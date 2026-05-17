"""CloudSyncService — local USER_CONFIG_DIR ↔ Supabase Storage orchestrator.

Phase 1 scope (matches storage-sync.spec.yaml):
    - Manual Push: walk local, apply include/exclude + transforms, upload via
      SupabaseStorageClient; record etags in a sync-state file.
    - Manual Pull: list remote tree, download every key into local paths;
      Phase 1 does not delete local files that no longer exist remotely
      (soft strategy deferred — see spec.runtime.on_delete).
    - Self-check: list-only, used by lifecycle hooks to surface a status
      number without transferring bytes.

Out of scope (deferred to next plan):
    - Auto-push file watcher + debounce
    - Conflict detection with `.conflict-<ts>` rename
    - Cloud-side deletion of objects whose local source disappeared
    - Background timer / periodic re-sync

The spec lives at PROJECT_ROOT/storage-sync.spec.yaml but is NOT read at
runtime. Its glob lists and transform rules are mirrored as module-level
constants below; keep them in lock-step when editing the spec.

This service is a QObject so we can emit Qt signals (progress / finished
/ status_changed) for the UserPanel to listen on. It does NOT spawn any
threads; the orchestration is driven by CloudSyncTask running on a
TasksManager worker, which calls into this service synchronously.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import io
import json
import re
from configparser import RawConfigParser
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from unitport_sdk import (
    Config,
    DataManager,
    Paths,
    log_error,
    log_info,
    log_warning,
)


# ---------------------------------------------------------------------------
# Spec constants — mirror of storage-sync.spec.yaml § 3-4. Edit BOTH files
# in sync. The spec is the human-readable source of truth; these constants
# are what runs.
# ---------------------------------------------------------------------------


_INCLUDE_GLOBS: Tuple[str, ...] = (
    "user.ini",
    "engines/*.json",
    "registers/*.json",
    "robot_presets/*.json",
    "projects/*/project.yaml",
    "projects/*/README.md",
    "projects/*/canvas/**/*.canvas.json",
    "projects/*/behaviors/**",
    "projects/*/datasets/**",
    "projects/*/resources/**",
    "projects/*/robots/**/*.yaml",
    "projects/*/scenarios/**",
    "projects/*/scripts/**",
    "projects/*/training/configs/**",
    "projects/*/training/exported/**",
    "projects/*/training/runs/**",
)

_EXCLUDE_GLOBS: Tuple[str, ...] = (
    "avatars/**",
    "session.json",
    ".cloud_sync_state.json",
    "**/.unitport/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "projects/*/training/runs/*/*/git/**",
    "projects/*/training/runs/*/*/events.out.tfevents.*",
)

_USER_INI_UPLOAD_SECTIONS: Tuple[str, ...] = (
    "Window",
    "Project",
    "Localisation",
    "System",
    "SimConfig",
)

_RUN_PER_KEEP_GLOBS: Tuple[str, ...] = (
    "deploy_meta.json",
    "unitport_run_meta.yaml",
    "amp_alignment.json",
    "unitport_env_cfg.py",
    "params/**",
)
_RUN_PER_DROP_GLOBS: Tuple[str, ...] = (
    "checkpoints/**",
    "git/**",
    "__pycache__/**",
    "events.out.tfevents.*",
)
_RUN_MODEL_PT_PATTERN = re.compile(r"^model_(\d+)\.pt$")

_CONTENT_TYPE_BY_EXT: Dict[str, str] = {
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".json": "application/json",
    ".ini": "text/plain",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".onnx": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".img": "application/octet-stream",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

_STATE_FILENAME = ".cloud_sync_state.json"
_DEFAULT_BUCKET = "user-data"


# ---------------------------------------------------------------------------
# Plan / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlanEntry:
    """One file scheduled for push or pull."""

    local_path: Path                # absolute local path
    rel_path: str                   # local-relative posix path
    cloud_key: str                  # cloud key WITHOUT the leading user-id prefix
    size: int = 0
    content_type: str = "application/octet-stream"
    # If set, push uses this exact body instead of reading local_path
    # (used by transforms like user_ini_split / engines_strip_local).
    body_override: Optional[bytes] = None


@dataclass
class SyncPlan:
    phase: str                                  # "push" | "pull"
    entries: List[PlanEntry] = field(default_factory=list)
    skipped_oversize: List[str] = field(default_factory=list)
    skipped_excluded: int = 0
    skipped_runs_topn: int = 0


@dataclass
class SyncResult:
    phase: str
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CloudSyncService(QObject):
    """Singleton. Stateless w.r.t. credentials — pulls from AuthManager per call."""

    # progress(done, total, current_key) — emitted from the worker thread.
    progress = pyqtSignal(int, int, str)
    # status_changed(payload) — UI listens for "summary changed" pings.
    status_changed = pyqtSignal(dict)
    # finished(phase, success, summary) — phase ∈ {"push","pull","self_check"}
    finished = pyqtSignal(str, bool, dict)
    # usage_changed(used_bytes, max_bytes) — emitted after every cloud op
    # that refreshes the server-side quota (push / pull / self_check).
    usage_changed = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._bucket = (
            Config.get_value("CloudSync", "bucket", fallback=_DEFAULT_BUCKET)
            or _DEFAULT_BUCKET
        )
        size_raw = Config.get_value(
            "CloudSync", "single_file_size_limit_mb", fallback=50,
        )
        try:
            size_mb = float(size_raw)
        except (TypeError, ValueError):
            size_mb = 50.0
        self._size_limit_bytes = int(size_mb * 1024 * 1024)
        topn_raw = Config.get_value(
            "CloudSync", "runs_keep_top_n", fallback=3,
        )
        try:
            self._runs_keep_top_n = int(topn_raw)
        except (TypeError, ValueError):
            self._runs_keep_top_n = 3
        # Last-known server-side quota result. None means "never fetched";
        # UI shows a placeholder until the first refresh completes.
        self._last_usage: Optional[Tuple[int, int]] = None

    # ----- public API ------------------------------------------------------

    @property
    def bucket(self) -> str:
        return self._bucket

    def get_status(self) -> dict:
        """Read the persisted state file; returns an empty dict if absent."""
        state = self._load_state()
        return {
            "last_push_ts": state.get("last_push_ts", ""),
            "last_pull_ts": state.get("last_pull_ts", ""),
            "last_self_check_ts": state.get("last_self_check_ts", ""),
            "remote_count": int(state.get("remote_count", 0) or 0),
            "synced_files": len(state.get("files", {}) or {}),
        }

    def cached_usage(self) -> Optional[Tuple[int, int]]:
        """Last fetched ``(used_bytes, max_bytes)`` or ``None`` if unknown."""
        return self._last_usage

    def fetch_storage_usage(self) -> Optional[Tuple[int, int]]:
        """Hit the ``get_storage_usage`` RPC, cache, emit ``usage_changed``.

        Synchronous — call from a worker thread. Returns the
        ``(used_bytes, max_bytes)`` tuple, or ``None`` on any failure
        (signed-out / no network / RPC missing). On success it both
        caches the value and emits ``usage_changed`` for the UI.
        """
        client, user_id, access = self._client_or_none()
        if client is None or not user_id or not access:
            return None
        try:
            payload = client.storage_usage_rpc(access, user_id)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] storage_usage_rpc failed: {exc}")
            return None
        if not isinstance(payload, dict):
            return None
        try:
            used = int(payload.get("used_bytes") or 0)
            total = int(payload.get("max_bytes") or 0)
        except (TypeError, ValueError):
            return None
        self._last_usage = (used, total)
        self.usage_changed.emit(used, total)
        return self._last_usage

    def list_remote(self) -> List[dict]:
        """List the signed-in user's full cloud prefix. Used by self-check.

        Returns a list of ``{"name","size","etag","content_type"}`` dicts
        suitable for logging / status display. Never raises — connection
        / auth errors are logged and the result is an empty list.
        """
        client, user_id, access = self._client_or_none()
        if client is None or not user_id or not access:
            return []
        try:
            objs = client.list(access, prefix=user_id)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] list_remote failed: {exc}")
            return []

        rows = [
            {
                "name": o.name,
                "size": o.size,
                "etag": o.etag,
                "content_type": o.content_type,
            }
            for o in objs
        ]
        # Persist the count so the status row can render after self-check.
        state = self._load_state()
        state["remote_count"] = len(rows)
        state["last_self_check_ts"] = _now_iso()
        self._save_state(state)
        self.status_changed.emit(self.get_status())
        self.fetch_storage_usage()
        return rows

    def plan_push(self) -> SyncPlan:
        """Walk USER_CONFIG_DIR, apply include/exclude/transforms.

        Does NOT consult cloud state — Phase 1 push uploads everything
        the plan covers with ``upsert=true``. (Diff-based push is a Phase
        2 optimisation; for first cut we trade bandwidth for simplicity.)
        """
        plan = SyncPlan(phase="push")
        base = self._base_dir()
        if base is None:
            return plan

        # 1. Walk all candidate files relative to base, applying include.
        included: List[str] = []
        for absolute in _iter_files(base):
            rel_posix = _to_posix_rel(absolute, base)
            if _matches_any(rel_posix, _EXCLUDE_GLOBS):
                plan.skipped_excluded += 1
                continue
            if not _matches_any(rel_posix, _INCLUDE_GLOBS):
                continue
            included.append(rel_posix)

        # 2. Apply runs_topn pruning across the runs subtree.
        kept, dropped = self._prune_runs_topn(base, included)
        plan.skipped_runs_topn = dropped

        # 3. Build entries, applying per-file transforms / size limit.
        for rel_posix in kept:
            absolute = base / Path(*rel_posix.split("/"))
            try:
                st = absolute.stat()
            except OSError:
                continue
            size = int(st.st_size)
            ext = absolute.suffix.lower()

            # transforms — may rewrite the body before upload
            body_override: Optional[bytes] = None
            if rel_posix == "user.ini":
                body_override = self._transform_user_ini_split(absolute)
                if body_override is None:
                    # All sections were excluded; nothing to upload.
                    continue
                size = len(body_override)
            elif fnmatch.fnmatch(rel_posix, "engines/*.json"):
                body_override = self._transform_engines_strip_local(absolute)
                if body_override is None:
                    continue
                size = len(body_override)

            if size > self._size_limit_bytes:
                plan.skipped_oversize.append(rel_posix)
                continue

            plan.entries.append(PlanEntry(
                local_path=absolute,
                rel_path=rel_posix,
                cloud_key=rel_posix,
                size=size,
                content_type=_CONTENT_TYPE_BY_EXT.get(
                    ext, "application/octet-stream",
                ),
                body_override=body_override,
            ))
        return plan

    def plan_pull(self) -> SyncPlan:
        """List the remote prefix and turn each object into a Pull entry."""
        plan = SyncPlan(phase="pull")
        client, user_id, access = self._client_or_none()
        if client is None or not user_id or not access:
            return plan
        base = self._base_dir()
        if base is None:
            return plan

        try:
            objs = client.list(access, prefix=user_id)
        except Exception as exc:                                  # noqa: BLE001
            log_error(f"[cloud-sync] plan_pull list failed: {exc}")
            return plan

        prefix = f"{user_id}/"
        for o in objs:
            if not o.name.startswith(prefix):
                continue
            rel_posix = o.name[len(prefix):]
            if not rel_posix:
                continue
            plan.entries.append(PlanEntry(
                local_path=base / Path(*rel_posix.split("/")),
                rel_path=rel_posix,
                cloud_key=rel_posix,
                size=o.size,
                content_type=o.content_type or "application/octet-stream",
            ))
        return plan

    def execute(self, plan: SyncPlan, *, progress_cb=None) -> dict:
        """Drive the plan to completion; emits per-file progress.

        ``progress_cb(done, total, current_key)`` is invoked alongside
        ``self.progress`` so a Task can also feed its own task_progress
        signal (avoiding a cross-thread emit through this QObject when
        the Task is the more natural progress channel).
        """
        if plan.phase == "push":
            return self._execute_push(plan, progress_cb=progress_cb)
        if plan.phase == "pull":
            return self._execute_pull(plan, progress_cb=progress_cb)
        log_error(f"[cloud-sync] execute: unknown phase {plan.phase!r}")
        return {"ok": 0, "failed": 0, "skipped": 0, "errors": []}

    def pull_single(
        self,
        uid: str,
        rel_path: str,
        access_token: str,
    ) -> Optional[bytes]:
        """Fetch one object from ``{uid}/{rel_path}`` and return its bytes.

        Used by the cross-user audit Reject path to restore a target user's
        file from their cloud namespace. Returns ``None`` on 404, on RLS
        denial (403), on missing config, or on network error — caller is
        expected to surface a "no cloud copy available" message rather than
        retry. Does not write to disk and does not touch ``_state``.
        """
        if not uid or not rel_path or not access_token:
            return None
        try:
            from application.service.auth.supabase_storage import (
                SupabaseStorageClient,
            )
            url = Config.get_value("auth", "supabase_url", fallback="") or ""
            anon = Config.get_value("auth", "supabase_anon_key", fallback="") or ""
            if not url or not anon:
                log_warning("[cloud-sync] pull_single: supabase url/anon missing")
                return None
            client = SupabaseStorageClient(
                url=url, anon_key=anon, bucket=self._bucket,
            )
            key = f"{uid}/{rel_path.lstrip('/')}"
            return client.download(access_token, key)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] pull_single({rel_path!r}) failed: {exc}")
            return None

    # ----- push / pull execution ------------------------------------------

    def _execute_push(self, plan: SyncPlan, *, progress_cb) -> dict:
        client, user_id, access = self._client_or_none()
        if client is None or not user_id or not access:
            return {
                "ok": 0,
                "failed": 0,
                "skipped": len(plan.entries),
                "errors": ["not signed in"],
            }

        result = SyncResult(phase="push")
        result.skipped += len(plan.skipped_oversize)
        state = self._load_state()
        files_state: dict = dict(state.get("files", {}) or {})

        total = len(plan.entries)
        for i, entry in enumerate(plan.entries, start=1):
            try:
                payload = entry.body_override
                if payload is None:
                    payload = entry.local_path.read_bytes()
                client.upload(
                    access,
                    f"{user_id}/{entry.cloud_key}",
                    payload,
                    content_type=entry.content_type,
                    upsert=True,
                )
                etag = hashlib.md5(payload, usedforsecurity=False).hexdigest()
                files_state[entry.rel_path] = {
                    "etag": etag,
                    "size": entry.size,
                    "local_mtime": _safe_mtime(entry.local_path),
                    "synced_ts": _now_iso(),
                }
                result.ok += 1
            except Exception as exc:                              # noqa: BLE001
                # Per-file failures are collected, not logged here:
                # logging inside the progress_cb loop would break the
                # SDK's overwrite-style progress_line. The caller
                # (CloudSyncTask) drains result.errors after the final
                # progress tick lands.
                result.failed += 1
                result.errors.append(f"{entry.rel_path}: {exc}")

            self.progress.emit(i, total, entry.cloud_key)
            if progress_cb is not None:
                try:
                    progress_cb(i, total, entry.cloud_key)
                except Exception:
                    pass

        state["files"] = files_state
        state["last_push_ts"] = _now_iso()
        self._save_state(state)
        self.status_changed.emit(self.get_status())
        self.fetch_storage_usage()
        return {
            "ok": result.ok,
            "failed": result.failed,
            "skipped": result.skipped,
            "errors": result.errors[:20],          # truncate for log
            "oversize": plan.skipped_oversize,
            "excluded": plan.skipped_excluded,
            "runs_pruned": plan.skipped_runs_topn,
            "total": total,
        }

    def _execute_pull(self, plan: SyncPlan, *, progress_cb) -> dict:
        client, user_id, access = self._client_or_none()
        if client is None or not user_id or not access:
            return {
                "ok": 0,
                "failed": 0,
                "skipped": len(plan.entries),
                "errors": ["not signed in"],
            }

        result = SyncResult(phase="pull")
        state = self._load_state()
        files_state: dict = dict(state.get("files", {}) or {})

        total = len(plan.entries)
        for i, entry in enumerate(plan.entries, start=1):
            try:
                data = client.download(access, f"{user_id}/{entry.cloud_key}")
                if data is None:
                    result.skipped += 1
                    continue
                target = entry.local_path
                target.parent.mkdir(parents=True, exist_ok=True)
                # Atomic-ish write: temp + rename. DataManager's write()
                # handles formats but here we have raw bytes from the
                # network — write directly. Tempfile lives next to the
                # target so os.replace stays on the same volume.
                tmp = target.with_suffix(target.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(target)
                etag = hashlib.md5(data, usedforsecurity=False).hexdigest()
                files_state[entry.rel_path] = {
                    "etag": etag,
                    "size": len(data),
                    "local_mtime": _safe_mtime(target),
                    "synced_ts": _now_iso(),
                }
                result.ok += 1
            except Exception as exc:                              # noqa: BLE001
                # Same rationale as _execute_push: collect, don't log
                # inside the progress_cb loop.
                result.failed += 1
                result.errors.append(f"{entry.rel_path}: {exc}")

            self.progress.emit(i, total, entry.cloud_key)
            if progress_cb is not None:
                try:
                    progress_cb(i, total, entry.cloud_key)
                except Exception:
                    pass

        state["files"] = files_state
        state["last_pull_ts"] = _now_iso()
        self._save_state(state)
        self.status_changed.emit(self.get_status())
        self.fetch_storage_usage()
        return {
            "ok": result.ok,
            "failed": result.failed,
            "skipped": result.skipped,
            "errors": result.errors[:20],
            "total": total,
        }

    # ----- transforms (spec §4) -------------------------------------------

    def _transform_user_ini_split(self, ini_path: Path) -> Optional[bytes]:
        """Return a filtered ``user.ini`` body containing only safe sections."""
        try:
            text = ini_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log_warning(f"[cloud-sync] user.ini read failed: {exc}")
            return None
        parser = RawConfigParser()
        parser.optionxform = str               # preserve key case (legacy ini)
        try:
            parser.read_string(text)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] user.ini parse failed: {exc}")
            return None

        keep = RawConfigParser()
        keep.optionxform = str
        kept_any = False
        for section in parser.sections():
            if section not in _USER_INI_UPLOAD_SECTIONS:
                continue
            keep.add_section(section)
            for k, v in parser.items(section):
                keep.set(section, k, v)
            kept_any = True
        if not kept_any:
            return None
        buf = io.StringIO()
        keep.write(buf)
        return buf.getvalue().encode("utf-8")

    def _transform_engines_strip_local(self, json_path: Path) -> Optional[bytes]:
        """Strip the per-machine ``$.local`` field from engines/*.json."""
        try:
            raw = json_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log_warning(f"[cloud-sync] engines read failed: {exc}")
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Malformed JSON — upload as-is so the user can fix and resync.
            return raw.encode("utf-8")
        if isinstance(data, dict) and "local" in data:
            data = {k: v for k, v in data.items() if k != "local"}
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    def _prune_runs_topn(
        self,
        base: Path,
        candidates: List[str],
    ) -> Tuple[List[str], int]:
        """Apply spec.transform.runs_topn — keep last N runs per engine.

        Group candidates under ``projects/*/training/runs/<engine>/<run>/...``
        by their ``<engine>/<run>`` directory, rank by run-dir mtime,
        keep ``self._runs_keep_top_n`` newest. Within each kept run,
        apply per_run_keep / per_run_drop / model_*.pt latest-step.
        """
        kept: List[str] = []
        run_buckets: Dict[Tuple[str, str], List[str]] = {}
        for rel in candidates:
            run_dir = _run_dir_for(rel)
            if run_dir is None:
                kept.append(rel)
                continue
            run_buckets.setdefault(run_dir, []).append(rel)

        dropped = 0
        # Group runs by their (project, engine) parent so we keep top-N
        # *per engine*, not globally.
        engine_groups: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = {}
        for (engine_dir, run_name), files in run_buckets.items():
            mtime = 0.0
            try:
                mtime = (base / engine_dir / run_name).stat().st_mtime
            except OSError:
                pass
            engine_groups.setdefault(engine_dir, []).append(
                (engine_dir, run_name, mtime)
            )

        kept_run_keys: set = set()
        for engine_dir, runs in engine_groups.items():
            runs.sort(key=lambda x: x[2], reverse=True)
            for engine_dir2, run_name, _mt in runs[: self._runs_keep_top_n]:
                kept_run_keys.add((engine_dir2, run_name))

        for (engine_dir, run_name), files in run_buckets.items():
            if (engine_dir, run_name) not in kept_run_keys:
                dropped += len(files)
                continue
            kept.extend(
                self._filter_inside_run(engine_dir, run_name, files)
            )
        return kept, dropped

    def _filter_inside_run(
        self,
        engine_dir: str,
        run_name: str,
        files: List[str],
    ) -> List[str]:
        prefix = f"{engine_dir}/{run_name}/"
        result: List[str] = []
        model_pts: List[Tuple[int, str]] = []
        for rel in files:
            inside = rel[len(prefix):] if rel.startswith(prefix) else rel
            if _matches_any(inside, _RUN_PER_DROP_GLOBS):
                continue
            # model_<N>.pt at the run root — keep only the latest step
            # (spec.transform.runs_topn.per_run_keep_latest_only).
            if "/" not in inside:
                m = _RUN_MODEL_PT_PATTERN.match(inside)
                if m is not None:
                    model_pts.append((int(m.group(1)), rel))
                    continue
            result.append(rel)
        if model_pts:
            model_pts.sort(key=lambda x: x[0], reverse=True)
            result.append(model_pts[0][1])
        return result

    # ----- helpers --------------------------------------------------------

    def _base_dir(self) -> Optional[Path]:
        try:
            base = Path(Paths.USER_CONFIG_DIR)
        except Exception:                                         # noqa: BLE001
            log_error("[cloud-sync] Paths.USER_CONFIG_DIR unavailable")
            return None
        if not base.exists():
            return None
        return base

    def _client_or_none(self):
        """Return ``(client, user_id, access_token)`` or ``(None, "", "")``.

        Refuses to operate when:
            - no AuthManager / not signed in
            - user_id is empty (guest workspace — by user requirement)
        """
        try:
            from application.service.auth import get_auth_manager
            from application.service.auth.supabase_storage import (
                SupabaseStorageClient,
            )
            mgr = get_auth_manager()
            if not mgr.is_signed_in():
                return None, "", ""
            user = mgr.current_user()
            if user is None or not user.user_id:
                return None, "", ""
            access = mgr.access_token()
            if not access:
                return None, "", ""
            url = Config.get_value("auth", "supabase_url", fallback="") or ""
            anon = Config.get_value("auth", "supabase_anon_key", fallback="") or ""
            client = SupabaseStorageClient(
                url=url, anon_key=anon, bucket=self._bucket,
            )
            return client, user.user_id, access
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] client_or_none failed: {exc}")
            return None, "", ""

    def _state_path(self) -> Path:
        return Path(Paths.USER_CONFIG_DIR) / _STATE_FILENAME

    def _load_state(self) -> dict:
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            data = DataManager.read(path)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] state read failed: {exc}")
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, state: dict) -> None:
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            DataManager.write(path, state)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-sync] state write failed: {exc}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_instance: Optional[CloudSyncService] = None


def get_cloud_sync_service() -> CloudSyncService:
    """Return the process-wide CloudSyncService, lazy-constructed."""
    global _instance
    if _instance is None:
        _instance = CloudSyncService()
    return _instance


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _iter_files(base: Path) -> Iterable[Path]:
    """Recursive file walk that skips the state file + .git automatically."""
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.name == _STATE_FILENAME:
            continue
        # Cheap top-level filter — `.git` and friends never carry user-
        # facing data and the exclude globs would catch them later anyway,
        # but we save a stat() per .git/objects entry by stopping here.
        rel = p.relative_to(base).parts
        if rel and rel[0] in {".git"}:
            continue
        yield p


def _to_posix_rel(absolute: Path, base: Path) -> str:
    return str(PurePosixPath(absolute.relative_to(base).as_posix()))


def _matches_any(rel_posix: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, p) for p in patterns)


def _run_dir_for(rel_posix: str) -> Optional[Tuple[str, str]]:
    """If ``rel_posix`` is under projects/*/training/runs/<engine>/<run>/...,
    return ``(<projects/*/training/runs/<engine>>, <run>)``; else None."""
    parts = rel_posix.split("/")
    # projects / <project> / training / runs / <engine> / <run> / ...
    if (
        len(parts) >= 7
        and parts[0] == "projects"
        and parts[2] == "training"
        and parts[3] == "runs"
    ):
        engine_dir = "/".join(parts[:5])      # projects/<P>/training/runs/<E>
        run_name = parts[5]
        return engine_dir, run_name
    return None


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


__all__ = [
    "CloudSyncService",
    "SyncPlan",
    "PlanEntry",
    "get_cloud_sync_service",
]
