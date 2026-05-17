"""CrossUserAuditService — log + review machinery for cross-user file edits.

Phase 1 problem statement
=========================

The Local Files homepage card surfaces canvas files belonging to *every*
local user workspace, not just the active one. The previous UI disabled
Load and Delete on rows owned by another user, but kept Reveal enabled —
so user B could always edit / delete A's canvas via the OS file explorer
anyway, making the in-UI gate pure theatre.

This module replaces that gate with an **accountability** model:

1. When B overwrites or deletes a file living under A's workspace dir
   (``<workspace_root>/<A_uid>/...``), an audit entry is appended to
   ``<workspace_root>/<A_uid>/.audit/pending.jsonl``.
2. When A next signs in on this machine, ``MainWindow`` reads the queue
   and shows a review dialog. A can ``accept()`` (drop the entry) or
   ``reject()`` (restore from A's cloud namespace; entry then drops).
3. The actual security perimeter is the Supabase RLS predicate
   ``(storage.foldername(name))[1] = auth.uid()::text`` — B's JWT can
   never write under A's ``{A_uid}/...`` cloud prefix, so A's cloud is
   authoritative and is the sole source for Reject.

This service is pure I/O (no QObject, no signals). All disk reads / writes
go through ``unitport_sdk.DataManager`` and ``unitport_sdk.Storage`` per
the SDK rule against raw ``open()``. JSONL is read/written with
``format='txt'`` so we ride the existing TextHandler (one atomic
temp-then-replace per write).

What is NOT here
================

- **Local pre-snapshots** — the user explicitly chose "cloud restore only"
  for Phase 1. ``capture_pre_state`` only computes sha256 + size; no
  ``.audit/snapshots/<sha>.bin`` files are written.
- **Cross-machine aggregation** — each machine's ``pending.jsonl`` is
  independent. Audit entries are never uploaded to the cloud.
- **History viewer UI** — ``history.jsonl`` is appended on accept/reject
  for future use but is never read back in Phase 1.
- **Audit for non-canvas files** — Phase 1 scopes itself to the two
  destructive entry points the homepage exposes: canvas save (overwrite)
  and the Local Files row delete button.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from unitport_sdk import (
    DataManager,
    Storage,
    log_debug,
    log_info,
    log_warning,
)

from application.service.user_workspace import (
    read_active_user,
    read_workspace_root,
)


# JSONL is read/written through the TextHandler (no native handler ships
# for ``.jsonl``). Passing ``format="txt"`` to DataManager forces it.
_JSONL_FORMAT = "txt"

# Special directory name that means "the local guest workspace". Guest
# never has a UUID, so cross-user audit excludes both sides whenever guest
# is involved (any guest↔X op is treated as same-user / unauditable).
_GUEST_DIR = "_guest"

# Layout under the target user's workspace dir.
_AUDIT_DIRNAME = ".audit"
_PENDING_FILENAME = "pending.jsonl"
_HISTORY_FILENAME = "history.jsonl"

# Bump when the JSON schema changes. ``list_pending_for`` ignores entries
# whose schema is newer than this constant (so an older binary doesn't
# choke on a future field) and logs a warning.
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """One row in ``<target_uid>/.audit/pending.jsonl``."""

    id: str
    schema: int
    ts: str                  # ISO-8601 UTC, e.g. ``2026-05-17T11:23:09Z``
    actor_uid: str
    actor_label: str         # display name / email at action time, frozen
    target_uid: str
    op: str                  # ``"overwrite"`` | ``"delete"``
    rel_path: str            # POSIX, relative to ``<target_uid>/``
    pre_sha256: str          # diagnostic only — not used to verify restore
    pre_size: int
    post_sha256: Optional[str]
    note: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "schema": self.schema,
            "ts": self.ts,
            "actor_uid": self.actor_uid,
            "actor_label": self.actor_label,
            "target_uid": self.target_uid,
            "op": self.op,
            "rel_path": self.rel_path,
            "pre_sha256": self.pre_sha256,
            "pre_size": self.pre_size,
            "post_sha256": self.post_sha256,
            "note": self.note,
        }


@dataclass(frozen=True)
class RejectResult:
    """Outcome of ``reject()`` — surfaced verbatim to the review dialog."""

    success: bool
    source: str              # ``"cloud"`` on success, ``"none"`` on failure
    error: str = ""          # human-readable; empty on success


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _audit_dir_for(target_uid: str) -> Optional[Path]:
    """Resolve ``<workspace_root>/<target_uid>/.audit/``; ``None`` for guest."""
    if not target_uid or target_uid == _GUEST_DIR:
        return None
    try:
        root = read_workspace_root()
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[audit] read_workspace_root failed: {exc!r}")
        return None
    return root / target_uid / _AUDIT_DIRNAME


def _pending_path(target_uid: str) -> Optional[Path]:
    d = _audit_dir_for(target_uid)
    return None if d is None else d / _PENDING_FILENAME


def _history_path(target_uid: str) -> Optional[Path]:
    d = _audit_dir_for(target_uid)
    return None if d is None else d / _HISTORY_FILENAME


def _read_lines(path: Path) -> List[str]:
    """Return non-empty lines from ``path``; empty list if file missing."""
    if not path.exists():
        return []
    try:
        # force_reload because audit files mutate under our feet between
        # calls (caller may have just rewritten via _write_lines) and the
        # DataManager cache key is shared.
        raw = DataManager.load(path, force_reload=True, format=_JSONL_FORMAT)
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[audit] read {path} failed: {exc!r}")
        return []
    if not isinstance(raw, str):
        return []
    return [ln for ln in raw.splitlines() if ln.strip()]


def _write_lines(path: Path, lines: List[str]) -> bool:
    """Atomic rewrite of ``path`` with ``lines`` joined by ``\\n``.

    Empty input produces an empty file (preserved so callers can detect
    "queue is empty" vs "queue file absent"). DataManager.write writes
    via temp-then-os.replace on the same volume (see SDK _atomic_write).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log_warning(f"[audit] mkdir {path.parent} failed: {exc!r}")
        return False
    body = ("\n".join(lines) + "\n") if lines else ""
    ok = DataManager.write(path, body, format=_JSONL_FORMAT)
    if not ok:
        log_warning(f"[audit] write {path} failed (DataManager returned False)")
    return ok


def _append_line(path: Path, line: str) -> bool:
    """Append one JSONL line — read-modify-write under the file lock.

    DataManager.write takes the per-path lock internally; combined with
    the prior _read_lines call we form a non-atomic compound, but writes
    from a single process are serialised by the GIL plus the per-path
    lock acquired inside write. Concurrent writers across processes are
    not a concern in this app (single Studio instance per machine).
    """
    lines = _read_lines(path)
    lines.append(line)
    return _write_lines(path, lines)


def _entry_from_obj(obj: dict) -> Optional[AuditEntry]:
    try:
        schema = int(obj.get("schema", 0))
        if schema > _SCHEMA_VERSION:
            log_warning(
                f"[audit] entry schema={schema} > {_SCHEMA_VERSION}; "
                "skipping (binary too old)"
            )
            return None
        return AuditEntry(
            id=str(obj["id"]),
            schema=schema,
            ts=str(obj.get("ts", "")),
            actor_uid=str(obj.get("actor_uid", "")),
            actor_label=str(obj.get("actor_label", "")),
            target_uid=str(obj.get("target_uid", "")),
            op=str(obj.get("op", "")),
            rel_path=str(obj.get("rel_path", "")),
            pre_sha256=str(obj.get("pre_sha256", "")),
            pre_size=int(obj.get("pre_size", 0) or 0),
            post_sha256=(
                str(obj["post_sha256"])
                if obj.get("post_sha256") is not None else None
            ),
            note=str(obj.get("note", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log_warning(f"[audit] malformed entry skipped: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_target(abs_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(target_uid, rel_path)`` if ``abs_path`` is a cross-user write.

    A path is "cross-user" when ALL hold:

    * the path lies under ``<workspace_root>/<some_uid>/...``,
    * ``some_uid`` ≠ the active user (the one logged in *now*; guest
      is treated as a distinct first-class actor with the sentinel
      ``_guest``),
    * ``some_uid`` is not ``_guest`` itself — the guest workspace has
      no UUID, no sign-in flow, no review surface, no cloud namespace,
      so nothing acts on an audit entry against it. Edits *to* guest
      files are silently allowed (and not recorded).

    Otherwise returns ``(None, None)``. This is the single source of
    truth for "should we record an audit entry" across the codebase —
    every UI hook (save / delete) consults it.
    """
    try:
        root = read_workspace_root()
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[audit] classify_target: read_workspace_root failed: {exc!r}")
        return (None, None)

    try:
        target_abs = Path(abs_path).resolve()
        root_abs = Path(root).resolve()
    except OSError as exc:
        log_warning(f"[audit] classify_target: resolve failed: {exc!r}")
        return (None, None)

    try:
        rel_to_root = target_abs.relative_to(root_abs)
    except ValueError:
        # Path is outside the workspace tree — not our concern.
        return (None, None)

    parts = rel_to_root.parts
    if len(parts) < 2:
        # File sits directly at the workspace root, not under any user.
        return (None, None)

    owner_dir = parts[0]
    if owner_dir == _GUEST_DIR:
        # Target is guest — can't review, can't restore from cloud.
        return (None, None)

    # Normalise empty active uid (signed-out / guest) to the guest
    # sentinel so the equality check below correctly identifies
    # "guest editing a signed-in user's file" as cross-user.
    active_raw = (read_active_user() or "").strip()
    active = active_raw if active_raw else _GUEST_DIR
    if owner_dir == active:
        # Same-user write — not cross-user.
        return (None, None)

    rel_posix = "/".join(parts[1:])
    return (owner_dir, rel_posix)


def capture_pre_state(abs_path: Path) -> Optional[dict]:
    """Read current bytes of ``abs_path`` and compute sha256 + size.

    Returns ``{"sha": hex, "size": int, "bytes_ok": bool}`` or ``None``
    when the file does not exist (deleting a missing file is a no-op
    and produces no audit entry). The bytes themselves are NOT written
    to disk — Phase 1 relies on cloud restore, not local snapshots.
    """
    try:
        if not abs_path.exists():
            return None
        data = DataManager.load(abs_path, force_reload=True)
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[audit] capture_pre_state {abs_path}: {exc!r}")
        return {"sha": "", "size": 0, "bytes_ok": False}

    # DataManager returns format-specific objects (dict for json/yaml,
    # str for txt, bytes for unknown). For sha purposes we want the
    # raw on-disk bytes — read once through the BytesHandler fallback.
    try:
        raw = DataManager.load(
            abs_path, force_reload=True, format="bytes",
        )
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[audit] re-read bytes {abs_path}: {exc!r}")
        # Fall back to whatever the typed read returned, best-effort.
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            return {"sha": "", "size": 0, "bytes_ok": False}

    if not isinstance(raw, (bytes, bytearray)):
        return {"sha": "", "size": 0, "bytes_ok": False}

    return {
        "sha": _sha256_hex(bytes(raw)),
        "size": len(raw),
        "bytes_ok": True,
    }


def record_overwrite(
    abs_path: Path,
    pre: dict,
    post_bytes: Optional[bytes],
    *,
    note: str,
) -> Optional[str]:
    """Append an ``op="overwrite"`` entry; returns the audit id or None.

    Call AFTER the write completes. ``pre`` is the dict returned by
    ``capture_pre_state`` BEFORE the write. ``post_bytes`` is the new
    on-disk content (used to compute ``post_sha256``); pass ``None`` if
    you couldn't capture it cheaply — sha will be empty.
    """
    target_uid, rel_path = classify_target(abs_path)
    if not target_uid or not rel_path:
        return None
    return _record(
        target_uid=target_uid,
        rel_path=rel_path,
        op="overwrite",
        pre=pre,
        post_sha=(_sha256_hex(post_bytes) if post_bytes is not None else None),
        note=note,
    )


def record_delete(
    abs_path: Path,
    pre: dict,
    *,
    note: str,
) -> Optional[str]:
    """Append an ``op="delete"`` entry; returns the audit id or None.

    Call AFTER the unlink completes. ``pre`` must come from a
    ``capture_pre_state`` call BEFORE the delete.
    """
    target_uid, rel_path = classify_target(abs_path)
    if not target_uid or not rel_path:
        return None
    return _record(
        target_uid=target_uid,
        rel_path=rel_path,
        op="delete",
        pre=pre,
        post_sha=None,
        note=note,
    )


def list_pending_for(target_uid: str) -> List[AuditEntry]:
    """Read the target user's pending queue; tolerant of malformed lines."""
    path = _pending_path(target_uid)
    if path is None:
        return []
    entries: List[AuditEntry] = []
    for line in _read_lines(path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            log_warning(f"[audit] malformed JSONL line skipped: {exc}")
            continue
        if not isinstance(obj, dict):
            continue
        entry = _entry_from_obj(obj)
        if entry is not None:
            entries.append(entry)
    return entries


def accept(target_uid: str, entry_id: str) -> bool:
    """Drop an entry from the pending queue; mirror it to history.jsonl."""
    return _decide(target_uid, entry_id, decision="accept")


def reject(target_uid: str, entry_id: str) -> RejectResult:
    """Restore the file from the target user's cloud namespace.

    Lifecycle assumption: this is called from the review dialog while
    the *target* user is the active user — meaning ``AuthManager`` has
    target's JWT in memory. We read that token live; never persist it.
    """
    # 1. Locate the entry without removing it yet — we only drop it if
    # the restore succeeded.
    pending_path = _pending_path(target_uid)
    if pending_path is None:
        return RejectResult(False, "none", "guest workspace has no audit")
    entry = _find_entry(target_uid, entry_id)
    if entry is None:
        return RejectResult(False, "none", "entry not found")

    # 2. Pull from cloud using the active user's access token.
    try:
        from application.service.auth import get_auth_manager
        from application.service.cloud_sync import get_cloud_sync_service
        token = get_auth_manager().access_token() or ""
    except Exception as exc:                                  # noqa: BLE001
        return RejectResult(False, "none", f"auth init failed: {exc!r}")
    if not token:
        return RejectResult(False, "none", "not signed in (no access token)")

    try:
        cloud = get_cloud_sync_service()
        body = cloud.pull_single(target_uid, entry.rel_path, token)
    except Exception as exc:                                  # noqa: BLE001
        return RejectResult(False, "none", f"cloud fetch error: {exc!r}")
    if body is None:
        return RejectResult(
            False, "none",
            "no cloud copy available (file was never synced or RLS denied)",
        )

    # 3. Write the cloud bytes back into the active user's local. We are
    # the target user when reviewing, so Storage.push (which writes under
    # Paths.USER_CONFIG_DIR) lands the bytes at the right address. Force
    # the bytes handler so the file extension (.canvas.json → JsonHandler
    # which expects a dict) doesn't reject the raw cloud payload.
    ok = Storage.push(entry.rel_path, body, channel="local", format="bytes")
    if not ok:
        return RejectResult(False, "none", "local write failed (see log)")

    # 4. Drop the entry from pending + mirror to history.
    _decide(target_uid, entry_id, decision="reject")
    return RejectResult(True, "cloud")


# ---------------------------------------------------------------------------
# Internal: record + decide
# ---------------------------------------------------------------------------


def _record(
    *,
    target_uid: str,
    rel_path: str,
    op: str,
    pre: Optional[dict],
    post_sha: Optional[str],
    note: str,
) -> Optional[str]:
    """Common path for record_overwrite + record_delete."""
    # Normalise empty / signed-out uid to the guest sentinel so guest
    # edits to signed-in users' files still produce a reviewable entry
    # (label surfaces as "Guest" on the target's review dialog).
    actor_raw = (read_active_user() or "").strip()
    actor_uid = actor_raw if actor_raw else _GUEST_DIR
    if actor_uid == _GUEST_DIR:
        actor_label = "Guest"
    else:
        try:
            from application.service.auth import get_auth_manager
            prof = get_auth_manager().current_user()
            actor_label = (
                getattr(prof, "display_name", "")
                or getattr(prof, "email", "")
                or actor_uid
            )
        except Exception:                                     # noqa: BLE001
            actor_label = actor_uid

    pre = pre or {"sha": "", "size": 0, "bytes_ok": False}
    entry = AuditEntry(
        id=str(uuid.uuid4()),
        schema=_SCHEMA_VERSION,
        ts=_now_iso(),
        actor_uid=actor_uid,
        actor_label=actor_label,
        target_uid=target_uid,
        op=op,
        rel_path=rel_path,
        pre_sha256=str(pre.get("sha", "")),
        pre_size=int(pre.get("size", 0) or 0),
        post_sha256=post_sha,
        note=note,
    )

    path = _pending_path(target_uid)
    if path is None:
        return None
    line = json.dumps(entry.to_dict(), ensure_ascii=False)
    if not _append_line(path, line):
        return None
    log_info(
        f"[audit] recorded {op} by {actor_label!r} on "
        f"{target_uid}/{rel_path} (id={entry.id})"
    )
    return entry.id


def _find_entry(target_uid: str, entry_id: str) -> Optional[AuditEntry]:
    for e in list_pending_for(target_uid):
        if e.id == entry_id:
            return e
    return None


def _decide(target_uid: str, entry_id: str, *, decision: str) -> bool:
    """Remove ``entry_id`` from pending.jsonl; append decision to history."""
    pending = _pending_path(target_uid)
    history = _history_path(target_uid)
    if pending is None or history is None:
        return False

    kept_lines: List[str] = []
    decided_obj: Optional[dict] = None
    for line in _read_lines(pending):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Preserve malformed lines so we don't silently lose data.
            kept_lines.append(line)
            continue
        if isinstance(obj, dict) and str(obj.get("id", "")) == entry_id:
            decided_obj = obj
            continue
        kept_lines.append(line)

    if decided_obj is None:
        log_debug(f"[audit] decide({decision}, {entry_id}): not found")
        return False

    if not _write_lines(pending, kept_lines):
        return False

    decided_obj["decision"] = decision
    decided_obj["decided_ts"] = _now_iso()
    decided_obj["decided_by"] = (read_active_user() or "")
    _append_line(history, json.dumps(decided_obj, ensure_ascii=False))
    return True
