# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CloudFileOpTask — single-file cloud operation worker.

Distinct from :class:`CloudSyncTask` (which orchestrates batch push /
pull) so the Manage dialog can fire one-off ``download_one`` /
``upload_one`` / ``delete`` operations without contending with the auto-
push pipeline or inheriting its progress / status broadcasting.

Each task instance runs exactly one operation:

* ``download_one`` — fetch ``{user_id}/{rel_path}`` and atomically write
  it to ``local_dst`` (caller resolves the absolute target path,
  typically under ``Paths.USER_CONFIG_DIR``).
* ``upload_one``   — PUT ``upload_bytes`` to ``{user_id}/{rel_path}``
  with ``upsert=True``. Caller is responsible for the 50 MB size guard.
* ``delete``       — DELETE ``{user_id}/{rel_path}``.

The task never raises out of ``run()``: per-op failures (Storage error,
network error, missing config, signed-out, etc.) are returned as
``{"ok": False, "error": "...", "phase": ..., "rel_path": ...}``. The
caller (the Manage dialog) is expected to surface the failure via a
QMessageBox rather than reattempt automatically.

I/O happens inside the TasksManager worker thread, so the main UI stays
responsive. No ``progress_line`` is emitted — single-file ops complete
in seconds and the dialog instead disables the affected row until the
``task_finished`` signal lands.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from unitport_sdk import Config, Task, log_warning

from application.service.auth import get_auth_manager


class CloudListTask(Task):
    """Fetch the user's full cloud prefix listing on a worker thread.

    Wraps :meth:`CloudSyncService.list_remote` so the Manage dialog can
    open without blocking the main thread on the HTTP round-trip
    (list + recursive folder walks can take seconds). The service emits
    ``usage_changed`` internally on success, so the dialog's quota row
    refreshes for free.

    Result shape: ``{"ok": bool, "rows": list[dict], "error": str|None}``.
    """

    def __init__(self) -> None:
        super().__init__(name="cloud-list")

    def run(self) -> Any:  # type: ignore[override]
        from application.service.cloud_sync import get_cloud_sync_service
        try:
            rows = get_cloud_sync_service().list_remote()
        except Exception as exc:                                  # noqa: BLE001
            self.log_debug(f"list_remote failed: {exc!r}")
            return {"ok": False, "rows": [], "error": str(exc)}
        self.log_debug(f"list_remote: {len(rows)} object(s)")
        return {"ok": True, "rows": rows, "error": None}


class CloudFileOpTask(Task):
    """One-shot cloud-file worker for the TasksManager queue."""

    PHASES = frozenset({"download_one", "upload_one", "delete", "delete_many"})

    def __init__(
        self,
        phase: str,
        rel_path: str = "",
        *,
        upload_bytes: Optional[bytes] = None,
        upload_ct: str = "application/octet-stream",
        local_dst: Optional[Path] = None,
        rel_paths: Optional[List[str]] = None,
    ) -> None:
        if phase not in self.PHASES:
            raise ValueError(
                f"CloudFileOpTask: unknown phase {phase!r}; "
                f"expected one of {sorted(self.PHASES)}"
            )
        if phase == "delete_many":
            cleaned = [p.lstrip("/") for p in (rel_paths or []) if p]
            if not cleaned:
                raise ValueError(
                    "CloudFileOpTask: delete_many requires non-empty "
                    "rel_paths list"
                )
            super().__init__(name=f"cloud-delete-many:{len(cleaned)}")
            self._rel_paths = cleaned
            self._rel_path = ""
        else:
            if not rel_path or rel_path.startswith("/"):
                raise ValueError(
                    "CloudFileOpTask: rel_path must be a non-empty relative "
                    f"path (got {rel_path!r})"
                )
            super().__init__(name=f"cloud-{phase}:{rel_path}")
            self._rel_path = rel_path.lstrip("/")
            self._rel_paths = []
        self._phase = phase
        self._upload_bytes = upload_bytes
        self._upload_ct = upload_ct or "application/octet-stream"
        self._local_dst = local_dst

    # ------------------------------------------------------------------
    # Run dispatch
    # ------------------------------------------------------------------

    def run(self) -> Any:  # type: ignore[override]
        ctx = self._resolve_context()
        if "error" in ctx:
            return self._fail(ctx["error"])

        client = ctx["client"]
        uid = ctx["user_id"]
        access = ctx["access_token"]

        try:
            if self._phase == "delete_many":
                return self._do_delete_many(client, access, uid)
            key = f"{uid}/{self._rel_path}"
            if self._phase == "download_one":
                return self._do_download(client, access, key)
            if self._phase == "upload_one":
                return self._do_upload(client, access, key)
            # delete
            return self._do_delete(client, access, key)
        except Exception as exc:                                  # noqa: BLE001
            # StorageError / StorageNetworkError / OSError all funnel
            # here; the dialog surfaces ``error`` via a QMessageBox so we
            # don't double-log at warning level inside the worker.
            self.log_debug(f"{self._phase} {self._rel_path}: {exc!r}")
            return self._fail(str(exc) or exc.__class__.__name__)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _do_download(self, client, access: str, key: str) -> dict:
        if self._local_dst is None:
            return self._fail("download_one requires local_dst")
        data = client.download(access, key)
        if data is None:
            return self._fail(f"cloud object not found: {key}")
        dst = Path(self._local_dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a sibling temp, then os.replace into
        # place. Same-volume guarantee holds because the temp file lives
        # in the same parent dir as the destination.
        tmp = dst.with_name(f".{dst.name}.cloud-partial.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
        except OSError as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return self._fail(f"local write failed: {exc}")
        return {
            "ok": True,
            "phase": "download_one",
            "rel_path": self._rel_path,
            "local_path": str(dst),
            "size": len(data),
        }

    def _do_upload(self, client, access: str, key: str) -> dict:
        payload = self._upload_bytes
        if payload is None:
            return self._fail("upload_one requires upload_bytes")
        client.upload(
            access, key, payload,
            content_type=self._upload_ct, upsert=True,
        )
        return {
            "ok": True,
            "phase": "upload_one",
            "rel_path": self._rel_path,
            "size": len(payload),
        }

    def _do_delete(self, client, access: str, key: str) -> dict:
        client.remove(access, [key])
        return {
            "ok": True,
            "phase": "delete",
            "rel_path": self._rel_path,
        }

    def _do_delete_many(self, client, access: str, uid: str) -> dict:
        keys = [f"{uid}/{rp}" for rp in self._rel_paths]
        client.remove(access, keys)
        return {
            "ok": True,
            "phase": "delete_many",
            "rel_paths": list(self._rel_paths),
            "count": len(self._rel_paths),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_context(self) -> Dict[str, Any]:
        """Resolve auth + storage client lazily on the worker thread.

        Returns a dict with ``client``, ``user_id``, ``access_token`` on
        success, or a dict carrying just ``error`` describing what was
        missing. Done here (not in ``__init__``) so a stale dialog
        reference still surfaces a clean error message when the user
        signed out between click and dispatch.
        """
        mgr = get_auth_manager()
        if not mgr.is_signed_in():
            return {"error": "not signed in"}
        user = mgr.current_user()
        uid = user.user_id if user is not None else ""
        access = mgr.access_token()
        if not uid or not access:
            return {"error": "auth context missing user_id or access token"}
        url = Config.get_value("auth", "supabase_url", fallback="") or ""
        anon = Config.get_value("auth", "supabase_anon_key", fallback="") or ""
        if not url or not anon:
            return {"error": "supabase url/anon key not configured"}
        try:
            from application.service.auth.supabase_storage import (
                SupabaseStorageClient,
            )
            client = SupabaseStorageClient(url=url, anon_key=anon)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud-manage-task] client init failed: {exc!r}")
            return {"error": f"client init failed: {exc}"}
        return {"client": client, "user_id": uid, "access_token": access}

    def _fail(self, message: str) -> dict:
        out: Dict[str, Any] = {
            "ok": False,
            "phase": self._phase,
            "error": message,
        }
        if self._phase == "delete_many":
            out["rel_paths"] = list(self._rel_paths)
        else:
            out["rel_path"] = self._rel_path
        return out


__all__ = ["CloudFileOpTask", "CloudListTask"]
