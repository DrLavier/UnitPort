# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AutoSyncPushTask — worker-thread uploader for persist-time cloud auto-sync.

Distinct from :class:`application.tools.cloud_sync_task.CloudSyncTask`, which
plans a FULL push/pull/self-check from the UserPanel buttons. This task uploads
only an explicit set of just-persisted rel-paths, handed to it by
:class:`application.service.auto_sync.AutoSyncController` after it debounces a
burst of ``DataManager.write`` calls under ``USER_CONFIG_DIR``.

It reuses :meth:`CloudSyncService.plan_push_paths` (same include/exclude/marker
gate, same privacy transforms, same content-diff as Manual Push) and
:meth:`CloudSyncService.execute`, so an auto-synced file is byte-for-byte what a
manual Push would have uploaded. Deliberately quiet: it does NOT claim the cmd
progress bar (no ``progress_cb``) so background sync never fights a foreground
task's progress line; per-file errors surface in the returned summary and are
logged once at the end.
"""

from __future__ import annotations

from typing import Any, List

from unitport_sdk import Task

from application.service.cloud_sync import get_cloud_sync_service


class AutoSyncPushTask(Task):
    """One-shot uploader for a debounced batch of touched rel-paths."""

    def __init__(self, rel_paths: List[str]) -> None:
        super().__init__(name="cloud-auto-sync")
        # Copy — the controller may keep mutating its own pending set.
        self._rels: List[str] = list(rel_paths)

    def run(self) -> Any:  # type: ignore[override]
        svc = get_cloud_sync_service()
        plan = svc.plan_push_paths(self._rels)
        total = len(plan.entries)
        if total == 0:
            self.log_debug(
                f"auto-sync: {len(self._rels)} file(s) touched, "
                f"nothing to upload "
                f"({plan.skipped_unchanged} unchanged, "
                f"{plan.skipped_excluded} not-syncable, "
                f"{len(plan.skipped_oversize)} oversize)"
            )
            return {"phase": "auto_push", "ok": 0, "failed": 0, "total": 0}

        self.log_info(f"auto-sync: uploading {total} changed file(s)")
        # No progress_cb: keep the background sync off the cmd progress line.
        summary = svc.execute(plan)
        for err in (summary.get("errors") or [])[:20]:
            self.log_warning(f"auto-sync failed: {err}")
        ok = int(summary.get("ok", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        line = f"auto-sync: ok={ok}/{total} failed={failed}"
        if failed:
            self.log_warning(line)
        else:
            self.log_success(line)
        summary["phase"] = "auto_push"
        return summary


__all__ = ["AutoSyncPushTask"]
