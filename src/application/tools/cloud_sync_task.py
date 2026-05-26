# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CloudSyncTask — Task wrapper around CloudSyncService for the
TasksManager queue.

Three phases share one class so the same submit path can be used from
the UserPanel buttons (push / pull) and the startup self-check hook:

    - "self_check" : list-only, no transfer. Used by main.py startup.
    - "push"       : planner + uploader. Triggered by UserPanel "Push".
    - "pull"       : planner + downloader. Triggered by UserPanel "Pull".

The Task obeys the SDK contract:
    - logs through ``self.log_*`` (auto-prefixed [cloud-sync-<phase>])
    - calls ``self.check_cancelled()`` between files
    - never touches ``time.sleep``

The heavy lifting is in CloudSyncService; this class is the worker-thread
adapter that hooks progress reporting into the Task's signals.
"""

from __future__ import annotations

from typing import Any

from unitport_sdk import Task

from application.service.cloud_sync import get_cloud_sync_service


class CloudSyncTask(Task):
    """One-shot cloud-sync worker for the TasksManager queue.

    ``phase`` must be one of ``"self_check"`` / ``"push"`` / ``"pull"``.
    Anything else raises so a typo at the call site fails loudly during
    development rather than silently no-oping at runtime.
    """

    _ALLOWED = frozenset({"self_check", "push", "pull", "usage"})

    def __init__(self, phase: str) -> None:
        if phase not in self._ALLOWED:
            raise ValueError(
                f"CloudSyncTask: unknown phase {phase!r}; "
                f"expected one of {sorted(self._ALLOWED)}"
            )
        super().__init__(name=f"cloud-sync-{phase}")
        self._phase = phase

    def run(self) -> Any:  # type: ignore[override]
        svc = get_cloud_sync_service()
        if self._phase == "usage":
            usage = svc.fetch_storage_usage()
            if usage is None:
                self.log_debug("usage: unavailable")
                return {"phase": "usage", "used_bytes": 0, "max_bytes": 0}
            used, total = usage
            self.log_debug(f"usage: {used}/{total} bytes")
            return {
                "phase": "usage", "used_bytes": used, "max_bytes": total,
            }

        if self._phase == "self_check":
            # Self-check is a single HTTP round-trip with no progress loop;
            # one info line for the final count is enough.
            self.log_debug("listing remote prefix")
            rows = svc.list_remote()
            self.log_info(f"cloud has {len(rows)} object(s)")
            return {"phase": "self_check", "remote_count": len(rows)}

        if self._phase == "push":
            self.log_debug("planning push")
            plan = svc.plan_push()
            total = len(plan.entries)
            self.log_debug(
                f"plan: upload={total} skip_unchanged={plan.skipped_unchanged} "
                f"skip_excluded={plan.skipped_excluded} "
                f"skip_oversize={len(plan.skipped_oversize)} "
                f"skip_runs_topn={plan.skipped_runs_topn}"
            )
            if total == 0:
                self.log_info(
                    f"push: nothing to upload "
                    f"({plan.skipped_unchanged} already up to date)"
                )
                return {
                    "phase": "push", "ok": 0, "failed": 0,
                    "skipped": 0, "total": 0,
                }
            # progress_line owns the cmd line for the duration of execute().
            # CloudSyncService.execute does NOT log inside the loop; any
            # per-file errors come back in summary["errors"].
            summary = svc.execute(plan, progress_cb=self._progress_cb)
            self._emit_summary("push", total, summary)
            summary["phase"] = "push"
            return summary

        # phase == "pull"
        self.log_debug("planning pull")
        plan = svc.plan_pull()
        total = len(plan.entries)
        self.log_debug(
            f"plan: download={total} "
            f"skip_unchanged={plan.skipped_unchanged}"
        )
        if total == 0:
            self.log_info(
                f"pull: nothing to download "
                f"({plan.skipped_unchanged} already up to date)"
            )
            return {
                "phase": "pull", "ok": 0, "failed": 0,
                "skipped": 0, "total": 0,
            }
        summary = svc.execute(plan, progress_cb=self._progress_cb)
        self._emit_summary("pull", total, summary)
        summary["phase"] = "pull"
        return summary

    def _emit_summary(self, verb: str, total: int, summary: dict) -> None:
        """Log accumulated errors + a one-line success/warn summary.

        Called AFTER ``progress_line(1.0, ...)`` has fired (execute()
        terminates with ratio=1.0 on the last entry), so the cmd
        progress slot is released and these lines won't fight with the
        overwrite-style bar.
        """
        for err in (summary.get("errors") or [])[:20]:
            self.log_warning(f"{verb} failed: {err}")
        ok = int(summary.get("ok", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        skipped = int(summary.get("skipped", 0) or 0)
        line = (
            f"{verb} ok={ok}/{total} failed={failed} skipped={skipped}"
        )
        if failed:
            self.log_warning(line)
        else:
            self.log_success(line)

    def _progress_cb(self, done: int, total: int, current_key: str) -> None:
        """Bridge CloudSyncService progress into the SDK progress line.

        ``progress_line`` is the only thing allowed to emit inside the
        loop; any log_info/log_warning here would interleave with its
        overwrite-style cmd line and produce stale leftover bars.
        """
        self.check_cancelled()
        ratio = float(done) / float(total) if total > 0 else 1.0
        self.progress_line(ratio, f"{done}/{total} {current_key}")


__all__ = ["CloudSyncTask"]
