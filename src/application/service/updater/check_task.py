"""CheckUpdateTask — TasksManager wrapper around ``UpdateService.check()``.

Submitted at Stage 2 of the boot pipeline alongside ``data_load`` so
the network round-trip happens off the main thread. Non-gating: a
failure here does NOT abort the rest of startup (the sidebar simply
keeps its "Latest" state).
"""

from __future__ import annotations

from typing import Any

from unitport_sdk import Task

from .service import get_update_service


class CheckUpdateTask(Task):
    def __init__(self, *, force: bool = False, name: str = "update_check") -> None:
        super().__init__(name=name)
        self._force = force

    def run(self) -> Any:  # type: ignore[override]
        self.log_info("checking for updates")
        release = get_update_service().check(force=self._force)
        if release is None:
            self.log_debug("no update available (or throttled / offline)")
        else:
            self.log_success(f"new version {release.version} available")
        return release


__all__ = ["CheckUpdateTask"]
