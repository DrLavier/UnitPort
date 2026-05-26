# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ApplyUpdateTask — TasksManager wrapper around ``UpdateService.apply()``.

Submitted when the user clicks Apply in ``UpdateAvailableDialog``. Heavy
work (download, extract, file IO) runs on a worker thread; progress is
broadcast through AppSignals so the modal progress dialog tracks it.
"""

from __future__ import annotations

from typing import Any

from unitport_sdk import Task

from .release_info import ApplyResult, ReleaseInfo
from .service import get_update_service


class ApplyUpdateTask(Task):
    def __init__(self, release: ReleaseInfo, name: str = "update_apply") -> None:
        super().__init__(name=name)
        self._release = release

    def run(self) -> Any:  # type: ignore[override]
        self.log_info(f"applying {self._release.tag}")

        def progress(frac: float, label: str) -> None:
            # progress_line drives the SDK's built-in cmd-screen progress
            # bar; the AppSignals broadcast (inside UpdateService.apply)
            # drives the modal dialog UI.
            try:
                self.progress_line(frac, label)
            except Exception:
                pass

        result: ApplyResult = get_update_service().apply(
            self._release, progress_cb=progress,
        )
        if result.ok:
            self.log_success(result.message or "update applied")
        else:
            self.log_warning(result.message or "update failed")
        return result


__all__ = ["ApplyUpdateTask"]
