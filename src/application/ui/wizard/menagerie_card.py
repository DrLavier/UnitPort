# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Wizard-side shim — keeps the historical import path live.

The card grid widgets moved to :mod:`application.ui.widgets.menagerie.card_grid`
so the sidebar Robot Asset browser dialog can share them. The wizard imports
keep working unchanged through the re-export below.

The QThread-based ``IconFetchWorker`` stays here because the wizard runs
before the Qt application is fully up — the SDK's ``TasksManager`` is not
yet primed at that point, so we keep the known-good ``QThread`` worker for
the wizard. The sidebar dialog (which opens long after start-up) uses the
new ``MenagerieIconFetchTask`` under :mod:`application.service.menagerie.tasks`
instead, so both paths run on the right primitive for their lifecycle.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal

from application.service.models import menagerie_manager as mm
from application.ui.widgets.menagerie.card_grid import (
    CARD_W,
    CARD_H,
    ICON_W,
    ICON_H,
    GRID_COLUMNS,
    GRID_HSPACE,
    GRID_VSPACE,
    GRID_MARGIN,
    MenagerieCard,
    MenagerieCardGrid,
)


class IconFetchWorker(QThread):
    """Background worker -- fills the icon cache for a list of names.

    Steps: load the cached ``_index.json``, refresh from GitHub Trees API if
    the index is incomplete, then download each missing PNG via
    ``ensure_cached_icon`` on a thread pool. Wizard-only — the sidebar dialog
    submits :class:`MenagerieIconFetchTask` through ``get_tasks_manager``.
    """

    icon_ready = pyqtSignal(str, str)        # name, cache_path
    progress = pyqtSignal(int, int)          # done, total
    failed = pyqtSignal(str)                 # human-readable error
    finished_all = pyqtSignal()

    MAX_WORKERS = 8

    def __init__(self, names: List[str], parent=None) -> None:
        super().__init__(parent)
        self._names = list(names)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            index = mm.load_icon_index()
            need_index_refresh = (
                not index
                or any(n not in index for n in self._names)
            )
            if need_index_refresh:
                try:
                    fresh = mm.fetch_icon_index()
                    if fresh:
                        index = fresh
                        mm.save_icon_index(index)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(f"icon index fetch failed: {exc}")

            todo = [
                n for n in self._names
                if not mm.has_cached_icon(n)
            ]
            total = len(todo)
            done = 0
            if total == 0:
                self.finished_all.emit()
                return

            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
                futures = {
                    pool.submit(
                        mm.ensure_cached_icon,
                        name,
                        index.get(name),
                    ): name
                    for name in todo
                }
                for fut in as_completed(futures):
                    if self._cancelled:
                        break
                    name = futures[fut]
                    try:
                        path = fut.result()
                    except Exception:  # noqa: BLE001
                        path = None
                    done += 1
                    self.progress.emit(done, total)
                    if path is not None:
                        self.icon_ready.emit(name, str(path))

            self.finished_all.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


__all__ = [
    "CARD_W", "CARD_H", "ICON_W", "ICON_H",
    "GRID_COLUMNS", "GRID_HSPACE", "GRID_VSPACE", "GRID_MARGIN",
    "MenagerieCard", "MenagerieCardGrid", "IconFetchWorker",
]
