# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Task subclasses that adapt menagerie_manager's pure-Python surface.

Three workers cover the sidebar Menagerie browser dialog's lifecycle:

* :class:`MenagerieSparseAddTask` — git sparse-checkout to materialise the
  selected package set under ``custom_mods/models/menagerie/``.
* :class:`MenagerieRefreshTask`   — fetch the live package list from GitHub
  (falls back to the snapshot when the API is unreachable).
* :class:`MenagerieIconFetchTask` — populate the per-package PNG cache.

All three submit through ``get_tasks_manager().submit(task)``; cancellation
honours :meth:`Task.check_cancelled`; logs auto-prefix ``[<task.name>]`` via
:meth:`Task.log_info`.

Note: the first-launch wizard keeps using the QThread-based
``IconFetchWorker`` (see :mod:`application.ui.wizard.menagerie_card`) because
it runs before the TasksManager is fully primed. Both code paths target the
same on-disk cache, so they interleave safely.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from unitport_sdk import Task

from application.service.models import menagerie_manager as _mm


class MenagerieSparseAddTask(Task):
    """Sparse-checkout add for selected packages.

    Wraps :func:`menagerie_manager.add_packages` so each git output line
    is forwarded to ``self.log_info`` and surfaces in ``CmdLogWidget``.
    Returns the list of newly-installed dir names (passed-through input).
    """

    def __init__(self, packages: Sequence[str]) -> None:
        super().__init__(name=f"menagerie-sparse-add[{len(packages)}]")
        self._packages = list(packages)

    def run(self) -> List[str]:
        self.check_cancelled()
        if not self._packages:
            return []

        def on_output(line: str) -> None:
            self.log_info(line)
            self.check_cancelled()

        try:
            installed = _mm.add_packages(self._packages, on_output=on_output)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "git executable not found on PATH; install Git for Windows "
                "(https://gitforwindows.org/) and restart UnitPort"
            ) from exc

        self.progress_line(1.0, f"installed {len(installed)} package(s)")
        return list(installed)


class MenagerieRefreshTask(Task):
    """Fetch the live menagerie package list from GitHub.

    Best-effort: callers should fall back to ``MENAGERIE_PACKAGES_SNAPSHOT``
    if this task fails (network down, GitHub rate-limit, etc.).
    """

    def __init__(self) -> None:
        super().__init__(name="menagerie-refresh")

    def run(self) -> List[str]:
        self.check_cancelled()
        names = _mm.fetch_remote_packages()
        self.log_info(f"fetched {len(names)} packages from GitHub")
        return names


class MenagerieIconFetchTask(Task):
    """Cache preview PNGs for the given package names.

    Uses an in-task ThreadPoolExecutor(8) to saturate network — most icons
    are <50 KB, so this completes well under 5s for ~70 packages on a warm
    connection. Yields per-icon progress via ``self.progress_line``.

    Returns ``{name: cache_path_or_None}`` so the dialog can refresh icons
    in one pass once the task finishes.
    """

    MAX_WORKERS = 8

    def __init__(self, names: Sequence[str]) -> None:
        super().__init__(name=f"menagerie-icons[{len(names)}]")
        self._names = list(names)

    def run(self) -> Dict[str, Optional[str]]:
        self.check_cancelled()
        if not self._names:
            return {}

        index: Dict[str, str] = _mm.load_icon_index()
        need_refresh = (not index) or any(n not in index for n in self._names)
        if need_refresh:
            try:
                fresh = _mm.fetch_icon_index()
                if fresh:
                    index = fresh
                    _mm.save_icon_index(index)
            except Exception as exc:  # noqa: BLE001
                # Network down — proceed with whatever local cache survives;
                # cards already-on-disk still render via load_icon_index().
                self.log_warning(f"icon index refresh failed: {exc}")

        todo = [n for n in self._names if not _mm.has_cached_icon(n)]
        results: Dict[str, Optional[str]] = {n: None for n in self._names}
        for n in self._names:
            if _mm.has_cached_icon(n):
                results[n] = str(_mm.cached_icon_path(n))

        if not todo:
            self.progress_line(1.0, "icons already cached")
            return results

        total = len(todo)
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futs = {
                pool.submit(_mm.ensure_cached_icon, n, index.get(n)): n
                for n in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                self.check_cancelled()
                n = futs[fut]
                try:
                    p = fut.result()
                    results[n] = str(p) if p else None
                except Exception as exc:  # noqa: BLE001
                    self.log_warning(f"icon fetch failed for {n}: {exc}")
                    results[n] = None
                self.progress_line(i / total, f"{i}/{total}")
        return results


__all__ = [
    "MenagerieSparseAddTask",
    "MenagerieRefreshTask",
    "MenagerieIconFetchTask",
]
