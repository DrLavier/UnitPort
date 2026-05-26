# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ProjectStore — directory-level project scanner for the Project Files panel.

Scan root: ``Paths.PROJECTS_DIR`` (default ``Paths.USER_CONFIG_DIR / "projects"``,
overlayable via ``[Resources] projects_dir`` in system.ini/user.ini). A project is
any sub-directory containing a ``project.yaml`` manifest (schema_version: project_v2).

The store keeps an in-memory snapshot that the UI reads synchronously. The
snapshot is refreshed:
    * once at startup by ``UnitPortMain._data_load_body`` (worker thread);
    * on every ``MainWindow.refresh()`` (main thread, fast: ~ms per yaml);
    * on demand via the panel's refresh button.

Recents (``[Project] recent`` in user.ini) are kept for future "recently
opened" UI; ``add_recent`` now records project root directory paths.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices

from unitport_sdk import (
    Config,
    DataManager,
    Paths,
    log_info,
    log_warning,
    save_data,
)


_MANIFEST_NAME = "project.yaml"
_RECENT_LIMIT = 10
_INI_SECTION = "Project"
_INI_RECENT_KEY = "recent"
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass(frozen=True)
class ProjectInfo:
    path: Path           # project root directory
    name: str            # manifest.name, fallback to dir basename
    project_id: str      # manifest.project_id, fallback to dir basename
    updated_at: float    # epoch seconds; 0.0 if absent / unparseable
    manifest: Dict       # raw parsed yaml (frozen via dict copy)


def _parse_updated_at(raw) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, datetime):
        return raw.timestamp()
    s = str(raw).strip()
    if not s:
        return 0.0
    iso = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return 0.0


class ProjectStore(QObject):
    """Directory-level project list under ``Paths.PROJECTS_DIR`` (per-user)."""

    snapshot_changed = pyqtSignal(list)   # list[ProjectInfo]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._snapshot: List[ProjectInfo] = []

    # ----- paths ----------------------------------------------------------

    @property
    def _root(self) -> Path:
        """Always re-read ``Paths.PROJECTS_DIR`` so a USER_CONFIG_DIR
        relocate (UserPanel workspace gear) is honoured without reaching
        into the singleton to mutate state."""
        return Paths.PROJECTS_DIR

    def projects_root(self) -> Path:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_warning(f"[projects] cannot create {self._root}: {exc}")
        return self._root

    # ----- scan -----------------------------------------------------------

    def scan(self) -> List[ProjectInfo]:
        root = self.projects_root()
        if not root.exists():
            return []
        out: List[ProjectInfo] = []
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            manifest_path = child / _MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                manifest = DataManager.load(manifest_path)
            except Exception as exc:
                log_warning(f"[projects] cannot parse {manifest_path}: {exc}")
                continue
            if not isinstance(manifest, dict):
                log_warning(f"[projects] {manifest_path} is not a mapping; skipped")
                continue
            name = str(manifest.get("name") or child.name)
            project_id = str(manifest.get("project_id") or child.name)
            updated_at = _parse_updated_at(manifest.get("updated_at"))
            out.append(
                ProjectInfo(
                    path=child,
                    name=name,
                    project_id=project_id,
                    updated_at=updated_at,
                    manifest=dict(manifest),
                )
            )
        # Newest first; ties fall back to case-insensitive dir name.
        out.sort(key=lambda p: (-p.updated_at, p.path.name.lower()))
        return out

    def snapshot(self) -> List[ProjectInfo]:
        return list(self._snapshot)

    def refresh_snapshot(self) -> List[ProjectInfo]:
        self._snapshot = self.scan()
        self.snapshot_changed.emit(list(self._snapshot))
        return list(self._snapshot)

    def first_project(self) -> Optional[ProjectInfo]:
        if not self._snapshot:
            return None
        return self._snapshot[0]

    def find_by_path(self, path: Path) -> Optional[ProjectInfo]:
        try:
            target = path.resolve()
        except OSError:
            return None
        for info in self._snapshot:
            try:
                if info.path.resolve() == target:
                    return info
            except OSError:
                continue
        return None

    # ----- create -------------------------------------------------------

    def create_project(self, name: str) -> Optional[ProjectInfo]:
        """Create a minimal v2 project under ``<projects_root>/<project_id>/``.

        ``project_id`` is the lowercase form of ``name`` (so display name
        ``NEW_TEST`` lands at directory ``new_test/``). Rejects names that
        clash case-insensitively with an existing project. Returns the
        freshly-scanned :class:`ProjectInfo` on success, or None on
        validation / IO failure (a warning is logged).
        """
        clean = name.strip()
        if not _NAME_RE.match(clean):
            log_warning(
                f"[projects] create_project: invalid name {clean!r} "
                "(letters, digits, _ and - only)"
            )
            return None
        project_id = clean.lower()
        # Case-insensitive collision check against current snapshot.
        for p in self._snapshot:
            if (
                p.project_id.lower() == project_id
                or p.name.strip().lower() == clean.lower()
            ):
                log_warning(
                    f"[projects] create_project: {clean!r} clashes with "
                    f"existing project {p.name!r} ({p.path})"
                )
                return None
        root = self.projects_root()
        target = root / project_id
        if target.exists():
            log_warning(
                f"[projects] create_project: directory already exists {target}"
            )
            return None
        manifest_path = target / _MANIFEST_NAME
        now_iso = datetime.now(timezone.utc).isoformat()
        manifest: Dict = {
            "schema_version": "project_v2",
            "project_id": project_id,
            "uuid": uuid.uuid4().hex,
            "name": clean,
            "created_at": now_iso,
            "updated_at": now_iso,
            "owner_id": None,
            "robot": {"brand": "", "model": ""},
            "assets": {
                "workflows": [],
                "checkpoints": [],
                "bundles": [],
            },
            "active_experiment": "",
            "experiments": [],
        }
        try:
            target.mkdir(parents=True, exist_ok=False)
            save_data(manifest_path, manifest)
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[projects] create_project: write failed at {manifest_path}: {exc}"
            )
            return None
        log_info(f"[projects] created project {clean!r} at {target}")
        self.refresh_snapshot()
        for p in self._snapshot:
            if p.project_id == project_id:
                return p
        log_warning(
            f"[projects] create_project: post-create snapshot missing "
            f"{project_id!r}; manifest may be malformed"
        )
        return None

    # ----- open in OS file browser ---------------------------------------

    def open_in_explorer(self, path: Optional[Path] = None) -> bool:
        target = path or self.projects_root()
        if not target.exists():
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    # ----- recents (persisted in user.ini) -------------------------------

    def recent(self) -> List[Path]:
        raw = Config.get_value(_INI_SECTION, _INI_RECENT_KEY, fallback="")
        if not raw:
            return []
        items: List[Path] = []
        for line in str(raw).splitlines():
            s = line.strip()
            if not s:
                continue
            p = Path(s)
            if p.exists():
                items.append(p)
        return items[:_RECENT_LIMIT]

    def add_recent(self, path: Path) -> None:
        try:
            entries = [str(p) for p in self.recent() if Path(p).resolve() != path.resolve()]
        except OSError:
            entries = []
        entries.insert(0, str(path))
        entries = entries[:_RECENT_LIMIT]
        try:
            Config.set_value(_INI_SECTION, _INI_RECENT_KEY, os.linesep.join(entries))
        except Exception as exc:
            log_warning(f"[projects] persist recent failed: {exc}")


_instance: Optional[ProjectStore] = None


def get_project_store() -> ProjectStore:
    global _instance
    if _instance is None:
        _instance = ProjectStore()
    return _instance


# ---------------------------------------------------------------------------
# Current project — module-level single source of truth.
#
# The historical pattern (e.g. RunSourceSelector) accepts a ``project_info_provider``
# callable so widgets can resolve the bound ProjectInfo on demand. That stays
# valid; this accessor is the canonical write side. ``set_current_project_info``
# both updates the module-level state and broadcasts ``AppSignals.project_changed``
# so subscribers (top-row dropdowns, training tasks at submit time, future deploy
# panel) can react without each having to know about MainWindow.
# ---------------------------------------------------------------------------

_current_project_info: Optional[ProjectInfo] = None


def current_project_info() -> Optional[ProjectInfo]:
    """Return the currently-bound project, or ``None`` if no project is active.

    Safe to call from any thread (read-only). Training tasks call this at
    construction time so the project captured stays stable through ``run()``
    even if the user switches projects mid-run.
    """
    return _current_project_info


def set_current_project_info(info: Optional[ProjectInfo]) -> None:
    """Set the current project + emit ``AppSignals.project_changed``.

    No-op (no emit) when ``info`` resolves to the same project root as the
    previous value — avoids UI churn on idempotent rebinds.
    """
    global _current_project_info
    prev = _current_project_info

    def _same(a: Optional[ProjectInfo], b: Optional[ProjectInfo]) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            return a.path.resolve() == b.path.resolve()
        except OSError:
            return a.path == b.path

    if _same(prev, info):
        return
    _current_project_info = info

    # Local import — service.signals imports nothing from this module, but
    # avoiding the top-level dep keeps store.py importable from contexts
    # where AppSignals' QObject base hasn't been touched yet.
    try:
        from application.service.signals import get_app_signals
        get_app_signals().project_changed.emit(info)
    except Exception as exc:                              # pragma: no cover
        log_warning(f"[projects] project_changed emit failed: {exc}")


__all__ = [
    "ProjectInfo",
    "ProjectStore",
    "get_project_store",
    "current_project_info",
    "set_current_project_info",
]
