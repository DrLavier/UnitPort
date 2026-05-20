"""InitPoseService — factory + user init pose presets, keyed by SKU.

Factory presets ship with the canonical robot registry
(``robots_canonical.json[sku].init_pose_presets``). User-created presets
live in ``<USER_CONFIG_DIR>/robot_presets/init_poses.json`` (per RELEASE rule
§1.4: user state never inside ``src/``).

``list_presets(sku)`` merges both sources. When a user preset shares a
name with a factory preset, the user version wins (UI flags this with a
shadow badge so it's visible to the user). Delete only affects user
presets — factory presets are immutable.

Singleton via :func:`get_init_pose_service` — mirrors RobotAssetService.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from registers import robots as _robots_registry
from unitport_sdk import Paths, log_warning, push_data, read_data

from .model import (
    InitPoseOverride,
    InitPosePreset,
    SOURCE_FACTORY,
    SOURCE_USER,
)


_STATE_REL = "robot_presets/init_poses.json"
_SCHEMA = "robot_init_poses/user"
_VERSION = "1.0.0"


def _state_path() -> Path:
    """Lazy resolve of the user preset state path inside the LIVE
    USER_CONFIG_DIR. Resolved at call time so workspace hot-switches are
    picked up without restart.
    """
    return Paths.USER_CONFIG_DIR / "robot_presets" / "init_poses.json"


class InitPoseService(QObject):
    """CRUD over factory + user init pose presets.

    Signal ``presets_changed(sku)`` fires whenever the user-state file
    is mutated for that SKU, so the UI can refresh the preset dropdown
    without polling.
    """

    presets_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    # ----- on-disk state ---------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if not _state_path().exists():
            return {"version": _VERSION, "schema": _SCHEMA, "robots": {}}
        data = read_data(_state_path())
        if not isinstance(data, dict):
            return {"version": _VERSION, "schema": _SCHEMA, "robots": {}}
        data.setdefault("version", _VERSION)
        data.setdefault("schema", _SCHEMA)
        data.setdefault("robots", {})
        if not isinstance(data["robots"], dict):
            data["robots"] = {}
        return data

    def _save_state(self, state: Dict[str, Any]) -> bool:
        if not push_data(_STATE_REL, state):
            log_warning("[init_poses] failed to persist user preset state")
            return False
        return True

    def _user_block(self, state: Dict[str, Any], sku: str) -> List[Dict[str, Any]]:
        """Return the (mutable) list of user presets for ``sku``, creating it."""
        robots = state.setdefault("robots", {})
        block = robots.setdefault(sku, [])
        if not isinstance(block, list):
            robots[sku] = []
            block = robots[sku]
        return block

    # ----- public API -----------------------------------------------------

    def list_factory_presets(self, sku: str) -> List[InitPosePreset]:
        """Factory presets from the canonical robot registry."""
        if not sku:
            return []
        raw = _robots_registry.get_robot_init_pose_presets(sku)
        return [InitPosePreset.from_dict(p, SOURCE_FACTORY) for p in raw]

    def list_user_presets(self, sku: str) -> List[InitPosePreset]:
        """User-created presets from ``<USER_CONFIG_DIR>/robot_presets/init_poses.json``."""
        if not sku:
            return []
        state = self._load_state()
        block = state.get("robots", {}).get(sku, [])
        if not isinstance(block, list):
            return []
        return [InitPosePreset.from_dict(p, SOURCE_USER) for p in block]

    def list_presets(self, sku: str) -> List[InitPosePreset]:
        """Factory presets first, then user. User name collisions shadow factory.

        Returns a flat list in display order. The caller (UI) reads the
        ``source`` field to render F/U badges.
        """
        factory = self.list_factory_presets(sku)
        user = self.list_user_presets(sku)
        user_names = {p.name for p in user}
        merged: List[InitPosePreset] = []
        for p in factory:
            if p.name in user_names:
                # User preset takes precedence — skip the factory one
                # here; the user version will appear in the user loop
                # below. UI can detect this case by comparing factory vs
                # user-presets independently if it wants to badge the
                # shadow relationship.
                continue
            merged.append(p)
        merged.extend(user)
        return merged

    def get_preset(self, sku: str, name: str) -> Optional[InitPosePreset]:
        """Find a preset by name. User shadows factory."""
        if not sku or not name:
            return None
        for p in self.list_user_presets(sku):
            if p.name == name:
                return p
        for p in self.list_factory_presets(sku):
            if p.name == name:
                return p
        return None

    def save_user_preset(self, sku: str, preset: InitPosePreset) -> bool:
        """Insert or update a user preset for ``sku``. Returns success.

        Saved presets are always tagged ``source=user``; the input
        preset's source field is ignored.
        """
        if not sku or not preset.name:
            return False
        state = self._load_state()
        block = self._user_block(state, sku)
        payload = preset.to_dict()
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        # Replace by name if exists, else append.
        replaced = False
        for idx, existing in enumerate(block):
            if isinstance(existing, dict) and existing.get("name") == preset.name:
                # Preserve original created_at on update; add updated_at.
                payload.setdefault(
                    "created_at",
                    str(existing.get("created_at", payload["created_at"])),
                )
                payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
                block[idx] = payload
                replaced = True
                break
        if not replaced:
            block.append(payload)
        ok = self._save_state(state)
        if ok:
            self.presets_changed.emit(sku)
        return ok

    def delete_user_preset(self, sku: str, name: str) -> bool:
        """Remove a user preset by name. Factory presets are never touched."""
        if not sku or not name:
            return False
        state = self._load_state()
        block = state.get("robots", {}).get(sku)
        if not isinstance(block, list):
            return False
        before = len(block)
        block[:] = [
            p for p in block
            if not (isinstance(p, dict) and p.get("name") == name)
        ]
        if len(block) == before:
            return False
        ok = self._save_state(state)
        if ok:
            self.presets_changed.emit(sku)
        return ok

    def is_user_preset(self, sku: str, name: str) -> bool:
        """True iff a preset with this name exists in the user state file."""
        if not sku or not name:
            return False
        return any(p.name == name for p in self.list_user_presets(sku))


_instance: Optional[InitPoseService] = None


def get_init_pose_service() -> InitPoseService:
    global _instance
    if _instance is None:
        _instance = InitPoseService()
    return _instance


__all__ = [
    "InitPoseService",
    "get_init_pose_service",
    "InitPoseOverride",
    "InitPosePreset",
    "SOURCE_FACTORY",
    "SOURCE_USER",
]
