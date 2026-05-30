# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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

from registers import init_pose_presets as _init_pose_presets_registry
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
        """Factory presets for ``sku`` from the family-default registry
        layered with the SKU-override layer from
        ``robots_canonical.json[sku].init_pose_presets``.

        Three-layer factory source:

          1. family-default -- ``registers.init_pose_presets.list_presets``
             returns name + base_pos + description for the SKU's family.
             ``joint_pos_by_ir`` is intentionally absent at this layer
             because init pose joint angles are per-robot physical
             quantities; a quadruped "stand" pose's thigh / calf angles
             depend on segment lengths, so there is no valid family-wide
             canonical to seed from.
          2. SKU-override -- ``robots_canonical[sku].init_pose_presets``
             carries the per-SKU ``joint_pos_by_ir`` (and may also
             override the base_pos and description). SKU-level entries
             with the same name as a family-default shadow it integrally
             (no dict merge -- the SKU-override replaces the entire
             preset).
          3. The user layer is composed on top by :meth:`list_presets`
             (factory + user).

        Selecting a family preset whose SKU has no SKU-override carries
        an empty ``joint_pos_by_ir``; apply-time consumers
        (RegistryPresetPickerRow._InitPosePresetAdapter.on_select,
        InitPoseSubsection apply) MUST fail-loud rather than fill
        zero / family-default angles -- the registry intentionally
        offers nothing to fall back to (§8 silent-fallback ban).
        """
        if not sku:
            return []
        family = self._resolve_sku_family(sku)
        family_presets = self._family_default_presets(family)
        sku_overrides_raw = list(
            _robots_registry.get_robot_init_pose_presets(sku) or []
        )
        # Index SKU-override by name; non-dict entries silently skipped
        # at this layer (robots.py validates the raw shape upstream).
        sku_overrides: Dict[str, Dict[str, Any]] = {}
        for raw in sku_overrides_raw:
            if not isinstance(raw, dict):
                continue
            nm = str(raw.get("name", "") or "")
            if nm:
                sku_overrides[nm] = raw
        merged: List[InitPosePreset] = []
        seen_names: set = set()
        for fp in family_presets:
            if fp["name"] in sku_overrides:
                merged.append(InitPosePreset.from_dict(
                    sku_overrides[fp["name"]], SOURCE_FACTORY,
                ))
            else:
                # Family-default only -- no joint angles available.
                # apply-time consumers raise on this state.
                merged.append(InitPosePreset(
                    name=fp["name"],
                    description=fp["description"],
                    base_pos=fp["base_pos"],
                    joint_pos_by_ir={},
                    source=SOURCE_FACTORY,
                ))
            seen_names.add(fp["name"])
        for name, raw in sku_overrides.items():
            if name in seen_names:
                continue
            merged.append(InitPosePreset.from_dict(raw, SOURCE_FACTORY))
        return merged

    @staticmethod
    def _resolve_sku_family(sku: str) -> str:
        """Return the first family declared on the SKU's canonical entry.

        Returns ``""`` when the SKU is unknown or has no families field.
        At list time this is a soft failure (empty preset list); apply
        time is where the fail-loud directive (§8) lives.
        """
        try:
            spec = _robots_registry.get_robot_spec(sku)
        except Exception:
            return ""
        families = list(getattr(spec, "families", []) or [])
        return str(families[0]) if families else ""

    @staticmethod
    def _family_default_presets(family: str) -> List[Dict[str, Any]]:
        """Family-default presets as raw dicts.

        Returned shape mirrors the catalog (name + description + base_pos
        as a 3-tuple) so the merge loop in :meth:`list_factory_presets`
        treats family-default and SKU-override symmetrically. Empty
        list when the family is unknown / unresolvable (matches the
        soft-fail-at-list-time semantics of :meth:`_resolve_sku_family`).
        """
        if not family:
            return []
        try:
            registry_presets = (
                _init_pose_presets_registry.list_presets(family)
            )
        except _init_pose_presets_registry.InitPosePresetValidationError:
            return []
        return [
            {
                "name": p.name,
                "description": p.description,
                "base_pos": p.base_pos,
            }
            for p in registry_presets
        ]

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
