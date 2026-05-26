# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Data models for the init pose override feature.

InitPosePreset is the persisted shape (factory in robots_canonical.json,
user in <USER_CONFIG_DIR>/robot_presets/init_poses.json).

InitPoseOverride is the immutable in-memory snapshot the UI hands to a
runtime Task. The Task never looks presets back up by name — once the
user clicks Review/Run, the value snapshot is what runs, even if the
underlying preset is later edited.

Both shapes key joint values by **IR role name** (hip_FL, thigh_RR, ...)
so the data is brand- and bundle-permutation-independent. Translation
into the bundle's joint order happens at the substrate boundary in
``InitPoseOverride.to_bundle_order``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple


SOURCE_FACTORY = "factory"
SOURCE_USER = "user"


@dataclass(frozen=True)
class InitPosePreset:
    """A named init pose, keyed by IR role names.

    ``source`` distinguishes shipped factory presets (read-only) from
    user-created ones (deletable). UI uses it to enable/disable the
    Delete button and to render an F/U badge.
    """

    name: str
    description: str
    base_pos: Tuple[float, float, float]
    joint_pos_by_ir: Mapping[str, float]
    source: str = SOURCE_USER

    @staticmethod
    def from_dict(raw: dict, source: str) -> "InitPosePreset":
        """Parse a raw dict (from canonical JSON or user state file) into a preset.

        Tolerates missing fields with reasonable defaults so a partial
        preset still loads — UI shows whatever it has. Caller is
        responsible for source attribution.
        """
        name = str(raw.get("name", "")).strip()
        description = str(raw.get("description", "") or "")
        bp_raw = raw.get("base_pos") or [0.0, 0.0, 0.0]
        try:
            base_pos = (
                float(bp_raw[0]),
                float(bp_raw[1]),
                float(bp_raw[2]),
            )
        except (TypeError, ValueError, IndexError):
            base_pos = (0.0, 0.0, 0.0)
        jp_raw = raw.get("joint_pos_by_ir") or {}
        joint_pos_by_ir: dict = {}
        if isinstance(jp_raw, dict):
            for k, v in jp_raw.items():
                try:
                    joint_pos_by_ir[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        return InitPosePreset(
            name=name,
            description=description,
            base_pos=base_pos,
            joint_pos_by_ir=joint_pos_by_ir,
            source=str(source) or SOURCE_USER,
        )

    def to_dict(self) -> dict:
        """Persistable shape — matches the JSON in robots_canonical.json
        and the user state file. ``source`` is excluded because it is
        derived from the storage layer, not stored in the file.
        """
        return {
            "name": self.name,
            "description": self.description,
            "base_pos": [float(v) for v in self.base_pos],
            "joint_pos_by_ir": {
                str(k): float(v) for k, v in self.joint_pos_by_ir.items()
            },
        }


@dataclass(frozen=True)
class InitPoseOverride:
    """Immutable snapshot the UI passes to a runtime Task.

    Decouples the runtime from the preset registry: once the Task has
    this object, it never re-queries the service. The UI freezes the
    current spinbox values into this object the moment the user clicks
    Review/Run.
    """

    base_pos: Tuple[float, float, float]
    joint_pos_by_ir: Mapping[str, float]

    @staticmethod
    def from_preset(preset: InitPosePreset) -> "InitPoseOverride":
        return InitPoseOverride(
            base_pos=tuple(float(v) for v in preset.base_pos),  # type: ignore[arg-type]
            joint_pos_by_ir=dict(preset.joint_pos_by_ir),
        )

    def to_bundle_order(
        self,
        bundle_ir_roles: Sequence[str],
        default_joint_pos: Sequence[float],
    ) -> List[float]:
        """Reorder ``joint_pos_by_ir`` into the bundle's joint order.

        For each IR role in ``bundle_ir_roles``, returns the matching
        override angle if present, else the corresponding fallback from
        ``default_joint_pos``. Extra IR roles in the override (not in
        the bundle) are silently ignored — they belong to a different
        SKU or schema and are not relevant here.
        """
        defaults = list(default_joint_pos)
        out: List[float] = []
        for i, role in enumerate(bundle_ir_roles):
            role_key = str(role)
            if role_key in self.joint_pos_by_ir:
                out.append(float(self.joint_pos_by_ir[role_key]))
            elif i < len(defaults):
                out.append(float(defaults[i]))
            else:
                out.append(0.0)
        return out

    def missing_ir_roles(self, bundle_ir_roles: Sequence[str]) -> List[str]:
        """List IR roles in the bundle that the override does NOT cover.

        Used by the UI to surface a "partial coverage — fallback to
        bundle defaults" hint after preset apply.
        """
        return [
            str(r) for r in bundle_ir_roles
            if str(r) not in self.joint_pos_by_ir
        ]


__all__ = [
    "InitPosePreset",
    "InitPoseOverride",
    "SOURCE_FACTORY",
    "SOURCE_USER",
]
