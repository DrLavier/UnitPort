#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — Training contract compatibility checker.

TrainingCompatibilityChecker
    Compares a TrainingAssetEntry (base checkpoint being resumed / warm-started)
    against the current canvas TrainingJobSpec and returns a list of
    CompatResult objects indicating mismatches.

Levels
------
PASS  — no issue
WARN  — mismatch is recoverable (training still makes sense but may under-perform)
FAIL  — mismatch would corrupt or crash training

Checks performed
----------------
framework     FAIL  — only SB3-on-SB3 is supported in Phase D
algorithm     FAIL  — SAC checkpoint cannot be resumed into PPO and vice-versa
obs_dim       FAIL  — observation vector size mismatch breaks the network forward pass
action_type   WARN  — torque vs position is conceptually different but training
                       will run (policy will learn; deployment may be wrong)
robot_type    WARN  — cross-robot resume is unusual but not an error

No Qt.  No SB3.  No file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.system.training.training_asset_registry import TrainingAssetEntry
    from src.system.training.training_spec import TrainingJobSpec


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class CompatLevel(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CompatResult:
    """One compatibility issue (or explicit pass) for a single field."""

    level:   CompatLevel
    field:   str    # human-readable field name, e.g. "obs_dim"
    message: str    # full description


# ---------------------------------------------------------------------------
# Observation dimension helpers
# ---------------------------------------------------------------------------

# Per-component observation dimensions.
# Components whose dim depends on the robot's action_dim are marked None.
_OBS_COMPONENT_DIM: Dict[str, Optional[int]] = {
    # joint-space components (= action_dim per robot)
    "joint_pos":         None,
    "joint_positions":   None,
    "joint_vel":         None,
    "joint_velocities":  None,
    "previous_action":   None,
    "last_action":       None,
    # reference motion obs (auto-injected when ReferenceMotionNode is connected)
    "reference_joint_positions":  None,   # = action_dim (num_joints)
    "ref_joint_pos":              None,
    "reference_joint_velocities": None,   # = action_dim (num_joints)
    "ref_joint_vel":              None,
    # fixed-size components
    "imu":               6,    # 3-axis accel + 3-axis gyro
    "base_angular_velocity": 3,
    "projected_gravity": 3,
    "velocity_command":  3,
    "command":           4,    # v_x, v_y, yaw_rate, height → community go2 layout
    "base_lin_vel":      3,
    "base_ang_vel":      3,
    "phase_sin_cos":     2,
    "phase":             2,
}

# Action dim by robot type (used when obs component dim = action_dim)
_ACTION_DIM_BY_ROBOT: Dict[str, int] = {
    "go2": 12, "go1": 12, "a1": 12, "b1": 12, "b2": 12,
    "h1": 10, "h1_2": 10, "g1": 23, "spot": 12, "humanoid": 10,
}


def compute_canvas_obs_dim(
    obs_components: List[str],
    robot_type: str,
    action_dim: int = 0,
) -> int:
    """
    Estimate the canvas observation vector dimension from component names.

    Returns 0 when the component list is empty or contains unknown names only
    (caller should treat 0 as "unknown — skip check").

    *action_dim*: when > 0, used directly instead of the static lookup table.
    """
    if action_dim <= 0:
        action_dim = _ACTION_DIM_BY_ROBOT.get(robot_type.lower(), 12)
    total = 0
    for comp in obs_components:
        dim = _OBS_COMPONENT_DIM.get(comp.strip().lower())
        if dim is None:
            # joint-space component
            total += action_dim
        elif dim == 0:
            pass   # unknown / future component — skip
        else:
            total += dim
    return total


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class TrainingCompatibilityChecker:
    """
    Check whether *entry* can be safely resumed/warm-started with *spec*.

    Usage::

        checker = TrainingCompatibilityChecker()
        results = checker.check(entry, spec)
        fails  = [r for r in results if r.level == CompatLevel.FAIL]
        warns  = [r for r in results if r.level == CompatLevel.WARN]

    Results are sorted: FAILs first, then WARNs, then PASSes.
    """

    def check(
        self,
        entry: "TrainingAssetEntry",
        spec:  "TrainingJobSpec",
    ) -> List[CompatResult]:
        """Return all compatibility results for (entry, spec)."""
        results: List[CompatResult] = []

        results.append(self._check_framework(entry, spec))
        results.append(self._check_algorithm(entry, spec))
        results.append(self._check_obs_dim(entry, spec))
        results.append(self._check_action_type(entry, spec))
        results.append(self._check_robot_type(entry, spec))

        # Sort: FAIL → WARN → PASS
        _order = {CompatLevel.FAIL: 0, CompatLevel.WARN: 1, CompatLevel.PASS: 2}
        results.sort(key=lambda r: _order[r.level])
        return results

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_framework(
        entry: "TrainingAssetEntry",
        spec:  "TrainingJobSpec",
    ) -> CompatResult:
        entry_fw  = (entry.framework or "").strip().lower()
        canvas_fw = "sb3"   # Phase D only supports SB3

        if not entry_fw:
            return CompatResult(
                CompatLevel.WARN,
                "framework",
                "Asset has no framework tag — assuming SB3 compatible.",
            )
        if entry_fw != canvas_fw:
            return CompatResult(
                CompatLevel.FAIL,
                "framework",
                f"Framework mismatch: asset='{entry_fw}'  canvas='{canvas_fw}'. "
                "Phase D training only supports Stable-Baselines3 (SB3) checkpoints.",
            )
        return CompatResult(CompatLevel.PASS, "framework", "SB3 ✔")

    @staticmethod
    def _check_algorithm(
        entry: "TrainingAssetEntry",
        spec:  "TrainingJobSpec",
    ) -> CompatResult:
        entry_algo  = (getattr(entry, "algorithm", "") or "").strip().upper()
        canvas_algo = (spec.algorithm_config.algorithm or "").strip().upper()

        if not entry_algo:
            return CompatResult(
                CompatLevel.WARN,
                "algorithm",
                f"Asset has no algorithm tag — canvas is configured for {canvas_algo}.",
            )
        if entry_algo != canvas_algo:
            return CompatResult(
                CompatLevel.FAIL,
                "algorithm",
                f"Algorithm mismatch: asset='{entry_algo}'  canvas='{canvas_algo}'. "
                "Resuming a SAC checkpoint into PPO (or vice-versa) is not supported.",
            )
        return CompatResult(CompatLevel.PASS, "algorithm", f"{canvas_algo} ✔")

    @staticmethod
    def _check_obs_dim(
        entry: "TrainingAssetEntry",
        spec:  "TrainingJobSpec",
    ) -> CompatResult:
        entry_obs_dim = int(getattr(entry, "obs_dim", 0) or 0)
        if entry_obs_dim == 0:
            return CompatResult(
                CompatLevel.WARN,
                "obs_dim",
                "Asset has no obs_dim recorded — cannot verify observation contract.",
            )

        # Phase 5-C: when a named contract preset is active, use its authoritative
        # obs_dim directly rather than estimating from the free-form component list.
        preset = getattr(spec.obs_action_config, "contract_preset", "custom") or "custom"
        canvas_obs_dim = 0
        preset_label = ""
        if preset != "custom":
            try:
                from src.system.training.obs_contracts import get_obs_contract
                contract = get_obs_contract(preset)
                if contract is not None:
                    canvas_obs_dim = int(contract["obs_dim"])
                    preset_label = f" (preset '{preset}')"
            except ImportError:
                pass

        if canvas_obs_dim == 0:
            # Fall back to component-based estimation. Use the same helper the
            # exporter uses (`resolve_obs_dim_from_components`) so canvas-side
            # and asset-side obs_dim can never disagree just because they were
            # computed by two slightly different lookup tables.
            frame_stack = max(1, int(getattr(spec.obs_action_config, "frame_stack", 1) or 1))
            try:
                from src.system.training.unitree_gym_env import (
                    resolve_obs_dim_from_components,
                )
                action_dim_for_calc = (
                    int(spec.robot_spec.action_dim)
                    if int(spec.robot_spec.action_dim or 0) > 0
                    else _ACTION_DIM_BY_ROBOT.get(
                        (spec.robot_spec.robot_type or "").lower(), 12
                    )
                )
                canvas_obs_dim = resolve_obs_dim_from_components(
                    list(spec.obs_action_config.obs_components or []),
                    num_joints=action_dim_for_calc,
                    action_dim=action_dim_for_calc,
                    frame_stack=frame_stack,
                )
            except ImportError:
                canvas_obs_dim = compute_canvas_obs_dim(
                    spec.obs_action_config.obs_components,
                    spec.robot_spec.robot_type,
                    action_dim=spec.robot_spec.action_dim,
                )
                canvas_obs_dim *= frame_stack

        if canvas_obs_dim == 0:
            return CompatResult(
                CompatLevel.WARN,
                "obs_dim",
                f"Canvas obs_dim could not be computed from obs_components "
                f"{spec.obs_action_config.obs_components!r} — "
                f"asset expects {entry_obs_dim}.",
            )
        if canvas_obs_dim != entry_obs_dim:
            return CompatResult(
                CompatLevel.FAIL,
                "obs_dim",
                f"Observation dimension mismatch: asset={entry_obs_dim}  "
                f"canvas={canvas_obs_dim}{preset_label}. "
                "The policy network input layer size must match exactly.",
            )
        return CompatResult(
            CompatLevel.PASS, "obs_dim", f"{canvas_obs_dim}{preset_label} ✔"
        )

    @staticmethod
    def _check_action_type(
        entry: "TrainingAssetEntry",
        spec:  "TrainingJobSpec",
    ) -> CompatResult:
        entry_atype  = (getattr(entry, "action_type", "") or "").strip().lower()
        canvas_atype = (spec.physics_config.action_type or "").strip().lower()

        if not entry_atype:
            return CompatResult(
                CompatLevel.PASS,
                "action_type",
                "Asset has no action_type recorded — skipping check.",
            )
        if entry_atype != canvas_atype:
            return CompatResult(
                CompatLevel.WARN,
                "action_type",
                f"Action type mismatch: asset='{entry_atype}'  "
                f"canvas='{canvas_atype}'. "
                "Policy will train but deployment behavior may differ.",
            )
        return CompatResult(
            CompatLevel.PASS, "action_type", f"{canvas_atype} ✔"
        )

    @staticmethod
    def _check_robot_type(
        entry: "TrainingAssetEntry",
        spec:  "TrainingJobSpec",
    ) -> CompatResult:
        entry_rt  = (getattr(entry, "robot_type", "") or "").strip().lower()
        canvas_rt = (spec.robot_spec.robot_type or "").strip().lower()

        if not entry_rt:
            return CompatResult(
                CompatLevel.PASS,
                "robot_type",
                "Asset has no robot_type recorded — skipping check.",
            )
        if entry_rt != canvas_rt:
            return CompatResult(
                CompatLevel.WARN,
                "robot_type",
                f"Robot type mismatch: asset='{entry_rt}'  "
                f"canvas='{canvas_rt}'. "
                "Cross-robot transfer is unusual; verify obs/action contracts manually.",
            )
        return CompatResult(
            CompatLevel.PASS, "robot_type", f"{canvas_rt} ✔"
        )
