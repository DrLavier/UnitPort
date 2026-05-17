"""Motion clip validator — structural + RobotSpec / IR cross-check.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/motion/validator.py``.
Stage 5 changes the cross-check target: DEMO compared against
``RobotAsset`` (an asset-registry object); RELEASE compares against
:class:`registers.robots.RobotSpec` + :func:`registers.ir.canonical_body_map`.

The contract stays the same — **no retargeting**. A clip whose joint
count or required IR roles do not match the robot is rejected with an
``unmapped_bodies`` error so the canvas can surface it. Robots map ONTO
the canonical IR; the IR set never auto-extends from a clip's body
names (CLAUDE.md §1.2 red line).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from application.training.motion.clip import MotionClip

if TYPE_CHECKING:
    from registers.robots import RobotSpec


class MotionValidationError(ValueError):
    """Raised when a MotionClip fails validation. User-facing message."""


# Minimum frames a clip must have to be useful — both tracking (needs at
# least 2 frames to cycle) and AMP (needs (s, s') pair).
_MIN_FRAMES = 2

# Tolerance for quaternion unit-norm check. Loaders normalize quats to
# 1e-12, so this is conservative — anything worse means corruption.
_QUAT_NORM_TOL = 1e-3


# ---------------------------------------------------------------------------
# Per-format expected IR roles (Stage 5; extended in Stage 8)
# ---------------------------------------------------------------------------

# 12-DoF quadruped, canonical order after AMPLegGymLoader leg-reorder
# (FR/FL/RR/RL → FL/FR/RL/RR). Contract pinned with
# ``loader._reorder_quad_legs``.
_QUADRUPED_AMP_IR_ROLES: List[str] = [
    "hip_FL", "thigh_FL", "calf_FL",
    "hip_FR", "thigh_FR", "calf_FR",
    "hip_RL", "thigh_RL", "calf_RL",
    "hip_RR", "thigh_RR", "calf_RR",
]

_FORMAT_IR_ROLES = {
    "amp_legged_gym": _QUADRUPED_AMP_IR_ROLES,
    # unitport_npy: no portable IR contract (clip carries env-native
    # joint order). Validator skips the IR gate for it.
}


@dataclass(frozen=True)
class IRMatchReport:
    """Outcome of comparing a clip against a robot's IR roles."""

    ok: bool
    clip_name: str = ""
    format_id: str = ""
    expected_roles: List[str] = field(default_factory=list)
    unmapped_bodies: List[str] = field(default_factory=list)  # red-line term
    message: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_clip(clip: MotionClip) -> None:
    """Structural + numeric sanity checks. Raises
    :exc:`MotionValidationError` on failure; otherwise returns ``None``.

    Use :func:`validate_clip_against_robot` to additionally check the
    clip is compatible with a target :class:`RobotSpec`.
    """
    if clip.n_frames < _MIN_FRAMES:
        raise MotionValidationError(
            f"MotionClip {clip.name!r} has only {clip.n_frames} frame(s); "
            f"at least {_MIN_FRAMES} are required."
        )
    if clip.fps <= 0:
        raise MotionValidationError(
            f"MotionClip {clip.name!r} has invalid fps={clip.fps} (must be > 0)."
        )
    if clip.dof <= 0:
        raise MotionValidationError(
            f"MotionClip {clip.name!r} has dof={clip.dof}; the joint_pos "
            f"array is empty or not 2-D."
        )
    if clip.joint_pos.shape[0] != clip.n_frames:
        raise MotionValidationError(
            f"MotionClip {clip.name!r}: joint_pos first dim "
            f"({clip.joint_pos.shape[0]}) != n_frames ({clip.n_frames})."
        )

    _check_finite(clip.joint_pos, "joint_pos", clip.name)
    for attr in ("root_pos", "root_quat", "joint_vel", "toe_pos_local",
                 "toe_vel_local", "lin_vel", "ang_vel"):
        arr = getattr(clip, attr)
        if arr is not None:
            _check_finite(arr, attr, clip.name)

    if clip.root_quat is not None:
        norms = np.linalg.norm(clip.root_quat, axis=-1)
        max_err = float(np.max(np.abs(norms - 1.0)))
        if max_err > _QUAT_NORM_TOL:
            raise MotionValidationError(
                f"MotionClip {clip.name!r}: root_quat has non-unit norms "
                f"(max deviation {max_err:.2e}, tol {_QUAT_NORM_TOL:.0e}). "
                f"Source file likely corrupted."
            )

    if clip.has_amp_payload():
        if clip.joint_vel is None or clip.joint_vel.shape != clip.joint_pos.shape:
            raise MotionValidationError(
                f"MotionClip {clip.name!r}: joint_vel shape does not match "
                f"joint_pos (needed for AMP payload)."
            )
        if clip.amp_obs_dim <= 0:
            raise MotionValidationError(
                f"MotionClip {clip.name!r}: claims AMP payload but "
                f"amp_obs_dim={clip.amp_obs_dim}."
            )


def validate_clip_against_robot(
    clip: MotionClip,
    robot_spec: "RobotSpec",
) -> IRMatchReport:
    """Compare *clip* against *robot_spec*; return a structured report.

    Behaviour:
      * Run :func:`validate_clip` first (raises on structural failure).
      * Compare clip ``dof`` to ``robot_spec.num_joints``; mismatch → error
        report (no retargeting).
      * For formats with an IR contract (``amp_legged_gym``), require every
        expected role to appear in ``robot_spec.joint_ir_roles``. Missing
        roles are surfaced as ``unmapped_bodies`` so the canvas can
        explain the gap (CLAUDE.md §1.2 red line: never auto-extend the
        IR catalog).

    Returns:
        :class:`IRMatchReport` with ``ok=True`` on success. ``ok=False``
        with populated ``unmapped_bodies`` is the canonical "the user
        must resolve this" outcome.
    """
    validate_clip(clip)

    expected = list(_FORMAT_IR_ROLES.get(clip.format_id, []))

    if expected and clip.dof != len(expected):
        return IRMatchReport(
            ok=False,
            clip_name=clip.name,
            format_id=clip.format_id,
            expected_roles=expected,
            unmapped_bodies=list(expected),
            message=(
                f"clip {clip.name!r} has dof={clip.dof}; format "
                f"{clip.format_id!r} requires {len(expected)} (got {clip.dof})"
            ),
        )

    # Joint count vs robot
    if robot_spec.num_joints > 0 and clip.dof != robot_spec.num_joints:
        return IRMatchReport(
            ok=False,
            clip_name=clip.name,
            format_id=clip.format_id,
            expected_roles=expected,
            unmapped_bodies=list(expected),
            message=(
                f"clip {clip.name!r} has dof={clip.dof} but RobotSpec "
                f"{robot_spec.sku!r} ({robot_spec.name!r}) has "
                f"{robot_spec.num_joints} joints. UnitPort does not retarget — "
                f"pick a clip with matching dof or a robot with matching dof."
            ),
        )

    if not expected:
        # Format has no IR contract (e.g. unitport_npy) — skip the role gate.
        return IRMatchReport(
            ok=True,
            clip_name=clip.name,
            format_id=clip.format_id,
            expected_roles=[],
            message=f"format {clip.format_id!r} has no portable IR contract — skipped",
        )

    # IR-role gate: every expected role must appear in the robot's joint roles.
    have = set(robot_spec.joint_ir_roles)
    unmapped = [r for r in expected if r not in have]
    if unmapped:
        return IRMatchReport(
            ok=False,
            clip_name=clip.name,
            format_id=clip.format_id,
            expected_roles=expected,
            unmapped_bodies=unmapped,
            message=(
                f"{len(unmapped)} IR role(s) unmapped for robot "
                f"{robot_spec.sku!r}: {unmapped!r}. Edit the robot's body "
                f"mapping or pick a robot whose joints cover the format's "
                f"required roles. (Auto-extending the IR catalog is forbidden — "
                f"see CLAUDE.md §1.2.)"
            ),
        )

    return IRMatchReport(
        ok=True,
        clip_name=clip.name,
        format_id=clip.format_id,
        expected_roles=expected,
        message="all IR roles resolve",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_finite(arr: np.ndarray, field_name: str, clip_name: str) -> None:
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise MotionValidationError(
            f"MotionClip {clip_name!r}: field {field_name!r} contains "
            f"{bad} non-finite values (NaN/Inf)."
        )


__all__ = [
    "MotionValidationError",
    "IRMatchReport",
    "validate_clip",
    "validate_clip_against_robot",
]
