"""Motion clip validator — cross-check vs optional RobotAsset.

Runs structural + numeric + (optional) robot-compatibility checks on a
``MotionClip`` **after** a loader has built it. Failures raise
``MotionValidationError`` with human-readable messages so the Canvas /
launcher can surface them directly.

Per AMP_design.yaml §3.reference_motion.new_modules.validator:

- No retargeting. Joint-count or joint-name mismatches against the
  target robot are rejected with a clear error, not silently fixed.
- Numeric sanity (no NaN/Inf, root quat unit norm within tolerance).
- Minimum frame count + strictly-positive frame_dt.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from src.system.training.motion.clip import MotionClip

if TYPE_CHECKING:
    from src.system.training.robot_assets.registry import RobotAsset


class MotionValidationError(ValueError):
    """Raised when a MotionClip fails validation. User-facing message."""


# Minimum frames a clip must have to be useful for anything — both
# tracking (needs at least 2 frames to cycle) and AMP (needs (s, s') pair).
_MIN_FRAMES = 2

# Tolerance for quaternion unit-norm check. Loaders normalize quats to
# 1e-12, so this is very conservative — anything worse means the file
# was corrupted.
_QUAT_NORM_TOL = 1e-3


def validate_clip(
    clip: MotionClip,
    robot_asset: Optional["RobotAsset"] = None,
) -> None:
    """Raise ``MotionValidationError`` on any problem; otherwise return.

    Parameters
    ----------
    clip:
        Freshly loaded ``MotionClip``.
    robot_asset:
        Optional. When provided, the clip's ``dof`` must equal
        ``robot_asset.dof_count``. Joint-name cross-check is attempted
        if the clip carries ``metadata['joint_names']`` but is NOT
        mandatory — most community mocap files don't embed joint names.
    """
    # ── Structural ──
    if clip.n_frames < _MIN_FRAMES:
        raise MotionValidationError(
            f"MotionClip '{clip.name}' has only {clip.n_frames} frame(s); "
            f"at least {_MIN_FRAMES} are required."
        )
    if clip.fps <= 0:
        raise MotionValidationError(
            f"MotionClip '{clip.name}' has invalid fps={clip.fps} "
            f"(must be > 0)."
        )
    if clip.dof <= 0:
        raise MotionValidationError(
            f"MotionClip '{clip.name}' has dof={clip.dof}; the joint_pos "
            f"array is empty or not 2-D."
        )
    if clip.joint_pos.shape[0] != clip.n_frames:
        raise MotionValidationError(
            f"MotionClip '{clip.name}': joint_pos first dim "
            f"({clip.joint_pos.shape[0]}) does not match n_frames "
            f"({clip.n_frames})."
        )

    # ── Numeric sanity ──
    _check_finite(clip.joint_pos, "joint_pos", clip.name)
    for attr in ("root_pos", "root_quat", "joint_vel", "toe_pos_local",
                 "toe_vel_local", "lin_vel", "ang_vel"):
        arr = getattr(clip, attr)
        if arr is not None:
            _check_finite(arr, attr, clip.name)

    # ── Quaternion unit norm ──
    if clip.root_quat is not None:
        norms = np.linalg.norm(clip.root_quat, axis=-1)
        max_err = float(np.max(np.abs(norms - 1.0)))
        if max_err > _QUAT_NORM_TOL:
            raise MotionValidationError(
                f"MotionClip '{clip.name}': root_quat has non-unit norms "
                f"(max deviation {max_err:.2e}, tol {_QUAT_NORM_TOL:.0e}). "
                f"The loader should have normalized these — this likely "
                f"means the source file is corrupted."
            )

    # ── AMP payload shape consistency ──
    if clip.has_amp_payload():
        if clip.joint_vel is None or clip.joint_vel.shape != clip.joint_pos.shape:
            raise MotionValidationError(
                f"MotionClip '{clip.name}': joint_vel shape does not match "
                f"joint_pos (needed for AMP payload)."
            )
        if clip.amp_obs_dim <= 0:
            raise MotionValidationError(
                f"MotionClip '{clip.name}': claims AMP payload but "
                f"amp_obs_dim={clip.amp_obs_dim}."
            )

    # ── Robot asset compatibility ──
    if robot_asset is not None:
        _check_vs_robot(clip, robot_asset)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_finite(arr: np.ndarray, field: str, clip_name: str) -> None:
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise MotionValidationError(
            f"MotionClip '{clip_name}': field '{field}' contains {bad} "
            f"non-finite values (NaN/Inf)."
        )


def _check_vs_robot(clip: MotionClip, robot_asset: "RobotAsset") -> None:
    """Dof / joint-name cross-check against a registered RobotAsset.

    We do NOT retarget. A mismatch is an error — the user must either
    pick a different motion file or provide a robot with matching dof.
    """
    expected_dof = int(robot_asset.dof_count)
    if expected_dof > 0 and clip.dof != expected_dof:
        raise MotionValidationError(
            f"MotionClip '{clip.name}' has {clip.dof} dof but RobotAsset "
            f"'{robot_asset.asset_id}' has {expected_dof}. "
            f"UnitPort does not retarget — pick a motion file matching "
            f"your robot's joint count, or use a robot with matching dof."
        )

    # Optional joint-name check — only if the clip carries names metadata.
    clip_names = clip.metadata.get("joint_names")
    if clip_names and robot_asset.joint_names:
        if list(clip_names) != list(robot_asset.joint_names):
            # Build a compact diff for the error message.
            diff_lines = []
            for i, (a, b) in enumerate(zip(clip_names, robot_asset.joint_names)):
                if a != b:
                    diff_lines.append(f"  [{i}] clip={a!r}  asset={b!r}")
                if len(diff_lines) >= 5:
                    diff_lines.append("  ...")
                    break
            raise MotionValidationError(
                f"MotionClip '{clip.name}' joint_names differ from RobotAsset "
                f"'{robot_asset.asset_id}':\n" + "\n".join(diff_lines)
            )
