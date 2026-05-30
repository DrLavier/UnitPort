# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Post-load augmentations applied uniformly across loaders.

Currently holds the left/right mirror op for quadruped AMP clips. The
op is loader-agnostic — it consumes a :class:`MotionClip` and emits a
mirrored copy — so it lives outside any specific format's module.

``_standardize_quats`` is imported privately from
:mod:`amp_legged_gym`: quaternion hemisphere standardisation is the
same operation everywhere; keeping a single implementation prevents
drift between the loader's quat handling and the mirror op's.
"""
from __future__ import annotations

from application.training.motion.clip import MotionClip
from application.training.motion.loaders.amp_legged_gym import _standardize_quats
from application.training.motion.loaders.base import LoaderError


# Left/right mirror augmentation tables for the 12-DoF quadruped layout
# (FL/FR/RL/RR × hip/thigh/calf, post-reorder).
_QUAD_LR_SWAP_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
_QUAD_HIP_INDICES = (0, 3, 6, 9)
_QUAD_FOOT_Y_INDICES = (1, 4, 7, 10)


def mirror_motion_clip(clip: MotionClip) -> MotionClip:
    """Reflect a quadruped MotionClip across the sagittal plane.

    Returns a new clip suffixed ``"_mirror"`` — input is not mutated.
    Only valid for 12-DoF FL/FR/RL/RR clips (typically produced by
    :class:`AMPLegGymLoader`).
    """
    if clip.joint_pos.shape[1] != 12:
        raise LoaderError(
            f"mirror_motion_clip: only 12-dof quadruped clips are supported, "
            f"got dof={clip.joint_pos.shape[1]} for {clip.name!r}."
        )

    swap = _QUAD_LR_SWAP_PERM
    hip_idx = list(_QUAD_HIP_INDICES)
    foot_y_idx = list(_QUAD_FOOT_Y_INDICES)

    new_joint_pos = clip.joint_pos[:, swap].copy()
    new_joint_pos[:, hip_idx] *= -1.0

    if clip.joint_vel is not None:
        new_joint_vel = clip.joint_vel[:, swap].copy()
        new_joint_vel[:, hip_idx] *= -1.0
    else:
        new_joint_vel = None

    if clip.toe_pos_local is not None and clip.toe_pos_local.shape[1] == 12:
        new_toe_pos = clip.toe_pos_local[:, swap].copy()
        new_toe_pos[:, foot_y_idx] *= -1.0
    else:
        new_toe_pos = None

    if clip.toe_vel_local is not None and clip.toe_vel_local.shape[1] == 12:
        new_toe_vel = clip.toe_vel_local[:, swap].copy()
        new_toe_vel[:, foot_y_idx] *= -1.0
    else:
        new_toe_vel = None

    if clip.root_pos is not None and clip.root_pos.shape[1] == 3:
        new_root_pos = clip.root_pos.copy()
        new_root_pos[:, 1] *= -1.0
    else:
        new_root_pos = None

    if clip.root_quat is not None and clip.root_quat.shape[1] == 4:
        new_root_quat = clip.root_quat.copy()
        new_root_quat[:, 0] *= -1.0
        new_root_quat[:, 2] *= -1.0
        new_root_quat = _standardize_quats(new_root_quat)
    else:
        new_root_quat = None

    if clip.lin_vel is not None and clip.lin_vel.shape[1] == 3:
        new_lin_vel = clip.lin_vel.copy()
        new_lin_vel[:, 1] *= -1.0
    else:
        new_lin_vel = None

    if clip.ang_vel is not None and clip.ang_vel.shape[1] == 3:
        new_ang_vel = clip.ang_vel.copy()
        new_ang_vel[:, 0] *= -1.0
        new_ang_vel[:, 2] *= -1.0
    else:
        new_ang_vel = None

    new_metadata = dict(clip.metadata)
    new_metadata["mirrored_from"] = clip.name

    return MotionClip(
        name=clip.name + "_mirror",
        format_id=clip.format_id,
        fps=clip.fps,
        loop_mode=clip.loop_mode,
        motion_weight=clip.motion_weight,
        task_tag=clip.task_tag,
        root_pos=new_root_pos,
        root_quat=new_root_quat,
        joint_pos=new_joint_pos,
        joint_vel=new_joint_vel,
        toe_pos_local=new_toe_pos,
        toe_vel_local=new_toe_vel,
        lin_vel=new_lin_vel,
        ang_vel=new_ang_vel,
        amp_obs_dim=clip.amp_obs_dim,
        metadata=new_metadata,
    )


__all__ = ["mirror_motion_clip"]
