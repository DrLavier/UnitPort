"""Joint Pose Tracking — Gaussian reward for joint positions matching the reference frame."""

from __future__ import annotations

from scripts.task_module import (
    ALG_AMP,
    ALG_PPO,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def joint_pose_tracking(cur_joints, ref_joints):
    """Gaussian reward for joint positions matching reference.

    score = exp(-5.0 * ||q_cur - q_ref||^2)
    Falls back to default standing pose when no reference is loaded.
    """
    sq_err = sum((cur_joints - ref_joints) ** 2)
    return exp(-5.0 * sq_err)
'''


ENTRY = reward_item(
    key='joint_pose_tracking',
    polarity='reward',
    title='Joint Pose Tracking',
    desc='Gaussian reward for joint positions matching the reference frame. Falls back to a default standing pose when no reference motion is loaded.',
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_inline=INLINE_SOURCE,
)
