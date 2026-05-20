"""Ang Vel Z Penalty — L2 penalty on yaw angular velocity — unconditional."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_ang_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on yaw angular velocity — unconditional.

    Mirrors ang_vel_xy_penalty for the Z axis. Use for items that
    should not spin (stand / strafe / pace).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b[:, 2]
    w = torch.nan_to_num(w, nan=0.0, posinf=20.0, neginf=-20.0)
    w = torch.clamp(w, min=-20.0, max=20.0)
    return torch.square(w)
'''


ENTRY = reward_item(
    key='ang_vel_z_penalty',
    polarity='penalty',
    title='Ang Vel Z Penalty',
    desc='L2 penalty on yaw angular velocity — unconditional. Use for items that should not spin (stand / strafe / pace).',
    default=-0.1,
    min_value=-10.0,
    max_value=0.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_ang_vel_z_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
