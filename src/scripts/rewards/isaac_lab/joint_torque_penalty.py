"""Joint Torque Penalty — L2 penalty on joint torques for energy efficiency."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_joint_torques_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on applied joint torques."""
    import torch
    asset = env.scene[asset_cfg.name]
    tq = asset.data.applied_torque[:, asset_cfg.joint_ids]
    tq = torch.nan_to_num(tq, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    tq = torch.clamp(tq, min=-1.0e3, max=1.0e3)
    return torch.sum(torch.square(tq), dim=1)
'''


ENTRY = reward_item(
    key='joint_torque_penalty',
    polarity='penalty',
    title='Joint Torque Penalty',
    desc='L2 penalty on joint torques for energy efficiency.',
    default=-0.0002,
    min_value=-0.01,
    max_value=0.0,
    step=5e-05,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_joint_torques_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
