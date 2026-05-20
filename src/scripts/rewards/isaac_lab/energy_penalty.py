"""Energy Penalty — L2 penalty on torque × velocity product — minimises mechanical energy expenditure."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_energy(env, asset_cfg=SceneEntityCfg("robot")):
    """Penalize energy used by joints (|vel| * |torque|)."""
    import torch
    asset = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)
'''


ENTRY = reward_item(
    key='energy_penalty',
    polarity='penalty',
    title='Energy Penalty',
    desc='L2 penalty on torque × velocity product — minimises mechanical energy expenditure.',
    default=-2e-05,
    min_value=-0.01,
    max_value=0.0,
    step=1e-05,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_energy',
    il_module=IL_MOD_INLINE,
    il_params='"asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
)
