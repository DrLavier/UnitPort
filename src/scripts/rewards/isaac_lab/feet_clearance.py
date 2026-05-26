# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Feet Clearance — Reward for swing-foot height reaching a target clearance above ground."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_foot_clearance_reward(env, asset_cfg=SceneEntityCfg("robot"),
                                     target_height=0.1, std=0.05, tanh_mult=2.0):
    """Reward swinging feet for clearing a specified height."""
    import torch
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)
'''


ENTRY = reward_item(
    key='feet_clearance',
    polarity='reward',
    title='Feet Clearance',
    desc='Reward for swing-foot height reaching a target clearance above ground. Needs params: target_height (m, default 0.1).',
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_foot_clearance_reward',
    il_module=IL_MOD_INLINE,
    il_params='"asset_cfg": SceneEntityCfg("robot", body_names={ir:feet}), "target_height": 0.1, "std": 0.05, "tanh_mult": 2.0',
    il_inline=INLINE_SOURCE,
)
