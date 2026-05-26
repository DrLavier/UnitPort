# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Track Lin Vel XY — Exponential tracking reward for commanded XY linear velocity."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_track_lin_vel_xy_exp(env, std=0.25, command_name="base_velocity",
                                    asset_cfg=SceneEntityCfg("robot")):
    """Exponential tracking reward for commanded XY linear velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum(torch.square(vel_xy - cmd_xy), dim=1)
    return torch.exp(-error / (std * std))
'''


ENTRY = reward_item(
    key='track_lin_vel_xy',
    polarity='reward',
    title='Track Lin Vel XY',
    desc='Exponential tracking reward for commanded XY linear velocity.',
    default=1.5,
    min_value=0.0,
    max_value=40.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_track_lin_vel_xy_exp',
    il_module=IL_MOD_INLINE,
    il_params='"std": {node_std}, "command_name": "base_velocity"',
    il_inline=INLINE_SOURCE,
)
