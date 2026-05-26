# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Track Ang Vel Z — Exponential tracking reward for commanded yaw angular velocity."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_track_ang_vel_z_exp(env, std=0.25, command_name="base_velocity",
                                   asset_cfg=SceneEntityCfg("robot")):
    """Exponential tracking reward for commanded yaw angular velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(ang_vel_z - cmd_yaw)
    return torch.exp(-error / (std * std))
'''


ENTRY = reward_item(
    key='track_ang_vel_z',
    polarity='reward',
    title='Track Ang Vel Z',
    desc='Exponential tracking reward for commanded yaw angular velocity.',
    default=0.75,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_track_ang_vel_z_exp',
    il_module=IL_MOD_INLINE,
    il_params='"std": {node_std}, "command_name": "base_velocity"',
    il_inline=INLINE_SOURCE,
)
