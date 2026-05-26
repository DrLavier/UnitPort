# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lin Vel Z Penalty — L2 penalty on vertical linear velocity to discourage bouncing."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on vertical linear velocity."""
    import torch
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_b[:, 2]
    # Cap |vz| at 10 m/s before squaring — protects against physics spikes
    # where the body velocity jumps to hundreds of m/s in one step.
    vz = torch.nan_to_num(vz, nan=0.0, posinf=10.0, neginf=-10.0)
    vz = torch.clamp(vz, min=-10.0, max=10.0)
    return torch.square(vz)
'''


ENTRY = reward_item(
    key='lin_vel_z_penalty',
    polarity='penalty',
    title='Lin Vel Z Penalty',
    desc='L2 penalty on vertical linear velocity to discourage bouncing.',
    default=-2.0,
    min_value=-20.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_lin_vel_z_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
