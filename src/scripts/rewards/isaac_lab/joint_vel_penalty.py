# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint Vel Penalty — L2 penalty on joint velocities — discourages overly fast joint motion."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_joint_vel_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on joint velocities."""
    import torch
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
'''


ENTRY = reward_item(
    key='joint_vel_penalty',
    polarity='penalty',
    title='Joint Vel Penalty',
    desc='L2 penalty on joint velocities — discourages overly fast joint motion.',
    default=-0.001,
    min_value=-1.0,
    max_value=0.0,
    step=0.0005,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_joint_vel_l2',
    il_module=IL_MOD_INLINE,
    il_params='"asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
)
