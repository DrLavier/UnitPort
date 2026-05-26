# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Flat Orientation — L2 penalty on projected gravity deviation from vertical — keeps the base level."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on projected gravity deviation from vertical (keeps base level)."""
    import torch
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
'''


ENTRY = reward_item(
    key='flat_orientation',
    polarity='penalty',
    title='Flat Orientation',
    desc='L2 penalty on projected gravity deviation from vertical — keeps the base level.',
    default=-5.0,
    min_value=-20.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_flat_orientation_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
