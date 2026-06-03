# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint Pos Limits — Penalty when joint positions approach or exceed soft limits — keeps joints within safe operating range."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_joint_pos_limits(env, asset_cfg=SceneEntityCfg("robot")):
    """Penalty when joint positions approach or exceed soft limits.

    For each joint, computes the excess beyond 95% of the joint range
    on both sides and sums the absolute violations.
    """
    import torch
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_lo = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    soft_hi = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    out_of_range = -(pos - soft_lo).clamp(max=0.0) + (pos - soft_hi).clamp(min=0.0)
    return torch.sum(out_of_range, dim=1)
'''


ENTRY = reward_item(
    key='dof_pos_limits',
    polarity='penalty',
    title='Joint Pos Limits',
    desc='Penalty when joint positions approach or exceed soft limits — keeps joints within safe operating range.',
    default=-5.0,
    min_value=-20.0,
    max_value=0.0,
    step=0.5,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_joint_pos_limits',
    il_module=IL_MOD_INLINE,
    il_params='"asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
    il_partition_source='pd_groups',  # 缺口③ — joint-subset paginable
)
