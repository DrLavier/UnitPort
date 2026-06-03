# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint Accel Penalty — L2 penalty on joint accelerations for smooth motion."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_joint_acc_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on joint accelerations."""
    import torch
    asset = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    # Guard against physics spikes / NaN: cap per-joint |accel| at 1000 rad/s^2
    # before squaring so a single bad step can't blow up the value function.
    acc = torch.nan_to_num(acc, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    acc = torch.clamp(acc, min=-1.0e3, max=1.0e3)
    return torch.sum(torch.square(acc), dim=1)
'''


ENTRY = reward_item(
    key='joint_accel_penalty',
    polarity='penalty',
    title='Joint Accel Penalty',
    desc='L2 penalty on joint accelerations for smooth motion.',
    default=-2.5e-07,
    min_value=-0.001,
    max_value=0.0,
    step=1e-07,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_joint_acc_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
    il_partition_source='pd_groups',  # 缺口③ — joint-subset paginable
)
