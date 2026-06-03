# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint Deviation L1 — L1 penalty on joint positions deviating from the asset's default pose."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_joint_deviation_l1(env, asset_cfg=SceneEntityCfg("robot")):
    """L1 penalty on joint position deviation from the asset's default pose.

    Anchors the policy to the nominal stance encoded in the articulation's
    ``default_joint_pos`` — counters left/right asymmetric drift and
    body-sinking that arises when the policy has no posture reference.
    """
    import torch
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] -             asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)
'''


ENTRY = reward_item(
    key='joint_deviation_l1',
    polarity='penalty',
    title='Joint Deviation L1',
    desc="L1 penalty on joint positions deviating from the asset's default pose. Anchors the policy to a symmetric nominal stance — counters left/right asymmetric drift and posture sinking.",
    default=-0.05,
    min_value=-2.0,
    max_value=0.0,
    step=0.01,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_joint_deviation_l1',
    il_module=IL_MOD_INLINE,
    il_params='"asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
    # 缺口③ — partitionable by the family's PD joint groups. With no partitions
    # the term runs on all joints (default above); with a ``partitions`` map on
    # the Rewards node it fans out into one instance per subset (hip / arm /
    # waist / ...), each on that subset's joints with its own weight —
    # legged_gym's hip_dof_deviation / arm_dof_deviation / waist_dof_deviation.
    il_partition_source='pd_groups',
)
