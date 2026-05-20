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
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - \\
            asset.data.default_joint_pos[:, asset_cfg.joint_ids]
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
)
