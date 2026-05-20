"""Base Height — L2 penalty on base height deviating from a target standing height."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_base_height_l2(env, target_height=0.34, asset_cfg=SceneEntityCfg("robot"),
                              sensor_cfg=None):
    """Penalize base height deviation from target using L2 squared kernel."""
    import torch
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any():
            adjusted = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted = target_height
    err = asset.data.root_pos_w[:, 2] - adjusted
    # Cap deviation at 1 m before squaring so a physics-clipping event
    # cannot push the per-step penalty past ~1.0.
    err = torch.nan_to_num(err, nan=0.0, posinf=1.0, neginf=-1.0)
    err = torch.clamp(err, min=-1.0, max=1.0)
    return torch.square(err)
'''


ENTRY = reward_item(
    key='base_height',
    polarity='penalty',
    title='Base Height',
    desc='L2 penalty on base height deviating from a target standing height. Needs params: target_height (default 0.34 for quadruped, 0.78 for biped).',
    default=-10.0,
    min_value=-50.0,
    max_value=0.0,
    step=0.5,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_base_height_l2',
    il_module=IL_MOD_INLINE,
    il_params='"target_height": {robot_target_height}, "asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
)
