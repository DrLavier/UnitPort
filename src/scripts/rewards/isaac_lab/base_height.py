# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
def _unitport_base_height_l2(env, target_height=None, asset_cfg=SceneEntityCfg("robot"),
                              sensor_cfg=None):
    """Penalize base height deviation from a brand-neutral standing target.

    target_height resolution (first non-None wins):
      1. The ``target_height`` kwarg passed by the canvas (the Rewards node's
         per-item "Value" chip). Caller-controlled override.
      2. ``asset.data.default_root_state[:, 2]`` — the spawn pose's
         z-coordinate, which is the asset's nominal standing height
         declared in its USD/MJCF init_state. Brand-neutral, no per-robot
         tuning needed; works for Go2 (~0.4 m), A1 (~0.32 m), Spot
         (~0.5 m), G1/H1 (~0.7 m).
      3. Final fallback 0.34 m — generic legged-quadruped value used
         only if neither of the above is available (test envs without
         spawn pose).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    # 0.0 / negative is treated as "use auto" sentinel because the canvas
    # compiler emits 0.0 when the reward's "Value" chip is unset.
    if target_height is None or target_height <= 0.0:
        try:
            spawn_z = asset.data.default_root_state[:, 2]
            target_height = float(spawn_z[0].item())
        except Exception:
            target_height = 0.34
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
    # ``{item_value}`` = the per-item "Value" chip (target standing height, m).
    # 0.0 = auto → inline reward resolves to the asset's nominal spawn z.
    il_params='"target_height": {item_value}, "asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
    il_value_label='Target Height',
    il_value_default=0.0,
    il_value_min=0.0,
    il_value_max=5.0,
    il_value_step=0.01,
    il_value_unit='m',
)
