# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Feet Slide — L2 penalty on foot velocity while in contact — discourages sliding/skating."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_feet_slide(env, sensor_cfg=None, asset_cfg=SceneEntityCfg("robot")):
    """Penalize feet sliding on the ground while in contact."""
    import torch
    import isaaclab.utils.math as math_utils
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(
        dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    cur_footvel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_body = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_body[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel[:, i, :])
    lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_body[:, :, :2]), dim=2)).view(env.num_envs, -1)
    return torch.sum(lateral_vel * contacts, dim=1)
'''


ENTRY = reward_item(
    key='feet_slide',
    polarity='penalty',
    title='Feet Slide',
    desc='L2 penalty on foot velocity while in contact — discourages sliding/skating.',
    default=-0.2,
    min_value=-5.0,
    max_value=0.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_feet_slide',
    il_module=IL_MOD_INLINE,
    il_params='"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), "asset_cfg": SceneEntityCfg("robot", body_names={ir:feet})',
    il_inline=INLINE_SOURCE,
)
