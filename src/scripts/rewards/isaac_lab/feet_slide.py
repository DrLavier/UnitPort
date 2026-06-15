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
    """Penalize feet sliding on the ground while in contact.

    "Slide" is GROUND SLIP: the WORLD-frame horizontal speed of a foot that
    is currently loaded. A correctly planted foot is the body's pivot -- its
    world velocity is ~0 -- so it is NOT penalized even while the base
    translates over it; only a foot that actually skates along the ground
    (non-zero world velocity while in contact) is penalized.

    NOTE: an earlier version measured foot velocity RELATIVE TO THE ROOT
    (``body_lin_vel_w - root_lin_vel_w``). That is wrong for this term: a
    planted foot necessarily moves backward relative to a forward-moving
    body, so the relative-velocity form charged every normal stance foot at
    the base speed -- a hidden locomotion penalty mislabeled as "slide".
    The reward now matches its name (and IsaacLab stock ``feet_slide``):
    world-frame foot velocity, gated by contact.
    """
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(
        dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(foot_vel_xy.norm(dim=-1) * contacts, dim=1)
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
