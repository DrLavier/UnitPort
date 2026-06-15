# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Feet Slide — Contact-force-based penalty on foot lateral velocity while in ground contact."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def feet_slide(foot_body_ids, contacts, foot_vel_w):
    """Contact-gated penalty on foot WORLD-frame lateral velocity while grounded.

    contact_indicator * ||foot_vel_xy_world||^2
    "Slide" is ground slip: a planted foot is the body's pivot (world velocity
    ~0) and must NOT be penalized while the base moves over it. Use the foot's
    world-frame velocity, NOT its velocity relative to the root -- the latter
    charges every normal stance foot at the base speed (a locomotion penalty,
    not a slide penalty). Mirrors IsaacLab ``_unitport_feet_slide``.
    Contact detection uses MuJoCo contact pairs (binary, not force magnitude).
    """
    cost = 0.0
    for bid in contacted_foot_bodies:
        v = foot_vel_w[bid][:2]
        cost += v[0]**2 + v[1]**2
    return cost
'''


ENTRY = reward_item(
    key='feet_slide',
    polarity='penalty',
    title='Feet Slide',
    desc='Contact-force-based penalty on foot lateral velocity while in ground contact. Uses contact detection to weight the sliding cost.',
    default=-0.1,
    min_value=-5.0,
    max_value=0.0,
    step=0.01,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
