# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Feet Air Time — Reward for maintaining contact schedule (gait pattern)."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_feet_air_time(env, threshold=0.5, sensor_cfg=None):
    """Reward long feet air time at touchdown -- PURE stride-duration shaping.

    Bonus proportional to how long each foot was airborne when it touches
    down (``last_air_time - threshold`` summed over feet at first contact).
    This rewards deliberate, long swings over rapid high-frequency shuffling
    -- it shapes STRIDE DURATION and nothing else, to match its name.

    By design it makes NO check on base velocity, displacement, or command
    magnitude: whether the robot is following the commanded velocity vector
    is the velocity-tracking rewards' job, kept as a separate concern so the
    terms compose freely. A robot that never lifts a foot scores zero here
    naturally (no touchdown fires first_contact), so a still stance needs no
    gate. Route this reward only to items where stepping is wanted (per-item
    wiring); the stand item, with no air-time edge, is left untouched. Pair
    it with the gait reward on the same item for the all-legs coordination
    guarantee -- air time shapes duration, gait shapes phasing, and the two
    stack without either policing velocity.
    """
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward
'''


ENTRY = reward_item(
    key='feet_air_time',
    polarity='reward',
    title='Feet Air Time',
    desc='Reward for maintaining contact schedule (gait pattern).',
    default=0.125,
    min_value=0.0,
    max_value=5.0,
    step=0.025,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_feet_air_time',
    il_module=IL_MOD_INLINE,
    il_params='"threshold": {node_threshold}, "sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
    il_inline=INLINE_SOURCE,
)
