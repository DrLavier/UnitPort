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
def _unitport_feet_air_time(env, threshold=0.5, command_name="base_velocity",
                             sensor_cfg=None, asset_cfg=SceneEntityCfg("robot")):
    """Reward feet air time when the robot is moving (velocity command above threshold).

    Mirrors velocity_mdp.feet_air_time: gives a bonus proportional to
    how long each foot was in the air at touchdown, gated by whether the
    velocity command magnitude exceeds a minimum.
    """
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    cmd = env.command_manager.get_command(command_name)
    is_moving = torch.norm(cmd[:, :2], dim=1) > 0.1
    return reward * is_moving
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
    il_params='"threshold": {node_threshold}, "command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
    il_inline=INLINE_SOURCE,
)
