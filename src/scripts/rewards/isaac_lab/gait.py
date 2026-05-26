# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Gait Rhythm — Periodic gait reward — encourages regular alternating footfalls with a target period and phase offset."""

from __future__ import annotations

from scripts.task_module import (
    ALG_AMP,
    ALG_PPO,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_feet_gait(env, period=0.8, offset=None, sensor_cfg=None,
                         threshold=0.5, command_name=None):
    """Enforce periodic gait patterns for legged robots.

    Per-leg contribution is +1 when the leg's contact state matches the
    expected stance/swing phase, and -1 when it mismatches. This makes
    "freeze all 4 legs while a velocity command is active" actively
    penalised (= -N_legs) rather than merely unrewarded (= 0), which
    was the prior behaviour and let policies skip gait collection to
    cheat e.g. track_ang_vel_z by twisting joints in place.

    The command_norm gate keeps gait inactive on zero-command stances
    (stand motion), so this stricter shaping does not collide with the
    intent to stay still when commanded.
    """
    import torch
    if offset is None:
        offset = [0.0, 0.5, 0.5, 0.0]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for off in offset:
        phases.append((global_phase + off) % 1.0)
    leg_phase = torch.cat(phases, dim=-1)
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        match = ~(is_stance ^ is_contact[:, i])
        reward += match.float() * 2.0 - 1.0
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= (cmd_norm > 0.1).float()
    return reward
'''


ENTRY = reward_item(
    key='gait',
    polarity='reward',
    title='Gait Rhythm',
    desc='Periodic gait reward — encourages regular alternating footfalls with a target period and phase offset. Needs params: period (s), offset (per-foot phase array).',
    default=0.5,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_func='_unitport_feet_gait',
    il_module=IL_MOD_INLINE,
    il_params='"period": 0.8, "offset": [0.0, 0.5, 0.5, 0.0], "sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), "threshold": 0.5, "command_name": "base_velocity"',
    il_inline=INLINE_SOURCE,
)
