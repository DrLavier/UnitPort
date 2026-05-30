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
    "freeze all N legs while a velocity command is active" actively
    penalised (= -N_legs) rather than merely unrewarded (= 0), which
    was the prior behaviour and let policies skip gait collection to
    cheat e.g. track_ang_vel_z by twisting joints in place.

    The command_norm gate keeps gait inactive on zero-command stances
    (stand motion), so this stricter shaping does not collide with the
    intent to stay still when commanded.

    The default ``offset`` is resolved from ``len(sensor_cfg.body_ids)``
    at runtime (4-foot -> trot [0, 0.5, 0.5, 0]; 2-foot -> alternating
    [0, 0.5]). This is dispatch on the runtime-discovered foot count
    from the sensor; the reward source carries no family-keyed literal.
    """
    import torch
    # period<=0 is the "auto" sentinel the canvas compiler emits when the
    # Rewards node "Value" chip is unset (parse_item_value -> None -> 0.0).
    # Fall back to the canonical default so the gait period stays
    # single-sourced here (the il_params template carries {item_value}, no
    # second 0.8 literal).
    if period is None or period <= 0.0:
        period = 0.8
    if offset is None:
        n_feet = len(sensor_cfg.body_ids)
        if n_feet == 4:
            offset = [0.0, 0.5, 0.5, 0.0]
        elif n_feet == 2:
            offset = [0.0, 0.5]
        else:
            raise ValueError(
                f"[_unitport_feet_gait] offset auto-default supports "
                f"n_feet in {{2, 4}} (resolved via len(sensor_cfg."
                f"body_ids)); got n_feet={n_feet}. Either provide an "
                f"explicit offset list of length {n_feet} in il_params "
                f"or fix the feet IR mapping for the bound robot."
            )
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
    # ``{item_value}`` = the per-item "Value" chip (gait period, s). 0.0 =
    # auto -> the inline reward falls back to its canonical 0.8 s default.
    # Period is robot-scale-dependent (longer/heavier legs swing slower), so
    # it is tunable per-canvas here rather than a hidden constant.
    # offset is no longer hardcoded in il_params -- the inline source
    # resolves the default per-family from len(sensor_cfg.body_ids) at
    # runtime (4-foot trot / 2-foot alternating). Callers may still
    # override by adding ``"offset": [..],`` here for non-default
    # phasings.
    il_params='"period": {item_value}, "sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}), "threshold": 0.5, "command_name": "base_velocity"',
    il_inline=INLINE_SOURCE,
    il_value_label='Gait Period',
    il_value_default=0.0,
    il_value_min=0.0,
    il_value_max=5.0,
    il_value_step=0.05,
    il_value_unit='s',
)
