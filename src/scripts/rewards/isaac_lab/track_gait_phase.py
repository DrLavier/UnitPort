# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Track Gait Phase — Reward foot contact matching the expected stance/swing phase from a Walk These Ways gait command term."""

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
def _unitport_track_gait_phase(env, command_name="gait_command", sensor_cfg=None):
    """Reward foot contact matching the expected stance/swing phase.

    Walk These Ways §3: each foot has a local phase in [0, 1); it
    should be in stance (on the ground) when phase < 0.5, in swing
    otherwise. This reward is the mean per-foot agreement between
    expected stance and actual contact.
    """
    import torch
    term = env.command_manager.get_term(command_name)
    per_foot = term.per_foot_phase()                            # (n, 4)
    expected_stance = (per_foot < 0.5).float()
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = (torch.norm(forces, dim=-1) > 1.0).float()         # (n, n_feet)
    n_match = min(contact.shape[1], 4)
    match = 1.0 - torch.abs(
        expected_stance[:, :n_match] - contact[:, :n_match]
    )
    return match.mean(dim=-1)
'''


ENTRY = reward_item(
    key='track_gait_phase',
    polarity='reward',
    title='Track Gait Phase',
    desc='Reward foot contact matching the expected stance/swing phase from a Walk These Ways gait command term. Requires Training Commands with gait_enabled.',
    default=0.5,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_func='_unitport_track_gait_phase',
    il_module=IL_MOD_INLINE,
    il_params='"command_name": "gait_command", "sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet})',
    il_inline=INLINE_SOURCE,
)
