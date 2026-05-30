# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Track Swing Height Cmd — Reward swing-phase foot apex height matching the commanded step_height."""

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
def _unitport_track_swing_height_cmd(env, command_name="gait_command",
                                      asset_cfg=SceneEntityCfg("robot"),
                                      std=0.02):
    """Reward swing-phase foot apex matching commanded step_height.

    Reads ``step_height_cmd()`` (family-agnostic via the CommandTerm
    abstract method) and ``per_foot_phase()`` (returns (n, n_feet) with
    n_feet auto-adapted: 4 for quadruped UniformGaitCommand, 2 for biped
    BipedGaitCommand). ``n_match`` is driven by ``per_foot.shape[1]`` so
    the slice width tracks the family count without a family-keyed
    branch.
    """
    import torch
    term = env.command_manager.get_term(command_name)
    target = term.step_height_cmd().unsqueeze(-1)                # (n, 1)
    per_foot = term.per_foot_phase()                             # (n, n_feet)
    is_swing = (per_foot >= 0.5).float()
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    n_match = min(foot_z.shape[1], per_foot.shape[1])
    foot_z = foot_z[:, :n_match]
    is_swing = is_swing[:, :n_match]
    error = torch.square(foot_z - target)
    reward = torch.exp(-error / (std * std)) * is_swing
    return reward.sum(dim=-1) / is_swing.sum(dim=-1).clamp(min=1.0)
'''


ENTRY = reward_item(
    key='track_swing_height_cmd',
    polarity='reward',
    title='Track Swing Height Cmd',
    desc='Reward swing-phase foot apex height matching the commanded step_height. Requires a gait command term; feet list auto-resolved via IR mapping.',
    default=0.3,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_func='_unitport_track_swing_height_cmd',
    il_module=IL_MOD_INLINE,
    il_params='"command_name": "gait_command", "asset_cfg": SceneEntityCfg("robot", body_names={ir:feet}), "std": 0.02',
    il_inline=INLINE_SOURCE,
)
