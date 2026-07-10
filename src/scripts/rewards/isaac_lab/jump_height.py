# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Jump Height — reward base height above the nominal standing height (a jump).

Added for skill_command_path_design.md Slice 3: the first skill (jump) needs a
reward that rewards leaving the ground. Existing height rewards track a *setpoint*
(``base_height`` penalises deviation from standing) and ``lin_vel_z_penalty`` actively
suppresses vertical velocity — none reward a jump. This one returns ``max(0, root_z -
nominal_z)`` so it is 0 while standing and grows as the base rises. It is meant to be
GATED on a skill trigger (post-pulse window with decay); ungated it would reward
permanent bouncing.
"""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_jump_height(env, asset_cfg=SceneEntityCfg("robot")):
    """Reward positive base height above the asset's nominal standing height.

    Returns ``max(0, root_z - nominal_z)`` (capped at 2 m), where ``nominal_z`` is
    the spawn z (``default_root_state``) — brand-neutral, no per-robot tuning. 0 while
    standing; grows as the base rises. Gate this on a skill trigger's post-pulse window
    so it rewards jumping only when commanded.
    """
    import torch
    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    try:
        nominal_z = asset.data.default_root_state[:, 2]
    except Exception:
        nominal_z = torch.full_like(root_z, 0.34)
    height = root_z - nominal_z
    height = torch.nan_to_num(height, nan=0.0, posinf=2.0, neginf=0.0)
    return torch.clamp(height, min=0.0, max=2.0)
'''


ENTRY = reward_item(
    key='jump_height',
    polarity='reward',
    title='Jump Height',
    desc='Reward base height above the nominal standing height (a jump). GATE this on a '
         'skill trigger (skill package gated_by) so it rewards jumping only when commanded — '
         'ungated it rewards permanent bouncing.',
    default=5.0,
    min_value=0.0,
    max_value=50.0,
    step=0.5,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_jump_height',
    il_module=IL_MOD_INLINE,
    il_params='"asset_cfg": SceneEntityCfg("robot")',
    il_inline=INLINE_SOURCE,
)
