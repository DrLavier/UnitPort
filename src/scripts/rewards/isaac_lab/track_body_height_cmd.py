"""Track Body Height Cmd — Exponential reward for base height matching the commanded body_height."""

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
def _unitport_track_body_height_cmd(env, command_name="gait_command",
                                     asset_cfg=SceneEntityCfg("robot"), std=0.05):
    """Exponential reward for base height matching commanded body_height."""
    import torch
    term = env.command_manager.get_term(command_name)
    target = term.command[:, 5]
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    error = torch.square(height - target)
    return torch.exp(-error / (std * std))
'''


ENTRY = reward_item(
    key='track_body_height_cmd',
    polarity='reward',
    title='Track Body Height Cmd',
    desc='Exponential reward for base height matching the commanded body_height. Pairs with the Walk These Ways gait command — tightens body_height tracking during training.',
    default=0.3,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_func='_unitport_track_body_height_cmd',
    il_module=IL_MOD_INLINE,
    il_params='"command_name": "gait_command", "std": 0.05',
    il_inline=INLINE_SOURCE,
)
