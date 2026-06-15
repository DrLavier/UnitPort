# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Stand Still — reward a clean four-square, level stance WHEN commanded to
stand (zero velocity command): all feet planted AND base level."""

from __future__ import annotations

from scripts.task_module import (
    ALG_PPO,
    ALG_AMP,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_stand_still(env, command_name="base_velocity", sensor_cfg=None,
                          asset_cfg=SceneEntityCfg("robot"),
                          cmd_threshold=0.1, level_std=0.25):
    """Reward a clean standing posture WHEN commanded to stand.

    Active only on near-zero velocity commands (``||cmd|| < cmd_threshold``):
    the velocity-tracking rewards already make a still robot the optimum there,
    but nothing in the basic set forbids it from balancing on two legs with the
    base level and CoM still — so a policy shared with walk/turn happily floats
    a diagonal pair while "standing". This term closes that hole by rewarding
    the posture the user actually wants:

        reward = standing * feet_down * level

      * standing  = 1 when the command is ~zero, else 0 (so it never fights
        locomotion — it is silent under any moving command, regardless of which
        items it is wired to),
      * feet_down = fraction of feet in ground contact (1.0 = all four planted),
      * level     = exp(-||projected_gravity_xy||^2 / level_std^2), 1.0 when the
        base is perfectly level, decaying as it tilts.

    The product is the unique maximum (=1.0) at "all feet planted AND level
    while commanded to stand", and any floated foot OR tilt pulls it toward 0 —
    a direct positive gradient toward a square, level stance that the floating-
    leg artifact lacked. "Not moving" is left to the velocity-tracking terms
    (zero command -> exp(0) tracking max), kept as a separate concern.
    """
    import torch
    cmd = env.command_manager.get_command(command_name)[:, :3]
    standing = (torch.norm(cmd, dim=1) < cmd_threshold).float()
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    ).float()
    feet_down = in_contact.mean(dim=1)                   # 1.0 = all feet planted
    asset = env.scene[asset_cfg.name]
    grav_xy = torch.sum(
        torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    level = torch.exp(-grav_xy / (level_std * level_std))
    return standing * feet_down * level
'''


ENTRY = reward_item(
    key='stand_still',
    polarity='reward',
    title='Stand Still',
    desc=(
        'Reward a clean four-square, level stance WHEN commanded to stand '
        '(zero velocity command): all feet planted AND base level. Silent '
        'under any moving command, so it is safe on the global page; route it '
        'to the stand item to fix the "floats two legs while standing" artifact.'
    ),
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_func='_unitport_stand_still',
    il_module=IL_MOD_INLINE,
    # cmd_threshold gates on a near-zero command; level_std sets how sharply the
    # level factor decays with base tilt. Feet are the IR-keyed contact bodies.
    il_params=(
        '"command_name": "base_velocity", '
        '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}, '
        'preserve_order=True), '
        '"cmd_threshold": 0.1, "level_std": 0.25'
    ),
    il_inline=INLINE_SOURCE,
)
