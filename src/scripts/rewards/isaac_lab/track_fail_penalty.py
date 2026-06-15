# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Track Fail Penalty — penalize the velocity-command tracking ERROR (failure to follow the commanded vel/yaw)."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_track_fail_penalty(env, std=0.25, command_name="base_velocity",
                                  asset_cfg=SceneEntityCfg("robot")):
    """Penalize FAILURE to follow the commanded velocity / yaw rate.

    UNBOUNDED command-relative squared error (negative velocity-tracking error):

        penalty_value = ||v_xy - cmd_xy||**2 + (w_z - cmd_z)**2

    summed over the linear-XY error and the yaw error. The canvas weight is
    NEGATIVE, so this DEDUCTS points in proportion to how far the robot's actual
    body-frame velocity is from the command. It is 0 only when the robot tracks
    perfectly, and grows without bound (quadratically) as it drifts.

    WHY UNBOUNDED (vs the earlier 1-exp bounded form): the bounded complement
    SATURATES — at a large velocity error (e.g. a robot standing still under a
    1.5 m/s command, error**2 = 2.25) ``1 - exp(-error/std**2)`` is already ~1.0
    and FLAT, so a stationary robot feels NO gradient pulling it toward moving.
    Both the exp tracking reward and the bounded penalty are flat in that
    region, which is exactly why the policy gets stuck standing. A raw squared
    error has a gradient EVERYWHERE — the farther from the command, the larger
    the penalty AND the steeper the pull toward following it. That is the
    "no-follow -> heavy penalty; the closer to the command, the smaller the
    deduction" positive-guidance signal.

    IMPORTANT — this penalizes the COMMAND-RELATIVE error (||actual - cmd||),
    NOT the absolute velocity. Penalizing absolute speed is the documented
    foot-gun that silently kills locomotion (a near-zero command would punish
    moving); here a zero command (the stand item) yields error ~= 0 -> ~0
    penalty for holding still, while a non-zero command yields a large penalty
    for NOT moving. ``std`` is accepted for emit-signature compatibility and is
    intentionally unused (the unbounded form has no saturation scale).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_err = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - cmd[:, :2]), dim=1)
    ang_err = torch.square(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2])
    return lin_err + ang_err
'''


ENTRY = reward_item(
    key='track_fail_penalty',
    polarity='penalty',
    title='Track Fail Penalty',
    desc='Penalty on the velocity-command tracking error (failure to follow commanded vel/yaw). UNBOUNDED command-relative squared error ||actual-cmd||² — 0 when perfectly tracking, grows quadratically (with a gradient everywhere, including from a dead standstill) as the robot drifts from the command. Penalizes command-relative error, NOT absolute speed (so a zero/stand command is ~free). Use a larger |weight| to force following harder.',
    default=-1.0,
    min_value=-5.0,
    max_value=0.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_track_fail_penalty',
    il_module=IL_MOD_INLINE,
    il_params='"std": {node_std}, "command_name": "base_velocity"',
    il_inline=INLINE_SOURCE,
)
