# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Track Fail Penalty — penalize the velocity-command tracking error (failure to follow)."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


# Doc stub — the real SB3 implementation is ``_r_track_fail_penalty`` in
# ``application/training/envs/reward_terms.py`` (bound by the ENTRY key below).
INLINE_SOURCE = '''\
def track_fail_penalty(lin_vel_xy, ang_vel_z, cmd):
    """UNBOUNDED penalty on the command-relative velocity error (NOT abs speed).

    ||lin_vel_xy - cmd_xy||^2 + (wz - cmd_wz)^2
    0 when tracking perfectly, grows quadratically (gradient everywhere,
    including from a standstill) as the robot drifts off command. Canvas weight
    is negative. Mirrors IsaacLab ``_unitport_track_fail_penalty``.
    """
    return 0.0
'''


ENTRY = reward_item(
    key='track_fail_penalty',
    polarity='penalty',
    title='Track Fail Penalty',
    desc='Penalty on the velocity-command tracking error (failure to follow commanded vel/yaw). UNBOUNDED command-relative squared error ||actual-cmd||² — gradient everywhere (incl. from a standstill), penalizes command-relative error not absolute speed. Use a larger |weight| to force following harder.',
    default=-1.0,
    min_value=-5.0,
    max_value=0.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
