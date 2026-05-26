# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lin Vel XY Penalty — L2 penalty on horizontal linear velocity — unconditional."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_lin_vel_xy_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """L2 penalty on horizontal linear velocity — unconditional base-motion penalty.

    Complements track_lin_vel_xy: tracking reward saturates with a Gaussian
    kernel, this penalty grows linearly with |v|^2, giving stronger gradient
    to suppress drift when the commanded velocity is zero (stand) or
    yaw-dominant (turn).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b[:, :2]
    v = torch.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)
    v = torch.clamp(v, min=-10.0, max=10.0)
    return torch.sum(torch.square(v), dim=1)
'''


ENTRY = reward_item(
    key='lin_vel_xy_penalty',
    polarity='penalty',
    title='Lin Vel XY Penalty',
    desc='L2 penalty on horizontal linear velocity — unconditional. Use heavy weight for stand (kill drift) or moderate for turn (discourage forward leak during yaw).',
    default=-0.5,
    min_value=-20.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_lin_vel_xy_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
