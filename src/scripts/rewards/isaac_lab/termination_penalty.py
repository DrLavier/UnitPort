# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Termination Penalty — one-shot penalty on non-timeout episode termination.

Registers ``termination_penalty`` through the canonical per-file channel so it
appears in the Rewards node's selectable list and carries its ``polarity``
metadata to the UI (negative → red square badge). It was previously defined
only in the compile-side ``application.training.isaac_lab.task_module_registry``
``IL_REWARD_REGISTRY`` — the compiler resolved it but the UI registry
(``scripts``) could not list or classify it, so it rendered as a green reward
circle and was unmanageable from the Rewards node. Metadata here is kept
byte-identical to the task_module_registry entry to avoid drift.
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
def _unitport_is_terminated(env):
    """One-shot terminal penalty (1.0 on the step a non-timeout termination fires).

    Mirrors ``isaaclab.envs.mdp.is_terminated``: fires on the exact step the
    termination_manager flips, excluding the natural time_out cause so the
    policy isn't punished for reaching the episode horizon. Pair with a
    large negative weight (e.g. -200) for IL G1's hard fall penalty.
    """
    import torch
    tm = env.termination_manager
    # tm.terminated == any non-timeout termination fired this step.
    # tm.time_outs is the timeout-only mask; subtracting is unnecessary
    # because terminated already excludes time_out per IsaacLab contract.
    return tm.terminated.float()
'''


ENTRY = reward_item(
    key='termination_penalty',
    polarity='penalty',
    title='Termination Penalty',
    desc='One-shot penalty on non-timeout termination — IsaacLab\'s stock '
         'humanoid hard-fall signal (G1/H1 use weight ≈ -200). Pair with '
         'a positive ``alive_reward`` for a balanced survival objective.',
    default=-200.0,
    min_value=-1000.0,
    max_value=0.0,
    step=1.0,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_is_terminated',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
