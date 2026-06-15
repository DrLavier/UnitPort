# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Command Follow Timeout — terminate if the robot still hasn't followed the commanded velocity by half the episode."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_ISAAC,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='command_follow_timeout',
    title='Command Follow Timeout',
    desc='Terminate when, past the HALFWAY point of the episode, the robot is still failing to follow a non-trivial velocity command — i.e. its command-relative velocity error stays above this threshold (m/s). Self-gates on command magnitude, so a near-zero (stand) command never triggers it. Survival-side pressure to actually move; emitted as a real failure termination (eats the termination penalty).',
    default=0.6,
    min_value=0.1,
    max_value=2.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
