# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Command Follow Timeout — terminate if the robot still hasn't followed the commanded velocity by half the episode (SB3)."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='command_follow_timeout',
    title='Command Follow Timeout',
    desc='Terminate when, past the HALFWAY point of the episode, the robot is still failing to follow a non-trivial velocity command (command-relative velocity error above this threshold, m/s). Self-gates on command magnitude (a stand command never triggers it). Mirrors the IsaacLab termination.',
    default=0.6,
    min_value=0.1,
    max_value=2.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
)
