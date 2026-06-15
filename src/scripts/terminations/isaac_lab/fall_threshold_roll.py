# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Roll Threshold (Isaac Lab) — Terminate when |roll| exceeds the upright margin.

Per-axis counterpart to the symmetric cone ``bad_orientation``. Pairs with
``fall_threshold_pitch`` for legged_gym-style asymmetric tilt limits (tolerate
fore/aft lean more than sideways tip-over). Compiled to an inline DoneTerm that
reads ``projected_gravity_b`` (see ``env_cfg_compiler``); roll = atan2(g_y, -g_z).

NOTE: this reuses the SB3 key string for code-management symmetry only. Canvases
are hard-bound to one engine and are never cross-engine portable.
"""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_ISAAC,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='fall_threshold_roll',
    title='Roll Threshold',
    desc='Terminate when the base roll (sideways tilt) exceeds this margin (rad). '
         'Per-axis; pair with Pitch Threshold for asymmetric tilt limits.',
    default=0.8,
    min_value=0.2,
    max_value=3.14,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
