# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Pitch Threshold (Isaac Lab) — Terminate when |pitch| exceeds the upright margin.

Per-axis counterpart to the symmetric cone ``bad_orientation``. Pairs with
``fall_threshold_roll`` for legged_gym-style asymmetric tilt limits (tolerate
fore/aft lean more than sideways tip-over). Compiled to an inline DoneTerm that
reads ``projected_gravity_b`` (see ``env_cfg_compiler``); pitch = atan2(g_x, -g_z).

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
    key='fall_threshold_pitch',
    title='Pitch Threshold',
    desc='Terminate when the base pitch (fore/aft tilt) exceeds this margin (rad). '
         'Per-axis; pair with Roll Threshold for asymmetric tilt limits.',
    default=1.0,
    min_value=0.2,
    max_value=3.14,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
