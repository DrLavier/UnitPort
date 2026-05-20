"""Illegal Contact — Terminate when net contact force on the configured bodies exceeds this Newton threshold."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_ISAAC,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='illegal_contact',
    title='Illegal Contact',
    desc="Terminate when net contact force on the configured bodies exceeds this Newton threshold. Body regex list is configured via the node's illegal_contact_bodies parameter.",
    default=1.0,
    min_value=0.1,
    max_value=200.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
