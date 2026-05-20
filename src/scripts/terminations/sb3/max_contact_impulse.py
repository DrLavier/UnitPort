"""Contact Impulse — Terminate on excessive impact indicating unstable collapse."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='max_contact_impulse',
    title='Contact Impulse',
    desc='Terminate on excessive impact indicating unstable collapse.',
    default=250.0,
    min_value=50.0,
    max_value=500.0,
    step=5.0,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
)
