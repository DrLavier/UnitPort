"""Undesired Contacts — Penalty when non-foot bodies make ground contact."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def undesired_contacts(contacts, foot_body_ids):
    """Count non-foot robot bodies in contact with the ground.

    Iterates MuJoCo contacts; skips foot bodies and world-world pairs.
    Returns float count for multiplying by a negative penalty weight.
    """
    hit = set()
    for c in contacts:
        b1, b2 = geom_bodyid[c.geom1], geom_bodyid[c.geom2]
        if b1 == 0 and b2 not in foot_body_ids:
            hit.add(b2)
        elif b2 == 0 and b1 not in foot_body_ids:
            hit.add(b1)
    return float(len(hit))
'''


ENTRY = reward_item(
    key='undesired_contacts',
    polarity='penalty',
    title='Undesired Contacts',
    desc='Penalty when non-foot bodies make ground contact.',
    default=-1.0,
    min_value=-10.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
