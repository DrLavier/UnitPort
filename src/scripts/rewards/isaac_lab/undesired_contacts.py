"""Undesired Contacts — Penalty when non-foot bodies (torso, thighs, shoulders) make ground contact."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_undesired_contacts(env, threshold=1.0, sensor_cfg=None):
    """Penalize undesired contacts (non-foot bodies touching ground)."""
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1).float()
'''


ENTRY = reward_item(
    key='undesired_contacts',
    polarity='penalty',
    title='Undesired Contacts',
    desc='Penalty when non-foot bodies (torso, thighs, shoulders) make ground contact. Needs params: sensor_cfg with body_names regex.',
    default=-1.0,
    min_value=-10.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_undesired_contacts',
    il_module=IL_MOD_INLINE,
    il_params='"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:thighs_hips_base})',
    il_inline=INLINE_SOURCE,
)
