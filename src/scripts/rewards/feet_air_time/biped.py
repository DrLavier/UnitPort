# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Biped feet_air_time reward — IsaacLab stock ``feet_air_time_positive_biped``.

Variant of the canonical ``feet_air_time`` reward, tuned for two-foot
humanoids. The ONE substantive difference vs the preset (quadruped/
generic) version is numerical, not a hidden gate:

* **Per-foot clamp at threshold.** The preset rewards ``(last_air_time
  - threshold) * first_contact``, which on a quadruped's 4-foot trot
  averages out into a stable trot signal. On a biped with only 2 feet,
  the same expression spikes negative when a foot just lifted (because
  ``last_air_time`` resets to 0). The biped variant uses
  ``clamp(last_air_time - threshold, min=0.0)`` so single-stance
  shorter than threshold contributes 0, not a penalty — matching the
  positive-only signal IL uses in ``Isaac-Velocity-Flat-G1-v0``.

Like the preset, this variant is PURE stride-duration shaping: it makes
NO check on base velocity, displacement, or command magnitude (the old
command gate was removed so the term does exactly what its name says and
composes freely). A still stance scores zero naturally — feet that never
leave the ground never fire ``first_contact``. Following the commanded
velocity is the velocity-tracking rewards' concern; route this reward only
to items where stepping is wanted (per-item wiring), not the stand item.

Function name MUST stay ``_unitport_feet_air_time`` — the env_cfg
compiler templates ``RewTerm(func=mdp.feet_air_time, ...)`` against
the preset's ``il_func`` and the variant must hot-swap into that
slot. Anchored by registry: ``IL_REWARD_REGISTRY['feet_air_time']``
in ``application/training/isaac_lab/task_module_registry.py``.
"""
def _unitport_feet_air_time(env, threshold=0.4, sensor_cfg=None):
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Per-foot air time at the moment of touchdown.
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # IL biped variant: clamp positive — single-stance under threshold
    # contributes 0 (no penalty), over-threshold air contributes the
    # excess. Sum across the (typically 2) feet.
    air_excess = (last_air_time - threshold).clamp(min=0.0)
    reward = torch.sum(air_excess * first_contact.float(), dim=1)
    return reward
