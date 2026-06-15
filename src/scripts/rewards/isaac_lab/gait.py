# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Gait Rhythm — clock-free diagonal-coordination reward: rewards the correct
gait PATTERN (diagonal pairs in antiphase) without dictating a period."""

from __future__ import annotations

from scripts.task_module import (
    ALG_AMP,
    ALG_PPO,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_feet_gait(env, offset=None, sensor_cfg=None):
    """Reward the correct gait PATTERN — clock-free, period-free.

    The previous form scored each foot against an ABSOLUTE phase clock
    (``episode_length_buf * step_dt % period``). That forced the policy to
    lock its footfalls to an external wall-clock unrelated to its own leg
    state, with a hand-picked period — the source of the "chaotic rhythm"
    the user saw: mid-training the policy can never match the phantom clock,
    so the reward flickers and never settles. This form judges ONLY the
    relationship between the legs, so a steady rhythm is free to self-organise
    at whatever frequency the robot's mass/legs prefer.

    The ``offset`` (IR-role-keyed, emitted from ``{ir_gait_phase:feet}``)
    partitions the feet into the two diagonal trot groups:
      * group A = feet whose phase offset is ~0.0  ({FL, RR}; biped {L}),
      * group B = feet whose phase offset is ~0.5  ({FR, RL}; biped {R}).
    A correct trot has the two groups in ANTIPHASE at every instant: one group
    planted while the other swings. We score that directly as the absolute
    difference of the two groups' mean ground contact:

        reward = | mean(contact[group A]) - mean(contact[group B]) |   in [0,1]

      * trot       (A planted, B airborne, or vice-versa) -> |1-0| = 1.0 (max),
      * standing   (all four planted)                     -> |1-1| = 0,
      * pronk      (all four airborne)                    -> |0-0| = 0,
      * bound      (front pair vs back pair) -> each diagonal group is split
        50/50 -> |0.5-0.5| = 0  (bound is correctly NOT a trot),
      * single-leg twitch (one of a pair lifts) -> |0.5-1| = 0.5 (< a real trot).

    Crucially this CANNOT be farmed statically: holding |diff|=1 requires one
    group airborne, and airborne feet must eventually land (physics), flipping
    the groups — so maximising the integral forces continuous alternation = a
    self-consistent rhythm, with NO imposed period. Stride DURATION is shaped
    separately by feet_air_time and lift height by feet_clearance; this term is
    pure phasing, exactly as before.

    ``offset`` is REQUIRED and keyed by IR role, never list position (§8): a
    positional guess is how a trot silently mis-pairs into a bound. The two
    groups are split at offset 0.25 so the {0.0, 0.5} trot phases land cleanly
    on either side regardless of feet ordering.
    """
    import torch
    if offset is None:
        raise ValueError(
            "[_unitport_feet_gait] offset is REQUIRED and must be the "
            "IR-role-keyed gait phase the compiler emits from "
            "{ir_gait_phase:feet} (one phase per foot, aligned to {ir:feet}). "
            "Refusing to guess a positional default — that is how a trot "
            "silently mis-pairs into a bound. Fix the feet IR mapping for the "
            "affected robot."
        )
    if len(offset) != len(sensor_cfg.body_ids):
        raise ValueError(
            f"[_unitport_feet_gait] offset length {len(offset)} != number of "
            f"feet {len(sensor_cfg.body_ids)} (body_ids). The IR-keyed offset "
            f"and the {{ir:feet}} body list must be the same ordered set."
        )
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    ).float()                                            # (n, n_feet) in {0,1}
    # Partition feet into the two diagonal groups by their IR-keyed phase
    # offset (~0.0 vs ~0.5). Built on the device/dtype of the contact tensor.
    group_a = [i for i, off in enumerate(offset) if float(off) < 0.25]
    group_b = [i for i, off in enumerate(offset) if float(off) >= 0.25]
    if not group_a or not group_b:
        raise ValueError(
            "[_unitport_feet_gait] gait offset did not split into two diagonal "
            f"groups (offset={list(offset)}). Expected ~0.0 and ~0.5 phases "
            "from {ir_gait_phase:feet}."
        )
    a_idx = torch.as_tensor(group_a, device=is_contact.device, dtype=torch.long)
    b_idx = torch.as_tensor(group_b, device=is_contact.device, dtype=torch.long)
    mean_a = is_contact.index_select(1, a_idx).mean(dim=1)
    mean_b = is_contact.index_select(1, b_idx).mean(dim=1)
    # Antiphase between the two diagonal groups: 1.0 for a clean trot instant,
    # 0 for standing / pronk / bound. Always in [0, 1] (no "don't move" valley).
    return torch.abs(mean_a - mean_b)
'''


ENTRY = reward_item(
    key='gait',
    polarity='reward',
    title='Gait Rhythm',
    desc=(
        'Clock-free diagonal-coordination reward: rewards the correct trot '
        'PATTERN (diagonal foot pairs in antiphase) at every instant, with no '
        'imposed period — a steady rhythm self-organises. Pair with Feet Air '
        'Time (stride duration) and Feet Clearance (lift height).'
    ),
    default=0.5,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_func='_unitport_feet_gait',
    il_module=IL_MOD_INLINE,
    # ``offset`` is emitted from ``{ir_gait_phase:feet}`` — the per-foot gait
    # phase ``body_ir.gait_phase_offsets_for_roles`` derives from each foot's
    # IR ROLE (not its list position), aligned 1:1 with the ``{ir:feet}``
    # body_names. ``preserve_order=True`` keeps the contact ``body_ids`` in that
    # same IR-resolved order so ``offset[i]`` lines up with foot i. No period /
    # Value chip: the reward judges only the inter-leg phase relationship.
    il_params=(
        '"offset": {ir_gait_phase:feet}, '
        '"sensor_cfg": SceneEntityCfg("contact_forces", body_names={ir:feet}, '
        'preserve_order=True)'
    ),
    il_inline=INLINE_SOURCE,
)
