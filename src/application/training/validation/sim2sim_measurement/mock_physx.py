# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Mock PhysX side — synthesizes ``physx_results`` from the real MuJoCo results
by injecting a CONTROLLED cross-engine residual, so the whole Stage-1 pipeline
(discriminator → range table → residual report) is verifiable end-to-end in
``.venv311`` WITHOUT the Kit venv. The user later replaces this with the real
``il_sim2sim_launcher`` output.

Per dimension the injected behaviour is either:
  * **systematic** — PhysX = MuJoCo × a fixed factor (+ tiny deterministic
    epsilon): the cross-engine residual is a stable function of the input, so
    the discriminator's IC-sensitivity is LOW → classified systematic, factor
    recovered ≈ the injected one;
  * **chaotic** — PhysX = MuJoCo × (1 + large IC-seed-dependent noise): the
    residual swings across repeats → HIGH sensitivity → classified chaotic.

This lets the test assert the discriminator routes each dimension correctly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .protocol import EngineResult, ProbeSet, StepRecord


def _stable_int(s: str) -> int:
    """Deterministic per-string seed contribution. Python's built-in ``hash``
    is per-process randomized (PYTHONHASHSEED), which would make the mock — and
    thus the end-to-end test — non-reproducible. hashlib is stable."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "big")


@dataclass
class DimBehavior:
    mode: str = "systematic"   # "systematic" | "chaotic"
    factor: float = 1.2        # systematic multiplicative bias
    noise: float = 0.0         # systematic: tiny; chaotic: large IC-dependent


# Default injected behaviour: contact stiffness/transient/trajectory are
# systematic (engine-bias, randomizable); the friction cone is chaotic (slip is
# IC-sensitive) — exercises BOTH Layer-2 and Layer-3 routing.
DEFAULT_BEHAVIOR: Dict[str, DimBehavior] = {
    "contact_stiffness": DimBehavior("systematic", 1.30, 0.01),
    "contact_transient": DimBehavior("systematic", 1.15, 0.01),
    "trajectory": DimBehavior("systematic", 1.05, 0.005),
    "friction_cone": DimBehavior("chaotic", 1.0, 0.6),
}


def _primary_channel_ref(er: EngineResult, channel: str) -> float:
    """Characteristic magnitude of the discriminator's PRIMARY channel for this
    result. The chaotic injection scales its noise by THIS (not a global max),
    so the noise is commensurate with the channel the discriminator measures —
    a contact-force-scale noise must not swamp a slip-scale signal. Floored so a
    quasi-static channel still gets a visible, IC-varying perturbation."""
    if channel == "contact_force":
        m = max((abs(s.contact_force_norm) for s in er.steps), default=0.0)
    elif channel == "slip":
        m = max((abs(s.slip) for s in er.steps), default=0.0)
    else:  # joint_state
        m = max((max((abs(x) for x in s.next_qpos), default=0.0)
                 for s in er.steps), default=0.0)
    return max(m, 1e-2)


def synthesize_physx_results(
    probe_set: ProbeSet,
    mujoco_results: List[EngineResult],
    *,
    behavior: Optional[Dict[str, DimBehavior]] = None,
    seed: int = 0,
) -> List[EngineResult]:
    """Produce a deterministic mock ``physx_results`` from ``mujoco_results``.

    Systematic dimensions: PhysX = MuJoCo × fixed factor (stable residual →
    classified systematic, factor recovered). Chaotic dimensions: the
    discriminator's PRIMARY channel for the dimension gets per-step ADDITIVE
    noise scaled by that channel's own magnitude, drawn fresh per IC repeat
    (residual swings across repeats → classified chaotic). Targeting the primary
    channel (not all channels) keeps the noise commensurate with what the
    discriminator measures."""
    from .discriminator import PRIMARY_CHANNEL

    behavior = behavior or DEFAULT_BEHAVIOR
    dim_of: Dict[str, str] = {p.probe_id: p.dimension for p in probe_set.probes}

    out: List[EngineResult] = []
    for er in mujoco_results:
        dim = dim_of.get(er.probe_id, "trajectory")
        beh = behavior.get(dim, DimBehavior())
        chaotic = beh.mode == "chaotic"
        channel = PRIMARY_CHANNEL.get(dim, "joint_state")
        rng = (
            np.random.default_rng(seed + _stable_int(er.probe_id) + er.repeat)
            if chaotic else None
        )
        ref = _primary_channel_ref(er, channel) if chaotic else 0.0
        # Chaos signature = IC-sensitivity: draw ONE offset per repeat (not per
        # step) so the rollout's residual VARIES across IC-perturbed repeats. A
        # per-step draw would average out over the rollout (low cross-repeat
        # variance → mis-classified systematic); the per-repeat draw is what the
        # discriminator's std(R_r)/mean(R_r) is built to catch.
        offset = (beh.noise * ref * float(rng.normal())) if chaotic else 0.0

        def sysf(v: float) -> float:
            return float(v) * (beh.factor + beh.noise * 0.1)

        def chaos(v: float) -> float:
            return float(v) + offset

        steps: List[StepRecord] = []
        for s in er.steps:
            if chaotic:
                # Perturb only the primary channel chaotically; leave the rest
                # equal to MuJoCo (factor 1.0) so the injected chaos is exactly
                # on the measured channel.
                cf = chaos(s.contact_force_norm) if channel == "contact_force" else s.contact_force_norm
                sl = chaos(s.slip) if channel == "slip" else s.slip
                qp = [chaos(x) for x in s.next_qpos] if channel == "joint_state" else list(s.next_qpos)
                steps.append(StepRecord(
                    next_qpos=qp, next_qvel=list(s.next_qvel),
                    next_qacc=list(s.next_qacc),
                    contact_force_norm=cf, n_contacts=s.n_contacts, slip=sl))
            else:
                steps.append(StepRecord(
                    next_qpos=[sysf(x) for x in s.next_qpos],
                    next_qvel=[sysf(x) for x in s.next_qvel],
                    next_qacc=[sysf(x) for x in s.next_qacc],
                    contact_force_norm=sysf(s.contact_force_norm),
                    n_contacts=s.n_contacts, slip=sysf(s.slip)))
        out.append(EngineResult(
            probe_id=er.probe_id, repeat=er.repeat, engine="physx", steps=steps))
    return out


__all__ = ["DimBehavior", "DEFAULT_BEHAVIOR", "synthesize_physx_results"]
