# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo side of the Stage-1 open-loop measurement (runs in ``.venv311``).

Builds the SAME Layer-1 aligned plant the deploy stack realizes — ``MjActor``
neutralizes MJCF PD so ``ctrl`` is a direct joint torque and applies the
USD→MJCF inertia overlay; we additionally apply the Stage-0 single-source
ground friction (method A) and any per-probe contact override (``solref`` /
``solimp`` / ``friction``). Then it replays each probe's torque sequence
open-loop and records the realized next-state + contact response.

Determinism: a probe ``repeat`` is seeded by its ``ic_perturb_seeds[r]`` — same
seed → byte-identical rollout — so the discriminator's sensitivity estimate is
reproducible and the mock-PhysX end-to-end test is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from unitport_sdk import log_debug

from .protocol import (
    AlignedPlant,
    ContactCfg,
    EngineResult,
    Probe,
    ProbeSet,
    StepRecord,
    write_results_jsonl,
)


# ---------------------------------------------------------------------------
# plant construction
# ---------------------------------------------------------------------------

def load_mujoco_plant(plant: AlignedPlant) -> Any:
    """Build the aligned MuJoCo actor for ``plant``.

    ``mjcf_path`` (if set) wins over ``sku`` so tests stay self-contained; a
    bare path skips the overlay (no SKU to key it on), which is correct for a
    synthetic test MJCF. The Stage-0 ground friction is applied here (the
    runtime PolicyRunner does the same at deploy — we mirror it so the measured
    plant equals the deployed plant).
    """
    # Box plant (clean friction-cone probe): a single free rigid body, no
    # articulation/overlay — built from the synthetic box MJCF, not a SKU.
    if getattr(plant, "plant_kind", "robot") == "box":
        from .blockslide import build_box_plant
        return build_box_plant(plant)

    from application.service.runtime.simulation.mujoco.mj_actor import MjActor

    if plant.mjcf_path:
        actor = MjActor.from_path(plant.mjcf_path, robot_id=plant.sku or None)
    elif plant.sku:
        actor = MjActor.from_sku(plant.sku)
    else:
        raise ValueError(
            "load_mujoco_plant: AlignedPlant needs either mjcf_path or sku (§8)"
        )

    # Stage-0 single-source ground friction (method A): apply to every geom's
    # sliding column so the realized μ equals the canvas value (matches
    # generic_mujoco_env + PolicyRunner).
    if plant.friction_static is not None:
        mu = float(plant.friction_static)
        if np.isfinite(mu) and mu > 0.0:
            actor.mj_model.geom_friction[:, 0] = mu
    return actor


def plant_dims(actor: Any) -> Tuple[int, int, int]:
    """(n_qpos, n_qvel, nu) of the actor's model."""
    m = actor.mj_model
    return int(m.nq), int(m.nv), int(m.nu)


def actuator_joint_names(actor: Any) -> List[str]:
    """Per-ACTUATOR joint names in actuator order (length nu) — the canonical
    order ``torque_seq`` columns are in, NOT the full qpos joint list (which
    includes the free base). The Kit launcher maps these names onto the USD
    articulation so torque column j hits the same physical joint on both
    engines (joint-order parity)."""
    import mujoco

    m = actor.mj_model
    names: List[str] = []
    for i in range(int(m.nu)):
        jid = int(m.actuator_trnid[i, 0])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        names.append(nm if nm else f"act{i}")
    return names


def nominal_qpos(actor: Any) -> List[float]:
    """The nominal-stance full qpos — keyframe-0 if the MJCF declares one, else
    qpos0. Same source the gain solver uses; a valid standing pose (NOT the
    zero pose, which leaves e.g. the Go2 calf straight / out of the USD limit)."""
    m = actor.mj_model
    src = m.key_qpos[0] if int(m.nkey) > 0 else m.qpos0
    return [float(x) for x in src]


def nominal_actuator_joint_pos(actor: Any) -> dict:
    """Nominal stance angle per ACTUATOR joint (name → radians), from
    keyframe-0/qpos0. Baked into the probe file so the Kit launcher seeds the
    ArticulationCfg with a limit-valid default pose."""
    import mujoco

    m = actor.mj_model
    src = m.key_qpos[0] if int(m.nkey) > 0 else m.qpos0
    out: dict = {}
    for i in range(int(m.nu)):
        jid = int(m.actuator_trnid[i, 0])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"act{i}"
        qadr = int(m.jnt_qposadr[jid])
        out[nm] = float(src[qadr])
    return out


# ---------------------------------------------------------------------------
# open-loop rollout
# ---------------------------------------------------------------------------

def _apply_contact_cfg(model: Any, cfg: ContactCfg) -> None:
    """Apply a probe's Type-II contact override to the live model (all geoms)."""
    if cfg.solref is not None:
        model.geom_solref[:, : len(cfg.solref)] = np.asarray(cfg.solref, dtype=float)
    if cfg.solimp is not None:
        model.geom_solimp[:, : len(cfg.solimp)] = np.asarray(cfg.solimp, dtype=float)
    if cfg.friction is not None:
        model.geom_friction[:, 0] = float(cfg.friction)


def _gear(model: Any) -> np.ndarray:
    """Per-actuator transmission gear (ctrl = joint_torque / gear). A zero gear
    is a degenerate transmission — fail loud rather than divide-by-zero (§8)."""
    nu = int(model.nu)
    gear = np.asarray(model.actuator_gear[:nu, 0], dtype=float).copy()
    if np.any(np.abs(gear) < 1e-12):
        raise ValueError(
            "mujoco_driver: actuator_gear has a (near-)zero entry — degenerate "
            "transmission, cannot map joint torque to ctrl."
        )
    return gear


def _contact_force_and_slip(mujoco: Any, model: Any, data: Any) -> Tuple[float, int, float]:
    """NET total contact-force magnitude, contact count, and slip — measured the
    SAME WAY as the PhysX launcher (cross-engine comparability, 硬化一).

    * force = magnitude of the NET contact force vector on the robot: each
      contact's force is rotated from its contact frame to WORLD and accumulated
      into one net vector, then ``norm`` (matches ``norm(sum net_forces_w)``).
    * slip = max horizontal speed among BODIES IN CONTACT with the ground
      (matches the PhysX ``max horiz speed over contacting bodies``). NOT the
      tangential relative velocity at the contact point, and NOT a max over all
      bodies — the latter is dominated by the trunk and is not foot-ground slip.
    """
    ncon = int(data.ncon)
    if ncon == 0:
        return 0.0, 0, 0.0
    net = np.zeros(3, dtype=np.float64)
    out6 = np.zeros(6, dtype=np.float64)
    contacted_bodies: set = set()
    for i in range(ncon):
        mujoco.mj_contactForce(model, data, i, out6)
        con = data.contact[i]
        # contact-frame force [normal, tan1, tan2] → world via the 3×3 contact
        # frame (rows = normal, tangent1, tangent2 in world).
        frame = np.asarray(con.frame, dtype=float).reshape(3, 3)
        net += out6[0] * frame[0] + out6[1] * frame[1] + out6[2] * frame[2]
        # Record the robot (non-world) bodies participating in this contact.
        for g in (int(con.geom1), int(con.geom2)):
            bid = int(model.geom_bodyid[g])
            if bid != 0:  # 0 = world body
                contacted_bodies.add(bid)
    # Slip = max horizontal speed of the contacting bodies.
    slip = 0.0
    v6 = np.zeros(6, dtype=np.float64)
    for bid in contacted_bodies:
        try:
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)
            slip = max(slip, float(np.linalg.norm(v6[3:5])))  # xy linear speed
        except Exception:
            pass
    return float(np.linalg.norm(net)), ncon, slip


def run_probe(
    mujoco: Any,
    actor: Any,
    probe: Probe,
    *,
    friction_static: Optional[float],
) -> List[EngineResult]:
    """Open-loop replay of one probe across all its perturbed-IC repeats."""
    model = actor.mj_model
    data = actor.mj_data
    nu = int(model.nu)
    if probe.nu != nu:
        raise ValueError(
            f"run_probe[{probe.probe_id}]: torque width {probe.nu} != model nu {nu}"
        )
    if len(probe.init_qpos) != int(model.nq):
        raise ValueError(
            f"run_probe[{probe.probe_id}]: init_qpos width {len(probe.init_qpos)} "
            f"!= model nq {model.nq}"
        )
    if len(probe.init_qvel) != int(model.nv):
        raise ValueError(
            f"run_probe[{probe.probe_id}]: init_qvel width {len(probe.init_qvel)} "
            f"!= model nv {model.nv}"
        )

    gear = _gear(model)
    base_qpos = np.asarray(probe.init_qpos, dtype=float)
    base_qvel = np.asarray(probe.init_qvel, dtype=float)
    torque_seq = np.asarray(probe.torque_seq, dtype=float)  # (T, nu)

    # Snapshot the geom contact params so each repeat starts from the SAME plant
    # before applying this probe's override (no cross-repeat leakage).
    solref0 = model.geom_solref.copy()
    solimp0 = model.geom_solimp.copy()
    friction0 = model.geom_friction.copy()

    results: List[EngineResult] = []
    for r, seed in enumerate(probe.ic_perturb_seeds):
        # Restore plant, re-apply Stage-0 friction + this probe's override.
        model.geom_solref[:] = solref0
        model.geom_solimp[:] = solimp0
        model.geom_friction[:] = friction0
        if friction_static is not None and np.isfinite(friction_static) and friction_static > 0:
            model.geom_friction[:, 0] = float(friction_static)
        _apply_contact_cfg(model, probe.contact_cfg)

        # Seeded IC perturbation (seed 0 → no perturbation: the baseline run).
        mujoco.mj_resetData(model, data)
        qpos = base_qpos.copy()
        qvel = base_qvel.copy()
        if int(seed) != 0 and probe.ic_perturb_scale > 0.0:
            rng = np.random.default_rng(int(seed))
            qpos = qpos + rng.normal(0.0, probe.ic_perturb_scale, size=qpos.shape)
            qvel = qvel + rng.normal(0.0, probe.ic_perturb_scale, size=qvel.shape)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        steps: List[StepRecord] = []
        for t in range(torque_seq.shape[0]):
            data.ctrl[:nu] = torque_seq[t] / gear
            mujoco.mj_step(model, data)
            f, ncon, slip = _contact_force_and_slip(mujoco, model, data)
            steps.append(StepRecord(
                next_qpos=[float(x) for x in data.qpos],
                next_qvel=[float(x) for x in data.qvel],
                next_qacc=[float(x) for x in data.qacc],
                contact_force_norm=f,
                n_contacts=ncon,
                slip=slip,
            ))
        results.append(EngineResult(
            probe_id=probe.probe_id, repeat=r, engine="mujoco", steps=steps))
    return results


def run_probe_set(probe_set: ProbeSet, out_path: Path | str) -> List[EngineResult]:
    """Run every probe×repeat on MuJoCo and write ``mujoco_results.jsonl``.

    Returns the in-memory results too (handy for tests / the mock pipeline)."""
    import mujoco

    actor = load_mujoco_plant(probe_set.aligned_plant)
    nq, nv, nu = plant_dims(actor)
    log_debug(
        f"[sim2sim] MuJoCo plant built: sku={probe_set.aligned_plant.sku!r} "
        f"nq={nq} nv={nv} nu={nu} friction={probe_set.aligned_plant.friction_static}")
    if (nq, nv, nu) != (probe_set.n_qpos, probe_set.n_qvel, probe_set.nu):
        raise ValueError(
            f"run_probe_set: plant dims (nq={nq}, nv={nv}, nu={nu}) disagree with "
            f"the probe set (nq={probe_set.n_qpos}, nv={probe_set.n_qvel}, "
            f"nu={probe_set.nu}). Regenerate probes against this plant (§8)."
        )
    all_results: List[EngineResult] = []
    for probe in probe_set.probes:
        rs = run_probe(
            mujoco, actor, probe,
            friction_static=probe_set.aligned_plant.friction_static,
        )
        all_results.extend(rs)
        log_debug(
            f"[sim2sim] MuJoCo probe={probe.probe_id} "
            f"({probe.scenario}/{probe.dimension}) done: "
            f"{len(rs)} repeats × {probe.n_steps} steps")
    write_results_jsonl(out_path, all_results)
    return all_results


__all__ = ["load_mujoco_plant", "plant_dims", "run_probe", "run_probe_set"]
