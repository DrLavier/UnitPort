# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Single-rigid-body lateral-slide friction probe (the CLEAN friction-cone test).

The legged ``lateral_slip`` probe is confounded: a zero-torque robot COLLAPSES
while sliding, and the two engines collapse differently, so the measured slip is
dominated by limb dynamics, not foot-ground friction. This module replaces the
robot with a SINGLE free rigid body (a box) sliding on a plane — no joints, no
collapse — so the slip is pure Coulomb foot-ground friction. Both engines build
the SAME box from ``AlignedPlant(plant_kind="box")`` (no SKU/USD needed); the
MuJoCo side reuses ``mujoco_driver.run_probe`` and the PhysX side runs
``il_blockslide_launcher`` (a box, not the robot articulation).

Acceptance (behavioural, per the user): both engines' slip must fall
monotonically with μ and stop at high μ (slip→0); the two slip-μ curves must be
the SAME shape differing only by a bounded factor. If either keeps sliding at
high μ, the probe is not yet clean.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

from .protocol import AlignedPlant, ContactCfg, Probe, ProbeSet

# A 5 kg, 0.1 m cube resting on a plane. Friction is overwritten per-probe; the
# placeholder "1" is replaced by the driver / launcher.
BOX_MJCF = """
<mujoco model="blockslide">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" contype="1" conaffinity="1"
          friction="1 0.005 0.0001"/>
    <body name="box" pos="0 0 0.05">
      <freejoint/>
      <geom name="box" type="box" size="0.05 0.05 0.05" mass="5.0"
            contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

# Box geometry the PhysX launcher must match (half-extents, mass, rest height).
BOX_HALF_EXTENT = 0.05
BOX_MASS = 5.0
BOX_REST_Z = 0.05


def build_box_plant(plant: AlignedPlant) -> Any:
    """Build the box MuJoCo model as a lightweight actor (mj_model/mj_data/
    joint_names) the existing ``mujoco_driver`` can drive unchanged."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(BOX_MJCF)
    if plant.friction_static is not None:
        model.geom_friction[:, 0] = float(plant.friction_static)
    data = mujoco.MjData(model)
    return SimpleNamespace(
        mj_model=model, mj_data=data, joint_names=[], robot_id="blockslide")


def build_block_slide_probe_set(
    *,
    friction_static: float = 0.8,
    vels: Sequence[float] = (0.5, 1.0, 2.0),
    mus: Sequence[float] = (0.1, 0.3, 0.6, 1.0, 1.5),
    n_steps: int = 300,
    n_repeats: int = 4,
) -> ProbeSet:
    """Box-slide battery: each (velocity × μ) is one probe — the box is launched
    laterally at ``v`` with sliding friction ``μ`` and ZERO control, decelerating
    by friction alone. The slip-μ curve at each velocity is the clean
    friction-cone signature."""
    plant = AlignedPlant(plant_kind="box", friction_static=friction_static)
    actor = build_box_plant(plant)
    nq, nv, nu = int(actor.mj_model.nq), int(actor.mj_model.nv), int(actor.mj_model.nu)
    qpos0 = [0.0, 0.0, BOX_REST_Z, 1.0, 0.0, 0.0, 0.0]   # box at rest, identity quat
    probes = []
    for vi, v in enumerate(vels):
        for fi, mu in enumerate(mus):
            qvel = [float(v), 0.0, 0.0, 0.0, 0.0, 0.0]    # lateral vx
            probes.append(Probe(
                probe_id=f"box_v{vi}_m{fi}",
                scenario="block_slide",
                dimension="friction_cone",
                init_qpos=list(qpos0),
                init_qvel=qvel,
                torque_seq=[[] for _ in range(n_steps)],  # nu=0, no control
                contact_cfg=ContactCfg(friction=float(mu)),
                n_repeats=n_repeats,
                ic_perturb_scale=0.005,
            ))
    return ProbeSet(aligned_plant=plant, probes=probes, n_qpos=nq, n_qvel=nv, nu=nu)


__all__ = [
    "BOX_MJCF", "BOX_HALF_EXTENT", "BOX_MASS", "BOX_REST_Z",
    "build_box_plant", "build_block_slide_probe_set",
]
