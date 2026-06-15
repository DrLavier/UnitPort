# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PhysX side of the CLEAN friction-cone probe: a single rigid box sliding on a
plane (NO articulation → no robot collapse confound). Mirrors the MuJoCo box in
``…sim2sim_measurement.blockslide.BOX_MJCF``: a 5 kg, 0.1 m cube launched
laterally and decelerated by friction alone.

Runs inside the Kit venv (app-stack-free: only the pure ``protocol`` module is
imported, so no ``registers``/``unitport_sdk``/PyQt6). The coordinator prints the
command. Acceptance: the box stopping distance must fall monotonically with μ and
the box must STOP at high μ (slip→0).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path as _Path

_SRC_ROOT = _Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _precheck() -> None:
    import importlib.util

    missing = [m for m in ("isaaclab", "isaaclab.sim", "torch")
               if importlib.util.find_spec(m) is None]
    if missing:
        print("[UnitPort][BLOCKSLIDE][PRECHECK] Kit venv missing: "
              + ", ".join(missing), flush=True)
        sys.exit(2)


_precheck()

try:
    import h5py  # noqa: F401
except Exception:
    pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UnitPort blockslide friction probe")
    p.add_argument("--probes", required=True)
    p.add_argument("--out", required=True)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    except Exception:
        p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from application.training.validation.sim2sim_measurement.protocol import (
        EngineResult, ProbeSet, StepRecord, write_results_jsonl,
    )
    from application.training.validation.sim2sim_measurement.blockslide import (
        BOX_HALF_EXTENT, BOX_MASS, BOX_REST_Z,
    )

    probe_set = ProbeSet.read_json(args.probes)
    plant = probe_set.aligned_plant
    if plant.plant_kind != "box":
        raise ValueError(
            f"il_blockslide_launcher: aligned_plant.plant_kind={plant.plant_kind!r} "
            "is not 'box' — use il_sim2sim_launcher for robot probes.")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    mu0 = float(plant.friction_static) if plant.friction_static is not None else 1.0

    @configclass
    class _BoxSceneCfg(InteractiveSceneCfg):
        # Static ground slab (kinematic) — material set per-probe at runtime.
        ground = RigidObjectCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(50.0, 50.0, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu0, dynamic_friction=mu0, restitution=0.0,
                    friction_combine_mode="min"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        )
        # The sliding box (dynamic).
        box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(2 * BOX_HALF_EXTENT, 2 * BOX_HALF_EXTENT, 2 * BOX_HALF_EXTENT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu0, dynamic_friction=mu0, restitution=0.0,
                    friction_combine_mode="min"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BOX_REST_Z)),
        )

    print("[UnitPort][BLOCKSLIDE] building scene...", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002))
    scene = InteractiveScene(_BoxSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    print("[UnitPort][BLOCKSLIDE] sim ready.", flush=True)

    box: RigidObject = scene["box"]
    ground: RigidObject = scene["ground"]
    device = sim.device

    def _set_mu(view: object, mu: float) -> str:
        mat = view.get_material_properties().clone()
        mat[..., 0] = float(mu)
        mat[..., 1] = float(mu)
        view.set_material_properties(mat, torch.arange(mat.shape[0], dtype=torch.int32))
        rb = view.get_material_properties()
        return f"{round(float(rb[0, 0, 0].item()), 3)}(n={rb.shape[1]})"

    results = []
    for pi, probe in enumerate(probe_set.probes):
        mu = (probe.contact_cfg.friction
              if probe.contact_cfg.friction is not None else mu0)
        rb_box = _set_mu(box.root_physx_view, mu)
        rb_gnd = _set_mu(ground.root_physx_view, mu)
        T = len(probe.torque_seq)
        print(f"[UnitPort][BLOCKSLIDE] probe {pi+1}/{len(probe_set.probes)} "
              f"{probe.probe_id} mu={mu:.3f} box_rb={rb_box} gnd_rb={rb_gnd} "
              f"steps={T}", flush=True)
        for r, seed in enumerate(probe.ic_perturb_seeds):
            qp = np.asarray(probe.init_qpos, dtype=np.float32).copy()   # (7,)
            qv = np.asarray(probe.init_qvel, dtype=np.float32).copy()   # (6,)
            if int(seed) != 0 and probe.ic_perturb_scale > 0.0:
                rng = np.random.default_rng(int(seed))
                qp = qp + rng.normal(0.0, probe.ic_perturb_scale, size=qp.shape).astype(np.float32)
                qv = qv + rng.normal(0.0, probe.ic_perturb_scale, size=qv.shape).astype(np.float32)
            box.write_root_pose_to_sim(torch.tensor(qp[:7], device=device).unsqueeze(0))
            box.write_root_velocity_to_sim(torch.tensor(qv[:6], device=device).unsqueeze(0))
            scene.write_data_to_sim()
            steps = []
            for t in range(T):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                pose = box.data.root_pose_w[0].cpu().numpy() \
                    if hasattr(box.data, "root_pose_w") else box.data.root_state_w[0, :7].cpu().numpy()
                vel = box.data.root_vel_w[0].cpu().numpy() \
                    if hasattr(box.data, "root_vel_w") else box.data.root_state_w[0, 7:13].cpu().numpy()
                slip = float(np.linalg.norm(vel[:2]))      # box horizontal speed
                steps.append(StepRecord(
                    next_qpos=[float(x) for x in pose[:7]],
                    next_qvel=[float(x) for x in vel[:6]],
                    next_qacc=[0.0] * 6,
                    contact_force_norm=0.0, n_contacts=0, slip=slip))
            results.append(EngineResult(
                probe_id=probe.probe_id, repeat=r, engine="physx", steps=steps))

    write_results_jsonl(args.out, results)
    print(f"[UnitPort][BLOCKSLIDE] wrote {len(results)} results -> {args.out}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
