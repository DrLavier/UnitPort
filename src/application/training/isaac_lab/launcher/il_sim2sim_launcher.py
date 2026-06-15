# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UnitPort sim2sim Stage-1 PhysX-side measurement launcher (Kit venv).

The PhysX half of the §4 cross-engine measurement. It runs INSIDE the Isaac Lab
``.venv`` under AppLauncher (the app venv cannot import ``pxr``/``isaaclab``),
reads the coordinator's ``sim2sim_probes.json``, replays each probe's joint
torque sequence OPEN-LOOP on the SAME Layer-1 aligned plant (armature emitted
into the actuator cfg, friction from the probe), and writes
``physx_results.jsonl`` — the batch-file contract the .venv311 coordinator reads
back for the residual analysis. No live IPC.

Run it (the coordinator prints this command)::

    <Engines/isaac_lab/.venv python> il_sim2sim_launcher.py \\
        --probes /path/sim2sim_probes.json \\
        --out    /path/physx_results.jsonl --headless

NOTE (env-gated): this launcher is built to the documented IsaacLab APIs and
syntax-verified, but is exercised only inside the Kit venv. The points marked
``KIT-API`` below (effort control, root/joint state writes, contact-sensor
readout) should be sanity-checked on the first real run — they are the places
the IsaacLab version on the user's machine may differ.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path as _Path

# ── Bootstrap RELEASE/src onto sys.path (mirror il_train_launcher) ──
_SRC_ROOT = _Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ── Auto-accept Omniverse Kit EULA BEFORE any isaacsim/omni import ──
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _precheck() -> None:
    import importlib.util

    required = ["isaaclab", "isaaclab.sim", "torch"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        print(
            "[UnitPort][SIM2SIM][PRECHECK] Isaac venv missing: "
            + ", ".join(missing)
            + " — install isaaclab + torch into the Kit venv first.",
            flush=True,
        )
        sys.exit(2)


_precheck()

try:  # pin our HDF5 before Kit loads its own (matches il_train_launcher)
    import h5py  # noqa: F401
except Exception:
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UnitPort sim2sim PhysX measurement")
    parser.add_argument("--probes", required=True, help="sim2sim_probes.json path")
    parser.add_argument("--out", required=True, help="physx_results.jsonl output path")
    parser.add_argument(
        "--experience", default="",
        help="optional Kit experience file (minimal headless by default)")
    # AppLauncher contributes --headless and friends.
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        # AppLauncher import is deferred to main(); argparse still works for
        # --probes/--out so a syntax / --help check outside Kit doesn't crash.
        parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Read the probe set FIRST (pure-python protocol module — Kit-safe import).
    from application.training.validation.sim2sim_measurement.protocol import (
        EngineResult,
        ProbeSet,
        StepRecord,
        write_results_jsonl,
    )

    probe_set = ProbeSet.read_json(args.probes)
    plant = probe_set.aligned_plant

    # ── Start Isaac Sim ──
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── Post-launch imports (need the sim running) ──
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import (
        Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg)
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass

    # ── Resolve the robot USD from the probe file's pre-baked ``usd_source`` ──
    # The COORDINATOR (.venv311, which has the registry/asset service) already
    # resolved this and baked it in, so this launcher imports NO app stack
    # (no ``registers``/``unitport_sdk`` → no PyQt6 in the headless Kit venv).
    # A ``nucleus:`` marker resolves against ISAAC_NUCLEUS_DIR here (Kit-side).
    usd_src = str(plant.usd_source or "")
    if not usd_src:
        raise ValueError(
            "il_sim2sim_launcher: aligned_plant.usd_source is empty — the "
            "coordinator could not resolve a USD for this SKU (a bare mjcf_path "
            "is MuJoCo-only). Regenerate probes with a SKU whose asset ships a "
            "USD."
        )
    if usd_src.startswith("nucleus:"):
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        usd_src = f"{ISAAC_NUCLEUS_DIR}/{usd_src[len('nucleus:'):]}"
    usd_path = usd_src

    # ── Build the aligned articulation: PURE EFFORT control (no PD), so
    # set_joint_efforts applies the raw joint torque exactly like MuJoCo's
    # neutralized ctrl. friction goes on the ground material; armature is folded
    # into the actuator cfg so PhysX realizes link+armature (Layer-1 parity). ──
    mu = float(plant.friction_static) if plant.friction_static is not None else 1.0
    # Nominal stance per joint (limit-valid) so the spawned articulation passes
    # IsaacLab's joint-limit validation — a default of 0 fails for joints whose
    # limits exclude 0 (e.g. the Go2 calf). Per-probe states override at runtime.
    default_joint_pos = {str(k): float(v) for k, v in plant.default_joint_pos.items()}

    @configclass
    class _MeasureSceneCfg(InteractiveSceneCfg):
        # Ground: a STATIC cuboid RigidObject (NOT GroundPlaneCfg, whose
        # physics_material is silently ignored in this IsaacLab — proven by the
        # px_slip being byte-identical across two different ground cfgs). As a
        # RigidObject it has a root_physx_view, so its material friction is
        # set+readback-confirmed PER PROBE just like the robot's — both surfaces
        # set to the probe μ ⇒ effective foot-ground friction = μ. Top face at
        # z=0 (center at -0.1, half-height 0.1).
        ground = RigidObjectCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(50.0, 50.0, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu, dynamic_friction=mu, restitution=0.0,
                    friction_combine_mode="min"),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.3, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            # KIT-API: activate_contact_sensors=True adds the PhysX contact-
            # reporter API to every rigid body so the ContactSensor below can
            # bind (else "could not find any bodies with contact reporter API").
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path), activate_contact_sensors=True),
            # Spawn-time placeholder only — the real per-probe (base + joint)
            # state is written after reset via _set_state(). Must be a non-empty
            # dict: Isaac Lab's _process_cfg feeds init_state.joint_pos to
            # resolve_matching_names_values(), which raises TypeError on None (a
            # bare ``or None`` here silently disabled the whole self-check when
            # the probe carried no default_joint_pos). ".*" -> 0.0 spawns at the
            # zero pose; _set_state immediately overwrites it for every probe.
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=default_joint_pos or {".*": 0.0}),
            # KIT-API: zero stiffness/damping ⇒ the implicit actuator passes the
            # commanded effort straight through (no PD), matching MuJoCo's
            # direct-torque ctrl. armature lifts PhysX onto the Layer-1 plant.
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0, damping=0.0,
                ),
            },
        )
        # KIT-API: contact sensor over all robot bodies for the contact channel.
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)

    print("[UnitPort][SIM2SIM] building scene + sim...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.002)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(_MeasureSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[UnitPort][SIM2SIM] sim ready (scene initialized).", flush=True)

    robot: Articulation = scene["robot"]
    ground_obj: RigidObject = scene["ground"]
    contact_sensor = scene["contact"]
    device = sim.device

    def _set_one_friction(view: Any, mu_probe: float, tag: str) -> str:
        """Set every shape's static+dynamic friction on ``view`` to ``mu_probe``
        and read back (KIT-API: root_physx_view.set_material_properties — the
        3-column friction tensor randomize_rigid_body_material writes)."""
        mat = view.get_material_properties().clone()       # (n_envs, n_shapes, 3)
        before = float(mat[0, 0, 0].item()) if mat.numel() else float("nan")
        mat[..., 0] = float(mu_probe)                      # static_friction
        mat[..., 1] = float(mu_probe)                      # dynamic_friction
        view.set_material_properties(
            mat, torch.arange(mat.shape[0], dtype=torch.int32))
        rb = view.get_material_properties()
        got = [round(float(rb[0, s, 0].item()), 3) for s in range(min(rb.shape[1], 4))]
        return (f"{tag} set->{mu_probe:.3f} (was {before:.3f}); "
                f"readback[:4]={got} n_shapes={rb.shape[1]}")

    def _set_robot_friction(mu_probe: float) -> None:
        """Set BOTH the robot foot material AND the ground material to ``mu_probe``
        (combine='min' on both, equal values ⇒ effective foot-ground friction =
        mu_probe). The ground is a RigidObject (not GroundPlaneCfg, whose material
        was ignored), so its friction is genuinely settable + readback-confirmed.
        Behavioural acceptance: px_slip must fall as μ rises."""
        lines = []
        for view, tag in ((robot.root_physx_view, "robot"),
                          (ground_obj.root_physx_view, "ground")):
            try:
                lines.append(_set_one_friction(view, mu_probe, tag))
            except Exception as exc:  # noqa: BLE001
                lines.append(f"{tag} set->{mu_probe:.3f} FAILED: {exc}")
        for ln in lines:
            print(f"[UnitPort][SIM2SIM]   {ln}", flush=True)
        try:
            with open(_Path(args.out).with_name("friction_diag.txt"), "a",
                      encoding="utf-8") as _fh:
                _fh.write(" | ".join(lines) + "\n")
        except Exception:
            pass

    # Map the probe's canonical joint order (MuJoCo) → USD articulation order.
    usd_joint_names = list(robot.joint_names)
    canon = list(plant.joint_names) or usd_joint_names
    name_to_usd = {n: i for i, n in enumerate(usd_joint_names)}
    try:
        canon_to_usd = [name_to_usd[n] for n in canon]
    except KeyError as exc:
        raise ValueError(
            f"il_sim2sim_launcher: probe joint {exc!s} not found in the USD "
            f"articulation joints {usd_joint_names}. The probe's joint_names "
            f"must match the deployed robot (joint-order parity by name)."
        ) from exc
    n_joints = len(canon)
    n_base_q = max(int(probe_set.n_qpos) - n_joints, 0)  # 7 for a free base, else 0

    def _set_state(init_qpos, init_qvel) -> None:
        """Write the probe's initial (base + joint) state into the sim.
        KIT-API: root pose/velocity + joint pos/vel writes."""
        q = np.asarray(init_qpos, dtype=np.float32)
        v = np.asarray(init_qvel, dtype=np.float32)
        if n_base_q >= 7:
            root_pose = q[:7].copy()           # [x,y,z, qw,qx,qy,qz] (MuJoCo wxyz)
            root_vel = v[:6].copy()            # [vx,vy,vz, wx,wy,wz]
            rp = torch.tensor(root_pose, device=device).unsqueeze(0)
            rv = torch.tensor(root_vel, device=device).unsqueeze(0)
            robot.write_root_pose_to_sim(rp)
            robot.write_root_velocity_to_sim(rv)
        jp_canon = q[n_base_q:n_base_q + n_joints]
        jv_canon = v[(n_base_q - 1 if n_base_q >= 7 else 0):][:n_joints] \
            if n_base_q >= 7 else v[:n_joints]
        # Reorder canonical → USD by name.
        jp = np.zeros(n_joints, dtype=np.float32)
        jv = np.zeros(n_joints, dtype=np.float32)
        for ci, ui in enumerate(canon_to_usd):
            jp[ui] = jp_canon[ci]
            jv[ui] = jv_canon[ci] if ci < len(jv_canon) else 0.0
        robot.write_joint_state_to_sim(
            torch.tensor(jp, device=device).unsqueeze(0),
            torch.tensor(jv, device=device).unsqueeze(0))
        scene.write_data_to_sim()

    def _read_step() -> StepRecord:
        """Read the realized next-state + contact (post-step). KIT-API."""
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        ja = robot.data.joint_acc[0].cpu().numpy() if hasattr(robot.data, "joint_acc") \
            else np.zeros_like(jp)
        # Reorder USD → canonical so the residual aligns with the MuJoCo side.
        jp_c = np.array([jp[canon_to_usd[ci]] for ci in range(n_joints)], dtype=float)
        jv_c = np.array([jv[canon_to_usd[ci]] for ci in range(n_joints)], dtype=float)
        ja_c = np.array([ja[canon_to_usd[ci]] for ci in range(n_joints)], dtype=float)
        # Base + joint qpos/qvel in the MuJoCo layout the protocol expects.
        if n_base_q >= 7:
            rp = robot.data.root_pose_w[0].cpu().numpy() \
                if hasattr(robot.data, "root_pose_w") else robot.data.root_state_w[0, :7].cpu().numpy()
            rv = robot.data.root_vel_w[0].cpu().numpy() \
                if hasattr(robot.data, "root_vel_w") else robot.data.root_state_w[0, 7:13].cpu().numpy()
            next_qpos = list(rp[:7]) + list(jp_c)
            next_qvel = list(rv[:6]) + list(jv_c)
            next_qacc = [0.0] * 6 + list(ja_c)
        else:
            next_qpos = list(jp_c)
            next_qvel = list(jv_c)
            next_qacc = list(ja_c)
        # Contact + slip channels (硬化一 comparability).
        #  * contact force = NET total contact-force magnitude (norm of the sum
        #    over bodies — matches the MuJoCo driver's norm-of-net).
        #  * slip = horizontal speed of bodies IN CONTACT only (matches the
        #    MuJoCo foot-ground tangential slip). The old "max over ALL bodies"
        #    was dominated by the TRUNK's lateral velocity (not foot-ground
        #    slip) — a measurement-口径 mismatch that inflated the friction factor.
        force, ncon, slip = 0.0, 0, 0.0
        try:
            nfw = contact_sensor.data.net_forces_w[0]    # (n_bodies, 3)
            force = float(torch.linalg.norm(nfw.sum(dim=0)).item())
            mask = torch.linalg.norm(nfw, dim=-1) > 1e-6  # bodies in contact
            ncon = int(mask.sum().item())
            horiz = torch.linalg.norm(robot.data.body_lin_vel_w[0][:, :2], dim=-1)
            slip = float(horiz[mask].max().item()) if bool(mask.any()) else 0.0
        except Exception:
            force, ncon, slip = 0.0, 0, 0.0
        return StepRecord(
            next_qpos=[float(x) for x in next_qpos],
            next_qvel=[float(x) for x in next_qvel],
            next_qacc=[float(x) for x in next_qacc],
            contact_force_norm=force, n_contacts=ncon, slip=slip)

    # ── Open-loop replay over every probe × repeat ──
    # NO per-repeat sim.reset(): write_root_pose/write_joint_state set the start
    # state directly — sim.reset() re-runs the full init + render() per call
    # (slow / can stall headless). Heartbeats are FLUSHED so progress is visible
    # even when stdout is redirected to a file. sim.step(render=False) skips the
    # viewport (we only need physics + sensors via scene.update).
    n_probes = len(probe_set.probes)
    print(f"[UnitPort][SIM2SIM] starting measurement: {n_probes} probes, "
          f"n_joints={n_joints}, free_base={n_base_q >= 7}", flush=True)
    results = []
    for pi, probe in enumerate(probe_set.probes):
        torque_seq = np.asarray(probe.torque_seq, dtype=np.float32)  # (T, nu)
        T = int(torque_seq.shape[0])
        nrep = len(probe.ic_perturb_seeds)
        # Realize this probe's contact_cfg.friction on PhysX (the per-probe μ
        # sweep). Absent override → the plant's default μ. (solref/solimp are
        # MuJoCo-only contact-stiffness knobs PhysX can't runtime-set — ignored
        # here; the drop probes' contact_stiffness factor is ≈1× regardless.)
        pf = (probe.contact_cfg.friction
              if probe.contact_cfg.friction is not None else mu)
        _set_robot_friction(pf)
        print(f"[UnitPort][SIM2SIM] probe {pi + 1}/{n_probes} {probe.probe_id} "
              f"({probe.scenario}, mu={pf:.3f}): {nrep} repeats × {T} steps",
              flush=True)
        for r, seed in enumerate(probe.ic_perturb_seeds):
            qp = np.asarray(probe.init_qpos, dtype=np.float32).copy()
            qv = np.asarray(probe.init_qvel, dtype=np.float32).copy()
            if int(seed) != 0 and probe.ic_perturb_scale > 0.0:
                rng = np.random.default_rng(int(seed))
                qp = qp + rng.normal(0.0, probe.ic_perturb_scale, size=qp.shape).astype(np.float32)
                qv = qv + rng.normal(0.0, probe.ic_perturb_scale, size=qv.shape).astype(np.float32)
            _set_state(qp, qv)
            steps = []
            for t in range(T):
                # Reorder canonical torque → USD, apply as raw effort.
                tau_usd = np.zeros(n_joints, dtype=np.float32)
                for ci, ui in enumerate(canon_to_usd):
                    tau_usd[ui] = torque_seq[t, ci]
                robot.set_joint_effort_target(
                    torch.tensor(tau_usd, device=device).unsqueeze(0))
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                steps.append(_read_step())
                if (t + 1) % 40 == 0:
                    print(f"[UnitPort][SIM2SIM]   {probe.probe_id} r{r} "
                          f"step {t + 1}/{T}", flush=True)
            results.append(EngineResult(
                probe_id=probe.probe_id, repeat=r, engine="physx", steps=steps))
            print(f"[UnitPort][SIM2SIM]   {probe.probe_id} repeat {r + 1}/{nrep} "
                  f"done", flush=True)

    write_results_jsonl(args.out, results)
    print(f"[UnitPort][SIM2SIM] wrote {len(results)} results → {args.out}", flush=True)
    # Isaac Sim's ``simulation_app.close()`` can hang for minutes — or never
    # return — on Windows during renderer / USD-context teardown. The result
    # file is already on disk (``write_results_jsonl`` is atomic: temp +
    # ``os.replace``), so there is nothing left to flush. Blocking on the
    # graceful close here is exactly what wedged the coordinator's wait: it
    # would burn the full measurement timeout AFTER a complete, expensive
    # measurement had been written, then the parent discarded the results. Skip
    # the hang-prone close and HARD-EXIT — the OS reclaims the GPU/USD contexts.
    # This mirrors the documented Kit reality (``discover_usd_bodies.py`` notes
    # Kit itself ``os._exit(0)``s during teardown on Isaac Sim 5.x); we make it
    # deterministic. (§11: the launcher half of the file-contract fix; the
    # coordinator half waits on the result FILE, not on this process exiting.)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
