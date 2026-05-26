# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UnitPort Isaac Lab POSE-ONLY review launcher.

Sibling of :mod:`il_review_launcher` for a strictly bundle-free use case:
the user wants to visualise an Actor Setting init pose (canvas
``init_pos_{x,y,z}`` + ``init_joint_angles``) inside Isaac Sim **before**
training. No policy, no bundle, no deploy_contract — just spawn the USD
articulation, write the override pose into its initial state, hold the
joints with implicit-PD, and idle the viewport.

Flow:
  1. argparse (--sku, --scene, --base_pos, --joint_pos_by_ir, --hold_seconds)
  2. Bootstrap sys.path to RELEASE/src
  3. Start Isaac Sim (viewer ON)
  4. Resolve SKU → spec → USD path via registers.robots
  5. Build minimal InteractiveScene (ground + robot articulation),
     map IR roles → USD-format physical joint names, write the
     override pose into ``init_state.joint_pos``
  6. Idle loop: hold position target, render, exit when viewport closed
     OR (if ``--hold_seconds > 0``) when the hold budget elapses

Fail-loud (CLAUDE.md §1.8): unregistered SKU, missing USD path, no
USD joint table for this robot, IR-role lookup misses, → raise. No
silent zero substitution for unmapped joints.

Usage (called by IsaacSimPoseReviewTask, not directly)::

    python il_pose_review_launcher.py \\
        --sku unitree.go2 \\
        --scene flat_ground \\
        --base_pos 0.0,0.0,0.4 \\
        --joint_pos_by_ir '{"FL_hip":0.0,"FL_thigh":0.9,...}' \\
        --hold_seconds 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pathlib import Path as _Path
_SRC_ROOT = _Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

try:
    import h5py  # noqa: F401  — side-effect import only
except Exception:
    pass

from isaaclab.app import AppLauncher

# ── 1. CLI args ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="UnitPort Isaac Lab POSE-ONLY Review Launcher"
)
parser.add_argument("--sku", type=str, required=True,
                    help="Canonical robot SKU (e.g. 'unitree.go2')")
parser.add_argument("--scene", type=str, default="flat_ground",
                    help="Scene identifier. Phase 1 only supports 'flat_ground'.")
parser.add_argument("--base_pos", type=str, default="0,0,0.4",
                    help="Initial base pose as 'x,y,z' (m). Default 0,0,0.4")
parser.add_argument("--joint_pos_by_ir", type=str, default="{}",
                    help="IR-role keyed joint angles dict, JSON-encoded. "
                         "Empty dict (default) → robot stays at zero pose.")
parser.add_argument("--hold_seconds", type=float, default=0.0,
                    help="If > 0, exit after this many seconds of wall-clock "
                         "time. Default 0 = run until the viewport is closed.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── 2. Start Isaac Sim ───────────────────────────────────────────────────

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 3. Post-launch imports (need sim running) ────────────────────────────

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from registers import RegistryHub
from registers.robots import get_robot_spec, resolve_id

# ── 4. Parse args ────────────────────────────────────────────────────────

base_pos_parts = [p.strip() for p in args_cli.base_pos.split(",")]
if len(base_pos_parts) != 3:
    raise ValueError(
        f"--base_pos must be 'x,y,z'; got {args_cli.base_pos!r}"
    )
base_pos = (
    float(base_pos_parts[0]),
    float(base_pos_parts[1]),
    float(base_pos_parts[2]),
)

try:
    joint_pos_by_ir_raw = json.loads(args_cli.joint_pos_by_ir)
except json.JSONDecodeError as exc:
    raise ValueError(
        f"--joint_pos_by_ir is not valid JSON: {exc}; got "
        f"{args_cli.joint_pos_by_ir!r}"
    ) from exc
if not isinstance(joint_pos_by_ir_raw, dict):
    raise ValueError(
        f"--joint_pos_by_ir must JSON-decode to an object; got "
        f"{type(joint_pos_by_ir_raw).__name__}"
    )
joint_pos_by_ir: dict = {}
for k, v in joint_pos_by_ir_raw.items():
    try:
        joint_pos_by_ir[str(k)] = float(v)
    except (TypeError, ValueError):
        continue

# ── 5. Resolve SKU + USD + joint map ─────────────────────────────────────

RegistryHub.load_all()
sku_canonical = resolve_id(args_cli.sku) or args_cli.sku
spec = get_robot_spec(sku_canonical)
if spec is None:
    raise ValueError(
        f"SKU {args_cli.sku!r} (canonical {sku_canonical!r}) is not "
        f"registered in registers.robots; cannot resolve USD asset."
    )

usd_path = (
    str(getattr(spec, "usd_path", "") or "").strip()
    or str(getattr(spec, "usd_url", "") or "").strip()
)
if not usd_path:
    raise ValueError(
        f"Robot {sku_canonical!r} has no USD asset registered "
        f"(usd_path/usd_url both empty). Add a USD path to the robot's "
        f"registry entry before using the IsaacSim pose review."
    )

# USD joint table is the source of truth for IsaacSim. Build the
# IR-role → physical-joint-name map. CLAUDE.md §1.8: any joint whose
# IR-role appears in the override but cannot be mapped to a USD joint
# is a registry mismatch — raise instead of silently dropping it.
usd_joints = spec.joints_for("USD")
if not usd_joints:
    raise RuntimeError(
        f"Robot {sku_canonical!r} has no joints_per_format['USD'] entries "
        f"in the registry. The IsaacSim pose review needs USD joint names "
        f"to populate ArticulationCfg.init_state.joint_pos. Run 'Dump USD' "
        f"on the Robot Node to populate the USD table, or pick the MuJoCo "
        f"pose review backend for this robot."
    )

ir_to_usd_name: dict = {}
all_usd_names: list = []
for jspec in usd_joints.values():
    nm = str(jspec.get("name", ""))
    ir = str(jspec.get("ir_role", ""))
    if nm:
        all_usd_names.append(nm)
        if ir:
            ir_to_usd_name[ir] = nm

# Default-joint-pos array: walk the USD joint order; pull the override
# value for that joint's IR role if present, else 0. Same shape that
# ImplicitActuatorCfg + InitialStateCfg below need (dicts keyed by USD
# physical joint name).
init_joint_pos_by_name: dict = {nm: 0.0 for nm in all_usd_names}
unmapped_ir_roles: list = []
for ir_role, value in joint_pos_by_ir.items():
    nm = ir_to_usd_name.get(ir_role)
    if nm is None:
        unmapped_ir_roles.append(ir_role)
        continue
    init_joint_pos_by_name[nm] = float(value)

if unmapped_ir_roles:
    raise RuntimeError(
        f"{len(unmapped_ir_roles)} IR role(s) in --joint_pos_by_ir have "
        f"no matching joint in registers.robots[{sku_canonical!r}]"
        f".joints_per_format['USD']: {unmapped_ir_roles}. Either fix "
        f"the registry's USD joint table for this SKU or remove these "
        f"roles from the Actor Setting init_joint_angles."
    )

# Resolve SKU-specific PD defaults from the spec.default_actuator_params
# block (populated by RobotSpecRef.from_registry from
# registers/data/robots_canonical.json[<sku>].default_actuator_params).
#
# CLAUDE.md §1.8: previously fell back to mid-range defaults (kp=50/kd=2/
# effort=30) "so pose review is never blocked by missing tuning" — but
# those defaults are Go2-class and silently mistune Spot (42 kg) or G1
# (38 kg humanoid). Pose review with wrong PD looks visually plausible
# while the joints respond at the wrong speed; users have no way to
# notice. Refuse to fabricate when the registry didn't ship defaults
# for this SKU, with an actionable error pointing at the registry entry.
default_actuator = dict(getattr(spec, "default_actuator_params", {}) or {})
if not default_actuator:
    raise RuntimeError(
        f"[pose_review] sku={sku_canonical!r} has no "
        f"default_actuator_params in the registry (RobotSpec.default_"
        f"actuator_params is None / empty). Refusing to substitute "
        f"Go2-class PD defaults (kp=50, kd=2, effort=30) — they would "
        f"silently mistune pose review for this robot. Add a "
        f"default_actuator_params block to robots_canonical.json for "
        f"this SKU (or its variant parent) with stiffness/damping/"
        f"effort_limit values matched to the robot's mass + actuator "
        f"spec (CLAUDE.md §1.8)."
    )
for required_key in ("stiffness", "damping", "effort_limit"):
    if required_key not in default_actuator:
        raise RuntimeError(
            f"[pose_review] sku={sku_canonical!r} default_actuator_params "
            f"is missing required key {required_key!r}. Found keys: "
            f"{sorted(default_actuator)}. Refusing to substitute a "
            f"hardcoded default."
        )
default_kp = float(default_actuator["stiffness"])
default_kd = float(default_actuator["damping"])
default_effort = float(default_actuator["effort_limit"])

print(
    f"[pose_review] sku={sku_canonical} name={spec.name!r} usd={usd_path}",
    flush=True,
)
print(
    f"[pose_review] n_joints={len(all_usd_names)} base_pos={base_pos} "
    f"overridden_ir={len(joint_pos_by_ir)} kp={default_kp} kd={default_kd}",
    flush=True,
)

# ── 6. Build scene ───────────────────────────────────────────────────────


@configclass
class _PoseReviewSceneCfg(InteractiveSceneCfg):
    """Minimal pose-review scene: ground plane + robot articulation."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=base_pos,
            joint_pos=init_joint_pos_by_name,
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=default_kp,
                damping=default_kd,
                effort_limit=default_effort,
            ),
        },
    )


# Standard physics dt (matches il_review_launcher.py default).
sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0, render_interval=4)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.5))

scene = InteractiveScene(_PoseReviewSceneCfg(num_envs=1, env_spacing=2.0))
robot: Articulation = scene["robot"]

sim.reset()
print(
    f"[pose_review] scene built: {robot.num_joints} joints in articulation",
    flush=True,
)

if robot.num_joints != len(all_usd_names):
    print(
        f"[pose_review] [Warning] USD articulation has {robot.num_joints} "
        f"joints, registry declares {len(all_usd_names)} for SKU "
        f"{sku_canonical!r}. The viewer will still run but the registry "
        f"USD joint table is out of sync — re-run 'Dump USD' on the "
        f"Robot Node.",
        flush=True,
    )

# ── 7. Hold-pose loop ───────────────────────────────────────────────────

# Build the target tensor once. Implicit PD will pull the articulation
# toward this every step, so the override pose is held against gravity.
import torch
target_tensor = torch.tensor(
    [init_joint_pos_by_name[nm] for nm in robot.joint_names],
    dtype=torch.float32,
    device=sim.device,
).unsqueeze(0)

start_wall = time.perf_counter()
hold_budget = float(args_cli.hold_seconds)

print(
    f"[pose_review] holding pose "
    f"({'until viewport close' if hold_budget <= 0 else f'for {hold_budget}s'})",
    flush=True,
)

step_count = 0
while simulation_app.is_running():
    robot.set_joint_position_target(target_tensor)
    scene.write_data_to_sim()
    sim.step()
    scene.update(1.0 / 200.0)
    step_count += 1
    if hold_budget > 0 and (time.perf_counter() - start_wall) >= hold_budget:
        print(
            f"[pose_review] hold_seconds={hold_budget} elapsed; exiting",
            flush=True,
        )
        break

print(f"[pose_review] viewport loop done (ran {step_count} steps)", flush=True)
simulation_app.close()
