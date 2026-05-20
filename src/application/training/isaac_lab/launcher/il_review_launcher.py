"""UnitPort Isaac Lab BUNDLE review launcher.

Loads a v1 bundle (manifest.yaml + policy.onnx + deploy_contract) and replays
the trained policy inside an Isaac Sim viewport. Bundle-only contract per
RELEASE/CLAUDE.md §1.9: the launcher reads ONLY the bundle directory + the
robot registry (for the SKU → USD path lookup). No reach-back into
``<project>/training/runs/``, no canvas state, no in-memory spec.

Flow:
  1. argparse (--bundle, --scene, --max_play_steps)
  2. Bootstrap sys.path to RELEASE/src
  3. Start Isaac Sim (AppLauncher with viewer on — review = visible)
  4. Load bundle manifest + deploy_contract + ONNX policy
  5. Resolve USD spawn path from manifest.robot.sku via registers.robots
  6. Build minimal InteractiveScene (ground + robot articulation)
  7. Replay loop:
       - compute obs (base_ang_vel / projected_gravity / joint_pos
         / joint_vel / last_action / commands) matching deploy_contract layout
       - run ONNX inference → raw action
       - scale + offset action → joint position target
       - inner physics step loop (decimation times)
       - render

Fail-loud throughout (CLAUDE.md §1.8): missing manifest fields, ONNX
shape mismatches, unregistered SKU → raise immediately with the
specific cause. No silent zero-substitutions for failed sensor reads
or shape mismatches.

Usage (called by IsaacSimReviewTask, not directly)::

    python il_review_launcher.py \\
        --bundle /abs/path/to/<bundle_dir> \\
        --scene flat_ground \\
        --max_play_steps 3000
"""
from __future__ import annotations

import argparse
import os
import sys

# Bootstrap sys.path to RELEASE/src so `from application... / from registers...`
# imports resolve in the Isaac Sim subprocess. Mirrors il_train_launcher.py
# parents[4] math: <RELEASE>/src/application/training/isaac_lab/launcher/<this>.
from pathlib import Path as _Path
_SRC_ROOT = _Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# h5py first — same race-condition workaround as il_train_launcher.
try:
    import h5py  # noqa: F401  — side-effect import only
except Exception:
    pass

from isaaclab.app import AppLauncher

# ── 1. CLI args ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="UnitPort Isaac Lab Bundle Review Launcher"
)
parser.add_argument("--bundle", type=str, required=True,
                    help="Absolute path to a v1 bundle directory")
parser.add_argument("--scene", type=str, default="flat_ground",
                    help="Scene identifier. Phase 1 only supports 'flat_ground'.")
parser.add_argument("--max_play_steps", type=int, default=3000,
                    help="Hard upper bound on the replay loop. At 50Hz this "
                         "is ~60s. Set to 0 to disable (run until viewport closed).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Review = visible viewport. AppLauncher reads --headless from args_cli;
# we leave it at its default False so the user sees the robot.

# ── 2. Start Isaac Sim ───────────────────────────────────────────────────

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 3. Post-launch imports (need sim running) ────────────────────────────

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Read bundle dict (post-import so any yaml lib load is after Kit boot,
# matching the order training_launcher does its env_cfg load).
import yaml

# Registry lookup for robot SKU → USD path.
from registers import RegistryHub
from registers.robots import get_robot_spec, resolve_id

# ── 4. Bundle load + validation ──────────────────────────────────────────

bundle_dir = _Path(args_cli.bundle).resolve()
if not bundle_dir.is_dir():
    raise FileNotFoundError(f"--bundle is not a directory: {bundle_dir}")

manifest_path = bundle_dir / "manifest.yaml"
if not manifest_path.is_file():
    raise FileNotFoundError(f"manifest.yaml missing in bundle: {manifest_path}")
onnx_path = bundle_dir / "policy.onnx"
if not onnx_path.is_file():
    raise FileNotFoundError(f"policy.onnx missing in bundle: {onnx_path}")

with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = yaml.safe_load(fh)
if not isinstance(manifest, dict):
    raise ValueError(f"manifest.yaml is not a mapping: {manifest_path}")

robot_section = manifest.get("robot")
if not isinstance(robot_section, dict):
    raise ValueError("manifest.robot section missing or not a mapping")
robot_sku = str(robot_section.get("sku") or "").strip()
if not robot_sku:
    raise ValueError("manifest.robot.sku is empty — re-export the bundle")

deploy_contract = manifest.get("deploy_contract")
if not isinstance(deploy_contract, dict):
    raise ValueError(
        "manifest.deploy_contract is missing — required for replay (it "
        "carries PD gains, joint order, action scale, and obs layout)"
    )

joint_names: list = list(deploy_contract.get("joint_sdk_names") or [])
if not joint_names:
    raise ValueError("deploy_contract.joint_sdk_names is empty")
n_joints = len(joint_names)

stiffness = list(deploy_contract.get("stiffness") or [])
damping = list(deploy_contract.get("damping") or [])
effort_limit = list(deploy_contract.get("effort_limit") or [])
default_joint_pos = list(deploy_contract.get("default_joint_pos") or [])
for name, val in (
    ("stiffness", stiffness),
    ("damping", damping),
    ("effort_limit", effort_limit),
    ("default_joint_pos", default_joint_pos),
):
    if len(val) != n_joints:
        raise ValueError(
            f"deploy_contract.{name} length {len(val)} != n_joints {n_joints}"
        )

# Stage I (sim2sim PD framework): when the bundle carries pd_param
# (schema v2 with ActuatorPDNode wired), re-derive PhysX-side gains via
# the canonical solver instead of trusting the stored stiffness/damping
# arrays. Catches the case where the contract has stale gains because
# someone edited pd_param by hand or pd_param drifted from the derived
# arrays during a bundle hand-edit. v1 bundles (no pd_param) skip this
# block silently — their stiffness/damping flow through unchanged.
pd_param_raw = deploy_contract.get("pd_param")
if isinstance(pd_param_raw, dict) and pd_param_raw:
    try:
        from application.physics.pd_param import PDParam as _PDParam
        from application.physics.physx_gain_solver import solve as _solve_physx
        _pd_param = _PDParam.from_dict(pd_param_raw)
        _phys = [str(n) for n in joint_names]
        # joint_sdk_names ARE IR roles (Phase 5 contract); the IsaacLab
        # review subprocess maps them onto the USD articulation by name.
        # For the gain solver we just need parallel arrays.
        _ir_roles = _phys
        _gains = _solve_physx(
            joint_order_physical=_phys,
            joint_ir_roles=_ir_roles,
            pd_param=_pd_param,
        )
        # Replace the cached arrays with freshly-derived ones; the
        # ImplicitActuatorCfg below will pick them up.
        stiffness = list(_gains.kp)
        damping = list(_gains.kd)
        print(
            f"[review] pd_param present — re-derived PhysX gains via "
            f"physx_gain_solver (family={_pd_param.family!r}, n={n_joints})",
            flush=True,
        )
    except Exception as _exc:
        print(
            f"[review] pd_param re-derive failed ({_exc!r}); falling back "
            f"to stored stiffness/damping arrays. WARN: bundle may have "
            f"stale gains.",
            flush=True,
        )
else:
    print(
        f"[review] no pd_param in deploy_contract — legacy v1 bundle. "
        f"Using stored stiffness/damping. Re-export to graduate onto the "
        f"(omega_n, zeta) PD framework.",
        flush=True,
    )

# CLAUDE.md §1.8: strict-mode deploy_contract — every field below is
# required and pre-validated by DeployContract.from_dict at bundle load
# time. The previous `or 0.0` / `or 0` / `or 1.0` chains were redundant
# defensive casts that masked "field missing" as "field is zero" until
# the next line happened to raise. Read missing fields explicitly so the
# error attribution is clean.
def _required_dc_field(field: str, caster):
    val = deploy_contract.get(field)
    if val is None:
        raise ValueError(
            f"[review] deploy_contract.{field} is missing. The bundle's "
            f"manifest.yaml::deploy_contract section is incomplete; "
            f"re-export the bundle through the current finalizer."
        )
    try:
        return caster(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[review] deploy_contract.{field}={val!r} is not coercible "
            f"to {caster.__name__}."
        ) from exc


sim_dt = _required_dc_field("sim_dt", float)
step_dt = _required_dc_field("step_dt", float)
decimation = _required_dc_field("decimation", int)
if sim_dt <= 0 or step_dt <= 0 or decimation < 1:
    raise ValueError(
        f"deploy_contract physics fields invalid: sim_dt={sim_dt}, "
        f"step_dt={step_dt}, decimation={decimation}"
    )

action_cfg = deploy_contract.get("action")
if not isinstance(action_cfg, dict) or "scale" not in action_cfg:
    raise ValueError(
        "[review] deploy_contract.action.scale is missing — refusing to "
        "substitute scale=1.0 default (CLAUDE.md §1.8). Action scale is "
        "applied to every deploy step (target = last_action * scale + "
        "default_joint_pos); wrong scale produces wrong control magnitude."
    )
action_scale = float(action_cfg["scale"])

observations = deploy_contract.get("observations")
if not isinstance(observations, dict) or not observations:
    raise ValueError(
        "[review] deploy_contract.observations is missing or empty — "
        "bundle is corrupt; re-export."
    )

# Resolve SKU → robot spec → USD path.
RegistryHub.load_all()
sku_canonical = resolve_id(robot_sku) or robot_sku
spec = get_robot_spec(sku_canonical)
if spec is None:
    raise ValueError(
        f"manifest.robot.sku={robot_sku!r} (canonical {sku_canonical!r}) "
        f"is not registered in registers.robots; cannot resolve USD path"
    )

usd_path = (
    str(getattr(spec, "usd_path", "") or "").strip()
    or str(getattr(spec, "usd_url", "") or "").strip()
)
if not usd_path:
    raise ValueError(
        f"robot {sku_canonical!r} has no usd_path / usd_url in registry — "
        f"IsaacSim review requires a USD asset"
    )

print(f"[review] bundle={bundle_dir.name} sku={sku_canonical} "
      f"name={spec.name!r} usd={usd_path}", flush=True)
print(f"[review] n_joints={n_joints} sim_dt={sim_dt} decimation={decimation} "
      f"action_scale={action_scale}", flush=True)

# ── 5. Build minimal scene ───────────────────────────────────────────────

@configclass
class _ReviewSceneCfg(InteractiveSceneCfg):
    """Minimal review scene: ground plane + robot articulation."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(getattr(spec, "target_height", 0.5) or 0.5)),
            joint_pos={name: float(p) for name, p in zip(joint_names, default_joint_pos)},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={name: float(k) for name, k in zip(joint_names, stiffness)},
                damping={name: float(d) for name, d in zip(joint_names, damping)},
                effort_limit={name: float(e) for name, e in zip(joint_names, effort_limit)},
            ),
        },
    )


sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, render_interval=decimation)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.5))

scene = InteractiveScene(_ReviewSceneCfg(num_envs=1, env_spacing=2.0))
robot: Articulation = scene["robot"]

sim.reset()
print(f"[review] scene built: {robot.num_joints} joints in articulation", flush=True)

# Sanity check: articulation joint count matches bundle joint count.
if robot.num_joints != n_joints:
    raise RuntimeError(
        f"USD articulation has {robot.num_joints} joints but bundle "
        f"deploy_contract declares {n_joints}. Re-train this robot or "
        f"check that the registered USD path matches the bundle's SKU."
    )

# ── 6. ONNX policy ───────────────────────────────────────────────────────

try:
    import onnxruntime as ort
except ImportError as exc:
    raise RuntimeError(
        "onnxruntime is required for IsaacSim review — install via "
        "`<isaac_python> -m pip install onnxruntime`"
    ) from exc

ort_session = ort.InferenceSession(
    str(onnx_path),
    providers=["CPUExecutionProvider"],  # CPU inference is fast enough for 1 env
)
ort_input_name = ort_session.get_inputs()[0].name
ort_input_shape = ort_session.get_inputs()[0].shape
ort_output_name = ort_session.get_outputs()[0].name

# Compute expected obs dim from deploy_contract.observations (sum of dim *
# history_length per term, dict-insertion order).
total_obs_dim = 0
obs_term_specs: list = []  # list of (name, dim, history_length)
for term_name, term_cfg in observations.items():
    if not isinstance(term_cfg, dict):
        raise ValueError(f"observations[{term_name}] is not a mapping")
    # CLAUDE.md §1.8: previously `term_cfg.get("dim") or 0` / `... or 1`
    # — the subsequent `dim <= 0` check turned that into a fail-loud, but
    # the error message lost track of whether `dim` was missing vs.
    # explicitly 0 (corrupt vs. unset bundle). Read explicitly so the
    # error names the missing field.
    if "dim" not in term_cfg:
        raise ValueError(
            f"observations[{term_name}] is missing required 'dim' field — "
            f"bundle's deploy_contract is corrupt."
        )
    dim = int(term_cfg["dim"])
    # history_length is genuinely optional — default to 1 (single-step
    # observation, the canonical case). WHY KEPT (§1.8 c): documented
    # default in the contract schema; substitution is not silent because
    # the bundle's manifest serialises history_length explicitly when
    # set, and 1 is the documented "no history" value.
    hist = int(term_cfg.get("history_length", 1))
    if dim <= 0 or hist < 1:
        raise ValueError(
            f"observations[{term_name}] invalid dim={dim} history_length={hist}"
        )
    obs_term_specs.append((str(term_name), dim, hist))
    total_obs_dim += dim * hist

# ONNX input shape sanity. Typical shape: [batch, total_obs_dim] or
# [batch, history, total_obs_dim]; we check the trailing dim.
trailing = ort_input_shape[-1]
if isinstance(trailing, int) and trailing != total_obs_dim:
    raise RuntimeError(
        f"ONNX input trailing dim {trailing} != computed total obs dim "
        f"{total_obs_dim} from deploy_contract.observations. Bundle's "
        f"manifest/contract is internally inconsistent — re-export."
    )

print(f"[review] policy.onnx loaded: input={ort_input_shape} "
      f"output={ort_session.get_outputs()[0].shape} "
      f"expected_obs_dim={total_obs_dim}", flush=True)

# ── 7. Replay loop ───────────────────────────────────────────────────────

device = sim.device
default_pos_tensor = torch.tensor(default_joint_pos, dtype=torch.float32, device=device)
last_action = np.zeros(n_joints, dtype=np.float32)
# Phase 1: commands defaults to zero (stand still). Mirrors MujocoReviewTask
# default. Future: read from --command argv or a UI subsection.
command = np.zeros(3, dtype=np.float32)


def _compute_obs_vec() -> np.ndarray:
    """Build obs vector matching deploy_contract.observations layout.

    Fails loud (raise) on any sensor read mismatch — no silent zero fill.
    Only the 6 standard rsl_rl locomotion terms are supported in Phase 1;
    a bundle requesting an unknown term gets a clear NotImplementedError.
    """
    parts: list = []
    for term_name, dim, hist in obs_term_specs:
        if hist != 1:
            raise NotImplementedError(
                f"obs term {term_name!r} has history_length={hist}; "
                f"Phase 1 IsaacSim review only supports history_length=1"
            )
        if term_name == "base_ang_vel":
            v = robot.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
        elif term_name == "projected_gravity":
            v = robot.data.projected_gravity_b[0].cpu().numpy().astype(np.float32)
        elif term_name == "joint_pos":
            v = (robot.data.joint_pos[0] - default_pos_tensor).cpu().numpy().astype(np.float32)
        elif term_name == "joint_vel":
            v = robot.data.joint_vel[0].cpu().numpy().astype(np.float32)
        elif term_name == "last_action":
            v = last_action.copy()
        elif term_name in ("commands", "velocity_command"):
            v = command.copy()
        else:
            raise NotImplementedError(
                f"obs term {term_name!r} is not supported by Phase 1 "
                f"IsaacSim review. Supported: base_ang_vel, projected_gravity, "
                f"joint_pos, joint_vel, last_action, commands. "
                f"Use the Mujoco review backend for this bundle."
            )
        if v.shape[0] != dim:
            raise RuntimeError(
                f"obs term {term_name!r}: computed dim={v.shape[0]} != "
                f"expected dim={dim} (deploy_contract.observations)"
            )
        parts.append(v)
    flat = np.concatenate(parts, axis=0)
    if flat.shape[0] != total_obs_dim:
        raise RuntimeError(
            f"obs flat dim={flat.shape[0]} != expected {total_obs_dim}"
        )
    return flat


step_count = 0
max_steps = args_cli.max_play_steps if args_cli.max_play_steps > 0 else float("inf")

print(f"[review] replay loop starting (max_steps={args_cli.max_play_steps})",
      flush=True)
while simulation_app.is_running() and step_count < max_steps:
    obs_vec = _compute_obs_vec()
    # Match ONNX expected input shape — typically [1, total_obs_dim].
    onnx_input = obs_vec.reshape(1, -1).astype(np.float32)
    raw_action = ort_session.run([ort_output_name], {ort_input_name: onnx_input})[0]
    if raw_action.shape[-1] != n_joints:
        raise RuntimeError(
            f"ONNX output trailing dim {raw_action.shape[-1]} != n_joints "
            f"{n_joints}. Bundle is internally inconsistent."
        )
    last_action = raw_action.reshape(n_joints).astype(np.float32)

    # Action -> joint position target: scale + offset by default_pos.
    target = (last_action * action_scale) + np.asarray(default_joint_pos,
                                                       dtype=np.float32)
    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    robot.set_joint_position_target(target_t)

    # Inner physics decimation loop.
    for _ in range(decimation):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    step_count += 1
    if step_count % 100 == 0:
        print(f"[review] step={step_count}", flush=True)

print(f"[review] replay loop done (ran {step_count} steps)", flush=True)
simulation_app.close()
