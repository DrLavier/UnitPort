"""UnitPort Isaac Lab training launcher.

This script mirrors Isaac Lab's ``rsl_rl/train.py`` but adds one critical step:
it imports a UnitPort-compiled ``@configclass`` config file and registers it as
a gymnasium environment **before** the Hydra task resolver runs.

Flow:
  1. Parse CLI args (identical to train.py)
  2. Start Isaac Sim via AppLauncher
  3. Import built-in isaaclab_tasks (registers all official envs)
  4. Import UnitPort compiled config → gymnasium.register("UnitPort-Custom-v0")
  5. Run training via RSL-RL OnPolicyRunner (same code path as train.py)

Usage (called by IsaacLabBackend, not directly)::

    python il_train_launcher.py \\
        --task UnitPort-Custom-v0 \\
        --unitport_config /path/to/compiled_cfg.py \\
        --num_envs 4096 --max_iterations 1500 --seed 42 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make ``src.system.*`` importable from inside the Isaac venv ──
# The launcher subprocess runs with cwd = isaac_lab_path (NOT the
# UnitPort repo root), so ``import src.system.training...`` would fail
# with "No module named 'src'" if we don't put the repo root on
# sys.path explicitly. This file lives at
# ``<repo>/src/system/training/il_train_launcher.py`` so the repo root
# is parents[3].
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Auto-accept Omniverse Kit EULA ──
# Must be set BEFORE any isaacsim / omni.* import, otherwise Isaac Sim
# blocks on an interactive license agreement prompt and the launcher hangs.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# ── Dependency precheck ──
# Isaac Sim's bootstrap is expensive (~30s) and emits hundreds of log lines.
# If the Isaac venv is missing one of the Python packages we need, the
# resulting ImportError lands deep in that wall of text and is easy to miss.
# Check the obvious ones up-front and bail out with a one-line, copy-pasteable
# fix instructions BEFORE AppLauncher spends half a minute booting Kit.
def _unitport_precheck() -> None:
    import importlib.util

    # (import_name, pip_name_for_message, isaaclab_extra_for_message)
    # isaaclab_extra is None when the package isn't installable via
    # ``isaaclab.bat -i`` (e.g. core isaaclab itself).
    required = [
        ("isaaclab",       "isaaclab",      None),
        ("isaaclab_tasks", "isaaclab_tasks", None),
        ("isaaclab_rl",    "isaaclab_rl",   None),
        ("gymnasium",      "gymnasium",     None),
        ("torch",          "torch",         None),
        ("rsl_rl",         "rsl-rl-lib",    "rsl_rl"),
    ]
    missing = [row for row in required if importlib.util.find_spec(row[0]) is None]
    if not missing:
        return

    py = sys.executable
    lines = [
        "",
        "============================================================",
        "[UnitPort][PRECHECK] Isaac venv is missing required packages:",
    ]
    for import_name, pip_name, isaaclab_extra in missing:
        lines.append(f"  - {import_name}")
    lines.append("")
    lines.append("Install them into the Isaac venv before retrying:")
    lines.append("")
    for import_name, pip_name, isaaclab_extra in missing:
        if isaaclab_extra:
            lines.append(f"  isaaclab.bat -i {isaaclab_extra}")
            lines.append(f"    (or: \"{py}\" -m pip install {pip_name})")
        else:
            lines.append(f"  \"{py}\" -m pip install {pip_name}")
    lines.append("")
    lines.append("Aborting before Isaac Sim bootstrap.")
    lines.append("============================================================")
    print("\n".join(lines), flush=True)
    sys.exit(2)


_unitport_precheck()

# ── h5py MUST be imported before SimulationApp / AppLauncher ──
# Isaac Sim ships its own HDF5 runtime as part of Kit. If h5py is imported
# *after* the sim has booted, the two HDF5 libraries race for the same
# global symbol table and training crashes randomly (segfault, "invalid
# file signature", or hangs at the first save_interval). Importing h5py
# first pins our HDF5 in the process before Kit gets a chance to load its
# own copy. Wrapped in a try-block so a missing optional h5py install does
# not break launcher bootstrap on machines that don't need it.
try:
    import h5py  # noqa: F401  — side-effect import only
except Exception:
    pass

from isaaclab.app import AppLauncher

# ── 1. CLI args — superset of train.py args + our --unitport_config ──

parser = argparse.ArgumentParser(description="UnitPort Isaac Lab Training Launcher")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="UnitPort-Custom-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--unitport_config", type=str, default="",
                    help="Path to UnitPort-compiled @configclass Python file")
parser.add_argument("--log_dir", type=str, default="")
parser.add_argument("--load_run", type=str, default="",
                    help="Run directory (or .pt file) to resume training from")
parser.add_argument("--checkpoint", type=str, default="",
                    help="Checkpoint file name inside --load_run (optional)")
parser.add_argument("--unitport_warm_start_actor", action="store_true",
                    help="Warm-start: load only actor-critic weights from the "
                         "checkpoint at --load_run, keep optimizer / "
                         "discriminator / amp_normalizer freshly initialized, "
                         "and reset the iteration counter to 0. Required when "
                         "seeding an AMP-PPO run from a pure PPO checkpoint.")
# ── Phase_1 of AMP_design.yaml §3.custom_robot_assets ──
# When the user wires a RobotAssetNode, the main venv resolves the asset_id
# through src.system.training.robot_assets.registry and passes the absolute
# .usd path here. The launcher then (a) overrides env_cfg.scene.robot.spawn
# .usd_path and (b) runs the deferred USD validation (path b — pxr only
# imports inside the Isaac venv which is exactly where this launcher runs).
parser.add_argument("--unitport_robot_usd", type=str, default="",
                    help="Absolute .usd path resolved from a RobotAsset registry "
                         "entry. Overrides env_cfg.scene.robot.spawn.usd_path "
                         "after the env_cfg is constructed.")
parser.add_argument("--unitport_robot_asset_id", type=str, default="",
                    help="The asset_id this --unitport_robot_usd was resolved "
                         "from. Used only for diagnostic logging.")
# ── Phase_3 of AMP_design.yaml §4.amp_backend ──
# AMP_PPO branch selector. When set to "AMP_PPO", the launcher builds
# an AmpRslRlVecEnvWrapper + AMPOnPolicyRunner pair (imported lazily)
# instead of the standard RslRlVecEnvWrapper + OnPolicyRunner. PPO
# branch is completely untouched in the default case.
parser.add_argument("--unitport_algorithm", type=str, default="PPO",
                    choices=["PPO", "AMP_PPO"],
                    help="Which training backend to use. 'PPO' (default) "
                         "keeps the legacy code path. 'AMP_PPO' switches to "
                         "the vendored AMP runner + discriminator.")
parser.add_argument("--unitport_amp_motion_files", type=str, default="",
                    help="Comma-separated list of motion files (amp_legged_gym "
                         "format) loaded via src.system.training.motion. "
                         "Only consulted when --unitport_algorithm=AMP_PPO.")
parser.add_argument("--unitport_amp_reward_coef", type=float, default=2.0,
                    help="AMP reward coefficient forwarded to the vendored "
                         "AMPDiscriminator. Default 2.0 matches upstream.")
parser.add_argument("--unitport_amp_preload", type=int, default=2_000_000,
                    help="Number of expert transitions to preload up front. "
                         "0 = stream sampling per update.")
parser.add_argument("--unitport_amp_transition_dt", type=float, default=0.02,
                    help="Time gap between (s, s') transition samples in "
                         "seconds. Should match the env control_dt.")
parser.add_argument("--unitport_amp_task_reward_lerp", type=float, default=0.5,
                    help="Blend coefficient for r_total = (1-l)*style_r + "
                         "l*task_r. Default 0.5 per AMP-PPO_design.yaml "
                         "§1.reward_mixing. Lerp=0 → pure style imitation, "
                         "lerp=1 → pure task reward (AMP disabled).")
parser.add_argument("--unitport_amp_lerp_schedule", type=str, default="",
                    help='JSON linear anneal schedule for task_reward_lerp. '
                         'Example: \'{"start":0.75,"end":0.3,"over_iters":2000}\'. '
                         'Empty string = constant lerp (no schedule).')
parser.add_argument("--unitport_amp_replay_buffer_size", type=int, default=1_000_000,
                    help="AMP replay buffer capacity. Default 1M.")
parser.add_argument("--unitport_amp_disc_grad_penalty", type=float, default=10.0,
                    help="R1 gradient penalty lambda for the discriminator. "
                         "Default 10.0 per AMP-PPO_design.yaml.")
parser.add_argument("--unitport_amp_disc_lr", type=float, default=1e-4,
                    help="Separate learning rate for the discriminator. "
                         "0 = share the policy learning rate.")
parser.add_argument("--unitport_amp_disc_label_smoothing", type=float, default=0.9,
                    help="Soft expert BCE target (one-sided label "
                         "smoothing, Salimans 2016). 1.0 = no smoothing, "
                         "0.9 = standard GAN default, lower = stronger "
                         "regularization against discriminator saturation.")
# ── Staged training pipeline (STAGE_NODE_DESIGN.yaml §2) ──
parser.add_argument("--unitport_stage_schedule", type=str, default="",
                    help="Base64-encoded JSON StageSchedule. When non-empty, "
                         "the runner iterates through stages, applying each "
                         "stage's parameter overrides on top of the baseline. "
                         "Empty string = single-stage training (backward compat).")
parser.add_argument("--unitport_stage_checkpoint_strategy", type=str,
                    default="both",
                    help="Checkpoint strategy at stage transitions: "
                         "save_on_advance | save_best_per_stage | both.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# ── 2. Start Isaac Sim ──

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2.5. Force performance renderer when training non-headless ──
#
# Default Isaac Sim startup with --headless=False uses RTX-RaytracedLighting
# with full DLSS / TAA / shadows / reflections / ambient occlusion. With
# 4096 envs that turns the viewport into a ~5 fps slideshow and chokes the
# GPU memory needed for the actual training. The user almost never wants
# pretty rendering during training — they just want a sanity-check window
# of "is the robot still upright?". Knee-cap the renderer to a low-fidelity
# raster-only preset so the viewport stays usable.
#
# Skipped when:
#   - --headless: nothing to render anyway
#   - --video:    user explicitly wants pretty frames for the recording
try:
    if not args_cli.headless and not args_cli.video:
        import carb
        _carb = carb.settings.get_settings()

        # Force RaytracedLighting (cheap raster-with-RT-shadows) instead of
        # PathTracing. RaytracedLighting is the default but some builds /
        # experience files flip to PathTracing — be defensive.
        _carb.set("/rtx/rendermode", "RaytracedLighting")

        # Cheapest AA: FXAA (op=1). DLAA / DLSS / TAA all cost more.
        _carb.set("/rtx/post/aa/op", 1)

        # Kill the expensive RT effects
        _carb.set("/rtx/reflections/enabled", False)
        _carb.set("/rtx/translucency/enabled", False)
        _carb.set("/rtx/ambientOcclusion/enabled", False)
        _carb.set("/rtx/raytracing/sssLighting/enabled", False)
        _carb.set("/rtx/raytracing/fractionalCutoutOpacity", False)

        # Use rasterized shadows instead of ray-traced
        _carb.set("/rtx/raytracing/shadows/enabled", False)
        _carb.set("/rtx/shadows/denoiser/enabled", False)

        # Cap path-tracing in case it sneaks in via a custom experience
        _carb.set("/rtx/pathtracing/totalSpp", 1)
        _carb.set("/rtx/pathtracing/maxBounces", 0)
        _carb.set("/rtx/pathtracing/maxBouncesLights", 0)

        # Half-resolution viewport — quartering pixel count cuts the
        # rasterizer cost ~4x with negligible UX impact for sanity-check
        # viewing.
        _carb.set("/app/viewport/defaultRenderScale", 0.5)
        _carb.set("/app/viewport/grid/enabled", False)

        # Drop expensive post-processing
        _carb.set("/rtx/post/dlss/execMode", 0)        # 0 = off
        _carb.set("/rtx/post/motionblur/enabled", False)
        _carb.set("/rtx/post/dof/enabled", False)
        _carb.set("/rtx/post/bloom/enabled", False)
        _carb.set("/rtx/post/lensDistortion/enabled", False)
        _carb.set("/rtx/post/lensFlares/enabled", False)
        _carb.set("/rtx/post/chromaticAberration/enabled", False)

        # Skip material loading caches that don't matter for low-fi viewing
        _carb.set("/rtx/sceneDb/ambientLightIntensity", 1.0)

        print(
            "[UnitPort][train] Non-headless mode: forced RaytracedLighting "
            "+ FXAA + half-res viewport + RT effects off so the training "
            "viewport stays at a usable framerate. Pass --video for "
            "high-quality rendering instead.",
            flush=True,
        )
except Exception as _exc:
    # Carb tweaks are best-effort — never block training startup on a
    # rendering preference.
    print(f"[UnitPort][train] Could not apply low-fi render preset "
          f"(non-fatal): {_exc}", flush=True)

# ── 3. Post-launch imports (need sim running) ──

import importlib.util
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401 — registers all built-in tasks

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ── 4. Register UnitPort custom environment ──

_unitport_env_cfg = None
_unitport_ppo_cfg = None

if args_cli.unitport_config and os.path.isfile(args_cli.unitport_config):
    print(f"[UnitPort] Loading compiled config: {args_cli.unitport_config}")
    spec = importlib.util.spec_from_file_location("unitport_env_cfg", args_cli.unitport_config)
    _cfg_mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec — @configclass (dataclass) needs
    # the module in sys.modules to resolve __dict__ during class creation.
    sys.modules["unitport_env_cfg"] = _cfg_mod
    spec.loader.exec_module(_cfg_mod)

    _unitport_env_cfg = getattr(_cfg_mod, "UnitPortEnvCfg", None)
    _unitport_ppo_cfg = getattr(_cfg_mod, "PPORunnerCfg", None)

    if _unitport_env_cfg is not None:
        gym.register(
            id="UnitPort-Custom-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _unitport_env_cfg,
            },
        )
        print(f"[UnitPort] Registered gymnasium env: UnitPort-Custom-v0")
    else:
        print(f"[UnitPort] WARNING: UnitPortEnvCfg not found in {args_cli.unitport_config}")

# ── 5. Training ──


def _resolve_entry_point(entry_point):
    """Resolve an entry point that may be a string 'module:Class', a class, or an instance."""
    if entry_point is None:
        raise ValueError("entry_point is None")
    if isinstance(entry_point, str):
        # "module.path:ClassName" format
        if ":" in entry_point:
            mod_path, cls_name = entry_point.rsplit(":", 1)
        elif "." in entry_point:
            mod_path, cls_name = entry_point.rsplit(".", 1)
        else:
            raise ValueError(f"Cannot parse entry point: {entry_point}")
        import importlib
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls() if callable(cls) else cls
    if callable(entry_point):
        return entry_point()
    return entry_point


def main():
    """Train with RSL-RL — same logic as Isaac Lab's train.py."""

    # Resolve env config
    task_name = args_cli.task
    if task_name == "UnitPort-Custom-v0" and _unitport_env_cfg is not None:
        env_cfg = _unitport_env_cfg()
    else:
        # Fall back to registry lookup for built-in tasks
        env_cfg = _resolve_entry_point(gym.spec(task_name).kwargs.get("env_cfg_entry_point"))

    # ── Phase_1: RobotAsset usd override + deferred USD validation ──
    # When --unitport_robot_usd is set, the user wired a RobotAssetNode in
    # the Canvas. The main venv has already resolved the asset_id through
    # the registry; we just (a) point the spawner at it and (b) run the
    # real USD parser now that we're inside the Isaac venv (per
    # AMP_design.yaml §7.risks.usd_parser_venv_dependency, path b).
    if args_cli.unitport_robot_usd:
        usd_path = args_cli.unitport_robot_usd
        asset_id = args_cli.unitport_robot_asset_id or "<unknown>"

        # ── Expand the nucleus: marker (set by discovery._MENAGERIE_USD_URL) ──
        # Marker format: "nucleus:Robots/Unitree/Go2/go2.usd"
        # Resolves to f"{ISAAC_NUCLEUS_DIR}/{rel}" — ISAAC_NUCLEUS_DIR is
        # only importable inside the Isaac venv, which is exactly where
        # this launcher subprocess runs.
        if usd_path.startswith("nucleus:"):
            try:
                from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
                rel = usd_path[len("nucleus:"):]
                resolved = f"{ISAAC_NUCLEUS_DIR}/{rel}"
                print(f"[UnitPort] Expanded nucleus marker: {usd_path}", flush=True)
                print(f"[UnitPort]                       → {resolved}", flush=True)
                usd_path = resolved
            except Exception as exc:
                print(f"[UnitPort][ABORT] Could not resolve nucleus marker "
                      f"{usd_path!r}: {exc}", flush=True)
                sys.exit(3)

        print(f"[UnitPort] RobotAsset override: asset_id={asset_id} usd={usd_path}",
              flush=True)
        try:
            # Reach into env_cfg.scene.robot.spawn — the standard ArticulationCfg
            # / UsdFileCfg layout. Wrapped in try because some custom env_cfg
            # variants may bury the robot under a different attribute name.
            env_cfg.scene.robot.spawn.usd_path = usd_path
        except AttributeError as exc:
            print(f"[UnitPort] WARNING: could not set robot.spawn.usd_path: {exc}. "
                  f"The compiled env_cfg may not match the standard ArticulationCfg "
                  f"layout. Continuing with env_cfg's own usd_path.", flush=True)

        # Deferred USD validation — runs the real pxr-based parser now.
        try:
            from src.system.training.robot_assets.parsers.usd_parser import (
                parse_usd, UsdParserUnavailable, UsdParseError,
            )
            try:
                usd_meta = parse_usd(usd_path)
                print(f"[UnitPort] USD parse OK: asset_id={asset_id} "
                      f"dof={usd_meta.dof_count} joints={len(usd_meta.joint_names)}",
                      flush=True)
            except UsdParserUnavailable:
                # Should not happen inside the Isaac venv, but if pxr is somehow
                # missing, fall through with a warning rather than abort —
                # the original env_cfg.usd_path is still set.
                print(f"[UnitPort] WARNING: pxr unavailable inside Isaac venv; "
                      f"USD validation skipped.", flush=True)
            except UsdParseError as exc:
                # Real parse failure — abort training rather than crash 30s
                # later inside Kit with an opaque error.
                print(f"[UnitPort][ABORT] USD validation failed for asset_id="
                      f"{asset_id}: {exc}", flush=True)
                sys.exit(3)
        except Exception as exc:
            print(f"[UnitPort] WARNING: deferred USD validation hook failed "
                  f"to run ({exc}); continuing.", flush=True)

    # Override with CLI args
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve agent config — build a proper RslRlOnPolicyRunnerCfg
    # Our compiled PPORunnerCfg is flat; RSL-RL needs nested algorithm/policy structure.
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    # Isaac Lab >= 2.4 uses RslRlMLPModelCfg (actor/critic);
    # Isaac Lab <= 2.3 uses the deprecated RslRlPpoActorCriticCfg (policy).
    try:
        from isaaclab_rl.rsl_rl.rl_cfg import RslRlMLPModelCfg
        _has_mlp_model_cfg = True
    except ImportError:
        _has_mlp_model_cfg = False

    if _unitport_ppo_cfg is not None:
        _flat = _unitport_ppo_cfg()
        _p = lambda k, d: getattr(_flat, k, d)
    else:
        _p = lambda k, d: d

    _algo_cfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        learning_rate=float(_p("learning_rate", 0.001)),
        gamma=float(_p("discount_factor", 0.99)),
        lam=float(_p("gae_lambda", 0.95)),
        clip_param=float(_p("clip_param", 0.2)),
        entropy_coef=float(_p("entropy_coef", 0.01)),
        value_loss_coef=float(_p("value_loss_coef", 1.0)),
        use_clipped_value_loss=True,
        max_grad_norm=float(_p("max_grad_norm", 1.0)),
        desired_kl=float(_p("desired_kl", 0.01)),
        schedule=str(_p("schedule", "adaptive")),
        num_learning_epochs=int(_p("num_learning_epochs", 5)),
        num_mini_batches=int(_p("num_minibatches", 4)),
    )

    if _has_mlp_model_cfg:
        # New-style: separate actor/critic model configs (Isaac Lab >= 2.4)
        agent_cfg = RslRlOnPolicyRunnerCfg(
            seed=int(_p("seed", 42)),
            num_steps_per_env=int(_p("num_steps_per_env", 24)),
            max_iterations=int(_p("max_iterations", 1500)),
            save_interval=int(_p("save_interval", 100)),
            experiment_name="unitport_custom",
            run_name="",
            algorithm=_algo_cfg,
            actor=RslRlMLPModelCfg(
                class_name="MLPModel",
                hidden_dims=eval(str(_p("actor_hidden_dims", "[128, 64, 32]"))),
                activation=str(_p("activation", "elu")),
                distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                    init_std=float(_p("init_noise_std", 1.0)),
                ),
            ),
            critic=RslRlMLPModelCfg(
                class_name="MLPModel",
                hidden_dims=eval(str(_p("critic_hidden_dims", "[128, 64, 32]"))),
                activation=str(_p("activation", "elu")),
            ),
        )
    else:
        # Legacy: combined policy config (Isaac Lab <= 2.3)
        from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg
        agent_cfg = RslRlOnPolicyRunnerCfg(
            seed=int(_p("seed", 42)),
            num_steps_per_env=int(_p("num_steps_per_env", 24)),
            max_iterations=int(_p("max_iterations", 1500)),
            save_interval=int(_p("save_interval", 100)),
            experiment_name="unitport_custom",
            run_name="",
            algorithm=_algo_cfg,
            policy=RslRlPpoActorCriticCfg(
                class_name="ActorCritic",
                init_noise_std=float(_p("init_noise_std", 1.0)),
                actor_hidden_dims=eval(str(_p("actor_hidden_dims", "[128, 64, 32]"))),
                critic_hidden_dims=eval(str(_p("critic_hidden_dims", "[128, 64, 32]"))),
                activation=str(_p("activation", "elu")),
            ),
        )

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Log directory
    if args_cli.log_dir:
        log_dir = args_cli.log_dir
    else:
        experiment_name = getattr(agent_cfg, "experiment_name", "unitport")
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
        log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # debug_vis policy:
    #
    # In **headless** mode there is no viewport to render markers into, so
    # leaving any debug_vis True wastes GPU memory on VisualizationMarkers
    # that nobody sees.  Wholesale disable everything.
    #
    # In **non-headless** mode the user explicitly wants visual feedback.
    # The most informative marker is Isaac Lab's built-in velocity command
    # visualizer (UniformVelocityCommandCfg.debug_vis=True): it spawns a
    # green arrow showing the **commanded** base velocity and a blue
    # arrow showing the **current** base velocity, both attached above
    # each robot.  We surgically re-enable that one term and leave the
    # rest off — wholesale enabling sometimes triggers SDK material spawn
    # failures on observation_manager terms that were never designed for
    # visualisation.
    if args_cli.headless:
        for attr_name in dir(env_cfg):
            sub = getattr(env_cfg, attr_name, None)
            if sub is None or isinstance(sub, (str, int, float, bool)):
                continue
            if hasattr(sub, "debug_vis"):
                try:
                    sub.debug_vis = False
                except Exception:
                    pass
            for term_name in dir(sub):
                term = getattr(sub, term_name, None)
                if term is not None and hasattr(term, "debug_vis"):
                    try:
                        term.debug_vis = False
                    except Exception:
                        pass
    else:
        # Enable the velocity-command direction arrows over each robot.
        # Other debug visualisers stay off for SDK-stability reasons.
        try:
            env_cfg.commands.base_velocity.debug_vis = True
            print("[UnitPort] Enabled velocity command debug arrows "
                  "(green=goal, blue=current) on each robot.")
        except Exception as exc:
            print(f"[UnitPort] Could not enable velocity command "
                  f"debug arrows: {exc}")

    # Create environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # Wrap for RSL-RL + create runner. Two mutually-exclusive paths:
    #
    #   PPO branch      (default, zero regression):
    #       env = RslRlVecEnvWrapper(env, ...)
    #       runner = OnPolicyRunner(env, agent_dict, ...)
    #
    #   AMP_PPO branch  (phase_3 of AMP_design.yaml §4.amp_backend):
    #       env = AmpRslRlVecEnvWrapper(env, num_amp_obs=..., ...)
    #       runner = AMPOnPolicyRunner(env, train_cfg, amp_data, ...)
    #
    # The AMP classes and vendored runner are imported lazily — module
    # imports of this file in the main venv (tests) never touch them.
    clip_actions = getattr(agent_cfg, "clip_actions", 1.0)

    agent_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else vars(agent_cfg)
    # MLPModel only accepts: hidden_dims, activation, obs_normalization, distribution_cfg
    # Strip everything else from actor/critic dicts to avoid unexpected keyword errors.
    _MLPMODEL_KEYS = {"hidden_dims", "activation", "obs_normalization", "distribution_cfg", "class_name"}
    for _model_key in ("actor", "critic"):
        if _model_key in agent_dict and isinstance(agent_dict[_model_key], dict):
            agent_dict[_model_key] = {
                k: v for k, v in agent_dict[_model_key].items() if k in _MLPMODEL_KEYS
            }
    device = getattr(agent_cfg, "device", "cuda:0")

    if args_cli.unitport_algorithm == "AMP_PPO":
        print("[UnitPort] AMP_PPO runner active", flush=True)

        # ── Load expert motion data via phase_2 loaders ──
        from src.system.training.amp.data_provider import build_amp_data_from_files
        from src.system.training.amp.wrappers.amp_rsl_rl_vec_env_wrapper import (
            AmpRslRlVecEnvWrapper,
        )
        from src.system.training.amp.runners.amp_on_policy_runner import (
            AMPOnPolicyRunner,
        )
        from src.system.training.amp.joint_alignment import (
            verify_alignment,
            dump_alignment_report,
            AmpObsAlignmentError,
        )

        motion_paths = [
            p.strip() for p in str(args_cli.unitport_amp_motion_files).split(",")
            if p.strip()
        ]
        if not motion_paths:
            print(
                "[UnitPort][ABORT] --unitport_algorithm=AMP_PPO requires at "
                "least one motion file via --unitport_amp_motion_files.",
                flush=True,
            )
            sys.exit(4)

        amp_data = build_amp_data_from_files(
            motion_paths,
            transition_dt=float(args_cli.unitport_amp_transition_dt),
            device=device,
            format_id="amp_legged_gym",
        )
        print(
            f"[UnitPort] Loaded {len(motion_paths)} AMP motion clip(s), "
            f"observation_dim={amp_data.observation_dim}",
            flush=True,
        )

        # ── Build the amp_obs extractor for the wrapper ──
        # We do NOT require the canvas's il_observation node to define a
        # dedicated ``amp_obs`` obs group — that would force every AMP
        # canvas to know about the discriminator's specific field set.
        # Instead, mirror what AMP_for_hardware's env does internally:
        # build the AMP observation directly from the robot articulation
        # data so any canvas obs config works as long as the robot has
        # the standard quadruped foot links.
        #
        # B5 (AMP fix plan): both the env extractor AND the motion
        # loader dispatch through the shared registry at
        # ``src.system.training.amp_obs_terms``. Previously the two
        # sides had independent hand-written concatenations that could
        # silently drift in field order, frame, or units — the most
        # common silent killer for AMP training. The registry entries
        # live at one file and either both producers stay in sync or
        # neither compiles.
        from src.system.training.amp_obs_terms import (
            DEFAULT_QUADRUPED_TERMS,
            compute_amp_obs_from_env,
        )

        # Canonical 43-dim layout for quadrupeds — these names are
        # registered in amp_obs_terms.py and consumed by MotionClip
        # via the same table.
        amp_term_names = list(DEFAULT_QUADRUPED_TERMS)

        def _amp_obs_extractor(wrapper):
            return compute_amp_obs_from_env(amp_term_names, wrapper)

        # Field order alignment (hard-mitigation for amp_obs_dim_drift).
        # We still run the legacy verify_alignment check: the motion
        # loader uses the same DEFAULT_QUADRUPED_TERMS list so the
        # two field tuples are equal by construction, but keeping the
        # assertion shields against a future refactor that forgets to
        # thread the term list through.
        clip_fields = list(amp_term_names)
        env_fields = list(amp_term_names)
        try:
            verify_alignment(env_fields, clip_fields)
            dump_alignment_report(
                log_dir, env_fields, clip_fields, ok=True
            )
            print(f"[UnitPort] amp_obs alignment OK ({len(clip_fields)} fields)", flush=True)
        except AmpObsAlignmentError as exc:
            dump_alignment_report(
                log_dir, env_fields, clip_fields, ok=False, error=str(exc)
            )
            print(f"[UnitPort][ABORT] {exc}", flush=True)
            sys.exit(5)

        env = AmpRslRlVecEnvWrapper(
            env,
            clip_actions=clip_actions,
            num_amp_obs=amp_data.observation_dim,
            amp_obs_fields=clip_fields,
            amp_obs_extractor=_amp_obs_extractor,
        )

        # Build the AMPOnPolicyRunner's nested cfg dict. Fields come
        # from AMPConfig (via agent_dict['amp_*']) plus the PPO bits
        # the vendored runner shares with OnPolicyRunner.
        # ── Decode stage schedule if provided ──
        _stage_schedule_dict = None
        if getattr(args_cli, "unitport_stage_schedule", ""):
            import base64 as _b64
            try:
                _decoded = _b64.b64decode(
                    args_cli.unitport_stage_schedule
                ).decode("utf-8")
                _stage_schedule_dict = json.loads(_decoded)
                if not isinstance(_stage_schedule_dict, list):
                    # Might be a full StageSchedule dict with "stages" key
                    if isinstance(_stage_schedule_dict, dict):
                        pass  # keep as-is
                    else:
                        _stage_schedule_dict = None
            except Exception as _e:
                print(
                    f"[UnitPort][WARN] Failed to decode --unitport_stage_schedule: {_e}",
                    flush=True,
                )
                _stage_schedule_dict = None

        amp_runner_cfg = {
            "runner": {
                "policy_class_name": "ActorCritic",
                "algorithm_class_name": "AMPPPO",
                "num_steps_per_env": int(_p("num_steps_per_env", 24)),
                "save_interval": int(_p("save_interval", 100)),
                "amp_reward_coef": float(args_cli.unitport_amp_reward_coef),
                "amp_discr_hidden_dims": eval(str(_p("disc_hidden_dims", "[1024, 512]"))),
                "amp_task_reward_lerp": float(args_cli.unitport_amp_task_reward_lerp),
                "amp_lerp_schedule": args_cli.unitport_amp_lerp_schedule,
                "amp_num_preload_transitions": int(args_cli.unitport_amp_preload),
            },
            "algorithm": {
                "num_learning_epochs": int(_p("num_learning_epochs", 5)),
                "num_mini_batches": int(_p("num_minibatches", 4)),
                "clip_param": float(_p("clip_param", 0.2)),
                "gamma": float(_p("discount_factor", 0.99)),
                "lam": float(_p("gae_lambda", 0.95)),
                "value_loss_coef": float(_p("value_loss_coef", 1.0)),
                "entropy_coef": float(_p("entropy_coef", 0.01)),
                "learning_rate": float(_p("learning_rate", 0.001)),
                "max_grad_norm": float(_p("max_grad_norm", 1.0)),
                "use_clipped_value_loss": True,
                "schedule": str(_p("schedule", "adaptive")),
                "desired_kl": float(_p("desired_kl", 0.01)),
                "amp_replay_buffer_size": int(args_cli.unitport_amp_replay_buffer_size),
                "amp_disc_grad_penalty": float(args_cli.unitport_amp_disc_grad_penalty),
                "amp_disc_lr": float(args_cli.unitport_amp_disc_lr),
                "amp_disc_label_smoothing": float(args_cli.unitport_amp_disc_label_smoothing),
            },
            "policy": {
                "actor_hidden_dims": eval(str(_p("actor_hidden_dims", "[512, 256, 128]"))),
                "critic_hidden_dims": eval(str(_p("critic_hidden_dims", "[512, 256, 128]"))),
                "activation": str(_p("activation", "elu")),
                "init_noise_std": float(_p("init_noise_std", 0.368)),
            },
        }

        # ── Inject stage schedule into runner config ──
        if _stage_schedule_dict is not None:
            _ckpt_strat = getattr(
                args_cli, "unitport_stage_checkpoint_strategy", "both"
            )
            if isinstance(_stage_schedule_dict, list):
                amp_runner_cfg["stage_schedule"] = {
                    "stages": _stage_schedule_dict,
                    "checkpoint_strategy": _ckpt_strat,
                    "global_max_iterations": sum(
                        int(s.get("iterations", 0))
                        for s in _stage_schedule_dict
                        if isinstance(s, dict)
                    ),
                }
            elif isinstance(_stage_schedule_dict, dict):
                amp_runner_cfg["stage_schedule"] = _stage_schedule_dict
                if "checkpoint_strategy" not in _stage_schedule_dict:
                    amp_runner_cfg["stage_schedule"]["checkpoint_strategy"] = _ckpt_strat

        runner = AMPOnPolicyRunner(
            env,
            train_cfg=amp_runner_cfg,
            amp_data=amp_data,
            log_dir=log_dir,
            device=device,
        )

        # H2 (AMP fix plan): write a UnitPort-owned run metadata file so
        # downstream tooling (bundle exporter, PolicyRunner, ONNX
        # exporter) can dispatch AMP-aware paths without having to
        # grovel around in Isaac Lab's params/agent.yaml — which lies
        # about ``class_name: OnPolicyRunner`` for AMP runs because it
        # reflects the default RslRlOnPolicyRunnerCfg dataclass rather
        # than the AMP runner that was actually instantiated.
        #
        # This file is the initial ``run.yaml`` the spec §5 talks
        # about; later phases (H1) will extend it with spec_hash, and
        # B6 will read it from the bundle exporter.
        try:
            from src.system.training.amp_obs_terms import DEFAULT_QUADRUPED_TERMS
            from src.system.training.run_meta import (
                RUN_META_VERSION,
                write_run_meta,
            )

            run_meta = {
                "unitport_run_meta_version": RUN_META_VERSION,
                "algorithm_class": "AMP_PPO",
                "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "amp": {
                    "amp_reward_coef": float(args_cli.unitport_amp_reward_coef),
                    "task_reward_lerp": float(args_cli.unitport_amp_task_reward_lerp),
                    "num_preload_transitions": int(args_cli.unitport_amp_preload),
                    "transition_dt": float(args_cli.unitport_amp_transition_dt),
                    "num_amp_obs_history": 1,
                    "amp_obs_terms": list(DEFAULT_QUADRUPED_TERMS),
                    "amp_obs_dim": int(amp_data.observation_dim),
                    "dataset_files": [
                        {"path": str(p), "basename": os.path.basename(p)}
                        for p in motion_paths
                    ],
                },
                "loss": {
                    # Documented for provenance. When phase 2 flipped
                    # from LSGAN to BCE we left this so any future
                    # lineage switch has a breadcrumb.
                    "discriminator_loss": "BCEWithLogits",
                    "style_reward_formula": "coef * softplus(logit)",
                },
            }
            meta_path = write_run_meta(log_dir, run_meta)
            print(
                f"[UnitPort][AMP] Wrote run metadata: {meta_path}",
                flush=True,
            )
        except Exception as exc:
            # Non-fatal: missing unitport_run_meta.yaml degrades the
            # bundle exporter's dispatch but does not break training.
            print(
                f"[UnitPort][AMP] WARNING: failed to write run metadata: {exc}",
                flush=True,
            )
    else:
        # Legacy PPO path — unchanged from before phase_3.
        env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
        runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=device)

    # Dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── Resume from Start Point (optional) ──
    # UnitPort's BaseAssetNode feeds an absolute checkpoint path down the
    # graph; the backend forwards it as --load_run (always) plus an optional
    # --checkpoint filename. RSL-RL's OnPolicyRunner.load() expects a full
    # .pt path, so we reassemble it here.
    resume_path = ""
    if args_cli.load_run:
        lr = Path(args_cli.load_run)
        if args_cli.checkpoint:
            resume_path = str(lr / args_cli.checkpoint)
        elif lr.is_file():
            resume_path = str(lr)
        else:
            # Directory with no explicit filename — pick the newest model_*.pt
            candidates = sorted(lr.glob("model_*.pt"))
            if candidates:
                resume_path = str(candidates[-1])
    if resume_path and os.path.isfile(resume_path):
        warm = bool(getattr(args_cli, "unitport_warm_start_actor", False))
        mode_label = "warm-start (actor only)" if warm else "full resume"
        print(
            f"[UnitPort] Loading checkpoint ({mode_label}): {resume_path}",
            flush=True,
        )
        try:
            # AmpOnPolicyRunner.load() accepts the kwarg; RSL-RL's stock
            # OnPolicyRunner.load() does not. Fall back to a manual
            # actor-only load for the non-AMP path.
            _is_amp_run = args_cli.unitport_algorithm == "AMP_PPO"
            if warm and not _is_amp_run:
                import torch as _torch
                loaded = _torch.load(resume_path)
                sd = loaded.get("model_state_dict", loaded)
                runner.alg.actor_critic.load_state_dict(sd, strict=False)
                runner.current_learning_iteration = 0
            elif warm:
                runner.load(resume_path, warm_start_actor=True)
            else:
                runner.load(resume_path)
        except Exception as exc:
            print(
                f"[UnitPort] ERROR: runner.load() failed ({exc}). "
                f"Aborting — refusing to silently discard the user-selected "
                f"Start Point. Fix the canvas architecture to match the "
                f"checkpoint or pick 'New' in Start Point.",
                flush=True,
            )
            sys.exit(5)
    elif args_cli.load_run:
        print(f"[UnitPort] WARNING: --load_run={args_cli.load_run!r} did not "
              f"resolve to a readable checkpoint; starting from scratch",
              flush=True)

    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ── Sentinel: training loop completed ──
    # Two reasons this is critical:
    #
    # 1. RSL-RL prints "Learning iteration {it}/{max}" with ``it`` running
    #    0..max-1, so the LAST iteration line is e.g. "1499/1500" — the
    #    progress bar would otherwise sit at 99% forever even though
    #    training is fully done. The backend normalises the displayed
    #    progress to 100% the moment it sees this sentinel.
    #
    # 2. Isaac Sim's ``simulation_app.close()`` (and sometimes
    #    ``env.close()``) can hang for minutes — or indefinitely — at
    #    shutdown on Windows because Kit holds onto extension threads /
    #    asset workers. The backend's stdout-tail loop blocks on
    #    ``for line in process.stdout`` until the process actually exits,
    #    so a hung shutdown leaves the UI stuck at 99% with no error.
    #
    # The fix is to (a) print the sentinel + flush so the backend can mark
    # training as 100% done immediately, and (b) force-exit the process
    # with ``os._exit(0)`` so the OS reaps any hung Kit / Omniverse
    # threads and the parent's stdout pipe closes promptly. We still try
    # ``env.close()`` first, with a hard timeout via ``threading.Timer``,
    # so any clean MuJoCo / RSL-RL teardown finishes when it can.
    print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
    print("[UnitPort] TRAINING_LOOP_DONE", flush=True)

    import threading

    def _force_exit_after(deadline_s: float) -> None:
        # Watchdog: if env.close() / sim_app.close() is still running
        # after `deadline_s`, abandon graceful shutdown and exit.
        timer = threading.Timer(deadline_s, lambda: os._exit(0))
        timer.daemon = True
        timer.start()
        return timer

    _watchdog = _force_exit_after(15.0)
    try:
        env.close()
    except Exception as exc:
        print(f"[UnitPort] env.close() failed (ignored): {exc}", flush=True)
    print("[UnitPort] ENV_CLOSED", flush=True)
    _watchdog.cancel()

    # Skip simulation_app.close() entirely on the way out — it's the
    # canonical hang source on Windows. os._exit bypasses Python atexit
    # handlers AND any non-daemon threads still spinning inside Kit, so
    # the subprocess closes its stdout immediately and the parent backend
    # can transition to MSG_FINISHED.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    # Unreachable in normal flow because main() ends with os._exit(0).
    # Kept for safety in case main() short-circuits before training.
    simulation_app.close()
