# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
import json
import os
import sys

# ── Bootstrap RELEASE/src onto sys.path ──
# The compiled per-run env_cfg file emitted by env_cfg_compiler.py contains
# imports like ``from application.training.amp import mdp_events`` — these
# resolve fine in the main UnitPort app (where ``src/`` is on sys.path via
# main.py / venv entry) but the Isaac Sim subprocess is launched with a
# vanilla environment whose sys.path doesn't include our project tree. The
# launcher itself lives at:
#     <RELEASE>/src/application/training/isaac_lab/launcher/il_train_launcher.py
# so parents[4] is the project ``src/`` root we need to prepend. Doing this
# unconditionally at the top of the launcher (before any other import that
# might reach into ``application.*``) makes the launcher portable: whatever
# cwd / PYTHONPATH the orchestrator hands us, framework imports always work.
from pathlib import Path as _Path
_SRC_ROOT = _Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

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

parser = argparse.ArgumentParser(
    description="UnitPort Isaac Lab Training Launcher",
    # ``fromfile_prefix_chars='@'`` enables argparse's response-file
    # convention: an argv token ``@path/to/argsfile`` is replaced by the
    # whitespace-stripped lines of that file. The Windows bat wrapper
    # (config.py:_write_win_wrapper) uses this to bypass cmd.exe's 8191
    # char line-length limit — without it, AMP runs (large stage_schedule
    # base64 + per-item AMP task_items + N motion file paths) silently
    # truncate mid-line and argparse fails with "expected one argument"
    # for whichever flag happened to land at the cut.
    fromfile_prefix_chars="@",
)
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
# When the user wires a RobotAssetNode, the main venv resolves the SKU
# through application.service.robot_assets and passes the absolute
# .usd path here. The launcher then (a) overrides env_cfg.scene.robot.spawn
# .usd_path and (b) runs the deferred USD validation (path b — pxr only
# imports inside the Isaac venv which is exactly where this launcher runs).
parser.add_argument("--unitport_robot_usd", type=str, default="",
                    help="Absolute .usd path resolved from a RobotAsset registry "
                         "entry. Overrides env_cfg.scene.robot.spawn.usd_path "
                         "after the env_cfg is constructed.")
parser.add_argument("--unitport_robot_sku", type=str, default="",
                    help="Canonical robot SKU as produced by "
                         "registers.robots.build_sku(brand, model). The "
                         "single identity carried across the subprocess "
                         "boundary — drives MJCF / USD lookup, RSI joint "
                         "naming, and diagnostic logging. Empty when the "
                         "caller has not wired a robot.")
parser.add_argument("--unitport_robot_family", type=str, default="",
                    help="Primary family slug (documented "
                         "registers.families convention: "
                         "RobotSpec.families[0]). Used by the launcher's "
                         "downstream alignment / motion-format dispatch. "
                         "Empty when the caller has not wired a robot.")
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
                         "format) loaded via application.training.motion. "
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
parser.add_argument("--unitport_amp_task_filter", type=str, default="",
                    help="Filter motion clips by task tag before building "
                         "the AMP dataset. Only clips whose inferred or "
                         "manifest-declared task_tag matches this value are "
                         "kept. Comma-separated for multi-tag (e.g. 'walk,trot'). "
                         "Empty = no filtering (all clips used).")
parser.add_argument("--unitport_amp_task_labels", type=str, default="",
                    help="JSON dict {filename_stem: tag} from canvas tag "
                         "table editor. Overrides auto-inferred tags.")
parser.add_argument("--unitport_amp_task_items", type=str, default="",
                    help="Base64-encoded JSON list of TaskItem dicts "
                         "(id, motion_tag, command_ranges, clip_ref, ...) "
                         "from CommandSchema.task_items. Enables "
                         "command-conditioned AMP expert sampling: "
                         "each env's current velocity command is mapped "
                         "to a task_id and the discriminator's expert "
                         "transitions are drawn from the matching "
                         "sub-buffer. Empty = legacy uniform sampling.")
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
parser.add_argument("--unitport_amp_disc_logit_clamp_max", type=float, default=4.0,
                    help="Upper clamp on discriminator logit BEFORE "
                         "softplus → AMP reward. Defends against runaway "
                         "disc that drives style_reward to 1e4+. Default "
                         "4.0 → softplus(4)≈4.02. Lower for tighter "
                         "capping; raise only if disc legitimately needs "
                         "more headroom.")
parser.add_argument("--unitport_amp_reward_clamp_per_step", type=float, default=50.0,
                    help="Per-step reward bound applied in the AMP "
                         "runner. Defends against physics-explosion "
                         "rewards (1e3-1e6/step) that destroy critic "
                         "learning. ±50 leaves ~1.5× headroom on "
                         "legitimate signal.")
parser.add_argument("--unitport_amp_policy_std_clamp_max", type=float, default=1.5,
                    help="Upper clamp on actor exploration std (AMP "
                         "path). Without it, KL-adaptive LR can push "
                         "std → ∞ → NaN crash. At action_scale=0.25, "
                         "std=1.5 → 0.375 rad/step joint sample.")
parser.add_argument("--unitport_amp_obs_fields", type=str, default="",
                    help="Comma-separated AMP obs term names overriding the "
                         "family-keyed default from "
                         "registers.amp_obs_templates. Each name must be "
                         "registered in application.training.amp.obs_terms. "
                         "Empty = use the dispatch table entry for the "
                         "robot's primary family (quadruped: 43-d default).")
parser.add_argument("--unitport_amp_no_auto_inject_ref_obs", action="store_true",
                    help="Disable auto-injection of reference-clip observations "
                         "into the discriminator input. Defaults to enabled.")
parser.add_argument("--unitport_amp_phase_mode", type=str, default="loop",
                    choices=["loop", "once", "clamp"],
                    help="Motion clip playback mode forwarded from "
                         "training_motion.phase_mode. Metadata-only today "
                         "(amp_data_provider does not branch on this yet); "
                         "stamped onto run_meta for provenance.")
parser.add_argument("--unitport_amp_no_random_start_phase", action="store_true",
                    help="Disable randomising the start phase per episode. "
                         "Metadata-only today (build_rsi_pool always samples "
                         "n_frames uniformly across clips).")
parser.add_argument("--unitport_amp_disc_overrides_file", type=str, default="",
                    help="Path to a JSON sidecar containing user-edited "
                         "AMP discriminator function bodies "
                         "(disc_forward / disc_compute_grad_pen / "
                         "disc_predict_reward). The main process emits "
                         "this when the user opens the discriminator "
                         "Source Code editor and saves changes; the "
                         "subprocess re-imports the registry fresh, so "
                         "without the sidecar the edits would be lost. "
                         "Empty / missing file = use registry defaults.")
parser.add_argument("--unitport_amp_rsi_prob", type=float, default=0.0,
                    help="AMP Reference-State Initialization probability per "
                         "reset. 0.0 disables (default). 0.8 recommended for "
                         "warm AMP training. Distinct namespace from the "
                         "legacy --unitport_rsi_prob which drives the "
                         "reference_frame_0 init_pose mode.")
parser.add_argument("--unitport_amp_rsi_n_frames", type=int, default=20000,
                    help="Number of frames sampled from expert motion clips "
                         "into the RSI pool at launcher startup.")
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
# ── RSI (Reference State Initialization) ──
parser.add_argument("--unitport_init_pose_mode", type=str, default="default",
                    help="Init pose mode: default | reference_frame_0 | keyframe | custom. "
                         "When 'reference_frame_0', env resets use AMP clip frames.")
parser.add_argument("--unitport_rsi_prob", type=float, default=1.0,
                    help="Probability [0,1] that a reset uses RSI. 1.0 = always.")
parser.add_argument("--unitport_rsi_sample_mode", type=str, default="frame_0",
                    help="RSI frame selection: frame_0 | uniform_phase.")
parser.add_argument("--unitport_body_mapping", type=str, default="",
                    help="Base64-encoded JSON of the Robot node's body_mapping "
                         "(body_names + IR roles). Legacy RSI fallback for "
                         "robots whose joint table isn't declared per-format "
                         "yet — superseded by --unitport_joint_names_by_ir.")
parser.add_argument("--unitport_active_format", type=str, default="",
                    help="Active asset format the spec compiled against "
                         "(MJCF / USD / URDF). Stage 4: required for "
                         "deterministic clip→robot joint name lookup.")
parser.add_argument("--unitport_joint_names_by_ir", type=str, default="",
                    help="Base64-encoded JSON {ir_role: joint_name} for the "
                         "robot in --unitport_active_format. RSI uses this "
                         "to map clip IR roles → physical joint names "
                         "directly (no body+suffix heuristic).")
parser.add_argument("--unitport_init_pose_keyframe_name", type=str, default="home",
                    help="MJCF keyframe name (when --unitport_init_pose_mode=keyframe). "
                         "Resolved against the robot asset's MJCF (looked up via "
                         "--unitport_robot_sku). Joint angles from the keyframe's "
                         "qpos overwrite env_cfg.scene.robot.init_state.joint_pos.")
# Post-training eval (from eval_config canvas node)
parser.add_argument("--unitport_eval_episodes", type=int, default=0,
                    help="Number of eval episodes to run after training. "
                         "0 = skip eval phase entirely.")
parser.add_argument("--unitport_eval_success_threshold", type=float, default=0.8,
                    help="Episode reward / max_reward ratio counted as success "
                         "in eval_results.json.")
parser.add_argument("--unitport_eval_stochastic", action="store_true",
                    help="Use stochastic actions in eval. Default is "
                         "deterministic (mean of the Gaussian policy).")
parser.add_argument("--unitport_save_best_model", action="store_true",
                    help="During training, save the iter with the highest "
                         "reward as model_best.pt alongside the periodic "
                         "save_interval checkpoints.")
parser.add_argument("--unitport_eval_video_dir", type=str, default="",
                    help="If set, eval phase records video into this dir. "
                         "Metadata-only today — gym.wrappers.RecordVideo "
                         "wrapping is not wired in the eval branch yet.")

parser.add_argument(
    "--unitport_headed_experience", type=str, default="",
    help="Isaac Lab install root. When set and the run is headed "
         "(not --headless) and not --video, the launcher generates a minimal "
         "viewport-only experience under <root>/apps/ and selects it — "
         "avoiding the full Kit GUI omni.kit.menu.utils crash.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# ── Headed run: select a minimal viewport-only experience ──
# The default headed experience (isaaclab.python.kit) loads the full Kit GUI,
# which crashes during menu construction (omni.kit.menu.utils._build_menu →
# Windows access violation) and maps far more buffers through the scarce GPU
# BAR1 aperture. For a headed (not --headless), non-video run, generate a
# stripped viewport-only experience under the selected install's apps/ and
# point AppLauncher at it. Fail loud — never fall back to the crashing default.
if (not args_cli.headless and not args_cli.video
        and args_cli.unitport_headed_experience):
    from application.training.isaac_lab.launcher.headed_experience import (
        ensure_headed_experience,
    )
    args_cli.experience = ensure_headed_experience(
        _Path(args_cli.unitport_headed_experience)
    )
    print(
        f"[UnitPort][train] Headed run: using minimal experience "
        f"'{args_cli.experience}' (full Kit GUI skipped to avoid the "
        f"menu-build crash).",
        flush=True,
    )

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

        # Viewport render scale. 1.0 = render at native viewport resolution
        # (sharp). The earlier 0.5 (quarter pixel count) was too blurry for a
        # usable sanity window — rendering at half res and upscaling smears the
        # whole image, made worse by the fast-forward motion of training. This
        # only affects viewport cost, never physics/training; lower it (e.g.
        # 0.75) if the viewport stutters. Future UI: expose as a user knob.
        _carb.set("/app/viewport/defaultRenderScale", 1.0)
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
    # Path-traversal boundary (Plan finding P2-2). The launcher will
    # ``exec_module`` whatever Python file ``--unitport_config`` points
    # at, so the path must be inside PROJECTS_DIR (where the canvas
    # compiler emits compiled @configclass files). A path outside that
    # tree is either a developer typo or a hostile launcher invocation —
    # fail fast either way.
    #
    # Why we don't ``from unitport_sdk import Paths`` here (CLAUDE.md §1.8):
    # the SDK pulls PyQt6 at module-import time (``unitport_sdk.logger``),
    # which is NOT installed in Isaac Sim's bundled Python. The previous
    # code swallowed the ImportError and silently used
    # ``<RELEASE>/projects`` as PROJECTS_DIR — that path is the SOURCE
    # tree default, NOT the user's actual workspace. When the user
    # configured USER_CONFIG_DIR to an external drive (e.g. A:\UnitPort\
    # <workspace_uuid>\), every emitted config file lived under there,
    # but this boundary check compared against the source-tree default
    # and rejected EVERY file. Resolve PROJECTS_DIR with stdlib only,
    # mirroring ``unitport_sdk.sys.Paths._resolve_dynamic`` exactly so
    # the subprocess sees the same workspace the main app does.
    def _resolve_projects_dir() -> _Path:
        import configparser

        project_root = _SRC_ROOT.parent  # <RELEASE>/

        def _read_resource(ini_path: _Path, key: str) -> str:
            try:
                cp = configparser.ConfigParser(strict=False)
                if ini_path.exists():
                    cp.read(ini_path, encoding="utf-8")
                return cp.get("Resources", key, fallback="").strip()
            except Exception:
                return ""

        # 1) USER_CONFIG_DIR — bootstrap shim first, then system.ini.
        #    user.ini cannot determine its own location (chicken-egg),
        #    so user_config_dir is read from system.ini ONLY.
        shim_path = _Path.home() / ".unitport_paths.ini"
        user_cfg_raw = _read_resource(shim_path, "user_config_dir")
        if not user_cfg_raw:
            system_ini = _SRC_ROOT / "config" / "system.ini"
            user_cfg_raw = _read_resource(system_ini, "user_config_dir")
        if not user_cfg_raw:
            # CLAUDE.md §1.8: no silent fallback to a source-tree default.
            # USER_CONFIG_DIR has no factory default (parent CLAUDE.md §1.4).
            # If we reach here, the install wizard never ran — which means
            # the subprocess was launched from an unconfigured app state,
            # a real bug worth surfacing.
            raise RuntimeError(
                "[UnitPort] USER_CONFIG_DIR not configured — neither "
                f"{shim_path} nor system.ini[Resources].user_config_dir "
                "is set. The install wizard's DataDirectoryPage must run "
                "before any training subprocess is launched."
            )
        user_cfg = _Path(user_cfg_raw).expanduser()
        if not user_cfg.is_absolute():
            user_cfg = project_root / user_cfg
        # USER_INI overlay for projects_dir (user.ini wins over system.ini
        # for Tier-B paths, matching Paths._read_resource).
        user_ini = user_cfg / "user.ini"
        projects_raw = _read_resource(user_ini, "projects_dir") or _read_resource(
            _SRC_ROOT / "config" / "system.ini", "projects_dir"
        )
        if projects_raw:
            p = _Path(projects_raw).expanduser()
            return (p if p.is_absolute() else (project_root / p)).resolve()
        return (user_cfg / "projects").resolve()

    _cfg_resolved = _Path(args_cli.unitport_config).resolve()
    try:
        _projects_root = _resolve_projects_dir()
    except Exception as exc:
        # Loud failure — never substitute a wrong default (CLAUDE.md §1.8).
        print(
            f"[UnitPort] Failed to resolve PROJECTS_DIR in subprocess: {exc}",
            flush=True,
        )
        sys.exit(2)
    if not _cfg_resolved.is_relative_to(_projects_root):
        print(
            f"[UnitPort] Refusing to load unitport_config outside PROJECTS_DIR: "
            f"{_cfg_resolved} (root: {_projects_root})",
            flush=True,
        )
        sys.exit(2)

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
    # the Canvas. The main venv has already resolved the SKU through the
    # registry; we just (a) point the spawner at it and (b) run the real
    # USD parser now that we're inside the Isaac venv (per
    # AMP_design.yaml §7.risks.usd_parser_venv_dependency, path b).
    if args_cli.unitport_robot_usd:
        usd_path = args_cli.unitport_robot_usd
        sku = args_cli.unitport_robot_sku or "<unknown>"

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

        print(f"[UnitPort] RobotAsset override: sku={sku} usd={usd_path}",
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
            from application.service.robot_assets.usd_parser import (
                parse_usd, UsdParserUnavailable, UsdParseError,
            )
            try:
                usd_meta = parse_usd(usd_path)
                print(f"[UnitPort] USD parse OK: sku={sku} "
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
                print(f"[UnitPort][ABORT] USD validation failed for sku="
                      f"{sku}: {exc}", flush=True)
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

    # ``init_noise_std`` has historically been interpreted two ways across
    # UnitPort's training backends: the Isaac Lab config compiler treats it
    # as ``log_std`` and applies ``exp()`` (so ``-1.0`` → 0.368), while the
    # AMP vendored ActorCritic uses the value as raw std directly. The UI
    # preset defaults to ``-1.0`` (log-space). A raw negative std silently
    # flips exploration noise sign (Normal's validate_args is disabled for
    # speed), then the policy diverges into NaN a few iters later. To make
    # both paths agree with the UI convention, any non-positive value is
    # treated as log_std and exponentiated.
    import math as _math

    def _init_std(default: float) -> float:
        v = float(_p("init_noise_std", default))
        return _math.exp(v) if v <= 0.0 else v

    # ── std parameterization (rsl_rl) ──
    # rsl_rl's GaussianDistribution has two modes:
    #   * ``scalar`` (rsl_rl default) — std is a *direct* trainable parameter
    #     with no positivity constraint. PPO updates can drive it negative,
    #     after which ``Normal(mean, negative_std).sample()`` raises
    #     ``RuntimeError: normal expects all elements of std >= 0.0`` and
    #     training crashes mid-run. Hit on PPO-from-scratch + adaptive LR.
    #   * ``log`` — std is parameterized as ``exp(log_std_param)``, so the
    #     sampled std is always strictly positive regardless of update
    #     magnitude. Same expressive capacity, gradient-stable.
    # We force ``log`` for every UnitPort PPO/AMP run; there is no scenario
    # where ``scalar`` is preferable. Override via canvas if ever needed.
    _STD_TYPE = "log"

    # ── NaN guard on rsl_rl 5.x GaussianDistribution ──
    # std_type="log" guarantees positivity *only while log_std_param is finite*.
    # exp(NaN)=NaN, and Normal(mean, NaN).sample() raises
    # "RuntimeError: normal expects all elements of std >= 0.0". Stock rsl_rl
    # has no nan_to_num/clamp on mean or log_std_param, so a single Inf in a
    # reward/value loss can NaN-poison the policy hours into training. The AMP
    # path's vendored ActorCritic already guards this (amp/algorithms/
    # actor_critic.py:184-208); port the same pattern to the from-scratch PPO
    # path here. Idempotent — safe to re-enter on multiple launcher calls.
    if _has_mlp_model_cfg:
        from rsl_rl.modules.distribution import GaussianDistribution as _GD
        if not getattr(_GD, "_unitport_nan_guarded", False):
            from torch.distributions import Normal as _Normal
            _STD_MIN, _STD_MAX = 1e-3, 1.5
            _LOG_STD_MIN, _LOG_STD_MAX = _math.log(_STD_MIN), _math.log(_STD_MAX)
            _LOG_STD_FALLBACK = _math.log(0.3)

            def _guarded_update(self, mlp_output):
                mean = mlp_output
                if not torch.isfinite(mean).all():
                    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
                if self.std_type == "log":
                    with torch.no_grad():
                        self.log_std_param.data = torch.nan_to_num(
                            self.log_std_param.data,
                            nan=_LOG_STD_FALLBACK,
                            posinf=_LOG_STD_MAX,
                            neginf=_LOG_STD_MIN,
                        ).clamp(min=_LOG_STD_MIN, max=_LOG_STD_MAX)
                    std = torch.exp(self.log_std_param).expand_as(mean)
                else:
                    with torch.no_grad():
                        self.std_param.data = torch.nan_to_num(
                            self.std_param.data,
                            nan=0.3,
                            posinf=_STD_MAX,
                            neginf=_STD_MIN,
                        ).clamp(min=_STD_MIN, max=_STD_MAX)
                    std = self.std_param.expand_as(mean)
                self._distribution = _Normal(mean, std)

            _GD.update = _guarded_update
            _GD._unitport_nan_guarded = True

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
                    init_std=_init_std(1.0),
                    std_type=_STD_TYPE,
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
                init_noise_std=_init_std(1.0),
                noise_std_type=_STD_TYPE,
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

    # ── RSI (Reference State Initialization) ──
    # Override the robot's default joint positions with a reference motion
    # frame so the env resets into the reference pose instead of the stock
    # standing pose. Accelerates AMP convergence (APEX, Peng 2022).
    #
    # The clip's flat joint array is routed through the IR layer:
    #
    #   clip[i] → QUADRUPED_CLIP_IR_ROLES[i] → BodyIRMapper.get(role).body
    #           → body + "_joint" → env_cfg.scene.robot.init_state.joint_pos
    #
    # All robot-specific naming (Go2 "FL_hip" vs A1 "FL_hip" vs H1 knee) is
    # resolved by the Robot node's body_mapping, serialized into
    # --unitport_body_mapping. No positional-index hacks.
    _rsi_mode = str(getattr(args_cli, "unitport_init_pose_mode", "default")).strip().lower()
    _rsi_prob = float(getattr(args_cli, "unitport_rsi_prob", 1.0))
    _rsi_sample_mode = str(getattr(args_cli, "unitport_rsi_sample_mode", "frame_0")).strip().lower()
    _body_mapping_b64 = str(getattr(args_cli, "unitport_body_mapping", "") or "")
    _active_format = str(getattr(args_cli, "unitport_active_format", "") or "").upper()
    _joint_names_by_ir_b64 = str(getattr(args_cli, "unitport_joint_names_by_ir", "") or "")

    # Stage 4: register the per-format IR→joint-name table with the AMP
    # obs/RSI permutation resolver so robots whose USD naming diverges
    # from Unitree convention (Spot fl_hy/fl_kn, ...) build the canonical
    # permutation correctly. Safe no-op when the table is absent: the
    # resolver falls back to the legacy {leg}_{slot}_joint pattern.
    if _joint_names_by_ir_b64:
        try:
            import base64, json as _json
            from application.training.amp.obs_terms import (
                set_canonical_joint_perm_override,
            )
            _joint_names_by_ir_for_perm = _json.loads(
                base64.b64decode(_joint_names_by_ir_b64).decode("utf-8")
            )
            set_canonical_joint_perm_override(_joint_names_by_ir_for_perm)
            print(f"[UnitPort] canonical joint perm override registered "
                  f"({len(_joint_names_by_ir_for_perm)} IR roles, "
                  f"active_format={_active_format or '(none)'})")
        except Exception as exc:
            print(f"[UnitPort] failed to register joint perm override: {exc}")

    # RSI also benefits PPO-from-scratch runs whose canvas wires a
    # training_motion node with motion clips — used to be AMP_PPO-only,
    # which silently dropped the canvas selection for any PPO training
    # that wanted to warm-start episodes from a reference pose.
    if _rsi_mode == "reference_frame_0":
        _motion_paths = [
            p.strip() for p in str(args_cli.unitport_amp_motion_files).split(",")
            if p.strip()
        ]
        _applied = False
        if not _motion_paths:
            print("[UnitPort] RSI: no motion files — skipped.")
        elif not _joint_names_by_ir_b64 and not _body_mapping_b64:
            print("[UnitPort] RSI: no IR-role joint table — cannot route "
                  "clip joints through IR layer. Skipped.")
        else:
            try:
                import base64, json as _json
                from application.training.motion.loaders import get_loader
                from application.training.isaac_lab.reference_state_init import (
                    build_joint_pos_dict, RSIMappingError,
                )

                # Stage 4: prefer the per-format IR-role → joint name
                # table from --unitport_joint_names_by_ir. Legacy
                # --unitport_body_mapping is kept for backwards-compat
                # with subprocess invocations still pinned to the old
                # body+suffix path (mid-rollout safety).
                joint_names_by_ir = None
                if _joint_names_by_ir_b64:
                    joint_names_by_ir = _json.loads(
                        base64.b64decode(_joint_names_by_ir_b64).decode("utf-8")
                    )
                elif _body_mapping_b64:
                    # Legacy: derive {ir_role: joint_name} by body+"_joint"
                    # suffix from the old body_mapping payload.
                    from application.training.body_ir import BodyIRMapper
                    body_mapping_dict = _json.loads(
                        base64.b64decode(_body_mapping_b64).decode("utf-8")
                    )
                    mapper = BodyIRMapper.from_dict(body_mapping_dict)
                    joint_names_by_ir = {}
                    for role in mapper.roles:
                        if role.body:
                            joint_names_by_ir[role.role_id] = f"{role.body}_joint"
                    print("[UnitPort] RSI: using legacy body+suffix joint-name "
                          "derivation (no --unitport_joint_names_by_ir provided)")

                _loader = get_loader("amp_legged_gym")
                # loader.load returns a ReferenceMotionContract; RSI
                # seeds env init_state from the wrapped MotionClip's
                # frame 0. target_sku / target_family are required
                # keyword-only; both come from the subprocess CLI
                # (canvas SKU + primary family slug travelled across
                # via --unitport_robot_sku / --unitport_robot_family).
                _first_clip = _loader.load(
                    _motion_paths[0],
                    target_sku=str(args_cli.unitport_robot_sku),
                    target_family=str(args_cli.unitport_robot_family),
                ).clip

                _jp = getattr(_first_clip, "joint_pos", None) if _first_clip else None
                if _jp is None:
                    print("[UnitPort] RSI: clip has no joint_pos, skipped.")
                else:
                    _frame = _jp[0]  # (num_joints,)
                    # Clip's own ir_roles take precedence over the global
                    # default (lets a non-quadruped clip carry its own
                    # layout in metadata['ir_roles']).
                    clip_roles = getattr(_first_clip, "ir_roles", None)
                    joint_pos_dict = build_joint_pos_dict(
                        _frame, joint_names_by_ir, clip_ir_roles=clip_roles,
                    )
                    # Replace the regex-keyed default with a per-joint dict.
                    # Specific joint names take precedence over Isaac Lab's
                    # built-in regex patterns at articulation init.
                    env_cfg.scene.robot.init_state.joint_pos = joint_pos_dict
                    print(f"[UnitPort] RSI via IR (format={_active_format or 'auto'}): "
                          f"{len(joint_pos_dict)} joints set from clip frame 0 "
                          f"(prob={_rsi_prob}, sample={_rsi_sample_mode})")
                    for jn, val in list(joint_pos_dict.items())[:6]:
                        print(f"[UnitPort] RSI:   {jn} = {val:.4f}")
                    if len(joint_pos_dict) > 6:
                        print(f"[UnitPort] RSI:   ... ({len(joint_pos_dict) - 6} more)")
                    _applied = True
            except RSIMappingError as exc:
                print(f"[UnitPort] RSI: IR mapping failed — {exc}")
            except Exception as exc:
                print(f"[UnitPort] RSI: unexpected error — {exc}. Skipped.")

        if not _applied:
            print("[UnitPort] RSI: not applied — env will use stock init pose.")

    elif _rsi_mode == "keyframe":
        # MJCF keyframe init_pose: resolve robot asset → MJCF file →
        # mujoco loads keyframe.qpos → write joint_pos dict into env_cfg.
        # Skips the 7-element root (3 pos + 4 quat) and keeps only joint
        # angles. Assumes MJCF joint names match the USD articulation
        # joint names (true for menagerie assets where USD is derived
        # from the same MJCF).
        _kf_name = str(getattr(args_cli, "unitport_init_pose_keyframe_name", "home"))
        _kf_sku = str(args_cli.unitport_robot_sku or "")
        _kf_applied = False
        if not _kf_sku:
            print("[UnitPort] keyframe init_pose: no robot_sku — skipped.")
        else:
            try:
                import mujoco
                from application.service.robot_assets import get_robot_asset_service
                from registers.robots import resolve_id as _resolve_robot_id
                _sku = _resolve_robot_id(_kf_sku) or _kf_sku
                _asset = get_robot_asset_service().resolve(_sku)
                _mjcf_path = str(_asset.mjcf_path) if _asset and _asset.mjcf_path else ""
                if not _mjcf_path or not os.path.isfile(_mjcf_path):
                    print(
                        f"[UnitPort] keyframe init_pose: robot {_sku!r} has no "
                        f"MJCF (mjcf_path={_mjcf_path!r}); skipped.",
                        flush=True,
                    )
                else:
                    _mj_model = mujoco.MjModel.from_xml_path(_mjcf_path)
                    _key_id = mujoco.mj_name2id(
                        _mj_model, mujoco.mjtObj.mjOBJ_KEY, _kf_name,
                    )
                    if _key_id < 0:
                        print(
                            f"[UnitPort] keyframe init_pose: keyframe "
                            f"{_kf_name!r} not in {_mjcf_path}; skipped.",
                            flush=True,
                        )
                    else:
                        _qpos = _mj_model.key_qpos[_key_id]
                        _joint_pos_dict = {}
                        for _ji in range(_mj_model.njnt):
                            _jname = mujoco.mj_id2name(
                                _mj_model, mujoco.mjtObj.mjOBJ_JOINT, _ji,
                            )
                            _jtype = int(_mj_model.jnt_type[_ji])
                            if _jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                                continue
                            _qpos_addr = int(_mj_model.jnt_qposadr[_ji])
                            _joint_pos_dict[str(_jname)] = float(_qpos[_qpos_addr])
                        if _joint_pos_dict:
                            env_cfg.scene.robot.init_state.joint_pos = _joint_pos_dict
                            print(
                                f"[UnitPort] keyframe init_pose: loaded "
                                f"{_kf_name!r} ({len(_joint_pos_dict)} joints) "
                                f"from {_mjcf_path}",
                                flush=True,
                            )
                            _kf_applied = True
            except ImportError:
                print(
                    "[UnitPort] keyframe init_pose: mujoco not installed in "
                    "Isaac venv; skipped.",
                    flush=True,
                )
            except Exception as _kf_exc:
                print(
                    f"[UnitPort] keyframe init_pose: failed ({_kf_exc}); "
                    "skipped.",
                    flush=True,
                )
        if not _kf_applied:
            print("[UnitPort] keyframe init_pose: not applied — using stock init pose.")

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
        from application.training.amp.data_provider import (
            build_amp_data_from_files,
            build_rsi_pool,
        )
        from application.training.amp.mdp_events import register_rsi_pool
        from application.training.amp.wrappers.amp_rsl_rl_vec_env_wrapper import (
            AmpRslRlVecEnvWrapper,
        )
        from application.training.amp.runners.amp_on_policy_runner import (
            AMPOnPolicyRunner,
        )
        from application.training.amp.joint_alignment import (
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

        # Parse canvas-side label overrides
        _label_overrides = {}
        _raw_labels = str(getattr(args_cli, "unitport_amp_task_labels", "") or "")
        if _raw_labels:
            try:
                _label_overrides = json.loads(_raw_labels)
                if not isinstance(_label_overrides, dict):
                    _label_overrides = {}
            except Exception:
                _label_overrides = {}

        # Resolve the AMP obs term list ONCE here from the robot's primary
        # family (--unitport_robot_family carries families[0]). The same
        # list flows into both the motion-data builder (expert side, via
        # MotionAmpData) and the env-side AMP obs extractor below, so the
        # discriminator's expert and policy distributions are guaranteed
        # to share field order. Canvas-side override
        # (--unitport_amp_obs_fields) wins when non-empty.
        from registers.amp_obs_templates import get_obs_template
        _family = str(args_cli.unitport_robot_family or "")
        if not _family:
            print(
                "[UnitPort][ABORT] --unitport_robot_family is empty; "
                "cannot resolve the AMP obs template / preflight perm. "
                "Pass the robot family slug (e.g. 'quadruped' for Go2, "
                "'humanoid' for H1) so registers.amp_obs_templates can "
                "return the canonical term list.",
                flush=True,
            )
            sys.exit(4)
        _user_obs_fields = str(getattr(args_cli, "unitport_amp_obs_fields", "") or "").strip()
        if _user_obs_fields:
            amp_term_names = [
                t.strip() for t in _user_obs_fields.split(",") if t.strip()
            ]
            print(
                f"[UnitPort][AMP] obs fields overridden by canvas: {amp_term_names}",
                flush=True,
            )
        else:
            amp_term_names = list(
                get_obs_template(_family).template_terms
            )
            print(
                f"[UnitPort][AMP] obs fields resolved from "
                f"registers.amp_obs_templates[{_family!r}]: {amp_term_names}",
                flush=True,
            )

        amp_data = build_amp_data_from_files(
            motion_paths,
            transition_dt=float(args_cli.unitport_amp_transition_dt),
            robot_sku=str(args_cli.unitport_robot_sku),
            robot_family=str(args_cli.unitport_robot_family),
            term_names=amp_term_names,
            device=device,
            format_id="amp_legged_gym",
            task_filter=str(getattr(args_cli, "unitport_amp_task_filter", "")),
            task_label_overrides=_label_overrides or None,
        )
        print(
            f"[UnitPort] Loaded {len(motion_paths)} AMP motion clip(s), "
            f"observation_dim={amp_data.observation_dim}",
            flush=True,
        )

        # ── RSI (Reference-State Initialization) pool ──
        # AMP trains much faster when episodes start from a random expert
        # frame (pose + joint state + velocity) instead of a fixed default
        # pose plus uniform noise. The compiler already emits a
        # reset_from_reference_motion EventTerm wired to pool_id "default";
        # our job here is to populate that pool before env construction.
        # getattr with defaults keeps this safe if the compiler-side CLI
        # args land in a later change.
        _rsi_prob = float(getattr(args_cli, "unitport_amp_rsi_prob", 0.0) or 0.0)
        _rsi_n_frames = int(getattr(args_cli, "unitport_amp_rsi_n_frames", 20000) or 20000)
        if _rsi_prob > 0.0:
            _random_start_phase = not bool(
                getattr(args_cli, "unitport_amp_no_random_start_phase", False)
            )
            _rsi_pool = build_rsi_pool(
                amp_data.clips,
                n_frames=_rsi_n_frames,
                random_start_phase=_random_start_phase,
            )
            register_rsi_pool("default", _rsi_pool)
            print(
                f"[UnitPort][AMP] RSI pool registered: {_rsi_n_frames} frames "
                f"from {len(amp_data.clips)} clips (rsi_prob={_rsi_prob})",
                flush=True,
            )
        else:
            print(
                "[UnitPort][AMP] RSI disabled (rsi_prob=0); using default reset events.",
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
        # ``application.training.amp.obs_terms``. Previously the two
        # sides had independent hand-written concatenations that could
        # silently drift in field order, frame, or units — the most
        # common silent killer for AMP training. The registry entries
        # live at one file and either both producers stay in sync or
        # neither compiles.
        from application.training.amp.obs_terms import (
            compute_amp_obs_from_env,
            preflight_canonical_mapping,
        )

        # amp_term_names was resolved above (single dispatch point for
        # both motion-data builder and env extractor). Canvas discriminator
        # .amp_obs_fields can override per-run via --unitport_amp_obs_fields
        # (handled in the resolution block above).
        # auto_inject_ref_obs is a discriminator-level flag that, when
        # disabled, would skip the auto-injection of reference clip obs
        # into the discriminator's input. The current AMP wrapper already
        # builds both env-side and motion-side obs from the same field
        # list, so "auto-inject" is implicit; disabling it would require
        # a wrapper-side branch we have not built yet. Log so the user
        # sees the flag was received but is currently inert.
        if getattr(args_cli, "unitport_amp_no_auto_inject_ref_obs", False):
            print(
                "[UnitPort][AMP] WARNING: --unitport_amp_no_auto_inject_ref_obs "
                "received but currently inert (AMP wrapper has no opt-out "
                "branch for reference-obs injection). Effect: identical to "
                "default. Track in canvas if this needs to actually toggle.",
                flush=True,
            )

        def _amp_obs_extractor(wrapper):
            return compute_amp_obs_from_env(amp_term_names, wrapper)

        # Field order alignment (hard-mitigation for amp_obs_dim_drift).
        # Both the motion data builder (above) and the env extractor here
        # consume the same amp_term_names; the two field tuples are equal
        # by construction. Keeping verify_alignment as a belt-and-braces
        # check shields against a future refactor that re-introduces two
        # independent term-list sources.
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

        # Preflight: resolve canonical joint permutation + foot ids eagerly
        # so any naming mismatch (new robot brand, renamed URDF, a newly
        # registered per-DoF term that forgot to apply the permutation)
        # aborts training at step 0 with a clear error — instead of
        # silently degenerating into disc_acc=1.00 from iter 0. Also
        # captures the asset→canonical mapping in amp_alignment.json for
        # post-mortem.
        try:
            canonical_mapping = preflight_canonical_mapping(
                env, family=_family,
            )
            dump_alignment_report(
                log_dir, env_fields, clip_fields, ok=True,
                canonical_mapping=canonical_mapping,
            )
            print(
                "[UnitPort] amp_obs canonical mapping resolved: "
                f"family={canonical_mapping['family']!r} "
                f"joint_perm={canonical_mapping['canonical_joint_perm']} "
                f"foot_names={canonical_mapping['canonical_foot_names']}",
                flush=True,
            )
        except Exception as exc:
            dump_alignment_report(
                log_dir, env_fields, clip_fields, ok=False,
                error=f"canonical_mapping_failed: {exc}",
            )
            print(
                f"[UnitPort][ABORT] amp_obs canonical mapping failed: {exc}",
                flush=True,
            )
            sys.exit(5)

        # Build the AMPOnPolicyRunner's nested cfg dict. Fields come
        # from AMPConfig (via agent_dict['amp_*']) plus the PPO bits
        # the vendored runner shares with OnPolicyRunner.
        # ── Decode stage schedule if provided ──
        # H0 contract: the payload is always a serialized
        # StageScheduleConfig dict (schema_version + stages + ...). Decode
        # failures and shape mismatches are fail-loud — the previous
        # silent ``except: stage_schedule_dict = None`` masked corrupt
        # base64 / JSON as "no schedule wired" and routed to the
        # single-stage path. The H0 dict guard re-checks schema_version
        # and per-stage defaults before injection.
        _stage_schedule_dict = None
        if getattr(args_cli, "unitport_stage_schedule", ""):
            import base64 as _b64
            from application.training.training_spec import (
                validate_stage_schedule_dict_h0,
            )
            _decoded = _b64.b64decode(
                args_cli.unitport_stage_schedule
            ).decode("utf-8")
            _stage_schedule_dict = json.loads(_decoded)
            if not isinstance(_stage_schedule_dict, dict):
                raise ValueError(
                    f"--unitport_stage_schedule decoded payload must be a "
                    f"dict (serialized StageScheduleConfig), got "
                    f"{type(_stage_schedule_dict).__name__}."
                )
            validate_stage_schedule_dict_h0(_stage_schedule_dict)

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
                # P2 circuit-breakers — discriminator + per-step reward clamp.
                "amp_discr_logit_clamp_max": float(
                    args_cli.unitport_amp_disc_logit_clamp_max
                ),
                "amp_reward_clamp_per_step": float(
                    args_cli.unitport_amp_reward_clamp_per_step
                ),
                # P0.3 — user-edited discriminator function bodies, if any.
                "amp_disc_overrides_file": str(
                    args_cli.unitport_amp_disc_overrides_file or ""
                ),
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
                "init_noise_std": _init_std(0.368),
                # P2 circuit-breaker — std clamp passed via **policy_cfg
                # to ActorCritic.__init__ (only the AMP path's vendored
                # ActorCritic accepts these; rsl_rl 2.x ignores via **kwargs).
                "std_clamp_max": float(args_cli.unitport_amp_policy_std_clamp_max),
            },
        }

        # ── Inject stage schedule into runner config ──
        # The decoder above guarantees ``_stage_schedule_dict`` is a dict
        # (legacy list shape now fail-loud raises). The H0 guard already
        # confirmed schema_version + per-stage defaults.
        if _stage_schedule_dict is not None:
            _ckpt_strat = getattr(
                args_cli, "unitport_stage_checkpoint_strategy", "both"
            )
            amp_runner_cfg["stage_schedule"] = _stage_schedule_dict
            if "checkpoint_strategy" not in _stage_schedule_dict:
                amp_runner_cfg["stage_schedule"]["checkpoint_strategy"] = _ckpt_strat

            # Loud confirmation so the user can verify staged dispatch
            # before the runner starts. Silent schedule loss (e.g. canvas
            # missing a StageSwitchNode edge) is easy to miss.
            _sched_for_log = amp_runner_cfg.get("stage_schedule", {})
            _stages_for_log = _sched_for_log.get("stages", []) if isinstance(_sched_for_log, dict) else []
            _ver = _sched_for_log.get("version", 1) if isinstance(_sched_for_log, dict) else 1
            print(
                f"[UnitPort][STAGE] schedule DETECTED: version={_ver}, "
                f"{len(_stages_for_log)} stage(s), "
                f"total_iters={_sched_for_log.get('global_max_iterations', '?')}",
                flush=True,
            )
            for _i, _s in enumerate(_stages_for_log):
                _tp = _s.get("train_pipe", {}) if isinstance(_s, dict) else {}
                print(
                    f"[UnitPort][STAGE]   stage[{_i}] port=stage_{_s.get('stage_index', '?')} "
                    f"mode={_tp.get('training_mode', '?')} "
                    f"iters={_s.get('iterations', '?')} "
                    f"lerp={_tp.get('task_reward_lerp', '?')} "
                    f"disc_lr={_tp.get('disc_lr', '?')} "
                    f"disc_gp={_tp.get('disc_grad_penalty', '?')}",
                    flush=True,
                )
        else:
            print(
                "[UnitPort][STAGE] NO stage_schedule — running single-stage. "
                "If you wired a StageSwitchNode, check --unitport_stage_schedule.",
                flush=True,
            )

        # Decode optional task_items payload for command-conditioned AMP.
        # Base64-wrapped JSON — the CLI arg carrier avoids shell-escape
        # issues with the raw braces/quotes inside the dicts.
        _task_items_list = None
        _ti_raw = str(getattr(args_cli, "unitport_amp_task_items", "") or "").strip()
        if _ti_raw:
            import base64 as _b64
            try:
                _ti_decoded = _b64.b64decode(_ti_raw).decode("utf-8")
                _ti_json = json.loads(_ti_decoded)
                if isinstance(_ti_json, list) and _ti_json:
                    from application.training.command_schema import CommandSchema
                    _cs = CommandSchema.from_dict({"task_items": _ti_json})
                    _task_items_list = list(_cs.task_items) or None
                    if _task_items_list:
                        print(
                            f"[UnitPort][AMP] Decoded {len(_task_items_list)} "
                            f"task_item(s) for command-conditioned sampling: "
                            f"{[t.id for t in _task_items_list]}",
                            flush=True,
                        )
            except Exception as _exc:
                print(
                    f"[UnitPort][AMP][WARN] failed to decode "
                    f"--unitport_amp_task_items ({_exc}); falling back to "
                    f"uniform sampling.",
                    flush=True,
                )
                _task_items_list = None

        # UnitPort curriculum (cold-start ramps) — compiled into the
        # generated config file as a module-level ``UNITPORT_CURRICULUM``
        # dict. See ``IsaacLabConfigCompiler._unitport_curriculum_cfg``.
        # Absent/empty dict = no active curriculum.
        _unitport_curriculum = {}
        if '_cfg_mod' in globals():
            _unitport_curriculum = getattr(_cfg_mod, "UNITPORT_CURRICULUM", {}) or {}
        if _unitport_curriculum:
            _enabled_dims = sorted(_unitport_curriculum.keys())
            print(
                f"[UnitPort][Curriculum] Active dimensions: {_enabled_dims}",
                flush=True,
            )

        runner = AMPOnPolicyRunner(
            env,
            train_cfg=amp_runner_cfg,
            amp_data=amp_data,
            log_dir=log_dir,
            device=device,
            task_items=_task_items_list,
            unitport_curriculum=_unitport_curriculum,
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
            from application.training.run_meta import (
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
                    "amp_obs_terms": list(amp_term_names),
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
                # training_motion playback knobs — metadata only, no
                # current runtime branch. Recorded so the deploy stack /
                # post-mortem tooling knows the user's intent even though
                # the data_provider treats all clips uniformly today.
                "motion_playback": {
                    "phase_mode": str(getattr(args_cli, "unitport_amp_phase_mode", "loop")),
                    "random_start_phase": not bool(
                        getattr(args_cli, "unitport_amp_no_random_start_phase", False)
                    ),
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
                # resume_path is user-selected via the GUI's checkpoint picker
                # → untrusted-possible pickle. Funnel through the SHA-256 /
                # consent gate. Plan P1-1.
                from application.training._ckpt_safety import load_checkpoint_safely
                loaded = load_checkpoint_safely(resume_path)
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
    # AMP_PPO path: stage_schedule (when present) was injected into
    # amp_runner_cfg["stage_schedule"] above and the vendored
    # AMPOnPolicyRunner dispatches stages internally. The PPO path uses
    # stock rsl_rl OnPolicyRunner which has no stage awareness, so we
    # decode the schedule here and drive stages with an explicit loop
    # that mutates runner.alg between learn() calls.
    _ppo_stages = []
    if args_cli.unitport_algorithm != "AMP_PPO" and getattr(
        args_cli, "unitport_stage_schedule", ""
    ):
        import base64 as _b64
        try:
            _decoded = _b64.b64decode(
                args_cli.unitport_stage_schedule
            ).decode("utf-8")
            _ppo_sched = json.loads(_decoded)
            if isinstance(_ppo_sched, list):
                _ppo_stages = _ppo_sched
            elif isinstance(_ppo_sched, dict):
                _ppo_stages = _ppo_sched.get("stages") or []
        except Exception as _e:
            print(
                f"[UnitPort][STAGE/PPO] decode failed: {_e}; "
                f"falling back to single-stage training.",
                flush=True,
            )

    if _ppo_stages:
        print(
            f"[UnitPort][STAGE/PPO] {len(_ppo_stages)} stage(s) detected; "
            f"running explicit stage loop.",
            flush=True,
        )
        for _i, _stage in enumerate(_ppo_stages):
            if not isinstance(_stage, dict):
                continue
            # CLAUDE.md §1.8: distinguish "iterations field missing" (config
            # bug, raise) from "iterations explicitly 0" (intentional disable
            # sentinel, skip with WARN). The previous chained `get(..., 0)
            # or 0` collapsed both into "skip silently", masking
            # curriculum_schedule_json wiring bugs that produced 0-iter
            # stages by accident.
            if "iterations" not in _stage:
                raise ValueError(
                    f"[UnitPort][STAGE/PPO] stage[{_i}] is missing the "
                    f"'iterations' field — refusing to skip silently. Fix "
                    f"the curriculum_schedule_json on the canvas."
                )
            try:
                _iters = int(_stage["iterations"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[UnitPort][STAGE/PPO] stage[{_i}].iterations="
                    f"{_stage['iterations']!r} is not an integer."
                ) from exc
            if _iters < 0:
                raise ValueError(
                    f"[UnitPort][STAGE/PPO] stage[{_i}].iterations={_iters} "
                    f"is negative — invalid."
                )
            if _iters == 0:
                print(
                    f"[UnitPort][STAGE/PPO] stage[{_i}] iterations=0 "
                    f"(explicit disable sentinel), skipped",
                    flush=True,
                )
                continue
            _tp = _stage.get("train_pipe", {}) if isinstance(_stage, dict) else {}
            # Apply PPO overrides on runner.alg (rsl_rl PPO class). Only
            # the handful of fields that have a direct mutable counterpart
            # are touched here; anything else (num_minibatches, schedule)
            # is structurally baked at runner construction and ignored.
            if isinstance(_tp, dict):
                if "learning_rate" in _tp:
                    try:
                        new_lr = float(_tp["learning_rate"])
                        for _pg in runner.alg.optimizer.param_groups:
                            _pg["lr"] = new_lr
                        if hasattr(runner.alg, "learning_rate"):
                            runner.alg.learning_rate = new_lr
                        print(
                            f"[UnitPort][STAGE/PPO] stage[{_i}] learning_rate -> {new_lr}",
                            flush=True,
                        )
                    except (TypeError, ValueError):
                        pass
                if "entropy_coef" in _tp and hasattr(runner.alg, "entropy_coef"):
                    try:
                        runner.alg.entropy_coef = float(_tp["entropy_coef"])
                        print(
                            f"[UnitPort][STAGE/PPO] stage[{_i}] entropy_coef -> "
                            f"{runner.alg.entropy_coef}",
                            flush=True,
                        )
                    except (TypeError, ValueError):
                        pass
                if "clip_param" in _tp and hasattr(runner.alg, "clip_param"):
                    try:
                        runner.alg.clip_param = float(_tp["clip_param"])
                        print(
                            f"[UnitPort][STAGE/PPO] stage[{_i}] clip_param -> "
                            f"{runner.alg.clip_param}",
                            flush=True,
                        )
                    except (TypeError, ValueError):
                        pass
            print(
                f"[UnitPort][STAGE/PPO] stage[{_i}] training {_iters} iter(s)",
                flush=True,
            )
            runner.learn(
                num_learning_iterations=_iters,
                init_at_random_ep_len=(_i == 0),
            )
    else:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )

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

    # ── Post-training eval (eval_config canvas node) ──
    # Runs N deterministic episodes on the *training* env (we do not tear
    # down + rebuild — Isaac Sim shutdown is the canonical hang source on
    # Windows, and the existing 4096-env training scene is more than
    # adequate to collect a representative sample). Writes eval_results.json
    # next to the run dir so the bundle finalizer can include it. Skipped
    # entirely when eval_episodes==0 (the canvas had no eval_config node).
    _eval_target = int(getattr(args_cli, "unitport_eval_episodes", 0) or 0)
    if _eval_target > 0:
        print(
            f"[UnitPort][EVAL] starting: target={_eval_target} episodes "
            f"deterministic={not args_cli.unitport_eval_stochastic}",
            flush=True,
        )
        # Sentinel — runner shape mismatch (newer rsl_rl, SB3 wrappers,
        # custom AMP runner without an ``.alg.actor_critic`` attr) should
        # silently skip eval instead of surfacing as a "(non-fatal) failed"
        # noise line. Training output (checkpoints + save_best_model) is
        # unaffected, so this is genuinely a routing decision, not a fault.
        class _EvalSkipped(RuntimeError):
            pass

        try:
            _actor_critic = getattr(runner, "alg", None)
            _actor_critic = getattr(_actor_critic, "actor_critic", None) if _actor_critic is not None else None
            if _actor_critic is None:
                raise _EvalSkipped(
                    f"runner does not expose .alg.actor_critic "
                    f"(runner type={type(runner).__name__})"
                )
            _actor_critic.eval()
            _obs, _ = env.get_observations()
            _n_envs = _obs.shape[0]
            _ep_reward = torch.zeros(_n_envs, device=device)
            _ep_length = torch.zeros(_n_envs, device=device, dtype=torch.long)
            _completed_rewards = []
            _completed_lengths = []
            _max_eval_steps = max(500, _eval_target * 50 // max(1, _n_envs) + 200)
            with torch.no_grad():
                for _step in range(_max_eval_steps):
                    if args_cli.unitport_eval_stochastic:
                        _actions = _actor_critic.act(_obs)
                    else:
                        _actions = _actor_critic.act_inference(_obs)
                    _obs, _rew, _dones, _info = env.step(_actions)
                    _ep_reward = _ep_reward + _rew
                    _ep_length = _ep_length + 1
                    if _dones.any():
                        _done_idx = _dones.nonzero(as_tuple=False).squeeze(-1)
                        for _i in _done_idx.tolist():
                            _completed_rewards.append(float(_ep_reward[_i].item()))
                            _completed_lengths.append(int(_ep_length[_i].item()))
                        _ep_reward[_done_idx] = 0.0
                        _ep_length[_done_idx] = 0
                        if len(_completed_rewards) >= _eval_target:
                            break
            _completed_rewards = _completed_rewards[:_eval_target]
            _completed_lengths = _completed_lengths[:_eval_target]
            if _completed_rewards:
                _mean_r = sum(_completed_rewards) / len(_completed_rewards)
                _mean_l = sum(_completed_lengths) / len(_completed_lengths)
                _max_r = max(_completed_rewards)
                _min_r = min(_completed_rewards)
                _success_thresh = float(args_cli.unitport_eval_success_threshold) * (_max_r if _max_r > 0 else 1.0)
                _success_rate = sum(1 for _r in _completed_rewards if _r >= _success_thresh) / len(_completed_rewards)
                _eval_results = {
                    "eval_episodes": len(_completed_rewards),
                    "mean_reward": _mean_r,
                    "max_reward": _max_r,
                    "min_reward": _min_r,
                    "mean_episode_length": _mean_l,
                    "success_threshold_ratio": float(args_cli.unitport_eval_success_threshold),
                    "success_rate": _success_rate,
                    "deterministic": not bool(args_cli.unitport_eval_stochastic),
                    "video_dir": str(getattr(args_cli, "unitport_eval_video_dir", "") or ""),
                }
                _eval_path = os.path.join(log_dir, "eval_results.json")
                with open(_eval_path, "w", encoding="utf-8") as _fh:
                    json.dump(_eval_results, _fh, indent=2)
                print(
                    f"[UnitPort][EVAL] done: mean_reward={_mean_r:.3f} "
                    f"mean_ep_len={_mean_l:.1f} success_rate={_success_rate:.2f} "
                    f"→ {_eval_path}",
                    flush=True,
                )
            else:
                print(
                    "[UnitPort][EVAL] no episodes completed within "
                    f"{_max_eval_steps} steps; eval_results.json not written.",
                    flush=True,
                )
        except _EvalSkipped as _skip_exc:
            print(
                f"[UnitPort][EVAL] skipped: {_skip_exc}; training output "
                f"is still valid.",
                flush=True,
            )
        except Exception as _eval_exc:
            print(f"[UnitPort][EVAL] failed (non-fatal): {_eval_exc}", flush=True)

    # ── save_best_model: post-hoc best-checkpoint snapshot ──
    # rsl_rl's OnPolicyRunner has no per-iter best-tracking callback API,
    # so the closest we can do post-hoc is identify the highest-iter
    # checkpoint (under save_interval cadence, training reward is usually
    # monotonic enough that the last checkpoint is also the best). Copy
    # it to model_best.pt so the bundle exporter / play launcher can
    # reference it by name regardless of how many model_<N>.pt files
    # accumulated. When save_best_model is False, do nothing.
    if getattr(args_cli, "unitport_save_best_model", False):
        try:
            import shutil
            _ckpt_dir = Path(log_dir)
            _candidates = sorted(
                _ckpt_dir.glob("model_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else -1,
                reverse=True,
            )
            if _candidates:
                _best = _candidates[0]
                _best_dst = _ckpt_dir / "model_best.pt"
                shutil.copy2(str(_best), str(_best_dst))
                print(
                    f"[UnitPort][EVAL] save_best_model: {_best.name} → "
                    f"model_best.pt (post-hoc; last-checkpoint heuristic)",
                    flush=True,
                )
            else:
                print(
                    "[UnitPort][EVAL] save_best_model: no model_*.pt found; "
                    "nothing to snapshot.",
                    flush=True,
                )
        except Exception as _exc:
            print(f"[UnitPort][EVAL] save_best_model failed: {_exc}", flush=True)

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
