"""UnitPort Isaac Lab play launcher.

Replays a trained policy in the Isaac Sim viewport. Mirrors Isaac Lab's
``rsl_rl/play.py`` but accepts an **absolute** ``--load_run`` path instead
of a subdirectory name relative to ``<isaac_root>/logs/rsl_rl/<exp>/``.

Why this exists
---------------
Isaac Lab's stock ``play.py`` calls
``get_checkpoint_path(log_root_path, agent_cfg.load_run, ...)`` which:
  1. Computes ``log_root_path = <isaac_root>/logs/rsl_rl/<experiment_name>``
     from the **task's own** agent_cfg, ignoring whatever the caller passed.
  2. ``os.scandir`` that directory and uses ``agent_cfg.load_run`` as a
     **regex pattern** to filter the subdirectory names — not as an
     absolute path.

UnitPort writes its checkpoints under ``projects/<slug>/training/runs/<run>/``
which lives outside Isaac Lab's tree, so the stock launcher can never find
them. This launcher bypasses ``get_checkpoint_path`` entirely and feeds the
absolute ``.pt`` path directly into ``OnPolicyRunner.load()``.

Usage (called by IsaacLabBackend / vis_check_runner, not directly)::

    python il_play_launcher.py \\
        --task Isaac-Velocity-Rough-Unitree-Go2-v0 \\
        --load_run "D:/.../runs/isaac_training__20260408T110524Z" \\
        --num_envs 50
"""

from __future__ import annotations

import argparse
import os
import sys

# Same EULA / precheck / h5py-first dance as il_train_launcher.py.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

try:
    import h5py  # noqa: F401  — side-effect import only
except Exception:
    pass

from isaaclab.app import AppLauncher

# ── 1. CLI args ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="UnitPort Isaac Lab Play Launcher")
parser.add_argument("--task", type=str, required=True,
                    help="Gym task ID, e.g. Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_run", type=str, required=True,
                    help="Absolute path to the run directory containing model_*.pt "
                         "(or directly to a .pt file).")
parser.add_argument("--checkpoint", type=str, default="",
                    help="Optional checkpoint file name inside --load_run. "
                         "When omitted the newest model_*.pt is used.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--max_play_steps", type=int, default=3000,
                    help="Hard upper bound on the playback loop. At 50Hz "
                         "this is ~60s. Set to 0 to disable (run until "
                         "the user closes the viewport).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Hydra args are forwarded to the env_cfg / agent_cfg defaults via
# Isaac Lab's hydra wrapper. We don't use Hydra in this launcher (we load
# the checkpoint directly), so just drop them from sys.argv to keep the
# AppLauncher and gym.make code from getting confused.
sys.argv = [sys.argv[0]] + hydra_args

# ── 2. Start Isaac Sim ───────────────────────────────────────────────────

# Force rendering on — this is the review path, the user wants to SEE the
# robot. AppLauncher reads --headless from args_cli, so explicitly clear it
# even if it leaked in via the parent CLI.
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2.5. Tame RTX auto-exposure ──────────────────────────────────────────
# With num_envs=1 the scene is sparse enough that Isaac Sim's histogram-
# based auto-exposure can't find a stable convergence point and drifts
# wide-open, blowing the entire viewport to white. Disable auto-exposure
# and pin a sensible fixed-exposure tonemap so the scene matches what the
# env was rendered with at training time. Must run AFTER AppLauncher so
# carb is initialised, but BEFORE gym.make / scene spawn so the first
# frame already uses the new settings.
try:
    import carb
    _carb = carb.settings.get_settings()
    # 1 = ACES filmic tonemapper (matches Isaac Lab training default)
    _carb.set("/rtx/post/tonemap/op", 1)
    # Kill the auto-exposure histogram entirely
    _carb.set("/rtx/post/histogram/enabled", False)
    _carb.set("/rtx/post/histogram/autoExposure", False)
    # Pin a neutral fixed exposure (ISO 100, 1/100s, f/8 — daylight outdoors)
    _carb.set("/rtx/post/tonemap/filmIso", 100.0)
    _carb.set("/rtx/post/tonemap/cameraShutter", 100.0)
    _carb.set("/rtx/post/tonemap/fNumber", 8.0)
    # Some RTX builds use a separate "auto exposure" toggle under the
    # post-processing path — set both to be safe.
    _carb.set("/rtx/post/aa/op", 3)  # DLAA, sharper edges than default TAA
    print("[UnitPort][play] RTX auto-exposure disabled; using fixed "
          "ISO100 / 1/100s / f8 tonemap.", flush=True)
except Exception as exc:
    print(f"[UnitPort][play] Could not tweak RTX exposure settings "
          f"(non-fatal): {exc}", flush=True)

# ── 3. Post-launch imports (need sim running) ────────────────────────────

from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401  — registers all built-in tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Hydra wrapper that resolves env_cfg + agent_cfg from the gym task ID
# without forcing us through play.py's get_checkpoint_path codepath.
from isaaclab_tasks.utils.hydra import hydra_task_config


def _resolve_checkpoint_path(load_run: str, checkpoint: str = "") -> Path:
    """Find the .pt file to feed into ``OnPolicyRunner.load()``.

    Accepts:
      - A direct path to a .pt file               → returned as-is
      - A run directory (containing model_*.pt)   → newest model_*.pt
      - A run directory + explicit checkpoint name
    """
    p = Path(load_run)
    if not p.exists():
        raise FileNotFoundError(f"--load_run path does not exist: {load_run}")
    if p.is_file():
        return p
    if checkpoint:
        candidate = p / checkpoint
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"--checkpoint {checkpoint!r} not found inside {load_run}"
        )
    # Directory: pick the highest-numbered model_*.pt
    def _iter_num(path: Path) -> int:
        try:
            return int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    candidates = sorted(p.glob("model_*.pt"), key=_iter_num, reverse=True)
    if not candidates or _iter_num(candidates[0]) < 0:
        # Some RSL-RL setups nest under params/ or checkpoints/. Search one
        # level deeper before giving up.
        for sub in ("checkpoints", "params"):
            sub_dir = p / sub
            if sub_dir.is_dir():
                deeper = sorted(sub_dir.glob("model_*.pt"), key=_iter_num, reverse=True)
                if deeper and _iter_num(deeper[0]) >= 0:
                    return deeper[0]
        raise FileNotFoundError(
            f"No model_*.pt found in {load_run} (or its checkpoints/, params/ subdirs)"
        )
    return candidates[0]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Replay logic — same shape as Isaac Lab's play.py main() but with our
    checkpoint resolver instead of ``get_checkpoint_path``."""

    # Resolve checkpoint BEFORE building the env so we fail fast on a
    # missing path instead of after a 30s sim spin-up.
    resume_path = _resolve_checkpoint_path(args_cli.load_run, args_cli.checkpoint)
    print(f"[UnitPort][play] Resolved checkpoint: {resume_path}", flush=True)

    # Override env config with CLI args (smaller envs for replay).
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed

    # Disable obs corruption (noise) so the user sees a clean signal during
    # review. Everything else stays at the env's training defaults — we
    # don't touch events, debug_vis, etc., because partial overrides risk
    # leaving the manager in an inconsistent state.
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        try:
            env_cfg.observations.policy.enable_corruption = False
        except Exception:
            pass

    # ── Force a constant non-zero velocity command ──────────────────────
    # The trained policy is just a function: command + state → action. If
    # the env's CommandManager happens to sample a zero command (which
    # ``UniformVelocityCommand`` does with probability ``rel_standing_envs``,
    # plus it only resamples every ~10 seconds), the policy will correctly
    # output "stand still" and the user sees a motionless robot. For review
    # we want continuous walking, so pin the command to a forward 0.7 m/s
    # for the entire session.
    try:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            cmd_cfg = env_cfg.commands.base_velocity
            if hasattr(cmd_cfg, "ranges"):
                cmd_cfg.ranges.lin_vel_x = (0.7, 0.7)
                cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
                cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
                if hasattr(cmd_cfg.ranges, "heading"):
                    cmd_cfg.ranges.heading = (0.0, 0.0)
            # Never sample a "standing still" command
            if hasattr(cmd_cfg, "rel_standing_envs"):
                cmd_cfg.rel_standing_envs = 0.0
            if hasattr(cmd_cfg, "rel_heading_envs"):
                cmd_cfg.rel_heading_envs = 0.0
            # Don't resample during the demo (10000s ≈ never)
            if hasattr(cmd_cfg, "resampling_time_range"):
                cmd_cfg.resampling_time_range = (10000.0, 10000.0)
            # Show the green/blue velocity arrows over the robot during
            # review — Isaac Lab's built-in goal_vel + current_vel
            # visualizers, gated by ``debug_vis``.
            if hasattr(cmd_cfg, "debug_vis"):
                cmd_cfg.debug_vis = True
            print("[UnitPort][play] Pinned velocity command to "
                  "vx=0.7 m/s, vy=0, wz=0 for the entire review session "
                  "(direction arrows enabled).",
                  flush=True)
    except Exception as exc:
        print(f"[UnitPort][play] Could not pin velocity command "
              f"(non-fatal): {exc}", flush=True)

    # ── Make the viewer camera follow the robot ─────────────────────────
    # Default ViewerCfg looks at world origin (0,0,0), but rough-terrain
    # tasks spawn the robot somewhere inside a procedural terrain grid that
    # may be tens of metres from origin — the user sees an empty floor and
    # has to manually fly the camera to find the robot. Switch the viewer
    # to "follow the robot's articulation root" so the camera stays glued
    # to it for the entire session.
    try:
        if hasattr(env_cfg, "viewer"):
            v = env_cfg.viewer
            # Camera position is interpreted relative to the asset root.
            v.eye = (2.5, 2.5, 1.5)     # 2.5m back-right, 1.5m up
            v.lookat = (0.0, 0.0, 0.3)  # aim at the robot's mid-body
            v.origin_type = "asset_root"
            v.env_index = 0
            v.asset_name = "robot"
            print("[UnitPort][play] Viewer set to follow robot "
                  "(eye offset = 2.5,2.5,1.5).", flush=True)
    except Exception as exc:
        print(f"[UnitPort][play] Could not pin viewer to robot "
              f"(non-fatal): {exc}", flush=True)

    # Build the env (rendering enabled — review mode).
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder="/tmp/unitport_play_videos",
                                       step_trigger=lambda s: s == 0,
                                       video_length=args_cli.video_length,
                                       disable_logger=True)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", 1.0))

    # Create runner. ``log_dir=None`` tells RSL-RL not to write training
    # logs (we're not training). Pass agent_cfg through ``to_dict()`` AS-IS
    # — do NOT pre-filter actor/critic keys. The built-in Isaac Lab tasks
    # ship a deprecated ``policy`` config that rsl_rl 4.x auto-converts to
    # ``actor``/``critic`` internally; any client-side filtering breaks
    # that conversion (we lose ``class_name`` and OnPolicyRunner blows up
    # in construct_algorithm). The previous filter was copy-pasted from
    # ``il_train_launcher.py`` where it's needed because UnitPort builds
    # its own RslRlMLPModelCfg with extra keys — that situation does not
    # apply on the play path.
    agent_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else vars(agent_cfg)
    device = getattr(agent_cfg, "device", "cuda:0")

    # Wrap construction in the same hold-viewport-open guard as runner.load
    # below — without this, an exception in OnPolicyRunner.__init__ escapes
    # past hydra and tears down Isaac Sim before the user can see anything
    # (we'd lose the env scene that was just successfully built).
    try:
        runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=device)
    except Exception:
        import traceback
        print("[UnitPort][play] FATAL: OnPolicyRunner construction raised:",
              flush=True)
        traceback.print_exc()
        print("[UnitPort][play] Holding viewport open for inspection — "
              "close the Isaac Sim window to exit. The scene above is "
              "valid; only the policy runner failed to instantiate.",
              flush=True)
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except Exception:
            pass
        os._exit(1)

    # Load checkpoint by absolute path — the whole point of this launcher.
    print(f"[UnitPort][play] Loading checkpoint into OnPolicyRunner ...", flush=True)
    try:
        runner.load(str(resume_path))
    except Exception as exc:
        import traceback
        print(f"[UnitPort][play] FATAL: runner.load() raised an exception:",
              flush=True)
        traceback.print_exc()
        # Keep the viewport alive so the user at least sees the env that
        # was successfully built. Idle-spin until they close the window.
        print("[UnitPort][play] Holding viewport open for inspection — "
              "close the Isaac Sim window to exit.", flush=True)
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except Exception:
            pass
        os._exit(1)
    print(f"[UnitPort][play] Checkpoint loaded. Resolving first observation ...",
          flush=True)

    # Initialise observations. RslRlVecEnvWrapper.get_observations() returns
    # ``(obs_tensor, extras_dict)``; the underlying env was already reset
    # by the wrapper's __init__, so we don't need a second reset() call.
    try:
        if hasattr(env, "get_observations"):
            obs, _extras = env.get_observations()
        else:
            obs, _ = env.reset()
    except Exception as exc:
        import traceback
        print(f"[UnitPort][play] FATAL: could not obtain initial obs:",
              flush=True)
        traceback.print_exc()
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except Exception:
            pass
        os._exit(1)

    print(f"[UnitPort][play] First obs shape: "
          f"{tuple(obs.shape) if hasattr(obs, 'shape') else type(obs).__name__}. "
          f"Starting playback loop.", flush=True)

    policy = runner.get_inference_policy(device=device)
    step_count = 0
    last_log_step = 0

    # Diagnostic: print first action so we can tell whether the policy is
    # actually producing non-zero outputs vs. the env not advancing.
    try:
        with torch.inference_mode():
            _first_action = policy(obs)
        if hasattr(_first_action, "shape"):
            _flat = _first_action.reshape(-1).cpu().tolist() if hasattr(_first_action, "cpu") else list(_first_action)
            _preview = ", ".join(f"{v:+.3f}" for v in _flat[:6])
            print(f"[UnitPort][play] First action shape={tuple(_first_action.shape)} "
                  f"sample=[{_preview}{', ...' if len(_flat) > 6 else ''}]",
                  flush=True)
    except Exception as exc:
        print(f"[UnitPort][play] First-action diagnostic failed "
              f"(non-fatal): {exc}", flush=True)

    # Inference loop — render until the user closes the window or hits
    # Ctrl+C. We catch step()/policy() exceptions INSIDE the loop so a
    # transient runtime issue (shape mismatch, NaN, etc.) doesn't tear
    # down the viewport — instead we log it once and idle-spin so the user
    # can still see what was built.
    runtime_error_logged = False
    max_steps = int(args_cli.max_play_steps or 0)
    if max_steps > 0:
        print(f"[UnitPort][play] Playback will auto-stop after "
              f"{max_steps} steps (~{max_steps / 50.0:.0f}s @50Hz). "
              f"Pass --max_play_steps 0 to disable the cap.", flush=True)
    try:
        while simulation_app.is_running():
            # Hard cap so the viewport always self-terminates even if the
            # window-close event from Kit doesn't propagate (Windows +
            # detached subprocess + redirected stdout sometimes eats it).
            if max_steps > 0 and step_count >= max_steps:
                print(f"[UnitPort][play] Reached --max_play_steps "
                      f"({max_steps}); closing viewport.", flush=True)
                break
            try:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                step_count += 1
                if step_count - last_log_step >= 100:
                    print(f"[UnitPort][play] step={step_count}", flush=True)
                    last_log_step = step_count
            except Exception as exc:
                if not runtime_error_logged:
                    import traceback
                    print(f"[UnitPort][play] Inference step raised an "
                          f"exception (step={step_count}):", flush=True)
                    traceback.print_exc()
                    print("[UnitPort][play] Holding viewport open — close "
                          "the Isaac Sim window to exit. Most likely cause: "
                          "the trained policy was produced with a UnitPort-"
                          "compiled env_cfg whose obs/action layout differs "
                          "from the built-in task this fallback is using.",
                          flush=True)
                    runtime_error_logged = True
                # Idle-update so the window stays interactive.
                try:
                    simulation_app.update()
                except Exception:
                    break
    except KeyboardInterrupt:
        print("[UnitPort][play] Interrupted — closing.", flush=True)
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[UnitPort][play] env.close() raised (ignored): {exc}",
                  flush=True)
        try:
            simulation_app.close()
        except Exception:
            pass
        # Hard exit to bypass any lingering Kit / Omniverse threads (same
        # rationale as il_train_launcher.py).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
