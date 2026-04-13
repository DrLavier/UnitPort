"""UnitPort Isaac Lab — Persistent Review Session launcher.

Per ``knowledge_base/IsaacSim_design.yaml`` §2 and §4.

This script is a long-lived sibling of ``il_train_launcher.py`` and
``il_play_launcher.py``.  Unlike them it does NOT exit after one
training run / one playback session — it boots Isaac Sim Kit ONCE
and then enters an IPC main loop that reads JSON commands from
stdin.  Each command can:

  * load a new policy checkpoint into the live actor (hot-swap)
  * override the env velocity command (manual gamepad / keyboard control)
  * switch between AUTO (scheduled demo) and MANUAL modes
  * reset / pause / resume the simulation
  * snapshot the current robot state
  * shutdown gracefully

The first review action pays the ~30 s Kit boot cost; every action
afterwards is < 1 s because the same process keeps the env, the actor,
and the Kit window alive.

Usage (called by ``IsaacReviewSession`` in UnitPort, NOT directly)::

    python il_review_session.py \\
        --task UnitPort-Custom-v0 \\
        --unitport_config /path/to/compiled_cfg.py \\
        --num_envs 1

The session prints structured JSON status messages to stdout — one per
line — that the parent process consumes via its stdout reader thread.
See §3.session_to_manager in IsaacSim_design.yaml for the message
schema (ready, loaded, stage_changed, error, telemetry).

Architecture decisions
----------------------
1. **Single Kit boot** — AppLauncher is invoked once, simulation_app
   stays alive for the entire session lifetime.
2. **stdin reader thread** — sys.stdin.readline() is blocking and
   Windows pipes don't support select; a daemon thread reads stdin
   into a queue.Queue and the main loop drains it non-blocking.
3. **Command override via vel_command_b** — Isaac Lab's
   UniformVelocityCommand exposes ``vel_command_b`` as the live
   command tensor.  We disable resampling (huge resampling_time_range)
   and standing-env probability so our writes survive _update_command.
4. **Hot-swap via direct ActorCritic.load_state_dict** — we skip
   OnPolicyRunner / AMPOnPolicyRunner entirely (they require nested
   cfg dicts and runtime infrastructure we don't need for inference).
   Instead we build a vendored ActorCritic once at startup, then on
   each load_checkpoint IPC call we ``torch.load`` the .pt and call
   ``actor_critic.load_state_dict`` in-place.  The same code path
   works for BOTH PPO and AMP_PPO checkpoints because the actor's
   state_dict layout is identical (AMP's discriminator is a separate
   top-level key in the .pt and we ignore it).
5. **AUTO scheduler is wall-clock based** — uses ``time.monotonic()``
   so it's robust to env step time variations.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# ── Make ``src.system.*`` importable from inside the Isaac venv ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Auto-accept Omniverse Kit EULA
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# h5py-first dance to avoid HDF5 symbol races (same as il_train_launcher)
try:
    import h5py  # noqa: F401
except Exception:
    pass

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="UnitPort Isaac Lab — Persistent Review Session"
)
parser.add_argument("--task", type=str, default="UnitPort-Custom-v0",
                    help="Gym task ID (matches the one registered by --unitport_config)")
parser.add_argument("--unitport_config", type=str, required=True,
                    help="Path to UnitPort-compiled @configclass Python file. "
                         "Determines obs/action layout and env contents.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs in the review viewer. "
                         "Default 1 — single robot for clearest visualization.")
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force non-headless: this script's whole purpose is the visualization
args_cli.headless = False
sys.argv = [sys.argv[0]] + hydra_args


# ---------------------------------------------------------------------------
# Stdout messaging — structured JSON, one per line
# ---------------------------------------------------------------------------

# Use a unique sentinel prefix so the parent stdout reader can filter
# UnitPort messages from Kit's own log spam.
_MSG_PREFIX = "[UNITPORT_REVIEW_MSG]"


def emit(msg: str, **fields) -> None:
    """Send a structured message to the parent process via stdout."""
    payload = {"msg": msg, **fields}
    try:
        print(f"{_MSG_PREFIX}{json.dumps(payload, ensure_ascii=False)}",
              flush=True)
    except Exception:
        pass


def emit_error(where: str, detail: str) -> None:
    emit("error", where=where, detail=detail)


# ---------------------------------------------------------------------------
# Stdin reader thread
# ---------------------------------------------------------------------------

_stdin_queue: "queue.Queue[dict]" = queue.Queue(maxsize=2048)
_stdin_stop = threading.Event()


def _stdin_reader_loop() -> None:
    """Daemon thread: read newline-delimited JSON from stdin into a queue.

    Windows pipes don't support select() on stdin and sys.stdin.readline()
    is blocking, so we run this in a thread and the main loop drains the
    queue non-blocking once per env step.
    """
    while not _stdin_stop.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:
            # EOF — parent closed the pipe.  Push a synthetic shutdown.
            try:
                _stdin_queue.put_nowait({"op": "shutdown"})
            except queue.Full:
                pass
            return
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as exc:
            emit_error("stdin_parse", f"{exc}: {line[:200]}")
            continue
        if not isinstance(cmd, dict) or "op" not in cmd:
            emit_error("stdin_parse", f"missing 'op' field: {line[:200]}")
            continue
        try:
            _stdin_queue.put_nowait(cmd)
        except queue.Full:
            emit_error("stdin_queue", "queue full, dropping command")


# ---------------------------------------------------------------------------
# Boot Isaac Sim
# ---------------------------------------------------------------------------

print(f"[UnitPort][review] Booting Isaac Sim (headless=False, num_envs={args_cli.num_envs})...",
      flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Tame RTX auto-exposure — same defensive setup as il_play_launcher.py
try:
    import carb
    _carb = carb.settings.get_settings()
    _carb.set("/rtx/post/tonemap/op", 1)
    _carb.set("/rtx/post/histogram/enabled", False)
    _carb.set("/rtx/post/histogram/autoExposure", False)
    _carb.set("/rtx/post/tonemap/filmIso", 100.0)
    _carb.set("/rtx/post/tonemap/cameraShutter", 100.0)
    _carb.set("/rtx/post/tonemap/fNumber", 8.0)
    _carb.set("/rtx/post/aa/op", 3)  # DLAA
except Exception:
    pass


# ---------------------------------------------------------------------------
# Post-launch imports (need sim running)
# ---------------------------------------------------------------------------

import importlib.util
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  — registers built-in tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from src.system.training.auto_command_schedule import (  # noqa: E402
    build_auto_command_schedule,
    format_stage_for_hud,
)
# We use the vendored UnitPort ActorCritic so the same instance can
# load checkpoints from BOTH PPO and AMP_PPO training runs (the
# state_dict layout is identical because the AMP discriminator is a
# separate top-level key in the .pt file, not part of actor_critic).
from src.system.training.amp.algorithms.actor_critic import ActorCritic


# ---------------------------------------------------------------------------
# Load UnitPort-compiled env_cfg (same pattern as il_train_launcher)
# ---------------------------------------------------------------------------

print(f"[UnitPort][review] Loading compiled config: {args_cli.unitport_config}",
      flush=True)
spec = importlib.util.spec_from_file_location(
    "unitport_review_env_cfg", args_cli.unitport_config
)
_cfg_mod = importlib.util.module_from_spec(spec)
sys.modules["unitport_review_env_cfg"] = _cfg_mod
spec.loader.exec_module(_cfg_mod)

UnitPortEnvCfg = getattr(_cfg_mod, "UnitPortEnvCfg", None)
PPORunnerCfg = getattr(_cfg_mod, "PPORunnerCfg", None)
if UnitPortEnvCfg is None:
    emit_error("config_load", "UnitPortEnvCfg not found in compiled config")
    print("[UnitPort][review] FATAL: UnitPortEnvCfg missing", flush=True)
    sys.exit(2)

gym.register(
    id="UnitPort-Review-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": UnitPortEnvCfg},
)


# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

env_cfg = UnitPortEnvCfg()

# Force single-robot review
env_cfg.scene.num_envs = max(1, int(args_cli.num_envs))
if hasattr(env_cfg, "seed"):
    env_cfg.seed = args_cli.seed

# Disable observation corruption — review wants clean signal
if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
    try:
        env_cfg.observations.policy.enable_corruption = False
    except Exception:
        pass

# Direction arrows on
try:
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        cmd_cfg = env_cfg.commands.base_velocity
        if hasattr(cmd_cfg, "debug_vis"):
            cmd_cfg.debug_vis = True
        # Disable auto-resample so our IPC overrides survive
        if hasattr(cmd_cfg, "resampling_time_range"):
            cmd_cfg.resampling_time_range = (1e9, 1e9)
        if hasattr(cmd_cfg, "rel_standing_envs"):
            cmd_cfg.rel_standing_envs = 0.0
        if hasattr(cmd_cfg, "rel_heading_envs"):
            cmd_cfg.rel_heading_envs = 0.0
except Exception as exc:
    print(f"[UnitPort][review] Could not tune base_velocity cfg: {exc}",
          flush=True)

# Camera follow robot
try:
    if hasattr(env_cfg, "viewer"):
        v = env_cfg.viewer
        v.eye = (2.5, 2.5, 1.5)
        v.lookat = (0.0, 0.0, 0.3)
        v.origin_type = "asset_root"
        v.env_index = 0
        v.asset_name = "robot"
except Exception:
    pass

# Capture velocity ranges BEFORE building the env (to feed AUTO scheduler).
# Use defaults if the cfg doesn't expose them.
def _range(obj, attr, default):
    try:
        val = getattr(obj.ranges, attr, None)
        if val is not None and len(val) >= 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return default

vel_ranges = (
    _range(env_cfg.commands.base_velocity, "lin_vel_x", (-0.5, 2.0)),
    _range(env_cfg.commands.base_velocity, "lin_vel_y", (-0.5, 0.5)),
    _range(env_cfg.commands.base_velocity, "ang_vel_z", (-1.5, 1.5)),
)
print(f"[UnitPort][review] Velocity ranges: vx={vel_ranges[0]} "
      f"vy={vel_ranges[1]} wz={vel_ranges[2]}", flush=True)

env = gym.make("UnitPort-Review-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=1.0)


# ---------------------------------------------------------------------------
# Resolve velocity command term + neutralise auto-update behaviour
# ---------------------------------------------------------------------------

def _get_velocity_command_term():
    """Return the env's UniformVelocityCommand term, or None."""
    try:
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        cm = unwrapped.command_manager
        if hasattr(cm, "get_term"):
            return cm.get_term("base_velocity")
        # Some Isaac Lab versions store terms in a private dict
        terms = getattr(cm, "_terms", {}) or {}
        return terms.get("base_velocity")
    except Exception:
        return None


_vel_term = _get_velocity_command_term()
if _vel_term is not None:
    try:
        # Never trigger standing override — always honour our writes
        if hasattr(_vel_term, "is_standing_env"):
            _vel_term.is_standing_env.zero_()
    except Exception:
        pass


def _override_velocity_command(vx: float, vy: float, wz: float) -> bool:
    """Write (vx, vy, wz) to the env's command tensor in-place."""
    term = _vel_term if _vel_term is not None else _get_velocity_command_term()
    if term is None:
        return False
    try:
        cmd = term.vel_command_b  # shape (num_envs, 3)
        cmd[:, 0] = float(vx)
        cmd[:, 1] = float(vy)
        cmd[:, 2] = float(wz)
        if hasattr(term, "is_standing_env"):
            term.is_standing_env.zero_()
        return True
    except Exception as exc:
        emit_error("override_command", str(exc))
        return False


# ---------------------------------------------------------------------------
# Build a vendored ActorCritic — will be loaded with real weights via
# the load_checkpoint IPC op.
#
# We DELIBERATELY skip OnPolicyRunner / AMPOnPolicyRunner here because
# both require nested cfg dicts ({runner, algorithm, policy, ...}) plus
# either expert motion data or a properly-built RslRlPpoAlgorithmCfg —
# none of which we need for inference-only review.  Constructing the
# ActorCritic directly is simpler and supports BOTH PPO and AMP_PPO
# checkpoints (the actor's state_dict layout is identical; AMP's
# discriminator is a separate top-level key in the .pt file).
# ---------------------------------------------------------------------------

if PPORunnerCfg is None:
    emit_error("config_load", "PPORunnerCfg missing in compiled config")
    print("[UnitPort][review] FATAL: PPORunnerCfg missing", flush=True)
    sys.exit(2)

# Pull architectural params off the compiled cfg.
try:
    _cfg = PPORunnerCfg()
    _actor_hidden = list(getattr(_cfg, "actor_hidden_dims", [128, 64, 32]))
    _critic_hidden = list(getattr(_cfg, "critic_hidden_dims", [128, 64, 32]))
    _activation = str(getattr(_cfg, "activation", "elu"))
    _init_noise_std = float(getattr(_cfg, "init_noise_std", 0.37))
    _algorithm_class = str(getattr(_cfg, "algorithm_class", "PPO"))
except Exception as exc:
    emit_error("config_load", f"PPORunnerCfg instantiation failed: {exc}")
    print(f"[UnitPort][review] FATAL: PPORunnerCfg() raised: {exc}",
          flush=True)
    sys.exit(2)

# Resolve obs/action dims from the live env.  RslRlVecEnvWrapper
# stores num_actions as an attribute (set at __init__) and exposes
# obs dims via the underlying ManagerBasedRLEnv.observation_manager.
# env.observation_space is a TensorDict in modern Isaac Lab and
# does NOT have a usable .shape — fall through several query paths.
def _extract_obs_tensor(obs):
    """Normalize an obs return into a plain (num_envs, obs_dim) torch.Tensor.

    Modern Isaac Lab's RslRlVecEnvWrapper returns a TensorDict (with
    a ``policy`` key) instead of a flat tensor.  Earlier versions
    returned a tuple ``(obs_tensor, extras)``.  This helper handles
    every shape we've seen so the inference / scheduler / step loop
    don't need to know which Isaac Lab version is in use.
    """
    if obs is None:
        return None
    # Tuple/list — first element is the obs (rest is extras)
    if isinstance(obs, (tuple, list)):
        if len(obs) == 0:
            return None
        return _extract_obs_tensor(obs[0])
    # TensorDict / dict-like — pull the "policy" group
    if hasattr(obs, "__getitem__") and not torch.is_tensor(obs):
        try:
            inner = obs["policy"]
            return _extract_obs_tensor(inner)
        except (KeyError, TypeError, IndexError):
            pass
    # Plain tensor
    if torch.is_tensor(obs):
        # Some TensorDict subclasses inherit Tensor — force a plain
        # contiguous Tensor so onnxruntime / nn.Linear don't trip on
        # __torch_function__ overrides.
        try:
            return obs.as_subclass(torch.Tensor).clone().contiguous()
        except Exception:
            return obs
    return obs


def _query_env_dims():
    num_obs, num_actions = None, None

    # Try wrapper attributes first (set at __init__ for RslRlVecEnvWrapper)
    if hasattr(env, "num_actions") and env.num_actions:
        num_actions = int(env.num_actions)

    # Best path for obs: ManagerBasedRLEnv.observation_manager.group_obs_dim
    try:
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        obs_mgr = getattr(unwrapped, "observation_manager", None)
        if obs_mgr is not None:
            group_dims = getattr(obs_mgr, "group_obs_dim", None)
            if group_dims and "policy" in group_dims:
                shape = group_dims["policy"]
                if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                    num_obs = int(shape[0])
    except Exception:
        pass

    # Fallback: action_manager.total_action_dim
    if num_actions is None:
        try:
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            act_mgr = getattr(unwrapped, "action_manager", None)
            if act_mgr is not None:
                num_actions = int(act_mgr.total_action_dim)
        except Exception:
            pass

    # Fallback: single_*_space (gym vectorized API)
    if num_obs is None:
        try:
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            sos = getattr(unwrapped, "single_observation_space", None)
            if sos is not None:
                if hasattr(sos, "shape") and sos.shape:
                    num_obs = int(sos.shape[-1])
                elif hasattr(sos, "__getitem__"):
                    # TensorDict / Dict[group_name → space]
                    pol = sos["policy"] if "policy" in sos else None
                    if pol is not None and hasattr(pol, "shape"):
                        num_obs = int(pol.shape[-1])
        except Exception:
            pass

    if num_actions is None:
        try:
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            sas = getattr(unwrapped, "single_action_space", None)
            if sas is not None and hasattr(sas, "shape") and sas.shape:
                num_actions = int(sas.shape[-1])
        except Exception:
            pass

    # Final fallback: actually call get_observations
    if num_obs is None:
        try:
            raw0 = None
            if hasattr(env, "get_observations"):
                raw0 = env.get_observations()
            obs0 = _extract_obs_tensor(raw0)
            if obs0 is not None and hasattr(obs0, "shape"):
                num_obs = int(obs0.shape[-1])
        except Exception:
            pass

    return num_obs, num_actions


_num_obs, _num_actions = _query_env_dims()
if not _num_obs or not _num_actions:
    emit_error(
        "env_query",
        f"could not read obs/action shape (num_obs={_num_obs}, "
        f"num_actions={_num_actions}). The env might use a non-standard "
        f"observation manager."
    )
    print(f"[UnitPort][review] FATAL: env shape query failed: "
          f"num_obs={_num_obs} num_actions={_num_actions}", flush=True)
    sys.exit(2)

device = str(getattr(args_cli, "device", None) or "cuda:0")

print(f"[UnitPort][review] Building ActorCritic: "
      f"num_obs={_num_obs} num_actions={_num_actions} "
      f"actor_hidden={_actor_hidden} critic_hidden={_critic_hidden} "
      f"activation={_activation} init_noise_std={_init_noise_std} "
      f"algorithm={_algorithm_class}", flush=True)

try:
    actor_critic = ActorCritic(
        num_actor_obs=_num_obs,
        num_critic_obs=_num_obs,  # critic same shape for inference
        num_actions=_num_actions,
        actor_hidden_dims=_actor_hidden,
        critic_hidden_dims=_critic_hidden,
        activation=_activation,
        init_noise_std=_init_noise_std,
        fixed_std=False,
    ).to(device)
    actor_critic.eval()
except Exception as exc:
    import traceback
    emit_error("actor_critic_init", str(exc))
    print("[UnitPort][review] FATAL: ActorCritic construction raised:",
          flush=True)
    traceback.print_exc()
    print("[UnitPort][review] Holding viewport open — close window to exit.",
          flush=True)
    while simulation_app.is_running():
        try:
            simulation_app.update()
        except Exception:
            break
    sys.exit(1)


# ---------------------------------------------------------------------------
# Initial env reset
# ---------------------------------------------------------------------------

try:
    raw_obs = None
    # RslRlVecEnvWrapper.get_observations() returns a single TensorDict
    # in modern Isaac Lab — NOT a (obs, extras) tuple.  ``reset()``
    # still returns the tuple.  Handle both via the helper.
    if hasattr(env, "get_observations"):
        raw_obs = env.get_observations()
    else:
        reset_ret = env.reset()
        if isinstance(reset_ret, (tuple, list)) and len(reset_ret) >= 1:
            raw_obs = reset_ret[0]
        else:
            raw_obs = reset_ret
    obs = _extract_obs_tensor(raw_obs)
    if obs is None:
        raise RuntimeError(
            f"could not extract obs tensor from env return type "
            f"{type(raw_obs).__name__}"
        )
except Exception as exc:
    import traceback
    traceback.print_exc()
    emit_error("env_reset", str(exc))
    print(f"[UnitPort][review] FATAL: initial reset/get_obs failed: {exc}",
          flush=True)
    sys.exit(1)

# Determine obs/action dims from the runner / env
try:
    obs_dim = int(obs.shape[-1]) if hasattr(obs, "shape") else 0
except Exception:
    obs_dim = 0
try:
    action_dim = int(env.action_space.shape[-1])
except Exception:
    action_dim = 0


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class SessionState:
    """Mutable state for the IPC main loop."""

    def __init__(self):
        self.policy = None       # set by first load_checkpoint
        self.checkpoint_path: str = ""
        self.paused: bool = True  # idle until first checkpoint loaded
        self.control_mode: str = "AUTO"
        self.scheduler = None    # AutoCommandSchedule instance
        self.scheduler_start_t: float = 0.0
        self.last_stage_idx: int = -1
        self.shutdown_requested: bool = False
        # Telemetry
        self.last_telemetry_t: float = 0.0
        self.step_count: int = 0
        self.last_step_count: int = 0


state = SessionState()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _activate_auto_scheduler() -> None:
    state.scheduler = build_auto_command_schedule(
        vx_range=vel_ranges[0],
        vy_range=vel_ranges[1],
        wz_range=vel_ranges[2],
    )
    state.scheduler_start_t = time.monotonic()
    state.last_stage_idx = -1
    print(f"[UnitPort][review] AUTO scheduler armed: "
          f"{len(state.scheduler.stages)} stages, "
          f"{state.scheduler.cycle_time_s:.1f}s cycle", flush=True)


def _deactivate_auto_scheduler() -> None:
    state.scheduler = None
    state.last_stage_idx = -1


def _load_actor_from_pt(path: str) -> Optional[Callable]:
    """Load a torch .pt checkpoint into the live ActorCritic.

    Returns an inference callable, or None on failure (caller emits
    the error).  Used for direct training-run checkpoints
    (model_*.pt under params/ or checkpoints/).
    """
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit("loaded", path=path, ok=False,
             error=f"torch.load failed: {exc}")
        return None

    state_dict = None
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "actor_critic_state_dict",
                    "model", "state_dict"):
            if key in loaded and isinstance(loaded[key], dict):
                state_dict = loaded[key]
                break
    if state_dict is None:
        emit("loaded", path=path, ok=False,
             error=f"no model_state_dict / actor_critic_state_dict / model "
                   f"key in checkpoint (top-level keys: "
                   f"{list(loaded.keys()) if isinstance(loaded, dict) else type(loaded).__name__})")
        return None

    try:
        missing, unexpected = actor_critic.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            print(f"[UnitPort][review] WARNING: missing actor_critic keys "
                  f"(first 5): {list(missing)[:5]}", flush=True)
        if unexpected:
            print(f"[UnitPort][review] WARNING: unexpected actor_critic keys "
                  f"(first 5): {list(unexpected)[:5]}", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit("loaded", path=path, ok=False,
             error=f"load_state_dict failed: {exc}")
        return None

    actor_critic.eval()
    return lambda obs: actor_critic.act_inference(obs)


def _load_actor_from_onnx(path: str) -> Optional[Callable]:
    """Load a policy.onnx file via onnxruntime and return an inference callable.

    UnitPort bundles export the actor as ONNX (not PyTorch) — this is
    the path that handles them.  Inference is done with onnxruntime
    (CPU or CUDA provider depending on availability).  We wrap the
    output in a torch.Tensor on the env's device so the rest of the
    review loop doesn't need to know which backend produced the action.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        emit("loaded", path=path, ok=False,
             error=f"onnxruntime not available: {exc}")
        return None
    except Exception as exc:
        emit("loaded", path=path, ok=False,
             error=f"onnxruntime import failed: {exc}")
        return None

    try:
        # Prefer CUDA provider if available so inference matches the
        # device the env runs on; fall back to CPU silently.
        providers = ort.get_available_providers()
        preferred = []
        if "CUDAExecutionProvider" in providers:
            preferred.append("CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")
        sess = ort.InferenceSession(str(path), providers=preferred)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit("loaded", path=path, ok=False,
             error=f"InferenceSession init failed: {exc}")
        return None

    input_name = sess.get_inputs()[0].name
    expected_input_dim = None
    try:
        expected_input_dim = int(sess.get_inputs()[0].shape[-1])
    except Exception:
        pass

    print(f"[UnitPort][review] ONNX session ready: providers={providers}, "
          f"input='{input_name}' shape={sess.get_inputs()[0].shape}",
          flush=True)

    if expected_input_dim is not None and expected_input_dim != _num_obs:
        print(f"[UnitPort][review] WARNING: ONNX expects obs_dim="
              f"{expected_input_dim} but env produces obs_dim={_num_obs}. "
              f"This will fail at inference time — the canvas's "
              f"observation layout doesn't match the bundle's training "
              f"observation layout.", flush=True)

    def _inference(obs):
        # Coerce to a contiguous (N, obs_dim) float32 numpy array
        if torch.is_tensor(obs):
            arr = obs.detach().cpu().numpy()
        else:
            arr = obs
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Some bundles export with float32 inputs even though Isaac Lab
        # produces float64 envs — cast to be safe.
        if arr.dtype != "float32":
            arr = arr.astype("float32")
        out = sess.run(None, {input_name: arr})[0]
        return torch.as_tensor(out, device=device)

    return _inference


def _handle_load_checkpoint(cmd: dict) -> None:
    path = str(cmd.get("path", "")).strip()
    auto_mode = bool(cmd.get("auto_mode", True))
    if not path or not os.path.isfile(path):
        emit("loaded", path=path, ok=False, error=f"file not found: {path}")
        return
    print(f"[UnitPort][review] Loading checkpoint: {path}", flush=True)

    # Dispatch on file extension.  UnitPort exports bundles as .onnx
    # (the deployable actor) but training runs save raw .pt files.
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".onnx":
        policy_fn = _load_actor_from_onnx(path)
    elif suffix == ".pt":
        policy_fn = _load_actor_from_pt(path)
    else:
        emit("loaded", path=path, ok=False,
             error=f"unsupported checkpoint extension: {suffix} "
                   f"(expected .onnx or .pt)")
        return

    if policy_fn is None:
        return  # _load_* already emitted the error

    state.checkpoint_path = path
    state.policy = policy_fn
    state.paused = False
    state.control_mode = "AUTO" if auto_mode else "MANUAL"
    if auto_mode:
        _activate_auto_scheduler()
    else:
        _deactivate_auto_scheduler()
    emit("loaded", path=path, ok=True, control_mode=state.control_mode)


def _handle_command(cmd: dict) -> None:
    if state.control_mode != "MANUAL":
        # Silently ignore — AUTO scheduler owns the command tensor.
        return
    try:
        vx = float(cmd.get("vx", 0.0))
        vy = float(cmd.get("vy", 0.0))
        wz = float(cmd.get("wz", 0.0))
    except (TypeError, ValueError) as exc:
        emit_error("command", f"bad payload: {exc}")
        return
    _override_velocity_command(vx, vy, wz)


def _handle_control_mode(cmd: dict) -> None:
    mode = str(cmd.get("mode", "AUTO")).upper()
    if mode not in ("AUTO", "MANUAL"):
        emit_error("control_mode", f"invalid mode: {mode}")
        return
    state.control_mode = mode
    if mode == "AUTO":
        _activate_auto_scheduler()
    else:
        _deactivate_auto_scheduler()
    print(f"[UnitPort][review] Control mode → {mode}", flush=True)


def _handle_reset(_cmd: dict) -> None:
    global obs
    try:
        raw_obs = None
        if hasattr(env, "reset"):
            ret = env.reset()
            raw_obs = ret[0] if isinstance(ret, (tuple, list)) and len(ret) >= 1 else ret
        elif hasattr(env, "get_observations"):
            raw_obs = env.get_observations()
        new_obs = _extract_obs_tensor(raw_obs)
        if new_obs is not None:
            obs = new_obs
    except Exception as exc:
        emit_error("reset", str(exc))
        return
    if state.scheduler is not None:
        state.scheduler_start_t = time.monotonic()
        state.last_stage_idx = -1
    print("[UnitPort][review] Env reset.", flush=True)


def _handle_pause(_cmd: dict) -> None:
    state.paused = True


def _handle_resume(_cmd: dict) -> None:
    state.paused = False


def _handle_snapshot(cmd: dict) -> None:
    label = str(cmd.get("label", "")).strip() or time.strftime("%Y%m%dT%H%M%SZ")
    try:
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        robot = unwrapped.scene["robot"]
        snap = {
            "label": label,
            "checkpoint": state.checkpoint_path,
            "control_mode": state.control_mode,
            "joint_pos": robot.data.joint_pos[0].cpu().tolist(),
            "joint_vel": robot.data.joint_vel[0].cpu().tolist(),
            "base_pos_w": robot.data.root_pos_w[0].cpu().tolist(),
            "base_quat_w": robot.data.root_quat_w[0].cpu().tolist(),
            "base_lin_vel_b": robot.data.root_lin_vel_b[0].cpu().tolist(),
            "base_ang_vel_b": robot.data.root_ang_vel_b[0].cpu().tolist(),
        }
        snap_dir = Path(state.checkpoint_path).parent / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{label}.json"
        with snap_path.open("w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2, ensure_ascii=False)
        emit("snapshot", path=str(snap_path), ok=True)
        print(f"[UnitPort][review] Snapshot saved: {snap_path}", flush=True)
    except Exception as exc:
        emit_error("snapshot", str(exc))


def _handle_shutdown(_cmd: dict) -> None:
    state.shutdown_requested = True


_HANDLERS = {
    "load_checkpoint": _handle_load_checkpoint,
    "command":         _handle_command,
    "control_mode":    _handle_control_mode,
    "reset":           _handle_reset,
    "pause":           _handle_pause,
    "resume":          _handle_resume,
    "snapshot":        _handle_snapshot,
    "shutdown":        _handle_shutdown,
}


def _drain_stdin_queue() -> None:
    """Process all pending IPC commands without blocking."""
    while True:
        try:
            cmd = _stdin_queue.get_nowait()
        except queue.Empty:
            return
        op = str(cmd.get("op", "")).strip()
        handler = _HANDLERS.get(op)
        if handler is None:
            emit_error("dispatch", f"unknown op: {op}")
            continue
        try:
            handler(cmd)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            emit_error("handler", f"{op}: {exc}")


def _tick_auto_scheduler() -> None:
    if state.scheduler is None or state.control_mode != "AUTO":
        return
    elapsed = time.monotonic() - state.scheduler_start_t
    idx, stage = state.scheduler.stage_at_time(elapsed)
    if idx != state.last_stage_idx:
        state.last_stage_idx = idx
        _override_velocity_command(stage.vx, stage.vy, stage.wz)
        emit("stage_changed", index=idx, name=stage.name,
             vx=stage.vx, vy=stage.vy, wz=stage.wz)
        print(f"[UnitPort][review] {format_stage_for_hud(stage, idx, len(state.scheduler.stages))}",
              flush=True)


def _emit_telemetry() -> None:
    now = time.monotonic()
    if now - state.last_telemetry_t < 1.0:
        return
    dt = now - state.last_telemetry_t if state.last_telemetry_t > 0 else 1.0
    fps = (state.step_count - state.last_step_count) / dt if dt > 0 else 0.0
    emit("telemetry", fps=round(fps, 1), step_count=state.step_count,
         control_mode=state.control_mode, paused=state.paused)
    state.last_telemetry_t = now
    state.last_step_count = state.step_count


# ---------------------------------------------------------------------------
# Start IPC and emit READY
# ---------------------------------------------------------------------------

threading.Thread(target=_stdin_reader_loop, daemon=True,
                 name="UnitPortReviewStdinReader").start()

emit("ready",
     num_envs=int(args_cli.num_envs),
     obs_dim=obs_dim,
     action_dim=action_dim,
     vx_range=vel_ranges[0],
     vy_range=vel_ranges[1],
     wz_range=vel_ranges[2])
print(f"[UnitPort][review] READY  obs_dim={obs_dim}  action_dim={action_dim}",
      flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

try:
    while simulation_app.is_running() and not state.shutdown_requested:
        _drain_stdin_queue()
        if state.shutdown_requested:
            break

        _tick_auto_scheduler()

        if state.paused or state.policy is None:
            try:
                simulation_app.update()
            except Exception:
                break
            time.sleep(0.001)
            continue

        try:
            with torch.inference_mode():
                actions = state.policy(obs)
                step_ret = env.step(actions)
                # Modern RslRlVecEnvWrapper.step returns
                # (TensorDict, rew, dones, extras) — 4 values.
                # Older versions returned 5 (with terminated/truncated
                # split).  Be defensive.
                if isinstance(step_ret, (tuple, list)) and len(step_ret) >= 1:
                    raw_next_obs = step_ret[0]
                else:
                    raw_next_obs = step_ret
                obs = _extract_obs_tensor(raw_next_obs)
            state.step_count += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            emit_error("step", str(exc))
            state.paused = True
            try:
                simulation_app.update()
            except Exception:
                break
            continue

        _emit_telemetry()

except KeyboardInterrupt:
    print("[UnitPort][review] Interrupted — shutting down.", flush=True)
finally:
    _stdin_stop.set()
    try:
        env.close()
    except Exception:
        pass
    try:
        simulation_app.close()
    except Exception:
        pass
    emit("shutdown_complete")
    print("[UnitPort][review] Session closed.", flush=True)
