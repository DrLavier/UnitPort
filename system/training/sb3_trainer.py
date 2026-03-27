#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D 閳?Stable-Baselines3 trainer for UnitPort Training Ground.

SB3Trainer
    Builds a Gymnasium env and SB3 model from a compiled TrainingJobSpec
    and runs a real training loop with callbacks that forward progress/log
    updates to the TrainRunThread signal interface.

Callbacks
    SB3ProgressCallback  閳?periodically emits (step, total, reward_mean,
                           best_reward) + log lines.  Supports clean
                           cancellation at each callback tick.

    SB3EvalCallback      閳?lightweight eval after each SB3 checkpoint interval
                           (optional; currently disabled in short-run mode).

Usage (called from TrainRunThread)::

    trainer = SB3Trainer(spec, progress_fn=..., log_fn=..., cancel_fn=...)
    model   = trainer.train()   # returns trained BaseAlgorithm

Dependencies
    stable-baselines3 >= 2.0  (gymnasium >= 0.26 included via SB3)
    gymnasium
    mujoco
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from system.training.training_spec import TrainingJobSpec


# ---------------------------------------------------------------------------
# Types for callbacks
# ---------------------------------------------------------------------------

ProgressFn  = Callable[[int, int, float, float, float, str], None]
LogFn       = Callable[[str], None]
CancelFn    = Callable[[], bool]          # returns True when training should stop
VisCheckFn  = Callable[[Any], None]       # fn(model) 閳?called at vis check milestones


# ---------------------------------------------------------------------------
# SB3ProgressCallback
# ---------------------------------------------------------------------------

class SB3ProgressCallback:
    """
    Stable-Baselines3 ``BaseCallback`` subclass that bridges SB3's internal
    training loop to the TrainRunThread signal interface.

    Parameters
    ----------
    total_timesteps:
        Total training steps (used to compute % progress).
    progress_fn:
        ``fn(step, total, reward_mean, best_reward, status)`` 閳?mapped to
        ``TrainRunThread.progress`` signal.
    log_fn:
        ``fn(text)`` 閳?mapped to ``TrainRunThread.log_line`` signal.
    cancel_fn:
        ``fn() -> bool`` 閳?returns True when the user requests cancellation.
    log_interval:
        Emit a log line every *log_interval* SB3 rollout callbacks.
    """

    def __init__(
        self,
        total_timesteps: int,
        progress_fn: Optional[ProgressFn] = None,
        log_fn: Optional[LogFn] = None,
        cancel_fn: Optional[CancelFn] = None,
        log_interval: int = 5,
        progress_interval_s: float = 2.0,
    ) -> None:
        try:
            import time as _time
            from stable_baselines3.common.callbacks import BaseCallback

            class _Inner(BaseCallback):
                def __init__(inner_self, **kwargs):
                    super().__init__(verbose=0, **kwargs)
                    inner_self._total = total_timesteps
                    inner_self._progress_fn = progress_fn
                    inner_self._log_fn = log_fn
                    inner_self._cancel_fn = cancel_fn
                    inner_self._log_interval = log_interval
                    inner_self._progress_interval_s = progress_interval_s
                    inner_self._rollout_count = 0
                    inner_self._best_reward = -float("inf")
                    inner_self._last_progress_time = 0.0

                def _on_step(inner_self) -> bool:
                    # Cancel check per step — returning False stops SB3's collect_rollouts
                    if inner_self._cancel_fn and inner_self._cancel_fn():
                        if inner_self._log_fn:
                            inner_self._log_fn("[cancelled] Training cancelled by user.")
                        return False
                    return True

                def _on_rollout_end(inner_self) -> None:
                    inner_self._rollout_count += 1
                    step = inner_self.num_timesteps
                    total = inner_self._total

                    # Compute mean episodic reward and ep length from SB3's ep_info_buffer
                    buf = list(inner_self.model.ep_info_buffer)
                    rewards = [ep["r"] for ep in buf if "r" in ep]
                    lengths = [ep["l"] for ep in buf if "l" in ep]
                    reward_mean = float(np.mean(rewards)) if rewards else 0.0
                    ep_len_mean = float(np.mean(lengths)) if lengths else 0.0
                    if reward_mean > inner_self._best_reward:
                        inner_self._best_reward = reward_mean

                    algo = type(inner_self.model).__name__
                    status = f"{algo} step {step:,}"

                    # Throttle Qt signal emissions — emit at most once per
                    # progress_interval_s to prevent flooding the UI event queue.
                    now = _time.monotonic()
                    if inner_self._progress_fn and (
                        now - inner_self._last_progress_time >= inner_self._progress_interval_s
                    ):
                        inner_self._last_progress_time = now
                        inner_self._progress_fn(
                            step, total,
                            reward_mean, inner_self._best_reward,
                            ep_len_mean, status,
                        )

            self._callback_class = _Inner

        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for SB3ProgressCallback. "
                "Install it with: pip install stable-baselines3"
            ) from exc

    def build(self):
        """Return a new instance of the inner callback class."""
        return self._callback_class()


def resolve_training_device(
    requested: str,
    log_fn: Optional[LogFn] = None,
) -> str:
    """
    Resolve a UI/spec device choice into an SB3/PyTorch device string.

    Supported inputs:
      - "cpu"
      - "gpu" / "cuda"
      - "auto"
    """
    choice = str(requested or "auto").strip().lower()
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if choice in {"gpu", "cuda"}:
        if cuda_available:
            return "cuda"
        if log_fn:
            log_fn("[SB3] GPU requested but CUDA is unavailable 閳?falling back to CPU")
        return "cpu"

    if choice == "cpu":
        return "cpu"

    return "cuda" if cuda_available else "cpu"


# ---------------------------------------------------------------------------
# SB3VisCheckCallback
# ---------------------------------------------------------------------------

class SB3VisCheckCallback:
    """
    SB3 callback that triggers visualization milestone checks at configurable
    step intervals.

    Parameters
    ----------
    vis_start_step:
        Training step at which the first vis check fires.
    vis_interval_steps:
        Repeat interval after the first check.
    vis_check_fn:
        ``fn(model)`` 閳?called at each milestone. May block (e.g. waiting for
        a viewer window to close); training is paused until it returns.
    log_fn:
        Optional log line emitter.
    """

    def __init__(
        self,
        vis_start_step: int,
        vis_interval_steps: int,
        vis_check_fn: VisCheckFn,
        log_fn: Optional[LogFn] = None,
        milestone_steps: Optional[list] = None,
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback

            class _Inner(BaseCallback):
                def __init__(inner_self, **kwargs):
                    super().__init__(verbose=0, **kwargs)
                    inner_self._vis_start = max(1, vis_start_step)
                    inner_self._vis_interval = max(1, vis_interval_steps)
                    inner_self._vis_check_fn = vis_check_fn
                    inner_self._log_fn = log_fn
                    inner_self._milestones = [
                        max(1, int(step))
                        for step in (milestone_steps or [])
                        if int(step) > 0
                    ]
                    inner_self._milestone_idx = 0
                    inner_self._next_vis_step = (
                        inner_self._milestones[0]
                        if inner_self._milestones
                        else max(1, vis_start_step)
                    )

                def _on_step(inner_self) -> bool:
                    step = inner_self.num_timesteps
                    if step >= inner_self._next_vis_step:
                        fired_step = inner_self._next_vis_step
                        if inner_self._milestones:
                            inner_self._milestone_idx += 1
                            if inner_self._milestone_idx < len(inner_self._milestones):
                                inner_self._next_vis_step = inner_self._milestones[inner_self._milestone_idx]
                            else:
                                inner_self._next_vis_step = 10**18
                        else:
                            inner_self._next_vis_step = step + inner_self._vis_interval
                        try:
                            inner_self._vis_check_fn(inner_self.model)
                        except Exception as exc:
                            if inner_self._log_fn:
                                inner_self._log_fn(
                                    f"[vis] Vis check error (non-fatal): {exc}"
                                )
                    return True

            self._callback_class = _Inner

        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for SB3VisCheckCallback."
            ) from exc

    def build(self):
        """Return a new instance of the inner callback class."""
        return self._callback_class()


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

def get_obs_action_dims(spec: "TrainingJobSpec") -> Tuple[int, int]:
    """
    Derive (obs_dim, action_dim) from a compiled TrainingJobSpec.

    When a named obs/action contract preset is active, its declared
    ``obs_dim`` / ``action_dim`` are authoritative. This is required for
    ``resume_sb3`` because the restored policy network input size must match
    the rebuilt environment exactly.

    Observation breakdown for go2 (following DEFAULT_COMPONENT_ORDER):
        base_angular_velocity (3) + projected_gravity (3)
        + joint_positions (12) + joint_velocities (12)
        + last_action (12) + velocity_command (3)
        = 45

    Action dim = number of robot joints.

    For non-go2 robots this function uses a best-effort heuristic;
    ``obs_dim`` can be overridden by setting ``spec.obs_action_config``
    directly if needed.
    """
    _ACTION_DIM_BY_ROBOT: Dict[str, int] = {
        "go2": 12, "go1": 12, "b1": 12, "b2": 12, "h1": 10, "g1": 23,
    }

    preset_name = (
        getattr(getattr(spec, "obs_action_config", None), "contract_preset", "")
        or "custom"
    )
    if preset_name != "custom":
        try:
            from system.training.obs_contracts import get_obs_contract

            contract = get_obs_contract(str(preset_name))
        except ImportError:
            contract = None

        if contract is not None:
            obs_dim = int(contract.get("obs_dim", 0) or 0)
            action_dim = int(contract.get("action_dim", 0) or 0)
            if obs_dim > 0 and action_dim > 0:
                return obs_dim, action_dim

    robot_type = spec.robot_spec.robot_type.lower()
    action_dim = _ACTION_DIM_BY_ROBOT.get(robot_type, 12)
    try:
        from system.training.unitree_gym_env import resolve_obs_dim_from_components
        components = list(getattr(spec.obs_action_config, "obs_components", []) or [])
        frame_stack = int(getattr(spec.obs_action_config, "frame_stack", 1) or 1)
        obs_dim = resolve_obs_dim_from_components(
            components,
            num_joints=action_dim,
            action_dim=action_dim,
            frame_stack=frame_stack,
        )
    except Exception:
        obs_dim = 0
    if obs_dim <= 0:
        # Fixed breakdown: 3 + 3 + action_dim + action_dim + action_dim + 3
        obs_dim = 3 + 3 + action_dim + action_dim + action_dim + 3  # = 45 for 12-dof
    return obs_dim, action_dim


def _activation_fn(name: str):
    import torch.nn as nn

    key = str(name or "relu").strip().lower()
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    return mapping.get(key, nn.ReLU)


def _parse_ent_coef(value: Any, algo_name: str) -> Any:
    raw = str(value if value is not None else "").strip().lower()
    if algo_name == "SAC":
        if raw == "auto":
            return "auto"
        try:
            return float(value)
        except (TypeError, ValueError):
            return "auto"
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# P1: Learning-rate and clip-range schedule builders
# ---------------------------------------------------------------------------

def _make_lr_schedule(base_lr: float, schedule: str) -> Any:
    """
    Return a learning-rate value or callable for SB3.

    SB3 passes ``progress_remaining`` (float, 1.0 → 0.0) to callables.

    Schedules
    ---------
    'constant'  — return ``base_lr`` (a plain float, no callable overhead).
    'linear'    — linearly decay ``base_lr → base_lr * 0.1``.
    'cosine'    — cosine-anneal  ``base_lr → base_lr * 0.1``  (smooth).
    """
    s = str(schedule or "constant").strip().lower()
    if s == "linear":
        def _linear(p: float) -> float:
            return base_lr * (0.1 + 0.9 * p)
        return _linear
    if s == "cosine":
        import math as _math
        def _cosine(p: float) -> float:
            cos_w = 0.5 * (1.0 + _math.cos(_math.pi * (1.0 - p)))
            return base_lr * (0.1 + 0.9 * cos_w)
        return _cosine
    return base_lr   # constant


def _make_clip_range_schedule(start: float, final: float) -> Any:
    """
    Return a clip-range value or callable for SB3 PPO.

    When ``final <= 0`` or ``final >= start`` the schedule is constant
    (returns a plain float).  Otherwise returns a linear-decay callable
    from ``start`` (at progress=1.0) to ``final`` (at progress=0.0).
    """
    s = max(0.0, float(start))
    f = float(final)
    if f <= 0.0 or f >= s:
        return s   # constant — no decay configured
    def _clip(p: float) -> float:
        return f + (s - f) * p
    return _clip


# ---------------------------------------------------------------------------
# P0: Best-model tracker callback (always-on, no EvalConfig required)
# ---------------------------------------------------------------------------

class SB3BestModelTracker:
    """
    Always-on SB3 callback that saves model weights to disk whenever the
    rolling mean episodic reward reaches a new high.

    Unlike ``EvalCallback`` this runs purely on the training ep_info_buffer
    so it adds zero extra simulation overhead.

    Writes inside *save_path*:
        best_model.zip         — SB3 zip checkpoint
        best_model_meta.json   — {"step": int, "reward_mean": float}
    """

    def __init__(self, save_path: "Path", log_fn: Optional[LogFn] = None) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
            import json as _json

            _save = save_path
            _log  = log_fn

            class _Inner(BaseCallback):
                def __init__(inner_self):
                    super().__init__(verbose=0)
                    inner_self._best = -float("inf")

                def _on_rollout_end(inner_self) -> None:
                    buf = list(inner_self.model.ep_info_buffer)
                    rewards = [ep["r"] for ep in buf if "r" in ep]
                    if not rewards:
                        return
                    mean_r = float(np.mean(rewards))
                    if mean_r <= inner_self._best:
                        return
                    inner_self._best = mean_r
                    step = inner_self.num_timesteps
                    try:
                        _save.mkdir(parents=True, exist_ok=True)
                        inner_self.model.save(str(_save / "best_model"))
                        with open(_save / "best_model_meta.json", "w") as fh:
                            _json.dump(
                                {"step": step, "reward_mean": round(mean_r, 4)},
                                fh,
                            )
                        if _log:
                            _log(
                                f"[best] New best at step {step:,}: "
                                f"reward_mean={mean_r:.3f}"
                            )
                    except Exception as exc:
                        if _log:
                            _log(f"[best] Save failed (non-fatal): {exc}")

                def _on_step(inner_self) -> bool:
                    return True

            self._callback_class = _Inner

        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for SB3BestModelTracker."
            ) from exc

    def build(self):
        """Return a new instance of the inner callback class."""
        return self._callback_class()


# ---------------------------------------------------------------------------
# SB3CollapseGuard  — rollback-on-collapse callback
# ---------------------------------------------------------------------------

class SB3CollapseGuard:
    """
    SB3 callback that detects reward collapse and rolls back to the best
    saved checkpoint.

    Collapse condition (checked at every rollout end):
        rolling_mean < best_ever_seen * threshold
        AND consecutive_bad_checks >= patience
        AND rollouts_since_last_rollback >= cooldown_rollouts

    On rollback:
        1. Reload actor/critic weights from best_model.zip.
        2. Multiply all optimizer LRs by ``lr_factor``.
        3. Start a linear LR warmup back to the pre-rollback rate
           over ``recovery_rollouts`` rollout-ends.

    Parameters
    ----------
    best_model_path : Path
        Directory containing ``best_model.zip`` (written by SB3BestModelTracker).
    threshold : float
        Fraction of the best-ever rolling mean below which a check counts as
        "bad".  Default 0.65 — rollback if reward drops below 65 % of best.
    patience : int
        Consecutive bad checks required before triggering rollback.
        Default 5 rollout-ends.
    lr_factor : float
        LR multiplier applied immediately after rollback (0 < lr_factor < 1).
        Default 0.3 — reduce to 30 % while the buffer flushes bad experiences.
    recovery_rollouts : int
        Rollout-ends over which LR linearly returns to the pre-rollback value.
        Default 20.
    cooldown_rollouts : int
        Minimum rollout-ends that must pass before a second rollback can fire.
        Default 30 — prevents rapid re-rollbacks while the buffer is still bad.
    log_fn : callable, optional
        One-argument log sink forwarded from SB3Trainer.
    """

    def __init__(
        self,
        best_model_path: "Path",
        threshold: float = 0.65,
        patience: int = 5,
        lr_factor: float = 0.3,
        recovery_rollouts: int = 20,
        cooldown_rollouts: int = 30,
        log_fn: Optional[LogFn] = None,
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
            import json as _json

            _best_dir  = Path(best_model_path)
            _best_zip  = _best_dir / "best_model.zip"
            _threshold = float(threshold)
            _patience  = max(1, int(patience))
            _lr_factor = max(0.01, min(1.0, float(lr_factor)))
            _recovery  = max(1, int(recovery_rollouts))
            _cooldown  = max(0, int(cooldown_rollouts))
            _log       = log_fn

            class _Inner(BaseCallback):
                def __init__(inner_self):
                    super().__init__(verbose=0)
                    inner_self._best_seen     = -float("inf")
                    inner_self._bad_streak    = 0
                    inner_self._since_rollback = _cooldown  # start ready
                    inner_self._recovery_left = 0
                    inner_self._lr_before_rollback: Optional[float] = None
                    inner_self._rollback_count = 0

                # ----------------------------------------------------------
                # Helpers
                # ----------------------------------------------------------

                def _lr_param_groups(inner_self):
                    seen = set()
                    for obj in (
                        getattr(getattr(inner_self.model, "policy", None), "optimizer", None),
                        getattr(getattr(inner_self.model, "actor", None), "optimizer", None),
                        getattr(getattr(inner_self.model, "critic", None), "optimizer", None),
                        getattr(getattr(inner_self.model, "critic_target", None), "optimizer", None),
                        getattr(inner_self.model, "actor", None),
                        getattr(inner_self.model, "critic", None),
                        getattr(inner_self.model, "critic_target", None),
                    ):
                        if obj is None:
                            continue
                        pgs = getattr(obj, "param_groups", None)
                        if not pgs:
                            continue
                        ident = id(pgs)
                        if ident in seen:
                            continue
                        seen.add(ident)
                        yield pgs

                def _current_lr(inner_self) -> Optional[float]:
                    try:
                        for pgs in inner_self._lr_param_groups():
                            if pgs:
                                lr = pgs[0].get("lr")
                                if lr is not None:
                                    return float(lr)
                    except Exception:
                        pass
                    return None

                def _set_lr(inner_self, lr: float) -> None:
                    try:
                        for pgs in inner_self._lr_param_groups():
                            for pg in pgs:
                                pg["lr"] = lr
                    except Exception:
                        pass

                def _rollback(inner_self) -> None:
                    if not _best_zip.exists():
                        if _log:
                            _log("[guard] No best_model.zip yet — skipping rollback")
                        inner_self._bad_streak = 0
                        return
                    try:
                        from stable_baselines3 import PPO, SAC, TD3
                        algo_name = type(inner_self.model).__name__.upper()
                        algo_cls  = {"PPO": PPO, "SAC": SAC, "TD3": TD3}.get(algo_name)
                        if algo_cls is None:
                            return
                        snap = algo_cls.load(str(_best_zip), device=str(inner_self.model.device))
                        inner_self.model.set_parameters(snap.get_parameters())
                        del snap
                        inner_self._rollback_count += 1
                        inner_self._since_rollback = 0
                        inner_self._bad_streak = 0

                        # LR reduction
                        current_lr = inner_self._current_lr()
                        rollback_lr: Optional[float] = None
                        if current_lr is not None and current_lr > 0:
                            inner_self._lr_before_rollback = current_lr
                            rollback_lr = current_lr * _lr_factor
                            inner_self._set_lr(rollback_lr)
                            inner_self._recovery_left = _recovery
                        if _log:
                            if rollback_lr is not None:
                                _log(
                                    f"[guard] Rollback #{inner_self._rollback_count} at "
                                    f"step {inner_self.num_timesteps:,} — "
                                    f"LR → {rollback_lr:.2e} for "
                                    f"{_recovery} rollouts"
                                )
                            else:
                                _log(
                                    f"[guard] Rollback #{inner_self._rollback_count} at "
                                    f"step {inner_self.num_timesteps:,} — "
                                    f"LR unchanged (unavailable) for "
                                    f"{_recovery} rollouts"
                                )
                    except Exception as exc:
                        if _log:
                            _log(f"[guard] Rollback failed (non-fatal): {exc}")
                        inner_self._bad_streak = 0

                # ----------------------------------------------------------
                # Callbacks
                # ----------------------------------------------------------

                def _on_rollout_end(inner_self) -> None:
                    inner_self._since_rollback += 1

                    # ── LR warm-up after rollback ──────────────────────────
                    if inner_self._recovery_left > 0 and inner_self._lr_before_rollback is not None:
                        inner_self._recovery_left -= 1
                        ratio = 1.0 - inner_self._recovery_left / _recovery
                        restored = inner_self._lr_before_rollback
                        post_rollback = restored * _lr_factor
                        new_lr = post_rollback + (restored - post_rollback) * ratio
                        inner_self._set_lr(new_lr)
                        if inner_self._recovery_left == 0:
                            inner_self._lr_before_rollback = None
                            if _log:
                                _log(f"[guard] LR fully restored to {restored:.2e}")

                    # ── Collapse detection ─────────────────────────────────
                    buf = list(inner_self.model.ep_info_buffer)
                    rewards = [ep["r"] for ep in buf if "r" in ep]
                    if not rewards:
                        return
                    mean_r = float(np.mean(rewards))

                    if mean_r > inner_self._best_seen:
                        inner_self._best_seen = mean_r
                        inner_self._bad_streak = 0
                        return

                    if (inner_self._best_seen > -float("inf")
                            and mean_r < inner_self._best_seen * _threshold):
                        inner_self._bad_streak += 1
                    else:
                        inner_self._bad_streak = 0

                    if (inner_self._bad_streak >= _patience
                            and inner_self._since_rollback >= _cooldown):
                        inner_self._rollback()

                def _on_step(inner_self) -> bool:
                    return True

            self._callback_class = _Inner

        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for SB3CollapseGuard."
            ) from exc

    def build(self):
        return self._callback_class()


# ---------------------------------------------------------------------------
# SB3Trainer
# ---------------------------------------------------------------------------

class SB3Trainer:
    """
    Real training orchestrator for Training Ground Phase D.

    Parameters
    ----------
    spec:
        Fully compiled TrainingJobSpec from the canvas.
    progress_fn:
        Called every rollout with (step, total, reward_mean, best_reward, status).
    log_fn:
        Called with a single log line string.
    cancel_fn:
        Called before each rollout; returning True stops training cleanly.
    """

    def __init__(
        self,
        spec: "TrainingJobSpec",
        progress_fn: Optional[ProgressFn] = None,
        log_fn: Optional[LogFn] = None,
        cancel_fn: Optional[CancelFn] = None,
        vis_check_fn: Optional[VisCheckFn] = None,
    ) -> None:
        self._spec        = spec
        self._progress    = progress_fn
        self._log         = log_fn
        self._cancel      = cancel_fn
        self._vis_check   = vis_check_fn

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self):
        """
        Run training and return the trained SB3 model.

        Returns
        -------
        stable_baselines3.common.base_class.BaseAlgorithm
            The trained model ready for ONNX export.

        Raises
        ------
        ImportError
            If stable-baselines3 or gymnasium are not installed.
        CancelledError
            (Custom) if the cancel_fn returns True before training starts.
        """
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
            from stable_baselines3.common.vec_env import (
                DummyVecEnv,
                SubprocVecEnv,
                VecMonitor,
                VecNormalize,
            )
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 >= 2.0 is required for Phase D training. "
                "Install with: pip install stable-baselines3"
            ) from exc

        from system.training.unitree_gym_env import UnitreeGymEnv

        from system.training.training_spec import resolve_effective_task_terms

        algo   = self._spec.algorithm_config
        robot  = self._spec.robot_spec
        env_cfg = self._spec.env_config
        physics = self._spec.physics_config
        reward_terms, termination_conditions = resolve_effective_task_terms(
            self._spec
        )

        obs_dim, action_dim = get_obs_action_dims(self._spec)

        # Prefer top-level spec.resume_mode (Phase 3-B); fall back to algo_config
        resume_mode = (
            getattr(self._spec, "resume_mode", None)
            or algo.resume_mode
            or "scratch"
        )
        base_asset_path = getattr(self._spec, "base_asset_path", "") or ""

        if self._log:
            self._log(
                f"[SB3] Starting {algo.algorithm} training: "
                f"policy_id_out={algo.policy_id_out}  "
                f"timesteps={algo.total_timesteps:,}  "
                f"obs={obs_dim}  action={action_dim}  "
                f"resume_mode={resume_mode}"
            )

        # ── GPU optimisations ───────────────────────────────────────────
        # Enable TF32 on Ampere/Ada GPUs + cudnn auto-tuner.
        # matmul TF32 gives ~40 % speedup with negligible precision loss for RL.
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32       = True
                torch.backends.cudnn.benchmark        = True   # auto-tune conv kernels
                if self._log:
                    self._log("[SB3] CUDA opts: TF32 matmul + cudnn.benchmark enabled")
        except Exception:
            pass

        # 閳光偓閳光偓 Build environment 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        mjcf_path = robot.mjcf_path or None
        from system.training.unitree_gym_env import (
            _resolve_scene_xml,
            resolve_action_clip_range,
            resolve_obs_components,
            resolve_task_commands,
        )
        _scene_xml = _resolve_scene_xml(
            self._spec.scene_config,
            mjcf_path or "",
            getattr(robot, "robot_type", ""),
        )
        _commands = resolve_task_commands(self._spec.task_config)
        _gravity_z = getattr(self._spec.scene_config, "gravity_z", -9.81)
        _obs_components = resolve_obs_components(self._spec.obs_action_config)
        _frame_stack = int(getattr(self._spec.obs_action_config, "frame_stack", 1) or 1)
        _action_clip = resolve_action_clip_range(self._spec.obs_action_config, env_cfg)
        _action_scale = float(getattr(self._spec.obs_action_config, "action_scale", 1.0) or 1.0)
        _episode_max_steps = int(getattr(env_cfg, "time_limit_override", 0) or 0)
        if _episode_max_steps <= 0:
            _episode_max_steps = max(100, physics.episode_max_steps)
        n_envs = max(1, int(getattr(env_cfg, "n_envs", 1) or 1))
        vec_type = str(getattr(env_cfg, "vec_type", "dummy") or "dummy").strip().lower()

        # Extract all spec references into plain local variables so that
        # _make_env captures only picklable data, not `self` (which holds
        # Qt callbacks that cannot cross a multiprocessing spawn boundary).
        _sim_dt          = getattr(physics, "sim_dt", 0.002)
        _control_dt      = getattr(physics, "control_dt", 0.02)
        _action_type     = getattr(physics, "action_type", "joint_position")
        _joint_config    = dict(getattr(robot, "joint_config", {}) or {})
        _domain_rand_cfg    = self._spec.domain_rand_config
        _ref_motion_cfg     = self._spec.reference_motion_config
        _init_pose_cfg      = self._spec.init_pose_config
        _gated_reward_cfg   = getattr(self._spec, "gated_reward_config", None)
        _command_mode    = getattr(self._spec.task_config, "command_mode", "fixed")
        _curriculum      = bool(getattr(self._spec.task_config, "curriculum", False))
        _success_thr     = float(getattr(self._spec.task_config, "success_threshold", 0.8))
        _trunc_steps     = int(getattr(self._spec.task_config, "truncation_max_steps", 0))
        _curriculum_sched = dict(getattr(self._spec.task_config, "curriculum_schedule", {}) or {})

        def _make_env(seed_offset: int = 0, *, use_domain_rand: bool = True):
            return UnitreeGymEnv(
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_episode_steps=_episode_max_steps,
                mjcf_path=mjcf_path,
                scene_xml_path=_scene_xml,
                sim_dt=_sim_dt,
                control_dt=_control_dt,
                gravity_z=_gravity_z,
                commands=_commands,
                reward_terms=reward_terms,
                termination_conditions=termination_conditions,
                obs_components=_obs_components,
                frame_stack=_frame_stack,
                action_type=_action_type,
                action_scale=_action_scale,
                action_clip=_action_clip,
                joint_config=_joint_config,
                domain_rand_config=(_domain_rand_cfg if use_domain_rand else None),
                reference_motion_config=_ref_motion_cfg,
                init_pose_config=_init_pose_cfg,
                gated_reward_config=_gated_reward_cfg,
                command_mode=_command_mode,
                curriculum=_curriculum,
                success_threshold=_success_thr,
                truncation_max_steps=_trunc_steps,
                curriculum_schedule=_curriculum_sched,
            )

        env_fns = [lambda idx=i: _make_env(seed_offset=idx, use_domain_rand=True) for i in range(n_envs)]
        if vec_type == "subproc" and n_envs > 1:
            try:
                vec_env = SubprocVecEnv(env_fns)
            except Exception as exc:
                if self._log:
                    self._log(f"[SB3] SubprocVecEnv unavailable ({exc}); falling back to DummyVecEnv")
                vec_env = DummyVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

        if bool(getattr(env_cfg, "enable_monitor", True)):
            vec_env = VecMonitor(vec_env)
        if bool(getattr(env_cfg, "obs_normalize", False)) or bool(getattr(env_cfg, "reward_normalize", False)):
            vec_env = VecNormalize(
                vec_env,
                norm_obs=bool(getattr(env_cfg, "obs_normalize", False)),
                norm_reward=bool(getattr(env_cfg, "reward_normalize", False)),
                clip_obs=float(getattr(env_cfg, "clip_obs", 10.0) or 10.0),
                clip_reward=float(getattr(env_cfg, "clip_reward", 10.0) or 10.0),
            )

        if self._log:
            self._log(
                f"[SB3] Environment built: n_envs={n_envs}  obs_dim={obs_dim}  "
                f"vec_type={vec_type}  frame_stack={_frame_stack}"
            )

        # 閳光偓閳光偓 Build model 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        algo_name    = algo.algorithm.upper()
        total_steps  = max(1, algo.total_timesteps)
        learning_rate = algo.learning_rate
        batch_size    = algo.batch_size
        gamma         = algo.gamma
        seed          = algo.seed
        device        = resolve_training_device(algo.device, self._log)

        policy_kwargs: Dict[str, Any] = {}
        if algo.hidden_dims:
            policy_kwargs["net_arch"] = list(algo.hidden_dims)
        policy_kwargs["activation_fn"] = _activation_fn(getattr(algo, "activation", "relu"))
        ent_coef = _parse_ent_coef(getattr(algo, "ent_coef", 0.0), algo_name)

        # ── P1: LR schedule + clip-range decay ──────────────────────────
        lr_sched_name  = str(getattr(algo, "lr_schedule", "cosine") or "cosine").lower()
        clip_range_start = float(getattr(algo, "clip_range",       0.2)  or 0.2)
        clip_range_end   = float(getattr(algo, "clip_range_final", 0.05) or 0.05)

        lr_fn         = _make_lr_schedule(learning_rate, lr_sched_name)
        clip_range_fn = _make_clip_range_schedule(clip_range_start, clip_range_end)

        if self._log:
            lr_tag = (
                f"{lr_sched_name} ({learning_rate:.2e} -> {learning_rate*0.1:.2e})"
                if lr_sched_name != "constant"
                else f"constant ({learning_rate:.2e})"
            )
            cr_tag = (
                f"{clip_range_start} -> {clip_range_end}"
                if callable(clip_range_fn)
                else f"constant ({clip_range_start})"
            )
            self._log(f"[SB3] LR schedule: {lr_tag}  |  clip_range: {cr_tag}")

        # ── Build or resume model ────────────────────────────────────────
        gradient_steps = int(getattr(algo, "gradient_steps", 1) or 1)
        n_epochs       = max(1, int(getattr(algo, "n_epochs", 10) or 10))

        model = self._build_or_resume_model(
            algo_name=algo_name,
            algo_cls_ppo=PPO,
            algo_cls_sac=SAC,
            vec_env=vec_env,
            learning_rate=lr_fn,
            n_steps=getattr(algo, "n_steps", 2048),
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=getattr(algo, "gae_lambda", 0.95),
            buffer_size=getattr(algo, "buffer_size", 1_000_000),
            learning_starts=getattr(algo, "learning_starts", 100),
            ent_coef=ent_coef,
            total_steps=total_steps,
            n_envs=n_envs,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs or None,
            resume_mode=resume_mode,
            base_asset_path=base_asset_path,
            gradient_steps=gradient_steps,
            n_epochs=n_epochs,
            clip_range=clip_range_fn,
        )

        # Silence SB3's internal stdout logger — the rollout table is captured
        # via our callback; printing it to CMD is redundant and noisy.
        try:
            from stable_baselines3.common.logger import configure as _sb3_cfg
            model.set_logger(_sb3_cfg(folder=None, format_strings=[]))
        except Exception:
            pass

        if self._log:
            self._log(f"[SB3] Model built: {algo_name}  seed={seed}  device={device}  "
                      f"gradient_steps={gradient_steps}  n_epochs={n_epochs}")

        # 閳光偓閳光偓 Callbacks 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        callbacks = []
        if self._progress or self._log or self._cancel:
            cb_builder = SB3ProgressCallback(
                total_timesteps=total_steps,
                progress_fn=self._progress,
                log_fn=self._log,
                cancel_fn=self._cancel,
            )
            callbacks.append(cb_builder.build())

        # Vis check callback 閳?fires at milestone steps when configured
        if self._vis_check is not None and self._spec.vis_check_config is not None:
            vis_cfg = self._spec.vis_check_config
            milestone_steps = None
            if vis_cfg.trigger_mode == "episode_split":
                # Anchor the final vis check at the end of training and place
                # the earlier milestones at equal spacing before it.
                n_checks = max(1, vis_cfg.num_vis_checks)
                milestone_steps = sorted({
                    max(1, int(round(total_steps * idx / n_checks)))
                    for idx in range(1, n_checks + 1)
                })
                vc_start = milestone_steps[0]
                vc_interval = (
                    max(1, milestone_steps[1] - milestone_steps[0])
                    if len(milestone_steps) > 1 else max(1, total_steps)
                )
                if self._log:
                    self._log(
                        f"[vis] episode_split milestones: {', '.join(f'{s:,}' for s in milestone_steps)}"
                    )
            else:
                vc_start = max(1, vis_cfg.vis_start_step)
                vc_interval = max(1, vis_cfg.vis_interval_steps)
            vis_cb_builder = SB3VisCheckCallback(
                vis_start_step=vc_start,
                vis_interval_steps=vc_interval,
                vis_check_fn=self._vis_check,
                log_fn=self._log,
                milestone_steps=milestone_steps,
            )
            callbacks.append(vis_cb_builder.build())

        checkpoint_interval = max(0, int(getattr(algo, "checkpoint_interval", 0) or 0))
        checkpoint_dir = Path(os.getcwd()) / "training_checkpoints" / (algo.policy_id_out or "trained_policy")

        # ── P0: Always-on best-model tracker ────────────────────────────
        # Saves best_model.zip + best_model_meta.json whenever rolling reward
        # improves.  export_bundle() will prefer this over the final model.
        try:
            best_tracker = SB3BestModelTracker(
                save_path=checkpoint_dir / "best_model",
                log_fn=self._log,
            )
            callbacks.append(best_tracker.build())
        except Exception as _bt_exc:
            if self._log:
                self._log(f"[best] Tracker unavailable (non-fatal): {_bt_exc}")

        # ── Collapse guard (optional) ─────────────────────────────────────
        _cg_enabled = bool(getattr(algo, "collapse_guard", True))
        if _cg_enabled:
            try:
                guard = SB3CollapseGuard(
                    best_model_path=checkpoint_dir / "best_model",
                    threshold=float(getattr(algo, "collapse_threshold", 0.65)),
                    patience=int(getattr(algo, "collapse_patience", 5)),
                    lr_factor=float(getattr(algo, "collapse_lr_factor", 0.3)),
                    recovery_rollouts=20,
                    cooldown_rollouts=30,
                    log_fn=self._log,
                )
                callbacks.append(guard.build())
            except Exception as _cg_exc:
                if self._log:
                    self._log(f"[guard] CollapseGuard unavailable (non-fatal): {_cg_exc}")

        if checkpoint_interval > 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(1, checkpoint_interval // max(1, n_envs)),
                    save_path=str(checkpoint_dir),
                    name_prefix="checkpoint",
                    save_vecnormalize=bool(getattr(env_cfg, "obs_normalize", False) or getattr(env_cfg, "reward_normalize", False)),
                )
            )

        if self._spec.eval_config is not None:
            eval_cfg = self._spec.eval_config

            class _PeriodicEvalCallback(BaseCallback):
                def __init__(inner_self):
                    super().__init__(verbose=0)
                    inner_self._next_eval = max(1, int(getattr(eval_cfg, "eval_interval", 50_000) or 50_000))
                    inner_self._best_mean = -float("inf")

                def _on_step(inner_self) -> bool:
                    if inner_self.num_timesteps < inner_self._next_eval:
                        return True
                    result = self.evaluate(
                        inner_self.model,
                        step_tag=f"step_{inner_self.num_timesteps}",
                    )
                    mean_reward = float(result.get("mean_reward", 0.0))
                    if bool(getattr(eval_cfg, "save_best_model", True)) and mean_reward >= inner_self._best_mean:
                        inner_self._best_mean = mean_reward
                        best_dir = checkpoint_dir / "best_model"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        inner_self.model.save(str(best_dir / "best_model"))
                    inner_self._next_eval += max(1, int(getattr(eval_cfg, "eval_interval", 50_000) or 50_000))
                    return True

            callbacks.append(_PeriodicEvalCallback())

        # 閳光偓閳光偓 Train 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks or None,
            reset_num_timesteps=True,
            progress_bar=False,
        )

        vec_env.close()

        if self._log:
            self._log(f"[SB3] Training loop complete: {model.num_timesteps:,} steps")

        return model

    # ------------------------------------------------------------------
    # Internal: model construction / resume
    # ------------------------------------------------------------------

    def _build_or_resume_model(
        self,
        *,
        algo_name: str,
        algo_cls_ppo,
        algo_cls_sac,
        vec_env,
        learning_rate: float,
        n_steps: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        buffer_size: int,
        learning_starts: int,
        ent_coef: Any,
        total_steps: int,
        n_envs: int,
        seed: int,
        device: str,
        policy_kwargs,
        resume_mode: str,
        base_asset_path: str,
        gradient_steps: int = 1,
        n_epochs: int = 10,
        clip_range: Any = 0.2,
    ):
        """
        Build a new model or resume from a base asset checkpoint.

        resume_mode:
          'scratch'         閳?build a new model from random weights (default)
          'resume_sb3'      閳?load SB3 checkpoint (.zip); continue training
                             from saved weights + optimizer state
          'warm_start_actor' 閳?build a new model then copy actor (policy)
                              parameters only from checkpoint; optimizer starts
                              fresh, replay buffer is NOT restored
        """
        if resume_mode == "resume_sb3" and base_asset_path:
            if self._log:
                self._log(
                    f"[SB3] resume_sb3 閳?loading checkpoint: {base_asset_path}"
                )
            if algo_name == "PPO":
                model = algo_cls_ppo.load(
                    base_asset_path,
                    env=vec_env,
                    device=device,
                )
            elif algo_name == "SAC":
                model = algo_cls_sac.load(
                    base_asset_path,
                    env=vec_env,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Unsupported algorithm '{algo_name}' for resume_sb3. "
                    "Phase D supports: PPO, SAC"
                )
            if self._log:
                self._log(
                    f"[SB3] Checkpoint loaded 閳?"
                    f"base_asset={base_asset_path}  "
                    f"load_mode=resume_sb3  replay_buffer=not_restored"
                )
            return model

        if resume_mode == "warm_start_actor" and base_asset_path:
            if self._log:
                self._log(
                    f"[SB3] warm_start_actor 閳?loading actor weights only: "
                    f"{base_asset_path}"
                )
            # Build a fresh model first (fresh optimizer + replay buffer)
            model = self._build_fresh_model(
                algo_name=algo_name,
                algo_cls_ppo=algo_cls_ppo,
                algo_cls_sac=algo_cls_sac,
                vec_env=vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                ent_coef=ent_coef,
                total_steps=total_steps,
                n_envs=n_envs,
                seed=seed,
                device=device,
                policy_kwargs=policy_kwargs,
                gradient_steps=gradient_steps,
                n_epochs=n_epochs,
                clip_range=clip_range,
            )
            try:
                if algo_name == "SAC":
                    src = algo_cls_sac.load(base_asset_path, device=device)
                else:
                    src = algo_cls_ppo.load(base_asset_path, device=device)
                copied = self._copy_actor_weights_only(
                    src, model, algo_name, log_fn=self._log
                )
                if self._log:
                    self._log(
                        f"[SB3] Actor weights copied ({copied} tensors) 閳?"
                        f"base_asset={base_asset_path}  "
                        f"load_mode=warm_start_actor  "
                        f"replay_buffer=not_restored"
                    )
            except Exception as exc:
                if self._log:
                    self._log(
                        f"[SB3] warm_start_actor failed ({exc}); "
                        "falling back to scratch"
                    )
            return model

        # scratch (default)
        return self._build_fresh_model(
            algo_name=algo_name,
            algo_cls_ppo=algo_cls_ppo,
            algo_cls_sac=algo_cls_sac,
            vec_env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            ent_coef=ent_coef,
            total_steps=total_steps,
            n_envs=n_envs,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
            gradient_steps=gradient_steps,
            n_epochs=n_epochs,
            clip_range=clip_range,
        )

    @staticmethod
    def _copy_actor_weights_only(
        src_model,
        target_model,
        algo_name: str,
        log_fn: Optional[LogFn] = None,
    ) -> int:
        """
        Copy only the actor (policy) network weights from *src_model* to
        *target_model* using PyTorch state-dict surgery.

        Key selection by algorithm:
          SAC  閳?policy keys starting with ``"actor."``
                 (critic.*, critic_target.*, log_std_init are excluded)
          PPO  閳?policy keys NOT starting with ``"value_net."``
                 (mlp_extractor + action_net + log_std; value head excluded)
          other 閳?all policy keys (best-effort, shape-guarded)

        Only keys that exist in the target state dict with matching tensor
        shapes are transferred.  Optimizer state is never touched.

        Returns the number of tensors successfully copied.
        """
        src_state  = src_model.policy.state_dict()
        tgt_state  = target_model.policy.state_dict()

        if algo_name == "SAC":
            candidate = {k: v for k, v in src_state.items()
                         if k.startswith("actor.")}
        elif algo_name == "PPO":
            candidate = {k: v for k, v in src_state.items()
                         if not k.startswith("value_net.")}
        else:
            candidate = dict(src_state)

        updates: dict = {}
        skipped: list = []
        for k, v in candidate.items():
            if k in tgt_state:
                if tgt_state[k].shape == v.shape:
                    updates[k] = v
                else:
                    skipped.append(f"{k}: src{list(v.shape)} tgt{list(tgt_state[k].shape)}")
            else:
                skipped.append(f"{k}: not in target")

        if skipped and log_fn:
            log_fn(
                f"[SB3] warm_start_actor: skipped {len(skipped)} tensors "
                f"(shape mismatch or absent): {skipped[:3]}{'...' if len(skipped) > 3 else ''}"
            )

        tgt_state.update(updates)
        target_model.policy.load_state_dict(tgt_state)
        return len(updates)

    def _build_fresh_model(
        self,
        *,
        algo_name: str,
        algo_cls_ppo,
        algo_cls_sac,
        vec_env,
        learning_rate: float,
        n_steps: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        buffer_size: int,
        learning_starts: int,
        ent_coef: Any,
        total_steps: int,
        n_envs: int,
        seed: int,
        device: str,
        policy_kwargs,
        gradient_steps: int = 1,
        n_epochs: int = 10,
        clip_range: Any = 0.2,
    ):
        if algo_name == "PPO":
            n_steps = max(64, n_steps)
            return algo_cls_ppo(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=min(batch_size, n_steps * n_envs),
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=float(ent_coef),
                n_epochs=n_epochs,
                clip_range=clip_range,
                verbose=0,
                seed=seed,
                device=device,
                policy_kwargs=policy_kwargs,
            )
        elif algo_name == "SAC":
            # Auto-tune gradient_steps for GPU saturation.
            # User setting of 1 almost never saturates a GPU — match n_envs so
            # each collected batch triggers n_envs gradient updates.
            # Users can override by setting gradient_steps explicitly != 1.
            effective_gradient_steps = gradient_steps
            if gradient_steps == 1 and n_envs > 1:
                effective_gradient_steps = n_envs
                if self._log:
                    self._log(
                        f"[SB3] SAC gradient_steps auto-tuned: 1 → {n_envs} "
                        f"(= n_envs; set gradient_steps explicitly to override)"
                    )
            # Batch size: warn if too small for GPU
            effective_batch = batch_size
            if device.startswith("cuda") and batch_size < 512 and self._log:
                self._log(
                    f"[SB3] Note: batch_size={batch_size} is small for GPU. "
                    "Consider 1024+ for better throughput (AlgorithmConfig → batch_size)."
                )
            return algo_cls_sac(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=learning_rate,
                batch_size=effective_batch,
                gamma=gamma,
                buffer_size=min(buffer_size, total_steps + 10_000),
                learning_starts=min(learning_starts, total_steps // 2),
                ent_coef=ent_coef,
                gradient_steps=effective_gradient_steps,
                verbose=0,
                seed=seed,
                device=device,
                policy_kwargs=policy_kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported algorithm '{algo_name}'. "
                "Phase D supports: PPO, SAC"
            )

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model, step_tag: str = "final") -> Dict[str, Any]:
        """
        Run post-training evaluation episodes and return summary metrics.

        Driven by ``spec.eval_config``; returns a best-effort result dict
        even when eval_config is absent (uses safe defaults).

        Parameters
        ----------
        model:
            Trained SB3 model returned by :meth:`train`.

        Returns
        -------
        dict with keys:
            eval_episodes (int), mean_reward (float), std_reward (float),
            success_rate (float), success_threshold (float), passed (bool)
        """
        from system.training.unitree_gym_env import (
            UnitreeGymEnv,
            resolve_action_clip_range,
            resolve_obs_components,
            resolve_task_commands,
        )

        eval_cfg = self._spec.eval_config
        n_episodes  = (eval_cfg.eval_episodes   if eval_cfg else 5)
        deterministic = (eval_cfg.deterministic if eval_cfg else True)
        threshold   = (eval_cfg.success_threshold if eval_cfg else 0.8)

        from system.training.training_spec import resolve_effective_task_terms

        robot  = self._spec.robot_spec
        physics = self._spec.physics_config
        reward_terms, termination_conditions = resolve_effective_task_terms(
            self._spec
        )
        obs_dim, action_dim = get_obs_action_dims(self._spec)

        from system.training.unitree_gym_env import _resolve_scene_xml
        _eval_scene_xml = _resolve_scene_xml(
            self._spec.scene_config,
            robot.mjcf_path or "",
            getattr(robot, "robot_type", ""),
        )
        _eval_commands = resolve_task_commands(self._spec.task_config)
        _eval_gravity_z = getattr(self._spec.scene_config, "gravity_z", -9.81)
        _eval_obs_components = resolve_obs_components(self._spec.obs_action_config)
        _eval_frame_stack = int(getattr(self._spec.obs_action_config, "frame_stack", 1) or 1)
        _eval_action_clip = resolve_action_clip_range(self._spec.obs_action_config, self._spec.env_config)
        _eval_action_scale = float(getattr(self._spec.obs_action_config, "action_scale", 1.0) or 1.0)
        _use_domain_rand = not bool(getattr(self._spec.env_config, "eval_disable_rand", True))
        env = UnitreeGymEnv(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_episode_steps=max(100, int(getattr(self._spec.env_config, "time_limit_override", 0) or physics.episode_max_steps)),
            mjcf_path=robot.mjcf_path or None,
            scene_xml_path=_eval_scene_xml,
            sim_dt=getattr(physics, "sim_dt", 0.002),
            control_dt=getattr(physics, "control_dt", 0.02),
            gravity_z=_eval_gravity_z,
            commands=_eval_commands,
            reward_terms=reward_terms,
            termination_conditions=termination_conditions,
            obs_components=_eval_obs_components,
            frame_stack=_eval_frame_stack,
            action_type=getattr(physics, "action_type", "joint_position"),
            action_scale=_eval_action_scale,
            action_clip=_eval_action_clip,
            joint_config=getattr(robot, "joint_config", {}),
            domain_rand_config=(self._spec.domain_rand_config if _use_domain_rand else None),
            reference_motion_config=self._spec.reference_motion_config,
            init_pose_config=self._spec.init_pose_config,
            command_mode=getattr(self._spec.task_config, "command_mode", "fixed"),
            curriculum=getattr(self._spec.task_config, "curriculum", False),
            success_threshold=getattr(self._spec.task_config, "success_threshold", 0.8),
            truncation_max_steps=getattr(self._spec.task_config, "truncation_max_steps", 0),
            curriculum_schedule=getattr(self._spec.task_config, "curriculum_schedule", {}),
        )

        if self._log:
            self._log(
                f"[eval] Starting post-training evaluation: "
                f"episodes={n_episodes}  deterministic={deterministic}  "
                f"threshold={threshold}"
            )

        episode_rewards: list = []
        saved_videos: list = []
        frame_capture_enabled = bool(eval_cfg.record_video) if eval_cfg else False
        frame_capture_dir = Path(
            getattr(eval_cfg, "video_dir", "") or (Path(os.getcwd()) / "training_eval_videos")
        )
        if frame_capture_enabled:
            frame_capture_dir.mkdir(parents=True, exist_ok=True)
        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            captured_frames = []
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += float(reward)
                if frame_capture_enabled:
                    frame = env.render_frame()
                    if frame is not None:
                        captured_frames.append(frame)
                done = terminated or truncated
            episode_rewards.append(ep_reward)
            if frame_capture_enabled and captured_frames:
                video_path = frame_capture_dir / f"{step_tag}_episode_{ep + 1}.gif"
                try:
                    import imageio.v2 as imageio
                    imageio.mimsave(video_path, captured_frames, duration=max(env._control_dt, 0.02))
                    saved_videos.append(str(video_path))
                except Exception as exc:
                    if self._log:
                        self._log(f"[eval] Video export skipped for episode {ep + 1}: {exc}")

            if self._cancel and self._cancel():
                if self._log:
                    self._log("[eval] Evaluation cancelled by user.")
                break

        env.close()

        mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        std_r  = float(np.std(episode_rewards))  if episode_rewards else 0.0
        above  = sum(1 for r in episode_rewards if r >= threshold * mean_r)
        success_rate = above / max(len(episode_rewards), 1)
        passed = mean_r >= threshold

        if self._log:
            self._log(
                f"[eval] Results: episodes={len(episode_rewards)}  "
                f"mean_reward={mean_r:.3f}  std={std_r:.3f}  "
                f"success_rate={success_rate:.2%}  passed={passed}"
            )

        return {
            "eval_episodes":    len(episode_rewards),
            "mean_reward":      mean_r,
            "std_reward":       std_r,
            "success_rate":     success_rate,
            "success_threshold": threshold,
            "passed":           passed,
            "videos":           saved_videos,
        }


