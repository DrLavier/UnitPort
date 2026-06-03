# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SB3 trainer — PPO / SAC / TD3 against :class:`GenericMujocoEnv`.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/sb3_trainer.py``
(2k lines), narrowed to the Stage 7 acceptance contract:

    train_sb3(spec) → (model, info)

Stage 7 v1 scope:
    * Build ``DummyVecEnv`` over ``GenericMujocoEnv`` (subprocess
      vectorisation lands in Stage 10's launcher).
    * Optional ``VecNormalize`` wrapper driven by ``spec.env``.
    * Construct PPO / SAC / TD3 model honouring the SB3 hyperparam
      surface in ``AlgorithmConfig`` (learning_rate, batch_size, gamma,
      n_steps, gae_lambda, ent_coef, …).
    * Run ``model.learn(total_timesteps)`` with optional callback.
    * Pass-through for cancel via the standard SB3 callback mechanism;
      Stage 10 wraps this whole module in a subprocess + IPC bridge.

Out of Stage 7:
    * Best-model tracker / collapse guard / vis-check callbacks (DEMO
      had three full sub-classes; deferred until end-to-end is proven).
    * Curriculum schedule wiring (deferred).
    * VecNormalize stats round-trip into the manifest (Stage 7 still
      writes a vanilla manifest; Stage 12 acceptance test will surface
      the gap).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from unitport_sdk import log_info, log_success, log_warning

from application.training.training_spec import validate_stage_schedule_dict_h0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Returned by :func:`train_sb3`."""

    model: Any                  # stable_baselines3.PPO/SAC/TD3
    vec_env: Any                # SB3 VecEnv (raw) — caller closes
    obs_dim: int
    action_dim: int
    algorithm: str              # "PPO" / "SAC" / "TD3"
    total_timesteps: int
    seed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _activation_fn(name: str):
    import torch.nn as nn
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }.get(str(name or "relu").strip().lower(), nn.ReLU)


def _parse_ent_coef(value: Any, algo: str) -> Any:
    """``"auto"`` for SAC/TD3; float for PPO."""
    if value is None:
        return "auto" if algo in ("SAC", "TD3") else 0.0
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return "auto" if algo in ("SAC", "TD3") else 0.0
        try:
            return float(v)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_target_entropy(value: Any) -> Any:
    """SAC target entropy: ``"auto"`` (SB3 default) or a float literal."""
    if value is None:
        return "auto"
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "" or v == "auto":
            return "auto"
        try:
            return float(v)
        except ValueError:
            return "auto"
    try:
        return float(value)
    except (TypeError, ValueError):
        return "auto"


def _resolve_device(pref: str) -> str:
    pref = (pref or "auto").strip().lower()
    if pref == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return pref


def _build_lr_schedule(lr: float, schedule: str) -> Any:
    """Return either a float (constant) or a SB3-compatible ``Callable``.

    SB3 calls the callable with ``progress_remaining`` ∈ [1.0, 0.0] each
    rollout. ``"linear"`` decays to 0 over training; ``"cosine"`` follows a
    half-cosine from ``lr`` to 0; ``"constant"`` (or anything else) keeps
    the float verbatim so SB3 takes its hot path.
    """
    lr = float(lr)
    s = (schedule or "constant").strip().lower()
    if s == "linear":
        def _linear(progress_remaining: float) -> float:
            return lr * float(progress_remaining)
        return _linear
    if s == "cosine":
        import math

        def _cosine(progress_remaining: float) -> float:
            progress = 1.0 - float(progress_remaining)
            return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return _cosine
    return lr


def _resolve_buffer_size(a) -> int:
    """Off-policy replay buffer sizing — honors ``buffer_size_mode``.

    ``"manual"`` uses the user-supplied ``buffer_size`` verbatim.
    ``"auto"`` derives a buffer capped at 1M but no larger than the planned
    training budget, so short smoke runs don't allocate gigabytes upfront.
    """
    mode = (getattr(a, "buffer_size_mode", "auto") or "auto").strip().lower()
    if mode == "manual":
        return int(a.buffer_size or 1_000_000)
    total = int(getattr(a, "total_timesteps", 1_000_000) or 1_000_000)
    return max(10_000, min(1_000_000, total))


def _warn_unimplemented_features(spec) -> None:
    """Log loud warnings for canvas fields the SB3 path does not honor yet.

    Silent no-ops are the canvas-audit anti-pattern this is built to kill:
    every UI field that does not reach the SB3 environment must either
    work or fail visibly so the operator knows their setting was ignored.
    """
    oa = getattr(spec, "obs_action", None)
    if oa is not None:
        # contract_preset warn removed (2026-06): the field is gone (obs is
        # il_terms-driven for both backends); the old warn also falsely claimed
        # the exporter honored it. action_type below is belt-and-suspenders —
        # spec_validator._check_sb3_unsupported already fail-louds on it.
        action_type = (getattr(oa, "action_type", "joint_position") or "").lower()
        if action_type and action_type not in ("joint_position", "position"):
            log_warning(
                f"[sb3_trainer] obs_action_config.action_type={action_type!r} "
                f"is not supported by GenericMujocoEnv; the env always runs "
                f"PD-driven joint-position control."
            )

    physics = getattr(spec, "physics", None)
    if physics is not None:
        phys_action_type = (getattr(physics, "action_type", "joint_position") or "").lower()
        if phys_action_type and phys_action_type not in ("joint_position", "position"):
            log_warning(
                f"[sb3_trainer] physics_config.action_type={phys_action_type!r} "
                f"is not supported by GenericMujocoEnv; the env always runs "
                f"PD-driven joint-position control."
            )

    # domain_rand is honored by GenericMujocoEnv directly (mass / friction /
    # motor_strength / joint_damping / obs_noise / push). No warning is
    # emitted on opt-in; the env-side log line on first reset confirms
    # the configured ranges took effect.

    sched = getattr(spec, "stage_schedule", None)
    if sched is not None:
        # H0 lock still enforced; the 2-stage multigated_reward cross-fade is
        # now CONSUMED by the SB3 path (P6 — _RewardStageCallback ramps the env
        # reward blend across max_step_stage0 + blend_steps). No warn here.
        validate_stage_schedule_dict_h0(asdict(sched))

    # training_motion (per-item command ranges, gait, command_curriculum) and
    # task_config.command_mode are now CONSUMED by the SB3 env (P3) — gait is
    # gated fail-loud at validation (F2). No warn-and-continue here anymore.

    task = getattr(spec, "task", None)
    if task is not None:
        task_type = (getattr(task, "task_type", "velocity_tracking") or "").lower()
        if task_type and task_type != "velocity_tracking":
            log_warning(
                f"[sb3_trainer] task_config.task_type={task_type!r} is not "
                f"implemented; the env still runs the velocity-tracking task."
            )


def _load_existing_model(algo: str, ckpt_path: str, vec_env, *, device: str):
    """Instantiate one of PPO/SAC/TD3 from a saved ``.zip``/``.pt`` file.

    Returns the loaded model bound to ``vec_env``. Caller is responsible
    for deciding whether to keep loaded hyperparameters (resume) or
    overwrite them (warm_start_actor).
    """
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(ckpt_path, env=vec_env, device=device)
    if algo == "SAC":
        from stable_baselines3 import SAC
        return SAC.load(ckpt_path, env=vec_env, device=device)
    if algo == "TD3":
        from stable_baselines3 import TD3
        return TD3.load(ckpt_path, env=vec_env, device=device)
    raise ValueError(f"_load_existing_model: unsupported algorithm {algo!r}")


# ---------------------------------------------------------------------------
# Env factory (DummyVecEnv over GenericMujocoEnv)
# ---------------------------------------------------------------------------


import gymnasium as _gym


class _ActionClipWrapper(_gym.ActionWrapper):
    """Outer safety action clip driven by ``env_assembler.action_clip_range``.

    Sits OUTSIDE the obs_action contract clip applied inside
    :class:`GenericMujocoEnv`. The contract clip is part of the deploy bundle
    (so the runtime applier reproduces it); this wrapper is a training-only
    guard that lets users tighten exploration without touching the contract.

    MUST be a real ``gymnasium.ActionWrapper`` subclass: gymnasium's
    ``Monitor`` / wrapper chain assert ``isinstance(env, gymnasium.Env)`` at
    construction (gymnasium >= 1.x), so a plain duck-typed wrapper is rejected
    with ``AssertionError: Expected env to be a gymnasium.Env``. ActionWrapper
    forwards reset/step/render/close/spaces/metadata and lets us transform the
    action via :meth:`action`.
    """

    def __init__(self, env, clip_range: float):
        super().__init__(env)
        self._clip_range = float(clip_range)

    def action(self, action):
        import numpy as _np
        return _np.clip(action, -self._clip_range, self._clip_range)


def _make_env_factory(
    spec,
    *,
    env_index: int,
    base_seed: int,
    render_mode: Optional[str] = None,
) -> Callable:
    """Return a thunk that builds one :class:`GenericMujocoEnv` instance.

    Each VecEnv slot gets a deterministic per-index seed so subprocess
    vectorisation (Stage 10) keeps episodes reproducible.
    """
    from application.training.envs import make_env

    def _thunk():
        return make_env(spec, seed=base_seed + env_index, render_mode=render_mode)

    return _thunk


def build_vec_env(
    spec,
    *,
    n_envs: int,
    seed: int,
    vec_type: str = "dummy",
    render_mode: Optional[str] = None,
    video_record_dir: Optional[Path] = None,
):
    """Construct a (Dummy|Subproc)VecEnv with optional VecNormalize.

    ``vec_type``:
        * ``"subproc"`` (+ ``n_envs > 1``) → :class:`SubprocVecEnv` with the
          ``"spawn"`` start method, so each env steps on its own CPU core in
          a separate process. ``"spawn"`` is mandatory on Windows and the
          safe choice cross-platform (fork + the torch/mujoco thread pools is
          unsafe). The launcher (``sb3_entry``) provides the required
          ``if __name__ == "__main__"`` guard; the env factory closures + the
          compiled ``spec`` are cloudpickled across the spawn boundary (the
          env builds from the spec alone — it touches no registry/global).
        * ``"dummy"`` or ``n_envs == 1`` → :class:`DummyVecEnv` (in-process;
          required for eval/video where rendering must stay in this process).

    ``vec_type`` is normally resolved by
    :func:`application.training.envs.auto_parallelism.resolve_parallelism`
    (hardware-adaptive) before this is called.

    Wrapper stack (outermost-first as the policy sees them):
        VecFrameStack (if ``obs_action.frame_stack > 1``)
          → VecNormalize (if ``env.obs_normalize`` / ``reward_normalize``)
          → Dummy|SubprocVecEnv
          → Monitor
          → GenericMujocoEnv
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    n_envs = max(1, int(n_envs))

    # Outer safety clip from env_assembler.action_clip_range. This sits OUTSIDE
    # the obs_action contract clip (which is applied inside GenericMujocoEnv.step
    # and round-trips through the deploy bundle). The env_assembler value is a
    # training-only guard — users tightening it produces tighter exploration
    # without affecting the deploy contract. A value of 0 disables the wrapper.
    env_cfg = getattr(spec, "env", None)
    action_clip_range = float(getattr(env_cfg, "action_clip_range", 1.0) or 0.0) if env_cfg is not None else 0.0

    def _wrap(thunk: Callable, *, video_dir: Optional[Path] = None) -> Callable:
        # Monitor populates ``info["episode"]`` (and thus ``model.ep_info_buffer``)
        # with per-episode return ``r`` + length ``l``. The progress callback in
        # sb3_entry.py reads ``l`` to emit ``ep_len_mean``.
        def _factory():
            env = thunk()
            if action_clip_range > 0.0:
                env = _ActionClipWrapper(env, action_clip_range)
            if video_dir is not None:
                try:
                    from gymnasium.wrappers import RecordVideo
                    video_dir.mkdir(parents=True, exist_ok=True)
                    env = RecordVideo(
                        env,
                        video_folder=str(video_dir),
                        episode_trigger=lambda ep: True,
                        name_prefix="eval",
                        disable_logger=True,
                    )
                except Exception as exc:                      # pragma: no cover
                    log_warning(
                        f"[sb3_trainer] RecordVideo wiring skipped: {exc}"
                    )
            return Monitor(env)
        return _factory

    fns = [
        _wrap(
            _make_env_factory(spec, env_index=i, base_seed=seed, render_mode=render_mode),
            video_dir=video_record_dir if i == 0 else None,
        )
        for i in range(n_envs)
    ]
    use_subproc = str(vec_type).strip().lower() == "subproc" and n_envs > 1
    if use_subproc:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        # "spawn" — workers are fresh interpreters that re-import the env
        # modules and inherit PYTHONPATH from the sb3_entry subprocess. fork
        # would copy this process's torch/mujoco thread pools into each child
        # (deadlock-prone), so we force spawn on every platform.
        vec = SubprocVecEnv(fns, start_method="spawn")
        log_info(
            f"[sb3_trainer] SubprocVecEnv: {n_envs} worker processes "
            f"(start_method=spawn)"
        )
    else:
        vec = DummyVecEnv(fns)
        if n_envs > 1:
            log_info(f"[sb3_trainer] DummyVecEnv: {n_envs} envs in-process (serial)")

    if env_cfg is not None and getattr(env_cfg, "obs_normalize", False):
        clip_obs = float(getattr(env_cfg, "clip_obs", 10.0))
        clip_reward = float(getattr(env_cfg, "clip_reward", 10.0))
        norm_reward = bool(getattr(env_cfg, "reward_normalize", False))
        vec = VecNormalize(
            vec,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
        )

    # obs_action.frame_stack > 1 ⇒ wrap with VecFrameStack so the policy
    # network observes ``frame_stack`` consecutive observations concatenated
    # along the feature axis. SB3's VecFrameStack stacks on the env side so
    # the bundle exporter still receives the stacked obs_dim (k × base_dim)
    # — that matches what the trained ONNX expects at deploy.
    oa = getattr(spec, "obs_action", None)
    frame_stack = int(getattr(oa, "frame_stack", 1) or 1) if oa is not None else 1
    if frame_stack > 1:
        from stable_baselines3.common.vec_env import VecFrameStack
        vec = VecFrameStack(vec, n_stack=frame_stack)

    return vec


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_model(spec, vec_env, *, run_dir: Optional[Path] = None):
    """Construct an SB3 PPO / SAC / TD3 model from ``spec.algorithm``.

    When ``run_dir`` is provided, the model's ``tensorboard_log`` is set
    to it so events.tfevents land under ``<project>/training/runs/<run_id>/``.

    Honors ``spec.algorithm.checkpoint`` (populated by spec_compiler from
    the canvas base_asset node): when ``load_mode`` is ``"resume"`` the
    saved model + optimizer state is loaded and training continues from
    ``num_timesteps`` carried in the checkpoint. ``"warm_start_actor"``
    loads the same way but resets the timestep counter so the canvas
    schedule (lr_schedule, total_timesteps) re-applies from zero.
    """
    a = spec.algorithm
    algo = (a.algorithm or "PPO").upper().strip()
    device = _resolve_device(a.device)

    # ── Checkpoint resume path (Phase 1 release audit fix) ──
    ckpt_cfg = getattr(a, "checkpoint", None)
    ckpt_file = str(getattr(ckpt_cfg, "checkpoint_file", "") or "") if ckpt_cfg else ""
    load_mode = str(getattr(ckpt_cfg, "load_mode", "scratch") or "scratch").lower() if ckpt_cfg else "scratch"
    if ckpt_file and load_mode in ("resume", "warm_start_actor"):
        if not Path(ckpt_file).exists():
            log_warning(
                f"[sb3_trainer] checkpoint_file={ckpt_file!r} missing on disk; "
                f"falling back to scratch training"
            )
        else:
            model = _load_existing_model(algo, ckpt_file, vec_env, device=device)
            if load_mode == "warm_start_actor":
                # Re-apply the canvas's current hyperparameters so the loaded
                # weights act as a seed rather than dictating the continuation
                # schedule. We can't fully re-init the SB3 optimizer here, but
                # learning_rate and gamma drive the next gradient step.
                model.learning_rate = _build_lr_schedule(
                    float(a.learning_rate or 3e-4),
                    str(getattr(a, "lr_schedule", "constant") or "constant"),
                )
                model.gamma = float(a.gamma or 0.99)
                try:
                    model._setup_lr_schedule()
                except Exception:
                    pass
                model.num_timesteps = 0
                log_success(
                    f"[sb3_trainer] warm_start_actor: loaded {algo} weights "
                    f"from {ckpt_file}"
                )
            else:
                log_success(
                    f"[sb3_trainer] resume: continuing {algo} from "
                    f"{ckpt_file} (num_timesteps={model.num_timesteps})"
                )
            if run_dir is not None:
                model.tensorboard_log = str(run_dir)
            return model

    policy_kwargs: Dict[str, Any] = {
        "net_arch": [int(a.hidden_dim_1), int(a.hidden_dim_2)],
        "activation_fn": _activation_fn(a.activation),
    }

    # PPO uses the canvas-selected lr_schedule (cosine / linear / constant);
    # SAC and TD3 keep a constant LR because their UI gates lr_schedule off.
    if algo == "PPO":
        learning_rate: Any = _build_lr_schedule(
            float(a.learning_rate or 3e-4),
            str(getattr(a, "lr_schedule", "constant") or "constant"),
        )
    else:
        learning_rate = float(a.learning_rate or 3e-4)

    common_kwargs: Dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "learning_rate": learning_rate,
        "gamma": float(a.gamma or 0.99),
        "seed": int(a.seed or 42),
        # verbose=1 makes SB3 configure its native HumanOutputFormat logger and
        # dump the full rollout/train diagnostics table (ep_rew_mean,
        # ep_len_mean, explained_variance, entropy_loss, value_loss, fps, ...)
        # to stdout every log_interval (PPO: every rollout iteration). The
        # subprocess forwards every stdout line verbatim as MSG_LOG
        # (sb3_subprocess._process_line), so that table surfaces in the UI log
        # alongside the machine-parsed [UnitPort][PROGRESS]/[METRICS] lines —
        # the latter drive the progress bar / chart, the former give the human
        # the real training signal. With tensorboard_log set the logger format
        # becomes ["stdout", "tensorboard"], so TB output is preserved.
        "verbose": 1,
        "device": device,
        "policy_kwargs": policy_kwargs,
    }
    if run_dir is not None:
        common_kwargs["tensorboard_log"] = str(run_dir)

    if algo == "PPO":
        from stable_baselines3 import PPO

        return PPO(
            **common_kwargs,
            n_steps=int(a.n_steps or 2048),
            batch_size=int(a.batch_size or 256),
            n_epochs=int(a.n_epochs or 10),
            gae_lambda=float(a.gae_lambda or 0.95),
            ent_coef=_parse_ent_coef(a.ent_coef, algo),
        )

    if algo == "SAC":
        from stable_baselines3 import SAC

        return SAC(
            **common_kwargs,
            batch_size=int(a.batch_size or 256),
            buffer_size=_resolve_buffer_size(a),
            learning_starts=int(a.learning_starts or 10_000),
            gradient_steps=int(a.gradient_steps or 1),
            tau=float(a.tau or 5e-3),
            ent_coef=_parse_ent_coef(a.ent_coef, algo),
            target_entropy=_parse_target_entropy(getattr(a, "target_entropy", "auto")),
        )

    if algo == "TD3":
        from stable_baselines3 import TD3

        return TD3(
            **common_kwargs,
            batch_size=int(a.batch_size or 256),
            buffer_size=_resolve_buffer_size(a),
            learning_starts=int(a.learning_starts or 10_000),
            gradient_steps=int(a.gradient_steps or 1),
            tau=float(a.tau or 5e-3),
        )

    raise ValueError(
        f"sb3_trainer: unsupported algorithm {algo!r}; expected PPO/SAC/TD3"
    )


try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
except Exception:  # pragma: no cover - sb3 optional at import time
    _BaseCallback = object  # type: ignore


class _CommandCurriculumCallback(_BaseCallback):  # type: ignore[misc]
    """Ramp the env command-magnitude scale from ``start`` to ``end`` over
    ``ramp_iters`` rollouts (P3 command_curriculum).

    Uses ``env_method`` so the scale reaches the underlying GenericMujocoEnv
    through the Monitor / VecNormalize / VecFrameStack wrapper chain.
    """

    def __init__(self, *, start: float, end: float, ramp_iters: int):
        super().__init__()
        self._start = float(start)
        self._end = float(end)
        self._ramp = max(1, int(ramp_iters))
        self._iter = 0

    def _set_scale(self, scale: float) -> None:
        try:
            self.training_env.env_method("set_command_curriculum_scale", float(scale))
        except Exception:  # pragma: no cover - never fail training on this
            pass

    def _on_training_start(self) -> None:
        self._set_scale(self._start)

    def _on_rollout_start(self) -> None:
        frac = min(1.0, self._iter / self._ramp)
        self._set_scale(self._start + (self._end - self._start) * frac)
        self._iter += 1

    def _on_step(self) -> bool:
        return True


class _RewardStageCallback(_BaseCallback):  # type: ignore[misc]
    """Cross-fade the env reward from stage 0 → stage 1 (P6 multigated_reward).

    Holds pure stage 0 until ``max_step_stage0`` global timesteps, then ramps
    the blend alpha 0 → 1 linearly over ``blend_steps``. Pushed to the env via
    ``env_method`` once per rollout (sub-rollout drift is negligible over a
    multi-thousand-step blend window).
    """

    def __init__(self, *, max_step_stage0: int, blend_steps: int):
        super().__init__()
        self._start = max(0, int(max_step_stage0))
        self._blend = max(1, int(blend_steps))

    def _alpha(self) -> float:
        t = int(self.num_timesteps)
        if t < self._start:
            return 0.0
        return min(1.0, (t - self._start) / self._blend)

    def _on_training_start(self) -> None:
        try:
            self.training_env.env_method("set_reward_blend", 0.0)
        except Exception:  # pragma: no cover
            pass

    def _on_rollout_start(self) -> None:
        try:
            self.training_env.env_method("set_reward_blend", self._alpha())
        except Exception:  # pragma: no cover
            pass

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def train_sb3(
    spec,
    *,
    total_timesteps: Optional[int] = None,
    callback: Any = None,
    progress_bar: bool = False,
    run_dir: Optional[Path] = None,
) -> TrainingResult:
    """Run an SB3 training session against :class:`GenericMujocoEnv`.

    Args:
        spec: populated :class:`TrainingSpec` (Stage 3).
        total_timesteps: override ``spec.algorithm.total_timesteps``.
            Useful for smoke tests (e.g. 50 steps).
        callback: optional SB3 callback (e.g. progress IPC bridge in
            Stage 10). Defaults to ``None``.
        progress_bar: forwarded to ``model.learn``.
        run_dir: project-scoped run directory
            (``<project>/training/runs/<run_id>/``). When provided,
            TensorBoard events land under it and a ``CheckpointCallback``
            is appended that snapshots ``model_*.pt`` into
            ``run_dir/checkpoints/``. Caller (``sb3_entry``) is expected
            to ``mkdir`` the dir before calling.

    Returns:
        :class:`TrainingResult` with the trained model + vec_env. Caller
        is responsible for ``vec_env.close()``.
    """
    if spec is None or not getattr(spec, "robot", None) or not spec.robot.sku:
        raise ValueError("train_sb3: spec.robot is missing or unresolved")

    a = spec.algorithm
    # Hardware-adaptive vectorisation: resolve (n_envs, vec_type) on THIS
    # machine (auto) or take the canvas values (manual). Done here — at launch,
    # on the training box — so cloud/remote workers adapt to their own CPU/RAM
    # rather than the editor's. The PPO budget below uses the RESOLVED n_envs.
    from application.training.envs.auto_parallelism import resolve_parallelism

    _env_cfg = getattr(spec, "env", None)
    _plan = resolve_parallelism(
        mode=str(getattr(_env_cfg, "parallelism_mode", "auto") or "auto"),
        manual_n_envs=int(getattr(_env_cfg, "n_envs", 8) or 8) if _env_cfg else 8,
        manual_vec_type=str(getattr(_env_cfg, "vec_type", "subproc") or "subproc"),
    )
    n_envs = _plan.n_envs
    vec_type = _plan.vec_type
    log_info(f"[sb3_trainer] parallelism {_plan.rationale}")
    seed = int(a.seed or 42)
    algo = (a.algorithm or "PPO").upper()
    # PPO is iteration-based: the canvas sets max_iterations, and total env
    # steps = iterations × rollout(n_steps × n_envs). This keeps the budget
    # stable when n_steps / n_envs change (matches the IsaacLab mental model).
    # SAC/TD3 are off-policy (no rollout iteration) → use total_timesteps.
    # An explicit total_timesteps arg (smoke runs) always overrides.
    if total_timesteps is not None:
        steps = int(total_timesteps)
    elif algo == "PPO":
        n_steps = max(1, int(a.n_steps or 2048))
        max_iters = max(1, int(getattr(a, "max_iterations", 1500) or 1500))
        steps = max_iters * n_steps * max(1, n_envs)
    else:
        steps = int(a.total_timesteps)

    log_info(
        f"[sb3_trainer] starting {a.algorithm} run "
        f"(total_timesteps={steps}"
        + (f" = {max(1,int(getattr(a,'max_iterations',1500) or 1500))} iters × "
           f"{max(1,int(a.n_steps or 2048))} n_steps × {n_envs} envs" if algo == 'PPO'
           and total_timesteps is None else "")
        + f", n_envs={n_envs}, seed={seed}, robot={spec.robot.sku!r})"
    )

    # Surface unimplemented hyperparameter switches so a user who enabled
    # them on the canvas is not misled by silent no-ops. PER is not in
    # stable-baselines3 stock; collapse_guard is deferred (see module
    # docstring). These warnings only fire on opt-in, never on defaults.
    if bool(getattr(a, "use_per", False)):
        log_warning(
            "[sb3_trainer] use_per=True is not supported by stable-baselines3 "
            "stock; PrioritizedReplayBuffer is ignored for this run."
        )

    _warn_unimplemented_features(spec)

    vec_env = build_vec_env(spec, n_envs=n_envs, seed=seed, vec_type=vec_type)
    obs_dim = int(vec_env.observation_space.shape[0])
    action_dim = int(vec_env.action_space.shape[0])

    # Compose the user-supplied progress callback with a CheckpointCallback
    # rooted at run_dir/checkpoints/, when a run_dir is bound.
    final_callback = callback
    # Per-term + per-item reward breakdown logger. Always wired (no run_dir
    # gate) — TensorBoard records land wherever the SB3 logger is pointed,
    # and a no-op tick is cheaper than missing the diagnostic when a user
    # debugs locally without a run_dir.
    try:
        from stable_baselines3.common.callbacks import CallbackList as _CL_R
        from application.training.sb3_reward_log import RewardBreakdownCallback

        reward_log_cb = RewardBreakdownCallback(log_every=100)
        if final_callback is None:
            final_callback = reward_log_cb
        else:
            final_callback = _CL_R([final_callback, reward_log_cb])
    except Exception as exc:                                  # pragma: no cover
        log_warning(f"[sb3_trainer] RewardBreakdownCallback wiring skipped: {exc}")

    # Command-magnitude curriculum (P3): ramp the env's command scale from
    # ``start`` to ``end`` over ``ramp_iters`` rollouts so early training sees
    # gentle velocity targets. Injected into the env via env_method (reaches
    # the GenericMujocoEnv through the Monitor / VecNormalize wrappers).
    motion_for_cmd = getattr(spec, "motion", None)
    cmd_curr = getattr(motion_for_cmd, "command_curriculum", None) if motion_for_cmd else None
    if cmd_curr is not None and bool(getattr(cmd_curr, "enabled", False)):
        try:
            from stable_baselines3.common.callbacks import CallbackList as _CL_C
            cmd_cb = _CommandCurriculumCallback(
                start=float(getattr(cmd_curr, "start", 0.25) or 0.25),
                end=float(getattr(cmd_curr, "end", 1.0) or 1.0),
                ramp_iters=int(getattr(cmd_curr, "ramp_iters", 800) or 800),
            )
            final_callback = (
                cmd_cb if final_callback is None
                else _CL_C([final_callback, cmd_cb])
            )
        except Exception as exc:                              # pragma: no cover
            log_warning(f"[sb3_trainer] command-curriculum callback skipped: {exc}")

    # Multi-stage reward cross-fade (P6): when a multigated_reward node set a
    # 2-stage schedule with a stage-0 budget, ramp the env reward blend.
    sched_for_stage = getattr(spec, "stage_schedule", None)
    rewards_for_stage = getattr(spec, "rewards", None)
    if (
        sched_for_stage is not None
        and rewards_for_stage is not None
        and int(getattr(sched_for_stage, "max_step_stage0", 0) or 0) > 0
        and len(getattr(rewards_for_stage, "stages", None) or []) >= 2
    ):
        try:
            from stable_baselines3.common.callbacks import CallbackList as _CL_S
            stage_cb = _RewardStageCallback(
                max_step_stage0=int(getattr(sched_for_stage, "max_step_stage0", 0) or 0),
                blend_steps=int(getattr(sched_for_stage, "blend_steps", 3000) or 3000),
            )
            final_callback = (
                stage_cb if final_callback is None
                else _CL_S([final_callback, stage_cb])
            )
        except Exception as exc:                              # pragma: no cover
            log_warning(f"[sb3_trainer] reward-stage callback skipped: {exc}")
    if run_dir is not None:
        try:
            from stable_baselines3.common.callbacks import (
                CallbackList,
                CheckpointCallback,
            )

            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # ``algorithm.checkpoint_interval`` is the canvas UI key (global
            # timesteps between snapshots). Fall back to ~20 evenly spaced
            # snapshots with a 10k floor when the user leaves it at 0.
            desired_global = int(getattr(a, "checkpoint_interval", 0) or 0)
            if desired_global <= 0:
                desired_global = max(steps // 20, 10_000)
            # SB3's CheckpointCallback save_freq counts _on_step() calls,
            # which under a VecEnv tick once per `n_envs` global steps.
            save_freq = max(1, desired_global // max(1, n_envs))

            ckpt_cb = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(ckpt_dir),
                name_prefix="model",
            )
            if final_callback is None:
                final_callback = ckpt_cb
            else:
                final_callback = CallbackList([final_callback, ckpt_cb])
        except Exception as exc:                             # pragma: no cover
            log_warning(f"[sb3_trainer] CheckpointCallback wiring skipped: {exc}")

    # EvalCallback honors eval_config from the canvas. SB3 stock supports
    # periodic eval episodes against a separate env and an optional
    # best-model checkpoint; eval_episodes / eval_interval / deterministic
    # / save_best_model on the canvas drive these knobs.
    eval_cfg = getattr(spec, "eval", None)
    if eval_cfg is not None and run_dir is not None:
        eval_interval = int(getattr(eval_cfg, "eval_interval", 0) or 0)
        eval_episodes = int(getattr(eval_cfg, "eval_episodes", 0) or 0)
        save_best = bool(getattr(eval_cfg, "save_best_model", True))
        deterministic = bool(getattr(eval_cfg, "deterministic", True))
        record_video = bool(getattr(eval_cfg, "record_video", False))
        # Custom video_dir overrides the in-run-dir default. When the
        # canvas leaves video_dir blank we land videos beside the
        # checkpoints so a user inspecting the run_dir sees them next to
        # the policy artifacts.
        video_dir_str = (getattr(eval_cfg, "video_dir", "") or "").strip()
        if record_video:
            video_dir = Path(video_dir_str) if video_dir_str else (run_dir / "videos")
        else:
            video_dir = None
        if eval_interval > 0 and eval_episodes > 0:
            try:
                from stable_baselines3.common.callbacks import (
                    CallbackList as _CL,
                    EvalCallback,
                )

                class _LoggingEvalCallback(EvalCallback):
                    """Announce the eval BEFORE it runs.

                    Periodic eval steps ``n_eval_episodes`` episodes on a
                    SINGLE env; when the policy keeps the robot alive each
                    episode runs to the full horizon, so the eval can take
                    tens of seconds with no rollout/iteration output — which
                    reads as a frozen run in the UI log. Printing a marker
                    here (forwarded verbatim as MSG_LOG by the subprocess)
                    makes the pause legible. The base class already logs the
                    eval result afterwards under verbose=1. To shorten the
                    pause itself, lower EvalConfig.eval_episodes or raise
                    eval_interval on the canvas.
                    """

                    def _on_step(inner) -> bool:
                        if (
                            inner.eval_freq > 0
                            and inner.n_calls > 0
                            and inner.n_calls % inner.eval_freq == 0
                        ):
                            print(
                                f"[UnitPort] eval: running "
                                f"{inner.n_eval_episodes} episode(s) on 1 env "
                                f"(training paused until eval completes) ...",
                                flush=True,
                            )
                        return super()._on_step()

                eval_env = build_vec_env(
                    spec,
                    n_envs=1,
                    seed=seed + 10_000,
                    render_mode="rgb_array" if record_video else None,
                    video_record_dir=video_dir,
                )
                best_dir = run_dir / "best_model" if save_best else None
                if best_dir is not None:
                    best_dir.mkdir(parents=True, exist_ok=True)
                eval_freq = max(1, eval_interval // max(1, n_envs))
                eval_cb = _LoggingEvalCallback(
                    eval_env,
                    n_eval_episodes=eval_episodes,
                    eval_freq=eval_freq,
                    best_model_save_path=str(best_dir) if best_dir else None,
                    deterministic=deterministic,
                    render=False,
                )
                if final_callback is None:
                    final_callback = eval_cb
                else:
                    final_callback = _CL([final_callback, eval_cb])
            except Exception as exc:                         # pragma: no cover
                log_warning(f"[sb3_trainer] EvalCallback wiring skipped: {exc}")

    vis_cfg = getattr(spec, "vis", None)
    if vis_cfg is not None:
        num_checks = int(getattr(vis_cfg, "num_vis_checks", 0) or 0)
        if num_checks > 0:
            log_warning(
                "[sb3_trainer] vis_check is not implemented in the SB3 path; "
                "the MuJoCo viewer will not pop during training. The bundle "
                "is still produced at the end of training for review."
            )

    # export.include_torchscript → SB3 bundle now ships a policy.pt companion
    # (bundle_exporter); export.include_normalization → VecNormalize obs stats
    # are round-tripped to normalization.json + applied at deploy
    # (PolicyRunner._load_bundle_normalizer). Both are honoured, so no
    # warn-and-continue here anymore.

    try:
        model = _build_model(spec, vec_env, run_dir=run_dir)
        # For ``load_mode=resume`` we want SB3 to keep the loaded timestep
        # counter so the lr schedule and total_timesteps target are computed
        # against cumulative progress. Warm-start and scratch both start at 0.
        ckpt_cfg = getattr(a, "checkpoint", None)
        load_mode_runtime = (
            str(getattr(ckpt_cfg, "load_mode", "scratch") or "scratch").lower()
            if ckpt_cfg else "scratch"
        )
        reset_num_timesteps = load_mode_runtime != "resume"
        model.learn(
            total_timesteps=steps,
            callback=final_callback,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
    except Exception:
        try:
            vec_env.close()
        except Exception:
            pass
        raise

    log_success(
        f"[sb3_trainer] {a.algorithm} run completed "
        f"(obs_dim={obs_dim}, action_dim={action_dim})"
    )

    return TrainingResult(
        model=model,
        vec_env=vec_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        algorithm=a.algorithm.upper(),
        total_timesteps=steps,
        seed=seed,
    )


__all__ = [
    "TrainingResult",
    "train_sb3",
    "build_vec_env",
]
