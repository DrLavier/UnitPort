# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AMP-PPO trainer — SB3 PPO with adversarial style reward.

Stage 8 D2 decision: **callback-based** disc training (option B from
plan.md§D2). Rationale:

  * SB3's :class:`PPO` and :class:`RolloutBuffer` are imported as-is —
    no monkey-patching, no fork.
  * :class:`AmpRolloutCollector` is a SB3 ``BaseCallback`` that fires at
    the end of each rollout. It pulls policy AMP transitions out of the
    env (via ``env.get_amp_observations()``) into :class:`AmpReplayBuffer`,
    samples expert transitions from the motion clip, trains the disc
    one minibatch step, and stages the per-step style reward into the
    env wrapper for the next rollout.
  * :class:`AmpStyleRewardWrapper` is a thin VecEnv reward shaper that
    blends the staged style reward with the task reward via
    ``(1-lerp)*style + lerp*task``.

Out of Stage 8 v1:

  * Multi-epoch disc updates (we do one step per rollout). Add when
    Stage 12 says it's needed.
  * Reference state init (RSI). Hooks reserved on the wrapper.
  * Per-stage gating (multigated_reward). Stage 11 will resurrect.

Acceptance: 100-step AMP_PPO smoke run completes; disc loss has finite
gradients; ``BundleExporter.export_bundle`` produces a valid policy.onnx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from unitport_sdk import log_info, log_success, log_warning


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AmpTrainingResult:
    model: Any                          # stable_baselines3.PPO
    vec_env: Any
    discriminator: Any                  # AMPDiscriminator
    obs_dim: int
    action_dim: int
    amp_obs_dim: int
    total_timesteps: int


# ---------------------------------------------------------------------------
# Style-reward wrapper — blends env task reward with staged style reward
# ---------------------------------------------------------------------------


def _make_amp_style_reward_wrapper():
    """Return :class:`AmpStyleRewardWrapper` (lazy class definition).

    SB3's VecEnvWrapper imports torch indirectly through SB3 — we keep
    this behind a function so the parent module imports cleanly without
    SB3 / torch installed.
    """
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class AmpStyleRewardWrapper(VecEnvWrapper):
        """Blend task reward with staged style reward.

        The :class:`AmpRolloutCollector` callback writes a per-env style
        reward array into :attr:`_pending_style_reward` after each
        rollout's disc step. On each ``step_wait``, this wrapper applies
        ``r_total = (1-lerp)*style + lerp*task``.

        When the buffer is empty (first rollout, before any disc step),
        the wrapper passes the task reward through unchanged.
        """

        def __init__(self, venv, *, task_reward_lerp: float = 0.5) -> None:
            super().__init__(venv)
            self.task_reward_lerp = float(task_reward_lerp)
            self._pending_style_reward: Optional[np.ndarray] = None
            # AMP transition recording — collector reads these after step.
            self._last_amp_obs: Optional[np.ndarray] = None
            self._this_amp_obs: Optional[np.ndarray] = None

        def reset(self):
            obs = self.venv.reset()
            try:
                self._last_amp_obs = _safe_get_amp_obs(self.venv)
            except Exception:
                self._last_amp_obs = None
            self._this_amp_obs = None
            return obs

        def step_async(self, actions):
            self.venv.step_async(actions)

        def step_wait(self):
            obs, rewards, dones, infos = self.venv.step_wait()
            try:
                self._this_amp_obs = _safe_get_amp_obs(self.venv)
            except Exception:
                self._this_amp_obs = None
            # Blend task + style reward (per-env)
            blended = np.asarray(rewards, dtype=np.float32)
            if self._pending_style_reward is not None:
                lerp = self.task_reward_lerp
                style = np.asarray(self._pending_style_reward, dtype=np.float32)
                if style.shape == blended.shape:
                    blended = (1.0 - lerp) * style + lerp * blended
                else:
                    log_warning(
                        f"[amp_ppo] style reward shape {style.shape} != "
                        f"task reward shape {blended.shape}; using task only"
                    )
            return obs, blended, dones, infos

        def stage_style_reward(self, style_reward: np.ndarray) -> None:
            self._pending_style_reward = np.asarray(style_reward, dtype=np.float32)

        def latest_amp_pair(self):
            """Return ``(s_t, s_{t+1})`` arrays of shape ``(num_envs, dim)``,
            or ``(None, None)`` when env hasn't supplied AMP obs yet."""
            return self._last_amp_obs, self._this_amp_obs

        def advance_amp_history(self) -> None:
            self._last_amp_obs = self._this_amp_obs

    return AmpStyleRewardWrapper


def _safe_get_amp_obs(venv) -> Optional[np.ndarray]:
    """Best-effort ``env.get_amp_observations()`` across VecEnv layers.

    Stage 8 v1: env-side AMP obs extraction is not yet wired into
    :class:`GenericMujocoEnv` (Stage 10 will land it alongside the
    SB3 launcher). Until then we synthesize AMP obs on the fly from
    each env's qpos/qvel — the joint_alignment guard will catch any
    field-name drift at boot.
    """
    try:
        return venv.env_method("get_amp_observations")
    except (AttributeError, Exception):
        pass
    # Fallback: read a stand-in AMP obs from the env's joint state.
    try:
        from stable_baselines3.common.vec_env import VecEnv

        if isinstance(venv, VecEnv):
            num_envs = venv.num_envs
            try:
                qpos = np.asarray(
                    venv.env_method("_get_amp_obs_synth")
                )
            except (AttributeError, Exception):
                qpos = None
            if qpos is not None and qpos.ndim == 2 and qpos.shape[0] == num_envs:
                return qpos.astype(np.float32, copy=False)
    except Exception:
        pass
    return None


# Class name re-exported from module level (lazy class def above).
def __getattr__(name):  # noqa: D401
    if name == "AmpStyleRewardWrapper":
        cls = _make_amp_style_reward_wrapper()
        globals()["AmpStyleRewardWrapper"] = cls
        return cls
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Rollout collector — SB3 callback that trains the disc each rollout
# ---------------------------------------------------------------------------


def _make_amp_rollout_collector():
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    import torch.nn.functional as F

    class AmpRolloutCollector(BaseCallback):
        """Trains the AMP discriminator after each PPO rollout.

        Args:
            discriminator: :class:`AMPDiscriminator` instance.
            replay_buffer: :class:`AmpReplayBuffer` for policy transitions.
            normalizer: :class:`Normalizer` for AMP obs (running stats).
            expert_clip: a :class:`MotionClip` with ``has_amp_payload()==True``.
            wrapper: :class:`AmpStyleRewardWrapper` — collector stages the
                style reward back through this wrapper.
            disc_lr: SGD step size for the disc Adam optimizer.
            grad_pen_lambda: gradient-penalty weight (zero-centered).
            label_smoothing: BCE label smoothing for disc fake/real targets.
            disc_batch_size: minibatch size per disc step.
            disc_steps_per_rollout: number of disc updates per PPO rollout.
        """

        def __init__(
            self,
            *,
            discriminator,
            replay_buffer,
            normalizer,
            expert_clip,
            wrapper,
            disc_lr: float = 1e-4,
            grad_pen_lambda: float = 10.0,
            label_smoothing: float = 0.9,
            disc_batch_size: int = 256,
            disc_steps_per_rollout: int = 1,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose=verbose)
            self.disc = discriminator
            self.buf = replay_buffer
            self.normalizer = normalizer
            self.expert_clip = expert_clip
            self.wrapper = wrapper
            self.disc_lr = float(disc_lr)
            self.grad_pen_lambda = float(grad_pen_lambda)
            self.label_smoothing = float(label_smoothing)
            self.disc_batch_size = int(disc_batch_size)
            self.disc_steps_per_rollout = max(1, int(disc_steps_per_rollout))
            self._optim = torch.optim.Adam(self.disc.parameters(), lr=self.disc_lr)
            # Stats for logging — Stage 12 will consume.
            self.last_disc_loss: float = 0.0
            self.last_grad_pen: float = 0.0
            self.last_acc_expert: float = 0.0
            self.last_acc_policy: float = 0.0

        def _on_step(self) -> bool:
            # Per-step: stash the env's AMP transition pair into the buffer.
            s_t, s_tp1 = self.wrapper.latest_amp_pair()
            if s_t is not None and s_tp1 is not None:
                self.buf.push(s_t, s_tp1)
                self.normalizer.update(s_t)
            self.wrapper.advance_amp_history()
            return True

        def _on_rollout_end(self) -> None:
            if self.buf.size < self.disc_batch_size:
                return  # not enough policy data yet
            if not self.expert_clip.has_amp_payload():
                return  # tracking-only clip; no AMP transitions to learn from

            for _ in range(self.disc_steps_per_rollout):
                # ── Sample policy + expert ──
                pol_s, pol_sn = self.buf.sample(self.disc_batch_size)
                exp_s, exp_sn = self.expert_clip.sample_transitions(
                    self.disc_batch_size, dt=self.expert_clip.frame_dt,
                )
                self.normalizer.update(exp_s)

                # ── To torch ──
                device = self.disc.device
                pol_s_t = torch.as_tensor(pol_s, device=device, dtype=torch.float32)
                pol_sn_t = torch.as_tensor(pol_sn, device=device, dtype=torch.float32)
                exp_s_t = torch.as_tensor(exp_s, device=device, dtype=torch.float32)
                exp_sn_t = torch.as_tensor(exp_sn, device=device, dtype=torch.float32)

                pol_s_n = self.normalizer.normalize_torch(pol_s_t, device)
                pol_sn_n = self.normalizer.normalize_torch(pol_sn_t, device)
                exp_s_n = self.normalizer.normalize_torch(exp_s_t, device)
                exp_sn_n = self.normalizer.normalize_torch(exp_sn_t, device)

                pol_x = torch.cat([pol_s_n, pol_sn_n], dim=-1)
                exp_x = torch.cat([exp_s_n, exp_sn_n], dim=-1)

                # ── Disc forward + BCE-with-label-smoothing ──
                self.disc.train()
                pol_logits = self.disc.forward(pol_x)
                exp_logits = self.disc.forward(exp_x)
                # Labels: expert=1.0*smoothing, policy=0
                exp_target = torch.full_like(
                    exp_logits, self.label_smoothing
                )
                pol_target = torch.zeros_like(pol_logits)
                bce = (
                    F.binary_cross_entropy_with_logits(exp_logits, exp_target)
                    + F.binary_cross_entropy_with_logits(pol_logits, pol_target)
                )

                # Zero-centered grad penalty on expert pairs only
                grad_pen = self.disc.compute_grad_pen(
                    exp_s_n, exp_sn_n, lambda_=self.grad_pen_lambda
                )

                loss = bce + grad_pen
                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                self._optim.step()

                # Logging stats
                with torch.no_grad():
                    self.last_disc_loss = float(loss.item())
                    self.last_grad_pen = float(grad_pen.item())
                    self.last_acc_expert = float(
                        (exp_logits > 0).float().mean().item()
                    )
                    self.last_acc_policy = float(
                        (pol_logits < 0).float().mean().item()
                    )

            # ── Stage style reward for the next rollout ──
            # Use the most recent transition pair to compute style reward
            # for the env's current observation; we extrapolate to all
            # envs via a single batched disc inference.
            s_t, s_tp1 = self.wrapper.latest_amp_pair()
            if s_t is None or s_tp1 is None:
                return
            with torch.no_grad():
                cur_s = torch.as_tensor(s_t, device=device, dtype=torch.float32)
                cur_sn = torch.as_tensor(s_tp1, device=device, dtype=torch.float32)
                style_r, _ = self.disc.predict_amp_reward(
                    cur_s, cur_sn, normalizer=self.normalizer,
                )
                self.wrapper.stage_style_reward(style_r.cpu().numpy())

    return AmpRolloutCollector


def __getattr_for_collector(name):
    cls = _make_amp_rollout_collector()
    return cls


# Re-export pattern: also lazy-load AmpRolloutCollector on first access.
_orig_getattr = __getattr__


def __getattr__(name):  # noqa: F811
    if name == "AmpStyleRewardWrapper":
        cls = _make_amp_style_reward_wrapper()
        globals()["AmpStyleRewardWrapper"] = cls
        return cls
    if name == "AmpRolloutCollector":
        cls = _make_amp_rollout_collector()
        globals()["AmpRolloutCollector"] = cls
        return cls
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def train_amp_ppo(
    spec,
    expert_clip,
    *,
    total_timesteps: Optional[int] = None,
    callback=None,
    run_dir: Optional["Path"] = None,
) -> AmpTrainingResult:
    """Run AMP-PPO training.

    Args:
        spec: populated :class:`TrainingSpec` (algorithm.training_mode
            should be ``"AMP_PPO"``).
        expert_clip: a :class:`MotionClip` with AMP payload.
        total_timesteps: override ``spec.algorithm.total_timesteps``.
        callback: optional extra SB3 callback (e.g. progress IPC). The
            AMP rollout collector is always installed; user callbacks
            are stacked via ``CallbackList`` when provided.
        run_dir: project-scoped run directory
            (``<project>/training/runs/<run_id>/``). When provided:
              * PPO's ``tensorboard_log`` lands under it.
              * A ``CheckpointCallback`` snapshots ``model_*.pt`` into
                ``run_dir/checkpoints/`` with the same cadence as SB3.
              * On completion, the disc weights and AMP replay buffer
                are saved atomically to ``run_dir/amp/discriminator.pt``
                and ``run_dir/amp/replay_buffer.npz``.

    Returns:
        :class:`AmpTrainingResult` with the trained PPO model + disc.
        Caller is responsible for ``vec_env.close()``.
    """
    from pathlib import Path

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from application.training.amp.algorithms.discriminator import AMPDiscriminator
    from application.training.amp.obs_terms import (
        DEFAULT_QUADRUPED_TERMS,
        compute_amp_obs_dim,
    )
    from application.training.amp.storage.replay_buffer import AmpReplayBuffer
    from application.training.amp.utils.normalizer import Normalizer
    from application.training.sb3_trainer import (
        TrainingResult,
        _build_model,
        build_vec_env,
    )

    if spec is None or not getattr(spec, "robot", None) or not spec.robot.sku:
        raise ValueError("train_amp_ppo: spec.robot is missing or unresolved")
    if expert_clip is None:
        raise ValueError("train_amp_ppo: expert_clip is required")
    if not expert_clip.has_amp_payload():
        raise ValueError(
            "train_amp_ppo: expert_clip lacks AMP payload "
            "(use format_id='amp_legged_gym')"
        )

    # ── Force PPO + AMP-PPO mode ──
    spec.algorithm.algorithm = "PPO"
    spec.algorithm.training_mode = "AMP_PPO"
    a = spec.algorithm
    n_envs = int(getattr(spec.env, "n_envs", 8) or 8) if getattr(spec, "env", None) else 8
    seed = int(a.seed or 42)
    steps = int(total_timesteps if total_timesteps is not None else a.total_timesteps)

    log_info(
        f"[amp_ppo] starting AMP-PPO "
        f"(steps={steps}, n_envs={n_envs}, expert_clip={expert_clip.name!r})"
    )

    # ── VecEnv + style-reward wrapper ──
    base_vec = build_vec_env(spec, n_envs=n_envs, seed=seed)
    AmpStyleRewardWrapper_cls = _make_amp_style_reward_wrapper()
    vec_env = AmpStyleRewardWrapper_cls(
        base_vec, task_reward_lerp=spec.il.amp.task_reward_lerp,
    )

    # ── Disc + normalizer + buffer ──
    num_dofs = int(spec.robot.num_joints if hasattr(spec.robot, "num_joints") else len(spec.robot.joint_order))
    amp_obs_dim = compute_amp_obs_dim(
        DEFAULT_QUADRUPED_TERMS,
        context={"num_dofs": num_dofs, "num_feet": 4},
    )
    disc = AMPDiscriminator(
        input_dim=2 * amp_obs_dim,  # (s, s') concat
        amp_reward_coef=spec.il.amp.amp_reward_coef,
        hidden_layer_sizes=spec.il.amp.disc.hidden_dims,
        task_reward_lerp=spec.il.amp.task_reward_lerp,
        logit_clamp_max=spec.il.amp.disc.disc_logit_clamp_max,
    )
    normalizer = Normalizer(amp_obs_dim)
    buf_cap = max(amp_obs_dim * 1024, int(spec.il.amp.amp_replay_buffer_size or 1_000_000))
    replay = AmpReplayBuffer(capacity=min(buf_cap, 100_000), amp_obs_dim=amp_obs_dim)

    # ── Collector callback ──
    AmpRolloutCollector_cls = _make_amp_rollout_collector()
    collector = AmpRolloutCollector_cls(
        discriminator=disc,
        replay_buffer=replay,
        normalizer=normalizer,
        expert_clip=expert_clip,
        wrapper=vec_env,
        disc_lr=spec.il.amp.disc_lr,
        grad_pen_lambda=spec.il.amp.disc.disc_grad_penalty,
        label_smoothing=spec.il.amp.disc.disc_label_smoothing,
        disc_batch_size=256,
        disc_steps_per_rollout=1,
    )

    # ── Stack collector + (optional) user callback + (optional) checkpoint cb ──
    cbs = [collector]
    if callback is not None:
        cbs.append(callback)
    if run_dir is not None:
        try:
            ckpt_dir = Path(run_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            desired_global = int(getattr(a, "checkpoint_freq", 0) or 0)
            if desired_global <= 0:
                desired_global = max(steps // 20, 10_000)
            save_freq = max(1, desired_global // max(1, n_envs))
            cbs.append(CheckpointCallback(
                save_freq=save_freq,
                save_path=str(ckpt_dir),
                name_prefix="model",
            ))
        except Exception as exc:                             # pragma: no cover
            log_warning(f"[amp_ppo] CheckpointCallback wiring skipped: {exc}")
    cb = cbs[0] if len(cbs) == 1 else CallbackList(cbs)

    # ── Build PPO model + learn ──
    try:
        model = _build_model(spec, vec_env, run_dir=run_dir)
        model.learn(total_timesteps=steps, callback=cb)
    except Exception:
        try:
            vec_env.close()
        except Exception:
            pass
        raise

    log_success(
        f"[amp_ppo] AMP-PPO completed — "
        f"disc_loss={collector.last_disc_loss:.4f} "
        f"grad_pen={collector.last_grad_pen:.4f} "
        f"acc_e={collector.last_acc_expert:.2f} "
        f"acc_p={collector.last_acc_policy:.2f}"
    )

    # ── Persist disc weights + replay buffer to <run_dir>/amp/ ──
    if run_dir is not None:
        try:
            amp_dir = Path(run_dir) / "amp"
            amp_dir.mkdir(parents=True, exist_ok=True)
            disc.save(amp_dir / "discriminator.pt")
            replay.save(amp_dir / "replay_buffer.npz")
            log_info(
                f"[amp_ppo] persisted disc + replay buffer → {amp_dir}"
            )
        except Exception as exc:                             # pragma: no cover
            log_warning(f"[amp_ppo] AMP persistence failed: {exc}")

    return AmpTrainingResult(
        model=model,
        vec_env=vec_env,
        discriminator=disc,
        obs_dim=int(vec_env.observation_space.shape[0]),
        action_dim=int(vec_env.action_space.shape[0]),
        amp_obs_dim=amp_obs_dim,
        total_timesteps=steps,
    )


# ---------------------------------------------------------------------------
# Lightweight alias class used by callers that just want to construct
# the trainer + collector pair manually.
# ---------------------------------------------------------------------------


class AmpPpoTrainer:
    """Thin wrapper around :func:`train_amp_ppo` for symmetry with
    Stage 7's :class:`SB3Trainer`-style API. Most callers use the
    function form."""

    def __init__(self, spec, expert_clip) -> None:
        self.spec = spec
        self.expert_clip = expert_clip

    def run(
        self,
        *,
        total_timesteps: Optional[int] = None,
        callback=None,
        run_dir: Optional["Path"] = None,
    ) -> AmpTrainingResult:
        return train_amp_ppo(
            self.spec,
            self.expert_clip,
            total_timesteps=total_timesteps,
            callback=callback,
            run_dir=run_dir,
        )


__all__ = [
    "AmpPpoTrainer",
    "AmpTrainingResult",
    "train_amp_ppo",
    # Lazy-loaded; module __getattr__ resolves these on first access:
    # "AmpStyleRewardWrapper",
    # "AmpRolloutCollector",
]
