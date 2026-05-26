# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Behavioral Cloning (BC) — Phase 1 pure BC pre-training.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/behavioral_cloning.py``,
split into two files per the Stage 9 plan:

    * :mod:`application.training.behavioral_cloning` — :class:`DemonstrationBuffer`
      + :class:`BehavioralCloningTrainer` (this file).
    * :mod:`application.training.il_blend` — :class:`BCBlendCallback` (Phase 2
      blend during RL fine-tune).

The 2-phase imitation pipeline (DAPG / DeepMimic / AMP):

    Phase 1 — Pure BC: supervised MSE / Huber loss on (obs, a_ref) pairs
              extracted from a reference :class:`MotionClip` and replayed
              through :class:`GenericMujocoEnv`.
    Phase 2 — BC + RL blend: auxiliary BC loss decays linearly over N
              RL timesteps; see :mod:`il_blend` for the SB3 callback.

Stage 9 changes vs DEMO:

    * Buffer collection: DEMO required ``env.get_reference_action_for_bc()``
      and ``env._action_clip`` (DEMO env-specific). RELEASE drives the
      replay from a :class:`MotionClip` directly via
      :meth:`DemonstrationBuffer.from_motion_clip` — joint-pos frames in
      the clip are mapped to the env's action vector through
      ``RobotSpec.joint_order``.
    * Logging via :func:`unitport_sdk.log_*` instead of ``logging`` module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from unitport_sdk import log_info, log_success, log_warning


# ---------------------------------------------------------------------------
# DemonstrationBuffer
# ---------------------------------------------------------------------------


@dataclass
class DemonstrationBuffer:
    """Stores ``(observation, reference_action)`` pairs for BC supervision.

    Two factories:
      * :meth:`from_motion_clip` — replay a clip through an env (the
        canonical Stage 9 path; clip joint frames → action vector via
        ``RobotSpec.joint_order``).
      * :meth:`from_env_callback` — call ``env.get_reference_action_for_bc()``
        if the env exposes it (legacy DEMO compat). Stage 9 leaves this
        as a thin shim; new code should prefer the clip-based path.
    """

    observations: np.ndarray    # (N, obs_dim) float32
    actions: np.ndarray         # (N, action_dim) float32
    episode_starts: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return int(self.observations.shape[0]) if self.observations.size else 0

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1]) if self.observations.size else 0

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1]) if self.actions.size else 0

    @staticmethod
    def from_motion_clip(
        env,
        clip,
        *,
        num_trajectories: int = 50,
        noise_std: float = 0.0,
        action_scale: Optional[float] = None,
        action_clip: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "DemonstrationBuffer":
        """Collect demos by replaying a reference :class:`MotionClip`.

        The clip's ``joint_pos`` is sampled at the env's control rate
        and mapped to the env's action vector under ``action = (q_ref -
        q_default) / action_scale``. The result mirrors the policy
        action output for the same observation, so a downstream BC
        trainer can fit ``π(obs) ≈ a_ref``.

        Args:
            env: a :class:`GenericMujocoEnv` instance.
            clip: a :class:`MotionClip` whose ``dof`` matches
                ``env.num_joints``.
            num_trajectories: episodes to roll.
            noise_std: optional Gaussian noise on the reference action
                before stepping (DAgger-like distribution broadening).
            action_scale: divide ``q_err`` by this when forming the
                action. Defaults to ``env._d.action_scale`` (the env's
                position-control scale).
            action_clip: clip the (possibly-noisy) action to ``[-clip,
                +clip]`` before stepping the env. Set to ``1.0`` for
                the default Box(-1,1) action space.
        """
        if clip is None:
            raise ValueError("from_motion_clip: clip is required")
        if clip.dof != int(env.num_joints):
            raise ValueError(
                f"from_motion_clip: clip.dof={clip.dof} != env.num_joints="
                f"{env.num_joints}; UnitPort does not retarget"
            )
        if rng is None:
            rng = np.random.default_rng()
        if action_scale is None:
            action_scale = float(getattr(env, "_d", None).action_scale) if getattr(env, "_d", None) else 0.25

        all_obs: List[np.ndarray] = []
        all_act: List[np.ndarray] = []
        ep_starts: List[int] = []
        idx = 0

        # control_dt for time tracking through episode
        control_dt = float(getattr(env, "_d", None).control_dt) if getattr(env, "_d", None) else 0.02

        for traj_i in range(int(num_trajectories)):
            obs, _ = env.reset()
            ep_starts.append(idx)
            done = False
            t = 0.0

            while not done:
                frame = clip.frame_at(t)
                q_ref = np.asarray(frame.get("joint_pos"), dtype=np.float32)
                if q_ref.shape[0] != env.num_joints:
                    log_warning(
                        f"[BC] motion frame dof={q_ref.shape[0]} != env.num_joints="
                        f"{env.num_joints}; truncating/padding"
                    )
                    if q_ref.shape[0] < env.num_joints:
                        pad = np.zeros(env.num_joints - q_ref.shape[0], dtype=np.float32)
                        q_ref = np.concatenate([q_ref, pad])
                    else:
                        q_ref = q_ref[: env.num_joints]

                # Default joint angles (env._default_qpos_actuated): zero in
                # Stage 6; later actor.joint_init feeds robot-specific defaults.
                q_default = getattr(env, "_default_qpos_actuated", np.zeros_like(q_ref))
                q_default = q_default.astype(np.float32)
                ref_action = np.clip(
                    (q_ref - q_default) / max(action_scale, 1e-6),
                    -action_clip, action_clip,
                ).astype(np.float32)

                if noise_std > 0.0:
                    noisy = ref_action + rng.normal(
                        0.0, noise_std, size=ref_action.shape,
                    ).astype(np.float32)
                    noisy = np.clip(noisy, -action_clip, action_clip)
                else:
                    noisy = ref_action

                all_obs.append(np.asarray(obs, dtype=np.float32).copy())
                all_act.append(ref_action.copy())
                idx += 1

                obs, _, terminated, truncated, _ = env.step(noisy)
                done = bool(terminated) or bool(truncated)
                t += control_dt

        observations = np.stack(all_obs, axis=0).astype(np.float32) if all_obs else np.zeros((0, 0), dtype=np.float32)
        actions = np.stack(all_act, axis=0).astype(np.float32) if all_act else np.zeros((0, 0), dtype=np.float32)
        log_info(
            f"[BC] demo buffer ready — {observations.shape[0]:,} transitions, "
            f"{num_trajectories} traj, obs_dim={observations.shape[1] if observations.size else 0}, "
            f"action_dim={actions.shape[1] if actions.size else 0}"
        )
        return DemonstrationBuffer(
            observations=observations,
            actions=actions,
            episode_starts=ep_starts,
        )

    @staticmethod
    def from_env_callback(
        env,
        *,
        num_trajectories: int = 50,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "DemonstrationBuffer":
        """Legacy DEMO path — env exposes ``get_reference_action_for_bc()``.

        Kept for compatibility; new code should prefer
        :meth:`from_motion_clip`.
        """
        if not hasattr(env, "get_reference_action_for_bc"):
            raise AttributeError(
                "from_env_callback: env has no get_reference_action_for_bc(); "
                "use from_motion_clip(env, clip, ...) instead"
            )
        if rng is None:
            rng = np.random.default_rng()

        all_obs: List[np.ndarray] = []
        all_act: List[np.ndarray] = []
        ep_starts: List[int] = []
        idx = 0

        action_clip_val = float(getattr(env, "_action_clip", 1.0))

        for _ in range(int(num_trajectories)):
            obs, _ = env.reset()
            ep_starts.append(idx)
            done = False
            while not done:
                ref_action = np.asarray(
                    env.get_reference_action_for_bc(), dtype=np.float32,
                )
                if noise_std > 0.0:
                    noisy = np.clip(
                        ref_action + rng.normal(0.0, noise_std, size=ref_action.shape),
                        -action_clip_val, action_clip_val,
                    ).astype(np.float32)
                else:
                    noisy = ref_action
                all_obs.append(np.asarray(obs, dtype=np.float32).copy())
                all_act.append(ref_action.copy())
                idx += 1
                obs, _, terminated, truncated, _ = env.step(noisy)
                done = bool(terminated) or bool(truncated)

        observations = np.stack(all_obs, axis=0).astype(np.float32) if all_obs else np.zeros((0, 0), dtype=np.float32)
        actions = np.stack(all_act, axis=0).astype(np.float32) if all_act else np.zeros((0, 0), dtype=np.float32)
        return DemonstrationBuffer(
            observations=observations,
            actions=actions,
            episode_starts=ep_starts,
        )


# ---------------------------------------------------------------------------
# BehavioralCloningTrainer
# ---------------------------------------------------------------------------


class BehavioralCloningTrainer:
    """Phase 1 — pure BC pre-training on the SB3 model's actor network.

    Optimises the actor with supervised loss::

        L = MSE(π(obs), a_ref)   or   Huber(π(obs), a_ref)

    The critic / value networks are left untouched — they will be
    trained from scratch during the subsequent RL phase. Works with
    SAC and PPO policy architectures.
    """

    def __init__(
        self,
        model,
        demo_buffer: DemonstrationBuffer,
        *,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        loss_type: str = "mse",
        cancel_fn: Optional[Callable[[], bool]] = None,
        progress_fn: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        self._model = model
        self._buffer = demo_buffer
        self._epochs = max(1, int(epochs))
        self._lr = float(learning_rate)
        self._batch_size = max(1, int(batch_size))
        self._loss_type = (loss_type or "mse").strip().lower()
        self._cancel = cancel_fn
        self._progress = progress_fn

    def train(self) -> Dict[str, float]:
        """Run pure-BC training. Returns metrics dict."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self._buffer.size == 0:
            log_warning("[BC] empty demo buffer — skipping pre-training")
            return {"bc_loss": float("nan"), "bc_epochs": 0}

        device = next(self._model.policy.parameters()).device
        obs_t = torch.as_tensor(self._buffer.observations, device=device)
        act_t = torch.as_tensor(self._buffer.actions, device=device)
        dataset = TensorDataset(obs_t, act_t)
        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, drop_last=False,
        )

        loss_fn = nn.SmoothL1Loss() if self._loss_type == "huber" else nn.MSELoss()

        actor_params = self._get_actor_parameters()
        if not actor_params:
            log_warning("[BC] could not identify actor parameters — skipping")
            return {"bc_loss": float("nan"), "bc_epochs": 0}

        optimizer = torch.optim.Adam(actor_params, lr=self._lr)

        log_info(
            f"[BC] starting Phase 1 — {self._epochs} epochs, "
            f"{self._buffer.size:,} demos, lr={self._lr:.1e}, "
            f"batch={self._batch_size}, loss={self._loss_type}"
        )

        best_loss = float("inf")
        final_loss = float("inf")

        for epoch in range(self._epochs):
            if self._cancel and self._cancel():
                log_info("[BC] cancelled during pre-training")
                break

            epoch_loss = 0.0
            n_batches = 0
            for obs_batch, act_batch in loader:
                predicted = self._predict_action(obs_batch)
                loss = loss_fn(predicted, act_batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            final_loss = avg_loss
            best_loss = min(best_loss, avg_loss)

            if self._progress:
                self._progress(epoch + 1, self._epochs, avg_loss)

        log_success(
            f"[BC] Phase 1 complete — final_loss={final_loss:.6f} best={best_loss:.6f}"
        )
        return {
            "bc_loss": final_loss,
            "bc_best_loss": best_loss,
            "bc_epochs": self._epochs,
            "bc_demos": self._buffer.size,
        }

    def _get_actor_parameters(self) -> List:
        """Extract actor parameters — works for both SAC and PPO."""
        policy = self._model.policy

        # SAC: separate actor network
        if hasattr(policy, "actor") and hasattr(policy.actor, "parameters"):
            return list(policy.actor.parameters())

        # PPO: action_net + (optional) action_dist + shared MLP + features
        params = []
        if hasattr(policy, "action_net"):
            params.extend(policy.action_net.parameters())
        if hasattr(policy, "action_dist") and hasattr(policy.action_dist, "parameters"):
            try:
                params.extend(policy.action_dist.parameters())
            except (AttributeError, TypeError):
                pass
        if hasattr(policy, "mlp_extractor") and hasattr(policy.mlp_extractor, "policy_net"):
            params.extend(policy.mlp_extractor.policy_net.parameters())
        if hasattr(policy, "features_extractor"):
            params.extend(policy.features_extractor.parameters())
        return params

    def _predict_action(self, obs):
        """Forward pass through the actor's mean head (deterministic)."""
        policy = self._model.policy

        # SAC actor
        if hasattr(policy, "actor"):
            actor = policy.actor
            if hasattr(actor, "get_action_dist_params"):
                features = actor.extract_features(obs, actor.features_extractor)
                latent = actor.latent_pi(features)
                return actor.mu(latent)
            actions, _ = actor.action_log_prob(obs)
            return actions

        # PPO
        features = policy.extract_features(obs, policy.features_extractor)
        if hasattr(policy, "mlp_extractor"):
            latent_pi, _ = policy.mlp_extractor(features)
        else:
            latent_pi = features
        return policy.action_net(latent_pi)

    def build_blend_callback(
        self,
        blend_steps: int = 200_000,
        coef_start: float = 0.5,
    ):
        """Build the Phase 2 :class:`BCBlendCallback` (lives in
        :mod:`il_blend`)."""
        from application.training.il_blend import BCBlendCallback

        return BCBlendCallback(
            model=self._model,
            demo_buffer=self._buffer,
            blend_steps=blend_steps,
            coef_start=coef_start,
            loss_type=self._loss_type,
            batch_size=self._batch_size,
        )


__all__ = [
    "DemonstrationBuffer",
    "BehavioralCloningTrainer",
]
