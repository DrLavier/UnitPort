#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioral Cloning (BC) pre-training for motion imitation.

Implements the industry-standard 2-phase imitation learning pipeline
(DeepMimic / AMP / DReCon):

    Phase 1 — Pure BC: supervised MSE/Huber loss on (obs, a_ref) pairs
              collected by replaying the reference motion through the env.
    Phase 2 — BC + RL blend: auxiliary BC loss decays linearly over N RL
              timesteps via an SB3 callback, ensuring smooth handoff.

Usage (called by SB3Trainer before ``model.learn()``):

    from src.system.training.behavioral_cloning import (
        DemonstrationBuffer,
        BehavioralCloningTrainer,
        BCBlendCallback,
    )

    demo_buf = DemonstrationBuffer.collect(env, num_trajectories=50)
    bc = BehavioralCloningTrainer(model, demo_buf, cfg)
    bc.train()                    # Phase 1: pure BC
    blend_cb = bc.build_blend_callback()  # Phase 2 callback for model.learn()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DemonstrationBuffer
# ---------------------------------------------------------------------------

@dataclass
class DemonstrationBuffer:
    """Stores (observation, reference_action) pairs for BC supervision.

    Collected by rolling the reference motion through the environment and
    recording what the *expert* (reference trajectory) would do at each step.
    """

    observations: np.ndarray    # (N, obs_dim)  float32
    actions: np.ndarray         # (N, act_dim)  float32
    episode_starts: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.observations)

    @staticmethod
    def collect(
        env,
        num_trajectories: int = 50,
        noise_std: float = 0.0,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> "DemonstrationBuffer":
        """Collect demonstration data by replaying reference motion.

        For each trajectory the env is reset, then at every control step the
        reference action is read via ``env.get_reference_action_for_bc()`` and
        applied (with optional noise) to step the env.  The resulting
        (observation, reference_action) pairs form the BC dataset.

        Parameters
        ----------
        env
            A single (unwrapped) ``UnitreeGymEnv`` instance.  Must have a
            reference motion loaded (``_ref_frames is not None``).
        num_trajectories
            How many episodes to collect.
        noise_std
            Gaussian noise σ added to the reference action before stepping.
            Small values (0.01–0.05) broaden the state distribution visited
            during collection (DAgger-like) without degrading quality.
        log_fn
            Optional logging callback.
        """
        all_obs: List[np.ndarray] = []
        all_act: List[np.ndarray] = []
        ep_starts: List[int] = []
        idx = 0

        for traj_i in range(num_trajectories):
            obs, _ = env.reset()
            ep_starts.append(idx)
            done = False

            while not done:
                # The reference action in policy action-space coordinates
                ref_action = env.get_reference_action_for_bc()

                # Optional noise for distribution broadening
                if noise_std > 0.0:
                    noisy_action = ref_action + np.random.normal(
                        0.0, noise_std, size=ref_action.shape
                    ).astype(np.float32)
                    noisy_action = np.clip(
                        noisy_action,
                        -env._action_clip,
                        env._action_clip,
                    )
                else:
                    noisy_action = ref_action

                all_obs.append(obs.copy())
                all_act.append(ref_action.copy())
                idx += 1

                obs, _, terminated, truncated, _ = env.step(noisy_action)
                done = terminated or truncated

            if log_fn and (traj_i + 1) % max(1, num_trajectories // 5) == 0:
                log_fn(
                    f"[BC] Demo collection: {traj_i + 1}/{num_trajectories} "
                    f"trajectories ({idx:,} transitions)"
                )

        observations = np.array(all_obs, dtype=np.float32)
        actions = np.array(all_act, dtype=np.float32)

        if log_fn:
            log_fn(
                f"[BC] Demo buffer ready: {observations.shape[0]:,} transitions, "
                f"{num_trajectories} trajectories, "
                f"obs_dim={observations.shape[1]}, act_dim={actions.shape[1]}"
            )

        return DemonstrationBuffer(
            observations=observations,
            actions=actions,
            episode_starts=ep_starts,
        )


# ---------------------------------------------------------------------------
# BehavioralCloningTrainer
# ---------------------------------------------------------------------------

class BehavioralCloningTrainer:
    """Phase 1: pure BC pre-training on the SB3 model's actor network.

    Directly optimises the actor (policy) network with supervised loss::

        L = MSE(π(obs), a_ref)   or   Huber(π(obs), a_ref)

    The critic/value networks are left untouched — they will be trained
    from scratch during the subsequent RL phase.

    Works with both SAC and PPO policy architectures from SB3.
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
        log_fn: Optional[Callable[[str], None]] = None,
        cancel_fn: Optional[Callable[[], bool]] = None,
        progress_fn: Optional[Callable[[int, int, float], None]] = None,
    ):
        self._model = model
        self._buffer = demo_buffer
        self._epochs = max(1, epochs)
        self._lr = learning_rate
        self._batch_size = max(1, batch_size)
        self._loss_type = loss_type.strip().lower()
        self._log = log_fn
        self._cancel = cancel_fn
        self._progress = progress_fn

    def train(self) -> Dict[str, float]:
        """Run the pure-BC training loop.  Returns final metrics dict."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self._buffer.size == 0:
            if self._log:
                self._log("[BC] Empty demo buffer — skipping BC pre-training")
            return {"bc_loss": float("nan"), "bc_epochs": 0}

        device = next(self._model.policy.parameters()).device

        obs_t = torch.as_tensor(self._buffer.observations, device=device)
        act_t = torch.as_tensor(self._buffer.actions, device=device)
        dataset = TensorDataset(obs_t, act_t)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Build loss function
        if self._loss_type == "huber":
            loss_fn = nn.SmoothL1Loss()
        else:
            loss_fn = nn.MSELoss()

        # Collect actor parameters — works for both SAC and PPO
        actor_params = self._get_actor_parameters()
        if not actor_params:
            if self._log:
                self._log("[BC] Could not identify actor parameters — skipping")
            return {"bc_loss": float("nan"), "bc_epochs": 0}

        optimizer = torch.optim.Adam(actor_params, lr=self._lr)

        if self._log:
            self._log(
                f"[BC] Starting Phase 1 — Pure BC: {self._epochs} epochs, "
                f"{self._buffer.size:,} demos, lr={self._lr:.1e}, "
                f"batch={self._batch_size}, loss={self._loss_type}"
            )

        best_loss = float("inf")
        final_loss = float("inf")

        for epoch in range(self._epochs):
            if self._cancel and self._cancel():
                if self._log:
                    self._log("[BC] Cancelled during BC pre-training")
                break

            epoch_loss = 0.0
            n_batches = 0

            for obs_batch, act_batch in loader:
                predicted = self._predict_action(obs_batch)
                loss = loss_fn(predicted, act_batch)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            final_loss = avg_loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            if self._progress:
                self._progress(epoch + 1, self._epochs, avg_loss)

            if self._log and (epoch + 1) % max(1, self._epochs // 10) == 0:
                self._log(
                    f"[BC] Epoch {epoch + 1}/{self._epochs}  "
                    f"loss={avg_loss:.6f}  best={best_loss:.6f}"
                )

        if self._log:
            self._log(
                f"[BC] Phase 1 complete — final_loss={final_loss:.6f}  "
                f"best_loss={best_loss:.6f}"
            )

        return {
            "bc_loss": final_loss,
            "bc_best_loss": best_loss,
            "bc_epochs": self._epochs,
            "bc_demos": self._buffer.size,
        }

    def _get_actor_parameters(self) -> List:
        """Extract actor network parameters from the SB3 model."""
        import torch

        policy = self._model.policy

        # SAC: policy has a separate actor network
        if hasattr(policy, "actor") and hasattr(policy.actor, "parameters"):
            return list(policy.actor.parameters())

        # PPO: the action_net + action_dist are the "actor"
        params: List[torch.nn.Parameter] = []
        if hasattr(policy, "action_net"):
            params.extend(policy.action_net.parameters())
        if hasattr(policy, "action_dist"):
            params.extend(policy.action_dist.parameters())
        # Also include the shared MLP features extractor + mlp_extractor.policy_net
        if hasattr(policy, "mlp_extractor") and hasattr(policy.mlp_extractor, "policy_net"):
            params.extend(policy.mlp_extractor.policy_net.parameters())
        if hasattr(policy, "features_extractor"):
            params.extend(policy.features_extractor.parameters())

        return params

    def _predict_action(self, obs: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the actor to get the mean action prediction.

        For SAC: uses actor.action_dist to get the mean (deterministic mode).
        For PPO: uses the policy's forward to extract action logits/mean.
        """
        policy = self._model.policy

        # SAC actor
        if hasattr(policy, "actor"):
            # SB3 SAC actor forward returns (actions, log_prob)
            # We use the mu_net directly for deterministic BC training
            actor = policy.actor
            if hasattr(actor, "get_action_dist_params"):
                # SB3 >= 2.0: get_action_dist_params returns the mean + log_std
                features = actor.extract_features(obs, actor.features_extractor)
                latent = actor.latent_pi(features)
                mean_actions = actor.mu(latent)
                return mean_actions
            # Fallback: full forward
            actions, _ = actor.action_log_prob(obs)
            return actions

        # PPO: forward through shared extractor + policy net + action_net
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
    ) -> "BCBlendCallback":
        """Create the Phase 2 callback for BC+RL blended training."""
        return BCBlendCallback(
            model=self._model,
            demo_buffer=self._buffer,
            blend_steps=blend_steps,
            coef_start=coef_start,
            loss_type=self._loss_type,
            batch_size=self._batch_size,
            log_fn=self._log,
        )


# ---------------------------------------------------------------------------
# BCBlendCallback  (Phase 2: BC + RL blend)
# ---------------------------------------------------------------------------

class BCBlendCallback:
    """SB3-compatible callback that adds an auxiliary BC loss during RL.

    The BC coefficient λ decays linearly from ``coef_start`` to 0 over
    ``blend_steps`` RL timesteps.  At each gradient update the callback
    samples a mini-batch from the demo buffer, computes the BC loss, and
    adds ``λ · BC_loss`` to the actor's gradient.

    After ``blend_steps`` the callback becomes a no-op (pure RL).

    This implements the standard practice from:
    - Rajeswaran et al., "Learning Complex Dexterous Manipulation with
      Deep Reinforcement Learning and Demonstrations" (DAPG)
    - Peng et al., DeepMimic / AMP
    """

    def __init__(
        self,
        model,
        demo_buffer: DemonstrationBuffer,
        blend_steps: int = 200_000,
        coef_start: float = 0.5,
        loss_type: str = "mse",
        batch_size: int = 256,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self._model = model
        self._buffer = demo_buffer
        self._blend_steps = max(1, blend_steps)
        self._coef_start = coef_start
        self._batch_size = min(batch_size, max(1, demo_buffer.size))
        self._loss_type = loss_type
        self._log = log_fn
        self._step = 0
        self._active = demo_buffer.size > 0 and blend_steps > 0 and coef_start > 0.0
        self._last_log_step = -1

    def build(self):
        """Return an SB3 BaseCallback instance for use with model.learn()."""
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Inner(BaseCallback):
            def __init__(inner_self):
                super().__init__(verbose=0)

            def _on_step(inner_self) -> bool:
                if not outer._active:
                    return True
                outer._step = inner_self.num_timesteps
                if outer._step > outer._blend_steps:
                    # Phase 2 done — deactivate permanently
                    if outer._active:
                        outer._active = False
                        if outer._log:
                            outer._log(
                                f"[BC] Phase 2 complete at step {outer._step:,} — "
                                "BC auxiliary loss deactivated, entering pure RL"
                            )
                    return True
                outer._apply_bc_gradient()
                return True

            def _on_rollout_end(inner_self) -> None:
                # Log blend status periodically
                if not outer._active:
                    return
                coef = outer._current_coef()
                step = outer._step
                interval = max(1, outer._blend_steps // 10)
                if step - outer._last_log_step >= interval:
                    outer._last_log_step = step
                    if outer._log:
                        pct = min(100.0, 100.0 * step / outer._blend_steps)
                        outer._log(
                            f"[BC] Phase 2 blend: step {step:,}/{outer._blend_steps:,} "
                            f"({pct:.0f}%)  λ={coef:.4f}"
                        )

        return _Inner()

    def _current_coef(self) -> float:
        """Linearly decaying BC coefficient."""
        progress = min(1.0, self._step / self._blend_steps)
        return self._coef_start * (1.0 - progress)

    def _apply_bc_gradient(self) -> None:
        """Sample a BC mini-batch and inject auxiliary gradient."""
        import torch
        import torch.nn as nn

        coef = self._current_coef()
        if coef < 1e-6:
            return

        policy = self._model.policy
        device = next(policy.parameters()).device

        # Sample random mini-batch from demo buffer
        indices = np.random.randint(0, self._buffer.size, size=self._batch_size)
        obs_batch = torch.as_tensor(
            self._buffer.observations[indices], device=device
        )
        act_batch = torch.as_tensor(
            self._buffer.actions[indices], device=device
        )

        # Forward pass through actor
        predicted = self._predict_action(obs_batch)

        if self._loss_type == "huber":
            loss = nn.functional.smooth_l1_loss(predicted, act_batch)
        else:
            loss = nn.functional.mse_loss(predicted, act_batch)

        # Scale by blend coefficient and backprop through actor only
        scaled_loss = coef * loss
        scaled_loss.backward()
        # Note: we do NOT call optimizer.step() — the gradient accumulates
        # into the actor parameters and will be applied by SB3's own
        # optimizer on the next gradient step.  This is the standard
        # approach for auxiliary losses in SB3.

    def _predict_action(self, obs: "torch.Tensor") -> "torch.Tensor":
        """Forward pass — mirrors BehavioralCloningTrainer._predict_action."""
        policy = self._model.policy
        if hasattr(policy, "actor"):
            actor = policy.actor
            if hasattr(actor, "get_action_dist_params"):
                features = actor.extract_features(obs, actor.features_extractor)
                latent = actor.latent_pi(features)
                return actor.mu(latent)
            actions, _ = actor.action_log_prob(obs)
            return actions
        features = policy.extract_features(obs, policy.features_extractor)
        if hasattr(policy, "mlp_extractor"):
            latent_pi, _ = policy.mlp_extractor(features)
        else:
            latent_pi = features
        return policy.action_net(latent_pi)
