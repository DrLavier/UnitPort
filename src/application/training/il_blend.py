"""IL+RL fusion — Phase 2 BC blend during RL fine-tune.

REWRITE-WITH-DEMO-REF from DEMO ``BCBlendCallback`` in
``src/system/training/behavioral_cloning.py`` (lines 357-498), split out
into its own module per Stage 9 plan.

The callback adds an auxiliary BC loss to the actor's gradient during
SB3 policy updates, with the BC coefficient λ decaying linearly from
``coef_start`` to 0 over ``blend_steps`` RL timesteps. Once
``num_timesteps > blend_steps`` the callback becomes a no-op (pure RL).

Implements the standard practice from:
    * Rajeswaran et al., DAPG ("Learning Complex Dexterous Manipulation
      with Deep Reinforcement Learning and Demonstrations")
    * Peng et al., DeepMimic / AMP

The callback does NOT call ``optimizer.step()`` — it only calls
``loss.backward()`` so the gradient accumulates into the actor
parameters and SB3's own optimizer applies it on the next gradient
step. This is the standard approach for SB3 auxiliary losses.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from unitport_sdk import log_info


class BCBlendCallback:
    """SB3 callback that blends a decaying BC loss into the actor update.

    Lazy-built: :meth:`build` returns a ``BaseCallback`` subclass that
    SB3's ``model.learn(callback=...)`` accepts.
    """

    def __init__(
        self,
        model,
        demo_buffer,
        *,
        blend_steps: int = 200_000,
        coef_start: float = 0.5,
        loss_type: str = "mse",
        batch_size: int = 256,
    ) -> None:
        self._model = model
        self._buffer = demo_buffer
        self._blend_steps = max(1, int(blend_steps))
        self._coef_start = float(coef_start)
        self._batch_size = min(int(batch_size), max(1, demo_buffer.size))
        self._loss_type = (loss_type or "mse").strip().lower()
        self._step = 0
        self._active = (
            demo_buffer.size > 0
            and self._blend_steps > 0
            and self._coef_start > 0.0
        )
        self._last_log_step = -1

    # ------------------------------------------------------------------
    # Inspection (testable without SB3)
    # ------------------------------------------------------------------

    def current_coef(self, step: Optional[int] = None) -> float:
        """Linear-decay BC coefficient at *step* (defaults to internal)."""
        if step is None:
            step = self._step
        if not self._active:
            return 0.0
        progress = min(1.0, step / self._blend_steps)
        return float(self._coef_start * (1.0 - progress))

    @property
    def is_active(self) -> bool:
        return bool(self._active)

    @property
    def step_count(self) -> int:
        return int(self._step)

    # ------------------------------------------------------------------
    # SB3 callback
    # ------------------------------------------------------------------

    def build(self):
        """Return a ``BaseCallback`` instance for ``model.learn(callback=...)``."""
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Inner(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__(verbose=0)

            def _on_step(inner_self) -> bool:
                if not outer._active:
                    return True
                outer._step = int(inner_self.num_timesteps)
                if outer._step > outer._blend_steps:
                    if outer._active:
                        outer._active = False
                        log_info(
                            f"[BC] Phase 2 complete at step {outer._step:,} — "
                            "auxiliary BC loss deactivated, entering pure RL"
                        )
                    return True
                outer._apply_bc_gradient()
                return True

            def _on_rollout_end(inner_self) -> None:
                if not outer._active:
                    return
                step = outer._step
                interval = max(1, outer._blend_steps // 10)
                if step - outer._last_log_step >= interval:
                    outer._last_log_step = step
                    coef = outer.current_coef(step)
                    pct = min(100.0, 100.0 * step / outer._blend_steps)
                    log_info(
                        f"[BC] Phase 2 blend: step {step:,}/{outer._blend_steps:,} "
                        f"({pct:.0f}%)  λ={coef:.4f}"
                    )

        return _Inner()

    def _apply_bc_gradient(self) -> None:
        """Sample a BC mini-batch and inject auxiliary gradient."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        coef = self.current_coef()
        if coef < 1e-6:
            return

        policy = self._model.policy
        device = next(policy.parameters()).device

        idx = np.random.randint(0, self._buffer.size, size=self._batch_size)
        obs_batch = torch.as_tensor(
            self._buffer.observations[idx], device=device,
        )
        act_batch = torch.as_tensor(
            self._buffer.actions[idx], device=device,
        )

        predicted = self._predict_action(obs_batch)

        if self._loss_type == "huber":
            loss = F.smooth_l1_loss(predicted, act_batch)
        else:
            loss = F.mse_loss(predicted, act_batch)

        scaled_loss = coef * loss
        scaled_loss.backward()
        # Note: no optimizer.step() — SB3's optimizer applies the
        # accumulated gradient on its next update.

    def _predict_action(self, obs):
        """Mirror :meth:`BehavioralCloningTrainer._predict_action`."""
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


__all__ = ["BCBlendCallback"]
