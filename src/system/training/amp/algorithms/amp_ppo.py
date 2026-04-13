# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#   3. Neither the name of the copyright holder nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# ----------------------------------------------------------------------------
# Vendored into UnitPort from:
#   custom_mods/archives/AMP_for_hardware/rsl_rl/rsl_rl/algorithms/amp_ppo.py
#
# Changes from upstream:
#   - Replay buffer import points at the vendored UnitPort copy
#     (src.system.training.amp.storage.replay_buffer) instead of rsl_rl's
#     fork. ActorCritic and RolloutStorage are still pulled from the
#     system rsl_rl (isaac venv) so we pick up whatever version Isaac Lab
#     ships with.
#   - ``from rsl_rl.modules import ActorCritic`` and
#     ``from rsl_rl.storage import RolloutStorage`` are deferred to
#     __init__ to avoid a module-import-time failure in the main UnitPort
#     venv (which does not have rsl_rl).
#
# This module will only run inside the Isaac Sim venv. Main-venv code
# (tests, UI) must not import it.
# ----------------------------------------------------------------------------

"""AMP-flavored PPO: vanilla PPO + discriminator loss on replay transitions.

The public shape is the same as the upstream ``AMPPPO`` class. Our
``AMPOnPolicyRunner`` (next door) constructs one of these per training
run. Main-venv code should NEVER import this file — guard imports
behind a lazy loader inside the launcher's AMP branch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

# Vendored UnitPort copies — fully version-agnostic vs whatever rsl_rl
# Isaac Lab happens to ship. The AMP_for_hardware fork was written
# against rsl_rl 1.x's ``RolloutStorage`` which doesn't exist in
# rsl_rl 2.x; we ship the 1.x file in-tree under
# ``src/system/training/amp/storage/rollout_storage.py``.
from src.system.training.amp.storage.replay_buffer import ReplayBuffer
from src.system.training.amp.storage.rollout_storage import RolloutStorage


class AMPPPO:
    """PPO update loop with an adversarial style-reward discriminator.

    Apart from the discriminator + replay buffer arguments, this class
    is structurally identical to upstream PPO — same rollout/update
    split, same KL-adaptive learning rate schedule, same clipped
    surrogate + value losses. All AMP-specific code lives in
    ``update()``.
    """

    # The real ActorCritic class is resolved at construction time, not
    # at import time, because rsl_rl lives only inside the Isaac venv.
    actor_critic: object

    def __init__(
        self,
        actor_critic,
        discriminator,
        amp_data,
        amp_normalizer,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        amp_replay_buffer_size: int = 1_000_000,
        amp_disc_grad_penalty: float = 10.0,
        amp_disc_lr: float = 0.0,
        amp_disc_label_smoothing: float = 0.9,
        min_std=None,
    ) -> None:
        # ``RolloutStorage`` is the vendored UnitPort copy — fully
        # version-agnostic relative to whatever rsl_rl Isaac Lab ships.
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.amp_disc_grad_penalty = amp_disc_grad_penalty
        self.amp_disc_label_smoothing = float(amp_disc_label_smoothing)
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later via init_storage()

        # Optimizer for policy and discriminator.
        # When amp_disc_lr > 0, the discriminator uses its own learning
        # rate; otherwise it shares the policy LR.
        disc_lr = amp_disc_lr if amp_disc_lr > 0 else learning_rate
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "lr": disc_lr,
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.amp_linear.parameters(),
                "lr": disc_lr,
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ) -> None:
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self) -> None:
        self.actor_critic.test()

    def train_mode(self) -> None:
        self.actor_critic.train()

    # ------------------------------------------------------------------
    # Rollout hooks — called from AMPOnPolicyRunner
    # ------------------------------------------------------------------

    def act(self, obs, critic_obs, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs) -> None:
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ------------------------------------------------------------------
    # Main update — combined PPO + discriminator loss
    # ------------------------------------------------------------------

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        # B4 (AMP fix plan): discriminator accuracy. The current LSGAN
        # loss targets policy→−1 / expert→+1 so midpoint = 0. After B1/B2
        # we move to BCEWithLogits where the test (logit>0 ≡ sigmoid>0.5)
        # is identical — so this thresholding stays stable across the
        # lineage switch.
        mean_accuracy_policy = 0.0
        mean_accuracy_expert = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(
                aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(
                aug_critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL-adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Discriminator loss
            #
            # NBE1 (AMP fix plan): the pre-fix code rebound
            # ``policy_state``/``expert_state`` to their normalized
            # versions at the top of this block, then fed the same
            # (now-normalized) tensors back to ``amp_normalizer.update``
            # at the bottom of the block. That made the running mean/
            # std collapse toward (0, 1) regardless of the true AMP obs
            # distribution — the normalizer was effectively frozen from
            # iter 0.
            #
            # Fix: keep raw copies under distinct names, normalize into
            # separate tensors used for both the main forward pass AND
            # the gradient penalty (so the two are computed in the same
            # input space), and feed the RAW copies to the normalizer
            # update after ``optimizer.step()``.
            #
            # Also fixes a latent inconsistency: ``compute_grad_pen``
            # was previously called with the raw ``sample_amp_expert``
            # while the main disc forward pass used normalized inputs.
            # Passing the normalized expert tensors here aligns the two
            # paths so the penalty and the loss land on the same
            # manifold.
            raw_policy_state, raw_policy_next_state = sample_amp_policy
            raw_expert_state, raw_expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(
                        raw_policy_state, self.device
                    )
                    policy_next_state = self.amp_normalizer.normalize_torch(
                        raw_policy_next_state, self.device
                    )
                    expert_state = self.amp_normalizer.normalize_torch(
                        raw_expert_state, self.device
                    )
                    expert_next_state = self.amp_normalizer.normalize_torch(
                        raw_expert_next_state, self.device
                    )
            else:
                policy_state = raw_policy_state
                policy_next_state = raw_policy_next_state
                expert_state = raw_expert_state
                expert_next_state = raw_expert_next_state

            policy_d = self.discriminator(
                torch.cat([policy_state, policy_next_state], dim=-1)
            )
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_next_state], dim=-1)
            )
            # B1 (AMP fix plan): BCEWithLogits replaces the AMP_for_hardware
            # LSGAN targets (+1/−1 with MSE). The discriminator head
            # already outputs a raw logit (no sigmoid) so it is a drop-in
            # input to ``binary_cross_entropy_with_logits``. Targets flip
            # from {+1, −1} to {1, 0}. The accuracy threshold (logit > 0
            # ⇔ sigmoid > 0.5) is unchanged — see B4 accuracy metric.
            #
            # Two-sided label smoothing (adapted from Salimans 2016).
            #
            # Salimans recommends ONE-sided smoothing for standard GANs
            # because the adversarial generator can exploit a softened
            # policy boundary to produce "fake" samples that fool the
            # disc.  For AMP that concern does NOT apply — the policy is
            # not adversarially exploiting the disc, it is genuinely
            # trying to match the expert motion distribution.  One-sided
            # smoothing creates a pathological equilibrium where the
            # disc drives the policy logit to -∞ (no upper bound on
            # ``BCE(logit, 0)`` as logit → -∞, the loss just asymptotes
            # to zero).  Once the policy logit is stuck at -6 or lower,
            # the BCE gradient vanishes on both sides and the disc
            # freezes — observed as ``amp=0.164`` constant across many
            # iters, ``pol`` stuck near -6, and style reward collapsing
            # as the policy drifts under entropy regularization alone.
            #
            # Two-sided fix: expert target = ``s`` (0.9 default),
            # policy target = ``1 - s`` (0.1 default).  The disc has a
            # clear, finite equilibrium at expert_logit = +log(s/(1-s))
            # and policy_logit = -log(s/(1-s)) — for s=0.9, both sit at
            # ±2.2, which keeps softplus(pol) = 0.105 per step, giving
            # a strong and stable style reward gradient for the policy
            # to learn from.  This also keeps the disc's BCE loss
            # nonzero and actively moving throughout training.
            bce = torch.nn.functional.binary_cross_entropy_with_logits
            expert_target = self.amp_disc_label_smoothing
            policy_target = 1.0 - expert_target
            expert_loss = bce(
                expert_d,
                expert_target * torch.ones_like(expert_d),
            )
            policy_loss = bce(
                policy_d,
                policy_target * torch.ones_like(policy_d),
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state,
                lambda_=self.amp_disc_grad_penalty,
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            if (not self.actor_critic.fixed_std) and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(
                    min=self.min_std
                )

            if self.amp_normalizer is not None:
                # Feed RAW tensors to the normalizer so running mean /
                # std track the real AMP obs distribution rather than
                # collapsing toward (0, 1). Per spec §1.discriminator.
                # normalizer: "Update using RAW (un-normalized) tensors
                # AFTER each optimizer step, under torch.no_grad()."
                self.amp_normalizer.update(raw_policy_state.cpu().numpy())
                self.amp_normalizer.update(raw_expert_state.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            with torch.no_grad():
                mean_accuracy_policy += (policy_d < 0).float().mean().item()
                mean_accuracy_expert += (expert_d > 0).float().mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_accuracy_policy /= num_updates
        mean_accuracy_expert /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
        )
