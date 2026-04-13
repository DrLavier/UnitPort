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
#   custom_mods/archives/AMP_for_hardware/rsl_rl/rsl_rl/modules/actor_critic.py
#
# Why vendor instead of importing from rsl_rl?
#
#   Modern rsl_rl 2.x (the version Isaac Lab 5.x ships with) dropped the
#   ``ActorCritic`` god-class entirely. ``rsl_rl.modules`` only exposes
#   low-level building blocks now (CNN / MLP / RNN / GaussianDistribution)
#   and the high-level RL algorithm (``PPO``) constructs the actor + critic
#   internally from a config dict — there is no longer a single object
#   the AMP_for_hardware ``AMPPPO`` class can attach to. Vendoring this
#   1.x-style ActorCritic in-tree makes the entire AMP path version-agnostic
#   relative to whatever rsl_rl Isaac Lab happens to ship.
#
# Changes from upstream:
#   - Removed unused ``from torch.nn.modules import rnn`` import.
#   - Replaced ``print(f"Actor MLP: ...")`` debug prints at construction
#     time with a single ``[UnitPort][AMP] AC built ...`` line so the
#     training log isn't polluted with the same multi-line dump per env.
# ----------------------------------------------------------------------------

"""Combined actor + critic MLP for the AMP_for_hardware-style runner.

A single ``nn.Module`` that bundles:
  - actor MLP    (obs → action_mean)
  - critic MLP   (critic_obs → value)
  - learnable per-action ``std`` parameter for the Gaussian policy
  - the standard rsl_rl 1.x ``act / evaluate / get_actions_log_prob /
    action_mean / action_std / entropy / reset / update_distribution``
    surface that ``AMPPPO`` (also vendored) calls into.

Non-recurrent only — ``ActorCriticRecurrent`` is not vendored because
phase_3 only ships the feed-forward MLP path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        fixed_std: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "[UnitPort][AMP] ActorCritic ignored unexpected kwargs: "
                + str(list(kwargs.keys())),
                flush=True,
            )
        super().__init__()

        act_fn = _get_activation(activation)

        # ── Policy (actor) MLP ──
        actor_layers = [nn.Linear(num_actor_obs, actor_hidden_dims[0]), act_fn]
        for layer_idx in range(len(actor_hidden_dims)):
            if layer_idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_idx], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_idx], actor_hidden_dims[layer_idx + 1])
                )
                actor_layers.append(act_fn)
        self.actor = nn.Sequential(*actor_layers)

        # ── Value function (critic) MLP ──
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act_fn]
        for layer_idx in range(len(critic_hidden_dims)):
            if layer_idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_idx], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[layer_idx], critic_hidden_dims[layer_idx + 1])
                )
                critic_layers.append(act_fn)
        self.critic = nn.Sequential(*critic_layers)

        # Single concise summary line — much friendlier than upstream's
        # full-tree dump per env on a 4096-env launch.
        print(
            f"[UnitPort][AMP] ActorCritic: actor={list(actor_hidden_dims)} → {num_actions}, "
            f"critic={list(critic_hidden_dims)} → 1, activation={activation}",
            flush=True,
        )

        # ── Action noise ──
        # Per-action std vector. Either a fixed buffer or a learned
        # nn.Parameter. The PPO update rules in AMPPPO match the
        # learned-std behaviour (entropy / KL terms read from .std).
        self.fixed_std = bool(fixed_std)
        std_init = init_noise_std * torch.ones(num_actions)
        self.std = (
            torch.tensor(std_init) if self.fixed_std else nn.Parameter(std_init)
        )

        self.distribution = None
        # disable args validation for speedup (matches upstream)
        Normal.set_default_validate_args = False

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def init_weights(sequential, scales):
        # not used at the moment, kept for upstream API compatibility
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None) -> None:
        # No-op for non-recurrent policies. AMPOnPolicyRunner calls this
        # after each env step in case the underlying policy needs to
        # reset hidden states.
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0.0 + std)

    # ------------------------------------------------------------------
    # Public API used by AMPPPO
    # ------------------------------------------------------------------

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        # Deterministic inference (used at policy export / playback time).
        return self.actor(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)


def _get_activation(name: str) -> nn.Module:
    name = (name or "elu").lower()
    table = {
        "elu":     nn.ELU(),
        "selu":    nn.SELU(),
        "relu":    nn.ReLU(),
        "crelu":   nn.ReLU(),
        "lrelu":   nn.LeakyReLU(),
        "tanh":    nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name not in table:
        print(
            f"[UnitPort][AMP] Unknown activation {name!r}; defaulting to ELU.",
            flush=True,
        )
        return nn.ELU()
    return table[name]
