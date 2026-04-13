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
#   custom_mods/archives/AMP_for_hardware/rsl_rl/rsl_rl/algorithms/
#     amp_discriminator.py
#
# Changes from upstream:
#   - Removed ``from rsl_rl.utils import utils`` (unused in this file).
#   - No behavioral changes.
# ----------------------------------------------------------------------------

"""AMP discriminator network.

A small MLP trunk + linear head that scores transition pairs
``(s_t, s_{t+1})`` as either "expert" (mocap) or "policy" (generated
by the current actor). The AMP loss drives the policy to produce
transitions that look expert-like, yielding a style reward in addition
to any task reward.

Per AMP_design.yaml §4.amp_backend, this module is main-venv safe —
it depends only on torch. The actual training loop that uses this
discriminator (``amp_ppo.py``) requires rsl_rl and runs inside the
Isaac Sim venv.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import autograd


class AMPDiscriminator(nn.Module):
    """Transition-pair discriminator for AMP.

    Parameters
    ----------
    input_dim:
        Size of the concatenated ``(s_t, s_{t+1})`` input — i.e. twice
        the ``amp_obs_dim`` of the motion clip.
    amp_reward_coef:
        Scale applied to the clipped reward in ``predict_amp_reward``.
    hidden_layer_sizes:
        Iterable of hidden MLP widths. ReLU between each.
    device:
        Torch device the network lives on.
    task_reward_lerp:
        Config pass-through only. Blend coefficient between style and
        task reward: ``r_total = (1 - lerp) * style_r + lerp * task_r``.
        Stored on this module so ``AMPOnPolicyRunner`` can read it, but
        the actual mixing happens in the runner (per spec §1.reward_mixing
        in ``AMP_EXAMPLE/AMP-PPO_design.yaml`` — env stays AMP-agnostic,
        mixing is the runner's job). Default 0.5.
    """

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: Iterable[int],
        device,
        task_reward_lerp: float = 0.5,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.amp_reward_coef = float(amp_reward_coef)

        # LeakyReLU instead of ReLU: GAN discriminators are prone to
        # dying-ReLU collapse because the initial policy-vs-expert
        # distribution is so dissimilar that the first few updates push
        # many trunk neurons into permanently negative pre-activations
        # (∂ReLU/∂x = 0 for x<0 → dead neuron, unrecoverable).  Once
        # enough neurons die, the R1 gradient penalty silently becomes
        # zero (observed as ``gp=0.000`` in training logs) because
        # ∂d/∂x can only flow through the remaining live neurons.  The
        # surviving subnetwork is small enough to classify policy-vs-
        # expert perfectly (``acc_p=acc_e=1.00``), yet too constrained
        # to provide a useful style-reward gradient — the classic AMP
        # "style reward peaks then crashes" failure mode.
        #
        # LeakyReLU with negative slope 0.2 keeps a small gradient
        # flowing through every neuron regardless of sign, preventing
        # permanent death.  Slope 0.2 is the PatchGAN / WGAN default;
        # see Radford et al. 2015 ("DCGAN") and every modern GAN trunk
        # since.  This is also what amp-rsl-rl uses for the same task.
        layers = []
        curr_in = int(input_dim)
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in, int(hidden_dim)))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            curr_in = int(hidden_dim)
        self.trunk = nn.Sequential(*layers).to(device)
        self.amp_linear = nn.Linear(curr_in, 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = float(task_reward_lerp)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of concatenated ``(s_t, s_{t+1})`` pairs."""
        h = self.trunk(x)
        return self.amp_linear(h)

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP style, applied to expert samples)
    # ------------------------------------------------------------------

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """Penalise non-zero gradient norms on expert transitions.

        Unlike standard WGAN-GP, the penalty targets ``|grad| = 0``
        (not 1). This stabilises AMP training empirically — the paper
        calls it a "zero-centered" gradient penalty.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return float(lambda_) * (grad.norm(2, dim=1) - 0.0).pow(2).mean()

    # ------------------------------------------------------------------
    # Inference: disc score → style reward (no grad)
    # ------------------------------------------------------------------

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        normalizer=None,
    ):
        """Compute the AMP **style** reward for a batch of policy transitions.

        Returns a tuple ``(style_reward, disc_score)`` where ``style_reward``
        has shape ``(batch,)`` and ``disc_score`` is the raw MLP logit
        for logging.

        Task-reward mixing is intentionally NOT done here. Per spec
        §1.reward_mixing the runner owns the `(1-l)*style + l*task`
        blend so that the env stays AMP-agnostic and the two components
        can be logged separately. Callers should read
        ``self.task_reward_lerp`` and do the lerp themselves.

        Reward formula (B2 in ``knowledge_base/amp_fix.yaml``):
            ``style_r = coef * softplus(logit)``

        This is the amp-rsl-rl / Peng 2021 formula, equivalent to
        ``-log(1 - sigmoid(logit))``, which is the natural paired
        reward when the discriminator is trained with BCEWithLogits
        (B1). It is smooth and strictly positive everywhere; the
        previous AMP_for_hardware formula ``max(0, 1 − 0.25*(d−1)²)``
        had zero gradient outside ``|d−1| > 2`` and misread a BCE logit
        as an LSGAN score.
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            style_reward = self.amp_reward_coef * torch.nn.functional.softplus(d)
            self.train()
        return style_reward.squeeze(-1), d
