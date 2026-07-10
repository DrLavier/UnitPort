# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ActorCriticRecurrent — GRU/LSTM recurrent policy for the AMP-PPO path.

缺口① — legged_gym's G1 trains an ``ActorCriticRecurrent`` (GRU). The
vendored :class:`~application.training.amp.algorithms.actor_critic.ActorCritic`
is feed-forward only (``is_recurrent = False``); the AMP PPO algorithm
(``rsl_amp_ppo.py``) and rollout storage already carry dormant
``is_recurrent`` branches + hidden-state buffers copied from upstream rsl_rl.
This class lights them up.

Design (decision R5): **recurrent actor + MLP critic**. The actor is a memory
cell (GRU or LSTM) followed by an MLP head; the critic stays a plain MLP over
the (possibly privileged) critic obs — the value function is only used at
training time and does not need to be exported / replayed.

State-dict layout (the C5↔B1 export contract — DO NOT drift without updating
``onnx_export._build_recurrent_actor``): the whole actor — memory + head —
lives under the single ``actor.`` namespace so the exporter's existing
``actor.``-prefix extraction captures both::

    actor.rnn.weight_ih_l0   actor.rnn.weight_hh_l0   actor.rnn.bias_ih_l0 ...
    actor.mlp.0.weight       actor.mlp.0.bias         actor.mlp.2.weight   ...
    critic.0.weight ...   std

The API surface matches what ``AMPPPO`` / ``AMPOnPolicyRunner`` call on the
non-recurrent class, PLUS the recurrent extras the dormant branches expect:
``is_recurrent = True``, ``get_hidden_states()``, ``reset(dones)``, and
``act`` / ``evaluate`` accepting ``masks=`` / ``hidden_states=`` kwargs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from application.training.amp.algorithms.actor_critic import _get_activation


_RNN_CLASSES = {"gru": nn.GRU, "lstm": nn.LSTM}


def unpad_trajectories(trajectories: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_split_and_pad_trajectories`` (rollout_storage).

    Verbatim port of ``rsl_rl.utils.utils.unpad_trajectories`` from the
    AMP_for_hardware fork — the exact counterpart of the
    ``_split_and_pad_trajectories`` already inlined in
    ``storage/rollout_storage.py``.

    During the recurrent PPO update the mini-batch generator pads obs into
    ``[T_pad, n_traj, D]`` so the RNN can re-run whole trajectories from the
    saved start hidden state. The actor/critic forward therefore produces a
    ``[T_pad, n_traj, *]`` output, but ``actions_batch`` / ``returns_batch``
    stay in the raw ``[T, n_envs, *]`` layout. This collapses the padded
    output back to ``[T, n_envs, *]`` using the trajectory masks so the two
    align (otherwise ``Normal.log_prob`` sees ``[T_pad, n_traj]`` vs
    ``[T, n_envs]`` and aborts — the historical crash).
    """
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


class RecurrentActor(nn.Module):
    """Memory cell (GRU/LSTM) → MLP head, producing the action mean.

    ``forward`` is the deterministic inference path used at export/replay:
    ``(obs[N, obs_dim], hidden) -> (mean[N, num_actions], next_hidden)``.
    Hidden shapes: GRU ``h[L, N, H]``; LSTM ``(h, c)`` each ``[L, N, H]``.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims,
        activation: str,
        rnn_type: str,
        rnn_hidden_size: int,
        rnn_num_layers: int,
    ) -> None:
        super().__init__()
        rnn_type = (rnn_type or "gru").strip().lower()
        rnn_cls = _RNN_CLASSES.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(
                f"ActorCriticRecurrent: unsupported rnn_type {rnn_type!r}; "
                f"expected one of {sorted(_RNN_CLASSES)}"
            )
        self.rnn_type = rnn_type
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.rnn_num_layers = int(rnn_num_layers)
        self.rnn = rnn_cls(
            input_size=num_obs,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=False,
        )
        act_fn = _get_activation(activation)
        dims = list(hidden_dims)
        layers = [nn.Linear(self.rnn_hidden_size, dims[0]), act_fn]
        for i in range(len(dims)):
            if i == len(dims) - 1:
                layers.append(nn.Linear(dims[i], num_actions))
            else:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(act_fn)
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, hidden=None):
        # obs may be (N, obs_dim) for a single inference step, or
        # (T, N, obs_dim) for a batched training rollout. Normalise to a
        # time dim of 1 for the single-step case.
        single_step = obs.dim() == 2
        x = obs.unsqueeze(0) if single_step else obs
        out, new_hidden = self.rnn(x, hidden)
        if single_step:
            out = out.squeeze(0)
        mean = self.mlp(out)
        return mean, new_hidden


class ActorCriticRecurrent(nn.Module):
    """Recurrent actor + MLP critic. Drop-in for the vendored ``ActorCritic``."""

    is_recurrent = True

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
        std_clamp_max: float = 1.5,
        std_clamp_min: float = 1e-3,
        rnn_type: str = "gru",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "[UnitPort][AMP] ActorCriticRecurrent ignored unexpected kwargs: "
                + str(list(kwargs.keys())),
                flush=True,
            )
        super().__init__()
        self._std_clamp_max = float(std_clamp_max)
        self._std_clamp_min = float(std_clamp_min)

        self.actor = RecurrentActor(
            num_actor_obs, num_actions, actor_hidden_dims, activation,
            rnn_type, rnn_hidden_size, rnn_num_layers,
        )

        # Critic — plain MLP over (possibly privileged) critic obs (R5).
        act_fn = _get_activation(activation)
        cdims = list(critic_hidden_dims)
        critic_layers = [nn.Linear(num_critic_obs, cdims[0]), act_fn]
        for i in range(len(cdims)):
            if i == len(cdims) - 1:
                critic_layers.append(nn.Linear(cdims[i], 1))
            else:
                critic_layers.append(nn.Linear(cdims[i], cdims[i + 1]))
                critic_layers.append(act_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(
            f"[UnitPort][AMP] ActorCriticRecurrent: rnn={rnn_type}"
            f"(hidden={rnn_hidden_size}, layers={rnn_num_layers}) → "
            f"actor={list(actor_hidden_dims)} → {num_actions}, "
            f"critic={list(critic_hidden_dims)} → 1, activation={activation}",
            flush=True,
        )

        self.fixed_std = bool(fixed_std)
        std_init = init_noise_std * torch.ones(num_actions)
        self.std = (
            torch.tensor(std_init) if self.fixed_std else nn.Parameter(std_init)
        )
        self.distribution = None
        # Per-(actor) hidden state, lazily shaped to the batch on first act().
        self._hidden = None
        Normal.set_default_validate_args = False

    # ── hidden-state lifecycle (the dormant rsl_amp_ppo branches call these) ──

    def reset(self, dones=None) -> None:
        """Zero the hidden state for envs that just terminated.

        ``dones`` is a bool/0-1 tensor over the batch; entries that are True
        get their hidden column zeroed. ``None`` / no hidden yet → full reset.
        """
        if self._hidden is None or dones is None:
            self._hidden = None
            return
        mask = (~dones.bool()).to(self._hidden_ref().dtype) if hasattr(dones, "bool") else None
        if mask is None:
            self._hidden = None
            return
        # mask shape (N,) → (1, N, 1) broadcast over [L, N, H]
        m = mask.view(1, -1, 1)
        if isinstance(self._hidden, tuple):
            self._hidden = tuple(h * m for h in self._hidden)
        else:
            self._hidden = self._hidden * m

    def _hidden_ref(self):
        return self._hidden[0] if isinstance(self._hidden, tuple) else self._hidden

    def get_hidden_states(self):
        """Return the current actor hidden state (rsl_amp_ppo reads this into
        the rollout transition). Critic shares no recurrence (R5) → None."""
        return self._hidden, None

    # ── distribution helpers (mirror ActorCritic) ──

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if not torch.isfinite(mean).all():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = self.std.to(mean.device)
        std = torch.nan_to_num(
            std, nan=0.3, posinf=self._std_clamp_max, neginf=0.3
        ).clamp(min=self._std_clamp_min, max=self._std_clamp_max)
        self.distribution = Normal(mean, mean * 0.0 + std)

    # ── public API used by AMPPPO ──

    def act(self, observations: torch.Tensor, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            # Recurrent PPO UPDATE pass: obs arrived padded ([T_pad, n_traj, *])
            # so the RNN re-runs whole trajectories from the SAVED start hidden
            # (``hidden_states``). The resulting hidden is per-trajectory
            # (n_traj) batch state — it MUST NOT overwrite the persistent
            # rollout hidden (sized to num_envs), or the next single-step
            # rollout act fails with "Expected hidden (1, num_envs, H), got
            # (1, n_traj, H)". So discard the new hidden here. Then collapse the
            # padded mean back to the raw [T, n_envs, *] layout actions_batch
            # uses (Normal.log_prob can't broadcast otherwise).
            mean, _ = self.actor(observations, hidden_states)
            mean = unpad_trajectories(mean, masks)
        else:
            # Single-step ROLLOUT/inference: persist the recurrent state across
            # steps (the runner zeroes terminated envs via reset(dones)).
            if hidden_states is not None:
                self._hidden = hidden_states
            mean, self._hidden = self.actor(observations, self._hidden)
        self._update_distribution(mean)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mean, self._hidden = self.actor(observations, self._hidden)
        return mean

    def evaluate(self, critic_observations: torch.Tensor, masks=None,
                 hidden_states=None, **kwargs) -> torch.Tensor:
        # Critic is non-recurrent (R5): hidden_states ignored. But the recurrent
        # mini-batch generator still feeds PADDED critic obs ([T_pad, n_traj, *])
        # during the update, so the value output must be unpadded back to the
        # raw [T, n_envs, *] layout to align with returns_batch / values_batch
        # (the actor side does the same). masks is None on the single-step
        # rollout path → plain forward.
        value = self.critic(critic_observations)
        if masks is not None:
            value = unpad_trajectories(value, masks)
        return value


__all__ = ["ActorCriticRecurrent", "RecurrentActor"]
