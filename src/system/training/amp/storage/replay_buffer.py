# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Vendored into UnitPort from
#   custom_mods/archives/AMP_for_hardware/rsl_rl/rsl_rl/storage/replay_buffer.py
# See the header in amp/algorithms/discriminator.py for the full BSD-3
# text (Copyright 2021 NVIDIA / ETH Zurich / Nikita Rudin).
#
# Changes from upstream:
#   - Added type hints + docstrings.
#   - ``feed_forward_generator`` uses ``np.random.default_rng()`` for
#     determinism-friendly tests (upstream used the legacy np.random).

"""Fixed-size ring buffer of AMP ``(s_t, s_{t+1})`` policy transitions.

The buffer lives on the AMPPPO side — every rollout step adds the
current policy's amp observation pair, and every update call samples
minibatches from it to train the discriminator against the expert
motion data.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Ring buffer storing ``(state, next_state)`` tensor pairs."""

    def __init__(
        self,
        obs_dim: int,
        buffer_size: int,
        device,
    ) -> None:
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = int(buffer_size)
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Add new state pairs. Wraps around when the buffer is full."""
        num_states = int(states.shape[0])
        start_idx = self.step
        end_idx = self.step + num_states

        if end_idx > self.buffer_size:
            # Split across the wrap
            head = self.buffer_size - self.step
            self.states[self.step:self.buffer_size] = states[:head]
            self.next_states[self.step:self.buffer_size] = next_states[:head]
            tail = end_idx - self.buffer_size
            self.states[:tail] = states[head:]
            self.next_states[:tail] = next_states[head:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(
            self.buffer_size, max(end_idx, self.num_samples)
        )
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``num_mini_batch`` random minibatches of stored pairs."""
        if self.num_samples == 0:
            raise RuntimeError(
                "ReplayBuffer.feed_forward_generator: buffer is empty"
            )
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(num_mini_batch)):
            idxs = rng.integers(0, self.num_samples, size=int(mini_batch_size))
            yield (
                self.states[idxs].to(self.device),
                self.next_states[idxs].to(self.device),
            )
