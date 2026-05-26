# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AMP replay buffers — two parallel classes for the two AMP chains.

The two AMP code paths in RELEASE need **different** replay buffer
implementations and live in this single module because the import path
``application.training.amp.storage.replay_buffer`` is fixed by both
upstream forks (AMP_for_hardware for IL, our own SB3 callback for SB3):

* :class:`AmpReplayBuffer` — **SB3 chain** (``amp_ppo.py`` callback,
  runs in the main UnitPort venv). Pure numpy ring buffer. Caller hands
  in numpy arrays; torch is materialised inside the trainer. Signature:
  ``__init__(capacity, amp_obs_dim)`` + ``push / sample / save / load``.

* :class:`ReplayBuffer` — **IL chain** (``rsl_amp_ppo.py``, runs only
  inside the Isaac Sim subprocess). Torch tensor ring buffer carrying a
  third aligned ``motion_tag_ids`` track so the disc update can sample
  expert transitions per-policy-row from the matching task sub-buffer.
  Signature mirrors AMP_for_hardware's ``ReplayBuffer`` 1.x
  (``__init__(obs_dim, buffer_size, device)`` + ``insert /
  feed_forward_generator``) — RELEASE's only extension is that ``insert``
  takes an optional ``motion_tag_ids`` and ``feed_forward_generator``
  yields a 3-tuple including the aligned tag slice. ``None`` /
  unspecified tags are stored as :data:`UNKNOWN_TAG_ID` (``-1``) so the
  IL disc gracefully falls back to global expert sampling when the
  runner hasn't threaded tags through.

Neither class depends on ``unitport_sdk`` / PyQt6 — both must remain
importable inside Isaac Sim's vanilla Python venv (the lazy-import
contract enforced by :mod:`application.training.amp.algorithms.__init__`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


class AmpReplayBuffer:
    """Ring buffer of policy AMP transitions.

    Layout: two parallel ``(capacity, amp_obs_dim)`` float32 arrays.
    """

    def __init__(self, capacity: int, amp_obs_dim: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if amp_obs_dim <= 0:
            raise ValueError(f"amp_obs_dim must be positive, got {amp_obs_dim}")
        self.capacity = int(capacity)
        self.amp_obs_dim = int(amp_obs_dim)
        self._states = np.zeros((self.capacity, self.amp_obs_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.amp_obs_dim), dtype=np.float32)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def push(self, states: np.ndarray, next_states: np.ndarray) -> None:
        """Insert one or more (s, s') pairs.

        Shapes: ``states.shape == next_states.shape == (B, amp_obs_dim)``.
        Wraps around when the buffer fills up.
        """
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.amp_obs_dim:
            raise ValueError(
                f"states.shape={states.shape}; expected (B, {self.amp_obs_dim})"
            )
        if states.shape != next_states.shape:
            raise ValueError(
                f"states {states.shape} and next_states {next_states.shape} "
                f"shapes differ"
            )
        n = states.shape[0]
        if n == 0:
            return
        # Handle wrap-around in two slices
        end = self._cursor + n
        if end <= self.capacity:
            self._states[self._cursor:end] = states
            self._next_states[self._cursor:end] = next_states
        else:
            split = self.capacity - self._cursor
            self._states[self._cursor:] = states[:split]
            self._next_states[self._cursor:] = next_states[:split]
            tail = n - split
            self._states[:tail] = states[split:]
            self._next_states[:tail] = next_states[split:]
        self._cursor = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a uniform random minibatch. Returns ``(states, next_states)``."""
        if self._size == 0:
            raise RuntimeError("AmpReplayBuffer is empty; cannot sample")
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self._size, size=int(batch_size))
        return self._states[idx].copy(), self._next_states[idx].copy()

    def clear(self) -> None:
        self._cursor = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Persistence — atomic .npz dump of the live ring slice
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Dump the live (filled) portion of the ring to a .npz file.

        Atomic via temp-file + ``os.replace``. Only the first ``self._size``
        rows are persisted — replay is resumed from a fresh cursor on load.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # ``np.savez_compressed`` auto-appends ``.npz`` if the filename
        # doesn't end with it, so name the temp ``...tmp.npz`` directly
        # to keep the on-disk filename predictable for the rename.
        tmp = target.parent / (target.name + ".tmp.npz")
        np.savez_compressed(
            str(tmp),
            states=self._states[: self._size],
            next_states=self._next_states[: self._size],
            capacity=np.asarray(self.capacity, dtype=np.int64),
            amp_obs_dim=np.asarray(self.amp_obs_dim, dtype=np.int64),
            cursor=np.asarray(self._cursor, dtype=np.int64),
            size=np.asarray(self._size, dtype=np.int64),
        )
        os.replace(str(tmp), str(target))

    @classmethod
    def load(cls, path) -> "AmpReplayBuffer":
        """Reconstruct an :class:`AmpReplayBuffer` from a saved .npz file."""
        with np.load(str(path)) as bundle:
            capacity = int(bundle["capacity"])
            amp_obs_dim = int(bundle["amp_obs_dim"])
            states = np.asarray(bundle["states"], dtype=np.float32)
            next_states = np.asarray(bundle["next_states"], dtype=np.float32)
            saved_size = int(bundle["size"])
        inst = cls(capacity=capacity, amp_obs_dim=amp_obs_dim)
        n = min(saved_size, capacity)
        if n > 0:
            inst._states[:n] = states[:n]
            inst._next_states[:n] = next_states[:n]
            inst._size = n
            inst._cursor = n % capacity
        return inst


# ---------------------------------------------------------------------------
# IL chain — torch-tensor ring buffer with aligned motion_tag_ids
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Torch-tensor ring buffer used by :class:`AMPPPO` (IL chain).

    Layout: three parallel ``(buffer_size, …)`` tensors held on
    ``device`` — ``states``, ``next_states``, ``motion_tag_ids``. The
    tag track lets :meth:`AMPPPO.update` sample expert transitions
    aligned to each policy row's task (see
    :meth:`MotionAmpData.sample_tagged`); when a caller omits tags they
    are stored as :data:`UNKNOWN_TAG_ID` (``-1``) and the disc falls
    back to global uniform expert sampling.

    Mirrors AMP_for_hardware's ``rsl_rl.storage.ReplayBuffer`` (1.x) so
    the vendored :class:`AMPPPO` works without modification; the only
    extension is the third tag track, which AMP_for_hardware didn't have
    and which is required by RELEASE's command-conditioned expert
    sampling.
    """

    #: Sentinel for "no tag known" — matches
    #: ``application.training.amp.tag_router.UNKNOWN_TAG_ID``. Inlined
    #: as a literal here rather than imported so this module stays
    #: dependency-free and importable inside Isaac Sim's venv even if
    #: ``tag_router`` later grows non-portable imports.
    UNKNOWN_TAG_ID: int = -1

    def __init__(self, obs_dim: int, buffer_size: int, device) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        # Local import — torch isn't a hard dep of this module's top
        # level so the SB3 numpy class above can still be used in
        # contexts that haven't paid torch's import cost yet.
        import torch
        self._torch = torch
        self.obs_dim = int(obs_dim)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.states = torch.zeros(self.buffer_size, self.obs_dim, device=device)
        self.next_states = torch.zeros(
            self.buffer_size, self.obs_dim, device=device,
        )
        self.motion_tag_ids = torch.full(
            (self.buffer_size,),
            fill_value=self.UNKNOWN_TAG_ID,
            dtype=torch.int64,
            device=device,
        )
        self.step = 0
        self.num_samples = 0

    def insert(
        self,
        states,
        next_states,
        motion_tag_ids=None,
    ) -> None:
        """Append a batch of transitions to the ring.

        ``states`` / ``next_states``: ``(B, obs_dim)`` torch tensors.
        ``motion_tag_ids``: optional ``(B,)`` int64 tensor of task tags.
        When ``None`` (or any element ``< 0``) the entry is recorded as
        :attr:`UNKNOWN_TAG_ID`, signalling the disc updater to sample
        the expert side from the global pool.
        """
        torch = self._torch
        num_states = int(states.shape[0])
        if num_states == 0:
            return
        if motion_tag_ids is None:
            motion_tag_ids = torch.full(
                (num_states,),
                fill_value=self.UNKNOWN_TAG_ID,
                dtype=torch.int64,
                device=self.device,
            )
        else:
            if motion_tag_ids.device != self.motion_tag_ids.device:
                motion_tag_ids = motion_tag_ids.to(self.motion_tag_ids.device)
            if motion_tag_ids.dtype != torch.int64:
                motion_tag_ids = motion_tag_ids.to(torch.int64)

        start = self.step
        end = self.step + num_states
        if end > self.buffer_size:
            head = self.buffer_size - start
            self.states[start:self.buffer_size] = states[:head]
            self.next_states[start:self.buffer_size] = next_states[:head]
            self.motion_tag_ids[start:self.buffer_size] = motion_tag_ids[:head]
            tail = end - self.buffer_size
            self.states[:tail] = states[head:]
            self.next_states[:tail] = next_states[head:]
            self.motion_tag_ids[:tail] = motion_tag_ids[head:]
        else:
            self.states[start:end] = states
            self.next_states[start:end] = next_states
            self.motion_tag_ids[start:end] = motion_tag_ids

        self.num_samples = min(
            self.buffer_size, max(end, self.num_samples),
        )
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
    ) -> Iterator[Tuple]:
        """Yield ``num_mini_batch`` random minibatches of size
        ``mini_batch_size``. Each yielded item is the 3-tuple
        ``(states, next_states, motion_tag_ids)``, every tensor sliced
        on the buffer's ``device``.

        Uses ``numpy.random.choice`` (matches AMP_for_hardware) — the
        IL trainer doesn't seed this directly; reproducibility comes
        from the global torch / numpy seeds set by the launcher.
        """
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(
                self.num_samples, size=int(mini_batch_size),
            )
            yield (
                self.states[sample_idxs].to(self.device),
                self.next_states[sample_idxs].to(self.device),
                self.motion_tag_ids[sample_idxs].to(self.device),
            )


__all__ = ["AmpReplayBuffer", "ReplayBuffer"]
