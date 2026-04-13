"""Bridge: phase_2 MotionClip → AmpOnPolicyRunner ``amp_data`` contract.

AMPOnPolicyRunner expects its ``amp_data`` object to expose:

- ``.observation_dim`` (int) — size of the concatenated ``(s_t, s_{t+1})``
  input to the discriminator divided by 2.
- ``.feed_forward_generator(num_mini_batch, mini_batch_size)`` yielding
  tuples ``(expert_state, expert_next_state)`` of torch tensors on the
  runner's device.
- ``.preload_transitions(num_preload_transitions)`` — optional. The
  vendored ``amp_on_policy_runner`` calls this via the
  ``preload_transitions=True`` runner flag. We provide it as a no-op
  when transitions were already materialized at ``build_amp_data`` time
  (which is the common case), and as a real preload pass otherwise.

The bridge keeps motion data on CPU as float32 numpy until the
generator is called, then moves minibatches to the target device. This
is both memory-friendly for long mocap datasets and test-friendly
(tests can run on CPU without a CUDA toolchain).

Per AMP_design.yaml §4.amp_backend.data_provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.system.training.motion.clip import MotionClip
from src.system.training.motion.loader import AMPLegGymLoader


# ---------------------------------------------------------------------------
# MotionAmpData — the object AmpOnPolicyRunner consumes
# ---------------------------------------------------------------------------


@dataclass
class MotionAmpData:
    """AMP data source for the vendored ``AmpOnPolicyRunner``.

    Constructed from one or more ``MotionClip`` objects (typically by
    ``build_amp_data`` or ``build_amp_data_from_files``). The important
    invariants:

    - All clips must share the same ``amp_obs_dim`` (we reject the
      build at mixing time).
    - At least one clip must have ``has_amp_payload() == True``
      (tracking-only clips cannot produce discriminator transitions).
    """

    clips: List[MotionClip]
    device: torch.device
    transition_dt: float
    #: Clip-level sampling probabilities, derived from MotionClip.motion_weight.
    clip_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Cache of preloaded transitions; populated lazily when the
    #: vendored runner calls ``preload_transitions``.
    _preloaded: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    #: Dimension reported to the discriminator constructor.
    _obs_dim: int = 0

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("MotionAmpData requires at least one MotionClip")
        dims = {c.amp_obs_dim for c in self.clips}
        if len(dims) != 1:
            raise ValueError(
                f"MotionAmpData: clips have inconsistent amp_obs_dim: {dims}. "
                f"All clips must share the same AMP observation layout."
            )
        obs_dim = next(iter(dims))
        if obs_dim <= 0:
            raise ValueError(
                "MotionAmpData: clips do not carry an AMP payload "
                "(amp_obs_dim == 0). Use format_id='amp_legged_gym'."
            )
        self._obs_dim = int(obs_dim)

        weights = np.array(
            [max(1e-6, float(c.motion_weight)) for c in self.clips],
            dtype=np.float64,
        )
        weights /= weights.sum()
        self.clip_weights = weights

        if self.transition_dt <= 0:
            raise ValueError(
                f"MotionAmpData.transition_dt must be > 0, got {self.transition_dt}"
            )

    # ------------------------------------------------------------------
    # AmpOnPolicyRunner contract
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        """Size of one ``s_t`` observation (NOT the concatenated pair)."""
        return self._obs_dim

    @property
    def num_motions(self) -> int:
        return len(self.clips)

    def preload_transitions(
        self,
        num_preload_transitions: int,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Materialize a large batch of transitions up-front.

        Mirrors ``AMPLoader.preload_transitions`` in the vendored
        ``motion_loader.py``. Useful for runs where the clip set is
        small enough to fit in memory and you want per-minibatch
        sampling to be a pure index lookup.
        """
        n = int(num_preload_transitions)
        if n <= 0:
            self._preloaded = None
            return
        s_np, s_next_np = self._sample_np(n, rng=rng)
        self._preloaded = (
            torch.as_tensor(s_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(s_next_np, dtype=torch.float32, device=self.device),
        )

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``num_mini_batch`` minibatches of expert transitions.

        If ``preload_transitions`` was called earlier, samples come from
        the cached tensor (much faster, deterministic with a seeded rng).
        Otherwise each minibatch re-samples from the raw clips.
        """
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(int(num_mini_batch)):
            if self._preloaded is not None:
                s_cache, s_next_cache = self._preloaded
                total = s_cache.shape[0]
                idxs = rng.integers(0, total, size=int(mini_batch_size))
                idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
                yield s_cache.index_select(0, idxs_t), s_next_cache.index_select(0, idxs_t)
            else:
                s_np, s_next_np = self._sample_np(int(mini_batch_size), rng=rng)
                yield (
                    torch.as_tensor(s_np, dtype=torch.float32, device=self.device),
                    torch.as_tensor(s_next_np, dtype=torch.float32, device=self.device),
                )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_np(
        self, n: int, *, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` transitions across clips weighted by motion_weight.

        Each transition is sourced from a single clip (sampled with
        replacement proportionally to ``clip_weights``) and uses
        ``MotionClip.sample_transitions`` to draw ``(s_t, s_{t+dt})``.
        """
        if rng is None:
            rng = np.random.default_rng()

        clip_idxs = rng.choice(
            len(self.clips), size=n, replace=True, p=self.clip_weights
        )
        # Bucket by clip so each clip gets one vectorised sample_transitions call.
        s_out = np.empty((n, self._obs_dim), dtype=np.float32)
        s_next_out = np.empty((n, self._obs_dim), dtype=np.float32)

        for ci, clip in enumerate(self.clips):
            mask = clip_idxs == ci
            k = int(mask.sum())
            if k == 0:
                continue
            s_t, s_tp1 = clip.sample_transitions(k, self.transition_dt, rng=rng)
            s_out[mask] = s_t
            s_next_out[mask] = s_tp1
        return s_out, s_next_out


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def build_amp_data(
    clips: Sequence[MotionClip],
    *,
    transition_dt: float,
    device: Union[str, torch.device] = "cpu",
) -> MotionAmpData:
    """Wrap an existing list of ``MotionClip`` objects into a ``MotionAmpData``.

    Use this when the caller has already loaded and validated clips (as
    in unit tests). The launcher typically uses
    :func:`build_amp_data_from_files` instead.
    """
    dev = torch.device(device)
    return MotionAmpData(
        clips=list(clips),
        device=dev,
        transition_dt=float(transition_dt),
    )


def build_amp_data_from_files(
    paths: Sequence[Union[Path, str]],
    *,
    transition_dt: float,
    device: Union[str, torch.device] = "cpu",
    format_id: str = "amp_legged_gym",
) -> MotionAmpData:
    """Load + validate + wrap motion files in one call.

    Used by the launcher's AMP branch. Only ``amp_legged_gym`` is
    supported for AMP consumption (``unitport_npy`` is tracking-only
    by definition).
    """
    if format_id != "amp_legged_gym":
        raise ValueError(
            f"build_amp_data_from_files: only format_id='amp_legged_gym' can "
            f"produce an AMP-capable dataset, got {format_id!r}."
        )
    if not paths:
        raise ValueError("build_amp_data_from_files: at least one file is required")

    from src.system.training.motion.validator import validate_clip

    loader = AMPLegGymLoader()
    clips: List[MotionClip] = []
    for p in paths:
        clip = loader.load(p)
        validate_clip(clip)
        clips.append(clip)

    return build_amp_data(clips, transition_dt=transition_dt, device=device)
