from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_EPSILON = 1e-8  # guards against near-zero std


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


class Normalizer:
    """Lightweight observation/action normalization.

    If stats are absent the operation is a no-op except dtype coercion to
    float32.  Clipping is applied only when clip_min / clip_max are set.
    Convention: normalize_obs and denormalize_action are both safe to call
    with None stats.
    """

    def __init__(
        self,
        obs_stats: Optional[NormalizationStats] = None,
        action_stats: Optional[NormalizationStats] = None,
    ):
        self._obs_stats = obs_stats
        self._action_stats = action_stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation vector to float32.

        z = (obs - mean) / max(std, epsilon)
        Then optionally clip to [clip_min, clip_max].
        """
        out = obs.astype(np.float32)
        if self._obs_stats is None:
            return out
        stats = self._obs_stats
        std_safe = np.where(
            np.abs(stats.std) < _EPSILON,
            _EPSILON,
            stats.std.astype(np.float32),
        )
        out = (out - stats.mean.astype(np.float32)) / std_safe
        out = self._clip(out, stats)
        return out

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action vector from normalized space to raw space.

        raw = action * max(std, epsilon) + mean
        Then optionally clip.
        """
        out = action.astype(np.float32)
        if self._action_stats is None:
            return out
        stats = self._action_stats
        std_safe = np.where(
            np.abs(stats.std) < _EPSILON,
            _EPSILON,
            stats.std.astype(np.float32),
        )
        out = out * std_safe + stats.mean.astype(np.float32)
        out = self._clip(out, stats)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _clip(arr: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        if stats.clip_min is not None or stats.clip_max is not None:
            arr = np.clip(arr, stats.clip_min, stats.clip_max)
        return arr
