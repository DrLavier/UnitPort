"""Running mean/std normalizer for AMP discriminator inputs.

DIRECT-MIGRATE from DEMO ``src/system/training/amp/utils/normalizer.py``,
which itself vendors :class:`RunningMeanStd` + :class:`Normalizer` from
``custom_mods/archives/AMP_for_hardware/rsl_rl`` (BSD-3, NVIDIA + ETH
Zurich; full license header in DEMO ``amp/algorithms/discriminator.py``).
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np


class RunningMeanStd:
    """Streaming estimator of per-dimension mean and variance.

    Uses the parallel update rule from
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance —
    numerically stable for long training runs.
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Union[int, Tuple[int, ...]] = (),
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.var: np.ndarray = np.ones(shape, dtype=np.float64)
        self.count: float = float(epsilon)

    def update(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class Normalizer(RunningMeanStd):
    """Clip-and-standardize wrapper for AMP discriminator inputs."""

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
    ) -> None:
        super().__init__(shape=input_dim)
        self.epsilon = float(epsilon)
        self.clip_obs = float(clip_obs)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def normalize_torch(self, x, device):
        import torch
        mean_t = torch.as_tensor(self.mean, device=device, dtype=torch.float32)
        std_t = torch.sqrt(
            torch.as_tensor(self.var + self.epsilon, device=device, dtype=torch.float32)
        )
        return torch.clamp(
            (x - mean_t) / std_t, -self.clip_obs, self.clip_obs
        )

    def state_dict(self) -> Dict[str, Union[np.ndarray, float]]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "epsilon": self.epsilon,
            "clip_obs": self.clip_obs,
        }

    def load_state_dict(self, state: Dict[str, Union[np.ndarray, float]]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
        self.epsilon = float(state["epsilon"])
        self.clip_obs = float(state["clip_obs"])


__all__ = ["RunningMeanStd", "Normalizer"]
