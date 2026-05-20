"""SB3 callback that pipes the env's per-term / per-item reward breakdown
into the SB3 logger so it lands in TensorBoard.

``GenericMujocoEnv.step()`` writes three families of keys into the
``info`` dict:

    info["reward/<term>"]       — weighted contribution of one reward term
    info["reward/item/<id>"]    — blended total attributed to one motion item
    info["reward/active_item"]  — string id of the argmax-alpha item

This callback aggregates them across the vec-env rollout and emits the
mean every ``log_every`` SB3 steps. Without it, the rich info dict is
silently consumed by SB3's Monitor wrapper (which only retains ``r`` /
``l`` per episode) and the per-term signal never reaches the user.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
    _HAS_SB3 = True
except ImportError:  # pragma: no cover
    _BaseCallback = object  # type: ignore[assignment,misc]
    _HAS_SB3 = False


class RewardBreakdownCallback(_BaseCallback):
    """Aggregate per-term / per-item reward info and record to SB3 logger.

    Parameters
    ----------
    log_every:
        Emit a logger.record() snapshot every N SB3 callback ticks
        (where one tick = one VecEnv step across all sub-envs). Defaults
        to 100 so the TensorBoard scalar curves stay readable; users
        wanting finer grain can pass a smaller value at callsite.
    """

    def __init__(self, log_every: int = 100, verbose: int = 0) -> None:
        if not _HAS_SB3:
            raise ImportError(
                "stable_baselines3 is required for RewardBreakdownCallback."
            )
        super().__init__(verbose)
        self._log_every: int = max(1, int(log_every))
        self._buf: Dict[str, List[float]] = {}
        self._active_item_counts: Dict[str, int] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key, value in info.items():
                if not isinstance(key, str):
                    continue
                if key == "reward/active_item":
                    if isinstance(value, str) and value:
                        self._active_item_counts[value] = (
                            self._active_item_counts.get(value, 0) + 1
                        )
                    continue
                if not key.startswith("reward/"):
                    continue
                try:
                    self._buf.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue
        if self.n_calls % self._log_every == 0:
            self._flush()
        return True

    def _on_training_end(self) -> None:  # pragma: no cover
        self._flush()

    def _flush(self) -> None:
        if not (self._buf or self._active_item_counts):
            return
        for key, vals in self._buf.items():
            if not vals:
                continue
            self.logger.record(f"rollout/{key}", float(np.mean(vals)))
        if self._active_item_counts:
            total = float(sum(self._active_item_counts.values()))
            if total > 0:
                for iid, count in self._active_item_counts.items():
                    self.logger.record(
                        f"rollout/reward/active_item_frac/{iid}",
                        float(count) / total,
                    )
        self._buf.clear()
        self._active_item_counts.clear()


__all__ = ["RewardBreakdownCallback"]
