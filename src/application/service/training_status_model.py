"""application.service.training_status_model — 训练状态聚合.

订阅 ``signals.training_run_started / training_metrics / training_run_finished``
三联信号，在主线程聚合出一个 status dict 供 mission_panel "训练状态" 卡片消费：

    {
        "state":          "idle" | "running" | "paused" | "finished" | "failed",
        "run_id":         str,
        "label":          str,             # 训练 run 的人类可读名
        "elapsed_s":      int,             # 已训练秒数
        "iterations":     int,             # 当前迭代步
        "current_reward": float,
        "mean_reward":    float,           # 滚动均值（最近 100 sample）
        "episode_length": float,           # 平均回合时长 (steps)
        "fps":            float,           # 来自 metrics["fps"]，未上报则保持 0
    }

未运行（idle）和初始化时 status dict 用 "全 0 + state=idle" 占位，让 UI 渲染
"未运行 / 0 / 0.00..." 兜底状态。

模型本身是**单例 QObject**：通过 ``get_training_status_model()`` 获取；
UI 订阅 ``signals.training_status_changed(dict)``（在 ``signals.py`` 上）。
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from PyQt6.QtCore import QObject

from application.service.signals import get_app_signals


_REWARD_WINDOW = 100                 # 计算 mean_reward 的滚动窗口


def _empty_status() -> Dict[str, Any]:
    return {
        "state": "idle",
        "run_id": "",
        "label": "",
        "elapsed_s": 0,
        "iterations": 0,
        "current_reward": 0.0,
        "mean_reward": 0.0,
        "best_reward": 0.0,
        "episode_length": 0.0,
        "fps": 0.0,
    }


class TrainingStatusModel(QObject):
    """聚合 training_metrics 流为 status dict.

    线程契约：所有 emit 都在主线程发生（信号源是 Qt queued connection），
    所以内部状态字段不加锁。
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._status: Dict[str, Any] = _empty_status()
        self._reward_window: Deque[float] = deque(maxlen=_REWARD_WINDOW)
        self._run_started_at: float = 0.0
        self._best_reward: Optional[float] = None  # None = 还未收到任何 reward 样本

        sigs = get_app_signals()
        sigs.training_run_started.connect(self._on_run_started)
        sigs.training_metrics.connect(self._on_metrics)
        sigs.training_run_finished.connect(self._on_run_finished)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Snapshot of the current status dict (defensive copy)."""
        return dict(self._status)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_run_started(self, run_id: str, label: str) -> None:
        self._reward_window.clear()
        self._run_started_at = time.monotonic()
        self._best_reward = None
        self._status = _empty_status()
        self._status["state"] = "running"
        self._status["run_id"] = str(run_id or "")
        self._status["label"] = str(label or "")
        self._broadcast()

    def _on_metrics(self, run_id: str, sample: Dict[str, Any]) -> None:
        # Drop stale samples for a different run_id (e.g. previous run still
        # finishing while a new one started).
        if self._status["state"] == "running" and self._status["run_id"]:
            if str(run_id or "") != self._status["run_id"]:
                return
        if not isinstance(sample, dict):
            return

        # Iterations / step
        step = sample.get("step")
        if step is None:
            step = sample.get("iteration")
        if isinstance(step, (int, float)):
            self._status["iterations"] = int(step)

        # Current reward — canonical key is ``reward_mean`` (used by both the
        # SB3 launcher's MSG_PROGRESS bridge and the Isaac Lab MetricsAccumulator).
        # ``reward`` / ``ep_rew_mean`` kept as fallbacks for legacy callers.
        reward = sample.get("reward_mean")
        if reward is None:
            reward = sample.get("reward")
        if reward is None:
            reward = sample.get("ep_rew_mean")
        if isinstance(reward, (int, float)) and not math.isnan(float(reward)):
            r = float(reward)
            self._status["current_reward"] = r
            self._reward_window.append(r)
            if self._reward_window:
                self._status["mean_reward"] = (
                    sum(self._reward_window) / len(self._reward_window)
                )
            # Best reward — monotonic max across the entire run.
            if self._best_reward is None or r > self._best_reward:
                self._best_reward = r
        # The task wrapper also injects an authoritative ``best_reward``
        # (rolling max tracked across MSG_PROGRESS + MSG_METRICS + MSG_FINISHED);
        # prefer that when present so the badge agrees with the bundle export.
        sample_best = sample.get("best_reward")
        if isinstance(sample_best, (int, float)) and not math.isnan(float(sample_best)):
            sb = float(sample_best)
            if self._best_reward is None or sb > self._best_reward:
                self._best_reward = sb
        if self._best_reward is not None:
            self._status["best_reward"] = self._best_reward

        # Mean episode length (steps). Both SB3 (sb3_entry.py) and Isaac Lab
        # (isaac_lab/backend.py _KEY_MAP) emit this as ``ep_len_mean`` — already
        # a rolling mean from the trainer, so we copy it through verbatim.
        ep_len = sample.get("ep_len_mean")
        if ep_len is None:
            ep_len = sample.get("episode_length")
        if isinstance(ep_len, (int, float)) and not math.isnan(float(ep_len)):
            self._status["episode_length"] = float(ep_len)

        # FPS
        fps = sample.get("fps")
        if isinstance(fps, (int, float)):
            self._status["fps"] = float(fps)

        # Elapsed
        if self._run_started_at > 0:
            self._status["elapsed_s"] = int(
                max(0.0, time.monotonic() - self._run_started_at)
            )

        self._broadcast()

    def _on_run_finished(self, run_id: str, success: bool) -> None:
        if self._status["run_id"] and str(run_id or "") != self._status["run_id"]:
            return
        self._status["state"] = "finished" if success else "failed"
        if self._run_started_at > 0:
            self._status["elapsed_s"] = int(
                max(0.0, time.monotonic() - self._run_started_at)
            )
        self._broadcast()

    def _broadcast(self) -> None:
        get_app_signals().training_status_changed.emit(dict(self._status))


_instance: Optional[TrainingStatusModel] = None


def get_training_status_model() -> TrainingStatusModel:
    """Lazy singleton — must be called after QApplication exists."""
    global _instance
    if _instance is None:
        _instance = TrainingStatusModel()
    return _instance


def initial_status() -> Dict[str, Any]:
    """Return an idle/zero status dict — useful for first paint before any
    real metrics arrive."""
    return _empty_status()


__all__ = [
    "TrainingStatusModel",
    "get_training_status_model",
    "initial_status",
]
