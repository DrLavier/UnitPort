"""application.service.sensor_health — 6 项固定传感器的健康状态 model.

为 mission_panel "传感器状态" 卡片提供统一数据源。当前是 stub：所有传感器初始
``level="idle"``（灰），后端补齐时通过 ``set_status(name, level)`` 推数据，
信号 ``signals.sensor_status_changed(name, level)`` 由 model 转发。

固定 6 项：``imu`` / ``joints`` / ``lidar`` / ``camera`` / ``torque`` / ``battery``。
``level`` 取值 ``"ok"`` / ``"warn"`` / ``"error"`` / ``"idle"`` —— UI 把它们映射
到 system.ini 的 ``safe_zone`` / ``highlight`` / ``danger_zone`` / ``sub_t2``
颜色槽。
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import QObject

from application.service.signals import get_app_signals


SENSOR_NAMES = ("imu", "joints", "lidar", "camera", "torque", "battery")
LEVELS = ("ok", "warn", "error", "idle")


def _normalize_level(level: str) -> str:
    s = str(level or "").strip().lower()
    return s if s in LEVELS else "idle"


class SensorHealthModel(QObject):
    """单例 model；保存当前各传感器 level 并广播变更."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._levels: Dict[str, str] = {n: "idle" for n in SENSOR_NAMES}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def status_map(self) -> Dict[str, str]:
        """Snapshot of current per-sensor levels."""
        return dict(self._levels)

    def get(self, name: str) -> str:
        return self._levels.get(str(name), "idle")

    def set_status(self, name: str, level: str) -> None:
        """更新一项传感器状态并 emit 信号（即使 level 未变也强制 emit，
        方便 UI 在订阅之前已设置过的项首次刷新时拉到最新值）."""
        n = str(name).strip().lower()
        if n not in SENSOR_NAMES:
            return
        lv = _normalize_level(level)
        self._levels[n] = lv
        get_app_signals().sensor_status_changed.emit(n, lv)

    def reset_all_idle(self) -> None:
        for n in SENSOR_NAMES:
            self.set_status(n, "idle")


_instance: Optional[SensorHealthModel] = None


def get_sensor_health_model() -> SensorHealthModel:
    """Lazy singleton — must be called after QApplication exists."""
    global _instance
    if _instance is None:
        _instance = SensorHealthModel()
    return _instance


__all__ = [
    "SENSOR_NAMES",
    "LEVELS",
    "SensorHealthModel",
    "get_sensor_health_model",
]
