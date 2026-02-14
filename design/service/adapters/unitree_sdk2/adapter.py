"""Unitree SDK2 service adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

from design.service.adapters.base_adapter import BaseAdapter
from models.unitree import UnitreeModel
from .mapper import map_action


class UnitreeAdapter(BaseAdapter):
    """Adapter for Unitree robots via SDK2."""

    def __init__(self, robot_type: str = "go2"):
        self.robot_type = (robot_type or "go2").lower()
        self._model: Optional[UnitreeModel] = None

    def connect(self, **kwargs: Any) -> bool:
        """Create or refresh an underlying Unitree model instance."""
        requested_type = kwargs.get("robot_type")
        force_reinit = bool(kwargs.get("force_reinit", False))
        if requested_type:
            self.robot_type = str(requested_type).lower()

        if self._model is not None and not force_reinit:
            if getattr(self._model, "robot_type", "").lower() == self.robot_type:
                return True

        self._model = UnitreeModel(self.robot_type)
        return True

    def get_model(self) -> Optional[UnitreeModel]:
        """Compatibility helper for legacy callers that still need model instance."""
        if self._model is None:
            self.connect()
        return self._model

    def run_action(self, action: str, **params: Any) -> Any:
        model = self.get_model()
        if model is None:
            return False
        return model.run_action(map_action(action), **params)

    def stop(self) -> None:
        model = self.get_model()
        if model is not None:
            model.stop()

    def get_sensor_data(self) -> Dict[str, Any]:
        model = self.get_model()
        if model is None:
            return {"error": "Unitree model not available"}
        return model.get_sensor_data()

    def health(self) -> Dict[str, Any]:
        model = self.get_model()
        if model is None:
            return {"connected": False, "robot_type": self.robot_type}
        return {
            "connected": True,
            "robot_type": self.robot_type,
            "available": bool(getattr(model, "is_available", False)),
        }
