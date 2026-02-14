"""Abstract base adapter for robot services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAdapter(ABC):
    """Interface that every robot service adapter must implement."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> bool:
        """Establish a connection to the robot or simulator."""
        ...

    @abstractmethod
    def run_action(self, action: str, **params: Any) -> Any:
        """Execute a named action with the given parameters."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop all robot activity."""
        ...

    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]:
        """Return the latest sensor readings."""
        ...

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return adapter health / connectivity status."""
        ...
