"""Service action routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .service_registry import ServiceRegistry


@dataclass
class ServiceRouter:
    """Routes actions and queries to the appropriate service adapter."""

    registry: ServiceRegistry = field(default_factory=ServiceRegistry)

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register an adapter to the underlying registry."""
        self.registry.register(name, adapter)

    def get_adapter(self, adapter_name: str) -> Any:
        """Return adapter instance or raise KeyError if missing."""
        return self.registry.require(adapter_name)

    def run_action(self, adapter_name: str, action: str, **params: Any) -> Any:
        """Route an action to the named adapter."""
        return self.get_adapter(adapter_name).run_action(action, **params)

    def stop(self, adapter_name: str) -> None:
        """Stop the named adapter."""
        self.get_adapter(adapter_name).stop()

    def get_sensor_data(self, adapter_name: str) -> Dict[str, Any]:
        """Retrieve sensor data from the named adapter."""
        return self.get_adapter(adapter_name).get_sensor_data()

    def health(self, adapter_name: str) -> Dict[str, Any]:
        """Retrieve health information from the named adapter."""
        return self.get_adapter(adapter_name).health()
