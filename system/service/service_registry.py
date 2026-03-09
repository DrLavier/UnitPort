"""Service adapter registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ServiceRegistry:
    """Registry for robot service adapters."""

    _adapters: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def register(self, name: str, adapter: Any) -> None:
        """Register a service adapter under *name*."""
        if not name:
            raise ValueError("Adapter name must be non-empty")
        if adapter is None:
            raise ValueError("Adapter instance must not be None")
        self._adapters[name] = adapter

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a registered adapter by *name*."""
        return self._adapters.get(name)

    def require(self, name: str) -> Any:
        """Retrieve adapter or raise KeyError if missing."""
        adapter = self.get(name)
        if adapter is None:
            raise KeyError(f"Adapter not found: {name}")
        return adapter

    def list_adapters(self) -> List[str]:
        """Return names of all registered adapters."""
        return sorted(self._adapters.keys())
