"""Safety policy model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SafetyPolicy:
    max_loop_iterations: int = 100
    require_robot_for_actions: bool = False
    block_when_simulation_running: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyPolicy":
        if not isinstance(data, dict):
            return cls()
        return cls(
            max_loop_iterations=int(data.get("max_loop_iterations", 100)),
            require_robot_for_actions=bool(data.get("require_robot_for_actions", False)),
            block_when_simulation_running=bool(data.get("block_when_simulation_running", True)),
        )

