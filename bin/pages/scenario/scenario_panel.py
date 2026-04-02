"""Scenario panel placeholder for runtime configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ScenarioPanelState:
    target: str = "simulation"
    robot_type: str = "go2"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_runtime_scenario(self, **overrides: Any) -> Dict[str, Any]:
        scenario = {
            "target": self.target,
            "robot_type": self.robot_type,
            **self.params,
        }
        scenario.update(overrides)
        return scenario
