"""Scenario runtime console placeholder."""

from dataclasses import dataclass
from typing import Any, Dict

from design.runtime import RuntimeEngine


@dataclass
class RuntimeConsoleState:
    running: bool = False
    last_message: str = ""


def execute_with_runtime(runtime: RuntimeEngine, mission: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Execute mission through runtime engine and return output payload."""
    return runtime.execute(mission, scenario)
