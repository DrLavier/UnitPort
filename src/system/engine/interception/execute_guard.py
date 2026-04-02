"""Execute-time guard checks."""

from __future__ import annotations

from typing import Any, Dict


class ExecuteGuard:
    """Checks runtime scenario preconditions before execution begins."""

    def check(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(scenario, dict):
            return {"ok": False, "reason": "scenario_invalid"}

        if scenario.get("simulation_running", False):
            return {"ok": False, "reason": "simulation_already_running"}

        target = scenario.get("target", "simulation")
        if target not in ("simulation", "hardware"):
            return {"ok": False, "reason": "scenario_target_invalid"}

        return {"ok": True}

