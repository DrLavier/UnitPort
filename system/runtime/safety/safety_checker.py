"""Safety checks for runtime execution."""

from __future__ import annotations

from typing import Any, Dict

from .safety_policy import SafetyPolicy


class SafetyChecker:
    """Evaluate mission/scenario against a safety policy."""

    def check(self, mission_ir: Any, scenario: Dict[str, Any], policy: SafetyPolicy) -> Dict[str, Any]:
        if policy.block_when_simulation_running and scenario.get("simulation_running", False):
            return {"ok": False, "reason": "simulation_running"}

        if policy.require_robot_for_actions and self._has_action_nodes(mission_ir):
            if scenario.get("robot_model") is None:
                return {"ok": False, "reason": "robot_model_required"}

        return {"ok": True}

    @staticmethod
    def _has_action_nodes(mission_ir: Any) -> bool:
        if not isinstance(mission_ir, dict):
            return False
        nodes = mission_ir.get("nodes", {})
        if isinstance(nodes, dict):
            return any(
                n.get("type") in ("action_execution", "stop")
                or "Action Execution" in n.get("name", "")
                for n in nodes.values()
            )
        if isinstance(nodes, list):
            return any(n.get("type") in ("action_execution", "stop") for n in nodes)
        return False

