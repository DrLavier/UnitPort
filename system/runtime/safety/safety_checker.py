"""Safety checks for runtime execution."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

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

    def check_policy_step(
        self,
        action: np.ndarray,
        step_index: int,
        policy_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Per-step safety gate called during policy episode execution.

        Checks whether the action produced at *step_index* is safe to apply.
        Returns ``{"ok": True}`` when the step may proceed, or
        ``{"ok": False, "reason": <str>}`` to trigger a controlled safety stop.

        Extend this method to add numeric bounds, velocity limits, or
        workspace constraint checks without modifying the policy stack.

        Parameters
        ----------
        action:
            The action vector produced by the inference engine this step.
        step_index:
            Zero-based count of policy steps completed so far in the episode.
        policy_id:
            Identifier of the running policy (for logging / audit trail).
        context:
            Optional extra key/value pairs the caller can forward (e.g. env
            state, per-policy safety limits).  Currently unused; reserved for
            future constraint checks.

        Returns
        -------
        dict with keys:
          ``ok``     — bool; True = proceed, False = stop episode
          ``reason`` — str; human-readable explanation when ok=False, else ""
        """
        ctx = context or {}

        # Guard: action must be a finite numpy array
        if not isinstance(action, np.ndarray):
            return {"ok": False, "reason": "safety_stop: action is not a numpy array"}

        if not np.all(np.isfinite(action)):
            return {
                "ok": False,
                "reason": (
                    f"safety_stop: non-finite action detected at step {step_index} "
                    f"(policy_id={policy_id!r})"
                ),
            }

        # Per-policy max action magnitude (optional; caller injects via context)
        max_magnitude = ctx.get("max_action_magnitude")
        if max_magnitude is not None:
            magnitude = float(np.max(np.abs(action)))
            if magnitude > float(max_magnitude):
                return {
                    "ok": False,
                    "reason": (
                        f"safety_stop: action magnitude {magnitude:.4f} exceeds "
                        f"limit {max_magnitude} at step {step_index} "
                        f"(policy_id={policy_id!r})"
                    ),
                }

        return {"ok": True, "reason": ""}

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

