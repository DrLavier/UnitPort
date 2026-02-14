"""Emergency stop/degrade/rollback handler."""

from __future__ import annotations

from typing import Any, Dict


class EmergencyHandler:
    """Generate emergency actions when safety checks fail."""

    def handle(self, reason: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        context = context or {}
        action = "stop"
        if reason in ("simulation_running", "simulation_already_running"):
            action = "abort"
        return {
            "action": action,
            "reason": reason,
            "context": context,
        }

