"""Compile-time guard checks."""

from __future__ import annotations

from typing import Any, Dict


class CompileGuard:
    """Lightweight structural checks before runtime execution."""

    def check(self, mission_ir: Any) -> Dict[str, Any]:
        if mission_ir is None:
            return {"ok": False, "reason": "mission_ir_missing"}
        if not isinstance(mission_ir, dict):
            return {"ok": True}

        nodes = mission_ir.get("nodes")
        if nodes is None:
            return {"ok": False, "reason": "nodes_missing"}
        if not isinstance(nodes, (list, dict)):
            return {"ok": False, "reason": "nodes_invalid_type"}
        return {"ok": True}

