"""Bridge behavior layer to compiler outputs."""

from __future__ import annotations

from typing import List, Tuple

from compiler.semantic.diagnostics import Diagnostic
from design.mission.mission_planner import MissionPlanner
from shared.ir.workflow_ir import WorkflowIR


class BehaviorCompilerBridge:
    """Unify compiler/canvas behavior inputs to shared IR."""

    def __init__(self):
        self._planner = MissionPlanner()

    def from_source(self, source: str, robot_type: str = "go2") -> Tuple[WorkflowIR, List[Diagnostic]]:
        return self._planner.from_source(source, robot_type=robot_type)

