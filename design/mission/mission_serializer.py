"""Mission IR serialization helpers."""

from __future__ import annotations

from shared.ir.workflow_ir import WorkflowIR


class MissionSerializer:
    """Serialize/deserialize mission IR."""

    @staticmethod
    def to_json(ir: WorkflowIR, indent: int = 2) -> str:
        return ir.to_json(indent=indent)

    @staticmethod
    def from_json(payload: str) -> WorkflowIR:
        return WorkflowIR.from_json(payload)

