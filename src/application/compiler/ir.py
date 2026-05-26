# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.compiler.ir — 分层 IR 契约 + 桥/序列化器（单文件）.

合并自原 / Merged from former:
- application/tools/ir/layered_contracts.py    （分层 IR 数据类与版本元数据）
- application/tools/ir/layered_interfaces.py   （桥与序列化器接口/默认实现）

来源 / Source: DEMO/src/system/ir/

⚠ workbench/ 不在本文件 —— IR 工作台输出归 ``src/runtime/workbench/``
（``Paths.RUNTIME_DIR``），不是源码。

后续 / Pending（阶段 C 搬 compiler 时）:
- WorkflowIR / IRNode / IREdge / NodeKind / IRParam / IRVariable / SourceSpan
  等核心类型从 DEMO/src/system/compiler/ir/workflow_ir.py 一并搬迁，届时与
  本文件分层契约共存。
"""

from __future__ import annotations

import datetime
from dataclasses import asdict as _asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# =============================================================================
# Section 1 — Layered IR contracts（合并自 layered_contracts.py）
# =============================================================================

@dataclass
class MissionIR:
    """Mission-level orchestration contract."""

    entry_node_id: str = ""
    exit_policy: str = "complete_all"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorStateTransition:
    """State transition contract for behavior state machines."""

    from_state: str
    to_state: str
    guard: str = ""
    action_ref: str = ""


@dataclass
class MotionParamSpec:
    """Fine-control parameter metadata for behavior tuning."""

    name: str
    value: Any = None
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    tunable_realtime: bool = True


@dataclass
class SensorBinding:
    """Sensor/event input binding used by behavior logic."""

    channel: str
    event_type: str = ""
    sample_rate_hz: float = 0.0
    filter_ref: str = ""


@dataclass
class BehaviorIR:
    """Behavior-level contract for fine-grained control."""

    initial_state: str = "idle"
    terminal_states: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    transitions: List[BehaviorStateTransition] = field(default_factory=list)
    motion_params: List[MotionParamSpec] = field(default_factory=list)
    sensor_bindings: List[SensorBinding] = field(default_factory=list)
    custom_logic_refs: List[str] = field(default_factory=list)


@dataclass
class CommandIR:
    """Normalized command model contract for service adapters."""

    command_id: str
    command_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timeout_sec: float = 0.0
    retry_max_attempts: int = 0
    preemption_policy: str = "disabled"


@dataclass
class SafetyIR:
    """Safety policy contract."""

    limits: Dict[str, Any] = field(default_factory=dict)
    guards: Dict[str, Any] = field(default_factory=dict)
    estop: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityIR:
    """Telemetry/audit contract."""

    event_channels: List[str] = field(default_factory=list)
    metrics_interval_hz: float = 0.0
    audit_enabled: bool = True


@dataclass
class ExecutionIR:
    """Execution policy contract."""

    scheduler_mode: str = "event_driven"
    cancel_grace_period_sec: float = 0.0
    failure_strategy: str = "fail_fast"
    concurrency_policy: str = "single_thread"


@dataclass
class SubgraphPort:
    """Public subgraph IO port contract."""

    name: str
    direction: str
    data_type: str = "any"
    required: bool = False
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "data_type": self.data_type,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubgraphPort":
        return cls(
            name=str(d.get("name", "")),
            direction=str(d.get("direction", "")),
            data_type=str(d.get("data_type", "any")),
            required=bool(d.get("required", False)),
            default=d.get("default"),
        )


@dataclass
class MigrationInfo:
    """Minimal migration provenance for a package version upgrade."""

    migrated_from: str = ""
    migrated_at: str = ""
    migration_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migrated_from": self.migrated_from,
            "migrated_at": self.migrated_at,
            "migration_reason": self.migration_reason,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "MigrationInfo":
        if not isinstance(d, dict):
            return cls()
        return cls(
            migrated_from=str(d.get("migrated_from", "")),
            migrated_at=str(d.get("migrated_at", "")),
            migration_reason=str(d.get("migration_reason", "")),
        )

    @classmethod
    def stamp(cls, migrated_from: str, reason: str = "") -> "MigrationInfo":
        return cls(
            migrated_from=migrated_from,
            migrated_at=datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            migration_reason=reason,
        )


@dataclass
class PackageMetadata:
    """Versioned metadata for a packaged/composite node (wrapper node)."""

    package_id: str = ""
    package_version: str = "0.1.0"
    schema_version: str = "1.0"
    migration_info: MigrationInfo = field(default_factory=MigrationInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "package_version": self.package_version,
            "schema_version": self.schema_version,
            "migration_info": self.migration_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "PackageMetadata":
        if not isinstance(d, dict):
            return cls()
        return cls(
            package_id=str(d.get("package_id", "")),
            package_version=str(d.get("package_version", "0.1.0")),
            schema_version=str(d.get("schema_version", "1.0")),
            migration_info=MigrationInfo.from_dict(d.get("migration_info", {})),
        )

    @classmethod
    def create(
        cls,
        package_id: str,
        package_version: str = "0.1.0",
        schema_version: str = "1.0",
    ) -> "PackageMetadata":
        return cls(
            package_id=package_id,
            package_version=package_version,
            schema_version=schema_version,
        )

    def is_empty(self) -> bool:
        return not bool(self.package_id.strip())

    def with_migration(self, reason: str) -> "PackageMetadata":
        return PackageMetadata(
            package_id=self.package_id,
            package_version=self.package_version,
            schema_version=self.schema_version,
            migration_info=MigrationInfo.stamp(
                migrated_from=self.package_version,
                reason=reason,
            ),
        )


@dataclass
class SemanticValidationResult:
    """Result of a semantic-equivalence comparison between two SubgraphIR instances."""

    is_equivalent: bool
    differences: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_equivalent": self.is_equivalent,
            "differences": list(self.differences),
            "warnings": list(self.warnings),
        }


def _compare_ports(
    ports_a: List[SubgraphPort],
    ports_b: List[SubgraphPort],
    direction: str,
    diffs: List[str],
) -> None:
    names_a = {p.name: p for p in ports_a}
    names_b = {p.name: p for p in ports_b}
    only_a = set(names_a) - set(names_b)
    only_b = set(names_b) - set(names_a)
    if only_a:
        diffs.append(f"{direction} ports only in subgraph_a: {sorted(only_a)}")
    if only_b:
        diffs.append(f"{direction} ports only in subgraph_b: {sorted(only_b)}")
    for name in set(names_a) & set(names_b):
        if names_a[name].data_type != names_b[name].data_type:
            diffs.append(
                f"{direction} port '{name}' data_type mismatch: "
                f"{names_a[name].data_type!r} vs {names_b[name].data_type!r}"
            )


class PackageSemanticValidator:
    """Lightweight semantic-equivalence validator for SubgraphIR expand/collapse."""

    @staticmethod
    def validate(
        subgraph_a: "SubgraphIR",
        subgraph_b: "SubgraphIR",
    ) -> SemanticValidationResult:
        diffs: List[str] = []
        warnings: List[str] = []

        _compare_ports(subgraph_a.inputs, subgraph_b.inputs, "input", diffs)
        _compare_ports(subgraph_a.outputs, subgraph_b.outputs, "output", diffs)

        set_a = set(subgraph_a.internal_node_ids)
        set_b = set(subgraph_b.internal_node_ids)
        only_a = set_a - set_b
        only_b = set_b - set_a
        if only_a:
            diffs.append(f"internal nodes only in subgraph_a: {sorted(only_a)}")
        if only_b:
            diffs.append(f"internal nodes only in subgraph_b: {sorted(only_b)}")

        meta_a: PackageMetadata = subgraph_a.package_metadata
        meta_b: PackageMetadata = subgraph_b.package_metadata
        if (
            not meta_a.is_empty()
            and not meta_b.is_empty()
            and meta_a.package_id != meta_b.package_id
        ):
            warnings.append(
                f"package_id mismatch: {meta_a.package_id!r} vs {meta_b.package_id!r}"
            )

        if meta_a.schema_version != meta_b.schema_version:
            warnings.append(
                f"schema_version mismatch: {meta_a.schema_version!r} vs {meta_b.schema_version!r}"
            )

        return SemanticValidationResult(
            is_equivalent=len(diffs) == 0,
            differences=diffs,
            warnings=warnings,
        )

    @staticmethod
    def validate_roundtrip(subgraph: "SubgraphIR") -> SemanticValidationResult:
        d = subgraph.to_dict()
        restored = SubgraphIR.from_dict(d)
        return PackageSemanticValidator.validate(subgraph, restored)


@dataclass
class SubgraphIR:
    """Packaged subgraph contract (ComfyUI-style composition target)."""

    subgraph_id: str
    display_name: str
    version: str = "0.1.0"
    inputs: List[SubgraphPort] = field(default_factory=list)
    outputs: List[SubgraphPort] = field(default_factory=list)
    internal_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    package_metadata: PackageMetadata = field(default_factory=PackageMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "display_name": self.display_name,
            "version": self.version,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "internal_node_ids": list(self.internal_node_ids),
            "metadata": dict(self.metadata),
            "package_metadata": self.package_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "SubgraphIR":
        if not isinstance(d, dict):
            return cls(subgraph_id="", display_name="")
        raw_inputs = d.get("inputs") or []
        raw_outputs = d.get("outputs") or []
        return cls(
            subgraph_id=str(d.get("subgraph_id", "")),
            display_name=str(d.get("display_name", "")),
            version=str(d.get("version", "0.1.0")),
            inputs=[SubgraphPort.from_dict(p) for p in raw_inputs if isinstance(p, dict)],
            outputs=[SubgraphPort.from_dict(p) for p in raw_outputs if isinstance(p, dict)],
            internal_node_ids=[str(n) for n in (d.get("internal_node_ids") or [])],
            metadata=dict(d.get("metadata") or {}),
            package_metadata=PackageMetadata.from_dict(d.get("package_metadata", {})),
        )


@dataclass
class LayeredIRBundle:
    """Composition root for layered IR contracts."""

    mission: MissionIR = field(default_factory=MissionIR)
    behavior: BehaviorIR = field(default_factory=BehaviorIR)
    commands: List[CommandIR] = field(default_factory=list)
    safety: SafetyIR = field(default_factory=SafetyIR)
    observability: ObservabilityIR = field(default_factory=ObservabilityIR)
    execution: ExecutionIR = field(default_factory=ExecutionIR)
    subgraphs: List[SubgraphIR] = field(default_factory=list)


# =============================================================================
# Section 2 — Layered IR interfaces / bridge / serializer（合并自 layered_interfaces.py）
# =============================================================================

class LayeredIRSerializer(Protocol):
    """Serializer contract for layered IR bundles."""

    def to_dict(self, bundle: LayeredIRBundle) -> Dict[str, Any]: ...
    def from_dict(self, payload: Dict[str, Any]) -> LayeredIRBundle: ...


class LayeredIRBridge(Protocol):
    """Bridge contract between WorkflowIR and layered bundle."""

    def from_workflow_ir(self, workflow_ir: Any) -> LayeredIRBundle: ...
    def apply_to_workflow_ir(self, workflow_ir: Any, bundle: LayeredIRBundle) -> Any: ...


class DefaultLayeredIRSerializer:
    """Minimal working serializer for LayeredIRBundle round-trips."""

    def to_dict(self, bundle: LayeredIRBundle) -> Dict[str, Any]:
        return {
            "mission":       _asdict(bundle.mission),
            "behavior":      _asdict(bundle.behavior),
            "commands":      [_asdict(c) for c in bundle.commands],
            "safety":        _asdict(bundle.safety),
            "observability": _asdict(bundle.observability),
            "execution":     _asdict(bundle.execution),
            "subgraphs":     [sg.to_dict() for sg in bundle.subgraphs],
        }

    def from_dict(self, payload: Dict[str, Any]) -> LayeredIRBundle:
        if not isinstance(payload, dict):
            return LayeredIRBundle()
        subgraphs = [
            SubgraphIR.from_dict(sg)
            for sg in (payload.get("subgraphs") or [])
            if isinstance(sg, dict)
        ]
        bundle = LayeredIRBundle()
        bundle.subgraphs = subgraphs
        return bundle


class DefaultLayeredIRBridge:
    """Minimal working bridge between WorkflowIR and LayeredIRBundle."""

    _serializer = DefaultLayeredIRSerializer()

    def from_workflow_ir(self, workflow_ir: Any) -> LayeredIRBundle:
        contracts: Any = None
        if isinstance(workflow_ir, dict):
            contracts = workflow_ir.get("layered_contracts")
        elif hasattr(workflow_ir, "layered_contracts"):
            contracts = workflow_ir.layered_contracts
        if isinstance(contracts, dict):
            return self._serializer.from_dict(contracts)
        return LayeredIRBundle()

    def apply_to_workflow_ir(self, workflow_ir: Any, bundle: LayeredIRBundle) -> Any:
        serialized = self._serializer.to_dict(bundle)
        if isinstance(workflow_ir, dict):
            workflow_ir["layered_contracts"] = serialized
        elif hasattr(workflow_ir, "layered_contracts"):
            workflow_ir.layered_contracts = serialized
        elif hasattr(workflow_ir, "__dict__"):
            workflow_ir.__dict__["layered_contracts"] = serialized
        return workflow_ir


__all__ = [
    # contracts
    "MissionIR", "BehaviorStateTransition", "MotionParamSpec", "SensorBinding",
    "BehaviorIR", "CommandIR", "SafetyIR", "ObservabilityIR", "ExecutionIR",
    "SubgraphPort", "SubgraphIR", "LayeredIRBundle",
    "MigrationInfo", "PackageMetadata",
    "SemanticValidationResult", "PackageSemanticValidator",
    # interfaces
    "LayeredIRSerializer", "LayeredIRBridge",
    "DefaultLayeredIRSerializer", "DefaultLayeredIRBridge",
]
