# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.compiler.lowering — 画布 ↔ IR 双向转换 / Canvas ↔ IR roundtrip.

合并自 DEMO ``src/system/compiler/lowering/{canvas_to_ir.py, ir_to_canvas.py}``。

本模块同时承载 **WorkflowIR 数据契约** 与 **canvas ↔ IR 转换函数**：
- ``IRParam`` / ``IRNodeUI`` / ``IRNode`` / ``IREdge`` / ``EdgeType`` / ``WorkflowIR``
  迁自 DEMO ``src/system/compiler/ir/workflow_ir.py``。
- ``canvas_to_ir(canvas_data) -> WorkflowIR``    画布 JSON dict → IR
- ``ir_to_canvas(ir) -> canvas_data``             IR → 画布 JSON dict

调用方传入 dict（``.canvas.json`` 的解析结果）而非 ``QGraphicsScene``，
保持 ``application.compiler`` 层无 PyQt6 依赖（"无 I/O 的纯计算层"契约）。

阶段状态 / Phase status:
- Phase 0: 数据契约 + 空函数桩（占位返回 + 单元测试可跑）
- Phase 1: 实装 ``canvas_to_ir`` / ``ir_to_canvas``，三份真实画布 byte-stable roundtrip
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# Canonical set of param_type strings IRParam.to_dict / from_dict will accept.
# Adding a new param_type is a schema change; do it in lockstep with the node
# manifest format and the readers in env_cfg_compiler so unknown types don't
# silently round-trip and break compilation later.
_IR_PARAM_TYPES = frozenset({"string", "int", "float", "bool", "enum", "list", "json"})


# =============================================================================
# WorkflowIR 数据契约（迁自 DEMO ``workflow_ir.py``）
# =============================================================================


class EdgeType(Enum):
    """边类型：控制流 vs 数据流 / Control vs data flow."""

    FLOW = "flow"
    DATA = "data"


@dataclass
class IRParam:
    """IR 节点上的强类型参数 / Typed parameter on an IR node."""

    name: str
    value: Any
    param_type: str = "string"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to canvas-JSON shape. Strict on ``param_type``.

        Validates ``param_type`` against the canonical set and, for
        ``param_type == 'json'``, ensures ``value`` is one of the two
        encodings canvas files use historically:

          - native dict / list (newer canvases; written by IRParam
            round-trip through Python without intermediate stringify)
          - JSON-encoded ``str`` (older canvases; the on-disk form
            that the canvas authoring tool emitted for some json-typed
            fields like ``obs_terms``, ``training_items``, …)

        Both shapes are read back losslessly by downstream consumers
        (notably ``env_cfg_compiler._parse_json_param``). What we
        block here is *truly* malformed json: non-serialisable Python
        objects, unparseable JSON strings, or types other than the two
        encodings above. That preserves the 2026-05-11 contract — bad
        data must not round-trip silently — without breaking the dual
        encoding the canvas format has always supported.
        """
        if self.param_type not in _IR_PARAM_TYPES:
            raise ValueError(
                f"IRParam.to_dict: param_type {self.param_type!r} is not in "
                f"the canonical set {sorted(_IR_PARAM_TYPES)!r} "
                f"(name={self.name!r})"
            )
        if self.param_type == "json":
            v = self.value
            if isinstance(v, (dict, list)):
                try:
                    json.dumps(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"IRParam.to_dict: param_type='json' dict/list value is not "
                        f"json-serialisable (name={self.name!r}): {exc}"
                    ) from exc
            elif isinstance(v, str):
                try:
                    parsed = json.loads(v)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"IRParam.to_dict: param_type='json' string value is not "
                        f"valid JSON (name={self.name!r}): {exc}"
                    ) from exc
                if not isinstance(parsed, (dict, list)):
                    raise ValueError(
                        f"IRParam.to_dict: param_type='json' string value parsed "
                        f"to {type(parsed).__name__}, expected dict or list "
                        f"(name={self.name!r})"
                    )
            else:
                raise ValueError(
                    f"IRParam.to_dict: param_type='json' requires value to be "
                    f"dict, list, or JSON-encoded string; got "
                    f"{type(v).__name__} (name={self.name!r}, value={v!r})"
                )
        return {"name": self.name, "value": self.value, "param_type": self.param_type}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IRParam":
        """Deserialize from canvas-JSON shape. Strict on ``param_type``.

        ``param_type`` is now required (no silent 'string' default — that
        let stale canvas files round-trip with a wrong type tag and
        masked compile-time bugs). ``name`` and ``value`` are required.
        """
        if "param_type" not in d:
            raise ValueError(
                f"IRParam.from_dict: missing required 'param_type' field "
                f"(got keys={sorted(d)!r})"
            )
        if "name" not in d:
            raise ValueError(
                f"IRParam.from_dict: missing required 'name' field "
                f"(got keys={sorted(d)!r})"
            )
        if "value" not in d:
            raise ValueError(
                f"IRParam.from_dict: missing required 'value' field "
                f"(name={d.get('name')!r}, keys={sorted(d)!r})"
            )
        pt = d["param_type"]
        if pt not in _IR_PARAM_TYPES:
            raise ValueError(
                f"IRParam.from_dict: param_type {pt!r} is not in the "
                f"canonical set {sorted(_IR_PARAM_TYPES)!r} "
                f"(name={d['name']!r})"
            )
        return cls(name=d["name"], value=d["value"], param_type=pt)


@dataclass
class IRNodeUI:
    """节点 UI 元数据；不参与语义比较 / UI metadata; not part of semantic equality."""

    x: float = 0.0
    y: float = 0.0
    width: float = 180.0
    height: float = 110.0
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IRNodeUI":
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            width=float(d.get("width", 180.0)),
            height=float(d.get("height", 110.0)),
            collapsed=bool(d.get("collapsed", False)),
        )


@dataclass
class IRNode:
    """工作流 IR 中的一个节点 / A node in the workflow IR."""

    id: str
    schema_id: str  # 节点类型字符串（与 ``NodeManifest.id`` 对齐）
    kind: str  # ``NodeKind`` 的字符串值（保持 IR 与 canvas 解耦）
    params: Dict[str, IRParam] = field(default_factory=dict)
    ui: Optional[IRNodeUI] = None
    opaque_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "schema_id": self.schema_id,
            "kind": self.kind,
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "ui": self.ui.to_dict() if self.ui else None,
            "opaque_code": self.opaque_code,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IRNode":
        return cls(
            id=d["id"],
            schema_id=d["schema_id"],
            kind=d.get("kind", "custom"),
            params={
                k: IRParam.from_dict(v) for k, v in (d.get("params") or {}).items()
            },
            ui=IRNodeUI.from_dict(d["ui"]) if d.get("ui") else None,
            opaque_code=d.get("opaque_code"),
        )


@dataclass
class IREdge:
    """工作流 IR 中的一条边 / An edge in the workflow IR."""

    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    edge_type: EdgeType = EdgeType.FLOW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "edge_type": self.edge_type.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IREdge":
        return cls(
            id=d["id"],
            source_node=d["source_node"],
            source_port=d["source_port"],
            target_node=d["target_node"],
            target_port=d["target_port"],
            edge_type=EdgeType(d.get("edge_type", "flow")),
        )


@dataclass
class WorkflowIR:
    """工作流 IR 顶层容器 / Top-level workflow IR container."""

    nodes: List[IRNode] = field(default_factory=list)
    edges: List[IREdge] = field(default_factory=list)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Canvas-bound training backend engine id (e.g. ``"sb3_mujoco"`` /
    # ``"isaac_lab"``). Sole source of truth for the per-node backend kind on
    # rewards / terminations / domain_rand — those nodes no longer expose
    # ``backend`` as a ParamSpec; the canvas binds it once and downstream
    # populates derive from here via ``registers.backends.node_backend_kind``.
    backend: Optional[str] = None
    # Canonical robot SKU (mirrors top-level ``robot_id`` in canvas JSON).
    # Derived in ``CanvasPage.to_workflow_dict`` from the Robot node's
    # ``asset_id`` parameter via ``registers.robots.resolve_to_sku``; carried
    # here so bundle exporter / spec compiler can read SKU without re-scanning
    # nodes. Always equal to the SKU on the canvas's (single) Robot node
    # when set; ``None`` if the canvas has no robot node or asset_id failed
    # to resolve. See rule §1.6 (Robot identity in runtime artifacts is
    # SKU-only).
    robot_id: Optional[str] = None

    def new_node_id(self) -> str:
        return f"n_{uuid.uuid4().hex[:12]}"

    def new_edge_id(self) -> str:
        return f"e_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        # Keep human-readable diff order: version → backend → robot_id →
        # metadata → nodes → edges (matches CanvasPage.to_workflow_dict).
        out: Dict[str, Any] = {"version": self.version}
        if self.backend:
            out["backend"] = self.backend
        if self.robot_id:
            out["robot_id"] = self.robot_id
        out["metadata"] = dict(self.metadata)
        out["nodes"] = [n.to_dict() for n in self.nodes]
        out["edges"] = [e.to_dict() for e in self.edges]
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkflowIR":
        backend_raw = d.get("backend")
        backend = backend_raw.strip() if isinstance(backend_raw, str) and backend_raw.strip() else None
        robot_raw = d.get("robot_id")
        robot_id = robot_raw.strip() if isinstance(robot_raw, str) and robot_raw.strip() else None
        # Bypass / disable filter (ComfyUI 风格):
        # 顶层 ``disabled: true`` 的节点整体跳过,**不进入 ir.nodes** —— 下游所有
        # backend (spec_compiler / Isaac Lab env_cfg / SB3 trainer / bundle exporter)
        # 都按 ``ir.nodes`` 迭代,在这里过滤一次就够。
        # 任一端点落在 disabled 集合上的边也一并剥掉,避免 spec_validator 把
        # "另一端必填 port" 误报为 DANGLING_PORT。关键节点被禁则由
        # spec_validator.MISSING_REQUIRED_NODE 自然触发 —— 用户契约要求与"未添加
        # 该节点"的错误一致,无需新增错误码。
        raw_nodes = [n for n in (d.get("nodes") or [])
                     if isinstance(n, dict) and not bool(n.get("disabled", False))]
        enabled_ids = {n["id"] for n in raw_nodes if n.get("id")}
        raw_edges = [e for e in (d.get("edges") or [])
                     if isinstance(e, dict)
                     and e.get("source_node") in enabled_ids
                     and e.get("target_node") in enabled_ids]
        return cls(
            version=str(d.get("version", "1.0.0")),
            metadata=dict(d.get("metadata", {}) or {}),
            nodes=[IRNode.from_dict(n) for n in raw_nodes],
            edges=[IREdge.from_dict(e) for e in raw_edges],
            backend=backend,
            robot_id=robot_id,
        )


# =============================================================================
# 双向转换函数（Phase 0 桩；Phase 1 实装）
# =============================================================================


def canvas_to_ir(canvas_data: Dict[str, Any]) -> WorkflowIR:
    """画布 JSON dict → ``WorkflowIR`` / Canvas-shaped dict → workflow IR.

    Args:
        canvas_data: ``.canvas.json`` 解析结果，或 ``CanvasScene.serialize()`` 输出

    Returns:
        ``WorkflowIR`` 实例

    Raises:
        ValueError: ``canvas_data`` 不是 dict，或缺少 ``nodes`` / ``edges`` 键
            （Phase 0 仅支持 IR-shape 直传；GraphScene-shape Phase 1 才实装）。
            硬失败比静默退化为空 IR 更好——后者会让 spec_validator 的
            "no recognized trainer wired" 兜底盖住真因。

    阶段 / Phase:
        Phase 0: 仅接受 IR-shape dict（``nodes`` + ``edges`` 必需），from_dict 透传。
        Phase 1: 实装 GraphScene-shape → IR 翻译（DEMO ``canvas_to_ir.py`` 697 行）。
    """
    if not isinstance(canvas_data, dict):
        raise ValueError(
            f"canvas_to_ir: expected dict, got {type(canvas_data).__name__}"
        )
    if "nodes" not in canvas_data or "edges" not in canvas_data:
        raise ValueError(
            "canvas_to_ir: canvas_data is not IR-shape "
            f"(missing 'nodes'/'edges'); got keys={sorted(canvas_data)!r}. "
            "GraphScene-shape → IR translation is Phase 1; until then, callers "
            "must hand IR-shape dicts (e.g. CanvasPage.to_workflow_dict())."
        )
    return WorkflowIR.from_dict(canvas_data)


def ir_to_canvas(ir: WorkflowIR) -> Dict[str, Any]:
    """``WorkflowIR`` → 画布 JSON dict / Workflow IR → canvas-shaped dict.

    Args:
        ir: ``WorkflowIR`` 实例

    Returns:
        画布 JSON 字典；可直接 ``save_data(path, ...)`` 落盘为 ``.canvas.json``
        或交给 ``CanvasScene.deserialize()``

    阶段 / Phase:
        Phase 0: 占位实现——直接 ``ir.to_dict()`` 返回 IR-shape 字典。
        Phase 1: 实装从 DEMO ``ir_to_canvas.py`` (377 行) 行为，加 canvas 视觉布局信息。
    """
    return ir.to_dict()


__all__ = [
    # WorkflowIR 数据契约
    "EdgeType",
    "IRParam",
    "IRNodeUI",
    "IRNode",
    "IREdge",
    "WorkflowIR",
    # 转换函数
    "canvas_to_ir",
    "ir_to_canvas",
]
