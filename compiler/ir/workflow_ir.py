#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow IR (Intermediate Representation) data structures.

The IR is the single source of truth for any workflow.
Both Canvas and Code map to/from IR, never directly to each other.
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum


class NodeKind(Enum):
    """The kind/category of an IR node."""
    ACTION = "action"
    SENSOR = "sensor"
    LOGIC = "logic"
    MATH = "math"
    TIMER = "timer"
    VARIABLE = "variable"
    COMPARISON = "comparison"
    STOP = "stop"
    CUSTOM = "custom"
    OPAQUE = "opaque"

    @classmethod
    def from_string(cls, s: str) -> 'NodeKind':
        try:
            return cls(s.lower())
        except ValueError:
            return cls.CUSTOM


class EdgeType(Enum):
    """Edge type: control flow or data flow."""
    FLOW = "flow"
    DATA = "data"


@dataclass
class IRParam:
    """A typed parameter value on an IR node."""
    name: str
    value: Any
    param_type: str = "string"

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "param_type": self.param_type}

    @classmethod
    def from_dict(cls, d: dict) -> 'IRParam':
        return cls(name=d["name"], value=d["value"], param_type=d.get("param_type", "string"))


@dataclass
class IRNodeUI:
    """UI metadata for a node. Not part of semantic comparison."""
    x: float = 0.0
    y: float = 0.0
    width: float = 180.0
    height: float = 110.0
    collapsed: bool = False

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width,
                "height": self.height, "collapsed": self.collapsed}

    @classmethod
    def from_dict(cls, d: dict) -> 'IRNodeUI':
        return cls(
            x=d.get("x", 0.0), y=d.get("y", 0.0),
            width=d.get("width", 180.0), height=d.get("height", 110.0),
            collapsed=d.get("collapsed", False),
        )


@dataclass
class SourceSpan:
    """Code location for source mapping."""
    line_start: int = 0
    line_end: int = 0
    col_start: int = 0
    col_end: int = 0

    def to_dict(self) -> dict:
        return {"line_start": self.line_start, "line_end": self.line_end,
                "col_start": self.col_start, "col_end": self.col_end}

    @classmethod
    def from_dict(cls, d: dict) -> 'SourceSpan':
        return cls(
            line_start=d.get("line_start", 0), line_end=d.get("line_end", 0),
            col_start=d.get("col_start", 0), col_end=d.get("col_end", 0),
        )


@dataclass
class IRNode:
    """A node in the workflow IR."""
    id: str
    schema_id: str
    kind: NodeKind
    params: Dict[str, IRParam] = field(default_factory=dict)
    ui: Optional[IRNodeUI] = None
    source_span: Optional[SourceSpan] = None
    opaque_code: Optional[str] = None

    @staticmethod
    def new_id() -> str:
        """Generate a new unique node ID."""
        return str(uuid.uuid4())[:8]

    def get_param_value(self, name: str, default: Any = None) -> Any:
        """Get a parameter value by name."""
        p = self.params.get(name)
        return p.value if p else default

    def set_param(self, name: str, value: Any, param_type: str = "string"):
        """Set a parameter value."""
        self.params[name] = IRParam(name=name, value=value, param_type=param_type)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "schema_id": self.schema_id,
            "kind": self.kind.value,
            "params": {k: v.to_dict() for k, v in self.params.items()},
        }
        if self.ui:
            d["ui"] = self.ui.to_dict()
        if self.source_span:
            d["source_span"] = self.source_span.to_dict()
        if self.opaque_code is not None:
            d["opaque_code"] = self.opaque_code
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'IRNode':
        params = {k: IRParam.from_dict(v) for k, v in d.get("params", {}).items()}
        ui = IRNodeUI.from_dict(d["ui"]) if "ui" in d else None
        source_span = SourceSpan.from_dict(d["source_span"]) if "source_span" in d else None
        return cls(
            id=d["id"],
            schema_id=d["schema_id"],
            kind=NodeKind.from_string(d["kind"]),
            params=params,
            ui=ui,
            source_span=source_span,
            opaque_code=d.get("opaque_code"),
        )


@dataclass
class IREdge:
    """A directed edge in the workflow IR."""
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    edge_type: EdgeType = EdgeType.FLOW

    def to_dict(self) -> dict:
        return {
            "from_node": self.from_node, "from_port": self.from_port,
            "to_node": self.to_node, "to_port": self.to_port,
            "edge_type": self.edge_type.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'IREdge':
        return cls(
            from_node=d["from_node"], from_port=d["from_port"],
            to_node=d["to_node"], to_port=d["to_port"],
            edge_type=EdgeType(d.get("edge_type", "flow")),
        )


@dataclass
class IRVariable:
    """A workflow-level variable declaration."""
    name: str
    initial_value: Any = None
    value_type: str = "number"

    def to_dict(self) -> dict:
        return {"name": self.name, "initial_value": self.initial_value,
                "value_type": self.value_type}

    @classmethod
    def from_dict(cls, d: dict) -> 'IRVariable':
        return cls(name=d["name"], initial_value=d.get("initial_value"),
                   value_type=d.get("value_type", "number"))


@dataclass
class WorkflowIR:
    """
    The complete Workflow Intermediate Representation.
    This is the canonical, serializable workflow format.
    """
    ir_version: str = "1.0"
    name: str = ""
    robot_type: str = "go2"
    brand: str = "unitree"
    nodes: List[IRNode] = field(default_factory=list)
    edges: List[IREdge] = field(default_factory=list)
    variables: List[IRVariable] = field(default_factory=list)

    def get_node(self, node_id: str) -> Optional[IRNode]:
        """Find a node by ID."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_nodes_by_kind(self, kind: NodeKind) -> List[IRNode]:
        """Get all nodes of a specific kind."""
        return [n for n in self.nodes if n.kind == kind]

    def get_outgoing_edges(self, node_id: str) -> List[IREdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.from_node == node_id]

    def get_incoming_edges(self, node_id: str) -> List[IREdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.to_node == node_id]

    def get_entry_nodes(self) -> List[IRNode]:
        """Find nodes with no incoming flow edges."""
        nodes_with_flow_in = {e.to_node for e in self.edges if e.edge_type == EdgeType.FLOW}
        return [n for n in self.nodes if n.id not in nodes_with_flow_in
                and n.kind != NodeKind.COMPARISON]

    def add_node(self, node: IRNode):
        """Add a node to the IR."""
        self.nodes.append(node)

    def add_edge(self, edge: IREdge):
        """Add an edge to the IR."""
        self.edges.append(edge)

    def to_dict(self) -> dict:
        """Serialize the entire IR to a JSON-compatible dict."""
        return {
            "ir_version": self.ir_version,
            "name": self.name,
            "robot_type": self.robot_type,
            "brand": self.brand,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "variables": [v.to_dict() for v in self.variables],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WorkflowIR':
        """Deserialize from a dict."""
        return cls(
            ir_version=d.get("ir_version", "1.0"),
            name=d.get("name", ""),
            robot_type=d.get("robot_type", "go2"),
            brand=d.get("brand", "unitree"),
            nodes=[IRNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[IREdge.from_dict(e) for e in d.get("edges", [])],
            variables=[IRVariable.from_dict(v) for v in d.get("variables", [])],
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'WorkflowIR':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
