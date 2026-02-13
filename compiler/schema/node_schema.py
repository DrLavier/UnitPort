#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node Schema definitions.

A schema describes a node type's ports, parameters, constraints,
code template, and robot compatibility. It is the "knowledge base"
that drives compilation, validation, and code generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from compiler.ir.types import IRType, PortDirection


@dataclass
class ParamConstraint:
    """Constraints for a parameter value."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None
    regex: Optional[str] = None

    def to_dict(self) -> dict:
        d = {}
        if self.min_value is not None:
            d["min_value"] = self.min_value
        if self.max_value is not None:
            d["max_value"] = self.max_value
        if self.choices is not None:
            d["choices"] = self.choices
        if self.regex is not None:
            d["regex"] = self.regex
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ParamConstraint':
        return cls(
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            choices=d.get("choices"),
            regex=d.get("regex"),
        )


@dataclass
class PortSchema:
    """Schema for a single port on a node."""
    name: str
    direction: PortDirection
    data_type: IRType = IRType.ANY
    required: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "data_type": self.data_type.value,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PortSchema':
        return cls(
            name=d["name"],
            direction=PortDirection(d["direction"]),
            data_type=IRType.from_string(d.get("data_type", "any")),
            required=d.get("required", False),
            description=d.get("description", ""),
        )


@dataclass
class ParamSchema:
    """Schema for a configurable parameter."""
    name: str
    param_type: IRType
    default: Any = None
    constraints: Optional[ParamConstraint] = None
    unit: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "param_type": self.param_type.value,
            "default": self.default,
        }
        if self.constraints:
            d["constraints"] = self.constraints.to_dict()
        if self.unit:
            d["unit"] = self.unit
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ParamSchema':
        constraints = None
        if "constraints" in d:
            constraints = ParamConstraint.from_dict(d["constraints"])
        return cls(
            name=d["name"],
            param_type=IRType.from_string(d.get("param_type", "string")),
            default=d.get("default"),
            constraints=constraints,
            unit=d.get("unit", ""),
            description=d.get("description", ""),
        )


@dataclass
class NodeSchema:
    """
    Complete schema for a node type.

    Defines ports, parameters, code generation template,
    robot compatibility, and safety constraints.
    """
    schema_id: str
    display_name: str
    node_type: str
    kind: str
    ports: List[PortSchema] = field(default_factory=list)
    parameters: List[ParamSchema] = field(default_factory=list)
    code_template: str = ""
    robot_compat: List[str] = field(default_factory=list)
    safety: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def get_input_ports(self) -> List[PortSchema]:
        return [p for p in self.ports if p.direction == PortDirection.INPUT]

    def get_output_ports(self) -> List[PortSchema]:
        return [p for p in self.ports if p.direction == PortDirection.OUTPUT]

    def get_parameter(self, name: str) -> Optional[ParamSchema]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "schema_id": self.schema_id,
            "display_name": self.display_name,
            "node_type": self.node_type,
            "kind": self.kind,
            "ports": [p.to_dict() for p in self.ports],
            "parameters": [p.to_dict() for p in self.parameters],
            "code_template": self.code_template,
            "robot_compat": self.robot_compat,
            "safety": self.safety,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeSchema':
        return cls(
            schema_id=d["schema_id"],
            display_name=d["display_name"],
            node_type=d["node_type"],
            kind=d["kind"],
            ports=[PortSchema.from_dict(p) for p in d.get("ports", [])],
            parameters=[ParamSchema.from_dict(p) for p in d.get("parameters", [])],
            code_template=d.get("code_template", ""),
            robot_compat=d.get("robot_compat", []),
            safety=d.get("safety", {}),
            version=d.get("version", "1.0"),
        )
