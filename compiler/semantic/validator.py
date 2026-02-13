#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic validator for WorkflowIR.
Validates the IR against schema constraints.
"""

from typing import List

from compiler.ir.workflow_ir import WorkflowIR, IRNode, NodeKind, EdgeType
from compiler.schema.registry import SchemaRegistry
from compiler.semantic.diagnostics import (
    Diagnostic, DiagnosticLevel, DiagnosticLocation,
    make_error, make_warning, make_info,
)


class SemanticValidator:
    """Validate a WorkflowIR against its node schemas."""

    def validate(self, ir: WorkflowIR) -> List[Diagnostic]:
        """
        Run all validation checks on the IR.

        Returns:
            List of diagnostics (errors, warnings, info).
        """
        diags: List[Diagnostic] = []

        diags.extend(self._check_schemas_exist(ir))
        diags.extend(self._check_param_types(ir))
        diags.extend(self._check_param_constraints(ir))
        diags.extend(self._check_dangling_edges(ir))
        diags.extend(self._check_robot_compat(ir))

        if not any(d.level == DiagnosticLevel.ERROR for d in diags):
            diags.append(make_info(
                "I4001",
                f"Validation passed ({len(ir.nodes)} nodes, {len(ir.edges)} edges)",
            ))

        return diags

    def _check_schemas_exist(self, ir: WorkflowIR) -> List[Diagnostic]:
        """Check that every node references a valid schema."""
        diags = []
        for node in ir.nodes:
            if node.kind == NodeKind.OPAQUE:
                continue
            schema = SchemaRegistry.get(node.schema_id)
            if schema is None:
                diags.append(make_error(
                    "E2001",
                    f"Unknown schema: {node.schema_id}",
                    node_id=node.id,
                    suggestion=f"Available schemas: {', '.join(SchemaRegistry.list_schema_ids())}",
                ))
        return diags

    def _check_param_types(self, ir: WorkflowIR) -> List[Diagnostic]:
        """Check parameter value types match schema expectations."""
        diags = []
        for node in ir.nodes:
            schema = SchemaRegistry.get(node.schema_id)
            if not schema:
                continue

            for param in schema.parameters:
                ir_param = node.params.get(param.name)
                if ir_param is None:
                    continue

                value = ir_param.value
                if value is None:
                    continue

                expected_type = param.param_type.value
                if expected_type == "int":
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        diags.append(make_error(
                            "E2003",
                            f"Parameter '{param.name}' expects int, got '{value}'",
                            node_id=node.id,
                        ))
                elif expected_type == "float":
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        diags.append(make_error(
                            "E2003",
                            f"Parameter '{param.name}' expects float, got '{value}'",
                            node_id=node.id,
                        ))
                elif expected_type == "bool":
                    if not isinstance(value, bool):
                        diags.append(make_warning(
                            "E2003",
                            f"Parameter '{param.name}' expects bool, got '{type(value).__name__}'",
                            node_id=node.id,
                        ))

        return diags

    def _check_param_constraints(self, ir: WorkflowIR) -> List[Diagnostic]:
        """Check parameter values against schema constraints."""
        diags = []
        for node in ir.nodes:
            schema = SchemaRegistry.get(node.schema_id)
            if not schema:
                continue

            for param_schema in schema.parameters:
                ir_param = node.params.get(param_schema.name)
                if ir_param is None or ir_param.value is None:
                    continue

                constraints = param_schema.constraints
                if not constraints:
                    continue

                value = ir_param.value

                # Check choices
                if constraints.choices and value not in constraints.choices:
                    diags.append(make_error(
                        "E2004",
                        f"Parameter '{param_schema.name}' value '{value}' "
                        f"not in allowed choices: {constraints.choices}",
                        node_id=node.id,
                    ))

                # Check numeric range
                if constraints.min_value is not None or constraints.max_value is not None:
                    try:
                        num_val = float(value)
                        if constraints.min_value is not None and num_val < constraints.min_value:
                            diags.append(make_error(
                                "E2003",
                                f"Parameter '{param_schema.name}' value {num_val} "
                                f"below minimum {constraints.min_value}",
                                node_id=node.id,
                            ))
                        if constraints.max_value is not None and num_val > constraints.max_value:
                            diags.append(make_error(
                                "E2003",
                                f"Parameter '{param_schema.name}' value {num_val} "
                                f"above maximum {constraints.max_value}",
                                node_id=node.id,
                            ))
                    except (ValueError, TypeError):
                        pass

        return diags

    def _check_dangling_edges(self, ir: WorkflowIR) -> List[Diagnostic]:
        """Check for edges referencing nonexistent nodes."""
        diags = []
        node_ids = {n.id for n in ir.nodes}
        for edge in ir.edges:
            if edge.from_node not in node_ids:
                diags.append(make_error(
                    "E2005",
                    f"Edge references nonexistent source node: {edge.from_node}",
                ))
            if edge.to_node not in node_ids:
                diags.append(make_error(
                    "E2005",
                    f"Edge references nonexistent target node: {edge.to_node}",
                ))
        return diags

    def _check_robot_compat(self, ir: WorkflowIR) -> List[Diagnostic]:
        """Check that action/sensor nodes are compatible with the robot type."""
        diags = []
        robot_type = ir.robot_type

        for node in ir.nodes:
            schema = SchemaRegistry.get(node.schema_id)
            if not schema:
                continue

            if schema.robot_compat and robot_type not in schema.robot_compat:
                diags.append(make_warning(
                    "E2007",
                    f"Node '{schema.display_name}' may not be compatible "
                    f"with robot '{robot_type}'",
                    node_id=node.id,
                ))

        return diags
