#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR to Canvas conversion.
Converts a WorkflowIR back to canvas graph_data format
that can be loaded by GraphScene.load_workflow().
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, NodeKind, EdgeType,
)
from compiler.lowering.layout import LayoutEngine
from compiler.semantic.diagnostics import Diagnostic, make_warning, make_info


# Reverse map: action ID to UI display name
_ACTION_ID_TO_UI = {
    "lift_right_leg": "Lift Right Leg",
    "stand": "Stand",
    "sit": "Sit",
    "walk": "Walk",
    "stop": "Stop",
}

# Reverse map: sensor type to UI display name
_SENSOR_ID_TO_UI = {
    "ultrasonic": "Read Ultrasonic",
    "infrared": "Read Infrared",
    "camera": "Read Camera",
    "imu": "Read IMU",
    "odometry": "Read Odometry",
}

# Reverse map: operator symbol to UI display name
_OP_TO_COMPARISON_UI = {
    "==": "Equal",
    "!=": "Not Equal",
    ">": "Greater Than",
    "<": "Less Than",
    ">=": "Greater Equal",
    "<=": "Less Equal",
}

# Reverse map: math operation to UI display name
_MATH_OP_TO_UI = {
    "add": "Add", "subtract": "Subtract", "multiply": "Multiply",
    "divide": "Divide", "power": "Power", "modulo": "Modulo",
    "min": "Min", "max": "Max", "abs": "Abs",
    "sum": "Sum", "average": "Average",
}


class IRToCanvas:
    """Convert a WorkflowIR to canvas-compatible graph_data."""

    def convert(self, ir: WorkflowIR) -> Tuple[Dict, List[Diagnostic]]:
        """
        Convert IR to graph_data dict suitable for GraphScene.load_workflow().

        Returns:
            (graph_data, diagnostics)
        """
        diags: List[Diagnostic] = []

        # Auto-layout if nodes don't have positions
        needs_layout = any(n.ui is None or (n.ui.x == 0 and n.ui.y == 0)
                           for n in ir.nodes)
        if needs_layout:
            engine = LayoutEngine()
            engine.layout(ir)

        nodes = []
        id_map: Dict[str, int] = {}

        for idx, ir_node in enumerate(ir.nodes):
            canvas_node, node_diags = self._convert_node(ir_node, idx)
            diags.extend(node_diags)
            if canvas_node:
                nodes.append(canvas_node)
                id_map[ir_node.id] = idx

        connections = []
        for edge in ir.edges:
            from_id = id_map.get(edge.from_node)
            to_id = id_map.get(edge.to_node)
            if from_id is not None and to_id is not None:
                connections.append({
                    "from_node": from_id,
                    "from_port": edge.from_port,
                    "to_node": to_id,
                    "to_port": edge.to_port,
                })

        graph_data = {
            "nodes": nodes,
            "connections": connections,
        }

        diags.append(make_info(
            "I4003",
            f"IR to canvas: {len(nodes)} nodes, {len(connections)} connections",
        ))

        return graph_data, diags

    def _convert_node(self, ir_node: IRNode, canvas_id: int
                      ) -> Tuple[Optional[Dict], List[Diagnostic]]:
        """Convert a single IR node to canvas node dict."""
        diags: List[Diagnostic] = []

        pos = {"x": 100, "y": 100}
        if ir_node.ui:
            pos = {"x": ir_node.ui.x, "y": ir_node.ui.y}

        base = {
            "id": canvas_id,
            "position": pos,
        }

        if ir_node.kind == NodeKind.ACTION:
            action = ir_node.get_param_value("action", "stand")
            ui_name = _ACTION_ID_TO_UI.get(action, action.replace("_", " ").title())
            base.update({
                "display_name": "Action Execution",
                "node_type": "action_execution",
                "ui_selection": ui_name,
            })

        elif ir_node.kind == NodeKind.STOP:
            base.update({
                "display_name": "Action Execution",
                "node_type": "action_execution",
                "ui_selection": "Stop",
            })

        elif ir_node.kind == NodeKind.SENSOR:
            sensor = ir_node.get_param_value("sensor_type", "imu")
            ui_name = _SENSOR_ID_TO_UI.get(sensor, f"Read {sensor.title()}")
            base.update({
                "display_name": "Sensor Input",
                "node_type": "sensor_input",
                "ui_selection": ui_name,
            })

        elif ir_node.kind == NodeKind.TIMER:
            duration = ir_node.get_param_value("duration", 1.0)
            base.update({
                "display_name": "Timer",
                "node_type": "timer",
                "duration": str(duration),
            })

        elif ir_node.kind == NodeKind.LOGIC and ir_node.schema_id == "builtin.if":
            condition = ir_node.get_param_value("condition_expr", "")
            elif_conds = ir_node.get_param_value("elif_conditions", [])
            base.update({
                "display_name": "Logic Control",
                "node_type": "if",
                "ui_selection": "If",
                "condition_expr": condition,
            })
            if elif_conds:
                base["elif_conditions"] = elif_conds

        elif ir_node.kind == NodeKind.LOGIC and ir_node.schema_id == "builtin.while_loop":
            loop_type = ir_node.get_param_value("loop_type", "while")
            condition = ir_node.get_param_value("condition_expr", "")
            if loop_type == "for":
                base.update({
                    "display_name": "Logic Control",
                    "node_type": "while_loop",
                    "ui_selection": "While Loop",
                    "loop_type": "For",
                    "condition_expr": condition,
                    "for_start": str(ir_node.get_param_value("for_start", 0)),
                    "for_end": str(ir_node.get_param_value("for_end", 10)),
                    "for_step": str(ir_node.get_param_value("for_step", 1)),
                })
            else:
                base.update({
                    "display_name": "Logic Control",
                    "node_type": "while_loop",
                    "ui_selection": "While Loop",
                    "loop_type": "While",
                    "condition_expr": condition,
                })

        elif ir_node.kind == NodeKind.COMPARISON:
            operator = ir_node.get_param_value("operator", "==")
            ui_name = _OP_TO_COMPARISON_UI.get(operator, "Equal")
            base.update({
                "display_name": "Condition",
                "node_type": "comparison",
                "ui_selection": ui_name,
                "left_value": ir_node.get_param_value("input_expr", ""),
                "right_value": ir_node.get_param_value("compare_value", "0"),
            })

        elif ir_node.kind == NodeKind.MATH:
            operation = ir_node.get_param_value("operation", "add")
            ui_name = _MATH_OP_TO_UI.get(operation, operation.title())
            base.update({
                "display_name": "Math",
                "node_type": "math",
                "ui_selection": ui_name,
            })

        elif ir_node.kind == NodeKind.VARIABLE:
            name = ir_node.get_param_value("name", "var")
            value = ir_node.get_param_value("initial_value", 0)
            base.update({
                "display_name": "Variable",
                "node_type": "variable",
                "name": name,
                "initial_value": value,
            })

        elif ir_node.kind == NodeKind.OPAQUE:
            code = ir_node.opaque_code or ir_node.get_param_value("code", "")
            base.update({
                "display_name": "Opaque Code",
                "node_type": "opaque",
                "code": code,
            })
            diags.append(make_warning(
                "W3002",
                f"Opaque code block: cannot fully reconstruct canvas node",
                node_id=ir_node.id,
            ))

        else:
            base.update({
                "display_name": f"Unknown ({ir_node.schema_id})",
                "node_type": "unknown",
            })
            diags.append(make_warning(
                "W3003",
                f"Unknown node kind: {ir_node.kind}",
                node_id=ir_node.id,
            ))

        return base, diags
