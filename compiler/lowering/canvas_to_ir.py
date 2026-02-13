#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canvas to IR conversion.
Converts the GraphScene export data into a WorkflowIR.
"""

from typing import Dict, Any, List, Tuple

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, IRParam, IRNodeUI, NodeKind, EdgeType,
)
from compiler.schema.registry import SchemaRegistry
from compiler.semantic.diagnostics import Diagnostic, make_error, make_warning

# Maps display names used in the canvas to schema node types
_DISPLAY_NAME_TO_NODE_TYPE = {
    "Action Execution": "action_execution",
    "Sensor Input": "sensor_input",
    "Logic Control": "if",
    "Condition": "comparison",
    "Math": "math",
    "Timer": "timer",
    "Variable": "variable",
    "Stop": "stop",
}

# Maps UI action display text to robot action identifiers
_ACTION_UI_TO_ID = {
    "Lift Right Leg": "lift_right_leg",
    "Stand": "stand",
    "Sit": "sit",
    "Walk": "walk",
    "Stop": "stop",
}

# Maps UI sensor display text to sensor type identifiers
_SENSOR_UI_TO_ID = {
    "Read Ultrasonic": "ultrasonic",
    "Read Infrared": "infrared",
    "Read Camera": "camera",
    "Read IMU": "imu",
    "Read Odometry": "odometry",
}

# Maps UI comparison display text to operator symbols
_COMPARISON_UI_TO_OP = {
    "Equal": "==",
    "Not Equal": "!=",
    "Greater Than": ">",
    "Less Than": "<",
    "Greater Equal": ">=",
    "Less Equal": "<=",
}

# Maps UI math display text to operation identifiers
_MATH_UI_TO_OP = {
    "Add": "add", "Subtract": "subtract", "Multiply": "multiply",
    "Divide": "divide", "Power": "power", "Modulo": "modulo",
    "Min": "min", "Max": "max", "Abs": "abs",
    "Sum": "sum", "Average": "average",
}

# Ports that carry control flow
_FLOW_PORTS = {
    "flow_in", "flow_out",
    "out_if", "out_else",
    "loop_body", "loop_end",
}


class CanvasToIR:
    """Convert GraphScene exported data to a WorkflowIR."""

    def convert(self, graph_data: Dict[str, Any],
                robot_type: str = "go2") -> Tuple[WorkflowIR, List[Diagnostic]]:
        """
        Convert canvas graph data to IR.

        Args:
            graph_data: Dict from GraphScene.export_graph_data() or serialize_workflow()
            robot_type: Current robot type

        Returns:
            (WorkflowIR, diagnostics) tuple
        """
        diags: List[Diagnostic] = []
        ir = WorkflowIR(robot_type=robot_type, brand=self._brand_for(robot_type))

        # Map old canvas node IDs to new IR node IDs
        id_map: Dict[int, str] = {}

        for node_data in graph_data.get("nodes", []):
            ir_node, node_diags = self._convert_node(node_data)
            diags.extend(node_diags)
            if ir_node:
                ir.add_node(ir_node)
                old_id = node_data.get("id")
                if old_id is not None:
                    id_map[old_id] = ir_node.id

        for conn_data in graph_data.get("connections", []):
            edge, edge_diags = self._convert_edge(conn_data, id_map)
            diags.extend(edge_diags)
            if edge:
                ir.add_edge(edge)

        return ir, diags

    def _convert_node(self, node_data: Dict[str, Any]) -> Tuple[IRNode, List[Diagnostic]]:
        """Convert a single canvas node to an IR node."""
        diags = []
        display_name = node_data.get("display_name", "")
        node_id = str(node_data.get("id", IRNode.new_id()))
        ui_selection = node_data.get("ui_selection", "")

        # Determine node type and resolve Logic Control special case
        node_type = node_data.get("node_type", "unknown")
        if node_type == "unknown":
            node_type = _DISPLAY_NAME_TO_NODE_TYPE.get(display_name, "unknown")

        # Handle Logic Control type resolution (if vs while_loop)
        if "Logic Control" in display_name:
            if ui_selection.lower().startswith("while") or ui_selection.lower().startswith("for"):
                node_type = "while_loop"
            else:
                node_type = "if"

        # Handle Stop node disguised as Action Execution with Stop preset
        if node_type == "action_execution" and ui_selection == "Stop":
            node_type = "stop"

        # Find schema
        schema = SchemaRegistry.get_by_node_type(node_type)
        if schema is None:
            diags.append(make_warning(
                "E2001",
                f"No schema found for node type '{node_type}' (display: '{display_name}')",
                node_id=node_id,
            ))
            schema_id = f"unknown.{node_type}"
            kind = NodeKind.CUSTOM
        else:
            schema_id = schema.schema_id
            kind = NodeKind.from_string(schema.kind)

        # Build params from UI state
        params = self._extract_params(node_data, node_type, ui_selection)

        # UI metadata
        pos = node_data.get("position", {})
        ui = IRNodeUI(
            x=pos.get("x", 0),
            y=pos.get("y", 0),
            width=node_data.get("width", 180),
            height=node_data.get("height", 110),
        )

        ir_node = IRNode(
            id=node_id,
            schema_id=schema_id,
            kind=kind,
            params=params,
            ui=ui,
        )
        return ir_node, diags

    def _extract_params(self, node_data: Dict, node_type: str,
                        ui_selection: str) -> Dict[str, IRParam]:
        """Extract IR parameters from canvas node data."""
        params: Dict[str, IRParam] = {}

        if node_type == "action_execution":
            action = _ACTION_UI_TO_ID.get(ui_selection, ui_selection.lower().replace(" ", "_"))
            params["action"] = IRParam("action", action, "string")

        elif node_type == "stop":
            pass  # No parameters

        elif node_type == "sensor_input":
            sensor = _SENSOR_UI_TO_ID.get(ui_selection, "imu")
            params["sensor_type"] = IRParam("sensor_type", sensor, "string")

        elif node_type == "if":
            cond = node_data.get("condition_expr", "")
            params["condition_expr"] = IRParam("condition_expr", cond, "string")
            elif_conds = node_data.get("elif_conditions", [])
            if elif_conds:
                params["elif_conditions"] = IRParam("elif_conditions", elif_conds, "string")

        elif node_type == "while_loop":
            loop_type = (node_data.get("loop_type", "While") or "While").lower()
            params["loop_type"] = IRParam("loop_type", loop_type, "string")
            cond = node_data.get("condition_expr", "")
            params["condition_expr"] = IRParam("condition_expr", cond, "string")
            params["for_start"] = IRParam("for_start",
                                          self._safe_int(node_data.get("for_start", "0"), 0), "int")
            params["for_end"] = IRParam("for_end",
                                        self._safe_int(node_data.get("for_end", "10"), 10), "int")
            params["for_step"] = IRParam("for_step",
                                         self._safe_int(node_data.get("for_step", "1"), 1), "int")

        elif node_type == "comparison":
            operator = _COMPARISON_UI_TO_OP.get(ui_selection, "==")
            params["operator"] = IRParam("operator", operator, "string")
            params["input_expr"] = IRParam("input_expr",
                                           node_data.get("left_value", ""), "string")
            params["compare_value"] = IRParam("compare_value",
                                              node_data.get("right_value", "0"), "string")
            params["output_name"] = IRParam("output_name",
                                            f"condition_{node_data.get('id', 0)}", "string")

        elif node_type == "math":
            operation = _MATH_UI_TO_OP.get(ui_selection, "add")
            params["operation"] = IRParam("operation", operation, "string")

        elif node_type == "timer":
            duration_text = node_data.get("duration", "1.0")
            try:
                duration = float(duration_text) if duration_text else 1.0
            except (ValueError, TypeError):
                duration = 1.0
            params["duration"] = IRParam("duration", duration, "float")
            params["unit"] = IRParam("unit", "seconds", "string")

        elif node_type == "variable":
            params["name"] = IRParam("name", node_data.get("name", "var"), "string")
            params["initial_value"] = IRParam("initial_value",
                                              node_data.get("initial_value", 0), "any")

        return params

    def _convert_edge(self, conn_data: Dict, id_map: Dict[int, str]
                      ) -> Tuple[IREdge, List[Diagnostic]]:
        """Convert a canvas connection to an IR edge."""
        diags = []
        from_id = id_map.get(conn_data.get("from_node"))
        to_id = id_map.get(conn_data.get("to_node"))

        if not from_id or not to_id:
            diags.append(make_warning(
                "W3001",
                f"Skipping edge with unmapped node ID: "
                f"{conn_data.get('from_node')} -> {conn_data.get('to_node')}",
            ))
            return None, diags

        from_port = conn_data.get("from_port", "flow_out")
        to_port = conn_data.get("to_port", "flow_in")

        # Determine edge type
        is_flow = (from_port in _FLOW_PORTS or to_port in _FLOW_PORTS
                   or from_port.startswith("out_elif"))
        edge_type = EdgeType.FLOW if is_flow else EdgeType.DATA

        edge = IREdge(
            from_node=from_id,
            from_port=from_port,
            to_node=to_id,
            to_port=to_port,
            edge_type=edge_type,
        )
        return edge, diags

    @staticmethod
    def _safe_int(val, default: int = 0) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _brand_for(robot_type: str) -> str:
        brands = {
            "go2": "unitree", "a1": "unitree", "b1": "unitree",
            "b2": "unitree", "h1": "unitree",
        }
        return brands.get(robot_type, "unknown")
