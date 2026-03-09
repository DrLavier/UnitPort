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
    "Start": "start",
    "End": "end",
    "Action Execution": "action_execution",
    "Sensor Input": "sensor_input",
    "Logic Control": "if",
    "Loop": "while_loop",
    "While Loop": "while_loop",
    "Condition": "comparison",
    "Math": "math",
    "Timer": "timer",
    "Variable": "variable",
    "Stop": "stop",
    "Wait": "wait",
    "Gate": "gate",
    "Timeout": "timeout",
    "Retry": "retry",
    "Cancel": "cancel",
    "Abort": "abort",
    "Break": "break",
    "For Loop": "while_loop",
    "Switch": "switch",
    "Try": "try_catch",
    "Parallel": "parallel",
    "Join": "join",
    "Script Box": "opaque_code",
    "OnEvent": "opaque_code",
    "PublishEvent": "opaque_code",
    "SetState": "opaque_code",
    "CheckState": "opaque_code",
    "HumanConfirm": "opaque_code",
    # Circle 1 Step 1.3: explicit behavior node — previously ambiguously mapped
    # to action_execution with external_kind=behavior; now has its own type so
    # canvas→IR lowering and runtime dispatch are unambiguous.
    "Behavior": "behavior_call",
    # Circle 4: protocol emit node — compiles node outputs to motor-weight protocol payload.
    "Protocol Emit": "protocol_emit",
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
    "loop_break_in", "break_out",
    "out_pass", "out_block",
    "out_timeout",
    "retry_body", "out_success", "out_failed",
    "try_body", "except_body", "finally_body",
    "case_default", "join_out",
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

        # Sync robot_type and brand from Start node if present
        start_nodes = ir.get_nodes_by_kind(NodeKind.START)
        if start_nodes:
            rt = start_nodes[0].get_param_value("robot_type", "go2")
            ir.robot_type = rt
            rb = start_nodes[0].get_param_value("robot_brand", "")
            ir.brand = rb.lower() if rb else self._brand_for(rt)

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

        # Legacy compatibility: old logic node used one display name + selection.
        if "Logic Control" in display_name:
            if ui_selection.lower().startswith("while") or ui_selection.lower().startswith("for"):
                node_type = "while_loop"
            else:
                node_type = "if"

        # Handle Stop node disguised as Action Execution with Stop preset
        if node_type == "action_execution" and ui_selection == "Stop":
            node_type = "stop"

        # Circle 1 Step 1.3: behavior_call has an explicit IR kind;
        # handle before the schema registry lookup so it never falls back to CUSTOM.
        # Circle 4: protocol_emit similarly bypasses the registry.
        if node_type == "behavior_call":
            schema_id = "behavior_call"
            kind = NodeKind.BEHAVIOR_CALL
        elif node_type == "protocol_emit":
            # Use builtin schema id so SemanticValidator can resolve schema metadata.
            schema_id = "builtin.protocol_emit"
            kind = NodeKind.PROTOCOL_EMIT
        else:
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

        # Circle 1 Step 1.3: behavior_call extracts behavior_ref from ui_selection
        # (operator sets the ref name) or falls back to display_name.
        if node_type == "behavior_call":
            behavior_ref = (
                node_data.get("behavior_ref")
                or ui_selection
                or node_data.get("display_name", "")
            ).strip()
            params["behavior_ref"] = IRParam("behavior_ref", behavior_ref, "string")

        elif node_type == "start":
            robot_brand = node_data.get("robot_brand", "Unitree")
            params["robot_brand"] = IRParam("robot_brand", robot_brand, "string")
            robot_type = node_data.get("robot_type", ui_selection or "go2")
            params["robot_type"] = IRParam("robot_type", robot_type, "string")

        elif node_type == "end":
            pass  # No parameters

        elif node_type == "action_execution":
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

        elif node_type == "wait":
            external_kind = str(node_data.get("external_kind", "") or "").lower()
            is_trigger = external_kind == "trigger"
            wait_mode = "event" if is_trigger else node_data.get("wait_mode", "duration")
            params["mode"] = IRParam("mode", wait_mode, "string")
            dur_text = node_data.get("duration", "1.0")
            try:
                dur = float(dur_text) if dur_text else 1.0
            except (ValueError, TypeError):
                dur = 1.0
            params["duration"] = IRParam("duration", dur, "float")
            event_name = node_data.get("event_name", "")
            if is_trigger:
                event_name = node_data.get("trigger_event", event_name)
            params["event_name"] = IRParam("event_name", event_name, "string")
            to_text = node_data.get("timeout", "0")
            if is_trigger:
                to_text = node_data.get("trigger_timeout", to_text)
            try:
                to_val = float(to_text) if to_text else 0.0
            except (ValueError, TypeError):
                to_val = 0.0
            params["timeout"] = IRParam("timeout", to_val, "float")

        elif node_type == "gate":
            params["mode"] = IRParam("mode", node_data.get("gate_mode", "condition"), "string")
            params["condition_expr"] = IRParam("condition_expr",
                                               node_data.get("condition_expr", ""), "string")
            params["event_name"] = IRParam("event_name", node_data.get("event_name", ""), "string")

        elif node_type == "timeout":
            dur_text = node_data.get("duration", "5.0")
            try:
                dur = float(dur_text) if dur_text else 5.0
            except (ValueError, TypeError):
                dur = 5.0
            params["duration"] = IRParam("duration", dur, "float")
            params["on"] = IRParam("on", node_data.get("timeout_on", "next_step"), "string")

        elif node_type == "retry":
            ma_text = node_data.get("max_attempts", "3")
            try:
                ma = int(ma_text) if ma_text else 3
            except (ValueError, TypeError):
                ma = 3
            params["max_attempts"] = IRParam("max_attempts", ma, "int")
            bs_text = node_data.get("backoff_sec", "1.0")
            try:
                bs = float(bs_text) if bs_text else 1.0
            except (ValueError, TypeError):
                bs = 1.0
            params["backoff_sec"] = IRParam("backoff_sec", bs, "float")
            params["strategy"] = IRParam("strategy", node_data.get("strategy", "fixed"), "string")

        elif node_type == "cancel":
            params["target"] = IRParam("target", node_data.get("cancel_target", "current"), "string")
            params["tag"] = IRParam("tag", node_data.get("cancel_tag", ""), "string")

        elif node_type == "abort":
            params["reason"] = IRParam("reason", node_data.get("abort_reason", ""), "string")
            emergency = node_data.get("abort_emergency", "false")
            params["emergency"] = IRParam("emergency",
                                          str(emergency).lower() in ("true", "1", "yes"),
                                          "bool")

        elif node_type == "break":
            params["break_level"] = IRParam(
                "break_level",
                self._safe_int(node_data.get("break_level", "1"), 1),
                "int",
            )

        elif node_type == "switch":
            params["input_expr"] = IRParam("input_expr",
                                           node_data.get("switch_expr", ""), "string")
            cases = node_data.get("switch_cases", [])
            params["cases"] = IRParam("cases", cases, "list")

        elif node_type == "try_catch":
            params["exception_types"] = IRParam("exception_types",
                                                 node_data.get("exception_types", "Exception"),
                                                 "string")

        elif node_type == "parallel":
            count_raw = node_data.get("branch_count", "2")
            try:
                count = int(count_raw)
            except (ValueError, TypeError):
                count = 2
            params["branch_count"] = IRParam("branch_count", count, "int")

        elif node_type == "join":
            count_raw = node_data.get("join_count") or node_data.get("branch_count", "2")
            try:
                count = int(count_raw)
            except (ValueError, TypeError):
                count = 2
            params["branch_count"] = IRParam("branch_count", count, "int")

        elif node_type == "protocol_emit":
            # Circle 4: extract protocol emission params from canvas node data.
            params["motor_key"]   = IRParam("motor_key",   node_data.get("motor_key",   ""),        "string")
            params["param_key"]   = IRParam("param_key",   node_data.get("param_key",   "gain"),    "string")
            params["address"]     = IRParam("address",     node_data.get("address",     0),         "int")
            params["dtype"]       = IRParam("dtype",       node_data.get("dtype",       "float32"), "string")
            params["max_age_ms"]  = IRParam("max_age_ms",  node_data.get("max_age_ms",  1000),      "int")
            params["signal"]      = IRParam("signal",      node_data.get("signal",      ""),        "string")
            params["sensor_field"]= IRParam("sensor_field",node_data.get("sensor_field","accel_z"), "string")

        elif node_type == "opaque_code":
            params["code"] = IRParam("code", node_data.get("code", ""), "string")
            params["script_ref"] = IRParam("script_ref", node_data.get("script_ref", ""), "string")
            params["script_io_spec"] = IRParam(
                "script_io_spec",
                node_data.get("script_io_spec", {"inputs": [], "outputs": []}),
                "dict",
            )
            params["shell_name"] = IRParam(
                "shell_name",
                node_data.get("shell_name", node_data.get("display_name", "Script Box")),
                "string",
            )
            params["state_event_fields"] = IRParam(
                "state_event_fields",
                node_data.get("state_event_fields", {}),
                "dict",
            )
            params["behavior_extension_notes"] = IRParam(
                "behavior_extension_notes",
                node_data.get("behavior_extension_notes", {}),
                "dict",
            )

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
                   or from_port.startswith("out_elif")
                   or from_port.startswith("branch_")
                   or to_port.startswith("branch_")
                   or from_port.startswith("case_"))
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
        try:
            from models.brand_registry import BrandRegistry
            mapping = BrandRegistry().get_robot_brand_map()
            if robot_type in mapping:
                return mapping[robot_type]
        except Exception:
            pass
        _fallback = {
            "go2": "unitree", "a1": "unitree", "b1": "unitree",
            "b2": "unitree", "h1": "unitree",
        }
        return _fallback.get(robot_type, "unknown")
