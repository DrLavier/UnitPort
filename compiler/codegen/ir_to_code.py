#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR to Python code generator.
Generates a Python script from a WorkflowIR.
"""

from typing import List, Dict, Set, Tuple, Optional

from compiler.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind, EdgeType
from compiler.schema.registry import SchemaRegistry
from compiler.semantic.diagnostics import Diagnostic, make_warning, make_info


class SourceMap:
    """Maps IR node IDs to generated code line ranges."""

    def __init__(self):
        self._map: Dict[str, Tuple[int, int]] = {}

    def record(self, node_id: str, line_start: int, line_end: int):
        self._map[node_id] = (line_start, line_end)

    def get(self, node_id: str) -> Optional[Tuple[int, int]]:
        return self._map.get(node_id)

    def to_dict(self) -> dict:
        return self._map.copy()


class IRToCode:
    """Generate Python code from a WorkflowIR."""

    # Maps math operation to Python operator symbol
    _MATH_OP_SYMBOLS = {
        "add": "+", "subtract": "-", "multiply": "*",
        "divide": "/", "power": "**", "modulo": "%",
    }

    def generate(self, ir: WorkflowIR) -> Tuple[str, List[Diagnostic], SourceMap]:
        """
        Generate Python code from the IR.

        Returns:
            (code_string, diagnostics, source_map) tuple
        """
        self._ir = ir
        self._diags: List[Diagnostic] = []
        self._source_map = SourceMap()
        self._generated: Set[str] = set()

        # Build adjacency for code generation
        self._outgoing: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self._incoming: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        for node in ir.nodes:
            self._outgoing[node.id] = {}
            self._incoming[node.id] = {}

        for edge in ir.edges:
            self._outgoing.setdefault(edge.from_node, {}).setdefault(
                edge.from_port, []).append((edge.to_node, edge.to_port))
            self._incoming.setdefault(edge.to_node, {}).setdefault(
                edge.to_port, []).append((edge.from_node, edge.from_port))

        lines: List[str] = []

        # File header
        lines.extend([
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            '"""Auto-generated workflow code"""',
            "",
            "import time",
            "from bin.core.robot_context import RobotContext",
            "",
            "",
            "def execute_workflow(robot=None):",
            "    '''Execute the visual workflow'''",
        ])

        # Generate condition nodes first (they provide data to if nodes)
        for node in ir.nodes:
            if node.kind == NodeKind.COMPARISON:
                out_edges = self._outgoing.get(node.id, {})
                result_targets = out_edges.get("result", [])
                for target_id, target_port in result_targets:
                    if target_port == "condition":
                        code = self._generate_node_code(node.id, indent=1)
                        if code:
                            lines.extend(code)
                            lines.append("")
                        break

        # Generate from entry nodes
        entry_nodes = ir.get_entry_nodes()
        # Sort by x position for deterministic order
        entry_nodes.sort(key=lambda n: (n.ui.x if n.ui else 0))

        for entry in entry_nodes:
            if entry.id not in self._generated:
                code = self._generate_node_code(entry.id, indent=1)
                if code:
                    lines.extend(code)
                    lines.append("")

        # Check if any code was generated
        body_lines = [l for l in lines[10:] if l.strip()]
        if not body_lines:
            lines.append("    pass  # No connected workflow")

        # Main block
        lines.extend([
            "",
            "if __name__ == '__main__':",
            "    # Initialize robot (simulation or real)",
            "    # from models import get_robot_model",
            "    # robot = get_robot_model('go2')",
            "    robot = None  # Replace with actual robot instance",
            "    execute_workflow(robot)",
        ])

        code = "\n".join(lines)

        self._diags.append(make_info(
            "I4001",
            f"Code generated: {len(ir.nodes)} nodes, {len(ir.edges)} edges",
        ))

        return code, self._diags, self._source_map

    def _generate_node_code(self, node_id: str, indent: int = 1) -> List[str]:
        """Recursively generate code for a node and its downstream flow."""
        if node_id in self._generated:
            return []

        node = self._ir.get_node(node_id)
        if node is None:
            return []

        self._generated.add(node_id)
        indent_str = "    " * indent
        lines: List[str] = []
        line_start = -1  # Will be set later with actual line numbers

        outgoing = self._outgoing.get(node_id, {})

        if node.kind == NodeKind.START:
            robot_type = node.get_param_value("robot_type", "go2")
            lines.append(f"{indent_str}RobotContext.set_robot_type('{robot_type}')")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.END:
            pass  # End node produces no code and does not follow flow

        elif node.kind == NodeKind.LOGIC and node.schema_id == "builtin.if":
            lines.extend(self._gen_if(node, indent))

        elif node.kind == NodeKind.LOGIC and node.schema_id == "builtin.while_loop":
            loop_type = node.get_param_value("loop_type", "while")
            if loop_type == "for":
                lines.extend(self._gen_for(node, indent))
            else:
                lines.extend(self._gen_while(node, indent))

        elif node.kind == NodeKind.COMPARISON:
            lines.extend(self._gen_comparison(node, indent))

        elif node.kind == NodeKind.ACTION:
            action = node.get_param_value("action", "stand")
            lines.append(f"{indent_str}RobotContext.run_action('{action}')")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.STOP:
            lines.append(f"{indent_str}RobotContext.stop()")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.SENSOR:
            sensor_type = node.get_param_value("sensor_type", "imu")
            lines.append(f"{indent_str}# Sensor read: {sensor_type}")
            lines.append(f"{indent_str}sensor_data = RobotContext.get_sensor_data()")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.TIMER:
            duration = node.get_param_value("duration", 1.0)
            unit = node.get_param_value("unit", "seconds")
            if unit == "milliseconds":
                lines.append(f"{indent_str}time.sleep({duration} / 1000)")
            else:
                lines.append(f"{indent_str}time.sleep({duration})")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.MATH:
            lines.extend(self._gen_math(node, indent))
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.VARIABLE:
            name = node.get_param_value("name", "var")
            value = node.get_param_value("initial_value", 0)
            lines.append(f"{indent_str}{name} = {repr(value)}")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.WAIT:
            mode = node.get_param_value("mode", "duration")
            if mode == "event":
                event_name = node.get_param_value("event_name", "my_event")
                timeout = node.get_param_value("timeout", 0)
                lines.append(f"{indent_str}event_bus.wait('{event_name}', timeout={timeout})")
            else:
                duration = node.get_param_value("duration", 1.0)
                lines.append(f"{indent_str}time.sleep({duration})")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.GATE:
            mode = node.get_param_value("mode", "condition")
            if mode == "event":
                event_name = node.get_param_value("event_name", "")
                lines.append(f"{indent_str}if event_bus.check('{event_name}'):")
            else:
                cond = node.get_param_value("condition_expr", "True")
                if not cond:
                    cond = "True"
                lines.append(f"{indent_str}if {cond}:")
            # out_pass branch
            pass_targets = outgoing.get("out_pass", [])
            if pass_targets:
                for target_id, _ in pass_targets:
                    code = self._generate_node_code(target_id, indent + 1)
                    lines.extend(code)
            else:
                lines.append(f"{indent_str}    pass")
            # out_block branch
            block_targets = outgoing.get("out_block", [])
            if block_targets:
                lines.append(f"{indent_str}else:")
                for target_id, _ in block_targets:
                    code = self._generate_node_code(target_id, indent + 1)
                    lines.extend(code)

        elif node.kind == NodeKind.TIMEOUT:
            duration = node.get_param_value("duration", 5.0)
            lines.append(f"{indent_str}_timeout_start = time.time()")
            # flow_out (normal path)
            self._follow_flow(node_id, "flow_out", indent, lines)
            # out_timeout branch
            lines.append(f"{indent_str}if time.time() - _timeout_start > {duration}:")
            timeout_targets = outgoing.get("out_timeout", [])
            if timeout_targets:
                for target_id, _ in timeout_targets:
                    code = self._generate_node_code(target_id, indent + 1)
                    lines.extend(code)
            else:
                lines.append(f"{indent_str}    pass  # timeout")

        elif node.kind == NodeKind.RETRY:
            max_attempts = node.get_param_value("max_attempts", 3)
            backoff = node.get_param_value("backoff_sec", 1.0)
            strategy = node.get_param_value("strategy", "fixed")
            lines.append(f"{indent_str}for _attempt in range(1, {max_attempts} + 1):")
            lines.append(f"{indent_str}    try:")
            # retry_body
            body_targets = outgoing.get("retry_body", [])
            if body_targets:
                for target_id, _ in body_targets:
                    code = self._generate_node_code(target_id, indent + 2)
                    lines.extend(code)
            else:
                lines.append(f"{indent_str}        pass  # retry body")
            lines.append(f"{indent_str}        break  # success")
            lines.append(f"{indent_str}    except Exception:")
            if strategy == "exp":
                lines.append(f"{indent_str}        time.sleep({backoff} * (2 ** _attempt))")
            else:
                lines.append(f"{indent_str}        time.sleep({backoff})")
            lines.append(f"{indent_str}else:")
            # out_failed
            fail_targets = outgoing.get("out_failed", [])
            if fail_targets:
                for target_id, _ in fail_targets:
                    code = self._generate_node_code(target_id, indent + 1)
                    lines.extend(code)
            else:
                lines.append(f"{indent_str}    pass  # all retries failed")
            # out_success (code after the loop)
            self._follow_flow(node_id, "out_success", indent, lines)

        elif node.kind == NodeKind.CANCEL:
            target = node.get_param_value("target", "current")
            tag = node.get_param_value("tag", "")
            if target == "tag" and tag:
                lines.append(f"{indent_str}scheduler.cancel('tag', '{tag}')")
            else:
                lines.append(f"{indent_str}scheduler.cancel('{target}')")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.ABORT:
            reason = node.get_param_value("reason", "")
            emergency = node.get_param_value("emergency", False)
            lines.append(f"{indent_str}raise AbortWorkflow('{reason}', emergency={emergency})")
            # Terminal node — no follow_flow

        elif node.kind == NodeKind.SWITCH:
            lines.extend(self._gen_switch(node, indent))

        elif node.kind == NodeKind.TRY_CATCH:
            lines.extend(self._gen_try_catch(node, indent))

        elif node.kind == NodeKind.PARALLEL:
            lines.extend(self._gen_parallel(node, indent))

        elif node.kind == NodeKind.JOIN:
            # Join is a synchronization point; downstream code follows flow_out
            lines.append(f"{indent_str}# Join: all parallel branches synchronized")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.OPAQUE:
            code = node.opaque_code or node.get_param_value("code", "")
            if code:
                lines.append(f"{indent_str}# [opaque code block]")
                for code_line in code.split("\n"):
                    lines.append(f"{indent_str}{code_line}")
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.PROTOCOL_EMIT:
            lines.extend(self._gen_protocol_emit(node, indent))
            self._follow_flow(node_id, "flow_out", indent, lines)

        elif node.kind == NodeKind.RUN_POLICY:
            lines.extend(self._gen_run_policy(node, indent))
            self._follow_flow(node_id, "flow_out", indent, lines)

        else:
            # Unknown node type - try to use schema code_template
            schema = SchemaRegistry.get(node.schema_id)
            if schema and schema.code_template:
                template = schema.code_template
                for pname, pparam in node.params.items():
                    template = template.replace(f"{{{pname}}}", str(pparam.value))
                lines.append(f"{indent_str}{template}")
            else:
                lines.append(f"{indent_str}# Unknown node: {node.schema_id}")
            self._follow_flow(node_id, "flow_out", indent, lines)

        return lines

    def _gen_protocol_emit(self, node: "IRNode", indent: int) -> List[str]:
        """Generate protocol payload emission code for a PROTOCOL_EMIT node.

        Circle 4: produces Python code that builds a motor-weight protocol
        payload and assigns it to ``protocol_payload``.

        If a ``signal`` param is set, uses the signal-convention path
        (ProtocolPayloadCompiler.for_signal).  Otherwise uses the sensor-IMU
        path (ProtocolPayloadCompiler.from_sensor_imu) assuming upstream
        ``sensor_data`` variable is in scope from a SENSOR node.
        """
        indent_str = "    " * indent
        lines: List[str] = []
        motor_key    = node.get_param_value("motor_key",    "")
        param_key    = node.get_param_value("param_key",    "gain")
        address      = node.get_param_value("address",      0)
        max_age_ms   = node.get_param_value("max_age_ms",   1000)
        signal       = node.get_param_value("signal",       "")
        sensor_field = node.get_param_value("sensor_field", "accel_z")

        lines.append(f"{indent_str}# Protocol emit: motor_key={motor_key!r}")
        lines.append(
            f"{indent_str}from compiler.lowering.protocol_compiler import ProtocolPayloadCompiler"
        )

        if signal:
            # Signal-convention path (event_enable / event_disable / event_switch)
            lines.append(
                f"{indent_str}protocol_payload = ProtocolPayloadCompiler.for_signal("
                f"signal={signal!r}, motor_key={motor_key!r}, "
                f"address={address!r}, max_age_ms={max_age_ms!r})"
            )
        else:
            # Sensor→protocol path: reads sensor_data from prior SENSOR node.
            lines.append(
                f"{indent_str}protocol_payload = ProtocolPayloadCompiler.from_sensor_imu("
                f"sensor_data=sensor_data, motor_key={motor_key!r}, "
                f"param_key={param_key!r}, sensor_field={sensor_field!r}, "
                f"address={address!r}, max_age_ms={max_age_ms!r})"
            )
        return lines

    def _gen_run_policy(self, node: "IRNode", indent: int) -> List[str]:
        """Generate a run_policy command dict for a RUN_POLICY node.

        Circle 6: emits a structured command payload carrying all lowered params.
        Circle 7 will connect this dict to real runtime dispatch via
        RobotContext.run_policy(**_run_policy_cmd) or an equivalent executor.
        """
        indent_str = "    " * indent
        policy_id        = node.get_param_value("policy_id",        "")
        max_steps        = node.get_param_value("max_steps",        1000)
        velocity_command = node.get_param_value("velocity_command", [0.0, 0.0, 0.0])
        render           = node.get_param_value("render",           False)

        return [
            f"{indent_str}# run_policy: policy_id={policy_id!r}",
            f"{indent_str}_run_policy_cmd = {{",
            f"{indent_str}    'kind': 'run_policy',",
            f"{indent_str}    'policy_id': {policy_id!r},",
            f"{indent_str}    'max_steps': {max_steps!r},",
            f"{indent_str}    'velocity_command': {velocity_command!r},",
            f"{indent_str}    'render': {render!r},",
            f"{indent_str}}}",
        ]

    def _follow_flow(self, node_id: str, port: str, indent: int,
                     lines: List[str]):
        """Follow flow_out connections and generate downstream code."""
        outgoing = self._outgoing.get(node_id, {})
        targets = outgoing.get(port, [])
        for target_id, _ in targets:
            code = self._generate_node_code(target_id, indent)
            lines.extend(code)

    def _gen_if(self, node: IRNode, indent: int) -> List[str]:
        """Generate if/elif/else code."""
        indent_str = "    " * indent
        lines = []
        outgoing = self._outgoing.get(node.id, {})

        # Get condition
        condition = self._get_condition_text(node)
        lines.append(f"{indent_str}if {condition}:")

        # True branch
        true_targets = outgoing.get("out_if", [])
        if true_targets:
            for target_id, _ in true_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        else:
            lines.append(f"{indent_str}    pass")

        # Elif branches
        elif_conditions = node.get_param_value("elif_conditions", [])
        if isinstance(elif_conditions, list):
            for i, elif_cond in enumerate(elif_conditions):
                elif_cond = elif_cond.strip() if elif_cond else "False"
                if not elif_cond:
                    elif_cond = "False"
                lines.append(f"{indent_str}elif {elif_cond}:")
                elif_targets = outgoing.get(f"out_elif_{i}", [])
                if elif_targets:
                    for target_id, _ in elif_targets:
                        code = self._generate_node_code(target_id, indent + 1)
                        lines.extend(code)
                else:
                    lines.append(f"{indent_str}    pass")

        # Else branch
        false_targets = outgoing.get("out_else", [])
        if false_targets:
            lines.append(f"{indent_str}else:")
            for target_id, _ in false_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)

        return lines

    def _gen_while(self, node: IRNode, indent: int) -> List[str]:
        """Generate while loop code."""
        indent_str = "    " * indent
        lines = []
        outgoing = self._outgoing.get(node.id, {})

        condition = self._get_condition_text(node)
        lines.append(f"{indent_str}while {condition}:")

        body_targets = outgoing.get("loop_body", [])
        if body_targets:
            for target_id, _ in body_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        else:
            lines.append(f"{indent_str}    pass")

        # Loop end (code after the while)
        end_targets = outgoing.get("loop_end", [])
        for target_id, _ in end_targets:
            code = self._generate_node_code(target_id, indent)
            lines.extend(code)

        return lines

    def _gen_for(self, node: IRNode, indent: int) -> List[str]:
        """Generate for loop code."""
        indent_str = "    " * indent
        lines = []
        outgoing = self._outgoing.get(node.id, {})

        start = node.get_param_value("for_start", 0)
        end = node.get_param_value("for_end", 10)
        step = node.get_param_value("for_step", 1)
        lines.append(f"{indent_str}for i in range({start}, {end}, {step}):")

        body_targets = outgoing.get("loop_body", [])
        if body_targets:
            for target_id, _ in body_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        else:
            lines.append(f"{indent_str}    pass")

        end_targets = outgoing.get("loop_end", [])
        for target_id, _ in end_targets:
            code = self._generate_node_code(target_id, indent)
            lines.extend(code)

        return lines

    def _gen_comparison(self, node: IRNode, indent: int) -> List[str]:
        """Generate comparison assignment code."""
        indent_str = "    " * indent
        input_expr = node.get_param_value("input_expr", "0")
        compare_value = node.get_param_value("compare_value", "0")
        operator = node.get_param_value("operator", "==")
        output_name = node.get_param_value("output_name", f"condition_{node.id}")

        if not input_expr:
            input_expr = "0"

        return [f"{indent_str}{output_name} = {input_expr} {operator} {compare_value}"]

    def _gen_math(self, node: IRNode, indent: int) -> List[str]:
        """Generate math operation code."""
        indent_str = "    " * indent
        operation = node.get_param_value("operation", "add")
        value_a = node.get_param_value("value_a", 0)
        value_b = node.get_param_value("value_b", 0)

        if operation in self._MATH_OP_SYMBOLS:
            symbol = self._MATH_OP_SYMBOLS[operation]
            return [f"{indent_str}result = {value_a} {symbol} {value_b}"]
        elif operation == "abs":
            return [f"{indent_str}result = abs({value_a})"]
        elif operation == "min":
            return [f"{indent_str}result = min({value_a}, {value_b})"]
        elif operation == "max":
            return [f"{indent_str}result = max({value_a}, {value_b})"]
        elif operation == "sum":
            return [f"{indent_str}result = sum(values)"]
        elif operation == "average":
            return [f"{indent_str}result = sum(values) / len(values)"]
        else:
            return [f"{indent_str}# Unknown math operation: {operation}"]

    def _gen_switch(self, node: IRNode, indent: int) -> List[str]:
        """Generate switch/case code."""
        indent_str = "    " * indent
        lines = []
        outgoing = self._outgoing.get(node.id, {})
        input_expr = node.get_param_value("input_expr", "") or "value"
        cases = node.get_param_value("cases", [])
        if not isinstance(cases, list):
            cases = []
        lines.append(f"{indent_str}_switch_val = {input_expr}")
        for idx, case_val in enumerate(cases):
            kw = "if" if idx == 0 else "elif"
            lines.append(f"{indent_str}{kw} _switch_val == {repr(str(case_val))}:")
            case_targets = outgoing.get(f"case_{idx}", [])
            if case_targets:
                for target_id, _ in case_targets:
                    code = self._generate_node_code(target_id, indent + 1)
                    lines.extend(code)
            else:
                lines.append(f"{indent_str}    pass")
        lines.append(f"{indent_str}else:")
        default_targets = outgoing.get("case_default", [])
        if default_targets:
            for target_id, _ in default_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        else:
            lines.append(f"{indent_str}    pass  # default")
        return lines

    def _gen_try_catch(self, node: IRNode, indent: int) -> List[str]:
        """Generate try/except/finally code."""
        indent_str = "    " * indent
        lines = []
        outgoing = self._outgoing.get(node.id, {})
        exc_types = node.get_param_value("exception_types", "Exception") or "Exception"
        lines.append(f"{indent_str}try:")
        try_targets = outgoing.get("try_body", [])
        if try_targets:
            for target_id, _ in try_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        else:
            lines.append(f"{indent_str}    pass")
        lines.append(f"{indent_str}except {exc_types}:")
        except_targets = outgoing.get("except_body", [])
        if except_targets:
            for target_id, _ in except_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        else:
            lines.append(f"{indent_str}    pass")
        finally_targets = outgoing.get("finally_body", [])
        if finally_targets:
            lines.append(f"{indent_str}finally:")
            for target_id, _ in finally_targets:
                code = self._generate_node_code(target_id, indent + 1)
                lines.extend(code)
        # flow_out: code after the try/except block
        self._follow_flow(node.id, "flow_out", indent, lines)
        return lines

    def _gen_parallel(self, node: IRNode, indent: int) -> List[str]:
        """Generate parallel branch code using threading."""
        indent_str = "    " * indent
        lines = []
        outgoing = self._outgoing.get(node.id, {})
        count = int(node.get_param_value("branch_count", 2))
        lines.append(f"{indent_str}import threading")
        lines.append(f"{indent_str}_parallel_threads = []")
        for i in range(count):
            branch_targets = outgoing.get(f"branch_{i}", [])
            lines.append(f"{indent_str}def _branch_{i}():")
            if branch_targets:
                for target_id, _ in branch_targets:
                    code = self._generate_node_code(target_id, indent + 1)
                    lines.extend(code)
            else:
                lines.append(f"{indent_str}    pass")
        for i in range(count):
            lines.append(f"{indent_str}_t{i} = threading.Thread(target=_branch_{i})")
            lines.append(f"{indent_str}_parallel_threads.append(_t{i})")
        lines.append(f"{indent_str}for _t in _parallel_threads: _t.start()")
        lines.append(f"{indent_str}for _t in _parallel_threads: _t.join()")
        # join_out: optional merge point
        self._follow_flow(node.id, "join_out", indent, lines)
        return lines

    def _get_condition_text(self, node: IRNode) -> str:
        """Get condition text for if/while nodes."""
        # Check if condition port is connected
        incoming = self._incoming.get(node.id, {})
        condition_sources = incoming.get("condition", [])

        if condition_sources:
            source_id, source_port = condition_sources[0]
            source_node = self._ir.get_node(source_id)
            if source_node and source_node.kind == NodeKind.COMPARISON:
                output_name = source_node.get_param_value("output_name", "")
                if output_name:
                    return output_name

        # Fallback: use condition_expr parameter
        expr = node.get_param_value("condition_expr", "")
        if expr:
            return expr

        return "condition"
