#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logic Control Nodes
"""

from typing import Dict, Any, List
from .base_node import BaseNode


class IfNode(BaseNode):
    """Conditional branch node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "if")
        self.inputs = {'condition': None}
        self.outputs = {'out_if': None, 'out_else': None}
        self.parameters = {
            'condition_expr': '',
            'elif_conditions': []  # list of condition expressions
        }

    def add_elif(self, condition_expr: str = ""):
        """Add an elif branch"""
        elif_index = len(self.parameters.get('elif_conditions', []))
        self.parameters.setdefault('elif_conditions', []).append(condition_expr)
        self.inputs[f'elif_{elif_index}'] = None
        self.outputs[f'out_elif_{elif_index}'] = None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional branch"""
        condition = inputs.get('condition', False)

        if condition:
            return {'out_if': {'value': True}, 'out_else': None}

        # Check elif conditions (if any)
        for idx in range(len(self.parameters.get('elif_conditions', []))):
            elif_cond = inputs.get(f'elif_{idx}', False)
            if elif_cond:
                return {f'out_elif_{idx}': {'value': True}, 'out_else': None}

        return {'out_if': None, 'out_else': {'value': False}}

    def get_display_name(self) -> str:
        return "If"

    def get_description(self) -> str:
        return "Select execution path based on condition"

    def to_code(self) -> str:
        condition_expr = self.get_parameter('condition_expr', '') or 'condition'
        elif_blocks = ""
        for idx, expr in enumerate(self.parameters.get('elif_conditions', [])):
            cond_expr = expr or f"elif_condition_{idx}"
            elif_blocks += f"elif {cond_expr}:\n    # elif branch {idx}\n    pass\n"
        return (
            f"# Conditional branch\n"
            f"if {condition_expr}:\n"
            f"    # true branch\n"
            f"    pass\n"
            f"{elif_blocks}"
            f"else:\n"
            f"    # false branch\n"
            f"    pass\n"
        )


class WhileLoopNode(BaseNode):
    """While loop node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "while_loop")
        self.inputs = {
            'condition': None,
            'for_start': None,
            'for_end': None,
            'for_step': None
        }
        self.outputs = {'loop_body': None, 'loop_end': None}
        self.parameters = {
            'loop_type': 'while',  # while | for
            'for_start': 0,
            'for_end': 1,
            'for_step': 1
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute loop"""
        loop_type = self.get_parameter('loop_type', 'while')
        if loop_type == 'for':
            start = inputs.get('for_start', self.get_parameter('for_start', 0))
            end = inputs.get('for_end', self.get_parameter('for_end', 1))
            step = inputs.get('for_step', self.get_parameter('for_step', 1))
            should_run = start < end if step > 0 else start > end
        else:
            should_run = inputs.get('condition', False)

        if should_run:
            return {'loop_body': {'continue': True}, 'loop_end': None}
        return {'loop_body': None, 'loop_end': {'finished': True}}

    def get_display_name(self) -> str:
        return "While Loop"

    def get_description(self) -> str:
        return "Repeat execution while condition is true"

    def to_code(self) -> str:
        loop_type = self.get_parameter('loop_type', 'while')
        if loop_type == 'for':
            start = self.get_parameter('for_start', 0)
            end = self.get_parameter('for_end', 1)
            step = self.get_parameter('for_step', 1)
            return f"# For loop\nfor i in range({start}, {end}, {step}):\n    # loop body\n    pass\n"
        condition_expr = self.get_parameter('condition_expr', '') or 'condition'
        return f"# While loop\nwhile {condition_expr}:\n    # loop body\n    pass\n"


class SwitchNode(BaseNode):
    """Switch/case node - multi-way conditional branch"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "switch")
        self.inputs = {'flow_in': None, 'value_in': None}
        self.outputs = {'case_default': None}
        self.parameters = {
            'input_expr': '',
            'cases': [],  # list of case value strings
        }

    def add_case(self, case_value: str = ""):
        idx = len(self.parameters.get('cases', []))
        self.parameters.setdefault('cases', []).append(case_value)
        self.outputs[f'case_{idx}'] = None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get('value_in', None)
        cases = self.parameters.get('cases', [])
        for idx, case_val in enumerate(cases):
            if str(value) == str(case_val):
                return {f'case_{idx}': {'matched': True, 'value': value}, 'case_default': None}
        return {'case_default': {'matched': True, 'value': value}}

    def get_display_name(self) -> str:
        return "Switch"

    def get_description(self) -> str:
        return "Multi-way conditional branch on value"

    def to_code(self) -> str:
        input_expr = self.get_parameter('input_expr', '') or 'value'
        cases = self.parameters.get('cases', [])
        lines = [f"_switch_val = {input_expr}"]
        for idx, case_val in enumerate(cases):
            kw = "if" if idx == 0 else "elif"
            lines.append(f"{kw} _switch_val == {repr(case_val)}:")
            lines.append(f"    pass  # case_{idx}")
        lines.append("else:")
        lines.append("    pass  # default")
        return "\n".join(lines) + "\n"


class TryCatchNode(BaseNode):
    """Try/Catch/Finally node - exception handling.

    Execution is driven by the engine's ``_handle_try_catch`` handler:
      1. Engine executes nodes downstream of ``try_body``.
      2. If any downstream node produces a failure result, the engine
         routes execution to ``except_body`` (caught error info is stored
         in ``results[node_id + "._caught_error"]``).
      3. ``finally_body`` always executes.
      4. ``flow_out`` continues the main workflow.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "try_catch")
        self.inputs = {'flow_in': None}
        self.outputs = {
            'try_body': None,
            'except_body': None,
            'finally_body': None,
            'flow_out': None,
        }
        self.parameters = {
            'exception_types': 'Exception',
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Return all output port keys so the engine's _handle_try_catch
        # handler can detect this node type and drive the flow.
        return {
            'try_body': {'active': True},
            'except_body': {'active': True},
            'finally_body': {'active': True},
            'flow_out': {'active': True},
        }

    def get_display_name(self) -> str:
        return "Try / Catch"

    def get_description(self) -> str:
        return "Execute with exception handling"

    def to_code(self) -> str:
        exc_types = self.get_parameter('exception_types', 'Exception') or 'Exception'
        return (
            f"try:\n"
            f"    pass  # try body\n"
            f"except {exc_types}:\n"
            f"    pass  # except body\n"
            f"finally:\n"
            f"    pass  # finally body\n"
        )


class ParallelBranchNode(BaseNode):
    """Parallel Branch node - fork execution into concurrent branches.

    Execution is driven by the engine's ``_handle_parallel`` handler which
    launches each ``branch_N`` in a separate thread.  After all branches
    complete, the engine follows ``join_out`` edges.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "parallel")
        self.inputs = {'flow_in': None}
        self.outputs = {'branch_0': None, 'branch_1': None, 'join_out': None}
        self.parameters = {
            'branch_count': 2,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        count = max(1, int(self.get_parameter('branch_count', 2)))
        result: Dict[str, Any] = {
            f'branch_{i}': {'parallel': True, 'index': i} for i in range(count)
        }
        result['join_out'] = {'pending': True}
        return result

    def get_display_name(self) -> str:
        return "Parallel Branch"

    def get_description(self) -> str:
        return "Fork execution into concurrent branches"

    def to_code(self) -> str:
        count = int(self.get_parameter('branch_count', 2))
        lines = ["import threading", "_parallel_threads = []"]
        for i in range(count):
            lines.append(f"def _branch_{i}():")
            lines.append(f"    pass  # branch_{i}")
        for i in range(count):
            lines.append(f"_t{i} = threading.Thread(target=_branch_{i})")
            lines.append(f"_parallel_threads.append(_t{i})")
        lines.append("for _t in _parallel_threads: _t.start()")
        lines.append("for _t in _parallel_threads: _t.join()")
        return "\n".join(lines) + "\n"


class JoinNode(BaseNode):
    """Join node - synchronization barrier for parallel branches.

    Collects results from all incoming branch edges and aggregates them
    into a single ``flow_out`` payload.  Works with ``ParallelBranchNode``
    whose engine handler ensures all branches complete before reaching here.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "join")
        self.inputs = {'branch_0': None, 'branch_1': None}
        self.outputs = {'flow_out': None}
        self.parameters = {
            'branch_count': 2,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        branch_count = int(self.get_parameter('branch_count', 2))
        branch_results: Dict[str, Any] = {}
        all_ok = True
        for i in range(branch_count):
            key = f'branch_{i}'
            val = inputs.get(key)
            branch_results[key] = val
            if isinstance(val, dict) and val.get('error'):
                all_ok = False
        return {
            'flow_out': {
                'joined': True,
                'status': 'success' if all_ok else 'partial_failure',
                'branch_count': branch_count,
                'branch_results': branch_results,
            }
        }

    def get_display_name(self) -> str:
        return "Join"

    def get_description(self) -> str:
        return "Synchronize parallel branches"

    def to_code(self) -> str:
        return "# Join: all parallel branches synchronized\n"


class ComparisonNode(BaseNode):
    """Comparison node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "comparison")
        self.inputs = {'left': None, 'right': None, 'value_in': None, 'compare_value': None}
        self.outputs = {'result': None}
        self.parameters = {
            'operator': '==',  # ==, !=, >, <, >=, <=
            'compare_value': 0,
            'input_expr': '',
            'output_name': ''
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comparison"""
        value = inputs.get('left', inputs.get('value_in', 0))
        operator = self.get_parameter('operator', '==')
        compare_value = inputs.get('right', inputs.get('compare_value', self.get_parameter('compare_value', 0)))

        operators = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b
        }

        result = operators.get(operator, operators['=='])(value, compare_value)

        return {'result': {'value': result}}

    def get_display_name(self) -> str:
        return "Comparison"

    def get_description(self) -> str:
        return "Compare two values"

    def to_code(self) -> str:
        operator = self.get_parameter('operator', '==')
        compare_value = self.get_parameter('compare_value', 0)
        input_expr = self.get_parameter('input_expr', '') or 'left_value'
        output_name = self.get_parameter('output_name', '') or 'result'
        return f"# Comparison: {input_expr} {operator} {compare_value}\n{output_name} = {input_expr} {operator} {compare_value}\n"
