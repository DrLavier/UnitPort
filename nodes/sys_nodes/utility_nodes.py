#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Nodes - Math and Timer operations

These nodes provide basic computational and timing utilities for workflows.
"""

import time
from typing import Dict, Any, List
from .base_node import BaseNode


class MathNode(BaseNode):
    """
    Math operation node - performs mathematical operations on inputs.

    Supports:
    - Basic: add, subtract, multiply, divide
    - Advanced: power, modulo, abs, min, max
    - Aggregation: sum, average (for multiple inputs)
    """

    OPERATIONS = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else float('inf'),
        'power': lambda a, b: a ** b,
        'modulo': lambda a, b: a % b if b != 0 else 0,
        'min': lambda a, b: min(a, b),
        'max': lambda a, b: max(a, b),
    }

    def __init__(self, node_id: str):
        super().__init__(node_id, "math")
        self.inputs = {
            'a': None,      # First operand
            'b': None,      # Second operand
            'values': None  # For aggregation operations (list)
        }
        self.outputs = {'result': None}
        self.parameters = {
            'operation': 'add',  # add, subtract, multiply, divide, etc.
            'value_a': 0,       # Default value for a
            'value_b': 0,       # Default value for b
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute math operation."""
        operation = self.get_parameter('operation', 'add')

        # Get input values (from connection or parameter)
        a = inputs.get('a')
        if a is None:
            a = self.get_parameter('value_a', 0)
        if isinstance(a, dict):
            a = a.get('value', 0)

        b = inputs.get('b')
        if b is None:
            b = self.get_parameter('value_b', 0)
        if isinstance(b, dict):
            b = b.get('value', 0)

        # Convert to numbers
        try:
            a = float(a)
            b = float(b)
        except (ValueError, TypeError):
            return {'result': {'value': 0, 'error': 'Invalid input'}}

        # Handle aggregation operations
        if operation == 'sum':
            values = inputs.get('values', [a, b])
            if isinstance(values, dict):
                values = values.get('value', [a, b])
            if not isinstance(values, (list, tuple)):
                values = [a, b]
            result = sum(float(v) for v in values)
        elif operation == 'average':
            values = inputs.get('values', [a, b])
            if isinstance(values, dict):
                values = values.get('value', [a, b])
            if not isinstance(values, (list, tuple)):
                values = [a, b]
            values = [float(v) for v in values]
            result = sum(values) / len(values) if values else 0
        elif operation == 'abs':
            result = abs(a)
        elif operation in self.OPERATIONS:
            result = self.OPERATIONS[operation](a, b)
        else:
            result = a + b  # Default to add

        return {'result': {'value': result, 'operation': operation}}

    def get_display_name(self) -> str:
        return "Math"

    def get_description(self) -> str:
        return "Perform mathematical operations (add, subtract, multiply, divide, etc.)"

    def to_code(self) -> str:
        operation = self.get_parameter('operation', 'add')
        value_a = self.get_parameter('value_a', 0)
        value_b = self.get_parameter('value_b', 0)

        op_symbols = {
            'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/',
            'power': '**', 'modulo': '%'
        }

        if operation in op_symbols:
            symbol = op_symbols[operation]
            return (
                f"# Math: {operation}\n"
                f"a = {value_a}  # or from input\n"
                f"b = {value_b}  # or from input\n"
                f"result = a {symbol} b\n"
            )
        elif operation == 'abs':
            return f"# Math: absolute value\nresult = abs({value_a})\n"
        elif operation == 'sum':
            return f"# Math: sum\nresult = sum(values)\n"
        elif operation == 'average':
            return f"# Math: average\nresult = sum(values) / len(values)\n"
        elif operation in ('min', 'max'):
            return f"# Math: {operation}\nresult = {operation}(a, b)\n"
        else:
            return f"# Math: {operation}\nresult = a + b\n"


class TimerNode(BaseNode):
    """
    Timer node - delays execution for specified duration.

    Similar to time.sleep() but can be interrupted and reports progress.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "timer")
        self.inputs = {'flow_in': None, 'duration': None}
        self.outputs = {'flow_out': None}
        self.parameters = {
            'duration': 1.0,  # Duration in seconds
            'unit': 'seconds'  # seconds, milliseconds
        }
        self._interrupted = False

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute timer (wait for duration)."""
        # Get duration from input or parameter
        duration = inputs.get('duration')
        if duration is None:
            duration = self.get_parameter('duration', 1.0)
        if isinstance(duration, dict):
            duration = duration.get('value', 1.0)

        try:
            duration = float(duration)
        except (ValueError, TypeError):
            duration = 1.0

        # Convert based on unit
        unit = self.get_parameter('unit', 'seconds')
        if unit == 'milliseconds':
            duration = duration / 1000.0

        # Clamp duration
        duration = max(0, min(duration, 60))  # Max 60 seconds

        # Execute wait
        self._interrupted = False
        start_time = time.time()

        # Sleep in small increments to allow interruption
        while time.time() - start_time < duration:
            if self._interrupted:
                break
            time.sleep(0.01)  # 10ms increments

        elapsed = time.time() - start_time

        return {
            'flow_out': {
                'status': 'interrupted' if self._interrupted else 'completed',
                'requested_duration': duration,
                'actual_duration': elapsed
            }
        }

    def interrupt(self):
        """Interrupt the timer."""
        self._interrupted = True

    def get_display_name(self) -> str:
        return "Timer"

    def get_description(self) -> str:
        return "Wait for specified duration (like sleep)"

    def to_code(self) -> str:
        duration = self.get_parameter('duration', 1.0)
        unit = self.get_parameter('unit', 'seconds')

        if unit == 'milliseconds':
            return (
                f"# Timer: {duration}ms\n"
                f"import time\n"
                f"time.sleep({duration} / 1000)\n"
            )
        else:
            return (
                f"# Timer: {duration}s\n"
                f"import time\n"
                f"time.sleep({duration})\n"
            )


class VariableNode(BaseNode):
    """
    Variable node - stores and passes values.

    Useful for storing intermediate results or constants.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "variable")
        self.inputs = {'set_value': None}
        self.outputs = {'value': None}
        self.parameters = {
            'name': 'var',
            'initial_value': 0,
            'value_type': 'number'  # number, string, boolean
        }
        self._stored_value = None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute variable node."""
        # Check if value is being set from input
        if 'set_value' in inputs and inputs['set_value'] is not None:
            value = inputs['set_value']
            if isinstance(value, dict):
                value = value.get('value', value)
            self._stored_value = value
        elif self._stored_value is None:
            self._stored_value = self.get_parameter('initial_value', 0)

        return {
            'value': {
                'value': self._stored_value,
                'name': self.get_parameter('name', 'var')
            }
        }

    def get_display_name(self) -> str:
        return "Variable"

    def get_description(self) -> str:
        return "Store and pass values"

    def to_code(self) -> str:
        name = self.get_parameter('name', 'var')
        initial = self.get_parameter('initial_value', 0)
        return f"# Variable: {name}\n{name} = {repr(initial)}\n"
