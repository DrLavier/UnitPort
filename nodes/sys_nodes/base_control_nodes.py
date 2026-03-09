#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Control Nodes - Wait, Gate, Timeout, Retry, Cancel, Abort, Break

These nodes provide fundamental flow-control primitives for workflow orchestration.
"""

import time
from typing import Dict, Any
from .base_node import BaseNode


class WaitNode(BaseNode):
    """
    Wait node - pauses execution for a duration or until an event fires.

    Modes:
    - duration: sleeps for the specified number of seconds
    - event: waits for a named event with optional timeout
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "wait")
        self.inputs = {'flow_in': None}
        self.outputs = {'flow_out': None}
        self.parameters = {
            'mode': 'duration',       # duration | event
            'duration': 1.0,          # seconds (duration mode)
            'event_name': '',         # event identifier (event mode)
            'timeout': 0.0,           # 0 = no timeout (event mode)
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.get_parameter('mode', 'duration')
        if mode == 'event':
            event_name = self.get_parameter('event_name', '')
            timeout = float(self.get_parameter('timeout', 0))
            return {'flow_out': {'waited_event': event_name, 'timeout': timeout}}
        else:
            duration = float(self.get_parameter('duration', 1.0))
            duration = max(0, min(duration, 60))
            time.sleep(duration)
            return {'flow_out': {'waited_duration': duration}}

    def get_display_name(self) -> str:
        return "Wait"

    def get_description(self) -> str:
        return "Pause execution for a duration or until an event"

    def to_code(self) -> str:
        mode = self.get_parameter('mode', 'duration')
        if mode == 'event':
            name = self.get_parameter('event_name', 'my_event')
            timeout = self.get_parameter('timeout', 0)
            return f"# Wait for event\nevent_bus.wait('{name}', timeout={timeout})\n"
        duration = self.get_parameter('duration', 1.0)
        return f"# Wait\nimport time\ntime.sleep({duration})\n"


class GateNode(BaseNode):
    """
    Gate node - routes flow based on a condition or event.

    Modes:
    - condition: evaluates an expression; flow goes to out_pass or out_block
    - event: checks if a named event has occurred
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "gate")
        self.inputs = {'flow_in': None}
        self.outputs = {'out_pass': None, 'out_block': None}
        self.parameters = {
            'mode': 'condition',       # condition | event
            'condition_expr': '',      # expression string
            'event_name': '',          # event identifier
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.get_parameter('mode', 'condition')
        if mode == 'event':
            return {'out_pass': True, 'out_block': False}
        cond = self.get_parameter('condition_expr', '')
        passed = bool(cond)
        return {'out_pass': passed, 'out_block': not passed}

    def get_display_name(self) -> str:
        return "Gate"

    def get_description(self) -> str:
        return "Route flow based on a condition or event"

    def to_code(self) -> str:
        mode = self.get_parameter('mode', 'condition')
        if mode == 'event':
            name = self.get_parameter('event_name', '')
            return f"# Gate (event)\nif event_bus.check('{name}'):\n    pass  # out_pass\nelse:\n    pass  # out_block\n"
        cond = self.get_parameter('condition_expr', 'True')
        return f"# Gate\nif {cond}:\n    pass  # out_pass\nelse:\n    pass  # out_block\n"


class TimeoutNode(BaseNode):
    """
    Timeout node - executes flow with a time limit.

    If downstream completes within duration, follows flow_out.
    If duration expires, follows out_timeout.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "timeout")
        self.inputs = {'flow_in': None}
        self.outputs = {'flow_out': None, 'out_timeout': None}
        self.parameters = {
            'duration': 5.0,           # timeout duration in seconds
            'on': 'next_step',         # next_step | event
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        duration = float(self.get_parameter('duration', 5.0))
        return {'flow_out': {'timeout_duration': duration}, 'out_timeout': None}

    def get_display_name(self) -> str:
        return "Timeout"

    def get_description(self) -> str:
        return "Execute with a time limit; branch on timeout"

    def to_code(self) -> str:
        duration = self.get_parameter('duration', 5.0)
        return (
            f"# Timeout: {duration}s\n"
            f"import time\n"
            f"_timeout_start = time.time()\n"
            f"# ... body ...\n"
            f"if time.time() - _timeout_start > {duration}:\n"
            f"    pass  # out_timeout\n"
        )


class RetryNode(BaseNode):
    """
    Retry node - retries a body block up to max_attempts with backoff.

    On success follows out_success; after all attempts fail follows out_failed.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "retry")
        self.inputs = {'flow_in': None}
        self.outputs = {'retry_body': None, 'out_success': None, 'out_failed': None}
        self.parameters = {
            'max_attempts': 3,
            'backoff_sec': 1.0,
            'strategy': 'fixed',       # fixed | exp
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        max_attempts = int(self.get_parameter('max_attempts', 3))
        return {
            'retry_body': None,
            'out_success': {'attempts': max_attempts},
            'out_failed': None,
        }

    def get_display_name(self) -> str:
        return "Retry"

    def get_description(self) -> str:
        return "Retry a block with configurable attempts and backoff"

    def to_code(self) -> str:
        ma = self.get_parameter('max_attempts', 3)
        bs = self.get_parameter('backoff_sec', 1.0)
        strategy = self.get_parameter('strategy', 'fixed')
        return (
            f"# Retry: max {ma} attempts, {strategy} backoff {bs}s\n"
            f"for _attempt in range(1, {ma} + 1):\n"
            f"    try:\n"
            f"        pass  # retry_body\n"
            f"        break  # success\n"
            f"    except Exception:\n"
            f"        time.sleep({bs}{'  * (2 ** _attempt)' if strategy == 'exp' else ''})\n"
            f"else:\n"
            f"    pass  # out_failed\n"
        )


class CancelNode(BaseNode):
    """
    Cancel node - cancels running tasks.

    Targets:
    - current: cancel the current task
    - tag: cancel tasks with a specific tag
    - all: cancel all running tasks
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "cancel")
        self.inputs = {'flow_in': None}
        self.outputs = {'flow_out': None}
        self.parameters = {
            'target': 'current',       # current | tag | all
            'tag': '',                 # tag name when target=tag
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        target = self.get_parameter('target', 'current')
        tag = self.get_parameter('tag', '')
        return {'flow_out': {'cancelled': target, 'tag': tag}}

    def get_display_name(self) -> str:
        return "Cancel"

    def get_description(self) -> str:
        return "Cancel running tasks by target or tag"

    def to_code(self) -> str:
        target = self.get_parameter('target', 'current')
        tag = self.get_parameter('tag', '')
        if target == 'tag':
            return f"# Cancel by tag\nscheduler.cancel('tag', '{tag}')\n"
        return f"# Cancel: {target}\nscheduler.cancel('{target}')\n"


class AbortNode(BaseNode):
    """
    Abort node - immediately terminates the workflow.

    Terminal node with no output ports. Raises AbortWorkflow.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "abort")
        self.inputs = {'flow_in': None}
        self.outputs = {}
        self.parameters = {
            'reason': '',
            'emergency': False,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        reason = self.get_parameter('reason', 'Workflow aborted')
        raise RuntimeError(f"AbortWorkflow: {reason}")

    def get_display_name(self) -> str:
        return "Abort"

    def get_description(self) -> str:
        return "Immediately terminate the workflow"

    def to_code(self) -> str:
        reason = self.get_parameter('reason', '')
        emergency = self.get_parameter('emergency', False)
        return (
            f"# Abort workflow\n"
            f"raise AbortWorkflow('{reason}', emergency={emergency})\n"
        )


class BreakNode(BaseNode):
    """
    Break node - exits an enclosing loop.

    The actual target loop is resolved by the graph scene auto-wiring logic.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "break")
        self.inputs = {'flow_in': None}
        self.outputs = {'break_out': None}
        self.parameters = {
            'break_level': 1,  # 1 = nearest enclosing loop
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {'break_out': {'break_level': int(self.get_parameter('break_level', 1) or 1)}}

    def get_display_name(self) -> str:
        return "Break"

    def get_description(self) -> str:
        return "Exit an enclosing loop"

    def to_code(self) -> str:
        return "break\n"
