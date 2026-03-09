#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow runner for legacy graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Set

try:
    from bin.core.logger import log_info, log_debug, log_warning, log_error
except (ImportError, AttributeError):
    def log_info(*a, **k): pass    # type: ignore[misc]
    def log_debug(*a, **k): pass   # type: ignore[misc]
    def log_warning(*a, **k): pass # type: ignore[misc]
    def log_error(*a, **k): pass   # type: ignore[misc]


@dataclass
class WorkflowRunner:
    """Execute graph-scene execution graph with control-flow support."""

    max_loop_iterations: int = 100

    def run(
        self,
        exec_graph: Dict[str, Any],
        robot_model: Any,
        graph_scene: Any = None,
        action_mapping: Dict[str, str] | None = None,
        node_status_callback=None,
        cancel_check=None,
    ) -> Dict[str, Any]:
        """Run an execution graph and return a summary dict.

        Args:
            node_status_callback: Optional callable(node_id, status) fired for
                each node lifecycle event.  status ∈ {"running", "success",
                "failed"}.  When None, behaviour is unchanged (backward-compat).
            cancel_check: Optional callable() -> bool injected by RuntimeEngine
                (Cycle 2 STAGE-06).  Returns True when cancellation is requested.
                Checked at the start of each node execution — None → no cancel.
        """
        if not exec_graph.get("nodes"):
            return {"ok": False, "reason": "no_nodes"}

        has_action = any(
            node.get("type") in ("action_execution", "stop")
            or "Action Execution" in node.get("name", "")
            for node in exec_graph["nodes"].values()
        )

        if has_action and robot_model is not None:
            try:
                reset_ok = True
                if hasattr(robot_model, "reset_simulation"):
                    reset_ok = robot_model.reset_simulation()
                if not reset_ok:
                    return {"ok": False, "reason": "simulation_reset_failed", "has_action": has_action}
            except Exception as exc:
                log_error(f"Simulation reset failed: {exc}")
                return {"ok": False, "reason": "simulation_reset_failed", "has_action": has_action}

        results: Dict[str, Any] = {}
        executed: Set[str] = set()
        executed_count = 0
        cancelled = False
        failed = False

        def execute_node(node_id: str):
            nonlocal executed_count, cancelled, failed
            if node_id in executed:
                return
            if failed:
                return
            # Cycle 2 STAGE-06: stop traversal when cancellation is requested.
            if cancel_check is not None and cancel_check():
                cancelled = True
                return
            if node_id not in exec_graph["nodes"]:
                return

            node_data = exec_graph["nodes"][node_id]
            node_name = node_data.get("name", "")
            node_type = node_data.get("type", "unknown")
            logic_node = node_data.get("logic_node")
            outgoing = exec_graph["outgoing"].get(node_id, {})
            incoming = exec_graph["incoming"].get(node_id, {})

            executed.add(node_id)
            executed_count += 1
            log_info(f"Executing node: {node_name} (ID: {node_id}, Type: {node_type})")

            # Notify UI that this node is now running (Cycle 1 STAGE-03).
            # Backward-compat: callback is optional; absent → no-op.
            if node_status_callback is not None:
                node_status_callback(node_id, "running")

            # Circle 1 Step 1.3: explicit behavior node detection in exec_graph path.
            # behavior_call nodes cannot execute in WorkflowRunner (no behavior bridge);
            # return a deterministic "skipped" result with a clear reason code and log
            # a warning so operators know the WorkflowIR path is required.
            _is_behavior_node = (
                node_type in ("behavior_call", "behavior")
                or node_data.get("external_kind") == "behavior"
            )
            if _is_behavior_node:
                behavior_ref = node_data.get("behavior_ref") or node_data.get("ui_selection", "")
                log_warning(
                    f"WorkflowRunner: behavior node '{node_id}' (ref='{behavior_ref}') "
                    "cannot execute on exec_graph path — "
                    "set UNITPORT_BEHAVIOR_ENABLED=1 or add a behavior node to trigger "
                    "the WorkflowIR path.  Producing skipped result with reason code."
                )
                results[node_id] = {
                    "status": "skipped",
                    "reason": "behavior_call_requires_workflowir",
                    "behavior_ref": behavior_ref,
                }
                if node_status_callback is not None:
                    node_status_callback(node_id, "success")
                # Continue flow traversal so downstream non-behavior nodes still run.
                flow_targets = outgoing.get("flow_out", [])
                for target_id, _ in flow_targets:
                    execute_node(target_id)
                return

            if "Logic Control" in node_name or node_name == "Loop" or node_type == "while_loop":
                self._execute_logic_control(
                    node_id=node_id,
                    node_data=node_data,
                    exec_graph=exec_graph,
                    outgoing=outgoing,
                    results=results,
                    executed=executed,
                    execute_node=execute_node,
                )
                # Logic control nodes do not produce a failure result themselves.
                if node_status_callback is not None:
                    node_status_callback(node_id, "success")
                return

            if "Condition" in node_name:
                results[node_id] = self._execute_condition_node(node_id, node_data, exec_graph, results)
                if node_status_callback is not None:
                    _r = results.get(node_id, {})
                    _s = "failed" if self._is_failure_result(_r) else "success"
                    node_status_callback(node_id, _s)
                return

            if logic_node:
                if node_type in ("action_execution", "sensor_input", "stop"):
                    logic_node.set_parameter("robot_model", robot_model)

                inputs = {}
                for port_name, sources in incoming.items():
                    for source_id, source_port in sources:
                        if source_id in results:
                            source_result = results[source_id]
                            if isinstance(source_result, dict) and source_port in source_result:
                                inputs[port_name] = source_result[source_port]

                try:
                    results[node_id] = logic_node.execute(inputs)
                    log_debug(f"Node {node_id} result: {results[node_id]}")
                except Exception as exc:
                    log_error(f"Node {node_id} execution failed: {exc}")
                    results[node_id] = {"error": str(exc)}
            else:
                ui_selection = node_data.get("ui_selection", "")
                if "Action Execution" in node_name and robot_model:
                    mapping = action_mapping or {}
                    if not mapping and graph_scene is not None:
                        mapping = getattr(graph_scene, "_action_mapping", {})
                    action = mapping.get(ui_selection, ui_selection.lower().replace(" ", "_"))
                    log_info(f"Executing action: {action}")
                    try:
                        success = robot_model.run_action(action)
                        if success:
                            results[node_id] = {"status": "success", "action": action}
                        else:
                            results[node_id] = {
                                "status": "failed",
                                "action": action,
                                "error": f"action_execution_failed:{action}",
                            }
                    except Exception as exc:
                        log_error(f"Action execution failed: {exc}")
                        results[node_id] = {"error": str(exc)}

            # Fire completion callback for regular nodes (logic + action branches).
            if node_status_callback is not None:
                _r = results.get(node_id, {})
                _s = "failed" if self._is_failure_result(_r) else "success"
                node_status_callback(node_id, _s)
            _node_result = results.get(node_id, {})
            if self._is_failure_result(_node_result):
                failed = True
                return

            flow_targets = outgoing.get("flow_out", [])
            for target_id, _ in flow_targets:
                execute_node(target_id)

        for entry_id in exec_graph.get("entry_nodes", []):
            execute_node(entry_id)

        return {
            "ok": not cancelled and not failed,
            "reason": "mission_cancelled" if cancelled else ("node_execution_failed" if failed else ""),
            "results": results,
            "executed_count": executed_count,
            "has_action": has_action,
            "cancelled": cancelled,
        }

    def _execute_logic_control(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        exec_graph: Dict[str, Any],
        outgoing: Dict[str, Any],
        results: Dict[str, Any],
        executed: Set[str],
        execute_node,
    ) -> None:
        ui_selection = node_data.get("ui_selection", "If")
        # Loop nodes store type/range in logic_node.parameters (surfaced as
        # node_data["parameters"]), not at the top level of node_data.
        params = node_data.get("parameters", {})

        if ui_selection.lower().startswith("if"):
            condition_result = self._evaluate_condition(node_id, node_data, exec_graph, results)
            log_debug(f"If condition evaluated to: {condition_result}")
            branch = "out_if" if condition_result else "out_else"
            for target_id, _ in outgoing.get(branch, []):
                execute_node(target_id)
            return

        # A "Loop" canvas node has no ui_selection; detect it via node type or
        # an explicit loop_type parameter, in addition to the legacy "while"
        # ui_selection from Logic Control nodes.
        loop_type = (
            params.get("loop_type")
            or node_data.get("loop_type")
            or ("while" if ui_selection.lower().startswith("while") else None)
        )
        if loop_type is not None:
            if loop_type == "for":
                try:
                    start = int(params.get("for_start") or node_data.get("for_start", "0") or "0")
                    end   = int(params.get("for_end")   or node_data.get("for_end",   "10") or "10")
                    step  = int(params.get("for_step")  or node_data.get("for_step",  "1")  or "1")
                except ValueError:
                    start, end, step = 0, 10, 1

                log_debug(f"For loop: range({start}, {end}, {step})")
                for i in range(start, end, step):
                    results[f"{node_id}_i"] = i
                    for target_id, _ in outgoing.get("loop_body", []):
                        executed.discard(target_id)
                        execute_node(target_id)
            else:
                iteration = 0
                while iteration < self.max_loop_iterations:
                    if not self._evaluate_condition(node_id, node_data, exec_graph, results):
                        break
                    for target_id, _ in outgoing.get("loop_body", []):
                        executed.discard(target_id)
                        execute_node(target_id)
                    iteration += 1

                if iteration >= self.max_loop_iterations:
                    log_warning(f"While loop exceeded max iterations ({self.max_loop_iterations})")

            for target_id, _ in outgoing.get("loop_end", []):
                execute_node(target_id)

    def _evaluate_condition(
        self, node_id: str, node_data: Dict[str, Any], exec_graph: Dict[str, Any], results: Dict[str, Any]
    ) -> bool:
        incoming = exec_graph["incoming"].get(node_id, {})
        condition_sources = incoming.get("condition", [])

        if condition_sources:
            source_id, _ = condition_sources[0]
            if source_id not in results and source_id in exec_graph["nodes"]:
                source_data = exec_graph["nodes"][source_id]
                results[source_id] = self._execute_condition_node(source_id, source_data, exec_graph, results)

            if source_id in results:
                source_result = results[source_id]
                if isinstance(source_result, dict):
                    value = source_result.get("result", {}).get("value", False)
                    return bool(value)

        condition_expr = node_data.get("condition_expr", "")
        if condition_expr:
            try:
                return self._safe_eval_condition(condition_expr, results)
            except Exception as exc:
                log_warning(f"Condition evaluation failed: {exc}")
                return False
        return False

    def _execute_condition_node(
        self, node_id: str, node_data: Dict[str, Any], exec_graph: Dict[str, Any], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        logic_node = node_data.get("logic_node")
        if logic_node:
            incoming = exec_graph["incoming"].get(node_id, {})
            inputs = {}

            for port_name, sources in incoming.items():
                for source_id, source_port in sources:
                    if source_id in results:
                        source_result = results[source_id]
                        if isinstance(source_result, dict) and source_port in source_result:
                            inputs[port_name] = source_result[source_port]

            if "left" not in inputs:
                left_val = node_data.get("left_value", "0")
                try:
                    inputs["left"] = float(left_val) if "." in left_val else int(left_val)
                except ValueError:
                    inputs["left"] = left_val

            if "right" not in inputs:
                right_val = node_data.get("right_value", "0")
                try:
                    inputs["right"] = float(right_val) if "." in right_val else int(right_val)
                except ValueError:
                    inputs["right"] = right_val

            try:
                return logic_node.execute(inputs)
            except Exception as exc:
                log_error(f"Condition node {node_id} execution failed: {exc}")
                return {"result": {"value": False}}

        return {"result": {"value": False}}

    @staticmethod
    def _is_failure_result(result: Any) -> bool:
        """Return True when a node result payload encodes a failure state."""
        if not isinstance(result, dict):
            return False
        if "error" in result or result.get("status") in {"failed", "error"}:
            return True
        flow_out = result.get("flow_out")
        if isinstance(flow_out, dict) and flow_out.get("status") in {"failed", "error"}:
            return True
        return False

    @staticmethod
    def _safe_eval_condition(expr: str, results: Dict[str, Any]) -> bool:
        expr = expr.strip()
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False

        try:
            allowed_names = {"True": True, "False": False, "None": None}
            for node_id, result in results.items():
                if isinstance(result, dict):
                    for key, val in result.items():
                        if isinstance(val, dict) and "value" in val:
                            allowed_names[f"result_{node_id}_{key}"] = val["value"]
            return bool(eval(expr, {"__builtins__": {}}, allowed_names))
        except Exception:
            return False
