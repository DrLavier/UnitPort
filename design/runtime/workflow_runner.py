#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow runner for legacy graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Set

from bin.core.logger import log_info, log_debug, log_warning, log_error


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
    ) -> Dict[str, Any]:
        """Run an execution graph and return a summary dict."""
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

        def execute_node(node_id: str):
            nonlocal executed_count
            if node_id in executed:
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

            if "Logic Control" in node_name:
                self._execute_logic_control(
                    node_id=node_id,
                    node_data=node_data,
                    exec_graph=exec_graph,
                    outgoing=outgoing,
                    results=results,
                    executed=executed,
                    execute_node=execute_node,
                )
                return

            if "Condition" in node_name:
                results[node_id] = self._execute_condition_node(node_id, node_data, exec_graph, results)
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
                        results[node_id] = {"status": "success" if success else "failed", "action": action}
                    except Exception as exc:
                        log_error(f"Action execution failed: {exc}")
                        results[node_id] = {"error": str(exc)}

            flow_targets = outgoing.get("flow_out", [])
            for target_id, _ in flow_targets:
                execute_node(target_id)

        for entry_id in exec_graph.get("entry_nodes", []):
            execute_node(entry_id)

        return {
            "ok": True,
            "results": results,
            "executed_count": executed_count,
            "has_action": has_action,
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

        if ui_selection.lower().startswith("if"):
            condition_result = self._evaluate_condition(node_id, node_data, exec_graph, results)
            log_debug(f"If condition evaluated to: {condition_result}")
            branch = "out_if" if condition_result else "out_else"
            for target_id, _ in outgoing.get(branch, []):
                execute_node(target_id)
            return

        if ui_selection.lower().startswith("while"):
            loop_type = node_data.get("loop_type", "while")
            if loop_type == "for":
                try:
                    start = int(node_data.get("for_start", "0") or "0")
                    end = int(node_data.get("for_end", "10") or "10")
                    step = int(node_data.get("for_step", "1") or "1")
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
