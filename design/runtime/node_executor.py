#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime node executor."""

from typing import Dict, Any, Optional
from bin.core.logger import log_info, log_error, log_debug, log_warning


class NodeExecutor:
    """Execute a simple DAG-like node graph."""

    def __init__(self):
        self.nodes = []
        self.connections = []
        self.execution_order = []

    def add_node(self, node_id: str, node_type: str, node_data: Dict[str, Any]):
        node = {
            "id": node_id,
            "type": node_type,
            "data": node_data,
            "inputs": [],
            "outputs": [],
        }
        self.nodes.append(node)
        log_debug(f"add node: {node_id} ({node_type})")

    def add_connection(self, from_node: str, from_port: str, to_node: str, to_port: str):
        connection = {
            "from": {"node": from_node, "port": from_port},
            "to": {"node": to_node, "port": to_port},
        }
        self.connections.append(connection)
        log_debug(f"add connection: {from_node}.{from_port} -> {to_node}.{to_port}")

    def build_execution_order(self) -> bool:
        graph = {node["id"]: [] for node in self.nodes}
        in_degree = {node["id"]: 0 for node in self.nodes}

        for conn in self.connections:
            from_id = conn["from"]["node"]
            to_id = conn["to"]["node"]
            if from_id in graph and to_id in graph:
                graph[from_id].append(to_id)
                in_degree[to_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            log_error("cycle detected in node graph")
            return False

        self.execution_order = result
        log_info(f"execution order: {' -> '.join(result)}")
        return True

    def execute(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.execution_order:
            if not self.build_execution_order():
                raise RuntimeError("failed to build execution order")

        context = context or {}
        results: Dict[str, Any] = {}

        for node_id in self.execution_order:
            node = self._find_node(node_id)
            if not node:
                log_warning(f"node not found: {node_id}")
                continue

            log_info(f"execute node: {node_id} ({node['type']})")
            inputs = self._collect_inputs(node_id, results)
            try:
                output = self._execute_node(node, inputs, context)
                results[node_id] = output
                log_debug(f"node {node_id} output: {output}")
            except Exception as exc:
                log_error(f"node {node_id} failed: {exc}")
                results[node_id] = {"error": str(exc)}

        return results

    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None

    def _collect_inputs(self, node_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for conn in self.connections:
            if conn["to"]["node"] == node_id:
                from_id = conn["from"]["node"]
                from_port = conn["from"]["port"]
                to_port = conn["to"]["port"]
                if from_id in results:
                    inputs[to_port] = results[from_id].get(from_port)
        return inputs

    def _execute_node(self, node: Dict[str, Any], inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "output": f"Executed {node['type']}",
            "data": node["data"],
        }

    def clear(self):
        self.nodes.clear()
        self.connections.clear()
        self.execution_order.clear()
        log_info("node executor cleared")

    def to_code(self) -> str:
        if not self.execution_order:
            self.build_execution_order()

        code_lines = [
            "# Auto-generated workflow code",
            "",
            "def execute_workflow():",
        ]
        for node_id in self.execution_order:
            node = self._find_node(node_id)
            if node:
                code_lines.append(f"    # {node['type']}: {node_id}")
                code_lines.append(f"    # TODO: implement {node['type']}")
                code_lines.append("")

        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    execute_workflow()")
        return "\n".join(code_lines)

