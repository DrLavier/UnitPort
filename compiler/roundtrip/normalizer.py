#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR normalizer for round-trip comparison.

Strips UI metadata, canonicalizes IDs, sorts edges,
and compares two IRs for semantic equivalence.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, IRParam, IRNodeUI, NodeKind, EdgeType,
)


class IRNormalizer:
    """Normalize and compare WorkflowIR instances."""

    def normalize(self, ir: WorkflowIR) -> WorkflowIR:
        """
        Create a normalized copy of the IR.

        Normalization steps:
        1. Strip UI metadata (positions, dimensions)
        2. Topologically sort nodes
        3. Reassign sequential IDs (0, 1, 2, ...)
        4. Sort edges by (from_node, from_port, to_node, to_port)
        5. Normalize parameter types (string numbers -> actual numbers)
        """
        normalized = WorkflowIR(
            robot_type=ir.robot_type,
            brand=ir.brand,
        )

        # Topological sort
        sorted_nodes = self._topo_sort(ir)

        # Build old-to-new ID mapping
        id_map: Dict[str, str] = {}
        for idx, node in enumerate(sorted_nodes):
            id_map[node.id] = str(idx)

        # Create normalized nodes
        for idx, node in enumerate(sorted_nodes):
            norm_node = IRNode(
                id=str(idx),
                schema_id=node.schema_id,
                kind=node.kind,
                params=self._normalize_params(node.params),
                ui=None,  # Strip UI
                opaque_code=node.opaque_code,
            )
            normalized.add_node(norm_node)

        # Create normalized edges with remapped IDs
        norm_edges = []
        for edge in ir.edges:
            from_id = id_map.get(edge.from_node)
            to_id = id_map.get(edge.to_node)
            if from_id is not None and to_id is not None:
                norm_edges.append(IREdge(
                    from_node=from_id,
                    from_port=edge.from_port,
                    to_node=to_id,
                    to_port=edge.to_port,
                    edge_type=edge.edge_type,
                ))

        # Sort edges deterministically
        norm_edges.sort(key=lambda e: (e.from_node, e.from_port,
                                        e.to_node, e.to_port))
        for edge in norm_edges:
            normalized.add_edge(edge)

        return normalized

    def compare(self, ir_a: WorkflowIR, ir_b: WorkflowIR) -> float:
        """
        Compare two IRs and return an equivalence score (0.0 to 1.0).

        1.0 = perfect match
        0.0 = completely different
        """
        norm_a = self.normalize(ir_a)
        norm_b = self.normalize(ir_b)

        if not norm_a.nodes and not norm_b.nodes:
            return 1.0
        if not norm_a.nodes or not norm_b.nodes:
            return 0.0

        # Compare nodes
        node_score = self._compare_nodes(norm_a.nodes, norm_b.nodes)

        # Compare edges
        edge_score = self._compare_edges(norm_a.edges, norm_b.edges)

        # Weighted average: nodes are more important than edges
        return 0.7 * node_score + 0.3 * edge_score

    def _compare_nodes(self, nodes_a: List[IRNode],
                       nodes_b: List[IRNode]) -> float:
        """Compare two node lists and return similarity score."""
        max_len = max(len(nodes_a), len(nodes_b))
        if max_len == 0:
            return 1.0

        matches = 0
        min_len = min(len(nodes_a), len(nodes_b))

        for i in range(min_len):
            na, nb = nodes_a[i], nodes_b[i]
            if na.kind == nb.kind and na.schema_id == nb.schema_id:
                param_sim = self._compare_params(na.params, nb.params)
                matches += param_sim

        return matches / max_len

    def _compare_edges(self, edges_a: List[IREdge],
                       edges_b: List[IREdge]) -> float:
        """Compare two edge lists and return similarity score."""
        max_len = max(len(edges_a), len(edges_b))
        if max_len == 0:
            return 1.0

        # Convert to sets of tuples for comparison
        set_a = {(e.from_node, e.from_port, e.to_node, e.to_port) for e in edges_a}
        set_b = {(e.from_node, e.from_port, e.to_node, e.to_port) for e in edges_b}

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 1.0

    def _compare_params(self, params_a: Dict[str, IRParam],
                        params_b: Dict[str, IRParam]) -> float:
        """Compare two param dicts and return similarity score."""
        all_keys = set(params_a.keys()) | set(params_b.keys())
        if not all_keys:
            return 1.0

        matches = 0
        for key in all_keys:
            if key in params_a and key in params_b:
                va = self._normalize_value(params_a[key].value)
                vb = self._normalize_value(params_b[key].value)
                if va == vb:
                    matches += 1
                elif str(va) == str(vb):
                    matches += 0.8  # Close match
            # Missing key counts as 0

        return matches / len(all_keys)

    def _normalize_params(self, params: Dict[str, IRParam]) -> Dict[str, IRParam]:
        """Normalize parameter values."""
        normalized = {}
        for key, param in params.items():
            value = self._normalize_value(param.value)
            normalized[key] = IRParam(param.name, value, param.param_type)
        return normalized

    @staticmethod
    def _normalize_value(value):
        """Normalize a value for comparison."""
        if isinstance(value, str):
            # Try converting numeric strings
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except (ValueError, TypeError):
                pass
            return value.lower().strip()
        return value

    def _topo_sort(self, ir: WorkflowIR) -> List[IRNode]:
        """Topologically sort nodes. Falls back to kind+schema sort."""
        # Build adjacency
        outgoing: Dict[str, List[str]] = {n.id: [] for n in ir.nodes}
        in_degree: Dict[str, int] = {n.id: 0 for n in ir.nodes}
        node_map = {n.id: n for n in ir.nodes}

        for edge in ir.edges:
            if edge.from_node in outgoing and edge.to_node in in_degree:
                outgoing[edge.from_node].append(edge.to_node)
                in_degree[edge.to_node] += 1

        # Kahn's algorithm
        queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            nid = queue.pop(0)
            result.append(node_map[nid])
            for target in sorted(outgoing.get(nid, [])):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
                    queue.sort()

        # Add any remaining nodes (cycles or disconnected)
        visited = {n.id for n in result}
        for node in ir.nodes:
            if node.id not in visited:
                result.append(node)

        return result
