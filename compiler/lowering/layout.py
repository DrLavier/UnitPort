#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-layout algorithm for placing IR nodes on the canvas.
Simple left-to-right layered layout (Sugiyama-inspired).
Nodes are centered vertically within each layer.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from compiler.ir.workflow_ir import WorkflowIR, IRNode, IRNodeUI


# Layout constants
NODE_WIDTH = 180
NODE_HEIGHT = 110
LOGIC_WIDTH = 240
LOGIC_HEIGHT = 200
H_GAP = 100       # Horizontal gap between layers
V_GAP = 50        # Vertical gap between nodes in same layer
CANVAS_CENTER_X = 600   # Center X of the canvas viewport
CANVAS_CENTER_Y = 400   # Center Y of the canvas viewport


class LayoutEngine:
    """Compute x, y positions for IR nodes using layered layout."""

    def layout(self, ir: WorkflowIR):
        """
        Assign positions to all IR nodes in the IR.
        Modifies node.ui in place.
        Nodes are centered on the canvas.
        """
        if not ir.nodes:
            return

        # Build adjacency
        outgoing: Dict[str, List[str]] = {n.id: [] for n in ir.nodes}
        incoming: Dict[str, List[str]] = {n.id: [] for n in ir.nodes}
        for edge in ir.edges:
            if edge.from_node in outgoing:
                outgoing[edge.from_node].append(edge.to_node)
            if edge.to_node in incoming:
                incoming[edge.to_node].append(edge.from_node)

        # Assign layers using topological ordering (longest path)
        layers = self._assign_layers(ir, outgoing, incoming)

        # Group nodes by layer
        layer_groups: Dict[int, List[IRNode]] = {}
        for node in ir.nodes:
            layer = layers.get(node.id, 0)
            layer_groups.setdefault(layer, []).append(node)

        # Calculate total dimensions for centering
        max_layer = max(layer_groups.keys()) if layer_groups else 0
        num_layers = max_layer + 1

        # Calculate total width
        total_width = 0
        for layer_idx in range(num_layers):
            nodes_in_layer = layer_groups.get(layer_idx, [])
            max_w = max((self._node_size(n)[0] for n in nodes_in_layer),
                        default=NODE_WIDTH)
            total_width += max_w
        total_width += H_GAP * max(0, num_layers - 1)

        # Calculate max total height across all layers
        max_total_height = 0
        for layer_idx in range(num_layers):
            nodes_in_layer = layer_groups.get(layer_idx, [])
            layer_height = sum(self._node_size(n)[1] for n in nodes_in_layer)
            layer_height += V_GAP * max(0, len(nodes_in_layer) - 1)
            max_total_height = max(max_total_height, layer_height)

        # Starting X to center horizontally
        start_x = CANVAS_CENTER_X - total_width / 2

        # Position nodes layer by layer
        current_x = start_x
        for layer_idx in range(num_layers):
            nodes_in_layer = layer_groups.get(layer_idx, [])
            if not nodes_in_layer:
                continue

            # Layer width = max node width in this layer
            layer_max_w = max(self._node_size(n)[0] for n in nodes_in_layer)

            # Total height of this layer
            layer_height = sum(self._node_size(n)[1] for n in nodes_in_layer)
            layer_height += V_GAP * max(0, len(nodes_in_layer) - 1)

            # Starting Y to center vertically
            start_y = CANVAS_CENTER_Y - layer_height / 2

            current_y = start_y
            for node in nodes_in_layer:
                w, h = self._node_size(node)
                # Center node within layer column
                x = current_x + (layer_max_w - w) / 2
                y = current_y

                if node.ui is None:
                    node.ui = IRNodeUI(x=x, y=y, width=w, height=h)
                else:
                    node.ui.x = x
                    node.ui.y = y
                    node.ui.width = w
                    node.ui.height = h

                current_y += h + V_GAP

            current_x += layer_max_w + H_GAP

    def _assign_layers(self, ir: WorkflowIR,
                       outgoing: Dict[str, List[str]],
                       incoming: Dict[str, List[str]]) -> Dict[str, int]:
        """Assign layer numbers using longest-path from entry nodes."""
        layers: Dict[str, int] = {}

        # Find entry nodes (no incoming edges)
        entry_ids = [n.id for n in ir.nodes if not incoming.get(n.id)]
        if not entry_ids:
            entry_ids = [ir.nodes[0].id] if ir.nodes else []

        # BFS-based longest path
        def assign(node_id: str, layer: int):
            if node_id in layers and layers[node_id] >= layer:
                return
            layers[node_id] = layer
            for target in outgoing.get(node_id, []):
                assign(target, layer + 1)

        for eid in entry_ids:
            assign(eid, 0)

        # Assign unvisited nodes to layer 0
        for node in ir.nodes:
            if node.id not in layers:
                layers[node.id] = 0

        return layers

    @staticmethod
    def _node_size(node: IRNode) -> Tuple[int, int]:
        """Get the width/height for a node based on its kind."""
        from compiler.ir.workflow_ir import NodeKind
        if node.kind == NodeKind.LOGIC:
            return LOGIC_WIDTH, LOGIC_HEIGHT
        if node.kind == NodeKind.COMPARISON:
            return 260, 170
        return NODE_WIDTH, NODE_HEIGHT
