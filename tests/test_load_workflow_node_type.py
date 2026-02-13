#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for GraphScene.load_workflow node_type restoration."""

import sys
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.components.graph_scene import GraphScene
from bin.components.code_editor import CodeEditor


def ensure_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestLoadWorkflowNodeType(unittest.TestCase):
    def setUp(self):
        ensure_qapp()
        self.scene = GraphScene()
        self.editor = CodeEditor()
        self.scene.set_code_editor(self.editor)
        self.editor.set_graph_scene(self.scene)

    def test_if_node_type_takes_priority_over_display_name(self):
        # Intentionally conflicting display_name to verify node_type-driven restore.
        data = {
            "robot_type": "go2",
            "nodes": [
                {
                    "id": 0,
                    "display_name": "While Loop",
                    "node_type": "if",
                    "position": {"x": 100, "y": 100},
                    "condition_expr": "x > 0",
                    "elif_conditions": ["x == 0"],
                }
            ],
            "connections": [],
        }

        self.scene.load_workflow(data)

        self.assertIn(0, self.scene._logic_nodes)
        logic_node = self.scene._logic_nodes[0]
        self.assertEqual(logic_node.node_type, "if")
        self.assertEqual(logic_node.parameters.get("elif_conditions"), ["x == 0"])


if __name__ == "__main__":
    unittest.main()

