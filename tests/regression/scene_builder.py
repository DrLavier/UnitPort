#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene Builder Utility
Constructs a GraphScene from a workflow JSON dict for regression testing.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def ensure_qapp():
    """Ensure a QApplication exists (required for QWidget creation)."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def load_sample(sample_path: str) -> Dict[str, Any]:
    """Load a sample workflow JSON file."""
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_scene_from_sample(sample_data: Dict[str, Any]):
    """
    Build a GraphScene from a sample workflow dict.

    Args:
        sample_data: Workflow dict (as from sample JSON files).

    Returns:
        (scene, code_editor) tuple.
    """
    from bin.components.graph_scene import GraphScene
    from bin.components.code_editor import CodeEditor

    scene = GraphScene()
    editor = CodeEditor()
    scene.set_code_editor(editor)

    if "robot_type" in sample_data:
        scene.set_robot_type(sample_data["robot_type"])

    scene.load_workflow(sample_data)

    return scene, editor


def get_generated_code(scene, editor) -> str:
    """Get the currently generated code from the editor."""
    return editor.get_code()


def get_all_sample_paths():
    """Get paths to all regression sample JSON files."""
    samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
    return sorted(samples_dir.glob("sample_*.json"))
