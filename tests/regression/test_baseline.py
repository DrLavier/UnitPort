#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Regression Tests
Verifies that the current code generation produces expected output for each sample workflow.
"""

import json
import sys
import os
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must import and initialize before any Qt widgets
from tests.regression.scene_builder import (
    ensure_qapp,
    load_sample,
    build_scene_from_sample,
    get_generated_code,
    get_all_sample_paths,
)


class TestBaseline(unittest.TestCase):
    """Regression baseline tests for Canvas -> Code generation."""

    @classmethod
    def setUpClass(cls):
        """Initialize QApplication and theme manager."""
        cls.app = ensure_qapp()
        # Initialize theme manager with default config
        try:
            from bin.core.theme_manager import init_theme_manager
            ui_config = PROJECT_ROOT / "config" / "ui.ini"
            if ui_config.exists():
                init_theme_manager(str(ui_config))
        except Exception:
            pass

    def _run_sample(self, sample_path: Path):
        """Load a sample, build scene, and verify generated code."""
        sample = load_sample(str(sample_path))
        scene, editor = build_scene_from_sample(sample)
        code = get_generated_code(scene, editor)

        self.assertIsNotNone(code, f"No code generated for {sample['name']}")
        self.assertGreater(len(code.strip()), 0, f"Empty code for {sample['name']}")

        # Check expected code fragments
        for expected in sample.get("expected_code_contains", []):
            self.assertIn(
                expected, code,
                f"Expected '{expected}' in generated code for {sample['name']}.\n"
                f"Generated code:\n{code}"
            )

        return code

    def test_sample_01_single_action(self):
        """Single action node produces correct code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_01_single_action.json")

    def test_sample_02_two_actions_chained(self):
        """Two chained actions produce sequential code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_02_two_actions_chained.json")

    def test_sample_03_if_true_false(self):
        """If node with true/false branches produces if/else code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_03_if_true_false.json")

    def test_sample_04_if_elif_else(self):
        """If with elif produces if/elif/else code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_04_if_elif_else.json")

    def test_sample_05_while_loop(self):
        """While loop produces while code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_05_while_loop.json")

    def test_sample_06_for_loop(self):
        """For loop produces for-range code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_06_for_loop.json")

    def test_sample_07_comparison_to_if(self):
        """Comparison connected to If produces comparison + if code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_07_comparison_to_if.json")

    def test_sample_08_timer_in_flow(self):
        """Timer between actions produces sleep code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_08_timer_in_flow.json")

    def test_sample_09_sensor_standalone(self):
        """Sensor node produces sensor code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_09_sensor_standalone.json")

    def test_sample_10_complex_workflow(self):
        """Complex workflow produces correct nested code."""
        samples_dir = PROJECT_ROOT / "tests" / "regression" / "samples"
        self._run_sample(samples_dir / "sample_10_complex_workflow.json")

    def test_all_samples_loadable(self):
        """All sample files are valid JSON and can be loaded."""
        for path in get_all_sample_paths():
            with self.subTest(sample=path.name):
                sample = load_sample(str(path))
                self.assertIn("nodes", sample)
                self.assertIn("connections", sample)
                self.assertIn("name", sample)


if __name__ == "__main__":
    unittest.main()
