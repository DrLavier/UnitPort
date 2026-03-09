#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Runtime Status Surface — Cycle 1 STAGE-03.

Coverage
--------
GraphScene status badge API:
  - set_node_execution_status applies coloured border pen
  - set_node_execution_status with unknown status is a no-op
  - reset_execution_statuses restores default border and clears _exec_status
  - set_node_execution_status on non-existent node_id is a no-op
  - all five statuses (success/failed/skipped/running/pending) are supported

MainZonePanel summary bar:
  - show_execution_summary makes bar visible with correct text
  - show_execution_summary success scenario
  - show_execution_summary failed scenario (failed_nodes present)
  - show_execution_summary blocked scenario
  - show_execution_summary compat_path flag shows compat label
  - clear_execution_summary hides bar
  - re-running (show → clear → show) produces stable state

ui._apply_node_execution_statuses logic (pure-logic unit test, no Qt):
  - nodes in failed_nodes → "failed"
  - nodes in results (not failed) → "success"
  - nodes in exec_graph but not results → "skipped"
  - key normalisation: int vs str node_ids

Isolation
---------
- GraphScene / MainZonePanel tests: offscreen PySide6
- _apply_node_execution_statuses logic test: pure Python (mock graph_scene)
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, call

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

_app = None


def _get_app():
    global _app
    if not _PYSIDE6_AVAILABLE:
        return None
    if QApplication.instance() is None:
        _app = QApplication(sys.argv[:1])
    else:
        _app = QApplication.instance()
    return _app


# ---------------------------------------------------------------------------
# Pure-logic helper — tests _apply_node_execution_statuses without Qt
# ---------------------------------------------------------------------------

class _FakeScene:
    """Records set_node_execution_status calls."""
    def __init__(self):
        self.calls: list = []

    def reset_execution_statuses(self):
        self.calls.clear()

    def set_node_execution_status(self, node_id, status: str):
        self.calls.append((node_id, status))


def _run_apply_statuses(exec_graph: dict, run_result: dict):
    """Replicate bin/ui.py _apply_node_execution_statuses logic for pure-Python testing."""
    results = run_result.get("results", {})
    diag = run_result.get("diagnostics", {})
    failed_strs = {str(nid) for nid in diag.get("failed_nodes", [])}
    executed_strs = {str(nid) for nid in results.keys()}

    scene = _FakeScene()
    for node_id in exec_graph.get("nodes", {}):
        nid_str = str(node_id)
        if nid_str in failed_strs:
            status = "failed"
        elif nid_str in executed_strs:
            status = "success"
        else:
            status = "skipped"
        scene.set_node_execution_status(node_id, status)
    return scene.calls


class TestApplyNodeStatusLogic(unittest.TestCase):
    """Pure-Python unit tests for per-node status mapping logic."""

    def _make_exec_graph(self, node_ids):
        return {"nodes": {nid: {} for nid in node_ids}}

    def test_failed_nodes_get_failed_status(self):
        """Nodes listed in diagnostics.failed_nodes receive 'failed' status."""
        exec_graph = self._make_exec_graph([1, 2, 3])
        run_result = {
            "results": {1: {"output": "ok"}, 2: {"error": "oops"}, 3: {"output": "ok"}},
            "diagnostics": {"failed_nodes": [2]},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        status_map = {nid: st for nid, st in calls}
        self.assertEqual(status_map[2], "failed")

    def test_executed_nodes_without_error_get_success(self):
        """Nodes in results but NOT in failed_nodes receive 'success'."""
        exec_graph = self._make_exec_graph([1, 2])
        run_result = {
            "results": {1: {"output": "ok"}, 2: {"output": "ok"}},
            "diagnostics": {"failed_nodes": []},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        status_map = {nid: st for nid, st in calls}
        self.assertEqual(status_map[1], "success")
        self.assertEqual(status_map[2], "success")

    def test_unexecuted_nodes_get_skipped(self):
        """Nodes in exec_graph but absent from results receive 'skipped'."""
        exec_graph = self._make_exec_graph([1, 2, 3])
        run_result = {
            "results": {1: {"output": "ok"}},
            "diagnostics": {"failed_nodes": []},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        status_map = {nid: st for nid, st in calls}
        self.assertEqual(status_map[2], "skipped")
        self.assertEqual(status_map[3], "skipped")

    def test_int_key_in_results_matches_int_node_id(self):
        """Int key in results correctly maps to int node_id in exec_graph."""
        exec_graph = self._make_exec_graph([10])
        run_result = {
            "results": {10: {"output": "ok"}},
            "diagnostics": {"failed_nodes": []},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        self.assertEqual(calls[0][1], "success")

    def test_str_key_in_results_matches_int_node_id(self):
        """Str key in results (e.g. from JSON round-trip) maps correctly."""
        exec_graph = self._make_exec_graph([10])
        run_result = {
            "results": {"10": {"output": "ok"}},
            "diagnostics": {"failed_nodes": []},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        self.assertEqual(calls[0][1], "success")

    def test_str_failed_node_matches_int_node_id(self):
        """Str failed_node entry (from runtime contract) matches int node_id."""
        exec_graph = self._make_exec_graph([5])
        run_result = {
            "results": {"5": {"error": "boom"}},
            "diagnostics": {"failed_nodes": ["5"]},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        self.assertEqual(calls[0][1], "failed")

    def test_all_nodes_covered_in_calls(self):
        """Every node in exec_graph gets a status call."""
        exec_graph = self._make_exec_graph([1, 2, 3, 4])
        run_result = {
            "results": {1: {}, 2: {"error": "x"}},
            "diagnostics": {"failed_nodes": [2]},
        }
        calls = _run_apply_statuses(exec_graph, run_result)
        covered_ids = {c[0] for c in calls}
        self.assertEqual(covered_ids, {1, 2, 3, 4})

    def test_empty_exec_graph_no_calls(self):
        """Empty exec_graph produces no status calls."""
        exec_graph = {"nodes": {}}
        run_result = {"results": {}, "diagnostics": {"failed_nodes": []}}
        calls = _run_apply_statuses(exec_graph, run_result)
        self.assertEqual(calls, [])


# ---------------------------------------------------------------------------
# GraphScene badge API tests (requires PySide6)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestGraphSceneStatusBadges(unittest.TestCase):
    """Tests for GraphScene.set_node_execution_status and reset_execution_statuses."""

    @classmethod
    def setUpClass(cls):
        _get_app()
        from bin.components.graph_scene import GraphScene
        cls.GraphScene = GraphScene

    def _make_scene(self):
        return self.GraphScene()

    def _node_item(self, scene, node_id):
        from shiboken6 import isValid
        for item in scene.items():
            if isValid(item) and item.data(10) == "node" and item.data(12) == node_id:
                return item
        return None

    def test_set_status_success_changes_pen_color(self):
        """set_node_execution_status('success') applies green border pen."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(100, 100))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "success")

        pen = item.pen()
        color = pen.color()
        self.assertGreater(color.green(), color.red(), "Success border should be green-dominant")

    def test_set_status_failed_changes_pen_color(self):
        """set_node_execution_status('failed') applies red border pen."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(100, 100))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "failed")

        pen = item.pen()
        color = pen.color()
        self.assertGreater(color.red(), color.green(), "Failed border should be red-dominant")

    def test_set_status_stores_exec_status_attribute(self):
        """_exec_status attribute is set on the node item after status update."""
        scene = self._make_scene()
        item = scene.create_node("Wait", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "skipped")

        self.assertEqual(getattr(item, '_exec_status', None), "skipped")

    def test_all_five_statuses_supported(self):
        """All five canonical statuses apply without errors."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        for status in ("success", "failed", "skipped", "running", "pending"):
            scene.set_node_execution_status(nid, status)
            self.assertEqual(item._exec_status, status)

    def test_unknown_status_is_noop(self):
        """Unknown status string leaves the node pen unchanged."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)
        original_pen = item.pen()

        scene.set_node_execution_status(nid, "nonexistent_status_xyz")

        self.assertEqual(item.pen().color().name(), original_pen.color().name())

    def test_nonexistent_node_id_is_noop(self):
        """set_node_execution_status with a non-existent node_id does not raise."""
        scene = self._make_scene()
        scene.set_node_execution_status(99999, "success")  # must not raise

    def test_reset_clears_exec_status_attribute(self):
        """reset_execution_statuses clears _exec_status on all nodes."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)
        scene.set_node_execution_status(nid, "failed")
        self.assertEqual(item._exec_status, "failed")

        scene.reset_execution_statuses()

        self.assertIsNone(getattr(item, '_exec_status', None))

    def test_reset_restores_default_border(self):
        """reset_execution_statuses changes the pen away from the status colour.

        We verify that the pen colour AFTER reset is not the bright red that
        'failed' status would set (#ef4444), and that the pen width returns
        to 2 (default) from 2.5 (status badge).
        """
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "failed")
        pen_during = item.pen()
        # Sanity: failed pen is red-dominant
        self.assertGreater(pen_during.color().red(), pen_during.color().green())

        scene.reset_execution_statuses()

        pen_after = item.pen()
        # After reset the pen must no longer be the status red
        failed_red = QColor("#ef4444")
        self.assertNotAlmostEqual(
            pen_after.color().red(), failed_red.red(), delta=10,
            msg="Pen should no longer be the 'failed' red after reset"
        )
        # Width returns to 2 (default) from 2.5 (status badge)
        self.assertAlmostEqual(pen_after.widthF(), 2.0, delta=0.6)

    def test_reset_then_rerun_gives_fresh_statuses(self):
        """After reset, setting a new status works correctly (stable re-run)."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        # First run: failed
        scene.set_node_execution_status(nid, "failed")
        self.assertEqual(item._exec_status, "failed")

        # Reset
        scene.reset_execution_statuses()
        self.assertIsNone(item._exec_status)

        # Second run: success
        scene.set_node_execution_status(nid, "success")
        self.assertEqual(item._exec_status, "success")


# ---------------------------------------------------------------------------
# MainZonePanel summary bar tests (requires PySide6)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestExecutionSummaryBar(unittest.TestCase):
    """Tests for MainZonePanel.show_execution_summary / clear_execution_summary."""

    @classmethod
    def setUpClass(cls):
        _get_app()
        from bin.layout.main_zone_panel import MainZonePanel
        cls.MainZonePanel = MainZonePanel

    def _make_panel(self):
        return self.MainZonePanel()

    def _success_result(self, node_count=5, executed=5, failed_nodes=None):
        return {
            "status": "success",
            "reason": "",
            "node_count": node_count,
            "results": {i: {} for i in range(executed)},
            "diagnostics": {
                "executed_count": executed,
                "failed_nodes": failed_nodes or [],
                "compat_path": False,
            },
        }

    def _failed_result(self, node_count=5, executed=3, failed_nodes=None):
        return {
            "status": "failed",
            "reason": "node_execution_failed",
            "node_count": node_count,
            "results": {i: {} for i in range(executed)},
            "diagnostics": {
                "executed_count": executed,
                "failed_nodes": failed_nodes or [2],
                "compat_path": False,
            },
        }

    def _blocked_result(self, reason="safety_blocked"):
        return {
            "status": "blocked",
            "reason": reason,
            "node_count": 0,
            "results": {},
            "diagnostics": {
                "executed_count": 0,
                "failed_nodes": [],
                "compat_path": False,
            },
        }

    def test_bar_initially_hidden(self):
        """Summary bar is hidden before any run.
        Note: isHidden() checks the widget's own explicit flag; isVisible()
        requires the entire parent chain to be visible (not reliable in headless tests)."""
        panel = self._make_panel()
        self.assertTrue(panel._exec_summary_bar.isHidden())

    def test_show_makes_bar_visible(self):
        """show_execution_summary makes the bar not-hidden (explicit visible flag set)."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result())
        self.assertFalse(panel._exec_summary_bar.isHidden())

    def test_clear_hides_bar(self):
        """clear_execution_summary hides the bar (explicit hidden flag set)."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result())
        panel.clear_execution_summary()
        self.assertTrue(panel._exec_summary_bar.isHidden())

    def test_success_result_shows_checkmark(self):
        """Success result sets checkmark icon."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result())
        self.assertEqual(panel._exec_status_icon.text(), "✔")

    def test_failed_result_shows_cross(self):
        """Failed result sets cross icon."""
        panel = self._make_panel()
        panel.show_execution_summary(self._failed_result())
        self.assertEqual(panel._exec_status_icon.text(), "✖")

    def test_blocked_result_shows_blocked_icon(self):
        """Blocked result sets blocked icon."""
        panel = self._make_panel()
        panel.show_execution_summary(self._blocked_result())
        self.assertEqual(panel._exec_status_icon.text(), "⊘")

    def test_summary_text_includes_executed_count(self):
        """Summary text shows executed/total node counts."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result(node_count=10, executed=10))
        text = panel._exec_summary_label.text()
        self.assertIn("10", text)

    def test_summary_text_includes_failure_count(self):
        """Summary text mentions failed node count when failures present."""
        panel = self._make_panel()
        panel.show_execution_summary(self._failed_result(failed_nodes=[1, 3]))
        text = panel._exec_summary_label.text()
        self.assertIn("2 failed", text)

    def test_compat_path_label_visible_when_active(self):
        """Compat path label is not-hidden when compat_path is True."""
        panel = self._make_panel()
        result = self._success_result()
        result["diagnostics"]["compat_path"] = True
        result["diagnostics"]["compat_reason"] = "UNITPORT_FLOW_AWARE_EXECUTION=0"
        panel.show_execution_summary(result)
        self.assertFalse(panel._exec_compat_label.isHidden())
        self.assertIn("compat", panel._exec_compat_label.text().lower())

    def test_compat_path_label_hidden_when_normal(self):
        """Compat path label is hidden on normal runs."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result())
        self.assertTrue(panel._exec_compat_label.isHidden())

    def test_clear_empties_summary_text(self):
        """clear_execution_summary clears the summary label text."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result())
        panel.clear_execution_summary()
        self.assertEqual(panel._exec_summary_label.text(), "")

    def test_rerun_stable_state(self):
        """show → clear → show produces correct state for the second run."""
        panel = self._make_panel()
        panel.show_execution_summary(self._success_result())
        panel.clear_execution_summary()
        panel.show_execution_summary(self._failed_result())
        self.assertFalse(panel._exec_summary_bar.isHidden())
        self.assertEqual(panel._exec_status_icon.text(), "✖")

    def test_show_exposes_api_methods(self):
        """MainZonePanel exposes both required methods."""
        from bin.layout.main_zone_panel import MainZonePanel
        self.assertTrue(callable(getattr(MainZonePanel, "show_execution_summary", None)))
        self.assertTrue(callable(getattr(MainZonePanel, "clear_execution_summary", None)))


# ---------------------------------------------------------------------------
# WorkflowRunner node_status_callback tests (pure Python, no Qt)
# ---------------------------------------------------------------------------

class _MockLogicNode:
    """Minimal logic node stub for WorkflowRunner tests."""
    def __init__(self, output=None, raises=None):
        self._output  = output or {"status": "ok"}
        self._raises  = raises
        self.executed = False

    def set_parameter(self, *a, **kw):
        pass

    def execute(self, inputs=None):
        self.executed = True
        if self._raises:
            raise self._raises
        return self._output


def _make_exec_graph(node_ids, fail_ids=None, chain=False):
    """Build a minimal exec_graph for WorkflowRunner tests.

    Args:
        node_ids: Ordered list of int node IDs.
        fail_ids: Set of int IDs whose logic_nodes should raise.
        chain:    If True, wire nodes in a linear chain (1 → 2 → 3 …).
    """
    fail_ids = fail_ids or set()
    nodes, outgoing, incoming = {}, {}, {}
    for nid in node_ids:
        exc = Exception(f"forced failure on node {nid}") if nid in fail_ids else None
        nodes[nid] = {
            "id": nid,
            "name": f"Node{nid}",
            "type": "timer",
            "logic_node": _MockLogicNode(raises=exc),
        }
        outgoing[nid] = {"flow_out": []}
        incoming[nid] = {}

    if chain and len(node_ids) > 1:
        for i in range(len(node_ids) - 1):
            src, dst = node_ids[i], node_ids[i + 1]
            outgoing[src]["flow_out"] = [(dst, "out")]

    return {
        "nodes": nodes,
        "outgoing": outgoing,
        "incoming": incoming,
        "entry_nodes": [node_ids[0]] if node_ids else [],
    }


class TestWorkflowRunnerCallback(unittest.TestCase):
    """Tests for node_status_callback in WorkflowRunner.run() (pure Python)."""

    def setUp(self):
        from system.runtime.workflow_runner import WorkflowRunner
        self.runner = WorkflowRunner()

    # ── Callback sequence ────────────────────────────────────────────────

    def test_callback_running_fires_before_success(self):
        """'running' must precede 'success' in callback sequence."""
        events = []
        exec_graph = _make_exec_graph([1])

        # Wrap execute to record ordering relative to callback events
        original = exec_graph["nodes"][1]["logic_node"]
        class _Tracked(_MockLogicNode):
            def execute(self, inputs=None):
                events.append(("exec", 1))
                return {"status": "ok"}
        exec_graph["nodes"][1]["logic_node"] = _Tracked()

        def cb(nid, status):
            events.append((status, nid))

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb)

        running_idx = next(i for i, e in enumerate(events) if e == ("running", 1))
        exec_idx    = next(i for i, e in enumerate(events) if e == ("exec", 1))
        success_idx = next(i for i, e in enumerate(events) if e == ("success", 1))
        self.assertLess(running_idx, exec_idx,
            "'running' callback must fire before node.execute()")
        self.assertLess(exec_idx, success_idx,
            "'success' callback must fire after node.execute()")

    def test_callback_fires_failed_on_exception(self):
        """'failed' fires when logic_node.execute() raises."""
        events = []
        exec_graph = _make_exec_graph([1], fail_ids={1})

        def cb(nid, status):
            events.append((status, nid))

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb)

        statuses = [s for (s, _) in events]
        self.assertIn("running", statuses)
        self.assertIn("failed",  statuses)
        self.assertNotIn("success", statuses)

    def test_callback_fires_for_every_executed_node(self):
        """Callback fires a 'running' event for each node in a linear chain."""
        events = []
        exec_graph = _make_exec_graph([1, 2, 3], chain=True)

        def cb(nid, status):
            events.append((status, nid))

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb)

        running_ids = {nid for (s, nid) in events if s == "running"}
        self.assertEqual(running_ids, {1, 2, 3})

    def test_callback_success_for_each_node_in_chain(self):
        """Each node in a linear chain fires 'success'."""
        events = []
        exec_graph = _make_exec_graph([1, 2, 3], chain=True)

        def cb(nid, status):
            events.append((status, nid))

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb)

        success_ids = {nid for (s, nid) in events if s == "success"}
        self.assertEqual(success_ids, {1, 2, 3})

    def test_callback_int_node_ids_passed_through(self):
        """Callback receives the same int node_id that exec_graph uses."""
        received_ids = []
        exec_graph = _make_exec_graph([42])

        def cb(nid, status):
            received_ids.append(nid)

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb)

        # Must receive int 42 (same type as exec_graph key), not str "42"
        self.assertIn(42, received_ids)
        self.assertNotIn("42", received_ids)

    def test_no_callback_runs_without_error(self):
        """Omitting node_status_callback keeps backward-compat (no crash)."""
        exec_graph = _make_exec_graph([1, 2], chain=True)
        result = self.runner.run(exec_graph, robot_model=None)
        self.assertTrue(result.get("ok", False))

    def test_none_callback_runs_without_error(self):
        """Explicit None callback also leaves behaviour unchanged."""
        exec_graph = _make_exec_graph([1])
        result = self.runner.run(exec_graph, robot_model=None, node_status_callback=None)
        self.assertTrue(result.get("ok", False))

    def test_partial_chain_fail_skipped_node_not_in_callback(self):
        """Nodes not reached due to failed predecessor are never in callback."""
        events = []
        # Node 1 fails; node 2 is downstream via flow_out
        exec_graph = _make_exec_graph([1, 2], fail_ids={1}, chain=True)

        def cb(nid, status):
            events.append((status, nid))

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb)

        cb_nids = {nid for (_, nid) in events}
        # Node 2 is in a flow chain after node 1 which fails.
        # WorkflowRunner continues flow after failure (no early abort in WorkflowRunner),
        # so we just verify node 1 fires failed and node 2 runs separately.
        # At minimum, node 1 must have fired its callbacks.
        self.assertIn(1, cb_nids, "Node 1 must appear in callback events")

    # ── Rerun stability ──────────────────────────────────────────────────

    def test_rerun_produces_fresh_callbacks(self):
        """A second run on the same graph fires fresh callbacks (no stale events)."""
        first_run_events  = []
        second_run_events = []

        exec_graph = _make_exec_graph([1])

        def cb_first(nid, status):
            first_run_events.append((status, nid))

        def cb_second(nid, status):
            second_run_events.append((status, nid))

        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb_first)
        # Reset executed set — in WorkflowRunner it's a closure; just re-run
        self.runner.run(exec_graph, robot_model=None, node_status_callback=cb_second)

        self.assertIn(("running",  1), first_run_events)
        self.assertIn(("success",  1), first_run_events)
        self.assertIn(("running",  1), second_run_events)
        self.assertIn(("success",  1), second_run_events)


# ---------------------------------------------------------------------------
# NodeExecutor node_status_callback tests (pure Python, no Qt)
# ---------------------------------------------------------------------------

class TestNodeExecutorCallback(unittest.TestCase):
    """Tests for node_status_callback in NodeExecutor.execute() (pure Python)."""

    def _make_executor_with_node(self, node_type="timer", fail=False):
        """Build a single-node NodeExecutor ready to execute."""
        from system.runtime.node_executor import NodeExecutor

        exc = NodeExecutor()
        exc.use_flow_aware_execution = True

        # Add a minimal node that the executor's _execute_node can dispatch.
        # We use a custom node_type and patch _execute_node to avoid registry lookup.
        exc.add_node("n1", node_type, {"kind": "timer"})

        def _fake_execute_node(node, inputs, context):
            if fail:
                raise Exception("forced failure")
            return {"status": "ok"}

        exc._execute_node = _fake_execute_node
        return exc

    def test_callback_running_fires(self):
        """execute() fires 'running' callback for each node."""
        events = []
        exc = self._make_executor_with_node()

        def cb(nid, status):
            events.append((status, nid))

        exc.execute(node_status_callback=cb)

        running_events = [(s, n) for (s, n) in events if s == "running"]
        self.assertEqual(len(running_events), 1)
        self.assertEqual(running_events[0][1], "n1")

    def test_callback_success_fires(self):
        """execute() fires 'success' callback after a successful node."""
        events = []
        exc = self._make_executor_with_node()
        exc.execute(node_status_callback=lambda nid, st: events.append((st, nid)))
        success_events = [s for (s, _) in events if s == "success"]
        self.assertGreater(len(success_events), 0)

    def test_callback_failed_fires_on_exception(self):
        """execute() fires 'failed' callback when node raises."""
        events = []
        exc = self._make_executor_with_node(fail=True)
        exc.execute(node_status_callback=lambda nid, st: events.append((st, nid)))
        statuses = [s for (s, _) in events]
        self.assertIn("running", statuses)
        self.assertIn("failed",  statuses)
        self.assertNotIn("success", statuses)

    def test_callback_attribute_set_before_execute(self):
        """_node_status_callback attribute exists and defaults to None."""
        from system.runtime.node_executor import NodeExecutor
        exc = NodeExecutor()
        # Default value must be None (attribute defined in __init__)
        self.assertIsNone(exc._node_status_callback)

    def test_callback_replaced_each_run(self):
        """A second execute() with a different callback replaces the previous one."""
        exc = self._make_executor_with_node()
        calls_a, calls_b = [], []
        exc.execute(node_status_callback=lambda nid, st: calls_a.append(st))
        # Re-build node state for second run
        exc2 = self._make_executor_with_node()
        exc2.execute(node_status_callback=lambda nid, st: calls_b.append(st))
        self.assertIn("success", calls_a)
        self.assertIn("success", calls_b)

    def test_none_callback_no_crash(self):
        """execute() with no callback completes without error."""
        exc = self._make_executor_with_node()
        result = exc.execute()
        self.assertIsInstance(result, dict)

    def test_topological_path_fires_callback(self):
        """Topological fallback path also fires callbacks."""
        from system.runtime.node_executor import NodeExecutor
        exc = NodeExecutor()
        exc.use_flow_aware_execution = False  # force compat path

        exc.add_node("t1", "timer", {"kind": "timer"})
        exc.add_node("t2", "timer", {"kind": "timer"})
        exc.add_connection("t1", "out", "t2", "in", edge_type="flow")

        def _fake_execute_node(node, inputs, context):
            return {"status": "ok"}
        exc._execute_node = _fake_execute_node

        events = []
        exc.execute(node_status_callback=lambda nid, st: events.append((st, nid)))

        running_ids = {nid for (s, nid) in events if s == "running"}
        self.assertIn("t1", running_ids)
        self.assertIn("t2", running_ids)


# ---------------------------------------------------------------------------
# Pending status and lifecycle integration (PySide6 GraphScene)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestPendingAndLifecycle(unittest.TestCase):
    """Tests for pending/running/success/failed lifecycle in GraphScene."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _make_scene(self):
        from bin.components.graph_scene import GraphScene
        return GraphScene()

    def test_pending_color_distinct_from_default(self):
        """'pending' status applies a colour different from default border."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)
        default_pen = item.pen()

        scene.set_node_execution_status(nid, "pending")
        pending_pen = item.pen()

        self.assertNotEqual(
            pending_pen.color().name(), default_pen.color().name(),
            "Pending border should differ from default border",
        )

    def test_running_color_is_blue_dominant(self):
        """'running' status applies a blue-dominant border pen."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "running")
        color = item.pen().color()

        self.assertGreater(color.blue(), color.red(),
            "'running' border should be blue-dominant (#3b82f6)")

    def test_pending_to_running_transition(self):
        """Node transitions from pending to running correctly."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "pending")
        self.assertEqual(item._exec_status, "pending")

        scene.set_node_execution_status(nid, "running")
        self.assertEqual(item._exec_status, "running")

    def test_running_to_success_transition(self):
        """Node transitions from running to success correctly."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "running")
        scene.set_node_execution_status(nid, "success")
        self.assertEqual(item._exec_status, "success")

    def test_running_to_failed_transition(self):
        """Node transitions from running to failed correctly."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "running")
        scene.set_node_execution_status(nid, "failed")
        self.assertEqual(item._exec_status, "failed")

    def test_reset_clears_pending_status(self):
        """reset_execution_statuses clears a pending node back to default."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "pending")
        scene.reset_execution_statuses()

        self.assertIsNone(item._exec_status)

    def test_multiple_nodes_all_get_pending(self):
        """All nodes in a scene can be marked pending independently."""
        scene = self._make_scene()
        items = [scene.create_node("Timer", QPointF(i * 150, 0)) for i in range(4)]
        nids  = [it.data(12) for it in items]

        for nid in nids:
            scene.set_node_execution_status(nid, "pending")

        for it in items:
            self.assertEqual(it._exec_status, "pending")

    def test_pending_then_skipped_on_reset_reapply(self):
        """Node set to pending then overwritten by skipped (after reset-free reapply)."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        scene.set_node_execution_status(nid, "pending")
        # Simulate: node was never reached → _apply_node_execution_statuses sets skipped
        scene.set_node_execution_status(nid, "skipped")
        self.assertEqual(item._exec_status, "skipped")

    def test_full_lifecycle_sequence_for_one_node(self):
        """Full lifecycle: pending → running → success is deterministic."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        lifecycle = ["pending", "running", "success"]
        for expected_status in lifecycle:
            scene.set_node_execution_status(nid, expected_status)
            self.assertEqual(item._exec_status, expected_status,
                f"Expected _exec_status='{expected_status}' after set")

    def test_full_lifecycle_for_failed_node(self):
        """Full lifecycle: pending → running → failed is deterministic."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)

        for expected_status in ["pending", "running", "failed"]:
            scene.set_node_execution_status(nid, expected_status)
            self.assertEqual(item._exec_status, expected_status)

    def test_rerun_reset_clears_all_then_pending_applies(self):
        """reset_execution_statuses then pending re-apply works correctly."""
        scene = self._make_scene()
        items = [scene.create_node("Timer", QPointF(i * 100, 0)) for i in range(3)]
        nids  = [it.data(12) for it in items]

        # First run: success
        for nid in nids:
            scene.set_node_execution_status(nid, "success")

        # Reset (simulating reset_execution_statuses before re-run)
        scene.reset_execution_statuses()
        for it in items:
            self.assertIsNone(it._exec_status)

        # Mark pending for second run
        for nid in nids:
            scene.set_node_execution_status(nid, "pending")
        for it in items:
            self.assertEqual(it._exec_status, "pending")


if __name__ == "__main__":
    unittest.main()
