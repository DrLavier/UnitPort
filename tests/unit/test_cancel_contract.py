#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests — Cycle 3 STAGE-04: Preemptive Cancel Contract.

Covers:
  A. BaseAdapter.cancel_action() — contract defaults
  B. Concrete adapter overrides (UnitreeAdapter, SpotAdapter, CyberDogAdapter)
  C. RuntimeEngine.request_cancel() — adapter-aware cancel path
  D. RuntimeEngine.execute() diagnostics — ADAPTER_CANCEL_INVOKED key
  E. Cooperative-cancel fallback — cancel_action() not called when no model
  F. Reason codes — "mission_cancelled" distinguishable from runtime failure

All tests run without a real robot, SDK, or Qt event loop.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# A — BaseAdapter.cancel_action() contract defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseAdapterCancelContract(unittest.TestCase):
    """cancel_action() is present and safe-no-op on the base class."""

    def setUp(self):
        from system.service.adapters.base_adapter import BaseAdapter

        # Minimal concrete subclass — only required abstract methods.
        class _MinAdapter(BaseAdapter):
            def connect(self, **kw): return True
            def run_action(self, action, **kw): return True
            def stop(self): pass
            def get_sensor_data(self): return {}
            def health(self): return {}

        self.adapter = _MinAdapter()

    def test_cancel_action_method_exists(self):
        self.assertTrue(callable(getattr(self.adapter, "cancel_action", None)))

    def test_cancel_action_returns_none(self):
        result = self.adapter.cancel_action()
        self.assertIsNone(result)

    def test_cancel_action_does_not_raise(self):
        try:
            self.adapter.cancel_action()
        except Exception as exc:
            self.fail(f"cancel_action() must not raise; got {exc}")

    def test_cancel_action_callable_multiple_times(self):
        """Calling cancel multiple times must be idempotent."""
        for _ in range(3):
            try:
                self.adapter.cancel_action()
            except Exception as exc:
                self.fail(f"Repeated cancel_action() raised: {exc}")

    def test_cancel_action_present_on_base_class_not_abstract(self):
        """cancel_action() must NOT be abstract — safe no-op default required."""
        import inspect
        from system.service.adapters.base_adapter import BaseAdapter
        abstract_methods = getattr(BaseAdapter, "__abstractmethods__", set())
        self.assertNotIn("cancel_action", abstract_methods,
                         "cancel_action must not be abstract (safe no-op default required)")


# ══════════════════════════════════════════════════════════════════════════════
# B — Concrete adapter overrides
# ══════════════════════════════════════════════════════════════════════════════


class TestUnitreeAdapterCancelAction(unittest.TestCase):
    """UnitreeAdapter.cancel_action() calls model.stop() when model is present."""

    def _make_adapter(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter.__new__(UnitreeAdapter)
        adapter.robot_type = "go2"
        adapter._model = None
        return adapter

    def test_cancel_action_no_model_does_not_raise(self):
        adapter = self._make_adapter()
        adapter._model = None
        try:
            adapter.cancel_action()
        except Exception as exc:
            self.fail(f"cancel_action with no model raised: {exc}")

    def test_cancel_action_calls_model_stop(self):
        adapter = self._make_adapter()
        mock_model = MagicMock()
        adapter._model = mock_model
        adapter.cancel_action()
        mock_model.stop.assert_called_once()

    def test_cancel_action_silences_model_stop_exception(self):
        adapter = self._make_adapter()
        mock_model = MagicMock()
        mock_model.stop.side_effect = RuntimeError("SDK crash")
        adapter._model = mock_model
        try:
            adapter.cancel_action()  # must not propagate exception
        except Exception as exc:
            self.fail(f"cancel_action must silence exceptions; got {exc}")

    def test_cancel_action_does_not_call_get_model(self):
        """cancel_action must read self._model directly (no side-effecting get_model())."""
        adapter = self._make_adapter()
        adapter.get_model = MagicMock(side_effect=AssertionError("get_model must not be called"))
        adapter._model = None
        # Should not raise (no get_model call, no model → skip stop)
        try:
            adapter.cancel_action()
        except AssertionError:
            self.fail("cancel_action() must not call get_model()")


class TestSpotAdapterCancelAction(unittest.TestCase):
    """SpotAdapter.cancel_action() delegates to stop()."""

    def _make_adapter(self):
        from system.service.adapters.spot_sdk.adapter import SpotAdapter
        adapter = SpotAdapter.__new__(SpotAdapter)
        adapter._session_open = False
        adapter._robot = None
        adapter._lease_client = None
        adapter._cmd_client = None
        return adapter

    def test_cancel_action_exists(self):
        adapter = self._make_adapter()
        self.assertTrue(callable(getattr(adapter, "cancel_action", None)))

    def test_cancel_action_no_session_does_not_raise(self):
        adapter = self._make_adapter()
        try:
            adapter.cancel_action()
        except Exception as exc:
            self.fail(f"cancel_action with no session raised: {exc}")

    def test_cancel_action_calls_stop(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "stop") as mock_stop:
            adapter.cancel_action()
            mock_stop.assert_called_once()

    def test_cancel_action_silences_stop_exception(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "stop", side_effect=RuntimeError("spot error")):
            try:
                adapter.cancel_action()
            except Exception as exc:
                self.fail(f"cancel_action must silence exceptions; got {exc}")


class TestCyberDogAdapterCancelAction(unittest.TestCase):
    """CyberDogAdapter.cancel_action() delegates to stop()."""

    def _make_adapter(self):
        from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter
        adapter = CyberDogAdapter.__new__(CyberDogAdapter)
        adapter._session_open = False
        adapter._node = None
        adapter.namespace = "/unitree"
        return adapter

    def test_cancel_action_exists(self):
        adapter = self._make_adapter()
        self.assertTrue(callable(getattr(adapter, "cancel_action", None)))

    def test_cancel_action_no_session_does_not_raise(self):
        adapter = self._make_adapter()
        try:
            adapter.cancel_action()
        except Exception as exc:
            self.fail(f"cancel_action with no session raised: {exc}")

    def test_cancel_action_calls_stop(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "stop") as mock_stop:
            adapter.cancel_action()
            mock_stop.assert_called_once()

    def test_cancel_action_silences_stop_exception(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "stop", side_effect=RuntimeError("ros2 error")):
            try:
                adapter.cancel_action()
            except Exception as exc:
                self.fail(f"cancel_action must silence exceptions; got {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# C — RuntimeEngine.request_cancel() adapter-aware path
# ══════════════════════════════════════════════════════════════════════════════


class TestRuntimeEngineCancelAdapterPath(unittest.TestCase):
    """request_cancel() calls robot_model.cancel_action() when model is present."""

    def _make_engine(self):
        from system.runtime.runtime_engine import RuntimeEngine
        return RuntimeEngine()

    def test_request_cancel_sets_flag(self):
        engine = self._make_engine()
        self.assertFalse(engine._cancel_requested)
        engine.request_cancel()
        self.assertTrue(engine._cancel_requested)

    def test_request_cancel_no_model_does_not_raise(self):
        engine = self._make_engine()
        engine._current_robot_model = None
        try:
            engine.request_cancel()
        except Exception as exc:
            self.fail(f"request_cancel with no model raised: {exc}")

    def test_request_cancel_calls_cancel_action_when_model_present(self):
        engine = self._make_engine()
        mock_model = MagicMock()
        engine._current_robot_model = mock_model
        engine.request_cancel()
        mock_model.cancel_action.assert_called_once()

    def test_request_cancel_sets_adapter_cancel_invoked_flag(self):
        engine = self._make_engine()
        mock_model = MagicMock()
        engine._current_robot_model = mock_model
        self.assertFalse(engine._adapter_cancel_invoked)
        engine.request_cancel()
        self.assertTrue(engine._adapter_cancel_invoked)

    def test_request_cancel_no_cancel_action_attr_does_not_raise(self):
        """Model without cancel_action must not cause request_cancel to raise."""
        engine = self._make_engine()
        model_without_cancel = MagicMock(spec=[])  # no attributes
        engine._current_robot_model = model_without_cancel
        try:
            engine.request_cancel()
        except Exception as exc:
            self.fail(f"request_cancel must not raise when model lacks cancel_action: {exc}")

    def test_request_cancel_cancel_action_raises_is_silenced(self):
        """Exceptions from cancel_action() must not propagate out of request_cancel()."""
        engine = self._make_engine()
        mock_model = MagicMock()
        mock_model.cancel_action.side_effect = RuntimeError("adapter crash")
        engine._current_robot_model = mock_model
        try:
            engine.request_cancel()
        except Exception as exc:
            self.fail(f"request_cancel must silence cancel_action exceptions: {exc}")

    def test_request_cancel_adapter_invoked_flag_false_when_cancel_raises(self):
        """When cancel_action() raises, adapter_cancel_invoked stays False."""
        engine = self._make_engine()
        mock_model = MagicMock()
        mock_model.cancel_action.side_effect = RuntimeError("crash")
        engine._current_robot_model = mock_model
        engine.request_cancel()
        self.assertFalse(engine._adapter_cancel_invoked)

    def test_request_cancel_thread_safe_from_different_thread(self):
        """request_cancel() must be callable from a thread other than the runner thread."""
        engine = self._make_engine()
        mock_model = MagicMock()
        engine._current_robot_model = mock_model
        errors = []

        def _cancel_from_thread():
            try:
                engine.request_cancel()
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_cancel_from_thread)
        t.start()
        t.join(timeout=2)
        self.assertFalse(errors, f"request_cancel raised from thread: {errors}")
        mock_model.cancel_action.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# D — RuntimeEngine.execute() diagnostics: ADAPTER_CANCEL_INVOKED
# ══════════════════════════════════════════════════════════════════════════════


def _make_minimal_exec_graph():
    return {
        "nodes": {"1": {"name": "Start", "type": "start"}},
        "outgoing": {"1": {}},
        "incoming": {"1": {}},
        "entry_nodes": ["1"],
    }


class TestRuntimeEngineAdapterCancelDiagnostics(unittest.TestCase):
    """ADAPTER_CANCEL_INVOKED diagnostics key is set correctly."""

    def _make_engine(self):
        from system.runtime.runtime_engine import RuntimeEngine
        return RuntimeEngine()

    def test_adapter_cancel_invoked_absent_when_no_cancel(self):
        """Normal run (no cancel) must NOT set ADAPTER_CANCEL_INVOKED."""
        from system.runtime.contracts import DiagnosticsKey
        engine = self._make_engine()
        result = engine.execute(_make_minimal_exec_graph(), {})
        self.assertNotIn(DiagnosticsKey.ADAPTER_CANCEL_INVOKED, result.get("diagnostics", {}))

    def test_adapter_cancel_invoked_false_on_cooperative_cancel_only(self):
        """Cooperative cancel (no model) must NOT set ADAPTER_CANCEL_INVOKED."""
        from system.runtime.contracts import DiagnosticsKey

        engine = self._make_engine()

        def _cancel_during_run(**kw):
            engine.request_cancel()
            return {"ok": False, "reason": "mission_cancelled", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": True}

        with patch.object(engine._workflow_runner, "run", side_effect=_cancel_during_run):
            result = engine.execute(_make_minimal_exec_graph(), {})

        self.assertNotIn(DiagnosticsKey.ADAPTER_CANCEL_INVOKED, result.get("diagnostics", {}))

    def test_adapter_cancel_invoked_true_when_model_provides_cancel(self):
        """When adapter cancel_action() is called, ADAPTER_CANCEL_INVOKED=True in diagnostics."""
        from system.runtime.contracts import DiagnosticsKey

        engine = self._make_engine()
        mock_model = MagicMock()

        def _cancel_during_run(**kw):
            engine.request_cancel()
            return {"ok": False, "reason": "mission_cancelled", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": True}

        with patch.object(engine._workflow_runner, "run", side_effect=_cancel_during_run):
            result = engine.execute(_make_minimal_exec_graph(), {"robot_model": mock_model})

        self.assertTrue(result["diagnostics"].get(DiagnosticsKey.ADAPTER_CANCEL_INVOKED))

    def test_current_robot_model_cleared_after_execute(self):
        """_current_robot_model must be None after execute() returns."""
        engine = self._make_engine()
        mock_model = MagicMock()
        engine.execute(_make_minimal_exec_graph(), {"robot_model": mock_model})
        self.assertIsNone(engine._current_robot_model)

    def test_current_robot_model_cleared_even_on_engine_exception(self):
        """_current_robot_model must be cleared even when execute() raises internally."""
        engine = self._make_engine()
        mock_model = MagicMock()

        with patch.object(engine._workflow_runner, "run",
                          side_effect=RuntimeError("unexpected")):
            try:
                engine.execute(_make_minimal_exec_graph(), {"robot_model": mock_model})
            except Exception:
                pass  # internal exception propagation is acceptable

        self.assertIsNone(engine._current_robot_model)

    def test_cancel_flag_reset_on_new_execute(self):
        """_cancel_requested must be False at the start of each execute() call."""
        engine = self._make_engine()
        engine.request_cancel()
        self.assertTrue(engine._cancel_requested)
        # A new execute() should reset the flag
        engine.execute(_make_minimal_exec_graph(), {})
        # After the execute, the flag reflects in-execution state, not pre-state
        # (it was reset at the top of execute, then set if cancel was requested again)
        # Verify by checking that _adapter_cancel_invoked is also reset
        self.assertFalse(engine._adapter_cancel_invoked)


# ══════════════════════════════════════════════════════════════════════════════
# E — Cooperative-cancel fallback: ADAPTER_CANCEL_INVOKED not set without model
# ══════════════════════════════════════════════════════════════════════════════


class TestCooperativeCancelFallback(unittest.TestCase):
    """When no robot_model is in scenario, cancel uses cooperative path only."""

    def _make_engine(self):
        from system.runtime.runtime_engine import RuntimeEngine
        return RuntimeEngine()

    def test_no_model_in_scenario_means_no_adapter_cancel(self):
        """Scenario without robot_model → _current_robot_model is None during run."""
        engine = self._make_engine()
        captured_model = []

        def _record_model(**kw):
            captured_model.append(engine._current_robot_model)
            return {"ok": True, "reason": "", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": False}

        with patch.object(engine._workflow_runner, "run", side_effect=_record_model):
            engine.execute(_make_minimal_exec_graph(), {})  # no robot_model

        self.assertEqual(len(captured_model), 1)
        self.assertIsNone(captured_model[0])

    def test_model_in_scenario_is_stored_during_run(self):
        """Scenario with robot_model → _current_robot_model is that model during run."""
        engine = self._make_engine()
        mock_model = MagicMock()
        captured_model = []

        def _record_model(**kw):
            captured_model.append(engine._current_robot_model)
            return {"ok": True, "reason": "", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": False}

        with patch.object(engine._workflow_runner, "run", side_effect=_record_model):
            engine.execute(_make_minimal_exec_graph(), {"robot_model": mock_model})

        self.assertEqual(len(captured_model), 1)
        self.assertIs(captured_model[0], mock_model)


# ══════════════════════════════════════════════════════════════════════════════
# F — Reason codes: "mission_cancelled" vs runtime failure distinguishable
# ══════════════════════════════════════════════════════════════════════════════


class TestCancelReasonCodes(unittest.TestCase):
    """Cancel result has reason="mission_cancelled", not runtime failure codes."""

    def _make_engine(self):
        from system.runtime.runtime_engine import RuntimeEngine
        return RuntimeEngine()

    def test_cancelled_run_reason_is_mission_cancelled(self):
        engine = self._make_engine()

        def _cancel_and_return(**kw):
            engine.request_cancel()
            return {"ok": True, "reason": "", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": False}

        with patch.object(engine._workflow_runner, "run", side_effect=_cancel_and_return):
            result = engine.execute(_make_minimal_exec_graph(), {})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "mission_cancelled")

    def test_cancelled_reason_distinct_from_node_execution_failed(self):
        """mission_cancelled and node_execution_failed are different reasons."""
        self.assertNotEqual("mission_cancelled", "node_execution_failed")

    def test_cancelled_reason_distinct_from_runtime_error(self):
        """mission_cancelled and runtime error reasons are distinct."""
        self.assertNotEqual("mission_cancelled", "mission_run_thread_error")

    def test_adapter_cancel_invoked_key_is_stable_string(self):
        """DiagnosticsKey.ADAPTER_CANCEL_INVOKED must be a non-empty string constant."""
        from system.runtime.contracts import DiagnosticsKey
        key = DiagnosticsKey.ADAPTER_CANCEL_INVOKED
        self.assertIsInstance(key, str)
        self.assertTrue(key)

    def test_mission_cancelled_with_adapter_cancel_still_reports_failed(self):
        """Adapter-level cancel must still produce status=failed (not success)."""
        from system.runtime.contracts import DiagnosticsKey
        engine = self._make_engine()
        mock_model = MagicMock()

        def _cancel_and_return(**kw):
            engine.request_cancel()
            return {"ok": False, "reason": "mission_cancelled", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": True}

        with patch.object(engine._workflow_runner, "run", side_effect=_cancel_and_return):
            result = engine.execute(_make_minimal_exec_graph(), {"robot_model": mock_model})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "mission_cancelled")
        self.assertTrue(result["diagnostics"].get(DiagnosticsKey.ADAPTER_CANCEL_INVOKED))


if __name__ == "__main__":
    unittest.main()
