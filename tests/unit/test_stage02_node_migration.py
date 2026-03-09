"""Phase 7 STAGE-02 — Node direct-call migration tests.

Verifies that ActionExecutionNode, StopNode, and SensorInputNode execute()
methods now route through RobotContext class methods (lifecycle-routed path)
instead of calling robot_model.* directly.

Sections:
    A — ActionExecutionNode.execute() calls RobotContext.run_action()
    B — StopNode.execute() calls RobotContext.stop()
    C — SensorInputNode.execute() calls RobotContext.get_sensor_data()
    D — Result shape invariants
    E — ImportError handling (RobotContext unavailable)
    F — Exception propagation
    G — Regression: no direct robot_model.* calls in execute()
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_action_node(action: str = "stand"):
    from nodes.sys_nodes.action_nodes import ActionExecutionNode
    node = ActionExecutionNode("n1")
    node.parameters["action"] = action
    return node


def _make_stop_node():
    from nodes.sys_nodes.action_nodes import StopNode
    return StopNode("n2")


def _make_sensor_node(sensor_type: str = "imu"):
    from nodes.sys_nodes.sensor_nodes import SensorInputNode
    node = SensorInputNode("n3")
    node.parameters["sensor_type"] = sensor_type
    return node


# ── Section A: ActionExecutionNode calls RobotContext.run_action() ────────────

class TestActionNodeRobotContextRouting(unittest.TestCase):

    def test_calls_robotcontext_run_action_not_model(self):
        node = _make_action_node("walk")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.run_action.assert_called_once_with("walk")

    def test_does_not_call_get_robot_model(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.get_robot_model.assert_not_called()

    def test_action_parameter_forwarded(self):
        node = _make_action_node("sit")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.run_action.assert_called_once_with("sit")

    def test_success_true_gives_status_success(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["status"], "success")

    def test_success_false_gives_status_failed(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = False
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["status"], "failed")

    def test_robot_type_from_robotcontext_get_robot_type(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "b1"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["robot_type"], "b1")

    def test_action_in_result(self):
        node = _make_action_node("dance")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["action"], "dance")

    def test_default_action_is_stand(self):
        from nodes.sys_nodes.action_nodes import ActionExecutionNode
        node = ActionExecutionNode("nx")
        # don't override action param — leave default
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.run_action.assert_called_once_with("stand")


# ── Section B: StopNode calls RobotContext.stop() ────────────────────────────

class TestStopNodeRobotContextRouting(unittest.TestCase):

    def test_calls_robotcontext_stop(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.stop.assert_called_once()

    def test_does_not_call_get_robot_model(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.get_robot_model.assert_not_called()

    def test_returns_stopped_status(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["status"], "stopped")

    def test_stop_called_even_when_stop_returns_none(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        mock_rc.stop.return_value = None
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        mock_rc.stop.assert_called_once()
        self.assertEqual(result["flow_out"]["status"], "stopped")

    def test_importerror_still_returns_stopped(self):
        node = _make_stop_node()
        # Simulate RobotContext import failure
        saved = sys.modules.pop("bin.core.robot_context", None)
        sys.modules["bin.core.robot_context"] = None  # makes import raise ImportError
        try:
            result = node.execute({})
        finally:
            if saved is not None:
                sys.modules["bin.core.robot_context"] = saved
            else:
                sys.modules.pop("bin.core.robot_context", None)
        # StopNode silently swallows ImportError and still returns stopped
        self.assertEqual(result["flow_out"]["status"], "stopped")


# ── Section C: SensorInputNode calls RobotContext.get_sensor_data() ──────────

class TestSensorNodeRobotContextRouting(unittest.TestCase):

    def test_calls_robotcontext_get_sensor_data(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {"imu": {"x": 0.0}}
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.get_sensor_data.assert_called_once()

    def test_does_not_call_get_robot_model(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {}
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})
        mock_rc.get_robot_model.assert_not_called()

    def test_sensor_data_in_result(self):
        node = _make_sensor_node("imu")
        payload = {"imu": {"ax": 1.0, "ay": 0.0, "az": 9.8}}
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = payload
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["data"], payload)

    def test_sensor_type_in_result(self):
        node = _make_sensor_node("camera")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {"camera": {}}
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["sensor_type"], "camera")

    def test_status_success_on_ok(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {}
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["status"], "success")

    def test_robot_type_from_robotcontext(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {}
        mock_rc.get_robot_type.return_value = "a1"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["robot_type"], "a1")

    def test_default_sensor_type_is_imu(self):
        from nodes.sys_nodes.sensor_nodes import SensorInputNode
        node = SensorInputNode("nx")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {}
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["sensor_type"], "imu")


# ── Section D: Result shape invariants ───────────────────────────────────────

class TestResultShapeInvariants(unittest.TestCase):

    def _action_result(self, success=True):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = success
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            return node.execute({})

    def _sensor_result(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {"k": "v"}
        mock_rc.get_robot_type.return_value = "go2"
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            return node.execute({})

    def test_action_result_has_flow_out_key(self):
        self.assertIn("flow_out", self._action_result())

    def test_action_result_flow_out_has_status(self):
        self.assertIn("status", self._action_result()["flow_out"])

    def test_action_result_flow_out_has_action(self):
        self.assertIn("action", self._action_result()["flow_out"])

    def test_action_result_flow_out_has_robot_type(self):
        self.assertIn("robot_type", self._action_result()["flow_out"])

    def test_stop_result_has_flow_out_key(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertIn("flow_out", result)

    def test_stop_result_flow_out_has_status(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertIn("status", result["flow_out"])

    def test_sensor_result_has_out_key(self):
        self.assertIn("out", self._sensor_result())

    def test_sensor_result_out_has_status(self):
        self.assertIn("status", self._sensor_result()["out"])

    def test_sensor_result_out_has_sensor_type(self):
        self.assertIn("sensor_type", self._sensor_result()["out"])

    def test_sensor_result_out_has_data(self):
        self.assertIn("data", self._sensor_result()["out"])

    def test_sensor_result_out_has_robot_type(self):
        self.assertIn("robot_type", self._sensor_result()["out"])


# ── Section E: ImportError handling ──────────────────────────────────────────

class TestImportErrorHandling(unittest.TestCase):

    def _patch_no_import(self):
        """Context: bin.core.robot_context raises ImportError."""
        return patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
            (_ for _ in ()).throw(ImportError(f"No module named '{name}'"))
            if name == "bin.core.robot_context"
            else __import__(name, *a, **kw)
        ))

    def test_action_importerror_returns_error_status(self):
        node = _make_action_node("stand")
        with self._patch_no_import():
            result = node.execute({})
        self.assertEqual(result["flow_out"]["status"], "error")

    def test_action_importerror_message_mentions_robotcontext(self):
        node = _make_action_node("stand")
        with self._patch_no_import():
            result = node.execute({})
        self.assertIn("RobotContext", result["flow_out"]["message"])

    def test_sensor_importerror_returns_error_status(self):
        node = _make_sensor_node("imu")
        with self._patch_no_import():
            result = node.execute({})
        self.assertEqual(result["out"]["status"], "error")

    def test_sensor_importerror_message_mentions_robotcontext(self):
        node = _make_sensor_node("imu")
        with self._patch_no_import():
            result = node.execute({})
        self.assertIn("RobotContext", result["out"]["message"])


# ── Section F: Exception propagation ─────────────────────────────────────────

class TestExceptionPropagation(unittest.TestCase):

    def test_action_exception_in_run_action_gives_error_status(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.side_effect = RuntimeError("hardware fault")
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["status"], "error")
        self.assertIn("hardware fault", result["flow_out"]["message"])

    def test_action_exception_message_preserved(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.side_effect = ValueError("bad param")
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["flow_out"]["message"], "bad param")

    def test_sensor_exception_gives_error_status(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.side_effect = RuntimeError("sensor offline")
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["status"], "error")
        self.assertIn("sensor offline", result["out"]["message"])

    def test_sensor_exception_message_preserved(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.side_effect = OSError("network unreachable")
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            result = node.execute({})
        self.assertEqual(result["out"]["message"], "network unreachable")


# ── Section G: Regression — no direct robot_model.* calls in execute() ───────

class TestNoDirectModelCalls(unittest.TestCase):
    """Structural regression: execute() must not call robot_model.run_action,
    robot_model.get_sensor_data, or robot_model.stop directly.
    We verify by injecting a mock model that raises if called."""

    def _trap_model(self):
        model = MagicMock()
        model.run_action.side_effect = AssertionError("run_action called on model directly")
        model.get_sensor_data.side_effect = AssertionError("get_sensor_data called on model directly")
        model.stop.side_effect = AssertionError("stop called on model directly")
        return model

    def test_action_node_does_not_call_model_run_action(self):
        node = _make_action_node("stand")
        mock_rc = MagicMock()
        mock_rc.run_action.return_value = True
        mock_rc.get_robot_type.return_value = "go2"
        mock_rc.get_robot_model.return_value = self._trap_model()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            # Must not raise (trap model is registered but should never be called)
            node.execute({})

    def test_stop_node_does_not_call_model_stop(self):
        node = _make_stop_node()
        mock_rc = MagicMock()
        mock_rc.get_robot_model.return_value = self._trap_model()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})

    def test_sensor_node_does_not_call_model_get_sensor_data(self):
        node = _make_sensor_node("imu")
        mock_rc = MagicMock()
        mock_rc.get_sensor_data.return_value = {}
        mock_rc.get_robot_type.return_value = "go2"
        mock_rc.get_robot_model.return_value = self._trap_model()
        with patch.dict("sys.modules", {"bin.core.robot_context": MagicMock(RobotContext=mock_rc)}):
            node.execute({})


if __name__ == "__main__":
    unittest.main()
