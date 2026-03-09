"""Phase 7 STAGE-04 — Fallback and duplicate logic cleanup tests.

Verifies that the three silent fallback branches (M3, M4, M5) and the
equivalent stop() double-fallback have been removed from RobotContext.

Removal guarantees:
  - M3: _create_model_for_brand dead code removed; unknown brand never
        silently resolves to UnitreeModel
  - M4: run_action() lifecycle error → False (no router.run_action fallback,
        no robot.run_action fallback)
  - M5: get_sensor_data() lifecycle error → error dict (no router fallback,
        no robot.get_sensor_data fallback)
  - stop(): lifecycle error → log_error only (no router.stop fallback,
            no robot.stop fallback)

Sections:
    A — M3: _create_model_for_brand removed; _create_adapter_for_brand intact
    B — M4: run_action() lifecycle-error path returns False without fallback
    C — M4: run_action() exception path returns False without fallback
    D — M5: get_sensor_data() lifecycle-error path returns error dict without fallback
    E — M5: get_sensor_data() exception path returns error dict without fallback
    F — stop() lifecycle-error and exception paths contain no fallback calls
    G — Result-shape invariants preserved after cleanup
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_context():
    from bin.core.robot_context import RobotContext
    RobotContext.reset()


def _setup_unitree(action_result=True, sensor_result=None):
    """Wire up a mock unitree adapter that succeeds by default."""
    from bin.core.robot_context import RobotContext
    _reset_context()
    mock_adapter = MagicMock()
    mock_adapter.open_session.return_value  = {"status": "ok"}
    mock_adapter.preflight.return_value     = {"status": "ok"}
    mock_adapter.close_session.return_value = {"status": "ok"}
    mock_adapter.run_action.return_value    = action_result
    mock_adapter.get_sensor_data.return_value = sensor_result or {"imu": {}}
    mock_adapter.stop.return_value          = None
    mock_adapter.connect.return_value       = None
    RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
    RobotContext.set_robot_type("go2")
    return mock_adapter


# ── Section A: M3 — _create_model_for_brand removed ──────────────────────────

class TestCreateModelForBrandRemoved(unittest.TestCase):
    """_create_model_for_brand was dead code with a silent UnitreeModel fallback.
    It must no longer exist on RobotContext."""

    def test_create_model_for_brand_method_does_not_exist(self):
        from bin.core.robot_context import RobotContext
        self.assertFalse(
            hasattr(RobotContext, "_create_model_for_brand"),
            "_create_model_for_brand should have been removed (dead code, M3)",
        )

    def test_create_adapter_for_brand_still_exists(self):
        from bin.core.robot_context import RobotContext
        self.assertTrue(hasattr(RobotContext, "_create_adapter_for_brand"))

    def test_create_adapter_for_brand_unknown_returns_none(self):
        from bin.core.robot_context import RobotContext
        result = RobotContext._create_adapter_for_brand("totally_unknown_brand", "robot1")
        self.assertIsNone(result)

    def test_create_adapter_for_brand_unknown_does_not_return_unitree_model(self):
        """Confirm the silent UnitreeModel fallback is gone from the adapter factory too."""
        from bin.core.robot_context import RobotContext
        result = RobotContext._create_adapter_for_brand("totally_unknown_brand", "robot1")
        # Must be None — never a UnitreeModel instance
        if result is not None:
            from models.Unitree import UnitreeModel
            self.assertNotIsInstance(result, UnitreeModel)

    def test_create_adapter_for_brand_unitree_still_works(self):
        from bin.core.robot_context import RobotContext
        result = RobotContext._create_adapter_for_brand("unitree", "go2")
        self.assertIsNotNone(result)


# ── Section B: M4 — run_action() lifecycle-error path ────────────────────────

class TestRunActionLifecycleErrorNoFallback(unittest.TestCase):
    """When _lifecycle_route returns (False, ...), run_action must return False
    without calling router.run_action or robot_model.run_action."""

    def setUp(self):
        _reset_context()

    def _trap_adapter(self):
        """A trap adapter that raises AssertionError if run_action is called directly."""
        adapter = MagicMock()
        adapter.open_session.return_value  = {"status": "ok"}
        adapter.preflight.return_value     = {"status": "ok"}
        adapter.close_session.return_value = {"status": "ok"}
        adapter.run_action.side_effect = AssertionError(
            "run_action called on adapter directly (double-fallback M4)"
        )
        adapter.connect.return_value = None
        return adapter

    def test_lifecycle_error_returns_false(self):
        from bin.core.robot_context import RobotContext
        _reset_context()
        mock_adapter = self._trap_adapter()
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        # Make the lifecycle route itself return error
        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            result = RobotContext.run_action("stand")

        self.assertFalse(result)

    def test_lifecycle_error_does_not_call_router_run_action(self):
        from bin.core.robot_context import RobotContext
        _reset_context()
        mock_adapter = self._trap_adapter()
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        router_spy = MagicMock(side_effect=AssertionError("router.run_action called (M4 fallback)"))
        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            with patch.object(RobotContext._service_router, "run_action", router_spy):
                result = RobotContext.run_action("stand")

        # router.run_action must NOT have been called
        router_spy.assert_not_called()
        self.assertFalse(result)

    def test_lifecycle_error_does_not_call_get_robot_model(self):
        from bin.core.robot_context import RobotContext
        _reset_context()
        mock_adapter = self._trap_adapter()
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            with patch.object(RobotContext, "get_robot_model") as mock_get:
                mock_get.side_effect = AssertionError("get_robot_model called (M4 fallback)")
                result = RobotContext.run_action("stand")

        mock_get.assert_not_called()
        self.assertFalse(result)


# ── Section C: M4 — run_action() exception path ───────────────────────────────

class TestRunActionExceptionNoFallback(unittest.TestCase):

    def setUp(self):
        _reset_context()

    def test_lifecycle_exception_returns_false(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route",
                          side_effect=RuntimeError("network fault")):
            result = RobotContext.run_action("stand")

        self.assertFalse(result)

    def test_lifecycle_exception_does_not_call_robot_run_action(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        mock_adapter.run_action.side_effect = AssertionError("model.run_action called (M4 exception fallback)")
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route",
                          side_effect=RuntimeError("network fault")):
            with patch.object(RobotContext, "get_robot_model") as mock_get:
                mock_get.side_effect = AssertionError("get_robot_model called (M4 exception fallback)")
                result = RobotContext.run_action("stand")

        mock_get.assert_not_called()
        self.assertFalse(result)


# ── Section D: M5 — get_sensor_data() lifecycle-error path ───────────────────

class TestGetSensorDataLifecycleErrorNoFallback(unittest.TestCase):

    def setUp(self):
        _reset_context()

    def test_lifecycle_error_returns_error_dict(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            result = RobotContext.get_sensor_data()

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_lifecycle_error_does_not_call_router_get_sensor_data(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        router_spy = MagicMock(side_effect=AssertionError("router.get_sensor_data called (M5 fallback)"))
        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            with patch.object(RobotContext._service_router, "get_sensor_data", router_spy):
                result = RobotContext.get_sensor_data()

        router_spy.assert_not_called()
        self.assertIn("error", result)

    def test_lifecycle_error_does_not_call_get_robot_model(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            with patch.object(RobotContext, "get_robot_model") as mock_get:
                mock_get.side_effect = AssertionError("get_robot_model called (M5 fallback)")
                result = RobotContext.get_sensor_data()

        mock_get.assert_not_called()
        self.assertIn("error", result)


# ── Section E: M5 — get_sensor_data() exception path ─────────────────────────

class TestGetSensorDataExceptionNoFallback(unittest.TestCase):

    def setUp(self):
        _reset_context()

    def test_lifecycle_exception_returns_error_dict(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route",
                          side_effect=OSError("connection lost")):
            result = RobotContext.get_sensor_data()

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_lifecycle_exception_error_message_is_string(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route",
                          side_effect=OSError("connection lost")):
            result = RobotContext.get_sensor_data()

        self.assertIsInstance(result["error"], str)

    def test_lifecycle_exception_does_not_call_robot_get_sensor_data(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        mock_adapter.get_sensor_data.side_effect = AssertionError(
            "model.get_sensor_data called (M5 exception fallback)"
        )
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route",
                          side_effect=OSError("connection lost")):
            with patch.object(RobotContext, "get_robot_model") as mock_get:
                mock_get.side_effect = AssertionError("get_robot_model called (M5 exception fallback)")
                result = RobotContext.get_sensor_data()

        mock_get.assert_not_called()
        self.assertIn("error", result)


# ── Section F: stop() double-fallback removed ─────────────────────────────────

class TestStopNoFallback(unittest.TestCase):

    def setUp(self):
        _reset_context()

    def test_stop_lifecycle_error_does_not_call_router_stop(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        router_spy = MagicMock(side_effect=AssertionError("router.stop called (stop fallback)"))
        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            with patch.object(RobotContext._service_router, "stop", router_spy):
                RobotContext.stop()  # must not raise

        router_spy.assert_not_called()

    def test_stop_lifecycle_error_does_not_call_robot_stop(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            with patch.object(RobotContext, "get_robot_model") as mock_get:
                mock_get.side_effect = AssertionError("get_robot_model called (stop fallback)")
                RobotContext.stop()

        mock_get.assert_not_called()

    def test_stop_exception_does_not_call_robot_stop(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route",
                          side_effect=RuntimeError("hw fault")):
            with patch.object(RobotContext, "get_robot_model") as mock_get:
                mock_get.side_effect = AssertionError("get_robot_model called (stop exception fallback)")
                RobotContext.stop()

        mock_get.assert_not_called()

    def test_stop_returns_none_on_lifecycle_error(self):
        from bin.core.robot_context import RobotContext
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = None
        RobotContext._service_registry.register("unitree_sdk2", mock_adapter)
        RobotContext.set_robot_type("go2")

        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            result = RobotContext.stop()

        self.assertIsNone(result)


# ── Section G: Result-shape invariants preserved ─────────────────────────────

class TestResultShapePreserved(unittest.TestCase):
    """The public API return types must be unchanged after cleanup."""

    def setUp(self):
        _reset_context()

    def test_run_action_success_returns_bool_true(self):
        mock_adapter = _setup_unitree(action_result=True)
        mock_adapter.open_session.return_value  = {"status": "ok"}
        mock_adapter.close_session.return_value = {"status": "ok"}
        from bin.core.robot_context import RobotContext
        with patch.object(RobotContext, "_lifecycle_route", return_value=(True, True)):
            result = RobotContext.run_action("stand")
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_run_action_error_returns_bool_false(self):
        from bin.core.robot_context import RobotContext
        _setup_unitree()
        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            result = RobotContext.run_action("stand")
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_get_sensor_data_success_returns_dict(self):
        from bin.core.robot_context import RobotContext
        _setup_unitree()
        payload = {"imu": {"ax": 0.1}}
        with patch.object(RobotContext, "_lifecycle_route", return_value=(True, payload)):
            result = RobotContext.get_sensor_data()
        self.assertIsInstance(result, dict)
        self.assertEqual(result, payload)

    def test_get_sensor_data_error_returns_dict_with_error_key(self):
        from bin.core.robot_context import RobotContext
        _setup_unitree()
        with patch.object(RobotContext, "_lifecycle_route", return_value=(False, None)):
            result = RobotContext.get_sensor_data()
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_stop_always_returns_none(self):
        from bin.core.robot_context import RobotContext
        _setup_unitree()
        with patch.object(RobotContext, "_lifecycle_route", return_value=(True, None)):
            result = RobotContext.stop()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
