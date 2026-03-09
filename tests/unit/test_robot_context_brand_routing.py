#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 STAGE-05: RobotContext multi-brand routing unit tests.

Tests cover:
  A. Brand-map constants (BRAND_ADAPTER_MAP, _FALLBACK_BRAND_MAP)
  B. _create_adapter_for_brand factory
  C. _ensure_adapter multi-brand registration
  D. get_robot_model() for non-Unitree adapters
  E. run_action brand dispatch path
  F. (originally section F — stop/get_sensor_data dispatch)
  G. Unknown-brand routing safety (STAGE-05 regression)
"""

import pytest
from unittest.mock import MagicMock, patch, call

from bin.core.robot_context import RobotContext
from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter


# ─── helpers ──────────────────────────────────────────────────────────────────

def _reset():
    """Reset RobotContext to clean slate before each test."""
    RobotContext.reset()


def _make_mock_adapter(connect_ok: bool = True):
    """Return a mock adapter whose connect() succeeds or raises."""
    m = MagicMock()
    if connect_ok:
        m.connect.return_value = True
    else:
        m.connect.side_effect = RuntimeError("connection refused")
    m.get_model = MagicMock(return_value=None)   # non-Unitree: no model
    return m


# ─── Section A: brand map constants ───────────────────────────────────────────

class TestBrandAdapterMap:
    """BRAND_ADAPTER_MAP keys must match BrandRegistry lowercased directory names."""

    def test_unitree_maps_to_unitree_sdk2(self):
        assert RobotContext.BRAND_ADAPTER_MAP["unitree"] == "unitree_sdk2"

    def test_bostiondynamics_maps_to_spot_sdk(self):
        assert RobotContext.BRAND_ADAPTER_MAP["bostiondynamics"] == "spot_sdk"

    def test_xiaomi_maps_to_cyberdog_sdk(self):
        assert RobotContext.BRAND_ADAPTER_MAP["xiaomi"] == "cyberdog_sdk"

    def test_no_legacy_boston_dynamics_key(self):
        """Old 'boston_dynamics' key must NOT exist (Phase 4 correction)."""
        assert "boston_dynamics" not in RobotContext.BRAND_ADAPTER_MAP

    def test_no_legacy_cyberdog_key(self):
        """Old 'cyberdog' key must NOT exist as a brand (it's a robot type, not a brand)."""
        assert "cyberdog" not in RobotContext.BRAND_ADAPTER_MAP

    def test_only_three_brand_entries(self):
        assert len(RobotContext.BRAND_ADAPTER_MAP) == 3


# ─── Section B: _FALLBACK_BRAND_MAP completeness ──────────────────────────────

class TestFallbackBrandMap:
    """_FALLBACK_BRAND_MAP must cover all supported robot types."""

    def test_unitree_robots_present(self):
        m = RobotContext._FALLBACK_BRAND_MAP
        for rt in ("go2", "a1", "b1", "b2", "h1"):
            assert m[rt] == "unitree", f"Expected {rt} → unitree"

    def test_spot_maps_to_bostiondynamics(self):
        assert RobotContext._FALLBACK_BRAND_MAP["spot"] == "bostiondynamics"

    def test_cyberdog_maps_to_xiaomi(self):
        assert RobotContext._FALLBACK_BRAND_MAP["cyberdog"] == "xiaomi"

    def test_cyberdog2_maps_to_xiaomi(self):
        assert RobotContext._FALLBACK_BRAND_MAP["cyberdog2"] == "xiaomi"

    def test_fallback_brand_robots_has_bostiondynamics(self):
        assert "bostiondynamics" in RobotContext._FALLBACK_BRAND_ROBOTS
        assert "spot" in RobotContext._FALLBACK_BRAND_ROBOTS["bostiondynamics"]

    def test_fallback_brand_robots_has_xiaomi(self):
        assert "xiaomi" in RobotContext._FALLBACK_BRAND_ROBOTS
        assert "cyberdog" in RobotContext._FALLBACK_BRAND_ROBOTS["xiaomi"]
        assert "cyberdog2" in RobotContext._FALLBACK_BRAND_ROBOTS["xiaomi"]


# ─── Section C: _create_adapter_for_brand factory ─────────────────────────────

class TestCreateAdapterForBrand:
    def setup_method(self):
        _reset()

    def test_unitree_returns_unitree_adapter(self):
        with patch.object(UnitreeAdapter, "__init__", return_value=None) as mock_init:
            adapter = RobotContext._create_adapter_for_brand("unitree", "go2")
        assert isinstance(adapter, UnitreeAdapter)
        mock_init.assert_called_once_with("go2")

    def test_bostiondynamics_returns_spot_adapter(self):
        with patch(
            "system.service.adapters.spot_sdk.adapter.SpotAdapter"
        ) as MockSpot:
            result = RobotContext._create_adapter_for_brand("bostiondynamics", "spot")
        MockSpot.assert_called_once_with()
        assert result is MockSpot.return_value

    def test_xiaomi_returns_cyberdog_adapter(self):
        with patch(
            "system.service.adapters.cyberdog_sdk.adapter.CyberDogAdapter"
        ) as MockCyber:
            result = RobotContext._create_adapter_for_brand("xiaomi", "cyberdog")
        MockCyber.assert_called_once_with()
        assert result is MockCyber.return_value

    def test_unknown_brand_returns_none(self):
        result = RobotContext._create_adapter_for_brand("nonexistent_brand", "robot1")
        assert result is None

    def test_unknown_brand_logs_warning(self):
        with patch("bin.core.robot_context.log_warning") as mock_warn:
            RobotContext._create_adapter_for_brand("nonexistent_brand", "robot1")
        mock_warn.assert_called_once()
        assert "nonexistent_brand" in mock_warn.call_args[0][0]


# ─── Section D: _ensure_adapter multi-brand ───────────────────────────────────

class TestEnsureAdapterMultiBrand:
    def setup_method(self):
        _reset()

    def _patch_factory(self, mock_adapter):
        return patch.object(
            RobotContext, "_create_adapter_for_brand", return_value=mock_adapter
        )

    def test_creates_and_registers_adapter_for_unitree(self):
        mock_adapter = _make_mock_adapter()
        with self._patch_factory(mock_adapter):
            result = RobotContext._ensure_adapter("unitree", "go2")
        assert result is mock_adapter
        registered = RobotContext._service_registry.get("unitree_sdk2")
        assert registered is mock_adapter

    def test_creates_and_registers_adapter_for_bostiondynamics(self):
        mock_adapter = _make_mock_adapter()
        with self._patch_factory(mock_adapter):
            result = RobotContext._ensure_adapter("bostiondynamics", "spot")
        assert result is mock_adapter
        registered = RobotContext._service_registry.get("spot_sdk")
        assert registered is mock_adapter

    def test_creates_and_registers_adapter_for_xiaomi(self):
        mock_adapter = _make_mock_adapter()
        with self._patch_factory(mock_adapter):
            result = RobotContext._ensure_adapter("xiaomi", "cyberdog")
        assert result is mock_adapter
        registered = RobotContext._service_registry.get("cyberdog_sdk")
        assert registered is mock_adapter

    def test_returns_none_when_factory_returns_none(self):
        with self._patch_factory(None):
            result = RobotContext._ensure_adapter("unknown_brand", "bot")
        assert result is None

    def test_returns_none_when_connect_raises(self):
        mock_adapter = _make_mock_adapter(connect_ok=False)
        with self._patch_factory(mock_adapter):
            result = RobotContext._ensure_adapter("unitree", "go2")
        assert result is None

    def test_reuses_registered_adapter_on_second_call(self):
        mock_adapter = _make_mock_adapter()
        with self._patch_factory(mock_adapter):
            RobotContext._ensure_adapter("unitree", "go2")
        # Factory returns a NEW mock, but the registered one should be reused
        mock_adapter2 = _make_mock_adapter()
        with self._patch_factory(mock_adapter2):
            result = RobotContext._ensure_adapter("unitree", "go2")
        # connect() called on the original mock (reuse path), not on mock_adapter2
        assert result is mock_adapter
        mock_adapter2.connect.assert_not_called()

    def test_factory_called_with_correct_brand_and_robot_type(self):
        mock_adapter = _make_mock_adapter()
        with patch.object(
            RobotContext, "_create_adapter_for_brand", return_value=mock_adapter
        ) as mock_factory:
            RobotContext._ensure_adapter("bostiondynamics", "spot")
        mock_factory.assert_called_once_with("bostiondynamics", "spot")

    def test_connect_called_with_robot_type_kwarg(self):
        mock_adapter = _make_mock_adapter()
        with self._patch_factory(mock_adapter):
            RobotContext._ensure_adapter("xiaomi", "cyberdog2")
        mock_adapter.connect.assert_called_once_with(
            robot_type="cyberdog2", force_reinit=False
        )

    def test_logs_error_when_factory_returns_none(self):
        with self._patch_factory(None):
            with patch("bin.core.robot_context.log_error") as mock_err:
                RobotContext._ensure_adapter("badBrand", "bot")
        mock_err.assert_called_once()
        assert "badBrand" in mock_err.call_args[0][0]


# ─── Section E: get_robot_model non-Unitree paths ─────────────────────────────

class TestGetRobotModelNonUnitree:
    def setup_method(self):
        _reset()

    def _patch_ensure(self, mock_adapter):
        return patch.object(
            RobotContext, "_ensure_adapter", return_value=mock_adapter
        )

    def test_non_unitree_adapter_no_get_model_returns_none_model(self):
        """Adapters without get_model() should not crash; model is None."""
        mock_adapter = MagicMock(spec=[])  # no get_model attribute
        mock_adapter.connect = MagicMock(return_value=True)
        with self._patch_ensure(mock_adapter):
            RobotContext._current_robot_type = "spot"
            model = RobotContext.get_robot_model()
        assert model is None

    def test_non_unitree_adapter_sets_initialized_true(self):
        """get_robot_model() must mark initialized=True even when model is None."""
        mock_adapter = MagicMock(spec=[])
        with self._patch_ensure(mock_adapter):
            RobotContext._current_robot_type = "spot"
            RobotContext.get_robot_model()
        assert RobotContext._initialized is True

    def test_ensure_adapter_none_returns_none(self):
        """If _ensure_adapter returns None, get_robot_model returns None."""
        with self._patch_ensure(None):
            model = RobotContext.get_robot_model()
        assert model is None

    def test_unitree_adapter_with_get_model_returns_model(self):
        """Adapters with get_model() should return the model object."""
        mock_model = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.get_model.return_value = mock_model
        with self._patch_ensure(mock_adapter):
            result = RobotContext.get_robot_model()
        assert result is mock_model
        assert RobotContext._current_robot_model is mock_model


# ─── Section F: run_action brand dispatch ─────────────────────────────────────

class TestRunActionBrandDispatch:
    def setup_method(self):
        _reset()

    def test_run_action_uses_correct_adapter_name_for_spot(self):
        """run_action for 'spot' robot should route through 'spot_sdk' adapter."""
        mock_adapter = _make_mock_adapter()
        with patch.object(RobotContext, "_ensure_adapter", return_value=mock_adapter):
            with patch.object(
                RobotContext, "_lifecycle_route", return_value=(True, True)
            ) as mock_route:
                RobotContext._current_robot_type = "spot"
                RobotContext.run_action("stand")
        # The adapter_name passed to _lifecycle_route must be "spot_sdk"
        mock_route.assert_called_once()
        _, adapter_name_arg = mock_route.call_args[0][:2]
        assert adapter_name_arg == "spot_sdk"

    def test_run_action_uses_correct_adapter_name_for_cyberdog(self):
        """run_action for 'cyberdog' robot should route through 'cyberdog_sdk'."""
        mock_adapter = _make_mock_adapter()
        with patch.object(RobotContext, "_ensure_adapter", return_value=mock_adapter):
            with patch.object(
                RobotContext, "_lifecycle_route", return_value=(True, True)
            ) as mock_route:
                RobotContext._current_robot_type = "cyberdog"
                RobotContext.run_action("sit")
        mock_route.assert_called_once()
        _, adapter_name_arg = mock_route.call_args[0][:2]
        assert adapter_name_arg == "cyberdog_sdk"

    def test_run_action_uses_correct_adapter_name_for_go2(self):
        """run_action for 'go2' should route through 'unitree_sdk2'."""
        mock_adapter = _make_mock_adapter()
        with patch.object(RobotContext, "_ensure_adapter", return_value=mock_adapter):
            with patch.object(
                RobotContext, "_lifecycle_route", return_value=(True, True)
            ) as mock_route:
                RobotContext._current_robot_type = "go2"
                RobotContext.run_action("walk")
        mock_route.assert_called_once()
        _, adapter_name_arg = mock_route.call_args[0][:2]
        assert adapter_name_arg == "unitree_sdk2"

    def test_run_action_returns_false_when_adapter_unavailable(self):
        with patch.object(RobotContext, "_ensure_adapter", return_value=None):
            result = RobotContext.run_action("stand")
        assert result is False

    def test_get_sensor_data_uses_correct_adapter_name_for_spot(self):
        mock_adapter = _make_mock_adapter()
        with patch.object(RobotContext, "_ensure_adapter", return_value=mock_adapter):
            with patch.object(
                RobotContext, "_lifecycle_route", return_value=(True, {"battery": 99})
            ) as mock_route:
                RobotContext._current_robot_type = "spot"
                result = RobotContext.get_sensor_data()
        mock_route.assert_called_once()
        _, adapter_name_arg = mock_route.call_args[0][:2]
        assert adapter_name_arg == "spot_sdk"
        assert result == {"battery": 99}

    def test_stop_uses_correct_adapter_name_for_cyberdog(self):
        mock_adapter = _make_mock_adapter()
        with patch.object(RobotContext, "_ensure_adapter", return_value=mock_adapter):
            with patch.object(
                RobotContext, "_lifecycle_route", return_value=(True, None)
            ) as mock_route:
                RobotContext._current_robot_type = "cyberdog"
                RobotContext.stop()
        mock_route.assert_called_once()
        _, adapter_name_arg = mock_route.call_args[0][:2]
        assert adapter_name_arg == "cyberdog_sdk"


# ─── Section G: Unknown-brand routing safety (STAGE-05 regression) ────────────

class TestUnknownBrandRoutingSafety:
    """Regression: unknown brand must NEVER silently route to a pre-registered
    unitree_sdk2 adapter.  All four public dispatch paths must fail fast."""

    def setup_method(self):
        _reset()

    def _pre_register_unitree(self):
        """Register a mock unitree_sdk2 adapter to prove it is NOT reused."""
        mock_unitree = _make_mock_adapter()
        RobotContext._service_registry.register("unitree_sdk2", mock_unitree)
        return mock_unitree

    def _patch_brand(self, brand: str):
        return patch.object(RobotContext, "get_current_brand", return_value=brand)

    # ── _ensure_adapter ──────────────────────────────────────────────────────

    def test_ensure_adapter_unknown_brand_returns_none(self):
        """_ensure_adapter must return None for any brand not in BRAND_ADAPTER_MAP."""
        result = RobotContext._ensure_adapter("completely_unknown_brand", "bot1")
        assert result is None

    def test_ensure_adapter_unknown_brand_does_not_reuse_registered_unitree(self):
        """Even when unitree_sdk2 is pre-registered, unknown brand → None, NOT Unitree."""
        unitree_mock = self._pre_register_unitree()
        result = RobotContext._ensure_adapter("completely_unknown_brand", "bot1")
        assert result is None
        unitree_mock.connect.assert_not_called()

    def test_ensure_adapter_logs_error_for_unknown_brand(self):
        with patch("bin.core.robot_context.log_error") as mock_err:
            RobotContext._ensure_adapter("completely_unknown_brand", "bot1")
        mock_err.assert_called_once()
        msg = mock_err.call_args[0][0]
        assert "completely_unknown_brand" in msg

    # ── run_action ───────────────────────────────────────────────────────────

    def test_run_action_returns_false_for_unknown_brand(self):
        with self._patch_brand("completely_unknown_brand"):
            result = RobotContext.run_action("stand")
        assert result is False

    def test_run_action_unknown_brand_does_not_route_to_registered_unitree(self):
        """Pre-registered Unitree adapter must NOT be called for unknown brand."""
        unitree_mock = self._pre_register_unitree()
        with self._patch_brand("completely_unknown_brand"):
            result = RobotContext.run_action("stand")
        assert result is False
        unitree_mock.connect.assert_not_called()
        unitree_mock.run_action.assert_not_called()

    def test_run_action_unknown_brand_logs_error(self):
        with self._patch_brand("completely_unknown_brand"):
            with patch("bin.core.robot_context.log_error") as mock_err:
                RobotContext.run_action("stand")
        assert any("completely_unknown_brand" in str(c) for c in mock_err.call_args_list)

    # ── get_sensor_data ──────────────────────────────────────────────────────

    def test_get_sensor_data_returns_error_dict_for_unknown_brand(self):
        with self._patch_brand("completely_unknown_brand"):
            result = RobotContext.get_sensor_data()
        assert isinstance(result, dict)
        assert "error" in result

    def test_get_sensor_data_error_dict_mentions_brand(self):
        with self._patch_brand("completely_unknown_brand"):
            result = RobotContext.get_sensor_data()
        assert "completely_unknown_brand" in result["error"]

    def test_get_sensor_data_unknown_brand_does_not_route_to_registered_unitree(self):
        unitree_mock = self._pre_register_unitree()
        with self._patch_brand("completely_unknown_brand"):
            result = RobotContext.get_sensor_data()
        assert "error" in result
        unitree_mock.connect.assert_not_called()

    # ── stop ─────────────────────────────────────────────────────────────────

    def test_stop_is_noop_for_unknown_brand(self):
        """stop() must not raise for unknown brand."""
        with self._patch_brand("completely_unknown_brand"):
            RobotContext.stop()  # must not raise

    def test_stop_returns_none_for_unknown_brand(self):
        with self._patch_brand("completely_unknown_brand"):
            result = RobotContext.stop()
        assert result is None

    def test_stop_unknown_brand_does_not_route_to_registered_unitree(self):
        unitree_mock = self._pre_register_unitree()
        with self._patch_brand("completely_unknown_brand"):
            RobotContext.stop()
        unitree_mock.connect.assert_not_called()
        unitree_mock.run_action.assert_not_called()
