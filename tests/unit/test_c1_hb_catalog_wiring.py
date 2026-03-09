#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 1.4 compliance gap fix — HB catalog wiring.

Verifies that _refresh_capability_inspector() propagates capability data
to BehaviorPanel.set_capability_profile() on every code path, without
requiring a live Qt application or robot SDK.

Coverage
--------
_update_hb_catalog()
  - success: delegates to behavior_panel.set_capability_profile(cap_dict)
  - exception from set_capability_profile is suppressed (never propagates)

_refresh_capability_inspector() wiring
  - path A (adapter=None): set_capability_profile called with {}
  - path B (adapter fetch raises): set_capability_profile called with {}
  - path C (success): set_capability_profile called with real cap_dict
  - failure in set_capability_profile does not raise from _refresh_capability_inspector

Isolation
---------
- No QApplication. No live robot SDK.
- MainWindow constructor is NOT invoked; we build a minimal stub.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Minimal stub that mimics only the parts of MainWindow we exercise
# ---------------------------------------------------------------------------

class _FakeMainZone:
    """Stub for MainWindow.main_zone."""

    def __init__(self):
        self.behavior_panel = MagicMock()
        self.behavior_panel.set_capability_profile = MagicMock()
        # run_compat_audit returns a clean HBCompatReport so _run_compat_audit completes
        from system.behavior.hb_compat_audit import HBCompatReport
        self.behavior_panel.run_compat_audit = MagicMock(return_value=HBCompatReport("", ""))

    def set_adapter_capabilities(self, cap_dict, missing):
        pass  # no-op — not under test here

    def get_sdk_settings(self):
        return {}

    def show_compat_report(self, report, diag_dicts=None):
        pass  # no-op — not under test here


def _make_stub(adapter=None, adapter_raises=False):
    """Return a stub object that exposes _refresh_capability_inspector and
    _update_hb_catalog bound as unbound helper (mirroring MainWindow behaviour).

    We import the two methods from bin.ui lazily and bind them to the stub so
    we never instantiate MainWindow.
    """
    # Lazily import the real method implementations
    import bin.ui as ui_mod

    stub = types.SimpleNamespace()
    stub.main_zone = _FakeMainZone()

    # Bind real methods to stub
    stub._update_hb_catalog = lambda cap: ui_mod.MainWindow._update_hb_catalog(stub, cap)
    stub._refresh_capability_inspector = lambda: ui_mod.MainWindow._refresh_capability_inspector(stub)

    # Patch RobotContext calls and adapter behaviour
    stub._adapter = adapter
    stub._adapter_raises = adapter_raises
    return stub


# ===========================================================================
# _update_hb_catalog — unit tests
# ===========================================================================

class TestUpdateHBCatalog(unittest.TestCase):
    """Direct unit tests for MainWindow._update_hb_catalog."""

    def _make(self):
        import bin.ui as ui_mod
        stub = types.SimpleNamespace()
        stub.main_zone = _FakeMainZone()
        # Bind the real method
        stub._update_hb_catalog = lambda cap: ui_mod.MainWindow._update_hb_catalog(stub, cap)
        return stub

    def test_calls_set_capability_profile_with_cap_dict(self):
        stub = self._make()
        cap = {"brand": "unitree", "sensors": ["imu"], "flags": {}}
        stub._update_hb_catalog(cap)
        stub.main_zone.behavior_panel.set_capability_profile.assert_called_once_with(cap)

    def test_calls_set_capability_profile_with_empty_dict(self):
        stub = self._make()
        stub._update_hb_catalog({})
        stub.main_zone.behavior_panel.set_capability_profile.assert_called_once_with({})

    def test_exception_in_set_capability_profile_is_suppressed(self):
        stub = self._make()
        stub.main_zone.behavior_panel.set_capability_profile.side_effect = RuntimeError("boom")
        # Must not raise
        try:
            stub._update_hb_catalog({"brand": "x"})
        except Exception as exc:
            self.fail(f"_update_hb_catalog raised unexpectedly: {exc}")

    def test_exception_logged_not_silently_eaten(self):
        """log_warning should be called when set_capability_profile raises."""
        import bin.ui as ui_mod
        stub = self._make()
        stub.main_zone.behavior_panel.set_capability_profile.side_effect = ValueError("bad")
        with patch("bin.ui.log_warning") as mock_warn:
            stub._update_hb_catalog({})
            mock_warn.assert_called_once()
            warn_msg = mock_warn.call_args[0][0]
            self.assertIn("hb_catalog", warn_msg)


# ===========================================================================
# _refresh_capability_inspector wiring — paths A / B / C
# ===========================================================================

_UNITREE_CAPS = {
    "brand": "unitree",
    "robot_type": "go2",
    "sensors": ["imu", "qpos", "qvel", "foot_contact", "battery"],
    "flags": {"high_level_control": True},
    "required_settings": [],
}


def _patch_robot_context(brand="unitree", robot_type="go2"):
    """Return a context manager that patches RobotContext helpers."""
    import contextlib

    @contextlib.contextmanager
    def _cm(adapter_obj=None, create_raises=False):
        def _create(b, rt):
            if create_raises:
                raise RuntimeError("adapter fail")
            return adapter_obj

        with patch("bin.ui.RobotContext.get_current_brand", return_value=brand), \
             patch("bin.ui.RobotContext.get_robot_type", return_value=robot_type), \
             patch("bin.ui.RobotContext._create_adapter_for_brand", side_effect=_create):
            yield

    return _cm


class TestRefreshCapabilityInspectorWiring(unittest.TestCase):
    """Verify that _refresh_capability_inspector propagates caps to BehaviorPanel."""

    def _make_stub(self):
        import bin.ui as ui_mod
        stub = types.SimpleNamespace()
        stub.main_zone = _FakeMainZone()
        stub._update_hb_catalog = lambda cap: ui_mod.MainWindow._update_hb_catalog(stub, cap)
        stub._run_compat_audit  = lambda b, rt: ui_mod.MainWindow._run_compat_audit(stub, b, rt)
        stub._update_hb_channel = MagicMock()  # Circle 2: no-op for catalog-wiring tests
        stub._refresh_capability_inspector = lambda: ui_mod.MainWindow._refresh_capability_inspector(stub)
        return stub

    # ------------------------------------------------------------------ path A
    def test_path_a_adapter_none_calls_set_capability_profile_empty(self):
        """When adapter is None, set_capability_profile({}) is called."""
        stub = self._make_stub()
        cm = _patch_robot_context()
        with cm(adapter_obj=None):
            stub._refresh_capability_inspector()
        stub.main_zone.behavior_panel.set_capability_profile.assert_called_once_with({})

    # ------------------------------------------------------------------ path B
    def test_path_b_adapter_raises_calls_set_capability_profile_empty(self):
        """When adapter fetch raises, set_capability_profile({}) is called."""
        stub = self._make_stub()
        cm = _patch_robot_context()
        with cm(create_raises=True):
            stub._refresh_capability_inspector()
        stub.main_zone.behavior_panel.set_capability_profile.assert_called_once_with({})

    # ------------------------------------------------------------------ path C
    def test_path_c_success_calls_set_capability_profile_with_cap_dict(self):
        """On success, set_capability_profile is called with the real cap_dict."""
        stub = self._make_stub()
        mock_adapter = MagicMock()
        mock_adapter.capabilities.return_value = dict(_UNITREE_CAPS)

        cm = _patch_robot_context()
        with cm(adapter_obj=mock_adapter), \
             patch("bin.core.settings_form.compute_missing_settings", return_value=[]):
            stub._refresh_capability_inspector()

        stub.main_zone.behavior_panel.set_capability_profile.assert_called_once()
        passed_cap = stub.main_zone.behavior_panel.set_capability_profile.call_args[0][0]
        self.assertEqual(passed_cap["brand"], "unitree")

    def test_path_c_set_capability_profile_receives_full_cap_dict(self):
        """Cap dict passed to set_capability_profile matches adapter.capabilities()."""
        stub = self._make_stub()
        mock_adapter = MagicMock()
        mock_adapter.capabilities.return_value = dict(_UNITREE_CAPS)

        cm = _patch_robot_context()
        with cm(adapter_obj=mock_adapter), \
             patch("bin.core.settings_form.compute_missing_settings", return_value=[]):
            stub._refresh_capability_inspector()

        passed_cap = stub.main_zone.behavior_panel.set_capability_profile.call_args[0][0]
        self.assertIn("imu", passed_cap.get("sensors", []))

    # ------------------------------------------------------------------ fault tolerance
    def test_set_capability_profile_failure_does_not_propagate_on_path_c(self):
        """Exception from set_capability_profile is swallowed on success path."""
        stub = self._make_stub()
        stub.main_zone.behavior_panel.set_capability_profile.side_effect = RuntimeError("panel err")
        mock_adapter = MagicMock()
        mock_adapter.capabilities.return_value = dict(_UNITREE_CAPS)

        cm = _patch_robot_context()
        try:
            with cm(adapter_obj=mock_adapter), \
                 patch("bin.core.settings_form.compute_missing_settings", return_value=[]):
                stub._refresh_capability_inspector()
        except Exception as exc:
            self.fail(f"_refresh_capability_inspector propagated: {exc}")

    def test_path_b_set_capability_profile_failure_does_not_propagate(self):
        """Exception from set_capability_profile is swallowed on exception path."""
        stub = self._make_stub()
        stub.main_zone.behavior_panel.set_capability_profile.side_effect = RuntimeError("panel err")
        cm = _patch_robot_context()
        try:
            with cm(create_raises=True):
                stub._refresh_capability_inspector()
        except Exception as exc:
            self.fail(f"_refresh_capability_inspector propagated: {exc}")

    # ------------------------------------------------------------------ Issue 7: audit on ALL paths
    def test_path_a_run_compat_audit_called(self):
        """Issue 7: _run_compat_audit must fire even when adapter is None."""
        stub = self._make_stub()
        stub._run_compat_audit = MagicMock()
        stub._refresh_capability_inspector = lambda: __import__("bin.ui", fromlist=["MainWindow"]).MainWindow._refresh_capability_inspector(stub)
        cm = _patch_robot_context()
        with cm(adapter_obj=None):
            stub._refresh_capability_inspector()
        stub._run_compat_audit.assert_called_once()

    def test_path_b_run_compat_audit_called(self):
        """Issue 7: _run_compat_audit must fire even when adapter fetch raises."""
        stub = self._make_stub()
        stub._run_compat_audit = MagicMock()
        stub._refresh_capability_inspector = lambda: __import__("bin.ui", fromlist=["MainWindow"]).MainWindow._refresh_capability_inspector(stub)
        cm = _patch_robot_context()
        with cm(create_raises=True):
            stub._refresh_capability_inspector()
        stub._run_compat_audit.assert_called_once()

    def test_path_c_run_compat_audit_called(self):
        """Issue 7: _run_compat_audit must fire on the success path."""
        stub = self._make_stub()
        stub._run_compat_audit = MagicMock()
        stub._refresh_capability_inspector = lambda: __import__("bin.ui", fromlist=["MainWindow"]).MainWindow._refresh_capability_inspector(stub)
        mock_adapter = MagicMock()
        mock_adapter.capabilities.return_value = dict(_UNITREE_CAPS)
        cm = _patch_robot_context()
        with cm(adapter_obj=mock_adapter), \
             patch("bin.core.settings_form.compute_missing_settings", return_value=[]):
            stub._refresh_capability_inspector()
        stub._run_compat_audit.assert_called_once()

    # ------------------------------------------------------------------ call count invariant
    def test_set_capability_profile_called_exactly_once_per_refresh(self):
        """set_capability_profile is called exactly once per refresh call."""
        stub = self._make_stub()
        mock_adapter = MagicMock()
        mock_adapter.capabilities.return_value = dict(_UNITREE_CAPS)

        cm = _patch_robot_context()
        with cm(adapter_obj=mock_adapter), \
             patch("bin.core.settings_form.compute_missing_settings", return_value=[]):
            stub._refresh_capability_inspector()

        self.assertEqual(
            stub.main_zone.behavior_panel.set_capability_profile.call_count, 1
        )


if __name__ == "__main__":
    unittest.main()
