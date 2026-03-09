"""Tests for Cycle 2 STAGE-04 capability refresh wiring.

Coverage:

    A. compute_missing_settings — unit tests for the helper in settings_form
    B. Real adapter capabilities — capability dicts contain expected keys
    C. _refresh_capability_inspector — updates inspector from live adapter data
    D. Failure degradation — adapter/capability errors do not crash or raise
    E. Brand switch triggers — set_sdk_brand + inspector refresh on robot change
    F. Settings-change triggers — refresh on settings_applied / settings_reset
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from bin.core.settings_form import compute_missing_settings
from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
from system.service.adapters.spot_sdk.adapter import SpotAdapter
from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter


# ── A. compute_missing_settings ───────────────────────────────────────────────

class TestComputeMissingSettings(unittest.TestCase):
    """Unit tests for compute_missing_settings()."""

    # ── absent keys ──────────────────────────────────────────────────────────

    def test_absent_key_is_missing(self):
        missing = compute_missing_settings("unitree", {}, ["brand"])
        self.assertIn("brand", missing)

    def test_empty_required_list_returns_empty(self):
        self.assertEqual(compute_missing_settings("unitree", {}, []), [])

    # ── empty-string text/choice ──────────────────────────────────────────────

    def test_empty_string_text_field_is_missing(self):
        # "robot_type" is a choice field; empty string = missing
        config = {"brand": ""}
        missing = compute_missing_settings("unitree", config, ["brand"])
        self.assertIn("brand", missing)

    def test_nonempty_string_text_field_is_not_missing(self):
        config = {"brand": "unitree"}
        missing = compute_missing_settings("unitree", config, ["brand"])
        self.assertNotIn("brand", missing)

    # ── empty list ────────────────────────────────────────────────────────────

    def test_empty_list_field_is_missing(self):
        # "action_endpoints" is a list field in the xiaomi schema
        config = {"action_endpoints": []}
        missing = compute_missing_settings("xiaomi", config, ["action_endpoints"])
        self.assertIn("action_endpoints", missing)

    def test_nonempty_list_field_is_not_missing(self):
        config = {"action_endpoints": ["192.168.1.1"]}
        missing = compute_missing_settings("xiaomi", config, ["action_endpoints"])
        self.assertNotIn("action_endpoints", missing)

    # ── int / float / bool — zero / False are valid ───────────────────────────

    def test_zero_integer_not_missing(self):
        config = {"timeout": 0}
        missing = compute_missing_settings("unitree", config, ["timeout"])
        self.assertNotIn("timeout", missing)

    def test_zero_float_not_missing(self):
        config = {"some_float": 0.0}
        missing = compute_missing_settings("unitree", config, ["some_float"])
        self.assertNotIn("some_float", missing)

    def test_false_bool_not_missing(self):
        config = {"safety_enabled": False}
        missing = compute_missing_settings("unitree", config, ["safety_enabled"])
        self.assertNotIn("safety_enabled", missing)

    # ── multiple fields ───────────────────────────────────────────────────────

    def test_mixed_missing_and_present(self):
        config = {"brand": "unitree", "robot_type": ""}
        missing = compute_missing_settings(
            "unitree", config, ["brand", "robot_type", "timeout"]
        )
        self.assertNotIn("brand", missing)       # "unitree" ≠ ""
        self.assertIn("robot_type", missing)     # "" → missing
        self.assertIn("timeout", missing)        # absent → missing

    def test_preserves_order_of_required_settings(self):
        config = {}
        required = ["timeout", "brand", "safety_enabled"]
        missing = compute_missing_settings("unitree", config, required)
        self.assertEqual(missing, ["timeout", "brand", "safety_enabled"])

    # ── unknown brand degrades gracefully ─────────────────────────────────────

    def test_unknown_brand_does_not_crash(self):
        config = {"x": ""}
        missing = compute_missing_settings("nonexistent_brand_xyz", config, ["x"])
        self.assertIn("x", missing)

    # ── real adapters' required_settings pass the computation ─────────────────

    def test_unitree_all_required_settings_absent_returns_all(self):
        cap = UnitreeAdapter().capabilities()
        required = cap.get("required_settings", [])
        missing = compute_missing_settings("unitree", {}, required)
        self.assertEqual(missing, required)

    def test_spot_all_required_settings_absent_returns_all(self):
        cap = SpotAdapter().capabilities()
        required = cap.get("required_settings", [])
        missing = compute_missing_settings("bostiondynamics", {}, required)
        self.assertEqual(missing, required)

    def test_cyberdog_all_required_settings_absent_returns_all(self):
        cap = CyberDogAdapter().capabilities()
        required = cap.get("required_settings", [])
        missing = compute_missing_settings("xiaomi", {}, required)
        self.assertEqual(missing, required)


# ── B. Real adapter capabilities ──────────────────────────────────────────────

class TestRealAdapterCapabilities(unittest.TestCase):
    """Verify real adapter capability dicts have the expected structure."""

    def _check_cap(self, cap: dict) -> None:
        self.assertIsInstance(cap, dict)
        self.assertIn("brand", cap)
        self.assertIn("adapter", cap)
        for key in ("actions", "sensors"):
            self.assertIsInstance(cap.get(key, []), list)
        self.assertIsInstance(cap.get("flags", {}), dict)
        self.assertIsInstance(cap.get("required_settings", []), list)

    def test_unitree_capabilities_structure(self):
        self._check_cap(UnitreeAdapter().capabilities())

    def test_spot_capabilities_structure(self):
        self._check_cap(SpotAdapter().capabilities())

    def test_cyberdog_capabilities_structure(self):
        self._check_cap(CyberDogAdapter().capabilities())

    def test_unitree_brand_is_unitree(self):
        cap = UnitreeAdapter().capabilities()
        self.assertEqual(cap["brand"], "unitree")

    def test_spot_brand_is_bostiondynamics(self):
        cap = SpotAdapter().capabilities()
        self.assertEqual(cap["brand"], "bostiondynamics")

    def test_cyberdog_brand_is_xiaomi(self):
        cap = CyberDogAdapter().capabilities()
        self.assertEqual(cap["brand"], "xiaomi")


# ── C. _refresh_capability_inspector ─────────────────────────────────────────

class TestRefreshCapabilityInspector(unittest.TestCase):
    """Integration: MainWindow._refresh_capability_inspector() populates inspector."""

    def _make_window(self):
        """Create a minimal MainWindow for testing."""
        from bin.core.config_manager import ConfigManager
        from bin.ui import MainWindow
        config = ConfigManager()
        return MainWindow(config)

    def test_refresh_populates_inspector_brand(self):
        win = self._make_window()
        win._refresh_capability_inspector()
        header_text = win.main_zone.capability_inspector._header_label.text()
        # Should contain a brand name, not the default "Capability Inspector"
        self.assertNotEqual(header_text, "Capability Inspector")

    def test_refresh_populates_with_unitree_by_default(self):
        win = self._make_window()
        win._refresh_capability_inspector()
        header_text = win.main_zone.capability_inspector._header_label.text()
        self.assertIn("unitree", header_text.lower())

    def test_refresh_does_not_raise_on_clean_state(self):
        win = self._make_window()
        try:
            win._refresh_capability_inspector()
        except Exception as exc:
            self.fail(f"_refresh_capability_inspector raised: {exc}")

    def test_missing_settings_computed_and_passed(self):
        """Capability inspector receives a list (possibly empty) for missing."""
        win = self._make_window()
        win._refresh_capability_inspector()
        inspector = win.main_zone.capability_inspector
        # After refresh, _missing should be a list (not None)
        self.assertIsInstance(inspector._missing, list)


# ── D. Failure degradation ────────────────────────────────────────────────────

class TestRefreshFailureDegradation(unittest.TestCase):
    """Adapter/capability errors must not propagate or crash."""

    def _make_window(self):
        from bin.core.config_manager import ConfigManager
        from bin.ui import MainWindow
        return MainWindow(ConfigManager())

    def test_adapter_factory_returns_none_does_not_crash(self):
        win = self._make_window()
        with patch(
            "bin.ui.RobotContext._create_adapter_for_brand",
            return_value=None,
        ):
            try:
                win._refresh_capability_inspector()
            except Exception as exc:
                self.fail(f"Should not raise when adapter is None: {exc}")

    def test_adapter_capabilities_raises_does_not_crash(self):
        win = self._make_window()
        bad_adapter = MagicMock()
        bad_adapter.capabilities.side_effect = RuntimeError("SDK not available")
        with patch(
            "bin.ui.RobotContext._create_adapter_for_brand",
            return_value=bad_adapter,
        ):
            try:
                win._refresh_capability_inspector()
            except Exception as exc:
                self.fail(f"Should not raise when capabilities() raises: {exc}")

    def test_set_adapter_capabilities_raises_does_not_crash(self):
        win = self._make_window()
        with patch.object(
            win.main_zone,
            "set_adapter_capabilities",
            side_effect=RuntimeError("widget error"),
        ):
            try:
                win._refresh_capability_inspector()
            except Exception as exc:
                self.fail(f"Should not raise when set_adapter_capabilities raises: {exc}")

    def test_inspector_state_unchanged_when_adapter_unavailable(self):
        """Inspector keeps previous content when adapter fetch fails."""
        win = self._make_window()
        # Populate once with real data
        win._refresh_capability_inspector()
        header_before = win.main_zone.capability_inspector._header_label.text()

        with patch(
            "bin.ui.RobotContext._create_adapter_for_brand",
            return_value=None,
        ):
            win._refresh_capability_inspector()  # should be no-op

        # Header unchanged
        self.assertEqual(
            win.main_zone.capability_inspector._header_label.text(),
            header_before,
        )


# ── E. Brand switch triggers ──────────────────────────────────────────────────

class TestBrandSwitchTrigger(unittest.TestCase):
    """_on_robot_type_changed must sync settings panel brand + refresh inspector."""

    def _make_window(self):
        from bin.core.config_manager import ConfigManager
        from bin.ui import MainWindow
        return MainWindow(ConfigManager())

    def test_brand_switch_updates_capability_inspector(self):
        win = self._make_window()
        # Start with unitree (go2 default) — already refreshed at startup
        # Switch to spot
        win._on_robot_type_changed("spot")
        header = win.main_zone.capability_inspector._header_label.text()
        # Should now show bostiondynamics / spot brand
        self.assertTrue(
            "bostiondynamics" in header.lower() or "spot" in header.lower(),
            f"Expected spot brand in header after switch: {header!r}",
        )

    def test_brand_switch_updates_settings_panel_brand(self):
        win = self._make_window()
        win._on_robot_type_changed("spot")
        # SettingsPanel title should reflect bostiondynamics
        title = win.main_zone.settings_panel._title_label.text()
        self.assertIn("bostiondynamics", title)

    def test_brand_switch_to_cyberdog(self):
        win = self._make_window()
        win._on_robot_type_changed("cyberdog")
        header = win.main_zone.capability_inspector._header_label.text()
        self.assertTrue(
            "xiaomi" in header.lower() or "cyberdog" in header.lower(),
            f"Expected xiaomi/cyberdog brand in header: {header!r}",
        )


# ── F. Settings-change triggers ───────────────────────────────────────────────

class TestSettingsChangeTriggers(unittest.TestCase):
    """settings_applied and settings_reset signals must trigger inspector refresh."""

    def _make_window(self):
        from bin.core.config_manager import ConfigManager
        from bin.ui import MainWindow
        return MainWindow(ConfigManager())

    def test_settings_applied_triggers_refresh(self):
        win = self._make_window()
        refresh_calls = []
        original = win._refresh_capability_inspector
        win._refresh_capability_inspector = lambda: refresh_calls.append(1) or original()

        # Simulate settings_applied emission
        win.main_zone.settings_panel.settings_applied.emit({})
        self.assertGreater(len(refresh_calls), 0, "Refresh should be called on settings_applied")

    def test_settings_reset_triggers_refresh(self):
        win = self._make_window()
        refresh_calls = []
        original = win._refresh_capability_inspector
        win._refresh_capability_inspector = lambda: refresh_calls.append(1) or original()

        # Simulate settings_reset emission
        win.main_zone.settings_panel.settings_reset.emit()
        self.assertGreater(len(refresh_calls), 0, "Refresh should be called on settings_reset")

    def test_on_sdk_settings_changed_calls_refresh(self):
        win = self._make_window()
        called = []
        win._refresh_capability_inspector = lambda: called.append(True)
        win._on_sdk_settings_changed({"brand": "unitree"})
        self.assertTrue(called)

    def test_on_sdk_settings_changed_no_args_calls_refresh(self):
        win = self._make_window()
        called = []
        win._refresh_capability_inspector = lambda: called.append(True)
        win._on_sdk_settings_changed()
        self.assertTrue(called)


if __name__ == "__main__":
    unittest.main()
