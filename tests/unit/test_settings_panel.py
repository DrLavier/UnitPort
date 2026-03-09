"""Tests for bin.components.settings_panel — Cycle 2 STAGE-03.

Coverage:

    A. Instantiation — all three brands, default brand fallback
    B. get_current_config — keys, value types, all schema keys present
    C. has_unsaved_changes — False at rest, True after widget mutation
    D. Apply flow — success path (signal, applied_config updated)
    E. Apply flow — validation error path (error labels shown, signal NOT emitted)
    F. Apply flow — error labels cleared on subsequent successful Apply
    G. Reset flow — widget values restored, error labels cleared, signal emitted
    H. set_brand — form rebuilt, field set matches new brand schema
    I. Widget type mapping — QComboBox/QCheckBox/QSpinBox/QLineEdit per type
    J. Required / Optional grouping — QGroupBox titles
    K. Initial config pre-population — values loaded from constructor arg
    L. MainZonePanel integration — settings_panel attribute exists, tab present
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QSpinBox,
)

_app = QApplication.instance() or QApplication(sys.argv)

from bin.components.settings_panel import SettingsPanel
from system.service.settings_schema import KNOWN_BRANDS, get_settings_schema


# ── Helpers ──────────────────────────────────────────────────────────────────

def _panel(brand: str = "unitree", config: dict = None) -> SettingsPanel:
    return SettingsPanel(brand, config or {})


def _find_groupboxes(panel: SettingsPanel):
    return panel.findChildren(QGroupBox)


def _groupbox_titles(panel: SettingsPanel):
    return {gb.title() for gb in _find_groupboxes(panel)}


# ─────────────────────────────────────────────────────────────────────────────
# A. Instantiation
# ─────────────────────────────────────────────────────────────────────────────

class TestInstantiation(unittest.TestCase):
    def test_unitree_instantiates(self):
        p = _panel("unitree")
        self.assertIsInstance(p, SettingsPanel)

    def test_bostiondynamics_instantiates(self):
        p = _panel("bostiondynamics")
        self.assertIsInstance(p, SettingsPanel)

    def test_xiaomi_instantiates(self):
        p = _panel("xiaomi")
        self.assertIsInstance(p, SettingsPanel)

    def test_unknown_brand_falls_back_to_first_known(self):
        p = SettingsPanel("nonexistent_brand")
        self.assertEqual(p._brand, KNOWN_BRANDS[0])

    def test_title_label_contains_brand(self):
        p = _panel("unitree")
        self.assertIn("unitree", p._title_label.text())


# ─────────────────────────────────────────────────────────────────────────────
# B. get_current_config
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCurrentConfig(unittest.TestCase):
    def setUp(self):
        self.panel = _panel("unitree")

    def test_returns_dict(self):
        self.assertIsInstance(self.panel.get_current_config(), dict)

    def test_all_schema_keys_present(self):
        schema_keys = set(get_settings_schema("unitree").keys())
        config_keys = set(self.panel.get_current_config().keys())
        self.assertEqual(config_keys, schema_keys)

    def test_choice_field_returns_string(self):
        cfg = self.panel.get_current_config()
        self.assertIsInstance(cfg["brand"], str)
        self.assertIsInstance(cfg["connection_mode"], str)

    def test_bool_field_returns_bool(self):
        cfg = self.panel.get_current_config()
        self.assertIsInstance(cfg["safety_enabled"], bool)

    def test_integer_field_returns_int(self):
        cfg = self.panel.get_current_config()
        self.assertIsInstance(cfg["timeout"], int)
        self.assertIsInstance(cfg["dds_domain_id"], int)

    def test_list_field_returns_list_for_xiaomi(self):
        p = _panel("xiaomi")
        cfg = p.get_current_config()
        self.assertIsInstance(cfg["action_endpoints"], list)


# ─────────────────────────────────────────────────────────────────────────────
# C. has_unsaved_changes
# ─────────────────────────────────────────────────────────────────────────────

class TestHasUnsavedChanges(unittest.TestCase):
    def test_false_when_nothing_changed(self):
        p = _panel("unitree")
        self.assertFalse(p.has_unsaved_changes())

    def test_true_after_spinbox_change(self):
        p = _panel("unitree")
        spin = p._field_widgets.get("timeout")
        self.assertIsNotNone(spin)
        spin.setValue(99)
        self.assertTrue(p.has_unsaved_changes())

    def test_false_after_apply(self):
        p = _panel("unitree")
        # Simulate a change then apply
        p._field_widgets["timeout"].setValue(42)
        p._on_apply()
        self.assertFalse(p.has_unsaved_changes())


# ─────────────────────────────────────────────────────────────────────────────
# D. Apply — success path
# ─────────────────────────────────────────────────────────────────────────────

class TestApplySuccess(unittest.TestCase):
    def test_settings_applied_signal_emitted(self):
        p = _panel("unitree")
        emitted = []
        p.settings_applied.connect(emitted.append)
        p._on_apply()
        self.assertEqual(len(emitted), 1)
        self.assertIsInstance(emitted[0], dict)

    def test_applied_config_updated_after_apply(self):
        p = _panel("unitree")
        p._field_widgets["timeout"].setValue(77)
        p._on_apply()
        self.assertEqual(p._applied_config["timeout"], 77)

    def test_status_label_shows_applied(self):
        p = _panel("unitree")
        p._on_apply()
        self.assertIn("applied", p._status_label.text().lower())

    def test_applied_config_matches_get_current_config(self):
        p = _panel("unitree")
        p._on_apply()
        self.assertEqual(p._applied_config, p.get_current_config())


# ─────────────────────────────────────────────────────────────────────────────
# E. Apply — validation error path (mocked)
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_ERROR = {
    "status": "error",
    "brand": "unitree",
    "errors": ["required setting missing: 'robot_type'",
               "'timeout' invalid value"],
    "missing": ["robot_type"],
    "invalid": ["timeout"],
}


class TestApplyValidationError(unittest.TestCase):
    def _apply_with_error(self):
        p = _panel("unitree")
        emitted = []
        p.settings_applied.connect(emitted.append)
        with patch("bin.components.settings_panel.validate_settings",
                   return_value=_MOCK_ERROR):
            p._on_apply()
        return p, emitted

    def test_signal_not_emitted_on_error(self):
        _, emitted = self._apply_with_error()
        self.assertEqual(emitted, [])

    def test_missing_field_error_label_shown(self):
        p, _ = self._apply_with_error()
        lbl = p._error_labels.get("robot_type")
        self.assertIsNotNone(lbl)
        self.assertFalse(lbl.isHidden(), "error label should not be hidden")
        self.assertTrue(lbl.text())

    def test_invalid_field_error_label_shown(self):
        p, _ = self._apply_with_error()
        lbl = p._error_labels.get("timeout")
        self.assertIsNotNone(lbl)
        self.assertFalse(lbl.isHidden(), "error label should not be hidden")
        self.assertTrue(lbl.text())

    def test_status_label_shows_error_summary(self):
        p, _ = self._apply_with_error()
        self.assertTrue(p._status_label.text())


# ─────────────────────────────────────────────────────────────────────────────
# F. Apply — error labels cleared on subsequent success
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyErrorClearedOnSuccess(unittest.TestCase):
    def test_error_labels_hidden_after_successful_apply(self):
        p = _panel("unitree")
        # First apply → error
        with patch("bin.components.settings_panel.validate_settings",
                   return_value=_MOCK_ERROR):
            p._on_apply()
        # Confirm error label is shown (not hidden)
        self.assertFalse(p._error_labels["robot_type"].isHidden())
        # Second apply → success (real validator)
        p._on_apply()
        # All error labels should be hidden again
        for lbl in p._error_labels.values():
            self.assertTrue(lbl.isHidden())


# ─────────────────────────────────────────────────────────────────────────────
# G. Reset flow
# ─────────────────────────────────────────────────────────────────────────────

class TestResetFlow(unittest.TestCase):
    def test_reset_emits_settings_reset_signal(self):
        p = _panel("unitree")
        emitted = []
        p.settings_reset.connect(lambda: emitted.append(True))
        p._on_reset()
        self.assertEqual(len(emitted), 1)

    def test_reset_restores_applied_config_values(self):
        p = _panel("unitree")
        # Apply with timeout=55
        p._field_widgets["timeout"].setValue(55)
        p._on_apply()
        # Change timeout to 99
        p._field_widgets["timeout"].setValue(99)
        self.assertEqual(p.get_current_config()["timeout"], 99)
        # Reset → should revert to 55
        p._on_reset()
        self.assertEqual(p.get_current_config()["timeout"], 55)

    def test_reset_clears_error_labels(self):
        p = _panel("unitree")
        with patch("bin.components.settings_panel.validate_settings",
                   return_value=_MOCK_ERROR):
            p._on_apply()
        self.assertFalse(p._error_labels["robot_type"].isHidden())
        p._on_reset()
        for lbl in p._error_labels.values():
            self.assertTrue(lbl.isHidden())

    def test_reset_does_not_emit_settings_applied(self):
        p = _panel("unitree")
        applied = []
        p.settings_applied.connect(applied.append)
        p._on_reset()
        self.assertEqual(applied, [])


# ─────────────────────────────────────────────────────────────────────────────
# H. set_brand
# ─────────────────────────────────────────────────────────────────────────────

class TestSetBrand(unittest.TestCase):
    def test_set_brand_rebuilds_field_widgets(self):
        p = _panel("unitree")
        unitree_keys = set(p._field_widgets.keys())
        p.set_brand("bostiondynamics")
        spot_keys = set(p._field_widgets.keys())
        self.assertNotEqual(unitree_keys, spot_keys)

    def test_set_brand_field_keys_match_schema(self):
        p = _panel("unitree")
        p.set_brand("xiaomi")
        expected = set(get_settings_schema("xiaomi").keys())
        self.assertEqual(set(p._field_widgets.keys()), expected)

    def test_set_brand_updates_title(self):
        p = _panel("unitree")
        p.set_brand("bostiondynamics")
        self.assertIn("bostiondynamics", p._title_label.text())

    def test_set_brand_with_config_pre_populates(self):
        p = _panel("unitree")
        p.set_brand("unitree", config={"timeout": 123})
        self.assertEqual(p.get_current_config()["timeout"], 123)

    def test_set_unknown_brand_falls_back(self):
        p = _panel("unitree")
        p.set_brand("bad_brand")
        self.assertEqual(p._brand, KNOWN_BRANDS[0])


# ─────────────────────────────────────────────────────────────────────────────
# I. Widget type mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestWidgetTypeMapping(unittest.TestCase):
    def setUp(self):
        self.unitree = _panel("unitree")
        self.xiaomi  = _panel("xiaomi")

    def test_choice_field_is_qcombobox(self):
        self.assertIsInstance(self.unitree._field_widgets["brand"], QComboBox)
        self.assertIsInstance(self.unitree._field_widgets["robot_type"], QComboBox)
        self.assertIsInstance(self.unitree._field_widgets["connection_mode"], QComboBox)
        self.assertIsInstance(self.unitree._field_widgets["control_level"], QComboBox)

    def test_bool_field_is_qcheckbox(self):
        self.assertIsInstance(self.unitree._field_widgets["safety_enabled"], QCheckBox)

    def test_integer_field_is_qspinbox(self):
        self.assertIsInstance(self.unitree._field_widgets["timeout"], QSpinBox)
        self.assertIsInstance(self.unitree._field_widgets["dds_domain_id"], QSpinBox)

    def test_text_field_is_qlineedit(self):
        self.assertIsInstance(self.unitree._field_widgets["network_interface"], QLineEdit)

    def test_list_field_is_qlineedit(self):
        self.assertIsInstance(self.xiaomi._field_widgets["action_endpoints"], QLineEdit)

    def test_bool_optional_field_is_qcheckbox(self):
        spot = _panel("bostiondynamics")
        self.assertIsInstance(spot._field_widgets["timesync_required"], QCheckBox)
        self.assertIsInstance(spot._field_widgets["lease_required"], QCheckBox)


# ─────────────────────────────────────────────────────────────────────────────
# J. Required / Optional grouping
# ─────────────────────────────────────────────────────────────────────────────

class TestGrouping(unittest.TestCase):
    def test_required_group_exists_for_unitree(self):
        p = _panel("unitree")
        titles = _groupbox_titles(p)
        self.assertIn("Required Settings", titles)

    def test_optional_group_exists_for_spot(self):
        p = _panel("bostiondynamics")
        titles = _groupbox_titles(p)
        self.assertIn("Optional Settings", titles)

    def test_optional_group_exists_for_xiaomi(self):
        p = _panel("xiaomi")
        titles = _groupbox_titles(p)
        self.assertIn("Optional Settings", titles)

    def test_no_optional_group_when_all_required(self):
        # unitree has no optional fields — no "Optional Settings" group
        p = _panel("unitree")
        titles = _groupbox_titles(p)
        self.assertNotIn("Optional Settings", titles)

    def test_field_widget_count_matches_schema(self):
        for brand in KNOWN_BRANDS:
            p = _panel(brand)
            schema_count = len(get_settings_schema(brand))
            self.assertEqual(
                len(p._field_widgets), schema_count,
                f"Widget count mismatch for brand={brand!r}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# K. Initial config pre-population
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialConfigPrePopulation(unittest.TestCase):
    def test_integer_field_pre_populated(self):
        p = SettingsPanel("unitree", {"timeout": 42})
        self.assertEqual(p.get_current_config()["timeout"], 42)

    def test_text_field_pre_populated(self):
        p = SettingsPanel("unitree", {"robot_type": "go2"})
        self.assertEqual(p.get_current_config()["robot_type"], "go2")

    def test_choice_field_pre_populated(self):
        p = SettingsPanel("unitree", {"connection_mode": "hardware"})
        self.assertEqual(p.get_current_config()["connection_mode"], "hardware")

    def test_bool_field_pre_populated(self):
        p = SettingsPanel("unitree", {"safety_enabled": True})
        self.assertTrue(p.get_current_config()["safety_enabled"])

    def test_unknown_initial_config_keys_ignored(self):
        # Extra keys that are not in the schema should not crash
        p = SettingsPanel("unitree", {"nonexistent_key": "value"})
        self.assertIsInstance(p, SettingsPanel)


# ─────────────────────────────────────────────────────────────────────────────
# K2. robot_type dynamic choices / legacy compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestRobotTypeDynamicChoices(unittest.TestCase):
    def test_robot_type_choices_follow_brand(self):
        p = _panel("unitree")
        unitree_models = [p._field_widgets["robot_type"].itemText(i) for i in range(p._field_widgets["robot_type"].count())]
        self.assertIn("go2", unitree_models)

        p.set_brand("bostiondynamics")
        spot_models = [p._field_widgets["robot_type"].itemText(i) for i in range(p._field_widgets["robot_type"].count())]
        self.assertIn("spot", spot_models)
        self.assertNotIn("go2", spot_models)

    def test_legacy_unknown_robot_type_is_preserved(self):
        p = SettingsPanel("unitree", {"robot_type": "legacy_unknown_model"})
        self.assertEqual(p.get_current_config()["robot_type"], "legacy_unknown_model")


# ─────────────────────────────────────────────────────────────────────────────
# L. MainZonePanel integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainZonePanelIntegration(unittest.TestCase):
    def setUp(self):
        from bin.layout.main_zone_panel import MainZonePanel
        self.mzp = MainZonePanel()

    def test_settings_panel_attribute_exists(self):
        self.assertIsInstance(self.mzp.settings_panel, SettingsPanel)

    def test_settings_tab_in_workspace_tabs(self):
        tabs = self.mzp.workspace_tabs
        tab_titles = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertIn("Settings", tab_titles)

    def test_get_sdk_settings_returns_dict(self):
        self.assertIsInstance(self.mzp.get_sdk_settings(), dict)

    def test_set_sdk_brand_changes_panel_brand(self):
        self.mzp.set_sdk_brand("bostiondynamics")
        self.assertEqual(self.mzp.settings_panel._brand, "bostiondynamics")


if __name__ == "__main__":
    unittest.main()
