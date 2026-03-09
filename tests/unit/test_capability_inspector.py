"""Tests for bin.components.capability_inspector — Cycle 2 STAGE-04.

Coverage:

    A. Instantiation — empty, None, valid, partial cap dict
    B. Section rendering — brand/adapter header, actions, sensors, flags,
       required_settings
    C. Partial-data resilience — individual missing keys render gracefully
    D. focus_setting signal — clicking a required-settings button emits
       the correct field_id
    E. Missing-settings highlighting — missing items styled differently
    F. set_capabilities — refreshes content with new cap dict
    G. SettingsPanel.scroll_to_field — no crash, unknown field is no-op
    H. MainZonePanel integration — capability_inspector attribute, Capabilities
       tab present, focus_setting switches to Settings tab
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton

_app = QApplication.instance() or QApplication(sys.argv)

from bin.components.capability_inspector import CapabilityInspector
from bin.components.settings_panel import SettingsPanel
from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
from system.service.adapters.spot_sdk.adapter import SpotAdapter
from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter


# ── Fixture capability dicts ─────────────────────────────────────────────────

_UNITREE_CAP = UnitreeAdapter().capabilities()
_SPOT_CAP    = SpotAdapter().capabilities()
_CYBER_CAP   = CyberDogAdapter().capabilities()

_FULL_CAP = {
    "brand":   "testbrand",
    "adapter": "test_sdk",
    "actions": ["stand", "sit", "walk"],
    "sensors": ["battery", "imu"],
    "flags":   {"sdk_available": True, "sim_mode": False},
    "required_settings": ["brand", "robot_type", "timeout"],
}


def _inspector(cap=None, missing=None) -> CapabilityInspector:
    return CapabilityInspector(cap or {}, missing or [])


def _groupbox_titles(widget) -> set:
    return {gb.title() for gb in widget.findChildren(QGroupBox)}


def _find_buttons(widget) -> list:
    return widget.findChildren(QPushButton)


# ─────────────────────────────────────────────────────────────────────────────
# A. Instantiation
# ─────────────────────────────────────────────────────────────────────────────

class TestInstantiation(unittest.TestCase):
    def test_empty_dict_instantiates(self):
        i = CapabilityInspector({})
        self.assertIsInstance(i, CapabilityInspector)

    def test_none_instantiates(self):
        i = CapabilityInspector(None)
        self.assertIsInstance(i, CapabilityInspector)

    def test_full_cap_dict_instantiates(self):
        i = CapabilityInspector(_FULL_CAP)
        self.assertIsInstance(i, CapabilityInspector)

    def test_non_dict_instantiates_without_crash(self):
        i = CapabilityInspector("not a dict")  # type: ignore
        self.assertIsInstance(i, CapabilityInspector)

    def test_real_unitree_capabilities_instantiates(self):
        i = CapabilityInspector(_UNITREE_CAP)
        self.assertIsInstance(i, CapabilityInspector)

    def test_real_spot_capabilities_instantiates(self):
        i = CapabilityInspector(_SPOT_CAP)
        self.assertIsInstance(i, CapabilityInspector)

    def test_real_cyberdog_capabilities_instantiates(self):
        i = CapabilityInspector(_CYBER_CAP)
        self.assertIsInstance(i, CapabilityInspector)


# ─────────────────────────────────────────────────────────────────────────────
# B. Section rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionRendering(unittest.TestCase):
    def setUp(self):
        self.inspector = CapabilityInspector(_FULL_CAP)

    def test_actions_groupbox_present(self):
        self.assertIn("Actions", _groupbox_titles(self.inspector))

    def test_sensors_groupbox_present(self):
        self.assertIn("Sensors", _groupbox_titles(self.inspector))

    def test_flags_groupbox_present(self):
        self.assertIn("Feature Flags", _groupbox_titles(self.inspector))

    def test_required_settings_groupbox_present(self):
        self.assertIn("Required Settings", _groupbox_titles(self.inspector))

    def test_brand_in_header_label(self):
        self.assertIn("testbrand", self.inspector._header_label.text())

    def test_actions_rendered_as_labels(self):
        labels = [
            lbl.text() for lbl in self.inspector.findChildren(QLabel)
            if lbl.text() in _FULL_CAP["actions"]
        ]
        self.assertEqual(sorted(labels), sorted(_FULL_CAP["actions"]))

    def test_sensors_rendered_as_labels(self):
        label_texts = {lbl.text() for lbl in self.inspector.findChildren(QLabel)}
        for sensor in _FULL_CAP["sensors"]:
            self.assertIn(sensor, label_texts)

    def test_required_settings_rendered_as_buttons(self):
        btn_texts = {btn.text() for btn in _find_buttons(self.inspector)}
        for fid in _FULL_CAP["required_settings"]:
            self.assertTrue(
                any(fid in t for t in btn_texts),
                f"No button found for required_setting {fid!r}",
            )

    def test_real_unitree_all_sections_present(self):
        i = CapabilityInspector(_UNITREE_CAP)
        titles = _groupbox_titles(i)
        for section in ("Actions", "Sensors", "Feature Flags", "Required Settings"):
            self.assertIn(section, titles)


# ─────────────────────────────────────────────────────────────────────────────
# C. Partial-data resilience
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialDataResilience(unittest.TestCase):
    def test_missing_actions_key_renders_not_declared(self):
        cap = {k: v for k, v in _FULL_CAP.items() if k != "actions"}
        i = CapabilityInspector(cap)
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertIn("(not declared)", label_texts)

    def test_missing_sensors_key_renders_not_declared(self):
        cap = {k: v for k, v in _FULL_CAP.items() if k != "sensors"}
        i = CapabilityInspector(cap)
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertIn("(not declared)", label_texts)

    def test_missing_flags_key_renders_not_declared(self):
        cap = {k: v for k, v in _FULL_CAP.items() if k != "flags"}
        i = CapabilityInspector(cap)
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertIn("(not declared)", label_texts)

    def test_missing_required_settings_key_renders_not_declared(self):
        cap = {k: v for k, v in _FULL_CAP.items() if k != "required_settings"}
        i = CapabilityInspector(cap)
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertIn("(not declared)", label_texts)

    def test_empty_cap_dict_shows_no_data_message(self):
        i = CapabilityInspector({})
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertTrue(
            any("No capability data" in t for t in label_texts),
            "Expected 'No capability data' message for empty dict",
        )

    def test_none_cap_shows_no_data_message(self):
        i = CapabilityInspector(None)
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertTrue(
            any("No capability data" in t for t in label_texts),
        )

    def test_partial_cap_does_not_crash(self):
        # Only brand and adapter — all list/dict sections missing
        cap = {"brand": "x", "adapter": "y"}
        i = CapabilityInspector(cap)
        self.assertIsInstance(i, CapabilityInspector)

    def test_empty_actions_list_renders_not_declared(self):
        cap = dict(_FULL_CAP)
        cap["actions"] = []
        i = CapabilityInspector(cap)
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertIn("(not declared)", label_texts)


# ─────────────────────────────────────────────────────────────────────────────
# D. focus_setting signal
# ─────────────────────────────────────────────────────────────────────────────

class TestFocusSettingSignal(unittest.TestCase):
    def test_clicking_required_setting_btn_emits_signal(self):
        i = CapabilityInspector(_FULL_CAP)
        emitted = []
        i.focus_setting.connect(emitted.append)

        # Find the button for "brand"
        btn = next(
            (b for b in _find_buttons(i) if "brand" in b.text()),
            None,
        )
        self.assertIsNotNone(btn, "No button found for 'brand'")
        btn.click()
        self.assertEqual(emitted, ["brand"])

    def test_clicking_different_buttons_emits_correct_field_ids(self):
        i = CapabilityInspector(_FULL_CAP)
        emitted = []
        i.focus_setting.connect(emitted.append)

        for fid in _FULL_CAP["required_settings"]:
            btn = next(
                (b for b in _find_buttons(i) if fid in b.text()),
                None,
            )
            if btn is not None:
                btn.click()

        # Each required_settings field_id should appear in emitted (in order)
        for fid in _FULL_CAP["required_settings"]:
            self.assertIn(fid, emitted)

    def test_signal_emits_exact_field_id_string(self):
        cap = {"brand": "x", "adapter": "y",
               "actions": [], "sensors": [], "flags": {},
               "required_settings": ["network_interface"]}
        i = CapabilityInspector(cap)
        emitted = []
        i.focus_setting.connect(emitted.append)
        btn = next((b for b in _find_buttons(i) if "network_interface" in b.text()), None)
        self.assertIsNotNone(btn)
        btn.click()
        self.assertEqual(emitted, ["network_interface"])


# ─────────────────────────────────────────────────────────────────────────────
# E. Missing-settings highlighting
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingSettingsHighlighting(unittest.TestCase):
    def test_missing_button_contains_warning_indicator(self):
        i = CapabilityInspector(_FULL_CAP, missing_settings=["timeout"])
        btn = next(
            (b for b in _find_buttons(i) if "timeout" in b.text()),
            None,
        )
        self.assertIsNotNone(btn)
        # Missing button should show a warning prefix or "(missing)" suffix
        self.assertTrue(
            "missing" in btn.text().lower() or "⚠" in btn.text(),
            f"Expected missing indicator in button text: {btn.text()!r}",
        )

    def test_non_missing_button_has_no_warning(self):
        i = CapabilityInspector(_FULL_CAP, missing_settings=["timeout"])
        # "brand" is NOT in missing_settings
        btn = next(
            (b for b in _find_buttons(i) if b.text().strip() == "brand"),
            None,
        )
        self.assertIsNotNone(btn)
        self.assertNotIn("missing", btn.text().lower())
        self.assertNotIn("⚠", btn.text())

    def test_all_missing_when_all_listed(self):
        fids = _FULL_CAP["required_settings"]
        i = CapabilityInspector(_FULL_CAP, missing_settings=fids)
        for fid in fids:
            btn = next(
                (b for b in _find_buttons(i) if fid in b.text()),
                None,
            )
            self.assertIsNotNone(btn, f"No button for {fid!r}")
            self.assertTrue(
                "missing" in btn.text().lower() or "⚠" in btn.text(),
                f"Expected missing indicator for {fid!r}: {btn.text()!r}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# F. set_capabilities
# ─────────────────────────────────────────────────────────────────────────────

class TestSetCapabilities(unittest.TestCase):
    def test_set_capabilities_updates_brand_in_header(self):
        i = CapabilityInspector({})
        i.set_capabilities(_UNITREE_CAP)
        self.assertIn("unitree", i._header_label.text())

    def test_set_capabilities_replaces_old_content(self):
        i = CapabilityInspector(_FULL_CAP)
        i.set_capabilities(_SPOT_CAP)
        # Should now show spot brand, not "testbrand"
        self.assertNotIn("testbrand", i._header_label.text())

    def test_set_capabilities_with_missing_updates_highlight(self):
        i = CapabilityInspector(_FULL_CAP, missing_settings=[])
        # Re-set with missing_settings
        i.set_capabilities(_FULL_CAP, missing_settings=["brand"])
        btn = next(
            (b for b in _find_buttons(i) if "brand" in b.text()),
            None,
        )
        self.assertIsNotNone(btn)
        self.assertTrue(
            "missing" in btn.text().lower() or "⚠" in btn.text(),
        )

    def test_set_capabilities_empty_resets_to_no_data_message(self):
        i = CapabilityInspector(_FULL_CAP)
        i.set_capabilities({})
        label_texts = {lbl.text() for lbl in i.findChildren(QLabel)}
        self.assertTrue(any("No capability data" in t for t in label_texts))


# ─────────────────────────────────────────────────────────────────────────────
# G. SettingsPanel.scroll_to_field
# ─────────────────────────────────────────────────────────────────────────────

class TestScrollToField(unittest.TestCase):
    def test_scroll_to_known_field_does_not_crash(self):
        p = SettingsPanel("unitree", {})
        p.scroll_to_field("timeout")   # should not raise

    def test_scroll_to_unknown_field_does_not_crash(self):
        p = SettingsPanel("unitree", {})
        p.scroll_to_field("nonexistent_field")  # should not raise

    def test_scroll_to_field_works_for_all_unitree_fields(self):
        from system.service.settings_schema import get_settings_schema
        p = SettingsPanel("unitree", {})
        for fid in get_settings_schema("unitree"):
            p.scroll_to_field(fid)  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# H. MainZonePanel integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainZonePanelIntegration(unittest.TestCase):
    def setUp(self):
        from bin.layout.main_zone_panel import MainZonePanel
        self.mzp = MainZonePanel()

    def test_capability_inspector_attribute_exists(self):
        self.assertIsInstance(self.mzp.capability_inspector, CapabilityInspector)

    def test_capabilities_tab_in_workspace_tabs(self):
        tabs = self.mzp.workspace_tabs
        tab_titles = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertIn("Capabilities", tab_titles)

    def test_set_adapter_capabilities_updates_inspector(self):
        self.mzp.set_adapter_capabilities(_UNITREE_CAP)
        self.assertIn("unitree", self.mzp.capability_inspector._header_label.text())

    def test_focus_setting_switches_to_settings_tab(self):
        # Find the Settings tab index
        tabs = self.mzp.workspace_tabs
        settings_idx = next(
            i for i in range(tabs.count()) if tabs.tabText(i) == "Settings"
        )
        # Start on a different tab
        tabs.setCurrentIndex(0)
        # Fire focus_setting
        self.mzp._on_capability_focus_setting("timeout")
        self.assertEqual(tabs.currentIndex(), settings_idx)

    def test_focus_setting_does_not_crash_on_unknown_field(self):
        self.mzp._on_capability_focus_setting("nonexistent_field")  # no raise


if __name__ == "__main__":
    unittest.main()
