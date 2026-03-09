#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle 1 integration test matrix — Steps 1.5–1.8.

Validates cross-module behaviour for the HeartBeat frontend productization
work introduced in Circle 1 Steps 1.5–1.7:

  Step 1.5  Event-state / IO-mapping visualization (hb_display_state.py)
  Step 1.6  HeartBeat frontend state persistence (mission_persistence.py)
  Step 1.7  Model-switch compatibility audit (hb_compat_audit.py)
  Step 1.8  Full-suite acceptance gate (this file)

Each test class exercises a distinct cross-module integration boundary.
No QApplication, no Qt, no robot SDK required.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports — pure Python modules only
# ---------------------------------------------------------------------------

from system.behavior.hb_display_state import (
    HBEventStatus, HBIOStatus,
    badge_for_event_state, badge_for_io_status,
    detect_io_conflicts, compute_event_summary, compute_io_summary,
    HBConflictHint, HBEventSummary, HBIOSummary,
)
from system.behavior.hb_node_catalog import (
    HBNodeCatalog, HBNodeAvailability, HBNodeCatalogEntry,
)
from system.behavior.hb_compat_audit import (
    HBCompatIssue, HBCompatReport,
    audit_catalog_compatibility, report_to_behavior_diagnostics,
)
from system.behavior.behavior_artifact import BehaviorDiagnostic
from system.runtime.contracts import DiagnosticsKey
from bin.core.mission_persistence import (
    MISSION_SCHEMA_VERSION,
    build_behavior_drafts_payload, extract_behavior_drafts_payload,
    inject_snapshot_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap_dict(brand="unitree", robot_type="go2", sensors=None, flags=None):
    return {
        "brand": brand,
        "robot_type": robot_type,
        "sensors": sensors if sensors is not None else ["imu", "qpos", "qvel"],
        "flags":   flags   if flags   is not None else {"high_level_control": True},
    }


def _full_catalog():
    """Catalog built from a rich capability dict — all nodes AVAILABLE."""
    return HBNodeCatalog.from_capability_dict(_cap_dict())


def _sparse_catalog():
    """Catalog built from an empty sensors list — sensor nodes UNSUPPORTED."""
    return HBNodeCatalog.from_capability_dict(_cap_dict(sensors=[], flags={}))


def _io_mapping(signal, direction, status="active", target="joint_vel"):
    """Return a minimal HBIOMapping-like namespace."""
    return types.SimpleNamespace(
        signal=signal, direction=direction, status=status, target_param=target
    )


def _event_entry(event_name, state):
    """Return a minimal HBEventStateEntry-like namespace."""
    return types.SimpleNamespace(
        event_name=event_name, state=state,
        source="test", timestamp="00:00", reason="",
    )


# ===========================================================================
# Step 1.5 — display state badge & analysis pipeline
# ===========================================================================

class TestDisplayStateBadgePipeline(unittest.TestCase):
    """End-to-end: raw state strings → badges → color_keys."""

    def test_event_active_badge_color_key_is_success(self):
        b = badge_for_event_state(HBEventStatus.ACTIVE)
        self.assertEqual(b.color_key, "success")

    def test_event_error_badge_color_key_is_error(self):
        b = badge_for_event_state(HBEventStatus.ERROR)
        self.assertEqual(b.color_key, "error")

    def test_io_active_badge_label_is_ok(self):
        b = badge_for_io_status(HBIOStatus.ACTIVE)
        self.assertEqual(b.label, "OK")

    def test_io_unsupported_badge_color_key_is_error(self):
        b = badge_for_io_status(HBIOStatus.UNSUPPORTED)
        self.assertEqual(b.color_key, "error")

    def test_io_unmapped_badge_color_key_is_warning(self):
        b = badge_for_io_status(HBIOStatus.UNMAPPED)
        self.assertEqual(b.color_key, "warning")

    def test_unknown_event_state_returns_badge_no_raise(self):
        b = badge_for_event_state("completely_unknown_xyz")
        self.assertIsInstance(b.label, str)
        self.assertIsInstance(b.color_key, str)


class TestDisplayStateAnalysis(unittest.TestCase):
    """Conflict detection and summary computation."""

    def test_no_conflicts_when_all_distinct_active(self):
        mappings = [
            _io_mapping("sig_a", "inbound"),
            _io_mapping("sig_b", "outbound"),
        ]
        self.assertEqual(detect_io_conflicts(mappings), [])

    def test_bidirectional_same_signal_is_conflict(self):
        mappings = [
            _io_mapping("pos_x", "inbound"),
            _io_mapping("pos_x", "outbound"),
        ]
        conflicts = detect_io_conflicts(mappings)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].signal, "pos_x")
        self.assertEqual(conflicts[0].severity, "error")

    def test_unsupported_mapping_is_error_conflict(self):
        mappings = [_io_mapping("vel_z", "inbound", status="unsupported")]
        conflicts = detect_io_conflicts(mappings)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].severity, "error")

    def test_compute_event_summary_counts_correctly(self):
        entries = [
            _event_entry("e1", HBEventStatus.ACTIVE),
            _event_entry("e2", HBEventStatus.ACTIVE),
            _event_entry("e3", HBEventStatus.ERROR),
            _event_entry("e4", HBEventStatus.INACTIVE),
        ]
        s = compute_event_summary(entries)
        self.assertEqual(s.total, 4)
        self.assertEqual(s.active, 2)
        self.assertEqual(s.errors, 1)

    def test_compute_io_summary_counts_directions(self):
        mappings = [
            _io_mapping("a", "inbound"),
            _io_mapping("b", "inbound"),
            _io_mapping("c", "outbound"),
        ]
        conflicts = detect_io_conflicts(mappings)
        s = compute_io_summary(mappings, conflicts)
        self.assertEqual(s.total, 3)
        self.assertEqual(s.inbound, 2)
        self.assertEqual(s.outbound, 1)


# ===========================================================================
# Step 1.6 — mission persistence round-trip
# ===========================================================================

class TestMissionPersistenceRoundTrip(unittest.TestCase):
    """build + extract must be perfectly inverse."""

    def test_schema_version_is_1_3(self):
        self.assertEqual(MISSION_SCHEMA_VERSION, "1.4")

    def test_schema_version_injected_by_inject_snapshot_metadata(self):
        data = {}
        inject_snapshot_metadata(data)
        self.assertEqual(data.get("schema_version"), "1.4")

    def test_build_extract_round_trip_preserves_drafts(self):
        drafts = {
            "0": {"core": "# node 0 code", "hb": {}},
            "5": {"core": "# node 5 code", "hb": {"authoring_mode": "canvas"}},
        }
        payload = build_behavior_drafts_payload(drafts)
        restored = extract_behavior_drafts_payload({"behavior_drafts": payload})
        self.assertEqual(restored["0"]["core"], "# node 0 code")
        self.assertEqual(restored["5"]["hb"]["authoring_mode"], "canvas")

    def test_extract_returns_none_for_missing_key(self):
        """Old mission files without 'behavior_drafts' must not error."""
        old_data = {"schema_version": "1.1", "canvas": {}}
        self.assertIsNone(extract_behavior_drafts_payload(old_data))

    def test_extract_returns_none_for_non_dict_value(self):
        self.assertIsNone(extract_behavior_drafts_payload({"behavior_drafts": "bad"}))

    def test_build_converts_int_keys_to_strings(self):
        drafts = {3: {"core": "x"}, 7: {"core": "y"}}
        payload = build_behavior_drafts_payload(drafts)
        self.assertIn("3", payload)
        self.assertIn("7", payload)

    def test_extract_never_raises_on_garbage_input(self):
        for bad in [None, 42, [], "str", {"behavior_drafts": [1, 2, 3]}]:
            try:
                extract_behavior_drafts_payload(bad)
            except Exception as exc:
                self.fail(f"extract raised on {bad!r}: {exc}")


# ===========================================================================
# Step 1.7 — catalog → audit → BehaviorDiagnostic pipeline (Issue 9)
# ===========================================================================

class TestCompatAuditPipeline(unittest.TestCase):
    """Full pipeline: capability dict → catalog → audit → IR diagnostics."""

    def test_full_sensors_catalog_yields_clean_audit(self):
        catalog = _full_catalog()
        report  = audit_catalog_compatibility(catalog, "unitree", "go2")
        self.assertTrue(report.is_clean())

    def test_empty_sensors_catalog_yields_errors(self):
        catalog = _sparse_catalog()
        report  = audit_catalog_compatibility(catalog, "unitree", "go2")
        self.assertFalse(report.is_clean())
        self.assertGreater(report.error_count, 0)

    def test_report_to_behavior_diagnostics_empty_on_clean(self):
        catalog = _full_catalog()
        report  = audit_catalog_compatibility(catalog)
        diags   = report_to_behavior_diagnostics(report)
        self.assertEqual(diags, [])

    def test_report_to_behavior_diagnostics_types(self):
        catalog = _sparse_catalog()
        report  = audit_catalog_compatibility(catalog, "unitree", "go2")
        diags   = report_to_behavior_diagnostics(report)
        self.assertGreater(len(diags), 0)
        for d in diags:
            self.assertIsInstance(d, BehaviorDiagnostic)
            self.assertIn(d.level, ("error", "warning"))

    def test_error_node_yields_error_diagnostic(self):
        catalog = _sparse_catalog()
        report  = audit_catalog_compatibility(catalog)
        diags   = report_to_behavior_diagnostics(report)
        levels  = {d.level for d in diags}
        self.assertIn("error", levels)

    def test_diagnostic_code_contains_availability_and_kind(self):
        catalog = _sparse_catalog()
        report  = audit_catalog_compatibility(catalog)
        diags   = report_to_behavior_diagnostics(report)
        for d in diags:
            self.assertIn("compat.", d.code)

    def test_diagnostic_location_matches_node_kind(self):
        catalog = _sparse_catalog()
        report  = audit_catalog_compatibility(catalog)
        diags   = report_to_behavior_diagnostics(report)
        entry_kinds = {e.kind for e in catalog.all_entries()}
        for d in diags:
            self.assertIn(d.location, entry_kinds)

    def test_to_dict_produces_compat_diagnostics_format(self):
        """BehaviorDiagnostic.to_dict() must include all COMPAT_DIAGNOSTICS fields."""
        catalog   = _sparse_catalog()
        report    = audit_catalog_compatibility(catalog)
        diag_dicts = [d.to_dict() for d in report_to_behavior_diagnostics(report)]
        for d in diag_dicts:
            self.assertIn("level",    d)
            self.assertIn("code",     d)
            self.assertIn("message",  d)
            self.assertIn("location", d)

    def test_compat_diagnostics_key_is_string(self):
        self.assertIsInstance(DiagnosticsKey.COMPAT_DIAGNOSTICS, str)
        self.assertGreater(len(DiagnosticsKey.COMPAT_DIAGNOSTICS), 0)

    def test_compat_diagnostics_key_distinct_from_other_keys(self):
        ck = DiagnosticsKey.COMPAT_DIAGNOSTICS
        other_keys = [
            DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
            DiagnosticsKey.HEARTBEAT_DIAGNOSTICS,
            DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS,
        ]
        for ok in other_keys:
            self.assertNotEqual(ck, ok)


# ===========================================================================
# Step 1.7 — _run_compat_audit wiring on all paths (Issue 7)
# ===========================================================================

class TestCompatAuditAllPaths(unittest.TestCase):
    """_run_compat_audit must be called on adapter=None, exception, and success paths."""

    def _make_stub(self, adapter_obj=None, create_raises=False, cap_raises=False):
        import bin.ui as ui_mod
        from unittest.mock import patch

        stub = types.SimpleNamespace()

        mz = MagicMock()
        clean_report = HBCompatReport("", "")
        mz.behavior_panel.run_compat_audit.return_value = clean_report
        mz.show_compat_report = MagicMock()
        mz.get_sdk_settings.return_value = {}
        stub.main_zone = mz

        stub._run_compat_audit  = MagicMock()  # spy
        stub._update_hb_catalog = lambda cap: ui_mod.MainWindow._update_hb_catalog(stub, cap)
        stub._update_hb_channel = MagicMock()  # Circle 2: no-op for compat-audit tests
        stub._refresh_capability_inspector = lambda: ui_mod.MainWindow._refresh_capability_inspector(stub)

        return stub

    def _patch_robot_context(self, adapter_obj=None, create_raises=False):
        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _cm():
            def _create(b, rt):
                if create_raises:
                    raise RuntimeError("adapter error")
                return adapter_obj

            with patch("bin.ui.RobotContext.get_current_brand", return_value="unitree"), \
                 patch("bin.ui.RobotContext.get_robot_type",    return_value="go2"), \
                 patch("bin.ui.RobotContext._create_adapter_for_brand", side_effect=_create):
                yield
        return _cm()

    def test_adapter_none_path_triggers_audit(self):
        stub = self._make_stub()
        with self._patch_robot_context(adapter_obj=None):
            stub._refresh_capability_inspector()
        stub._run_compat_audit.assert_called_once_with("unitree", "go2")

    def test_exception_path_triggers_audit(self):
        stub = self._make_stub()
        with self._patch_robot_context(create_raises=True):
            stub._refresh_capability_inspector()
        stub._run_compat_audit.assert_called_once_with("unitree", "go2")

    def test_success_path_triggers_audit(self):
        from unittest.mock import patch
        stub = self._make_stub()
        mock_adapter = MagicMock()
        mock_adapter.capabilities.return_value = _cap_dict()
        with self._patch_robot_context(adapter_obj=mock_adapter), \
             patch("bin.core.settings_form.compute_missing_settings", return_value=[]):
            stub._refresh_capability_inspector()
        stub._run_compat_audit.assert_called_once_with("unitree", "go2")


# ===========================================================================
# Step 1.7 — show_compat_report wiring (Issue 8)
# ===========================================================================

class TestShowCompatReportWiring(unittest.TestCase):
    """_run_compat_audit must route through show_compat_report on main_zone."""

    def _make_stub(self, has_issues=False):
        import bin.ui as ui_mod

        stub = types.SimpleNamespace()
        mz = MagicMock()
        if has_issues:
            report = HBCompatReport("unitree", "go2", issues=[
                HBCompatIssue("sensor_read", "Sensor Read", "unsupported", "missing", "error"),
            ])
        else:
            report = HBCompatReport("unitree", "go2")
        mz.behavior_panel.run_compat_audit.return_value = report
        mz.show_compat_report = MagicMock()
        stub.main_zone = mz
        stub._run_compat_audit = lambda b, rt: ui_mod.MainWindow._run_compat_audit(stub, b, rt)
        return stub

    def test_show_compat_report_called_on_clean_report(self):
        stub = self._make_stub()
        stub._run_compat_audit("unitree", "go2")
        stub.main_zone.show_compat_report.assert_called_once()

    def test_show_compat_report_receives_compat_report_object(self):
        stub = self._make_stub()
        stub._run_compat_audit("unitree", "go2")
        args = stub.main_zone.show_compat_report.call_args[0]
        self.assertIsInstance(args[0], HBCompatReport)

    def test_show_compat_report_receives_empty_diag_dicts_for_clean(self):
        stub = self._make_stub()
        stub._run_compat_audit("unitree", "go2")
        args = stub.main_zone.show_compat_report.call_args[0]
        self.assertEqual(args[1], [])

    def test_show_compat_report_receives_diag_dicts_for_issues(self):
        stub = self._make_stub(has_issues=True)
        stub._run_compat_audit("unitree", "go2")
        args = stub.main_zone.show_compat_report.call_args[0]
        self.assertGreater(len(args[1]), 0)
        # Each dict must have BehaviorDiagnostic-format keys
        for d in args[1]:
            self.assertIn("level",   d)
            self.assertIn("code",    d)
            self.assertIn("message", d)


# ===========================================================================
# Circle 1 regression guard — Steps 1.1–1.4 still intact
# ===========================================================================

class TestCircle1RuntimePathRegression(unittest.TestCase):
    """Ensure Steps 1.1–1.4 runtime path primitives are unbroken."""

    def test_behavior_run_flags_importable(self):
        from system.runtime.migration import BehaviorRunFlags
        flags = BehaviorRunFlags.from_env()
        self.assertIsInstance(flags.use_workflowir_for_behavior, bool)

    def test_node_kind_behavior_call_exists(self):
        from compiler.ir.workflow_ir import NodeKind
        self.assertTrue(hasattr(NodeKind, "BEHAVIOR_CALL"))
        self.assertEqual(NodeKind.BEHAVIOR_CALL.value, "behavior_call")

    def test_diagnostics_keys_all_distinct(self):
        keys = [
            DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
            DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS,
            DiagnosticsKey.HEARTBEAT_DIAGNOSTICS,
            DiagnosticsKey.HEARTBEAT_STATUS,
            DiagnosticsKey.COMPAT_DIAGNOSTICS,
        ]
        self.assertEqual(len(keys), len(set(keys)),
                         "DiagnosticsKey constants must all be distinct strings")

    def test_behavior_diagnostic_factory_methods(self):
        e = BehaviorDiagnostic.error("test.code", "error msg")
        w = BehaviorDiagnostic.warning("test.code", "warn msg")
        self.assertEqual(e.level, "error")
        self.assertEqual(w.level, "warning")
        self.assertIn("level", e.to_dict())

    def test_hb_node_catalog_from_empty_cap_dict_does_not_raise(self):
        try:
            catalog = HBNodeCatalog.from_capability_dict({})
            self.assertIsNotNone(catalog)
        except Exception as exc:
            self.fail(f"from_capability_dict({{}}) raised: {exc}")

    def test_audit_catalog_compatibility_never_raises(self):
        for cat in [HBNodeCatalog.unknown(), _full_catalog(), _sparse_catalog()]:
            try:
                audit_catalog_compatibility(cat)
            except Exception as exc:
                self.fail(f"audit_catalog_compatibility raised: {exc}")


if __name__ == "__main__":
    unittest.main()
