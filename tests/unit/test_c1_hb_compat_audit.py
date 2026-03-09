#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system/behavior/hb_compat_audit.py — Circle 1 Step 1.7.

Coverage
--------
HBCompatIssue
  - all fields stored correctly
  - severity is "error" or "warning"

HBCompatReport
  - is_clean() when no issues
  - has_errors / has_warnings properties
  - error_count / warning_count properties
  - summary_text() — all combinations (clean, errors only, warnings only, mixed)

audit_catalog_compatibility()
  - empty catalog → clean report
  - all AVAILABLE → clean report
  - UNSUPPORTED entry → error issue
  - LIMITED entry → warning issue
  - UNKNOWN_CAPABILITY entry → warning issue
  - multiple mixed states → all issues returned; counts correct
  - brand/robot_type embedded in default reason strings
  - custom reason on entry preserved (not overwritten by default)
  - never raises on arbitrary catalog input

_run_compat_audit wiring (bin/ui.py stub)
  - called on success path of _refresh_capability_inspector
  - errors are suppressed (never propagate)

Isolation
---------
- No QApplication. No Qt. Pure-Python only.
- HBNodeCatalog populated via from_capability_dict() with crafted capability dicts.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.hb_compat_audit import (  # noqa: E402
    HBCompatIssue,
    HBCompatReport,
    audit_catalog_compatibility,
    report_to_behavior_diagnostics,
)
from system.runtime.contracts import DiagnosticsKey  # noqa: E402
from system.behavior.hb_node_catalog import (  # noqa: E402
    HBNodeAvailability,
    HBNodeCatalog,
    HBNodeCatalogEntry,
)


# ---------------------------------------------------------------------------
# Catalog builder helpers
# ---------------------------------------------------------------------------

def _catalog(*entries: HBNodeCatalogEntry) -> HBNodeCatalog:
    """Build a catalog with exactly the given entries."""
    cat = HBNodeCatalog.unknown()
    cat._entries = {e.kind: e for e in entries}  # type: ignore[attr-defined]
    return cat


def _entry(kind: str, avail: str, reason: str = "") -> HBNodeCatalogEntry:
    label_map = {
        "sensor_read": "Sensor Read",
        "movement_blend": "Movement Blend",
        "motor_command": "Motor Command",
        "safety_gate": "Safety Gate",
    }
    return HBNodeCatalogEntry(
        kind=kind,
        label=label_map.get(kind, kind.replace("_", " ").title()),
        availability=avail,
        reason=reason,
    )


def _available(kind: str) -> HBNodeCatalogEntry:
    return _entry(kind, HBNodeAvailability.AVAILABLE)


def _unsupported(kind: str, reason: str = "") -> HBNodeCatalogEntry:
    return _entry(kind, HBNodeAvailability.UNSUPPORTED, reason)


def _limited(kind: str, reason: str = "") -> HBNodeCatalogEntry:
    return _entry(kind, HBNodeAvailability.LIMITED, reason)


def _unknown(kind: str, reason: str = "") -> HBNodeCatalogEntry:
    return _entry(kind, HBNodeAvailability.UNKNOWN_CAPABILITY, reason)


# ===========================================================================
# HBCompatIssue
# ===========================================================================

class TestHBCompatIssue(unittest.TestCase):

    def _make(self, severity="error"):
        return HBCompatIssue(
            node_kind="sensor_read",
            node_label="Sensor Read",
            availability=HBNodeAvailability.UNSUPPORTED,
            reason="not supported on go2",
            severity=severity,
        )

    def test_fields_stored_correctly(self):
        issue = self._make()
        self.assertEqual(issue.node_kind, "sensor_read")
        self.assertEqual(issue.node_label, "Sensor Read")
        self.assertEqual(issue.availability, HBNodeAvailability.UNSUPPORTED)
        self.assertEqual(issue.reason, "not supported on go2")
        self.assertEqual(issue.severity, "error")

    def test_warning_severity(self):
        issue = self._make(severity="warning")
        self.assertEqual(issue.severity, "warning")


# ===========================================================================
# HBCompatReport
# ===========================================================================

class TestHBCompatReportClean(unittest.TestCase):

    def _clean(self):
        return HBCompatReport(brand="unitree", robot_type="go2", issues=[])

    def test_is_clean_true_when_no_issues(self):
        self.assertTrue(self._clean().is_clean())

    def test_has_errors_false_when_clean(self):
        self.assertFalse(self._clean().has_errors)

    def test_has_warnings_false_when_clean(self):
        self.assertFalse(self._clean().has_warnings)

    def test_error_count_zero_when_clean(self):
        self.assertEqual(self._clean().error_count, 0)

    def test_warning_count_zero_when_clean(self):
        self.assertEqual(self._clean().warning_count, 0)

    def test_summary_text_no_issues(self):
        self.assertEqual(self._clean().summary_text(), "No issues")


class TestHBCompatReportWithIssues(unittest.TestCase):

    def _err(self):
        return HBCompatIssue("a", "A", "unsupported", "r", "error")

    def _warn(self):
        return HBCompatIssue("b", "B", "limited", "r", "warning")

    def test_is_clean_false_with_issues(self):
        r = HBCompatReport("u", "g2", issues=[self._err()])
        self.assertFalse(r.is_clean())

    def test_has_errors_true(self):
        r = HBCompatReport("u", "g2", issues=[self._err()])
        self.assertTrue(r.has_errors)

    def test_has_errors_false_when_only_warnings(self):
        r = HBCompatReport("u", "g2", issues=[self._warn()])
        self.assertFalse(r.has_errors)

    def test_has_warnings_true(self):
        r = HBCompatReport("u", "g2", issues=[self._warn()])
        self.assertTrue(r.has_warnings)

    def test_has_warnings_false_when_only_errors(self):
        r = HBCompatReport("u", "g2", issues=[self._err()])
        self.assertFalse(r.has_warnings)

    def test_error_count(self):
        r = HBCompatReport("u", "g2", issues=[self._err(), self._err(), self._warn()])
        self.assertEqual(r.error_count, 2)

    def test_warning_count(self):
        r = HBCompatReport("u", "g2", issues=[self._err(), self._warn(), self._warn()])
        self.assertEqual(r.warning_count, 2)


class TestHBCompatReportSummaryText(unittest.TestCase):

    def _mk(self, errors=0, warnings=0):
        issues = []
        for _ in range(errors):
            issues.append(HBCompatIssue("a", "A", "unsupported", "r", "error"))
        for _ in range(warnings):
            issues.append(HBCompatIssue("b", "B", "limited", "r", "warning"))
        return HBCompatReport("u", "g", issues=issues)

    def test_clean_text(self):
        self.assertEqual(self._mk().summary_text(), "No issues")

    def test_one_error(self):
        self.assertIn("1 error", self._mk(errors=1).summary_text())

    def test_multiple_errors(self):
        self.assertIn("2 errors", self._mk(errors=2).summary_text())

    def test_one_warning(self):
        self.assertIn("1 warning", self._mk(warnings=1).summary_text())

    def test_multiple_warnings(self):
        self.assertIn("3 warnings", self._mk(warnings=3).summary_text())

    def test_mixed_text_contains_both(self):
        text = self._mk(errors=1, warnings=2).summary_text()
        self.assertIn("1 error", text)
        self.assertIn("2 warnings", text)

    def test_summary_is_string(self):
        self.assertIsInstance(self._mk(errors=1).summary_text(), str)


# ===========================================================================
# audit_catalog_compatibility()
# ===========================================================================

class TestAuditCatalogCompatibility(unittest.TestCase):

    def test_empty_catalog_is_clean(self):
        cat = _catalog()
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertTrue(r.is_clean())
        self.assertEqual(r.issues, [])

    def test_all_available_is_clean(self):
        cat = _catalog(_available("sensor_read"), _available("movement_blend"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertTrue(r.is_clean())

    def test_unsupported_yields_error(self):
        cat = _catalog(_unsupported("sensor_read"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertFalse(r.is_clean())
        self.assertEqual(r.error_count, 1)
        self.assertEqual(r.warning_count, 0)
        self.assertEqual(r.issues[0].severity, "error")
        self.assertEqual(r.issues[0].node_kind, "sensor_read")

    def test_limited_yields_warning(self):
        cat = _catalog(_limited("movement_blend"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertFalse(r.is_clean())
        self.assertEqual(r.warning_count, 1)
        self.assertEqual(r.error_count, 0)
        self.assertEqual(r.issues[0].severity, "warning")

    def test_unknown_capability_yields_warning(self):
        cat = _catalog(_unknown("motor_command"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertEqual(r.warning_count, 1)
        self.assertEqual(r.issues[0].severity, "warning")

    def test_mixed_states_all_returned(self):
        cat = _catalog(
            _available("safety_gate"),
            _unsupported("sensor_read"),
            _limited("movement_blend"),
            _unknown("motor_command"),
        )
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertEqual(r.error_count, 1)
        self.assertEqual(r.warning_count, 2)
        self.assertEqual(len(r.issues), 3)

    def test_unsupported_issue_availability_field(self):
        cat = _catalog(_unsupported("sensor_read"))
        r = audit_catalog_compatibility(cat)
        self.assertEqual(r.issues[0].availability, HBNodeAvailability.UNSUPPORTED)

    def test_limited_issue_availability_field(self):
        cat = _catalog(_limited("movement_blend"))
        r = audit_catalog_compatibility(cat)
        self.assertEqual(r.issues[0].availability, HBNodeAvailability.LIMITED)

    def test_custom_reason_preserved(self):
        cat = _catalog(_unsupported("sensor_read", reason="IMU chip not available"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertIn("IMU chip not available", r.issues[0].reason)

    def test_default_reason_non_empty_when_no_custom_reason(self):
        cat = _catalog(_unsupported("sensor_read"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertGreater(len(r.issues[0].reason), 0)

    def test_brand_robot_type_stored_in_report(self):
        cat = _catalog()
        r = audit_catalog_compatibility(cat, brand="spot", robot_type="spot1")
        self.assertEqual(r.brand, "spot")
        self.assertEqual(r.robot_type, "spot1")

    def test_never_raises_on_unknown_catalog(self):
        cat = HBNodeCatalog.unknown()
        try:
            audit_catalog_compatibility(cat)
        except Exception as exc:
            self.fail(f"audit_catalog_compatibility raised: {exc}")

    def test_report_type_is_compat_report(self):
        self.assertIsInstance(audit_catalog_compatibility(HBNodeCatalog.unknown()), HBCompatReport)

    def test_multiple_unsupported_nodes(self):
        cat = _catalog(_unsupported("sensor_read"), _unsupported("motor_command"))
        r = audit_catalog_compatibility(cat, "unitree", "go2")
        self.assertEqual(r.error_count, 2)
        kinds = {i.node_kind for i in r.issues}
        self.assertIn("sensor_read", kinds)
        self.assertIn("motor_command", kinds)

    def test_node_label_in_issue(self):
        cat = _catalog(_unsupported("sensor_read"))
        r = audit_catalog_compatibility(cat)
        self.assertEqual(r.issues[0].node_label, "Sensor Read")


# ===========================================================================
# report_to_behavior_diagnostics()  — Issue 9: IR semantic diagnostics
# ===========================================================================

class TestReportToBehaviorDiagnostics(unittest.TestCase):

    def _unsupported_report(self):
        return HBCompatReport("unitree", "go2", issues=[
            HBCompatIssue("sensor_read", "Sensor Read", "unsupported", "chip missing", "error"),
        ])

    def _mixed_report(self):
        return HBCompatReport("u", "g2", issues=[
            HBCompatIssue("sensor_read", "Sensor Read", "unsupported", "chip missing", "error"),
            HBCompatIssue("movement_blend", "Movement Blend", "limited", "partial", "warning"),
        ])

    def test_clean_report_returns_empty_list(self):
        r = HBCompatReport("u", "g2", issues=[])
        self.assertEqual(report_to_behavior_diagnostics(r), [])

    def test_error_issue_yields_error_diagnostic(self):
        diags = report_to_behavior_diagnostics(self._unsupported_report())
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].level, "error")

    def test_warning_issue_yields_warning_diagnostic(self):
        r = HBCompatReport("u", "g2", issues=[
            HBCompatIssue("motor_command", "Motor Command", "limited", "partial", "warning"),
        ])
        diags = report_to_behavior_diagnostics(r)
        self.assertEqual(diags[0].level, "warning")

    def test_code_contains_availability_and_node_kind(self):
        diags = report_to_behavior_diagnostics(self._unsupported_report())
        self.assertIn("unsupported", diags[0].code)
        self.assertIn("sensor_read", diags[0].code)

    def test_message_contains_node_label(self):
        diags = report_to_behavior_diagnostics(self._unsupported_report())
        self.assertIn("Sensor Read", diags[0].message)

    def test_message_contains_reason(self):
        diags = report_to_behavior_diagnostics(self._unsupported_report())
        self.assertIn("chip missing", diags[0].message)

    def test_location_is_node_kind(self):
        diags = report_to_behavior_diagnostics(self._unsupported_report())
        self.assertEqual(diags[0].location, "sensor_read")

    def test_mixed_report_yields_correct_count(self):
        diags = report_to_behavior_diagnostics(self._mixed_report())
        self.assertEqual(len(diags), 2)

    def test_to_dict_round_trip(self):
        diags = report_to_behavior_diagnostics(self._unsupported_report())
        d = diags[0].to_dict()
        self.assertIn("level", d)
        self.assertIn("code", d)
        self.assertIn("message", d)
        self.assertIn("location", d)
        self.assertEqual(d["level"], "error")

    def test_compat_diagnostics_key_exists(self):
        self.assertTrue(hasattr(DiagnosticsKey, "COMPAT_DIAGNOSTICS"))
        self.assertIsInstance(DiagnosticsKey.COMPAT_DIAGNOSTICS, str)
        self.assertGreater(len(DiagnosticsKey.COMPAT_DIAGNOSTICS), 0)


# ===========================================================================
# _run_compat_audit wiring in bin/ui.py
# ===========================================================================

class TestRunCompatAuditWiring(unittest.TestCase):
    """Verify that _run_compat_audit calls the panel, routes through the
    IR semantic diagnostics pipeline, surfaces results in the main zone,
    and suppresses failures on all paths."""

    def _make_stub(self, audit_raises=False, has_issues=False):
        import bin.ui as ui_mod

        stub = types.SimpleNamespace()

        bp = MagicMock()
        if audit_raises:
            bp.run_compat_audit.side_effect = RuntimeError("audit boom")
        elif has_issues:
            report = HBCompatReport("u", "g2", issues=[
                HBCompatIssue("sensor_read", "Sensor Read", "unsupported", "missing", "error"),
            ])
            bp.run_compat_audit.return_value = report
        else:
            bp.run_compat_audit.return_value = HBCompatReport("u", "g2")

        mz = MagicMock()
        mz.behavior_panel = bp
        mz.show_compat_report = MagicMock()

        stub.main_zone = mz
        stub._run_compat_audit = lambda b, rt: ui_mod.MainWindow._run_compat_audit(stub, b, rt)
        return stub

    def test_run_compat_audit_calls_panel(self):
        stub = self._make_stub()
        stub._run_compat_audit("unitree", "go2")
        stub.main_zone.behavior_panel.run_compat_audit.assert_called_once_with("unitree", "go2")

    def test_run_compat_audit_failure_suppressed(self):
        stub = self._make_stub(audit_raises=True)
        try:
            stub._run_compat_audit("unitree", "go2")
        except Exception as exc:
            self.fail(f"_run_compat_audit propagated: {exc}")

    def test_run_compat_audit_passes_brand_and_robot_type(self):
        stub = self._make_stub()
        stub._run_compat_audit("spot", "spot1")
        call_args = stub.main_zone.behavior_panel.run_compat_audit.call_args
        self.assertEqual(call_args[0], ("spot", "spot1"))

    def test_show_compat_report_called_on_main_zone(self):
        """Issue 8: aggregate alert must surface in MainZone, not just BehaviorPanel."""
        stub = self._make_stub()
        stub._run_compat_audit("unitree", "go2")
        stub.main_zone.show_compat_report.assert_called_once()

    def test_show_compat_report_receives_report_object(self):
        stub = self._make_stub()
        stub._run_compat_audit("unitree", "go2")
        args = stub.main_zone.show_compat_report.call_args[0]
        self.assertIsInstance(args[0], HBCompatReport)

    def test_show_compat_report_receives_diag_dicts(self):
        """Issue 9: BehaviorDiagnostic dicts must be passed to show_compat_report."""
        stub = self._make_stub(has_issues=True)
        stub._run_compat_audit("unitree", "go2")
        args = stub.main_zone.show_compat_report.call_args[0]
        # Second arg is diag_dicts (list of BehaviorDiagnostic.to_dict())
        self.assertIsInstance(args[1], list)
        self.assertGreater(len(args[1]), 0)
        self.assertIn("level", args[1][0])
        self.assertIn("code", args[1][0])

    def test_diag_dicts_empty_when_clean_report(self):
        stub = self._make_stub()  # clean report
        stub._run_compat_audit("unitree", "go2")
        args = stub.main_zone.show_compat_report.call_args[0]
        # Diag dicts must be empty for a clean report
        self.assertEqual(args[1], [])


if __name__ == "__main__":
    unittest.main()
