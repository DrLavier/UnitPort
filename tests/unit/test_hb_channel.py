#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system/behavior/hb_channel.py — Circle 1 Step 1.3.

Coverage
--------
- All seven envelope dataclasses: field presence, types, and defaults
- IHBChannel protocol: HBMockChannel satisfies isinstance check
- HBMockChannel.compile(): returns ok response with unique artifact_id
- HBMockChannel.dry_run(): returns correct tick count; populates event_updates
- HBMockChannel.run(): sets mock_running; returns started status
- HBMockChannel.poll_diagnostics(): reflects running state; tick_count grows
- HBMockChannel.reset(): clears all mutable state
- State sequencing: compile -> run -> poll -> reset roundtrip

Isolation
---------
- No QApplication. No Qt. Pure-Python only.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.hb_channel import (  # noqa: E402
    HBCompileRequest,
    HBCompileResponse,
    HBDiagnosticsSnapshot,
    HBDryRunRequest,
    HBDryRunResponse,
    HBMockChannel,
    HBRunRequest,
    HBRunResponse,
    IHBChannel,
)


# ===========================================================================
# Envelope: HBCompileRequest / HBCompileResponse
# ===========================================================================

class TestHBCompileEnvelopes(unittest.TestCase):

    def test_request_fields(self):
        req = HBCompileRequest(behavior_ref="node_a", draft_payload={"script": ""})
        self.assertEqual(req.behavior_ref, "node_a")
        self.assertEqual(req.draft_payload, {"script": ""})

    def test_response_defaults(self):
        resp = HBCompileResponse(ok=True, artifact_id="a1", error_count=0, warning_count=0)
        self.assertEqual(resp.diagnostics, [])
        self.assertEqual(resp.message, "")

    def test_response_with_diagnostics(self):
        diag = [{"severity": "error", "message": "bad node"}]
        resp = HBCompileResponse(ok=False, artifact_id="", error_count=1,
                                 warning_count=0, diagnostics=diag)
        self.assertFalse(resp.ok)
        self.assertEqual(len(resp.diagnostics), 1)


# ===========================================================================
# Envelope: HBDryRunRequest / HBDryRunResponse
# ===========================================================================

class TestHBDryRunEnvelopes(unittest.TestCase):

    def test_request_default_tick_count(self):
        req = HBDryRunRequest(behavior_ref="n", draft_payload=None)
        self.assertEqual(req.tick_count, 5)

    def test_request_custom_tick_count(self):
        req = HBDryRunRequest(behavior_ref="n", draft_payload=None, tick_count=3)
        self.assertEqual(req.tick_count, 3)

    def test_response_defaults(self):
        resp = HBDryRunResponse(ok=True, simulated_ticks=5)
        self.assertEqual(resp.event_updates, [])
        self.assertEqual(resp.sensor_snapshot_summary, {})
        self.assertEqual(resp.control_output_summary, {})
        self.assertEqual(resp.diagnostics, [])
        self.assertEqual(resp.message, "")


# ===========================================================================
# Envelope: HBRunRequest / HBRunResponse
# ===========================================================================

class TestHBRunEnvelopes(unittest.TestCase):

    def test_request_default_scenario(self):
        req = HBRunRequest(behavior_ref="n", draft_payload=None)
        self.assertEqual(req.scenario, {})

    def test_response_defaults(self):
        resp = HBRunResponse(ok=True, status="started", heartbeat_status="active")
        self.assertEqual(resp.tick_count, 0)
        self.assertEqual(resp.event_updates, [])
        self.assertEqual(resp.diagnostics, [])
        self.assertEqual(resp.message, "")


# ===========================================================================
# Envelope: HBDiagnosticsSnapshot
# ===========================================================================

class TestHBDiagnosticsSnapshot(unittest.TestCase):

    def test_required_fields(self):
        snap = HBDiagnosticsSnapshot(
            behavior_ref="n",
            status="idle",
            heartbeat_status="inactive",
        )
        self.assertEqual(snap.behavior_ref, "n")
        self.assertEqual(snap.status, "idle")
        self.assertEqual(snap.heartbeat_status, "inactive")

    def test_defaults(self):
        snap = HBDiagnosticsSnapshot(behavior_ref="n", status="idle",
                                     heartbeat_status="inactive")
        self.assertEqual(snap.event_updates, [])
        self.assertEqual(snap.sensor_snapshot_summary, {})
        self.assertEqual(snap.control_output_summary, {})
        self.assertEqual(snap.tick_count, 0)
        self.assertAlmostEqual(snap.last_tick_ms, 0.0)
        self.assertEqual(snap.timestamp, "")


# ===========================================================================
# IHBChannel protocol satisfaction
# ===========================================================================

class TestIHBChannelProtocol(unittest.TestCase):

    def test_mock_satisfies_protocol(self):
        ch = HBMockChannel()
        self.assertIsInstance(ch, IHBChannel)

    def test_mock_has_all_protocol_methods(self):
        ch = HBMockChannel()
        for method in ("compile", "dry_run", "run", "poll_diagnostics"):
            self.assertTrue(callable(getattr(ch, method, None)),
                            msg=f"HBMockChannel missing method: {method}")


# ===========================================================================
# HBMockChannel.compile()
# ===========================================================================

class TestHBMockChannelCompile(unittest.TestCase):

    def setUp(self):
        self.ch = HBMockChannel()

    def test_returns_ok_true(self):
        resp = self.ch.compile(HBCompileRequest("ref", None))
        self.assertTrue(resp.ok)

    def test_returns_compile_response_type(self):
        resp = self.ch.compile(HBCompileRequest("ref", None))
        self.assertIsInstance(resp, HBCompileResponse)

    def test_artifact_id_contains_ref(self):
        resp = self.ch.compile(HBCompileRequest("my_node", None))
        self.assertIn("my_node", resp.artifact_id)

    def test_zero_errors_and_warnings(self):
        resp = self.ch.compile(HBCompileRequest("ref", None))
        self.assertEqual(resp.error_count, 0)
        self.assertEqual(resp.warning_count, 0)

    def test_artifact_id_unique_across_calls(self):
        r1 = self.ch.compile(HBCompileRequest("ref", None))
        r2 = self.ch.compile(HBCompileRequest("ref", None))
        self.assertNotEqual(r1.artifact_id, r2.artifact_id)

    def test_compile_count_increments(self):
        self.ch.compile(HBCompileRequest("ref", None))
        self.ch.compile(HBCompileRequest("ref", None))
        self.assertEqual(self.ch.compile_count, 2)

    def test_message_non_empty(self):
        resp = self.ch.compile(HBCompileRequest("ref", None))
        self.assertGreater(len(resp.message), 0)


# ===========================================================================
# HBMockChannel.dry_run()
# ===========================================================================

class TestHBMockChannelDryRun(unittest.TestCase):

    def setUp(self):
        self.ch = HBMockChannel()

    def test_returns_ok_true(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None))
        self.assertTrue(resp.ok)

    def test_returns_dry_run_response_type(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None))
        self.assertIsInstance(resp, HBDryRunResponse)

    def test_simulated_ticks_matches_request(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None, tick_count=3))
        self.assertEqual(resp.simulated_ticks, 3)

    def test_event_updates_count_matches_ticks(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None, tick_count=4))
        self.assertEqual(len(resp.event_updates), 4)

    def test_event_updates_have_required_keys(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None, tick_count=2))
        for ev in resp.event_updates:
            for key in ("event_name", "state", "source", "timestamp", "reason"):
                self.assertIn(key, ev)

    def test_sensor_snapshot_summary_non_empty(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None))
        self.assertGreater(len(resp.sensor_snapshot_summary), 0)

    def test_control_output_summary_non_empty(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None))
        self.assertGreater(len(resp.control_output_summary), 0)

    def test_tick_count_clamped_to_20(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None, tick_count=100))
        self.assertEqual(resp.simulated_ticks, 20)

    def test_tick_count_minimum_1(self):
        resp = self.ch.dry_run(HBDryRunRequest("ref", None, tick_count=0))
        self.assertEqual(resp.simulated_ticks, 1)


# ===========================================================================
# HBMockChannel.run()
# ===========================================================================

class TestHBMockChannelRun(unittest.TestCase):

    def setUp(self):
        self.ch = HBMockChannel()

    def test_returns_ok_true(self):
        resp = self.ch.run(HBRunRequest("ref", None))
        self.assertTrue(resp.ok)

    def test_returns_run_response_type(self):
        resp = self.ch.run(HBRunRequest("ref", None))
        self.assertIsInstance(resp, HBRunResponse)

    def test_status_is_started(self):
        resp = self.ch.run(HBRunRequest("ref", None))
        self.assertEqual(resp.status, "started")

    def test_heartbeat_status_active(self):
        resp = self.ch.run(HBRunRequest("ref", None))
        self.assertEqual(resp.heartbeat_status, "active")

    def test_sets_mock_running(self):
        self.assertFalse(self.ch.mock_running)
        self.ch.run(HBRunRequest("ref", None))
        self.assertTrue(self.ch.mock_running)

    def test_run_count_increments(self):
        self.ch.run(HBRunRequest("ref", None))
        self.ch.run(HBRunRequest("ref", None))
        self.assertEqual(self.ch.run_count, 2)

    def test_message_contains_ref(self):
        resp = self.ch.run(HBRunRequest("my_ref", None))
        self.assertIn("my_ref", resp.message)


# ===========================================================================
# HBMockChannel.poll_diagnostics()
# ===========================================================================

class TestHBMockChannelPoll(unittest.TestCase):

    def setUp(self):
        self.ch = HBMockChannel()

    def test_idle_before_run(self):
        snap = self.ch.poll_diagnostics("ref")
        self.assertEqual(snap.status, "idle")
        self.assertEqual(snap.heartbeat_status, "inactive")

    def test_running_after_run(self):
        self.ch.run(HBRunRequest("ref", None))
        snap = self.ch.poll_diagnostics("ref")
        self.assertEqual(snap.status, "running")
        self.assertEqual(snap.heartbeat_status, "active")

    def test_tick_count_increments_on_each_poll(self):
        self.ch.run(HBRunRequest("ref", None))
        self.ch.poll_diagnostics("ref")
        self.ch.poll_diagnostics("ref")
        snap = self.ch.poll_diagnostics("ref")
        self.assertEqual(snap.tick_count, 3)

    def test_behavior_ref_preserved(self):
        snap = self.ch.poll_diagnostics("my_node")
        self.assertEqual(snap.behavior_ref, "my_node")

    def test_timestamp_non_empty(self):
        snap = self.ch.poll_diagnostics("ref")
        self.assertGreater(len(snap.timestamp), 0)

    def test_returns_snapshot_type(self):
        snap = self.ch.poll_diagnostics("ref")
        self.assertIsInstance(snap, HBDiagnosticsSnapshot)

    def test_sensor_summary_populated_when_running(self):
        self.ch.run(HBRunRequest("ref", None))
        snap = self.ch.poll_diagnostics("ref")
        self.assertGreater(len(snap.sensor_snapshot_summary), 0)

    def test_sensor_summary_empty_when_idle(self):
        snap = self.ch.poll_diagnostics("ref")
        self.assertEqual(snap.sensor_snapshot_summary, {})


# ===========================================================================
# HBMockChannel.reset()
# ===========================================================================

class TestHBMockChannelReset(unittest.TestCase):

    def test_reset_clears_compile_count(self):
        ch = HBMockChannel()
        ch.compile(HBCompileRequest("ref", None))
        ch.reset()
        self.assertEqual(ch.compile_count, 0)

    def test_reset_clears_run_count(self):
        ch = HBMockChannel()
        ch.run(HBRunRequest("ref", None))
        ch.reset()
        self.assertEqual(ch.run_count, 0)

    def test_reset_clears_mock_running(self):
        ch = HBMockChannel()
        ch.run(HBRunRequest("ref", None))
        ch.reset()
        self.assertFalse(ch.mock_running)

    def test_reset_clears_tick_count(self):
        ch = HBMockChannel()
        ch.run(HBRunRequest("ref", None))
        ch.poll_diagnostics("ref")
        ch.reset()
        self.assertEqual(ch.mock_tick_count, 0)

    def test_after_reset_poll_reports_idle(self):
        ch = HBMockChannel()
        ch.run(HBRunRequest("ref", None))
        ch.reset()
        snap = ch.poll_diagnostics("ref")
        self.assertEqual(snap.status, "idle")


# ===========================================================================
# Full sequence: compile -> run -> poll -> reset
# ===========================================================================

class TestHBMockChannelSequence(unittest.TestCase):

    def test_full_sequence(self):
        ch = HBMockChannel()

        # 1. compile
        compile_resp = ch.compile(HBCompileRequest("seq_node", None))
        self.assertTrue(compile_resp.ok)
        self.assertEqual(ch.compile_count, 1)

        # 2. run
        run_resp = ch.run(HBRunRequest("seq_node", None))
        self.assertTrue(run_resp.ok)
        self.assertTrue(ch.mock_running)

        # 3. poll x3
        for i in range(3):
            snap = ch.poll_diagnostics("seq_node")
            self.assertEqual(snap.status, "running")
            self.assertEqual(snap.tick_count, i + 1)

        # 4. reset
        ch.reset()
        snap = ch.poll_diagnostics("seq_node")
        self.assertEqual(snap.status, "idle")
        self.assertEqual(snap.tick_count, 0)


if __name__ == "__main__":
    unittest.main()
