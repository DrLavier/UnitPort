#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration test — Circle 1 Step 1.8 supplement.

Validates the full mock-backed heartbeat channel chain:
  HBDraftPayload (UI edit simulation)
  → compile (HBMockChannel)
  → run (HBMockChannel)
  → poll_diagnostics (HBMockChannel)
  → event state update (HBEventStateEntry.from_dict)
  → IO mapping derivation (_build_io_mappings_from_snapshot)

All tests are pure Python — no QApplication, no Qt widget instantiation.
A ConfigurableMockChannel subclass injects event_updates on poll_diagnostics()
to exercise the event-state update path (HBMockChannel always returns [] on poll).
"""

import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pure Python channel types — no Qt dependency
from system.behavior.hb_channel import (
    HBMockChannel,
    HBCompileRequest,
    HBDryRunRequest,
    HBRunRequest,
    HBDiagnosticsSnapshot,
    IHBChannel,
)


# ---------------------------------------------------------------------------
# Lazy import helpers — behavior_panel pure-Python types
# ---------------------------------------------------------------------------

def _get_panel_module():
    """Import bin.layout.behavior_panel once; reuse on subsequent calls."""
    if "bin.layout.behavior_panel" not in sys.modules:
        import bin.layout.behavior_panel  # noqa: F401
    return sys.modules["bin.layout.behavior_panel"]


def _panel_types():
    """Return (HBDraftPayload, HBPolicyDraft, HBEventStateEntry, HBIOMapping)."""
    mod = _get_panel_module()
    return (
        mod.HBDraftPayload,
        mod.HBPolicyDraft,
        mod.HBEventStateEntry,
        mod.HBIOMapping,
    )


# ---------------------------------------------------------------------------
# ConfigurableMockChannel — injects event_updates on poll_diagnostics()
# ---------------------------------------------------------------------------

class ConfigurableMockChannel(HBMockChannel):
    """Extends HBMockChannel to return configurable event_updates on poll.

    The base HBMockChannel always returns event_updates=[] from
    poll_diagnostics().  This subclass injects a supplied list so that
    tests can exercise the event-state update path without Qt.
    """

    def __init__(self, poll_event_updates=None):
        super().__init__()
        self._poll_event_updates = list(poll_event_updates or [])

    def set_poll_event_updates(self, updates):
        self._poll_event_updates = list(updates)

    def poll_diagnostics(self, behavior_ref: str) -> HBDiagnosticsSnapshot:
        snap = super().poll_diagnostics(behavior_ref)
        if self._poll_event_updates and self._mock_running:
            return HBDiagnosticsSnapshot(
                behavior_ref=snap.behavior_ref,
                status=snap.status,
                heartbeat_status=snap.heartbeat_status,
                event_updates=list(self._poll_event_updates),
                sensor_snapshot_summary=snap.sensor_snapshot_summary,
                control_output_summary=snap.control_output_summary,
                tick_count=snap.tick_count,
                last_tick_ms=snap.last_tick_ms,
                timestamp=snap.timestamp,
            )
        return snap


# ===========================================================================
# TestHBDraftPayloadConstruction — HBDraftPayload construction + mutation
# ===========================================================================

class TestHBDraftPayloadConstruction(unittest.TestCase):
    """Simulate UI edit by constructing and mutating HBDraftPayload."""

    def setUp(self):
        self.HBDraftPayload, self.HBPolicyDraft, _, _ = _panel_types()

    def test_default_has_non_empty_script(self):
        draft = self.HBDraftPayload.default()
        self.assertIsInstance(draft.script, str)
        self.assertGreater(len(draft.script), 0)

    def test_default_has_policy(self):
        draft = self.HBDraftPayload.default()
        self.assertIsInstance(draft.policy, self.HBPolicyDraft)

    def test_default_authoring_mode_is_navigator(self):
        # Default authoring mode updated to "navigator" (product simplification)
        draft = self.HBDraftPayload.default()
        self.assertEqual(draft.authoring_mode, "navigator")

    def test_ui_edit_script_changes_payload(self):
        """Simulate user editing the script field in the UI."""
        draft = self.HBDraftPayload.default()
        new_script = "# user-edited\nlisten_sensor('imu', interval=0.1)"
        draft.script = new_script
        self.assertEqual(draft.script, new_script)

    def test_ui_edit_policy_tick_ms(self):
        """Simulate user changing tick_ms in the policy UI."""
        draft = self.HBDraftPayload.default()
        draft.policy.tick_ms = 250
        self.assertEqual(draft.policy.tick_ms, 250)

    def test_ui_edit_authoring_mode_switch(self):
        """Simulate switching from canvas to script mode."""
        draft = self.HBDraftPayload.default()
        draft.authoring_mode = "script"
        self.assertEqual(draft.authoring_mode, "script")

    def test_to_dict_contains_script(self):
        draft = self.HBDraftPayload.default()
        d = draft.to_dict()
        self.assertIn("script", d)
        self.assertIsInstance(d["script"], str)

    def test_to_dict_contains_policy_with_tick_ms(self):
        draft = self.HBDraftPayload.default()
        d = draft.to_dict()
        self.assertIn("policy", d)
        self.assertIn("tick_ms", d["policy"])

    def test_to_dict_contains_io_bindings_list(self):
        draft = self.HBDraftPayload.default()
        d = draft.to_dict()
        self.assertIn("io_bindings", d)
        self.assertIsInstance(d["io_bindings"], list)

    def test_from_dict_roundtrip_preserves_edits(self):
        draft = self.HBDraftPayload.default()
        draft.script = "# roundtrip"
        draft.policy.tick_ms = 500
        draft.authoring_mode = "script"
        restored = self.HBDraftPayload.from_dict(draft.to_dict())
        self.assertEqual(restored.script, draft.script)
        self.assertEqual(restored.policy.tick_ms, draft.policy.tick_ms)
        self.assertEqual(restored.authoring_mode, draft.authoring_mode)

    def test_mutation_does_not_affect_subsequent_defaults(self):
        """Mutating one draft does not affect fresh defaults."""
        draft = self.HBDraftPayload.default()
        draft.script = "mutated"
        fresh = self.HBDraftPayload.default()
        self.assertNotEqual(fresh.script, "mutated")


# ===========================================================================
# TestHBCompileChain — compile() request/response chain
# ===========================================================================

class TestHBCompileChain(unittest.TestCase):

    def setUp(self):
        self.HBDraftPayload, *_ = _panel_types()
        self.channel = HBMockChannel()

    def _draft(self, script="# test", tick_ms=100):
        d = self.HBDraftPayload.default()
        d.script = script
        d.policy.tick_ms = tick_ms
        return d

    def test_compile_returns_ok_true(self):
        resp = self.channel.compile(HBCompileRequest("node_42", self._draft()))
        self.assertTrue(resp.ok)

    def test_compile_produces_unique_artifact_ids(self):
        req = HBCompileRequest("node_42", self._draft())
        r1 = self.channel.compile(req)
        r2 = self.channel.compile(req)
        self.assertNotEqual(r1.artifact_id, r2.artifact_id)

    def test_compile_artifact_id_contains_behavior_ref(self):
        resp = self.channel.compile(HBCompileRequest("walk_node", self._draft()))
        self.assertIn("walk_node", resp.artifact_id)

    def test_compile_error_count_zero(self):
        resp = self.channel.compile(HBCompileRequest("node_1", self._draft()))
        self.assertEqual(resp.error_count, 0)

    def test_compile_warning_count_zero(self):
        resp = self.channel.compile(HBCompileRequest("node_1", self._draft()))
        self.assertEqual(resp.warning_count, 0)

    def test_compile_increments_compile_count(self):
        req = HBCompileRequest("node_1", self._draft())
        self.assertEqual(self.channel.compile_count, 0)
        self.channel.compile(req)
        self.assertEqual(self.channel.compile_count, 1)
        self.channel.compile(req)
        self.assertEqual(self.channel.compile_count, 2)

    def test_compile_request_preserves_draft_payload(self):
        """compile() receives the exact draft that was passed in."""
        draft = self._draft(script="# custom", tick_ms=200)
        req = HBCompileRequest("node_7", draft)
        self.assertIs(req.draft_payload, draft)
        self.assertEqual(req.draft_payload.script, "# custom")
        self.assertEqual(req.draft_payload.policy.tick_ms, 200)

    def test_compile_message_non_empty(self):
        resp = self.channel.compile(HBCompileRequest("node_1", self._draft()))
        self.assertGreater(len(resp.message), 0)


# ===========================================================================
# TestHBRunChain — run() request/response chain
# ===========================================================================

class TestHBRunChain(unittest.TestCase):

    def setUp(self):
        self.HBDraftPayload, *_ = _panel_types()
        self.channel = HBMockChannel()

    def _compile_then_run(self, behavior_ref="node_42"):
        draft = self.HBDraftPayload.default()
        self.channel.compile(HBCompileRequest(behavior_ref, draft))
        return self.channel.run(HBRunRequest(behavior_ref, draft))

    def test_run_returns_ok_true(self):
        self.assertTrue(self._compile_then_run().ok)

    def test_run_status_is_started(self):
        self.assertEqual(self._compile_then_run().status, "started")

    def test_run_heartbeat_status_is_active(self):
        self.assertEqual(self._compile_then_run().heartbeat_status, "active")

    def test_run_sets_mock_running(self):
        self.assertFalse(self.channel.mock_running)
        self._compile_then_run()
        self.assertTrue(self.channel.mock_running)

    def test_run_increments_run_count(self):
        self.assertEqual(self.channel.run_count, 0)
        self._compile_then_run()
        self.assertEqual(self.channel.run_count, 1)

    def test_run_resets_tick_count_on_second_run(self):
        draft = self.HBDraftPayload.default()
        run_req = HBRunRequest("n1", draft)
        self.channel.run(run_req)
        self.channel.poll_diagnostics("n1")
        self.channel.poll_diagnostics("n1")
        self.assertEqual(self.channel.mock_tick_count, 2)
        self.channel.run(run_req)  # second run resets
        self.assertEqual(self.channel.mock_tick_count, 0)

    def test_run_sensor_summary_non_empty(self):
        resp = self._compile_then_run()
        self.assertIsInstance(resp.sensor_snapshot_summary, dict)
        self.assertGreater(len(resp.sensor_snapshot_summary), 0)

    def test_run_control_summary_non_empty(self):
        resp = self._compile_then_run()
        self.assertIsInstance(resp.control_output_summary, dict)
        self.assertGreater(len(resp.control_output_summary), 0)

    def test_run_initial_tick_count_is_zero(self):
        self.assertEqual(self._compile_then_run().tick_count, 0)


# ===========================================================================
# TestHBPollDiagnosticsChain — poll_diagnostics() sequence
# ===========================================================================

class TestHBPollDiagnosticsChain(unittest.TestCase):

    def setUp(self):
        self.HBDraftPayload, *_ = _panel_types()
        self.channel = HBMockChannel()
        draft = self.HBDraftPayload.default()
        self.channel.run(HBRunRequest("test_node", draft))

    def test_poll_status_is_running_after_run(self):
        snap = self.channel.poll_diagnostics("test_node")
        self.assertEqual(snap.status, "running")

    def test_poll_heartbeat_status_is_active_after_run(self):
        snap = self.channel.poll_diagnostics("test_node")
        self.assertEqual(snap.heartbeat_status, "active")

    def test_poll_increments_tick_count(self):
        s1 = self.channel.poll_diagnostics("test_node")
        s2 = self.channel.poll_diagnostics("test_node")
        s3 = self.channel.poll_diagnostics("test_node")
        self.assertEqual(s1.tick_count, 1)
        self.assertEqual(s2.tick_count, 2)
        self.assertEqual(s3.tick_count, 3)

    def test_poll_behavior_ref_matches_argument(self):
        snap = self.channel.poll_diagnostics("test_node")
        self.assertEqual(snap.behavior_ref, "test_node")

    def test_poll_sensor_summary_non_empty_when_running(self):
        snap = self.channel.poll_diagnostics("test_node")
        self.assertGreater(len(snap.sensor_snapshot_summary), 0)

    def test_poll_control_summary_non_empty_when_running(self):
        snap = self.channel.poll_diagnostics("test_node")
        self.assertGreater(len(snap.control_output_summary), 0)

    def test_poll_status_idle_before_run(self):
        ch = HBMockChannel()
        snap = ch.poll_diagnostics("any")
        self.assertEqual(snap.status, "idle")
        self.assertEqual(snap.heartbeat_status, "inactive")

    def test_poll_sensor_summary_empty_before_run(self):
        ch = HBMockChannel()
        snap = ch.poll_diagnostics("any")
        self.assertEqual(snap.sensor_snapshot_summary, {})

    def test_poll_timestamp_is_non_empty_string(self):
        snap = self.channel.poll_diagnostics("test_node")
        self.assertIsInstance(snap.timestamp, str)
        self.assertGreater(len(snap.timestamp), 0)

    def test_three_polls_accumulate_monotonically(self):
        counts = [self.channel.poll_diagnostics("test_node").tick_count for _ in range(3)]
        self.assertEqual(counts, [1, 2, 3])


# ===========================================================================
# TestHBDryRunChain — dry_run() event_updates path
# ===========================================================================

class TestHBDryRunChain(unittest.TestCase):

    def setUp(self):
        self.HBDraftPayload, _, self.HBEventStateEntry, _ = _panel_types()
        self.channel = HBMockChannel()

    def _dry_run(self, tick_count=5):
        draft = self.HBDraftPayload.default()
        return self.channel.dry_run(HBDryRunRequest("walk_node", draft, tick_count))

    def test_dry_run_returns_ok_true(self):
        self.assertTrue(self._dry_run().ok)

    def test_dry_run_simulated_ticks_matches_request(self):
        resp = self._dry_run(tick_count=3)
        self.assertEqual(resp.simulated_ticks, 3)

    def test_dry_run_event_updates_length_matches_ticks(self):
        resp = self._dry_run(tick_count=5)
        self.assertEqual(len(resp.event_updates), 5)

    def test_dry_run_event_updates_parseable_as_entries(self):
        """Each event_update dict is parseable via HBEventStateEntry.from_dict()."""
        for raw in self._dry_run(tick_count=4).event_updates:
            entry = self.HBEventStateEntry.from_dict(raw)
            self.assertIsInstance(entry.event_name, str)
            self.assertIsInstance(entry.state, str)

    def test_dry_run_event_names_are_tick(self):
        for raw in self._dry_run(tick_count=3).event_updates:
            self.assertEqual(raw.get("event_name"), "tick")

    def test_dry_run_event_states_are_active(self):
        for raw in self._dry_run(tick_count=3).event_updates:
            self.assertEqual(self.HBEventStateEntry.from_dict(raw).state, "active")

    def test_dry_run_event_source_is_mock_hb(self):
        for raw in self._dry_run(tick_count=2).event_updates:
            self.assertEqual(self.HBEventStateEntry.from_dict(raw).source, "mock_hb")

    def test_dry_run_sensor_snapshot_has_imu(self):
        self.assertIn("imu", self._dry_run().sensor_snapshot_summary)

    def test_dry_run_control_summary_has_base_action(self):
        self.assertIn("base_action", self._dry_run().control_output_summary)

    def test_dry_run_clamps_tick_count_max_20(self):
        self.assertLessEqual(self._dry_run(tick_count=100).simulated_ticks, 20)

    def test_dry_run_clamps_tick_count_min_1(self):
        self.assertGreaterEqual(self._dry_run(tick_count=0).simulated_ticks, 1)


# ===========================================================================
# TestHBEventStateEntryParsing — HBEventStateEntry.from_dict()
# ===========================================================================

class TestHBEventStateEntryParsing(unittest.TestCase):

    def setUp(self):
        _, _, self.HBEventStateEntry, _ = _panel_types()

    def _parse(self, d):
        return self.HBEventStateEntry.from_dict(d)

    def test_parse_full_dict(self):
        entry = self._parse({
            "event_name": "posture_check", "state": "active",
            "source": "imu", "timestamp": "T42", "reason": "nominal",
        })
        self.assertEqual(entry.event_name, "posture_check")
        self.assertEqual(entry.state, "active")
        self.assertEqual(entry.source, "imu")
        self.assertEqual(entry.timestamp, "T42")
        self.assertEqual(entry.reason, "nominal")

    def test_parse_missing_reason_defaults_empty_string(self):
        entry = self._parse({"event_name": "ev", "state": "inactive",
                              "source": "s", "timestamp": "T0"})
        self.assertEqual(entry.reason, "")

    def test_parse_missing_state_defaults_inactive(self):
        entry = self._parse({"event_name": "ev", "source": "s", "timestamp": "T0"})
        self.assertEqual(entry.state, "inactive")

    def test_parse_empty_dict_no_raise(self):
        entry = self._parse({})
        self.assertIsInstance(entry.event_name, str)
        self.assertIsInstance(entry.state, str)

    def test_parse_extra_keys_ignored(self):
        entry = self._parse({
            "event_name": "ev", "state": "active", "source": "s",
            "timestamp": "T1", "extra_field": "ignored",
        })
        self.assertEqual(entry.event_name, "ev")

    def test_parse_non_string_values_coerced_to_str(self):
        entry = self._parse({"event_name": 42, "state": True,
                              "source": None, "timestamp": 0})
        self.assertIsInstance(entry.event_name, str)
        self.assertIsInstance(entry.state, str)
        self.assertIsInstance(entry.source, str)


# ===========================================================================
# TestBuildIOMappingsFromSnapshot — _build_io_mappings_from_snapshot
# ===========================================================================

class TestBuildIOMappingsFromSnapshot(unittest.TestCase):
    """Test _build_io_mappings_from_snapshot via a stub binding (no Qt instantiation)."""

    def setUp(self):
        _, _, _, self.HBIOMapping = _panel_types()
        mod = _get_panel_module()
        stub = types.SimpleNamespace()
        self._build = lambda sensor, ctrl: mod.BehaviorPanel._build_io_mappings_from_snapshot(
            stub, sensor, ctrl
        )

    def test_empty_both_returns_empty_list(self):
        self.assertEqual(self._build({}, {}), [])

    def test_flat_sensor_key_creates_inbound_mapping(self):
        result = self._build({"battery": 85}, {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].signal, "battery")
        self.assertEqual(result[0].direction, "inbound")
        self.assertEqual(result[0].status, "active")

    def test_nested_sensor_key_expands_to_sub_keys(self):
        result = self._build({"imu": {"pitch": 0.02, "roll": -0.01}}, {})
        signals = {m.signal for m in result}
        self.assertIn("imu.pitch", signals)
        self.assertIn("imu.roll", signals)
        self.assertEqual(len(result), 2)

    def test_nested_sensor_all_directions_are_inbound(self):
        result = self._build({"imu": {"pitch": 0.0, "yaw": 0.0}}, {})
        for m in result:
            self.assertEqual(m.direction, "inbound")

    def test_control_key_creates_outbound_mapping(self):
        result = self._build({}, {"base_action": "walk"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].signal, "base_action")
        self.assertEqual(result[0].direction, "outbound")

    def test_mixed_sensor_and_control_both_directions(self):
        result = self._build({"battery": 85}, {"base_action": "walk", "confidence": 1.0})
        directions = {m.direction for m in result}
        self.assertIn("inbound", directions)
        self.assertIn("outbound", directions)

    def test_mock_poll_snapshot_data_produces_expected_counts(self):
        """Mock poll snapshot → 3 inbound (imu.pitch, imu.roll, battery) + 2 outbound."""
        sensor = {"imu": {"pitch": 0.0, "roll": 0.0}, "battery": 85}
        control = {"base_action": "walk", "confidence": 1.0}
        result = self._build(sensor, control)
        inbound = [m for m in result if m.direction == "inbound"]
        outbound = [m for m in result if m.direction == "outbound"]
        self.assertEqual(len(inbound), 3)
        self.assertEqual(len(outbound), 2)

    def test_all_mappings_are_hbiomapping_instances(self):
        result = self._build({"imu": {"pitch": 0.0}}, {"base_action": "walk"})
        for m in result:
            self.assertIsInstance(m, self.HBIOMapping)

    def test_target_param_equals_signal_for_mock_data(self):
        result = self._build({"battery": 85}, {})
        self.assertEqual(result[0].target_param, result[0].signal)


# ===========================================================================
# TestHBFullChain — end-to-end compile → run → poll sequence
# ===========================================================================

class TestHBFullChain(unittest.TestCase):

    def setUp(self):
        self.HBDraftPayload, *_ = _panel_types()

    def test_compile_run_three_polls_correct_state(self):
        """Complete chain: draft → compile → run → poll × 3 with state assertions."""
        ch = HBMockChannel()
        draft = self.HBDraftPayload.default()
        ref = "walk_behavior"

        compile_resp = ch.compile(HBCompileRequest(ref, draft))
        self.assertTrue(compile_resp.ok)
        self.assertEqual(ch.compile_count, 1)

        run_resp = ch.run(HBRunRequest(ref, draft))
        self.assertTrue(run_resp.ok)
        self.assertEqual(run_resp.status, "started")
        self.assertTrue(ch.mock_running)

        for i in range(1, 4):
            snap = ch.poll_diagnostics(ref)
            self.assertEqual(snap.status, "running")
            self.assertEqual(snap.tick_count, i)

    def test_two_channels_maintain_independent_state(self):
        ch1 = HBMockChannel()
        ch2 = HBMockChannel()
        draft = self.HBDraftPayload.default()

        ch1.run(HBRunRequest("n1", draft))
        ch1.poll_diagnostics("n1")
        ch1.poll_diagnostics("n1")

        snap2 = ch2.poll_diagnostics("n1")
        self.assertEqual(snap2.status, "idle")
        self.assertEqual(snap2.tick_count, 0)
        self.assertEqual(ch1.mock_tick_count, 2)

    def test_reset_clears_all_accumulated_state(self):
        ch = HBMockChannel()
        draft = self.HBDraftPayload.default()
        ch.compile(HBCompileRequest("nx", draft))
        ch.run(HBRunRequest("nx", draft))
        ch.poll_diagnostics("nx")
        ch.reset()
        self.assertEqual(ch.compile_count, 0)
        self.assertEqual(ch.run_count, 0)
        self.assertFalse(ch.mock_running)
        self.assertEqual(ch.mock_tick_count, 0)

    def test_edited_draft_propagates_into_compile_request(self):
        """UI edits to draft are carried into the compile/run request envelopes."""
        ch = HBMockChannel()
        draft = self.HBDraftPayload.default()
        draft.script = "# user-modified"
        draft.policy.tick_ms = 200
        draft.authoring_mode = "script"

        req = HBCompileRequest("node_edit", draft)
        compile_resp = ch.compile(req)
        self.assertTrue(compile_resp.ok)
        self.assertEqual(req.draft_payload.script, "# user-modified")
        self.assertEqual(req.draft_payload.policy.tick_ms, 200)
        self.assertEqual(req.draft_payload.authoring_mode, "script")

    def test_hbmockchannel_satisfies_ihbchannel_protocol(self):
        self.assertIsInstance(HBMockChannel(), IHBChannel)

    def test_configurable_channel_satisfies_ihbchannel_protocol(self):
        self.assertIsInstance(ConfigurableMockChannel(), IHBChannel)


# ===========================================================================
# TestHBEventUpdateIntegration — poll event_updates → event state objects
# ===========================================================================

class TestHBEventUpdateIntegration(unittest.TestCase):
    """Test the event-state update path using ConfigurableMockChannel."""

    def setUp(self):
        self.HBDraftPayload, _, self.HBEventStateEntry, _ = _panel_types()

    def _start(self, poll_event_updates):
        ch = ConfigurableMockChannel(poll_event_updates=poll_event_updates)
        draft = self.HBDraftPayload.default()
        ch.run(HBRunRequest("node_ev", draft))
        return ch

    def test_poll_delivers_configured_event_updates_when_running(self):
        updates = [{"event_name": "posture_check", "state": "active",
                    "source": "imu", "timestamp": "T1", "reason": ""}]
        ch = self._start(updates)
        snap = ch.poll_diagnostics("node_ev")
        self.assertEqual(len(snap.event_updates), 1)
        self.assertEqual(snap.event_updates[0]["event_name"], "posture_check")

    def test_poll_event_updates_parse_into_entry_objects(self):
        updates = [
            {"event_name": "balance_monitor", "state": "active",
             "source": "sensor_pipeline", "timestamp": "T10", "reason": "ok"},
            {"event_name": "velocity_limit", "state": "warning",
             "source": "control_pipeline", "timestamp": "T10", "reason": "near limit"},
        ]
        ch = self._start(updates)
        snap = ch.poll_diagnostics("node_ev")
        entries = [self.HBEventStateEntry.from_dict(u) for u in snap.event_updates]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].event_name, "balance_monitor")
        self.assertEqual(entries[0].state, "active")
        self.assertEqual(entries[1].event_name, "velocity_limit")
        self.assertEqual(entries[1].state, "warning")

    def test_all_state_values_preserved_through_parse(self):
        """active, inactive, warning, error all round-trip correctly."""
        for state in ("active", "inactive", "warning", "error"):
            with self.subTest(state=state):
                updates = [{"event_name": "ev", "state": state,
                             "source": "s", "timestamp": "T"}]
                ch = self._start(updates)
                snap = ch.poll_diagnostics("node_ev")
                entry = self.HBEventStateEntry.from_dict(snap.event_updates[0])
                self.assertEqual(entry.state, state)

    def test_poll_returns_no_event_updates_when_not_running(self):
        """Configured updates are NOT injected until run() is called."""
        updates = [{"event_name": "ev", "state": "active",
                    "source": "s", "timestamp": "T"}]
        ch = ConfigurableMockChannel(poll_event_updates=updates)
        snap = ch.poll_diagnostics("node_ev")  # never ran
        self.assertEqual(snap.event_updates, [])

    def test_full_diagnostics_pipeline_simulation(self):
        """compile → run → poll (with event updates) → parse event state entries."""
        updates = [{"event_name": "tick", "state": "active",
                    "source": "mock_hb", "timestamp": "T0", "reason": ""}]
        ch = ConfigurableMockChannel(poll_event_updates=updates)
        draft = self.HBDraftPayload.default()

        compile_resp = ch.compile(HBCompileRequest("sim_node", draft))
        self.assertTrue(compile_resp.ok)

        run_resp = ch.run(HBRunRequest("sim_node", draft))
        self.assertTrue(run_resp.ok)
        self.assertEqual(run_resp.status, "started")

        snap = ch.poll_diagnostics("sim_node")
        self.assertEqual(snap.status, "running")
        self.assertGreater(snap.tick_count, 0)
        self.assertGreater(len(snap.event_updates), 0)

        entries = [self.HBEventStateEntry.from_dict(u) for u in snap.event_updates]
        self.assertEqual(entries[0].event_name, "tick")
        self.assertEqual(entries[0].state, "active")
        self.assertEqual(entries[0].source, "mock_hb")


if __name__ == "__main__":
    unittest.main()
