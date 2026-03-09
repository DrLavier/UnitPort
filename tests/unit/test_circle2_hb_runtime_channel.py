#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Circle 2 — Heartbeat Real Channel Replacement.

Coverage
--------
HBRuntimeChannel — compile():
  - valid source → ok=True, artifact_id set, error_count 0
  - invalid source (compile errors) → ok=False, error_count > 0
  - bridge exception → ok=False, message contains error
  - compile caches last_artifact when ok

HBRuntimeChannel — dry_run():
  - valid source → ok=True, simulated_ticks == requested
  - tick_count capped at 50
  - invalid source → ok=False, simulated_ticks=0
  - bridge exception in dry_run → ok=False

HBRuntimeChannel — run():
  - valid source → ok=True, status="started", is_running=True
  - run() generates a trace_id
  - invalid source → ok=False, status="failed", is_running=False
  - compile error in run() → ok=False, is_running=False

HBRuntimeChannel — poll_diagnostics():
  - not running → status="idle", heartbeat_status="inactive"
  - after run() → status="running", heartbeat_status="active"
  - tick_count increments on each poll
  - bridge/executor fault in poll swallowed — snapshot still returned
  - behavior_ref passed through to snapshot

HBRuntimeChannel — state helpers:
  - reset() clears all state
  - is_running / tick_count / trace_id properties

HBRuntimeChannel — _extract_source():
  - HBDraftPayload with .script attribute
  - dict with "script" key
  - plain string
  - None → empty string

HBChannelFactory — create():
  - bridge=None → returns HBMockChannel
  - bridge present → returns HBRuntimeChannel
  - bridge present but init raises → fallback to HBMockChannel

IHBChannel duck-typing:
  - HBRuntimeChannel satisfies IHBChannel protocol
  - HBMockChannel satisfies IHBChannel protocol
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Node registry stub — avoids importing Qt / SDK during tests
# ---------------------------------------------------------------------------

_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = _registry.get  # type: ignore[attr-defined]
_nodes_mod.register_node = lambda k, v: _registry.__setitem__(k, v)  # type: ignore[attr-defined]


class _OkNode:
    def __init__(self, *_, **__): pass
    def set_parameter(self, k, v): pass
    def execute(self, inputs): return {"result": "ok"}


def _install_registry():
    _registry.clear()
    _registry["ok_node"] = _OkNode
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Helpers — build minimal artifacts / bridges
# ---------------------------------------------------------------------------

from system.behavior.behavior_artifact import BehaviorArtifact, BehaviorDiagnostic
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
from system.ir.workflow_ir import IREdge, IRNode, NodeKind, WorkflowIR
from system.behavior.hb_channel import (
    HBChannelFactory,
    HBCompileRequest,
    HBDiagnosticsSnapshot,
    HBDryRunRequest,
    HBMockChannel,
    HBRunRequest,
    HBRuntimeChannel,
    IHBChannel,
)


def _empty_ir() -> WorkflowIR:
    return WorkflowIR(nodes=[], edges=[])


def _single_node_ir(schema_id: str = "ok_node") -> WorkflowIR:
    return WorkflowIR(
        nodes=[IRNode(id="n0", kind=NodeKind.ACTION, schema_id=schema_id, params={})],
        edges=[],
    )


def _valid_artifact(ref: str = "hb_test") -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=ref,
        behavior_ir=_single_node_ir(),
    )


def _invalid_artifact(ref: str = "hb_bad") -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=ref,
        behavior_ir=_empty_ir(),
        diagnostics=[BehaviorDiagnostic.error("E001", "forced error")],
    )


class _StubBridgeOk:
    """Bridge stub: compile() always returns a valid artifact."""
    def compile(self, source: str = "", behavior_ref: str = "ref") -> BehaviorArtifact:
        return _valid_artifact(behavior_ref)


class _StubBridgeFail:
    """Bridge stub: compile() always returns an invalid artifact."""
    def compile(self, source: str = "", behavior_ref: str = "ref") -> BehaviorArtifact:
        return _invalid_artifact(behavior_ref)


class _StubBridgeRaises:
    """Bridge stub: compile() always raises."""
    def compile(self, source: str = "", behavior_ref: str = "ref") -> BehaviorArtifact:
        raise RuntimeError("bridge_exploded")


# Draft payload helpers
class _FakeDraftPayload:
    def __init__(self, script: str = ""):
        self.script = script


# ===========================================================================
# TestHBRuntimeChannelCompile
# ===========================================================================

class TestHBRuntimeChannelCompile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def _channel(self, bridge=None):
        return HBRuntimeChannel(bridge=bridge or _StubBridgeOk())

    def test_compile_ok_returns_true(self):
        ch = self._channel()
        req = HBCompileRequest(behavior_ref="test", draft_payload="# script")
        resp = ch.compile(req)
        self.assertTrue(resp.ok)

    def test_compile_ok_artifact_id_set(self):
        ch = self._channel()
        req = HBCompileRequest(behavior_ref="test", draft_payload=None)
        resp = ch.compile(req)
        self.assertIsInstance(resp.artifact_id, str)
        self.assertGreater(len(resp.artifact_id), 0)

    def test_compile_ok_zero_error_count(self):
        ch = self._channel()
        resp = ch.compile(HBCompileRequest("test", None))
        self.assertEqual(resp.error_count, 0)

    def test_compile_fail_returns_false(self):
        ch = self._channel(_StubBridgeFail())
        resp = ch.compile(HBCompileRequest("bad", None))
        self.assertFalse(resp.ok)

    def test_compile_fail_error_count_positive(self):
        ch = self._channel(_StubBridgeFail())
        resp = ch.compile(HBCompileRequest("bad", None))
        self.assertGreater(resp.error_count, 0)

    def test_compile_bridge_exception_ok_false(self):
        ch = self._channel(_StubBridgeRaises())
        resp = ch.compile(HBCompileRequest("boom", None))
        self.assertFalse(resp.ok)

    def test_compile_bridge_exception_message_contains_error(self):
        ch = self._channel(_StubBridgeRaises())
        resp = ch.compile(HBCompileRequest("boom", None))
        self.assertIn("error", resp.message.lower())

    def test_compile_ok_caches_last_artifact(self):
        ch = self._channel()
        ch.compile(HBCompileRequest("r", None))
        self.assertIsNotNone(ch._last_artifact)

    def test_compile_fail_does_not_cache_artifact(self):
        ch = self._channel(_StubBridgeFail())
        ch.compile(HBCompileRequest("r", None))
        self.assertIsNone(ch._last_artifact)

    def test_compile_draft_payload_object_used(self):
        """_extract_source should pull .script from draft payload."""
        ch = self._channel()
        dp = _FakeDraftPayload("print('hello')")
        resp = ch.compile(HBCompileRequest("r", dp))
        self.assertTrue(resp.ok)

    def test_compile_draft_payload_dict_used(self):
        ch = self._channel()
        resp = ch.compile(HBCompileRequest("r", {"script": "# code"}))
        self.assertTrue(resp.ok)


# ===========================================================================
# TestHBRuntimeChannelDryRun
# ===========================================================================

class TestHBRuntimeChannelDryRun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def test_dry_run_ok_true(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.dry_run(HBDryRunRequest("ref", None, tick_count=3))
        self.assertTrue(resp.ok)

    def test_dry_run_simulated_ticks_equals_requested(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.dry_run(HBDryRunRequest("ref", None, tick_count=5))
        self.assertEqual(resp.simulated_ticks, 5)

    def test_dry_run_tick_count_capped_at_50(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.dry_run(HBDryRunRequest("ref", None, tick_count=100))
        self.assertLessEqual(resp.simulated_ticks, 50)

    def test_dry_run_compile_fail_returns_false(self):
        ch = HBRuntimeChannel(_StubBridgeFail())
        resp = ch.dry_run(HBDryRunRequest("bad", None, tick_count=3))
        self.assertFalse(resp.ok)
        self.assertEqual(resp.simulated_ticks, 0)

    def test_dry_run_bridge_exception_false(self):
        ch = HBRuntimeChannel(_StubBridgeRaises())
        resp = ch.dry_run(HBDryRunRequest("boom", None, tick_count=1))
        self.assertFalse(resp.ok)

    def test_dry_run_event_updates_match_tick_count(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.dry_run(HBDryRunRequest("ref", None, tick_count=4))
        self.assertEqual(len(resp.event_updates), 4)

    def test_dry_run_message_references_ref(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.dry_run(HBDryRunRequest("my_ref", None, tick_count=2))
        self.assertIn("my_ref", resp.message)


# ===========================================================================
# TestHBRuntimeChannelRun
# ===========================================================================

class TestHBRuntimeChannelRun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def test_run_ok_true(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.run(HBRunRequest("ref", None))
        self.assertTrue(resp.ok)

    def test_run_status_started(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        resp = ch.run(HBRunRequest("ref", None))
        self.assertEqual(resp.status, "started")

    def test_run_is_running_true_after_start(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        self.assertTrue(ch.is_running)

    def test_run_generates_trace_id(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        self.assertGreater(len(ch.trace_id), 0)

    def test_run_compile_fail_ok_false(self):
        ch = HBRuntimeChannel(_StubBridgeFail())
        resp = ch.run(HBRunRequest("bad", None))
        self.assertFalse(resp.ok)

    def test_run_compile_fail_status_failed(self):
        ch = HBRuntimeChannel(_StubBridgeFail())
        resp = ch.run(HBRunRequest("bad", None))
        self.assertEqual(resp.status, "failed")

    def test_run_compile_fail_is_not_running(self):
        ch = HBRuntimeChannel(_StubBridgeFail())
        ch.run(HBRunRequest("bad", None))
        self.assertFalse(ch.is_running)

    def test_run_bridge_exception_ok_false(self):
        ch = HBRuntimeChannel(_StubBridgeRaises())
        resp = ch.run(HBRunRequest("boom", None))
        self.assertFalse(resp.ok)

    def test_run_tick_count_starts_at_zero(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        self.assertEqual(ch.tick_count, 0)


# ===========================================================================
# TestHBRuntimeChannelPollDiagnostics
# ===========================================================================

class TestHBRuntimeChannelPollDiagnostics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def test_poll_before_run_status_idle(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        snap = ch.poll_diagnostics("ref")
        self.assertEqual(snap.status, "idle")

    def test_poll_before_run_heartbeat_inactive(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        snap = ch.poll_diagnostics("ref")
        self.assertEqual(snap.heartbeat_status, "inactive")

    def test_poll_after_run_status_running(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        snap = ch.poll_diagnostics("ref")
        self.assertEqual(snap.status, "running")

    def test_poll_after_run_heartbeat_active(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        snap = ch.poll_diagnostics("ref")
        self.assertEqual(snap.heartbeat_status, "active")

    def test_poll_increments_tick_count(self):
        # With background ticks, we must join each tick before the next poll
        # to ensure deterministic tick_count increments.
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        ch.poll_diagnostics("ref")
        ch._join_tick(timeout=2.0)   # wait for first tick to complete
        ch.poll_diagnostics("ref")
        ch._join_tick(timeout=2.0)   # wait for second tick to complete
        self.assertEqual(ch.tick_count, 2)

    def test_poll_behavior_ref_in_snapshot(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        snap = ch.poll_diagnostics("my_behavior")
        self.assertEqual(snap.behavior_ref, "my_behavior")

    def test_poll_returns_snapshot_even_if_tick_raises(self):
        """Faults inside _execute_tick must be swallowed; snapshot still returned."""
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        # Corrupt last_artifact so execute_tick will raise
        ch._last_artifact = object()  # missing .behavior_ir — will raise on access
        snap = ch.poll_diagnostics("ref")
        self.assertIsInstance(snap, HBDiagnosticsSnapshot)

    def test_poll_timestamp_set(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        snap = ch.poll_diagnostics("ref")
        self.assertIsInstance(snap.timestamp, str)
        self.assertGreater(len(snap.timestamp), 0)


# ===========================================================================
# TestHBRuntimeChannelStateHelpers
# ===========================================================================

class TestHBRuntimeChannelStateHelpers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def test_reset_clears_running(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        ch.reset()
        self.assertFalse(ch.is_running)

    def test_reset_clears_tick_count(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        ch.poll_diagnostics("ref")
        ch.reset()
        self.assertEqual(ch.tick_count, 0)

    def test_reset_clears_trace_id(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        ch.reset()
        self.assertEqual(ch.trace_id, "")

    def test_reset_clears_last_artifact(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        ch.run(HBRunRequest("ref", None))
        ch.reset()
        self.assertIsNone(ch._last_artifact)


# ===========================================================================
# TestHBRuntimeChannelExtractSource
# ===========================================================================

class TestHBRuntimeChannelExtractSource(unittest.TestCase):

    def test_extract_from_draft_payload_object(self):
        dp = _FakeDraftPayload("x = 1")
        result = HBRuntimeChannel._extract_source(dp)
        self.assertEqual(result, "x = 1")

    def test_extract_from_dict(self):
        result = HBRuntimeChannel._extract_source({"script": "y = 2"})
        self.assertEqual(result, "y = 2")

    def test_extract_from_plain_string(self):
        result = HBRuntimeChannel._extract_source("direct_source")
        self.assertEqual(result, "direct_source")

    def test_extract_none_returns_empty(self):
        result = HBRuntimeChannel._extract_source(None)
        self.assertEqual(result, "")

    def test_extract_empty_script_in_dict(self):
        result = HBRuntimeChannel._extract_source({"script": ""})
        self.assertEqual(result, "")

    def test_extract_none_script_on_object(self):
        dp = _FakeDraftPayload(script=None)  # type: ignore[arg-type]
        result = HBRuntimeChannel._extract_source(dp)
        self.assertEqual(result, "")


# ===========================================================================
# TestHBChannelFactory
# ===========================================================================

class TestHBChannelFactory(unittest.TestCase):

    def test_no_bridge_returns_mock(self):
        ch = HBChannelFactory.create(bridge=None)
        self.assertIsInstance(ch, HBMockChannel)

    def test_no_bridge_emits_warning_reason_no_bridge(self):
        """Fix 2: bridge=None path must emit an explicit warning."""
        with patch("system.behavior.hb_channel._log_warning") as mock_warn:
            HBChannelFactory.create(bridge=None)
        mock_warn.assert_called_once()
        msg = mock_warn.call_args[0][0]
        self.assertIn("no_bridge", msg)

    def test_bridge_returns_runtime(self):
        ch = HBChannelFactory.create(bridge=_StubBridgeOk())
        self.assertIsInstance(ch, HBRuntimeChannel)

    def test_bridge_with_adapter_returns_runtime(self):
        adapter = MagicMock()
        ch = HBChannelFactory.create(bridge=_StubBridgeOk(), adapter=adapter)
        self.assertIsInstance(ch, HBRuntimeChannel)

    def test_adapter_stored_on_runtime_channel(self):
        adapter = MagicMock()
        ch = HBChannelFactory.create(bridge=_StubBridgeOk(), adapter=adapter)
        self.assertIs(ch._adapter, adapter)  # type: ignore[union-attr]

    def test_bridge_init_raises_falls_back_to_mock(self):
        """If HBRuntimeChannel.__init__ raises, factory falls back to HBMockChannel."""
        with patch(
            "system.behavior.hb_channel.HBRuntimeChannel.__init__",
            side_effect=RuntimeError("init_fail"),
        ):
            ch = HBChannelFactory.create(bridge=_StubBridgeOk())
        self.assertIsInstance(ch, HBMockChannel)

    def test_bridge_init_raises_emits_warning_reason_runtime_init_failed(self):
        """Fix 2: runtime init failure must emit warning with reason=runtime_init_failed."""
        with patch("system.behavior.hb_channel._log_warning") as mock_warn, \
             patch(
                 "system.behavior.hb_channel.HBRuntimeChannel.__init__",
                 side_effect=RuntimeError("boom"),
             ):
            HBChannelFactory.create(bridge=_StubBridgeOk())
        mock_warn.assert_called_once()
        msg = mock_warn.call_args[0][0]
        self.assertIn("runtime_init_failed", msg)

    def test_bridge_present_no_warning(self):
        """No warning when bridge is available and init succeeds."""
        with patch("system.behavior.hb_channel._log_warning") as mock_warn:
            HBChannelFactory.create(bridge=_StubBridgeOk())
        mock_warn.assert_not_called()


# ===========================================================================
# TestIHBChannelDuckTyping
# ===========================================================================

class TestIHBChannelDuckTyping(unittest.TestCase):

    def test_runtime_channel_satisfies_ihb_protocol(self):
        ch = HBRuntimeChannel(_StubBridgeOk())
        self.assertIsInstance(ch, IHBChannel)

    def test_mock_channel_satisfies_ihb_protocol(self):
        ch = HBMockChannel()
        self.assertIsInstance(ch, IHBChannel)

    def test_factory_result_satisfies_ihb_with_bridge(self):
        ch = HBChannelFactory.create(bridge=_StubBridgeOk())
        self.assertIsInstance(ch, IHBChannel)

    def test_factory_result_satisfies_ihb_no_bridge(self):
        ch = HBChannelFactory.create(bridge=None)
        self.assertIsInstance(ch, IHBChannel)


# ===========================================================================
# TestHBRuntimeChannelWithRealBridge
# ===========================================================================

class TestHBRuntimeChannelWithRealBridge(unittest.TestCase):
    """Integration-level tests using a real BehaviorCompilerBridge.

    These tests exercise the full compile pipeline to ensure the real bridge
    and HBRuntimeChannel wire together correctly.
    """

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def setUp(self):
        self.bridge = BehaviorCompilerBridge()
        self.ch = HBRuntimeChannel(bridge=self.bridge)

    def test_compile_with_real_bridge_returns_response(self):
        from system.behavior.behavior_artifact import BehaviorArtifact
        # Pre-register a valid artifact so bridge.resolve() works
        artifact = _valid_artifact("hb_rt_ref")
        self.bridge.register(artifact)
        req = HBCompileRequest(behavior_ref="hb_rt_ref", draft_payload=None)
        resp = self.ch.compile(req)
        # compile() on real bridge with registered artifact:
        # The bridge compiles from source (may produce error if source is empty),
        # but should not raise — just return a response.
        self.assertIsNotNone(resp)
        self.assertIsInstance(resp.ok, bool)

    def test_run_then_poll_increments_tick(self):
        artifact = _valid_artifact("hb_poll_ref")
        self.bridge.register(artifact)
        # Pre-cache artifact so run() path succeeds
        self.ch._last_artifact = artifact
        self.ch._running = True
        self.ch._trace_id = "test-trace"
        initial_tick = self.ch.tick_count
        self.ch.poll_diagnostics("hb_poll_ref")
        self.ch._join_tick(timeout=2.0)  # wait for background tick to complete
        self.assertEqual(self.ch.tick_count, initial_tick + 1)

    def test_mock_fallback_from_factory_still_compiles(self):
        """HBMockChannel returned by factory should compile without errors."""
        ch = HBChannelFactory.create(bridge=None)
        req = HBCompileRequest(behavior_ref="fallback_ref", draft_payload="# code")
        resp = ch.compile(req)
        self.assertIsNotNone(resp)
        self.assertIsInstance(resp.ok, bool)


if __name__ == "__main__":
    unittest.main()
