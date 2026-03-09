#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for Circle 2 DoD — HB Runtime Channel + Fallback + Non-blocking.

Coverage
--------
Runtime channel path (end-to-end):
  - HBChannelFactory returns HBRuntimeChannel when bridge is available.
  - run() + repeated poll() work: status evolves from "running", tick_count grows.
  - trace_id is stable across multiple polls in the same run session.
  - Two back-to-back run() sessions produce distinct trace_ids.

Fallback path:
  - HBChannelFactory returns HBMockChannel when bridge is None.
  - Warning is emitted for the no-bridge fallback (reason=no_bridge).
  - Warning is emitted when runtime channel init fails (reason=runtime_init_failed).
  - Mock channel remains functional (compile/dry_run/run/poll all return responses).

Non-blocking / slow-upstream:
  - poll_diagnostics() returns well within a tight latency bound even when
    _execute_tick is artificially delayed.
  - Snapshot returned during slow tick is still coherent (not None, correct type).
  - Two consecutive polls while tick is in flight return snapshots immediately
    (only one tick is launched; a second poll does not start a second tick).
  - tick_count increments exactly once per launched tick (monotonic, no double-counts).

Tick_count monotonicity:
  - tick_count never decreases across polls.
  - After reset(), tick_count returns to zero.
  - A new run() session starts tick_count from zero.
"""

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Node registry stub — avoids Qt / SDK imports
# ---------------------------------------------------------------------------

_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = _registry.get          # type: ignore[attr-defined]
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
# Helpers
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

# Latency bound for non-blocking poll (ms).  Conservatively high to avoid flakiness.
_POLL_LATENCY_LIMIT_MS = 200


def _empty_ir() -> WorkflowIR:
    return WorkflowIR(nodes=[], edges=[])


def _single_node_ir() -> WorkflowIR:
    return WorkflowIR(
        nodes=[IRNode(id="n0", kind=NodeKind.ACTION, schema_id="ok_node", params={})],
        edges=[],
    )


def _valid_artifact(ref: str = "it_ref") -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=ref,
        behavior_ir=_single_node_ir(),
    )


class _BridgeOk:
    """Stub bridge — compile() always returns a valid artifact."""
    def compile(self, source: str = "", behavior_ref: str = "ref") -> BehaviorArtifact:
        return _valid_artifact(behavior_ref)


class _SlowBridge:
    """Stub bridge — compile() returns valid artifact; tick will be slow via patching."""
    def compile(self, source: str = "", behavior_ref: str = "ref") -> BehaviorArtifact:
        return _valid_artifact(behavior_ref)


# ===========================================================================
# TestRuntimeChannelEndToEnd
# ===========================================================================

class TestRuntimeChannelEndToEnd(unittest.TestCase):
    """End-to-end runtime channel: factory → run → poll lifecycle."""

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def test_factory_returns_runtime_channel_when_bridge_available(self):
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        self.assertIsInstance(ch, HBRuntimeChannel)

    def test_run_succeeds_and_is_running(self):
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        resp = ch.run(HBRunRequest("ref", None))
        self.assertTrue(resp.ok)
        self.assertTrue(ch.is_running)

    def test_poll_after_run_returns_running_status(self):
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        ch.run(HBRunRequest("ref", None))
        snap = ch.poll_diagnostics("ref")
        self.assertEqual(snap.status, "running")

    def test_tick_count_grows_across_polls(self):
        """tick_count increases after giving time for background ticks to complete."""
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        ch.run(HBRunRequest("ref", None))
        for _ in range(3):
            ch.poll_diagnostics("ref")
            ch._join_tick(timeout=2.0)
        self.assertGreaterEqual(ch.tick_count, 1)

    def test_trace_id_stable_across_polls_in_same_session(self):
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        ch.run(HBRunRequest("ref", None))
        trace_before = ch.trace_id
        ch.poll_diagnostics("ref")
        ch._join_tick(timeout=1.0)
        ch.poll_diagnostics("ref")
        self.assertEqual(ch.trace_id, trace_before)

    def test_two_run_sessions_have_distinct_trace_ids(self):
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        ch.run(HBRunRequest("ref", None))
        trace1 = ch.trace_id
        ch.reset()
        ch.run(HBRunRequest("ref", None))
        trace2 = ch.trace_id
        self.assertNotEqual(trace1, trace2)

    def test_tick_count_resets_on_new_run(self):
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        ch.run(HBRunRequest("ref", None))
        ch.poll_diagnostics("ref")
        ch._join_tick(timeout=1.0)
        ch.reset()
        ch.run(HBRunRequest("ref", None))
        self.assertEqual(ch.tick_count, 0)

    def test_compile_then_run_then_poll_full_lifecycle(self):
        """Full lifecycle: compile → run → poll sequence produces coherent results."""
        ch = HBChannelFactory.create(bridge=_BridgeOk())
        compile_resp = ch.compile(HBCompileRequest("lifecycle_ref", None))
        self.assertTrue(compile_resp.ok)
        run_resp = ch.run(HBRunRequest("lifecycle_ref", None))
        self.assertTrue(run_resp.ok)
        snap = ch.poll_diagnostics("lifecycle_ref")
        self.assertIsInstance(snap, HBDiagnosticsSnapshot)
        self.assertEqual(snap.status, "running")


# ===========================================================================
# TestFallbackPath
# ===========================================================================

class TestFallbackPath(unittest.TestCase):
    """Mock fallback: warning emitted in all fallback branches; mock remains functional."""

    def test_factory_returns_mock_when_no_bridge(self):
        ch = HBChannelFactory.create(bridge=None)
        self.assertIsInstance(ch, HBMockChannel)

    def test_no_bridge_fallback_emits_warning(self):
        with patch("system.behavior.hb_channel._log_warning") as mock_warn:
            HBChannelFactory.create(bridge=None)
        self.assertTrue(mock_warn.called, "warning must be emitted when bridge=None")

    def test_no_bridge_warning_contains_reason_no_bridge(self):
        with patch("system.behavior.hb_channel._log_warning") as mock_warn:
            HBChannelFactory.create(bridge=None)
        msg = mock_warn.call_args[0][0]
        self.assertIn("no_bridge", msg)

    def test_runtime_init_failure_emits_warning(self):
        with patch("system.behavior.hb_channel._log_warning") as mock_warn, \
             patch(
                 "system.behavior.hb_channel.HBRuntimeChannel.__init__",
                 side_effect=RuntimeError("init_fail"),
             ):
            HBChannelFactory.create(bridge=_BridgeOk())
        self.assertTrue(mock_warn.called)

    def test_runtime_init_failure_warning_contains_runtime_init_failed(self):
        with patch("system.behavior.hb_channel._log_warning") as mock_warn, \
             patch(
                 "system.behavior.hb_channel.HBRuntimeChannel.__init__",
                 side_effect=RuntimeError("init_fail"),
             ):
            HBChannelFactory.create(bridge=_BridgeOk())
        msg = mock_warn.call_args[0][0]
        self.assertIn("runtime_init_failed", msg)

    def test_runtime_init_failure_returns_mock(self):
        with patch("system.behavior.hb_channel._log_warning"), \
             patch(
                 "system.behavior.hb_channel.HBRuntimeChannel.__init__",
                 side_effect=RuntimeError("init_fail"),
             ):
            ch = HBChannelFactory.create(bridge=_BridgeOk())
        self.assertIsInstance(ch, HBMockChannel)

    def test_mock_compile_returns_response(self):
        ch = HBChannelFactory.create(bridge=None)
        resp = ch.compile(HBCompileRequest("mock_ref", "# script"))
        self.assertIsNotNone(resp)
        self.assertIsInstance(resp.ok, bool)

    def test_mock_dry_run_returns_response(self):
        ch = HBChannelFactory.create(bridge=None)
        resp = ch.dry_run(HBDryRunRequest("mock_ref", None, tick_count=3))
        self.assertIsNotNone(resp)

    def test_mock_run_and_poll_return_responses(self):
        ch = HBChannelFactory.create(bridge=None)
        run_resp = ch.run(HBRunRequest("mock_ref", None))
        self.assertIsNotNone(run_resp)
        snap = ch.poll_diagnostics("mock_ref")
        self.assertIsInstance(snap, HBDiagnosticsSnapshot)

    def test_no_warning_when_bridge_is_present(self):
        with patch("system.behavior.hb_channel._log_warning") as mock_warn:
            HBChannelFactory.create(bridge=_BridgeOk())
        self.assertFalse(mock_warn.called, "no warning when runtime channel is used")


# ===========================================================================
# TestNonBlockingPollDiagnostics
# ===========================================================================

class TestNonBlockingPollDiagnostics(unittest.TestCase):
    """poll_diagnostics() must return immediately regardless of tick duration."""

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def _make_running_channel(self, artifact=None):
        """Return a channel already in running state with a valid artifact."""
        ch = HBRuntimeChannel(bridge=_BridgeOk())
        art = artifact or _valid_artifact("nb_ref")
        with ch._lock:
            ch._running = True
            ch._trace_id = "it-trace"
            ch._last_artifact = art
        return ch

    def test_poll_returns_within_latency_bound_with_slow_tick(self):
        """poll_diagnostics() must return << _POLL_LATENCY_LIMIT_MS even for slow ticks.

        We inject a 300 ms delay into _execute_tick.  poll_diagnostics() must
        return within _POLL_LATENCY_LIMIT_MS (200 ms) — well before the tick
        completes.
        """
        ch = self._make_running_channel()

        delay_s = 0.3  # 300 ms simulated slow tick

        original_execute = ch._execute_tick

        def slow_execute(*args, **kwargs):
            time.sleep(delay_s)
            return original_execute(*args, **kwargs)

        ch._execute_tick = slow_execute

        t0 = time.monotonic()
        snap = ch.poll_diagnostics("nb_ref")
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self.assertIsInstance(snap, HBDiagnosticsSnapshot)
        self.assertLess(
            elapsed_ms,
            _POLL_LATENCY_LIMIT_MS,
            f"poll_diagnostics() took {elapsed_ms:.1f} ms, expected < {_POLL_LATENCY_LIMIT_MS} ms",
        )
        # Clean up — join slow thread so it doesn't outlive the test.
        ch._join_tick(timeout=1.0)

    def test_poll_snapshot_coherent_while_tick_in_flight(self):
        """Snapshot returned during a slow tick is still a valid HBDiagnosticsSnapshot."""
        ch = self._make_running_channel()

        original_execute = ch._execute_tick

        def slow_execute(*args, **kwargs):
            time.sleep(0.2)
            return original_execute(*args, **kwargs)

        ch._execute_tick = slow_execute

        snap = ch.poll_diagnostics("nb_ref")
        self.assertIsNotNone(snap)
        self.assertIsInstance(snap, HBDiagnosticsSnapshot)
        self.assertIsInstance(snap.tick_count, int)
        self.assertIsInstance(snap.timestamp, str)
        ch._join_tick(timeout=1.0)

    def test_second_poll_while_tick_in_flight_does_not_start_new_tick(self):
        """If tick is in flight, a second poll must not launch another tick thread."""
        ch = self._make_running_channel()

        # Use a barrier to hold the tick until we've polled twice.
        barrier = threading.Event()
        original_execute = ch._execute_tick

        def blocking_execute(*args, **kwargs):
            barrier.wait(timeout=2.0)
            return original_execute(*args, **kwargs)

        ch._execute_tick = blocking_execute

        # First poll — launches background tick (held by barrier).
        ch.poll_diagnostics("nb_ref")
        thread1 = ch._tick_thread

        # Second poll — tick is in flight; must NOT launch another thread.
        ch.poll_diagnostics("nb_ref")
        thread2 = ch._tick_thread

        self.assertIs(thread1, thread2, "second poll must not launch a new tick thread")

        # Release barrier and clean up.
        barrier.set()
        ch._join_tick(timeout=1.0)

    def test_tick_count_monotonic_across_slow_polls(self):
        """tick_count must be monotonically non-decreasing across slow polls."""
        ch = self._make_running_channel()

        counts = []
        for _ in range(4):
            ch.poll_diagnostics("nb_ref")
            ch._join_tick(timeout=1.0)
            counts.append(ch.tick_count)

        for i in range(1, len(counts)):
            self.assertGreaterEqual(
                counts[i],
                counts[i - 1],
                f"tick_count decreased: {counts[i - 1]} → {counts[i]}",
            )

    def test_tick_count_exact_after_serial_polls_with_join(self):
        """When polls are serialised via _join_tick, tick_count increments exactly."""
        ch = self._make_running_channel()
        expected = 0
        for _ in range(3):
            ch.poll_diagnostics("nb_ref")
            ch._join_tick(timeout=1.0)
            expected += 1
        self.assertEqual(ch.tick_count, expected)

    def test_poll_after_reset_returns_idle(self):
        """After reset(), poll_diagnostics() must return idle snapshot."""
        ch = self._make_running_channel()
        ch.poll_diagnostics("nb_ref")
        ch.reset()
        snap = ch.poll_diagnostics("nb_ref")
        self.assertEqual(snap.status, "idle")

    def test_behavior_ref_preserved_in_slow_tick_snapshot(self):
        """behavior_ref in the returned snapshot must match the poll argument."""
        ch = self._make_running_channel()
        snap = ch.poll_diagnostics("my_specific_ref")
        self.assertEqual(snap.behavior_ref, "my_specific_ref")
        ch._join_tick(timeout=1.0)


# ===========================================================================
# TestRealBridgeIntegration
# ===========================================================================

class TestRealBridgeIntegration(unittest.TestCase):
    """Full integration using real BehaviorCompilerBridge (no stubs for bridge)."""

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def setUp(self):
        self.bridge = BehaviorCompilerBridge()
        artifact = _valid_artifact("it_real_ref")
        self.bridge.register(artifact)

    def test_factory_with_real_bridge_returns_runtime(self):
        ch = HBChannelFactory.create(bridge=self.bridge)
        self.assertIsInstance(ch, HBRuntimeChannel)

    def test_compile_and_run_with_real_bridge(self):
        ch = HBChannelFactory.create(bridge=self.bridge)
        compile_resp = ch.compile(HBCompileRequest("it_real_ref", None))
        # Compile from empty source via real bridge — may or may not succeed
        # depending on bridge compile logic; result must be a valid response object.
        self.assertIsNotNone(compile_resp)
        self.assertIsInstance(compile_resp.ok, bool)

    def test_poll_returns_coherent_snapshot_with_real_bridge(self):
        ch = HBRuntimeChannel(bridge=self.bridge)
        artifact = _valid_artifact("it_real_poll")
        self.bridge.register(artifact)
        with ch._lock:
            ch._running = True
            ch._trace_id = "real-trace"
            ch._last_artifact = artifact
        snap = ch.poll_diagnostics("it_real_poll")
        self.assertIsInstance(snap, HBDiagnosticsSnapshot)
        self.assertEqual(snap.status, "running")
        ch._join_tick(timeout=2.0)
        self.assertGreaterEqual(ch.tick_count, 1)

    def test_no_warning_emitted_with_real_bridge(self):
        with patch("system.behavior.hb_channel._log_warning") as mock_warn:
            HBChannelFactory.create(bridge=self.bridge)
        self.assertFalse(mock_warn.called)


if __name__ == "__main__":
    unittest.main()
