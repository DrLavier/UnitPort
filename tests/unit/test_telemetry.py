#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 STAGE-03: Structured telemetry schema tests.

Covers:
  A. TelemetryEvent dataclass schema and to_dict() output
  B. TelemetryCollector: start/end/events behaviour
  C. emit_event() logging helper
  D. new_trace_id() UUID4 generation
  E. _scrub_credentials() privacy filter
  F. ServiceRouter: telemetry injected into every result path
"""

import sys
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.telemetry import (          # noqa: E402
    TelemetryEvent, TelemetryCollector,
    emit_event, new_trace_id, _scrub_credentials,
)
from system.service.lifecycle import (          # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult, SafetyPolicy,
)
from system.service.service_router import RouteOp, ServiceRouter  # noqa: E402
from system.service.service_registry import ServiceRegistry        # noqa: E402
from system.service.adapters.base_adapter import BaseAdapter       # noqa: E402

# ── Telemetry schema constants ────────────────────────────────────────────

_EVENT_KEYS = frozenset({
    "trace_id", "adapter_name", "brand", "operation", "stage",
    "event", "timestamp", "duration_ms", "status", "reason",
    "retryable", "attempt", "diagnostics",
})


# ── Minimal mock adapter ──────────────────────────────────────────────────

class _MockAdapter(BaseAdapter):
    def __init__(self):
        self.open_status   = "ok"
        self.pre_status    = "ok"
        self.close_status  = "ok"
        self.action_raises = None

    def connect(self, **kw):   return True
    def run_action(self, a, **p):
        if self.action_raises:
            raise self.action_raises
        return True
    def stop(self):            pass
    def get_sensor_data(self): return {"x": 1}
    def health(self):          return {"connected": True}

    def open_session(self, config=None):
        if self.open_status == "ok":
            return LifecycleResult.ok("open_session").to_dict()
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "mock failure"},
        ).to_dict()

    def preflight(self, context=None):
        if self.pre_status == "ok":
            return LifecycleResult.ok("preflight").to_dict()
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED,
            {"message": "mock failure"},
        ).to_dict()

    def close_session(self):
        if self.close_status == "ok":
            return LifecycleResult.ok("close_session").to_dict()
        return LifecycleResult.error(
            "close_session", LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "mock failure"},
        ).to_dict()


def _router_with(adapter, name="myAdapter"):
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


def _get_telemetry(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result.get("diagnostics", {}).get("telemetry", [])


# ══════════════════════════════════════════════════════════════════════════
# A — TelemetryEvent schema
# ══════════════════════════════════════════════════════════════════════════

class TestTelemetryEventSchema(unittest.TestCase):

    def _make_start(self, **overrides) -> TelemetryEvent:
        defaults = dict(
            trace_id="t1", adapter_name="a", brand="b",
            operation="run_action", stage="open_session",
            event="start", timestamp=1.0, duration_ms=None,
            status="pending", reason="", retryable=False,
            attempt=1, diagnostics={},
        )
        defaults.update(overrides)
        return TelemetryEvent(**defaults)

    def test_to_dict_has_all_13_keys(self):
        d = self._make_start().to_dict()
        self.assertEqual(frozenset(d.keys()), _EVENT_KEYS)

    def test_to_dict_start_duration_ms_is_none(self):
        d = self._make_start().to_dict()
        self.assertIsNone(d["duration_ms"])

    def test_to_dict_end_duration_ms_is_float(self):
        ev = self._make_start(event="end", duration_ms=3.14)
        self.assertIsInstance(ev.to_dict()["duration_ms"], float)

    def test_to_dict_trace_id_is_string(self):
        d = self._make_start().to_dict()
        self.assertIsInstance(d["trace_id"], str)

    def test_to_dict_retryable_is_bool(self):
        d = self._make_start().to_dict()
        self.assertIsInstance(d["retryable"], bool)

    def test_to_dict_attempt_is_int(self):
        d = self._make_start().to_dict()
        self.assertIsInstance(d["attempt"], int)
        self.assertEqual(d["attempt"], 1)

    def test_to_dict_diagnostics_is_dict(self):
        d = self._make_start().to_dict()
        self.assertIsInstance(d["diagnostics"], dict)

    def test_status_pending_for_start(self):
        d = self._make_start().to_dict()
        self.assertEqual(d["status"], "pending")

    def test_event_field_start(self):
        d = self._make_start().to_dict()
        self.assertEqual(d["event"], "start")

    def test_event_field_end(self):
        ev = self._make_start(event="end", duration_ms=1.0, status="ok")
        self.assertEqual(ev.to_dict()["event"], "end")


# ══════════════════════════════════════════════════════════════════════════
# B — TelemetryCollector
# ══════════════════════════════════════════════════════════════════════════

class TestTelemetryCollector(unittest.TestCase):

    def _collector(self, **kw) -> TelemetryCollector:
        defaults = dict(
            trace_id="tr1", adapter_name="ada", brand="brd",
            operation="run_action",
        )
        defaults.update(kw)
        return TelemetryCollector(**defaults)

    # ── start/end produce events ──────────────────────────────────────────

    def test_start_appends_one_event(self):
        c = self._collector()
        c.start("open_session")
        self.assertEqual(len(c.events), 1)

    def test_end_appends_one_event(self):
        c = self._collector()
        c.start("open_session")
        c.end("open_session", "ok")
        self.assertEqual(len(c.events), 2)

    def test_start_event_dict_has_all_keys(self):
        c = self._collector()
        c.start("open_session")
        ev = c.events[0]
        self.assertEqual(frozenset(ev.keys()), _EVENT_KEYS)

    def test_end_event_dict_has_all_keys(self):
        c = self._collector()
        c.start("execute")
        c.end("execute", "ok")
        ev = c.events[1]
        self.assertEqual(frozenset(ev.keys()), _EVENT_KEYS)

    def test_start_event_field_is_start(self):
        c = self._collector()
        c.start("execute")
        self.assertEqual(c.events[0]["event"], "start")

    def test_end_event_field_is_end(self):
        c = self._collector()
        c.start("execute")
        c.end("execute", "ok")
        self.assertEqual(c.events[1]["event"], "end")

    def test_start_duration_ms_is_none(self):
        c = self._collector()
        c.start("execute")
        self.assertIsNone(c.events[0]["duration_ms"])

    def test_end_duration_ms_is_non_negative_float(self):
        c = self._collector()
        c.start("execute")
        c.end("execute", "ok")
        self.assertIsInstance(c.events[1]["duration_ms"], float)
        self.assertGreaterEqual(c.events[1]["duration_ms"], 0.0)

    def test_end_status_propagated(self):
        c = self._collector()
        c.start("preflight")
        c.end("preflight", "error", "preflight_failed")
        self.assertEqual(c.events[1]["status"], "error")

    def test_end_reason_propagated(self):
        c = self._collector()
        c.start("preflight")
        c.end("preflight", "error", "preflight_failed")
        self.assertEqual(c.events[1]["reason"], "preflight_failed")

    def test_trace_id_consistent_across_events(self):
        c = self._collector(trace_id="abc123")
        c.start("open_session")
        c.end("open_session", "ok")
        for ev in c.events:
            self.assertEqual(ev["trace_id"], "abc123")

    def test_adapter_name_consistent(self):
        c = self._collector(adapter_name="spot_sdk")
        c.start("execute")
        c.end("execute", "ok")
        for ev in c.events:
            self.assertEqual(ev["adapter_name"], "spot_sdk")

    def test_events_property_returns_copy(self):
        c = self._collector()
        c.start("execute")
        ev1 = c.events
        c.end("execute", "ok")
        # The list retrieved before end() should not grow
        self.assertEqual(len(ev1), 1)

    def test_multiple_stages_accumulate(self):
        c = self._collector()
        for stage in ("open_session", "execute", "close_session"):
            c.start(stage)
            c.end(stage, "ok")
        self.assertEqual(len(c.events), 6)

    def test_attempt_default_is_1(self):
        c = self._collector()
        c.start("execute")
        self.assertEqual(c.events[0]["attempt"], 1)

    def test_custom_attempt_propagated(self):
        c = self._collector(attempt=3)
        c.start("execute")
        self.assertEqual(c.events[0]["attempt"], 3)

    def test_credentials_scrubbed_from_end_diagnostics(self):
        c = self._collector()
        c.start("execute")
        c.end("execute", "error", diagnostics={"auth_token": "SECRET", "op": "run_action"})
        diag = c.events[1]["diagnostics"]
        self.assertNotIn("auth_token", diag)
        self.assertIn("op", diag)

    def test_hostname_scrubbed_from_end_diagnostics(self):
        c = self._collector()
        c.start("open_session")
        c.end("open_session", "error", diagnostics={"hostname": "192.168.1.1", "code": 1})
        diag = c.events[1]["diagnostics"]
        self.assertNotIn("hostname", diag)
        self.assertIn("code", diag)


# ══════════════════════════════════════════════════════════════════════════
# C — emit_event
# ══════════════════════════════════════════════════════════════════════════

class TestEmitEvent(unittest.TestCase):

    def _make_event(self) -> TelemetryEvent:
        return TelemetryEvent(
            trace_id="t", adapter_name="a", brand="",
            operation="stop", stage="execute",
            event="start", timestamp=0.0, duration_ms=None,
            status="pending", reason="", retryable=False,
            attempt=1, diagnostics={},
        )

    def test_emit_does_not_raise(self):
        emit_event(self._make_event())

    def test_emit_end_event_does_not_raise(self):
        ev = self._make_event()
        object.__setattr__(ev, "event", "end")
        object.__setattr__(ev, "duration_ms", 1.5)
        object.__setattr__(ev, "status", "ok")
        emit_event(ev)

    def test_emit_error_event_does_not_raise(self):
        ev = self._make_event()
        object.__setattr__(ev, "event", "end")
        object.__setattr__(ev, "status", "error")
        object.__setattr__(ev, "reason", "execute_failed")
        emit_event(ev)


# ══════════════════════════════════════════════════════════════════════════
# D — new_trace_id
# ══════════════════════════════════════════════════════════════════════════

class TestNewTraceId(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(new_trace_id(), str)

    def test_is_valid_uuid4(self):
        tid = new_trace_id()
        parsed = uuid.UUID(tid, version=4)
        self.assertEqual(str(parsed), tid)

    def test_successive_calls_differ(self):
        self.assertNotEqual(new_trace_id(), new_trace_id())

    def test_nonempty(self):
        self.assertTrue(new_trace_id())


# ══════════════════════════════════════════════════════════════════════════
# E — _scrub_credentials
# ══════════════════════════════════════════════════════════════════════════

class TestScrubCredentials(unittest.TestCase):

    def test_auth_token_removed(self):
        d = _scrub_credentials({"auth_token": "abc", "x": 1})
        self.assertNotIn("auth_token", d)

    def test_hostname_removed(self):
        d = _scrub_credentials({"hostname": "10.0.0.1", "x": 1})
        self.assertNotIn("hostname", d)

    def test_safe_keys_preserved(self):
        d = _scrub_credentials({"brand": "spot", "message": "ok"})
        self.assertEqual(d, {"brand": "spot", "message": "ok"})

    def test_empty_dict_passthrough(self):
        self.assertEqual(_scrub_credentials({}), {})

    def test_both_credential_keys_removed(self):
        d = _scrub_credentials({"auth_token": "x", "hostname": "y", "z": 1})
        self.assertNotIn("auth_token", d)
        self.assertNotIn("hostname", d)
        self.assertIn("z", d)

    def test_returns_new_dict(self):
        orig = {"brand": "u"}
        result = _scrub_credentials(orig)
        self.assertIsNot(result, orig)


# ══════════════════════════════════════════════════════════════════════════
# F — ServiceRouter: telemetry in every result path
# ══════════════════════════════════════════════════════════════════════════

class TestRouterTelemetryPresence(unittest.TestCase):
    """Every execute_with_lifecycle result must contain diagnostics["telemetry"]."""

    _NAME = "myAdapter"

    def _run(self, adapter=None, op=RouteOp.RUN_ACTION, policy=None):
        if adapter is None:
            adapter = _MockAdapter()
        router = _router_with(adapter, self._NAME)
        return router.execute_with_lifecycle(self._NAME, op, policy=policy)

    # ── Happy path ───────────────────────────────────────────────────────

    def test_success_result_has_telemetry_key(self):
        result = self._run()
        self.assertIn("telemetry", result["diagnostics"])

    def test_success_telemetry_is_list(self):
        result = self._run()
        self.assertIsInstance(_get_telemetry(result), list)

    def test_success_telemetry_nonempty(self):
        result = self._run()
        self.assertGreater(len(_get_telemetry(result)), 0)

    # ── Adapter not found ────────────────────────────────────────────────

    def test_not_found_has_telemetry_key(self):
        router = ServiceRouter()
        result = router.execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self.assertIn("telemetry", result["diagnostics"])

    def test_not_found_telemetry_is_list(self):
        router = ServiceRouter()
        result = router.execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self.assertIsInstance(_get_telemetry(result), list)

    # ── open_session failure ─────────────────────────────────────────────

    def test_open_session_fail_has_telemetry(self):
        adapter = _MockAdapter()
        adapter.open_status = "error"
        result = self._run(adapter)
        self.assertIn("telemetry", result["diagnostics"])

    def test_open_session_fail_telemetry_nonempty(self):
        adapter = _MockAdapter()
        adapter.open_status = "error"
        result = self._run(adapter)
        self.assertGreater(len(_get_telemetry(result)), 0)

    # ── preflight failure ────────────────────────────────────────────────

    def test_preflight_fail_has_telemetry(self):
        adapter = _MockAdapter()
        adapter.pre_status = "error"
        result = self._run(adapter, policy=LifecyclePolicy(run_preflight=True))
        self.assertIn("telemetry", result["diagnostics"])

    # ── execute failure ──────────────────────────────────────────────────

    def test_execute_fail_has_telemetry(self):
        adapter = _MockAdapter()
        adapter.action_raises = RuntimeError("boom")
        result = self._run(adapter)
        self.assertIn("telemetry", result["diagnostics"])

    # ── close_session failure (result still ok) ──────────────────────────

    def test_close_fail_has_telemetry(self):
        adapter = _MockAdapter()
        adapter.close_status = "error"
        result = self._run(adapter, policy=LifecyclePolicy(close_after=True))
        self.assertIn("telemetry", result["diagnostics"])

    # ── close_session ok ─────────────────────────────────────────────────

    def test_close_ok_has_telemetry(self):
        result = self._run(policy=LifecyclePolicy(close_after=True))
        self.assertIn("telemetry", result["diagnostics"])


class TestRouterTelemetryContent(unittest.TestCase):
    """Telemetry event dicts contain the 13 schema keys and correct values."""

    _NAME = "myAdapter"

    def _run(self, adapter=None, op=RouteOp.RUN_ACTION, policy=None):
        if adapter is None:
            adapter = _MockAdapter()
        router = _router_with(adapter, self._NAME)
        return router.execute_with_lifecycle(self._NAME, op, policy=policy)

    def _events(self, result):
        return _get_telemetry(result)

    # ── Schema completeness ───────────────────────────────────────────────

    def test_every_event_has_all_13_keys(self):
        result = self._run()
        for ev in self._events(result):
            with self.subTest(ev=ev):
                self.assertEqual(frozenset(ev.keys()), _EVENT_KEYS)

    # ── open_session events ───────────────────────────────────────────────

    def test_open_session_start_event_present(self):
        result = self._run()
        stages = [(e["stage"], e["event"]) for e in self._events(result)]
        self.assertIn(("open_session", "start"), stages)

    def test_open_session_end_event_present(self):
        result = self._run()
        stages = [(e["stage"], e["event"]) for e in self._events(result)]
        self.assertIn(("open_session", "end"), stages)

    # ── execute events ────────────────────────────────────────────────────

    def test_execute_start_event_present(self):
        result = self._run()
        stages = [(e["stage"], e["event"]) for e in self._events(result)]
        self.assertIn(("execute", "start"), stages)

    def test_execute_end_status_ok_on_success(self):
        result = self._run()
        end_events = [e for e in self._events(result)
                      if e["stage"] == "execute" and e["event"] == "end"]
        self.assertTrue(end_events)
        self.assertEqual(end_events[0]["status"], "ok")

    def test_execute_end_status_error_on_failure(self):
        adapter = _MockAdapter()
        adapter.action_raises = RuntimeError("crash")
        result = self._run(adapter)
        end_events = [e for e in self._events(result)
                      if e["stage"] == "execute" and e["event"] == "end"]
        self.assertTrue(end_events)
        self.assertEqual(end_events[0]["status"], "error")

    def test_execute_end_reason_on_failure(self):
        adapter = _MockAdapter()
        adapter.action_raises = RuntimeError("crash")
        result = self._run(adapter)
        end_events = [e for e in self._events(result)
                      if e["stage"] == "execute" and e["event"] == "end"]
        self.assertEqual(end_events[0]["reason"], LifecycleReason.EXECUTE_FAILED)

    # ── close_session events ──────────────────────────────────────────────

    def test_close_session_events_present_when_close_after(self):
        result = self._run(policy=LifecyclePolicy(close_after=True))
        stages = [e["stage"] for e in self._events(result)]
        self.assertIn("close_session", stages)

    def test_no_close_session_events_without_close_after(self):
        result = self._run()
        stages = [e["stage"] for e in self._events(result)]
        self.assertNotIn("close_session", stages)

    # ── trace_id consistency ──────────────────────────────────────────────

    def test_all_events_share_same_trace_id(self):
        result = self._run(policy=LifecyclePolicy(close_after=True))
        trace_ids = {e["trace_id"] for e in self._events(result)}
        self.assertEqual(len(trace_ids), 1)

    def test_trace_id_is_valid_uuid(self):
        result = self._run()
        tid = self._events(result)[0]["trace_id"]
        parsed = uuid.UUID(tid, version=4)
        self.assertEqual(str(parsed), tid)

    def test_two_calls_have_different_trace_ids(self):
        adapter = _MockAdapter()
        router = _router_with(adapter, self._NAME)
        r1 = router.execute_with_lifecycle(self._NAME, RouteOp.RUN_ACTION)
        r2 = router.execute_with_lifecycle(self._NAME, RouteOp.RUN_ACTION)
        tid1 = _get_telemetry(r1)[0]["trace_id"]
        tid2 = _get_telemetry(r2)[0]["trace_id"]
        self.assertNotEqual(tid1, tid2)

    # ── adapter_name in events ────────────────────────────────────────────

    def test_events_adapter_name_matches(self):
        adapter = _MockAdapter()
        reg = ServiceRegistry()
        reg.register("spot_sdk", adapter)
        router = ServiceRouter(registry=reg)
        result = router.execute_with_lifecycle("spot_sdk", RouteOp.GET_SENSOR_DATA)
        for ev in _get_telemetry(result):
            self.assertEqual(ev["adapter_name"], "spot_sdk")

    # ── preflight events ──────────────────────────────────────────────────

    def test_preflight_events_present_when_run_preflight(self):
        result = self._run(policy=LifecyclePolicy(run_preflight=True))
        stages = [e["stage"] for e in self._events(result)]
        self.assertIn("preflight", stages)

    def test_no_preflight_events_without_run_preflight(self):
        result = self._run()
        stages = [e["stage"] for e in self._events(result)]
        self.assertNotIn("preflight", stages)

    # ── settings_validation events ────────────────────────────────────────

    def test_no_settings_validation_events_without_brand(self):
        result = self._run()
        stages = [e["stage"] for e in self._events(result)]
        self.assertNotIn("settings_validation", stages)

    # ── open_session error emits error end event ──────────────────────────

    def test_open_session_fail_emits_error_end_event(self):
        adapter = _MockAdapter()
        adapter.open_status = "error"
        result = self._run(adapter)
        end_events = [e for e in self._events(result)
                      if e["stage"] == "open_session" and e["event"] == "end"]
        self.assertTrue(end_events)
        self.assertEqual(end_events[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
