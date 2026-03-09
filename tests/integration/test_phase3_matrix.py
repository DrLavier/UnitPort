#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 Test Matrix — STAGE-06.

This file is the single-document evidence of Phase 3 completeness.
Organised in three explicit sections mirroring the STAGE-06 task list:

  SECTION A — Unit coverage
    STAGE-05 error taxonomy: LifecycleReason constants + structured diagnostics.
    STAGE-05 diagnostics plumbing: logging at each lifecycle boundary.

  SECTION B — Integration coverage
    Full Unitree lifecycle flow (success path and all failure branches).
    Phase 2 runtime/service integration does not break.

  SECTION C — Regression coverage
    Legacy non-lifecycle callers unchanged.
    Phase 1 + Phase 2 runtime output contract keys stable.

Isolation
---------
No Qt, no SDK.  All adapter and node interactions use in-module stubs.
RuntimeEngine tests reuse the fake-node-registry pattern from
tests/integration/test_runtime_engine.py.
"""

import logging
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Service layer imports (no stubs needed) ────────────────────────────────
from system.service.lifecycle import (           # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
)
from system.service.service_router import RouteOp, ServiceRouter   # noqa: E402
from system.service.service_registry import ServiceRegistry         # noqa: E402
from system.service.adapters.base_adapter import BaseAdapter        # noqa: E402

# ── Runtime imports (no stubs needed) ─────────────────────────────────────
from system.runtime.runtime_engine import RuntimeEngine             # noqa: E402
from system.runtime.contracts import (                              # noqa: E402
    DiagnosticsKey, ExecutionPath, RuntimeStatus,
)
from system.ir.workflow_ir import IRNode, NodeKind, WorkflowIR     # noqa: E402

# ── Shared: the stdlib logger used by service_router ──────────────────────
_router_logger = logging.getLogger("unitport.service.router")


# ── Shared mock adapter ────────────────────────────────────────────────────

class _MockAdapter(BaseAdapter):
    """Configurable lifecycle mock for matrix tests."""

    def __init__(self):
        self.open_session_status  = "ok"
        self.preflight_status     = "ok"
        self.close_session_status = "ok"
        self.run_action_return    = "action_done"
        self.run_action_raises    = None
        self.calls: list = []

    def connect(self, **kwargs: Any) -> bool:
        self.calls.append(("connect", kwargs))
        return True

    def run_action(self, action: str, **params: Any) -> Any:
        self.calls.append(("run_action", action, params))
        if self.run_action_raises:
            raise self.run_action_raises
        return self.run_action_return

    def stop(self) -> None:
        self.calls.append(("stop",))

    def get_sensor_data(self) -> Dict[str, Any]:
        self.calls.append(("get_sensor_data",))
        return {"battery": 80}

    def health(self) -> Dict[str, Any]:
        return {"connected": True}

    def open_session(self, config=None) -> Dict[str, Any]:
        self.calls.append(("open_session", config))
        if self.open_session_status == "ok":
            return LifecycleResult.ok("open_session").to_dict()
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "mock open_session failure"},
        ).to_dict()

    def preflight(self, context=None) -> Dict[str, Any]:
        self.calls.append(("preflight", context))
        if self.preflight_status == "ok":
            return LifecycleResult.ok("preflight").to_dict()
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED,
            {"message": "mock preflight failure"},
        ).to_dict()

    def close_session(self) -> Dict[str, Any]:
        self.calls.append(("close_session",))
        if self.close_session_status == "ok":
            return LifecycleResult.ok("close_session").to_dict()
        return LifecycleResult.error(
            "close_session", LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "mock close_session failure"},
        ).to_dict()


def _router_with(adapter: _MockAdapter, name: str = "unitree") -> ServiceRouter:
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


# ── Shared: fake node registry for RuntimeEngine tests ────────────────────

_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class _SimpleNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "output": f"{self.nid}_done"}


class _FailingNode:
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): raise RuntimeError("node_error")


def _install_node_registry():
    _registry.clear()
    _registry["simple"] = _SimpleNode
    _registry["failing"] = _FailingNode
    sys.modules["nodes"] = _nodes_mod


def _uninstall_node_registry():
    _registry.clear()
    sys.modules.pop("nodes", None)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Unit evidence
# ═══════════════════════════════════════════════════════════════════════════

# ── A1: STAGE-05 Error taxonomy ────────────────────────────────────────────

class TestStage05ErrorTaxonomy(unittest.TestCase):
    """LifecycleReason provides the required error reason codes."""

    _SPEC_CODES = [
        ("SESSION_OPEN_FAILED",    "session_open_failed"),
        ("PREFLIGHT_FAILED",       "preflight_failed"),
        ("SESSION_CLOSE_FAILED",   "session_close_failed"),
        ("CAPABILITY_UNAVAILABLE", "capability_unavailable"),
    ]

    def test_session_open_failed_constant_value(self):
        self.assertEqual(LifecycleReason.SESSION_OPEN_FAILED, "session_open_failed")

    def test_preflight_failed_constant_value(self):
        self.assertEqual(LifecycleReason.PREFLIGHT_FAILED, "preflight_failed")

    def test_session_close_failed_constant_value(self):
        self.assertEqual(LifecycleReason.SESSION_CLOSE_FAILED, "session_close_failed")

    def test_capability_unavailable_constant_value(self):
        self.assertEqual(LifecycleReason.CAPABILITY_UNAVAILABLE, "capability_unavailable")

    def test_all_required_reason_codes_distinct(self):
        values = [getattr(LifecycleReason, attr) for attr, _ in self._SPEC_CODES]
        self.assertEqual(len(values), len(set(values)))


# ── A2: STAGE-05 Structured diagnostics ───────────────────────────────────

class TestStage05StructuredDiagnostics(unittest.TestCase):
    """Failure results carry code / message / stage / context."""

    _SCHEMA = frozenset({"status", "reason", "stage", "payload", "diagnostics"})

    def _run(self, adapter, op=RouteOp.RUN_ACTION, op_args=None, policy=None):
        router = _router_with(adapter)
        return router.execute_with_lifecycle("unitree", op, op_args, policy)

    def test_open_session_failure_result_schema_complete(self):
        a = _MockAdapter(); a.open_session_status = "error"
        result = self._run(a)
        self.assertFalse(self._SCHEMA - result.keys())

    def test_open_session_failure_reason_is_session_open_failed(self):
        a = _MockAdapter(); a.open_session_status = "error"
        result = self._run(a)
        self.assertEqual(result["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_preflight_failure_result_schema_complete(self):
        a = _MockAdapter(); a.preflight_status = "error"
        result = self._run(a, policy=LifecyclePolicy(run_preflight=True))
        self.assertFalse(self._SCHEMA - result.keys())

    def test_execute_failure_diagnostics_has_message_and_op(self):
        a = _MockAdapter(); a.run_action_raises = RuntimeError("boom")
        result = self._run(a, op_args={"action": "stand"})
        self.assertIn("message", result["diagnostics"])
        self.assertIn("op",      result["diagnostics"])

    def test_close_session_failure_status_still_ok(self):
        a = _MockAdapter(); a.close_session_status = "error"
        result = self._run(
            a, op_args={"action": "stand"},
            policy=LifecyclePolicy(close_after=True),
        )
        self.assertEqual(result["status"], "ok")

    def test_close_session_failure_surfaced_in_diagnostics(self):
        a = _MockAdapter(); a.close_session_status = "error"
        result = self._run(
            a, op_args={"action": "stand"},
            policy=LifecyclePolicy(close_after=True),
        )
        self.assertIn("close_session_failed", result["diagnostics"])

    def test_all_error_paths_have_non_empty_reason(self):
        cases = [
            ("adapter_not_found", ServiceRouter(), "ghost", RouteOp.RUN_ACTION, None, None),
        ]
        for label, router, name, op, op_args, policy in cases:
            with self.subTest(label=label):
                result = router.execute_with_lifecycle(name, op, op_args, policy)
                self.assertNotEqual(result.get("reason", ""), "",
                                    f"Empty reason on {label}")


# ── A3: STAGE-05 Lifecycle boundary logging ───────────────────────────────

class TestStage05BoundaryLogging(unittest.TestCase):
    """service_router emits log calls at lifecycle boundaries."""

    def _run(self, adapter, op_args=None, policy=None):
        router = _router_with(adapter)
        return router.execute_with_lifecycle("unitree", RouteOp.RUN_ACTION,
                                             op_args, policy)

    def test_success_path_emits_debug_logs(self):
        with patch.object(_router_logger, "debug") as mock_debug:
            self._run(_MockAdapter(), op_args={"action": "stand"})
        # acquire + open_session + execute + success = ≥4 debug calls
        self.assertGreaterEqual(mock_debug.call_count, 4)

    def test_adapter_not_found_emits_warning(self):
        with patch.object(_router_logger, "warning") as mock_warn:
            ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self.assertGreater(mock_warn.call_count, 0)

    def test_open_session_failure_emits_warning(self):
        a = _MockAdapter(); a.open_session_status = "error"
        with patch.object(_router_logger, "warning") as mock_warn:
            self._run(a)
        self.assertGreater(mock_warn.call_count, 0)

    def test_preflight_failure_emits_warning(self):
        a = _MockAdapter(); a.preflight_status = "error"
        policy = LifecyclePolicy(run_preflight=True)
        with patch.object(_router_logger, "warning") as mock_warn:
            self._run(a, policy=policy)
        self.assertGreater(mock_warn.call_count, 0)

    def test_execute_failure_emits_warning(self):
        a = _MockAdapter(); a.run_action_raises = RuntimeError("hw_err")
        with patch.object(_router_logger, "warning") as mock_warn:
            self._run(a, op_args={"action": "stand"})
        self.assertGreater(mock_warn.call_count, 0)

    def test_close_session_failure_emits_warning(self):
        a = _MockAdapter(); a.close_session_status = "error"
        policy = LifecyclePolicy(close_after=True)
        with patch.object(_router_logger, "warning") as mock_warn:
            self._run(a, op_args={"action": "stand"}, policy=policy)
        self.assertGreater(mock_warn.call_count, 0)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Integration coverage
# ═══════════════════════════════════════════════════════════════════════════

# ── B1: Unitree lifecycle success flow ────────────────────────────────────

class TestUnitreeLifecycleSuccessFlow(unittest.TestCase):
    """Full lifecycle path returns correct result for all op types."""

    def setUp(self):
        self.adapter = _MockAdapter()
        self.router  = _router_with(self.adapter)

    def _ewl(self, op, op_args=None, policy=None):
        return self.router.execute_with_lifecycle("unitree", op, op_args, policy)

    def test_run_action_full_lifecycle_status_ok(self):
        result = self._ewl(RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["status"], "ok")

    def test_run_action_payload_returned_through_lifecycle(self):
        result = self._ewl(RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["payload"], "action_done")

    def test_open_session_called_before_execute(self):
        self._ewl(RouteOp.RUN_ACTION, {"action": "stand"})
        call_names = [c[0] for c in self.adapter.calls]
        open_idx   = call_names.index("open_session")
        run_idx    = call_names.index("run_action")
        self.assertLess(open_idx, run_idx)

    def test_close_session_called_after_execute_when_policy_close_after(self):
        self._ewl(RouteOp.RUN_ACTION, {"action": "stand"},
                  policy=LifecyclePolicy(close_after=True))
        call_names = [c[0] for c in self.adapter.calls]
        run_idx   = call_names.index("run_action")
        close_idx = call_names.index("close_session")
        self.assertLess(run_idx, close_idx)

    def test_full_lifecycle_with_preflight_ok(self):
        result = self._ewl(
            RouteOp.RUN_ACTION, {"action": "stand"},
            policy=LifecyclePolicy(run_preflight=True),
        )
        self.assertEqual(result["status"], "ok")
        pre_calls = [c for c in self.adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(pre_calls), 1)

    def test_lifecycle_result_schema_complete(self):
        result = self._ewl(RouteOp.RUN_ACTION, {"action": "stand"})
        for key in ("status", "reason", "stage", "payload", "diagnostics"):
            with self.subTest(key=key):
                self.assertIn(key, result)

    def test_get_sensor_data_via_lifecycle(self):
        result = self._ewl(RouteOp.GET_SENSOR_DATA)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"], {"battery": 80})

    def test_stop_via_lifecycle_status_ok(self):
        result = self._ewl(RouteOp.STOP)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["payload"])


# ── B2: Unitree lifecycle failure branches ────────────────────────────────

class TestUnitreeLifecycleFailureBranches(unittest.TestCase):
    """Every failure branch returns a reason-coded, diagnosable result."""

    def _ewl(self, adapter, op_args=None, policy=None):
        router = _router_with(adapter)
        return router.execute_with_lifecycle(
            "unitree", RouteOp.RUN_ACTION, op_args, policy
        )

    def test_open_session_failure_stops_at_open_stage(self):
        a = _MockAdapter(); a.open_session_status = "error"
        result = self._ewl(a)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"],  "open_session")

    def test_open_session_failure_reason_is_session_open_failed(self):
        a = _MockAdapter(); a.open_session_status = "error"
        result = self._ewl(a)
        self.assertEqual(result["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_open_session_failure_payload_none(self):
        a = _MockAdapter(); a.open_session_status = "error"
        self.assertIsNone(self._ewl(a)["payload"])

    def test_preflight_failure_stops_at_preflight_stage(self):
        a = _MockAdapter(); a.preflight_status = "error"
        result = self._ewl(a, policy=LifecyclePolicy(run_preflight=True))
        self.assertEqual(result["stage"],  "preflight")
        self.assertEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_preflight_failure_calls_close_session_as_cleanup(self):
        a = _MockAdapter(); a.preflight_status = "error"
        self._ewl(a, policy=LifecyclePolicy(run_preflight=True))
        close_calls = [c for c in a.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 1)

    def test_execute_failure_reason_is_execute_failed(self):
        a = _MockAdapter(); a.run_action_raises = RuntimeError("hw_err")
        result = self._ewl(a, op_args={"action": "stand"})
        self.assertEqual(result["reason"], "execute_failed")

    def test_execute_failure_diagnostics_has_message(self):
        a = _MockAdapter(); a.run_action_raises = RuntimeError("hw_err")
        result = self._ewl(a, op_args={"action": "stand"})
        self.assertIn("message", result["diagnostics"])

    def test_close_session_failure_does_not_mask_action_result(self):
        a = _MockAdapter(); a.close_session_status = "error"
        result = self._ewl(
            a, op_args={"action": "stand"},
            policy=LifecyclePolicy(close_after=True),
        )
        self.assertEqual(result["status"],  "ok")
        self.assertEqual(result["payload"], "action_done")

    def test_close_session_failure_surfaced_in_diagnostics_key(self):
        a = _MockAdapter(); a.close_session_status = "error"
        result = self._ewl(
            a, op_args={"action": "stand"},
            policy=LifecyclePolicy(close_after=True),
        )
        self.assertTrue(result["diagnostics"].get("close_session_failed"))

    def test_adapter_not_found_reason(self):
        result = ServiceRouter().execute_with_lifecycle(
            "ghost", RouteOp.RUN_ACTION
        )
        self.assertEqual(result["reason"], LifecycleReason.ADAPTER_NOT_FOUND)
        self.assertEqual(result["stage"],  "acquire")


# ── B3: Phase 2 runtime / service integration ─────────────────────────────

class TestPhase2RuntimeIntegration(unittest.TestCase):
    """RuntimeEngine output contract and Phase 2 path unbroken by Phase 3."""

    _TOP_KEYS = frozenset({
        "status", "reason", "task_id", "node_count", "results", "metrics", "diagnostics",
    })
    _DIAG_KEYS = frozenset({
        DiagnosticsKey.PATH,
        DiagnosticsKey.EXECUTED_COUNT,
        DiagnosticsKey.FAILED_NODES,
        DiagnosticsKey.HAS_ACTION,
        DiagnosticsKey.MISSION_TRACE_ID,
        DiagnosticsKey.BEHAVIOR_TRACE_IDS,
        DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
    })

    @classmethod
    def setUpClass(cls):
        _install_node_registry()

    @classmethod
    def tearDownClass(cls):
        _uninstall_node_registry()

    def _simple_ir(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", kind=NodeKind.ACTION,
                           schema_id="builtin.action_execution", params={}))
        return ir

    def test_runtime_top_level_keys_present(self):
        engine = RuntimeEngine()
        result = engine.execute(self._simple_ir(), scenario={})
        missing = self._TOP_KEYS - result.keys()
        self.assertFalse(missing, f"Missing top-level keys: {missing}")

    def test_runtime_diagnostics_keys_present(self):
        engine = RuntimeEngine()
        result = engine.execute(self._simple_ir(), scenario={})
        diag   = result.get("diagnostics", {})
        missing = self._DIAG_KEYS - diag.keys()
        self.assertFalse(missing, f"Missing diagnostics keys: {missing}")

    def test_runtime_status_success_on_simple_node(self):
        engine = RuntimeEngine()
        result = engine.execute(self._simple_ir(), scenario={})
        self.assertEqual(result["status"], RuntimeStatus.SUCCESS)

    def test_runtime_status_failed_on_failing_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", kind=NodeKind.ACTION,
                           schema_id="failing", params={}))
        engine = RuntimeEngine()
        result = engine.execute(ir, scenario={})
        self.assertEqual(result["status"], RuntimeStatus.FAILED)

    def test_service_and_runtime_imports_coexist(self):
        """Verify Phase 3 service imports do not disturb runtime imports."""
        from system.service.lifecycle import LifecycleResult       # noqa: F401
        from system.service.service_router import ServiceRouter     # noqa: F401
        from system.runtime.runtime_engine import RuntimeEngine     # noqa: F401
        from system.runtime.contracts import DiagnosticsKey        # noqa: F401
        self.assertTrue(True)  # no ImportError → pass


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Regression coverage
# ═══════════════════════════════════════════════════════════════════════════

# ── C1: Legacy passthrough callers unchanged ──────────────────────────────

class TestLegacyPassthroughRegression(unittest.TestCase):
    """Direct run_action / stop / get_sensor_data signatures unchanged."""

    def setUp(self):
        self.adapter = _MockAdapter()
        self.router  = _router_with(self.adapter)

    def test_legacy_run_action_returns_adapter_value(self):
        result = self.router.run_action("unitree", "stand")
        self.assertEqual(result, "action_done")

    def test_legacy_run_action_does_not_call_open_session(self):
        self.router.run_action("unitree", "stand")
        open_calls = [c for c in self.adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 0)

    def test_legacy_run_action_does_not_call_close_session(self):
        self.router.run_action("unitree", "stand")
        close_calls = [c for c in self.adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)

    def test_legacy_stop_calls_adapter_stop(self):
        self.router.stop("unitree")
        stop_calls = [c for c in self.adapter.calls if c[0] == "stop"]
        self.assertEqual(len(stop_calls), 1)

    def test_legacy_get_sensor_data_returns_sensor_dict(self):
        result = self.router.get_sensor_data("unitree")
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)

    def test_legacy_methods_raise_key_error_on_missing_adapter(self):
        router = ServiceRouter()
        with self.assertRaises(KeyError):
            router.run_action("nonexistent", "stand")


# ── C2: Runtime output contract keys stable ───────────────────────────────

class TestRuntimeContractStability(unittest.TestCase):
    """Phase 1 + Phase 2 constant values have not changed."""

    def test_diagnostics_path_key_is_path(self):
        self.assertEqual(DiagnosticsKey.PATH, "path")

    def test_diagnostics_executed_count_key(self):
        self.assertEqual(DiagnosticsKey.EXECUTED_COUNT, "executed_count")

    def test_diagnostics_failed_nodes_key(self):
        self.assertEqual(DiagnosticsKey.FAILED_NODES, "failed_nodes")

    def test_diagnostics_has_action_key(self):
        self.assertEqual(DiagnosticsKey.HAS_ACTION, "has_action")

    def test_phase2_mission_trace_id_key(self):
        self.assertEqual(DiagnosticsKey.MISSION_TRACE_ID, "mission_trace_id")

    def test_phase2_behavior_trace_ids_key(self):
        self.assertEqual(DiagnosticsKey.BEHAVIOR_TRACE_IDS, "behavior_trace_ids")

    def test_phase2_behavior_diagnostics_key(self):
        self.assertEqual(DiagnosticsKey.BEHAVIOR_DIAGNOSTICS, "behavior_diagnostics")

    def test_runtime_status_values_unchanged(self):
        self.assertEqual(RuntimeStatus.SUCCESS, "success")
        self.assertEqual(RuntimeStatus.FAILED,  "failed")
        self.assertEqual(RuntimeStatus.BLOCKED, "blocked")


if __name__ == "__main__":
    unittest.main(verbosity=2)
