#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle 5 acceptance bootstrap: Simple/Advanced UX flow tests + error UX."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _simple_ir():
    from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind

    ir = WorkflowIR(name="circle5_simple")
    start = IRNode(id="s", schema_id="builtin.start", kind=NodeKind.START)
    end = IRNode(id="e", schema_id="builtin.end", kind=NodeKind.END)
    ir.add_node(start)
    ir.add_node(end)
    ir.add_edge(IREdge("s", "flow_out", "e", "flow_in"))
    return ir


def _behavior_stub_ir():
    from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind

    ir = WorkflowIR(name="circle5_behavior_stub")
    start = IRNode(id="bs", schema_id="builtin.start", kind=NodeKind.START)
    end = IRNode(id="be", schema_id="builtin.end", kind=NodeKind.END)
    ir.add_node(start)
    ir.add_node(end)
    ir.add_edge(IREdge("bs", "flow_out", "be", "flow_in"))
    return ir


def _advanced_ir_with_behavior():
    from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, IRParam, NodeKind

    ir = WorkflowIR(name="circle5_advanced")
    start = IRNode(id="s", schema_id="builtin.start", kind=NodeKind.START)
    behavior = IRNode(
        id="b1",
        schema_id="behavior_call",
        kind=NodeKind.BEHAVIOR_CALL,
        params={"behavior_ref": IRParam("behavior_ref", "ux_adv_behavior", "string")},
    )
    end = IRNode(id="e", schema_id="builtin.end", kind=NodeKind.END)
    ir.add_node(start)
    ir.add_node(behavior)
    ir.add_node(end)
    ir.add_edge(IREdge("s", "flow_out", "b1", "flow_in"))
    ir.add_edge(IREdge("b1", "flow_out", "e", "flow_in"))
    return ir


def _bridge_for_advanced_behavior():
    from system.behavior.behavior_artifact import (
        BehaviorArtifact,
        BehaviorResolveResult,
    )

    artifact = BehaviorArtifact.create(
        behavior_ref="ux_adv_behavior",
        behavior_ir=_behavior_stub_ir(),
        diagnostics=[],
    )

    class _Bridge:
        def resolve(self, behavior_ref, trace_id=None):
            if behavior_ref != "ux_adv_behavior":
                return BehaviorResolveResult.missing(behavior_ref, trace_id or "tid")
            return BehaviorResolveResult.success(artifact=artifact, trace_id=trace_id or "tid")

    return _Bridge()


class TestCircle5SimplePath(unittest.TestCase):
    def test_simple_path_has_trace_and_empty_package_metadata(self):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.runtime.contracts import DiagnosticsKey

        engine = RuntimeEngine()
        scenario = {
            "brand": "unitree",
            "robot_type": "go2",
            "_behavior_path_reason": "circle5_simple_mode",
        }
        result = engine.execute(_simple_ir(), scenario)
        diag = result.get("diagnostics", {})

        self.assertEqual(result.get("status"), "success")
        self.assertTrue(diag.get(DiagnosticsKey.BEHAVIOR_ENABLED_RUN))
        self.assertEqual(diag.get(DiagnosticsKey.EXECUTION_PATH_REASON), "circle5_simple_mode")
        self.assertTrue(diag.get(DiagnosticsKey.MISSION_TRACE_ID))
        self.assertEqual(diag.get(DiagnosticsKey.PACKAGE_METADATA_TRACE), {
            "package_id": "",
            "package_version": "",
            "schema_version": "",
        })


class TestCircle5AdvancedPath(unittest.TestCase):
    def test_advanced_path_has_package_trace_and_behavior_trace(self):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.runtime.contracts import DiagnosticsKey

        engine = RuntimeEngine(behavior_bridge=_bridge_for_advanced_behavior())
        scenario = {
            "brand": "unitree",
            "robot_type": "go2",
            "_behavior_path_reason": "circle5_advanced_package_expanded",
            "package_metadata_trace": {
                "package_id": "pkg.advanced.walk",
                "package_version": "2.4.1",
                "schema_version": "3",
            },
        }
        result = engine.execute(_advanced_ir_with_behavior(), scenario)
        diag = result.get("diagnostics", {})

        self.assertIn(result.get("status"), ("success", "failed"))
        self.assertEqual(diag.get(DiagnosticsKey.EXECUTION_PATH_REASON), "circle5_advanced_package_expanded")
        self.assertEqual(diag.get(DiagnosticsKey.PACKAGE_METADATA_TRACE), scenario["package_metadata_trace"])
        self.assertTrue(diag.get(DiagnosticsKey.MISSION_TRACE_ID))
        btids = diag.get(DiagnosticsKey.BEHAVIOR_TRACE_IDS, {})
        self.assertIn("b1", btids)
        self.assertEqual(btids["b1"], diag.get(DiagnosticsKey.MISSION_TRACE_ID))


class TestCircle5ErrorMessageConsistency(unittest.TestCase):
    def test_failed_node_has_machine_code_and_human_message_and_trace(self):
        from bin.core.error_ux import extract_failed_nodes_info, get_operator_text
        from nodes import REGISTERED_NODES
        from nodes.sys_nodes.base_node import BaseNode
        from system.runtime.runtime_engine import RuntimeEngine
        from system.runtime.contracts import DiagnosticsKey
        from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind

        class _FailNode(BaseNode):
            def __init__(self, node_id: str):
                super().__init__(node_id, "ux_fail_node")

            def execute(self, inputs):
                return {
                    "error": "execute_failed",
                    "reason": "execute_failed",
                    "message": "Action execution failed.",
                    "trace_id": "node-trace-fixed",
                }

            def get_display_name(self) -> str:
                return "UX Fail Node"

            def get_description(self) -> str:
                return "Deterministic failure node for Circle 5 UX acceptance."

        original = REGISTERED_NODES.get("ux_fail_node")
        REGISTERED_NODES["ux_fail_node"] = _FailNode
        try:
            ir = WorkflowIR(name="circle5_error_consistency")
            s = IRNode(id="s", schema_id="builtin.start", kind=NodeKind.START)
            f = IRNode(id="f", schema_id="ux_fail_node", kind=NodeKind.CUSTOM)
            e = IRNode(id="e", schema_id="builtin.end", kind=NodeKind.END)
            ir.add_node(s)
            ir.add_node(f)
            ir.add_node(e)
            ir.add_edge(IREdge("s", "flow_out", "f", "flow_in"))
            ir.add_edge(IREdge("f", "flow_out", "e", "flow_in"))

            result = RuntimeEngine().execute(ir, {"brand": "unitree", "robot_type": "go2"})
            self.assertEqual(result.get("status"), "failed")

            infos = extract_failed_nodes_info(result, {"f": "UX Fail Node"})
            self.assertEqual(len(infos), 1)
            info = infos[0]

            self.assertEqual(info.get("reason"), "execute_failed")  # machine-readable code
            self.assertEqual(info.get("operator_text"), get_operator_text("execute_failed"))  # human message
            self.assertTrue(info.get("message"))
            self.assertEqual(info.get("trace_id"), "node-trace-fixed")
            self.assertEqual(
                info.get("mission_trace_id"),
                result.get("diagnostics", {}).get(DiagnosticsKey.MISSION_TRACE_ID),
            )
        finally:
            if original is None:
                REGISTERED_NODES.pop("ux_fail_node", None)
            else:
                REGISTERED_NODES["ux_fail_node"] = original


if __name__ == "__main__":
    unittest.main()

