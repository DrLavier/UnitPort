"""Integration tests: Circle 3 package metadata end-to-end traceability.

Covers DoD items:
1. Package metadata persisted on save and restored on load.
2. Old files (no package_metadata key) still load successfully.
3. WorkflowIR.layered_contracts carries package metadata in serializable form.
4. Execution path exposes package_metadata_trace in diagnostics.
5. backward-compat: files without package_metadata load silently.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from bin.core.mission_persistence import (
    build_package_metadata_payload,
    extract_package_metadata_payload,
    migrate_mission_payload,
    needs_package_metadata_migration,
)
from system.ir.layered_contracts import LayeredIRBundle, PackageMetadata, SubgraphIR
from system.ir.layered_interfaces import DefaultLayeredIRBridge, DefaultLayeredIRSerializer
from system.runtime.contracts import DiagnosticsKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_mission(include_package_metadata: bool = True, pkg_id: str = "test.pkg") -> Dict[str, Any]:
    """Build a minimal, schema-valid mission dict."""
    data: Dict[str, Any] = {
        "nodes": [],
        "connections": [],
        "schema_version": "1.4",
    }
    if include_package_metadata:
        pm = PackageMetadata.create(pkg_id, "1.0.0")
        data["package_metadata"] = build_package_metadata_payload(pm)
    return data


# ---------------------------------------------------------------------------
# 1. Persistence helpers — save/load round-trip
# ---------------------------------------------------------------------------

class TestPackageMetadataPersistenceRoundTrip(unittest.TestCase):

    def test_build_then_extract_preserves_package_id(self):
        pm = PackageMetadata.create("persist.pkg", "2.0.0")
        payload = build_package_metadata_payload(pm)
        restored = extract_package_metadata_payload({"package_metadata": payload})
        self.assertIsNotNone(restored)
        self.assertEqual(restored["package_id"], "persist.pkg")
        self.assertEqual(restored["package_version"], "2.0.0")

    def test_build_then_extract_via_file(self):
        """Simulate a save → read from disk → restore cycle."""
        pm = PackageMetadata.create("file.pkg", "3.0.0")
        mission = _minimal_mission(pkg_id="file.pkg")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            json.dump(mission, f)
            fname = f.name

        try:
            with open(fname, encoding="utf-8") as f:
                loaded = json.load(f)
            restored = extract_package_metadata_payload(loaded)
            self.assertIsNotNone(restored)
            self.assertEqual(restored["package_id"], "file.pkg")
        finally:
            os.unlink(fname)

    def test_build_with_migration_info_survives_file(self):
        pm = PackageMetadata.create("migrated.pkg", "1.0.0")
        pm = pm.with_migration("test migration")
        payload = build_package_metadata_payload(pm)
        restored = extract_package_metadata_payload({"package_metadata": payload})
        mi = restored["migration_info"]
        self.assertEqual(mi["migration_reason"], "test migration")
        self.assertNotEqual(mi["migrated_at"], "")


# ---------------------------------------------------------------------------
# 2. Backward compatibility — old files without package_metadata
# ---------------------------------------------------------------------------

class TestBackwardCompatOldFiles(unittest.TestCase):

    def test_extract_returns_none_for_old_file(self):
        old_file = {"nodes": [], "connections": [], "schema_version": "1.0"}
        result = extract_package_metadata_payload(old_file)
        self.assertIsNone(result)

    def test_extract_does_not_raise_for_non_dict(self):
        result = extract_package_metadata_payload(None)
        self.assertIsNone(result)

    def test_migrate_returns_false_upgrade_when_no_subgraphs(self):
        old_file = {"nodes": [], "connections": [], "schema_version": "1.4"}
        result = migrate_mission_payload(old_file)
        self.assertFalse(result["needs_package_metadata_upgrade"])

    def test_migrate_returns_true_upgrade_when_subgraph_lacks_key(self):
        data = {
            "nodes": [], "connections": [], "schema_version": "1.4",
            "subgraphs": {"sg1": {"display_name": "Test"}},
        }
        result = migrate_mission_payload(data)
        self.assertTrue(result["needs_package_metadata_upgrade"])
        # Warning text must be present
        self.assertTrue(any("package_metadata" in w for w in result["warnings"]))

    def test_old_file_loads_gracefully_no_exception(self):
        old_file = {"nodes": [], "connections": []}
        # Should return None, not raise
        pm = extract_package_metadata_payload(old_file)
        self.assertIsNone(pm)
        # Migration check also silent
        info = migrate_mission_payload(old_file)
        self.assertFalse(info["needs_package_metadata_upgrade"])


# ---------------------------------------------------------------------------
# 3. WorkflowIR.layered_contracts carries package metadata
# ---------------------------------------------------------------------------

class TestWorkflowIRLayeredContracts(unittest.TestCase):

    def test_apply_bridge_sets_serializable_contracts(self):
        from compiler.ir.workflow_ir import WorkflowIR
        bridge = DefaultLayeredIRBridge()
        bundle = LayeredIRBundle()
        bundle.subgraphs = [SubgraphIR(
            subgraph_id="sg", display_name="SG",
            package_metadata=PackageMetadata.create("wf.pkg", "1.0.0"),
        )]
        wf = WorkflowIR(nodes=[], edges=[])
        bridge.apply_to_workflow_ir(wf, bundle)

        self.assertIsInstance(wf.layered_contracts, dict)
        sg_dict = wf.layered_contracts["subgraphs"][0]
        self.assertEqual(sg_dict["package_metadata"]["package_id"], "wf.pkg")

    def test_workflowir_to_dict_includes_layered_contracts(self):
        from compiler.ir.workflow_ir import WorkflowIR
        bridge = DefaultLayeredIRBridge()
        bundle = LayeredIRBundle()
        bundle.subgraphs = [SubgraphIR(
            subgraph_id="sg2", display_name="SG2",
            package_metadata=PackageMetadata.create("serial.pkg"),
        )]
        wf = WorkflowIR(nodes=[], edges=[])
        bridge.apply_to_workflow_ir(wf, bundle)
        d = wf.to_dict()
        self.assertIn("layered_contracts", d)
        self.assertEqual(
            d["layered_contracts"]["subgraphs"][0]["package_metadata"]["package_id"],
            "serial.pkg",
        )

    def test_workflowir_from_dict_restores_layered_contracts(self):
        from compiler.ir.workflow_ir import WorkflowIR
        bridge = DefaultLayeredIRBridge()
        bundle = LayeredIRBundle()
        bundle.subgraphs = [SubgraphIR(
            subgraph_id="sg3", display_name="SG3",
            package_metadata=PackageMetadata.create("restore.pkg", "2.5.0"),
        )]
        wf_orig = WorkflowIR(nodes=[], edges=[])
        bridge.apply_to_workflow_ir(wf_orig, bundle)
        d = wf_orig.to_dict()

        wf_restored = WorkflowIR.from_dict(d)
        self.assertIsNotNone(wf_restored.layered_contracts)
        pm = wf_restored.layered_contracts["subgraphs"][0]["package_metadata"]
        self.assertEqual(pm["package_id"], "restore.pkg")
        self.assertEqual(pm["package_version"], "2.5.0")


# ---------------------------------------------------------------------------
# 4. Execution path exposes package_metadata_trace in diagnostics
# ---------------------------------------------------------------------------

class TestExecutionPackageMetadataTrace(unittest.TestCase):
    """Test that RuntimeEngine surfaces package_metadata_trace in diagnostics."""

    def _make_engine(self):
        from system.runtime.runtime_engine import RuntimeEngine
        return RuntimeEngine()

    def _minimal_exec_graph(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": "start", "type": "Start", "params": {}},
                {"id": "end",   "type": "End",   "params": {}},
            ],
            "edges": [{"from": "start", "to": "end"}],
        }

    def _minimal_scenario(self, pkg_trace: Dict) -> Dict[str, Any]:
        return {
            "target": "simulation",
            "safety_policy": {},
            "package_metadata_trace": pkg_trace,
        }

    def test_package_metadata_trace_present_in_result_diagnostics(self):
        engine = self._make_engine()
        trace = {"package_id": "exec.pkg", "package_version": "1.0", "schema_version": "1.0"}
        scenario = self._minimal_scenario(trace)
        result = engine.execute(self._minimal_exec_graph(), scenario)
        pkg_trace = result.get("diagnostics", {}).get(DiagnosticsKey.PACKAGE_METADATA_TRACE)
        self.assertIsNotNone(pkg_trace)
        self.assertEqual(pkg_trace["package_id"], "exec.pkg")
        self.assertEqual(pkg_trace["package_version"], "1.0")

    def test_package_metadata_trace_empty_strings_when_absent(self):
        engine = self._make_engine()
        scenario = self._minimal_scenario({})
        result = engine.execute(self._minimal_exec_graph(), scenario)
        pkg_trace = result.get("diagnostics", {}).get(DiagnosticsKey.PACKAGE_METADATA_TRACE)
        self.assertIsNotNone(pkg_trace)
        self.assertEqual(pkg_trace["package_id"], "")
        self.assertEqual(pkg_trace["package_version"], "")
        self.assertEqual(pkg_trace["schema_version"], "")

    def test_package_metadata_trace_present_when_scenario_has_no_key(self):
        engine = self._make_engine()
        scenario: Dict[str, Any] = {"target": "simulation", "safety_policy": {}}
        result = engine.execute(self._minimal_exec_graph(), scenario)
        pkg_trace = result.get("diagnostics", {}).get(DiagnosticsKey.PACKAGE_METADATA_TRACE)
        self.assertIsNotNone(pkg_trace)
        self.assertIsInstance(pkg_trace["package_id"], str)

    def test_package_metadata_trace_all_keys_present(self):
        engine = self._make_engine()
        result = engine.execute(
            self._minimal_exec_graph(),
            {"target": "simulation", "safety_policy": {}},
        )
        pkg_trace = result.get("diagnostics", {}).get(DiagnosticsKey.PACKAGE_METADATA_TRACE, {})
        for key in ("package_id", "package_version", "schema_version"):
            self.assertIn(key, pkg_trace, f"Missing key: {key}")


# ---------------------------------------------------------------------------
# 5. Full save → load → execute traceability (metadata-only; no Qt)
# ---------------------------------------------------------------------------

class TestFullMetadataTraceability(unittest.TestCase):
    """Simulate the complete pipeline without Qt: build payload → persist →
    reload → inject into scenario → run → check diagnostics."""

    def test_save_load_execute_trace(self):
        # Step 1: create package metadata and "save" it into a mission dict
        pm = PackageMetadata.create("full.trace.pkg", "1.2.3")
        mission = {
            "nodes": [
                {"id": "start", "type": "Start", "params": {}},
                {"id": "end",   "type": "End",   "params": {}},
            ],
            "connections": [{"from": "start", "to": "end"}],
            "schema_version": "1.4",
            "package_metadata": build_package_metadata_payload(pm),
        }

        # Step 2: serialize to JSON and reload (simulates file I/O)
        raw = json.dumps(mission)
        loaded = json.loads(raw)

        # Step 3: extract metadata (simulates _on_open)
        restored_pkg = extract_package_metadata_payload(loaded)
        self.assertIsNotNone(restored_pkg)
        self.assertEqual(restored_pkg["package_id"], "full.trace.pkg")

        # Step 4: build exec_graph and inject package_metadata_trace (simulates _on_run)
        exec_graph = {
            "nodes": [
                {"id": "start", "type": "Start", "params": {}},
                {"id": "end",   "type": "End",   "params": {}},
            ],
            "edges": [{"from": "start", "to": "end"}],
        }
        scenario = {
            "target": "simulation",
            "safety_policy": {},
            "package_metadata_trace": {
                "package_id":      restored_pkg.get("package_id", ""),
                "package_version": restored_pkg.get("package_version", ""),
                "schema_version":  restored_pkg.get("schema_version", ""),
            },
        }

        # Step 5: execute and verify diagnostics carry the trace
        from system.runtime.runtime_engine import RuntimeEngine
        engine = RuntimeEngine()
        result = engine.execute(exec_graph, scenario)

        pkg_trace = result.get("diagnostics", {}).get(DiagnosticsKey.PACKAGE_METADATA_TRACE)
        self.assertIsNotNone(pkg_trace)
        self.assertEqual(pkg_trace["package_id"], "full.trace.pkg")
        self.assertEqual(pkg_trace["package_version"], "1.2.3")


if __name__ == "__main__":
    unittest.main()
