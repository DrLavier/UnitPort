"""Unit tests for Circle 3 DefaultLayeredIRBridge and DefaultLayeredIRSerializer.

Covers:
- Serializer round-trip (to_dict / from_dict) with SubgraphIR + PackageMetadata
- Bridge from_workflow_ir: reads layered_contracts dict → LayeredIRBundle
- Bridge apply_to_workflow_ir: writes serialized dict → WorkflowIR.layered_contracts
- Bridge from_workflow_ir: missing/None layered_contracts → empty bundle (backward-compat)
- PackageMetadata survives bridge round-trip
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from system.ir.layered_contracts import (
    LayeredIRBundle,
    MigrationInfo,
    PackageMetadata,
    SubgraphIR,
    SubgraphPort,
)
from system.ir.layered_interfaces import (
    DefaultLayeredIRBridge,
    DefaultLayeredIRSerializer,
)


# ---------------------------------------------------------------------------
# Minimal WorkflowIR stub for bridge tests
# ---------------------------------------------------------------------------

@dataclass
class _StubWorkflowIR:
    """Minimal stand-in for WorkflowIR — has layered_contracts field."""
    nodes: list = field(default_factory=list)
    layered_contracts: Optional[Dict[str, Any]] = None


class _DictWorkflowIR(dict):
    """Dict-style workflow IR (legacy compat path)."""


# ---------------------------------------------------------------------------
# Serializer tests
# ---------------------------------------------------------------------------

class TestDefaultLayeredIRSerializer(unittest.TestCase):

    def setUp(self):
        self.ser = DefaultLayeredIRSerializer()

    def _make_bundle(self, pkg_id: str = "pkg.test") -> LayeredIRBundle:
        b = LayeredIRBundle()
        sg = SubgraphIR(
            subgraph_id="sg1",
            display_name="Test SG",
            inputs=[SubgraphPort(name="in", direction="input", data_type="float")],
            outputs=[SubgraphPort(name="out", direction="output", data_type="float")],
            internal_node_ids=["n1", "n2"],
            package_metadata=PackageMetadata.create(pkg_id, "1.2.3"),
        )
        b.subgraphs = [sg]
        return b

    def test_to_dict_has_subgraphs_key(self):
        d = self.ser.to_dict(self._make_bundle())
        self.assertIn("subgraphs", d)

    def test_to_dict_subgraph_has_package_metadata(self):
        d = self.ser.to_dict(self._make_bundle("my.pkg"))
        self.assertEqual(d["subgraphs"][0]["package_metadata"]["package_id"], "my.pkg")

    def test_to_dict_is_json_safe(self):
        import json
        d = self.ser.to_dict(self._make_bundle())
        # Must not raise
        json.dumps(d)

    def test_from_dict_empty_payload_returns_empty_bundle(self):
        b = self.ser.from_dict({})
        self.assertIsInstance(b, LayeredIRBundle)
        self.assertEqual(b.subgraphs, [])

    def test_from_dict_none_returns_empty_bundle(self):
        b = self.ser.from_dict(None)  # type: ignore[arg-type]
        self.assertIsInstance(b, LayeredIRBundle)

    def test_from_dict_reconstructs_subgraph(self):
        d = self.ser.to_dict(self._make_bundle("round.trip"))
        restored = self.ser.from_dict(d)
        self.assertEqual(len(restored.subgraphs), 1)
        self.assertEqual(restored.subgraphs[0].subgraph_id, "sg1")

    def test_from_dict_reconstructs_package_metadata(self):
        d = self.ser.to_dict(self._make_bundle("rt.pkg"))
        restored = self.ser.from_dict(d)
        pm = restored.subgraphs[0].package_metadata
        self.assertEqual(pm.package_id, "rt.pkg")
        self.assertEqual(pm.package_version, "1.2.3")

    def test_roundtrip_preserves_ports(self):
        d = self.ser.to_dict(self._make_bundle())
        restored = self.ser.from_dict(d)
        sg = restored.subgraphs[0]
        self.assertEqual(len(sg.inputs), 1)
        self.assertEqual(sg.inputs[0].name, "in")
        self.assertEqual(sg.outputs[0].name, "out")

    def test_roundtrip_preserves_internal_node_ids(self):
        d = self.ser.to_dict(self._make_bundle())
        restored = self.ser.from_dict(d)
        self.assertEqual(sorted(restored.subgraphs[0].internal_node_ids), ["n1", "n2"])

    def test_from_dict_with_migration_info(self):
        b = LayeredIRBundle()
        pm = PackageMetadata.create("migrated.pkg")
        pm = pm.with_migration("upgrade test")
        b.subgraphs = [SubgraphIR(subgraph_id="s", display_name="S", package_metadata=pm)]
        d = self.ser.to_dict(b)
        restored = self.ser.from_dict(d)
        mi = restored.subgraphs[0].package_metadata.migration_info
        self.assertEqual(mi.migration_reason, "upgrade test")
        self.assertNotEqual(mi.migrated_at, "")


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------

class TestDefaultLayeredIRBridgeApply(unittest.TestCase):

    def setUp(self):
        self.bridge = DefaultLayeredIRBridge()

    def test_apply_sets_layered_contracts_on_dataclass(self):
        wf = _StubWorkflowIR()
        bundle = LayeredIRBundle()
        self.bridge.apply_to_workflow_ir(wf, bundle)
        self.assertIsInstance(wf.layered_contracts, dict)

    def test_apply_subgraph_package_metadata_in_contracts(self):
        wf = _StubWorkflowIR()
        bundle = LayeredIRBundle()
        bundle.subgraphs = [SubgraphIR(
            subgraph_id="sg1", display_name="SG",
            package_metadata=PackageMetadata.create("wired.pkg", "2.0.0"),
        )]
        self.bridge.apply_to_workflow_ir(wf, bundle)
        pm = wf.layered_contracts["subgraphs"][0]["package_metadata"]
        self.assertEqual(pm["package_id"], "wired.pkg")
        self.assertEqual(pm["package_version"], "2.0.0")

    def test_apply_dict_style_workflow_ir(self):
        wf = _DictWorkflowIR()
        self.bridge.apply_to_workflow_ir(wf, LayeredIRBundle())
        self.assertIn("layered_contracts", wf)

    def test_apply_returns_workflow_ir(self):
        wf = _StubWorkflowIR()
        result = self.bridge.apply_to_workflow_ir(wf, LayeredIRBundle())
        self.assertIs(result, wf)

    def test_apply_does_not_touch_other_fields(self):
        wf = _StubWorkflowIR(nodes=["existing_node"])
        self.bridge.apply_to_workflow_ir(wf, LayeredIRBundle())
        self.assertEqual(wf.nodes, ["existing_node"])


class TestDefaultLayeredIRBridgeFrom(unittest.TestCase):

    def setUp(self):
        self.bridge = DefaultLayeredIRBridge()
        self.ser = DefaultLayeredIRSerializer()

    def test_from_workflow_ir_with_no_contracts_returns_empty_bundle(self):
        wf = _StubWorkflowIR()
        bundle = self.bridge.from_workflow_ir(wf)
        self.assertIsInstance(bundle, LayeredIRBundle)
        self.assertEqual(bundle.subgraphs, [])

    def test_from_workflow_ir_with_none_contracts_returns_empty(self):
        wf = _StubWorkflowIR(layered_contracts=None)
        bundle = self.bridge.from_workflow_ir(wf)
        self.assertEqual(bundle.subgraphs, [])

    def test_from_workflow_ir_reconstructs_subgraph(self):
        # Simulate a previously-applied bundle
        wf = _StubWorkflowIR()
        bundle_in = LayeredIRBundle()
        bundle_in.subgraphs = [SubgraphIR(
            subgraph_id="sg_restore", display_name="Restored",
            package_metadata=PackageMetadata.create("restore.pkg"),
        )]
        self.bridge.apply_to_workflow_ir(wf, bundle_in)
        bundle_out = self.bridge.from_workflow_ir(wf)
        self.assertEqual(len(bundle_out.subgraphs), 1)
        self.assertEqual(bundle_out.subgraphs[0].package_metadata.package_id, "restore.pkg")

    def test_from_workflow_ir_dict_style(self):
        wf = _DictWorkflowIR()
        bundle_in = LayeredIRBundle()
        bundle_in.subgraphs = [SubgraphIR(
            subgraph_id="x", display_name="X",
            package_metadata=PackageMetadata.create("dict.pkg"),
        )]
        self.bridge.apply_to_workflow_ir(wf, bundle_in)
        bundle_out = self.bridge.from_workflow_ir(wf)
        self.assertEqual(bundle_out.subgraphs[0].package_metadata.package_id, "dict.pkg")

    def test_from_workflow_ir_with_non_dict_contracts_returns_empty(self):
        wf = _StubWorkflowIR()
        wf.layered_contracts = "bad_value"  # type: ignore[assignment]
        bundle = self.bridge.from_workflow_ir(wf)
        self.assertEqual(bundle.subgraphs, [])


# ---------------------------------------------------------------------------
# Full apply → from round-trip
# ---------------------------------------------------------------------------

class TestBridgeRoundTrip(unittest.TestCase):

    def test_apply_then_from_preserves_package_metadata(self):
        bridge = DefaultLayeredIRBridge()
        wf = _StubWorkflowIR()
        pm_in = PackageMetadata.create("roundtrip.pkg", "3.1.4", "2.0")
        bundle_in = LayeredIRBundle()
        bundle_in.subgraphs = [SubgraphIR(
            subgraph_id="sg", display_name="SG",
            inputs=[SubgraphPort(name="a", direction="input")],
            package_metadata=pm_in,
        )]
        bridge.apply_to_workflow_ir(wf, bundle_in)
        bundle_out = bridge.from_workflow_ir(wf)
        pm_out = bundle_out.subgraphs[0].package_metadata
        self.assertEqual(pm_out.package_id, "roundtrip.pkg")
        self.assertEqual(pm_out.package_version, "3.1.4")
        self.assertEqual(pm_out.schema_version, "2.0")

    def test_apply_then_from_preserves_ports(self):
        bridge = DefaultLayeredIRBridge()
        wf = _StubWorkflowIR()
        bundle_in = LayeredIRBundle()
        bundle_in.subgraphs = [SubgraphIR(
            subgraph_id="sg", display_name="SG",
            inputs=[SubgraphPort(name="x", direction="input", data_type="int")],
            outputs=[SubgraphPort(name="y", direction="output", data_type="float")],
            package_metadata=PackageMetadata.create("port.pkg"),
        )]
        bridge.apply_to_workflow_ir(wf, bundle_in)
        bundle_out = bridge.from_workflow_ir(wf)
        sg = bundle_out.subgraphs[0]
        self.assertEqual(sg.inputs[0].data_type, "int")
        self.assertEqual(sg.outputs[0].data_type, "float")


# ---------------------------------------------------------------------------
# DiagnosticsKey surface test
# ---------------------------------------------------------------------------

class TestPackageMetadataTraceKey(unittest.TestCase):

    def test_key_exists_and_has_correct_value(self):
        from system.runtime.contracts import DiagnosticsKey
        self.assertEqual(DiagnosticsKey.PACKAGE_METADATA_TRACE, "package_metadata_trace")


if __name__ == "__main__":
    unittest.main()
