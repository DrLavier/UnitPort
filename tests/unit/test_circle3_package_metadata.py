#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Circle 3 — Wrapper Node Package Metadata.

Coverage
--------
MigrationInfo:
  - Default construction → all fields empty
  - to_dict() round-trip
  - from_dict() reconstructs correctly; missing keys default gracefully
  - from_dict(non-dict) → default instance
  - stamp() generates current timestamp and migrated_from

PackageMetadata:
  - Default construction: empty package_id, default versions
  - create() factory sets fields
  - is_empty() True when package_id blank, False otherwise
  - to_dict() round-trip (all fields preserved)
  - from_dict() reconstructs correctly; missing keys default gracefully
  - from_dict(non-dict) → default instance
  - from_dict preserves migration_info
  - with_migration() returns copy stamped with MigrationInfo

SubgraphPort:
  - to_dict() / from_dict() round-trip

SubgraphIR (Circle 3 additions):
  - Default package_metadata is empty PackageMetadata
  - to_dict() includes package_metadata key
  - from_dict() reconstructs package_metadata from persisted dict
  - from_dict(dict without package_metadata) → empty PackageMetadata (backward compat)
  - from_dict(non-dict) → empty subgraph

SemanticValidationResult:
  - to_dict() structure

PackageSemanticValidator.validate():
  - Identical subgraphs → is_equivalent=True, no diffs
  - Port name difference → diff reported, not equivalent
  - Port data_type difference → diff reported
  - Extra input port in B → diff reported
  - Extra internal node in A → diff reported
  - Different package_id on both metadata → warning (not diff)
  - Different schema_version → warning (not diff)
  - Empty metadata on one side → no package_id warning
  - Same internal_node_ids different order → still equivalent

PackageSemanticValidator.validate_roundtrip():
  - Roundtrip of a fully populated SubgraphIR → is_equivalent=True
  - Roundtrip of minimal SubgraphIR → is_equivalent=True

mission_persistence — package metadata helpers:
  - build_package_metadata_payload(PackageMetadata) → dict with all keys
  - build_package_metadata_payload(dict) → shallow copy
  - build_package_metadata_payload(None) → {}
  - build_package_metadata_payload(object with bad to_dict) → {}
  - extract_package_metadata_payload — top-level key present → dict
  - extract_package_metadata_payload — top-level key missing → None
  - extract_package_metadata_payload — subgraph_id lookup found → dict
  - extract_package_metadata_payload — subgraph_id not found → None
  - extract_package_metadata_payload — non-dict data → None
  - needs_package_metadata_migration — list-style subgraphs without key → True
  - needs_package_metadata_migration — dict-style subgraphs without key → True
  - needs_package_metadata_migration — all subgraphs have key → False
  - needs_package_metadata_migration — no subgraphs → False
  - needs_package_metadata_migration — non-dict data → False

migrate_mission_payload (Circle 3 extension):
  - Returns needs_package_metadata_upgrade=True when subgraphs lack metadata
  - Returns needs_package_metadata_upgrade=False when subgraphs have metadata
  - Warning text appended when migration needed
  - Existing needs_protocol_upgrade logic still works alongside new key
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.ir.layered_contracts import (
    MigrationInfo,
    PackageMetadata,
    PackageSemanticValidator,
    SemanticValidationResult,
    SubgraphIR,
    SubgraphPort,
)
from bin.core.mission_persistence import (
    build_package_metadata_payload,
    extract_package_metadata_payload,
    migrate_mission_payload,
    needs_package_metadata_migration,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _port(name: str, direction: str = "input", data_type: str = "float") -> SubgraphPort:
    return SubgraphPort(name=name, direction=direction, data_type=data_type)


def _subgraph(
    sid: str = "sg1",
    inputs=None,
    outputs=None,
    nodes=None,
    pkg_id: str = "",
) -> SubgraphIR:
    meta = PackageMetadata.create(pkg_id) if pkg_id else PackageMetadata()
    return SubgraphIR(
        subgraph_id=sid,
        display_name=sid,
        inputs=inputs or [],
        outputs=outputs or [],
        internal_node_ids=nodes or [],
        package_metadata=meta,
    )


# ===========================================================================
# TestMigrationInfo
# ===========================================================================

class TestMigrationInfo(unittest.TestCase):

    def test_default_fields_empty(self):
        m = MigrationInfo()
        self.assertEqual(m.migrated_from, "")
        self.assertEqual(m.migrated_at, "")
        self.assertEqual(m.migration_reason, "")

    def test_to_dict_has_all_keys(self):
        m = MigrationInfo(migrated_from="0.9", migrated_at="2026-01-01T00:00:00Z", migration_reason="test")
        d = m.to_dict()
        self.assertIn("migrated_from", d)
        self.assertIn("migrated_at", d)
        self.assertIn("migration_reason", d)

    def test_to_dict_values(self):
        m = MigrationInfo(migrated_from="0.9", migrated_at="ts", migration_reason="reason")
        d = m.to_dict()
        self.assertEqual(d["migrated_from"], "0.9")
        self.assertEqual(d["migrated_at"], "ts")
        self.assertEqual(d["migration_reason"], "reason")

    def test_from_dict_round_trip(self):
        m = MigrationInfo(migrated_from="0.9", migrated_at="ts", migration_reason="r")
        m2 = MigrationInfo.from_dict(m.to_dict())
        self.assertEqual(m2.migrated_from, "0.9")
        self.assertEqual(m2.migration_reason, "r")

    def test_from_dict_missing_keys_default(self):
        m = MigrationInfo.from_dict({})
        self.assertEqual(m.migrated_from, "")

    def test_from_dict_non_dict_returns_default(self):
        m = MigrationInfo.from_dict(None)
        self.assertEqual(m.migrated_from, "")
        m2 = MigrationInfo.from_dict("bad")
        self.assertEqual(m2.migrated_from, "")

    def test_stamp_sets_migrated_from(self):
        m = MigrationInfo.stamp("1.0.0", reason="auto")
        self.assertEqual(m.migrated_from, "1.0.0")
        self.assertEqual(m.migration_reason, "auto")

    def test_stamp_timestamp_not_empty(self):
        m = MigrationInfo.stamp("1.0.0")
        self.assertGreater(len(m.migrated_at), 0)
        self.assertIn("T", m.migrated_at)


# ===========================================================================
# TestPackageMetadata
# ===========================================================================

class TestPackageMetadata(unittest.TestCase):

    def test_default_package_id_empty(self):
        pm = PackageMetadata()
        self.assertEqual(pm.package_id, "")

    def test_default_versions_set(self):
        pm = PackageMetadata()
        self.assertEqual(pm.package_version, "0.1.0")
        self.assertEqual(pm.schema_version, "1.0")

    def test_create_sets_fields(self):
        pm = PackageMetadata.create("pkg_walk", "1.2.0", "2.0")
        self.assertEqual(pm.package_id, "pkg_walk")
        self.assertEqual(pm.package_version, "1.2.0")
        self.assertEqual(pm.schema_version, "2.0")

    def test_is_empty_true_when_no_id(self):
        pm = PackageMetadata()
        self.assertTrue(pm.is_empty())

    def test_is_empty_false_with_id(self):
        pm = PackageMetadata.create("my_pkg")
        self.assertFalse(pm.is_empty())

    def test_is_empty_true_whitespace_only(self):
        pm = PackageMetadata(package_id="   ")
        self.assertTrue(pm.is_empty())

    def test_to_dict_has_all_keys(self):
        pm = PackageMetadata.create("pkg")
        d = pm.to_dict()
        for key in ("package_id", "package_version", "schema_version", "migration_info"):
            self.assertIn(key, d)

    def test_to_dict_migration_info_is_dict(self):
        pm = PackageMetadata.create("pkg")
        d = pm.to_dict()
        self.assertIsInstance(d["migration_info"], dict)

    def test_from_dict_round_trip(self):
        pm = PackageMetadata.create("pkg_x", "2.0.0", "3.0")
        pm2 = PackageMetadata.from_dict(pm.to_dict())
        self.assertEqual(pm2.package_id, "pkg_x")
        self.assertEqual(pm2.package_version, "2.0.0")
        self.assertEqual(pm2.schema_version, "3.0")

    def test_from_dict_missing_keys_defaults(self):
        pm = PackageMetadata.from_dict({})
        self.assertEqual(pm.package_id, "")
        self.assertEqual(pm.package_version, "0.1.0")

    def test_from_dict_non_dict_returns_default(self):
        pm = PackageMetadata.from_dict(None)
        self.assertTrue(pm.is_empty())
        pm2 = PackageMetadata.from_dict("bad")
        self.assertTrue(pm2.is_empty())

    def test_from_dict_migration_info_restored(self):
        mi = MigrationInfo.stamp("0.9.0", "upgrade")
        pm = PackageMetadata(package_id="pkg", migration_info=mi)
        pm2 = PackageMetadata.from_dict(pm.to_dict())
        self.assertEqual(pm2.migration_info.migrated_from, "0.9.0")
        self.assertEqual(pm2.migration_info.migration_reason, "upgrade")

    def test_with_migration_preserves_id(self):
        pm = PackageMetadata.create("pkg_y", "1.0.0")
        pm2 = pm.with_migration("test upgrade")
        self.assertEqual(pm2.package_id, "pkg_y")
        self.assertEqual(pm2.migration_info.migrated_from, "1.0.0")
        self.assertEqual(pm2.migration_info.migration_reason, "test upgrade")

    def test_with_migration_does_not_mutate_original(self):
        pm = PackageMetadata.create("pkg_z")
        pm.with_migration("x")
        self.assertEqual(pm.migration_info.migrated_from, "")


# ===========================================================================
# TestSubgraphPort
# ===========================================================================

class TestSubgraphPort(unittest.TestCase):

    def test_to_dict_round_trip(self):
        p = SubgraphPort(name="speed", direction="input", data_type="float", required=True)
        p2 = SubgraphPort.from_dict(p.to_dict())
        self.assertEqual(p2.name, "speed")
        self.assertEqual(p2.direction, "input")
        self.assertEqual(p2.data_type, "float")
        self.assertTrue(p2.required)

    def test_from_dict_defaults(self):
        p = SubgraphPort.from_dict({})
        self.assertEqual(p.name, "")
        self.assertEqual(p.data_type, "any")
        self.assertFalse(p.required)


# ===========================================================================
# TestSubgraphIR
# ===========================================================================

class TestSubgraphIR(unittest.TestCase):

    def test_default_package_metadata_is_empty(self):
        sg = SubgraphIR(subgraph_id="s", display_name="S")
        self.assertTrue(sg.package_metadata.is_empty())

    def test_to_dict_includes_package_metadata(self):
        sg = _subgraph(pkg_id="my_pkg")
        d = sg.to_dict()
        self.assertIn("package_metadata", d)
        self.assertEqual(d["package_metadata"]["package_id"], "my_pkg")

    def test_from_dict_round_trip(self):
        sg = _subgraph(
            sid="test_sg",
            inputs=[_port("vel", "input")],
            outputs=[_port("out", "output", "bool")],
            nodes=["n1", "n2"],
            pkg_id="pkg_rrt",
        )
        sg.package_metadata = PackageMetadata.create("pkg_rrt", "1.0.0", "2.0")
        d = sg.to_dict()
        sg2 = SubgraphIR.from_dict(d)
        self.assertEqual(sg2.subgraph_id, "test_sg")
        self.assertEqual(sg2.package_metadata.package_id, "pkg_rrt")
        self.assertEqual(len(sg2.inputs), 1)
        self.assertEqual(sg2.inputs[0].name, "vel")
        self.assertIn("n1", sg2.internal_node_ids)

    def test_from_dict_missing_package_metadata_backward_compat(self):
        """Old dict without package_metadata key → empty PackageMetadata."""
        d = {
            "subgraph_id": "old_sg",
            "display_name": "Old",
            "inputs": [],
            "outputs": [],
            "internal_node_ids": [],
            "metadata": {},
        }
        sg = SubgraphIR.from_dict(d)
        self.assertTrue(sg.package_metadata.is_empty())

    def test_from_dict_non_dict_returns_empty(self):
        sg = SubgraphIR.from_dict(None)
        self.assertEqual(sg.subgraph_id, "")

    def test_from_dict_ports_restored(self):
        sg = _subgraph(inputs=[_port("a"), _port("b")], outputs=[_port("c", "output")])
        sg2 = SubgraphIR.from_dict(sg.to_dict())
        self.assertEqual(len(sg2.inputs), 2)
        self.assertEqual(len(sg2.outputs), 1)


# ===========================================================================
# TestSemanticValidationResult
# ===========================================================================

class TestSemanticValidationResult(unittest.TestCase):

    def test_to_dict_structure(self):
        r = SemanticValidationResult(is_equivalent=True, differences=[], warnings=["w"])
        d = r.to_dict()
        self.assertIn("is_equivalent", d)
        self.assertIn("differences", d)
        self.assertIn("warnings", d)
        self.assertTrue(d["is_equivalent"])
        self.assertEqual(d["warnings"], ["w"])


# ===========================================================================
# TestPackageSemanticValidatorValidate
# ===========================================================================

class TestPackageSemanticValidatorValidate(unittest.TestCase):

    def test_identical_subgraphs_equivalent(self):
        sg = _subgraph(inputs=[_port("x")], outputs=[_port("y", "output")], nodes=["n1"])
        result = PackageSemanticValidator.validate(sg, sg)
        self.assertTrue(result.is_equivalent)
        self.assertEqual(result.differences, [])

    def test_same_structure_different_instances_equivalent(self):
        sg_a = _subgraph(inputs=[_port("x")], nodes=["n1", "n2"])
        sg_b = _subgraph(inputs=[_port("x")], nodes=["n1", "n2"])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertTrue(result.is_equivalent)

    def test_missing_input_port_in_b_not_equivalent(self):
        sg_a = _subgraph(inputs=[_port("x"), _port("y")])
        sg_b = _subgraph(inputs=[_port("x")])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertFalse(result.is_equivalent)
        self.assertTrue(any("input" in d for d in result.differences))

    def test_extra_input_port_in_b_not_equivalent(self):
        sg_a = _subgraph(inputs=[_port("x")])
        sg_b = _subgraph(inputs=[_port("x"), _port("z")])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertFalse(result.is_equivalent)

    def test_port_data_type_difference_not_equivalent(self):
        sg_a = _subgraph(inputs=[SubgraphPort("v", "input", "float")])
        sg_b = _subgraph(inputs=[SubgraphPort("v", "input", "int")])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertFalse(result.is_equivalent)
        self.assertTrue(any("data_type" in d for d in result.differences))

    def test_output_port_missing_not_equivalent(self):
        sg_a = _subgraph(outputs=[_port("out", "output")])
        sg_b = _subgraph(outputs=[])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertFalse(result.is_equivalent)

    def test_extra_internal_node_in_a_not_equivalent(self):
        sg_a = _subgraph(nodes=["n1", "n2"])
        sg_b = _subgraph(nodes=["n1"])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertFalse(result.is_equivalent)
        self.assertTrue(any("internal nodes" in d for d in result.differences))

    def test_different_package_id_is_warning_not_diff(self):
        sg_a = _subgraph(pkg_id="pkg_alpha")
        sg_b = _subgraph(pkg_id="pkg_beta")
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertTrue(result.is_equivalent)   # no structural diff
        self.assertTrue(any("package_id" in w for w in result.warnings))

    def test_different_schema_version_is_warning_not_diff(self):
        sg_a = _subgraph(pkg_id="pkg")
        sg_b = _subgraph(pkg_id="pkg")
        sg_a.package_metadata = PackageMetadata.create("pkg", schema_version="1.0")
        sg_b.package_metadata = PackageMetadata.create("pkg", schema_version="2.0")
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertTrue(result.is_equivalent)
        self.assertTrue(any("schema_version" in w for w in result.warnings))

    def test_empty_metadata_on_one_side_no_package_id_warning(self):
        """If one or both metadata are empty, no package_id mismatch warning."""
        sg_a = _subgraph()             # empty metadata
        sg_b = _subgraph(pkg_id="p")
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertTrue(result.is_equivalent)
        self.assertFalse(any("package_id" in w for w in result.warnings))

    def test_internal_node_order_does_not_matter(self):
        sg_a = _subgraph(nodes=["n2", "n1", "n3"])
        sg_b = _subgraph(nodes=["n1", "n3", "n2"])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertTrue(result.is_equivalent)

    def test_empty_subgraphs_equivalent(self):
        sg_a = _subgraph()
        sg_b = _subgraph()
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertTrue(result.is_equivalent)

    def test_multiple_diffs_all_reported(self):
        sg_a = _subgraph(inputs=[_port("x")], nodes=["n1"])
        sg_b = _subgraph(inputs=[_port("y")], nodes=["n2"])
        result = PackageSemanticValidator.validate(sg_a, sg_b)
        self.assertFalse(result.is_equivalent)
        self.assertGreater(len(result.differences), 1)


# ===========================================================================
# TestPackageSemanticValidatorRoundtrip
# ===========================================================================

class TestPackageSemanticValidatorRoundtrip(unittest.TestCase):

    def test_roundtrip_populated_subgraph(self):
        sg = SubgraphIR(
            subgraph_id="rt_sg",
            display_name="RT SG",
            version="1.0.0",
            inputs=[_port("in1"), _port("in2", data_type="bool")],
            outputs=[_port("out1", "output", "float")],
            internal_node_ids=["a", "b", "c"],
            package_metadata=PackageMetadata.create("pkg_rt", "1.0.0", "2.0"),
        )
        result = PackageSemanticValidator.validate_roundtrip(sg)
        self.assertTrue(result.is_equivalent, result.differences)

    def test_roundtrip_minimal_subgraph(self):
        sg = SubgraphIR(subgraph_id="min", display_name="Minimal")
        result = PackageSemanticValidator.validate_roundtrip(sg)
        self.assertTrue(result.is_equivalent)

    def test_roundtrip_subgraph_with_migration_info(self):
        pm = PackageMetadata.create("pkg_m", "2.0.0")
        pm = pm.with_migration("version bump")
        sg = SubgraphIR(
            subgraph_id="mig_sg",
            display_name="Mig SG",
            internal_node_ids=["x"],
            package_metadata=pm,
        )
        result = PackageSemanticValidator.validate_roundtrip(sg)
        self.assertTrue(result.is_equivalent, result.differences)


# ===========================================================================
# TestPersistenceHelpers
# ===========================================================================

class TestBuildPackageMetadataPayload(unittest.TestCase):

    def test_with_package_metadata_instance(self):
        pm = PackageMetadata.create("pkg_persist", "1.0.0", "1.0")
        d = build_package_metadata_payload(pm)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["package_id"], "pkg_persist")
        self.assertIn("migration_info", d)

    def test_with_plain_dict(self):
        d = build_package_metadata_payload({"package_id": "x"})
        self.assertEqual(d["package_id"], "x")

    def test_with_none_returns_empty(self):
        d = build_package_metadata_payload(None)
        self.assertEqual(d, {})

    def test_shallow_copy_of_dict(self):
        original = {"package_id": "y"}
        d = build_package_metadata_payload(original)
        d["package_id"] = "mutated"
        self.assertEqual(original["package_id"], "y")

    def test_bad_to_dict_returns_empty(self):
        class Bad:
            def to_dict(self): raise RuntimeError("oops")
        d = build_package_metadata_payload(Bad())
        self.assertEqual(d, {})


class TestExtractPackageMetadataPayload(unittest.TestCase):

    def test_top_level_key_found(self):
        data = {"package_metadata": {"package_id": "top_pkg"}}
        result = extract_package_metadata_payload(data)
        self.assertIsNotNone(result)
        self.assertEqual(result["package_id"], "top_pkg")

    def test_top_level_key_missing_returns_none(self):
        result = extract_package_metadata_payload({"other": 1})
        self.assertIsNone(result)

    def test_subgraph_id_lookup_found(self):
        data = {
            "subgraphs": {
                "sg_abc": {"package_metadata": {"package_id": "found_pkg"}}
            }
        }
        result = extract_package_metadata_payload(data, subgraph_id="sg_abc")
        self.assertIsNotNone(result)
        self.assertEqual(result["package_id"], "found_pkg")

    def test_subgraph_id_lookup_not_found(self):
        data = {"subgraphs": {"sg_xyz": {"package_metadata": {}}}}
        result = extract_package_metadata_payload(data, subgraph_id="nonexistent")
        self.assertIsNone(result)

    def test_subgraph_id_no_package_metadata_key(self):
        data = {"subgraphs": {"sg1": {"other": "value"}}}
        result = extract_package_metadata_payload(data, subgraph_id="sg1")
        self.assertIsNone(result)

    def test_non_dict_data_returns_none(self):
        self.assertIsNone(extract_package_metadata_payload(None))
        self.assertIsNone(extract_package_metadata_payload("bad"))
        self.assertIsNone(extract_package_metadata_payload(123))

    def test_returns_shallow_copy(self):
        data = {"package_metadata": {"package_id": "copy_test"}}
        result = extract_package_metadata_payload(data)
        result["package_id"] = "mutated"
        self.assertEqual(data["package_metadata"]["package_id"], "copy_test")


class TestNeedsPackageMetadataMigration(unittest.TestCase):

    def test_list_subgraphs_without_key_true(self):
        data = {"subgraphs": [{"subgraph_id": "s1"}, {"subgraph_id": "s2"}]}
        self.assertTrue(needs_package_metadata_migration(data))

    def test_dict_subgraphs_without_key_true(self):
        data = {"subgraphs": {"s1": {"subgraph_id": "s1"}}}
        self.assertTrue(needs_package_metadata_migration(data))

    def test_all_subgraphs_have_key_false(self):
        data = {"subgraphs": [{"package_metadata": {}}]}
        self.assertFalse(needs_package_metadata_migration(data))

    def test_dict_subgraphs_all_have_key_false(self):
        data = {"subgraphs": {"s1": {"package_metadata": {"package_id": "x"}}}}
        self.assertFalse(needs_package_metadata_migration(data))

    def test_no_subgraphs_key_false(self):
        self.assertFalse(needs_package_metadata_migration({"nodes": []}))

    def test_empty_subgraphs_false(self):
        self.assertFalse(needs_package_metadata_migration({"subgraphs": []}))
        self.assertFalse(needs_package_metadata_migration({"subgraphs": {}}))

    def test_non_dict_data_false(self):
        self.assertFalse(needs_package_metadata_migration(None))
        self.assertFalse(needs_package_metadata_migration("bad"))


# ===========================================================================
# TestMigrateMissionPayloadCircle3
# ===========================================================================

class TestMigrateMissionPayloadCircle3(unittest.TestCase):

    def _base_data(self, **extra) -> dict:
        d = {"nodes": [], "connections": [], "schema_version": "1.4"}
        d.update(extra)
        return d

    def test_needs_package_metadata_upgrade_true_when_subgraph_lacks_key(self):
        data = self._base_data(subgraphs=[{"subgraph_id": "sg1"}])
        result = migrate_mission_payload(data)
        self.assertTrue(result["needs_package_metadata_upgrade"])

    def test_needs_package_metadata_upgrade_false_when_subgraphs_have_key(self):
        data = self._base_data(
            subgraphs=[{"subgraph_id": "sg1", "package_metadata": {"package_id": "p"}}]
        )
        result = migrate_mission_payload(data)
        self.assertFalse(result["needs_package_metadata_upgrade"])

    def test_needs_package_metadata_upgrade_false_when_no_subgraphs(self):
        data = self._base_data()
        result = migrate_mission_payload(data)
        self.assertFalse(result["needs_package_metadata_upgrade"])

    def test_warning_added_when_migration_needed(self):
        data = self._base_data(subgraphs=[{"subgraph_id": "sg_old"}])
        result = migrate_mission_payload(data)
        self.assertTrue(
            any("package_metadata" in w for w in result["warnings"])
        )

    def test_no_warning_when_no_migration_needed(self):
        data = self._base_data()
        result = migrate_mission_payload(data)
        self.assertFalse(
            any("package_metadata" in w for w in result["warnings"])
        )

    def test_protocol_upgrade_still_detected_alongside_new_key(self):
        # Old file without protocol upgrade AND with untracked subgraphs
        data = {
            "nodes": [], "connections": [],
            "schema_version": "1.3",  # predates 1.4 protocol contract
            "subgraphs": [{"subgraph_id": "sg_old"}],
        }
        result = migrate_mission_payload(data)
        self.assertTrue(result["needs_protocol_upgrade"])
        self.assertTrue(result["needs_package_metadata_upgrade"])
        self.assertGreaterEqual(len(result["warnings"]), 2)

    def test_result_includes_required_keys(self):
        result = migrate_mission_payload(self._base_data())
        for key in ("needs_protocol_upgrade", "needs_package_metadata_upgrade",
                    "prior_schema_version", "warnings"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
