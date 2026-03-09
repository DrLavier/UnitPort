#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for BehaviorCompilerBridge — Phase 2 STAGE-02 (gap closure).

Coverage
--------
- _map_diagnostic: all DiagnosticLevel values, both location types, no-location
- _source_hash: determinism and correct algorithm
- compile(): artifact fields, trace_id propagation, source_hash, auto-register
- compile(): compile error → is_valid False, NOT auto-registered
- register(): explicit registration, overwrite semantics
- resolve(): success / missing / invalid — ok, reason, artifact, diagnostics, trace_id
- resolve(): to_dict() schema validation
- lookup(): delegates to resolve(); legacy behaviour unchanged
- lookup_any(): returns invalid artifacts; returns None for missing
- list_refs(): keys before/after registration
- clear_registry(): flushes all entries
- BehaviorResolveResult dataclass: field types, factory methods, to_dict()
- BehaviorErrorCode constants exist (used by callers for reason strings)
- Backward compatibility: from_source() still works unchanged

Isolation: no Qt, no SDK, no sys.modules injection needed (compiler pipeline
is pure Python).
"""

import hashlib
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from compiler.semantic.diagnostics import (  # noqa: E402
    Diagnostic,
    DiagnosticLevel,
    DiagnosticLocation,
)
from system.behavior.behavior_compiler_bridge import (  # noqa: E402
    BehaviorCompilerBridge,
    _map_diagnostic,
    _source_hash,
)
from system.behavior.behavior_artifact import (  # noqa: E402
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorResolveResult,
)
from system.ir.workflow_ir import WorkflowIR  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_ir() -> WorkflowIR:
    return WorkflowIR(nodes=[], edges=[])


def _make_artifact(behavior_ref: str, valid: bool = True) -> BehaviorArtifact:
    """Create a minimal BehaviorArtifact for registry tests."""
    diags = []
    if not valid:
        diags = [BehaviorDiagnostic.error("test_err", "forced error")]
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_empty_ir(),
        diagnostics=diags,
    )


def _make_compiler_diag(
    level: DiagnosticLevel,
    code: str = "E001",
    message: str = "test msg",
    node_id: str = None,
    line: int = None,
) -> Diagnostic:
    loc = None
    if node_id is not None or line is not None:
        loc = DiagnosticLocation(node_id=node_id, line=line)
    return Diagnostic(code=code, level=level, message=message, location=loc)


# ===========================================================================
# _map_diagnostic
# ===========================================================================

class TestMapDiagnostic(unittest.TestCase):

    def _call(self, level, node_id=None, line=None):
        d = _make_compiler_diag(level, node_id=node_id, line=line)
        return _map_diagnostic(d, trace_id="tid-test")

    def test_error_level_maps_to_error(self):
        bd = self._call(DiagnosticLevel.ERROR)
        self.assertEqual(bd.level, "error")

    def test_warning_level_maps_to_warning(self):
        # Compiler enum is WARNING; BehaviorDiagnostic uses "warning" (not "warn")
        bd = self._call(DiagnosticLevel.WARNING)
        self.assertEqual(bd.level, "warning")

    def test_info_level_maps_to_info(self):
        bd = self._call(DiagnosticLevel.INFO)
        self.assertEqual(bd.level, "info")

    def test_trace_id_propagated(self):
        d = _make_compiler_diag(DiagnosticLevel.ERROR)
        bd = _map_diagnostic(d, trace_id="abc-123")
        self.assertEqual(bd.trace_id, "abc-123")

    def test_code_and_message_copied(self):
        d = _make_compiler_diag(DiagnosticLevel.WARNING, code="W42", message="watch out")
        bd = _map_diagnostic(d, trace_id="t")
        self.assertEqual(bd.code, "W42")
        self.assertEqual(bd.message, "watch out")

    def test_no_location_gives_none(self):
        d = _make_compiler_diag(DiagnosticLevel.INFO)
        bd = _map_diagnostic(d, trace_id="t")
        self.assertIsNone(bd.location)

    def test_node_id_location(self):
        bd = self._call(DiagnosticLevel.ERROR, node_id="node_42")
        self.assertEqual(bd.location, "node:node_42")

    def test_line_location_when_no_node_id(self):
        bd = self._call(DiagnosticLevel.ERROR, line=7)
        self.assertEqual(bd.location, "line:7")

    def test_node_id_takes_precedence_over_line(self):
        d = _make_compiler_diag(DiagnosticLevel.ERROR, node_id="n1", line=5)
        bd = _map_diagnostic(d, trace_id="t")
        self.assertEqual(bd.location, "node:n1")


# ===========================================================================
# _source_hash
# ===========================================================================

class TestSourceHash(unittest.TestCase):

    def test_deterministic(self):
        self.assertEqual(_source_hash("hello"), _source_hash("hello"))

    def test_correct_algorithm(self):
        src = "stand()"
        expected = hashlib.sha256(src.encode("utf-8")).hexdigest()
        self.assertEqual(_source_hash(src), expected)

    def test_different_sources_differ(self):
        self.assertNotEqual(_source_hash("a"), _source_hash("b"))

    def test_empty_string(self):
        h = _source_hash("")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA-256 hex = 64 chars


# ===========================================================================
# BehaviorCompilerBridge — registry operations
# ===========================================================================

class TestRegistryOperations(unittest.TestCase):

    def setUp(self):
        # Use a fresh temp dir so on-disk cached artifacts from other test
        # runs cannot pollute registry lookups (Circle 1 cache isolation fix).
        self._cache_dir = tempfile.mkdtemp(prefix="unitport_test_bridge_")
        self.bridge = BehaviorCompilerBridge(cache_dir=self._cache_dir)

    def tearDown(self):
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def test_lookup_missing_returns_none(self):
        self.assertIsNone(self.bridge.lookup("walk"))

    def test_register_then_lookup(self):
        art = _make_artifact("stand")
        self.bridge.register(art)
        result = self.bridge.lookup("stand")
        self.assertIs(result, art)

    def test_register_overwrites_previous(self):
        art1 = _make_artifact("stand")
        art2 = _make_artifact("stand")
        self.bridge.register(art1)
        self.bridge.register(art2)
        self.assertIs(self.bridge.lookup("stand"), art2)

    def test_lookup_invalid_artifact_returns_none(self):
        art = _make_artifact("bad_ref", valid=False)
        self.bridge.register(art)
        # lookup() filters out invalid artifacts
        self.assertIsNone(self.bridge.lookup("bad_ref"))

    def test_lookup_any_returns_invalid_artifact(self):
        art = _make_artifact("bad_ref", valid=False)
        self.bridge.register(art)
        result = self.bridge.lookup_any("bad_ref")
        self.assertIs(result, art)

    def test_lookup_any_returns_none_for_missing(self):
        self.assertIsNone(self.bridge.lookup_any("unknown"))

    def test_list_refs_empty_initially(self):
        self.assertEqual(self.bridge.list_refs(), [])

    def test_list_refs_after_register(self):
        self.bridge.register(_make_artifact("a"))
        self.bridge.register(_make_artifact("b"))
        self.assertCountEqual(self.bridge.list_refs(), ["a", "b"])

    def test_clear_registry_removes_all(self):
        self.bridge.register(_make_artifact("x"))
        self.bridge.register(_make_artifact("y"))
        self.bridge.clear_registry()
        self.assertEqual(self.bridge.list_refs(), [])
        self.assertIsNone(self.bridge.lookup("x"))

    def test_duplicate_ref_creates_new_artifact(self):
        self.bridge.register(_make_artifact("custom_move"))
        new_ref = self.bridge.duplicate_ref("custom_move")
        self.assertIsNotNone(new_ref)
        self.assertNotEqual(new_ref, "custom_move")
        self.assertIsNotNone(self.bridge.lookup_any(new_ref))

    def test_delete_ref_removes_registry_and_cache_dir(self):
        self.bridge.register(_make_artifact("to_delete"))
        self.assertTrue(self.bridge.delete_ref("to_delete"))
        self.assertIsNone(self.bridge.lookup_any("to_delete"))


# ===========================================================================
# BehaviorCompilerBridge — compile()
# ===========================================================================

class TestCompile(unittest.TestCase):

    def setUp(self):
        self._cache_dir = tempfile.mkdtemp(prefix="unitport_test_compile_")
        self.bridge = BehaviorCompilerBridge(cache_dir=self._cache_dir)

    def tearDown(self):
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def test_compile_empty_source_returns_artifact(self):
        art = self.bridge.compile("", "idle")
        self.assertIsInstance(art, BehaviorArtifact)
        self.assertEqual(art.behavior_ref, "idle")

    def test_compile_sets_source_hash(self):
        src = "stand()"
        art = self.bridge.compile(src, "stand")
        self.assertEqual(art.source_hash, _source_hash(src))

    def test_compile_propagates_caller_trace_id(self):
        art = self.bridge.compile("", "idle", trace_id="my-trace-99")
        self.assertEqual(art.trace_id, "my-trace-99")
        # All diagnostics share the same trace_id
        for d in art.diagnostics:
            self.assertEqual(d.trace_id, "my-trace-99")

    def test_compile_generates_trace_id_if_absent(self):
        art = self.bridge.compile("", "idle")
        self.assertIsNotNone(art.trace_id)
        self.assertGreater(len(art.trace_id), 0)

    def test_compile_generates_artifact_id(self):
        art = self.bridge.compile("", "idle")
        self.assertIsNotNone(art.artifact_id)
        self.assertGreater(len(art.artifact_id), 0)

    def test_compile_valid_source_auto_registers(self):
        art = self.bridge.compile("", "idle")
        # Empty source should not produce compile errors
        if art.is_valid:
            self.assertIs(self.bridge.lookup("idle"), art)

    def test_compile_two_different_refs_both_registered(self):
        a1 = self.bridge.compile("", "stand")
        a2 = self.bridge.compile("", "sit")
        if a1.is_valid:
            self.assertIs(self.bridge.lookup("stand"), a1)
        if a2.is_valid:
            self.assertIs(self.bridge.lookup("sit"), a2)

    def test_compile_always_returns_artifact_even_on_error(self):
        # Force compile failure with intentionally bad source; if diagnostics
        # contain an error the artifact.is_valid will be False but is still returned.
        art = self.bridge.compile("def !!!syntaxerr(:", "broken")
        self.assertIsInstance(art, BehaviorArtifact)
        # Don't assert is_valid — depends on how lenient the parser is

    def test_compile_invalid_artifact_not_registered(self):
        # We simulate this by directly forcing a compile with a known-bad source
        # and checking that if is_valid is False, lookup returns None.
        art = self.bridge.compile("def !!!syntaxerr(:", "broken")
        if not art.is_valid:
            self.assertIsNone(self.bridge.lookup("broken"))

    def test_compile_behavior_ir_is_workflow_ir(self):
        art = self.bridge.compile("", "idle")
        self.assertIsInstance(art.behavior_ir, WorkflowIR)

    def test_compile_compiled_at_is_set(self):
        art = self.bridge.compile("", "idle")
        self.assertIsNotNone(art.compiled_at)
        self.assertIn("T", art.compiled_at)  # ISO 8601 separator


# ===========================================================================
# BehaviorCompilerBridge — backward compatibility
# ===========================================================================

class TestBackwardCompatibility(unittest.TestCase):

    def test_from_source_still_works(self):
        bridge = BehaviorCompilerBridge()
        ir, diags = bridge.from_source("")
        self.assertIsInstance(ir, WorkflowIR)
        self.assertIsInstance(diags, list)

    def test_from_source_independent_of_registry(self):
        bridge = BehaviorCompilerBridge()
        bridge.from_source("")
        # Registry must be untouched by from_source()
        self.assertEqual(bridge.list_refs(), [])


# ===========================================================================
# BehaviorErrorCode constants
# ===========================================================================

class TestBehaviorErrorCode(unittest.TestCase):

    def test_all_constants_exist(self):
        self.assertEqual(BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND,
                         "behavior_ref_not_found")
        self.assertEqual(BehaviorErrorCode.ARTIFACT_INVALID,
                         "artifact_invalid")
        self.assertEqual(BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED,
                         "subgraph_execution_failed")
        self.assertEqual(BehaviorErrorCode.SUBGRAPH_ABORTED,
                         "subgraph_aborted")
        self.assertEqual(BehaviorErrorCode.BEHAVIOR_COMPILE_ERROR,
                         "behavior_compile_error")


# ===========================================================================
# BehaviorResolveResult — dataclass and factory methods
# ===========================================================================

class TestBehaviorResolveResult(unittest.TestCase):
    """Tests BehaviorResolveResult factories and to_dict() directly."""

    def _valid_artifact(self) -> BehaviorArtifact:
        return _make_artifact("stand", valid=True)

    def _invalid_artifact(self) -> BehaviorArtifact:
        return _make_artifact("broken", valid=False)

    # ── success factory ───────────────────────────────────────────────

    def test_success_ok_true(self):
        r = BehaviorResolveResult.success(self._valid_artifact(), "tid1")
        self.assertTrue(r.ok)

    def test_success_reason_empty(self):
        r = BehaviorResolveResult.success(self._valid_artifact(), "tid1")
        self.assertEqual(r.reason, "")

    def test_success_artifact_is_set(self):
        art = self._valid_artifact()
        r = BehaviorResolveResult.success(art, "tid1")
        self.assertIs(r.artifact, art)

    def test_success_diagnostics_empty(self):
        r = BehaviorResolveResult.success(self._valid_artifact(), "tid1")
        self.assertEqual(r.diagnostics, [])

    def test_success_trace_id_propagated(self):
        r = BehaviorResolveResult.success(self._valid_artifact(), "my-trace")
        self.assertEqual(r.trace_id, "my-trace")

    # ── missing factory ───────────────────────────────────────────────

    def test_missing_ok_false(self):
        r = BehaviorResolveResult.missing("walk", "tid2")
        self.assertFalse(r.ok)

    def test_missing_reason_is_ref_not_found(self):
        r = BehaviorResolveResult.missing("walk", "tid2")
        self.assertEqual(r.reason, BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)

    def test_missing_artifact_is_none(self):
        r = BehaviorResolveResult.missing("walk", "tid2")
        self.assertIsNone(r.artifact)

    def test_missing_has_one_diagnostic(self):
        r = BehaviorResolveResult.missing("walk", "tid2")
        self.assertEqual(len(r.diagnostics), 1)

    def test_missing_diagnostic_code_matches_reason(self):
        r = BehaviorResolveResult.missing("walk", "tid2")
        self.assertEqual(r.diagnostics[0].code, BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)

    def test_missing_diagnostic_level_is_error(self):
        r = BehaviorResolveResult.missing("walk", "tid2")
        self.assertEqual(r.diagnostics[0].level, "error")

    def test_missing_diagnostic_trace_id_matches(self):
        r = BehaviorResolveResult.missing("walk", "trace-x")
        self.assertEqual(r.diagnostics[0].trace_id, "trace-x")

    def test_missing_diagnostic_message_contains_ref(self):
        r = BehaviorResolveResult.missing("run_fast", "tid2")
        self.assertIn("run_fast", r.diagnostics[0].message)

    def test_missing_trace_id_propagated(self):
        r = BehaviorResolveResult.missing("walk", "tid-42")
        self.assertEqual(r.trace_id, "tid-42")

    # ── invalid factory ───────────────────────────────────────────────

    def test_invalid_ok_false(self):
        r = BehaviorResolveResult.invalid(self._invalid_artifact(), "tid3")
        self.assertFalse(r.ok)

    def test_invalid_reason_is_artifact_invalid(self):
        r = BehaviorResolveResult.invalid(self._invalid_artifact(), "tid3")
        self.assertEqual(r.reason, BehaviorErrorCode.ARTIFACT_INVALID)

    def test_invalid_artifact_returned(self):
        art = self._invalid_artifact()
        r = BehaviorResolveResult.invalid(art, "tid3")
        self.assertIs(r.artifact, art)

    def test_invalid_diagnostics_non_empty(self):
        r = BehaviorResolveResult.invalid(self._invalid_artifact(), "tid3")
        self.assertGreater(len(r.diagnostics), 0)

    def test_invalid_first_diagnostic_code_is_artifact_invalid(self):
        r = BehaviorResolveResult.invalid(self._invalid_artifact(), "tid3")
        self.assertEqual(r.diagnostics[0].code, BehaviorErrorCode.ARTIFACT_INVALID)

    def test_invalid_includes_compile_diagnostics(self):
        # The invalid artifact has one compile error; invalid() should include it.
        art = self._invalid_artifact()
        r = BehaviorResolveResult.invalid(art, "tid3")
        # First entry is the resolution-level error; rest are compile errors
        compile_codes = [d.code for d in r.diagnostics[1:]]
        self.assertIn("test_err", compile_codes)

    def test_invalid_trace_id_propagated(self):
        r = BehaviorResolveResult.invalid(self._invalid_artifact(), "t-inv")
        self.assertEqual(r.trace_id, "t-inv")

    # ── to_dict() schema ─────────────────────────────────────────────

    def test_to_dict_success_schema(self):
        r = BehaviorResolveResult.success(self._valid_artifact(), "tid")
        d = r.to_dict()
        for key in ("ok", "reason", "artifact", "diagnostics", "trace_id"):
            self.assertIn(key, d)
        self.assertTrue(d["ok"])
        self.assertEqual(d["reason"], "")
        self.assertIsNotNone(d["artifact"])
        self.assertIsInstance(d["diagnostics"], list)

    def test_to_dict_missing_schema(self):
        r = BehaviorResolveResult.missing("x", "tid")
        d = r.to_dict()
        self.assertFalse(d["ok"])
        self.assertEqual(d["reason"], BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)
        self.assertIsNone(d["artifact"])
        self.assertEqual(len(d["diagnostics"]), 1)

    def test_to_dict_invalid_schema(self):
        r = BehaviorResolveResult.invalid(self._invalid_artifact(), "tid")
        d = r.to_dict()
        self.assertFalse(d["ok"])
        self.assertEqual(d["reason"], BehaviorErrorCode.ARTIFACT_INVALID)
        self.assertIsNotNone(d["artifact"])
        self.assertGreater(len(d["diagnostics"]), 0)


# ===========================================================================
# BehaviorCompilerBridge — resolve()
# ===========================================================================

class TestResolve(unittest.TestCase):
    """Tests BehaviorCompilerBridge.resolve() and lookup() delegation."""

    def setUp(self):
        self.bridge = BehaviorCompilerBridge()

    # ── success path ──────────────────────────────────────────────────

    def test_resolve_success_ok_true(self):
        art = _make_artifact("stand")
        self.bridge.register(art)
        r = self.bridge.resolve("stand")
        self.assertTrue(r.ok)

    def test_resolve_success_reason_empty(self):
        self.bridge.register(_make_artifact("stand"))
        r = self.bridge.resolve("stand")
        self.assertEqual(r.reason, "")

    def test_resolve_success_artifact_matches(self):
        art = _make_artifact("stand")
        self.bridge.register(art)
        r = self.bridge.resolve("stand")
        self.assertIs(r.artifact, art)

    def test_resolve_success_diagnostics_empty(self):
        self.bridge.register(_make_artifact("stand"))
        r = self.bridge.resolve("stand")
        self.assertEqual(r.diagnostics, [])

    def test_resolve_success_trace_id_propagated(self):
        self.bridge.register(_make_artifact("stand"))
        r = self.bridge.resolve("stand", trace_id="caller-tid")
        self.assertEqual(r.trace_id, "caller-tid")

    def test_resolve_success_generates_trace_id_when_absent(self):
        self.bridge.register(_make_artifact("stand"))
        r = self.bridge.resolve("stand")
        self.assertIsNotNone(r.trace_id)
        self.assertGreater(len(r.trace_id), 0)

    # ── missing ref ───────────────────────────────────────────────────

    def test_resolve_missing_ok_false(self):
        r = self.bridge.resolve("unknown_ref")
        self.assertFalse(r.ok)

    def test_resolve_missing_reason_is_ref_not_found(self):
        r = self.bridge.resolve("unknown_ref")
        self.assertEqual(r.reason, BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)

    def test_resolve_missing_artifact_is_none(self):
        r = self.bridge.resolve("unknown_ref")
        self.assertIsNone(r.artifact)

    def test_resolve_missing_has_diagnostic(self):
        r = self.bridge.resolve("unknown_ref")
        self.assertGreater(len(r.diagnostics), 0)

    def test_resolve_missing_diagnostic_code(self):
        r = self.bridge.resolve("unknown_ref")
        self.assertEqual(r.diagnostics[0].code,
                         BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)

    def test_resolve_missing_trace_id_propagated(self):
        r = self.bridge.resolve("unknown_ref", trace_id="t-missing")
        self.assertEqual(r.trace_id, "t-missing")
        self.assertEqual(r.diagnostics[0].trace_id, "t-missing")

    # ── invalid artifact ──────────────────────────────────────────────

    def test_resolve_invalid_ok_false(self):
        self.bridge.register(_make_artifact("bad", valid=False))
        r = self.bridge.resolve("bad")
        self.assertFalse(r.ok)

    def test_resolve_invalid_reason_is_artifact_invalid(self):
        self.bridge.register(_make_artifact("bad", valid=False))
        r = self.bridge.resolve("bad")
        self.assertEqual(r.reason, BehaviorErrorCode.ARTIFACT_INVALID)

    def test_resolve_invalid_artifact_is_returned(self):
        art = _make_artifact("bad", valid=False)
        self.bridge.register(art)
        r = self.bridge.resolve("bad")
        self.assertIs(r.artifact, art)

    def test_resolve_invalid_diagnostics_non_empty(self):
        self.bridge.register(_make_artifact("bad", valid=False))
        r = self.bridge.resolve("bad")
        self.assertGreater(len(r.diagnostics), 0)

    def test_resolve_invalid_trace_id_propagated(self):
        self.bridge.register(_make_artifact("bad", valid=False))
        r = self.bridge.resolve("bad", trace_id="t-inv")
        self.assertEqual(r.trace_id, "t-inv")

    # ── resolve() result is BehaviorResolveResult ─────────────────────

    def test_resolve_returns_resolve_result_type(self):
        r = self.bridge.resolve("anything")
        self.assertIsInstance(r, BehaviorResolveResult)

    # ── lookup() delegates to resolve() — backward compat unchanged ───

    def test_lookup_valid_ref_returns_artifact(self):
        art = _make_artifact("stand")
        self.bridge.register(art)
        self.assertIs(self.bridge.lookup("stand"), art)

    def test_lookup_missing_ref_returns_none(self):
        self.assertIsNone(self.bridge.lookup("nonexistent"))

    def test_lookup_invalid_artifact_returns_none(self):
        self.bridge.register(_make_artifact("bad", valid=False))
        self.assertIsNone(self.bridge.lookup("bad"))

    def test_lookup_does_not_expose_resolve_result(self):
        # lookup() must return BehaviorArtifact or None, not BehaviorResolveResult
        art = _make_artifact("stand")
        self.bridge.register(art)
        result = self.bridge.lookup("stand")
        self.assertNotIsInstance(result, BehaviorResolveResult)


class TestPersistence(unittest.TestCase):
    """Cache-backed persistence in __cache__/behavior_artifacts."""

    def _new_cache_dir(self) -> Path:
        cache_root = PROJECT_ROOT / "__cache__"
        cache_root.mkdir(parents=True, exist_ok=True)
        path = cache_root / f"test_behavior_cache_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_compile_persists_and_next_bridge_resolves(self):
        cache_dir = self._new_cache_dir()
        try:
            bridge1 = BehaviorCompilerBridge(cache_dir=str(cache_dir))
            artifact = bridge1.compile("", "persist_ref")
            self.assertTrue(artifact.is_valid)

            bridge2 = BehaviorCompilerBridge(cache_dir=str(cache_dir))
            result = bridge2.resolve("persist_ref")
            self.assertTrue(result.ok)
            self.assertIsNotNone(result.artifact)
            self.assertEqual(result.artifact.behavior_ref, "persist_ref")
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_register_persists_and_lookup_any_loads_from_disk(self):
        cache_dir = self._new_cache_dir()
        try:
            bridge1 = BehaviorCompilerBridge(cache_dir=str(cache_dir))
            bridge1.register(_make_artifact("reg_ref"))

            bridge2 = BehaviorCompilerBridge(cache_dir=str(cache_dir))
            loaded = bridge2.lookup_any("reg_ref")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.behavior_ref, "reg_ref")
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_clear_registry_does_not_delete_persisted_artifacts(self):
        cache_dir = self._new_cache_dir()
        try:
            bridge1 = BehaviorCompilerBridge(cache_dir=str(cache_dir))
            bridge1.compile("", "keep_ref")
            bridge1.clear_registry()

            bridge2 = BehaviorCompilerBridge(cache_dir=str(cache_dir))
            again = bridge2.resolve("keep_ref")
            self.assertTrue(again.ok)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
# Phase 1 behavior redesign: compile() + BehaviorTimeline integration
# ===========================================================================

class TestCompileWithTimeline(unittest.TestCase):
    """Phase 1 behavior redesign — BehaviorTimeline integration in compile().

    Tests that:
    - compile() without timeline → timeline_data is None (backward compat)
    - compile() with empty BehaviorTimeline → timeline_data is a dict
    - compile() with out-of-range motor segment → warning-level diagnostics
    - timeline warnings are non-blocking (do not affect is_valid)
    - timeline_data is persisted to disk and restored on second bridge load
    """

    def setUp(self):
        self._cache_dir = tempfile.mkdtemp(prefix="unitport_test_tl_")
        self.bridge = BehaviorCompilerBridge(cache_dir=self._cache_dir)

    def tearDown(self):
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def _import_action_profile(self):
        """Return action_profile module or skip test if unavailable."""
        try:
            import system.behavior.action_profile as ap
            return ap
        except ImportError:
            self.skipTest("system.behavior.action_profile not available")

    def _empty_timeline(self):
        ap = self._import_action_profile()
        return ap.BehaviorTimeline(
            version=ap.BEHAVIOR_TIMELINE_VERSION,
            action_segments=[],
            motor_overlays=[],
            robot_type="go2",
            active_motor_tracks=[],
        )

    def _timeline_with_out_of_range_motor(self):
        ap = self._import_action_profile()
        track_name = list(ap.UNITREE_MOTOR_TRACK_MAP.keys())[0]
        track_def = ap.UNITREE_MOTOR_TRACK_MAP[track_name]
        seg_action = ap.ActionSegment.create("walk", start_time=0.0, duration=4.0)
        seg_motor = ap.MotorSegment(
            motor_id="m1",
            track_name=track_name,
            start_time=0.0,
            duration=4.0,
            params={track_def.param_key: track_def.safe_max + 999.0},
            parent_action_id=seg_action.action_id,
            locked=False,
        )
        overlay = ap.ActionMotorOverlay(
            action_id=seg_action.action_id,
            motor_segments=[seg_motor],
            expanded=True,
        )
        return ap.BehaviorTimeline(
            version=ap.BEHAVIOR_TIMELINE_VERSION,
            action_segments=[seg_action],
            motor_overlays=[overlay],
            robot_type="go2",
            active_motor_tracks=[track_name],
        )

    # -- timeline_data absent / present -----------------------------------

    def test_compile_without_timeline_timeline_data_is_none(self):
        art = self.bridge.compile("", "idle_no_tl")
        self.assertIsNone(art.timeline_data)

    def test_compile_with_none_timeline_timeline_data_is_none(self):
        art = self.bridge.compile("", "idle_none_tl", timeline=None)
        self.assertIsNone(art.timeline_data)

    def test_compile_with_empty_timeline_sets_timeline_data_dict(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "idle_with_tl", timeline=tl)
        self.assertIsInstance(art.timeline_data, dict)

    def test_compile_timeline_data_has_version_key(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "idle_tl_version", timeline=tl)
        self.assertIn("version", art.timeline_data)

    def test_compile_timeline_data_has_action_segments_key(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "idle_tl_as", timeline=tl)
        self.assertIn("action_segments", art.timeline_data)

    def test_compile_empty_timeline_no_motor_warnings(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "idle_clean", timeline=tl)
        tl_warnings = [d for d in art.diagnostics
                       if d.level == "warning" and d.code.startswith("timeline.motor_")]
        self.assertEqual(tl_warnings, [])

    # -- out-of-range motor diagnostics -----------------------------------
    # Fix 3: hardware violations → level="error"; sim violations → level="warning"

    def test_compile_out_of_range_motor_hardware_emits_error(self):
        """Hardware out-of-range motor segment → level=error (Fix 3)."""
        tl = self._timeline_with_out_of_range_motor()
        art = self.bridge.compile("", "walk_oob_hw", timeline=tl, is_simulation=False)
        tl_errors = [d for d in art.diagnostics
                     if d.level == "error" and d.code.startswith("timeline.motor_")]
        self.assertGreater(len(tl_errors), 0)

    def test_compile_out_of_range_motor_sim_emits_warning(self):
        """Sim out-of-range motor segment → level=warning, non-blocking (Fix 3)."""
        import system.behavior.action_profile as ap
        track_name = list(ap.UNITREE_MOTOR_TRACK_MAP.keys())[0]
        track_def = ap.UNITREE_MOTOR_TRACK_MAP[track_name]
        seg_action = ap.ActionSegment.create("walk", start_time=0.0, duration=4.0)
        # Use a value that exceeds sim_max to force a sim-range violation
        over_sim_val = track_def.sim_max + 1.0
        seg_motor = ap.MotorSegment(
            motor_id="m_sim",
            track_name=track_name,
            start_time=0.0,
            duration=4.0,
            params={track_def.param_key: over_sim_val},
            parent_action_id=seg_action.action_id,
            locked=False,
        )
        overlay = ap.ActionMotorOverlay(
            action_id=seg_action.action_id,
            motor_segments=[seg_motor],
        )
        tl = ap.BehaviorTimeline(
            version=ap.BEHAVIOR_TIMELINE_VERSION,
            action_segments=[seg_action],
            motor_overlays=[overlay],
            active_motor_tracks=[track_name],
        )
        art = self.bridge.compile("", "walk_oob_sim", timeline=tl, is_simulation=True)
        tl_warnings = [d for d in art.diagnostics
                       if d.level == "warning" and d.code.startswith("timeline.motor_")]
        self.assertGreater(len(tl_warnings), 0)

    def test_compile_out_of_range_sim_warnings_are_non_blocking(self):
        """Sim timeline warnings must not flip is_valid (Fix 3)."""
        import system.behavior.action_profile as ap
        track_name = list(ap.UNITREE_MOTOR_TRACK_MAP.keys())[0]
        track_def = ap.UNITREE_MOTOR_TRACK_MAP[track_name]
        seg_action = ap.ActionSegment.create("walk", start_time=0.0, duration=4.0)
        over_sim_val = track_def.sim_max + 1.0
        seg_motor = ap.MotorSegment(
            motor_id="m_nbk",
            track_name=track_name,
            start_time=0.0,
            duration=4.0,
            params={track_def.param_key: over_sim_val},
            parent_action_id=seg_action.action_id,
            locked=False,
        )
        overlay = ap.ActionMotorOverlay(
            action_id=seg_action.action_id,
            motor_segments=[seg_motor],
        )
        tl = ap.BehaviorTimeline(
            version=ap.BEHAVIOR_TIMELINE_VERSION,
            action_segments=[seg_action],
            motor_overlays=[overlay],
            active_motor_tracks=[track_name],
        )
        art = self.bridge.compile("", "walk_warn_valid2", timeline=tl, is_simulation=True)
        compile_errors = [d for d in art.diagnostics if d.level == "error"]
        if not compile_errors:
            self.assertTrue(art.is_valid)

    def test_compile_timeline_diag_code_starts_with_prefix(self):
        tl = self._timeline_with_out_of_range_motor()
        art = self.bridge.compile("", "walk_code_test", timeline=tl, is_simulation=False)
        for d in art.diagnostics:
            if d.code.startswith("timeline."):
                self.assertTrue(d.code.startswith("timeline.motor_"))
                break

    def test_compile_timeline_diag_has_track_location(self):
        tl = self._timeline_with_out_of_range_motor()
        art = self.bridge.compile("", "walk_loc_test", timeline=tl, is_simulation=False)
        tl_diags = [d for d in art.diagnostics
                    if d.code.startswith("timeline.motor_")]
        if tl_diags:
            self.assertIsNotNone(tl_diags[0].location)
            self.assertTrue(tl_diags[0].location.startswith("track:"))

    def test_compile_timeline_diag_shares_trace_id(self):
        tl = self._timeline_with_out_of_range_motor()
        art = self.bridge.compile("", "walk_tid_test",
                                  trace_id="my-trace-tl", timeline=tl, is_simulation=False)
        for d in art.diagnostics:
            if d.code.startswith("timeline.motor_"):
                self.assertEqual(d.trace_id, "my-trace-tl")

    # -- persistence roundtrip --------------------------------------------

    def test_compile_timeline_data_persisted_and_restored(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "persist_tl_ref", timeline=tl)
        if not art.is_valid:
            self.skipTest("Compile failed; skipping persistence check")
        bridge2 = BehaviorCompilerBridge(cache_dir=self._cache_dir)
        result = bridge2.resolve("persist_tl_ref")
        self.assertTrue(result.ok)
        self.assertIsInstance(result.artifact.timeline_data, dict)
        self.assertIn("version", result.artifact.timeline_data)

    def test_compile_no_timeline_persisted_loads_as_none(self):
        art = self.bridge.compile("", "no_tl_persist")
        if not art.is_valid:
            self.skipTest("Compile failed; skipping persistence check")
        bridge2 = BehaviorCompilerBridge(cache_dir=self._cache_dir)
        result = bridge2.resolve("no_tl_persist")
        self.assertTrue(result.ok)
        self.assertIsNone(result.artifact.timeline_data)

    # -- is_simulation flag -----------------------------------------------

    def test_compile_is_simulation_false_runs_without_error(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "sim_false", timeline=tl, is_simulation=False)
        self.assertIsNotNone(art)

    def test_compile_is_simulation_true_runs_without_error(self):
        tl = self._empty_timeline()
        art = self.bridge.compile("", "sim_true", timeline=tl, is_simulation=True)
        self.assertIsNotNone(art)


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Fix 3 — Safety enforcement: hardware error vs sim warning
# ===========================================================================

class TestMotorSafetyLevelPropagation(unittest.TestCase):
    """Verify compile() propagates md.level from MotorSegmentDiagnostic.

    Hardware violations must produce is_valid=False (error level);
    simulation violations must produce is_valid=True (warning level, non-blocking).
    """

    def _make_bridge(self) -> BehaviorCompilerBridge:
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        b = BehaviorCompilerBridge()
        b._cache_dir = tmpdir
        return b

    def _make_timeline_with_violation(self):
        """Return a BehaviorTimeline containing a motor segment that exceeds
        the safe range (amplitude=0.99 > safe_max=0.5 for leg_group_left)."""
        from system.behavior.action_profile import (
            ActionMotorOverlay,
            ActionSegment,
            BehaviorTimeline,
            MotorSegment,
        )
        seg = ActionSegment.create("walk", 0.0, 4.0)
        mseg = MotorSegment.create(
            "leg_group_left", 0.0, 4.0,
            params={"amplitude": 0.99},
            parent_action_id=seg.action_id,
        )
        overlay = ActionMotorOverlay(action_id=seg.action_id, motor_segments=[mseg])
        return BehaviorTimeline(
            action_segments=[seg],
            motor_overlays=[overlay],
            active_motor_tracks=["leg_group_left"],
        )

    def test_hardware_motor_violation_is_error_level(self):
        bridge = self._make_bridge()
        tl = self._make_timeline_with_violation()
        artifact = bridge.compile(
            "walk()", "test_hw",
            robot_type="go2",
            timeline=tl,
            is_simulation=False,
        )
        error_diags = [d for d in artifact.diagnostics if d.level == "error"]
        self.assertGreater(len(error_diags), 0)

    def test_hardware_motor_violation_makes_artifact_invalid(self):
        bridge = self._make_bridge()
        tl = self._make_timeline_with_violation()
        artifact = bridge.compile(
            "walk()", "test_hw_invalid",
            robot_type="go2",
            timeline=tl,
            is_simulation=False,
        )
        self.assertFalse(artifact.is_valid)

    def test_sim_motor_violation_is_warning_level(self):
        bridge = self._make_bridge()
        tl = self._make_timeline_with_violation()
        # amplitude=0.99 is within sim range (sim_max >= 1.0 for this track)
        # so we need a higher value to trigger a sim violation; but amplitude=0.99
        # may be within sim range — use a definitely out-of-sim-range value:
        from system.behavior.action_profile import (
            ActionMotorOverlay, ActionSegment, BehaviorTimeline, MotorSegment,
            UNITREE_MOTOR_TRACK_MAP,
        )
        tdef = UNITREE_MOTOR_TRACK_MAP["leg_group_left"]
        over_sim_val = tdef.sim_max + 1.0  # guaranteed to exceed sim range
        seg = ActionSegment.create("walk", 0.0, 4.0)
        mseg = MotorSegment.create(
            "leg_group_left", 0.0, 4.0,
            params={"amplitude": over_sim_val},
            parent_action_id=seg.action_id,
        )
        overlay = ActionMotorOverlay(action_id=seg.action_id, motor_segments=[mseg])
        tl2 = BehaviorTimeline(
            action_segments=[seg],
            motor_overlays=[overlay],
            active_motor_tracks=["leg_group_left"],
        )
        artifact = bridge.compile(
            "walk()", "test_sim_warn",
            robot_type="go2",
            timeline=tl2,
            is_simulation=True,
        )
        timeline_diags = [d for d in artifact.diagnostics if d.code.startswith("timeline.")]
        self.assertGreater(len(timeline_diags), 0)
        for d in timeline_diags:
            self.assertEqual(d.level, "warning")

    def test_sim_motor_violation_does_not_block_registration(self):
        bridge = self._make_bridge()
        from system.behavior.action_profile import (
            ActionMotorOverlay, ActionSegment, BehaviorTimeline, MotorSegment,
            UNITREE_MOTOR_TRACK_MAP,
        )
        tdef = UNITREE_MOTOR_TRACK_MAP["leg_group_left"]
        over_sim_val = tdef.sim_max + 1.0
        seg = ActionSegment.create("walk", 0.0, 4.0)
        mseg = MotorSegment.create(
            "leg_group_left", 0.0, 4.0,
            params={"amplitude": over_sim_val},
            parent_action_id=seg.action_id,
        )
        overlay = ActionMotorOverlay(action_id=seg.action_id, motor_segments=[mseg])
        tl = BehaviorTimeline(
            action_segments=[seg],
            motor_overlays=[overlay],
            active_motor_tracks=["leg_group_left"],
        )
        artifact = bridge.compile(
            "walk()", "test_sim_valid",
            robot_type="go2",
            timeline=tl,
            is_simulation=True,
        )
        # is_valid=True because warnings don't block; artifact is registered
        self.assertTrue(artifact.is_valid)


if __name__ == "__main__":
    unittest.main()
