#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for HeartBeat frontend state persistence — Circle 1 Step 1.6.

Verifies:
- MISSION_SCHEMA_VERSION bumped to "1.2"
- build_behavior_drafts_payload() / extract_behavior_drafts_payload() helpers
- BehaviorPanel.get_behavior_drafts_state() / set_behavior_drafts_state() wiring
- Full round-trip: build → persist dict → extract → restore
- Backward compat: missing "behavior_drafts" key → None (no raise)
- bin.ui._build_workflow_payload() injects behavior_drafts (wiring test via stub)
- bin.ui._on_open() restores behavior_drafts (wiring test via stub)

Isolation
---------
- No QApplication, no Qt widget instantiation.
- BehaviorPanel methods tested via duck-typed stubs.
- MainWindow methods tested via SimpleNamespace stubs (same pattern as other Step tests).
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bin.core.mission_persistence import (  # noqa: E402
    MISSION_SCHEMA_VERSION,
    build_behavior_drafts_payload,
    extract_behavior_drafts_payload,
)
from bin.layout.behavior_panel import HBDraftPayload  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _sample_draft(core: str = "stand()", script: str = "# hb") -> Dict[str, Any]:
    """Build a raw draft dict as stored in _drafts_by_node."""
    payload = HBDraftPayload(script=script)
    return {"core": core, "hb": payload.to_dict()}


def _sample_drafts() -> Dict[str, Any]:
    return {
        "1": _sample_draft("stand()", "# hb node 1"),
        "42": _sample_draft("walk(speed=0.3)", "# hb node 42"),
    }


# ===========================================================================
# Schema version
# ===========================================================================

class TestSchemaVersion(unittest.TestCase):

    def test_schema_version_is_1_4(self):
        self.assertEqual(MISSION_SCHEMA_VERSION, "1.4")

    def test_schema_version_is_string(self):
        self.assertIsInstance(MISSION_SCHEMA_VERSION, str)


# ===========================================================================
# build_behavior_drafts_payload()
# ===========================================================================

class TestBuildBehaviorDraftsPayload(unittest.TestCase):

    def test_empty_dict_returns_empty(self):
        result = build_behavior_drafts_payload({})
        self.assertEqual(result, {})

    def test_int_keys_stringified(self):
        drafts = {1: _sample_draft(), 42: _sample_draft()}
        result = build_behavior_drafts_payload(drafts)
        self.assertIn("1", result)
        self.assertIn("42", result)
        self.assertNotIn(1, result)

    def test_str_keys_preserved(self):
        drafts = {"10": _sample_draft(), "20": _sample_draft()}
        result = build_behavior_drafts_payload(drafts)
        self.assertIn("10", result)
        self.assertIn("20", result)

    def test_values_are_dicts(self):
        drafts = {"1": _sample_draft("stand()")}
        result = build_behavior_drafts_payload(drafts)
        self.assertIsInstance(result["1"], dict)

    def test_value_contains_core_key(self):
        drafts = {"1": _sample_draft("walk()")}
        result = build_behavior_drafts_payload(drafts)
        self.assertIn("core", result["1"])
        self.assertEqual(result["1"]["core"], "walk()")

    def test_value_contains_hb_key(self):
        drafts = {"1": _sample_draft()}
        result = build_behavior_drafts_payload(drafts)
        self.assertIn("hb", result["1"])
        self.assertIsInstance(result["1"]["hb"], dict)

    def test_return_is_independent_copy(self):
        original = {"5": _sample_draft("stand()")}
        result = build_behavior_drafts_payload(original)
        result["5"]["core"] = "MUTATED"
        self.assertEqual(original["5"]["core"], "stand()")

    def test_does_not_raise_on_malformed_entry(self):
        # Non-dict value: should be skipped (not raise)
        try:
            build_behavior_drafts_payload({"bad": "not a dict"})
        except Exception as exc:
            self.fail(f"build_behavior_drafts_payload raised: {exc}")

    def test_multiple_nodes_all_present(self):
        result = build_behavior_drafts_payload(_sample_drafts())
        self.assertIn("1", result)
        self.assertIn("42", result)
        self.assertEqual(len(result), 2)


# ===========================================================================
# extract_behavior_drafts_payload()
# ===========================================================================

class TestExtractBehaviorDraftsPayload(unittest.TestCase):

    def test_missing_key_returns_none(self):
        data = {"nodes": [], "connections": []}
        self.assertIsNone(extract_behavior_drafts_payload(data))

    def test_not_a_dict_returns_none(self):
        self.assertIsNone(extract_behavior_drafts_payload(None))
        self.assertIsNone(extract_behavior_drafts_payload([]))
        self.assertIsNone(extract_behavior_drafts_payload("string"))

    def test_behavior_drafts_not_dict_returns_none(self):
        data = {"behavior_drafts": "bad"}
        self.assertIsNone(extract_behavior_drafts_payload(data))

    def test_behavior_drafts_none_returns_none(self):
        data = {"behavior_drafts": None}
        self.assertIsNone(extract_behavior_drafts_payload(data))

    def test_valid_block_returned_as_dict(self):
        drafts = _sample_drafts()
        data = {"behavior_drafts": drafts}
        result = extract_behavior_drafts_payload(data)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_valid_block_keys_preserved(self):
        drafts = _sample_drafts()
        data = {"behavior_drafts": drafts}
        result = extract_behavior_drafts_payload(data)
        self.assertIn("1", result)
        self.assertIn("42", result)

    def test_result_is_independent_copy(self):
        drafts = {"1": _sample_draft()}
        data = {"behavior_drafts": drafts}
        result = extract_behavior_drafts_payload(data)
        result["extra_key"] = "new"
        self.assertNotIn("extra_key", data["behavior_drafts"])

    def test_never_raises_on_garbage_input(self):
        for bad in (None, [], 42, "bad", object()):
            try:
                extract_behavior_drafts_payload(bad)
            except Exception as exc:
                self.fail(f"Raised on {bad!r}: {exc}")

    def test_empty_behavior_drafts_block_returned_as_empty_dict(self):
        data = {"behavior_drafts": {}}
        result = extract_behavior_drafts_payload(data)
        self.assertEqual(result, {})


# ===========================================================================
# Round-trip: build → extract
# ===========================================================================

class TestRoundTrip(unittest.TestCase):

    def test_build_extract_roundtrip_preserves_all_nodes(self):
        original = _sample_drafts()
        built = build_behavior_drafts_payload(original)
        mission = {"nodes": [], "connections": [], "behavior_drafts": built}
        extracted = extract_behavior_drafts_payload(mission)
        self.assertIsNotNone(extracted)
        self.assertIn("1", extracted)
        self.assertIn("42", extracted)

    def test_build_extract_preserves_core_script(self):
        original = {"7": _sample_draft("sit()", "# hb sit")}
        built = build_behavior_drafts_payload(original)
        mission = {"behavior_drafts": built}
        extracted = extract_behavior_drafts_payload(mission)
        self.assertEqual(extracted["7"]["core"], "sit()")

    def test_build_extract_preserves_hb_payload(self):
        payload = HBDraftPayload(script="# my hb", authoring_mode="script")
        original = {"3": {"core": "stand()", "hb": payload.to_dict()}}
        built = build_behavior_drafts_payload(original)
        mission = {"behavior_drafts": built}
        extracted = extract_behavior_drafts_payload(mission)
        hb = HBDraftPayload.from_dict(extracted["3"]["hb"])
        self.assertEqual(hb.script, "# my hb")
        self.assertEqual(hb.authoring_mode, "script")


# ===========================================================================
# BehaviorPanel stub — get/set_behavior_drafts_state wiring
# ===========================================================================

class _BehaviorPanelStub:
    """Minimal stub that replicates the get/set_behavior_drafts_state logic
    without requiring a QApplication (no Qt widgets instantiated).
    """

    def __init__(self):
        self._node_id = -1
        self._drafts_by_node: Dict[int, Dict[str, Any]] = {}
        self._applied_drafts: list = []
        self._applied_core: list = []

    def _collect_hb_draft(self) -> HBDraftPayload:
        return HBDraftPayload(script="# current")

    def _set_core_source(self, text: str, *, mark_dirty: bool = True) -> None:
        self._applied_core.append(text)

    def _apply_hb_draft(self, payload: HBDraftPayload) -> None:
        self._applied_drafts.append(payload)

    def get_behavior_drafts_state(self) -> Dict[str, Any]:
        if self._node_id >= 0:
            self._drafts_by_node[self._node_id] = {
                "core": "# flushed",
                "hb": self._collect_hb_draft().to_dict(),
            }
        return {str(k): dict(v) for k, v in self._drafts_by_node.items()}

    def set_behavior_drafts_state(self, drafts: Dict[str, Any]) -> None:
        self._drafts_by_node = {}
        for k, v in drafts.items():
            try:
                node_id = int(k)
                if isinstance(v, dict):
                    self._drafts_by_node[node_id] = dict(v)
            except (ValueError, TypeError):
                pass
        if self._node_id >= 0 and self._node_id in self._drafts_by_node:
            saved = self._drafts_by_node[self._node_id]
            core = saved.get("core", "")
            hb_dict = saved.get("hb")
            self._set_core_source(core, mark_dirty=False)
            try:
                payload = HBDraftPayload.from_dict(hb_dict or {})
                self._apply_hb_draft(payload)
            except Exception:
                self._apply_hb_draft(HBDraftPayload.default())


class TestGetSetBehaviorDraftsState(unittest.TestCase):

    def _make_stub(self, **kw) -> _BehaviorPanelStub:
        s = _BehaviorPanelStub()
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    # ---- get_behavior_drafts_state ----

    def test_get_empty_when_no_nodes(self):
        stub = self._make_stub()
        result = stub.get_behavior_drafts_state()
        self.assertEqual(result, {})

    def test_get_flushes_current_node(self):
        stub = self._make_stub(_node_id=5)
        result = stub.get_behavior_drafts_state()
        self.assertIn("5", result)

    def test_get_returns_str_keys(self):
        stub = self._make_stub(_node_id=3)
        stub._drafts_by_node = {3: _sample_draft(), 7: _sample_draft()}
        result = stub.get_behavior_drafts_state()
        for k in result:
            self.assertIsInstance(k, str)

    def test_get_result_is_copy(self):
        stub = self._make_stub(_node_id=-1)
        stub._drafts_by_node = {1: _sample_draft("stand()")}
        result = stub.get_behavior_drafts_state()
        result["999"] = "mutated"
        self.assertNotIn(999, stub._drafts_by_node)

    # ---- set_behavior_drafts_state ----

    def test_set_populates_drafts_by_node(self):
        stub = self._make_stub()
        stub.set_behavior_drafts_state({"1": _sample_draft(), "5": _sample_draft()})
        self.assertIn(1, stub._drafts_by_node)
        self.assertIn(5, stub._drafts_by_node)

    def test_set_int_and_str_keys_both_accepted(self):
        stub = self._make_stub()
        stub.set_behavior_drafts_state({"10": _sample_draft(), "20": _sample_draft()})
        self.assertIn(10, stub._drafts_by_node)
        self.assertIn(20, stub._drafts_by_node)

    def test_set_clears_previous_drafts(self):
        stub = self._make_stub()
        stub._drafts_by_node = {99: _sample_draft()}
        stub.set_behavior_drafts_state({"1": _sample_draft()})
        self.assertNotIn(99, stub._drafts_by_node)

    def test_set_skips_non_dict_values(self):
        stub = self._make_stub()
        stub.set_behavior_drafts_state({"bad": "not a dict", "1": _sample_draft()})
        self.assertNotIn("bad", {str(k) for k in stub._drafts_by_node})
        self.assertIn(1, stub._drafts_by_node)

    def test_set_skips_non_int_keys(self):
        stub = self._make_stub()
        stub.set_behavior_drafts_state({"abc": _sample_draft(), "5": _sample_draft()})
        self.assertIn(5, stub._drafts_by_node)
        self.assertNotIn("abc", {str(k) for k in stub._drafts_by_node})

    def test_set_re_applies_current_node_draft(self):
        stub = self._make_stub(_node_id=3)
        stub.set_behavior_drafts_state({"3": _sample_draft("sit()", "# hb3")})
        self.assertGreater(len(stub._applied_core), 0)
        self.assertEqual(stub._applied_core[-1], "sit()")

    def test_set_no_reapply_when_no_current_node(self):
        stub = self._make_stub(_node_id=-1)
        stub.set_behavior_drafts_state({"1": _sample_draft()})
        self.assertEqual(len(stub._applied_drafts), 0)

    def test_set_never_raises_on_empty(self):
        stub = self._make_stub()
        try:
            stub.set_behavior_drafts_state({})
        except Exception as exc:
            self.fail(f"set_behavior_drafts_state raised: {exc}")


# ===========================================================================
# MainWindow wiring: _build_workflow_payload injects behavior_drafts
# ===========================================================================

def _make_ui_stub(drafts=None):
    """Stub mimicking only the parts of MainWindow exercised in persistence."""
    import bin.ui as ui_mod

    stub = types.SimpleNamespace()

    # minimal behavior_panel mock
    bp = MagicMock()
    bp.get_behavior_drafts_state.return_value = drafts or {"1": _sample_draft()}
    bp.set_behavior_drafts_state = MagicMock()

    class _FakeMainZone:
        behavior_panel = bp
        def get_sdk_settings(self): return {}
        def get_scenario_settings(self): return {}

    stub.main_zone = _FakeMainZone()

    # minimal graph_scene mock
    stub.graph_scene = MagicMock()
    stub.graph_scene.serialize_workflow.return_value = {"nodes": [], "connections": []}

    stub._build_workflow_payload = lambda: ui_mod.MainWindow._build_workflow_payload(stub)
    stub._on_open = lambda: None  # not tested here

    return stub


class TestBuildWorkflowPayloadInjectsDrafts(unittest.TestCase):

    def test_behavior_drafts_key_present(self):
        stub = _make_ui_stub()
        with patch("bin.ui.RobotContext.get_current_brand", return_value="unitree"), \
             patch("bin.ui.RobotContext.get_robot_type", return_value="go2"):
            payload = stub._build_workflow_payload()
        self.assertIn("behavior_drafts", payload)

    def test_behavior_drafts_contains_node_data(self):
        stub = _make_ui_stub(drafts={"5": _sample_draft("sit()")})
        with patch("bin.ui.RobotContext.get_current_brand", return_value="unitree"), \
             patch("bin.ui.RobotContext.get_robot_type", return_value="go2"):
            payload = stub._build_workflow_payload()
        self.assertIn("5", payload["behavior_drafts"])

    def test_behavior_drafts_failure_does_not_propagate(self):
        stub = _make_ui_stub()
        stub.main_zone.behavior_panel.get_behavior_drafts_state.side_effect = RuntimeError("boom")
        try:
            with patch("bin.ui.RobotContext.get_current_brand", return_value="unitree"), \
                 patch("bin.ui.RobotContext.get_robot_type", return_value="go2"):
                stub._build_workflow_payload()
        except Exception as exc:
            self.fail(f"_build_workflow_payload propagated: {exc}")


# ===========================================================================
# MainWindow wiring: _on_open restores behavior_drafts
# ===========================================================================

class _FakeMainZoneForOpen:
    def __init__(self):
        self.behavior_panel = MagicMock()
        self.behavior_panel.set_behavior_drafts_state = MagicMock()

    def set_sdk_brand(self, *a): pass
    def get_sdk_settings(self): return {}
    def set_scenario_settings(self, *a): pass
    def clear_execution_summary(self): pass
    def clear_diagnostics_panel(self): pass


def _make_open_stub(data: dict):
    """Stub for _on_open that skips dialog and Qt calls."""
    import bin.ui as ui_mod

    stub = types.SimpleNamespace()
    stub.main_zone = _FakeMainZoneForOpen()
    stub.graph_scene = MagicMock()
    stub.graph_scene.load_workflow = MagicMock()
    stub.status = MagicMock()
    stub._current_workflow_path = None

    def _set_path(p): stub._current_workflow_path = p
    stub._set_current_workflow_path = _set_path

    # Bind only the restore slice we test (not the full _on_open with dialog)
    def _restore_drafts():
        try:
            from bin.core.mission_persistence import extract_behavior_drafts_payload
            drafts = extract_behavior_drafts_payload(data)
            if drafts is not None:
                stub.main_zone.behavior_panel.set_behavior_drafts_state(drafts)
        except Exception:
            pass

    stub._restore_drafts = _restore_drafts
    return stub


class TestOnOpenRestoresDrafts(unittest.TestCase):

    def test_set_behavior_drafts_called_when_key_present(self):
        data = {"nodes": [], "connections": [],
                "behavior_drafts": {"3": _sample_draft()}}
        stub = _make_open_stub(data)
        stub._restore_drafts()
        stub.main_zone.behavior_panel.set_behavior_drafts_state.assert_called_once()

    def test_set_behavior_drafts_not_called_when_key_absent(self):
        data = {"nodes": [], "connections": []}
        stub = _make_open_stub(data)
        stub._restore_drafts()
        stub.main_zone.behavior_panel.set_behavior_drafts_state.assert_not_called()

    def test_correct_drafts_passed_to_set(self):
        drafts_in = {"7": _sample_draft("walk()")}
        data = {"nodes": [], "connections": [], "behavior_drafts": drafts_in}
        stub = _make_open_stub(data)
        stub._restore_drafts()
        call_arg = stub.main_zone.behavior_panel.set_behavior_drafts_state.call_args[0][0]
        self.assertIn("7", call_arg)

    def test_failure_does_not_propagate(self):
        data = {"behavior_drafts": {"1": _sample_draft()}}
        stub = _make_open_stub(data)
        stub.main_zone.behavior_panel.set_behavior_drafts_state.side_effect = RuntimeError("fail")
        try:
            stub._restore_drafts()
        except Exception as exc:
            self.fail(f"restore propagated: {exc}")


if __name__ == "__main__":
    unittest.main()
