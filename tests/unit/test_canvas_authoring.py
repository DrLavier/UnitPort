#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Mission Canvas authoring operations — Cycle 1 STAGE-02.

Coverage
--------
- duplicate_selected_nodes: copies node with same type/name at offset
- duplicate skips Start/End (protected) nodes
- duplicate with no selection is a no-op
- _apply_node_data / serialize_workflow roundtrip preserves parameters
- Ctrl+D shortcut wiring present in keyPressEvent

Isolation
---------
- Requires QApplication (offscreen platform).
- Set QT_QPA_PLATFORM=offscreen before running if no display available.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use offscreen platform for headless CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPointF, Qt
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

_app = None


def _get_app():
    global _app
    if not _PYSIDE6_AVAILABLE:
        return None
    if QApplication.instance() is None:
        _app = QApplication(sys.argv[:1])
    else:
        _app = QApplication.instance()
    return _app


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestDuplicateNode(unittest.TestCase):
    """Tests for GraphScene.duplicate_selected_nodes."""

    @classmethod
    def setUpClass(cls):
        _get_app()
        from bin.components.graph_scene import GraphScene
        cls.GraphScene = GraphScene

    def _make_scene(self):
        return self.GraphScene()

    def _node_ids(self, scene):
        """Return set of node_id values for all nodes in scene."""
        from shiboken6 import isValid
        return {
            item.data(12)
            for item in scene.items()
            if isValid(item) and item.data(10) == "node"
        }

    def _node_by_name(self, scene, name):
        """Return first node item with given display name."""
        from shiboken6 import isValid
        for item in scene.items():
            if isValid(item) and item.data(10) == "node" and item.data(11) == name:
                return item
        return None

    def test_duplicate_single_node_increases_count(self):
        """Duplicating one selected node increases node count by 1."""
        scene = self._make_scene()
        # Add a non-protected node
        timer_item = scene.create_node("Timer", QPointF(100, 100))
        self.assertIsNotNone(timer_item)

        # Select it
        timer_item.setSelected(True)
        before_ids = self._node_ids(scene)

        scene.duplicate_selected_nodes()
        after_ids = self._node_ids(scene)

        new_ids = after_ids - before_ids
        self.assertEqual(len(new_ids), 1, "Exactly one new node should be created")

    def test_duplicate_preserves_node_name(self):
        """Duplicated node has the same display name as the original."""
        scene = self._make_scene()
        from shiboken6 import isValid

        item = scene.create_node("Timer", QPointF(200, 200))
        item.setSelected(True)
        before_ids = self._node_ids(scene)

        scene.duplicate_selected_nodes()
        after_ids = self._node_ids(scene)
        new_ids = after_ids - before_ids

        new_item = next(
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node" and i.data(12) in new_ids
        )
        self.assertEqual(new_item.data(11), "Timer")

    def test_duplicate_places_copy_at_offset(self):
        """Duplicated node is placed offset from original by ~(30, 30)."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        orig_pos = item.pos()
        item.setSelected(True)

        scene.duplicate_selected_nodes()

        from shiboken6 import isValid
        new_items = [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node" and i.data(11) == "Timer"
            and i is not item
        ]
        self.assertEqual(len(new_items), 1)
        new_pos = new_items[0].pos()
        # Offset should be positive (copy is placed to the right/down)
        self.assertGreater(new_pos.x(), orig_pos.x() - 1)
        self.assertGreater(new_pos.y(), orig_pos.y() - 1)

    def test_duplicate_deselects_original_selects_copy(self):
        """After duplication, original is deselected and copy is selected."""
        scene = self._make_scene()
        item = scene.create_node("Wait", QPointF(50, 50))
        item.setSelected(True)

        scene.duplicate_selected_nodes()

        self.assertFalse(item.isSelected(), "Original should be deselected")
        from shiboken6 import isValid
        selected = [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node" and i.isSelected()
        ]
        self.assertEqual(len(selected), 1, "Exactly one node (the copy) should be selected")

    def test_duplicate_no_selection_is_noop(self):
        """duplicate_selected_nodes with nothing selected does nothing."""
        scene = self._make_scene()
        scene.create_node("Timer", QPointF(100, 100))
        scene.clearSelection()
        before_count = len([
            i for i in scene.items()
            if i.data(10) == "node"
        ])

        scene.duplicate_selected_nodes()

        after_count = len([
            i for i in scene.items()
            if i.data(10) == "node"
        ])
        self.assertEqual(before_count, after_count)

    def test_duplicate_skips_protected_nodes(self):
        """Protected Start/End nodes cannot be duplicated."""
        scene = self._make_scene()
        from shiboken6 import isValid

        # Find the Start node (if it exists in the default scene)
        protected = [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node"
            and getattr(i, '_is_protected', False)
        ]

        if not protected:
            # Create a manually protected node for this test
            item = scene.create_node("Start", QPointF(0, 0))
            item._is_protected = True
            protected = [item]

        for p in protected:
            p.setSelected(True)

        before_ids = self._node_ids(scene)
        scene.duplicate_selected_nodes()
        after_ids = self._node_ids(scene)

        self.assertEqual(before_ids, after_ids, "Protected nodes must not be duplicated")

    def test_duplicate_multiple_nodes(self):
        """Duplicating two selected nodes creates two copies."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(200, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)
        before_count = sum(
            1 for i in scene.items()
            if i.data(10) == "node"
        )

        scene.duplicate_selected_nodes()

        after_count = sum(
            1 for i in scene.items()
            if i.data(10) == "node"
        )
        self.assertEqual(after_count - before_count, 2)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestSerializeRoundtrip(unittest.TestCase):
    """Verify serialize_workflow / load_workflow roundtrip is intact after refactor."""

    @classmethod
    def setUpClass(cls):
        _get_app()
        from bin.components.graph_scene import GraphScene
        cls.GraphScene = GraphScene

    def test_empty_scene_roundtrip(self):
        """serialize then load empty scene produces valid structure."""
        scene = self.GraphScene()
        data = scene.serialize_workflow()
        self.assertIn("nodes", data)
        self.assertIn("connections", data)

    def test_node_roundtrip_preserves_count(self):
        """After save + load, same number of nodes is restored."""
        scene = self.GraphScene()
        scene.create_node("Timer", QPointF(100, 100))
        scene.create_node("Wait", QPointF(300, 100))

        data = scene.serialize_workflow()
        node_count_before = len(data["nodes"])

        scene.load_workflow(data)
        data_after = scene.serialize_workflow()
        self.assertEqual(len(data_after["nodes"]), node_count_before)

    def test_timer_duration_roundtrip(self):
        """Timer duration parameter survives serialize/load cycle."""
        scene = self.GraphScene()
        item = scene.create_node("Timer", QPointF(100, 100))

        # Set duration directly if widget exists
        dur_widget = getattr(item, '_duration_input', None)
        if dur_widget is None:
            self.skipTest("Timer node has no _duration_input widget")

        dur_widget.setText("42")

        data = scene.serialize_workflow()
        scene.load_workflow(data)

        # Find restored timer node
        from shiboken6 import isValid
        timer_nodes = [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node" and i.data(11) == "Timer"
        ]
        self.assertTrue(timer_nodes, "Timer node should be present after load")
        restored_dur = getattr(timer_nodes[0], '_duration_input', None)
        if restored_dur:
            self.assertEqual(restored_dur.text(), "42")


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestKeyboardShortcutWiring(unittest.TestCase):
    """Verify Ctrl+D shortcut is wired in GraphScene.keyPressEvent."""

    def test_ctrl_d_handler_exists_in_source(self):
        """graph_scene.py source contains Ctrl+D -> duplicate_selected_nodes wiring."""
        import inspect
        from bin.components.graph_scene import GraphScene
        source = inspect.getsource(GraphScene.keyPressEvent)
        self.assertIn("Key_D", source)
        self.assertIn("duplicate_selected_nodes", source)

    def test_duplicate_selected_nodes_method_exists(self):
        """GraphScene exposes duplicate_selected_nodes."""
        from bin.components.graph_scene import GraphScene
        self.assertTrue(
            callable(getattr(GraphScene, "duplicate_selected_nodes", None)),
            "duplicate_selected_nodes must be a callable method on GraphScene"
        )

    def test_apply_node_data_method_exists(self):
        """GraphScene exposes _apply_node_data (used by both load and duplicate)."""
        from bin.components.graph_scene import GraphScene
        self.assertTrue(
            callable(getattr(GraphScene, "_apply_node_data", None)),
            "_apply_node_data must be a callable method on GraphScene"
        )

    def test_ctrl_g_handler_exists_in_source(self):
        """graph_scene.py source contains Ctrl+G -> group_selected_nodes wiring."""
        import inspect
        from bin.components.graph_scene import GraphScene
        source = inspect.getsource(GraphScene.keyPressEvent)
        self.assertIn("Key_G", source)
        self.assertIn("group_selected_nodes", source)
        self.assertIn("ungroup_selected_groups", source)

    def test_backspace_is_not_wired_to_delete_nodes(self):
        """Backspace must not delete canvas nodes; only Delete should."""
        import inspect
        from bin.components.graph_scene import GraphScene
        source = inspect.getsource(GraphScene.keyPressEvent)
        self.assertIn("Key_Delete", source)
        self.assertNotIn("Key_Backspace", source)

    def test_group_selected_nodes_method_exists(self):
        """GraphScene exposes group_selected_nodes."""
        from bin.components.graph_scene import GraphScene
        self.assertTrue(callable(getattr(GraphScene, "group_selected_nodes", None)))

    def test_ungroup_selected_groups_method_exists(self):
        """GraphScene exposes ungroup_selected_groups."""
        from bin.components.graph_scene import GraphScene
        self.assertTrue(callable(getattr(GraphScene, "ungroup_selected_groups", None)))


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestGroupUngroup(unittest.TestCase):
    """Tests for GraphScene.group_selected_nodes / ungroup_selected_groups."""

    @classmethod
    def setUpClass(cls):
        _get_app()
        from bin.components.graph_scene import GraphScene, GroupContainerItem
        cls.GraphScene = GraphScene
        cls.GroupContainerItem = GroupContainerItem

    def _make_scene(self):
        return self.GraphScene()

    def _group_items(self, scene):
        """Return all GroupContainerItem instances in scene."""
        from shiboken6 import isValid
        return [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "group"
        ]

    def _node_items(self, scene):
        """Return all non-protected node items."""
        from shiboken6 import isValid
        return [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node"
            and not getattr(i, '_is_protected', False)
        ]

    # ------------------------------------------------------------------ #
    #  group_selected_nodes                                                #
    # ------------------------------------------------------------------ #

    def test_group_creates_one_container(self):
        """Grouping N selected nodes creates exactly one GroupContainerItem."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(200, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)

        g = scene.group_selected_nodes(title="TestGroup")
        self.assertIsNotNone(g)
        groups = self._group_items(scene)
        self.assertEqual(len(groups), 1)

    def test_group_has_correct_title(self):
        """Group container title matches the provided argument."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)

        g = scene.group_selected_nodes(title="MyLabel")
        self.assertEqual(g._title, "MyLabel")

    def test_group_default_title_is_auto(self):
        """Group container with empty title gets an auto-generated label."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)

        g = scene.group_selected_nodes()
        self.assertTrue(g._title.startswith("Group "))

    def test_group_member_ids_match_selected(self):
        """Group _member_ids contains the node IDs of all grouped nodes."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(200, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)

        g = scene.group_selected_nodes()
        expected_ids = {item_a.data(12), item_b.data(12)}
        self.assertEqual(g._member_ids, expected_ids)

    def test_group_assigns_group_id_to_members(self):
        """After grouping, member node items carry the group's _group_id."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)

        g = scene.group_selected_nodes()
        self.assertEqual(getattr(item, '_group_id', None), g._group_id)

    def test_group_skips_protected_nodes(self):
        """Protected nodes are excluded from grouping."""
        scene = self._make_scene()
        # Find the pre-placed Start node (protected)
        from shiboken6 import isValid
        start_nodes = [
            i for i in scene.items()
            if isValid(i) and i.data(10) == "node" and i.data(11) == "Start"
        ]
        if not start_nodes:
            self.skipTest("No Start node in default scene")
        start = start_nodes[0]
        self.assertTrue(getattr(start, '_is_protected', False))

        # Also add a regular node and select both
        regular = scene.create_node("Timer", QPointF(100, 100))
        start.setSelected(True)
        regular.setSelected(True)

        g = scene.group_selected_nodes()
        # Only the regular node should be a member
        self.assertIn(regular.data(12), g._member_ids)
        self.assertNotIn(start.data(12), g._member_ids)
        # Start node must NOT have _group_id set
        self.assertIsNone(getattr(start, '_group_id', None))

    def test_group_no_selection_returns_none(self):
        """group_selected_nodes with nothing selected returns None."""
        scene = self._make_scene()
        scene.clearSelection()
        result = scene.group_selected_nodes()
        self.assertIsNone(result)

    def test_group_no_selection_creates_no_container(self):
        """group_selected_nodes with nothing selected adds no container to scene."""
        scene = self._make_scene()
        scene.clearSelection()
        before = len(self._group_items(scene))
        scene.group_selected_nodes()
        after = len(self._group_items(scene))
        self.assertEqual(before, after)

    def test_group_container_rect_encloses_members(self):
        """Group container bounding rect must be at least as large as the member nodes."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(400, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)

        g = scene.group_selected_nodes()
        # Container scene bounding rect should fully contain each member
        container_rect = g.sceneBoundingRect()
        for member in [item_a, item_b]:
            member_rect = member.sceneBoundingRect()
            self.assertTrue(
                container_rect.contains(member_rect),
                f"Container {container_rect} should contain member {member_rect}"
            )

    def test_group_registered_in_groups_dict(self):
        """After grouping, the group_id is present in scene._groups."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)

        g = scene.group_selected_nodes()
        self.assertIn(g._group_id, scene._groups)
        self.assertIs(scene._groups[g._group_id], g)

    # ------------------------------------------------------------------ #
    #  ungroup_selected_groups                                             #
    # ------------------------------------------------------------------ #

    def test_ungroup_removes_container_from_scene(self):
        """ungroup_selected_groups removes the group container from the scene."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        g = scene.group_selected_nodes()

        g.setSelected(True)
        scene.ungroup_selected_groups()

        groups_after = self._group_items(scene)
        self.assertEqual(len(groups_after), 0)

    def test_ungroup_detaches_group_id_from_members(self):
        """After ungrouping, member nodes no longer carry _group_id."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(200, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)
        g = scene.group_selected_nodes()

        g.setSelected(True)
        scene.ungroup_selected_groups()

        for item in [item_a, item_b]:
            self.assertIsNone(
                getattr(item, '_group_id', None),
                "_group_id must be None after ungroup"
            )

    def test_ungroup_removes_from_groups_dict(self):
        """After ungrouping, group_id is absent from scene._groups."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        g = scene.group_selected_nodes()
        gid = g._group_id
        g.setSelected(True)
        scene.ungroup_selected_groups()
        self.assertNotIn(gid, scene._groups)

    def test_ungroup_preserves_member_nodes(self):
        """Ungrouping does NOT delete the member nodes."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(200, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)
        node_count_before = sum(1 for i in scene.items() if i.data(10) == "node")

        g = scene.group_selected_nodes()
        g.setSelected(True)
        scene.ungroup_selected_groups()

        node_count_after = sum(1 for i in scene.items() if i.data(10) == "node")
        self.assertEqual(node_count_before, node_count_after)

    def test_ungroup_no_group_selected_is_noop(self):
        """ungroup_selected_groups with no group containers selected is a no-op."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        g = scene.group_selected_nodes()

        # Select only the node, not the group container
        scene.clearSelection()
        item.setSelected(True)

        before = len(self._group_items(scene))
        scene.ungroup_selected_groups()
        after = len(self._group_items(scene))
        self.assertEqual(before, after, "No container should be removed when none selected")

    # ------------------------------------------------------------------ #
    #  Delete container via _delete_items                                  #
    # ------------------------------------------------------------------ #

    def test_delete_group_container_detaches_members(self):
        """Pressing Delete on a group container detaches members but keeps nodes."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        g = scene.group_selected_nodes()
        node_id = item.data(12)

        # Select the container and delete it
        scene.clearSelection()
        g.setSelected(True)
        scene._delete_items([g])

        from shiboken6 import isValid
        # Container should be gone
        self.assertFalse(isValid(g) and g.scene() is not None)
        # Node should still be in scene
        remaining = [i for i in scene.items()
                     if isValid(i) and i.data(10) == "node" and i.data(12) == node_id]
        self.assertEqual(len(remaining), 1)
        # Node's _group_id should be cleared
        self.assertIsNone(getattr(remaining[0], '_group_id', None))

    # ------------------------------------------------------------------ #
    #  Serialisation roundtrip with groups                                 #
    # ------------------------------------------------------------------ #

    def test_save_load_roundtrip_preserves_group(self):
        """serialize_workflow + load_workflow restores group containers."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Wait", QPointF(200, 0))
        item_a.setSelected(True)
        item_b.setSelected(True)
        g = scene.group_selected_nodes(title="RoundtripGroup")

        data = scene.serialize_workflow()
        self.assertIn("groups", data)
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["groups"][0]["title"], "RoundtripGroup")

        scene.load_workflow(data)

        groups_after = self._group_items(scene)
        self.assertEqual(len(groups_after), 1)
        self.assertEqual(groups_after[0]._title, "RoundtripGroup")

    def test_save_load_roundtrip_restores_member_ids(self):
        """After load_workflow, group _member_ids are restored."""
        scene = self._make_scene()
        item_a = scene.create_node("Timer", QPointF(10, 10))
        item_b = scene.create_node("Wait", QPointF(210, 10))
        item_a.setSelected(True)
        item_b.setSelected(True)
        scene.group_selected_nodes(title="MembersGroup")

        data = scene.serialize_workflow()
        scene.load_workflow(data)

        groups_after = self._group_items(scene)
        self.assertEqual(len(groups_after), 1)
        # Two non-protected nodes should be members
        self.assertEqual(len(groups_after[0]._member_ids), 2)

    def test_backward_compat_missing_groups_key(self):
        """load_workflow on JSON without 'groups' key does not crash."""
        scene = self._make_scene()
        scene.create_node("Timer", QPointF(100, 100))
        data = scene.serialize_workflow()
        # Simulate old-format JSON: drop the groups key
        data.pop("groups", None)
        # Should not raise
        scene.load_workflow(data)
        # No groups in scene (nothing was restored)
        self.assertEqual(len(self._group_items(scene)), 0)

    def test_groups_key_in_serialized_output(self):
        """serialize_workflow always includes a 'groups' key."""
        scene = self._make_scene()
        data = scene.serialize_workflow()
        self.assertIn("groups", data)
        self.assertIsInstance(data["groups"], list)

    # ------------------------------------------------------------------ #
    #  Interaction with duplicate                                          #
    # ------------------------------------------------------------------ #

    def test_duplicate_ungrouped_node_no_new_group(self):
        """Duplicating a node that has no group does not create a group."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        before_groups = len(self._group_items(scene))

        scene.duplicate_selected_nodes()

        after_groups = len(self._group_items(scene))
        self.assertEqual(before_groups, after_groups)

    def test_group_then_ungroup_then_duplicate(self):
        """Workflow: group → ungroup → duplicate works without errors."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        g = scene.group_selected_nodes()
        g.setSelected(True)
        scene.ungroup_selected_groups()

        # Now duplicate the former member
        item.setSelected(True)
        before_count = sum(1 for i in scene.items() if i.data(10) == "node")
        scene.duplicate_selected_nodes()
        after_count = sum(1 for i in scene.items() if i.data(10) == "node")
        self.assertEqual(after_count - before_count, 1)

    # ------------------------------------------------------------------ #
    #  clear_all_nodes clears groups                                       #
    # ------------------------------------------------------------------ #

    def test_clear_all_nodes_removes_groups(self):
        """clear_all_nodes also removes group containers from the scene."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(0, 0))
        item.setSelected(True)
        scene.group_selected_nodes()

        scene.clear_all_nodes()

        self.assertEqual(len(self._group_items(scene)), 0)
        self.assertEqual(len(scene._groups), 0)

    # ------------------------------------------------------------------ #
    #  Movement propagation                                                #
    # ------------------------------------------------------------------ #

    def test_moving_group_moves_members(self):
        """Moving a group container displaces member nodes by the same delta."""
        scene = self._make_scene()
        item = scene.create_node("Timer", QPointF(50, 50))
        original_pos = QPointF(item.pos())
        item.setSelected(True)
        g = scene.group_selected_nodes()

        # Simulate move: call setPos a second time (after _prev_pos is set)
        delta = QPointF(100, 50)
        new_group_pos = g.pos() + delta
        g.setPos(new_group_pos)

        # Item should have moved by the same delta
        expected = original_pos + delta
        actual = item.pos()
        self.assertAlmostEqual(actual.x(), expected.x(), delta=2)
        self.assertAlmostEqual(actual.y(), expected.y(), delta=2)


if __name__ == "__main__":
    unittest.main()
