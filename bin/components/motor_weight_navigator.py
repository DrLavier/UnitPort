#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor Weight Navigator widget — Step 2.

A read-optimized hierarchical inspector that shows:
  1. Body regions
  2. Motors under each region
  3. Which movements call each motor
  4. For each motor param: source status (Constant / External / Unresolved)
     with color-coded badges and values

This widget is the replacement for the HeartbeatCanvas placeholder.
It is intentionally NOT an authoring canvas — interaction is minimal
(expand/collapse, no drag-drop, no node editing).

Public API
----------
MotorWeightNavigator(parent=None)
    .refresh(regions: List[NavRegion]) -> None
    .refresh_from_timeline_and_io(timeline, io_mappings, available_tracks) -> None
    .set_placeholder(text: str) -> None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from PySide6.QtWidgets import (
        QTreeWidget,
        QTreeWidgetItem,
        QAbstractItemView,
        QHeaderView,
        QLabel,
        QVBoxLayout,
        QWidget,
    )
    from PySide6.QtGui import QBrush, QColor, QFont
    from PySide6.QtCore import Qt
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    # Fallback stubs — allow import in test environments without PySide6
    class QWidget: pass
    class QTreeWidget(QWidget): pass


# Source badge colors — aligned with _BADGE_COLOR_MAP in behavior_panel.py
_SOURCE_COLOR: dict = {
    "constant":   "#4caf50",   # green — authored constant value
    "external":   "#ff9800",   # amber — driven by upstream Script/protocol
    "unresolved": "#f44336",   # red   — param exists but cannot resolve
}
_REGION_FG       = "#e0e0e0"
_MOTOR_FG        = "#b0bec5"
_MOTOR_UNUSED_FG = "#78909c"  # blue-grey — motor in topology, using SDK default
_MOVEMENT_FG     = "#90a4ae"
_ISSUE_FG        = "#f44336"
_DEFAULT_COLOR   = "#78909c"  # blue-grey — "Default" badge (SDK-controlled motor)


if _QT_AVAILABLE:

    class MotorWeightNavigator(QTreeWidget):
        """Read-only hierarchical motor weight inspector.

        Tree columns
        ------------
        0 — Name  (region / motor / movement / param label)
        1 — Source badge  (Constant | External | Unresolved | Unused)
        2 — Value  (numeric for constants; "← signal" for externals; "—" otherwise)
        3 — Spacer (always empty; stretches to keep layout clean)
        """

        COLUMN_NAME   = 0
        COLUMN_SOURCE = 1
        COLUMN_VALUE  = 2
        COLUMN_SPACER = 3

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.setColumnCount(4)
            self.setHeaderLabels(["Motor / Movement", "Source", "Value", ""])
            self.setRootIsDecorated(True)
            self.setAlternatingRowColors(False)
            self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.setUniformRowHeights(True)
            self.setExpandsOnDoubleClick(True)
            self.setAnimated(False)

            # Column width hints
            hdr = self.header()
            hdr.setStretchLastSection(False)
            hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

            self._placeholder_text = "No timeline loaded — open a behavior node to inspect motor routing."
            self._show_placeholder()

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def refresh(self, regions: "List[Any]") -> None:  # List[NavRegion]
            """Rebuild the tree from a NavRegion list.

            Canonical mode  : region → limb → motor/joint → movement → param
            Degraded mode   : region → motor → movement → param  (flat, no limb tier)

            Clears and repopulates on every call (simple, correct, cheap
            for the expected region/motor counts).
            """
            self.clear()
            if not regions:
                self._show_placeholder()
                return

            for region in regions:
                region_item = self._make_region_item(region)
                self.addTopLevelItem(region_item)
                region_item.setExpanded(True)

                if getattr(region, "is_degraded", False):
                    # Degraded notice: no children, just the label row
                    continue

                if region.limbs:
                    # CANONICAL path — render limb tier
                    for limb in region.limbs:
                        limb_item = self._make_limb_item(limb)
                        region_item.addChild(limb_item)
                        limb_item.setExpanded(True)
                        self._populate_motors(limb_item, limb.motors)
                else:
                    # DEGRADED path — flat motors directly under region
                    self._populate_motors(region_item, region.motors)

        def refresh_from_timeline_and_io(
            self,
            timeline: Optional[Any],
            io_mappings: Optional[list] = None,
            available_tracks: Optional[Dict[str, Any]] = None,
            brand: str = "",
            robot_type: str = "",
        ) -> None:
            """Convenience wrapper: build model then refresh."""
            from system.behavior.motor_weight_nav_model import build_navigator_model
            regions = build_navigator_model(
                timeline,
                io_mappings or [],
                available_tracks=available_tracks,
                brand=brand,
                robot_type=robot_type,
            )
            self.refresh(regions)

        def set_placeholder(self, text: str) -> None:
            self._placeholder_text = text

        # ------------------------------------------------------------------
        # Private render helpers
        # ------------------------------------------------------------------

        def _populate_motors(
            self, parent: QTreeWidgetItem, motors: "List[Any]"
        ) -> None:
            """Append NavMotor items (and their movement/param children) to parent."""
            for motor in motors:
                motor_item = self._make_motor_item(motor)
                parent.addChild(motor_item)
                if not getattr(motor, "is_unused", False):
                    motor_item.setExpanded(True)

                for movement in motor.movements:
                    movement_item = self._make_movement_item(movement)
                    motor_item.addChild(movement_item)
                    movement_item.setExpanded(True)

                    for param in movement.params:
                        param_item = self._make_param_item(param)
                        movement_item.addChild(param_item)

        # ------------------------------------------------------------------
        # Item builders
        # ------------------------------------------------------------------

        def _make_region_item(self, region: Any) -> QTreeWidgetItem:
            label = region.region_label
            is_degraded = getattr(region, "is_degraded", False)
            if is_degraded:
                label = f"~ {label}"
            elif region.has_issue:
                label = f"? {label}"
            item = QTreeWidgetItem([label, "", "", ""])
            f = item.font(self.COLUMN_NAME)
            f.setBold(True)
            if is_degraded:
                f.setItalic(True)
            item.setFont(self.COLUMN_NAME, f)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            return item

        def _make_limb_item(self, limb: Any) -> QTreeWidgetItem:
            """NavLimb row — body-part/limb header below the region."""
            label = f"  {limb.limb_label}"
            item = QTreeWidgetItem([label, "", "", ""])
            f = item.font(self.COLUMN_NAME)
            f.setBold(True)
            item.setFont(self.COLUMN_NAME, f)
            # Use the widget/theme default text color for navigator content.
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            return item

        def _make_motor_item(self, motor: Any) -> QTreeWidgetItem:
            # Prefer joint_name (canonical) -> track_label -> track_name
            display = getattr(motor, "display_label", None) or motor.track_label
            label = f"    {display}  ({motor.track_name})"
            is_unused = getattr(motor, "is_unused", False)

            if is_unused:
                item = QTreeWidgetItem([label, "Default", "-", ""])
                item.setForeground(self.COLUMN_SOURCE, QBrush(QColor(_DEFAULT_COLOR)))
            else:
                item = QTreeWidgetItem([label, "", "", ""])

            f = item.font(self.COLUMN_NAME)
            f.setItalic(True)
            item.setFont(self.COLUMN_NAME, f)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            return item

        def _make_movement_item(self, movement: Any) -> QTreeWidgetItem:
            label = f"    ▶ {movement.movement_name}"
            item = QTreeWidgetItem([label, "", "", ""])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            return item

        def _make_param_item(self, param: Any) -> QTreeWidgetItem:
            # Col 0: param key (indented)
            name_label = f"        {param.param_key}"
            # Col 1: source badge
            source_text = param.source_badge
            # Col 2: value
            value_text = param.display_value

            item = QTreeWidgetItem([name_label, source_text, value_text, ""])

            # Source badge color
            source_color = _SOURCE_COLOR.get(param.source, "#757575")
            item.setForeground(self.COLUMN_SOURCE, QBrush(QColor(source_color)))

            return item

        # ------------------------------------------------------------------
        # Placeholder
        # ------------------------------------------------------------------

        def _show_placeholder(self) -> None:
            item = QTreeWidgetItem([self._placeholder_text, "", "", ""])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setForeground(0, QBrush(QColor("#6b7280")))
            self.addTopLevelItem(item)

else:
    # Headless stub for environments without PySide6 (CI / pure-Python tests)
    class MotorWeightNavigator:  # type: ignore[no-redef]
        def __init__(self, parent=None):
            self._regions = []

        def refresh(self, regions):
            self._regions = list(regions)

        def refresh_from_timeline_and_io(
            self,
            timeline,
            io_mappings=None,
            available_tracks=None,
            brand: str = "",
            robot_type: str = "",
        ):
            from system.behavior.motor_weight_nav_model import build_navigator_model
            regions = build_navigator_model(
                timeline,
                io_mappings or [],
                available_tracks=available_tracks,
                brand=brand,
                robot_type=robot_type,
            )
            self.refresh(regions)

        def set_placeholder(self, text):
            pass
