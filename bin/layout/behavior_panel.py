"""Behavior workspace with timeline-oriented sequence editing."""

from __future__ import annotations

import re
import uuid as _uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from system.behavior.timeline_view_model import TimelineSelection, SelectionLevel

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QPointF, QRectF, QEvent
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from system.behavior.action_profile import (
    ActionMotorOverlay,
    ActionSegment,
    BehaviorTimeline,
    MotorSegment,
    MotorTrackDef,
    UNITREE_ACTION_PROFILES,
    build_timeline_from_modules,
    validate_timeline,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
from system.behavior.timeline_migration import migrate_timeline
from system.behavior.action_registry import get_semantic_actions
from system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
from system.behavior.hb_channel import (
    HBCompileRequest,
    HBDiagnosticsSnapshot,
    HBDryRunRequest,
    HBMockChannel,
    HBRunRequest,
    IHBChannel,
)
from system.behavior.hb_node_catalog import HBNodeAvailability, HBNodeCatalog
from system.behavior.hb_compat_audit import (
    HBCompatReport,
    audit_catalog_compatibility,
    audit_action_compatibility,
)
from system.behavior.hb_display_state import (
    badge_for_event_state,
    badge_for_io_status,
    compute_event_summary,
    compute_io_summary,
    detect_io_conflicts,
)
from bin.core.localisation import tr
from bin.core.theme_manager import get_color
from bin.components.motor_weight_navigator import MotorWeightNavigator  # Step 2
from system.behavior.motor_param_source import (                         # Step 3
    STRUCTURAL_KEYS,
    get_param_source,
)

# Color token 閳?hex string map for status badges (Step 1.5)
_BADGE_COLOR_MAP: Dict[str, str] = {
    "success": "#4caf50",
    "error":   "#f44336",
    "warning": "#ff9800",
    "info":    "#2196f3",
    "neutral": "#757575",
}


@dataclass
class SequenceModule:
    kind: str  # movement | behavior
    name: str
    args: str = ""
    duration: float = 1.0


class BehaviorCompileWorker(QThread):
    """Execute BehaviorCompilerBridge.compile() on a background thread."""

    compile_done = Signal(object)

    def __init__(
        self,
        bridge: BehaviorCompilerBridge,
        source: str,
        behavior_ref: str,
        robot_type: str = "go2",
        timeline=None,
        is_simulation: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._bridge = bridge
        self._source = source
        self._behavior_ref = behavior_ref
        self._robot_type = robot_type
        self._timeline = timeline
        self._is_simulation = is_simulation

    def run(self) -> None:
        artifact = self._bridge.compile(
            self._source,
            self._behavior_ref,
            robot_type=self._robot_type,
            timeline=self._timeline,
            is_simulation=self._is_simulation,
        )
        self.compile_done.emit(artifact)


class _MotorParamEditDialog(QDialog):
    """Fix 2: minimal dialog to edit motor segment parameters."""

    def __init__(self, seg: "MotorSegment", track_def: Optional["MotorTrackDef"], parent=None):
        super().__init__(parent)
        self._seg = seg
        self._track_def = track_def
        self.setWindowTitle(f"Edit Motor Segment: {seg.motor_id}")
        self.setMinimumWidth(280)

        form = QFormLayout(self)
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}

        # Start time
        start_spin = QDoubleSpinBox()
        start_spin.setRange(0.0, 9999.0)
        start_spin.setDecimals(3)
        start_spin.setValue(seg.start_time)
        self._spinboxes["start_time"] = start_spin
        form.addRow("start_time (s):", start_spin)

        # Track-specific primary param
        if track_def is not None:
            pk = track_def.param_key
            spin = QDoubleSpinBox()
            lo = track_def.sim_min if track_def.sim_min is not None else -360.0
            hi = track_def.sim_max if track_def.sim_max is not None else 360.0
            spin.setRange(lo, hi)
            spin.setDecimals(4)
            spin.setValue(float(seg.params.get(pk, 0.0)))
            self._spinboxes[pk] = spin
            form.addRow(f"{pk}:", spin)
        else:
            for k, v in seg.params.items():
                spin = QDoubleSpinBox()
                spin.setRange(-9999.0, 9999.0)
                spin.setDecimals(4)
                try:
                    spin.setValue(float(v))
                except (TypeError, ValueError):
                    spin.setValue(0.0)
                self._spinboxes[k] = spin
                form.addRow(f"{k}:", spin)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self._apply_and_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _apply_and_accept(self) -> None:
        if "start_time" in self._spinboxes:
            self._seg.start_time = self._spinboxes["start_time"].value()
        for k, spin in self._spinboxes.items():
            if k == "start_time":
                continue
            self._seg.params[k] = spin.value()
        self.accept()


class TimelineView(QGraphicsView):
    """Video-editor-like timeline with ruler, primary track, and secondary track."""

    module_selected = Signal(int)
    motor_segment_selected = Signal(str, str)  # (motor_id, track_name), empty strings = none
    body_part_selected = Signal(str, str)      # (action_id, limb_id); "" = deselected/none
    module_reordered = Signal(int, int)
    delete_requested = Signal()
    timeline_edited = Signal()  # Fix 6: emitted after any motor seg mutation or track toggle
    playhead_moved = Signal(float)             # emitted when playhead position changes (seconds)
    _ZOOM_TICK_STEPS: ClassVar[List[float]] = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(self.renderHints())
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._modules: List[SequenceModule] = []
        self._secondary: List[SequenceModule] = []
        self._selected_index: int = -1
        self._sec_source_name: str = ""
        self._press_index: Optional[int] = None
        self._press_scene_x: float = 0.0
        self._press_scene_y: float = 0.0
        self._drag_started: bool = False
        self._drag_from_index: Optional[int] = None
        self._drag_grab_offset_x: float = 0.0
        self._drag_left_x: float = 0.0
        self._drag_insert_pos: Optional[int] = None
        self._drag_cursor_x: float = 0.0

        # Multi-track timeline state (Phase 1 redesign)
        self._behavior_timeline: Optional["BehaviorTimeline"] = None
        self._motor_track_names: List[str] = []   # ordered active motor tracks
        self._motor_track_expanded: Dict[str, bool] = {}  # track_name 閳?expanded
        self._motor_track_rows: Dict[str, Tuple[float, float, bool]] = {}
        # Step 5: body-part grouping state
        self._timeline_vm: Optional[Any] = None          # TimelineViewModel
        self._available_tracks: Dict[str, Any] = {}      # track_name 閳?MotorTrackDef
        self._body_part_expanded: Dict[str, bool] = {}   # limb_id 閳?expanded
        self._body_part_rows: Dict[str, Tuple[float, float]] = {}  # limb_id 閳?(y, h)
        # Selection state is now owned by _timeline_vm.selection (TimelineSelection).
        # Local _selected_* fields removed; all highlight reads go through the VM.

        # Fix 2: motor segment drag state
        self._motor_press_id: Optional[str] = None
        self._motor_press_track: Optional[str] = None
        self._motor_drag_started: bool = False
        self._motor_press_scene_x: float = 0.0
        self._motor_orig_start: float = 0.0

        # Timeline density baseline.
        self._base_scale_px_per_sec = 180.0
        self._base_tick_sec = 0.05
        # Init at the same visual zoom level as 0.01s tick (Ctrl+wheel equivalent).
        self._tick_sec = 0.01
        self._scale_px_per_sec = (
            self._base_scale_px_per_sec * (self._base_tick_sec / self._tick_sec)
        )
        self._ruler_h = 26
        self._track_h = 44
        self._main_y = self._ruler_h
        self._secondary_y = self._main_y + self._track_h
        self._scene_width = 24000.0
        self._content_height = self._secondary_y + self._track_h
        self._drag_start_threshold_px = 6.0

        self._main_track_bg = get_color("behavior_timeline_main_track_bg", "#1f1f1f")
        self._secondary_track_bg = get_color("behavior_timeline_secondary_track_bg", "#1b1b1b")

        # Playhead state
        self._playhead_time: float = 0.0

        self.horizontalScrollBar().valueChanged.connect(self._ensure_infinite_width)
        self._redraw()

    def apply_theme(self) -> None:
        self._main_track_bg = get_color("behavior_timeline_main_track_bg", "#1f1f1f")
        self._secondary_track_bg = get_color("behavior_timeline_secondary_track_bg", "#1b1b1b")
        self._redraw()

    def set_timeline(
        self,
        modules: List[SequenceModule],
        secondary: List[SequenceModule],
        selected_index: int,
        secondary_source_name: str = "",
    ) -> None:
        self._modules = list(modules)
        self._secondary = list(secondary)
        self._selected_index = selected_index
        self._sec_source_name = secondary_source_name

        # Clear multi-track state when legacy set_timeline is called
        self._behavior_timeline = None
        self._motor_track_names = []
        self._motor_track_expanded = {}
        self._motor_track_rows = {}
        self._clear_motor_selection(notify=True)

        total_duration = self._total_duration(self._modules)
        needed = max(24000.0, total_duration * self._scale_px_per_sec + 1600.0)
        if needed > self._scene_width:
            self._scene_width = needed
        self._redraw()

    def set_multi_track_timeline(
        self,
        timeline: "BehaviorTimeline",
        selected_index: int = -1,
        brand: str = "",
        robot_type: str = "",
        available_tracks: Optional[Dict[str, Any]] = None,
        selection: Optional[Any] = None,
    ) -> None:
        """Load a structured BehaviorTimeline for multi-track rendering.

        Populates both the legacy _modules (Action Track) and the structured
        motor sub-tracks.  Existing drag/reorder logic continues to operate
        on _modules for backward-compat edit operations.

        Parameters
        ----------
        timeline         : BehaviorTimeline containing action_segments and
                           motor_overlays.
        selected_index   : Index of the selected ActionSegment (-1 for none).
        brand            : Robot brand 閳?used to resolve canonical topology.
        robot_type       : Robot type 閳?used to resolve canonical topology.
        available_tracks : Full motor track catalog for label/color resolution.
        """
        self._behavior_timeline = timeline
        self._selected_index = selected_index
        self._available_tracks = available_tracks or {}

        # Sync legacy _modules from ActionSegments for backward-compat
        self._modules = [
            SequenceModule(
                kind=seg.kind,
                name=seg.name,
                args=", ".join(f"{k}={v}" for k, v in seg.params.items()),
                duration=seg.duration,
            )
            for seg in timeline.action_segments
        ]
        self._secondary = []
        self._sec_source_name = ""
        if selected_index >= 0:
            self._clear_motor_selection(notify=True)

        # Update active motor tracks; preserve expanded state across calls
        self._motor_track_names = list(timeline.active_motor_tracks)
        for tname in self._motor_track_names:
            if tname not in self._motor_track_expanded:
                self._motor_track_expanded[tname] = True  # expanded by default

        # Step 5: build canonical body-part view model when brand/robot_type known
        self._timeline_vm = None
        if brand or robot_type:
            try:
                from system.behavior.timeline_view_model import build_timeline_view_model
                self._timeline_vm = build_timeline_view_model(
                    timeline, brand=brand, robot_type=robot_type,
                    available_tracks=self._available_tracks,
                )
            except Exception:
                self._timeline_vm = None

        # Restore the caller-provided selection into the freshly-built VM so
        # that highlight state survives a full timeline refresh.
        if self._timeline_vm is not None and selection is not None:
            self._timeline_vm.selection = selection

        total_dur = timeline.total_duration() if not timeline.is_empty() else 0.0
        needed = max(24000.0, total_dur * self._scale_px_per_sec + 1600.0)
        if needed > self._scene_width:
            self._scene_width = needed
        self._redraw()

    def toggle_motor_track(self, track_name: str) -> None:
        """Expand or collapse a motor sub-track.  UI-only state; no model mutation."""
        current = self._motor_track_expanded.get(track_name, True)
        self._motor_track_expanded[track_name] = not current
        self._redraw()

    def toggle_body_part(self, limb_id: str) -> None:
        """Expand or collapse a body-part group.  UI-only state; no model mutation."""
        current = self._body_part_expanded.get(limb_id, True)
        self._body_part_expanded[limb_id] = not current
        self._redraw()

    def _action_id_for_index(self, idx: int) -> str:
        """Return the action_id for the given action segment index.

        Falls back to the first action's id when idx < 0.  Used to attach
        action context to body-part selection events so the single selection
        model carries full ``(action_id, limb_id)`` semantics.
        """
        segs = []
        if self._behavior_timeline is not None:
            segs = getattr(self._behavior_timeline, "action_segments", []) or []
        if 0 <= idx < len(segs):
            return getattr(segs[idx], "action_id", "") or ""
        if segs:
            return getattr(segs[0], "action_id", "") or ""
        return ""

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())

        # Click in ruler area → set playhead
        if scene_pos.y() <= self._ruler_h:
            label_w = self._label_lane_w if self._behavior_timeline is not None else 0
            raw_x = scene_pos.x() - label_w
            if raw_x >= 0 and self._scale_px_per_sec > 0:
                self.set_playhead(raw_x / self._scale_px_per_sec)
            event.accept()
            return

        if self._behavior_timeline is not None:
            items = self._scene.items(scene_pos)
            for item in items:
                if item.data(1) == "body_part_toggle":
                    limb_id = str(item.data(2) or "")
                    if limb_id:
                        self.toggle_body_part(limb_id)
                    event.accept()
                    return
                if item.data(1) == "body_part_module":
                    limb_id = str(item.data(2) or "")
                    action_id = str(item.data(3) or "")
                    self._selected_index = -1
                    if self._timeline_vm is not None:
                        from system.behavior.timeline_view_model import TimelineSelection
                        self._timeline_vm.selection = (
                            TimelineSelection.body_part(action_id, limb_id)
                            if limb_id else TimelineSelection.none()
                        )
                    self.body_part_selected.emit(action_id, limb_id)
                    self._redraw()
                    event.accept()
                    return
                if item.data(1) == "body_part_select":
                    limb_id = str(item.data(2) or "")
                    # Capture action context BEFORE clearing _selected_index.
                    action_id = self._action_id_for_index(self._selected_index)
                    self._selected_index = -1
                    if self._timeline_vm is not None:
                        from system.behavior.timeline_view_model import TimelineSelection
                        self._timeline_vm.selection = (
                            TimelineSelection.body_part(action_id, limb_id)
                            if limb_id else TimelineSelection.none()
                        )
                    self.body_part_selected.emit(action_id, limb_id)
                    self._redraw()
                    event.accept()
                    return
            # Collapsed tracks stay visible but shield content interactions.
            if float(scene_pos.x()) >= float(self._label_lane_w):
                collapsed_track = self._motor_track_name_at_y(
                    float(scene_pos.y()),
                    only_collapsed=True,
                )
                if collapsed_track:
                    self.module_selected.emit(-1)
                    self._clear_motor_selection(notify=True)
                    self.setFocus()
                    event.accept()
                    return
                # Collapsed body-part group 閳?click selects without toggling
                limb_id = self._body_part_limb_at_y(float(scene_pos.y()), only_collapsed=True)
                if limb_id:
                    action_id = self._action_id_for_index(self._selected_index)
                    if self._timeline_vm is not None:
                        from system.behavior.timeline_view_model import TimelineSelection
                        self._timeline_vm.selection = (
                            TimelineSelection.body_part(action_id, limb_id)
                            if limb_id else TimelineSelection.none()
                        )
                    self.body_part_selected.emit(action_id, limb_id)
                    self._redraw()
                    event.accept()
                    return

            # Fix 2: check for motor segment hit first (motor tracks are below main track)
            for item in items:
                if item.data(1) == "motor":
                    motor_id = item.data(0)
                    track_name = item.data(2)
                    motor_id_str = str(motor_id or "")
                    track_name_str = str(track_name or "")
                    self._selected_index = -1
                    if self._timeline_vm is not None:
                        from system.behavior.timeline_view_model import TimelineSelection
                        self._timeline_vm.selection = TimelineSelection.motor(
                            action_id="", limb_id="",
                            track_name=track_name_str, motor_id=motor_id_str,
                        )
                    self.motor_segment_selected.emit(motor_id_str, track_name_str)
                    self._motor_press_id = motor_id
                    self._motor_press_track = track_name
                    self._motor_drag_started = False
                    self._motor_press_scene_x = float(scene_pos.x())
                    seg = self._find_motor_seg(motor_id, track_name)
                    self._motor_orig_start = seg.start_time if seg is not None else 0.0
                    self.setFocus()
                    event.accept()
                    return

        if self._main_y <= scene_pos.y() <= (self._main_y + self._track_h):
            content_x = self._content_x_from_scene_x(float(scene_pos.x()))
            idx = self._module_index_at_x(content_x)
            if idx >= 0:
                self._clear_motor_selection(notify=True)
                self.module_selected.emit(idx)
                self._press_index = idx
                self._press_scene_x = float(scene_pos.x())
                self._press_scene_y = float(scene_pos.y())
                self._drag_started = False
                self._drag_from_index = None
                module_left = self._module_left_x(idx)
                self._drag_grab_offset_x = content_x - module_left
                self._drag_left_x = module_left
                self._drag_insert_pos = None
                self.setFocus()
                event.accept()
                return
        self.module_selected.emit(-1)
        self._clear_motor_selection(notify=True)
        self._press_index = None
        self._drag_started = False
        self._drag_from_index = None
        self._drag_insert_pos = None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # Fix 2: clear motor drag state
        if self._motor_press_id is not None:
            self._motor_press_id = None
            self._motor_press_track = None
            self._motor_drag_started = False
            return
        if not self._drag_started or self._drag_from_index is None:
            self._press_index = None
            self._drag_started = False
            self._drag_from_index = None
            self._drag_insert_pos = None
            return
        to_index = self._drag_drop_index()
        from_index = self._drag_from_index
        self._press_index = None
        self._drag_started = False
        self._drag_from_index = None
        self._drag_insert_pos = None
        if to_index != from_index:
            self.module_reordered.emit(from_index, to_index)
        else:
            self._redraw()

    def mouseDoubleClickEvent(self, event):
        """Fix 2: double-click on a motor segment opens param edit dialog."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        if self._behavior_timeline is None:
            super().mouseDoubleClickEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if float(scene_pos.x()) >= float(self._label_lane_w):
            collapsed_track = self._motor_track_name_at_y(
                float(scene_pos.y()),
                only_collapsed=True,
            )
            if collapsed_track:
                event.accept()
                return
        items = self._scene.items(scene_pos)
        for item in items:
            if item.data(1) == "motor":
                motor_id = item.data(0)
                track_name = item.data(2)
                seg = self._find_motor_seg(motor_id, track_name)
                if seg is not None:
                    track_def = self._available_tracks.get(track_name)
                    dlg = _MotorParamEditDialog(seg, track_def, parent=self)
                    if dlg.exec() == QDialog.DialogCode.Accepted:
                        self._redraw()
                        self.timeline_edited.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        # Fix 2: motor segment drag
        if self._motor_press_id is not None:
            dx = abs(float(scene_pos.x()) - self._motor_press_scene_x)
            if not self._motor_drag_started:
                if dx < self._drag_start_threshold_px:
                    return
                self._motor_drag_started = True
            delta_sec = (float(scene_pos.x()) - self._motor_press_scene_x) / self._scale_px_per_sec
            new_start = max(0.0, self._motor_orig_start + delta_sec)
            self._update_motor_seg_start(self._motor_press_id, self._motor_press_track, new_start)
            self._redraw()
            return

        if self._press_index is None:
            return

        cur_x = float(max(0.0, scene_pos.x()))
        cur_content_x = self._content_x_from_scene_x(cur_x)
        cur_y = float(scene_pos.y())
        dx = abs(cur_x - self._press_scene_x)
        dy = abs(cur_y - self._press_scene_y)
        if not self._drag_started:
            if max(dx, dy) < self._drag_start_threshold_px:
                return
            self._drag_started = True
            self._drag_from_index = self._press_index

        if self._drag_from_index is None:
            return
        drag_w = self._duration_to_width(self._modules[self._drag_from_index].duration)
        content_w = self._content_width()
        self._drag_left_x = max(0.0, min(cur_content_x - self._drag_grab_offset_x, content_w - drag_w))
        self._drag_cursor_x = cur_x
        self._drag_insert_pos = self._drag_insert_pos_for_x(self._drag_left_x, self._drag_left_x + drag_w)
        self._redraw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self._selected_index >= 0:
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Ctrl + wheel zooms timeline tick granularity inside ruler/track box."""
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            super().wheelEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if not self._is_in_timeline_box(float(scene_pos.x()), float(scene_pos.y())):
            super().wheelEvent(event)
            return

        delta = int(event.angleDelta().y())
        if delta == 0:
            event.accept()
            return

        cur_idx = self._nearest_zoom_tick_index(self._tick_sec)
        # Wheel up => zoom in => finer tick.
        new_idx = cur_idx + (1 if delta > 0 else -1)
        new_idx = max(0, min(new_idx, len(self._ZOOM_TICK_STEPS) - 1))
        new_tick = self._ZOOM_TICK_STEPS[new_idx]
        if abs(new_tick - self._tick_sec) < 1e-12:
            event.accept()
            return

        old_scale = self._scale_px_per_sec
        content_origin = self._content_origin_x()
        anchor_content_x = max(0.0, float(scene_pos.x()) - content_origin)
        anchor_time = anchor_content_x / old_scale if old_scale > 1e-9 else 0.0

        self._tick_sec = new_tick
        self._scale_px_per_sec = self._base_scale_px_per_sec * (self._base_tick_sec / self._tick_sec)

        min_width = self._minimum_scene_width_for_scale()
        if min_width > self._scene_width:
            self._scene_width = min_width

        self._redraw()

        new_anchor_scene_x = content_origin + anchor_time * self._scale_px_per_sec
        sb = self.horizontalScrollBar()
        sb.setValue(int(round(sb.value() + (new_anchor_scene_x - float(scene_pos.x())))))
        event.accept()

    def _ensure_infinite_width(self, value: int) -> None:
        sb = self.horizontalScrollBar()
        if value >= sb.maximum() - 120:
            self._scene_width += 4000.0
            self._redraw()

    def _content_origin_x(self) -> float:
        return float(self._label_lane_w if self._behavior_timeline is not None else 0.0)

    def _content_width(self) -> float:
        return max(0.0, float(self._scene_width - self._content_origin_x()))

    def _content_x_from_scene_x(self, scene_x: float) -> float:
        return max(0.0, float(scene_x) - self._content_origin_x())

    def _is_in_timeline_box(self, scene_x: float, scene_y: float) -> bool:
        if scene_x < 0.0 or scene_y < 0.0:
            return False
        if scene_x > self._scene_width:
            return False
        return scene_y <= float(self._content_height)

    @classmethod
    def _nearest_zoom_tick_index(cls, tick: float) -> int:
        return min(
            range(len(cls._ZOOM_TICK_STEPS)),
            key=lambda i: abs(cls._ZOOM_TICK_STEPS[i] - float(tick)),
        )

    def _minimum_scene_width_for_scale(self) -> float:
        if self._behavior_timeline is not None:
            total_dur = self._behavior_timeline.total_duration() if not self._behavior_timeline.is_empty() else 0.0
        else:
            total_dur = self._total_duration(self._modules)
        # Keep a right-side padding roughly proportional to current scale.
        return max(24000.0, total_dur * self._scale_px_per_sec + (self._scale_px_per_sec * 9.0))

    # Fix 2: motor segment helpers
    def _find_motor_seg(self, motor_id: str, track_name: str) -> Optional["MotorSegment"]:
        """Walk _behavior_timeline overlays to find a segment by motor_id+track_name."""
        if self._behavior_timeline is None:
            return None
        for overlay in self._behavior_timeline.motor_overlays:
            for seg in overlay.motor_segments:
                if seg.motor_id == motor_id and seg.track_name == track_name:
                    return seg
        return None

    def _update_motor_seg_start(self, motor_id: str, track_name: Optional[str], new_start: float) -> None:
        """Mutate segment.start_time in place and re-emit timeline_edited signal."""
        if self._behavior_timeline is None:
            return
        for overlay in self._behavior_timeline.motor_overlays:
            for seg in overlay.motor_segments:
                if seg.motor_id == motor_id and seg.track_name == (track_name or seg.track_name):
                    seg.start_time = max(0.0, new_start)
                    self.timeline_edited.emit()
                    return

    def _redraw(self) -> None:
        if self._behavior_timeline is not None:
            self._redraw_multi_track()
            return

        self._scene.clear()
        self._content_height = self._secondary_y + self._track_h
        self.setSceneRect(0.0, 0.0, self._scene_width, float(self._content_height))

        # Ruler background
        self._scene.addRect(0, 0, self._scene_width, self._ruler_h, QPen(QColor("#454545")), QBrush(QColor("#272727")))

        # Track backgrounds
        self._scene.addRect(
            0,
            self._main_y,
            self._scene_width,
            self._track_h,
            QPen(QColor("#4f4f4f")),
            QBrush(QColor(self._main_track_bg)),
        )
        self._scene.addRect(
            0,
            self._secondary_y,
            self._scene_width,
            self._track_h,
            QPen(QColor("#4f4f4f")),
            QBrush(QColor(self._secondary_track_bg)),
        )

        self._draw_ruler()
        self._draw_modules(self._modules, y=self._main_y, selected=self._selected_index, track="main")
        self._draw_modules(self._secondary, y=self._secondary_y, selected=-1, track="secondary")
        self._draw_playhead(label_offset=0)

        # No track titles by design.

    def set_playhead(self, t: float) -> None:
        """Set the playhead to time *t* (seconds) and redraw."""
        self._playhead_time = max(0.0, t)
        self._redraw()
        self.playhead_moved.emit(self._playhead_time)

    def _draw_playhead(self, label_offset: int = 0) -> None:
        """Draw a vertical playhead line at the current playhead time."""
        x = self._playhead_time * self._scale_px_per_sec + label_offset
        pen = QPen(QColor("#ff4444"))
        pen.setWidth(2)
        line = self._scene.addLine(x, 0.0, x, float(self._content_height), pen)
        line.setZValue(10)  # always on top

    # Label lane width 閳?recomputed each redraw to fit the widest label text.
    # Stored as instance variable so mouse-event coordinate translation stays consistent.
    _label_lane_w: int = 80  # initial default; overwritten by _redraw_multi_track()
    _motor_track_h: int = 36
    _motor_track_collapsed_h: int = 14

    # Body-part header row height constants
    _body_part_h: int = 22
    _body_part_indent: int = 8  # motor-row label indent when under a body-part header

    def _redraw_multi_track(self) -> None:
        """Render Action + body-part sub-tracks from BehaviorTimeline."""
        timeline = self._behavior_timeline
        if timeline is None:
            return

        self._scene.clear()
        self._motor_track_rows = {}
        self._body_part_rows = {}

        vm = self._timeline_vm
        use_canonical = (
            vm is not None and not vm.is_degraded and len(vm.actions) > 0
        )

        if use_canonical:
            render_groups = list(vm.layout_body_parts)
        else:
            render_groups = []
            track_labels = {
                t: (self._available_tracks[t].label if t in self._available_tracks else t)
                for t in self._motor_track_names
            }

        from PySide6.QtGui import QFontMetrics
        _fm = QFontMetrics(self.font())
        all_texts = ["Action"]
        if use_canonical:
            for bp in render_groups:
                all_texts.append(bp.limb_label)
        else:
            for lbl in track_labels.values():
                all_texts.append(lbl)
        _max_text_px = max((_fm.horizontalAdvance(s) for s in all_texts), default=40)
        label_w = max(_max_text_px + 24, 72)
        self._label_lane_w = label_w

        total_body_h = 0
        if use_canonical:
            total_body_h = len(render_groups) * (self._motor_track_h + 2)
        else:
            if vm is not None and vm.is_degraded:
                total_body_h += 22
            total_body_h += len(self._motor_track_names) * (self._motor_track_h + 2)
        total_h = self._ruler_h + self._track_h + total_body_h + 8
        self._content_height = total_h
        self.setSceneRect(0.0, 0.0, self._scene_width, float(total_h))

        self._scene.addRect(
            0, 0, self._scene_width, self._ruler_h,
            QPen(QColor("#454545")), QBrush(QColor("#272727")),
        )
        self._scene.addRect(
            0, 0, label_w, self._ruler_h,
            QPen(QColor("#3a3a3a")), QBrush(QColor("#1a1a1a")),
        )
        self._draw_ruler(label_offset=label_w)

        action_y = float(self._ruler_h)
        self._scene.addRect(
            0, action_y, label_w, self._track_h,
            QPen(QColor("#4a4a4a")), QBrush(QColor("#1a1a1a")),
        )
        lbl = self._scene.addSimpleText("Action")
        lbl.setBrush(QBrush(QColor("#888888")))
        lbl.setPos(4, action_y + 4)
        self._scene.addRect(
            label_w, action_y, self._scene_width - label_w, self._track_h,
            QPen(QColor("#4f4f4f")), QBrush(QColor(self._main_track_bg)),
        )
        for i, seg in enumerate(timeline.action_segments):
            x = seg.start_time * self._scale_px_per_sec + label_w
            w = self._duration_to_width(seg.duration)
            selected = (i == self._selected_index)
            self._draw_action_segment(seg, i, x, action_y, w, selected)

        row_y = action_y + self._track_h + 2
        total_dur = timeline.total_duration() if not timeline.is_empty() else 0.0

        if use_canonical:
            _sel = vm.selection if vm is not None else None
            for bp in render_groups:
                row_h = self._motor_track_h
                self._body_part_rows[bp.limb_id] = (row_y, float(row_h))
                is_selected_bp = (
                    _sel is not None and _sel.limb_id == bp.limb_id
                    and _sel.level.value in ("body_part", "motor")
                )
                row_pen = QPen(QColor("#6b7280"), 1.2) if is_selected_bp else QPen(QColor("#3a3a3a"))
                self._scene.addRect(
                    0, row_y, label_w, row_h,
                    row_pen, QBrush(QColor("#141414")),
                ).setData(1, "body_part_select")
                label_item = self._scene.addSimpleText(bp.limb_label)
                label_item.setBrush(QBrush(QColor("#e2e8f0" if is_selected_bp else "#a8b0ba")))
                label_item.setPos(8, row_y + max(0, (row_h - 14) // 2))
                label_item.setData(1, "body_part_select")
                label_item.setData(2, bp.limb_id)
                label_rect = self._scene.items(label_item.boundingRect().translated(label_item.pos()))
                self._scene.addRect(
                    label_w, row_y, self._scene_width - label_w, row_h,
                    row_pen, QBrush(QColor(self._secondary_track_bg)),
                )
                # Click-anywhere selection lane on the row.
                lane = self._scene.addRect(
                    0, row_y, self._scene_width, row_h,
                    QPen(Qt.PenStyle.NoPen), QBrush(Qt.BrushStyle.NoBrush),
                )
                lane.setData(1, "body_part_select")
                lane.setData(2, bp.limb_id)
                self._render_body_part_track_content(bp, row_y, row_h, label_w, total_dur, timeline)
                row_y += row_h + 2
        else:
            if vm is not None and vm.is_degraded:
                notice_h = 20
                reason = vm.degraded_reason or "Topology unavailable - showing flat track view"
                notice_bg = self._scene.addRect(
                    0, row_y, self._scene_width, notice_h,
                    QPen(QColor("#5a4a00")), QBrush(QColor("#2a2000")),
                )
                notice_bg.setData(1, "degraded_notice")
                notice_bg.setData(2, reason)
                notice_lbl = self._scene.addSimpleText(f"~ {reason}")
                notice_lbl.setBrush(QBrush(QColor("#c8a030")))
                notice_lbl.setPos(4, row_y + 3)
                notice_lbl.setData(1, "degraded_notice_text")
                row_y += notice_h + 2

            for tname in self._motor_track_names:
                row_h = self._motor_track_h
                self._motor_track_rows[tname] = (row_y, float(row_h), True)
                track_label = track_labels.get(tname, tname)
                self._scene.addRect(
                    0, row_y, label_w, row_h,
                    QPen(QColor("#3a3a3a")), QBrush(QColor("#1a1a1a")),
                )
                track_lbl = self._scene.addSimpleText(track_label)
                track_lbl.setBrush(QBrush(QColor("#aaaaaa")))
                track_lbl.setPos(4, row_y + max(0, (row_h - 14) // 2))
                self._scene.addRect(
                    label_w, row_y, self._scene_width - label_w, row_h,
                    QPen(QColor("#3a3a3a")), QBrush(QColor(self._secondary_track_bg)),
                )
                self._render_motor_track_content(
                    tname, tname, "#5f7fbf", row_y, row_h, label_w, total_dur, timeline
                )
                row_y += row_h + 2

    def _render_body_part_track_content(
        self,
        body_part: Any,
        row_y: float,
        row_h: int,
        label_w: int,
        total_dur: float,
        timeline: Any,
    ) -> None:
        track_names = {
            mr.track_name for mr in getattr(body_part, "motors", [])
        }
        track_names.update(
            getattr(mr, "lookup_track_name", "") or ""
            for mr in getattr(body_part, "motors", [])
        )
        track_names.discard("")
        if not track_names:
            return

        for action_seg in getattr(timeline, "action_segments", []) or []:
            overlay = timeline.get_overlay_for_action(getattr(action_seg, "action_id", ""))
            if overlay is None:
                continue
            matched = [
                seg for seg in getattr(overlay, "motor_segments", []) or []
                if getattr(seg, "track_name", "") in track_names
            ]
            if not matched:
                continue
            start_time = min(float(seg.start_time) for seg in matched)
            end_time = max(float(seg.start_time) + float(seg.duration) for seg in matched)
            x = start_time * self._scale_px_per_sec + label_w
            w = max(18.0, (end_time - start_time) * self._scale_px_per_sec)
            selected = (
                self._timeline_vm is not None
                and self._timeline_vm.selection.matches_body_part(action_seg.action_id, body_part.limb_id)
            )
            self._draw_body_part_segment(
                limb_id=body_part.limb_id,
                action_id=action_seg.action_id,
                action_name=action_seg.name,
                x=x,
                y=row_y,
                width=w,
                height=row_h,
                selected=selected,
            )

        self._draw_playhead(label_offset=self._label_lane_w)

    def _draw_body_part_segment(
        self,
        limb_id: str,
        action_id: str,
        action_name: str,
        x: float,
        y: float,
        width: float,
        height: float,
        selected: bool,
    ) -> None:
        fill = QColor("#4d84c4")
        fill.setAlpha(185)
        pen = QPen(QColor("#f4d03f"), 2) if selected else QPen(QColor("#7f8ea3"), 1)
        rect = self._scene.addRect(x, y + 3, width, height - 6, pen, QBrush(fill))
        rect.setData(1, "body_part_module")
        rect.setData(2, limb_id)
        rect.setData(3, action_id)
        display_name = self._action_display_name(action_name)
        rect.setToolTip(f"{display_name} | {limb_id}")
        if width > 42:
            title = self._scene.addSimpleText(display_name)
            title.setBrush(QBrush(QColor("#ffffff")))
            title.setPos(x + 6, y + 6)
            title.setToolTip(f"{display_name} | {limb_id}")
            title.setData(1, "body_part_module")
            title.setData(2, limb_id)
            title.setData(3, action_id)

    def _render_motor_track_content(
        self,
        row_track_name: str,
        lookup_track_name: str,
        track_color: str,
        motor_y: float,
        row_h: int,
        label_w: int,
        total_dur: float,
        timeline: Any,
    ) -> None:
        """Render motor segments for degraded fallback rows only."""
        motor_segs = timeline.get_motor_segments_for_track(lookup_track_name)
        for mseg in motor_segs:
            mx = mseg.start_time * self._scale_px_per_sec + label_w
            mw = max(float(row_h), mseg.duration * self._scale_px_per_sec)
            self._draw_motor_segment(
                mseg, mx, motor_y, mw, row_h, track_color, row_track_name, lookup_track_name
            )

    def _motor_track_name_at_y(self, scene_y: float, only_collapsed: bool = False) -> Optional[str]:
        for tname, (top, height, expanded) in self._motor_track_rows.items():
            if not (top <= scene_y <= (top + height)):
                continue
            if only_collapsed and expanded:
                continue
            return tname
        return None

    def _body_part_limb_at_y(self, scene_y: float, only_collapsed: bool = False) -> Optional[str]:
        """Return limb_id if scene_y falls within a body-part header row."""
        for limb_id, (top, height) in self._body_part_rows.items():
            if not (top <= scene_y <= (top + height)):
                continue
            if only_collapsed and self._body_part_expanded.get(limb_id, True):
                continue
            return limb_id
        return None

    def _draw_action_segment(
        self,
        seg: "ActionSegment",
        index: int,
        x: float,
        y: float,
        width: float,
        selected: bool,
    ) -> None:
        fill = QColor("#4d84c4") if seg.kind == "movement" else QColor("#7a5db6")
        pen = QPen(QColor("#f4d03f"), 2) if selected else QPen(QColor("#9b9b9b"), 1)
        rect = self._scene.addRect(x, y, width, self._track_h, pen, QBrush(fill))
        rect.setData(0, index)
        rect.setData(1, "main")
        display_name = self._action_display_name(seg.name)
        tooltip = f"{display_name} ({seg.name})"
        rect.setToolTip(tooltip)
        title = self._scene.addSimpleText(display_name)
        title.setBrush(QBrush(QColor("#ffffff")))
        title.setPos(x + 6, y + 6)
        title.setToolTip(tooltip)
        dur_txt = self._scene.addSimpleText(f"{seg.duration:.2f}s")
        dur_txt.setBrush(QBrush(QColor("#e9e9e9")))
        dur_txt.setPos(x + 6, y + 24)
        dur_txt.setToolTip(tooltip)

    @staticmethod
    def _action_display_name(action_id: str) -> str:
        """Return a UI-friendly action name for the Action Track title."""
        profile = UNITREE_ACTION_PROFILES.get(action_id)
        if profile is not None and str(profile.label or "").strip():
            label = str(profile.label).strip()
            if "/" in label:
                tail = label.split("/")[-1].strip()
                if tail:
                    return tail
            return label
        pretty = str(action_id or "").replace("_", " ").strip()
        if not pretty:
            return "Action"
        return pretty.title()

    def _draw_motor_segment(
        self,
        seg: "MotorSegment",
        x: float,
        y: float,
        width: float,
        height: float,
        track_color: str,
        track_name: str = "",
        lookup_track_name: str = "",
    ) -> None:
        fill = QColor(track_color)
        fill.setAlpha(160)
        _vm_sel = self._timeline_vm.selection if self._timeline_vm is not None else None
        selected = (
            _vm_sel is not None
            and _vm_sel.level.value == "motor"
            and _vm_sel.motor_id == seg.motor_id
            and _vm_sel.track_name == (track_name or seg.track_name)
        )
        pen = QPen(QColor("#f4d03f"), 2) if selected else QPen(QColor(track_color), 1)
        rect = self._scene.addRect(x, y + 2, width, height - 4, pen, QBrush(fill))
        rect.setData(0, seg.motor_id)
        rect.setData(1, "motor")
        rect.setData(2, track_name or seg.track_name)
        rect.setData(3, lookup_track_name or seg.track_name)
        display_name = self._motor_segment_display_name(seg)
        tooltip = f"{display_name} ({seg.motor_id})`nduration={seg.duration:.2f}s"
        rect.setToolTip(tooltip)
        if width > 20:
            title_txt = self._scene.addSimpleText(display_name)
            title_txt.setBrush(QBrush(QColor("#dddddd")))
            title_txt.setPos(x + 4, y + 2)
            title_txt.setToolTip(tooltip)
        if width > 26:
            dur_txt = self._scene.addSimpleText(f"{seg.duration:.2f}s")
            dur_txt.setBrush(QBrush(QColor("#f0f0f0")))
            dur_txt.setPos(x + 4, y + height - 16)
            dur_txt.setToolTip(tooltip)

    def _clear_motor_selection(self, notify: bool = False) -> None:
        had_sel = (
            self._timeline_vm is not None
            and self._timeline_vm.selection.level.value == "motor"
        )
        if self._timeline_vm is not None:
            from system.behavior.timeline_view_model import TimelineSelection
            self._timeline_vm.selection = TimelineSelection.none()
        if notify and had_sel:
            self.motor_segment_selected.emit("", "")

    def _motor_segment_display_name(self, seg: "MotorSegment") -> str:
        """Resolve a readable label for a motor segment."""
        tl = self._behavior_timeline
        if tl is not None and seg.parent_action_id:
            for action in tl.action_segments:
                if action.action_id == seg.parent_action_id:
                    return self._action_display_name(action.name)
        track_def = self._available_tracks.get(seg.track_name)
        if track_def is not None:
            return track_def.label
        return "Segment"

    def _draw_ruler(self, label_offset: int = 0) -> None:
        """Draw ruler ticks and second labels.

        label_offset: pixels reserved on the left for track labels.  Ticks
        start at x = label_offset so the t=0 mark always lines up with the
        left edge of the track content region.
        """
        content_width = self._scene_width - label_offset
        max_tick = int(content_width / (self._tick_sec * self._scale_px_per_sec))
        for tick_idx in range(max_tick + 1):
            sec = tick_idx * self._tick_sec
            x = label_offset + sec * self._scale_px_per_sec

            if self._is_time_multiple(sec, 1.0):  # 1.0s
                h = 20
                color = QColor("#d7d7d7")
                show_label = True
            elif self._is_time_multiple(sec, 0.1):  # 0.1s
                h = 12
                color = QColor("#7f7f7f")
                show_label = False
            else:  # 0.01s minimal tick
                h = 8
                color = QColor("#5f5f5f")
                show_label = False

            self._scene.addLine(x, self._ruler_h - h, x, self._ruler_h, QPen(color, 1))
            if show_label:
                t = self._scene.addSimpleText(str(int(round(sec))))
                t.setBrush(QBrush(QColor("#d7d7d7")))
                t.setPos(x + 2, 2)

    @staticmethod
    def _is_time_multiple(value: float, unit: float) -> bool:
        if unit <= 0:
            return False
        q = value / unit
        return abs(q - round(q)) < 1e-6

    def _draw_modules(
        self,
        modules: List[SequenceModule],
        y: float,
        selected: int,
        track: str,
    ) -> None:
        if track == "main" and self._drag_from_index is not None and 0 <= self._drag_from_index < len(modules):
            self._draw_main_modules_with_drag(modules, y, selected)
            return

        x = 0.0
        for i, mod in enumerate(modules):
            width = self._duration_to_width(mod.duration)
            self._draw_module_rect(mod, i, track, x, y, width, i == selected and track == "main")

            x += width

    def _draw_main_modules_with_drag(self, modules: List[SequenceModule], y: float, selected: int) -> None:
        from_idx = self._drag_from_index
        if from_idx is None:
            return
        widths = [self._duration_to_width(m.duration) for m in modules]
        others = [i for i in range(len(modules)) if i != from_idx]
        current_pos = sum(1 for i in others if i < from_idx)
        insert_pos = self._drag_insert_pos if self._drag_insert_pos is not None else current_pos
        insert_pos = max(0, min(insert_pos, len(others)))

        # Render a live "gap" at target insertion position as drop preview.
        x = 0.0
        for pos, idx in enumerate(others):
            if pos == insert_pos:
                x += widths[from_idx]
            w = widths[idx]
            self._draw_module_rect(modules[idx], idx, "main", x, y, w, idx == selected)
            x += w
        if insert_pos == len(others):
            x += widths[from_idx]

        drag_width = widths[from_idx]
        drag_x = max(0.0, min(self._drag_left_x, self._scene_width - drag_width))
        self._draw_module_rect(
            modules[from_idx],
            from_idx,
            "main",
            drag_x,
            y,
            drag_width,
            from_idx == selected,
            dragging=True,
        )

    def _draw_module_rect(
        self,
        mod: SequenceModule,
        index: int,
        track: str,
        x: float,
        y: float,
        width: float,
        selected: bool,
        dragging: bool = False,
    ) -> None:
        fill = QColor("#4d84c4") if mod.kind == "movement" else QColor("#7a5db6")
        if selected and track == "main":
            pen = QPen(QColor("#f4d03f"), 2)
        else:
            pen = QPen(QColor("#9b9b9b"), 1)
        if dragging:
            pen = QPen(QColor("#f4d03f"), 2, Qt.PenStyle.DashLine)
        rect = self._scene.addRect(x, y, width, self._track_h, pen, QBrush(fill))
        rect.setData(0, index)
        rect.setData(1, track)

        title = self._scene.addSimpleText(mod.name)
        title.setBrush(QBrush(QColor("#ffffff")))
        title.setPos(x + 6, y + 6)

        dur = self._scene.addSimpleText(f"{mod.duration:.2f}s")
        dur.setBrush(QBrush(QColor("#e9e9e9")))
        dur.setPos(x + 6, y + 24)

    def _duration_to_width(self, duration: float) -> float:
        quant = max(self._tick_sec, round(duration / self._tick_sec) * self._tick_sec)
        return max(float(self._track_h), quant * self._scale_px_per_sec)

    @staticmethod
    def _total_duration(modules: List[SequenceModule]) -> float:
        total = 0.0
        for m in modules:
            total += max(0.25, m.duration)
        return total

    def _module_left_x(self, index: int) -> float:
        cursor = 0.0
        for i, mod in enumerate(self._modules):
            if i == index:
                return cursor
            cursor += self._duration_to_width(mod.duration)
        return cursor

    def _module_index_at_x(self, x: float) -> int:
        cursor = 0.0
        for i, mod in enumerate(self._modules):
            w = self._duration_to_width(mod.duration)
            if cursor <= x < cursor + w:
                return i
            cursor += w
        return -1

    def _drag_insert_pos_for_x(self, drag_left: float, drag_right: float) -> int:
        if self._drag_from_index is None:
            return 0
        others = [i for i in range(len(self._modules)) if i != self._drag_from_index]
        if not others:
            return 0

        drag_center = (drag_left + drag_right) / 2.0
        desired_final_index = len(self._modules)
        for idx in range(len(self._modules)):
            if idx == self._drag_from_index:
                continue
            left = self._module_left_x(idx)
            w = self._duration_to_width(self._modules[idx].duration)
            center = left + (w / 2.0)
            if drag_center < center:
                desired_final_index = idx
                break
        return sum(1 for i in others if i < desired_final_index)

    def _drag_drop_index(self) -> int:
        if self._drag_from_index is None:
            return -1
        if self._drag_insert_pos is None:
            return self._drag_from_index
        others_count = max(0, len(self._modules) - 1)
        return max(0, min(self._drag_insert_pos, others_count))


def _format_compile_output(artifact) -> str:
    status_tag = "OK" if artifact.is_valid else "ERROR"
    lines = [
        f"[{status_tag}] behavior_ref='{artifact.behavior_ref}' artifact_id={artifact.artifact_id[:8]}...",
        f"       nodes={len(artifact.behavior_ir.nodes)} edges={len(artifact.behavior_ir.edges)}",
        f"       compiled_at={artifact.compiled_at}",
    ]
    if artifact.source_hash:
        lines.append(f"       source_hash={artifact.source_hash[:16]}...")
    if artifact.diagnostics:
        lines.append(
            f"       {artifact.error_count} error(s) {artifact.warning_count} warning(s):"
        )
        for d in artifact.diagnostics:
            loc = f" [{d.location}]" if d.location else ""
            lines.append(f"         [{d.level.upper()}] {d.code}: {d.message}{loc}")
    else:
        lines.append("       0 diagnostics")
    return "\n".join(lines)


def _extract_steps_from_source(source: str) -> List[Dict[str, str]]:
    """Legacy helper: parse call-like lines into step dicts.

    Returns:
        [{"action": "<name>", "args": "<arg_text>"}, ...]
    """
    steps: List[Dict[str, str]] = []
    if not source:
        return steps
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", line)
        if not m:
            continue
        steps.append({
            "action": m.group(1).strip(),
            "args": m.group(2).strip(),
        })
    return steps


def _render_steps_to_source(steps: List[Dict[str, str]]) -> str:
    """Legacy helper: render step dicts into call-like source lines."""
    if not steps:
        return ""
    lines: List[str] = []
    for step in steps:
        action = str(step.get("action", "")).strip()
        args = str(step.get("args", "")).strip()
        if not action:
            action = "noop"
        lines.append(f"{action}({args})" if args else f"{action}()")
    return "\n".join(lines)


def _default_core_source_for_behavior_ref(behavior_ref: str) -> str:
    """Return initial editor text seeded from the selected behavior/action ref."""
    legacy_default = (
        "stand(duration=2.0)\n"
        "forward(speed=0.3, duration=4.0)\n"
        "wait(duration=1.0)\n"
        "sit()"
    )
    ref = str(behavior_ref or "").strip()
    if not ref or ref.startswith("<"):
        return legacy_default
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ref):
        return legacy_default
    if ref in {"walk", "forward"}:
        return "forward(speed=0.3, duration=4.0)"
    if ref == "wait":
        return "wait(duration=1.0)"
    return f"{ref}()"


def _semantic_descriptor_for_behavior(
    intent_id: str,
    display_name: str = "",
    brand: str = "",
    robot_type: str = "",
) -> Optional[SemanticActionDescriptor]:
    """Resolve the authored IR descriptor for a Behavior node selection.
    
    Args:
        intent_id: The semantic intent ID to look up.
        display_name: Fallback display name for matching.
        brand: Robot brand for filtering (e.g., "unitree").
        robot_type: Robot type for filtering (e.g., "go2").
    """
    wanted_intent = str(intent_id or "").strip()
    wanted_label = str(display_name or "").strip().lower()
    try:
        # Circle 4 Step 4.1: Pass brand/robot_type for机型-specific filtering
        descriptors = get_semantic_actions(brand, robot_type)
    except Exception:
        return None

    for desc in descriptors:
        if wanted_intent and str(getattr(desc, "intent_id", "") or "").strip() == wanted_intent:
            return desc
    for desc in descriptors:
        label = str(getattr(desc, "display_name", "") or "").strip().lower()
        raw = str(getattr(desc, "raw_action", "") or "").strip().lower()
        if wanted_label and wanted_label in {label, raw}:
            return desc
    return None


def _default_core_source_for_behavior_selection(
    intent_id: str,
    display_name: str = "",
    brand: str = "",
    robot_type: str = "",
) -> str:
    """Seed editor source from semantic intent rather than behavior_ref."""
    descriptor = _semantic_descriptor_for_behavior(intent_id, display_name, brand, robot_type)
    if descriptor is None:
        return _default_core_source_for_behavior_ref(display_name or intent_id)

    raw_action = str(getattr(descriptor, "raw_action", "") or "").strip()
    if not raw_action:
        return _default_core_source_for_behavior_ref(display_name or intent_id)

    arg_parts: List[str] = []
    for key, meta in (getattr(descriptor, "parameters", {}) or {}).items():
        if "default" in meta:
            arg_parts.append(f"{key}={meta['default']}")
    return f"{raw_action}({', '.join(arg_parts)})" if arg_parts else f"{raw_action}()"


def _default_timeline_for_behavior_selection(
    intent_id: str,
    display_name: str = "",
    brand: str = "",
    robot_type: str = "",
) -> BehaviorTimeline:
    """Build a default timeline that preserves semantic intent metadata."""
    descriptor = _semantic_descriptor_for_behavior(intent_id, display_name, brand, robot_type)
    if descriptor is None:
        return BehaviorTimeline()

    raw_action = str(getattr(descriptor, "raw_action", "") or "").strip()
    if not raw_action:
        return BehaviorTimeline()

    arg_parts: List[str] = []
    for key, meta in (getattr(descriptor, "parameters", {}) or {}).items():
        if "default" in meta:
            arg_parts.append(f"{key}={meta['default']}")
    timeline = build_timeline_from_modules(
        [SequenceModule(kind="movement", name=raw_action, args=", ".join(arg_parts), duration=1.0)],
        robot_type=robot_type or "go2",
        auto_decompose=True,
    )
    return timeline or BehaviorTimeline()

# =============================================================================
# Heartbeat Canvas 閳?data model (pure Python, no Qt dependency)
# =============================================================================

class HBNodeKind:
    """Canonical kind identifiers for heartbeat canvas nodes."""
    SOURCE           = "heartbeat_source"
    SENSOR_READ      = "sensor_read"
    SENSOR_TRANSFORM = "sensor_transform"
    MOVEMENT_BLEND   = "movement_blend"
    SAFETY_GATE      = "safety_gate"
    ALL: List[str] = [SOURCE, SENSOR_READ, SENSOR_TRANSFORM, MOVEMENT_BLEND, SAFETY_GATE]


_HB_NODE_COLORS: Dict[str, str] = {
    HBNodeKind.SOURCE:           "#1a3a5c",
    HBNodeKind.SENSOR_READ:      "#1a4a2e",
    HBNodeKind.SENSOR_TRANSFORM: "#2e1a4a",
    HBNodeKind.MOVEMENT_BLEND:   "#5c3a00",
    HBNodeKind.SAFETY_GATE:      "#5c1a1a",
}

_HB_NODE_LABELS: Dict[str, str] = {
    HBNodeKind.SOURCE:           "Heartbeat Source",
    HBNodeKind.SENSOR_READ:      "Sensor Read",
    HBNodeKind.SENSOR_TRANSFORM: "Transform",
    HBNodeKind.MOVEMENT_BLEND:   "Movement Blend",
    HBNodeKind.SAFETY_GATE:      "Safety Gate",
}

# -- Step 1.4: availability-state rendering maps for _refresh_hb_library() --
# Badges appended to node labels; empty string = no badge for available nodes.
_HB_CATALOG_BADGE: Dict[str, str] = {
    HBNodeAvailability.AVAILABLE:          "",
    HBNodeAvailability.LIMITED:            " [!]",
    HBNodeAvailability.UNSUPPORTED:        " [x]",
    HBNodeAvailability.UNKNOWN_CAPABILITY: " [?]",
}

# Tooltip text prefix; reason string appended when non-empty.
_HB_CATALOG_TOOLTIP: Dict[str, str] = {
    HBNodeAvailability.AVAILABLE:          "Supported by current robot profile",
    HBNodeAvailability.LIMITED:            "Limited support",
    HBNodeAvailability.UNSUPPORTED:        "Not supported",
    HBNodeAvailability.UNKNOWN_CAPABILITY: "Capability unknown 閳?load a capability profile to resolve",
}


@dataclass
class HBGraphNode:
    """Serializable representation of one heartbeat canvas node."""
    node_id: str
    kind: str
    label: str
    x: float
    y: float
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HBGraphNode:
        return cls(
            node_id=str(d.get("node_id", "")),
            kind=str(d.get("kind", HBNodeKind.SOURCE)),
            label=str(d.get("label", "")),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            params=dict(d.get("params", {})),
        )


@dataclass
class HBGraphEdge:
    """Serializable directed edge between two heartbeat canvas nodes."""
    edge_id: str
    source_id: str
    target_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HBGraphEdge:
        return cls(
            edge_id=str(d.get("edge_id", "")),
            source_id=str(d.get("source_id", "")),
            target_id=str(d.get("target_id", "")),
        )


@dataclass
class HBGraphData:
    """Serializable heartbeat canvas graph: nodes + directed edges."""
    nodes: List[HBGraphNode] = field(default_factory=list)
    edges: List[HBGraphEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HBGraphData:
        nodes = [HBGraphNode.from_dict(n) for n in d.get("nodes", [])]
        edges = [HBGraphEdge.from_dict(e) for e in d.get("edges", [])]
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def default_template(cls) -> HBGraphData:
        """Return a minimal starter graph: source -> sensor_read -> movement_blend."""
        src_id = "hb_src"
        sensor_id = "hb_imu"
        blend_id = "hb_blend"
        return cls(
            nodes=[
                HBGraphNode(node_id=src_id, kind=HBNodeKind.SOURCE,
                            label="Heartbeat Source", x=40.0, y=80.0),
                HBGraphNode(node_id=sensor_id, kind=HBNodeKind.SENSOR_READ,
                            label="IMU Read", x=240.0, y=80.0,
                            params={"sensor": "imu"}),
                HBGraphNode(node_id=blend_id, kind=HBNodeKind.MOVEMENT_BLEND,
                            label="Movement Blend", x=440.0, y=80.0),
            ],
            edges=[
                HBGraphEdge(edge_id="e0", source_id=src_id, target_id=sensor_id),
                HBGraphEdge(edge_id="e1", source_id=sensor_id, target_id=blend_id),
            ],
        )


# =============================================================================
# Heartbeat panel state containers (pure Python, no Qt)
# =============================================================================

@dataclass
class HBEventStateEntry:
    """One row in the Event State panel 閳?readable by non-technical users."""
    event_name: str
    state: str       # "active" | "inactive" | "pending" | "error"
    source: str
    timestamp: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_name": self.event_name,
            "state": self.state,
            "source": self.source,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HBEventStateEntry:
        return cls(
            event_name=str(d.get("event_name", "")),
            state=str(d.get("state", "inactive")),
            source=str(d.get("source", "")),
            timestamp=str(d.get("timestamp", "")),
            reason=str(d.get("reason", "")),
        )


@dataclass
class HBIOMapping:
    """One inbound/outbound signal <-> movement-param binding in the IO panel."""
    signal: str        # e.g. "imu.pitch", "imu.roll"
    target_param: str  # e.g. "posture.pitch_compensation"
    direction: str     # "inbound" | "outbound"
    status: str = "active"  # "active" | "unmapped" | "unsupported"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "target_param": self.target_param,
            "direction": self.direction,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HBIOMapping:
        return cls(
            signal=str(d.get("signal", "")),
            target_param=str(d.get("target_param", "")),
            direction=str(d.get("direction", "inbound")),
            status=str(d.get("status", "active")),
        )


@dataclass
class HBPolicyDraft:
    """Heartbeat policy fields editable in the authoring UI."""
    tick_ms: int = 100
    timeout_ms: int = 5000
    fail_policy: str = "stop_behavior"
    max_override: float = 0.2
    safety_mode: str = "strict"

    FAIL_POLICIES: ClassVar[Tuple[str, ...]] = ("stop_behavior", "continue", "warn_only")
    SAFETY_MODES: ClassVar[Tuple[str, ...]] = ("strict", "relaxed", "disabled")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_ms": self.tick_ms,
            "timeout_ms": self.timeout_ms,
            "fail_policy": self.fail_policy,
            "max_override": self.max_override,
            "safety_mode": self.safety_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HBPolicyDraft":
        return cls(
            tick_ms=int(d.get("tick_ms", 100)),
            timeout_ms=int(d.get("timeout_ms", 5000)),
            fail_policy=str(d.get("fail_policy", "stop_behavior")),
            max_override=float(d.get("max_override", 0.2)),
            safety_mode=str(d.get("safety_mode", "strict")),
        )

    @classmethod
    def default(cls) -> "HBPolicyDraft":
        return cls()


@dataclass
class HBDraftPayload:
    """Canonical heartbeat draft payload 閳?one per mission node context.

    Covers all authoring state: script, canvas graph, policy, IO bindings,
    event schema, and which authoring mode (canvas/script) was last active.
    Draft state (this class) is always kept separate from compiled/runtime
    snapshot state (``BehaviorPanel._hb_compiled_snapshot``).
    """
    script: str = ""
    canvas_graph: Dict[str, Any] = field(default_factory=dict)
    policy: HBPolicyDraft = field(default_factory=HBPolicyDraft)
    io_bindings: List[Dict[str, Any]] = field(default_factory=list)
    event_schema: List[Dict[str, Any]] = field(default_factory=list)
    authoring_mode: str = "navigator"  # "navigator" | "script" | "canvas"(legacy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script": self.script,
            "canvas_graph": dict(self.canvas_graph),
            "policy": self.policy.to_dict(),
            "io_bindings": [dict(b) for b in self.io_bindings],
            "event_schema": [dict(e) for e in self.event_schema],
            "authoring_mode": self.authoring_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HBDraftPayload":
        return cls(
            script=str(d.get("script", "")),
            canvas_graph=dict(d.get("canvas_graph") or {}),
            policy=HBPolicyDraft.from_dict(d.get("policy") or {}),
            io_bindings=[dict(b) for b in (d.get("io_bindings") or [])],
            event_schema=[dict(e) for e in (d.get("event_schema") or [])],
            authoring_mode=str(d.get("authoring_mode", "navigator")),
        )

    @classmethod
    def default(cls) -> "HBDraftPayload":
        return cls(
            script="# background heartbeat\nlisten_sensor(sensor='imu', interval=0.2)",
            canvas_graph=HBGraphData.default_template().to_dict(),
            policy=HBPolicyDraft.default(),
        )


# =============================================================================
# Heartbeat Canvas 閳?Qt graphics items
# =============================================================================

class HBNodeItem(QGraphicsRectItem):
    """Draggable, selectable heartbeat node rendered in HeartbeatCanvas."""

    NODE_W = 140.0
    NODE_H = 52.0

    def __init__(self, graph_node: HBGraphNode, parent=None):
        super().__init__(QRectF(0.0, 0.0, self.NODE_W, self.NODE_H))
        if parent is not None:
            self.setParentItem(parent)

        self.graph_node = graph_node
        self.setPos(graph_node.x, graph_node.y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        color = QColor(_HB_NODE_COLORS.get(graph_node.kind, "#1f2937"))
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor("#4b5563"), 1.5))

        self._label_item = QGraphicsTextItem(graph_node.label, self)
        self._label_item.setDefaultTextColor(QColor("#e5e7eb"))
        self._label_item.setPos(8.0, 5.0)

        kind_font = QFont()
        kind_font.setPointSize(7)
        self._kind_item = QGraphicsTextItem(graph_node.kind, self)
        self._kind_item.setDefaultTextColor(QColor("#9ca3af"))
        self._kind_item.setFont(kind_font)
        self._kind_item.setPos(8.0, 30.0)

        self._edges: List[HBEdgeItem] = []

    def register_edge(self, edge: HBEdgeItem) -> None:
        self._edges.append(edge)

    def unregister_edge(self, edge: HBEdgeItem) -> None:
        if edge in self._edges:
            self._edges.remove(edge)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.graph_node.x = self.x()
            self.graph_node.y = self.y()
            for e in list(self._edges):
                e.update_line()
        return super().itemChange(change, value)

    def center_right(self) -> QPointF:
        return QPointF(self.x() + self.NODE_W, self.y() + self.NODE_H / 2.0)

    def center_left(self) -> QPointF:
        return QPointF(self.x(), self.y() + self.NODE_H / 2.0)


class HBEdgeItem(QGraphicsLineItem):
    """Visual directed edge between two HBNodeItems."""

    def __init__(self, src: HBNodeItem, tgt: HBNodeItem, edge: HBGraphEdge):
        super().__init__()
        self._src = src
        self._tgt = tgt
        self._edge = edge
        self.setPen(QPen(QColor("#6b7280"), 1.5))
        self.setZValue(-1)
        self.update_line()
        src.register_edge(self)
        tgt.register_edge(self)

    def update_line(self) -> None:
        p0 = self._src.center_right()
        p1 = self._tgt.center_left()
        self.setLine(p0.x(), p0.y(), p1.x(), p1.y())

    @property
    def edge(self) -> HBGraphEdge:
        return self._edge

    @property
    def src_item(self) -> HBNodeItem:
        return self._src

    @property
    def tgt_item(self) -> HBNodeItem:
        return self._tgt


class HeartbeatCanvas(QGraphicsView):
    """Interactive canvas for heartbeat graph authoring.

    Features:
    - Draggable node items (heartbeat_source, sensor_read, sensor_transform,
      movement_blend, safety_gate)
    - Visual directed edges between nodes
    - Right-click context menu to add or delete nodes
    - Delete key to remove selected nodes
    - to_graph() / from_graph() for draft state persistence
    """

    graph_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor("#111827")))

        self._graph_data: HBGraphData = HBGraphData()
        self._node_items: Dict[str, HBNodeItem] = {}
        self._edge_items: List[HBEdgeItem] = []

        self.from_graph(HBGraphData.default_template())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, kind: str, label: str = "",
                 x: float = 100.0, y: float = 100.0) -> HBNodeItem:
        """Add a new heartbeat node at the given canvas position."""
        node_id = _uuid.uuid4().hex[:8]
        label = label or _HB_NODE_LABELS.get(kind, kind)
        gn = HBGraphNode(node_id=node_id, kind=kind, label=label, x=x, y=y)
        self._graph_data.nodes.append(gn)
        item = HBNodeItem(gn)
        self._scene.addItem(item)
        self._node_items[node_id] = item
        self.graph_changed.emit()
        return item

    def to_graph(self) -> HBGraphData:
        """Serialize current canvas state (sync positions from items first)."""
        for node_id, item in self._node_items.items():
            item.graph_node.x = item.x()
            item.graph_node.y = item.y()
        return self._graph_data

    def from_graph(self, data: HBGraphData) -> None:
        """Load an HBGraphData into the canvas, replacing current content."""
        self.clear_graph()
        self._graph_data = data
        for gn in data.nodes:
            item = HBNodeItem(gn)
            self._scene.addItem(item)
            self._node_items[gn.node_id] = item
        for ge in data.edges:
            src_item = self._node_items.get(ge.source_id)
            tgt_item = self._node_items.get(ge.target_id)
            if src_item and tgt_item:
                edge_item = HBEdgeItem(src_item, tgt_item, ge)
                self._scene.addItem(edge_item)
                self._edge_items.append(edge_item)

    def clear_graph(self) -> None:
        """Remove all items from the canvas."""
        self._scene.clear()
        self._graph_data = HBGraphData()
        self._node_items.clear()
        self._edge_items.clear()

    def reset_to_template(self) -> None:
        """Reset canvas to the default heartbeat template."""
        self.from_graph(HBGraphData.default_template())

    def apply_theme(self) -> None:
        bg = get_color("behavior_heartbeat_canvas_bg", "#111827")
        self.setBackgroundBrush(QBrush(QColor(bg)))

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def contextMenuEvent(self, event) -> None:
        """Right-click: delete node under cursor, or add a new node."""
        scene_pos = self.mapToScene(event.pos())
        item = self._scene.itemAt(scene_pos, self.transform())
        node_item = item if isinstance(item, HBNodeItem) else (
            item.parentItem() if item is not None and isinstance(item.parentItem(), HBNodeItem) else None
        )
        if node_item is not None:
            menu = QMenu(self)
            del_action = menu.addAction("Delete Node")
            chosen = menu.exec(event.globalPos())
            if chosen is del_action:
                self._remove_node(node_item)
            return

        menu = QMenu(self)
        menu.addSection("Add Heartbeat Node")
        actions: Dict[object, tuple] = {}
        for kind in HBNodeKind.ALL:
            label = _HB_NODE_LABELS.get(kind, kind)
            act = menu.addAction(label)
            actions[act] = (kind, float(scene_pos.x()), float(scene_pos.y()))
        chosen = menu.exec(event.globalPos())
        if chosen in actions:
            kind, x, y = actions[chosen]
            self.add_node(kind, x=x, y=y)

    def keyPressEvent(self, event) -> None:
        """Delete key removes all selected nodes."""
        if event.key() == Qt.Key.Key_Delete:
            for item in list(self._scene.selectedItems()):
                if isinstance(item, HBNodeItem):
                    self._remove_node(item)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_node(self, item: HBNodeItem) -> None:
        to_remove = [e for e in self._edge_items
                     if e.src_item is item or e.tgt_item is item]
        for edge in to_remove:
            edge.src_item.unregister_edge(edge)
            edge.tgt_item.unregister_edge(edge)
            self._scene.removeItem(edge)
            self._edge_items.remove(edge)
            eid = edge.edge.edge_id
            self._graph_data.edges = [
                e for e in self._graph_data.edges if e.edge_id != eid
            ]
        self._scene.removeItem(item)
        node_id = item.graph_node.node_id
        self._node_items.pop(node_id, None)
        self._graph_data.nodes = [
            n for n in self._graph_data.nodes if n.node_id != node_id
        ]
        self.graph_changed.emit()


class LimbDiagramView(QGraphicsView):
    """Placeholder canvas for the robot limb diagram.

    Displays a simple schematic of the active robot type using circles and
    rectangles.  Accepts ``update_time(float)`` to animate limb state as
    the timeline playhead moves.  Full kinematic animation is deferred to a
    future cycle; this class establishes the signal interface so the wiring
    is in place.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(self.renderHints())  # keep Qt default
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor("#1a1f2e")))
        self._robot_type: str = "go2"
        self._current_time: float = 0.0
        self._draw_placeholder()

    def set_robot_type(self, robot_type: str) -> None:
        self._robot_type = robot_type
        self._draw_placeholder()

    def update_time(self, t: float) -> None:
        """Slot: called by playback timer with current playhead time (seconds)."""
        self._current_time = t
        # Full animation deferred; redraw placeholder to show time is received.
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self._scene.clear()
        w, h = 260.0, 180.0
        self._scene.setSceneRect(0, 0, w, h)

        # Central body rectangle
        body_pen = QPen(QColor("#4a90d9"))
        body_pen.setWidth(2)
        body_brush = QBrush(QColor("#1e2d45"))
        body = self._scene.addRect(QRectF(w / 2 - 30, h / 2 - 40, 60, 80), body_pen, body_brush)

        # Head circle
        head_pen = QPen(QColor("#7ec8e3"))
        head_pen.setWidth(2)
        head_brush = QBrush(QColor("#1e2d45"))
        self._scene.addEllipse(QRectF(w / 2 - 16, h / 2 - 68, 32, 28), head_pen, head_brush)

        # Four legs as lines (simplified for go2 / quadruped)
        leg_pen = QPen(QColor("#4a90d9"))
        leg_pen.setWidth(3)
        leg_coords = [
            (w / 2 - 30, h / 2 - 20, w / 2 - 50, h / 2 + 50),
            (w / 2 + 30, h / 2 - 20, w / 2 + 50, h / 2 + 50),
            (w / 2 - 30, h / 2 + 30, w / 2 - 50, h / 2 + 90),
            (w / 2 + 30, h / 2 + 30, w / 2 + 50, h / 2 + 90),
        ]
        for x1, y1, x2, y2 in leg_coords:
            self._scene.addLine(x1, y1, x2, y2, leg_pen)
            # Foot circle
            foot_brush = QBrush(QColor("#4a90d9"))
            self._scene.addEllipse(QRectF(x2 - 5, y2 - 5, 10, 10), leg_pen, foot_brush)

        # Time indicator
        time_item = self._scene.addText(f"t = {self._current_time:.2f} s")
        time_item.setDefaultTextColor(QColor("#6b7280"))
        time_item.setPos(4, 4)

        # Robot type label
        type_item = self._scene.addText(self._robot_type)
        type_item.setDefaultTextColor(QColor("#4a90d9"))
        type_item.setPos(4, h - 22)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class BehaviorPanel(QWidget):
    """Behavior tab with timeline-focused sequence authoring."""

    back_requested = Signal()
    action_parameters_changed = Signal(int)
    playback_time_changed = Signal(float)  # emitted on every playback tick (seconds)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node_name = ""
        self._behavior_ref = ""
        self._intent_id = ""
        self._node_id = -1
        self._bridge = BehaviorCompilerBridge()
        self._compile_worker = None

        self._dirty_nodes: set = set()
        self._suspend_dirty_tracking = False
        # Per-node+ref transient draft state. Key format: "<node_id>::<ref>".
        # Legacy payloads keyed only by node_id are migrated on first access.
        self._drafts_by_node: Dict[str, Dict[str, Any]] = {}
        # Snapshot taken at set_node_context time; used by Reset to restore initial state.
        self._init_snapshot: Optional[Dict[str, Any]] = None

        self._movement_registry: List[str] = []
        self._movement_descriptors: List[SemanticActionDescriptor] = []  # Circle 5
        self._behavior_registry: List[str] = []
        self._modules: List[SequenceModule] = []
        self._selected_module_index: int = -1  # for module ops (remove/reorder); not selection display
        self._movement_track_groups_expanded: Dict[str, bool] = {}
        self._movement_group_header_rows: Dict[int, str] = {}
        # Single authoritative selection state 閳?always kept in sync with _timeline_vm.selection.
        # Replaces the previous scattered _selected_limb_id / _selected_body_part_action_id /
        # _selected_motor_segment_id / _selected_motor_track fields.
        from system.behavior.timeline_view_model import TimelineSelection, SelectionLevel
        self._panel_selection: "TimelineSelection" = TimelineSelection.none()
        self._SelectionLevel = SelectionLevel  # cached for use without re-import

        # Phase 1 redesign: structured timeline per node (None until first sync)
        # _behavior_timeline  閳?active structured model for the current node
        # _timelines_by_node  閳?persisted per-node timelines (key format same as _drafts_by_node)
        self._behavior_timeline: Optional[BehaviorTimeline] = None
        self._timelines_by_node: Dict[str, Dict[str, Any]] = {}

        # Fix 1: robot type and simulation mode (updated by set_capability_profile /
        # set_simulation_mode from bin/ui.py scenario settings wiring).
        self._brand: str = ""
        self._robot_type: str = "go2"
        self._is_simulation: bool = False

        # Runtime/compile snapshot state 閳?separate from draft
        self._hb_compiled_snapshot: Optional[Dict[str, Any]] = None
        # Event state and IO mapping panels 閳?populated by external callers or draft restore
        self._hb_event_states: List[HBEventStateEntry] = []
        self._hb_io_mappings: List[HBIOMapping] = []
        # Dual-mode authoring state (Step 1.2)
        self._hb_authoring_mode: str = "navigator"  # "navigator" | "script" | "canvas"(legacy)
        # Productized panel no longer depends on the legacy internal canvas widget.
        # Keep a dict slot only so old draft payloads with "canvas_graph" round-trip.
        self._hb_legacy_canvas_graph: Dict[str, Any] = HBGraphData.default_template().to_dict()
        self._hb_policy_draft: HBPolicyDraft = HBPolicyDraft.default()
        # Execution chain channel 閳?mock by default; swapped via set_hb_channel() (Step 1.3)
        self._hb_channel: IHBChannel = HBMockChannel()
        # Step 1.4: Node catalog 閳?model-aware availability; starts as unknown
        self._hb_catalog: HBNodeCatalog = HBNodeCatalog.unknown()

        # Playback state (timeline preview playback)
        self._playback_time: float = 0.0
        self._is_playing: bool = False

        self._init_ui()
        # Diagnostics poll timer 閳?started on first set_node_context() call (Step 1.3)
        self._hb_diag_timer = QTimer(self)
        self._hb_diag_timer.setInterval(2000)
        self._hb_diag_timer.timeout.connect(self._poll_hb_diagnostics)
        # Playback timer — 50 ms tick → ~20 fps scrub rate
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(50)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._refresh_registries()
        self._refresh_module_library()
        self._refresh_hb_library()
        self._refresh_timeline()
        self._refresh_navigator()
        self._refresh_movement_settings()
        self._refresh_event_state()
        self._refresh_movement_io_mapping()
        self.refresh_texts()

    def set_node_context(
        self,
        node_name: str,
        node_id: int,
        behavior_ref: str = "",
        intent_id: str = "",
    ) -> None:
        # Save current node's full canonical draft before switching
        if self._node_id >= 0:
            self._persist_current_behavior_draft()

        self._node_name = node_name
        self._behavior_ref = str(behavior_ref or "").strip() or node_name
        self._intent_id = str(intent_id or "").strip()
        self._node_id = node_id
        self._refresh_breadcrumb()
        self._ctx_node.setText(node_name)
        self._ctx_node_id.setText(str(node_id))
        self._ctx_ref.setText(self._behavior_ref)

        slot_key = self._draft_slot_key(node_id, self._behavior_ref)
        saved = self._drafts_by_node.get(slot_key, {})
        if not saved:
            # Backward compat: migrate legacy node-id-only drafts to the
            # currently selected ref once, so action switching no longer
            # reuses one stale draft across all refs.
            legacy = self._drafts_by_node.get(str(node_id)) or self._drafts_by_node.get(node_id)  # type: ignore[arg-type]
            if isinstance(legacy, dict):
                saved = dict(legacy)
                self._drafts_by_node[slot_key] = saved
                self._drafts_by_node.pop(str(node_id), None)
                self._drafts_by_node.pop(node_id, None)  # type: ignore[arg-type]
        core = saved.get(
            "core",
            _default_core_source_for_behavior_selection(self._intent_id, node_name, self._brand, self._robot_type),
        )
        hb_dict = saved.get("hb")

        self._set_core_source(core, mark_dirty=False)
        # Restore HB draft (lossless: both canvas and script restored independently)
        if hb_dict is not None:
            try:
                self._apply_hb_draft(HBDraftPayload.from_dict(hb_dict))
            except Exception:
                self._apply_hb_draft(HBDraftPayload.default())
        else:
            self._apply_hb_draft(HBDraftPayload.default())
        self._hb_status_label.setText("Draft 閳?uncompiled")
        self._hb_compiled_snapshot = None
        self._hb_diag_indicator.setText("")
        # _hb_event_states / _hb_io_mappings / mode already reset by _apply_hb_draft

        # Restore structured timeline for this node using the draft as the
        # single persisted behavior state. Legacy external timeline registries are
        # only consulted as a migration fallback.
        timeline_dict = self._timeline_dict_for_slot(slot_key)
        if timeline_dict is not None:
            self._behavior_timeline = self._load_timeline_with_migration(timeline_dict)
            if self._behavior_timeline.is_empty() and not timeline_dict.get("action_segments"):
                self._behavior_timeline = None
        else:
            self._behavior_timeline = _default_timeline_for_behavior_selection(
                self._intent_id,
                node_name,
                self._brand,
                self._robot_type,
            )

        self._sync_views_from_timeline(mark_dirty=False)
        self._refresh_registries()
        self._refresh_module_library()
        self._refresh_timeline()
        self._refresh_navigator()

        # Record init snapshot for Reset 閳?captures state right after load.
        self._init_snapshot = {
            "core": self._core_editor.toPlainText(),
            "hb": self._collect_hb_draft().to_dict(),
            "timeline": self._behavior_timeline.to_dict() if self._behavior_timeline is not None else None,
        }

        # Start diagnostics poll timer when a node is active (Step 1.3)
        if not self._hb_diag_timer.isActive():
            self._hb_diag_timer.start()
        self._refresh_save_button_state()

    @staticmethod
    def _draft_slot_key(node_id: int, node_name: str) -> str:
        return f"{int(node_id)}::{str(node_name or '').strip()}"

    def ensure_behavior_artifact(
        self,
        *,
        node_name: str,
        node_id: int,
        behavior_ref: str,
        intent_id: str,
        brand: str,
        robot_type: str,
        is_simulation: bool,
    ):
        """Compile/register the artifact for one Behavior node using semantic defaults."""
        ref = str(behavior_ref or "").strip() or str(node_name or "").strip()
        slot_key = self._draft_slot_key(node_id, ref)
        saved = self._drafts_by_node.get(slot_key, {})

        source = str(
            saved.get("core")
            or _default_core_source_for_behavior_selection(intent_id, node_name, brand, robot_type)
        )

        timeline_raw = self._timeline_dict_for_slot(slot_key)
        if isinstance(timeline_raw, dict):
            timeline = self._load_timeline_with_migration(timeline_raw)
        else:
            timeline = _default_timeline_for_behavior_selection(intent_id, node_name, brand, robot_type)

        return self._bridge.compile(
            source=source,
            behavior_ref=ref,
            robot_type=robot_type or "go2",
            timeline=timeline,
            is_simulation=bool(is_simulation),
        )


    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty_nodes)

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_content(), 1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("behaviorHeader")
        header.setFixedHeight(40)
        header.setStyleSheet("border-radius: 0px;")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 4, 8, 4)
        row.setSpacing(6)

        self._breadcrumb = QLabel("")
        self._breadcrumb.setObjectName("behaviorBreadcrumb")
        self._breadcrumb.setStyleSheet("background: transparent; border-radius: 0px;")
        row.addWidget(self._breadcrumb)
        row.addStretch()

        self._back_btn = QPushButton("")
        self._back_btn.setObjectName("behaviorBackBtn")
        self._back_btn.setFixedHeight(28)
        self._back_btn.clicked.connect(self.back_requested.emit)
        row.addWidget(self._back_btn)
        return header

    def _build_content(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_library_pane())
        splitter.addWidget(self._build_main_pane())
        splitter.setSizes([250, 950])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_library_pane(self) -> QFrame:
        pane = QFrame()
        pane.setObjectName("behaviorLeftPane")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 閳光偓閳光偓 Context region 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        self._context_title = _plain_title(layout, "")
        self._ctx_node_key, self._ctx_node = _kv_row(layout, "", "-")
        self._ctx_node_id_key, self._ctx_node_id = _kv_row(layout, "", "-")
        self._ctx_ref_key, self._ctx_ref = _kv_row(layout, "", "-")
        self._ctx_artifact_key, self._ctx_artifact = _kv_row(layout, "", "-")
        self._ctx_compile_time_key, self._ctx_compile_time = _kv_row(layout, "", "-")
        self._ctx_diag_key, self._ctx_diag = _kv_row(layout, "", "-")

        layout.addSpacing(14)
        self._context_core_divider = QFrame()
        self._context_core_divider.setFrameShape(QFrame.Shape.HLine)
        self._context_core_divider.setFrameShadow(QFrame.Shadow.Plain)
        self._context_core_divider.setStyleSheet("color: #303030; background: #303030; min-height: 1px; max-height: 1px;")
        layout.addWidget(self._context_core_divider)
        layout.addSpacing(14)

        # 閳光偓閳光偓 Core Library region (movements + behavior refs) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        self._core_library_title = _plain_title(layout, "")
        self._module_library_tree = QTreeWidget()
        self._module_library_tree.setHeaderHidden(True)
        self._module_library_tree.setIndentation(12)
        self._module_library_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._module_library_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._module_library_tree.customContextMenuRequested.connect(
            self._on_module_library_context_menu
        )
        self._module_library_tree.installEventFilter(self)
        self._module_library_tree.itemDoubleClicked.connect(
            lambda item, _column: self._insert_module_from_library(item)
        )
        layout.addWidget(self._module_library_tree, 1)

        layout.addSpacing(4)

        # 閳光偓閳光偓 HeartBeat Library region 閳?hidden in product UI (not part of main flow) 閳光偓閳光偓
        self._hb_library_title = _plain_title(layout, "")
        self._hb_library_tree = QTreeWidget()
        self._hb_library_tree.setHeaderHidden(True)
        self._hb_library_tree.setIndentation(12)
        self._hb_library_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._hb_library_tree.itemDoubleClicked.connect(
            lambda item, _column: self._insert_hb_node_from_library(item)
        )
        layout.addWidget(self._hb_library_tree, 1)
        self._hb_library_title.hide()
        self._hb_library_tree.hide()

        # Compatibility alert bar 閳?Step 1.7 (hidden until issues detected)
        self._compat_summary_label = QLabel("")
        self._compat_summary_label.setObjectName("hbCompatSummaryLabel")
        self._compat_summary_label.setWordWrap(True)
        self._compat_summary_label.setStyleSheet(
            "font-size: 11px; padding: 2px 4px; border-radius: 3px;"
        )
        self._compat_summary_label.hide()
        layout.addWidget(self._compat_summary_label)

        self._compat_issues_list = QListWidget()
        self._compat_issues_list.setObjectName("hbCompatIssuesList")
        self._compat_issues_list.setMaximumHeight(96)
        self._compat_issues_list.setStyleSheet("font-size: 10px;")
        self._compat_issues_list.itemClicked.connect(self._on_compat_issue_clicked)
        self._compat_issues_list.hide()
        layout.addWidget(self._compat_issues_list)

        self._apply_module_library_tree_style()
        return pane

    def _build_main_pane(self) -> QWidget:
        wrapper = QWidget()
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(0)

        # Vertical splitter: top = Limb Diagram (user-resizable), bottom = rest.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        limb_diagram = self._build_limb_diagram_section()
        splitter.addWidget(limb_diagram)

        # Lower pane: timeline + bottom tabs stacked vertically.
        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(0, 6, 0, 0)
        lower_layout.setSpacing(6)

        timeline = self._build_timeline_section()
        lower_layout.addWidget(timeline, 0)

        # Bottom region: Movement Settings+IO (tab 0) + hidden compat tabs.
        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.addTab(self._build_movement_io_tab(), "")   # tab 0
        self._bottom_tabs.addTab(self._build_event_state_tab(), "")   # tab 1 (hidden)
        self._bottom_tabs.addTab(self._build_source_diag_section(), "")  # tab 2 (hidden)
        self._bottom_tabs.setTabVisible(1, False)
        self._bottom_tabs.setTabVisible(2, False)
        self._bottom_tabs.tabBar().hide()  # Single visible tab – no bar needed
        lower_layout.addWidget(self._bottom_tabs, 2)

        splitter.addWidget(lower)
        # Initial size ratio: Limb diagram ~30%, lower area ~70%.
        splitter.setSizes([200, 400])

        root.addWidget(splitter, 1)
        return wrapper

    def _build_movement_io_tab(self) -> QWidget:
        """Tab 0: left = tabbed settings (Parameters / IO Mappings), right = Navigator."""
        pane = QWidget()
        layout = QHBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left side: QTabWidget with Parameters and IO Mappings tabs
        self._movement_settings_tabs = QTabWidget()
        self._movement_settings_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        params_tab = self._build_movement_settings_section()
        io_tab = self._build_io_mapping_section()
        self._movement_settings_tabs.addTab(params_tab, "Parameters")
        self._movement_settings_tabs.addTab(io_tab, "IO Mappings")
        layout.addWidget(self._movement_settings_tabs, 2)

        # Right side: Navigator (heartbeat section)
        heartbeat = self._build_heartbeat_section()
        heartbeat.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(heartbeat, 1)
        return pane

    def _build_event_state_tab(self) -> QFrame:
        """Tab 1: event state panel 閳?event name, state, source, timestamp, reason."""
        pane = QFrame()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._event_state_title = _plain_title(layout, "")

        # Summary label: "N events | N active | N errors" (Step 1.5)
        self._event_state_summary_label = QLabel("")
        self._event_state_summary_label.setObjectName("hbEventSummaryLabel")
        self._event_state_summary_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        layout.addWidget(self._event_state_summary_label)

        self._event_state_table = QTableWidget(0, 5)
        self._event_state_table.setHorizontalHeaderLabels(["", "", "", "", ""])
        self._event_state_table.verticalHeader().setVisible(False)
        self._event_state_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._event_state_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._event_state_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self._event_state_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents)
        self._event_state_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)
        self._event_state_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._event_state_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._event_state_table, 1)
        return pane

    def _build_io_mapping_section(self) -> QFrame:
        """Movement IO mapping: inbound sensor/signal -> movement param bindings."""
        pane = QFrame()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(4)

        self._io_mapping_title = _plain_title(layout, "")

        self._io_mapping_table = QTableWidget(0, 4)
        self._io_mapping_table.setHorizontalHeaderLabels(["", "", "", ""])
        self._io_mapping_table.verticalHeader().setVisible(False)
        self._io_mapping_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._io_mapping_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._io_mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self._io_mapping_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents)
        self._io_mapping_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._io_mapping_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._io_mapping_table, 1)

        # Conflict / warning hint label 閳?hidden until conflicts exist (Step 1.5)
        self._io_conflict_label = QLabel("")
        self._io_conflict_label.setObjectName("hbIOConflictLabel")
        self._io_conflict_label.setStyleSheet(
            "color: #ff9800; font-size: 11px; padding: 2px 0;"
        )
        self._io_conflict_label.setWordWrap(True)
        self._io_conflict_label.hide()
        layout.addWidget(self._io_conflict_label)
        return pane

    def _build_heartbeat_section(self) -> QFrame:
        pane = QFrame()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(4)
        self._hb_canvas_btn = QPushButton("")
        self._hb_canvas_btn.setCheckable(True)
        self._hb_canvas_btn.setChecked(False)
        self._hb_compiler_btn = QPushButton("")
        self._hb_compiler_btn.setCheckable(True)
        self._reset_btn = QPushButton("")
        self._reset_btn.setObjectName("behaviorResetBtn")
        self._reset_btn.setFixedHeight(28)
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._on_reset)

        # Legacy canvas/compiler/navigator toggles hidden in product UI.
        row.addWidget(self._hb_canvas_btn)
        row.addWidget(self._hb_compiler_btn)
        row.addStretch()
        row.addWidget(self._reset_btn)
        layout.addLayout(row)
        self._hb_canvas_btn.hide()
        self._hb_compiler_btn.hide()  # Script mode retired from product UI

        # Execution chain buttons, status label, and policy row are retained as attributes
        # for backend compat (compile/dryrun/run logic still works internally) but are not
        # shown in the product UI 閳?Behavior panel is now "Navigator + Timeline + Settings".
        _hidden_ctrl = QWidget()
        _hidden_ctrl.hide()
        _hidden_layout = QVBoxLayout(_hidden_ctrl)
        _hidden_layout.setContentsMargins(0, 0, 0, 0)

        exec_row = QHBoxLayout()
        exec_row.setSpacing(4)
        self._hb_compile_btn = QPushButton("")
        self._hb_compile_btn.setFixedHeight(24)
        self._hb_compile_btn.clicked.connect(self._on_hb_compile)
        exec_row.addWidget(self._hb_compile_btn)
        self._hb_simulate_btn = QPushButton("")
        self._hb_simulate_btn.setFixedHeight(24)
        self._hb_simulate_btn.clicked.connect(self._on_hb_dryrun)
        exec_row.addWidget(self._hb_simulate_btn)
        self._hb_run_btn = QPushButton("")
        self._hb_run_btn.setFixedHeight(24)
        self._hb_run_btn.clicked.connect(self._on_hb_run)
        exec_row.addWidget(self._hb_run_btn)
        exec_row.addStretch()
        self._hb_diag_indicator = QLabel("")
        self._hb_diag_indicator.setStyleSheet(
            "color: #6b7280; font-size: 10px; background: transparent;"
        )
        exec_row.addWidget(self._hb_diag_indicator)
        _hidden_layout.addLayout(exec_row)

        self._hb_status_label = QLabel("Draft 閳?uncompiled")
        self._hb_status_label.setStyleSheet(
            "color: #9ca3af; font-size: 11px; background: transparent;"
        )
        _hidden_layout.addWidget(self._hb_status_label)

        # Policy row 閳?tick_ms, timeout_ms, fail_policy (retained for persistence compat)
        _lbl_style = "color: #9ca3af; font-size: 11px; background: transparent;"
        policy_row = QHBoxLayout()
        policy_row.setSpacing(4)
        _tick_lbl = QLabel("Tick")
        _tick_lbl.setStyleSheet(_lbl_style)
        policy_row.addWidget(_tick_lbl)
        self._hb_policy_tick_spin = QSpinBox()
        self._hb_policy_tick_spin.setRange(10, 10000)
        self._hb_policy_tick_spin.setValue(100)
        self._hb_policy_tick_spin.setSuffix(" ms")
        self._hb_policy_tick_spin.setMaximumWidth(90)
        self._hb_policy_tick_spin.valueChanged.connect(self._on_policy_changed)
        policy_row.addWidget(self._hb_policy_tick_spin)
        policy_row.addSpacing(6)
        _to_lbl = QLabel("Timeout")
        _to_lbl.setStyleSheet(_lbl_style)
        policy_row.addWidget(_to_lbl)
        self._hb_policy_timeout_spin = QSpinBox()
        self._hb_policy_timeout_spin.setRange(100, 300000)
        self._hb_policy_timeout_spin.setValue(5000)
        self._hb_policy_timeout_spin.setSuffix(" ms")
        self._hb_policy_timeout_spin.setMaximumWidth(100)
        self._hb_policy_timeout_spin.valueChanged.connect(self._on_policy_changed)
        policy_row.addWidget(self._hb_policy_timeout_spin)
        policy_row.addSpacing(6)
        _fp_lbl = QLabel("On fail")
        _fp_lbl.setStyleSheet(_lbl_style)
        policy_row.addWidget(_fp_lbl)
        self._hb_policy_fail_combo = QComboBox()
        for _fp in HBPolicyDraft.FAIL_POLICIES:
            self._hb_policy_fail_combo.addItem(_fp)
        self._hb_policy_fail_combo.currentTextChanged.connect(self._on_policy_changed)
        policy_row.addWidget(self._hb_policy_fail_combo)
        policy_row.addStretch()
        _hidden_layout.addLayout(policy_row)
        layout.addWidget(_hidden_ctrl)

        self._hb_stack = QStackedWidget()
        # Legacy canvas view retired: keep an inert placeholder only to preserve
        # index compatibility for old authoring_mode values ("canvas").
        self._hb_canvas_placeholder = QWidget()
        self._hb_stack.addWidget(self._hb_canvas_placeholder)

        self._heartbeat_editor = QTextEdit()
        self._heartbeat_editor.setAcceptRichText(False)
        self._heartbeat_editor.setPlaceholderText("")
        self._heartbeat_editor.textChanged.connect(self._on_editor_changed)
        self._hb_stack.addWidget(self._heartbeat_editor)

        # Step 2: Motor Weight Navigator 閳?read-only inspection mode (index 2)
        self._motor_weight_navigator = MotorWeightNavigator()
        self._hb_stack.addWidget(self._motor_weight_navigator)

        layout.addWidget(self._hb_stack, 1)

        self._hb_canvas_btn.clicked.connect(lambda: self._set_hb_mode(0))
        self._hb_compiler_btn.clicked.connect(lambda: self._set_hb_mode(1))
        # Product default: navigator is the only visible mode.
        self._set_hb_mode(2)
        return pane

    def _build_limb_diagram_section(self) -> QFrame:
        """Top panel: robot limb schematic that animates with the timeline playhead."""
        pane = QFrame()
        pane.setObjectName("limbDiagramSection")
        pane.setStyleSheet("#limbDiagramSection { border: 1px solid #2d3748; border-radius: 4px; }")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        lbl = QLabel("Limb Diagram")
        lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        header.addWidget(lbl)
        header.addStretch()
        layout.addLayout(header)

        self._limb_diagram_view = LimbDiagramView()
        self._limb_diagram_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._limb_diagram_view, 1)
        return pane

    def _build_timeline_section(self) -> QFrame:
        pane = QFrame()
        pane.setMinimumHeight(250)
        pane.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        top = QHBoxLayout()
        self._timeline_title = _plain_title(top, "")
        top.addStretch()
        layout.addLayout(top)

        self._timeline_status_label = QLabel("")
        self._timeline_status_label.setWordWrap(True)
        self._timeline_status_label.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 0 2px 4px 2px;"
        )
        layout.addWidget(self._timeline_status_label)

        # Playback controls row — video-editor style
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        ctrl_row.setContentsMargins(4, 0, 4, 2)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(28, 24)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._play_cmd_toggle)
        ctrl_row.addWidget(self._play_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedSize(28, 24)
        self._stop_btn.setToolTip("Stop and rewind")
        self._stop_btn.clicked.connect(self._play_cmd_stop)
        ctrl_row.addWidget(self._stop_btn)

        ctrl_row.addSpacing(6)
        self._play_time_label = QLabel("0.00 s")
        self._play_time_label.setStyleSheet(
            "color: #94a3b8; font-size: 11px; font-family: monospace; background: transparent;"
        )
        ctrl_row.addWidget(self._play_time_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self._timeline = TimelineView()
        self._timeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._timeline.module_selected.connect(self._on_timeline_module_selected)
        self._timeline.motor_segment_selected.connect(self._on_timeline_motor_segment_selected)
        self._timeline.module_reordered.connect(self._on_timeline_module_reordered)
        self._timeline.delete_requested.connect(self._remove_selected_module)
        self._timeline.timeline_edited.connect(self._on_timeline_edited)
        self._timeline.body_part_selected.connect(self._on_timeline_body_part_selected)
        # Update time label when user scrubs playhead directly on ruler
        self._timeline.playhead_moved.connect(
            lambda t: self._play_time_label.setText(f"{t:.2f} s")
        )
        layout.addWidget(self._timeline, 1)

        return pane

    def _build_movement_settings_section(self) -> QFrame:
        pane = QFrame()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._movement_settings_title = _plain_title(layout, "")
        self._movement_selected_label = QLabel("")
        self._movement_selected_label.setWordWrap(True)
        self._movement_selected_label.setStyleSheet(
            "color: #e2e8f0; font-weight: 600; padding: 0 0 2px 0;"
        )
        layout.addWidget(self._movement_selected_label)

        self._movement_params_table = QTableWidget(0, 3)   # Step 3: Parameter/Source/Value
        self._movement_params_table.setHorizontalHeaderLabels(["Parameter", "Source", "Value"])
        self._movement_params_table.verticalHeader().setVisible(False)
        self._movement_params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._movement_params_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._movement_params_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._movement_params_table.horizontalHeader().setStretchLastSection(False)
        self._movement_params_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._movement_params_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._movement_params_table.cellClicked.connect(self._on_movement_param_cell_clicked)
        layout.addWidget(self._movement_params_table, 1)

        self._movement_hint = QLabel("")
        self._movement_hint.setWordWrap(True)
        self._movement_hint.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 2px 0 0 0;"
        )
        layout.addWidget(self._movement_hint)
        return pane

    def _build_source_diag_section(self) -> QFrame:
        pane = QFrame()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._core_source_title = _plain_title(layout, "")
        self._core_editor = QTextEdit()
        self._core_editor.setAcceptRichText(False)
        self._core_editor.setPlaceholderText("")
        self._core_editor.textChanged.connect(self._on_editor_changed)
        layout.addWidget(self._core_editor, 1)

        self._diagnostics_title = _plain_title(layout, "")
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFixedHeight(90)
        layout.addWidget(self._output)
        return pane

    def _set_hb_mode(self, idx: int) -> None:
        self._hb_stack.setCurrentIndex(idx)
        self._hb_canvas_btn.setChecked(idx == 0)
        self._hb_compiler_btn.setChecked(idx == 1)
        if idx == 0:
            self._hb_authoring_mode = "canvas"
        elif idx == 1:
            self._hb_authoring_mode = "script"
        else:
            self._hb_authoring_mode = "navigator"
        if idx == 2:                                    # Step 2: refresh on switch
            self._refresh_navigator()

    def _refresh_registries(self) -> None:
        try:
            self._behavior_registry = sorted(self._bridge.list_refs())
        except Exception:
            self._behavior_registry = []

        # Circle 5 STEP 5.1: source Movement library from central semantic registry.
        # brand="" / robot_type="" returns the brand-independent base catalog 閳?
        # no hardcoded fallback string list needed.
        self._movement_descriptors = [d for d in get_semantic_actions(self._brand, self._robot_type) if d.availability == ActionAvailability.AVAILABLE]
        self._movement_registry = [d.raw_action for d in self._movement_descriptors]

    def _refresh_module_library(self) -> None:
        self._module_library_tree.clear()

        movement_root = QTreeWidgetItem([tr("behavior.library.movement", "System")])
        movement_root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        movement_root.setExpanded(True)
        self._module_library_tree.addTopLevelItem(movement_root)
        for desc in self._movement_descriptors:
            label = desc.display_name or desc.raw_action
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "kind": "movement",
                "name": desc.raw_action,
                "intent_id": desc.intent_id,
                "availability": desc.availability,
            })
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            movement_root.addChild(item)

        behavior_root = QTreeWidgetItem([tr("behavior.library.behavior", "Customs")])
        behavior_root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        behavior_root.setExpanded(True)
        self._module_library_tree.addTopLevelItem(behavior_root)
        for name in self._behavior_registry:
            item = QTreeWidgetItem([f"behavior.{name}()"])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "behavior", "name": name, "args": ""})
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            behavior_root.addChild(item)

        self._module_library_tree.expandAll()

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_module_library_tree", None):
            if (
                event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Delete
            ):
                if self._delete_selected_custom_behavior_ref():
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def _is_custom_behavior_item(self, item: Optional[QTreeWidgetItem]) -> bool:
        if item is None:
            return False
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        return data.get("kind") == "behavior" and bool(str(data.get("name") or "").strip())

    def _delete_selected_custom_behavior_ref(self) -> bool:
        item = self._module_library_tree.currentItem()
        if not self._is_custom_behavior_item(item):
            return False
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        ref = str(data.get("name") or "").strip()
        if not ref:
            return False
        if not self._bridge.delete_ref(ref):
            return False
        self._refresh_registries()
        self._refresh_module_library()
        return True

    def _on_module_library_context_menu(self, pos) -> None:
        item = self._module_library_tree.itemAt(pos)
        if not self._is_custom_behavior_item(item):
            return
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        src_ref = str(data.get("name") or "").strip()
        if not src_ref:
            return

        menu = QMenu(self._module_library_tree)
        dup_action = menu.addAction("Duplicate")
        del_action = menu.addAction("Delete")
        chosen = menu.exec(self._module_library_tree.viewport().mapToGlobal(pos))
        if chosen is dup_action:
            new_ref = self._bridge.duplicate_ref(src_ref)
            if new_ref:
                self._refresh_registries()
                self._refresh_module_library()
        elif chosen is del_action:
            if self._bridge.delete_ref(src_ref):
                self._refresh_registries()
                self._refresh_module_library()

    def _refresh_hb_library(self) -> None:
        """Refresh the HeartBeat Library region with model-aware availability states.

        All catalog entries are always shown 閳?nodes are never silently hidden.
        The availability state (available / limited / unsupported / unknown) is
        conveyed via a text badge and tooltip on each item (Step 1.4).
        """
        self._hb_library_tree.clear()
        root = QTreeWidgetItem([tr("behavior.hb_library.nodes", "Heartbeat Nodes")])
        root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        root.setExpanded(True)
        self._hb_library_tree.addTopLevelItem(root)

        for entry in self._hb_catalog.all_entries():
            badge = _HB_CATALOG_BADGE.get(entry.availability, " [?]")
            display_label = entry.label + badge
            item = QTreeWidgetItem([display_label])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": entry.kind, "label": entry.label})
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            # Build tooltip 閳?always explicit, never silent
            prefix = _HB_CATALOG_TOOLTIP.get(entry.availability, "Capability unknown")
            tooltip = (prefix + ": " + entry.reason) if entry.reason else prefix
            item.setToolTip(0, tooltip)

            root.addChild(item)

        self._hb_library_tree.expandAll()

    def _refresh_event_state(self) -> None:
        """Refresh the Event State panel with status badges (Step 1.5)."""
        self._event_state_table.setRowCount(0)
        if not self._hb_event_states:
            # Show placeholder row so the panel is never silently empty
            self._event_state_table.insertRow(0)
            placeholder = QTableWidgetItem(
                tr("hb.no_events", "No events recorded")
            )
            placeholder.setForeground(QColor("#9e9e9e"))
            self._event_state_table.setItem(0, 0, placeholder)
            self._event_state_table.setSpan(0, 0, 1, 5)
            self._event_state_summary_label.setText("")
            return

        summary = compute_event_summary(self._hb_event_states)
        self._event_state_summary_label.setText(
            tr(
                "hb.event_summary",
                "{total} events  |  {active} active  |  {errors} errors",
                total=summary.total,
                active=summary.active,
                errors=summary.errors,
            )
        )

        for entry in self._hb_event_states:
            row = self._event_state_table.rowCount()
            self._event_state_table.insertRow(row)

            # Col 0: event_name (plain)
            self._event_state_table.setItem(row, 0, QTableWidgetItem(entry.event_name))

            # Col 1: state badge (colored background)
            badge = badge_for_event_state(entry.state)
            bg_hex = _BADGE_COLOR_MAP.get(badge.color_key, "#757575")
            state_item = QTableWidgetItem(badge.label)
            state_item.setBackground(QColor(bg_hex))
            state_item.setForeground(QColor("#ffffff"))
            state_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._event_state_table.setItem(row, 1, state_item)

            # Col 2閳?: source, timestamp, reason (plain)
            self._event_state_table.setItem(row, 2, QTableWidgetItem(entry.source))
            self._event_state_table.setItem(row, 3, QTableWidgetItem(entry.timestamp))
            self._event_state_table.setItem(row, 4, QTableWidgetItem(entry.reason))

    def _refresh_navigator(self) -> None:
        """Step 2: Rebuild MotorWeightNavigator from current timeline + IO mappings."""
        try:
            from system.behavior.action_profile import get_motor_track_map
            robot_type = self._robot_type or ""
            brand = self._brand or ""
            available_tracks = get_motor_track_map(robot_type, brand=brand)
            self._motor_weight_navigator.refresh_from_timeline_and_io(
                self._behavior_timeline,
                self._hb_io_mappings,
                available_tracks=available_tracks,
                brand=brand,
                robot_type=robot_type,
            )
        except Exception:
            pass  # Never crash the panel

    def _refresh_movement_io_mapping(self) -> None:
        """Refresh the Movement IO Mapping panel with direction/status badges (Step 1.5)."""
        self._io_mapping_table.setRowCount(0)
        if not self._hb_io_mappings:
            # Show placeholder row so the panel is never silently empty
            self._io_mapping_table.insertRow(0)
            placeholder = QTableWidgetItem(
                tr("hb.no_io_bindings", "No IO bindings active")
            )
            placeholder.setForeground(QColor("#9e9e9e"))
            self._io_mapping_table.setItem(0, 0, placeholder)
            self._io_mapping_table.setSpan(0, 0, 1, 4)
            self._io_conflict_label.hide()
            return

        # Conflict detection 閳?update hint label
        conflicts = detect_io_conflicts(self._hb_io_mappings)
        if conflicts:
            msgs = [f"{h.signal}: {h.reason}" for h in conflicts]
            self._io_conflict_label.setText("\u26a0 " + "  |  ".join(msgs))
            self._io_conflict_label.show()
        else:
            self._io_conflict_label.hide()

        for mapping in self._hb_io_mappings:
            row = self._io_mapping_table.rowCount()
            self._io_mapping_table.insertRow(row)

            # Col 0: signal (plain)
            self._io_mapping_table.setItem(row, 0, QTableWidgetItem(mapping.signal))

            # Col 1: target_param (plain)
            self._io_mapping_table.setItem(row, 1, QTableWidgetItem(mapping.target_param))

            # Col 2: direction badge (colored foreground, 閳?In / 閳?Out)
            if mapping.direction == "inbound":
                dir_label = "\u2193 In"
                dir_color = QColor("#2196f3")
            elif mapping.direction == "outbound":
                dir_label = "\u2191 Out"
                dir_color = QColor("#ff9800")
            else:
                dir_label = mapping.direction
                dir_color = QColor("#9e9e9e")
            dir_item = QTableWidgetItem(dir_label)
            dir_item.setForeground(dir_color)
            dir_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._io_mapping_table.setItem(row, 2, dir_item)

            # Col 3: status badge (colored background)
            badge = badge_for_io_status(mapping.status)
            bg_hex = _BADGE_COLOR_MAP.get(badge.color_key, "#757575")
            status_item = QTableWidgetItem(badge.label)
            status_item.setBackground(QColor(bg_hex))
            status_item.setForeground(QColor("#ffffff"))
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._io_mapping_table.setItem(row, 3, status_item)

    def set_hb_event_states(self, entries: List[HBEventStateEntry]) -> None:
        """Public API: update event state region (called by runtime/compile path)."""
        self._hb_event_states = list(entries)
        self._refresh_event_state()

    def set_hb_io_mappings(self, mappings: List[HBIOMapping]) -> None:
        """Public API: update movement IO mapping region (called by runtime/compile path)."""
        self._hb_io_mappings = list(mappings)
        self._refresh_movement_io_mapping()
        self._refresh_navigator()  # Step 2

    def set_hb_compiled_snapshot(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Public API: store compiled/runtime snapshot state (separate from draft)."""
        self._hb_compiled_snapshot = snapshot

    # ------------------------------------------------------------------ Step 1.6
    def get_behavior_drafts_state(self) -> Dict[str, Any]:
        """Return the full per-node draft state as a JSON-serializable dict.

        Flushes the currently active node's draft into ``_drafts_by_node``
        before returning so the caller always gets a complete snapshot.

        Keys are node IDs converted to strings for JSON compatibility.
        Values are raw draft dicts ``{"core": str, "hb": {...}}``.
        Phase 1: the current node's "timeline" key is also flushed when a
        BehaviorTimeline is active.
        """
        # Flush current node so it is included in the snapshot
        if self._node_id >= 0:
            self._persist_current_behavior_draft()
        return {str(k): dict(v) for k, v in self._drafts_by_node.items()}

    def set_behavior_drafts_state(self, drafts: Dict[str, Any]) -> None:
        """Restore per-node draft state from a serialised dict (e.g. from mission file).

        Both int and str node IDs are accepted as keys.
        Entries that cannot be parsed are silently skipped (backward compat).
        After restoring, re-applies the current node's draft if one is present.
        Phase 1: entries with a "timeline" key also restore the BehaviorTimeline.
        """
        self._drafts_by_node = {}
        self._timelines_by_node = {}
        for k, v in drafts.items():
            if not isinstance(v, dict):
                continue
            key = str(k).strip()
            if not key:
                continue
            self._drafts_by_node[key] = dict(v)
            timeline_raw = v.get("timeline")
            if isinstance(timeline_raw, dict):
                self._drafts_by_node[key]["timeline"] = dict(timeline_raw)

        # Re-apply the current node's draft if the panel already has a node context
        if self._node_id >= 0:
            cur_key = self._draft_slot_key(self._node_id, self._behavior_ref or self._node_name)
            saved = self._drafts_by_node.get(cur_key)
            if saved is None:
                # One-time compatibility path for old "<node_id>" keys.
                legacy = self._drafts_by_node.get(str(self._node_id))
                if isinstance(legacy, dict):
                    saved = dict(legacy)
                    self._drafts_by_node[cur_key] = saved
                    self._drafts_by_node.pop(str(self._node_id), None)
            if saved is None:
                return
            core = saved.get("core", "")
            hb_dict = saved.get("hb")
            self._set_core_source(core, mark_dirty=False)
            try:
                payload = HBDraftPayload.from_dict(hb_dict or {})
                self._apply_hb_draft(payload)
            except Exception:  # noqa: BLE001
                self._apply_hb_draft(HBDraftPayload.default())
            # Restore timeline (Phase 1) 閳?run Circle 4 migration on load
            timeline_raw = self._timeline_dict_for_slot(cur_key)
            if isinstance(timeline_raw, dict):
                self._behavior_timeline = self._load_timeline_with_migration(timeline_raw)
                if self._behavior_timeline.is_empty() and not timeline_raw.get("action_segments"):
                    self._behavior_timeline = None
            else:
                self._behavior_timeline = None
            self._sync_views_from_timeline(mark_dirty=False)
            # Fix 6: restore expand/collapse state
            expanded_raw = saved.get("motor_track_expanded")
            if isinstance(expanded_raw, dict):
                self._timeline._motor_track_expanded.update(
                    {k: bool(v) for k, v in expanded_raw.items()}
                )

    # ------------------------------------------------------------------ Phase 1
    def get_behavior_timelines_state(self) -> Dict[str, Any]:
        """Return per-node BehaviorTimeline dicts for mission save.

        Flushes the current node's timeline before returning.
        """
        out: Dict[str, Any] = {}
        if self._node_id >= 0:
            self._persist_current_behavior_draft()
        for key, entry in self._drafts_by_node.items():
            tl = entry.get("timeline")
            if isinstance(tl, dict):
                out[str(key)] = dict(tl)
        return out

    def _load_timeline_with_migration(
        self, timeline_dict: Dict[str, Any]
    ) -> "BehaviorTimeline":
        """Load a BehaviorTimeline dict and automatically run Circle 4 migration.

        Deserialises via ``BehaviorTimeline.from_dict`` then passes the result
        through ``migrate_timeline`` so legacy segments that store raw action
        names (no intent_id) are resolved to semantic intent IDs before the
        timeline is used.  Warnings from ambiguous or unavailable mappings are
        emitted to the output log but never raise.

        Args:
            timeline_dict: Raw dict from a saved draft or mission file.

        Returns:
            Migrated ``BehaviorTimeline`` with intent_id populated where
            possible.  Falls back to an empty timeline on any exception.
        """
        try:
            tl = BehaviorTimeline.from_dict(timeline_dict)
            report = migrate_timeline(tl, brand=self._brand,
                                      robot_type=self._robot_type)
            for warning in report.warnings:
                if hasattr(self, "_output"):
                    self._output.append(f"[timeline migration] {warning}")
            return report.timeline
        except Exception:  # noqa: BLE001
            return BehaviorTimeline()

    def set_behavior_timelines_state(self, timelines: Dict[str, Any]) -> None:
        """Restore per-node BehaviorTimeline dicts from a mission file.

        Silently skips malformed entries.
        """
        for k, v in timelines.items():
            if not isinstance(v, dict):
                continue
            key = str(k).strip()
            if not key:
                continue
            entry = dict(self._drafts_by_node.get(key, {}))
            entry["timeline"] = dict(v)
            self._drafts_by_node[key] = entry
        # Apply to current node if context is already set
        if self._node_id >= 0:
            cur_key = self._draft_slot_key(self._node_id, self._behavior_ref or self._node_name)
            tl_raw = self._timeline_dict_for_slot(cur_key)
            if tl_raw is not None:
                try:
                    self._behavior_timeline = self._load_timeline_with_migration(tl_raw)
                    self._sync_views_from_timeline(mark_dirty=False)
                    self._refresh_timeline()
                except Exception:  # noqa: BLE001
                    pass

    def set_simulation_mode(self, is_sim: bool) -> None:
        """Update whether the current target is simulation or hardware.

        Called from bin/ui.py after scenario settings change.
        Affects is_simulation passed to BehaviorCompileWorker.
        """
        self._is_simulation = bool(is_sim)

    def set_capability_profile(self, cap: Dict[str, Any]) -> None:
        """Update the HeartBeat node catalog from an adapter capability dict.

        Called by MainWindow when the robot brand / type changes or after an
        adapter refresh.  Only the HeartBeat Library region is refreshed; all
        other panel regions are unaffected (Step 1.4).

        Args
        ----
        cap : The dict returned by ``adapter.capabilities()``, or an empty
              dict to reset to unknown-capability state.  Non-dict input is
              treated as an empty dict.
        """
        prev_brand = self._brand
        prev_robot_type = self._robot_type
        if isinstance(cap, dict):
            self._brand = cap.get("brand", prev_brand) or prev_brand or ""
            # robot_type may be at top-level (preferred) or nested inside "flags"
            # (some adapter versions put it there).  Check both for robustness.
            _flags = cap.get("flags") or {}
            self._robot_type = (
                cap.get("robot_type")
                or _flags.get("robot_type")
                or prev_robot_type
                or "go2"
            )
        else:
            self._brand = prev_brand or ""
            self._robot_type = prev_robot_type or "go2"
        self._hb_catalog = HBNodeCatalog.from_capability_dict(cap)
        self._refresh_hb_library()
        # Circle 5 STEP 5.2: refresh Movement library immediately on brand/model switch.
        self._refresh_registries()
        self._refresh_module_library()
        # Update limb diagram robot type
        if hasattr(self, "_limb_diagram_view"):
            self._limb_diagram_view.set_robot_type(self._robot_type)
        # Step 8: model-switch migration 閳?diagnose stale MotorSegment tracks
        # explicitly before rebuilding the timeline.  The timeline is preserved
        # (no silent drops); warnings surface in the output log so the user
        # knows which motor tracks may not render correctly in the new model.
        model_switched = (
            self._brand != prev_brand or self._robot_type != prev_robot_type
        )
        if model_switched and self._behavior_timeline is not None:
            try:
                from system.behavior.timeline_migration import (
                    migrate_timeline_on_model_switch,
                )
                switch_report = migrate_timeline_on_model_switch(
                    self._behavior_timeline,
                    new_brand=self._brand,
                    new_robot_type=self._robot_type,
                )
                for warning in switch_report.warnings:
                    if hasattr(self, "_output"):
                        self._output.append(f"[model switch] {warning}")
            except Exception:  # noqa: BLE001
                pass

        # Rebuild timeline only when the robot model actually changed (Fix 5 / bug-fix).
        # Unconditional rebuild was wiping user-edited action_seg.params on every
        # capability refresh (e.g. settings_changed 250 ms timer) even when the brand
        # and robot_type were identical.
        if self._behavior_timeline is not None and model_switched:
            self._sync_views_from_timeline(mark_dirty=False)
            self._behavior_timeline = self._build_or_update_timeline(self._modules, preserve_action_edits=False)
            self._sync_views_from_timeline(mark_dirty=False)
            self._refresh_timeline()
        # Always refresh navigator on model switch 閳?it must show the correct
        # body-part hierarchy for the new model even when no modules are authored.
        self._refresh_navigator()

    # ------------------------------------------------------------------ Step 1.7
    def run_compat_audit(self, brand: str = "", robot_type: str = "") -> HBCompatReport:
        """Run a compatibility audit against the current HB node catalog.

        Produces an :class:`HBCompatReport` from the active ``_hb_catalog``
        and updates the alert bar immediately.  Safe to call at any time;
        never raises.

        Parameters
        ----------
        brand      : Robot brand (e.g. ``"unitree"``).  Used in reason strings.
        robot_type : Robot type (e.g. ``"go2"``).  Used in reason strings.

        Returns
        -------
        :class:`HBCompatReport`
        """
        # HB node catalog audit (Step 1.7)
        hb_report = audit_catalog_compatibility(self._hb_catalog, brand, robot_type)

        # Circle 6 Step 6.1: semantic action compatibility audit
        try:
            action_issues, action_report = audit_action_compatibility(
                self._behavior_timeline, brand, robot_type
            )
        except Exception:
            action_issues, action_report = [], None

        # Merge both into a single report
        merged_issues = list(hb_report.issues) + action_issues
        report = HBCompatReport(
            brand=brand,
            robot_type=robot_type,
            issues=merged_issues,
            action_report=action_report,
        )
        self._apply_compat_report(report)
        return report

    def _apply_compat_report(self, report: HBCompatReport) -> None:
        """Update the compat alert bar and issues list from *report*."""
        if report.is_clean():
            self._compat_summary_label.hide()
            self._compat_issues_list.hide()
            return

        # Summary label
        if report.has_errors:
            color = "#f44336"
        else:
            color = "#ff9800"
        self._compat_summary_label.setStyleSheet(
            f"font-size: 11px; padding: 2px 4px; border-radius: 3px;"
            f" color: {color}; border: 1px solid {color};"
        )
        self._compat_summary_label.setText(
            f"\u26a0 {report.summary_text()} 閳?click item to highlight node"
        )
        self._compat_summary_label.show()

        # Issues list
        self._compat_issues_list.clear()
        for issue in report.issues:
            prefix = "[ERR]" if issue.severity == "error" else "[WARN]"
            if issue.intent_id is not None:
                # Circle 6 Step 6.2: action-level 閳?show intent + raw action context
                raw_info = ""
                if issue.source_raw_action and issue.target_raw_action:
                    raw_info = f" ({issue.source_raw_action!r} -> {issue.target_raw_action!r})"
                elif issue.source_raw_action:
                    raw_info = f" (src={issue.source_raw_action!r})"
                text = f"{prefix} [action] {issue.intent_id}{raw_info}: {issue.reason}"
            else:
                text = f"{prefix} {issue.node_label}: {issue.reason}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, issue.node_kind)
            if issue.severity == "error":
                item.setForeground(QColor("#f44336"))
            else:
                item.setForeground(QColor("#ff9800"))
            self._compat_issues_list.addItem(item)
        self._compat_issues_list.show()

    def _on_compat_issue_clicked(self, item: QListWidgetItem) -> None:
        """Click on a compat issue 閳?highlight matching node in HB library tree."""
        node_kind = item.data(Qt.ItemDataRole.UserRole)
        if node_kind:
            self._select_hb_library_item(node_kind)

    def _select_hb_library_item(self, node_kind: str) -> None:
        """Select and scroll to the HB library tree entry matching *node_kind*."""
        root = self._hb_library_tree.topLevelItem(0)
        if root is None:
            return
        for i in range(root.childCount()):
            child = root.child(i)
            data = child.data(0, Qt.ItemDataRole.UserRole) or {}
            if data.get("kind") == node_kind:
                self._hb_library_tree.setCurrentItem(child)
                self._hb_library_tree.scrollToItem(child)
                return

    def _insert_hb_node_from_library(self, item: QTreeWidgetItem) -> None:
        """Double-click on HeartBeat Library: add node to canvas (or no-op in script mode)."""
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        kind = data.get("kind")
        if not kind:
            return
        # Canvas mode is retired from product UI: no internal canvas mutation.
        # Legacy drafts still preserve "canvas_graph" for backward compatibility.
        _ = kind

    # ------------------------------------------------------------------
    # Step 1.2 閳?Canonical draft model and dual-mode authoring contract
    # ------------------------------------------------------------------

    def _on_policy_changed(self, _value=None) -> None:
        """Update _hb_policy_draft when any policy widget changes."""
        self._hb_policy_draft = HBPolicyDraft(
            tick_ms=self._hb_policy_tick_spin.value(),
            timeout_ms=self._hb_policy_timeout_spin.value(),
            fail_policy=self._hb_policy_fail_combo.currentText(),
            max_override=self._hb_policy_draft.max_override,
            safety_mode=self._hb_policy_draft.safety_mode,
        )

    def _collect_hb_draft(self) -> HBDraftPayload:
        """Collect current UI state into one canonical HBDraftPayload."""
        return HBDraftPayload(
            script=self._heartbeat_editor.toPlainText(),
            canvas_graph=dict(self._hb_legacy_canvas_graph),
            policy=HBPolicyDraft(
                tick_ms=self._hb_policy_tick_spin.value(),
                timeout_ms=self._hb_policy_timeout_spin.value(),
                fail_policy=self._hb_policy_fail_combo.currentText(),
                max_override=self._hb_policy_draft.max_override,
                safety_mode=self._hb_policy_draft.safety_mode,
            ),
            io_bindings=[m.to_dict() for m in self._hb_io_mappings],
            event_schema=[e.to_dict() for e in self._hb_event_states],
            authoring_mode=self._hb_authoring_mode,
        )

    def _apply_hb_draft(self, payload: HBDraftPayload) -> None:
        """Restore all HB authoring UI state from a canonical HBDraftPayload.

        Lossless: both canvas graph and script text are restored independently;
        neither representation is discarded when switching authoring modes.
        """
        # Restore script (Script mode widget)
        self._set_heartbeat_source(payload.script, mark_dirty=False)
        # Restore legacy canvas graph payload (data-only; no UI canvas dependency).
        try:
            self._hb_legacy_canvas_graph = HBGraphData.from_dict(payload.canvas_graph).to_dict()
        except Exception:
            self._hb_legacy_canvas_graph = HBGraphData.default_template().to_dict()
        # Restore policy widgets (block signals to avoid re-entrant _on_policy_changed)
        self._hb_policy_tick_spin.blockSignals(True)
        self._hb_policy_tick_spin.setValue(payload.policy.tick_ms)
        self._hb_policy_tick_spin.blockSignals(False)
        self._hb_policy_timeout_spin.blockSignals(True)
        self._hb_policy_timeout_spin.setValue(payload.policy.timeout_ms)
        self._hb_policy_timeout_spin.blockSignals(False)
        _fp_idx = self._hb_policy_fail_combo.findText(payload.policy.fail_policy)
        self._hb_policy_fail_combo.blockSignals(True)
        if _fp_idx >= 0:
            self._hb_policy_fail_combo.setCurrentIndex(_fp_idx)
        self._hb_policy_fail_combo.blockSignals(False)
        self._hb_policy_draft = HBPolicyDraft.from_dict(payload.policy.to_dict())
        # Restore IO bindings and event schema into display lists
        self._hb_io_mappings = [HBIOMapping.from_dict(b) for b in payload.io_bindings]
        self._hb_event_states = [HBEventStateEntry.from_dict(e) for e in payload.event_schema]
        self._refresh_movement_io_mapping()
        self._refresh_event_state()
        self._refresh_navigator()   # Step 2
        # Restore authoring mode last (sets stack index + _hb_authoring_mode).
        # Legacy "canvas" drafts map to navigator in product UI.
        if payload.authoring_mode == "script":
            self._set_hb_mode(1)
        else:
            self._set_hb_mode(2)

    def get_hb_draft_payload(self) -> HBDraftPayload:
        """Public API: return the current canonical heartbeat draft payload.

        Used by compile/run/save paths to get the complete authored state.
        """
        return self._collect_hb_draft()

    def set_hb_channel(self, channel: IHBChannel) -> None:
        """Public API: swap the execution channel (replace mock with real runtime).

        Called by the host (bin/ui.py or test harness) once the real backend
        channel is available.  The diagnostics poll timer automatically uses
        the new channel on the next tick.
        """
        self._hb_channel = channel

    def set_bridge(self, bridge: BehaviorCompilerBridge) -> None:
        """Inject the canonical compiler bridge owned by the host runtime."""
        if bridge is None:
            return
        self._bridge = bridge

    # ------------------------------------------------------------------
    # Step 1.3 閳?Execution chain handlers
    # ------------------------------------------------------------------

    def _on_hb_compile(self) -> None:
        """HB Compile button: compile the current heartbeat draft via the channel."""
        if not self._node_name:
            self._output.append("[HB Compile] No behavior node selected.")
            return
        draft = self._collect_hb_draft()
        req = HBCompileRequest(behavior_ref=self._behavior_ref or self._node_name, draft_payload=draft)
        self._hb_compile_btn.setEnabled(False)
        try:
            resp = self._hb_channel.compile(req)
        except Exception as exc:
            self._hb_status_label.setText(f"HB compile error: {exc}")
            self._output.append(f"[HB Compile] ERROR: {exc}")
            self._hb_compile_btn.setEnabled(True)
            return
        finally:
            self._hb_compile_btn.setEnabled(True)

        if resp.ok:
            self._hb_status_label.setText(
                f"HB compiled OK 閳?artifact {resp.artifact_id[:12]}... "
                f"{resp.warning_count} warning(s)"
            )
            self._hb_compiled_snapshot = {
                "artifact_id": resp.artifact_id,
                "error_count": resp.error_count,
                "warning_count": resp.warning_count,
                "diagnostics": resp.diagnostics,
                "message": resp.message,
            }
        else:
            self._hb_status_label.setText(
                f"HB compile failed 閳?{resp.error_count} error(s)"
            )
        self._output.append(f"[HB Compile] {resp.message}")
        for diag in resp.diagnostics:
            self._output.append(f"  [{diag.get('severity','info')}] {diag.get('message','')}")

    def _on_hb_dryrun(self) -> None:
        """Simulate button: dry-run the heartbeat and populate diagnostics panels."""
        if not self._node_name:
            self._output.append("[Simulate] No behavior node selected.")
            return
        draft = self._collect_hb_draft()
        tick_count = max(1, draft.policy.tick_ms and 5)  # default 5 ticks
        req = HBDryRunRequest(
            behavior_ref=self._behavior_ref or self._node_name,
            draft_payload=draft,
            tick_count=tick_count,
        )
        self._hb_simulate_btn.setEnabled(False)
        try:
            resp = self._hb_channel.dry_run(req)
        except Exception as exc:
            self._output.append(f"[Simulate] ERROR: {exc}")
            self._hb_simulate_btn.setEnabled(True)
            return
        finally:
            self._hb_simulate_btn.setEnabled(True)

        self._output.append(f"[Simulate] {resp.message}")
        if resp.ok:
            self._hb_status_label.setText(
                f"Simulated {resp.simulated_ticks} tick(s) 閳?"
                f"{len(resp.event_updates)} event(s)"
            )
            # Populate event state panel from simulation
            events = [HBEventStateEntry.from_dict(e) for e in resp.event_updates]
            self.set_hb_event_states(events)
            # Populate IO mapping from sensor/control summaries
            mappings = self._build_io_mappings_from_snapshot(
                resp.sensor_snapshot_summary,
                resp.control_output_summary,
            )
            self.set_hb_io_mappings(mappings)
            # Update diagnostics indicator
            sensor_keys = list(resp.sensor_snapshot_summary.keys())
            self._hb_diag_indicator.setText(
                f"sensors: {', '.join(sensor_keys[:3])}"
            )

    def _on_hb_run(self) -> None:
        """Run HB button: start live heartbeat execution via the channel."""
        if not self._node_name:
            self._output.append("[Run HB] No behavior node selected.")
            return
        draft = self._collect_hb_draft()
        req = HBRunRequest(behavior_ref=self._behavior_ref or self._node_name, draft_payload=draft)
        self._hb_run_btn.setEnabled(False)
        try:
            resp = self._hb_channel.run(req)
        except Exception as exc:
            self._output.append(f"[Run HB] ERROR: {exc}")
            self._hb_run_btn.setEnabled(True)
            return
        finally:
            self._hb_run_btn.setEnabled(True)

        self._output.append(f"[Run HB] {resp.message}")
        if resp.ok:
            self._hb_status_label.setText(
                f"HB {resp.status} 閳?heartbeat: {resp.heartbeat_status}"
            )
            if not self._hb_diag_timer.isActive():
                self._hb_diag_timer.start()

    # ------------------------------------------------------------------
    # Playback controls (timeline preview)
    # ------------------------------------------------------------------

    def _play_cmd_play(self) -> None:
        """Start or resume playback."""
        if self._is_playing:
            return
        self._is_playing = True
        self._play_btn.setText("⏸")
        self._play_timer.start()

    def _play_cmd_pause(self) -> None:
        """Pause playback (keeps current position)."""
        self._is_playing = False
        self._play_btn.setText("▶")
        self._play_timer.stop()

    def _play_cmd_stop(self) -> None:
        """Stop playback and rewind to start."""
        self._is_playing = False
        self._play_btn.setText("▶")
        self._play_timer.stop()
        self._playback_time = 0.0
        self._timeline.set_playhead(0.0)
        self._play_time_label.setText("0.00 s")
        self.playback_time_changed.emit(0.0)
        if hasattr(self, "_limb_diagram_view"):
            self._limb_diagram_view.update_time(0.0)

    def _play_cmd_toggle(self) -> None:
        """Toggle play/pause."""
        if self._is_playing:
            self._play_cmd_pause()
        else:
            self._play_cmd_play()

    def _on_play_tick(self) -> None:
        """50 ms playback tick — advance playhead and update display."""
        if not self._is_playing:
            return
        # Determine total duration from active timeline
        total_duration = 0.0
        if self._behavior_timeline is not None:
            for seg in self._behavior_timeline.action_segments:
                end = seg.start_time + seg.duration
                if end > total_duration:
                    total_duration = end
        if total_duration <= 0.0:
            total_duration = 10.0  # fallback for empty/unloaded timeline

        self._playback_time += 0.05  # 50 ms step
        if self._playback_time >= total_duration:
            self._playback_time = total_duration
            self._play_cmd_stop()
        else:
            self._timeline.set_playhead(self._playback_time)
            self._play_time_label.setText(f"{self._playback_time:.2f} s")
            self.playback_time_changed.emit(self._playback_time)
            if hasattr(self, "_limb_diagram_view"):
                self._limb_diagram_view.update_time(self._playback_time)

    def _poll_hb_diagnostics(self) -> None:
        """Timer callback: poll channel diagnostics and refresh status display."""
        if not self._node_name or self._node_id < 0:
            return
        try:
            snap: HBDiagnosticsSnapshot = self._hb_channel.poll_diagnostics(
                self._behavior_ref or self._node_name
            )
        except Exception:
            return
        # Update indicator label with live status
        if snap.status == "running":
            self._hb_diag_indicator.setText(
                f"HB active 閳?tick #{snap.tick_count}"
            )
        elif snap.status == "idle":
            self._hb_diag_indicator.setText("HB idle")
        else:
            self._hb_diag_indicator.setText(f"HB {snap.status}")
        # Push event updates if present
        if snap.event_updates:
            events = [HBEventStateEntry.from_dict(e) for e in snap.event_updates]
            self.set_hb_event_states(events)
        # Push IO mapping from sensor/control data if running
        if snap.status == "running" and (
            snap.sensor_snapshot_summary or snap.control_output_summary
        ):
            mappings = self._build_io_mappings_from_snapshot(
                snap.sensor_snapshot_summary,
                snap.control_output_summary,
            )
            if mappings:
                self.set_hb_io_mappings(mappings)

    def _build_io_mappings_from_snapshot(
        self,
        sensor_summary: Dict[str, Any],
        control_summary: Dict[str, Any],
    ) -> List[HBIOMapping]:
        """Convert sensor + control summary dicts into display HBIOMapping list."""
        mappings: List[HBIOMapping] = []
        for key, val in sensor_summary.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    mappings.append(HBIOMapping(
                        signal=f"{key}.{sub_key}",
                        target_param=f"{key}.{sub_key}",
                        direction="inbound",
                        status="active",
                    ))
            else:
                mappings.append(HBIOMapping(
                    signal=key,
                    target_param=key,
                    direction="inbound",
                    status="active",
                ))
        for key, val in control_summary.items():
            mappings.append(HBIOMapping(
                signal=key,
                target_param=key,
                direction="outbound",
                status="active",
            ))
        return mappings

    def _insert_module_from_library(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        kind = data.get("kind")
        if kind not in {"movement", "behavior"}:
            return

        name = str(data.get("name") or "")
        self._ensure_behavior_timeline()
        module = SequenceModule(kind=kind, name=name, args="duration=1.0", duration=1.0)
        new_seg = self._build_or_update_timeline([module], preserve_action_edits=False).action_segments[0]
        self._behavior_timeline.action_segments.append(new_seg)
        self._selected_module_index = len(self._behavior_timeline.action_segments) - 1
        self._rebuild_timeline_from_action_segments(mark_dirty=True)

    def _remove_selected_module(self) -> None:
        self._ensure_behavior_timeline()
        segs = self._behavior_timeline.action_segments
        if self._selected_module_index < 0 or self._selected_module_index >= len(segs):
            return
        segs.pop(self._selected_module_index)
        if self._selected_module_index >= len(segs):
            self._selected_module_index = len(segs) - 1
        self._rebuild_timeline_from_action_segments(mark_dirty=True)

    def _move_selected_module(self, delta: int) -> None:
        self._ensure_behavior_timeline()
        segs = self._behavior_timeline.action_segments
        i = self._selected_module_index
        if i < 0 or i >= len(segs):
            return
        j = i + delta
        if j < 0 or j >= len(segs):
            return
        segs[i], segs[j] = segs[j], segs[i]
        cursor = 0.0
        for seg in segs:
            seg.start_time = cursor
            cursor += max(0.01, float(seg.duration))
        self._selected_module_index = j
        self._rebuild_timeline_from_action_segments(mark_dirty=True)

    # ------------------------------------------------------------------ selection model

    def _apply_selection(self, sel: "TimelineSelection") -> None:
        """Set the authoritative selection state.

        Writes to ``_panel_selection`` and keeps the widget's
        ``_timeline_vm.selection`` in sync.  All selection changes must flow
        through this method so that there is exactly one source of truth for
        what is currently selected.
        """
        self._panel_selection = sel
        _vm = getattr(self._timeline, "_timeline_vm", None)
        if _vm is not None:
            _vm.selection = sel

    def _action_id_from_module_index(self, idx: int) -> str:
        """Return the action_id for the action segment at *idx*."""
        if self._behavior_timeline is None:
            return ""
        segs = getattr(self._behavior_timeline, "action_segments", []) or []
        if 0 <= idx < len(segs):
            return getattr(segs[idx], "action_id", "") or ""
        return ""

    def _resolve_motor_context(self, motor_id: str, track_name: str) -> "Tuple[str, str]":
        """Return ``(action_id, limb_id)`` for the motor segment identified by *motor_id*.

        Looks up the overlay that owns *motor_id* (gives ``action_id``), then
        queries the current VM body-part rows for the limb that owns *track_name*
        (gives ``limb_id``).  Returns empty strings when context is unavailable.
        """
        action_id = ""
        if self._behavior_timeline is not None:
            for overlay in self._behavior_timeline.motor_overlays:
                for seg in getattr(overlay, "motor_segments", []):
                    if getattr(seg, "motor_id", "") == motor_id:
                        action_id = getattr(overlay, "action_id", "") or ""
                        break
                if action_id:
                    break

        limb_id = ""
        _vm = getattr(self._timeline, "_timeline_vm", None)
        if _vm is not None:
            for bp in _vm.layout_body_parts:
                if any(mr.track_name == track_name for mr in bp.motors):
                    limb_id = bp.limb_id
                    break

        return action_id, limb_id

    def _on_timeline_module_selected(self, index: int) -> None:
        self._selected_module_index = index
        action_id = self._action_id_from_module_index(index)
        from system.behavior.timeline_view_model import TimelineSelection
        self._apply_selection(
            TimelineSelection.action(action_id) if action_id else TimelineSelection.none()
        )
        self._refresh_timeline()
        self._refresh_movement_settings()

    def _on_timeline_motor_segment_selected(self, motor_id: str, track_name: str) -> None:
        from system.behavior.timeline_view_model import TimelineSelection
        if motor_id and track_name:
            action_id, limb_id = self._resolve_motor_context(motor_id, track_name)
            self._apply_selection(
                TimelineSelection.motor(
                    action_id=action_id,
                    limb_id=limb_id,
                    track_name=track_name,
                    motor_id=motor_id,
                )
            )
            self._selected_module_index = -1
        else:
            self._apply_selection(TimelineSelection.none())
        self._refresh_timeline()
        self._refresh_movement_settings()

    def _on_timeline_body_part_selected(self, action_id: str, limb_id: str) -> None:
        """Called when user clicks a body-part header row in the timeline."""
        from system.behavior.timeline_view_model import TimelineSelection
        self._apply_selection(
            TimelineSelection.body_part(action_id, limb_id) if limb_id
            else TimelineSelection.none()
        )
        self._selected_module_index = -1
        self._refresh_timeline()
        self._refresh_movement_settings()

    def _on_timeline_module_reordered(self, from_index: int, to_index: int) -> None:
        if from_index < 0 or to_index < 0:
            return
        self._ensure_behavior_timeline()
        segs = self._behavior_timeline.action_segments
        if from_index >= len(segs) or to_index >= len(segs):
            return
        if from_index == to_index:
            return
        seg = segs.pop(from_index)
        segs.insert(to_index, seg)
        cursor = 0.0
        for action_seg in segs:
            action_seg.start_time = cursor
            cursor += max(0.01, float(action_seg.duration))
        self._selected_module_index = to_index
        self._rebuild_timeline_from_action_segments(mark_dirty=True)

    def _on_timeline_edited(self) -> None:
        """Called when the timeline emits timeline_edited (Fix 6)."""
        self._mark_dirty()
        self._refresh_movement_settings()
        self._refresh_navigator()  # Step 2

    def _refresh_timeline(self) -> None:
        # Phase 1: use multi-track timeline when structured model is available
        if self._behavior_timeline is not None and not self._behavior_timeline.is_empty():
            try:
                from system.behavior.action_profile import get_motor_track_map
                available_tracks = get_motor_track_map(self._robot_type, brand=self._brand)
            except Exception:
                available_tracks = {}
            self._timeline.set_multi_track_timeline(
                timeline=self._behavior_timeline,
                selected_index=self._selected_module_index,
                brand=self._brand,
                robot_type=self._robot_type,
                available_tracks=available_tracks,
                selection=self._panel_selection,
            )
            self._refresh_timeline_status()
            return

        # Legacy single-track path
        secondary: List[SequenceModule] = []
        secondary_name = ""

        if 0 <= self._selected_module_index < len(self._modules):
            selected = self._modules[self._selected_module_index]
            if selected.kind == "behavior":
                secondary = self._resolve_behavior_children(selected.name)
                secondary_name = selected.name

        self._timeline.set_timeline(
            modules=self._modules,
            secondary=secondary,
            selected_index=self._selected_module_index,
            secondary_source_name=secondary_name,
        )
        self._refresh_timeline_status()

    # 閳光偓閳光偓 Step 3: IO mutation helpers for source-switch 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

    def _io_upsert_external(self, track_name: str, param_key: str, signal: str) -> None:
        """Add or update an inbound HBIOMapping for track_name.param_key 閳?signal."""
        target = f"{track_name}.{param_key}"
        for m in self._hb_io_mappings:
            if m.target_param == target and m.direction == "inbound":
                if m.signal == signal:
                    return
                m.signal = signal
                self._refresh_movement_io_mapping()
                self._refresh_navigator()
                self._refresh_movement_settings()
                self._mark_dirty()
                return
        self._hb_io_mappings.append(HBIOMapping(
            signal=signal,
            target_param=target,
            direction="inbound",
            status="active",
        ))
        self._refresh_movement_io_mapping()
        self._refresh_navigator()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _io_remove_external(self, track_name: str, param_key: str) -> None:
        """Remove all inbound HBIOMappings targeting track_name.param_key."""
        target = f"{track_name}.{param_key}"
        before = len(self._hb_io_mappings)
        self._hb_io_mappings = [
            m for m in self._hb_io_mappings
            if not (m.target_param == target and m.direction == "inbound")
        ]
        if len(self._hb_io_mappings) == before:
            return
        self._refresh_movement_io_mapping()
        self._refresh_navigator()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _set_param_editor_3col(
        self,
        row: int,
        track_name: str,
        key: str,
        value: Any,
        on_commit_constant,
    ) -> None:
        """Step 3: populate Source (col 1) + Value (col 2) for a motor segment param row."""
        is_structural = key in STRUCTURAL_KEYS
        src_info = get_param_source(track_name, key, self._hb_io_mappings)

        # 閳光偓閳光偓 Column 1: Source indicator 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        if is_structural:
            # Structural params always Constant 閳?show a read-only label
            src_label = QLabel("Constant")
            src_label.setStyleSheet("color: #9e9e9e; font-size: 11px; padding: 0 4px;")
            self._movement_params_table.setCellWidget(row, 1, src_label)
        else:
            src_combo = QComboBox()
            src_combo.addItems(["Constant", "External"])
            src_combo.setCurrentIndex(1 if src_info.source == "external" else 0)
            src_combo.currentTextChanged.connect(
                lambda txt, tn=track_name, pk=key: (
                    self._io_upsert_external(tn, pk, "")
                    if txt == "External"
                    else self._io_remove_external(tn, pk)
                )
            )
            self._movement_params_table.setCellWidget(row, 1, src_combo)

        # 閳光偓閳光偓 Column 2: Value editor (depends on source) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        if src_info.source == "external":
            sig_edit = QLineEdit(src_info.signal)
            sig_edit.setPlaceholderText("signal key (e.g. imu.pitch)")
            sig_edit.editingFinished.connect(
                lambda e=sig_edit, tn=track_name, pk=key: (
                    self._io_upsert_external(tn, pk, e.text())
                )
            )
            self._movement_params_table.setCellWidget(row, 2, sig_edit)
        else:
            # Constant path 閳?reuse high-precision editor
            num_val = self._try_float(value)
            if num_val is None:
                editor = QLineEdit(str(value) if value is not None else "")
                editor.editingFinished.connect(
                    lambda e=editor, k=key: on_commit_constant(k, e.text())
                )
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(2)
                editor.setSingleStep(0.01)
                editor.setRange(-1_000_000.0, 1_000_000.0)
                editor.setValue(num_val)
                editor.editingFinished.connect(
                    lambda e=editor, k=key: on_commit_constant(k, e.value())
                )
            self._movement_params_table.setCellWidget(row, 2, editor)

    # 閳光偓閳光偓 Step 3 end 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

    # Circle 5 STEP 5.3 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def _get_descriptor_for_segment(
        self, action_seg: ActionSegment
    ) -> Optional[SemanticActionDescriptor]:
        """Return the semantic descriptor for an ActionSegment.

        Lookup order:
        1. Match by ``intent_id`` (exact semantic match).
        2. Fall back to matching by ``raw_action`` == segment name.
        Returns ``None`` when no descriptor is found.
        """
        descriptors = get_semantic_actions(self._brand, self._robot_type)
        if action_seg.intent_id:
            for d in descriptors:
                if d.intent_id == action_seg.intent_id:
                    return d
        for d in descriptors:
            if d.raw_action == action_seg.name:
                return d
        return None

    def _descriptor_param_rows(
        self,
        descriptor: SemanticActionDescriptor,
        live_params: Optional[Dict[str, Any]],
        parsed_args: Optional[List[Tuple[str, Any]]] = None,
    ) -> List[Tuple[str, Any]]:
        """Build ``(key, current_value)`` rows from descriptor.parameters.

        Resolution priority for each declared key:
        1. ``live_params`` dict (ActionSegment.params or equivalent).
        2. ``parsed_args`` list of (key, value) pairs (legacy flat-module path).
        3. Descriptor ``default`` value.

        Args:
            descriptor:  The resolved semantic descriptor for this action.
            live_params: Live parameter dict (may be None or empty).
            parsed_args: Fallback ``[(key, value)]`` from ``_parse_args``
                         (used only when ``live_params`` is absent/empty).
        Returns:
            Ordered ``[(key, value)]`` list matching declaration order in
            ``descriptor.parameters``.
        """
        args_map: Dict[str, Any] = dict(parsed_args) if parsed_args else {}
        params_map: Dict[str, Any] = live_params if live_params else {}
        rows: List[Tuple[str, Any]] = []
        for key, meta in descriptor.parameters.items():
            if key in params_map:
                val = params_map[key]
            elif key in args_map:
                val = args_map[key]
            else:
                val = meta.get("default")
            rows.append((key, val))
        return rows

    def _commit_action_seg_param(
        self, action_seg: ActionSegment, key: str, value: Any
    ) -> None:
        """Persist a parameter edit back into an ActionSegment.

        Structural fields such as ``start_time`` and ``duration`` must update the
        ActionSegment itself so timeline geometry stays in sync with the editor.
        """
        changed = False
        if key == "start_time":
            v = self._try_float(value)
            if v is not None:
                new_val = max(0.0, v)
                changed = abs(float(action_seg.start_time) - float(new_val)) > 1e-9
                if changed:
                    action_seg.start_time = new_val
                    action_seg.params.pop("start_time", None)
        elif key == "duration":
            v = self._try_float(value)
            if v is not None:
                new_val = max(0.01, v)
                changed = abs(float(action_seg.duration) - float(new_val)) > 1e-9
                if changed:
                    action_seg.duration = new_val
                    action_seg.params.pop("duration", None)
        else:
            new_val = self._coerce_param_value(value)
            old_val = action_seg.params.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                changed = abs(float(old_val) - float(new_val)) > 1e-9
            else:
                changed = old_val != new_val
            if changed:
                action_seg.params[key] = new_val
        if not changed:
            return
        self._persist_current_timeline_draft()
        self._refresh_timeline()
        self._refresh_movement_settings()
        if self._node_id >= 0:
            self.action_parameters_changed.emit(int(self._node_id))
        self._mark_dirty()

    def _timeline_slot_key(self, node_id: int, behavior_ref: str, node_name: str = "") -> str:
        ref = str(behavior_ref or "").strip() or str(node_name or "").strip() or f"behavior_node_{int(node_id)}"
        return self._draft_slot_key(node_id, ref)

    def _timeline_for_node(
        self,
        *,
        node_id: int,
        behavior_ref: str,
        intent_id: str = "",
        display_name: str = "",
        robot_type: str = "",
    ) -> BehaviorTimeline:
        slot_key = self._timeline_slot_key(node_id, behavior_ref, display_name)
        current_key = self._timeline_slot_key(
            self._node_id,
            self._behavior_ref or self._node_name,
            self._node_name,
        )
        if self._behavior_timeline is not None and self._node_id >= 0 and slot_key == current_key:
            return self._behavior_timeline

        saved = self._drafts_by_node.get(slot_key, {})
        timeline_raw = self._timeline_dict_for_slot(slot_key)
        if isinstance(timeline_raw, dict):
            return self._load_timeline_with_migration(timeline_raw)
        return _default_timeline_for_behavior_selection(
            intent_id,
            display_name,
            self._brand,
            robot_type or self._robot_type,
        )

    @staticmethod
    def _coerce_param_value(raw_value: Any) -> Any:
        text = str(raw_value if raw_value is not None else "").strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            num = float(text)
        except ValueError:
            return text
        if abs(num - int(num)) <= 1e-9:
            return int(num)
        return num

    def _display_name_for_action_segment(self, action_seg: ActionSegment) -> str:
        descriptor = self._get_descriptor_for_segment(action_seg)
        if descriptor is not None and str(descriptor.display_name or "").strip():
            return str(descriptor.display_name).strip()
        action_name = str(getattr(action_seg, "name", "") or "").strip()
        pretty = action_name.replace("_", " ").strip()
        return pretty.title() if pretty else "Action"

    def _resolve_node_action_segment(
        self,
        timeline: BehaviorTimeline,
        *,
        intent_id: str = "",
        display_name: str = "",
    ) -> Optional[ActionSegment]:
        segments = getattr(timeline, "action_segments", []) or []
        if not segments:
            return None
        wanted_intent = str(intent_id or "").strip()
        wanted_name = str(display_name or "").strip().lower()
        if wanted_intent:
            for seg in segments:
                if str(getattr(seg, "intent_id", "") or "").strip() == wanted_intent:
                    return seg
        if wanted_name:
            for seg in segments:
                seg_name = str(getattr(seg, "name", "") or "").strip().lower()
                disp_name = self._display_name_for_action_segment(seg).strip().lower()
                if wanted_name in {seg_name, disp_name}:
                    return seg
        return segments[0]

    def get_node_exposed_action_parameters(
        self,
        *,
        node_id: int,
        behavior_ref: str,
        intent_id: str = "",
        display_name: str = "",
        robot_type: str = "",
    ) -> List[Dict[str, Any]]:
        timeline = self._timeline_for_node(
            node_id=node_id,
            behavior_ref=behavior_ref,
            intent_id=intent_id,
            display_name=display_name,
            robot_type=robot_type,
        )
        action_seg = self._resolve_node_action_segment(
            timeline,
            intent_id=intent_id,
            display_name=display_name,
        )
        if action_seg is None:
            return []
        descriptor = self._get_descriptor_for_segment(action_seg)
        if descriptor is not None and descriptor.parameters:
            rows = self._descriptor_param_rows(
                descriptor,
                live_params={
                    **dict(action_seg.params),
                    "start_time": action_seg.start_time,
                    "duration": action_seg.duration,
                },
            )
        else:
            rows = sorted((dict(action_seg.params) or {}).items(), key=lambda kv: str(kv[0]))
        row_keys = {str(key) for key, _value in rows}
        if "duration" not in row_keys:
            rows = [("duration", action_seg.duration)] + rows
        return [
            {
                "key": str(key),
                "label": str(key),
                "value": "" if value is None else str(value),
            }
            for key, value in rows
        ]

    def update_node_exposed_action_parameter(
        self,
        *,
        node_id: int,
        behavior_ref: str,
        intent_id: str = "",
        display_name: str = "",
        robot_type: str = "",
        key: str,
        value: Any,
    ) -> bool:
        slot_key = self._timeline_slot_key(node_id, behavior_ref, display_name)
        timeline = self._timeline_for_node(
            node_id=node_id,
            behavior_ref=behavior_ref,
            intent_id=intent_id,
            display_name=display_name,
            robot_type=robot_type,
        )
        action_seg = self._resolve_node_action_segment(
            timeline,
            intent_id=intent_id,
            display_name=display_name,
        )
        if action_seg is None:
            return False

        key_name = str(key)
        changed = False
        if key_name == "start_time":
            v = self._try_float(value)
            if v is not None:
                new_val = max(0.0, v)
                changed = abs(float(action_seg.start_time) - float(new_val)) > 1e-9
                if changed:
                    action_seg.start_time = new_val
                    action_seg.params.pop("start_time", None)
        elif key_name == "duration":
            v = self._try_float(value)
            if v is not None:
                new_val = max(0.01, v)
                changed = abs(float(action_seg.duration) - float(new_val)) > 1e-9
                if changed:
                    action_seg.duration = new_val
                    action_seg.params.pop("duration", None)
        else:
            new_val = self._coerce_param_value(value)
            old_val = action_seg.params.get(key_name)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                changed = abs(float(old_val) - float(new_val)) > 1e-9
            else:
                changed = old_val != new_val
            if changed:
                action_seg.params[key_name] = new_val
        if not changed:
            return True

        entry = dict(self._drafts_by_node.get(slot_key, {}))
        entry["timeline"] = timeline.to_dict()
        self._drafts_by_node[slot_key] = entry

        is_current_context = self._node_id >= 0 and int(self._node_id) == int(node_id)
        if is_current_context:
            self._behavior_timeline = timeline
            self._refresh_timeline()
            self._refresh_movement_settings()
            self._mark_dirty()

        self.action_parameters_changed.emit(int(node_id))
        return True

    def _render_descriptor_param_table(
        self,
        rows: List[Tuple[str, Any]],
        descriptor: SemanticActionDescriptor,
        commit_fn,
    ) -> None:
        """Populate ``_movement_params_table`` from descriptor-derived rows.

        Args:
            rows:       ``[(key, value)]`` from ``_descriptor_param_rows``.
            descriptor: The resolved descriptor (used for min/max metadata).
            commit_fn:  Callable ``(key, value) -> None`` invoked on edit.
        """
        self._movement_params_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self._movement_params_table.setItem(r, 0, QTableWidgetItem(k))
            src_label = QLabel("Semantic")
            src_label.setStyleSheet("color: #2196f3; font-size: 11px; padding: 0 4px;")
            self._movement_params_table.setCellWidget(r, 1, src_label)
            num_val = self._try_float(v)
            if num_val is None:
                editor = QLineEdit(str(v) if v is not None else "")
                editor.editingFinished.connect(
                    lambda e=editor, key_name=k: commit_fn(key_name, e.text())
                )
            else:
                meta = descriptor.parameters.get(k, {})
                editor = QDoubleSpinBox()
                editor.setDecimals(2)
                editor.setSingleStep(0.01)
                editor.setRange(
                    float(meta.get("min", -1_000_000.0)),
                    float(meta.get("max",  1_000_000.0)),
                )
                editor.setValue(num_val)
                editor.editingFinished.connect(
                    lambda e=editor, key_name=k: commit_fn(key_name, e.value())
                )
            self._movement_params_table.setCellWidget(r, 2, editor)

    def _refresh_movement_settings(self) -> None:
        self._movement_params_table.setRowCount(0)
        self._movement_params_table.clearContents()
        self._movement_params_table.clearSpans()
        self._movement_group_header_rows = {}
        self._refresh_timeline_status()
        # Priority: when a motor sub-track segment is selected, edit it directly.
        seg = self._get_selected_motor_segment()
        if seg is not None:
            self._movement_selected_label.setText(
                f"Selected Limb Module: {self._get_active_track_label(seg.track_name)}"
            )
            self._movement_hint.setText(
                "This panel is editing the selected timeline module directly. Changes apply immediately."
            )
            rows: List[Tuple[str, Any]] = [("start_time", seg.start_time), ("duration", seg.duration)]
            track_def = self._get_active_track_def(seg.track_name)
            if track_def and track_def.param_key in seg.params:
                rows.append((track_def.param_key, seg.params.get(track_def.param_key)))
            for k, v in seg.params.items():
                if k not in {rk for rk, _ in rows}:
                    rows.append((k, v))
            self._movement_params_table.setRowCount(len(rows))
            for r, (k, v) in enumerate(rows):
                self._movement_params_table.setItem(r, 0, QTableWidgetItem(str(k)))
                self._set_param_editor_3col(
                    row=r,
                    track_name=seg.track_name,
                    key=str(k),
                    value=v,
                    on_commit_constant=lambda key_name, val, s=seg: self._commit_motor_param(s, key_name, val),
                )
            return

        # Step 6: body-part selected path 閳?highest priority after motor-segment.
        sel = self._panel_selection
        if sel.level.value == "body_part" and sel.limb_id and self._behavior_timeline is not None:
            self._render_body_part_settings(sel.limb_id, action_id=sel.action_id)
            return

        # Multi-track mode: timeline widget emits indices into action_segments, NOT
        # panel._modules (which tracks the high-level module sequence and has fewer
        # entries when a package action expands into multiple phases).
        if self._behavior_timeline is not None and not self._behavior_timeline.is_empty():
            n_actions = len(self._behavior_timeline.action_segments)
            if 0 <= self._selected_module_index < n_actions:
                action_seg = self._behavior_timeline.action_segments[self._selected_module_index]
                # Circle 5 STEP 5.3: resolve semantic descriptor for display name + availability.
                descriptor = self._get_descriptor_for_segment(action_seg)
                display_name = (
                    descriptor.display_name if descriptor is not None else action_seg.name
                )
                self._movement_selected_label.setText(
                    tr("behavior.movement.selected_format", "Selected Action: {node}").format(
                        node=display_name
                    )
                )
                # Surface degraded / unsupported state before showing parameters.
                if descriptor is not None:
                    avail = descriptor.availability
                    if avail == ActionAvailability.UNSUPPORTED:
                        self._movement_hint.setText(tr(
                            "behavior.movement.unsupported",
                            "Action '{action}' is not supported on {brand}/{model}.",
                        ).format(
                            action=display_name,
                            brand=self._brand or "?",
                            model=self._robot_type or "?",
                        ))
                    elif avail == ActionAvailability.DEGRADED:
                        self._movement_hint.setText(tr(
                            "behavior.movement.degraded",
                            "Action '{action}' has limited support on {brand}/{model}.",
                        ).format(
                            action=display_name,
                            brand=self._brand or "?",
                            model=self._robot_type or "?",
                        ))
                # STEP 5.3: descriptor parameters are primary; motor overlay is fallback.
                if descriptor is not None and descriptor.parameters:
                    rows = self._descriptor_param_rows(
                        descriptor,
                        live_params={
                            **dict(action_seg.params),
                            "start_time": action_seg.start_time,
                            "duration": action_seg.duration,
                        },
                    )
                    # Clear hint only when AVAILABLE (don't overwrite avail warning).
                    if descriptor.availability == ActionAvailability.AVAILABLE:
                        self._movement_hint.setText("")
                    self._render_descriptor_param_table(
                        rows,
                        descriptor,
                        commit_fn=lambda k, v, seg=action_seg: self._commit_action_seg_param(seg, k, v),
                    )
                    return
                # Fallback: motor overlay (for actions without descriptor params).
                if self._refresh_movement_settings_for_action_overlay(self._selected_module_index):
                    return
                if descriptor is None or descriptor.availability == ActionAvailability.AVAILABLE:
                    self._movement_hint.setText(tr(
                        "behavior.movement.no_motor_params",
                        "No motor parameters for this action.",
                    ))
            else:
                self._movement_selected_label.setText(tr("behavior.movement.selected_none", "Selected: None"))
                self._movement_hint.setText(tr(
                    "behavior.movement.select_hint",
                    "Select an Action block or a limb module in the timeline to inspect its parameters.",
                ))
            return

        # Legacy flat-module path (no structured timeline).
        if self._selected_module_index < 0 or self._selected_module_index >= len(self._modules):
            self._movement_selected_label.setText(tr("behavior.movement.selected_none", "Selected: None"))
            self._movement_hint.setText(tr(
                "behavior.movement.select_hint",
                "Select an Action block or a limb module in the timeline to inspect its parameters.",
            ))
            return

        module = self._modules[self._selected_module_index]
        self._movement_selected_label.setText(
            tr("behavior.movement.selected_format", "Selected Action: {node}").format(
                node=f"{module.kind}.{module.name}"
            )
        )

        if self._refresh_movement_settings_for_action_overlay(self._selected_module_index):
            return

        if module.kind != "movement":
            self._movement_hint.setText(tr(
                "behavior.movement.behavior_selected_hint",
                "Selected node is a behavior node. Movement settings are available only for movement nodes.",
            ))
            return

        self._movement_hint.setText("")
        # STEP 5.3: resolve descriptor for this module (by raw_action name) and
        # use descriptor.parameters as primary; fall back to parsed args when absent.
        _temp_seg = ActionSegment(
            action_id="",
            name=module.name,
            start_time=0.0,
            duration=module.duration,
            intent_id="",
        )
        module_descriptor = self._get_descriptor_for_segment(_temp_seg)
        parsed = self._parse_args(module.args)
        if module_descriptor is not None and module_descriptor.parameters:
            rows = self._descriptor_param_rows(
                module_descriptor,
                live_params={**dict(parsed), "duration": module.duration},
                parsed_args=parsed,
            )
            idx = self._selected_module_index
            self._render_descriptor_param_table(
                rows,
                module_descriptor,
                commit_fn=lambda k, v, i=idx: self._commit_module_param(i, k, v),
            )
            return

        # Legacy fallback: derive rows from duration + parsed args.
        rows: List[Tuple[str, Any]] = [("duration", module.duration)]
        for (k, v) in parsed:
            if k == "duration":
                continue
            rows.append((k, v))

        self._movement_params_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self._movement_params_table.setItem(r, 0, QTableWidgetItem(k))
            # Step 3: action-module params have no track context 閳?always constant, col 1 = label, col 2 = editor
            src_label = QLabel("Constant")
            src_label.setStyleSheet("color: #9e9e9e; font-size: 11px; padding: 0 4px;")
            self._movement_params_table.setCellWidget(r, 1, src_label)
            num_val = self._try_float(v)
            if num_val is None:
                editor = QLineEdit(str(v) if v is not None else "")
                editor.editingFinished.connect(
                    lambda e=editor, key_name=k, idx=self._selected_module_index: self._commit_module_param(idx, key_name, e.text())
                )
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(2)
                editor.setSingleStep(0.01)
                editor.setRange(-1_000_000.0, 1_000_000.0)
                editor.setValue(num_val)
                editor.editingFinished.connect(
                    lambda e=editor, key_name=k, idx=self._selected_module_index: self._commit_module_param(idx, key_name, e.value())
                )
            self._movement_params_table.setCellWidget(r, 2, editor)

    def _get_active_track_def(self, track_name: str) -> Optional[MotorTrackDef]:
        """Return the model-aware MotorTrackDef for track_name, or None.

        Never falls back to UNITREE_MOTOR_TRACK_MAP 閳?using Unitree metadata
        for a non-Unitree model would produce wrong labels and param keys.
        Returns None when the active model has no entry for this track.
        """
        from system.behavior.action_profile import get_motor_track_map
        try:
            tmap = get_motor_track_map(self._robot_type, brand=self._brand) or {}
            return tmap.get(track_name)
        except Exception:
            return None

    def _get_active_track_label(self, track_name: str) -> str:
        """Return a model-aware display label for track_name (falls back to track_name)."""
        td = self._get_active_track_def(track_name)
        return td.label if td is not None else track_name

    def _selected_action_segment(self) -> Optional[ActionSegment]:
        if self._behavior_timeline is not None and not self._behavior_timeline.is_empty():
            segs = getattr(self._behavior_timeline, "action_segments", []) or []
            if 0 <= self._selected_module_index < len(segs):
                return segs[self._selected_module_index]
        if 0 <= self._selected_module_index < len(self._modules):
            module = self._modules[self._selected_module_index]
            return ActionSegment(
                action_id="",
                name=module.name,
                start_time=0.0,
                duration=module.duration,
                intent_id="",
            )
        return None

    def _refresh_timeline_status(self) -> None:
        if not hasattr(self, "_timeline_status_label"):
            return
        sel = self._panel_selection
        text = "Flow: Action timeline -> limb modules -> Movement Settings"
        if sel.level == self._SelectionLevel.MOTOR and sel.track_name:
            text = f"Selected limb module: {self._get_active_track_label(sel.track_name)}"
        elif sel.level == self._SelectionLevel.BODY_PART and sel.limb_id:
            text = f"Selected body part: {sel.limb_id}"
        elif sel.level == self._SelectionLevel.ACTION:
            seg = self._selected_action_segment()
            if seg is not None:
                descriptor = self._get_descriptor_for_segment(seg)
                text = (
                    f"Selected action: {descriptor.display_name}"
                    if descriptor is not None and descriptor.display_name
                    else f"Selected action: {seg.name}"
                )
        elif self._behavior_timeline is None or self._behavior_timeline.is_empty():
            text = "Add or select an action package to author limb behavior over time."
        self._timeline_status_label.setText(text)

    def _render_body_part_settings(self, limb_id: str, action_id: Optional[str] = None) -> None:
        """Step 6: render Movement Settings for the selected body-part.

        Layout:
          [Body Part: <limb_label>]           閳?spanning header
            [Motor: <joint_name> (<track>)]   閳?per-motor header (collapsible)
              start_time | Constant | editor
              duration   | Constant | editor
              <param_key>| Constant | editor
              ...
          [Motor not yet authored]            閳?when no segment found
        """
        from system.behavior.motor_topology import resolve_topology

        table = self._movement_params_table
        table.setRowCount(0)
        table.clearContents()
        table.clearSpans()
        self._movement_group_header_rows = {}

        # 閳光偓閳光偓 1. Resolve canonical topology for the active model 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        topology = resolve_topology(self._brand, self._robot_type)
        limb: Optional[Any] = None
        for l in topology.all_limbs:
            if l.limb_id == limb_id:
                limb = l
                break

        if topology.is_degraded or limb is None:
            reason = topology.degraded_reason or "No canonical topology available"
            self._movement_selected_label.setText(
                f"Body Part: {limb_id}"
            )
            self._movement_hint.setText(
                f"Degraded: {reason}. Canonical motor detail is unavailable."
            )
            return

        self._movement_selected_label.setText(
            f"Selected Body Part: {limb.label}"
        )
        self._movement_hint.setText(
            "Showing authored limb modules for this body part. Edits update the timeline directly."
        )

        # 閳光偓閳光偓 2. Build track 閳?motor segments map (scoped to action when known) 閳光偓
        timeline = self._behavior_timeline
        track_to_segs: Dict[str, List[Any]] = {}
        if timeline is not None:
            overlays = timeline.motor_overlays
            if action_id:
                overlays = [o for o in overlays if getattr(o, "action_id", "") == action_id]
            for overlay in overlays:
                for seg in overlay.motor_segments:
                    track_to_segs.setdefault(seg.track_name, []).append(seg)

        # 閳光偓閳光偓 3. Render body-part header spanning all 3 columns 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        row = 0
        body_header = QTableWidgetItem(f"Body Part: {limb.label}")
        body_header.setFlags(Qt.ItemFlag.ItemIsEnabled)
        body_header.setBackground(QColor(get_color("behavior_movement_group_header_bg", "#243447")))
        body_header.setForeground(QColor(get_color("behavior_movement_group_header_text", "#dbe7f5")))
        table.insertRow(row)
        table.setItem(row, 0, body_header)
        table.setSpan(row, 0, 1, 3)
        row += 1

        # 閳光偓閳光偓 4. Render each motor in the limb 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
        for motor in limb.motors:
            tname = motor.track_name
            joint_label = motor.joint_name or motor.track_name
            segs = track_to_segs.get(tname, [])
            expanded = self._movement_track_groups_expanded.get(
                f"bp:{limb_id}:{tname}", True
            )

            # Motor header row (collapsible)
            authored_note = f"  ({len(segs)} segment{'s' if len(segs) != 1 else ''})" if segs else "  [not yet authored]"
            chevron = "▼" if expanded else "▶"
            motor_header = QTableWidgetItem(f"  {chevron} {joint_label}  ({tname}){authored_note}")
            motor_header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            motor_header.setBackground(QColor("#1e293b"))
            motor_header.setForeground(QColor("#94a3b8"))
            table.insertRow(row)
            table.setItem(row, 0, motor_header)
            table.setSpan(row, 0, 1, 3)
            # Store collapse key so _on_movement_param_cell_clicked can toggle
            self._movement_group_header_rows[row] = f"bp:{limb_id}:{tname}"
            row += 1

            if not expanded:
                continue

            if not segs:
                # Motor exists in topology but nothing authored yet
                note_item = QTableWidgetItem("    (no segments authored for this motor)")
                note_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                note_item.setForeground(QColor("#6b7280"))
                table.insertRow(row)
                table.setItem(row, 0, note_item)
                table.setSpan(row, 0, 1, 3)
                row += 1
                continue

            multiple_segs = len(segs) > 1
            for seg in segs:
                if multiple_segs:
                    seg_label = QTableWidgetItem(f"    Segment {seg.motor_id[:8]}")
                    seg_label.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    seg_label.setForeground(QColor("#64748b"))
                    table.insertRow(row)
                    table.setItem(row, 0, seg_label)
                    table.setSpan(row, 0, 1, 3)
                    row += 1

                # Param rows: start_time, duration, primary param_key, extras
                param_rows: List[Tuple[str, Any]] = [
                    ("start_time", seg.start_time),
                    ("duration", seg.duration),
                ]
                # Canonical param_key from topology
                if motor.param_key and motor.param_key in seg.params:
                    param_rows.append((motor.param_key, seg.params[motor.param_key]))
                # Remaining params not already listed
                already = {k for k, _ in param_rows}
                for k, v in seg.params.items():
                    if k not in already:
                        param_rows.append((k, v))

                for k, v in param_rows:
                    table.insertRow(row)
                    key_label = f"    {k}" if not multiple_segs else f"    {seg.motor_id[:6]}.{k}"
                    table.setItem(row, 0, QTableWidgetItem(key_label))
                    self._set_param_editor_3col(
                        row=row,
                        track_name=tname,
                        key=str(k),
                        value=v,
                        on_commit_constant=lambda key_name, val, s=seg: self._commit_motor_param(s, key_name, val),
                    )
                    row += 1

    def _refresh_movement_settings_for_action_overlay(self, action_index: int) -> bool:
        timeline = self._behavior_timeline
        if timeline is None:
            return False
        if action_index < 0 or action_index >= len(timeline.action_segments):
            return False
        action_seg = timeline.action_segments[action_index]
        overlay = timeline.get_overlay_for_action(action_seg.action_id)
        if overlay is None:
            # Backward-compatible fallback for legacy timelines where overlay.action_id
            # is stale/missing after rebuild. Keep index-aligned behavior usable.
            if 0 <= action_index < len(timeline.motor_overlays):
                overlay = timeline.motor_overlays[action_index]
        if overlay is None or not overlay.motor_segments:
            return False

        segments_by_track: Dict[str, List[MotorSegment]] = {}
        for seg in overlay.motor_segments:
            segments_by_track.setdefault(seg.track_name, []).append(seg)

        ordered_tracks: List[str] = []
        for tname in timeline.active_motor_tracks:
            if tname in segments_by_track:
                ordered_tracks.append(tname)
        for tname in segments_by_track.keys():
            if tname not in ordered_tracks:
                ordered_tracks.append(tname)
        if not ordered_tracks:
            return False

        self._movement_hint.setText(
            "Main action selected: parameters grouped by body part. "
            "Click group headers to fold/unfold."
        )

        row = 0
        for track_name in ordered_tracks:
            track_segments = segments_by_track.get(track_name, [])
            if not track_segments:
                continue
            track_label = self._get_active_track_label(track_name)
            expanded = self._movement_track_groups_expanded.get(track_name, True)

            self._movement_params_table.insertRow(row)
            chevron = "▼" if expanded else "▶"
            header_item = QTableWidgetItem(f"{chevron} {track_label}")
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header_item.setBackground(QColor(get_color("behavior_movement_group_header_bg", "#243447")))
            header_item.setForeground(QColor(get_color("behavior_movement_group_header_text", "#dbe7f5")))
            self._movement_params_table.setItem(row, 0, header_item)
            self._movement_params_table.setSpan(row, 0, 1, 3)
            self._movement_group_header_rows[row] = track_name
            row += 1

            if not expanded:
                continue

            multiple_segments = len(track_segments) > 1
            for seg in track_segments:
                if multiple_segments:
                    self._movement_params_table.insertRow(row)
                    seg_item = QTableWidgetItem(f"Motor {seg.motor_id[:8]}")
                    seg_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self._movement_params_table.setItem(row, 0, seg_item)
                    self._movement_params_table.setSpan(row, 0, 1, 3)
                    row += 1

                rows: List[Tuple[str, Any]] = [("start_time", seg.start_time), ("duration", seg.duration)]
                track_def = self._get_active_track_def(seg.track_name)
                if track_def and track_def.param_key in seg.params:
                    rows.append((track_def.param_key, seg.params.get(track_def.param_key)))
                for k, v in seg.params.items():
                    if k not in {rk for rk, _ in rows}:
                        rows.append((k, v))

                for k, v in rows:
                    self._movement_params_table.insertRow(row)
                    key_label = str(k) if not multiple_segments else f"{seg.motor_id[:6]}.{k}"
                    self._movement_params_table.setItem(row, 0, QTableWidgetItem(key_label))
                    self._set_param_editor_3col(
                        row=row,
                        track_name=seg.track_name,
                        key=str(k),
                        value=v,
                        on_commit_constant=lambda key_name, val, s=seg: self._commit_motor_param(s, key_name, val),
                    )
                    row += 1

        return row > 0

    def _on_movement_param_cell_clicked(self, row: int, _column: int) -> None:
        track_name = self._movement_group_header_rows.get(row)
        if not track_name:
            return
        current = self._movement_track_groups_expanded.get(track_name, True)
        self._movement_track_groups_expanded[track_name] = not current
        self._refresh_movement_settings()

    def _get_selected_motor_segment(self) -> Optional[MotorSegment]:
        if self._behavior_timeline is None:
            return None
        sel = self._panel_selection
        if sel.level.value != "motor" or not sel.motor_id or not sel.track_name:
            return None
        for overlay in self._behavior_timeline.motor_overlays:
            for seg in overlay.motor_segments:
                if seg.motor_id == sel.motor_id and seg.track_name == sel.track_name:
                    return seg
        return None

    def _set_param_editor(
        self,
        row: int,
        key: str,
        value: Any,
        on_commit,
    ) -> None:
        num_val = self._try_float(value)
        if num_val is None:
            editor = QLineEdit(str(value))
            editor.editingFinished.connect(
                lambda e=editor, k=key: on_commit(k, e.text())
            )
        else:
            editor = QDoubleSpinBox()
            editor.setDecimals(2)
            editor.setSingleStep(0.01)
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setValue(num_val)
            editor.editingFinished.connect(
                lambda e=editor, k=key: on_commit(k, e.value())
            )
        self._movement_params_table.setCellWidget(row, 1, editor)

    def _commit_motor_param(self, seg: MotorSegment, key: str, value: Any) -> None:
        changed = False
        if key == "start_time":
            v = self._try_float(value)
            if v is not None:
                new_val = max(0.0, v)
                changed = abs(float(seg.start_time) - float(new_val)) > 1e-9
                if changed:
                    seg.start_time = new_val
        elif key == "duration":
            v = self._try_float(value)
            if v is not None:
                new_val = max(0.01, v)
                changed = abs(float(seg.duration) - float(new_val)) > 1e-9
                if changed:
                    seg.duration = new_val
        else:
            f = self._try_float(value)
            new_val = f if f is not None else value
            old_val = seg.params.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                changed = abs(float(old_val) - float(new_val)) > 1e-9
            else:
                changed = old_val != new_val
            if changed:
                seg.params[key] = new_val
        if not changed:
            return
        self._persist_current_timeline_draft()
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _commit_module_param(self, index: int, key: str, value: Any) -> None:
        self._ensure_behavior_timeline()
        segs = self._behavior_timeline.action_segments
        if index < 0 or index >= len(segs):
            return
        seg = segs[index]
        changed = False
        if key == "duration":
            f = self._try_float(value)
            if f is not None:
                new_val = max(0.01, f)
                changed = abs(float(seg.duration) - float(new_val)) > 1e-9
                if changed:
                    seg.duration = new_val
                    seg.params.pop("duration", None)
        else:
            new_val = self._coerce_param_value(value)
            old_val = seg.params.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                changed = abs(float(old_val) - float(new_val)) > 1e-9
            else:
                changed = old_val != new_val
            if changed:
                seg.params[key] = new_val
        if not changed:
            return
        self._rebuild_timeline_from_action_segments(mark_dirty=True)

    @staticmethod
    def _try_float(v: Any) -> Optional[float]:
        try:
            if isinstance(v, str) and not v.strip():
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    def _sync_modules_from_source(self) -> None:
        source = self._core_editor.toPlainText()
        parsed = self._parse_modules(source)
        self._selected_module_index = -1
        self._behavior_timeline = self._build_or_update_timeline(
            parsed, preserve_action_edits=False
        )
        self._sync_views_from_timeline(mark_dirty=False)
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._refresh_navigator()

    def _build_or_update_timeline(
        self,
        modules: List[SequenceModule],
        preserve_action_edits: bool = True,
    ) -> Optional[BehaviorTimeline]:
        """Build a BehaviorTimeline from the current SequenceModule list.

        Preserve authored motor overlays only when the rebuilt timeline stays in
        the same authored track space. A model switch can change track keys from
        abstract limb tracks to joint-level tracks, and carrying old overlays
        across that boundary would leave visible rows with no matching segments.
        """
        new_timeline = build_timeline_from_modules(
            modules,
            robot_type=self._robot_type or "go2",
            auto_decompose=True,
        )
        if self._behavior_timeline is not None and preserve_action_edits:
            existing = self._behavior_timeline
            same = (
                len(existing.action_segments) == len(new_timeline.action_segments)
                and all(
                    a.name == b.name
                    for a, b in zip(existing.action_segments, new_timeline.action_segments)
                )
            )
            if same:
                # Carry forward user-edited ActionSegment fields from the existing
                # timeline so a rebuild never silently resets authored values back
                # to descriptor/module defaults.
                for old_seg, new_seg in zip(
                    existing.action_segments, new_timeline.action_segments
                ):
                    new_seg.start_time = float(old_seg.start_time)
                    new_seg.duration = float(old_seg.duration)
                    if old_seg.params:
                        new_seg.params = dict(old_seg.params)
                compatible_track_space = (
                    (existing.robot_type or "") == (new_timeline.robot_type or "")
                    and list(existing.active_motor_tracks or []) == list(new_timeline.active_motor_tracks or [])
                )
                rebound_overlays: List[ActionMotorOverlay] = []
                for idx, new_action in enumerate(new_timeline.action_segments):
                    if compatible_track_space and idx < len(existing.motor_overlays):
                        old_overlay = existing.motor_overlays[idx]
                        if old_overlay.motor_segments:
                            rebound_segments: List[MotorSegment] = []
                            for old_seg in old_overlay.motor_segments:
                                seg_copy = MotorSegment.from_dict(old_seg.to_dict())
                                seg_copy.parent_action_id = new_action.action_id
                                rebound_segments.append(seg_copy)
                            rebound_overlays.append(ActionMotorOverlay(
                                action_id=new_action.action_id,
                                motor_segments=rebound_segments,
                                expanded=old_overlay.expanded,
                            ))
                            continue
                    if idx < len(new_timeline.motor_overlays):
                        fresh = new_timeline.motor_overlays[idx]
                        rebound_overlays.append(ActionMotorOverlay(
                            action_id=new_action.action_id,
                            motor_segments=list(fresh.motor_segments),
                        ))
                    else:
                        rebound_overlays.append(
                            ActionMotorOverlay(action_id=new_action.action_id)
                        )
                new_timeline.motor_overlays = rebound_overlays
                if compatible_track_space and existing.active_motor_tracks:
                    new_timeline.active_motor_tracks = existing.active_motor_tracks
        return new_timeline

    def _timeline_dict_for_slot(self, slot_key: str) -> Optional[Dict[str, Any]]:
        entry = self._drafts_by_node.get(slot_key, {})
        timeline_raw = entry.get("timeline")
        if isinstance(timeline_raw, dict):
            return dict(timeline_raw)
        legacy = self._timelines_by_node.get(slot_key)
        if isinstance(legacy, dict):
            return dict(legacy)
        return None

    @staticmethod
    def _format_module_arg_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    def _module_from_action_segment(self, seg: ActionSegment) -> SequenceModule:
        pairs: List[Tuple[str, str]] = [
            ("duration", self._format_module_arg_value(float(seg.duration)))
        ]
        for key, value in dict(seg.params).items():
            if key in STRUCTURAL_KEYS:
                continue
            pairs.append((str(key), self._format_module_arg_value(value)))
        args = ", ".join(f"{k}={v}" if v != "" else k for k, v in pairs)
        return SequenceModule(
            kind=str(getattr(seg, "kind", "movement") or "movement"),
            name=str(getattr(seg, "name", "") or ""),
            args=args,
            duration=float(getattr(seg, "duration", 1.0) or 1.0),
        )

    def _ensure_behavior_timeline(self) -> None:
        if self._behavior_timeline is None:
            self._behavior_timeline = _default_timeline_for_behavior_selection(
                self._intent_id,
                self._node_name,
                self._brand,
                self._robot_type,
            )

    def _sync_views_from_timeline(self, mark_dirty: bool = False) -> None:
        if self._behavior_timeline is None:
            self._modules = []
            self._set_core_source("", mark_dirty=mark_dirty)
            return
        self._modules = [
            self._module_from_action_segment(seg)
            for seg in self._behavior_timeline.action_segments
        ]
        lines = []
        for mod in self._modules:
            args = mod.args.strip()
            lines.append(f"{mod.name}({args})" if args else f"{mod.name}()")
        self._set_core_source("\n".join(lines), mark_dirty=mark_dirty)

    def _persist_current_behavior_draft(self) -> None:
        if self._node_id < 0:
            return
        slot_key = self._draft_slot_key(
            self._node_id, self._behavior_ref or self._node_name
        )
        self._sync_views_from_timeline(mark_dirty=False)
        hb_payload = (
            self._collect_hb_draft().to_dict()
            if hasattr(self, "_heartbeat_editor")
            else HBDraftPayload.default().to_dict()
        )
        entry: Dict[str, Any] = {
            "core": self._core_editor.toPlainText(),
            "hb": hb_payload,
            "motor_track_expanded": dict(getattr(self._timeline, "_motor_track_expanded", {})),
        }
        if self._behavior_timeline is not None:
            entry["timeline"] = self._behavior_timeline.to_dict()
        self._drafts_by_node[slot_key] = entry

    def _persist_current_timeline_draft(self) -> None:
        """Persist the active structured timeline into the current node draft."""
        if self._node_id < 0 or self._behavior_timeline is None:
            return
        self._persist_current_behavior_draft()

    def _rebuild_timeline_from_action_segments(self, mark_dirty: bool = True) -> None:
        self._sync_views_from_timeline(mark_dirty=mark_dirty)
        self._behavior_timeline = self._build_or_update_timeline(
            self._modules, preserve_action_edits=True
        )
        self._persist_current_timeline_draft()
        self._refresh_timeline()
        self._refresh_movement_settings()
        if mark_dirty:
            self._mark_dirty()

    def _sync_source_from_modules(self) -> None:
        self._behavior_timeline = self._build_or_update_timeline(
            list(self._modules), preserve_action_edits=False
        )
        self._sync_views_from_timeline(mark_dirty=True)
        self._persist_current_timeline_draft()

    def _on_reset(self) -> None:
        """Restore the panel to the snapshot taken at init (set_node_context time)."""
        if self._init_snapshot is None:
            return
        snap = self._init_snapshot
        # Restore core source
        self._set_core_source(snap.get("core", ""), mark_dirty=False)
        # Restore HB draft
        hb_dict = snap.get("hb")
        if hb_dict is not None:
            try:
                self._apply_hb_draft(HBDraftPayload.from_dict(hb_dict))
            except Exception:
                self._apply_hb_draft(HBDraftPayload.default())
        else:
            self._apply_hb_draft(HBDraftPayload.default())
        # Restore structured timeline 閳?run Circle 4 migration on load
        tl_dict = snap.get("timeline")
        if tl_dict is not None:
            self._behavior_timeline = self._load_timeline_with_migration(tl_dict)
            if self._behavior_timeline.is_empty() and not tl_dict.get("action_segments"):
                self._behavior_timeline = None
        else:
            self._behavior_timeline = None
        # Clear dirty state for this node
        if self._node_id >= 0:
            self._dirty_nodes.discard(self._node_id)
        self._sync_views_from_timeline(mark_dirty=False)
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._refresh_navigator()
        self._refresh_registries()
        self._refresh_module_library()
        self._refresh_save_button_state()

    def _on_save_as(self) -> None:
        """Save current behavior settings to a new custom ref with user-supplied name."""
        name, ok = QInputDialog.getText(
            self,
            tr("behavior.save_as.dialog_title", "Save As Custom Behavior"),
            tr("behavior.save_as.dialog_label", "Behavior name (saved to Customs):"),
        )
        if not ok or not name.strip():
            return
        new_ref = name.strip()
        # Disallow overwriting system behaviors via Save as
        if new_ref in UNITREE_ACTION_PROFILES:
            QMessageBox.warning(
                self,
                tr("behavior.save_as.system_conflict_title", "Name Conflict"),
                tr(
                    "behavior.save_as.system_conflict_msg",
                    "'{name}' is a built-in system behavior and cannot be used as a custom name. "
                    "Please choose a different name.",
                ).format(name=new_ref),
            )
            return
        if self._compile_worker is not None and self._compile_worker.isRunning():
            QMessageBox.information(
                self,
                tr("behavior.save_as.busy_title", "Busy"),
                tr("behavior.save_as.busy_msg", "A compile is already in progress. Please wait."),
            )
            return
        self._sync_views_from_timeline(mark_dirty=False)
        source = self._core_editor.toPlainText()
        self._output.append(f"[Save As] behavior_ref='{new_ref}' source_len={len(source)} chars")
        if hasattr(self, "_save_as_btn"):
            self._save_as_btn.setEnabled(False)
        robot_type = self._robot_type or "go2"
        self._compile_worker = BehaviorCompileWorker(
            bridge=self._bridge,
            source=source,
            behavior_ref=new_ref,
            robot_type=robot_type,
            timeline=self._behavior_timeline,
            is_simulation=self._is_simulation,
            parent=self,
        )
        self._compile_worker.compile_done.connect(self._on_compile_done)
        self._compile_worker.finished.connect(self._on_save_as_finished)
        self._compile_worker.start()

    def _on_save_as_finished(self) -> None:
        if hasattr(self, "_save_as_btn"):
            self._save_as_btn.setEnabled(True)
        self._compile_worker = None
        self._refresh_save_button_state()

    def _run_compile(self) -> None:
        if not self._node_name:
            self._output.append("[Compile] No behavior node selected; open from Mission first.")
            return
        if self._compile_worker is not None and self._compile_worker.isRunning():
            self._output.append("[Compile] Compile already in progress.")
            return

        # Warn before overwriting a system (built-in) behavior
        if self._node_name in UNITREE_ACTION_PROFILES:
            reply = QMessageBox.question(
                self,
                tr("behavior.save.system_confirm_title", "Save System Behavior"),
                tr(
                    "behavior.save.system_confirm_msg",
                    "'{name}' is a built-in system behavior.\n"
                    "Saving will override the system definition with your edits.\n\n"
                    "Continue?",
                ).format(name=self._node_name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._sync_views_from_timeline(mark_dirty=False)
        behavior_ref = self._behavior_ref or self._node_name
        source = self._core_editor.toPlainText()
        self._output.append(f"[Compiling] behavior_ref='{behavior_ref}' source_len={len(source)} chars")

        if hasattr(self, "_compile_btn"):
            self._compile_btn.setEnabled(False)
            self._compile_btn.setText(tr("behavior.header.saving", "Saving..."))

        robot_type = self._robot_type or "go2"
        self._compile_worker = BehaviorCompileWorker(
            bridge=self._bridge,
            source=source,
            behavior_ref=behavior_ref,
            robot_type=robot_type,
            timeline=self._behavior_timeline,
            is_simulation=self._is_simulation,
            parent=self,
        )
        self._compile_worker.compile_done.connect(self._on_compile_done)
        self._compile_worker.finished.connect(self._on_compile_finished)
        self._compile_worker.start()

    def _on_compile_done(self, artifact) -> None:
        self._output.append(_format_compile_output(artifact))
        self._ctx_artifact.setText(f"{artifact.artifact_id[:8]}...")
        self._ctx_compile_time.setText(str(artifact.compiled_at))
        self._ctx_diag.setText(f"{artifact.error_count} error(s), {artifact.warning_count} warning(s)")
        if artifact.is_valid:
            self._hb_status_label.setText(
                f"Core compiled OK 閳?0 errors, {artifact.warning_count} warning(s)"
            )
        else:
            self._hb_status_label.setText(
                f"Core compiled 閳?{artifact.error_count} error(s), {artifact.warning_count} warning(s)"
            )
        # Store compiled snapshot (separate from transient draft state)
        self._hb_compiled_snapshot = artifact.to_dict()

        self._refresh_registries()
        self._refresh_module_library()

        if artifact.is_valid and self._node_id >= 0:
            self._dirty_nodes.discard(self._node_id)

    def _on_compile_finished(self) -> None:
        self._refresh_save_button_state()
        self._compile_worker = None

    def _refresh_breadcrumb(self) -> None:
        node = self._node_name or "-"
        self._breadcrumb.setText(
            tr("behavior.header.breadcrumb", "Mission / {node} / Behavior").format(node=node)
        )

    def apply_theme(self) -> None:
        self._timeline.apply_theme()
        self._apply_module_library_tree_style()
        self._apply_button_styles()

    def _apply_action_button_style(
        self,
        button: QPushButton,
        *,
        bg: str,
        text: str,
        border: str,
        hover_bg: str,
        pressed_bg: str,
        disabled_bg: str,
        disabled_text: str,
        disabled_border: str,
    ) -> None:
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(
            f"""
            QPushButton#{button.objectName()} {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 0 12px;
                text-align: left;
            }}
            QPushButton#{button.objectName()}:hover:!disabled {{
                background: {hover_bg};
            }}
            QPushButton#{button.objectName()}:pressed:!disabled {{
                background: {pressed_bg};
            }}
            QPushButton#{button.objectName()}:disabled {{
                background: {disabled_bg};
                color: {disabled_text};
                border: 1px solid {disabled_border};
            }}
            """
        )

    def _apply_button_styles(self) -> None:
        base_bg = get_color("button_bg", "#111827")
        base_text = get_color("button_text", "#e5e7eb")
        base_border = get_color("button_border", "#4b5563")
        hover_bg = get_color("button_hover_bg", get_color("hover_bg", "#1f2937"))
        pressed_bg = get_color("button_pressed_bg", "#0f172a")
        disabled_bg = get_color("button_disabled_bg", "#2f2f2f")
        disabled_text = get_color("button_disabled_text", "#8a8a8a")
        disabled_border = get_color("button_disabled_border", "#4b5563")

        buttons = [
            self._back_btn,
            self._hb_canvas_btn,
            self._hb_compiler_btn,
            self._reset_btn,
            self._hb_compile_btn,
            self._hb_simulate_btn,
            self._hb_run_btn,
        ]
        for button in buttons:
            if not button.objectName():
                button.setObjectName(f"behaviorBtn_{id(button)}")
            self._apply_action_button_style(
                button,
                bg=base_bg,
                text=base_text,
                border=base_border,
                hover_bg=hover_bg,
                pressed_bg=pressed_bg,
                disabled_bg=disabled_bg,
                disabled_text=disabled_text,
                disabled_border=disabled_border,
            )


    def _apply_module_library_tree_style(self) -> None:
        text_primary = get_color("text_primary", "#e5e7eb")
        hover_bg = get_color("hover_bg", "#111827")
        selected_bg = get_color("card_bg", "#1f2937")
        style = f"""
            QTreeWidget {{
                background: transparent;
                border: none;
                color: {text_primary};
            }}
            QTreeWidget::item {{
                padding: 4px 6px;
                margin: 1px 0px;
                border: 1px solid transparent;
                border-radius: 6px;
            }}
            QTreeWidget::item:hover {{
                background: {hover_bg};
                border: 1px solid {hover_bg};
            }}
            QTreeWidget::item:selected {{
                background: {selected_bg};
                border: 1px solid {selected_bg};
                color: {text_primary};
            }}
            """
        self._module_library_tree.setStyleSheet(style)
        if hasattr(self, "_hb_library_tree"):
            self._hb_library_tree.setStyleSheet(style)

    def refresh_texts(self) -> None:
        self._refresh_breadcrumb()
        self._reset_btn.setText(tr("behavior.header.reset", "Reset"))
        self._back_btn.setText(tr("behavior.header.back_mission", "<- Mission"))

        self._context_title.setText(tr("behavior.library.context", "Context"))
        self._ctx_node_key.setText(f"{tr('behavior.context.mission_node', 'Mission Node')}:")
        self._ctx_node_id_key.setText(f"{tr('behavior.context.node_id', 'Node ID')}:")
        self._ctx_ref_key.setText(f"{tr('behavior.context.behavior_ref', 'Behavior Ref')}:")
        self._ctx_artifact_key.setText(f"{tr('behavior.context.last_artifact', 'Last Artifact')}:")
        self._ctx_compile_time_key.setText(f"{tr('behavior.context.last_compile', 'Last Compile')}:")
        self._ctx_diag_key.setText(f"{tr('behavior.context.diagnostics', 'Diagnostics')}:")
        self._core_library_title.setText(tr("behavior.library.core_library", "Core Library"))
        self._hb_library_title.setText(tr("behavior.library.hb_library", "HeartBeat Library"))

        self._hb_canvas_btn.setText(tr("behavior.heartbeat.canvas", "Canvas"))
        self._hb_compiler_btn.setText(tr("behavior.heartbeat.compiler", "Compiler"))
        self._heartbeat_editor.setPlaceholderText(tr("behavior.heartbeat.script_placeholder", "# HeartBeat scripts"))
        self._hb_compile_btn.setText(tr("behavior.heartbeat.hb_compile", "HB Compile"))
        self._hb_simulate_btn.setText(tr("behavior.heartbeat.simulate", "Simulate"))
        self._hb_run_btn.setText(tr("behavior.heartbeat.run_hb", "Run HB"))

        self._timeline_title.setText(tr("behavior.timeline.title", "Behavior Timeline"))

        self._movement_settings_title.setText(tr("behavior.movement.title", "Movement Settings"))
        self._movement_params_table.setHorizontalHeaderLabels([
            tr("behavior.movement.parameter", "Parameter"),
            tr("behavior.movement.source", "Source"),
            tr("behavior.movement.value", "Value"),
        ])

        self._io_mapping_title.setText(tr("behavior.io_mapping.title", "Movement IO Mapping"))
        self._io_mapping_table.setHorizontalHeaderLabels([
            tr("behavior.io_mapping.signal", "Signal"),
            tr("behavior.io_mapping.target", "Target Param"),
            tr("behavior.io_mapping.direction", "Direction"),
            tr("behavior.io_mapping.status", "Status"),
        ])

        self._event_state_title.setText(tr("behavior.event_state.title", "Event State"))
        self._event_state_table.setHorizontalHeaderLabels([
            tr("behavior.event_state.event", "Event"),
            tr("behavior.event_state.state", "State"),
            tr("behavior.event_state.source", "Source"),
            tr("behavior.event_state.timestamp", "Timestamp"),
            tr("behavior.event_state.reason", "Reason"),
        ])

        self._bottom_tabs.setTabText(0, tr("behavior.tabs.movement_io", "Movement IO"))
        self._bottom_tabs.setTabText(1, tr("behavior.tabs.event_state", "Event State"))
        self._bottom_tabs.setTabText(2, tr("behavior.tabs.source", "Source"))

        self._core_source_title.setText(tr("behavior.source.title", "Core Source"))
        self._core_editor.setPlaceholderText(tr("behavior.source.placeholder", "# movement and behavior calls"))
        self._diagnostics_title.setText(tr("behavior.diagnostics.title", "Diagnostics"))

        self._refresh_module_library()
        self._refresh_hb_library()
        self._refresh_movement_settings()
        self._refresh_timeline_status()
        # Re-render panels so placeholder / summary text uses updated locale (Step 1.5)
        self._refresh_event_state()
        self._refresh_movement_io_mapping()
        self.apply_theme()
        self._refresh_save_button_state()

    def _on_editor_changed(self) -> None:
        if self._suspend_dirty_tracking:
            return
        self._mark_dirty()

    def _set_core_source(self, source: str, mark_dirty: bool = False) -> None:
        self._suspend_dirty_tracking = True
        self._core_editor.blockSignals(True)
        self._core_editor.setPlainText(source)
        self._core_editor.blockSignals(False)
        self._suspend_dirty_tracking = False
        if mark_dirty:
            self._mark_dirty()

    def _set_heartbeat_source(self, source: str, mark_dirty: bool = False) -> None:
        self._suspend_dirty_tracking = True
        self._heartbeat_editor.blockSignals(True)
        self._heartbeat_editor.setPlainText(source)
        self._heartbeat_editor.blockSignals(False)
        self._suspend_dirty_tracking = False
        if mark_dirty:
            self._mark_dirty()

    def _mark_dirty(self) -> None:
        if self._node_id >= 0:
            self._dirty_nodes.add(self._node_id)
        self._refresh_save_button_state()

    def _refresh_save_button_state(self) -> None:
        running = self._compile_worker is not None and self._compile_worker.isRunning()
        if hasattr(self, "_reset_btn"):
            self._reset_btn.setEnabled((self._node_id >= 0) and (self._init_snapshot is not None) and (not running))

    def _parse_modules(self, source: str) -> List[SequenceModule]:
        modules: List[SequenceModule] = []
        for raw in source.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", line)
            if not m:
                continue
            name = m.group(1).strip()
            args = m.group(2).strip()
            duration = self._extract_duration(args)
            kind = "movement" if name in self._movement_registry else "behavior"
            modules.append(SequenceModule(kind=kind, name=name, args=args, duration=duration))
        return modules

    @staticmethod
    def _extract_duration(args: str) -> float:
        if not args:
            return 1.0
        m = re.search(r"(?:^|,)\s*duration\s*=\s*([0-9]+(?:\.[0-9]+)?)", args)
        if not m:
            return 1.0
        try:
            v = float(m.group(1))
            return max(0.25, v)
        except ValueError:
            return 1.0

    @staticmethod
    def _parse_args(args: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if not args.strip():
            return out
        for part in args.split(","):
            text = part.strip()
            if not text:
                continue
            if "=" in text:
                k, v = text.split("=", 1)
                out.append((k.strip(), v.strip()))
            else:
                out.append((text, ""))
        return out

    def _resolve_behavior_children(self, behavior_ref: str) -> List[SequenceModule]:
        children: List[SequenceModule] = []
        try:
            result = self._bridge.resolve(behavior_ref)
            if not result.ok or result.artifact is None:
                return children
            ir = result.artifact.behavior_ir
        except Exception:
            return children

        for node in getattr(ir, "nodes", []):
            schema_id = str(getattr(node, "schema_id", ""))
            params = getattr(node, "params", {}) or {}

            if "action_execution" in schema_id:
                action = self._extract_param_value(params, "action") or "movement"
                children.append(SequenceModule(kind="movement", name=str(action), args="duration=1.0", duration=1.0))
            elif schema_id == "behavior" or "external_kind" in str(getattr(node, "to_dict", lambda: {})()):
                ref = self._extract_param_value(params, "behavior_ref") or "behavior"
                children.append(SequenceModule(kind="behavior", name=str(ref), args="duration=1.0", duration=1.0))

        return children

    @staticmethod
    def _extract_param_value(params: Dict, key: str) -> Optional[str]:
        raw = params.get(key)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return str(raw.get("value", ""))
        value = getattr(raw, "value", None)
        if value is not None:
            return str(value)
        return str(raw)


def _title(layout, text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("behaviorPaneTitle")
    if hasattr(layout, "addWidget"):
        layout.addWidget(label)
    return label


def _plain_title(layout, text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("background: transparent; border-radius: 0px;")
    if hasattr(layout, "addWidget"):
        layout.addWidget(label)
    return label


def _kv_row(layout, key: str, value: str) -> Tuple[QLabel, QLabel]:
    row = QHBoxLayout()
    row.setSpacing(8)
    k = QLabel(f"{key}:")
    k.setObjectName("behaviorDetailKey")
    k.setStyleSheet("background: transparent; padding: 0; margin: 0;")
    v = QLabel(value)
    v.setObjectName("behaviorDetailVal")
    v.setWordWrap(True)
    v.setStyleSheet(
        "background: rgba(255, 255, 255, 0.04); border-radius: 4px; padding: 2px 6px;"
    )
    row.addWidget(k)
    row.addWidget(v, 1)
    layout.addLayout(row)
    return k, v
















