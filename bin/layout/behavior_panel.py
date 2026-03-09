"""Behavior workspace with timeline-oriented sequence editing."""

from __future__ import annotations

import re
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

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
    UNITREE_MOTOR_TRACK_MAP,
    build_timeline_from_modules,
    validate_timeline,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
from system.behavior.hb_channel import (
    HBCompileRequest,
    HBDiagnosticsSnapshot,
    HBDryRunRequest,
    HBMockChannel,
    HBRunRequest,
    IHBChannel,
)
from system.behavior.hb_node_catalog import HBNodeAvailability, HBNodeCatalog
from system.behavior.hb_compat_audit import HBCompatReport, audit_catalog_compatibility
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

# Color token → hex string map for status badges (Step 1.5)
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
    module_reordered = Signal(int, int)
    delete_requested = Signal()
    timeline_edited = Signal()  # Fix 6: emitted after any motor seg mutation or track toggle
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
        self._motor_track_expanded: Dict[str, bool] = {}  # track_name → expanded
        self._motor_track_rows: Dict[str, Tuple[float, float, bool]] = {}

        # Fix 2: motor segment drag state
        self._motor_press_id: Optional[str] = None
        self._motor_press_track: Optional[str] = None
        self._motor_drag_started: bool = False
        self._motor_press_scene_x: float = 0.0
        self._motor_orig_start: float = 0.0
        self._selected_motor_id: Optional[str] = None
        self._selected_motor_track: Optional[str] = None

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
    ) -> None:
        """Load a structured BehaviorTimeline for multi-track rendering.

        Populates both the legacy _modules (Action Track) and the structured
        motor sub-tracks.  Existing drag/reorder logic continues to operate
        on _modules for backward-compat edit operations.

        Parameters
        ----------
        timeline        : BehaviorTimeline containing action_segments and
                          motor_overlays.
        selected_index  : Index of the selected ActionSegment (-1 for none).
        """
        self._behavior_timeline = timeline
        self._selected_index = selected_index

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

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())

        if self._behavior_timeline is not None:
            items = self._scene.items(scene_pos)
            for item in items:
                if item.data(1) == "track_toggle":
                    track_name = str(item.data(2) or "")
                    if track_name:
                        self.toggle_motor_track(track_name)
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

            # Fix 2: check for motor segment hit first (motor tracks are below main track)
            for item in items:
                if item.data(1) == "motor":
                    motor_id = item.data(0)
                    track_name = item.data(2)
                    self._selected_motor_id = str(motor_id or "")
                    self._selected_motor_track = str(track_name or "")
                    self._selected_index = -1
                    self.motor_segment_selected.emit(
                        self._selected_motor_id,
                        self._selected_motor_track,
                    )
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
                    track_def = UNITREE_MOTOR_TRACK_MAP.get(track_name)
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

        # No track titles by design.

    # Label lane width — recomputed each redraw to fit the widest label text.
    # Stored as instance variable so mouse-event coordinate translation stays consistent.
    _label_lane_w: int = 80  # initial default; overwritten by _redraw_multi_track()
    _motor_track_h: int = 36
    _motor_track_collapsed_h: int = 14

    def _redraw_multi_track(self) -> None:
        """Render main Action Track + motor sub-tracks from BehaviorTimeline."""
        timeline = self._behavior_timeline
        if timeline is None:
            return

        self._scene.clear()

        # Compute motor track labels first (needed for label width calculation)
        motor_track_labels = {
            t: UNITREE_MOTOR_TRACK_MAP[t].label if t in UNITREE_MOTOR_TRACK_MAP else t
            for t in self._motor_track_names
        }

        # Dynamically size the label lane to fit the widest label text.
        # Reserve space for fold chevrons and add extra safety padding so titles
        # do not clip on scaled displays.
        from PySide6.QtGui import QFontMetrics
        _fm = QFontMetrics(self.font())
        _all_label_texts = ["Action"] + [f"▼ {txt}" for txt in motor_track_labels.values()]
        _max_text_px = max((_fm.horizontalAdvance(s) for s in _all_label_texts), default=40)
        chevron_reserve_px = 10
        extra_redundancy_px = 18
        label_w = max(_max_text_px + 16 + chevron_reserve_px + extra_redundancy_px, 48)
        self._label_lane_w = label_w  # keep in sync for mouse-event coordinate translation

        # Calculate total height: ruler + action track + one row per motor track.
        total_motor_h = 0
        for tname in self._motor_track_names:
            expanded = self._motor_track_expanded.get(tname, True)
            total_motor_h += self._motor_track_h if expanded else self._motor_track_collapsed_h
            total_motor_h += 2
        total_h = self._ruler_h + self._track_h + total_motor_h + 8
        self._content_height = total_h
        self.setSceneRect(0.0, 0.0, self._scene_width, float(total_h))
        self._motor_track_rows = {}

        # Ruler: full-width background, then a dark placeholder over the label lane,
        # then ticks/labels starting at x=label_w so they align with track content.
        self._scene.addRect(0, 0, self._scene_width, self._ruler_h,
                            QPen(QColor("#454545")), QBrush(QColor("#272727")))
        self._scene.addRect(0, 0, label_w, self._ruler_h,
                            QPen(QColor("#3a3a3a")), QBrush(QColor("#1a1a1a")))
        self._draw_ruler(label_offset=label_w)

        # Action Track: label lane + content region
        action_y = float(self._ruler_h)
        # Label lane background
        self._scene.addRect(0, action_y, label_w, self._track_h,
                            QPen(QColor("#4a4a4a")), QBrush(QColor("#1a1a1a")))
        lbl = self._scene.addSimpleText("Action")
        lbl.setBrush(QBrush(QColor("#888888")))
        lbl.setPos(4, action_y + 4)
        # Content region background (offset by label_w)
        self._scene.addRect(label_w, action_y, self._scene_width - label_w, self._track_h,
                            QPen(QColor("#4f4f4f")), QBrush(QColor(self._main_track_bg)))

        # Draw ActionSegments on the main track (x offset by label_w)
        for i, seg in enumerate(timeline.action_segments):
            x = seg.start_time * self._scale_px_per_sec + label_w
            w = self._duration_to_width(seg.duration)
            selected = (i == self._selected_index)
            self._draw_action_segment(seg, i, x, action_y, w, selected)

        # Motor sub-tracks
        motor_y = action_y + self._track_h + 2
        motor_track_colors = {
            t: UNITREE_MOTOR_TRACK_MAP[t].color if t in UNITREE_MOTOR_TRACK_MAP else "#5f7fbf"
            for t in self._motor_track_names
        }
        # motor_track_labels already computed above for label-width calculation

        for tname in self._motor_track_names:
            expanded = self._motor_track_expanded.get(tname, True)
            row_h = self._motor_track_h if expanded else self._motor_track_collapsed_h
            self._motor_track_rows[tname] = (motor_y, float(row_h), expanded)
            track_color = motor_track_colors.get(tname, "#5f7fbf")
            track_label = motor_track_labels.get(tname, tname)

            # Label lane background (click-to-toggle control).
            label_bg = self._scene.addRect(
                0,
                motor_y,
                label_w,
                row_h,
                QPen(QColor("#3a3a3a")),
                QBrush(QColor("#1a1a1a")),
            )
            label_bg.setData(1, "track_toggle")
            label_bg.setData(2, tname)
            chevron = "▼" if expanded else "▶"
            track_lbl = self._scene.addSimpleText(f"{chevron} {track_label}")
            track_lbl.setBrush(QBrush(QColor("#aaaaaa")))
            track_lbl.setPos(4, motor_y + max(0, (row_h - 14) // 2))
            track_lbl.setData(1, "track_toggle")
            track_lbl.setData(2, tname)

            # Content region background (offset by label_w)
            bg_color = QColor(track_color)
            bg_color.setAlpha(40)
            self._scene.addRect(
                label_w,
                motor_y,
                self._scene_width - label_w,
                row_h,
                QPen(QColor("#3f3f3f")),
                QBrush(bg_color),
            )

            if not expanded:
                # Collapsed rows remain visible: dark mask + no interactive motor nodes.
                self._scene.addRect(
                    label_w,
                    motor_y,
                    self._scene_width - label_w,
                    row_h,
                    QPen(Qt.PenStyle.NoPen),
                    QBrush(QColor(8, 8, 8, 180)),
                )
                motor_y += row_h + 2
                continue

            # Render motor segments for this track (x offset by label_w)
            motor_segs = timeline.get_motor_segments_for_track(tname)
            total_dur = timeline.total_duration() if not timeline.is_empty() else 0.0

            # Collect covered ranges for empty-zone hatch
            covered_ranges = []
            for mseg in motor_segs:
                mx = mseg.start_time * self._scale_px_per_sec + label_w
                mw = max(float(row_h), mseg.duration * self._scale_px_per_sec)
                self._draw_motor_segment(mseg, mx, motor_y, mw, row_h, track_color, tname)
                covered_ranges.append((mx - label_w, mx - label_w + mw))

            # Diagonal hatch for uncovered zones within total_duration
            if total_dur > 0:
                total_px = total_dur * self._scale_px_per_sec
                covered_ranges.sort()
                gaps: List[Tuple[float, float]] = []
                cursor_px = 0.0
                for (seg_start, seg_end) in covered_ranges:
                    if seg_start > cursor_px:
                        gaps.append((cursor_px, seg_start))
                    cursor_px = max(cursor_px, seg_end)
                if cursor_px < total_px:
                    gaps.append((cursor_px, total_px))
                # Empty zones: light-gray base + diagonal hatch overlay so
                # uncovered ranges remain visible without overpowering segments.
                base_brush = QBrush(QColor("#d9d9d9"))
                base_pen = QPen(QColor("#bcbcbc"), 0.6)
                hatch_brush = QBrush(QColor("#a7a7a7"), Qt.BrushStyle.BDiagPattern)
                for (gap_start, gap_end) in gaps:
                    gx = gap_start + label_w
                    gw = gap_end - gap_start
                    if gw > 0:
                        self._scene.addRect(
                            gx, motor_y + 2, gw, row_h - 4,
                            base_pen, base_brush,
                        )
                        self._scene.addRect(
                            gx, motor_y + 2, gw, row_h - 4,
                            QPen(Qt.PenStyle.NoPen), hatch_brush,
                        )

            motor_y += row_h + 2

    def _motor_track_name_at_y(self, scene_y: float, only_collapsed: bool = False) -> Optional[str]:
        for tname, (top, height, expanded) in self._motor_track_rows.items():
            if not (top <= scene_y <= (top + height)):
                continue
            if only_collapsed and expanded:
                continue
            return tname
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
            # For package-expanded labels like "Lift Right Leg / Prepare",
            # keep only the actionable phase text on the axis.
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
    ) -> None:
        fill = QColor(track_color)
        fill.setAlpha(160)
        selected = (
            self._selected_motor_id == seg.motor_id
            and self._selected_motor_track == (track_name or seg.track_name)
        )
        pen = QPen(QColor("#f4d03f"), 2) if selected else QPen(QColor(track_color), 1)
        rect = self._scene.addRect(x, y + 2, width, height - 4, pen, QBrush(fill))
        # Fix 2: tag segment for interaction
        rect.setData(0, seg.motor_id)
        rect.setData(1, "motor")
        rect.setData(2, track_name or seg.track_name)
        display_name = self._motor_segment_display_name(seg)
        tooltip = f"{display_name} ({seg.motor_id})\nduration={seg.duration:.2f}s"
        rect.setToolTip(tooltip)
        if width > 20:
            title_txt = self._scene.addSimpleText(display_name)
            title_txt.setBrush(QBrush(QColor("#dddddd")))
            title_txt.setPos(x + 4, y + 2)
            title_txt.setToolTip(tooltip)
        if width > 26:
            dur_txt = self._scene.addSimpleText(f"{seg.duration:.2f}s")
            dur_txt.setBrush(QBrush(QColor("#ffffff")))
            dur_txt.setPos(x + 4, y + 16)
            dur_txt.setToolTip(tooltip)

    def _clear_motor_selection(self, notify: bool = False) -> None:
        had_sel = self._selected_motor_id is not None or self._selected_motor_track is not None
        self._selected_motor_id = None
        self._selected_motor_track = None
        if notify and had_sel:
            self.motor_segment_selected.emit("", "")

    def _motor_segment_display_name(self, seg: "MotorSegment") -> str:
        """Resolve a readable label for a motor segment."""
        tl = self._behavior_timeline
        if tl is not None and seg.parent_action_id:
            for action in tl.action_segments:
                if action.action_id == seg.parent_action_id:
                    return self._action_display_name(action.name)
        track_def = UNITREE_MOTOR_TRACK_MAP.get(seg.track_name)
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
        "walk(speed=0.3, duration=4.0)\n"
        "wait(duration=1.0)\n"
        "sit()"
    )
    ref = str(behavior_ref or "").strip()
    if not ref or ref.startswith("<"):
        return legacy_default
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ref):
        return legacy_default
    if ref == "walk":
        return "walk(speed=0.3, duration=4.0)"
    if ref == "wait":
        return "wait(duration=1.0)"
    return f"{ref}()"


# =============================================================================
# Heartbeat Canvas — data model (pure Python, no Qt dependency)
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
    HBNodeAvailability.UNKNOWN_CAPABILITY: "Capability unknown — load a capability profile to resolve",
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
    """One row in the Event State panel — readable by non-technical users."""
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
    """Canonical heartbeat draft payload — one per mission node context.

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
# Heartbeat Canvas — Qt graphics items
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


class BehaviorPanel(QWidget):
    """Behavior tab with timeline-focused sequence authoring."""

    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node_name = ""
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
        self._behavior_registry: List[str] = []
        self._modules: List[SequenceModule] = []
        self._selected_module_index: int = -1
        self._selected_motor_segment_id: Optional[str] = None
        self._selected_motor_track: Optional[str] = None
        self._movement_track_groups_expanded: Dict[str, bool] = {}
        self._movement_group_header_rows: Dict[int, str] = {}

        # Phase 1 redesign: structured timeline per node (None until first sync)
        # _behavior_timeline  — active structured model for the current node
        # _timelines_by_node  — persisted per-node timelines (key format same as _drafts_by_node)
        self._behavior_timeline: Optional[BehaviorTimeline] = None
        self._timelines_by_node: Dict[str, Dict[str, Any]] = {}

        # Fix 1: robot type and simulation mode (updated by set_capability_profile /
        # set_simulation_mode from bin/ui.py scenario settings wiring).
        self._robot_type: str = "go2"
        self._is_simulation: bool = False

        # Runtime/compile snapshot state — separate from draft
        self._hb_compiled_snapshot: Optional[Dict[str, Any]] = None
        # Event state and IO mapping panels — populated by external callers or draft restore
        self._hb_event_states: List[HBEventStateEntry] = []
        self._hb_io_mappings: List[HBIOMapping] = []
        # Dual-mode authoring state (Step 1.2)
        self._hb_authoring_mode: str = "navigator"  # "navigator" | "script" | "canvas"(legacy)
        # Productized panel no longer depends on the legacy internal canvas widget.
        # Keep a dict slot only so old draft payloads with "canvas_graph" round-trip.
        self._hb_legacy_canvas_graph: Dict[str, Any] = HBGraphData.default_template().to_dict()
        self._hb_policy_draft: HBPolicyDraft = HBPolicyDraft.default()
        # Execution chain channel — mock by default; swapped via set_hb_channel() (Step 1.3)
        self._hb_channel: IHBChannel = HBMockChannel()
        # Step 1.4: Node catalog — model-aware availability; starts as unknown
        self._hb_catalog: HBNodeCatalog = HBNodeCatalog.unknown()

        self._init_ui()
        # Diagnostics poll timer — started on first set_node_context() call (Step 1.3)
        self._hb_diag_timer = QTimer(self)
        self._hb_diag_timer.setInterval(2000)
        self._hb_diag_timer.timeout.connect(self._poll_hb_diagnostics)
        self._refresh_registries()
        self._refresh_module_library()
        self._refresh_hb_library()
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._refresh_event_state()
        self._refresh_movement_io_mapping()
        self.refresh_texts()

    def set_node_context(self, node_name: str, node_id: int) -> None:
        # Save current node's full canonical draft before switching
        if self._node_id >= 0:
            prev_key = self._draft_slot_key(self._node_id, self._node_name)
            self._drafts_by_node[prev_key] = {
                "core": self._core_editor.toPlainText(),
                "hb": self._collect_hb_draft().to_dict(),
            }
            # Save structured timeline for the outgoing node
            if self._behavior_timeline is not None:
                self._timelines_by_node[prev_key] = self._behavior_timeline.to_dict()

        self._node_name = node_name
        self._node_id = node_id
        self._refresh_breadcrumb()
        self._ctx_node.setText(node_name)
        self._ctx_node_id.setText(str(node_id))
        self._ctx_ref.setText(node_name)

        slot_key = self._draft_slot_key(node_id, node_name)
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
        core = saved.get("core", _default_core_source_for_behavior_ref(node_name))
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
        self._hb_status_label.setText("Draft — uncompiled")
        self._hb_compiled_snapshot = None
        self._hb_diag_indicator.setText("")
        # _hb_event_states / _hb_io_mappings / mode already reset by _apply_hb_draft

        # Restore structured timeline for this node (Phase 1)
        timeline_dict = self._timelines_by_node.get(slot_key)
        if timeline_dict is not None:
            try:
                self._behavior_timeline = BehaviorTimeline.from_dict(timeline_dict)
            except Exception:
                self._behavior_timeline = None
        else:
            self._behavior_timeline = None

        self._sync_modules_from_source()
        self._refresh_registries()
        self._refresh_module_library()

        # Record init snapshot for Reset — captures state right after load.
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

        # ── Context region ────────────────────────────────────────────────
        self._context_title = _plain_title(layout, "")
        self._ctx_node_key, self._ctx_node = _kv_row(layout, "", "-")
        self._ctx_node_id_key, self._ctx_node_id = _kv_row(layout, "", "-")
        self._ctx_ref_key, self._ctx_ref = _kv_row(layout, "", "-")
        self._ctx_artifact_key, self._ctx_artifact = _kv_row(layout, "", "-")
        self._ctx_compile_time_key, self._ctx_compile_time = _kv_row(layout, "", "-")
        self._ctx_diag_key, self._ctx_diag = _kv_row(layout, "", "-")

        layout.addSpacing(12)

        # ── Core Library region (movements + behavior refs) ───────────────
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

        # ── HeartBeat Library region — hidden in product UI (not part of main flow) ──
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

        # Compatibility alert bar — Step 1.7 (hidden until issues detected)
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
        root.setSpacing(6)

        heartbeat = self._build_heartbeat_section()
        timeline = self._build_timeline_section()
        root.addWidget(heartbeat, 3)   # Navigator — smaller stretch after cleanup
        root.addWidget(timeline, 0)

        # Bottom region: only Movement IO is shown in product UI.
        # Event State (tab 1) and Source/Diagnostics (tab 2) are built for compat
        # but hidden; tab bar is hidden since only one tab remains.
        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.addTab(self._build_movement_io_tab(), "")   # tab 0
        self._bottom_tabs.addTab(self._build_event_state_tab(), "")   # tab 1 (hidden)
        self._bottom_tabs.addTab(self._build_source_diag_section(), "")  # tab 2 (hidden)
        self._bottom_tabs.setTabVisible(1, False)
        self._bottom_tabs.setTabVisible(2, False)
        self._bottom_tabs.tabBar().hide()  # Single visible tab — no bar needed
        root.addWidget(self._bottom_tabs, 2)
        return wrapper

    def _build_movement_io_tab(self) -> QWidget:
        """Tab 0: movement settings (left) + IO signal mapping (right), side by side."""
        pane = QWidget()
        layout = QHBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        movement_settings = self._build_movement_settings_section()
        io_mapping = self._build_io_mapping_section()
        movement_settings.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        io_mapping.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Keep both panes with equal weight.
        layout.addWidget(movement_settings, 1)
        layout.addWidget(io_mapping, 1)
        return pane

    def _build_event_state_tab(self) -> QFrame:
        """Tab 1: event state panel — event name, state, source, timestamp, reason."""
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

        # Conflict / warning hint label — hidden until conflicts exist (Step 1.5)
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
        self._hb_nav_btn = QPushButton("Navigator")  # Step 2
        self._hb_nav_btn.setCheckable(True)
        self._hb_nav_btn.setFixedHeight(28)

        self._compile_btn = QPushButton("")
        self._compile_btn.setObjectName("behaviorSaveBtn")
        self._compile_btn.setFixedHeight(28)
        self._compile_btn.setEnabled(False)
        self._compile_btn.clicked.connect(self._run_compile)

        self._save_as_btn = QPushButton("")
        self._save_as_btn.setObjectName("behaviorSaveAsBtn")
        self._save_as_btn.setFixedHeight(28)
        self._save_as_btn.clicked.connect(self._on_save_as)

        self._reset_btn = QPushButton("")
        self._reset_btn.setObjectName("behaviorResetBtn")
        self._reset_btn.setFixedHeight(28)
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._on_reset)

        # Legacy canvas and compiler toggles hidden in product UI; only Navigator shown.
        row.addWidget(self._hb_canvas_btn)
        row.addWidget(self._hb_compiler_btn)
        row.addWidget(self._hb_nav_btn)
        row.addStretch()
        row.addWidget(self._compile_btn)
        row.addWidget(self._save_as_btn)
        row.addWidget(self._reset_btn)
        layout.addLayout(row)
        self._hb_canvas_btn.hide()
        self._hb_compiler_btn.hide()  # Script mode retired from product UI

        # Execution chain buttons, status label, and policy row are retained as attributes
        # for backend compat (compile/dryrun/run logic still works internally) but are not
        # shown in the product UI — Behavior panel is now "Navigator + Timeline + Settings".
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

        self._hb_status_label = QLabel("Draft — uncompiled")
        self._hb_status_label.setStyleSheet(
            "color: #9ca3af; font-size: 11px; background: transparent;"
        )
        _hidden_layout.addWidget(self._hb_status_label)

        # Policy row — tick_ms, timeout_ms, fail_policy (retained for persistence compat)
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

        # Step 2: Motor Weight Navigator — read-only inspection mode (index 2)
        self._motor_weight_navigator = MotorWeightNavigator()
        self._hb_stack.addWidget(self._motor_weight_navigator)

        layout.addWidget(self._hb_stack, 1)

        self._hb_canvas_btn.clicked.connect(lambda: self._set_hb_mode(0))
        self._hb_compiler_btn.clicked.connect(lambda: self._set_hb_mode(1))
        self._hb_nav_btn.clicked.connect(lambda: self._set_hb_mode(2))  # Step 2
        # Product default: quick navigator replaces legacy placeholder canvas.
        self._set_hb_mode(2)
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

        self._timeline = TimelineView()
        self._timeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # self._timeline.setMinimumHeight(300)
        self._timeline.module_selected.connect(self._on_timeline_module_selected)
        self._timeline.motor_segment_selected.connect(self._on_timeline_motor_segment_selected)
        self._timeline.module_reordered.connect(self._on_timeline_module_reordered)
        self._timeline.delete_requested.connect(self._remove_selected_module)
        self._timeline.timeline_edited.connect(self._on_timeline_edited)
        layout.addWidget(self._timeline, 1)

        return pane

    def _build_movement_settings_section(self) -> QFrame:
        pane = QFrame()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._movement_settings_title = _plain_title(layout, "")
        self._movement_selected_label = QLabel("")
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
        self._hb_nav_btn.setChecked(idx == 2)          # Step 2
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

        try:
            from bin.core.robot_context import RobotContext
            self._movement_registry = sorted(set(RobotContext.get_available_actions() or []))
        except Exception:
            self._movement_registry = []

        if not self._movement_registry:
            self._movement_registry = ["stand", "walk", "sit", "wait", "stop"]

    def _refresh_module_library(self) -> None:
        self._module_library_tree.clear()

        movement_root = QTreeWidgetItem([tr("behavior.library.movement", "System")])
        movement_root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        movement_root.setExpanded(True)
        self._module_library_tree.addTopLevelItem(movement_root)
        for name in self._movement_registry:
            item = QTreeWidgetItem([f"movement.{name}()"])
            item.setData(0, Qt.ItemDataRole.UserRole, {"kind": "movement", "name": name, "args": ""})
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

        All catalog entries are always shown — nodes are never silently hidden.
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

            # Build tooltip — always explicit, never silent
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

            # Col 2–4: source, timestamp, reason (plain)
            self._event_state_table.setItem(row, 2, QTableWidgetItem(entry.source))
            self._event_state_table.setItem(row, 3, QTableWidgetItem(entry.timestamp))
            self._event_state_table.setItem(row, 4, QTableWidgetItem(entry.reason))

    def _refresh_navigator(self) -> None:
        """Step 2: Rebuild MotorWeightNavigator from current timeline + IO mappings."""
        try:
            from system.behavior.action_profile import get_motor_track_map
            robot_type = getattr(self._behavior_timeline, "robot_type", "") or ""
            available_tracks = get_motor_track_map(robot_type)
            self._motor_weight_navigator.refresh_from_timeline_and_io(
                self._behavior_timeline,
                self._hb_io_mappings,
                available_tracks=available_tracks,
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

        # Conflict detection — update hint label
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

            # Col 2: direction badge (colored foreground, ↓ In / ↑ Out)
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
            cur_key = self._draft_slot_key(self._node_id, self._node_name)
            entry: Dict[str, Any] = {
                "core": self._core_editor.toPlainText(),
                "hb": self._collect_hb_draft().to_dict(),
            }
            if self._behavior_timeline is not None:
                entry["timeline"] = self._behavior_timeline.to_dict()
            # Fix 6: persist expand/collapse state
            entry["motor_track_expanded"] = dict(self._timeline._motor_track_expanded)
            self._drafts_by_node[cur_key] = entry
            # Also flush timeline to the timelines registry
            if self._behavior_timeline is not None:
                self._timelines_by_node[cur_key] = self._behavior_timeline.to_dict()
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
            # Extract timeline sub-key if present
            timeline_raw = v.get("timeline")
            if isinstance(timeline_raw, dict):
                self._timelines_by_node[key] = dict(timeline_raw)

        # Re-apply the current node's draft if the panel already has a node context
        if self._node_id >= 0:
            cur_key = self._draft_slot_key(self._node_id, self._node_name)
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
            # Restore timeline (Phase 1)
            timeline_raw = saved.get("timeline") or self._timelines_by_node.get(cur_key)
            if isinstance(timeline_raw, dict):
                try:
                    self._behavior_timeline = BehaviorTimeline.from_dict(timeline_raw)
                except Exception:  # noqa: BLE001
                    self._behavior_timeline = None
            else:
                self._behavior_timeline = None
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
        if self._node_id >= 0 and self._behavior_timeline is not None:
            cur_key = self._draft_slot_key(self._node_id, self._node_name)
            self._timelines_by_node[cur_key] = self._behavior_timeline.to_dict()
        return {str(k): dict(v) for k, v in self._timelines_by_node.items()}

    def set_behavior_timelines_state(self, timelines: Dict[str, Any]) -> None:
        """Restore per-node BehaviorTimeline dicts from a mission file.

        Silently skips malformed entries.
        """
        self._timelines_by_node = {}
        for k, v in timelines.items():
            if not isinstance(v, dict):
                continue
            key = str(k).strip()
            if not key:
                continue
            self._timelines_by_node[key] = dict(v)
        # Apply to current node if context is already set
        if self._node_id >= 0:
            cur_key = self._draft_slot_key(self._node_id, self._node_name)
            tl_raw = self._timelines_by_node.get(cur_key)
            if tl_raw is not None:
                try:
                    self._behavior_timeline = BehaviorTimeline.from_dict(tl_raw)
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
        if isinstance(cap, dict):
            self._robot_type = cap.get("robot_type", "go2") or "go2"
        else:
            self._robot_type = "go2"
        self._hb_catalog = HBNodeCatalog.from_capability_dict(cap)
        self._refresh_hb_library()
        # Rebuild timeline with updated robot_type (Fix 5)
        if self._modules:
            self._behavior_timeline = self._build_or_update_timeline(self._modules)
            self._refresh_timeline()

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
        report = audit_catalog_compatibility(self._hb_catalog, brand, robot_type)
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
            f"\u26a0 {report.summary_text()} — click item to highlight node"
        )
        self._compat_summary_label.show()

        # Issues list
        self._compat_issues_list.clear()
        for issue in report.issues:
            prefix = "[ERR]" if issue.severity == "error" else "[WARN]"
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
        """Click on a compat issue → highlight matching node in HB library tree."""
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
    # Step 1.2 — Canonical draft model and dual-mode authoring contract
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

    # ------------------------------------------------------------------
    # Step 1.3 — Execution chain handlers
    # ------------------------------------------------------------------

    def _on_hb_compile(self) -> None:
        """HB Compile button: compile the current heartbeat draft via the channel."""
        if not self._node_name:
            self._output.append("[HB Compile] No behavior node selected.")
            return
        draft = self._collect_hb_draft()
        req = HBCompileRequest(behavior_ref=self._node_name, draft_payload=draft)
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
                f"HB compiled OK — artifact {resp.artifact_id[:12]}... "
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
                f"HB compile failed — {resp.error_count} error(s)"
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
            behavior_ref=self._node_name,
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
                f"Simulated {resp.simulated_ticks} tick(s) — "
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
        req = HBRunRequest(behavior_ref=self._node_name, draft_payload=draft)
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
                f"HB {resp.status} — heartbeat: {resp.heartbeat_status}"
            )
            if not self._hb_diag_timer.isActive():
                self._hb_diag_timer.start()

    def _poll_hb_diagnostics(self) -> None:
        """Timer callback: poll channel diagnostics and refresh status display."""
        if not self._node_name or self._node_id < 0:
            return
        try:
            snap: HBDiagnosticsSnapshot = self._hb_channel.poll_diagnostics(
                self._node_name
            )
        except Exception:
            return
        # Update indicator label with live status
        if snap.status == "running":
            self._hb_diag_indicator.setText(
                f"HB active — tick #{snap.tick_count}"
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
        module = SequenceModule(kind=kind, name=name, args="duration=1.0", duration=1.0)
        self._modules.append(module)
        self._selected_module_index = len(self._modules) - 1
        self._sync_source_from_modules()
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _remove_selected_module(self) -> None:
        if self._selected_module_index < 0 or self._selected_module_index >= len(self._modules):
            return
        self._modules.pop(self._selected_module_index)
        if self._selected_module_index >= len(self._modules):
            self._selected_module_index = len(self._modules) - 1
        self._sync_source_from_modules()
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _move_selected_module(self, delta: int) -> None:
        i = self._selected_module_index
        if i < 0 or i >= len(self._modules):
            return
        j = i + delta
        if j < 0 or j >= len(self._modules):
            return
        self._modules[i], self._modules[j] = self._modules[j], self._modules[i]
        self._selected_module_index = j
        self._sync_source_from_modules()
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _on_timeline_module_selected(self, index: int) -> None:
        self._selected_module_index = index
        self._selected_motor_segment_id = None
        self._selected_motor_track = None
        self._refresh_timeline()
        self._refresh_movement_settings()

    def _on_timeline_motor_segment_selected(self, motor_id: str, track_name: str) -> None:
        if motor_id and track_name:
            self._selected_motor_segment_id = motor_id
            self._selected_motor_track = track_name
            self._selected_module_index = -1
        else:
            self._selected_motor_segment_id = None
            self._selected_motor_track = None
        self._refresh_timeline()
        self._refresh_movement_settings()

    def _on_timeline_module_reordered(self, from_index: int, to_index: int) -> None:
        if from_index < 0 or to_index < 0:
            return
        if from_index >= len(self._modules) or to_index >= len(self._modules):
            return
        if from_index == to_index:
            return
        module = self._modules.pop(from_index)
        self._modules.insert(to_index, module)
        self._selected_module_index = to_index
        self._sync_source_from_modules()
        self._refresh_timeline()
        self._refresh_movement_settings()
        self._mark_dirty()

    def _on_timeline_edited(self) -> None:
        """Called when the timeline emits timeline_edited (Fix 6)."""
        self._mark_dirty()
        self._refresh_movement_settings()
        self._refresh_navigator()  # Step 2

    def _refresh_timeline(self) -> None:
        # Phase 1: use multi-track timeline when structured model is available
        if self._behavior_timeline is not None and not self._behavior_timeline.is_empty():
            self._timeline.set_multi_track_timeline(
                timeline=self._behavior_timeline,
                selected_index=self._selected_module_index,
            )
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

    # ── Step 3: IO mutation helpers for source-switch ────────────────────────

    def _io_upsert_external(self, track_name: str, param_key: str, signal: str) -> None:
        """Add or update an inbound HBIOMapping for track_name.param_key → signal."""
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

        # ── Column 1: Source indicator ───────────────────────────────────────
        if is_structural:
            # Structural params always Constant — show a read-only label
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

        # ── Column 2: Value editor (depends on source) ───────────────────────
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
            # Constant path — reuse high-precision editor
            num_val = self._try_float(value)
            if num_val is None:
                editor = QLineEdit(str(value) if value is not None else "")
                editor.editingFinished.connect(
                    lambda e=editor, k=key: on_commit_constant(k, e.text())
                )
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(6)
                editor.setSingleStep(0.0005)
                editor.setRange(-1_000_000.0, 1_000_000.0)
                editor.setValue(num_val)
                editor.editingFinished.connect(
                    lambda e=editor, k=key: on_commit_constant(k, e.value())
                )
            self._movement_params_table.setCellWidget(row, 2, editor)

    # ── Step 3 end ───────────────────────────────────────────────────────────

    def _refresh_movement_settings(self) -> None:
        self._movement_params_table.setRowCount(0)
        self._movement_params_table.clearContents()
        self._movement_params_table.clearSpans()
        self._movement_group_header_rows = {}
        # Priority: when a motor sub-track segment is selected, edit it directly.
        seg = self._get_selected_motor_segment()
        if seg is not None:
            self._movement_selected_label.setText(
                f"Selected Node: motor.{seg.track_name}.{seg.motor_id[:8]}"
            )
            self._movement_hint.setText(
                "Fine-tune selected motor segment parameters. Changes apply immediately."
            )
            rows: List[Tuple[str, Any]] = [("start_time", seg.start_time), ("duration", seg.duration)]
            track_def = UNITREE_MOTOR_TRACK_MAP.get(seg.track_name)
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

        # Multi-track mode: timeline widget emits indices into action_segments, NOT
        # panel._modules (which tracks the high-level module sequence and has fewer
        # entries when a package action expands into multiple phases).
        if self._behavior_timeline is not None and not self._behavior_timeline.is_empty():
            n_actions = len(self._behavior_timeline.action_segments)
            if 0 <= self._selected_module_index < n_actions:
                action_seg = self._behavior_timeline.action_segments[self._selected_module_index]
                self._movement_selected_label.setText(
                    tr("behavior.movement.selected_format", "Selected Node: {node}").format(
                        node=action_seg.name
                    )
                )
                if self._refresh_movement_settings_for_action_overlay(self._selected_module_index):
                    return
                self._movement_hint.setText(tr(
                    "behavior.movement.no_motor_params",
                    "No motor parameters for this action.",
                ))
            else:
                self._movement_selected_label.setText(tr("behavior.movement.selected_none", "Selected Node: None"))
                self._movement_hint.setText(tr(
                    "behavior.movement.select_hint",
                    "Select a movement module in the main track to inspect its parameters.",
                ))
            return

        # Legacy flat-module path (no structured timeline).
        if self._selected_module_index < 0 or self._selected_module_index >= len(self._modules):
            self._movement_selected_label.setText(tr("behavior.movement.selected_none", "Selected Node: None"))
            self._movement_hint.setText(tr(
                "behavior.movement.select_hint",
                "Select a movement module in the main track to inspect its parameters.",
            ))
            return

        module = self._modules[self._selected_module_index]
        self._movement_selected_label.setText(
            tr("behavior.movement.selected_format", "Selected Node: {node}").format(
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
        parsed = self._parse_args(module.args)
        rows: List[Tuple[str, Any]] = [("duration", module.duration)]
        for (k, v) in parsed:
            if k == "duration":
                continue
            rows.append((k, v))

        self._movement_params_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self._movement_params_table.setItem(r, 0, QTableWidgetItem(k))
            # Step 3: action-module params have no track context → always constant, col 1 = label, col 2 = editor
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
                editor.setDecimals(6)
                editor.setSingleStep(0.0005)
                editor.setRange(-1_000_000.0, 1_000_000.0)
                editor.setValue(num_val)
                editor.editingFinished.connect(
                    lambda e=editor, key_name=k, idx=self._selected_module_index: self._commit_module_param(idx, key_name, e.value())
                )
            self._movement_params_table.setCellWidget(r, 2, editor)

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
            track_label = UNITREE_MOTOR_TRACK_MAP.get(track_name).label if track_name in UNITREE_MOTOR_TRACK_MAP else track_name
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
                track_def = UNITREE_MOTOR_TRACK_MAP.get(seg.track_name)
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
        if not self._selected_motor_segment_id or not self._selected_motor_track:
            return None
        for overlay in self._behavior_timeline.motor_overlays:
            for seg in overlay.motor_segments:
                if (
                    seg.motor_id == self._selected_motor_segment_id
                    and seg.track_name == self._selected_motor_track
                ):
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
            editor.setDecimals(6)
            editor.setSingleStep(0.0005)
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
                new_val = max(0.005, v)
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
        self._refresh_timeline()
        self._mark_dirty()

    def _commit_module_param(self, index: int, key: str, value: Any) -> None:
        if index < 0 or index >= len(self._modules):
            return
        module = self._modules[index]
        old_args = module.args
        old_duration = module.duration
        pairs = [(k, v) for (k, v) in self._parse_args(module.args) if k]

        def _upsert(p_key: str, p_val: str) -> None:
            for i, (k, _v) in enumerate(pairs):
                if k == p_key:
                    pairs[i] = (k, p_val)
                    break
            else:
                pairs.append((p_key, p_val))

        if key == "duration":
            f = self._try_float(value)
            if f is not None:
                module.duration = max(0.005, f)
                _upsert("duration", f"{module.duration:.6f}".rstrip("0").rstrip("."))
        else:
            f = self._try_float(value)
            val_txt = (
                f"{f:.6f}".rstrip("0").rstrip(".")
                if f is not None
                else str(value)
            )
            _upsert(key, val_txt)
            if key == "duration":
                module.duration = max(0.005, f or module.duration)

        module.args = ", ".join([f"{k}={v}" if v != "" else k for (k, v) in pairs])
        if module.args == old_args and abs(float(module.duration) - float(old_duration)) <= 1e-9:
            return
        self._sync_source_from_modules()
        self._refresh_timeline()
        self._mark_dirty()

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
        self._modules = parsed
        self._selected_module_index = -1
        # Phase 1: build/rebuild structured timeline from parsed modules.
        # If a saved timeline exists (restored from draft/file), only rebuild
        # when the source has changed (different action names or count).
        self._behavior_timeline = self._build_or_update_timeline(parsed)
        self._refresh_timeline()
        self._refresh_movement_settings()
        # Keep navigator in sync with the latest timeline build, especially
        # during first open where _apply_hb_draft() may have refreshed before
        # timeline restoration/rebuild completed.
        self._refresh_navigator()

    def _build_or_update_timeline(
        self, modules: List[SequenceModule]
    ) -> Optional[BehaviorTimeline]:
        """Build a BehaviorTimeline from the current SequenceModule list.

        If a saved timeline already has the same action sequence (names +
        order), preserve its motor overlay data.  Otherwise, build fresh
        from Unitree profiles.
        """
        if not modules:
            return None
        robot_type = self._robot_type or ""
        new_timeline = build_timeline_from_modules(
            modules, robot_type=robot_type, auto_decompose=(robot_type != "")
        )
        if self._behavior_timeline is not None:
            # Try to preserve existing motor overlays if the action list matches
            existing = self._behavior_timeline
            same = (
                len(existing.action_segments) == len(new_timeline.action_segments)
                and all(
                    a.name == b.name
                    for a, b in zip(existing.action_segments, new_timeline.action_segments)
                )
            )
            if same:
                # Rebind preserved overlays to the newly generated action_ids so
                # action-click lookup stays correct after source sync/rebuild.
                rebound_overlays: List[ActionMotorOverlay] = []
                for idx, new_action in enumerate(new_timeline.action_segments):
                    if idx >= len(existing.motor_overlays):
                        break
                    old_overlay = existing.motor_overlays[idx]
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
                new_timeline.motor_overlays = rebound_overlays
                new_timeline.active_motor_tracks = existing.active_motor_tracks
        return new_timeline

    def _sync_source_from_modules(self) -> None:
        lines = []
        for m in self._modules:
            args = m.args.strip()
            lines.append(f"{m.name}({args})" if args else f"{m.name}()")
        self._set_core_source("\n".join(lines), mark_dirty=True)

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
        # Restore structured timeline
        tl_dict = snap.get("timeline")
        if tl_dict is not None:
            try:
                self._behavior_timeline = BehaviorTimeline.from_dict(tl_dict)
            except Exception:
                self._behavior_timeline = None
        else:
            self._behavior_timeline = None
        # Clear dirty state for this node
        if self._node_id >= 0:
            self._dirty_nodes.discard(self._node_id)
        self._sync_modules_from_source()
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
        self._sync_source_from_modules()
        source = self._core_editor.toPlainText()
        self._output.append(f"[Save As] behavior_ref='{new_ref}' source_len={len(source)} chars")
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

        self._sync_source_from_modules()
        behavior_ref = self._node_name
        source = self._core_editor.toPlainText()
        self._output.append(f"[Compiling] behavior_ref='{behavior_ref}' source_len={len(source)} chars")

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
                f"Core compiled OK — 0 errors, {artifact.warning_count} warning(s)"
            )
        else:
            self._hb_status_label.setText(
                f"Core compiled — {artifact.error_count} error(s), {artifact.warning_count} warning(s)"
            )
        # Store compiled snapshot (separate from transient draft state)
        self._hb_compiled_snapshot = artifact.to_dict()

        self._refresh_registries()
        self._refresh_module_library()

        if artifact.is_valid and self._node_id >= 0:
            self._dirty_nodes.discard(self._node_id)

    def _on_compile_finished(self) -> None:
        self._compile_btn.setText(tr("behavior.header.save", "Save"))
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
        self._apply_save_button_style()

    def _apply_save_button_style(self) -> None:
        bg = get_color("behavior_save_button_bg", get_color("button_bg", "#111827"))
        text = get_color("behavior_save_button_text", get_color("button_text", "#e5e7eb"))
        border = get_color("behavior_save_button_border", get_color("button_border", "#4b5563"))
        disabled_bg = get_color("behavior_save_button_disabled_bg", "#3a3a3a")
        disabled_text = get_color("behavior_save_button_disabled_text", "#8a8a8a")
        disabled_border = get_color("behavior_save_button_disabled_border", "#565656")
        hover_bg = get_color("behavior_save_button_hover_bg", bg)
        self._compile_btn.setStyleSheet(
            f"""
            QPushButton#behaviorSaveBtn {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 0 12px;
                text-align: left;
            }}
            QPushButton#behaviorSaveBtn:hover:!disabled {{
                background: {hover_bg};
            }}
            QPushButton#behaviorSaveBtn:disabled {{
                background: {disabled_bg};
                color: {disabled_text};
                border: 1px solid {disabled_border};
            }}
            """
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
        self._compile_btn.setText(tr("behavior.header.save", "Save"))
        self._save_as_btn.setText(tr("behavior.header.save_as", "Save as"))
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
        self._hb_nav_btn.setText(tr("behavior.heartbeat.navigator", "Navigator"))
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
        if not hasattr(self, "_compile_btn"):
            return
        running = self._compile_worker is not None and self._compile_worker.isRunning()
        can_save = (self._node_id >= 0) and (self._node_id in self._dirty_nodes) and (not running)
        self._compile_btn.setEnabled(can_save)
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
    k = QLabel(f"{key}:")
    k.setObjectName("behaviorDetailKey")
    v = QLabel(value)
    v.setObjectName("behaviorDetailVal")
    v.setWordWrap(True)
    row.addWidget(k)
    row.addWidget(v, 1)
    layout.addLayout(row)
    return k, v
