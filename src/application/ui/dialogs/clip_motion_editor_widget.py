# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ClipMotionEditorWidget — the reusable **Clip Motion Editor** surface.

A motion file (e.g. a LAFAN1 capture) is often a long sequence containing
many un-segmented sub-motions. This widget lets the user:

* watch the clip play back in an embedded **offscreen MuJoCo render**
  (no MuJoCo UI — pure pixels into a scaled ``QLabel``; the starting frame
  is byte-identical to the clip's first frame);
* scrub / play with a video-editor-style :class:`ClipTimeline` (frame ruler
  on top, seconds ruler below, draggable playhead + in/out selection box,
  Ctrl-wheel zoom, already-saved segments coloured on the track);
* mark **segments** by dragging the in/out box or typing Start/End frames,
  then tag / crop them (persisted as user state via the assets-browser seam);
* (embedded mode) **Apply** a saved segment as the training item's reference.

Two host contexts share this one surface (§11 — one implementation, two
consumers):

* **Resources → Clip Motion Editor** (:class:`ClipCropLabelDialog`) —
  ``show_robot_picker=True``: a render-robot dropdown lives in the top row
  and the user drives everything from here. No Apply column (there is no
  training item to bind to).
* **Training Motion Editor** (:mod:`registry_module_editor_panel`) —
  ``show_robot_picker=False``: the render robot is *fixed* to the canvas's
  bound robot (passed as ``robot_sku=``), the top row is hidden (a ``Clip:``
  dropdown sits above the widget in the host), and the segment table gains
  an **Apply** button per row. Segment add/delete and Apply are surfaced to
  the host through :pyattr:`segmentsChanged` / :pyattr:`segmentApplied`.

Rendering runs on a dedicated :class:`ClipRenderWorker` ``QThread`` because
``mujoco.Renderer`` owns a thread-affine GL context (see that module). The
host MUST call :meth:`teardown` when it closes so the GL context is released.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QMetaObject, Q_ARG, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QBrush, QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    i18n_bind,
    log_warning,
    setButton,
    setLineEdit,
    setSpinBox,
    tr,
)

from application.ui.widgets.clip_timeline import ClipTimeline

#: Offscreen render resolution (displayed scaled-to-fit, aspect preserved).
_RENDER_W = 1200
_RENDER_H = 900
#: Timeline range fallback when the clip length can't be probed.
_FALLBACK_MAX_FRAME = 1000
#: Suggested tag labels (free text still allowed).
_TASK_TAG_SUGGESTIONS = ("", "walk", "run", "turn", "stand", "jump", "dance")

#: Mouse-drag → degrees of orbit per pixel, and wheel zoom step.
_ORBIT_DEG_PER_PX = 0.3
_ZOOM_IN = 0.9
_ZOOM_OUT = 1.1


class _RenderView(QLabel):
    """Render surface: left/middle drag → orbit, mouse-wheel → zoom.

    Emits incremental camera deltas; the widget forwards them to the render
    worker (which owns the GL camera) and re-renders the current frame.
    """

    orbit = pyqtSignal(float, float)  # (d_azimuth_deg, d_elevation_deg)
    zoomed = pyqtSignal(float)        # distance multiplier (<1 = closer)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._last: Optional[object] = None

    def mousePressEvent(self, e):  # type: ignore[override]
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._last = e.position()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  # type: ignore[override]
        if self._last is not None:
            p = e.position()
            dx = p.x() - self._last.x()
            dy = p.y() - self._last.y()
            self._last = p
            # drag right → azimuth+, drag up → elevation+ (MuJoCo-viewer feel)
            self.orbit.emit(dx * _ORBIT_DEG_PER_PX, -dy * _ORBIT_DEG_PER_PX)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):  # type: ignore[override]
        self._last = None
        e.accept()

    def wheelEvent(self, e):  # type: ignore[override]
        dy = e.angleDelta().y()
        if dy != 0:
            self.zoomed.emit(_ZOOM_IN if dy > 0 else _ZOOM_OUT)
            e.accept()
            return
        super().wheelEvent(e)


def _ss() -> int:
    # The Clip Motion Editor's own chrome is size_small; only the embedded
    # ClipTimeline keeps size_mini for its dense dual rulers.
    return int(Config.get_font_size("size_small"))


def _make_icon(name: str) -> QIcon:
    """Resolve a themed asset icon by bare name (e.g. ``"icon_play"``)."""
    path = Assets.find_icon(name)
    return QIcon(str(path)) if path is not None else QIcon()


class ClipMotionEditorWidget(QWidget):
    """Embedded-render clip editor surface for a single clip ref.

    Signals:
        segmentsChanged: emitted after any segment add / crop / delete /
            rename / tag edit — the host repopulates its own clip picker so
            a newly-cut segment becomes selectable immediately (req 5).
        segmentApplied(str): emitted when the user clicks a segment row's
            **Apply** button — carries the encoded ``<base>#seg=lo-hi``
            reference the host binds as the training item's clip (req 5).
    """

    segmentsChanged = pyqtSignal()
    segmentApplied = pyqtSignal(str)

    def __init__(
        self,
        *,
        provider,
        parent: Optional[QWidget] = None,
        clip_ref: Optional[str] = None,
        initial_selection: Optional[tuple] = None,
        show_robot_picker: bool = True,
        show_reference_select: bool = False,
        robot_sku: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._clip_ref = str(clip_ref) if clip_ref else ""
        # Optional (lo, hi) to pre-select on first load — used when the host
        # drills in on an existing segment. None → whole clip.
        self._initial_selection = initial_selection
        # Picker mode = standalone Resources editor (render-robot dropdown in
        # the top row); fixed mode = embedded in the Training Motion Editor
        # (robot pinned to the canvas robot, no top row).
        self._show_robot_picker = bool(show_robot_picker)
        # Reference-select checkbox column on the segment table — only
        # meaningful when a host training item exists to bind the reference to
        # (embedded mode). Checking a segment applies it as the training
        # reference; unchecking falls back to the whole clip.
        self._show_reference = bool(show_reference_select)
        # The full ref currently bound as the training reference (host-set):
        # a ``<base>#seg=lo-hi`` segment ref, or the base clip ref (whole clip),
        # or "" (none). Drives which segment row shows checked + highlighted.
        self._applied_seg_ref: str = ""
        # Fixed render robot (embedded mode). Ignored when the picker is shown.
        self._robot_sku: Optional[str] = robot_sku or None

        # Segment-table column layout (a leading checkbox column exists only in
        # reference-select mode).
        if self._show_reference:
            (self._C_CHECK, self._C_NAME, self._C_TAG,
             self._C_IN, self._C_OUT, self._C_ACT) = 0, 1, 2, 3, 4, 5
            self._n_cols = 6
        else:
            self._C_CHECK = -1
            (self._C_NAME, self._C_TAG,
             self._C_IN, self._C_OUT, self._C_ACT) = 0, 1, 2, 3, 4
            self._n_cols = 5
        # Segment rows currently shown (indexed by table row) — lets the check
        # handler / visual refresh resolve a row back to its SegmentRow without
        # re-reading cells.
        self._row_segs: List = []

        self._populating = False
        self._syncing = False
        self._robot_skus: List[str] = []

        # Render worker / thread state.
        self._thread: Optional[QThread] = None
        self._worker = None
        self._worker_ready = False
        self._range_initialised = False
        self._last_qimage: Optional[QImage] = None
        self._in_flight = False
        # Coalesced pending request: the latest wanted frame + accumulated
        # camera deltas to apply when the in-flight render returns.
        self._pending_frame: Optional[int] = None
        self._pending_d_az = 0.0
        self._pending_d_el = 0.0
        self._pending_zoom = 1.0

        # Clip length — refined by the render worker's ``ready`` (or the CPU
        # probe when no robot is bound). Defaults keep the timeline sane
        # before the first clip is loaded.
        self._n_frames = 0
        self._fps = 0.0
        self._frames_known = False

        self.setObjectName("clipMotionEditor")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.setInterval(self._frame_interval_ms())

        self._init_ui()
        # Load the initial clip (probe → timeline range → render worker). For
        # the embedded editor this is usually None on construction; the host
        # drives ``set_clip`` once the user picks one.
        self._load_clip(self._clip_ref or None, initial_selection=initial_selection)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Top row: clip name (left) + render-robot picker (right) ──
        # Shown only in picker (standalone Resources) mode. In embedded mode
        # the host owns a ``Clip:`` dropdown above this widget and the robot
        # is fixed to the canvas robot, so the whole row is suppressed.
        self._robot_combo: Optional[QComboBox] = None
        if self._show_robot_picker:
            top = QHBoxLayout()
            top.setSpacing(8)
            self._top_clip_label = QLabel(
                Path(self._clip_ref).name if self._clip_ref else "", self
            )
            self._top_clip_label.setStyleSheet(
                f"color: {Config.get_color('sub_t1')};"
                f" background: transparent; font-size: {_ss()}px;"
            )
            self._top_clip_label.setToolTip(self._clip_ref)
            top.addWidget(self._top_clip_label, 1)
            top.addStretch(1)
            top.addWidget(self._label("clip_editor.render_robot", "Render robot"))
            self._robot_combo = self._build_robot_combo()
            if self._robot_combo is not None:
                # Fixed width, stretch 0 → never grows past 200px (CLAUDE.md
                # sidebar/fixed-width discipline; combobox must not stretch).
                top.addWidget(self._robot_combo, 0)
            root.addLayout(top)
        else:
            self._top_clip_label = None

        # ── Render region: a stack of [render view | loading page] ──
        # Page 0 is the offscreen-MuJoCo render surface (drag=orbit,
        # wheel=zoom). Page 1 is a centred "Loading…" text + small progress
        # bar shown while the host pre-loads motion assets on a background
        # thread — the main UI thread is never blocked (host req 3).
        self._render_label = _RenderView(self)
        self._render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._render_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._render_label.setStyleSheet(
            f"QLabel {{ background: {Config.get_color('canvas_bg')};"
            f" color: {Config.get_color('sub_t1')};"
            f" border: 1px solid {Config.get_color('border_1')};"
            f" font-size: {_ss()}px; }}"
        )
        self._render_label.setToolTip(
            tr("clip_editor.view_hint", "Drag to rotate · scroll to zoom")
        )
        self._render_label.setText(tr("clip_editor.rendering", "Rendering…"))
        self._render_label.orbit.connect(
            lambda az, el: self._request(d_az=az, d_el=el)
        )
        self._render_label.zoomed.connect(lambda f: self._request(zoom=f))

        self._render_stack = QStackedWidget(self)
        self._render_stack.setMinimumHeight(300)
        self._render_stack.addWidget(self._render_label)         # page 0
        self._render_stack.addWidget(self._build_loading_page())  # page 1
        root.addWidget(self._render_stack, 1)

        # ── Transport row: play/pause + stop (icon buttons) + frame readout ──
        transport = QHBoxLayout()
        transport.setSpacing(6)
        self._play_icon = _make_icon("icon_play")
        self._pause_icon = _make_icon("icon_pause")
        self._play_btn = setButton(
            "clip_editor.play", 30, 28, kind="light",
            icon="icon_play", icon_only=True, default="",
        )
        self._play_btn.setToolTip(tr("clip_editor.play", "Play"))
        self._play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self._play_btn)
        self._stop_btn = setButton(
            "clip_editor.stop", 30, 28, kind="light",
            icon="icon_stop", icon_only=True, default="",
        )
        self._stop_btn.setToolTip(tr("clip_editor.stop", "Stop"))
        self._stop_btn.clicked.connect(self._on_stop)
        transport.addWidget(self._stop_btn)
        self._info_label = QLabel("", self)
        self._info_label.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        transport.addWidget(self._info_label, 1)
        self._range_label = QLabel("", self)
        self._range_label.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        transport.addWidget(self._range_label, 0)
        root.addLayout(transport)
        self._update_info_label()

        # ── Timeline ──
        self._timeline = ClipTimeline(self)
        self._timeline.set_frame_count(_FALLBACK_MAX_FRAME)
        self._timeline.set_fps(self._fps)
        self._timeline.set_known(False)
        self._timeline.seeked.connect(self._on_seek)
        self._timeline.playheadMoved.connect(self._on_playhead_moved)
        self._timeline.rangeChanged.connect(self._on_range_changed)
        root.addWidget(self._timeline)

        # ── Marker form: Start / End / Frame … [Save as Clip] ──
        form = QHBoxLayout()
        form.setSpacing(6)
        form.addWidget(self._label("clip_editor.seg_start", "Start"))
        self._start_spin = setSpinBox(
            0, minimum=0, maximum=_FALLBACK_MAX_FRAME, parent=self
        )
        # Hide the up/down arrows — they render as bare white rectangles here
        # (the SDK spin box has no themed step buttons). Typing / timeline drag
        # drive the value instead.
        self._start_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._apply_small_font(self._start_spin)
        self._start_spin.valueChanged.connect(self._on_spin_changed)
        form.addWidget(self._start_spin)
        form.addWidget(self._label("clip_editor.seg_end", "End"))
        self._end_spin = setSpinBox(
            _FALLBACK_MAX_FRAME, minimum=0, maximum=_FALLBACK_MAX_FRAME, parent=self
        )
        self._end_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._apply_small_font(self._end_spin)
        self._end_spin.valueChanged.connect(self._on_spin_changed)
        form.addWidget(self._end_spin)
        # Current playhead frame — the live "where am I" readout.
        form.addWidget(self._label("clip_editor.cur_frame", "Frame"))
        self._frame_value = QLabel("0", self)
        self._frame_value.setStyleSheet(
            f"color: {Config.get_color('highlight')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        self._frame_value.setMinimumWidth(60)
        form.addWidget(self._frame_value)
        form.addStretch(1)
        # Crop — one-click quick-save of the marked range as a segment with an
        # auto-derived name (no dialog).
        self._crop_btn = setButton(
            "clip_editor.crop", 0, 26, kind="border", spec="notice",
            default=tr("clip_editor.crop", "Crop"),
        )
        self._crop_btn.setToolTip(
            tr("clip_editor.crop_tip", "Quick-save the marked range as a segment")
        )
        self._crop_btn.clicked.connect(self._on_crop)
        form.addWidget(self._crop_btn)
        # Mark Segment — opens a naming dialog (name + optional tag;
        # overwrite-confirm on a duplicate name) that persists the marked
        # in/out range as a named segment.
        self._mark_btn = setButton(
            "clip_editor.mark_segment", 0, 26, kind="normal", spec="save",
            default=tr("clip_editor.mark_segment", "Mark Segment"),
        )
        self._mark_btn.setToolTip(
            tr("clip_editor.mark_segment_tip", "Mark the range as a named segment")
        )
        self._mark_btn.clicked.connect(self._on_save_as_clip)
        form.addWidget(self._mark_btn)
        root.addLayout(form)

        # ── Existing segments ──
        # Columns: [☐ (reference-select, embedded only)] Name · Tag · In · Out ·
        # [✗ delete]. The leading checkbox column exists only in embedded mode.
        headers: List[str] = []
        if self._show_reference:
            headers.append("")  # checkbox column — no header
        headers += [
            tr("clip_editor.col_name", "Name"),
            tr("clip_editor.col_tag", "Tag"),
            tr("clip_editor.col_in", "In"),
            tr("clip_editor.col_out", "Out"),
            tr("clip_editor.col_action", ""),
        ]
        self._seg_table = QTableWidget(0, self._n_cols, self)
        self._seg_table.setHorizontalHeaderLabels(headers)
        self._seg_table.verticalHeader().setVisible(False)
        self._seg_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._seg_table.setMinimumHeight(110)
        self._seg_table.setStyleSheet(
            f"QTableWidget {{ background: {Config.get_color('bg_2')};"
            f" color: {Config.get_color('main_t1')};"
            f" gridline-color: {Config.get_color('border_1')};"
            f" border: 1px solid {Config.get_color('border_1')};"
            f" font-size: {_ss()}px; }}"
            f"QHeaderView::section {{ background: {Config.get_color('bg_3')};"
            f" color: {Config.get_color('sub_t1')}; padding: 2px 4px;"
            f" border: none; font-size: {_ss()}px; }}"
        )
        hdr = self._seg_table.horizontalHeader()
        if self._show_reference:
            hdr.setSectionResizeMode(
                self._C_CHECK, QHeaderView.ResizeMode.ResizeToContents
            )
        hdr.setSectionResizeMode(self._C_NAME, QHeaderView.ResizeMode.Stretch)
        for col in (self._C_TAG, self._C_IN, self._C_OUT, self._C_ACT):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._seg_table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._seg_table)

    def _label(self, key: str, default: str) -> QLabel:
        lbl = QLabel(self)
        i18n_bind(lbl, "setText", key, default)
        lbl.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        return lbl

    def _apply_small_font(self, w: QWidget) -> None:
        """Pin an SDK input widget (spin box / line edit) to ``size_small``.

        Those widgets default to ``size_normal`` and size their font via
        ``setFont`` (not QSS), so a later ``setFont`` wins — keeping the whole
        surface at ``size_small`` (LaviButton is already ``size_small``).
        """
        f = w.font()
        f.setPixelSize(_ss())
        w.setFont(f)

    def _build_robot_combo(self) -> Optional[QComboBox]:
        from registers import robots

        skus = robots.list_skus()
        if not skus:
            return None
        # Plain themed QComboBox (not setComboBox): fixed width + stretch 0 so it
        # never grows, and full control over font — LaviComboBox bakes size_small
        # into QSS it rebuilds on every selection, which would wipe a font override.
        combo = QComboBox(self)
        for sku in skus:
            spec = robots.get_robot_spec(sku)
            combo.addItem(spec.name if spec and spec.name else sku)
        self._robot_skus = list(skus)
        combo.setFixedSize(200, 26)
        combo.setStyleSheet(
            f"QComboBox {{ background: {Config.get_color('bg_2')};"
            f" color: {Config.get_color('main_t1')};"
            f" border: 1px solid {Config.get_color('border_1')};"
            f" font-size: {_ss()}px; padding: 1px 4px; }}"
            f"QComboBox QAbstractItemView {{ background: {Config.get_color('bg_2')};"
            f" color: {Config.get_color('main_t1')};"
            f" selection-background-color: {Config.get_color('row_1')};"
            f" font-size: {_ss()}px; }}"
        )
        # Default = the robot whose joints actually match the clip (SKUs are
        # opaque hashes, so a path-string guess can't find them). Fall back to
        # a brand/model/name token match, then leave index 0.
        default_sku = None
        try:
            if self._clip_ref:
                default_sku = self._provider.suggest_sku(self._clip_ref)
        except Exception as exc:
            log_warning(f"[clip_editor] suggest_sku failed: {exc}")
        if default_sku not in skus:
            default_sku = self._guess_sku(self._clip_ref, skus)
        if default_sku in skus:
            combo.setCurrentIndex(skus.index(default_sku))
        # Connect AFTER setting the initial index so it doesn't double-fire
        # the worker start (which _load_clip does explicitly).
        combo.currentIndexChanged.connect(self._on_robot_changed)
        return combo

    @staticmethod
    def _guess_sku(clip_ref: str, skus: List[str]) -> Optional[str]:
        """Fallback guess: match the robot's brand/model/name against the path.

        Used only if :meth:`suggest_sku` (IR-overlap) is unavailable. Matches
        delimited tokens (``/g1/`` not a stray substring) and prefers the
        longest match so ``go2w`` wins over ``go2`` when both appear.
        """
        from registers import robots

        low = (clip_ref or "").lower().replace("\\", "/")

        def delimited(token: str) -> bool:
            token = token.lower().strip()
            if len(token) < 2:
                return False
            return re.search(
                r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", low
            ) is not None

        best_sku: Optional[str] = None
        best_len = 0
        for sku in skus:
            spec = robots.get_robot_spec(sku)
            tokens = {sku.split(":")[-1].lower()}
            if spec is not None:
                for field in (spec.model, spec.brand):
                    if field:
                        tokens.add(str(field).lower())
                        tokens.update(str(field).lower().replace("_", " ").split())
                if spec.name:
                    tokens.update(str(spec.name).lower().split())
            for t in tokens:
                if delimited(t) and len(t) > best_len:
                    best_sku, best_len = sku, len(t)
        return best_sku

    def _build_loading_page(self) -> QWidget:
        """Centred "Loading…" text (top) over a small progress bar (bottom)."""
        page = QWidget(self)
        page.setStyleSheet(
            f"background: {Config.get_color('canvas_bg')};"
            f" border: 1px solid {Config.get_color('border_1')};"
        )
        outer = QVBoxLayout(page)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addStretch(1)

        self._loading_text = QLabel(
            tr("clip_editor.loading", "Loading motion assets…"), page
        )
        self._loading_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_text.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        outer.addWidget(self._loading_text, 0, Qt.AlignmentFlag.AlignHCenter)

        # Small, fixed-width, text-less bar centred under the label. Colours
        # come from theme slots (§5) — no literals.
        self._loading_bar = QProgressBar(page)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedSize(220, 8)
        self._loading_bar.setRange(0, 0)  # indeterminate until first progress
        self._loading_bar.setStyleSheet(
            f"QProgressBar {{ background: {Config.get_color('bg_2')};"
            f" border: 1px solid {Config.get_color('border_1')};"
            f" border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {Config.get_color('highlight')};"
            f" border-radius: 3px; }}"
        )
        bar_row = QHBoxLayout()
        bar_row.addStretch(1)
        bar_row.addWidget(self._loading_bar)
        bar_row.addStretch(1)
        outer.addSpacing(8)
        outer.addLayout(bar_row)
        outer.addStretch(1)
        return page

    def begin_loading(self, text: Optional[str] = None) -> None:
        """Switch the render region to the centred loading page (host-driven)."""
        if text:
            self._loading_text.setText(text)
        self._loading_bar.setRange(0, 0)  # indeterminate spinner
        self._render_stack.setCurrentIndex(1)

    def set_loading_progress(self, done: int, total: int) -> None:
        """Advance the loading bar + count while the host loads assets."""
        if total > 0:
            self._loading_bar.setRange(0, int(total))
            self._loading_bar.setValue(int(done))
            self._loading_text.setText(
                tr("clip_editor.loading_n", "Loading motion assets…  {d}/{t}")
                .format(d=int(done), t=int(total))
            )

    def end_loading(self) -> None:
        """Return the render region to the render view (host-driven)."""
        self._render_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Public API — host-driven clip / robot switching
    # ------------------------------------------------------------------

    def set_clip(
        self,
        clip_ref: Optional[str],
        *,
        initial_selection: Optional[tuple] = None,
        applied_ref: Optional[str] = None,
    ) -> None:
        """Switch the editor to ``clip_ref`` (or clear when falsy).

        ``applied_ref`` is the full reference currently bound as the training
        item's clip (a ``#seg=`` segment ref or the whole-clip base ref) — it
        drives which segment row shows checked + highlighted.

        Switching to a *different* base clip tears down and rebuilds the render
        worker. Re-selecting the *same* base clip (e.g. checking a different
        segment of the clip already loaded) is cheap: it only re-points the
        selection + repaints the checkbox column, so the GL worker is never
        thrashed.
        """
        new_ref = str(clip_ref) if clip_ref else ""
        cur = self._clip_ref or ""
        self._applied_seg_ref = str(applied_ref) if applied_ref else ""
        if new_ref == cur and new_ref:
            # Same clip already loaded — just re-point selection + checkmarks.
            self._initial_selection = initial_selection
            if self._worker_ready and initial_selection is not None:
                self._apply_selection(initial_selection)
            self._refresh_seg_selection_visuals()
            return
        self._load_clip(new_ref or None, initial_selection=initial_selection)

    def set_robot_sku(self, sku: Optional[str]) -> None:
        """Re-pin the fixed render robot (embedded mode) and re-render."""
        self._robot_sku = sku or None
        if self._show_robot_picker:
            return  # picker mode owns robot selection itself
        self._last_qimage = None
        cur_sku = self._current_sku()
        if self._clip_ref and cur_sku:
            self._start_worker(cur_sku)  # keeps existing markers (_range_initialised)
        else:
            self._teardown_worker()
            self._render_label.setPixmap(QPixmap())
            self._render_label.setText(
                self._no_robot_message() if self._clip_ref
                else tr("clip_editor.pick_clip", "Pick a clip to preview.")
            )

    def teardown(self) -> None:
        """Release the render worker + GL context. Host MUST call on close."""
        self._teardown_worker()

    # ------------------------------------------------------------------
    # Clip loading
    # ------------------------------------------------------------------

    def _load_clip(self, clip_ref: Optional[str], *, initial_selection: Optional[tuple] = None) -> None:
        self._teardown_worker()
        self._clip_ref = str(clip_ref) if clip_ref else ""
        self._update_marker_actions_enabled()  # gate Crop / Mark Segment (req 1)
        self._initial_selection = initial_selection
        self._range_initialised = False
        self._last_qimage = None
        if self._top_clip_label is not None:
            self._top_clip_label.setText(Path(self._clip_ref).name if self._clip_ref else "")
            self._top_clip_label.setToolTip(self._clip_ref)

        if not self._clip_ref:
            self._n_frames, self._fps, self._frames_known = 0, 0.0, False
            self._timeline.set_frame_count(_FALLBACK_MAX_FRAME)
            self._timeline.set_fps(0.0)
            self._timeline.set_known(False)
            self._render_label.setPixmap(QPixmap())
            self._render_label.setText(
                tr("clip_editor.pick_clip", "Pick a clip to preview.")
            )
            self._reload_segments()
            self._update_info_label()
            self._update_range_label()
            self._update_frame_label()
            return

        sku = self._current_sku()
        if sku:
            # A render worker will parse the clip on its OWN thread and report
            # the true ``(n_frames, fps)`` via ``ready`` (which then
            # initialises the in/out selection). We deliberately do NOT probe
            # the clip on the GUI thread here — parsing a large motion file is
            # exactly the kind of blocking work that must stay off the main
            # thread (host req 3). The timeline shows a fallback range for the
            # brief moment until ``ready`` arrives.
            self._n_frames, self._fps, self._frames_known = 0, 0.0, False
            self._timeline.set_frame_count(_FALLBACK_MAX_FRAME)
            self._timeline.set_fps(0.0)
            self._timeline.set_known(False)
            self._syncing = True
            try:
                self._start_spin.setMaximum(_FALLBACK_MAX_FRAME)
                self._end_spin.setMaximum(_FALLBACK_MAX_FRAME)
            finally:
                self._syncing = False
            self._reload_segments()
            self._update_info_label()
            self._start_worker(sku)
            return

        # No robot to render with — there is no worker thread to lean on, so
        # probe the length on the GUI thread (one clip) to give the timeline a
        # real range and let the user mark segments blind. Then initialise the
        # selection here (no ``ready`` will arrive to do it).
        try:
            self._n_frames, self._fps = self._provider.probe_clip(self._clip_ref)
            self._frames_known = self._n_frames > 0
        except Exception as exc:  # fail-soft in the UI; provider stays loud
            log_warning(f"[clip_editor] probe_clip failed for {self._clip_ref!r}: {exc}")
            self._n_frames, self._fps, self._frames_known = 0, 0.0, False
        self._timeline.set_frame_count(
            self._n_frames if self._frames_known else _FALLBACK_MAX_FRAME
        )
        self._timeline.set_fps(self._fps)
        self._timeline.set_known(self._frames_known)
        smax = (self._n_frames - 1) if self._frames_known else _FALLBACK_MAX_FRAME
        self._syncing = True
        try:
            self._start_spin.setMaximum(smax)
            self._end_spin.setMaximum(smax)
        finally:
            self._syncing = False
        self._timer.setInterval(self._frame_interval_ms())
        self._reload_segments()
        self._update_info_label()
        self._render_label.setPixmap(QPixmap())
        self._render_label.setText(self._no_robot_message())
        self._apply_selection(self._initial_selection)

    def _no_robot_message(self) -> str:
        if self._show_robot_picker:
            return tr("clip_editor.no_robot", "Pick a render robot first.")
        return tr(
            "clip_editor.no_canvas_robot",
            "No robot bound — connect a Robot node on the canvas to preview.",
        )

    def _apply_selection(self, sel: Optional[tuple]) -> None:
        """Set the in/out selection (spins + timeline) from ``sel`` or whole clip."""
        last = (self._n_frames - 1) if self._frames_known else _FALLBACK_MAX_FRAME
        last = max(0, last)
        if sel is not None:
            lo = max(0, min(int(sel[0]), last))
            hi = max(lo, min(int(sel[1]), last))
        else:
            lo, hi = 0, last
        prev = self._syncing
        self._syncing = True
        try:
            self._start_spin.setValue(lo)
            self._end_spin.setValue(hi)
            self._timeline.set_selection(lo, hi)
            self._timeline.set_playhead(lo)
        finally:
            self._syncing = prev
        self._range_initialised = True
        self._update_range_label()
        self._update_frame_label()
        if self._worker_ready:
            self._request(lo)

    # ------------------------------------------------------------------
    # Render worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(self, sku: str) -> None:
        self._teardown_worker()
        if not sku:
            self._render_label.setText(self._no_robot_message())
            return
        from application.ui.dialogs.clip_render_worker import ClipRenderWorker

        self._render_label.setText(tr("clip_editor.rendering", "Rendering…"))
        self._worker_ready = False
        self._thread = QThread()
        self._worker = ClipRenderWorker(sku, self._clip_ref, _RENDER_W, _RENDER_H)
        self._worker.moveToThread(self._thread)
        self._worker.ready.connect(self._on_render_ready)
        self._worker.frameReady.connect(self._on_frame_ready)
        self._worker.failed.connect(self._on_render_failed)
        self._thread.started.connect(self._worker.open)
        # Canonical Qt cleanup: once the worker thread's event loop exits, the
        # thread drives deletion of both the worker (on its own thread, via the
        # finish-time DeferredDelete flush) and itself (on the GUI thread).
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _teardown_worker(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._set_play_button(False)
        self._in_flight = False
        self._pending_frame = None
        self._pending_d_az = self._pending_d_el = 0.0
        self._pending_zoom = 1.0
        self._worker_ready = False
        worker, thread = self._worker, self._thread
        self._worker = None
        self._thread = None
        if worker is not None:
            # Drop our slots first so in-flight queued signals from this worker
            # can't reach the widget after a robot switch (stale-frame flicker).
            try:
                worker.ready.disconnect(self._on_render_ready)
                worker.frameReady.disconnect(self._on_frame_ready)
                worker.failed.disconnect(self._on_render_failed)
            except (TypeError, RuntimeError):
                pass
        if thread is not None:
            if worker is not None and thread.isRunning():
                # Release the thread-affine GL context ON the worker thread,
                # synchronously, BEFORE the loop exits. Blocking is safe:
                # teardown always runs on the GUI thread, never the worker's.
                QMetaObject.invokeMethod(
                    worker, "shutdown", Qt.ConnectionType.BlockingQueuedConnection
                )
            thread.quit()
            if not thread.wait(5000):
                log_warning(
                    "[clip_editor] render thread did not stop within 5s during "
                    "teardown — proceeding (GL context may not be released)."
                )
            # Deletion is driven by thread.finished (wired in _start_worker);
            # no manual deleteLater here (it would post to a dead event loop).

    def _request(
        self, frame: Optional[int] = None, *,
        d_az: float = 0.0, d_el: float = 0.0, zoom: float = 1.0,
    ) -> None:
        """Render a frame (optionally applying an orbit/zoom delta), coalesced."""
        if self._worker is None or not self._worker_ready:
            return
        if frame is None:
            frame = self._timeline.playhead()
        frame = int(frame)
        if self._in_flight:
            self._pending_frame = frame
            self._pending_d_az += d_az
            self._pending_d_el += d_el
            self._pending_zoom *= zoom
            return
        self._in_flight = True
        d_az += self._pending_d_az
        d_el += self._pending_d_el
        zoom *= self._pending_zoom
        self._pending_frame = None
        self._pending_d_az = self._pending_d_el = 0.0
        self._pending_zoom = 1.0
        QMetaObject.invokeMethod(
            self._worker, "render", Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, frame), Q_ARG(float, float(d_az)),
            Q_ARG(float, float(d_el)), Q_ARG(float, float(zoom)),
        )

    @pyqtSlot(int, float)
    def _on_render_ready(self, n_frames: int, fps: float) -> None:
        self._n_frames = int(n_frames)
        self._fps = float(fps)
        self._frames_known = self._n_frames > 0
        self._worker_ready = True
        self._timeline.set_frame_count(self._n_frames)
        self._timeline.set_fps(self._fps)
        self._timeline.set_known(True)
        last = max(0, self._n_frames - 1)
        self._syncing = True
        try:
            self._start_spin.setMaximum(last)
            self._end_spin.setMaximum(last)
        finally:
            self._syncing = False
        if not self._range_initialised:
            # First load: default the in/out selection to the whole clip,
            # unless the host drilled in on a specific segment range.
            self._apply_selection(self._initial_selection)
        else:
            # Robot switch (same clip, same n_frames) — preserve the user's
            # markers, just re-clamp them to the (unchanged) range.
            self._syncing = True
            try:
                lo = min(self._start_spin.value(), last)
                hi = min(self._end_spin.value(), last)
                self._start_spin.setValue(lo)
                self._end_spin.setValue(hi)
                self._timeline.set_selection(lo, hi)
            finally:
                self._syncing = False
        self._timer.setInterval(self._frame_interval_ms())
        self._update_info_label()
        self._update_range_label()
        self._update_frame_label()
        self._reload_segments()
        self._request(self._timeline.playhead())

    @pyqtSlot(int, QImage)
    def _on_frame_ready(self, idx: int, img: QImage) -> None:
        self._last_qimage = img
        self._update_render_pixmap()
        self._in_flight = False
        # Flush any request that arrived while this one was rendering (a newer
        # frame and/or accumulated orbit/zoom).
        if (
            self._pending_frame is not None
            or self._pending_d_az or self._pending_d_el
            or self._pending_zoom != 1.0
        ):
            self._request(self._pending_frame)

    @pyqtSlot(str)
    def _on_render_failed(self, msg: str) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._set_play_button(False)
        # A failed worker can't render — disarm the request state machine so a
        # later scrub/seek isn't stuck coalescing into a frame that never comes.
        self._worker_ready = False
        self._in_flight = False
        self._pending_frame = None
        self._pending_d_az = self._pending_d_el = 0.0
        self._pending_zoom = 1.0
        self._render_label.setText(
            tr("clip_editor.render_failed", "Render failed:\n{msg}").format(msg=msg)
        )
        log_warning(f"[clip_editor] render failed: {msg}")

    def _update_render_pixmap(self) -> None:
        if self._last_qimage is None or self._last_qimage.isNull():
            return
        pm = QPixmap.fromImage(self._last_qimage)
        self._render_label.setPixmap(pm.scaled(
            self._render_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    # ------------------------------------------------------------------
    # Playback + scrubbing
    # ------------------------------------------------------------------

    def _frame_interval_ms(self) -> int:
        """Playback tick interval; falls back to 30 fps when fps is unknown."""
        fps = self._fps if self._fps and self._fps > 0 else 30.0
        return max(1, int(round(1000.0 / fps)))

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._set_play_button(False)
            return
        # Only play once a render worker is ready (avoids a 0-ms busy-loop and
        # posting frames to a not-yet-ready / failed worker).
        if self._worker is None or not self._worker_ready or self._n_frames <= 1:
            return
        self._timer.setInterval(self._frame_interval_ms())
        self._timer.start()
        self._set_play_button(True)

    def _set_play_button(self, playing: bool) -> None:
        # Icon-only transport: swap play ⇄ pause glyph (no text — §3).
        self._play_btn.setIcon(self._pause_icon if playing else self._play_icon)
        self._play_btn.setToolTip(
            tr("clip_editor.pause", "Pause") if playing
            else tr("clip_editor.play", "Play")
        )

    def _on_stop(self) -> None:
        """Stop playback and rewind to frame 0 (video-editor stop semantics)."""
        if self._timer.isActive():
            self._timer.stop()
        self._set_play_button(False)
        self._timeline.set_playhead(0)
        self._update_frame_label()
        self._request(0)

    def _on_tick(self) -> None:
        nxt = self._timeline.playhead() + 1
        if nxt >= self._n_frames:
            nxt = 0
        self._timeline.set_playhead(nxt)
        self._update_frame_label()
        self._request(nxt)

    def _on_seek(self, frame: int) -> None:
        # A click-to-seek pauses playback too, so a single gesture has the same
        # playback semantics whether or not the cursor jitters into a drag.
        if self._timer.isActive():
            self._timer.stop()
            self._set_play_button(False)
        self._update_frame_label()
        self._request(frame)

    def _on_playhead_moved(self, frame: int) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._set_play_button(False)
        self._update_frame_label()
        self._request(frame)

    def _on_range_changed(self, lo: int, hi: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._start_spin.setValue(int(lo))
            self._end_spin.setValue(int(hi))
        finally:
            self._syncing = False
        self._update_range_label()

    def _on_spin_changed(self, _v: int = 0) -> None:
        if self._syncing:
            return
        lo, hi = self._start_spin.value(), self._end_spin.value()
        self._syncing = True
        try:
            self._timeline.set_selection(lo, hi)
        finally:
            self._syncing = False
        self._update_range_label()

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _update_info_label(self) -> None:
        if self._frames_known and self._fps > 0:
            self._info_label.setText(
                tr("clip_editor.frames_fps", "{n} frames · {fps:.0f} fps · {dur:.1f}s")
                .format(n=self._n_frames, fps=self._fps,
                        dur=self._n_frames / self._fps)
            )
        elif self._frames_known:
            self._info_label.setText(
                tr("clip_editor.frames_only", "{n} frames").format(n=self._n_frames)
            )
        else:
            self._info_label.setText(
                tr("clip_editor.length_unknown", "clip length unknown")
            )

    def _update_range_label(self) -> None:
        lo, hi = self._timeline.selection()
        self._range_label.setText(tr(
            "clip_editor.range", "In {a} · Out {b} · {n} frames"
        ).format(a=lo, b=hi, n=max(0, hi - lo)))

    def _update_frame_label(self) -> None:
        cur = self._timeline.playhead()
        total = max(0, self._n_frames - 1)
        self._frame_value.setText(f"{cur} / {total}")

    # ------------------------------------------------------------------
    # Segment CRUD
    # ------------------------------------------------------------------

    def _markers(self) -> tuple:
        return int(self._start_spin.value()), int(self._end_spin.value())

    def _current_sku(self) -> str:
        if self._robot_combo is not None and self._robot_skus:
            idx = self._robot_combo.currentIndex()
            if 0 <= idx < len(self._robot_skus):
                return self._robot_skus[idx]
            return ""
        # Fixed (embedded) mode.
        return self._robot_sku or ""

    def _on_robot_changed(self, _idx: int) -> None:
        self._last_qimage = None
        self._start_worker(self._current_sku())

    def _update_marker_actions_enabled(self) -> None:
        """Crop / Mark Segment only make sense once a clip is loaded (req 1)."""
        has_clip = bool(self._clip_ref)
        for btn in (getattr(self, "_crop_btn", None), getattr(self, "_mark_btn", None)):
            if btn is not None:
                btn.setEnabled(has_clip)

    def _on_crop(self) -> None:
        """Quick-save the marked in/out range as a segment with an auto-name.

        Unlike "Mark Segment" this does not prompt: the name is derived from the
        clip stem + frame range so a segment can be cut in one click.
        """
        if not self._clip_ref:
            return
        lo, hi = self._markers()
        name = f"{Path(self._clip_ref).stem}_{lo}-{hi}"
        try:
            self._provider.add_segment(
                self._clip_ref, name=name, start_frame=lo, end_frame=hi, task_tag=""
            )
        except (ValueError, KeyError, RuntimeError) as e:
            QMessageBox.critical(self, self._dialog_title(), f"{type(e).__name__}: {e}")
            return
        self._reload_segments()
        self.segmentsChanged.emit()

    def _on_save_as_clip(self) -> None:
        """Open the naming dialog and persist the marked range as a segment.

        The dialog collects a required name + optional tag and confirms an
        overwrite when the name collides with an existing segment (otherwise it
        keeps the user on the name field). Overwrite = replace the same-named
        segment's range/tag.
        """
        if not self._clip_ref:
            return
        lo, hi = self._markers()
        try:
            existing = {s.name: s for s in self._provider.list_segments(self._clip_ref)}
        except Exception as exc:
            log_warning(f"[clip_editor] list_segments failed: {exc}")
            existing = {}
        dlg = _SaveSegmentDialog(self, existing_names=set(existing.keys()))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.segment_name()
        tag = dlg.segment_tag()
        try:
            if dlg.overwrite() and name in existing:
                self._provider.delete_segment(
                    self._clip_ref, existing[name].segment_id
                )
            self._provider.add_segment(
                self._clip_ref, name=name, start_frame=lo, end_frame=hi, task_tag=tag
            )
        except (ValueError, KeyError, RuntimeError) as e:
            QMessageBox.critical(self, self._dialog_title(), f"{type(e).__name__}: {e}")
            return
        self._reload_segments()
        self.segmentsChanged.emit()

    def _on_delete(self, segment_id: str) -> None:
        try:
            self._provider.delete_segment(self._clip_ref, segment_id)
        except (KeyError, RuntimeError) as e:
            QMessageBox.critical(self, self._dialog_title(), f"{type(e).__name__}: {e}")
            return
        self._reload_segments()
        self.segmentsChanged.emit()

    def _seg_ref_of(self, seg) -> str:
        """Encoded ``<base>#seg=lo-hi`` ref for ``seg`` (or "" on bad range)."""
        from application.training.motion.segment_ref import build_segment_ref
        try:
            return build_segment_ref(
                self._clip_ref, int(seg.start_frame), int(seg.end_frame)
            )
        except ValueError as exc:
            log_warning(f"[clip_editor] build_segment_ref failed: {exc}")
            return ""

    def _on_seg_check_toggled(self, row: int, checked: bool) -> None:
        """A segment's reference checkbox toggled — rebind the training reference.

        Checking a segment applies it (single-select — the host round-trips and
        re-paints the column so peers uncheck); unchecking the active one falls
        back to the whole clip (req 4 / req 5).
        """
        if not (0 <= row < len(self._row_segs)):
            return
        if checked:
            ref = self._seg_ref_of(self._row_segs[row])
            if ref:
                self.segmentApplied.emit(ref)
        else:
            # Unchecked the applied segment → reference becomes the whole clip.
            if self._clip_ref:
                self.segmentApplied.emit(self._clip_ref)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating:
            return
        col = item.column()
        if col == self._C_CHECK:
            self._on_seg_check_toggled(
                item.row(), item.checkState() == Qt.CheckState.Checked
            )
            return
        seg_id = item.data(Qt.ItemDataRole.UserRole)
        if not seg_id:
            return
        text = (item.text() or "").strip()
        try:
            if col == self._C_NAME:
                if not text:  # empty name → revert by reloading
                    self._reload_segments()
                    return
                self._provider.rename_segment(self._clip_ref, str(seg_id), text)
            elif col == self._C_TAG:
                self._provider.set_segment_tag(self._clip_ref, str(seg_id), text)
        except (ValueError, KeyError, RuntimeError) as e:
            QMessageBox.critical(self, self._dialog_title(), f"{type(e).__name__}: {e}")
            self._reload_segments()
            return
        self.segmentsChanged.emit()

    def _reload_segments(self) -> None:
        self._populating = True
        try:
            segs: List = []
            if self._clip_ref:
                try:
                    segs = self._provider.list_segments(self._clip_ref)
                except Exception as exc:
                    log_warning(
                        f"[clip_editor] list_segments failed for {self._clip_ref!r}: {exc}"
                    )
                    segs = []
            self._row_segs = list(segs)
            self._seg_table.setRowCount(len(segs))
            for r, s in enumerate(segs):
                checked = (
                    self._show_reference
                    and self._applied_seg_ref
                    and self._seg_ref_of(s) == self._applied_seg_ref
                )
                if self._show_reference:
                    chk = QTableWidgetItem()
                    chk.setFlags(
                        Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                    )
                    chk.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
                    chk.setToolTip(
                        tr("clip_editor.ref_tip",
                           "Use this segment as the training reference")
                    )
                    self._seg_table.setItem(r, self._C_CHECK, chk)

                name_item = QTableWidgetItem(s.name)
                name_item.setData(Qt.ItemDataRole.UserRole, s.segment_id)
                self._seg_table.setItem(r, self._C_NAME, name_item)

                tag_item = QTableWidgetItem(s.task_tag)
                tag_item.setData(Qt.ItemDataRole.UserRole, s.segment_id)
                self._seg_table.setItem(r, self._C_TAG, tag_item)

                for col, val in ((self._C_IN, s.start_frame), (self._C_OUT, s.end_frame)):
                    cell = QTableWidgetItem(str(val))
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._seg_table.setItem(r, col, cell)

                self._paint_seg_row(r, bool(checked))
                self._seg_table.setCellWidget(
                    r, self._C_ACT, self._build_action_cell(r, s)
                )
        finally:
            self._populating = False
        # Paint saved segments as coloured bands on the timeline.
        self._timeline.set_segments(
            [(s.start_frame, s.end_frame, i) for i, s in enumerate(segs)]
        )

    def _paint_seg_row(self, row: int, checked: bool) -> None:
        """Colour a segment row: selected → safe_zone bold, else sub_t1."""
        color = Config.get_color("safe_zone") if checked else Config.get_color("sub_t1")
        brush = QBrush(QColor(color))
        for col in (self._C_NAME, self._C_TAG, self._C_IN, self._C_OUT):
            it = self._seg_table.item(row, col)
            if it is None:
                continue
            it.setForeground(brush)
            f = it.font()
            f.setBold(checked)
            it.setFont(f)

    def _refresh_seg_selection_visuals(self) -> None:
        """Re-sync the reference checkbox + row colour of every row in place.

        Called after the applied reference changes (host round-trip) without a
        full table rebuild — so the checkbox that fired isn't destroyed under
        its own signal.
        """
        if not self._show_reference:
            return
        self._populating = True
        try:
            for r, s in enumerate(self._row_segs):
                checked = bool(
                    self._applied_seg_ref
                    and self._seg_ref_of(s) == self._applied_seg_ref
                )
                chk = self._seg_table.item(r, self._C_CHECK)
                if chk is not None:
                    chk.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
                self._paint_seg_row(r, checked)
        finally:
            self._populating = False

    def _build_action_cell(self, row: int, seg) -> QWidget:
        """Action cell: a single icon-only Delete (icon_no) button per row."""
        cell = QWidget(self._seg_table)
        cell.setStyleSheet("background: transparent;")
        h = QHBoxLayout(cell)
        h.setContentsMargins(2, 1, 2, 1)
        h.setSpacing(4)
        h.addStretch(1)
        del_btn = setButton(
            f"clip_editor.seg_del.{row}", 24, 20, kind="border", spec="danger",
            icon="icon_no", icon_only=True, default="",
        )
        del_btn.setToolTip(tr("clip_editor.delete", "Delete"))
        del_btn.clicked.connect(
            lambda _c=False, sid=seg.segment_id: self._on_delete(sid)
        )
        h.addWidget(del_btn)
        h.addStretch(1)
        return cell

    def _dialog_title(self) -> str:
        w = self.window()
        title = w.windowTitle() if w is not None else ""
        return title or tr("clip_editor.title", "Clip Motion Editor")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._update_render_pixmap()


class _SaveSegmentDialog(QDialog):
    """Modal "Save as Clip" naming popup: name (required) + tag (optional).

    On Save it validates the name and, when it collides with an existing
    segment, asks the user to confirm an overwrite — declining keeps the dialog
    open so they can change the name (never a silent no-op). Exposes
    :meth:`segment_name` / :meth:`segment_tag` / :meth:`overwrite` on accept.
    """

    def __init__(self, parent: QWidget, *, existing_names: set) -> None:
        super().__init__(parent)
        self._existing = set(existing_names or set())
        self._name_val = ""
        self._tag_val = ""
        self._overwrite = False

        self.setModal(True)
        self.setWindowTitle(tr("clip_editor.mark_seg_title", "Mark Segment"))
        self.setStyleSheet(
            f"QDialog {{ background: {Config.get_color('bg_1')};"
            f" color: {Config.get_color('main_t1')}; }}"
        )
        self.setMinimumWidth(360)

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        def _lbl(text: str) -> QLabel:
            lab = QLabel(text, self)
            lab.setStyleSheet(
                f"color: {Config.get_color('main_t1')};"
                f" background: transparent; font-size: {_ss()}px;"
            )
            return lab

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(_lbl(tr("clip_editor.seg_name", "Name")))
        self._name = setLineEdit(
            placeholder=tr("clip_editor.seg_name_ph", "e.g. walk_loop"), parent=self
        )
        f = self._name.font()
        f.setPixelSize(_ss())
        self._name.setFont(f)
        name_row.addWidget(self._name, 1)
        v.addLayout(name_row)

        tag_row = QHBoxLayout()
        tag_row.setSpacing(6)
        tag_row.addWidget(_lbl(tr("clip_editor.seg_tag", "Tag")))
        self._tag = QComboBox(self)
        self._tag.setEditable(True)
        self._tag.addItems(list(_TASK_TAG_SUGGESTIONS))
        self._tag.setCurrentText("")
        self._tag.setStyleSheet(
            f"QComboBox {{ background: {Config.get_color('bg_2')};"
            f" color: {Config.get_color('main_t1')};"
            f" border: 1px solid {Config.get_color('border_1')};"
            f" font-size: {_ss()}px; padding: 1px 4px; }}"
        )
        tag_opt = _lbl(tr("clip_editor.optional", "(optional)"))
        tag_opt.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        tag_row.addWidget(self._tag, 1)
        tag_row.addWidget(tag_opt, 0)
        v.addLayout(tag_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = setButton(
            "clip_editor.cancel", 92, 28, kind="border", spec="danger",
            default=tr("clip_editor.cancel", "Cancel"),
        )
        cancel_btn.clicked.connect(self.reject)
        save_btn = setButton(
            "clip_editor.save", 92, 28, kind="border", spec="save",
            default=tr("clip_editor.save", "Save"),
        )
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        v.addLayout(btn_row)

    def _on_save(self) -> None:
        name = (self._name.text() or "").strip()
        if not name:
            QMessageBox.warning(
                self, self.windowTitle(),
                tr("clip_editor.err_name", "Segment name is required."),
            )
            return
        if name in self._existing:
            r = QMessageBox.question(
                self, self.windowTitle(),
                tr("clip_editor.overwrite_q",
                   "A clip named '{name}' already exists. Overwrite it?")
                .format(name=name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                # Declined → keep the dialog open so the user renames.
                self._name.setFocus()
                self._name.selectAll()
                return
            self._overwrite = True
        else:
            self._overwrite = False
        self._name_val = name
        self._tag_val = (self._tag.currentText() or "").strip()
        self.accept()

    def segment_name(self) -> str:
        return self._name_val

    def segment_tag(self) -> str:
        return self._tag_val

    def overwrite(self) -> bool:
        return self._overwrite


__all__ = ["ClipMotionEditorWidget"]
