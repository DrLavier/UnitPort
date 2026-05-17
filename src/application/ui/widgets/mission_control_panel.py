"""MissionControlPanel -- top-level main work area shell.

This panel is now an **overlay** raised above an external CanvasPage. The
host (``MainWindow.main_panel``) parents the canvas at the bottom layer
(QVBoxLayout-managed, fills the panel) and parents this MissionControlPanel
as a free-floating sibling raised on top. The host calls
:meth:`set_canvas` to give this panel a reference to the canvas it
overlays, then watches :attr:`overlay_compact_changed` to resize this
panel between two states::

    Mission Control mode  → fills the entire main_panel
                            (chart visible at user-set opacity over canvas)
    Training Canva mode   → shrinks to top_row strip only
    + canvas loaded       (canvas fully exposed below the strip)

Layout::

    +------------------------------ top_row (always visible, opaque) ----+
    | [run dropdown | opacity slider | node lib btn]      <slider switch>|
    +-------------- vertical splitter (drag, cursor) -- hidden in TC ----+
    |        main_screen (top, flex, transparent — chart applies opacity)|
    +-------------- vertical split ---------------------------------------+
    |     mission_panel (init H = 400, opaque)                            |
    +---------------------------------------------------------------------+

The ``files_list`` QWidget is constructed by this panel but **not** placed
inside its own layout — it is an orphan widget that ``MainWindow`` adopts
into the sidebar's Project Files panel via :meth:`take_files_list_widget`.

* ``files_list``  — single ``LaviTabTable`` (single-select) carrying
  the ``Canvas | Script`` tab head.
* ``top_row``     — fixed-height opaque strip. Hosts:
  - Mission Control middle widgets (RunSourceSelector + opacity QSlider).
    Training Canva mode shows no middle widgets here — its Node Library
    palette is permanently mounted on the canvas itself.
  - SliderSwitch on the right, hidden until a canvas file is loaded into
    the external canvas.
* ``main_screen`` — transparent QWidget hosting the content stack:
  - Canvas tab inner stack: ``new_file_panel`` placeholder (no selection)
    or the ``TrainingChartPanel`` chart (plain QWidget composite, no
    QGraphicsView inside) wrapped in a ``QGraphicsOpacityEffect`` driven
    by the top_row's opacity slider (90-100%).
  - Script tab: standby placeholder ↔ CodeEditorWidget toolbar+editor.
* ``mission_panel`` — opaque bottom region, currently a placeholder.

In Training Canva mode the splitter (and therefore both ``main_screen``
and ``mission_panel``) is hidden via ``setVisible(False)`` — the host
shrinks this panel's geometry to ``top_row.height`` so the canvas behind
becomes fully visible and click-through.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    CodeEditorWidget,
    Config,
    I18nLabel,
    LaviTabTable,
    SliderSwitch,
    i18n_bind,
    log_debug,
    log_warning,
    setButton,
    setLaviTabTable,
    tr,
)

from application.service.projects import ProjectInfo, resolve_file
from application.ui.canvas import CanvasPage
from application.ui.widgets.policy_simulation_card import (
    PolicySimulationCard,
)
from application.ui.widgets.real_robot_connection_card import (
    RealRobotConnectionCard,
)
from application.ui.widgets.run_source_selector import RunSourceSelector
from application.ui.widgets.tc_tool_buttons import TCToolButtonsCluster
from application.ui.widgets.training_chart_panel import TrainingChartPanel
from application.ui.widgets.training_config_card import (
    TrainingConfigPerspectiveCard,
)


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
MODE_MISSION_CONTROL = "mission_control"
MODE_TRAINING_CANVA = "training_canva"

_MODE_OPTIONS = [
    (MODE_MISSION_CONTROL, "Mission Control"),
    (MODE_TRAINING_CANVA, "Training Canva"),
]


# ---------------------------------------------------------------------------
# MissionControlPanel
# ---------------------------------------------------------------------------
class MissionControlPanel(QWidget):
    """Two-pane main work area: top_row + (main_screen / mission_panel).

    Used as an overlay above an external ``CanvasPage`` — the host sets the
    canvas via :meth:`set_canvas` and listens to
    :attr:`overlay_compact_changed` to resize the panel between
    full-cover (MC mode) and top_row-only (TC mode).
    """

    # -- signals -------------------------------------------------------
    tab_changed = pyqtSignal(str)              # "Canvas" | "Script"
    selection_changed = pyqtSignal(str, str)   # (tab, file_id)
    mode_changed = pyqtSignal(str)             # "mission_control" | "training_canva"
    # Whenever the splitter visibility flips → host should re-apply
    # this overlay's geometry (full-cover vs top_row-only).
    overlay_compact_changed = pyqtSignal(bool)  # True = compact (top_row only)
    # Fired after a canvas file_id has fully loaded into the embedded
    # CanvasPage (success path only). Carried payload is the file_id, e.g.
    # "canvas/sb3_mujoco/main.canvas.json" — same shape used by the sidebar
    # Project Files list. MainWindow uses this to persist the
    # (project, canvas) pair to user.ini for next-launch auto-open and to
    # flip the host _MainPanel from "picker" mode to "canvas" mode.
    canvas_loaded = pyqtSignal(str)

    _FILES_LIST_MIN_W = 180
    _MISSION_PANEL_DEFAULT_H = 450
    _MISSION_PANEL_MIN_H = 80
    _MAIN_SCREEN_MIN_H = 120
    _HANDLE_W = 4

    # Top-row geometry.
    _TOP_ROW_H = 43
    _OPACITY_SLIDER_W = 110

    # Chart overlay opacity slider range (percent).
    _OPACITY_MIN = 90
    _OPACITY_MAX = 100
    _OPACITY_DEFAULT = 95

    # Tab ids.
    _TAB_CANVAS_KEY = "missioncontrol.tab.canvas"
    _TAB_SCRIPT_KEY = "missioncontrol.tab.script"
    _TAB_KEY_TO_NAME = {
        _TAB_CANVAS_KEY: "Canvas",
        _TAB_SCRIPT_KEY: "Script",
    }
    _TAB_NAME_TO_KEY = {v: k for k, v in _TAB_KEY_TO_NAME.items()}

    # Inner Canvas-body stack indices.
    _CANVAS_BODY_NEW_FILE = 0
    _CANVAS_BODY_CHART = 1

    _SCRIPT_PLACEHOLDER_KEY = "missioncontrol.script.placeholder"
    _SCRIPT_PLACEHOLDER_DEFAULT = "Select a script file to edit"
    _SCRIPT_SAVE_KEY = "missioncontrol.btn.save_script"
    _SCRIPT_SAVE_DEFAULT = "Save"
    _NEW_FILE_PLACEHOLDER_KEY = "missioncontrol.canvas.new_file_placeholder"
    _NEW_FILE_PLACEHOLDER_DEFAULT = "Select a canvas file to begin"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionControlPanel")
        # Transparent background — the chart overlay's opacity reveals the
        # canvas below; only top_row + mission_panel paint opaque chrome.
        self.setStyleSheet(
            "QWidget#missionControlPanel { background: transparent; }"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # ---- public state ------------------------------------------
        self.current_tab: str = "Canvas"
        self.current_canvas: Optional[str] = None
        self.current_script: Optional[str] = None

        # ---- data caches ------------------------------------------
        self._canvas_groups: List[Dict] = []
        self._script_groups: List[Dict] = []

        # ---- project context ----------------------------------------
        self._project_info: Optional[ProjectInfo] = None
        self._script_loaded_id: Optional[str] = None
        self._script_target_path: Optional[Path] = None
        # Mirrors current_canvas after a successful auto-load.
        self._canvas_loaded_id: Optional[str] = None
        # External canvas (set by host via set_canvas).
        self._external_canvas: Optional[CanvasPage] = None

        # ---- mode state --------------------------------------------
        # Persisted; effective mode = _mode if canvas loaded else MC.
        self._mode: str = self._restore_mode()
        # First-canvas-load-per-session forces MC mode once.
        self._first_load_force_mc_done: bool = False
        # Cache of last splitter-visible state, so we only emit
        # overlay_compact_changed on actual flips.
        self._is_compact_state: bool = False
        # Chart overlay opacity (percent in [_OPACITY_MIN, _OPACITY_MAX]).
        self._chart_opacity_pct: int = self._restore_opacity()

        # ---- widget slots ------------------------------------------
        self._files_list: Optional[QWidget] = None
        self._files_list_layout: Optional[QVBoxLayout] = None
        self._files_table: Optional[LaviTabTable] = None

        self._top_row_host: Optional[QFrame] = None
        self._mode_switch: Optional[SliderSwitch] = None
        self._run_source_selector: Optional[RunSourceSelector] = None
        self._opacity_icon: Optional[QLabel] = None
        self._opacity_slider: Optional[QSlider] = None
        self._tc_tools: Optional[TCToolButtonsCluster] = None

        self._v_splitter: Optional[QSplitter] = None
        self._main_screen: Optional[QWidget] = None
        self._mission_panel: Optional[QWidget] = None

        self._content_stack: Optional[QStackedWidget] = None
        self._canvas_page_widget: Optional[QWidget] = None
        self._canvas_body_stack: Optional[QStackedWidget] = None
        self._chart_panel: Optional[TrainingChartPanel] = None
        self._chart_opacity_effect: Optional[QGraphicsOpacityEffect] = None
        self._new_file_panel: Optional[QWidget] = None

        self._script_page: Optional[QWidget] = None
        self._script_stack: Optional[QStackedWidget] = None
        self._script_placeholder: Optional[I18nLabel] = None
        self._script_editor: Optional[CodeEditorWidget] = None
        self._script_save_btn = None
        self._script_path_label: Optional[QLabel] = None
        self._script_status: Optional[QLabel] = None
        self._script_toolbar: Optional[QFrame] = None

        self._train_config_card: Optional[TrainingConfigPerspectiveCard] = None
        self._policy_sim_card: Optional[PolicySimulationCard] = None
        self._real_robot_card: Optional[RealRobotConnectionCard] = None
        # Mission-panel card-visibility toggle row (3 checkable buttons,
        # left-aligned, fit-content; checked = corresponding card visible).
        self._mission_toggle_row: Optional[QFrame] = None
        self._btn_show_train_config: Optional[QPushButton] = None
        self._btn_show_policy_sim: Optional[QPushButton] = None
        self._btn_show_real_robot: Optional[QPushButton] = None

        self._build_ui()
        self.apply_theme()
        self._render_main_screen()
        # Sync mode visibility on construction (no canvas yet → all
        # mode-specific widgets hidden, splitter visible showing
        # new_file_panel / script standby).
        self._apply_mode()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _restore_mode(self) -> str:
        try:
            raw = Config.get_value(
                "Window", "canvas_view_mode", MODE_MISSION_CONTROL
            )
        except Exception:
            raw = MODE_MISSION_CONTROL
        if str(raw) == MODE_TRAINING_CANVA:
            return MODE_TRAINING_CANVA
        return MODE_MISSION_CONTROL

    def _restore_opacity(self) -> int:
        try:
            raw = Config.get_value(
                "Window",
                "chart_overlay_opacity",
                self._OPACITY_DEFAULT,
                value_type=int,
            )
        except Exception:
            raw = self._OPACITY_DEFAULT
        try:
            v = int(raw)
        except Exception:
            v = self._OPACITY_DEFAULT
        return max(self._OPACITY_MIN, min(self._OPACITY_MAX, v))

    def _save_mode(self, mode: str) -> None:
        try:
            Config.set_value("Window", "canvas_view_mode", mode)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[mission] save mode failed: {exc!r}")

    def _save_opacity(self, pct: int) -> None:
        try:
            Config.set_value("Window", "chart_overlay_opacity", int(pct))
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[mission] save opacity failed: {exc!r}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 1) Top row — opaque, fixed height, always at the top of MC.
        self._build_top_row(root)

        # 2) Splitter — main_screen (transparent, holds chart/script) over
        #    mission_panel (opaque). Hidden in Training Canva mode (the
        #    host then resizes this whole overlay to top_row only).
        self._v_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._v_splitter.setObjectName("missionControlVSplit")
        self._v_splitter.setHandleWidth(self._HANDLE_W)
        self._v_splitter.setChildrenCollapsible(False)

        self._main_screen = QWidget(self._v_splitter)
        self._main_screen.setObjectName("mainScreen")
        self._main_screen.setMinimumHeight(self._MAIN_SCREEN_MIN_H)
        self._build_main_screen(self._main_screen)

        self._mission_panel = QWidget(self._v_splitter)
        self._mission_panel.setObjectName("missionPanel")
        self._mission_panel.setMinimumHeight(self._MISSION_PANEL_MIN_H)
        self._build_mission_panel(self._mission_panel)

        self._v_splitter.addWidget(self._main_screen)
        self._v_splitter.addWidget(self._mission_panel)
        self._v_splitter.setStretchFactor(0, 1)
        self._v_splitter.setStretchFactor(1, 0)
        self._v_splitter.setSizes([10_000, self._MISSION_PANEL_DEFAULT_H])

        root.addWidget(self._v_splitter, 1)

        # Files list — orphan widget; sidebar Project Files panel adopts it.
        self._build_files_list_widget()

    def _build_top_row(self, root: QVBoxLayout) -> None:
        self._top_row_host = QFrame(self)
        self._top_row_host.setObjectName("missionTopRow")
        self._top_row_host.setFrameShape(QFrame.Shape.NoFrame)
        self._top_row_host.setFixedHeight(self._TOP_ROW_H)
        self._top_row_host.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
        top_row = QHBoxLayout(self._top_row_host)
        top_row.setContentsMargins(8, 6, 8, 6)
        top_row.setSpacing(8)

        # MC mode middle: run-source dropdown + opacity-icon + opacity slider.
        self._run_source_selector = RunSourceSelector(
            self._top_row_host,
            project_info_provider=lambda: self._project_info,
        )
        top_row.addWidget(
            self._run_source_selector, 0, Qt.AlignmentFlag.AlignVCenter
        )

        # Spacing between dropdown and opacity affordance.
        top_row.addSpacing(10)

        # icon_opacity prefix — purely decorative QLabel hosting the SVG pixmap.
        # Visibility is gated alongside the slider in ``_apply_mode``.
        self._opacity_icon = QLabel(self._top_row_host)
        self._opacity_icon.setObjectName("opacityIcon")
        self._opacity_icon.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._opacity_icon.setStyleSheet("background: transparent;")
        icon_path = Assets.find_icon("icon_opacity")
        if icon_path is not None:
            side = max(12, self._TOP_ROW_H - 24)
            pm = QPixmap(str(icon_path))
            if not pm.isNull():
                self._opacity_icon.setPixmap(
                    pm.scaled(
                        side, side,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            self._opacity_icon.setFixedSize(side, side)
        top_row.addWidget(
            self._opacity_icon, 0, Qt.AlignmentFlag.AlignVCenter
        )

        self._opacity_slider = QSlider(
            Qt.Orientation.Horizontal, self._top_row_host
        )
        self._opacity_slider.setObjectName("chartOverlayOpacity")
        self._opacity_slider.setRange(self._OPACITY_MIN, self._OPACITY_MAX)
        self._opacity_slider.setValue(self._chart_opacity_pct)
        self._opacity_slider.setFixedWidth(self._OPACITY_SLIDER_W)
        # Explicit tracking=True so valueChanged fires continuously while
        # the user drags the handle — required for real-time chart-opacity
        # follow-through (Qt's default is True but be defensive).
        self._opacity_slider.setTracking(True)
        i18n_bind(
            self._opacity_slider, "setToolTip",
            "missioncontrol.tooltip.chart_opacity", "Chart overlay opacity",
        )
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        top_row.addWidget(
            self._opacity_slider, 0, Qt.AlignmentFlag.AlignVCenter
        )

        # TC mode left cluster — visible only in Training Canva + Canvas tab +
        # canvas loaded. Hosts [Node Library | Undo | Redo | Save]. See
        # ``_apply_mode`` and ``set_canvas`` for wiring.
        self._tc_tools = TCToolButtonsCluster(self._top_row_host)
        top_row.addWidget(
            self._tc_tools, 0, Qt.AlignmentFlag.AlignVCenter
        )

        # Stretch pushes the SliderSwitch to the right edge.
        top_row.addStretch(1)

        # Mode switch — right-aligned, hidden until a canvas file is loaded.
        self._mode_switch = SliderSwitch(
            _MODE_OPTIONS,
            height=30,
            min_segment_width=120,
            parent=self._top_row_host,
        )
        if self._mode == MODE_TRAINING_CANVA:
            self._mode_switch.setCurrentIndex(1, animated=False, emit=False)
        else:
            self._mode_switch.setCurrentIndex(0, animated=False, emit=False)
        self._mode_switch.current_changed.connect(self._on_mode_switch_changed)
        top_row.addWidget(
            self._mode_switch,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        root.addWidget(self._top_row_host, 0)

    def _build_files_list_widget(self) -> None:
        self._files_list = QWidget()
        self._files_list.setObjectName("filesList")
        self._files_list.setMinimumWidth(self._FILES_LIST_MIN_W)
        self._files_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._files_list_layout = QVBoxLayout(self._files_list)
        self._files_list_layout.setContentsMargins(0, 0, 0, 0)
        self._files_list_layout.setSpacing(0)
        self._rebuild_files_table()

    def _build_table_spec(self) -> List[dict]:
        return [
            {
                "id": self._TAB_CANVAS_KEY,
                "default": "Canvas",
                "groups": list(self._canvas_groups),
            },
            {
                "id": self._TAB_SCRIPT_KEY,
                "default": "Script",
                "groups": list(self._script_groups),
            },
        ]

    def _rebuild_files_table(self) -> None:
        if self._files_list_layout is None:
            return
        if self._files_table is not None:
            self._files_list_layout.removeWidget(self._files_table)
            self._files_table.setParent(None)
            self._files_table.deleteLater()
            self._files_table = None

        self._files_table = setLaviTabTable(
            self._build_table_spec(),
            kind="single",
            parent=self._files_list,
        )
        self._files_table.setObjectName("filesListTable")
        self._files_table.setTabEmptyHint(
            self._TAB_CANVAS_KEY,
            "missioncontrol.list.empty.canvas",
            "(no canvas files)",
        )
        self._files_table.tabChanged.connect(self._on_table_tab_changed)
        self._files_table.selectionChanged.connect(self._on_table_selection_changed)
        self._files_list_layout.addWidget(self._files_table, 1)

        active_key = self._TAB_NAME_TO_KEY.get(self.current_tab, self._TAB_CANVAS_KEY)
        self._files_table.setCurrentTab(active_key)
        if self.current_canvas:
            self._files_table.setSelection(
                self._TAB_CANVAS_KEY, [self.current_canvas]
            )
            survivors = self._files_table.selectionMap().get(
                self._TAB_CANVAS_KEY, []
            )
            self.current_canvas = survivors[0] if survivors else None
        if self.current_script:
            self._files_table.setSelection(
                self._TAB_SCRIPT_KEY, [self.current_script]
            )
            survivors = self._files_table.selectionMap().get(
                self._TAB_SCRIPT_KEY, []
            )
            self.current_script = survivors[0] if survivors else None

    def _build_main_screen(self, host: QWidget) -> None:
        # main_screen background is transparent so the chart's opacity
        # reveals the canvas below. Top_row + mission_panel chrome paint
        # opaque via their own stylesheets.
        host.setStyleSheet("QWidget#mainScreen { background: transparent; }")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._content_stack = QStackedWidget(host)

        # Canvas tab page → inner stack [new_file_panel | chart_panel].
        self._canvas_page_widget = QWidget(self._content_stack)
        cp_layout = QVBoxLayout(self._canvas_page_widget)
        cp_layout.setContentsMargins(0, 0, 0, 0)
        cp_layout.setSpacing(0)

        self._canvas_body_stack = QStackedWidget(self._canvas_page_widget)

        # idx 0: placeholder when no canvas selected. This panel is only
        # shown briefly during the no-canvas-loaded transient inside
        # canvas-mode; the user-facing picker (HomepagePage) is a
        # *sibling* of CanvasPage + MissionControlPanel in the host
        # _MainPanel, not a child of this panel — see MainWindow.
        self._new_file_panel = QWidget(self._canvas_body_stack)
        self._new_file_panel.setObjectName("newFilePanel")
        nf_layout = QVBoxLayout(self._new_file_panel)
        nf_layout.setContentsMargins(24, 24, 24, 24)
        nf_placeholder = I18nLabel(
            self._NEW_FILE_PLACEHOLDER_KEY,
            default=self._NEW_FILE_PLACEHOLDER_DEFAULT,
            parent=self._new_file_panel,
        )
        nf_placeholder.setObjectName("newFilePanelPlaceholder")
        nf_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nf_layout.addStretch(1)
        nf_layout.addWidget(nf_placeholder, 0, Qt.AlignmentFlag.AlignCenter)
        nf_layout.addStretch(1)
        self._canvas_body_stack.addWidget(self._new_file_panel)

        # idx 1: chart panel (plain QWidget composite — series checkbox
        # column + QPainter-drawn chart canvas, no QGraphicsView inside).
        # A single QGraphicsOpacityEffect on the panel composes through
        # the whole subtree cleanly, mirroring the DEMO build's overlay
        # opacity slider behavior.
        self._chart_panel = TrainingChartPanel(self._canvas_body_stack)
        self._chart_opacity_effect = QGraphicsOpacityEffect(self._chart_panel)
        self._chart_opacity_effect.setOpacity(1.0)
        self._chart_panel.setGraphicsEffect(self._chart_opacity_effect)
        self._canvas_body_stack.addWidget(self._chart_panel)

        # Wire run-source dropdown → chart panel.
        self._run_source_selector.selected_runs_changed.connect(
            self._chart_panel.set_visible_runs
        )
        self._chart_panel.set_visible_runs(
            self._run_source_selector.selected_run_ids()
        )

        self._apply_chart_opacity(self._chart_opacity_pct)

        cp_layout.addWidget(self._canvas_body_stack, 1)
        # Initial: no canvas → placeholder.
        self._canvas_body_stack.setCurrentIndex(self._CANVAS_BODY_NEW_FILE)

        self._content_stack.addWidget(self._canvas_page_widget)

        # Script tab page (unchanged structurally).
        self._script_page = QWidget(self._content_stack)
        sp_layout = QVBoxLayout(self._script_page)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        sp_layout.setSpacing(0)

        self._script_stack = QStackedWidget(self._script_page)

        standby = QWidget(self._script_stack)
        standby_layout = QVBoxLayout(standby)
        standby_layout.setContentsMargins(24, 24, 24, 24)
        self._script_placeholder = I18nLabel(
            self._SCRIPT_PLACEHOLDER_KEY,
            default=self._SCRIPT_PLACEHOLDER_DEFAULT,
            parent=standby,
        )
        self._script_placeholder.setObjectName("scriptCompilerPlaceholder")
        self._script_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._script_placeholder.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        standby_layout.addStretch(1)
        standby_layout.addWidget(
            self._script_placeholder, 0, Qt.AlignmentFlag.AlignCenter
        )
        standby_layout.addStretch(1)
        self._script_stack.addWidget(standby)

        editor_pane = QWidget(self._script_stack)
        ep_layout = QVBoxLayout(editor_pane)
        ep_layout.setContentsMargins(0, 0, 0, 0)
        ep_layout.setSpacing(0)

        self._script_toolbar = QFrame(editor_pane)
        self._script_toolbar.setObjectName("scriptCompilerToolbar")
        self._script_toolbar.setFrameShape(QFrame.Shape.NoFrame)
        tb_layout = QHBoxLayout(self._script_toolbar)
        tb_layout.setContentsMargins(12, 6, 12, 6)
        tb_layout.setSpacing(8)

        self._script_path_label = QLabel("", self._script_toolbar)
        self._script_path_label.setObjectName("scriptCompilerPath")
        self._script_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        tb_layout.addWidget(self._script_path_label, 0, Qt.AlignmentFlag.AlignVCenter)
        tb_layout.addStretch(1)

        self._script_status = QLabel("", self._script_toolbar)
        self._script_status.setObjectName("scriptCompilerStatus")
        tb_layout.addWidget(self._script_status, 0, Qt.AlignmentFlag.AlignVCenter)

        self._script_save_btn = setButton(
            self._SCRIPT_SAVE_KEY,
            72,
            28,
            kind="normal",
            spec="save",
            default=self._SCRIPT_SAVE_DEFAULT,
        )
        self._script_save_btn.clicked.connect(self._save_script)
        self._script_save_btn.setEnabled(False)
        tb_layout.addWidget(self._script_save_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        ep_layout.addWidget(self._script_toolbar, 0)

        self._script_editor = CodeEditorWidget(editor_pane, mode="python")
        self._script_editor.setObjectName("scriptCompilerEditor")
        self._script_editor.dirtyChanged.connect(self._script_save_btn.setEnabled)
        ep_layout.addWidget(self._script_editor, 1)

        self._script_stack.addWidget(editor_pane)
        sp_layout.addWidget(self._script_stack, 1)
        self._content_stack.addWidget(self._script_page)

        layout.addWidget(self._content_stack, 1)

    def _build_mission_panel(self, host: QWidget) -> None:
        layout = QVBoxLayout(host)
        layout.setContentsMargins(16, 8, 8, 16)
        layout.setSpacing(8)

        # ---- Card-visibility toggle row ---------------------------------
        # Transparent / no border / no background — purely a row of three
        # checkable buttons that hide/show the cards below. Left-aligned
        # via a trailing stretch; each button fits its own text content.
        self._mission_toggle_row = QFrame(host)
        self._mission_toggle_row.setObjectName("missionPanelToggleRow")
        self._mission_toggle_row.setFrameShape(QFrame.Shape.NoFrame)
        self._mission_toggle_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
        toggle_layout = QHBoxLayout(self._mission_toggle_row)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(8)

        def _make_toggle_btn(i18n_key: str, default: str) -> QPushButton:
            btn = QPushButton(self._mission_toggle_row)
            btn.setObjectName("missionPanelToggleBtn")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
            )
            # QPushButton.sizeHint 用 widget 自身 QFont 的度量算宽度，QSS
            # 改 font-weight 不会触发 sizeHint 重算 → 把 QFont 直接设成
            # bold，这样常规态 / 选中态共用 "粗体宽度"，选中加粗时不会
            # 被截。实际是否绘制粗体由下方 QSS 的 font-weight 决定。
            f = btn.font()
            f.setBold(True)
            btn.setFont(f)
            i18n_bind(btn, "setText", i18n_key, default)
            return btn

        self._btn_show_train_config = _make_toggle_btn(
            "missioncontrol.toggle.training_config", "Training Config",
        )
        self._btn_show_policy_sim = _make_toggle_btn(
            "missioncontrol.toggle.simulation", "Simulation",
        )
        self._btn_show_real_robot = _make_toggle_btn(
            "missioncontrol.toggle.ros2_connection", "ROS2 Connection",
        )

        toggle_layout.addWidget(self._btn_show_train_config, 0)
        toggle_layout.addWidget(self._btn_show_policy_sim, 0)
        toggle_layout.addWidget(self._btn_show_real_robot, 0)
        toggle_layout.addStretch(1)

        layout.addWidget(self._mission_toggle_row, 0)

        # ---- Cards row --------------------------------------------------
        cards_row = QWidget(host)
        cards_row.setObjectName("missionPanelCardsRow")
        cards_row.setStyleSheet(
            "QWidget#missionPanelCardsRow { background: transparent; }"
        )
        cards_layout = QHBoxLayout(cards_row)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)
        self._train_config_card = TrainingConfigPerspectiveCard(cards_row)
        self._policy_sim_card = PolicySimulationCard(cards_row)
        self._real_robot_card = RealRobotConnectionCard(cards_row)
        cards_layout.addWidget(self._train_config_card, 2)
        cards_layout.addWidget(self._policy_sim_card, 1)
        cards_layout.addWidget(self._real_robot_card, 1)
        layout.addWidget(cards_row, 1)

        # Wire toggles → card visibility (Qt's layout drops hidden widgets
        # so the surviving cards reclaim space via their stretch factors).
        self._btn_show_train_config.toggled.connect(
            self._train_config_card.setVisible
        )
        self._btn_show_policy_sim.toggled.connect(
            self._policy_sim_card.setVisible
        )
        self._btn_show_real_robot.toggled.connect(
            self._real_robot_card.setVisible
        )

        # Robot Config → Policy Simulation: keep the simulation card's
        # SKU in sync with whichever robot the user has selected in the
        # Robot Config section. Without this wire, Run / Review Robot
        # would always re-resolve the SKU from the policy bundle's
        # manifest (baked at training time), so changing the canvas's
        # robot asset would have no effect on the preview / playback.
        self._train_config_card.robot_sku_changed.connect(
            self._policy_sim_card.set_robot_sku
        )
        # Initial sync — _RobotConfigSection may have already resolved
        # its SKU silently before this wire was hooked up (e.g. if a
        # canvas was bound during construction). Pull it once now.
        self._policy_sim_card.set_robot_sku(
            self._train_config_card.current_robot_sku()
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _on_table_tab_changed(self, tab_key: str) -> None:
        name = self._TAB_KEY_TO_NAME.get(tab_key)
        if not name:
            return
        self.current_tab = name
        self._render_main_screen()
        self.tab_changed.emit(name)
        cur = self._current_id_for_tab()
        self.selection_changed.emit(name, cur or "")
        # Tab change affects mode-conditional visibility (TC compactness
        # only applies on Canvas tab).
        self._apply_mode()

    def _on_table_selection_changed(self, tab_key: str, items: list) -> None:
        name = self._TAB_KEY_TO_NAME.get(tab_key)
        if not name:
            return
        item_id = items[0] if items else ""
        if name == "Canvas":
            self.current_canvas = item_id or None
        else:
            self.current_script = item_id or None
        if self.current_tab == name:
            self._render_main_screen()
            self.selection_changed.emit(name, item_id)

    def _current_id_for_tab(self) -> Optional[str]:
        return (
            self.current_canvas
            if self.current_tab == "Canvas"
            else self.current_script
        )

    @staticmethod
    def _basename_of(file_id: Optional[str]) -> str:
        if not file_id:
            return ""
        name = file_id.replace("\\", "/").rsplit("/", 1)[-1]
        for suffix in (".canvas.json", ".py"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _render_main_screen(self) -> None:
        if self._content_stack is None:
            return
        if self.current_tab == "Canvas":
            self._content_stack.setCurrentIndex(0)
            self._sync_canvas_view()
        else:
            self._content_stack.setCurrentIndex(1)
            self._sync_script_view()

    # ------------------------------------------------------------------
    # Canvas auto-load
    # ------------------------------------------------------------------
    def _sync_canvas_view(self) -> None:
        if self._canvas_body_stack is None:
            return
        if self._project_info is None or not self.current_canvas:
            self._canvas_body_stack.setCurrentIndex(self._CANVAS_BODY_NEW_FILE)
            if self._canvas_loaded_id is not None:
                self._canvas_loaded_id = None
                if self._external_canvas is not None:
                    self._external_canvas.clear_scene()
                if self._train_config_card is not None:
                    try:
                        self._train_config_card.set_canvas(None, None)
                    except Exception:
                        pass
            self._apply_mode()
            return
        if self.current_canvas != self._canvas_loaded_id:
            self._load_canvas_into_external()
        if self._canvas_loaded_id == self.current_canvas:
            self._canvas_body_stack.setCurrentIndex(self._CANVAS_BODY_CHART)
        # Reapply mode visibility now that the canvas-loaded state may
        # have flipped.
        self._apply_mode()

    def _load_canvas_into_external(self) -> None:
        """Resolve current_canvas → abs path, load into the external CanvasPage.

        On any failure (unresolved id, malformed file, backend mismatch),
        log + revert the body stack to the new-file placeholder.
        """
        if (
            self._external_canvas is None
            or self._project_info is None
            or not self.current_canvas
        ):
            return
        file_id = self.current_canvas
        try:
            abs_path = resolve_file(self._project_info, file_id)
        except ValueError as exc:
            log_warning(f"[mission] cannot resolve canvas {file_id!r}: {exc}")
            self._canvas_loaded_id = None
            if self._canvas_body_stack is not None:
                self._canvas_body_stack.setCurrentIndex(self._CANVAS_BODY_NEW_FILE)
            if self._train_config_card is not None:
                try:
                    self._train_config_card.set_canvas(None, None)
                except Exception:
                    pass
            return

        try:
            self._external_canvas.clear_scene()
            n = self._external_canvas.load_from_file(abs_path)
        except (ValueError, RuntimeError) as exc:
            log_warning(f"[mission] load canvas failed {abs_path}: {exc}")
            self._canvas_loaded_id = None
            if self._canvas_body_stack is not None:
                self._canvas_body_stack.setCurrentIndex(self._CANVAS_BODY_NEW_FILE)
            if self._train_config_card is not None:
                try:
                    self._train_config_card.set_canvas(None, None)
                except Exception:
                    pass
            return

        self._canvas_loaded_id = file_id
        log_debug(f"[mission] external canvas loaded: {abs_path} ({n} nodes)")
        # 把 (file_id, project, save_name) 写到 page，供 set_node_param 自动落盘 +
        # canvas_param_changed emit 携带正确 file_id。save_name 形如
        # "sb3_mujoco/main"（即 file_id 去掉 "canvas/" 前缀 + ".canvas.json" 后缀）。
        try:
            self._external_canvas.set_file_id(file_id)
            save_name = self._derive_save_name(file_id)
            if save_name:
                self._external_canvas.set_save_target(self._project_info, save_name)
        except Exception as exc:
            log_warning(f"[mission] canvas save_target wire failed: {exc!r}")
        # 通知左卡（训练配置透视）切换到这块画布
        if self._train_config_card is not None:
            try:
                self._train_config_card.set_canvas(self._external_canvas, file_id)
            except Exception as exc:
                log_warning(f"[mission] train_config_card.set_canvas failed: {exc!r}")
        # First canvas-load per session forces Mission Control mode so the
        # chart overlay covers any flash while the canvas builds.
        self.notify_canvas_loaded()
        # Notify external subscribers (MainWindow → user.ini persistence,
        # ...) that a canvas has fully loaded. Emit only on the success
        # path so failures don't blow away last_canvas state.
        self.canvas_loaded.emit(file_id)

    @staticmethod
    def _derive_save_name(file_id: str) -> str:
        """从 canvas file_id 推导 save_to_project 需要的 ``<subdir>/<stem>`` 名。

        e.g. ``"canvas/sb3_mujoco/main.canvas.json"`` → ``"sb3_mujoco/main"``。
        非 canvas/ 前缀或无 .canvas.json 后缀返回空串。
        """
        s = (file_id or "").replace("\\", "/")
        if s.startswith("canvas/"):
            s = s[len("canvas/"):]
        if s.endswith(".canvas.json"):
            s = s[: -len(".canvas.json")]
        return s

    # ------------------------------------------------------------------
    # Mode apply (master state applicator)
    # ------------------------------------------------------------------
    def _effective_mode(self) -> str:
        """The mode actually in effect right now.

        Equal to :attr:`_mode` when a canvas is loaded; otherwise forced
        to Mission Control (TC mode is meaningless without a canvas, and
        the SliderSwitch is hidden in that state anyway).
        """
        if self._canvas_loaded_id is None:
            return MODE_MISSION_CONTROL
        return self._mode

    def _on_mode_switch_changed(self, _index: int, key: str) -> None:
        new_mode = (
            MODE_TRAINING_CANVA if key == MODE_TRAINING_CANVA else MODE_MISSION_CONTROL
        )
        if new_mode == self._mode:
            return
        self._mode = new_mode
        self._save_mode(new_mode)
        self._apply_mode()
        self.mode_changed.emit(new_mode)

    def _apply_mode(self) -> None:
        """Apply visibility rules from current (mode, tab, canvas-loaded).

        Drives:
          - top_row mode-conditional widgets (run dropdown, opacity slider,
            node lib button) per effective mode;
          - SliderSwitch visibility (canvas-loaded gate);
          - splitter visibility (TC + canvas-loaded + Canvas tab → hidden);
          - external canvas minimap + interactive flag;
          - emits ``overlay_compact_changed(True/False)`` only on actual flips
            so the host can resize the panel between full-cover and top_row
            strip.
        """
        eff_mode = self._effective_mode()
        is_mc = (eff_mode == MODE_MISSION_CONTROL)
        on_canvas_tab = (self.current_tab == "Canvas")
        canvas_loaded = self._canvas_loaded_id is not None

        # SliderSwitch — canvas-loaded gate. Hide on Script tab too: mode
        # toggling has no visible effect there.
        if self._mode_switch is not None:
            self._mode_switch.setVisible(canvas_loaded and on_canvas_tab)

        # Top row middle widgets.
        # RunSourceSelector + opacity slider → MC mode (and only meaningful
        # when chart can show: Canvas tab + canvas loaded).
        show_mc_middle = is_mc and on_canvas_tab and canvas_loaded
        if self._run_source_selector is not None:
            self._run_source_selector.setVisible(show_mc_middle)
        if self._opacity_icon is not None:
            self._opacity_icon.setVisible(show_mc_middle)
        if self._opacity_slider is not None:
            self._opacity_slider.setVisible(show_mc_middle)

        # TC tools cluster — opposite gate: only in Training Canva + Canvas
        # tab + canvas loaded. Mutually exclusive with the MC middle widgets.
        show_tc_tools = (not is_mc) and on_canvas_tab and canvas_loaded
        if self._tc_tools is not None:
            self._tc_tools.setVisible(show_tc_tools)

        # Splitter (main_screen + mission_panel) — visible only when a canvas
        # is loaded; hidden in Training Canva + canvas-loaded + Canvas tab
        # (compact mode covers full screen). Without a canvas the bottom
        # mission_panel has nothing meaningful to show, so the entire
        # splitter collapses. (Note: the empty-state picker is *not* hosted
        # here — it's a sibling of canvas+mission_control in the host
        # _MainPanel, mutually exclusive with the canvas mode.)
        compact = (not is_mc) and canvas_loaded and on_canvas_tab
        splitter_visible = canvas_loaded and not compact
        if self._v_splitter is not None:
            self._v_splitter.setVisible(splitter_visible)

        # External canvas state — non-interactive when MC mode
        # (chart covers it); interactive only in TC + Canvas tab + loaded.
        # Node Library 已迁移到 Sidebar 弹出面板：可见性由
        # ``Sidebar.set_node_library_visible`` 控制（MainWindow 桥接
        # ``tab_changed`` → sidebar），这里不再触碰 canvas overlay。
        if self._external_canvas is not None:
            in_tc_canvas = (not is_mc) and on_canvas_tab and canvas_loaded
            self._external_canvas.set_interactive(in_tc_canvas)
            self._external_canvas.set_minimap_visible(in_tc_canvas)

        # Notify host (MainWindow) of compact-state flips only.
        if compact != self._is_compact_state:
            self._is_compact_state = compact
            self.overlay_compact_changed.emit(compact)

    def _on_opacity_slider_changed(self, value: int) -> None:
        v = max(self._OPACITY_MIN, min(self._OPACITY_MAX, int(value)))
        self._chart_opacity_pct = v
        self._apply_chart_opacity(v)
        self._save_opacity(v)

    def _apply_chart_opacity(self, pct: int) -> None:
        if self._chart_opacity_effect is None:
            return
        opacity = max(0.0, min(1.0, float(pct) / 100.0))
        self._chart_opacity_effect.setOpacity(opacity)

    # ------------------------------------------------------------------
    # Public API consumed by MainWindow
    # ------------------------------------------------------------------
    def set_canvas(self, canvas: CanvasPage) -> None:
        """Wire the external CanvasPage that this overlay sits above.

        Should be called by the host immediately after constructing both
        widgets, before any project is bound.
        """
        self._external_canvas = canvas
        # Forward to the TC tools cluster so it can wire undo/redo + dirty
        # signals to this canvas. Safe to call with same canvas (cluster
        # short-circuits).
        if self._tc_tools is not None:
            self._tc_tools.set_canvas(canvas)
        # Apply current state to the freshly-wired canvas (interactive /
        # minimap / library card visibility).
        self._apply_mode()

    def bind_run_buttons(self, start_btn, stop_btn) -> None:
        """转发 MainWindow 顶部 [开始/停止] 按钮给左卡，做状态 + 行为联动。"""
        if self._train_config_card is not None:
            try:
                self._train_config_card.bind_run_buttons(start_btn, stop_btn)
            except Exception as exc:
                log_warning(f"[mission] bind_run_buttons failed: {exc!r}")

    def bind_link_combo(self, combo) -> None:
        """转发 MainWindow 顶部 [Local/Cloud] combo 给左卡，做双向同步。"""
        if self._train_config_card is not None:
            try:
                self._train_config_card.bind_link_combo(combo)
            except Exception as exc:
                log_warning(f"[mission] bind_link_combo failed: {exc!r}")

    def top_row_height(self) -> int:
        """Used by the host to size this overlay in compact (TC) mode."""
        return self._TOP_ROW_H

    def is_overlay_compact(self) -> bool:
        return self._is_compact_state

    @property
    def current_canvas_file_id(self) -> str:
        return self._canvas_loaded_id or ""

    def current_view_mode(self) -> str:
        return self._mode

    def notify_canvas_loaded(self) -> None:
        """First canvas-load per session: force Training Canva mode.

        UX spec: user lands directly in the editing view (TC mode →
        canvas exposed). The Node Library card is permanently mounted
        and shown by ``_apply_mode`` whenever the canvas is interactive,
        so no explicit open call is needed here.
        """
        if self._first_load_force_mc_done:
            return
        self._first_load_force_mc_done = True
        if self._mode != MODE_TRAINING_CANVA:
            self._mode = MODE_TRAINING_CANVA
            self._save_mode(MODE_TRAINING_CANVA)
            if self._mode_switch is not None:
                self._mode_switch.setCurrentIndex(1, animated=False, emit=False)
            self.mode_changed.emit(MODE_TRAINING_CANVA)

    def notify_project_loaded(self) -> None:
        """Reset the per-session force-MC flag so the next canvas re-arms."""
        self._first_load_force_mc_done = False
        self._render_main_screen()

    def enter_mission_control_mode(self) -> None:
        """Programmatically flip into Mission Control mode.

        Called when Training is started: the chart + bottom mission_panel
        should be visible so the user can watch progress. No-op when
        already in MC mode or when no canvas is loaded (TC mode is
        meaningless without a canvas and the switch is hidden anyway).
        """
        if self._canvas_loaded_id is None:
            return
        if self._mode == MODE_MISSION_CONTROL:
            self._apply_mode()
            return
        self._mode = MODE_MISSION_CONTROL
        self._save_mode(MODE_MISSION_CONTROL)
        if self._mode_switch is not None:
            self._mode_switch.setCurrentIndex(0, animated=False, emit=False)
        self._apply_mode()
        self.mode_changed.emit(MODE_MISSION_CONTROL)

    def unbind_canvas(self) -> None:
        """完全解除当前画布绑定 / Fully detach from the currently loaded canvas.

        Used by the main_row "New" button so that flipping to the picker
        leaves no dangling references to the previously open canvas (file_id,
        save_target, files_table selection, training-config card binding,
        and the loaded scene itself).
        """
        # Drop the files_table selection (both tabs to be safe) so the
        # sidebar list no longer highlights the closed canvas.
        if self._files_table is not None:
            try:
                self._files_table.clearSelection()
            except Exception:
                pass
        self.current_canvas = None
        self.current_script = None
        self._canvas_loaded_id = None
        self._script_loaded_id = None
        self._script_target_path = None
        # Wipe the embedded canvas so the next bind starts from a blank slate.
        if self._external_canvas is not None:
            try:
                self._external_canvas.clear_scene()
                self._external_canvas.set_file_id("")
                self._external_canvas.set_save_target(None, "")
            except Exception as exc:                                # noqa: BLE001
                log_warning(f"[mission] unbind_canvas clear failed: {exc!r}")
        if self._train_config_card is not None:
            try:
                self._train_config_card.set_canvas(None, None)
            except Exception:
                pass
        # Reset the per-session "first canvas-load forces MC" gate so the
        # next canvas opened after the picker again forces TC mode.
        self._first_load_force_mc_done = False
        # Refresh the visible body (placeholder) + mode visibility.
        if self._canvas_body_stack is not None:
            self._canvas_body_stack.setCurrentIndex(self._CANVAS_BODY_NEW_FILE)
        self._apply_mode()
        # canvas 卸载会让 _effective_mode 回到 mission_control（TC 在无 canvas
        # 时不成立）。mode_changed 信号原本只覆盖用户主动切换的场景，这里补发
        # 一次让 Sidebar 在卸载瞬间也重排导航按钮。
        self.mode_changed.emit(self._effective_mode())

    def open_canvas(self, file_id: str) -> None:
        """Programmatically open ``file_id`` as if the user picked the row.

        Drives the sidebar files list (kept by reference even after
        :meth:`take_files_list_widget` reparents it) to the Canvas tab and
        selects the row, then lets ``_on_table_selection_changed`` trigger
        the existing auto-load path which terminates in
        ``canvas_loaded.emit(file_id)`` (used by MainWindow to persist the
        last-opened (project, canvas) pair).

        Falls back to driving panel state directly if ``_files_list`` is
        not yet built — supports the very early startup auto-open case.
        """
        if not file_id:
            return
        # Make sure the panel itself is on the Canvas tab so the
        # selection-changed handler routes through canvas auto-load.
        self.current_tab = "Canvas"
        # _files_list is a wrapping QWidget; the actual LaviTabTable lives
        # at _files_table (which keeps signal-slot wiring across the sidebar
        # reparent done by take_files_list_widget).
        if self._files_table is not None:
            self._files_table.setCurrentTab(self._TAB_CANVAS_KEY)
            self._files_table.setSelection(self._TAB_CANVAS_KEY, [file_id])
            return
        # Fallback path — drive state directly.
        self.current_canvas = file_id
        self._render_main_screen()
        self.selection_changed.emit("Canvas", file_id)

# ------------------------------------------------------------------
    # Script tab plumbing (unchanged)
    # ------------------------------------------------------------------
    def _sync_script_view(self) -> None:
        if self._script_stack is None:
            return
        if self._project_info is None or not self.current_script:
            self._script_stack.setCurrentIndex(0)
            self._script_loaded_id = None
            return
        if self.current_script != self._script_loaded_id:
            self._load_current_script_into_editor()
        self._script_stack.setCurrentIndex(1)

    def _load_current_script_into_editor(self) -> None:
        if self._script_editor is None or self._project_info is None:
            return
        file_id = self.current_script or ""
        if self._script_editor.is_modified():
            log_warning(
                f"[mission] script switch discards unsaved edits in "
                f"{self._script_loaded_id!r}"
            )
        try:
            path = resolve_file(self._project_info, file_id)
        except ValueError as exc:
            self._set_script_status(f"unresolved: {exc}", error=True)
            self._script_loaded_id = None
            self._script_target_path = None
            return
        if self._script_path_label is not None:
            self._script_path_label.setText(path.name)
        self._script_target_path = path
        if not path.exists():
            self._script_editor.set_text("")
            self._set_script_status(f"missing on disk: {path.name}", error=True)
            self._script_loaded_id = file_id
            return
        ok = self._script_editor.load_file(path, mode="python")
        if ok:
            self._script_loaded_id = file_id
            self._set_script_status("loaded", error=False)
        else:
            self._script_loaded_id = None
            self._set_script_status("load failed", error=True)

    def _save_script(self) -> None:
        if self._script_editor is None:
            return
        target = self._script_target_path
        if target is None:
            self._set_script_status("no path", error=True)
            return
        ok = self._script_editor.save_file(target)
        self._set_script_status("saved" if ok else "save failed", error=not ok)

    def _set_script_status(self, text: str, *, error: bool = False) -> None:
        if self._script_status is None:
            return
        self._script_status.setText(text)
        color_slot = "danger_zone" if error else "sub_t2"
        fallback = "#FF6B6B" if error else "#777777"
        color = Config.get_color(color_slot, fallback)
        font_small = Config.get_font_size("size_small", 12)
        self._script_status.setStyleSheet(
            f"color: {color}; font-size: {font_small}px; background: transparent;"
        )

    # ------------------------------------------------------------------
    # Public API for project binding
    # ------------------------------------------------------------------
    def set_project(self, info: Optional[ProjectInfo]) -> None:
        self._project_info = info
        self._canvas_loaded_id = None
        if self._external_canvas is not None:
            self._external_canvas.clear_scene()
        if info is None:
            self._script_loaded_id = None
            self._script_target_path = None
            if self._script_editor is not None:
                self._script_editor.set_text("")
            if self._script_path_label is not None:
                self._script_path_label.setText("")
            if self._script_status is not None:
                self._script_status.setText("")
        if self.current_tab == "Script":
            self._render_main_screen()
        # 解除 project 绑定时 _canvas_loaded_id 已置 None，effective_mode 会
        # 强制回到 mission_control；补发 mode_changed 让 Sidebar 重排——否则
        # 登出/账号切换走到这里时 Sidebar 仍然停留在 training_canva 样式。
        if info is None:
            self.mode_changed.emit(self._effective_mode())

    def set_canvas_groups(self, groups: List[Dict]) -> None:
        self._canvas_groups = list(groups)
        self._rebuild_files_table()
        if self.current_tab == "Canvas":
            self._render_main_screen()

    def set_script_groups(self, groups: List[Dict]) -> None:
        self._script_groups = list(groups)
        self._rebuild_files_table()
        if self.current_tab == "Script":
            self._render_main_screen()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def files_list(self) -> QWidget:
        assert self._files_list is not None
        return self._files_list

    def take_files_list_widget(self) -> QWidget:
        assert self._files_list is not None
        return self._files_list

    @property
    def main_screen(self) -> QWidget:
        assert self._main_screen is not None
        return self._main_screen

    @property
    def mission_panel(self) -> QWidget:
        assert self._mission_panel is not None
        return self._mission_panel

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        bg_main = Config.get_color("bg_1", "#1E1E1E")
        bg_alt = Config.get_color("bg_2", "#1A1A1A")
        border = Config.get_color("border_1", "#444444")
        handle_hover = Config.get_color("hover_1", "#525252")
        sub = Config.get_color("sub_t2", "#777777")
        font_normal = Config.get_font_size("size_normal", 14)
        font_small = Config.get_font_size("size_small", 12)
        slider_track = Config.get_color("opacity_slider_track", "#2A2C33")
        slider_handle = Config.get_color("highlight", "#F6D393")
        # Toggle row buttons (mission-panel card visibility).
        toggle_bg = Config.get_color("btn_1")
        toggle_fg = Config.get_color("main_t1")
        toggle_hover = Config.get_color("hover_1")
        
        # toggle_checked_bg = Config.get_color("highlight")
        toggle_checked_bg = Config.get_color("safe_zone")
        
        toggle_checked_fg = Config.get_color("bg_1")
        toggle_family = Config.get_value("Font", "family", "Microsoft YaHei")

        # Note: ``QWidget#missionControlPanel`` keeps the transparent
        # background set in __init__; apply_theme styles only opaque
        # children (top_row, mission_panel) and the script chrome.
        self.setStyleSheet(
            f"QWidget#missionControlPanel {{ background: transparent; }}"
            f"QFrame#missionTopRow {{ background-color: {bg_main}; "
            f"border-bottom: 1px solid {border}; }}"
            f"QWidget#mainScreen {{ background: transparent; }}"
            f"QWidget#missionPanel {{ background-color: {bg_alt}; "
            f"border-top: 1px solid {border}; }}"
            f"QLabel#scriptCompilerPlaceholder, QLabel#newFilePanelPlaceholder {{ "
            f"color: {sub}; font-size: {font_normal}px; background: transparent; }}"
            f"QFrame#scriptCompilerToolbar {{ background-color: {bg_alt}; "
            f"border-bottom: 1px solid {border}; }}"
            f"QLabel#scriptCompilerPath {{ color: {sub}; "
            f"font-family: Consolas, 'Courier New', monospace; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#scriptCompilerStatus {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QSplitter::handle {{ background-color: {border}; }}"
            f"QSplitter::handle:hover {{ background-color: {handle_hover}; }}"
            f"QWidget#newFilePanel {{ background-color: {bg_main}; }}"
            f"QSlider#chartOverlayOpacity::groove:horizontal {{ "
            f"background: {slider_track}; height: 4px; border-radius: 2px; }}"
            f"QSlider#chartOverlayOpacity::handle:horizontal {{ "
            f"background: {slider_handle}; width: 12px; height: 12px; "
            f"margin: -4px 0; border-radius: 6px; }}"
            f"QSlider#chartOverlayOpacity::sub-page:horizontal {{ "
            f"background: {slider_handle}; height: 4px; border-radius: 2px; }}"
            f"QFrame#missionPanelToggleRow {{ background: transparent; "
            f"border: none; }}"
            f"QPushButton#missionPanelToggleBtn {{ "
            f"background-color: {toggle_bg}; color: {toggle_fg}; "
            f"border: none; border-radius: 6px; padding: 0; "
            f"padding: 4px 12px; font-family: \"{toggle_family}\"; "
            f"font-size: {font_small}px; font-weight: bold; }}"
            f"QPushButton#missionPanelToggleBtn:hover:!checked {{ "
            f"background-color: {toggle_hover}; color: {toggle_fg}; }}"
            f"QPushButton#missionPanelToggleBtn:checked {{ "
            f"background-color: {toggle_checked_bg}; "
            f"color: {toggle_checked_fg}; font-weight: bold; }}"
        )
        if self._files_list is not None:
            # No backing fill — the host (MissionControlPanel left column or
            # ProjectsPanel files host) provides its own background, and the
            # inner LaviTabTable owns its own theming.
            self._files_list.setStyleSheet(
                "QWidget#filesList { background: transparent; }"
            )
        if self._files_table is not None:
            self._files_table.refresh_style()
        if self._script_editor is not None:
            self._script_editor.refresh_style()
        if self._mode_switch is not None:
            self._mode_switch.refresh_style()
        if self._run_source_selector is not None:
            self._run_source_selector.apply_theme()
        if self._chart_panel is not None:
            self._chart_panel.apply_theme()
        if self._train_config_card is not None:
            self._train_config_card.apply_theme()
        if self._policy_sim_card is not None:
            self._policy_sim_card.apply_theme()
        if self._real_robot_card is not None:
            self._real_robot_card.apply_theme()
        if self._tc_tools is not None:
            self._tc_tools.apply_theme()


__all__ = ["MissionControlPanel"]
