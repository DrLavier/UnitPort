# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MissionControlPanel -- top-level main work area shell.

This panel is an **overlay** raised above an external CanvasPage. The host
(``MainWindow.main_panel``) parents the canvas at the bottom layer
(QVBoxLayout-managed, fills the panel) and parents this MissionControlPanel
as a free-floating sibling raised on top. The host calls
:meth:`set_canvas` to give this panel a reference to the canvas it
overlays, then watches :attr:`overlay_compact_changed` to resize this
panel between **full-cover** and **top-row-only** states.

Three view modes (3-way SliderSwitch on the top-right, gated by
``canvas_loaded``):

    Mission Control  → full-cover; chart visible at user-set opacity
                       over the (hidden) canvas, cards in mission_panel
    Training Canva   → compact (top_row strip only); canvas fully
                       exposed below for editing
    Scripts          → full-cover; built-in script compiler fills
                       main_screen, mission_panel hidden, undo/redo/save
                       cluster in the top_row (mirrors TC tool cluster)

Layout::

    +------------------------------ top_row (always visible, opaque) ----+
    | <MC: run dropdown | opacity slider> | <TC: undo/redo/save canvas> |
    | <Scripts: undo/redo/save script>                  <slider switch> |
    +-------------- vertical splitter (drag, cursor) -- hidden in TC ----+
    |   main_screen (transparent — chart applies opacity in MC; editor   |
    |                fills full height in Scripts mode)                  |
    +-------------- vertical split ---------------------------------------+
    |   mission_panel (cards row; hidden in Scripts)                      |
    +---------------------------------------------------------------------+

The Project Files sidebar panel and the four Scripts-mode sidebar panels
(Training / Rewards / Termins / Observs) own their own data widgets now —
this panel only listens for ``set_project`` / ``open_canvas`` /
``load_script`` from MainWindow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
    Paths,
    SliderSwitch,
    i18n_bind,
    log_debug,
    log_warning,
    setButton,
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
from application.ui.widgets.scripts_tool_buttons import ScriptsToolbarCluster
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
MODE_SCRIPTS = "scripts"

# The SliderSwitch options carry (i18n_key, default) pairs so the SDK
# re-translates each segment on ``language_changed``. The mode-constant
# string is recovered from the index via ``_MODE_KEYS``; the emitted
# ``key`` field of ``current_changed`` is the i18n key, not the mode.
_MODE_OPTIONS = [
    ("missioncontrol.mode.mission_control", "Mission Control"),
    ("missioncontrol.mode.training_canva",  "Training Canva"),
    ("missioncontrol.mode.scripts",         "Scripts"),
]
_MODE_KEYS = [MODE_MISSION_CONTROL, MODE_TRAINING_CANVA, MODE_SCRIPTS]

_VALID_MODES = frozenset({MODE_MISSION_CONTROL, MODE_TRAINING_CANVA, MODE_SCRIPTS})


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
    mode_changed = pyqtSignal(str)             # one of _VALID_MODES
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

    _MISSION_PANEL_DEFAULT_H = 530
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

    # Inner Canvas-body stack indices.
    _CANVAS_BODY_NEW_FILE = 0
    _CANVAS_BODY_CHART = 1

    # Outer content_stack indices.
    _CONTENT_CANVAS = 0
    _CONTENT_SCRIPT = 1

    _SCRIPT_PLACEHOLDER_KEY = "missioncontrol.script.placeholder"
    _SCRIPT_PLACEHOLDER_DEFAULT = "Select a script file to edit"
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
        self.current_canvas: Optional[str] = None
        # ``current_script`` carries the file_id form when loaded from
        # disk (``project:<rel>`` / ``system:<rel>``) or a virtual id
        # (``registry:<kind>:<key>``) when loaded from a TaskModuleItem.
        self.current_script: Optional[str] = None

        # ---- project context ----------------------------------------
        self._project_info: Optional[ProjectInfo] = None
        self._script_loaded_id: Optional[str] = None
        self._script_target_path: Optional[Path] = None
        # True while the loaded script came from a registry virtual id
        # (``registry:<kind>:<key>[:<variant>]``) — _save_script then
        # routes through application.service.scripts.resolver for
        # variant writes (preset writes are rejected).
        self._script_is_virtual: bool = False
        # When ``_script_is_virtual`` is True these record the resolver
        # tuple. ``_script_variant=None`` means the editor is showing a
        # preset (read-only conceptually); a non-None value names the
        # user variant that Save will write back into.
        self._script_kind: Optional[str] = None
        self._script_key: Optional[str] = None
        self._script_variant: Optional[str] = None
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
        # Reentry guard for _apply_mode. ``_sync_canvas_view`` calls
        # ``_apply_mode`` from inside its own body when the canvas-loaded
        # state flips mid-sync, and ``_apply_mode`` itself calls
        # ``_sync_canvas_view`` — without this flag the two recurse.
        self._applying_mode: bool = False
        # Chart overlay opacity (percent in [_OPACITY_MIN, _OPACITY_MAX]).
        self._chart_opacity_pct: int = self._restore_opacity()

        # ---- widget slots ------------------------------------------
        self._top_row_host: Optional[QFrame] = None
        self._mode_switch: Optional[SliderSwitch] = None
        self._run_source_selector: Optional[RunSourceSelector] = None
        self._opacity_icon: Optional[QLabel] = None
        self._opacity_slider: Optional[QSlider] = None
        self._tc_tools: Optional[TCToolButtonsCluster] = None
        self._scripts_tools: Optional[ScriptsToolbarCluster] = None

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
        # Sync mode visibility on construction. With no canvas yet,
        # _effective_mode falls back to MC, mode_switch + tools clusters
        # are all hidden, and the splitter shows the new_file placeholder.
        # _apply_mode also drives the content_stack to the right page and
        # invokes _sync_canvas_view / _sync_script_view.
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
        s = str(raw)
        return s if s in _VALID_MODES else MODE_MISSION_CONTROL

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

        # Wire the scripts toolbar cluster to the editor now that both
        # halves exist (the cluster is built inside _build_top_row before
        # _build_main_screen creates the editor).
        if self._scripts_tools is not None and self._script_editor is not None:
            self._scripts_tools.bind_editor(self._script_editor)
            self._scripts_tools.save_requested.connect(self._save_script)

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

        # TC mode left cluster — visible only in Training Canva + canvas loaded.
        # Hosts [Undo | Redo | Save]. See ``_apply_mode`` and ``set_canvas``
        # for wiring.
        self._tc_tools = TCToolButtonsCluster(self._top_row_host)
        top_row.addWidget(
            self._tc_tools, 0, Qt.AlignmentFlag.AlignVCenter
        )

        # Scripts mode left cluster — same layout as TC tools but bound to
        # the script CodeEditorWidget. Editor binding is deferred to
        # _build_ui()'s tail once _build_main_screen has constructed the
        # editor. Mutually exclusive with both _tc_tools and the MC middle
        # widgets above.
        self._scripts_tools = ScriptsToolbarCluster(self._top_row_host)
        top_row.addWidget(
            self._scripts_tools, 0, Qt.AlignmentFlag.AlignVCenter
        )

        # Stretch pushes the SliderSwitch to the right edge.
        top_row.addStretch(1)

        # Mode switch — right-aligned, hidden until a canvas file is loaded.
        # 3-way (Mission Control | Training Canva | Scripts). min_segment_width
        # dropped from 120→92 so three segments still fit the top row at
        # reasonable window widths; drop further if clipping shows up.
        self._mode_switch = SliderSwitch(
            _MODE_OPTIONS,
            height=30,
            min_segment_width=92,
            parent=self._top_row_host,
        )
        try:
            initial_index = _MODE_KEYS.index(self._mode)
        except ValueError:
            initial_index = 0
        self._mode_switch.setCurrentIndex(initial_index, animated=False, emit=False)
        self._mode_switch.current_changed.connect(self._on_mode_switch_changed)
        top_row.addWidget(
            self._mode_switch,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        root.addWidget(self._top_row_host, 0)

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

        # Script page — opaque so it fully covers the canvas underneath
        # (main_screen is transparent for MC mode's chart-opacity effect,
        # so we have to paint our own fill here).
        self._script_page = QWidget(self._content_stack)
        self._script_page.setObjectName("scriptPage")
        self._script_page.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
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

        # Thin info row above the editor: file path on the left, status
        # (loaded / saved / error) on the right. Save/Undo/Redo live in
        # the top-row ScriptsToolbarCluster now, NOT here.
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

        ep_layout.addWidget(self._script_toolbar, 0)

        self._script_editor = CodeEditorWidget(editor_pane, mode="python")
        self._script_editor.setObjectName("scriptCompilerEditor")
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
        # ROS2 connection is a development-preview feature: ship the toggle
        # off by default and warn the user on first manual enable.
        self._btn_show_real_robot.setChecked(False)
        self._ros2_dev_notice_shown = False

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
        # Keep the card hidden until the user opts in via the toggle above.
        self._real_robot_card.setVisible(False)
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
        self._btn_show_real_robot.toggled.connect(
            self._maybe_show_ros2_dev_notice
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

    def _maybe_show_ros2_dev_notice(self, checked: bool) -> None:
        # Fire once, only on a positive (user-initiated) toggle. The notice
        # is informational and never blocks the toggle from taking effect.
        if not checked or self._ros2_dev_notice_shown:
            return
        self._ros2_dev_notice_shown = True
        QMessageBox.information(
            self,
            tr(
                "missioncontrol.ros2_dev_notice.title",
                "ROS2 Connection (Preview)",
            ),
            tr(
                "missioncontrol.ros2_dev_notice.body",
                "ROS2 connection is still under development. Basic "
                "connection and control are available; the full feature "
                "set will land in a later release.",
            ),
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    @staticmethod
    def _basename_of(file_id: Optional[str]) -> str:
        if not file_id:
            return ""
        name = file_id.replace("\\", "/").rsplit("/", 1)[-1]
        for suffix in (".canvas.json", ".py"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

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

    def cycle_mode(self, direction: int) -> bool:
        """Advance the mode SliderSwitch by ``direction`` (+1 / -1) with wrap.

        Bound to PageUp (prev, -1) / PageDown (next, +1) at the MainWindow
        level. No-ops when the switch is hidden (canvas not loaded) or the
        panel is mid-apply; returns False so callers can fall through to
        the next handler if needed.
        """
        sw = self._mode_switch
        if sw is None or not sw.isVisible():
            return False
        opts = [m for m, _ in _MODE_OPTIONS]
        if not opts:
            return False
        cur = sw.currentIndex()
        if cur < 0:
            cur = 0
        new_idx = (cur + int(direction)) % len(opts)
        if new_idx == cur:
            return False
        sw.setCurrentIndex(new_idx, animated=True, emit=True)
        return True

    def _on_mode_switch_changed(self, index: int, _key: str) -> None:
        # The SliderSwitch's emitted ``_key`` is the i18n key
        # (``missioncontrol.mode.*``) since the segments are i18n-bound.
        # Recover the mode constant by index lookup.
        if 0 <= index < len(_MODE_KEYS):
            new_mode = _MODE_KEYS[index]
        else:
            new_mode = MODE_MISSION_CONTROL
        if new_mode == self._mode:
            return
        self._mode = new_mode
        self._save_mode(new_mode)
        self._apply_mode()
        self.mode_changed.emit(new_mode)

    def _apply_mode(self) -> None:
        """Apply visibility rules from current (mode, canvas-loaded).

        Single source of truth for the 3-mode UI. Drives:
          - top_row mode-conditional clusters (run-dropdown + opacity =
            MC; tc_tools = TC; scripts_tools = Scripts);
          - SliderSwitch visibility (canvas-loaded gate; no tab gate);
          - outer content_stack (Canvas page idx 0 vs Scripts page idx 1);
          - splitter visibility (visible in MC + Scripts; hidden in TC);
          - mission_panel visibility (visible only in MC + canvas loaded);
          - external canvas interactivity + minimap (only TC);
          - emits ``overlay_compact_changed`` on actual flips so the host
            resizes the panel between full-cover and top_row strip.

        Loops at most twice: ``_sync_canvas_view`` can flip
        ``_canvas_loaded_id`` mid-call (canvas load success/failure), and
        the visibility we computed before the sync goes stale. The
        ``_applying_mode`` guard prevents a true re-entry from within
        ``_sync_canvas_view`` — we detect the flip after sync and just
        loop once more inside the same outer call.
        """
        if self._applying_mode:
            return
        self._applying_mode = True
        try:
            for _ in range(2):
                eff_mode = self._effective_mode()
                canvas_loaded = self._canvas_loaded_id is not None
                is_mc = eff_mode == MODE_MISSION_CONTROL
                is_tc = eff_mode == MODE_TRAINING_CANVA
                is_scripts = eff_mode == MODE_SCRIPTS

                if self._mode_switch is not None:
                    self._mode_switch.setVisible(canvas_loaded)

                show_mc_middle = is_mc and canvas_loaded
                if self._run_source_selector is not None:
                    self._run_source_selector.setVisible(show_mc_middle)
                if self._opacity_icon is not None:
                    self._opacity_icon.setVisible(show_mc_middle)
                if self._opacity_slider is not None:
                    self._opacity_slider.setVisible(show_mc_middle)

                if self._tc_tools is not None:
                    # Save stays accessible in MC + TC; undo/redo only in TC.
                    self._tc_tools.setVisible((is_tc or is_mc) and canvas_loaded)
                    self._tc_tools.set_undo_redo_visible(is_tc and canvas_loaded)
                if self._scripts_tools is not None:
                    self._scripts_tools.setVisible(is_scripts and canvas_loaded)

                if self._content_stack is not None:
                    self._content_stack.setCurrentIndex(
                        self._CONTENT_SCRIPT if is_scripts else self._CONTENT_CANVAS
                    )

                compact = is_tc and canvas_loaded
                splitter_visible = canvas_loaded and not compact
                if self._v_splitter is not None:
                    self._v_splitter.setVisible(splitter_visible)
                if self._mission_panel is not None:
                    self._mission_panel.setVisible(is_mc and canvas_loaded)

                if self._external_canvas is not None:
                    in_tc_canvas = is_tc and canvas_loaded
                    self._external_canvas.set_interactive(in_tc_canvas)
                    self._external_canvas.set_minimap_visible(in_tc_canvas)

                pre_loaded = self._canvas_loaded_id
                if is_scripts:
                    self._sync_script_view()
                else:
                    self._sync_canvas_view()

                if compact != self._is_compact_state:
                    self._is_compact_state = compact
                    self.overlay_compact_changed.emit(compact)

                # If the sync flipped canvas-loaded state, run once more
                # to refresh the stale visibility decisions above.
                if self._canvas_loaded_id == pre_loaded:
                    return
        finally:
            self._applying_mode = False

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
        """转发 MainWindow 顶部 [Local(版本)…/Cloud] combo 给左卡引用。"""
        if self._train_config_card is not None:
            try:
                self._train_config_card.bind_link_combo(combo)
            except Exception as exc:
                log_warning(f"[mission] bind_link_combo failed: {exc!r}")

    def mirror_link_options(self, opts: list, current_data: str) -> None:
        """转发：用 MainWindow 同一份选项重建左卡 combo。"""
        if self._train_config_card is not None:
            try:
                self._train_config_card.mirror_link_options(opts, current_data)
            except Exception as exc:
                log_warning(f"[mission] mirror_link_options failed: {exc!r}")

    def set_link_current(self, data: str) -> None:
        """转发：把左卡 combo 选中项对齐到 data 令牌（无回声）。"""
        if self._train_config_card is not None:
            try:
                self._train_config_card.set_link_current(data)
            except Exception as exc:
                log_warning(f"[mission] set_link_current failed: {exc!r}")

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

    def effective_mode(self) -> str:
        """Public accessor for :meth:`_effective_mode`.

        Returns the mode actually in effect right now (forced to MC when
        no canvas is loaded). Used by the app shell's Ctrl+S dispatcher
        to decide between canvas save and script save.
        """
        return self._effective_mode()

    def try_save_active_script(self) -> bool:
        """Ctrl+S dispatch hook for Scripts mode.

        Returns True when the call was *consumed* by the Scripts editor
        (i.e. effective mode is Scripts and a script is currently loaded
        in the editor). The caller (MainWindow Ctrl+S handler) must skip
        the canvas-save path in that case. Returns False when there is
        no script context to save against — the caller then falls through
        to the canvas save.

        Always rule §1.8 compliant: a real save failure surfaces to the
        editor's status row via _save_script's _set_script_status calls
        and still returns True (consumed), so Ctrl+S does NOT silently
        fall through to save the canvas behind a failed script save.
        """
        if self._effective_mode() != MODE_SCRIPTS:
            return False
        if self._script_editor is None or not self.current_script:
            return False
        self._save_script()
        return True

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
        self._apply_mode()

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
        save_target, training-config card binding,
        and the loaded scene itself).
        """
        self.current_canvas = None
        self.current_script = None
        self._canvas_loaded_id = None
        self._script_loaded_id = None
        self._script_target_path = None
        self._script_is_virtual = False
        self._script_kind = None
        self._script_key = None
        self._script_variant = None
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
        # canvas 卸载会让 _effective_mode 回到 mission_control（TC/Scripts 在无
        # canvas 时不成立）。补发一次 mode_changed 让 Sidebar 在卸载瞬间重排导航。
        self.mode_changed.emit(self._effective_mode())

    def open_canvas(self, file_id: str) -> None:
        """Programmatically open ``file_id`` as if the user picked the row.

        Drives the canvas auto-load path. The sidebar Project Files panel
        owns the canvas list now — selection there reaches MissionControl
        via MainWindow's ``canvas_selected`` → ``open_canvas`` wire. The
        return path is symmetric: ``canvas_loaded.emit`` lets MainWindow
        re-highlight the row via ``ProjectsPanel.set_current_canvas``.
        """
        if not file_id:
            return
        self.current_canvas = file_id
        self._apply_mode()

    def load_script(self, file_id_or_virtual: str) -> None:
        """Public entry point for the four Scripts-mode sidebar panels.

        Accepts either a real-file id (``project:<rel>`` / ``system:<rel>``),
        a registry virtual id (``registry:<kind>:<key>[:<variant>]``), or
        the synthetic create-variant sentinel
        ``registry:<kind>:<key>:__new__``. Auto-flips the panel into
        Scripts mode so the editor is visible — clicking an item in any
        Scripts-rail panel is an explicit intent to view it.
        """
        if not file_id_or_virtual:
            return
        # Intercept ``__new__`` sentinel — open the variant creation
        # dialog and short-circuit the normal editor flow. The dialog's
        # ``accepted_variant`` signal hands us back ``(kind, key, name)``
        # so we can pivot to the new variant.
        if file_id_or_virtual.endswith(":__new__"):
            self._open_variant_create_dialog(file_id_or_virtual)
            return
        self.current_script = file_id_or_virtual
        if self._canvas_loaded_id is not None and self._mode != MODE_SCRIPTS:
            self._mode = MODE_SCRIPTS
            self._save_mode(MODE_SCRIPTS)
            if self._mode_switch is not None:
                try:
                    idx = [m for m, _ in _MODE_OPTIONS].index(MODE_SCRIPTS)
                    self._mode_switch.setCurrentIndex(idx, animated=True, emit=False)
                except ValueError:
                    pass
            self._apply_mode()
            self.mode_changed.emit(MODE_SCRIPTS)
        else:
            # Already in Scripts (or no canvas yet → _effective_mode forces
            # MC and the switch is hidden); still refresh the script view.
            self._apply_mode()

    # ------------------------------------------------------------------
    # Variant create — opened via the synthetic ``:__new__`` sentinel
    # ------------------------------------------------------------------
    def _open_variant_create_dialog(self, virtual_id: str) -> None:
        """Parse ``registry:<kind>:<key>:__new__`` and open the modal.

        On accept the dialog already wrote the variant via the resolver
        (so the sidebar will refresh automatically via
        ``AppSignals.user_scripts_changed``); we then pivot the editor to
        load the freshly-created variant.
        """
        parts = virtual_id.split(":", 3)
        if len(parts) != 4 or parts[0] != "registry":
            log_warning(
                f"[mission] malformed __new__ virtual id: {virtual_id!r}"
            )
            return
        kind, key = parts[1], parts[2]
        try:
            from application.ui.dialogs.variant_create_dialog import (
                VariantCreateDialog,
            )
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[mission] variant create dialog import: {exc!r}")
            return
        dlg = VariantCreateDialog(kind=kind, key=key, parent=self)

        def _on_accepted(k: str, ky: str, vname: str) -> None:
            self.load_script(f"registry:{k}:{ky}:{vname}")

        dlg.accepted_variant.connect(_on_accepted)
        dlg.exec()

    # ------------------------------------------------------------------
    # Script page plumbing
    # ------------------------------------------------------------------
    def _sync_script_view(self) -> None:
        if self._script_stack is None:
            return
        if self._project_info is None or not self.current_script:
            self._script_stack.setCurrentIndex(0)
            self._script_loaded_id = None
            self._script_is_virtual = False
            self._script_kind = None
            self._script_key = None
            self._script_variant = None
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
        if file_id.startswith("registry:"):
            self._load_registry_virtual_into_editor(file_id)
            return
        # File id path — resolve to disk via the same routine ProjectsPanel
        # uses for canvas resolution.
        try:
            path = resolve_file(self._project_info, file_id)
        except ValueError as exc:
            self._set_script_status(f"unresolved: {exc}", error=True)
            self._script_loaded_id = None
            self._script_target_path = None
            self._script_is_virtual = False
            self._script_kind = None
            self._script_key = None
            self._script_variant = None
            return
        if self._script_path_label is not None:
            self._script_path_label.setText(path.name)
        self._script_target_path = path
        self._script_is_virtual = False
        self._script_kind = None
        self._script_key = None
        self._script_variant = None
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

    def _load_registry_virtual_into_editor(self, virtual_id: str) -> None:
        """Resolve ``registry:<kind>:<key>[:<variant>]`` → editor buffer.

        Routes every load through
        :func:`application.service.scripts.resolver.resolve`. The fourth
        segment of the virtual id is optional — missing or ``"preset"``
        loads the factory preset (read-only, conceptually); anything else
        is treated as a user variant under
        ``Paths.USER_CONFIG_DIR / scripts / <kind> / <key> / <variant>.py``.

        Editor state set on success:

        * ``_script_kind`` / ``_script_key`` / ``_script_variant`` — drive
          :meth:`_save_script` (variant=None ⇒ save rejected).
        * ``_script_target_path`` — informational for the variant case;
          unused for the preset case (preset writes never occur).
        """
        from application.service.scripts import resolver as _resolver
        from application.service.signals import current_backend

        parts = virtual_id.split(":", 3)
        if len(parts) < 3 or parts[0] != "registry":
            self._set_script_status(f"bad virtual id: {virtual_id}", error=True)
            self._script_loaded_id = None
            self._script_is_virtual = False
            self._script_kind = None
            self._script_key = None
            self._script_variant = None
            return
        kind = parts[1]
        key = parts[2]
        variant_token = parts[3] if len(parts) == 4 else ""
        variant = None if variant_token in ("", "preset") else variant_token

        backend = current_backend()
        resolved = _resolver.resolve(
            kind, key, variant=variant, backend=backend
        )
        # If a specific variant was requested but missing, fall back to
        # preset rather than blanking the editor — gives the user a
        # readable source even when their variant file has been deleted
        # behind the app's back.
        fallback_used = False
        if resolved is None and variant is not None:
            resolved = _resolver.resolve(kind, key, variant=None, backend=backend)
            fallback_used = True
        if resolved is None:
            self._set_script_status(
                f"registry miss: {kind}/{key} (backend={backend or 'none'})",
                error=True,
            )
            self._script_loaded_id = None
            self._script_is_virtual = False
            self._script_kind = None
            self._script_key = None
            self._script_variant = None
            if self._script_editor is not None:
                self._script_editor.set_text("")
            return

        # Drop into the editor; CodeEditorWidget.set_text re-baselines
        # the dirty tracker so dirtyChanged starts at False.
        self._script_editor.set_text(resolved.source)
        self._script_loaded_id = virtual_id
        self._script_is_virtual = True
        self._script_kind = kind
        self._script_key = key
        # If we fell back to preset, the editor is showing preset source —
        # mark the variant slot as None so Save can't accidentally write
        # the preset source back as a fresh variant.
        self._script_variant = None if (variant is None or fallback_used) else variant

        # Path-label: ``[Rewards|Termins|Observs] <key> · <variant>``.
        tag = {
            "reward": "Rewards",
            "termination": "Termins",
            "observation": "Observs",
            "discriminator": "Disc",
        }.get(kind, kind.title())
        label_suffix = f" · {self._script_variant}" if self._script_variant else " · preset"
        if self._script_path_label is not None:
            self._script_path_label.setText(f"[{tag}] {key}{label_suffix}")

        # Informational target path (variant case only); presets never
        # write so we leave this None to make the assertion in
        # ``_save_script`` straightforward.
        if self._script_variant:
            try:
                self._script_target_path = (
                    Paths.USER_CONFIG_DIR
                    / "scripts"
                    / {
                        "reward": "rewards",
                        "termination": "terminations",
                        "observation": "observations",
                        "discriminator": "discriminator",
                    }.get(kind, kind)
                    / key
                    / f"{self._script_variant}.py"
                )
            except Exception:                                     # noqa: BLE001
                self._script_target_path = None
        else:
            self._script_target_path = None

        if fallback_used:
            self._set_script_status(
                f"variant {variant!r} missing — showing preset", error=True
            )
        elif self._script_variant:
            self._set_script_status(
                f"variant: {self._script_variant}", error=False
            )
        else:
            self._set_script_status("preset (clone to edit)", error=False)

    def _save_script(self) -> None:
        if self._script_editor is None:
            return

        # Virtual-id (registry preset / user variant) save path.
        if self._script_is_virtual:
            if not self._script_kind or not self._script_key:
                self._set_script_status("no script context", error=True)
                return
            if not self._script_variant:
                # Preset rows are read-only — users create a variant via
                # the sidebar "+ new variant" entrypoint (Stage 2).
                self._set_script_status(
                    "preset is read-only — clone as a variant first",
                    error=True,
                )
                return
            from application.service.scripts import resolver as _resolver
            source = self._script_editor.text()
            # Preserve existing meta so a Save doesn't wipe families /
            # description silently. The sidebar's edit-meta dialog is
            # the place to mutate those.
            existing = next(
                (
                    m for m in _resolver.list_variants(
                        self._script_kind, self._script_key
                    )
                    if m.name == self._script_variant
                ),
                None,
            )
            try:
                _resolver.save_variant(
                    self._script_kind,
                    self._script_key,
                    self._script_variant,
                    source,
                    families=(
                        sorted(existing.families) if existing else None
                    ),
                    description=(existing.description if existing else ""),
                    based_on=(existing.based_on if existing else "preset"),
                )
            except ValueError as exc:
                self._set_script_status(f"save rejected: {exc}", error=True)
                return
            except OSError as exc:
                self._set_script_status(f"save failed: {exc}", error=True)
                return
            # Rebaseline the editor's dirty tracker so subsequent script
            # switches don't fire the "discards unsaved edits" warning.
            # ``_refresh_baseline`` is the SDK-internal hook used by
            # ``save_file`` after a successful write — we have to call it
            # ourselves because we wrote through resolver, not save_file.
            try:
                self._script_editor._refresh_baseline()           # noqa: SLF001
            except Exception:                                     # noqa: BLE001
                pass
            self._set_script_status(
                f"variant saved: {self._script_variant}", error=False
            )
            return

        # Real-file save path (project: / system:) — unchanged from
        # legacy behaviour.
        target = self._script_target_path
        if target is None:
            self._set_script_status("no path", error=True)
            return
        ok = self._script_editor.save_file(target)
        if not ok:
            self._set_script_status("save failed", error=True)
            return
        self._set_script_status("saved", error=False)

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
            self._script_is_virtual = False
            self.current_script = None
            if self._script_editor is not None:
                self._script_editor.set_text("")
            if self._script_path_label is not None:
                self._script_path_label.setText("")
            if self._script_status is not None:
                self._script_status.setText("")
        self._apply_mode()
        # 解除 project 绑定时 _canvas_loaded_id 已置 None，effective_mode 会
        # 强制回到 mission_control；补发 mode_changed 让 Sidebar 重排——否则
        # 登出/账号切换走到这里时 Sidebar 仍然停留在 training_canva 样式。
        if info is None:
            self.mode_changed.emit(self._effective_mode())

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
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
            # Scripts page is opaque — it fully covers the canvas in
            # Scripts mode, so its standby placeholder and editor sit on
            # ``bg_1`` rather than a transparent main_screen.
            f"QWidget#scriptPage {{ background-color: {bg_main}; }}"
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
        if self._scripts_tools is not None:
            self._scripts_tools.apply_theme()


__all__ = ["MissionControlPanel"]
