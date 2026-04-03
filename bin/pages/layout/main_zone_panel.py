"""Main zone panel for mission workspace."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QEvent, QPoint, QTimer, QSize
from PySide6.QtGui import QKeySequence, QShortcut, QIcon

def _zone_icon(icon_name: str) -> QIcon:
    """Load an SVG icon via the global theme-aware resolver."""
    from src.system.core.theme_manager import get_icon
    return get_icon(icon_name)


def _zone_toggle_button_extent() -> int:
    """Square toggle sized to fit the header row."""
    return 28


def _zone_toggle_icon_size(extent: int | None = None) -> QSize:
    """Stretch the glyph to nearly fill the square toggle button."""
    btn_extent = extent or _zone_toggle_button_extent()
    icon_extent = max(12, btn_extent - 4)
    return QSize(icon_extent, icon_extent)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QSplitter,
    QTabWidget,
    QSizePolicy,
    QLabel,
    QPushButton,
    QDialog,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QComboBox,
    QAbstractSpinBox,
    QFrame,
)

from bin.pages.canvas.code_editor import CodeEditor
from bin.pages.canvas.diagnostics_panel import DiagnosticsPanel
from bin.pages.canvas.graph_scene import GraphScene
from bin.pages.canvas.graph_view import GraphView
from bin.pages.layout.module_cards import ModulePalette
from bin.pages.settings.capability_inspector import CapabilityInspector
from bin.pages.settings.settings_panel import SettingsPanel
from src.system.core.config_manager import ConfigManager
from src.system.core.localisation import tr
from bin.pages.training.training_panel import TrainingPanel
from bin.pages.layout.behavior_panel import BehaviorPanel


class MainZonePanel(QWidget):
    """Main workspace panel containing mission controls + mission editor."""

    start_requested    = Signal()
    training_requested = Signal()
    exit_requested     = Signal()
    pause_requested    = Signal()
    abort_requested    = Signal()
    reset_requested    = Signal()
    # Emitted when DiagnosticsPanel requests canvas navigation (node_id: int)
    navigate_to_node   = Signal(object)
    workflow_load_requested = Signal()
    workflow_save_requested = Signal()
    workflow_save_as_requested = Signal()
    workflow_browser_requested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scenario_settings = {
            "mujoco_gl_backend": "glfw",
            "mujoco_realtime": True,
            "mujoco_timestep": 0.0020,
            "mujoco_scene_xml": "",
            # Cycle 3 STAGE-06: advanced MuJoCo settings (bounded subset)
            "mujoco_gravity_z": -9.81,
            "mujoco_solver_iterations": 100,
            "mujoco_render_width": 640,
            "mujoco_render_height": 480,
        }
        self.config = ConfigManager()
        self._mission_control_dragging = False
        self._mission_control_drag_offset = QPoint()
        self._mission_control_position_initialized = False
        self._mission_control_parent = None
        self._mission_control_icon_size = QSize(20, 20)
        # Fallback widths used when layout state cannot be read from user.ini
        self._saved_canvas_width = 920
        self._saved_compiler_width = 680
        # node_id → ScriptEditor widget (for open script tabs)
        self._script_editor_tabs: dict = {}
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("mainZone")
        self._layout_mode = "default"  # default | canvas | compiler

        main_zone_layout = QVBoxLayout(self)
        main_zone_layout.setContentsMargins(0, 0, 0, 0)
        main_zone_layout.setSpacing(0)

        # Execution summary bar (Cycle 1 STAGE-03) — hidden until after first run
        self._exec_summary_bar = QWidget()
        self._exec_summary_bar.setObjectName("execSummaryBar")
        _summary_layout = QHBoxLayout(self._exec_summary_bar)
        _summary_layout.setContentsMargins(10, 3, 10, 3)
        _summary_layout.setSpacing(12)
        self._exec_status_icon = QLabel("")
        self._exec_status_icon.setFixedWidth(18)
        _summary_layout.addWidget(self._exec_status_icon)
        self._exec_summary_label = QLabel("")
        self._exec_summary_label.setObjectName("execSummaryLabel")
        _summary_layout.addWidget(self._exec_summary_label, 1)
        self._exec_compat_label = QLabel("")
        self._exec_compat_label.setObjectName("execCompatLabel")
        _summary_layout.addWidget(self._exec_compat_label)
        self._details_btn = QPushButton("Details")
        self._details_btn.setObjectName("execDetailsBtn")
        self._details_btn.setFixedWidth(56)
        self._details_btn.setToolTip("Show diagnostics for failed nodes")
        self._details_btn.setVisible(False)
        self._details_btn.clicked.connect(self._on_details_clicked)
        _summary_layout.addWidget(self._details_btn)
        _dismiss_btn = QPushButton("×")
        _dismiss_btn.setFixedWidth(20)
        _dismiss_btn.setToolTip("Dismiss")
        _dismiss_btn.clicked.connect(self.clear_execution_summary)
        _summary_layout.addWidget(_dismiss_btn)
        self._exec_summary_bar.setVisible(False)
        main_zone_layout.addWidget(self._exec_summary_bar)

        self.navigation_header = QWidget()
        self.navigation_header.setObjectName("navigationHeader")
        navigation_header_layout = QHBoxLayout(self.navigation_header)
        navigation_header_layout.setContentsMargins(10, 6, 10, 6)
        navigation_header_layout.setSpacing(0)
        navigation_header_layout.addStretch()
        main_zone_layout.addWidget(self.navigation_header)

        # Diagnostics panel disabled by request: keep an unattached instance only
        # so existing call sites can no-op against a stable attribute.
        self.diag_panel = DiagnosticsPanel()
        self.diag_panel.navigate_requested.connect(self.navigate_to_node)
        self.diag_panel.closed.connect(self._on_diag_panel_closed)
        self.diag_panel.setVisible(False)
        # Compatibility alias; runtime zone no longer exists as a separate panel.
        self.runtime_zone = None

        self.mission_content_row = QWidget()
        self.mission_content_row.setObjectName("missionContentRow")
        mission_content_row_layout = QHBoxLayout(self.mission_content_row)
        mission_content_row_layout.setContentsMargins(0, 0, 0, 0)
        mission_content_row_layout.setSpacing(0)

        self.mission_content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Canvas zone ───────────────────────────────────────────────────────
        self.module_palette = ModulePalette()
        self.graph_scene = GraphScene()
        self.graph_view = GraphView(self.graph_scene)
        self.canvas_zone = QWidget()
        self.canvas_zone.setObjectName("canvasZone")
        canvas_zone_layout = QVBoxLayout(self.canvas_zone)
        canvas_zone_layout.setContentsMargins(0, 0, 0, 0)
        canvas_zone_layout.setSpacing(0)

        self.canvas_header = QWidget()
        self.canvas_header.setObjectName("canvasHeader")
        canvas_header_layout = QHBoxLayout(self.canvas_header)
        canvas_header_layout.setContentsMargins(10, 6, 10, 6)
        canvas_header_layout.setSpacing(8)
        self.canvas_save_btn = QPushButton("Save")
        self.canvas_save_btn.setObjectName("canvasHeaderButton")
        self.canvas_save_btn.clicked.connect(self.workflow_save_requested.emit)
        canvas_header_layout.addWidget(self.canvas_save_btn)
        self.canvas_save_as_btn = QPushButton("Save as...")
        self.canvas_save_as_btn.setObjectName("canvasHeaderButton")
        self.canvas_save_as_btn.clicked.connect(self.workflow_save_as_requested.emit)
        canvas_header_layout.addWidget(self.canvas_save_as_btn)
        canvas_header_layout.addStretch()
        self.canvas_toggle_btn = QPushButton()
        self.canvas_toggle_btn.setObjectName("zoneToggleButton")
        canvas_toggle_extent = _zone_toggle_button_extent()
        self.canvas_toggle_btn.setFixedSize(canvas_toggle_extent, canvas_toggle_extent)
        self.canvas_toggle_btn.setIconSize(_zone_toggle_icon_size(canvas_toggle_extent))
        self.canvas_toggle_btn.clicked.connect(self._toggle_canvas_fullscreen)
        canvas_header_layout.addWidget(self.canvas_toggle_btn)
        canvas_zone_layout.addWidget(self.canvas_header)

        # canvas splitter (graph_view only; node library now lives in sidebar)
        self.canvas_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas_splitter.addWidget(self.graph_view)
        self.canvas_splitter.setSizes([1000])
        self.canvas_splitter.setStretchFactor(0, 1)

        # mission_zone = canvas content area (parent for floating mission control)
        self.mission_zone = QWidget()
        self.mission_zone.setObjectName("missionZone")
        mission_zone_layout = QVBoxLayout(self.mission_zone)
        mission_zone_layout.setContentsMargins(0, 0, 0, 0)
        mission_zone_layout.setSpacing(0)
        mission_zone_layout.addWidget(self.canvas_splitter, 1)
        self.tool_row = None
        self._init_floating_mission_control()

        # workspace_tabs lives INSIDE canvas_zone so its tab bar stays within canvas area
        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setObjectName("workspaceTabs")
        self.workspace_tabs.addTab(self.mission_zone, "[New File]")
        self.workspace_tabs.setTabsClosable(True)
        self.workspace_tabs.tabCloseRequested.connect(self._on_workspace_tab_close_requested)
        self.workspace_tabs.tabBar().hide()

        self.behavior_panel = BehaviorPanel()
        self.behavior_panel.back_requested.connect(self.open_mission_tab)

        # Training panel — permanent tab, closeable only programmatically
        self.training_panel = TrainingPanel()
        self._training_tab_index: int = -1   # set when first shown

        # Start node settings surface — a single popup containing both panels.
        self.settings_panel = SettingsPanel("unitree", {})
        self.capability_inspector = CapabilityInspector({})
        self.capability_inspector.focus_setting.connect(self._on_capability_focus_setting)
        self._init_start_settings_dialog()

        # Right-align the tab bar so it stays clear of the left edge
        self.workspace_tabs.tabBar().setExpanding(False)
        _tab_left_spacer = QWidget()
        _tab_left_spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.workspace_tabs.setCornerWidget(_tab_left_spacer, Qt.Corner.TopLeftCorner)
        self._init_workspace_tab_shortcuts()
        self._refresh_workspace_tab_close_buttons()

        canvas_zone_layout.addWidget(self.workspace_tabs, 1)

        # ── Compiler zone ─────────────────────────────────────────────────────
        self.code_editor = CodeEditor()
        self.compiler_zone = QWidget()
        self.compiler_zone.setObjectName("compilerZone")
        compiler_zone_layout = QVBoxLayout(self.compiler_zone)
        compiler_zone_layout.setContentsMargins(0, 0, 0, 0)
        compiler_zone_layout.setSpacing(0)

        compiler_header = QWidget()
        compiler_header.setObjectName("compilerHeader")
        compiler_header_layout = QHBoxLayout(compiler_header)
        compiler_header_layout.setContentsMargins(10, 6, 10, 6)
        compiler_header_layout.setSpacing(8)
        compiler_title = QLabel("Compiler")
        compiler_header_layout.addWidget(compiler_title)
        self.compile_to_canvas_btn = QPushButton("Compile to Canvas")
        self.compile_to_canvas_btn.setObjectName("compileToCanvasBtn")
        self.compile_to_canvas_btn.setFixedHeight(28)
        self.compile_to_canvas_btn.clicked.connect(self.code_editor.compile_code)
        compiler_header_layout.addWidget(self.compile_to_canvas_btn)
        compiler_header_layout.addStretch()
        self.compiler_toggle_btn = QPushButton()
        self.compiler_toggle_btn.setObjectName("zoneToggleButton")
        compiler_toggle_extent = _zone_toggle_button_extent()
        self.compiler_toggle_btn.setFixedSize(compiler_toggle_extent, compiler_toggle_extent)
        self.compiler_toggle_btn.setIconSize(_zone_toggle_icon_size(compiler_toggle_extent))
        self.compiler_toggle_btn.clicked.connect(self._toggle_compiler_fullscreen)
        compiler_header_layout.addWidget(self.compiler_toggle_btn)
        compiler_zone_layout.addWidget(compiler_header)

        # Compiler tab widget: Tab 0 = Main Workflow (non-closeable), Tab 1+ = Script Box editors
        self.compiler_tabs = QTabWidget()
        self.compiler_tabs.setObjectName("compilerTabs")
        self.compiler_tabs.setTabsClosable(True)
        self.compiler_tabs.tabCloseRequested.connect(self._on_compiler_tab_close_requested)
        self.compiler_tabs.addTab(self.code_editor, "[New File]: Main Workflow")
        # Make Tab 0 non-closeable by removing its close button
        self.compiler_tabs.tabBar().setTabButton(
            0, self.compiler_tabs.tabBar().ButtonPosition.RightSide, None
        )
        compiler_zone_layout.addWidget(self.compiler_tabs, 1)

        # Graph scene and code editor (bidirectional)
        self.graph_scene.set_code_editor(self.code_editor)
        self.code_editor.set_graph_scene(self.graph_scene)

        self.mission_content_splitter.addWidget(self.canvas_zone)
        self.mission_content_splitter.addWidget(self.compiler_zone)
        self.mission_content_splitter.setSizes([920, 680])
        self.mission_content_splitter.setStretchFactor(0, 1)
        self.mission_content_splitter.setStretchFactor(1, 1)

        mission_content_row_layout.addWidget(self.mission_content_splitter, 1)
        main_zone_layout.addWidget(self.mission_content_row, 1)
        self._restore_layout_from_config()
        QTimer.singleShot(0, self._ensure_mission_control_position)

    def _init_floating_mission_control(self) -> None:
        self._mission_control_parent = self.mission_zone
        self.mission_control_float = QWidget(self._mission_control_parent)
        self.mission_control_float.setObjectName("missionControlFloat")
        mission_control_layout = QHBoxLayout(self.mission_control_float)
        mission_control_layout.setContentsMargins(10, 6, 10, 6)
        mission_control_layout.setSpacing(8)

        self.mission_control_drag_btn = QPushButton("")
        self.mission_control_drag_btn.setObjectName("missionControlDragHandle")
        self.mission_control_drag_btn.setToolTip("Drag")
        self.mission_control_drag_btn.setFlat(True)
        self.mission_control_drag_btn.setCursor(Qt.OpenHandCursor)
        self.mission_control_drag_btn.installEventFilter(self)
        mission_control_layout.addWidget(self.mission_control_drag_btn)
        mission_control_layout.addSpacing(8)

        self.start_btn = QPushButton("?")
        self.start_btn.setToolTip("Start")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_requested.emit)
        mission_control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("?")
        self.pause_btn.setToolTip("Pause")
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.clicked.connect(self.pause_requested.emit)
        mission_control_layout.addWidget(self.pause_btn)

        self.abort_btn = QPushButton("?")
        self.abort_btn.setToolTip("Abort")
        self.abort_btn.setCursor(Qt.PointingHandCursor)
        self.abort_btn.clicked.connect(self.abort_requested.emit)
        mission_control_layout.addWidget(self.abort_btn)

        self.reset_btn = QPushButton("?")
        self.reset_btn.setToolTip("Reset")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        mission_control_layout.addWidget(self.reset_btn)

        self.scenario_settings_btn = QPushButton("?")
        self.scenario_settings_btn.setObjectName("scenarioSettingsButton")
        self.scenario_settings_btn.setToolTip(tr("ui.mujoco_setting", "Mujoco Setting"))
        self.scenario_settings_btn.setCursor(Qt.PointingHandCursor)
        self.scenario_settings_btn.setStyleSheet("padding: 0px;")
        self.scenario_settings_btn.clicked.connect(self._open_scenario_settings)
        mission_control_layout.addWidget(self.scenario_settings_btn)

        # Separator + PageSwitcher
        from PySide6.QtWidgets import QFrame as _QFrame
        _mc_sep2 = _QFrame()
        _mc_sep2.setFrameShape(_QFrame.Shape.VLine)
        _mc_sep2.setObjectName("missionControlSep")
        mission_control_layout.addWidget(_mc_sep2)

        from bin.pages.layout.misc import PageSwitcher
        self.page_switcher = PageSwitcher(self.mission_control_float)
        mission_control_layout.addWidget(self.page_switcher)

        self._apply_mission_control_button_sizes()
        self._apply_mission_control_icons()

        self.mission_control_float.adjustSize()
        self.mission_control_float.raise_()
        self.runtime_zone = self.mission_control_float

        # ── Overview panel (child of mission_zone, hidden by default) ─
        from bin.components.overview_panel import ControlPanel
        from bin.components.mission_overview_content import MissionOverviewContent
        self._overview_panel = ControlPanel(self._mission_control_parent)
        self._mc_overview_content = MissionOverviewContent()
        self._overview_panel.set_content(self._mc_overview_content)
        self._overview_panel.set_title("Mission Overview")
        self._overview_panel.collapse_requested.connect(self._toggle_overview_panel)
        self._overview_panel.hide()

        # Bidirectional robot sync: GraphScene ↔ RobotPanel
        self.graph_scene.robot_type_changed.connect(
            self._overview_panel.robot_panel.set_robot_type
        )
        self._overview_panel.robot_changed.connect(self._on_robot_panel_changed)
        # Initialize RobotPanel with current graph scene robot
        self._overview_panel.robot_panel.set_robot(
            getattr(self.graph_scene, '_robot_brand', 'unitree'),
            getattr(self.graph_scene, '_robot_type', 'go2'),
        )

    def _apply_mission_control_button_sizes(self) -> None:
        buttons = [
            self.mission_control_drag_btn,
            self.start_btn,
            self.pause_btn,
            self.abort_btn,
            self.reset_btn,
            self.scenario_settings_btn,
        ]
        size = 30
        for btn in buttons:
            btn.setFixedSize(size, size)
            btn.setIconSize(self._mission_control_icon_size)

    def _apply_mission_control_icons(self) -> None:
        from src.system.core.theme_manager import get_icon
        icon_bindings = [
            (self.mission_control_drag_btn, "move", ""),
            (self.start_btn, "play", "?"),
            (self.pause_btn, "pause", "?"),
            (self.abort_btn, "stop", "?"),
            (self.reset_btn, "reset", "?"),
            (self.scenario_settings_btn, "setting", "?"),
        ]
        for button, icon_name, fallback_text in icon_bindings:
            icon = get_icon(icon_name)
            if not icon.isNull():
                button.setIcon(icon)
                button.setText("")
            else:
                button.setIcon(QIcon())
                button.setText(fallback_text)

    def _default_mission_control_pos(self) -> QPoint:
        panel = self.mission_control_float
        right_margin = 40
        top_margin = 12
        x = max(0, self._mission_control_parent.width() - panel.width() - right_margin)
        y = top_margin  # top-right
        return QPoint(x, y)

    def _save_mission_control_position(self) -> None:
        try:
            pos = self.mission_control_float.pos()
            self.config.set("LAYOUT", "mission_control_float_x", str(pos.x()), config_type="user")
            self.config.set("LAYOUT", "mission_control_float_y", str(pos.y()), config_type="user")
            self.config.save_user_config()
        except Exception:
            pass

    def _load_mission_control_position(self) -> QPoint:
        # Match Training Ground behavior: dock to the parent's top-right when
        # the bar is first shown instead of restoring a persisted position.
        return self._default_mission_control_pos()

    def _clamp_mission_control_pos(self, pos: QPoint) -> QPoint:
        panel = self.mission_control_float
        max_x = max(0, self._mission_control_parent.width() - panel.width())
        max_y = max(0, self._mission_control_parent.height() - panel.height())
        x = max(0, min(pos.x(), max_x))
        y = max(0, min(pos.y(), max_y))
        return QPoint(x, y)

    def _ensure_mission_control_position(self) -> None:
        if not hasattr(self, "mission_control_float"):
            return
        if not self._mission_control_position_initialized:
            # Defer until the parent widget has been laid out and has a real width.
            # This handles the case where the mission page is hidden inside a
            # QStackedWidget at startup (parent width is 0 while hidden).
            if self._mission_control_parent.width() == 0:
                return
            target = self._load_mission_control_position()
            self._mission_control_position_initialized = True
        else:
            target = self.mission_control_float.pos()
        self.mission_control_float.move(self._clamp_mission_control_pos(target))
        self.mission_control_float.raise_()

    def eventFilter(self, watched, event):
        if watched is self.mission_control_drag_btn:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._mission_control_dragging = True
                self._mission_control_drag_offset = event.position().toPoint()
                self.mission_control_drag_btn.setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self._mission_control_dragging:
                parent_pos = self._mission_control_parent.mapFromGlobal(event.globalPosition().toPoint())
                new_pos = parent_pos - self._mission_control_drag_offset
                self.mission_control_float.move(self._clamp_mission_control_pos(new_pos))
                return True
            if event.type() == QEvent.MouseButtonRelease and self._mission_control_dragging:
                self._mission_control_dragging = False
                self.mission_control_drag_btn.setCursor(Qt.OpenHandCursor)
                self._save_mission_control_position()
                return True
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._mission_control_position_initialized:
            QTimer.singleShot(0, self._ensure_mission_control_position)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ensure_mission_control_position()

    def _on_robot_panel_changed(self, brand_id: str, model_id: str) -> None:
        """RobotPanel selection → update GraphScene Start node."""
        self.graph_scene._robot_brand = brand_id
        self.graph_scene.set_robot_type(model_id)

    def _toggle_overview_panel(self) -> None:
        if self._overview_panel.isVisible():
            self._overview_panel.hide()
        else:
            self._overview_panel.place_center_bottom()
            self._overview_panel.show()
            self._overview_panel.raise_()
        self.mission_control_float.raise_()

    def set_page_mode(self, page: str) -> None:
        """Update float bar border color to match the active page accent."""
        from src.system.core.theme_manager import get_color
        if page == "training":
            accent = get_color("tab_bg_training", "#F6D393")
        else:
            accent = get_color("tab_bg_checked", "#A390FC")
        card_bg = get_color("card_bg", "#272829")
        if hasattr(self, "mission_control_float"):
            self.mission_control_float.setStyleSheet(
                f"#missionControlFloat {{"
                f" background-color: {card_bg};"
                f" border: 1px solid {accent};"
                f" border-radius: 10px;"
                f"}}"
            )
        if hasattr(self, "page_switcher"):
            self.page_switcher.set_current_page(page)

    def dock_mission_control_top_right(self) -> None:
        """Force the floating mission control bar to re-dock at the canvas top-right."""
        if not hasattr(self, "mission_control_float"):
            return
        self.mission_control_float.adjustSize()
        self._mission_control_position_initialized = False
        self._ensure_mission_control_position()

    def _apply_default_layout(self):
        """Restore split view using last-known widths."""
        self.mission_content_splitter.setSizes([self._saved_canvas_width, self._saved_compiler_width])
        self._layout_mode = "default"
        self._sync_zone_toggle_buttons()

    def _toggle_canvas_fullscreen(self):
        if self._layout_mode == "canvas":
            self._apply_default_layout()
        else:
            self._capture_default_widths()
            self.mission_content_splitter.setSizes([1, 0])
            self._layout_mode = "canvas"
            self._sync_zone_toggle_buttons()
        self._save_layout_to_config()

    def _toggle_compiler_fullscreen(self):
        if self._layout_mode == "compiler":
            self._apply_default_layout()
        else:
            self._capture_default_widths()
            self.mission_content_splitter.setSizes([0, 1])
            self._layout_mode = "compiler"
            self._sync_zone_toggle_buttons()
        self._save_layout_to_config()

    def _sync_zone_toggle_buttons(self):
        # Canvas button: WW = canvas-fullscreen state, FW = split/windowed state
        icon_size = _zone_toggle_icon_size(self.canvas_toggle_btn.width())
        canvas_icon = _zone_icon("ICON_WW") if self._layout_mode == "canvas" else _zone_icon("ICON_FW")
        self.canvas_toggle_btn.setIcon(canvas_icon)
        self.canvas_toggle_btn.setIconSize(icon_size)
        # Compiler button: same icon logic mirrored
        compiler_icon = _zone_icon("ICON_WW") if self._layout_mode == "compiler" else _zone_icon("ICON_FW")
        self.compiler_toggle_btn.setIcon(compiler_icon)
        self.compiler_toggle_btn.setIconSize(icon_size)

    def _capture_default_widths(self):
        """Snapshot current splitter widths before leaving default (split) mode."""
        if self._layout_mode == "default":
            sizes = self.mission_content_splitter.sizes()
            if len(sizes) >= 2 and sizes[0] > 10 and sizes[1] > 10:
                self._saved_canvas_width = sizes[0]
                self._saved_compiler_width = sizes[1]

    def _save_layout_to_config(self):
        """Persist layout mode and split widths to user.ini."""
        try:
            self.config.set("LAYOUT", "mode", self._layout_mode, config_type="user")
            self.config.set("LAYOUT", "canvas_width", str(self._saved_canvas_width), config_type="user")
            self.config.set("LAYOUT", "compiler_width", str(self._saved_compiler_width), config_type="user")
            self.config.save_user_config()
        except Exception:
            pass  # layout persistence is non-critical

    def _restore_layout_from_config(self):
        """Read layout state from user.ini and apply. Defaults to canvas-fullscreen."""
        try:
            mode = self.config.get("LAYOUT", "mode", fallback="canvas", config_type="user")
            self._saved_canvas_width = self.config.get_int(
                "LAYOUT", "canvas_width", fallback=920, config_type="user"
            )
            self._saved_compiler_width = self.config.get_int(
                "LAYOUT", "compiler_width", fallback=680, config_type="user"
            )
        except Exception:
            mode = "canvas"

        if mode == "compiler":
            self.mission_content_splitter.setSizes([0, 1])
            self._layout_mode = "compiler"
        elif mode == "default":
            self.mission_content_splitter.setSizes([self._saved_canvas_width, self._saved_compiler_width])
            self._layout_mode = "default"
        else:  # "canvas" or unknown → canvas fullscreen
            self.mission_content_splitter.setSizes([1, 0])
            self._layout_mode = "canvas"
        self._sync_zone_toggle_buttons()

    def _init_start_settings_dialog(self) -> None:
        """Create the Start-node popup that groups settings and capabilities."""
        self.start_settings_dialog = QDialog(self)
        self.start_settings_dialog.setObjectName("startSettingsDialog")
        self.start_settings_dialog.setWindowTitle("Start Settings")
        self.start_settings_dialog.setModal(True)
        self.start_settings_dialog.resize(1120, 760)

        root = QVBoxLayout(self.start_settings_dialog)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Start Node Settings")
        title.setObjectName("startSettingsDialogTitle")
        root.addWidget(title)

        subtitle = QLabel("Settings and capabilities are managed together from the Start node.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("startSettingsDialogSubtitle")
        root.addWidget(subtitle)

        content_splitter = QSplitter(Qt.Orientation.Horizontal, self.start_settings_dialog)
        content_splitter.setChildrenCollapsible(False)

        settings_container = QWidget(content_splitter)
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)
        settings_label = QLabel("Settings")
        settings_label.setObjectName("startSettingsSectionTitle")
        settings_layout.addWidget(settings_label)
        settings_layout.addWidget(self.settings_panel, 1)

        capability_container = QWidget(content_splitter)
        capability_layout = QVBoxLayout(capability_container)
        capability_layout.setContentsMargins(0, 0, 0, 0)
        capability_layout.setSpacing(8)
        capability_label = QLabel("Capabilities")
        capability_label.setObjectName("startSettingsSectionTitle")
        capability_layout.addWidget(capability_label)
        capability_layout.addWidget(self.capability_inspector, 1)

        content_splitter.addWidget(settings_container)
        content_splitter.addWidget(capability_container)
        content_splitter.setSizes([680, 440])
        root.addWidget(content_splitter, 1)

        separator = QFrame(self.start_settings_dialog)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(separator)

        actions = QHBoxLayout()
        actions.addStretch()
        close_btn = QPushButton("Close", self.start_settings_dialog)
        close_btn.clicked.connect(self.start_settings_dialog.accept)
        actions.addWidget(close_btn)
        root.addLayout(actions)

    def _refresh_workspace_tab_close_buttons(self) -> None:
        _permanent = {self.mission_zone}
        tab_bar = self.workspace_tabs.tabBar()
        for i in range(self.workspace_tabs.count()):
            if self.workspace_tabs.widget(i) in _permanent:
                tab_bar.setTabButton(i, tab_bar.ButtonPosition.RightSide, None)

    def _find_workspace_tab_index(self, page: QWidget) -> int:
        return self.workspace_tabs.indexOf(page)

    def _ensure_workspace_tab(self, tab_id: str, title_override: str = "") -> int:
        if tab_id == "behavior":
            page = self.behavior_panel
            title = title_override.strip() or "Behavior"
        else:
            return -1

        idx = self._find_workspace_tab_index(page)
        if idx >= 0:
            if title_override.strip():
                self.workspace_tabs.setTabText(idx, title)
            return idx
        idx = self.workspace_tabs.addTab(page, title)
        self.workspace_tabs.setCurrentIndex(idx)
        self._refresh_workspace_tab_close_buttons()
        return idx

    def open_training_tab(self) -> None:
        """Switch workspace to the Training tab, creating it if necessary."""
        existing = self._find_workspace_tab_index(self.training_panel)
        if existing >= 0:
            self.workspace_tabs.setCurrentIndex(existing)
            self._training_tab_index = existing
            return
        idx = self.workspace_tabs.addTab(self.training_panel, "?? Training")
        self._training_tab_index = idx
        self.workspace_tabs.setCurrentIndex(idx)
        self._refresh_workspace_tab_close_buttons()

    def close_training_tab(self) -> None:
        """Remove the Training tab and return to the mission tab."""
        idx = self._find_workspace_tab_index(self.training_panel)
        if idx >= 0:
            self.workspace_tabs.removeTab(idx)
            self._training_tab_index = -1
            self._refresh_workspace_tab_close_buttons()
        self.open_mission_tab()

    def open_mission_tab(self) -> None:
        idx = self._find_workspace_tab_index(self.mission_zone)
        if idx >= 0:
            self.workspace_tabs.setCurrentIndex(idx)

    def set_workflow_tab_title(self, title: str) -> None:
        idx = self._find_workspace_tab_index(self.mission_zone)
        if idx >= 0:
            display_title = title or "[New File]"
            self.workspace_tabs.setTabText(idx, display_title)

    def _on_workspace_tab_close_requested(self, index: int) -> None:
        page = self.workspace_tabs.widget(index)
        _permanent = {self.mission_zone, self.training_panel}
        if page in _permanent:
            return
        self.workspace_tabs.removeTab(index)
        self.open_mission_tab()
        self._refresh_workspace_tab_close_buttons()

    def _on_compiler_tab_close_requested(self, index: int) -> None:
        if index == 0:
            return  # Main Workflow tab is permanent
        page = self.compiler_tabs.widget(index)
        for node_id, editor in list(self._script_editor_tabs.items()):
            if editor is page:
                del self._script_editor_tabs[node_id]
                break
        self.compiler_tabs.removeTab(index)

    def open_behavior_tab(
        self,
        node_name: str,
        node_id: int,
        behavior_ref: str = "",
        intent_id: str = "",
    ):
        """Switch workspace to Behavior tab and load Mission node context."""
        self.behavior_panel.set_node_context(
            node_name,
            node_id,
            behavior_ref=behavior_ref,
            intent_id=intent_id,
        )
        idx = self._ensure_workspace_tab("behavior", title_override=node_name)
        if idx >= 0:
            self.workspace_tabs.setCurrentIndex(idx)

    def open_script_tab(
        self,
        node_id: int,
        script_name: str,
        io_spec: dict,
        node_item=None,
        graph_scene=None,
    ) -> None:
        """Open (or focus) a Script editor tab inside the Compiler zone."""
        from bin.pages.canvas.script_editor import ScriptEditor

        # If already open, switch to it
        existing = self._script_editor_tabs.get(node_id)
        if existing is not None:
            idx = self.compiler_tabs.indexOf(existing)
            if idx >= 0:
                if script_name:
                    self.compiler_tabs.setTabText(idx, script_name)
                    if hasattr(existing, "set_script_name"):
                        existing.set_script_name(script_name)
                self._ensure_compiler_visible()
                self.compiler_tabs.setCurrentIndex(idx)
                return
            # Stale entry — tab was closed without going through close handler
            del self._script_editor_tabs[node_id]

        editor = ScriptEditor(node_id, script_name, io_spec)

        # Wire Save & Compile → graph_scene full IO update (inputs + outputs).
        if graph_scene is not None:
            editor.compile_requested.connect(
                lambda nid, result, gs=graph_scene: gs.script_update_io(nid, result)
            )

        # Wire Canvas port changes → signature update in editor
        if node_item is not None:
            node_item._on_io_spec_updated = lambda spec, ed=editor: ed.update_signature(spec)

        self._script_editor_tabs[node_id] = editor
        idx = self.compiler_tabs.addTab(editor, script_name)
        self._ensure_compiler_visible()
        self.compiler_tabs.setCurrentIndex(idx)
        # Initial compile to sync default template outputs (e.g. init()->return result)
        QTimer.singleShot(0, lambda ed=editor: ed.save_and_compile(show_dialogs=False))

    def rename_script_tab(self, node_id: int, script_name: str) -> None:
        """Rename an open Script editor tab by node id (no-op if tab not open)."""
        editor = self._script_editor_tabs.get(node_id)
        if editor is None:
            return
        idx = self.compiler_tabs.indexOf(editor)
        if idx < 0:
            return
        safe_name = str(script_name or "").strip() or f"script_{node_id}"
        self.compiler_tabs.setTabText(idx, safe_name)
        if hasattr(editor, "set_script_name"):
            editor.set_script_name(safe_name)

    def close_script_tab(self, node_id: int) -> None:
        """Close the script editor tab for *node_id* in the Compiler zone."""
        editor = self._script_editor_tabs.pop(node_id, None)
        if editor is None:
            return
        idx = self.compiler_tabs.indexOf(editor)
        if idx >= 0:
            self.compiler_tabs.removeTab(idx)

    def set_compiler_main_tab_title(self, filename: str) -> None:
        """Update the Main Workflow tab title to reflect the current file name."""
        title = f"[{filename or 'New File'}]: Main Workflow"
        self.compiler_tabs.setTabText(0, title)

    def _ensure_compiler_visible(self) -> None:
        """Expand the compiler zone if currently collapsed (canvas-fullscreen mode)."""
        if self._layout_mode == "canvas":
            self._apply_default_layout()
            self._save_layout_to_config()

    def open_settings_tab(self) -> None:
        self.start_settings_dialog.show()
        self.start_settings_dialog.raise_()
        self.start_settings_dialog.activateWindow()

    def open_capabilities_tab(self) -> None:
        self.open_settings_tab()

    def _init_workspace_tab_shortcuts(self):
        """Bind workspace tab-switch shortcuts from user.ini [SHORTCUTS]."""
        defaults = {
            "workspace_prev_tab": "Left",
            "workspace_next_tab": "Right",
        }
        changed = False
        resolved = {}
        for key, default in defaults.items():
            value = str(self.config.get("SHORTCUTS", key, fallback="", config_type="user") or "").strip()
            if not value:
                value = default
                self.config.set("SHORTCUTS", key, value, config_type="user")
                changed = True
            resolved[key] = value
        if changed:
            self.config.save_user_config()

        self._workspace_shortcuts = []
        prev_seq = resolved.get("workspace_prev_tab", "Left")
        next_seq = resolved.get("workspace_next_tab", "Right")

        prev_shortcut = QShortcut(QKeySequence(prev_seq), self)
        prev_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        prev_shortcut.activated.connect(self._activate_prev_workspace_tab)
        self._workspace_shortcuts.append(prev_shortcut)

        next_shortcut = QShortcut(QKeySequence(next_seq), self)
        next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        next_shortcut.activated.connect(self._activate_next_workspace_tab)
        self._workspace_shortcuts.append(next_shortcut)

    def _is_text_editing_focus(self) -> bool:
        """Avoid stealing Left/Right when user is editing text."""
        focus_widget = QApplication.focusWidget()
        return isinstance(
            focus_widget,
            (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox),
        )

    def _activate_prev_workspace_tab(self):
        if self._is_text_editing_focus():
            return
        count = self.workspace_tabs.count()
        if count <= 1:
            return
        idx = self.workspace_tabs.currentIndex()
        self.workspace_tabs.setCurrentIndex((idx - 1) % count)

    def _activate_next_workspace_tab(self):
        if self._is_text_editing_focus():
            return
        count = self.workspace_tabs.count()
        if count <= 1:
            return
        idx = self.workspace_tabs.currentIndex()
        self.workspace_tabs.setCurrentIndex((idx + 1) % count)

    def activate_runtime_fullscreen(self):
        """Compatibility no-op; runtime zone was removed by design."""
        return

    # ------------------------------------------------------------------
    # Execution Summary Bar (Cycle 1 STAGE-03)
    # ------------------------------------------------------------------

    def show_execution_summary(self, run_result: dict) -> None:
        """Populate and show the execution summary bar from a RuntimeEngine result dict.

        Args:
            run_result: The dict returned by RuntimeEngine.execute().
        """
        status = run_result.get("status", "")
        node_count = run_result.get("node_count", 0)
        diag = run_result.get("diagnostics", {})
        executed_count = diag.get("executed_count", len(run_result.get("results", {})))
        failed_nodes = diag.get("failed_nodes", [])
        compat_path = diag.get("compat_path", False)
        reason = run_result.get("reason", "")

        # Icon and colour
        if status == "success":
            icon = "?"
            icon_color = "#22c55e"
        elif status == "failed":
            icon = "?"
            icon_color = "#ef4444"
        else:  # blocked or unknown
            icon = "?"
            icon_color = "#f59e0b"

        self._exec_status_icon.setText(icon)
        self._exec_status_icon.setStyleSheet(
            f"color: {icon_color}; font-weight: bold; background: transparent;"
        )

        # Main summary text
        failed_part = f" | {len(failed_nodes)} failed" if failed_nodes else ""
        reason_part = f" ({reason})" if reason and status != "success" else ""
        summary = (
            f"{executed_count}/{node_count} nodes executed"
            f"{failed_part}{reason_part}"
        )
        self._exec_summary_label.setText(summary)
        self._exec_summary_label.setStyleSheet("background: transparent;")

        # Compat path warning
        if compat_path:
            compat_reason = diag.get("compat_reason", "compat gate active")
            self._exec_compat_label.setText(f"[compat: {compat_reason}]")
            self._exec_compat_label.setStyleSheet(
                "color: #f59e0b; font-style: italic; background: transparent;"
            )
            self._exec_compat_label.setVisible(True)
        else:
            self._exec_compat_label.setVisible(False)

        # Diagnostics UI is disabled; never expose the drill-down entry point.
        self._details_btn.setVisible(False)

        self._exec_summary_bar.setVisible(True)

    def clear_execution_summary(self) -> None:
        """Hide the execution summary bar and clear its content."""
        self._exec_summary_bar.setVisible(False)
        self._exec_summary_label.setText("")
        self._exec_status_icon.setText("")
        self._exec_compat_label.setVisible(False)
        self._details_btn.setVisible(False)
        self.diag_panel.clear()

    def clear_compat_report(self) -> None:
        """Hide compat-only summary state without touching execution diagnostics."""
        self._exec_compat_label.setText("")
        self._exec_compat_label.setVisible(False)

    # ------------------------------------------------------------------
    # Diagnostics Panel (Cycle 1 STAGE-04)
    # ------------------------------------------------------------------

    def show_diagnostics_panel(self, node_infos: list, start_index: int = 0) -> None:
        """Show the diagnostics panel populated with *node_infos*.

        Args:
            node_infos:  List from error_ux.extract_failed_nodes_info().
            start_index: Which node to display first.
        """
        _ = (node_infos, start_index)
        return


    def show_compat_report(self, report, compat_diag_dicts: list = None) -> None:
        """Surface compatibility audit results in the summary bar and diagnostics panel.

        Called by :meth:`MainWindow._run_compat_audit` on every model-switch event
        so that compat issues are visible in the main zone — not just the
        BehaviorPanel sidebar.  BehaviorDiagnostic-formatted dicts are routed
        through :class:`DiagnosticsPanel` to satisfy the IR semantic diagnostics
        contract (``DiagnosticsKey.COMPAT_DIAGNOSTICS``).

        Parameters
        ----------
        report           : ``HBCompatReport`` from ``audit_catalog_compatibility()``.
        compat_diag_dicts: ``BehaviorDiagnostic.to_dict()`` list produced by
                           ``report_to_behavior_diagnostics(report)``.
                           Pass ``[]`` or ``None`` when ``report.is_clean()``.
        """
        if report.is_clean():
            self._exec_compat_label.setText("")
            self._exec_compat_label.setVisible(False)
            return

        summary = report.summary_text()
        color = "#f44336" if report.has_errors else "#ff9800"
        self._exec_compat_label.setText(f"[compat: {summary}]")
        self._exec_compat_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._exec_compat_label.setVisible(True)

        if not self._exec_summary_bar.isVisible():
            self._exec_summary_label.setText(summary)
            self._exec_summary_bar.setVisible(True)

        # Route BehaviorDiagnostic dicts into DiagnosticsPanel as node_infos so
        # that compat issues share the same aggregated diagnostic surface as
        # execution failures (IR semantic diagnostics pipeline).
        node_infos = []
        for d in (compat_diag_dicts or []):
            code = str(d.get("code", "") or "")
            location = str(d.get("location", "") or "")
            # sensor_read is not implemented yet; suppress its limited-capability
            # compat warning in the canvas diagnostics surface until the node lands.
            if code == "compat.limited.sensor_read" and location == "sensor_read":
                continue
            node_infos.append({
                "node_name": location or code or "?",
                "node_id":   location,
                "status":    d.get("level", "error"),
                "reason":    d.get("message", ""),
                "code":      code,
            })
        if node_infos:
            self._details_btn.setVisible(False)


    def clear_diagnostics_panel(self) -> None:
        """Hide and clear the diagnostics panel."""
        return

    def _on_details_clicked(self) -> None:
        """Toggle diagnostics panel visibility from summary bar button."""
        return




    def _on_diag_panel_closed(self) -> None:
        """Reset the Details button state when the panel is closed."""
        self._details_btn.setVisible(False)

    def _open_scenario_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("ui.mujoco_settings_title", "MuJoCo Settings"))
        dialog.setModal(True)
        dialog.resize(420, 280)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        backend_combo = QComboBox(dialog)
        backend_combo.addItems(["glfw", "egl", "osmesa"])
        backend_combo.setCurrentText(self._scenario_settings.get("mujoco_gl_backend", "glfw"))
        form.addRow(tr("ui.mujoco_gl_backend", "GL Backend"), backend_combo)

        realtime_check = QCheckBox(tr("ui.mujoco_realtime", "Enable realtime pacing"), dialog)
        realtime_check.setChecked(bool(self._scenario_settings.get("mujoco_realtime", True)))
        form.addRow(tr("ui.mujoco_timing", "Timing"), realtime_check)

        timestep_spin = QDoubleSpinBox(dialog)
        timestep_spin.setDecimals(4)
        timestep_spin.setRange(0.0005, 0.0500)
        timestep_spin.setSingleStep(0.0005)
        timestep_spin.setSuffix(" s")
        timestep_spin.setValue(float(self._scenario_settings.get("mujoco_timestep", 0.0020)))
        form.addRow(tr("ui.mujoco_timestep", "Timestep"), timestep_spin)

        scene_edit = QLineEdit(dialog)
        scene_edit.setPlaceholderText(tr("ui.mujoco_scene_placeholder", "Optional scene XML path"))
        scene_edit.setText(str(self._scenario_settings.get("mujoco_scene_xml", "")))
        form.addRow(tr("ui.mujoco_scene_xml", "Scene XML"), scene_edit)

        # Cycle 3 STAGE-06: advanced MuJoCo settings (bounded subset)
        gravity_spin = QDoubleSpinBox(dialog)
        gravity_spin.setDecimals(2)
        gravity_spin.setRange(-50.0, 0.0)
        gravity_spin.setSingleStep(0.1)
        gravity_spin.setSuffix(" m/s2")
        gravity_spin.setValue(float(self._scenario_settings.get("mujoco_gravity_z", -9.81)))
        form.addRow(tr("ui.mujoco_gravity_z", "Gravity Z"), gravity_spin)

        solver_spin = QSpinBox(dialog)
        solver_spin.setRange(1, 1000)
        solver_spin.setValue(int(self._scenario_settings.get("mujoco_solver_iterations", 100)))
        form.addRow(tr("ui.mujoco_solver_iterations", "Solver Iterations"), solver_spin)

        render_w_spin = QSpinBox(dialog)
        render_w_spin.setRange(64, 4096)
        render_w_spin.setSingleStep(64)
        render_w_spin.setValue(int(self._scenario_settings.get("mujoco_render_width", 640)))
        form.addRow(tr("ui.mujoco_render_width", "Render Width (px)"), render_w_spin)

        render_h_spin = QSpinBox(dialog)
        render_h_spin.setRange(64, 4096)
        render_h_spin.setSingleStep(64)
        render_h_spin.setValue(int(self._scenario_settings.get("mujoco_render_height", 480)))
        form.addRow(tr("ui.mujoco_render_height", "Render Height (px)"), render_h_spin)

        layout.addLayout(form)
        layout.addStretch()

        buttons_row = QHBoxLayout()
        buttons_row.addStretch()
        confirm_btn = QPushButton(tr("ui.confirm", "Confirm"), dialog)
        cancel_btn = QPushButton(tr("ui.cancel", "Cancel"), dialog)
        cancel_btn.clicked.connect(dialog.reject)

        def _save_and_close():
            self._scenario_settings = {
                "mujoco_gl_backend": backend_combo.currentText().strip() or "glfw",
                "mujoco_realtime": bool(realtime_check.isChecked()),
                "mujoco_timestep": float(timestep_spin.value()),
                "mujoco_scene_xml": scene_edit.text().strip(),
                # Advanced fields (STAGE-06)
                "mujoco_gravity_z": float(gravity_spin.value()),
                "mujoco_solver_iterations": int(solver_spin.value()),
                "mujoco_render_width": int(render_w_spin.value()),
                "mujoco_render_height": int(render_h_spin.value()),
            }
            dialog.accept()

        confirm_btn.clicked.connect(_save_and_close)
        buttons_row.addWidget(confirm_btn)
        buttons_row.addWidget(cancel_btn)
        layout.addLayout(buttons_row)

        dialog.exec()

    def get_scenario_settings(self) -> dict:
        """Collect current Scenario/MuJoCo settings."""
        return dict(self._scenario_settings)

    def set_scenario_settings(self, settings: dict) -> None:
        """Restore Scenario/MuJoCo settings (e.g., from a loaded mission file).

        Only keys that belong to the known scenario settings namespace are applied
        (mujoco_* prefix).  Unknown keys are silently ignored for forward compat.
        """
        for key, val in settings.items():
            if key.startswith("mujoco_"):
                self._scenario_settings[key] = val

    # ------------------------------------------------------------------
    # SDK Settings Panel (Cycle 2 STAGE-03)
    # ------------------------------------------------------------------

    def get_sdk_settings(self) -> dict:
        """Return the current SDK settings config from the settings panel."""
        return self.settings_panel.get_current_config()

    def set_sdk_brand(self, brand: str, config: dict = None) -> None:
        """Rebuild the settings panel form for *brand* with optional *config*."""
        self.settings_panel.set_brand(brand, config or {})

    # ------------------------------------------------------------------
    # Capability Inspector (Cycle 2 STAGE-04)
    # ------------------------------------------------------------------

    def set_adapter_capabilities(
        self,
        cap_dict: dict,
        missing_settings: list = None,
    ) -> None:
        """Reload the capability inspector with *cap_dict*.

        Args:
            cap_dict:         Dict from ``adapter.capabilities()``.
            missing_settings: Optional list of field_ids not yet filled.
        """
        self.capability_inspector.set_capabilities(cap_dict, missing_settings or [])

    def _on_capability_focus_setting(self, field_id: str) -> None:
        """Open the Start settings popup and scroll to *field_id*."""
        self.open_settings_tab()
        self.settings_panel.scroll_to_field(field_id)

    def refresh_texts(self):
        """Refresh i18n texts in this panel."""
        self._apply_mission_control_icons()
        self.scenario_settings_btn.setToolTip(tr("ui.mujoco_setting", "Mujoco Setting"))
        if hasattr(self, "behavior_panel"):
            self.behavior_panel.refresh_texts()

    def refresh_style(self) -> None:
        """Refresh theme-driven styles for dynamic panels owned by MainZonePanel."""
        self._sync_zone_toggle_buttons()
        if hasattr(self, "code_editor"):
            self.code_editor.refresh_style()
        for editor in list(getattr(self, "_script_editor_tabs", {}).values()):
            if hasattr(editor, "refresh_style"):
                editor.refresh_style()



























