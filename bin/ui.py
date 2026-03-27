#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main UI Module
Contains MainWindow and main UI components
"""

from PySide6.QtCore import Qt, QTimer, QUrl, QEasingCurve, QPropertyAnimation, QEvent
from PySide6.QtGui import QDesktopServices, QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QLabel, QComboBox, QMessageBox, QGraphicsOpacityEffect,
    QFileDialog, QPushButton, QStackedWidget, QSizePolicy, QTextEdit,
)
from PySide6.QtSvgWidgets import QSvgWidget
import inspect
import json
import os
from pathlib import Path

from bin.layout import MainZonePanel
from bin.components import SidebarDock
from bin.components.sidebar_dock import ProjectFilesPanel
from bin.scenario import ScenarioPanelState
from system.runtime import RuntimeEngine
from bin.core.simulation_thread import SimulationThread
from bin.core.mission_run_thread import MissionRunThread  # Cycle 2 STAGE-06
from bin.core.config_manager import ConfigManager
from bin.core.data_manager import get_value, load_data, up_data
from bin.core.theme_manager import get_color, get_color_slot, get_font_size, set_theme
from bin.core.logger import CmdLogWidget, get_log_signal, log_info, log_success, log_warning, log_error, log_debug
from bin.core.localisation import get_localisation, tr
from bin.core.robot_context import RobotContext
from bin.components.misc import MainRow
from bin.components.homepage import WindowControlButtons
from system.behavior.action_registry import get_semantic_actions



class MainWindow(QMainWindow):
    """Main Window"""

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.config = config
        self.robot_model = None
        self.simulation_thread = None
        self._runtime_paused = False
        self._current_workflow_path = ""
        # Cycle 2 STAGE-06: background mission run thread + cached exec_graph.
        self._mission_run_thread = None
        # AppShell: single Training Ground page (created lazily)
        self._training_page = None
        self._nav_busy = False
        self._fade_pending_cb = None  # currently-connected finished callback
        self._last_exec_graph: dict = {}
        self._last_runtime_target = ""
        self._last_runtime_backend = ""
        self.runtime_engine = RuntimeEngine()
        self.scenario_state = ScenarioPanelState()
        self.project_files_panel = None

        # Circle 2: shared BehaviorCompilerBridge 鈥?owned here, injected into
        # both the RuntimeEngine (for mission-time behavior dispatch) and the
        # HBChannelFactory (for HB panel compile/run).
        from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
        self._behavior_bridge = BehaviorCompilerBridge()
        self.runtime_engine.behavior_bridge = self._behavior_bridge

        # Circle 3: last-loaded package metadata; round-trips through save/load
        # and surfaces in execution diagnostics as package_metadata_trace.
        self._loaded_package_metadata: dict = {}

        # Load UI config
        self._load_ui_config()

        self._init_ui()
        self._init_statusbar()
        # Keep startup behavior consistent with a manual theme toggle:
        # ensure all nested components (especially GraphScene node widgets)
        # receive a full theme refresh on first render.
        self._refresh_theme()

        log_info(tr("log.main_window_init", "Main window initialized"))

    def _prepare_behavior_artifacts_for_run(self, exec_graph: dict, robot_type: str) -> bool:
        """Compile/register all mission behavior artifacts under their internal refs."""
        panel = self.main_zone.behavior_panel
        failures = []
        brand = RobotContext.get_current_brand()

        for node_data in (exec_graph.get("nodes") or {}).values():
            if not isinstance(node_data, dict):
                continue
            params = node_data.get("parameters") or {}
            node_type = str(node_data.get("type") or "").strip().lower()
            behavior_ref = str(params.get("behavior_ref") or "").strip()
            if node_type not in {"behavior_call", "behavior"} and not behavior_ref:
                continue

            node_id = int(node_data.get("id") or 0)
            display_name = str(node_data.get("ui_selection") or node_data.get("name") or "Behavior").strip()
            if not behavior_ref:
                behavior_ref = f"behavior_node_{node_id}"
            intent_id = str(params.get("behavior_intent_id") or node_data.get("behavior_intent_id") or "").strip()

            try:
                artifact = panel.ensure_behavior_artifact(
                    node_name=display_name,
                    node_id=node_id,
                    behavior_ref=behavior_ref,
                    intent_id=intent_id,
                    brand=brand,
                    robot_type=robot_type or "go2",
                    is_simulation=True,
                )
            except Exception as exc:
                failures.append(f"node {node_id}: compile error: {exc}")
                continue

            if not bool(getattr(artifact, "is_valid", False)):
                failures.append(f"node {node_id}: artifact {behavior_ref!r} is invalid")

        if failures:
            message = "\n".join(failures)
            log_error(f"Behavior artifact preparation failed:\n{message}")
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                f"Behavior artifact preparation failed.\n{message}",
            )
            return False
        return True

    def _load_ui_config(self):
        """Load UI configuration"""
        ui_config_path = self.config.project_root / "config" / "ui.ini"
        load_data(str(ui_config_path))

        # Set theme
        theme = self.config.get('PREFERENCES', 'theme', fallback='dark', config_type='user')
        theme = (theme or "dark").lower()
        if theme not in ("light", "dark"):
            theme = "dark"
        self._theme = theme
        set_theme(theme)

    def _init_ui(self):
        """Initialize UI with single-window app shell."""
        width = self.config.get_int('UI', 'window_width', fallback=1400)
        height = self.config.get_int('UI', 'window_height', fallback=900)

        self.setWindowTitle(tr("app.title", "UnitPort - Robot Visual Programming Platform"))
        self.resize(width, height)

        # App shell: single central widget holding the page stack
        shell = QWidget()
        shell.setObjectName("appShell")
        self.setCentralWidget(shell)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self._page_stack = QStackedWidget()
        shell_layout.addWidget(self._page_stack, 1)

        # Page 0: Homepage
        from bin.components.homepage import HomepageWidget
        self._homepage_page = HomepageWidget(theme=self._theme)
        self._homepage_page.mission_requested.connect(self.go_mission)
        self._homepage_page.training_requested.connect(self.go_training)
        self._homepage_page.exit_requested.connect(QApplication.quit)
        self._page_stack.addWidget(self._homepage_page)

        # Page 1: Mission Control
        self._build_mission_page()
        self._page_stack.addWidget(self._mission_page)

        # In-window fade overlay: child of shell, covers the page stack.
        # Absolute-positioned; resized in resizeEvent.
        self._fade_overlay = QWidget(shell)
        self._fade_overlay.setObjectName("pageTransitionOverlay")
        self._fade_overlay.setStyleSheet("background: #000000;")
        self._fade_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._fade_effect = QGraphicsOpacityEffect(self._fade_overlay)
        self._fade_effect.setOpacity(0.0)
        self._fade_overlay.setGraphicsEffect(self._fade_effect)
        self._fade_anim = QPropertyAnimation(self._fade_effect, b"opacity", self)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_overlay.hide()

        # ── Startup log background ────────────────────────────────────────
        # Fullscreen, lowest z-order. Acts as a dynamic loading wallpaper;
        # visible through the semi-transparent overlay above it.
        self._startup_log_bg = QWidget(shell)
        self._startup_log_bg.setObjectName("startupLogBg")
        _log_bg_layout = QVBoxLayout(self._startup_log_bg)
        _log_bg_layout.setContentsMargins(0, 0, 0, 0)
        _log_bg_layout.setSpacing(0)
        self._startup_log = QTextEdit(self._startup_log_bg)
        self._startup_log.setObjectName("startupLogDisplay")
        self._startup_log.setReadOnly(True)
        self._startup_log.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._startup_log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._startup_log.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        _log_bg_layout.addWidget(self._startup_log)
        self._startup_log_bg.setGeometry(shell.rect())
        # Do NOT raise_() — keep behind the overlay

        # ── Startup loading overlay ───────────────────────────────────────
        # Semi-transparent (85 % opacity) dark layer above the log; shows logo only.
        self._startup_loading_overlay = QWidget(shell)
        self._startup_loading_overlay.setObjectName("startupLoadingOverlay")
        startup_loading_layout = QVBoxLayout(self._startup_loading_overlay)
        startup_loading_layout.setContentsMargins(0, 0, 0, 0)
        startup_loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._startup_loading_logo = QSvgWidget("assets/icon/logo_w.svg", self._startup_loading_overlay)
        logo_size = self._startup_loading_logo.renderer().defaultSize()
        logo_height = 200
        logo_width = logo_height
        if logo_size.height() > 0:
            logo_width = max(1, int(logo_size.width() * (logo_height / logo_size.height())))
        self._startup_loading_logo.setFixedSize(logo_width, logo_height)
        startup_loading_layout.addWidget(
            self._startup_loading_logo,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self._startup_loading_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._startup_loading_overlay.setGeometry(shell.rect())
        self._startup_loading_overlay.raise_()
        get_log_signal().log_message.connect(self._on_startup_log)

        # Start on homepage (page 0)
        self._page_stack.setCurrentIndex(0)
        self._page_stack.hide()

        self._apply_stylesheet()
        self._set_current_workflow_path("")
        log_debug(tr("log.ui_layout_created", "UI layout created"))
        log_info(tr("log.graph_editor_ready", "Graph editor ready, drag modules from left panel to canvas"))
        self._refresh_policy_registry_ui()
        QTimer.singleShot(0, self.graph_view.recenter_to_origin)

    def finish_startup_loading(self) -> None:
        try:
            get_log_signal().log_message.disconnect(self._on_startup_log)
        except Exception:
            pass
        if hasattr(self, "_page_stack") and self._page_stack is not None:
            self._page_stack.show()
        if hasattr(self, "_startup_loading_overlay") and self._startup_loading_overlay is not None:
            self._startup_loading_overlay.hide()
        if hasattr(self, "_startup_log_bg") and self._startup_log_bg is not None:
            self._startup_log_bg.hide()

    _STARTUP_LOG_COLORS = {
        "debug": "#8B8B8B",
        "info": "#FFFFFF",
        "success": "#00D26A",
        "warning": "#FFC107",
        "error": "#FF6B6B",
    }

    def _on_startup_log(self, text: str, log_type: str, wrap: bool, typer: bool) -> None:
        """Append incoming log messages to the startup overlay log display."""
        if not hasattr(self, "_startup_log") or self._startup_log is None:
            return
        color = self._STARTUP_LOG_COLORS.get(log_type, "#FFFFFF")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self._startup_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if cursor.position() > 0:
            cursor.insertText("\n", QTextCharFormat())
        cursor.insertText(text, fmt)
        self._startup_log.setTextCursor(cursor)
        self._startup_log.ensureCursorVisible()

    def run_startup_prewarm(self) -> None:
        self._prime_training_shell_cache("")

    def _build_mission_page(self):
        """Build the Mission Control page widget and store as self._mission_page."""
        self._mission_page = QWidget()
        self._mission_page.setObjectName("missionPage")
        main_layout = QHBoxLayout(self._mission_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: fixed sidebar rail + contextual slide-out content panel
        self.sidebar = SidebarDock(
            nav_items=[
                ("user", "User", "icon_acc", "assets/icon/icon_acc.svg"),
                ("projects", "Project Files", "icon_prj", "assets/icon/icon_prj.svg"),
                ("nodes", "Nodes", "icon_nod", "assets/icon/icon_nod.svg"),
            ],
            config=self.config,
        )
        self.sidebar.panel_requested.connect(self._on_sidebar_panel_requested)
        self.sidebar.theme_toggle_requested.connect(self._on_theme_toggle)
        self.sidebar.language_toggle_requested.connect(self._on_language_button_clicked)
        self.theme_button = self.sidebar.theme_button
        self.language_button = self.sidebar.language_button
        self._sync_theme_button()
        main_layout.addWidget(self.sidebar)

        self.mission_shell = QWidget()
        mission_shell_layout = QVBoxLayout(self.mission_shell)
        mission_shell_layout.setContentsMargins(0, 0, 0, 0)
        mission_shell_layout.setSpacing(0)
        main_layout.addWidget(self.mission_shell, 1)

        self.mission_nav_header = MainRow(theme=self._theme)
        self.mission_nav_header.page_selected.connect(self._on_page_switcher_selected)
        self._page_switcher = self.mission_nav_header.page_switcher

        self._mc_nav_content = QWidget()
        mc_nav_layout = QHBoxLayout(self._mc_nav_content)
        mc_nav_layout.setContentsMargins(0, 0, 0, 0)
        mc_nav_layout.setSpacing(0)
        self.window_controls = WindowControlButtons(theme=self._theme, parent=self._mc_nav_content)
        self.window_controls.minimize_requested.connect(self.showMinimized)
        self.window_controls.fullscreen_requested.connect(self._toggle_window_fullscreen)
        self.window_controls.close_requested.connect(self.close)
        mc_nav_layout.addWidget(self.window_controls)
        self.mission_nav_header.set_right_widget(self._mc_nav_content)

        mission_shell_layout.addWidget(self.mission_nav_header)

        # Shared workspace stack (left pane of the shell splitter)
        self._shell_content_stack = QStackedWidget()
        self._shell_content_stack.setObjectName("shellContentStack")
        self._mission_content_placeholder = QWidget()
        self._shell_content_stack.addWidget(self._mission_content_placeholder)  # index 0
        self._tg_content_placeholder = QWidget()                          # index 1 placeholder
        self._shell_content_stack.addWidget(self._tg_content_placeholder)
        self._shell_content_stack.setCurrentIndex(0)
        self._cmd_collapsed = False
        self._cmd_last_width = 340

        # Right: built-in CMD console
        self.cmd_log = CmdLogWidget()
        self.cmd_log.setMinimumWidth(0)

        # Main and right-zone splitter shared by Mission + Training
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self._shell_content_stack)
        self.main_splitter.addWidget(self.cmd_log)
        self.main_splitter.setSizes([1260, self._cmd_last_width])
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        mission_shell_layout.addWidget(self.main_splitter, 1)

        # Main zone: delegated to dedicated panel component
        self.main_zone = MainZonePanel()
        if hasattr(self.main_zone, "navigation_header"):
            self.main_zone.navigation_header.setMaximumHeight(0)
        self.main_zone.behavior_panel.set_bridge(self._behavior_bridge)
        self.runtime_zone = self.main_zone.runtime_zone
        self.mission_zone = self.main_zone.mission_zone
        self.tool_row = self.main_zone.tool_row
        self.module_palette = self.main_zone.module_palette
        self.graph_scene = self.main_zone.graph_scene
        self.graph_view = self.main_zone.graph_view
        self.code_editor = self.main_zone.code_editor
        self.module_palette.set_embedded_mode(True)
        self.project_files_panel = ProjectFilesPanel(self._workflows_root())
        self.project_files_panel.file_activated.connect(self._open_workflow_from_path)
        self.project_files_panel.load_requested.connect(self._on_open)
        self.project_files_panel.open_folder_requested.connect(self._open_workflows_folder)
        self.sidebar.set_panel_widget("projects", self.project_files_panel, "Project Files")
        self.sidebar.set_panel_widget("nodes", self.module_palette, "Node Library")
        self.graph_scene.set_subgraph_opener(self._open_nested_editor)
        self.graph_scene.set_script_tab_closer(self.main_zone.close_script_tab)
        self.graph_scene.set_script_tab_renamer(self.main_zone.rename_script_tab)
        self.graph_scene.set_behavior_param_provider(
            self.main_zone.behavior_panel.get_node_exposed_action_parameters
        )
        self.graph_scene.set_behavior_param_writer(
            self.main_zone.behavior_panel.update_node_exposed_action_parameter
        )
        self.main_zone.behavior_panel.action_parameters_changed.connect(
            self.graph_scene.refresh_behavior_node_parameter_rows
        )
        self.module_palette.node_requested.connect(self._on_node_requested)
        self.graph_scene.robot_type_changed.connect(self._on_robot_type_changed)
        self.graph_scene.workspace_open_requested.connect(self._on_open_training_workspace)
        self.main_zone.start_requested.connect(self._on_run)
        self.main_zone.training_requested.connect(self.go_training)
        self.main_zone.exit_requested.connect(self.close)
        self.main_zone.pause_requested.connect(self._on_runtime_pause)
        self.main_zone.abort_requested.connect(self._on_runtime_abort)
        self.main_zone.reset_requested.connect(self._on_runtime_reset)
        self.main_zone.workflow_load_requested.connect(self._on_open)
        self.main_zone.workflow_save_requested.connect(self._on_save)
        self.main_zone.workflow_save_as_requested.connect(self._on_save_as)
        self.main_zone.workflow_browser_requested.connect(self._open_project_files_sidebar)
        self.main_zone.navigate_to_node.connect(self._on_navigate_to_node)
        # Capability inspector refresh on settings change (Cycle 2 STAGE-04)
        self.main_zone.settings_panel.settings_applied.connect(self._on_sdk_settings_changed)
        self.main_zone.settings_panel.settings_reset.connect(self._refresh_capability_inspector)
        self._capability_refresh_timer = QTimer(self)
        self._capability_refresh_timer.setSingleShot(True)
        self._capability_refresh_timer.timeout.connect(self._refresh_capability_inspector)
        self.main_zone.settings_panel.settings_changed.connect(
            lambda _cfg: self._capability_refresh_timer.start(250)
        )

        old_mission_content = self._shell_content_stack.widget(0)
        if old_mission_content is not self.main_zone:
            self._shell_content_stack.removeWidget(old_mission_content)
            self._shell_content_stack.insertWidget(0, self.main_zone)
        self._shell_content_stack.setCurrentIndex(0)

        # Store MC sidebar panel widgets and icon bindings for later restoration
        self._mc_sidebar_user_widget = self.sidebar._panel_meta.get("user", {}).get("widget")
        self._mc_sidebar_icon_bindings = dict(self.sidebar._nav_icon_bindings)
        self._shell_mode = "mission"

    # ------------------------------------------------------------------
    # App shell navigation
    # ------------------------------------------------------------------

    def go_home(self) -> None:
        """Navigate to the Homepage."""
        self._navigate_to_page(0)

    def go_mission(self) -> None:
        """Navigate to Mission Control (restores MC mode if in TG mode)."""
        from_home = self._page_stack.currentIndex() == 0
        self._enter_mission_mode()
        if hasattr(self, "main_zone"):
            self.main_zone.open_mission_tab()
        # Auto-load last mission file when coming from homepage and canvas is empty
        if from_home and not self._current_workflow_path:
            last_file = self._load_last_mission_file()
            if last_file and os.path.isfile(last_file):
                self._load_workflow_from_path(last_file)
        self._navigate_to_page(1)

    def go_training(self, policy_id: str = "") -> None:
        """Switch to Training Ground for *policy_id* inside the shared shell."""
        policy_id = str(policy_id or "")
        self._enter_training_mode(policy_id)
        self._navigate_to_page(1)

    def _on_page_switcher_selected(self, page: str) -> None:
        page = str(page or "").strip().lower()
        if page == "training":
            self.go_training()
        else:
            self.go_mission()

    # ------------------------------------------------------------------
    # Shell mode switching (MC 鈫?TG within page 1)
    # ------------------------------------------------------------------

    def _ensure_training_content(self, policy_id: str) -> None:
        """Build (or rebuild) the embedded Training Ground widget.

        Empty policy_id means "keep whatever training page is current" — existing
        page is always preserved when no specific policy is requested.  A non-empty
        policy_id that differs from the current one triggers a rebuild.
        """
        if self._training_page is not None:
            # No specific policy requested → preserve current training state
            if not policy_id:
                return
            # Same policy already loaded → nothing to do
            if getattr(self._training_page, '_policy_id', None) == policy_id:
                return
            # Different specific policy → rebuild
            idx = self._shell_content_stack.indexOf(self._training_page)
            if idx >= 0:
                self._shell_content_stack.removeWidget(self._training_page)
            self._training_page.deleteLater()
            self._training_page = None

        # First build: resolve policy_id — explicit > last saved > ""
        if not policy_id:
            policy_id = self._load_last_training_policy()

        from bin.components.training_workspace_window import TrainingWorkspaceWindow
        page = TrainingWorkspaceWindow(
            policy_id,
            initial_robot_type=RobotContext.get_robot_type(),
            initial_runtime_scenario=self._get_runtime_scenario_settings(),
            embedded=True,
        )
        page.mission_control_requested.connect(self.go_mission)
        page.exit_requested.connect(self.close)
        page.checkpoint_exported.connect(
            lambda bp, pid=policy_id: self._on_checkpoint_exported(bp, pid)
        )
        self._training_page = page
        if policy_id:
            self._save_last_training_policy(policy_id)

    def _prime_training_shell_cache(self, policy_id: str = "") -> None:
        """Prebuild hidden Training Ground UI so the first page switch is instant."""
        self._ensure_training_content(policy_id)
        page = self._training_page
        if page is None:
            return

        old_tg_content = self._shell_content_stack.widget(1)
        if old_tg_content is not page:
            self._shell_content_stack.removeWidget(old_tg_content)
            self._shell_content_stack.insertWidget(1, page)

        if hasattr(page, "apply_theme"):
            try:
                page.apply_theme()
            except Exception:
                pass
        if hasattr(page, "warm_cache"):
            try:
                page.warm_cache()
            except Exception:
                pass

    def _enter_training_mode(self, policy_id: str) -> None:
        """Switch the shared mission shell to Training Ground content."""
        # Build or reuse TG widget
        self._ensure_training_content(policy_id)
        page = self._training_page

        # --- Sidebar panels: swap to TG panels ---
        self.sidebar.set_panel_widget("user", page._canvas_browser_panel, "Experiments")
        self.sidebar.set_panel_widget("projects", page._export_browser_panel, "Exports")
        self.sidebar.set_panel_widget("nodes", page._palette_panel, "Config Nodes")
        for key, title in [("user", "Experiments"), ("projects", "Exports"), ("nodes", "Config Nodes")]:
            btn = self.sidebar._nav_buttons.get(key)
            if btn is not None:
                btn.set_title(title)
        # Update sidebar nav button icons to TG equivalents
        self.sidebar._nav_icon_bindings["user"] = ("icon_prj", "assets/icon/icon_prj.svg")
        self.sidebar._nav_icon_bindings["projects"] = ("icon_cp", "assets/icon/icon_cp.svg")
        self.sidebar._nav_icon_bindings["nodes"] = ("icon_nod", "assets/icon/icon_nod.svg")
        self.sidebar.refresh_icons(self._theme)
        # Wire sidebar panel changes to TG handler
        try:
            self.sidebar.panel_changed.disconnect(self._on_tg_sidebar_panel_changed)
        except RuntimeError:
            pass
        self.sidebar.panel_changed.connect(self._on_tg_sidebar_panel_changed)

        # --- Nav header: swap to TG header widget ---
        self.mission_nav_header.set_right_widget(page._header_widget)
        self.mission_nav_header.sync_layout()

        # --- Shell content: replace TG placeholder with TG canvas widget ---
        old_tg_content = self._shell_content_stack.widget(1)
        if old_tg_content is not page:
            self._shell_content_stack.removeWidget(old_tg_content)
            self._shell_content_stack.insertWidget(1, page)
        self._shell_content_stack.setCurrentIndex(1)

        self._shell_mode = "training"
        self._page_switcher.set_current_page("training")

        # Apply theme so TG nav header labels get correct colours immediately.
        if hasattr(page, 'apply_theme'):
            try:
                page.apply_theme()
            except Exception:
                pass

    def _enter_mission_mode(self) -> None:
        """Switch the shared mission shell back to Mission Control content."""
        if self._shell_mode == "mission":
            return

        # Persist the active training workspace policy before leaving Training Ground.
        # Prefer _workspace_policy_id (may differ from _policy_id for derived checkpoints).
        if self._training_page is not None:
            _ws_pid = (
                getattr(self._training_page, "_workspace_policy_id", "") or
                getattr(self._training_page, "_selected_policy_id", "")
            )
            if _ws_pid:
                self._save_last_training_policy(_ws_pid)

        # --- Sidebar panels: restore MC panels ---
        if self._mc_sidebar_user_widget is not None:
            self.sidebar.set_panel_widget("user", self._mc_sidebar_user_widget, "User")
        self.sidebar.set_panel_widget("projects", self.project_files_panel, "Project Files")
        self.sidebar.set_panel_widget("nodes", self.module_palette, "Node Library")
        for key, title in [("user", "User"), ("projects", "Project Files"), ("nodes", "Nodes")]:
            btn = self.sidebar._nav_buttons.get(key)
            if btn is not None:
                btn.set_title(title)
        try:
            self.sidebar.panel_changed.disconnect(self._on_tg_sidebar_panel_changed)
        except RuntimeError:
            pass
        # Restore MC nav button icons
        self.sidebar._nav_icon_bindings.update(self._mc_sidebar_icon_bindings)
        self.sidebar.refresh_icons(self._theme)

        # --- Nav header: restore MC nav content ---
        self.mission_nav_header.set_right_widget(self._mc_nav_content)
        self.mission_nav_header.sync_layout()

        # --- Shell content: restore MC splitter ---
        self._shell_content_stack.setCurrentIndex(0)
        self.main_zone.open_mission_tab()
        QTimer.singleShot(0, self.main_zone.dock_mission_control_top_right)

        self._shell_mode = "mission"
        self._page_switcher.set_current_page("mission")

    def _on_tg_sidebar_panel_changed(self, panel_key: str) -> None:
        """Forward sidebar panel-change events to the active Training Ground page."""
        if self._training_page is not None:
            self._training_page._on_sidebar_panel_changed(panel_key)

    def _navigate_to_page(self, index: int) -> None:
        """Switch to the page at index using a fade-to-black transition."""
        if self._nav_busy or self._page_stack.currentIndex() == index:
            return
        self._nav_busy = True

        shell = self.centralWidget()
        if shell:
            self._fade_overlay.setGeometry(shell.rect())
        self._fade_overlay.raise_()
        self._fade_overlay.show()

        def _finish():
            self._fade_overlay.hide()
            self._nav_busy = False
            self._fade_pending_cb = None

        def _do_switch():
            self._page_stack.setCurrentIndex(index)
            if index == 1 and getattr(self, "_shell_mode", "") == "mission" and hasattr(self, "main_zone"):
                QTimer.singleShot(0, self.main_zone.dock_mission_control_top_right)
            # Disconnect this callback, then connect the fade-out callback.
            if self._fade_pending_cb is not None:
                try:
                    self._fade_anim.finished.disconnect(self._fade_pending_cb)
                except RuntimeError:
                    pass
            self._fade_pending_cb = _finish
            self._fade_anim.setStartValue(1.0)
            self._fade_anim.setEndValue(0.0)
            self._fade_anim.setDuration(110)
            self._fade_anim.finished.connect(self._fade_pending_cb)
            self._fade_anim.start()

        # Disconnect any previously connected callback before adding the new one.
        if self._fade_pending_cb is not None:
            try:
                self._fade_anim.finished.disconnect(self._fade_pending_cb)
            except RuntimeError:
                pass
        self._fade_pending_cb = _do_switch
        self._fade_effect.setOpacity(0.0)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setDuration(90)
        self._fade_anim.finished.connect(self._fade_pending_cb)
        self._fade_anim.start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_fade_overlay') and self.centralWidget():
            self._fade_overlay.setGeometry(self.centralWidget().rect())
        if hasattr(self, '_startup_log_bg') and self.centralWidget():
            self._startup_log_bg.setGeometry(self.centralWidget().rect())
        if hasattr(self, '_startup_loading_overlay') and self.centralWidget():
            self._startup_loading_overlay.setGeometry(self.centralWidget().rect())
        self._sync_window_controls()

    # ------------------------------------------------------------------
    # Frameless window: edge-resize + title-bar drag via WM_NCHITTEST
    # ------------------------------------------------------------------

    _RESIZE_MARGIN = 6  # pixels from window edge that trigger resize cursor/op

    def nativeEvent(self, event_type, message):  # type: ignore[override]
        import sys
        if sys.platform != "win32":
            return super().nativeEvent(event_type, message)
        if event_type != b"windows_generic_MSG":
            return super().nativeEvent(event_type, message)
        try:
            import ctypes

            WM_NCHITTEST = 0x0084
            HTCLIENT      = 1
            HTCAPTION     = 2
            HTLEFT        = 10
            HTRIGHT       = 11
            HTTOP         = 12
            HTTOPLEFT     = 13
            HTTOPRIGHT    = 14
            HTBOTTOM      = 15
            HTBOTTOMLEFT  = 16
            HTBOTTOMRIGHT = 17

            class _MSG(ctypes.Structure):
                _fields_ = [
                    ("hwnd",    ctypes.c_void_p),
                    ("message", ctypes.c_uint),
                    ("wParam",  ctypes.c_size_t),
                    ("lParam",  ctypes.c_ssize_t),
                ]

            msg = ctypes.cast(int(message), ctypes.POINTER(_MSG)).contents
            if msg.message != WM_NCHITTEST:
                return super().nativeEvent(event_type, message)

            # Screen coordinates of the cursor
            x = ctypes.c_short(msg.lParam & 0xFFFF).value
            y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value

            geo = self.frameGeometry()
            m   = self._RESIZE_MARGIN

            # Edge-resize hits — only when windowed (not maximised, not fullscreen)
            if not self.isMaximized() and not self.isFullScreen():
                on_l = x < geo.left()   + m
                on_r = x > geo.right()  - m
                on_t = y < geo.top()    + m
                on_b = y > geo.bottom() - m
                if on_t and on_l: return True, HTTOPLEFT
                if on_t and on_r: return True, HTTOPRIGHT
                if on_b and on_l: return True, HTBOTTOMLEFT
                if on_b and on_r: return True, HTBOTTOMRIGHT
                if on_t: return True, HTTOP
                if on_b: return True, HTBOTTOM
                if on_l: return True, HTLEFT
                if on_r: return True, HTRIGHT

            # Caption drag — works when windowed or maximised (restore on drag)
            if not self.isFullScreen() and hasattr(self, "mission_nav_header"):
                h  = self.mission_nav_header
                g  = h.mapToGlobal(h.rect().topLeft())
                lx = x - g.x()
                ly = y - g.y()
                if 0 <= lx < h.width() and 0 <= ly < h.height():
                    child = h.childAt(lx, ly)
                    # Only treat as caption if cursor is over empty space or _left_mirror
                    mirror = getattr(h, "_left_mirror", None)
                    if child is None or child is mirror:
                        return True, HTCAPTION

            return True, HTCLIENT
        except Exception:
            pass
        return super().nativeEvent(event_type, message)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_window_controls()

    def _apply_stylesheet(self):
        """Apply stylesheet"""
        try:
            bg = get_color('bg', '#1e1e1e')
            homepage_1 = get_color('homepage_1', bg)
            homepage_2 = get_color('homepage_2', bg)
            card_bg = get_color('card_bg', '#2d2d2d')
            border = get_color('border', '#444444')
            sidebar_right_border = get_color('sidebar_right_border', border)
            top_row_bottom_border = get_color('top_row_bottom_border', border)
            text_primary = get_color('text_primary', '#ffffff')
            text_secondary = get_color('text_secondary', '#cccccc')
            hover_bg = get_color('hover_bg', '#3d3d3d')
            sidebar_rail_bg = get_color('sidebar_rail_bg', card_bg)
            sidebar_panel_bg = get_color('sidebar_panel_bg', bg)
            sidebar_button_hover_bg = get_color('sidebar_button_hover_bg', hover_bg)
            sidebar_button_checked_bg = get_color('sidebar_button_checked_bg', sidebar_button_hover_bg)
            sidebar_tooltip_bg = get_color('sidebar_tooltip_bg', card_bg)
            sidebar_tooltip_text = get_color('sidebar_tooltip_text', text_primary)
            sidebar_tooltip_border = get_color('sidebar_tooltip_border', border)
            tab_bg = get_color('tab_bg', '#252525')
            tab_bg_hover = get_color('tab_bg_hover', '#3d3d3d')
            tab_bg_checked = get_color('tab_bg_checked', '#4CAF50')
            tab_text = get_color('tab_text', '#aaaaaa')
            tab_text_hover = get_color('tab_text_hover', '#ffffff')
            tab_text_checked = get_color('tab_text_checked', '#ffffff')
        except:
            # Fallback
            bg = '#1e1e1e'
            homepage_1 = bg
            homepage_2 = '#2d2d2d'
            card_bg = '#2d2d2d'
            border = '#444444'
            sidebar_right_border = border
            top_row_bottom_border = border
            text_primary = '#ffffff'
            text_secondary = '#cccccc'
            hover_bg = '#3d3d3d'
            sidebar_rail_bg = card_bg
            sidebar_panel_bg = bg
            sidebar_button_hover_bg = hover_bg
            sidebar_button_checked_bg = hover_bg
            sidebar_tooltip_bg = card_bg
            sidebar_tooltip_text = text_primary
            sidebar_tooltip_border = border
            tab_bg = '#252525'
            tab_bg_hover = '#3d3d3d'
            tab_bg_checked = '#4CAF50'
            tab_text = '#aaaaaa'
            tab_text_hover = '#ffffff'
            tab_text_checked = '#ffffff'

        mission_panel_gradient = (
            f"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {homepage_1}, stop:1 {homepage_2})"
        )

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg};
            }}
            QWidget {{
                color: {text_primary};
            }}
            #startupLogBg {{
                background-color: #060A0D;
            }}
            #startupLogDisplay {{
                background-color: transparent;
                border: none;
                color: #FFFFFF;
                font-family: Consolas, "Courier New", monospace;
                font-size: 13px;
                padding: 18px;
                selection-background-color: transparent;
            }}
            #startupLoadingOverlay {{
                background-color: rgba(0, 0, 0, 217);
            }}
            QLabel {{
                background-color: {card_bg};
                border-radius: 12px;
                padding: 2px;
            }}
            #sidebarRail {{
                background-color: {sidebar_rail_bg};
                border-right: none;
            }}
            #mainZone {{
                /* background-color: {bg}; */
                background: {mission_panel_gradient};
            }}
            #navigationHeader {{
                background-color: {card_bg};
                border-bottom: 1px solid {top_row_bottom_border};
            }}
            #windowControlsRow {{
                background-color: {card_bg};
                border-bottom: 1px solid {border};
            }}
            #windowControlButton {{
                min-width: 28px;
                max-width: 28px;
                min-height: 22px;
                max-height: 22px;
                padding: 0px;
                border-radius: 4px;
            }}
            #missionZone {{
                /* background-color: {bg}; */
                background: {mission_panel_gradient};
                border: 1px solid {border};
            }}
            #missionControlRow {{
                background-color: {card_bg};
                border-bottom: 1px solid {border};
            }}
            #missionControlFloat {{
                background-color: {card_bg};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            #canvasHeader, #compilerHeader {{
                background-color: {card_bg};
                border-bottom: 1px solid {border};
            }}
            #canvasZone, #compilerZone {{
                border: 1px solid {border};
                border-top: none;
                /* background-color: {bg}; */
                background: {mission_panel_gradient};
            }}
            QPushButton {{
                background-color: {card_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
            #sidebarNavButton, #sidebarUtilityButton {{
                background-color: {sidebar_rail_bg};
                color: {text_primary};
                border: none;
                border-radius: 8px;
                padding: 0px;
                text-align: center;
                font-weight: 600;
            }}
            #sidebarNavButton:hover, #sidebarUtilityButton:hover {{
                background-color: {sidebar_button_hover_bg};
            }}
            #sidebarNavButton:checked {{
                background-color: {sidebar_button_checked_bg};
                border: 1px solid {border};
            }}
            #sidebarContentPanel {{
                background-color: {sidebar_panel_bg};
                border-left: none;
                border-right: none;
            }}
            #sidebarHoverTooltip {{
                background-color: {sidebar_tooltip_bg};
                color: {sidebar_tooltip_text};
                border: 1px solid {sidebar_tooltip_border};
                border-radius: 8px;
                padding: 0px 10px;
                font-weight: 600;
            }}
            #sidebarContentTitle {{
                background-color: transparent;
                border-radius: 0px;
                padding: 0px;
                font-weight: 700;
            }}
            #sidebarContentStack, #sidebarPlaceholderPage {{
                background-color: transparent;
                border: none;
            }}
            #projectFilesPanel {{
                background-color: transparent;
            }}
            #projectFilesSearchInput {{
                background-color: {card_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            #projectFilesList {{
                background-color: transparent;
                color: {text_primary};
                border: none;
                border-radius: 0px;
                padding: 0px;
            }}
            #projectFilesList::item {{
                padding: 6px 8px;
                border-radius: 6px;
            }}
            #projectFilesList::item:hover {{
                background-color: {hover_bg};
                border: 1px solid {border};
            }}
            #projectFilesList::item:selected {{
                background-color: {sidebar_button_checked_bg};
                color: {text_primary};
            }}
            #sidebarPlaceholderLabel {{
                background-color: transparent;
                border-radius: 0px;
                padding: 0px;
                color: {text_secondary};
            }}
            #zoneToggleButton {{
                min-width: 28px;
                max-width: 34px;
                padding: 2px 4px;
                border-radius: 4px;
            }}
            #missionControlDragHandle {{
                background-color: transparent;
                border: none;
                padding: 0px 2px;
                font-size: 16px;
            }}
            #missionControlDragHandle:hover {{
                color: {tab_text_checked};
            }}
            QComboBox {{
                background-color: {card_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 2px 6px;
                min-height: 20px;
            }}
            QComboBox::item {{
                color: {text_primary};
                background-color: {card_bg};
            }}
            QComboBox::item:selected {{
                background-color: {hover_bg};
                color: {text_primary};
            }}
            QComboBox QAbstractItemView {{
                background-color: {card_bg};
                color: {text_primary};
                selection-background-color: {hover_bg};
            }}
            QStatusBar {{
                background-color: {bg};
                color: {text_secondary};
                border-top: 1px solid {border};
            }}
            QSplitter::handle {{
                background-color: {border};
            }}
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
            QSplitter::handle:vertical {{
                height: 2px;
            }}
            #workspaceTabs {{
                background-color: {tab_bg};
                border: none;
            }}
            #workspaceTabs::pane {{
                background-color: {bg};
                border: none;
            }}
            #canvasFileBadge {{
                background-color: {tab_bg_checked};
                color: {tab_text_checked};
                border: none;
                border-radius: 2px;
                padding: 0px;
                font-weight: 600;
            }}
            #workspaceTabs QTabBar::tab {{
                background-color: {tab_bg};
                color: {tab_text};
                border: 1px solid {border};
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 2px 14px;
                min-height: 18px;
                margin-right: 0px;
                font-weight: 400;
            }}
            #workspaceTabs QTabBar::close-button {{
                width: 10px;
                height: 10px;
            }}
            #workspaceTabs QTabBar::tab:hover {{
                background-color: {tab_bg_hover};
                color: {tab_text_hover};
            }}
            #workspaceTabs QTabBar::tab:selected {{
                background-color: {tab_bg_checked};
                color: {tab_text_checked};
                font-weight: 700;
            }}
            #compilerTabs {{
                background-color: {tab_bg};
                border: none;
            }}
            #compilerTabs::pane {{
                background-color: {bg};
                border: 1px solid {border};
                border-top: none;
            }}
            #compilerTabs QTabBar::tab {{
                background-color: {tab_bg};
                color: {tab_text};
                border: 1px solid {border};
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 2px 14px;
                min-height: 18px;
                margin-right: 0px;
                font-weight: 400;
            }}
            #compilerTabs QTabBar::close-button {{
                width: 10px;
                height: 10px;
            }}
            #compilerTabs QTabBar::tab:hover {{
                background-color: {tab_bg_hover};
                color: {tab_text_hover};
            }}
            #compilerTabs QTabBar::tab:selected {{
                background-color: {tab_bg_checked};
                color: {tab_text_checked};
                font-weight: 700;
            }}
        """)
        # Apply PointingHandCursor to every QPushButton in the window that
        # still has the default ArrowCursor (buttons with intentional custom
        # cursors 鈥?drag handles etc. 鈥?are unaffected).
        for _btn in self.findChildren(QPushButton):
            if _btn.cursor().shape() == Qt.CursorShape.ArrowCursor:
                _btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if hasattr(self, "mission_nav_header"):
            self.mission_nav_header.apply_theme()

    def _open_project_files_sidebar(self) -> None:
        if not hasattr(self, "sidebar") or self.sidebar is None:
            return
        self.sidebar.show_panel("projects")
        self._on_sidebar_panel_requested("projects")

    def _on_sidebar_panel_requested(self, panel_key: str):
        panel_name = panel_key.replace("_", " ").title()
        if panel_key == "projects" and self.project_files_panel is not None:
            self.project_files_panel.refresh_files()
        log_info(f"sidebar panel requested: {panel_name}")

    def _refresh_policy_registry_ui(self):
        """Sync registered checkpoints into Canvas policy_id dropdown models."""
        if hasattr(self, "graph_scene") and self.graph_scene is not None:
            try:
                self.graph_scene.refresh_policy_registry()
            except Exception:  # noqa: BLE001
                pass

    def _toggle_cmd_zone(self):
        if self._cmd_collapsed:
            self.cmd_log.show()
            self.main_splitter.setSizes([max(200, self.width() - self._cmd_last_width), self._cmd_last_width])
            self._cmd_collapsed = False
        else:
            current_sizes = self.main_splitter.sizes()
            if len(current_sizes) >= 2 and current_sizes[1] > 0:
                self._cmd_last_width = current_sizes[1]
            self.cmd_log.hide()
            self.main_splitter.setSizes([1, 0])
            self._cmd_collapsed = True
        self._sync_cmd_toggle_button()

    def _sync_cmd_toggle_button(self):
        if not hasattr(self, "cmd_toggle_button"):
            return
        # Expanded: show collapse symbol; collapsed: show expand symbol.
        self.cmd_toggle_button.setText(">>" if not self._cmd_collapsed else "<<")
        self.cmd_toggle_button.setToolTip("CMD")

    def _init_statusbar(self):
        """Initialize status bar (hidden 鈥?kept for internal showMessage calls)."""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setVisible(False)

        # Initialize RobotContext with default robot type
        robot_type = self.config.get('SIMULATION', 'default_robot', fallback='go2')
        RobotContext.set_robot_type(robot_type)
        self.robot_model = RobotContext.get_robot_model()
        self.scenario_state.robot_type = robot_type

        # Populate capability inspector after the event loop starts (Cycle 2 STAGE-04)
        QTimer.singleShot(0, self._refresh_capability_inspector)

        # Show initial status
        self.status.showMessage(
            tr("status.ready", "Ready | Robot: {robot}", robot=robot_type)
        )

    def set_robot_model(self, robot_model):
        """Set robot model"""
        self.robot_model = robot_model

        # Also set graph scene robot type
        if hasattr(self, 'graph_scene') and robot_model:
            robot_type = getattr(robot_model, 'robot_type', 'go2')
            self.graph_scene.set_robot_type(robot_type)

        log_success(tr("log.robot_model_set", "Robot model set: {model}", model=robot_model))

    def _on_robot_type_changed(self, robot_type: str):
        """Robot type changed - updates global RobotContext"""
        log_info(tr("log.robot_type_changed", "Robot type changed: {type}", type=robot_type))
        self.status.showMessage(
            tr("status.robot_changed", "Robot type changed: {robot}", robot=robot_type),
            2000
        )

        # Update global RobotContext (CRITICAL: This is the global state)
        RobotContext.set_robot_type(robot_type)
        self.scenario_state.robot_type = robot_type

        # Keep graph-scene cache aligned without pushing a second UI mutation
        # back into the Start node picker that emitted this change.
        if hasattr(self, 'graph_scene'):
            self.graph_scene._robot_type = robot_type

        # Get robot model from context
        self.robot_model = RobotContext.get_robot_model()

        # Sync settings panel brand + refresh capability inspector (Cycle 2 STAGE-04)
        brand = RobotContext.get_current_brand()
        self.main_zone.set_sdk_brand(brand)
        self._refresh_capability_inspector()

        # Refresh Behavior node action dropdowns to match the new model's
        # available actions (filters out model-unsupported actions like
        # lift_right_leg which is Go2-only).
        self._refresh_behavior_node_actions(brand, robot_type)

    def _get_runtime_scenario_settings(self) -> dict:
        """Read current settings from Scenario panel and apply env overrides."""
        settings = {}
        if hasattr(self.main_zone, "get_scenario_settings"):
            settings = self.main_zone.get_scenario_settings() or {}

        runtime_scene_xml = str(settings.get("mujoco_scene_xml", "") or "").strip()
        resolved_runtime_scene_xml = runtime_scene_xml
        if not resolved_runtime_scene_xml:
            robot_type = self._get_selected_robot_type()
            brand_map = {
                "go2": ("unitree", "go2"),
                "go2w": ("unitree", "go2w"),
                "go1": ("unitree", "go1"),
                "a1": ("unitree", "a1"),
                "g1": ("unitree", "g1"),
                "h1": ("unitree", "h1"),
                "h1_2": ("unitree", "h1_2"),
                "spot": ("bostiondynamics", "spot"),
            }
            brand_info = brand_map.get(str(robot_type or "").strip().lower())
            if brand_info is not None:
                try:
                    from system.mujoco_asset_registry import resolve_mujoco_asset
                    loc = resolve_mujoco_asset(brand_info[0], brand_info[1])
                    if loc is not None:
                        resolved_runtime_scene_xml = str(loc.scene_path)
                except Exception:
                    pass
        settings["resolved_runtime_scene_xml"] = resolved_runtime_scene_xml

        gl_backend = settings.get("mujoco_gl_backend")
        if gl_backend:
            os.environ["MUJOCO_GL"] = str(gl_backend)
        self.scenario_state.params = dict(settings)

        return settings

    def _get_selected_robot_type(self) -> str:
        """Return the current UI-selected robot type for runtime decisions."""
        if hasattr(self, "graph_scene") and self.graph_scene is not None:
            _brand, start_robot_type = self.graph_scene.get_start_node_selection()
            if start_robot_type:
                return start_robot_type
        graph_robot_type = str(getattr(self.graph_scene, "_robot_type", "") or "").strip().lower()
        if graph_robot_type:
            return graph_robot_type
        settings_robot_type = str((self.main_zone.get_sdk_settings() or {}).get("robot_type") or "").strip().lower()
        if settings_robot_type:
            return settings_robot_type
        return str(getattr(self.scenario_state, "robot_type", "go2") or "go2").strip().lower()

    def _resolve_execution_backend(self, runtime_model) -> str:
        """Return the execution backend to use for a mission run.

        Current policy: always MuJoCo simulation ("mujoco").
        Hardware SDK path ("sdk") is not yet wired into the UI; when it is,
        this method is the single place to add the routing logic 鈥?e.g.:

            if self._sdk_hardware_connected():
                return "sdk"

        Callers inject the returned value into ``scenario["backend"]`` so that
        ``_determine_execution_backend()`` in behavior_invoker picks it up with
        the highest priority (before the legacy ``target`` field fallback).

        Returns:
            "mujoco" 鈥?simulation (current implementation)
            "sdk"    鈥?real hardware (future; not yet implemented)
        """
        # TODO(sdk-wiring): when hardware deployment is supported, detect
        #   active adapter presence here and return "sdk" accordingly.
        return "mujoco"

    def _sync_runtime_robot_selection(self) -> str:
        """Sync RobotContext with the current UI-selected robot type."""
        robot_type = self._get_selected_robot_type() or "go2"
        RobotContext.set_robot_type(robot_type)
        self.scenario_state.robot_type = robot_type
        if hasattr(self, "graph_scene"):
            self.graph_scene._robot_type = robot_type
        return robot_type

    def _on_language_changed(self, index: int):
        """Language changed"""
        lang_code = "en"
        loc = get_localisation()
        if loc.load_language(lang_code):
            log_info(f"Language changed to: {lang_code}")
            # Note: Full UI refresh would require more extensive changes
            # For now, new text will appear on next widget creation
            self._refresh_theme()

    def _on_language_button_clicked(self):
        """Language quick switch button."""
        self._on_language_changed(0)
        self.language_button.setText("EN")

    def _on_new(self):
        """New project"""
        log_info(tr("log.new_project", "New project"))
        if hasattr(self, "graph_scene"):
            self.graph_scene.clear_all_nodes()
        if hasattr(self, "main_zone"):
            self.main_zone.clear_execution_summary()
            self.main_zone.clear_diagnostics_panel()
        self.code_editor.clear()
        self._set_current_workflow_path("")
        self.status.showMessage(tr("status.new_project", "New project"), 2000)

    def _workflows_root(self) -> str:
        workflows_root = Path(self.config.project_root) / "workflows"
        workflows_root.mkdir(parents=True, exist_ok=True)
        return str(workflows_root)

    # ------------------------------------------------------------------
    # Recent-file persistence (user.ini [RECENT])
    # ------------------------------------------------------------------

    def _save_last_mission_file(self, path: str) -> None:
        try:
            self.config.set("RECENT", "last_mission_file", path, config_type="user")
            self.config.save_user_config()
        except Exception:
            pass

    def _load_last_mission_file(self) -> str:
        return self.config.get("RECENT", "last_mission_file", fallback="", config_type="user") or ""

    def _save_last_training_policy(self, policy_id: str) -> None:
        try:
            self.config.set("RECENT", "last_training_policy", policy_id, config_type="user")
            self.config.save_user_config()
        except Exception:
            pass

    def _load_last_training_policy(self) -> str:
        return self.config.get("RECENT", "last_training_policy", fallback="", config_type="user") or ""

    def _set_current_workflow_path(self, path: str) -> None:
        normalized_path = os.path.abspath(path) if path else ""
        self._current_workflow_path = normalized_path
        if normalized_path:
            self._save_last_mission_file(normalized_path)
        title = os.path.basename(normalized_path) if normalized_path else "[New File]"
        compiler_title = os.path.basename(normalized_path) if normalized_path else "New File"
        if hasattr(self, "main_zone"):
            self.main_zone.set_workflow_tab_title(title or "[New File]")
            self.main_zone.set_compiler_main_tab_title(compiler_title)
        if self.project_files_panel is not None:
            self.project_files_panel.refresh_files()

    def _build_workflow_payload(self) -> dict:
        from bin.core.mission_persistence import inject_snapshot_metadata

        data = self.graph_scene.serialize_workflow()

        # Cycle 3 STAGE-02: inject current SDK settings into the mission payload
        # before metadata stamping so the settings survive a save/load roundtrip.
        try:
            from bin.core.mission_persistence import build_settings_payload
            _brand = RobotContext.get_current_brand()
            _config = self.main_zone.get_sdk_settings()
            data["settings"] = build_settings_payload(_brand, _config)
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not inject settings into mission payload: {_exc}")

        # Cycle 3 STAGE-06: inject advanced MuJoCo/scenario settings.
        try:
            from bin.core.mission_persistence import build_scenario_payload
            _scenario_cfg = self.main_zone.get_scenario_settings()
            data["scenario_settings"] = build_scenario_payload(_scenario_cfg)
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not inject scenario settings into mission payload: {_exc}")

        # Circle 1 Step 1.6: inject per-node behavior/heartbeat drafts.
        try:
            from bin.core.mission_persistence import build_behavior_drafts_payload
            _drafts = self.main_zone.behavior_panel.get_behavior_drafts_state()
            data["behavior_drafts"] = build_behavior_drafts_payload(_drafts)
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not inject behavior drafts into mission payload: {_exc}")

        # Phase 1 behavior redesign: inject per-node structured behavior timelines.
        # Silent no-op when the panel has no timelines (fallback path unchanged).
        try:
            from bin.core.mission_persistence import build_behavior_timeline_payload
            _timelines = self.main_zone.behavior_panel.get_behavior_timelines_state()
            data["behavior_timelines"] = build_behavior_timeline_payload(_timelines)
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not inject behavior timelines into mission payload: {_exc}")

        # Circle 3: persist package metadata so it survives save/load round-trips.
        try:
            from bin.core.mission_persistence import build_package_metadata_payload
            data["package_metadata"] = build_package_metadata_payload(
                self._loaded_package_metadata or {}
            )
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not inject package metadata into mission payload: {_exc}")

        inject_snapshot_metadata(data)
        return data

    def _save_workflow_to_path(self, path: str) -> bool:
        if not hasattr(self, "graph_scene"):
            log_error("Graph scene not available")
            return False

        data = self._build_workflow_payload()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            log_error(tr("log.save_error", "Save failed: {error}", error=str(exc)))
            QMessageBox.critical(
                self,
                tr("messages.error", "Error"),
                tr("messages.save_write_error", "Could not write file: {error}", error=str(exc)),
            )
            return False

        self._set_current_workflow_path(path)
        log_info(tr("log.save_project", "Save project"))
        log_info(tr("log.saved", "Mission saved: {path}", path=path))
        self.status.showMessage(
            tr("status.saved", "Saved: {path}", path=path), 3000
        )
        return True

    def _load_workflow_from_path(self, path: str) -> bool:
        """Load a workflow file from disk into the current canvas."""
        # Stage 4: unified unsaved guard (settings + behavior) before navigation.
        if not self._handle_unsaved_settings_guard(
            tr("messages.before_open", "before opening a file")
        ):
            return False

        from bin.core.mission_persistence import validate_mission_schema
        if not path:
            return False

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            log_error(tr("log.open_error", "Open failed: {error}", error=str(exc)))
            QMessageBox.critical(
                self,
                tr("messages.error", "Error"),
                tr("messages.open_read_error", "Could not read file: {error}", error=str(exc)),
            )
            return False

        ok, reason = validate_mission_schema(data)
        if not ok:
            log_error(tr("log.open_schema_error", "Mission schema invalid: {reason}", reason=reason))
            QMessageBox.warning(
                self,
                tr("messages.schema_mismatch", "Schema Mismatch"),
                tr(
                    "messages.schema_mismatch_detail",
                    "Cannot load mission 鈥?schema check failed:\n{reason}",
                    reason=reason,
                ),
            )
            return False

        if not hasattr(self, "graph_scene"):
            log_error("Graph scene not available")
            return False

        # Step 5 鈥?emit migration warnings for pre-1.4 files before loading.
        try:
            from bin.core.mission_persistence import migrate_mission_payload
            _migration = migrate_mission_payload(data)
            for _w in _migration.get("warnings", []):
                log_info(_w)
        except Exception:
            pass

        self.graph_scene.load_workflow(data)
        self._refresh_policy_registry_ui()
        self._sync_runtime_robot_selection()
        self._set_current_workflow_path(path)
        self.main_zone.clear_execution_summary()
        self.main_zone.clear_diagnostics_panel()

        # Cycle 3 STAGE-02: restore SDK settings when brand matches.
        # Silent no-op for old files without a "settings" key (backward compat).
        try:
            from bin.core.mission_persistence import extract_settings_payload
            _settings_pair = extract_settings_payload(data)
            if _settings_pair is not None:
                _loaded_brand, _loaded_config = _settings_pair
                _current_brand = RobotContext.get_current_brand()
                if _loaded_brand == _current_brand:
                    self.main_zone.set_sdk_brand(_loaded_brand, _loaded_config)
                    log_info(
                        tr(
                            "log.settings_restored",
                            "SDK settings restored from mission (brand={brand})",
                            brand=_loaded_brand,
                        )
                    )
                else:
                    log_info(
                        tr(
                            "log.settings_brand_mismatch",
                            "Mission settings brand '{loaded}' differs from active brand '{current}' 鈥?skipping restore",
                            loaded=_loaded_brand,
                            current=_current_brand,
                        )
                    )
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not restore settings from mission: {_exc}")

        # Cycle 3 STAGE-06: restore advanced MuJoCo/scenario settings when present.
        try:
            from bin.core.mission_persistence import extract_scenario_payload
            _scenario_cfg = extract_scenario_payload(data)
            if _scenario_cfg is not None:
                self.main_zone.set_scenario_settings(_scenario_cfg)
                log_info(tr("log.scenario_settings_restored",
                            "Scenario settings restored from mission"))
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not restore scenario settings from mission: {_exc}")

        # Circle 1 Step 1.6: restore per-node behavior/heartbeat drafts when present.
        # Silent no-op for old mission files that lack the "behavior_drafts" key.
        try:
            from bin.core.mission_persistence import extract_behavior_drafts_payload
            _drafts = extract_behavior_drafts_payload(data)
            if _drafts is not None:
                self.main_zone.behavior_panel.set_behavior_drafts_state(_drafts)
                log_info(tr("log.behavior_drafts_restored",
                            "Behavior drafts restored from mission"))
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not restore behavior drafts from mission: {_exc}")

        # Phase 1 behavior redesign: restore per-node structured behavior timelines.
        # Silent no-op for missions saved before Phase 1 (no "behavior_timelines" key).
        try:
            from bin.core.mission_persistence import extract_behavior_timeline_payload
            _timelines = extract_behavior_timeline_payload(data)
            if _timelines is not None:
                self.main_zone.behavior_panel.set_behavior_timelines_state(_timelines)
                self.graph_scene.refresh_behavior_node_parameter_rows()
                log_info(tr("log.behavior_timelines_restored",
                            "Behavior timelines restored from mission"))
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not restore behavior timelines from mission: {_exc}")

        # Circle 3: restore package metadata for compile/execute traceability.
        # Silent no-op for old files that pre-date package metadata (backward-compat).
        try:
            from bin.core.mission_persistence import extract_package_metadata_payload
            _pkg_meta = extract_package_metadata_payload(data)
            self._loaded_package_metadata = _pkg_meta if _pkg_meta is not None else {}
            if _pkg_meta:
                log_info(
                    f"Package metadata restored: "
                    f"package_id={_pkg_meta.get('package_id', '')!r} "
                    f"version={_pkg_meta.get('package_version', '')!r}"
                )
        except Exception as _exc:  # noqa: BLE001
            log_warning(f"Could not restore package metadata from mission: {_exc}")

        log_info(tr("log.open_project", "Open project"))
        log_info(tr("log.opened", "Mission opened: {path}", path=path))
        self.status.showMessage(
            tr("status.opened", "Opened: {path}", path=path), 3000
        )
        self._refresh_capability_inspector()
        if self.project_files_panel is not None:
            self.project_files_panel.refresh_files()
        return True

    def _open_workflow_from_path(self, path: str):
        self._load_workflow_from_path(path)

    def _on_open(self):
        """Open project 鈥?shows file dialog and loads a mission file."""
        if not self._handle_unsaved_settings_guard(tr("toolbar.open", "Open")):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("toolbar.open", "Open Mission"),
            self._workflows_root(),
            "Mission Files (*.unitport);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self._load_workflow_from_path(path)

    def _open_workflows_folder(self):
        workflows_root = self._workflows_root()
        if os.name == "nt":
            try:
                os.startfile(workflows_root)
                return
            except OSError:
                pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(workflows_root))

    def _on_open_training_workspace(self, policy_id: str):
        """Navigate to Training Ground for the given policy_id."""
        self.go_training(policy_id)

    def _on_checkpoint_exported(
        self,
        bundle_path: str,
        base_policy_id: str = "",
    ) -> None:
        """
        Refresh Mission-side runtime checkpoint pickers after a training export.

        When the Training Ground was opened from a Checkpoint node bound to
        ``base_policy_id``, automatically retarget any Mission-side Checkpoint
        pickers still selecting that base policy so the next MuJoCo run uses
        the freshly exported checkpoint.
        """
        exported_policy_id = Path(str(bundle_path or "")).name.strip()
        self._refresh_policy_registry_ui()
        if (
            exported_policy_id
            and base_policy_id
            and exported_policy_id != base_policy_id
            and hasattr(self, "graph_scene")
            and self.graph_scene is not None
        ):
            try:
                self.graph_scene.retarget_checkpoint_policy_references(
                    base_policy_id,
                    exported_policy_id,
                )
            except Exception:  # noqa: BLE001
                pass

    # 鈹€鈹€ Circle 7 S2: mock training shell 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _on_save(self):
        """Save current workflow to current path; fallback to Save As for new files."""
        if self._current_workflow_path:
            self._save_workflow_to_path(self._current_workflow_path)
            return
        self._on_save_as()

    def _on_save_as(self):
        """Save workflow to a selected file path."""
        initial_path = self._current_workflow_path or self._workflows_root()
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("toolbar.save_as", "Save Workflow As"),
            initial_path,
            "Mission Files (*.unitport);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self._save_workflow_to_path(path)

    def _on_export_code(self):
        """Export code"""
        log_info(tr("log.export_code", "Export code"))
        code = self.code_editor.get_code()
        QMessageBox.information(
            self,
            tr("toolbar.export_code", "Export Code"),
            tr(
                "messages.export_code_length",
                "Code length: {length} characters\n(Export feature not implemented)",
                length=len(code)
            )
        )

    def _on_run(self):
        """Run the connected workflow (async, non-blocking 鈥?Cycle 2 STAGE-06)."""
        log_info(tr("log.run", "Run"))

        # Stage 5: drop stale finished thread handles before a new run starts.
        if self._mission_run_thread is not None and not self._mission_run_thread.isRunning():
            self._mission_run_thread = None

        # Stage 4: unified unsaved guard (settings + behavior) before run.
        if not self._handle_unsaved_settings_guard(
            tr("messages.before_run", "before running")
        ):
            return

        # Refresh capability inspector before execution so inspector stays in sync
        self._refresh_capability_inspector()

        # Block run if current SDK settings fail validation (Cycle 2 STAGE-05)
        if not self._validate_settings_pre_run():
            return

        # Guard: do not start a second run while one is already active.
        if self._mission_run_thread and self._mission_run_thread.isRunning():
            log_warning("Mission is already running")
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                tr("messages.simulation_running", "Simulation is already running"),
            )
            return

        # Expand runtime zone before execution starts.
        self.main_zone.activate_runtime_fullscreen()
        QApplication.processEvents()

        if not hasattr(self, 'graph_scene'):
            log_error("Graph scene not available")
            return

        # Training Ground may have produced new checkpoints since the Mission
        # canvas last refreshed its registry-backed dropdowns. Refresh right
        # before export/compilation so Checkpoint nodes never run with a stale
        # empty policy_id after a successful training/export cycle.
        self._refresh_policy_registry_ui()

        exec_graph = self.graph_scene.get_execution_graph()

        if not exec_graph['nodes']:
            QMessageBox.information(
                self,
                tr("messages.info", "Info"),
                tr(
                    "messages.no_connected_nodes",
                    "No connected nodes in workflow. Please connect nodes to create a workflow."
                )
            )
            return

        # Guard: simulation thread (MuJoCo) must not be running concurrently.
        if self.simulation_thread and self.simulation_thread.isRunning():
            log_warning(tr("log.simulation_running", "Simulation is already running"))
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                tr("messages.simulation_running", "Simulation is already running")
            )
            return

        log_info(f"Executing workflow with {len(exec_graph['nodes'])} nodes")
        self.status.showMessage(tr("status.executing_workflow", "Executing workflow..."))

        # Reset per-node status badges before each run so stale state is never shown
        self.graph_scene.reset_execution_statuses()
        self.main_zone.clear_execution_summary()
        self.main_zone.clear_diagnostics_panel()
        # Stage 4: route operator to Mission tab for truthful live status context.
        self.main_zone.workspace_tabs.setCurrentIndex(0)

        # Mark every mission node as "pending" before execution starts (STAGE-03).
        for _nid in exec_graph.get("nodes", {}):
            self.graph_scene.set_node_execution_status(_nid, "pending")
        QApplication.processEvents()

        selected_robot_type = self._sync_runtime_robot_selection()
        if not self._prepare_behavior_artifacts_for_run(exec_graph, selected_robot_type):
            return
        runtime_model = RobotContext.get_robot_model()
        self.robot_model = runtime_model

        scenario_settings = self._get_runtime_scenario_settings()
        runtime_brand = RobotContext.get_current_brand()
        if runtime_model is None:
            if runtime_brand != "unitree":
                log_warning(
                    f"Simulation requested for unsupported MuJoCo brand/model: "
                    f"{runtime_brand}/{selected_robot_type}"
                )
                QMessageBox.warning(
                    self,
                    tr("messages.warning", "Warning"),
                    f"MuJoCo simulation is not implemented for {runtime_brand}/{selected_robot_type}.",
                )
                return
            log_error(
                f"Simulation requested but no Unitree runtime model is available for "
                f"{selected_robot_type}"
            )
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                f"Failed to initialize MuJoCo model for unitree/{selected_robot_type}.",
            )
            return

        # Phase F: Behavior-node runs require a live MuJoCo model/data pair even
        # when the workflow contains no legacy action nodes.  Keep this prep
        # headless so the replay viewer remains the only MuJoCo window.
        if self._has_behavior_nodes(exec_graph):
            try:
                close_viewer = getattr(runtime_model, "close_viewer", None)
                if callable(close_viewer):
                    close_viewer()
                if hasattr(runtime_model, "initialize") and callable(runtime_model.initialize):
                    if not runtime_model.initialize():
                        raise RuntimeError("runtime_model.initialize() failed")
                if (
                    getattr(runtime_model, "mj_model", None) is None
                    and getattr(runtime_model, "model", None) is None
                ):
                    if hasattr(runtime_model, "load_model") and callable(runtime_model.load_model):
                        if not runtime_model.load_model():
                            raise RuntimeError("runtime_model.load_model() failed")
                if hasattr(runtime_model, "reset_simulation") and callable(runtime_model.reset_simulation):
                    reset_fn = runtime_model.reset_simulation
                    reset_kwargs = {}
                    try:
                        if "ensure_viewer" in inspect.signature(reset_fn).parameters:
                            reset_kwargs["ensure_viewer"] = False
                    except (TypeError, ValueError):
                        pass
                    if not reset_fn(**reset_kwargs):
                        raise RuntimeError("runtime_model.reset_simulation() failed")
            except Exception as exc:  # noqa: BLE001
                log_error(f"Failed to initialize MuJoCo runtime for Behavior node: {exc}")
                QMessageBox.warning(
                    self,
                    tr("messages.warning", "Warning"),
                    "Failed to initialize MuJoCo simulation for Behavior execution.",
                )
                return

        scenario = self.scenario_state.to_runtime_scenario(
            target="simulation",
            robot_model=runtime_model,
            graph_scene=self.graph_scene,
            simulation_running=bool(self.simulation_thread and self.simulation_thread.isRunning()),
            **scenario_settings,
        )

        # Phase F: BehaviorNode executes checkpoint policies through PolicyRunner,
        # which expects a SimEnvContext in runtime context. Build it eagerly for
        # simulation runs so Checkpoint -> Behavior can open and drive MuJoCo.
        try:
            from system.policy.policy_command_executor import _build_sim_env_context
            scenario["sim_env"] = _build_sim_env_context(runtime_model, render=True)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"Could not build SimEnvContext for mission run: {exc}")

        # Task 1 (STAGE-06): inject validated SDK settings into scenario so the
        # service lifecycle can access them without bypassing RobotContext.
        scenario["sdk_settings"] = self.main_zone.get_sdk_settings()
        scenario["brand"]        = runtime_brand
        # SDK/MuJoCo backend routing 鈥?resolved via _resolve_execution_backend().
        # Currently always "mujoco"; future SDK hardware support hooks here.
        scenario["backend"]      = self._resolve_execution_backend(runtime_model)
        self._last_runtime_target = str(scenario.get("target", ""))
        self._last_runtime_backend = str(scenario.get("backend", ""))

        # Cycle 3 STAGE-03 (fix): build a per-run policy and pass it directly to
        # MissionRunThread.  The thread activates it as a thread-local so all
        # RobotContext calls from the worker use the run-scoped settings without
        # ever mutating the class-level _lifecycle_policy.
        _run_policy = None
        try:
            _sdk = self.main_zone.get_sdk_settings()
            _brand = runtime_brand
            _run_policy = RobotContext.make_run_policy(
                {**_sdk, "brand": _brand, "robot_type": selected_robot_type}
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(f"Could not build run-scoped policy: {exc}")

        # Circle 1 Step 1.1: choose execution path.
        # mission_ir is what RuntimeEngine actually receives; exec_graph dict is
        # always kept in _last_exec_graph for UI status-badge bookkeeping.
        mission_ir = exec_graph  # default: exec_graph compat path
        _behavior_path_reason = "exec_graph_compat"
        if self._should_use_workflowir_run(exec_graph):
            _compiled = self._compile_canvas_to_workflowir()
            if _compiled is not None:
                # Circle 3: attach layered_contracts (including package metadata)
                # to the compiled WorkflowIR so it flows through to execution.
                try:
                    from system.ir.layered_interfaces import DefaultLayeredIRBridge
                    from system.ir.layered_contracts import LayeredIRBundle, SubgraphIR, PackageMetadata
                    _bridge = DefaultLayeredIRBridge()
                    _bundle = LayeredIRBundle()
                    _pkg = getattr(self, "_loaded_package_metadata", {}) or {}
                    if _pkg:
                        _pm = PackageMetadata.from_dict(_pkg)
                        _sg = SubgraphIR(subgraph_id="mission", display_name="Mission",
                                         package_metadata=_pm)
                        _bundle.subgraphs = [_sg]
                    _bridge.apply_to_workflow_ir(_compiled, _bundle)
                except Exception as _exc:  # noqa: BLE001
                    log_warning(f"Circle 3: bridge attach failed 鈥?{_exc}")
                mission_ir = _compiled
                _behavior_path_reason = "workflowir_behavior_enabled"
                log_info("Circle 1: behavior-enabled run 鈥?using WorkflowIR execution path")
            else:
                _behavior_path_reason = "workflowir_compile_failed_fallback"
                log_warning(
                    "Circle 1: WorkflowIR compilation failed 鈥?"
                    "falling back to exec_graph path"
                )

        # Cache exec_graph (dict) always for UI status badges (_on_mission_finished).
        self._last_exec_graph = exec_graph
        # Carry path reason into the scenario so RuntimeEngine can surface it
        # in diagnostics (picked up in _on_mission_finished).
        scenario["_behavior_path_reason"] = _behavior_path_reason

        # Circle 3: inject package metadata trace so RuntimeEngine can surface
        # traceability fields in execution diagnostics.  Always present (never None).
        _pkg = getattr(self, "_loaded_package_metadata", {}) or {}
        scenario["package_metadata_trace"] = {
            "package_id":      _pkg.get("package_id", ""),
            "package_version": _pkg.get("package_version", ""),
            "schema_version":  _pkg.get("schema_version", ""),
        }

        # Task 2 (STAGE-06): run on background thread; UI remains responsive.
        # run_policy is activated as thread-local inside the thread (STAGE-03 fix).
        self._mission_run_thread = MissionRunThread(mission_ir, scenario, self.runtime_engine, run_policy=_run_policy)
        self._mission_run_thread.node_status_changed.connect(self._on_mission_node_status)
        self._mission_run_thread.execution_finished.connect(self._on_mission_finished)
        self._mission_run_thread.start()

    # 鈹€鈹€ Async mission execution slots (Cycle 2 STAGE-06) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _on_mission_node_status(self, node_id: object, status: str) -> None:
        """Slot: per-node status signal from MissionRunThread (main thread).

        Qt's queued connection guarantees delivery on the main thread, so
        graph_scene calls are safe here without explicit locking.
        """
        sender = self.sender()
        if sender is not None and sender is not self._mission_run_thread:
            return
        self.graph_scene.set_node_execution_status(node_id, status)

    def _on_mission_finished(self, run_result: dict) -> None:
        """Slot: mission execution completed (main thread).

        Mirrors the post-execution UX logic previously inline in _on_run().
        Called via queued signal from MissionRunThread when execute() returns.
        """
        sender = self.sender()
        if sender is not None and sender is not self._mission_run_thread:
            return
        self._runtime_paused = False

        exec_graph = self._last_exec_graph

        # Apply final per-node execution status badges (STAGE-03).
        self._apply_node_execution_statuses(exec_graph, run_result)
        # Show session summary bar regardless of success/failure (STAGE-03).
        self.main_zone.show_execution_summary(run_result)
        # Populate diagnostics panel for failed nodes (STAGE-04).
        self._populate_diagnostics_panel(exec_graph, run_result)

        # Simulation runs should not leave terminal success borders latched on the canvas.
        if self._last_runtime_target == "simulation" or self._last_runtime_backend == "mujoco":
            QTimer.singleShot(0, self.graph_scene.reset_execution_statuses)

        reason = run_result.get("reason", "")
        diag = run_result.get("diagnostics", {})
        executed_count = diag.get("executed_count", len(run_result.get("results", {})))
        if run_result.get("status") != "success":
            if reason in ("simulation_reset_failed", "safety:simulation_reset_failed"):
                QMessageBox.warning(
                    self,
                    tr("messages.warning", "Warning"),
                    tr(
                        "messages.simulation_reset_failed",
                        "Failed to reset simulation. Please check MuJoCo setup."
                    )
                )
            elif reason in ("simulation_already_running", "simulation_running"):
                QMessageBox.warning(
                    self,
                    tr("messages.warning", "Warning"),
                    tr("messages.simulation_running", "Simulation is already running")
                )
            elif reason == "mission_cancelled":
                self.status.showMessage(
                    tr("status.mission_cancelled", "Mission cancelled"),
                    2000,
                )
                log_info("Mission execution cancelled.")
            else:
                failed_nodes = diag.get("failed_nodes", [])
                failed_suffix = f" Failed nodes: {', '.join(map(str, failed_nodes))}." if failed_nodes else ""
                reason_suffix = f" Reason: {reason}." if reason else ""
                log_error(
                    f"Workflow failed. Executed {executed_count} nodes."
                    f"{reason_suffix}{failed_suffix}"
                )
            # Cycle 3 STAGE-06: refresh capability inspector on failure/cancel
            # so live adapter state changes (e.g., disconnect) are reflected.
            self._refresh_capability_inspector()
            if self._mission_run_thread is not None and not self._mission_run_thread.isRunning():
                self._mission_run_thread = None
            return

        # Success path
        has_action = any(
            node.get('type') in ('action_execution', 'stop')
            or "Action Execution" in node.get('name', '')
            for node in exec_graph.get('nodes', {}).values()
        )
        if has_action and self.robot_model is None:
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                tr("messages.no_robot_model", "Robot model not set. Actions were not executed on hardware.")
            )

        self.status.showMessage(
            tr("status.workflow_completed", "Workflow execution completed"),
            5000
        )
        log_success(f"Workflow completed. Executed {executed_count} nodes.")

        # Cycle 3 STAGE-06: refresh capability inspector after run so live
        # adapter connection state changes (disconnect on finish) are reflected.
        self._refresh_capability_inspector()
        if self._mission_run_thread is not None and not self._mission_run_thread.isRunning():
            self._mission_run_thread = None

    def _populate_diagnostics_panel(self, exec_graph: dict, run_result: dict) -> None:
        """Auto-populate DiagnosticsPanel when failures exist (STAGE-04).

        Builds a node_names mapping from exec_graph so the panel shows human-
        readable node names rather than raw IDs.
        """
        diag = run_result.get("diagnostics", {})
        if not diag.get("failed_nodes"):
            self.main_zone.clear_diagnostics_panel()
            return
        # Build node_names from exec_graph for display
        node_names = {}
        for nid, node_data in exec_graph.get("nodes", {}).items():
            node_names[nid] = node_data.get("name", str(nid))
            node_names[str(nid)] = node_data.get("name", str(nid))
        from bin.core.error_ux import extract_failed_nodes_info  # noqa: PLC0415
        node_infos = extract_failed_nodes_info(run_result, node_names)
        self.main_zone.show_diagnostics_panel(node_infos)

    def _on_navigate_to_node(self, node_id) -> None:
        """Navigate the canvas to the node with *node_id* (STAGE-04)."""
        item = self.graph_scene.get_node_item(node_id)
        if item is not None:
            self.graph_view.centerOn(item)
            item.setSelected(True)

    def _apply_node_execution_statuses(self, exec_graph: dict, run_result: dict) -> None:
        """Apply per-node coloured border badges from a runtime result (STAGE-03).

        Nodes present in results without errors 鈫?"success".
        Nodes in diagnostics.failed_nodes             鈫?"failed".
        Nodes in exec_graph but absent from results   鈫?"skipped".
        """
        results = run_result.get("results", {})
        diag = run_result.get("diagnostics", {})
        # Normalize to str for safe comparison across int/str key formats
        failed_strs = {str(nid) for nid in diag.get("failed_nodes", [])}
        executed_strs = {str(nid) for nid in results.keys()}

        for node_id in exec_graph.get("nodes", {}):
            nid_str = str(node_id)
            if nid_str in failed_strs:
                status = "failed"
            elif nid_str in executed_strs:
                status = "success"
            else:
                status = "skipped"
            self.graph_scene.set_node_execution_status(node_id, status)

        # Apply runtime-authoritative protocol border states to Behavior nodes.
        self.graph_scene.apply_behavior_protocol_states_from_run_result(run_result)

    # 鈹€鈹€ Circle 1 Step 1.1 helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _has_behavior_nodes(exec_graph: dict) -> bool:
        """Return True if exec_graph contains any behavior-type canvas nodes."""
        for node in exec_graph.get("nodes", {}).values():
            if (
                node.get("type") in ("behavior", "behavior_call")
                or node.get("external_kind") == "behavior"
            ):
                return True
        return False

    def _should_use_workflowir_run(self, exec_graph: dict) -> bool:
        """Return True when this run should use WorkflowIR + NodeExecutor path.

        Activated by:
          1. UNITPORT_BEHAVIOR_ENABLED=1 env var (explicit opt-in for all runs).
          2. Auto-detection of behavior nodes inside exec_graph.
        """
        from system.runtime.migration import BehaviorRunFlags
        flags = BehaviorRunFlags.from_env()
        return flags.use_workflowir_for_behavior or self._has_behavior_nodes(exec_graph)

    def _compile_canvas_to_workflowir(self):
        """Compile the current canvas to WorkflowIR; returns None on any failure.

        Uses CanvasToIR pipeline.  Falls back gracefully so the caller can
        revert to the exec_graph compat path without interrupting the run.
        """
        try:
            from compiler.lowering.canvas_to_ir import CanvasToIR
            graph_data = self.graph_scene.export_graph_data()
            robot_type = getattr(self.graph_scene, "_robot_type", "go2")
            converter = CanvasToIR()
            ir, diags = converter.convert(graph_data, robot_type)
            errors = [d for d in diags if getattr(d.level, "value", str(d.level)) == "error"]
            if errors:
                log_warning(
                    f"Circle 1: Canvas鈫扞R had {len(errors)} error(s);"
                    " falling back to exec_graph path"
                )
                return None
            return ir
        except Exception as exc:  # noqa: BLE001
            log_warning(f"Circle 1: Canvas鈫扞R compilation raised {exc}; using exec_graph fallback")
            return None

    # 鈹€鈹€ end Circle 1 helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _on_runtime_pause(self):
        """Toggle pause/resume for active runtime execution."""
        has_active_runtime = bool(
            (self._mission_run_thread and self._mission_run_thread.isRunning())
            or (self.simulation_thread and self.simulation_thread.isRunning())
        )
        if not has_active_runtime:
            self.status.showMessage("No active runtime to pause/resume", 2000)
            return

        if not self._runtime_paused:
            if RobotContext.pause():
                self._runtime_paused = True
                log_info("Runtime paused")
                self.status.showMessage("Runtime paused", 2000)
                return
            self.status.showMessage("Pause is not supported by current robot model", 2000)
            return

        if RobotContext.resume():
            self._runtime_paused = False
            log_info("Runtime resumed")
            self.status.showMessage("Runtime resumed", 2000)
            return
        self.status.showMessage("Resume is not supported by current robot model", 2000)

    def _on_runtime_abort(self):
        """Abort runtime execution (Cycle 2 STAGE-06: also cancels mission thread)."""
        log_warning("Runtime abort requested")
        aborted = False

        # Cancel background mission run thread if active (STAGE-06).
        if self._mission_run_thread and self._mission_run_thread.isRunning():
            self._mission_run_thread.request_cancel()
            if not self._mission_run_thread.wait(5000):
                log_warning("Mission run thread did not finish within 5 s timeout")
            aborted = True
            # No policy restore needed: MissionRunThread's finally block clears
            # the thread-local run-scoped policy; class-level policy was never mutated.
            if not self._mission_run_thread.isRunning():
                self._mission_run_thread = None
        elif self._mission_run_thread is not None and not self._mission_run_thread.isRunning():
            self._mission_run_thread = None

        # Stop MuJoCo simulation thread if active.
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait(3000)
            self._clear_simulation_runtime_state()
            aborted = True
        # Best-effort direct interrupt for in-flight action loops on the
        # adapter-backed model currently used by action execution.
        if RobotContext.cancel_action():
            aborted = True
        RobotContext.stop()

        if aborted:
            self._runtime_paused = False
            self.status.showMessage("Runtime aborted", 2000)
            # Cycle 3 STAGE-06: refresh capability inspector after abort so live
            # adapter state (disconnected) is reflected immediately.
            self._refresh_capability_inspector()
        else:
            self.status.showMessage("No active runtime to abort", 2000)

    def _on_runtime_reset(self):
        """Reset runtime state and clear per-node status badges."""
        log_info("Runtime reset requested")
        if (
            (self._mission_run_thread and self._mission_run_thread.isRunning())
            or (self.simulation_thread and self.simulation_thread.isRunning())
        ):
            self._on_runtime_abort()
        reset_ok = RobotContext.reset_simulation()
        self.graph_scene.reset_execution_statuses()
        self._runtime_paused = False
        self.main_zone.clear_execution_summary()
        self.main_zone.clear_diagnostics_panel()
        self._refresh_capability_inspector()
        if self._mission_run_thread is not None and not self._mission_run_thread.isRunning():
            self._mission_run_thread = None
        if reset_ok:
            self.status.showMessage("Runtime reset", 2000)
        else:
            self.status.showMessage("Runtime reset (simulation reset not supported)", 2500)

    def _clear_simulation_runtime_state(self) -> None:
        """Clear transient canvas execution highlights left by simulation-only runs."""
        self.graph_scene.reset_execution_statuses()
        if self.simulation_thread is not None and not self.simulation_thread.isRunning():
            self.simulation_thread = None

    def _on_simulation_started(self, message: str) -> None:
        """Reflect simulation start in the status bar."""
        self.status.showMessage(message)

    def _on_simulation_finished(self, message: str) -> None:
        """Clear simulation runtime UI state once the worker exits."""
        self._runtime_paused = False
        self._clear_simulation_runtime_state()
        self.status.showMessage(message, 3000)

    def _on_simulation_error(self, message: str) -> None:
        """Surface simulation errors while keeping canvas runtime state consistent."""
        self._runtime_paused = False
        self._clear_simulation_runtime_state()
        QMessageBox.critical(self, "Error", message)

    def _test_lift_leg(self):
        """Test lift leg action"""
        self.main_zone.activate_runtime_fullscreen()
        QApplication.processEvents()
        self._get_runtime_scenario_settings()

        if self.robot_model is None:
            log_warning(tr("log.no_robot_model", "Robot model not set"))
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                tr("messages.no_robot_model", "Robot model not set")
            )
            return

        if self.simulation_thread and self.simulation_thread.isRunning():
            log_warning(tr("log.simulation_running", "Simulation is already running"))
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                tr("messages.simulation_running", "Simulation is already running")
            )
            return

        log_info(tr("log.test_lift_leg_start", "Starting lift leg action test"))
        self.status.showMessage(
            tr("status.executing_action", "Executing lift leg action...")
        )

        # Create simulation thread
        self.simulation_thread = SimulationThread(
            self.robot_model,
            "lift_right_leg"
        )

        # Connect signals
        self.simulation_thread.simulation_started.connect(
            self._on_simulation_started
        )
        self.simulation_thread.simulation_finished.connect(
            self._on_simulation_finished
        )
        self.simulation_thread.error_occurred.connect(
            self._on_simulation_error
        )

        # Start thread
        self.simulation_thread.start()

    def _on_node_requested(self, payload: dict):
        """Create node from node library double-click"""
        if not payload:
            return
        title = payload.get("title", "Unknown")
        grad = tuple(payload.get("grad", ["#45a049", "#4CAF50"]))
        features = payload.get("features", [])
        preset = payload.get("preset")

        if not hasattr(self, "graph_view") or not hasattr(self, "graph_scene"):
            return

        center = self.graph_view.viewport().rect().center()
        scene_pos = self.graph_view.mapToScene(center)
        node_item = self.graph_scene.create_node(title, scene_pos, features, grad)
        if preset and hasattr(node_item, "_combo") and node_item._combo:
            node_item._combo.setCurrentText(preset)

    def _open_nested_editor(
        self,
        kind: str,
        node_id: int,
        ref: str = "",
        node_item=None,
        display_name: str = "",
        intent_id: str = "",
    ):
        """Open a nested editor for Behavior/Script nodes, or Settings."""
        if kind == "settings":
            self.main_zone.open_settings_tab()
            return

        if kind == "behavior":
            node_name = display_name if display_name else f"Node {node_id}"
            self.main_zone.open_behavior_tab(
                node_name=node_name,
                node_id=node_id,
                behavior_ref=ref,
                intent_id=intent_id,
            )
            return

        if kind == "script":
            script_name = (ref or "").strip() or f"script_{node_id}"
            io_spec = (
                getattr(node_item, "_script_io_spec", None)
                if node_item is not None
                else None
            ) or {"inputs": [], "outputs": []}
            self.main_zone.open_script_tab(
                node_id=node_id,
                script_name=script_name,
                io_spec=io_spec,
                node_item=node_item,
                graph_scene=self.graph_scene,
            )
            return

        detail = f"{kind}#{node_id}"
        if ref:
            detail = f"{detail} ({ref})"
        msg = f"Nested editor requested: {detail}"
        log_info(msg)
        self.status.showMessage(msg, 3000)

    # 鈹€鈹€ Settings validation pre-flight (Cycle 2 STAGE-05) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _validate_settings_pre_run(self) -> bool:
        """Validate current SDK settings before execution starts.

        Returns:
            True  鈥?validation passed; run may proceed.
            False 鈥?validation failed; run is blocked and error UX was shown.

        On failure:
        - Surfaces error in ``ExecutionSummaryBar`` and ``DiagnosticsPanel``.
        - Refreshes the capability inspector so missing fields are highlighted.
        - Switches to the Settings tab and scrolls to the first missing field.

        Resilient: if the validator itself raises, returns True (do not block
        the run on an unexpected validator failure).
        """
        from system.service.settings_validator import validate_settings
        from bin.core.error_ux import format_settings_validation_error

        brand  = RobotContext.get_current_brand()
        config = self.main_zone.get_sdk_settings()

        try:
            result = validate_settings(brand, config)
        except Exception as exc:
            log_warning(f"settings pre-run validator raised unexpectedly: {exc}")
            return True  # Resilient: don't block on unexpected validator error

        if result.get("status") == "ok":
            return True

        # 鈹€鈹€ Validation failed 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        missing: list = result.get("missing", [])
        invalid: list = result.get("invalid", [])
        log_warning(
            f"Run blocked 鈥?settings validation failed: brand={brand!r}, "
            f"missing={missing}, invalid={invalid}"
        )

        # Surface in ExecutionSummaryBar
        settings_run_result = {
            "status":    "failed",
            "reason":    "settings_validation_failed",
            "node_count": 0,
            "results":   {},
            "diagnostics": {
                "executed_count": 0,
                "failed_nodes":   ["settings"],
                "compat_path":    False,
            },
        }
        self.main_zone.clear_execution_summary()
        self.main_zone.show_execution_summary(settings_run_result)

        # Surface raw details in DiagnosticsPanel
        diag_info = format_settings_validation_error(brand, result)
        self.main_zone.show_diagnostics_panel([diag_info])

        # Refresh capability inspector so missing fields are highlighted
        self._refresh_capability_inspector()

        # Navigate to Settings tab 鈫?first problem field
        first_field = (missing or invalid or [None])[0]
        if first_field:
            self.main_zone._on_capability_focus_setting(first_field)

        return False

    # 鈹€鈹€ Unsaved settings guardrail (Cycle 3 STAGE-05) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _handle_unsaved_settings_guard(self, action_label: str = "") -> bool:
        """Prompt operator when there are unapplied settings / behavior changes.

        Shows a modal dialog with three choices: Apply & Continue, Continue
        Without Applying, or Cancel.  Returns True when it is safe to proceed
        with the calling action; False when the action must be aborted.

        Thread-safety / no-op contract
        --------------------------------
        Must only be called from the Qt main thread.  Returns True immediately
        (no dialog) when no unsaved changes are detected, ensuring zero overhead
        on the normal (already-applied) path.

        Args:
            action_label: Short description appended to the dialog text so the
                operator knows which action is gated.  Empty string 鈫?generic text.

        Returns:
            True  鈥?safe to proceed (no unsaved changes, or operator applied /
                    discarded changes).
            False 鈥?operator cancelled, or Apply was attempted but failed
                    validation (SettingsPanel inline errors are already shown).
        """
        settings_panel = getattr(self.main_zone, "settings_panel", None)
        settings_dirty = bool(
            settings_panel is not None and settings_panel.has_unsaved_changes()
        )

        if not settings_dirty:
            return True

        desc = f" {action_label}" if action_label else ""
        msg = QMessageBox(self)
        msg.setWindowTitle(
            tr("messages.unsaved_settings_title", "Unapplied Changes")
        )

        detail = (
            "You have unapplied settings changes. "
            f"What would you like to do{desc}?"
        )
        msg.setText(detail)

        apply_btn = None
        if settings_dirty:
            apply_btn = msg.addButton(
                tr("messages.apply_and_continue", "Apply && Continue"),
                QMessageBox.ButtonRole.AcceptRole,
            )
        discard_btn = msg.addButton(
                tr(
                    "messages.discard_and_continue",
                    "Continue Without Applying",
                ),
            QMessageBox.ButtonRole.DestructiveRole,
        )
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(apply_btn or discard_btn)
        msg.exec()

        clicked = msg.clickedButton()
        if apply_btn is not None and clicked is apply_btn:
            settings_panel._on_apply()
            if settings_panel.has_unsaved_changes():
                # Apply failed validation 鈥?SettingsPanel already shows inline errors.
                # Switch to Settings tab so operator sees the error labels.
                self.main_zone._on_capability_focus_setting("")
                return False
            return True
        if clicked is discard_btn:
            return True
        # Cancel or dialog dismissed
        return False

    def _refresh_behavior_node_actions(self, brand: str, robot_type: str) -> None:
        """Rebuild Behavior-node action dropdowns for the active robot model.

        Behavior authoring is sourced from the IR action registry, not from
        adapter-declared SDK actions. The active brand/model is used to filter
        the registry and show both available and unsupported actions with clear
        status indicators.
        """
        if not hasattr(self, "graph_scene") or self.graph_scene is None:
            return
        try:
            # Circle 4 Step 4.3: Show both available and unsupported actions with status indicators
            # This ensures the UI reflects the actual capability state per model_status_checklist.md
            all_descriptors = get_semantic_actions(brand, robot_type)
            
            # Log unsupported actions for diagnostics
            unsupported = [d for d in all_descriptors if d.availability == "unsupported"]
            if unsupported:
                for d in unsupported:
                    log_info(f"Forward not supported on {brand}/{robot_type}: {d.intent_id}")
            
            self.graph_scene.refresh_behavior_node_actions(all_descriptors)
        except Exception as exc:
            log_warning(f"behavior_node_actions: refresh failed 鈥?{exc}")

    # 鈹€鈹€ Capability inspector refresh (Cycle 2 STAGE-04) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _refresh_capability_inspector(self) -> None:
        """Fetch capability data from the active adapter and update the inspector.

        Trigger points: startup, brand/model change, settings apply/reset, run.
        All errors degrade gracefully 鈥?the inspector is left empty rather than
        crashing.  Errors are logged but never propagated to callers.
        """
        try:
            brand      = RobotContext.get_current_brand()
            robot_type = RobotContext.get_robot_type()
            _gs = getattr(self, "graph_scene", None)
            if _gs is not None:
                _gs_brand, _gs_rt = _gs.get_start_node_selection()
                if _gs_rt:
                    robot_type = _gs_rt
                else:
                    _gs_rt = str(getattr(_gs, "_robot_type", "") or "").strip().lower()
                    if _gs_rt:
                        robot_type = _gs_rt
                if _gs_brand:
                    brand = _gs_brand
                else:
                    _gs_brand = RobotContext._get_robot_brand_map().get(robot_type, "")
                    if _gs_brand:
                        brand = _gs_brand
            adapter    = RobotContext._create_adapter_for_brand(brand, robot_type)
            if adapter is None:
                # Adapter unavailable 鈥?clear inspector content while keeping
                # header label unchanged (last-known state preserved for operator).
                self.main_zone.set_adapter_capabilities({}, [])
                # Always pass brand/robot_type so BehaviorPanel topology stays
                # aligned with the selected model even without a live adapter.
                self._update_hb_catalog({"brand": brand, "robot_type": robot_type})
                self._update_hb_channel(adapter=None)  # Circle 2: fallback to mock
                self._run_compat_audit(brand, robot_type)
                return
            cap_dict: dict = adapter.capabilities()
        except Exception as exc:
            log_warning(f"capability_inspector: adapter fetch failed 鈥?{exc}")
            # Degrade gracefully 鈥?clear inspector content, keep header unchanged.
            try:
                self.main_zone.set_adapter_capabilities({}, [])
                self._update_hb_catalog({"brand": brand, "robot_type": robot_type})
                self._update_hb_channel(adapter=None)  # Circle 2: fallback to mock
                self._run_compat_audit(brand, robot_type)
            except Exception:
                pass
            return

        try:
            from bin.core.settings_form import compute_missing_settings
            required       = cap_dict.get("required_settings") or []
            current_config = self.main_zone.get_sdk_settings()
            missing        = compute_missing_settings(brand, current_config, required)
            self.main_zone.set_adapter_capabilities(cap_dict, missing)
            self._update_hb_catalog(cap_dict)
            self._update_hb_channel(adapter=adapter)  # Circle 2: runtime-backed channel
            # Step 1.7: run compat audit after catalog update (success path only)
            self._run_compat_audit(brand, robot_type)
        except Exception as exc:
            log_warning(f"capability_inspector: inspector update failed 鈥?{exc}")

    def _update_hb_catalog(self, cap_dict: dict) -> None:
        """Propagate capability data to HeartBeat node catalog in BehaviorPanel.

        Called from every code path inside _refresh_capability_inspector() so
        the HeartBeat Library always reflects the active robot's capability profile.
        Errors are logged and suppressed 鈥?never propagated to callers.
        """
        try:
            self.main_zone.behavior_panel.set_capability_profile(cap_dict)
        except Exception as exc:
            log_warning(f"hb_catalog: update failed 鈥?{exc}")

    def _update_hb_channel(self, adapter: object = None) -> None:
        """Swap the HB execution channel to a runtime-backed instance (Circle 2).

        Uses HBChannelFactory to prioritise HBRuntimeChannel when the shared
        behavior bridge is available; falls back to HBMockChannel otherwise.
        Errors are logged and suppressed 鈥?channel swap is best-effort.
        """
        try:
            from system.behavior.hb_channel import HBChannelFactory
            channel = HBChannelFactory.create(
                bridge=self._behavior_bridge,
                adapter=adapter,
            )
            self.main_zone.behavior_panel.set_hb_channel(channel)
        except Exception as exc:
            log_warning(f"hb_channel: channel update failed 鈥?{exc}")

    def _run_compat_audit(self, brand: str, robot_type: str) -> None:
        """Trigger model-switch compatibility audit on every brand/model change (Step 1.7).

        Fires on ALL paths of _refresh_capability_inspector() 鈥?including
        adapter=None and exception paths 鈥?so the compat alert bar and the
        MainZone DiagnosticsPanel always reflect the current capability profile.

        Pipeline:
            BehaviorPanel.run_compat_audit()
                鈫?HBCompatReport
                鈫?report_to_behavior_diagnostics()   [IR semantic diagnostics]
                鈫?DiagnosticsKey.COMPAT_DIAGNOSTICS dicts
                鈫?MainZonePanel.show_compat_report()  [aggregate alert surface]

        Errors are logged and suppressed 鈥?never propagated to callers.
        """
        try:
            report = self.main_zone.behavior_panel.run_compat_audit(brand, robot_type)
        except Exception as exc:
            log_warning(f"compat_audit: panel audit failed 鈥?{exc}")
            return
        try:
            from system.behavior.hb_compat_audit import report_to_behavior_diagnostics
            from system.runtime.contracts import DiagnosticsKey
            self.main_zone.clear_compat_report()
            diagnostics = report_to_behavior_diagnostics(report)
            diag_dicts = []
            for issue, diag in zip(report.issues, diagnostics):
                d = diag.to_dict()
                if issue.intent_id is not None:
                    d["intent_id"] = issue.intent_id
                    d["source_raw_action"] = issue.source_raw_action
                    d["target_raw_action"] = issue.target_raw_action
                diag_dicts.append(d)
            if diag_dicts:
                log_warning(
                    f"compat_audit: {report.summary_text()} - "
                    f"{len(diag_dicts)} {DiagnosticsKey.COMPAT_DIAGNOSTICS} record(s)"
                )
                for d in diag_dicts:
                    level = str(d.get("level", "warning") or "warning").lower()
                    code = str(d.get("code", "") or "")
                    location = str(d.get("location", "") or "")
                    message = str(d.get("message", "") or "")
                    line = f"compat_audit[{code}] {location}: {message}" if location else f"compat_audit[{code}] {message}"
                    if level == "error":
                        log_error(line)
                    else:
                        log_warning(line)
        except Exception as exc:
            log_warning(f"compat_audit: diagnostic routing failed ? {exc}")

    def _on_sdk_settings_changed(self, _config: dict = None) -> None:
        """Called when SDK settings are applied; refreshes the capability inspector."""
        config = _config if isinstance(_config, dict) else {}
        robot_type = str(config.get("robot_type") or "").strip().lower()
        if robot_type:
            RobotContext.set_robot_type(robot_type)
            self.scenario_state.robot_type = robot_type
            if hasattr(self, 'graph_scene'):
                # Update the cache only 鈥?do NOT call set_robot_type() which
                # would also overwrite the Start node combo and erase the
                # user's canvas model selection.
                self.graph_scene._robot_type = robot_type
        self._refresh_capability_inspector()
        self._sync_behavior_simulation_mode()

    def _sync_behavior_simulation_mode(self) -> None:
        """Push current scenario target to BehaviorPanel.set_simulation_mode()."""
        try:
            scenario_cfg = self.main_zone.get_scenario_settings() or {}
            is_sim = scenario_cfg.get("target", "simulation") == "simulation"
            self.main_zone.behavior_panel.set_simulation_mode(is_sim)
        except Exception:  # noqa: BLE001
            pass

    def closeEvent(self, event):
        """Window close event"""
        # Cycle 3 STAGE-05: guard against unapplied settings before close.
        if not self._handle_unsaved_settings_guard(
            tr("messages.before_close", "before closing")
        ):
            event.ignore()
            return

        # Stop simulation thread
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait(3000)  # Wait up to 3 seconds
        if self.robot_model and hasattr(self.robot_model, "close_viewer"):
            try:
                self.robot_model.close_viewer()
            except Exception:
                pass
        try:
            from system.training.training_run_cache import get_training_run_cache

            get_training_run_cache().clear_all()
        except Exception:
            pass

        log_info(tr("log.main_window_closed", "Main window closed"))
        event.accept()

    def _on_theme_toggle(self):
        """Toggle theme between light/dark"""
        next_theme = "light" if self._theme == "dark" else "dark"
        self._apply_theme(next_theme, persist=True)

    def _apply_theme(self, theme: str, persist: bool = True):
        """Apply theme and refresh UI"""
        theme = (theme or "dark").lower()
        if theme not in ("light", "dark"):
            theme = "dark"
        self._theme = theme
        set_theme(theme)
        if persist:
            self.config.set('PREFERENCES', 'theme', theme, config_type='user')
            self.config.save_user_config()
        self._refresh_theme()
        self._sync_theme_button()

    def _sync_theme_button(self):
        """Sync theme toggle button label"""
        if not hasattr(self, "theme_button"):
            return
        self.theme_button.setText("")
        self.theme_button.setToolTip(tr("toolbar.theme_toggle", "Toggle theme"))

    def _toggle_window_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._sync_window_controls()

    def _sync_window_controls(self) -> None:
        if hasattr(self, "window_controls") and self.window_controls is not None:
            self.window_controls.set_fullscreen(self.isFullScreen())
        if getattr(self, "_training_page", None) is not None and hasattr(self._training_page, "_sync_window_controls"):
            self._training_page._sync_window_controls()

    def _refresh_theme(self):
        """Refresh theme styles across components"""
        # Reload ui.ini color table so startup and runtime refresh paths are consistent.
        get_color_slot().reload()
        self._apply_stylesheet()
        if hasattr(self, "sidebar"):
            self.sidebar.refresh_icons(self._theme)
        if hasattr(self, "window_controls") and self.window_controls is not None:
            self.window_controls.apply_theme()
            self._sync_window_controls()
        if hasattr(self, "cmd_log"):
            self.cmd_log.refresh_style()
        if hasattr(self, "module_palette"):
            self.module_palette.refresh_style()
        if hasattr(self, "code_editor"):
            self.code_editor.refresh_style()
        if hasattr(self, "graph_scene"):
            self.graph_scene.refresh_style()
        if hasattr(self, "main_zone") and hasattr(self.main_zone, "refresh_style"):
            self.main_zone.refresh_style()
        if hasattr(self, "main_zone") and hasattr(self.main_zone, "refresh_texts"):
            self.main_zone.refresh_texts()
        if getattr(self, '_training_page', None) is not None:
            try:
                if hasattr(self._training_page, 'apply_theme'):
                    self._training_page.apply_theme()
            except RuntimeError:
                pass








