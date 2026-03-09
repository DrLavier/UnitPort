#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main UI Module
Contains MainWindow and main UI components
"""

from PySide6.QtCore import Qt, QEvent, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QLabel, QComboBox, QMessageBox, QPushButton, QSizePolicy,
    QFileDialog,
)
import json
import os
from pathlib import Path

from bin.layout import MainZonePanel
from bin.scenario import ScenarioPanelState
from system.runtime import RuntimeEngine
from bin.core.simulation_thread import SimulationThread
from bin.core.mission_run_thread import MissionRunThread  # Cycle 2 STAGE-06
from bin.core.config_manager import ConfigManager
from bin.core.data_manager import get_value, load_data, up_data
from bin.core.theme_manager import get_color, get_color_slot, get_font_size, set_theme
from bin.core.logger import CmdLogWidget, log_info, log_success, log_warning, log_error, log_debug
from bin.core.localisation import get_localisation, tr
from bin.core.robot_context import RobotContext


class MainWindow(QMainWindow):
    """Main Window"""

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.robot_model = None
        self.simulation_thread = None
        self._runtime_paused = False
        self._current_workflow_path = ""
        # Cycle 2 STAGE-06: background mission run thread + cached exec_graph.
        self._mission_run_thread = None
        self._last_exec_graph: dict = {}
        self.runtime_engine = RuntimeEngine()
        self.scenario_state = ScenarioPanelState()

        # Circle 2: shared BehaviorCompilerBridge — owned here, injected into
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
        """Initialize UI"""
        # Read window size from config
        width = self.config.get_int('UI', 'window_width', fallback=1400)
        height = self.config.get_int('UI', 'window_height', fallback=900)

        self.setWindowTitle(tr("app.title", "UnitPort - Robot Visual Programming Platform"))
        self.resize(width, height)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Root layout: workspace row only
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Workspace layout: user zone + (main zone + cmd zone)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        root_layout.addLayout(main_layout, 1)

        # Left: user zone (collapsed by default, expands on hover)
        self._user_zone_collapsed_width = 56
        self._user_zone_expanded_width = 180
        self.user_zone = QWidget()
        self.user_zone.setObjectName("userZone")
        self.user_zone.setFixedWidth(self._user_zone_collapsed_width)
        self.user_zone.installEventFilter(self)
        user_layout = QVBoxLayout(self.user_zone)
        user_layout.setContentsMargins(8, 10, 8, 10)
        user_layout.setSpacing(8)

        self._user_zone_buttons = []
        placeholders = [
            ("U", "User"),
            ("P", "Projects"),
            ("A", "Assets"),
            ("S", "Settings"),
        ]
        for short_text, full_text in placeholders:
            btn = QPushButton(short_text)
            btn.setProperty("short_text", short_text)
            btn.setProperty("full_text", full_text)
            btn.setObjectName("userZoneButton")
            btn.setToolTip(full_text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.installEventFilter(self)
            btn.clicked.connect(lambda _, t=full_text: log_info(f"user_zone placeholder clicked: {t}"))
            user_layout.addWidget(btn)
            self._user_zone_buttons.append(btn)
        user_layout.addStretch()

        # Bottom controls: theme + language buttons only (no title text).
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("userZoneButton")
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.clicked.connect(self._on_theme_toggle)
        user_layout.addWidget(self.theme_button)

        self.language_button = QPushButton("EN")
        self.language_button.setObjectName("userZoneButton")
        self.language_button.setCursor(Qt.PointingHandCursor)
        self.language_button.clicked.connect(self._on_language_button_clicked)
        user_layout.addWidget(self.language_button)
        self._sync_theme_button()
        main_layout.addWidget(self.user_zone)

        # Main and right-zone splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter, 1)
        self._cmd_collapsed = False
        self._cmd_last_width = 340

        # Right: built-in CMD console
        self.cmd_log = CmdLogWidget()
        self.cmd_log.setMinimumWidth(0)

        # Main zone: delegated to dedicated panel component
        self.main_zone = MainZonePanel()
        self.runtime_zone = self.main_zone.runtime_zone
        self.mission_zone = self.main_zone.mission_zone
        self.tool_row = self.main_zone.tool_row
        self.module_palette = self.main_zone.module_palette
        self.graph_scene = self.main_zone.graph_scene
        self.graph_view = self.main_zone.graph_view
        self.code_editor = self.main_zone.code_editor
        self.graph_scene.set_subgraph_opener(self._open_nested_editor)
        self.graph_scene.set_script_tab_closer(self.main_zone.close_script_tab)
        self.graph_scene.set_script_tab_renamer(self.main_zone.rename_script_tab)
        self.module_palette.node_requested.connect(self._on_node_requested)
        self.main_zone.start_requested.connect(self._on_run)
        self.main_zone.pause_requested.connect(self._on_runtime_pause)
        self.main_zone.abort_requested.connect(self._on_runtime_abort)
        self.main_zone.reset_requested.connect(self._on_runtime_reset)
        self.main_zone.workflow_load_requested.connect(self._on_open)
        self.main_zone.workflow_save_requested.connect(self._on_save)
        self.main_zone.workflow_save_as_requested.connect(self._on_save_as)
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

        # Add to main splitter
        self.main_splitter.addWidget(self.main_zone)
        self.main_splitter.addWidget(self.cmd_log)
        self.main_splitter.setSizes([1260, self._cmd_last_width])
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)

        # Apply stylesheet
        self._apply_stylesheet()
        self._set_current_workflow_path("")
        log_debug(tr("log.ui_layout_created", "UI layout created"))
        log_info(tr("log.graph_editor_ready", "Graph editor ready, drag modules from left panel to canvas"))
        QTimer.singleShot(0, self.graph_view.recenter_to_origin)

    def _apply_stylesheet(self):
        """Apply stylesheet"""
        try:
            bg = get_color('bg', '#1e1e1e')
            card_bg = get_color('card_bg', '#2d2d2d')
            border = get_color('border', '#444444')
            text_primary = get_color('text_primary', '#ffffff')
            text_secondary = get_color('text_secondary', '#cccccc')
            hover_bg = get_color('hover_bg', '#3d3d3d')
            tab_bg = get_color('tab_bg', '#252525')
            tab_bg_hover = get_color('tab_bg_hover', '#3d3d3d')
            tab_bg_checked = get_color('tab_bg_checked', '#4CAF50')
            tab_text = get_color('tab_text', '#aaaaaa')
            tab_text_hover = get_color('tab_text_hover', '#ffffff')
            tab_text_checked = get_color('tab_text_checked', '#ffffff')
        except:
            # Fallback
            bg = '#1e1e1e'
            card_bg = '#2d2d2d'
            border = '#444444'
            text_primary = '#ffffff'
            text_secondary = '#cccccc'
            hover_bg = '#3d3d3d'
            tab_bg = '#252525'
            tab_bg_hover = '#3d3d3d'
            tab_bg_checked = '#4CAF50'
            tab_text = '#aaaaaa'
            tab_text_hover = '#ffffff'
            tab_text_checked = '#ffffff'

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg};
            }}
            QWidget {{
                color: {text_primary};
            }}
            QLabel {{
                background-color: {card_bg};
                border-radius: 12px;
                padding: 2px;
            }}
            #userZone {{
                background-color: {card_bg};
                border-right: 1px solid {border};
            }}
            #mainZone {{
                background-color: {bg};
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
                background-color: {bg};
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
                background-color: {bg};
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
            #userZoneButton {{
                background-color: rgba(34, 34, 34, 0.22);
                color: {text_primary};
                border: 1px solid rgba(20, 20, 20, 0.20);
                border-radius: 8px;
                padding: 6px 8px;
                text-align: left;
                font-weight: 600;
            }}
            #userZoneButton:hover {{
                background-color: rgba(34, 34, 34, 0.34);
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
                border: 1px solid {border};
                border-top: none;
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

    def eventFilter(self, obj, event):
        """Handle hover behavior for collapsible user zone."""
        if hasattr(self, "user_zone"):
            if obj is self.user_zone:
                if event.type() == QEvent.Type.Leave:
                    QTimer.singleShot(60, self._collapse_user_zone_if_needed)
            elif obj in getattr(self, "_user_zone_buttons", []):
                if event.type() == QEvent.Type.Enter:
                    self._set_user_zone_expanded(True)
                elif event.type() == QEvent.Type.Leave:
                    QTimer.singleShot(60, self._collapse_user_zone_if_needed)
        return super().eventFilter(obj, event)

    def _collapse_user_zone_if_needed(self):
        if not self.user_zone.underMouse():
            self._set_user_zone_expanded(False)

    def _set_user_zone_expanded(self, expanded: bool):
        width = self._user_zone_expanded_width if expanded else self._user_zone_collapsed_width
        if self.user_zone.width() != width:
            self.user_zone.setFixedWidth(width)
        for btn in self._user_zone_buttons:
            btn.setText(btn.property("full_text") if expanded else btn.property("short_text"))

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
        """Initialize status bar"""
        self.status = QStatusBar()
        self.setStatusBar(self.status)

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

        # Update graph scene robot type
        if hasattr(self, 'graph_scene'):
            self.graph_scene.set_robot_type(robot_type)

        # Get robot model from context
        self.robot_model = RobotContext.get_robot_model()

        # Sync settings panel brand + refresh capability inspector (Cycle 2 STAGE-04)
        brand = RobotContext.get_current_brand()
        self.main_zone.set_sdk_brand(brand)
        self._refresh_capability_inspector()

    def _get_runtime_scenario_settings(self) -> dict:
        """Read current settings from Scenario panel and apply env overrides."""
        settings = {}
        if hasattr(self.main_zone, "get_scenario_settings"):
            settings = self.main_zone.get_scenario_settings() or {}

        gl_backend = settings.get("mujoco_gl_backend")
        if gl_backend:
            os.environ["MUJOCO_GL"] = str(gl_backend)
        self.scenario_state.params = dict(settings)

        return settings

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

    def _set_current_workflow_path(self, path: str) -> None:
        normalized_path = os.path.abspath(path) if path else ""
        self._current_workflow_path = normalized_path
        title = os.path.basename(normalized_path) if normalized_path else "[New File]"
        compiler_title = os.path.basename(normalized_path) if normalized_path else "New File"
        if hasattr(self, "main_zone"):
            self.main_zone.set_workflow_tab_title(title or "[New File]")
            self.main_zone.set_compiler_main_tab_title(compiler_title)

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

    def _on_open(self):
        """Open project — shows file dialog and loads a mission file."""
        # Stage 4: unified unsaved guard (settings + behavior) before navigation.
        if not self._handle_unsaved_settings_guard(
            tr("messages.before_open", "before opening a file")
        ):
            return

        from bin.core.mission_persistence import validate_mission_schema

        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("toolbar.open", "Open Mission"),
            self._workflows_root(),
            "Mission Files (*.unitport);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

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
            return

        ok, reason = validate_mission_schema(data)
        if not ok:
            log_error(tr("log.open_schema_error", "Mission schema invalid: {reason}", reason=reason))
            QMessageBox.warning(
                self,
                tr("messages.schema_mismatch", "Schema Mismatch"),
                tr(
                    "messages.schema_mismatch_detail",
                    "Cannot load mission — schema check failed:\n{reason}",
                    reason=reason,
                ),
            )
            return

        if not hasattr(self, "graph_scene"):
            log_error("Graph scene not available")
            return

        # Step 5 — emit migration warnings for pre-1.4 files before loading.
        try:
            from bin.core.mission_persistence import migrate_mission_payload
            _migration = migrate_mission_payload(data)
            for _w in _migration.get("warnings", []):
                log_info(_w)
        except Exception:
            pass

        self.graph_scene.load_workflow(data)
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
                            "Mission settings brand '{loaded}' differs from active brand '{current}' — skipping restore",
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
        """Run the connected workflow (async, non-blocking — Cycle 2 STAGE-06)."""
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

        runtime_model = RobotContext.get_robot_model()
        self.robot_model = runtime_model

        scenario_settings = self._get_runtime_scenario_settings()
        scenario = self.scenario_state.to_runtime_scenario(
            target="simulation",
            robot_model=runtime_model,
            graph_scene=self.graph_scene,
            simulation_running=bool(self.simulation_thread and self.simulation_thread.isRunning()),
            **scenario_settings,
        )

        # Task 1 (STAGE-06): inject validated SDK settings into scenario so the
        # service lifecycle can access them without bypassing RobotContext.
        scenario["sdk_settings"] = self.main_zone.get_sdk_settings()
        scenario["brand"]        = RobotContext.get_current_brand()

        # Cycle 3 STAGE-03 (fix): build a per-run policy and pass it directly to
        # MissionRunThread.  The thread activates it as a thread-local so all
        # RobotContext calls from the worker use the run-scoped settings without
        # ever mutating the class-level _lifecycle_policy.
        _run_policy = None
        try:
            _sdk = self.main_zone.get_sdk_settings()
            _brand = RobotContext.get_current_brand()
            _run_policy = RobotContext.make_run_policy({**_sdk, "brand": _brand})
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
                    log_warning(f"Circle 3: bridge attach failed — {_exc}")
                mission_ir = _compiled
                _behavior_path_reason = "workflowir_behavior_enabled"
                log_info("Circle 1: behavior-enabled run — using WorkflowIR execution path")
            else:
                _behavior_path_reason = "workflowir_compile_failed_fallback"
                log_warning(
                    "Circle 1: WorkflowIR compilation failed — "
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

    # ── Async mission execution slots (Cycle 2 STAGE-06) ─────────────────────

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

        reason = run_result.get("reason", "")
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
        executed_count = len(run_result.get("results", {}))
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

        Nodes present in results without errors → "success".
        Nodes in diagnostics.failed_nodes             → "failed".
        Nodes in exec_graph but absent from results   → "skipped".
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

    # ── Circle 1 Step 1.1 helpers ─────────────────────────────────────────────

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
                    f"Circle 1: Canvas→IR had {len(errors)} error(s);"
                    " falling back to exec_graph path"
                )
                return None
            return ir
        except Exception as exc:  # noqa: BLE001
            log_warning(f"Circle 1: Canvas→IR compilation raised {exc}; using exec_graph fallback")
            return None

    # ── end Circle 1 helpers ──────────────────────────────────────────────────

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
            lambda msg: self.status.showMessage(msg)
        )
        self.simulation_thread.simulation_finished.connect(
            lambda msg: self.status.showMessage(msg, 3000)
        )
        self.simulation_thread.error_occurred.connect(
            lambda msg: QMessageBox.critical(self, "Error", msg)
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

    def _open_nested_editor(self, kind: str, node_id: int, ref: str = "", node_item=None):
        """Open a nested editor for Behavior/Script nodes, or Settings."""
        if kind == "settings":
            self.main_zone.open_settings_tab()
            return

        if kind == "behavior":
            node_name = ref if ref else f"Node {node_id}"
            self.main_zone.open_behavior_tab(node_name=node_name, node_id=node_id)
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

    # ── Settings validation pre-flight (Cycle 2 STAGE-05) ───────────────────

    def _validate_settings_pre_run(self) -> bool:
        """Validate current SDK settings before execution starts.

        Returns:
            True  — validation passed; run may proceed.
            False — validation failed; run is blocked and error UX was shown.

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

        # ── Validation failed ────────────────────────────────────────────
        missing: list = result.get("missing", [])
        invalid: list = result.get("invalid", [])
        log_warning(
            f"Run blocked — settings validation failed: brand={brand!r}, "
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

        # Navigate to Settings tab → first problem field
        first_field = (missing or invalid or [None])[0]
        if first_field:
            self.main_zone._on_capability_focus_setting(first_field)

        return False

    # ── Unsaved settings guardrail (Cycle 3 STAGE-05) ────────────────────────

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
                operator knows which action is gated.  Empty string → generic text.

        Returns:
            True  — safe to proceed (no unsaved changes, or operator applied /
                    discarded changes).
            False — operator cancelled, or Apply was attempted but failed
                    validation (SettingsPanel inline errors are already shown).
        """
        settings_panel = getattr(self.main_zone, "settings_panel", None)
        behavior_panel = getattr(self.main_zone, "behavior_panel", None)
        settings_dirty = bool(
            settings_panel is not None and settings_panel.has_unsaved_changes()
        )
        behavior_dirty = bool(
            behavior_panel is not None
            and hasattr(behavior_panel, "has_unsaved_changes")
            and behavior_panel.has_unsaved_changes()
        )

        if not settings_dirty and not behavior_dirty:
            return True

        desc = f" {action_label}" if action_label else ""
        msg = QMessageBox(self)
        msg.setWindowTitle(
            tr("messages.unsaved_settings_title", "Unapplied Changes")
        )

        if settings_dirty and behavior_dirty:
            detail = (
                "You have unapplied settings changes and uncompiled behavior edits. "
                f"What would you like to do{desc}?"
            )
        elif settings_dirty:
            detail = (
                "You have unapplied settings changes. "
                f"What would you like to do{desc}?"
            )
        else:
            detail = (
                "You have uncompiled behavior edits. "
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
                "Continue Without Applying/Compiling",
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
                # Apply failed validation — SettingsPanel already shows inline errors.
                # Switch to Settings tab so operator sees the error labels.
                self.main_zone._on_capability_focus_setting("")
                return False
            return True
        if clicked is discard_btn:
            return True
        # Cancel or dialog dismissed
        return False

    # ── Capability inspector refresh (Cycle 2 STAGE-04) ──────────────────────

    def _refresh_capability_inspector(self) -> None:
        """Fetch capability data from the active adapter and update the inspector.

        Trigger points: startup, brand/model change, settings apply/reset, run.
        All errors degrade gracefully — the inspector is left empty rather than
        crashing.  Errors are logged but never propagated to callers.
        """
        try:
            brand      = RobotContext.get_current_brand()
            robot_type = RobotContext.get_robot_type()
            adapter    = RobotContext._create_adapter_for_brand(brand, robot_type)
            if adapter is None:
                # Cycle 3 STAGE-06: clear inspector when no adapter is available
                # so stale capability data from a prior brand is not shown.
                self.main_zone.set_adapter_capabilities({}, [])
                self._update_hb_catalog({})
                self._update_hb_channel(adapter=None)  # Circle 2: fallback to mock
                # Step 1.7: audit must fire on every model-switch path, including
                # adapter=None (empty catalog → clean report clears compat bar).
                self._run_compat_audit(brand, robot_type)
                return
            cap_dict: dict = adapter.capabilities()
        except Exception as exc:
            log_warning(f"capability_inspector: adapter fetch failed — {exc}")
            # Cycle 3 STAGE-06: degrade gracefully — clear inspector on error
            # rather than leaving stale capability data from a previous run.
            try:
                self.main_zone.set_adapter_capabilities({}, [])
                self._update_hb_catalog({})
                self._update_hb_channel(adapter=None)  # Circle 2: fallback to mock
                # Step 1.7: audit on exception path too (empty catalog).
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
            log_warning(f"capability_inspector: inspector update failed — {exc}")

    def _update_hb_catalog(self, cap_dict: dict) -> None:
        """Propagate capability data to HeartBeat node catalog in BehaviorPanel.

        Called from every code path inside _refresh_capability_inspector() so
        the HeartBeat Library always reflects the active robot's capability profile.
        Errors are logged and suppressed — never propagated to callers.
        """
        try:
            self.main_zone.behavior_panel.set_capability_profile(cap_dict)
        except Exception as exc:
            log_warning(f"hb_catalog: update failed — {exc}")

    def _update_hb_channel(self, adapter: object = None) -> None:
        """Swap the HB execution channel to a runtime-backed instance (Circle 2).

        Uses HBChannelFactory to prioritise HBRuntimeChannel when the shared
        behavior bridge is available; falls back to HBMockChannel otherwise.
        Errors are logged and suppressed — channel swap is best-effort.
        """
        try:
            from system.behavior.hb_channel import HBChannelFactory
            channel = HBChannelFactory.create(
                bridge=self._behavior_bridge,
                adapter=adapter,
            )
            self.main_zone.behavior_panel.set_hb_channel(channel)
        except Exception as exc:
            log_warning(f"hb_channel: channel update failed — {exc}")

    def _run_compat_audit(self, brand: str, robot_type: str) -> None:
        """Trigger model-switch compatibility audit on every brand/model change (Step 1.7).

        Fires on ALL paths of _refresh_capability_inspector() — including
        adapter=None and exception paths — so the compat alert bar and the
        MainZone DiagnosticsPanel always reflect the current capability profile.

        Pipeline:
            BehaviorPanel.run_compat_audit()
                → HBCompatReport
                → report_to_behavior_diagnostics()   [IR semantic diagnostics]
                → DiagnosticsKey.COMPAT_DIAGNOSTICS dicts
                → MainZonePanel.show_compat_report()  [aggregate alert surface]

        Errors are logged and suppressed — never propagated to callers.
        """
        try:
            report = self.main_zone.behavior_panel.run_compat_audit(brand, robot_type)
        except Exception as exc:
            log_warning(f"compat_audit: panel audit failed — {exc}")
            return

        try:
            from system.behavior.hb_compat_audit import report_to_behavior_diagnostics
            from system.runtime.contracts import DiagnosticsKey
            diag_dicts = [d.to_dict() for d in report_to_behavior_diagnostics(report)]
            self.main_zone.show_compat_report(report, diag_dicts)
            if diag_dicts:
                log_warning(
                    f"compat_audit: {report.summary_text()} — "
                    f"{len(diag_dicts)} {DiagnosticsKey.COMPAT_DIAGNOSTICS} record(s)"
                )
        except Exception as exc:
            log_warning(f"compat_audit: diagnostic routing failed — {exc}")

    def _on_sdk_settings_changed(self, _config: dict = None) -> None:
        """Called when SDK settings are applied; refreshes the capability inspector."""
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
        if self._theme == "dark":
            self.theme_button.setText("🌙")
        else:
            self.theme_button.setText("☀️")
        self.theme_button.setToolTip(tr("toolbar.theme_toggle", "Toggle theme"))

    def _refresh_theme(self):
        """Refresh theme styles across components"""
        # Reload ui.ini color table so startup and runtime refresh paths are consistent.
        get_color_slot().reload()
        self._apply_stylesheet()
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

