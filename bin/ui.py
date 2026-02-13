#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main UI Module
Contains MainWindow and main UI components
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QStatusBar, QLabel, QComboBox, QMessageBox, QPushButton
)

from bin.components.code_editor import CodeEditor
from bin.components.graph_scene import GraphScene
from bin.components.graph_view import GraphView
from bin.components.module_cards import ModulePalette
from bin.core.simulation_thread import SimulationThread
from bin.core.config_manager import ConfigManager
from bin.core.data_manager import get_value, load_data, up_data
from bin.core.theme_manager import get_color, get_font_size, set_theme
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

        # Load UI config
        self._load_ui_config()

        self._init_ui()
        self._init_toolbar()
        self._init_statusbar()

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

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main splitter: Log + Middle workspace + Right code editor
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Log display
        self.cmd_log = CmdLogWidget()
        self.cmd_log.setMinimumWidth(300)

        # Middle: Module palette + Graph editor
        middle_widget = QWidget()
        middle_layout = QHBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)

        # Middle splitter
        middle_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Module palette
        self.module_palette = ModulePalette()
        self.module_palette.node_requested.connect(self._on_node_requested)

        # Graph editor
        self.graph_scene = GraphScene()
        self.graph_view = GraphView(self.graph_scene)

        middle_splitter.addWidget(self.module_palette)
        middle_splitter.addWidget(self.graph_view)
        middle_splitter.setSizes([240, 760])
        middle_splitter.setStretchFactor(0, 0)  # Palette does not stretch
        middle_splitter.setStretchFactor(1, 1)  # Graph takes remaining space

        middle_layout.addWidget(middle_splitter)

        # Right: Code editor
        self.code_editor = CodeEditor()

        # Connect graph scene and code editor (bidirectional)
        self.graph_scene.set_code_editor(self.code_editor)
        self.code_editor.set_graph_scene(self.graph_scene)

        # Add to main splitter
        self.main_splitter.addWidget(self.cmd_log)
        self.main_splitter.addWidget(middle_widget)
        self.main_splitter.addWidget(self.code_editor)

        # Set main splitter ratio (Log:Graph editor:Code editor = 1:3:1.5)
        self.main_splitter.setSizes([300, 900, 400])

        main_layout.addWidget(self.main_splitter)

        # Apply stylesheet
        self._apply_stylesheet()

        log_debug(tr("log.ui_layout_created", "UI layout created"))
        log_info(tr("log.graph_editor_ready", "Graph editor ready, drag modules from left panel to canvas"))

    def _apply_stylesheet(self):
        """Apply stylesheet"""
        try:
            bg = get_color('bg', '#1e1e1e')
            card_bg = get_color('card_bg', '#2d2d2d')
            border = get_color('border', '#444444')
            text_primary = get_color('text_primary', '#ffffff')
            text_secondary = get_color('text_secondary', '#cccccc')
            hover_bg = get_color('hover_bg', '#3d3d3d')
        except:
            # Fallback
            bg = '#1e1e1e'
            card_bg = '#2d2d2d'
            border = '#444444'
            text_primary = '#ffffff'
            text_secondary = '#cccccc'
            hover_bg = '#3d3d3d'

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
            QToolBar {{
                background-color: {bg};
                border-bottom: 1px solid {border};
                spacing: 6px;
            }}
            QToolButton, QPushButton {{
                background-color: {card_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QToolButton:hover, QPushButton:hover {{
                background-color: {hover_bg};
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
        """)

    def _init_toolbar(self):
        """Initialize toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        # Robot type selection
        t1 = QLabel(f" {tr('toolbar.robot', 'Robot:')} ")
        t1.setMaximumHeight(35)
        toolbar.addWidget(t1)

        self.robot_combo = QComboBox()
        available_robots = self.config.get_available_robots()
        self.robot_combo.addItems(available_robots)
        default_robot = self.config.get('SIMULATION', 'default_robot', fallback='go2')
        self.robot_combo.setCurrentText(default_robot)
        self.robot_combo.currentTextChanged.connect(self._on_robot_type_changed)
        self.robot_combo.setMinimumWidth(80)
        toolbar.addWidget(self.robot_combo)

        toolbar.addSeparator()

        # Toolbar buttons
        actions = [
            (tr("toolbar.new", "New"), self._on_new),
            (tr("toolbar.open", "Open"), self._on_open),
            (tr("toolbar.save", "Save"), self._on_save),
            (tr("toolbar.export_code", "Export Code"), self._on_export_code),
            (tr("toolbar.run", "Run"), self._on_run)
        ]

        for text, handler in actions:
            action = QAction(text, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)

        toolbar.addSeparator()

        # Test button
        test_action = QAction(tr("toolbar.test_lift_leg", "Test Lift Leg"), self)
        test_action.triggered.connect(self._test_lift_leg)
        toolbar.addAction(test_action)

        # Add spacer to push language combo to the right
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        spacer.setMinimumWidth(20)
        toolbar.addWidget(spacer)

        # Flexible spacer
        flexible_spacer = QWidget()
        flexible_spacer.setSizePolicy(
            flexible_spacer.sizePolicy().Policy.Expanding,
            flexible_spacer.sizePolicy().Policy.Preferred
        )
        toolbar.addWidget(flexible_spacer)

        # Theme toggle (left of language selection)
        self.theme_button = QPushButton()
        self.theme_button.clicked.connect(self._on_theme_toggle)
        toolbar.addWidget(self.theme_button)
        self._sync_theme_button()

        # Language selection (right side)
        toolbar.addWidget(QLabel(f" {tr('toolbar.language', 'Language:')} "))

        self.language_combo = QComboBox()
        # Currently only English is available
        self.language_combo.addItem("EN", "en")
        self.language_combo.setMinimumWidth(60)
        self.language_combo.setCurrentIndex(0)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        toolbar.addWidget(self.language_combo)

    def _init_statusbar(self):
        """Initialize status bar"""
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Initialize RobotContext with default robot type
        robot_type = self.robot_combo.currentText()
        RobotContext.set_robot_type(robot_type)
        self.robot_model = RobotContext.get_robot_model()

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

        # Update graph scene robot type
        if hasattr(self, 'graph_scene'):
            self.graph_scene.set_robot_type(robot_type)

        # Get robot model from context
        self.robot_model = RobotContext.get_robot_model()

    def _on_language_changed(self, index: int):
        """Language changed"""
        lang_code = self.language_combo.currentData()
        loc = get_localisation()
        if loc.load_language(lang_code):
            log_info(f"Language changed to: {lang_code}")
            # Note: Full UI refresh would require more extensive changes
            # For now, new text will appear on next widget creation
            self._refresh_theme()

    def _on_new(self):
        """New project"""
        log_info(tr("log.new_project", "New project"))
        self.code_editor.clear()
        self.status.showMessage(tr("status.new_project", "New project"), 2000)

    def _on_open(self):
        """Open project"""
        log_info(tr("log.open_project", "Open project"))
        QMessageBox.information(
            self,
            tr("messages.info", "Info"),
            tr("messages.open_not_implemented", "Open project feature not implemented")
        )

    def _on_save(self):
        """Save project"""
        log_info(tr("log.save_project", "Save project"))
        QMessageBox.information(
            self,
            tr("messages.info", "Info"),
            tr("messages.save_not_implemented", "Save project feature not implemented")
        )

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
        """Run the connected workflow with control flow support"""
        log_info(tr("log.run", "Run"))

        # Get workflow data from graph scene
        if not hasattr(self, 'graph_scene'):
            log_error("Graph scene not available")
            return

        # Get execution graph with control flow information
        exec_graph = self.graph_scene.get_execution_graph()

        # Check if there are nodes
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

        # Check if simulation is already running
        if self.simulation_thread and self.simulation_thread.isRunning():
            log_warning(tr("log.simulation_running", "Simulation is already running"))
            QMessageBox.warning(
                self,
                tr("messages.warning", "Warning"),
                tr("messages.simulation_running", "Simulation is already running")
            )
            return

        # Check for action nodes and reset simulation
        has_action = any(
            node.get('type') in ('action_execution', 'stop')
            or "Action Execution" in node.get('name', '')
            for node in exec_graph['nodes'].values()
        )

        if has_action and self.robot_model is not None:
            try:
                reset_ok = True
                if hasattr(self.robot_model, "reset_simulation"):
                    reset_ok = self.robot_model.reset_simulation()
                if not reset_ok:
                    log_error("Simulation reset failed")
                    QMessageBox.warning(
                        self,
                        tr("messages.warning", "Warning"),
                        tr(
                            "messages.simulation_reset_failed",
                            "Failed to reset simulation. Please check MuJoCo setup."
                        )
                    )
                    return
            except Exception as e:
                log_error(f"Simulation reset failed: {e}")
                QMessageBox.warning(
                    self,
                    tr("messages.warning", "Warning"),
                    tr(
                        "messages.simulation_reset_failed",
                        "Failed to reset simulation. Please check MuJoCo setup."
                    )
                )
                return

        # Execute workflow with control flow
        log_info(f"Executing workflow with {len(exec_graph['nodes'])} nodes")
        self.status.showMessage(
            tr("status.executing_workflow", "Executing workflow...")
        )

        # Execute using recursive control flow
        results = {}
        executed = set()
        executed_count = [0]  # Use list to allow modification in nested function

        def execute_node(node_id):
            """Recursively execute a node and its downstream nodes"""
            if node_id in executed:
                return
            if node_id not in exec_graph['nodes']:
                return

            node_data = exec_graph['nodes'][node_id]
            node_name = node_data.get('name', '')
            node_type = node_data.get('type', 'unknown')
            logic_node = node_data.get('logic_node')
            outgoing = exec_graph['outgoing'].get(node_id, {})
            incoming = exec_graph['incoming'].get(node_id, {})

            executed.add(node_id)
            executed_count[0] += 1
            log_info(f"Executing node: {node_name} (ID: {node_id}, Type: {node_type})")

            # Handle Logic Control nodes (if/while/for)
            if "Logic Control" in node_name:
                ui_selection = node_data.get('ui_selection', 'If')

                if ui_selection.lower().startswith('if'):
                    # Evaluate condition
                    condition_result = self._evaluate_condition(node_id, node_data, exec_graph, results)
                    log_debug(f"If condition evaluated to: {condition_result}")

                    if condition_result:
                        # Execute true branch
                        true_targets = outgoing.get('out_if', [])
                        for target_id, _ in true_targets:
                            execute_node(target_id)
                    else:
                        # Execute false branch
                        false_targets = outgoing.get('out_else', [])
                        for target_id, _ in false_targets:
                            execute_node(target_id)

                elif ui_selection.lower().startswith('while'):
                    loop_type = node_data.get('loop_type', 'while')

                    if loop_type == 'for':
                        # For loop execution
                        try:
                            start = int(node_data.get('for_start', '0') or '0')
                            end = int(node_data.get('for_end', '10') or '10')
                            step = int(node_data.get('for_step', '1') or '1')
                        except ValueError:
                            start, end, step = 0, 10, 1

                        log_debug(f"For loop: range({start}, {end}, {step})")
                        for i in range(start, end, step):
                            results[f'{node_id}_i'] = i
                            # Execute loop body
                            body_targets = outgoing.get('loop_body', [])
                            for target_id, _ in body_targets:
                                executed.discard(target_id)  # Allow re-execution
                                execute_node(target_id)

                        # Execute loop end
                        end_targets = outgoing.get('loop_end', [])
                        for target_id, _ in end_targets:
                            execute_node(target_id)
                    else:
                        # While loop execution (with max iterations for safety)
                        max_iterations = 100
                        iteration = 0
                        while iteration < max_iterations:
                            condition_result = self._evaluate_condition(node_id, node_data, exec_graph, results)
                            if not condition_result:
                                break

                            # Execute loop body
                            body_targets = outgoing.get('loop_body', [])
                            for target_id, _ in body_targets:
                                executed.discard(target_id)  # Allow re-execution
                                execute_node(target_id)
                            iteration += 1

                        if iteration >= max_iterations:
                            log_warning(f"While loop exceeded max iterations ({max_iterations})")

                        # Execute loop end
                        end_targets = outgoing.get('loop_end', [])
                        for target_id, _ in end_targets:
                            execute_node(target_id)

                return  # Logic Control nodes don't have flow_out

            # Handle Condition nodes (comparison)
            elif "Condition" in node_name:
                result = self._execute_condition_node(node_id, node_data, exec_graph, results)
                results[node_id] = result
                # Condition nodes don't continue flow, they provide data
                return

            # Handle Action/Sensor/Other nodes
            else:
                if logic_node:
                    # Set robot model for action/sensor nodes
                    if node_type in ('action_execution', 'sensor_input', 'stop'):
                        logic_node.set_parameter('robot_model', self.robot_model)

                    # Collect inputs from connected nodes
                    inputs = {}
                    for port_name, sources in incoming.items():
                        for source_id, source_port in sources:
                            if source_id in results:
                                source_result = results[source_id]
                                if isinstance(source_result, dict) and source_port in source_result:
                                    inputs[port_name] = source_result[source_port]

                    # Execute the node
                    try:
                        result = logic_node.execute(inputs)
                        results[node_id] = result
                        log_debug(f"Node {node_id} result: {result}")
                    except Exception as e:
                        log_error(f"Node {node_id} execution failed: {e}")
                        results[node_id] = {'error': str(e)}
                else:
                    # Handle nodes without logic implementation
                    ui_selection = node_data.get('ui_selection', '')
                    if "Action Execution" in node_name and self.robot_model:
                        action = self.graph_scene._action_mapping.get(
                            ui_selection,
                            ui_selection.lower().replace(" ", "_")
                        )
                        log_info(f"Executing action: {action}")
                        try:
                            success = self.robot_model.run_action(action)
                            results[node_id] = {'status': 'success' if success else 'failed', 'action': action}
                        except Exception as e:
                            log_error(f"Action execution failed: {e}")
                            results[node_id] = {'error': str(e)}

                # Continue with flow_out
                flow_targets = outgoing.get('flow_out', [])
                for target_id, _ in flow_targets:
                    execute_node(target_id)

        # Execute from entry nodes
        for entry_id in exec_graph['entry_nodes']:
            execute_node(entry_id)

        # Show completion message
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
        log_success(f"Workflow completed. Executed {executed_count[0]} nodes.")

    def _evaluate_condition(self, node_id, node_data, exec_graph, results):
        """Evaluate condition for Logic Control nodes"""
        # First check if there's a Condition node connected
        incoming = exec_graph['incoming'].get(node_id, {})
        condition_sources = incoming.get('condition', [])

        if condition_sources:
            source_id, source_port = condition_sources[0]
            # Execute condition node if not already done
            if source_id not in results and source_id in exec_graph['nodes']:
                source_data = exec_graph['nodes'][source_id]
                result = self._execute_condition_node(source_id, source_data, exec_graph, results)
                results[source_id] = result

            # Get result from condition node
            if source_id in results:
                source_result = results[source_id]
                if isinstance(source_result, dict):
                    value = source_result.get('result', {}).get('value', False)
                    return bool(value)

        # Fallback: evaluate condition expression directly
        condition_expr = node_data.get('condition_expr', '')
        if condition_expr:
            try:
                # Safe evaluation of simple expressions
                return self._safe_eval_condition(condition_expr, results)
            except Exception as e:
                log_warning(f"Condition evaluation failed: {e}")
                return False

        return False

    def _execute_condition_node(self, node_id, node_data, exec_graph, results):
        """Execute a Condition (comparison) node"""
        logic_node = node_data.get('logic_node')

        if logic_node:
            # Collect inputs
            incoming = exec_graph['incoming'].get(node_id, {})
            inputs = {}

            for port_name, sources in incoming.items():
                for source_id, source_port in sources:
                    if source_id in results:
                        source_result = results[source_id]
                        if isinstance(source_result, dict) and source_port in source_result:
                            inputs[port_name] = source_result[source_port]

            # Use values from UI if not connected
            if 'left' not in inputs:
                left_val = node_data.get('left_value', '0')
                try:
                    inputs['left'] = float(left_val) if '.' in left_val else int(left_val)
                except ValueError:
                    inputs['left'] = left_val

            if 'right' not in inputs:
                right_val = node_data.get('right_value', '0')
                try:
                    inputs['right'] = float(right_val) if '.' in right_val else int(right_val)
                except ValueError:
                    inputs['right'] = right_val

            try:
                return logic_node.execute(inputs)
            except Exception as e:
                log_error(f"Condition node {node_id} execution failed: {e}")
                return {'result': {'value': False}}

        return {'result': {'value': False}}

    def _safe_eval_condition(self, expr, results):
        """Safely evaluate a condition expression"""
        expr = expr.strip()

        # Handle simple boolean literals
        if expr.lower() == 'true':
            return True
        if expr.lower() == 'false':
            return False

        # Handle simple numeric comparisons
        try:
            # Try to evaluate as a simple expression
            # Only allow safe operations
            allowed_names = {'True': True, 'False': False, 'None': None}
            # Add results variables
            for node_id, result in results.items():
                if isinstance(result, dict):
                    for key, val in result.items():
                        if isinstance(val, dict) and 'value' in val:
                            allowed_names[f'result_{node_id}_{key}'] = val['value']

            return eval(expr, {"__builtins__": {}}, allowed_names)
        except Exception:
            # If evaluation fails, return False
            return False

    def _test_lift_leg(self):
        """Test lift leg action"""
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

    def closeEvent(self, event):
        """Window close event"""
        # Stop simulation thread
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait(3000)  # Wait up to 3 seconds

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
            self.theme_button.setText(tr("toolbar.theme_dark", "Dark"))
        else:
            self.theme_button.setText(tr("toolbar.theme_light", "Light"))
        self.theme_button.setToolTip(tr("toolbar.theme_toggle", "Toggle theme"))

    def _refresh_theme(self):
        """Refresh theme styles across components"""
        self._apply_stylesheet()
        if hasattr(self, "cmd_log"):
            self.cmd_log.refresh_style()
        if hasattr(self, "module_palette"):
            self.module_palette.refresh_style()
        if hasattr(self, "code_editor"):
            self.code_editor.refresh_style()
        if hasattr(self, "graph_scene"):
            self.graph_scene.refresh_style()
