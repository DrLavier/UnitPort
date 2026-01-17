#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主UI模块
包含MainWindow和主要UI组件
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


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.robot_model = None
        self.simulation_thread = None
        
        # 加载UI配置
        self._load_ui_config()
        
        self._init_ui()
        self._init_toolbar()
        self._init_statusbar()
        
        log_info("主窗口初始化完成")
    
    def _load_ui_config(self):
        """加载UI配置"""
        ui_config_path = self.config.project_root / "config" / "ui.ini"
        load_data(str(ui_config_path))
        
        # 设置主题
        theme = self.config.get('PREFERENCES', 'theme', fallback='dark', config_type='user')
        set_theme(theme)
    
    def _init_ui(self):
        """初始化UI"""
        # 从配置读取窗口大小
        width = self.config.get_int('UI', 'window_width', fallback=1400)
        height = self.config.get_int('UI', 'window_height', fallback=900)
        
        self.setWindowTitle("Celebrimbor - 机器人可视化编程平台")
        self.resize(width, height)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建主分割器：日志 + 中间工作区 + 右侧代码编辑器
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：日志显示器
        self.cmd_log = CmdLogWidget()
        self.cmd_log.setMinimumWidth(300)
        
        # 中间：模块面板 + 图形编辑器
        middle_widget = QWidget()
        middle_layout = QHBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)
        
        # 中间分割器
        middle_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 模块面板
        self.module_palette = ModulePalette()
        
        # 图形编辑器
        self.graph_scene = GraphScene()
        self.graph_view = GraphView(self.graph_scene)
        
        middle_splitter.addWidget(self.module_palette)
        middle_splitter.addWidget(self.graph_view)
        middle_splitter.setSizes([280, 720])
        
        middle_layout.addWidget(middle_splitter)
        
        # 右侧：代码编辑器
        self.code_editor = CodeEditor()
        
        # 连接图形场景和代码编辑器
        self.graph_scene.set_code_editor(self.code_editor)
        
        # 添加到主分割器
        self.main_splitter.addWidget(self.cmd_log)
        self.main_splitter.addWidget(middle_widget)
        self.main_splitter.addWidget(self.code_editor)
        
        # 设置主分割器比例 (日志:图形编辑器:代码编辑器 = 1:3:1.5)
        self.main_splitter.setSizes([300, 900, 400])
        
        main_layout.addWidget(self.main_splitter)
        
        # 应用样式
        self._apply_stylesheet()
        
        log_debug("UI布局创建完成")
        log_info("图形编辑器已就绪，可以从左侧拖拽模块到画布")
    
    def _apply_stylesheet(self):
        """应用样式表"""
        try:
            bg = get_color('bg', '#1e1e1e')
            card_bg = get_color('card_bg', '#2d2d2d')
            border = get_color('border', '#444444')
            text_primary = get_color('text_primary', '#ffffff')
        except:
            # 降级方案
            bg = '#1e1e1e'
            card_bg = '#2d2d2d'
            border = '#444444'
            text_primary = '#ffffff'
        
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
                padding: 20px;
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
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # 机器人类型选择
        toolbar.addWidget(QLabel(" 机器人: "))
        
        self.robot_combo = QComboBox()
        available_robots = self.config.get_available_robots()
        self.robot_combo.addItems(available_robots)
        default_robot = self.config.get('SIMULATION', 'default_robot', fallback='go2')
        self.robot_combo.setCurrentText(default_robot)
        self.robot_combo.currentTextChanged.connect(self._on_robot_type_changed)
        self.robot_combo.setMinimumWidth(80)
        toolbar.addWidget(self.robot_combo)
        
        toolbar.addSeparator()
        
        # 工具栏按钮
        actions = [
            ("新建", self._on_new),
            ("打开", self._on_open),
            ("保存", self._on_save),
            ("导出源码", self._on_export_code),
            ("运行", self._on_run)
        ]
        
        for text, handler in actions:
            action = QAction(text, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)
        
        toolbar.addSeparator()
        
        # 测试按钮
        test_action = QAction("测试抬右腿", self)
        test_action.triggered.connect(self._test_lift_leg)
        toolbar.addAction(test_action)
    
    def _init_statusbar(self):
        """初始化状态栏"""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        # 显示初始状态
        robot_type = self.robot_combo.currentText()
        self.status.showMessage(f"就绪 | 机器人: {robot_type}")
    
    def set_robot_model(self, robot_model):
        """设置机器人模型"""
        self.robot_model = robot_model
        
        # 同时设置图形场景的机器人类型
        if hasattr(self, 'graph_scene') and robot_model:
            robot_type = getattr(robot_model, 'robot_type', 'go2')
            self.graph_scene.set_robot_type(robot_type)
        
        log_success(f"机器人模型已设置: {robot_model}")
    
    def _on_robot_type_changed(self, robot_type: str):
        """机器人类型改变"""
        log_info(f"切换机器人类型: {robot_type}")
        self.status.showMessage(f"已切换机器人类型: {robot_type}", 2000)
        
        # 更新图形场景的机器人类型
        if hasattr(self, 'graph_scene'):
            self.graph_scene.set_robot_type(robot_type)
        
        # 如果有模型，更新模型类型
        if self.robot_model:
            self.robot_model.robot_type = robot_type
    
    def _on_new(self):
        """新建项目"""
        log_info("新建项目")
        self.code_editor.clear()
        self.status.showMessage("新建项目", 2000)
    
    def _on_open(self):
        """打开项目"""
        log_info("打开项目")
        QMessageBox.information(self, "提示", "打开项目功能尚未实现")
    
    def _on_save(self):
        """保存项目"""
        log_info("保存项目")
        QMessageBox.information(self, "提示", "保存项目功能尚未实现")
    
    def _on_export_code(self):
        """导出源码"""
        log_info("导出源码")
        code = self.code_editor.get_code()
        QMessageBox.information(self, "导出源码", f"代码长度: {len(code)} 字符\n（导出功能尚未实现）")
    
    def _on_run(self):
        """运行"""
        log_info("运行")
        QMessageBox.information(self, "提示", "运行功能需要在图形编辑器中选择动作节点")
    
    def _test_lift_leg(self):
        """测试抬右腿动作"""
        if self.robot_model is None:
            log_warning("未设置机器人模型")
            QMessageBox.warning(self, "警告", "未设置机器人模型")
            return
        
        if self.simulation_thread and self.simulation_thread.isRunning():
            log_warning("仿真正在运行中")
            QMessageBox.warning(self, "警告", "仿真正在运行中")
            return
        
        log_info("开始测试抬右腿动作")
        self.status.showMessage("正在执行抬右腿动作...")
        
        # 创建仿真线程
        self.simulation_thread = SimulationThread(
            self.robot_model,
            "lift_right_leg"
        )
        
        # 连接信号
        self.simulation_thread.simulation_started.connect(
            lambda msg: self.status.showMessage(msg)
        )
        self.simulation_thread.simulation_finished.connect(
            lambda msg: self.status.showMessage(msg, 3000)
        )
        self.simulation_thread.error_occurred.connect(
            lambda msg: QMessageBox.critical(self, "错误", msg)
        )
        
        # 启动线程
        self.simulation_thread.start()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止仿真线程
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait(3000)  # 等待最多3秒
        
        logger.info("主窗口关闭")
        event.accept()
