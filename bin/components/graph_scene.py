#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图形编辑场景
包含节点、连线、网格等元素
"""

import json
from typing import Optional, List, Dict, Any
from shiboken6 import isValid

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QLinearGradient, QGradient, QPainterPath
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPathItem, QGraphicsProxyWidget, QComboBox, QLineEdit, QWidget,
    QHBoxLayout, QPushButton, QMessageBox
)

from bin.core.logger import log_info, log_error, log_debug, log_warning, log_success


class ConnectionItem(QGraphicsPathItem):
    """连接线项 - 支持自动更新和端点编辑"""
    
    def __init__(self, out_port, in_port, parent=None):
        super().__init__(parent)
        
        self.out_port = out_port
        self.in_port = in_port
        
        # 设置样式
        self.setPen(QPen(QColor("#60a5fa"), 2.5))
        self.setZValue(-1)
        self.setData(0, "connection")
        
        # 可选中和可点击
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
        # 端点标记
        self._start_marker = None
        self._end_marker = None
        self._create_markers()
        
        # 更新路径
        self.update_path()
    
    def _create_markers(self):
        """创建端点标记(用于重新连接)"""
        # 起点标记
        self._start_marker = QGraphicsEllipseItem(-4, -4, 8, 8, self)
        self._start_marker.setBrush(QBrush(QColor("#60a5fa")))
        self._start_marker.setPen(QPen(QColor("#ffffff"), 1))
        self._start_marker.setZValue(10)
        self._start_marker.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._start_marker.setData(0, "connection_marker")
        self._start_marker.setData(1, "start")
        self._start_marker.setData(2, self)  # 关联的连接线
        self._start_marker.setAcceptedMouseButtons(Qt.LeftButton)
        self._start_marker.setVisible(False)
        
        # 终点标记
        self._end_marker = QGraphicsEllipseItem(-4, -4, 8, 8, self)
        self._end_marker.setBrush(QBrush(QColor("#60a5fa")))
        self._end_marker.setPen(QPen(QColor("#ffffff"), 1))
        self._end_marker.setZValue(10)
        self._end_marker.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._end_marker.setData(0, "connection_marker")
        self._end_marker.setData(1, "end")
        self._end_marker.setData(2, self)  # 关联的连接线
        self._end_marker.setAcceptedMouseButtons(Qt.LeftButton)
        self._end_marker.setVisible(False)
    
    def update_path(self):
        """更新连接路径"""
        if not self.out_port or not self.in_port:
            return
        
        # 检查端口是否有效
        if not isValid(self.out_port) or not isValid(self.in_port):
            return
        
        # 获取端口中心位置
        try:
            start = self.out_port.mapToScene(self.out_port.boundingRect().center())
            end = self.in_port.mapToScene(self.in_port.boundingRect().center())
        except RuntimeError:
            return
        
        # 创建贝塞尔曲线路径
        path = QPainterPath()
        path.moveTo(start)
        
        dx = end.x() - start.x()
        path.cubicTo(
            start.x() + dx * 0.5, start.y(),
            end.x() - dx * 0.5, end.y(),
            end.x(), end.y()
        )
        
        self.setPath(path)
        
        # 更新端点标记位置
        if self._start_marker:
            self._start_marker.setPos(start)
        if self._end_marker:
            self._end_marker.setPos(end)
    
    def hoverEnterEvent(self, event):
        """鼠标悬停 - 显示端点标记"""
        self.setPen(QPen(QColor("#3b82f6"), 3.5))
        if self._start_marker:
            self._start_marker.setVisible(True)
        if self._end_marker:
            self._end_marker.setVisible(True)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """鼠标离开 - 隐藏端点标记"""
        if not self.isSelected():
            self.setPen(QPen(QColor("#60a5fa"), 2.5))
        if self._start_marker:
            self._start_marker.setVisible(False)
        if self._end_marker:
            self._end_marker.setVisible(False)
        super().hoverLeaveEvent(event)
    
    def itemChange(self, change, value):
        """项变化事件"""
        if change == QGraphicsItem.ItemSelectedHasChanged:
            if value:  # 选中
                self.setPen(QPen(QColor("#3b82f6"), 3.5))
                if self._start_marker:
                    self._start_marker.setVisible(True)
                if self._end_marker:
                    self._end_marker.setVisible(True)
            else:  # 未选中
                self.setPen(QPen(QColor("#60a5fa"), 2.5))
                if self._start_marker:
                    self._start_marker.setVisible(False)
                if self._end_marker:
                    self._end_marker.setVisible(False)
        
        return super().itemChange(change, value)


class GraphScene(QGraphicsScene):
    """图形编辑场景"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 场景配置
        self.setSceneRect(-2500, -2500, 5000, 5000)
        self.grid_small = 20
        self.grid_big = self.grid_small * 5
        
        # 颜色配置
        self.color_bg = QColor(30, 31, 34)
        self.color_grid_small = QColor(50, 51, 55)
        self.color_grid_big = QColor(60, 62, 67)
        
        # 节点和连接
        self._node_seq = 0
        self._temp_connection = None
        self._temp_start_port = None
        
        # 重连接状态
        self._reconnecting = False
        self._reconnect_connection = None
        self._reconnect_end = None  # "start" or "end"
        
        # 动作映射
        self._action_mapping = {
            "抬右腿": "lift_right_leg",
            "站立": "stand",
            "坐下": "sit",
            "行走": "walk",
            "停止": "stop"
        }
        
        # 引用
        self._code_editor = None
        self._simulation_thread = None
        self._robot_type = "go2"
        
        # 定时器 - 用于更新连接线
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_all_connections)
        self._update_timer.start(16)  # 60fps
        
        log_debug("GraphScene 初始化完成")
    
    def set_code_editor(self, editor):
        """设置代码编辑器引用"""
        self._code_editor = editor
    
    def set_robot_type(self, robot_type: str):
        """设置机器人类型"""
        self._robot_type = robot_type
        log_info(f"机器人类型设置为: {robot_type}")
    
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """绘制网格背景"""
        painter.fillRect(rect, self.color_bg)
        
        # 绘制小网格
        left = int(rect.left()) - (int(rect.left()) % self.grid_small)
        top = int(rect.top()) - (int(rect.top()) % self.grid_small)
        
        lines = []
        
        # 垂直线
        x = left
        while x < rect.right():
            lines.append((x, rect.top(), x, rect.bottom()))
            x += self.grid_small
        
        # 水平线
        y = top
        while y < rect.bottom():
            lines.append((rect.left(), y, rect.right(), y))
            y += self.grid_small
        
        # 绘制小网格
        painter.setPen(QPen(self.color_grid_small, 1))
        for x1, y1, x2, y2 in lines:
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # 绘制大网格
        left = int(rect.left()) - (int(rect.left()) % self.grid_big)
        top = int(rect.top()) - (int(rect.top()) % self.grid_big)
        
        big_lines = []
        x = left
        while x < rect.right():
            big_lines.append((x, rect.top(), x, rect.bottom()))
            x += self.grid_big
        
        y = top
        while y < rect.bottom():
            big_lines.append((rect.left(), y, rect.right(), y))
            y += self.grid_big
        
        painter.setPen(QPen(self.color_grid_big, 2))
        for x1, y1, x2, y2 in big_lines:
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        pos = event.scenePos()
        item = self.itemAt(pos, self.views()[0].transform() if self.views() else None)
        
        # 检查是否点击了连接线标记(端点)
        if item and item.data(0) == "connection_marker":
            connection = item.data(2)
            end_type = item.data(1)
            self._start_reconnection(connection, end_type, pos)
            return
        
        # 检查是否点击了端口
        if self._is_port(item):
            self._start_connection(item, pos)
            return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self._temp_connection:
            # 更新临时连线
            self._update_temp_connection(event.scenePos())
            return
        
        if self._reconnecting and self._temp_connection:
            # 更新重连接的临时线
            self._update_temp_connection(event.scenePos())
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self._reconnecting:
            pos = event.scenePos()
            target_port = self._find_port_near(pos)
            
            if target_port:
                self._finish_reconnection(target_port)
            else:
                self._cancel_reconnection()
            return
        
        if self._temp_connection:
            pos = event.scenePos()
            target_port = self._find_port_near(pos)
            
            if target_port and target_port != self._temp_start_port:
                self._finish_connection(target_port)
            else:
                self._cancel_connection()
            
            return
        
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """键盘按下事件"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            # 删除选中的项
            selected_items = self.selectedItems()
            if selected_items:
                self._delete_items(selected_items)
                event.accept()
                return
        
        super().keyPressEvent(event)
    
    def _delete_items(self, items):
        """删除选中的项"""
        deleted_nodes = []
        deleted_connections = []
        
        for item in items:
            # 删除节点
            if item.data(10) == "node":
                node_name = item.data(11)
                node_id = item.data(12)
                
                # 删除与节点相关的所有连接
                self._delete_node_connections(item)
                
                # 删除节点本身
                self.removeItem(item)
                deleted_nodes.append(f"{node_name} (ID: {node_id})")
                log_info(f"删除节点: {node_name} (ID: {node_id})")
            
            # 删除连接
            elif item.data(0) == "connection" or isinstance(item, ConnectionItem):
                # 从端口移除连接引用
                if isinstance(item, ConnectionItem):
                    self._detach_connection(item)
                self.removeItem(item)
                deleted_connections.append("连接线")
        
        if deleted_nodes:
            log_success(f"已删除 {len(deleted_nodes)} 个节点")
        if deleted_connections:
            log_info(f"已删除 {len(deleted_connections)} 条连接")
        
        # 重新生成代码
        self.regenerate_code()
    
    def _delete_node_connections(self, node_item):
        """删除与节点相关的所有连接"""
        # 查找所有端口
        ports = []
        for child in node_item.childItems():
            if self._is_port(child):
                ports.append(child)
        
        # 删除每个端口的连接
        for port in ports:
            connections = port.data(2) or []
            for conn in list(connections):  # 使用list()创建副本避免迭代时修改
                if conn and isValid(conn) and conn.scene() is not None:
                    self.removeItem(conn)
    
    def _detach_connection(self, connection):
        """从端口移除连接引用"""
        if isinstance(connection, ConnectionItem):
            # 从输出端口移除
            if connection.out_port and isValid(connection.out_port):
                conns = connection.out_port.data(2) or []
                if connection in conns:
                    conns.remove(connection)
                    connection.out_port.setData(2, conns)
            
            # 从输入端口移除
            if connection.in_port and isValid(connection.in_port):
                conns = connection.in_port.data(2) or []
                if connection in conns:
                    conns.remove(connection)
                    connection.in_port.setData(2, conns)
    
    def _start_reconnection(self, connection, end_type, pos):
        """开始重新连接"""
        if not isinstance(connection, ConnectionItem):
            return
        
        self._reconnecting = True
        self._reconnect_connection = connection
        self._reconnect_end = end_type
        
        # 确定起始端口
        if end_type == "start":
            self._temp_start_port = connection.out_port
            # 临时移除连接的输出端
            connection.out_port = None
        else:
            self._temp_start_port = connection.in_port
            # 临时移除连接的输入端
            connection.in_port = None
        
        # 创建临时连线
        path = QPainterPath()
        center = self._port_center(self._temp_start_port)
        path.moveTo(center)
        path.lineTo(pos)
        
        self._temp_connection = QGraphicsPathItem(path)
        self._temp_connection.setPen(QPen(QColor("#f59e0b"), 3))
        self.addItem(self._temp_connection)
        
        log_debug(f"开始重新连接: {end_type} 端")
    
    def _finish_reconnection(self, target_port):
        """完成重新连接"""
        if not self._reconnect_connection or not target_port:
            self._cancel_reconnection()
            return
        
        # 检查连接方向
        start_io = self._temp_start_port.data(1)
        target_io = target_port.data(1)
        
        if start_io == target_io:
            log_warning("不能连接相同类型的端口")
            self._cancel_reconnection()
            return
        
        # 更新连接端口
        if self._reconnect_end == "start":
            # 重连起点
            if start_io == "out":
                self._reconnect_connection.out_port = target_port
            else:
                self._reconnect_connection.in_port = target_port
        else:
            # 重连终点
            if start_io == "out":
                self._reconnect_connection.in_port = target_port
            else:
                self._reconnect_connection.out_port = target_port
        
        # 附加到新端口
        self._attach_connection_safe(target_port, self._reconnect_connection)
        
        # 更新路径
        self._reconnect_connection.update_path()
        
        # 清理临时状态
        self._cancel_reconnection()
        
        log_info("重新连接成功")
        self.regenerate_code()
    
    def _cancel_reconnection(self):
        """取消重新连接"""
        if self._reconnect_connection:
            # 恢复原连接
            if self._reconnect_end == "start":
                # 恢复被移除的端口
                pass  # 连接已被删除或保持原状
            
            self._reconnect_connection = None
        
        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None
        
        self._temp_start_port = None
        self._reconnecting = False
    
    def _start_connection(self, port_item, pos):
        """开始创建连接"""
        self._temp_start_port = port_item
        
        # 创建临时连线
        path = QPainterPath()
        center = self._port_center(port_item)
        path.moveTo(center)
        path.lineTo(pos)
        
        self._temp_connection = QGraphicsPathItem(path)
        self._temp_connection.setPen(QPen(QColor("#60a5fa"), 3))
        self.addItem(self._temp_connection)
    
    def _update_temp_connection(self, pos):
        """更新临时连线"""
        if not self._temp_connection or not self._temp_start_port:
            return
        
        start = self._port_center(self._temp_start_port)
        path = QPainterPath()
        path.moveTo(start)
        
        # 贝塞尔曲线
        dx = pos.x() - start.x()
        path.cubicTo(
            start.x() + dx * 0.5, start.y(),
            pos.x() - dx * 0.5, pos.y(),
            pos.x(), pos.y()
        )
        
        self._temp_connection.setPath(path)
    
    def _finish_connection(self, target_port):
        """完成连接"""
        if not self._temp_start_port or not target_port:
            self._cancel_connection()
            return
        
        # 检查连接方向
        start_io = self._temp_start_port.data(1)
        target_io = target_port.data(1)
        
        if start_io == target_io:
            log_warning("不能连接相同类型的端口")
            self._cancel_connection()
            return
        
        # 确定输出端口和输入端口
        out_port = self._temp_start_port if start_io == "out" else target_port
        in_port = target_port if start_io == "out" else self._temp_start_port
        
        # 创建连接线
        self._create_connection(out_port, in_port)
        self._cancel_connection()
        
        # 更新代码
        self.regenerate_code()
    
    def _cancel_connection(self):
        """取消连接"""
        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None
        self._temp_start_port = None
    
    def _create_connection(self, out_port, in_port):
        """创建连接线 - 使用ConnectionItem"""
        conn = ConnectionItem(out_port, in_port)
        self.addItem(conn)
        
        # 附加到端口
        self._attach_connection_safe(out_port, conn)
        self._attach_connection_safe(in_port, conn)
        
        log_debug(f"创建连接: {out_port.data(3)} -> {in_port.data(3)}")
    
    def _update_all_connections(self):
        """更新所有连接线的路径"""
        for item in self.items():
            if isinstance(item, ConnectionItem):
                item.update_path()
    
    def create_node(self, name: str, scene_pos: QPointF, 
                   features: List[str] = None, grad: tuple = None):
        """
        创建节点
        
        Args:
            name: 节点名称
            scene_pos: 场景位置
            features: 功能列表
            grad: 渐变色 (color1, color2)
        """
        # 根据节点类型调整宽度
        if "逻辑控制" in name or "条件判断" in name:
            w, h = 200, 120
        else:
            w, h = 180, 110
        
        # 创建节点矩形
        rect = QGraphicsRectItem(0, 0, w, h)
        
        # 渐变背景
        if grad and len(grad) == 2:
            g = QLinearGradient(0, 0, 1, 1)
            g.setCoordinateMode(QGradient.ObjectBoundingMode)
            g.setColorAt(0.0, QColor(grad[0]))
            g.setColorAt(1.0, QColor(grad[1]))
            rect.setBrush(QBrush(g))
        else:
            rect.setBrush(QBrush(QColor(45, 50, 60)))
        
        rect.setPen(QPen(QColor(120, 130, 140), 2))
        rect.setFlag(QGraphicsItem.ItemIsMovable, True)
        rect.setFlag(QGraphicsItem.ItemIsSelectable, True)
        rect.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)  # 重要:发送几何变化信号
        rect.setPos(scene_pos - QPointF(w/2, h/2))
        self.addItem(rect)
        
        # 标题 - 使用固定宽度确保显示
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        label = self.addText(str(name), f)
        label.setDefaultTextColor(QColor("#ffffff"))
        label.setParentItem(rect)
        label.setZValue(2)
        label.setPos(8, 6)
        
        # 如果标题过长,裁剪显示
        label_width = label.boundingRect().width()
        if label_width > w - 16:
            # 调整字体大小
            f.setPointSize(8)
            label.setFont(f)
        
        # 创建端口函数
        port_r = 6
        def _mk_port(x, y, io, slot):
            p = QGraphicsEllipseItem(x - port_r, y - port_r, port_r*2, port_r*2, rect)
            p.setBrush(QBrush(QColor("#1f2937")))
            p.setPen(QPen(QColor("#60a5fa"), 2))
            p.setData(0, "port")
            p.setData(1, io)
            p.setData(2, [])
            p.setData(3, slot)
            p.setZValue(3)
            p.setAcceptedMouseButtons(Qt.LeftButton)
            p.setAcceptHoverEvents(True)
            return p
        
        # 根据节点类型创建不同的UI和端口
        combo = None
        
        if "逻辑控制" in name:
            features = features or ["如果", "当循环", "直到循环"]
            combo = QComboBox()
            combo.addItems(features)
            combo.setMinimumWidth(int(w * 0.8))
            combo.setMaximumWidth(int(w - 16))
            combo.setStyleSheet("""
                QComboBox { 
                    background: #0f1115; 
                    color: #e5e7eb; 
                    border: 1px solid #4b5563; 
                    border-radius: 4px; 
                    padding: 2px 4px;
                    font-size: 11px;
                }
                QComboBox QAbstractItemView { 
                    background: #111827; 
                    color: #e5e7eb; 
                    selection-background-color: #334155; 
                }
            """)
            
            # 端口
            _mk_port(0, h/2, "in", "condition")
            _mk_port(w, h*0.35, "out", "out_true")
            _mk_port(w, h*0.65, "out", "out_false")
            
            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(combo)
            proxy.setPos(8, 38)
            proxy.setZValue(2)
            
        elif "条件判断" in name:
            features = features or ["等于", "不等于", "大于", "小于"]
            combo = QComboBox()
            combo.addItems(features)
            combo.setMinimumWidth(60)
            combo.setMaximumWidth(70)
            combo.setStyleSheet("""
                QComboBox { 
                    background: #0f1115; 
                    color: #e5e7eb; 
                    border: 1px solid #4b5563; 
                    border-radius: 4px; 
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            
            value_input = QLineEdit()
            value_input.setPlaceholderText("值")
            value_input.setStyleSheet("""
                QLineEdit {
                    background: #0f1115; 
                    color: #e5e7eb; 
                    border: 1px solid #4b5563; 
                    border-radius: 4px; 
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            value_input.setMaximumWidth(60)
            
            widget_container = QWidget()
            hbox = QHBoxLayout(widget_container)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(4)
            hbox.addWidget(combo)
            hbox.addWidget(value_input)
            
            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(widget_container)
            proxy.setPos(8, 38)
            proxy.setZValue(2)
            
            _mk_port(0, h/2, "in", "value_in")
            _mk_port(w, h/2, "out", "result")
            
            rect._value_input = value_input
            rect._combo = combo
            
        else:
            # 其他节点类型
            if "动作执行" in name:
                features = features or ["抬右腿", "站立", "坐下", "行走", "停止"]
            elif "传感输入" in name:
                features = features or ["读超声波", "读红外", "读摄像头", "读IMU", "读里程计"]
            elif "功能运算" in name:
                features = features or ["加", "减", "乘", "除"]
            
            combo = QComboBox()
            combo.addItems(features)
            combo.setMinimumWidth(int(w * 0.85))
            combo.setMaximumWidth(int(w - 16))
            combo.setStyleSheet("""
                QComboBox { 
                    background: #0f1115; 
                    color: #e5e7eb; 
                    border: 1px solid #4b5563; 
                    border-radius: 4px; 
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            
            _mk_port(0, h/2, "in", "in")
            _mk_port(w, h/2, "out", "out")
            
            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(combo)
            proxy.setPos(8, 38)
            proxy.setZValue(2)
        
        # 节点元数据
        node_id = self._node_seq
        self._node_seq += 1
        rect.setData(10, "node")
        rect.setData(11, name)
        rect.setData(12, node_id)
        
        # 保存combo引用
        if combo:
            rect._combo = combo
            combo.currentTextChanged.connect(lambda _t: self.regenerate_code())
        
        self.regenerate_code()
        log_info(f"创建节点: {name} (ID: {node_id})")
        
        return rect
    
    def _find_port_near(self, pos, radius=14):
        """查找附近的端口"""
        search_rect = QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        candidates = []
        
        for it in self.items(search_rect):
            if self._is_port(it):
                c = self._port_center(it)
                dist2 = (c.x() - pos.x())**2 + (c.y() - pos.y())**2
                candidates.append((dist2, it))
        
        return min(candidates, key=lambda t: t[0])[1] if candidates else None
    
    def _is_port(self, item):
        """检查是否是端口"""
        return bool(item) and item.data(0) == "port"
    
    def _port_center(self, port_item):
        """获取端口中心位置"""
        return port_item.mapToScene(port_item.boundingRect().center())
    
    def _attach_connection_safe(self, port_item, conn_item):
        """安全地附加连接到端口"""
        try:
            conns = port_item.data(2) or []
            cleaned = []
            for c in conns:
                if c and isValid(c) and (c.scene() is not None):
                    cleaned.append(c)
            cleaned.append(conn_item)
            port_item.setData(2, cleaned)
        except Exception:
            pass
    
    def regenerate_code(self):
        """重新生成代码"""
        if not self._code_editor:
            return
        
        code_lines = [
            "# 自动生成的代码",
            "# Generated by Celebrimbor",
            "",
            "def execute_workflow():",
        ]
        
        # 获取所有节点
        nodes = []
        for item in self.items():
            if item.data(10) == "node":
                node_info = {
                    'id': item.data(12),
                    'name': item.data(11),
                    'combo': getattr(item, '_combo', None)
                }
                nodes.append(node_info)
        
        # 生成代码
        for node in nodes:
            code_lines.append(f"    # {node['name']} (ID: {node['id']})")
            if node['combo']:
                action = node['combo'].currentText()
                code_lines.append(f"    # 动作: {action}")
            code_lines.append("")
        
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    execute_workflow()")
        
        # 使用set_code方法
        self._code_editor.set_code("\n".join(code_lines))