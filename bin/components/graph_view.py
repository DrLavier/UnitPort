#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图形编辑视图
支持拖放、缩放、平移等操作
"""

import json
from PySide6.QtCore import Qt, QMimeData, QPointF
from PySide6.QtGui import QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from bin.core.logger import log_info, log_debug, log_warning, log_error
from bin.components.graph_scene import GraphScene


class GraphView(QGraphicsView):
    """图形编辑视图"""
    
    def __init__(self, scene: GraphScene, parent=None):
        super().__init__(scene, parent)
        
        # 视图配置
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # 优化性能
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        
        # 拖拽模式 - 默认使用RubberBandDrag
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        # 接受拖放
        self.setAcceptDrops(True)
        
        # 缩放范围
        self._zoom_factor = 1.0
        self._zoom_min = 0.3
        self._zoom_max = 3.0
        
        # 平移
        self._is_panning = False
        self._pan_start_pos = None
        
        # 连线状态标记 - 用于禁用RubberBand
        self._is_connecting = False
        
        log_debug("GraphView 初始化完成")
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        # 获取滚轮角度
        delta = event.angleDelta().y()
        
        if delta > 0:
            factor = 1.15
        else:
            factor = 0.85
        
        # 计算新的缩放因子
        new_zoom = self._zoom_factor * factor
        
        # 限制缩放范围
        if new_zoom < self._zoom_min or new_zoom > self._zoom_max:
            return
        
        self._zoom_factor = new_zoom
        
        # 以鼠标位置为中心缩放
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        # 检查场景是否正在连线
        scene = self.scene()
        if isinstance(scene, GraphScene):
            # 获取点击位置的item
            scene_pos = self.mapToScene(event.position().toPoint())
            item = scene.itemAt(scene_pos, self.transform())
            
            # 如果点击了端口,禁用RubberBand拖拽
            if scene._is_port(item):
                self._is_connecting = True
                self.setDragMode(QGraphicsView.NoDrag)
                log_debug("开始连线,禁用多选框")
        
        # 中键或右键平移
        if event.button() == Qt.MiddleButton or (event.button() == Qt.RightButton):
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self._is_panning:
            # 平移视图
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        # 如果正在连线,恢复RubberBand模式
        if self._is_connecting:
            self._is_connecting = False
            self.setDragMode(QGraphicsView.RubberBandDrag)
            log_debug("连线结束,恢复多选框")
        
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        
        super().mouseReleaseEvent(event)
    
    def dragEnterEvent(self, event):
        """拖拽进入"""
        # 检查是否有模块卡片数据
        if event.mimeData().hasFormat("application/x-module-card"):
            event.acceptProposedAction()
            log_debug("拖拽进入视图")
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动"""
        if event.mimeData().hasFormat("application/x-module-card"):
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """拖放事件"""
        if not event.mimeData().hasFormat("application/x-module-card"):
            event.ignore()
            return
        
        # 解析拖拽数据
        try:
            data = event.mimeData().data("application/x-module-card")
            payload = json.loads(bytes(data).decode("utf-8"))
            
            title = payload.get("title", "未知模块")
            grad = tuple(payload.get("grad", ["#45a049", "#4CAF50"]))
            features = payload.get("features", [])
            preset = payload.get("preset")
            
            # 获取场景位置
            scene_pos = self.mapToScene(event.position().toPoint())
            
            # 创建节点
            scene = self.scene()
            if isinstance(scene, GraphScene):
                node_item = scene.create_node(title, scene_pos, features, grad)
                if preset and hasattr(node_item, "_combo") and node_item._combo:
                    node_item._combo.setCurrentText(preset)
            log_info(f"创建节点: {title} at ({scene_pos.x():.0f}, {scene_pos.y():.0f})")
            
            event.acceptProposedAction()
            
        except Exception as e:
            log_error(f"拖放失败: {e}")
            event.ignore()
    
    def reset_view(self):
        """重置视图"""
        self.resetTransform()
        self._zoom_factor = 1.0
        log_info("视图已重置")
    
    def fit_to_contents(self):
        """适应内容"""
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        log_info("视图已适应内容")
