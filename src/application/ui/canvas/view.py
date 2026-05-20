"""CanvasView — 画布视图（pan/zoom + LOD 跟踪）/ canvas view (pan/zoom + LOD tracking).

Mouse contract:
- Left button   : RubberBand 框选（默认 dragMode）/ rubber-band select.
- Middle button : 按住拖动 → 以按下点为锚的缩放；水平方向 dx>0 放大，dx<0 缩小
                  (drag-to-zoom anchored at press point; +x → zoom in, -x → zoom out).
- Right button  : 按住拖动 → 平移 / hold-drag pan.
- Wheel         : 以光标位置为锚的缩放（保留）/ wheel zoom under cursor.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QMouseEvent,
    QPainter,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

from .items import ConnectionItem
from .lod import TIER_WORKING, tier_for_zoom
from .node_library_panel import MIME_NODE_ID


class CanvasView(QGraphicsView):
    """画布视图 / Canvas view.

    - 鼠标滚轮缩放（在光标点为锚）
    - 中键拖拽平移
    - 关闭抗锯齿调整以减开销（性能 hint）
    - **跟踪可见 scene rect** 与 **zoom tier** —— 用于 LOD / 视口剔除
    """

    _ZOOM_FACTOR = 1.15
    _MIN_ZOOM = 0.1
    _MAX_ZOOM = 4.0
    # 中键拖拽缩放灵敏度：每 1px 水平位移对应的 zoom 因子。
    # 1.005**100 ≈ 1.65；1.005**-100 ≈ 0.61，覆盖一般操作幅度。
    _MIDDLE_ZOOM_PER_PX = 1.005

    # 边缘自动滚动（左键拖动节点群 / 框选 / DnD 都生效；端口拖期间跳过）
    _EDGE_SCROLL_MARGIN = 30  # px，距 viewport 任一边小于此值即触发
    _EDGE_SCROLL_STEP = 16    # px/move event，单步推进量

    # zoom tier 跨边界时发出，参数为新 tier (lod.TIER_*)
    zoomChanged = pyqtSignal(int)

    def __init__(
        self,
        scene: QGraphicsScene,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True
        )
        # SmartViewportUpdate 在节点+边同时移动时会漏判旧 region，留残影。
        # BoundingRectViewportUpdate 把所有脏 region 合并成一个外包矩形重绘——
        # 对 ≤50 节点几乎没多余成本，但能保证 ConnectionItem 拖动后无残影。
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Phase 2: 默认 RubberBandDrag —— 框选 NodeItem / ConnectionItem。
        # 端口拖拽期间会被切到 NoDrag（避免误触发 marquee）；释放后切回。
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # Node Library 拖入接收：仅接 application/x-unitport-node MIME
        self.setAcceptDrops(True)

        # 右键被画布消费用于平移，不要让任何子组件弹出系统/Scene 上下文菜单。
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        # 右键拖拽平移状态
        self._panning = False
        self._pan_start: QPoint = QPoint()
        # 中键拖拽缩放状态
        self._zooming = False
        self._zoom_anchor: QPoint = QPoint()  # 按下点（缩放锚点，viewport 坐标）
        self._zoom_last_x: int = 0            # 上一次 move 的水平像素值
        # 端口拖期间为 True；用于 mouseMove 边缘自动滚动跳过端口拖（端口拖不
        # 需要滚屏，临时连线只在视口内有效）。
        self._port_drag_active = False

        self._zoom_tier: int = TIER_WORKING  # zoom 默认 1.0 落在 T2
        self._visible_scene_rect: QRectF = QRectF()

        # 端口拖拽期间临时切 NoDrag；scene.port_drag 触发回调
        port_drag = getattr(scene, "port_drag", None)
        if port_drag is not None:
            port_drag.set_state_callback(self._on_port_drag_state)

    # ---- LOD / visible rect 公开 API ----

    def visible_scene_rect(self) -> QRectF:
        """当前可见 rect 在场景坐标 / Current visible rect in scene coords."""
        return QRectF(self._visible_scene_rect)

    def current_zoom(self) -> float:
        """当前 zoom 因子 / Current zoom factor."""
        return self.transform().m11()

    def current_zoom_tier(self) -> int:
        """当前 LOD tier / Current LOD tier (TIER_OVERVIEW..TIER_DETAIL)."""
        return self._zoom_tier

    # ---- 缩放 ----

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self._ZOOM_FACTOR if delta > 0 else 1.0 / self._ZOOM_FACTOR
        current = self.transform().m11()
        new = current * factor
        if new < self._MIN_ZOOM or new > self._MAX_ZOOM:
            event.accept()
            return
        self.scale(factor, factor)
        self._update_visible_rect()
        self._update_zoom_tier()
        event.accept()

    # ---- 中键缩放 / 右键平移 ----

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._zooming = True
            self._zoom_anchor = event.position().toPoint()
            self._zoom_last_x = self._zoom_anchor.x()
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            # 命中连线 → 直接断开（不进入平移模式）。批量选中时整组一起断，
            # 单条 undo macro 内合并。命中节点 / 端口 / 空白 → 走平移。
            if self._try_disconnect_edge_at(event.position().toPoint()):
                event.accept()
                return
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def _try_disconnect_edge_at(self, viewport_pt: QPoint) -> bool:
        """Right-button press hit-test. Returns True iff a ConnectionItem was
        hit and the disconnect path was taken — in which case the caller
        must accept the event and skip panning. Returns False when the
        cursor is over a node, port, or empty canvas (caller proceeds with
        the pan path).

        Hit-testing reuses :class:`PortDragController`'s scene-pos hit-test
        for ports first — a right-click that lands on a port should still
        feel like "port territory" and start panning, not silently delete
        an adjacent edge. Then walks the scene item stack to look for a
        ``ConnectionItem`` ancestor.
        """
        scene = self.scene()
        if scene is None:
            return False
        # Avoid the ConnectionItem-vs-PortItem ambiguity at port centres:
        # PortItem's bounding rect is small but the connection's stroker
        # shape extends into the port. Prefer "port wins" — same rule the
        # context menu provider applies (see CanvasPage._build_context_menu).
        from PyQt6.QtGui import QTransform
        scene_pos = self.mapToScene(viewport_pt)
        for it in scene.items(scene_pos, deviceTransform=QTransform()):
            if it.data(0) == "port":
                return False
        edge = None
        for it in scene.items(scene_pos, deviceTransform=QTransform()):
            cur = it
            while cur is not None:
                if isinstance(cur, ConnectionItem):
                    edge = cur
                    break
                cur = cur.parentItem()
            if edge is not None:
                break
        if edge is None:
            return False
        page = getattr(scene, "_page_ref", None)
        if page is None:
            edge.disconnect()
            return True
        page._disconnect_edge_via_right_click(edge)
        return True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._zooming:
            current = event.position().toPoint()
            dx = current.x() - self._zoom_last_x
            self._zoom_last_x = current.x()
            if dx != 0:
                self._zoom_by_factor(self._MIDDLE_ZOOM_PER_PX ** dx, self._zoom_anchor)
            event.accept()
            return
        if self._panning:
            current = event.position().toPoint()
            delta = current - self._pan_start
            self._pan_start = current
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            # scrollbar set 会触发 scrollContentsBy → _update_visible_rect
            event.accept()
            return
        # 左键按下 + 不在端口拖 → 节点拖动 / RubberBand 中，距视口边缘 < 30px
        # 触发自动滚动一格。
        if (
            bool(event.buttons() & Qt.MouseButton.LeftButton)
            and not self._port_drag_active
        ):
            self._auto_scroll_at(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._zooming:
            self._zooming = False
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _zoom_by_factor(self, factor: float, viewport_anchor: QPoint) -> None:
        """以 viewport 坐标点 ``viewport_anchor`` 为锚做缩放.

        wheelEvent 走 AnchorUnderMouse + ``self.scale``；中键拖拽期间光标会移动，
        不能用 AnchorUnderMouse，故这里显式把锚下方的 scene 点保持不动。
        """
        if factor <= 0.0:
            return
        current = self.transform().m11()
        new = current * factor
        if new < self._MIN_ZOOM:
            factor = self._MIN_ZOOM / current
        elif new > self._MAX_ZOOM:
            factor = self._MAX_ZOOM / current
        if factor == 1.0:
            return
        scene_anchor = self.mapToScene(viewport_anchor)
        self.scale(factor, factor)
        new_screen = self.mapFromScene(scene_anchor)
        delta = new_screen - viewport_anchor
        if delta.x() != 0:
            hbar = self.horizontalScrollBar()
            hbar.setValue(hbar.value() + delta.x())
        if delta.y() != 0:
            vbar = self.verticalScrollBar()
            vbar.setValue(vbar.value() + delta.y())
        self._update_visible_rect()
        self._update_zoom_tier()

    # ---- visible rect / zoom tier 跟踪 ----

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_visible_rect()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._update_visible_rect()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._update_visible_rect()
        self._update_zoom_tier(force=True)

    def _update_visible_rect(self) -> None:
        vp_rect = self.viewport().rect()
        self._visible_scene_rect = self.mapToScene(vp_rect).boundingRect()

    def _update_zoom_tier(self, force: bool = False) -> None:
        new_tier = tier_for_zoom(self.transform().m11())
        if force or new_tier != self._zoom_tier:
            self._zoom_tier = new_tier
            self.zoomChanged.emit(new_tier)

    # ---- Node Library drop（外部 widget 拖入节点）----

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(MIME_NODE_ID):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        # NodeLibrary DnD 期间也触发边缘滚动，跨大画布拖图标更顺手
        self._auto_scroll_at(event.position().toPoint())
        if event.mimeData().hasFormat(MIME_NODE_ID):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if not md.hasFormat(MIME_NODE_ID):
            super().dropEvent(event)
            return
        node_id = bytes(md.data(MIME_NODE_ID)).decode("utf-8", errors="ignore").strip()
        if not node_id:
            event.ignore()
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        page = self.parent()  # CanvasPage
        spawn = getattr(page, "spawn_node", None)
        if callable(spawn):
            spawn(node_id, x=scene_pos.x(), y=scene_pos.y())
            event.acceptProposedAction()
        else:
            event.ignore()

    # ---- Phase 2: rubber-band 与端口拖拽协调 ----

    def _on_port_drag_state(self, active: bool) -> None:
        """端口拖拽进入 / 退出时切换 dragMode.

        - active=True   → NoDrag，让 scene 接管鼠标（避免被 RubberBandDrag 抢）
        - active=False  → RubberBandDrag 恢复，框选可用
        """
        self._port_drag_active = active
        if active:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    # ---- Phase 2: 视口边缘自动滚动 ----

    def _auto_scroll_at(self, viewport_pos: QPoint) -> None:
        """光标位置距 viewport 任一边 < ``_EDGE_SCROLL_MARGIN`` 时，对应方向
        滚动条单步推进 ``_EDGE_SCROLL_STEP`` 像素，实现拖动到边缘自动跟随."""
        rect = self.viewport().rect()
        margin = self._EDGE_SCROLL_MARGIN
        step = self._EDGE_SCROLL_STEP
        dx = 0
        dy = 0
        if viewport_pos.x() < margin:
            dx = -step
        elif viewport_pos.x() > rect.width() - margin:
            dx = step
        if viewport_pos.y() < margin:
            dy = -step
        elif viewport_pos.y() > rect.height() - margin:
            dy = step
        if dx:
            sb = self.horizontalScrollBar()
            sb.setValue(sb.value() + dx)
        if dy:
            sb = self.verticalScrollBar()
            sb.setValue(sb.value() + dy)


__all__ = ["CanvasView"]
