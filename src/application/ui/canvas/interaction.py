"""application.ui.canvas.interaction — 画布交互层 / Canvas interaction layer.

托管 drag-to-connect / rubber-band / 键盘 / 右键菜单的状态与协调，避免在
view.py、scene.py、page.py 之间散落事件处理。

设计参考 DEMO ``bin/pages/canvas/graph_scene.py:2123–2891``（不复制代码）：
- ``PortDragController`` 持 ``_start_port`` / ``_snap_port`` / ``_temp_path``
- 半径式端口命中（``_find_port_near`` 12px 容差）
- 校验函数 ``can_connect`` 纯函数（方向 / 类型 / 通道 / max_connections / 自连接）
- temp 连线：合法 snap 时端口色实线，无 snap / 非法 snap 时虚线（无效染红）
- ESC 取消 / drop 在空白处 → 静默取消
- input 端口已连接时拖出 → reconnect 模式：detach 原边、把 src 当作新拖动起点
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsScene

from unitport_sdk import Config, log_debug, log_warning

from .items import ConnectionItem, PortItem, ROLE_META


# =============================================================================
# 校验函数 / Connection validator —— 纯函数，无副作用
# =============================================================================

def can_connect(src: PortItem, dst: PortItem) -> Tuple[bool, str]:
    """判定两端口是否可连 / Check if two ports can connect.

    Returns:
        ``(True, "")`` 合法 / ``(False, "<reason>")`` 拒绝（reason 用于 UI 提示）
    """
    if src is dst:
        return False, "同一端口"
    if src.parent_node() is dst.parent_node():
        return False, "不能连同一节点"
    # 方向：必须 out + in（顺序无所谓，但不能两端同向）
    if src.direction == dst.direction:
        return False, f"端口方向相同 ({src.direction})"

    src_meta = src.data(ROLE_META) or {}
    dst_meta = dst.data(ROLE_META) or {}
    # 通道
    if src_meta.get("channel") != dst_meta.get("channel"):
        return False, "通道不匹配 (data vs flow)"
    # 类型：完全相同 或 一方是 "any"
    if (
        src.port_type != dst.port_type
        and "any" not in (src.port_type, dst.port_type)
    ):
        return False, f"类型不匹配 ({src.port_type} vs {dst.port_type})"
    # max_connections —— 仅检查 input 端的限制
    in_port = dst if dst.direction == "in" else src
    in_meta = in_port.data(ROLE_META) or {}
    max_conn = in_meta.get("max_connections")
    if max_conn is not None and len(in_port.connections) >= int(max_conn):
        return False, f"input 端已达最大连接数 {max_conn}"
    return True, ""


# =============================================================================
# 半径式端口命中
# =============================================================================

def _find_port_near(
    scene: QGraphicsScene,
    scene_pos: QPointF,
    radius: float = 12.0,
) -> Optional[PortItem]:
    """在 scene 中以 ``scene_pos`` 为中心、``radius`` 为半径查最近的 PortItem.

    比 ``scene.itemAt(...)`` 宽容——画布缩放后小圆点像素难点中，半径式让
    用户点击端口"附近" 12px 都算命中。
    """
    box = QRectF(
        scene_pos.x() - radius,
        scene_pos.y() - radius,
        radius * 2.0,
        radius * 2.0,
    )
    closest: Optional[PortItem] = None
    best_d2 = radius * radius
    for it in scene.items(box):
        if not isinstance(it, PortItem):
            continue
        sp = it.scenePos()
        dx = sp.x() - scene_pos.x()
        dy = sp.y() - scene_pos.y()
        d2 = dx * dx + dy * dy
        if d2 <= best_d2:
            best_d2 = d2
            closest = it
    return closest


# =============================================================================
# PortDragController —— 端口拖拽连线状态机
# =============================================================================

class PortDragController:
    """端口拖拽连线状态机 / Port-drag connection state machine.

    生命周期：
        try_press(scene_pos)        → 命中端口 → 开始（返回 True 表示接管事件）
        move(scene_pos)             → 更新 temp 边 + snap 端口高亮
        release(scene_pos)          → 完成连线（合法）/ 静默取消（非法 / 空白）
        cancel()                    → ESC 取消

    并发：单画布同时只能有一个拖拽进行中——多 view 共用一个 scene 时
    PortDragController 应全局唯一。
    """

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._start_port: Optional[PortItem] = None
        self._snap_port: Optional[PortItem] = None
        self._temp_path: Optional[QGraphicsPathItem] = None
        self._reconnect_edge: Optional[ConnectionItem] = None
        self._state_callback: Optional[Callable[[bool], None]] = None

    # ---- 公开 API ----

    @property
    def is_active(self) -> bool:
        return self._start_port is not None

    def set_state_callback(self, cb: Callable[[bool], None]) -> None:
        """注册 active 状态回调；``cb(True)`` 开始拖拽，``cb(False)`` 结束.

        view.py 用它在拖拽期间把 dragMode 切到 NoDrag、结束后切回 RubberBandDrag。
        """
        self._state_callback = cb

    def try_press(self, scene_pos: QPointF) -> bool:
        """尝试在 ``scene_pos`` 处接管按下事件.

        Returns:
            ``True`` 命中端口已接管（调用方应 ``event.accept()``）；
            ``False`` 未命中（让事件继续冒泡到默认 rubber-band 选择）。
        """
        if self.is_active:
            # 已经在拖拽中（异常路径，重复 press）—— 静默忽略，让原事件流处理
            return False
        port = _find_port_near(self._scene, scene_pos, radius=12.0)
        if port is None:
            return False

        # input 端口已有连接 → 进入 reconnect 模式：detach 原边，把对端 (out) 当起点
        if port.is_input() and port.connections:
            edge = port.connections[0]
            other = edge.src_port  # input 的对端必是 out
            self._reconnect_edge = edge
            self._start_port = other
            edge.disconnect()
        else:
            self._start_port = port

        self._begin_temp(scene_pos)
        if self._state_callback:
            self._state_callback(True)
        return True

    def move(self, scene_pos: QPointF) -> bool:
        """``mouseMoveEvent`` 入口；返回 True 表示已处理."""
        if not self.is_active:
            return False
        # 查 snap 候选
        candidate = _find_port_near(self._scene, scene_pos, radius=12.0)
        if candidate is self._start_port:
            candidate = None
        # 切换 snap：先复位旧 snap 视觉
        if self._snap_port is not None and self._snap_port is not candidate:
            self._snap_port.set_snap_status("normal")
            self._snap_port = None
        # 评估新 snap
        if candidate is not None and self._snap_port is not candidate:
            ok, _ = can_connect(self._start_port, candidate)  # type: ignore[arg-type]
            candidate.set_snap_status("snap_valid" if ok else "snap_invalid")
            self._snap_port = candidate
        # 重画 temp path
        self._update_temp(scene_pos)
        return True

    def release(self, scene_pos: QPointF) -> Optional[ConnectionItem]:
        """``mouseReleaseEvent`` 入口；返回新 ConnectionItem 或 None."""
        if not self.is_active:
            return None
        target = self._snap_port
        new_edge: Optional[ConnectionItem] = None
        if target is not None and self._start_port is not None:
            ok, reason = can_connect(self._start_port, target)
            if ok:
                # 方向规整：始终 out → in
                if self._start_port.direction == "out":
                    src, dst = self._start_port, target
                else:
                    src, dst = target, self._start_port
                new_edge = ConnectionItem(src, dst)
                self._scene.addItem(new_edge)
                log_debug(
                    f"[ui.canvas] connect: {src.parent_node().node_id}.{src.port_name} → "  # type: ignore[union-attr]
                    f"{dst.parent_node().node_id}.{dst.port_name}"  # type: ignore[union-attr]
                )
            else:
                log_warning(f"[ui.canvas] connect denied: {reason}")
        self._end()
        return new_edge

    def cancel(self) -> None:
        """ESC / 外部主动取消 / Cancel from ESC or external."""
        if not self.is_active:
            return
        log_debug("[ui.canvas] connect cancelled")
        self._end()

    # ---- 内部 ----

    def _begin_temp(self, scene_pos: QPointF) -> None:
        assert self._start_port is not None
        path = QPainterPath()
        path.moveTo(self._start_port.scenePos())
        path.lineTo(scene_pos)
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(self._start_port.color), 1.8)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        item.setPen(pen)
        item.setZValue(10.0)
        self._scene.addItem(item)
        self._temp_path = item

    def _update_temp(self, scene_pos: QPointF) -> None:
        if self._temp_path is None or self._start_port is None:
            return
        s = self._start_port.scenePos()
        e = scene_pos
        dx = e.x() - s.x()
        path = QPainterPath()
        path.moveTo(s)
        path.cubicTo(
            QPointF(s.x() + dx * 0.5, s.y()),
            QPointF(e.x() - dx * 0.5, e.y()),
            e,
        )
        self._temp_path.setPath(path)
        # 颜色 / 样式根据 snap 校验切换
        pen = self._temp_path.pen()
        if self._snap_port is not None:
            ok, _ = can_connect(self._start_port, self._snap_port)
            if ok:
                pen.setColor(QColor(self._start_port.color))
                pen.setStyle(Qt.PenStyle.SolidLine)
            else:
                pen.setColor(QColor(Config.get_color("canvas_safety_error", "#EF4444")))
                pen.setStyle(Qt.PenStyle.DashLine)
        else:
            pen.setColor(QColor(self._start_port.color))
            pen.setStyle(Qt.PenStyle.DashLine)
        self._temp_path.setPen(pen)

    def _end(self) -> None:
        if self._snap_port is not None:
            self._snap_port.set_snap_status("normal")
            self._snap_port = None
        if self._temp_path is not None:
            sc = self._temp_path.scene()
            if sc is not None:
                sc.removeItem(self._temp_path)
            self._temp_path = None
        self._start_port = None
        self._reconnect_edge = None
        if self._state_callback:
            self._state_callback(False)


__all__ = [
    "PortDragController",
    "can_connect",
]
