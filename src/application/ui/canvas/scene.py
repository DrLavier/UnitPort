"""CanvasScene — Training Ground 画布场景 / canvas scene.

Phase 0: 仅栅格背景 + 空场景。
Phase 1a: 接 NodeItem / PortItem / ConnectionItem。
Phase 1b: + 背景 LOD（T0 切到 big-only 大格）。
Phase 2  : + 端口 drag-connect / 右键 context menu / 鼠标事件转发到 interaction.py。
"""

from __future__ import annotations

from typing import Optional

from typing import Any, Dict

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneContextMenuEvent,
    QGraphicsSceneMouseEvent,
    QMenu,
    QWidget,
)

from unitport_sdk import Config

from .grid import CachedGrid
from .items import ConnectionItem, NodeItem
from .lod import TIER_OVERVIEW


class CanvasScene(QGraphicsScene):
    """画布场景 / Canvas scene.

    - 用 ``BspTreeIndex`` 替代 DEMO 的 ``NoIndex``（性能 playbook §1）
    - ``drawBackground`` 走缓存栅格 brush，不每帧重画线段
    - ``apply_theme`` 在主题切换时由外部调用，重新读 Config 颜色
    - ``on_zoom_tier_changed`` 由 ``CanvasView.zoomChanged`` 触发，T0 时切到 big-only 栅格
    """

    _SMALL_STEP = 20
    _BIG_STEP = 100
    _SCENE_HALF = 10000  # ±10000 共 20000×20000，足够 ≤50 节点画布平移

    # ---- DEMO-aligned scene-level signals ----
    # Emitted by ParamRow.set_value end-of-pipeline so widgets on *other*
    # nodes can react (e.g. Export node's output_file_list refreshes when
    # a sibling's bundle_name commits). DEMO 对应：
    # ``TrainingGraphScene.node_param_changed`` (training_workspace_window.py:3013).
    node_param_changed = pyqtSignal(str, str, str)   # node_type, key, value (str)

    # Emitted by ReviewLaunchButtonRow when the user clicks "▶ Launch Review"
    # AND SafetyReview passes. The MainWindow / workspace listens for the
    # payload and spawns the review subprocess. DEMO 对应：
    # ``TrainingGraphScene.review_launch_requested`` (line 3012).
    # Payload: {backend, scene_id, bundle_name, version, overwrite}.
    review_launch_requested = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.setSceneRect(
            -self._SCENE_HALF,
            -self._SCENE_HALF,
            self._SCENE_HALF * 2,
            self._SCENE_HALF * 2,
        )
        self._grid = CachedGrid(small_step=self._SMALL_STEP, big_step=self._BIG_STEP)
        # 端口拖拽控制器（Phase 2）；CanvasPage 构造完会注入
        from .interaction import PortDragController
        self._port_drag = PortDragController(self)
        # 右键菜单回调（CanvasPage 在构造后注入）
        self._context_menu_provider: Optional[
            "Callable[[QGraphicsSceneContextMenuEvent], Optional[QMenu]]"
        ] = None
        self.apply_theme()

    # ---- Phase 2 公开访问 ----

    @property
    def port_drag(self):
        """暴露 PortDragController 给 view（用于 dragMode 协调）."""
        return self._port_drag

    def set_context_menu_provider(self, provider) -> None:
        """注入右键菜单回调；signature: (event) -> QMenu | None."""
        self._context_menu_provider = provider

    # ---- DEMO-aligned graph serialization ----

    def serialize_training_graph(self, *, for_compiler: bool = True) -> Optional[Dict[str, Any]]:
        """Snapshot the canvas to an IR-shape graph dict.

        Delegates to ``CanvasPage.to_workflow_dict()`` via the ``_page_ref``
        backref CanvasPage installs at construction time. Returns ``None``
        if no page is bound (e.g. test-only headless scene).

        Used by ``TrainingContext.compat_report`` and ``safety_review`` —
        the same graph shape RELEASE persists to ``.canvas.json``.

        ``for_compiler`` is accepted for DEMO API parity but currently
        ignored — RELEASE's ``to_workflow_dict`` already returns the IR
        shape spec_compiler consumes.
        """
        page = getattr(self, "_page_ref", None)
        if page is None:
            return None
        try:
            return page.to_workflow_dict()
        except Exception:
            return None

    def apply_theme(self) -> None:
        """从 Config 读 canvas_* 主题槽并刷新栅格 brush / Re-read theme colors."""
        bg = QColor(Config.get_color("canvas_bg", "#1A1A1A"))
        small = QColor(Config.get_color("canvas_grid_small", "#2A2A2A"))
        big = QColor(Config.get_color("canvas_grid_big", "#3d3d3d"))
        self._grid.set_colors(bg, small, big)
        self.setBackgroundBrush(self._grid.brush())
        self.update()

    def on_zoom_tier_changed(self, tier: int) -> None:
        """``CanvasView.zoomChanged`` 信号槽 / Slot for view zoom-tier change.

        T0 时把小格关掉只剩大格；T1+ 切回完整双层。
        变化时重生成缓存 brush 并 invalidate 背景层。
        """
        big_only = tier == TIER_OVERVIEW
        self._grid.set_visible_tiers(big_only)
        self.setBackgroundBrush(self._grid.brush())
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """缓存 brush 平铺；DEMO 的逐线绘制改为单次 fillRect."""
        painter.fillRect(rect, self._grid.brush())

    # ---- 鼠标事件：先给 PortDragController 接管 ----

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # PyQt6 注意：Qt.MouseButton.LeftButton 的 .value 是 1，但 Python 层
        # `event.button() == 1` 比较返回 False（PyQt6 enum 协议不走 IntEnum 的
        # __eq__）。必须直接用枚举做比较。
        if event.button() == Qt.MouseButton.LeftButton:
            # Ctrl+LeftClick 在"空白"或 GroupItem 上 → 快速放 Note 节点。
            # 命中 NodeItem/NoteItem/ConnectionItem 时不拦截,保留 Ctrl+Click 多选语义。
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                page = getattr(self, "_page_ref", None)
                if page is not None and self._is_empty_spawn_target(event.scenePos()):
                    try:
                        page.spawn_node("note", event.scenePos().x(), event.scenePos().y())
                        event.accept()
                        return
                    except Exception:
                        pass  # spawn 失败兜底 → 走默认 super
            if self._port_drag.try_press(event.scenePos()):
                event.accept()
                return
        super().mousePressEvent(event)

    def _is_empty_spawn_target(self, scene_pos) -> bool:
        """返回 True 当命中点没有 NodeItem/NoteItem/ConnectionItem。

        GroupItem 视为"空白"目标 —— 用户在 group 框内空白点 Ctrl+LeftClick 即
        spawn Note 到该位置(随后由几何成员关系自动加入该 group)。
        """
        from PyQt6.QtGui import QTransform
        from .note_item import NoteItem
        for it in self.items(scene_pos, deviceTransform=QTransform()):
            if isinstance(it, (NodeItem, NoteItem, ConnectionItem)):
                return False
        return True

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._port_drag.is_active:
            self._port_drag.move(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._port_drag.is_active:
            new_edge = self._port_drag.release(event.scenePos())
            event.accept()
            if new_edge is not None:
                page = getattr(self, "_page_ref", None)
                if page is not None and not getattr(page, "_in_undo", False):
                    sn = new_edge.src_port.parent_node()
                    dn = new_edge.dst_port.parent_node()
                    if sn is not None and dn is not None:
                        try:
                            from .undo import ConnectCmd
                            page._undo_stack.push(ConnectCmd(
                                page, new_edge.edge_id,
                                sn.node_id, new_edge.src_port.port_name,
                                dn.node_id, new_edge.dst_port.port_name,
                                already_created=True,
                            ))
                        except Exception:
                            # undo bookkeeping 失败不该阻塞连线本身
                            pass
            return
        super().mouseReleaseEvent(event)

    # ---- 右键菜单 ----

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent) -> None:
        if self._context_menu_provider is None:
            super().contextMenuEvent(event)
            return
        menu = self._context_menu_provider(event)
        if menu is None:
            super().contextMenuEvent(event)
            return
        menu.exec(event.screenPos())
        event.accept()


__all__ = ["CanvasScene"]
