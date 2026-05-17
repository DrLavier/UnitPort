"""application.ui.canvas.group_item — 视觉分组框 GraphicsItem.

GroupItem 把若干 NodeItem / NoteItem 视觉上包在一个带 title 的圆角矩形里。约束：

- Z = -2（低于 edge=-1、node=0、temp=10）:视觉背景,不挡边线 / 节点。
- ``shape()`` 仅覆盖 title bar + 边框宽条 + 8 个 resize handle 区域
  → 框内空白鼠标穿透,成员节点照常点击 / 连线 / 拖拽。
- ``itemChange(ItemPositionHasChanged)`` 时把同一位移 delta 应用到每个 member
  item —— 拖 group 等价于同步拖动所有成员。
- 鼠标 8 向 resize(N/S/E/W + 4 corners):拖动边/角 → ``_rect`` 直接调整;
  W/N 方向同步移动 ``self.pos()`` 来维持 ``_rect.left()/top() == 0`` 不变量。
- 双击 title 进入 inline 改名(QLineEdit via QGraphicsProxyWidget);
  focusOut/Enter 提交,Esc 取消。
- **几何成员**:resize 结束后由 CanvasPage 重新扫描全部节点,中心点落在 group 框
  scene-rect 内即视为 member。节点单独拖出框外则脱离;拖入则加入。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsProxyWidget,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QLineEdit,
    QStyleOptionGraphicsItem,
    QWidget,
)

from unitport_sdk import Config, tr


TITLE_H = 24
EDGE_GRAB = 8          # 边框可抓宽度(左/右/下侧)
CORNER_R = 8
PADDING = 16           # 成员 bounding 外扩多少作为框 rect
HANDLE = 12            # resize handle 边长(正方形)
MIN_W = 100
MIN_H = 60


def _parse_color_with_alpha(value: str, default: str) -> QColor:
    """支持 ``#RRGGBB`` / ``#RRGGBBAA`` 两种格式,未识别走 default."""
    raw = str(value or "").strip().lstrip("#")
    if len(raw) == 8:
        try:
            r = int(raw[0:2], 16)
            g = int(raw[2:4], 16)
            b = int(raw[4:6], 16)
            a = int(raw[6:8], 16)
            return QColor(r, g, b, a)
        except ValueError:
            pass
    c = QColor("#" + raw) if raw else QColor(default)
    if not c.isValid():
        c = QColor(default)
    return c


# Handle direction → Qt cursor mapping
_HANDLE_CURSOR = {
    "NW": Qt.CursorShape.SizeFDiagCursor,
    "SE": Qt.CursorShape.SizeFDiagCursor,
    "NE": Qt.CursorShape.SizeBDiagCursor,
    "SW": Qt.CursorShape.SizeBDiagCursor,
    "N":  Qt.CursorShape.SizeVerCursor,
    "S":  Qt.CursorShape.SizeVerCursor,
    "W":  Qt.CursorShape.SizeHorCursor,
    "E":  Qt.CursorShape.SizeHorCursor,
}


class _TitleLineEdit(QLineEdit):
    """Inline title editor — Enter/Esc/focusOut → commit.

    Why subclass rather than the previous eventFilter + ``clearFocus()`` chain:
    a QLineEdit wrapped in QGraphicsProxyWidget does **not** reliably emit
    ``editingFinished`` from a ``clearFocus()`` call — the scene's focus item
    state is decoupled from the inner widget's focus state, so the chain
    silently dead-ends and the cursor stays parked in the edit box. Handling
    Enter/Esc in ``keyPressEvent`` and committing on ``focusOutEvent`` here is
    the same shape that ``note_item._BodyEdit`` already uses successfully for
    the body editor.
    """

    def __init__(self, on_commit, parent=None) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._firing = False  # reentrancy guard: commit hides proxy → focusOut

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self._fire()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self._fire()

    def _fire(self) -> None:
        if self._firing:
            return
        self._firing = True
        try:
            self._on_commit()
        finally:
            self._firing = False


class GroupItem(QGraphicsItem):
    """视觉分组框 —— title + 圆角矩形 + 成员联动 + 鼠标 resize + inline 改名。"""

    def __init__(
        self,
        group_id: str,
        title: str,
        member_ids: List[str],
        rect: QRectF,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.group_id = str(group_id)
        self.title = str(title or tr("canvas.group.default_title", "Group"))
        self.member_ids: List[str] = list(member_ids)

        # _rect 以 (0, 0) 起算 —— 永远归一化,所有 resize 走 setPos + 调整 width/height。
        self._rect = QRectF(0, 0, max(MIN_W, rect.width()), max(MIN_H, rect.height()))
        self.setPos(rect.x(), rect.y())

        self.setZValue(-2.0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setData(0, "group")
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

        self._last_pos = QPointF(self.pos())
        self._suspend_member_sync = False

        # ---- resize 态 ----
        self._resize_handle: Optional[str] = None
        self._resize_start_scene_pos: Optional[QPointF] = None
        self._resize_start_pos: Optional[QPointF] = None
        self._resize_start_rect: Optional[QRectF] = None
        self._resize_start_members: List[str] = []

        # ---- inline 改名 ----
        self._title_edit: Optional[_TitleLineEdit] = None
        self._title_proxy: Optional[QGraphicsProxyWidget] = None
        self._editing_title: bool = False
        self._build_title_editor()

        # ---- per-instance 颜色覆写(SelectionToolbar 色盘按钮 → page.py 入口) ----
        # 空字符串 = 走默认 (canvas_group_bg / canvas_group_title_bg);
        # "#RRGGBB" = body alpha 固定 0x20、title alpha 固定 0x40 派生两层底色,
        # 保留 GroupItem 半透明覆盖语义,不影响 highlight (边框/handle)。
        self._color_override: str = ""

    # NodeItem 鸭子兼容
    @property
    def width(self) -> float:
        return float(self._rect.width())

    @property
    def height(self) -> float:
        return float(self._rect.height())

    # ------------------------------------------------------------------
    # 几何 helpers
    # ------------------------------------------------------------------
    def frame_scene_rect(self) -> QRectF:
        """返回 group 框在 scene 坐标的矩形(供成员包含检测使用)."""
        return QRectF(
            self.pos().x() + self._rect.left(),
            self.pos().y() + self._rect.top(),
            self._rect.width(),
            self._rect.height(),
        )

    def _handle_at(self, pos: QPointF) -> Optional[str]:
        """item-local 坐标 → handle name (NW/N/NE/E/SE/S/SW/W) 或 None。"""
        r = self._rect
        hs = HANDLE
        # 8 handle 矩形(item-local)。corner 是正方形;edge 是窄条横/纵向。
        cx = r.center().x()
        cy = r.center().y()
        handles = {
            "NW": QRectF(r.left(), r.top(), hs, hs),
            "NE": QRectF(r.right() - hs, r.top(), hs, hs),
            "SW": QRectF(r.left(), r.bottom() - hs, hs, hs),
            "SE": QRectF(r.right() - hs, r.bottom() - hs, hs, hs),
            "N":  QRectF(cx - hs * 0.5, r.top(), hs, hs * 0.5),
            "S":  QRectF(cx - hs * 0.5, r.bottom() - hs * 0.5, hs, hs * 0.5),
            "W":  QRectF(r.left(), cy - hs * 0.5, hs * 0.5, hs),
            "E":  QRectF(r.right() - hs * 0.5, cy - hs * 0.5, hs * 0.5, hs),
        }
        # 先 corner 后 edge —— corner 与 edge handle 在角落区域重叠时 corner 优先。
        for name in ("NW", "NE", "SW", "SE", "N", "S", "W", "E"):
            if handles[name].contains(pos):
                return name
        return None

    def _title_rect(self) -> QRectF:
        """title 文本区域(item-local)。改名命中检测用 —— 排除两端 handle 区。"""
        r = self._rect
        return QRectF(r.left() + HANDLE, r.top(), r.width() - 2 * HANDLE, TITLE_H)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def boundingRect(self) -> QRectF:  # noqa: N802
        # 选中态外发光 + handle 小方块需要外扩
        return self._rect.adjusted(-6, -6, 6, 6)

    def shape(self) -> QPainterPath:  # noqa: D401
        """完整 _rect 进入 hit shape —— 框内空白处也能点中 group 选中它。

        不会挡 node / connection 的交互:Z 排序保证 NodeItem(Z=0) > ConnectionItem
        (Z=-1) > GroupItem(Z=-2),Qt 命中检测自顶向下找,优先命中节点/边线;只有
        真正空白处才落到 group。
        """
        path = QPainterPath()
        path.addRoundedRect(self._rect, CORNER_R, CORNER_R)
        return path

    def paint(  # noqa: D401
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,  # noqa: ARG002
        widget: Optional[QWidget] = None,  # noqa: ARG002
    ) -> None:
        # GroupItem 默认配色全套来自黄色 highlight(body=alpha 0x20、title=alpha 0x40、
        # border=full);override 设置后,黄色被整体替换为 override hex,所有三层
        # 沿用同样的 alpha 阶梯派生 —— 不出现"黄色滤镜 + override 色"叠加。
        ov = self._color_override
        if ov and QColor(ov).isValid():
            base = QColor(ov)
            bg = QColor(base.red(), base.green(), base.blue(), 0x20)
            title_bg = QColor(base.red(), base.green(), base.blue(), 0x40)
            border = QColor(base.red(), base.green(), base.blue(), 0xFF)
            accent = QColor(base)  # 选中外发光 + handle fill 也跟着走 override
        else:
            bg = _parse_color_with_alpha(
                Config.get_color("canvas_group_bg", "#F6D39320"), "#F6D39320",
            )
            title_bg = _parse_color_with_alpha(
                Config.get_color("canvas_group_title_bg", "#F6D39340"), "#F6D39340",
            )
            border = QColor(Config.get_color("highlight", "#F6D393"))
            accent = QColor(Config.get_color("highlight", "#F6D393"))
        title_text = QColor(Config.get_color("canvas_group_title_text", "#E8E6DA"))
        selected = self.isSelected()

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 0. 选中态外发光 —— 用 accent 替代原来的 highlight。
        if selected:
            glow = QColor(accent)
            glow.setAlpha(100)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                self._rect.adjusted(-4, -4, 4, 4), CORNER_R + 2, CORNER_R + 2,
            )

        # 1. body + border
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 2.5 if selected else 1.5))
        painter.drawRoundedRect(self._rect, CORNER_R, CORNER_R)

        # 2. title bar(clip 到圆角顶部)
        painter.save()
        clip = QPainterPath()
        clip.addRoundedRect(self._rect, CORNER_R, CORNER_R)
        painter.setClipPath(clip)
        painter.setBrush(QBrush(title_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(self._rect.left(), self._rect.top(),
                                self._rect.width(), TITLE_H))
        painter.restore()

        # 3. title 文本(编辑中由 QLineEdit 顶替,这里不画)
        if not self._editing_title:
            font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
            font.setPixelSize(int(Config.get_font_size("size_small", 12)))
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(title_text))
            painter.drawText(
                self._title_rect(),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self.title,
            )

        # 4. resize handles —— 仅选中态可见,fill 用 accent (override 时也跟着变)。
        if selected:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            handle_fill = QColor(accent)
            handle_border = QColor(Config.get_color("canvas_group_title_text", "#E8E6DA"))
            painter.setBrush(QBrush(handle_fill))
            painter.setPen(QPen(handle_border, 1.0))
            r = self._rect
            hs = HANDLE
            # corner 用 hs × hs;edge 用窄条
            corners = [
                QRectF(r.left(), r.top(), hs, hs),
                QRectF(r.right() - hs, r.top(), hs, hs),
                QRectF(r.left(), r.bottom() - hs, hs, hs),
                QRectF(r.right() - hs, r.bottom() - hs, hs, hs),
            ]
            for h in corners:
                painter.drawRect(h)
            # edge 中点(简化为小方块,与 corner 视觉一致)
            cx = r.center().x()
            cy = r.center().y()
            edges = [
                QRectF(cx - hs * 0.5, r.top(), hs, hs * 0.5),
                QRectF(cx - hs * 0.5, r.bottom() - hs * 0.5, hs, hs * 0.5),
                QRectF(r.left(), cy - hs * 0.5, hs * 0.5, hs),
                QRectF(r.right() - hs * 0.5, cy - hs * 0.5, hs * 0.5, hs),
            ]
            for h in edges:
                painter.drawRect(h)

    # ------------------------------------------------------------------
    # hover 光标切换
    # ------------------------------------------------------------------
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        if self._editing_title:
            self.setCursor(Qt.CursorShape.IBeamCursor)
            super().hoverMoveEvent(event)
            return
        handle = self._handle_at(event.pos()) if self.isSelected() else None
        if handle is not None:
            self.setCursor(_HANDLE_CURSOR[handle])
        elif self._title_rect().contains(event.pos()):
            # title 区域(非 handle):双击改名,单击拖整体 → SizeAll
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    # ------------------------------------------------------------------
    # 拖动联动 + Undo 去抖
    # ------------------------------------------------------------------
    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and not self._suspend_member_sync
        ):
            new_pos = QPointF(self.pos())
            delta = new_pos - self._last_pos
            self._last_pos = new_pos
            if delta.x() != 0.0 or delta.y() != 0.0:
                self._apply_delta_to_members(delta)
                sc = self.scene()
                page = getattr(sc, "_page_ref", None) if sc is not None else None
                if page is not None and hasattr(page, "_reposition_selection_toolbar"):
                    try:
                        page._reposition_selection_toolbar()
                    except Exception:  # pragma: no cover
                        pass
        return super().itemChange(change, value)

    def _apply_delta_to_members(self, delta: QPointF) -> None:
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None:
            return
        instances = getattr(page, "_instances", {})
        for mid in self.member_ids:
            item = instances.get(mid)
            if item is None:
                continue
            item.setPos(item.pos() + delta)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        # Click outside the title rect while renaming → commit first. GroupItem
        # is not ItemIsFocusable, so a click on its own body would not transfer
        # scene focus and the editor would otherwise stay parked.
        if self._editing_title and not self._title_rect().contains(event.pos()):
            self._commit_rename()

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # 1) Resize handle?
        if self.isSelected():
            handle = self._handle_at(event.pos())
            if handle is not None:
                self._resize_handle = handle
                self._resize_start_scene_pos = QPointF(event.scenePos())
                self._resize_start_pos = QPointF(self.pos())
                self._resize_start_rect = QRectF(self._rect)
                self._resize_start_members = list(self.member_ids)
                event.accept()
                return

        # 2) 默认 move 路径:先 super 选中,再收集 _pending_move
        super().mousePressEvent(event)
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None or getattr(page, "_in_undo", False):
            return
        moves: List[tuple] = [(self._group_move_key(), QPointF(self.pos()))]
        seen = {self.group_id}
        instances = getattr(page, "_instances", {})
        for mid in self.member_ids:
            if mid in seen:
                continue
            item = instances.get(mid)
            if item is None:
                continue
            moves.append((mid, QPointF(item.pos())))
            seen.add(mid)
        page._pending_move = moves

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._resize_handle is not None and self._resize_start_scene_pos is not None:
            self._apply_resize_to(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _apply_resize_to(self, scene_pos: QPointF) -> None:
        """根据当前鼠标 scene 坐标 + handle + 起始快照 → 调整 self.pos + _rect。"""
        if (
            self._resize_handle is None
            or self._resize_start_scene_pos is None
            or self._resize_start_pos is None
            or self._resize_start_rect is None
        ):
            return
        delta = scene_pos - self._resize_start_scene_pos
        h = self._resize_handle
        sx = self._resize_start_pos.x()
        sy = self._resize_start_pos.y()
        sw = self._resize_start_rect.width()
        sh = self._resize_start_rect.height()

        new_x, new_y, new_w, new_h = sx, sy, sw, sh

        if "W" in h:
            new_x = sx + delta.x()
            new_w = sw - delta.x()
        if "E" in h:
            new_w = sw + delta.x()
        if "N" in h:
            new_y = sy + delta.y()
            new_h = sh - delta.y()
        if "S" in h:
            new_h = sh + delta.y()

        # 约束最小尺寸 —— 钉住对侧不动
        if new_w < MIN_W:
            if "W" in h:
                new_x = sx + (sw - MIN_W)
            new_w = MIN_W
        if new_h < MIN_H:
            if "N" in h:
                new_y = sy + (sh - MIN_H)
            new_h = MIN_H

        # 应用:先挂起 member-sync,再 setPos + 改 _rect,最后恢复 _last_pos
        self._suspend_member_sync = True
        try:
            self.prepareGeometryChange()
            if QPointF(new_x, new_y) != self.pos():
                self.setPos(new_x, new_y)
            self._last_pos = QPointF(self.pos())
            self._rect = QRectF(0, 0, new_w, new_h)
        finally:
            self._suspend_member_sync = False
        # title proxy 跟着 rect 移
        self._reposition_title_proxy()
        self.update()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        # Resize 完成 → push undo + 重算成员
        if self._resize_handle is not None:
            handle = self._resize_handle
            start_pos = self._resize_start_pos
            start_rect = self._resize_start_rect
            start_members = self._resize_start_members
            self._resize_handle = None
            self._resize_start_scene_pos = None
            self._resize_start_pos = None
            self._resize_start_rect = None
            self._resize_start_members = []

            sc = self.scene()
            page = getattr(sc, "_page_ref", None) if sc is not None else None
            if page is None:
                event.accept()
                return

            new_pos = QPointF(self.pos())
            new_rect = QRectF(self._rect)
            # 计算几何 + push undo —— 注意:跳过 in_undo,跳过零变化
            if (start_pos == new_pos and start_rect == new_rect):
                event.accept()
                return

            # 全局重算(嵌套 group 走最小框优先) → 收集 new_members
            if hasattr(page, "_recompute_all_group_memberships"):
                page._recompute_all_group_memberships()
            new_members = list(self.member_ids)

            if getattr(page, "_in_undo", False):
                event.accept()
                return

            from .undo import ResizeGroupCmd
            page._undo_stack.push(ResizeGroupCmd(
                page,
                self.group_id,
                old_pos=start_pos,
                old_rect=start_rect,
                old_members=list(start_members),
                new_pos=new_pos,
                new_rect=new_rect,
                new_members=list(new_members),
            ))
            event.accept()
            return

        # 普通 move 释放 —— 复用 MoveNodeCmd 路径
        super().mouseReleaseEvent(event)
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None:
            return
        moves = getattr(page, "_pending_move", None)
        if not moves:
            return
        page._pending_move = None
        if getattr(page, "_in_undo", False):
            return
        actual = []
        groups = getattr(page, "_groups", {})
        instances = getattr(page, "_instances", {})
        for key, old in moves:
            if isinstance(key, str) and key.startswith("__group__:"):
                gid = key.split(":", 1)[1]
                grp = groups.get(gid)
                if grp is None:
                    continue
                new = grp.pos()
                if QPointF(old) != QPointF(new):
                    actual.append((key, QPointF(old), QPointF(new)))
            else:
                item = instances.get(key)
                if item is None:
                    continue
                new = item.pos()
                if QPointF(old) != QPointF(new):
                    actual.append((key, QPointF(old), QPointF(new)))
        if not actual:
            return
        from .undo import MoveNodeCmd
        page._undo_stack.push(MoveNodeCmd(page, actual))

    def _group_move_key(self) -> str:
        return f"__group__:{self.group_id}"

    # ------------------------------------------------------------------
    # 双击改名
    # ------------------------------------------------------------------
    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._title_rect().contains(event.pos())
        ):
            self._enter_rename()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _build_title_editor(self) -> None:
        """构造内嵌 QLineEdit + ProxyWidget,默认隐藏。"""
        self._title_edit = _TitleLineEdit(on_commit=self._commit_rename)
        self._apply_title_style(self._title_edit)
        self._title_proxy = QGraphicsProxyWidget(self)
        self._title_proxy.setWidget(self._title_edit)
        self._title_proxy.setVisible(False)
        self._reposition_title_proxy()

    def _apply_title_style(self, w: QLineEdit) -> None:
        bg = Config.get_color("canvas_group_title_bg", "#F6D39340")
        fg = Config.get_color("canvas_group_title_text", "#E8E6DA")
        border = Config.get_color("checked_1", "#67CCF5")
        size = int(Config.get_font_size("size_small", 12))
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        w.setStyleSheet(
            f"QLineEdit {{ background: {bg}; color: {fg}; border: 1px solid {border};"
            f" border-radius: 3px; padding: 2px 4px;"
            f" font-family: '{family}'; font-size: {size}px; font-weight: bold; }}"
        )

    def _reposition_title_proxy(self) -> None:
        if self._title_proxy is None:
            return
        tr_rect = self._title_rect()
        self._title_proxy.setPos(tr_rect.left() - 4, tr_rect.top() + 2)
        self._title_proxy.resize(tr_rect.width() + 8, TITLE_H - 4)

    def _enter_rename(self) -> None:
        if self._title_edit is None or self._title_proxy is None:
            return
        self._title_edit.setText(self.title)
        self._editing_title = True
        self._reposition_title_proxy()
        self._title_proxy.setVisible(True)
        self._title_edit.setFocus(Qt.FocusReason.MouseFocusReason)
        self._title_edit.selectAll()
        self.update()

    def _commit_rename(self) -> None:
        if not self._editing_title or self._title_edit is None:
            return
        new_title = (self._title_edit.text() or "").strip()
        if not new_title:
            new_title = tr("canvas.group.default_title", "Group")
        old_title = self.title
        self._editing_title = False
        if self._title_proxy is not None:
            self._title_proxy.setVisible(False)
        if new_title == old_title:
            self.update()
            return
        # 写入 + 走 undo
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None or getattr(page, "_in_undo", False):
            self.title = new_title
            self.update()
            return
        from .undo import RenameCmd
        page._undo_stack.push(
            RenameCmd(page, "group", self.group_id, old_title, new_title)
        )

    # ------------------------------------------------------------------
    # 成员维护 / 序列化
    # ------------------------------------------------------------------
    def set_rect(self, rect: QRectF) -> None:
        """更新本地几何(item-local;归一化到 (0,0,w,h),同步 setPos)。"""
        self.prepareGeometryChange()
        # rect.x/y 视为新的 scene 位置(归一化策略)
        if rect.x() != 0.0 or rect.y() != 0.0:
            self._suspend_member_sync = True
            try:
                self.setPos(QPointF(rect.x(), rect.y()))
                self._last_pos = QPointF(rect.x(), rect.y())
            finally:
                self._suspend_member_sync = False
        self._rect = QRectF(0, 0, max(MIN_W, rect.width()), max(MIN_H, rect.height()))
        self._reposition_title_proxy()
        self.update()

    # ------------------------------------------------------------------
    # Per-instance color override (SelectionToolbar 色盘按钮 → page.py 入口)
    # ------------------------------------------------------------------
    def set_color_override(self, value: Optional[str]) -> None:
        """设置/清空 per-canvas 颜色覆写。空 → 走默认主题色。"""
        new = "" if value is None else str(value).strip()
        if new == self._color_override:
            return
        self._color_override = new
        self.update()

    @property
    def color_override(self) -> str:
        return self._color_override

    def serialize(self) -> dict:
        out = {
            "id": self.group_id,
            "title": self.title,
            "members": list(self.member_ids),
            "rect": {
                "x": float(self.pos().x()),
                "y": float(self.pos().y()),
                "w": float(self._rect.width()),
                "h": float(self._rect.height()),
            },
        }
        if self._color_override:
            out["color_override"] = self._color_override
        return out


__all__ = ["GroupItem", "TITLE_H", "PADDING"]
