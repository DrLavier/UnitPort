# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas.note_item — Note 注释节点 GraphicsItem.

NoteItem 是非执行节点（schema_id="note"）在画布上的可视化项。与 NodeItem 平级
但接口鸭子兼容（``.node_id`` / ``.manifest`` / ``.params`` / ``.pos()`` /
``.width`` / ``.height`` / ``.iter_ports()`` 返回空 / ``.setSelected``），使
:meth:`CanvasPage.to_workflow_dict` 的现有循环无需特殊分支即可序列化它的
``title`` / ``body`` 两个 string 参数。

交互：
- 默认状态：painter 直接画 title bar + body 文本，无任何嵌入 widget。
- 双击 title bar 区域 → 浮出 QLineEdit 编辑 title；focusOut / Esc 提交并隐藏。
- 双击 body 区域       → 浮出 QPlainTextEdit 编辑 body；同上。

颜色全部从 ``Config.get_color`` 取（system.ini [Theme]，违反 §1.5 即 bug）。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

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
    QPlainTextEdit,
    QStyleOptionGraphicsItem,
    QWidget,
)

from unitport_sdk import Config

from application.compiler.nodes import NodeManifest


# 几何 —— Note 卡片采用与 NodeItem 一致的视觉骨架但更紧凑
NOTE_W = 260
NOTE_H = 180
TITLE_H = 30
CORNER_R = 6
H_PAD = 12
V_PAD = 8
BORDER_W = 1.0

# Resize 配置 —— 与 GroupItem 8 向 handle 对齐(NW/N/NE/E/SE/S/SW/W)。
HANDLE = 10
MIN_W = 140
MIN_H = 90


# Handle direction → Qt cursor mapping (与 GroupItem 同表;保持一致体验)
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
    """Inline note-title editor — Enter/Esc/focusOut → commit.

    Why subclass rather than the previous eventFilter + ``clearFocus()`` chain:
    a QLineEdit inside QGraphicsProxyWidget does not reliably emit
    ``editingFinished`` from ``clearFocus()`` — scene focus is decoupled from
    inner-widget focus, so the chain silently dead-ends. Same shape as
    ``_BodyEdit`` below.
    """

    def __init__(self, on_commit, parent=None) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._firing = False

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


class _BodyEdit(QPlainTextEdit):
    """QPlainTextEdit 子类:Esc 退出编辑(clearFocus触发focusOut),focusOut → on_commit。

    NoteItem 是 QGraphicsItem(非 QObject),无法 installEventFilter;
    所以 body 的失焦提交走子类 focusOutEvent 直回调。
    """

    def __init__(self, on_commit, parent=None) -> None:
        super().__init__(parent)
        self._on_commit = on_commit

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.clearFocus()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        try:
            if callable(self._on_commit):
                self._on_commit()
        except Exception:  # pragma: no cover
            pass


class NoteItem(QGraphicsItem):
    """Note 注释节点 —— title + 多行 body，零端口，零执行。"""

    def __init__(
        self,
        manifest: NodeManifest,
        node_id: str,
        params: Optional[dict] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.manifest = manifest
        self.node_id = node_id
        self.params: dict = dict(params or {})

        # NodeItem 鸭子兼容字段 —— width/height 可由调用方覆盖(画布加载时
        # 从 ui.width / ui.height 恢复)。约束到 MIN_W / MIN_H 以下。
        self._width = float(max(MIN_W, width if width is not None else NOTE_W))
        self._height = float(max(MIN_H, height if height is not None else NOTE_H))

        # ---- resize 态 ----
        self._resize_handle: Optional[str] = None
        self._resize_start_scene_pos: Optional[QPointF] = None
        self._resize_start_pos: Optional[QPointF] = None
        self._resize_start_size: Optional[tuple] = None  # (w, h)

        # Qt flags —— 与 NodeItem 对齐
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setData(0, "note")  # ROLE_KIND = 0 in items.py 协议
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

        # Inline edit widgets（默认隐藏；双击进入编辑）
        self._editing_field: Optional[str] = None  # "title" | "body" | None
        self._title_edit: Optional[_TitleLineEdit] = None
        self._body_edit: Optional[_BodyEdit] = None
        self._title_proxy: Optional[QGraphicsProxyWidget] = None
        self._body_proxy: Optional[QGraphicsProxyWidget] = None
        self._build_edit_widgets()

        # Per-instance 颜色覆写 (SelectionToolbar 色盘按钮 → page.py 入口)。
        # 空 → 走默认 (bg_1 + canvas_note_title_bg);"#RRGGBB" →
        # body 用 base.darker(140)、title 用 base 本身,保留 Note 卡片
        # "标题色 vs 内容色有明暗对比" 的视觉。border 仍走 canvas_note_border /
        # highlight 不受覆写影响。
        self._color_override: str = ""

    # ------------------------------------------------------------------
    # NodeItem 鸭子兼容接口
    # ------------------------------------------------------------------
    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    def iter_ports(self) -> Iterable[Any]:
        return iter(())

    def port(self, name: str, direction: str = "in") -> Optional[Any]:  # noqa: ARG002
        return None

    # ------------------------------------------------------------------
    # 几何 helpers
    # ------------------------------------------------------------------
    def _handle_at(self, pos: QPointF) -> Optional[str]:
        """item-local 坐标 → handle name (NW/N/NE/E/SE/S/SW/W) 或 None。

        与 GroupItem._handle_at 同形:8 个矩形分布在 (0, 0, w, h) 的角与边中点;
        corner 优先于 edge(角落区域 corner 命中)。
        """
        w = self._width
        h = self._height
        hs = HANDLE
        cx = w * 0.5
        cy = h * 0.5
        handles = {
            "NW": QRectF(0, 0, hs, hs),
            "NE": QRectF(w - hs, 0, hs, hs),
            "SW": QRectF(0, h - hs, hs, hs),
            "SE": QRectF(w - hs, h - hs, hs, hs),
            "N":  QRectF(cx - hs * 0.5, 0, hs, hs * 0.5),
            "S":  QRectF(cx - hs * 0.5, h - hs * 0.5, hs, hs * 0.5),
            "W":  QRectF(0, cy - hs * 0.5, hs * 0.5, hs),
            "E":  QRectF(w - hs * 0.5, cy - hs * 0.5, hs * 0.5, hs),
        }
        for name in ("NW", "NE", "SW", "SE", "N", "S", "W", "E"):
            if handles[name].contains(pos):
                return name
        return None

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-2, -2, self._width + 4, self._height + 4)

    def paint(  # noqa: D401
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,  # noqa: ARG002
        widget: Optional[QWidget] = None,  # noqa: ARG002
    ) -> None:
        # Override 时:title = override 本色,body = title 上叠 60% 黑 (即每通道 ×0.4),
        # border + 选中 highlight + handle fill 一并切到 override,避免黄色滤镜残留。
        # 默认时:body / title / border 各自走原 Theme slot,选中态 border 用 highlight。
        ov = self._color_override
        if ov and QColor(ov).isValid():
            base = QColor(ov)
            title_bg = QColor(base)
            body_bg = QColor(
                int(base.red() * 0.4),
                int(base.green() * 0.4),
                int(base.blue() * 0.4),
            )
            border = QColor(base)
            accent = QColor(base)
        else:
            body_bg = QColor(Config.get_color("bg_1", "#1E1E1E"))
            title_bg = QColor(Config.get_color("canvas_note_title_bg", "#2A3140"))
            border = QColor(Config.get_color("canvas_note_border", "#3D4555"))
            accent = QColor(Config.get_color("highlight", "#F6D393"))
            if self.isSelected():
                border = QColor(accent)
        title_text = QColor(Config.get_color("canvas_note_title_text", "#E8E6DA"))
        body_text = QColor(Config.get_color("canvas_note_body_text", "#C0BCB0"))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # body 圆角矩形
        rect = QRectF(0, 0, self._width, self._height)
        painter.setBrush(QBrush(body_bg))
        painter.setPen(QPen(border, BORDER_W))
        painter.drawRoundedRect(rect, CORNER_R, CORNER_R)

        # title bar（clip 到圆角矩形顶部）
        painter.save()
        clip = QPainterPath()
        clip.addRoundedRect(rect, CORNER_R, CORNER_R)
        painter.setClipPath(clip)
        painter.setBrush(QBrush(title_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(0, 0, self._width, TITLE_H))
        painter.restore()

        # title bar 与 body 之间的细分隔线
        painter.setPen(QPen(border, 0.8))
        painter.drawLine(QPointF(0, TITLE_H), QPointF(self._width, TITLE_H))

        font_family = Config.get_value("Font", "family", "Microsoft YaHei")
        title_size = int(Config.get_font_size("size_small", 12))
        body_size = int(Config.get_font_size("size_small", 11))

        # title 文本（仅当未在编辑 title 时绘制；编辑中 QLineEdit 已遮住）
        if self._editing_field != "title":
            font = QFont(font_family)
            font.setPixelSize(title_size)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(title_text))
            title_rect = QRectF(H_PAD, 0, self._width - 2 * H_PAD, TITLE_H)
            painter.drawText(
                title_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                str(self.params.get("title") or "Note"),
            )

        # resize handles —— 仅选中态可见(与 GroupItem 同样约束)。编辑文本时不显示,
        # 避免 4 个 corner 落在 QLineEdit / QPlainTextEdit 上影响文本拖选。
        if self.isSelected() and self._editing_field is None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            # handle fill 沿用 accent —— override 时 = override 色,默认时 = highlight。
            handle_fill = QColor(accent)
            handle_border = QColor(Config.get_color("canvas_note_title_text", "#E8E6DA"))
            painter.setBrush(QBrush(handle_fill))
            painter.setPen(QPen(handle_border, 1.0))
            w = self._width
            h = self._height
            hs = HANDLE
            corners = [
                QRectF(0, 0, hs, hs),
                QRectF(w - hs, 0, hs, hs),
                QRectF(0, h - hs, hs, hs),
                QRectF(w - hs, h - hs, hs, hs),
            ]
            for hd in corners:
                painter.drawRect(hd)
            cx = w * 0.5
            cy = h * 0.5
            edges = [
                QRectF(cx - hs * 0.5, 0, hs, hs * 0.5),
                QRectF(cx - hs * 0.5, h - hs * 0.5, hs, hs * 0.5),
                QRectF(0, cy - hs * 0.5, hs * 0.5, hs),
                QRectF(w - hs * 0.5, cy - hs * 0.5, hs * 0.5, hs),
            ]
            for hd in edges:
                painter.drawRect(hd)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # body 文本
        if self._editing_field != "body":
            font = QFont(font_family)
            font.setPixelSize(body_size)
            painter.setFont(font)
            painter.setPen(QPen(body_text))
            body_rect = QRectF(
                H_PAD,
                TITLE_H + V_PAD,
                self._width - 2 * H_PAD,
                self._height - TITLE_H - 2 * V_PAD,
            )
            text = str(self.params.get("body") or "")
            painter.drawText(
                body_rect,
                int(
                    Qt.AlignmentFlag.AlignTop
                    | Qt.AlignmentFlag.AlignLeft
                    | Qt.TextFlag.TextWordWrap
                ),
                text,
            )

    # ------------------------------------------------------------------
    # hover 光标切换 —— 选中态时角/边显示 resize 光标
    # ------------------------------------------------------------------
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        if self._editing_field is not None:
            super().hoverMoveEvent(event)
            return
        handle = self._handle_at(event.pos()) if self.isSelected() else None
        if handle is not None:
            self.setCursor(_HANDLE_CURSOR[handle])
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        # resize handle 上的双击不进入编辑态(否则会卡在 handle 区域的小角落)
        if self.isSelected() and self._handle_at(event.pos()) is not None:
            event.accept()
            return
        local_y = event.pos().y()
        if local_y < TITLE_H:
            self._enter_edit("title")
        else:
            self._enter_edit("body")
        event.accept()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        # 单击:若处于编辑态且点到了 edit widget 之外,先 commit 再走默认逻辑。
        # NoteItem 不带 ItemIsFocusable,所以点在 note 自身 body 区不会自动转移
        # scene focus,这里必须显式 commit 否则编辑器一直留着。
        if self._editing_field is not None:
            local_y = event.pos().y()
            on_title = local_y < TITLE_H
            in_field = (self._editing_field == "title" and on_title) or (
                self._editing_field == "body" and not on_title
            )
            if not in_field:
                self._commit_field(self._editing_field)

        # 命中 resize handle(仅选中态) → 进入 resize,屏蔽默认 move 路径。
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.isSelected()
            and self._editing_field is None
        ):
            handle = self._handle_at(event.pos())
            if handle is not None:
                self._resize_handle = handle
                self._resize_start_scene_pos = QPointF(event.scenePos())
                self._resize_start_pos = QPointF(self.pos())
                self._resize_start_size = (self._width, self._height)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._resize_handle is not None and self._resize_start_scene_pos is not None:
            self._apply_resize_to(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _apply_resize_to(self, scene_pos: QPointF) -> None:
        """根据当前鼠标 scene 坐标 + handle + 起始快照 → 调整 self.pos + size。

        W/N handle 在拉伸时需同步移动 self.pos(),保持对侧角不动。
        """
        if (
            self._resize_handle is None
            or self._resize_start_scene_pos is None
            or self._resize_start_pos is None
            or self._resize_start_size is None
        ):
            return
        delta = scene_pos - self._resize_start_scene_pos
        h = self._resize_handle
        sx = self._resize_start_pos.x()
        sy = self._resize_start_pos.y()
        sw, sh = self._resize_start_size

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

        # 钉住对侧,夹到 MIN_W / MIN_H
        if new_w < MIN_W:
            if "W" in h:
                new_x = sx + (sw - MIN_W)
            new_w = MIN_W
        if new_h < MIN_H:
            if "N" in h:
                new_y = sy + (sh - MIN_H)
            new_h = MIN_H

        self.prepareGeometryChange()
        if QPointF(new_x, new_y) != self.pos():
            self.setPos(new_x, new_y)
        self._width = float(new_w)
        self._height = float(new_h)
        self._reposition_edit_proxies()
        self.update()

    def itemChange(self, change, value):  # noqa: N802, D401
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            sc = self.scene()
            page = getattr(sc, "_page_ref", None) if sc is not None else None
            if page is not None and hasattr(page, "_reposition_selection_toolbar"):
                try:
                    page._reposition_selection_toolbar()
                except Exception:  # pragma: no cover
                    pass
        return super().itemChange(change, value)

    # ---- Undo 去抖：与 NodeItem 一致，把自身纳入 _pending_move ----
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        # Resize 完成 → push ResizeNoteCmd
        if self._resize_handle is not None:
            start_pos = self._resize_start_pos
            start_size = self._resize_start_size
            self._resize_handle = None
            self._resize_start_scene_pos = None
            self._resize_start_pos = None
            self._resize_start_size = None
            event.accept()

            sc = self.scene()
            page = getattr(sc, "_page_ref", None) if sc is not None else None
            if page is None or getattr(page, "_in_undo", False):
                return
            new_pos = QPointF(self.pos())
            new_size = (self._width, self._height)
            if (
                start_pos is not None
                and start_size is not None
                and (start_pos != new_pos or start_size != new_size)
            ):
                from .undo import ResizeNoteCmd
                page._undo_stack.push(ResizeNoteCmd(
                    page,
                    self.node_id,
                    old_pos=start_pos,
                    old_size=start_size,
                    new_pos=new_pos,
                    new_size=new_size,
                ))
            return

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
        for inst, old in moves:
            item = page._instances.get(inst)
            if item is None:
                continue
            new = item.pos()
            if QPointF(old) != QPointF(new):
                actual.append((inst, QPointF(old), QPointF(new)))
        if not actual:
            return
        from .undo import MoveNodeCmd
        page._undo_stack.push(MoveNodeCmd(page, actual))
        if hasattr(page, "_recompute_all_group_memberships"):
            try:
                page._recompute_all_group_memberships()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    # 由 ResizeNoteCmd 调用,直接还原/重做 size
    # ------------------------------------------------------------------
    def set_size(self, width: float, height: float) -> None:
        new_w = float(max(MIN_W, width))
        new_h = float(max(MIN_H, height))
        if new_w == self._width and new_h == self._height:
            return
        self.prepareGeometryChange()
        self._width = new_w
        self._height = new_h
        self._reposition_edit_proxies()
        self.update()

    # ------------------------------------------------------------------
    # Edit widgets
    # ------------------------------------------------------------------
    def _build_edit_widgets(self) -> None:
        # Title — _TitleLineEdit (subclassed for Enter/Esc/focusOut → commit)
        self._title_edit = _TitleLineEdit(on_commit=lambda: self._commit_field("title"))
        self._title_edit.setText(str(self.params.get("title") or "Note"))
        self._apply_title_style(self._title_edit)
        self._title_proxy = QGraphicsProxyWidget(self)
        self._title_proxy.setWidget(self._title_edit)
        self._title_proxy.setVisible(False)

        # Body — QPlainTextEdit (subclassed for Esc + focusOut 回调提交)
        self._body_edit = _BodyEdit(on_commit=lambda: self._commit_field("body"))
        self._body_edit.setPlainText(str(self.params.get("body") or ""))
        self._body_edit.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self._apply_body_style(self._body_edit)
        self._body_proxy = QGraphicsProxyWidget(self)
        self._body_proxy.setWidget(self._body_edit)
        self._body_proxy.setVisible(False)

        # 初始尺寸/位置由 _reposition_edit_proxies 一并设置 —— resize 时复用同一路径
        self._reposition_edit_proxies()

    def _reposition_edit_proxies(self) -> None:
        """根据当前 _width / _height 同步两个 inline editor proxy 的位置与尺寸。"""
        if self._title_proxy is not None:
            self._title_proxy.setPos(H_PAD - 4, 4)
            self._title_proxy.resize(
                max(20.0, self._width - 2 * H_PAD + 8),
                max(8.0, TITLE_H - 8),
            )
        if self._body_proxy is not None:
            self._body_proxy.setPos(H_PAD - 4, TITLE_H + V_PAD - 4)
            self._body_proxy.resize(
                max(20.0, self._width - 2 * H_PAD + 8),
                max(8.0, self._height - TITLE_H - 2 * V_PAD + 8),
            )

    def _apply_title_style(self, w: QLineEdit) -> None:
        bg = Config.get_color("canvas_note_title_bg", "#2A3140")
        fg = Config.get_color("canvas_note_title_text", "#E8E6DA")
        border = Config.get_color("checked_1", "#67CCF5")
        size = int(Config.get_font_size("size_small", 12))
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        w.setStyleSheet(
            f"QLineEdit {{ background: {bg}; color: {fg}; border: 1px solid {border};"
            f" border-radius: 3px; padding: 2px 4px;"
            f" font-family: '{family}'; font-size: {size}px; font-weight: bold; }}"
        )

    def _apply_body_style(self, w: QPlainTextEdit) -> None:
        bg = Config.get_color("bg_1", "#1E1E1E")
        fg = Config.get_color("canvas_note_body_text", "#C0BCB0")
        border = Config.get_color("checked_1", "#67CCF5")
        size = int(Config.get_font_size("size_small", 11))
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        w.setStyleSheet(
            f"QPlainTextEdit {{ background: {bg}; color: {fg};"
            f" border: 1px solid {border}; border-radius: 3px;"
            f" font-family: '{family}'; font-size: {size}px; }}"
        )

    def _enter_edit(self, field: str) -> None:
        if field == "title":
            if self._title_proxy is None or self._title_edit is None:
                return
            self._title_edit.setText(str(self.params.get("title") or ""))
            self._editing_field = "title"
            self._title_proxy.setVisible(True)
            self._title_edit.setFocus(Qt.FocusReason.MouseFocusReason)
            self._title_edit.selectAll()
        elif field == "body":
            if self._body_proxy is None or self._body_edit is None:
                return
            self._body_edit.setPlainText(str(self.params.get("body") or ""))
            self._editing_field = "body"
            self._body_proxy.setVisible(True)
            self._body_edit.setFocus(Qt.FocusReason.MouseFocusReason)
        self.update()

    def _commit_field(self, field: str) -> None:
        if field == "title" and self._title_edit is not None:
            self.params["title"] = self._title_edit.text()
            if self._title_proxy is not None:
                self._title_proxy.setVisible(False)
        elif field == "body" and self._body_edit is not None:
            self.params["body"] = self._body_edit.toPlainText()
            if self._body_proxy is not None:
                self._body_proxy.setVisible(False)
        if self._editing_field == field:
            self._editing_field = None
        # 写入 dirty tracker（参照 NodeItem 模式，由 page 自动通过 undo_stack 监测）
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

    # ------------------------------------------------------------------
    # 兼容方法（CanvasPage 在某些场景下会按 NodeItem 接口调用）
    # ------------------------------------------------------------------
    def retranslate(self) -> None:
        """Note 节点 title 是用户输入，不走 i18n；占位以保持 NodeItem 接口对齐。"""
        self.update()
