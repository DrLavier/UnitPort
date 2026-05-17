"""application.ui.canvas.node_library_overlay — Node Library button + card.

Three pieces:

1. ``NodeLibraryButton``         — LaviButton(kind="border", spec="notice", icon=icon_node).
                                   Default 45×45; can be resized by external hosts via
                                   :meth:`set_button_size` (e.g. when Mission Control's
                                   top row hosts the button at row height).
2. ``NodeLibraryCard``           — QFrame, transparent borderless container,
                                   hosts ``TitleGroupBox("Node Library")`` + ``NodeLibraryPanel``.
3. ``NodeLibraryCardController`` — anchors the card to canvas top-left, owns visibility
                                   toggling, routes ``panel.node_requested`` to
                                   ``CanvasPage.spawn_node`` (gated by canvas interactive
                                   state). Parent on canvas_page so resize-tracking and
                                   destruction follow the page.
4. ``NodeLibraryOverlay``        — back-compat wrapper: builds a floating button anchored
                                   at (10,10) plus a controller below it. Used by
                                   non-embedded ``CanvasPage`` callers (standalone tests,
                                   future feature windows).

Embedded canvas (inside MissionControlPanel) constructs only the controller; the button
lives in MC's top row and drives ``controller.set_open(bool)`` via its ``toggled`` signal.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, TitleGroupBox, i18n_bind, log_debug, log_warning, setButton, tr

from .node_library_panel import NodeLibraryPanel


# ---- 常量 ----------------------------------------------------------------------

#: 按钮 / 卡片距画布左上角的 padding（用户要求 10px）
EDGE_PADDING = 10
#: 按钮与卡片之间的垂直缝隙
BUTTON_CARD_GAP = 8
#: 按钮尺寸（正方形）默认值；外部宿主可调
BUTTON_SIZE = 45
#: 卡片默认尺寸
CARD_W = 280
CARD_H = 840
#: 嵌入模式 (button-外置) 下卡片相对画布顶部的额外下移 — 给上层 MC 顶部 row
#: 留出视觉缓冲，避免卡片紧贴 row 下沿
EMBEDDED_CARD_TOP_OFFSET = 120


# =============================================================================
# Button
# =============================================================================

class NodeLibraryButton(QWidget):
    """Node Library 浮动开关按钮 / Floating toggle button.

    外观为 ``LaviButton`` (kind="border", spec="notice", icon="icon_node"),
    可 toggle，按下/弹起 emit ``toggled(bool)``。

    支持运行时 :meth:`set_button_size` 调整尺寸 — Mission Control 顶部 row 把按钮
    塞进 row 时按 row 高度收缩。
    """

    toggled = pyqtSignal(bool)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        size: int = BUTTON_SIZE,
    ) -> None:
        super().__init__(parent)
        self._size = int(size)
        self.setFixedSize(self._size, self._size)
        # 透明背景：本 widget 只是 LaviButton 的容器，让 LaviButton 自己画
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._btn = setButton(
            "canvas.btn.node_lib",
            self._size,
            self._size,
            kind="border",
            spec="notice",
            icon="icon_node",
            default="",
            checkable=True,
            icon_only=True,
        )
        i18n_bind(self._btn, "setToolTip",
                  "canvas.btn.node_lib.tip", "Node Library")
        self._btn.setToolTipDuration(-1)
        self._btn.toggled.connect(self._on_btn_toggled)

        layout.addWidget(self._btn)

    def _on_btn_toggled(self, checked: bool) -> None:
        self.toggled.emit(checked)

    def is_open(self) -> bool:
        return self._btn.isChecked()

    def set_open(self, open_: bool) -> None:
        if self._btn.isChecked() != open_:
            self._btn.setChecked(open_)

    def set_button_size(self, size: int) -> None:
        """Resize the button (and its inner LaviButton) at runtime."""
        size = max(16, int(size))
        if size == self._size:
            return
        self._size = size
        self.setFixedSize(size, size)
        # LaviButton exposes setFixedSize via QPushButton; safe to call.
        try:
            self._btn.setFixedSize(size, size)
        except Exception:
            pass


# =============================================================================
# Card
# =============================================================================

class NodeLibraryCard(QFrame):
    """透明无边框容器，承载 Node Library 面板."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("nodeLibraryCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(CARD_W, CARD_H)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        self._group = TitleGroupBox(
            "canvas.node_lib.title",
            default="Node Library",
        )
        # TitleGroupBox 内部布局：放 NodeLibraryPanel
        group_layout = QVBoxLayout(self._group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(0)

        self._panel = NodeLibraryPanel(self._group)
        group_layout.addWidget(self._panel, 1)

        layout.addWidget(self._group, 1)

        self.refresh_style()

    @property
    def panel(self) -> NodeLibraryPanel:
        return self._panel

    def refresh_style(self) -> None:
        """外层容器永远透明；内层 group/panel 仍跟随主题刷新."""
        self.setStyleSheet(
            """
            QFrame#nodeLibraryCard {
                background-color: transparent;
                border: 0;
            }
            """
        )
        self._group.refresh_style()
        # 'Node Library' 标题字号 size_small（SDK TitleGroupBox 默认 size_large）
        # 在 SDK refresh_style 之后追加同名规则覆盖；后写规则同特异性下生效
        small = int(Config.get_font_size("size_small"))
        self._group.setStyleSheet(
            self._group.styleSheet()
            + f"\nQGroupBox {{ font-size: {small}px; }}"
            f"\nQGroupBox::title {{ font-size: {small}px; }}\n"
        )
        self._panel.refresh_style()


# =============================================================================
# Card controller (no button) — used by embedded canvas inside MissionControlPanel
# =============================================================================

class NodeLibraryCardController(QObject):
    """协调 Node Library card：定位 + 显隐 + 单击放置.

    Card parent 在 ``canvas_page`` 上，所以随其 resize 自动跟随，析构时一并清理。
    本 controller 不创建按钮 —— 由外部宿主（MissionControlPanel 顶部 row 或
    backward-compat 的 ``NodeLibraryOverlay``）创建按钮并把 toggle 转发到
    :meth:`set_open`。

    Card 锚点：以 canvas 左上角 (EDGE_PADDING, anchor_top) 为基准；默认
    ``anchor_top = EDGE_PADDING + EMBEDDED_CARD_TOP_OFFSET`` —— 嵌入模式下
    上层 MC 面板有自己的顶部 row 覆盖在 canvas 之上，card 需要整体下移
    给 row 留视觉缓冲。Backward-compat 路径覆盖此默认，使用 "在浮动按钮下方"
    的偏移，由 ``NodeLibraryOverlay`` 显式调用 :meth:`set_anchor_top`。
    """

    def __init__(self, canvas_page) -> None:
        super().__init__(canvas_page)
        self._canvas_page = canvas_page
        self._card = NodeLibraryCard(canvas_page)
        self._card.hide()
        self._card.panel.node_requested.connect(self._on_node_requested)
        self._anchor_top = EDGE_PADDING + EMBEDDED_CARD_TOP_OFFSET

        canvas_page.installEventFilter(self)
        self._reposition()

    # ---- positioning ----

    def set_anchor_top(self, y: int) -> None:
        self._anchor_top = int(y)
        self._reposition()

    def _reposition(self) -> None:
        self._card.move(EDGE_PADDING, self._anchor_top)
        if self._card.isVisible():
            self._card.raise_()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched is self._canvas_page and event.type() == QEvent.Type.Resize:
            self._reposition()
        return False

    # ---- visibility ----

    def is_open(self) -> bool:
        return self._card.isVisible()

    def set_open(self, open_: bool) -> None:
        if open_:
            self._reposition()
            self._card.show()
            self._card.raise_()
        else:
            self._card.hide()

    # ---- spawn routing ----

    def _on_node_requested(self, node_id: str) -> None:
        """双击 panel 节点 → 在画布视口中心 spawn (受 canvas interactive 状态门控)."""
        page = self._canvas_page
        view = getattr(page, "view", None)
        if view is None:
            return
        # Gate spawn when the canvas is in non-interactive mode (Mission
        # Control mode). The user cannot see the spawn happen with the
        # chart overlay covering the canvas — silent success would be a
        # footgun. Return quietly with a debug log.
        if not getattr(page, "interactive", True):
            log_debug(
                f"[node_library] spawn '{node_id}' suppressed (canvas not interactive)"
            )
            return
        try:
            vp_center = view.viewport().rect().center()
            scene_pos = view.mapToScene(vp_center)
            page.spawn_node(node_id, x=scene_pos.x(), y=scene_pos.y())
        except Exception as e:
            log_warning(f"[node_library] spawn at viewport center failed: {e!r}")

    # ---- theme ----

    def apply_theme(self) -> None:
        self._card.refresh_style()

    @property
    def card(self) -> NodeLibraryCard:
        return self._card


# =============================================================================
# Backward-compat overlay (button + controller) for non-embedded canvas
# =============================================================================

class NodeLibraryOverlay(QObject):
    """协调 button + card：定位、toggle、单击放置、点击外部关闭.

    Backward-compat path used when ``CanvasPage`` is constructed standalone
    (``embedded=False``). Embedded usage skips this wrapper and assembles a
    bare :class:`NodeLibraryCardController` paired with an external button.
    """

    def __init__(self, canvas_page) -> None:
        super().__init__(canvas_page)
        self._canvas_page = canvas_page

        self._button = NodeLibraryButton(canvas_page)
        self._controller = NodeLibraryCardController(canvas_page)
        # Anchor card below the floating button (legacy behavior).
        self._controller.set_anchor_top(
            EDGE_PADDING + self._button.height() + BUTTON_CARD_GAP
        )

        self._button.toggled.connect(self._controller.set_open)

        # Reposition button on canvas resize (card is handled by controller).
        canvas_page.installEventFilter(self)
        self._reposition_button()
        self._button.raise_()
        self._button.show()

        log_debug("[ui.canvas] NodeLibraryOverlay mounted")

    # ---- positioning ----

    def _reposition_button(self) -> None:
        self._button.move(EDGE_PADDING, EDGE_PADDING)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched is self._canvas_page and event.type() == QEvent.Type.Resize:
            self._reposition_button()
        return False

    # ---- theme ----

    def apply_theme(self) -> None:
        self._controller.apply_theme()

    @property
    def controller(self) -> NodeLibraryCardController:
        return self._controller

    @property
    def button(self) -> NodeLibraryButton:
        return self._button


__all__ = [
    "NodeLibraryButton",
    "NodeLibraryCard",
    "NodeLibraryCardController",
    "NodeLibraryOverlay",
    "EDGE_PADDING",
    "BUTTON_CARD_GAP",
    "BUTTON_SIZE",
    "CARD_W",
    "CARD_H",
    "EMBEDDED_CARD_TOP_OFFSET",
]
