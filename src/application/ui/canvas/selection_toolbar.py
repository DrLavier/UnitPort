"""application.ui.canvas.selection_toolbar — 选区悬浮工具栏.

单实例 QWidget,parent=CanvasView.viewport(),浮在最靠上选中 item 上方 20px。

按钮(全部 LaviButton kind="light" — 无边框、透明底):
- [成组/解组]  icon_group ↔ icon_degroupe。
- [折叠/展开]  icon_fold ↔ icon_expend (选区全部已收纳→expand;其他→collapse)。
- [跳过/启用]  icon_disable ↔ icon_yes (选区全部已禁用→enable;其他→disable)。
- [颜色]       icon_color → 左键打开 ColorPickerPopup;右键清除覆写恢复默认。
- [删除]       icon_delete → 调 ``page._delete_selected()``。

**所有 mode 由 CanvasPage._reposition_selection_toolbar 权威驱动** —— 工具栏
不再本地翻转状态,clicked 仅纯 emit。
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from unitport_sdk import Config, log_debug, setButton, tr


BTN_W = 28
BTN_H = 28
PAD = 4
GAP = 2


class _ColorButton(QWidget):
    """Wrapper around a setButton-created button that re-routes left/right
    clicks to two distinct signals.

    We can't just connect to ``button.clicked`` because we need to discriminate
    Qt.LeftButton (= open color picker) from Qt.RightButton (= clear override).
    Instead we intercept ``mousePressEvent`` on the SDK button instance via a
    light forwarding wrapper that exposes two signals.
    """

    color_left_clicked = pyqtSignal(QPoint)   # button bottom-left in global coords
    color_right_clicked = pyqtSignal()

    def __init__(self, btn: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._btn = btn
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(btn)
        btn.installEventFilter(self)
        self.setFixedSize(btn.sizeHint())

    def eventFilter(self, obj, event):  # noqa: N802 — Qt signature
        if obj is self._btn and event.type() == QMouseEvent.Type.MouseButtonPress:
            btn = event.button()
            if btn == Qt.MouseButton.LeftButton:
                # Anchor = bottom-left of the SDK button in global coords so the
                # popup hangs just below the toolbar.
                anchor = self._btn.mapToGlobal(self._btn.rect().bottomLeft())
                self.color_left_clicked.emit(anchor)
                return True
            if btn == Qt.MouseButton.RightButton:
                self.color_right_clicked.emit()
                return True
        return super().eventFilter(obj, event)

    def button(self) -> QWidget:
        return self._btn


class SelectionToolbar(QWidget):
    """选区悬浮工具栏 —— 单实例 by CanvasPage."""

    group_clicked = pyqtSignal()
    collapse_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()
    color_clicked = pyqtSignal(QPoint)
    color_reset_clicked = pyqtSignal()
    delete_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("selection_toolbar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # 权威 mode —— 由 CanvasPage 驱动,工具栏不本地翻转。
        self._group_mode: str = "group"      # "group" | "ungroup"
        self._collapse_mode: str = "collapse"  # "collapse" | "expand"
        self._disable_mode: str = "disable"    # "disable" | "enable"

        self._apply_theme()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(PAD, PAD, PAD, PAD)
        layout.setSpacing(GAP)

        # 4 个按钮均走 LaviButton 的 light 样式(透明底 + 无 border),icon-only。
        self._btn_group = setButton(
            "canvas.toolbar.group", BTN_W, BTN_H,
            kind="light", spec="none", icon="icon_group",
            default="Group", icon_only=True,
        )
        self._btn_collapse = setButton(
            "canvas.toolbar.collapse", BTN_W, BTN_H,
            kind="light", spec="none", icon="icon_fold",
            default="Fold", icon_only=True,
        )
        self._btn_skip = setButton(
            "canvas.toolbar.skip", BTN_W, BTN_H,
            kind="light", spec="none", icon="icon_disable",
            default="Skip", icon_only=True,
        )
        # 色盘按钮 —— 左键打开 ColorPickerPopup, 右键清除覆写恢复默认。
        # 用 _ColorButton wrapper 拦截 mousePressEvent 区分左右键(setButton 的
        # clicked signal 无法直接区分);wrapper 暴露 color_left/right_clicked。
        _btn_color_inner = setButton(
            "canvas.toolbar.color", BTN_W, BTN_H,
            kind="light", spec="none", icon="icon_color",
            default="Color", icon_only=True,
        )
        self._btn_color = _ColorButton(_btn_color_inner, parent=self)
        self._btn_delete = setButton(
            "canvas.toolbar.delete", BTN_W, BTN_H,
            kind="light", spec="none", icon="icon_delete",
            default="Del", icon_only=True,
        )

        for b in (
            self._btn_group,
            self._btn_collapse,
            self._btn_skip,
            self._btn_color,
            self._btn_delete,
        ):
            layout.addWidget(b)

        # tooltips 跟 i18n key 走,选区/语言变化时由 _refresh_tooltips 刷新。
        self._refresh_tooltips()

        # clicked → 纯 emit;mode 切换全部由 CanvasPage 通过 set_*_mode 反向驱动。
        self._btn_group.clicked.connect(self.group_clicked.emit)
        self._btn_collapse.clicked.connect(self._on_collapse_clicked)
        self._btn_skip.clicked.connect(self._on_skip_clicked)
        self._btn_color.color_left_clicked.connect(self.color_clicked.emit)
        self._btn_color.color_right_clicked.connect(self.color_reset_clicked.emit)
        self._btn_delete.clicked.connect(self.delete_clicked.emit)

        self.adjustSize()
        self.hide()

    # ------------------------------------------------------------------
    # API — 权威 mode setters(由 CanvasPage._reposition_selection_toolbar 调)
    # ------------------------------------------------------------------
    def set_group_mode(self, mode: str) -> None:
        new = "ungroup" if mode == "ungroup" else "group"
        if new == self._group_mode:
            return
        self._group_mode = new
        self._set_button_icon(
            self._btn_group,
            "icon_degroupe" if new == "ungroup" else "icon_group",
        )
        self._refresh_tooltips()

    def set_collapse_mode(self, mode: str) -> None:
        """mode ∈ {"collapse", "expand"}:全部已收纳→expand;混合/全展开→collapse."""
        new = "expand" if mode == "expand" else "collapse"
        if new == self._collapse_mode:
            return
        self._collapse_mode = new
        self._set_button_icon(
            self._btn_collapse,
            "icon_expend" if new == "expand" else "icon_fold",
        )
        self._refresh_tooltips()

    def set_disable_mode(self, mode: str) -> None:
        """mode ∈ {"disable", "enable"}:全部已禁用→enable;混合/全启用→disable."""
        new = "enable" if mode == "enable" else "disable"
        if new == self._disable_mode:
            return
        self._disable_mode = new
        self._set_button_icon(
            self._btn_skip,
            "icon_yes" if new == "enable" else "icon_disable",
        )
        self._refresh_tooltips()

    def apply_theme(self) -> None:
        """主题切换时由 CanvasPage 调,刷新 QSS + tooltip(i18n 可能跟着切)。"""
        self._apply_theme()
        for b in (
            self._btn_group,
            self._btn_collapse,
            self._btn_skip,
            self._btn_color.button(),
            self._btn_delete,
        ):
            refresh = getattr(b, "refresh_style", None)
            if callable(refresh):
                refresh()
        self._refresh_tooltips()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1", "#1E1E1E")
        border = Config.get_color("border_1", "#444444")
        self.setStyleSheet(
            f"#selection_toolbar {{ background: {bg};"
            f" border: 1px solid {border}; border-radius: 6px; }}"
        )

    def _set_button_icon(self, btn, icon_stem: str) -> None:
        """重新设置按钮 icon(走 SDK Assets.find_icon → QIcon)。"""
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QIcon

        from unitport_sdk import Assets

        path = Assets.find_icon(icon_stem)
        if path is None:
            log_debug(f"[ui.canvas] selection_toolbar: icon not found '{icon_stem}'")
            return
        btn.setIcon(QIcon(str(path)))
        side = max(12, BTN_H - 12)
        btn.setIconSize(QSize(side, side))

    def _refresh_tooltips(self) -> None:
        if self._group_mode == "ungroup":
            self._btn_group.setToolTip(tr("canvas.toolbar.ungroup", "Ungroup"))
        else:
            self._btn_group.setToolTip(tr("canvas.toolbar.group", "Group"))
        if self._collapse_mode == "expand":
            self._btn_collapse.setToolTip(tr("canvas.toolbar.expand", "Expand"))
        else:
            self._btn_collapse.setToolTip(tr("canvas.toolbar.collapse", "Fold"))
        if self._disable_mode == "enable":
            self._btn_skip.setToolTip(tr("canvas.toolbar.enable", "Enable"))
        else:
            self._btn_skip.setToolTip(tr("canvas.toolbar.skip", "Disable"))
        self._btn_color.button().setToolTip(
            tr("canvas.toolbar.color", "Pick color (right-click to reset)")
        )
        self._btn_delete.setToolTip(tr("canvas.toolbar.delete", "Delete"))

    def _on_collapse_clicked(self) -> None:
        # 纯 emit;mode 切换由 page._reposition_selection_toolbar 反向同步。
        self.collapse_clicked.emit()

    def _on_skip_clicked(self) -> None:
        self.skip_clicked.emit()


__all__ = ["SelectionToolbar"]
