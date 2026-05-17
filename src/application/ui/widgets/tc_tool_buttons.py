"""application.ui.widgets.tc_tool_buttons — Training Canva 工具row 左侧按钮组.

仅 TC 模式下出现在 MissionControlPanel 工具row 左侧的 3 按钮集合:

    [icon_undo] [icon_redo] [Save 按钮]

- undo / redo:  light icon-only button — 与 CanvasPage.undo_stack 双向绑定;
                disabled 时 SVG icon 以 bg_3 着色(QIcon.Mode.Disabled 预生成)
- Save:         border+save button — enable 仅当当前 canvas 处于脏状态;
                点击调用 CanvasPage._maybe_save_after_edit 落盘

Node Library 按钮 + 面板已迁移到 Sidebar (``Sidebar.node_library``),不再属于
工具row 范畴。

本 widget 完全自包含:set_canvas(canvas) 完成所有信号挂载;canvas=None 时全部 disable。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from unitport_sdk import Assets, Config, LaviButton, i18n_bind, log_warning, setButton, tr


BUTTON_H = 30                # 工具row 内按钮统一高度(row 43px - 6*2 padding - 1 气泡)
ICON_BTN_W = 30              # undo / redo 方形纯图标按钮
ICON_SIDE = max(12, BUTTON_H - 12)  # 与 setButton 内部一致(line 184)


# ----------------------------------------------------------------------------
# SVG icon helpers
# ----------------------------------------------------------------------------

def _build_dual_mode_icon(
    svg_path: Path, side: int, disabled_color: str
) -> QIcon:
    """生成 Normal / Disabled 双模态 QIcon。

    Normal mode 使用 SVG 原始着色(QIcon(path) 直接加载);Disabled mode 把同一 SVG
    通过 ``CompositionMode_SourceIn`` 用 ``disabled_color`` 重染。Qt 在 QPushButton
    isEnabled() 翻转时自动切换 pixmap,无需手动 setIcon。
    """
    icon = QIcon(str(svg_path))
    renderer = QSvgRenderer(str(svg_path))
    pm = QPixmap(side, side)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    renderer.render(p)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(pm.rect(), QColor(disabled_color))
    p.end()
    icon.addPixmap(pm, QIcon.Mode.Disabled)
    return icon


# ----------------------------------------------------------------------------
# Cluster
# ----------------------------------------------------------------------------

class TCToolButtonsCluster(QWidget):
    """Training Canva 工具row 左侧 4 按钮集合。

    生命周期:
        __init__ 构造 3 按钮(无 canvas 绑定时全部 disable);
        set_canvas(canvas) 挂全部信号 + 同步初始状态;
        set_canvas(None) 解绑 + disable;
        apply_theme() 主题切换时重新生成 disabled 着色 pixmap + 刷 QSS。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._canvas = None  # type: ignore[assignment]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. undo —— light kind, icon only, hover-only effect
        self._undo_btn: LaviButton = setButton(
            "missioncontrol.btn.undo",
            ICON_BTN_W, BUTTON_H,
            kind="light",
            icon="icon_undo",
            default="",
            icon_only=True,
            disabled=True,
        )
        i18n_bind(self._undo_btn, "setToolTip",
                  "missioncontrol.btn.undo.tip", "Undo (Ctrl+Z)")
        layout.addWidget(self._undo_btn)

        # 2. redo
        self._redo_btn: LaviButton = setButton(
            "missioncontrol.btn.redo",
            ICON_BTN_W, BUTTON_H,
            kind="light",
            icon="icon_redo",
            default="",
            icon_only=True,
            disabled=True,
        )
        self._redo_btn.setToolTip(
            tr("missioncontrol.btn.redo.tip", "Redo (Ctrl+Shift+Z)")
        )
        layout.addWidget(self._redo_btn)

        # 3. Save —— border + save(safe 绿). 宽度走 sizeHint 拟合 "Save" 文本。
        # Disabled 状态下 border 改用与 disabled fg 同色(见 _patch_save_disabled_qss)
        # —— 默认 LaviButton 的 disabled 只换文字色,保留 spec 着色 border 显得违和。
        self._save_btn: LaviButton = setButton(
            "missioncontrol.btn.save",
            0, BUTTON_H,                     # width=0 → 跳过 setFixedSize, 宽度自适应
            kind="border", spec="save",
            default="Save",
            disabled=True,
        )
        self._save_btn.setFixedHeight(BUTTON_H)
        i18n_bind(self._save_btn, "setToolTip",
                  "missioncontrol.btn.save.tip", "Save canvas")
        self._patch_save_disabled_qss()
        layout.addWidget(self._save_btn)

        # 立刻铺一次 undo/redo 的双模态 icon(disabled 走 bg_3 着色)
        self._apply_undo_redo_icons()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_canvas(self, canvas) -> None:
        """绑定 / 解绑 CanvasPage。重复绑同一 canvas 时安全(内部 short-circuit)。

        None → 解绑旧 canvas + 全部 disable。
        """
        if self._canvas is canvas:
            # 同 canvas 重绑:仍同步一次状态(load_from_file 之后 dirty/undo 都重置)
            self._sync_initial_state()
            return

        # 解绑旧 canvas
        old = self._canvas
        if old is not None:
            try:
                stack = old.undo_stack
                stack.canUndoChanged.disconnect(self._undo_btn.setEnabled)
                stack.canRedoChanged.disconnect(self._redo_btn.setEnabled)
                self._undo_btn.clicked.disconnect(stack.undo)
                self._redo_btn.clicked.disconnect(stack.redo)
                old.dirty_state_changed.disconnect(self._save_btn.setEnabled)
                self._save_btn.clicked.disconnect(old._maybe_save_after_edit)
            except (TypeError, RuntimeError):
                # 没接过 / 已被销毁:静默,Qt 不允许双断
                pass

        self._canvas = canvas
        if canvas is None:
            self._undo_btn.setEnabled(False)
            self._redo_btn.setEnabled(False)
            self._save_btn.setEnabled(False)
            return

        # 挂新 canvas 信号
        stack = canvas.undo_stack
        stack.canUndoChanged.connect(self._undo_btn.setEnabled)
        stack.canRedoChanged.connect(self._redo_btn.setEnabled)
        self._undo_btn.clicked.connect(stack.undo)
        self._redo_btn.clicked.connect(stack.redo)

        canvas.dirty_state_changed.connect(self._save_btn.setEnabled)
        self._save_btn.clicked.connect(canvas._maybe_save_after_edit)

        self._sync_initial_state()

    def apply_theme(self) -> None:
        """主题切换:刷 LaviButton QSS,并重生成 undo/redo disabled-mode pixmap。

        ``bg_3`` slot 在新主题里可能变色,所以 disabled pixmap 必须重染。
        Save 按钮的 disabled-border patch 也要在 refresh_style 之后重贴
        (refresh_style 会用 SDK 的原始 QSS 覆盖此 patch)。
        """
        for btn in (self._undo_btn, self._redo_btn, self._save_btn):
            try:
                btn.refresh_style()
            except Exception:
                pass
        self._patch_save_disabled_qss()
        self._apply_undo_redo_icons()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_initial_state(self) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        try:
            stack = canvas.undo_stack
            self._undo_btn.setEnabled(stack.canUndo())
            self._redo_btn.setEnabled(stack.canRedo())
        except Exception:
            self._undo_btn.setEnabled(False)
            self._redo_btn.setEnabled(False)
        try:
            self._save_btn.setEnabled(bool(canvas.is_dirty()))
        except Exception:
            self._save_btn.setEnabled(False)

    def _patch_save_disabled_qss(self) -> None:
        """追加 ``QPushButton:disabled { border-color: <disabled_fg>; }`` 规则。

        LaviButton 的 spec="save" 在 enable / hover / checked 时用 ``safe_zone`` 着色
        border + fg,但 disabled 状态默认只把 fg 换成 ``main_c2``,border 仍保留
        spec 绿色 —— 视觉上像 "禁用但仍在召唤注意",与一般灰显按钮的设计语言冲突。
        本 patch 把 disabled border 拉回 ``main_c2``,与 disabled fg 同色。
        """
        disabled_fg = Config.get_color("main_c2", "#999999")
        base = self._save_btn.styleSheet() or ""
        extra = f"QPushButton:disabled {{ border-color: {disabled_fg}; }}"
        self._save_btn.setStyleSheet(base + "\n" + extra)

    def _apply_undo_redo_icons(self) -> None:
        disabled_color = Config.get_color("bg_3", "#101010")
        for btn, name in ((self._undo_btn, "icon_undo"),
                          (self._redo_btn, "icon_redo")):
            svg_path = Assets.find_icon(name)
            if svg_path is None:
                log_warning(f"[tc_tool_buttons] icon not found: {name}")
                continue
            icon = _build_dual_mode_icon(svg_path, ICON_SIDE, disabled_color)
            btn.setIcon(icon)
            btn.setIconSize(QSize(ICON_SIDE, ICON_SIDE))


__all__ = ["TCToolButtonsCluster"]
