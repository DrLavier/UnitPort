# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ui.py — SDK 内置 Qt 控件。

整合三个原本分散的 UI 组件：

- TyperThread   — 打字机效果线程（CmdLogWidget 内部使用）
- SwitchButton  — 带渐变动画的开关
- CmdLogWidget  — 内置 cmd 屏（DEBUG / INFO 双开关 + 单行进度覆写 + 历史过滤）
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional, Tuple

from PyQt6.QtCore import (
    Qt,
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QRectF,
    QSize,
    QThread,
    QTimer,
    QUrl,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QStandardItem,
    QStandardItemModel,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .logger import (
    get_debug_flag,
    get_info_flag,
    get_log_signal,
    log_error,
    log_warning,
    set_debug_flag,
    set_info_flag,
)
from .sys import Assets, Config, I18n, get_task_signal

try:
    from shiboken6 import isValid as _qt_is_valid
except ImportError:  # pragma: no cover - PyQt6 ships shiboken6
    def _qt_is_valid(_obj) -> bool:
        return True


# =============================================================================
# i18n 绑定：让 widget 字串响应 I18n.set_language()
# =============================================================================
def i18n_bind(
    widget: QWidget,
    setter: str,
    key: str,
    default: str = "",
) -> None:
    """把 ``widget.{setter}(text)`` 与 i18n key 挂钩，语种切换时自动重译。

    Examples:
        i18n_bind(window, "setWindowTitle", "app.title", default="My App")
        i18n_bind(group, "setTitle", "settings.network", default="Network")

    立即用当前语种调用一次 setter；之后挂到 ``I18n.language_changed`` 信号上，
    每次切换语种自动重新翻译。

    生命周期：当 widget 被 Qt 销毁（C++ 端释放）时自动从信号断开，避免
    `RuntimeError: wrapped C/C++ object has been deleted`。即便 destroyed
    信号未触发（极少数 race），_update 内层也吞 RuntimeError 并自我断开，
    确保不会污染后续切换语种的 emit。
    """
    sig = I18n.instance().language_changed
    state = {"alive": True}

    def _disconnect(*_):
        # Idempotent: 多路径都可能触发（_update 抓 RuntimeError 自清；widget
        # destroyed 兜底）。重复调用会让 Qt 在 stderr 打 qWarning，所以用 flag
        # 保证 sig.disconnect 只跑一次。
        if not state["alive"]:
            return
        state["alive"] = False
        # 应用退出阶段，I18n 单例的 C++ 端可能已被 Qt 销毁；此时再 disconnect
        # 会触发 qWarning "No such signal" 直接打到 stderr（不是 Python 异常，
        # try/except 抓不住）。先用 shiboken6 验活，避免污染关闭日志。
        inst = I18n._instance
        if inst is None or not _qt_is_valid(inst):
            return
        try:
            sig.disconnect(_update)
        except (TypeError, RuntimeError):
            pass

    def _update(*_):
        if not state["alive"]:
            return
        try:
            getattr(widget, setter)(I18n.tr(key, default))
        except RuntimeError:
            # widget's C++ side already gone; tear down this slot.
            _disconnect()

    _update()
    sig.connect(_update)
    try:
        widget.destroyed.connect(_disconnect)
    except (RuntimeError, AttributeError):
        pass


class I18nLabel(QLabel):
    """自动重译的 QLabel。"""

    def __init__(self, key: str, default: str = "", parent=None):
        super().__init__(parent=parent)
        self._i18n_key = key
        self._i18n_default = default
        i18n_bind(self, "setText", key, default)


class I18nButton(QPushButton):
    """自动重译的 QPushButton。"""

    def __init__(self, key: str, default: str = "", parent=None):
        super().__init__(parent=parent)
        self._i18n_key = key
        self._i18n_default = default
        i18n_bind(self, "setText", key, default)


# =============================================================================
# LaviButton / setButton — 主题化按钮工厂
# =============================================================================
class LaviButton(I18nButton):
    """主题化按钮：kind × spec 风格 + 用户 override + i18n 文本。

    继承 I18nButton 自动获得语种切换重译；子类只负责配色与 QSS。
    主题切换后调 ``refresh_style()`` 重新应用配色。
    """

    _SPEC_COLOR_KEY = {
        "notice": "highlight",
        "save":   "safe_zone",
        "danger": "danger_zone",
    }
    _SPEC_HOVER_KEY = {
        "notice": "highlight",
        "save":   "safe_zone_hover",
        "danger": "danger_zone_hover",
    }
    _SPEC_CHECKED_KEY = {
        "notice": "highlight_checked",
        "save":   "safe_zone_checked",
        "danger": "danger_zone_checked",
    }

    def __init__(
        self,
        id: str,
        *,
        default: str = "",
        kind: str = "normal",
        spec: str = "none",
        icon: Optional[str] = None,
        overrides: Optional[dict] = None,
        checkable: bool = False,
        disabled: bool = False,
        icon_only: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        parent=None,
    ):
        if icon_only:
            # 绕过 I18nButton 的 i18n_bind：纯图标按钮不需要文本绑定
            QPushButton.__init__(self, parent=parent)
            self._i18n_key = id
            self._i18n_default = default
            self.setText("")
        else:
            super().__init__(id, default, parent)
        self._kind = kind if kind in ("normal", "border", "light") else "normal"
        self._spec = spec if spec in ("none", "notice", "save", "danger") else "none"
        self._overrides = {k: v for k, v in (overrides or {}).items() if v}

        if width and height:
            self.setFixedSize(int(width), int(height))
        if checkable:
            self.setCheckable(True)
        if icon:
            resolved = Assets.find_icon(icon)
            if resolved is not None:
                self.setIcon(QIcon(str(resolved)))
                if height:
                    side = max(12, int(height) - 12)
                    self.setIconSize(QSize(side, side))
            else:
                log_warning(f"[setButton] icon not found: {icon}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setEnabled(not disabled)

        self.refresh_style()

    # ---- 颜色解析 -------------------------------------------------------
    def _resolve_palette(self) -> dict:
        """根据 kind × spec 返回配色。overrides 逐字段覆盖。

        非 ``none`` spec：hover/checked bg 来自 spec 自己的 ``*_hover`` / ``*_checked``
        slot，hover/checked 文字色统一切到 ``main_t2``，且 baseline 字体加粗。
        """
        spec_color_key   = self._SPEC_COLOR_KEY.get(self._spec)
        spec_hover_key   = self._SPEC_HOVER_KEY.get(self._spec)
        spec_checked_key = self._SPEC_CHECKED_KEY.get(self._spec)

        spec_color   = Config.get_color(spec_color_key) if spec_color_key   else None
        spec_hover   = Config.get_color(spec_hover_key) if spec_hover_key   else None
        spec_checked = Config.get_color(spec_checked_key) if spec_checked_key else None
        has_spec = spec_color is not None

        if self._kind == "normal":
            bg = spec_color if has_spec else Config.get_color("btn_1")
            fg = (
                Config.get_color("bg_1")
                if has_spec
                else Config.get_color("main_t2")
            )
            border = "transparent"
        elif self._kind == "border":
            bg = "transparent"
            fg = spec_color if has_spec else Config.get_color("main_t1", "#D6D3C7")
            border = spec_color if has_spec else Config.get_color("border_1", "#444444")
        else:  # light
            bg = "transparent"
            fg = spec_color if has_spec else Config.get_color("main_t1", "#D6D3C7")
            border = "transparent"

        if has_spec:
            hover      = spec_hover
            hover_fg   = Config.get_color("main_t2")
            checked    = spec_checked
            checked_fg = Config.get_color("bg_1")
        else:
            hover      = Config.get_color("hover_1", "#525252")
            hover_fg   = fg                                  # 无 spec：hover 不改文字色
            checked    = Config.get_color("theme_1", "#9989E0")
            checked_fg = Config.get_color("bg_1", "#1E1E1E")

        ov = self._overrides
        return {
            "bg":         ov.get("bg", bg),
            "fg":         ov.get("fg", fg),
            "border":     ov.get("border", border),
            "hover":      ov.get("hover", hover),
            "hover_fg":   ov.get("hover_fg", hover_fg),
            "checked":    ov.get("checked", checked),
            "checked_fg": ov.get("checked_fg", checked_fg),
            "bold":       has_spec,
        }

    def _build_qss(self) -> str:
        p = self._resolve_palette()
        size = Config.get_font_size("size_small")
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        border_width = "1px" if self._kind == "border" else "0"
        weight_line = "font-weight: bold;" if p["bold"] else ""
        disabled_fg = Config.get_color("main_c2")
        # disabled 状态强制走中性灰：不论 kind × spec × overrides 如何染色，
        # 禁用态都剥离 spec 底色 / hover / checked tint，只保留 kind 的几何
        # 形状（normal 有底、border 有描边、light 透明）。
        if self._kind == "normal":
            disabled_bg = Config.get_color("row_1")
            disabled_border = "transparent"
        elif self._kind == "border":
            disabled_bg = "transparent"
            disabled_border = Config.get_color("border_2")
        else:  # light
            disabled_bg = "transparent"
            disabled_border = "transparent"
        return f"""
        QPushButton {{
            background-color: {p['bg']};
            color: {p['fg']};
            border: {border_width} solid {p['border']};
            border-radius: 6px;
            padding: 4px 12px;
            font-family: "{family}";
            font-size: {size}px;
            {weight_line}
        }}
        QPushButton:enabled:hover {{
            background-color: {p['hover']};
            color: {p['hover_fg']};
        }}
        QPushButton:enabled:checked {{
            background-color: {p['checked']};
            color: {p['checked_fg']};
        }}
        QPushButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_fg};
            border: {border_width} solid {disabled_border};
            font-weight: normal;
        }}
        """

    def refresh_style(self):
        """主题切换时调用；重新解析配色并应用 QSS。"""
        self.setStyleSheet(self._build_qss())


def setButton(
    id: str,
    width: int,
    height: int,
    *,
    kind: str = "normal",
    spec: str = "none",
    icon: Optional[str] = None,
    default: str = "",
    font_color: Optional[str] = None,
    border_color: Optional[str] = None,
    bg_color: Optional[str] = None,
    hover_color: Optional[str] = None,
    hover_fg_color: Optional[str] = None,
    checked_color: Optional[str] = None,
    checked_fg_color: Optional[str] = None,
    checkable: bool = False,
    disabled: bool = False,
    icon_only: bool = False,
    parent=None,
) -> LaviButton:
    """工厂函数：返回一只挂好 QSS、i18n 绑定、kind × spec 配色的 LaviButton。

    Args:
        id: i18n key（``I18n.tr(id, default)`` 取文本）。推荐命名 ``btn.save`` 等。
        width / height: 固定尺寸。
        kind: ``normal`` 有底色无 border；``border`` 透明底有 border；``light`` 全透明。
        spec: 语义角色 ``none / notice / save / danger``；非 ``none`` 时字体加粗，
            hover/checked bg 走对应 ``*_hover`` / ``*_checked`` slot，文字色统一切到
            ``main_t2``。
        icon: 显式图标文件路径；不传即纯文本按钮。
        default: i18n 兜底文本。
        font_color / border_color / bg_color / hover_color / hover_fg_color / checked_color
        / checked_fg_color: 逐字段覆盖配色矩阵的对应 slot。
        checkable / disabled: 直通 Qt setter。
    """
    overrides = {
        "fg":         font_color,
        "border":     border_color,
        "bg":         bg_color,
        "hover":      hover_color,
        "hover_fg":   hover_fg_color,
        "checked":    checked_color,
        "checked_fg": checked_fg_color,
    }
    return LaviButton(
        id,
        default=default,
        kind=kind,
        spec=spec,
        icon=icon,
        overrides=overrides,
        checkable=checkable,
        disabled=disabled,
        icon_only=icon_only,
        width=width,
        height=height,
        parent=parent,
    )


# =============================================================================
# LaviLabel / setText — 主题化文本工厂
# =============================================================================
class LaviLabel(I18nLabel):
    """主题化标签：kind 决定 [Theme] 颜色 slot + [Font] 字号 slot；
    无边框、透明背景；继承 I18nLabel 自动获得语种切换重译。

    传 ``id=""`` 时跳过 i18n 绑定，仅用 ``default`` 作为初始文本——
    适合动态文本（运行时通过 ``label.setText(...)`` 持续刷新）或
    无需翻译的临时字串。主题切换后调 ``refresh_style()`` 重新应用 QSS。
    """

    _KIND_COLOR_KEY = {
        "title":   "main_t1",
        "content": "main_c1",
    }
    _KIND_SIZE_KEY = {
        "title":   "size_large",
        "content": "size_normal",
    }

    def __init__(
        self,
        id: str,
        *,
        default: str = "",
        kind: str = "title",
        font: Optional[str] = None,
        color: Optional[str] = None,
        size: Optional[int] = None,
        parent=None,
    ):
        if id:
            super().__init__(id, default, parent)
        else:
            # 跳过 I18nLabel 的 i18n_bind：动态/无翻译标签
            QLabel.__init__(self, parent=parent)
            self._i18n_key = ""
            self._i18n_default = default
            self.setText(default)

        self._kind = kind if kind in ("title", "content") else "title"
        self._font_override = font
        self._color_override = color
        self._size_override = size

        self.refresh_style()

    # ---- 颜色/字号解析 -------------------------------------------------
    def _resolve_palette(self) -> dict:
        color_key = self._KIND_COLOR_KEY.get(self._kind, "main_t1")
        size_key = self._KIND_SIZE_KEY.get(self._kind, "size_large")
        fg = self._color_override or Config.get_color(color_key, "#D6D3C7")
        size = self._size_override or Config.get_font_size(size_key)
        family = self._font_override or Config.get_value(
            "Font", "family", "Microsoft YaHei"
        )
        return {"fg": fg, "size": int(size), "family": family}

    def _build_qss(self) -> str:
        p = self._resolve_palette()
        return f"""
        QLabel {{
            background-color: transparent;
            border: 0;
            color: {p['fg']};
            font-family: "{p['family']}";
            font-size: {p['size']}px;
        }}
        """

    def refresh_style(self):
        """主题切换时调用；重新解析配色/字号并应用 QSS。"""
        self.setStyleSheet(self._build_qss())


def setText(
    id: str,
    *,
    default: str = "",
    kind: str = "title",
    font: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[int] = None,
    parent=None,
) -> LaviLabel:
    """工厂函数：返回挂好 QSS、（可选）i18n 绑定的 LaviLabel。

    Args:
        id: i18n key（``I18n.tr(id, default)`` 取文本）；推荐命名 ``demo.title`` 等。
            传空串 ``""`` 跳过 i18n 绑定，仅用 ``default`` 作为初始文本（适合动态文本）。
        default: i18n 兜底文本，或 id 为空时的初始文本。
        kind: ``title`` → main_t1 + size_large；``content`` → main_c1 + size_normal。
        font: 字体族 override；不传读 ``[Font].family``。
        color: 前景色 override（hex）；不传按 kind 取。
        size: 字号像素 override；不传按 kind 取。
    """
    return LaviLabel(
        id,
        default=default,
        kind=kind,
        font=font,
        color=color,
        size=size,
        parent=parent,
    )


# =============================================================================
# LaviComboBox — 主题化下拉框（只读：i18n + highlight 强调）
# =============================================================================
class _LaviComboItemDelegate(QStyledItemDelegate):
    """Popup 行绘制：committed 行恒亮 highlight；其它行 hover 跟手。

    采用 delegate 而非 QSS ``:selected``，是因为 ``:selected`` 跟随键盘 / 鼠标
    导航，无法把高亮"钉在"已提交项上。
    """

    def __init__(self, combo: "LaviComboBox"):
        super().__init__(combo)
        self._combo = combo

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_normal = Config.get_color("bg_2", "#1E1E1E")
        bg_hover = Config.get_color("hover_1", "#525252")
        bg_committed = Config.get_color("highlight", "#F6D393")
        fg_normal = Config.get_color("main_t1", "#D6D3C7")
        fg_committed = Config.get_color("bg_1", "#1E1E1E")

        # 直接读 QComboBox.currentIndex() ——它由 Qt 在 commit（点击 / 回车）时
        # 更新，popup hover 不会改它，所以正好就是"已提交"项。曾经存在的
        # ``_committed_index`` 镜像字段会和 Qt 的 currentIndex 漂移（Qt popup 内
        # 部点击处理早于 view.clicked 触发，某些平台 view.clicked 还可能根本不
        # 发），导致高亮卡在 init 那一项。
        is_committed = index.row() == self._combo.currentIndex()
        is_hover = bool(option.state & QStyle.StateFlag.State_MouseOver)

        if is_committed:
            bg, fg = bg_committed, fg_committed
        elif is_hover:
            bg, fg = bg_hover, fg_normal
        else:
            bg, fg = bg_normal, fg_normal

        # Per-item foreground override —— 任何通过 QStandardItem.setForeground(...) 或
        # 模型 setData(idx, brush, ForegroundRole) 给行设置了显式前景色的项,
        # 优先于上面按状态计算出的 fg。配合 LaviComboBox._build_qss() 一起,
        # 让外部可以独立于"committed/hover/normal"状态给某一行做标记色。
        fg_override = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(fg_override, QBrush):
            fg = fg_override.color().name()
        elif isinstance(fg_override, QColor):
            fg = fg_override.name()

        painter.fillRect(option.rect, QColor(bg))
        painter.setPen(QColor(fg))
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        painter.drawText(
            option.rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            str(text),
        )
        painter.restore()


class LaviComboBox(QComboBox):
    """主题化下拉框（只读）：

    - 配色对齐 ``LaviButton``（btn_1 / hover_1 / border_1，圆角 6px）
    - 列表项接入 ``I18n``，语种切换自动重译
    - ``highlight=True``：主体 border 与显示文本切到 ``highlight`` 槽位

    注：原本支持的 ``editable=True`` 已撤掉——QComboBox 的 popup 容器在多种平台
    上都会从 lineEdit 抢走键盘焦点，多种规避都不彻底，故移除该模式。
    """

    def __init__(
        self,
        items: list,
        *,
        highlight: bool = False,
        i18n: bool = True,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._highlight = bool(highlight)
        self._i18n_enabled = bool(i18n)

        # 归一化 items：list[str | (key, default)] → list[(key, default)]
        self._i18n_items: list[tuple[str, str]] = []
        for it in items:
            if isinstance(it, tuple) and len(it) == 2:
                self._i18n_items.append((str(it[0]), str(it[1])))
            else:
                self._i18n_items.append((str(it), str(it)))

        self._src_model = QStandardItemModel(self)
        self.setModel(self._src_model)
        self._build_items_model()

        # popup view + delegate
        view = QListView(self)
        view.setMouseTracking(True)
        self.setView(view)
        self._delegate = _LaviComboItemDelegate(self)
        view.setItemDelegate(self._delegate)
        view.clicked.connect(self._on_view_clicked)

        if self._src_model.rowCount() > 0:
            self.setCurrentIndex(0)

        # i18n 重译；i18n=False 时跳过订阅，items 文本始终原样显示
        if self._i18n_enabled:
            I18n.instance().language_changed.connect(self._retranslate_items)

        # 当 committed 行变化时,如果该行设了 ForegroundRole, 闭合态显示文本
        # 颜色要随之更新(_build_qss 会读 _current_item_fg_color)。
        self.currentIndexChanged.connect(lambda _i: self.refresh_style())

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_style()

    # ---- 数据 / i18n -------------------------------------------------------
    def _display_text(self, key: str, default: str) -> str:
        # i18n=False：原样显示 default（适用于不可翻译的标识符，例如语言代码列表）
        return I18n.tr(key, default) if self._i18n_enabled else default

    def _build_items_model(self):
        self._src_model.clear()
        for key, default in self._i18n_items:
            item = QStandardItem(self._display_text(key, default))
            item.setData(key, Qt.ItemDataRole.UserRole)
            self._src_model.appendRow(item)

    def _retranslate_items(self, *_):
        for row, (key, default) in enumerate(self._i18n_items):
            item = self._src_model.item(row)
            if item is not None:
                item.setText(self._display_text(key, default))
        self.update()
        view = self.view()
        if view is not None and view.viewport() is not None:
            view.viewport().update()

    def currentKey(self) -> str:
        """返回当前选中项对应的 i18n key。"""
        return self.keyAt(self.currentIndex())

    def keyAt(self, index: int) -> str:
        """按行号取 i18n key。"""
        if index < 0 or index >= self._src_model.rowCount():
            return ""
        item = self._src_model.item(index)
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def setCurrentKey(self, key: str) -> bool:
        """按 i18n key 选中某项。命中返回 True，未命中保持原样并返回 False。

        和 ``setCurrentText`` 不同：display 文本会被 i18n 翻译，因此随语种变化；
        key 才是稳定标识，应当用来做编程化选中。
        """
        for i in range(self._src_model.rowCount()):
            item = self._src_model.item(i)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == key:
                self._commit(i)
                return True
        return False

    def setItems(self, items: list):
        """重置选项列表；尽量保留旧 committed 行（按 i18n key 匹配）。"""
        prev_key = self.currentKey()

        new_items: list[tuple[str, str]] = []
        for it in items:
            if isinstance(it, tuple) and len(it) == 2:
                new_items.append((str(it[0]), str(it[1])))
            else:
                new_items.append((str(it), str(it)))
        self._i18n_items = new_items

        self.blockSignals(True)
        try:
            self._build_items_model()
            new_committed = 0
            for i, (key, _) in enumerate(self._i18n_items):
                if key == prev_key:
                    new_committed = i
                    break
            if self._src_model.rowCount() > 0:
                self.setCurrentIndex(new_committed)
        finally:
            self.blockSignals(False)

    # ---- popup 选中 --------------------------------------------------------
    def _on_view_clicked(self, index):
        if not index.isValid():
            return
        self._commit(index.row())
        self.hidePopup()

    def _commit(self, row: int):
        # 选中状态完全交给 QComboBox.setCurrentIndex / currentIndex（delegate
        # 也直接读它）。这里只负责把视口刷一下，让"committed 高亮"立刻可见。
        if row < 0 or row >= self._src_model.rowCount():
            return
        self.setCurrentIndex(row)
        view = self.view()
        if view is not None and view.viewport() is not None:
            view.viewport().update()

    # ---- 主题 --------------------------------------------------------------
    def _resolve_palette(self) -> dict:
        bg = Config.get_color("bg_3", "#2A2C33")
        hover = Config.get_color("hover_1", "#525252")
        border = Config.get_color("border_1", "#444444")
        text = Config.get_color("main_t1", "#D6D3C7")
        highlight = Config.get_color("highlight", "#F6D393")
        popup_bg = Config.get_color("bg_2", "#1E1E1E")
        if self._highlight:
            border = highlight
            text = highlight
        # 闭合态显示文本如果当前 committed 项设了 ForegroundRole(配套 delegate),
        # 用该颜色作为 QComboBox 的 color,这样 popup 与闭合态显示保持一致。
        override = self._current_item_fg_color()
        if override is not None:
            text = override
        return {
            "bg": bg, "hover": hover, "border": border,
            "text": text, "popup_bg": popup_bg,
            "popup_border": Config.get_color("border_1", "#444444"),
        }

    def _current_item_fg_color(self) -> Optional[str]:
        """Return the current item's ForegroundRole color (hex), else None.

        Default `QStandardItem.foreground()` returns a `QBrush` with
        `Qt.BrushStyle.NoBrush` —— treat that as "no override".
        """
        idx = self.currentIndex()
        if idx < 0:
            return None
        item = self._src_model.item(idx)
        if item is None:
            return None
        brush = item.foreground()
        if brush.style() == Qt.BrushStyle.NoBrush:
            return None
        return brush.color().name()

    def _build_qss(self) -> str:
        p = self._resolve_palette()
        size = Config.get_font_size("size_small")
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        return f"""
        QComboBox {{
            background-color: {p['bg']};
            color: {p['text']};
            border: 1px solid {p['border']};
            border-radius: 6px;
            padding: 4px 12px;
            font-family: "{family}";
            font-size: {size}px;
        }}
        QComboBox:hover {{ background-color: {p['hover']}; }}
        QComboBox::drop-down {{ border: 0; width: 18px; }}
        QComboBox QAbstractItemView {{
            background-color: {p['popup_bg']};
            border: 1px solid {p['popup_border']};
            border-radius: 6px;
            outline: 0;
            padding: 2px;
            selection-background-color: transparent;
            selection-color: {p['text']};
        }}
        """

    def refresh_style(self):
        """主题切换时调用：重建 QSS 并触发 popup 重绘以让 delegate 用新配色。"""
        self.setStyleSheet(self._build_qss())
        view = self.view()
        if view is not None and view.viewport() is not None:
            view.viewport().update()


def setComboBox(
    items: list,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    highlight: bool = False,
    i18n: bool = True,
    parent=None,
) -> LaviComboBox:
    """工厂函数：返回挂好主题、i18n 绑定的 ``LaviComboBox``（只读）。

    Args:
        items: 选项列表。元素可为 ``str``（同时充当 i18n key 与 default 文本）
            或 ``(key, default)`` 元组。
        width / height: 可选固定尺寸；只传 ``height`` 则只锁定高度，宽度自适应。
        highlight: ``True`` 时主体 border 与显示文本切换为 ``highlight`` 色。
        i18n: ``False`` 时跳过 i18n 翻译，items 始终按 default 文本显示，并且
            不订阅 language_changed。适用于内容本身就不该被翻译的列表，例如
            语言代码（EN / FR / ZH_S）。
    """
    cb = LaviComboBox(items, highlight=highlight, i18n=i18n, parent=parent)
    if width and height:
        cb.setFixedSize(int(width), int(height))
    elif height:
        cb.setFixedHeight(int(height))
    return cb


# =============================================================================
# SettingConfirm — 确定 / 取消 双按钮组合
# =============================================================================
class SettingConfirm(QWidget):
    """横排「确定 / 取消」双图标按钮，便于外部场景化复用。

    布局：两个 ``normal`` 样式的 ``LaviButton``，分别使用 ``icon_yes.svg`` /
    ``icon_no.svg``，正方形尺寸对齐 ``size`` 高度，间隔 5px；容器透明无边。

    信号：
        - ``clicked(bool)``  — True=确定，False=取消
        - ``confirmed()``    — 点击「确定」
        - ``cancelled()``    — 点击「取消」
    """

    clicked = pyqtSignal(bool)
    confirmed = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, size: int = 20, parent=None):
        super().__init__(parent)
        side = int(size)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.btn_yes = LaviButton(
            "",
            kind="normal",
            icon="icon_yes.svg",
            icon_only=True,
            width=side,
            height=side,
            parent=self,
        )
        self.btn_no = LaviButton(
            "",
            kind="normal",
            icon="icon_no.svg",
            icon_only=True,
            width=side,
            height=side,
            parent=self,
        )

        layout.addWidget(self.btn_yes)
        layout.addWidget(self.btn_no)

        self.setFixedHeight(side)
        self.setStyleSheet("background: transparent; border: 0;")

        self.btn_yes.clicked.connect(self._on_yes)
        self.btn_no.clicked.connect(self._on_no)

    def _on_yes(self):
        self.confirmed.emit()
        self.clicked.emit(True)

    def _on_no(self):
        self.cancelled.emit()
        self.clicked.emit(False)

    def refresh_style(self):
        self.btn_yes.refresh_style()
        self.btn_no.refresh_style()


# =============================================================================
# TitleGroupBox — 主题化带标题分组框
# =============================================================================
class TitleGroupBox(QGroupBox):
    """主题化 ``QGroupBox``：

    - 标题颜色 ``main_t2``、字号 ``size_large``
    - 边框 ``border_2`` + 10px 圆角，无底色
    - 标题接入 ``I18n``，语种切换自动重译

    用法与原生 ``QGroupBox`` 一致：调用方自行 ``QVBoxLayout(group)`` 等挂内容。
    """

    def __init__(
        self,
        id: str,
        *,
        default: str = "",
        parent=None,
    ):
        super().__init__(parent=parent)
        self._i18n_key = id
        self._i18n_default = default
        if id:
            i18n_bind(self, "setTitle", id, default)
        else:
            self.setTitle(default)
        self.refresh_style()

    def _build_qss(self) -> str:
        title_color = Config.get_color("main_t2", "#D6D3C7")
        border = Config.get_color("border_2", "#444444")
        size = int(Config.get_font_size("size_large"))
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        # margin-top ≈ 标题文本高度的一半 → 边框正好从标题中间穿过（原生 QGroupBox 视感）
        # padding-top 让内部布局避开标题占据的高度
        half = max(6, size // 2)
        return f"""
        QGroupBox {{
            background-color: transparent;
            border: 1px solid {border};
            border-radius: 10px;
            margin-top: {half}px;
            padding: {half + 6}px 10px 10px 10px;
            font-family: "{family}";
            font-size: {size}px;
            color: {title_color};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 6px;
            color: {title_color};
        }}
        """

    def refresh_style(self):
        """主题切换时调用：重建 QSS。"""
        self.setStyleSheet(self._build_qss())


def setTitleGroupBox(
    id: str,
    *,
    default: str = "",
    parent=None,
) -> TitleGroupBox:
    """工厂函数：返回挂好主题、i18n 绑定的 ``TitleGroupBox``。

    Args:
        id: i18n key（``I18n.tr(id, default)`` 取标题文本）；传空串则跳过 i18n
            绑定，使用 ``default`` 作为静态标题。
        default: i18n 兜底文本，或 id 为空时的静态标题。
    """
    return TitleGroupBox(id, default=default, parent=parent)


# =============================================================================
# TyperThread — 打字机效果
# =============================================================================
class TyperThread(QThread):
    """逐字发射文本的工作线程。"""

    char_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, text: str, interval: int = 30):
        super().__init__()
        self._text = text
        self._interval = interval
        self._stop = False

    def run(self):
        for ch in self._text:
            if self._stop:
                break
            self.char_ready.emit(ch)
            self.msleep(self._interval)
        self.finished.emit()

    def stop(self):
        self._stop = True


# =============================================================================
# SwitchButton — 带渐变动画的开关
# =============================================================================
class SwitchButton(QWidget):
    """带渐变动画的开关按钮。"""

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None, checked: bool = False):
        super().__init__(parent)

        self._checked = checked
        self._slider_pos = 1.0 if checked else 0.0

        self._width = Config.get_value("SwitchButton", "width", 44, int)
        self._height = Config.get_value("SwitchButton", "height", 22, int)
        self._slider_margin = Config.get_value("SwitchButton", "slider_margin", 2, int)
        self._animation_duration = Config.get_value(
            "SwitchButton", "animation_duration", 200, int
        )

        self._off_bg = Config.get_color("btn_1", "#2A2C33")
        self._on_bg = Config.get_color("safe_zone", "#F6D393")
        self._slider_off_color = Config.get_color("sub_t1", "#A0A0A0")
        self._slider_on_color = Config.get_color("bg_1", "#1E1E1E")

        self.setFixedSize(self._width, self._height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._animation = QPropertyAnimation(self, b"sliderPos")
        self._animation.setDuration(self._animation_duration)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, animated: bool = True):
        if self._checked == checked:
            return

        self._checked = checked
        target = 1.0 if checked else 0.0

        if animated:
            self._animation.stop()
            self._animation.setStartValue(self._slider_pos)
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self._slider_pos = target
            self.update()

        self.toggled.emit(checked)

    def toggle(self):
        self.setChecked(not self._checked)

    @pyqtProperty(float)
    def sliderPos(self) -> float:
        return self._slider_pos

    @sliderPos.setter
    def sliderPos(self, value: float):
        self._slider_pos = value
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        height = rect.height()
        radius = height / 2

        off_color = QColor(self._off_bg)
        on_color = QColor(self._on_bg)
        r = int(off_color.red() + (on_color.red() - off_color.red()) * self._slider_pos)
        g = int(off_color.green() + (on_color.green() - off_color.green()) * self._slider_pos)
        b = int(off_color.blue() + (on_color.blue() - off_color.blue()) * self._slider_pos)
        bg_color = QColor(r, g, b)

        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)
        painter.fillPath(path, QBrush(bg_color))

        slider_diameter = height - 2 * self._slider_margin
        slider_x = self._slider_margin + self._slider_pos * (
            rect.width() - slider_diameter - 2 * self._slider_margin
        )
        slider_y = self._slider_margin

        slider_path = QPainterPath()
        slider_path.addEllipse(QRectF(slider_x, slider_y, slider_diameter, slider_diameter))

        off_slider = QColor(self._slider_off_color)
        on_slider = QColor(self._slider_on_color)
        sr = int(off_slider.red() + (on_slider.red() - off_slider.red()) * self._slider_pos)
        sg = int(off_slider.green() + (on_slider.green() - off_slider.green()) * self._slider_pos)
        sb = int(off_slider.blue() + (on_slider.blue() - off_slider.blue()) * self._slider_pos)
        slider_color = QColor(sr, sg, sb)

        painter.fillPath(slider_path, QBrush(slider_color))

    def set_colors(
        self,
        off_bg: Optional[str] = None,
        on_bg: Optional[str] = None,
        slider_off: Optional[str] = None,
        slider_on: Optional[str] = None,
    ):
        if off_bg:
            self._off_bg = off_bg
        if on_bg:
            self._on_bg = on_bg
        if slider_off:
            self._slider_off_color = slider_off
        if slider_on:
            self._slider_on_color = slider_on
        self.update()

    def refresh_style(self):
        self._off_bg = Config.get_color("btn_1", "#2A2C33")
        self._on_bg = Config.get_color("theme_2", "#F6D393")
        self._slider_off_color = Config.get_color("row_1", "#2A2A2A")
        self._slider_on_color = Config.get_color("bg_1", "#101010")
        self.update()


# =============================================================================
# LaviProgressBar — 标准化任务进度条 + 状态徽标
# =============================================================================
class _LaviFillBar(QWidget):
    """LaviProgressBar 内部填充条：bg_3 底色 + 状态色填充，5px 高、2px 圆角。"""

    BAR_HEIGHT = 8
    BAR_RADIUS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ratio = 0.0
        self._bg = Config.get_color("bg_3", "#101010")
        self._fg = Config.get_color("highlight", "#F6D393")
        self.setFixedHeight(self.BAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_ratio(self, ratio: float) -> None:
        self._ratio = max(0.0, min(1.0, float(ratio)))
        self.update()

    def set_state_color(self, color: str) -> None:
        self._fg = color
        self.update()

    def refresh_style(self) -> None:
        self._bg = Config.get_color("bg_3", "#101010")
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(0, 0, float(self.width()), float(self.height()))
        radius = self.BAR_RADIUS

        bg_path = QPainterPath()
        bg_path.addRoundedRect(rect, radius, radius)
        painter.fillPath(bg_path, QBrush(QColor(self._bg)))

        if self._ratio > 0.0:
            fill_w = max(1.0, rect.width() * self._ratio)
            fg_path = QPainterPath()
            fg_path.addRoundedRect(QRectF(0, 0, fill_w, rect.height()), radius, radius)
            painter.fillPath(fg_path, QBrush(QColor(self._fg)))


class LaviProgressBar(QWidget):
    """标准化任务进度条。

    布局: ``Task: [bar] NN% | current/total | Status``。

    可选挂接 ``TaskSignal`` 自动渲染指定 slot 的实时进度与状态变化
    (``progress_updated`` / ``status_changed`` / ``slot_released``)；
    亦可通过 ``set_progress`` / ``set_state`` 手动驱动。

    状态: ``IDLE`` / ``RUNNING`` / ``PAUSE`` / ``SUCCESS`` / ``FAILED`` / ``CANCELLED``。
    填充色: 运行 ``highlight``、成功 ``safe_zone``、失败/取消 ``danger_zone``。
    取消状态显示 1 秒后自动回落到 ``IDLE``。
    """

    class State:
        IDLE = "idle"
        RUNNING = "running"
        PAUSE = "pause"
        SUCCESS = "success"
        FAILED = "failed"
        CANCELLED = "cancelled"

    _STATUS_I18N = {
        "idle":      ("progress.state.idle",      "Idle"),
        "running":   ("progress.state.running",   "Running"),
        "pause":     ("progress.state.pause",     "Pause"),
        "success":   ("progress.state.success",   "Success"),
        "failed":    ("progress.state.failed",    "Failed"),
        "cancelled": ("progress.state.cancelled", "Cancelled"),
    }

    _TASK_STATUS_MAP = {
        "pending":   "idle",
        "running":   "running",
        "completed": "success",
        "failed":    "failed",
        "cancelled": "cancelled",
    }

    def __init__(
        self,
        task_key: str = "",
        *,
        default_name: str = "Task",
        total: int = 100,
        slot_idx: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)

        self._task_key = task_key
        self._task_default = default_name
        self._total = max(1, int(total))
        self._ratio = 0.0
        self._current = 0
        self._state = self.State.IDLE
        self._slot: Optional[int] = None
        self._signal_conns: list = []
        self._cancel_generation = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._name_label = QLabel(self)
        self._name_label.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        if task_key:
            i18n_bind(self._name_label, "setText", task_key, default_name)
        else:
            self._name_label.setText(default_name)

        self._fill_bar = _LaviFillBar(self)

        self._percent_label = QLabel("0%", self)
        self._sep1 = QLabel("|", self)
        self._count_label = QLabel(f"0/{self._total}", self)
        self._sep2 = QLabel("|", self)
        self._status_label = QLabel("", self)

        for w in (
            self._percent_label,
            self._sep1,
            self._count_label,
            self._sep2,
            self._status_label,
        ):
            w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

        layout.addWidget(self._name_label)
        layout.addWidget(self._fill_bar, 1)
        layout.addWidget(self._percent_label)
        layout.addWidget(self._sep1)
        layout.addWidget(self._count_label)
        layout.addWidget(self._sep2)
        layout.addWidget(self._status_label)

        I18n.instance().language_changed.connect(self._refresh_status_text)

        self.refresh_style()
        self.set_state(self.State.IDLE)

        if slot_idx is not None:
            self.bind_slot(slot_idx)

    # ---- public API -----------------------------------------------------
    def set_total(self, total: int) -> None:
        self._total = max(1, int(total))
        self._current = round(self._ratio * self._total)
        self._update_count_label()

    def set_progress(
        self,
        current: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> None:
        """更新进度。``ratio`` (0-1) 或 ``current`` 任传其一即可。"""
        if ratio is None and current is None:
            return
        if ratio is not None:
            self._ratio = max(0.0, min(1.0, float(ratio)))
            self._current = round(self._ratio * self._total)
        else:
            self._current = max(0, min(self._total, int(current)))
            self._ratio = self._current / self._total if self._total else 0.0

        if self._state == self.State.IDLE and self._ratio > 0.0:
            self.set_state(self.State.RUNNING)

        self._fill_bar.set_ratio(self._ratio)
        self._update_percent_label()
        self._update_count_label()

    def set_state(self, state: str) -> None:
        if state not in self._STATUS_I18N:
            return

        self._cancel_generation += 1
        prev = self._state
        self._state = state

        if state == self.State.SUCCESS:
            self._ratio = 1.0
            self._current = self._total
        elif state == self.State.IDLE:
            self._ratio = 0.0
            self._current = 0
        elif state == self.State.RUNNING and prev in (
            self.State.IDLE,
            self.State.SUCCESS,
            self.State.FAILED,
            self.State.CANCELLED,
        ):
            self._ratio = 0.0
            self._current = 0

        self._fill_bar.set_state_color(self._state_color())
        self._fill_bar.set_ratio(self._ratio)
        self._update_percent_label()
        self._update_count_label()
        self._refresh_status_text()

        if state == self.State.CANCELLED:
            gen = self._cancel_generation
            QTimer.singleShot(1000, lambda g=gen: self._auto_reset_cancelled(g))

    def reset(self) -> None:
        self.set_state(self.State.IDLE)

    def set_task_name(self, key: str, default: str = "") -> None:
        self._task_key = key
        if default:
            self._task_default = default
        if key:
            i18n_bind(self._name_label, "setText", key, self._task_default)
        else:
            self._name_label.setText(self._task_default)

    def bind_slot(self, slot_idx: Optional[int]) -> None:
        for sig, cb in self._signal_conns:
            try:
                sig.disconnect(cb)
            except (TypeError, RuntimeError):
                pass
        self._signal_conns.clear()

        self._slot = slot_idx
        if slot_idx is None:
            return

        ts = get_task_signal()
        ts.progress_updated.connect(self._on_progress_updated)
        ts.status_changed.connect(self._on_status_changed)
        ts.slot_released.connect(self._on_slot_released)
        self._signal_conns = [
            (ts.progress_updated, self._on_progress_updated),
            (ts.status_changed, self._on_status_changed),
            (ts.slot_released, self._on_slot_released),
        ]

    def refresh_style(self) -> None:
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        size = Config.get_font_size("size_small", 12)
        text_color = Config.get_color("main_c1", "#D6D3C7")
        sep_color = Config.get_color("sub_t2", "#777777")

        text_qss = (
            f'QLabel {{ background: transparent; color: {text_color}; '
            f'font-family: "{family}"; font-size: {size}px; }}'
        )
        sep_qss = (
            f'QLabel {{ background: transparent; color: {sep_color}; '
            f'font-family: "{family}"; font-size: {size}px; }}'
        )
        for w in (self._name_label, self._percent_label, self._count_label):
            w.setStyleSheet(text_qss)
        for w in (self._sep1, self._sep2):
            w.setStyleSheet(sep_qss)

        self._fill_bar.refresh_style()
        self._fill_bar.set_state_color(self._state_color())
        self._refresh_status_text()

    # ---- TaskSignal slots ----------------------------------------------
    def _on_progress_updated(self, slot_idx: int, ratio: float, _text: str) -> None:
        if slot_idx == self._slot:
            self.set_progress(ratio=ratio)

    def _on_status_changed(self, slot_idx: int, _task_id: str, status: str) -> None:
        if slot_idx != self._slot:
            return
        mapped = self._TASK_STATUS_MAP.get(status)
        if mapped:
            self.set_state(mapped)

    def _on_slot_released(self, slot_idx: int) -> None:
        if slot_idx == self._slot and self._state == self.State.RUNNING:
            self.reset()

    # ---- internal helpers ----------------------------------------------
    def _auto_reset_cancelled(self, gen: int) -> None:
        if gen == self._cancel_generation and self._state == self.State.CANCELLED:
            self.reset()

    def _state_color(self) -> str:
        s = self._state
        if s == self.State.RUNNING:
            return Config.get_color("highlight", "#F6D393")
        if s == self.State.SUCCESS:
            return Config.get_color("safe_zone", "#36E38E")
        if s in (self.State.FAILED, self.State.CANCELLED):
            return Config.get_color("danger_zone", "#FF6B6B")
        if s == self.State.PAUSE:
            return Config.get_color("sub_t1", "#A0A0A0")
        return Config.get_color("sub_t2", "#777777")

    def _update_percent_label(self) -> None:
        # Floor (not round) so we never advertise "100%" before the run is
        # actually complete — e.g. step 999/1000 (ratio 0.999) would round up
        # to 100% and mislead the user. 100% is shown only when ratio reaches
        # 1.0, which set_state(SUCCESS) sets explicitly on completion.
        if self._ratio >= 1.0:
            pct = 100
        else:
            pct = min(99, int(self._ratio * 100))
        self._percent_label.setText(f"{pct}%")

    def _update_count_label(self) -> None:
        self._count_label.setText(f"{self._current}/{self._total}")

    def _refresh_status_text(self) -> None:
        key, default = self._STATUS_I18N.get(
            self._state, self._STATUS_I18N[self.State.IDLE]
        )
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        size = Config.get_font_size("size_small", 12)
        color = self._state_color()
        self._status_label.setText(I18n.tr(key, default))
        self._status_label.setStyleSheet(
            f'QLabel {{ background: transparent; color: {color}; '
            f'font-family: "{family}"; font-size: {size}px; }}'
        )


# =============================================================================
# CmdLogWidget — 内置 cmd 屏
# =============================================================================
class CmdLogWidget(QWidget):
    """带 DEBUG/INFO 双开关、单行进度覆写、历史过滤的 cmd 日志屏。"""

    HISTORY_LIMIT = 5000  # 切换开关时重渲染的最大历史条数

    @staticmethod
    def _level_color(log_type: str) -> str:
        # 颜色统一从 system.ini[Theme] 的 log_* 槽位读取，禁止在此处硬编码 hex。
        slot = f"log_{log_type}" if log_type else "log_info"
        return Config.get_color(slot)

    @staticmethod
    def _timestamp_color() -> str:
        return Config.get_color("log_time")

    def __init__(self, parent=None):
        super().__init__(parent)

        self._status_start_pos: Optional[int] = None
        self._typer_thread: Optional[TyperThread] = None
        # 持久日志历史（仅 wrap=True；wrap=False 进度条天然短暂不入史）
        self._history: list[tuple[str, str, bool]] = []

        self._init_ui()
        self._init_debug_switch()
        self._init_info_switch()
        get_log_signal().log_message.connect(self._on_log)

    # ---- UI 构造 -------------------------------------------------------
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(10, 5, 10, 5)

        # self.title_label = LaviLabel(
            # "console.title",
            # default="Console",
            # kind="title",
            # color=Config.get_color("highlight", "#F6D393"),
        # )
        # hl.addWidget(self.title_label)
        # hl.addSpacing(20)
        hl.addWidget(I18nLabel("console.debug_label", default="DEBUG: "))
        self.sw_debug = SwitchButton(checked=False)
        hl.addWidget(self.sw_debug)
        hl.addSpacing(15)
        hl.addWidget(I18nLabel("console.info_label", default="INFO: "))
        self.sw_info = SwitchButton(checked=True)
        hl.addWidget(self.sw_info)
        hl.addStretch()

        self.clear_btn = setButton(
            "console.clear_btn", 72, 28,
            kind="border", spec="danger",
            default="Clear",
        )
        self.clear_btn.clicked.connect(self.clear)
        hl.addWidget(self.clear_btn)

        layout.addWidget(header)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self._apply_style()

    def _init_debug_switch(self):
        try:
            debug_flag = Config.get_value("System", "cmd_log_debug", False, bool)
            self.sw_debug.setChecked(bool(debug_flag), animated=False)
            set_debug_flag(bool(debug_flag))
            self.sw_debug.toggled.connect(self._on_debug_switch_changed)
        except Exception as e:
            log_error(f"[CmdLogWidget] DEBUG config initialization failed: {e}")

    def _on_debug_switch_changed(self, checked: bool):
        try:
            set_debug_flag(checked)
            Config.set_value("System", "cmd_log_debug", str(bool(checked)).lower())
            self._rerender_history()
            log_warning(f"DEBUG mode state: {checked}")
        except Exception as e:
            log_error(f"[CmdLogWidget] DEBUG config save failed: {e}")

    def _init_info_switch(self):
        try:
            info_flag = Config.get_value("System", "cmd_log_info", True, bool)
            self.sw_info.setChecked(bool(info_flag), animated=False)
            set_info_flag(bool(info_flag))
            self.sw_info.toggled.connect(self._on_info_switch_changed)
        except Exception as e:
            log_error(f"[CmdLogWidget] INFO config initialization failed: {e}")

    def _on_info_switch_changed(self, checked: bool):
        try:
            set_info_flag(checked)
            Config.set_value("System", "cmd_log_info", str(bool(checked)).lower())
            self._rerender_history()
            log_warning(f"INFO mode state: {checked}")
        except Exception as e:
            log_error(f"[CmdLogWidget] INFO config save failed: {e}")

    # ---- 过滤与历史 ----------------------------------------------------
    def _is_filtered(self, log_type: str) -> bool:
        if log_type == "debug" and not get_debug_flag():
            return True
        if log_type == "info" and not get_info_flag():
            return True
        return False

    def _rerender_history(self):
        """根据当前 DEBUG/INFO 过滤状态重新渲染整个 cmd 屏。"""
        self.text_edit.clear()
        self._status_start_pos = None
        for text, log_type, _typer in self._history:
            if self._is_filtered(log_type):
                continue
            self._append_log(text, log_type, wrap=True)

    # ---- 日志槽 --------------------------------------------------------
    def _on_log(self, text, log_type, wrap, typer):
        # 1. 仅 wrap=True 进入历史
        if wrap:
            self._history.append((text, log_type, bool(typer)))
            if len(self._history) > self.HISTORY_LIMIT:
                drop = self.HISTORY_LIMIT // 4
                self._history = self._history[drop:]

        # 2. 应用过滤
        if self._is_filtered(log_type):
            return

        # 3. 渲染
        if typer:
            self._start_typer(text, log_type, wrap)
        else:
            self._append_log(text, log_type, wrap)

    # ---- 渲染细节 ------------------------------------------------------
    def _clear_status_line(self):
        if self._status_start_pos is None:
            return
        cursor = self.text_edit.textCursor()
        cursor.setPosition(self._status_start_pos)
        cursor.movePosition(
            QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
        self._status_start_pos = None

    def _append_log(self, text, log_type, wrap):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        ts_fmt = QTextCharFormat()
        ts_fmt.setForeground(QColor(self._timestamp_color()))
        body_fmt = QTextCharFormat()
        body_fmt.setForeground(QColor(self._level_color(log_type)))

        ts = datetime.now().strftime("%H:%M:%S")
        ts_part = f"[{ts}] "
        body_part = str(text) + ("\n" if wrap else "")

        self._clear_status_line()
        if not wrap:
            self._status_start_pos = cursor.position()

        cursor.setCharFormat(ts_fmt)
        cursor.insertText(ts_part)
        cursor.setCharFormat(body_fmt)
        cursor.insertText(body_part)

        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _start_typer(self, text, log_type, wrap):
        if self._typer_thread and self._typer_thread.isRunning():
            self._typer_thread.stop()
            self._typer_thread.wait()

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        ts_fmt = QTextCharFormat()
        ts_fmt.setForeground(QColor(self._timestamp_color()))
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._level_color(log_type)))

        ts = datetime.now().strftime("%H:%M:%S")
        header = f"[{ts}] "

        self._clear_status_line()
        self._status_start_pos = cursor.position() if not wrap else None

        cursor.setCharFormat(ts_fmt)
        cursor.insertText(header)
        self.text_edit.setTextCursor(cursor)

        self._typer_thread = TyperThread(text)
        self._typer_thread.char_ready.connect(lambda ch: self._append_char(ch, fmt))
        self._typer_thread.finished.connect(lambda: self._on_typer_finished(wrap))
        self._typer_thread.start()

    def _append_char(self, ch, fmt):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.setCharFormat(fmt)
        cursor.insertText(ch)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _on_typer_finished(self, wrap):
        if wrap:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n")
            self.text_edit.setTextCursor(cursor)
            self._status_start_pos = None

    # ---- 杂项 ----------------------------------------------------------
    def clear(self):
        self.text_edit.clear()
        self._status_start_pos = None
        self._history.clear()

    def _apply_style(self):
        # 仅设 QTextEdit + 普通 QLabel；LaviLabel / LaviButton 自带更高优先级 QSS
        cmd_bg = Config.get_color("spec_cmd", "#000000")
        border = Config.get_color("border_1", "#444444")
        text_primary = Config.get_color("main_c1", "#D6D3C7")
        size_small = Config.get_font_size("size_small")

        self.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {cmd_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                font-family: Consolas, Monaco, monospace;
                font-size: {size_small}px;
            }}
            QLabel {{
                color: {text_primary};
                font-family: Consolas, Monaco, monospace;
                font-size: {size_small}px;
            }}
            """
        )

    def refresh_style(self):
        self._apply_style()
        # title 颜色随主题刷新（沿用 highlight slot）
        # self.title_label._color_override = Config.get_color("highlight", "#F6D393")
        # self.title_label.refresh_style()
        self.clear_btn.refresh_style()
        self.sw_debug.refresh_style()
        self.sw_info.refresh_style()


# =============================================================================
# WheelSelector — 半圆转轮选择器（垂直）
# =============================================================================
class WheelSelector(QWidget):
    """转轮式选择器：垂直半圆布局，选中项居中放大；
    上下逐级缩放 + 虚化 + hover 高亮。

    items 接受三种形态（按是否启用 i18n 解读）：

    - ``str`` —— 单行；``i18n=True`` 视为 i18n key（default = key 自身），
      ``i18n=False`` 直接当显示文本。
    - ``(a, b)`` —— ``i18n=True`` 解读为 ``(key, default)`` 单行；
      ``i18n=False`` 解读为 ``(main, sub)`` 两行（向后兼容旧用法）。
    - ``[main_spec, sub_spec]`` —— **两行**，每个元素再按 str / (key, default)
      规则解读（``i18n=True`` 模式下需要两行时用此 list 形态以避免歧义）。

    信号 ``currentIndexChanged(int, str)`` / ``item_activated(int, str)`` 的
    第二个参数固定为「raw key（i18n 时）」或「默认显示文本（非 i18n 时）」。
    """

    currentIndexChanged = pyqtSignal(int, str)
    item_activated      = pyqtSignal(int, str)

    def __init__(
        self,
        items: Optional[list] = None,
        *,
        visible_count: int = 5,
        radius: int = 100,
        font_size: Optional[int] = None,
        i18n: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._i18n = bool(i18n)
        self._raw_items: list = list(items or [])
        self._norm_items: list = self._normalize_items(self._raw_items)
        self._current_index: int = 0
        self._offset: float = 0.0
        self._hover_index: int = -1

        self.visible_count = visible_count
        self.radius        = radius
        self._custom_font_size = font_size

        self.setMinimumWidth(120)
        self.setMinimumHeight(120)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)

        self._anim = QPropertyAnimation(self, b"offset")
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._angle_step = math.pi / max(1, self.visible_count)

        self._pending_steps  = 0
        self._anim_running   = False

        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(200)
        self._emit_timer.timeout.connect(self._emit_current_index)

        self._slip_effect = None
        self._beep_effect = None

        self.apply_theme()

        if self._i18n:
            I18n.instance().language_changed.connect(self._on_language_changed)

        if self._norm_items:
            QTimer.singleShot(0, self._emit_current_index)

    # ── Public API ──────────────────────────────────────────────────────────
    def setItems(self, items: list) -> None:
        self._raw_items = list(items or [])
        self._norm_items = self._normalize_items(self._raw_items)
        self._current_index = min(self._current_index, max(0, len(self._norm_items) - 1))
        self._recalc_item_height()
        self.update()

    def items(self) -> list:
        return list(self._raw_items)

    def currentIndex(self) -> int:
        return self._current_index

    def currentText(self) -> str:
        """返回当前项的「raw 主串」：i18n 模式下为 key，否则为默认显示文本。"""
        if not self._norm_items:
            return ""
        return self._norm_items[self._current_index]["main_emit"]

    def currentDisplay(self) -> str:
        """返回当前项的「翻译后主串」（用于显示场景）。"""
        if not self._norm_items:
            return ""
        n = self._norm_items[self._current_index]
        return self._render(n["main_key"], n["main_default"])

    def selectNext(self) -> None: self._enqueue_step(1)
    def selectPrev(self) -> None: self._enqueue_step(-1)

    def setCurrentIndex(self, index: int) -> None:
        if not self._norm_items:
            return
        index = max(0, min(index, len(self._norm_items) - 1))
        diff  = index - self._current_index
        if diff:
            self._enqueue_step(diff)

    # ── Item normalization ─────────────────────────────────────────────────
    def _normalize_items(self, items: list) -> list:
        """归一化为 ``[{is_two_line, main_key, main_default, sub_key, sub_default,
        main_emit}]``。详细规则见类 docstring。"""
        out = []
        for it in items:
            if isinstance(it, list) and len(it) == 2:
                # 显式两行：[main_spec, sub_spec]
                mk, md = self._spec_to_key_default(it[0])
                sk, sd = self._spec_to_key_default(it[1])
                out.append({
                    "is_two_line": True,
                    "main_key": mk, "main_default": md,
                    "sub_key": sk, "sub_default": sd,
                    "main_emit": mk if self._i18n else md,
                })
            elif isinstance(it, tuple) and len(it) == 2:
                if self._i18n:
                    # (key, default) 单行
                    k, d = str(it[0]), str(it[1])
                    out.append({
                        "is_two_line": False,
                        "main_key": k, "main_default": d,
                        "sub_key": "", "sub_default": "",
                        "main_emit": k,
                    })
                else:
                    # (main, sub) 两行（旧用法）
                    m, s = str(it[0]), str(it[1])
                    out.append({
                        "is_two_line": True,
                        "main_key": "", "main_default": m,
                        "sub_key": "",  "sub_default": s,
                        "main_emit": m,
                    })
            else:
                s = str(it)
                if self._i18n:
                    out.append({
                        "is_two_line": False,
                        "main_key": s, "main_default": s,
                        "sub_key": "", "sub_default": "",
                        "main_emit": s,
                    })
                else:
                    out.append({
                        "is_two_line": False,
                        "main_key": "", "main_default": s,
                        "sub_key": "", "sub_default": "",
                        "main_emit": s,
                    })
        return out

    @staticmethod
    def _spec_to_key_default(spec) -> tuple:
        if isinstance(spec, tuple) and len(spec) == 2:
            return str(spec[0]), str(spec[1])
        s = str(spec)
        return s, s

    # ── Animation property ──────────────────────────────────────────────────
    def _get_offset(self) -> float: return self._offset
    def _set_offset(self, v: float) -> None:
        self._offset = max(-1.5, min(1.5, v))
        self.update()

    offset = pyqtProperty(float, _get_offset, _set_offset)

    # ── Internal animation ──────────────────────────────────────────────────
    def _enqueue_step(self, step: int) -> None:
        if not self._norm_items:
            return
        self._pending_steps += step
        self._pending_steps = max(
            -self._current_index,
            min(self._pending_steps, len(self._norm_items) - 1 - self._current_index),
        )
        if not self._anim_running:
            self._dequeue_and_animate()

    def _dequeue_and_animate(self) -> None:
        if self._pending_steps == 0:
            self._anim_running = False
            return
        step = 1 if self._pending_steps > 0 else -1
        self._pending_steps -= step
        target = max(0, min(self._current_index + step, len(self._norm_items) - 1))
        self._anim_running = True
        self._play_sound("SLIP")
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(self._offset - step)
        try:
            self._anim.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._anim.finished.connect(lambda idx=target: self._commit_and_continue(idx))
        self._anim.start()

    def _commit_and_continue(self, index: int) -> None:
        self._offset = 0.0
        self._current_index = index
        self.update()
        self._emit_timer.start()
        self._dequeue_and_animate()

    def _emit_current_index(self) -> None:
        if self._norm_items:
            self.currentIndexChanged.emit(self._current_index, self.currentText())

    def _on_language_changed(self, *_):
        self._recalc_item_height()
        self.update()

    # ── Sound（QtMultimedia 可选；缺失时静音） ───────────────────────────────
    def _play_sound(self, key: str) -> None:
        try:
            from PyQt6.QtMultimedia import QSoundEffect  # noqa: WPS433 — lazy import
        except ImportError:
            return
        try:
            rel = "sound/slip.wav" if key == "SLIP" else "sound/beep.wav"
            path = Assets.find(rel)
            if path is None:
                return
            if key == "SLIP":
                if self._slip_effect is None:
                    self._slip_effect = QSoundEffect(self)
                    self._slip_effect.setSource(QUrl.fromLocalFile(str(path)))
                    self._slip_effect.setVolume(0.45)
                self._slip_effect.play()
            elif key == "BEEP":
                if self._beep_effect is None:
                    self._beep_effect = QSoundEffect(self)
                    self._beep_effect.setSource(QUrl.fromLocalFile(str(path)))
                    self._beep_effect.setVolume(0.55)
                self._beep_effect.play()
        except Exception:  # noqa: BLE001 — sound failures must never propagate
            pass

    # ── Input ────────────────────────────────────────────────────────────────
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self._enqueue_step(-1 if delta > 0 else 1)
        event.accept()

    def mouseMoveEvent(self, event):
        new = self._item_at_pos(event.pos().y())
        if new != self._hover_index:
            self._hover_index = new
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if new >= 0
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def leaveEvent(self, event):
        if self._hover_index != -1:
            self._hover_index = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        idx = self._item_at_pos(event.pos().y())
        if idx < 0:
            return
        if idx == self._current_index:
            self._play_sound("BEEP")
            self.item_activated.emit(idx, self._norm_items[idx]["main_emit"])
        else:
            self.setCurrentIndex(idx)

    def _item_at_pos(self, mouse_y: int) -> int:
        cy       = self.height() / 2
        hit_half = max(self._center_item_height / 2, 22)
        for i in range(-(self.visible_count // 2), self.visible_count // 2 + 1):
            idx = self._current_index + i
            if not (0 <= idx < len(self._norm_items)):
                continue
            theta = (i + self._offset) * self._angle_step
            if abs(theta) > math.pi / 2:
                continue
            y = cy + math.sin(theta) * self.radius
            if abs(mouse_y - y) <= hit_half:
                return idx
        return -1

    # ── Painting ─────────────────────────────────────────────────────────────
    def _render(self, key: str, default: str) -> str:
        """根据 i18n 模式产出一段显示文本。优先 ``key``（i18n 模式且 key 非空）→
        ``default`` → ``key``。"""
        if self._i18n and key:
            return I18n.tr(key, default or key)
        return default or key

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        bg = QColor(self._bg_color) if self._bg_color != "transparent" else None
        if bg is not None and bg.alpha() > 0:
            painter.setBrush(bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect())

        if not self._norm_items:
            return

        cy, cx = h / 2, w / 2
        item_half_w = cx - 12
        h_half = self._center_item_height / 2

        # 中心 hover 高亮：只在 hover 当前项时绘制；用半透明 highlight 防止盖住文字
        if self._hover_index == self._current_index:
            cr = QRectF(cx - item_half_w, cy - h_half,
                        item_half_w * 2, self._center_item_height)
            hb = QColor(self._hover_bg)
            hb.setAlpha(110)
            painter.setBrush(hb)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(cr, self._center_item_radius, self._center_item_radius)

        for i in range(-(self.visible_count // 2), self.visible_count // 2 + 1):
            idx = self._current_index + i
            if 0 <= idx < len(self._norm_items):
                self._draw_item(painter, idx, i + self._offset, cx, cy, item_half_w)

    def _draw_item(self, painter, index, offset, cx, cy, item_half_w):
        theta = offset * self._angle_step
        if abs(theta) > math.pi / 2:
            return
        y       = cy + math.sin(theta) * self.radius
        scale   = max(0.65, 1.0 - abs(offset) * 0.18)
        opacity = max(0.20, 1.0 - abs(offset) * 0.28)
        is_current = (index == self._current_index and abs(offset) < 0.01)
        is_hovered = (index == self._hover_index)
        h_half     = self._center_item_height / 2

        painter.save()
        painter.translate(cx, y)
        painter.scale(scale, scale)
        painter.setOpacity(opacity)

        if is_hovered and not is_current:
            painter.save()
            hb = QColor(self._hover_bg)
            hb.setAlpha(60)
            painter.setBrush(QBrush(hb))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                QRectF(-item_half_w, -h_half, item_half_w * 2, self._center_item_height),
                self._center_item_radius, self._center_item_radius,
            )
            painter.restore()

        n = self._norm_items[index]
        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size

        # 文字配色：当前项 hover 时用 bg_1（深色）以避免与黄底撞色
        if is_current and is_hovered:
            color = QColor(self._text_on_accent)
        elif is_current:
            color = QColor(self._text_main)
        elif is_hovered:
            color = QColor(self._text_main)
            color.setAlpha(190)
        else:
            color = QColor(self._text_sec)

        text_rect = QRectF(-item_half_w - 10, -h_half,
                           (item_half_w + 10) * 2, self._center_item_height)

        if n["is_two_line"]:
            main_text = self._render(n["main_key"], n["main_default"])
            sub_text  = self._render(n["sub_key"],  n["sub_default"])
            main_font = QFont(self.font())
            main_font.setBold(is_current)
            main_font.setPointSize(max(1, int(fs)))
            main_h = QFontMetrics(main_font).height()
            sub_font = QFont(self.font())
            sub_font.setBold(False)
            sub_font.setPointSize(max(1, int(fs) - 2))
            sub_h = QFontMetrics(sub_font).height()

            total_h = main_h + sub_h + 2
            top_y = -total_h / 2

            painter.setFont(main_font)
            painter.setPen(color)
            main_rect = QRectF(text_rect.left(), top_y, text_rect.width(), main_h)
            painter.drawText(main_rect, Qt.AlignmentFlag.AlignCenter, main_text)

            sub_color = QColor(color)
            sub_color.setAlpha(max(80, color.alpha() - 60))
            painter.setFont(sub_font)
            painter.setPen(sub_color)
            sub_rect = QRectF(text_rect.left(), top_y + main_h + 2,
                              text_rect.width(), sub_h)
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, sub_text)
        else:
            font = QFont(self.font())
            font.setBold(is_current)
            font.setPointSize(max(1, int(fs)))
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(
                text_rect, Qt.AlignmentFlag.AlignCenter,
                self._render(n["main_key"], n["main_default"]),
            )

        painter.restore()

    # ── Theme ────────────────────────────────────────────────────────────────
    def apply_theme(self) -> None:
        self._text_main      = Config.get_color("main_t1",    "#D6D3C7")
        self._text_sec       = Config.get_color("sub_t1",     "#A0A0A0")
        self._text_on_accent = Config.get_color("bg_1",     "#1E1E1E")
        self._hover_bg       = Config.get_color("highlight", "#F6D393")
        self._bg_color       = "transparent"
        self._theme_font_size = Config.get_font_size("size_normal")
        self._center_item_radius = 8
        self._recalc_item_height()
        self.update()

    def _recalc_item_height(self) -> None:
        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size
        has_two_line = any(n["is_two_line"] for n in self._norm_items)
        if has_two_line:
            main_f = QFont()
            main_f.setPointSize(max(1, int(fs)))
            main_f.setBold(True)
            sub_f = QFont()
            sub_f.setPointSize(max(1, int(fs) - 2))
            self._center_item_height = float(
                QFontMetrics(main_f).height()
                + QFontMetrics(sub_f).height()
                + 2 + 8
            )
        else:
            f = QFont()
            f.setPointSize(max(1, int(fs)))
            f.setBold(True)
            self._center_item_height = float(QFontMetrics(f).height() + 8)

    def refresh_style(self) -> None:
        self.apply_theme()


def setWheelSelector(
    items: Optional[list] = None,
    *,
    visible_count: int = 5,
    radius: int = 100,
    font_size: Optional[int] = None,
    i18n: bool = False,
    parent: Optional[QWidget] = None,
) -> WheelSelector:
    """工厂函数：返回挂好主题的 ``WheelSelector``。

    Args:
        items: 元素为 ``str``（单行）或 ``(main, sub)``（两行）。
        visible_count / radius: 视觉参数。
        font_size: 字号 override；不传读 ``[Font].size_normal``。
        i18n: ``True`` 时把所有显示字串当成 i18n key（``I18n.tr(s, s)``）。
    """
    return WheelSelector(
        items,
        visible_count=visible_count,
        radius=radius,
        font_size=font_size,
        i18n=i18n,
        parent=parent,
    )


# =============================================================================
# SliderSwitch — 多段独占开关
# =============================================================================
class SliderSwitch(QWidget):
    """多段开关：互斥单选，高亮药丸滑动到当前段；圆角固定。

    options 元素可为 ``str``（同时充当 i18n key + default）或 ``(key, default)``
    元组。语种切换时自动重新计算固定尺寸并重绘。

    信号：
        current_changed(int, str): 当前段变化，参数为 (index, key)
    """

    BORDER_RADIUS = 8

    current_changed = pyqtSignal(int, str)

    def __init__(
        self,
        options: List,
        *,
        parent: Optional[QWidget] = None,
        height: int = 44,
        min_segment_width: int = 84,
        font_size: Optional[int] = None,
        animation_ms: int = 180,
    ):
        super().__init__(parent)
        self.setObjectName("sliderSwitch")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        self._i18n_options: List[Tuple[str, str]] = self._normalize_options(options)
        self._current_index: int = 0 if self._i18n_options else -1
        self._slider_pos: float = float(max(0, self._current_index))
        self._hover_index: int = -1
        self._height = int(height)
        self._min_segment_width = int(min_segment_width)
        self._custom_font_size = font_size
        self._padding = 1

        self._anim = QPropertyAnimation(self, b"sliderPos")
        self._anim.setDuration(int(animation_ms))
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.apply_theme()
        self._update_fixed_size()

        I18n.instance().language_changed.connect(self._on_language_changed)

    # ── Public API ──────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_options(options) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for it in options or []:
            if isinstance(it, tuple) and len(it) == 2:
                out.append((str(it[0]), str(it[1])))
            else:
                out.append((str(it), str(it)))
        return out

    def options(self) -> List[Tuple[str, str]]:
        return list(self._i18n_options)

    def set_options(self, options: List) -> None:
        self._i18n_options = self._normalize_options(options)
        if self._i18n_options:
            self._current_index = max(0, min(self._current_index, len(self._i18n_options) - 1))
            self._slider_pos = float(self._current_index)
        else:
            self._current_index = -1
            self._slider_pos = 0.0
        self._update_fixed_size()
        self.update()

    def currentIndex(self) -> int:
        return self._current_index

    def currentKey(self) -> str:
        if not self._i18n_options or self._current_index < 0:
            return ""
        return self._i18n_options[self._current_index][0]

    def currentText(self) -> str:
        if not self._i18n_options or self._current_index < 0:
            return ""
        key, default = self._i18n_options[self._current_index]
        return I18n.tr(key, default)

    def setCurrentIndex(self, index: int, animated: bool = True, emit: bool = True) -> None:
        if not self._i18n_options:
            return
        new_idx = max(0, min(int(index), len(self._i18n_options) - 1))
        if new_idx == self._current_index:
            return
        self._current_index = new_idx
        if animated:
            self._anim.stop()
            self._anim.setStartValue(self._slider_pos)
            self._anim.setEndValue(float(new_idx))
            self._anim.start()
        else:
            self._slider_pos = float(new_idx)
            self.update()
        if emit:
            self.current_changed.emit(new_idx, self._i18n_options[new_idx][0])

    def setCurrentKey(self, key: str, animated: bool = True, emit: bool = True) -> None:
        for i, (k, _d) in enumerate(self._i18n_options):
            if k == str(key):
                self.setCurrentIndex(i, animated=animated, emit=emit)
                return

    # ── Animation property ──────────────────────────────────────────────────
    def _get_slider_pos(self) -> float: return self._slider_pos
    def _set_slider_pos(self, v: float) -> None:
        self._slider_pos = float(v)
        self.update()

    sliderPos = pyqtProperty(float, _get_slider_pos, _set_slider_pos)

    # ── Theme ───────────────────────────────────────────────────────────────
    def apply_theme(self) -> None:
        self._track_bg       = Config.get_color("btn_1",      "#2A2C33")
        self._border_color   = "transparent"
        self._inactive_text  = Config.get_color("sub_t1",     "#9CA3AF")
        self._active_pill_bg = Config.get_color("highlight", "#F6D393")
        self._active_text    = Config.get_color("bg_1",     "#1E1E1E")
        self._hover_text     = Config.get_color("main_t1",    "#E2E8F0")
        self._theme_font_size = Config.get_font_size("size_small")
        self.update()

    def refresh_style(self) -> None:
        self.apply_theme()
        self._update_fixed_size()

    # ── Sizing ──────────────────────────────────────────────────────────────
    def _make_font(self, bold: bool) -> QFont:
        font = QFont(self.font())
        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size
        font.setPointSize(max(1, int(fs)))
        font.setBold(bold)
        return font

    def _update_fixed_size(self) -> None:
        h = int(self._height)
        if not self._i18n_options:
            self.setFixedWidth(int(self._min_segment_width))
            self.setFixedHeight(h)
            return
        metrics = QFontMetrics(self._make_font(bold=True))
        chrome = 28  # 14px padding × 2
        widest = 0
        for key, default in self._i18n_options:
            text = I18n.tr(key, default)
            widest = max(widest, metrics.horizontalAdvance(text) + chrome)
        seg_w = max(self._min_segment_width, widest)
        total_w = int(seg_w * len(self._i18n_options) + self._padding * 2)
        self.setFixedWidth(total_w)
        self.setFixedHeight(h)

    def setHeight(self, h: int) -> None:
        """运行时调整高度；调用后立即重算固定尺寸。"""
        self._height = int(h)
        self._update_fixed_size()
        self.update()

    def _segment_width(self) -> float:
        if not self._i18n_options:
            return float(self.width())
        track_w = self.width() - self._padding * 2
        return track_w / max(1, len(self._i18n_options))

    def _index_at(self, x: int) -> int:
        if not self._i18n_options:
            return -1
        seg_w = self._segment_width()
        track_x = x - self._padding
        if track_x < 0:
            return 0
        idx = int(track_x // seg_w)
        return max(0, min(idx, len(self._i18n_options) - 1))

    def _on_language_changed(self, *_):
        self._update_fixed_size()
        self.update()

    # ── Input ───────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        idx = self._index_at(event.pos().x())
        if idx >= 0 and idx != self._current_index:
            self.setCurrentIndex(idx, animated=True)
        event.accept()

    def mouseMoveEvent(self, event):
        idx = self._index_at(event.pos().x())
        if idx != self._hover_index:
            self._hover_index = idx
            self.update()

    def leaveEvent(self, event):
        if self._hover_index != -1:
            self._hover_index = -1
            self.update()

    # ── Painting ────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect_f = QRectF(self.rect())
        radius = float(self.BORDER_RADIUS)

        painter.setPen(QPen(QColor(self._border_color), 1))
        painter.setBrush(QColor(self._track_bg))
        track_rect = rect_f.adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(track_rect, radius, radius)

        if not self._i18n_options:
            return

        seg_w = self._segment_width()
        pill_x = self._padding + self._slider_pos * seg_w
        pill_rect = QRectF(
            pill_x,
            self._padding,
            seg_w,
            rect_f.height() - self._padding * 2,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._active_pill_bg))
        painter.drawRoundedRect(pill_rect, radius, radius)

        for idx, (key, default) in enumerate(self._i18n_options):
            x = self._padding + idx * seg_w
            seg_rect = QRectF(x, 0, seg_w, rect_f.height())
            is_active = (idx == self._current_index)
            is_hover = (idx == self._hover_index and not is_active)
            painter.setFont(self._make_font(bold=is_active))
            if is_active:
                color = QColor(self._active_text)
            elif is_hover:
                color = QColor(self._hover_text)
            else:
                color = QColor(self._inactive_text)
            painter.setPen(color)
            painter.drawText(seg_rect, Qt.AlignmentFlag.AlignCenter, I18n.tr(key, default))


def setSliderSwitch(
    options: List,
    *,
    height: int = 32,
    min_segment_width: int = 84,
    font_size: Optional[int] = None,
    animation_ms: int = 180,
    parent: Optional[QWidget] = None,
) -> SliderSwitch:
    """工厂函数：返回挂好主题、i18n 绑定的 ``SliderSwitch``。

    Args:
        options: 元素 ``str`` 或 ``(key, default)``；i18n 自动重译。
    """
    return SliderSwitch(
        options,
        height=height,
        min_segment_width=min_segment_width,
        font_size=font_size,
        animation_ms=animation_ms,
        parent=parent,
    )


# =============================================================================
# TagInput — 标签输入框（标签内嵌于输入框）
# =============================================================================
class TagInput(QFrame):
    """标签输入框：标签 chip 内嵌于输入区，支持 input 模式（回车添加）或
    select 模式（下拉选择追加）。

    placeholder 可走 i18n（传 ``placeholder_key``）。
    """

    changed = pyqtSignal()

    def __init__(
        self,
        *,
        mode: str = "input",
        options: Optional[list] = None,
        placeholder_key: str = "",
        placeholder: str = "Type and press Enter",
        font_size: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._mode = mode
        self._options = list(options or [])
        self._placeholder_key = placeholder_key
        self._placeholder = placeholder
        self._font_size = font_size
        self._tags: List[str] = []
        self._tag_widgets: List[QWidget] = []

        self._build_ui()
        self.refresh_style()

    # ── Build ───────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setObjectName("tag_input_frame")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(32)

        self._main_layout = QHBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(4)

        self._tags_widget = QWidget()
        self._tags_widget.setObjectName("tags_container")
        self._tags_layout = QHBoxLayout(self._tags_widget)
        self._tags_layout.setContentsMargins(0, 0, 0, 0)
        self._tags_layout.setSpacing(4)
        self._tags_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._main_layout.addWidget(self._tags_widget)

        if self._mode == "input":
            self._input_widget = QLineEdit()
            self._input_widget.setObjectName("tag_input_field")
            self._input_widget.setFrame(False)
            self._input_widget.setMinimumWidth(60)
            self._input_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._input_widget.returnPressed.connect(self._on_input_enter)
            self._input_widget.installEventFilter(self)
            if self._placeholder_key:
                i18n_bind(
                    self._input_widget,
                    "setPlaceholderText",
                    self._placeholder_key,
                    self._placeholder,
                )
            else:
                self._input_widget.setPlaceholderText(self._placeholder)
        else:
            self._input_widget = QComboBox()
            self._input_widget.setObjectName("tag_combo_field")
            self._input_widget.setFrame(False)
            self._input_widget.setMinimumWidth(80)
            self._input_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._input_widget.addItem("")
            self._input_widget.addItems([str(o) for o in self._options])
            self._input_widget.currentTextChanged.connect(self._on_combo_changed)

        self._main_layout.addWidget(self._input_widget, 1)

    # ── Style ───────────────────────────────────────────────────────────────
    def _resolved_font_size(self) -> int:
        return int(self._font_size) if self._font_size else int(
            Config.get_font_size("size_normal")
        )

    def _frame_qss(self) -> str:
        border_color = Config.get_color("border_1", "#444444")
        bg_color     = Config.get_color("bg_3")
        text_color   = Config.get_color("main_t1",  "#D6D3C7")
        font_size    = self._resolved_font_size()
        return f"""
            QFrame#tag_input_frame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
            }}
            QWidget#tags_container {{
                background-color: transparent;
            }}
            QLineEdit#tag_input_field {{
                background-color: transparent;
                border: none;
                color: {text_color};
                font-size: {font_size}px;
                padding: 2px;
            }}
            QComboBox#tag_combo_field {{
                background-color: transparent;
                border: none;
                color: {text_color};
                font-size: {font_size}px;
                padding: 2px;
            }}
            QComboBox#tag_combo_field::drop-down {{
                border: none;
                width: 16px;
            }}
        """

    def _apply_chip_style(self, tag_widget: QWidget) -> None:
        chip_bg          = Config.get_color("misc_3",     "#4ec9b0")
        chip_text        = Config.get_color("bg_2",       "#1E1E1E")
        delete_hover_bg  = Config.get_color("danger_zone", "#FF6B6B")
        font_size        = self._resolved_font_size()
        tag_widget.setStyleSheet(f"""
            QFrame#tag_item {{
                background-color: {chip_bg};
                border: none;
                border-radius: 3px;
                max-height: 22px;
            }}
            QLabel#tag_label {{
                color: {chip_text};
                font-size: {font_size}px;
                font-weight: bold;
                background-color: transparent;
                border: none;
                padding: 0px;
            }}
            QPushButton#tag_delete_btn {{
                background-color: transparent;
                border: none;
                color: {chip_text};
                font-size: 12px;
                font-weight: bold;
                border-radius: 2px;
                padding: 0px;
            }}
            QPushButton#tag_delete_btn:hover {{
                background-color: {delete_hover_bg};
                color: white;
            }}
        """)

    def refresh_style(self) -> None:
        self.setStyleSheet(self._frame_qss())
        for tag_widget in self._tag_widgets:
            self._apply_chip_style(tag_widget)

    # ── Events ──────────────────────────────────────────────────────────────
    def _on_input_enter(self) -> None:
        text = self._input_widget.text().strip()
        if text and text not in self._tags:
            self._add_tag(text)
            self._input_widget.clear()

    def _on_combo_changed(self, text: str) -> None:
        if text and text not in self._tags:
            self._add_tag(text)
            QTimer.singleShot(10, lambda: self._input_widget.setCurrentIndex(0))

    def _add_tag(self, value: str) -> None:
        self._tags.append(value)
        tag_widget = self._create_tag_widget(value)
        self._tag_widgets.append(tag_widget)
        self._tags_layout.addWidget(tag_widget)
        self._tags_widget.adjustSize()
        self.updateGeometry()
        self.changed.emit()

    def _create_tag_widget(self, value: str) -> QWidget:
        tag = QFrame()
        tag.setObjectName("tag_item")
        tag.setProperty("value", value)
        tag.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(tag)
        layout.setContentsMargins(6, 2, 4, 2)
        layout.setSpacing(2)

        label = QLabel(str(value))
        label.setObjectName("tag_label")
        layout.addWidget(label)

        delete_btn = QPushButton("×")
        delete_btn.setObjectName("tag_delete_btn")
        delete_btn.setFixedSize(14, 14)
        delete_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        delete_btn.clicked.connect(lambda: self._remove_tag(tag, value))
        delete_btn.setAutoDefault(False)
        delete_btn.setDefault(False)
        layout.addWidget(delete_btn)

        self._apply_chip_style(tag)
        tag.adjustSize()
        return tag

    def _remove_tag(self, tag_widget: QWidget, value: str) -> None:
        if value in self._tags:
            self._tags.remove(value)
        if tag_widget in self._tag_widgets:
            self._tag_widgets.remove(tag_widget)
        self._tags_layout.removeWidget(tag_widget)
        tag_widget.deleteLater()
        self._tags_widget.adjustSize()
        self.updateGeometry()
        self.changed.emit()

    def eventFilter(self, obj, event):
        if self._mode == "input" and obj == self._input_widget:
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Backspace:
                    if not self._input_widget.text() and self._tags:
                        self._remove_last_tag()
                        return True
        return super().eventFilter(obj, event)

    def _remove_last_tag(self) -> None:
        if self._tag_widgets:
            self._remove_tag(self._tag_widgets[-1], self._tags[-1])

    def mousePressEvent(self, event):
        self._input_widget.setFocus()
        super().mousePressEvent(event)

    # ── Public API ──────────────────────────────────────────────────────────
    def get_values(self) -> list:
        return list(self._tags)

    def set_values(self, values: list) -> None:
        self.clear()
        for value in values:
            if value and str(value) not in self._tags:
                self._add_tag(str(value))

    def clear(self) -> None:
        self._tags.clear()
        for widget in self._tag_widgets:
            self._tags_layout.removeWidget(widget)
            widget.deleteLater()
        self._tag_widgets.clear()
        self._tags_widget.adjustSize()
        self.updateGeometry()
        self.changed.emit()

    def add_value(self, value) -> None:
        if value and str(value) not in self._tags:
            self._add_tag(str(value))

    def has_values(self) -> bool:
        return bool(self._tags)


def setTagInput(
    *,
    mode: str = "input",
    options: Optional[list] = None,
    placeholder_key: str = "",
    placeholder: str = "Type and press Enter",
    font_size: Optional[int] = None,
    parent: Optional[QWidget] = None,
) -> TagInput:
    """工厂函数：返回挂好主题与 i18n 占位符的 ``TagInput``。

    Args:
        mode: ``input`` 走文本输入回车追加；``select`` 走下拉选择追加。
        options: ``select`` 模式下的可选项。
        placeholder_key / placeholder: i18n key 与兜底占位符（仅 input 模式使用）。
        font_size: 字号 override；不传读 ``[Font].size_small``。
    """
    return TagInput(
        mode=mode,
        options=options,
        placeholder_key=placeholder_key,
        placeholder=placeholder,
        font_size=font_size,
        parent=parent,
    )


# =============================================================================
# CheckboxItem — [box] Title 行（无边框、hover/checked 文本变色）
# =============================================================================
class CheckboxItem(QWidget):
    """复选框 + 标题行：方形 box（与首字母同高，2px 圆角），勾选时填 ``safe_zone``。

    标题文字在 hover/checked 时切到 ``highlight``。
    """

    toggled = pyqtSignal(bool)

    def __init__(
        self,
        id: str,
        *,
        default: str = "",
        checked: bool = False,
        font_size: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._i18n_key = id
        self._i18n_default = default
        self._title = default or id
        self._checked = bool(checked)
        self._hover = False
        self._font_size = font_size

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._font = QFont(self.font())
        fs = int(self._font_size) if self._font_size else int(
            Config.get_font_size("size_small")
        )
        self._font.setPointSize(max(1, fs))
        fm = QFontMetrics(self._font)
        text_h = fm.ascent()
        self._box_side = max(12, text_h)
        self._gap = 6
        self.setMinimumHeight(max(self._box_side, fm.height()) + 2)

        if id:
            i18n_bind(self, "setTitle", id, default)

    # ── Public API ──────────────────────────────────────────────────────────
    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self.update()
        self.toggled.emit(checked)

    def toggle(self) -> None:
        self.setChecked(not self._checked)

    def setTitle(self, title: str) -> None:
        self._title = str(title)
        self.updateGeometry()
        self.update()

    def title(self) -> str:
        return self._title

    # ── Sizing ──────────────────────────────────────────────────────────────
    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self._font)
        w = self._box_side + self._gap + fm.horizontalAdvance(self._title) + 4
        h = max(self._box_side, fm.height()) + 2
        return QSize(w, h)

    # ── Events ──────────────────────────────────────────────────────────────
    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        idle_color  = QColor(Config.get_color("main_t1",    "#E5E7EB"))
        hover_color = QColor(Config.get_color("highlight", "#F6D393"))
        muted_color = QColor(Config.get_color("sub_t1",     "#9CA3AF"))
        check_fill  = QColor(Config.get_color("safe_zone",  "#22C55E"))
        check_glyph = QColor(Config.get_color("main_t2",    "#FFFFFF"))

        rect = self.rect()
        fm = QFontMetrics(self._font)

        box_x = 1
        box_y = (rect.height() - self._box_side) // 2
        box_rect = QRectF(box_x, box_y, self._box_side, self._box_side)
        radius = 2.0

        if self._checked:
            painter.setPen(QPen(check_fill, 1.0))
            painter.setBrush(QBrush(check_fill))
            painter.drawRoundedRect(box_rect, radius, radius)
            pen = QPen(check_glyph, max(1.5, self._box_side * 0.16))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            cx, cy = box_rect.left(), box_rect.top()
            s = self._box_side
            p1 = (cx + s * 0.22, cy + s * 0.52)
            p2 = (cx + s * 0.44, cy + s * 0.74)
            p3 = (cx + s * 0.78, cy + s * 0.30)
            path = QPainterPath()
            path.moveTo(*p1)
            path.lineTo(*p2)
            path.lineTo(*p3)
            painter.drawPath(path)
        else:
            stroke_color = hover_color if self._hover else muted_color
            painter.setPen(QPen(stroke_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box_rect, radius, radius)

        text_color = hover_color if (self._checked or self._hover) else idle_color
        painter.setPen(QPen(text_color))
        painter.setFont(self._font)
        text_x = box_x + self._box_side + self._gap
        text_rect = QRectF(text_x, 0, rect.width() - text_x - 2, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._title,
        )
        painter.end()

    def refresh_style(self) -> None:
        """主题切换时调用：本控件每帧从 Config 取色，仅触发重绘即可。"""
        self.update()


def setCheckboxItem(
    id: str,
    *,
    default: str = "",
    checked: bool = False,
    font_size: Optional[int] = None,
    parent: Optional[QWidget] = None,
) -> CheckboxItem:
    """工厂函数：返回挂好 i18n 标题的 ``CheckboxItem``。

    Args:
        id: i18n key（传空串则使用 ``default`` 作为静态标题）。
        default: i18n 兜底文本，或 id 为空时的初始标题。
        checked: 初始勾选状态。
        font_size: 字号 override；不传读 ``[Font].size_small``。
    """
    return CheckboxItem(
        id, default=default, checked=checked, font_size=font_size, parent=parent
    )


# =============================================================================
# LaviTabTable — Tab 头 + 可折叠分组列表（single/multi 选）
# =============================================================================

# 视觉常量（详见 plans/ui-py-tab-lavitabtable-...md 的"视觉规范"小节）
_LTT_TAB_BAR_H     = 28
_LTT_TAB_RADIUS    = 4
_LTT_TAB_PAD_X     = 12
_LTT_ROW_H         = 24
_LTT_GROUP_INDENT  = 16
_LTT_SUBGROUP_INDENT = 12  # extra indent for sub-group headers (top + sub-row stacks)
_LTT_LABEL_GAP     = 6
_LTT_CARET_SIZE    = 8
_LTT_GROUP_GAP     = 8   # vertical gap between top-level groups inside a tab page
_LTT_ITEM_EXTRA_INDENT = 12  # extra left indent on items so they pop out under their group


class _LaviTabBar(QWidget):
    """LaviTabTable 顶部 tab 头。

    水平排列若干 tab；hover/checked 状态下底色变化；checked 底色 = 列表区底色，
    并把底部圆角抹平，实现"无缝拼接"。文字色 idle=sub_t1, hover/checked=highlight。
    """

    tabChanged = pyqtSignal(int)

    def __init__(
        self,
        tab_keys: List[Tuple[str, str]],
        *,
        font_size: Optional[int] = None,
        list_bg_slot: str = "bg_2",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._tab_keys = list(tab_keys)
        self._current = 0 if self._tab_keys else -1
        self._hover = -1
        self._list_bg_slot = list_bg_slot

        self._font = QFont(self.font())
        fs = int(font_size) if font_size else int(Config.get_font_size("size_mini"))
        self._font.setPointSize(max(1, fs))

        self.setFixedHeight(_LTT_TAB_BAR_H)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # i18n 切换 → 重绘（tab 文字现取 I18n.tr）
        I18n.instance().language_changed.connect(self.update)

    # ── Public API ────────────────────────────────────────────────────────
    def currentIndex(self) -> int:
        return self._current

    def setCurrentIndex(self, idx: int) -> None:
        if not (0 <= idx < len(self._tab_keys)):
            return
        if idx == self._current:
            return
        self._current = idx
        self.update()
        self.tabChanged.emit(idx)

    # ── Geometry ──────────────────────────────────────────────────────────
    def _tab_rects(self) -> List[Tuple[int, int, str]]:
        """返回每个 tab 的 (x, width, rendered_text)，所有 tab 等权重均分容器宽度。"""
        rects: List[Tuple[int, int, str]] = []
        n = len(self._tab_keys)
        if n == 0:
            return rects
        total = max(0, int(self.width()))
        base = total // n
        remainder = total - base * n
        x = 0
        for i, (key, default) in enumerate(self._tab_keys):
            text = I18n.tr(key, default) if key else default
            # 把 remainder 像素均匀分配到前 remainder 个 tab，避免右侧留白
            w = base + (1 if i < remainder else 0)
            rects.append((x, w, text))
            x += w
        return rects

    def _index_at(self, x: int) -> int:
        for i, (rx, rw, _t) in enumerate(self._tab_rects()):
            if rx <= x < rx + rw:
                return i
        return -1

    # ── Events ────────────────────────────────────────────────────────────
    def mouseMoveEvent(self, event):
        idx = self._index_at(int(event.position().x()))
        if idx != self._hover:
            self._hover = idx
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover != -1:
            self._hover = -1
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._index_at(int(event.position().x()))
            if idx >= 0:
                self.setCurrentIndex(idx)
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        idle_bg    = QColor(Config.get_color("row_1",      "#2A2A2A"))
        hover_bg   = QColor(Config.get_color("hover_2",    "#212121"))
        checked_bg = QColor(Config.get_color(self._list_bg_slot, "#1A1A1A"))
        idle_fg    = QColor(Config.get_color("sub_t1",     "#A0A0A0"))
        active_fg  = QColor(Config.get_color("highlight", "#F6D393"))

        painter.fillRect(self.rect(), idle_bg)
        painter.setFont(self._font)

        h = self.height()
        radius = _LTT_TAB_RADIUS

        for i, (x, w, text) in enumerate(self._tab_rects()):
            rect = QRectF(x, 0, w, h)
            if i == self._current:
                bg, fg = checked_bg, active_fg
            elif i == self._hover:
                bg, fg = hover_bg, active_fg
            else:
                bg, fg = None, idle_fg

            if bg is not None:
                path = QPainterPath()
                path.addRoundedRect(rect, radius, radius)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(bg))
                painter.drawPath(path)
                # checked: 抹平底部 radius，让 tab 与列表无缝拼接
                if i == self._current:
                    flat = QRectF(rect.x(), rect.y() + h - radius, rect.width(), radius)
                    painter.fillRect(flat, bg)

            painter.setPen(QPen(fg))
            painter.drawText(
                rect, int(Qt.AlignmentFlag.AlignCenter), text
            )
        painter.end()


class _LaviTabTableItemRow(QWidget):
    """LaviTabTable 单 item 行（24px 固定高）。

    布局: ``[indicator-box][gap][prefix][prefix-gap][text]``，左侧 indent 像素缩进。
    选中时 indicator 用 ``safe_zone`` 填充、文字色变 ``safe_zone``。
    可选的 ``prefix`` (e.g. ``"[IsaacLab]"``) 总是用 ``Config.get_color(prefix_color)``
    渲染，不受 hover/checked 状态影响 —— 这是给调用方插一段固定颜色徽标的通用机制
    （见 application.service.projects.file_index.list_canvas_groups）。
    """

    clicked = pyqtSignal(str)  # item id

    def __init__(
        self,
        item_id: str,
        *,
        default: str = "",
        indent: int = 0,
        font_size: Optional[int] = None,
        parent: Optional[QWidget] = None,
        prefix: str = "",
        prefix_color: Optional[str] = None,
    ):
        super().__init__(parent)
        self._item_id = item_id
        self._label_text = default or item_id
        self._indent = max(0, int(indent))
        self._checked = False
        self._hover = False
        # Optional colored badge rendered before the label. Empty string =
        # no badge; ``prefix_color`` is a ``system.ini[Theme]`` slot name
        # (e.g. ``"theme_1"``). Both must be opted-in by the caller; no
        # fallback to fabricate a prefix from the label.
        self._prefix_text: str = str(prefix or "")
        self._prefix_color_slot: Optional[str] = (
            str(prefix_color) if prefix_color else None
        )

        self.setFixedHeight(_LTT_ROW_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self._font = QFont(self.font())
        fs = int(font_size) if font_size else int(Config.get_font_size("size_mini"))
        self._font.setPointSize(max(1, fs))
        fm = QFontMetrics(self._font)
        self._box_side = max(10, fm.ascent())

        if item_id:
            i18n_bind(self, "_setLabel", item_id, default)

    # ── Public API ────────────────────────────────────────────────────────
    def itemId(self) -> str:
        return self._item_id

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self.update()

    def _setLabel(self, text: str) -> None:
        self._label_text = str(text)
        self.update()

    # ── Events ────────────────────────────────────────────────────────────
    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._item_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        list_bg    = QColor(Config.get_color("bg_2",      "#1A1A1A"))
        hover_bg   = QColor(Config.get_color("hover_2",   "#212121"))
        idle_text  = QColor(Config.get_color("main_t1",   "#D6D3C7"))
        muted      = QColor(Config.get_color("sub_t1",    "#A0A0A0"))
        accent     = QColor(Config.get_color("safe_zone", "#36E38E"))
        hover_text = QColor(Config.get_color("highlight", "#F6D393"))

        rect = self.rect()
        painter.fillRect(rect, hover_bg if self._hover else list_bg)

        box_x = self._indent + 1
        box_y = (rect.height() - self._box_side) // 2
        box_rect = QRectF(box_x, box_y, self._box_side, self._box_side)
        radius = 2.0
        if self._checked:
            painter.setPen(QPen(accent, 1.0))
            painter.setBrush(QBrush(accent))
            painter.drawRoundedRect(box_rect, radius, radius)
        else:
            painter.setPen(QPen(muted, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box_rect, radius, radius)

        if self._hover:
            text_color = hover_text
        elif self._checked:
            text_color = accent
        else:
            text_color = idle_text
        font = QFont(self._font)
        if self._hover:
            font.setBold(True)
        painter.setFont(font)
        text_x = box_x + self._box_side + _LTT_LABEL_GAP

        # Optional colored prefix badge — fixed theme color, immune to
        # hover/checked styling so the backend label stays recognizable.
        if self._prefix_text and self._prefix_color_slot:
            prefix_color = QColor(Config.get_color(self._prefix_color_slot))
            fm = QFontMetrics(font)
            prefix_w = fm.horizontalAdvance(self._prefix_text)
            prefix_rect = QRectF(text_x, 0, prefix_w, rect.height())
            painter.setPen(QPen(prefix_color))
            painter.drawText(
                prefix_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                self._prefix_text,
            )
            text_x += prefix_w + _LTT_LABEL_GAP

        painter.setPen(QPen(text_color))
        text_rect = QRectF(text_x, 0, rect.width() - text_x - 2, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._label_text,
        )
        painter.end()


class _LaviTabTableGroupHeader(QWidget):
    """LaviTabTable group header 行（24px 固定高）。

    布局: ``[indent][caret][gap][title]``。点击切换 expanded 状态并发 ``expansionToggled``。
    ``indent`` > 0 用于二级（sub-）group header 的视觉缩进。
    """

    expansionToggled = pyqtSignal(bool)

    def __init__(
        self,
        group_id: str,
        *,
        default: str = "",
        expanded: bool = True,
        indent: int = 0,
        font_size: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._group_id = group_id
        self._label_text = default or group_id
        self._expanded = bool(expanded)
        self._indent = max(0, int(indent))
        self._hover = False

        self.setFixedHeight(_LTT_ROW_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self._font = QFont(self.font())
        fs = int(font_size) if font_size else int(Config.get_font_size("size_mini"))
        self._font.setPointSize(max(1, fs))

        if group_id:
            i18n_bind(self, "_setLabel", group_id, default)

    # ── Public API ────────────────────────────────────────────────────────
    def isExpanded(self) -> bool:
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self.update()
        self.expansionToggled.emit(expanded)

    def _setLabel(self, text: str) -> None:
        self._label_text = str(text)
        self.update()

    # ── Events ────────────────────────────────────────────────────────────
    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setExpanded(not self._expanded)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        list_bg    = QColor(Config.get_color("bg_2",      "#1A1A1A"))
        hover_bg   = QColor(Config.get_color("hover_2",   "#212121"))
        idle_text  = QColor(Config.get_color("main_t2",   "#FFFFFF"))
        hover_text = QColor(Config.get_color("highlight", "#F6D393"))

        rect = self.rect()
        painter.fillRect(rect, hover_bg if self._hover else list_bg)

        cs = _LTT_CARET_SIZE
        cx = 6 + self._indent
        cy = (rect.height() - cs) // 2
        text_col = hover_text if self._hover else idle_text
        path = QPainterPath()
        if self._expanded:
            # ▼
            path.moveTo(cx, cy + 1)
            path.lineTo(cx + cs, cy + 1)
            path.lineTo(cx + cs / 2.0, cy + cs - 1)
        else:
            # ▶
            path.moveTo(cx + 1, cy)
            path.lineTo(cx + 1, cy + cs)
            path.lineTo(cx + cs - 1, cy + cs / 2.0)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(text_col))
        painter.drawPath(path)

        font = QFont(self._font)
        if self._hover:
            font.setBold(True)
        painter.setPen(QPen(text_col))
        painter.setFont(font)
        text_x = cx + cs + _LTT_LABEL_GAP
        text_rect = QRectF(text_x, 0, rect.width() - text_x - 2, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._label_text,
        )
        painter.end()


class LaviTabTable(QWidget):
    """带 Tab 切换 + 可折叠分组 list 的多/单选列表控件。

    - 顶部 4px 圆角 tab 头，hover/checked 底色切换；checked 底色 = 列表区底色，
      与下方列表无缝拼接。
    - 每 tab 内支持「分组」(group header + 缩进 item) 与「扁平」(直接 item) 两种容器，
      互斥。一级 group 内还可再嵌一层 sub-group（top → sub → items），
      支持最多两层嵌套。所有 group header 点击折叠/展开；top 折叠会级联
      隐藏其下所有 sub header 与 item。
    - 选中模式：``single`` 取消其它项；``multi`` 翻转。每个 tab 独立维护选中集合。
    - i18n：tab 名 / group 名 / item 名 全部接 ``I18n.tr``，语种切换自动重译。

    Args:
        tabs: 列表，每项 ``{"id", "default", ("groups"|"items")}``。``groups`` 项再含
            ``{"id","default","expanded", ("groups"|"items")}``：
              - ``items``: ``[{"id","default"}, ...]`` —— 一级 group 直挂 item。
              - ``groups``: ``[{"id","default","expanded","items":[...]}, ...]`` ——
                二级嵌套；sub-group 自身仅可挂 ``items``，不可继续嵌套。
            每一层 ``groups`` 与 ``items`` 互斥。
        kind: ``"single"`` 或 ``"multi"``。
        width / height: 显式最小尺寸，可选。
        font_size: 全控件统一字号；不传走 ``[Font].size_mini``。

    Signals:
        tabChanged(str): 新激活的 tab id。
        selectionChanged(str, list): 该 tab 选中集变化时发 ``(tab_id, [item_id,...])``。
        itemClicked(str, str): 每次点击 item 都发 ``(tab_id, item_id)``。

    Examples:
        >>> tt = setLaviTabTable([
        ...     {"id": "tab.brand", "default": "Brand",
        ...      "groups": [{"id": "grp.h", "default": "Humanoid",
        ...                  "items": [{"id": "robot.h1", "default": "H1"}]}]},
        ...     {"id": "tab.script", "default": "Script",
        ...      "groups": [{"id": "buc.proj", "default": "Project Script",
        ...                  "groups": [
        ...                      {"id": "cat.gait", "default": "gait",
        ...                       "items": [{"id": "project:gait/walk.py",
        ...                                  "default": "walk.py"}]},
        ...                  ]}]},
        ...     {"id": "tab.layer", "default": "Layer",
        ...      "items": [{"id": "lyr.skill", "default": "Skill"}]},
        ... ], kind="multi")
        >>> tt.selectionChanged.connect(lambda tab, items: ...)
    """

    tabChanged       = pyqtSignal(str)
    selectionChanged = pyqtSignal(str, list)
    itemClicked      = pyqtSignal(str, str)

    def __init__(
        self,
        tabs: List[dict],
        *,
        kind: str = "single",
        width: Optional[int] = None,
        height: Optional[int] = None,
        font_size: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if kind not in ("single", "multi"):
            raise ValueError(
                f"LaviTabTable kind must be 'single' or 'multi', got {kind!r}"
            )
        self._kind = kind
        self._font_size = font_size
        self._tabs_spec: List[dict] = list(tabs or [])

        self._tab_ids: List[str] = [t.get("id", "") for t in self._tabs_spec]
        self._selection: dict = {tid: [] for tid in self._tab_ids}
        # tab_id → {item_id: row_widget}
        self._tab_rows: dict = {tid: {} for tid in self._tab_ids}
        # tab_id → list[(group_id, header_widget, [item_rows])]
        self._tab_groups: dict = {tid: [] for tid in self._tab_ids}
        # tab_id → {is_flat, stack?, hint?, scroll, content, layout} —— 用于
        # setTabItems / setTabEmptyHint 动态重建。stack/hint 仅在 flat 模式下存在。
        self._tab_pages: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        tab_keys = [(t.get("id", ""), t.get("default", "")) for t in self._tabs_spec]
        self._tab_bar = _LaviTabBar(
            tab_keys,
            font_size=font_size,
            list_bg_slot="bg_2",
            parent=self,
        )
        self._tab_bar.tabChanged.connect(self._on_tab_index_changed)
        outer.addWidget(self._tab_bar)

        self._stack = QStackedWidget(self)
        outer.addWidget(self._stack, 1)

        if not self._tabs_spec:
            log_warning("LaviTabTable: empty tabs list — showing empty placeholder")
            placeholder = self._make_list_bg_widget()
            self._stack.addWidget(placeholder)
        else:
            for spec in self._tabs_spec:
                page = self._build_tab_page(spec)
                self._stack.addWidget(page)

        if width is not None:
            self.setMinimumWidth(int(width))
        if height is not None:
            self.setMinimumHeight(int(height))

    # ── Public API ────────────────────────────────────────────────────────
    @property
    def kind(self) -> str:
        return self._kind

    def currentTab(self) -> str:
        idx = self._tab_bar.currentIndex()
        if 0 <= idx < len(self._tab_ids):
            return self._tab_ids[idx]
        return ""

    def setCurrentTab(self, tab_id: str) -> None:
        if tab_id in self._tab_ids:
            self._tab_bar.setCurrentIndex(self._tab_ids.index(tab_id))

    def selectionMap(self) -> dict:
        """返回 ``{tab_id: [item_id, ...]}``，包含所有 tab 的当前选中（每 tab 独立）。"""
        return {tid: list(items) for tid, items in self._selection.items()}

    def currentSelection(self) -> List[str]:
        """语法糖：仅返回当前 tab 的选中项。"""
        return list(self._selection.get(self.currentTab(), []))

    def setSelection(self, tab_id: str, items: List[str]) -> None:
        if tab_id not in self._tab_rows:
            return
        items = list(items)
        if self._kind == "single" and len(items) > 1:
            items = items[:1]
        rows = self._tab_rows[tab_id]
        for iid, row in rows.items():
            row.setChecked(iid in items)
        new_sel = [iid for iid in items if iid in rows]
        if new_sel != self._selection[tab_id]:
            self._selection[tab_id] = new_sel
            self.selectionChanged.emit(tab_id, list(new_sel))

    def clearSelection(self, tab_id: Optional[str] = None) -> None:
        targets = [tab_id] if tab_id is not None else list(self._tab_ids)
        for tid in targets:
            if tid in self._selection and self._selection[tid]:
                self.setSelection(tid, [])

    def setExpanded(self, tab_id: str, group_id: str, expanded: bool) -> None:
        for gid, header, _items in self._tab_groups.get(tab_id, []):
            if gid == group_id:
                header.setExpanded(bool(expanded))
                return

    def setTabItems(self, tab_id: str, items: List[dict]) -> None:
        """动态替换某 flat-mode tab 的 item 列表。

        - 仅作用于 flat 模式（构造时 ``items`` 路径，无 ``groups``）；groups
          模式下调用会被忽略并发警告。
        - 上一次的选中项若仍在新列表中则保留勾选；否则该 tab 选中清空，
          ``selectionChanged(tab_id, [])`` 会被发出。
        - 列表为空时自动切到空态 hint（参见 :meth:`setTabEmptyHint`）。

        Args:
            tab_id: tab 的 id（与构造 ``tabs`` spec 一致）。
            items: ``[{"id": str, "default": str}, ...]``。
        """
        if tab_id not in self._tab_pages:
            return
        page = self._tab_pages[tab_id]
        if not page["is_flat"]:
            log_warning(
                f"LaviTabTable.setTabItems: tab {tab_id!r} is in groups mode; "
                "dynamic replacement is only supported in flat mode."
            )
            return

        prev_sel = list(self._selection.get(tab_id, []))

        # 清掉旧 row（含 stretch）
        layout: QVBoxLayout = page["layout"]
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._tab_rows[tab_id] = {}

        # 新 row
        content: QWidget = page["content"]
        new_items = list(items or [])
        for spec in new_items:
            row = self._build_item_row(tab_id, spec, indent=0, parent=content)
            layout.addWidget(row)
        layout.addStretch(1)

        # 选中保留：仅保留新列表中仍存在的 id
        new_ids = set(self._tab_rows[tab_id].keys())
        new_sel = [iid for iid in prev_sel if iid in new_ids]
        self._selection[tab_id] = new_sel
        for iid in new_sel:
            row = self._tab_rows[tab_id].get(iid)
            if row is not None:
                row.setChecked(True)

        # 空态 ↔ 列表切换
        page["stack"].setCurrentIndex(1 if new_items else 0)

        if new_sel != prev_sel:
            self.selectionChanged.emit(tab_id, list(new_sel))

    def setTabEmptyHint(
        self, tab_id: str, hint_id: str, default: str = ""
    ) -> None:
        """为 flat-mode tab 绑定空态 hint 文本（i18n 联动）。

        当该 tab 的 ``items`` 为空时，列表区域会显示这段居中提示。空字符串
        ``hint_id`` 会让 hint 静态显示 ``default``，跳过 I18n 绑定。
        """
        if tab_id not in self._tab_pages:
            return
        page = self._tab_pages[tab_id]
        if not page["is_flat"]:
            return
        hint: QLabel = page["hint"]
        if hint_id:
            i18n_bind(hint, "setText", hint_id, default)
        else:
            hint.setText(default)

    def refresh_style(self) -> None:
        """主题切换时调用：所有子部件每帧重取色，仅触发重绘。"""
        self._tab_bar.update()
        for tid in self._tab_ids:
            for row in self._tab_rows[tid].values():
                row.update()
            for _gid, header, _items in self._tab_groups[tid]:
                header.update()
        # 重设页背景调色板 + hint 字色
        for tid, page in self._tab_pages.items():
            for key in ("stack", "scroll", "content"):
                w = page.get(key)
                if w is not None:
                    self._apply_list_bg(w)
            scroll = page.get("scroll")
            if scroll is not None:
                vp = scroll.viewport()
                if vp is not None:
                    self._apply_list_bg(vp)
            hint = page.get("hint")
            if hint is not None:
                self._apply_hint_style(hint)

    # ── Private ────────────────────────────────────────────────────────────
    def _make_list_bg_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        w = QWidget(parent if parent is not None else self)
        self._apply_list_bg(w)
        return w

    def _apply_list_bg(self, w: QWidget) -> None:
        # The list-surface colour comes from a dedicated semantic slot so the
        # host project can choose between a solid fill or show-through. An
        # empty / "transparent" value disables the autofill entirely, letting
        # the parent panel's background paint through.
        color = Config.get_color(
            "lavi_tab_table_bg",
            Config.get_color("bg_2", "#1A1A1A"),
        )
        if not color or color.strip().lower() == "transparent":
            w.setAutoFillBackground(False)
            return
        w.setAutoFillBackground(True)
        pal = w.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(color))
        w.setPalette(pal)

    def _build_tab_page(self, spec: dict) -> QWidget:
        tab_id = spec.get("id", "")
        groups = spec.get("groups")
        items = spec.get("items")
        if groups and items:
            raise ValueError(
                f"LaviTabTable tab {tab_id!r}: 'groups' and 'items' are mutually exclusive"
            )

        is_flat = not groups  # flat = items 列表（含空），可后续 setTabItems 动态替换

        # 内层内容容器（item/header 列表）
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._apply_list_bg(content)

        if groups:
            for _grp_idx, grp in enumerate(groups):
                if _grp_idx > 0:
                    layout.addSpacing(_LTT_GROUP_GAP)
                gid = grp.get("id", "")
                top_header = _LaviTabTableGroupHeader(
                    gid,
                    default=grp.get("default", ""),
                    expanded=bool(grp.get("expanded", True)),
                    font_size=self._font_size,
                    parent=content,
                )
                layout.addWidget(top_header)

                sub_groups = grp.get("groups")
                grp_items = grp.get("items")
                if sub_groups and grp_items:
                    raise ValueError(
                        f"LaviTabTable tab {tab_id!r} group {gid!r}: "
                        "'groups' and 'items' are mutually exclusive"
                    )

                if sub_groups:
                    # Two-level: top → sub-group headers → items.
                    sub_entries: List[
                        tuple
                    ] = []  # [(sub_header, [item_rows]), ...]
                    self._tab_groups[tab_id].append((gid, top_header, []))
                    for sgrp in sub_groups:
                        sgid = sgrp.get("id", "")
                        if sgrp.get("groups"):
                            raise ValueError(
                                f"LaviTabTable tab {tab_id!r} sub-group {sgid!r}: "
                                "nesting beyond two levels is not supported"
                            )
                        sub_header = _LaviTabTableGroupHeader(
                            sgid,
                            default=sgrp.get("default", ""),
                            expanded=bool(sgrp.get("expanded", True)),
                            indent=_LTT_GROUP_INDENT,
                            font_size=self._font_size,
                            parent=content,
                        )
                        layout.addWidget(sub_header)
                        sub_rows: List[_LaviTabTableItemRow] = []
                        for it in (sgrp.get("items") or []):
                            row = self._build_item_row(
                                tab_id,
                                it,
                                indent=_LTT_GROUP_INDENT
                                + _LTT_SUBGROUP_INDENT
                                + _LTT_ITEM_EXTRA_INDENT,
                                parent=content,
                            )
                            layout.addWidget(row)
                            sub_rows.append(row)
                        sub_entries.append((sub_header, sub_rows))
                        self._tab_groups[tab_id].append(
                            (sgid, sub_header, sub_rows)
                        )
                    self._wire_groups_nested(top_header, sub_entries)
                else:
                    # One-level: top group with direct items (legacy form).
                    row_list: List[_LaviTabTableItemRow] = []
                    for it in (grp_items or []):
                        row = self._build_item_row(
                            tab_id,
                            it,
                            indent=_LTT_GROUP_INDENT + _LTT_ITEM_EXTRA_INDENT,
                            parent=content,
                        )
                        layout.addWidget(row)
                        row_list.append(row)
                    self._wire_group(top_header, row_list)
                    self._tab_groups[tab_id].append((gid, top_header, row_list))
        else:
            for it in (items or []):
                row = self._build_item_row(tab_id, it, indent=0, parent=content)
                layout.addWidget(row)

        layout.addStretch(1)

        # 外层 scroll：item 列表过长时滚动；视觉上仍是同一底色
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        self._apply_list_bg(scroll)
        self._apply_list_bg(scroll.viewport())

        if is_flat:
            # 包一层 QStackedWidget：idx 0 = 空态 hint，idx 1 = 滚动列表。
            # setTabItems 时根据 items 数量切换；setTabEmptyHint 时绑定文本。
            wrapper = QStackedWidget()
            self._apply_list_bg(wrapper)
            hint = QLabel("", wrapper)
            hint.setObjectName("laviTabTableHint")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            self._apply_hint_style(hint)
            wrapper.addWidget(hint)
            wrapper.addWidget(scroll)
            wrapper.setCurrentIndex(1 if items else 0)
            self._tab_pages[tab_id] = {
                "is_flat": True,
                "stack": wrapper,
                "hint": hint,
                "scroll": scroll,
                "content": content,
                "layout": layout,
            }
            return wrapper

        self._tab_pages[tab_id] = {
            "is_flat": False,
            "stack": None,
            "hint": None,
            "scroll": scroll,
            "content": content,
            "layout": layout,
        }
        return scroll

    def _apply_hint_style(self, hint: QLabel) -> None:
        fs = int(Config.get_font_size("size_mini"))
        color = Config.get_color("sub_t1", "#A0A0A0")
        # 背景由父级 palette 提供（row_1）；只设置文字 + 字号 + padding。
        hint.setStyleSheet(
            "QLabel#laviTabTableHint {"
            f" color: {color}; font-size: {fs}px;"
            " padding: 16px; background: transparent; }"
        )

    def _wire_group(
        self,
        header: "_LaviTabTableGroupHeader",
        rows: List["_LaviTabTableItemRow"],
    ) -> None:
        def _apply(expanded: bool):
            for r in rows:
                r.setVisible(expanded)
        _apply(header.isExpanded())
        header.expansionToggled.connect(_apply)

    def _wire_groups_nested(
        self,
        top_header: "_LaviTabTableGroupHeader",
        sub_entries: List[tuple],   # [(sub_header, [item_rows]), ...]
    ) -> None:
        """级联折叠：top 折 → 隐藏所有 sub header + items；sub 折 → 仅隐藏自身 items。"""

        def _apply_top(top_expanded: bool) -> None:
            for sub_header, sub_rows in sub_entries:
                sub_header.setVisible(top_expanded)
                visible_items = top_expanded and sub_header.isExpanded()
                for r in sub_rows:
                    r.setVisible(visible_items)

        def _make_sub_handler(sub_header, sub_rows):
            def _apply_sub(sub_expanded: bool) -> None:
                for r in sub_rows:
                    r.setVisible(top_header.isExpanded() and sub_expanded)
            return _apply_sub

        _apply_top(top_header.isExpanded())
        top_header.expansionToggled.connect(_apply_top)
        for sub_header, sub_rows in sub_entries:
            sub_header.expansionToggled.connect(
                _make_sub_handler(sub_header, sub_rows)
            )

    def _build_item_row(
        self,
        tab_id: str,
        it: dict,
        *,
        indent: int,
        parent: QWidget,
    ) -> "_LaviTabTableItemRow":
        iid = it.get("id", "")
        row = _LaviTabTableItemRow(
            iid,
            default=it.get("default", ""),
            indent=indent,
            font_size=self._font_size,
            parent=parent,
            prefix=it.get("prefix", ""),
            prefix_color=it.get("prefix_color"),
        )
        if iid:
            self._tab_rows[tab_id][iid] = row
        row.clicked.connect(
            lambda item_id, tid=tab_id: self._on_item_clicked(tid, item_id)
        )
        return row

    def _on_tab_index_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._tab_ids) and idx < self._stack.count():
            self._stack.setCurrentIndex(idx)
            self.tabChanged.emit(self._tab_ids[idx])

    def _on_item_clicked(self, tab_id: str, item_id: str) -> None:
        self.itemClicked.emit(tab_id, item_id)
        rows = self._tab_rows.get(tab_id, {})
        if item_id not in rows:
            return
        row = rows[item_id]
        prev = list(self._selection.get(tab_id, []))

        if self._kind == "single":
            if row.isChecked():
                row.setChecked(False)
                self._selection[tab_id] = []
            else:
                for iid, r in rows.items():
                    if r is not row and r.isChecked():
                        r.setChecked(False)
                row.setChecked(True)
                self._selection[tab_id] = [item_id]
        else:  # multi
            if row.isChecked():
                row.setChecked(False)
                if item_id in self._selection[tab_id]:
                    self._selection[tab_id].remove(item_id)
            else:
                row.setChecked(True)
                if item_id not in self._selection[tab_id]:
                    self._selection[tab_id].append(item_id)

        if self._selection[tab_id] != prev:
            self.selectionChanged.emit(tab_id, list(self._selection[tab_id]))


def setLaviTabTable(
    tabs: List[dict],
    *,
    kind: str = "single",
    width: Optional[int] = None,
    height: Optional[int] = None,
    font_size: Optional[int] = None,
    parent: Optional[QWidget] = None,
) -> LaviTabTable:
    """工厂函数：返回挂好主题、i18n 绑定的 :class:`LaviTabTable`。

    Args:
        tabs: 见 :class:`LaviTabTable` 文档。
        kind: ``"single"`` 或 ``"multi"``。
        width / height: 显式最小尺寸（不传由布局决定）。
        font_size: 全控件统一字号；不传读 ``[Font].size_mini``。
    """
    return LaviTabTable(
        tabs,
        kind=kind,
        width=width,
        height=height,
        font_size=font_size,
        parent=parent,
    )
