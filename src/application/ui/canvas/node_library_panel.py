# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas.node_library_panel — Node Library 面板（移植自 DEMO ModulePalette）.

- 数据源 / data: ``registers.nodes.list_nodes()`` —— 启动期由 ``RegistryHub.load_all()``
  扫描 ``src/nodes/`` + ``custom_mods/nodes/`` 自动发现，运行期 ``reload()`` 后调用方
  emit ``app_signals.nodes_changed`` 通知本面板刷新。
- 归档 / grouping: 按 ``manifest.layer``（A/B/C/D/IL/CUSTOM）分组，与
  ``CanvasPage._populate_add_node_menu`` 已确立的层级语义一致。
- 引擎感知 / engine-aware: 订阅 ``app_signals.current_backend_changed``；刷新时按
  ``list_nodes(backend=current_backend())`` 过滤。``manifest.backends`` 是唯一权威
  （与 AI Build ``registry_projection`` 同一谓词）：isaac_lab 画布绝不显示
  SB3-only 节点（task_config / obs_action_config / …），反之亦然；universal
  （空 ``backends``）恒显示。未绑后端（``current_backend() == ""``，无画布
  打开）时显示全量目录。
- 交互 / interaction:
  - 双击叶节点 → emit ``node_requested(node_id)`` （上层调 ``CanvasPage.spawn_node``）
  - 拖动叶节点 → ``QDrag`` + MIME ``application/x-unitport-node`` payload = ``node_id``
    （由 ``canvas/view.py`` 的 ``dropEvent`` 接收 + ``mapToScene`` 后 spawn）

DEMO 参考：``DEMO/bin/pages/layout/module_cards.py:117``（``ModulePalette``）—— UI 壳
照搬，但 DEMO 的硬编码节点列表 + 无引擎订阅的限制都在本实现中替换为 registry 驱动。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QMimeData, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QDrag, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, I18n, log_warning, tr


# ---- MIME 协议 ------------------------------------------------------------------

#: drag/drop MIME type；payload = utf-8 编码的 node_id（``manifest.id``）
MIME_NODE_ID = "application/x-unitport-node"


# ---- 内部 roles + 常量 ----------------------------------------------------------

# QTreeWidgetItem.data(0, ROLE_NODE_ID) → str | None；叶行才有
ROLE_NODE_ID = Qt.ItemDataRole.UserRole + 1
# QTreeWidgetItem.data(0, ROLE_IS_GROUP) → bool；组行 True
ROLE_IS_GROUP = Qt.ItemDataRole.UserRole + 2

# layer 显示顺序（与 CanvasPage._populate_add_node_menu 对齐）
# TOOLS 放最后 —— 注释/分组/辅助类节点(Note 等)与编译流水线无关,
# 远离 A→D 配置→训练→导出主线,视觉降权。
_LAYER_ORDER = ["A", "B", "C", "D", "IL", "CUSTOM", "TOOLS"]

# layer 业务显示名 — 替代裸字母 A/B/C/D/IL/CUSTOM/TOOLS
_LAYER_DISPLAY_NAMES: Dict[str, str] = {
    "A": "Configuration",
    "B": "Environment",
    "C": "Training",
    "D": "Export",
    "IL": "Imitation Learning",
    "CUSTOM": "Custom",
    "TOOLS": "Tools",
}


# =============================================================================
# Delegate —— 组行底色 + 字号统一
# =============================================================================

class _NodeTreeDelegate(QStyledItemDelegate):
    """组行字号 + 文本色定制（DEMO ``module_cards.NodeTreeDelegate`` 的 RELEASE 化）.

    叶行走默认渲染；组行不再上底色——让 sidebar 弹出面板自己的背景色（bg_3）
    直接透出，与 §1.5 "颜色一律 system.ini[Theme]" 不冲突（这里只是不画自己
    的底色，让上层 panel 的底色生效）。文字仍走 highlight 色 + 加粗。
    """

    def initStyleOption(
        self,
        option: QStyleOptionViewItem,
        index,
    ) -> None:  # type: ignore[override]
        super().initStyleOption(option, index)
        is_group = bool(index.data(ROLE_IS_GROUP))
        if is_group:
            from PyQt6.QtGui import QColor, QPalette
            # 折叠栏文本走 highlight 色（system.ini[Theme] slot 名 highlight）
            hl_hex = Config.get_color("highlight", "#F6D393")
            hl_color = QColor(hl_hex)
            for role in (
                QPalette.ColorRole.Text,
                QPalette.ColorRole.WindowText,
                QPalette.ColorRole.ButtonText,
                QPalette.ColorRole.HighlightedText,
            ):
                option.palette.setColor(role, hl_color)
            # 折叠栏字号 size_small（与叶节点 stylesheet 同字号）
            font = option.font
            font.setBold(True)
            font.setPixelSize(int(Config.get_font_size("size_small")))
            option.font = font


# =============================================================================
# Tree —— 重写 startDrag 走自定义 MIME
# =============================================================================

class _NodeTree(QTreeWidget):
    """QTreeWidget 子类：仅叶行可拖；MIME = ``application/x-unitport-node``."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setAnimated(False)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setItemDelegate(_NodeTreeDelegate(self))
        self.setMouseTracking(True)
        self.refresh_style()

    # ---- 鼠标 cursor：叶节点显示手型 ----

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        item = self.itemAt(event.pos())
        if item is not None and not bool(item.data(0, ROLE_IS_GROUP)) \
                and item.data(0, ROLE_NODE_ID):
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.viewport().unsetCursor()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self.viewport().unsetCursor()
        super().leaveEvent(event)

    # ---- 主题刷新 ----

    def refresh_style(self) -> None:
        """重读主题色 + 字号；主题切换或挂载完成时调用.

        Tree 与 viewport 都走 transparent —— 让 Sidebar 弹出面板（``bg_3``）的
        底色直接透出，整列条目不再画自己的底色。hover / selected 仍用
        ``hover_1``，是交互态色，不属于"item 列表底色"。
        """
        fg = Config.get_color("main_t1", "#D6D3C7")
        # 叶节点 + 折叠栏字号统一 size_small（迁入 Sidebar 弹出面板后由用户敲定）
        size = int(Config.get_font_size("size_small"))
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        self.setStyleSheet(
            f"""
            QTreeWidget {{
                background-color: transparent;
                color: {fg};
                border: 0;
                font-family: "{family}";
                font-size: {size}px;
                outline: 0;
            }}
            QTreeWidget::viewport {{
                background-color: transparent;
            }}
            QTreeWidget::item {{
                padding: 3px 6px;
                background-color: transparent;
            }}
            QTreeWidget::item:selected {{
                background-color: {Config.get_color("hover_1", "#525252")};
                color: {Config.get_color("main_t2", "#FFFFFF")};
            }}
            QTreeWidget::item:hover {{
                background-color: {Config.get_color("hover_1", "#525252")};
            }}
            """
        )
        self.viewport().update()

    # ---- 拖拽 ----

    def startDrag(self, supportedActions) -> None:  # type: ignore[override]
        item = self.currentItem()
        if item is None:
            return
        node_id = item.data(0, ROLE_NODE_ID)
        if not node_id:
            # 组行 / 占位行 → 不发起 drag
            return

        mime = QMimeData()
        mime.setData(MIME_NODE_ID, str(node_id).encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)

        # pixmap：用 item 文本作为拖影标签（DEMO 同样无 icon）
        label_text = item.text(0) or str(node_id)
        pixmap = self._make_label_pixmap(label_text)
        drag.setPixmap(pixmap)
        drag.setHotSpot(pixmap.rect().center())

        drag.exec(Qt.DropAction.CopyAction)

    @staticmethod
    def _make_label_pixmap(text: str) -> QPixmap:
        """简易拖影：圆角矩形 + 文本居中."""
        from PyQt6.QtGui import QColor, QFont, QPainterPath, QPen

        family = Config.get_value("Font", "family", "Microsoft YaHei")
        size = int(Config.get_font_size("size_normal"))
        font = QFont(family, size - 2 if size > 12 else size)
        # 估算宽度
        from PyQt6.QtGui import QFontMetrics
        fm = QFontMetrics(font)
        w = max(80, fm.horizontalAdvance(text) + 24)
        h = max(24, fm.height() + 10)
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(0, 0, 0, 0))
        p = QPainter(pixmap)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            bg = QColor(Config.get_color("btn_1", "#3A3A3A"))
            bg.setAlpha(230)
            border = QColor(Config.get_color("border_1", "#444444"))
            path = QPainterPath()
            path.addRoundedRect(0.5, 0.5, w - 1, h - 1, 6, 6)
            p.fillPath(path, bg)
            p.setPen(QPen(border, 1))
            p.drawPath(path)
            p.setPen(QColor(Config.get_color("main_t1", "#D6D3C7")))
            p.setFont(font)
            p.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        finally:
            p.end()
        return pixmap


# =============================================================================
# Panel —— 主部件
# =============================================================================

class NodeLibraryPanel(QWidget):
    """Node Library 主面板 / Node Library main panel.

    Signals:
        node_requested(str): 双击某个节点叶行时发出，参数 ``node_id``（manifest.id）
    """

    node_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tree = _NodeTree(self)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._tree, 1)

        # 订阅 app 信号（registry 内容变 / 引擎切换 → 刷新）
        from application.service.signals import get_app_signals
        sig = get_app_signals()
        sig.nodes_changed.connect(self._refresh)
        sig.current_backend_changed.connect(self._on_backend_changed)
        # 语种切换 → 重建树（节点 display_name + layer 分组 label + 占位
        # 文案都是 tr() 在 _refresh 里一次性拿的，不重建会卡在旧语种）。
        I18n.instance().language_changed.connect(self._refresh)

        self._refresh()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(240, 360)

    # ---- 主题 ----

    def refresh_style(self) -> None:
        """主题切换时由上层（card / overlay）调用."""
        self._tree.refresh_style()

    # ---- 数据刷新 ----

    def _on_backend_changed(self, _engine_id: str) -> None:
        self._refresh()

    def _refresh(self) -> None:
        """重建 tree：query registers → engine 过滤 → 按 layer 分组."""
        # 延迟 import：避免画布 import 链在 registers 未就绪时炸
        try:
            from registers import nodes as nodes_registry
        except Exception as e:
            log_warning(f"[node_library] registers.nodes 未就绪: {e!r}")
            self._tree.clear()
            self._show_empty_placeholder()
            return

        # 后端过滤：manifest.backends 是唯一权威（与 AI Build 的
        # registry_projection 同一谓词）。engine_id == ""（无画布打开）→ 全量。
        from application.service.signals import current_backend
        engine_id = current_backend()
        try:
            manifests = list(nodes_registry.list_nodes(backend=engine_id or None))
        except Exception as e:
            log_warning(f"[node_library] list_nodes() 失败: {e!r}")
            self._tree.clear()
            self._show_empty_placeholder()
            return

        self._tree.clear()
        if not manifests:
            self._show_empty_placeholder()
            return

        # 按 layer 分组
        groups: Dict[str, List] = {}
        for m in manifests:
            layer = (getattr(m, "layer", None) or "CUSTOM").upper()
            groups.setdefault(layer, []).append(m)

        # 显示顺序：固定 A/B/C/D/IL/CUSTOM，其他按字母末尾追加
        ordered_keys: List[str] = [k for k in _LAYER_ORDER if k in groups]
        ordered_keys += sorted(k for k in groups if k not in _LAYER_ORDER)

        for layer in ordered_keys:
            label = self._layer_label(layer)
            group_item = QTreeWidgetItem([label])
            group_item.setData(0, ROLE_IS_GROUP, True)
            group_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled  # 不可选 + 不可拖
            )
            self._tree.addTopLevelItem(group_item)

            for m in sorted(groups[layer], key=lambda x: x.id):
                leaf = self._make_leaf_item(m)
                group_item.addChild(leaf)

            group_item.setExpanded(True)

    def _make_leaf_item(self, manifest) -> QTreeWidgetItem:
        node_id = str(manifest.id)
        # 显示名走 display_name_key — 与画布上 NodeItem._resolve_title 一致
        # （items.py:476-484），保证拖入前后标题完全一致；缺译时回退 node_id
        name_key = getattr(manifest, "display_name_key", "") or ""
        display = tr(name_key, node_id) if name_key else node_id
        item = QTreeWidgetItem([display])
        item.setData(0, ROLE_NODE_ID, node_id)
        item.setData(0, ROLE_IS_GROUP, False)
        # tooltip：node_id + category + kind（信息密度高于显示名）
        kind_val = getattr(manifest, "kind", None)
        kind_str = (
            kind_val.value if hasattr(kind_val, "value") else str(kind_val or "")
        )
        category = str(getattr(manifest, "category", "") or "")
        tip_lines = [f"id: {node_id}"]
        if category:
            tip_lines.append(f"category: {category}")
        if kind_str:
            tip_lines.append(f"kind: {kind_str}")
        item.setToolTip(0, "\n".join(tip_lines))
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        return item

    def _show_empty_placeholder(self) -> None:
        msg = tr("node_lib.empty", "No nodes registered")
        item = QTreeWidgetItem([msg])
        item.setData(0, ROLE_IS_GROUP, False)
        item.setFlags(Qt.ItemFlag.NoItemFlags)  # 不可选 / 不可拖 / 不可双击
        self._tree.addTopLevelItem(item)

    @staticmethod
    def _layer_label(layer: str) -> str:
        # 业务命名：A→Configuration / B→Environment / C→Training / D→Export
        # / IL→Imitation Learning / CUSTOM→Custom；i18n key 仍保留以便后续接翻译
        fallback = _LAYER_DISPLAY_NAMES.get(layer, layer.title())
        return tr(f"node_lib.layer.{layer.lower()}", fallback)

    # ---- 交互 ----

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        node_id = item.data(0, ROLE_NODE_ID)
        if not node_id:
            return
        self.node_requested.emit(str(node_id))


__all__ = [
    "MIME_NODE_ID",
    "NodeLibraryPanel",
]
