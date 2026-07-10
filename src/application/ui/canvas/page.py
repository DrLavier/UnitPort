# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CanvasPage — Training Ground 画布页面壳 / Canvas page shell.

Phase 0: ``QWidget`` 容器，里面挂 ``CanvasView`` + ``CanvasScene``，仅栅格背景。
Phase 1a (本阶段): + ``spawn_node`` / ``connect`` / ``populate_demo`` 接入位。
Phase 1b: 接入调色板（左侧抽屉）+ 拖拽连线 + LOD。
Phase 1c: lowering 接入（IR roundtrip）。

主题切换：上层（``MainWindow.apply_theme``）应调 ``self.apply_theme()`` 让画布刷栅格色。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from contextlib import contextmanager

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QUndoStack
from PyQt6.QtWidgets import (
    QGraphicsSceneContextMenuEvent,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    I18n,
    clearDirtyTracker,
    getDirtyTracker,
    i18n_bind,
    log_debug,
    log_error,
    log_success,
    log_warning,
    read_data,
    reportDirtyText,
    save_data,
    setDirtyBaseline,
    tr,
)

from application.compiler.lowering import (
    IREdge,
    IRNode,
    IRNodeUI,
    IRParam,
    WorkflowIR,
    canvas_to_ir,
    ir_to_canvas,
)
from registers import backends as backends_registry
from registers import robots as robots_registry

from .agent_bridge import CanvasAgentBridge
from .ai_build_panel import AiBuildPanel
from .control_guide import CanvasControlGuide
from .group_item import GroupItem
from .items import ConnectionItem, NodeItem, PlaceholderNodeItem, connect_ports
from .minimap import CanvasMiniMap
from .node_library_overlay import NodeLibraryCardController, NodeLibraryOverlay
from .note_item import NoteItem
from .scene import CanvasScene
from .selection_toolbar import SelectionToolbar
from .view import CanvasView


def _is_legacy_training_canvas(data: Dict[str, Any]) -> bool:
    """检测 DEMO 旧版 ``training_canvas_v1`` 存档 / Detect DEMO legacy schema.

    严格信号：顶层 ``schema == "training_canvas_v1"``。
    宽容信号：第一个节点带 ``node_type`` 且无 ``schema_id`` —— 用于 schema 串
    被旧分支漏写的样本。两条信号在 IR-shape 真实文件上都不可能成立。
    """
    if not isinstance(data, dict):
        return False
    if str(data.get("schema", "")) == "training_canvas_v1":
        return True
    nodes = data.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        return False
    first = nodes[0] if isinstance(nodes[0], dict) else {}
    return ("node_type" in first) and ("schema_id" not in first)


def _legacy_to_ir(data: Dict[str, Any]) -> Dict[str, Any]:
    """``training_canvas_v1`` dict → IR-shape dict（纯函数，不修改入参）.

    字段映射（已通过 DEMO 真实样本核实）：
        node: ``node_type`` → ``schema_id``，``pos:[x,y]`` → ``ui:{x,y,...}``，
              ``parameters`` → ``params``（内层 string 由 from_workflow_dict 兜底处理）
        edge: ``src_id/src_slot/dst_id/dst_slot`` → ``source_node/source_port/target_node/target_port``
              缺 edge id 由 ``connect_ports`` 自动生成 uuid12。
        ``display_name`` 丢弃 —— RELEASE NodeItem 标题严格从 manifest.display_name_key 取。
        顶层 ``backend`` 必须存在并透传到 IR-shape 顶层（画布 backend 是一等不可改属性）。
        ``schema/active_backend`` 留 ``metadata.legacy`` 作 provenance。
    """
    backend = data.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(
            "_legacy_to_ir: legacy canvas missing required top-level 'backend' field; "
            "RELEASE does not support unbacked canvases (project still in development)"
        )
    legacy_meta = {
        "schema": data.get("schema"),
        "active_backend": data.get("active_backend"),
    }
    nodes_out: List[Dict[str, Any]] = []
    for n in data.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        pos = n.get("pos") or [0.0, 0.0]
        x = float(pos[0]) if len(pos) >= 1 else 0.0
        y = float(pos[1]) if len(pos) >= 2 else 0.0
        nodes_out.append({
            "id": n.get("id"),
            "schema_id": n.get("node_type"),
            "kind": "custom",
            "params": dict(n.get("parameters") or {}),
            "ui": {"x": x, "y": y, "width": 180.0, "height": 110.0, "collapsed": False},
            "opaque_code": None,
        })
    edges_out: List[Dict[str, Any]] = []
    for e in data.get("edges") or []:
        if not isinstance(e, dict):
            continue
        edges_out.append({
            "source_node": e.get("src_id"),
            "source_port": e.get("src_slot"),
            "target_node": e.get("dst_id"),
            "target_port": e.get("dst_slot"),
        })
    return {
        "version": "1.0.0",
        "backend": backend.strip(),
        "metadata": {"legacy": legacy_meta},
        "nodes": nodes_out,
        "edges": edges_out,
    }


def _migrate_base_asset_params(node_dict: Dict[str, Any]) -> None:
    """In-place rewrite of legacy ``base_asset`` param tokens.

    base_asset v1 stored a flat token in ``start_point`` (``__new__`` /
    ``__latest_export__`` / ``run:<abs>`` / ``asset:<id>``) plus
    ``asset_id`` + ``checkpoint_file`` siblings. v2 (2026-05) collapses
    the picker to 3 semantic categories and routes specific checkpoint
    picks through a separate ``checkpoint_id`` field.

    Mapping:
      ``run:<abs>``         → start_point=``__load__``, checkpoint_id=``run:<abs>``
      ``asset:<id>``        → start_point=``__new__`` + log_warning (registry gone)
      ``__latest_export__`` → start_point=``__new__`` + log_warning (semantics shifted)
      ``__new__``           → unchanged
      v2 tokens             → unchanged (idempotent)

    Always strips the now-obsolete ``asset_id`` / ``checkpoint_file``
    IRParam entries — they're not in the v2 manifest.
    """
    if node_dict.get("schema_id") != "base_asset":
        return
    params = node_dict.get("params") or {}
    sp_entry = params.get("start_point")
    if not isinstance(sp_entry, dict):
        return
    sp_val = sp_entry.get("value")
    if not isinstance(sp_val, str):
        return
    if sp_val in ("__new__", "__latest__", "__load__"):
        # Already v2.
        params.pop("asset_id", None)
        params.pop("checkpoint_file", None)
        return
    if sp_val.startswith("run:"):
        sp_entry["value"] = "__load__"
        params["checkpoint_id"] = {
            "name": "checkpoint_id",
            "value": sp_val,
            "param_type": "string",
        }
    elif sp_val.startswith("asset:"):
        sp_entry["value"] = "__new__"
        log_warning(
            f"[base_asset migrate] dropped legacy asset:<id> token "
            f"{sp_val!r}; reset start_point to __new__ — re-pick via Load "
            f"if needed"
        )
    elif sp_val == "__latest_export__":
        sp_entry["value"] = "__new__"
        log_warning(
            "[base_asset migrate] '__latest_export__' is no longer "
            "supported (semantics changed: Latest now means 'this canvas's "
            "last run'); reset start_point to __new__ — re-pick via "
            "Latest (once a run is produced) or Load"
        )
    else:
        # Unknown legacy token — leave start_point as-is to surface the
        # bad value rather than silently rewriting. v2 picker's
        # _resolve_start_point_label will display it verbatim.
        log_warning(
            f"[base_asset migrate] unrecognised legacy start_point "
            f"token {sp_val!r}; left as-is for user to fix"
        )
        return
    params.pop("asset_id", None)
    params.pop("checkpoint_file", None)


def _strip_alpha(hex_str: str) -> str:
    """``#RRGGBBAA`` / ``#RRGGBB`` → 大写 ``#RRGGBB``;非法值返回 ``#67CCF5`` 兜底。

    色盘组件 (ColorPickerPopup) 只接受 ``#RRGGBB`` 6 位 hex,不识别 alpha;
    主题色 slot (``canvas_group_title_bg`` 等) 是 8 位带 alpha 的,所以在
    把它们作为色盘"初始 hex"之前要把 alpha 砍掉。
    """
    raw = str(hex_str or "").strip().lstrip("#")
    if len(raw) >= 6:
        try:
            int(raw[:6], 16)
            return "#" + raw[:6].upper()
        except ValueError:
            pass
    return "#67CCF5"


class CanvasPage(QWidget):
    """Training Ground 画布页 / Training Ground canvas page.

    ``embedded=True`` is the mode used when CanvasPage is hosted inside
    MissionControlPanel's main_screen (chart overlay layered on top, mode
    switched between Mission Control / Training Canva). In that mode the
    page does NOT construct the floating Node Library button — MC's top
    row hosts the button instead — and exposes :attr:`node_library_card`
    + :meth:`set_node_library_open` so the external button can drive card
    visibility.

    ``embedded=False`` keeps the legacy standalone layout: floating Node
    Library button at (10,10) plus card below it.
    """

    # Per-canvas dirty state signal — filtered from the global DirtyTracker so
    # external toolbars only react when *this* canvas flips. Payload True =
    # unsaved edits exist; False = clean baseline.
    dirty_state_changed = pyqtSignal(bool)

    # Emitted when the floating AI Build button toggles. The host (_MainPanel)
    # owns the mask's geometry/z-order: the AI Build overlay is a main_panel
    # sibling of this page (NOT a viewport child) so its opacity effect can
    # composite against the canvas behind it, exactly like MissionControlPanel.
    # Payload True = show the mask (full-cover, raised above MC); False = hide.
    ai_build_overlay_toggled = pyqtSignal(bool)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._embedded = bool(embedded)
        # Whether the canvas is currently interactive (accepts spawn/edit).
        # Mission Control mode flips this to False so spawn-via-card is gated.
        self._interactive: bool = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scene = CanvasScene(self)
        self._view = CanvasView(self._scene, self)
        layout.addWidget(self._view, 1)

        # 实例化的 NodeItem 索引：instance_id → NodeItem / NoteItem / Placeholder
        # （node_id 是节点类型，instance_id 是画布上具体一个实例）
        self._instances: Dict[str, NodeItem] = {}
        # 视觉分组框索引：group_id → GroupItem。groups 不进 _instances，
        # 走自己一套增删 + undo 命令；序列化到 canvas JSON 顶层 ``groups`` 字段。
        self._groups: Dict[str, GroupItem] = {}
        # demo populate 幂等防呼叫两次
        self._demo_populated: bool = False

        # 画布与训练后端的绑定 / Canvas-to-backend bind.
        # ``None`` until first ``bind_backend()`` (new canvas) or
        # ``from_workflow_dict()`` (loaded file). Once set inside the same
        # CanvasPage instance, ``bind_backend(other_eid)`` raises — backend
        # is a first-class immutable property of the canvas. Loading a
        # different ``.canvas.json`` resets via ``from_workflow_dict`` (it
        # nulls the field before re-binding).
        self._backend_id: Optional[str] = None

        # 当前画布在 ProjectStore 中的 file_id（mission_panel 监听 canvas_param_changed
        # 信号时按 file_id 过滤，避免多 page 实例下的串扰）。MissionControlPanel
        # 在 _canvas_loaded_id 切换后调 page.set_file_id(fid) 写入；为空时表示
        # 未关联到具体磁盘文件（demo / 临时画布）。
        self._file_id: str = ""
        # 落盘目标 ProjectInfo + canvas name（save_to_project 的两个必需参数）。
        # mission_panel load 完画布后通过 set_save_target(project, name) 写入；
        # set_node_param(save=True) 用此对触发自动落盘。两者任一为空则跳过保存。
        self._save_project: Any = None
        self._save_name: str = ""

        # 内存剪贴板（Phase 2 复制粘贴；不走系统剪贴板，避免格式协商）
        self._clipboard: Optional[Dict[str, Any]] = None
        # 最近一次右键位置（用于 "Paste Here" 锚点）
        self._last_context_scene_pos: Optional[QPointF] = None

        # ---- Undo / Redo ----
        # 单画布一只 stack；`_in_undo` 在命令 redo / undo 期间置 True，
        # 上层 mutating 路径（spawn_node / ParamRow.set_value / NodeItem.mouseRelease）
        # 在 True 时跳过 push、走 raw 路径，避免无限递归。
        self._undo_stack = QUndoStack(self)
        self._in_undo: bool = False
        # 全局 DirtyTracker 接入：只在 ``load_from_file`` 成功后 set baseline；
        # ``save_to_file`` 成功后 refresh baseline；``clear_scene`` 清除追踪。
        # ``_current_path`` 为 None 时，本 page 不参与全局脏状态广播。
        self._current_path: Optional[str] = None
        self._undo_stack.indexChanged.connect(self._refresh_dirty_state)
        # 监听全局 DirtyTracker:仅当 path 匹配本 canvas 时转发到本 widget 的
        # dirty_state_changed 信号(外部 toolbar 用它驱动 Save 按钮 enable)。
        getDirtyTracker().dirtyChanged.connect(self._on_global_dirty_changed)
        # 节点拖动去抖：mousePress 时记录所有被拖节点的旧位置；
        # mouseRelease 时若实际位移非零，push 一条 MoveNodeCmd。
        self._pending_move: Optional[List[Tuple[str, QPointF]]] = None
        # NodeItem / ParamRow 通过 scene._page_ref 反查 page，避免在
        # 每个 item 上都挂一份引用。
        setattr(self._scene, "_page_ref", self)

        # 加载时若边引用 placeholder 节点（其端口对象不存在），把原始 edge dict
        # 缓存在这里；to_workflow_dict 时与 scene 中真实边一并 emit，确保
        # save / load 往返不丢数据。
        self._deferred_edges: List[Dict[str, Any]] = []

        # LOD 信号桥：view 跨 zoom tier 时通知 scene 调整背景层 LOD
        self._view.zoomChanged.connect(self._scene.on_zoom_tier_changed)

        # Phase 2: 接 scene 右键菜单 + 焦点策略（接收键盘）
        self._scene.set_context_menu_provider(self._build_context_menu)
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Node Library — 旧的画布悬浮按钮 + 卡片已被搬到 Sidebar 弹出面板
        # (`Sidebar.node_library`)。这里保留属性槽位以兼容旧代码路径，但不再
        # 实例化任何 overlay：spawn 路由改由 MainWindow 把 sidebar.node_requested
        # 桥接到 :meth:`spawn_node`。
        self._node_library: Optional[NodeLibraryOverlay] = None
        self._node_library_controller: Optional[NodeLibraryCardController] = None

        # 右下角 overlay 容器：仅 minimap（control_guide 独立浮于左下角）。
        # 单点定位整体（viewport 子 widget，固定像素位置）。
        self._overlay = QWidget(self._view.viewport())
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        overlay_layout = QVBoxLayout(self._overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(6)

        # AI Build 开关已迁到 MissionControlPanel 顶栏（tab 滑块左侧）；此右下角
        # overlay 现在只承载 minimap。

        # minimap（保留鼠标交互）
        self._minimap = CanvasMiniMap(self._view)
        self._minimap.setParent(self._overlay)
        overlay_layout.addWidget(self._minimap, 0, Qt.AlignmentFlag.AlignRight)
        self._view.horizontalScrollBar().valueChanged.connect(self._refresh_minimap)
        self._view.verticalScrollBar().valueChanged.connect(self._refresh_minimap)
        self._scene.changed.connect(lambda *_: self._refresh_minimap())

        # AI Build 大遮罩 —— **不是** viewport 子 widget。它由宿主 _MainPanel
        # reparent 成与本页平级的兄弟控件（画布在底层 widget，遮罩 raise 在上），
        # 这样遮罩的 QGraphicsOpacityEffect 才能和背后的画布合成透出，和
        # MissionControlPanel 的图表遮罩机制完全一致。viewport 子控件做不到。
        # 创建时挂在本页下并隐藏（避免成顶层窗口闪现），等 set_children 时由
        # _MainPanel 接管 parent + 几何 + 层级。
        self._ai_build_panel = AiBuildPanel(self)
        self._ai_build_panel.hide()
        # 收起由 Mission Control 顶栏的 "AI Coding" 标签驱动（切走该标签即
        # collapse_ai_build），面板自身不再有关闭 (×) 按钮。

        # Agent bridge —— the single seam between the (future) NL-orchestration
        # harness and this live canvas: applies agent mutations to the scene and
        # owns the pre-prompt checkpoint. Reset reverts the canvas; a submitted
        # prompt captures a fresh checkpoint first (so Reset can undo a bad run).
        self._ai_agent_bridge = CanvasAgentBridge(self, self._ai_build_panel)
        self._ai_build_panel.reset_requested.connect(
            self._ai_agent_bridge.reset_to_checkpoint
        )
        self._ai_build_panel.prompt_submitted.connect(self._on_ai_prompt_submitted)
        self._ai_build_panel.settings_requested.connect(self._open_ai_api_settings)
        self._ai_build_panel.pause_toggled.connect(self._on_ai_pause_toggled)
        self._ai_build_panel.retry_requested.connect(self._on_ai_retry_requested)
        # Configuration-source combobox: Origin + one numbered version per run
        # that changed the canvas. The store is the Qt-free producer; selecting
        # an entry applies that snapshot to the live canvas (Origin = revert).
        # Lazy import mirrors the other application.service imports here (page.py
        # can be imported very early, before application.service is ready).
        from application.service.ai_orchestration.version_store import (
            AiBuildVersionStore,
        )
        self._ai_versions = AiBuildVersionStore()
        self._ai_build_panel.config_source_changed.connect(
            self._on_ai_config_source_changed
        )

        self._overlay.adjustSize()
        self._overlay.show()
        self._overlay.raise_()
        self._reposition_overlay()

        # AI Build 开关已从画布右上角悬浮按钮迁移到 MissionControlPanel 顶栏
        # （tab 滑块左侧的 "AI: [SwitchButton]"，仅 Training Canva 模式可见）。
        # 这里只保留遮罩的"开/收"状态与对外的纯逻辑入口
        # ``set_ai_build_overlay_open`` / ``is_ai_build_open``，由 MC 顶栏开关驱动。
        # ``_top_overlay_inset`` 仍由宿主经 ``set_top_overlay_inset`` 下传（用于
        # 未来右上角 overlay 的避让），当前无悬浮控件需重定位。
        self._top_overlay_inset = 0
        self._ai_overlay_open: bool = False

        # 左下角 overlay 容器：control_guide。结构与右下 self._overlay 镜像，
        # 唯一区别是锚定在 viewport 左下角。viewport 子 widget → 不随 scene
        # pan/zoom 移动，也不会随窗口压缩被裁剪到画面外。
        self._overlay_left = QWidget(self._view.viewport())
        self._overlay_left.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        overlay_left_layout = QVBoxLayout(self._overlay_left)
        overlay_left_layout.setContentsMargins(0, 0, 0, 0)
        overlay_left_layout.setSpacing(0)

        # control_guide（纯展示，找不到 SVG 则跳过）
        self._control_guide: Optional[CanvasControlGuide] = self._build_control_guide()
        if self._control_guide is not None:
            self._control_guide.setParent(self._overlay_left)
            overlay_left_layout.addWidget(
                self._control_guide, 0, Qt.AlignmentFlag.AlignLeft
            )

        self._overlay_left.adjustSize()
        self._overlay_left.show()
        self._overlay_left.raise_()
        self._reposition_overlay_left()
        # QGraphicsView pan 走 viewport.scroll(dx,dy)，会把 viewport 的子
        # widget 一起平移；和 minimap 一样，必须在滚动条 valueChanged 上
        # 重新钉住绝对位置，否则会看起来像被印在画布上跟着相机走。
        self._view.horizontalScrollBar().valueChanged.connect(
            self._reposition_overlay_left
        )
        self._view.verticalScrollBar().valueChanged.connect(
            self._reposition_overlay_left
        )

        # 选区悬浮工具栏 —— 单实例,parent=viewport,跟随选区在最靠上 item 上方 20px。
        self._selection_toolbar = SelectionToolbar(self._view.viewport())
        self._selection_toolbar.group_clicked.connect(self._on_group_button_clicked)
        self._selection_toolbar.collapse_clicked.connect(self._toggle_collapse_selected)
        self._selection_toolbar.skip_clicked.connect(self._toggle_disable_selected)
        self._selection_toolbar.color_clicked.connect(self._on_selection_color_clicked)
        self._selection_toolbar.color_reset_clicked.connect(self._on_selection_color_reset)
        self._selection_toolbar.delete_clicked.connect(self._delete_selected)
        # Per-canvas snapshot of color-overrides captured the moment the picker
        # opens; cancel restores from this dict. Cleared when picker closes.
        self._color_picker_snapshot: Dict[str, str] = {}
        self._color_picker_popup: Optional[Any] = None
        self._scene.selectionChanged.connect(self._reposition_selection_toolbar)
        self._view.horizontalScrollBar().valueChanged.connect(
            self._reposition_selection_toolbar
        )
        self._view.verticalScrollBar().valueChanged.connect(
            self._reposition_selection_toolbar
        )
        # zoom 切档时 view 自己 emit zoomChanged(tier);此外 transform 任意变化都
        # 可能挪动 scene→view 映射 → resizeEvent 也走一遍。
        self._view.zoomChanged.connect(
            lambda *_: self._reposition_selection_toolbar()
        )

        # 语种切换 → 重译画布上节点缓存的标题（NodeItem._title_text 是
        # __init__ 时一次性 cache 的，否则会一直显示旧语种）。同时刷一遍
        # viewport 让背景 / overlay paint 路径中的 tr() 也走一次新 lang。
        I18n.instance().language_changed.connect(self._retranslate_canvas)

        log_debug(
            f"[ui.canvas] CanvasPage constructed (embedded={self._embedded})"
        )

    def _retranslate_canvas(self, *_args) -> None:
        scene = getattr(self, "_scene", None)
        if scene is not None:
            for it in scene.items():
                fn = getattr(it, "retranslate", None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            scene.update()
        view = getattr(self, "_view", None)
        if view is not None:
            view.viewport().update()
        for name in ("_minimap", "_control_guide"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.update()
                except Exception:
                    pass

    # ---- canvas overlay 重排（minimap 右下、control_guide 左下，各自独立）----

    _OVERLAY_MARGIN = 12         # minimap 距 viewport 右 / 下边距
    _CONTROL_GUIDE_MARGIN = 10   # control_guide 距 viewport 左 / 下边距
    _MINIMAP_WIDTH = 220         # CanvasMiniMap 固定宽度

    def is_ai_build_open(self) -> bool:
        """Whether the AI Build overlay is currently shown.

        The MissionControlPanel ``AI Coding`` tab mirrors this: selecting the
        tab opens the overlay, switching away (or leaving canvas mode)
        collapses it."""
        return bool(getattr(self, "_ai_overlay_open", False))

    def set_ai_build_overlay_open(self, open_: bool) -> bool:
        """Open / collapse the AI Build overlay; return the resulting state.

        The single entry point that the MissionControlPanel ``AI Coding`` tab
        (and :meth:`collapse_ai_build`) drive. Geometry / z-order is the host's
        job (``_MainPanel`` listens to :attr:`ai_build_overlay_toggled`): the
        mask is a main_panel sibling of this page so its opacity effect reveals
        the canvas like MC.

        Opening gates on a saved LLM connection (opens the settings dialog if
        none yet). If still unconfigured, the overlay stays collapsed and we
        emit ``ai_build_overlay_toggled(False)`` so the tab selection reverts.
        The return value is the actual open-state so a direct caller can react
        too.
        """
        panel = getattr(self, "_ai_build_panel", None)
        if panel is None:
            return False
        if not bool(open_):
            if self._ai_overlay_open:
                self._ai_overlay_open = False
                self.ai_build_overlay_toggled.emit(False)
            return False
        # Entry gate: AI Build needs a saved LLM connection.
        from application.service.ai_orchestration.api_config import is_api_configured

        if not is_api_configured() and not self._open_ai_api_settings():
            # Refused — make sure we read as closed and tell the switch to revert.
            self._ai_overlay_open = False
            self.ai_build_overlay_toggled.emit(False)
            return False
        # Bind the overlay to this canvas's per-canvas conversation threads, then
        # ask the host to show it (full-cover, raised above MC). The signal is
        # delivered synchronously, so the mask is visible before focus_input.
        panel.set_canvas_id(self.file_id)
        # Reflect the active agent's model on the top-row button before showing.
        panel.refresh_agent_button()
        # Feed the run-target banner / config context with this canvas's robot
        # + backend (the rest of the banner is BACKEND TODO — see the panel).
        panel.set_requirements_context(self._backend_id or "", self.robot_id or "")
        # Fill the Configuration sections from the LIVE canvas right away, so the
        # panel reflects the real config (not an empty skeleton) before any run.
        self._push_ai_config_preview()
        self._ai_overlay_open = True
        self.ai_build_overlay_toggled.emit(True)
        panel.focus_input()
        return True

    def collapse_ai_build(self) -> None:
        """Collapse the AI Build overlay (mirrored onto the MC ``AI Coding`` tab).

        Driven by the host when leaving canvas mode (and when the user switches
        away from the AI Coding tab). Emits ``ai_build_overlay_toggled(False)``
        so the host hides the mask and the tab selection reverts."""
        self.set_ai_build_overlay_open(False)

    @property
    def ai_build_panel(self) -> "AiBuildPanel":
        """The AI Build mask. The host reparents it to ``_MainPanel`` (sibling
        of this page) so its opacity composites against the canvas, like MC."""
        return self._ai_build_panel

    def _open_ai_api_settings(self) -> bool:
        """Open the API settings dialog (gear button + entry gate).

        Returns True if a usable connection is configured after the dialog
        closes, so the entry gate can decide whether to proceed.
        """
        from application.service.ai_orchestration.api_config import is_api_configured
        from application.ui.dialogs.ai_api_settings_dialog import AiApiSettingsDialog

        dialog = AiApiSettingsDialog(self)
        dialog.exec()
        # The default agent (and its model) may have changed — re-label the
        # top-row agent button so it reflects the active connection.
        panel = getattr(self, "_ai_build_panel", None)
        if panel is not None:
            panel.refresh_agent_button()
        return is_api_configured()

    @property
    def ai_agent_bridge(self) -> CanvasAgentBridge:
        """The seam the NL-orchestration harness drives the canvas through."""
        return self._ai_agent_bridge

    def ai_build_report_progress(self, fraction: float, text: str = "") -> None:
        """GUI-thread hook the marshalling sink forwards run progress to.

        Drives the overlay's progress bar from the run's real ``set_progress``
        checkpoints (connect → run → save → done)."""
        panel = getattr(self, "_ai_build_panel", None)
        if panel is not None:
            panel.set_progress(fraction, text)

    def ai_build_begin_phase(self, phase_id: str, title: str) -> None:
        """Forward a structured-phase start to the overlay (marshaller → GUI)."""
        panel = getattr(self, "_ai_build_panel", None)
        if panel is not None:
            panel.begin_phase(phase_id, title)

    def ai_build_on_engaged(self) -> None:
        """GUI-thread hook: the run passed the silent intent gate and WILL act
        on the canvas. Only now surface the checkpoint notice — a gate-declined
        direct answer never shows build affordances (the checkpoint itself was
        captured before the task was submitted)."""
        panel = getattr(self, "_ai_build_panel", None)
        if panel is not None:
            panel.append_markup(
                tr(
                    "canvas.ai_build.checkpoint_made",
                    "[[success:Checkpoint saved — Reset reverts the canvas to this point.]]",
                )
            )

    def ai_build_set_phase_tokens(self, phase_total: int, cumulative: int) -> None:
        """Forward per-phase / cumulative token counts to the overlay."""
        panel = getattr(self, "_ai_build_panel", None)
        if panel is not None:
            panel.set_phase_tokens(phase_total, cumulative)

    def _on_ai_pause_toggled(self, paused: bool) -> None:
        """Pause / resume the running AI Build task at its next phase boundary."""
        task = getattr(self, "_ai_build_task", None)
        if task is None:
            return
        if paused:
            task.request_pause()
        else:
            task.request_resume()

    def _on_ai_retry_requested(self) -> None:
        """Re-run the last prompt from the pre-prompt checkpoint (whole-run retry).

        Ignored while a run is in flight (Pause/cancel that first). Reverts the
        canvas to the checkpoint, then re-submits the same prompt."""
        if getattr(self, "_ai_build_task_id", None):
            self._ai_build_panel.append_markup(
                tr(
                    "canvas.ai_build.retry_busy",
                    "[[warning:A run is in progress — pause or wait before retrying.]]",
                )
            )
            return
        last = str(getattr(self, "_ai_last_prompt", "") or "").strip()
        if not last:
            self._ai_build_panel.append_markup(
                tr(
                    "canvas.ai_build.retry_none",
                    "[[warning:Nothing to retry yet — send a prompt first.]]",
                )
            )
            return
        if self._ai_agent_bridge.has_checkpoint():
            self._ai_agent_bridge.reset_to_checkpoint()
        self._on_ai_prompt_submitted(last)

    def _on_ai_prompt_submitted(self, prompt: str) -> None:
        """User submitted an AI Build prompt — dispatch the orchestration harness.

        The panel has already recorded the user message bubble; here we capture a
        whole-canvas checkpoint BEFORE anything acts on the canvas (so Reset can
        revert a bad run), then run the harness on a worker thread reading the
        panel's run settings (max repair rounds / auto-save). The harness opens
        with a silent intent gate: a prompt that is not a canvas build/modify
        request is answered directly from the model's own knowledge (no canvas
        writes, nothing committed); only a gated-in build run calls back
        :meth:`ai_build_on_engaged` and proceeds to mutate. Every canvas write
        the harness performs is marshalled back to this GUI thread by the
        :class:`BridgeMutationSink`, which renders the live C4 event stream into
        the agent bubble and drives the progress bar. Fail-loud: the task reports
        a configuration / transport failure and is marked failed; a bad run is
        never auto-persisted (§8).
        """
        prompt = str(prompt or "").strip()
        if not prompt:
            return
        if getattr(self, "_ai_build_task_id", None):
            # A run is already in flight — do not race two harnesses on one canvas.
            self._ai_build_panel.append_markup(
                tr(
                    "canvas.ai_build.run_busy",
                    "[[warning:An AI Build run is already in progress — wait for it "
                    "to finish.]]",
                )
            )
            return

        self._ai_agent_bridge.checkpoint()
        # Capture "Origin" (the pre-first-run canvas) once, before any agent edit,
        # so the Configuration combobox can always offer a manual revert target.
        try:
            self._ai_versions.set_origin_if_absent(self.to_workflow_dict())
        except Exception as exc:  # snapshot is the canvas's own contract; non-fatal
            log_warning(f"[ui.canvas] AI Build origin snapshot failed: {exc!r}")
        self._ai_build_panel.begin_progress()
        # The checkpoint NOTICE is deferred to ai_build_on_engaged(): the run
        # first passes the silent intent gate, and a gate-declined prompt is
        # answered directly without ever acting on the canvas — showing build
        # affordances for it would be noise. The checkpoint itself is captured
        # above regardless (it must precede any possible mutation).

        # Build the GUI-thread marshalling sink + the worker task and submit,
        # carrying the panel's run settings.
        from unitport_sdk import get_task_signal, get_tasks_manager

        from application.service.ai_orchestration.harness.run_task import AiBuildRunTask
        from .ai_build_bridge_marshaller import BridgeMutationSink

        if not getattr(self, "_ai_build_signal_wired", False):
            get_task_signal().task_finished.connect(self._on_ai_build_task_finished)
            self._ai_build_signal_wired = True

        # Remember the prompt so Retry can re-run it from the checkpoint.
        self._ai_last_prompt = prompt

        # Keep a reference to the sink so it is not garbage-collected mid-run.
        self._ai_build_sink = BridgeMutationSink(self)
        task = AiBuildRunTask(
            self._ai_build_sink,
            prompt,
            max_repair_rounds=self._ai_build_panel.max_repair_rounds(),
            auto_commit=self._ai_build_panel.auto_commit(),
            requirements=self._ai_build_panel.requirements(),
            # In-chat history of the active thread (captured NOW, on the GUI
            # thread, before the worker starts) — the gate / direct answer /
            # Understanding phase keep conversational referents across turns.
            history=self._ai_build_panel.llm_conversation_history(),
        )
        # Keep a reference so the Pause button can reach the running task.
        self._ai_build_task = task
        self._ai_build_task_id = get_tasks_manager().submit(task)

    def _on_ai_build_task_finished(self, task_id: str, success: bool, result: object) -> None:
        """Reset the in-flight flag when our AI Build task ends (success or not).

        The task already logged the specific outcome (saved / blocking issues /
        API failure) into the panel; this clears the run guard, releases the sink
        reference, tops off the progress bar and persists the agent turn."""
        if task_id != getattr(self, "_ai_build_task_id", None):
            return
        self._ai_build_task_id = None
        self._ai_build_sink = None
        self._ai_build_task = None
        if not success:
            self._ai_build_panel.append_markup(
                tr(
                    "canvas.ai_build.run_failed",
                    "[[error:AI Build run failed — see the messages above.]]",
                )
            )
        # If the agent changed the canvas at all, publish a new Configuration
        # version into the dropdown (regardless of success — the live canvas was
        # mutated either way; a no-op run adds nothing).
        self._publish_ai_version()
        self._ai_build_panel.finish_progress()

    def _publish_ai_version(self) -> None:
        """Record a Configuration version for the just-finished run and refresh
        the source combobox. A run that changed nothing adds no version (no
        fabricated entry, §8)."""
        try:
            canvas = self.to_workflow_dict()
        except Exception as exc:  # serialization is the canvas's own contract
            log_warning(f"[ui.canvas] AI Build version snapshot failed: {exc!r}")
            return
        # Reflect the final post-run canvas in the Configuration sections.
        self._push_ai_config_preview()
        next_n = len(self._ai_versions.sources()) + 1
        prompt = str(getattr(self, "_ai_last_prompt", "") or "").strip()
        hint = prompt.splitlines()[0][:24].strip() if prompt else ""
        label = f"v{next_n} · {hint}" if hint else f"v{next_n}"
        version = self._ai_versions.record(canvas, label=label)
        if version is None:
            return  # nothing changed → no new version
        self._ai_build_panel.set_config_sources(
            self._ai_versions.sources(), select=version.key
        )
        msg = tr(
            "canvas.ai_build.version_saved",
            "[[success:Saved as config version {v} — pick it (or Origin to revert) "
            "in the Configuration list.]]",
        )
        try:
            msg = msg.format(v=version.key)
        except (KeyError, IndexError):
            msg = f"[[success:Saved as config version {version.key}.]]"
        self._ai_build_panel.append_markup(msg)

    def _push_ai_config_preview(self) -> None:
        """Project the live canvas into the panel's Configuration sections.

        Called on overlay open, after every agent mutation (debounced), and after
        a run / version switch — so the sections mirror the real config and follow
        the agent's edits in real time. Best-effort + cosmetic (never raises)."""
        panel = getattr(self, "_ai_build_panel", None)
        if panel is None:
            return
        try:
            from application.service.ai_orchestration.config_preview import (
                project_config_sections,
            )
            sections = project_config_sections(self.to_workflow_dict())
        except Exception as exc:  # cosmetic preview must not break anything
            log_warning(f"[ui.canvas] AI Build config preview failed: {exc!r}")
            return
        panel.set_config_preview(sections)

    def ai_build_schedule_config_refresh(self) -> None:
        """Coalesce rapid agent mutations into one Configuration-preview refresh.

        The agent applies many primitives per run; a single-shot debounce avoids
        rebuilding the sections (and flickering) on every one while still tracking
        the canvas in near-real-time. Called by the bridge after each mutation."""
        timer = getattr(self, "_ai_cfg_timer", None)
        if timer is None:
            from PyQt6.QtCore import QTimer

            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(80)
            timer.timeout.connect(self._push_ai_config_preview)
            self._ai_cfg_timer = timer
        timer.start()  # restart the debounce window on each mutation

    def _on_ai_config_source_changed(self, key: str) -> None:
        """User picked a Configuration source: apply that snapshot to the live
        canvas (Origin = manual revert to the pre-run canvas). Direct + immediate
        + persisted, per the AI Build design."""
        if getattr(self, "_ai_build_task_id", None):
            self._ai_build_panel.append_markup(
                tr(
                    "canvas.ai_build.version_busy",
                    "[[warning:A run is in progress — wait before switching config "
                    "versions.]]",
                )
            )
            return
        snap = self._ai_versions.snapshot_for(str(key or "origin"))
        if snap is None:
            return
        try:
            # snapshot_for already returns a deep copy — safe to hand to the loader.
            self.from_workflow_dict(snap)
        except Exception as exc:  # a bad snapshot must not wedge the editor
            log_warning(f"[ui.canvas] AI Build apply config {key!r} failed: {exc!r}")
            return
        self.commit_to_save_target()  # zero-lag sync + persist (best-effort save)
        self._push_ai_config_preview()  # sections mirror the applied version

    def _build_control_guide(self) -> Optional[CanvasControlGuide]:
        path = Assets.find_icon("control_guide")
        if path is None:
            return None
        return CanvasControlGuide(self._view, path)

    def _reposition_overlay(self) -> None:
        overlay = getattr(self, "_overlay", None)
        if overlay is None:
            return
        overlay.adjustSize()
        margin = self._OVERLAY_MARGIN
        vp = self._view.viewport()
        sz = overlay.size()
        overlay.move(
            max(0, vp.width() - sz.width() - margin),
            max(0, vp.height() - sz.height() - margin),
        )
        overlay.raise_()

    def _reposition_overlay_left(self) -> None:
        overlay = getattr(self, "_overlay_left", None)
        if overlay is None:
            return
        overlay.adjustSize()
        margin = self._CONTROL_GUIDE_MARGIN
        vp = self._view.viewport()
        sz = overlay.size()
        overlay.move(
            margin,
            max(0, vp.height() - sz.height() - margin),
        )
        overlay.raise_()

    def set_top_overlay_inset(self, px: int) -> None:
        """Host hook: height of the overlay strip covering the canvas top.

        MainWindow passes the MissionControlPanel top_row height. The floating
        AI Build button that used to avoid this strip has moved into the MC
        top_row (the "AI" switch), so there is currently no top-right viewport
        overlay to reposition — the value is stored for any future top-right
        affordance and to keep the host's ``_relayout`` call a harmless no-op.
        The AI Build mask itself is a main_panel sibling raised above MC, so it
        covers the strip and needs no inset (handled by ``_MainPanel``)."""
        self._top_overlay_inset = max(0, int(px))

    def _refresh_minimap(self) -> None:
        self._reposition_overlay()
        if getattr(self, "_minimap", None) is not None:
            self._minimap.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlay()
        self._reposition_overlay_left()
        # 选区工具栏跟 viewport 几何 → 窗口尺寸变化时一并重定位
        if getattr(self, "_selection_toolbar", None) is not None:
            self._reposition_selection_toolbar()

    # ---- 公开访问 ----

    @property
    def scene(self) -> CanvasScene:
        return self._scene

    @property
    def view(self) -> CanvasView:
        return self._view

    def instances(self) -> Dict[str, NodeItem]:
        """返回当前画布上所有 NodeItem 实例的浅拷贝映射."""
        return dict(self._instances)

    # ---- backend bind（画布的一等不可改属性）----

    @property
    def backend_id(self) -> Optional[str]:
        """Currently bound training-backend id (e.g. ``"isaac_lab"``).

        ``None`` for a fresh, never-bound CanvasPage (newly constructed,
        not yet loaded or assigned). Read-only; mutate via
        :meth:`bind_backend` exactly once per canvas lifetime.
        """
        return self._backend_id

    @property
    def robot_id(self) -> Optional[str]:
        """SKU of the robot selected on this canvas, or ``None``.

        Derived live from the (single) ``robot`` schema node's ``asset_id``
        parameter: canvas top-level ``robot_id`` is a function of the Robot
        node state, guaranteeing 100% sync at every serialization point —
        users edit ``asset_id`` only, and the top-level key is recomputed
        on each :meth:`to_workflow_dict` call. Resolution uses the shared
        ``registers.robots.resolve_to_sku`` helper (SKU / alias / name /
        ``brand.model``).

        Returns ``None`` when:
            * No ``robot`` node exists on the canvas
            * The ``robot`` node is a placeholder (manifest missing)
            * ``asset_id`` is empty or fails to resolve to a registered SKU

        Multiple ``robot`` nodes on one canvas: the first encountered wins
        and a warning is logged — keep the canvas to one robot node.
        """
        first_robot = None
        extras: List[str] = []
        for inst_id, node_item in self._instances.items():
            if isinstance(node_item, PlaceholderNodeItem):
                continue
            if getattr(node_item.manifest, "id", None) == "robot":
                if first_robot is None:
                    first_robot = node_item
                else:
                    extras.append(inst_id)
        if first_robot is None:
            return None
        if extras:
            log_warning(
                f"[ui.canvas] robot_id: canvas has multiple robot nodes "
                f"(picked first; ignored {extras!r}); top-level robot_id "
                f"reflects only the first robot node"
            )
        raw_id = str(first_robot.params.get("asset_id", "") or "").strip()
        if not raw_id:
            return None
        sku = robots_registry.resolve_to_sku(raw_id)
        if sku is None:
            log_warning(
                f"[ui.canvas] robot_id: asset_id {raw_id!r} on robot node "
                f"did not resolve to a registered SKU; top-level robot_id "
                f"will be omitted on save"
            )
        return sku

    def bind_backend(self, engine_id: str) -> None:
        """Bind the canvas to a training-backend id (immutable once set).

        Raises:
            ValueError: ``engine_id`` is empty / not a string.
            RuntimeError: this CanvasPage was already bound to a different
                ``engine_id``. To switch backends, load a different
                ``.canvas.json`` (``from_workflow_dict`` resets the bind
                before re-binding to the file's declared backend).
        """
        if not isinstance(engine_id, str) or not engine_id.strip():
            raise ValueError("bind_backend: engine_id must be a non-empty string")
        eid = engine_id.strip()
        if self._backend_id is not None and self._backend_id != eid:
            raise RuntimeError(
                f"bind_backend: canvas backend is immutable once bound "
                f"(current={self._backend_id!r}, requested={eid!r})"
            )
        self._backend_id = eid
        # Sync every existing NodeItem's params['backend'] to the canvas-bound
        # kind. Rewards / terminations / domain_rand no longer expose backend
        # as a user-editable ParamSpec; their conditional_on visibility reads
        # from this injected key.
        kind = backends_registry.node_backend_kind(eid)
        for item in self._instances.values():
            try:
                item.apply_canvas_backend(kind)
            except Exception:
                pass

    # ---- file_id 关联（ProjectStore 维护；用于 canvas_param_changed 过滤） ----

    @property
    def file_id(self) -> str:
        """ProjectStore 维护的 canvas file id；空串 = 未关联磁盘文件."""
        return self._file_id

    def set_file_id(self, fid: str) -> None:
        """关联画布到 ProjectStore 的 file_id。MissionControlPanel 在
        ``_canvas_loaded_id`` 改变时调用；后续 ``canvas_param_changed`` emit
        会携带此 file_id，订阅方据此过滤多 page 串扰.

        关联完成后 emit 一次 ``canvas_topology_changed`` 让订阅方做首次刷新
        （此时 _instances 已就绪，从外部 set_canvas 链路触发）。"""
        self._file_id = str(fid or "")
        # AI Build 对话线程按画布隔离：切画布时让浮层重新载入该画布的线程
        # （浮层打开时立即可见；关闭时也无害，下次打开即对的上）。
        panel = getattr(self, "_ai_build_panel", None)
        if panel is not None:
            panel.set_canvas_id(self._file_id)
        self._emit_topology_changed()

    def _emit_topology_changed(self) -> None:
        """节点增 / 减 / 切画布时广播；批量加载（``_batch_load=True``，由
        ``_suspend_undo`` 设置）期间静默 — 加载完成后调用方会手动 emit 一次。
        单命令 redo / undo 不抑制：用户 spawn / despawn 必须立即广播。"""
        if getattr(self, "_batch_load", False):
            return
        try:
            from application.service.signals import get_app_signals
            get_app_signals().canvas_topology_changed.emit(self._file_id)
        except Exception:
            pass
        # 拓扑也是一种编辑（spawn/despawn/clear），与 undo_stack.indexChanged 互补
        # 覆盖：indexChanged 抓 push/undo/redo，本路径抓 raw 路径 + bind_backend。
        self._refresh_dirty_state()

    def _emit_edge_changed(self) -> None:
        """边建立 / 断开时广播（ConnectionItem → scene.notify_edge_changed）.

        批量加载期间静默（与 ``_emit_topology_changed`` 同一约定）——
        from_workflow_dict 收尾的 topology_changed 兜底触发订阅方刷新。
        订阅方：CommandContractModel（契约预览随接线实时重算）。
        """
        if getattr(self, "_batch_load", False):
            return
        try:
            from application.service.signals import get_app_signals
            get_app_signals().canvas_edge_changed.emit(self._file_id)
        except Exception:
            pass

    # ---- DirtyTracker 接入（全局"当前画布文件"脏状态广播） ----

    def _content_fingerprint(self) -> str:
        """画布内容指纹（用于 hash 比对）.

        基于 ``to_workflow_dict()`` 序列化为稳定 JSON。**剔除** ``metadata.view``
        （zoom / center）—— 缩放和平移不是编辑，不应被判脏。
        """
        data = self.to_workflow_dict()
        md = data.get("metadata")
        if isinstance(md, dict):
            data = dict(data)
            new_md = {k: v for k, v in md.items() if k != "view"}
            if new_md:
                data["metadata"] = new_md
            else:
                data.pop("metadata", None)
        return json.dumps(
            data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    def _refresh_dirty_state(self) -> None:
        """把当前画布指纹推给全局 DirtyTracker；仅在本 page 是 tracker 当前所有者
        且未处于批量加载时生效。"""
        if self._current_path is None:
            return
        if getattr(self, "_batch_load", False):
            return
        try:
            tracker = getDirtyTracker()
        except Exception:
            return
        if tracker.getCurrentPath() != self._current_path:
            return
        try:
            reportDirtyText(self._content_fingerprint())
        except Exception as exc:
            log_warning(f"[ui.canvas] dirty report failed: {exc!r}")

    def is_dirty(self) -> bool:
        """供 UI 组件查询当前画布是否脏 / Return True if the loaded canvas has
        unsaved edits (hash != baseline). Returns False when no canvas file is
        loaded, or when another CanvasPage currently owns the global tracker.
        """
        if self._current_path is None:
            return False
        try:
            tracker = getDirtyTracker()
        except Exception:
            return False
        if tracker.getCurrentPath() != self._current_path:
            return False
        return tracker.isDirty()

    def _on_global_dirty_changed(self, path: str, is_dirty: bool) -> None:
        """全局 DirtyTracker 任意 canvas 翻转 → 仅本 canvas 路径匹配时
        转发到 ``dirty_state_changed`` 信号。"""
        if path and self._current_path and path == self._current_path:
            self.dirty_state_changed.emit(bool(is_dirty))

    @property
    def undo_stack(self) -> QUndoStack:
        """Expose the QUndoStack so external toolbars can wire
        ``canUndoChanged`` / ``canRedoChanged`` and trigger ``undo()`` /
        ``redo()`` directly."""
        return self._undo_stack

    def set_save_target(self, project: Any, name: str) -> None:
        """记录跨 widget 写入后自动落盘所需的 (project, name) 上下文。
        传入 None / 空串则禁用自动落盘。"""
        self._save_project = project
        self._save_name = str(name or "")

    # ---- 跨 widget 节点参数读写 API（mission_panel ↔ canvas 双向同步入口） ----

    def get_node_param(self, instance_id: str, key: str, default: Any = None) -> Any:
        """读取节点参数；缺节点 / 缺 key 时返回 default."""
        item = self._instances.get(instance_id)
        if item is None:
            return default
        if key not in item.params:
            return default
        return item.params[key]

    def set_node_param(
        self,
        instance_id: str,
        key: str,
        value: Any,
        *,
        save: bool = True,
    ) -> bool:
        """跨 widget 写入节点参数 —— 与画布双向同步入口.

        - 走 ``ParamChangeCmd``（push 进 ``_undo_stack``，所以 mission_panel 的编辑
          也能 Ctrl+Z 回退）；
        - 写入后 ``_set_param_raw`` 自动 emit ``canvas_param_changed``；
        - ``save=True`` 时尝试落盘（仅当本 page 已关联到当前 project + file_id 时
          可解析为目标路径；否则跳过——非项目 file 暂存不主动落盘）。

        Returns ``True`` 当节点存在并完成写入；找不到节点返回 ``False``。
        """
        item = self._instances.get(instance_id)
        if item is None:
            return False
        old = item.params.get(key)
        if old == value:
            return True
        # 走标准 ParamChangeCmd 路径：自动 push 到 undo stack，redo 立即生效。
        try:
            from .undo import ParamChangeCmd
            cmd = ParamChangeCmd(self, instance_id, str(key), old, value)
            self._undo_stack.push(cmd)
        except Exception:
            # 兜底：直接走 raw（无 undo 记录），仍 emit canvas_param_changed.
            self._set_param_raw(instance_id, str(key), value)

        if save:
            self._maybe_save_after_edit()
        return True

    def find_first_node_by_schema(self, schema_id: str) -> Optional[str]:
        """返回第一个 ``schema_id`` 匹配的节点 instance_id；找不到返回 ``None``。

        兼容两种节点类：
        * ``NodeItem`` —— schema_id 来自 ``manifest.id``；
        * ``PlaceholderNodeItem`` —— 直接挂在 ``schema_id`` 属性上。

        mission_panel 用 ``find_first_node_by_schema("algorithm_config")`` 拿主超参
        节点；画布约定：每个画布最多只有一个同 schema 的 trainer 节点。
        """
        sid = str(schema_id or "")
        if not sid:
            return None
        for inst_id, item in self._instances.items():
            try:
                # NodeItem path: manifest.id
                manifest = getattr(item, "manifest", None)
                if manifest is not None:
                    if str(getattr(manifest, "id", "") or "") == sid:
                        return inst_id
                # PlaceholderNodeItem path: schema_id
                if str(getattr(item, "schema_id", "") or "") == sid:
                    return inst_id
            except Exception:
                continue
        return None

    def _maybe_save_after_edit(self) -> None:
        """跨 widget 写入后尝试落盘到 ``set_save_target`` 配置的目标；
        失败 log_warning，永不抛。"""
        if self._save_project is None or not self._save_name:
            return
        try:
            self.save_to_project(self._save_project, self._save_name)
        except Exception as exc:
            log_warning(f"[ui.canvas] _maybe_save_after_edit: skip ({exc!r})")

    def commit_to_save_target(self) -> Optional[Path]:
        """Persist the canvas to the configured ``set_save_target`` target.

        The AI Build harness's deterministic commit (blueprint §6) calls this on
        the GUI thread after a clean compile. Returns the saved path, or ``None``
        when no project target is set (an unsaved scratch canvas — the user must
        save manually; surfaced as a warning, not a silent success, §8).
        """
        if self._save_project is None or not self._save_name:
            log_warning(
                "[ui.canvas] commit_to_save_target: no project save target set; "
                "canvas not auto-saved"
            )
            return None
        return self.save_to_project(self._save_project, self._save_name)

    # ---- 节点 spawn / connect ----

    def spawn_node(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        instance_id: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Optional[NodeItem]:
        """User-facing spawn entry — pushes ``SpawnNodeCmd`` when not in undo replay.

        外部调用者（NodeLibrary 拖入、右键菜单 Add Node、测试）经此进入。
        ``_in_undo=True`` 时（命令 redo / from_workflow_dict 加载 / populate_demo
        guard）旁路 stack，直接走 ``_spawn_node_raw``。

        已绑后端的画布拒绝跨后端节点（``manifest.backends`` 谓词）——palette /
        右键菜单在上游已过滤，这里兜底陈旧拖拽与程序化调用；文件加载 / undo
        replay 走 ``_spawn_node_raw``，不受影响（旧画布照常打开，编译期归
        spec_validator 把关）。
        """
        if self._in_undo:
            return self._spawn_node_raw(node_id, x, y, instance_id, params)
        if self._backend_id is not None:
            from registers import nodes as nodes_registry
            m = nodes_registry.get_manifest(node_id)
            if m is not None and not m.applies_to_backend(self._backend_id):
                log_error(
                    f"[ui.canvas] spawn_node: node '{node_id}' is not available "
                    f"for backend '{self._backend_id}' "
                    f"(manifest.backends={list(m.backends)!r}); refused"
                )
                return None
        from .undo import SpawnNodeCmd
        cmd = SpawnNodeCmd(self, node_id, x, y, params=params, instance_id=instance_id)
        self._undo_stack.push(cmd)
        # SpawnNodeCmd.redo 已锁定 instance_id；对外返回 NodeItem
        locked = getattr(cmd, "_instance_id", None)
        return self._instances.get(locked) if locked else None

    def _spawn_node_raw(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        instance_id: Optional[str] = None,
        params: Optional[dict] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Optional[NodeItem]:
        """Raw spawn — no undo bookkeeping. 内部 / 加载 / 命令 redo 共用.

        ``width`` / ``height`` 仅在 Note 节点路径生效(NoteItem 支持用户 resize,
        需要从 ui.width / ui.height 恢复)。其他 NodeItem 由 manifest 决定布局,
        不接受外部 size 覆盖。
        """
        from registers import nodes as nodes_registry

        manifest = nodes_registry.get_manifest(node_id)
        if manifest is None:
            log_warning(f"[ui.canvas] spawn_node: 未知节点 id '{node_id}'")
            return None

        if instance_id is None:
            # Gap-safe auto-id: a count-based ``{id}_{n+1}`` COLLIDES whenever the
            # existing ids have a gap (e.g. ``rewards_2`` present but ``rewards_1``
            # removed → count=1 → ``rewards_2`` again → false "duplicate"). Multi-
            # instance nodes (rewards, …) have no uniqueness limit; increment past
            # any taken id so a legitimate placement never gets rejected.
            n = sum(1 for k in self._instances if k.startswith(f"{node_id}_"))
            instance_id = f"{node_id}_{n + 1}"
            while instance_id in self._instances:
                n += 1
                instance_id = f"{node_id}_{n + 1}"
        if instance_id in self._instances:
            # Only reachable now when the CALLER passed an explicit, already-taken
            # id (a real conflict) — auto-ids are guaranteed free above.
            log_warning(
                f"[ui.canvas] spawn_node: instance_id '{instance_id}' 已存在；拒绝重复创建"
            )
            return None

        # Manifest-default backfill: the 2026-05-11 contract treats the
        # manifest as the single source of truth for parameter defaults.
        # We merge declared defaults UNDER any caller-provided params so
        # that NodeItem (and downstream consumers — to_workflow_dict,
        # env_cfg compiler, …) always sees a complete param dict. Older
        # canvas files that omit newly-added manifest fields still load
        # cleanly because the missing keys get filled here, not silently
        # defaulted inside the compiler.
        canvas_params: Dict[str, Any] = dict(params or {})
        merged_params: Dict[str, Any] = {}
        for spec in manifest.parameters:
            if spec.default is not None:
                merged_params[spec.key] = spec.default
        merged_params.update(canvas_params)

        # Note 节点走 NoteItem 路径：自有 inline 文本编辑;无端口;无 backend tint。
        # 鸭子兼容 NodeItem 关键字段(.node_id/.manifest/.params/.pos()/.width
        # /.height/.iter_ports() 返回空),page._instances 与 to_workflow_dict
        # 现有循环对其透明。
        if manifest.id == "note":
            note = NoteItem(
                manifest,
                instance_id,
                params=merged_params,
                width=width,
                height=height,
            )
            note.setPos(x, y)
            self._scene.addItem(note)
            self._instances[instance_id] = note  # type: ignore[assignment]
            log_debug(
                f"[ui.canvas] spawned note '{instance_id}' at ({x:.0f}, {y:.0f})"
            )
            self._emit_topology_changed()
            return note  # type: ignore[return-value]

        item = NodeItem(manifest, instance_id, params=merged_params)
        item.setPos(x, y)
        self._scene.addItem(item)
        self._instances[instance_id] = item
        if self._backend_id is not None:
            try:
                item.apply_canvas_backend(
                    backends_registry.node_backend_kind(self._backend_id)
                )
            except Exception:
                pass
        log_debug(f"[ui.canvas] spawned {node_id} as '{instance_id}' at ({x:.0f}, {y:.0f})")
        self._emit_topology_changed()
        return item

    def connect(
        self,
        src_instance_id: str,
        src_port_name: str,
        dst_instance_id: str,
        dst_port_name: str,
    ) -> Optional[ConnectionItem]:
        """Raw 连接入口（非 undo-aware）—— 保留现有内部调用语义.

        用户拖拽建边走 ``PortDragController.release`` + scene 释放路径，
        在 scene 那侧 push ``ConnectCmd(already_created=True)``；
        本方法只服务于 ``_apply_snapshot_raw`` 等内部场景。
        """
        return self._connect_raw(src_instance_id, src_port_name,
                                 dst_instance_id, dst_port_name)

    def _connect_raw(
        self,
        src_instance_id: str,
        src_port_name: str,
        dst_instance_id: str,
        dst_port_name: str,
        edge_id: Optional[str] = None,
    ) -> Optional[ConnectionItem]:
        src_node = self._instances.get(src_instance_id)
        dst_node = self._instances.get(dst_instance_id)
        if src_node is None or dst_node is None:
            log_warning(
                f"[ui.canvas] connect: 缺节点 src={src_instance_id} dst={dst_instance_id}"
            )
            return None
        src_port = src_node.port(src_port_name, "out") or src_node.port(src_port_name, "in")
        dst_port = dst_node.port(dst_port_name, "in") or dst_node.port(dst_port_name, "out")
        if src_port is None or dst_port is None:
            log_warning(
                f"[ui.canvas] connect: 缺端口 src={src_port_name} dst={dst_port_name}"
            )
            return None
        try:
            edge = connect_ports(src_port, dst_port, edge_id=edge_id)
        except ValueError as e:
            log_warning(f"[ui.canvas] connect: {e}")
            return None
        self._scene.addItem(edge)
        # Canvas-derived-keys rule: ActorSetting.init_joint_angles is a
        # field whose KEY SET is derived from the upstream Robot Node.
        # Connecting a robot_pipe to an actor_setting's input port
        # therefore must re-populate that field. Same goes for a fresh
        # canvas load that calls _connect_raw to rebuild edges from
        # storage. (Disconnect is handled in _disconnect_edge_by_id_raw.)
        try:
            if (
                str(dst_port_name) == "robot_pipe"
                and str(getattr(dst_node.manifest, "id", "") or "")
                    == "actor_setting"
            ):
                from .param_rows import _reconcile_actor_init_joint_angles
                _reconcile_actor_init_joint_angles(dst_node)
                # Actuator caps (effort/velocity) moved to the RobotNode and
                # are reconciled there on asset_id change — not on this
                # actor_setting connect (2026-05).
        except Exception as exc:
            log_warning(
                f"[ui.canvas] connect: actor_setting reconcile after "
                f"robot_pipe connect failed: {exc}"
            )
        return edge

    # ---- Undo helpers (raw mutation primitives used by command classes) ----

    @contextmanager
    def _suspend_undo(self):
        """Context manager: 把 ``_in_undo`` 置 True，跳过 push 路径.

        同时置 ``_batch_load=True`` 让 ``_emit_topology_changed`` 在加载批量
        spawn 期间静默；加载/复制粘贴等批量操作完成后，调用方应再 emit 一次。
        """
        prev_undo = self._in_undo
        prev_batch = getattr(self, "_batch_load", False)
        self._in_undo = True
        self._batch_load = True
        try:
            yield
        finally:
            self._in_undo = prev_undo
            self._batch_load = prev_batch

    def _find_edge_by_id(self, edge_id: str) -> Optional[ConnectionItem]:
        if not edge_id:
            return None
        for it in self._scene.items():
            if isinstance(it, ConnectionItem) and getattr(it, "edge_id", None) == edge_id:
                return it
        return None

    def _disconnect_edge_by_id_raw(self, edge_id: str) -> bool:
        """断开指定 edge_id 的边；不存在则静默返回 False."""
        edge = self._find_edge_by_id(edge_id)
        if edge is None:
            return False
        # Capture dst before disconnect (the edge object's port refs may
        # become stale once disconnected). Used to trigger ActorSetting
        # reconcile if the cut edge was a Robot→ActorSetting robot_pipe.
        dst_port = getattr(edge, "dst_port", None)
        dst_node = None
        dst_port_name = ""
        if dst_port is not None:
            try:
                dst_node = dst_port.parent_node()
                dst_port_name = str(
                    getattr(dst_port, "port_name", "") or ""
                )
            except Exception:
                dst_node = None
        edge.disconnect()
        try:
            if (
                dst_node is not None
                and dst_port_name == "robot_pipe"
                and str(getattr(dst_node.manifest, "id", "") or "")
                    == "actor_setting"
            ):
                from .param_rows import (
                    _reconcile_actor_init_joint_angles,
                )
                # After disconnect, _upstream_robot_sku raises → reconcile
                # collapses init_joint_angles to {}. Per canvas-derived-keys
                # rule: no pipe → no authoritative IR roles → no keys.
                _reconcile_actor_init_joint_angles(dst_node)
        except Exception as exc:
            log_warning(
                f"[ui.canvas] disconnect: actor_setting reconcile after "
                f"robot_pipe disconnect failed: {exc}"
            )
        return True

    def _delete_node_raw(self, instance_id: Optional[str]) -> None:
        """删除节点（连同其上的所有边）；找不到则静默 noop.

        Placeholder 节点同样可被删除：scene/instances 清掉后，把任何引用它的
        deferred edge 一并丢弃，保证 to_workflow_dict 不再 emit 悬空边。
        """
        if not instance_id:
            return
        item = self._instances.get(instance_id)
        if item is None:
            return
        for p in list(item.iter_ports()):
            for c in list(p.connections):
                c.disconnect()
        self._scene.removeItem(item)
        self._instances.pop(instance_id, None)
        if self._deferred_edges:
            self._deferred_edges = [
                e for e in self._deferred_edges
                if e.get("source_node") != instance_id
                and e.get("target_node") != instance_id
            ]
        # 节点若属于某 group → 从 group.member_ids 摘掉;若 group 因此变空则销毁。
        for gid in list(self._groups.keys()):
            grp = self._groups.get(gid)
            if grp is None:
                continue
            if instance_id in grp.member_ids:
                grp.member_ids = [m for m in grp.member_ids if m != instance_id]
                if not grp.member_ids:
                    self._scene.removeItem(grp)
                    self._groups.pop(gid, None)
        self._emit_topology_changed()

    def _set_param_raw(self, instance_id: str, key: str, value: Any) -> None:
        """命令 redo / undo 用的 param 写入 —— 透传到 ParamRow + 触发可见性重排.

        写入完成后 emit ``signals.canvas_param_changed`` 让 mission_panel 等订阅方
        刷新对应 UI；emit 携带当前 page 的 ``_file_id`` 用于按画布过滤。
        """
        item = self._instances.get(instance_id)
        if item is None:
            return
        item.params[key] = value
        # 同步 ParamRow 的内部值与显示（绕过 set_value 的 push 路径）
        for row in getattr(item, "_param_rows", []) or []:
            try:
                if row.spec.key == key:
                    row._value = value
                    row.update()
                    break
            except Exception:
                pass
        try:
            item.on_param_changed(key, value)
        except Exception:
            pass
        # 广播给跨 widget 订阅者（mission_panel 训练配置卡片等）。延迟导入
        # 避免 page.py 在 application.service 之前的极早期被 import 时打断。
        try:
            from application.service.signals import get_app_signals
            get_app_signals().canvas_param_changed.emit(
                self._file_id, instance_id, str(key), value
            )
        except Exception:
            pass

    def _reconnect_edge_raw(self, edge_data: Dict[str, Any]) -> Optional[ConnectionItem]:
        """按 edge_data dict（含 source_node/source_port/target_node/target_port/id）重建边."""
        return self._connect_raw(
            edge_data.get("source_node"),
            edge_data.get("source_port"),
            edge_data.get("target_node"),
            edge_data.get("target_port"),
            edge_id=edge_data.get("id"),
        )

    def _apply_snapshot_raw(
        self,
        snap: Dict[str, Any],
        offset: QPointF = QPointF(0.0, 0.0),
        fixed_node_ids: Optional[List[str]] = None,
        fixed_edge_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """落一份 ``_serialize_selection`` / Delete 快照到画布.

        - ``fixed_node_ids`` / ``fixed_edge_ids`` 为 None 时自动生成新 id（粘贴 / 复制选中）；
          为 list 时按顺序使用（PasteCmd / DeleteSelectionCmd 第二次 redo / undo 复位）。
        - 返回 (生成的 node id 顺序, 生成的 edge id 顺序)，PasteCmd 锁存供下次复用.
        """
        # 清原选中
        for it in self._scene.selectedItems():
            it.setSelected(False)

        node_ids_out: List[str] = []
        id_remap: Dict[str, str] = {}
        snap_nodes = snap.get("nodes", []) or []
        for idx, n in enumerate(snap_nodes):
            schema_id = n.get("schema_id")
            old_id = n.get("id")
            if not schema_id:
                continue
            ui = n.get("ui") or {}
            x = float(ui.get("x", 0.0)) + offset.x()
            y = float(ui.get("y", 0.0)) + offset.y()
            params_raw = n.get("params") or {}
            params = {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in params_raw.items()
            }
            target_id = (
                fixed_node_ids[idx]
                if fixed_node_ids is not None and idx < len(fixed_node_ids)
                else None
            )
            item = self._spawn_node_raw(
                schema_id, x=x, y=y,
                instance_id=target_id,
                params=params,
            )
            if item is not None:
                if old_id:
                    id_remap[old_id] = item.node_id
                node_ids_out.append(item.node_id)
                item.setSelected(True)

        edge_ids_out: List[str] = []
        snap_edges = snap.get("edges", []) or []
        for idx, e in enumerate(snap_edges):
            src_old = e.get("source_node")
            dst_old = e.get("target_node")
            src_new = id_remap.get(src_old, src_old)
            dst_new = id_remap.get(dst_old, dst_old)
            if not src_new or not dst_new:
                continue
            target_eid = (
                fixed_edge_ids[idx]
                if fixed_edge_ids is not None and idx < len(fixed_edge_ids)
                else e.get("id") if fixed_node_ids is not None else None
            )
            edge = self._connect_raw(
                src_new, e.get("source_port"),
                dst_new, e.get("target_port"),
                edge_id=target_eid,
            )
            if edge is not None:
                edge_ids_out.append(edge.edge_id)

        log_debug(
            f"[ui.canvas] applied snapshot: {len(node_ids_out)} 节点 + "
            f"{len(edge_ids_out)} 边 (offset=({offset.x():.0f},{offset.y():.0f}))"
        )
        return node_ids_out, edge_ids_out

    # ---- 序列化 / 反序列化（Phase 1c）----

    def to_workflow_dict(self) -> Dict[str, Any]:
        """画布当前状态 → IR-shape dict / Snapshot scene to IR-shape dict.

        输出 dict 与 ``WorkflowIR.to_dict()`` 完全同形，可直接：
            ``save_data(path, page.to_workflow_dict())`` 或
            ``ir = canvas_to_ir(page.to_workflow_dict())``
        """
        nodes_out: List[Dict[str, Any]] = []
        for inst_id, node_item in self._instances.items():
            # Placeholder：原 dict 透传，仅刷新位置（保留 params / kind / opaque_code 不丢）
            if isinstance(node_item, PlaceholderNodeItem):
                snap = dict(node_item.original_dict or {})
                ui_orig = dict(snap.get("ui") or {})
                ui_orig["x"] = float(node_item.pos().x())
                ui_orig["y"] = float(node_item.pos().y())
                # width/height 在 placeholder 里是固定值；保留原 ui 中可能存在的真实尺寸
                ui_orig.setdefault("width", float(node_item.width))
                ui_orig.setdefault("height", float(node_item.height))
                ui_orig.setdefault("collapsed", False)
                snap["id"] = inst_id
                snap["schema_id"] = node_item.schema_id
                snap["ui"] = ui_orig
                # Placeholder 也保留 disabled 标识(无 set_disabled,但 dict 里
                # 原值透传 + 默认 False;真实 NodeItem 重新可用时回到正常路径)
                snap.setdefault("disabled", False)
                nodes_out.append(snap)
                continue

            params: Dict[str, Any] = {}
            # 先扫 manifest.parameters 拿 type；没有 spec 的 param 走 string 兜底
            spec_by_key = {p.key: p for p in node_item.manifest.parameters}
            for key, val in node_item.params.items():
                # Skip transient / canvas-injected keys not declared in the
                # manifest (e.g. ``backend`` on rewards/terminations/
                # domain_rand — backend lives at canvas top-level only).
                if key not in spec_by_key:
                    continue
                ptype = spec_by_key[key].type
                params[key] = IRParam(name=key, value=val, param_type=ptype).to_dict()
            ui = IRNodeUI(
                x=float(node_item.pos().x()),
                y=float(node_item.pos().y()),
                width=float(node_item.width),
                height=float(node_item.height),
                collapsed=bool(getattr(node_item, "_collapsed", False)),
            ).to_dict()
            node_entry: Dict[str, Any] = {
                "id": inst_id,
                "schema_id": node_item.manifest.id,
                "kind": node_item.manifest.kind.value,
                "params": params,
                "ui": ui,
                "opaque_code": None,
                # disabled 是语义字段(影响 IR 过滤),与 collapsed (UI-only) 分离;
                # 放在顶层与 ``opaque_code`` 同层级,与 ComfyUI 习惯一致。
                "disabled": bool(getattr(node_item, "_disabled", False)),
            }
            # display_name_override —— 用户双击改名的字符串;空 / None 不落盘
            override = getattr(node_item, "_display_name_override", "") or ""
            if override:
                node_entry["display_name_override"] = override
            # color_override —— SelectionToolbar 色盘按钮设置的 per-canvas 颜色;
            # 空字符串视为"未设置",不落盘。NodeItem + NoteItem 共享此路径
            # (NoteItem 在 nodes 列表里),GroupItem 走自己的 serialize()。
            color_ov = getattr(node_item, "_color_override", "") or ""
            if color_ov:
                node_entry["color_override"] = color_ov
            nodes_out.append(node_entry)

        edges_out: List[Dict[str, Any]] = []
        for it in self._scene.items():
            if isinstance(it, ConnectionItem):
                src_node = it.src_port.parent_node()
                dst_node = it.dst_port.parent_node()
                if src_node is None or dst_node is None:
                    continue
                edges_out.append(IREdge(
                    id=it.edge_id,
                    source_node=src_node.node_id,
                    source_port=it.src_port.port_name,
                    target_node=dst_node.node_id,
                    target_port=it.dst_port.port_name,
                ).to_dict())
        # Deferred edges（指向 placeholder 节点的边）原样回传，保证 round-trip 不丢数据
        for de in self._deferred_edges:
            edges_out.append(dict(de))

        # view 状态进 metadata
        zoom = float(self._view.transform().m11())
        center = self._view.mapToScene(self._view.viewport().rect().center())
        metadata = {
            "view": {
                "zoom": zoom,
                "center_x": float(center.x()),
                "center_y": float(center.y()),
            },
        }

        groups_out: List[Dict[str, Any]] = [
            g.serialize() for g in self._groups.values()
        ]

        # Top-level keys ordered for human-readable diffs:
        #   version → backend → robot_id → metadata → nodes → edges → groups
        # ``robot_id`` is a derived projection of the Robot node's
        # ``asset_id`` (see :attr:`robot_id`); recomputed on every save so
        # canvas top-level and node state stay 100% in sync. Omitted when
        # no robot node exists / its asset_id fails to resolve to a SKU.
        out: Dict[str, Any] = {"version": "1.0.0"}
        if self._backend_id is not None:
            out["backend"] = self._backend_id
        robot_id = self.robot_id
        if robot_id is not None:
            out["robot_id"] = robot_id
        out["metadata"] = metadata
        out["nodes"] = nodes_out
        out["edges"] = edges_out
        out["groups"] = groups_out
        return out

    def from_workflow_dict(self, data: Dict[str, Any]) -> int:
        """从 IR-shape dict 重建画布 / Rebuild scene from IR-shape dict.

        清空当前画布，按 ``data`` 重新 spawn 节点 + 连边 + 恢复视口状态。
        Always extracts the top-level ``backend`` field and rebinds the
        canvas to it; missing or non-string ``backend`` is a hard error
        (RELEASE does not tolerate unbacked canvases).

        Returns: 成功重建的节点数（不算被拒的）
        """
        if _is_legacy_training_canvas(data):
            log_debug("[ui.canvas] from_workflow_dict: detected legacy 'training_canvas_v1' — migrating to IR-shape")
            data = _legacy_to_ir(data)

        # 缺口② — graph-level fold: merge a legacy two-node il_observation
        # (group_name policy + critic) into a single node before per-node shims.
        # Idempotent (metadata.obs_merge_v1). Shared with the bootstrap migrator.
        from application.compiler.obs_group_merge import merge_obs_groups
        merge_obs_groups(data)

        # Bug#1 — reconcile legacy log-space init_noise_std to the direct-std
        # convention (a <= 0 value is exp()'d to preserve the realized std and
        # WARNs; positive values are kept verbatim). Idempotent
        # (metadata.init_noise_std_direct_v1). Shared with the bootstrap migrator.
        from application.compiler.init_noise_std_migrate import (
            migrate_init_noise_std_direct,
        )
        migrate_init_noise_std_direct(data)

        # RC-4 — strip deprecated AMP hyperparams off the trainer node so
        # the discriminator node is the unambiguous single source (the
        # consumer reads il.amp.*, populated from the discriminator).
        # Idempotent (metadata.amp_trainer_fields_stripped_v1).
        from application.compiler.amp_trainer_field_strip import (
            strip_deprecated_amp_trainer_fields,
        )
        strip_deprecated_amp_trainer_fields(data)

        # "item id IS its name" — fold legacy ``new_item_*`` training-item keys
        # (opaque id + dead ``title`` field) to registered command ids, drop the
        # title, rewrite ``reward_in__<old>`` edges, and register the name as a
        # user command (inheriting a matching builtin's velocity template).
        # Idempotent (metadata.training_item_ids_v1). Shared with the bootstrap
        # migrator (bootstrap/migrate_canvas_training_item_ids.py).
        from application.compiler.training_item_id_migrate import (
            migrate_training_item_ids,
        )
        migrate_training_item_ids(data)

        # Per-schema migration shims — run once per load before any
        # IRParam construction so v2 nodes see only the canonical fields.
        for _n in (data.get("nodes") or []):
            if isinstance(_n, dict):
                _migrate_base_asset_params(_n)

        backend = data.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError(
                "from_workflow_dict: canvas missing required top-level 'backend' field "
                "(every canvas must declare its training backend; project still in development, no legacy compat)"
            )
        backend = backend.strip()

        self.clear_scene()
        # Reset the bind — loading a different file is a deliberate switch,
        # not a mutation of an existing canvas's backend.
        self._backend_id = None
        self.bind_backend(backend)

        nodes = data.get("nodes") or []
        edges = data.get("edges") or []

        spawned = 0
        placeholders = 0
        # 加载是非用户行为；用 _suspend_undo 把内部 spawn / connect_ports 全走 raw
        # 路径，加载完毕后清空 stack 以免跨文件污染。
        with self._suspend_undo():
            from registers import nodes as nodes_registry
            for n in nodes:
                schema_id = n.get("schema_id")
                inst_id = n.get("id")
                if not schema_id or not inst_id:
                    log_warning(f"[ui.canvas] from_workflow_dict: 跳过缺 id/schema_id 的节点 {n}")
                    continue
                ui = n.get("ui") or {}
                x = float(ui.get("x", 0.0))
                y = float(ui.get("y", 0.0))
                params_raw = n.get("params") or {}
                # Strict unwrap of IRParam-shaped {name, value, param_type}
                # dicts. The 2026-05-11 contract bans silent normalization
                # here — every entry must be a well-formed IRParam dict, or
                # we surface the bad canvas rather than dropping fields.
                # Exception: ``backend`` is a long-standing legacy key that
                # used to live on the node but now lives at canvas level;
                # we drop it with a WARNING so old saves still load.
                params: Dict[str, Any] = {}
                for k, v in params_raw.items():
                    if k == "backend":
                        log_warning(
                            f"[ui.canvas] from_workflow_dict: legacy "
                            f"per-node 'backend' key on {schema_id}/{inst_id} "
                            f"is ignored; backend now lives at canvas level"
                        )
                        continue
                    if not isinstance(v, dict):
                        log_warning(
                            f"[ui.canvas] from_workflow_dict: {schema_id}/"
                            f"{inst_id} param {k!r} is not an IRParam dict "
                            f"(got {type(v).__name__}); using raw value as-is"
                        )
                        params[k] = v
                        continue
                    if "value" not in v:
                        log_warning(
                            f"[ui.canvas] from_workflow_dict: {schema_id}/"
                            f"{inst_id} param {k!r} IRParam dict missing "
                            f"'value' field (keys={sorted(v)!r}); skipping"
                        )
                        continue
                    params[k] = v["value"]

                if nodes_registry.get_manifest(schema_id) is None:
                    # 缺 manifest → 构造 PlaceholderNodeItem，原 dict 透传保留
                    if inst_id in self._instances:
                        log_warning(
                            f"[ui.canvas] from_workflow_dict: instance_id '{inst_id}' "
                            f"已存在；跳过 placeholder 重复创建"
                        )
                        continue
                    placeholder = PlaceholderNodeItem(schema_id, inst_id, original_dict=n)
                    placeholder.setPos(x, y)
                    self._scene.addItem(placeholder)
                    self._instances[inst_id] = placeholder  # type: ignore[assignment]
                    placeholders += 1
                    log_warning(
                        f"[ui.canvas] from_workflow_dict: schema_id '{schema_id}' "
                        f"未在 registry 注册 → 构造 placeholder（参数 + 位置原样保留）"
                    )
                    continue

                # Note 节点恢复用户调整后的尺寸:其他节点的 ui.width/height 来自
                # manifest 布局,_spawn_node_raw 内部已忽略。
                ui_w = ui.get("width")
                ui_h = ui.get("height")
                item = self._spawn_node_raw(
                    schema_id,
                    x=x,
                    y=y,
                    instance_id=inst_id,
                    params=params,
                    width=float(ui_w) if isinstance(ui_w, (int, float)) else None,
                    height=float(ui_h) if isinstance(ui_h, (int, float)) else None,
                )
                if item is not None:
                    spawned += 1
                    # 恢复 disabled / collapsed —— 旧 canvas JSON 无字段时
                    # .get(...) 落到默认 False,行为退化为"全启用全展开"。
                    is_disabled = bool(n.get("disabled", False))
                    is_collapsed = bool((n.get("ui") or {}).get("collapsed", False))
                    if hasattr(item, "set_disabled") and is_disabled:
                        item.set_disabled(True)
                    if hasattr(item, "set_collapsed") and is_collapsed:
                        item.set_collapsed(True)
                    # 恢复用户改名 override —— 旧 canvas JSON 无该字段 → 跳过。
                    override = (n.get("display_name_override") or "").strip()
                    if override and hasattr(item, "set_display_name_override"):
                        item.set_display_name_override(override)
                    # 恢复 per-canvas 颜色覆写。旧 canvas JSON 无该字段 → 跳过;
                    # NodeItem + NoteItem 都走这条路径 (都在 nodes 列表里)。
                    color_ov = (n.get("color_override") or "").strip()
                    if color_ov and hasattr(item, "set_color_override"):
                        item.set_color_override(color_ov)

            for e in edges:
                src_id = e.get("source_node")
                src_p = e.get("source_port")
                dst_id = e.get("target_node")
                dst_p = e.get("target_port")
                edge_id = e.get("id")
                src_node = self._instances.get(src_id) if src_id else None
                dst_node = self._instances.get(dst_id) if dst_id else None
                if src_node is None or dst_node is None:
                    log_warning(f"[ui.canvas] from_workflow_dict: 边引用缺失节点 src={src_id} dst={dst_id}")
                    continue
                # 任一端是 placeholder → 边降级为 deferred（save 时原样吐回，避免数据丢失）
                if isinstance(src_node, PlaceholderNodeItem) or isinstance(dst_node, PlaceholderNodeItem):
                    self._deferred_edges.append(dict(e))
                    log_warning(
                        f"[ui.canvas] from_workflow_dict: 边 {edge_id} 引用 placeholder 节点 "
                        f"({src_id} → {dst_id})；降级为 deferred，save 时原样保留"
                    )
                    continue
                src_port = src_node.port(src_p, "out") or src_node.port(src_p, "in")
                dst_port = dst_node.port(dst_p, "in") or dst_node.port(dst_p, "out")
                if src_port is None or dst_port is None:
                    log_warning(f"[ui.canvas] from_workflow_dict: 边引用缺失端口 src={src_p} dst={dst_p}")
                    continue
                try:
                    edge = connect_ports(src_port, dst_port, edge_id=edge_id)
                except ValueError as ex:
                    log_warning(f"[ui.canvas] from_workflow_dict: {ex}")
                    continue
                self._scene.addItem(edge)

        # 恢复 groups —— 节点都 spawn 完才有效:GroupItem.member_ids 引用 node id。
        # 旧 canvas JSON 无 "groups" 字段时 .get 返回 [] 兼容退化。
        groups = data.get("groups") or []
        for g in groups:
            try:
                gid = str(g.get("id") or "")
                title = str(g.get("title") or "")
                members = list(g.get("members") or [])
                # 过滤掉已不存在(被同步删除)的 member
                members = [m for m in members if m in self._instances]
                if not gid or not members:
                    continue
                rect_dict = g.get("rect") or {}
                rect = QRectF(
                    float(rect_dict.get("x", 0.0)),
                    float(rect_dict.get("y", 0.0)),
                    float(rect_dict.get("w", 200.0)),
                    float(rect_dict.get("h", 200.0)),
                )
                grp = GroupItem(gid, title, members, rect)
                # 恢复 per-canvas 颜色覆写。旧 canvas JSON 无该字段 → 跳过。
                grp_color_ov = (g.get("color_override") or "").strip()
                if grp_color_ov:
                    grp.set_color_override(grp_color_ov)
                self._scene.addItem(grp)
                self._groups[gid] = grp
            except Exception as exc:
                log_warning(f"[ui.canvas] from_workflow_dict: group 恢复失败 {g}: {exc!r}")

        # 恢复视口
        view_meta = (data.get("metadata") or {}).get("view") or {}
        zoom = float(view_meta.get("zoom", 1.0))
        if 0.05 < zoom < 10.0:
            cur = self._view.transform().m11()
            if cur > 0:
                self._view.scale(zoom / cur, zoom / cur)
        cx = view_meta.get("center_x")
        cy = view_meta.get("center_y")
        if cx is not None and cy is not None:
            self._view.centerOn(QPointF(float(cx), float(cy)))

        # 加载是 history 边界——用户对加载结果再做的操作才进 undo stack
        self._undo_stack.clear()

        # Top-level ``robot_id`` consistency check: file value vs. the SKU
        # derived from the loaded Robot node. RobotNode params win (single
        # source of truth); mismatch logs a warning and the next save will
        # rewrite the file with the derived value.
        file_robot_id = data.get("robot_id")
        if isinstance(file_robot_id, str) and file_robot_id.strip():
            file_robot_id = file_robot_id.strip()
            derived = self.robot_id
            if derived is None:
                log_warning(
                    f"[ui.canvas] from_workflow_dict: top-level robot_id "
                    f"{file_robot_id!r} present but canvas has no robot node "
                    f"(or asset_id failed to resolve); save will drop the key"
                )
            elif derived != file_robot_id:
                log_warning(
                    f"[ui.canvas] from_workflow_dict: top-level robot_id "
                    f"mismatch — file says {file_robot_id!r} but robot node "
                    f"asset_id resolves to {derived!r}; using robot node as "
                    f"source of truth (save will rewrite to {derived!r})"
                )

        # 加载完成后广播一次拓扑变化:``clear_scene`` 入口时已经发过一次"空画
        # 布"事件,中途 spawn/connect 都在 ``_suspend_undo`` (``_batch_load=
        # True``) 下静默,这里是订阅方(MissionControlPanel 训练配置透视
        # 等)唯一能看到"完整重建"画布的时机。文件加载路径靠后续 ``set_file_id``
        # 的 emit 救场,但模板导入 (``replace_content_from_dict``) 不改 file_id
        # → 不会有第二次 emit,面板永远停在"空画布"快照上。把 emit 放在这里
        # 让所有调用方一视同仁。
        self._emit_topology_changed()

        return spawned + placeholders

    def replace_content_from_dict(self, data: Dict[str, Any]) -> int:
        """In-place replace: load ``data`` into the scene while keeping the
        canvas bound to its current file path.

        Semantically equivalent to "paste another file's content into the
        currently open document" in a text editor: scene + backend rebind
        to ``data``, but the save target (``_current_path``) stays on the
        file the user had open. The canvas is marked dirty so Ctrl+S
        persists ``data`` to that file. Caller is responsible for the
        save-or-discard prompt beforehand.

        Returns the number of nodes rebuilt (same contract as
        :meth:`from_workflow_dict`). Raises ``RuntimeError`` if no canvas
        file is currently bound — in-place replace is meaningless without
        a target file.
        """
        if not self._current_path:
            raise RuntimeError(
                "replace_content_from_dict: no canvas file bound; "
                "load one first"
            )
        saved_path = self._current_path
        spawned = self.from_workflow_dict(data)
        # from_workflow_dict → clear_scene → _current_path=None + clearDirtyTracker.
        # Rebind to the original file and seed an empty baseline so the
        # actual content fingerprint != baseline → canvas reports dirty.
        self._current_path = saved_path
        try:
            setDirtyBaseline(self._current_path, "")
            self._refresh_dirty_state()
        except Exception as exc:                                # pragma: no cover
            log_warning(
                f"[ui.canvas] replace_content_from_dict: dirty baseline "
                f"rebind failed: {exc!r}"
            )
        return spawned

    def save_to_file(self, path: str | Path) -> bool:
        """保存当前画布到 ``.canvas.json`` / Save scene to .canvas.json (via DataManager).

        路径走 ``DataManager.save_data``（原子写：temp + os.replace）。

        Dev-time soft check: warn (not error) if *path* doesn't end in
        ``.canvas.json`` or doesn't sit under any project's ``canvas/`` dir.
        Hard validation lives in ``save_to_project`` — this entry stays
        permissive so legacy callers / tests can still write to scratch
        paths without policy interference.
        """
        try:
            data = self.to_workflow_dict()
            p = Path(path)
            if not p.name.lower().endswith(".canvas.json"):
                log_warning(
                    f"[ui.canvas] save_to_file: 路径 {path} 不以 .canvas.json 结尾；"
                    f"file_index 不会发现该文件"
                )
            save_data(p, data)
            log_success(f"[ui.canvas] saved to {path}")
            # 保存成功 → 把当前内容固化为新基线（处理 Save As 改路径的情况：
            # setDirtyBaseline 会 emit pathChanged + 把 dirty 拉回 False）。
            try:
                self._current_path = str(p)
                setDirtyBaseline(self._current_path, self._content_fingerprint())
            except Exception as exc:
                log_warning(f"[ui.canvas] dirty baseline refresh after save failed: {exc!r}")
            return True
        except Exception as e:
            log_error(f"[ui.canvas] save_to_file failed: {e!r}")
            return False

    def save_to_project(
        self,
        project: "ProjectInfo",  # noqa: F821 — fwd ref to break import cycle
        name: str = "main",
    ) -> Optional[Path]:
        """Save canvas under ``<project>/canvas/<canvas_subdir>/<stem>.canvas.json``
        with hard path validation.

        ``name`` MUST start with this canvas's backend sub-directory
        (``registers.backends.canvas_subdir(self.backend_id)``) so the file
        lands under the right backend bucket — this is the storage
        contract the Mission Control panel + file_index discovery rely on.
        Callers that want the default placement should pass
        ``f"{canvas_subdir(eid)}/{stem}"``.

        Returns the absolute path on success, or ``None`` on failure (path
        validation rejected, write error). The path is logged on success so
        the caller can pass it to ``file_index`` discovery.

        This is the *policy-enforcing* save entry — use it from the app
        shell (Ctrl+S, File ▸ Save) to guarantee the canvas lands where
        the Mission Control panel + project store will discover it.
        """
        # Local import — service.projects already depends on UI-free symbols
        # (Paths, DataManager, Config) but importing it at module top would
        # introduce an import cycle once page.py is reused from a service-
        # side test that bootstraps ProjectStore via this file.
        from application.service.projects import (
            current_project_canvas_path,
            ensure_project_canvas_path,
        )
        from registers import backends as backends_registry

        if self._backend_id is None:
            log_error(
                "[ui.canvas] save_to_project rejected: canvas has no bound backend; "
                "call bind_backend() before saving"
            )
            return None

        expected_subdir = backends_registry.canvas_subdir(self._backend_id)
        first_segment = str(name).replace("\\", "/").split("/", 1)[0]
        if first_segment != expected_subdir:
            log_error(
                f"[ui.canvas] save_to_project rejected: name first segment "
                f"{first_segment!r} != backend subdir {expected_subdir!r} "
                f"(backend_id={self._backend_id!r})"
            )
            return None

        try:
            target = current_project_canvas_path(project, name)
            ensure_project_canvas_path(project, target)
        except ValueError as exc:
            log_error(f"[ui.canvas] save_to_project rejected: {exc}")
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.save_to_file(target):
            return None
        return target

    def load_from_file(self, path: str | Path) -> int:
        """从 ``.canvas.json`` 重建画布 / Load scene from .canvas.json.

        若检测到 DEMO ``training_canvas_v1`` 旧格式，则在加载成功后立刻把磁盘文件
        以 IR-shape 覆写（单向迁移落盘），下次直接走标准路径而不再触发 legacy 翻译。

        Returns: 成功重建的节点数（0 表示加载失败或文件为空）
        """
        try:
            data = read_data(Path(path))
        except Exception as e:
            log_error(f"[ui.canvas] load_from_file: 读取失败 {path}: {e!r}")
            return 0
        if not isinstance(data, dict):
            log_error(f"[ui.canvas] load_from_file: 文件内容非 dict {path}")
            return 0

        was_legacy = _is_legacy_training_canvas(data)
        # Pre-validate normalization: legacy schema → IR-shape, then run
        # per-schema migration shims (e.g. base_asset v1 → v2 strips
        # asset_id / checkpoint_file and rewrites run:<path> tokens into
        # __load__ + checkpoint_id). Applied in-place to ``data`` so the
        # subsequent validate AND from_workflow_dict both see the
        # normalised form — otherwise validate fires "manifest 未声明的参
        # 数 key" warnings on legacy keys the shim is about to strip.
        if was_legacy:
            data = _legacy_to_ir(data)
        for _n in (data.get("nodes") or []):
            if isinstance(_n, dict):
                _migrate_base_asset_params(_n)
        # 加载前 manifest 一致性校验：列出缺节点 / 端口名漂移 / 参数 key 漂移；
        # 不阻塞加载，只为用户察觉哪些节点会变 placeholder。
        try:
            from application.compiler.migrations import validate_against_manifests
            for w in validate_against_manifests(data):
                log_warning(f"[ui.canvas] {path}: {w}")
        except Exception as exc:
            log_warning(f"[ui.canvas] load_from_file: validate 失败 (非致命): {exc!r}")
        try:
            spawned = self.from_workflow_dict(data)
        except (ValueError, RuntimeError) as e:
            log_error(f"[ui.canvas] load_from_file: rejected {path}: {e}")
            return 0

        # 一次性标准化：原文件是 legacy 且加载成功 → 用画布反序列化结果覆写磁盘
        # 用 to_workflow_dict() 而非直接落盘翻译结果，可顺带带上真实视口元数据和
        # 已分配的 edge uuid，且 NodeItem 的 width/height 来自 manifest 计算值。
        if was_legacy and spawned > 0:
            try:
                save_data(Path(path), self.to_workflow_dict())
                log_debug(f"[ui.canvas] load_from_file: legacy 文件已就地标准化为 IR-shape: {path}")
            except Exception as e:
                log_warning(f"[ui.canvas] load_from_file: legacy 标准化写回失败（不影响本次加载）{path}: {e!r}")

        # 加载成功 → 接入全局 DirtyTracker：把刚加载的内容作为干净基线，
        # 后续 undo_stack indexChanged / topology_changed 都按此基线判脏。
        try:
            self._current_path = str(Path(path))
            setDirtyBaseline(self._current_path, self._content_fingerprint())
        except Exception as exc:
            log_warning(f"[ui.canvas] dirty baseline init after load failed: {exc!r}")

        return spawned

    def clear_scene(self) -> None:
        """清空所有节点和边 / Clear all nodes and edges (background grid 保留).

        同时清除全局 DirtyTracker 追踪 —— ``clear_scene`` 之后画布无文件背景，
        ``is_dirty()`` 返回 False。``mission_control_panel`` 在切换画布时先 clear
        再 load，``load_from_file`` 会重新 setBaseline。
        """
        for it in list(self._scene.items()):
            if isinstance(it, (NodeItem, NoteItem, GroupItem, PlaceholderNodeItem, ConnectionItem)):
                self._scene.removeItem(it)
        self._instances.clear()
        self._groups.clear()
        self._deferred_edges.clear()
        self._current_path = None
        try:
            clearDirtyTracker()
        except Exception:
            pass
        self._emit_topology_changed()

    # ---- demo populate（Phase 1a 临时；Phase 1b 接 palette+drag 后移除） ----

    def populate_demo(self) -> List[NodeItem]:
        """从 registry 自动拖入若干样例节点（演示 Phase 1a 端到端）.

        - 幂等：多次调用只生效一次
        - 不写存盘；只是给画布一个非空初始状态
        - Phase 1b 接入调色板 + 拖拽 + 文件加载后会移除

        Returns: 生成的 NodeItem 列表
        """
        if self._demo_populated:
            return []
        self._demo_populated = True

        from registers import nodes as nodes_registry

        # 优先选已知 builtin manifest
        builtin_ids = [m.id for m in nodes_registry.list_nodes() if m.id != "example.print_hello"]
        if not builtin_ids:
            log_debug("[ui.canvas] populate_demo: 暂无 builtin 节点，跳过 demo")
            return []

        spawned: List[NodeItem] = []
        # Phase 1a 起步只演示一个 task_config + 一个 example_node，相距摆放
        positions: List[Tuple[str, float, float]] = []
        if "task_config" in builtin_ids:
            positions.append(("task_config", -200.0, -100.0))
        # 第二个找别的可用节点
        for nid in builtin_ids:
            if nid != "task_config":
                positions.append((nid, 200.0, -100.0))
                break
        # 兜底：把 example_node 也拉一个进来证明 custom 路径同等工作
        positions.append(("example.print_hello", 0.0, 200.0))

        with self._suspend_undo():
            for nid, x, y in positions:
                item = self._spawn_node_raw(nid, x, y)
                if item is not None:
                    spawned.append(item)
        # demo 不进 history
        self._undo_stack.clear()

        log_debug(f"[ui.canvas] populate_demo: spawned {len(spawned)} 节点（Phase 1a 临时）")
        return spawned

    # ---- 主题 ----

    def apply_theme(self) -> None:
        """主题切换时由外部调用，让 scene 重读 Config 并刷栅格 brush."""
        self._scene.apply_theme()
        # 节点已建好的实例需要重绘以拿新主题色
        for item in self._instances.values():
            item.update()
        # Node Library card 的 QSS 也需要随主题刷新（embedded / 非 embedded 都有）
        if self._node_library is not None:
            self._node_library.apply_theme()
        elif self._node_library_controller is not None:
            self._node_library_controller.apply_theme()
        # AI Build 浮层随主题刷新（开关已迁到 MC 顶栏，由 MC apply_theme 刷新）
        if getattr(self, "_ai_build_panel", None) is not None:
            self._ai_build_panel.apply_theme()

    # ---- 嵌入模式公开 API（MissionControlPanel 使用） ----

    @property
    def embedded(self) -> bool:
        return self._embedded

    @property
    def interactive(self) -> bool:
        return self._interactive

    def set_interactive(self, interactive: bool) -> None:
        """Toggle whether the canvas accepts edits (spawn / drag-spawn / shortcuts).

        Mission Control mode flips this to False so the chart overlay
        cannot be edited blind. Training Canva mode flips it back to True.
        Currently gates: NodeLibraryCardController spawn (double-click in card).
        """
        self._interactive = bool(interactive)

    def set_node_library_open(self, open_: bool) -> None:
        """Deprecated no-op: Node Library 现在挂在 Sidebar 弹出面板上。

        旧 caller(MissionControlPanel._apply_mode / 测试)调用此方法不再有任何
        副作用;真正的显隐由 ``Sidebar.set_node_library_visible`` 控制。
        """
        return

    def is_node_library_open(self) -> bool:
        return False

    def set_minimap_visible(self, visible: bool) -> None:
        """Show/hide the bottom-right minimap overlay (no animation)."""
        if self._minimap is not None:
            self._minimap.setVisible(bool(visible))

    # ===========================================================
    # Phase 2: 键盘快捷键
    # ===========================================================

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

        # ESC 取消端口拖拽
        if key == Qt.Key.Key_Escape:
            self._scene.port_drag.cancel()
            event.accept()
            return

        # Delete / Backspace → 删除选中。
        # hover + Delete 删除连线（替代已移除的右键删除）：当鼠标悬停在一条
        # 连线上、且当前没有任何选中项时，Delete 只删这条线；否则按常规删除
        # 选中的节点 / 边。
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            hovered = getattr(self._scene, "_hovered_edge", None)
            if (
                hovered is not None
                and hovered.scene() is self._scene
                and not self._scene.selectedItems()
            ):
                self._scene._hovered_edge = None
                self._user_disconnect_edge(hovered)
                event.accept()
                return
            self._delete_selected()
            event.accept()
            return

        if ctrl:
            shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            # Ctrl+Z undo / Ctrl+Shift+Z | Ctrl+Y redo —— 优先于其他 Ctrl 快捷键
            if key == Qt.Key.Key_Z and not shift:
                self._undo_stack.undo()
                event.accept()
                return
            if (key == Qt.Key.Key_Z and shift) or key == Qt.Key.Key_Y:
                self._undo_stack.redo()
                event.accept()
                return
            # Ctrl+A 全选
            if key == Qt.Key.Key_A:
                for it in self._scene.items():
                    if isinstance(it, (NodeItem, ConnectionItem)):
                        it.setSelected(True)
                event.accept()
                return
            # Ctrl+D 复制 + 偏移粘贴
            if key == Qt.Key.Key_D:
                self._duplicate_selected()
                event.accept()
                return
            # Ctrl+C 复制
            if key == Qt.Key.Key_C:
                self._copy_selected()
                event.accept()
                return
            # Ctrl+V 粘贴
            if key == Qt.Key.Key_V:
                self._paste(self._last_context_scene_pos)
                event.accept()
                return
            # Ctrl+G 成组 / 解组 —— 与 toolbar group 按钮共享同一入口
            if key == Qt.Key.Key_G:
                self._on_group_button_clicked()
                event.accept()
                return
            # Ctrl+H 展开 / 收纳 —— 与 toolbar collapse 按钮共享同一入口
            if key == Qt.Key.Key_H:
                self._toggle_collapse_selected()
                event.accept()
                return
            # Ctrl+K 禁用 / 启用 —— 与 toolbar skip 按钮共享同一入口
            if key == Qt.Key.Key_K:
                self._toggle_disable_selected()
                event.accept()
                return

        super().keyPressEvent(event)

    # ---- 选中操作的内部 helpers ----

    def _selected_nodes(self) -> List[Any]:
        # NodeItem ∪ PlaceholderNodeItem — 都暴露 node_id / pos / iter_ports，
        # 让删除 / 拖动 / 复制 走同一条路径。NoteItem 鸭子兼容 NodeItem 关键
        # 接口(.node_id/.iter_ports() 返回空),纳入这条路径后删除/序列化无障碍。
        return [
            it for it in self._scene.selectedItems()
            if isinstance(it, (NodeItem, NoteItem, PlaceholderNodeItem))
        ]

    def _selected_edges(self) -> List[ConnectionItem]:
        return [it for it in self._scene.selectedItems() if isinstance(it, ConnectionItem)]

    def _delete_selected(self) -> None:
        snap = self._serialize_full_delete_snapshot()
        if snap is None:
            return
        if self._in_undo:
            for n in snap.get("nodes", []):
                self._delete_node_raw(n.get("id"))
            for e in snap.get("loose_edges", []):
                self._disconnect_edge_by_id_raw(e.get("id"))
            # 兜底:若有 group 没因为成员级联删而消失(成员快照与实际不一致等),
            # 显式清掉。_delete_node_raw 已经会在成员清空时自动销毁 group,
            # 此处只清残留。
            for g in snap.get("groups", []):
                gid = g.get("id")
                if gid in self._groups:
                    self._delete_group_raw(gid)
            return
        from .undo import DeleteSelectionCmd
        self._undo_stack.push(DeleteSelectionCmd(self, snap))

    # ------------------------------------------------------------------
    # 选区悬浮工具栏 + Group 命令
    # ------------------------------------------------------------------
    def _selected_groupable_items(self) -> List[Any]:
        """选区中可参与成组的 item(NodeItem / NoteItem / GroupItem)."""
        return [
            it for it in self._scene.selectedItems()
            if isinstance(it, (NodeItem, NoteItem, GroupItem, PlaceholderNodeItem))
        ]

    def _group_of(self, item: Any) -> Optional[GroupItem]:
        """Item → 所属 GroupItem(若有);GroupItem 自身返回 None."""
        if isinstance(item, GroupItem):
            return None
        nid = getattr(item, "node_id", None)
        if not nid:
            return None
        for grp in self._groups.values():
            if nid in grp.member_ids:
                return grp
        return None

    def _compute_group_btn_mode(self, sel: List[Any]) -> str:
        """根据选区返回 "group" 或 "ungroup"。

        - 仅选中一个 GroupItem 本身(无任何 node) → "ungroup" 该 group
        - 选区不含 node 也不含 group(理论上不应到这里) → "group" 占位
        - node 全部隶属于同一现有 group(且无未分组项) → "ungroup"
        - 其他 → "group" (含"从旧组分离 加入新组"语义)
        """
        groups_in_sel = [it for it in sel if isinstance(it, GroupItem)]
        node_like = [it for it in sel if not isinstance(it, GroupItem)]
        # 单独选中一个 group(无 node)→ 解该 group
        if groups_in_sel and not node_like and len(groups_in_sel) == 1:
            return "ungroup"
        if not node_like:
            return "group"
        groups_of = {self._group_of(it) for it in node_like}
        # 全部属于同一现有 group(且 None 不在集合中)→ 解组
        if None not in groups_of and len(groups_of) == 1:
            return "ungroup"
        return "group"

    def _reposition_selection_toolbar(self) -> None:
        """根据当前选区把 ``_selection_toolbar`` 放到最靠上 item 上方 20px。"""
        tb = getattr(self, "_selection_toolbar", None)
        if tb is None:
            return
        sel = self._selected_groupable_items()
        if not sel:
            tb.hide()
            return
        # 更新工具栏所有 mode-driven icon (group/collapse/disable) —— 权威驱动
        try:
            tb.set_group_mode(self._compute_group_btn_mode(sel))
            tb.set_collapse_mode(self._compute_collapse_btn_mode(sel))
            tb.set_disable_mode(self._compute_disable_btn_mode(sel))
        except Exception:
            pass
        # 选区 scene-bounding union
        bounds: Optional[QRectF] = None
        for it in sel:
            br = it.sceneBoundingRect()
            bounds = br if bounds is None else bounds.united(br)
        if bounds is None:
            tb.hide()
            return
        # scene → viewport 坐标
        top_center_view = self._view.mapFromScene(
            QPointF(bounds.center().x(), bounds.top())
        )
        bottom_center_view = self._view.mapFromScene(
            QPointF(bounds.center().x(), bounds.bottom())
        )
        tb.adjustSize()
        sz = tb.size()
        x = int(top_center_view.x() - sz.width() / 2)
        y = int(top_center_view.y() - 20 - sz.height())
        if y < 0:
            # 选区贴顶 → 翻转到下方
            y = int(bottom_center_view.y() + 20)
        # 限制在 viewport 范围内,避免跑出可视区
        vp = self._view.viewport()
        x = max(0, min(x, vp.width() - sz.width()))
        y = max(0, min(y, vp.height() - sz.height()))
        tb.move(x, y)
        tb.show()
        tb.raise_()

    def _on_group_button_clicked(self) -> None:
        sel = self._selected_groupable_items()
        if not sel:
            return
        mode = self._compute_group_btn_mode(sel)
        if mode == "ungroup":
            # 优先看选区里直接选中的 GroupItem(单击 group 框 + Ctrl+G 的路径)
            groups_in_sel = [it for it in sel if isinstance(it, GroupItem)]
            if groups_in_sel:
                self.ungroup_group(groups_in_sel[0].group_id)
                return
            # 否则成员路径:全部节点都属于同一 group → 解该 group
            node_like = [it for it in sel if not isinstance(it, GroupItem)]
            grp = self._group_of(node_like[0]) if node_like else None
            if grp is None:
                return
            self.ungroup_group(grp.group_id)
        else:
            self.create_group_from_selection()

    # ------------------------------------------------------------------
    # 收纳 / 禁用 mode 计算 + 批量切换
    # ------------------------------------------------------------------
    def _selected_executable_nodes(self) -> List[NodeItem]:
        """工具栏 collapse/disable 只作用于真正的 NodeItem 且排除 Placeholder/Note。

        Note(NoteItem) 不参与执行,无禁用语义;Placeholder 无 set_collapsed
        /set_disabled 方法。返回空列表时 toggle 是 no-op。
        """
        out: List[NodeItem] = []
        for it in self._scene.selectedItems():
            if isinstance(it, NodeItem) and not isinstance(it, PlaceholderNodeItem):
                # 排除 NoteItem(它没继承 NodeItem,但保险一道)
                if isinstance(it, NoteItem):
                    continue
                out.append(it)
        return out

    def _compute_collapse_btn_mode(self, sel: List[Any]) -> str:
        nodes = [it for it in sel if isinstance(it, NodeItem)
                 and not isinstance(it, (NoteItem, PlaceholderNodeItem))]
        if not nodes:
            return "collapse"
        if all(getattr(n, "_collapsed", False) for n in nodes):
            return "expand"
        return "collapse"

    def _compute_disable_btn_mode(self, sel: List[Any]) -> str:
        nodes = [it for it in sel if isinstance(it, NodeItem)
                 and not isinstance(it, (NoteItem, PlaceholderNodeItem))]
        if not nodes:
            return "disable"
        if all(getattr(n, "_disabled", False) for n in nodes):
            return "enable"
        return "disable"

    def _toggle_collapse_selected(self) -> None:
        nodes = self._selected_executable_nodes()
        if not nodes:
            return
        # 任一未收纳 → 全部收纳;全部已收纳 → 全部展开
        target = not all(getattr(n, "_collapsed", False) for n in nodes)
        changes = [(n.node_id, bool(n._collapsed), target) for n in nodes
                   if bool(n._collapsed) != target]
        if not changes:
            return
        if self._in_undo:
            for nid, _, new in changes:
                it = self._instances.get(nid)
                if isinstance(it, NodeItem) and hasattr(it, "set_collapsed"):
                    it.set_collapsed(new)
            return
        from .undo import ToggleNodeFlagCmd
        self._undo_stack.push(ToggleNodeFlagCmd(self, "_collapsed", changes))

    def _toggle_disable_selected(self) -> None:
        nodes = self._selected_executable_nodes()
        if not nodes:
            return
        target = not all(getattr(n, "_disabled", False) for n in nodes)
        changes = [(n.node_id, bool(n._disabled), target) for n in nodes
                   if bool(n._disabled) != target]
        if not changes:
            return
        if self._in_undo:
            for nid, _, new in changes:
                it = self._instances.get(nid)
                if isinstance(it, NodeItem) and hasattr(it, "set_disabled"):
                    it.set_disabled(new)
            return
        from .undo import ToggleNodeFlagCmd
        self._undo_stack.push(ToggleNodeFlagCmd(self, "_disabled", changes))

    # ------------------------------------------------------------------
    # 色盘按钮 —— per-canvas 颜色覆写
    # ------------------------------------------------------------------
    def _color_targets(self) -> List[Any]:
        """选区里所有支持 set_color_override 的 item (NodeItem / NoteItem /
        GroupItem)。Placeholder 不支持。"""
        return [
            it for it in self._selected_groupable_items()
            if hasattr(it, "set_color_override")
        ]

    def _initial_color_for_picker(self, target: Any) -> str:
        """挑选首个 target 的"当前实际渲染色"作为色盘初始 hex。

        NodeItem → 标题栏色 (_title_bar_color);
        GroupItem → canvas_group_title_bg (覆写时是 base.alpha=0x40,
            但色盘看的是 base 本身,所以直接给现有 override 或主题色);
        NoteItem → canvas_note_title_bg / override。
        """
        from PyQt6.QtGui import QColor

        ov = getattr(target, "_color_override", "") or ""
        if ov and QColor(ov).isValid():
            return QColor(ov).name().upper()
        if isinstance(target, NodeItem) and not isinstance(target, NoteItem):
            try:
                c = target._title_bar_color()
                return c.name().upper()
            except Exception:
                pass
        if isinstance(target, GroupItem):
            from unitport_sdk import Config
            raw = Config.get_color("canvas_group_title_bg", "#F6D39340")
            # 去掉 alpha,只把 RGB 作为色盘初始值
            return _strip_alpha(raw)
        if isinstance(target, NoteItem):
            from unitport_sdk import Config
            return _strip_alpha(Config.get_color("canvas_note_title_bg", "#2A3140"))
        return "#67CCF5"

    def _on_selection_color_clicked(self, anchor: QPoint) -> None:
        """色盘按钮左键 → 弹 ColorPickerPopup,作用于整个选区。

        - 实时预览:color_changed 信号 → 所有 target.set_color_override(hex)
        - 确认:accepted_color → push 一条 ColorOverrideCmd 进 undo stack
        - 取消:cancelled → 还原到 popup 打开前快照
        """
        targets = self._color_targets()
        if not targets:
            return
        # 关闭可能残留的旧 popup,避免双开。
        if self._color_picker_popup is not None:
            try:
                self._color_picker_popup.close()
            except Exception:
                pass
            self._color_picker_popup = None

        # Snapshot 每个 target 当前的 override 字符串,取消时回灌。
        snapshot: Dict[str, str] = {}
        for t in targets:
            tid = self._color_target_id(t)
            if tid:
                snapshot[tid] = getattr(t, "_color_override", "") or ""
        self._color_picker_snapshot = snapshot

        initial = self._initial_color_for_picker(targets[0])

        # 延迟 import,避免 page.py 顶层硬绑 widgets 子树。
        from application.ui.widgets.color_picker_popup import ColorPickerPopup

        popup = ColorPickerPopup(initial, parent=self._view)
        self._color_picker_popup = popup

        def _on_preview(hex_: str) -> None:
            for it in self._color_targets():
                try:
                    it.set_color_override(hex_)
                except Exception:
                    pass

        def _on_accept(hex_: str) -> None:
            self._color_picker_popup = None
            self._commit_color_override(snapshot, hex_)

        def _on_cancel() -> None:
            self._color_picker_popup = None
            # 还原:逐 id 找回原 item 并恢复其 override (item 可能在 popup
            # 打开期间被删除 → 直接跳过)。
            for tid, prev in snapshot.items():
                it = self._instances.get(tid) or self._groups.get(tid)
                if it is not None and hasattr(it, "set_color_override"):
                    try:
                        it.set_color_override(prev)
                    except Exception:
                        pass
            self._color_picker_snapshot = {}

        popup.color_changed.connect(_on_preview)
        popup.accepted_color.connect(_on_accept)
        popup.cancelled.connect(_on_cancel)
        popup.show_at(anchor)

    def _on_selection_color_reset(self) -> None:
        """色盘按钮右键 → 清空当前选区所有 item 的 color_override。"""
        targets = self._color_targets()
        if not targets:
            return
        snapshot: Dict[str, str] = {}
        for t in targets:
            tid = self._color_target_id(t)
            if not tid:
                continue
            cur = getattr(t, "_color_override", "") or ""
            if cur:
                snapshot[tid] = cur
        if not snapshot:
            return
        self._commit_color_override(snapshot, "")

    def _color_target_id(self, item: Any) -> Optional[str]:
        """统一 NodeItem / NoteItem / GroupItem 的 id 取法。"""
        if isinstance(item, GroupItem):
            return getattr(item, "group_id", None)
        return getattr(item, "node_id", None)

    def _commit_color_override(
        self,
        old_map: Dict[str, str],
        new_hex: str,
    ) -> None:
        """构造 ColorOverrideCmd 并 push;空选区 / 无变化时 no-op。

        old_map: {target_id: previous_override_hex_or_empty}
        new_hex: 全部 target 的新覆写;"" 表示清空
        """
        # 过滤无变化的 target —— 命令本身也会再过一遍,但提前过滤可以避免
        # 创建一个 .setObsolete(True) 的空命令。
        changes: List[Tuple[str, str, str]] = []
        for tid, prev in old_map.items():
            it = self._instances.get(tid) or self._groups.get(tid)
            if it is None or not hasattr(it, "set_color_override"):
                continue
            if (prev or "") == (new_hex or ""):
                continue
            changes.append((tid, prev or "", new_hex or ""))
        if not changes:
            # 已经全部 = new_hex,确保 UI 状态对齐 (preview 中途可能被改过)。
            for tid in old_map:
                it = self._instances.get(tid) or self._groups.get(tid)
                if it is not None and hasattr(it, "set_color_override"):
                    try:
                        it.set_color_override(new_hex)
                    except Exception:
                        pass
            return
        if self._in_undo:
            for tid, _, nv in changes:
                it = self._instances.get(tid) or self._groups.get(tid)
                if it is not None and hasattr(it, "set_color_override"):
                    it.set_color_override(nv)
            return
        from .undo import ColorOverrideCmd
        self._undo_stack.push(ColorOverrideCmd(self, changes))

    def _toggle_collapse_for_node(self, node_id: str) -> None:
        """chevron 单击 + 节点体双击共用入口。"""
        it = self._instances.get(node_id)
        if not isinstance(it, NodeItem) or isinstance(it, (NoteItem, PlaceholderNodeItem)):
            return
        old = bool(getattr(it, "_collapsed", False))
        new = not old
        if self._in_undo:
            if hasattr(it, "set_collapsed"):
                it.set_collapsed(new)
            return
        from .undo import ToggleNodeFlagCmd
        self._undo_stack.push(
            ToggleNodeFlagCmd(self, "_collapsed", [(node_id, old, new)])
        )

    # ---- 公开操作 ----
    def create_group_from_selection(self) -> Optional[str]:
        """把当前选区中的 NodeItem / NoteItem 包进一个新 GroupItem.

        Returns: 新 group_id;选区为空则 None。
        """
        sel = [
            it for it in self._scene.selectedItems()
            if isinstance(it, (NodeItem, NoteItem, PlaceholderNodeItem))
        ]
        if not sel:
            return None
        member_ids = [it.node_id for it in sel]

        # 计算外包矩形(成员 sceneBoundingRect union + padding)
        bounds: Optional[QRectF] = None
        for it in sel:
            br = it.sceneBoundingRect()
            bounds = br if bounds is None else bounds.united(br)
        if bounds is None:
            return None
        from .group_item import PADDING, TITLE_H as GROUP_TITLE_H
        pad = PADDING
        rect = QRectF(
            bounds.left() - pad,
            bounds.top() - pad - GROUP_TITLE_H,
            bounds.width() + 2 * pad,
            bounds.height() + 2 * pad + GROUP_TITLE_H,
        )

        # 旧 group 成员摘除快照(便于 undo)
        evicted: List[Dict[str, str]] = []
        removed_old_groups: List[Dict[str, Any]] = []
        seen_old_gids: set = set()
        for mid in member_ids:
            grp = next((g for g in self._groups.values() if mid in g.member_ids), None)
            if grp is None or grp.group_id in seen_old_gids:
                if grp is not None:
                    seen_old_gids.add(grp.group_id)
                continue
            seen_old_gids.add(grp.group_id)
            removed_old_groups.append(grp.serialize())
        for mid in member_ids:
            grp = next((g for g in self._groups.values() if mid in g.member_ids), None)
            if grp is not None:
                evicted.append({"old_group": grp.group_id, "node": mid})

        import uuid
        gid = uuid.uuid4().hex[:12]
        snapshot: Dict[str, Any] = {
            "id": gid,
            "title": tr("canvas.group.default_title", "Group"),
            "members": member_ids,
            "rect": {
                "x": float(rect.x()),
                "y": float(rect.y()),
                "w": float(rect.width()),
                "h": float(rect.height()),
            },
            "evicted": evicted,
            "removed_old_groups": removed_old_groups,
        }
        if self._in_undo:
            self._create_group_raw(snapshot)
            return gid
        from .undo import CreateGroupCmd
        self._undo_stack.push(CreateGroupCmd(self, snapshot))
        return gid

    def ungroup_group(self, group_id: str) -> None:
        grp = self._groups.get(group_id)
        if grp is None:
            return
        snap = grp.serialize()
        if self._in_undo:
            self._delete_group_raw(group_id)
            return
        from .undo import UngroupCmd
        self._undo_stack.push(UngroupCmd(self, snap))

    # ---- raw helpers(供 UndoCommand 调用) ----
    def _create_group_raw(self, snapshot: Dict[str, Any]) -> None:
        # 先把旧 group(若被掏空)从 scene 清掉,再为待加入成员把他们从旧 group 摘除
        for old in snapshot.get("removed_old_groups", []):
            ogid = str(old.get("id") or "")
            if ogid and ogid in self._groups:
                self._scene.removeItem(self._groups[ogid])
                self._groups.pop(ogid, None)
        for entry in snapshot.get("evicted", []):
            ogid = str(entry.get("old_group") or "")
            nid = str(entry.get("node") or "")
            if ogid in self._groups and nid:
                old_grp = self._groups[ogid]
                old_grp.member_ids = [m for m in old_grp.member_ids if m != nid]
                if not old_grp.member_ids:
                    self._scene.removeItem(old_grp)
                    self._groups.pop(ogid, None)
        rect_dict = snapshot.get("rect") or {}
        rect = QRectF(
            float(rect_dict.get("x", 0.0)),
            float(rect_dict.get("y", 0.0)),
            float(rect_dict.get("w", 200.0)),
            float(rect_dict.get("h", 200.0)),
        )
        grp = GroupItem(
            snapshot["id"],
            snapshot.get("title") or tr("canvas.group.default_title", "Group"),
            list(snapshot.get("members") or []),
            rect,
        )
        self._scene.addItem(grp)
        self._groups[snapshot["id"]] = grp

    def _undo_create_group_raw(self, snapshot: Dict[str, Any]) -> None:
        # 销毁新 group,重建被掏空的旧 group(若有)
        gid = snapshot.get("id")
        if gid and gid in self._groups:
            self._scene.removeItem(self._groups[gid])
            self._groups.pop(gid, None)
        for old in snapshot.get("removed_old_groups", []):
            ogid = str(old.get("id") or "")
            if not ogid or ogid in self._groups:
                continue
            rect_dict = old.get("rect") or {}
            rect = QRectF(
                float(rect_dict.get("x", 0.0)),
                float(rect_dict.get("y", 0.0)),
                float(rect_dict.get("w", 200.0)),
                float(rect_dict.get("h", 200.0)),
            )
            grp = GroupItem(
                ogid,
                old.get("title") or tr("canvas.group.default_title", "Group"),
                list(old.get("members") or []),
                rect,
            )
            self._scene.addItem(grp)
            self._groups[ogid] = grp
        # 把被 evicted 的成员归还到他们的旧 group(若旧 group 现在还在)
        # 因为旧 group 在 _create_group_raw 中可能已经被销毁(member 全空),
        # 此处先尝试找到对应的 removed_old_groups 条目重建。
        # removed_old_groups 重建已含原 members 列表 → 已恢复关系。

    def _delete_group_raw(self, group_id: Optional[str]) -> None:
        if not group_id:
            return
        grp = self._groups.pop(group_id, None)
        if grp is not None:
            self._scene.removeItem(grp)

    def _restore_group_raw(self, snapshot: Dict[str, Any]) -> None:
        gid = snapshot.get("id")
        if not gid or gid in self._groups:
            return
        rect_dict = snapshot.get("rect") or {}
        rect = QRectF(
            float(rect_dict.get("x", 0.0)),
            float(rect_dict.get("y", 0.0)),
            float(rect_dict.get("w", 200.0)),
            float(rect_dict.get("h", 200.0)),
        )
        grp = GroupItem(
            gid,
            snapshot.get("title") or tr("canvas.group.default_title", "Group"),
            list(snapshot.get("members") or []),
            rect,
        )
        self._scene.addItem(grp)
        self._groups[gid] = grp

    # ------------------------------------------------------------------
    # AI Build 确定性排布 (SYSTEM step) —— move nodes + 区块框 upsert
    # ------------------------------------------------------------------
    def _move_node_raw(self, instance_id: str, x: float, y: float) -> bool:
        """Reposition a live node (raw; NOT undo-aware). False if it's missing.

        Matches the AI Build run's raw-mutation model — the whole run is
        checkpoint/restore-able (bridge.checkpoint), not per-op undo."""
        item = self._instances.get(str(instance_id))
        if item is None:
            return False
        item.setPos(float(x), float(y))
        return True

    #: gid prefix for AI-Build-owned block region boxes (rebuilt every layout;
    #: user-created groups use uuid hex ids and are never touched here).
    AI_BUILD_REGION_PREFIX = "aibuild_region_"

    def _current_view_center(self) -> "QPointF":
        """Scene-coord centre of what the user is currently looking at."""
        return self._view.mapToScene(self._view.viewport().rect().center())

    #: vertical gap between stacked nodes in a column (scene units).
    AI_BUILD_ROW_GAP = 48.0

    def _restack_columns_live(
        self, positions: Dict[str, Any]
    ) -> Dict[str, "QPointF"]:
        """Re-derive a LOCAL layout that keeps the plan's columns/order but sets
        each node's ``y`` from its **live rendered height**, not the plan's.

        The plan groups nodes into columns by ``x`` (planned from node widths);
        we keep that ``x`` but re-stack each column top-down using the live
        ``sceneBoundingRect().height()``. This is what kills the "exaggerated
        vertical gap": a node whose serialized height exceeds what it actually
        renders (collapsed / stale) no longer pushes the next node far past its
        visible bottom — the gap is always exactly ``AI_BUILD_ROW_GAP``."""
        columns: Dict[float, List[Tuple[float, str]]] = {}
        for nid, xy in (positions or {}).items():
            if str(nid) not in self._instances:
                continue
            try:
                lx, ly = round(float(xy[0]), 1), float(xy[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            columns.setdefault(lx, []).append((ly, str(nid)))

        laid: Dict[str, QPointF] = {}
        for lx in sorted(columns):
            y = 0.0  # all columns share a common top (top-aligned dataflow)
            for _order_y, nid in sorted(columns[lx]):
                laid[nid] = QPointF(lx, y)
                h = float(self._instances[nid].sceneBoundingRect().height())
                y += h + self.AI_BUILD_ROW_GAP
        return laid

    def apply_ai_build_layout(
        self,
        positions: Dict[str, Any],
        regions: Any,
        *,
        draw_regions: bool = True,
    ) -> None:
        """Apply a deterministic AI-Build layout to the LIVE canvas.

        ``positions`` is ``{node_id: (x, y)}`` in a local frame; the whole plan is
        translated so its centre sits on the current view (issue: layout ignored
        the view centre). The order is strict — **all nodes are positioned first,
        then the region boxes are drawn from the final node geometry** (issue:
        boxes drawn before/around scattered nodes). ``draw_regions`` is False for
        the per-block passes (position only, no boxes) and True once at the end.

        Region boxes wrap each block's nodes tightly (live ``sceneBoundingRect`` +
        padding) and, because the columns are packed with clear gaps, never
        overlap (issue: box-in-box). Finally
        :meth:`_recompute_all_group_memberships` re-derives EVERY group's members
        from geometry, so a box owns exactly the nodes inside it and a box the
        layout emptied (e.g. a pre-existing user group whose nodes moved away)
        can no longer cascade-delete far-off nodes (issue: deleting an empty box
        deleted nodes). Raw + idempotent; only ``aibuild_region_*`` groups are
        (re)built — user groups are never removed here."""
        from .group_item import GroupItem, PADDING, TITLE_H as _G_TITLE_H

        if not positions:
            return

        # 1) re-stack columns using LIVE heights (kills exaggerated vertical gaps)
        #    and place every node at its local slot first.
        laid = self._restack_columns_live(positions)
        if not laid:
            return
        for nid, pt in laid.items():
            self._move_node_raw(str(nid), pt.x(), pt.y())

        # 2) measure the REAL bounding box (a node's sceneBoundingRect is offset
        #    from its pos by its port stubs) and translate the whole layout so its
        #    centre lands on the current view.
        bbox: Optional[QRectF] = None
        for nid in laid:
            br = self._instances[nid].sceneBoundingRect()
            bbox = br if bbox is None else bbox.united(br)
        if bbox is None:
            return
        center = self._current_view_center()
        dx = center.x() - bbox.center().x()
        dy = center.y() - bbox.center().y()
        if dx or dy:
            for nid in laid:
                p = self._instances[nid].pos()
                self._move_node_raw(str(nid), p.x() + dx, p.y() + dy)

        if not draw_regions:
            return

        # 3) rebuild the AI-Build region boxes from the FINAL node geometry.
        for gid in [
            g for g in list(self._groups.keys())
            if str(g).startswith(self.AI_BUILD_REGION_PREFIX)
        ]:
            self._delete_group_raw(gid)

        for region in (regions or ()):
            region_id = str(getattr(region, "region_id", "") or "")
            member_ids = [str(m) for m in (getattr(region, "member_ids", ()) or ())]
            if not region_id or not member_ids:
                continue
            bounds: Optional[QRectF] = None
            live_members: List[str] = []
            for mid in member_ids:
                item = self._instances.get(mid)
                if item is None:
                    continue
                live_members.append(mid)
                br = item.sceneBoundingRect()
                bounds = br if bounds is None else bounds.united(br)
            if bounds is None:
                continue
            rect = QRectF(
                bounds.left() - PADDING,
                bounds.top() - PADDING - _G_TITLE_H,
                bounds.width() + 2 * PADDING,
                bounds.height() + 2 * PADDING + _G_TITLE_H,
            )
            title = tr(
                getattr(region, "title_key", "") or "",
                getattr(region, "title", "") or "",
            )
            grp = GroupItem(region_id, title, live_members, rect)
            self._scene.addItem(grp)
            self._groups[region_id] = grp

        # 4) make membership authoritative-by-geometry for ALL groups: each box
        #    owns exactly the nodes inside it; a box the layout emptied owns
        #    nothing → its deletion can't cascade to nodes it no longer contains.
        self._recompute_all_group_memberships()

    # ------------------------------------------------------------------
    # 几何成员关系 —— 由 group resize / node 拖动结束触发
    # ------------------------------------------------------------------
    def _recompute_group_members(self, group: "GroupItem") -> List[str]:
        """以 group 当前 frame_scene_rect 重新扫描所有节点,设置 member_ids。

        包含规则:item 的 scene-bounding center 落在 group 框内 → 成员。
        节点同时落在多个 group 内 → 此函数只更新传入的 group(不解决跨 group 冲突,
        全局解冲突由 _recompute_all_group_memberships 在所有 group 都过一遍后做)。
        """
        scene_rect = group.frame_scene_rect()
        ids: List[str] = []
        for inst_id, item in self._instances.items():
            if not isinstance(item, (NodeItem, NoteItem, PlaceholderNodeItem)):
                continue
            center = item.sceneBoundingRect().center()
            if scene_rect.contains(center):
                ids.append(inst_id)
        group.member_ids = ids
        return ids

    def _recompute_all_group_memberships(self) -> None:
        """对所有 group 扫描重算成员;跨 group 冲突走"最小框优先"规则。

        触发源:节点单独拖动 mouseReleaseEvent 之后(经 page._pending_move 路径
        push 完 MoveNodeCmd 再调一遍);group resize 结束(在 ResizeGroupCmd
        push 之前已经预算了新 member_ids,但其它 group 可能因为这次 resize 失去
        成员所以也要扫一遍);ResizeGroupCmd.undo 复原后也要扫,让其它 group 重新
        拿回它们当时的成员。**幂等** —— 多次调用等价。
        """
        if not self._groups:
            return
        # 先按"框面积升序"列出所有 group,小框优先认领节点 → 嵌套关系下内层 group 赢。
        groups_by_area = sorted(
            self._groups.values(),
            key=lambda g: g.frame_scene_rect().width() * g.frame_scene_rect().height(),
        )
        claimed: Dict[str, str] = {}  # node_id → group_id
        for grp in groups_by_area:
            scene_rect = grp.frame_scene_rect()
            new_members: List[str] = []
            for inst_id, item in self._instances.items():
                if not isinstance(item, (NodeItem, NoteItem, PlaceholderNodeItem)):
                    continue
                if inst_id in claimed:
                    continue
                center = item.sceneBoundingRect().center()
                if scene_rect.contains(center):
                    new_members.append(inst_id)
                    claimed[inst_id] = grp.group_id
            grp.member_ids = new_members

    def _serialize_selection(self) -> Optional[Dict[str, Any]]:
        nodes = self._selected_nodes()
        if not nodes:
            return None
        # 复用 to_workflow_dict 的 IRNode 结构，但只取选中的
        full = self.to_workflow_dict()
        sel_ids = {n.node_id for n in nodes}
        nodes_out = [n for n in full["nodes"] if n["id"] in sel_ids]
        # 只保留两端都在选中集内的边
        edges_out = [
            e for e in full["edges"]
            if e.get("source_node") in sel_ids and e.get("target_node") in sel_ids
        ]
        return {
            "version": "1.0.0",
            "metadata": {"copy": True},
            "nodes": nodes_out,
            "edges": edges_out,
        }

    def _selected_groups(self) -> List[GroupItem]:
        return [it for it in self._scene.selectedItems() if isinstance(it, GroupItem)]

    def _serialize_full_delete_snapshot(self) -> Optional[Dict[str, Any]]:
        """构造删除选中所需的完整快照（节点 + 内部边 + 跨界边/独立选边 + 分组框）.

        - ``nodes`` / ``edges``: ``_serialize_selection`` 同结构（用于 undo 时
          ``_apply_snapshot_raw`` 原地复原）；
        - ``loose_edges``: 一端在选中节点集内 + 另一端在外（跨界边），或独立选中
          且两端都在外的边——undo 时单独 ``_reconnect_edge_raw``（外端节点仍在画布）。
        - ``groups``: 被选中的 GroupItem 快照 + 因为成员被一同删除而自然消失的
          其它 group 快照(用于精确 undo 还原)。**删 group 会级联删其全部 members。**
        """
        nodes = self._selected_nodes()
        edges = self._selected_edges()
        sel_groups = self._selected_groups()
        if not nodes and not edges and not sel_groups:
            return None

        # 级联展开:选中的 group 把其全部 member node 一并纳入删除范围
        cascade_node_ids: set = set()
        for grp in sel_groups:
            cascade_node_ids.update(grp.member_ids)
        # 选中的 node + 级联 node 合并
        sel_node_ids = {n.node_id for n in nodes} | cascade_node_ids

        full = self.to_workflow_dict()
        sel_edge_ids = {e.edge_id for e in edges}
        nodes_out = [n for n in full["nodes"] if n["id"] in sel_node_ids]
        inner: List[Dict[str, Any]] = []
        loose: List[Dict[str, Any]] = []
        for e in full["edges"]:
            s = e.get("source_node")
            t = e.get("target_node")
            in_s = s in sel_node_ids
            in_t = t in sel_node_ids
            if in_s and in_t:
                inner.append(e)
            elif in_s or in_t or e.get("id") in sel_edge_ids:
                loose.append(e)

        # groups 快照:被直接选中的 + 因成员清空而需要被销毁的 group
        groups_to_delete: Dict[str, Dict[str, Any]] = {
            grp.group_id: grp.serialize() for grp in sel_groups
        }
        for grp in self._groups.values():
            if grp.group_id in groups_to_delete:
                continue
            remaining = [m for m in grp.member_ids if m not in sel_node_ids]
            if not remaining and grp.member_ids:
                # 所有成员都被删 → group 因此自然销毁,需要进 undo 快照
                groups_to_delete[grp.group_id] = grp.serialize()

        return {
            "version": "1.0.0",
            "metadata": {"delete": True},
            "nodes": nodes_out,
            "edges": inner,
            "loose_edges": loose,
            "groups": list(groups_to_delete.values()),
        }

    def _copy_selected(self) -> None:
        snap = self._serialize_selection()
        if snap is None:
            log_debug("[ui.canvas] copy: 无选中节点")
            return
        self._clipboard = snap
        log_debug(f"[ui.canvas] copied {len(snap['nodes'])} 节点 + {len(snap['edges'])} 边")

    def _duplicate_selected(self) -> None:
        snap = self._serialize_selection()
        if snap is None:
            return
        offset = QPointF(20.0, 20.0)
        if self._in_undo:
            self._apply_snapshot_raw(snap, offset=offset)
            return
        from .undo import PasteCmd
        self._undo_stack.push(PasteCmd(self, snap, offset, text="Duplicate"))

    def _paste(self, anchor: Optional[QPointF]) -> None:
        if not self._clipboard:
            log_debug("[ui.canvas] paste: 剪贴板为空")
            return
        # 跨后端粘贴守卫：CanvasPage 实例跨画布加载复用，剪贴板不随加载清空，
        # 因此内容可能来自另一块（不同后端）画布。含跨后端节点时整批拒绝——
        # 不静默丢弃部分节点（§8），同后端跨画布复制粘贴不受影响。
        if self._backend_id is not None:
            from registers import nodes as nodes_registry
            offenders = sorted({
                str(n.get("schema_id", "") or "")
                for n in self._clipboard.get("nodes", [])
                if (m := nodes_registry.get_manifest(
                    str(n.get("schema_id", "") or ""))) is not None
                and not m.applies_to_backend(self._backend_id)
            })
            if offenders:
                log_error(
                    f"[ui.canvas] paste: refused — clipboard contains nodes not "
                    f"available for backend '{self._backend_id}': {offenders!r}"
                )
                return
        # 计算偏移：把 clip 节点的几何重心移到 anchor
        if anchor is None:
            offset = QPointF(20.0, 20.0)
        else:
            xs = [n.get("ui", {}).get("x", 0.0) for n in self._clipboard["nodes"]]
            ys = [n.get("ui", {}).get("y", 0.0) for n in self._clipboard["nodes"]]
            if xs and ys:
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                offset = QPointF(anchor.x() - cx, anchor.y() - cy)
            else:
                offset = QPointF(0.0, 0.0)
        if self._in_undo:
            self._apply_snapshot_raw(self._clipboard, offset=offset)
            return
        from .undo import PasteCmd
        self._undo_stack.push(PasteCmd(self, self._clipboard, offset, text="Paste"))

    # ===========================================================
    # Phase 2: 右键 context menu
    # ===========================================================

    def _build_context_menu(self, event: QGraphicsSceneContextMenuEvent) -> Optional[QMenu]:
        scene_pos = event.scenePos()
        self._last_context_scene_pos = scene_pos
        clicked = self._scene.itemAt(scene_pos, self._view.transform())

        # 端口右键不弹菜单（避免遮挡端口拖拽）
        if clicked is not None and clicked.data(0) == "port":
            return None

        menu = QMenu(self._view)

        # 找到 NodeItem / ConnectionItem 祖先
        node = self._ancestor_of_type(clicked, NodeItem)
        if node is not None:
            self._build_node_menu(menu, node)
            return menu

        # 连线右键不再弹菜单（删除连线改为 hover + Delete；右键删除已移除：太
        # 容易误触）。命中连线 → 不弹菜单，让右键回落为平移。
        if self._ancestor_of_type(clicked, ConnectionItem) is not None:
            return None

        self._build_blank_menu(menu, scene_pos)
        return menu

    @staticmethod
    def _ancestor_of_type(item, kind):
        cur = item
        while cur is not None:
            if isinstance(cur, kind):
                return cur
            cur = cur.parentItem()
        return None

    def _build_node_menu(self, menu: QMenu, node: NodeItem) -> None:
        act_dup = menu.addAction(tr("canvas.menu.duplicate", "Duplicate (Ctrl+D)"))
        act_copy = menu.addAction(tr("canvas.menu.copy", "Copy (Ctrl+C)"))
        act_disc = menu.addAction(tr("canvas.menu.disconnect_all", "Disconnect All Edges"))
        menu.addSeparator()
        act_del = menu.addAction(tr("canvas.menu.delete", "Delete"))

        def _focus_only(n):
            for it in self._scene.selectedItems():
                it.setSelected(False)
            n.setSelected(True)

        act_dup.triggered.connect(lambda: (_focus_only(node), self._duplicate_selected()))
        act_copy.triggered.connect(lambda: (_focus_only(node), self._copy_selected()))
        act_disc.triggered.connect(lambda: self._disconnect_node(node))
        act_del.triggered.connect(lambda: (_focus_only(node), self._delete_selected()))

    def _user_disconnect_edge(self, edge: ConnectionItem) -> None:
        sn = edge.src_port.parent_node()
        dn = edge.dst_port.parent_node()
        if sn is None or dn is None:
            edge.disconnect()
            return
        if self._in_undo:
            edge.disconnect()
            return
        from .undo import DisconnectCmd
        self._undo_stack.push(DisconnectCmd(
            self, edge.edge_id,
            sn.node_id, edge.src_port.port_name,
            dn.node_id, edge.dst_port.port_name,
        ))

    def _build_blank_menu(self, menu: QMenu, scene_pos: QPointF) -> None:
        # Add Node ▸（按 layer 分组，最多 27 个 builtin + 三方）
        add_menu = menu.addMenu(tr("canvas.menu.add_node", "Add Node"))
        self._populate_add_node_menu(add_menu, scene_pos)
        menu.addSeparator()
        if self._clipboard:
            act_paste = menu.addAction(tr("canvas.menu.paste_here", "Paste Here (Ctrl+V)"))
            act_paste.triggered.connect(lambda: self._paste(scene_pos))

    def _populate_add_node_menu(self, root: QMenu, scene_pos: QPointF) -> None:
        from registers import nodes as nodes_registry

        # 按 layer 分组。后端过滤与 Node Library 面板同一谓词
        # （manifest.backends）：已绑后端的画布绝不提供跨后端节点；
        # fresh 未绑画布（backend_id is None）显示全量。
        groups: Dict[str, List[Any]] = {}
        for m in nodes_registry.list_nodes(backend=self._backend_id):
            layer = (m.layer or "Custom").upper()
            groups.setdefault(layer, []).append(m)

        # layer 显示顺序
        order = ["A", "B", "C", "D", "IL", "CUSTOM"]
        for key in order + [k for k in groups if k not in order]:
            if key not in groups:
                continue
            sub = root.addMenu(f"Layer {key}" if key in ("A", "B", "C", "D", "IL") else key)
            for m in sorted(groups[key], key=lambda x: x.id):
                act = sub.addAction(m.id)
                # 闭包陷阱：用 lambda kwarg 锁定 m
                act.triggered.connect(
                    lambda checked=False, mid=m.id, sx=scene_pos.x(), sy=scene_pos.y():
                    self.spawn_node(mid, x=sx, y=sy)
                )

    def _disconnect_node(self, node: NodeItem) -> None:
        """断开节点的所有连线（节点本身保留）；走单条 DisconnectAllEdgesCmd."""
        edges_data: List[Dict[str, Any]] = []
        for p in node.iter_ports():
            for c in p.connections:
                sn = c.src_port.parent_node()
                dn = c.dst_port.parent_node()
                if sn is None or dn is None:
                    continue
                edges_data.append({
                    "id": c.edge_id,
                    "source_node": sn.node_id,
                    "source_port": c.src_port.port_name,
                    "target_node": dn.node_id,
                    "target_port": c.dst_port.port_name,
                })
        if not edges_data:
            return
        if self._in_undo:
            for e in edges_data:
                self._disconnect_edge_by_id_raw(e["id"])
            return
        from .undo import DisconnectAllEdgesCmd
        self._undo_stack.push(DisconnectAllEdgesCmd(self, node.node_id, edges_data))


__all__ = ["CanvasPage"]
