# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""RegistryModuleEditorPanel — 统一注册模块编辑器 / Unified registry-module editor.

DEMO 对应：
  - ``DEMO/bin/pages/training/reward_editor_panel.py``  (RewardEditorPanel)
  - ``DEMO/bin/pages/training/training_motion_editor_panel.py``  (TrainingMotionEditorPanel)

DEMO 这两个面板的 chrome 完全一致（左侧分组列表 + 中部 title/desc/tags +
中央内容区 + 底部状态栏 + OK/Cancel），只是中央内容区不同——前者是
源码编辑器（编辑 reward / termination / observation 函数定义），后者是
clip 路径选择器 + 每项 speed/advanced 字段。本文件把 chrome 抽成一套
``RegistryModuleEditorPanel``，content 区由 ``kind`` 切换。

视觉契约（按用户硬性要求）：
  - 所有可见文本 / 输入 / 按钮 **必须** 走 ``unitport_sdk`` 预制件 —
    ``setText`` / ``setButton`` / ``LaviLineEdit`` / ``LaviComboBox`` /
    ``LaviDoubleSpinBox``，禁止裸 ``QLabel`` / ``QLineEdit`` / ``QComboBox``。
  - "label: [input]" 同一行两端之间留 ``_LABEL_INPUT_SPACING`` 的固定间距，
    label 用 ``setText(kind="content")``，input 用对应 Lavi* 控件。
  - NEW / REMOVE / OK / Cancel 按钮一律 ``kind="border"``，spec 按语义：
        NEW    → spec="notice"  (highlight 黄)
        REMOVE → spec="danger"  (danger_zone 红)
        OK     → spec="save"    (safe_zone 绿)
        Cancel → spec="danger"  (danger_zone 红)
  - 源码编辑器 **必须** 是 SDK ``CodeEditorWidget`` —— 行号 / Python 高亮 /
    智能缩进 / Ctrl+Wheel 字号缩放等完整功能直接继承，禁止用普通
    ``QPlainTextEdit``。

入口工厂（thin wrapper）：
    ``open_reward_function_editor(parent, items, registry_id, *, kind="reward")``
    ``open_training_motion_editor(parent, items)``

颜色与字体走 ``Config.get_color`` / ``Config.get_font_size``（rule §1.5），
绝不在本文件内定义 hex 字面量。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDoubleValidator, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    CodeEditorWidget,
    Config,
    LaviComboBox,
    LaviDoubleSpinBox,
    LaviLineEdit,
    Paths,
    log_warning,
    setButton,
    setComboBox,
    setDoubleSpinBox,
    setLineEdit,
    setText,
    tr,
)

# Backed by ``src/scripts/`` — every preset's authoritative source / title /
# desc / backend tag / family tag lives in ``TaskModuleItem`` rows within
# the kind-specific sub-registries (REWARD_REGISTRY / IL_REWARD_REGISTRY /
# TERMINATION_REGISTRY / IL_TERMINATION_REGISTRY / IL_OBS_REGISTRY /
# IL_DISC_REGISTRY). ``scripts.lookup(key, kind, backend)`` resolves
# (kind, key, backend) → TaskModuleItem honouring the SB3 vs Isaac Lab
# disambiguation. The Editor reads ``il_inline`` from there as the saved
# source baseline; Save writes back via ``dataclasses.replace`` + emits a
# sidecar for the training subprocess to overlay.
try:
    from scripts import (  # type: ignore[import-not-found]
        emit_inline_overrides as _emit_inline_overrides,
        lookup as _scripts_lookup,
    )
    from scripts.query import ALL_REGISTRIES as _SCRIPTS_ALL_REGISTRIES  # type: ignore[import-not-found]
    from scripts.task_module import (  # type: ignore[import-not-found]
        TaskModuleItem as _TaskModuleItem,
        ALL_FAMILIES as _ALL_FAMILIES,
        LOCOMOTION_FAMILIES as _LOCOMOTION_FAMILIES,
        LEGGED_FAMILIES as _LEGGED_FAMILIES,
    )
    _SCRIPTS_AVAILABLE = True
except Exception as _scripts_import_err:  # pragma: no cover — defensive
    _scripts_lookup = None  # type: ignore[assignment]
    _emit_inline_overrides = None  # type: ignore[assignment]
    _SCRIPTS_ALL_REGISTRIES = {}
    _TaskModuleItem = None  # type: ignore[assignment]
    _ALL_FAMILIES = frozenset()  # type: ignore[assignment]
    _LOCOMOTION_FAMILIES = frozenset()  # type: ignore[assignment]
    _LEGGED_FAMILIES = frozenset()  # type: ignore[assignment]
    _SCRIPTS_AVAILABLE = False
    log_warning(f"[RegistryModuleEditorPanel] scripts import failed: {_scripts_import_err!r}")


# =============================================================================
# Public API
# =============================================================================


def open_reward_function_editor(
    parent: Optional[QWidget],
    items: Dict[str, Any],
    registry_id: str = "rewards",
    *,
    kind: str = "reward",
    canvas_backend: str = "sb3",
    conflicting_keys: Optional[set] = None,
    robot_sku: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """打开 Reward / Termination / Observation 函数编辑器（modal）.

    Args:
        parent: 父窗口（用于 modal centre）
        items: 当前节点的 ``{key: weight_or_meta}`` 字典
        registry_id: ``"rewards"`` / ``"il_rewards"`` / ``"terminations"`` /
                     ``"il_terminations"`` / ``"observations"`` / ``"il_observations"``
        kind: ``"reward"`` / ``"termination"`` / ``"observation"`` —
              切换中央内容区为源码编辑器并选择对应 stub 模板。
        canvas_backend: 所在 canvas 绑定的后端 kind (``"sb3"`` / ``"isaac_lab"``)。
            条目自身没有 ``engine`` 标签时，Engine 下拉默认值落到这个 kind，
            而不是固定 ``"sb3"``。
        conflicting_keys: term key set whose rows should render in
            ``danger_zone`` — those keys are also defined on a peer rewards
            node connected to the same training_motion item with a different
            value (see ``application.compiler.reward_conflicts``). ``None`` /
            empty = no highlighting.

    Returns:
        ``None``（用户取消）或更新后的 items dict。
    """
    dlg = RegistryModuleEditorPanel(
        kind=kind,
        registry_id=registry_id,
        items=items,
        canvas_backend=canvas_backend,
        conflicting_keys=conflicting_keys,
        robot_sku=robot_sku,
        parent=parent,
    )
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.applied_items()


def open_training_motion_editor(
    parent: Optional[QWidget],
    items: Dict[str, Any],
    *,
    canvas_backend: str = "sb3",
) -> Optional[Dict[str, Any]]:
    """打开 Training Motion 编辑器（modal）.

    Args:
        parent: 父窗口
        items: training_motion 节点的 ``training_items`` dict
        canvas_backend: canvas 绑定的后端 kind，用作 Engine 下拉缺省值

    Returns:
        ``None``（用户取消）或更新后的 items dict。
    """
    dlg = RegistryModuleEditorPanel(
        kind="training_motion",
        registry_id="training_items",
        items=items,
        canvas_backend=canvas_backend,
        parent=parent,
    )
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.applied_items()


# =============================================================================
# Internals
# =============================================================================


# Per-kind chrome / stub-code config — DEMO ``_KIND_TITLES`` / ``_KIND_STUB_TEMPLATES``.
_KIND_TITLES: Dict[str, str] = {
    "reward":         "Reward Function Editor",
    "termination":    "Termination Function Editor",
    "observation":    "Observation Function Editor",
    "discriminator":  "Discriminator Function Editor",
    "training_motion": "Training Motion Editor",
}

_REWARD_STUB = (
    "def {key}(env):\n"
    "    \"\"\"Custom {polarity} function — return shape (num_envs,).\"\"\"\n"
    "    import torch\n"
    "    return torch.zeros(env.num_envs, device=env.device)\n"
)
_TERMINATION_STUB = (
    "def {key}(env):\n"
    "    \"\"\"Custom termination — return bool tensor (num_envs,).\"\"\"\n"
    "    import torch\n"
    "    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)\n"
)
_OBSERVATION_STUB = (
    "def {key}(env):\n"
    "    \"\"\"Custom observation — return tensor (num_envs, dim).\"\"\"\n"
    "    import torch\n"
    "    return torch.zeros(env.num_envs, 1, device=env.device)\n"
)
_DISCRIMINATOR_STUB = (
    "def {key}(env):\n"
    "    \"\"\"Custom discriminator term.\"\"\"\n"
    "    import torch\n"
    "    return torch.zeros(env.num_envs, device=env.device)\n"
)

_KIND_STUB: Dict[str, str] = {
    "reward":         _REWARD_STUB,
    "termination":    _TERMINATION_STUB,
    "observation":    _OBSERVATION_STUB,
    "discriminator":  _DISCRIMINATOR_STUB,
}

_ENGINE_OPTIONS = ["sb3", "isaac_lab", "newton"]

# Layout / sizing constants — drive label↔input spacing rule.
_LABEL_INPUT_SPACING = 8     # px between "Label:" and its [input] on the same row
_FIELD_GAP = 14              # px between sibling field groups (e.g. Title group ↔ Desc group)
_ROW_VSPACING = 6            # px between stacked rows
_PANE_GAP = 15               # px between left list pane and right edit pane (splitter handle)
_BTN_W = 92
_BTN_H = 28
_DLG_PAD = 12


def _apply_small_font(widget: QWidget) -> None:
    """Force the widget's font to ``size_small`` (theme slot).

    Some Lavi* widgets hardcode ``size_normal`` in ``__init__`` (LaviLineEdit,
    LaviSpinBox, LaviDoubleSpinBox, LaviComboBox internals). The editor's
    visual contract uses ``size_small`` uniformly, so we pull the slot value
    once and overwrite the per-widget font after construction.
    """
    f = widget.font()
    f.setPixelSize(int(Config.get_font_size("size_small", 11)))
    widget.setFont(f)


class RegistryModuleEditorPanel(QDialog):
    """Unified 3-pane modal editor — see module docstring for layout.

    Signals:
        applied (dict): emitted when user clicks OK with the resulting items dict.
    """

    applied = pyqtSignal(dict)

    def __init__(
        self,
        *,
        kind: str,
        registry_id: str,
        items: Dict[str, Any],
        canvas_backend: str = "sb3",
        conflicting_keys: Optional[set] = None,
        robot_sku: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._registry_id = registry_id
        # Term keys flagged by reward_conflicts.compute_reward_term_conflicts —
        # rendered with danger_zone foreground in the list pane so the user
        # immediately sees which entries collide with a peer rewards node
        # sharing the same training_motion item. Stored as a frozenset so
        # mutation by the caller after construction doesn't leak in.
        self._conflicting_keys: set = (
            frozenset(conflicting_keys) if conflicting_keys else frozenset()
        )
        # Canvas-bound backend kind — used as the Engine combobox default
        # when an item has no explicit ``engine`` tag. Per project rule,
        # the canvas owns backend; the per-item engine tag is purely a
        # filtering hint, NOT an override.
        self._canvas_backend = canvas_backend if canvas_backend in _ENGINE_OPTIONS else "sb3"
        # Optional robot SKU — drives the Variant dropdown's family
        # hard-filter (rule §1.5 Stage 3). ``None`` keeps all variants
        # visible and shows a banner so the user knows filtering is off.
        self._robot_sku: Optional[str] = robot_sku or None
        # The Variant dropdown is built lazily inside ``_build_right_pane``;
        # bind the type slots here so static checkers stay happy.
        self._variant_combo: Optional[LaviComboBox] = None
        self._variant_banner: Optional[Any] = None
        # Deep copy items so any in-flight mutation (stub promotion on first
        # visit, edits before Cancel) doesn't leak back to the caller.
        import copy as _copy
        self._items: Dict[str, Any] = _copy.deepcopy(dict(items or {}))
        self._current_key: Optional[str] = None
        # Per-key edit-buffer cache — switching items NEVER discards
        # unsubmitted edits; we rehydrate the editor from this cache on
        # re-select. ``_items[key]['source']`` is the *saved* baseline,
        # written only by ``_save_current_script()``.
        self._content_cache: Dict[str, str] = {}
        # Per-key dirty set — authoritative source is the SDK editor's
        # SHA-1 hash compare (``CodeEditorWidget.is_modified()`` /
        # ``dirtyChanged(bool)``). On load we set the editor baseline to
        # the saved source via ``set_text(saved)``, then display the cached
        # buffer via ``setPlainText(buffer)`` — ``setPlainText`` does NOT
        # reset ``_baseline_hash``, so the hash check is precisely
        # ``buffer != saved``. Typing a space and deleting it returns the
        # buffer to the saved bytes → editor reports clean → key
        # un-marked dirty. No string compare ourselves.
        self._dirty_keys: set = set()
        # Suppress flag for our handlers during programmatic editor loads.
        # The load itself fires textChanged + dirtyChanged spuriously;
        # those are not user actions and shouldn't update _dirty_keys.
        self._suppress_editor_dirty: bool = False
        # Whether the current panel uses a source-code CodeEditorWidget +
        # Save button. Source-code kinds (reward / termination / observation /
        # discriminator) → True; training_motion → False (Advanced JSON
        # commits immediately, no per-key dirty tracking).
        self._has_save_workflow: bool = self._kind in _KIND_STUB

        # Full registry for (kind, canvas_backend) — drives the "show all
        # functions" view in the list pane. The list pane shows the union of
        # this with self._items, so the user can pick any registered function
        # to inspect / edit its source even when the node hasn't added it yet.
        # training_motion has no scripts registry → leave empty.
        self._full_registry: Dict[str, Any] = {}
        if self._has_save_workflow and _SCRIPTS_AVAILABLE:
            try:
                from scripts.query import query_registry as _qr  # type: ignore[import-not-found]
                self._full_registry = dict(
                    _qr(kind=self._kind, backend=self._canvas_backend) or {}
                )
            except Exception as _qr_err:
                log_warning(f"[RegistryModuleEditorPanel] query_registry failed: {_qr_err!r}")
                self._full_registry = {}

        # Per-key edit drafts for items NOT in self._items (the node's added
        # dict). When the user clicks an un-added function and edits its
        # title / desc / engine / family / source, the change lives here —
        # _save_current_script writes source to the registry; OK discards
        # everything in this dict (un-added stays un-added — per project
        # contract, Save only writes the registry, never adds to the node).
        self._unadded_overrides: Dict[str, Dict[str, Any]] = {}

        self.setWindowTitle(_KIND_TITLES.get(kind, "Module Editor"))
        self.setModal(True)
        self.resize(1080, 720)
        self._apply_chrome_style()

        self._build_ui()
        self._populate_list()
        # Initial selection: prefer the node's first added key; else first
        # registry entry; else nothing (empty registry + empty items).
        first_key: Optional[str] = None
        if self._items:
            first_key = next(iter(self._items.keys()))
        elif self._full_registry:
            first_key = next(iter(self._full_registry.keys()))
        if first_key is not None:
            self._select_key(first_key)
        else:
            self._refresh_content_panel()

    # ---- public ----

    def applied_items(self) -> Dict[str, Any]:
        """Return the in-memory items dict (post-edit)."""
        return dict(self._items)

    # ---- chrome / style ----

    def _apply_chrome_style(self) -> None:
        """Dialog chrome: bg + body QSS for QDialog frames + QListWidget background.

        Buttons / labels / inputs are individually themed by their Lavi*
        widgets (each carries its own QSS). Here we only paint the dialog
        background + the list pane + splitter handle.
        """
        bg = Config.get_color("bg_1", "#1E1E1E")
        bg2 = Config.get_color("bg_2", "#2A2A2A")
        fg = Config.get_color("main_t1", "#D6D3C7")
        border = Config.get_color("border_2", "#444444")
        hl = Config.get_color("highlight", "#F6D393")
        alt = Config.get_color("bg_1", "#1E1E1E")
        self.setStyleSheet(
            f"QDialog {{ background: {bg}; color: {fg}; }}"
            f"QFrame {{ background: transparent; }}"
            f"QListWidget {{ background: {bg2}; color: {fg}; "
            f"border: 1px solid {border}; border-radius: 4px; "
            f"selection-background-color: {hl}; selection-color: {alt}; "
            f"outline: 0; padding: 2px; }}"
            f"QListWidget::item {{ padding: 4px 6px; }}"
            f"QListWidget::item:hover {{ background: {Config.get_color('hover_2', '#4A4A4A')}; }}"
            f"QSplitter::handle {{ background: {border}; }}"
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(_DLG_PAD, _DLG_PAD, _DLG_PAD, _DLG_PAD)
        root.setSpacing(_ROW_VSPACING)

        # ── Splitter: left list / right content ──────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        # 15px breathing room between the file list pane and the editor pane.
        splitter.setHandleWidth(_PANE_GAP)
        root.addWidget(splitter, 1)

        # Left pane.
        splitter.addWidget(self._build_left_pane())
        # Right pane.
        splitter.addWidget(self._build_right_pane())
        splitter.setSizes([260, 820])

    def _build_left_pane(self) -> QWidget:
        wrap = QFrame()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(_ROW_VSPACING)

        # Filter row — "Filter:" + LaviComboBox.
        self._filter = setComboBox(["ALL", *_ENGINE_OPTIONS], height=24, i18n=False)
        _apply_small_font(self._filter)
        self._filter.currentTextChanged.connect(self._on_filter_changed)
        v.addLayout(self._labeled_row(
            label_id="editor.filter", label_default="Filter:",
            widget=self._filter,
        ))

        # List.
        self._list = QListWidget()
        _apply_small_font(self._list)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        v.addWidget(self._list, 1)

        # NEW / REMOVE row — both border + spec.
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(_LABEL_INPUT_SPACING)
        self._btn_new = setButton(
            "editor.btn.new", _BTN_W, _BTN_H,
            kind="border", spec="notice", default="NEW",
        )
        self._btn_remove = setButton(
            "editor.btn.remove", _BTN_W, _BTN_H,
            kind="border", spec="danger", default="REMOVE",
        )
        self._btn_new.clicked.connect(self._on_new_item)
        self._btn_remove.clicked.connect(self._on_remove_item)
        btn_row.addWidget(self._btn_new)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return wrap

    def _build_right_pane(self) -> QWidget:
        wrap = QFrame()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(_ROW_VSPACING)

        # Header — Title: [input]   (gap)   Desc: [input]
        self._title_edit = setLineEdit()
        self._desc_edit = setLineEdit()
        _apply_small_font(self._title_edit)
        _apply_small_font(self._desc_edit)
        self._title_edit.editingFinished.connect(self._on_title_changed)
        self._desc_edit.editingFinished.connect(self._on_desc_changed)
        title_group = self._labeled_row(
            label_id="editor.title", label_default="Title:",
            widget=self._title_edit, stretch=1,
        )
        desc_group = self._labeled_row(
            label_id="editor.desc", label_default="Desc:",
            widget=self._desc_edit, stretch=2,
        )
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(_FIELD_GAP)
        hdr_row.addLayout(title_group, 1)
        hdr_row.addLayout(desc_group, 2)
        v.addLayout(hdr_row)

        # Tags — Engine: [combo]   (gap)   Family: [label].
        # Engine combo is suppressed for ``training_motion`` because the
        # canvas's ``backend`` is the single source of truth for that kind
        # (param_rows.TrainingItemsRow:2550 plumbs it in as canvas_backend).
        # Showing the dropdown would invite the user to switch a value that
        # has no per-item effect.
        # Family is a *read-only derived label* — ``TaskModuleItem.applicable_families``
        # is a set, not a single choice, so a single-select combo would
        # always misrepresent multi-family scripts. The label renders the
        # full set (joined / collapsed to a preset name when one matches).
        self._family_label = setText(
            "", default="",
            kind="content",
            color=Config.get_color("main_c2", "#999999"),
            size=int(Config.get_font_size("size_small", 11)),
        )
        fam_group = self._labeled_row(
            label_id="editor.family", label_default="Family:",
            widget=self._family_label, stretch=1,
        )
        tag_row = QHBoxLayout()
        tag_row.setContentsMargins(0, 0, 0, 0)
        tag_row.setSpacing(_FIELD_GAP)
        if self._kind == "training_motion":
            self._engine_combo = None
        else:
            self._engine_combo = setComboBox(_ENGINE_OPTIONS, height=24, i18n=False)
            _apply_small_font(self._engine_combo)
            self._engine_combo.setCurrentText(self._canvas_backend)
            self._engine_combo.currentTextChanged.connect(self._on_engine_changed)
            eng_group = self._labeled_row(
                label_id="editor.engine", label_default="Engine:",
                widget=self._engine_combo,
            )
            tag_row.addLayout(eng_group)
        tag_row.addLayout(fam_group)

        # Variant dropdown — only meaningful for reward / termination /
        # observation / discriminator kinds (training_motion has no
        # script variants).  Populated lazily per selected key in
        # ``_refresh_variant_combo``; family-hard-filtered against the
        # canvas-bound robot via ``resolver.family_filter`` when
        # ``self._robot_sku`` is set.
        if self._kind != "training_motion":
            self._variant_combo = setComboBox(
                [("editor.variant.preset", "Preset")], height=24, i18n=False
            )
            _apply_small_font(self._variant_combo)
            self._variant_combo.currentIndexChanged.connect(
                self._on_variant_changed
            )
            var_group = self._labeled_row(
                label_id="editor.variant", label_default="Variant:",
                widget=self._variant_combo,
            )
            tag_row.addLayout(var_group)
        tag_row.addStretch(1)
        v.addLayout(tag_row)

        # Banner row that surfaces when ``self._robot_sku`` is unset —
        # tells the user the family hard-filter is off (so they
        # understand why incompatible variants are still selectable).
        if self._robot_sku is None and self._kind != "training_motion":
            self._variant_banner = setText(
                "editor.variant.banner",
                default="No robot bound — family filter disabled.",
                kind="content",
                color=Config.get_color("sub_t2"),
                size=int(Config.get_font_size("size_small")),
            )
            v.addWidget(self._variant_banner)

        # Central content panel — kind-specific.
        self._content_host = QFrame()
        self._content_host_layout = QVBoxLayout(self._content_host)
        self._content_host_layout.setContentsMargins(0, 0, 0, 0)
        self._content_host_layout.setSpacing(_ROW_VSPACING)
        v.addWidget(self._content_host, 1)

        # Status + OK / Cancel buttons.
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(_LABEL_INPUT_SPACING)
        self._status = setText(
            "", default="", kind="content",
            color=Config.get_color("main_c2", "#999999"),
            size=int(Config.get_font_size("size_small", 11)),
        )
        bottom.addWidget(self._status, 1)
        # OK uses the highlight slot (spec="notice" — LaviButton's spec
        # vocabulary maps notice→highlight, save→safe_zone, danger→danger_zone).
        self._btn_ok = setButton(
            "editor.btn.ok", _BTN_W, _BTN_H,
            kind="border", spec="notice", default="OK",
        )
        self._btn_cancel = setButton(
            "editor.btn.cancel", _BTN_W, _BTN_H,
            kind="border", spec="danger", default="Cancel",
        )
        self._btn_ok.clicked.connect(self._on_accept)
        self._btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self._btn_ok)
        bottom.addWidget(self._btn_cancel)
        v.addLayout(bottom)

        # Build kind-specific content panel inside _content_host.
        self._build_content_panel()
        return wrap

    # ---- helper: "Label: [input]" with consistent spacing ----

    @staticmethod
    def _labeled_row(
        *,
        label_id: str,
        label_default: str,
        widget: QWidget,
        stretch: int = 0,
    ) -> QHBoxLayout:
        """Compose a horizontal layout: ``setText(...) [spacing] widget``.

        Spacing between label and input is fixed to ``_LABEL_INPUT_SPACING``
        per the panel's visual contract. Label uses ``size_small`` to keep
        every visible string in the editor at one type size.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_LABEL_INPUT_SPACING)
        lab = setText(
            label_id, default=label_default, kind="content",
            size=int(Config.get_font_size("size_small", 11)),
        )
        row.addWidget(lab)
        if stretch > 0:
            row.addWidget(widget, stretch)
        else:
            row.addWidget(widget)
        return row

    # ---- list pane ----

    def _populate_list(self) -> None:
        """Repopulate the left list with the **full** registry + ad-hoc items.

        For source-code kinds (reward / termination / observation / discriminator)
        the list shows every registered function for ``(kind, canvas_backend)``
        plus any user-added keys not in the registry. Items that the node has
        actually added (``self._items``) render in ``safe_zone`` bold; items
        with unsaved source edits override that with ``highlight`` bold (dirty
        takes priority). The displayed label is the ``TaskModuleItem.title``
        (with items dict override) so the same human-readable name shows up
        on the Canvas row, in the add-item picker, and here.

        For ``training_motion`` (no scripts registry) the list falls back to
        ``self._items`` keys — same legacy behaviour, just routed through the
        new visual helper.
        """
        self._list.blockSignals(True)
        self._list.clear()
        keep_engine = self._filter.currentText().strip()

        # Build the ordered key set: registry order first, then ad-hoc keys
        # that aren't in the registry. ``training_motion`` has empty
        # ``_full_registry``, so this degenerates to items-only.
        # 然后按显示 label (= title) A-Z 排——用户在大量奖励/观测条目里翻找时,
        # 字母序比插入序好用得多。casefold 做 locale-agnostic 大小写折叠,
        # 同 label 时 fallback 到 key 保稳定。
        seen: set = set()
        ordered_keys: List[str] = []
        for k in self._full_registry.keys():
            if k not in seen:
                ordered_keys.append(k)
                seen.add(k)
        for k in self._items.keys():
            if k not in seen:
                ordered_keys.append(k)
                seen.add(k)
        ordered_keys.sort(key=lambda k: (self._display_label(k).casefold(), k))

        for k in ordered_keys:
            if keep_engine and keep_engine != "ALL":
                # Engine resolution for filter: items override > registry
                # backends. Items without an explicit ``engine`` fall back
                # to whatever backend tags the registry entry advertises.
                eng = ""
                v_item = self._items.get(k)
                if isinstance(v_item, dict):
                    eng = str(v_item.get("engine", "") or "").strip()
                if not eng:
                    reg_item = self._full_registry.get(k)
                    if reg_item is not None:
                        backends = sorted(
                            str(b) for b in (getattr(reg_item, "backends", set()) or set())
                        )
                        eng = backends[0] if backends else ""
                if eng and eng != keep_engine:
                    continue
            label = self._display_label(k)
            it = QListWidgetItem(label)
            # The visible label is the title; the authoritative key lives
            # on UserRole so selection / dirty tracking always work off the
            # registry id regardless of how the user renames the title.
            it.setData(Qt.ItemDataRole.UserRole, k)
            self._apply_item_visual(
                it,
                added=(k in self._items),
                dirty=(k in self._dirty_keys),
                conflicting=(k in self._conflicting_keys),
            )
            self._list.addItem(it)
        self._list.blockSignals(False)

    def _display_label(self, key: str) -> str:
        """Resolve the human label for ``key``.

        Resolution order — same rule the Canvas row and the add-item picker
        use: ``items[key].title`` (user override on the node) → registry
        ``TaskModuleItem.title`` → fall back to ``key`` itself.
        """
        meta = self._items.get(key)
        if isinstance(meta, dict):
            t = str(meta.get("title", "") or "").strip()
            if t:
                return t
        ov = self._unadded_overrides.get(key)
        if isinstance(ov, dict):
            t = str(ov.get("title", "") or "").strip()
            if t:
                return t
        item = self._full_registry.get(key)
        if item is not None:
            t = str(getattr(item, "title", "") or "").strip()
            if t:
                return t
        return key

    @staticmethod
    def _apply_item_visual(
        item: QListWidgetItem, *, added: bool, dirty: bool,
        conflicting: bool = False,
    ) -> None:
        """Paint ``item`` according to its added / dirty / conflicting state.

        Priority: conflicting (``danger_zone`` red bold — multi-rewards
        collision; the user must resolve before the canvas can compile) >
        dirty (``highlight`` yellow bold — unsaved edits) > added
        (``safe_zone`` green bold — function is on the node) > default
        (``main_t1`` regular — function exists in the registry but isn't on
        the node).
        """
        if conflicting:
            color = QColor(Config.get_color("danger_zone"))
            bold = True
        elif dirty:
            color = QColor(Config.get_color("highlight", "#F6D393"))
            bold = True
        elif added:
            color = QColor(Config.get_color("safe_zone", "#8FD99B"))
            bold = True
        else:
            color = QColor(Config.get_color("main_t1", "#D6D3C7"))
            bold = False
        item.setForeground(QBrush(color))
        f = item.font()
        f.setBold(bold)
        item.setFont(f)

    def _select_key(self, key: str) -> None:
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            it_key = str(it.data(Qt.ItemDataRole.UserRole) or it.text())
            if it_key == key:
                self._list.setCurrentItem(it)
                break
        self._current_key = key
        self._refresh_content_panel()

    # ---- content panel (kind-specific) ----

    def _build_content_panel(self) -> None:
        """Construct the central edit area based on ``self._kind``."""
        # Clear any existing children.
        while self._content_host_layout.count():
            item = self._content_host_layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if self._kind == "training_motion":
            self._build_training_motion_panel()
        else:
            # reward / termination / observation / discriminator → SDK code editor.
            self._build_source_code_panel()

    def _build_source_code_panel(self) -> None:
        # Section label + Save button on the same row.
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(_LABEL_INPUT_SPACING)
        lab = setText(
            "editor.source_code",
            default="Source Code",
            kind="content",
            color=Config.get_color("highlight", "#F6D393"),
            size=int(Config.get_font_size("size_small", 11)),
        )
        hdr.addWidget(lab)
        hdr.addStretch(1)
        # Save button — disabled by default; enables when the current
        # script's buffer differs from its saved ``source``.
        self._btn_save = setButton(
            "editor.btn.save", _BTN_W, _BTN_H,
            kind="border", spec="save", default="Save",
        )
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_current_script)
        hdr.addWidget(self._btn_save)
        self._content_host_layout.addLayout(hdr)
        # Full SDK CodeEditorWidget — line numbers, Python highlighter,
        # smart indentation, Ctrl+Wheel zoom, undo/redo, dirty tracking.
        self._code_edit = CodeEditorWidget(
            parent=self._content_host,
            mode="python",
        )
        _apply_small_font(self._code_edit)
        # Dirty tracking → SDK editor's SHA-1 hash compare (vs the baseline
        # we set via set_text). textChanged is used only to mirror the
        # current buffer into the per-key cache for cross-switch resume.
        self._code_edit.dirtyChanged.connect(self._on_editor_dirty_changed)
        self._code_edit.textChanged.connect(self._on_buffer_changed)
        self._content_host_layout.addWidget(self._code_edit, 1)

    def _build_training_motion_panel(self) -> None:
        # Clip dropdown — populated on every selection refresh from
        # ``scripts.training_motion.library.list_entries`` so newly-dropped
        # motion files under ``<USER_CONFIG_DIR>/scripts/training_motion/motions/``
        # surface without restarting the dialog.
        self._clip_combo = setComboBox([], height=24, i18n=False)
        _apply_small_font(self._clip_combo)
        self._clip_combo.currentIndexChanged.connect(self._on_clip_changed)
        self._content_host_layout.addLayout(self._labeled_row(
            label_id="editor.clip", label_default="Clip:",
            widget=self._clip_combo, stretch=1,
        ))

        # §1C — Speed-channel context hint. Repopulated in
        # ``_refresh_training_motion_fields`` whenever a new item is
        # selected. Shows: ``Channel: lin_vel_x (m/s) · template:
        # [0.0, 1.0] · zero: [...]`` so the user can tell at a glance
        # which channel the Speed lo/hi range controls, what units it
        # is in, and the registry template baseline. Uses empty i18n
        # key (per setText docstring) so the dynamic text isn't
        # overwritten on language-switch retranslation.
        self._channel_hint = setText(
            "", default="", kind="content",
            color=Config.get_color("sub_t2", "#888888"),
            size=int(Config.get_font_size("size_small", 11)),
        )
        self._content_host_layout.addWidget(self._channel_hint)

        # Speed range — plain numeric text fields with the unit moved out
        # of the input. QDoubleValidator clamps to the same [-100, 100]
        # range / 3-decimal resolution the previous spinboxes used.
        self._speed_lo = setLineEdit()
        self._speed_hi = setLineEdit()
        _apply_small_font(self._speed_lo)
        _apply_small_font(self._speed_hi)
        _v_lo = QDoubleValidator(-100.0, 100.0, 3, self._speed_lo)
        _v_hi = QDoubleValidator(-100.0, 100.0, 3, self._speed_hi)
        _v_lo.setNotation(QDoubleValidator.Notation.StandardNotation)
        _v_hi.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._speed_lo.setValidator(_v_lo)
        self._speed_hi.setValidator(_v_hi)
        self._speed_lo.editingFinished.connect(self._on_speed_changed)
        self._speed_hi.editingFinished.connect(self._on_speed_changed)
        self._speed_lo_unit = setText(
            "", default="", kind="content",
            color=Config.get_color("sub_t2", "#888888"),
            size=int(Config.get_font_size("size_small", 11)),
        )
        self._speed_hi_unit = setText(
            "", default="", kind="content",
            color=Config.get_color("sub_t2", "#888888"),
            size=int(Config.get_font_size("size_small", 11)),
        )

        def _spd_group(
            label_id: str, label_default: str, edit: QWidget, unit: QWidget,
        ) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(_LABEL_INPUT_SPACING)
            lab = setText(
                label_id, default=label_default, kind="content",
                size=int(Config.get_font_size("size_small", 11)),
            )
            row.addWidget(lab)
            row.addWidget(edit)
            row.addWidget(unit)
            return row

        lo_group = _spd_group(
            "editor.speed_lo", "Speed lo:", self._speed_lo, self._speed_lo_unit,
        )
        hi_group = _spd_group(
            "editor.speed_hi", "Speed hi:", self._speed_hi, self._speed_hi_unit,
        )
        spd_row = QHBoxLayout()
        spd_row.setContentsMargins(0, 0, 0, 0)
        spd_row.setSpacing(_FIELD_GAP)
        spd_row.addLayout(lo_group)
        spd_row.addLayout(hi_group)
        spd_row.addStretch(1)
        self._content_host_layout.addLayout(spd_row)

        # Advanced (JSON) — section label + JSON-mode SDK editor.
        adv_lab = setText(
            "editor.advanced", default="Advanced (JSON)", kind="content",
            color=Config.get_color("highlight", "#F6D393"),
            size=int(Config.get_font_size("size_small", 11)),
        )
        self._content_host_layout.addWidget(adv_lab)
        self._adv_edit = CodeEditorWidget(
            parent=self._content_host,
            mode="json",
        )
        _apply_small_font(self._adv_edit)
        self._adv_edit.textChanged.connect(self._on_advanced_changed)
        self._content_host_layout.addWidget(self._adv_edit, 1)

    # ---- selection / refresh ----

    def _on_selection_changed(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._current_key = None
        else:
            self._current_key = str(
                items[0].data(Qt.ItemDataRole.UserRole) or items[0].text()
            )
        self._refresh_content_panel()
        self._refresh_remove_button_state()

    def _refresh_remove_button_state(self) -> None:
        """REMOVE only removes from the node — disabled for un-added keys.

        The Editor's left list shows the full registry; REMOVE has no
        meaning for entries the node never added (and we must never let
        REMOVE mutate the registry). For ``training_motion`` (no registry)
        every visible key is in items, so this stays a pure pass-through.
        """
        btn = getattr(self, "_btn_remove", None)
        if btn is None:
            return
        cur = self._current_key
        btn.setEnabled(cur is not None and cur in self._items)

    def _refresh_content_panel(self) -> None:
        key = self._current_key
        meta = self._items.get(key, {}) if key else {}
        if not isinstance(meta, dict):
            # legacy ``{key: weight}`` shape — promote to dict for editing.
            meta = {"weight": meta}
        # Un-added draft (title/desc/engine/family edits made while
        # browsing an un-added function). Discarded on OK / Cancel; only
        # the source goes anywhere (via Save → registry).
        ov = self._unadded_overrides.get(key, {}) if key else {}

        # Pull registry metadata (title / desc / engine / family) — for
        # built-in keys (in REWARD_REGISTRY etc.) this is authoritative;
        # the items dict typically only carries the weight scalar so
        # without this lookup the form fields would be blank or default.
        reg_meta = self._registry_metadata(key) if key else {}

        def _resolve(field: str, default: str = "") -> str:
            """items dict > un-added override > registry value > default."""
            v = str(meta.get(field, "") or "").strip()
            if v:
                return v
            v = str(ov.get(field, "") or "").strip()
            if v:
                return v
            v = str(reg_meta.get(field, "") or "").strip()
            if v:
                return v
            return default

        # Block signals so programmatic setText doesn't fire change handlers.
        self._title_edit.blockSignals(True)
        self._desc_edit.blockSignals(True)
        if self._engine_combo is not None:
            self._engine_combo.blockSignals(True)

        self._title_edit.setText(_resolve("title", key or ""))
        self._desc_edit.setText(_resolve("desc"))
        # Engine: items override > registry tag > canvas-bound default.
        # Skipped for training_motion (combo not built — backend is fixed
        # at the canvas level).
        if self._engine_combo is not None:
            eng = _resolve("engine")
            if eng in _ENGINE_OPTIONS:
                self._engine_combo.setCurrentText(eng)
            else:
                self._engine_combo.setCurrentText(self._canvas_backend)
        # Family label — purely derived from registry (set of applicable
        # families). No per-item override: ``d["family"]`` has never been
        # consumed downstream, so the legacy single-pick combo was dead UI.
        self._family_label.setText(_resolve("family"))

        self._title_edit.blockSignals(False)
        self._desc_edit.blockSignals(False)
        if self._engine_combo is not None:
            self._engine_combo.blockSignals(False)

        if self._kind == "training_motion":
            self._refresh_training_motion_fields(meta)
        else:
            self._refresh_source_code_field(key, meta)
            # Variant dropdown depends on (kind, key) — refresh after the
            # source-code field so the user sees the new options without
            # a perceptible flicker.
            self._refresh_variant_combo()
        # REMOVE is "remove from node" — only meaningful for added keys.
        self._refresh_remove_button_state()

    def _refresh_source_code_field(self, key: Optional[str], meta: Dict[str, Any]) -> None:
        """Hydrate the editor and align SDK baseline with the saved source.

        Two-phase load — relies on a property of the SDK editor:
            * ``set_text(t)`` → updates ``_baseline_hash = sha1(t)`` (clean)
            * ``setPlainText(t)`` → updates the document but leaves
              ``_baseline_hash`` untouched

        So we:
            1. ``set_text(saved)`` — pin baseline to the saved bytes
            2. if cached buffer differs, ``setPlainText(buffer)`` — display
               the buffer; the SDK's hash check now reports
               ``dirty == (buffer != saved)`` directly.

        Result:
            * First visit, no edits → buffer == saved → clean.
            * Cross-switch resume of a previously-edited item → clean if
              buffer happens to match saved, dirty otherwise.
            * Type a space + backspace → editor's hash check returns to
              the baseline → clean. (This was the old custom-string-compare
              bug — Qt ``QTextDocument`` may normalise line endings, so the
              cached pre-set_text string and the post-set_text editor text
              differed; SHA-1 of post-load text vs post-load text always
              matches.)
        """
        if key is None:
            self._suppress_editor_dirty = True
            self._code_edit.set_text("")
            self._suppress_editor_dirty = False
            self._refresh_save_button_state()
            return

        # Saved source — resolution priority:
        #   1. Registry's TaskModuleItem.il_inline (built-in presets in
        #      src/scripts/<kind>/registry.py)
        #   2. items[key]['source'] (ad-hoc per-canvas override)
        #   3. Kind stub (only when neither registry nor items provides
        #      a source — happens for terminations whose entries are
        #      threshold-only, or for fresh user-added keys via NEW)
        # In case (3), promote the stub to items[key]['source'] so that
        # first-visit dirty comparison against saved is clean (otherwise
        # buffer == stub but saved == "" → flagged dirty on click).
        saved = self._saved_source_text(key)
        if not saved:
            stub = _KIND_STUB.get(self._kind, _REWARD_STUB)
            saved = stub.format(key=key, polarity="reward")
            d = self._items.get(key)
            if isinstance(d, dict):
                d["source"] = saved

        buffer = self._content_cache.get(key, saved)

        # Phase 1 — pin SDK baseline to saved bytes.
        self._suppress_editor_dirty = True
        self._code_edit.set_text(saved)
        # Phase 2 — display the cached buffer if it differs from saved.
        # setPlainText does NOT call _refresh_baseline; baseline stays at saved.
        if buffer != self._code_edit.text():
            self._code_edit.setPlainText(buffer)
        self._suppress_editor_dirty = False

        # Cache the editor-normalised text so cross-switch resume always
        # uses the same form Qt's QTextDocument settled on.
        self._content_cache[key] = self._code_edit.text()

        # Reconcile _dirty_keys with the SDK's authoritative is_modified()
        # (textChanged → _recompute_dirty fired during the load even though
        # our handler was suppressed, so is_modified() is up-to-date).
        if self._code_edit.is_modified():
            self._dirty_keys.add(key)
        else:
            self._dirty_keys.discard(key)
        self._refresh_list_item_dirty_visual(key)
        self._refresh_save_button_state()

    def _rebuild_clip_combo(self, current_ref: Optional[str]) -> None:
        """Re-scan the motion library and repopulate the Clip dropdown.

        Mirrors DEMO's ``_rebuild_ref_combo``: ``(None)`` at the top, then
        every ``MotionEntry`` from ``scripts.training_motion.library``
        (SB3 library + community archives). ``userData`` holds the
        absolute path string the downstream training pipeline consumes.
        If ``current_ref`` doesn't match any scanned entry it is preserved
        as a ``(custom)`` row so legacy ``pack:`` refs aren't silently
        dropped on load.
        """
        self._clip_combo.blockSignals(True)
        try:
            self._clip_combo.clear()
            self._clip_combo.addItem("(None)", userData=None)

            entries = []
            try:
                from scripts.training_motion.library import list_entries
                entries = list_entries(include_archives=True)
            except Exception as e:
                log_warning(
                    f"[RegistryModuleEditorPanel] motion library scan failed: {e!r}"
                )

            seen_refs: set = {None}
            for e in entries:
                try:
                    ref = str(e.path)
                    if ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    tag = (getattr(e, "task_tag", "") or "—").strip() or "—"
                    display = f"{e.robot_model} / {e.name}  [{tag}]"
                    self._clip_combo.addItem(display, userData=ref)
                except Exception:
                    continue

            target_idx = 0
            if current_ref:
                for i in range(self._clip_combo.count()):
                    if self._clip_combo.itemData(i) == current_ref:
                        target_idx = i
                        break
                else:
                    # Legacy ref (e.g. ``pack:unitport_builtin:...``) that
                    # isn't in the scanned library — preserve it so OK
                    # round-trips don't lose the user's value.
                    self._clip_combo.addItem(
                        f"(custom) {current_ref}", userData=current_ref,
                    )
                    target_idx = self._clip_combo.count() - 1
            self._clip_combo.setCurrentIndex(target_idx)
        finally:
            self._clip_combo.blockSignals(False)

    def _refresh_training_motion_fields(self, meta: Dict[str, Any]) -> None:
        self._speed_lo.blockSignals(True)
        self._speed_hi.blockSignals(True)
        self._adv_edit.blockSignals(True)

        clip = meta.get("clip")
        # Clip combo is rebuilt (re-scans library) on every refresh so
        # newly-dropped files in the motion library surface without a
        # dialog restart. The combo manages its own blockSignals.
        self._rebuild_clip_combo(None if clip is None else str(clip))

        speed = meta.get("speed")
        if isinstance(speed, (list, tuple)) and len(speed) >= 2:
            self._speed_lo.setText(f"{float(speed[0] or 0.0):g}")
            self._speed_hi.setText(f"{float(speed[1] or 0.0):g}")
        else:
            self._speed_lo.setText("0")
            self._speed_hi.setText("0")
        adv = meta.get("advanced") or {}
        try:
            self._adv_edit.set_text(json.dumps(adv, indent=2, ensure_ascii=False))
        except Exception:
            self._adv_edit.set_text("{}")

        # §1C — Resolve per-item channel/unit hint from registers.commands.
        # Falls back to a generic hint when the item is unknown (e.g. user
        # has added a custom id that isn't in the registry yet).
        hint_text = ""
        suffix = ""
        try:
            from registers import commands as _commands_reg
            reg_item = _commands_reg.get_item(self._current_key) if self._current_key else None
        except Exception:
            reg_item = None
        if reg_item is not None:
            channel = reg_item.speed_channel
            unit = "rad/s" if str(channel).startswith("ang_") else "m/s"
            suffix = " " + unit
            tpl_rng = reg_item.command_template.get(channel)
            tpl_str = (
                f"[{float(tpl_rng[0]):.2f}, {float(tpl_rng[1]):.2f}]"
                if isinstance(tpl_rng, (list, tuple)) and len(tpl_rng) >= 2
                else "[-, -]"
            )
            zero_str = (
                "[" + ", ".join(reg_item.zero_channels) + "]"
                if reg_item.zero_channels else "[]"
            )
            crosses_zero = (
                isinstance(tpl_rng, (list, tuple)) and len(tpl_rng) >= 2
                and float(tpl_rng[0]) < 0 < float(tpl_rng[1])
            )
            bidir_tag = " · bidirectional" if crosses_zero else ""
            hint_text = (
                f"Channel: {channel} ({unit}) · template: {tpl_str} · "
                f"zero: {zero_str}{bidir_tag}"
            )
        self._channel_hint.setText(hint_text)
        unit = suffix.strip()
        self._speed_lo_unit.setText(unit)
        self._speed_hi_unit.setText(unit)

        self._speed_lo.blockSignals(False)
        self._speed_hi.blockSignals(False)
        self._adv_edit.blockSignals(False)

    # ---- field handlers ----

    def _ensure_dict_entry(self) -> Optional[Dict[str, Any]]:
        """Return the in-memory dict to receive a field edit for the current key.

        - If the current key is already on the node (in ``self._items``): the
          editable dict is ``self._items[key]`` itself — edits flow back to
          the node on OK.
        - If the current key is registry-only (browsing an un-added function):
          edits go to ``self._unadded_overrides[key]`` so the form fields
          stay populated across switches, but OK discards them — the user
          must use the Canvas add-item picker (or NEW) to actually adopt
          a function. This keeps Save's contract clean: Save writes the
          registry, never adds to the node.
        """
        key = self._current_key
        if key is None:
            return None
        if key in self._items:
            cur = self._items.get(key)
            if not isinstance(cur, dict):
                cur = {"weight": cur} if cur is not None else {}
                self._items[key] = cur
            return cur
        ov = self._unadded_overrides.get(key)
        if not isinstance(ov, dict):
            ov = {}
            self._unadded_overrides[key] = ov
        return ov

    def _on_title_changed(self) -> None:
        d = self._ensure_dict_entry()
        if d is None:
            return
        d["title"] = self._title_edit.text().strip()
        # Title can be the displayed label — refresh the visible row text.
        key = self._current_key
        if key is not None:
            self._refresh_list_item_label(key)

    def _on_desc_changed(self) -> None:
        d = self._ensure_dict_entry()
        if d is None:
            return
        d["desc"] = self._desc_edit.text().strip()

    def _on_engine_changed(self, value: str) -> None:
        d = self._ensure_dict_entry()
        if d is None:
            return
        d["engine"] = value

    def _on_variant_changed(self, _index: int) -> None:
        """Write the picked variant back to ``self._items[key]["variant"]``.

        Only mutates items that the node has actually added — the
        un-added drafts are discarded on OK / Cancel per the panel's
        existing contract, so writing them is pointless.
        """
        if self._variant_combo is None:
            return
        key = self._current_key
        if not key or key not in self._items:
            return
        ikey = self._variant_combo.currentKey() or ""
        meta = self._items.get(key)
        if not isinstance(meta, dict):
            meta = {"weight": meta}
            self._items[key] = meta
        if ikey == "editor.variant.preset" or not ikey:
            # Drop the variant tag entirely — preset is the implicit
            # default, no need to add a "variant" key the resolver will
            # normalise to None anyway.
            meta.pop("variant", None)
        elif ikey.startswith("editor.variant.user:"):
            meta["variant"] = ikey.split(":", 1)[1]

    def _refresh_variant_combo(self) -> None:
        """Repopulate the Variant dropdown for the currently-selected key.

        Source: :func:`application.service.scripts.resolver.list_variants`.
        When ``self._robot_sku`` is set, variants whose families do not
        intersect the robot's families are excluded (hard filter, rule
        Stage 3). The current ``self._items[key]["variant"]`` value (if
        any) is reflected as the active selection; legacy items without
        a variant tag select "Preset".
        """
        if self._variant_combo is None or self._kind == "training_motion":
            return
        key = self._current_key
        if not key:
            return
        try:
            from application.service.scripts import resolver
        except Exception:                                         # noqa: BLE001
            return

        # Robot families (empty if no SKU bound).
        robot_fams = frozenset()
        if self._robot_sku:
            # Reuse the same lookup the resolver uses — never duplicate
            # the brand-/SKU-stripping logic, that lives in registers.
            from application.service.scripts.resolver import _robot_families
            robot_fams = _robot_families(self._robot_sku)

        entries = [("editor.variant.preset", "Preset")]
        for v in resolver.list_variants(self._kind, key):
            if not resolver.family_filter(v.families, robot_fams):
                continue
            entries.append((f"editor.variant.user:{v.name}", v.name))

        # Replace the combo. LaviComboBox exposes no public "reset items"
        # API, but the cheapest replacement is to clear + add: model is a
        # plain QStandardItemModel populated in __init__.
        self._variant_combo.blockSignals(True)
        try:
            model = self._variant_combo._src_model              # noqa: SLF001
            model.clear()
            from PyQt6.QtGui import QStandardItem
            self._variant_combo._i18n_items = list(entries)     # noqa: SLF001
            for ikey, default in entries:
                item = QStandardItem(default)
                from PyQt6.QtCore import Qt as _Qt
                item.setData(ikey, _Qt.ItemDataRole.UserRole)
                model.appendRow(item)

            # Determine which entry to select based on current state.
            meta = self._items.get(key)
            cur_variant = ""
            if isinstance(meta, dict):
                cur_variant = str(meta.get("variant") or "")
            target_ikey = (
                f"editor.variant.user:{cur_variant}"
                if cur_variant
                else "editor.variant.preset"
            )
            if not self._variant_combo.setCurrentKey(target_ikey):
                self._variant_combo.setCurrentIndex(0)
        finally:
            self._variant_combo.blockSignals(False)

    def _on_filter_changed(self, _value: str) -> None:
        cur = self._current_key
        self._populate_list()
        if cur is not None:
            self._select_key(cur)

    def _on_buffer_changed(self) -> None:
        """Editor text changed — mirror current text into the per-key cache.

        Pure cache write so cross-switch resume always replays the latest
        keystrokes. Dirty-state bookkeeping is the job of
        :meth:`_on_editor_dirty_changed`, driven by the SDK's hash signal.
        """
        if self._suppress_editor_dirty:
            return
        if self._current_key is None:
            return
        self._content_cache[self._current_key] = self._code_edit.text()

    def _on_editor_dirty_changed(self, dirty: bool) -> None:
        """SDK editor dirty signal — directly drives ``_dirty_keys`` membership.

        The SDK's ``CodeEditorWidget`` maintains a SHA-1 baseline reset only
        by ``set_text`` / ``save_file`` / ``__init__``; every subsequent
        textChanged recomputes ``hash(current) == hash(baseline)`` and
        emits this signal on flip. Because we pin the baseline to the
        saved source on every load (see :meth:`_refresh_source_code_field`),
        the boolean delivered here is exactly ``buffer != saved``.

        Critically: typing a space and deleting it produces identical bytes
        → identical hash → ``dirty=False``. No more spurious dirty marks
        from text-normalisation or whitespace round-trips.
        """
        if self._suppress_editor_dirty:
            return
        if self._current_key is None:
            return
        key = self._current_key
        was_dirty = key in self._dirty_keys
        if dirty:
            self._dirty_keys.add(key)
        else:
            self._dirty_keys.discard(key)
        if was_dirty != dirty:
            self._refresh_list_item_dirty_visual(key)
        self._refresh_save_button_state()

    def _saved_source_text(self, key: str) -> str:
        """Return the canonical saved source for ``key``.

        Resolution order:
            1. ``scripts.<kind>_REGISTRY[key].il_inline`` — built-in preset's
               authoritative function body (from ``src/scripts/<kind>/``).
               This is the truth for every shipped reward / observation /
               discriminator entry.
            2. ``self._items[key]['source']`` — ad-hoc per-canvas override
               (only present for keys NOT in any registry, e.g. user added
               via NEW button without saving back to registry).
            3. ``""`` — caller falls through to the kind stub.
        """
        item = self._lookup_registry_entry(key)
        if item is not None:
            inline = str(getattr(item, "il_inline", "") or "")
            if inline:
                return inline
        meta = self._items.get(key)
        if isinstance(meta, dict):
            return str(meta.get("source", "") or "")
        return ""

    def _lookup_registry_entry(self, key: str):
        """Look up the ``TaskModuleItem`` for ``(self._kind, key, canvas_backend)``.

        Returns ``None`` if scripts package is unreachable, kind doesn't map
        to a registry, or key isn't registered.
        """
        if not _SCRIPTS_AVAILABLE or _scripts_lookup is None:
            return None
        if not self._has_save_workflow:
            return None
        try:
            return _scripts_lookup(
                key, kind=self._kind, backend=self._canvas_backend,
            )
        except Exception:
            return None

    def _registry_metadata(self, key: str) -> Dict[str, Any]:
        """Pull display metadata (title / desc / engine / family) from the registry.

        Built-in keys are owned by ``src/scripts/`` — title and desc on
        ``TaskModuleItem`` are the source of truth. Engine defaults to the
        first member of ``item.backends`` (or canvas backend if the item is
        backend-agnostic). Family renders the *full*
        ``applicable_families`` set, collapsing well-known supersets to
        their preset name (``all`` / ``locomotion`` / ``legged``) so the
        common cases stay short.
        """
        item = self._lookup_registry_entry(key)
        if item is None:
            return {}
        engine = ""
        try:
            backends = list(getattr(item, "backends", set()) or set())
            if self._canvas_backend in backends:
                engine = self._canvas_backend
            elif backends:
                engine = sorted(backends)[0]
        except Exception:
            pass
        family = ""
        try:
            fams = frozenset(getattr(item, "applicable_families", set()) or set())
            family = self._format_family_set(fams)
        except Exception:
            pass
        return {
            "title":  str(getattr(item, "title", "") or ""),
            "desc":   str(getattr(item, "desc", "") or ""),
            "engine": engine,
            "family": family,
        }

    @staticmethod
    def _format_family_set(fams: frozenset) -> str:
        """Render an ``applicable_families`` set for display.

        Collapses well-known supersets (``ALL_FAMILIES`` / ``LOCOMOTION_FAMILIES``
        / ``LEGGED_FAMILIES``) to their semantic preset name so a script
        like ``track_lin_vel_xy_exp`` (which applies to four families)
        shows as ``locomotion`` rather than a four-element list. Other
        sets render as a sorted, comma-joined list of family names.
        """
        if not fams:
            return ""
        if _ALL_FAMILIES and fams == _ALL_FAMILIES:
            return "all"
        if _LOCOMOTION_FAMILIES and fams == _LOCOMOTION_FAMILIES:
            return "locomotion"
        if _LEGGED_FAMILIES and fams == _LEGGED_FAMILIES:
            return "legged"
        return ", ".join(sorted(fams))

    def _refresh_list_item_dirty_visual(self, key: str) -> None:
        """Re-apply added / dirty / conflicting styling to the list row backed by ``key``."""
        added = key in self._items
        dirty = key in self._dirty_keys
        conflicting = key in self._conflicting_keys
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            it_key = str(it.data(Qt.ItemDataRole.UserRole) or it.text())
            if it_key == key:
                self._apply_item_visual(
                    it, added=added, dirty=dirty, conflicting=conflicting,
                )
                break

    def _refresh_list_item_label(self, key: str) -> None:
        """Re-render the list row's visible text for ``key`` (after title edits)."""
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            it_key = str(it.data(Qt.ItemDataRole.UserRole) or it.text())
            if it_key == key:
                it.setText(self._display_label(key))
                break

    def _refresh_save_button_state(self) -> None:
        """Save button enables iff the current item's script is dirty."""
        if not getattr(self, "_btn_save", None):
            return
        cur = self._current_key
        self._btn_save.setEnabled(cur is not None and cur in self._dirty_keys)

    def _save_current_script(self) -> None:
        """Commit the current editor buffer to its authoritative location.

        Two-track persistence:
            * **Registry-backed key** (entry exists in
              ``ALL_REGISTRIES[kind]``): replace the ``TaskModuleItem``'s
              ``il_inline`` field via ``dataclasses.replace`` and put the
              new instance back into the same sub-registry that held the
              original. This is in-process mutation — every consumer of
              the registry (compiler, training launcher, the editor
              itself on next open) immediately sees the new source.
            * **Ad-hoc key** (no registry entry — typically a NEW item
              before its first save): persist to ``items[key]['source']``
              so the per-canvas dict carries the source forward.

        After in-memory commit, ``emit_inline_overrides`` writes a sidecar
        JSON under ``USER_CONFIG_DIR/scripts/<kind>_<backend>.json`` so the
        next training subprocess (which re-imports the registry from disk
        and would otherwise lose the edit) can overlay the user's source.

        UI bookkeeping:
            * mirror to ``_content_cache`` (post-edit text)
            * ``set_text`` on the SDK editor to reset its SHA-1 baseline
              (otherwise a subsequent space + delete would re-flag dirty
              against the stale baseline)
            * clear ``_dirty_keys[key]``, refresh list visual + Save state
        """
        if not self._has_save_workflow:
            return
        key = self._current_key
        if key is None:
            return
        text = self._code_edit.text()

        # 1. Persist to registry (in-process) when key is registry-backed,
        #    else fall back to items dict (ad-hoc).
        wrote_registry = self._write_source_to_registry(key, text)
        if not wrote_registry:
            d = self._ensure_dict_entry()
            if d is not None:
                d["source"] = text

        # 2. Cache buffer aligned to saved state.
        self._content_cache[key] = text

        # 3. Sidecar emit for subprocess overlay.
        self._emit_inline_sidecar()

        # 4. Reset SDK baseline so subsequent space + delete cycles return
        #    to the new saved hash → clean.
        self._suppress_editor_dirty = True
        self._code_edit.set_text(text)
        self._suppress_editor_dirty = False
        self._dirty_keys.discard(key)
        self._refresh_list_item_dirty_visual(key)
        self._refresh_save_button_state()
        self._set_status(f"Saved: {key}")

    def _write_source_to_registry(self, key: str, source: str) -> bool:
        """Write ``source`` into the matching ``TaskModuleItem.il_inline``.

        Returns True when an in-place registry mutation happened, False
        when the key isn't found in any sub-registry of the current kind
        (caller should then fall back to items-dict persistence).

        The write goes into the SAME sub-registry that already holds the
        key (preserves SB3 vs Isaac Lab origin). Backend filter ensures we
        don't accidentally overwrite an SB3 entry from an isaac_lab canvas
        when both registries happen to share the key name.
        """
        if not _SCRIPTS_AVAILABLE or _TaskModuleItem is None:
            return False
        subs = _SCRIPTS_ALL_REGISTRIES.get(self._kind) or []
        for sub in subs:
            existing = sub.get(key)
            if existing is None:
                continue
            try:
                if self._canvas_backend not in getattr(existing, "backends", set()):
                    continue
            except Exception:
                continue
            try:
                sub[key] = dataclasses.replace(existing, il_inline=source)
                return True
            except Exception as e:
                log_warning(f"[RegistryModuleEditorPanel] dataclasses.replace failed for {key!r}: {e!r}")
                return False
        return False

    def _emit_inline_sidecar(self) -> None:
        """Persist all in-memory ``il_inline`` edits for the subprocess.

        Sidecar path: ``USER_CONFIG_DIR/scripts/<kind>_<backend>.json``
        (per-user state, outside the project tree — rule §1.4 / §5).
        Failures log a warning but never block Save.
        """
        if not _SCRIPTS_AVAILABLE or _emit_inline_overrides is None:
            return
        try:
            out_dir = Paths.USER_CONFIG_DIR / "scripts"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{self._kind}_{self._canvas_backend}.json"
            n = _emit_inline_overrides(
                self._kind, out_path, backend=self._canvas_backend,
            )
            self._set_status(f"Saved: {self._current_key} (sidecar: {n} entries)")
        except Exception as e:
            log_warning(f"[RegistryModuleEditorPanel] emit_inline_overrides failed: {e!r}")

    def _on_clip_changed(self, *_args: Any) -> None:
        d = self._ensure_dict_entry()
        if d is None:
            return
        data = self._clip_combo.currentData()
        d["clip"] = None if data is None else str(data)

    def _on_speed_changed(self) -> None:
        d = self._ensure_dict_entry()
        if d is None:
            return

        def _parse(le) -> float:
            try:
                return float((le.text() or "").strip() or 0.0)
            except ValueError:
                return 0.0

        d["speed"] = [_parse(self._speed_lo), _parse(self._speed_hi)]

    def _on_advanced_changed(self) -> None:
        d = self._ensure_dict_entry()
        if d is None:
            return
        text = self._adv_edit.text().strip() or "{}"
        try:
            parsed = json.loads(text)
            d["advanced"] = parsed if isinstance(parsed, dict) else {}
            self._set_status("")
        except Exception as e:
            self._set_status(f"Advanced JSON error: {e}")

    def _set_status(self, text: str) -> None:
        # LaviLabel exposes setText via QLabel parent.
        try:
            self._status.setText(text)
        except Exception:
            pass

    # ---- list mutate ----

    def _on_new_item(self) -> None:
        # Generate a unique key.
        i = 1
        while True:
            key = f"new_item_{i}"
            if key not in self._items:
                break
            i += 1
        if self._kind == "training_motion":
            self._items[key] = {
                "enabled": True,
                "speed": [0.0, 0.0],
                "clip": None,
                "advanced": {},
            }
        else:
            self._items[key] = {"weight": 1.0}
        self._populate_list()
        self._select_key(key)

    def _on_remove_item(self) -> None:
        # REMOVE means "remove from the node" — never touches the registry.
        # Disabled in the UI when the current key is registry-only.
        if self._current_key is None or self._current_key not in self._items:
            return
        key = self._current_key
        del self._items[key]
        self._content_cache.pop(key, None)
        self._dirty_keys.discard(key)
        self._current_key = None
        self._populate_list()
        # Prefer the next added key for continuity; else fall back to the
        # first row still showing (registry-only or empty).
        next_key: Optional[str] = None
        if self._items:
            next_key = next(iter(self._items.keys()))
        elif self._full_registry:
            next_key = next(iter(self._full_registry.keys()))
        if next_key is not None:
            self._select_key(next_key)
        else:
            self._refresh_content_panel()
        self._refresh_save_button_state()

    # ---- accept / reject ----

    def _on_accept(self) -> None:
        """OK clicked — confirm if any scripts are still dirty.

        ``self._dirty_keys`` lists keys whose buffer differs from saved
        ``source``. When non-empty, surface an SDK-themed confirmation
        listing each unsaved script so the user can Save first or
        deliberately discard.
        """
        if self._dirty_keys and not self._confirm_discard_unsaved("accept"):
            return
        try:
            self.applied.emit(dict(self._items))
        except Exception as e:
            log_warning(f"[RegistryModuleEditorPanel] applied emit failed: {e!r}")
        self.accept()

    def reject(self) -> None:  # type: ignore[override]
        """Cancel / X / Esc — same dirty guard as OK."""
        if self._dirty_keys and not self._confirm_discard_unsaved("reject"):
            return
        super().reject()

    def _confirm_discard_unsaved(self, action: str) -> bool:
        """Modal "you have unsaved scripts" confirmation. True = proceed."""
        keys = sorted(self._dirty_keys)
        dlg = _UnsavedScriptsDialog(self, keys, action=action)
        return dlg.exec() == QDialog.DialogCode.Accepted


# =============================================================================
# Unsaved-scripts confirmation dialog
# =============================================================================


class _UnsavedScriptsDialog(QDialog):
    """SDK-themed "discard unsaved scripts" confirmation modal.

    Shown when the user clicks OK / Cancel with non-empty
    ``_dirty_keys``. Lists every unsaved script by name. Buttons:

        Discard & Continue  → accept (parent proceeds with original action)
        Back to editing     → reject (parent stays open)

    Visual contract follows the parent panel: ``setText`` for labels +
    ``setButton(kind="border", spec=...)`` for actions.
    """

    def __init__(
        self,
        parent: QWidget,
        keys: List[str],
        *,
        action: str,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Unsaved Scripts")
        self._apply_chrome(parent)

        action_word = "closing" if action == "reject" else "applying"
        small = int(Config.get_font_size("size_small", 11))
        v = QVBoxLayout(self)
        v.setContentsMargins(_DLG_PAD, _DLG_PAD, _DLG_PAD, _DLG_PAD)
        v.setSpacing(_ROW_VSPACING * 2)

        title = setText(
            "editor.unsaved.title",
            default=f"{len(keys)} script(s) have unsaved edits.",
            kind="content",
            color=Config.get_color("highlight", "#F6D393"),
            size=small,
        )
        v.addWidget(title)

        sub = setText(
            "editor.unsaved.sub",
            default=f"Discarding now will lose these edits before {action_word}:",
            kind="content",
            size=small,
        )
        v.addWidget(sub)

        # List of unsaved keys — read-only.
        lst = QListWidget()
        _apply_small_font(lst)
        lst.setStyleSheet(
            f"QListWidget {{ background: {Config.get_color('bg_2', '#2A2A2A')}; "
            f"color: {Config.get_color('main_t1', '#D6D3C7')}; "
            f"border: 1px solid {Config.get_color('border_2', '#444444')}; "
            f"border-radius: 4px; padding: 4px; outline: 0; }}"
            f"QListWidget::item {{ padding: 3px 6px; }}"
        )
        for k in keys:
            it = QListWidgetItem(k)
            it.setForeground(QBrush(QColor(Config.get_color("highlight", "#F6D393"))))
            f = it.font()
            f.setBold(True)
            it.setFont(f)
            lst.addItem(it)
        lst.setMinimumHeight(120)
        lst.setMaximumHeight(220)
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        v.addWidget(lst, 1)

        # Action row — short labels per UI contract.
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_LABEL_INPUT_SPACING)
        row.addStretch(1)
        btn_back = setButton(
            "editor.unsaved.return", _BTN_W, _BTN_H,
            kind="border", spec="notice", default="Return",
        )
        btn_discard = setButton(
            "editor.unsaved.discard", _BTN_W, _BTN_H,
            kind="border", spec="danger", default="Discard",
        )
        btn_back.clicked.connect(self.reject)
        btn_discard.clicked.connect(self.accept)
        row.addWidget(btn_back)
        row.addWidget(btn_discard)
        v.addLayout(row)

        self.resize(420, 320)

    def _apply_chrome(self, parent: QWidget) -> None:
        bg = Config.get_color("bg_1", "#1E1E1E")
        fg = Config.get_color("main_t1", "#D6D3C7")
        self.setStyleSheet(
            f"QDialog {{ background: {bg}; color: {fg}; }}"
            f"QFrame {{ background: transparent; }}"
        )


__all__ = [
    "RegistryModuleEditorPanel",
    "open_reward_function_editor",
    "open_training_motion_editor",
]
