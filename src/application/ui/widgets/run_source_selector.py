"""RunSourceSelector — multi-select dropdown for the chart's data source.

Replaces DEMO's flat checkbox column (``training_workspace_window.py`` line
1527+) with a compact ``QToolButton`` whose popup hosts a
``QListWidget`` of checkable items. Each item is one training run.

Sources unioned at refresh time:

  * :class:`MetricsCache` — runs that have been seen this session
    (including the on-going one). Source of live metrics.
  * :class:`TrainingStore` — historical runs persisted under
    ``<project>/training/runs.json`` (Stage 11). Selected historical
    runs render with no curves until ``progress.jsonl`` replay lands
    in a later sub-stage.

Emits :sig:`selected_runs_changed(list[str])` whenever the user-visible
selection changes. The button label compresses to ``"<label>"`` for one
selection, ``"<N> runs"`` for more, ``"No runs"`` when the union is empty.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Set

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMenu,
    QToolButton,
    QWidget,
    QWidgetAction,
)

from unitport_sdk import Config, I18n, tr

from application.service.metrics_cache import RunSummary, get_metrics_cache
from application.service.signals import get_app_signals


class RunSourceSelector(QToolButton):
    """Checklist popup for selecting which training runs the chart shows.

    Args:
        parent: optional parent widget.
        project_info_provider: callable returning the currently bound
            ``ProjectInfo`` (or ``None``). Used to enrich the run union
            with persisted runs from ``TrainingStore``. May itself
            return ``None`` if no project is bound — selector still
            functions on the in-memory cache alone.
    """

    selected_runs_changed = pyqtSignal(list)            # list[str] of run_ids
    _MIN_POPUP_WIDTH = 260
    # 选择器按钮本体最小宽度 —— 弹出列表仍保留 _MIN_POPUP_WIDTH 以便长 run 标签可读。
    _MIN_BUTTON_WIDTH = 156           # ≈ _MIN_POPUP_WIDTH * 0.6
    _POPUP_MAX_HEIGHT = 280

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        project_info_provider: Optional[Callable[[], object]] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("runSourceSelector")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setMinimumWidth(self._MIN_BUTTON_WIDTH)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText(tr("chart.runs.empty", "No runs"))

        self._project_info_provider = project_info_provider
        self._selected: Set[str] = set()
        self._all_run_ids: List[str] = []                # display order

        # Build the popup menu shell — the QListWidget lives inside a
        # QWidgetAction so we get standard click-outside-to-close.
        self._menu = QMenu(self)
        self._menu.setObjectName("runSourceMenu")
        self._list = QListWidget(self._menu)
        self._list.setObjectName("runSourceList")
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._list.setMaximumHeight(self._POPUP_MAX_HEIGHT)
        self._list.setMinimumWidth(self._MIN_POPUP_WIDTH)
        self._list.setCursor(Qt.CursorShape.PointingHandCursor)
        self._list.itemChanged.connect(self._on_item_changed)
        # 整行点击切换勾选 —— ItemIsUserCheckable 已移除以禁用 checkbox 原生
        # 切换(避免 row-click 与 checkbox-click 双重 toggle 抵消),所有 toggle
        # 走 itemClicked → setCheckState 路径,itemChanged 仍是状态真源。
        self._list.itemClicked.connect(self._on_item_clicked)
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(self._list)
        self._menu.addAction(action)

        # Quick-action footer — "Select all" / "Clear".
        self._menu.addSeparator()
        act_all = QAction(tr("chart.runs.select_all", "Select all"), self._menu)
        act_all.triggered.connect(self._select_all)
        self._menu.addAction(act_all)
        act_none = QAction(tr("chart.runs.clear", "Clear"), self._menu)
        act_none.triggered.connect(self._clear_all)
        self._menu.addAction(act_none)
        self.setMenu(self._menu)

        # Refresh on training lifecycle so newly-spawned runs auto-appear.
        sigs = get_app_signals()
        sigs.training_run_started.connect(self._on_run_lifecycle)
        sigs.training_run_finished.connect(self._on_run_lifecycle)

        self.apply_theme()
        self.refresh(auto_select_latest=True)

        # 按钮文本由 _update_button_label() 按选择态动态生成（"No runs" /
        # "Select runs" / "{n} runs" / 单 run 标签），不能 i18n_bind setText。
        # 语种切换后重跑一次该方法即可重译；菜单里的 "Select all" / "Clear"
        # action 是构造时一次性 tr()，会保持旧语种 —— 重建菜单成本高，先接受。
        I18n.instance().language_changed.connect(self._update_button_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def selected_run_ids(self) -> List[str]:
        """Snapshot of the currently checked run_ids."""
        return [rid for rid in self._all_run_ids if rid in self._selected]

    def set_selected(self, run_ids: Iterable[str]) -> None:
        """Programmatic select — rebuilds checkmarks + emits the change signal."""
        new = {str(rid) for rid in run_ids if rid}
        if new == self._selected:
            return
        self._selected = new
        self._sync_checkmarks()
        self._update_button_label()
        self.selected_runs_changed.emit(self.selected_run_ids())

    def refresh(self, *, auto_select_latest: bool = False) -> None:
        """Re-read sources + rebuild the popup list. Preserves prior selection."""
        runs = self._build_runs()
        self._all_run_ids = [r.run_id for r in runs]

        # Drop any selections whose runs are no longer in the union.
        self._selected = {rid for rid in self._selected if rid in self._all_run_ids}

        self._list.blockSignals(True)
        try:
            self._list.clear()
            for run in runs:
                item = QListWidgetItem(self._format_label(run), self._list)
                item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                # 显式去掉 ItemIsUserCheckable:checkbox 仍随 CheckState 渲染,
                # 但原生点击 checkbox 不再自动 toggle —— 全部 toggle 走 _on_item_clicked。
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if run.run_id in self._selected
                    else Qt.CheckState.Unchecked
                )
        finally:
            self._list.blockSignals(False)

        if auto_select_latest and not self._selected and self._all_run_ids:
            # `_build_runs` orders most-recent-first, so the head is what
            # the user is most likely to want on first open.
            head = self._all_run_ids[0]
            self._selected = {head}
            self._sync_checkmarks()
            self.selected_runs_changed.emit(self.selected_run_ids())

        self._update_button_label()

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        bg_hover = Config.get_color("hover_2")
        border = Config.get_color("border_1")
        text = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QToolButton#runSourceSelector {{"
            f"  background-color: {bg};"
            f"  color: {text};"
            f"  border: 1px solid {border};"
            f"  border-radius: 4px;"
            f"  padding: 4px 18px 4px 10px;"
            f"  font-size: {font_small}px;"
            f"  text-align: left;"
            f"}}"
            f"QToolButton#runSourceSelector:hover {{ background-color: {bg_hover}; }}"
            f"QToolButton#runSourceSelector::menu-indicator {{"
            f"  subcontrol-position: right center;"
            f"  subcontrol-origin: padding;"
            f"  right: 4px;"
            f"}}"
        )
        self._menu.setStyleSheet(
            f"QMenu#runSourceMenu {{"
            f"  background-color: {bg};"
            f"  border: 1px solid {border};"
            f"  color: {text};"
            f"  font-size: {font_small}px;"
            f"}}"
            f"QMenu#runSourceMenu::item:selected {{ background-color: {bg_hover}; }}"
            f"QListWidget#runSourceList {{"
            f"  background-color: {bg};"
            f"  color: {text};"
            f"  border: none;"
            f"  outline: 0;"
            f"  font-size: {font_small}px;"
            f"}}"
            f"QListWidget#runSourceList::item {{ padding: 4px 8px; }}"
            f"QListWidget#runSourceList::item:hover {{ background-color: {bg_hover}; }}"
        )
        # Silence Pylance "unused" warnings on font_normal / sub — slots
        # left here so future tweaks don't re-derive them.
        _ = font_normal, sub

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_runs(self) -> List[RunSummary]:
        """Union live cache + persisted store, dedup by run_id."""
        cache_runs = get_metrics_cache().list_runs()              # most-recent first
        persisted = self._collect_persisted_runs()
        seen: Set[str] = set()
        merged: List[RunSummary] = []
        for r in cache_runs:
            if r.run_id in seen:
                continue
            seen.add(r.run_id)
            merged.append(r)
        for r in persisted:
            if r.run_id in seen:
                continue
            seen.add(r.run_id)
            merged.append(r)
        return merged

    def _collect_persisted_runs(self) -> List[RunSummary]:
        """Pull historical runs from ``TrainingStore`` if a project is bound.

        The selector tolerates any missing dependency: no project, no
        store, store-throws — all silently fall back to "no historical
        runs". Live cache is always available.
        """
        provider = self._project_info_provider
        if provider is None:
            return []
        try:
            project_info = provider()
        except Exception:
            return []
        if project_info is None:
            return []
        try:
            from application.service.project_store import TrainingStore
            store = TrainingStore(project_info)
            records = store.list_runs()
        except Exception:
            return []
        out: List[RunSummary] = []
        for rec in records:
            run_id = getattr(rec, "run_id", None) or ""
            if not run_id:
                continue
            algo = getattr(rec, "algorithm", "") or ""
            label = f"{algo or 'run'} #{run_id[-8:]}".strip()
            status = getattr(rec, "status", "finished") or "finished"
            started = float(
                self._parse_iso(getattr(rec, "submitted_at", None)) or 0.0
            )
            finished = self._parse_iso(getattr(rec, "finished_at", None))
            out.append(RunSummary(
                run_id=str(run_id),
                label=label,
                status=str(status),
                started_at=started,
                finished_at=finished,
                sample_count=int(getattr(rec, "last_step", 0) or 0),
            ))
        return out

    @staticmethod
    def _parse_iso(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(value)).timestamp()
        except Exception:
            return None

    @staticmethod
    def _format_label(run: RunSummary) -> str:
        suffix_map = {
            "running": tr("chart.runs.tag.running", "[ON-GOING]"),
            "finished": tr("chart.runs.tag.finished", "[finished]"),
            "failed": tr("chart.runs.tag.failed", "[failed]"),
        }
        suffix = suffix_map.get(run.status, "")
        if suffix:
            return f"{run.label}  {suffix}"
        return run.label

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        run_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(run_id, str):
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._selected.add(run_id)
        else:
            self._selected.discard(run_id)
        self._update_button_label()
        self.selected_runs_changed.emit(self.selected_run_ids())

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """整行点击 → 翻转 checkState。itemChanged 自然 cascades 到 _on_item_changed。"""
        if item is None:
            return
        new = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new)

    def _on_run_lifecycle(self, *_args) -> None:
        # Train start/finish: rebuild list (auto-select the new run if
        # nothing is checked yet, mirroring DEMO's "force_default" path).
        had_selection = bool(self._selected)
        self.refresh(auto_select_latest=not had_selection)

    def _sync_checkmarks(self) -> None:
        self._list.blockSignals(True)
        try:
            for i in range(self._list.count()):
                item = self._list.item(i)
                run_id = item.data(Qt.ItemDataRole.UserRole)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if run_id in self._selected
                    else Qt.CheckState.Unchecked
                )
        finally:
            self._list.blockSignals(False)

    def _update_button_label(self) -> None:
        n = len(self._selected)
        if n == 0:
            if not self._all_run_ids:
                self.setText(tr("chart.runs.empty", "No runs"))
            else:
                self.setText(tr("chart.runs.none_selected", "Select runs"))
            return
        if n == 1:
            rid = next(iter(self._selected))
            summary = get_metrics_cache().get_summary(rid)
            label = summary.label if summary is not None else rid[-8:]
            self.setText(label)
            return
        self.setText(tr("chart.runs.n_selected", f"{n} runs"))

    def _select_all(self) -> None:
        if not self._all_run_ids:
            return
        self.set_selected(self._all_run_ids)

    def _clear_all(self) -> None:
        self.set_selected([])


__all__ = ["RunSourceSelector"]
