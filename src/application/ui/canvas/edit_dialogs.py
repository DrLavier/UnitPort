# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas.edit_dialogs — Popup re-exports + Coverage Inspector.

Historical "edit dialog" module. ParamRow editors have all migrated to
:mod:`.popups` (anchored ``Qt.WindowType.Popup``, click-outside closes).
This file keeps the compatibility re-exports for the old names plus the
Coverage Inspector — a *read-only* detail viewer launched by double-
clicking a Rewards-node Coverage Badge. Modal QDialog is appropriate
for the Inspector since it shows a structured report rather than an
inline edit; the Add recommended set button below it mutates state but
that mutation is a single discrete commit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, log_warning, tr

from .popups import (
    open_choice_popup,
    open_code_popup,
    open_number_popup,
    open_path_dialog,
    open_text_popup,
    open_widget_popup,
    screen_anchor_for_row,
)


# 历史名映射 —— 仅在外部代码尚有引用时不破坏
open_choice_menu = open_choice_popup  # 历史名：单选/多选


# =============================================================================
# Coverage Inspector — rewards × motion-phase isolation
# =============================================================================


class CoverageInspectorDialog(QDialog):
    """Reward × MotionPhase coverage detail viewer.

    Layout::

        ┌─ Reward Coverage ──────────────────────────┐
        │ ✗ Critical: static phase has no negative   │
        │   reward …                                  │
        │ ┌────────────────────────────────────────┐ │
        │ │ Phase       | +  | −  | ~  | Status    │ │
        │ │ Static      | 0  | 0  | 0  | ✗         │ │
        │ │ Locomotion  | 4  | 5  | 0  | ⚠         │ │
        │ │ Agile       | 2  | 3  | 0  | ✓         │ │
        │ │ Unconditional| 1 | 2  | 0  | ✓         │ │
        │ └────────────────────────────────────────┘ │
        │ Suggested fix for Static:                   │
        │   [+ Add recommended set]   [Close]         │
        └─────────────────────────────────────────────┘

    Mutation contract: clicking *Add recommended set* writes back through
    ``node.params['reward_terms']`` (kept canonical via the registry
    module setter); the dialog closes and the canvas repaints the new
    coverage on its next paint cycle.
    """

    def __init__(self, node_item, report, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._node = node_item
        self._report = report
        self.setWindowTitle(tr("canvas.rewards.coverage.inspector.title",
                               "Reward Coverage Inspector"))
        self.setModal(True)
        self.resize(640, 460)
        self._build_ui()
        self._populate()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Severity banner — themed via system.ini[Theme].
        self._banner = QLabel(self)
        self._banner.setWordWrap(True)
        self._banner.setMinimumHeight(40)
        self._banner.setProperty("class", "coverage-banner")
        self._banner.setStyleSheet(self._banner_stylesheet())
        root.addWidget(self._banner)

        # Per-phase table.
        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            tr("canvas.rewards.coverage.col.phase", "Phase"),
            tr("canvas.rewards.coverage.col.pos", "+"),
            tr("canvas.rewards.coverage.col.neg", "−"),
            tr("canvas.rewards.coverage.col.neu", "~"),
            tr("canvas.rewards.coverage.col.status", "Status"),
        ])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_phase_selected)
        root.addWidget(self._table, stretch=1)

        # Per-phase detail — terms by polarity + message.
        self._detail = QLabel(self)
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._detail.setMinimumHeight(80)
        self._detail.setStyleSheet(
            f"background:{Config.get_color('canvas_node_param_bg', '#232323')};"
            f"color:{Config.get_color('canvas_node_param_text', '#C8C8C8')};"
            f"padding:8px;border-radius:4px;"
        )
        root.addWidget(self._detail)

        # Button row.
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._add_btn = QPushButton(
            tr("canvas.rewards.coverage.add_set", "Add recommended set"),
            self,
        )
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._on_add_set)
        buttons.addWidget(self._add_btn)
        buttons.addStretch(1)
        close_btn = QPushButton(tr("canvas.button.close", "Close"), self)
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

    def _banner_stylesheet(self) -> str:
        sev = self._report.severity
        if sev == "critical":
            bg = Config.get_color("canvas_badge_critical", "#EF4444")
            text = Config.get_color("canvas_badge_text", "#0A0A0A")
        elif sev == "warning":
            bg = Config.get_color("canvas_badge_warning", "#F59E0B")
            text = Config.get_color("canvas_badge_text", "#0A0A0A")
        else:
            bg = Config.get_color("safe_zone", "#36E38E")
            text = Config.get_color("canvas_badge_text", "#0A0A0A")
        return (
            f"QLabel{{background:{bg};color:{text};"
            f"font-weight:bold;padding:8px;border-radius:4px;}}"
        )

    # ---- Data population -------------------------------------------------

    def _populate(self) -> None:
        report = self._report
        if not report.has_terms:
            self._banner.setText(tr(
                "canvas.rewards.coverage.empty",
                "No reward terms configured — add at least one term to get a coverage assessment.",
            ))
        elif report.severity == "critical":
            self._banner.setText(tr(
                "canvas.rewards.coverage.banner.critical",
                "Critical: one or more phases are missing required negative reward. "
                "Training may produce undesired behaviors such as idle limb-flailing.",
            ))
        elif report.severity == "warning":
            self._banner.setText(tr(
                "canvas.rewards.coverage.banner.warning",
                "Warning: at least one phase is missing recommended reward coverage.",
            ))
        else:
            self._banner.setText(tr(
                "canvas.rewards.coverage.banner.ok",
                "All phases have positive + negative reward coverage.",
            ))

        rows = report.phases
        self._table.setRowCount(len(rows))
        for r, ph in enumerate(rows):
            name_item = QTableWidgetItem(ph.display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, ph)
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, _centered(str(len(ph.positive_terms))))
            self._table.setItem(r, 2, _centered(str(len(ph.negative_terms))))
            self._table.setItem(r, 3, _centered(str(len(ph.neutral_terms))))
            status_item = _centered(_sev_glyph(ph.severity))
            status_item.setForeground(QColor(_sev_color(ph.severity)))
            self._table.setItem(r, 4, status_item)
        if rows:
            self._table.selectRow(0)

    def _on_phase_selected(self) -> None:
        sel = self._table.selectedItems()
        if not sel:
            self._detail.setText("")
            self._add_btn.setEnabled(False)
            return
        ph = sel[0].data(Qt.ItemDataRole.UserRole)
        if ph is None:
            return
        lines: List[str] = []
        lines.append(
            f"<b>{ph.display_name}</b> &nbsp; "
            f"<i>polarity_required = {ph.polarity_required}</i>"
        )
        if ph.message:
            lines.append(f"<span style='color:{_sev_color(ph.severity)};'>{ph.message}</span>")
        lines.append(
            f"<b>+</b> {', '.join(ph.positive_terms) if ph.positive_terms else '—'}"
        )
        lines.append(
            f"<b>−</b> {', '.join(ph.negative_terms) if ph.negative_terms else '—'}"
        )
        if ph.neutral_terms:
            lines.append(f"<b>~</b> {', '.join(ph.neutral_terms)}")
        if ph.suggestions:
            lines.append("")
            lines.append("<b>Suggested:</b> " + ", ".join(ph.suggestions))
        self._detail.setText("<br>".join(lines))
        self._add_btn.setEnabled(bool(ph.suggestions))

    # ---- Mutations -------------------------------------------------------

    def _on_add_set(self) -> None:
        """Patch reward_terms with the selected phase's suggested terms.

        v2: phase-aware ``applies_to`` was removed — reward fan-out is now
        expressed at the canvas level by connecting a Rewards node's
        ``reward_pipe`` to specific training_motion items. Each suggestion
        is inserted as ``{weight: <default>}``; phase selection is the
        user's job via canvas wiring. Existing terms with the same key
        are left untouched (non-destructive).
        """
        sel = self._table.selectedItems()
        if not sel:
            return
        ph = sel[0].data(Qt.ItemDataRole.UserRole)
        if ph is None or not ph.suggestions:
            return
        raw = self._node.params.get("reward_terms", "{}")
        if isinstance(raw, str):
            try:
                terms: Dict[str, Any] = json.loads(raw) if raw.strip() else {}
            except Exception:
                terms = {}
        elif isinstance(raw, dict):
            terms = dict(raw)
        else:
            terms = {}
        backend = str(self._node.params.get("backend", "sb3") or "sb3").strip()
        added = 0
        for term_key in ph.suggestions:
            if term_key in terms:
                continue
            default_weight = _lookup_default_weight(term_key, backend=backend)
            terms[term_key] = {"weight": default_weight}
            added += 1
        if not added:
            log_warning(
                f"[canvas] Coverage Inspector: no new suggestions to add for "
                f"phase {ph.phase_id!r} (all already present)"
            )
            return
        # Commit through the same channel the row uses so undo / page-side
        # invalidation pick it up. We go through the row's set_value to
        # keep _row_payloads in sync.
        new_val = json.dumps(terms, ensure_ascii=False)
        try:
            row = _find_reward_terms_row(self._node)
        except Exception:
            row = None
        if row is not None:
            row.set_value(new_val)
            row.update()
        else:
            # Last-ditch: write directly into node params so at least the
            # Badge re-evaluates correctly on next paint.
            self._node.params["reward_terms"] = new_val
            self._node.update()
        self.accept()


def _centered(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
    return item


def _sev_glyph(severity: str) -> str:
    return {"critical": "✗", "warning": "⚠", "ok": "✓"}.get(severity, "·")


def _sev_color(severity: str) -> str:
    if severity == "critical":
        return Config.get_color("canvas_badge_critical", "#EF4444")
    if severity == "warning":
        return Config.get_color("canvas_badge_warning", "#F59E0B")
    return Config.get_color("safe_zone", "#36E38E")


def _lookup_default_weight(term_key: str, *, backend: str) -> float:
    """Resolve a registered reward's default weight, or 1.0 on miss."""
    try:
        from scripts import lookup as _lookup
        item = _lookup(term_key, kind="reward", backend=backend)
    except Exception:
        return 1.0
    if item is None:
        return 1.0
    try:
        return float(getattr(item, "default", 1.0))
    except Exception:
        return 1.0


def _find_reward_terms_row(node_item):
    """Locate the ``reward_terms`` ParamRow attached to a NodeItem."""
    rows = getattr(node_item, "_param_rows", []) or []
    for row in rows:
        if getattr(row.spec, "key", "") == "reward_terms":
            return row
    return None


def open_coverage_inspector(event, node_item) -> None:
    """Entry point invoked by ``NodeItem.mouseDoubleClickEvent``.

    Resolves the parent QGraphicsView (so the dialog inherits the right
    window), pulls the cached report if available, otherwise recomputes
    via the evaluator. Dialog instances are deleteOnClose to avoid
    leaking widgets when the user re-opens the inspector repeatedly.
    """
    view: Optional[QGraphicsView] = None
    try:
        view = event.widget().parent() if event.widget() is not None else None
        if not isinstance(view, QGraphicsView):
            view = None
    except Exception:
        view = None
    report = getattr(node_item, "_coverage_report_cache", None)
    if report is None or not getattr(report, "phases", None):
        report = node_item._compute_coverage()
    if report is None:
        return
    parent_window = view.window() if view is not None else None
    dlg = CoverageInspectorDialog(node_item, report, parent=parent_window)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.exec()


__all__ = [
    "open_choice_popup",
    "open_choice_menu",  # 历史名
    "open_code_popup",
    "open_number_popup",
    "open_path_dialog",
    "open_text_popup",
    "open_widget_popup",
    "screen_anchor_for_row",
    "open_coverage_inspector",
    "CoverageInspectorDialog",
]
