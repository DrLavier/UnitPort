"""DiagnosticsPanel — per-node execution diagnostics drill-down widget (STAGE-04).

Shows structured diagnostics for failed canvas nodes after a mission run.
Emits navigate_requested(node_id) when the user clicks "Go to Node".

Layout::

    ┌─────────────────────────────────────────────────────┐
    │  ⚠ Diagnostics  [Node ▼]                  [Raw □]   │  ← header
    ├─────────────────────────────────────────────────────┤
    │  Stage:          Execution                          │
    │  Description:    Action execution failed.           │  ← friendly
    │  Message:        Robot timed out after 5s           │
    │  Adapter:        unitree_sdk2                       │
    │  Retryable:      No                                 │
    ├─────────────────────────────────────────────────────┤
    │  [Go to Node]                         [Close ×]     │  ← footer
    └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.system.core.error_ux import DISPLAY_KEY_ORDER, get_stage_category
from src.system.core.localisation import tr


class DiagnosticsPanel(QWidget):
    """Node-level diagnostics drill-down panel (Cycle 1 STAGE-04).

    Signals:
        navigate_requested(object): Emitted with the int node_id when the user
            clicks "Go to Node".  Connect to a handler that calls
            ``graph_view.centerOn(graph_scene.get_node_item(node_id))``.
        closed(): Emitted when the panel is dismissed.
    """

    navigate_requested = Signal(object)   # node_id (int or str)
    closed = Signal()

    # Human-readable labels for known diagnostic keys.
    # Values are plain strings — localisation of these labels (if needed) is
    # the display layer's responsibility.
    _KEY_LABELS: Dict[str, str] = {
        "node_id":          "Node ID",
        "node_name":        "Node Name",
        "status":           "Status",
        "stage":            "Stage",
        "error_category":   "Category",       # STAGE-06: error type distinction
        "reason":           "Reason Code",
        "operator_text":    "Description",
        "message":          "Message",
        "adapter_name":     "Adapter",
        "retryable":        "Retryable",
        "context":          "Context",
        "compat_path":      "Compat Path",
        "compat_reason":    "Compat Reason",
        # Telemetry pointers
        "trace_id":         "Trace ID",
        "mission_trace_id": "Mission Trace",
        "package_metadata_trace": "Package Metadata Trace",
    }

    # User-visible labels for the four error category constants (STAGE-06).
    # Kept here (not in error_ux) so only the display layer renders them;
    # error_ux returns raw constant strings for engineering use.
    _CATEGORY_LABELS: Dict[str, str] = {
        "validation":      "Validation Error",
        "preflight_safety": "Preflight / Safety Block",
        "runtime":         "Runtime Failure",
        "compat_warning":  "Compatibility Warning",
    }

    # Keys shown in "friendly" mode (excluding node meta already in header).
    # error_category appears right after stage so the operator sees error
    # type and lifecycle stage together.
    # Telemetry pointer keys are shown when present for log correlation.
    _FRIENDLY_KEYS: List[str] = [
        "stage",
        "error_category",   # STAGE-06
        "operator_text",
        "message",
        "adapter_name",
        "retryable",
        # Telemetry pointers — shown only when non-empty (absent keys are skipped)
        "trace_id",
        "mission_trace_id",
        "package_metadata_trace",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("diagnosticsPanel")
        self._node_infos: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._build_ui()
        self.setVisible(False)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        # ── Header row: title + node selector + raw toggle ────────────────
        header = QHBoxLayout()
        header.setSpacing(8)

        title_lbl = QLabel(tr("diagnostics_panel.title", "⚠  Diagnostics"))
        title_lbl.setObjectName("diagTitle")
        self._title_lbl = title_lbl
        header.addWidget(title_lbl)

        self._node_combo = QComboBox()
        self._node_combo.setObjectName("diagNodeCombo")
        self._node_combo.setMinimumWidth(160)
        self._node_combo.currentIndexChanged.connect(self._on_node_selected)
        header.addWidget(self._node_combo)

        header.addStretch(1)

        self._raw_toggle = QCheckBox(tr("diagnostics_panel.raw", "Raw"))
        self._raw_toggle.setObjectName("diagRawToggle")
        self._raw_toggle.toggled.connect(self._refresh_view)
        header.addWidget(self._raw_toggle)

        root.addLayout(header)

        # ── Scroll area for friendly field rows ───────────────────────────
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_area.setObjectName("diagScrollArea")

        self._fields_widget = QWidget()
        self._fields_widget.setObjectName("diagFieldsWidget")
        self._fields_layout = QVBoxLayout(self._fields_widget)
        self._fields_layout.setContentsMargins(0, 0, 0, 0)
        self._fields_layout.setSpacing(2)
        self._scroll_area.setWidget(self._fields_widget)
        root.addWidget(self._scroll_area)

        # ── Raw JSON text view (visible only when Raw toggle is checked) ──
        self._raw_text = QPlainTextEdit()
        self._raw_text.setObjectName("diagRawText")
        self._raw_text.setReadOnly(True)
        self._raw_text.setFixedHeight(160)
        self._raw_text.setVisible(False)
        root.addWidget(self._raw_text)

        # ── Footer row: Go to Node + Close ────────────────────────────────
        footer = QHBoxLayout()
        footer.setSpacing(8)

        self._goto_btn = QPushButton(tr("diagnostics_panel.goto_node", "Go to Node"))
        self._goto_btn.setObjectName("diagGotoBtn")
        self._goto_btn.clicked.connect(self._on_goto_clicked)
        footer.addWidget(self._goto_btn)

        footer.addStretch(1)

        close_btn = QPushButton(tr("diagnostics_panel.close", "Close ×"))
        close_btn.setObjectName("diagCloseBtn")
        close_btn.clicked.connect(self._on_close)
        self._close_btn = close_btn
        footer.addWidget(close_btn)

        root.addLayout(footer)

    # ── Public API ────────────────────────────────────────────────────────

    def show_diagnostics(
        self,
        node_infos: List[Dict[str, Any]],
        start_index: int = 0,
    ) -> None:
        """Populate and show the panel with formatted node diagnostic dicts.

        Args:
            node_infos:  List produced by error_ux.extract_failed_nodes_info().
            start_index: Which node to display first (default 0).
        """
        self._node_infos = list(node_infos)
        self._current_index = max(0, min(start_index, len(self._node_infos) - 1))

        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        for info in self._node_infos:
            label = info.get("node_name") or info.get("node_id", "?")
            self._node_combo.addItem(str(label))
        self._node_combo.setCurrentIndex(self._current_index)
        self._node_combo.blockSignals(False)

        # Only show selector when there are multiple failed nodes
        self._node_combo.setVisible(len(self._node_infos) > 1)

        self._refresh_view()
        self.setVisible(True)

    def clear(self) -> None:
        """Hide the panel and reset all content."""
        self._node_infos = []
        self._current_index = 0
        self._node_combo.clear()
        self._clear_fields()
        self._raw_text.clear()
        self.setVisible(False)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_node_selected(self, index: int) -> None:
        self._current_index = index
        self._refresh_view()

    def _on_goto_clicked(self) -> None:
        if not self._node_infos:
            return
        info = self._node_infos[self._current_index]
        raw_id = info.get("node_id", "")
        # Emit as int when possible — node items store data(12) as int
        try:
            node_id: Any = int(raw_id)
        except (ValueError, TypeError):
            node_id = raw_id
        self.navigate_requested.emit(node_id)

    def _on_close(self) -> None:
        self.clear()
        self.closed.emit()

    # ── Rendering ─────────────────────────────────────────────────────────

    def _refresh_view(self) -> None:
        if not self._node_infos:
            return
        self._refresh_static_texts()
        info = self._node_infos[self._current_index]
        show_raw = self._raw_toggle.isChecked()
        self._scroll_area.setVisible(not show_raw)
        self._raw_text.setVisible(show_raw)
        if show_raw:
            self._raw_text.setPlainText(json.dumps(info, indent=2, default=str))
        else:
            self._render_friendly(info)

    def _render_friendly(self, info: Dict[str, Any]) -> None:
        self._clear_fields()

        # Build ordered display key list: friendly subset first, then rest
        shown: List[str] = []
        for k in self._FRIENDLY_KEYS:
            if k in info:
                shown.append(k)
        for k in DISPLAY_KEY_ORDER:
            if k not in shown and k in info and k not in ("node_id", "node_name"):
                shown.append(k)
        for k in sorted(info.keys()):
            if k not in shown:
                shown.append(k)

        for key in shown:
            value = info[key]
            # Skip empty context dict
            if key == "context" and isinstance(value, dict) and not value:
                continue

            label_text = self._get_key_label(key)
            value_text = self._format_value(key, value)

            row = QHBoxLayout()
            row.setSpacing(8)

            key_lbl = QLabel(f"<b>{label_text}:</b>")
            key_lbl.setFixedWidth(110)
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

            val_lbl = QLabel(value_text)
            val_lbl.setWordWrap(True)
            val_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            row.addWidget(key_lbl)
            row.addWidget(val_lbl, 1)
            self._fields_layout.addLayout(row)

        self._fields_layout.addStretch(1)

    def _clear_fields(self) -> None:
        while self._fields_layout.count():
            item = self._fields_layout.takeAt(0)
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    w_item = layout.takeAt(0)
                    if w_item.widget():
                        w_item.widget().deleteLater()
                layout.deleteLater()
            elif item.widget():
                item.widget().deleteLater()

    @classmethod
    def _format_value(cls, key: str, value: Any) -> str:
        if key == "stage":
            stage = str(value)
            stage_key_map = {
                "acquire": "diagnostics_panel.stage.acquire",
                "open_session": "diagnostics_panel.stage.open_session",
                "preflight": "diagnostics_panel.stage.preflight",
                "execute": "diagnostics_panel.stage.execute",
                "close_session": "diagnostics_panel.stage.close_session",
                "any": "diagnostics_panel.stage.any",
            }
            if stage in stage_key_map:
                return tr(stage_key_map[stage], get_stage_category(stage))
            return get_stage_category(stage)
        if key == "error_category":
            # Map raw category constant → human-readable label (STAGE-06).
            cat = str(value)
            cat_key_map = {
                "validation": "diagnostics_panel.category.validation",
                "preflight_safety": "diagnostics_panel.category.preflight_safety",
                "runtime": "diagnostics_panel.category.runtime",
                "compat_warning": "diagnostics_panel.category.compat_warning",
            }
            if cat in cat_key_map:
                return tr(cat_key_map[cat], cls._CATEGORY_LABELS.get(cat, cat.replace("_", " ").title()))
            return cls._CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())
        if key == "retryable":
            return tr("diagnostics_panel.value.yes", "Yes") if value else tr("diagnostics_panel.value.no", "No")
        if key == "context" and isinstance(value, dict):
            return "(empty)" if not value else json.dumps(value, indent=2, default=str)
        if key == "package_metadata_trace" and isinstance(value, dict):
            return "(empty)" if not value else json.dumps(value, indent=2, default=str)
        if key in ("trace_id", "mission_trace_id"):
            # Telemetry UUIDs — return as-is; empty is not expected here.
            return str(value) if value else "(none)"
        if value is None or value == "":
            return "(none)"
        return str(value)

    @classmethod
    def _get_key_label(cls, key: str) -> str:
        key_map = {
            "node_id": "diagnostics_panel.key.node_id",
            "node_name": "diagnostics_panel.key.node_name",
            "status": "diagnostics_panel.key.status",
            "stage": "diagnostics_panel.key.stage",
            "error_category": "diagnostics_panel.key.error_category",
            "reason": "diagnostics_panel.key.reason",
            "operator_text": "diagnostics_panel.key.operator_text",
            "message": "diagnostics_panel.key.message",
            "adapter_name": "diagnostics_panel.key.adapter_name",
            "retryable": "diagnostics_panel.key.retryable",
            "context": "diagnostics_panel.key.context",
            "compat_path": "diagnostics_panel.key.compat_path",
            "compat_reason": "diagnostics_panel.key.compat_reason",
            "trace_id": "diagnostics_panel.key.trace_id",
            "mission_trace_id": "diagnostics_panel.key.mission_trace_id",
            "package_metadata_trace": "diagnostics_panel.key.package_metadata_trace",
        }
        default = cls._KEY_LABELS.get(key, key.replace("_", " ").title())
        tr_key = key_map.get(key)
        if not tr_key:
            return default
        return tr(tr_key, default)

    def _refresh_static_texts(self) -> None:
        self._title_lbl.setText(tr("diagnostics_panel.title", "⚠  Diagnostics"))
        self._raw_toggle.setText(tr("diagnostics_panel.raw", "Raw"))
        self._goto_btn.setText(tr("diagnostics_panel.goto_node", "Go to Node"))
        self._close_btn.setText(tr("diagnostics_panel.close", "Close ×"))
