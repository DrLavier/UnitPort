"""Capability inspector widget — Cycle 2 STAGE-04.

Provides :class:`CapabilityInspector`, a PySide6 widget that renders the
dict returned by ``adapter.capabilities()`` in a structured, operator-readable
form and links missing required settings to the settings panel.

Public API::

    CapabilityInspector(capabilities_dict, missing_settings, parent)
    CapabilityInspector.set_capabilities(cap, missing_settings) -> None
    CapabilityInspector.focus_setting   Signal(str)   # field_id clicked

Rendering policy
----------------
- Header row: Brand and Adapter identity.
- **Actions** section: one label per action name.
- **Sensors** section: one label per sensor key.
- **Feature Flags** section: key → value pairs.
- **Required Settings** section: one QPushButton per field_id; clicking emits
  ``focus_setting(field_id)``.  Fields in *missing_settings* are rendered with
  a warning indicator so operators know they still need to be filled.

Partial-data resilience
-----------------------
- Input that is not a dict: renders "No capability data available".
- Individual missing keys: that section renders "(not declared)" and the
  remaining sections still render normally.
- Unknown extra keys are silently ignored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ── Style constants ──────────────────────────────────────────────────────────

_STYLE_MISSING_BTN = (
    "text-align: left; color: #f59e0b; border: 1px solid #f59e0b; "
    "border-radius: 3px; padding: 2px 6px; background: transparent;"
)
_STYLE_NORMAL_BTN = (
    "text-align: left; color: #60a5fa; border: 1px solid #374151; "
    "border-radius: 3px; padding: 2px 6px; background: transparent;"
)
_STYLE_UNAVAILABLE = "color: #6b7280; font-style: italic;"


class CapabilityInspector(QWidget):
    """Structured display of an adapter capability descriptor.

    Args:
        capabilities_dict:  The dict returned by ``adapter.capabilities()``.
                            Pass ``{}`` or ``None`` for empty state.
        missing_settings:   Optional list of field_id strings that are
                            required but not yet filled in the settings panel.
                            These are highlighted in the Required Settings
                            section as needing attention.
        parent:             Optional parent widget.
    """

    #: Emitted when the operator clicks a required-settings button.
    focus_setting = Signal(str)

    def __init__(
        self,
        capabilities_dict: Optional[Dict[str, Any]] = None,
        missing_settings: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cap: Dict[str, Any] = (
            capabilities_dict if isinstance(capabilities_dict, dict) else {}
        )
        self._missing: List[str] = list(missing_settings) if missing_settings else []
        self._init_ui()

    # ── Initial layout ───────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Persistent header label (compatibility handle expected by tests)
        self._header_label = QLabel("")
        self._header_label.setObjectName("capHeaderLabel")
        root.addWidget(self._header_label)

        # Scrollable content
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setObjectName("capInspectorScroll")
        root.addWidget(self._scroll, 1)

        self._build_content()

    # ── Content builder ──────────────────────────────────────────────────────

    def _build_content(self) -> None:
        """Discard current content widget and build a fresh one."""
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()

        container = QWidget()
        container.setObjectName("capInspectorContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(10)

        if not isinstance(self._cap, dict) or not self._cap:
            # Empty cap — show placeholder but do NOT clear _header_label so
            # callers that want to preserve the last-known brand info can do so.
            lbl = QLabel("No capability data available.")
            lbl.setStyleSheet(_STYLE_UNAVAILABLE)
            layout.addWidget(lbl)
            layout.addStretch()
            self._scroll.setWidget(container)
            return

        # ── Identity row ─────────────────────────────────────────────────
        brand   = self._cap.get("brand",   "")
        adapter = self._cap.get("adapter", "")
        if brand or adapter:
            header_text = f"Brand: {brand or '—'}  |  Adapter: {adapter or '—'}"
            self._header_label.setText(header_text)
            id_label = QLabel(
                f"<b>Brand:</b> {brand or '—'}  &nbsp; "
                f"<b>Adapter:</b> {adapter or '—'}"
            )
            id_label.setObjectName("capIdentityLabel")
            layout.addWidget(id_label)

        # ── Sections ─────────────────────────────────────────────────────
        layout.addWidget(
            self._make_list_section(
                "Actions",
                self._cap.get("actions"),
            )
        )
        layout.addWidget(
            self._make_list_section(
                "Sensors",
                self._cap.get("sensors"),
            )
        )
        layout.addWidget(
            self._make_flags_section(self._cap.get("flags"))
        )
        layout.addWidget(
            self._make_required_section(self._cap.get("required_settings"))
        )

        layout.addStretch()
        self._scroll.setWidget(container)

    # ── Section factories ────────────────────────────────────────────────────

    def _make_list_section(
        self, title: str, items: Optional[List[str]]
    ) -> QGroupBox:
        """Render a list of strings as a group of labels."""
        box = QGroupBox(title)
        box.setObjectName("capSection")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 10, 8, 6)
        v.setSpacing(3)

        if not isinstance(items, list) or not items:
            lbl = QLabel("(not declared)")
            lbl.setStyleSheet(_STYLE_UNAVAILABLE)
            v.addWidget(lbl)
        else:
            for item in items:
                lbl = QLabel(str(item))
                lbl.setObjectName("capListItem")
                v.addWidget(lbl)

        return box

    def _make_flags_section(
        self, flags: Optional[Dict[str, Any]]
    ) -> QGroupBox:
        """Render feature flags as key → value pairs."""
        box = QGroupBox("Feature Flags")
        box.setObjectName("capSection")
        form = QFormLayout(box)
        form.setContentsMargins(8, 10, 8, 6)
        form.setSpacing(4)

        if not isinstance(flags, dict) or not flags:
            lbl = QLabel("(not declared)")
            lbl.setStyleSheet(_STYLE_UNAVAILABLE)
            form.addRow(lbl)
        else:
            for key, val in flags.items():
                val_lbl = QLabel(str(val))
                val_lbl.setObjectName("capFlagValue")
                form.addRow(QLabel(str(key)), val_lbl)

        return box

    def _make_required_section(
        self, required: Optional[List[str]]
    ) -> QGroupBox:
        """Render required settings as clickable buttons.

        Buttons in *self._missing* are highlighted in amber to indicate that
        the operator still needs to fill them in.
        """
        box = QGroupBox("Required Settings")
        box.setObjectName("capSection")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 10, 8, 6)
        v.setSpacing(3)

        if not isinstance(required, list) or not required:
            lbl = QLabel("(not declared)")
            lbl.setStyleSheet(_STYLE_UNAVAILABLE)
            v.addWidget(lbl)
            return box

        for field_id in required:
            is_missing = field_id in self._missing
            prefix = "⚠ " if is_missing else ""
            suffix = "  (missing)" if is_missing else ""
            btn = QPushButton(f"{prefix}{field_id}{suffix}")
            btn.setObjectName("capRequiredBtn")
            btn.setStyleSheet(
                _STYLE_MISSING_BTN if is_missing else _STYLE_NORMAL_BTN
            )
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            btn.setFlat(True)
            # Capture field_id in closure
            btn.clicked.connect(
                lambda checked=False, fid=field_id: self.focus_setting.emit(fid)
            )
            v.addWidget(btn)

        return box

    # ── Public API ────────────────────────────────────────────────────────────

    def set_capabilities(
        self,
        capabilities_dict: Optional[Dict[str, Any]],
        missing_settings: Optional[List[str]] = None,
    ) -> None:
        """Reload the inspector with new capability data.

        Args:
            capabilities_dict: New capability dict (or None/empty for reset).
            missing_settings:  Updated list of field_ids not yet filled.
        """
        self._cap = (
            capabilities_dict
            if isinstance(capabilities_dict, dict)
            else {}
        )
        self._missing = list(missing_settings) if missing_settings else []
        self._build_content()
