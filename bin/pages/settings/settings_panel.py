"""Behavior SDK settings panel — Cycle 2 STAGE-03.

Provides :class:`SettingsPanel`, a PySide6 widget that dynamically renders
a form from ``bin.core.settings_form.schema_to_form_descriptors()`` and
integrates ``system.service.settings_validator.validate_settings()`` on Apply.

Public API::

    SettingsPanel(brand, config, parent)
    SettingsPanel.get_current_config() -> dict
    SettingsPanel.has_unsaved_changes() -> bool
    SettingsPanel.set_brand(brand, config) -> None
    SettingsPanel.settings_applied   Signal(dict)   # after successful Apply
    SettingsPanel.settings_reset     Signal()        # after Reset

Form rendering policy
---------------------
- One widget per field descriptor produced by ``schema_to_form_descriptors()``.
- Fields are grouped into **Required Settings** and **Optional Settings** QGroupBoxes.
- Widget type → Qt widget mapping:
    ``"choice"``  → QComboBox
    ``"bool"``    → QCheckBox
    ``"integer"`` → QSpinBox
    ``"float"``   → QDoubleSpinBox
    ``"text"``    → QLineEdit
    ``"list"``    → QLineEdit (comma-separated; read/write converts to/from list)
- ``validation_hints`` is used as QLineEdit placeholder text and row label tooltip.
- Per-field error labels (hidden by default) appear below each widget on failed Apply.

Apply flow
----------
1. Collect current widget values into a config dict.
2. Call ``validate_settings(brand, config)``.
3. On error: show per-field error labels; update status bar.  Do not emit.
4. On ok:   persist ``_applied_config``; emit ``settings_applied(config)``; clear errors.

Reset flow
----------
- Restores widget values from ``_applied_config`` (last applied, or the ``config``
  argument passed at construction / ``set_brand()``).
- Clears all error labels.
- Emits ``settings_reset()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.system.core.settings_form import schema_to_form_descriptors
from src.system.service.settings_schema import KNOWN_BRANDS
from src.system.service.settings_validator import validate_settings


class SettingsPanel(QWidget):
    """Dynamic Behavior SDK settings panel.

    Args:
        brand:  Initial brand key (one of ``KNOWN_BRANDS``).  Falls back to
                the first known brand if not recognised.
        config: Initial settings dict used to pre-populate the form and as the
                baseline for :meth:`has_unsaved_changes`.  Missing keys are
                left at widget defaults.
        parent: Optional parent widget.
    """

    #: Emitted with the validated config dict after a successful Apply.
    settings_applied = Signal(dict)
    #: Emitted after a Reset restores the form to the last applied state.
    settings_reset = Signal()
    #: Emitted when any field value changes (live/draft settings).
    settings_changed = Signal(dict)

    def __init__(
        self,
        brand: str = "unitree",
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._brand: str = brand if brand in KNOWN_BRANDS else KNOWN_BRANDS[0]
        self._applied_config: Dict[str, Any] = dict(config) if config else {}
        self._field_widgets: Dict[str, QWidget] = {}
        self._error_labels: Dict[str, QLabel] = {}
        self._init_ui()

    # ── Initial layout ───────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setObjectName("settingsPanelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(6)
        self._title_label = QLabel(f"SDK Settings — {self._brand}")
        self._title_label.setObjectName("settingsPanelTitle")
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("settingsApplyBtn")
        self._apply_btn.setFixedWidth(70)
        self._apply_btn.clicked.connect(self._on_apply)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setObjectName("settingsResetBtn")
        self._reset_btn.setFixedWidth(70)
        self._reset_btn.clicked.connect(self._on_reset)
        header_layout.addWidget(self._apply_btn)
        header_layout.addWidget(self._reset_btn)
        root.addWidget(header)

        # Scrollable form area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setObjectName("settingsScrollArea")
        # _form_container and _form_layout are (re)created by _build_form
        self._form_container: QWidget = QWidget()
        self._form_layout: QVBoxLayout = QVBoxLayout(self._form_container)
        self._scroll_area.setWidget(self._form_container)
        root.addWidget(self._scroll_area, 1)

        # Status label at bottom
        self._status_label = QLabel("")
        self._status_label.setObjectName("settingsStatusLabel")
        self._status_label.setContentsMargins(12, 4, 12, 4)
        root.addWidget(self._status_label)

        self._build_form()

    # ── Form builder ─────────────────────────────────────────────────────────

    def _build_form(self) -> None:
        """Discard the current form container and build a fresh one."""
        self._field_widgets.clear()
        self._error_labels.clear()

        # Replace container widget so old children are garbage-collected.
        old = self._scroll_area.takeWidget()
        if old is not None:
            old.deleteLater()

        self._form_container = QWidget()
        self._form_container.setObjectName("settingsFormContainer")
        self._form_layout = QVBoxLayout(self._form_container)
        self._form_layout.setContentsMargins(12, 8, 12, 8)
        self._form_layout.setSpacing(8)
        self._scroll_area.setWidget(self._form_container)

        descriptors = schema_to_form_descriptors(self._brand)
        required_descs = [d for d in descriptors if d["required"]]
        optional_descs = [d for d in descriptors if not d["required"]]

        if required_descs:
            self._form_layout.addWidget(
                self._make_group("Required Settings", required_descs)
            )
        if optional_descs:
            self._form_layout.addWidget(
                self._make_group("Optional Settings", optional_descs)
            )
        self._form_layout.addStretch()

        # Pre-populate with the current applied config (missing keys → defaults)
        self._apply_config_to_widgets(self._applied_config)
        self._refresh_robot_type_choices(brand=self._brand, preserve_current=True)
        self._lock_brand_field()
        # Sync _applied_config with the actual widget state so that
        # has_unsaved_changes() returns False right after construction/rebuild.
        self._applied_config = self.get_current_config()

    def _lock_brand_field(self) -> None:
        """Brand is controlled by outer app flow; keep it read-only here."""
        widget = self._field_widgets.get("brand")
        if isinstance(widget, QComboBox):
            widget.setEnabled(False)
            widget.setToolTip("Brand is controlled by current robot selection.")

    def _refresh_robot_type_choices(self, brand: str, preserve_current: bool = True) -> None:
        """Refresh robot_type choices from schema for the active brand."""
        robot_widget = self._field_widgets.get("robot_type")
        if not isinstance(robot_widget, QComboBox):
            return

        current = robot_widget.currentText().strip()
        schema = schema_to_form_descriptors(brand)
        robot_desc = next((d for d in schema if d["field_id"] == "robot_type"), None)
        choices = [str(c) for c in (robot_desc.get("choices") if robot_desc else []) if c]
        default = str(robot_desc.get("default") or "") if robot_desc else ""

        robot_widget.blockSignals(True)
        robot_widget.clear()
        for c in choices:
            robot_widget.addItem(c)
        robot_widget.blockSignals(False)

        if preserve_current and current:
            self._set_choice_value(robot_widget, current, allow_unknown=True)
            return
        if default:
            self._set_choice_value(robot_widget, default, allow_unknown=True)
            return
        if robot_widget.count() > 0:
            robot_widget.setCurrentIndex(0)

    def _make_group(self, title: str, descriptors: list) -> QGroupBox:
        """Build a titled QGroupBox with a QFormLayout for *descriptors*."""
        box = QGroupBox(title)
        box.setObjectName("settingsGroup")
        form = QFormLayout(box)
        form.setContentsMargins(8, 12, 8, 8)
        form.setSpacing(6)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        for desc in descriptors:
            fid = desc["field_id"]
            widget = self._make_widget(desc)
            self._field_widgets[fid] = widget
            self._wire_field_change_signal(widget)

            # Inline error label (hidden until a failed Apply)
            err = QLabel("")
            err.setObjectName("settingsErrorLabel")
            err.setStyleSheet(
                "color: #ef4444; font-size: 11px; background: transparent;"
            )
            err.setVisible(False)
            self._error_labels[fid] = err

            # Stack widget + error vertically inside a cell widget
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            cell_layout.addWidget(widget)
            cell_layout.addWidget(err)

            # Row label with tooltip = description + hint
            row_label = QLabel(desc["label"])
            row_label.setObjectName("settingsFieldLabel")
            hint = desc.get("validation_hints", "")
            description = desc.get("description", "")
            tooltip_parts = [p for p in (description, hint) if p]
            if tooltip_parts:
                row_label.setToolTip("\n\n".join(tooltip_parts))

            form.addRow(row_label, cell)

        return box

    def _wire_field_change_signal(self, widget: QWidget) -> None:
        """Emit settings_changed on live field edits for capability refresh."""
        if isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(lambda _v: self._emit_settings_changed())
            return
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda _v: self._emit_settings_changed())
            return
        if isinstance(widget, QSpinBox):
            widget.valueChanged.connect(lambda _v: self._emit_settings_changed())
            return
        if isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(lambda _v: self._emit_settings_changed())
            return
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda _v: self._emit_settings_changed())
            return

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit(self.get_current_config())

    @staticmethod
    def _make_widget(desc: dict) -> QWidget:
        """Create the Qt widget appropriate for *desc[widget_type]*."""
        wt = desc["widget_type"]
        default = desc.get("default")
        choices = desc.get("choices") or []

        if wt == "choice":
            w = QComboBox()
            for c in choices:
                w.addItem(str(c))
            if default is not None and str(default) in [str(c) for c in choices]:
                w.setCurrentText(str(default))
            return w

        if wt == "bool":
            w = QCheckBox()
            w.setChecked(bool(default) if default is not None else False)
            return w

        if wt == "integer":
            w = QSpinBox()
            w.setRange(-2_147_483_648, 2_147_483_647)
            if default is not None:
                try:
                    w.setValue(int(default))
                except (TypeError, ValueError):
                    pass
            return w

        if wt == "float":
            w = QDoubleSpinBox()
            w.setRange(-1e12, 1e12)
            w.setDecimals(6)
            if default is not None:
                try:
                    w.setValue(float(default))
                except (TypeError, ValueError):
                    pass
            return w

        # "text" and "list" both use QLineEdit
        w = QLineEdit()
        hint = desc.get("validation_hints", "")
        if hint:
            w.setPlaceholderText(hint)
        if wt == "list" and isinstance(default, list):
            w.setText(", ".join(str(v) for v in default))
        elif isinstance(default, str):
            w.setText(default)
        return w

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_config(self) -> Dict[str, Any]:
        """Return a dict of {field_id: typed_value} for all form fields."""
        config: Dict[str, Any] = {}
        for desc in schema_to_form_descriptors(self._brand):
            fid = desc["field_id"]
            widget = self._field_widgets.get(fid)
            if widget is not None:
                config[fid] = self._read_widget(widget, desc)
        return config

    def has_unsaved_changes(self) -> bool:
        """Return True if any field differs from the last applied config."""
        current = self.get_current_config()
        for key, val in current.items():
            if val != self._applied_config.get(key):
                return True
        return False

    def scroll_to_field(self, field_id: str) -> None:
        """Scroll the settings form to make *field_id*'s widget visible.

        No-op if *field_id* is not in the current form.  Used by the
        capability inspector's ``focus_setting`` linkage.
        """
        widget = self._field_widgets.get(field_id)
        if widget is not None:
            self._scroll_area.ensureWidgetVisible(widget)

    def set_brand(
        self,
        brand: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Rebuild the form for *brand*, optionally pre-populating *config*.

        Args:
            brand:  New brand key.  Falls back to first known brand if unknown.
            config: Optional config dict to set as the new applied baseline.
        """
        self._brand = brand if brand in KNOWN_BRANDS else KNOWN_BRANDS[0]
        self._applied_config = dict(config) if config else {}
        self._title_label.setText(f"SDK Settings — {self._brand}")
        self._build_form()
        self._set_status("")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _read_widget(widget: QWidget, desc: dict) -> Any:
        """Extract a typed Python value from *widget* using *desc*."""
        wt = desc["widget_type"]
        if wt == "choice":
            return widget.currentText()       # type: ignore[attr-defined]
        if wt == "bool":
            return widget.isChecked()         # type: ignore[attr-defined]
        if wt == "integer":
            return widget.value()             # type: ignore[attr-defined]
        if wt == "float":
            return widget.value()             # type: ignore[attr-defined]
        if wt == "list":
            text = widget.text().strip()      # type: ignore[attr-defined]
            if not text:
                return []
            return [v.strip() for v in text.split(",") if v.strip()]
        # "text"
        return widget.text().strip()          # type: ignore[attr-defined]

    def _apply_config_to_widgets(self, config: Dict[str, Any]) -> None:
        """Write *config* values into the corresponding form widgets."""
        for desc in schema_to_form_descriptors(self._brand):
            fid = desc["field_id"]
            widget = self._field_widgets.get(fid)
            if widget is None or fid not in config:
                continue
            val = config[fid]
            wt = desc["widget_type"]
            if wt == "choice":
                self._set_choice_value(  # type: ignore[arg-type]
                    widget,
                    str(val),
                    allow_unknown=(fid == "robot_type"),
                )
            elif wt == "bool":
                widget.setChecked(bool(val))           # type: ignore[attr-defined]
            elif wt == "integer":
                try:
                    widget.setValue(int(val))          # type: ignore[attr-defined]
                except (TypeError, ValueError):
                    pass
            elif wt == "float":
                try:
                    widget.setValue(float(val))        # type: ignore[attr-defined]
                except (TypeError, ValueError):
                    pass
            elif wt == "list":
                if isinstance(val, list):
                    widget.setText(", ".join(str(v) for v in val))  # type: ignore[attr-defined]
                else:
                    widget.setText(str(val))           # type: ignore[attr-defined]
            else:  # "text"
                widget.setText(str(val) if val is not None else "")  # type: ignore[attr-defined]

    @staticmethod
    def _set_choice_value(widget: QComboBox, value: str, allow_unknown: bool = False) -> None:
        """Set QComboBox value, optionally preserving unknown legacy values."""
        idx = widget.findText(value)
        if idx >= 0:
            widget.setCurrentIndex(idx)
            return
        if allow_unknown and value:
            widget.addItem(value)
            widget.setCurrentText(value)

    def _clear_errors(self) -> None:
        for lbl in self._error_labels.values():
            lbl.setVisible(False)
            lbl.setText("")
        self._set_status("")

    def _show_field_error(self, field_id: str, message: str) -> None:
        lbl = self._error_labels.get(field_id)
        if lbl is not None:
            lbl.setText(message)
            lbl.setVisible(True)

    def _set_status(self, message: str, ok: bool = False) -> None:
        self._status_label.setText(message)
        if ok:
            self._status_label.setStyleSheet(
                "color: #22c55e; background: transparent; padding: 4px 12px;"
            )
        elif message:
            self._status_label.setStyleSheet(
                "color: #ef4444; background: transparent; padding: 4px 12px;"
            )
        else:
            self._status_label.setStyleSheet("background: transparent;")

    # ── Slot handlers ─────────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        """Collect → validate → persist or show errors."""
        self._clear_errors()
        config = self.get_current_config()
        result = validate_settings(self._brand, config)

        if result["status"] != "ok":
            missing: List[str] = result.get("missing", [])
            invalid: List[str] = result.get("invalid", [])

            for fid in missing:
                self._show_field_error(fid, "Required")
            for fid in invalid:
                self._show_field_error(fid, "Invalid value")

            parts: List[str] = []
            if missing:
                parts.append(f"Missing: {', '.join(missing)}")
            if invalid:
                parts.append(f"Invalid: {', '.join(invalid)}")
            self._set_status(" | ".join(parts) or "Validation failed")
            return

        self._applied_config = config
        self._set_status("Settings applied.", ok=True)
        self.settings_applied.emit(config)

    def _on_reset(self) -> None:
        """Restore form to last applied state and clear errors."""
        self._clear_errors()
        self._apply_config_to_widgets(self._applied_config)
        self.settings_reset.emit()
