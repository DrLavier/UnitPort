# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""VariantCreateDialog — create a new user variant under (kind, key).

Triggered from the per-key fold-out in the sidebar Scripts panels
when the user clicks the synthetic ``+ new variant`` row (item id
``registry:<kind>:<key>:__new__``). The dialog gathers:

* **variant name** — Python-identifier-shaped, unique within ``(kind, key)``
* **families**     — zero or more of the known
                     :data:`KNOWN_FAMILIES`; empty list = "any robot"
* **based on**     — ``Preset`` (factory) or one of the existing
                     variants; the resolver copies that source as the
                     starting point so the new ``.py`` is editable
                     immediately
* **description**  — one-line summary surfaced in the sidebar tooltip

On accept, calls
:func:`application.service.scripts.resolver.save_variant` and emits
:attr:`accepted_variant(kind, key, name)` so the host (sidebar +
mission control) can refresh and switch the editor to the new variant.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    CheckboxItem,
    Config,
    LaviComboBox,
    LaviLineEdit,
    log_warning,
    setButton,
    setComboBox,
    setLineEdit,
    tr,
)

from application.service.scripts import resolver as _resolver


# Known families used to populate the checkbox grid. Sourced from the
# canonical family vocabulary in ``scripts.task_module.ALL_FAMILIES``
# plus ``humanoid`` which appears in ``registers/data/robots_canonical.json``
# but not in ``ALL_FAMILIES`` (kept distinct from ``biped`` for clarity).
# Order is preserved in the UI.
KNOWN_FAMILIES: List[str] = [
    "quadruped",
    "humanoid",
    "biped",
    "wheeled",
    "manipulator",
    "generic_locomotion",
]


class VariantCreateDialog(QDialog):
    """Modal dialog to create a single variant under one ``(kind, key)``."""

    accepted_variant = pyqtSignal(str, str, str)   # (kind, key, variant_name)

    def __init__(
        self,
        *,
        kind: str,
        key: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._key = key

        self.setObjectName("variantCreateDialog")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setWindowTitle(
            tr("scripts.variant.create.title", "Create Script Variant")
        )
        self.setFixedWidth(440)

        self._name_edit: Optional[LaviLineEdit] = None
        self._desc_edit: Optional[LaviLineEdit] = None
        self._base_combo: Optional[LaviComboBox] = None
        self._family_checks: dict[str, CheckboxItem] = {}
        self._status: Optional[QLabel] = None

        self._init_ui()
        self.setStyleSheet(self._stylesheet())

    # ─── UI ────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        header = QLabel(self)
        header.setObjectName("variantDialogHeader")
        header.setText(f"{self._kind} / {self._key}")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        # Variant name
        grid.addWidget(self._make_field_label("Name"), 0, 0, Qt.AlignmentFlag.AlignRight)
        self._name_edit = setLineEdit(
            placeholder="e.g. aggressive_quadruped", parent=self
        )
        grid.addWidget(self._name_edit, 0, 1)

        # Description
        grid.addWidget(
            self._make_field_label("Description"), 1, 0, Qt.AlignmentFlag.AlignRight
        )
        self._desc_edit = setLineEdit(
            placeholder="optional one-liner", parent=self
        )
        grid.addWidget(self._desc_edit, 1, 1)

        # Based on
        grid.addWidget(
            self._make_field_label("Based on"), 2, 0, Qt.AlignmentFlag.AlignRight
        )
        items: List[tuple] = [("scripts.variant.base.preset", "Preset")]
        for v in _resolver.list_variants(self._kind, self._key):
            items.append((f"variant.{v.name}", v.name))
        self._base_combo = setComboBox(items, height=28, parent=self)
        # Anchor i18n=False so the variant names render literally (they
        # were never registered in the localisation catalogue).
        grid.addWidget(self._base_combo, 2, 1)

        root.addLayout(grid)

        # Families — checkbox grid (3 columns)
        fam_label = self._make_field_label("Families")
        root.addWidget(fam_label)
        fam_grid = QGridLayout()
        fam_grid.setContentsMargins(0, 0, 0, 0)
        fam_grid.setHorizontalSpacing(12)
        fam_grid.setVerticalSpacing(4)
        cols = 3
        for idx, fam in enumerate(KNOWN_FAMILIES):
            box = CheckboxItem(
                id=f"scripts.variant.family.{fam}",
                default=fam,
                checked=False,
                parent=self,
            )
            self._family_checks[fam] = box
            fam_grid.addWidget(box, idx // cols, idx % cols)
        root.addLayout(fam_grid)

        hint = QLabel(self)
        hint.setObjectName("variantDialogHint")
        hint.setText(
            "Leave Families empty for an any-robot variant. "
            "Hard family filtering happens in the Rewards Editor."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Status line (errors / progress)
        self._status = QLabel(self)
        self._status.setObjectName("variantDialogStatus")
        self._status.setText("")
        root.addWidget(self._status)

        # OK / Cancel — matches the editor panel pattern (kind="border",
        # spec="save"/"danger"). SettingConfirm's icon_yes / icon_no
        # assets aren't reachable via Assets.find_icon (SDK passes the
        # filename with extension; the loader expects just the stem when
        # the file lives under ``icon/``). Plain bordered buttons sidestep
        # the issue and keep visual parity with the rest of the app.
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.addStretch(1)
        btn_ok = setButton(
            "scripts.variant.create.ok", 80, 28,
            kind="border", spec="save", default="OK",
        )
        btn_cancel = setButton(
            "scripts.variant.create.cancel", 80, 28,
            kind="border", spec="danger", default="Cancel",
        )
        btn_ok.clicked.connect(self._on_accept)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        root.addLayout(btn_row)

    def _make_field_label(self, text: str) -> QLabel:
        lbl = QLabel(text, self)
        lbl.setObjectName("variantDialogFieldLabel")
        return lbl

    # ─── Behaviour ────────────────────────────────────────────────

    def _on_accept(self) -> None:
        if self._name_edit is None:
            return
        name = (self._name_edit.text() or "").strip()
        if not name:
            self._set_status("name is required", error=True)
            return

        # Resolver enforces the identifier shape; pre-check here so the
        # user sees the rule before the round-trip.
        if not name.isidentifier() or name == "preset":
            self._set_status(
                "name must be a Python identifier and not 'preset'",
                error=True,
            )
            return

        existing = {
            v.name for v in _resolver.list_variants(self._kind, self._key)
        }
        if name in existing:
            self._set_status(f"variant {name!r} already exists", error=True)
            return

        families = sorted(
            fam for fam, box in self._family_checks.items() if box.isChecked()
        )
        description = (self._desc_edit.text() or "").strip() if self._desc_edit else ""
        based_on = self._based_on_value()

        # Pull the base source for the initial commit.
        if based_on == "preset":
            base = _resolver.resolve(self._kind, self._key, variant=None)
        else:
            base = _resolver.resolve(self._kind, self._key, variant=based_on)
        if base is None or not base.source.strip():
            self._set_status(
                f"base source missing for {based_on!r}; cannot clone",
                error=True,
            )
            return

        try:
            _resolver.save_variant(
                self._kind,
                self._key,
                name,
                base.source,
                families=families,
                description=description,
                based_on=based_on,
            )
        except ValueError as exc:
            self._set_status(f"create rejected: {exc}", error=True)
            return
        except OSError as exc:
            self._set_status(f"write failed: {exc}", error=True)
            log_warning(f"[variant_create] write failed: {exc!r}")
            return

        self.accepted_variant.emit(self._kind, self._key, name)
        self.accept()

    def _based_on_value(self) -> str:
        """Return the variant name (or ``"preset"``) selected in the combo.

        :class:`LaviComboBox` stashes each item's i18n key under
        ``UserRole`` and exposes it via :meth:`currentKey`. Variant names
        are encoded as ``"variant.<name>"``; the preset slot uses
        ``"scripts.variant.base.preset"``.
        """
        if self._base_combo is None:
            return "preset"
        ikey = self._base_combo.currentKey() or ""
        if ikey.startswith("variant."):
            return ikey.split(".", 1)[1]
        return "preset"

    def _set_status(self, text: str, *, error: bool) -> None:
        if self._status is None:
            return
        self._status.setText(text)
        slot = "danger_zone" if error else "sub_t2"
        size = Config.get_font_size("size_small")
        self._status.setStyleSheet(
            f"color: {Config.get_color(slot)}; "
            f"font-size: {size}px; background: transparent;"
        )

    # ─── Theming ──────────────────────────────────────────────────

    def _stylesheet(self) -> str:
        bg = Config.get_color("bg_1")
        text = Config.get_color("main_t2")
        muted = Config.get_color("sub_t2")
        accent = Config.get_color("highlight")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")
        return f"""
            QDialog#variantCreateDialog {{
                background: {bg};
            }}
            QLabel#variantDialogHeader {{
                color: {accent};
                font-size: {size_normal}px;
                font-weight: 600;
            }}
            QLabel#variantDialogFieldLabel {{
                color: {text};
                font-size: {size_small}px;
            }}
            QLabel#variantDialogHint {{
                color: {muted};
                font-size: {size_small}px;
            }}
        """


__all__ = ["VariantCreateDialog", "KNOWN_FAMILIES"]
