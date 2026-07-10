# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ClassifyAssetDialog — manually classify an asset the downloader couldn't.

When a downloaded resource cannot be auto-detected, the user picks its kind
(Motion / Policy / Model) and target robot family here. The choice is
recorded as user state under USER_CONFIG_DIR via the assets-browser seam.

Frontend round: the choice is *recorded but not yet applied* — the backend
round consumes the recorded classification to actually route the asset into
the right registry. The dialog says so on accept (no §8 illusion of success).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, setButton, setComboBox, tr

from application.service.assets_browser import get_asset_browser_provider


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


def _mini() -> int:
    return int(Config.get_font_size("size_mini"))


class ClassifyAssetDialog(QDialog):
    """Modal form to record a manual asset classification."""

    def __init__(self, package_id: str, *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._package_id = package_id
        self._provider = get_asset_browser_provider()

        self.setObjectName("classifyAssetDialog")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setWindowTitle(tr("classify.title", "Classify asset"))
        self.setFixedWidth(460)

        self._kind_values: List[str] = ["motion", "policy", "model"]
        self._family_values: List[str] = [""]

        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        hint = QLabel(self)
        i18n_bind(
            hint, "setText", "classify.hint",
            "Pick the asset kind and target robot family. Your choice is "
            "recorded for the backend to apply.",
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        root.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(self._label("classify.kind", "Asset kind"),
                       0, 0, Qt.AlignmentFlag.AlignRight)
        kind_items = [
            tr("resources.kind.motion", "Motion"),
            tr("resources.kind.policy", "Policy"),
            tr("resources.kind.model", "Model"),
        ]
        self._kind_combo = setComboBox(kind_items, i18n=False, parent=self)
        grid.addWidget(self._kind_combo, 0, 1)

        grid.addWidget(self._label("classify.family", "Robot family"),
                       1, 0, Qt.AlignmentFlag.AlignRight)
        fam_items = [tr("classify.family_any", "(any)")]
        try:
            from scripts.training_motion.library import CATEGORIES
            for cat in CATEGORIES:
                self._family_values.append(cat)
                fam_items.append(cat)
        except Exception:
            pass  # WHY KEPT (§8(a)): motion library optional in some envs;
            # the picker still offers "(any)" so classification isn't blocked.
        self._family_combo = setComboBox(fam_items, i18n=False, parent=self)
        grid.addWidget(self._family_combo, 1, 1)

        root.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = setButton(
            "classify.ok", 110, 28, kind="normal", spec="save",
            default=tr("classify.ok", "Record"),
        )
        ok_btn.clicked.connect(self._on_submit)
        btn_row.addWidget(ok_btn)
        cancel_btn = setButton(
            "classify.cancel", 96, 28, kind="border", spec="none",
            default=tr("classify.cancel", "Cancel"),
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    def _label(self, key: str, default: str) -> QLabel:
        lbl = QLabel(self)
        i18n_bind(lbl, "setText", key, default)
        lbl.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        return lbl

    def _on_submit(self) -> None:
        kind = self._value(self._kind_combo, self._kind_values)
        family = self._value(self._family_combo, self._family_values)
        if not kind:
            QMessageBox.warning(
                self, self.windowTitle(),
                tr("classify.err_kind", "Pick an asset kind."),
            )
            return
        try:
            self._provider.classify_package(
                self._package_id, asset_kind=kind, robot_family=family
            )
        except Exception as e:  # noqa: BLE001 — surface any write failure
            QMessageBox.critical(self, self.windowTitle(), f"{type(e).__name__}: {e}")
            return
        QMessageBox.information(
            self, self.windowTitle(),
            tr(
                "classify.recorded",
                "Classification recorded. It is not applied yet — the backend "
                "round will route the asset into the right registry.",
            ),
        )
        self.accept()

    @staticmethod
    def _value(combo, values: List[str]) -> str:
        idx = combo.currentIndex()
        if 0 <= idx < len(values):
            return values[idx]
        return ""


__all__ = ["ClassifyAssetDialog"]
