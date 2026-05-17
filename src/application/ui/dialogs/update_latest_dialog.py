"""UpdateLatestDialog — shown when the Update button is clicked but no
newer release exists on GitHub. Pure informational + a single OK.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr


class UpdateLatestDialog(QDialog):
    """Confirm the user is already on the latest version."""

    def __init__(
        self,
        current_version: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._current_version = current_version
        self.setModal(True)
        self.setMinimumWidth(380)
        i18n_bind(
            self, "setWindowTitle",
            "updater.dialog.title_latest", "UnitPort is up to date",
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(14)

        body = QLabel(
            tr(
                "updater.label.latest_body",
                "You are on UnitPort v{current}. No newer release was found on GitHub.",
            ).replace("{current}", self._current_version),
            self,
        )
        body.setWordWrap(True)
        body.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(body)

        footer = QHBoxLayout()
        footer.addStretch(1)
        ok_btn = QPushButton(self)
        i18n_bind(ok_btn, "setText", "updater.btn.ok", "OK")
        ok_btn.setMinimumWidth(96)
        ok_btn.setStyleSheet(self._button_qss())
        ok_btn.clicked.connect(self.accept)
        footer.addWidget(ok_btn)
        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

    @staticmethod
    def _button_qss() -> str:
        return (
            f"QPushButton {{ "
            f"background-color: {Config.get_color('btn_1')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 6px 14px; "
            f"font-size: {Config.get_font_size('size_small')}pt; }} "
            f"QPushButton:hover {{ background-color: {Config.get_color('hover_1')}; }}"
        )


__all__ = ["UpdateLatestDialog"]
