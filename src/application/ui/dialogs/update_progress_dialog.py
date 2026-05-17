"""UpdateProgressDialog — modal progress display for ApplyUpdateTask.

Subscribes to ``AppSignals.update_apply_progress`` / ``update_apply_complete``.
On terminal success, swaps content to the restart prompt + a single
``Close UnitPort`` button that calls ``QApplication.quit()``.

The dialog is intentionally NOT cancellable — once a channel.apply()
starts, interrupting it leaves the project tree half-merged. The
``Close`` button is only enabled after a terminal result.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr

from application.service.signals import get_app_signals


class UpdateProgressDialog(QDialog):
    """Modal progress UI for the in-app updater."""

    def __init__(
        self,
        target_version: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._target_version = target_version
        self._terminal = False

        self.setModal(True)
        self.setMinimumWidth(460)
        i18n_bind(
            self, "setWindowTitle",
            "updater.dialog.title_progress", "Updating UnitPort",
        )
        # Disable the close button while the apply is running. After the
        # terminal signal arrives we re-enable + wire the Close button.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        self._build_ui()
        self._wire_signals()

    # ----- UI -----
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(12)

        self._heading = QLabel(self)
        self._heading.setText(
            f"v{self._target_version}"
        )
        self._heading.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('highlight')}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        root.addWidget(self._heading)

        self._status = QLabel(
            tr("updater.progress.connecting", "Connecting"),
            self,
        )
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(self._status)

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(
            f"QProgressBar {{ "
            f"background-color: {Config.get_color('bg_3')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; }} "
            f"QProgressBar::chunk {{ "
            f"background-color: {Config.get_color('highlight')}; "
            f"border-radius: 3px; }}"
        )
        root.addWidget(self._bar)

        # Footer — Close button. Disabled until terminal result.
        footer = QHBoxLayout()
        footer.addStretch(1)
        self._close_btn = QPushButton(self)
        i18n_bind(self._close_btn, "setText", "updater.btn.close", "Close UnitPort")
        self._close_btn.setEnabled(False)
        self._close_btn.setMinimumWidth(140)
        self._close_btn.setStyleSheet(self._button_qss())
        self._close_btn.clicked.connect(self._on_close_clicked)
        footer.addWidget(self._close_btn)
        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

    def _wire_signals(self) -> None:
        sig = get_app_signals()
        sig.update_apply_progress.connect(self._on_progress)
        sig.update_apply_complete.connect(self._on_complete)

    # ----- handlers -----
    def _on_progress(self, fraction: float, label: str) -> None:
        try:
            f = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            return
        self._bar.setValue(int(f * 1000))
        if label:
            self._status.setText(str(label))

    def _on_complete(self, ok: bool, message: str) -> None:
        self._terminal = True
        if ok:
            self._bar.setValue(1000)
            i18n_bind(
                self, "setWindowTitle",
                "updater.dialog.title_complete", "Update applied",
            )
            self._status.setText(
                tr(
                    "updater.label.restart_prompt",
                    "Update to v{version} applied. UnitPort will close now — "
                    "please relaunch via start.bat or start.sh.",
                ).replace("{version}", self._target_version)
            )
        else:
            self._bar.setValue(0)
            self._status.setText(
                str(message or tr("updater.btn.close", "Close UnitPort"))
            )
            self._status.setStyleSheet(
                f"QLabel {{ color: {Config.get_color('danger_zone')}; "
                f"font-size: {Config.get_font_size('size_small')}pt; }}"
            )
        self._close_btn.setEnabled(True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.show()  # re-show so the WindowFlag change takes effect

    def _on_close_clicked(self) -> None:
        # On a successful apply we quit the app so the user can relaunch
        # cleanly on the new code. On failure, we just close the dialog
        # and let the user keep using the old version.
        QApplication.quit() if self._was_successful() else self.accept()

    def _was_successful(self) -> bool:
        return self._bar.value() == 1000

    @staticmethod
    def _button_qss() -> str:
        return (
            f"QPushButton {{ "
            f"background-color: {Config.get_color('highlight')}; "
            f"color: {Config.get_color('bg_1')}; "
            f"border: none; border-radius: 4px; "
            f"padding: 7px 16px; font-weight: bold; "
            f"font-size: {Config.get_font_size('size_small')}pt; }} "
            f"QPushButton:hover {{ "
            f"background-color: {Config.get_color('highlight_checked')}; }} "
            f"QPushButton:disabled {{ "
            f"background-color: {Config.get_color('btn_1')}; "
            f"color: {Config.get_color('sub_t2')}; }}"
        )


__all__ = ["UpdateProgressDialog"]
