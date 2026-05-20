"""IsaacInstallProgressDialog — modal progress display for IsaacLabInstallTask.

Subscribes to the three signals emitted by
:class:`application.service.installers.isaac_lab_installer.IsaacLabInstaller`:

* ``isaac_install_phase(phase, label)`` — stage transitions
  (``preflight`` / ``download`` / ``extract`` / ``clone`` / ``install`` /
  ``register`` / ``fallback_external`` / ``done`` / ``error``). We
  update the heading + status label here.
* ``isaac_install_progress(fraction, label)`` — pumped throughout each
  stage. Drives the progress bar (mapped to 0..1000) and the per-stage
  caption.
* ``isaac_install_complete(ok, message)`` — terminal. Success: bar to
  100 %, swap heading to "Installed", enable Close (returns the dialog
  to Accepted). Failure: switch to error view with three buttons —
  Open log / Retry / Switch to Locate mode.

Patterned after :class:`UpdateProgressDialog` but intentionally NOT a
subclass — the update dialog is hard-wired to ``QApplication.quit()`` on
success (since an update is "swap codebase, restart"), which is wrong
for an Isaac install (we want to stay running and immediately surface
the now-available backend in the sidebar).

The "Retry" and "Switch to Locate mode" buttons fire signals only — the
caller (MainWindow or the wizard) is responsible for re-submitting the
Task / opening the locate dialog. This keeps the dialog free of
back-references into orchestration code.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, Paths, i18n_bind, log_info, tr

from application.service.signals import get_app_signals


class IsaacInstallProgressDialog(QDialog):
    """Modal progress UI for IsaacLabInstallTask.

    Caller (MainWindow on receiving ``isaac_install_phase("preflight",...)``)
    creates + ``.show()``s the dialog. The dialog subscribes to the
    install signals in its own constructor — the caller does NOT need to
    forward anything.
    """

    # Emitted when the user clicks "Retry" on the error screen. Caller
    # is expected to re-dispatch ``IsaacLabInstallTask`` with the same
    # install_dir / eula_ids; the partial-marker on disk makes the
    # retry skip already-completed stages.
    retry_requested = pyqtSignal()
    # Emitted when the user clicks "Switch to Locate mode". Caller is
    # expected to open the wizard back-end page with the "Locate
    # existing installation" radio pre-selected.
    locate_mode_requested = pyqtSignal()

    def __init__(
        self,
        *,
        install_dir: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._install_dir = str(install_dir or "")
        self._terminal_ok: Optional[bool] = None
        self._last_error_message: str = ""

        self.setModal(True)
        self.setMinimumWidth(520)
        self.setMinimumHeight(260)
        i18n_bind(
            self, "setWindowTitle",
            "isaac_install.dialog.title", "Installing Isaac Lab",
        )
        # Keep the user from closing mid-install; we re-enable the close
        # button when isaac_install_complete arrives.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(12)

        # Heading — replaced on completion.
        self._heading = QLabel(self)
        i18n_bind(
            self._heading, "setText",
            "isaac_install.heading.running", "Installing Isaac Lab",
        )
        self._heading.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('highlight')}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        root.addWidget(self._heading)

        # Phase line: e.g. "Stage: download / clone / install".
        self._phase_label = QLabel(self)
        self._phase_label.setText(
            tr("isaac_install.phase.preflight", "Preparing...")
        )
        self._phase_label.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('sub_t2')}; "
            f"font-size: {Config.get_font_size('size_mini')}pt; }}"
        )
        root.addWidget(self._phase_label)

        # Progress bar (running view).
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(
            f"QProgressBar {{ "
            f"background-color: {Config.get_color('bg_3')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; "
            f"}} "
            f"QProgressBar::chunk {{ "
            f"background-color: {Config.get_color('highlight')}; "
            f"border-radius: 3px; "
            f"}}"
        )
        root.addWidget(self._bar)

        # Status line (under the bar) — the chatty per-tick label.
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._status.setText(
            tr("isaac_install.status.idle", "Waiting for installer to start...")
        )
        self._status.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(self._status)

        # Hidden until error: stderr / message viewer.
        self._error_view = QTextEdit(self)
        self._error_view.setReadOnly(True)
        self._error_view.setVisible(False)
        self._error_view.setStyleSheet(
            f"QTextEdit {{ "
            f"background-color: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('danger_zone')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 8px; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}}"
        )
        root.addWidget(self._error_view, 1)

        # Footer — varies between running / success / error states.
        footer = QHBoxLayout()
        footer.setSpacing(8)

        self._open_log_btn = QPushButton(self)
        i18n_bind(self._open_log_btn, "setText", "isaac_install.btn.open_log", "Open log folder")
        self._open_log_btn.setStyleSheet(self._button_qss("muted"))
        self._open_log_btn.clicked.connect(self._on_open_log_clicked)
        self._open_log_btn.setVisible(False)
        footer.addWidget(self._open_log_btn)

        self._locate_btn = QPushButton(self)
        i18n_bind(
            self._locate_btn, "setText",
            "isaac_install.btn.switch_locate", "Switch to Locate mode",
        )
        self._locate_btn.setStyleSheet(self._button_qss("muted"))
        self._locate_btn.clicked.connect(self._on_locate_clicked)
        self._locate_btn.setVisible(False)
        footer.addWidget(self._locate_btn)

        footer.addStretch(1)

        self._retry_btn = QPushButton(self)
        i18n_bind(self._retry_btn, "setText", "isaac_install.btn.retry", "Retry")
        self._retry_btn.setStyleSheet(self._button_qss("primary"))
        self._retry_btn.clicked.connect(self._on_retry_clicked)
        self._retry_btn.setVisible(False)
        footer.addWidget(self._retry_btn)

        self._close_btn = QPushButton(self)
        i18n_bind(self._close_btn, "setText", "isaac_install.btn.close", "Close")
        self._close_btn.setStyleSheet(self._button_qss("primary"))
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        footer.addWidget(self._close_btn)

        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

    def _wire_signals(self) -> None:
        sig = get_app_signals()
        sig.isaac_install_phase.connect(self._on_phase)
        sig.isaac_install_progress.connect(self._on_progress)
        sig.isaac_install_complete.connect(self._on_complete)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _on_phase(self, phase: str, label: str) -> None:
        if phase == "fallback_external":
            self._phase_label.setText(
                tr(
                    "isaac_install.phase.fallback_external",
                    "Switching to external venv (pip conflicts detected)",
                )
            )
            self._phase_label.setStyleSheet(
                f"QLabel {{ color: {Config.get_color('log_warning')}; "
                f"font-size: {Config.get_font_size('size_mini')}pt; }}"
            )
            return
        if phase == "error":
            # The complete signal will arrive momentarily; the phase
            # signal carries the human-readable error for surface here.
            self._last_error_message = label or self._last_error_message
            return
        # Use the i18n key per known phase; fall back to the label.
        text = tr(f"isaac_install.phase.{phase}", label or phase)
        self._phase_label.setText(text)

    def _on_progress(self, fraction: float, label: str) -> None:
        try:
            f = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            return
        self._bar.setValue(int(f * 1000))
        if label:
            self._status.setText(str(label))

    def _on_complete(self, ok: bool, message: str) -> None:
        self._terminal_ok = bool(ok)
        if ok:
            self._show_success_view()
        else:
            self._show_error_view(message or self._last_error_message)
        # Re-allow the user to close the window via the title-bar X.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.show()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _show_success_view(self) -> None:
        self._bar.setValue(1000)
        # Repaint heading + bar + close button in safe_zone so the user
        # gets an unambiguous "success" visual cue — the running-state
        # uses ``highlight`` (orange / accent), reserving safe_zone
        # (green) for the terminal-ok state.
        safe = Config.get_color("safe_zone")
        i18n_bind(
            self._heading, "setText",
            "isaac_install.heading.success", "Isaac Lab installed",
        )
        self._heading.setStyleSheet(
            f"QLabel {{ color: {safe}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        # Re-skin the progress bar in safe_zone (replaces the orange
        # ``highlight`` chunk used during the running view).
        self._bar.setStyleSheet(
            f"QProgressBar {{ "
            f"background-color: {Config.get_color('bg_3')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; "
            f"}} "
            f"QProgressBar::chunk {{ "
            f"background-color: {safe}; "
            f"border-radius: 3px; "
            f"}}"
        )
        self._phase_label.setText("")
        details = self._install_dir or tr(
            "isaac_install.status.success_generic", "Installation complete"
        )
        self._status.setText(
            tr(
                "isaac_install.status.success",
                "Successfully installed at: {path}",
            ).replace("{path}", details)
        )
        self._close_btn.setStyleSheet(self._success_button_qss())
        self._close_btn.setEnabled(True)
        self._retry_btn.setVisible(False)
        self._locate_btn.setVisible(False)
        self._open_log_btn.setVisible(False)

    def _show_error_view(self, message: str) -> None:
        i18n_bind(
            self._heading, "setText",
            "isaac_install.heading.error", "Isaac Lab installation failed",
        )
        self._heading.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('danger_zone')}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        self._phase_label.setText("")
        self._status.setText(
            tr(
                "isaac_install.status.error_hint",
                "See log for full details. You can retry, or switch to locating "
                "an existing Isaac Lab installation.",
            )
        )

        # Surface the message itself in the (newly-visible) error view.
        self._error_view.setVisible(True)
        self._error_view.setPlainText(message or "(no error message provided)")

        # Hide the progress bar — it's misleading after a failure.
        self._bar.setVisible(False)

        self._open_log_btn.setVisible(True)
        self._retry_btn.setVisible(True)
        self._locate_btn.setVisible(True)
        self._close_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _on_retry_clicked(self) -> None:
        log_info("[isaac-install] retry requested from progress dialog")
        self.retry_requested.emit()
        self.accept()

    def _on_locate_clicked(self) -> None:
        log_info("[isaac-install] user chose Switch to Locate mode")
        self.locate_mode_requested.emit()
        self.accept()

    def _on_open_log_clicked(self) -> None:
        # Open the LOGS_DIR in the OS file browser. We rely on
        # ``QDesktopServices.openUrl`` rather than spawning a shell —
        # works cross-platform.
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        path = Paths.LOGS_DIR
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------
    @staticmethod
    def _success_button_qss() -> str:
        """Primary button styled with safe_zone — applied to Close after a
        successful install so the success state reads at a glance."""
        safe = Config.get_color("safe_zone")
        return (
            f"QPushButton {{ "
            f"background-color: {safe}; "
            f"color: {Config.get_color('bg_1')}; "
            f"border: none; border-radius: 4px; "
            f"padding: 7px 16px; font-weight: bold; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}} "
            f"QPushButton:hover {{ "
            f"background-color: {safe}; "
            f"}}"
        )

    @staticmethod
    def _button_qss(kind: str) -> str:
        if kind == "primary":
            return (
                f"QPushButton {{ "
                f"background-color: {Config.get_color('highlight')}; "
                f"color: {Config.get_color('bg_1')}; "
                f"border: none; border-radius: 4px; "
                f"padding: 7px 16px; font-weight: bold; "
                f"font-size: {Config.get_font_size('size_small')}pt; "
                f"}} "
                f"QPushButton:hover {{ "
                f"background-color: {Config.get_color('highlight_checked')}; "
                f"}} "
                f"QPushButton:disabled {{ "
                f"background-color: {Config.get_color('btn_1')}; "
                f"color: {Config.get_color('sub_t2')}; "
                f"}}"
            )
        return (
            f"QPushButton {{ "
            f"background-color: {Config.get_color('btn_1')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 6px 14px; "
            f"font-size: {Config.get_font_size('size_small')}pt; "
            f"}} "
            f"QPushButton:hover {{ "
            f"background-color: {Config.get_color('hover_1')}; "
            f"}}"
        )


__all__ = ["IsaacInstallProgressDialog"]
