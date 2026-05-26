# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""LoadingScreen -- the loading-stage page hosted inside MainWindow.

This is **not** a top-level window. It is a regular ``QWidget`` that lives as
page 0 of MainWindow's central ``QStackedWidget``; when startup tasks finish
MainWindow fades it out and swaps to its main-content page.

Visual layout (back-to-front, mirroring DEMO ``bin/pages/layout/ui.py``
``_startup_log_bg`` + ``_startup_loading_overlay`` block):

    1. **log wallpaper** -- a transparent, scroll-bar-less ``QTextEdit`` that
       receives every ``LogSignal.log_message`` emission and appends it with a
       per-level color. Acts as the dynamic loading wallpaper visible behind
       the dim overlay.
    2. **dim overlay** -- semi-transparent dark ``QWidget`` above the log so
       the wallpaper is readable but visually muted.
    3. **logo** -- centered ``LogoPulse`` on the front layer (icon_logo.svg
       sine-wave opacity pulse).

Implementation rules (intentionally matched to DEMO):

- **No SDK CmdLogWidget dependency.** The wallpaper is a plain ``QTextEdit``
  built locally; the LoadingScreen subscribes to ``get_log_signal()`` itself.
  Coupling to ``CmdLogWidget`` would drag in unrelated CMD-panel concerns and
  defeat the "minimum dependencies" goal of the loading stage.
- **Self-contained log routing.** ``LogSignal.log_message`` is connected with
  ``Qt.QueuedConnection`` so worker-thread emits (DepCheckTask, etc.) marshal
  cleanly back to the UI thread.
- **Owner-invoked teardown.** ``stop_logo()`` halts the pulse AND disconnects
  the log handler so the (subsequent) main page does not double-render every
  log line into both this widget and CmdLogWidget elsewhere.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QLabel, QStackedLayout, QTextEdit, QVBoxLayout, QWidget

from unitport_sdk import Assets, Config, LogSignal, i18n_bind

from .logo_pulse import LogoPulse


# Per-level foreground colors for the wallpaper. Mirrors DEMO's palette so the
# loading screen reads identically across both trees. Themable later via
# ``Config.get_color`` slots if a project-wide design pass needs it.
_LOG_COLORS: dict[str, str] = {
    "debug": "#8B8B8B",
    "info": "#FFFFFF",
    "success": "#00D26A",
    "warning": "#FFC107",
    "error": "#FF6B6B",
}


class LoadingScreen(QWidget):
    """3-layer loading content: log wallpaper + dim + pulsing logo."""

    _LOGO_SIZE = QSize(200, 200)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.init()
        self.build_ui()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self) -> None:
        self.setObjectName("loadingScreen")
        self._log_view: Optional[QTextEdit] = None
        self._dim: Optional[QWidget] = None
        self._logo: Optional[LogoPulse] = None
        self._install_label: Optional[QLabel] = None
        self._log_signal_connected: bool = False

    def build_ui(self) -> None:
        # ---- Stacked layout: all layers render together
        stack = QStackedLayout(self)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)

        # ---- Layer 1: log wallpaper (底层)
        self._log_view = QTextEdit(self)
        self._log_view.setObjectName("loadingLogView")
        self._log_view.setReadOnly(True)
        self._log_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._log_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._log_view.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._log_view.setFrameStyle(0)
        stack.addWidget(self._log_view)

        # ---- Layer 2: dim overlay（只画半透明颜色，不用 opacity）
        # WA_StyledBackground is required for a plain QWidget to honour the
        # rgba() stylesheet background; without it the dim layer renders
        # nothing and the log wallpaper shows through unmuted.
        self._dim = QWidget(self)
        self._dim.setObjectName("loadingDim")
        self._dim.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._dim.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._dim.setStyleSheet("""
        #loadingDim {
            background-color: rgba(0, 0, 0, 160);
        }
        """)
        stack.addWidget(self._dim)

        # ---- Layer 3: front (logo 层，完全不受遮罩影响)
        front = QWidget(self)
        front.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        front.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        front.setAutoFillBackground(False)

        front_layout = QVBoxLayout(front)
        front_layout.setContentsMargins(0, 0, 0, 0)
        front_layout.setSpacing(8)
        front_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_path = self._resolve_logo_path()
        if logo_path:
            self._logo = LogoPulse(logo_path, self._LOGO_SIZE, parent=front)
            front_layout.addWidget(self._logo, 0, Qt.AlignmentFlag.AlignCenter)

        # Installation message — hidden by default, shown via
        # ``set_install_message_visible(True)`` from UnitPortMain when the
        # current launch is a first-time install (setup_state.json absent).
        # Bound through i18n_bind so language picks made BEFORE the install
        # wizard opens immediately retranslate this label too.
        self._install_label = QLabel(front)
        self._install_label.setObjectName("loadingInstallLabel")
        self._install_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        i18n_bind(
            self._install_label,
            "setText",
            "loading.install.in_progress",
            default="Installation in progress",
        )
        self._install_label.hide()
        front_layout.addWidget(
            self._install_label, 0, Qt.AlignmentFlag.AlignCenter
        )

        stack.addWidget(front)

        # ---- 显式 z-order：log (底) < dim (中) < front (顶)
        # QStackedLayout.StackAll honours insertion order, but raise_() makes
        # the dim-over-text and logo-over-dim invariants explicit so a future
        # refactor that adds a sibling widget can't silently invert them.
        self._log_view.lower()
        self._dim.raise_()
        front.raise_()

        # ---- Log signal -> UI（线程安全）
        LogSignal.instance().log_message.connect(
            self._on_log_message,
            Qt.ConnectionType.QueuedConnection,
        )
        self._log_signal_connected = True

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_1", "#1E1E1E")
        dim_color = Config.get_color("bg_overlay", "rgba(0, 0, 0, 140)")
        font_size = Config.get_font_size("size_small", 12)
        install_size = Config.get_font_size("size_normal", 16)
        install_color = Config.get_color("main_t2", "#FFFFFF")
        # The log_view is intentionally transparent so the page background
        # color shows through; dim layer painted ON TOP of the log to mute it.
        self.setStyleSheet(
            f"QWidget#loadingScreen {{ background-color: {bg}; }}"
            f"QWidget#loadingDim {{ background-color: {dim_color}; }}"
            f"QTextEdit#loadingLogView {{ "
            f"background: transparent; border: none; "
            f"color: #FFFFFF; font-family: Consolas, 'Courier New', monospace; "
            f"font-size: {font_size}px; padding: 8px 12px; "
            f"}}"
            f"QLabel#loadingInstallLabel {{ "
            f"background: transparent; color: {install_color}; "
            f"font-size: {install_size}px; "
            f"}}"
        )

    def set_install_message_visible(self, visible: bool) -> None:
        """Show / hide the "Installation in progress" tag below the logo.

        Called from ``UnitPortMain`` on first-launch (visible=True) when
        the install wizard opens, and again on PostSetupTask completion
        (visible=False) once the heavy installs have finished.
        """
        if self._install_label is None:
            return
        self._install_label.setVisible(bool(visible))

    # ------------------------------------------------------------------
    # Animation control + log teardown (owner-invoked)
    # ------------------------------------------------------------------
    def stop_logo(self) -> None:
        """Halt the pulsing logo, detach its inner opacity effect, and
        disconnect the log subscription.

        Called by MainWindow before applying the parent-level fade-out
        effect on this page; nested ``QGraphicsOpacityEffect`` would
        otherwise trigger a Qt painter warning. Disconnecting the log
        handler here means subsequent log_* calls do not paint into a
        widget that is about to disappear.
        """
        if self._logo is not None:
            self._logo.stop()
        if self._log_signal_connected:
            try:
                LogSignal.instance().log_message.disconnect(self._on_log_message)
            except (RuntimeError, TypeError):
                # Already disconnected (e.g., second call) -- safe to ignore.
                pass
            self._log_signal_connected = False

    # ------------------------------------------------------------------
    # Log sink
    # ------------------------------------------------------------------
    def _on_log_message(
        self,
        text: str,
        log_type: str,
        wrap: bool,
        _typer: bool,
    ) -> None:
        """Append one log record to the wallpaper.

        Mirrors DEMO ``_on_startup_log`` (bin/pages/layout/ui.py): per-level
        color, newline-separated, auto-scroll to keep the latest visible.
        ``typer`` is ignored on the loading screen -- the typewriter effect
        only makes sense on a stable cmd panel, not a startup wallpaper that
        is about to fade out.
        """
        if self._log_view is None:
            return
        color = _LOG_COLORS.get(log_type, "#FFFFFF")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if wrap and cursor.position() > 0:
            cursor.insertText("\n", QTextCharFormat())
        cursor.insertText(str(text), fmt)
        self._log_view.setTextCursor(cursor)
        self._log_view.ensureCursorVisible()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_logo_path() -> Optional[str]:
        for rel in (
            # "icon/start_logo_w.svg",
            # "icon/start_logo.svg",
            "icon/logo_w.svg",
        ):
            p = Assets.find(rel)
            if p is not None:
                return str(p)
        return None
