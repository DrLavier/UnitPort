# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""LanguagePickerDialog — first-launch language chooser.

Shown by ``UnitPortMain.run`` BEFORE :class:`InstallConfigWizard` opens,
so the install wizard (and the "Installation in progress" tag on the
LoadingScreen) can render in the user's preferred language from the
very first frame.

The picker enumerates every language sub-directory under
``localisation/`` via :meth:`I18n.available_languages`. Each language is
shown by its native display name so the dialog is readable regardless
of which language is currently loaded. Clicking a language calls
:meth:`I18n.set_language` (which writes through Config and emits
``language_changed`` so every ``i18n_bind``-attached widget retranslates)
and persists the choice as the device-level locale preference via
``application.service.user_workspace.write_machine_locale``.
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

from unitport_sdk import Config, I18n, i18n_bind, log_info, log_warning, tr


# Display labels for known language codes. Unknown codes fall back to the
# raw directory name so newly-added localisation directories surface
# without a code change. Kept in sync with the equivalent map inside
# ``application.ui.sidebar`` (both lookups are read-only — duplicating the
# tiny constant beats coupling a startup-time dialog to a heavy UI
# module).
_LANG_DISPLAY = {
    "EN": "English",
    "FR": "Français",
    "ZH_S": "简体中文",
    "ZH_T": "繁體中文",
    "IT": "Italiano",
    "ES": "Español",
    "DE": "Deutsch",
    "RU": "Русский",
    "PT": "Português",
    "AR": "العربية",
    "JA": "日本語",
    "KO": "한국어",
}


class LanguagePickerDialog(QDialog):
    """Frameless modal: pick the UI language for first install."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("languagePickerDialog")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        # Window title needs a tr() at construction; refreshed when the
        # user picks (window title doesn't need live i18n_bind because the
        # dialog closes immediately on pick).
        self.setWindowTitle(tr("setup.language.title", "Select Language"))
        self.setFixedSize(360, 360)
        self._init_ui()
        self.setStyleSheet(self._stylesheet())

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        title = QLabel(self)
        title.setObjectName("langPickerTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        i18n_bind(title, "setText", "setup.language.title", default="Select Language")
        root.addWidget(title)

        hint = QLabel(self)
        hint.setObjectName("langPickerHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        i18n_bind(
            hint,
            "setText",
            "setup.language.hint",
            default=(
                "Choose your preferred language for UnitPort. "
                "You can change it later from the sidebar."
            ),
        )
        root.addWidget(hint)
        root.addSpacing(8)

        codes = list(I18n.available_languages())
        if not codes:
            codes = [I18n.get_language() or "EN"]
        current = I18n.get_language()

        for code in codes:
            label = _LANG_DISPLAY.get(code, code)
            btn = QPushButton(label, self)
            btn.setObjectName(
                "langPickerBtnActive" if code == current else "langPickerBtn"
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda _checked, c=code: self._on_picked(c))
            root.addWidget(btn)

        root.addStretch(1)

    def _on_picked(self, code: str) -> None:
        try:
            I18n.set_language(code)
        except Exception as exc:                                      # noqa: BLE001
            log_warning(f"[langpicker] I18n.set_language failed: {exc}")
        try:
            from application.service.user_workspace import write_machine_locale
            write_machine_locale(code)
        except Exception as exc:                                      # noqa: BLE001
            log_warning(f"[langpicker] write_machine_locale failed: {exc}")
        log_info(f"[langpicker] user selected language {code!r}")
        self.accept()

    # ---- theming -----------------------------------------------------
    def _stylesheet(self) -> str:
        bg = Config.get_color("bg_1")
        text = Config.get_color("main_t2")
        muted = Config.get_color("sub_t2")
        btn_bg = Config.get_color("btn_1")
        btn_hover = Config.get_color("hover_1")
        border = Config.get_color("border_1")
        accent = Config.get_color("highlight")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")
        return f"""
            QDialog#languagePickerDialog {{
                background: {bg};
            }}
            QLabel#langPickerTitle {{
                color: {text};
                font-size: {size_normal}px;
                font-weight: 600;
            }}
            QLabel#langPickerHint {{
                color: {muted};
                font-size: {size_small}px;
            }}
            QPushButton#langPickerBtn,
            QPushButton#langPickerBtnActive {{
                background: {btn_bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                font-size: {size_normal}px;
            }}
            QPushButton#langPickerBtnActive {{
                border-color: {accent};
            }}
            QPushButton#langPickerBtn:hover,
            QPushButton#langPickerBtnActive:hover {{
                background: {btn_hover};
                border-color: {accent};
            }}
        """


__all__ = ["LanguagePickerDialog"]
