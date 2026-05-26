# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CrossUserOpenChoiceDialog — modal shown before opening another user's
canvas from the homepage Local Files card.

The old UI gated Load on cross-user rows; the new accountability model
lets the user Load freely but asks them up front whether they want to
**Copy** the canvas into their own workspace (recommended, no audit
trail) or **Open in place** (their subsequent saves get logged to the
target user's pending-review queue, see ``cross_user_audit``).

Layout (compact, three buttons stacked on a single footer row):

.. code-block:: text

    +--------------------------------------------------+
    | (warning icon)  Open another user's canvas       |
    |                                                  |
    | This canvas belongs to {owner}. Copy it into     |
    | your own workspace …                             |
    |                                                  |
    |              [ Cancel ]  [ Open in place ]       |
    |                            [ Copy (recommended) ]|
    +--------------------------------------------------+

Theme rule (CLAUDE.md §1.5): every colour and font size resolves through
``Config.get_color`` / ``Config.get_font_size``; no literals.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, setButton, tr


class OpenChoice(Enum):
    """Possible outcomes of :class:`CrossUserOpenChoiceDialog`."""

    CANCEL = "cancel"
    COPY = "copy"
    IN_PLACE = "in_place"


class CrossUserOpenChoiceDialog(QDialog):
    """Three-way modal: Copy / Open in place / Cancel.

    Use :meth:`pick` as the convenience entry point — it constructs the
    dialog, runs ``exec()``, and returns the :class:`OpenChoice` value.
    Cancelling via Esc / window close maps to ``OpenChoice.CANCEL``.
    """

    def __init__(
        self,
        owner_label: str,
        canvas_name: str,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._choice = OpenChoice.CANCEL
        self.setObjectName("crossUserOpenChoiceDialog")
        self.setModal(True)
        self.setWindowTitle(tr(
            "homepage.projects.cross_user_open_title",
            default="Open another user's canvas",
        ))
        self.resize(480, 260)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(12)

        title = QLabel(
            tr(
                "homepage.projects.cross_user_open_title",
                default="Open another user's canvas",
            ),
            self,
        )
        title.setObjectName("crossUserOpenChoiceTitle")
        outer.addWidget(title, 0)

        body_text = tr(
            "homepage.projects.cross_user_open_body",
            default=(
                "This canvas belongs to {owner}. Copy it into your own "
                "workspace for an isolated backup, or open the original "
                "in place — any edit / delete will be logged for {owner} "
                "to review on their next sign-in."
            ),
        ).format(owner=owner_label or "(unknown)")
        body = QLabel(body_text, self)
        body.setObjectName("crossUserOpenChoiceBody")
        body.setWordWrap(True)
        outer.addWidget(body, 1)

        # The canvas name as a secondary line so the user knows exactly
        # which file they're acting on. Hidden when no name given.
        if canvas_name:
            file_line = QLabel(canvas_name, self)
            file_line.setObjectName("crossUserOpenChoiceFile")
            file_line.setWordWrap(True)
            outer.addWidget(file_line, 0)

        # --- primary action: Copy (own row, full width) ----------------
        # border + save spec — the green outline signals "this is the
        # recommended, non-destructive choice" without flooding the
        # dialog with solid colour. setButton hard-codes setFixedSize
        # internally; we undo that with explicit min/max overrides so
        # the button can stretch to fill the dialog width.
        self._btn_copy = setButton(
            "homepage.projects.cross_user_open_copy",
            220, 40,
            kind="border", spec="save",
            default="Copy to my workspace (recommended)", parent=self,
        )
        self._btn_copy.setMinimumWidth(0)
        self._btn_copy.setMaximumWidth(16777215)
        self._btn_copy.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._btn_copy.setDefault(True)
        self._btn_copy.setAutoDefault(True)
        self._btn_copy.clicked.connect(self._on_copy)
        outer.addWidget(self._btn_copy, 0)

        # --- footer row: [ Open in place ] [ Cancel ] -------------------
        # Equal-weight buttons that together fill the dialog width: each
        # gets its own QSizePolicy(Expanding, Fixed) and the layout adds
        # both at stretch=1. Spec varies — Open-in-place uses ``notice``
        # so its outline reads as the actionable choice; Cancel stays
        # neutral.
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)

        self._btn_inplace = setButton(
            "homepage.projects.cross_user_open_inplace",
            140, 32,
            kind="border", spec="notice",
            default="Open in place", parent=self,
        )
        self._btn_inplace.setMinimumWidth(0)
        self._btn_inplace.setMaximumWidth(16777215)
        self._btn_inplace.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._btn_inplace.clicked.connect(self._on_inplace)
        footer.addWidget(self._btn_inplace, 1)

        self._btn_cancel = setButton(
            "homepage.projects.cross_user_open_cancel",
            96, 32,
            kind="border", spec="none",
            default="Cancel", parent=self,
        )
        self._btn_cancel.setMinimumWidth(0)
        self._btn_cancel.setMaximumWidth(16777215)
        self._btn_cancel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._btn_cancel.clicked.connect(self._on_cancel)
        footer.addWidget(self._btn_cancel, 1)

        outer.addLayout(footer)

        self._apply_theme()

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    @classmethod
    def pick(
        cls,
        owner_label: str,
        canvas_name: str,
        *,
        parent: Optional[QWidget] = None,
    ) -> OpenChoice:
        """Construct, exec, and return the user's choice in one call."""
        dlg = cls(owner_label, canvas_name, parent=parent)
        dlg.exec()
        return dlg._choice

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_cancel(self) -> None:
        self._choice = OpenChoice.CANCEL
        self.reject()

    def _on_inplace(self) -> None:
        self._choice = OpenChoice.IN_PLACE
        self.accept()

    def _on_copy(self) -> None:
        self._choice = OpenChoice.COPY
        self.accept()

    # Esc / window-close → cancel (Qt default is reject(), which keeps
    # the default _choice value of CANCEL — no extra wiring needed).

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        main = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        accent = Config.get_color("safe_zone")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QDialog#crossUserOpenChoiceDialog {{ background-color: {bg}; }}"
            f"QLabel#crossUserOpenChoiceTitle {{ color: {main}; "
            f"font-size: {font_normal}px; font-weight: 700; }}"
            f"QLabel#crossUserOpenChoiceBody {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QLabel#crossUserOpenChoiceFile {{ color: {accent}; "
            f"font-size: {font_small}px; font-weight: 600; }}"
        )
