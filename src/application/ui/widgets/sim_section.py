# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Shared section/form widgets for the mission_panel Simulation cards.

Extracted from ``policy_simulation_card.py`` so siblings (the upcoming
``init_pose_subsection`` widget) can reuse the same visual treatment
without copy-paste.

All colors and font sizes are resolved at use-time via
``Config.get_color`` / ``Config.get_font_size`` (CLAUDE.md §1.5) — no
literal hex values anywhere. Subscribe to theme changes by calling
``apply_theme()`` after construction and on every theme refresh.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, I18nLabel


class SectionFrame(QFrame):
    """Card subsection with an i18n title and a vertical body layout.

    Re-used across the Simulation cards to keep section visuals
    identical. Subclasses extend the body via ``body_layout()``.
    """

    def __init__(
        self,
        title_key: str,
        title_default: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionSimSection")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self._title = I18nLabel(title_key, default=title_default, parent=self)
        self._title.setObjectName("missionSimSectionTitle")
        self._title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        outer.addWidget(self._title, 0)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        outer.addWidget(self._body, 1)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def body_widget(self) -> QWidget:
        return self._body

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        title_color = Config.get_color("sub_t1")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QFrame#missionSimSection {{ background-color: {bg}; "
            f"border-radius: 6px; }}"
        )
        self._title.setStyleSheet(
            f"QLabel#missionSimSectionTitle {{ color: {title_color}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


class FormRow(QWidget):
    """Two-line row: i18n label on top, control widget below."""

    def __init__(
        self,
        label_key: str,
        label_default: str,
        control: QWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = I18nLabel(label_key, default=label_default, parent=self)
        self.label.setObjectName("missionSimFormLabel")
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.label, 0)
        self.control = control
        layout.addWidget(self.control, 0)

    def apply_theme(self) -> None:
        sub = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        self.label.setStyleSheet(
            f"QLabel#missionSimFormLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


__all__ = ["SectionFrame", "FormRow"]
