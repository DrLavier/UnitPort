"""Homepage card base — rounded panel with a title row + body area.

All four homepage cards (user / news / projects / create) share this
chrome so they line up visually. The base widget:

* paints its own rounded background and 1-px border via ``Config.get_color``
* exposes a fixed-height title row with a left-aligned title label and an
  optional right-aligned "View all"-style link button
* exposes a body host (``content_layout``) that subclasses fill in

Theme rule (CLAUDE.md §1.5): every colour is read at use-time from
top-level ``Config.get_color`` slots (``bg_1`` / ``main_t1`` / ``highlight``
/ etc.). There are no hex literals or homepage-specific aliases here.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Assets, Config, I18n, tr


class HomepageCard(QFrame):
    """Rounded panel with title row + body host.

    Subclasses should populate ``self.content_layout`` (the body
    QVBoxLayout). Use :meth:`set_action_link` to wire the right-aligned
    link label (e.g. "View all"); the link is hidden by default.
    """

    action_clicked = pyqtSignal()

    _TITLE_ROW_H = 32

    def __init__(
        self,
        title_key: str,
        title_default: str,
        parent: Optional[QWidget] = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("homepageCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._title_key = title_key
        self._title_default = title_default
        self._action_key: str = ""
        self._action_default: str = ""
        # When True the card behaves as a *section* hosted inside another
        # card: no rounded background / border, zero outer padding so its
        # content sits flush with the host's content_layout. The title
        # row stays visible so subclasses that decorate it (search bars,
        # icon buttons) still render. Switched on by the merged
        # homepage workspace card.
        self._embedded = bool(embedded)

        root = QVBoxLayout(self)
        # Card interior padding — bumped from (16,14,16,16). The cards
        # sit on the bg_3 canvas with generous outer breathing room, so
        # their interiors need matching inside breathing room or the
        # body content reads as crammed against the rounded border.
        if self._embedded:
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(8)
        else:
            # Horizontal padding is wider than vertical: the cards sit
            # on a bg_3 canvas with generous outer gutters, and the
            # interior should mirror that breathing room sideways so
            # column-split content (merged Create + LocalFiles card)
            # has visible margin against the rounded border.
            root.setContentsMargins(40, 20, 40, 24)
            root.setSpacing(12)

        # ---- title row ----
        # Exposed as ``self.title_row`` so subclasses can append extras
        # (search inputs, badges, segmented controls) to the right of the
        # title label without having to rebuild the chrome.
        self.title_row = QHBoxLayout()
        self.title_row.setContentsMargins(0, 0, 0, 0)
        self.title_row.setSpacing(8)

        self._title_label = QLabel(tr(title_key, title_default))
        self._title_label.setObjectName("homepageCardTitle")
        self._title_label.setFixedHeight(self._TITLE_ROW_H)
        self.title_row.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignVCenter)
        # Stretch slot — subclasses can ``insertWidget(stretch_index, w)``
        # before this point, or ``addWidget`` after it, to place a widget
        # at the right edge of the title row.
        self.title_row.addStretch(1)

        self._action_btn = QPushButton("")
        self._action_btn.setObjectName("homepageCardAction")
        self._action_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._action_btn.setFlat(True)
        self._action_btn.setVisible(False)
        self._action_btn.clicked.connect(self.action_clicked.emit)
        self.title_row.addWidget(
            self._action_btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        root.addLayout(self.title_row)

        # ---- body host ----
        self._content_host = QWidget(self)
        self._content_host.setObjectName("homepageCardContent")
        self.content_layout = QVBoxLayout(self._content_host)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        root.addWidget(self._content_host, 1)

        # Re-render strings on language change.
        I18n.instance().language_changed.connect(self._retranslate)

        self._apply_theme()

    # ----- public --------------------------------------------------------

    def set_action_link(
        self,
        key: str,
        default: str,
        *,
        icon: Optional[str] = None,
    ) -> None:
        """Show the right-aligned link with text ``tr(key, default)``.

        ``icon``: optional bare stem passed to :meth:`Assets.find_icon`
        (e.g. ``"coffee"``); when present the icon is rendered to the
        left of the text at 16×16. Keyword-only so existing callsites
        keep their text-only behaviour.
        """
        self._action_key = key
        self._action_default = default
        self._action_btn.setText(tr(key, default))
        if icon:
            p = Assets.find_icon(icon)
            if p is not None:
                self._action_btn.setIcon(QIcon(str(p)))
                self._action_btn.setIconSize(QSize(16, 16))
        else:
            self._action_btn.setIcon(QIcon())
        self._action_btn.setVisible(True)

    def clear_action_link(self) -> None:
        self._action_key = ""
        self._action_default = ""
        self._action_btn.setVisible(False)

    def set_title(self, key: str, default: str) -> None:
        self._title_key = key
        self._title_default = default
        self._title_label.setText(tr(key, default))

    def apply_theme(self) -> None:
        """Re-read theme colours and rebuild the stylesheet."""
        self._apply_theme()

    # ----- internal ------------------------------------------------------

    def _retranslate(self) -> None:
        self._title_label.setText(tr(self._title_key, self._title_default))
        if self._action_key:
            self._action_btn.setText(tr(self._action_key, self._action_default))

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        border = Config.get_color("border_2")
        title_fg = Config.get_color("main_t1")
        link_fg = Config.get_color("highlight")
        link_hover = Config.get_color("highlight")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")

        # Embedded mode: no fill, no border — the card is a *section*
        # inside another HomepageCard and must not paint its own chrome.
        card_rule = (
            f"QFrame#homepageCard {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            if self._embedded else
            f"QFrame#homepageCard {{"
            f"  background: {bg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 20px;"
            f"}}"
        )
        self.setStyleSheet(
            card_rule
            + f"QLabel#homepageCardTitle {{"
            f"  color: {title_fg};"
            f"  background: transparent;"
            f"  font-size: {size_normal}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QPushButton#homepageCardAction {{"
            f"  color: {link_fg};"
            f"  background: transparent;"
            f"  border: none;"
            f"  padding: 2px 4px;"
            f"  font-size: {size_small}px;"
            f"}}"
            f"QPushButton#homepageCardAction:hover {{"
            f"  color: {link_hover};"
            f"}}"
            f"QWidget#homepageCardContent {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
        )
