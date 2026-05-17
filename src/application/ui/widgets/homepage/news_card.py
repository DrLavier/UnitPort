"""NewsCard — homepage community-news tile.

Empty for now. The card chrome (title row + "View all" link) is wired
through :class:`HomepageCard`; a single centred message tells the user
that news content has not been wired up yet. When the real feed lands,
replace the body of :meth:`_build_body` with the live list — every
other piece of the homepage is decoupled from this card.

Theme rule (CLAUDE.md §1.5): every colour read at use-time via
``Config.get_color``. No hex literals in this file.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget

from unitport_sdk import Config, I18n, tr

from .card_base import HomepageCard


class NewsCard(HomepageCard):
    """Top-right homepage card. Empty-state placeholder for now."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            title_key="homepage.news.title",
            title_default="News",
            parent=parent,
        )
        self.set_action_link("homepage.news.view_all", "View all")
        self._build_body()
        I18n.instance().language_changed.connect(self._retranslate)
        self._apply_news_theme()

    # ------------------------------------------------------------------

    def _build_body(self) -> None:
        self._coming_soon = QLabel(tr(
            "homepage.news.coming_soon",
            "News feed coming soon.",
        ))
        self._coming_soon.setObjectName("homepageNewsComingSoon")
        self._coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._coming_soon.setWordWrap(True)
        self.content_layout.addStretch(1)
        self.content_layout.addWidget(
            self._coming_soon, 0, Qt.AlignmentFlag.AlignCenter
        )
        self.content_layout.addStretch(1)

    def _retranslate(self) -> None:
        self._coming_soon.setText(tr(
            "homepage.news.coming_soon",
            "News feed coming soon.",
        ))

    # ------------------------------------------------------------------

    def apply_theme(self) -> None:                            # type: ignore[override]
        super().apply_theme()
        self._apply_news_theme()

    def _apply_news_theme(self) -> None:
        sub_fg = Config.get_color("sub_t2")
        size_small = Config.get_font_size("size_small")
        self.setStyleSheet(self.styleSheet() + (
            f"QLabel#homepageNewsComingSoon {{"
            f"  color: {sub_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
        ))
