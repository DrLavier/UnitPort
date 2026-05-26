# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""HomepageCommunityCard — top-right homepage card.

Replaces the legacy "News" placeholder. Two-column body split by a 1-px
vertical divider:

* Left column hosts a single large tutorial card (title + description +
  "Open Link" CTA on the text side, decorative SVG on the image side).
  The link target is a placeholder until the official tutorial is
  published — see ``_TUTORIAL_URL``.
* Right column hosts a timeline of the three most-recent GitHub
  Releases (fetched off-thread via :class:`CommunityReleasesFetchTask`).
  Each :class:`_ReleaseItem` paints a left-side dot at the title's
  vertical center, with a 2-px line extending down to connect to the
  next item's dot (the last item omits the line). The newest non-draft
  / non-pre-release item gets a ``safe_zone`` "Latest" badge next to its
  version. "Download" routes the chosen :class:`ReleaseInfo` back to
  ``HomepagePage`` → ``MainWindow`` so the existing
  ``ApplyUpdateTask`` + ``UpdateProgressDialog`` machinery handles the
  install (the user already saw release notes inline, so the in-between
  ``UpdateAvailableDialog`` is skipped).

Theme rule (CLAUDE.md §1.5): every colour is read at use-time via
``Config.get_color`` — no hex literals in this file. Font sizes only
use the closed ``[Font]`` set (size_mini / size_small / size_normal /
size_large).
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import List, Optional

from PyQt6.QtCore import QSize, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QDesktopServices, QFontMetrics, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsColorizeEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from packaging.version import InvalidVersion, Version

from unitport_sdk import (
    Assets,
    Config,
    I18n,
    get_task_signal,
    get_tasks_manager,
    log_warning,
    setButton,
    tr,
)

from application.service.updater import ReleaseInfo
from application.tools.community_fetch_task import CommunityReleasesFetchTask

from .card_base import HomepageCard


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORT_URL = "https://github.com/sponsors/DrLavier"
# Placeholder until the official tutorial site is published. Points at
# the repo README so the click does *something* useful in the meantime.
_TUTORIAL_URL = "https://github.com/DrLavier/UnitPort#readme"

_TUTORIAL_ICON = "icon_velocity_tracking"   # Assets.find_icon stem; swap later
_TUTORIAL_ICON_SIZE = 128       # Square side (= panel inner height)
_TUTORIAL_PADDING = 12

_MAX_RELEASES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MD_PREFIX = re.compile(r"^[\s>#*\-]+")
_MD_FENCE = re.compile(r"^```.*$", re.MULTILINE)


def _first_line(body: str) -> str:
    """Return the first non-empty markdown line of ``body`` as plain text.

    Strips fences, leading hashes / asterisks / dashes / quote-arrows so
    the result is readable as a one-liner. Does not attempt to render
    markdown — the goal is a single elided summary line.
    """
    if not body:
        return ""
    text = _MD_FENCE.sub("", body)
    for raw in text.splitlines():
        stripped = _MD_PREFIX.sub("", raw).strip()
        if stripped:
            return stripped
    return ""


def _parse_iso(s: str) -> Optional[_dt.datetime]:
    if not s:
        return None
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(t)
    except ValueError:
        return None


def _format_date(published_at: str) -> str:
    """Human-readable date for the release item right column.

    Within a week → "{n} days ago" (with today/yesterday/day_one specials).
    Otherwise → ISO short ``YYYY-MM-DD`` — the published date in the
    user's local timezone.
    """
    dt = _parse_iso(published_at)
    if dt is None:
        return ""
    now = _dt.datetime.now(dt.tzinfo) if dt.tzinfo else _dt.datetime.now()
    delta = now - dt
    days = int(delta.total_seconds() // 86400)
    if days < 0:
        days = 0
    if days == 0:
        return tr("homepage.community.today", "today")
    if days == 1:
        return tr("homepage.community.yesterday", "yesterday")
    if days < 7:
        return tr(
            "homepage.community.days_ago", "{n} days ago",
        ).format(n=days)
    # 1+ week: ISO short.
    local = dt.astimezone() if dt.tzinfo else dt
    return local.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Tutorial panel (left column)
# ---------------------------------------------------------------------------


class _TutorialPanel(QFrame):
    """Tutorial card item — compact two-column tile.

    Layout: a self-contained rounded card whose body is a horizontal
    ``[ text_col | image ]`` pair. The image is a fixed square of side
    ``_TUTORIAL_ICON_SIZE``; the text column is pinned to the same
    height (title at top, description fills, "Open Link" pinned at the
    bottom). The panel's overall height is therefore
    ``_TUTORIAL_ICON_SIZE + 2 * _TUTORIAL_PADDING`` — fixed, so this
    card never sprawls to fill the left half of the Community body.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("homepageTutorialPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Fixed total height = image side + top/bottom padding. Width
        # expands to fill the available column.
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(_TUTORIAL_ICON_SIZE + 2 * _TUTORIAL_PADDING)

        root = QHBoxLayout(self)
        root.setContentsMargins(
            _TUTORIAL_PADDING, _TUTORIAL_PADDING,
            _TUTORIAL_PADDING, _TUTORIAL_PADDING,
        )
        root.setSpacing(12)

        # ---- text column ------------------------------------------------
        # Pinned to image height so title / desc / button form a tight
        # column the same vertical extent as the square image.
        text_host = QWidget(self)
        text_host.setObjectName("homepageTutorialTextCol")
        text_host.setFixedHeight(_TUTORIAL_ICON_SIZE)
        text_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        text_col = QVBoxLayout(text_host)
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(4)

        self._title = QLabel(
            tr("homepage.community.tutorial_title", "Official Tutorial"),
            text_host,
        )
        self._title.setObjectName("homepageTutorialTitle")
        self._title.setWordWrap(False)
        text_col.addWidget(self._title)

        self._desc = QLabel(
            tr(
                "homepage.community.tutorial_desc",
                "Step-by-step guides for building, training, and deploying.",
            ),
            text_host,
        )
        self._desc.setObjectName("homepageTutorialDesc")
        self._desc.setWordWrap(True)
        self._desc.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self._desc.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        text_col.addWidget(self._desc, 1)

        # "Open Link" — highlight border + highlight text, transparent bg.
        # NOTE: official tutorial site not yet published; the button is
        # temporarily replaced by a "Coming soon..." placeholder. Re-enable
        # the block below once _TUTORIAL_URL points at the real tutorial.
        # self._open_btn = setButton(
        #     "homepage.community.tutorial_open",
        #     104, 26,
        #     kind="border",
        #     spec="none",
        #     default="Open Link",
        #     font_color=Config.get_color("highlight"),
        #     border_color=Config.get_color("highlight"),
        #     hover_color=Config.get_color("hover_2"),
        # )
        # self._open_btn.clicked.connect(self._open_tutorial)
        self._open_btn = None
        self._coming_soon = QLabel(
            tr("homepage.community.tutorial_coming_soon", "Coming soon..."),
            text_host,
        )
        self._coming_soon.setObjectName("homepageTutorialComingSoon")
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self._coming_soon, 0, Qt.AlignmentFlag.AlignLeft)
        btn_row.addStretch(1)
        text_col.addLayout(btn_row)

        root.addWidget(text_host, 1)

        # ---- icon column ------------------------------------------------
        # Strict square: width == height == _TUTORIAL_ICON_SIZE.
        self._icon_label = QLabel(self)
        self._icon_label.setObjectName("homepageTutorialIcon")
        self._icon_label.setFixedSize(_TUTORIAL_ICON_SIZE, _TUTORIAL_ICON_SIZE)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed,
        )
        self._load_icon()
        root.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignTop)

        I18n.instance().language_changed.connect(self._retranslate)
        self._apply_theme()

    # ------------------------------------------------------------------
    def _load_icon(self) -> None:
        p = Assets.find_icon(_TUTORIAL_ICON)
        if p is None:
            return
        pm = QPixmap(str(p))
        if pm.isNull():
            return
        scaled = pm.scaled(
            _TUTORIAL_ICON_SIZE, _TUTORIAL_ICON_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._icon_label.setPixmap(scaled)

    def _open_tutorial(self) -> None:
        QDesktopServices.openUrl(QUrl(_TUTORIAL_URL))

    def _retranslate(self) -> None:
        self._title.setText(
            tr("homepage.community.tutorial_title", "Official Tutorial"),
        )
        self._desc.setText(
            tr(
                "homepage.community.tutorial_desc",
                "Step-by-step guides for building, training, and deploying.",
            ),
        )
        if self._coming_soon is not None:
            self._coming_soon.setText(
                tr(
                    "homepage.community.tutorial_coming_soon",
                    "Coming soon...",
                ),
            )

    def apply_theme(self) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        title_fg = Config.get_color("main_t1")
        desc_fg = Config.get_color("sub_t1")
        card_bg = Config.get_color("row_1")
        card_border = Config.get_color("border_2")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QFrame#homepageTutorialPanel {{"
            f"  background: {card_bg};"
            f"  border: 1px solid {card_border};"
            f"  border-radius: 12px;"
            f"}}"
            f"QWidget#homepageTutorialTextCol {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageTutorialTitle {{"
            f"  color: {title_fg};"
            f"  background: transparent;"
            f"  font-size: {size_normal}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageTutorialDesc {{"
            f"  color: {desc_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageTutorialIcon {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageTutorialComingSoon {{"
            f"  color: {desc_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-style: italic;"
            f"  border: none;"
            f"}}"
        )


# ---------------------------------------------------------------------------
# Release item (right column)
# ---------------------------------------------------------------------------


class _TimelineGutter(QWidget):
    """Fixed-width gutter painting a dot + a vertical tail line.

    The dot sits at ``y_anchor`` (the vertical centre of the parent
    item's title row). A 2-px ``highlight`` line always continues from
    the bottom of the dot down to the bottom edge of the gutter — acts
    as a per-card decorative tail rather than a strict timeline link.
    """

    _DOT_R = 4
    _LINE_W = 2
    _WIDTH = 18

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._y_anchor = 12  # Updated by set_anchor() once title row sizes.
        self.setFixedWidth(self._WIDTH)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding,
        )
        # Background is transparent — the dot and line are painted
        # directly. No QSS needed.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_anchor(self, y: int) -> None:
        if y == self._y_anchor:
            return
        self._y_anchor = int(y)
        self.update()

    def paintEvent(self, _event) -> None:                          # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(Config.get_color("highlight"))
        cx = self._WIDTH // 2
        cy = self._y_anchor
        # Line first so the dot paints on top of its start point.
        p.fillRect(
            cx - self._LINE_W // 2,
            cy,
            self._LINE_W,
            max(0, self.height() - cy),
            color,
        )
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(
            cx - self._DOT_R, cy - self._DOT_R,
            self._DOT_R * 2, self._DOT_R * 2,
        )


class _ReleaseItem(QFrame):
    """One row in the releases timeline.

    Layout:

        [gutter] [ version + Latest? + stretch + date ]
                 [ elided one-line description         ]
                 [ Open page · Download · stretch      ]
    """

    apply_requested = pyqtSignal(object)     # ReleaseInfo

    _TITLE_ROW_H = 24
    _DESC_ROW_H = 20

    def __init__(
        self,
        release: ReleaseInfo,
        *,
        is_latest: bool,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._release = release
        self._is_latest = bool(is_latest)
        self._desc_text = _first_line(release.body)

        self.setObjectName("homepageReleaseItem")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )

        # Card-shaped wrapper — row_1 bg + border_2 border + 12 px radius,
        # mirrors the Tutorial card on the left side so the right column
        # also reads as a stack of item cards. Inner padding leaves
        # breathing room around the timeline gutter + content.
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 14, 10)
        root.setSpacing(8)

        self._gutter = _TimelineGutter(self)
        root.addWidget(self._gutter)

        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(4)
        root.addLayout(content, 1)

        # ---- title row --------------------------------------------------
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        version_text = release.name.strip() or release.tag or release.version
        self._version_label = QLabel(version_text, self)
        self._version_label.setObjectName("homepageReleaseVersion")
        self._version_label.setFixedHeight(self._TITLE_ROW_H)
        title_row.addWidget(self._version_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._latest_badge: Optional[QLabel] = None
        if self._is_latest:
            self._latest_badge = QLabel(
                tr("homepage.community.latest", "Latest"), self,
            )
            self._latest_badge.setObjectName("homepageReleaseLatest")
            self._latest_badge.setFixedHeight(self._TITLE_ROW_H)
            title_row.addWidget(
                self._latest_badge, 0, Qt.AlignmentFlag.AlignVCenter,
            )

        title_row.addStretch(1)

        self._date_label = QLabel(_format_date(release.published_at), self)
        self._date_label.setObjectName("homepageReleaseDate")
        self._date_label.setFixedHeight(self._TITLE_ROW_H)
        self._date_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        title_row.addWidget(
            self._date_label, 0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        content.addLayout(title_row)

        # ---- description row -------------------------------------------
        self._desc_label = QLabel(self._desc_text, self)
        self._desc_label.setObjectName("homepageReleaseDesc")
        self._desc_label.setFixedHeight(self._DESC_ROW_H)
        self._desc_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        content.addWidget(self._desc_label)

        # ---- buttons row -----------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 6, 0, 0)
        btn_row.setSpacing(8)

        self._open_page_btn = setButton(
            "homepage.community.open_page",
            96, 26,
            kind="border",
            spec="none",
            default="Open page",
            font_color=Config.get_color("sub_t1"),
            border_color=Config.get_color("border_1"),
            hover_color=Config.get_color("hover_2"),
            hover_fg_color=Config.get_color("main_t1"),
        )
        self._open_page_btn.clicked.connect(self._open_page)
        btn_row.addWidget(self._open_page_btn)

        # Per §1.6 the current build version is sourced exclusively from
        # system.ini[System].version. Compare via packaging.Version so
        # "0.10.0" > "0.9.0"; fall back to string equality only if either
        # side fails to parse (malformed release tag, etc).
        current_version = str(
            Config.get_value("System", "version", "0.0.0") or "0.0.0",
        ).strip()
        release_v_str = (release.version or "").strip()

        def _parse(s: str) -> Optional[Version]:
            try:
                return Version(s) if s else None
            except InvalidVersion:
                return None

        rv = _parse(release_v_str)
        cv = _parse(current_version)
        if rv is not None and cv is not None:
            if rv > cv:
                relation = "newer"
            elif rv < cv:
                relation = "older"
            else:
                relation = "equal"
        else:
            relation = "equal" if release_v_str == current_version else "newer"

        self._download_btn: Optional[object] = None
        self._current_indicator: Optional[QFrame] = None
        self._current_text_label: Optional[QLabel] = None

        if relation == "equal":
            self._current_indicator = self._build_current_indicator()
            btn_row.addWidget(self._current_indicator)
        else:
            # Newer-than-current (not installed locally) → safe_zone to
            # invite the upgrade. Older-than-current → highlight as a
            # warning tone for the rollback / regression case.
            if relation == "newer":
                tint = Config.get_color("safe_zone")
                hover = Config.get_color("safe_zone_hover")
            else:                                                  # older
                tint = Config.get_color("highlight")
                hover = Config.get_color("hover_2")
            download_icon = Assets.find_icon("icon_download")
            self._download_btn = setButton(
                "homepage.community.download",
                104, 26,
                kind="border",
                spec="none",
                icon=str(download_icon) if download_icon is not None else None,
                default="Download",
                font_color=tint,
                border_color=tint,
                hover_color=hover,
                hover_fg_color=Config.get_color("main_t2"),
            )
            self._download_btn.clicked.connect(self._on_download)
            btn_row.addWidget(self._download_btn)

        btn_row.addStretch(1)
        content.addLayout(btn_row)

        I18n.instance().language_changed.connect(self._retranslate)
        self._apply_theme()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _open_page(self) -> None:
        url = (self._release.html_url or "").strip()
        if not url:
            return
        QDesktopServices.openUrl(QUrl(url))

    def _on_download(self) -> None:
        self.apply_requested.emit(self._release)

    # ------------------------------------------------------------------
    # Current-version indicator (replaces Download when version matches)
    # ------------------------------------------------------------------
    def _build_current_indicator(self) -> QFrame:
        """Inert ``[✓] Current`` chip occupying the Download slot."""
        frame = QFrame(self)
        frame.setObjectName("homepageReleaseCurrent")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        frame.setFrameShape(QFrame.Shape.NoFrame)
        # Match the Download button footprint so swapping doesn't shift
        # neighbouring widgets in the buttons row.
        frame.setFixedSize(104, 26)
        frame.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed,
        )

        row = QHBoxLayout(frame)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(6)

        icon_label = QLabel(frame)
        icon_label.setObjectName("homepageReleaseCurrentIcon")
        icon_label.setFixedSize(14, 14)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        yes_icon = Assets.find_icon("icon_yes")
        if yes_icon is not None:
            pm = QPixmap(str(yes_icon))
            if not pm.isNull():
                icon_label.setPixmap(
                    pm.scaled(
                        14, 14,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        row.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._current_text_label = QLabel(
            tr("homepage.community.current", "Current"), frame,
        )
        self._current_text_label.setObjectName("homepageReleaseCurrentText")
        self._current_text_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        row.addWidget(self._current_text_label, 1)
        return frame

    def _retranslate(self) -> None:
        if self._latest_badge is not None:
            self._latest_badge.setText(
                tr("homepage.community.latest", "Latest"),
            )
        if self._current_text_label is not None:
            self._current_text_label.setText(
                tr("homepage.community.current", "Current"),
            )
        self._date_label.setText(_format_date(self._release.published_at))
        # Description text is the release body — not translated. Just
        # re-elide in case the layout settled differently.
        self._reelide()

    # ------------------------------------------------------------------
    # Layout / paint
    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:                          # noqa: N802
        super().resizeEvent(event)
        self._gutter.set_anchor(self._TITLE_ROW_H // 2 + 4)
        self._reelide()

    def _reelide(self) -> None:
        if not self._desc_text:
            self._desc_label.setText("")
            return
        fm = QFontMetrics(self._desc_label.font())
        width = max(1, self._desc_label.width() - 4)
        self._desc_label.setText(
            fm.elidedText(self._desc_text, Qt.TextElideMode.ElideRight, width),
        )

    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        version_fg = Config.get_color("main_t1")
        date_fg = Config.get_color("sub_t2")
        desc_fg = Config.get_color("sub_t1")
        latest_fg = Config.get_color("safe_zone")
        card_bg = Config.get_color("row_1")
        card_border = Config.get_color("border_2")
        size_small = Config.get_font_size("size_small")
        size_mini = Config.get_font_size("size_mini")
        self.setStyleSheet(
            f"QFrame#homepageReleaseItem {{"
            f"  background: {card_bg};"
            f"  border: 1px solid {card_border};"
            f"  border-radius: 12px;"
            f"}}"
            f"QLabel#homepageReleaseVersion {{"
            f"  color: {version_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageReleaseLatest {{"
            f"  color: {latest_fg};"
            f"  background: transparent;"
            f"  font-size: {size_mini}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"  padding: 0px 2px;"
            f"}}"
            f"QLabel#homepageReleaseDate {{"
            f"  color: {date_fg};"
            f"  background: transparent;"
            f"  font-size: {size_mini}px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageReleaseDesc {{"
            f"  color: {desc_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            f"QFrame#homepageReleaseCurrent {{"
            f"  background: transparent;"
            f"  border: 1px solid {latest_fg};"
            f"  border-radius: 4px;"
            f"}}"
            f"QLabel#homepageReleaseCurrentIcon {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageReleaseCurrentText {{"
            f"  color: {latest_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
        )
        # Force gutter repaint (highlight slot may have changed).
        self._gutter.update()


# ---------------------------------------------------------------------------
# Top-level card
# ---------------------------------------------------------------------------


class HomepageCommunityCard(HomepageCard):
    """Top-right homepage card: tutorial + GitHub releases timeline."""

    # Forwarded to ``HomepagePage.community_apply_requested`` so the
    # MainWindow can launch the existing apply-update flow with this
    # user-selected ReleaseInfo (skipping UpdateAvailableDialog).
    apply_requested = pyqtSignal(object)        # ReleaseInfo

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            title_key="homepage.community.title",
            title_default="Community",
            parent=parent,
        )
        self.set_action_link(
            "homepage.community.support_us", "Support Us",
            icon="icon_coffee",
        )
        self.action_clicked.connect(self._open_support)
        # Tint the action button icon to safe_zone so it matches the
        # safe_zone text colour applied via CSS in
        # ``_apply_community_theme``. The base ``HomepageCard`` factory
        # paints the icon as a raw SVG with no recolour; without this
        # effect the icon would keep its source colour and clash with
        # the surrounding safe_zone tint.
        self._tint_action_icon()

        self._fetch_task_id: str = ""
        self._releases: List[ReleaseInfo] = []
        self._item_widgets: List[_ReleaseItem] = []
        self._tutorial_panel: Optional[_TutorialPanel] = None
        self._divider: Optional[QFrame] = None
        self._releases_host: Optional[QWidget] = None
        self._releases_layout: Optional[QVBoxLayout] = None
        self._loading_label: Optional[QLabel] = None

        self._build_body()
        get_task_signal().task_finished.connect(self._on_task_finished)
        I18n.instance().language_changed.connect(self._retranslate)
        self._apply_community_theme()
        self._kick_fetch()

    # ------------------------------------------------------------------
    # Build / layout
    # ------------------------------------------------------------------
    def _build_body(self) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(20)

        # Wrap the tutorial in a top-aligned column so the fixed-height
        # tutorial card pins to the top instead of getting stretched
        # across the full body height by the parent QHBoxLayout.
        tutorial_col_host = QWidget(self)
        tutorial_col_host.setObjectName("homepageTutorialColHost")
        tutorial_col_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        tutorial_col = QVBoxLayout(tutorial_col_host)
        tutorial_col.setContentsMargins(0, 0, 0, 0)
        tutorial_col.setSpacing(0)
        self._tutorial_panel = _TutorialPanel(tutorial_col_host)
        tutorial_col.addWidget(self._tutorial_panel)
        tutorial_col.addStretch(1)
        row.addWidget(tutorial_col_host, 1)

        self._divider = QFrame(self)
        self._divider.setObjectName("homepageCommunityDivider")
        self._divider.setFrameShape(QFrame.Shape.NoFrame)
        self._divider.setFixedWidth(1)
        self._divider.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding,
        )
        row.addWidget(self._divider)

        self._releases_host = QWidget(self)
        self._releases_host.setObjectName("homepageCommunityReleases")
        self._releases_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self._releases_layout = QVBoxLayout(self._releases_host)
        self._releases_layout.setContentsMargins(0, 0, 0, 0)
        # 8 px between item cards so each one reads as a distinct rounded
        # tile instead of merging into a single block.
        self._releases_layout.setSpacing(8)

        self._loading_label = QLabel(
            tr("homepage.community.releases_loading", "Loading releases…"),
            self._releases_host,
        )
        self._loading_label.setObjectName("homepageReleasesPlaceholder")
        self._loading_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        self._releases_layout.addWidget(self._loading_label)
        self._releases_layout.addStretch(1)

        row.addWidget(self._releases_host, 1)

        self.content_layout.addLayout(row, 1)

    # ------------------------------------------------------------------
    # Fetch + populate
    # ------------------------------------------------------------------
    def _kick_fetch(self) -> None:
        try:
            self._fetch_task_id = get_tasks_manager().submit(
                CommunityReleasesFetchTask(n=_MAX_RELEASES),
            )
        except Exception as exc:                                   # noqa: BLE001
            log_warning(f"[homepage:community] failed to submit fetch: {exc}")
            self._show_placeholder("releases_unavailable")

    @pyqtSlot(str, bool, object)
    def _on_task_finished(
        self, task_id: str, ok: bool, result: object,
    ) -> None:
        if task_id != self._fetch_task_id:
            return
        self._fetch_task_id = ""
        if not ok or not isinstance(result, list):
            self._show_placeholder("releases_unavailable")
            return
        releases = [r for r in result if isinstance(r, ReleaseInfo)]
        if not releases:
            self._show_placeholder("releases_empty")
            return
        self._populate(releases[:_MAX_RELEASES])

    def _show_placeholder(self, suffix: str) -> None:
        """Replace the right column body with a single sub-t2 message."""
        self._clear_releases()
        if self._releases_layout is None or self._releases_host is None:
            return
        label = QLabel(
            tr(
                f"homepage.community.{suffix}",
                {
                    "releases_loading": "Loading releases…",
                    "releases_unavailable": "Could not load releases.",
                    "releases_empty": "No releases published yet.",
                }.get(suffix, ""),
            ),
            self._releases_host,
        )
        label.setObjectName("homepageReleasesPlaceholder")
        label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        self._loading_label = label
        self._releases_layout.addWidget(label)
        self._releases_layout.addStretch(1)
        self._apply_community_theme()

    def _populate(self, releases: List[ReleaseInfo]) -> None:
        self._clear_releases()
        if self._releases_layout is None or self._releases_host is None:
            return
        self._releases = list(releases)
        latest_idx = -1
        for i, r in enumerate(releases):
            if not r.prerelease:
                latest_idx = i
                break
        for i, release in enumerate(releases):
            item = _ReleaseItem(
                release,
                is_latest=(i == latest_idx),
                parent=self._releases_host,
            )
            item.apply_requested.connect(self._on_item_apply)
            self._releases_layout.addWidget(item)
            self._item_widgets.append(item)
        self._releases_layout.addStretch(1)

    def _clear_releases(self) -> None:
        if self._releases_layout is None:
            return
        for item in self._item_widgets:
            try:
                item.apply_requested.disconnect(self._on_item_apply)
            except (TypeError, RuntimeError):
                pass
        self._item_widgets.clear()
        while self._releases_layout.count():
            child = self._releases_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._loading_label = None

    def _on_item_apply(self, release: object) -> None:
        if isinstance(release, ReleaseInfo):
            self.apply_requested.emit(release)

    # ------------------------------------------------------------------
    # Title-row action
    # ------------------------------------------------------------------
    def _open_support(self) -> None:
        QDesktopServices.openUrl(QUrl(_SUPPORT_URL))

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------
    def _retranslate(self) -> None:
        if self._loading_label is not None:
            # Cheapest correctness: derive the key suffix from the live
            # English default the label was constructed with — but since
            # we may have set any of the three placeholders, repaint by
            # looking at whether a fetch is in flight.
            if self._fetch_task_id:
                self._loading_label.setText(
                    tr(
                        "homepage.community.releases_loading",
                        "Loading releases…",
                    ),
                )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:                                 # noqa: D401
        super().apply_theme()
        if self._tutorial_panel is not None:
            try:
                self._tutorial_panel.apply_theme()
            except Exception:                                      # noqa: BLE001
                pass
        for item in self._item_widgets:
            try:
                item.apply_theme()
            except Exception:                                      # noqa: BLE001
                pass
        self._apply_community_theme()

    def _apply_community_theme(self) -> None:
        divider_bg = Config.get_color("border_2")
        placeholder_fg = Config.get_color("sub_t2")
        action_fg = Config.get_color("safe_zone")
        size_small = Config.get_font_size("size_small")
        # Append to the inherited stylesheet so the base card chrome
        # stays intact. The Support Us action button defined in
        # HomepageCard defaults to ``highlight`` — re-target it here to
        # ``safe_zone`` so the title-row CTA matches the safe_zone
        # branding for community / supporter affordances.
        suffix = (
            f"QFrame#homepageCommunityDivider {{"
            f"  background: {divider_bg};"
            f"  border: none;"
            f"}}"
            f"QWidget#homepageCommunityReleases {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageReleasesPlaceholder {{"
            f"  color: {placeholder_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            f"QPushButton#homepageCardAction {{"
            f"  color: {action_fg};"
            f"  background: transparent;"
            f"  border: none;"
            f"  padding: 2px 4px;"
            f"  font-size: {size_small}px;"
            f"}}"
            f"QPushButton#homepageCardAction:hover {{"
            f"  color: {action_fg};"
            f"}}"
        )
        self.setStyleSheet(self.styleSheet() + suffix)
        # Re-apply the icon tint — apply_theme() refreshes get_color()
        # outputs, so the effect needs to follow with the latest slot.
        self._tint_action_icon()

    def _tint_action_icon(self) -> None:
        """Apply a safe_zone QGraphicsColorizeEffect to the title-row icon.

        ``HomepageCard._action_btn`` paints the icon as a raw SVG; a
        QGraphicsColorizeEffect on the button recolours both the text
        glyphs and the icon, but the QSS rules above already pin the
        text colour, so the visible result is just the icon adopting
        the safe_zone tint.
        """
        btn = getattr(self, "_action_btn", None)
        if btn is None:
            return
        effect = QGraphicsColorizeEffect(btn)
        effect.setColor(QColor(Config.get_color("safe_zone")))
        effect.setStrength(1.0)
        btn.setGraphicsEffect(effect)


__all__ = ["HomepageCommunityCard"]
