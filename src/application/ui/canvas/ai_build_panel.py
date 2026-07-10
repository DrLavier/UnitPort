# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AiBuildPanel — full-cover "AI Build" overlay (IDE-style three-column layout).

This is the **front-end contract** for natural-language canvas orchestration
(see ``knowledge_base/nl_canvas_orchestration_blueprint.md`` §4.B1). The overlay
is opened/collapsed by the MissionControlPanel top-row "AI" switch (Training
Canva mode only); it is a ``main_panel``-level sibling of the canvas (NOT a
viewport child) so the ``QGraphicsOpacityEffect`` over the body's black veil can
composite against the live canvas behind it — the opacity slider reveals the
canvas exactly like MissionControlPanel's chart overlay.

Layout (redesign — see ``界面设计.png``)::

    top_row   API-settings gear + title (no close button — the AI Coding tab
              in the Mission Control slider opens / collapses this panel)
    body  ┌ Chats card ┬ Chat column ──────────────┬ Configuration card ──────┐
          │  + New /    │  bubbles + prompt card     │  source combobox +       │
          │  search /   │  (chips + newline + Revert │  Robot/Backend info +    │
          │  time cards │   + Send on one row)       │  collapsible sections    │
          └─────────────┴────────────────────────────┴──────────────────────────┘

Conversation threads are **per-canvas** and persisted under
``Paths.USER_CONFIG_DIR`` via
:mod:`application.service.ai_orchestration.conversation_store` — threads never
cross canvases.

Backend seam (unchanged): the agent harness drives the chat column through
:meth:`append_segments` / :meth:`append_markup` / :meth:`append_line` (each call
appends to the in-progress agent bubble) and reads :attr:`prompt_submitted` /
:attr:`reset_requested` / :attr:`settings_requested`. :meth:`set_progress` drives
the chat-header progress bar.

⚠ BACKEND TODO (front-end is ahead of the backend here — a backend-focused pass
will wire these data seams):
  - :meth:`set_config_preview` renders the Configuration card sections, but no
    producer feeds it yet (it should project the live canvas / a chat-generated
    config into typed sections). Until then the preview shows its empty state.
  - :attr:`config_source_changed` / :attr:`config_save_requested` /
    :meth:`set_config_sources` are the combobox + "Save Config" seams; nothing
    consumes them yet.
  - Robot + Backend (shown in the Configuration card) come from
    :meth:`set_requirements_context` (already wired by the page).

§5 compliance: every color is read at use-time from ``Config.get_color`` against
a ``system.ini[Theme]`` slot; fonts via ``Config.get_font_size``. UI strings go
through ``tr`` / ``i18n_bind``.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, QTime, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, log_warning, tr

from application.service.ai_orchestration import conversation_store as convstore
from application.service.ai_orchestration.contracts import Requirements

# The chat surface (user/assistant cards) lives in its own widget. LogSegment /
# parse_markup are re-exported here so existing importers (agent_bridge, tests)
# keep resolving them from this module.
from .bot_chat_zone import (  # noqa: F401  (re-exported)
    BotChatZone,
    LogSegment,
    _DEFAULT_ROLE,
    _ElidedLabel,
    parse_markup,
)


# ---------------------------------------------------------------------------
# Time grouping for the conversation rail (Today / Yesterday / Previous 7 days)
# ---------------------------------------------------------------------------
_GROUP_TODAY = "today"
_GROUP_YESTERDAY = "yesterday"
_GROUP_WEEK = "week"
_GROUP_OLDER = "older"

#: Render order + i18n key/default for each time bucket header.
_GROUP_ORDER: List[Tuple[str, str, str]] = [
    (_GROUP_TODAY, "canvas.ai_build.group_today", "Today"),
    (_GROUP_YESTERDAY, "canvas.ai_build.group_yesterday", "Yesterday"),
    (_GROUP_WEEK, "canvas.ai_build.group_week", "Previous 7 days"),
    (_GROUP_OLDER, "canvas.ai_build.group_older", "Older"),
]

_DAY_SECONDS = 86_400


def _start_of_today(now: float) -> float:
    """Local midnight at/just-before ``now`` (DST-naive; fine for bucketing)."""
    lt = time.localtime(now)
    return now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)


def _time_group(updated: float, now: float) -> str:
    start_today = _start_of_today(now)
    if updated >= start_today:
        return _GROUP_TODAY
    if updated >= start_today - _DAY_SECONDS:
        return _GROUP_YESTERDAY
    if updated >= start_today - 6 * _DAY_SECONDS:
        return _GROUP_WEEK
    return _GROUP_OLDER


def _fmt_time_label(updated: float, now: float) -> str:
    """``HH:MM`` for today, ``MM/DD`` otherwise (compact card timestamp)."""
    if updated <= 0:
        return ""
    lt = time.localtime(updated)
    if updated >= _start_of_today(now):
        return f"{lt.tm_hour:02d}:{lt.tm_min:02d}"
    return f"{lt.tm_mon:02d}/{lt.tm_mday:02d}"


# ===========================================================================
# Small widgets
# ===========================================================================
class _ConvoCard(QFrame):
    """One conversation thread as a 12px-rounded card (redesign).

    Title + compact time on the top row, a one-line preview subtitle below, and
    a small ``×`` delete affordance. Single-click selects, double-click renames
    the title inline. Colors are read at use-time (§5); the card is rebuilt on
    theme change so it always tracks the active palette."""

    clicked = pyqtSignal(str)
    delete_clicked = pyqtSignal(str)
    renamed = pyqtSignal(str, str)

    def __init__(
        self,
        conv_id: str,
        title: str,
        subtitle: str = "",
        time_label: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._conv_id = conv_id
        self.setObjectName("aiConvoCard")
        self.setProperty("active", "false")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 10, 9)
        lay.setSpacing(3)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self._title = _ElidedLabel(title, self)
        self._title.setObjectName("aiConvoCardTitle")
        top.addWidget(self._title, 1)

        self._time = QLabel(time_label, self)
        self._time.setObjectName("aiConvoCardTime")
        self._time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._time, 0)

        self._del = QPushButton("×", self)
        self._del.setObjectName("aiConvoCardDelete")
        self._del.setFixedSize(18, 18)
        self._del.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(self._del, "setToolTip", "canvas.ai_build.delete_convo", default="Delete")
        self._del.clicked.connect(lambda: self.delete_clicked.emit(self._conv_id))
        top.addWidget(self._del, 0)
        lay.addLayout(top)

        # Inline rename editor (hidden until double-click).
        self._edit = QLineEdit(self)
        self._edit.setObjectName("aiConvoCardEdit")
        self._edit.hide()
        self._edit.editingFinished.connect(self._commit_rename)
        lay.addWidget(self._edit)

        self._subtitle = _ElidedLabel(subtitle, self)
        self._subtitle.setObjectName("aiConvoCardSub")
        self._subtitle.setVisible(bool(subtitle))
        lay.addWidget(self._subtitle)

        self._apply_style()

    def _apply_style(self) -> None:
        card_bg = Config.get_color("bg_1", "#1E1E1E")
        card_bg_hover = Config.get_color("hover_2", "#212121")
        border = Config.get_color("border_1", "#444444")
        accent = Config.get_color("safe_zone", "#36E38E")
        title_c = Config.get_color("main_t1", "#D6D3C7")
        sub_c = Config.get_color("sub_t2", "#777777")
        del_c = Config.get_color("sub_t2", "#777777")
        del_hover = Config.get_color("log_error", "#FF6B6B")
        edit_bg = Config.get_color("bg_3", "#101010")
        # Font tiers bumped up one notch across the panel (see apply_theme).
        font_small = Config.get_font_size("size_normal", 16)
        font_mini = Config.get_font_size("size_small", 12)
        self._title.set_color_hex(title_c)
        self._subtitle.set_color_hex(sub_c)
        self.setStyleSheet(
            f"QFrame#aiConvoCard {{ background: {card_bg}; border: 1px solid {border}; "
            f"border-radius: 12px; }}"
            f"QFrame#aiConvoCard:hover {{ background: {card_bg_hover}; }}"
            f'QFrame#aiConvoCard[active="true"] {{ border: 1px solid {accent}; }}'
            f"QLabel#aiConvoCardTitle {{ font-size: {font_small}px; font-weight: 600; "
            f"background: transparent; }}"
            f"QLabel#aiConvoCardSub {{ color: {sub_c}; font-size: {font_mini}px; "
            f"background: transparent; }}"
            f"QLabel#aiConvoCardTime {{ color: {sub_c}; font-size: {font_mini}px; "
            f"background: transparent; }}"
            f"QLineEdit#aiConvoCardEdit {{ background: {edit_bg}; color: {title_c}; "
            f"border: 1px solid {accent}; border-radius: 4px; padding: 1px 4px; "
            f"font-size: {font_small}px; }}"
            f"QPushButton#aiConvoCardDelete {{ background: transparent; color: {del_c}; "
            f"border: none; font-weight: bold; }}"
            f"QPushButton#aiConvoCardDelete:hover {{ color: {del_hover}; }}"
        )

    def conv_id(self) -> str:
        return self._conv_id

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and not self._edit.isVisible():
            self.clicked.emit(self._conv_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._begin_rename()
        super().mouseDoubleClickEvent(event)

    def _begin_rename(self) -> None:
        self._edit.setText(self._title.text())
        self._title.hide()
        self._edit.show()
        self._edit.setFocus(Qt.FocusReason.MouseFocusReason)
        self._edit.selectAll()

    def _commit_rename(self) -> None:
        if not self._edit.isVisible():
            return
        new = self._edit.text().strip()
        self._edit.hide()
        self._title.show()
        if new and new != self._title.text():
            self._title.setText(new)
            self.renamed.emit(self._conv_id, new)


class _GroupLabel(QLabel):
    """A small muted section header for a time bucket ("Today", "Yesterday")."""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("aiConvGroupLabel")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


class _PromptInput(QPlainTextEdit):
    """Prompt box that fills its card. Enter sends; the newline checkbox flips it.

    ``Ctrl+Enter`` always sends; ``Shift+Enter`` always inserts a newline. When
    ``newline_on_enter`` is set, a bare Enter inserts a newline and sending is
    via the Send button / Ctrl+Enter."""

    submit_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aiBuildInput")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._newline_on_enter = False

    def set_newline_on_enter(self, value: bool) -> None:
        self._newline_on_enter = bool(value)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                self.submit_requested.emit()
                event.accept()
                return
            if mods & Qt.KeyboardModifier.ShiftModifier or self._newline_on_enter:
                super().keyPressEvent(event)
                return
            self.submit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _SectionHeader(QFrame):
    """Clickable section header: title on the left, chevron on the right."""

    clicked = pyqtSignal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aiCfgSectionHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(8)
        self._title = QLabel(title, self)
        self._title.setObjectName("aiCfgSectionTitle")
        row.addWidget(self._title, 1, Qt.AlignmentFlag.AlignVCenter)
        self._chevron = QLabel("▾", self)
        self._chevron.setObjectName("aiCfgChevron")
        row.addWidget(self._chevron, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_expanded(self, expanded: bool) -> None:
        self._chevron.setText("▾" if expanded else "▸")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _CollapsibleSection(QFrame):
    """One collapsible card in the Configuration panel (Robot, Environment…).

    A clickable header (title left, chevron right) toggles a body of
    ``label  value`` rows. Built from a plain section model so the backend feeds
    typed data without touching Qt (see :meth:`AiBuildPanel.set_config_preview`).
    Colors at use-time (§5)."""

    def __init__(
        self,
        title: str,
        rows: Sequence[Tuple[str, str]],
        *,
        expanded: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("aiCfgSection")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._expanded = bool(expanded)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._header = _SectionHeader(title, self)
        self._header.clicked.connect(self._toggle)
        lay.addWidget(self._header)

        self._body = QWidget(self)
        self._body.setObjectName("aiCfgSectionBody")
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(12, 4, 12, 10)
        body_lay.setSpacing(6)
        for key, value in rows:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            k = QLabel(str(key), self._body)
            k.setObjectName("aiCfgKey")
            row.addWidget(k, 1, Qt.AlignmentFlag.AlignVCenter)
            v = QLabel(str(value), self._body)
            v.setObjectName("aiCfgValue")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(v, 0)
            body_lay.addLayout(row)
        lay.addWidget(self._body)

        self._body.setVisible(self._expanded)
        self._header.set_expanded(self._expanded)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._header.set_expanded(self._expanded)


class _ConvArea(QWidget):
    """Body area: a dimmed pure-black veil BEHIND fully-opaque cards/bubbles.

    A lowered, full-size background child (``aiConvVeil``, pure black via the
    ``ai_build_panel_bg`` slot) carries the ``QGraphicsOpacityEffect`` — the same
    reliable canvas-reveal mechanism MC uses. Only that veil is dimmed (default
    90%), so the cards + chat bubbles raised above it stay 100% opaque (crisp,
    readable — no grey wash). The slider drives the veil's opacity."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._veil = QWidget(self)
        self._veil.setObjectName("aiConvVeil")
        self._veil.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._effect = QGraphicsOpacityEffect(self._veil)
        self._effect.setOpacity(0.9)
        self._veil.setGraphicsEffect(self._effect)
        self._veil.lower()

    def veil_effect(self) -> QGraphicsOpacityEffect:
        return self._effect

    def set_veil_opacity(self, fraction: float) -> None:
        self._effect.setOpacity(max(0.0, min(1.0, fraction)))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._veil.setGeometry(0, 0, self.width(), self.height())
        self._veil.lower()


# ===========================================================================
# AiBuildPanel
# ===========================================================================
class AiBuildPanel(QWidget):
    """Full-cover AI-Build overlay: conversations + chat + config preview + banner."""

    #: Emitted when the user submits a prompt. Carries the raw text.
    prompt_submitted = pyqtSignal(str)
    #: Emitted on "Revert canvas" — undo the canvas to the pre-prompt checkpoint.
    reset_requested = pyqtSignal()
    #: Emitted on the gear — open the API settings dialog.
    settings_requested = pyqtSignal()
    #: Emitted when the Pause button toggles (True = pause requested).
    pause_toggled = pyqtSignal(bool)
    #: Emitted on "Retry" — re-run the last prompt from the pre-prompt checkpoint.
    retry_requested = pyqtSignal()
    #: BACKEND TODO — Configuration-Preview source combobox changed. Payload is
    #: the source key ("origin" or a chat/config id). No consumer yet.
    config_source_changed = pyqtSignal(str)
    #: BACKEND TODO — "Save Config" pressed (apply the previewed config to the
    #: canvas / persist it). No consumer yet.
    config_save_requested = pyqtSignal()

    _TOP_ROW_H = 43
    # The two side cards (Chats list / Config preview) sit at their preferred
    # width when the panel is wide, but shrink toward their min when the panel
    # is narrow so the middle chat column — where the bubbles live — keeps at
    # least ``_CHAT_MIN_WIDTH`` and never collapses to a sliver. Recomputed in
    # ``resizeEvent`` (``_relayout_columns``).
    _LIST_WIDTH = 268           # Chats card preferred/max width
    _LIST_MIN_WIDTH = 180       # …shrinks to this when the panel is tight
    _CONFIG_WIDTH = 320         # Config preview card preferred/max width
    _CONFIG_MIN_WIDTH = 200     # …shrinks to this when the panel is tight
    _CHAT_MIN_WIDTH = 420       # chat column floor — bubbles widen up to it first
    _BODY_MARGIN = 12           # _build_ui body contentsMargins (per side)
    _BODY_GAP = 12              # _build_ui body spacing between the 3 columns
    #: Width of the safe_zone left-accent stripe on section titles — reused
    #: verbatim from MissionControlPanel's card title decoration.
    _ACCENT_W = 4
    #: 12px rounded cards (Chats / Configuration), per the design.
    _CARD_RADIUS = 12
    #: Fixed body-veil opacity (fraction) — the canvas shows faintly through the
    #: background behind the opaque cards. No user slider.
    _VEIL_OPACITY = 0.9

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aiBuildPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Height of the host overlay strip covering the canvas top (the
        # MissionControlPanel top_row in TC mode); the mask starts below it so
        # its own top_row isn't hidden. Set by the host via set_top_inset.
        self._top_inset: int = 0

        # The body area's black veil sits at a fixed opacity (behind the opaque
        # cards/bubbles) so the canvas shows through the BACKGROUND only, exactly
        # like MC's chart. The opacity is fixed (no user slider).
        self._conv_area: Optional[_ConvArea] = None
        self._opacity_effect: Optional[QGraphicsOpacityEffect] = None

        # ---- conversation state ------------------------------------------
        self._canvas_id: str = ""
        self._conv_set: Optional[convstore.ConversationSet] = None
        self._convo_rows: Dict[str, _ConvoCard] = {}
        self._search_text: str = ""
        # The in-progress agent turn (the store Message). The chat zone owns the
        # matching card; the panel keeps the phase/token state that drives it.
        self._current_agent_msg: Optional[convstore.Message] = None
        # Title + token count for the active phase (begin_phase → set_phase_tokens).
        self._current_phase_title: str = ""
        # Tokens to stamp on the next agent card if set_phase_tokens fires before
        # it opens (e.g. the Summary phase logs its text after the token report).
        self._pending_tokens: int = 0

        # ---- robot / backend context (fed by the page) — shown in the
        # Configuration card (was the bottom banner, now removed) -----------
        self._ctx_robot_sku: str = ""
        self._ctx_backend: str = ""

        # ---- widget slots ------------------------------------------------
        self._rows_layout: Optional[QVBoxLayout] = None
        self._rows_host: Optional[QWidget] = None
        self._search_input: Optional[QLineEdit] = None
        self._chat_zone: Optional[BotChatZone] = None
        self._chat_title: Optional[_ElidedLabel] = None
        self._progress: Optional[QProgressBar] = None
        self._input: Optional[_PromptInput] = None
        self._newline_chk: Optional[QCheckBox] = None
        self._submit_btn: Optional[QPushButton] = None
        self._reset_btn: Optional[QPushButton] = None
        # Run controls + token meter (chat header).
        self._pause_btn: Optional[QPushButton] = None
        self._retry_btn: Optional[QPushButton] = None
        self._tokens_total_lbl: Optional[QLabel] = None
        # Configuration card — collapsible sections (Robot / Engine / Training
        # Ground / Environment / Rewards). ``_config_from_backend`` flips once a
        # real projection is pushed via set_config_preview so the default
        # skeleton stops re-seeding on context changes.
        self._config_combo: Optional[QComboBox] = None
        self._cfg_layout: Optional[QVBoxLayout] = None
        self._cfg_host: Optional[QWidget] = None
        self._cfg_placeholder: Optional[QLabel] = None
        self._save_cfg_btn: Optional[QPushButton] = None
        self._config_from_backend: bool = False
        # The two side cards — widths recomputed responsively in resizeEvent so
        # the middle chat column keeps a usable floor on narrow panels.
        self._conv_card: Optional[QFrame] = None
        self._cfg_card: Optional[QFrame] = None

        self._build_ui()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_top_row(root)

        # Body — three columns over the opacity veil (cards opaque; the gaps
        # between them reveal the dimmed canvas, exactly like MC's chart).
        conv_area = _ConvArea(self)
        conv_area.setObjectName("aiConvArea")
        body = QHBoxLayout(conv_area)
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)
        body.addWidget(self._build_conversations_card(conv_area), 0)
        body.addWidget(self._build_chat_column(conv_area), 1)
        body.addWidget(self._build_config_card(conv_area), 0)
        self._conv_area = conv_area
        self._opacity_effect = conv_area.veil_effect()
        conv_area.set_veil_opacity(self._VEIL_OPACITY)
        root.addWidget(conv_area, 1)

    def _make_stripe_title(self, key: str, default: str, parent: QWidget) -> QFrame:
        """A Mission-Control-style section title: a transparent bar carrying a
        safe_zone ``border-left`` accent stripe + a ``main_t1`` label (the
        "safe_zone title decoration" reused verbatim from the MC cards)."""
        host = QFrame(parent)
        host.setObjectName("aiStripeTitle")
        host.setFrameShape(QFrame.Shape.NoFrame)
        host.setFixedHeight(22)
        host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(host)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)
        label = QLabel(host)
        label.setObjectName("aiStripeTitleLabel")
        i18n_bind(label, "setText", key, default=default)
        lay.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        return host

    def _build_top_row(self, root: QVBoxLayout) -> None:
        bar = QFrame(self)
        bar.setObjectName("aiBuildTopRow")
        bar.setFixedHeight(self._TOP_ROW_H)
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(bar)
        # 2px top/bottom margin so the (square) close button fills the row.
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(8)

        # Agent button (replaces both the old gear icon and the static "AI Build"
        # title): shows the active agent's model name, or a "set up" prompt when
        # no connection is saved. Clicking opens the API-settings dialog (the
        # agent registry) — the single entry point now that the gear is gone.
        self._agent_btn = QPushButton(bar)
        self._agent_btn.setObjectName("aiBuildAgentBtn")
        self._agent_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._agent_btn, "setToolTip",
            "canvas.ai_build.agent_tip", default="Configure or switch the AI agent",
        )
        self._agent_btn.clicked.connect(lambda: self.settings_requested.emit())
        row.addWidget(self._agent_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.refresh_agent_button()

        row.addStretch(1)

        # The AI Build panel is a first-class tab (AI Coding) in the Mission
        # Control top-row slider — switching tabs opens / collapses it, so the
        # old right-aligned close (×) button was removed as redundant.

        root.addWidget(bar, 0)

    def refresh_agent_button(self) -> None:
        """Re-read the active agent's model and label the top-row button.

        Empty model = the documented "not configured yet" state → the button
        shows the set-up prompt (mirrors ``load_api_config``'s blank default;
        not a §8 silent fallback for a required input). A registry-not-ready
        error is logged and treated as the same empty state (same pattern as
        ``_resolve_robot`` / ``_resolve_backend``)."""
        btn = getattr(self, "_agent_btn", None)
        if btn is None:
            return
        model = ""
        try:
            from application.service.ai_orchestration.api_config import active_model_name

            model = active_model_name()
        except Exception as exc:  # registry / config not ready — show set-up prompt
            log_warning(f"[ai.panel] active agent lookup failed: {exc!r}")
        if model:
            btn.setText(model)
            btn.setProperty("configured", "true")
        else:
            btn.setText(tr("canvas.ai_build.agent_setup", "< Set Up Your Agent >"))
            btn.setProperty("configured", "false")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    # ---- left column: Conversations card ------------------------------
    def _build_conversations_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("aiConvCard")
        # Width is driven by ``_relayout_columns`` (responsive); start at the
        # preferred width so the first paint (before any resize) looks right.
        card.setFixedWidth(self._LIST_WIDTH)
        self._conv_card = card
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Header: green-stripe title (kept) + "+ New" button (per the design).
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(
            self._make_stripe_title("canvas.ai_build.conv_title", "Chats", card),
            1,
        )
        self._new_btn = QPushButton(card)
        self._new_btn.setObjectName("aiConvoAdd")
        self._new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._new_btn, "setText",
            "canvas.ai_build.new_conversation", default="+ New",
        )
        i18n_bind(
            self._new_btn, "setToolTip",
            "canvas.ai_build.new_conversation_tip", default="New conversation",
        )
        self._new_btn.clicked.connect(self._on_new_conversation)
        header.addWidget(
            self._new_btn, 0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        lay.addLayout(header, 0)

        # Search box.
        self._search_input = QLineEdit(card)
        self._search_input.setObjectName("aiConvSearch")
        self._search_input.setClearButtonEnabled(True)
        i18n_bind(
            self._search_input, "setPlaceholderText",
            "canvas.ai_build.search_ph", default="Search conversations…",
        )
        self._search_input.textChanged.connect(self._on_search_changed)
        lay.addWidget(self._search_input, 0)

        # Time-grouped card list.
        rows_scroll = QScrollArea(card)
        rows_scroll.setObjectName("aiConvScroll")
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rows_host = QWidget()
        self._rows_host.setObjectName("aiConvRowsHost")
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_layout.addStretch(1)
        rows_scroll.setWidget(self._rows_host)
        lay.addWidget(rows_scroll, 1)
        return card

    # ---- middle column: chat ------------------------------------------
    def _build_chat_column(self, parent: QWidget) -> QWidget:
        col = QWidget(parent)
        col.setObjectName("aiChatColumn")
        col.setAutoFillBackground(False)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Header: active-thread title + run controls (tokens / pause / retry).
        header = QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(8)
        self._chat_title = _ElidedLabel("", col)
        self._chat_title.setObjectName("aiChatTitle")
        header.addWidget(self._chat_title, 1)

        self._tokens_total_lbl = QLabel(col)
        self._tokens_total_lbl.setObjectName("aiTokensTotal")
        i18n_bind(
            self._tokens_total_lbl, "setText",
            "canvas.ai_build.tokens_total", default="Tokens: 0",
        )
        header.addWidget(self._tokens_total_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        self._pause_btn = QPushButton(col)
        self._pause_btn.setObjectName("aiPauseBtn")
        self._pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause_btn.setCheckable(True)
        i18n_bind(self._pause_btn, "setText", "canvas.ai_build.pause", default="Pause")
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        header.addWidget(self._pause_btn, 0)

        self._retry_btn = QPushButton(col)
        self._retry_btn.setObjectName("aiRetryBtn")
        self._retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(self._retry_btn, "setText", "canvas.ai_build.retry", default="Retry")
        self._retry_btn.clicked.connect(lambda: self.retry_requested.emit())
        header.addWidget(self._retry_btn, 0)
        lay.addLayout(header, 0)

        # Thin run-progress bar under the header.
        self._progress = QProgressBar(col)
        self._progress.setObjectName("aiBuildProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        lay.addWidget(self._progress, 0)

        # Chat surface — user/assistant cards (transparent so the dimmed veil
        # shows behind them). Owned by BotChatZone (specialised chat widget).
        self._chat_zone = BotChatZone(col)
        lay.addWidget(self._chat_zone, 1)

        # Prompt input card (quick-action chips live in its Send-button row).
        lay.addWidget(self._build_input_card(col), 0)
        return col

    def _build_input_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("aiInputCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self._input = _PromptInput(card)
        self._input.setFixedHeight(72)
        i18n_bind(
            self._input, "setPlaceholderText",
            "canvas.ai_build.input_placeholder",
            default="Ask AI to generate configs, scripts, or training strategies…",
        )
        self._input.submit_requested.connect(self._on_submit)
        self._input.textChanged.connect(self._sync_submit_enabled)
        lay.addWidget(self._input, 1)

        # Controls row: quick-action chips (left, light/borderless) … then the
        # newline checkbox + Revert + Send (right). 10px spacing throughout.
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)

        # Quick-action chips — front-end prompt presets that pre-fill the input.
        # BACKEND TODO: a backend pass may replace these with context-aware
        # suggestions.
        chips = [
            ("canvas.ai_build.chip_ppo", "Generate PPO config"),
            ("canvas.ai_build.chip_reward", "Add reward function"),
            ("canvas.ai_build.chip_tune", "Tune hyperparameters"),
            ("canvas.ai_build.chip_debug", "Debug training"),
        ]
        for key, default in chips:
            chip = QPushButton(card)
            chip.setObjectName("aiChip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            i18n_bind(chip, "setText", key, default=default)
            chip.clicked.connect(
                lambda _checked=False, k=key, d=default: self._on_chip_clicked(k, d)
            )
            controls.addWidget(chip, 0)

        controls.addStretch(1)

        self._newline_chk = QCheckBox(card)
        self._newline_chk.setObjectName("aiNewlineChk")
        i18n_bind(
            self._newline_chk, "setText",
            "canvas.ai_build.newline", default="Enter for newline",
        )
        self._newline_chk.toggled.connect(self._input.set_newline_on_enter)
        controls.addWidget(self._newline_chk, 0)

        # Revert canvas to the pre-prompt checkpoint (moved out of the removed
        # Requirements card; still drives ``reset_requested``).
        self._reset_btn = QPushButton(card)
        self._reset_btn.setObjectName("aiResetBtn")
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._reset_btn, "setText", "canvas.ai_build.reset", default="Revert canvas",
        )
        i18n_bind(
            self._reset_btn, "setToolTip", "canvas.ai_build.reset_tip",
            default="Revert the canvas to the checkpoint taken before your last prompt.",
        )
        self._reset_btn.clicked.connect(lambda: self.reset_requested.emit())
        controls.addWidget(self._reset_btn, 0)

        self._submit_btn = QPushButton(card)
        self._submit_btn.setObjectName("aiSubmitBtn")
        self._submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._submit_btn, "setText", "canvas.ai_build.submit", default="Send",
        )
        self._submit_btn.clicked.connect(self._on_submit)
        controls.addWidget(self._submit_btn, 0)
        lay.addLayout(controls, 0)
        self._sync_submit_enabled()
        return card

    def _on_chip_clicked(self, key: str, default: str) -> None:
        if self._input is None:
            return
        self._input.setPlainText(tr(key, default))
        self.focus_input()

    # ---- right column: Configuration Preview --------------------------
    def _build_config_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("aiCfgCard")
        # Width is driven by ``_relayout_columns`` (responsive); start at the
        # preferred width so the first paint (before any resize) looks right.
        card.setFixedWidth(self._CONFIG_WIDTH)
        self._cfg_card = card
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Header: green-stripe title + source combobox (replaces the old
        # "Validate" button — switches between the Origin canvas config and any
        # chat-generated config snapshot).
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(
            self._make_stripe_title(
                "canvas.ai_build.config_title", "Configuration", card
            ),
            1,
        )
        self._config_combo = QComboBox(card)
        self._config_combo.setObjectName("aiCfgCombo")
        self._config_combo.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._config_combo.setMinimumWidth(110)
        # "Origin" = the live canvas config. BACKEND TODO: append one entry per
        # chat-generated config via :meth:`set_config_sources`.
        self._config_combo.addItem(tr("canvas.ai_build.config_origin", "Origin"), "origin")
        self._config_combo.currentIndexChanged.connect(self._on_config_source_changed)
        header.addWidget(self._config_combo, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(header, 0)

        # Collapsible section list (Robot / Engine / Training Ground /
        # Environment / Rewards). Populated with the default skeleton on build;
        # the page fills Robot + Engine via set_requirements_context, and a
        # backend pass replaces the whole set via set_config_preview.
        cfg_scroll = QScrollArea(card)
        cfg_scroll.setObjectName("aiCfgScroll")
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cfg_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cfg_host = QWidget()
        self._cfg_host.setObjectName("aiCfgHost")
        self._cfg_layout = QVBoxLayout(self._cfg_host)
        self._cfg_layout.setContentsMargins(0, 0, 0, 0)
        self._cfg_layout.setSpacing(8)
        self._cfg_layout.addStretch(1)
        cfg_scroll.setWidget(self._cfg_host)
        lay.addWidget(cfg_scroll, 1)

        self._render_config_default()

        # Save Config (green, full-width).
        self._save_cfg_btn = QPushButton(card)
        self._save_cfg_btn.setObjectName("aiSaveCfgBtn")
        self._save_cfg_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._save_cfg_btn, "setText",
            "canvas.ai_build.save_config", default="Save Config",
        )
        self._save_cfg_btn.clicked.connect(lambda: self.config_save_requested.emit())
        lay.addWidget(self._save_cfg_btn, 0)
        return card

    def _sync_submit_enabled(self) -> None:
        """Enable Send only when the prompt has non-whitespace text."""
        if self._submit_btn is None or self._input is None:
            return
        self._submit_btn.setEnabled(bool(self._input.toPlainText().strip()))

    # ------------------------------------------------------------------
    # Run-settings seam (Requirements card removed — defaults until the backend
    # derives these from the Configuration card context).
    # ------------------------------------------------------------------
    def requirements(self) -> Requirements:
        """No-token, user-supplied decisions for the run.

        BACKEND TODO: the Requirements card was removed in the redesign. Until a
        backend pass derives these from the Configuration card, this returns
        all-defaults ("card unused" → the agent decides everything), so the
        existing :class:`AiBuildRunTask` seam keeps working unchanged."""
        return Requirements(robot_sku=self._ctx_robot_sku)

    def max_repair_rounds(self) -> int:
        """BACKEND TODO: was a Requirements spin; default repair budget for now."""
        return 3

    def auto_commit(self) -> bool:
        """BACKEND TODO: was a Requirements checkbox; auto-save on clean compile."""
        return True

    def set_requirements_context(self, backend: str, robot_sku: str = "") -> None:
        """Feed the Configuration card's Robot + Engine sections (page-driven).

        Robot name/family + backend display name are resolved from the registry
        (cosmetic; a registry miss simply shows the raw id / em-dash)."""
        self._ctx_backend = str(backend or "")
        self._ctx_robot_sku = str(robot_sku or "")
        # Only re-seed the default skeleton while the backend hasn't pushed a
        # real config projection (which owns the whole preview once it lands).
        if not self._config_from_backend:
            self._render_config_default()

    def _resolve_robot(self) -> Tuple[str, str, str]:
        """(name, family, sku) for the context robot; em-dash where unknown."""
        none = tr("canvas.ai_build.banner_none", "—")
        sku = self._ctx_robot_sku or none
        name, family = (self._ctx_robot_sku or none), none
        if self._ctx_robot_sku:
            try:
                from registers import robots as _robots

                spec = _robots.get_robot_spec(self._ctx_robot_sku)
                if spec is not None:
                    name = spec.name or self._ctx_robot_sku
                    family = (spec.families[0].title() if spec.families else none)
            except Exception as exc:  # registry not ready — show raw id
                log_warning(f"[ai.panel] robot spec lookup failed: {exc!r}")
        return name, family, sku

    def _resolve_backend(self) -> str:
        none = tr("canvas.ai_build.banner_none", "—")
        if not self._ctx_backend:
            return none
        try:
            from registers import backends as _backends

            return _backends.get_display_name(self._ctx_backend) or self._ctx_backend
        except Exception as exc:
            log_warning(f"[ai.panel] backend name lookup failed: {exc!r}")
            return self._ctx_backend

    def _default_config_sections(
        self,
    ) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """The Configuration skeleton (Robot / Engine / Training Ground /
        Environment / Rewards). Robot + Engine carry the real page context;
        the rest are pass-through placeholders until the backend fills them.

        BACKEND TODO: a backend pass should project the live canvas / a chat
        config into these sections via :meth:`set_config_preview` — the values
        marked ``—`` are the fields awaiting that wiring."""
        none = tr("canvas.ai_build.banner_none", "—")
        name, family, sku = self._resolve_robot()
        backend = self._resolve_backend()
        return [
            (tr("canvas.ai_build.cfg_sec_robot", "Robot"), [
                (tr("canvas.ai_build.cfg_f_name", "Name"), name),
                (tr("canvas.ai_build.cfg_f_sku", "SKU"), sku),
                (tr("canvas.ai_build.cfg_f_family", "Family"), family),
            ]),
            (tr("canvas.ai_build.cfg_sec_engine", "Engine"), [
                (tr("canvas.ai_build.cfg_f_backend", "Backend"), backend),
                (tr("canvas.ai_build.cfg_f_device", "Device"), none),
            ]),
            (tr("canvas.ai_build.cfg_sec_ground", "Training Ground"), [
                (tr("canvas.ai_build.cfg_f_terrain", "Terrain"), none),
                (tr("canvas.ai_build.cfg_f_difficulty", "Difficulty"), none),
                (tr("canvas.ai_build.cfg_f_curriculum", "Curriculum"), none),
            ]),
            (tr("canvas.ai_build.cfg_sec_env", "Environment"), [
                (tr("canvas.ai_build.cfg_f_num_envs", "Num Envs"), none),
                (tr("canvas.ai_build.cfg_f_episode", "Episode Length"), none),
                (tr("canvas.ai_build.cfg_f_spacing", "Env Spacing"), none),
            ]),
            (tr("canvas.ai_build.cfg_sec_rewards", "Rewards"), [
                (tr("canvas.ai_build.cfg_f_reward_terms", "Reward Terms"), none),
            ]),
        ]

    def _render_config_default(self) -> None:
        """Render the default section skeleton (used until backend data lands)."""
        self._render_config_sections(self._default_config_sections())

    def _render_config_sections(
        self, sections: Sequence[Tuple[str, Sequence[Tuple[str, str]]]]
    ) -> None:
        """(Re)build the collapsible sections in the config scroll area."""
        lay = self._cfg_layout
        if lay is None:
            return
        while lay.count() > 1:  # keep the trailing stretch
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cfg_placeholder = None
        if not sections:
            self._show_cfg_placeholder()
            return
        for title, rows in sections:
            section = _CollapsibleSection(
                str(title), [(str(k), str(v)) for k, v in rows],
                expanded=True, parent=self._cfg_host,
            )
            lay.insertWidget(lay.count() - 1, section)

    # ------------------------------------------------------------------
    # Configuration Preview (BACKEND TODO — no producer yet)
    # ------------------------------------------------------------------
    def set_config_sources(
        self, sources: Sequence[Tuple[str, str]], *, select: Optional[str] = None
    ) -> None:
        """Populate the source combobox: ``[(key, label), …]`` after Origin.

        Fed by the page's :class:`~application.service.ai_orchestration.
        version_store.AiBuildVersionStore` — one entry per AI Build run that
        changed the canvas. ``select`` (a source key) moves the current selection
        WITHOUT emitting ``config_source_changed`` (signals stay blocked), so the
        combobox can point at the just-produced version without re-applying it.
        """
        if self._config_combo is None:
            return
        self._config_combo.blockSignals(True)
        self._config_combo.clear()
        self._config_combo.addItem(tr("canvas.ai_build.config_origin", "Origin"), "origin")
        for key, label in sources:
            self._config_combo.addItem(str(label), str(key))
        if select is not None:
            idx = self._config_combo.findData(str(select))
            if idx >= 0:
                self._config_combo.setCurrentIndex(idx)
        self._config_combo.blockSignals(False)

    def set_config_preview(
        self, sections: Sequence[Tuple[str, Sequence[Tuple[str, str]]]]
    ) -> None:
        """Render the collapsible Configuration sections from backend data.

        ``sections`` is ``[(title, [(key, value), …]), …]`` — a plain, Qt-free
        model so the backend can project the live canvas / a chat config without
        importing this widget. This takes ownership of the preview: once called
        with real sections, the default skeleton is no longer re-seeded on
        context changes. An empty list reverts to the default skeleton.

        BACKEND TODO: nothing produces this yet; until a backend pass wires a
        canvas→sections projection through here, the panel shows the default
        Robot / Engine / Training Ground / Environment / Rewards skeleton."""
        if not sections:
            self._config_from_backend = False
            self._render_config_default()
            return
        self._config_from_backend = True
        self._render_config_sections(sections)

    def _show_cfg_placeholder(self) -> None:
        if self._cfg_layout is None:
            return
        self._cfg_placeholder = QLabel(self._cfg_host)
        self._cfg_placeholder.setObjectName("aiCfgEmpty")
        i18n_bind(
            self._cfg_placeholder, "setText",
            "canvas.ai_build.config_empty",
            default="No configuration to preview yet.\nDescribe a training run to generate one.",
        )
        self._cfg_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cfg_placeholder.setWordWrap(True)
        self._cfg_layout.insertWidget(
            self._cfg_layout.count() - 1, self._cfg_placeholder, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
        )

    def _on_config_source_changed(self, _index: int) -> None:
        if self._config_combo is None:
            return
        key = str(self._config_combo.currentData() or "origin")
        # BACKEND TODO: re-render set_config_preview for ``key``. For now just
        # surface the intent so a backend can connect.
        self.config_source_changed.emit(key)

    # ------------------------------------------------------------------
    # Conversation set wiring
    # ------------------------------------------------------------------
    def set_canvas_id(self, file_id: str) -> None:
        """Bind the panel to a canvas's thread set (loads it from disk).

        Threads are per-canvas (never cross canvases). No-op when the canvas is
        unchanged so reopening the overlay does not discard in-memory state."""
        fid = str(file_id or "")
        if fid == self._canvas_id and self._conv_set is not None:
            return
        self._canvas_id = fid
        self._conv_set = convstore.load(fid)
        self._reset_turn()
        self._rebuild_list()
        self._render_active()

    def _persist(self) -> None:
        if self._conv_set is not None:
            convstore.save(self._conv_set)

    def _ensure_active(self) -> convstore.Conversation:
        if self._conv_set is None:
            self._conv_set = convstore.ConversationSet(canvas_id=self._canvas_id)
        conv = self._conv_set.active()
        if conv is None:
            conv = self._conv_set.new_conversation()
            self._rebuild_list()
        return conv

    @staticmethod
    def _derive_title(text: str) -> str:
        first = (text or "").strip().splitlines()[0] if text.strip() else ""
        first = first.strip()
        return first if len(first) <= 40 else first[:39] + "…"

    @staticmethod
    def _preview_text(conv: convstore.Conversation) -> str:
        """A one-line preview = the first user message's plain text."""
        for msg in conv.messages:
            if msg.role == convstore.USER_ROLE:
                parts = [seg[0] for line in msg.lines for seg in line]
                joined = " ".join(p.strip() for p in parts if p.strip())
                if joined:
                    return joined
        return ""

    # ------------------------------------------------------------------
    # Left rail
    # ------------------------------------------------------------------
    def _on_search_changed(self, text: str) -> None:
        self._search_text = str(text or "").strip().lower()
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lay = self._rows_layout
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._convo_rows = {}
        untitled = tr("canvas.ai_build.untitled", "New conversation")
        now = time.time()
        # Bucket conversations (already newest-first) by time group.
        grouped: Dict[str, List[convstore.Conversation]] = {
            key: [] for key, _, _ in _GROUP_ORDER
        }
        if self._conv_set is not None:
            for conv in self._conv_set.ordered():
                title = conv.title or untitled
                preview = self._preview_text(conv)
                if self._search_text and (
                    self._search_text not in title.lower()
                    and self._search_text not in preview.lower()
                ):
                    continue
                grouped[_time_group(conv.updated or conv.created, now)].append(conv)

        for key, gkey, gdefault in _GROUP_ORDER:
            members = grouped.get(key, [])
            if not members:
                continue
            lay.addWidget(_GroupLabel(tr(gkey, gdefault), self._rows_host))
            for conv in members:
                card = _ConvoCard(
                    conv.id,
                    conv.title or untitled,
                    self._preview_text(conv),
                    _fmt_time_label(conv.updated or conv.created, now),
                    parent=self._rows_host,
                )
                card.clicked.connect(self._on_row_clicked)
                card.delete_clicked.connect(self._on_row_delete)
                card.renamed.connect(self._on_row_renamed)
                card.set_active(
                    self._conv_set is not None and conv.id == self._conv_set.active_id
                )
                lay.addWidget(card)
                self._convo_rows[conv.id] = card
        lay.addStretch(1)

    def _on_new_conversation(self) -> None:
        if self._conv_set is None:
            self._conv_set = convstore.ConversationSet(canvas_id=self._canvas_id)
        active = self._conv_set.active()
        # Avoid piling up empty threads — if the active one is still empty,
        # just focus the input rather than creating another blank thread.
        if active is not None and not active.messages:
            self.focus_input()
            return
        self._conv_set.new_conversation()
        self._reset_turn()
        self._rebuild_list()
        self._render_active()
        self._persist()
        self.focus_input()

    def _on_row_clicked(self, conv_id: str) -> None:
        if self._conv_set is None or conv_id == self._conv_set.active_id:
            return
        self._conv_set.active_id = conv_id
        for cid, row in self._convo_rows.items():
            row.set_active(cid == conv_id)
        self._reset_turn()
        self._render_active()
        self._persist()

    def _on_row_delete(self, conv_id: str) -> None:
        if self._conv_set is None:
            return
        conv = self._conv_set.get(conv_id)
        if conv is not None and conv.messages:
            reply = QMessageBox.question(
                self,
                tr("canvas.ai_build.delete_title", "Delete conversation"),
                tr(
                    "canvas.ai_build.delete_body",
                    "Delete this conversation and its history? This cannot be undone.",
                ),
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._conv_set.remove(conv_id)
        self._reset_turn()
        self._rebuild_list()
        self._render_active()
        self._persist()

    def _on_row_renamed(self, conv_id: str, new_title: str) -> None:
        if self._conv_set is None:
            return
        conv = self._conv_set.get(conv_id)
        if conv is not None:
            conv.title = new_title
            self._persist()
            self._update_chat_title()

    # ------------------------------------------------------------------
    # Chat (delegated to BotChatZone)
    # ------------------------------------------------------------------
    def _render_active(self) -> None:
        self._update_chat_title()
        conv = self._conv_set.active() if self._conv_set is not None else None
        if self._chat_zone is None:
            return
        if conv is None or not conv.messages:
            self._chat_zone.clear()
            self._chat_zone.show_placeholder(
                tr(
                    "canvas.ai_build.empty_hint",
                    "Describe the training you want and I'll build the canvas.",
                )
            )
        else:
            self._chat_zone.render(conv.messages)
        self._reset_turn()

    def _update_chat_title(self) -> None:
        if self._chat_title is None:
            return
        conv = self._conv_set.active() if self._conv_set is not None else None
        if conv is not None and (conv.title or conv.messages):
            self._chat_title.setText(
                conv.title or tr("canvas.ai_build.untitled", "New conversation")
            )
        else:
            self._chat_title.setText(tr("canvas.ai_build.title", "AI Build"))

    def llm_conversation_history(self) -> List[dict]:
        """OpenAI-style history of the ACTIVE thread for the harness.

        Excludes the trailing user message when present — at submit time the
        just-recorded prompt is the last entry and the run passes it
        separately (on a Retry there is no fresh user bubble, so nothing is
        dropped). Projection (chrome-stripping, merging, caps) is the service
        layer's :func:`conversation_store.to_llm_history` — single source.
        """
        conv = self._conv_set.active() if self._conv_set is not None else None
        if conv is None:
            return []
        messages = list(conv.messages)
        if messages and messages[-1].role == convstore.USER_ROLE:
            messages = messages[:-1]
        return convstore.to_llm_history(messages)

    # ------------------------------------------------------------------
    # Public log API — the agent harness renders MutationEvents through this
    # ------------------------------------------------------------------
    def append_segments(
        self, segments: List[LogSegment], *, timestamp: bool = True
    ) -> None:
        """Append one line of typed (text, role) segments to the agent card.

        Lazily opens an agent card on the first call after a user message; a
        leading ``[HH:MM:SS]`` stamp is prepended unless ``timestamp=False``."""
        conv = self._ensure_active()
        line: convstore.Line = []
        if timestamp:
            line.append([f"[{QTime.currentTime().toString('HH:mm:ss')}] ", "time"])
        line.extend([[str(text), str(role)] for text, role in segments])

        if self._current_agent_msg is None:
            msg = convstore.Message(role="agent", lines=[], ts=time.time())
            conv.messages.append(msg)
            self._current_agent_msg = msg
            if self._pending_tokens:
                msg.tokens = self._pending_tokens
            if self._chat_zone is not None:
                self._chat_zone.begin_agent_card(
                    header=self._current_phase_title,
                    tokens=self._pending_tokens,
                    ts=msg.ts,
                )

        self._current_agent_msg.lines.append(line)
        conv.updated = time.time()
        if self._chat_zone is not None:
            self._chat_zone.set_agent_lines(self._current_agent_msg.lines)

    def append_markup(self, text: str) -> None:
        """Append one line of ``[[role:text]]``-tagged markup."""
        self.append_segments(parse_markup(text))

    def append_line(self, text: str, role: str = _DEFAULT_ROLE) -> None:
        """Append one single-role line."""
        self.append_segments([(text, role)])

    def clear_log(self) -> None:
        """Re-render the active thread (kept for backend API compatibility)."""
        self._render_active()

    def focus_input(self) -> None:
        if self._input is not None:
            self._input.setFocus(Qt.FocusReason.OtherFocusReason)

    # ------------------------------------------------------------------
    # Run settings + progress (driven by the run task)
    # ------------------------------------------------------------------
    def _reset_turn(self) -> None:
        """Forget the in-progress agent card + phase/token state."""
        self._current_agent_msg = None
        self._current_phase_title = ""
        self._pending_tokens = 0
        if self._chat_zone is not None:
            self._chat_zone.close_agent_card()

    def begin_progress(self) -> None:
        if self._progress is not None:
            self._progress.setValue(0)
        self._pending_tokens = 0
        if self._tokens_total_lbl is not None:
            self._tokens_total_lbl.setText(
                tr("canvas.ai_build.tokens_total", "Tokens: {n}").format(n=0)
            )

    def set_progress(self, fraction: float, text: str = "") -> None:
        """Drive the progress bar (0..1) from the run's real checkpoints."""
        if self._progress is None:
            return
        pct = int(max(0.0, min(1.0, float(fraction))) * 100)
        self._progress.setValue(pct)

    def begin_phase(self, phase_id: str, title: str) -> None:
        """Open a new structured phase: close the current agent card so the next
        append opens a fresh one headed by ``title`` (the fixed 3-phase format).
        Driven by the harness sink."""
        self._reset_turn()
        self._current_phase_title = str(title or "")
        self._pending_tokens = 0

    def set_phase_tokens(self, phase_total: int, cumulative: int) -> None:
        """Show this phase's token total on its card + the running total."""
        phase_total = int(phase_total or 0)
        cumulative = int(cumulative or 0)
        self._pending_tokens = phase_total
        if self._current_agent_msg is not None:
            self._current_agent_msg.tokens = phase_total
        if self._chat_zone is not None:
            self._chat_zone.set_agent_header(self._current_phase_title, phase_total)
        if self._tokens_total_lbl is not None:
            self._tokens_total_lbl.setText(
                tr("canvas.ai_build.tokens_total", "Tokens: {n}").format(n=cumulative)
            )

    def finish_progress(self) -> None:
        """Top the bar off at the end of a run and persist the agent turn."""
        if self._progress is not None:
            self._progress.setValue(100)
        if self._pause_btn is not None and self._pause_btn.isChecked():
            self._pause_btn.setChecked(False)
        self._reset_turn()
        self._persist()

    def _on_pause_toggled(self, checked: bool) -> None:
        if self._pause_btn is not None:
            self._pause_btn.setText(
                tr("canvas.ai_build.resume", "Resume") if checked
                else tr("canvas.ai_build.pause", "Pause")
            )
        self.pause_toggled.emit(bool(checked))

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    def _on_submit(self) -> None:
        if self._input is None:
            return
        text = self._input.toPlainText().strip()
        if not text:
            return
        conv = self._ensure_active()
        if not conv.title:
            conv.title = self._derive_title(text)
        msg = convstore.Message(
            role=convstore.USER_ROLE,
            lines=[[[text, convstore.USER_ROLE]]],
            ts=time.time(),
        )
        conv.messages.append(msg)
        conv.updated = time.time()
        if self._chat_zone is not None:
            self._chat_zone.add_user_message(text, ts=msg.ts)
        # Next append (the agent's reply) opens a fresh agent card.
        self._reset_turn()
        self._rebuild_list()  # the derived title may now be visible
        self._update_chat_title()
        self._persist()
        self._input.clear()
        self.prompt_submitted.emit(text)

    # ------------------------------------------------------------------
    # Geometry / theme
    # ------------------------------------------------------------------
    def set_top_inset(self, px: int) -> None:
        """Height of the host strip over the canvas top; the mask starts below it."""
        px = max(0, int(px))
        if px == self._top_inset:
            return
        self._top_inset = px
        if self.isVisible():
            self.reanchor()

    def reanchor(self) -> None:
        """Cover the parent viewport below the host top strip (big mask)."""
        parent = self.parentWidget()
        if parent is None:
            return
        inset = self._top_inset
        self.setGeometry(0, inset, parent.width(), max(0, parent.height() - inset))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Reflow the three columns first (may widen the chat column), THEN the
        # chat zone re-fits its bubble widths against the new chat-column width
        # via its own resizeEvent (fired when we resize the side cards).
        self._relayout_columns()

    def _relayout_columns(self) -> None:
        """Keep the middle chat column (bubbles) at ``_CHAT_MIN_WIDTH`` by
        shrinking the two fixed side cards toward their mins when the panel is
        narrow. When the panel is wide both side cards sit at their preferred
        width and the chat column takes all the slack (unchanged from before).

        Deterministic + §8-safe: no silent clamp that hides a bad width — the
        chat column simply gets priority for the available space, and the side
        cards yield down to a readable minimum before anything overflows.
        """
        conv = self._conv_card
        cfg = self._cfg_card
        if conv is None or cfg is None:
            return
        # Available width for the 3 columns = body width minus its outer
        # margins (both sides) and the 2 inter-column gaps.
        avail = self.width() - 2 * self._BODY_MARGIN - 2 * self._BODY_GAP
        if avail <= 0:
            return
        list_w = self._LIST_WIDTH
        cfg_w = self._CONFIG_WIDTH
        chat = avail - list_w - cfg_w
        if chat < self._CHAT_MIN_WIDTH:
            # Not enough room at preferred widths — take the deficit out of the
            # side cards, capped so neither drops below its min.
            deficit = self._CHAT_MIN_WIDTH - chat
            list_room = self._LIST_WIDTH - self._LIST_MIN_WIDTH
            cfg_room = self._CONFIG_WIDTH - self._CONFIG_MIN_WIDTH
            total_room = list_room + cfg_room
            if total_room > 0:
                take = min(deficit, total_room)
                # Split the shrink proportionally to each card's available room.
                list_w = self._LIST_WIDTH - round(take * list_room / total_room)
                cfg_w = self._CONFIG_WIDTH - (take - (self._LIST_WIDTH - list_w))
        if conv.width() != list_w:
            conv.setFixedWidth(list_w)
        if cfg.width() != cfg_w:
            cfg.setFixedWidth(cfg_w)

    def apply_theme(self) -> None:
        """Re-read every color slot and restyle (§5: colors at use-time)."""
        bg_main = Config.get_color("bg_1", "#1E1E1E")
        bg_alt = Config.get_color("bg_2", "#1A1A1A")
        bg_deep = Config.get_color("bg_3", "#101010")
        border = Config.get_color("border_1", "#444444")
        title_color = Config.get_color("main_t1", "#D6D3C7")
        field_bg = Config.get_color("bg_1", "#1E1E1E")
        text = Config.get_color("main_t1", "#D6D3C7")
        label_color = Config.get_color("sub_t2", "#777777")
        sub_color = Config.get_color("sub_t1", "#A0A0A0")
        accent = Config.get_color("safe_zone", "#36E38E")
        accent_hover = Config.get_color("safe_zone_hover", "#19B86B")
        accent_text = Config.get_color("alt_t1", "#1E1E1E")
        progress_track = Config.get_color("opacity_slider_track", "#2A2C33")
        veil = Config.get_color("ai_build_panel_bg", "#000000")
        # Font tiers bumped up one notch across the panel (per design): the
        # former mini/small/normal now read as small/normal/large.
        sz_title = Config.get_font_size("size_large", 18)
        sz_body = Config.get_font_size("size_normal", 16)
        sz_mini = Config.get_font_size("size_small", 12)
        stripe = self._ACCENT_W
        radius = self._CARD_RADIUS

        self.setStyleSheet(
            # Root transparent — the opacity effect over the body reveals canvas.
            f"QWidget#aiBuildPanel {{ background: transparent; }}"
            f"QFrame#aiBuildTopRow {{ background-color: {bg_main}; "
            f"border-bottom: 1px solid {border}; }}"
            # Agent button (top-row): the active model name, or the set-up prompt
            # when unconfigured. Plain neutral button (no accent pill) so it blends
            # with the panel chrome; the configured state reads as normal text vs a
            # muted set-up prompt, not an accent outline.
            f"QPushButton#aiBuildAgentBtn {{ color: {sub_color}; "
            f"background: transparent; border: 1px solid {border}; "
            f"border-radius: 6px; padding: 4px 14px; font-size: {sz_title}px; "
            f"font-weight: 600; }}"
            f'QPushButton#aiBuildAgentBtn[configured="true"] {{ color: {text}; }}'
            f"QPushButton#aiBuildAgentBtn:hover {{ background-color: {bg_alt}; }}"
            # safe_zone left-accent stripe title (MC card title decoration).
            f"QFrame#aiStripeTitle {{ background: transparent; "
            f"border-left: {stripe}px solid {accent}; }}"
            f"QLabel#aiStripeTitleLabel {{ color: {title_color}; "
            f"font-size: {sz_title}px; font-weight: 600; background: transparent; }}"
            # Body: lowered pure-black veil (aiConvVeil) carries the opacity
            # effect; the cards + transparent chat area sit above it.
            f"QWidget#aiConvArea {{ background: transparent; }}"
            f"QWidget#aiConvVeil {{ background-color: {veil}; }}"
            f"QWidget#aiChatColumn {{ background: transparent; }}"
            # 12px rounded Conversations + Configuration cards.
            f"QFrame#aiConvCard, QFrame#aiCfgCard {{ background-color: {bg_alt}; "
            f"border: 1px solid {border}; border-radius: {radius}px; }}"
            f"QWidget#aiConvRowsHost, QScrollArea#aiConvScroll, "
            f"QWidget#aiCfgHost, QScrollArea#aiCfgScroll {{ background: transparent; }}"
            # "+ New" — plain neutral button (no accent outline/fill) to match the
            # rest of the panel chrome.
            f"QPushButton#aiConvoAdd {{ background: transparent; color: {text}; "
            f"border: 1px solid {border}; border-radius: 6px; padding: 3px 12px; "
            f"font-size: {sz_body}px; font-weight: 600; }}"
            f"QPushButton#aiConvoAdd:hover {{ background-color: {bg_alt}; }}"
            # Search box.
            f"QLineEdit#aiConvSearch {{ background-color: {bg_deep}; color: {text}; "
            f"border: 1px solid {border}; border-radius: 8px; padding: 5px 10px; "
            f"font-size: {sz_body}px; }}"
            f"QLineEdit#aiConvSearch:focus {{ border-color: {accent}; }}"
            # Time-group header.
            f"QLabel#aiConvGroupLabel {{ color: {label_color}; font-size: {sz_mini}px; "
            f"font-weight: 600; background: transparent; padding: 4px 2px 0px 2px; }}"
            # Chat-column title (the chat surface itself is styled by BotChatZone).
            f"QLabel#aiChatTitle {{ color: {title_color}; font-size: {sz_title}px; "
            f"font-weight: 600; background: transparent; }}"
            # Run progress bar.
            f"QProgressBar#aiBuildProgress {{ background-color: {progress_track}; "
            f"border: none; border-radius: 2px; }}"
            f"QProgressBar#aiBuildProgress::chunk {{ background-color: {accent}; "
            f"border-radius: 2px; }}"
            # Input card.
            f"QFrame#aiInputCard {{ background-color: {bg_alt}; border: 1px solid {border}; "
            f"border-radius: {radius}px; }}"
            f"QPlainTextEdit#aiBuildInput {{ background-color: {bg_deep}; color: {text}; "
            f"border: 1px solid {border}; border-radius: 8px; padding: 6px 8px; "
            f"font-size: {sz_body}px; }}"
            f"QPlainTextEdit#aiBuildInput:focus {{ border-color: {accent}; }}"
            f"QCheckBox#aiNewlineChk {{ color: {label_color}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            f"QPushButton#aiResetBtn {{ background-color: {bg_main}; color: {text}; "
            f"border: 1px solid {border}; border-radius: 6px; padding: 6px 12px; "
            f"font-size: {sz_body}px; }}"
            f"QPushButton#aiResetBtn:hover {{ border-color: {accent}; color: {accent}; }}"
            f"QPushButton#aiSubmitBtn {{ background-color: {accent}; color: {accent_text}; "
            f"border: none; border-radius: 8px; padding: 7px 20px; "
            f"font-size: {sz_body}px; font-weight: 600; }}"
            f"QPushButton#aiSubmitBtn:hover {{ background-color: {accent_hover}; }}"
            f"QPushButton#aiSubmitBtn:disabled {{ background-color: {field_bg}; "
            f"color: {label_color}; }}"
            # Quick chips — light/borderless prompt presets (left of the Send row).
            f"QPushButton#aiChip {{ background: transparent; color: {sub_color}; "
            f"border: none; padding: 4px 6px; font-size: {sz_mini}px; }}"
            f"QPushButton#aiChip:hover {{ color: {accent}; }}"
            # Token meter + pause / retry (chat header).
            f"QLabel#aiTokensTotal {{ color: {label_color}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            f"QPushButton#aiPauseBtn, QPushButton#aiRetryBtn {{ background-color: {bg_main}; "
            f"color: {text}; border: 1px solid {border}; border-radius: 6px; "
            f"padding: 5px 12px; font-size: {sz_body}px; }}"
            f"QPushButton#aiPauseBtn:hover, QPushButton#aiRetryBtn:hover {{ "
            f"border-color: {accent}; color: {accent}; }}"
            f"QPushButton#aiPauseBtn:checked {{ background-color: {accent}; "
            f"color: {accent_text}; border-color: {accent}; }}"
            # Configuration Preview combo + sections + save button.
            f"QComboBox#aiCfgCombo {{ background-color: {bg_deep}; color: {text}; "
            f"border: 1px solid {border}; border-radius: 6px; padding: 3px 8px; "
            f"font-size: {sz_body}px; }}"
            f"QComboBox#aiCfgCombo QAbstractItemView {{ background-color: {bg_deep}; "
            f"color: {text}; selection-background-color: {accent}; "
            f"selection-color: {accent_text}; }}"
            f"QLabel#aiCfgEmpty {{ color: {label_color}; font-size: {sz_body}px; "
            f"background: transparent; padding: 16px; }}"
            f"QFrame#aiCfgSection {{ background-color: {bg_main}; border: 1px solid {border}; "
            f"border-radius: 8px; }}"
            # Section header: title (left) + chevron (right), whole bar clickable.
            f"QFrame#aiCfgSectionHeader {{ background: transparent; }}"
            f"QFrame#aiCfgSectionHeader:hover {{ background-color: {bg_deep}; "
            f"border-top-left-radius: 8px; border-top-right-radius: 8px; }}"
            f"QLabel#aiCfgSectionTitle {{ color: {title_color}; "
            f"font-size: {sz_body}px; font-weight: 600; background: transparent; }}"
            f"QLabel#aiCfgChevron {{ color: {label_color}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            f"QWidget#aiCfgSectionBody {{ background: transparent; }}"
            f"QLabel#aiCfgKey {{ color: {label_color}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            f"QLabel#aiCfgValue {{ color: {text}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            f"QPushButton#aiSaveCfgBtn {{ background-color: {accent}; color: {accent_text}; "
            f"border: none; border-radius: 8px; padding: 9px 18px; "
            f"font-size: {sz_body}px; font-weight: 600; }}"
            f"QPushButton#aiSaveCfgBtn:hover {{ background-color: {accent_hover}; }}"
        )
        # Refresh the chat surface's own palette, then rebuild rows + re-render
        # the thread so every use-time color tracks the theme.
        if self._chat_zone is not None:
            self._chat_zone.apply_theme()
        self._rebuild_list()
        self._render_active()
        # Re-seed the default config skeleton unless a backend projection owns it.
        if not self._config_from_backend:
            self._render_config_default()
        # Re-label the agent button (active model may have changed).
        self.refresh_agent_button()


__all__ = ["AiBuildPanel", "LogSegment", "parse_markup"]
