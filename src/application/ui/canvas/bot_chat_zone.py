# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BotChatZone — the user/assistant chat surface of the AI Build overlay.

Extracted from :mod:`application.ui.canvas.ai_build_panel` so the chat itself
can be specialised without dragging the whole panel along. Instead of narrow
bubbles this renders **chat cards** (see ``界面设计.png``):

    user  ─────────────────────────────────────────────  ┌──────────────────┐
                                                          │ name  time   [U] │  (right-aligned)
                                                          │ your text …      │
                                                          └──────────────────┘
    ┌──────────────────────────────────────────────┐
    │ [🤖] AI Assistant  17:07:54                    │  (left-aligned)
    │ narration text …                               │
    │ ▸ algorithm     ┐ collapsible JSON sectors —   │
    │ ▸ training      ┘ one bar per top-level key     │
    └──────────────────────────────────────────────┘

The zone is a pure **rendering widget**: the panel owns the conversation store
(the source of truth) and drives the zone through a small API
(:meth:`render` / :meth:`add_user_message` / :meth:`begin_agent_card` /
:meth:`set_agent_lines` / :meth:`set_agent_header` / :meth:`close_agent_card`).

JSON handling: an agent line whose full text parses as a JSON object/array is
rendered as collapsible "sector" bars (one per top-level key) instead of a plain
text line — so a config the backend streams back reads as a tidy, collapsible
panel that folds away for long content. Ordinary typed-log narration renders as
before (rich text with the AI-Build role colors).

§5 compliance: every color is read at use-time from ``Config.get_color`` against
a ``system.ini[Theme]`` slot; fonts via ``Config.get_font_size``. UI strings go
through ``tr`` / ``i18n_bind``.
"""
from __future__ import annotations

import ast
import json
import re
import time
from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Assets, Config, log_warning, tr

from application.service.ai_orchestration import conversation_store as convstore

#: One log segment = (text, role). ``role`` selects the color slot.
LogSegment = Tuple[str, str]

#: role → (Theme slot, fallback hex). The normal / warning / error / success
#: roles reuse the shared CMD Log slots; node / param / value are AI-Build tones.
_ROLE_SLOT = {
    "time": ("log_time", "#20CBD4"),
    "node": ("ai_log_node", "#569CD6"),
    "param": ("ai_log_param", "#9CDCFE"),
    "value": ("ai_log_value", "#B5CEA8"),
    "info": ("log_info", "#FFFFFF"),
    "warning": ("log_warning", "#D9AB41"),
    "error": ("log_error", "#FF6B6B"),
    "success": ("log_success", "#2CD620"),
}
_DEFAULT_ROLE = "info"

#: Inside an agent card these roles keep their typed color (semantic tokens /
#: status); everything else renders in the card's base text color so ordinary
#: narration reads cleanly.
_AGENT_TYPED_ROLES = frozenset({"node", "param", "value", "warning", "error", "success"})

#: Inline role markup used by :func:`parse_markup`:
#: ``Set [[param:track_lin_vel_xy]] weight to [[value:10.0]]``. Untagged text
#: falls back to the ``info`` role. Matching is non-greedy up to the first ``]]``.
_TAG_RE = re.compile(r"\[\[(\w+):(.*?)\]\]")


def _escape_html(text: str) -> str:
    """Escape HTML metacharacters (spaces preserved so cards word-wrap)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_markup(text: str) -> List[LogSegment]:
    """Split ``[[role:text]]``-tagged markup into typed segments.

    Untagged spans become ``info`` segments. This is the transport the agent
    harness uses to render one MutationEvent / status line as colored spans.
    """
    segments: List[LogSegment] = []
    cursor = 0
    for match in _TAG_RE.finditer(text):
        if match.start() > cursor:
            segments.append((text[cursor:match.start()], _DEFAULT_ROLE))
        segments.append((match.group(2), match.group(1)))
        cursor = match.end()
    if cursor < len(text):
        segments.append((text[cursor:], _DEFAULT_ROLE))
    return segments


def _bubble_color(role: str, agent: bool) -> str:
    if not agent:
        return Config.get_color("alt_t1", "#1E1E1E")
    if role in _AGENT_TYPED_ROLES:
        slot, fallback = _ROLE_SLOT.get(role, _ROLE_SLOT[_DEFAULT_ROLE])
        return Config.get_color(slot, fallback)
    return Config.get_color("main_t1", "#D6D3C7")


def _line_plain_text(line: convstore.Line) -> str:
    """Join a line's segment texts into plain text."""
    return "".join(str(seg[0]) for seg in line)


def _lines_plain_text(lines: convstore.Lines) -> str:
    """Join a message's lines into newline-separated plain text (user cards)."""
    return "\n".join(_line_plain_text(line) for line in lines)


def _lines_to_html(lines: convstore.Lines, agent: bool) -> str:
    parts: List[str] = []
    for line in lines:
        spans = "".join(
            f'<span style="color:{_bubble_color(role, agent)};">'
            f"{_escape_html(text)}</span>"
            for text, role in line
        )
        parts.append(spans or "&nbsp;")
    return "<br>".join(parts)


def _fmt_ts(ts: float) -> str:
    """``HH:MM:SS`` local time, or empty when unknown (ts <= 0)."""
    if not ts or ts <= 0:
        return ""
    lt = time.localtime(ts)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}"


def _measure_width(px: int, text: str) -> int:
    """Natural (single-line) pixel width of the widest line of ``text``.

    Measured with a ``QFont`` built from the ``[Font]`` family at ``px`` — QSS
    ``font-size`` does NOT update ``widget.font()``, so measuring the widget's
    own font would under-count and the card would size too narrow."""
    if not text:
        return 0
    font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
    font.setPixelSize(int(px))
    fm = QFontMetrics(font)
    return max((fm.horizontalAdvance(ln) for ln in str(text).split("\n")), default=0)


def _px_body() -> int:
    return int(Config.get_font_size("size_normal", 16))


def _px_mini() -> int:
    return int(Config.get_font_size("size_small", 12))


def _current_username() -> str:
    """Human username for the chat (local-part of the account, else 'User')."""
    name = ""
    try:
        from application.service.user_workspace import read_active_username

        name = read_active_username() or ""
    except Exception as exc:  # user_workspace / config not ready — fall back
        log_warning(f"[ai.chat] username unavailable: {exc!r}")
        name = ""
    name = name.split("@")[0] if "@" in name else name
    return name.strip() or tr("canvas.ai_build.user_name", "User")


# ---------------------------------------------------------------------------
# JSON → collapsible-sector model
# ---------------------------------------------------------------------------
#: A connector segment (" = ", ": ") sitting just before a struct payload — it
#: is dropped from the narration so "Set x = {…} on 1012" reads "Set x on 1012".
_CONNECTOR_RE = re.compile(r"\s*[=:]\s*")


def _try_parse_struct(text: str):
    """Return the parsed object if ``text`` is a dict/list literal, else None.

    Accepts both JSON (double quotes / ``true`` / ``null``) and Python-``repr``
    dicts (single quotes / ``True`` / ``None``) — the harness logs the latter
    when it echoes a ``set_param`` value (e.g. ``Set training_items = {…}``)."""
    s = text.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, (dict, list)):
            return obj
    except (ValueError, TypeError):
        pass
    try:
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return obj if isinstance(obj, (dict, list)) else None


def _fmt_scalar(value) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def json_to_sections(obj) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Project a parsed JSON object into ``[(title, [(key, value), …]), …]``.

    Each top-level entry becomes one collapsible sector. A dict value expands
    to its own key/value rows; a scalar/list value renders as a single row."""
    sections: List[Tuple[str, List[Tuple[str, str]]]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                rows = [(str(k), _fmt_scalar(v)) for k, v in value.items()]
            else:
                rows = [("", _fmt_scalar(value))]
            sections.append((str(key), rows))
    elif isinstance(obj, list):
        rows = [(str(i), _fmt_scalar(v)) for i, v in enumerate(obj)]
        sections.append((tr("canvas.ai_build.json_items", "items"), rows))
    return sections


# ===========================================================================
# Small widgets
# ===========================================================================
class _ElidedLabel(QLabel):
    """A left-aligned label that elides on the right and never forces width."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full = text or ""
        self._color = "#D6D3C7"
        super().setText(self._full)
        self.setToolTip(self._full)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt signature)
        self._full = text or ""
        super().setText(self._full)
        self.setToolTip(self._full)
        self.update()

    def text(self) -> str:  # noqa: N802
        return self._full

    def set_color_hex(self, color: str) -> None:
        if color:
            self._color = color
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setPen(QColor(self._color))
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(
            self._full, Qt.TextElideMode.ElideRight, self.width()
        )
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )


class _Avatar(QLabel):
    """A small round avatar: an ``icon_*`` glyph or a single letter on a disc."""

    _SIZE = 30

    def __init__(
        self,
        *,
        icon: Optional[str] = None,
        letter: str = "",
        object_name: str = "aiAvatar",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if icon:
            path = Assets.find_icon(icon)
            if path is not None:
                pm = QPixmap(str(path))
                if not pm.isNull():
                    inner = int(self._SIZE * 0.58)
                    self.setPixmap(
                        pm.scaled(
                            inner, inner,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    return
        # Letter fallback (also the user avatar).
        self.setText((letter or "?")[:1].upper())


class _JsonSection(QFrame):
    """One collapsible JSON sector: a header bar toggling a body of rows."""

    def __init__(
        self,
        title: str,
        rows: Sequence[Tuple[str, str]],
        *,
        expanded: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("aiJsonSection")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._expanded = bool(expanded)
        self._title = title
        self._rows = list(rows)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._header = QPushButton(self)
        self._header.setObjectName("aiJsonSectionHeader")
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self._toggle)
        lay.addWidget(self._header)

        self._body = QWidget(self)
        self._body.setObjectName("aiJsonSectionBody")
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(12, 2, 10, 8)
        body_lay.setSpacing(4)
        for key, value in rows:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            if key:
                k = QLabel(str(key), self._body)
                k.setObjectName("aiJsonKey")
                row.addWidget(k, 0, Qt.AlignmentFlag.AlignTop)
            v = QLabel(str(value), self._body)
            v.setObjectName("aiJsonValue")
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(v, 1)
            body_lay.addLayout(row)
        lay.addWidget(self._body)

        self._body.setVisible(self._expanded)
        self._refresh_header()

    def _refresh_header(self) -> None:
        chevron = "▾" if self._expanded else "▸"
        self._header.setText(f"{chevron}  {self._title}")

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._refresh_header()

    def natural_width(self) -> int:
        """Unwrapped content width (header bar + widest expanded row)."""
        mini = _px_mini()
        w = _measure_width(mini, f"▾  {self._title}")
        for key, value in self._rows:
            row = _measure_width(mini, value)
            if key:
                row += _measure_width(mini, key) + 8
            w = max(w, row + 22)  # aiJsonSectionBody h-margins (12 + 10)
        return w


class _ChatCard(QFrame):
    """Base for user / agent cards: an avatar block beside a content column.

    The avatar is its OWN block (top-aligned) laid out horizontally next to the
    content column (name+time header over the text / sectors) — so the text is
    indented past the avatar rather than sharing a row with the name/time."""

    _GAP = 10  # avatar ↔ content spacing

    def __init__(self, object_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(12, 10, 12, 10)
        self._row.setSpacing(self._GAP)
        # Content column (header + body) — filled by the subclass, then placed
        # next to the avatar via _assemble.
        self._content = QVBoxLayout()
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(4)

    def _assemble(self, avatar: QWidget, *, avatar_left: bool) -> None:
        """Place ``avatar`` and the content column side by side (avatar top-aligned)."""
        if avatar_left:
            self._row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
            self._row.addLayout(self._content, 1)
        else:
            self._row.addLayout(self._content, 1)
            self._row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)


class _UserCard(_ChatCard):
    """User message: right-aligned header (name · time · avatar) over the text."""

    def __init__(
        self, text: str, username: str, ts: float, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__("aiUserCard", parent)
        self._text = str(text)
        self._username = str(username)
        self._stamp = _fmt_ts(ts)

        # Header (name + time) right-aligned within the content column so it sits
        # toward the avatar on the right.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addStretch(1)
        name = QLabel(username, self)
        name.setObjectName("aiCardName")
        header.addWidget(name, 0, Qt.AlignmentFlag.AlignVCenter)
        if self._stamp:
            t = QLabel(self._stamp, self)
            t.setObjectName("aiCardTime")
            header.addWidget(t, 0, Qt.AlignmentFlag.AlignVCenter)
        self._content.addLayout(header)

        body = QLabel(self)
        body.setObjectName("aiCardText")
        body.setWordWrap(True)
        # Plain text — user input must never be interpreted as rich text/HTML.
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setText(self._text)
        self._content.addWidget(body)

        # Avatar on the right (user is the right-aligned side).
        self._assemble(
            _Avatar(letter=username, object_name="aiAvatarUser", parent=self),
            avatar_left=False,
        )

    def natural_width(self) -> int:
        text_w = _measure_width(_px_body(), self._text)
        header_w = _measure_width(_px_body(), self._username)
        if self._stamp:
            header_w += 8 + _measure_width(_px_mini(), self._stamp)
        content_w = max(text_w, header_w)
        return content_w + _Avatar._SIZE + self._GAP + 24  # avatar + gap + margins


class _AgentCard(_ChatCard):
    """Assistant message: left-aligned header (avatar · name · time), a meta line
    (phase · tokens), narration text, and collapsible JSON sectors."""

    def __init__(self, name: str, ts: float, parent: Optional[QWidget] = None) -> None:
        super().__init__("aiAgentCard", parent)
        self._name = str(name)
        self._stamp = _fmt_ts(ts)
        # Content tracked for natural-width measurement (widest narration line +
        # widest JSON sector drive the card width, capped by the zone).
        self._narration_texts: List[str] = []
        self._sections: List[_JsonSection] = []

        # Header (name + time) left-aligned within the content column.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        nm = QLabel(name, self)
        nm.setObjectName("aiCardName")
        header.addWidget(nm, 0, Qt.AlignmentFlag.AlignVCenter)
        if self._stamp:
            t = QLabel(self._stamp, self)
            t.setObjectName("aiCardTime")
            header.addWidget(t, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        self._content.addLayout(header)

        # Phase · tokens meta line (hidden until set_header supplies content).
        self._meta = QLabel(self)
        self._meta.setObjectName("aiCardMeta")
        self._meta.setVisible(False)
        self._content.addWidget(self._meta)

        # Body container — rebuilt on each set_lines (narration + JSON sectors).
        self._body_host = QWidget(self)
        self._body_host.setObjectName("aiAgentBody")
        self._body = QVBoxLayout(self._body_host)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(6)
        self._content.addWidget(self._body_host)

        # Avatar on the left (assistant is the left-aligned side).
        self._assemble(
            _Avatar(icon="icon_assistant", object_name="aiAvatarAgent", parent=self),
            avatar_left=True,
        )

    def set_header(self, title: str, tokens: int) -> None:
        parts: List[str] = []
        if title:
            parts.append(str(title))
        if tokens and tokens > 0:
            parts.append(tr("canvas.ai_build.tokens_n", "{n} tokens").format(n=tokens))
        text = " · ".join(parts)
        self._meta.setText(text)
        self._meta.setVisible(bool(text))

    @staticmethod
    def _find_struct_payload(line: convstore.Line):
        """Locate a dict/list payload segment in ``line``.

        Returns ``(index, obj)`` for the first non-``time`` segment whose text
        parses as a dict/list, else ``None``. Segment-level (not whole-line) so a
        ``Set x = {…} on 1012`` log lifts its ``value`` segment into sectors
        while keeping the surrounding narration."""
        for i, seg in enumerate(line):
            if str(seg[1]) == "time":
                continue
            obj = _try_parse_struct(str(seg[0]))
            if obj is not None and json_to_sections(obj):
                return i, obj
        return None

    @staticmethod
    def _narration_line(line: convstore.Line, payload_idx: int) -> convstore.Line:
        """The narration segments of ``line`` with the payload (and the ``=``/``:``
        connector immediately before it, plus every ``time`` segment) removed."""
        skip = {payload_idx}
        j = payload_idx - 1
        while j >= 0 and str(line[j][1]) == "time":
            j -= 1
        if j >= 0 and _CONNECTOR_RE.fullmatch(str(line[j][0])):
            skip.add(j)
        return [
            seg for k, seg in enumerate(line)
            if k not in skip and str(seg[1]) != "time"
        ]

    def set_lines(self, lines: convstore.Lines) -> None:
        """Rebuild the body from ``lines`` — grouping narration text and lifting
        any dict/list payload into collapsible sectors."""
        while self._body.count():
            item = self._body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._narration_texts = []
        self._sections = []

        pending: convstore.Lines = []

        def flush() -> None:
            if not pending:
                return
            lbl = QLabel(self._body_host)
            lbl.setObjectName("aiCardText")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lbl.setText(_lines_to_html(list(pending), agent=True))
            self._body.addWidget(lbl)
            self._narration_texts.extend(_line_plain_text(ln) for ln in pending)
            pending.clear()

        for line in lines:
            found = self._find_struct_payload(line)
            if found is None:
                pending.append(line)
                continue
            payload_idx, obj = found
            # Narration around the payload keeps its typed colors; flush any
            # preceding plain-narration lines first so ordering is preserved.
            narration = self._narration_line(line, payload_idx)
            if any(str(seg[0]).strip() for seg in narration):
                pending.append(narration)
            flush()
            panel = QWidget(self._body_host)
            panel.setObjectName("aiJsonPanel")
            pv = QVBoxLayout(panel)
            pv.setContentsMargins(0, 0, 0, 0)
            pv.setSpacing(4)
            for title, rows in json_to_sections(obj):
                sec = _JsonSection(title, rows, parent=panel)
                self._sections.append(sec)
                pv.addWidget(sec)
            self._body.addWidget(panel)
        flush()

    def natural_width(self) -> int:
        body_px, mini = _px_body(), _px_mini()
        header_w = _measure_width(body_px, self._name)
        if self._stamp:
            header_w += 8 + _measure_width(mini, self._stamp)
        content_w = header_w
        if self._meta.isVisible() and self._meta.text():
            content_w = max(content_w, _measure_width(mini, self._meta.text()))
        for txt in self._narration_texts:
            content_w = max(content_w, _measure_width(body_px, txt))
        for sec in self._sections:
            content_w = max(content_w, sec.natural_width())
        return content_w + _Avatar._SIZE + self._GAP + 24  # avatar + gap + margins


# ===========================================================================
# BotChatZone
# ===========================================================================
class BotChatZone(QWidget):
    """Scrollable stack of user/assistant chat cards (the AI Build chat surface)."""

    #: MAX card width as a fraction of the zone width (a ceiling — cards size to
    #: their content and only wrap past this). Agent cards may run a touch wider
    #: (they carry JSON sectors); short messages stay narrow.
    _USER_FRAC = 0.94
    _AGENT_FRAC = 0.98

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aiChatZone")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("aiChatScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._host = QWidget()
        self._host.setObjectName("aiChatHost")
        self._layout = QVBoxLayout(self._host)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(12)
        self._layout.addStretch(1)
        self._scroll.setWidget(self._host)
        self._scroll.viewport().setAutoFillBackground(False)
        self._host.setAutoFillBackground(False)
        lay.addWidget(self._scroll, 1)

        # (card_widget, width_fraction) — for viewport-relative max width.
        self._cards: List[Tuple[QWidget, float]] = []
        self._current_agent_card: Optional[_AgentCard] = None
        self._placeholder: Optional[QLabel] = None

        self.apply_theme()

    # -- introspection (used by the panel + tests) ---------------------
    @property
    def placeholder(self) -> Optional[QLabel]:
        return self._placeholder

    @property
    def cards(self) -> List[QWidget]:
        return [c for c, _ in self._cards]

    @property
    def current_agent_card(self) -> Optional[_AgentCard]:
        return self._current_agent_card

    # -- lifecycle -----------------------------------------------------
    def clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards = []
        self._current_agent_card = None
        self._placeholder = None
        self._layout.addStretch(1)

    def show_placeholder(self, text: str) -> None:
        self._placeholder = QLabel(text, self._host)
        self._placeholder.setObjectName("aiChatEmptyHint")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._layout.insertWidget(
            self._layout.count() - 1, self._placeholder, 0, Qt.AlignmentFlag.AlignCenter
        )

    def render(self, messages: Sequence[convstore.Message]) -> None:
        """Full re-render of a thread's messages as finalized cards."""
        self.clear()
        username = _current_username()
        assistant = tr("canvas.ai_build.assistant_name", "AI Assistant")
        for msg in messages:
            if msg.role == convstore.USER_ROLE:
                self._insert_user_card(_lines_plain_text(msg.lines), username, msg.ts)
            else:
                card = self._open_agent_card(assistant, msg.ts)
                card.set_header("", int(getattr(msg, "tokens", 0) or 0))
                card.set_lines(msg.lines)
        self.scroll_to_bottom()

    # -- live turn -----------------------------------------------------
    def add_user_message(self, text: str, *, ts: float = 0.0) -> None:
        self._drop_placeholder()
        self._insert_user_card(str(text), _current_username(), ts)
        self.scroll_to_bottom()

    def begin_agent_card(
        self, *, header: str = "", tokens: int = 0, ts: float = 0.0
    ) -> None:
        self._drop_placeholder()
        card = self._open_agent_card(
            tr("canvas.ai_build.assistant_name", "AI Assistant"), ts
        )
        card.set_header(header, tokens)
        self._current_agent_card = card

    def set_agent_lines(self, lines: convstore.Lines) -> None:
        if self._current_agent_card is not None:
            self._current_agent_card.set_lines(lines)
            self._recompute_width(self._current_agent_card)
            self.scroll_to_bottom()

    def set_agent_header(self, title: str, tokens: int) -> None:
        if self._current_agent_card is not None:
            self._current_agent_card.set_header(title, tokens)
            self._recompute_width(self._current_agent_card)

    def close_agent_card(self) -> None:
        self._current_agent_card = None

    # -- internals -----------------------------------------------------
    def _drop_placeholder(self) -> None:
        if self._placeholder is not None:
            self._placeholder.deleteLater()
            self._placeholder = None

    def _insert_user_card(self, text: str, username: str, ts: float) -> None:
        card = _UserCard(text, username, ts, parent=self._host)
        self._insert_card(card, self._USER_FRAC, align_right=True)

    def _open_agent_card(self, name: str, ts: float) -> _AgentCard:
        card = _AgentCard(name, ts, parent=self._host)
        self._insert_card(card, self._AGENT_FRAC, align_right=False)
        return card

    def _insert_card(self, card: QWidget, frac: float, *, align_right: bool) -> None:
        row = QWidget(self._host)
        row.setObjectName("aiChatRow")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        if align_right:
            rl.addStretch(1)
            rl.addWidget(card, 0)
        else:
            rl.addWidget(card, 0)
            rl.addStretch(1)
        self._cards.append((card, frac))
        self._apply_card_width(card, frac)
        self._layout.insertWidget(self._layout.count() - 1, row)

    def _apply_card_width(self, card: QWidget, frac: float) -> None:
        # Fit-to-content, capped: the card width is the natural (unwrapped) width
        # of its content, so a short sentence sits on one line; only content that
        # would exceed the cap wraps. The cap is a fraction of the zone's OWN
        # width (reliable inside resizeEvent, unlike the scroll viewport which
        # resizes a beat later — reading that stale value was the earlier bug),
        # minus room for the host margins + vertical scrollbar.
        cap = max(160, int(max(0, self.width() - 24) * frac))
        natural = card.natural_width() if hasattr(card, "natural_width") else cap
        natural += 6  # slack so a one-line sentence never wraps on rounding
        card.setFixedWidth(max(120, min(cap, natural)))

    def _recompute_width(self, card: QWidget) -> None:
        for c, frac in self._cards:
            if c is card:
                self._apply_card_width(c, frac)
                return

    def _refresh_card_widths(self) -> None:
        for card, frac in self._cards:
            self._apply_card_width(card, frac)

    def scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_card_widths()

    # -- theme ---------------------------------------------------------
    def apply_theme(self) -> None:
        bg_card = Config.get_color("bg_2", "#1A1A1A")
        bg_deep = Config.get_color("bg_3", "#101010")
        border = Config.get_color("border_1", "#444444")
        text = Config.get_color("main_t1", "#D6D3C7")
        label_color = Config.get_color("sub_t2", "#777777")
        sub_color = Config.get_color("sub_t1", "#A0A0A0")
        accent = Config.get_color("safe_zone", "#36E38E")
        accent_text = Config.get_color("alt_t1", "#1E1E1E")
        avatar_user = Config.get_color("theme_1", "#4ec9b0")
        avatar_agent = Config.get_color("theme_2", "#9989E0")
        user_card_bg = Config.get_color("bg_1", "#1E1E1E")
        sz_title = Config.get_font_size("size_normal", 16)
        sz_body = Config.get_font_size("size_normal", 16)
        sz_mini = Config.get_font_size("size_small", 12)

        self.setStyleSheet(
            f"QWidget#aiChatZone, QScrollArea#aiChatScroll, QWidget#aiChatHost, "
            f"QWidget#aiChatRow, QWidget#aiAgentBody, QWidget#aiJsonPanel {{ "
            f"background: transparent; }}"
            f"QLabel#aiChatEmptyHint {{ color: {label_color}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            # Cards.
            f"QFrame#aiUserCard {{ background-color: {user_card_bg}; "
            f"border: 1px solid {border}; border-radius: 12px; }}"
            f"QFrame#aiAgentCard {{ background-color: {bg_card}; "
            f"border: 1px solid {border}; border-radius: 12px; }}"
            f"QLabel#aiCardText {{ color: {text}; font-size: {sz_body}px; "
            f"background: transparent; }}"
            f"QLabel#aiCardName {{ color: {text}; font-size: {sz_body}px; "
            f"font-weight: 600; background: transparent; }}"
            f"QLabel#aiCardTime {{ color: {label_color}; font-size: {sz_mini}px; "
            f"background: transparent; }}"
            f"QLabel#aiCardMeta {{ color: {label_color}; font-size: {sz_mini}px; "
            f"background: transparent; }}"
            # Round avatars.
            f"QLabel#aiAvatarUser {{ background-color: {avatar_user}; color: {accent_text}; "
            f"border-radius: 15px; font-size: {sz_mini}px; font-weight: 700; }}"
            f"QLabel#aiAvatarAgent {{ background-color: {avatar_agent}; color: {accent_text}; "
            f"border-radius: 15px; font-size: {sz_mini}px; font-weight: 700; }}"
            # Collapsible JSON sectors.
            f"QFrame#aiJsonSection {{ background-color: {bg_deep}; "
            f"border: 1px solid {border}; border-radius: 8px; }}"
            f"QPushButton#aiJsonSectionHeader {{ background: transparent; color: {accent}; "
            f"border: none; text-align: left; padding: 6px 10px; "
            f"font-size: {sz_mini}px; font-weight: 600; }}"
            f"QPushButton#aiJsonSectionHeader:hover {{ color: {text}; }}"
            f"QWidget#aiJsonSectionBody {{ background: transparent; }}"
            f"QLabel#aiJsonKey {{ color: {label_color}; font-size: {sz_mini}px; "
            f"background: transparent; }}"
            f"QLabel#aiJsonValue {{ color: {text}; font-size: {sz_mini}px; "
            f"background: transparent; }}"
        )


__all__ = ["BotChatZone", "LogSegment", "parse_markup", "_ElidedLabel"]
