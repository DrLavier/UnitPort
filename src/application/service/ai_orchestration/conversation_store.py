# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Per-canvas AI-Build conversation history (threads model).

The AI-Build overlay's left rail is a list of **conversation threads**; each
thread is a linear sequence of user/agent messages. Threads are isolated per
canvas — the conversations for ``canvas/isaac_lab/ppo_walk.canvas.json`` are
never shown for a different canvas (blueprint follow-up: the left rail "controls
branches", and branches never cross canvases).

Persistence follows the same idiom as :mod:`api_config`: per-user state lives
under ``Paths.USER_CONFIG_DIR`` (CLAUDE.md §4 — NO fallback, NOT in ``src/``,
NOT in ``system.ini``), one JSON file per canvas. The file path is resolved
against the LIVE ``USER_CONFIG_DIR`` at call time so a workspace switch is
picked up without a restart.

This module is **Qt-free** (it is pure data + SDK file I/O); the overlay widget
renders the threads. A missing / malformed file is the documented empty state
(a fresh :class:`ConversationSet`) — that is the optional-per-user-settings
empty state, NOT an §8 silent fallback for a required input.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from unitport_sdk import Paths, log_warning, push_data, read_data

#: One persisted segment = ``[text, role]`` (role selects the render color). A
#: message line is a list of segments; a message holds a list of lines. User
#: messages are a single line with a single ``"user"`` segment; agent messages
#: accumulate the run's typed log lines verbatim, so a reopened thread renders
#: identically to the live run.
Segment = List[str]            # [text, role]
Line = List[Segment]
Lines = List[Line]

#: Role tag stamped on the single segment of a user message.
USER_ROLE = "user"

#: Bumped if the on-disk schema changes; load migrates older files (§8(c)).
_SCHEMA_VERSION = 1

#: Conversations live under this dir (relative to ``USER_CONFIG_DIR``), one
#: JSON file per canvas.
_CONV_SUBDIR = "ai_orchestration/conversations"


def _canvas_key(canvas_id: str) -> str:
    """Filesystem-safe, collision-free key for a canvas file id.

    A readable slug of the id plus a short sha1 digest of the *full* id — the
    slug keeps the file browsable, the digest guarantees two different canvases
    never share a file even after the slug is truncated / sanitised.
    """
    raw = (canvas_id or "").strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")[:48] or "unbound"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _file_for(canvas_id: str) -> Path:
    """Resolve the per-canvas conversation file inside the LIVE USER_CONFIG_DIR."""
    return Paths.USER_CONFIG_DIR / _CONV_SUBDIR / f"{_canvas_key(canvas_id)}.json"


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Message:
    """One turn in a thread: a ``role`` plus its rendered lines (typed segments).

    ``tokens`` is the token cost surfaced above an agent bubble (AI Build v2).
    User messages and pre-v2 persisted messages carry ``0`` (the migration
    default — an absent field is simply read as 0, no re-export needed).
    """

    role: str                       # "user" | "agent"
    lines: Lines = field(default_factory=list)
    ts: float = 0.0
    tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "ts": self.ts,
            "lines": self.lines,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        role = str(d.get("role", "agent"))
        ts = float(d.get("ts", 0.0) or 0.0)
        # Absent on pre-v2 files → 0 (forward-compatible migration, not an §8
        # fabricated value: a missing cosmetic token count is genuinely zero).
        try:
            tokens = int(d.get("tokens", 0) or 0)
        except (TypeError, ValueError):
            tokens = 0
        # Coerce each segment to a [text, role] pair of strings — a malformed
        # entry is dropped rather than crashing the whole thread render.
        raw_lines = d.get("lines", [])
        lines: Lines = []
        if isinstance(raw_lines, list):
            for raw_line in raw_lines:
                if not isinstance(raw_line, list):
                    continue
                line: Line = []
                for seg in raw_line:
                    if isinstance(seg, (list, tuple)) and len(seg) == 2:
                        line.append([str(seg[0]), str(seg[1])])
                lines.append(line)
        return cls(role=role, lines=lines, ts=ts, tokens=tokens)


@dataclass
class Conversation:
    """A single linear thread of messages."""

    id: str
    title: str = ""
    created: float = 0.0
    updated: float = 0.0
    messages: List[Message] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex),
            title=str(d.get("title", "")),
            created=float(d.get("created", 0.0) or 0.0),
            updated=float(d.get("updated", 0.0) or 0.0),
            messages=[
                Message.from_dict(m)
                for m in d.get("messages", [])
                if isinstance(m, dict)
            ],
        )


@dataclass
class ConversationSet:
    """All threads for one canvas, plus which one is active."""

    canvas_id: str = ""
    active_id: str = ""
    conversations: List[Conversation] = field(default_factory=list)

    # -- queries --------------------------------------------------------
    def get(self, conv_id: str) -> Optional[Conversation]:
        return next((c for c in self.conversations if c.id == conv_id), None)

    def active(self) -> Optional[Conversation]:
        return self.get(self.active_id) if self.active_id else None

    def ordered(self) -> List[Conversation]:
        """Newest first (the rail inserts new threads at the top).

        Ties on ``created`` (coarse clocks — Windows ``time.time()`` resolves to
        ~15 ms, so two quickly-created threads can share a timestamp) are broken
        by insertion order, so a later-appended thread always sorts newer."""
        return [
            conv
            for _, conv in sorted(
                enumerate(self.conversations),
                key=lambda pair: (pair[1].created, pair[0]),
                reverse=True,
            )
        ]

    # -- mutations ------------------------------------------------------
    def new_conversation(self, title: str = "") -> Conversation:
        ts = _now()
        conv = Conversation(
            id=uuid.uuid4().hex, title=title, created=ts, updated=ts, messages=[]
        )
        self.conversations.append(conv)
        self.active_id = conv.id
        return conv

    def remove(self, conv_id: str) -> None:
        self.conversations = [c for c in self.conversations if c.id != conv_id]
        if self.active_id == conv_id:
            ordered = self.ordered()
            self.active_id = ordered[0].id if ordered else ""

    def to_dict(self) -> dict:
        return {
            "schema_version": _SCHEMA_VERSION,
            "canvas_id": self.canvas_id,
            "active_id": self.active_id,
            "conversations": [c.to_dict() for c in self.conversations],
        }


# ---------------------------------------------------------------------------
# LLM history projection
# ---------------------------------------------------------------------------

#: Segment roles that are pure render chrome, dropped from the LLM projection.
_CHROME_SEGMENT_ROLES = frozenset({"time"})

#: Defaults for the LLM history projection (token frugality: a long thread or
#: a mutation-heavy build card must not flood the next run's prompts).
_HISTORY_MAX_MESSAGES = 12
_HISTORY_MAX_CHARS_PER_MESSAGE = 2000

_TRUNCATION_MARK = " …[truncated]"


def to_llm_history(
    messages: List[Message],
    *,
    max_messages: int = _HISTORY_MAX_MESSAGES,
    max_chars_per_message: int = _HISTORY_MAX_CHARS_PER_MESSAGE,
) -> List[Dict[str, str]]:
    """Project thread messages into OpenAI-style chat history for the harness.

    This is the in-thread conversation context that every run carries — the
    intent gate / direct answer / Understanding phase splice it between their
    system prompt and the new user message, so follow-ups ("what about SAC?",
    "make that penalty stronger") keep their referents. It is deliberately NOT
    the project knowledge base: only what the user and the agent already said
    in this thread.

    Projection rules (deterministic):
    - ``role``: ``user`` → ``"user"``, anything else → ``"assistant"``;
    - render chrome segments (timestamps) are dropped; lines join with ``\\n``;
    - empty messages are dropped;
    - consecutive same-role messages merge into one entry (agent runs emit one
      card per phase; some chat/completions providers reject same-role runs);
    - each merged entry is capped at ``max_chars_per_message`` (head kept, a
      loud truncation mark appended), then only the last ``max_messages``
      entries are returned.
    """
    rendered: List[Dict[str, str]] = []
    for msg in messages:
        role = "user" if msg.role == USER_ROLE else "assistant"
        lines = []
        for line in msg.lines:
            text = "".join(
                seg[0] for seg in line
                if len(seg) == 2 and seg[1] not in _CHROME_SEGMENT_ROLES
            ).strip()
            if text:
                lines.append(text)
        content = "\n".join(lines).strip()
        if not content:
            continue
        if rendered and rendered[-1]["role"] == role:
            rendered[-1]["content"] += "\n\n" + content
        else:
            rendered.append({"role": role, "content": content})
    for entry in rendered:
        if len(entry["content"]) > max_chars_per_message:
            entry["content"] = (
                entry["content"][:max_chars_per_message] + _TRUNCATION_MARK
            )
    return rendered[-max_messages:]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load(canvas_id: str) -> ConversationSet:
    """Load the thread set for ``canvas_id`` (empty set when none on disk)."""
    path = _file_for(canvas_id)
    if not path.exists():
        return ConversationSet(canvas_id=canvas_id)
    data = read_data(path)
    if not isinstance(data, dict):
        log_warning(f"[ai.conversations] malformed file at {path}; using empty set")
        return ConversationSet(canvas_id=canvas_id)
    convs = [
        Conversation.from_dict(c)
        for c in data.get("conversations", [])
        if isinstance(c, dict)
    ]
    active = str(data.get("active_id", ""))
    if active and not any(c.id == active for c in convs):
        active = ""
    return ConversationSet(canvas_id=canvas_id, active_id=active, conversations=convs)


def save(conv_set: ConversationSet) -> bool:
    """Persist ``conv_set`` under ``USER_CONFIG_DIR``. False on write failure."""
    rel = f"{_CONV_SUBDIR}/{_canvas_key(conv_set.canvas_id)}.json"
    if not push_data(rel, conv_set.to_dict()):
        log_warning(f"[ai.conversations] failed to write {_file_for(conv_set.canvas_id)}")
        return False
    return True


__all__ = [
    "Segment",
    "Line",
    "Lines",
    "USER_ROLE",
    "Message",
    "Conversation",
    "ConversationSet",
    "load",
    "save",
    "to_llm_history",
]
