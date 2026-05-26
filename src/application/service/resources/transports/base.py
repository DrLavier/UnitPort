# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Transport ABC + shared helpers.

Each transport implements ``fetch(ctx)`` which downloads the asset
described by ``ctx.source`` into ``ctx.dest_dir``. The function is
synchronous and may take minutes; callers run it inside a
``unitport_sdk.Task`` so cancellation (via ``ctx.cancel_check``) and
progress emission (via ``ctx.on_progress``) flow through the SDK's
standard mechanisms.

Transports must NOT swallow errors silently — per RELEASE/CLAUDE.md §1.8,
any failure raises ``TransportError`` with a user-facing message. The
caller (DownloadResourceTask) translates that into an "error" entry state
and surfaces the message in the UI card. The only acceptable fallback is
the optional-import pattern (``huggingface_hub`` missing → loud raise at
import time, not silent fallback).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..index import ResourceSource


ProgressFn = Callable[[float, str], None]
"""Signature: ``(fraction in [0,1], one-line status) -> None``."""

CancelCheckFn = Callable[[], None]
"""Called at safe points; raises ``TaskCancelledException`` if the user cancelled."""


class TransportError(RuntimeError):
    """Loud, user-facing error from a transport.

    The message is shown on the resource card; keep it short and
    actionable (e.g. "git not on PATH" beats a 20-line stderr dump).
    """


@dataclass(frozen=True)
class TransportContext:
    """Per-fetch input passed to ``Transport.fetch``."""

    source: ResourceSource
    dest_dir: Path
    on_progress: ProgressFn
    cancel_check: CancelCheckFn


class Transport(ABC):
    """Synchronous fetch contract."""

    name: str = ""        # short label rendered on the resource card badge

    @abstractmethod
    def fetch(self, ctx: TransportContext) -> int:
        """Download ``ctx.source`` into ``ctx.dest_dir``.

        Returns the total size in bytes of the materialised tree (for the
        index's ``size_bytes`` field). Implementations may pass 0 when
        size is not naturally tracked — the manager will fall back to a
        ``du``-style walk in that case.

        Raises:
            TransportError: any user-actionable failure (network, parse,
                unsupported URL shape, missing tool on PATH).
            TaskCancelledException: re-raised from ``cancel_check`` so the
                Task wrapper can run partial-tree cleanup.
        """


# ---------------------------------------------------------------------------
# Utility helpers shared by transports
# ---------------------------------------------------------------------------


_GITHUB_REPO_RE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)(?P<owner>[^/]+)/(?P<repo>[^/\.]+)"
)


def repo_name_from_url(url: str) -> str:
    """Extract the bare repo name from a GitHub URL or path.

    ``https://github.com/escontrela/AMP_for_hardware.git`` -> ``AMP_for_hardware``
    ``git@github.com:foo/bar.git`` -> ``bar``
    Falls back to the last URL segment (minus trailing ``.git``) for
    non-github URLs.
    """
    s = (url or "").strip()
    if not s:
        raise TransportError("empty URL")
    m = _GITHUB_REPO_RE.match(s)
    if m:
        return m.group("repo")
    # Generic fallback: trailing path component, strip .git
    tail = s.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    if not tail:
        raise TransportError(f"cannot extract repo name from URL: {s!r}")
    return tail


_SAFE_DIR_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_dir_name(raw: str, *, fallback: str = "resource") -> str:
    """Squish a free-form string (HF repo_id, release tag, ...) into a
    filename-safe directory name.

    ``"org/name"`` -> ``"org__name"`` (double underscore so a re-split is
    unambiguous); other separators are collapsed into single underscores.
    """
    s = (raw or "").strip().replace("/", "__")
    s = _SAFE_DIR_RE.sub("_", s).strip("_.")
    return s or fallback


def directory_size_bytes(root: Path) -> int:
    """Sum of file sizes under ``root``; non-existent or unreadable -> 0."""
    if not root.exists():
        return 0
    total = 0
    try:
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


__all__ = [
    "ProgressFn",
    "CancelCheckFn",
    "TransportError",
    "TransportContext",
    "Transport",
    "repo_name_from_url",
    "safe_dir_name",
    "directory_size_bytes",
]
