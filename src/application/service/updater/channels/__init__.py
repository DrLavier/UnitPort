# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Update channel registry + runtime detection.

Channels are tried in priority order: binary (authoritative for packaged
builds) -> source-git (cleanest preserve story when a real checkout +
git binary are available) -> source-zip (lossy fallback). The first
channel whose ``is_available()`` returns True wins.
"""

from __future__ import annotations

from typing import List, Optional

from .base import UpdateChannel
from .binary import BinaryInstallerChannel
from .source_git import SourceGitChannel
from .source_zip import SourceZipChannel


def iter_channels() -> List[UpdateChannel]:
    """Return all known channels in priority order.

    The list is constructed fresh each call so each channel runs its own
    ``is_available()`` probe at the moment ``detect_channel()`` is asked;
    state (e.g. ``.git/`` appearing or disappearing between launches) is
    not cached.
    """
    return [
        BinaryInstallerChannel(),
        SourceGitChannel(),
        SourceZipChannel(),
    ]


def detect_channel() -> Optional[UpdateChannel]:
    """Pick the highest-priority channel that reports itself available.

    Returns None only when every channel refused; the source-zip fallback
    has no preconditions beyond httpx, so this should never happen on a
    real install.
    """
    for ch in iter_channels():
        try:
            if ch.is_available():
                return ch
        except Exception:
            continue
    return None


__all__ = [
    "UpdateChannel",
    "BinaryInstallerChannel",
    "SourceGitChannel",
    "SourceZipChannel",
    "iter_channels",
    "detect_channel",
]
