# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Local post-update notice reader.

Reads the project-root ``CHANGELOG.md`` — the *single* canonical history /
"what's new" file the maintainer updates on every release. The leading
license/copyright comment block (and any other leading HTML comment, e.g. a
``<!-- version: X.Y.Z -->`` marker carried over from the old notice file) is
stripped; the remaining Markdown body is what the post-update popup renders.

The notice is shown exactly once, on the first launch after the running app
version (``system.ini[System].version``) changes — see
``main.py:_maybe_show_version_notice``. The version shown in the popup is the
running app version (single source of truth), NOT parsed from the file; the
file contributes body content only. ``CHANGELOG.md`` is **not** on the
zip-update ``PRESERVE_PATHS`` allowlist, so each update overwrites it with the
new release's copy.
"""

from __future__ import annotations

import re
from typing import Optional

from unitport_sdk import DataManager, Paths, log_warning

# The single canonical notice file. Lives at PROJECT_ROOT so the zip-update
# channel (whose preserve-list does NOT include it) overwrites it with the
# new release's copy on update.
_NOTICE_REL = "CHANGELOG.md"

# A leading HTML comment block, possibly multi-line (SPDX header, or a stray
# ``<!-- version: X.Y.Z -->`` marker). Anchored at the start, non-greedy so it
# matches one block at a time.
_LEADING_COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*", re.DOTALL)


def _strip_leading_comments(text: str) -> str:
    """Drop every leading HTML comment block (and surrounding blank lines)."""
    while True:
        m = _LEADING_COMMENT_RE.match(text)
        if m is None:
            break
        text = text[m.end():]
    return text.strip()


def read_local_notice() -> Optional[str]:
    """Return the ``CHANGELOG.md`` body (license header stripped), or ``None``.

    ``None`` is returned when the file is absent, empty, or contains nothing
    but the license header. These are all legitimate "no notice to show"
    states — NOT swallowed errors — so the caller simply shows nothing.
    """
    path = Paths.PROJECT_ROOT / _NOTICE_REL
    # WHY KEPT (§8 c): a missing/empty changelog is a valid "no notice" state,
    # not a contract violation — return None and show nothing.
    if not path.exists():
        return None

    try:
        # ``.md`` resolves to the SDK text handler -> returns str.
        raw = DataManager.load(path)
    except Exception as exc:  # noqa: BLE001 — defensive file read
        log_warning(f"[updater] failed to read {_NOTICE_REL}: {exc}")
        return None

    text = (raw or "").strip()
    if not text:
        return None

    body = _strip_leading_comments(text)
    return body or None


__all__ = ["read_local_notice"]
