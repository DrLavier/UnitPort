# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Snapshot + restore user-customised keys inside ``system.ini`` across an update.

Per CLAUDE.md §1.4 a small set of keys inside ``system.ini`` are
runtime-written by the install wizard / ``AuthManager`` (rather than
shipped by the factory copy) and a populated value identifies *this*
install — workspace location, account UUID, account username.

A naive update flow (``git checkout`` of a tag, or ``source-zip`` smart
merge) restores the factory copy of ``system.ini``, which would wipe
those keys and orphan the user's ``USER_CONFIG_DIR`` while also signing
them out. This module captures those keys before the channel runs and
re-applies them once the channel has finished writing the new file.

The restore uses the comment-preserving line-based patcher from
``user_workspace`` so factory theme comments, section ordering, and
newly-shipped ``[Theme]`` slots all survive the round-trip.
"""

from __future__ import annotations

from typing import Dict, Tuple


# (section, key) pairs to preserve across an in-place update. Keep this
# tuple in lockstep with CLAUDE.md §1.4 — the four "by-design
# system.ini-backed" runtime-written keys whose repo-tracked value MUST
# stay empty (the pre-commit hook enforces the empty side of the rule).
PRESERVED_SYSTEM_INI_KEYS: Tuple[Tuple[str, str], ...] = (
    ("Workspace", "root"),
    ("Resources", "user_config_dir"),
    ("Session", "active_user"),
    ("Session", "active_username"),
)


def snapshot_user_keys() -> Dict[str, Dict[str, str]]:
    """Read current values of :data:`PRESERVED_SYSTEM_INI_KEYS` from system.ini.

    Returns ``{section: {key: value}}`` containing only keys whose
    current value is non-empty — unset factory values have nothing
    worth restoring and are omitted so the patch step is a true no-op
    for them.
    """
    from application.service.user_workspace import read_system_ini_value

    out: Dict[str, Dict[str, str]] = {}
    for section, key in PRESERVED_SYSTEM_INI_KEYS:
        val = read_system_ini_value(section, key)
        if val:
            out.setdefault(section, {})[key] = val
    return out


def restore_user_keys(snapshot: Dict[str, Dict[str, str]]) -> bool:
    """Re-apply ``snapshot`` into ``system.ini``. No-op when empty.

    Uses the comment-preserving patcher; factory comments / section
    ordering / freshly-shipped keys survive untouched.

    Returns ``True`` if any value was actually written.
    """
    if not snapshot:
        return False
    from application.service.user_workspace import patch_system_ini

    return patch_system_ini(snapshot)


__all__ = [
    "PRESERVED_SYSTEM_INI_KEYS",
    "snapshot_user_keys",
    "restore_user_keys",
]
