#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProjectSession — process-wide source of truth for the active project.

UnitPort assets (checkpoints, bundles, training runs) are written into the
active project's directory under ``projects/<project_id>/``.  Many backend
modules (CheckpointRegistry, bundle_exporter, sys_nodes, …) need to know
*which* project is currently active so they can resolve paths consistently.

Previously each call site had to thread a ``ProjectStore + project_id`` pair
through every constructor — and where that wasn't done, code silently fell
back to the legacy global path ``custom_mods/training/checkpoints/``.  That
fallback is wrong: ``custom_mods/`` is reserved for user-imported community
resources only.  Real training products belong inside the project.

This module fixes that hole by providing one place to store / read the
currently-active project for the running process.  The UI shell sets it
when the active project changes; backend code reads it when no explicit
project context was supplied.

Subprocess note
---------------
Module-level state is **not** inherited across ``multiprocessing.spawn``.
When a worker process needs the session (e.g. the SB3 training process
calling ``export_bundle``), the parent must serialise the active project
into the worker's task spec and the worker must call
:func:`set_active` early in its entry point.  See ``train_run_thread.py``
and ``training_process.py`` for the canonical pattern.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.system.core.project_store import ProjectStore


__all__ = [
    "set_active",
    "clear_active",
    "get_store",
    "get_project_id",
    "is_active",
    "get_active_checkpoints_dir",
    "get_active_bundles_dir",
]


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_store: Optional["ProjectStore"] = None
_project_id: str = ""


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

def set_active(store: "ProjectStore", project_id: str) -> None:
    """Mark *project_id* (resolved through *store*) as the active project.

    Idempotent — safe to call repeatedly with the same arguments.
    Pass an empty ``project_id`` to clear (equivalent to :func:`clear_active`).
    """
    global _store, _project_id
    with _lock:
        if not project_id:
            _store = None
            _project_id = ""
            return
        _store = store
        _project_id = str(project_id)


def clear_active() -> None:
    """Forget the currently active project, if any."""
    global _store, _project_id
    with _lock:
        _store = None
        _project_id = ""


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_store() -> Optional["ProjectStore"]:
    """Return the active ``ProjectStore`` instance, or ``None``."""
    with _lock:
        return _store


def get_project_id() -> str:
    """Return the active project id, or an empty string when none is set."""
    with _lock:
        return _project_id


def is_active() -> bool:
    """``True`` when a project is currently active in this process."""
    with _lock:
        return bool(_store is not None and _project_id)


def get_active_checkpoints_dir() -> Optional[Path]:
    """Return ``projects/<active_id>/training/exported/`` or ``None``.

    Does not create the directory. (The legacy alias name
    ``_checkpoints_dir`` resolves to the v2 ``training/exported/`` location.)
    """
    with _lock:
        if _store is None or not _project_id:
            return None
        return _store._checkpoints_dir(_project_id)


def get_active_bundles_dir() -> Optional[Path]:
    """Return ``projects/<active_id>/training/exported/`` or ``None``.

    In Layout v2 a project has no separate top-level ``bundles/`` directory;
    exported bundles live under ``training/exported/<policy_id>/``.
    ``_bundles_dir`` is kept as a legacy alias to that location.
    """
    with _lock:
        if _store is None or not _project_id:
            return None
        return _store._bundles_dir(_project_id)


def get_active_project_root() -> Optional[Path]:
    """Return ``projects/<active_id>/`` or ``None``.

    Used by P5-T4 (Mission Runtime) so MjFieldNode can resolve
    project-local field overrides via field_loader.load_field(...,
    project_root=...) without having to plumb the active project id
    through every node visit. Mirrors the convenience accessors above.
    """
    with _lock:
        if _store is None or not _project_id:
            return None
        return _store._project_dir(_project_id)
