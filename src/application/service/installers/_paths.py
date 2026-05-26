# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Path helpers for the installer namespace.

Two small primitives shared by every installer (currently only Isaac Lab):

* :func:`default_isaac_lab_install_root` — the wizard's pre-filled default
  for "where to install Isaac Lab". Per the user decision recorded in
  ``plans/isaaclab-1-2-sb3-isaaclab-3-isaaclab-is-jolly-dawn.md``, this is
  ``PROJECT_ROOT/Engines/isaac_lab/``. The ``Engines/<backend_id>/``
  convention is reserved for future backends (e.g. ``Engines/mujoco_xla``).
* :func:`is_install_path_internal` — "can we safely reuse ``.venv311`` for
  this install path?" The judgment is conservative: target must be inside
  the project tree AND on the same volume. Any failure mode (cross-drive,
  network mount, permission error) returns False so callers fall back to
  the external-mode Isaac Sim venv and never touch the main project venv.

Neither function performs I/O beyond a resolve+exists probe; they are safe
to call from the wizard's main-thread enter-event without blocking.
"""

from __future__ import annotations

import os
from pathlib import Path

from unitport_sdk import Paths

# ``Engines/`` is the project-level installation root for "heavy backends
# whose footprint dwarfs everything else" (Isaac Sim is ~30 GB). Distinct
# from ``custom_mods/runtime/sdk_extensions/`` which holds brand SDK git
# clones (~MB scale).
_ENGINES_DIR_NAME = "Engines"
_ISAAC_LAB_BACKEND_ID = "isaac_lab"


def engines_root() -> Path:
    """Return ``PROJECT_ROOT/Engines/`` — the per-backend project install root."""
    return Paths.PROJECT_ROOT / _ENGINES_DIR_NAME


def default_isaac_lab_install_root() -> Path:
    """Default install destination for Isaac Lab.

    Returned as an absolute :class:`Path`. Does NOT create the directory —
    that is the installer's job after the user accepts the EULA.
    """
    return engines_root() / _ISAAC_LAB_BACKEND_ID


def _posix_mount_of(p: Path) -> Path:
    """Walk up until :func:`os.path.ismount` is True. POSIX-only helper.

    Falls back to filesystem root if no mount point is found before we
    run out of parents (unusual; happens in chroot-style sandboxes).
    """
    current = p.resolve(strict=False)
    while not os.path.ismount(current):
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def is_install_path_internal(target: str | os.PathLike[str]) -> bool:
    """True iff ``target`` is safe to share the project ``.venv311``.

    Two conditions must hold:

    1. ``target`` is inside ``Paths.PROJECT_ROOT`` (relative_to() succeeds).
    2. ``target`` and ``Paths.PROJECT_ROOT`` are on the same volume — on
       Windows this means matching drive letters (case-insensitive); on
       POSIX it means sharing a mount point. A different volume rules
       out the reuse because pip-installed editable packages cross-drive
       under ``.venv311`` invariably break Python's site machinery on
       Windows and have surprise behaviour with bind mounts on Linux.

    Any failure (path can't be resolved, permission denied, etc.) returns
    False — the conservative default is "treat as external" so the
    installer falls back to Isaac Sim's bundled python.bat and never
    contaminates ``.venv311``.
    """
    try:
        target_path = Path(target).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        return False

    try:
        project_root = Paths.PROJECT_ROOT.resolve(strict=False)
    except (OSError, ValueError):
        return False

    try:
        target_path.relative_to(project_root)
        in_subtree = True
    except ValueError:
        in_subtree = False

    if not in_subtree:
        return False

    try:
        if os.name == "nt":
            same_vol = (
                target_path.drive.upper() == project_root.drive.upper()
                and bool(target_path.drive)
            )
        else:
            same_vol = _posix_mount_of(target_path) == _posix_mount_of(project_root)
    except OSError:
        return False

    return same_vol


__all__ = [
    "engines_root",
    "default_isaac_lab_install_root",
    "is_install_path_internal",
]
