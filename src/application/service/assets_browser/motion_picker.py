# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Unified motion-clip discovery for the canvas Training Motion editor.

The Training Motion node's clip dropdown must offer **every** motion clip the
user can train against, no matter how it arrived on disk. Two discovery
mechanisms exist and neither alone is complete:

* ``scripts.training_motion.library.list_entries`` recognises the two
  *shipped* layouts — SB3 ``<category>/<model>/*.npy`` clips and community
  AMP packs (``<package>/datasets/<subdir>/*.{txt,json}``). It is a pure
  filesystem scanner with no ``ResourceManager`` / PyQt dependency (so it
  stays importable from headless training workers), and is therefore blind
  to downloaded packs whose arbitrary ``<package>/<robot>/*.<ext>`` layout
  matches neither shipped pattern — e.g. a HuggingFace LAFAN1 ``*.csv``
  dataset that lands under ``custom_mods/motions/<owner>__<repo>/<robot>/``.

* :class:`ResourceManager` owns the authoritative download index
  (``custom_mods/.resources.json``) but does not itself enumerate the clip
  *files* inside each downloaded pack.

This module is the join: it unions the library scan with a
loader-extension-driven walk of every ``local`` motion ``ResourceEntry``'s
folder, so a clip becomes pickable on the canvas the moment its download
finishes — closing the download → register → invoke loop that was broken
(downloaded resources were registered and browsable in the Resources panel
but invisible to the training canvas).

It lives in the **application layer** rather than in ``scripts`` on purpose:
the ``ResourceManager`` dependency transitively pulls PyQt6 / ``TasksManager``
via the app-signals bus, which the ``scripts`` package (loaded inside the
IsaacLab Kit worker venv, no PyQt6) must stay free of.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from unitport_sdk import log_warning

if TYPE_CHECKING:  # avoid importing the training stack at module load
    from scripts.training_motion.library import MotionEntry


def list_downloaded_motion_entries() -> List["MotionEntry"]:
    """One ``MotionEntry`` per clip file inside every downloaded motion pack.

    Keyed off the ``ResourceManager`` index (the registration record), not a
    bare filesystem sweep, so only packs the user actually downloaded surface
    and the pack's display name groups its clips. Only ``state == local``
    entries are walked — a half-downloaded or missing pack has no usable
    files. Clip files are recognised by the loader-registry extension set —
    the same single source of truth the Resources panel's clip table uses
    (``_EXT_TO_FORMAT``) — so the canvas picker and the Resources panel never
    disagree on what counts as a clip.
    """
    from application.service.resources import (
        ResourceKind,
        ResourceState,
        get_resource_manager,
    )
    from application.training.motion.loaders import _EXT_TO_FORMAT
    from scripts.training_motion.labels import infer_task_tag
    from scripts.training_motion.library import MotionEntry

    exts = frozenset(_EXT_TO_FORMAT)
    out: List[MotionEntry] = []
    seen: set = set()
    mgr = get_resource_manager()
    for e in mgr.list_motion_entries():
        if e.asset_kind != ResourceKind.MOTION or e.state != ResourceState.LOCAL:
            continue
        try:
            root = e.absolute_path()
        except OSError:
            continue
        if not root.is_dir():
            continue
        # Pack display name groups its clips in the picker (the MotionEntry
        # robot_model slot doubles as the grouping label, mirroring how the
        # SB3 library groups by model dir name).
        group = (e.name or "").strip() or e.id[:8]
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            try:
                rel_parts = f.relative_to(root).parts
            except ValueError:
                rel_parts = (f.name,)
            # Skip tool/cache trees (HF's ``.cache/huggingface``, ``__pycache__``,
            # any dot-prefixed dir) — same skip convention as the library scanner.
            if any(part.startswith(".") or part == "__pycache__" for part in rel_parts):
                continue
            try:
                p = f.resolve()
            except OSError:
                continue
            if p in seen:
                continue
            seen.add(p)
            # Use the path tail relative to the pack root as the clip name so
            # nested layouts (e.g. LAFAN1's <robot>/<clip>.csv) stay
            # disambiguated in a flat dropdown.
            rel = "/".join(rel_parts)
            name = rel[: -len(f.suffix)] if f.suffix else rel
            out.append(MotionEntry(
                name=name or f.stem,
                path=p,
                robot_model=group,
                category="generic",
                task_tag=infer_task_tag(f.name),
            ))
    return out


def list_pickable_motion_entries() -> List["MotionEntry"]:
    """Unified clip set for the canvas Training Motion editor's clip dropdown.

    ``library.list_entries()`` (shipped SB3 ``.npy`` library + community AMP
    packs) unioned with :func:`list_downloaded_motion_entries` (downloaded
    packs from the ResourceManager index). Deduped by absolute path; the
    shipped-library entry wins on tie so a pack present under both a shipped
    root and the download index keeps its richer library metadata.

    Each source is guarded independently: a failure scanning one source logs
    a loud warning and still returns the other source's entries, so a broken
    download folder can never blank out the whole picker (and vice versa).
    """
    from scripts.training_motion.library import list_entries

    entries: List["MotionEntry"] = []
    seen: set = set()

    def _add(items: List["MotionEntry"]) -> None:
        for it in items:
            try:
                key = str(Path(it.path).resolve())
            except OSError:
                key = str(it.path)
            if key in seen:
                continue
            seen.add(key)
            entries.append(it)

    try:
        _add(list_entries(include_archives=True))
    except Exception as exc:  # noqa: BLE001 — one bad source must not blank the picker
        log_warning(f"[motion_picker] shipped-library scan failed: {exc!r}")
    try:
        _add(list_downloaded_motion_entries())
    except Exception as exc:  # noqa: BLE001
        log_warning(f"[motion_picker] downloaded-resource scan failed: {exc!r}")

    return entries


__all__ = [
    "list_downloaded_motion_entries",
    "list_pickable_motion_entries",
]
