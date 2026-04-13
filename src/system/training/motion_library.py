#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Library — manages custom reference motion .npy files.

Directory layout (under ``custom_mods/training/motions/``):

    quadruped/
        go2/
            walk.npy
            trot.npy
        a1/
            walk.npy
    biped/
        h1/
            walk.npy
    wheeled/
        go2w/
            drive.npy
    manipulator/
        gr1/
            pick.npy
    generic/
        generic/
            default.npy

Files are browsed by category (derived from robot family) so that a Go2
user sees all quadruped motions, with their own model's files pinned first.
"""
from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.system.training.robot_family import ROBOT_FAMILY_BY_TYPE, normalize_robot_type

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # D:/.../NEW/

#: Single canonical directory for all motion .npy files.
#: Lives under the user-writable ``custom_mods/`` tree at the repo root.
CUSTOM_MOTIONS_DIR: Path = _REPO_ROOT / "custom_mods" / "training" / "motions"

#: Root where downloaded community packages (AMP_for_hardware, MetalHead,
#: rl_amp, …) live. The archive scanner walks each ``<package>/datasets/``
#: subtree looking for amp_legged_gym-style ``.txt`` / ``.json`` clips so
#: the Reference Motion dropdown surfaces them alongside the SB3 library.
ARCHIVE_ROOT: Path = _REPO_ROOT / "custom_mods" / "archives"

#: Where community packages typically drop their mocap .txt/.json files.
#: Order is deterministic so the picker shows entries in a stable sequence.
_ARCHIVE_MOTION_SUBDIRS: tuple = (
    "datasets/mocap_motions",
    "datasets/mocap_motions_bak",
    "datasets/mocap_motions_jump",
    "datasets/mocap_motions_jump_cmd",
    "datasets/hopturn",
)

#: Extensions recognised as amp_legged_gym motion clips. Phase_2's
#: AMPLegGymLoader reads both — .txt is the AMP_for_hardware default,
#: .json is used by newer forks like MetalHead's jump dataset.
_ARCHIVE_MOTION_EXTS: tuple = (".txt", ".json")

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

#: Canonical directory names used as first-level subdirs under custom_motions/.
CATEGORIES = ("quadruped", "biped", "wheeled", "manipulator", "generic")

_FAMILY_TO_CATEGORY: Dict[str, str] = {
    "quadruped":          "quadruped",
    "biped":              "biped",
    "wheeled":            "wheeled",
    "manipulator":        "manipulator",
    "generic_locomotion": "generic",
}


def get_category(robot_type: str) -> str:
    """Return the motion library category for a robot type string."""
    family = ROBOT_FAMILY_BY_TYPE.get(normalize_robot_type(robot_type), "generic_locomotion")
    return _FAMILY_TO_CATEGORY.get(family, "generic")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MotionEntry:
    name: str          # file stem, e.g. "walk"
    path: Path         # absolute path to the .npy file
    robot_model: str   # model sub-directory name, e.g. "go2"
    category: str      # category directory name, e.g. "quadruped"

    @property
    def rel_label(self) -> str:
        """Short path relative to custom_motions/, used in UI detail lines."""
        return f"{self.category}/{self.robot_model}"


@dataclass
class MotionPack:
    """A *bundle* of motion clips that share a common origin.

    Packs are the natural granularity for AMP training: the discriminator
    learns the joint distribution of an entire pool of natural movements,
    not a single clip. The Reference Motion picker surfaces packs as
    📦 entries that the launcher expands into a comma-separated list of
    file paths just before subprocess launch.

    A ``pack_id`` has the form ``"pack:<package>:<subdir>"`` — e.g.
    ``"pack:AMP_for_hardware:mocap_motions"``. The picker writes this
    string into ``ReferenceMotionConfig.motion_source`` (which is the
    same field that already carries ``loco:`` and ``generate:`` source
    strings); the window-layer scanner detects the ``pack:`` prefix and
    calls :func:`expand_motion_pack` to materialise the file list.
    """

    package: str       # archive package name, e.g. "AMP_for_hardware"
    subdir: str        # mocap subdir name, e.g. "mocap_motions"
    file_count: int    # number of motion files in the pack
    files: List[Path] = field(default_factory=list)
    """Absolute paths to the motion files. Sorted, deterministic order."""

    @property
    def pack_id(self) -> str:
        return f"pack:{self.package}:{self.subdir}"

    @property
    def display_label(self) -> str:
        return f"{self.package} / {self.subdir} ({self.file_count} clips)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def list_entries(
    *,
    category: Optional[str] = None,
    robot_type: Optional[str] = None,
    include_archives: bool = True,
) -> List[MotionEntry]:
    """
    Return all discoverable reference motion entries.

    Scans two locations:

    1. ``custom_mods/training/motions/`` — the user-writable SB3 motion
       library (one .npy per entry, grouped by category/model).
    2. ``custom_mods/archives/<package>/datasets/...`` — community package
       mocap clips from AMP_for_hardware / MetalHead / rl_amp style repos.
       These are the amp_legged_gym .txt/.json clips consumed by phase_2's
       ``AMPLegGymLoader``. They show up under
       ``robot_model="<package_name>"`` and category ``"quadruped"``
       (all known community AMP packages target quadrupeds today).

    Set ``include_archives=False`` to restrict the scan to the SB3
    library (useful when a call site only cares about .npy tracking
    clips, e.g. the legacy import workflow).

    If *robot_type* is given the category is inferred from it; that category
    is the only one scanned. If neither is given, all categories are scanned.
    """
    scan_cat: Optional[str] = category
    if scan_cat is None and robot_type:
        scan_cat = get_category(robot_type)

    entries: List[MotionEntry] = []

    # ── SB3 library side ──────────────────────────────────────────────
    if CUSTOM_MOTIONS_DIR.exists():
        cats = [scan_cat] if scan_cat else list(CATEGORIES)
        for cat in cats:
            cat_dir = CUSTOM_MOTIONS_DIR / cat
            if not cat_dir.is_dir():
                continue
            for model_dir in sorted(cat_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for npy in sorted(model_dir.glob("*.npy")):
                    entries.append(MotionEntry(
                        name=npy.stem,
                        path=npy.resolve(),
                        robot_model=model_dir.name,
                        category=cat,
                    ))

    # ── Archive side (AMP_for_hardware, MetalHead, …) ─────────────────
    if include_archives and (scan_cat is None or scan_cat == "quadruped"):
        entries.extend(_list_archive_motion_entries())

    return entries


def _list_archive_motion_entries() -> List[MotionEntry]:
    """Walk ``custom_mods/archives/<pkg>/datasets/...`` for AMP mocap clips.

    Returns one ``MotionEntry`` per file found. Packages are grouped via
    ``robot_model = <package_name>`` so the picker sorts them next to each
    other; the category is fixed to ``quadruped`` because every community
    AMP package in scope today targets quadrupeds (A1, Go2, etc.).

    Missing archive root or individual subdirs are silently skipped.
    """
    if not ARCHIVE_ROOT.is_dir():
        return []

    out: List[MotionEntry] = []
    for pkg_dir in sorted(ARCHIVE_ROOT.iterdir()):
        if not pkg_dir.is_dir():
            continue
        if pkg_dir.name.startswith(".") or pkg_dir.name.startswith("_"):
            continue
        for sub in _ARCHIVE_MOTION_SUBDIRS:
            base = pkg_dir / sub
            if not base.is_dir():
                continue
            for f in sorted(base.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in _ARCHIVE_MOTION_EXTS:
                    continue
                out.append(MotionEntry(
                    name=f.stem,
                    path=f.resolve(),
                    # Package name serves as the "model" grouping so
                    # sorting pins archive clips together. Distinct from
                    # any user-defined SB3 library model (e.g. "go2")
                    # because no SB3 model directory ever uses a
                    # capitalised package name like "AMP_for_hardware".
                    robot_model=pkg_dir.name,
                    category="quadruped",
                ))
    return out


# ---------------------------------------------------------------------------
# Motion packs (the natural granularity for AMP training)
# ---------------------------------------------------------------------------


def list_motion_packs() -> List[MotionPack]:
    """Enumerate the motion packs found under ``custom_mods/archives/``.

    One ``MotionPack`` per ``<package>/datasets/<subdir>/`` directory that
    contains at least one .txt or .json clip. Used by the Reference
    Motion picker to surface 📦 entries that can be selected as a single
    UI action and expanded into the full file list at training launch.
    """
    if not ARCHIVE_ROOT.is_dir():
        return []

    packs: List[MotionPack] = []
    for pkg_dir in sorted(ARCHIVE_ROOT.iterdir()):
        if not pkg_dir.is_dir():
            continue
        if pkg_dir.name.startswith(".") or pkg_dir.name.startswith("_"):
            continue
        for sub in _ARCHIVE_MOTION_SUBDIRS:
            base = pkg_dir / sub
            if not base.is_dir():
                continue
            files = sorted(
                f.resolve() for f in base.iterdir()
                if f.is_file() and f.suffix.lower() in _ARCHIVE_MOTION_EXTS
            )
            if not files:
                continue
            packs.append(MotionPack(
                package=pkg_dir.name,
                subdir=sub.split("/")[-1],   # "datasets/foo" → "foo"
                file_count=len(files),
                files=files,
            ))
    return packs


def expand_motion_pack(pack_id: str) -> List[Path]:
    """Resolve a ``"pack:<package>:<subdir>"`` identifier to its file list.

    Returns absolute paths in deterministic sorted order. Raises
    ``ValueError`` on a malformed identifier or unknown package/subdir.
    The window-layer scanner calls this in
    ``training_workspace_window._compile_and_launch_il_graph`` when a
    ReferenceMotionNode's ``motion_source`` starts with ``pack:``.
    """
    if not pack_id.startswith("pack:"):
        raise ValueError(f"Not a pack identifier: {pack_id!r}")

    rest = pack_id[len("pack:"):]
    parts = rest.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Malformed pack identifier {pack_id!r}: "
            f"expected 'pack:<package>:<subdir>'"
        )
    package, subdir = parts

    base = ARCHIVE_ROOT / package / "datasets" / subdir
    if not base.is_dir():
        raise ValueError(
            f"Pack {pack_id!r} not found on disk: {base}"
        )
    files = sorted(
        f.resolve() for f in base.iterdir()
        if f.is_file() and f.suffix.lower() in _ARCHIVE_MOTION_EXTS
    )
    if not files:
        raise ValueError(
            f"Pack {pack_id!r} contains no .txt/.json files at {base}"
        )
    return files


def import_file(
    src: Path,
    *,
    robot_type: str,
    robot_model: Optional[str] = None,
) -> MotionEntry:
    """
    Copy *src* into the motion library under the correct category/model dir.

    *robot_model* defaults to the normalised robot_type when not given.
    If a file with the same stem already exists in that dir the name is made
    unique by appending ``_2``, ``_3`` … before the extension.

    Returns the newly registered ``MotionEntry``.
    """
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    cat = get_category(robot_type)
    model = str(robot_model or normalize_robot_type(robot_type) or "generic").lower()
    dest_dir = CUSTOM_MOTIONS_DIR / cat / model
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    if dest.exists():
        stem, ext = src.stem, src.suffix
        i = 2
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{ext}"
            i += 1

    shutil.copy2(src, dest)
    return MotionEntry(name=dest.stem, path=dest.resolve(),
                       robot_model=model, category=cat)
