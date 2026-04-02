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
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def list_entries(
    *,
    category: Optional[str] = None,
    robot_type: Optional[str] = None,
) -> List[MotionEntry]:
    """
    Return all registered .npy motion entries.

    Scans ``custom_mods/training/motions/`` (the single canonical location).

    If *robot_type* is given the category is inferred from it; that category
    is the only one scanned.  If neither is given, all categories are scanned.

    Results are returned in (category, model, name) order — callers may
    re-sort as needed.
    """
    scan_cat: Optional[str] = category
    if scan_cat is None and robot_type:
        scan_cat = get_category(robot_type)

    entries: List[MotionEntry] = []
    if not CUSTOM_MOTIONS_DIR.exists():
        return entries

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

    return entries


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
