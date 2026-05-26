# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Motion task-tag labelling — manifest files + filename auto-inference.

AMP training requires that the discriminator sees only motion clips
relevant to the target behavior. Without labels, all clips (walk, trot,
stand, turn, jump, …) are pooled into one discriminator that learns
a nonsensical mixed distribution — the policy then tries to blend
incompatible gaits (e.g. running posture while standing).

This module provides two mechanisms to assign a ``task_tag`` to each
motion clip:

1. **Manifest file** (``motion_labels.yaml``) — a human-authored YAML
   file placed alongside motion clips in a pack directory. Maps
   filename stems to task tags with optional per-clip weight overrides.

2. **Filename auto-inference** — regex-based extraction of the task
   tag from the clip filename. Covers the naming conventions of all
   major community AMP datasets (AMP_for_hardware, lifelike-agility,
   MetalHead, rl_amp).

Resolution order: manifest wins over auto-inference. Unresolved clips
get ``task_tag = ""`` (untagged).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Filename → task_tag inference rules
# ---------------------------------------------------------------------------

# Order matters: first match wins. Patterns are tested against the
# lowercase filename stem.  Each entry is ``(compiled_regex, tag)``.
_INFERENCE_RULES: List[Tuple[re.Pattern, str]] = [
    # lifelike-agility-and-play naming: dog_<behavior>_NNN_ret[_mir]
    (re.compile(r"^dog_idle[_\d]"),          "stand"),
    (re.compile(r"^dog_back[_\d]"),          "backward"),
    (re.compile(r"^dog_fast_run[_\d]"),      "run"),
    (re.compile(r"^dog_quad_run[_\d]"),      "run"),
    (re.compile(r"^dog_quad_walk[_\d]"),     "walk"),
    (re.compile(r"^dog_quad_walkrun[_\d]"),  "walkrun"),
    (re.compile(r"^dog_star_walk[_\d]"),     "walk"),
    (re.compile(r"^dog_zig_walk[_\d]"),      "walk"),
    (re.compile(r"^dog_jump_high[_\d]"),     "jump"),
    (re.compile(r"^dog_jump[_\d]"),          "jump"),
    (re.compile(r"^dog_hit[_\d]"),           "hit"),
    (re.compile(r"^dog_play[_\d]"),          "play"),
    # lifelike pose-format clips
    (re.compile(r"^dog_walk\d+_pose"),       "walk"),
    (re.compile(r"^dog_walk\d+_joint"),      "walk"),
    (re.compile(r"^dog_run\d+_pose"),        "run"),
    (re.compile(r"^dog_run\d+_joint"),       "run"),
    (re.compile(r"^dog_trot_"),              "trot"),
    (re.compile(r"^dog_pace_"),              "pace"),
    (re.compile(r"^dog_turn\d+"),            "turn"),
    # AMP_for_hardware naming
    (re.compile(r"^leftturn\d*$"),           "turn"),
    (re.compile(r"^rightturn\d*$"),          "turn"),
    (re.compile(r"^left_turn\d*$"),          "turn"),
    (re.compile(r"^right_turn\d*$"),         "turn"),
    (re.compile(r"^pace\d*$"),               "pace"),
    (re.compile(r"^trot\d*$"),               "trot"),
    (re.compile(r"^canter\d*$"),             "canter"),
    (re.compile(r"^hopturn"),                "hopturn"),
    # MetalHead / rl_amp naming
    (re.compile(r"^gallop_forward"),         "gallop"),
    (re.compile(r"^gallop_jump"),            "jump"),
    (re.compile(r"^trot_forward"),           "trot"),
    (re.compile(r"^turn_left"),              "turn"),
    (re.compile(r"^turn_right"),             "turn"),
    (re.compile(r"^jump\d*$"),               "jump"),
    (re.compile(r"^walk_jump"),              "jump"),
    # Generic fallback patterns
    (re.compile(r"stand"),                   "stand"),
    (re.compile(r"idle"),                    "stand"),
    (re.compile(r"walk"),                    "walk"),
    (re.compile(r"trot"),                    "trot"),
    (re.compile(r"run"),                     "run"),
    (re.compile(r"gallop"),                  "gallop"),
    (re.compile(r"canter"),                  "canter"),
    (re.compile(r"jump"),                    "jump"),
    (re.compile(r"turn"),                    "turn"),
    (re.compile(r"pace"),                    "pace"),
]


def infer_task_tag(filename: str) -> str:
    """Infer a task tag from a motion clip filename.

    Parameters
    ----------
    filename:
        Filename (with or without extension) — e.g. ``"trot0.txt"``
        or ``"dog_quad_walk_001_ret"``.

    Returns
    -------
    The inferred tag (e.g. ``"trot"``, ``"walk"``), or ``""`` if no
    rule matched.
    """
    stem = Path(filename).stem.lower()
    for pattern, tag in _INFERENCE_RULES:
        if pattern.search(stem):
            return tag
    return ""


# ---------------------------------------------------------------------------
# Manifest file (motion_labels.yaml)
# ---------------------------------------------------------------------------

# Example motion_labels.yaml:
#
#   # Labels for AMP_for_hardware/datasets/mocap_motions
#   labels:
#     trot0:    trot
#     trot1:    trot
#     pace0:    pace
#     pace1:    pace
#     leftturn0:  turn
#     rightturn0: turn
#
#   # Optional per-clip weight overrides (default 1.0)
#   weights:
#     trot0: 1.5
#     trot1: 0.8


def _load_manifest_dict(manifest_path: Path) -> Dict:
    """Load motion_labels.yaml via DataManager. Returns ``{}`` on any error."""
    if not manifest_path.is_file():
        return {}
    try:
        from unitport_sdk import load_data
    except ImportError:
        return {}
    try:
        data = load_data(manifest_path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """Load a ``motion_labels.yaml`` manifest file.

    Returns a dict mapping filename stem → task_tag. Missing or
    malformed files return an empty dict (never raises).
    """
    data = _load_manifest_dict(manifest_path)
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in labels.items() if v}


def load_manifest_weights(manifest_path: Path) -> Dict[str, float]:
    """Load per-clip weight overrides from ``motion_labels.yaml``.

    Returns a dict mapping filename stem → weight. Missing entries
    keep the default (1.0).
    """
    data = _load_manifest_dict(manifest_path)
    weights = data.get("weights", {})
    if not isinstance(weights, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in weights.items():
        if v is None:
            continue
        try:
            out[str(k).strip()] = float(v)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Combined resolution: manifest + auto-inference
# ---------------------------------------------------------------------------


def resolve_task_tags(
    file_paths: List[Path],
    *,
    manifest_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Resolve task tags for a list of motion files.

    Resolution order per file:
    1. ``motion_labels.yaml`` in *manifest_dir* (or the common parent
       of *file_paths* if not given)
    2. Filename auto-inference via :func:`infer_task_tag`
    3. ``""`` (untagged)

    Returns a dict mapping absolute path string → task_tag.
    """
    result: Dict[str, str] = {}

    # Find manifest directory
    if manifest_dir is None and file_paths:
        # Use the common parent of all files
        parents = {p.parent for p in file_paths}
        if len(parents) == 1:
            manifest_dir = parents.pop()
        else:
            # Multiple directories — try each file's own parent
            manifest_dir = None

    # Load manifest(s)
    manifest_labels: Dict[str, str] = {}
    if manifest_dir is not None:
        manifest_labels = load_manifest(manifest_dir / "motion_labels.yaml")

    for fp in file_paths:
        fp = Path(fp).resolve()
        key = str(fp)
        stem = fp.stem

        # Try manifest first
        tag = manifest_labels.get(stem, "")

        # Try per-directory manifest if we didn't have a global one
        if not tag and manifest_dir is None:
            local_manifest = load_manifest(fp.parent / "motion_labels.yaml")
            tag = local_manifest.get(stem, "")

        # Fall back to auto-inference
        if not tag:
            tag = infer_task_tag(fp.name)

        result[key] = tag

    return result


def list_available_tags(file_paths: List[Path]) -> List[str]:
    """Return sorted unique task tags for a set of motion files.

    Useful for populating a dropdown/filter UI in the training canvas.
    """
    tags = resolve_task_tags(file_paths)
    unique = sorted({v for v in tags.values() if v})
    return unique


__all__ = [
    "infer_task_tag",
    "load_manifest",
    "load_manifest_weights",
    "resolve_task_tags",
    "list_available_tags",
]
