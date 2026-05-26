# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Motion task-tag labelling — manifest files + filename auto-inference.

DIRECT-MIGRATE from DEMO ``src/system/training/motion/labels.py``.

AMP training requires that the discriminator sees only motion clips
relevant to the target behavior. Without labels, all clips (walk, trot,
stand, turn, jump, …) are pooled into one discriminator that learns
a nonsensical mixed distribution. Two mechanisms assign a ``task_tag``:

1. **Manifest file** (``motion_labels.yaml``) — placed alongside motion
   clips in a pack directory.
2. **Filename auto-inference** — regex-based extraction.

Resolution order: manifest wins over auto-inference. Unresolved clips
get ``task_tag = ""`` (untagged).

I/O note: this module reads ``motion_labels.yaml`` files that may live
either in factory packs (``registers/data/motions/``) or in user-supplied
directories (``<USER_CONFIG_DIR>/motions/``). Both are explicit user-handed
paths, so reading them directly via ``open()`` here stays consistent
with the SDK contract (DataManager handles in-process project I/O; this
is plain text-file inspection of arbitrary user paths).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Filename → task_tag inference rules
# ---------------------------------------------------------------------------

_INFERENCE_RULES: List[Tuple[re.Pattern, str]] = [
    # lifelike-agility-and-play naming: dog_<behavior>_NNN_ret[_mir]
    (re.compile(r"^dog_idle[_\d]"),         "stand"),
    (re.compile(r"^dog_back[_\d]"),         "backward"),
    (re.compile(r"^dog_fast_run[_\d]"),     "run"),
    (re.compile(r"^dog_quad_run[_\d]"),     "run"),
    (re.compile(r"^dog_quad_walk[_\d]"),    "walk"),
    (re.compile(r"^dog_quad_walkrun[_\d]"), "walkrun"),
    (re.compile(r"^dog_star_walk[_\d]"),    "walk"),
    (re.compile(r"^dog_zig_walk[_\d]"),     "walk"),
    (re.compile(r"^dog_jump_high[_\d]"),    "jump"),
    (re.compile(r"^dog_jump[_\d]"),         "jump"),
    (re.compile(r"^dog_hit[_\d]"),          "hit"),
    (re.compile(r"^dog_play[_\d]"),         "play"),
    (re.compile(r"^dog_walk\d+_pose"),      "walk"),
    (re.compile(r"^dog_walk\d+_joint"),     "walk"),
    (re.compile(r"^dog_run\d+_pose"),       "run"),
    (re.compile(r"^dog_run\d+_joint"),      "run"),
    (re.compile(r"^dog_trot_"),             "trot"),
    (re.compile(r"^dog_pace_"),             "pace"),
    (re.compile(r"^dog_turn\d+"),           "turn"),
    # AMP_for_hardware naming
    (re.compile(r"^leftturn\d*$"),          "turn"),
    (re.compile(r"^rightturn\d*$"),         "turn"),
    (re.compile(r"^left_turn\d*$"),         "turn"),
    (re.compile(r"^right_turn\d*$"),        "turn"),
    (re.compile(r"^pace\d*$"),              "pace"),
    (re.compile(r"^trot\d*$"),              "trot"),
    (re.compile(r"^canter\d*$"),            "canter"),
    (re.compile(r"^hopturn"),               "hopturn"),
    # MetalHead / rl_amp naming
    (re.compile(r"^gallop_forward"),        "gallop"),
    (re.compile(r"^gallop_jump"),           "jump"),
    (re.compile(r"^trot_forward"),          "trot"),
    (re.compile(r"^turn_left"),             "turn"),
    (re.compile(r"^turn_right"),            "turn"),
    (re.compile(r"^jump\d*$"),              "jump"),
    (re.compile(r"^walk_jump"),             "jump"),
    # Generic fallback patterns
    (re.compile(r"stand"),                  "stand"),
    (re.compile(r"idle"),                   "stand"),
    (re.compile(r"walk"),                   "walk"),
    (re.compile(r"trot"),                   "trot"),
    (re.compile(r"run"),                    "run"),
    (re.compile(r"gallop"),                 "gallop"),
    (re.compile(r"canter"),                 "canter"),
    (re.compile(r"jump"),                   "jump"),
    (re.compile(r"turn"),                   "turn"),
    (re.compile(r"pace"),                   "pace"),
]


def infer_task_tag(filename: str) -> str:
    """Infer a task tag from a motion clip filename.

    Returns the inferred tag (e.g. ``"trot"``, ``"walk"``) or ``""``
    when no rule matched.
    """
    stem = Path(filename).stem.lower()
    for pattern, tag in _INFERENCE_RULES:
        if pattern.search(stem):
            return tag
    return ""


# ---------------------------------------------------------------------------
# Manifest file (motion_labels.yaml)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path):
    """Best-effort YAML load via DataManager (handles dispatch)."""
    if not path.is_file():
        return None
    try:
        from unitport_sdk import read_data
        return read_data(path)
    except Exception:
        return None


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """Load a ``motion_labels.yaml`` manifest. Returns ``{stem: tag}``;
    missing/malformed files yield ``{}`` (never raises)."""
    data = _load_yaml(manifest_path)
    if not isinstance(data, dict):
        return {}
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in labels.items() if v}


def load_manifest_weights(manifest_path: Path) -> Dict[str, float]:
    """Load per-clip weight overrides. Returns ``{stem: weight}``."""
    data = _load_yaml(manifest_path)
    if not isinstance(data, dict):
        return {}
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

    Resolution per file:
        1. ``motion_labels.yaml`` in *manifest_dir* (or common parent
           when not given)
        2. :func:`infer_task_tag` on the filename
        3. ``""`` (untagged)

    Returns ``{absolute_path: task_tag}``.
    """
    result: Dict[str, str] = {}

    if manifest_dir is None and file_paths:
        parents = {p.parent for p in file_paths}
        if len(parents) == 1:
            manifest_dir = parents.pop()
        else:
            manifest_dir = None

    manifest_labels: Dict[str, str] = {}
    if manifest_dir is not None:
        manifest_labels = load_manifest(manifest_dir / "motion_labels.yaml")

    for fp in file_paths:
        fp = Path(fp).resolve()
        key = str(fp)
        stem = fp.stem

        tag = manifest_labels.get(stem, "")
        if not tag and manifest_dir is None:
            local_manifest = load_manifest(fp.parent / "motion_labels.yaml")
            tag = local_manifest.get(stem, "")
        if not tag:
            tag = infer_task_tag(fp.name)
        result[key] = tag

    return result


def list_available_tags(file_paths: List[Path]) -> List[str]:
    """Return sorted unique task tags for a set of motion files."""
    tags = resolve_task_tags(file_paths)
    return sorted({v for v in tags.values() if v})


__all__ = [
    "infer_task_tag",
    "load_manifest",
    "load_manifest_weights",
    "resolve_task_tags",
    "list_available_tags",
]
