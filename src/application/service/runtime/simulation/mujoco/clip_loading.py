# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Clip-ref resolution + family lookup + ``MotionClip`` loading.

Shared by the MuJoCo motion-review task (``motion_review_session``, the
"Review Motion" external viewer) and the embedded clip-render session
(``clip_render_session``, the Clip Motion Editor's offscreen view). The
three functions here were extracted from ``MotionReviewTask`` so both
consumers drive clip loading through ONE implementation (CLAUDE.md §11) —
a divergence between "what the viewer plays" and "what the editor renders"
would be exactly the kind of silent drift §8 forbids.

No ``mujoco`` import lives here: resolving a path or loading a clip must
not drag in a GL stack (the assets-browser ``probe_clip`` calls
:func:`load_clip` on the GUI thread in a venv that may have no display).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_clip_path(clip_ref: str) -> Path:
    """Resolve a stored clip reference to an existing file path.

    Accepts a ``pack:<package>:<subdir>/<file>`` reference (resolved via
    the motion library) or a plain absolute / relative path, and tolerates a
    ``#seg=lo-hi`` segment sub-range suffix (stripped here — the range is a
    sub-range of the SAME file, so path resolution always targets the base
    clip). Raises (CLAUDE.md §8) on an empty ref or a missing file — never
    returns a sentinel.
    """
    from application.training.motion.segment_ref import parse_segment_ref

    # Strip any ``#seg=lo-hi`` suffix: the file on disk is the base clip; the
    # range is applied after load (see load_clip_ref). Without this, a segment
    # ref reaches the filesystem as a literal name and fails "file not found".
    base_ref, _lo, _hi = parse_segment_ref(str(clip_ref or ""))
    ref = base_ref.strip()
    if not ref:
        raise RuntimeError(
            "[motion/clip] no clip reference was provided — cannot resolve "
            "a motion file."
        )
    if ref.startswith("pack:"):
        from scripts.training_motion.library import resolve_pack_ref
        return resolve_pack_ref(ref)
    p = Path(ref)
    if not p.is_file():
        raise RuntimeError(
            f"[motion/clip] clip file not found: {ref!r}. The reference may "
            f"have been moved or removed from the motion library."
        )
    return p.resolve()


def resolve_target_family(sku: str) -> str:
    """Resolve the robot's primary family slug (the loader requires it).

    Raises when the SKU is unregistered or declares no families — both
    leave the loader unable to stamp a valid contract (CLAUDE.md §8).
    """
    from registers.robots import get_robot_spec

    spec = get_robot_spec(sku)
    if spec is None:
        raise RuntimeError(
            f"[motion/clip] robot SKU {sku!r} is not registered — cannot "
            f"resolve a target family for the clip loader."
        )
    families = list(getattr(spec, "families", []) or [])
    if not families:
        raise RuntimeError(
            f"[motion/clip] robot SKU {sku!r} declares no families in the "
            f"registry — fill the 'families' field in "
            f"registers/data/robots_canonical.json so the clip loader can "
            f"bind the motion to a family."
        )
    return str(families[0])


def load_clip(clip_path: Path, *, sku: str, family: str) -> Any:
    """Format-detect + load ``clip_path`` into a :class:`MotionClip`.

    ``sku`` / ``family`` are keyword-only required (the loader contract
    propagates them onto the returned reference contract). Returns the
    wrapped :class:`~application.training.motion.clip.MotionClip`.
    """
    from application.training.motion.loaders import (
        detect_motion_format,
        get_loader,
    )

    fmt = detect_motion_format(clip_path)
    loader = get_loader(fmt)
    contract = loader.load(clip_path, target_sku=sku, target_family=family)
    return contract.clip


def load_clip_ref(clip_ref: str, *, sku: str, family: str) -> Any:
    """Full clip-ref → :class:`MotionClip`, honouring a ``#seg=lo-hi`` range.

    The single entry point for "give me the clip this reference denotes": it
    decodes a segment sub-range (if present), resolves + loads the base file,
    and slices to the inclusive ``[lo, hi]`` frames via
    :meth:`MotionClip.subrange`. A whole-clip ref returns the whole clip. Used
    by the Review Motion viewer so playing back a segment shows exactly the
    selected frames — the same slice the AMP consumer applies (CLAUDE.md §11),
    with the same fail-loud out-of-bounds behaviour (§8).
    """
    from application.training.motion.segment_ref import parse_segment_ref

    base_ref, lo, hi = parse_segment_ref(str(clip_ref or ""))
    path = resolve_clip_path(base_ref)
    clip = load_clip(path, sku=sku, family=family)
    if lo is not None and hi is not None:
        clip = clip.subrange(lo, hi)
    return clip


__all__ = [
    "resolve_clip_path",
    "resolve_target_family",
    "load_clip",
    "load_clip_ref",
]
