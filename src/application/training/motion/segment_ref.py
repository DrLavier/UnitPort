# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Segment reference encoding — a clip ref that carries a frame sub-range.

A **segment** is a named ``[start, end]`` frame sub-range of a larger motion
clip (created in the Clip Motion Editor). To let a segment flow through the
*single string channel* that already carries clip references — the training
item's ``clip`` field (canvas → ``spec_compiler._populate_motion``) — we
encode the range as a suffix on the base ref:

    <base_ref>#seg=<start>-<end>

``<base_ref>`` is whatever the whole-clip channel already accepts: a
``pack:<pkg>:<sub>/<file>`` token or an absolute file path. Neither contains
``#``, so the ``#seg=`` marker is unambiguous. ``start`` / ``end`` are
**inclusive** frame indices (matching the editor's in/out selection and the
persisted segment metadata).

This module is the single source of truth for that grammar (§11): the picker
UI builds the ref, and ``_populate_motion`` parses it back into
``(base_ref, start, end)``. The encoded ``#seg=`` string is decoded app-side
and **never** crosses the Isaac Kit subprocess boundary — the launcher
receives the range as a separate, positionally-aligned plain CLI argument.

Torch-free and registry-free: importable from both the PyQt app venv and the
Kit worker venv.
"""
from __future__ import annotations

from typing import Optional, Tuple

#: Delimiter separating a base clip ref from its ``start-end`` frame range.
_MARKER = "#seg="


def build_segment_ref(base_ref: str, start: int, end: int) -> str:
    """Encode ``base_ref`` + inclusive ``[start, end]`` into a segment ref.

    Raises ``ValueError`` (fail-loud, §8) on a malformed range or a base
    ref that already carries a ``#seg=`` marker (double-encoding).
    """
    base = str(base_ref)
    if not base:
        raise ValueError("build_segment_ref: base_ref is empty")
    if _MARKER in base:
        raise ValueError(
            f"build_segment_ref: base_ref already carries a segment marker: {base!r}"
        )
    s, e = int(start), int(end)
    if s < 0 or e < s:
        raise ValueError(
            f"build_segment_ref: invalid frame range [{s}, {e}] "
            f"(need 0 <= start <= end)"
        )
    return f"{base}{_MARKER}{s}-{e}"


def parse_segment_ref(ref: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Split a (possibly-ranged) clip ref into ``(base_ref, start, end)``.

    A ref without the ``#seg=`` marker is a whole-clip ref → returns
    ``(ref, None, None)``. A ref with the marker returns the base plus the
    inclusive integer bounds. Raises ``ValueError`` (§8) on a marker with a
    malformed range — never silently falls back to the whole clip, which
    would train on the wrong frames.
    """
    text = str(ref)
    if _MARKER not in text:
        return text, None, None
    base, _, rng = text.partition(_MARKER)
    if not base:
        raise ValueError(f"parse_segment_ref: empty base ref in {text!r}")
    lo_str, dash, hi_str = rng.partition("-")
    if not dash or not lo_str or not hi_str:
        raise ValueError(
            f"parse_segment_ref: malformed range {rng!r} in {text!r} "
            f"(expected '<start>-<end>')"
        )
    try:
        start, end = int(lo_str), int(hi_str)
    except ValueError as exc:
        raise ValueError(
            f"parse_segment_ref: non-integer range {rng!r} in {text!r}"
        ) from exc
    if start < 0 or end < start:
        raise ValueError(
            f"parse_segment_ref: invalid range [{start}, {end}] in {text!r} "
            f"(need 0 <= start <= end)"
        )
    return base, start, end


def has_segment_range(ref: str) -> bool:
    """True iff ``ref`` carries a ``#seg=`` frame sub-range."""
    return _MARKER in str(ref)


def parse_motion_ranges_arg(
    ranges_arg: Optional[str], n_files: int
) -> list:
    """Parse a positional ``lo:hi,-,lo:hi`` ranges string → per-file ranges.

    This is the CLI-side counterpart of the ``#seg=`` encoding: the app flattens
    ``clip_ranges`` to a comma-joined string (``isaac_lab/config.py``) that rides
    positionally with ``--unitport_amp_motion_files`` across the Isaac Kit
    subprocess boundary, and the Kit worker parses it back here. Kept in this
    torch-free / registry-free module so it is importable in BOTH the app venv
    (tests) and the Kit venv (the launcher, which cannot import ``registers``).

    Returns a list of length ``n_files``; each element is ``(lo, hi)`` (inclusive
    frame sub-range) or ``None`` (whole file). ``"-"`` / empty token → ``None``.
    An absent/empty arg → all ``None``. Fail-loud (§8) on a malformed token or a
    count mismatch — never silently trains on the wrong frames.
    """
    text = str(ranges_arg or "").strip()
    if not text:
        return [None] * n_files
    tokens = [t.strip() for t in text.split(",")]
    if len(tokens) != n_files:
        raise ValueError(
            f"motion ranges arg has {len(tokens)} tokens but there are "
            f"{n_files} motion files — must be positionally aligned."
        )
    out: list = []
    for tok in tokens:
        if tok in ("", "-"):
            out.append(None)
            continue
        lo_str, sep, hi_str = tok.partition(":")
        if not sep or not lo_str or not hi_str:
            raise ValueError(
                f"malformed motion range token {tok!r} (expected 'lo:hi' or '-')"
            )
        try:
            lo, hi = int(lo_str), int(hi_str)
        except ValueError as exc:
            raise ValueError(
                f"non-integer motion range token {tok!r}"
            ) from exc
        if lo < 0 or hi < lo:
            raise ValueError(
                f"invalid motion range [{lo}, {hi}] in {tok!r} (need 0 <= lo <= hi)"
            )
        out.append((lo, hi))
    return out


__all__ = [
    "build_segment_ref",
    "parse_segment_ref",
    "has_segment_range",
    "parse_motion_ranges_arg",
]
