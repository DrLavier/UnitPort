# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot fixer for SVG path arc commands that were minified with a buggy
optimizer which dropped the ``ry`` parameter when ``rx == ry``.

Symptom: Qt logs ``qt.svg: Invalid path data; path truncated.`` at startup
because elliptical-arc commands like ``a934 0 0 1 6.055 3.201`` have only
6 numbers (rx, xrot, large, sweep, x, y) instead of the required 7
(rx, ry, xrot, large, sweep, x, y). Qt parses the 5th token as the sweep
flag, sees ``6.055`` (not ``0``/``1``), aborts the path.

Run from RELEASE root::

    .venv311\\Scripts\\python.exe scripts\\fix_svg_arcs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ICON_DIR = Path(__file__).resolve().parent.parent / "src" / "application" / "ui" / "assets" / "icon"

TOKEN_RE = re.compile(
    r"""
    ([MmZzLlHhVvCcSsQqTtAa])                          # command letter
    |
    ([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)    # number
    """,
    re.VERBOSE,
)


def tokenize(d: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in TOKEN_RE.finditer(d):
        if m.group(1):
            out.append(("cmd", m.group(1)))
        else:
            out.append(("num", m.group(2)))
    return out


def fix_path_data(d: str) -> tuple[str, int]:
    """Return (rewritten_d, num_arcs_fixed)."""
    toks = tokenize(d)
    n = len(toks)
    i = 0
    out: list[str] = []
    fixed = 0
    current_cmd: str | None = None  # current implicit command for arg continuation

    def is_flag(s: str) -> bool:
        return s in ("0", "1")

    while i < n:
        kind, val = toks[i]
        if kind == "cmd":
            current_cmd = val
            out.append(val)
            i += 1
            continue
        # Number outside any command — emit as-is and continue
        if current_cmd is None:
            out.append(val)
            i += 1
            continue

        c = current_cmd
        if c in ("A", "a"):
            # Need 7 numeric tokens: rx, ry, xrot, large, sweep, x, y
            nums = []
            j = i
            while j < n and toks[j][0] == "num" and len(nums) < 7:
                nums.append(toks[j][1])
                j += 1
            if len(nums) < 6:
                # Malformed tail — emit verbatim and bail on this command.
                for v in nums:
                    out.append(v)
                i = j
                continue

            # Decide: 7-arg form (flags at positions 3,4) or 6-arg form
            # (flags at positions 2,3 — meaning ry was dropped).
            if len(nums) == 7 and is_flag(nums[3]) and is_flag(nums[4]):
                # Valid 7-arg arc
                consumed = 7
                emitted = nums
            elif is_flag(nums[2]) and is_flag(nums[3]) and len(nums) >= 6:
                # 6-arg arc: insert ry = rx
                consumed = 6
                emitted = [nums[0], nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]]
                fixed += 1
            else:
                # Ambiguous — emit as-is to avoid making it worse.
                consumed = len(nums)
                emitted = nums

            out.extend(emitted)
            i += consumed
            # After an A/a argument-group, subsequent groups continue implicitly
            # with the same command — leave current_cmd as 'A'/'a' so we keep
            # looking for more 7-num groups.
            continue

        # Non-arc command — emit the number as-is. Implicit continuation rules
        # vary per command but we don't need to interpret them here.
        out.append(val)
        i += 1

    # Re-emit with spaces. We avoid trying to preserve original minification
    # since the cost is a few hundred bytes and the gain is a guaranteed-valid
    # parse.
    pieces: list[str] = []
    for tok in out:
        if tok in "MmZzLlHhVvCcSsQqTtAa":
            pieces.append(tok)
        else:
            # Prepend a space unless previous char was a command letter or '-'
            if pieces and not pieces[-1][-1].isalpha() and not tok.startswith("-"):
                pieces.append(" ")
            pieces.append(tok)
    return "".join(pieces), fixed


D_ATTR_RE = re.compile(r'(\sd\s*=\s*")([^"]+)(")', re.DOTALL)


EMPTY_PATH_RE = re.compile(r'\s*<path\s+d=""[^/>]*/>')
PATH_ELEM_RE = re.compile(r'\s*<path\s+[^/>]*d="([^"]*)"[^/>]*/>')


def _drop_unparseable_paths(text: str) -> tuple[str, int]:
    """Render each <path> in isolation under QSvgRenderer; drop ones that emit
    'Invalid path data' warnings. These come from a minifier that occasionally
    dropped a coordinate (e.g. a cubic bezier with 5 numbers instead of 6),
    leaving Qt unable to parse the tail. The affected files are heavily
    overlapping stippled icons; removing one broken path of hundreds is not
    visually noticeable.
    """
    # Lazy import so the rest of the script runs in non-Qt environments.
    import os as _os
    _os.environ.setdefault("QT_LOGGING_RULES", "qt.svg.warning=true")
    from PyQt6.QtCore import qInstallMessageHandler
    from PyQt6.QtSvg import QSvgRenderer
    from PyQt6.QtWidgets import QApplication
    import xml.sax.saxutils as _su

    if QApplication.instance() is None:
        QApplication([])

    bad = {"count": 0}

    def _handler(_t, _ctx, msg):
        if "Invalid path data" in msg or "path truncated" in msg:
            bad["count"] += 1

    qInstallMessageHandler(_handler)

    removed = 0

    def _check(m: re.Match[str]) -> str:
        nonlocal removed
        d = m.group(1)
        if not d.strip():
            return ""
        bad["count"] = 0
        mini = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            f'<path d="{_su.escape(d)}"/></svg>'
        )
        QSvgRenderer(mini.encode())
        if bad["count"] > 0:
            removed += 1
            return ""
        return m.group(0)

    new_text = PATH_ELEM_RE.sub(_check, text)
    return new_text, removed


def fix_svg(path: Path) -> tuple[int, int, int]:
    """Return (arcs_fixed, empty_paths_removed, unparseable_removed)."""
    text = path.read_text(encoding="utf-8")
    total_fixed = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal total_fixed
        prefix, body, suffix = m.group(1), m.group(2), m.group(3)
        new_body, fixed = fix_path_data(body)
        total_fixed += fixed
        return f"{prefix}{new_body}{suffix}"

    new_text = D_ATTR_RE.sub(_sub, text)
    new_text, n_empty = EMPTY_PATH_RE.subn("", new_text)
    new_text, n_unparseable = _drop_unparseable_paths(new_text)
    if total_fixed > 0 or n_empty > 0 or n_unparseable > 0:
        path.write_text(new_text, encoding="utf-8")
    return total_fixed, n_empty, n_unparseable


def main() -> int:
    if not ICON_DIR.is_dir():
        print(f"icon dir not found: {ICON_DIR}", file=sys.stderr)
        return 2
    total_arcs = 0
    total_empty = 0
    total_unparseable = 0
    files_touched = 0
    for svg in sorted(ICON_DIR.glob("*.svg")):
        arcs, empties, unparseable = fix_svg(svg)
        if arcs or empties or unparseable:
            files_touched += 1
            total_arcs += arcs
            total_empty += empties
            total_unparseable += unparseable
            print(
                f"  {svg.name}: arcs fixed={arcs}, "
                f"empty <path/> removed={empties}, "
                f"unparseable <path/> removed={unparseable}"
            )
    print(
        f"Done. arcs_fixed={total_arcs}, empty_stripped={total_empty}, "
        f"unparseable_stripped={total_unparseable} across {files_touched} file(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
