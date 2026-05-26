# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Normalize ``src/config/system.ini`` to its factory-shipped state.

Invoked by ``reset.bat`` / ``reset.sh`` as part of a factory reset. Pure
stdlib so it can run under any Python 3 the OS exposes — independent of
the project's ``.venv311`` (which the reset step deletes right after
this).

What "factory" means here:

* ``[Resources]`` — keep the keys (shipped contract), blank every value.
  In particular ``user_config_dir`` is written at install / sign-in time
  to anchor the per-user store; on reset it must go back to empty so the
  next bootstrap re-derives it.
* ``[Workspace]`` — entire section is install-time state, drop it.
* ``[Session]`` — entire section is sign-in state, drop it.
* ``[Localisation] lang`` — force ``EN`` (factory default language).

Everything else (``[Theme] [Font] [SwitchButton] [System] [auth]
[CloudSync] [Canvas]``) is genuine factory content and is left untouched.

Implementation is a line-based transformer (not ``configparser.write``)
because the latter would drop every ``;``-prefixed comment in the file,
and those comments document the theme slot taxonomy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEM_INI = SCRIPT_DIR.parent / "src" / "config" / "system.ini"

DROP_SECTIONS = {"Workspace", "Session"}
BLANK_VALUE_SECTIONS = {"Resources"}
FORCED_VALUES = {
    ("Localisation", "lang"): "EN",
}

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
KV_RE = re.compile(r"^(\s*)([^=;#\s][^=]*?)(\s*=\s*)(.*?)(\s*)$")


def transform(lines: list[str]) -> tuple[list[str], list[str]]:
    out: list[str] = []
    changes: list[str] = []
    current: str | None = None

    for raw in lines:
        m = SECTION_RE.match(raw)
        if m:
            current = m.group(1).strip()
            if current in DROP_SECTIONS:
                changes.append(f"removed [{current}]")
                continue
            out.append(raw)
            continue

        if current in DROP_SECTIONS:
            continue

        kv = KV_RE.match(raw.rstrip("\r\n"))
        if kv and current is not None:
            indent, key, eq, value, _trail = kv.groups()
            key_stripped = key.strip()

            forced = FORCED_VALUES.get((current, key_stripped))
            if forced is not None and value.strip() != forced:
                newline_suffix = raw[len(raw.rstrip("\r\n")):]
                out.append(f"{indent}{key}{eq}{forced}{newline_suffix}")
                changes.append(f"set [{current}].{key_stripped} = {forced}")
                continue

            if current in BLANK_VALUE_SECTIONS and value.strip() != "":
                newline_suffix = raw[len(raw.rstrip("\r\n")):] or "\n"
                out.append(f"{indent}{key.rstrip()} ={newline_suffix}")
                changes.append(f"cleared [{current}].{key_stripped}")
                continue

        out.append(raw)

    return out, changes


def normalize(path: Path) -> int:
    if not path.is_file():
        print(f"[reset-ini] system.ini not found at {path}", file=sys.stderr)
        return 1

    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[reset-ini] failed to read {path}: {exc}", file=sys.stderr)
        return 1

    lines = original.splitlines(keepends=True)
    new_lines, changes = transform(lines)

    if not changes:
        print("[reset-ini] system.ini already in factory state, no changes")
        return 0

    while len(new_lines) >= 2 and new_lines[-1].strip() == "" and new_lines[-2].strip() == "":
        new_lines.pop()

    new_text = "".join(new_lines)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"[reset-ini] failed to write {path}: {exc}", file=sys.stderr)
        try:
            tmp.unlink()
        except OSError:
            pass
        return 1

    for line in changes:
        print(f"[reset-ini] {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(normalize(SYSTEM_INI))
