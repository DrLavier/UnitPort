# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot migration: add model_family / variant_label to robots_canonical.json.

Schema additions (all optional, defaults safe):
    + model_family : string  — groups variants under one UI card (defaults to entry["model"])
    + variant_label: string  — chip-display label (defaults to "Standard")
    + inherits_from: string  — parent SKU; resolver merges parent fields into the variant
                               (this script does NOT set inherits_from for any factory
                               entry — variants are added by users at runtime via the
                               "+ Add Variant" path; the schema field is reserved here
                               for forward compat)

For every canonical entry this script:
  - Adds `model_family` (skipped if already present). Defaults are by entry:
      * Go2-W (4179f56d3c9f) → "go2w" (separate model_family per design — wheels
        give it a different IR topology vs Go2 quadruped)
      * Everything else → entry["model"] (each factory robot is its own family)
  - Adds `variant_label` defaulting to "Standard" (skipped if already present).

Hard SKU-stability check (CLAUDE.md §1.7): before writing, verifies that every
existing (brand, model) pair still resolves to its current SKU via build_sku.
Any drift aborts the migration before touching the file.

Run once:
    cd D:\\Unitport\\EXE\\RELEASE
    .\\.venv311\\Scripts\\python.exe bootstrap\\migrate_canonical_add_variant_metadata.py

Idempotent. Re-running on already-migrated data is a no-op.
Safe: writes a ``.bak`` next to the canonical before overwriting.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
_CANONICAL_PATH = _PROJECT_ROOT / "src" / "registers" / "data" / "robots_canonical.json"

# --- SKU helper (inlined to keep the migrator dependency-free) -----------------

_SLUG_RE = re.compile(r"[\s\-_/.]+")


def _norm(s: str) -> str:
    return _SLUG_RE.sub("", str(s).strip().lower())


def _build_sku(*parts: str, length: int = 12) -> str:
    raw = ".".join(_norm(p) for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


# --- Migration constants -------------------------------------------------------

GO2W_SKU = "4179f56d3c9f"

# Default model_family overrides (when the entry doesn't already declare one).
# Anything not listed here defaults to entry["model"]. Add an entry here when
# the on-disk model slug differs from the family name we want in the UI
# (e.g. Go2-W's slug is "go2w" but we want it as its own family "go2w" rather
# than rolling under "go2" because its wheels change the IR topology).
_FAMILY_OVERRIDES = {
    GO2W_SKU: "go2w",
}


def _migrate_entry(sku: str, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (new_entry, change_log). Idempotent on already-migrated entries."""
    out = dict(entry)
    changes: List[str] = []

    # 1. model_family
    if "model_family" not in out or not out.get("model_family"):
        out["model_family"] = _FAMILY_OVERRIDES.get(sku, out.get("model", ""))
        changes.append(f"model_family={out['model_family']!r}")

    # 2. variant_label
    if "variant_label" not in out or not out.get("variant_label"):
        out["variant_label"] = "Standard"
        changes.append(f"variant_label={out['variant_label']!r}")

    return out, changes


def _assert_sku_stability(robots: Dict[str, Any]) -> None:
    """Hard-fail when build_sku(brand, model) drifts from any current key."""
    failures: List[str] = []
    for sku, entry in robots.items():
        if not isinstance(entry, dict):
            continue
        brand = entry.get("brand", "")
        model = entry.get("model", "")
        if not brand or not model:
            continue
        expected = _build_sku(brand, model)
        if expected != sku:
            failures.append(
                f"  SKU drift: build_sku({brand!r}, {model!r}) -> {expected} "
                f"but current key is {sku}"
            )
    if failures:
        msg = (
            "Migration aborted: SKU drift detected. Historical manifests "
            "reference these SKUs; rewriting would break compatibility.\n"
            + "\n".join(failures)
        )
        print(msg, file=sys.stderr)
        sys.exit(2)


def main() -> int:
    if not _CANONICAL_PATH.exists():
        print(f"[migrate] canonical file missing: {_CANONICAL_PATH}", file=sys.stderr)
        return 1
    raw = _CANONICAL_PATH.read_text(encoding="utf-8")
    payload = json.loads(raw)
    robots = payload.get("robots", {})
    if not isinstance(robots, dict):
        print("[migrate] payload.robots is not a dict — aborting", file=sys.stderr)
        return 1

    _assert_sku_stability(robots)

    changed_any = False
    for sku in list(robots.keys()):
        entry = robots[sku]
        if not isinstance(entry, dict):
            continue
        new_entry, changes = _migrate_entry(sku, entry)
        if changes:
            robots[sku] = new_entry
            changed_any = True
            print(f"[migrate] {sku} ({entry.get('name', '<unnamed>')}): {', '.join(changes)}")

    if not changed_any:
        print("[migrate] no changes — every entry already carries variant metadata")
        return 0

    backup = _CANONICAL_PATH.with_suffix(_CANONICAL_PATH.suffix + ".bak")
    backup.write_text(raw, encoding="utf-8")
    print(f"[migrate] backup written: {backup.name}")

    payload["robots"] = robots
    _CANONICAL_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[migrate] wrote {_CANONICAL_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
