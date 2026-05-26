# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: il_observation obs_terms → strict scale form.

Phase: the il_observation node was redesigned so that each entry in
``obs_terms`` is now strictly the Isaac Lab ``ObservationTermCfg.scale``.
Pre-redesign canvases shipped a ``{name: number}`` shape where the number
was a widget default (1.0 across all terms) that the compiler silently
dropped — sidecar then recorded ``scale=null`` and bundle finalize
rejected the run in strict mode.

This script:

* parses each ``*.canvas.json`` (single file or recursive directory),
* finds every ``il_observation`` node,
* rewrites ``params.obs_terms.value`` so each entry's value becomes the
  per-term registry default scale (Isaac Lab convention, e.g.
  base_lin_vel=2.0, base_ang_vel=0.25, joint_vel=0.05). Items already in
  dict form (``{"scale": ...}``) are left untouched; unknown term keys
  are left untouched and reported on stderr.
* stamps ``metadata.obs_terms_scale_v1: true`` so a second invocation is
  a no-op,
* writes ``<file>.bak`` next to the original, then overwrites with the
  rewritten copy.

**Important** — the original ``1.0`` values are NOT carried over as
scales. They were the widget's flat default, not a user decision; using
them as scale would silently break observation normalisation at deploy
time (Isaac Lab's training-time ObservationManager applies its own
convention scale that differs from 1.0 for base_lin_vel etc.). The
migrator replaces them with the registry's new conventional default so
that the bundle's recorded scale matches what Isaac Lab actually applied
during training.

Usage::

    cd D:/Unitport/EXE/RELEASE
    .venv311/Scripts/python.exe bootstrap/migrate_il_observation_obs_terms.py \\
        projects/new_test/canvas/IsaacLab/ppo_walk.canvas.json
    # Or sweep an entire project tree (recursive):
    .venv311/Scripts/python.exe bootstrap/migrate_il_observation_obs_terms.py \\
        projects/new_test/canvas
    # Dry run — print what would change without touching files:
    .venv311/Scripts/python.exe bootstrap/migrate_il_observation_obs_terms.py \\
        path --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from application.training.isaac_lab.task_module_registry import (  # noqa: E402
    IL_OBS_REGISTRY,
)

_MIGRATION_FLAG = "obs_terms_scale_v1"


def _resolve_default_scale(term_key: str) -> Optional[float]:
    item = IL_OBS_REGISTRY.get(term_key)
    if item is None:
        return None
    return float(item.default)


def _migrate_obs_terms_payload(
    payload: Any, *, where: str,
) -> Tuple[Any, bool, List[str]]:
    """Rewrite one obs_terms payload to strict scale form.

    Accepts both ``Dict[str, ...]`` and JSON-encoded string of the same.
    Returns ``(new_payload, changed, unknown_terms)`` where ``new_payload``
    is in the same encoding (string/dict) as the input.
    """
    unknown: List[str] = []
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return payload, False, unknown
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(f"  ! {where}: invalid JSON ({exc}); skipping", file=sys.stderr)
            return payload, False, unknown
        new_parsed, changed, unknown = _migrate_obs_terms_payload(
            parsed, where=where,
        )
        if not changed:
            return payload, False, unknown
        return json.dumps(new_parsed, ensure_ascii=False), True, unknown

    if not isinstance(payload, dict):
        return payload, False, unknown

    new_dict: Dict[str, Any] = {}
    changed = False
    for term_key, value in payload.items():
        key = str(term_key).strip()
        # dict already-strict — keep verbatim
        if isinstance(value, dict):
            new_dict[key] = value
            continue
        default_scale = _resolve_default_scale(key)
        if default_scale is None:
            unknown.append(key)
            new_dict[key] = value
            continue
        # legacy number / bool / None → registry default scale
        if value is None or isinstance(value, (bool, int, float)):
            new_dict[key] = default_scale
            if value != default_scale:
                changed = True
            continue
        # list of numbers — accept verbatim (per-component scale shorthand)
        if isinstance(value, list):
            new_dict[key] = value
            continue
        # anything else — leave it; compiler will reject it
        new_dict[key] = value
    return new_dict, changed, unknown


def _migrate_canvas(path: Path, *, dry_run: bool) -> bool:
    """Rewrite one canvas file in-place; returns True if rewrites happened."""
    try:
        text = path.read_text(encoding="utf-8")
        canvas = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    metadata = canvas.setdefault("metadata", {})
    if isinstance(metadata, dict) and metadata.get(_MIGRATION_FLAG) is True:
        print(f"# {path}  (already migrated; skipping)")
        return False

    print(f"# {path}")

    any_change = False
    for node in canvas.get("nodes", []):
        if node.get("schema_id") != "il_observation":
            continue
        params = node.get("params", {}) or {}
        spec = params.get("obs_terms")
        if not isinstance(spec, dict) or "value" not in spec:
            continue
        new_value, changed, unknown = _migrate_obs_terms_payload(
            spec["value"],
            where=f"{path.name}::{node.get('id','?')}.obs_terms",
        )
        if changed:
            spec["value"] = new_value
            any_change = True
            print(f"  + node {node.get('id','?')} obs_terms rewritten")
        if unknown:
            print(
                f"  ! node {node.get('id','?')} obs_terms unknown keys "
                f"(not in IL_OBS_REGISTRY): {unknown}",
                file=sys.stderr,
            )

    # Always stamp the flag — even if no changes were needed — so future
    # invocations can fast-skip already-strict canvases.
    if isinstance(metadata, dict):
        if metadata.get(_MIGRATION_FLAG) is not True:
            metadata[_MIGRATION_FLAG] = True
            any_change = True

    if not any_change:
        print("  = no changes")
        return False

    if dry_run:
        print("  [dry-run] would rewrite")
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text(
        json.dumps(canvas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [ok] rewrote {path.name}; backup at {backup.name}")
    return True


def _iter_canvas_files(target: Path):
    if target.is_file():
        yield target
        return
    if target.is_dir():
        yield from sorted(target.rglob("*.canvas.json"))
        return
    print(f"! target does not exist: {target}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Translate canvas il_observation obs_terms to the strict scale "
            "form (each term value = ObservationTermCfg.scale, defaults "
            "sourced from IL_OBS_REGISTRY)."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Single .canvas.json file or a directory to walk recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report intended rewrites without modifying any files.",
    )
    args = parser.parse_args(argv)

    files = list(_iter_canvas_files(args.target))
    if not files:
        print("no .canvas.json files found", file=sys.stderr)
        return 2

    rewrites = 0
    for f in files:
        if _migrate_canvas(f, dry_run=args.dry_run):
            rewrites += 1
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\nrewrote {rewrites}/{len(files)} canvas files{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
