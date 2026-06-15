# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: fold legacy ``new_item_*`` training-motion item
keys (opaque id + dead ``title`` field) to registered command ids.

The actual fold lives in
``application.compiler.training_item_id_migrate.migrate_training_item_ids``
(shared with the canvas load-time hook ``CanvasPage.from_workflow_dict``). This
script just sweeps files: it renames the keys, drops ``title``, rewrites
``reward_in__<old>`` edges, and registers each name as a user command via
``registers.commands.register_named_command`` (writing
``<USER_CONFIG_DIR>/registers/commands_custom.json``). Stamps
``metadata.training_item_ids_v1`` for idempotency; writes ``<file>.bak`` before
rewrite.

Usage::

    .venv311/Scripts/python.exe bootstrap/migrate_canvas_training_item_ids.py PATH [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from application.compiler.training_item_id_migrate import (  # noqa: E402
    migrate_training_item_ids,
    TRAINING_ITEM_IDS_FLAG,
)


def _migrate_canvas(path: Path, *, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        canvas = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    meta = canvas.get("metadata")
    if isinstance(meta, dict) and meta.get(TRAINING_ITEM_IDS_FLAG) is True:
        print(f"# {path}  (already migrated; skipping)")
        return False

    changed = migrate_training_item_ids(canvas)
    if not changed:
        print(f"# {path}  = no changes")
        return False

    print(f"# {path}  → renamed legacy training-item keys to registered ids")
    if dry_run:
        print("  [dry-run] would rewrite")
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text(json.dumps(canvas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  [ok] rewrote {path.name}; backup at {backup.name}")
    return True


def _iter_canvas_files(target: Path):
    if target.is_file():
        yield target
    elif target.is_dir():
        yield from sorted(target.rglob("*.canvas.json"))
    else:
        print(f"! target does not exist: {target}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fold legacy new_item_* training-motion keys to registered command ids."
    )
    parser.add_argument("target", type=Path, help="A .canvas.json file or a directory.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args(argv)

    files = list(_iter_canvas_files(args.target))
    if not files:
        print("no .canvas.json files found", file=sys.stderr)
        return 2
    rewrites = sum(1 for f in files if _migrate_canvas(f, dry_run=args.dry_run))
    print(f"\nrewrote {rewrites}/{len(files)} canvas files"
          f"{' (dry-run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
