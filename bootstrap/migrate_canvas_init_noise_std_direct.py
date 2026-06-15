# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: fold legacy log-space ``init_noise_std`` to the
direct-std convention (Bug#1).

Before: the Isaac Lab compiler exp()'d ``il_policy_network.init_noise_std``
(treating it as log_std), so a user's ``1.0`` became std ``e≈2.718``. After: the
value is the DIRECT std (legged_gym: ``1.0`` == std ``1.0``), emitted verbatim,
with a fail-loud on non-positive values.

The actual fold lives in
``application.compiler.init_noise_std_migrate.migrate_init_noise_std_direct``
(shared with the canvas load-time hook). This script just sweeps files. Stamps
``metadata.init_noise_std_direct_v1`` for idempotency; writes ``<file>.bak``
before rewrite.

Usage::

    .venv311/Scripts/python.exe bootstrap/migrate_canvas_init_noise_std_direct.py PATH [--dry-run]
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

from application.compiler.init_noise_std_migrate import (  # noqa: E402
    migrate_init_noise_std_direct,
    INIT_STD_DIRECT_FLAG,
)


def _migrate_canvas(path: Path, *, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        canvas = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    meta = canvas.get("metadata")
    if isinstance(meta, dict) and meta.get(INIT_STD_DIRECT_FLAG) is True:
        print(f"# {path}  (already migrated; skipping)")
        return False

    changed = migrate_init_noise_std_direct(canvas)
    if not changed:
        print(f"# {path}  = no changes (already direct-std / positive)")
        return False

    print(f"# {path}  → converted legacy log-space init_noise_std")
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
        description="Fold legacy log-space init_noise_std canvases to direct std."
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
