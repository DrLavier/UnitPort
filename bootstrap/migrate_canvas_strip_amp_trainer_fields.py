# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: strip deprecated AMP hyperparams off the
trainer node (RC-4).

The 9 AMP core hyperparams (amp_reward_coef … lerp_schedule_json) live
authoritatively on the ``discriminator`` node. The ``il_ppo_trainer`` /
``amp_trainer`` nodes historically carried duplicate (now deprecated)
copies that ``spec_compiler`` ignores; this sweep removes the stale
serialized values so the discriminator node is the unambiguous source
and the PARAM_AUTHORITY_CONFLICT warning stops firing.

The actual strip lives in
``application.compiler.amp_trainer_field_strip.strip_deprecated_amp_trainer_fields``
(shared with the canvas load-time hook). This script just sweeps files.
Stamps ``metadata.amp_trainer_fields_stripped_v1`` for idempotency;
writes ``<file>.bak`` before rewrite.

Usage::

    .venv311/Scripts/python.exe bootstrap/migrate_canvas_strip_amp_trainer_fields.py PATH [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from application.compiler.amp_trainer_field_strip import (  # noqa: E402
    strip_deprecated_amp_trainer_fields,
    AMP_TRAINER_STRIP_FLAG,
)


def _migrate_canvas(path: Path, *, dry_run: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        canvas = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    meta = canvas.get("metadata")
    if isinstance(meta, dict) and meta.get(AMP_TRAINER_STRIP_FLAG) is True:
        print(f"# {path}  (already migrated; skipping)")
        return False

    changed = strip_deprecated_amp_trainer_fields(canvas)
    if not changed:
        print(f"# {path}  = no changes")
        return False

    print(f"# {path}  → stripped deprecated trainer AMP fields")
    if dry_run:
        return True
    path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
    path.write_text(json.dumps(canvas, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _iter_canvas_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.canvas.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="canvas file or directory to sweep")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    files = _iter_canvas_files(args.path)
    if not files:
        print(f"no .canvas.json files under {args.path}", file=sys.stderr)
        return 1
    n_changed = sum(int(_migrate_canvas(f, dry_run=args.dry_run)) for f in files)
    print(f"\n{n_changed}/{len(files)} canvas file(s) "
          f"{'would change' if args.dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
