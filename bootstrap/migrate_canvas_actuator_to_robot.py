# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""One-shot canvas migrator: actuator caps ActorSetting → RobotNode.

2026-05 consolidation: the scalar PD/actuator fields moved off the
``actor_setting`` node.

* ``stiffness`` / ``damping`` were DROPPED — both engines now derive per-joint
  gains from the RobotNode's canonical ``(omega_n, zeta)`` via the mass-weighted
  solver (CLAUDE.md §10). Raw kp/kd no longer exist in the canvas.
* ``effort_limit`` / ``velocity_limit`` MOVED to the ``robot`` node (hidden
  params, edited via its pd_param_table panel).

This script moves ``effort_limit`` / ``velocity_limit`` from each
``actor_setting`` node to the (single) ``robot`` node in the same canvas, and
drops the obsolete ``stiffness`` / ``damping`` params. Without it, an old
canvas loses its effort/velocity values silently (unknown params are kept
in-memory but dropped on the next save — see page.py from_workflow_dict).

Usage::

    cd A:/111/UnitPort
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_actuator_to_robot.py \
        <USER>/projects/new_test/canvas/IsaacLab/Spot.canvas.json
    # Or sweep an entire project / canvas dir:
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_actuator_to_robot.py \
        <USER>/projects/new_test/canvas

The script is idempotent (stamps ``metadata.actuator_to_robot_v1``); re-running
is a no-op. A ``.bak`` backup is written next to each rewritten canvas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Params dropped outright (raw kp/kd no longer exist).
_DROP_KEYS = ("stiffness", "damping")
# Params relocated to the robot node.
_MOVE_KEYS = ("effort_limit", "velocity_limit")
_STAMP = "actuator_to_robot_v1"


def _nodes_of(canvas: dict) -> List[dict]:
    return canvas.get("nodes", []) or []


def _schema_id(node: dict) -> str:
    # Canvas nodes carry either "schema_id" (newer) or "kind"/"type"; the
    # joint-names migrator keys off "schema_id", so we match that.
    return str(node.get("schema_id", "") or "")


def _migrate_canvas(path: Path) -> bool:
    """Move actuator caps in one canvas in-place; returns True if rewritten."""
    try:
        canvas = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    meta = canvas.get("metadata", {}) or {}
    if meta.get(_STAMP):
        print(f"# {path}\n  = already migrated ({_STAMP})")
        return False

    print(f"# {path}")
    nodes = _nodes_of(canvas)
    actor_nodes = [n for n in nodes if _schema_id(n) == "actor_setting"]
    robot_nodes = [n for n in nodes if _schema_id(n) == "robot"]

    any_change = False

    # Harvest effort/velocity from the FIRST actor_setting that carries them
    # (canvases have one actor_setting; defend against more by taking the
    # first non-empty value).
    harvested: dict = {}
    for an in actor_nodes:
        params = an.get("params", {}) or {}
        for k in _MOVE_KEYS:
            if k in params and k not in harvested:
                harvested[k] = params[k]
        # Drop moved + obsolete keys off the actor node.
        for k in _MOVE_KEYS + _DROP_KEYS:
            if k in params:
                del params[k]
                any_change = True
                print(f"  - actor_setting.{k} removed")

    # Write harvested caps onto the robot node (don't clobber a value the
    # user already set on the robot node).
    if harvested:
        if not robot_nodes:
            print(
                "  ! no robot node found — effort/velocity could not be "
                "relocated; they were removed from actor_setting. Re-set them "
                "in the RobotNode pd_param_table panel.",
                file=sys.stderr,
            )
        else:
            rparams = robot_nodes[0].setdefault("params", {})
            for k, pobj in harvested.items():
                if k in rparams:
                    continue  # robot node already has its own value
                rparams[k] = pobj
                any_change = True
                val = pobj.get("value") if isinstance(pobj, dict) else pobj
                print(f"  + robot.{k} = {val}")

    # Stamp regardless (so a canvas with nothing to move is still marked done).
    canvas.setdefault("metadata", {})[_STAMP] = True

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(
        json.dumps(canvas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state = "rewrote" if any_change else "stamped (no caps to move)"
    print(f"  [ok] {state} {path.name}; backup at {backup.name}")
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
            "Move effort_limit/velocity_limit from actor_setting to the robot "
            "node and drop stiffness/damping (2026-05 PD consolidation)."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Single .canvas.json file or a directory to walk recursively.",
    )
    args = parser.parse_args(argv)

    files = list(_iter_canvas_files(args.target))
    if not files:
        print("no .canvas.json files found", file=sys.stderr)
        return 2

    rewrites = 0
    for f in files:
        if _migrate_canvas(f):
            rewrites += 1
    print(f"\nmigrated {rewrites}/{len(files)} canvas files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
