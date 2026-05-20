"""One-shot canvas migrator: vendor / physical joint names → IR roles.

Phase 5 of the IR-only joint contract: every ``actor_setting`` node's
``init_joint_angles`` and ``action_joint_names_expr`` must use IR role
names (``hip_FL``, ``thigh_FL``, ``calf_FL``). Pre-Phase-5 canvases
shipped with vendor abbreviations (``fl_hx`` / ``fl_hy`` / ``fl_kn``)
or physical USD names (``FL_hip_joint``); the new spec_validator R8
rule + ``ActorSettingNode.validate()`` reject these. This script does
the one-shot translation.

Usage::

    .venv311/Scripts/python.exe bootstrap/migrate_canvas_joint_names_to_ir.py \
        projects/new_test/canvas/IsaacLab/ppo_walk.canvas.json
    # Or sweep an entire project:
    .venv311/Scripts/python.exe bootstrap/migrate_canvas_joint_names_to_ir.py \
        projects/new_test/canvas

The script:
* parses each ``*.canvas.json`` (single file or recursive directory),
* finds every ``actor_setting`` node,
* translates ``init_joint_angles`` keys via the body_ir tokenizer
  (``fl_hx`` → tokens ``["fl","hx"]`` → part=hips position=FL → IR=hip_FL),
* applies the quadruped ``knee_FL → calf_FL`` family alias when the
  bound robot's family is quadruped,
* translates literal entries in ``action_joint_names_expr`` (regex
  patterns containing metachars are left untouched),
* writes ``<file>.bak`` next to the original, then overwrites with the
  translated copy.

The family is inferred from the canvas's RobotNode ``asset_id`` →
``registers.robots.get_robot(sku).families[0]``. Pass ``--family
<name>`` to override (useful when the canvas has no RobotNode or the
asset is not yet registered).

Untranslatable keys are left untouched and reported on stderr — the
canvas will still fail validation downstream, prompting the user to
fix them by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The script must run from RELEASE/ root so the .venv311 finds src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from application.training.body_ir import (  # noqa: E402
    _CATEGORY_TO_ROLE_PREFIX,
    _FAMILY_ROLE_ALIASES,
    detect_part,
    detect_position,
    tokenize,
)


_REGEX_METACHARS = set(".^$*+?()[]{}|\\")


def _is_regex_pattern(s: str) -> bool:
    return any(ch in _REGEX_METACHARS for ch in s)


def vendor_name_to_ir_role(name: str, family: str) -> Optional[str]:
    """Translate one vendor / physical joint name to its IR role.

    Returns the IR role (e.g. ``"hip_FL"``) on success, or ``None`` when
    body_ir cannot infer both part and position.
    """
    tokens = tokenize(name)
    part = detect_part(tokens)
    pos = detect_position(tokens)
    if not part or not pos:
        return None
    prefix = _CATEGORY_TO_ROLE_PREFIX.get(part)
    if not prefix:
        return None
    role = f"{prefix}_{pos}"
    aliases = _FAMILY_ROLE_ALIASES.get(family, {})
    return aliases.get(role, role)


def _resolve_family_from_canvas(canvas: Dict[str, Any]) -> Optional[str]:
    """Walk canvas nodes for a RobotNode; resolve its asset → family."""
    for node in canvas.get("nodes", []):
        if node.get("schema_id") != "robot":
            continue
        params = node.get("params", {}) or {}
        asset_param = params.get("asset_id", {}) or {}
        asset_id = str(asset_param.get("value", "") or "").strip()
        if not asset_id:
            continue
        try:
            from registers import robots as _robots_registry  # noqa: WPS433
            sku = (
                asset_id
                if _robots_registry.get_robot(asset_id) is not None
                else _robots_registry.resolve_id(asset_id)
            )
            if not sku:
                continue
            spec = _robots_registry.get_robot(sku)
            families = list(getattr(spec, "families", []) or [])
            if families:
                return str(families[0])
        except Exception as exc:  # registry not bootable in this venv
            print(f"  ! family auto-detect failed: {exc}", file=sys.stderr)
            return None
    return None


def _translate_init_joint_angles(
    raw_value: str,
    family: str,
    *,
    where: str,
) -> Tuple[str, List[str]]:
    """Translate the JSON-string init_joint_angles dict; return (new_str, untranslated)."""
    untranslated: List[str] = []
    try:
        parsed = json.loads(raw_value) if isinstance(raw_value, str) and raw_value else {}
    except json.JSONDecodeError as exc:
        print(f"  ! {where}: invalid JSON ({exc}); skipping", file=sys.stderr)
        return raw_value, []
    if not isinstance(parsed, dict):
        return raw_value, []

    new_dict: Dict[str, float] = {}
    changed = False
    for key, value in parsed.items():
        ir = vendor_name_to_ir_role(str(key), family)
        if ir is None:
            untranslated.append(str(key))
            new_dict[str(key)] = value
            continue
        if ir != key:
            changed = True
        new_dict[ir] = value
    if not changed:
        return raw_value, untranslated
    return json.dumps(new_dict, ensure_ascii=False), untranslated


def _translate_action_joint_names_expr(
    raw_value: str,
    family: str,
    *,
    where: str,
) -> Tuple[str, List[str]]:
    """Translate JSON-string list of joint name patterns; preserve regexes."""
    untranslated: List[str] = []
    try:
        parsed = json.loads(raw_value) if isinstance(raw_value, str) and raw_value else []
    except json.JSONDecodeError as exc:
        print(f"  ! {where}: invalid JSON ({exc}); skipping", file=sys.stderr)
        return raw_value, []
    if not isinstance(parsed, list):
        return raw_value, []

    new_list: List[str] = []
    changed = False
    for item in parsed:
        if not isinstance(item, str):
            new_list.append(item)
            continue
        if _is_regex_pattern(item):
            new_list.append(item)
            continue
        ir = vendor_name_to_ir_role(item, family)
        if ir is None:
            untranslated.append(item)
            new_list.append(item)
            continue
        if ir != item:
            changed = True
        new_list.append(ir)
    if not changed:
        return raw_value, untranslated
    return json.dumps(new_list, ensure_ascii=False), untranslated


def _migrate_canvas(path: Path, family_override: Optional[str]) -> bool:
    """Translate one canvas file in-place; returns True if rewrites happened."""
    try:
        canvas = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"! cannot read {path}: {exc}", file=sys.stderr)
        return False

    family = family_override or _resolve_family_from_canvas(canvas) or "quadruped"
    print(f"# {path}  (family={family}{'' if family_override else ' inferred'})")

    any_change = False
    nodes = canvas.get("nodes", [])
    for node in nodes:
        if node.get("schema_id") != "actor_setting":
            continue
        params = node.get("params", {}) or {}

        ija = params.get("init_joint_angles")
        if isinstance(ija, dict) and "value" in ija:
            new_v, missed = _translate_init_joint_angles(
                ija["value"], family,
                where=f"{path.name}::{node.get('id','?')}.init_joint_angles",
            )
            if new_v != ija["value"]:
                ija["value"] = new_v
                any_change = True
                print(f"  + init_joint_angles translated to IR roles")
            if missed:
                print(
                    f"  ! init_joint_angles untranslated keys: {missed}",
                    file=sys.stderr,
                )

        expr = params.get("action_joint_names_expr")
        if isinstance(expr, dict) and "value" in expr:
            new_v, missed = _translate_action_joint_names_expr(
                expr["value"], family,
                where=f"{path.name}::{node.get('id','?')}.action_joint_names_expr",
            )
            if new_v != expr["value"]:
                expr["value"] = new_v
                any_change = True
                print(f"  + action_joint_names_expr translated to IR roles")
            if missed:
                print(
                    f"  ! action_joint_names_expr untranslated literals: {missed}",
                    file=sys.stderr,
                )

    if not any_change:
        print("  = no changes")
        return False

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
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
        description="Translate canvas actor_setting joint names to IR roles."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Single .canvas.json file or a directory to walk recursively.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help=(
            "Override morphology family (quadruped/biped/humanoid/manipulator). "
            "Default: auto-detect from the canvas's RobotNode asset_id, "
            "fallback to 'quadruped'."
        ),
    )
    args = parser.parse_args(argv)

    files = list(_iter_canvas_files(args.target))
    if not files:
        print("no .canvas.json files found", file=sys.stderr)
        return 2

    rewrites = 0
    for f in files:
        if _migrate_canvas(f, args.family):
            rewrites += 1
    print(f"\nrewrote {rewrites}/{len(files)} canvas files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
