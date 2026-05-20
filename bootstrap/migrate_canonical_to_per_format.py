"""One-shot migration: robots_canonical.json → per-format schema.

Old shape:
    {
      "<sku>": {
        ...,
        "joints": {"<uid>": {"name": "FL_thigh_joint", "ir_role": "thigh_FL"}, ...},
        "isaac_lab_joint_order": [...]  # optional
      }
    }

New shape:
    {
      "<sku>": {
        ...,
        "joints_per_format": {
          "MJCF": {"<uid>": {"name": "FL_thigh_joint", "ir_role": "thigh_FL"}, ...},
          "USD":  null,
          "URDF": null
        },
        "bodies_per_format": {
          "MJCF": {"<uid>": {"name": "FL_thigh", "ir_role": "thigh_FL"}, ...},
          "USD":  null,
          "URDF": null
        },
        "isaac_lab_joint_order": [...]  # KEPT as transitional USD-ordering hint
      }
    }

Body auto-fill (MJCF only):
- mujoco.MjModel.from_xml_path(asset.MJCF) → real body list
- jnt_bodyid[i] → which body each joint actuates
- canonical joints[*].ir_role + joint_to_body → body→ir_role assignment
- Foot bodies (no joint) auto-assigned via name keyword (foot/toe/sole)
- Base body via keyword (body/base/trunk/pelvis/torso)

Run once:
    cd D:\\Unitport\\EXE\\RELEASE
    .\\.venv311\\Scripts\\python.exe bootstrap\\migrate_canonical_to_per_format.py

Idempotent: re-running on already-migrated data is a no-op (detects new shape).
Safe: writes a .bak file before overwriting; backs out on any per-robot error.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
_CANONICAL_PATH = _PROJECT_ROOT / "src" / "registers" / "data" / "robots_canonical.json"

_FORMATS = ("MJCF", "USD", "URDF")

# ---------------------------------------------------------------------------
# SKU helpers (inlined to avoid import-time SDK dependency)
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[\s\-_/.]+")


def _norm(s: str) -> str:
    return _SLUG_RE.sub("", str(s).strip().lower())


def _build_sku(*parts: str, length: int = 12) -> str:
    raw = ".".join(_norm(p) for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# MJCF body discovery (lightweight, no app dependencies)
# ---------------------------------------------------------------------------
_PART_KEYWORDS: Dict[str, str] = {
    "foot": "feet", "toe": "feet", "sole": "feet",
    "body": "base", "base": "base", "trunk": "base",
    "pelvis": "base", "torso": "base",
}


def _resolve_mjcf_path(mjcf_rel: Optional[str]) -> Optional[Path]:
    """Resolve a canonical-relative MJCF path to an absolute path on disk."""
    if not mjcf_rel:
        return None
    rel = Path(mjcf_rel)
    if rel.is_absolute() and rel.exists():
        return rel
    # canonical paths are "menagerie/<robot>/scene.xml" — anchor under
    # custom_mods/models (the menagerie checkout location)
    candidates = [
        _PROJECT_ROOT / "custom_mods" / "models" / rel,
        _PROJECT_ROOT / "src" / "runtime" / "models" / rel,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _discover_mjcf_bodies(
    mjcf_path: Path,
) -> Tuple[List[str], Dict[str, str]]:
    """Return (body_names, joint_to_body) from a MJCF file via mujoco."""
    import mujoco  # type: ignore

    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    bodies: List[str] = []
    # skip body 0 ("world")
    for i in range(1, m.nbody):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if bn and bn not in bodies:
            bodies.append(bn)

    joint_to_body: Dict[str, str] = {}
    for i in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not jn:
            continue
        bid = int(m.jnt_bodyid[i])
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bn:
            joint_to_body[jn] = bn
    return bodies, joint_to_body


def _detect_position(body_name: str) -> str:
    """Quick position heuristic for foot-from-keyword (FL/FR/RL/RR/L/R)."""
    low = body_name.lower()
    if "fl" in low and "fr" not in low:
        return "FL"
    if "fr" in low:
        return "FR"
    if any(t in low for t in ("hl_", "rl_", "_rl")) and "hr" not in low:
        return "RL"
    if any(t in low for t in ("hr_", "rr_", "_rr")):
        return "RR"
    if "left" in low or low.endswith("_l"):
        return "L"
    if "right" in low or low.endswith("_r"):
        return "R"
    return ""


def _build_bodies_per_format(
    robot_sku: str,
    joints_dict: Dict[str, Dict[str, Any]],
    bodies_list: List[str],
    joint_to_body: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Construct bodies_per_format entry for one format.

    Algorithm:
      1. For each joint with ir_role: assign body (joint actuates) → same ir_role
      2. For remaining bodies matching foot/base keywords: assign foot_{POS} / base
    """
    body_to_ir: Dict[str, str] = {}

    # Step 1: actuated bodies via joint→body
    for jspec in joints_dict.values():
        if not isinstance(jspec, dict):
            continue
        jn = str(jspec.get("name", ""))
        ir_role = str(jspec.get("ir_role", ""))
        if not jn or not ir_role:
            continue
        bn = joint_to_body.get(jn)
        if bn and bn not in body_to_ir:
            body_to_ir[bn] = ir_role

    # Step 2: structural bodies (feet/base) via keyword + position
    for bn in bodies_list:
        if bn in body_to_ir:
            continue
        low = bn.lower()
        # Base detection (only assign once)
        for kw, cat in _PART_KEYWORDS.items():
            if kw in low and cat == "base":
                if "base" not in body_to_ir.values():
                    body_to_ir[bn] = "base"
                break
        if bn in body_to_ir:
            continue
        # Foot detection
        for kw, cat in _PART_KEYWORDS.items():
            if kw in low and cat == "feet":
                pos = _detect_position(bn)
                role = f"foot_{pos}" if pos else "foot"
                if role not in body_to_ir.values():
                    body_to_ir[bn] = role
                break

    # Emit as {uid: {name, ir_role}} dict (uid = build_sku(robot_sku, body_name))
    out: Dict[str, Dict[str, Any]] = {}
    for bn in bodies_list:
        if bn not in body_to_ir:
            continue
        uid = _build_sku(robot_sku, bn)
        out[uid] = {"name": bn, "ir_role": body_to_ir[bn]}
    return out


# ---------------------------------------------------------------------------
# Per-robot migration
# ---------------------------------------------------------------------------


def _migrate_one(sku: str, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (migrated_entry, notes). Idempotent."""
    notes: List[str] = []
    out = dict(entry)

    # Already migrated? Pass through.
    if "joints_per_format" in out and "joints" not in out:
        notes.append("already migrated; pass-through")
        return out, notes

    old_joints = dict(out.pop("joints", {}) or {})

    # joints_per_format: MJCF gets the old joints (current names assumed MJCF)
    joints_pf: Dict[str, Any] = {"MJCF": None, "USD": None, "URDF": None}
    if old_joints:
        joints_pf["MJCF"] = old_joints
        notes.append(f"joints → joints_per_format.MJCF ({len(old_joints)} entries)")
    else:
        notes.append("no joints declared (was empty)")

    # bodies_per_format: try MJCF auto-discovery
    bodies_pf: Dict[str, Any] = {"MJCF": None, "USD": None, "URDF": None}
    assets = out.get("assets", {}) or {}
    mjcf_rel = assets.get("MJCF")
    mjcf_path = _resolve_mjcf_path(mjcf_rel)
    if mjcf_path is not None and old_joints:
        try:
            bodies_list, jnt2body = _discover_mjcf_bodies(mjcf_path)
            bodies_pf["MJCF"] = _build_bodies_per_format(
                sku, old_joints, bodies_list, jnt2body
            )
            notes.append(
                f"MJCF body discovery: {len(bodies_list)} bodies → "
                f"{len(bodies_pf['MJCF'])} mapped to IR roles"
            )
        except Exception as exc:
            notes.append(f"MJCF body discovery FAILED: {exc!r}")
    elif mjcf_path is None and mjcf_rel:
        notes.append(f"MJCF path '{mjcf_rel}' not found on disk; bodies.MJCF=null")
    elif not old_joints:
        notes.append("no joints to anchor body discovery; bodies.MJCF=null")

    out["joints_per_format"] = joints_pf
    out["bodies_per_format"] = bodies_pf
    return out, notes


def main() -> int:
    if not _CANONICAL_PATH.exists():
        print(f"ERROR: {_CANONICAL_PATH} not found", file=sys.stderr)
        return 1

    raw = _CANONICAL_PATH.read_text(encoding="utf-8")
    payload = json.loads(raw)
    robots = payload.get("robots", {})

    print(f"=== Migrating {_CANONICAL_PATH.name} ({len(robots)} robots) ===\n")

    new_robots: Dict[str, Any] = {}
    errors: List[str] = []
    for sku, entry in robots.items():
        try:
            migrated, notes = _migrate_one(sku, entry)
            new_robots[sku] = migrated
            name = entry.get("name", sku)
            print(f"[{sku}] {name}")
            for n in notes:
                print(f"  -{n}")
            print()
        except Exception as exc:
            errors.append(f"{sku}: {exc!r}")
            new_robots[sku] = entry  # leave original

    if errors:
        print("=== ERRORS ===", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 2

    # Backup + write
    bak = _CANONICAL_PATH.with_suffix(".json.bak")
    bak.write_text(raw, encoding="utf-8")
    print(f"backup: {bak}")

    payload["robots"] = new_robots
    _CANONICAL_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote:  {_CANONICAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
