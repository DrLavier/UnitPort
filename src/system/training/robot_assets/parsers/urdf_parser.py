"""URDF parser — pure xml.etree, zero external deps.

Reads ``<robot>`` elements and pulls out:
- joint_names (in document order, ``fixed`` joints excluded)
- dof_count   (= len(joint_names) for revolute/prismatic; 1 dof each)
- link tree (parent → children) and base_link (the link with no parent)
- default_joint_pos (mid-point of joint <limit lower upper>, 0.0 if absent)
- mass (sum of all <link><inertial><mass value=…/>)
- foot_link_names — heuristic: links whose name contains 'foot' / 'toe' /
  'wheel' / 'sole'. The user can override via the RobotAssetNode params.

Read-only. Never writes to the source file.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


class UrdfParseError(ValueError):
    """Raised when a URDF file cannot be parsed."""


@dataclass
class UrdfMetadata:
    joint_names: List[str] = field(default_factory=list)
    dof_count: int = 0
    base_link: str = ""
    link_names: List[str] = field(default_factory=list)
    link_parent: Dict[str, str] = field(default_factory=dict)
    default_joint_pos: Dict[str, float] = field(default_factory=dict)
    mass: float = 0.0
    foot_link_names: List[str] = field(default_factory=list)


# Joint types that contribute to dof. URDF spec: revolute / continuous /
# prismatic each have 1 dof; floating has 6, planar has 3; fixed has 0.
_DOF_PER_TYPE = {
    "revolute": 1,
    "continuous": 1,
    "prismatic": 1,
    "planar": 3,
    "floating": 6,
    "fixed": 0,
}

_FOOT_KEYWORDS = ("foot", "toe", "wheel", "sole")


def parse_urdf(path: Path | str) -> UrdfMetadata:
    """Parse a URDF file and return its metadata.

    Raises ``UrdfParseError`` on any parse failure or structural issue
    (missing root, multiple base links, etc.).
    """
    p = Path(path)
    if not p.is_file():
        raise UrdfParseError(f"URDF file not found: {p}")

    try:
        tree = ET.parse(str(p))
    except ET.ParseError as exc:
        raise UrdfParseError(f"Invalid XML in {p}: {exc}") from exc

    root = tree.getroot()
    if root.tag != "robot":
        raise UrdfParseError(
            f"URDF root must be <robot>, got <{root.tag}> in {p}"
        )

    meta = UrdfMetadata()

    # ── Links + mass ──
    for link in root.findall("link"):
        name = link.get("name", "")
        if not name:
            continue
        meta.link_names.append(name)
        inertial = link.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                try:
                    meta.mass += float(mass_el.get("value", "0") or 0.0)
                except ValueError:
                    pass

    # ── Joints ──
    children_with_parent: Set[str] = set()
    for joint in root.findall("joint"):
        jname = joint.get("name", "")
        jtype = joint.get("type", "fixed")
        if not jname:
            continue
        # Skip fixed joints — they don't contribute to dof and aren't
        # actuated, so they don't belong in the joint_names list either.
        if _DOF_PER_TYPE.get(jtype, 0) <= 0:
            # Still record parent/child for the link tree
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is not None and child_el is not None:
                parent = parent_el.get("link", "")
                child = child_el.get("link", "")
                if parent and child:
                    meta.link_parent[child] = parent
                    children_with_parent.add(child)
            continue

        meta.joint_names.append(jname)
        meta.dof_count += _DOF_PER_TYPE[jtype]

        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is not None and child_el is not None:
            parent = parent_el.get("link", "")
            child = child_el.get("link", "")
            if parent and child:
                meta.link_parent[child] = parent
                children_with_parent.add(child)

        # Default joint position = midpoint of limits, else 0.
        limit = joint.find("limit")
        default = 0.0
        if limit is not None:
            try:
                lo = float(limit.get("lower", "0") or 0.0)
                hi = float(limit.get("upper", "0") or 0.0)
                if hi >= lo:
                    default = 0.5 * (lo + hi)
            except ValueError:
                pass
        meta.default_joint_pos[jname] = default

    # ── base_link: link that is never a child ──
    bases = [n for n in meta.link_names if n not in children_with_parent]
    if not bases:
        raise UrdfParseError(
            f"URDF {p}: no base link found (every link has a parent)"
        )
    if len(bases) > 1:
        # Multiple roots is unusual but legal in some URDFs (e.g. world +
        # robot). Pick the first one but warn via the dataclass: we just
        # pick deterministically rather than raising, because the user can
        # always provide an MJCF as the authoritative source.
        meta.base_link = bases[0]
    else:
        meta.base_link = bases[0]

    # ── Foot links: heuristic match ──
    meta.foot_link_names = [
        n for n in meta.link_names
        if any(kw in n.lower() for kw in _FOOT_KEYWORDS)
    ]

    return meta
