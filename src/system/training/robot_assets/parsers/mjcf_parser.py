"""MJCF parser — uses the mujoco python wheel.

Per AMP_design.yaml §7.risks.usd_parser_venv_dependency (path b), this is
the **primary** metadata source for RobotAsset in the main venv. The
mujoco wheel installs cleanly outside Isaac Sim, so this parser works
both in the main venv and the launcher subprocess.

Read-only. Never writes to the source file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


class MjcfParseError(ValueError):
    """Raised when an MJCF file cannot be loaded by mujoco.MjModel."""


@dataclass
class MjcfMetadata:
    joint_names: List[str] = field(default_factory=list)
    dof_count: int = 0
    base_link: str = ""
    link_names: List[str] = field(default_factory=list)
    default_joint_pos: Dict[str, float] = field(default_factory=dict)
    mass: float = 0.0
    foot_link_names: List[str] = field(default_factory=list)


_FOOT_KEYWORDS = ("foot", "toe", "wheel", "sole")


def parse_mjcf(path: Path | str) -> MjcfMetadata:
    """Parse an MJCF file via mujoco.MjModel and return its metadata.

    Raises ``MjcfParseError`` on any load failure. The mujoco wheel must
    be installed (it is in both main venv and Isaac venv).
    """
    p = Path(path)
    if not p.is_file():
        raise MjcfParseError(f"MJCF file not found: {p}")

    try:
        import mujoco
    except ImportError as exc:
        raise MjcfParseError(
            "mujoco python package is required for MJCF parsing. "
            "Install it with: pip install mujoco"
        ) from exc

    try:
        model = mujoco.MjModel.from_xml_path(str(p))
    except Exception as exc:
        raise MjcfParseError(f"mujoco.MjModel failed to load {p}: {exc}") from exc

    meta = MjcfMetadata()

    # ── Joint names (in MuJoCo's iteration order = qpos order) ──
    # Skip the implicit free joint of a floating-base robot from the
    # actuated-joint list, but DO count it in dof_count for transparency.
    # This matches what unitree_gym_env / policy_runner already do.
    free_joint_count = 0
    for j in range(int(model.njnt)):
        try:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        except Exception:
            name = None
        jtype = int(model.jnt_type[j])
        # mjJNT_FREE = 0, mjJNT_BALL = 1, mjJNT_SLIDE = 2, mjJNT_HINGE = 3
        if jtype == 0:  # free joint — floating base, skip in named list
            free_joint_count += 1
            continue
        meta.joint_names.append(name or f"joint_{j}")

    meta.dof_count = len(meta.joint_names)

    # ── Default joint pos: from model.qpos0, indexed by jnt_qposadr ──
    # qpos0 is the keyframe-zero pose; for non-free joints it's a single
    # scalar at jnt_qposadr[j].
    qpos0 = list(getattr(model, "qpos0", []))
    if qpos0:
        idx = 0
        for j in range(int(model.njnt)):
            jtype = int(model.jnt_type[j])
            if jtype == 0:  # free
                continue
            try:
                addr = int(model.jnt_qposadr[j])
                if 0 <= addr < len(qpos0) and idx < len(meta.joint_names):
                    meta.default_joint_pos[meta.joint_names[idx]] = float(qpos0[addr])
            except Exception:
                pass
            idx += 1

    # ── Body / link names ──
    for b in range(int(model.nbody)):
        try:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        except Exception:
            name = None
        if name:
            meta.link_names.append(name)

    # base_link: body 1 by convention (body 0 is the implicit world)
    if int(model.nbody) >= 2:
        try:
            base = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
            meta.base_link = base or ""
        except Exception:
            meta.base_link = ""

    # ── Total mass ──
    try:
        meta.mass = float(sum(model.body_mass))
    except Exception:
        meta.mass = 0.0

    # ── Foot links: heuristic ──
    meta.foot_link_names = [
        n for n in meta.link_names
        if any(kw in n.lower() for kw in _FOOT_KEYWORDS)
    ]

    return meta
