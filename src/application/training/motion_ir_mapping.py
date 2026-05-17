"""Motion clip ↔ IR layer contract.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/motion_ir_mapping.py``.

Single source of truth for "which IR role does clip joint index N
correspond to, for format X". Consumed by:

  * Reference State Init (RSI) — frame 0 of the chosen clip is routed
    through the IR layer when seeding ``init_state.joint_pos``.
  * Reference Motion UI — per-clip ✔/❌ auto-match indicators read
    :func:`validate_clip_ir_match` to surface IR-misalignment to the
    user before training submit.
  * Training launch pre-flight — guard against starting a run whose
    motion pack has unresolvable joints.

Stage 5 changes vs DEMO:
  * Imports — ``BodyIRMapper`` (DEMO ``src.system.training.body_ir``) is
    replaced by :class:`registers.robots.RobotSpec`. No more ``hf_downloader``
    / ``il_manifest_compat`` references (red-line deletion targets).
  * The IR role table is shared with :mod:`motion.validator` so adding a
    new format means editing exactly one file.

Design rule (carries forward from DEMO)::

    NO code that touches robot joints by name or index may bypass this
    contract. Every clip→robot mapping goes
        clip_index → IR role → joint name (RobotSpec.joint_ir_roles)
    Adding a new clip format = adding a new entry to FORMAT_IR_ROLES;
    no other files change.

Contracts:
  * ``amp_legged_gym`` — after ``AMPLegGymLoader(reorder_legs=True)``
    the 12-DoF quadruped joint array is in FL/FR/RL/RR × (hip, thigh, calf)
    order.
  * ``unitport_npy`` — legacy format; joint order matches the env's
    native ordering. Each clip carries its own assumed order (no
    cross-robot contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from registers.robots import RobotSpec


# ════════════════════════════════════════════════════════════════════════════
# Per-format IR role tables
# ════════════════════════════════════════════════════════════════════════════

# 12-DoF quadruped, canonical order after AMPLegGymLoader reorder
# (FR/FL/RR/RL → FL/FR/RL/RR). Contract pinned with
# ``loader._reorder_quad_legs``.
QUADRUPED_AMP_IR_ROLES: List[str] = [
    "hip_FL",   "thigh_FL", "calf_FL",   # clip indices 0-2
    "hip_FR",   "thigh_FR", "calf_FR",   # clip indices 3-5
    "hip_RL",   "thigh_RL", "calf_RL",   # clip indices 6-8
    "hip_RR",   "thigh_RR", "calf_RR",   # clip indices 9-11
]

# Future: BIPED_AMP_IR_ROLES, HUMANOID_AMP_IR_ROLES, etc.

FORMAT_IR_ROLES: Dict[str, List[str]] = {
    "amp_legged_gym": QUADRUPED_AMP_IR_ROLES,
    # unitport_npy is intentionally absent — those clips carry no portable
    # joint-order contract and must be matched by the env directly.
}


# ════════════════════════════════════════════════════════════════════════════
# Validation result
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ClipIRStatus:
    """Result of validating one motion clip against a robot's IR mapping."""

    ok: bool                      # True iff every expected IR role resolves
    clip_name: str = ""
    format_id: str = ""
    joint_count: int = 0          # DoF present in the clip
    expected_roles: List[str] = field(default_factory=list)
    missing_roles: List[str] = field(default_factory=list)
    unmapped_bodies: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def badge(self) -> str:
        """One-character indicator for UI (✔ / ❌ / —)."""
        if not self.expected_roles:
            return "—"
        return "✔" if self.ok else "❌"


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════


def get_ir_roles(format_id: str) -> List[str]:
    """Return the canonical IR role list for a clip format.

    Empty list when the format has no portable contract (e.g.
    ``unitport_npy``). Callers must treat an empty result as
    "cannot validate, skip IR gate".
    """
    return list(FORMAT_IR_ROLES.get(format_id, []))


def map_clip_index_to_joint(
    format_id: str,
    clip_index: int,
    robot_spec: "RobotSpec",
) -> Optional[str]:
    """Resolve clip joint index → robot joint name via the IR layer.

    Returns ``None`` if the format has no IR contract, the index is
    out of range, or the role is not present on this robot.
    """
    roles = FORMAT_IR_ROLES.get(format_id)
    if not roles or not (0 <= clip_index < len(roles)):
        return None
    role = roles[clip_index]
    for jname, jrole in zip(robot_spec.joint_order, robot_spec.joint_ir_roles):
        if jrole == role:
            return jname
    return None


def validate_clip_ir_match(
    clip_path: Union[str, Path],
    robot_spec: "RobotSpec",
    format_id: str = "amp_legged_gym",
    _cached_joint_count: Optional[int] = None,
) -> ClipIRStatus:
    """Check whether a single clip's joints resolve cleanly against the
    robot's IR mapping.

    The heavy path (loading the clip file) is only hit when
    ``_cached_joint_count`` is not provided. Callers iterating over many
    clips should pre-load joint counts to avoid repeated disk reads.
    """
    p = Path(clip_path)
    clip_name = p.stem or p.name or str(p)
    expected = get_ir_roles(format_id)

    if not expected:
        return ClipIRStatus(
            ok=False,
            clip_name=clip_name,
            format_id=format_id,
            joint_count=0,
            expected_roles=[],
            message=f"format {format_id!r} has no portable IR contract",
        )

    if _cached_joint_count is None:
        try:
            from application.training.motion.loader import get_loader
            loader = get_loader(format_id)
            clip = loader.load(clip_path)
            jp = getattr(clip, "joint_pos", None)
            joint_count = (
                int(jp.shape[1]) if jp is not None and getattr(jp, "ndim", 0) >= 2
                else 0
            )
        except Exception as exc:  # noqa: BLE001
            return ClipIRStatus(
                ok=False,
                clip_name=clip_name,
                format_id=format_id,
                joint_count=0,
                expected_roles=expected,
                missing_roles=list(expected),
                message=f"failed to load clip: {exc}",
            )
    else:
        joint_count = int(_cached_joint_count)

    if joint_count != len(expected):
        return ClipIRStatus(
            ok=False,
            clip_name=clip_name,
            format_id=format_id,
            joint_count=joint_count,
            expected_roles=expected,
            missing_roles=list(expected),
            message=(
                f"clip has {joint_count} joints but format {format_id!r} "
                f"expects {len(expected)}"
            ),
        )

    have = set(robot_spec.joint_ir_roles)
    unmapped = [r for r in expected if r not in have]
    if unmapped:
        return ClipIRStatus(
            ok=False,
            clip_name=clip_name,
            format_id=format_id,
            joint_count=joint_count,
            expected_roles=expected,
            unmapped_bodies=unmapped,
            message=(
                f"{len(unmapped)} IR role(s) unmapped for robot "
                f"{robot_spec.sku!r}: {unmapped!r}. Edit the robot's body "
                f"mapping or pick a robot whose joints cover the required roles."
            ),
        )

    return ClipIRStatus(
        ok=True,
        clip_name=clip_name,
        format_id=format_id,
        joint_count=joint_count,
        expected_roles=expected,
        message="all IR roles resolve",
    )


def validate_clips_ir_match(
    clip_paths: List[Union[str, Path]],
    robot_spec: "RobotSpec",
    format_id: str = "amp_legged_gym",
) -> Dict[str, ClipIRStatus]:
    """Batch-validate a list of clips. Returns ``{clip_path: status}``."""
    return {
        str(path): validate_clip_ir_match(path, robot_spec, format_id)
        for path in clip_paths
    }


__all__ = [
    "QUADRUPED_AMP_IR_ROLES",
    "FORMAT_IR_ROLES",
    "ClipIRStatus",
    "get_ir_roles",
    "map_clip_index_to_joint",
    "validate_clip_ir_match",
    "validate_clips_ir_match",
]
