# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Frame → ``qpos`` application for kinematic clip playback (shared impl).

Extracted from ``MotionReviewTask`` so the external MuJoCo viewer and the
embedded :class:`~application.service.runtime.simulation.mujoco.clip_render_session.ClipRenderSession`
write reference frames into ``mj_data.qpos`` through ONE code path
(CLAUDE.md §11). The IR-role joint routing and the scalar-last→scalar-first
quaternion reorder are the kind of detail that, if duplicated, drifts
silently between two consumers and produces a wrong-but-plausible pose
(the §8 "illusion of success" failure) — so it lives exactly once, here.

``mujoco`` is passed in as an argument and never imported at module top:
this keeps the module importable in venvs without the GL stack (Kit
worker / headless CI), matching ``MotionReviewTask``'s lazy-import rule.

Joint routing is by **IR role** (CLAUDE.md canvas-ir-only / §1):
the clip stores ``joint_pos`` columns in canonical IR-role order
(``MotionClip.ir_roles``); the MJCF stores joints under physical names.
``build_name_to_role`` maps physical-name → IR-role via the registry and
``build_joint_plan`` routes each clip column to the matching MJCF qpos slot.

Quaternion convention: ``MotionClip.root_quat`` is scalar-last
``(x, y, z, w)``; MuJoCo's free-joint qpos quaternion is scalar-first
``(w, x, y, z)``. :func:`apply_frame` reorders on write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FrameApplyPlan:
    """Precomputed routing for writing clip frames into one model's qpos.

    Attributes
    ----------
    free_qposadr:
        qpos address of the free-joint root, or ``-1`` when the model has
        no free root.
    has_freejoint_root:
        Whether the model has a free-joint root (root pose is written only
        then).
    joint_plan:
        ``(qposadr, clip_col | None)`` over the actuated joints (free / ball
        roots excluded, in MuJoCo joint order). ``None`` columns keep the
        MJCF rest value.
    matched:
        Count of joints that found a clip column (``0`` means the clip does
        not target this robot — :func:`build_frame_apply_plan` raises).
    """

    free_qposadr: int
    has_freejoint_root: bool
    joint_plan: List[Tuple[int, Optional[int]]]
    matched: int


def build_name_to_role(sku: str) -> Dict[str, str]:
    """Build a physical-MJCF-joint-name → IR-role lookup for ``sku``.

    Reads ``registers.robots.get_robot(sku)['joints_per_format']['MJCF']``.
    Raises loud (CLAUDE.md §8) when the SKU is unregistered or has no MJCF
    joint table — a silent empty map would route zero clip columns and
    render a frozen zero pose.
    """
    from registers import robots as _robots_registry

    entry = _robots_registry.get_robot(sku) or {}
    if not entry:
        raise RuntimeError(
            f"[motion/clip] robot SKU {sku!r} is not registered — cannot "
            f"build the IR→MJCF joint mapping for playback."
        )
    joints_dict = entry.get("joints_per_format", {}).get("MJCF", {}) or {}
    if not isinstance(joints_dict, dict) or not joints_dict:
        raise RuntimeError(
            f"[motion/clip] robot {sku!r} has no joints_per_format['MJCF'] "
            f"entries in the registry — cannot map IR roles to MJCF physical "
            f"joint names. Add MJCF joints to "
            f"registers/data/robots_canonical.json (or the user overlay)."
        )
    name_to_role: Dict[str, str] = {}
    for jspec in joints_dict.values():
        if not isinstance(jspec, dict):
            continue
        n = str(jspec.get("name", ""))
        r = str(jspec.get("ir_role", ""))
        if n:
            name_to_role[n] = r
    return name_to_role


def discover_free_root(mujoco: Any, model: Any) -> Tuple[int, bool]:
    """Locate the free-joint root: return ``(free_qposadr, has_root)``."""
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid]), True
    return -1, False


def build_joint_plan(
    mujoco: Any, model: Any, clip: Any, sku: str,
) -> Tuple[List[Tuple[int, Optional[int]]], int]:
    """Map each actuated MJCF joint to a clip column by IR role.

    Returns ``(plan, matched)``. ``plan`` is a list of
    ``(qposadr, clip_col | None)`` over the actuated joints (free / ball
    roots excluded, in MuJoCo joint order); ``matched`` counts joints that
    found a clip column. Joints with no clip column (``None``) keep their
    MJCF rest value during playback.
    """
    name_to_role = build_name_to_role(sku)
    clip_roles = list(clip.ir_roles)
    role_to_col: Dict[str, int] = {}
    for i, role in enumerate(clip_roles):
        # First occurrence wins — IR roles are unique per skeleton.
        role_to_col.setdefault(str(role), i)

    plan: List[Tuple[int, Optional[int]]] = []
    matched = 0
    for jid in range(int(model.njnt)):
        jtype = int(model.jnt_type[jid])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
            continue
        qposadr = int(model.jnt_qposadr[jid])
        # mj_id2name returns the name or None for a valid jid (it does not
        # raise) — a falsy name simply has no IR role (§8: no try/except→None
        # masking a partial failure into a silently frozen joint).
        jname = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_JOINT), jid)
        ir_role = name_to_role.get(str(jname or ""), "")
        col = role_to_col.get(ir_role) if ir_role else None
        if col is not None:
            matched += 1
        plan.append((qposadr, col))
    return plan, matched


def build_frame_apply_plan(
    mujoco: Any, model: Any, clip: Any, sku: str, *, clip_label: str = "",
) -> FrameApplyPlan:
    """Compose the free-root + joint-routing plan, failing loud (§8).

    Raises when none of the clip's joint channels map onto the robot — a
    clip prepared for a different robot / family would otherwise render a
    frozen zero pose instead of telling the user why.
    """
    free_qposadr, has_root = discover_free_root(mujoco, model)
    joint_plan, matched = build_joint_plan(mujoco, model, clip, sku)
    if matched == 0:
        label = clip_label or getattr(clip, "name", "?")
        roles = [str(r) for r in (clip.ir_roles or [])]
        sample = ", ".join(roles[:4]) + (" …" if len(roles) > 4 else "")
        raise RuntimeError(
            f"[motion/clip] clip {label!r} does not match robot {sku!r}: none "
            f"of its {len(roles)} joint channels (e.g. {sample or '∅'}) map onto "
            f"the robot's MJCF joints. This clip targets a different robot / "
            f"family — pick the matching render robot."
        )
    return FrameApplyPlan(
        free_qposadr=free_qposadr,
        has_freejoint_root=has_root,
        joint_plan=joint_plan,
        matched=matched,
    )


def apply_frame(
    mujoco: Any, model: Any, data: Any, clip: Any, t: float, plan: FrameApplyPlan,
) -> None:
    """Write the clip frame at time ``t`` into ``data.qpos`` + forward.

    Root pose (when both the clip carries one and the MJCF has a free root)
    goes to the free-joint slots, reordering the clip's scalar-last
    quaternion into MuJoCo's scalar-first layout. Joint angles route by the
    precomputed IR-role plan. Ends with ``mj_forward`` so ``xpos`` (and the
    rendered geometry) reflects the new ``qpos``.

    Note: ``clip.frame_at(N * clip.frame_dt)`` returns the *raw* frame N
    (interpolation alpha is 0 at an integer multiple), so a caller passing
    ``frame_index * frame_dt`` renders that exact frame, not a blend.
    """
    frame = clip.frame_at(t)

    if plan.has_freejoint_root and plan.free_qposadr >= 0:
        a = plan.free_qposadr
        root_pos = frame.get("root_pos")
        if root_pos is not None and len(root_pos) >= 3:
            data.qpos[a + 0] = float(root_pos[0])
            data.qpos[a + 1] = float(root_pos[1])
            data.qpos[a + 2] = float(root_pos[2])
        root_quat = frame.get("root_quat")
        if root_quat is not None and len(root_quat) >= 4:
            # Clip is (x, y, z, w); MuJoCo qpos is (w, x, y, z).
            data.qpos[a + 3] = float(root_quat[3])
            data.qpos[a + 4] = float(root_quat[0])
            data.qpos[a + 5] = float(root_quat[1])
            data.qpos[a + 6] = float(root_quat[2])

    joint_pos = frame.get("joint_pos")
    if joint_pos is not None:
        n_cols = len(joint_pos)
        for qposadr, col in plan.joint_plan:
            if col is None or col >= n_cols:
                continue
            data.qpos[qposadr] = float(joint_pos[col])

    mujoco.mj_forward(model, data)


__all__ = [
    "FrameApplyPlan",
    "build_name_to_role",
    "discover_free_root",
    "build_joint_plan",
    "build_frame_apply_plan",
    "apply_frame",
]
