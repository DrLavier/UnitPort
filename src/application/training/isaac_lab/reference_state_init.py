# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reference State Initialization (RSI) — IR-based clip→robot joint mapping.

Maps a flat AMP motion clip frame (canonical FL/FR/RL/RR leg order after
``reorder_legs=True`` in the loader) onto Isaac Lab's
``env_cfg.scene.robot.init_state.joint_pos`` dict via the IR layer.

Mapping chain (Stage 4 — per-format)::

    clip_flat_index
      → canonical IR role id          (clip.ir_roles[i] OR QUADRUPED_CLIP_IR_ROLES)
      → physical joint name           (joint_names_by_ir[role])
      → init_state.joint_pos[name]    (per-joint override)

The ``joint_names_by_ir`` table is built from the robot's
``joints_per_format[active_format]`` — the same data the env_cfg uses to
emit JointPositionActionCfg.joint_names — so RSI lands on the **exact
physical joint names** the runtime is loading. This drops the legacy
body+suffix heuristic (``body_name + "_joint"``) which only worked for
Unitree-style naming (FL_thigh → FL_thigh_joint) and silently failed on
Boston Dynamics Spot (joint ``fl_hy`` doesn't share a stem with body
``fl_uleg``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from application.training.body_ir import BodyIRMapper
from application.training.motion_ir_mapping import (
    QUADRUPED_AMP_IR_ROLES,
    get_ir_roles,
)

# Backwards-compatible alias. Prefer ``get_ir_roles(format_id)`` from
# motion_ir_mapping for new code — it's format-aware and covers future
# biped / humanoid clip contracts.
QUADRUPED_CLIP_IR_ROLES: List[str] = QUADRUPED_AMP_IR_ROLES


class RSIMappingError(RuntimeError):
    """Raised when the clip → IR → robot joint chain cannot be resolved."""


def build_joint_pos_dict(
    clip_frame: Any,
    joint_names_by_ir: Dict[str, str],
    *,
    clip_ir_roles: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Translate a flat clip frame into ``{joint_name: value}`` for Isaac Lab.

    Stage 4 signature: takes an IR-role → physical-joint-name lookup
    table directly (built from ``RobotSpecRef.joints_role_map_for(
    active_format)`` and inverted). No body+suffix heuristic.

    Parameters
    ----------
    clip_frame
        1-D iterable of length ``len(clip_ir_roles)`` (12 for quadrupeds).
        Values are target joint positions in the clip's canonical order.
    joint_names_by_ir
        ``{ir_role: joint_name}`` for the robot in its active format. Build
        via:

            joint_names_by_ir = {
                ir: jn for jn, ir in
                spec.robot.joints_role_map_for(active_format).items()
            }

    clip_ir_roles
        IR role id for each clip frame index. Defaults to
        :data:`QUADRUPED_CLIP_IR_ROLES`. New clips should set this from
        ``MotionClip.ir_roles`` so the caller doesn't pin a quadruped
        assumption.

    Returns
    -------
    dict
        ``{joint_name: value}`` ready to assign to
        ``env_cfg.scene.robot.init_state.joint_pos``.

    Raises
    ------
    RSIMappingError
        If any required IR role is missing from ``joint_names_by_ir``, or
        if the clip frame length does not match ``clip_ir_roles``.
    """
    roles = list(clip_ir_roles or QUADRUPED_CLIP_IR_ROLES)
    frame = list(clip_frame)
    if len(frame) != len(roles):
        raise RSIMappingError(
            f"clip frame length {len(frame)} does not match expected "
            f"{len(roles)} (one per IR role {roles!r})"
        )
    if not isinstance(joint_names_by_ir, dict) or not joint_names_by_ir:
        raise RSIMappingError(
            "joint_names_by_ir is empty; pass the robot's per-format "
            "joints_role_map_for(active_format) inverted to {ir_role: "
            "joint_name}"
        )

    out: Dict[str, float] = {}
    missing: List[str] = []
    for i, role_id in enumerate(roles):
        joint_name = joint_names_by_ir.get(role_id)
        if not joint_name:
            missing.append(role_id)
            continue
        out[joint_name] = float(frame[i])

    if missing:
        raise RSIMappingError(
            f"unresolved IR roles: {missing!r} — not present in the robot's "
            f"per-format joint table. Open Robot Asset → Dump MJCF/USD or "
            f"verify the active format declares these joint roles."
        )

    return out


def validate_mapping(
    joint_names_by_ir: Dict[str, str],
    clip_ir_roles: Optional[List[str]] = None,
) -> List[str]:
    """Return a list of unresolved IR roles (empty = all good).

    Purely static — does not require a clip frame. Useful for canvas-side
    pre-flight gating.
    """
    roles = list(clip_ir_roles or QUADRUPED_CLIP_IR_ROLES)
    if not isinstance(joint_names_by_ir, dict):
        return list(roles)
    return [r for r in roles if not joint_names_by_ir.get(r)]
