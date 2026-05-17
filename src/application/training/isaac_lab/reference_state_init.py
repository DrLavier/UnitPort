"""Reference State Initialization (RSI) — IR-based clip→robot joint mapping.

Maps a flat AMP motion clip frame (canonical FL/FR/RL/RR leg order after
``reorder_legs=True`` in the loader) onto Isaac Lab's
``env_cfg.scene.robot.init_state.joint_pos`` dict, via the single authoritative
IR (intermediate representation) layer: ``BodyIRMapper``.

Mapping chain::

    clip_flat_index
      → canonical IR role id          (QUADRUPED_CLIP_IR_ROLES)
      → USD body name                 (BodyIRMapper.get(role).body)
      → Isaac Lab joint name          (body + joint_suffix)
      → init_state.joint_pos[name]    (per-joint override)

This routes ALL robot-specific naming through the IR mapper stored in the
Robot node — consistent with the rest of the compiler. No hardcoded joint
orderings, no positional index matching against regex dicts.
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
    body_mapping_dict: Dict[str, Any],
    clip_frame: Any,
    joint_suffix: str = "_joint",
    clip_ir_roles: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Translate a flat clip frame into ``{joint_name: value}`` for Isaac Lab.

    Parameters
    ----------
    body_mapping_dict
        The Robot node's ``body_mapping`` param, parsed from JSON. Must have
        ``body_names`` and ``roles`` keys matching :py:meth:`BodyIRMapper.from_dict`.
    clip_frame
        A 1-D iterable of length ``len(clip_ir_roles)`` (12 for quadrupeds).
        Values are target joint positions in the clip's canonical order.
    joint_suffix
        Suffix appended to each IR-resolved body name to form the Isaac Lab
        joint name. Default ``"_joint"`` matches Go2/A1/Unitree convention;
        pass ``""`` for assets where the body name IS the joint name.
    clip_ir_roles
        IR role id for each clip frame index. Defaults to
        :data:`QUADRUPED_CLIP_IR_ROLES`.

    Returns
    -------
    dict
        ``{joint_name: value}`` ready to assign to
        ``env_cfg.scene.robot.init_state.joint_pos``.

    Raises
    ------
    RSIMappingError
        If any required IR role is missing or unresolved in ``body_mapping_dict``,
        or if the clip frame length does not match ``clip_ir_roles``.
    """
    roles = list(clip_ir_roles or QUADRUPED_CLIP_IR_ROLES)
    frame = list(clip_frame)
    if len(frame) != len(roles):
        raise RSIMappingError(
            f"clip frame length {len(frame)} does not match expected "
            f"{len(roles)} (one per IR role {roles!r})"
        )

    try:
        mapper = BodyIRMapper.from_dict(body_mapping_dict)
    except Exception as exc:
        raise RSIMappingError(f"failed to parse body_mapping: {exc}") from exc

    out: Dict[str, float] = {}
    missing: List[str] = []
    for i, role_id in enumerate(roles):
        role = mapper.get(role_id)
        body = role.body if role is not None else None
        if not body:
            missing.append(role_id)
            continue
        joint_name = f"{body}{joint_suffix}"
        out[joint_name] = float(frame[i])

    if missing:
        raise RSIMappingError(
            f"unresolved IR roles: {missing!r}. Open the Robot node's "
            f"Body Mapping editor on the canvas and assign these roles to "
            f"concrete USD body names."
        )

    return out


def validate_mapping(
    body_mapping_dict: Dict[str, Any],
    clip_ir_roles: Optional[List[str]] = None,
) -> List[str]:
    """Return a list of unresolved IR roles (empty = all good).

    Purely static — does not require a clip frame. Useful for canvas-side
    pre-flight gating (same pattern as ``IsaacLabConfigCompiler.validate_body_mapping``).
    """
    roles = list(clip_ir_roles or QUADRUPED_CLIP_IR_ROLES)
    try:
        mapper = BodyIRMapper.from_dict(body_mapping_dict)
    except Exception:
        return list(roles)  # can't parse → everything is unresolved

    return [r for r in roles
            if not (mapper.get(r) and mapper.get(r).body)]
