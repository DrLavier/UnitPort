"""Joint name canonicalization + IR translation helpers.

Two layers:
  1. :func:`canonicalize_joint_name` — strips ``_joint`` suffix and
     normalises case for legacy string-match fallbacks.  Used when an
     incoming bundle/asset uses slightly different spellings of the same
     physical joint (e.g. ``FL_hip_joint`` vs ``FL_hip``).
  2. :func:`ir_role_to_physical_name` / :func:`ir_roles_to_physical_names`
     / :func:`physical_to_ir_role` — Phase 5 IR Joint contract.  These
     resolve a registered IR role (``hip_FL``) to the bound robot's
     physical joint name (``FL_hip_joint``) by reading
     ``registers.robots.RobotSpec.joint_order ↔ joint_ir_roles`` parallel
     arrays.  Deploy-side code (PolicyRunner / ActionApplier / PDController
     / joint_space) calls these at the substrate boundary so the bundle's
     IR-only ``robot.joint_names`` can bind to a concrete MJCF / URDF /
     SDK actuator order.
"""
from __future__ import annotations

from typing import Iterable, List


def canonicalize_joint_name(name: str) -> str:
    """Normalize equivalent joint/actuator names to a shared runtime form."""
    text = str(name or "").strip()
    if text.endswith("_joint"):
        return text[:-6]
    return text


def canonicalize_joint_names(names: Iterable[str]) -> List[str]:
    """Return canonicalized joint names preserving input order."""
    return [canonicalize_joint_name(name) for name in names]


# ---------------------------------------------------------------------------
# Phase 5 — IR-only joint contract helpers
# ---------------------------------------------------------------------------


def ir_role_to_physical_name(ir_role: str, robot_sku: str) -> str:
    """Translate one IR role to the bound robot's physical joint name.

    Looks up ``robot_sku`` via :func:`registers.robots.resolve_id` +
    :func:`registers.robots.get_robot_spec`, then walks the parallel
    ``joint_ir_roles`` ↔ ``joint_order`` arrays.

    Raises :class:`ValueError` listing all valid IR roles when the lookup
    fails — message is designed to land in deploy-side logs and tell the
    operator exactly what's wrong (typically: bundle was trained against
    a robot whose IR set differs from the deploy target's).
    """
    if not ir_role:
        raise ValueError("ir_role_to_physical_name: ir_role must be non-empty")
    if not robot_sku:
        raise ValueError("ir_role_to_physical_name: robot_sku must be non-empty")

    from registers.robots import get_robot_spec, resolve_id

    canonical = resolve_id(robot_sku) or robot_sku
    rs = get_robot_spec(canonical)
    if rs is None:
        raise ValueError(
            f"ir_role_to_physical_name: robot_sku {robot_sku!r} (canonical "
            f"{canonical!r}) is not registered. Use registers.robots.list_skus() "
            f"to enumerate available robots."
        )
    try:
        idx = rs.joint_ir_roles.index(ir_role)
    except ValueError:
        raise ValueError(
            f"ir_role_to_physical_name: IR role {ir_role!r} not declared for "
            f"robot {canonical!r} ({rs.name!r}). Valid IR roles: "
            f"{sorted(rs.joint_ir_roles)}"
        ) from None
    return rs.joint_order[idx]


def ir_roles_to_physical_names(
    ir_roles: Iterable[str],
    robot_sku: str,
) -> List[str]:
    """Translate a list of IR roles to physical joint names (order preserved).

    Validates the entire list first by collecting all bad tokens, so the
    operator gets one error listing all problems instead of failing on
    the first miss.
    """
    if not robot_sku:
        raise ValueError("ir_roles_to_physical_names: robot_sku must be non-empty")
    ir_roles = list(ir_roles)
    if not ir_roles:
        return []

    from registers.robots import get_robot_spec, resolve_id

    canonical = resolve_id(robot_sku) or robot_sku
    rs = get_robot_spec(canonical)
    if rs is None:
        raise ValueError(
            f"ir_roles_to_physical_names: robot_sku {robot_sku!r} (canonical "
            f"{canonical!r}) is not registered."
        )
    valid = list(rs.joint_ir_roles)
    valid_set = set(valid)
    bad = [r for r in ir_roles if r not in valid_set]
    if bad:
        raise ValueError(
            f"ir_roles_to_physical_names: the following IR roles are not "
            f"declared for robot {canonical!r} ({rs.name!r}): {sorted(bad)}. "
            f"Valid IR roles for this robot: {sorted(valid)}"
        )
    return [rs.joint_order[valid.index(r)] for r in ir_roles]


def physical_to_ir_role(physical_name: str, robot_sku: str) -> str:
    """Translate a physical joint name back to its IR role.

    Used by deploy-side code that reads physical names from a runtime
    asset (MJCF actuator name, URDF joint name) and needs to look up the
    corresponding IR role to compare with bundle metadata.
    """
    if not physical_name:
        raise ValueError("physical_to_ir_role: physical_name must be non-empty")
    if not robot_sku:
        raise ValueError("physical_to_ir_role: robot_sku must be non-empty")

    from registers.robots import get_robot_spec, resolve_id

    canonical = resolve_id(robot_sku) or robot_sku
    rs = get_robot_spec(canonical)
    if rs is None:
        raise ValueError(
            f"physical_to_ir_role: robot_sku {robot_sku!r} (canonical "
            f"{canonical!r}) is not registered."
        )
    ir = rs.body_role_map.get(physical_name)
    if ir is None:
        raise ValueError(
            f"physical_to_ir_role: physical joint {physical_name!r} not "
            f"declared for robot {canonical!r} ({rs.name!r}). Valid physical "
            f"joints: {sorted(rs.joint_order)}"
        )
    return ir


__all__ = [
    "canonicalize_joint_name",
    "canonicalize_joint_names",
    "ir_role_to_physical_name",
    "ir_roles_to_physical_names",
    "physical_to_ir_role",
]
