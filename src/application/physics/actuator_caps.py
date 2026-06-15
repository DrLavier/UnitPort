# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Per-joint actuator caps (effort / velocity) from the robot's captured physics.

legged_gym clips each joint's torque to its own URDF ``effort`` limit; UnitPort
historically applied a SINGLE scalar ``effort_limit`` / ``velocity_limit`` to
every joint (RobotNode hidden params, manifest default 30 N·m). For a humanoid
like G1 (hip/knee real limits 88–139 N·m) a flat 30 N·m torque-starves the
load-bearing joints — the robot cannot resist gravity and collapses ("limp").

The real per-joint limits are already captured by *Dump USD* into the registry
``RobotSpec.physics_per_format["USD"]`` (DriveAPI ``max_force`` + per-joint
``max_velocity``; see the USD-drive AUTO pipeline). This module is the single
source that turns that capture into ``{ir_role: {effort, velocity}}`` so every
consumer (IsaacLab env_cfg emit, deploy bundle, SB3 MuJoCo env) clips each joint
to its true motor limit, keyed by the engine-agnostic IR role.

Returns an EMPTY dict when the robot has no captured USD physics — the caller
then falls back to its scalar (and should WARN loudly per CLAUDE.md §8, never
silently ship a wrong cap).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from unitport_sdk import log_warning


def resolve_per_joint_caps(
    sku: str, *, fmt: str = "USD"
) -> Dict[str, Dict[str, Optional[float]]]:
    """Map ``ir_role -> {"effort": max_force, "velocity": max_velocity}``.

    Reads ``RobotSpec.physics_for(fmt)`` (per-joint DriveAPI ``max_force`` and
    ``max_velocity``) and re-keys it from the format's physical joint names to
    IR roles via ``RobotSpec.joints_role_map_for(fmt)`` — so the result is
    engine-agnostic and a consumer in any joint order can look up a cap by role.

    A joint missing from the physics capture, or whose role is unknown, is
    simply absent from the result (the caller decides the fallback). Returns
    ``{}`` when the robot or its physics capture is absent.
    """
    try:
        from registers.robots import get_robot_spec
    except Exception:  # noqa: BLE001 — registry not importable (non-app context)
        return {}
    rs = get_robot_spec(sku) if sku else None
    if rs is None:
        return {}
    try:
        phys = rs.physics_for(fmt) or {}
        role_map = rs.joints_role_map_for(fmt) or {}
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for joint_name, block in phys.items():
        if not isinstance(block, dict):
            continue
        role = role_map.get(joint_name)
        if not role:
            continue
        drive = block.get("drive") if isinstance(block.get("drive"), dict) else {}
        eff = drive.get("max_force")
        vel = block.get("max_velocity")
        if eff is None and vel is None:
            continue
        out[str(role)] = {
            "effort": float(eff) if eff is not None else None,
            "velocity": float(vel) if vel is not None else None,
        }
    return out


# PD-Group config keys that are model-bound and cached per SKU on the canvas
# RobotNode. effort_limit / velocity_limit are the actuator caps; pd_groups are
# the per-group (omega_n, zeta) overrides; pd_param_mode is the parameterization.
PD_GROUP_KEYS: Tuple[str, ...] = (
    "pd_param_mode",
    "pd_groups",
    "effort_limit",
    "velocity_limit",
)


def _sanitize_pd_groups(config: Dict[str, object], valid_group_ids: Optional[set]) -> Dict[str, object]:
    """Drop PD-group overrides whose id isn't valid for the target family.

    A canvas cloned/copied across morphologies (e.g. a humanoid RobotNode
    pasted into a quadruped canvas) carries the source family's groups — a
    quadruped ending up with a ``finger`` override. ``PDParam.with_overrides``
    then raises ``group 'finger' is not in the base parameterization``, which
    BRICKS the PD panel (can't even open it to fix). The reconcile is the
    single funnel that writes ``pd_groups`` onto the node, so sanitize here:
    keep only groups the target family actually has, dropping aliens loudly.
    ``valid_group_ids=None`` (caller couldn't resolve the family) is a no-op —
    never silently empty a config we can't validate (CLAUDE.md §8).
    """
    if valid_group_ids is None:
        return config
    pg = config.get("pd_groups")
    if not isinstance(pg, dict) or not pg:
        return config
    kept = {gid: v for gid, v in pg.items() if gid in valid_group_ids}
    dropped = [gid for gid in pg if gid not in valid_group_ids]
    if dropped:
        log_warning(
            f"[pd-group] dropped alien PD groups {dropped} not in the target "
            f"family's parameterization (valid: {sorted(valid_group_ids)}) — "
            f"the RobotNode was likely cloned across morphologies. Re-run AUTO "
            f"to repopulate from the bound robot."
        )
        config = dict(config)
        config["pd_groups"] = kept
    return config


def plan_pd_group_transition(
    *,
    params: Dict[str, object],
    cache: Dict[str, Dict[str, object]],
    prev_sku: Optional[str],
    new_sku: str,
    resolve: Callable[[str], Optional[Dict[str, object]]],
    valid_group_ids: Optional[set] = None,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    """Pure decision for the per-SKU PD-Group cache + force-load on model switch.

    The canvas RobotNode holds ONE active PD-Group config (the user-approved
    rule: "画布落盘只保存选中的那个机型"). To avoid losing a user's hand-tuned
    PD when they flip the robot model and back, each visited SKU gets an
    in-memory cache slot; switching saves the slot we leave and restores (or
    force-loads from the registry) the slot we enter.

    Args:
        params: the RobotNode's CURRENT params (active config) — read-only here.
        cache: ``{sku: {pd-group key: value}}`` in-memory cache (session-only,
            never serialized to the canvas).
        prev_sku: the SKU ``params`` currently describe, or ``None`` on the
            first reconcile of a freshly loaded / created node.
        new_sku: the SKU being switched to.
        resolve: ``sku -> {pd-group key: value}`` registry force-load (the
            recommended PD-Group config), or ``None`` when the registry has
            nothing (caller then fails loud at training, CLAUDE.md §8).

    Returns ``(updates, new_cache)``:
        * ``updates`` — keys to WRITE onto the RobotNode (empty = no change).
        * ``new_cache`` — the updated cache (caller stashes it on the node).
    """
    cache = {k: dict(v) for k, v in (cache or {}).items()}
    # Snapshot the slot we're leaving so switching back restores the user's edits.
    cur = {k: params[k] for k in PD_GROUP_KEYS if params.get(k) is not None}
    if prev_sku and prev_sku != new_sku:
        cache[prev_sku] = cur

    if not new_sku or new_sku == prev_sku:
        return {}, cache  # no model switch — leave the active config untouched

    if new_sku in cache:
        chosen = dict(cache[new_sku])              # restore the user's slot
    elif prev_sku is None and cur:
        # First reconcile of an already-populated canvas (load) — TRUST what the
        # canvas carries; do not clobber the user's saved config with registry
        # recommendations (this overwrite-on-load was the reported data loss).
        chosen = cur
        cache[new_sku] = dict(cur)
    else:
        chosen = dict(resolve(new_sku) or {})      # genuine switch → force-load
        cache[new_sku] = dict(chosen)

    # Sanitize against the target family's groups. The trusted-canvas branch
    # above can carry a cross-morphology contamination (a quadruped node with a
    # humanoid 'finger' group) that bricks the PD panel; drop aliens here so the
    # node only ever holds groups its family actually parameterizes.
    chosen = _sanitize_pd_groups(chosen, valid_group_ids)
    cache[new_sku] = dict(chosen)

    updates = {k: v for k, v in chosen.items() if params.get(k) != v}
    return updates, cache


__all__ = ["resolve_per_joint_caps", "plan_pd_group_transition", "PD_GROUP_KEYS"]
