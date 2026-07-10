# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""C3 Cross-Domain Invariant Catalog + Integrator (blueprint §3.C3 / §4.B4.3).

Invariants that **span two or more domains** and therefore cannot be seen by any
single param expert. Each entry pairs versioned metadata (id / coupled domains /
rationale / responsible domains) with a machine-checkable ``predicate`` over the
canvas snapshot dict. The catalog is **data + predicate registry** (hybrid):
metadata/routing is data, the predicate is code keyed by id — growable by
appending one row + one function (blueprint §3.C3 "grows incrementally from
known footguns, not pre-enumerated").

Seeded with I1–I4 from the blueprint. The Integrator is **read-only**: it
detects violations and re-dispatches (blueprint §5); it never mutates.

Predicates tolerate a partially-built canvas — an absent node is "no violation
yet" (the compile gate owns missing-required-node), not a swallowed error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from ..contracts import AgentRole, IssueCode, RoutedIssue, Severity, ValidationIssue
from ..routing import collapse

__all__ = ["Invariant", "INVARIANTS", "run_integrator"]

Predicate = Callable[[Dict[str, Any]], List[ValidationIssue]]


@dataclass(frozen=True)
class Invariant:
    """One cross-domain invariant."""

    id: str
    domains: Tuple[AgentRole, ...]
    rationale: str
    responsible: Tuple[AgentRole, ...]
    predicate: Predicate


# =============================================================================
# Predicates (I1–I4)
# =============================================================================

def _i1_velocity_command_scale(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """I1 (motion × obs): velocity_command obs scale must be per-channel, not a
    flat scalar that over-amplifies yaw (footgun #5)."""
    obs_node = _first_node(canvas, "il_observation")
    if obs_node is None:
        return []
    obs_terms = _json_param(obs_node, "obs_terms") or {}
    if "velocity_command" not in obs_terms:
        return []
    scale = obs_terms["velocity_command"]
    if isinstance(scale, (list, tuple)):
        return []
    return [_issue(
        IssueCode.GENERIC, Severity.WARNING,
        "velocity_command obs scale is a flat scalar; use a per-channel vector "
        "(e.g. [2.0, 2.0, 0.25]) so yaw is not over-amplified",
        node_id=str(obs_node.get("id", "")), field="velocity_command",
    )]


def _i2_base_lin_vel_privileged(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """I2 (obs × trainer): base_lin_vel must live ONLY in critic_privileged_terms
    (unobservable on hardware; in deployed obs it poisons sim2real — footgun
    #2, a top killer)."""
    obs_node = _first_node(canvas, "il_observation")
    if obs_node is None:
        return []
    obs_terms = _json_param(obs_node, "obs_terms") or {}
    if "base_lin_vel" in obs_terms:
        return [_issue(
            IssueCode.GENERIC, Severity.ERROR,
            "base_lin_vel is in the deployed obs_terms — it is unobservable on "
            "real hardware and poisons sim2real. Move it to "
            "critic_privileged_terms.",
            node_id=str(obs_node.get("id", "")), field="base_lin_vel",
        )]
    return []


def _i3_item_reward_coverage(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """I3 (motion × rewards): every enabled command item needs a reward edge; a
    zero-velocity item (e.g. stand) needs at least one negative term (footgun
    #3 / #6)."""
    motion = _first_node(canvas, "training_motion")
    if motion is None:
        return []
    items = _json_param(motion, "training_items") or {}
    motion_id = str(motion.get("id", ""))
    issues: List[ValidationIssue] = []
    for item_id, entry in items.items():
        if not (isinstance(entry, dict) and entry.get("enabled")):
            continue
        if not _has_reward_edge(canvas, motion_id, item_id):
            issues.append(_issue(
                IssueCode.GENERIC, Severity.ERROR,
                f"enabled command item {item_id!r} has no reward edge — it would "
                f"train on constant-zero reward. Wire a rewards node into "
                f"reward_in__{item_id}.",
                node_id=motion_id, field=item_id,
            ))
            continue
        if _is_zero_velocity_item(item_id) and not _item_has_negative_term(
            canvas, motion_id, item_id
        ):
            issues.append(_issue(
                IssueCode.GENERIC, Severity.WARNING,
                f"zero-velocity item {item_id!r} has no negative reward term — "
                f"the policy may flail limbs in place. Add a penalty.",
                node_id=motion_id, field=item_id,
            ))
    return issues


def _i5_contact_bodies_valid(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """I5 (actor × robot): ``actor_setting.contact_body_names`` may reference
    only body IR roles the robot actually exposes (or the ``.*``/``*`` wildcard =
    all bodies). A role not on the robot is SILENTLY DROPPED by the env_cfg
    compiler's ``_resolve_bodies`` (§8) — flag it here so the agent fixes it to a
    valid subset or ``.*``."""
    actor = _first_node(canvas, "actor_setting")
    if actor is None:
        return []
    roles = _list_param(actor, "contact_body_names")
    explicit = [str(r) for r in roles if str(r) not in (".*", "*")]
    if not explicit:
        return []  # empty / wildcard = all bodies → always valid
    legal = _robot_body_roles(canvas)
    if not legal:
        return []  # robot bodies not resolvable yet → defer to the compile gate
    unknown = [r for r in explicit if r not in legal]
    if not unknown:
        return []
    sample = sorted(legal)[:12]
    return [_issue(
        IssueCode.GENERIC, Severity.ERROR,
        f"actor_setting.contact_body_names references body role(s) {unknown} the "
        f"robot does not have — the env_cfg compiler would silently drop them. Use "
        f"only the robot's body IR roles (e.g. {sample}) or '.*' for all bodies.",
        node_id=str(actor.get("id", "")), field="contact_body_names",
    )]


def _i4_backend_feature_compat(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """I4 (backend × all): a node from the wrong backend family must not be
    wired (thin cross-check; the compile gate is authoritative on the full
    SB3/IL feature matrix)."""
    backend = str(canvas.get("backend", ""))
    schema_ids = {str(n.get("schema_id")) for n in canvas.get("nodes", [])}
    if backend == "sb3_mujoco" and "il_ppo_trainer" in schema_ids:
        return [_issue(
            IssueCode.BACKEND_ALGORITHM_MISMATCH, Severity.ERROR,
            "sb3_mujoco canvas wires an il_ppo_trainer (IsaacLab-only). Use the "
            "SB3 trainer path (algorithm_config + train).",
        )]
    if backend == "isaac_lab" and "algorithm_config" in schema_ids and "train" in schema_ids:
        return [_issue(
            IssueCode.AMBIGUOUS_ALGORITHM_SOURCE, Severity.ERROR,
            "isaac_lab canvas wires the SB3 path (algorithm_config + train). "
            "Use il_ppo_trainer instead.",
        )]
    return []


# =============================================================================
# Catalog
# =============================================================================

INVARIANTS: List[Invariant] = [
    Invariant(
        id="I1",
        domains=(AgentRole.MOTION, AgentRole.OBS_ROBUSTNESS),
        rationale="velocity_command obs scale must agree in units with the "
                  "command channel scale",
        responsible=(AgentRole.OBS_ROBUSTNESS, AgentRole.MOTION),
        predicate=_i1_velocity_command_scale,
    ),
    Invariant(
        id="I2",
        domains=(AgentRole.OBS_ROBUSTNESS, AgentRole.TRAINER),
        rationale="base_lin_vel belongs only in critic_privileged_terms",
        responsible=(AgentRole.OBS_ROBUSTNESS,),
        predicate=_i2_base_lin_vel_privileged,
    ),
    Invariant(
        id="I3",
        domains=(AgentRole.MOTION, AgentRole.REWARDS),
        rationale="every enabled command item has a reward edge; a zero-velocity "
                  "item carries at least one negative term",
        responsible=(AgentRole.MOTION, AgentRole.REWARDS),
        predicate=_i3_item_reward_coverage,
    ),
    Invariant(
        id="I4",
        domains=(AgentRole.DESIGN,),
        rationale="no feature incompatible with the locked backend",
        responsible=(AgentRole.DESIGN,),
        predicate=_i4_backend_feature_compat,
    ),
    Invariant(
        id="I5",
        domains=(AgentRole.OBS_ROBUSTNESS, AgentRole.DESIGN),
        rationale="contact_body_names references only body IR roles the robot has",
        responsible=(AgentRole.OBS_ROBUSTNESS,),
        predicate=_i5_contact_bodies_valid,
    ),
]


def run_integrator(
    canvas: Dict[str, Any], active_roles: FrozenSet[AgentRole]
) -> List[RoutedIssue]:
    """Run every invariant; return routed issues (responsible role collapsed
    onto the active roster)."""
    routed: List[RoutedIssue] = []
    for inv in INVARIANTS:
        try:
            found = inv.predicate(canvas)
        except Exception as exc:  # a predicate bug must not crash the run
            # Surface loudly as a generic issue rather than silently dropping —
            # a broken invariant is a framework bug worth seeing (§8).
            found = [_issue(
                IssueCode.GENERIC, Severity.WARNING,
                f"invariant {inv.id} predicate raised: {exc!r}",
            )]
        # Route to the invariant's first responsible domain (collapsed).
        target = inv.responsible[0] if inv.responsible else AgentRole.ORCHESTRATOR
        responsible = collapse(target, active_roles)
        for issue in found:
            routed.append(RoutedIssue(issue=issue, responsible=responsible))
    return routed


# =============================================================================
# snapshot helpers (pure)
# =============================================================================

def _first_node(canvas: Dict[str, Any], schema_id: str) -> Optional[Dict[str, Any]]:
    for n in canvas.get("nodes", []):
        if str(n.get("schema_id")) == schema_id and not n.get("disabled"):
            return n
    return None


def _json_param(node: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    cell = (node.get("params") or {}).get(key)
    if cell is None:
        return None
    raw = cell.get("value") if isinstance(cell, dict) else cell
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    return raw if isinstance(raw, dict) else None


def _list_param(node: Dict[str, Any], key: str) -> List[Any]:
    """Parse a JSON-list param cell (contact_body_names etc.); [] on any miss."""
    cell = (node.get("params") or {}).get(key)
    if cell is None:
        return []
    raw = cell.get("value") if isinstance(cell, dict) else cell
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _robot_body_roles(canvas: Dict[str, Any]) -> FrozenSet[str]:
    """Body IR roles the canvas' robot exposes (union across asset formats).

    Empty when there is no robot node or its asset_id can't be resolved to a
    registered SKU with a body table — the caller then defers (no false flag)."""
    robot = _first_node(canvas, "robot")
    if robot is None:
        return frozenset()
    cell = (robot.get("params") or {}).get("asset_id")
    raw = cell.get("value") if isinstance(cell, dict) else cell
    asset_id = str(raw or "").strip()
    if not asset_id:
        return frozenset()
    from registers import robots as _robots

    sku = _robots.resolve_to_sku(asset_id)
    if not sku:
        return frozenset()
    roles: set = set()
    for fmt in ("USD", "MJCF", "URDF"):
        for role in _robots.get_body_role_map(sku, fmt).values():
            if role:
                roles.add(str(role))
    return frozenset(roles)


def _has_reward_edge(canvas: Dict[str, Any], motion_id: str, item_id: str) -> bool:
    port = f"reward_in__{item_id}"
    return any(
        str(e.get("target_node")) == motion_id and str(e.get("target_port")) == port
        for e in canvas.get("edges", [])
    )


def _reward_nodes_for_item(
    canvas: Dict[str, Any], motion_id: str, item_id: str
) -> List[str]:
    port = f"reward_in__{item_id}"
    srcs: List[str] = []
    for e in canvas.get("edges", []):
        if str(e.get("target_node")) == motion_id and str(e.get("target_port")) == port:
            srcs.append(str(e.get("source_node")))
    return srcs


def _item_has_negative_term(
    canvas: Dict[str, Any], motion_id: str, item_id: str
) -> bool:
    node_by_id = {str(n.get("id")): n for n in canvas.get("nodes", [])}
    for rid in _reward_nodes_for_item(canvas, motion_id, item_id):
        node = node_by_id.get(rid)
        if node is None:
            continue
        terms = _json_param(node, "reward_terms") or {}
        for page in terms.values():
            if not isinstance(page, dict):
                continue
            for payload in page.values():
                w = _weight_of(payload)
                if w is not None and w < 0:
                    return True
    return False


def _is_zero_velocity_item(item_id: str) -> bool:
    """True if the registered command for this item has an all-zero envelope."""
    from registers import commands as _commands

    item = _commands.get_item(item_id)
    if item is None:
        return False
    template = item.command_template or {}
    if not template:
        return False
    return all(
        float(lo) == 0.0 and float(hi) == 0.0 for (lo, hi) in template.values()
    )


def _weight_of(payload: Any) -> Optional[float]:
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict) and "weight" in payload:
        w = payload.get("weight")
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            return float(w)
        if isinstance(w, str):
            try:
                return float(w.strip())
            except ValueError:
                return None
    return None


def _issue(
    code: IssueCode, severity: Severity, message: str, *,
    node_id: str = "", field: str = "",
) -> ValidationIssue:
    return ValidationIssue(
        code=code, severity=severity, message=message, node_id=node_id, field=field
    )
