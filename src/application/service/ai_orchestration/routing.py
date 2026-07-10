# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""C5 issue routing — map a :class:`ValidationIssue` to the agent role
responsible for repairing it (blueprint §3.C5 ``responsible`` field).

Every correctness tier (the C1 tier-1 check, the compile gate, the C3
integrator, the critic) emits the same :class:`ValidationIssue` shape, so the
scheduler can route them uniformly. The mapping is **deterministic** and keyed
by :class:`IssueCode`; it returns a role from the *full* blueprint roster so the
table stays valid as the team grows. :func:`collapse` folds a granular role onto
the set of roles actually instantiated in the current run (v1: every producer
role collapses to ``ORCHESTRATOR``, which holds all write tools).
"""
from __future__ import annotations

from typing import Dict, FrozenSet

from .contracts import AgentRole, IssueCode, RoutedIssue, ValidationIssue

__all__ = ["route", "collapse", "to_routed", "PRODUCER_FALLBACK"]

# The universal producer: any granular producer role not active in a run is
# repaired by the Orchestrator (which owns place/connect/set_param + semantic
# ops). The blueprint's read-only roles (Integrator/Critic) never produce, so
# they are never a repair target.
PRODUCER_FALLBACK = AgentRole.ORCHESTRATOR


# Granular responsible role per issue code (blueprint §4 Integrator-vs-expert
# split + §3.C3 responsible-domains). Codes absent here fall back to the
# Orchestrator (structural / unclassified → the producer that owns the graph).
_BY_CODE: Dict[IssueCode, AgentRole] = {
    # ---- structure / wiring → the graph owner ----
    IssueCode.MISSING_REQUIRED_NODE: AgentRole.ORCHESTRATOR,
    IssueCode.MISSING_REQUIRED_PORT: AgentRole.ORCHESTRATOR,
    IssueCode.DANGLING_PORT: AgentRole.ORCHESTRATOR,
    IssueCode.OBS_OUTPUT_UNWIRED: AgentRole.ORCHESTRATOR,
    IssueCode.INCOMPLETE_AMP_WIRING: AgentRole.ORCHESTRATOR,
    # ---- backend / robot identity → the Design lock ----
    IssueCode.AMBIGUOUS_ALGORITHM_SOURCE: AgentRole.DESIGN,
    IssueCode.BACKEND_ALGORITHM_MISMATCH: AgentRole.DESIGN,
    IssueCode.BACKEND_REGISTRY_MISMATCH: AgentRole.DESIGN,
    IssueCode.BACKEND_UNAVAILABLE: AgentRole.DESIGN,
    IssueCode.UNKNOWN_ROBOT: AgentRole.DESIGN,
    IssueCode.UNRESOLVED_ROBOT_ASSET: AgentRole.DESIGN,
    IssueCode.BASE_ASSET_UNRESOLVED: AgentRole.DESIGN,
    # ---- rewards domain ----
    IssueCode.REWARD_TERM_CONFLICT: AgentRole.REWARDS,
    # PARAM_OUT_OF_RANGE is most often a reward weight; reward bounds are the
    # #1 safety footgun, so default it to the rewards expert.
    IssueCode.PARAM_OUT_OF_RANGE: AgentRole.REWARDS,
    # ---- motion / command domain ----
    IssueCode.MISSING_CONSUMPTION_MODE: AgentRole.MOTION,
    IssueCode.INVALID_CONSUMPTION_MODE: AgentRole.MOTION,
    IssueCode.INVALID_MOTION_CLIP: AgentRole.MOTION,
    IssueCode.INVALID_COMMAND_TEMPLATE: AgentRole.MOTION,
    IssueCode.INVALID_HEADING_COMMAND: AgentRole.MOTION,
    # ---- obs + robustness domain (obs / domain_rand / terminations / scene) ----
    IssueCode.CONTACT_SENSOR_COVERAGE_INSUFFICIENT: AgentRole.OBS_ROBUSTNESS,
    IssueCode.CONTACT_SENSOR_BODY_DATA_MISSING: AgentRole.OBS_ROBUSTNESS,
    IssueCode.TERRAIN_CURRICULUM_INVALID: AgentRole.OBS_ROBUSTNESS,
    IssueCode.CUSTOM_TERRAIN_INVALID: AgentRole.OBS_ROBUSTNESS,
    IssueCode.SIM_DT_CONFLICT: AgentRole.OBS_ROBUSTNESS,
    # ---- generic param errors → graph owner (refined later by node domain) ----
    IssueCode.UNKNOWN_PARAM_VALUE: AgentRole.ORCHESTRATOR,
    IssueCode.INVALID_JSON_PARAM: AgentRole.ORCHESTRATOR,
    IssueCode.INVALID_PARAM_TYPE: AgentRole.ORCHESTRATOR,
    IssueCode.UNMAPPED_ACTION_JOINTS: AgentRole.ORCHESTRATOR,
    IssueCode.NON_IR_JOINT_NAME: AgentRole.ORCHESTRATOR,
    IssueCode.GENERIC: AgentRole.ORCHESTRATOR,
}


def route(issue: ValidationIssue) -> AgentRole:
    """Return the granular responsible role for an issue (full roster)."""
    return _BY_CODE.get(issue.code, AgentRole.ORCHESTRATOR)


def collapse(role: AgentRole, active_roles: FrozenSet[AgentRole]) -> AgentRole:
    """Fold a granular producer role onto the roles active in this run.

    A role that is instantiated stays; any other producer role is repaired by
    the Orchestrator. ``DESIGN`` stays if active (it is in v1), else also folds
    to the Orchestrator.
    """
    if role in active_roles:
        return role
    return PRODUCER_FALLBACK


def to_routed(
    issue: ValidationIssue, active_roles: FrozenSet[AgentRole]
) -> RoutedIssue:
    """Wrap an issue with its collapsed responsible role."""
    return RoutedIssue(issue=issue, responsible=collapse(route(issue), active_roles))
