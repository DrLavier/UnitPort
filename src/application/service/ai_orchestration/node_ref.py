# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Node-identity presentation + corrective-affordance rendering (actionability).

The orchestration model must reference nodes by their real ``node_id`` (e.g.
``actor_setting_1``), never by the bare ``schema_id`` (``actor_setting``) — the
two never coincide (``page._spawn_node_raw`` assigns ``f"{schema}_{n}"``). When
feedback is not actionable — a snapshot that puts ``id`` next to a look-alike
``schema_id``, or a "no node X" reject with no candidate ids — the model retries
the same wrong id. These pure helpers make identity round-trip unambiguously and
make every rejection / blocking issue carry the ids needed to recover:

  * :func:`present_snapshot` — the canvas keyed BY ``node_id`` (schema under
    ``type``), for the Orchestrator/repair prompt (sub-step A).
  * :func:`candidates_hint` — the real node_ids a "no node <ref>" reject likely
    meant (sub-step B).
  * :func:`render_actionable_issue` / :func:`render_repair_message` — an unwired
    required input rendered with the port's pipe TYPE + candidate SOURCE node_ids
    to connect from (sub-step C).

All functions are Qt-free and read only the snapshot dict + the (duck-typed) C2
projection (``.port`` / ``.ports``) — no fork of MutationResult/ValidationIssue.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .contracts import IssueCode, ValidationIssue

__all__ = [
    "all_node_ids",
    "nodes_of_schema",
    "node_exists",
    "candidates_hint",
    "present_snapshot",
    "source_candidates",
    "render_actionable_issue",
    "render_repair_message",
]

#: Cap id lists surfaced into a prompt/message so a big canvas can't blow it up.
_MAX_IDS = 24


def all_node_ids(snap: Dict[str, Any]) -> List[str]:
    return [
        str(n.get("id"))
        for n in (snap.get("nodes") or [])
        if n.get("id") is not None
    ]


def nodes_of_schema(snap: Dict[str, Any], schema_id: str) -> List[str]:
    """node_ids of every node whose schema_id == ``schema_id`` (the common
    "model used the schema name as an id" case)."""
    return [
        str(n.get("id"))
        for n in (snap.get("nodes") or [])
        if str(n.get("schema_id")) == str(schema_id) and n.get("id") is not None
    ]


def node_exists(snap: Dict[str, Any], node_id: str) -> bool:
    return any(str(n.get("id")) == str(node_id) for n in (snap.get("nodes") or []))


def _node_schema(snap: Dict[str, Any], node_id: str) -> str:
    for n in snap.get("nodes") or []:
        if str(n.get("id")) == str(node_id):
            return str(n.get("schema_id") or "")
    return ""


def candidates_hint(snap: Dict[str, Any], ref: str) -> str:
    """Corrective suffix for a "no node <ref>" reject.

    If ``ref`` matches a schema present on the canvas, list that schema's real
    node_ids (the model almost always meant one of these); else point at the
    known node_ids so it can pick a valid target. Never auto-picks — it only
    surfaces candidates (fail-loud preserved)."""
    same = nodes_of_schema(snap, ref)
    if same:
        return f"; on canvas of type {str(ref)!r}: {same[:_MAX_IDS]} — reference one of these node_ids"
    known = all_node_ids(snap)
    if known:
        return f"; reference an existing node_id (never a schema name), e.g. {known[:_MAX_IDS]}"
    return "; the canvas has no nodes yet — place one first"


def present_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    """Canvas view for the model, PRIMARY-KEYED by ``node_id``.

    The exact string to pass to ``set_param`` / ``connect`` is the dict key; the
    node's schema lives under ``type`` (deliberately not beside a look-alike
    ``id`` field, which is what invited schema-as-id confusion)."""
    nodes: Dict[str, Any] = {}
    for n in snap.get("nodes") or []:
        nid = str(n.get("id"))
        nodes[nid] = {
            "type": str(n.get("schema_id") or ""),
            "params": n.get("params") or {},
        }
    return {
        "_note": (
            "reference nodes by these node_id KEYS (e.g. 'actor_setting_1'); "
            "'type' is the node's schema, NOT a reference id"
        ),
        "backend": snap.get("backend"),
        "robot_id": snap.get("robot_id"),
        "nodes": nodes,
        "edges": snap.get("edges", []),
    }


def source_candidates(snap: Dict[str, Any], projection: Any, pipe_type: str) -> List[str]:
    """node_ids on canvas that expose an OUTPUT port able to feed ``pipe_type``.

    Compatibility mirrors tier-1: exact type match or ``any`` on either side.
    These are the nodes a "connect from a node providing <pipe_type>" fix can
    draw from (e.g. the robot node for ``robot_pipe``)."""
    out: List[str] = []
    for n in snap.get("nodes") or []:
        schema = str(n.get("schema_id") or "")
        for p in projection.ports(schema).get("outputs", []):
            if p.type == pipe_type or p.type == "any" or pipe_type == "any":
                nid = str(n.get("id"))
                if nid not in out:
                    out.append(nid)
                break
    return out


def render_actionable_issue(
    issue: ValidationIssue, snap: Dict[str, Any], projection: Any
) -> str:
    """Render one blocking issue as an ACTIONABLE instruction for the model.

    An unwired required input becomes "node `<id>`: required input '<port>'
    unwired; connect a <pipe_type> output into it; candidate source node_ids:
    [...]" — naming the target node, the port, the pipe type, and the real
    source ids, and dropping the training-owned message's misleading ``schema=``
    wording. Other issues fall back to ``str(issue)`` (which already carries the
    ``node=/field=`` prefix, 0.5) — no fork of the shared type."""
    if issue.code == IssueCode.MISSING_REQUIRED_PORT and issue.node_id and issue.field:
        schema = _node_schema(snap, issue.node_id)
        port = projection.port(schema, issue.field) if schema else None
        pipe_type = port.type if port is not None else "the required type"
        sources = source_candidates(snap, projection, pipe_type)
        candidates = (
            sources[:_MAX_IDS]
            if sources
            else "none on canvas — place a node providing it first"
        )
        return (
            f"node {issue.node_id!r}: required input {issue.field!r} is unwired; "
            f"connect a node output of type {pipe_type!r} into it; "
            f"candidate source node_ids: {candidates}"
        )
    return str(issue)


def render_repair_message(
    blocking: List[ValidationIssue], snap: Dict[str, Any], projection: Any
) -> str:
    """The repair user-message: the id-keyed snapshot + one actionable line per
    blocking issue. Replaces the id-blind ``- {issue}`` list so the repair
    Orchestrator has both the real node_ids and the corrective wiring."""
    lines = "\n".join(
        f"- {render_actionable_issue(i, snap, projection)}" for i in blocking
    )
    return (
        "The canvas has blocking issues. Fix each by calling the tools; do not "
        "repeat a rejected call verbatim. Reference nodes ONLY by a node_id key "
        "in the snapshot below (or one returned by place_node), never by a "
        "schema/type name.\n\nCurrent canvas (nodes keyed by node_id):\n"
        + json.dumps(present_snapshot(snap), ensure_ascii=False)
        + "\n\nIssues:\n"
        + lines
    )
