# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.compiler.reward_conflicts — cross-rewards term conflict detection.

A canvas may carry multiple ``rewards`` nodes that all fan into the same
``training_motion`` item (via ``reward_pipe → reward_in__<item_id>``). The
``reward_terms`` dict on each rewards node is unioned into one per-item term
map at compile-time; when two rewards define the **same term key with
different values** on the same item, the result of the union is
non-deterministic (last loop iteration wins). This module surfaces that
collision as a typed conflict so the validator can reject the compile and the
UI can paint the affected term rows red.

Same key + identical value across rewards is **not** a conflict — the rewards
simply agree on that term and the union collapses cleanly.

Contract:
    ``compute_reward_term_conflicts(ir) → {rewards_node_id: {term_key: TermConflict}}``

Rewards nodes that contribute zero conflicting terms do not appear in the
returned dict. Each ``TermConflict`` carries the item_id where the clash
surfaces and a ``{rewards_node_id: value}`` map of contributors so the UI can
show which other rewards a given key collides with.

Pure function: walks ``ir.edges`` and reads ``rewards`` node params; does not
mutate the IR.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from application.compiler.lowering import WorkflowIR, IRNode


REWARD_PORT_PREFIX = "reward_in__"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TermConflict:
    """One reward-term collision across multiple rewards on the same item.

    ``values`` is ``{rewards_node_id: normalized_value}`` for every rewards
    node that contributes this key on this item; len(values) ≥ 2 and the
    set of distinct values has size ≥ 2 (otherwise it would not be a
    conflict).
    """

    item_id: str
    key: str
    values: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward_term_conflicts(
    ir: "WorkflowIR",
) -> Dict[str, Dict[str, TermConflict]]:
    """Return ``{rewards_node_id: {term_key: TermConflict}}``.

    A conflict exists iff ≥2 rewards nodes drive the same training_motion
    item via ``reward_pipe → reward_in__<item_id>`` AND define the same
    term key with at least two distinct normalized values.
    """
    rewards_nodes_by_id: Dict[str, "IRNode"] = {
        n.id: n for n in ir.nodes if n.schema_id == "rewards"
    }
    training_motion_ids = {
        n.id for n in ir.nodes if n.schema_id == "training_motion"
    }
    if len(rewards_nodes_by_id) < 2 or not training_motion_ids:
        return {}

    feeders_by_item: Dict[str, List[str]] = {}
    for e in ir.edges:
        if e.source_port != "reward_pipe":
            continue
        if e.source_node not in rewards_nodes_by_id:
            continue
        if e.target_node not in training_motion_ids:
            continue
        tport = e.target_port or ""
        if not tport.startswith(REWARD_PORT_PREFIX):
            continue
        item_id = tport[len(REWARD_PORT_PREFIX):]
        if not item_id:
            continue
        bucket = feeders_by_item.setdefault(item_id, [])
        if e.source_node not in bucket:
            bucket.append(e.source_node)

    out: Dict[str, Dict[str, TermConflict]] = {}
    for item_id, feeders in feeders_by_item.items():
        if len(feeders) < 2:
            continue

        terms_by_feeder: Dict[str, Dict[str, Any]] = {
            rid: _read_reward_terms(rewards_nodes_by_id[rid]) for rid in feeders
        }

        all_keys: set[str] = set()
        for terms in terms_by_feeder.values():
            all_keys.update(terms.keys())

        for key in all_keys:
            contributors: Dict[str, Any] = {}
            normalized: Dict[str, Tuple[str, Any]] = {}
            for rid, terms in terms_by_feeder.items():
                if key not in terms:
                    continue
                raw = terms[key]
                contributors[rid] = raw
                normalized[rid] = _normalize_term_value(raw)
            if len(contributors) < 2:
                continue
            distinct = {sig for sig in normalized.values()}
            if len(distinct) < 2:
                continue
            for rid in contributors:
                out.setdefault(rid, {})[key] = TermConflict(
                    item_id=item_id,
                    key=key,
                    values=dict(contributors),
                )

    return out


def conflicting_keys_for(
    ir: "WorkflowIR",
    rewards_node_id: str,
) -> set:
    """Convenience: just the set of conflicting term keys for one rewards
    node. Empty set if no conflicts. UI uses this to tint editor rows."""
    table = compute_reward_term_conflicts(ir)
    return set(table.get(rewards_node_id, {}).keys())


def normalize_term_value(v: Any) -> Tuple[str, Any]:
    """Public alias of the term-value normalization used internally.

    Exposed for UI-side conflict computation that walks the canvas
    ``ConnectionItem`` graph directly (avoiding a full IR conversion on
    every paint). Same signature contract as the private form.
    """
    return _normalize_term_value(v)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_reward_terms(node: "IRNode") -> Dict[str, Any]:
    """Pull the ``reward_terms`` param off an IR node.

    Raises ``ValueError`` on malformed JSON or wrong shape so the validator
    can translate the failure into a typed ``INVALID_JSON_PARAM``
    ``ValidationIssue`` — silently treating a broken value as ``{}`` would
    let a real user error pass the conflict check undetected.

    Missing param / ``None`` / empty string are legitimate "no terms"
    states and return ``{}`` without raising (a rewards node may be wired
    but not yet configured; that's a different rule the topology check
    handles).
    """
    # 缺口③ — reward_terms is paged ({page: {func: payload}}); flatten across
    # pages so conflict detection compares actual term values (the global page
    # wins on key collision). Legacy flat dicts pass through unchanged.
    def _flat(d: Dict[str, Any]) -> Dict[str, Any]:
        from application.compiler.term_payload import (
            is_paged_reward_terms, iter_reward_pages,
        )
        if not is_paged_reward_terms(d):
            return dict(d)
        out: Dict[str, Any] = {}
        for _pid, page_terms in iter_reward_pages(d):
            for func, payload in page_terms.items():
                out.setdefault(func, payload)
        return out

    p = node.params.get("reward_terms")
    if p is None:
        return {}
    raw = p.value
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return _flat(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reward_terms on node {node.id!r} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"reward_terms on node {node.id!r} parses to "
                f"{type(parsed).__name__}, expected dict"
            )
        return _flat(parsed)
    raise ValueError(
        f"reward_terms on node {node.id!r} has unsupported type "
        f"{type(raw).__name__}; expected dict or JSON string"
    )


def _normalize_term_value(v: Any) -> Tuple[str, Any]:
    """Reduce ``reward_terms`` value shapes to a comparable signature.

    Schema is either ``{key: weight_float}`` or
    ``{key: {"weight": weight_float, ...other...}}``. Two terms agree iff
    their weight values match (other fields like ``std`` per-term overrides
    are advisory and not part of the merge-collision check). Numeric values
    compare uniformly via ``float()`` so ``1`` and ``1.0`` collapse.
    """
    if isinstance(v, dict):
        w = v.get("weight")
        return ("dict", _scalar_sig(w))
    return ("scalar", _scalar_sig(v))


def _scalar_sig(v: Any) -> Any:
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    return ("raw", repr(v))
