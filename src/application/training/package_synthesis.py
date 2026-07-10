# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Training-package reward synthesis (Method A, Slice 1b).

A training package (see ``training_spec.TrainingPackage``) authors its reward set
INLINE on the ``training_motion`` node — the common case wires no ``rewards`` node.
But the two backends read rewards from different places:

  * SB3 / ``spec_compiler`` walks a :class:`WorkflowIR` and reads each ``rewards``
    node's paged ``reward_terms``;
  * IsaacLab / ``IsaacLabConfigCompiler`` parses the raw canvas dict and reads each
    ``rewards`` node's ``reward_terms`` by node id (and hard-fails if there is no
    ``rewards`` node).

So an inline-package canvas would train nothing on IsaacLab. This module closes that
gap by *synthesising a virtual ``rewards`` node per inline-reward package* into the
representation each backend consumes, with synthetic ``reward_pipe -> reward_in__<item>``
edges to the package's member items. Both backends' EXISTING per-item reward readers
then consume packages unchanged — downstream never learns about packages
(``training_package_schema_design.md`` §3).

``package_weight`` is folded into the term weights exactly ONCE, here, in
:func:`_fold_package_weight` — the single multiply site. Both backends read the folded
weights off the synthetic node; neither re-derives the product (design §3 / P6).

Presence-gated: a canvas with no inline-reward package produces NO synthetic node and is
byte-identical to today (P1). The implicit default package (``package_weight == 1.0``,
empty ``reward_terms``) never synthesises.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from application.compiler.term_payload import migrate_reward_terms_to_paged
from application.training.training_spec import (
    DEFAULT_PACKAGE_ID,
    MotionConfig,
    TrainingPackage,
    package_members,
    resolve_effective_packages,
)

__all__ = [
    "SYNTH_NODE_PREFIX",
    "SYNTH_EDGE_PREFIX",
    "SYNTH_METADATA_FLAG",
    "packages_from_param",
    "plan_package_reward_synthesis",
    "compose_reward_preview",
    "synthesize_into_ir",
    "synthesize_into_graph",
]

# Deterministic id prefixes (no uuid/random — synthesis must be reproducible for
# idempotency and byte-equality tests).
SYNTH_NODE_PREFIX = "__pkg_rewards__"
SYNTH_EDGE_PREFIX = "__pkg_edge__"
# Idempotency marker stamped on the IR / graph after synthesis so a second pass is
# a no-op (never double-adds the virtual nodes).
SYNTH_METADATA_FLAG = "__package_reward_synth_v1__"

_REWARD_IN_PREFIX = "reward_in__"


# ---------------------------------------------------------------------------
# Core (representation-neutral)
# ---------------------------------------------------------------------------


@dataclass
class _SynthPlan:
    """A planned virtual rewards node: its id, the effective (package_weight-folded)
    paged reward_terms, and the member item ids it feeds via reward_in__ edges."""

    node_id: str
    reward_terms: Dict[str, Any]
    member_item_ids: List[str] = field(default_factory=list)
    # skill_id (== trigger command-term NAME) gating this package's reward, or ""
    # (Slice 3). When set, the IsaacLab reward emit wraps every term in the
    # trigger-window gate keyed on this command name.
    gate_command_name: str = ""


def _fold_package_weight(paged_reward_terms: Dict[str, Any], package_weight: float) -> Dict[str, Any]:
    """THE single site where ``package_weight`` multiplies term weights.

    Operates on the paged ``{page_id: {func: payload}}`` shape. For a dict payload
    only the ``weight`` field is scaled — the ``value`` field (a function-internal
    target, e.g. a base-height set-point) is bound into the reward fn and must NOT be
    scaled. For a bare-scalar payload the scalar IS the weight and is scaled directly.

    No other code multiplies by ``package_weight`` (P6 single-multiply-site guard).
    """
    pw = float(package_weight)
    out: Dict[str, Any] = {}
    for page_id, terms in (paged_reward_terms or {}).items():
        new_terms: Dict[str, Any] = {}
        for func, payload in (terms or {}).items():
            if isinstance(payload, dict):
                p = dict(payload)
                if p.get("weight") is not None:
                    p["weight"] = float(p["weight"]) * pw
                new_terms[func] = p
            else:
                new_terms[func] = float(payload) * pw
        out[page_id] = new_terms
    return out


def plan_package_reward_synthesis(
    training_items: Dict[str, Any],
    packages: Dict[str, TrainingPackage],
    wired_reward_item_ids: Set[str],
) -> List[_SynthPlan]:
    """Resolve effective packages and return the virtual-node plan (fail-loud, §8).

    Reuses :func:`resolve_effective_packages` — the single source of truth for
    membership (it raises on a missing/disabled package, an item with no package_id
    under an authored layer, a zero-member enabled package, or the reserved id).

    Only packages with a NON-EMPTY inline ``reward_terms`` synthesise a node. A
    package with empty ``reward_terms`` but ``package_weight != 1.0`` raises — the
    lever would otherwise be silently ignored (its members' rewards come from wired
    nodes). A member fed by BOTH an inline package reward and a wired ``reward_in__``
    edge raises (P5 — single reward source per item).
    """
    motion = MotionConfig(
        training_items=dict(training_items or {}),
        packages=dict(packages or {}),
    )
    effective = resolve_effective_packages(motion)  # fail-loud membership

    plans: List[_SynthPlan] = []
    for pid, pkg in effective.items():
        raw_terms = getattr(pkg, "reward_terms", None) or {}
        pw = float(getattr(pkg, "package_weight", 1.0))
        gate_name = str(getattr(pkg, "gated_by", "")).strip()
        if not raw_terms:
            if pw != 1.0:
                raise ValueError(
                    f"package {pid!r} sets package_weight={pw} but has no inline "
                    f"reward_terms to scale — the weight lever would be silently "
                    f"ignored (its members' rewards come from wired rewards nodes). "
                    f"Author inline rewards on the package or set package_weight=1.0 "
                    f"(§8 fail-loud)."
                )
            if gate_name:
                raise ValueError(
                    f"package {pid!r} is gated by skill {gate_name!r} but has no "
                    f"inline reward_terms to gate — a skill package must author the "
                    f"task reward that the trigger gates (§8 fail-loud)."
                )
            continue  # implicit default / wired-reward package: no synthetic node
        members = package_members(motion, pid)
        if not members:
            raise ValueError(
                f"package {pid!r} has inline reward_terms but no enabled member "
                f"items to apply them to (§8 fail-loud)."
            )
        for item_id in members:
            if item_id in wired_reward_item_ids:
                raise ValueError(
                    f"training item {item_id!r} is fed by BOTH package {pid!r}'s "
                    f"inline reward_terms AND a wired rewards node "
                    f"(reward_in__{item_id}). An item's rewards must come from exactly "
                    f"one source — remove the wired edge or the inline package reward "
                    f"(§8 fail-loud)."
                )
        paged = migrate_reward_terms_to_paged(raw_terms)
        plans.append(_SynthPlan(
            node_id=f"{SYNTH_NODE_PREFIX}{pid}",
            reward_terms=_fold_package_weight(paged, pw),
            member_item_ids=list(members),
            gate_command_name=gate_name,
        ))
    return plans


# ---------------------------------------------------------------------------
# Param extraction helpers
# ---------------------------------------------------------------------------


def _as_json_obj(value: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a param value to a dict. Absent/None/blank -> default (presence gate);
    a present-but-malformed value raises (§8 — never silently swallow bad authored data)."""
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        parsed = json.loads(s)  # JSONDecodeError propagates (§8)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed
    raise ValueError(f"expected dict / JSON-string / None, got {type(value).__name__}")


def packages_from_param(raw: Any) -> Dict[str, TrainingPackage]:
    """Reconstruct ``{pid: TrainingPackage}`` from a training_motion ``packages`` param."""
    obj = _as_json_obj(raw, {})
    out: Dict[str, TrainingPackage] = {}
    for pid, v in obj.items():
        if isinstance(v, TrainingPackage):
            out[str(pid)] = v
            continue
        if not isinstance(v, dict):
            raise ValueError(
                f"package {pid!r} entry must be an object, got {type(v).__name__} (§8)."
            )
        out[str(pid)] = TrainingPackage(
            package_id=str(v.get("package_id", pid)),
            name=str(v.get("name", "")),
            enabled=bool(v.get("enabled", True)),
            package_weight=float(v.get("package_weight", 1.0)),
            reward_terms=dict(v.get("reward_terms") or {}),
            gated_by=str(v.get("gated_by", "")).strip(),
        )
    return out


def _item_id_from_reward_port(port: str) -> str:
    return port[len(_REWARD_IN_PREFIX):] if (port or "").startswith(_REWARD_IN_PREFIX) else ""


def compose_reward_preview(
    training_items: Dict[str, Any],
    packages: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Per-item composed inline-package reward, flat ``{item_id: {func: payload}}``.

    This is EXACTLY what export lowers (same planner + same single ``_fold_package_weight``
    site), so a UI preview built from it is byte-faithful to the trained reward (P7
    WYSIWYG). Items whose rewards come from a wired rewards node (not an inline package)
    do not appear here — the preview shows only the package-composed portion. Raises on
    invalid membership (the caller surfaces it); ``packages`` may be a dict-of-dict or a
    dict-of-:class:`TrainingPackage`.
    """
    plans = plan_package_reward_synthesis(
        dict(training_items or {}), packages_from_param(packages), set()
    )
    out: Dict[str, Dict[str, Any]] = {}
    for plan in plans:
        flat: Dict[str, Any] = {}
        for _page, terms in plan.reward_terms.items():
            flat.update(terms)
        for item_id in plan.member_item_ids:
            out[item_id] = dict(flat)
    return out


# ---------------------------------------------------------------------------
# Adapter — WorkflowIR (SB3 / spec_compiler side)
# ---------------------------------------------------------------------------


def _ir_param_value(node: Any, key: str) -> Any:
    p = node.params.get(key) if getattr(node, "params", None) else None
    return p.value if p is not None else None


def _wired_reward_items_ir(ir: Any) -> Set[str]:
    return {
        _item_id_from_reward_port(e.target_port)
        for e in ir.edges
        if _item_id_from_reward_port(e.target_port)
    }


def synthesize_into_ir(ir: Any) -> None:
    """Append virtual rewards :class:`IRNode` / :class:`IREdge` for inline-reward
    packages into ``ir`` in place. No-op (zero mutation) when no package synthesises.
    Idempotent via ``ir.metadata`` flag."""
    if ir.metadata.get(SYNTH_METADATA_FLAG):
        return
    tm = next((n for n in ir.nodes if n.schema_id == "training_motion"), None)
    if tm is None:
        return
    training_items = _as_json_obj(_ir_param_value(tm, "training_items"), {})
    packages = packages_from_param(_ir_param_value(tm, "packages"))
    plans = plan_package_reward_synthesis(training_items, packages, _wired_reward_items_ir(ir))
    if not plans:
        return

    from application.compiler.lowering import IRNode, IREdge, IRParam, EdgeType

    for plan in plans:
        node_params = {"reward_terms": IRParam(
            name="reward_terms", value=plan.reward_terms, param_type="json",
        )}
        if plan.gate_command_name:
            node_params["skill_gate"] = IRParam(
                name="skill_gate", value=plan.gate_command_name, param_type="string",
            )
        ir.nodes.append(IRNode(
            id=plan.node_id,
            schema_id="rewards",
            kind="process",
            params=node_params,
        ))
        for item_id in plan.member_item_ids:
            ir.edges.append(IREdge(
                id=f"{SYNTH_EDGE_PREFIX}{plan.node_id}__{item_id}",
                source_node=plan.node_id,
                source_port="reward_pipe",
                target_node=tm.id,
                target_port=f"{_REWARD_IN_PREFIX}{item_id}",
                edge_type=EdgeType.FLOW,
            ))
    ir.metadata[SYNTH_METADATA_FLAG] = True


# ---------------------------------------------------------------------------
# Adapter — canvas/workflow dict (IsaacLab / IsaacLabConfigCompiler side)
# ---------------------------------------------------------------------------


def _dict_param_value(params: Dict[str, Any], key: str) -> Any:
    v = (params or {}).get(key)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _node_type(n: Dict[str, Any]) -> str:
    return str(n.get("schema_id") or n.get("node_type") or "")


def _wired_reward_items_graph(graph: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for e in graph.get("edges") or []:
        port = str(e.get("target_port") or e.get("dst_slot") or "")
        item = _item_id_from_reward_port(port)
        if item:
            out.add(item)
    return out


def synthesize_into_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Return a graph with virtual rewards node/edge dicts added for inline-reward
    packages. Returns the ORIGINAL object unchanged (no copy) when no package
    synthesises — byte-identical to today (P1). Deep-copies before mutating so the
    caller's dict (and any saved canvas) is never touched. Idempotent via metadata."""
    if not isinstance(graph, dict):
        return graph
    meta = graph.get("metadata")
    if isinstance(meta, dict) and meta.get(SYNTH_METADATA_FLAG):
        return graph
    nodes = graph.get("nodes") or []
    tm = next((n for n in nodes if _node_type(n) == "training_motion"), None)
    if tm is None:
        return graph
    tm_params = tm.get("params") or tm.get("parameters") or {}
    training_items = _as_json_obj(_dict_param_value(tm_params, "training_items"), {})
    packages = packages_from_param(_dict_param_value(tm_params, "packages"))
    plans = plan_package_reward_synthesis(training_items, packages, _wired_reward_items_graph(graph))
    if not plans:
        return graph  # byte-identical: no copy, no mutation

    g = copy.deepcopy(graph)
    g.setdefault("nodes", [])
    g.setdefault("edges", [])
    tm_id = str(tm.get("id"))
    for plan in plans:
        node_params = {"reward_terms": {
            "name": "reward_terms", "value": plan.reward_terms, "param_type": "json",
        }}
        if plan.gate_command_name:
            node_params["skill_gate"] = {
                "name": "skill_gate", "value": plan.gate_command_name, "param_type": "string",
            }
        g["nodes"].append({
            "id": plan.node_id,
            "schema_id": "rewards",
            "kind": "process",
            "params": node_params,
            "ui": None,
            "opaque_code": None,
        })
        for item_id in plan.member_item_ids:
            g["edges"].append({
                "id": f"{SYNTH_EDGE_PREFIX}{plan.node_id}__{item_id}",
                "source_node": plan.node_id,
                "source_port": "reward_pipe",
                "target_node": tm_id,
                "target_port": f"{_REWARD_IN_PREFIX}{item_id}",
                "edge_type": "flow",
            })
    md = g.setdefault("metadata", {})
    if isinstance(md, dict):
        md[SYNTH_METADATA_FLAG] = True
    return g
