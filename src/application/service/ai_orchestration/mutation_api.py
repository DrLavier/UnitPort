# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""C1 Mutation API — the single write entry + tier-1 interface-alignment +
semantic ops (blueprint §2 / §3.C1 / §4.B4 tier-1).

This is the service-side layer ABOVE the GUI-resident
:class:`~application.ui.canvas.agent_bridge.CanvasAgentBridge`. The bridge owns
the five raw primitives over the live editor (and must run on the GUI thread);
this layer adds, on the worker thread:

  * **tier-1 interface-alignment** (blueprint §4.B4.1): ``connect`` checks port
    type compatibility + ``multi``/``optional`` arity against the C2 projection;
    ``set_param`` checks enum membership, numeric ``[min, max]``, and reward
    **hard weight bounds**. A failing check is rejected fail-loud with a
    :class:`ValidationIssue` and the blackboard is NOT touched (the sink is
    never called). This realises "constrained as it writes" (blueprint §3.C2).
  * **semantic ops** (blueprint §3.C1): the ONLY sanctioned cross-domain write.
    :meth:`assign_reward_bundle` expands into a fixed primitive sequence so no
    agent tracks cross-agent state.

The bridge already renders the human-readable C4 line for an applied primitive;
this layer captures the structured :class:`MutationEvent` in parallel and does
NOT re-render. Tier-1 rejections (which never reach the bridge) are surfaced to
the panel here via ``sink.log_markup`` so the user sees the fail-loud reason.

Dynamic ports (e.g. ``reward_in__<item>`` materialised on ``training_motion``
when an item is enabled) are not in the static manifest; tier-1 cannot resolve
them and therefore DEFERS their existence/type check to the sink (the live node
has the port) + the compile gate. This avoids false rejection of legitimate
dynamic-port connects.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from application.compiler.term_payload import PAGE_GLOBAL

from .contracts import (
    AgentRole,
    IssueCode,
    MutationEvent,
    MutationResult,
    MutationSink,
    Severity,
    ValidationIssue,
)
from .node_ref import candidates_hint, node_exists, nodes_of_schema
from .registry_projection import RegistryProjection

__all__ = ["MutationApi"]

# Dynamic-port prefixes whose existence/type is resolved at the live sink, not
# the static C2 projection (tier-1 defers).
_DYNAMIC_PORT_PREFIXES = ("reward_in__",)


class MutationApi:
    """The single write entry over a :class:`MutationSink`, with tier-1."""

    def __init__(
        self,
        sink: MutationSink,
        projection: RegistryProjection,
        *,
        agent: AgentRole = AgentRole.ORCHESTRATOR,
    ) -> None:
        self._sink = sink
        self._proj = projection
        self._agent = agent

    def set_agent(self, agent: AgentRole) -> None:
        """Attribute subsequent mutations to ``agent`` (for C4 events)."""
        self._agent = agent

    # =====================================================================
    # C1 primitives
    # =====================================================================

    def place_node(
        self,
        schema_id: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
        x: float = 0.0,
        y: float = 0.0,
    ) -> MutationResult:
        if schema_id not in self._proj.node_ids:
            return self._reject(
                "place_node",
                IssueCode.UNKNOWN_PARAM_VALUE,
                f"unknown node type {schema_id!r} for this backend/family",
                node_id=schema_id,
            )
        new_id = self._sink.place_node(
            schema_id, params=params, x=x, y=y, instance_id=instance_id
        )
        if not new_id:
            # The sink (bridge) already logged the fail-loud line. Make the
            # MODEL feedback actionable: name the EXISTING node it should
            # reference instead of re-placing (real cause = an instance_id that
            # already exists, page._spawn_node_raw). Surface — never auto-pick.
            snap = self._sink.snapshot()
            if instance_id and node_exists(snap, instance_id):
                hint = f"; node {instance_id!r} already exists — reference it directly"
            else:
                existing = nodes_of_schema(snap, schema_id)
                hint = (
                    f"; a node of type {schema_id!r} already exists: {existing} — "
                    f"reference it directly instead of re-placing"
                    if existing else ""
                )
            return MutationResult(
                ok=False,
                issues=[self._issue(
                    IssueCode.GENERIC,
                    f"place_node {schema_id!r} rejected by editor{hint}",
                )],
            )
        self._flash(new_id)
        return MutationResult(
            ok=True,
            node_id=new_id,
            event=self._event("place_node", target_node=new_id, value=schema_id),
        )

    def remove_node(self, instance_id: str) -> MutationResult:
        ok = self._sink.remove_node(instance_id)
        if not ok:
            return MutationResult(
                ok=False,
                issues=[self._issue(IssueCode.GENERIC, f"remove_node {instance_id!r} rejected")],
            )
        return MutationResult(
            ok=True,
            node_id=instance_id,
            event=self._event("remove_node", target_node=instance_id),
        )

    def connect(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str
    ) -> MutationResult:
        snap = self._sink.snapshot()
        issue = self._check_connect(snap, src_node, src_port, dst_node, dst_port)
        if issue is not None:
            return self._reject_issue("connect", issue)
        edge_id = self._sink.connect(src_node, src_port, dst_node, dst_port)
        if not edge_id:
            return MutationResult(
                ok=False,
                issues=[self._issue(
                    IssueCode.GENERIC,
                    f"connect {src_node}.{src_port} -> {dst_node}.{dst_port} rejected by editor",
                )],
            )
        self._flash(dst_node)
        return MutationResult(
            ok=True,
            edge_id=edge_id,
            event=self._event(
                "connect", target_node=dst_node, target_port=dst_port,
                value=f"{src_node}.{src_port}",
            ),
        )

    def disconnect(self, edge_id: str) -> MutationResult:
        ok = self._sink.disconnect(edge_id)
        if not ok:
            return MutationResult(
                ok=False,
                issues=[self._issue(IssueCode.GENERIC, f"disconnect {edge_id!r} rejected")],
            )
        return MutationResult(
            ok=True,
            edge_id=edge_id,
            event=self._event("disconnect", value=edge_id),
        )

    def set_param(self, instance_id: str, key: str, value: Any) -> MutationResult:
        snap = self._sink.snapshot()
        schema_id = _node_schema(snap, instance_id)
        if schema_id is None:
            return self._reject(
                "set_param", IssueCode.GENERIC,
                f"no node {instance_id!r} on canvas{candidates_hint(snap, instance_id)}",
                node_id=instance_id, field=key,
            )
        issue = self._check_set_param(schema_id, instance_id, key, value)
        if issue is not None:
            return self._reject_issue("set_param", issue)
        ok = self._sink.set_param(instance_id, key, value)
        if not ok:
            return MutationResult(
                ok=False,
                issues=[self._issue(
                    IssueCode.GENERIC, f"set_param {key!r} on {instance_id!r} rejected",
                    node_id=instance_id, field=key,
                )],
            )
        self._flash(instance_id)
        return MutationResult(
            ok=True,
            node_id=instance_id,
            event=self._event("set_param", target_node=instance_id, target_port=key, value=value),
        )

    # =====================================================================
    # Semantic op — assign a reward bundle to a command item (blueprint §3.C1)
    # =====================================================================

    def assign_reward_bundle(
        self, item_id: str, terms: Dict[str, Any], *, page: str = PAGE_GLOBAL
    ) -> MutationResult:
        """Expand to: enable item -> ensure a rewards node dedicated to this
        item -> write terms to ``page`` -> connect ``reward_in__<item>``.

        ``page`` selects which ``reward_terms`` page the terms land on:
        ``PAGE_GLOBAL`` (``__global__``, default — unchanged behaviour) or one
        of the family's pd-group pages (a joint partition). Per-item
        differentiation is realised by giving each item its OWN rewards node —
        reused via that item's ``reward_in__<item>`` edge when it already exists,
        else a fresh node is placed — so distinct bundles assigned to distinct
        items become distinct terms instead of one shared global bundle. Fixed
        order so the dynamic port exists before the connect. Returns an aggregate
        result (ok == all sub-ok)."""
        results: List[MutationResult] = []

        snap = self._sink.snapshot()
        motion_id = _first_node_of(snap, "training_motion")
        if motion_id is None:
            return self._reject(
                "assign_reward_bundle", IssueCode.MISSING_REQUIRED_NODE,
                "no training_motion node — place and wire it before assigning rewards",
                node_id="training_motion",
            )

        # Page must be the global page or a known pd-group page for the locked
        # family (fail-loud §8: an unknown page would silently create a dead
        # partition the compiler later rejects).
        page_key = str(page or PAGE_GLOBAL)
        valid_pages = (PAGE_GLOBAL, *self._proj.pd_group_pages)
        if page_key not in valid_pages:
            return self._reject(
                "assign_reward_bundle", IssueCode.UNKNOWN_PARAM_VALUE,
                f"reward page {page_key!r} is not valid for this family; "
                f"valid pages: {list(valid_pages)}",
                node_id="rewards", field="page",
            )

        # 1) enable the item in training_items (merge, keep other items).
        items = _json_param(snap, motion_id, "training_items") or {}
        entry = dict(items.get(item_id) or {})
        entry["enabled"] = True
        merged_items = dict(items)
        merged_items[item_id] = entry
        results.append(self.set_param(motion_id, "training_items", merged_items))

        # 2) ensure a rewards node dedicated to THIS item. Reuse the node already
        #    feeding reward_in__<item> (re-assignment / a second page for the same
        #    item); else place a fresh one so a second item does not overwrite the
        #    first item's bundle (per-item differentiation).
        snap = self._sink.snapshot()
        rewards_id = _rewards_node_for_item(snap, motion_id, item_id)
        if rewards_id is None:
            placed = self.place_node("rewards")
            results.append(placed)
            rewards_id = placed.node_id or None
        if rewards_id is None:
            return self._aggregate(results)

        # 3) merge the terms into the chosen page of that rewards node.
        existing = _json_param(self._sink.snapshot(), rewards_id, "reward_terms") or {}
        page_terms = dict(existing.get(page_key) or {})
        for name, weight in terms.items():
            num = _try_number(weight)
            if num is None:
                return self._reject(
                    "assign_reward_bundle", IssueCode.INVALID_PARAM_TYPE,
                    f"reward {name!r} weight must be a number, got {weight!r}",
                    field=str(name),
                )
            page_terms[name] = {"weight": num}
        new_terms = dict(existing)
        new_terms[page_key] = page_terms
        results.append(self.set_param(rewards_id, "reward_terms", new_terms))

        # 4) wire reward_pipe -> reward_in__<item> (dynamic port now exists),
        #    unless it is already wired (avoid a duplicate edge on re-assignment).
        dst_port = f"reward_in__{item_id}"
        if not _edge_exists(
            self._sink.snapshot(), rewards_id, "reward_pipe", motion_id, dst_port
        ):
            results.append(
                self.connect(rewards_id, "reward_pipe", motion_id, dst_port)
            )
        return self._aggregate(results)

    # =====================================================================
    # tier-1 checks (pure; over the C2 projection + snapshot)
    # =====================================================================

    def _check_connect(
        self, snap: Dict[str, Any], src_node: str, src_port: str,
        dst_node: str, dst_port: str,
    ) -> Optional[ValidationIssue]:
        src_schema = _node_schema(snap, src_node)
        dst_schema = _node_schema(snap, dst_node)
        if src_schema is None:
            return self._issue(
                IssueCode.GENERIC,
                f"connect: no source node {src_node!r}{candidates_hint(snap, src_node)}",
            )
        if dst_schema is None:
            return self._issue(
                IssueCode.GENERIC,
                f"connect: no target node {dst_node!r}{candidates_hint(snap, dst_node)}",
            )

        src_spec = self._proj.port(src_schema, src_port)
        dst_spec = self._proj.port(dst_schema, dst_port)

        # Static port resolved? Then enforce existence + type. Dynamic/unknown
        # ports defer to the live sink + compile gate (see module docstring).
        if src_spec is None and not _is_dynamic_port(src_port):
            return self._issue(
                IssueCode.MISSING_REQUIRED_PORT,
                f"connect: source port {src_port!r} not on {src_schema!r}",
                node_id=src_node, field=src_port,
            )
        if dst_spec is None and not _is_dynamic_port(dst_port):
            return self._issue(
                IssueCode.MISSING_REQUIRED_PORT,
                f"connect: target port {dst_port!r} not on {dst_schema!r}",
                node_id=dst_node, field=dst_port,
            )

        # type compatibility (only when both static specs are known).
        if src_spec is not None and dst_spec is not None:
            if not _types_compatible(src_spec.type, dst_spec.type):
                return self._issue(
                    IssueCode.GENERIC,
                    f"connect: port type mismatch {src_port}:{src_spec.type} -> "
                    f"{dst_port}:{dst_spec.type}",
                    node_id=dst_node, field=dst_port,
                )

        # arity: a single-connection input port may not already be wired.
        if dst_spec is not None and not dst_spec.multi:
            if _count_incoming(snap, dst_node, dst_port) >= 1:
                return self._issue(
                    IssueCode.GENERIC,
                    f"connect: input port {dst_port!r} on {dst_node!r} already "
                    f"wired (single-connection)",
                    node_id=dst_node, field=dst_port,
                )
        return None

    def _check_set_param(
        self, schema_id: str, instance_id: str, key: str, value: Any
    ) -> Optional[ValidationIssue]:
        # Reward weights are the #1 safety footgun — validate hard bounds.
        if schema_id == "rewards" and key == "reward_terms":
            return self._check_reward_terms(instance_id, value)

        schema = self._proj.param_schema(schema_id).get(key)
        if schema is None:
            # Unknown / backend-gated param — defer to compile gate (it knows
            # the full param surface incl. conditional siblings). Not a tier-1
            # hard reject.
            return None

        # A json param whose manifest default is an OBJECT (e.g. obs_terms="{}")
        # must receive an object, never a list/scalar: the compiler does
        # ``dict(value)`` on it, and ``dict(["base_lin_vel", ...])`` raises deep
        # in compilation. Reject at write-time with an actionable message so the
        # agent supplies a mapping instead (§8 — caught at the seam, not a crash).
        if schema.param_type == "json" and isinstance(_coerce_json(schema.default), dict):
            if not isinstance(_coerce_json(value), dict):
                return self._issue(
                    IssueCode.INVALID_JSON_PARAM,
                    f"set_param: {key!r} must be a JSON OBJECT (a mapping "
                    f"{{name: payload}}), not a {type(_coerce_json(value)).__name__} "
                    f"— e.g. obs_terms = {{\"base_ang_vel\": {{\"scale\": 1.0}}}}",
                    node_id=instance_id, field=key,
                )

        # A training item's `clip` may carry a segment ref (`<clip>#seg=lo-hi`).
        # Validate the SYNTAX at the write seam so a malformed suffix gets
        # actionable feedback here (routed to the MOTION agent) instead of only
        # surfacing as INVALID_MOTION_CLIP deep in the compile gate.
        if schema_id == "training_motion" and key == "training_items":
            seg_issue = self._check_training_item_clips(instance_id, value)
            if seg_issue is not None:
                return seg_issue

        if schema.choices is not None and value not in schema.choices:
            return self._issue(
                IssueCode.UNKNOWN_PARAM_VALUE,
                f"set_param: {value!r} not in allowed values {schema.choices} for {key!r}",
                node_id=instance_id, field=key,
            )

        if schema.minimum is not None or schema.maximum is not None:
            num = _try_number(value)
            if num is None:
                # Non-numeric value for a bounded field — let the compile gate
                # decide (some bounded params accept list/expr forms).
                return None
            if schema.minimum is not None and num < schema.minimum:
                return self._issue(
                    IssueCode.PARAM_OUT_OF_RANGE,
                    f"set_param: {key}={num} below min {schema.minimum}",
                    node_id=instance_id, field=key,
                )
            if schema.maximum is not None and num > schema.maximum:
                return self._issue(
                    IssueCode.PARAM_OUT_OF_RANGE,
                    f"set_param: {key}={num} above max {schema.maximum}",
                    node_id=instance_id, field=key,
                )
        return None

    def _check_training_item_clips(
        self, instance_id: str, value: Any
    ) -> Optional[ValidationIssue]:
        """Validate the ``#seg=`` SYNTAX of each training item's ``clip``.

        Only the segment-suffix grammar is checked here (fast, file-free). The
        range-vs-clip-length and file-existence checks stay at the compile gate,
        which loads the clip. A whole-clip ref (no ``#seg=``) is untouched.
        """
        items = _coerce_json(value)
        if not isinstance(items, dict):
            return None  # object-shape already handled by the caller
        from application.training.motion.segment_ref import (
            has_segment_range,
            parse_segment_ref,
        )
        for item_id, payload in items.items():
            if not isinstance(payload, dict):
                continue
            clip = payload.get("clip")
            if not isinstance(clip, str) or not has_segment_range(clip):
                continue
            try:
                parse_segment_ref(clip)
            except ValueError as exc:
                return self._issue(
                    IssueCode.INVALID_MOTION_CLIP,
                    f"set_param: training item {item_id!r} clip {clip!r} has a "
                    f"malformed segment ref: {exc}. Use `<clip_ref>#seg=lo-hi` "
                    f"(inclusive, 0<=lo<=hi), or reference an exact ref listed "
                    f"in the motion_segments vocabulary.",
                    node_id=instance_id, field="training_items",
                )
        return None

    def _check_reward_terms(
        self, instance_id: str, value: Any
    ) -> Optional[ValidationIssue]:
        parsed = _coerce_json(value)
        if not isinstance(parsed, dict):
            return self._issue(
                IssueCode.INVALID_JSON_PARAM,
                "reward_terms must be a paged dict {page: {func: payload}}",
                node_id=instance_id, field="reward_terms",
            )
        for page_id, page in parsed.items():
            if not isinstance(page, dict):
                return self._issue(
                    IssueCode.INVALID_JSON_PARAM,
                    f"reward_terms page {page_id!r} must be a dict",
                    node_id=instance_id, field="reward_terms",
                )
            for func, payload in page.items():
                weight = _weight_of(payload)
                bounds = self._proj.rewards.get(func)
                if bounds is None:
                    return self._issue(
                        IssueCode.UNKNOWN_PARAM_VALUE,
                        f"reward {func!r} is not a valid reward for this "
                        f"backend/family",
                        node_id=instance_id, field=func,
                    )
                if weight is None:
                    continue  # payload carries no weight (e.g. value-only) — skip
                if weight < bounds.minimum or weight > bounds.maximum:
                    return self._issue(
                        IssueCode.PARAM_OUT_OF_RANGE,
                        f"reward {func!r} weight {weight} outside "
                        f"[{bounds.minimum}, {bounds.maximum}] — would distort or "
                        f"freeze training",
                        node_id=instance_id, field=func,
                    )
        return None

    # =====================================================================
    # helpers
    # =====================================================================

    def _flash(self, node_id: str) -> None:
        """Highlight the just-mutated node (on-only; the scheduler's block exit
        + run-end ``clear_highlights`` turn it back off). Best-effort UI."""
        if not node_id:
            return
        fn = getattr(self._sink, "highlight_node", None)
        if callable(fn):
            try:
                fn(node_id, True)
            except Exception:  # cosmetic feedback must never break a mutation
                pass

    def _event(
        self, op: str, *, target_node: str = "", target_port: str = "",
        value: Any = None,
    ) -> MutationEvent:
        return MutationEvent(
            op=op, agent=self._agent, timestamp=time.time(),
            target_node=target_node, target_port=target_port, value=value,
        )

    def _issue(
        self, code: IssueCode, message: str, *, node_id: str = "", field: str = "",
    ) -> ValidationIssue:
        return ValidationIssue(
            code=code, severity=Severity.ERROR, message=message,
            node_id=node_id, field=field,
        )

    def _reject(
        self, op: str, code: IssueCode, message: str, *, node_id: str = "",
        field: str = "",
    ) -> MutationResult:
        return self._reject_issue(op, self._issue(code, message, node_id=node_id, field=field))

    def _reject_issue(self, op: str, issue: ValidationIssue) -> MutationResult:
        # Surface the fail-loud reason to the panel (the bridge never saw this
        # op, so it could not log it).
        self._sink.log_markup(f"[[error:{op} rejected — {issue.message}]]")
        return MutationResult(ok=False, issues=[issue])

    def _aggregate(self, results: List[MutationResult]) -> MutationResult:
        issues: List[ValidationIssue] = []
        ok = True
        last_event = None
        for r in results:
            ok = ok and r.ok
            issues.extend(r.issues)
            if r.event is not None:
                last_event = r.event
        return MutationResult(ok=ok, issues=issues, event=last_event)


# =============================================================================
# snapshot / value helpers (module-level, pure)
# =============================================================================

def _node_schema(snap: Dict[str, Any], instance_id: str) -> Optional[str]:
    for n in snap.get("nodes", []):
        if str(n.get("id")) == str(instance_id):
            return str(n.get("schema_id") or "")
    return None


def _first_node_of(snap: Dict[str, Any], schema_id: str) -> Optional[str]:
    for n in snap.get("nodes", []):
        if str(n.get("schema_id")) == schema_id:
            return str(n.get("id"))
    return None


def _rewards_node_for_item(
    snap: Dict[str, Any], motion_id: str, item_id: str
) -> Optional[str]:
    """Return the id of a ``rewards`` node already feeding ``motion_id``'s
    ``reward_in__<item>`` port, or None.

    Lets a re-assignment (or a second page) for the same item reuse that item's
    own rewards node, while a different item resolves to None and gets its own
    node — the seam that makes per-item bundles distinct rather than shared.
    """
    port = f"reward_in__{item_id}"
    reward_ids = {
        str(n.get("id"))
        for n in snap.get("nodes", [])
        if str(n.get("schema_id")) == "rewards"
    }
    for e in snap.get("edges", []):
        if (
            str(e.get("target_node")) == str(motion_id)
            and str(e.get("target_port")) == port
            and str(e.get("source_node")) in reward_ids
        ):
            return str(e.get("source_node"))
    return None


def _edge_exists(
    snap: Dict[str, Any], src_node: str, src_port: str, dst_node: str, dst_port: str
) -> bool:
    for e in snap.get("edges", []):
        if (
            str(e.get("source_node")) == str(src_node)
            and str(e.get("source_port")) == str(src_port)
            and str(e.get("target_node")) == str(dst_node)
            and str(e.get("target_port")) == str(dst_port)
        ):
            return True
    return False


def _count_incoming(snap: Dict[str, Any], node: str, port: str) -> int:
    return sum(
        1
        for e in snap.get("edges", [])
        if str(e.get("target_node")) == str(node)
        and str(e.get("target_port")) == str(port)
    )


def _json_param(
    snap: Dict[str, Any], instance_id: str, key: str
) -> Optional[Dict[str, Any]]:
    for n in snap.get("nodes", []):
        if str(n.get("id")) != str(instance_id):
            continue
        params = n.get("params") or {}
        cell = params.get(key)
        if cell is None:
            return None
        raw = cell.get("value") if isinstance(cell, dict) else cell
        parsed = _coerce_json(raw)
        return parsed if isinstance(parsed, dict) else None
    return None


def _is_dynamic_port(port_name: str) -> bool:
    return any(port_name.startswith(p) for p in _DYNAMIC_PORT_PREFIXES)


def _types_compatible(src_type: str, dst_type: str) -> bool:
    """Port type compatibility. ``any`` matches anything; otherwise exact."""
    if src_type == dst_type:
        return True
    return "any" in (src_type, dst_type)


def _coerce_json(value: Any) -> Any:
    """Accept native dict/list or a JSON string (the loader tolerates both)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _weight_of(payload: Any) -> Optional[float]:
    """Extract a weight from a reward payload (bare float | {"weight": x})."""
    num = _try_number(payload)
    if num is not None:
        return num
    if isinstance(payload, dict) and "weight" in payload:
        return _try_number(payload.get("weight"))
    return None


def _try_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None
    return None
