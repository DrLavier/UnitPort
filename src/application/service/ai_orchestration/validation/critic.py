# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""B4 tier-4 Critic — baseline-relative pathology review (blueprint §4.B4.4).

The compile gate (tier-2) catches schema / bounds / wiring; the C3 integrator
(tier-3) catches *locatable* cross-domain conflicts. The critic catches the
"valid JSON, broken training" pathologies (``training_config.md`` §9) that need
a same-family baseline to judge — reward imbalance, premature terminations,
composite-item waste. It is **read-only**: it emits issues and the scheduler
re-dispatches; it never mutates (blueprint §5).

v1 = deterministic rule checks for the highest-value footguns + one optional
scoped LLM review. To stay non-redundant with the integrator, the critic does
NOT re-check ``base_lin_vel`` (that is I2's job). Findings are WARNING-severity
by default (heuristic, advisory) — only the compile gate / I2 raise blocking
ERRORs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from unitport_sdk import log_warning

from ..contracts import (
    Completion,
    IssueCode,
    LlmClient,
    LlmTransportError,
    Severity,
    ValidationIssue,
)
from ..registry_projection import RegistryProjection

__all__ = ["run_critic", "rule_findings"]


# =============================================================================
# Deterministic rule findings (footguns #1, #9)
# =============================================================================

def rule_findings(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """Cheap, deterministic baseline-relative checks (no LLM)."""
    issues: List[ValidationIssue] = []
    issues.extend(_check_reward_balance(canvas))
    issues.extend(_check_premature_terminations(canvas))
    issues.extend(_check_uniform_rewards_multi_item(canvas))
    return issues


def _check_reward_balance(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """Footgun #1: tracking-vs-penalty imbalance.

    Aggregates weights across all rewards nodes' ``__global__`` page. Flags:
      * track_ang_vel_z stronger than track_lin_vel_xy (should be ~half), and
      * a positive tracking reward with no negative penalty term at all
        (nothing discourages degenerate motion).
    """
    weights: Dict[str, float] = {}
    for node in _nodes_of(canvas, "rewards"):
        terms = _json_param(node, "reward_terms") or {}
        for page in terms.values():
            if not isinstance(page, dict):
                continue
            for func, payload in page.items():
                w = _weight_of(payload)
                if w is not None:
                    # last writer wins for the summary view (conflicts are a
                    # separate compile-gate concern: REWARD_TERM_CONFLICT).
                    weights[func] = w
    if not weights:
        return []

    out: List[ValidationIssue] = []
    tl = weights.get("track_lin_vel_xy")
    ta = weights.get("track_ang_vel_z")
    if tl is not None and ta is not None and ta > tl:
        out.append(_issue(
            f"track_ang_vel_z ({ta}) is stronger than track_lin_vel_xy ({tl}); "
            f"angular tracking should be ~half of linear, or the robot is "
            f"rewarded for turning over moving (footgun #1).",
            field="track_ang_vel_z",
        ))
    has_positive_track = any(
        weights.get(k, 0.0) > 0 for k in ("track_lin_vel_xy", "track_ang_vel_z")
    )
    has_penalty = any(w < 0 for w in weights.values())
    if has_positive_track and not has_penalty:
        out.append(_issue(
            "tracking rewards are present but there is no negative penalty term; "
            "without penalties the policy can exploit degenerate motions "
            "(footgun #1). Add action-rate / orientation / height penalties.",
            field="reward_terms",
        ))
    return out


def _check_uniform_rewards_multi_item(
    canvas: Dict[str, Any],
) -> List[ValidationIssue]:
    """Composite-reward affordance: multiple enabled command items served by a
    single uniform global bundle (no per-item differentiation).

    Advisory only (WARNING). Fires when: (a) ≥2 command items are enabled on the
    canvas, AND (b) some rewards node carries global terms, AND (c) there is no
    per-item differentiation — neither ≥2 distinct rewards nodes feeding distinct
    items (the multi-node norm) nor any non-global (pd-group) page. A single
    command item, or an already-composite config, never fires (composite and
    single are both valid — this only nudges, never blocks).
    """
    enabled_items = _enabled_command_items(canvas)
    if len(enabled_items) < 2:
        return []

    has_global_terms = False
    has_non_global_page = False
    for node in _nodes_of(canvas, "rewards"):
        terms = _json_param(node, "reward_terms") or {}
        for page_id, page in terms.items():
            if not isinstance(page, dict) or not page:
                continue
            if str(page_id) == "__global__":
                has_global_terms = True
            else:
                has_non_global_page = True
    if not has_global_terms:
        # Nothing that looks like a uniform global bundle (missing-edge cases are
        # the integrator's I3 job, not this advisory).
        return []

    distinct_feeding_nodes = _distinct_reward_feed_nodes(canvas)
    differentiated = len(distinct_feeding_nodes) >= 2 or has_non_global_page
    if differentiated:
        return []

    return [_issue(
        f"{len(enabled_items)} command items are enabled "
        f"({', '.join(sorted(enabled_items))}) but rewards are a single uniform "
        f"global bundle with no per-item differentiation; consider composite "
        f"rewards — a separate bundle per item (e.g. walk/turn/stand) and "
        f"pd-group pages for joint-scoped terms (footgun: one bundle for all "
        f"items reinforces the easiest command).",
        field="reward_terms",
    )]


def _check_premature_terminations(canvas: Dict[str, Any]) -> List[ValidationIssue]:
    """Footgun #9: hard terminations that fire before the skill is learned pin
    the horizon into a 'survive-to-wall' degenerate policy."""
    out: List[ValidationIssue] = []
    for node in _nodes_of(canvas, "terminations"):
        conds = _json_param(node, "termination_conditions") or {}
        if "command_follow_timeout" in conds:
            out.append(_issue(
                "termination 'command_follow_timeout' is enabled; a hard kill "
                "for not-yet-learned command following can pin the learning "
                "horizon (footgun #9). Prefer continuous reward pressure.",
                node_id=str(node.get("id", "")), field="command_follow_timeout",
            ))
    return out


# =============================================================================
# Optional LLM review pass
# =============================================================================

_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_findings",
        "description": (
            "Report training-config pathologies you found in the canvas, judged "
            "relative to a same-family validated baseline. Empty list = clean."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "node_id": {"type": "string"},
                            "field": {"type": "string"},
                        },
                        "required": ["message"],
                    },
                }
            },
            "required": ["findings"],
        },
    },
}

_CRITIC_SYSTEM = (
    "You are a reinforcement-learning training-config critic for a legged-robot "
    "platform. Review the canvas for pathologies that compile cleanly but ruin "
    "training, judged against a same-family validated baseline: reward imbalance "
    "(a tracking reward several times stronger than its penalties; angular "
    "tracking that should be ~half of linear), premature hard terminations, and "
    "composite command items enabled alongside atomic ones (wastes samples, "
    "reinforces standing). Also flag reward-doctrine violations that create "
    "degenerate optima: a command-tracking reward (track_lin_vel_xy / "
    "track_ang_vel_z) with no track_fail_penalty counterpart — the bounded "
    "tracking reward saturates, so the policy gets stuck standing; a reward set "
    "with no posture / joint-regularization terms (flat_orientation, "
    "joint_deviation_l1, dof_pos_limits) — limb flailing / pose drift; "
    "feet_air_time with no gait — a single-leg twitch farms the stride reward; "
    "an alive/termination bonus dominating weak command pressure — survive-to-"
    "wall. These are heuristics judged for the task, not hard rules — a task may "
    "legitimately omit gait. Do NOT flag base_lin_vel placement (handled "
    "elsewhere). Call report_findings with a concise list; empty if clean."
)


def run_critic(
    canvas: Dict[str, Any],
    projection: RegistryProjection,
    llm: Optional[LlmClient] = None,
    *,
    baseline: Optional[Dict[str, Any]] = None,
) -> List[ValidationIssue]:
    """Return critic findings (WARNING severity).

    Always runs the deterministic rule checks; if ``llm`` is provided, also runs
    one scoped LLM review and merges its findings. An LLM transport failure is
    NOT fatal to the critic — the deterministic findings still stand and the
    failure is logged (the compile gate remains the authoritative blocking
    gate). This is a sanctioned degradation of an *advisory* tier, not of a
    required input (§8).
    """
    findings = rule_findings(canvas)
    if llm is None:
        return findings

    user_payload = {
        "backend": projection.backend,
        "families": sorted(projection.families),
        "canvas": canvas,
        "baseline_reference": baseline or "(no same-family baseline available)",
        "reward_bounds": projection.vocabulary_summary().get("rewards", {}),
    }
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        completion = llm.complete(messages, tools=[_REPORT_TOOL])
    except LlmTransportError as exc:
        log_warning(f"[ai.critic] LLM review skipped (transport error): {exc}")
        return findings
    findings.extend(_parse_llm_findings(completion))
    return findings


def _parse_llm_findings(completion: Completion) -> List[ValidationIssue]:
    out: List[ValidationIssue] = []
    for call in completion.tool_calls:
        if call.name != "report_findings":
            continue
        for f in call.arguments.get("findings", []) or []:
            if not isinstance(f, dict):
                continue
            msg = str(f.get("message", "")).strip()
            if not msg:
                continue
            out.append(_issue(
                msg, node_id=str(f.get("node_id", "")), field=str(f.get("field", ""))
            ))
    return out


# =============================================================================
# helpers
# =============================================================================

def _nodes_of(canvas: Dict[str, Any], schema_id: str) -> List[Dict[str, Any]]:
    return [
        n for n in canvas.get("nodes", [])
        if str(n.get("schema_id")) == schema_id and not n.get("disabled")
    ]


def _enabled_command_items(canvas: Dict[str, Any]) -> List[str]:
    """Command item ids marked ``enabled`` on the canvas' training_motion
    node(s). Anchors "multiple items" to what is actually enabled, not the
    registry vocabulary."""
    out: List[str] = []
    for node in _nodes_of(canvas, "training_motion"):
        items = _json_param(node, "training_items") or {}
        for item_id, cfg in items.items():
            if isinstance(cfg, dict) and cfg.get("enabled") and item_id not in out:
                out.append(str(item_id))
    return out


def _distinct_reward_feed_nodes(canvas: Dict[str, Any]) -> set:
    """Set of ``rewards`` node ids that feed at least one ``reward_in__<item>``
    port. ≥2 distinct nodes ⇒ per-item differentiation (the multi-node norm)."""
    reward_ids = {str(n.get("id")) for n in _nodes_of(canvas, "rewards")}
    feeding: set = set()
    for e in canvas.get("edges", []):
        if not str(e.get("target_port", "")).startswith("reward_in__"):
            continue
        src = str(e.get("source_node"))
        if src in reward_ids:
            feeding.add(src)
    return feeding


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


def _issue(message: str, *, node_id: str = "", field: str = "") -> ValidationIssue:
    return ValidationIssue(
        code=IssueCode.GENERIC, severity=Severity.WARNING, message=message,
        node_id=node_id, field=field,
    )
