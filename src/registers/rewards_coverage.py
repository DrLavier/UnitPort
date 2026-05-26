# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.rewards_coverage — Reward × MotionPhase coverage evaluation.

Backed by ``data/reward_coverage_rules.json`` (factory defaults) plus an
optional user overlay at ``<USER_CONFIG_DIR>/registers/reward_coverage_rules_custom.json``.
Consumed by the canvas Rewards node's Coverage Badge and the compile-time
``CanvasPage.compile_spec`` block dialog.

Concept
-------
For each *motion phase* declared in :mod:`registers.motion_phases`, the
evaluator counts the reward terms that ``applies_to`` that phase (or are
unconditional, which counts against every phase) and bins their weights
into ``positive`` / ``negative`` / ``neutral`` polarity. Combining the
phase's ``polarity_required`` hint with the per-rule severity thresholds
in ``reward_coverage_rules.json``, the evaluator returns a single overall
:class:`CoverageSeverity` plus a per-phase :class:`PhaseCoverage` report.

The default rules ship with one critical rule (a phase declaring
``polarity_required = "negative"`` with zero negative-weight terms — the
Go2 idle limb-flailing pattern) and one warning rule (a phase declaring
``polarity_required = "both"`` missing one polarity).

Calling code
------------
* Canvas paints the Badge color from :func:`evaluate` ``.severity``.
* Inspector dialog renders the per-phase table from ``.phases``.
* Compile-path reads ``.blocking`` to decide whether to halt training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from unitport_sdk import Paths, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_DEFAULTS_PATH = _DATA_DIR / "reward_coverage_rules.json"


def _user_overlay_path():
    return Paths.USER_CONFIG_DIR / "registers" / "reward_coverage_rules_custom.json"


# =============================================================================
# Severity enum-like
# =============================================================================


SEVERITY_OK = "ok"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

_SEVERITY_ORDER = {SEVERITY_OK: 0, SEVERITY_WARNING: 1, SEVERITY_CRITICAL: 2}


def _severity_max(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class PhaseCoverage:
    """Per-phase coverage breakdown produced by :func:`evaluate`.

    The counts split each polarity into two buckets:

    * ``*_terms``                — phase-specific terms (declared ``applies_to``
      contains this phase id).
    * ``unconditional_*_terms`` — terms with empty ``applies_to`` (apply
      to every phase at runtime).

    The Inspector renders both columns; the Coverage Badge uses
    ``rule.when.phase_specific_*_count`` matchers to distinguish "phase is
    only governed by unconditional reward — likely under-isolated" from
    "phase has its own dedicated reward shaping".
    """

    phase_id: str
    display_name: str
    polarity_required: str
    positive_terms: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)
    neutral_terms: List[str] = field(default_factory=list)
    unconditional_positive_terms: List[str] = field(default_factory=list)
    unconditional_negative_terms: List[str] = field(default_factory=list)
    unconditional_neutral_terms: List[str] = field(default_factory=list)
    severity: str = SEVERITY_OK
    triggered_rule_ids: List[str] = field(default_factory=list)
    message: str = ""
    suggestions: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Aggregated views (used by the Inspector for display + by rules)
    # ------------------------------------------------------------------

    @property
    def all_positive_terms(self) -> List[str]:
        return list(self.positive_terms) + list(self.unconditional_positive_terms)

    @property
    def all_negative_terms(self) -> List[str]:
        return list(self.negative_terms) + list(self.unconditional_negative_terms)

    @property
    def all_neutral_terms(self) -> List[str]:
        return list(self.neutral_terms) + list(self.unconditional_neutral_terms)


@dataclass
class CoverageReport:
    """Full coverage report — feed both the Badge and the Inspector."""

    severity: str = SEVERITY_OK
    blocking: bool = False
    phases: List[PhaseCoverage] = field(default_factory=list)
    has_terms: bool = False

    def critical_phases(self) -> List[PhaseCoverage]:
        return [p for p in self.phases if p.severity == SEVERITY_CRITICAL]

    def warning_phases(self) -> List[PhaseCoverage]:
        return [p for p in self.phases if p.severity == SEVERITY_WARNING]


# =============================================================================
# Rule loading
# =============================================================================


_state: Dict[str, Any] = {
    "loaded": False,
    "rules": [],  # list of dict
}


def load() -> int:
    """Read factory + user rules into the module cache. Returns rule count."""
    if _state["loaded"]:
        return len(_state["rules"])
    rules: List[Dict[str, Any]] = []
    rules.extend(_read_rules_from(_DEFAULTS_PATH))
    user_rules = _read_rules_from(_user_overlay_path())
    if user_rules:
        # User rules take precedence on id clash.
        ids = {r["id"] for r in user_rules if isinstance(r, dict) and "id" in r}
        rules = [r for r in rules if r.get("id") not in ids] + user_rules
    _state["rules"] = rules
    _state["loaded"] = True
    return len(rules)


def reload() -> int:
    _state["loaded"] = False
    _state["rules"] = []
    return load()


def _read_rules_from(path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = read_data(path)
    except Exception as exc:
        log_warning(f"[registers.rewards_coverage] failed to read {path}: {exc}")
        return []
    if isinstance(data, dict):
        return list(data.get("rules", []) or [])
    if isinstance(data, list):
        return list(data)
    log_warning(
        f"[registers.rewards_coverage] unexpected payload shape in {path}: "
        f"{type(data).__name__}"
    )
    return []


# =============================================================================
# Helpers
# =============================================================================


def _term_polarity(weight: float, item_polarity: Optional[str]) -> str:
    """Decide a reward term's polarity: positive / negative / neutral.

    Priority: explicit ``item.polarity`` from the rewards registry if it
    maps cleanly (``reward`` → positive, ``penalty`` → negative); fall
    back to the weight's sign. A weight of zero with no registry hint is
    neutral.
    """
    if item_polarity:
        p = str(item_polarity).strip().lower()
        if p in ("reward", "positive", "+"):
            return "positive"
        if p in ("penalty", "negative", "-"):
            return "negative"
        if p in ("bidirectional", "neutral", "info"):
            pass  # fall through to weight sign
    if weight > 1e-9:
        return "positive"
    if weight < -1e-9:
        return "negative"
    return "neutral"


def _resolve_item_polarity(term_key: str, kind: str, backend: str) -> Optional[str]:
    """Look up the registered TaskModuleItem polarity. Returns None on miss."""
    try:
        from scripts import lookup as _lookup
        item = _lookup(term_key, kind=kind, backend=backend)
    except Exception:
        return None
    if item is None:
        return None
    pol = getattr(item, "polarity", None)
    return str(pol) if pol else None


# =============================================================================
# Public API
# =============================================================================


def evaluate(
    reward_terms: Dict[str, Any],
    phases: Iterable[Dict[str, Any]],
    *,
    kind: str = "reward",
    backend: str = "sb3",
    applies_to_key: str = "applies_to",
) -> CoverageReport:
    """Score a reward_terms dict against a phase layout.

    Parameters
    ----------
    reward_terms:
        Canvas-shape dict ``{term_key: {weight, ...other..., applies_to: [...]}}``
        or ``{term_key: weight}`` for legacy flat entries.
    phases:
        Iterable of dicts with at least ``id`` / ``display_name`` /
        ``polarity_required`` (matches the shape produced by
        :func:`registers.motion_phases.list_phases` after dict-ifying).
    kind / backend:
        Passed through to ``scripts.lookup`` for registry-driven polarity
        resolution. Falls back to weight-sign on miss.
    applies_to_key:
        Field name on each reward entry that lists phase ids the term
        contributes to. Missing or empty list = unconditional.

    Returns
    -------
    CoverageReport
        ``.severity`` collapses every phase into one badge color;
        ``.phases`` holds the detailed breakdown for the Inspector;
        ``.blocking`` is True when at least one critical rule was hit.
    """
    if not _state["loaded"]:
        load()
    rules = _state["rules"]

    phase_list = list(phases)
    # Pre-bin terms into two buckets per phase:
    #   * phase-specific: applies_to contains this phase id
    #   * unconditional : applies_to is empty (active everywhere at runtime)
    # We do NOT auto-bucket unconditional terms into each phase's specific
    # bucket. Doing so would mask the Go2 bug: there, every reward term is
    # unconditional, so every phase would appear to have positive +
    # negative coverage even though no reward is *isolated* to the static
    # phase — and the unconditional positives (feet_air_time / gait)
    # actively encourage limb-flailing while standing.
    specific: Dict[str, List[Dict[str, Any]]] = {p["id"]: [] for p in phase_list}
    unconditional: List[Dict[str, Any]] = []
    has_terms = False
    for term_key, val in (reward_terms or {}).items():
        has_terms = True
        if isinstance(val, dict):
            weight = float(val.get("weight", 0.0) or 0.0)
            applies = val.get(applies_to_key) or []
        else:
            try:
                weight = float(val if val is not None else 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            applies = []
        if isinstance(applies, str):
            applies = [s.strip() for s in applies.split(",") if s.strip()]
        elif isinstance(applies, (list, tuple)):
            applies = [str(s).strip() for s in applies if str(s).strip()]
        else:
            applies = []
        item_polarity = _resolve_item_polarity(term_key, kind=kind, backend=backend)
        polarity = _term_polarity(weight, item_polarity)
        bucket_entry = {
            "key": term_key,
            "weight": weight,
            "polarity": polarity,
        }
        if applies:
            for pid in applies:
                if pid in specific:
                    specific[pid].append(bucket_entry)
        else:
            unconditional.append(bucket_entry)

    uncond_pos = [e["key"] for e in unconditional if e["polarity"] == "positive"]
    uncond_neg = [e["key"] for e in unconditional if e["polarity"] == "negative"]
    uncond_neu = [e["key"] for e in unconditional if e["polarity"] == "neutral"]

    overall_severity = SEVERITY_OK
    overall_blocking = False
    phase_reports: List[PhaseCoverage] = []
    for phase in phase_list:
        pid = phase["id"]
        bucket = specific.get(pid, [])
        positive = [e["key"] for e in bucket if e["polarity"] == "positive"]
        negative = [e["key"] for e in bucket if e["polarity"] == "negative"]
        neutral = [e["key"] for e in bucket if e["polarity"] == "neutral"]
        pc = PhaseCoverage(
            phase_id=pid,
            display_name=str(phase.get("display_name") or pid),
            polarity_required=str(phase.get("polarity_required") or "any"),
            positive_terms=positive,
            negative_terms=negative,
            neutral_terms=neutral,
            unconditional_positive_terms=list(uncond_pos),
            unconditional_negative_terms=list(uncond_neg),
            unconditional_neutral_terms=list(uncond_neu),
        )
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            try:
                if _rule_matches(rule, pc):
                    sev = str(rule.get("severity") or SEVERITY_WARNING)
                    pc.severity = _severity_max(pc.severity, sev)
                    pc.triggered_rule_ids.append(str(rule.get("id") or "?"))
                    msg = str(rule.get("message") or "").strip()
                    if msg and not pc.message:
                        pc.message = msg
                    sugg = rule.get("suggest_add_terms") or []
                    if isinstance(sugg, list):
                        for s in sugg:
                            if s and s not in pc.suggestions:
                                pc.suggestions.append(str(s))
            except Exception as exc:
                log_warning(
                    f"[registers.rewards_coverage] rule "
                    f"{rule.get('id', '?')!r} failed on phase {pid!r}: {exc}"
                )
        if pc.severity == SEVERITY_CRITICAL:
            overall_blocking = True
        overall_severity = _severity_max(overall_severity, pc.severity)
        phase_reports.append(pc)

    return CoverageReport(
        severity=overall_severity,
        blocking=overall_blocking,
        phases=phase_reports,
        has_terms=has_terms,
    )


def _rule_matches(rule: Dict[str, Any], pc: PhaseCoverage) -> bool:
    """Evaluate a rule's ``when`` clause against a single phase report.

    Supported when fields:
        * ``polarity_required_in``         — list[str] of phase.polarity_required to scope to
        * ``positive_count``               — phase-specific positive terms (dict({"min"/"max"/"eq"}))
        * ``negative_count``               — phase-specific negative terms
        * ``neutral_count``                — phase-specific neutral terms
        * ``total_count``                  — phase-specific (positive + negative + neutral)
        * ``unconditional_positive_count`` — terms with empty applies_to + positive weight
        * ``unconditional_negative_count`` — terms with empty applies_to + negative weight
        * ``all_negative_count``           — phase-specific + unconditional negatives
        * ``all_positive_count``           — phase-specific + unconditional positives

    Unknown fields are ignored so future rules can ship without breaking
    older runtimes.
    """
    when = rule.get("when", {}) or {}
    pol_req_in = when.get("polarity_required_in")
    if isinstance(pol_req_in, list) and pc.polarity_required not in pol_req_in:
        return False
    counts = {
        "positive_count": len(pc.positive_terms),
        "negative_count": len(pc.negative_terms),
        "neutral_count": len(pc.neutral_terms),
        "total_count": len(pc.positive_terms) + len(pc.negative_terms) + len(pc.neutral_terms),
        "unconditional_positive_count": len(pc.unconditional_positive_terms),
        "unconditional_negative_count": len(pc.unconditional_negative_terms),
        "unconditional_neutral_count": len(pc.unconditional_neutral_terms),
        "all_positive_count": len(pc.all_positive_terms),
        "all_negative_count": len(pc.all_negative_terms),
    }
    for key, expected in when.items():
        if key == "polarity_required_in":
            continue
        if key not in counts:
            continue
        actual = counts[key]
        if isinstance(expected, dict):
            if "min" in expected and actual < int(expected["min"]):
                return False
            if "max" in expected and actual > int(expected["max"]):
                return False
            if "eq" in expected and actual != int(expected["eq"]):
                return False
        elif isinstance(expected, int):
            if actual != expected:
                return False
    return True


__all__ = [
    "PhaseCoverage",
    "CoverageReport",
    "SEVERITY_OK",
    "SEVERITY_WARNING",
    "SEVERITY_CRITICAL",
    "load",
    "reload",
    "evaluate",
]
