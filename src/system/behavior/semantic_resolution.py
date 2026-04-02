#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic intent resolution for runtime behavior execution — Circle 7 Step 7.1.

Provides late binding of semantic intent IDs to raw backend actions against
the *currently active* brand/model at execution time.  Resolution is performed
through the central :func:`ActionRegistry.resolve_raw_action` function so it
always reflects the live capability profile, not stale authoring state.

Public API
----------
SemanticResolutionResult  — result dataclass (resolved map, unresolvable list,
                            per-intent diagnostics)
resolve_timeline_intents(timeline_data, brand, robot_type, trace_id)
                          → SemanticResolutionResult

Design
------
- Pure Python; no Qt, no SDK calls, no file I/O.
- ``resolve_timeline_intents`` never raises; all errors are captured in the
  returned result.
- When ``brand`` is empty the function returns a no-op result so legacy paths
  (no active brand/model) pass through without failing.
- UNSUPPORTED intents (resolver returns ``None``) are classified as
  unresolvable and trigger a ``resolution.unsupported.<intent_id>`` error
  diagnostic.
- Resolved intents (resolver returns a raw_action string) emit an info-level
  diagnostic: ``resolution.resolved.<intent_id>``, including both the intent
  ID and the resolved raw action for audit/logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SemanticResolutionResult:
    """Outcome of resolving all semantic intents in a behavior timeline.

    Attributes
    ----------
    resolved       : ``{intent_id: raw_action}`` for every successfully
                     resolved intent.  Empty when no movement segments had
                     intent IDs, or when brand was empty.
    unresolvable   : List of intent IDs that could not be resolved
                     (UNSUPPORTED or absent from the target model's registry).
                     Non-empty means the invocation MUST be blocked.
    diagnostics    : Per-intent info/error diagnostics suitable for surfacing
                     in :class:`BehaviorInvokeOutput.resolution_diagnostics`.
    skipped        : True when resolution was skipped entirely because no
                     brand was provided (legacy / unknown-model path).
    """

    resolved:     Dict[str, str] = field(default_factory=dict)
    unresolvable: List[str]      = field(default_factory=list)
    diagnostics:  list           = field(default_factory=list)   # List[BehaviorDiagnostic]
    skipped:      bool           = False

    @property
    def is_clean(self) -> bool:
        """True when all intents resolved successfully (or were skipped)."""
        return len(self.unresolvable) == 0

    @property
    def resolved_count(self) -> int:
        return len(self.resolved)

    @property
    def unresolvable_count(self) -> int:
        return len(self.unresolvable)


# ---------------------------------------------------------------------------
# Resolution function
# ---------------------------------------------------------------------------

def resolve_timeline_intents(
    timeline_data: Optional[Dict],
    brand: str,
    robot_type: str,
    trace_id: str = "",
) -> SemanticResolutionResult:
    """Resolve semantic intent IDs from a timeline against the active model.

    Extracts all unique ``intent_id`` values from the timeline's movement
    action segments, then calls :func:`resolve_raw_action` for each against
    the given ``brand``/``robot_type``.

    Emits one diagnostic per intent:
    - ``resolution.resolved.<intent_id>``  (info) on success — includes both
      the intent ID and the resolved raw action so operators can trace the
      late-binding mapping.
    - ``resolution.unsupported.<intent_id>`` (error) on failure — blocks
      the invocation; includes the intent ID so the operator knows exactly
      which action is not supported on the target model.

    Parameters
    ----------
    timeline_data : Serialised :class:`BehaviorTimeline` dict (from
                    ``BehaviorArtifact.timeline_data``).  ``None`` or
                    non-dict input → skipped result (no error).
    brand         : Active robot brand (e.g. ``"unitree"``).  Empty string
                    → skipped result (legacy path, no brand context).
    robot_type    : Active robot type (e.g. ``"go2"``).
    trace_id      : Correlation ID propagated to each diagnostic.

    Returns
    -------
    :class:`SemanticResolutionResult` — always returned; never raises.
    """
    from src.system.behavior.behavior_artifact import BehaviorDiagnostic
    from src.system.behavior.action_registry import resolve_raw_action

    # ------------------------------------------------------------------
    # Guard: no brand → legacy path, no resolution needed
    # ------------------------------------------------------------------
    if not brand:
        return SemanticResolutionResult(skipped=True)

    # ------------------------------------------------------------------
    # Guard: no timeline data → nothing to resolve
    # ------------------------------------------------------------------
    if not isinstance(timeline_data, dict):
        return SemanticResolutionResult(skipped=True)

    action_segments = timeline_data.get("action_segments") or []
    if not isinstance(action_segments, list):
        return SemanticResolutionResult(skipped=True)

    # ------------------------------------------------------------------
    # Collect unique intent IDs from movement segments
    # ------------------------------------------------------------------
    seen: set = set()
    unique_intents: List[tuple] = []  # (intent_id, source_raw_action)
    for seg in action_segments:
        if not isinstance(seg, dict):
            continue
        if seg.get("kind", "movement") != "movement":
            continue
        intent_id = (seg.get("intent_id") or "").strip()
        if not intent_id or intent_id in seen:
            continue
        seen.add(intent_id)
        unique_intents.append((intent_id, str(seg.get("name") or intent_id)))

    if not unique_intents:
        return SemanticResolutionResult(skipped=True)

    # ------------------------------------------------------------------
    # Resolve each intent against the active brand/model
    # ------------------------------------------------------------------
    resolved:     Dict[str, str] = {}
    unresolvable: List[str]      = []
    diagnostics:  list           = []

    for intent_id, source_raw in unique_intents:
        raw_action = resolve_raw_action(intent_id, brand, robot_type)

        if raw_action is not None:
            resolved[intent_id] = raw_action
            diagnostics.append(
                BehaviorDiagnostic.info(
                    code=f"resolution.resolved.{intent_id}",
                    message=(
                        f"intent {intent_id!r} resolved to raw_action={raw_action!r} "
                        f"on {brand}/{robot_type}"
                    ),
                    location=intent_id,
                    trace_id=trace_id or None,
                )
            )
        else:
            unresolvable.append(intent_id)
            diagnostics.append(
                BehaviorDiagnostic.error(
                    code=f"resolution.unsupported.{intent_id}",
                    message=(
                        f"intent {intent_id!r} (source raw={source_raw!r}) has no "
                        f"executable mapping on {brand}/{robot_type} — action is "
                        f"UNSUPPORTED or absent from the target model's registry"
                    ),
                    location=intent_id,
                    trace_id=trace_id or None,
                )
            )

    return SemanticResolutionResult(
        resolved=resolved,
        unresolvable=unresolvable,
        diagnostics=diagnostics,
        skipped=False,
    )
