#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Action profile comparison and compatibility report — Circle 3 STEP 3.2.

Provides semantic-level comparison between two adapter action profiles (source
and target).  Comparison is performed at the ``intent_id`` level — a raw-name
difference alone is never classified as incompatible.

Public API
----------
CompatStatus                   — string constants for classification outcomes
ActionCompatEntry              — one intent's compatibility result (dataclass)
ActionProfileReport            — full comparison result containing all entries
compare_action_profiles(...)   — build a report from two descriptor lists

Classification rules:
    preserved   — intent exists in both profiles; target is AVAILABLE;
                  raw_action may differ (remap is transparent to authoring).
    remapped    — same as preserved but raw_action actually differs, surfaced
                  explicitly for diagnostics.
    degraded    — intent exists in target with DEGRADED availability.
    unsupported — intent exists in target with UNSUPPORTED availability, or
                  the intent is completely absent from the target profile.
    ambiguous   — multiple source entries share the same intent_id
                  (registry invariant violation; reported but not fatal).

Design constraints:
    - Pure Python; no Qt imports; no adapter instances.
    - Dataclasses are immutable (frozen=True).
    - Comparison never mutates input lists.
    - classify_intent() is a pure function, independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor


# ---------------------------------------------------------------------------
# Classification status constants
# ---------------------------------------------------------------------------

class CompatStatus:
    """String constants for ``ActionCompatEntry.status``.

    Ordered from best to worst outcome:
        PRESERVED   — full semantic + execution compatibility.
        REMAPPED    — compatible intent; raw backend action differs (still ok).
        DEGRADED    — intent executes on target but with reduced capability.
        UNSUPPORTED — no executable path on target model.
        AMBIGUOUS   — source profile has conflicting entries for the same intent.
    """

    PRESERVED   = "preserved"
    REMAPPED    = "remapped"
    DEGRADED    = "degraded"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS   = "ambiguous"

    _ALL: frozenset = frozenset({
        "preserved", "remapped", "degraded", "unsupported", "ambiguous",
    })


# ---------------------------------------------------------------------------
# Entry and report dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionCompatEntry:
    """Compatibility result for a single semantic intent.

    Fields
    ------
    intent_id        : The canonical semantic intent being compared.
    status           : One of the ``CompatStatus`` constants.
    source_raw_action: Raw backend action on the source model (or ``""``).
    target_raw_action: Raw backend action on the target model (``None`` when
                       unsupported or absent from target).
    reason           : Human-readable explanation of the classification.
                       Machine-readable prefix hints are not guaranteed; use
                       ``status`` for programmatic branching.
    """

    intent_id: str
    status: str
    source_raw_action: str
    target_raw_action: Optional[str]
    reason: Optional[str] = None


@dataclass(frozen=True)
class ActionProfileReport:
    """Full compatibility report between a source and target action profile.

    Attributes are read-only lists, one per ``CompatStatus`` outcome.
    Access ``all_entries`` for the flat ordered list.

    Usage::

        report = compare_action_profiles(source_list, target_list)
        for entry in report.unsupported:
            print(f"{entry.intent_id}: {entry.reason}")
    """

    preserved:   List[ActionCompatEntry] = field(default_factory=list)
    remapped:    List[ActionCompatEntry] = field(default_factory=list)
    degraded:    List[ActionCompatEntry] = field(default_factory=list)
    unsupported: List[ActionCompatEntry] = field(default_factory=list)
    ambiguous:   List[ActionCompatEntry] = field(default_factory=list)

    @property
    def all_entries(self) -> List[ActionCompatEntry]:
        """Flat list of all entries, ordered: preserved, remapped, degraded,
        unsupported, ambiguous."""
        return (
            list(self.preserved)
            + list(self.remapped)
            + list(self.degraded)
            + list(self.unsupported)
            + list(self.ambiguous)
        )

    @property
    def is_fully_compatible(self) -> bool:
        """True iff every source intent is preserved or remapped on target."""
        return len(self.degraded) == 0 and len(self.unsupported) == 0 and len(self.ambiguous) == 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for diagnostics / logging."""
        def _entry_list(entries):
            return [
                {
                    "intent_id":         e.intent_id,
                    "status":            e.status,
                    "source_raw_action": e.source_raw_action,
                    "target_raw_action": e.target_raw_action,
                    "reason":            e.reason,
                }
                for e in entries
            ]
        return {
            "preserved":         _entry_list(self.preserved),
            "remapped":          _entry_list(self.remapped),
            "degraded":          _entry_list(self.degraded),
            "unsupported":       _entry_list(self.unsupported),
            "ambiguous":         _entry_list(self.ambiguous),
            "is_fully_compatible": self.is_fully_compatible,
        }


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_intent(
    source: SemanticActionDescriptor,
    target: Optional[SemanticActionDescriptor],
) -> ActionCompatEntry:
    """Classify compatibility for one *source* intent against its *target* match.

    Args:
        source: Descriptor from the source (authoring) profile.
        target: Corresponding descriptor from the target profile (same
                ``intent_id``), or ``None`` when absent from target.

    Returns:
        An ``ActionCompatEntry`` with the appropriate ``CompatStatus``.
    """
    intent_id = source.intent_id
    src_raw   = source.raw_action

    # Intent completely absent from target profile.
    if target is None:
        return ActionCompatEntry(
            intent_id=intent_id,
            status=CompatStatus.UNSUPPORTED,
            source_raw_action=src_raw,
            target_raw_action=None,
            reason=f"intent {intent_id!r} has no declaration on target model",
        )

    tgt_raw = target.raw_action

    if target.availability == ActionAvailability.UNSUPPORTED:
        return ActionCompatEntry(
            intent_id=intent_id,
            status=CompatStatus.UNSUPPORTED,
            source_raw_action=src_raw,
            target_raw_action=tgt_raw,
            reason=(
                f"intent {intent_id!r} is declared UNSUPPORTED on target model "
                f"(source raw={src_raw!r})"
            ),
        )

    if target.availability == ActionAvailability.DEGRADED:
        return ActionCompatEntry(
            intent_id=intent_id,
            status=CompatStatus.DEGRADED,
            source_raw_action=src_raw,
            target_raw_action=tgt_raw,
            reason=(
                f"intent {intent_id!r} is DEGRADED on target model "
                f"(source raw={src_raw!r}, target raw={tgt_raw!r})"
            ),
        )

    # Target is AVAILABLE — preserved or remapped.
    if src_raw == tgt_raw:
        return ActionCompatEntry(
            intent_id=intent_id,
            status=CompatStatus.PRESERVED,
            source_raw_action=src_raw,
            target_raw_action=tgt_raw,
            reason=None,  # no explanation needed for clean preservation
        )

    # Raw action differs but semantic intent is the same — remapped.
    return ActionCompatEntry(
        intent_id=intent_id,
        status=CompatStatus.REMAPPED,
        source_raw_action=src_raw,
        target_raw_action=tgt_raw,
        reason=(
            f"intent {intent_id!r} remapped: source raw={src_raw!r} → "
            f"target raw={tgt_raw!r}"
        ),
    )


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

def compare_action_profiles(
    source: List[SemanticActionDescriptor],
    target: List[SemanticActionDescriptor],
) -> ActionProfileReport:
    """Build a full compatibility report between *source* and *target* profiles.

    Comparison is performed at the ``intent_id`` level.  A difference in
    ``raw_action`` alone produces a ``REMAPPED`` (compatible) result, never an
    incompatibility error.

    Ambiguity detection: if *source* contains multiple entries with the same
    ``intent_id``, those entries are classified as ``AMBIGUOUS`` and excluded
    from the normal comparison path.

    Args:
        source: Descriptor list from the authoring / source model profile.
        target: Descriptor list from the destination / target model profile.

    Returns:
        An immutable ``ActionProfileReport`` instance.
    """
    # Index target by intent_id for O(1) lookup.
    target_index: Dict[str, SemanticActionDescriptor] = {}
    for d in target:
        if isinstance(d, SemanticActionDescriptor):
            # Last entry wins when target has duplicates (same as registry behaviour).
            target_index[d.intent_id] = d

    # Detect duplicate intent_ids in source.
    source_count: Dict[str, int] = {}
    for d in source:
        if isinstance(d, SemanticActionDescriptor):
            source_count[d.intent_id] = source_count.get(d.intent_id, 0) + 1

    preserved:   List[ActionCompatEntry] = []
    remapped:    List[ActionCompatEntry] = []
    degraded:    List[ActionCompatEntry] = []
    unsupported: List[ActionCompatEntry] = []
    ambiguous:   List[ActionCompatEntry] = []

    seen_source: set = set()

    for d in source:
        if not isinstance(d, SemanticActionDescriptor):
            continue

        intent_id = d.intent_id

        # Ambiguous: multiple source entries share the same intent_id.
        if source_count[intent_id] > 1:
            if intent_id not in seen_source:
                ambiguous.append(ActionCompatEntry(
                    intent_id=intent_id,
                    status=CompatStatus.AMBIGUOUS,
                    source_raw_action=d.raw_action,
                    target_raw_action=None,
                    reason=(
                        f"source profile has {source_count[intent_id]} entries "
                        f"for intent {intent_id!r} — registry invariant violated"
                    ),
                ))
            seen_source.add(intent_id)
            continue

        if intent_id in seen_source:
            continue
        seen_source.add(intent_id)

        entry = classify_intent(d, target_index.get(intent_id))

        if entry.status == CompatStatus.PRESERVED:
            preserved.append(entry)
        elif entry.status == CompatStatus.REMAPPED:
            remapped.append(entry)
        elif entry.status == CompatStatus.DEGRADED:
            degraded.append(entry)
        elif entry.status == CompatStatus.UNSUPPORTED:
            unsupported.append(entry)
        else:
            ambiguous.append(entry)

    return ActionProfileReport(
        preserved=preserved,
        remapped=remapped,
        degraded=degraded,
        unsupported=unsupported,
        ambiguous=ambiguous,
    )
