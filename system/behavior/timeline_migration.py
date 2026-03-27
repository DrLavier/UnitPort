#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior timeline legacy migration — Circle 4 STEP 4.2.

Detects legacy ``ActionSegment`` entries that store a raw action name in
``name`` but have no ``intent_id`` set (old v1.0 format), and resolves them
to semantic intent IDs using the central ActionRegistry.

Migration status codes
----------------------
ALREADY_SET   — intent_id was already populated; no action taken.
RESOLVED      — legacy raw action name successfully mapped to an intent_id.
AMBIGUOUS     — raw action matched but brand context was unavailable; the
                base catalog was used as a best-effort fallback.  Manual
                review recommended.
UNAVAILABLE   — no intent mapping found for the raw action name in any
                context; intent_id left empty with a warning recorded.

Public API
----------
MigrationStatus                  — string constants for migration outcomes
ActionSegmentMigrationResult     — result for one ActionSegment
TimelineMigrationReport          — aggregated result for a full timeline
migrate_action_segment(...)      — migrate one ActionSegment
migrate_timeline(...)            — migrate all segments in a BehaviorTimeline

Design constraints:
    - Pure Python; no Qt imports; no adapter instances.
    - Never raises; ambiguous/unavailable cases produce diagnostics, not exceptions.
    - Input objects are not mutated; migrated copies are returned.
    - Re-save after migration produces the new semantic format (v1.1+).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from system.behavior.action_profile import ActionSegment, BehaviorTimeline, BEHAVIOR_TIMELINE_VERSION
from system.behavior.action_registry import resolve_intent
from system.behavior.intent_catalog import get_base_catalog
from system.behavior.motor_topology import resolve_topology


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

class MigrationStatus:
    """String constants for ``ActionSegmentMigrationResult.status``."""

    ALREADY_SET  = "already_set"   # intent_id was present — no change made
    RESOLVED     = "resolved"       # raw action → intent_id via brand registry
    AMBIGUOUS    = "ambiguous"      # resolved via base catalog (no brand context)
    UNAVAILABLE  = "unavailable"    # no mapping found; intent_id left empty

    _ALL: frozenset = frozenset({"already_set", "resolved", "ambiguous", "unavailable"})


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSegmentMigrationResult:
    """Migration result for a single ActionSegment.

    Fields
    ------
    action_id   : Identifies which segment this result applies to.
    raw_action  : The ``name`` field value from the legacy segment.
    intent_id   : The resolved intent_id (empty string when UNAVAILABLE).
    status      : One of the ``MigrationStatus`` constants.
    warning     : Human-readable explanation when status is AMBIGUOUS or
                  UNAVAILABLE; None when ALREADY_SET or RESOLVED cleanly.
    """

    action_id: str
    raw_action: str
    intent_id: str
    status: str
    warning: Optional[str] = None


@dataclass(frozen=True)
class TimelineMigrationReport:
    """Aggregated migration report for a full BehaviorTimeline.

    Attributes
    ----------
    timeline    : The migrated BehaviorTimeline (new object; input unchanged).
    results     : One result per ActionSegment in declaration order.
    """

    timeline: BehaviorTimeline
    results: List[ActionSegmentMigrationResult]

    @property
    def already_set_count(self) -> int:
        return sum(1 for r in self.results if r.status == MigrationStatus.ALREADY_SET)

    @property
    def resolved_count(self) -> int:
        return sum(1 for r in self.results if r.status == MigrationStatus.RESOLVED)

    @property
    def ambiguous_count(self) -> int:
        return sum(1 for r in self.results if r.status == MigrationStatus.AMBIGUOUS)

    @property
    def unavailable_count(self) -> int:
        return sum(1 for r in self.results if r.status == MigrationStatus.UNAVAILABLE)

    @property
    def warnings(self) -> List[str]:
        """All non-None warning strings from results."""
        return [r.warning for r in self.results if r.warning is not None]

    @property
    def needs_manual_review(self) -> bool:
        """True when any segment is AMBIGUOUS or UNAVAILABLE."""
        return self.ambiguous_count > 0 or self.unavailable_count > 0

    def to_dict(self) -> dict:
        return {
            "already_set":   self.already_set_count,
            "resolved":      self.resolved_count,
            "ambiguous":     self.ambiguous_count,
            "unavailable":   self.unavailable_count,
            "warnings":      self.warnings,
            "needs_manual_review": self.needs_manual_review,
        }


# ---------------------------------------------------------------------------
# Base catalog raw-action → intent_id reverse index (for ambiguous fallback)
# ---------------------------------------------------------------------------

def _build_base_raw_index() -> dict:
    """Build a {raw_action: intent_id} index from the brand-independent catalog.

    When multiple base entries share the same raw_action (unlikely but
    possible), the first entry wins.
    """
    index: dict = {}
    for d in get_base_catalog():
        if d.raw_action and d.raw_action not in index:
            index[d.raw_action] = d.intent_id
    return index


_BASE_RAW_INDEX: dict = _build_base_raw_index()


# ---------------------------------------------------------------------------
# Core migration helpers
# ---------------------------------------------------------------------------

def migrate_action_segment(
    segment: ActionSegment,
    brand: str = "",
    robot_type: str = "",
) -> tuple:
    """Migrate one ``ActionSegment`` to semantic intent storage.

    Args:
        segment:    The legacy (or already-migrated) ActionSegment.
        brand:      Robot brand context for registry lookup (e.g. ``"unitree"``).
                    Pass empty string when brand context is unavailable.
        robot_type: Model identifier within the brand (e.g. ``"go2"``).

    Returns:
        ``(migrated_segment, result)`` where ``migrated_segment`` is a new
        ``ActionSegment`` instance with ``intent_id`` populated (or unchanged
        when ALREADY_SET/UNAVAILABLE) and ``result`` is an
        ``ActionSegmentMigrationResult``.
    """
    # Already migrated — return as-is.
    if segment.intent_id:
        result = ActionSegmentMigrationResult(
            action_id=segment.action_id,
            raw_action=segment.name,
            intent_id=segment.intent_id,
            status=MigrationStatus.ALREADY_SET,
        )
        return segment, result

    raw = segment.name

    # Attempt brand-aware registry lookup first.
    if brand:
        descriptor = resolve_intent(raw, brand, robot_type)
        if descriptor is not None and descriptor.intent_id:
            migrated = dataclasses.replace(segment, intent_id=descriptor.intent_id)
            result = ActionSegmentMigrationResult(
                action_id=segment.action_id,
                raw_action=raw,
                intent_id=descriptor.intent_id,
                status=MigrationStatus.RESOLVED,
            )
            return migrated, result

    # Fallback: try the brand-independent base catalog.
    base_intent = _BASE_RAW_INDEX.get(raw)
    if base_intent:
        migrated = dataclasses.replace(segment, intent_id=base_intent)
        warning = (
            f"action {raw!r} resolved to {base_intent!r} via base catalog "
            f"(no brand context); verify this is the correct intent"
            if not brand
            else (
                f"action {raw!r} not found in {brand!r}/{robot_type!r} registry; "
                f"resolved to {base_intent!r} via base catalog fallback"
            )
        )
        result = ActionSegmentMigrationResult(
            action_id=segment.action_id,
            raw_action=raw,
            intent_id=base_intent,
            status=MigrationStatus.AMBIGUOUS,
            warning=warning,
        )
        return migrated, result

    # No mapping available.
    warning = (
        f"action {raw!r} could not be mapped to any semantic intent"
        + (f" for brand={brand!r} robot_type={robot_type!r}" if brand else "")
        + "; intent_id left empty — manual review required"
    )
    result = ActionSegmentMigrationResult(
        action_id=segment.action_id,
        raw_action=raw,
        intent_id="",
        status=MigrationStatus.UNAVAILABLE,
        warning=warning,
    )
    return segment, result


def migrate_timeline(
    timeline: BehaviorTimeline,
    brand: str = "",
    robot_type: str = "",
) -> TimelineMigrationReport:
    """Migrate all ``ActionSegment`` entries in a ``BehaviorTimeline``.

    Iterates over every segment, calls ``migrate_action_segment``, and
    assembles a new ``BehaviorTimeline`` with updated segments.  The input
    timeline is not mutated.

    Args:
        timeline:   The BehaviorTimeline to migrate (may be legacy v1.0 format).
        brand:      Robot brand context for registry lookup.  When empty the
                    migration falls back to the base catalog only.
        robot_type: Model identifier passed through to the registry.

    Returns:
        A ``TimelineMigrationReport`` containing the new timeline and per-
        segment results.  Re-serialising the returned timeline via
        ``timeline.to_dict()`` produces a v1.1 semantic-intent draft.
    """
    # Use robot_type from timeline as fallback when caller doesn't specify.
    effective_robot_type = robot_type or timeline.robot_type

    migrated_segments: List[ActionSegment] = []
    results: List[ActionSegmentMigrationResult] = []

    for seg in timeline.action_segments:
        new_seg, result = migrate_action_segment(seg, brand, effective_robot_type)
        migrated_segments.append(new_seg)
        results.append(result)

    # Build new timeline preserving all other fields; bump version to current.
    new_timeline = dataclasses.replace(
        timeline,
        action_segments=migrated_segments,
        version=BEHAVIOR_TIMELINE_VERSION,
    )
    return TimelineMigrationReport(timeline=new_timeline, results=results)


# ---------------------------------------------------------------------------
# Model-switch migration (Step 8) — explicit diagnostics, no silent drops
# ---------------------------------------------------------------------------

class ModelSwitchStatus:
    """String constants for ``MotorSegmentSwitchResult.status``."""

    COMPATIBLE   = "compatible"    # track_name exists in new model topology
    UNSUPPORTED  = "unsupported"   # track_name absent from new model topology
    UNCHECKED    = "unchecked"     # topology unavailable; no verdict possible

    _ALL: frozenset = frozenset({"compatible", "unsupported", "unchecked"})


@dataclass(frozen=True)
class MotorSegmentSwitchResult:
    """Compatibility verdict for one MotorSegment after a model switch.

    Fields
    ------
    motor_id     : Identifies the MotorSegment.
    track_name   : The ``track_name`` that was checked.
    action_id    : Parent action_id of the overlay.
    status       : One of the ``ModelSwitchStatus`` constants.
    warning      : Human-readable explanation when UNSUPPORTED or UNCHECKED;
                   None when COMPATIBLE.
    """

    motor_id:  str
    track_name: str
    action_id: str
    status:    str
    warning:   Optional[str] = None


@dataclass(frozen=True)
class ModelSwitchMigrationReport:
    """Aggregated result of checking a timeline after a model switch.

    The timeline is **never mutated** — authored segments are preserved
    regardless of compatibility.  Callers use the diagnostics to notify the
    user and decide whether to prune stale segments or leave them intact.

    Attributes
    ----------
    new_brand      : Target brand that was checked against.
    new_robot_type : Target robot_type that was checked against.
    results        : One result per MotorSegment across all overlays.
    topology_degraded : True when the target topology itself was degraded
                       (no canonical data for new_brand/new_robot_type).
    """

    new_brand:          str
    new_robot_type:     str
    results:            List[MotorSegmentSwitchResult]
    topology_degraded:  bool = False

    @property
    def compatible_count(self) -> int:
        return sum(1 for r in self.results if r.status == ModelSwitchStatus.COMPATIBLE)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for r in self.results if r.status == ModelSwitchStatus.UNSUPPORTED)

    @property
    def unchecked_count(self) -> int:
        return sum(1 for r in self.results if r.status == ModelSwitchStatus.UNCHECKED)

    @property
    def warnings(self) -> List[str]:
        return [r.warning for r in self.results if r.warning is not None]

    @property
    def has_stale_tracks(self) -> bool:
        """True when any authored motor segment references an unsupported track."""
        return self.unsupported_count > 0

    def to_dict(self) -> dict:
        return {
            "new_brand":         self.new_brand,
            "new_robot_type":    self.new_robot_type,
            "compatible":        self.compatible_count,
            "unsupported":       self.unsupported_count,
            "unchecked":         self.unchecked_count,
            "has_stale_tracks":  self.has_stale_tracks,
            "topology_degraded": self.topology_degraded,
            "warnings":          self.warnings,
        }


def migrate_timeline_on_model_switch(
    timeline: BehaviorTimeline,
    new_brand: str,
    new_robot_type: str,
) -> ModelSwitchMigrationReport:
    """Check authored MotorSegments against a new model's canonical topology.

    This function is called after the user switches robot brand or model.
    It does **not** mutate the timeline — the authored data is always preserved.
    Instead it returns a ``ModelSwitchMigrationReport`` whose ``warnings``
    field lists every stale track reference.  Callers should surface these
    warnings to the user (e.g. via the output log) so the mismatch is
    explicit, never silent.

    Design constraints:
        - Pure Python; no Qt imports; no adapter instances.
        - Never raises; topology load failures produce UNCHECKED verdicts.
        - Input timeline is not mutated.
        - Stale tracks are diagnosed, not silently dropped.

    Args:
        timeline:      The BehaviorTimeline whose motor segments to check.
        new_brand:     Target robot brand (e.g. "unitree").
        new_robot_type: Target robot model (e.g. "go2").

    Returns:
        A ``ModelSwitchMigrationReport`` describing compatibility.
    """
    try:
        topology = resolve_topology(new_brand, new_robot_type)
        topology_degraded = topology.is_degraded
        valid_tracks: Set[str] = set(topology.all_track_names)
    except Exception:
        topology_degraded = True
        valid_tracks = set()

    results: List[MotorSegmentSwitchResult] = []

    for overlay in timeline.motor_overlays:
        for seg in overlay.motor_segments:
            if topology_degraded:
                status = ModelSwitchStatus.UNCHECKED
                warning = (
                    f"track {seg.track_name!r} (action={overlay.action_id!r}): "
                    f"topology unavailable for {new_brand!r}/{new_robot_type!r}; "
                    f"compatibility unchecked"
                )
            elif seg.track_name in valid_tracks:
                status = ModelSwitchStatus.COMPATIBLE
                warning = None
            else:
                status = ModelSwitchStatus.UNSUPPORTED
                warning = (
                    f"track {seg.track_name!r} (action={overlay.action_id!r}): "
                    f"not found in {new_brand!r}/{new_robot_type!r} topology; "
                    f"segment preserved but may not render correctly — "
                    f"manual review recommended"
                )
            results.append(MotorSegmentSwitchResult(
                motor_id=seg.motor_id,
                track_name=seg.track_name,
                action_id=overlay.action_id,
                status=status,
                warning=warning,
            ))

    return ModelSwitchMigrationReport(
        new_brand=new_brand,
        new_robot_type=new_robot_type,
        results=results,
        topology_degraded=topology_degraded,
    )


# ---------------------------------------------------------------------------
# Compile-time topology validation (Step 9) — wired into BehaviorCompilerBridge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyValidationDiagnostic:
    """One topology-validation finding for a MotorSegment at compile time.

    Fields
    ------
    level      : "error" (blocks compile validity) or "warning" (advisory).
    code       : Machine-readable diagnostic code, e.g. "topology.unknown_track".
    track_name : The MotorSegment.track_name that was checked.
    motor_id   : The MotorSegment.motor_id being diagnosed.
    action_id  : Parent ActionMotorOverlay.action_id.
    param_key  : Authored param key involved (empty when not param-specific).
    limb_label : Canonical body-part label from the topology (e.g. "Front Left Leg").
                 Empty when the track is unknown or topology is degraded.
    joint_name : Canonical joint name from the MotorEntry (e.g. "FL_hip").
                 Empty when not resolvable.
    message    : Human-readable explanation, using canonical labels where possible.
    """

    level:      str   # "error" | "warning"
    code:       str
    track_name: str
    motor_id:   str
    action_id:  str
    param_key:  str = ""
    limb_label: str = ""
    joint_name: str = ""
    message:    str = ""

    def to_dict(self) -> dict:
        return {
            "level":      self.level,
            "code":       self.code,
            "track_name": self.track_name,
            "motor_id":   self.motor_id,
            "action_id":  self.action_id,
            "param_key":  self.param_key,
            "limb_label": self.limb_label,
            "joint_name": self.joint_name,
            "message":    self.message,
        }


def validate_timeline_topology(
    timeline: BehaviorTimeline,
    brand: str,
    robot_type: str,
) -> List[TopologyValidationDiagnostic]:
    """Validate authored MotorSegments against the canonical topology at compile time.

    Checks
    ------
    1. **Unknown track** (ERROR) — ``track_name`` is not present in the canonical
       topology for the given brand/robot_type.  Compile must not silently
       tolerate stale track references; this blocks ``artifact.is_valid``.

    2. **Param-key mismatch** (WARNING) — authored ``params`` dict contains a key
       that does not match the canonical ``param_key`` for any MotorEntry on that
       track.  Non-blocking; callers should surface as advisory diagnostics.

    3. **Topology degraded** (WARNING, one entry) — when the topology itself
       cannot be resolved (unknown brand/robot_type), the check is skipped but a
       single warning is emitted so the caller knows validation was incomplete.

    Design constraints:
        - Pure Python; no Qt imports.
        - Never raises; exceptions produce a single UNCHECKED warning.
        - Input timeline is not mutated.

    Args:
        timeline:   BehaviorTimeline whose motor segments to check.
        brand:      Target robot brand (e.g. ``"unitree"``).
        robot_type: Target robot model (e.g. ``"go2"``).

    Returns:
        List of ``TopologyValidationDiagnostic`` in overlay/segment order.
    """
    diags: List[TopologyValidationDiagnostic] = []

    try:
        topology = resolve_topology(brand, robot_type)
    except Exception:
        # Topology load failed — emit one umbrella warning, skip all segment checks.
        diags.append(TopologyValidationDiagnostic(
            level="warning",
            code="topology.unavailable",
            track_name="",
            motor_id="",
            action_id="",
            message=(
                f"Topology unavailable for {brand!r}/{robot_type!r}; "
                f"motor-track validation skipped"
            ),
        ))
        return diags

    if topology.is_degraded:
        diags.append(TopologyValidationDiagnostic(
            level="warning",
            code="topology.degraded",
            track_name="",
            motor_id="",
            action_id="",
            message=(
                f"Topology degraded for {brand!r}/{robot_type!r} "
                f"({topology.degraded_reason}); motor-track validation skipped"
            ),
        ))
        return diags

    # Build lookup structures from the resolved topology.
    valid_tracks: Set[str] = set(topology.all_track_names)
    # Map: track_name → set of canonical param_keys (from MotorEntry.param_key)
    canonical_params: Dict[str, Set[str]] = {}
    # Map: track_name → (limb_label, joint_name) for human-readable diagnostics
    track_labels: Dict[str, tuple] = {}  # track_name → (limb_label, joint_name)
    for limb in topology.all_limbs:
        for motor in limb.motors:
            if motor.track_name not in canonical_params:
                canonical_params[motor.track_name] = set()
            if motor.param_key:
                canonical_params[motor.track_name].add(motor.param_key)
            if motor.track_name not in track_labels:
                track_labels[motor.track_name] = (
                    getattr(limb, "label", "") or "",
                    getattr(motor, "joint_name", "") or "",
                )

    for overlay in timeline.motor_overlays:
        for seg in overlay.motor_segments:
            # Rule 1: unknown track — ERROR.
            if seg.track_name not in valid_tracks:
                diags.append(TopologyValidationDiagnostic(
                    level="error",
                    code="topology.unknown_track",
                    track_name=seg.track_name,
                    motor_id=seg.motor_id,
                    action_id=overlay.action_id,
                    message=(
                        f"Track {seg.track_name!r} is not defined in any "
                        f"{robot_type!r} body-part; "
                        f"authored segment cannot be dispatched to hardware"
                    ),
                ))
                continue  # Skip param check for unknown tracks.

            # Resolve canonical labels for this track.
            _limb_label, _joint_name = track_labels.get(seg.track_name, ("", ""))
            _label_ctx = (
                f"{_limb_label!r} ({_joint_name})"
                if _limb_label and _joint_name
                else (_limb_label or seg.track_name)
            )

            # Rule 2: param-key mismatch — WARNING.
            expected_keys = canonical_params.get(seg.track_name, set())
            if expected_keys and seg.params:
                authored_keys = set(seg.params.keys())
                unrecognised = authored_keys - expected_keys
                for bad_key in sorted(unrecognised):
                    diags.append(TopologyValidationDiagnostic(
                        level="warning",
                        code="topology.param_key_mismatch",
                        track_name=seg.track_name,
                        motor_id=seg.motor_id,
                        action_id=overlay.action_id,
                        param_key=bad_key,
                        limb_label=_limb_label,
                        joint_name=_joint_name,
                        message=(
                            f"Param {bad_key!r} on {_label_ctx} is not a "
                            f"canonical parameter for {robot_type!r} "
                            f"(expected: {sorted(expected_keys)})"
                        ),
                    ))

    return diags
