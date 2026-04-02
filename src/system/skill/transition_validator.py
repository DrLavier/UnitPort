"""TransitionValidator — pre-flight validation of a skill chain.

Given a topologically-sorted list of :class:`SkillManifest` instances (from
the Mission DAG), this module checks every adjacent pair for compatibility
and attempts to resolve gaps via :mod:`transition_library`.

Usage::

    from src.system.skill.transition_validator import validate_skill_chain

    report = validate_skill_chain([manifest_a, manifest_b, manifest_c])
    if not report.is_valid:
        # at least one INCOMPATIBLE pair — block execution
        ...
    for tid in report.insertions:
        # auto-insert these transition behaviours into the execution graph
        ...
"""

from __future__ import annotations

import logging
from typing import List

from src.system.skill.skill_manifest import SkillManifest
from src.system.skill.transition_library import lookup_transition
from src.system.skill.transition_result import (
    TransitionCheckResult,
    TransitionStatus,
    TransitionValidationReport,
    TransitionWarning,
    TransitionWarningKind,
)

log = logging.getLogger(__name__)

# If control_frequency_hz ratio between adjacent skills exceeds this,
# emit a warning.
_FREQUENCY_RATIO_THRESHOLD = 4.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_skill_chain(
    manifests: List[SkillManifest],
) -> TransitionValidationReport:
    """Validate an ordered chain of skill manifests.

    Returns a :class:`TransitionValidationReport` covering every adjacent
    pair ``(manifests[i], manifests[i+1])``.
    """
    results: List[TransitionCheckResult] = []

    for i in range(len(manifests) - 1):
        a = manifests[i]
        b = manifests[i + 1]
        result = _check_pair(a, b)
        results.append(result)

    report = TransitionValidationReport(results=results)

    if not report.is_valid:
        log.warning(
            "Skill chain has %d incompatible transition(s)",
            sum(1 for r in results if r.status == TransitionStatus.INCOMPATIBLE),
        )

    return report


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _check_pair(a: SkillManifest, b: SkillManifest) -> TransitionCheckResult:
    """Check a single (A → B) transition."""
    warnings: List[TransitionWarning] = []
    reasons: List[str] = []
    needs_transition = False

    # 1. Posture compatibility
    post = a.postcondition.posture
    pre = b.precondition.posture

    if pre != "any" and post != pre:
        needs_transition = True
        reasons.append(
            f"posture mismatch: {a.skill_id} ends '{post}', "
            f"{b.skill_id} expects '{pre}'"
        )

    # 2. Velocity compatibility
    if a.postcondition.velocity_max_mps > b.precondition.velocity_max_mps:
        needs_transition = True
        reasons.append(
            f"velocity mismatch: {a.skill_id} may exit at "
            f"{a.postcondition.velocity_max_mps} m/s, "
            f"{b.skill_id} requires ≤{b.precondition.velocity_max_mps} m/s"
        )

    # 3. Action space type compatibility (warning if different)
    if a.action_space_type != b.action_space_type:
        warnings.append(TransitionWarning(
            kind=TransitionWarningKind.ACTION_SPACE_MISMATCH,
            message=(
                f"action_space_type changes: {a.skill_id} uses "
                f"{a.action_space_type.value}, {b.skill_id} uses "
                f"{b.action_space_type.value}"
            ),
        ))

    # 4. Control frequency ratio (warning if >4x)
    if a.control_frequency_hz > 0 and b.control_frequency_hz > 0:
        ratio = max(
            a.control_frequency_hz / b.control_frequency_hz,
            b.control_frequency_hz / a.control_frequency_hz,
        )
        if ratio > _FREQUENCY_RATIO_THRESHOLD:
            warnings.append(TransitionWarning(
                kind=TransitionWarningKind.FREQUENCY_RATIO,
                message=(
                    f"control frequency ratio {ratio:.1f}x between "
                    f"{a.skill_id} ({a.control_frequency_hz} Hz) and "
                    f"{b.skill_id} ({b.control_frequency_hz} Hz)"
                ),
            ))

    # --- Resolve ---
    if not needs_transition:
        return TransitionCheckResult(
            from_skill_id=a.skill_id,
            to_skill_id=b.skill_id,
            status=TransitionStatus.COMPATIBLE,
            warnings=warnings,
        )

    # Try to find a transition behaviour
    target_posture = pre if pre != "any" else "standing"
    entry = lookup_transition(post, target_posture)

    if entry is not None:
        return TransitionCheckResult(
            from_skill_id=a.skill_id,
            to_skill_id=b.skill_id,
            status=TransitionStatus.NEEDS_TRANSITION,
            reason="; ".join(reasons),
            transition_id=entry.skill_manifest.skill_id,
            warnings=warnings,
        )

    # No transition available — incompatible
    return TransitionCheckResult(
        from_skill_id=a.skill_id,
        to_skill_id=b.skill_id,
        status=TransitionStatus.INCOMPATIBLE,
        reason="; ".join(reasons) + " — no suitable transition behaviour found",
        warnings=warnings,
    )
