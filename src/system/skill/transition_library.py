"""TransitionLibrary — registry of transition behaviours.

Ships with built-in transitions for common posture changes.  Users can
register custom transitions from ``custom_mods/training/transitions/``.

Each transition behaviour is itself described by a :class:`SkillManifest`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.system.skill.skill_manifest import (
    ActionSpaceType,
    InferenceBackend,
    Postcondition,
    Precondition,
    SkillManifest,
    SourceType,
)

log = logging.getLogger(__name__)

# Registry key: (from_posture, to_posture)
_TransitionKey = Tuple[str, str]


@dataclass
class TransitionEntry:
    """A registered transition behaviour."""
    skill_manifest: SkillManifest
    bundle_path: Optional[Path] = None      # None for built-in PD controllers


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_registry: Dict[_TransitionKey, TransitionEntry] = {}


def register_transition(
    from_posture: str,
    to_posture: str,
    skill_manifest: SkillManifest,
    bundle_path: Optional[Path] = None,
) -> None:
    """Register a transition behaviour for (from_posture → to_posture)."""
    key: _TransitionKey = (from_posture, to_posture)
    _registry[key] = TransitionEntry(
        skill_manifest=skill_manifest,
        bundle_path=bundle_path,
    )
    log.debug("Registered transition: %s → %s  (skill=%s)", from_posture, to_posture, skill_manifest.skill_id)


def lookup_transition(from_posture: str, to_posture: str) -> Optional[TransitionEntry]:
    """Find a registered transition for the given posture pair.

    Tries exact match first, then falls back to wildcard ``"any"`` sources.
    """
    # Exact match
    entry = _registry.get((from_posture, to_posture))
    if entry is not None:
        return entry
    # Wildcard source
    return _registry.get(("any", to_posture))


def list_transitions() -> List[Tuple[str, str, str]]:
    """Return ``[(from, to, skill_id), ...]`` for all registered transitions."""
    return [
        (k[0], k[1], v.skill_manifest.skill_id)
        for k, v in _registry.items()
    ]


def clear_registry() -> None:
    """Remove all registered transitions (useful for testing)."""
    _registry.clear()


# ---------------------------------------------------------------------------
# Built-in transitions
# ---------------------------------------------------------------------------

def _make_builtin(
    skill_id: str,
    name: str,
    from_posture: str,
    to_posture: str,
    description: str,
) -> SkillManifest:
    """Create a SkillManifest for a built-in PD-based transition controller."""
    return SkillManifest(
        skill_id=skill_id,
        skill_name=name,
        version="1.0.0",
        source_type=SourceType.MANUAL_SCRIPT,
        action_space_type=ActionSpaceType.JOINT_POSITION,
        action_dim=12,
        control_frequency_hz=50.0,
        required_sensors=["imu", "joint_encoder"],
        observation_dim=45,
        target_robot_family="quadruped",
        precondition=Precondition(posture=from_posture),
        postcondition=Postcondition(posture=to_posture),
        inference_backend=InferenceBackend.PYTHON_CALLABLE,
        model_path="",
        description=description,
        tags=["builtin_transition"],
    )


def _register_builtins() -> None:
    """Register the built-in transition behaviours."""
    # any → standing: recovery controller (PD-based, no ML)
    register_transition(
        "any", "standing",
        _make_builtin(
            "builtin_recovery_to_standing",
            "Recovery → Standing",
            "any", "standing",
            "PD-based recovery controller that brings the robot to a standing posture from any state.",
        ),
    )

    # standing → walking_ready: shift weight forward
    register_transition(
        "standing", "walking_ready",
        _make_builtin(
            "builtin_standing_to_walking_ready",
            "Standing → Walking Ready",
            "standing", "walking_ready",
            "Shifts weight forward and adjusts stance for locomotion initiation.",
        ),
    )

    # walking → standing: decelerate + stabilise
    register_transition(
        "walking", "standing",
        _make_builtin(
            "builtin_walking_to_standing",
            "Walking → Standing",
            "walking", "standing",
            "Decelerates and stabilises the robot to a standing posture.",
        ),
    )


# Auto-register builtins on module import
_register_builtins()
