"""obs_adapter — Isaac→MuJoCo observation reproducibility classifier.

This module is the heart of the sim2sim story for the Mission runtime.
A bundle trained in Isaac Lab declares its required observations through
``SkillManifest.required_sensors``. Some of those observations have
direct MuJoCo equivalents (joint encoders, on-body IMU); others are
Isaac-specific (height_scan against a USD terrain) or scene-dependent
(lidar / camera) and need a documented degradation strategy when the
field cannot reproduce them.

Per knowledge_base/mission_design.yaml §6 + Q6 resolution:
    "Switch first, registry later. v1 obs_adapter.py uses a hard-coded
     switch over ~5 obs types. Refactor to a registry once ≥8 distinct
     obs types are in use."

Contract
--------
- ``classify(sensor, field)`` returns an :class:`ObsRemapDecision`
  describing whether the sensor is producible, what strategy will be
  used, and an optional warning string.
- ``classify_required_sensors(manifest, field)`` walks
  ``manifest.required_sensors`` and returns the full
  :class:`ObsRemapReport`.
- The adapter NEVER silently fakes data — every degradation produces
  an entry in ``ObsRemapReport.warnings`` so the BehaviorNode can
  surface them through ``RuntimeResult.diagnostics``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ObsProduceability(str, Enum):
    """How an obs component will be obtained at runtime."""

    NATIVE = "native"           # produced by mj_data directly (joints, IMU)
    FIELD_DEPENDENT = "field_dependent"   # needs a non-flat field; substituted otherwise
    DEGRADED = "degraded"       # not implemented yet; zero-filled with WARN
    UNKNOWN = "unknown"         # not in the switch table; falls through to ObsBuilder


@dataclass(frozen=True)
class ObsRemapDecision:
    sensor: str
    produceability: ObsProduceability
    strategy: str          # human-readable explanation
    warning: Optional[str] = None   # None when no degradation


@dataclass
class ObsRemapReport:
    decisions: List[ObsRemapDecision] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_degradation(self) -> bool:
        return any(d.warning for d in self.decisions)


# ---------------------------------------------------------------------------
# Switch table — v1 hard-codes the small set of sensors that
# SkillManifest.required_sensors actually carries today. Per Q6 the
# registry refactor waits until ≥8 distinct types are in use.
# ---------------------------------------------------------------------------

# Sensors that mj_data exposes natively, regardless of field.
#
# v1.4 architectural correction: ``height_scan`` is now NATIVE because
# ObsBuilder._get_height_scan implements it as an mj_ray-based
# geometric ray-cast against whatever geometry the active field has
# (flat ground, stairs, hfield, prop obstacles — any of them work).
# Mission re-derives the obs from the env, never from the training
# scene. This is the user's stated requirement: "Mission inherits the
# trained behavior policy and applies it to whatever scene the user
# assigns".
_NATIVE_SENSORS = frozenset(
    {
        "imu",
        "joint_encoder",
        "joint_encoders",
        "joint_pos",
        "joint_vel",
        "joint_position",
        "joint_velocity",
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "actions",            # last-action loopback; available from any policy state
        "last_action",
        "height_scan",        # P5++: implemented via mj_ray + analytical plane fallback
        "heightfield_scan",   # alias
        "terrain_height",     # alias
    }
)

# Sensors that need explicit field-side configuration (none in v1.4
# now that height_scan is native — kept as a deliberate empty set so
# the classify() switch table stays explicit).
_FIELD_DEPENDENT_SENSORS = frozenset()

# Sensors that require ray-casting / rendering against scene geoms.
# P4 ships them as DEGRADED (zero-filled + WARN); P5 may upgrade some
# of them once richer fields exist.
_SCENE_DEPENDENT_SENSORS = frozenset(
    {
        "lidar",
        "depth_camera",
        "rgb_camera",
        "front_camera",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(sensor: str, field_obj: Any = None) -> ObsRemapDecision:
    """Classify a single required sensor.

    Parameters
    ----------
    sensor:
        The sensor name as it appears in
        ``SkillManifest.required_sensors``.
    field_obj:
        Optional :class:`MjField` so the adapter can up-grade
        ``height_scan`` from DEGRADED to NATIVE when the field is
        actually a heightfield. Pass ``None`` and the adapter assumes
        the worst case (flat ground).
    """
    name = (sensor or "").strip().lower()
    if not name:
        return ObsRemapDecision(
            sensor=sensor,
            produceability=ObsProduceability.UNKNOWN,
            strategy="empty sensor name; ObsBuilder will zero-fill",
            warning=f"obs_remap.empty_name",
        )

    if name in _NATIVE_SENSORS:
        return ObsRemapDecision(
            sensor=sensor,
            produceability=ObsProduceability.NATIVE,
            strategy="read directly from mj_data",
            warning=None,
        )

    if name in _FIELD_DEPENDENT_SENSORS:
        # Inspect the field if one was provided. P5 will introduce
        # heightfield-style fields; until then every field is flat and
        # this branch always degrades.
        base_terrain = ""
        if field_obj is not None:
            base_terrain = str(getattr(field_obj, "base_terrain", "") or "").lower()
        if base_terrain and base_terrain != "flat":
            return ObsRemapDecision(
                sensor=sensor,
                produceability=ObsProduceability.FIELD_DEPENDENT,
                strategy=f"sampled from field terrain ({base_terrain})",
                warning=None,
            )
        return ObsRemapDecision(
            sensor=sensor,
            produceability=ObsProduceability.DEGRADED,
            strategy="substituted with zero vector (field is flat ground)",
            warning=(
                f"obs_remap.degraded.{name}: field has no terrain; "
                "policy will see a zero vector for this obs and may behave "
                "out-of-distribution"
            ),
        )

    if name in _SCENE_DEPENDENT_SENSORS:
        return ObsRemapDecision(
            sensor=sensor,
            produceability=ObsProduceability.DEGRADED,
            strategy="substituted with zero vector (ray-cast not implemented in P4)",
            warning=(
                f"obs_remap.degraded.{name}: ray-cast / camera rendering "
                "not yet implemented in the Mission runtime; policy will "
                "see a zero vector for this obs"
            ),
        )

    return ObsRemapDecision(
        sensor=sensor,
        produceability=ObsProduceability.UNKNOWN,
        strategy="unknown sensor type; ObsBuilder will zero-fill",
        warning=f"obs_remap.unknown.{name}",
    )


def classify_required_sensors(
    required_sensors: List[str],
    field_obj: Any = None,
) -> ObsRemapReport:
    """Walk every required sensor and aggregate decisions + warnings.

    Returns an :class:`ObsRemapReport` whose ``warnings`` field is the
    list of warning strings (one per degraded sensor) that
    BehaviorNode forwards into ``RuntimeResult.diagnostics``.
    """
    report = ObsRemapReport()
    for sensor in required_sensors or []:
        decision = classify(sensor, field_obj=field_obj)
        report.decisions.append(decision)
        if decision.warning:
            report.warnings.append(decision.warning)
    return report
