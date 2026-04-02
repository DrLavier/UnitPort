#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocomotionProfile — per-brand/model locomotion capability data (Step 6).

Architectural Decisions (from DEVELOP_TODO.txt)
------------------------------------------------
- Decision 1: Humanoid robots (g1, h1, h1_2) use independent walking semantics,
  NOT quadruped trot gait. Profile must reflect humanoid walking parameters.
- Decision 4: SDK-only path is NOT acceptance criteria. Profile registration
  is required ONLY for MuJoCo runtime support. SDK-only models (a1, b1) should
  NOT be registered here unless MuJoCo support is added later.

Used by ActionBindings to convert normalised IR parameters into
backend-ready values without hardcoding any model-specific numbers inline.

Profile lookup
--------------
    profile = get_locomotion_profile(brand, robot_type)
    if profile is None:
        # Unknown model — caller must use a safe fallback or emit a diagnostic.
        # Per Decision 4: this indicates NO MuJoCo support.

Built-in profiles
-----------------
    ("unitree", "go2")  — go2 quadruped, full velocity channel, MuJoCo supported

NOT registered (SDK-only, Decision 4)
--------------------------------------
    ("unitree", "a1")  — no MuJoCo model, SDK-only
    ("unitree", "b1")  — no MuJoCo model, SDK-only
    ("unitree", "go2w") — no MuJoCo support (Decision 2)
    ("unitree", "b2w")  — no velocity_move_mujoco implementation

Adding new models
-----------------
Register an entry in ``_PROFILES`` only.  No binding or IR code changes needed.
NOTE: Only register models with MuJoCo support. Per Decision 4, SDK-only models
should NOT be registered unless MuJoCo is added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.system.models.model_registry import iter_model_specs

# ---------------------------------------------------------------------------
# LocomotionProfile dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocomotionProfile:
    """Locomotion capability descriptor for a specific brand/model.

    Attributes
    ----------
    max_forward_vx:
        Maximum forward linear velocity in m/s.  Bindings multiply the
        normalised ``speed`` IR parameter by this value to obtain ``vx``.
    supports_lateral_vy:
        True when the model can move laterally (non-zero ``vy``).
    supports_yaw_rate:
        True when the model supports in-place rotation (non-zero ``vyaw``).
    preferred_backend_ops:
        Ordered tuple of preferred backend operation names.  The first entry
        is the primary operation for forward locomotion.
    gait_type:
        Locomotion semantics for this model. Used to distinguish quadruped
        trot gait from humanoid walking (Decision 1).
        - "trot": quadruped trot gait (go2, b2)
        - "walk": humanoid walking (g1, h1, h1_2)
    robot_category:
        Robot morphology category for UI filtering.
        - "quadruped": four-legged robot
        - "humanoid": bipedal robot
    """

    max_forward_vx: float
    supports_lateral_vy: bool
    supports_yaw_rate: bool
    preferred_backend_ops: Tuple[str, ...]
    gait_type: str = "trot"  # "trot" | "walk"
    robot_category: str = "quadruped"  # "quadruped" | "humanoid"


# ---------------------------------------------------------------------------
# Built-in profile registry
# ---------------------------------------------------------------------------

_PROFILES: dict = {
    (spec.brand_id, spec.model_id): LocomotionProfile(
        max_forward_vx=float(spec.forward.max_forward_vx or 0.0),
        supports_lateral_vy=(spec.forward.robot_category != "humanoid"),
        supports_yaw_rate=True,
        preferred_backend_ops=("velocity_move",),
        gait_type=str(spec.forward.gait_type or "trot"),
        robot_category=str(spec.forward.robot_category or "quadruped"),
    )
    for spec in iter_model_specs()
    if spec.forward is not None
    and spec.forward.max_forward_vx is not None
    and spec.forward.gait_type is not None
    and spec.forward.robot_category is not None
    and spec.mujoco_supported
}


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

def get_locomotion_profile(
    brand: str,
    robot_type: str,
) -> Optional[LocomotionProfile]:
    """Return the locomotion profile for the given brand/model, or ``None``.

    Lookup is case-insensitive.  Returns ``None`` for unknown brand/model
    combinations; callers must handle the missing-profile case gracefully
    (use a safe default or emit a diagnostic — do not raise).
    """
    key = (str(brand or "").strip().lower(), str(robot_type or "").strip().lower())
    return _PROFILES.get(key)
