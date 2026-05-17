"""Motion clip ↔ IR layer contract — static role tables.

Single source of truth for "which IR role does clip joint index N correspond
to, for format X".

Design rule::

    NO code that touches robot joints by name or index may bypass this
    contract. Every clip→robot mapping goes clip_index → IR role → body_name
    → joint name. Adding a new clip format = adding a new entry to
    FORMAT_IR_ROLES; no other files change.

Contracts:

  - ``amp_legged_gym`` — after ``AMPLegGymLoader(reorder_legs=True)``, the
    12-DoF quadruped joint array is in FL/FR/RL/RR × (hip, thigh, calf) order.
  - ``unitport_npy`` — legacy format; joint order matches the env's native
    ordering. Each clip carries its own assumed order (no cross-robot
    contract).

Migration note (RELEASE)
------------------------
DEMO's ``validate_clip_ir_match`` / ``validate_clips_ir_match`` validators
were tightly coupled to two not-yet-migrated runtime modules:

* ``src.system.training.body_ir.BodyIRMapper`` — the IR mapper class
  that turns a Robot canvas node's body_mapping into a queryable object.
* ``src.system.training.motion.loader.get_loader`` — the clip loader
  registry used to resolve a clip's joint count.

Both will land alongside the IR runtime (``application/engine/`` and
``registers/ir.py``). Until then, this module ships only the static
data layer + result dataclass — the validator functions will return
to this file once their dependencies exist in RELEASE. See plan §3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ════════════════════════════════════════════════════════════════════════════
# Per-format IR role tables
# ════════════════════════════════════════════════════════════════════════════

# 12-DoF quadruped, canonical order after AMPLegGymLoader reorder
# (FR/FL/RR/RL → FL/FR/RL/RR). Contract with loader._reorder_quad_legs.
QUADRUPED_AMP_IR_ROLES: List[str] = [
    "hip_FL",   "thigh_FL", "calve_FL",   # clip indices 0-2
    "hip_FR",   "thigh_FR", "calve_FR",   # clip indices 3-5
    "hip_RL",   "thigh_RL", "calve_RL",   # clip indices 6-8
    "hip_RR",   "thigh_RR", "calve_RR",   # clip indices 9-11
]

# Future: BIPED_AMP_IR_ROLES, HUMANOID_AMP_IR_ROLES, etc.

FORMAT_IR_ROLES: Dict[str, List[str]] = {
    "amp_legged_gym": QUADRUPED_AMP_IR_ROLES,
    # unitport_npy is intentionally absent — those clips carry no portable
    # joint-order contract and must be matched by the env directly.
}


# ════════════════════════════════════════════════════════════════════════════
# Validation result
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ClipIRStatus:
    """Result of validating one motion clip against a robot's IR mapping."""

    ok: bool                      # True iff every expected IR role resolves
    clip_name: str = ""
    format_id: str = ""
    joint_count: int = 0          # DoF present in the clip
    expected_roles: List[str] = field(default_factory=list)
    missing_roles: List[str] = field(default_factory=list)
    unresolved_roles: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def badge(self) -> str:
        """One-character indicator for UI (✔ / ❌ / — )."""
        if not self.expected_roles:
            return "—"
        return "✔" if self.ok else "❌"


# ════════════════════════════════════════════════════════════════════════════
# Pure data API
# ════════════════════════════════════════════════════════════════════════════


def get_ir_roles(format_id: str) -> List[str]:
    """Return the canonical IR role list for a clip format.

    Returns an empty list when the format has no portable contract
    (e.g. ``unitport_npy``). Callers must treat an empty result as
    "cannot validate, skip IR gate".
    """
    return list(FORMAT_IR_ROLES.get(format_id, []))


__all__ = [
    "QUADRUPED_AMP_IR_ROLES",
    "FORMAT_IR_ROLES",
    "ClipIRStatus",
    "get_ir_roles",
]
