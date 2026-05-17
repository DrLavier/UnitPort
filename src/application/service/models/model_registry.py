"""DEMO model_registry shim — RELEASE has ``registers.robots`` instead.

Why this file exists
--------------------

DEMO's ``src/system/skill/isaac_lab_manifest_parser.py`` (now ported to
``application.training.isaac_lab.manifest_parser``) calls into DEMO's
``src/system/models/model_registry.py`` for five operations:

    UnsupportedRobotForTrainingError    (exception class)
    iter_model_specs()                  -> ModelSpec iterator
    get_robot_joint_names(brand, mid)   -> List[str]
    get_ir_role_to_joint(brand, mid)    -> Dict[str, str]   (IR role → joint name)
    is_training_ready(brand, mid)       -> bool

RELEASE's authoritative robot catalog is ``registers/robots.py`` (single-
SKU keyed, ``RobotSpec`` typed). Porting DEMO's 740-line ``model_registry``
verbatim would duplicate the registers layer and bring along DEMO's
``brand_packages.<brand>.joint_mappings`` import path that does not exist
in RELEASE.

Instead, this module is a **thin forwarding shim** that translates
``(brand, model_id)`` ↔ SKU ``f"{brand}.{model_id}"`` and forwards to
``registers.robots`` API. Behaviour is semantically equivalent for the
five operations manifest_parser uses.

Boundary rule (CLAUDE.md §1.1 multi-brand inclusiveness): no brand strings
are hardcoded here — every lookup goes through registers.robots, which
itself is brand-agnostic.

Boundary rule (Phase 3 plan §B6): this module sits in
``application.service.models`` not ``application.training.*`` because
``robots`` is the canonical model catalog (used by SB3 chain too, via
``current_project_info`` / ProjectStore-adjacent helpers); routing the
shim through service/models keeps the IL chain from importing
``registers.robots`` directly under a non-IL namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from registers import robots as _robots


class UnsupportedRobotForTrainingError(RuntimeError):
    """Raised when (brand, model_id) cannot satisfy the IL bundle contract.

    Same semantic as DEMO's ``model_registry.UnsupportedRobotForTrainingError``:
    the (brand, model_id) does not resolve to any registered SKU, OR the
    resolved RobotSpec lacks joint_order / IR role mapping required by the
    Isaac Lab manifest parser.
    """


# ---------------------------------------------------------------------------
# Internal helpers — SKU is the registers-side primary key.
# ---------------------------------------------------------------------------

def _sku_of(brand: str, model_id: str) -> str:
    """``(brand, model_id)`` → ``"<brand>.<model_id>"`` SKU.

    Mirrors :func:`registers.robots.build_sku` (we don't import it to keep
    the shim's dependency surface minimal).
    """
    b = str(brand or "").strip().lower()
    m = str(model_id or "").strip().lower()
    return f"{b}.{m}" if b and m else (b or m)


def _split_sku(sku: str) -> Tuple[str, str]:
    """``"<brand>.<model_id>"`` → ``(brand, model_id)``.

    Falls back to ``("", sku)`` when the SKU does not contain a dot — keeps
    the shim from raising on legacy entries; manifest_parser will treat the
    empty brand as "unknown" and surface its own error.
    """
    raw = str(sku or "").strip()
    if "." not in raw:
        return ("", raw)
    brand, model = raw.split(".", 1)
    return (brand, model)


# ---------------------------------------------------------------------------
# DEMO ModelSpec compatibility — we expose the two attributes manifest_parser
# reads (``brand_id`` and ``model_id``). Everything else DEMO's ModelSpec
# carries is unused by manifest_parser's USD-path heuristic at line ~388.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Minimal subset of DEMO ``ModelSpec`` exposing only what manifest_parser
    needs (``brand_id`` + ``model_id``)."""
    brand_id: str
    model_id: str


def iter_model_specs(*, brand: str = "") -> Iterable[ModelSpec]:
    """Iterate every (brand, model_id) pair registered in RELEASE.

    Optional ``brand`` filter keeps the API drop-in compatible with DEMO's
    signature; passing the empty string yields every robot.
    """
    brand_filter = str(brand or "").strip().lower()
    for sku in _robots.list_skus():
        b, m = _split_sku(sku)
        if not m:
            continue
        if brand_filter and b != brand_filter:
            continue
        yield ModelSpec(brand_id=b, model_id=m)


def get_robot_joint_names(brand: str, model_id: str) -> Optional[List[str]]:
    """Canonical runtime joint order for ``(brand, model_id)``.

    Forwards to :class:`registers.robots.RobotSpec.joint_order`. Returns
    ``None`` (matching DEMO contract) when the SKU is unknown — caller
    is responsible for raising :exc:`UnsupportedRobotForTrainingError`.
    """
    sku = _sku_of(brand, model_id)
    spec = _robots.get_robot_spec(sku)
    if spec is None:
        return None
    return list(spec.joint_order)


def get_ir_role_to_joint(brand: str, model_id: str) -> Optional[Dict[str, str]]:
    """Canonical IR role → runtime joint name mapping for ``(brand, model_id)``.

    DEMO returns ``{ir_role: joint_name}``. RELEASE's RobotSpec stores
    ``body_role_map = {raw_joint_name: ir_role}`` (the inverse), so we
    invert. ``None`` when the SKU is unknown or has no IR roles declared.
    """
    sku = _sku_of(brand, model_id)
    spec = _robots.get_robot_spec(sku)
    if spec is None or not spec.body_role_map:
        return None
    inverted: Dict[str, str] = {}
    for raw_joint, ir_role in spec.body_role_map.items():
        if not ir_role:
            continue
        # Last-write-wins on duplicate IR roles — same convention as
        # DEMO when JOINT_ORDER + IR_ROLE_TO_JOINT have ambiguous fan-in.
        inverted[ir_role] = raw_joint
    return inverted or None


def is_training_ready(brand: str, model_id: str) -> bool:
    """True iff ``(brand, model_id)`` has both joint_order and an IR role
    mapping with the value set being a subset of joint_order.

    Mirrors DEMO :func:`is_training_ready`; manifest_parser uses this as
    a hard gate before walking observation / action spaces.
    """
    joints = get_robot_joint_names(brand, model_id)
    mapping = get_ir_role_to_joint(brand, model_id)
    if not joints or not mapping:
        return False
    joint_set = set(joints)
    return all(v in joint_set for v in mapping.values())


__all__ = [
    "ModelSpec",
    "UnsupportedRobotForTrainingError",
    "get_ir_role_to_joint",
    "get_robot_joint_names",
    "is_training_ready",
    "iter_model_specs",
]
