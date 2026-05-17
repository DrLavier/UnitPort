"""Body-name IR layer — canonical-catalog driven, brand-neutral.

Each robot body is mapped onto a slot in a fixed canonical IR catalog
(``registers/data/ir_canonical.json``). The catalog is keyed by morphology
family (quadruped, biped, manipulator, …) and NEVER auto-extends from a
robot's URDF/USDA — that would let one robot's vendor-specific naming
pollute the IR layer. Bodies the keyword tables can't auto-route to a
canonical slot land in ``unmapped_bodies`` for the user to resolve via
the Robot node UI (assign to a canonical role, or mark explicitly
out-of-scope).

Keyword tables (``PART_KEYWORDS`` / ``POSITION_KEYWORDS``) only drive
the *suggestion* step. They never define what counts as a valid IR
role — that authority lives in the catalog.

Ported from DEMO/src/system/training/body_ir.py with:
- IR catalog import rewired to ``registers.ir`` (RELEASE's frozen catalog)
- ``from_robot_asset`` adapted to RELEASE's :class:`RobotAsset` shape
  (``joints: dict[name, ir_role]`` + ``families: list[str]``)
- ``detect_family`` reads ``asset.families[0]`` directly (RELEASE canonical
  entries authoritatively declare the family — no detection heuristic needed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from registers import ir as _ir_registry


# ═══════════════════════════════════════════════════════════════════════
# Catalog adapters — bridge registers.ir's dict-based API to the IRRole
# dataclass expected by the BodyIRMapper internals (which were originally
# written against DEMO's ir_role_catalog dataclass API).
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalRole:
    """Local mirror of one canonical role row from registers.ir.

    Mirrors DEMO ir_role_catalog.IRRole (role_id / category / label /
    position / required) so the mapper internals stay verbatim.
    """
    role_id: str
    category: str
    label: str
    position: str
    required: bool = True


def _adapt_role_dict(d: Dict[str, Any]) -> CanonicalRole:
    return CanonicalRole(
        role_id=str(d.get("id", "")),
        category=str(d.get("category", "")),
        label=str(d.get("label", "")),
        position=str(d.get("position", "")),
        required=bool(d.get("required", True)),
    )


def get_canonical_roles(family: str) -> List[CanonicalRole]:
    return [_adapt_role_dict(d) for d in _ir_registry.list_roles(family)]


def get_role(family: str, role_id: str) -> Optional[CanonicalRole]:
    d = _ir_registry.get_role(family, role_id)
    return _adapt_role_dict(d) if d is not None else None


def list_known_families() -> Set[str]:
    return set(_ir_registry.list_families())


# IR-canonical categories that map to actuated joints (vs kinematic body
# links). Filters get_canonical_roles() down to the IR roles a user can
# actuate via init_joint_angles / action_joint_names_expr.
_JOINT_CATEGORIES_BY_FAMILY: Dict[str, FrozenSet[str]] = {
    "quadruped":   frozenset({"hips", "thighs", "calves"}),
    "biped":       frozenset({"hips", "knees", "shoulders", "elbows", "wrists"}),
    "humanoid":    frozenset({"hips", "knees", "shoulders", "elbows", "wrists"}),
    "manipulator": frozenset({"shoulders", "elbows", "wrists"}),
    "wheeled":     frozenset(),
    "generic":     frozenset(),
}


def get_joint_ir_roles(family: str) -> List[str]:
    """Return the family's canonical IR roles that correspond to actuated joints.

    Used by canvas UI (JointInitTableRow dropdown, validate()) to enumerate
    the legal IR role names a user can name in ``init_joint_angles`` /
    ``action_joint_names_expr`` for a robot of *family*. Bodies (base, feet,
    head, ...) are filtered out — they are kinematic links, not joints.

    Unknown family → empty list (caller decides whether to raise).
    """
    joint_cats = _JOINT_CATEGORIES_BY_FAMILY.get(family, frozenset())
    if not joint_cats:
        return []
    return [
        r.role_id for r in get_canonical_roles(family)
        if r.category in joint_cats
    ]


def detect_family(asset: Any) -> str:
    """Return the morphology family for ``asset`` (RobotAsset).

    RELEASE canonical entries authoritatively declare the family list
    (``asset.families``). No URDF heuristic is needed — if the user
    imported a robot with no family field the canonical entry is
    incomplete and we fall back to ``generic`` (catalog will then have
    only ``base``, every other body lands in ``unmapped_bodies``).
    """
    families = list(getattr(asset, "families", []) or [])
    if families:
        first = str(families[0])
        if first in list_known_families():
            return first
    return "generic"


# ═══════════════════════════════════════════════════════════════════════
# Keyword tables  (suggestion-only — NOT a role registry)
# ═══════════════════════════════════════════════════════════════════════

# Part keywords → category name. Used to *suggest* which canonical role
# slot a body should fill. A miss here does NOT create a new IR role —
# the body is surfaced as unmapped instead.
PART_KEYWORDS: Dict[str, str] = {
    "foot":     "feet",
    "toe":      "feet",
    "ankle":    "feet",
    "sole":     "feet",
    "wheel":    "feet",
    "thigh":    "thighs",
    "upper_leg": "thighs",
    "hip":      "hips",
    "calf":     "calves",
    "shank":    "calves",
    "lower_leg": "calves",
    "knee":     "knees",
    "shoulder": "shoulders",
    "upper_arm": "shoulders",
    "elbow":    "elbows",
    "forearm":  "elbows",
    "wrist":    "wrists",
    "hand":     "hands",
    "finger":   "fingers",
    "waist":    "waist",
    "torso":    "base",
    "trunk":    "base",
    "base":     "base",
    "pelvis":   "base",
    "body":     "base",
}

PART_TOKEN_KEYWORDS: Dict[str, str] = {
    "hx":         "hips",
    "haa":        "hips",
    "abad":       "hips",
    "abduction":  "hips",
    "ad":         "hips",
    "hy":         "thighs",
    "hfe":        "thighs",
    "flexion":    "thighs",
    "kn":         "knees",
    "kfe":        "knees",
}

POSITION_KEYWORDS: Dict[str, str] = {
    "fl": "FL", "fr": "FR", "rl": "RL", "rr": "RR",
    "lf": "FL", "rf": "FR",
    "lb": "RL", "rb": "RR",
    "lh": "RL", "rh": "RR",
    "hl": "RL", "hr": "RR",
    "left": "L", "right": "R",
    "l": "L", "r": "R",
    "front": "F", "rear": "R_",
    "back": "R_", "hind": "R_",
}

_FRONT_TOKENS: FrozenSet[str] = frozenset({"front", "fore"})
_REAR_TOKENS: FrozenSet[str] = frozenset({"rear", "back", "hind"})
_LEFT_TOKENS: FrozenSet[str] = frozenset({"left"})
_RIGHT_TOKENS: FrozenSet[str] = frozenset({"right"})

AUTO_FROM_ASSET_CATEGORIES: FrozenSet[str] = frozenset({"base", "feet"})


# ═══════════════════════════════════════════════════════════════════════
# Tokeniser
# ═══════════════════════════════════════════════════════════════════════

def joint_name_to_link_name(joint_name: str) -> str:
    """Derive the likely USD link/body name from a joint name."""
    for suffix in ("_joint", "_Joint", "_actuator", "_motor"):
        if joint_name.endswith(suffix):
            return joint_name[: -len(suffix)]
    return joint_name


_GLUED_PREFIXES: FrozenSet[str] = frozenset({
    "fl", "fr", "rl", "rr",
    "lf", "rf", "lb", "rb",
    "lh", "rh", "hl", "hr",
})


def tokenize(name: str) -> List[str]:
    """Split a body name into lowercase tokens.

    ``"FL_thigh_joint"`` → ``["fl", "thigh", "joint"]``
    ``"LeftHip"``        → ``["left", "hip"]``
    ``"lffoot"``         → ``["lf", "foot"]``  (glued prefix split)
    """
    known_part_fragments = (
        set(PART_KEYWORDS.keys()) | set(PART_TOKEN_KEYWORDS.keys())
    )
    parts = re.split(r"[_\-\.]+", name)
    tokens: List[str] = []
    for p in parts:
        sub = re.sub(r"([a-z])([A-Z])", r"\1_\2", p).split("_")
        for s in sub:
            if not s:
                continue
            sl = s.lower()
            if (len(sl) > 2
                    and sl[:2] in _GLUED_PREFIXES
                    and sl[2:] in known_part_fragments):
                tokens.append(sl[:2])
                tokens.append(sl[2:])
            else:
                tokens.append(sl)
    return tokens


def detect_part(tokens: List[str]) -> Optional[str]:
    """Return the category name from part keywords, or None."""
    for t in tokens:
        if t in PART_TOKEN_KEYWORDS:
            return PART_TOKEN_KEYWORDS[t]
    joined = "_".join(tokens)
    for kw, cat in PART_KEYWORDS.items():
        if kw in joined:
            return cat
    for t in tokens:
        if t in PART_KEYWORDS:
            return PART_KEYWORDS[t]
    return None


def detect_position(tokens: List[str]) -> str:
    """Return a normalised position tag (FL, FR, RL, RR, L, R, …) or ''."""
    has_front = any(t in _FRONT_TOKENS for t in tokens)
    has_rear  = any(t in _REAR_TOKENS  for t in tokens)
    has_left  = any(t in _LEFT_TOKENS  for t in tokens)
    has_right = any(t in _RIGHT_TOKENS for t in tokens)

    if has_front and has_left:
        return "FL"
    if has_front and has_right:
        return "FR"
    if has_rear and has_left:
        return "RL"
    if has_rear and has_right:
        return "RR"

    for t in tokens:
        if t in POSITION_KEYWORDS:
            return POSITION_KEYWORDS[t]
    return ""


_CATEGORY_TO_ROLE_PREFIX: Dict[str, str] = {
    "hips":      "hip",
    "thighs":    "thigh",
    "calves":    "calf",
    "knees":     "knee",
    "feet":      "foot",
    "shoulders": "shoulder",
    "elbows":    "elbow",
    "wrists":    "wrist",
    "hands":     "hand",
    "fingers":   "finger",
    "waist":     "waist",
    "base":      "base",
}


_SEGMENT_JOINT_RE = re.compile(
    r"^(?P<a>[a-z]+?)(?P<na>\d+)_(?P<b>[a-z]+?)(?P<nb>\d+)$"
)
_BASE_TO_SEG_RE = re.compile(r"^base_(?P<pfx>[a-z]+?)(?P<n>\d+)$")

_SEGMENT_TO_CATEGORY: Dict[Tuple[str, str], str] = {
    ("1", "2"): "thighs",
    ("2", "3"): "calves",
}


def detect_segment_pair_role(link_name: str) -> Optional[Tuple[str, str]]:
    """Recognise CHAMP-style segment-numbered URDF joint names.

    Returns ``(category, position_tag)`` or ``None``.
    """
    name = link_name.lower()

    m = _BASE_TO_SEG_RE.match(name)
    if m:
        pos = POSITION_KEYWORDS.get(m.group("pfx"), "")
        if pos and m.group("n") == "1":
            return ("hips", pos)

    m = _SEGMENT_JOINT_RE.match(name)
    if m:
        pfx_a, pfx_b = m.group("a"), m.group("b")
        if pfx_a == pfx_b:
            pos = POSITION_KEYWORDS.get(pfx_a, "")
            cat = _SEGMENT_TO_CATEGORY.get((m.group("na"), m.group("nb")))
            if pos and cat:
                return (cat, pos)

    return None


_FAMILY_ROLE_ALIASES: Dict[str, Dict[str, str]] = {
    "quadruped": {
        f"knee_{p}": f"calf_{p}" for p in ("FL", "FR", "RL", "RR")
    },
}


def _suggest_role_id(link_name: str) -> str:
    """Compute the canonical role_id a link name *might* fill."""
    pair = detect_segment_pair_role(link_name)
    if pair is not None:
        cat, pos = pair
        prefix = _CATEGORY_TO_ROLE_PREFIX.get(
            cat,
            cat.rstrip("s") if cat.endswith("s") else cat,
        )
        return f"{prefix}_{pos}" if pos else prefix

    tokens = tokenize(link_name)
    cat = detect_part(tokens) or ""
    pos = detect_position(tokens)
    if not cat:
        return ""
    if cat == "base":
        return "base"
    prefix = _CATEGORY_TO_ROLE_PREFIX.get(
        cat,
        cat.rstrip("s") if cat.endswith("s") else cat,
    )
    return f"{prefix}_{pos}" if pos else prefix


# ═══════════════════════════════════════════════════════════════════════
# IR Role  (one slot in the canonical catalog, possibly unfilled)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IRRole:
    """One IR role slot. ``body`` is the link name it resolves to (or None)."""
    role_id: str
    category: str
    label: str
    position: str
    auto_from_asset: bool = False
    body: Optional[str] = None
    manual: bool = False

    @property
    def resolved(self) -> bool:
        return self.body is not None


# ═══════════════════════════════════════════════════════════════════════
# Mapper
# ═══════════════════════════════════════════════════════════════════════

class BodyIRMapper:
    """Robot bodies → canonical IR role slots.

    Construction never invents new IR roles. Every entry in ``roles``
    comes from the canonical catalog for the requested family. Bodies
    that don't keyword-match any catalog slot land in
    ``unmapped_bodies`` until the user resolves them via the Robot node
    validator (assign to a canonical role, or mark out-of-scope).
    """

    def __init__(self, body_names: List[str], family: str = "generic"):
        self._body_names = list(body_names)
        # Normalise unknown families to ``generic`` so the validator UI
        # can still render *something* for novel morphologies.
        self._family = family if family in list_known_families() else "generic"
        self._roles: List[IRRole] = []
        self._by_id: Dict[str, IRRole] = {}
        self._unmapped_bodies: List[str] = []
        self._out_of_scope: Set[str] = set()

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_robot_asset(cls, asset: Any) -> "BodyIRMapper":
        """Build from a RELEASE :class:`RobotAsset`.

        Reads ``asset.joints`` (dict[name, ir_role]) for joint names and
        ``asset.families`` for morphology family. Foot links are inferred
        from calf joint names (``FL_calf_joint`` → ``FL_foot``) since the
        canonical entry only declares actuated joints, not contact bodies.
        """
        joints_map = dict(getattr(asset, "joints", {}) or {})
        joint_names = list(joints_map.keys())

        bodies: List[str] = []

        # Joint-derived link names (FL_hip_joint → FL_hip etc.)
        for jn in joint_names:
            ln = joint_name_to_link_name(jn)
            if ln not in bodies:
                bodies.append(ln)

        # Foot link inference: every calf/shank/ankle joint implies a foot
        # body sharing the same position prefix.
        foot_links_inferred: List[str] = []
        for jn in joint_names:
            ln = joint_name_to_link_name(jn)
            if "calf" in ln.lower() or "shank" in ln.lower() or "ankle" in ln.lower():
                candidate = re.sub(r"(?i)calf|shank|ankle", "foot", ln)
                if candidate != ln and candidate not in bodies:
                    bodies.append(candidate)
                    foot_links_inferred.append(candidate)

        family = detect_family(asset)
        mapper = cls(bodies, family=family)

        # Seed roles from the canonical IR mapping that the registers entry
        # already provides for joints. This is authoritative — no keyword
        # guessing for joint-derived bodies.
        for jn, ir_role in joints_map.items():
            if not ir_role:
                continue
            ln = joint_name_to_link_name(jn)
            mapper._roles  # ensure present (filled below)
            # Stash the (body, role_id) — applied after _build_roles_from_bodies.

        mapper._build_roles_from_bodies()

        # Override auto-suggestions with the canonical joint→ir_role map.
        # This is the RELEASE-specific path: the canonical entry asserts
        # which IR role each joint occupies, so we trust it over the
        # keyword tokenizer.
        for jn, ir_role in joints_map.items():
            if not ir_role:
                continue
            slot = mapper._by_id.get(ir_role)
            if slot is None:
                continue
            ln = joint_name_to_link_name(jn)
            if slot.body is not None and slot.body != ln:
                mapper._send_to_unmapped(slot.body)
            slot.body = ln
            mapper._unmark_unmapped(ln)

        # Foot-from-calf inference: assign each inferred foot body to the
        # matching foot_<POS> slot (auto_from_asset = True so UI hides it
        # from the validator).
        for fl in foot_links_inferred:
            slot = next((r for r in mapper._roles if r.body == fl), None)
            if slot is not None:
                if slot.category == "feet":
                    slot.auto_from_asset = True
                continue
            pos = detect_position(tokenize(fl))
            target = next(
                (r for r in mapper._roles
                 if r.category == "feet" and r.body is None and r.position == pos),
                None,
            ) or next(
                (r for r in mapper._roles
                 if r.category == "feet" and r.body is None),
                None,
            )
            if target is not None:
                target.body = fl
                target.auto_from_asset = True
                mapper._unmark_unmapped(fl)

        return mapper

    @classmethod
    def from_body_list(cls, body_names: List[str],
                       family: str = "quadruped") -> "BodyIRMapper":
        mapper = cls(body_names, family=family)
        mapper._build_roles_from_bodies()
        return mapper

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyIRMapper":
        body_names = list(d.get("body_names", []))
        family = str(d.get("family") or "quadruped")

        has_feet_hint = any(
            ("foot" in bn.lower() or "toe" in bn.lower() or "ankle" in bn.lower())
            for bn in body_names
        )
        if not has_feet_hint:
            for bn in list(body_names):
                ln = joint_name_to_link_name(bn)
                if "calf" in ln.lower() or "shank" in ln.lower():
                    cand = re.sub(r"(?i)calf|shank", "foot", ln)
                    if cand != ln and cand not in body_names:
                        body_names.append(cand)

        mapper = cls(body_names, family=family)
        for oos in d.get("out_of_scope_bodies", []) or []:
            mapper._out_of_scope.add(str(oos))

        mapper._build_roles_from_bodies()

        roles_data = d.get("roles", {}) or {}
        for rid, rd in roles_data.items():
            if not isinstance(rd, dict):
                continue
            body = rd.get("body")
            if not body:
                continue
            canonical = mapper._by_id.get(rid)
            if canonical is None:
                if not mapper._is_assigned(body) and body not in mapper._out_of_scope:
                    mapper._mark_unmapped(body)
                continue
            if canonical.body is not None and canonical.body != body:
                mapper._send_to_unmapped(canonical.body)
            canonical.body = body
            canonical.manual = bool(rd.get("manual", False))
            canonical.auto_from_asset = bool(
                rd.get("auto_from_asset", canonical.auto_from_asset)
            )
            mapper._unmark_unmapped(body)

        for um in d.get("unmapped_bodies", []) or []:
            um_str = str(um)
            if (not mapper._is_assigned(um_str)
                    and um_str not in mapper._out_of_scope):
                mapper._mark_unmapped(um_str)

        return mapper

    # ── Core: build roles from the canonical catalog ──────────────────

    def _build_roles_from_bodies(self) -> None:
        """Seed ``_roles`` from the canonical catalog and auto-suggest
        body→role assignments via keyword matching.
        """
        self._roles.clear()
        self._by_id.clear()
        self._unmapped_bodies.clear()

        # 1. Seed catalog slots (all empty initially)
        for canon in get_canonical_roles(self._family):
            role = IRRole(
                role_id=canon.role_id,
                category=canon.category,
                label=canon.label,
                position=canon.position,
                auto_from_asset=False,
                body=None,
            )
            self._roles.append(role)
            self._by_id[canon.role_id] = role

        # 2. Auto-suggest assignments via keyword matching
        for raw_name in self._body_names:
            link_name = joint_name_to_link_name(raw_name)
            if link_name in self._out_of_scope:
                continue
            if self._is_assigned(link_name):
                continue
            role_id = _suggest_role_id(link_name)
            if role_id and role_id not in self._by_id:
                role_id = _FAMILY_ROLE_ALIASES.get(
                    self._family, {}
                ).get(role_id, role_id)
            slot = self._by_id.get(role_id) if role_id else None
            if slot is not None and slot.body is None:
                slot.body = link_name
                slot.auto_from_asset = (
                    slot.category in AUTO_FROM_ASSET_CATEGORIES
                )
            else:
                self._mark_unmapped(link_name)

    # ── Internal helpers ──────────────────────────────────────────────

    def _is_assigned(self, body: str) -> bool:
        return any(r.body == body for r in self._roles)

    def _mark_unmapped(self, body: str) -> None:
        if body and body not in self._unmapped_bodies:
            self._unmapped_bodies.append(body)

    def _unmark_unmapped(self, body: str) -> None:
        if body in self._unmapped_bodies:
            self._unmapped_bodies.remove(body)

    def _send_to_unmapped(self, body: str) -> None:
        for r in self._roles:
            if r.body == body:
                r.body = None
                r.manual = False
                r.auto_from_asset = False
        if body not in self._out_of_scope:
            self._mark_unmapped(body)

    # ── Public API ────────────────────────────────────────────────────

    def override(self, role_id: str, body: Optional[str]) -> None:
        if body is None or body == "":
            r = self._by_id.get(role_id)
            if r and r.body is not None:
                prev = r.body
                r.body = None
                r.manual = True
                if (prev not in self._out_of_scope
                        and not self._is_assigned(prev)):
                    self._mark_unmapped(prev)
            return
        self.reassign_role(body, role_id)

    def reassign_role(self, body_link: str,
                      new_role_id: Optional[str]) -> None:
        if not body_link:
            return
        for r in self._roles:
            if r.body == body_link:
                r.body = None
                r.manual = False
                r.auto_from_asset = False
        self._unmark_unmapped(body_link)
        self._out_of_scope.discard(body_link)

        if not new_role_id:
            self._mark_unmapped(body_link)
            return

        role = self._by_id.get(new_role_id)
        if role is None:
            self._mark_unmapped(body_link)
            return

        if role.body is not None and role.body != body_link:
            self._send_to_unmapped(role.body)

        role.body = body_link
        role.manual = True
        role.auto_from_asset = False

    def mark_out_of_scope(self, body_link: str) -> None:
        if not body_link:
            return
        for r in self._roles:
            if r.body == body_link:
                r.body = None
                r.manual = False
                r.auto_from_asset = False
        self._unmark_unmapped(body_link)
        self._out_of_scope.add(body_link)

    def clear_out_of_scope(self, body_link: str) -> None:
        if body_link in self._out_of_scope:
            self._out_of_scope.discard(body_link)
            self._mark_unmapped(body_link)

    def get(self, role_id: str) -> Optional[IRRole]:
        return self._by_id.get(role_id)

    def get_category_bodies(self, category: str) -> List[str]:
        return [r.body for r in self._roles
                if r.category == category and r.body is not None]

    def all_resolved(self, required_only: bool = True) -> bool:
        for r in self._roles:
            if r.resolved:
                continue
            if required_only:
                canon = get_role(self._family, r.role_id)
                if canon is not None and not canon.required:
                    continue
            return False
        return True

    def unresolved_roles(self, required_only: bool = True) -> List[str]:
        out: List[str] = []
        for r in self._roles:
            if r.resolved:
                continue
            if required_only:
                canon = get_role(self._family, r.role_id)
                if canon is not None and not canon.required:
                    continue
            out.append(r.role_id)
        return out

    def unmapped_bodies(self) -> List[str]:
        return list(self._unmapped_bodies)

    def out_of_scope_bodies(self) -> List[str]:
        return sorted(self._out_of_scope)

    @property
    def roles(self) -> List[IRRole]:
        return list(self._roles)

    @property
    def body_names(self) -> List[str]:
        return list(self._body_names)

    @property
    def family(self) -> str:
        return self._family

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_names": list(self._body_names),
            "family": self._family,
            "roles": {
                r.role_id: {
                    "body": r.body,
                    "manual": r.manual,
                    "auto_from_asset": r.auto_from_asset,
                }
                for r in self._roles
            },
            "unmapped_bodies": list(self._unmapped_bodies),
            "out_of_scope_bodies": sorted(self._out_of_scope),
        }

    def categories_present(self) -> Set[str]:
        return {r.category for r in self._roles
                if r.category and r.body is not None}


# ═══════════════════════════════════════════════════════════════════════
# User-override extract / apply
# ═══════════════════════════════════════════════════════════════════════

def extract_user_overrides(mapper: BodyIRMapper) -> Dict[str, Any]:
    """Capture only the deliberate edits the user made on top of auto-detection."""
    manual_roles = {
        r.role_id: r.body
        for r in mapper.roles
        if r.manual and r.body is not None
    }
    out_of_scope = list(mapper.out_of_scope_bodies())
    if not manual_roles and not out_of_scope:
        return {}
    payload: Dict[str, Any] = {}
    if manual_roles:
        payload["manual_roles"] = manual_roles
    if out_of_scope:
        payload["out_of_scope"] = out_of_scope
    return payload


def apply_user_overrides(mapper: BodyIRMapper,
                         overrides: Optional[Dict[str, Any]]) -> None:
    """Replay :func:`extract_user_overrides` output onto a fresh mapper."""
    if not overrides:
        return
    valid_bodies = set()
    for raw in mapper.body_names:
        valid_bodies.add(raw)
        valid_bodies.add(joint_name_to_link_name(raw))

    for body in (overrides.get("out_of_scope") or []):
        if body in valid_bodies:
            mapper.mark_out_of_scope(str(body))

    manual_roles = overrides.get("manual_roles") or {}
    if isinstance(manual_roles, dict):
        for role_id, body in manual_roles.items():
            if body and body in valid_bodies:
                mapper.reassign_role(str(body), str(role_id))


# ═══════════════════════════════════════════════════════════════════════
# Reward params helper
# ═══════════════════════════════════════════════════════════════════════

def resolve_body_params(il_params_template: str,
                        mapper: BodyIRMapper) -> str:
    """Substitute ``{ir:category}`` placeholders.

    ``{ir:feet}`` → list of all bodies in the "feet" category.
    ``{ir:thighs_hips_base}`` → merged list from thighs + hips + base.
    """
    def _replace(match: re.Match) -> str:
        expr = match.group(1)
        categories = split_compound_categories(expr)
        bodies: List[str] = []
        for cat in categories:
            bodies.extend(mapper.get_category_bodies(cat))
        if not bodies:
            return f'"(UNRESOLVED:{expr})"'
        if len(bodies) == 1:
            return f'"{bodies[0]}"'
        return "[" + ", ".join(f'"{b}"' for b in bodies) + "]"

    return re.sub(r"\{ir:(\w+)\}", _replace, il_params_template)


def split_compound_categories(expr: str) -> List[str]:
    """Split ``"thighs_hips_base"`` → ``["thighs", "hips", "base"]``."""
    known = set(PART_KEYWORDS.values())
    known_sorted = sorted(known, key=len, reverse=True)
    cats: List[str] = []
    rem = expr
    while rem:
        matched = False
        for cat in known_sorted:
            if rem.startswith(cat):
                cats.append(cat)
                rem = rem[len(cat):]
                if rem.startswith("_"):
                    rem = rem[1:]
                matched = True
                break
        if not matched:
            idx = rem.find("_")
            if idx > 0:
                cats.append(rem[:idx])
                rem = rem[idx + 1:]
            else:
                cats.append(rem)
                break
    return cats


__all__ = [
    "BodyIRMapper",
    "IRRole",
    "CanonicalRole",
    "detect_family",
    "get_canonical_roles",
    "get_role",
    "joint_name_to_link_name",
    "tokenize",
    "extract_user_overrides",
    "apply_user_overrides",
    "resolve_body_params",
    "split_compound_categories",
]
