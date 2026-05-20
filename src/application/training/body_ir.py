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
    "quadruped":   frozenset({"hips", "thighs", "calves", "sensors", "misc"}),
    "biped":       frozenset({"hips", "knees", "ankles", "shoulders", "elbows", "wrists", "waist", "neck", "fingers", "sensors", "misc"}),
    "humanoid":    frozenset({"hips", "knees", "ankles", "shoulders", "elbows", "wrists", "waist", "neck", "fingers", "sensors", "misc"}),
    "manipulator": frozenset({"shoulders", "elbows", "wrists", "sensors", "misc"}),
    "wheeled":     frozenset({"sensors", "misc"}),
    "generic":     frozenset({"sensors", "misc"}),
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
    "sole":     "feet",
    "wheel":    "feet",
    "ankle":    "ankles",
    "thigh":    "thighs",
    "upper_leg": "thighs",
    "uleg":     "thighs",
    "uarm":     "shoulders",
    "hip":      "hips",
    "calf":     "calves",
    "shank":    "calves",
    "lower_leg": "calves",
    "lleg":     "calves",
    "larm":     "elbows",
    "knee":     "knees",
    "shoulder": "shoulders",
    "upper_arm": "shoulders",
    "elbow":    "elbows",
    "forearm":  "elbows",
    "wrist":    "wrists",
    "hand":     "hands",
    "finger":   "fingers",
    "thumb":    "fingers",
    "index":    "fingers",
    "middle":   "fingers",
    "ring":     "fingers",
    "little":   "fingers",
    "pinky":    "fingers",
    "waist":    "waist",
    "neck":     "neck",
    "head":     "head",
    "torso":    "torso",
    "trunk":    "torso",
    "spine":    "torso",
    "pelvis":   "pelvis",
    "base":     "base",
    "body":     "base",
    # ── Sensor mount keywords ─────────────────────────────────────────
    # Map every common sensor stem to the "sensors" category; the
    # _suggest_role_id sensor branch then picks the specific role
    # (sensor_imu / sensor_lidar / ...) from the matched token.
    "imu":      "sensors",
    "accel":    "sensors",
    "gyro":     "sensors",
    "lidar":    "sensors",
    "camera":   "sensors",
    "depth":    "sensors",
    "rgb":      "sensors",
    "rgbd":     "sensors",
    "radar":    "sensors",
    "gps":      "sensors",
    "sonar":    "sensors",
    "ultrasonic": "sensors",
    "sensor":   "sensors",
}

# Axis-suffix tokens for multi-DOF humanoid joints (left_hip_PITCH_joint, ...).
# Stored separately from PART_KEYWORDS because they don't denote a category by
# themselves — they qualify a part stem (hip + pitch, ankle + roll, ...).
_AXIS_TOKENS: FrozenSet[str] = frozenset({"pitch", "roll", "yaw"})

# Finger names used for humanoid hand decomposition (thumb_0_L, index_1_R, ...).
_FINGER_NAMES: FrozenSet[str] = frozenset({
    "thumb", "index", "middle", "ring", "little", "pinky"
})

# Family-conditional category remapping. Biped/humanoid get their own
# pelvis / torso / neck slots in the catalog; on every other family those
# tokens collapse back into the generic "base" slot.
_BIPED_LIKE_FAMILIES: FrozenSet[str] = frozenset({"biped", "humanoid"})
# Categories that ONLY exist on biped/humanoid catalogs — when seen on
# any other family, the tokeniser collapses them back to "base" so they
# don't produce orphan role_ids. ``ankles`` is biped-only: quadrupeds
# (Spot / Go2 / A1 / ...) have 3 actuated DOFs per leg (hip / thigh /
# calf) and no ankle joint. Any "*_ank" prim found in a quadruped USD
# is a FixedJoint attaching the foot rigid body to the lower leg, not
# an articulation DOF — and the discovery script now rejects FixedJoints
# upstream so it never reaches the tokeniser anyway.
_BIPED_ONLY_CATEGORIES: FrozenSet[str] = frozenset({"pelvis", "torso", "neck", "waist", "fingers", "head", "ankles"})

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
    # NOTE: "ank" was previously mapped to "ankles" on the false premise
    # that BD Spot has a 4th-DOF ankle per leg. Spot has 3 actuated DOFs
    # per leg; the "*_ank" prim in Spot's USD is a non-actuated
    # FixedJoint attaching the foot to the lower leg. discover_usd_bodies
    # now strips those phantom joints, so this entry has been removed.
}

# Sensor-type → role_id suffix. Used by the sensors branch of
# _suggest_role_id to refine "category=sensors" into a specific
# role_id (sensor_imu / sensor_camera / ...). Ordered list because
# tokenise() may produce both "depth" and "camera" for a depth-RGB
# camera body name; we want the more specific "camera" winner-first.
_SENSOR_TYPE_PRIORITY: Tuple[str, ...] = (
    "imu",      # accel/gyro packaged
    "lidar",
    "camera",   # incl. rgb/depth/rgbd → camera variant
    "depth",
    "radar",
    "gps",
    "sonar",
    "ultrasonic",
    "accel",
    "gyro",
)
_SENSOR_TYPE_REMAP: Dict[str, str] = {
    "rgb":  "camera",
    "rgbd": "camera",
    "ultrasonic": "sonar",
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

AUTO_FROM_ASSET_CATEGORIES: FrozenSet[str] = frozenset({"base", "feet", "pelvis"})

# Semantic alias table for ``BodyIRMapper.get_category_bodies``. When a
# requested category has zero direct hits on the bound robot, fall back
# to the listed IR role ids in declaration order. Each entry is a
# deliberate cross-family equivalence — kept SMALL and EXPLICIT.
#
# Why this exists: presets in ``task_module_registry`` template body
# regex via ``{ir:feet}`` (the rewards-oriented "ground contact" body
# set). Biped morphologies have NO ``feet`` category in the IR catalog
# (the lower limb ends at ankle_roll, which IS the foot end-effector),
# so a direct lookup returns []. Without this fallback, every IL
# locomotion preset emits ``(UNRESOLVED:feet)`` and IsaacLab dies at
# sensor regex resolve time. Quadrupeds keep their direct ``feet`` hit
# (the table only fires on empty direct result), so this is a zero-impact
# addition for existing morphologies.
#
# DO NOT add hierarchy here — this is alias resolution, not inheritance.
_CATEGORY_FAMILY_FALLBACK: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("biped",    "feet"): ("ankle_roll_L", "ankle_roll_R"),
    ("humanoid", "feet"): ("ankle_roll_L", "ankle_roll_R"),
    # Add wheeled / manipulator / other families here when their reward
    # presets start referencing categories the IR catalog doesn't expose.
}


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
    "ankles":    "ankle",
    "shoulders": "shoulder",
    "elbows":    "elbow",
    "wrists":    "wrist",
    "hands":     "hand",
    "fingers":   "finger",
    "waist":     "waist",
    "neck":      "neck",
    "head":      "head",
    "torso":     "torso",
    "pelvis":    "pelvis",
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
    # Bare-name aliases for biped joints whose name doesn't decompose into
    # part+axis cleanly. Keyed by the joint's tokenised bare name (joints
    # with no _joint suffix only — body names retain their _link suffix,
    # see _suggest_role_id step 5).
    #
    # H1 quirk: its single waist DOF is named "torso" (Z-axis = G1 waist_yaw
    # same kinematic spot). The body H1 actuates is named "torso_link" which
    # tokenises differently (bare="torso_link") and falls through to the
    # "torso" body role; the joint (bare="torso") gets re-routed here.
    "biped": {
        "torso": "waist_yaw",
    },
    "humanoid": {
        "torso": "waist_yaw",
    },
}


def _suggest_role_id(link_name: str, family: str = "generic") -> str:
    """Compute the canonical role_id a link/joint name *might* fill.

    ``family`` enables family-aware decisions:
      * biped/humanoid: recognise multi-DOF axis-split joints
        (``hip_pitch_L``, ``ankle_roll_R``, ``waist_yaw``, ...) +
        finger roles (``thumb_2_L``, ``index_tip_R``, ...);
        ``pelvis``/``torso``/``neck`` get dedicated role slots.
      * other families: collapse pelvis/torso/neck back to ``base``,
        and ignore axis tokens (no axis-split slots exist in the
        catalog for non-biped morphologies).
    """
    pair = detect_segment_pair_role(link_name)
    if pair is not None:
        cat, pos = pair
        prefix = _CATEGORY_TO_ROLE_PREFIX.get(
            cat,
            cat.rstrip("s") if cat.endswith("s") else cat,
        )
        return f"{prefix}_{pos}" if pos else prefix

    tokens = tokenize(link_name)
    is_biped_like = family in _BIPED_LIKE_FAMILIES
    family_aliases = _FAMILY_ROLE_ALIASES.get(family, {})

    # 1. Family alias by bare joint name. The bare form filters _joint
    #    suffix but keeps _link so the H1 joint "torso" (bare="torso")
    #    routes to waist_yaw while the body "torso_link" (bare="torso_link")
    #    falls through to the dedicated torso body role below.
    bare = "_".join(t for t in tokens if t != "joint")
    if bare in family_aliases:
        return family_aliases[bare]

    # 2. Finger names take priority on biped (thumb/index/middle/ring/
    #    little/pinky + numeric index, or "tip" for fingertip body roles).
    if is_biped_like:
        finger_name = next((t for t in tokens if t in _FINGER_NAMES), None)
        if finger_name:
            if finger_name == "pinky":
                finger_name = "little"
            side = detect_position(tokens)
            side = side if side in ("L", "R") else ""
            if "tip" in tokens or "fingertip" in tokens:
                return f"{finger_name}_tip_{side}" if side else f"{finger_name}_tip"
            idx_tok = next((t for t in tokens if t.isdigit()), None)
            if idx_tok is not None:
                return f"{finger_name}_{idx_tok}_{side}" if side else f"{finger_name}_{idx_tok}"
            seg_idx = next(
                (str(i) for i, kw in enumerate(("proximal", "intermediate", "distal"))
                 if kw in tokens), None,
            )
            if seg_idx is not None:
                return f"{finger_name}_{seg_idx}_{side}" if side else f"{finger_name}_{seg_idx}"
            return f"hand_{side}" if side else ""

    cat = detect_part(tokens) or ""
    pos = detect_position(tokens)
    axis_tok = next((t for t in tokens if t in _AXIS_TOKENS), None)

    # 3. Sensor mounts — family-agnostic. Refine "sensors" category into
    #    a specific role_id by looking for the most-specific sensor-type
    #    token (imu / lidar / camera / radar / gps / sonar / depth).
    #    Generic mount with no type token → bare "sensor".
    if cat == "sensors":
        for stype in _SENSOR_TYPE_PRIORITY:
            if stype in tokens:
                final = _SENSOR_TYPE_REMAP.get(stype, stype)
                return f"sensor_{final}"
        return "sensor"

    # 4. Family-conditional remapping for biped-only categories on
    #    non-biped families.
    #    - head / neck / fingers: have no kinematic meaning on a
    #      quadruped / wheeled / manipulator; legs/wheels/arms don't
    #      have necks or fingers. Decorative head shells (Go2's
    #      Head_upper / Head_lower) belong in the misc bucket — they
    #      shouldn't compete with the actual base body for the base
    #      slot.
    #    - pelvis / torso / waist: kinematically ARE the chassis on
    #      non-biped robots, so fold them into the base slot.
    if not is_biped_like and cat in _BIPED_ONLY_CATEGORIES:
        if cat in {"head", "neck", "fingers"}:
            return "misc"
        cat = "base"
        axis_tok = None

    if not cat:
        return ""

    # 4. Multi-DOF axis-split joints (biped/humanoid only).
    if is_biped_like and axis_tok:
        if cat == "waist":
            return f"waist_{axis_tok}"
        if cat == "neck":
            return f"neck_{axis_tok}"
        if cat == "head":
            return f"head_{axis_tok}"
        if cat in {"hips", "ankles", "shoulders", "wrists"} and pos:
            prefix = _CATEGORY_TO_ROLE_PREFIX.get(cat, cat.rstrip("s"))
            return f"{prefix}_{axis_tok}_{pos}"

    # 5. Dedicated body roles with no side qualifier.
    if cat == "base":
        return "base"
    if cat == "pelvis":
        return "pelvis"
    if cat == "torso":
        return "torso"
    if cat == "head":
        return "head"
    if cat == "neck":
        return "neck"

    # 6. Standard prefix_{pos} fallback.
    prefix = _CATEGORY_TO_ROLE_PREFIX.get(
        cat,
        cat.rstrip("s") if cat.endswith("s") else cat,
    )
    role = f"{prefix}_{pos}" if pos else prefix
    if role and role in family_aliases:
        return family_aliases[role]
    return role


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
        # Bodies tagged with a "multi-allowed" role (``misc`` /
        # ``sensor_*``) bypass the 1-slot-per-role model entirely:
        # they are kinematic-only links the user has explicitly
        # classified (cosmetic shells, sensor mounts) and must NOT
        # compete for the limb slot, NOT be surfaced as unmapped, and
        # NOT trigger any contact-sensor / IK plumbing. Kept here as a
        # plain set so callers that need "all decorative/sensor links
        # on this robot" can query directly (collision filtering, debug
        # visualisation, etc.).
        self._silent_bodies: Set[str] = set()

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_robot_asset(
        cls, asset: Any, *, active_format: Optional[str] = None,
    ) -> "BodyIRMapper":
        """Build a mapper by **pure lookup** from the asset's per-format tables.

        Stage 3: the mapper no longer parses MJCF / USD files at runtime.
        Body and IR-role data come from the registry's
        ``bodies_per_format[active_format]`` (and ``joints_per_format`` as
        fallback for actuated bodies). The Dump MJCF / Dump USD buttons
        in the Robot Asset card are the only place real assets are
        parsed; their output lands in the registry, then this method
        reads it back deterministically — no runtime heuristics.

        ``active_format`` is required for unambiguous lookup. When a
        RobotAsset only declares one format, callers may pass ``None``
        and we auto-pick the single declared format. When more than one
        format is declared and no preference is given, we raise — the
        caller must commit to a format (env_cfg_compiler passes
        ``spec.robot.active_format``, UI passes the selected tab).

        :func:`discover_from_mjcf` (utility) is what migration / Dump
        MJCF use to extract body lists from a live MuJoCo model — the
        only place ``mujoco.MjModel`` is loaded by this module.
        """
        family = detect_family(asset)

        # ── 1. Pick the format ────────────────────────────────────────
        declared = list(asset.available_formats()) if hasattr(asset, "available_formats") else []
        fmt = str(active_format or "").strip().upper()
        if not fmt:
            if len(declared) == 1:
                fmt = declared[0]
            elif declared:
                # Prefer MJCF when unspecified (existing canvas
                # display assumption); the env_cfg path always pins
                # active_format explicitly.
                fmt = "MJCF" if "MJCF" in declared else declared[0]
            else:
                # No format declared — produce an empty mapper rather
                # than raising. UI uses this to render "no data yet,
                # run Dump MJCF/USD" placeholders without crashing.
                return cls([], family=family)

        # ── 2. Pull the per-format body/joint tables ─────────────────
        bodies_map = asset.bodies_for(fmt) if hasattr(asset, "bodies_for") else {}
        joints_map = asset.joints_for(fmt) if hasattr(asset, "joints_for") else {}

        body_names: List[str] = list(bodies_map.keys())
        mapper = cls(body_names, family=family)
        mapper._build_roles_from_bodies()

        # ── 3. Apply explicit body→ir_role assignments from the table ─
        # bodies_map[name] = ir_role is the authoritative per-format
        # declaration; respect it over keyword tokeniser suggestions.
        from registers.robots import OUT_OF_SCOPE_IR_ROLE
        from application.service.robot_assets.discovery_subprocess import (
            _is_non_unique_role,
        )
        for body_name, ir_role in bodies_map.items():
            if not ir_role:
                continue
            # OOS sentinel — user explicitly tagged this body as
            # "completely ignore at training time". Sentinel isn't a
            # real catalog role (mapper._by_id has no slot), so route
            # to the dedicated out_of_scope set instead of letting it
            # fall through to "slot is None → continue → stuck in
            # unmapped from step 2".
            if ir_role == OUT_OF_SCOPE_IR_ROLE:
                # Drop any speculative step-2 assignment to this body
                # first so it doesn't double-count.
                if mapper._is_assigned(body_name):
                    mapper._send_to_unmapped(body_name)
                mapper._silent_bodies.discard(body_name)
                mapper._unmark_unmapped(body_name)
                mapper._out_of_scope.add(body_name)
                continue
            # Multi-allowed roles (misc / sensor_*) — register the body
            # as silently classified, never compete for a slot. Drop
            # any speculative tokeniser slot assignment to this body.
            if _is_non_unique_role(ir_role):
                for r in mapper._roles:
                    if r.body == body_name:
                        r.body = None
                        r.manual = False
                        r.auto_from_asset = False
                mapper._silent_bodies.add(body_name)
                mapper._unmark_unmapped(body_name)
                continue
            slot = mapper._by_id.get(ir_role)
            if slot is None:
                continue
            if slot.body is not None and slot.body != body_name:
                mapper._send_to_unmapped(slot.body)
            slot.body = body_name
            # Foot bodies (no actuated joint, declared in body table) are
            # marked auto_from_asset so the UI validator hides them.
            if slot.category == "feet":
                slot.auto_from_asset = True
            # Once we explicitly bind, this body is resolved — drop any
            # silent-bucket / unmapped tags left by step 2.
            mapper._silent_bodies.discard(body_name)
            mapper._unmark_unmapped(body_name)

        return mapper

    # ------------------------------------------------------------------
    # Discovery utilities — used by migration script + Dump MJCF/USD
    # buttons. NOT part of the runtime lookup path.
    # ------------------------------------------------------------------

    @staticmethod
    def discover_from_mjcf(
        mjcf_path: Any,
        joints_role_map: Dict[str, str],
    ) -> Dict[str, str]:
        """Parse an MJCF file and return ``{body_name: ir_role}``.

        Used by:
          * ``bootstrap/migrate_canonical_to_per_format.py`` to seed
            ``bodies_per_format.MJCF`` from the existing joint table
          * UI Dump MJCF button (if added) to refresh the table after a
            menagerie update

        Logic:
          1. ``mujoco.MjModel.from_xml_path(mjcf_path)`` → real body list
             + ``jnt_bodyid`` joint↔body relationship
          2. For each joint in ``joints_role_map``, assign its IR role to
             the body it actuates (jnt_bodyid lookup)
          3. For unassigned bodies, keyword-match "foot/toe/sole" / "base/
             trunk/pelvis/torso" → assign to the matching foot_{POS} or
             base IR role

        Returns ``{body_name: ir_role}`` ready to feed back into
        ``bodies_per_format[<FORMAT>]`` (after uid generation).
        """
        import mujoco  # type: ignore
        from pathlib import Path as _Path

        mp = _Path(str(mjcf_path))
        if not mp.exists():
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
        # tinyxml2 / fopen on Windows mis-translates non-ASCII bytes via the
        # ANSI codepage; route through the short-path helper. See
        # application/physics/mujoco_path_compat.py for the full rationale.
        from application.physics.mujoco_path_compat import safe_mjcf_path
        m = mujoco.MjModel.from_xml_path(safe_mjcf_path(mp))

        bodies: List[str] = []
        for i in range(1, m.nbody):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if bn and bn not in bodies:
                bodies.append(bn)

        joint_to_body: Dict[str, str] = {}
        for i in range(m.njnt):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not jn:
                continue
            bid = int(m.jnt_bodyid[i])
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bn:
                joint_to_body[jn] = bn

        body_to_ir: Dict[str, str] = {}
        # Step 1: actuated bodies via joint→body
        for jn, ir_role in joints_role_map.items():
            if not ir_role:
                continue
            bn = joint_to_body.get(jn)
            if bn and bn not in body_to_ir:
                body_to_ir[bn] = ir_role

        # Step 2: structural bodies — feet / base via keyword + position
        _BASE_KW = ("body", "base", "trunk", "pelvis", "torso")
        _FOOT_KW = ("foot", "toe", "sole")
        for bn in bodies:
            if bn in body_to_ir:
                continue
            low = bn.lower()
            if any(k in low for k in _BASE_KW) and "base" not in body_to_ir.values():
                body_to_ir[bn] = "base"
                continue
            if any(k in low for k in _FOOT_KW):
                pos = detect_position(tokenize(bn))
                role = f"foot_{pos}" if pos else "foot"
                if role not in body_to_ir.values():
                    body_to_ir[bn] = role

        return {bn: body_to_ir[bn] for bn in bodies if bn in body_to_ir}

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
        from application.service.robot_assets.discovery_subprocess import (
            _is_cosmetic_name,
            _is_non_unique_role,
        )
        for raw_name in self._body_names:
            link_name = joint_name_to_link_name(raw_name)
            if link_name in self._out_of_scope:
                continue
            if self._is_assigned(link_name):
                continue
            # Cosmetic substring (protector / calflower / shroud ...)
            # short-circuits straight to silent bucket — overrides any
            # limb-stem match the tokeniser would have produced.
            if _is_cosmetic_name(link_name):
                self._silent_bodies.add(link_name)
                continue
            role_id = _suggest_role_id(link_name, self._family)
            if role_id and role_id not in self._by_id:
                role_id = _FAMILY_ROLE_ALIASES.get(
                    self._family, {}
                ).get(role_id, role_id)
            # Multi-allowed roles (misc / sensor_*) bypass the slot
            # model — they go into the silent bucket so the body is
            # marked "resolved" without competing for a 1:1 slot. The
            # explicit step 3 pass below will overwrite this for bodies
            # whose registry ir_role disagrees with the tokeniser.
            if role_id and _is_non_unique_role(role_id):
                self._silent_bodies.add(link_name)
                continue
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
        # Silent bodies (misc / sensor mounts) are already classified —
        # never demote them to unmapped just because a slot was taken
        # by another body. OOS bodies are explicitly excluded too.
        if body not in self._out_of_scope and body not in self._silent_bodies:
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
        direct = [r.body for r in self._roles
                  if r.category == category and r.body is not None]
        if direct:
            return direct
        # Family-aware semantic fallback. The IR catalog does not always
        # expose a category by the same name across morphologies — e.g.
        # biped has no "feet" category (the lower limb ends at
        # ankle_roll, which IS the foot end-effector). Without this
        # fallback, presets templated as ``body_names={ir:feet}`` emit
        # ``(UNRESOLVED:feet)`` on biped and IsaacLab dies. See
        # ``_CATEGORY_FAMILY_FALLBACK`` for the alias table.
        fallback_role_ids = _CATEGORY_FAMILY_FALLBACK.get(
            (self._family, category), ()
        )
        out: List[str] = []
        for alt_role_id in fallback_role_ids:
            role = self._by_id.get(alt_role_id)
            if role is not None and role.body is not None:
                out.append(role.body)
        return out

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
        # Silent bucket holds bodies the user explicitly tagged
        # misc/sensor — they're resolved and must not show up as
        # "needs attention" in the canvas Body Mapping table.
        return [b for b in self._unmapped_bodies
                if b not in self._silent_bodies]

    def silent_bodies(self) -> List[str]:
        """Bodies classified as misc / sensor mounts — kinematic-only
        links resolved by the user but not bound to any limb slot.

        Spec-compiler / collision-filter callers can query this list to
        e.g. exclude these bodies from contact reward terms while still
        keeping them in the simulation."""
        return list(self._silent_bodies)

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
