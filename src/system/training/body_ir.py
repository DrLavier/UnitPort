"""Body-name IR layer — keyword-driven, robot-agnostic.

No hardcoded limb templates.  IR roles are generated dynamically from
the robot's actual body names via keyword decomposition::

    USD body "FL_thigh"  →  tokens {fl, thigh}
                          →  part=thigh, position=FL
                          →  auto-maps to category "thighs"

Categories referenced by rewards (``{ir:feet}``, ``{ir:thighs}``, …)
collect all IR roles whose part keyword matches.  Non-standard bodies
(caterpillar segments, tentacles, …) that don't match any keyword
appear as unmapped entries for manual assignment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Keyword tables
# ═══════════════════════════════════════════════════════════════════════

# Part keywords → category name.
# A USD body containing any of these tokens is auto-assigned to the
# category.  Order matters: first match wins.
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

# Position keywords → normalised position tag.
POSITION_KEYWORDS: Dict[str, str] = {
    "fl": "FL", "fr": "FR", "rl": "RL", "rr": "RR",
    "left": "L", "right": "R",
    "l": "L", "r": "R",
    "front": "F", "rear": "R_",
    "back": "R_",
}

# Categories that are auto-populated from RobotAsset metadata and
# hidden from the validator UI (users don't need to touch these).
AUTO_FROM_ASSET_CATEGORIES: FrozenSet[str] = frozenset({"base", "feet"})


# ═══════════════════════════════════════════════════════════════════════
# Tokeniser
# ═══════════════════════════════════════════════════════════════════════

def joint_name_to_link_name(joint_name: str) -> str:
    """Derive the likely USD link/body name from a joint name.

    Common conventions:
    - ``FL_thigh_joint`` → ``FL_thigh``   (strip ``_joint`` suffix)
    - ``left_hip_yaw``   → ``left_hip_yaw`` (no suffix to strip)
    - ``joint_0``        → ``joint_0``     (no recognisable suffix)
    """
    # Strip common joint-name suffixes
    for suffix in ("_joint", "_Joint", "_actuator", "_motor"):
        if joint_name.endswith(suffix):
            return joint_name[: -len(suffix)]
    return joint_name


def tokenize(name: str) -> List[str]:
    """Split a body name into lowercase tokens.

    ``"FL_thigh_joint"`` → ``["fl", "thigh", "joint"]``
    ``"LeftHip"``        → ``["left", "hip"]``
    """
    # Split on underscores, hyphens, dots, digits-to-alpha boundaries
    parts = re.split(r"[_\-\.]+", name)
    tokens: List[str] = []
    for p in parts:
        # CamelCase split: "LeftHip" → ["Left", "Hip"]
        sub = re.sub(r"([a-z])([A-Z])", r"\1_\2", p).split("_")
        tokens.extend(s.lower() for s in sub if s)
    return tokens


def detect_part(tokens: List[str]) -> Optional[str]:
    """Return the category name from part keywords, or None."""
    # Try multi-token matches first (e.g. "upper_leg")
    joined = "_".join(tokens)
    for kw, cat in PART_KEYWORDS.items():
        if kw in joined:
            return cat
    # Single-token scan
    for t in tokens:
        if t in PART_KEYWORDS:
            return PART_KEYWORDS[t]
    return None


def detect_position(tokens: List[str]) -> str:
    """Return a normalised position tag (FL, FR, L, R, …) or ''."""
    # Multi-char position tokens first (fl, fr, rl, rr)
    for t in tokens:
        if t in POSITION_KEYWORDS:
            return POSITION_KEYWORDS[t]
    return ""


# ═══════════════════════════════════════════════════════════════════════
# IR Role
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IRRole:
    """One IR role → one USD body name (or None)."""
    role_id: str              # e.g. "thigh_FL", "base", "segment_3"
    category: str             # e.g. "thighs", "base", ""
    label: str                # e.g. "Thigh FL", "Base"
    position: str             # e.g. "FL", "L", ""
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
    """Keyword-driven body-name mapper.  No hardcoded limb templates."""

    def __init__(self, body_names: List[str]):
        self._body_names = list(body_names)
        self._roles: List[IRRole] = []
        self._by_id: Dict[str, IRRole] = {}

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_robot_asset(cls, asset: Any) -> "BodyIRMapper":
        bodies: List[str] = []
        if hasattr(asset, "base_link") and asset.base_link:
            bodies.append(asset.base_link)
        for jn in (getattr(asset, "joint_names", None) or []):
            if jn not in bodies:
                bodies.append(jn)
        for fn in (getattr(asset, "foot_link_names", None) or []):
            if fn not in bodies:
                bodies.append(fn)
        # If parser didn't detect foot links, infer from calf joints:
        # *_calf_joint → *_foot (standard quadruped/biped convention)
        foot_links = list(getattr(asset, "foot_link_names", None) or [])
        if not foot_links:
            for jn in (getattr(asset, "joint_names", None) or []):
                ln = joint_name_to_link_name(jn)
                if "calf" in ln.lower() or "shank" in ln.lower() or "ankle" in ln.lower():
                    # Derive foot link: replace calf→foot, shank→foot
                    import re as _re
                    candidate = _re.sub(r"(?i)calf|shank|ankle", "foot", ln)
                    if candidate != ln and candidate not in bodies:
                        bodies.append(candidate)
                        foot_links.append(candidate)

        mapper = cls(bodies)
        mapper._build_roles_from_bodies()

        # Seed auto_from_asset entries from parsed asset metadata.
        # base_link and foot_link_names are already link names (not
        # joint names), so match them directly.
        base_link = getattr(asset, "base_link", "") or ""
        if base_link:
            r = mapper._by_id.get("base")
            if r:
                r.body = base_link
                r.auto_from_asset = True
        foot_links = set(getattr(asset, "foot_link_names", None) or [])
        if foot_links:
            for r in mapper._roles:
                if r.category == "feet" and r.body in foot_links:
                    r.auto_from_asset = True

        return mapper

    @classmethod
    def from_body_list(cls, body_names: List[str],
                       family: str = "quadruped") -> "BodyIRMapper":
        mapper = cls(body_names)
        mapper._build_roles_from_bodies()
        return mapper

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyIRMapper":
        body_names = d.get("body_names", [])
        mapper = cls(body_names)
        mapper._build_roles_from_bodies()
        # Restore saved overrides
        roles_data = d.get("roles", {})
        for rid, rd in roles_data.items():
            r = mapper._by_id.get(rid)
            if r:
                r.body = rd.get("body")
                r.manual = rd.get("manual", False)
                r.auto_from_asset = rd.get("auto_from_asset", r.auto_from_asset)
        return mapper

    # ── Core: build roles from actual body names ──────────────────────

    def _build_roles_from_bodies(self) -> None:
        """Scan body names, decompose keywords, create one IRRole per body.

        The ``body`` field stores the **link name** (what USD scene
        exposes to contact sensors), not the raw joint name.  Joint names
        like ``FL_thigh_joint`` are converted to ``FL_thigh``.
        """
        self._roles.clear()
        self._by_id.clear()
        seen_ids: Set[str] = set()

        for raw_name in self._body_names:
            # Derive link name (strip _joint suffix etc.)
            link_name = joint_name_to_link_name(raw_name)
            tokens = tokenize(link_name)
            cat = detect_part(tokens) or ""
            pos = detect_position(tokens)

            # Build role_id
            if cat and cat != "base":
                singular = cat.rstrip("s") if cat.endswith("s") else cat
                role_id = f"{singular}_{pos}" if pos else singular
            elif cat == "base":
                role_id = "base"
            else:
                role_id = link_name

            # Deduplicate
            if role_id in seen_ids:
                i = 2
                while f"{role_id}_{i}" in seen_ids:
                    i += 1
                role_id = f"{role_id}_{i}"
            seen_ids.add(role_id)

            # Label
            if cat:
                singular = cat.rstrip("s").title() if cat.endswith("s") else cat.title()
                label = f"{singular} {pos}" if pos else singular
            else:
                label = link_name

            auto = cat in AUTO_FROM_ASSET_CATEGORIES
            role = IRRole(
                role_id=role_id,
                category=cat,
                label=label,
                position=pos,
                auto_from_asset=auto,
                body=link_name,   # link name, NOT joint name
            )
            self._roles.append(role)
            self._by_id[role_id] = role

    # ── Public API ────────────────────────────────────────────────────

    def override(self, role_id: str, body: Optional[str]) -> None:
        r = self._by_id.get(role_id)
        if r:
            r.body = body
            r.manual = True

    def get(self, role_id: str) -> Optional[IRRole]:
        return self._by_id.get(role_id)

    def get_category_bodies(self, category: str) -> List[str]:
        return [r.body for r in self._roles
                if r.category == category and r.body is not None]

    def all_resolved(self, required_only: bool = True) -> bool:
        # All roles with a category are "required" by default
        for r in self._roles:
            if required_only and not r.category:
                continue
            if not r.resolved:
                return False
        return True

    def unresolved_roles(self, required_only: bool = True) -> List[str]:
        return [r.role_id for r in self._roles
                if (not required_only or r.category) and not r.resolved]

    @property
    def roles(self) -> List[IRRole]:
        return list(self._roles)

    @property
    def body_names(self) -> List[str]:
        return list(self._body_names)

    @property
    def family(self) -> str:
        return ""  # no longer template-based

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_names": self._body_names,
            "roles": {
                r.role_id: {
                    "body": r.body,
                    "manual": r.manual,
                    "auto_from_asset": r.auto_from_asset,
                }
                for r in self._roles
            },
        }

    def categories_present(self) -> Set[str]:
        """Return the set of categories detected in this robot's bodies."""
        return {r.category for r in self._roles if r.category}


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
    """Split ``"thighs_hips_base"`` → ``["thighs", "hips", "base"]``.

    Greedy-matches known category names.  Unknown fragments are kept
    as-is so novel categories from future extensions still work.
    """
    # Build known category set from PART_KEYWORDS values
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
            # Unknown — take until next underscore or end
            idx = rem.find("_")
            if idx > 0:
                cats.append(rem[:idx])
                rem = rem[idx + 1:]
            else:
                cats.append(rem)
                break
    return cats
