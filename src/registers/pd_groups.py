"""registers.pd_groups — PD 关节组注册表 / Joint-group registry for PD parameterization.

Provides family-level joint groups (e.g. ``hip_x``, ``hip_y``, ``knee`` for
quadruped) and per-group ``(omega_n, zeta)`` defaults that feed into the
mass-matrix-adaptive sim2sim PD framework (see
``knowledge_base/sim2sim_mass-matrix-adaptive.yaml`` and the canvas
``ActuatorPDNode``).

Design invariants (Stage A of the sim2sim refactor):

* Definitions (which IR roles belong to which group) live in
  ``data/pd_groups_definitions.json`` — factory-shipped, read-only at
  runtime.
* Defaults (``omega_n``, ``zeta`` per group per family) live in
  ``data/pd_groups_defaults.json`` — factory-shipped, read-only at
  runtime.
* User overlays land in ``<USER_CONFIG_DIR>/registers/pd_groups_custom.json``
  and are merged at load time via :func:`merge_user_extensions`. Only
  defaults can be overridden user-side — group regex definitions are
  fixed by the family to keep the IR-canonical contract stable
  (rule §1.2: no auto-extension).
* Cross-registry validation (run from :class:`RegistryHub.validate`):
    - every group regex must compile;
    - every group's regex must match at least one role in its family's
      IR canonical (otherwise the group is dead);
    - two groups within the same family must not match the same IR
      role (overlap detection — compile-time PD assignment would be
      ambiguous).
* No engine-specific ``kp``/``kd`` values are stored in this registry.
  Solver helpers in ``application.physics.{mujoco_gain_solver,
  physx_gain_solver}`` derive engine gains from ``(omega_n, zeta)``
  at runtime.

API (kept symmetric with sibling registers — ``ir``, ``commands``,
``robots``):

    load() -> int                                      # total groups across all families
    list_families() -> list[str]
    list_groups(family) -> list[dict]                  # each: {id, label, ir_role_regex, rationale}
    get_group(family, group_id) -> dict | None
    get_defaults(family) -> dict[str, dict]            # group_id -> {omega_n, zeta}
    get_default(family, group_id) -> dict | None
    find_group_for_role(family, role_id) -> tuple[str, dict] | None
    merge_user_extensions(payload) -> int              # returns # overrides applied
    validate() -> list[str]                            # called from RegistryHub.validate()
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_DEFINITIONS_PATH = _DATA_DIR / "pd_groups_definitions.json"
_DEFAULTS_PATH = _DATA_DIR / "pd_groups_defaults.json"


_state: Dict[str, Any] = {
    "loaded": False,
    "definitions_version": "",
    "defaults_version": "",
    # family -> {group_id: {id, label, ir_role_regex, rationale, _compiled}}
    "definitions": {},
    # family -> {group_id: {omega_n: float, zeta: float}}
    "defaults": {},
    # family -> {group_id: {omega_n?: float, zeta?: float}} — user overrides
    "user_overrides": {},
}


def _resolve_alias(family: str, raw_families: Dict[str, Any]) -> str:
    """Follow ``alias_of`` chain in a family map. Mirrors ``ir._resolve_alias``."""
    seen: set = set()
    cur = family
    while cur not in seen:
        seen.add(cur)
        spec = raw_families.get(cur, {})
        target = spec.get("alias_of") if isinstance(spec, dict) else None
        if not target:
            return cur
        cur = target
    return cur


def load() -> int:
    """Load definitions + defaults JSON; return total group count.

    Idempotent — repeat calls short-circuit on the cached state.
    """
    if _state["loaded"]:
        return sum(len(groups) for groups in _state["definitions"].values())

    defs_payload = read_data(_DEFINITIONS_PATH)
    if not isinstance(defs_payload, dict):
        raise RuntimeError(
            f"pd_groups_definitions.json failed to parse: {_DEFINITIONS_PATH}"
        )

    defaults_payload = read_data(_DEFAULTS_PATH)
    if not isinstance(defaults_payload, dict):
        raise RuntimeError(
            f"pd_groups_defaults.json failed to parse: {_DEFAULTS_PATH}"
        )

    _state["definitions_version"] = str(defs_payload.get("version", ""))
    _state["defaults_version"] = str(defaults_payload.get("version", ""))

    raw_def_families = defs_payload.get("families", {})
    raw_dflt_families = defaults_payload.get("families", {})

    if not isinstance(raw_def_families, dict) or not isinstance(raw_dflt_families, dict):
        raise RuntimeError(
            "pd_groups: 'families' must be a mapping in both definitions and defaults"
        )

    resolved_defs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fam_id in raw_def_families:
        target = _resolve_alias(fam_id, raw_def_families)
        spec = raw_def_families.get(target, {}) or {}
        groups_raw = spec.get("groups", {}) if isinstance(spec, dict) else {}
        if not isinstance(groups_raw, dict):
            log_warning(
                f"[pd_groups] family={fam_id!r} has non-dict 'groups' "
                f"(target={target!r}); skipping"
            )
            continue
        compiled_groups: Dict[str, Dict[str, Any]] = {}
        for gid, gspec in groups_raw.items():
            if not isinstance(gspec, dict):
                log_warning(
                    f"[pd_groups] family={fam_id!r} group={gid!r} is not a dict; skipping"
                )
                continue
            regex_str = str(gspec.get("ir_role_regex", "")).strip()
            if not regex_str:
                log_warning(
                    f"[pd_groups] family={fam_id!r} group={gid!r} has empty "
                    f"ir_role_regex; skipping"
                )
                continue
            try:
                compiled = re.compile(regex_str)
            except re.error as exc:
                raise RuntimeError(
                    f"pd_groups: family={fam_id!r} group={gid!r} ir_role_regex "
                    f"{regex_str!r} failed to compile: {exc}"
                ) from exc
            compiled_groups[gid] = {
                "id": gid,
                "label": str(gspec.get("label", gid)),
                "ir_role_regex": regex_str,
                "rationale": str(gspec.get("rationale", "")),
                "_compiled": compiled,
            }
        resolved_defs[fam_id] = compiled_groups

    _state["definitions"] = resolved_defs

    resolved_defaults: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fam_id, defaults in raw_dflt_families.items():
        if not isinstance(defaults, dict):
            continue
        fam_target = _resolve_alias(fam_id, raw_def_families)
        groups_in_family = resolved_defs.get(fam_target, {})
        resolved_defaults[fam_id] = {}
        for gid, vals in defaults.items():
            if not isinstance(vals, dict):
                continue
            if gid not in groups_in_family:
                # Defaults file declares a group that has no definition.
                # Fail-loud (rule §1.8): silent drop would mask typos.
                raise RuntimeError(
                    f"pd_groups: family={fam_id!r} defaults declare group "
                    f"{gid!r} but no matching group exists in "
                    f"pd_groups_definitions.json (resolved family={fam_target!r}). "
                    f"Fix the typo or add the group to definitions."
                )
            try:
                omega_n = float(vals["omega_n"])
                zeta = float(vals["zeta"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"pd_groups: family={fam_id!r} group={gid!r} defaults "
                    f"must declare numeric omega_n and zeta: {exc}"
                ) from exc
            if not (0.0 < omega_n <= 200.0):
                raise RuntimeError(
                    f"pd_groups: family={fam_id!r} group={gid!r} omega_n="
                    f"{omega_n} out of range (0, 200] rad/s"
                )
            if not (0.1 <= zeta <= 2.0):
                raise RuntimeError(
                    f"pd_groups: family={fam_id!r} group={gid!r} zeta="
                    f"{zeta} out of range [0.1, 2.0]"
                )
            resolved_defaults[fam_id][gid] = {"omega_n": omega_n, "zeta": zeta}

    _state["defaults"] = resolved_defaults
    _state["user_overrides"] = {}
    _state["loaded"] = True
    return sum(len(groups) for groups in resolved_defs.values())


def list_families() -> List[str]:
    if not _state["loaded"]:
        load()
    return list(_state["definitions"].keys())


def list_groups(family: str) -> List[Dict[str, Any]]:
    """Return the group definitions for ``family`` (excluding the compiled regex object).

    Returns ``[]`` for unknown families; the caller decides whether that
    is an error in context.
    """
    if not _state["loaded"]:
        load()
    groups = _state["definitions"].get(family, {})
    return [
        {k: v for k, v in g.items() if not k.startswith("_")}
        for g in groups.values()
    ]


def get_group(family: str, group_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    groups = _state["definitions"].get(family, {})
    g = groups.get(group_id)
    if g is None:
        return None
    return {k: v for k, v in g.items() if not k.startswith("_")}


def get_defaults(family: str) -> Dict[str, Dict[str, float]]:
    """Return ``{group_id: {omega_n, zeta}}`` for ``family``, with user overrides applied.

    Falls back to ``{}`` for unknown families (caller decides whether to
    raise).
    """
    if not _state["loaded"]:
        load()
    base = dict(_state["defaults"].get(family, {}))
    overrides = _state["user_overrides"].get(family, {})
    for gid, partial in overrides.items():
        if gid not in base:
            continue
        merged = dict(base[gid])
        if "omega_n" in partial:
            merged["omega_n"] = float(partial["omega_n"])
        if "zeta" in partial:
            merged["zeta"] = float(partial["zeta"])
        base[gid] = merged
    return base


def get_default(family: str, group_id: str) -> Optional[Dict[str, float]]:
    """Return ``{omega_n, zeta}`` for one group, or ``None`` if undefined."""
    defaults = get_defaults(family)
    return defaults.get(group_id)


def find_group_for_role(family: str, role_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(group_id, group_dict)`` for the first matching group, or ``None``.

    Used by ``application.physics.joint_group_resolver`` to expand IR
    roles to groups at compile time. Returns the first match by
    insertion order — overlap detection is the registry validator's job
    (we don't pay regex cost twice per call).
    """
    if not _state["loaded"]:
        load()
    groups = _state["definitions"].get(family, {})
    for gid, g in groups.items():
        if g["_compiled"].fullmatch(role_id):
            return gid, {k: v for k, v in g.items() if not k.startswith("_")}
    return None


def merge_user_extensions(payload: Dict[str, Any]) -> int:
    """Merge a user overlay payload into the cached defaults.

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/pd_groups_custom.json``)::

        {
          "version": "1.0.0",
          "overrides": {
            "<family>": {
              "<group_id>": { "omega_n": <float>, "zeta": <float> }
            }
          }
        }

    Returns the number of ``(family, group_id)`` override entries
    successfully applied. Unknown families / groups are logged and
    skipped (a typo there is a user mistake, not a fatal app boot
    failure).
    """
    if not _state["loaded"]:
        load()
    if not isinstance(payload, dict):
        return 0
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        return 0
    applied = 0
    for family, group_map in overrides.items():
        if family not in _state["definitions"]:
            log_warning(
                f"[pd_groups] user overlay declares unknown family {family!r}; "
                f"skipping (valid families: {list(_state['definitions'].keys())})"
            )
            continue
        if not isinstance(group_map, dict):
            continue
        fam_groups = _state["definitions"][family]
        fam_overrides = _state["user_overrides"].setdefault(family, {})
        for gid, vals in group_map.items():
            if gid not in fam_groups:
                log_warning(
                    f"[pd_groups] user overlay declares unknown group "
                    f"{family!r}.{gid!r}; skipping (valid: {list(fam_groups.keys())})"
                )
                continue
            if not isinstance(vals, dict):
                continue
            entry = dict(fam_overrides.get(gid, {}))
            if "omega_n" in vals:
                try:
                    omega_n = float(vals["omega_n"])
                    if not (0.0 < omega_n <= 200.0):
                        raise ValueError(f"omega_n={omega_n} out of range (0, 200]")
                    entry["omega_n"] = omega_n
                except (TypeError, ValueError) as exc:
                    log_warning(
                        f"[pd_groups] user overlay {family!r}.{gid!r} omega_n "
                        f"is invalid ({exc}); skipping that value"
                    )
            if "zeta" in vals:
                try:
                    zeta = float(vals["zeta"])
                    if not (0.1 <= zeta <= 2.0):
                        raise ValueError(f"zeta={zeta} out of range [0.1, 2.0]")
                    entry["zeta"] = zeta
                except (TypeError, ValueError) as exc:
                    log_warning(
                        f"[pd_groups] user overlay {family!r}.{gid!r} zeta "
                        f"is invalid ({exc}); skipping that value"
                    )
            if entry:
                fam_overrides[gid] = entry
                applied += 1
    return applied


def validate() -> List[str]:
    """Cross-registry integrity check.

    Verifies (per family that has ≥1 group defined):
      1. Every group's regex matches at least one role in the family's
         IR canonical. A group with zero matches is dead code.
      2. No two groups within the same family match the same IR role
         (overlap = ambiguous PD assignment at compile time).
      3. Every default declared in ``pd_groups_defaults.json`` has a
         matching definition (caught at load time, but re-verified here
         in case a user overlay added a new family without a default).

    Returns a list of human-readable problem strings. Empty list = OK.
    The caller (RegistryHub.validate) decides whether to raise.
    """
    from . import ir as _ir

    if not _state["loaded"]:
        load()

    problems: List[str] = []

    for family, groups in _state["definitions"].items():
        if not groups:
            continue  # generic family declares no groups; nothing to validate
        ir_roles = [r.get("id", "") for r in _ir.list_roles(family) if isinstance(r, dict)]
        ir_roles = [rid for rid in ir_roles if rid]
        if not ir_roles:
            problems.append(
                f"  pd_groups: family={family!r} has groups but the IR canonical "
                f"declares no roles for it — pd_groups will be unreachable. "
                f"Either drop the family from pd_groups_definitions.json or add "
                f"roles to ir_canonical.json."
            )
            continue

        role_to_groups: Dict[str, List[str]] = {}
        for gid, g in groups.items():
            compiled: re.Pattern = g["_compiled"]
            matched = [rid for rid in ir_roles if compiled.fullmatch(rid)]
            if not matched:
                problems.append(
                    f"  pd_groups: family={family!r} group={gid!r} regex "
                    f"{g['ir_role_regex']!r} matches no IR role for the family. "
                    f"Available roles: {ir_roles[:8]}{'...' if len(ir_roles) > 8 else ''}"
                )
                continue
            for rid in matched:
                role_to_groups.setdefault(rid, []).append(gid)

        for rid, gids in role_to_groups.items():
            if len(gids) > 1:
                problems.append(
                    f"  pd_groups: family={family!r} IR role {rid!r} is matched "
                    f"by multiple groups {gids}. Resolve the overlap so PD "
                    f"assignment is unambiguous at compile time."
                )

        # Defaults coverage: every group with a definition should have a default.
        family_defaults = _state["defaults"].get(family, {})
        for gid in groups:
            if gid not in family_defaults:
                problems.append(
                    f"  pd_groups: family={family!r} group={gid!r} has no default "
                    f"in pd_groups_defaults.json — every defined group needs a "
                    f"factory (omega_n, zeta) pair."
                )

    return problems


def get_version() -> str:
    """Return the loaded definitions file version string."""
    if not _state["loaded"]:
        load()
    return str(_state["definitions_version"])
