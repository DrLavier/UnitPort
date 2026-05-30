# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.symmetry — Symmetry mirror op declarative data registry.

Purpose
-------
- Single source of truth for the symmetry mirror operation. This module
  ships **data + validation only** — the actual ``apply_mirror()``
  consumer is implemented by the training-side code that needs it.
- Per-family partition of every IR role in the family slice into one of
  three sets: ``role_swaps`` (L↔R bilateral pairs), ``role_self_map``
  (centerline / self-symmetric), ``role_unmapped`` (non-policy non-mirror
  — sensors, decorative misc).
- Cross-registry validate() enforces partition closure (every role in
  exactly one set; duplicates raise) and that referenced roles all exist
  in the family's ``ir_canonical`` slice.

Field naming
------------
- ``role_swaps`` / ``role_self_map`` / ``role_unmapped`` cover both joint
  and body IR roles. The biped bilateral set includes body roles
  (``foot_L/R``, ``hand_L/R``, fingertip pairs) so a ``joint_*`` name
  would mis-describe them; ``role_*`` is the accurate term.
- ``obs_channel_sign_flips`` / ``action_channel_sign_flips`` keep their
  names because these process **channels**, not roles. Both ship empty
  here and are filled by the consumer that defines the channel layout
  (AMP obs templates, gait command).

alias_of vs inherits_from
-------------------------
- ``alias_of`` here = **data identity** at load time (load-time hard
  expansion, no override). Mirrors ``ir._resolve_alias`` /
  ``pd_groups._resolve_alias`` pattern.
- ``inherits_from`` (from ``families_catalog.json``) = **dispatch
  fallback chain** opt-in by the caller. A different concept; this module
  does NOT use ``inherits_from``.
- ``humanoid`` declares ``alias_of: "biped"`` here, mirroring ir_canonical
  + pd_groups_definitions. The three-layer alignment (ir, pd_groups,
  symmetry all declare ``humanoid alias_of biped``) is a by-design
  observation, NOT a cross-registry hard constraint — each layer is
  validated independently.

Data layout
-----------
- Factory catalog: ``registers/data/symmetry_catalog.json`` (read-only at
  runtime).
- User overlay: ``<USER_CONFIG_DIR>/registers/symmetry_custom.json``
  merged at load by ``RegistryHub._merge_user_overlays`` →
  :func:`merge_user_extensions`.
- The overlay can add new families OR override existing ones; same
  load-time L-rule validation applies.

Fail-loud — no silent fallbacks
-------------------------------
- Load-time L1-L7 raise on malformed catalog.
- Cross-registry validate() returns problem list for S1/S3/S4; the hub
  wraps non-empty problem list into ``RegistryValidationError``.
- Runtime API errors raise :class:`SymmetryValidationError` with
  actionable messages.
- :func:`has_symmetry` returns False ONLY when the family is absent from
  the catalog; catalog corruption (broken alias chain) raises instead of
  silently returning False — that case must surface via
  RegistryHub.validate, not get masked.

API
---
::

    load() -> int
    merge_user_extensions(payload) -> int
    list_families() -> list[str]
    get_symmetry(family) -> SymmetryData
    has_symmetry(family) -> bool
    validate() -> list[str]
    get_version() -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_debug, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CATALOG_PATH = _DATA_DIR / "symmetry_catalog.json"
_USER_CUSTOM_REL = "registers/symmetry_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user-overlay path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart (mirrors :func:`registers.families._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "symmetry_custom.json"


# ---------------------------------------------------------------------------
# Public dataclasses / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetryData:
    """Resolved symmetry data for one family.

    ``@dataclass(frozen=True)`` so downstream consumers (AMP obs / gait
    command / future apply_mirror) can cache safely. Tuple-of-tuple shape
    (instead of list) reflects the immutability contract.

    Fields:
        family: family id passed to :func:`get_symmetry`. When the query
            target is an alias, this is still the queried id.
        alias_of: ``alias_of`` value declared on the QUERIED family entry
            (``None`` if the queried family had its own data).
        resolved_from: family id whose data this struct contains. Equal
            to ``family`` when not aliased; otherwise the chain root.
        role_swaps: tuple of ``(left, right)`` string pairs.
        role_self_map: tuple of role ids that mirror to themselves
            (centerline / self-symmetric). Sign-flip per channel is
            captured separately by the ``*_sign_flips`` fields below.
        role_unmapped: tuple of role ids that do not participate in
            mirror (non-policy: sensor mounts, decorative misc).
        obs_channel_sign_flips: filled by the consumer that defines the
            observation channel layout (empty until then).
        action_channel_sign_flips: filled by the consumer that defines
            the action channel layout (empty until then).
    """

    family: str
    alias_of: Optional[str]
    resolved_from: str
    role_swaps: Tuple[Tuple[str, str], ...]
    role_self_map: Tuple[str, ...]
    role_unmapped: Tuple[str, ...]
    obs_channel_sign_flips: Tuple[Any, ...]
    action_channel_sign_flips: Tuple[Any, ...]


class SymmetryValidationError(RuntimeError):
    """Raised by the runtime API on:

    * unknown family in :func:`get_symmetry` (catalog has no entry for
      the queried name);
    * defensive belt for broken alias chain (missing target / cycle)
      encountered at runtime — :class:`RegistryHub.validate` should have
      caught it earlier; this raise surfaces it if validate has not run.

    Cross-registry validate() problems surface via
    :class:`RegistryValidationError` from the hub instead — this exception
    is for runtime API contract violations only (mirrors
    :class:`families.FamiliesValidationError`).
    """


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------


_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    # fid -> spec dict:
    #   alias entries: {"alias_of": <str>}
    #   data entries: {"alias_of": None, "role_swaps", "role_self_map",
    #                  "role_unmapped", "obs_channel_sign_flips",
    #                  "action_channel_sign_flips"}
    "families": {},
    # fid -> original on-disk spec from user overlay (round-trip preservation).
    "user_overrides": {},
}


# ---------------------------------------------------------------------------
# Per-entry schema validation (L1-L7)
# ---------------------------------------------------------------------------


_DATA_FIELDS = (
    "role_swaps",
    "role_self_map",
    "role_unmapped",
    "obs_channel_sign_flips",
    "action_channel_sign_flips",
)


def _validate_entry(fid: str, spec: Dict[str, Any], *, source: str) -> None:
    """Per-family schema validation. Raises RuntimeError on malformed entry.

    Covers L3 (lowercase ASCII fid), L4 (alias_of conflict), L5 (swap pair
    shape), L6 (missing data fields on non-alias), L7 (sign_flips list type).
    """
    # L3: lowercase ASCII fid.
    if fid != fid.lower() or not fid.isascii():
        raise RuntimeError(
            f"symmetry: family id {fid!r} must be lowercase ASCII; "
            f"rename in {source}"
        )
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"symmetry: family={fid!r} in {source} is not a dict "
            f"(got {type(spec).__name__})"
        )
    alias = spec.get("alias_of")
    if alias is not None and not isinstance(alias, str):
        raise RuntimeError(
            f"symmetry: family={fid!r}.alias_of must be a string or null "
            f"(got {type(alias).__name__}) in {source}"
        )
    if alias:
        # L4: alias entries must NOT also carry data fields.
        conflicting = [f for f in _DATA_FIELDS if f in spec]
        if conflicting:
            raise RuntimeError(
                f"symmetry: family={fid!r} declares alias_of={alias!r} but "
                f"also has {conflicting!r} — choose one (data identity "
                f"alias means the target's data is reused verbatim); "
                f"in {source}"
            )
        return
    # Non-alias entries: every required field must be present + typed.
    for field in _DATA_FIELDS:
        if field not in spec:
            raise RuntimeError(
                f"symmetry: family={fid!r} missing required field "
                f"{field!r}; declare it as an empty array if not "
                f"applicable; in {source}"
            )
    # L5: swap pairs are 2-element string lists.
    swaps = spec["role_swaps"]
    if not isinstance(swaps, list):
        raise RuntimeError(
            f"symmetry: family={fid!r}.role_swaps must be a list (got "
            f"{type(swaps).__name__}) in {source}"
        )
    for i, pair in enumerate(swaps):
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError(
                f"symmetry: family={fid!r}.role_swaps[{i}]={pair!r} is "
                f"not a 2-element [left, right] list in {source}"
            )
        if not all(isinstance(x, str) for x in pair):
            raise RuntimeError(
                f"symmetry: family={fid!r}.role_swaps[{i}]={pair!r} "
                f"contains non-string elements in {source}"
            )
    # role_self_map / role_unmapped: list of str.
    for field in ("role_self_map", "role_unmapped"):
        value = spec[field]
        if not isinstance(value, list):
            raise RuntimeError(
                f"symmetry: family={fid!r}.{field} must be a list (got "
                f"{type(value).__name__}) in {source}"
            )
        for j, item in enumerate(value):
            if not isinstance(item, str):
                raise RuntimeError(
                    f"symmetry: family={fid!r}.{field}[{j}]={item!r} is "
                    f"not a string in {source}"
                )
    # L7: sign_flips fields exist + are lists (H0 may be empty).
    for field in ("obs_channel_sign_flips", "action_channel_sign_flips"):
        value = spec[field]
        if not isinstance(value, list):
            raise RuntimeError(
                f"symmetry: family={fid!r}.{field} must be a list (H0 may "
                f"be empty) in {source} (got {type(value).__name__})"
            )


def _load_payload_into_state(
    families_block: Dict[str, Any],
    *,
    source: str,
    track_user_overrides: bool = False,
) -> int:
    """Validate + merge a parsed families block into :data:`_state['families']`.

    Used by both :func:`load` (factory) and :func:`merge_user_extensions`
    (user overlay). Returns the number of families added / overridden.
    """
    if not isinstance(families_block, dict):
        raise RuntimeError(
            f"symmetry: 'families' must be a mapping in {source} "
            f"(got {type(families_block).__name__})"
        )
    added = 0
    for fid_raw, spec in families_block.items():
        fid = str(fid_raw)
        _validate_entry(fid, spec, source=source)
        if spec.get("alias_of"):
            _state["families"][fid] = {"alias_of": str(spec["alias_of"])}
        else:
            _state["families"][fid] = {
                "alias_of": None,
                "role_swaps": [list(pair) for pair in spec["role_swaps"]],
                "role_self_map": list(spec["role_self_map"]),
                "role_unmapped": list(spec["role_unmapped"]),
                "obs_channel_sign_flips": list(spec["obs_channel_sign_flips"]),
                "action_channel_sign_flips": list(spec["action_channel_sign_flips"]),
            }
        if track_user_overrides:
            _state["user_overrides"][fid] = dict(spec)
        added += 1
    return added


# ---------------------------------------------------------------------------
# Public API: load / overlay
# ---------------------------------------------------------------------------


def load() -> int:
    """Load ``symmetry_catalog.json`` into :data:`_state`. Returns family count.

    Idempotent — repeat calls short-circuit on the cached state. Pattern
    mirrors :func:`ir.load` / :func:`pd_groups.load` / :func:`families.load`.
    """
    if _state["loaded"]:
        return len(_state["families"])

    payload = read_data(_CATALOG_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"symmetry_catalog.json failed to parse: {_CATALOG_PATH}"
        )

    _state["version"] = str(payload.get("version", ""))
    families_block = payload.get("families")
    _load_payload_into_state(
        families_block if isinstance(families_block, dict) else {},
        source=str(_CATALOG_PATH),
    )
    _state["loaded"] = True
    return len(_state["families"])


def merge_user_extensions(payload: Dict[str, Any]) -> int:
    """Merge a user overlay payload into the catalog cache.

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/symmetry_custom.json``)::

        {
          "version": "1.0.0",
          "families": {
            "<fid>": {
              "alias_of": null,
              "role_swaps":     [[L, R], ...],
              "role_self_map":  [...],
              "role_unmapped":  [...],
              "obs_channel_sign_flips":    [],
              "action_channel_sign_flips": []
            }
          }
        }

    Returns the number of families added / overridden. Per-entry validation
    (L*) raises RuntimeError on malformed overlay so a typo cannot silently
    destabilise the catalog (fail-loud invariant). Cross-family rules S1/S3/S4
    are re-checked by :meth:`RegistryHub.validate`.
    """
    if not _state["loaded"]:
        load()
    if not isinstance(payload, dict):
        return 0
    families_block = payload.get("families")
    if not isinstance(families_block, dict):
        return 0
    return _load_payload_into_state(
        families_block,
        source=str(_user_custom_path()),
        track_user_overrides=True,
    )


def get_version() -> str:
    """Return the loaded catalog file version string."""
    if not _state["loaded"]:
        load()
    return str(_state["version"])


# ---------------------------------------------------------------------------
# Public API: queries
# ---------------------------------------------------------------------------


def list_families() -> List[str]:
    """All declared family ids in catalog insertion order."""
    if not _state["loaded"]:
        load()
    return list(_state["families"].keys())


def has_symmetry(family: str) -> bool:
    """True iff symmetry data is declared (and alias chain resolves) for ``family``.

    Returns False ONLY when ``family`` is absent from the catalog — that case
    is the legitimate "no symmetry data shipped for this family yet"
    signal for callers that have a meaningful fallback (e.g. an apply_mirror
    consumer that skips silently).

    Catalog corruption (broken alias chain — cycle or missing target) RAISES
    :class:`SymmetryValidationError` rather than silently returning False;
    that case must be surfaced by :meth:`RegistryHub.validate`, not masked
    here.
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        return False
    # If chain is broken, raise instead of swallowing — registry corruption
    # must surface (fail-loud invariant).
    _resolve_alias_chain(fid)
    return True


def get_symmetry(family: str) -> SymmetryData:
    """Return resolved :class:`SymmetryData` for ``family``.

    Raises :class:`SymmetryValidationError` when ``family`` is not declared
    in the catalog (NOT a soft Optional — callers with unknown families
    are a bug we want to fail loud on).
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        raise SymmetryValidationError(
            f"get_symmetry: {family!r} not in symmetry_catalog.json; "
            f"declared families: {sorted(_state['families'].keys())}"
        )
    chain = _resolve_alias_chain(fid)
    root = chain[-1]
    root_spec = _state["families"][root]
    if root_spec.get("alias_of"):
        # Defensive — _resolve_alias_chain walks until alias_of is falsy,
        # so this only fires if the resolved root itself declared alias_of
        # to a missing target after we left the chain. Should be caught
        # by validate(); raise as belt-and-braces.
        raise SymmetryValidationError(
            f"get_symmetry: chain for {family!r} resolves to alias-only "
            f"entry {root!r}; RegistryHub.validate should have flagged "
            f"this — please re-run."
        )
    declared_alias = _state["families"][fid].get("alias_of")
    return SymmetryData(
        family=fid,
        alias_of=declared_alias,
        resolved_from=root,
        role_swaps=tuple(tuple(pair) for pair in root_spec["role_swaps"]),
        role_self_map=tuple(root_spec["role_self_map"]),
        role_unmapped=tuple(root_spec["role_unmapped"]),
        obs_channel_sign_flips=tuple(root_spec["obs_channel_sign_flips"]),
        action_channel_sign_flips=tuple(root_spec["action_channel_sign_flips"]),
    )


def _resolve_alias_chain(family: str) -> List[str]:
    """Walk the ``alias_of`` chain from ``family`` to the data-bearing root.

    Returns ``[family, parent, ..., root]``. Raises
    :class:`SymmetryValidationError` on cycle or missing intermediate
    target (defensive belt for catalog corruption).
    """
    chain: List[str] = []
    visited: set = set()
    cur: Optional[str] = family
    while cur is not None:
        if cur in visited:
            raise SymmetryValidationError(
                f"_resolve_alias_chain: cycle detected on {family!r} — "
                f"chain={chain + [cur]!r}; this should have been caught "
                f"by RegistryHub.validate, please re-run it"
            )
        if cur not in _state["families"]:
            raise SymmetryValidationError(
                f"_resolve_alias_chain: chain from {family!r} reached "
                f"{cur!r} which is not declared in symmetry_catalog.json"
            )
        visited.add(cur)
        chain.append(cur)
        cur = _state["families"][cur].get("alias_of")
    return chain


# ---------------------------------------------------------------------------
# Cross-registry validation (called from RegistryHub.validate)
# ---------------------------------------------------------------------------


def validate() -> List[str]:
    """Cross-family integrity check. Returns problem strings; ``[]`` = OK.

    The caller (:meth:`RegistryHub.validate`) wraps a non-empty problem
    list into :class:`RegistryValidationError`.

    Rules:
      * S1: For every non-alias family, every IR role in the family's
        slice (via :func:`ir.list_roles`) appears in EXACTLY ONE of
        ``role_swaps`` / ``role_self_map`` / ``role_unmapped``. Duplicate
        across sets → problem; missing from all → problem.
      * S3: Every role referenced by ``role_*`` sets must exist in the
        family's IR canonical slice.
      * S4: ``alias_of`` target must exist + alias graph must be acyclic.

    L1-L7 (per-entry schema) were enforced at :func:`load` so they are
    not re-checked here.
    """
    from . import ir as _ir
    if not _state["loaded"]:
        load()
    problems: List[str] = []

    # S4: alias_of target exists + no cycles. Run first because cycle
    # detection prevents downstream S1 / S3 from walking corrupted chains.
    problems.extend(_validate_alias_chains())

    # S1 + S3: family slice partition (only for non-alias families).
    for fid, spec in _state["families"].items():
        if spec.get("alias_of"):
            # Alias entries — validation on root handles partition; only
            # the alias_of target check (above) applies.
            continue
        ir_roles_raw = [
            r.get("id", "") for r in _ir.list_roles(fid) if isinstance(r, dict)
        ]
        ir_roles = [rid for rid in ir_roles_raw if rid]
        if not ir_roles:
            # Family has no IR canonical roles — symmetry data has nothing
            # to mirror against. Informational, not a problem.
            continue
        ir_role_set = set(ir_roles)

        swap_roles: List[str] = []
        for pair in spec["role_swaps"]:
            swap_roles.extend(pair)
        self_map_roles = list(spec["role_self_map"])
        unmapped_roles = list(spec["role_unmapped"])

        # S3: every referenced role must be in the family's IR slice.
        for role in swap_roles + self_map_roles + unmapped_roles:
            if role not in ir_role_set:
                problems.append(
                    f"  symmetry: family={fid!r} references role {role!r} "
                    f"which is not in ir_canonical for {fid!r}; either "
                    f"remove the role from symmetry_catalog.json or "
                    f"extend ir_canonical (which is hand-edited, never "
                    f"auto-extended from imports)"
                )

        # S1: build per-role assignment map for duplicate + completeness check.
        role_assignment: Dict[str, List[str]] = {}
        for role in swap_roles:
            role_assignment.setdefault(role, []).append("role_swaps")
        for role in self_map_roles:
            role_assignment.setdefault(role, []).append("role_self_map")
        for role in unmapped_roles:
            role_assignment.setdefault(role, []).append("role_unmapped")

        # S1 duplicates.
        for role, sets in role_assignment.items():
            if len(sets) > 1:
                problems.append(
                    f"  symmetry: family={fid!r} role {role!r} appears in "
                    f"multiple sets {sets!r}; each role must appear in "
                    f"exactly one of role_swaps / role_self_map / "
                    f"role_unmapped"
                )

        # S1 missing roles (in IR slice but unassigned).
        missing = [r for r in ir_roles if r not in role_assignment]
        if missing:
            problems.append(
                f"  symmetry: family={fid!r} partition incomplete — IR "
                f"roles missing from {{role_swaps, role_self_map, "
                f"role_unmapped}}: {sorted(missing)}; add them to the "
                f"appropriate set in symmetry_catalog.json"
            )

    return problems


def _validate_alias_chains() -> List[str]:
    """S4: ``alias_of`` target exists + no cycles. Returns problem strings."""
    problems: List[str] = []
    families = _state["families"]

    # S4a: target exists.
    for fid, spec in families.items():
        target = spec.get("alias_of")
        if target is None:
            continue
        if target not in families:
            problems.append(
                f"  symmetry: family={fid!r} alias_of={target!r} but "
                f"target is not declared in symmetry_catalog.json; fix "
                f"typo or add the missing family"
            )

    # S4b: cycle detection (tri-state DFS — same shape as families._detect_cycles).
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {fid: WHITE for fid in families}
    found_cycles: set = set()

    def dfs(node: str, path: List[str]) -> None:
        color[node] = GRAY
        path.append(node)
        parent = families[node].get("alias_of")
        if parent and parent in families:
            if color[parent] == GRAY:
                cycle_start = path.index(parent)
                cycle = path[cycle_start:] + [parent]
                core = cycle[:-1]
                if core:
                    pivot = core.index(min(core))
                    rotated = core[pivot:] + core[:pivot]
                    key = tuple(rotated)
                    if key not in found_cycles:
                        found_cycles.add(key)
                        problems.append(
                            f"  symmetry: alias cycle detected: "
                            f"{' -> '.join(rotated + [rotated[0]])}; "
                            f"break the cycle in symmetry_catalog.json"
                        )
            elif color[parent] == WHITE:
                dfs(parent, path)
        path.pop()
        color[node] = BLACK

    for fid in families:
        if color[fid] == WHITE:
            dfs(fid, [])

    return problems


# ---------------------------------------------------------------------------


__all__ = [
    "SymmetryData",
    "SymmetryValidationError",
    "load",
    "merge_user_extensions",
    "list_families",
    "get_symmetry",
    "has_symmetry",
    "validate",
    "get_version",
]
