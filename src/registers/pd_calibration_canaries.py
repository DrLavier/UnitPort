# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.pd_calibration_canaries — Per-family PD calibration canary group registry.

Purpose
-------
- Single source of truth for the per-family canary group recipe that
  ``sim2sim_calibration.run_calibration`` uses for its ``(omega_n, zeta)``
  round-trip check on the joints most sensitive to mass-matrix coupling
  (``knee`` + ``hip_y`` for legged platforms; ``elbow`` + ``wrist`` for
  manipulators).
- Each list entry is a ``pd_groups`` group id (e.g. ``"knee"``, **NOT** an
  IR role like ``"knee_L"``). Resolution from group id to specific joint
  indices is the consumer's job, performed via
  ``application.physics.joint_group_resolver.expand_groups_to_joints``.

Safety contract — asymmetric overlay
------------------------------------
The cross-engine TORQUE check in ``run_calibration`` always runs over all
joints; this canary registry only scopes the (omega_n, zeta) round-trip
gate. The factory list singles out joints with the largest mass off-
diagonal coupling; these joints must always be in the gate. User overlays
can therefore ONLY EXTEND a family's canary list — they cannot SHRINK
it. A shrinking overlay is rejected at merge time with
:class:`PDCalibrationValidationError`, so an overlay cannot silently
exempt sentinel joints from the round-trip check.

Adding a family or extending one in user overlay is allowed (more checks
= safer). Removing factory groups is not. The factory snapshot is held
in :data:`_state['factory_canaries']` and consulted by
:func:`merge_user_extensions` before accepting the overlay.

alias_of vs inherits_from
-------------------------
- ``alias_of`` here = **data identity** at load time (hard expansion, no
  override). Mirrors :func:`ir._resolve_alias` /
  :func:`pd_groups._resolve_alias` / :class:`symmetry.SymmetryData` pattern.
- ``inherits_from`` (from ``families_catalog.json``) = **dispatch fallback
  chain** opt-in by the caller. A different concept; this module does NOT
  use ``inherits_from``.
- ``humanoid`` declares ``alias_of: "biped"`` here, mirroring
  ``ir_canonical``, ``pd_groups_definitions``, and ``symmetry_catalog``.
  The four-layer alignment is a by-design observation, NOT a cross-
  registry hard constraint — each layer is validated independently.

Data layout
-----------
- Factory catalog: ``registers/data/pd_calibration_canaries.json``
  (read-only at runtime).
- User overlay: ``<USER_CONFIG_DIR>/registers/pd_calibration_canaries_custom.json``
  merged at load by ``RegistryHub._merge_user_overlays`` →
  :func:`merge_user_extensions`.
- The overlay can add new families OR extend an existing family's canary
  list; same load-time L-rule validation applies, plus the factory-
  subset rule enforced at merge time.

Fail-loud — no silent fallbacks
-------------------------------
- Load-time L1-L5 raise on malformed catalog.
- Cross-registry :func:`validate` returns problem list for S1/S2; the
  hub wraps non-empty problem list into ``RegistryValidationError``.
- Runtime API errors raise :class:`PDCalibrationValidationError` with
  actionable messages.
- :func:`get_canaries` raises on absent family (no soft Optional).
- :func:`has_canaries` returns False ONLY when the family is absent from
  the catalog; catalog corruption (broken alias chain) raises instead of
  silently returning False — that case must surface via
  ``RegistryHub.validate``, not get masked.

API
---
::

    load() -> int
    merge_user_extensions(payload) -> int
    list_families() -> list[str]
    get_canaries(family) -> CanaryData
    has_canaries(family) -> bool
    validate() -> list[str]
    get_version() -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_debug, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CATALOG_PATH = _DATA_DIR / "pd_calibration_canaries.json"
_USER_CUSTOM_REL = "registers/pd_calibration_canaries_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user overlay path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart (mirrors :func:`registers.symmetry._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "pd_calibration_canaries_custom.json"


# ---------------------------------------------------------------------------
# Public dataclasses / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryData:
    """Resolved canary data for one family.

    ``@dataclass(frozen=True)`` so downstream consumers
    (``sim2sim_calibration.run_calibration``) can cache safely. Tuple
    shape (instead of list) reflects the immutability contract — the
    canary set is a safety invariant; runtime mutation would be a bug.

    Fields:
        family: family id passed to :func:`get_canaries`. When the query
            target is an alias, this is still the queried id.
        alias_of: ``alias_of`` value declared on the QUERIED family entry
            (``None`` if the queried family had its own ``canary_groups``).
        resolved_from: family id whose data this struct contains. Equal
            to ``family`` when not aliased; otherwise the chain root.
        canary_groups: tuple of pd_groups group ids participating in the
            (omega_n, zeta) round-trip gate. Empty tuple is legal (e.g.
            ``wheeled`` / ``generic`` declare no PD canary).
    """

    family: str
    alias_of: Optional[str]
    resolved_from: str
    canary_groups: Tuple[str, ...]


class PDCalibrationValidationError(RuntimeError):
    """Raised by the runtime API on:

    * unknown family in :func:`get_canaries`;
    * defensive belt for broken alias chain (cycle / missing target)
      encountered at runtime — :class:`RegistryHub.validate` should have
      caught it; this raise surfaces it if validate has not run;
    * user overlay that shrinks the factory canary set (asymmetric
      overlay rule — overlays may only extend, never shrink).

    Cross-registry validate() problems surface via
    ``RegistryValidationError`` from the hub instead — this exception
    is for runtime API contract violations only (mirrors
    :class:`families.FamiliesValidationError` /
    :class:`symmetry.SymmetryValidationError`).
    """


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------


_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    # fid -> spec dict:
    #   alias entries: {"alias_of": <str>}
    #   data entries:  {"alias_of": None, "canary_groups": [<str>, ...]}
    "families": {},
    # Snapshot of factory canary sets keyed by fid (resolved through alias).
    # Used by merge_user_extensions to enforce the factory-subset rule
    # before accepting an overlay. Populated at the end of load() so it
    # always reflects the factory state, even after subsequent overlays.
    "factory_canaries": {},
    # fid -> original on-disk spec from user overlay (round-trip preservation).
    "user_overrides": {},
}


# ---------------------------------------------------------------------------
# Per-entry schema validation (L1-L5)
# ---------------------------------------------------------------------------


def _validate_entry(fid: str, spec: Dict[str, Any], *, source: str) -> None:
    """Per-family schema validation. Raises RuntimeError on malformed entry.

    Covers:
      * L1: lowercase ASCII fid.
      * L2: spec is a dict.
      * L3: exactly one of ``canary_groups`` / ``alias_of`` is present
            (mutually exclusive — having both is an authoring bug).
      * L4: ``alias_of`` if present is a non-empty string.
      * L5: ``canary_groups`` if present is a list[str] (possibly empty).
    """
    # L1: lowercase ASCII fid.
    if fid != fid.lower() or not fid.isascii():
        raise RuntimeError(
            f"pd_calibration_canaries: family id {fid!r} must be lowercase "
            f"ASCII; rename in {source}"
        )
    # L2: spec is a dict.
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"pd_calibration_canaries: family={fid!r} in {source} is not a "
            f"dict (got {type(spec).__name__})"
        )
    has_alias = "alias_of" in spec and spec["alias_of"] is not None
    has_data = "canary_groups" in spec
    # L3: exactly one of canary_groups / alias_of (mutually exclusive).
    if has_alias and has_data:
        raise RuntimeError(
            f"pd_calibration_canaries: family={fid!r} declares both "
            f"alias_of and canary_groups — choose one (data identity alias "
            f"means the target's data is reused verbatim); in {source}"
        )
    if not has_alias and not has_data:
        raise RuntimeError(
            f"pd_calibration_canaries: family={fid!r} must declare either "
            f"canary_groups (a list, possibly empty) or alias_of (a family "
            f"id string); in {source}"
        )
    # L4: alias_of is a non-empty string.
    if has_alias:
        alias = spec["alias_of"]
        if not isinstance(alias, str) or not alias.strip():
            raise RuntimeError(
                f"pd_calibration_canaries: family={fid!r}.alias_of must be "
                f"a non-empty string (got {alias!r}) in {source}"
            )
        return
    # L5: canary_groups is a list of strings.
    groups = spec["canary_groups"]
    if not isinstance(groups, list):
        raise RuntimeError(
            f"pd_calibration_canaries: family={fid!r}.canary_groups must "
            f"be a list (got {type(groups).__name__}) in {source}"
        )
    for i, gid in enumerate(groups):
        if not isinstance(gid, str):
            raise RuntimeError(
                f"pd_calibration_canaries: family={fid!r}.canary_groups[{i}]="
                f"{gid!r} is not a string in {source}"
            )
        if not gid.strip():
            raise RuntimeError(
                f"pd_calibration_canaries: family={fid!r}.canary_groups[{i}] "
                f"is empty in {source}; drop the entry or use a valid group id"
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
            f"pd_calibration_canaries: 'families' must be a mapping in "
            f"{source} (got {type(families_block).__name__})"
        )
    added = 0
    for fid_raw, spec in families_block.items():
        fid = str(fid_raw)
        _validate_entry(fid, spec, source=source)
        if "alias_of" in spec and spec["alias_of"] is not None:
            _state["families"][fid] = {"alias_of": str(spec["alias_of"])}
        else:
            _state["families"][fid] = {
                "alias_of": None,
                "canary_groups": list(spec["canary_groups"]),
            }
        if track_user_overrides:
            _state["user_overrides"][fid] = dict(spec)
        added += 1
    return added


# ---------------------------------------------------------------------------
# Public API: load / overlay
# ---------------------------------------------------------------------------


def load() -> int:
    """Load ``pd_calibration_canaries.json`` into :data:`_state`.

    Returns the family entry count (alias + data combined).
    Idempotent — repeat calls short-circuit on the cached state. Pattern
    mirrors :func:`symmetry.load` / :func:`families.load`.
    """
    if _state["loaded"]:
        return len(_state["families"])

    payload = read_data(_CATALOG_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"pd_calibration_canaries.json failed to parse: {_CATALOG_PATH}"
        )

    _state["version"] = str(payload.get("version", ""))
    families_block = payload.get("families")
    _load_payload_into_state(
        families_block if isinstance(families_block, dict) else {},
        source=str(_CATALOG_PATH),
    )

    # Snapshot factory canary sets (resolved through alias) for the
    # asymmetric overlay rule. Done after load to capture the final
    # post-alias-expansion state without re-walking chains at merge time.
    _state["factory_canaries"] = _snapshot_factory_canaries()
    _state["loaded"] = True
    return len(_state["families"])


def merge_user_extensions(payload: Dict[str, Any]) -> int:
    """Merge a user overlay payload into the catalog cache.

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/pd_calibration_canaries_custom.json``)::

        {
          "version": "1.0.0",
          "families": {
            "<fid>": {
              "canary_groups": ["knee", "hip_y", "ankle_pitch", "new_group"]
            }
          }
        }

    Returns the number of families added / overridden.

    Enforces the asymmetric overlay rule: for any family already declared
    in the factory catalog, the overlay's resolved canary set must be a
    SUPERSET of the factory canary set. An overlay that drops any factory
    group raises :class:`PDCalibrationValidationError` — the entire
    overlay file is rejected so the canary registry is never left in a
    half-merged state.

    Per-entry validation (L1-L5) raises on malformed overlay so a typo
    cannot silently destabilise the catalog. Cross-registry rules S1/S2
    are re-checked by :meth:`RegistryHub.validate`.
    """
    if not _state["loaded"]:
        load()
    if not isinstance(payload, dict):
        return 0
    families_block = payload.get("families")
    if not isinstance(families_block, dict):
        return 0

    # First pass — schema + factory-subset check on every entry before
    # mutating state. If any entry violates either, we raise and the
    # catalog stays untouched. This is the merge-time enforcement of the
    # safety contract.
    factory = _state.get("factory_canaries", {})
    source = str(_user_custom_path())
    for fid_raw, spec in families_block.items():
        fid = str(fid_raw)
        _validate_entry(fid, spec, source=source)
        # Subset check only applies to families that exist in the factory
        # snapshot. Brand-new families introduced by the overlay are not
        # subject to any factory set (there is none).
        if fid not in factory:
            continue
        factory_set = set(factory[fid])
        if "alias_of" in spec and spec["alias_of"] is not None:
            # Overlay turns the family into an alias. Resolve target's
            # factory snapshot (if available) and check superset.
            target = str(spec["alias_of"])
            if target not in factory:
                raise PDCalibrationValidationError(
                    f"pd_calibration_canaries overlay: family={fid!r} aliases "
                    f"to {target!r} which has no factory canary snapshot, so "
                    f"the asymmetric-overlay safety check cannot verify the "
                    f"factory subset is preserved; reject"
                )
            overlay_set = set(factory[target])
        else:
            overlay_set = set(spec["canary_groups"])
        missing = factory_set - overlay_set
        if missing:
            raise PDCalibrationValidationError(
                f"pd_calibration_canaries overlay: family={fid!r} would drop "
                f"factory canary groups {sorted(missing)!r}. The canary set "
                f"is a safety invariant — overlays may only EXTEND it, never "
                f"SHRINK it (the omitted groups are the joints most sensitive "
                f"to mass-matrix mismatch). Remove the offending overlay "
                f"entry, or extend the list instead of replacing it"
            )

    # Second pass — apply (schema + subset are both clean by now).
    return _load_payload_into_state(
        families_block,
        source=source,
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


def has_canaries(family: str) -> bool:
    """True iff canary data is declared (and alias chain resolves) for ``family``.

    Returns False ONLY when ``family`` is absent from the catalog — that
    case is the legitimate "no canary entry for this family" signal for
    callers that have a meaningful fallback (e.g. a caller that
    fall-throughs to "skip calibration round-trip for this family").

    Catalog corruption (broken alias chain — cycle or missing target)
    RAISES :class:`PDCalibrationValidationError` rather than silently
    returning False; that case must be surfaced by
    :meth:`RegistryHub.validate`, not masked here.
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        return False
    # If chain is broken, raise instead of swallowing — registry
    # corruption must surface (fail-loud invariant).
    _resolve_alias_chain(fid)
    return True


def get_canaries(family: str) -> CanaryData:
    """Return resolved :class:`CanaryData` for ``family``.

    Raises :class:`PDCalibrationValidationError` when ``family`` is not
    declared in the catalog (NOT a soft Optional — callers with unknown
    families are a bug we want to fail loud on, matching the safety-
    sensitive nature of the calibration gate).
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        raise PDCalibrationValidationError(
            f"get_canaries: {family!r} not in pd_calibration_canaries.json; "
            f"declared families: {sorted(_state['families'].keys())}"
        )
    chain = _resolve_alias_chain(fid)
    root = chain[-1]
    root_spec = _state["families"][root]
    if root_spec.get("alias_of"):
        # Defensive — _resolve_alias_chain walks until alias_of is
        # falsy, so this only fires if the resolved root itself
        # declared alias_of to a missing target after we left the
        # chain. Should be caught by validate(); raise as
        # belt-and-braces.
        raise PDCalibrationValidationError(
            f"get_canaries: chain for {family!r} resolves to alias-only "
            f"entry {root!r}; RegistryHub.validate should have flagged "
            f"this — please re-run."
        )
    declared_alias = _state["families"][fid].get("alias_of")
    return CanaryData(
        family=fid,
        alias_of=declared_alias,
        resolved_from=root,
        canary_groups=tuple(root_spec["canary_groups"]),
    )


def _resolve_alias_chain(family: str) -> List[str]:
    """Walk the ``alias_of`` chain from ``family`` to the data-bearing root.

    Returns ``[family, parent, ..., root]``. Raises
    :class:`PDCalibrationValidationError` on cycle or missing intermediate
    target (defensive belt for catalog corruption).
    """
    chain: List[str] = []
    visited: set = set()
    cur: Optional[str] = family
    while cur is not None:
        if cur in visited:
            raise PDCalibrationValidationError(
                f"_resolve_alias_chain: cycle detected on {family!r} — "
                f"chain={chain + [cur]!r}; this should have been caught "
                f"by RegistryHub.validate, please re-run it"
            )
        if cur not in _state["families"]:
            raise PDCalibrationValidationError(
                f"_resolve_alias_chain: chain from {family!r} reached "
                f"{cur!r} which is not declared in "
                f"pd_calibration_canaries.json"
            )
        visited.add(cur)
        chain.append(cur)
        cur = _state["families"][cur].get("alias_of")
    return chain


def _snapshot_factory_canaries() -> Dict[str, List[str]]:
    """Build the {fid: canary_groups} factory snapshot, expanding aliases.

    Called from :func:`load` after the factory catalog has been parsed
    into :data:`_state['families']` but BEFORE any user overlay merges,
    so the snapshot reflects the on-disk factory state only.
    """
    snapshot: Dict[str, List[str]] = {}
    for fid in _state["families"]:
        try:
            chain = _resolve_alias_chain(fid)
        except PDCalibrationValidationError:
            # Catalog corruption — surface via validate(). Skip the entry
            # here so a single bad chain does not corrupt the snapshot of
            # the rest.
            continue
        root = chain[-1]
        root_spec = _state["families"][root]
        if root_spec.get("alias_of"):
            # Root itself is alias-only — caught by validate(). Skip.
            continue
        snapshot[fid] = list(root_spec["canary_groups"])
    return snapshot


# ---------------------------------------------------------------------------
# Cross-registry validation (called from RegistryHub.validate)
# ---------------------------------------------------------------------------


def validate() -> List[str]:
    """Cross-family integrity check. Returns problem strings; ``[]`` = OK.

    The caller (:meth:`RegistryHub.validate`) wraps a non-empty problem
    list into ``RegistryValidationError``.

    Rules:
      * S1: For every non-alias family, every group id in
        ``canary_groups`` must exist in the family's
        :func:`pd_groups.list_groups` set. The check uses the family
        from the canary catalog directly (alias resolution happens
        through ``_resolve_alias_chain`` — for alias families the root
        family's pd_groups are checked).
      * S2: Every family declared in the canary catalog must exist in
        :func:`families.list_families`. This keeps the canary registry
        aligned with the central family vocabulary.

    Soft rule (emitted as ``log_warning`` rather than added to problem
    list, so it does not block boot):
      * Any family known to ``pd_groups.list_families`` but absent from
        the canary catalog gets a one-line warning. ``wheeled`` /
        ``generic`` already declare empty arrays to opt out of the
        warning loud-and-clear, so the warning fires only on omissions.

    L1-L5 (per-entry schema) and the asymmetric-overlay subset rule
    were enforced at :func:`load` / :func:`merge_user_extensions` so they
    are not re-checked here.
    """
    from . import pd_groups as _pd_groups
    from . import families as _families
    if not _state["loaded"]:
        load()
    problems: List[str] = []

    # S4-style: alias_of target exists + no cycles. Same shape as
    # symmetry._validate_alias_chains; run first because cycle detection
    # prevents downstream S1 from walking corrupted chains.
    problems.extend(_validate_alias_chains())

    # S2: every catalog family must be in families_catalog.json.
    catalog_families = set(_families.list_families())
    for fid in _state["families"]:
        if fid not in catalog_families:
            problems.append(
                f"  pd_calibration_canaries: family={fid!r} is declared in "
                f"pd_calibration_canaries.json but not in families_catalog.json; "
                f"add the family to families_catalog.json or remove the "
                f"canary entry"
            )

    # S1: every canary group must exist in pd_groups[resolved_family]
    # group set. For alias entries the resolved root family's pd_groups
    # are used (matches runtime behaviour of get_canaries).
    for fid, spec in _state["families"].items():
        if spec.get("alias_of"):
            # Alias entries — S1 is enforced on the root only (alias
            # data is identical by definition).
            continue
        # Resolve which pd_groups family slice applies. The canary
        # registry's family id should map directly to the pd_groups
        # family id (one-to-one); ``pd_groups.list_groups`` returns the
        # post-alias-resolved group set for that family.
        pd_groups_for_family = _pd_groups.list_groups(fid)
        if not pd_groups_for_family:
            # pd_groups declares no groups for this family — informational
            # (e.g. ``generic`` family). Canary should be empty too, which
            # the next loop will verify implicitly (any non-empty canary
            # on a no-pd-groups family raises S1).
            if spec["canary_groups"]:
                problems.append(
                    f"  pd_calibration_canaries: family={fid!r} declares "
                    f"canary_groups={spec['canary_groups']!r} but pd_groups "
                    f"declares no groups for the family; either drop the "
                    f"canary entries or add the groups to "
                    f"pd_groups_definitions.json"
                )
            continue
        pd_group_ids = {g.get("id", "") for g in pd_groups_for_family if isinstance(g, dict)}
        pd_group_ids.discard("")
        for gid in spec["canary_groups"]:
            if gid not in pd_group_ids:
                problems.append(
                    f"  pd_calibration_canaries: family={fid!r} canary group "
                    f"{gid!r} is not a defined pd_groups group for the "
                    f"family. pd_groups[{fid!r}] declares: "
                    f"{sorted(pd_group_ids)!r}. Add the group to "
                    f"pd_groups_definitions.json or correct the canary id"
                )

    # Soft warning — pd_groups family present, canary catalog absent.
    # Emitted via log_warning (not added to problems) so boot is not
    # blocked; the omission is a coverage gap, not a corruption.
    canary_fams = set(_state["families"].keys())
    for pf in _pd_groups.list_families():
        if pf not in canary_fams:
            log_warning(
                f"[pd_calibration_canaries] family={pf!r} has pd_groups "
                f"definitions but no canary entry — declare canary_groups: [] "
                f"explicitly if the family has no PD canary, or add the "
                f"appropriate groups"
            )

    return problems


def _validate_alias_chains() -> List[str]:
    """``alias_of`` target exists + no cycles. Returns problem strings.

    Same tri-state DFS shape as :func:`symmetry._validate_alias_chains`.
    """
    problems: List[str] = []
    families = _state["families"]

    # Target exists.
    for fid, spec in families.items():
        target = spec.get("alias_of")
        if target is None:
            continue
        if target not in families:
            problems.append(
                f"  pd_calibration_canaries: family={fid!r} alias_of="
                f"{target!r} but target is not declared in "
                f"pd_calibration_canaries.json; fix typo or add the missing "
                f"family"
            )

    # Cycle detection (tri-state DFS — same shape as
    # families._detect_cycles / symmetry._validate_alias_chains).
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
                            f"  pd_calibration_canaries: alias cycle detected: "
                            f"{' -> '.join(rotated + [rotated[0]])}; break "
                            f"the cycle in pd_calibration_canaries.json"
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
    "CanaryData",
    "PDCalibrationValidationError",
    "load",
    "merge_user_extensions",
    "list_families",
    "get_canaries",
    "has_canaries",
    "validate",
    "get_version",
]
