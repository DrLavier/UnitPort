# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.amp_obs_templates — Per-family AMP discriminator observation template registry.

Purpose
-------
- Single source of truth for the per-family AMP observation term list
  that the discriminator consumes. Each entry's ``template_terms`` is an
  ordered tuple of term names registered in
  :mod:`application.training.amp.obs_terms`; the consumer (motion-side
  ``compute_amp_obs_from_motion`` / env-side ``compute_amp_obs_from_env``)
  walks the tuple in order to assemble the obs vector. Term-name order is
  load-bearing: a permutation of the same terms produces a different
  vector layout, which a downstream discriminator separates trivially on
  index drift.
- Replaces the historical ``DEFAULT_QUADRUPED_TERMS`` hardcoded constant
  with a family-keyed dispatch table. Callers select the template via
  :func:`get_obs_template` (or the families-side
  :func:`registers.families.resolve_family_dispatch`) keyed on
  ``robot.families[0]``, never on a ``family == "..."`` string compare.

Design boundaries
-----------------
- Term names must be a subset of
  :func:`application.training.amp.obs_terms.list_terms`. Templates do not
  define new terms; they only choose from the registry. New term names
  require a registration in ``obs_terms.py``.
- ``template_terms`` are term NAMES (e.g. ``"joint_pos"``), NOT IR roles
  (e.g. ``"knee_L"``). Per-DoF dimensionality of each term resolves from
  the obs-term registry's ``dim_fn`` against the live robot morphology
  (``num_dofs`` / ``num_feet`` ctx); this registry has no opinion on
  per-joint membership.
- Empty ``template_terms`` is legal — declares a family "registered but
  not AMP-trainable" (mirrors ``pd_calibration_canaries``'s empty arrays
  for wheeled / generic).

alias_of vs inherits_from
-------------------------
- ``alias_of`` here = **data identity** at load time (hard expansion, no
  override). Mirrors :func:`registers.ir._resolve_alias` /
  :func:`registers.pd_groups._resolve_alias` /
  :class:`registers.symmetry.SymmetryData` /
  :class:`registers.pd_calibration_canaries.CanaryData` pattern.
- ``inherits_from`` (from ``families_catalog.json``) = **dispatch fallback
  chain** opt-in by the caller via
  ``resolve_family_dispatch(allow_fallback=True)``. A different concept;
  this module does NOT use ``inherits_from``.
- ``humanoid`` declares ``alias_of: "biped"`` here, mirroring
  ``ir_canonical``, ``pd_groups_definitions``, ``symmetry_catalog``, and
  ``pd_calibration_canaries`` (five-layer alignment is by-design
  observation, NOT a cross-registry hard constraint — each layer
  validates independently).

Data layout
-----------
- Factory catalog: ``registers/data/amp_obs_templates.json``
  (read-only at runtime).
- User overlay: ``<USER_CONFIG_DIR>/registers/amp_obs_templates_custom.json``
  merged at load by ``RegistryHub._merge_user_overlays`` ->
  :func:`merge_user_extensions`.
- The overlay can add new families OR override existing ones; the same
  load-time L1-L5 validation applies.

Fail-loud — no silent fallbacks
-------------------------------
- Load-time L1-L5 raise on malformed catalog.
- Cross-registry :func:`validate` returns problem list for A1/A2/A3; the
  hub wraps non-empty problem list into
  :class:`RegistryValidationError`.
- Runtime API errors raise :class:`AmpObsTemplatesValidationError` with
  actionable messages.
- :func:`get_obs_template` raises on absent family (no soft Optional).
- :func:`has_obs_template` returns False ONLY when the family is absent;
  catalog corruption (broken alias chain) raises instead of silently
  returning False -- that case must surface via
  :meth:`RegistryHub.validate`, not get masked.

API
---
::

    load() -> int
    merge_user_extensions(payload) -> int
    list_families() -> list[str]
    get_obs_template(family) -> ObsTemplate
    has_obs_template(family) -> bool
    validate() -> list[str]
    get_version() -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_debug, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CATALOG_PATH = _DATA_DIR / "amp_obs_templates.json"
_USER_CUSTOM_REL = "registers/amp_obs_templates_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user overlay path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart (mirrors :func:`registers.pd_calibration_canaries._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "amp_obs_templates_custom.json"


# ---------------------------------------------------------------------------
# Public dataclasses / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObsTemplate:
    """Resolved AMP obs template for one family.

    ``@dataclass(frozen=True)`` so downstream consumers (launcher,
    motion-clip term lookup) can cache safely. ``Tuple`` shape (instead of
    list) reflects the immutability contract — term-name order is
    load-bearing for byte-identity between motion and env sides, so
    runtime mutation would be a bug.

    Fields:
        family: family id passed to :func:`get_obs_template`. When the
            query target is an alias, this is still the queried id.
        alias_of: ``alias_of`` value declared on the QUERIED family entry
            (``None`` if the queried family had its own ``template_terms``).
        resolved_from: family id whose data this struct contains. Equal
            to ``family`` when not aliased; otherwise the chain root.
        template_terms: ordered tuple of AMP obs term names. Each entry
            must exist in :func:`application.training.amp.obs_terms.list_terms`
            (verified at :func:`validate` time, Rule A1). Empty tuple is
            legal (e.g. ``wheeled`` / ``generic`` / ``manipulator``
            declare empty arrays to opt out of AMP training).
    """

    family: str
    alias_of: Optional[str]
    resolved_from: str
    template_terms: Tuple[str, ...]


class AmpObsTemplatesValidationError(RuntimeError):
    """Raised by the runtime API on:

    * unknown family in :func:`get_obs_template`;
    * defensive belt for broken alias chain (cycle / missing target)
      encountered at runtime -- :class:`RegistryHub.validate` should
      have caught it; this raise surfaces it if validate has not run.

    Cross-registry validate() problems surface via
    :class:`RegistryValidationError` from the hub instead -- this
    exception is for runtime API contract violations only (mirrors
    :class:`registers.families.FamiliesValidationError` /
    :class:`registers.pd_calibration_canaries.PDCalibrationValidationError`).
    """


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------


_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    # fid -> spec dict:
    #   alias entries: {"alias_of": <str>}
    #   data entries:  {"alias_of": None, "template_terms": [<str>, ...]}
    "families": {},
    # fid -> original on-disk spec from user overlay (round-trip preservation;
    # ``families`` already has the merged view).
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
      * L3: exactly one of ``template_terms`` / ``alias_of`` is present
            (mutually exclusive -- having both is an authoring bug).
      * L4: ``alias_of`` if present is a non-empty string.
      * L5: ``template_terms`` if present is a list[str] (possibly empty).
    """
    # L1: lowercase ASCII fid.
    if fid != fid.lower() or not fid.isascii():
        raise RuntimeError(
            f"amp_obs_templates: family id {fid!r} must be lowercase "
            f"ASCII; rename in {source}"
        )
    # L2: spec is a dict.
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"amp_obs_templates: family={fid!r} in {source} is not a "
            f"dict (got {type(spec).__name__})"
        )
    has_alias = "alias_of" in spec and spec["alias_of"] is not None
    has_data = "template_terms" in spec
    # L3: exactly one of template_terms / alias_of (mutually exclusive).
    if has_alias and has_data:
        raise RuntimeError(
            f"amp_obs_templates: family={fid!r} declares both alias_of "
            f"and template_terms -- choose one (data identity alias "
            f"means the target's data is reused verbatim); in {source}"
        )
    if not has_alias and not has_data:
        raise RuntimeError(
            f"amp_obs_templates: family={fid!r} must declare either "
            f"template_terms (a list of obs term names, possibly empty) "
            f"or alias_of (a family id string); in {source}"
        )
    # L4: alias_of is a non-empty string.
    if has_alias:
        alias = spec["alias_of"]
        if not isinstance(alias, str) or not alias.strip():
            raise RuntimeError(
                f"amp_obs_templates: family={fid!r}.alias_of must be a "
                f"non-empty string (got {alias!r}) in {source}"
            )
        return
    # L5: template_terms is a list of strings.
    terms = spec["template_terms"]
    if not isinstance(terms, list):
        raise RuntimeError(
            f"amp_obs_templates: family={fid!r}.template_terms must be "
            f"a list (got {type(terms).__name__}) in {source}"
        )
    for i, tname in enumerate(terms):
        if not isinstance(tname, str):
            raise RuntimeError(
                f"amp_obs_templates: family={fid!r}.template_terms[{i}]="
                f"{tname!r} is not a string in {source}"
            )
        if not tname.strip():
            raise RuntimeError(
                f"amp_obs_templates: family={fid!r}.template_terms[{i}] "
                f"is empty in {source}; drop the entry or use a valid "
                f"obs term name"
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
            f"amp_obs_templates: 'families' must be a mapping in "
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
                "template_terms": list(spec["template_terms"]),
            }
        if track_user_overrides:
            _state["user_overrides"][fid] = dict(spec)
        added += 1
    return added


# ---------------------------------------------------------------------------
# Public API: load / overlay
# ---------------------------------------------------------------------------


def load() -> int:
    """Load ``amp_obs_templates.json`` into :data:`_state`.

    Returns the family entry count (alias + data combined). Idempotent --
    repeat calls short-circuit on the cached state. Pattern mirrors
    :func:`registers.pd_calibration_canaries.load` /
    :func:`registers.families.load`.
    """
    if _state["loaded"]:
        return len(_state["families"])

    payload = read_data(_CATALOG_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"amp_obs_templates.json failed to parse: {_CATALOG_PATH}"
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

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/amp_obs_templates_custom.json``)::

        {
          "version": "1.0.0",
          "families": {
            "<fid>": {
              "template_terms": ["joint_pos", "joint_vel", ...]
            }
          }
        }

    Returns the number of families added / overridden.

    Per-entry validation (L1-L5) raises on malformed overlay so a typo
    cannot silently destabilise the catalog. Cross-registry rules A1-A3
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


def has_obs_template(family: str) -> bool:
    """True iff an obs template is declared (and alias chain resolves) for ``family``.

    Returns False ONLY when ``family`` is absent from the catalog -- that
    case is the legitimate "no obs template entry for this family" signal
    for callers that have a meaningful fallback.

    Catalog corruption (broken alias chain -- cycle or missing target)
    RAISES :class:`AmpObsTemplatesValidationError` rather than silently
    returning False; that case must be surfaced by
    :meth:`RegistryHub.validate`, not masked here.
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        return False
    # If chain is broken, raise instead of swallowing -- registry
    # corruption must surface (fail-loud invariant).
    _resolve_alias_chain(fid)
    return True


def get_obs_template(family: str) -> ObsTemplate:
    """Return resolved :class:`ObsTemplate` for ``family``.

    Raises :class:`AmpObsTemplatesValidationError` when ``family`` is not
    declared in the catalog (NOT a soft Optional -- callers with unknown
    families are a bug we want to fail loud on, matching the consistency-
    sensitive nature of AMP obs construction).
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        raise AmpObsTemplatesValidationError(
            f"get_obs_template: {family!r} not in amp_obs_templates.json; "
            f"declared families: {sorted(_state['families'].keys())}"
        )
    chain = _resolve_alias_chain(fid)
    root = chain[-1]
    root_spec = _state["families"][root]
    if root_spec.get("alias_of"):
        # Defensive -- _resolve_alias_chain walks until alias_of is
        # falsy, so this only fires if the resolved root itself declared
        # alias_of to a missing target after we left the chain. Should
        # be caught by validate(); raise as belt-and-braces.
        raise AmpObsTemplatesValidationError(
            f"get_obs_template: chain for {family!r} resolves to "
            f"alias-only entry {root!r}; RegistryHub.validate should have "
            f"flagged this -- please re-run."
        )
    declared_alias = _state["families"][fid].get("alias_of")
    return ObsTemplate(
        family=fid,
        alias_of=declared_alias,
        resolved_from=root,
        template_terms=tuple(root_spec["template_terms"]),
    )


def _resolve_alias_chain(family: str) -> List[str]:
    """Walk the ``alias_of`` chain from ``family`` to the data-bearing root.

    Returns ``[family, parent, ..., root]``. Raises
    :class:`AmpObsTemplatesValidationError` on cycle or missing
    intermediate target (defensive belt for catalog corruption).
    """
    chain: List[str] = []
    visited: set = set()
    cur: Optional[str] = family
    while cur is not None:
        if cur in visited:
            raise AmpObsTemplatesValidationError(
                f"_resolve_alias_chain: cycle detected on {family!r} -- "
                f"chain={chain + [cur]!r}; this should have been caught "
                f"by RegistryHub.validate, please re-run it"
            )
        if cur not in _state["families"]:
            raise AmpObsTemplatesValidationError(
                f"_resolve_alias_chain: chain from {family!r} reached "
                f"{cur!r} which is not declared in "
                f"amp_obs_templates.json"
            )
        visited.add(cur)
        chain.append(cur)
        cur = _state["families"][cur].get("alias_of")
    return chain


# ---------------------------------------------------------------------------
# Cross-registry validation (called from RegistryHub.validate)
# ---------------------------------------------------------------------------


def validate() -> List[str]:
    """Cross-registry integrity check. Returns problem strings; ``[]`` = OK.

    The caller (:meth:`RegistryHub.validate`) wraps a non-empty problem
    list into :class:`RegistryValidationError`.

    Rules:
      * A1: For every non-alias family, every term name in
        ``template_terms`` must exist in
        :func:`application.training.amp.obs_terms.list_terms`. The check
        uses the family from the obs-templates catalog directly (alias
        resolution happens through ``_resolve_alias_chain`` -- for alias
        families the root family's terms are checked).
      * A2: Every family declared in the obs-templates catalog must
        exist in :func:`registers.families.list_families`. Keeps the
        registry aligned with the central family vocabulary.
      * A3: ``alias_of`` target exists + no cycles (same tri-state DFS
        shape as :func:`registers.pd_calibration_canaries.
        _validate_alias_chains` /
        :func:`registers.families._detect_cycles`).

    Soft rule (emitted as ``log_warning`` rather than added to problem
    list, so it does not block boot):
      * A4: any family declared in ``families_catalog.json`` but absent
        from the obs-templates catalog gets a one-line warning.
        ``wheeled`` / ``generic`` / ``manipulator`` already declare
        empty arrays to opt out of the warning loud-and-clear, so the
        warning fires only on omissions.

    L1-L5 (per-entry schema) were enforced at :func:`load` /
    :func:`merge_user_extensions` so they are not re-checked here.
    """
    from . import families as _families
    if not _state["loaded"]:
        load()
    problems: List[str] = []

    # A3-first: alias_of target exists + no cycles. Same shape as
    # pd_calibration_canaries._validate_alias_chains; run first because
    # cycle detection prevents downstream A1 from walking corrupted chains.
    problems.extend(_validate_alias_chains())

    # A2: every catalog family must be in families_catalog.json.
    catalog_families = set(_families.list_families())
    for fid in _state["families"]:
        if fid not in catalog_families:
            problems.append(
                f"  amp_obs_templates: family={fid!r} is declared in "
                f"amp_obs_templates.json but not in families_catalog.json; "
                f"add the family to families_catalog.json or remove the "
                f"obs template entry"
            )

    # A1: every term name must exist in the obs-terms registry. Import
    # at call time (not module top) so RegistryHub.load_all() does not
    # eagerly pull the AMP backend during early boot when only registry
    # data is being read.
    from application.training.amp.obs_terms import list_terms as _list_obs_terms
    registered_terms = set(_list_obs_terms())
    for fid, spec in _state["families"].items():
        if spec.get("alias_of"):
            # Alias entries -- A1 is enforced on the root only (alias
            # data is identical by definition).
            continue
        for tname in spec["template_terms"]:
            if tname not in registered_terms:
                problems.append(
                    f"  amp_obs_templates: family={fid!r} template term "
                    f"{tname!r} is not registered in "
                    f"application.training.amp.obs_terms. Registered "
                    f"terms: {sorted(registered_terms)!r}. Register the "
                    f"term via amp.obs_terms.register() or correct the "
                    f"name in amp_obs_templates.json"
                )

    # A4 (soft warn) -- families_catalog declares a family but obs
    # templates catalog has no entry. Emitted via log_warning (not added
    # to problems) so boot is not blocked; the omission is a coverage
    # gap, not a corruption.
    template_fams = set(_state["families"].keys())
    for cf in catalog_families:
        if cf not in template_fams:
            log_warning(
                f"[amp_obs_templates] family={cf!r} is declared in "
                f"families_catalog.json but has no obs template entry -- "
                f"declare template_terms: [] explicitly if the family is "
                f"not AMP-trainable, or add the appropriate term list"
            )

    return problems


def _validate_alias_chains() -> List[str]:
    """``alias_of`` target exists + no cycles. Returns problem strings.

    Same tri-state DFS shape as
    :func:`registers.pd_calibration_canaries._validate_alias_chains` /
    :func:`registers.families._detect_cycles`.
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
                f"  amp_obs_templates: family={fid!r} alias_of="
                f"{target!r} but target is not declared in "
                f"amp_obs_templates.json; fix typo or add the missing "
                f"family"
            )

    # Cycle detection (tri-state DFS).
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
                            f"  amp_obs_templates: alias cycle detected: "
                            f"{' -> '.join(rotated + [rotated[0]])}; "
                            f"break the cycle in amp_obs_templates.json"
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
    "ObsTemplate",
    "AmpObsTemplatesValidationError",
    "load",
    "merge_user_extensions",
    "list_families",
    "get_obs_template",
    "has_obs_template",
    "validate",
    "get_version",
]
