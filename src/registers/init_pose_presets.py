# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.init_pose_presets -- Per-family init pose preset registry.

Purpose
-------
- Single source of truth for the family-keyed init pose preset name
  catalog. Each entry declares the family-level preset NAMES + base_pos
  + description; the ``joint_pos_by_ir`` field is intentionally absent
  at the family layer because init pose joint angles are per-robot
  physical quantities (a quadruped "stand" pose's thigh / calf angles
  depend on segment lengths, so Go2's 0.9 / -1.8 is not a valid
  quadruped canonical -- §8 forbids the silent fallback that would let
  the same numbers appear on Spot / ANYmal at a different leg geometry).
- The SKU-override layer lives in
  ``robots_canonical.json[sku].init_pose_presets`` and carries the
  ``joint_pos_by_ir`` per SKU. Three-layer merge
  (family-default -> SKU-override -> user) happens in
  :class:`application.service.robot_init_poses.InitPoseService`;
  selecting a family preset on a SKU that has no SKU-override is
  fail-loud at apply time (no silent fallback to zero / family-default
  angles -- the registry intentionally has none to fall back to).

Design boundaries
-----------------
- This registry carries NAMES, descriptions, and the base_pos
  (translation-only); it does NOT carry per-joint angles. Callers
  needing the full pose tuple read through
  :class:`InitPoseService` which performs the three-layer merge.
- ``InitPosePreset`` here is the registry-side dataclass (no
  ``source`` field). The runtime layered view kept by
  :class:`application.service.robot_init_poses.InitPoseService` uses
  its own :class:`application.service.robot_init_poses.model.InitPosePreset`
  with the ``source`` tag -- a deliberate layer separation (registry
  data shape != runtime layered view shape).
- Missing families are legal in the soft-warn sense: every family in
  ``families_catalog`` should declare at least one preset (warn at
  load), but no family is forced to. I6 surfaces the omission rather
  than fabricating an empty preset.

alias_of vs inherits_from
-------------------------
- ``alias_of`` here = **data identity** at load time (hard expansion,
  no override). Mirrors :func:`registers.ir._resolve_alias` /
  :func:`registers.pd_groups._resolve_alias` /
  :class:`registers.symmetry.SymmetryData` /
  :class:`registers.pd_calibration_canaries.CanaryData` /
  :class:`registers.amp_obs_templates.ObsTemplate` /
  :class:`registers.gait_commands.GaitCommandSpec` pattern.
- ``humanoid`` declares ``alias_of: "biped"`` here, mirroring the
  same alias in the six prior family-keyed registries -- seven-layer
  alignment is by-design observation, NOT a cross-registry hard
  constraint (each layer validates independently).

Data layout
-----------
- Factory catalog: ``registers/data/init_pose_presets_catalog.json``
  (read-only at runtime).
- User overlay: ``<USER_CONFIG_DIR>/registers/init_pose_presets_custom.json``
  merged at load by ``RegistryHub._merge_user_overlays`` ->
  :func:`merge_user_extensions`.
- The overlay can add new families OR override existing ones; the
  same load-time L1-L9 validation applies.

Fail-loud -- no silent fallbacks
--------------------------------
- Load-time L1-L9 raise on malformed catalog.
- Cross-registry :func:`validate` returns problem list for I1-I5; the
  hub wraps non-empty problem list into
  :class:`RegistryValidationError`.
- Runtime API errors raise :class:`InitPosePresetValidationError`
  with actionable messages.
- :func:`get_preset` raises on absent family or absent preset name.
- :func:`has_preset` returns False ONLY when the family or name is
  absent; catalog corruption (broken alias chain) raises instead of
  silently returning False -- that case must be surfaced by
  :meth:`RegistryHub.validate`, not get masked.

API
---
::

    load() -> int
    merge_user_extensions(payload) -> int
    list_families() -> list[str]
    list_presets(family) -> list[InitPosePreset]
    get_preset(family, name) -> InitPosePreset
    has_preset(family, name) -> bool
    validate() -> list[str]
    get_version() -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_debug, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CATALOG_PATH = _DATA_DIR / "init_pose_presets_catalog.json"
_USER_CUSTOM_REL = "registers/init_pose_presets_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user overlay path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart (mirrors :func:`registers.gait_commands._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "init_pose_presets_custom.json"


# ---------------------------------------------------------------------------
# Public dataclasses / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitPosePreset:
    """Resolved family-default init pose preset.

    ``@dataclass(frozen=True)`` so downstream consumers (canvas widget
    adapter, InitPoseService merger) can cache safely. The
    family layer carries name + base_pos + description only; the
    per-joint angles live in the SKU-override layer
    (``robots_canonical.json[sku].init_pose_presets``) and are merged
    by :class:`InitPoseService`.

    Fields:
        family: family id passed to :func:`get_preset` /
            :func:`list_presets`. When the query target is an alias,
            this is still the queried id.
        alias_of: ``alias_of`` value declared on the QUERIED family
            entry (``None`` if the queried family had its own data).
        resolved_from: family id whose data this struct contains.
            Equal to ``family`` when not aliased; otherwise the chain
            root.
        name: preset name unique within the resolved family.
        description: human-readable one-liner shown in the UI.
        base_pos: ``(x, y, z)`` in metres. Translation-only -- joint
            angles are SKU-override layer.
    """

    family: str
    alias_of: Optional[str]
    resolved_from: str
    name: str
    description: str
    base_pos: Tuple[float, float, float]


class InitPosePresetValidationError(RuntimeError):
    """Raised by the runtime API on:

    * unknown family in :func:`get_preset` / :func:`list_presets`;
    * unknown preset name in :func:`get_preset`;
    * defensive belt for broken alias chain (cycle / missing target)
      encountered at runtime -- :class:`RegistryHub.validate` should
      have caught it; this raise surfaces it if validate has not run.

    Cross-registry validate() problems surface via
    :class:`RegistryValidationError` from the hub instead -- this
    exception is for runtime API contract violations only (mirrors
    :class:`registers.gait_commands.GaitCommandValidationError`).
    """


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------


_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    # fid -> spec dict:
    #   alias entries: {"alias_of": <str>}
    #   data entries:  {
    #       "alias_of": None,
    #       "presets": [
    #           {"name": <str>, "description": <str>,
    #            "base_pos": [<float>, <float>, <float>]},
    #           ...
    #       ],
    #   }
    "families": {},
    # fid -> original on-disk spec from user overlay (round-trip
    # preservation; ``families`` already has the merged view).
    "user_overrides": {},
}


# ---------------------------------------------------------------------------
# Per-entry schema validation (L1-L9)
# ---------------------------------------------------------------------------


def _validate_entry(fid: str, spec: Dict[str, Any], *, source: str) -> None:
    """Per-family schema validation. Raises RuntimeError on malformed entry.

    Covers:
      * L1: lowercase ASCII fid.
      * L2: spec is a dict.
      * L3: exactly one of ``presets`` / ``alias_of`` is present
            (mutually exclusive -- having both is an authoring bug).
      * L4: ``alias_of`` if present is a non-empty string.
      * L5: ``presets`` is a list of dicts.
      * L6: each preset dict has ``name`` (non-empty str), ``description``
            (str), ``base_pos`` (3-element list / tuple of numbers).
      * L7: preset names unique within the family.
      * L8: ``base_pos`` numbers coerce to float (no NaN / non-numeric).
      * L9: no ``joint_pos_by_ir`` smuggled into the family layer
            (per-SKU angles belong to
            ``robots_canonical.json[sku].init_pose_presets`` because
            init pose joint angles are per-robot physical quantities).
    """
    # L1: lowercase ASCII fid.
    if fid != fid.lower() or not fid.isascii():
        raise RuntimeError(
            f"init_pose_presets: family id {fid!r} must be lowercase "
            f"ASCII; rename in {source}"
        )
    # L2: spec is a dict.
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"init_pose_presets: family={fid!r} in {source} is not a "
            f"dict (got {type(spec).__name__})"
        )
    has_alias = "alias_of" in spec and spec["alias_of"] is not None
    has_data = "presets" in spec
    # L3: exactly one of presets / alias_of (mutually exclusive).
    if has_alias and has_data:
        raise RuntimeError(
            f"init_pose_presets: family={fid!r} declares both alias_of "
            f"and presets -- choose one (data identity alias means the "
            f"target's presets are reused verbatim); in {source}"
        )
    if not has_alias and not has_data:
        raise RuntimeError(
            f"init_pose_presets: family={fid!r} must declare either "
            f"'presets' (a list of preset dicts) or alias_of (a family "
            f"id string); in {source}"
        )
    # L4: alias_of is a non-empty string.
    if has_alias:
        alias = spec["alias_of"]
        if not isinstance(alias, str) or not alias.strip():
            raise RuntimeError(
                f"init_pose_presets: family={fid!r}.alias_of must be a "
                f"non-empty string (got {alias!r}) in {source}"
            )
        return
    # L5: presets is a list of dicts.
    presets = spec["presets"]
    if not isinstance(presets, list):
        raise RuntimeError(
            f"init_pose_presets: family={fid!r}.presets must be a list "
            f"(got {type(presets).__name__}) in {source}"
        )
    seen_names: set = set()
    for i, p in enumerate(presets):
        if not isinstance(p, dict):
            raise RuntimeError(
                f"init_pose_presets: family={fid!r}.presets[{i}] must be "
                f"a dict (got {type(p).__name__}) in {source}"
            )
        # L6: name + description + base_pos shape.
        name = p.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(
                f"init_pose_presets: family={fid!r}.presets[{i}].name "
                f"must be a non-empty string (got {name!r}) in {source}"
            )
        description = p.get("description", "")
        if not isinstance(description, str):
            raise RuntimeError(
                f"init_pose_presets: family={fid!r}.presets[{i}]"
                f"({name!r}).description must be a string (got "
                f"{type(description).__name__}) in {source}"
            )
        bp = p.get("base_pos")
        if not isinstance(bp, (list, tuple)) or len(bp) != 3:
            raise RuntimeError(
                f"init_pose_presets: family={fid!r}.presets[{i}]"
                f"({name!r}).base_pos must be a 3-element list / tuple "
                f"(got {bp!r}) in {source}"
            )
        # L8: base_pos numbers coerce to float.
        for j, v in enumerate(bp):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise RuntimeError(
                    f"init_pose_presets: family={fid!r}.presets[{i}]"
                    f"({name!r}).base_pos[{j}]={v!r} is not a number "
                    f"in {source}"
                )
        # L7: name unique within the family.
        if name in seen_names:
            raise RuntimeError(
                f"init_pose_presets: family={fid!r} declares preset "
                f"name={name!r} twice; preset names must be unique "
                f"within a family in {source}"
            )
        seen_names.add(name)
        # L9: the family layer carries name + base_pos + description
        # by design. A user overlay that smuggles ``joint_pos_by_ir``
        # into a family-level preset is a contract violation --
        # per-SKU joint angles belong to
        # ``robots_canonical.json[sku].init_pose_presets``, NOT here
        # (init pose joint angles are per-robot physical quantities,
        # not family canonicals; §8 forbids the silent fallback that
        # would let a quadruped 'stand' pose's Go2-tuned thigh angle
        # surface on Spot or ANYmal). The merge-time normalisation
        # would silently drop the field; surface it as a fail-loud
        # raise at load instead.
        if "joint_pos_by_ir" in p:
            raise RuntimeError(
                f"init_pose_presets: family={fid!r}.presets[{i}]"
                f"({name!r}) carries 'joint_pos_by_ir' but the family "
                f"layer is name + base_pos only -- move per-SKU joint "
                f"angles to robots_canonical.json[sku].init_pose_presets "
                f"(init pose joint angles are per-robot physical "
                f"quantities) in {source}"
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
            f"init_pose_presets: 'families' must be a mapping in "
            f"{source} (got {type(families_block).__name__})"
        )
    added = 0
    for fid_raw, spec in families_block.items():
        fid = str(fid_raw)
        _validate_entry(fid, spec, source=source)
        if "alias_of" in spec and spec["alias_of"] is not None:
            _state["families"][fid] = {"alias_of": str(spec["alias_of"])}
        else:
            normalised_presets: List[Dict[str, Any]] = []
            for p in spec["presets"]:
                normalised_presets.append({
                    "name": str(p["name"]),
                    "description": str(p.get("description", "") or ""),
                    "base_pos": [float(v) for v in p["base_pos"]],
                })
            _state["families"][fid] = {
                "alias_of": None,
                "presets": normalised_presets,
            }
        if track_user_overrides:
            _state["user_overrides"][fid] = dict(spec)
        added += 1
    return added


# ---------------------------------------------------------------------------
# Public API: load / overlay
# ---------------------------------------------------------------------------


def load() -> int:
    """Load ``init_pose_presets_catalog.json`` into :data:`_state`.

    Returns the family entry count (alias + data combined). Idempotent --
    repeat calls short-circuit on the cached state. Pattern mirrors
    :func:`registers.gait_commands.load` /
    :func:`registers.amp_obs_templates.load`.
    """
    if _state["loaded"]:
        return len(_state["families"])

    payload = read_data(_CATALOG_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"init_pose_presets_catalog.json failed to parse: "
            f"{_CATALOG_PATH}"
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

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/init_pose_presets_custom.json``)::

        {
          "version": "1.0.0",
          "families": {
            "<fid>": {
              "presets": [
                {"name": "...", "description": "...",
                 "base_pos": [..., ..., ...]},
                ...
              ]
            }
          }
        }

    Returns the number of families added / overridden.

    Per-entry validation (L1-L9) raises on malformed overlay so a typo
    cannot silently destabilise the catalog. Cross-registry rules I1-I5
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


def list_presets(family: str) -> List[InitPosePreset]:
    """Return the resolved family-default presets for ``family``.

    Resolves the alias chain (humanoid -> biped) before returning so
    consumers can treat aliased families identically to root families.
    Raises :class:`InitPosePresetValidationError` when ``family`` is
    not declared.
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        raise InitPosePresetValidationError(
            f"list_presets: {family!r} not in "
            f"init_pose_presets_catalog.json; declared families: "
            f"{sorted(_state['families'].keys())}"
        )
    chain = _resolve_alias_chain(fid)
    root = chain[-1]
    root_spec = _state["families"][root]
    if root_spec.get("alias_of"):
        raise InitPosePresetValidationError(
            f"list_presets: chain for {family!r} resolves to alias-only "
            f"entry {root!r}; RegistryHub.validate should have flagged "
            f"this -- please re-run."
        )
    declared_alias = _state["families"][fid].get("alias_of")
    out: List[InitPosePreset] = []
    for p in root_spec["presets"]:
        out.append(InitPosePreset(
            family=fid,
            alias_of=declared_alias,
            resolved_from=root,
            name=str(p["name"]),
            description=str(p.get("description", "") or ""),
            base_pos=(
                float(p["base_pos"][0]),
                float(p["base_pos"][1]),
                float(p["base_pos"][2]),
            ),
        ))
    return out


def has_preset(family: str, name: str) -> bool:
    """True iff ``family`` declares a preset with ``name``.

    Returns False ONLY when the family is absent or the preset is not
    in the resolved family's preset list. Catalog corruption (broken
    alias chain) RAISES :class:`InitPosePresetValidationError` rather
    than silently returning False; that case must be surfaced by
    :meth:`RegistryHub.validate`, not masked here.
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        return False
    # If chain is broken, raise instead of swallowing.
    _resolve_alias_chain(fid)
    for p in list_presets(fid):
        if p.name == str(name):
            return True
    return False


def get_preset(family: str, name: str) -> InitPosePreset:
    """Return the resolved :class:`InitPosePreset` for ``(family, name)``.

    Raises :class:`InitPosePresetValidationError` when ``family`` is
    not declared or when ``name`` is not a preset in the resolved
    family (NOT a soft Optional -- callers with unknown families /
    preset names are a bug we want to fail loud on, matching the
    consistency-sensitive nature of init pose application).
    """
    if not _state["loaded"]:
        load()
    presets = list_presets(family)
    for p in presets:
        if p.name == str(name):
            return p
    raise InitPosePresetValidationError(
        f"get_preset: family={family!r} declares no preset named "
        f"{name!r}; declared preset names: {[p.name for p in presets]!r}"
    )


def _resolve_alias_chain(family: str) -> List[str]:
    """Walk the ``alias_of`` chain from ``family`` to the data-bearing root.

    Returns ``[family, parent, ..., root]``. Raises
    :class:`InitPosePresetValidationError` on cycle or missing
    intermediate target (defensive belt for catalog corruption).
    """
    chain: List[str] = []
    visited: set = set()
    cur: Optional[str] = family
    while cur is not None:
        if cur in visited:
            raise InitPosePresetValidationError(
                f"_resolve_alias_chain: cycle detected on {family!r} -- "
                f"chain={chain + [cur]!r}; this should have been caught "
                f"by RegistryHub.validate, please re-run it"
            )
        if cur not in _state["families"]:
            raise InitPosePresetValidationError(
                f"_resolve_alias_chain: chain from {family!r} reached "
                f"{cur!r} which is not declared in "
                f"init_pose_presets_catalog.json"
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
      * I1: every non-alias family has unique preset names (L7 catches
        at load time; I1 is the cross-registry-time belt against
        in-memory mutation between load and validate).
      * I2: every non-alias family's preset ``base_pos`` is a 3-tuple
        of finite floats (L6 / L8 catch at load time; I2 is the
        cross-registry-time belt).
      * I3: ``alias_of`` target exists + no cycles (same tri-state DFS
        shape as
        :func:`registers.gait_commands._validate_alias_chains` /
        :func:`registers.amp_obs_templates._validate_alias_chains`).
      * I4: every family declared in the init_pose_presets catalog
        must exist in :func:`registers.families.list_families`. Keeps
        the registry aligned with the central family vocabulary.

    Soft rule (emitted as ``log_warning`` rather than added to problem
    list, so it does not block boot):
      * I6: any family declared in ``families_catalog.json`` but
        absent from the init_pose_presets catalog gets a one-line
        warning. Some families (e.g. manipulator) may legitimately
        have no canonical init pose presets and the warning is the
        loud-and-clear coverage-gap signal (mirrors
        :func:`registers.amp_obs_templates.validate` A4 /
        :func:`registers.gait_commands.validate` G6 shape).

    Note on I5 (joint_pos_by_ir IR slice check):
      The family-default layer carries NAMES + base_pos + description
      only -- ``joint_pos_by_ir`` is intentionally absent here because
      init pose joint angles are per-robot physical quantities. The
      SKU-override layer (``robots_canonical.json[sku].init_pose_presets``)
      carries the joint angles per SKU. IR slice validation for
      SKU-override data is performed inside
      :meth:`RegistryHub.validate` Rule 3 (every joints[*].ir_role
      must be a member of the IR canonical role set for that robot's
      family) for the existing per-format joints table; the same
      ir_canonical pattern catches malformed
      ``init_pose_presets[*].joint_pos_by_ir`` keys before they reach
      a runtime apply. The init_pose_presets registry itself has
      nothing to cross-check on this axis -- I5 is therefore a
      defensive belt for the user-overlay path: if a user overlay
      ever ships a ``presets[*].joint_pos_by_ir`` field on a
      family entry, surface it as a contract violation (the family
      layer is name + base_pos by design; per-SKU angles go to
      robots_canonical).

    L1-L9 (per-entry schema) were enforced at :func:`load` /
    :func:`merge_user_extensions` so they are not re-checked
    exhaustively here -- I1-I2 are the cross-registry-time defensive
    belts only.
    """
    from . import families as _families
    if not _state["loaded"]:
        load()
    problems: List[str] = []

    # I3-first: alias_of target exists + no cycles. Same shape as
    # gait_commands._validate_alias_chains; run first because cycle
    # detection prevents downstream rules from walking corrupted chains.
    problems.extend(_validate_alias_chains())

    # I4: every catalog family must be in families_catalog.json.
    catalog_families = set(_families.list_families())
    for fid in _state["families"]:
        if fid not in catalog_families:
            problems.append(
                f"  init_pose_presets: family={fid!r} is declared in "
                f"init_pose_presets_catalog.json but not in "
                f"families_catalog.json; add the family to "
                f"families_catalog.json or remove the init pose entry"
            )

    # I1-I2 + I5: per-entry defensive belts on the data families.
    for fid, spec in _state["families"].items():
        if spec.get("alias_of"):
            continue
        presets = spec.get("presets", []) or []
        names_seen: set = set()
        for i, p in enumerate(presets):
            name = str(p.get("name", "") or "")
            # I1: name unique.
            if name in names_seen:
                problems.append(
                    f"  init_pose_presets: family={fid!r} declares "
                    f"preset name={name!r} twice"
                )
            names_seen.add(name)
            # I2: base_pos shape + finite floats.
            bp = p.get("base_pos", [])
            if not isinstance(bp, list) or len(bp) != 3:
                problems.append(
                    f"  init_pose_presets: family={fid!r}.presets[{i}]"
                    f"({name!r}).base_pos must be a 3-element list "
                    f"(got {bp!r})"
                )
            else:
                for j, v in enumerate(bp):
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        problems.append(
                            f"  init_pose_presets: family={fid!r}."
                            f"presets[{i}]({name!r}).base_pos[{j}]="
                            f"{v!r} is not a number"
                        )
            # I5 (defensive belt for user overlay): the family layer
            # carries no joint angles. A user overlay that smuggles
            # ``joint_pos_by_ir`` into a family-level preset is a
            # contract violation (per-SKU angles go to
            # robots_canonical, not here).
            if "joint_pos_by_ir" in p:
                problems.append(
                    f"  init_pose_presets: family={fid!r}.presets[{i}]"
                    f"({name!r}) carries 'joint_pos_by_ir' but the "
                    f"family layer is name + base_pos only -- move "
                    f"per-SKU joint angles to "
                    f"robots_canonical.json[sku].init_pose_presets"
                )

    # I6 (soft warn) -- families_catalog declares a family but
    # init_pose_presets catalog has no entry. Emitted via log_warning
    # (not added to problems) so boot is not blocked; the omission is
    # a coverage gap (or a legitimate family without canonical
    # presets), not a corruption.
    pose_fams = set(_state["families"].keys())
    for cf in catalog_families:
        if cf not in pose_fams:
            log_warning(
                f"[init_pose_presets] family={cf!r} is declared in "
                f"families_catalog.json but has no init pose preset "
                f"entry -- add a presets list if the family has canonical "
                f"init poses, otherwise canvas users will see an empty "
                f"preset dropdown for SKUs of this family"
            )

    return problems


def _validate_alias_chains() -> List[str]:
    """``alias_of`` target exists + no cycles. Returns problem strings.

    Same tri-state DFS shape as
    :func:`registers.gait_commands._validate_alias_chains` /
    :func:`registers.amp_obs_templates._validate_alias_chains` /
    :func:`registers.pd_calibration_canaries._validate_alias_chains`.
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
                f"  init_pose_presets: family={fid!r} alias_of="
                f"{target!r} but target is not declared in "
                f"init_pose_presets_catalog.json; fix typo or add the "
                f"missing family"
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
                            f"  init_pose_presets: alias cycle detected: "
                            f"{' -> '.join(rotated + [rotated[0]])}; "
                            f"break the cycle in "
                            f"init_pose_presets_catalog.json"
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
    "InitPosePreset",
    "InitPosePresetValidationError",
    "load",
    "merge_user_extensions",
    "list_families",
    "list_presets",
    "get_preset",
    "has_preset",
    "validate",
    "get_version",
]
