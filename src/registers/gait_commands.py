# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.gait_commands -- Per-family parameterised gait command spec registry.

Purpose
-------
- Single source of truth for the family-keyed parameterised gait command
  layout. Each entry's ``class_name`` identifies the env-cfg compiler
  CommandTerm class to emit (``UniformGaitCommand`` for quadruped,
  ``BipedGaitCommand`` for biped/humanoid); ``command_dim`` /
  ``phase_count`` / ``phase_obs_dim`` describe the per-env command tensor
  layout and the sin/cos-encoded obs term width so consumers can dispatch
  without any ``if family == "..."`` compare.
- Replaces the historical hardcoded quadruped-only UniformGaitCommand
  emission path with a family-keyed dispatch table. Callers select the
  spec via :func:`get_gait_command` (or the families-side
  :func:`registers.families.resolve_family_dispatch`) keyed on
  ``robot.families[0]``, never on a ``family == "..."`` string compare.

Design boundaries
-----------------
- ``class_name`` names a Python class in the env-cfg compiler emit space;
  this registry has no opinion on the class' implementation. Mismatch
  between catalog ``class_name`` and the actually-emitted class is a
  consumer bug, not a registry bug.
- ``obs_layout`` is a documentation pin and byte-identity assertion lever
  (each entry is ``"<term>:<width>"``); the consumer that emits obs
  terms is the source of truth for actual obs tensor shape. The two are
  cross-checked in tests at the consumer site, not here.
- Missing families are legal: gait command is legged-only by construction;
  ``manipulator`` / ``wheeled`` / ``generic`` deliberately have no entry,
  and the validate G6 soft warning surfaces the omission rather than
  fabricating an empty entry that would imply the family is gait-trainable.

alias_of vs inherits_from
-------------------------
- ``alias_of`` here = **data identity** at load time (hard expansion, no
  override). Mirrors :func:`registers.ir._resolve_alias` /
  :func:`registers.pd_groups._resolve_alias` /
  :class:`registers.symmetry.SymmetryData` /
  :class:`registers.pd_calibration_canaries.CanaryData` /
  :class:`registers.amp_obs_templates.ObsTemplate` pattern.
- ``inherits_from`` (from ``families_catalog.json``) = **dispatch fallback
  chain** opt-in by the caller via
  ``resolve_family_dispatch(allow_fallback=True)``. A different concept;
  this module does NOT use ``inherits_from``.
- ``humanoid`` declares ``alias_of: "biped"`` here, mirroring
  ``ir_canonical``, ``pd_groups_definitions``, ``symmetry_catalog``,
  ``pd_calibration_canaries``, and ``amp_obs_templates`` (six-layer
  alignment is by-design observation, NOT a cross-registry hard
  constraint -- each layer validates independently).

Data layout
-----------
- Factory catalog: ``registers/data/gait_commands_catalog.json``
  (read-only at runtime).
- User overlay: ``<USER_CONFIG_DIR>/registers/gait_commands_custom.json``
  merged at load by ``RegistryHub._merge_user_overlays`` ->
  :func:`merge_user_extensions`.
- The overlay can add new families OR override existing ones; the same
  load-time L1-L7 validation applies.

Fail-loud -- no silent fallbacks
--------------------------------
- Load-time L1-L7 raise on malformed catalog.
- Cross-registry :func:`validate` returns problem list for G1-G5; the
  hub wraps non-empty problem list into
  :class:`RegistryValidationError`.
- Runtime API errors raise :class:`GaitCommandValidationError` with
  actionable messages.
- :func:`get_gait_command` raises on absent family (no soft Optional).
- :func:`has_gait_command` returns False ONLY when the family is absent;
  catalog corruption (broken alias chain) raises instead of silently
  returning False -- that case must be surfaced by
  :meth:`RegistryHub.validate`, not get masked.

API
---
::

    load() -> int
    merge_user_extensions(payload) -> int
    list_families() -> list[str]
    get_gait_command(family) -> GaitCommandSpec
    has_gait_command(family) -> bool
    validate() -> list[str]
    get_version() -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_debug, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CATALOG_PATH = _DATA_DIR / "gait_commands_catalog.json"
_USER_CUSTOM_REL = "registers/gait_commands_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user overlay path inside the LIVE USER_CONFIG_DIR.

    Resolved at call time so workspace hot-switches are picked up without
    restart (mirrors :func:`registers.amp_obs_templates._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "gait_commands_custom.json"


# ---------------------------------------------------------------------------
# Public dataclasses / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaitCommandSpec:
    """Resolved parameterised gait command spec for one family.

    ``@dataclass(frozen=True)`` so downstream consumers (env-cfg compiler
    dispatch path, deploy-side command schema builder) can cache safely.
    ``Tuple`` shape (instead of list) reflects the immutability contract --
    obs_layout / phase_names order is load-bearing for the policy obs
    sub-vector layout, so runtime mutation would be a bug.

    Fields:
        family: family id passed to :func:`get_gait_command`. When the
            query target is an alias, this is still the queried id.
        alias_of: ``alias_of`` value declared on the QUERIED family entry
            (``None`` if the queried family had its own data).
        resolved_from: family id whose data this struct contains. Equal
            to ``family`` when not aliased; otherwise the chain root.
        class_name: Python class name the env-cfg compiler emits as the
            gait CommandTerm for this family.
        command_dim: per-env command tensor width (the
            ``CommandTerm.command`` tensor has shape ``(num_envs,
            command_dim)``).
        phase_count: number of per-foot phase channels embedded in the
            command tensor (occupy ``command[:, 1:1 + phase_count]``).
        phase_obs_dim: width of the sin/cos-encoded phase obs term
            (= ``2 * phase_count``).
        obs_layout: ordered tuple of obs sub-terms emitted to the policy
            group as ``"<name>:<width>"`` strings (e.g.
            ``"frequency:1"``, ``"phase_sin_cos:8"``).
        phase_names: ordered tuple of per-leg labels keyed by phase
            index (length must equal ``phase_count``).
        default_ranges: per-family default sampling ranges
            ``{"frequency" | "body_height" | "step_height": (lo, hi)}``
            — the SINGLE source of the bundled gait range numbers
            (catalog 1.1.0). ``None`` when the entry declares none;
            consumers (``gait_presets.default_ranges_for_family``)
            fail loud on ``None`` instead of substituting a
            wrong-geometry family's values.
    """

    family: str
    alias_of: Optional[str]
    resolved_from: str
    class_name: str
    command_dim: int
    phase_count: int
    phase_obs_dim: int
    obs_layout: Tuple[str, ...]
    phase_names: Tuple[str, ...]
    default_ranges: Optional[Dict[str, Tuple[float, float]]] = None


class GaitCommandValidationError(RuntimeError):
    """Raised by the runtime API on:

    * unknown family in :func:`get_gait_command`;
    * defensive belt for broken alias chain (cycle / missing target)
      encountered at runtime -- :class:`RegistryHub.validate` should
      have caught it; this raise surfaces it if validate has not run.

    Cross-registry validate() problems surface via
    :class:`RegistryValidationError` from the hub instead -- this
    exception is for runtime API contract violations only (mirrors
    :class:`registers.amp_obs_templates.AmpObsTemplatesValidationError` /
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
    #   data entries:  {
    #       "alias_of": None,
    #       "class_name": <str>,
    #       "command_dim": <int>,
    #       "phase_count": <int>,
    #       "phase_obs_dim": <int>,
    #       "obs_layout": [<str>, ...],
    #       "phase_names": [<str>, ...],
    #   }
    "families": {},
    # fid -> original on-disk spec from user overlay (round-trip preservation;
    # ``families`` already has the merged view).
    "user_overrides": {},
}


# ---------------------------------------------------------------------------
# Per-entry schema validation (L1-L7)
# ---------------------------------------------------------------------------


_DATA_FIELDS = (
    "class_name",
    "command_dim",
    "phase_count",
    "phase_obs_dim",
    "obs_layout",
    "phase_names",
)


def _validate_entry(fid: str, spec: Dict[str, Any], *, source: str) -> None:
    """Per-family schema validation. Raises RuntimeError on malformed entry.

    Covers:
      * L1: lowercase ASCII fid.
      * L2: spec is a dict.
      * L3: exactly one of data fields / ``alias_of`` is present
            (mutually exclusive -- having both is an authoring bug).
      * L4: ``alias_of`` if present is a non-empty string.
      * L5: every required data field is present and of the right type
            (``class_name`` non-empty str; the three int fields are int
            and >= 0; the two list fields are list[str]).
      * L6: ``phase_count == len(phase_names)`` (catalog-internal
            consistency; G2 re-checks at cross-registry time but the
            tighter per-entry check at load() catches typos before they
            corrupt cross-registry inputs).
      * L7: ``phase_obs_dim == 2 * phase_count`` (sin + cos encoding;
            G3 re-checks at cross-registry time).
    """
    # L1: lowercase ASCII fid.
    if fid != fid.lower() or not fid.isascii():
        raise RuntimeError(
            f"gait_commands: family id {fid!r} must be lowercase "
            f"ASCII; rename in {source}"
        )
    # L2: spec is a dict.
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"gait_commands: family={fid!r} in {source} is not a "
            f"dict (got {type(spec).__name__})"
        )
    has_alias = "alias_of" in spec and spec["alias_of"] is not None
    has_data = any(k in spec for k in _DATA_FIELDS)
    # L3: exactly one of data fields / alias_of (mutually exclusive).
    if has_alias and has_data:
        raise RuntimeError(
            f"gait_commands: family={fid!r} declares both alias_of "
            f"and data fields ({', '.join(_DATA_FIELDS)}) -- choose one "
            f"(data identity alias means the target's data is reused "
            f"verbatim); in {source}"
        )
    if not has_alias and not has_data:
        raise RuntimeError(
            f"gait_commands: family={fid!r} must declare either the data "
            f"field set ({', '.join(_DATA_FIELDS)}) or alias_of (a family "
            f"id string); in {source}"
        )
    # L4: alias_of is a non-empty string.
    if has_alias:
        alias = spec["alias_of"]
        if not isinstance(alias, str) or not alias.strip():
            raise RuntimeError(
                f"gait_commands: family={fid!r}.alias_of must be a "
                f"non-empty string (got {alias!r}) in {source}"
            )
        # L8 (alias half): default_ranges on an alias entry would be
        # silently ignored (aliases reuse the root's data verbatim) —
        # refuse instead of letting the author believe it took effect.
        if "default_ranges" in spec:
            raise RuntimeError(
                f"gait_commands: family={fid!r} declares alias_of and "
                f"default_ranges; aliases reuse the target's data "
                f"verbatim, so the ranges would be silently ignored — "
                f"move them to the target family or drop alias_of; "
                f"in {source}"
            )
        return
    # L5: every data field is present + typed.
    missing = [k for k in _DATA_FIELDS if k not in spec]
    if missing:
        raise RuntimeError(
            f"gait_commands: family={fid!r} missing data fields "
            f"{missing!r} in {source}; declare every field of "
            f"{list(_DATA_FIELDS)!r} or use alias_of instead"
        )
    class_name = spec["class_name"]
    if not isinstance(class_name, str) or not class_name.strip():
        raise RuntimeError(
            f"gait_commands: family={fid!r}.class_name must be a "
            f"non-empty string (got {class_name!r}) in {source}"
        )
    for k in ("command_dim", "phase_count", "phase_obs_dim"):
        v = spec[k]
        if not isinstance(v, int) or isinstance(v, bool):
            raise RuntimeError(
                f"gait_commands: family={fid!r}.{k} must be an int "
                f"(got {type(v).__name__}={v!r}) in {source}"
            )
        if v < 0:
            raise RuntimeError(
                f"gait_commands: family={fid!r}.{k}={v} is negative; "
                f"must be >= 0 in {source}"
            )
    for k in ("obs_layout", "phase_names"):
        seq = spec[k]
        if not isinstance(seq, list):
            raise RuntimeError(
                f"gait_commands: family={fid!r}.{k} must be a list "
                f"(got {type(seq).__name__}) in {source}"
            )
        for i, item in enumerate(seq):
            if not isinstance(item, str):
                raise RuntimeError(
                    f"gait_commands: family={fid!r}.{k}[{i}]={item!r} "
                    f"is not a string in {source}"
                )
            if not item.strip():
                raise RuntimeError(
                    f"gait_commands: family={fid!r}.{k}[{i}] is empty "
                    f"in {source}; drop the entry or use a valid label"
                )
    # L6: phase_count == len(phase_names) (catalog-internal consistency).
    if int(spec["phase_count"]) != len(spec["phase_names"]):
        raise RuntimeError(
            f"gait_commands: family={fid!r} phase_count="
            f"{spec['phase_count']} != len(phase_names)="
            f"{len(spec['phase_names'])} in {source}; the two must agree "
            f"or downstream dispatch will key off mismatched widths"
        )
    # L7: phase_obs_dim == 2 * phase_count (sin + cos encoding).
    expected_obs = 2 * int(spec["phase_count"])
    if int(spec["phase_obs_dim"]) != expected_obs:
        raise RuntimeError(
            f"gait_commands: family={fid!r} phase_obs_dim="
            f"{spec['phase_obs_dim']} != 2 * phase_count="
            f"{expected_obs} in {source}; sin/cos encoding doubles the "
            f"channel count -- fix the value to {expected_obs}"
        )
    # L8 (data half): default_ranges, when present, is exactly
    # {frequency, body_height, step_height} -> [lo, hi] numeric with
    # lo <= hi. Optional per entry -- a family without it fail-louds at
    # gait_presets.default_ranges_for_family time instead (no silent
    # substitute of another family's geometry-class values).
    if "default_ranges" in spec:
        _validate_default_ranges(fid, spec["default_ranges"], source=source)


_DEFAULT_RANGE_KEYS = ("frequency", "body_height", "step_height")


def _validate_default_ranges(fid: str, ranges: Any, *, source: str) -> None:
    """L8 shape check for a ``default_ranges`` block. Raises RuntimeError."""
    if not isinstance(ranges, dict):
        raise RuntimeError(
            f"gait_commands: family={fid!r}.default_ranges must be a "
            f"mapping (got {type(ranges).__name__}) in {source}"
        )
    missing = [k for k in _DEFAULT_RANGE_KEYS if k not in ranges]
    extra = [k for k in ranges if k not in _DEFAULT_RANGE_KEYS]
    if missing or extra:
        raise RuntimeError(
            f"gait_commands: family={fid!r}.default_ranges must declare "
            f"exactly {list(_DEFAULT_RANGE_KEYS)!r} (missing={missing!r}, "
            f"unexpected={extra!r}) in {source}"
        )
    for k in _DEFAULT_RANGE_KEYS:
        pair = ranges[k]
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in pair)
        ):
            raise RuntimeError(
                f"gait_commands: family={fid!r}.default_ranges[{k!r}] must "
                f"be a [lo, hi] pair of numbers (got {pair!r}) in {source}"
            )
        if float(pair[0]) > float(pair[1]):
            raise RuntimeError(
                f"gait_commands: family={fid!r}.default_ranges[{k!r}]="
                f"{pair!r} has lo > hi in {source}"
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
            f"gait_commands: 'families' must be a mapping in "
            f"{source} (got {type(families_block).__name__})"
        )
    added = 0
    for fid_raw, spec in families_block.items():
        fid = str(fid_raw)
        _validate_entry(fid, spec, source=source)
        if "alias_of" in spec and spec["alias_of"] is not None:
            _state["families"][fid] = {"alias_of": str(spec["alias_of"])}
        else:
            entry: Dict[str, Any] = {
                "alias_of": None,
                "class_name": str(spec["class_name"]),
                "command_dim": int(spec["command_dim"]),
                "phase_count": int(spec["phase_count"]),
                "phase_obs_dim": int(spec["phase_obs_dim"]),
                "obs_layout": list(spec["obs_layout"]),
                "phase_names": list(spec["phase_names"]),
            }
            if "default_ranges" in spec:
                entry["default_ranges"] = {
                    k: [float(v[0]), float(v[1])]
                    for k, v in spec["default_ranges"].items()
                }
            _state["families"][fid] = entry
        if track_user_overrides:
            _state["user_overrides"][fid] = dict(spec)
        added += 1
    return added


# ---------------------------------------------------------------------------
# Public API: load / overlay
# ---------------------------------------------------------------------------


def load() -> int:
    """Load ``gait_commands_catalog.json`` into :data:`_state`.

    Returns the family entry count (alias + data combined). Idempotent --
    repeat calls short-circuit on the cached state. Pattern mirrors
    :func:`registers.amp_obs_templates.load` /
    :func:`registers.pd_calibration_canaries.load`.
    """
    if _state["loaded"]:
        return len(_state["families"])

    payload = read_data(_CATALOG_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"gait_commands_catalog.json failed to parse: {_CATALOG_PATH}"
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

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/gait_commands_custom.json``)::

        {
          "version": "1.0.0",
          "families": {
            "<fid>": {
              "class_name": "MyGaitCommand",
              "command_dim": 5,
              "phase_count": 2,
              "phase_obs_dim": 4,
              "obs_layout": ["frequency:1", "phase_sin_cos:4", ...],
              "phase_names": ["L", "R"]
            }
          }
        }

    Returns the number of families added / overridden.

    Per-entry validation (L1-L7) raises on malformed overlay so a typo
    cannot silently destabilise the catalog. Cross-registry rules G1-G5
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


def has_gait_command(family: str) -> bool:
    """True iff a gait command spec is declared (and alias chain resolves) for ``family``.

    Returns False ONLY when ``family`` is absent from the catalog -- that
    case is the legitimate "no gait command for this family" signal for
    callers that have a meaningful fallback (e.g. manipulator / wheeled
    deliberately have no entry).

    Catalog corruption (broken alias chain -- cycle or missing target)
    RAISES :class:`GaitCommandValidationError` rather than silently
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


def get_gait_command(family: str) -> GaitCommandSpec:
    """Return resolved :class:`GaitCommandSpec` for ``family``.

    Raises :class:`GaitCommandValidationError` when ``family`` is not
    declared in the catalog (NOT a soft Optional -- callers with unknown
    families are a bug we want to fail loud on, matching the consistency-
    sensitive nature of gait command emit).
    """
    if not _state["loaded"]:
        load()
    fid = str(family or "")
    if fid not in _state["families"]:
        raise GaitCommandValidationError(
            f"get_gait_command: {family!r} not in "
            f"gait_commands_catalog.json; declared families: "
            f"{sorted(_state['families'].keys())}"
        )
    chain = _resolve_alias_chain(fid)
    root = chain[-1]
    root_spec = _state["families"][root]
    if root_spec.get("alias_of"):
        # Defensive -- _resolve_alias_chain walks until alias_of is
        # falsy, so this only fires if the resolved root itself declared
        # alias_of to a missing target after we left the chain. Should
        # be caught by validate(); raise as belt-and-braces.
        raise GaitCommandValidationError(
            f"get_gait_command: chain for {family!r} resolves to "
            f"alias-only entry {root!r}; RegistryHub.validate should "
            f"have flagged this -- please re-run."
        )
    declared_alias = _state["families"][fid].get("alias_of")
    raw_ranges = root_spec.get("default_ranges")
    default_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    if raw_ranges is not None:
        default_ranges = {
            k: (float(v[0]), float(v[1])) for k, v in raw_ranges.items()
        }
    return GaitCommandSpec(
        family=fid,
        alias_of=declared_alias,
        resolved_from=root,
        class_name=str(root_spec["class_name"]),
        command_dim=int(root_spec["command_dim"]),
        phase_count=int(root_spec["phase_count"]),
        phase_obs_dim=int(root_spec["phase_obs_dim"]),
        obs_layout=tuple(root_spec["obs_layout"]),
        phase_names=tuple(root_spec["phase_names"]),
        default_ranges=default_ranges,
    )


def _resolve_alias_chain(family: str) -> List[str]:
    """Walk the ``alias_of`` chain from ``family`` to the data-bearing root.

    Returns ``[family, parent, ..., root]``. Raises
    :class:`GaitCommandValidationError` on cycle or missing intermediate
    target (defensive belt for catalog corruption).
    """
    chain: List[str] = []
    visited: set = set()
    cur: Optional[str] = family
    while cur is not None:
        if cur in visited:
            raise GaitCommandValidationError(
                f"_resolve_alias_chain: cycle detected on {family!r} -- "
                f"chain={chain + [cur]!r}; this should have been caught "
                f"by RegistryHub.validate, please re-run it"
            )
        if cur not in _state["families"]:
            raise GaitCommandValidationError(
                f"_resolve_alias_chain: chain from {family!r} reached "
                f"{cur!r} which is not declared in "
                f"gait_commands_catalog.json"
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
      * G1: every non-alias family declares ``class_name`` non-empty
        (L5 already catches "missing or wrong type"; this is the
        cross-registry-time re-check kept for parity with the documented
        G-rule numbering).
      * G2: ``phase_count == len(phase_names)`` for every non-alias
        family (L6 catches at load time; G2 is the cross-registry-time
        belt against in-memory mutation between load and validate).
      * G3: ``phase_obs_dim == 2 * phase_count`` (L7 catches at load
        time; G3 is the cross-registry-time belt).
      * G4: ``alias_of`` target exists + no cycles (same tri-state DFS
        shape as :func:`registers.amp_obs_templates.
        _validate_alias_chains` /
        :func:`registers.pd_calibration_canaries.
        _validate_alias_chains` /
        :func:`registers.families._detect_cycles`).
      * G5: every family declared in the gait_commands catalog must
        exist in :func:`registers.families.list_families`. Keeps the
        registry aligned with the central family vocabulary.

    Soft rule (emitted as ``log_warning`` rather than added to problem
    list, so it does not block boot):
      * G6: any family declared in ``families_catalog.json`` but absent
        from the gait_commands catalog gets a one-line warning. Gait
        command is legged-only by construction, so ``manipulator`` /
        ``wheeled`` / ``generic`` legitimately have no entry and the
        warning is the loud-and-clear coverage-gap signal (mirrors
        amp_obs_templates A4 / pd_calibration_canaries soft-warn shape).

    L1-L7 (per-entry schema) were enforced at :func:`load` /
    :func:`merge_user_extensions` so they are not re-checked exhaustively
    here -- G1-G3 are the cross-registry-time defensive belts only.
    """
    from . import families as _families
    if not _state["loaded"]:
        load()
    problems: List[str] = []

    # G4-first: alias_of target exists + no cycles. Same shape as
    # amp_obs_templates._validate_alias_chains; run first because cycle
    # detection prevents downstream rules from walking corrupted chains.
    problems.extend(_validate_alias_chains())

    # G5: every catalog family must be in families_catalog.json.
    catalog_families = set(_families.list_families())
    for fid in _state["families"]:
        if fid not in catalog_families:
            problems.append(
                f"  gait_commands: family={fid!r} is declared in "
                f"gait_commands_catalog.json but not in "
                f"families_catalog.json; add the family to "
                f"families_catalog.json or remove the gait command entry"
            )

    # G1-G3: per-entry defensive belts on the data families.
    for fid, spec in _state["families"].items():
        if spec.get("alias_of"):
            continue
        # G1
        if not str(spec.get("class_name", "")).strip():
            problems.append(
                f"  gait_commands: family={fid!r} class_name is empty; "
                f"declare the env-cfg compiler CommandTerm class name"
            )
        # G2
        pcount = int(spec.get("phase_count", -1))
        pnames = spec.get("phase_names", []) or []
        if pcount != len(pnames):
            problems.append(
                f"  gait_commands: family={fid!r} phase_count={pcount} "
                f"!= len(phase_names)={len(pnames)}; the two must agree"
            )
        # G3
        pobs = int(spec.get("phase_obs_dim", -1))
        if pobs != 2 * pcount:
            problems.append(
                f"  gait_commands: family={fid!r} phase_obs_dim={pobs} "
                f"!= 2 * phase_count={2 * pcount}; sin/cos encoding "
                f"doubles the channel count"
            )
        # G7 (belt for L8): default_ranges shape re-check against
        # in-memory mutation between load and validate.
        dr = spec.get("default_ranges")
        if dr is not None:
            try:
                _validate_default_ranges(fid, dr, source="in-memory state")
            except RuntimeError as exc:
                problems.append(f"  {exc}")

    # G6 (soft warn) -- families_catalog declares a family but
    # gait_commands catalog has no entry. Emitted via log_warning (not
    # added to problems) so boot is not blocked; the omission is a
    # coverage gap (or a legitimate non-legged family), not a corruption.
    gait_fams = set(_state["families"].keys())
    for cf in catalog_families:
        if cf not in gait_fams:
            log_warning(
                f"[gait_commands] family={cf!r} is declared in "
                f"families_catalog.json but has no gait command entry -- "
                f"add a data entry if the family is gait-trainable, "
                f"otherwise the family will reject Training Commands "
                f"node gait_enabled=true at compile time"
            )

    return problems


def _validate_alias_chains() -> List[str]:
    """``alias_of`` target exists + no cycles. Returns problem strings.

    Same tri-state DFS shape as
    :func:`registers.amp_obs_templates._validate_alias_chains` /
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
                f"  gait_commands: family={fid!r} alias_of="
                f"{target!r} but target is not declared in "
                f"gait_commands_catalog.json; fix typo or add the "
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
                            f"  gait_commands: alias cycle detected: "
                            f"{' -> '.join(rotated + [rotated[0]])}; "
                            f"break the cycle in "
                            f"gait_commands_catalog.json"
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
    "GaitCommandSpec",
    "GaitCommandValidationError",
    "load",
    "merge_user_extensions",
    "list_families",
    "get_gait_command",
    "has_gait_command",
    "validate",
    "get_version",
]
