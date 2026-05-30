# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.families — Family metadata registry / Family vocabulary + dispatch resolver.

Single source of truth for the enumerable set of robot families
(quadruped, biped, humanoid, manipulator, wheeled, generic) and the
family-keyed dispatch resolver consumers across the codebase call
instead of hand-rolling ``robot.families[0]`` lookups.

Purpose
-------
- All ``RobotSpec.families`` values must be drawn from this catalog,
  enforced by ``RegistryHub.validate()`` Rule C5.
- Family-keyed dispatch resolver (``resolve_family_dispatch``).
- Inheritance chain (``humanoid -> biped``) for opt-in fallback
  (``allow_fallback=True``).

Design boundaries
-----------------
- ``inherits_from`` in this catalog is **distinct** from ``alias_of`` in
  ``ir_canonical.json`` / ``pd_groups_definitions.json``:
    * ``alias_of`` = "my data set is identical to target's" (load-time hard
      expansion, no override possible).
    * ``inherits_from`` = "at dispatch time, fall back along this chain when
      the table lacks my own key" (caller opts in per call).
  The two coexist; ``families.py`` only owns ``inherits_from``.
- SKU multi-family superset (e.g. ``["humanoid", "biped"]``) and catalog
  inheritance are TWO INDEPENDENT mechanisms. The validator does NOT
  cross-check them.
- Dispatch priority within ``RobotSpec.families`` is list order. A SKU
  declaring ``["humanoid", "biped"]`` means "prefer humanoid; fall
  through to biped". Matches existing ``env_cfg_compiler``
  ``primary_family = families[0]`` semantics.

Data layout
-----------
- Factory catalog: ``registers/data/families_catalog.json`` (read-only at
  runtime).
- User overlay: ``<USER_CONFIG_DIR>/registers/families_custom.json`` merged at
  load by ``RegistryHub._merge_user_overlays`` -> ``merge_user_extensions``.
- The overlay can add new families OR override existing ones; the same
  load-time validation rules (L3/L4/L5) apply.

Fail-loud — no silent fallbacks
-------------------------------
- Load-time L1-L5 raise on malformed catalog.
- Cross-registry validate() returns problem list for C2/C3; the hub raises
  ``RegistryValidationError`` when problems is non-empty (Rule C4/C5 are
  in the hub body, not here).
- Runtime API errors raise ``FamiliesValidationError`` with actionable
  messages.

API
---
::

    load() -> int
    merge_user_extensions(payload) -> int
    list_families() -> list[str]
    get_family_meta(name) -> FamilyMeta
    get_inheritance_chain(name) -> list[str]
    resolve_family_dispatch(sku, table_name, table, *, allow_fallback=False) -> T
    validate() -> list[str]
    get_version() -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TypeVar

from unitport_sdk import Paths, log_debug, log_warning, read_data


T = TypeVar("T")


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_CATALOG_PATH = _DATA_DIR / "families_catalog.json"
_USER_CUSTOM_REL = "registers/families_custom.json"


def _user_custom_path() -> Path:
    """Lazy resolve of the user catalog overlay path inside the LIVE
    USER_CONFIG_DIR. Resolved at call time so workspace hot-switches are
    picked up without restart (mirrors :func:`registers.robots._user_custom_path`).
    """
    return Paths.USER_CONFIG_DIR / "registers" / "families_custom.json"


# ---------------------------------------------------------------------------
# Public dataclasses / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyMeta:
    """Catalog metadata for one family. Frozen so consumers can cache safely.

    Fields:
        id: family id (the catalog dict key); always lowercase ASCII.
        label: human-facing display string (UI sidebar labels, badges, ...).
        inherits_from: parent family id, or None for a root family. Used only
            by ``resolve_family_dispatch(..., allow_fallback=True)``.
        doc: informational free-text from the catalog ``_doc`` field. Empty
            string when the catalog entry omitted it.
    """

    id: str
    label: str
    inherits_from: Optional[str]
    doc: str


class FamiliesValidationError(RuntimeError):
    """Raised at runtime by the families.py public API on:

    * unknown family name (``get_family_meta`` / ``get_inheritance_chain``);
    * SKU lookup failures in ``resolve_family_dispatch`` (unregistered SKU,
      empty families, family not in catalog, dispatch miss after fallback).

    Cross-registry validate() problems are surfaced via
    ``RegistryValidationError`` from the hub instead (the hub wraps the
    problems list); this exception is for runtime API contract violations
    only.
    """


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------


_state: Dict[str, Any] = {
    "loaded": False,
    "version": "",
    # fid -> {"label": str, "inherits_from": str|None, "_doc": str}
    "families": {},
    # fid -> original on-disk spec from user overlay (round-trip preservation;
    # ``families`` already has the merged view).
    "user_overrides": {},
}


# ---------------------------------------------------------------------------
# Internal load helpers
# ---------------------------------------------------------------------------


def _validate_entry(fid: str, spec: Dict[str, Any], *, source: str) -> None:
    """Per-family schema validation (L3 / L4 / L5 of the fail-loud list).

    Raises RuntimeError on malformed entries — these are factory or user-overlay
    errors that should never pass silently (fail-loud invariant).
    """
    # L5: lowercase ASCII fid.
    if fid != fid.lower() or not fid.isascii():
        raise RuntimeError(
            f"families: family id {fid!r} must be lowercase ASCII; "
            f"rename in {source}"
        )
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"families: family={fid!r} in {source} is not a dict "
            f"(got {type(spec).__name__})"
        )
    # L3: label non-empty.
    label = spec.get("label")
    if not isinstance(label, str) or not label.strip():
        raise RuntimeError(
            f"families: family={fid!r} has empty or missing 'label' field "
            f"in {source}"
        )
    # L4: inherits_from key present (value may be null).
    if "inherits_from" not in spec:
        raise RuntimeError(
            f"families: family={fid!r} missing 'inherits_from' field in "
            f"{source}; declare it as null if no parent"
        )
    parent = spec["inherits_from"]
    if parent is not None and not isinstance(parent, str):
        raise RuntimeError(
            f"families: family={fid!r} inherits_from must be a string or "
            f"null (got {type(parent).__name__}) in {source}"
        )


def _load_payload_into_state(
    families_block: Dict[str, Any],
    *,
    source: str,
    track_user_overrides: bool = False,
) -> int:
    """Validate + merge a parsed families block into _state['families'].

    Used by both :func:`load` (factory catalog) and
    :func:`merge_user_extensions` (user overlay). Per-entry rules L3/L4/L5
    are enforced here; cross-family rules C2/C3 are :func:`validate`'s job.

    Returns the number of families added or overridden.
    """
    if not isinstance(families_block, dict):
        raise RuntimeError(
            f"families: 'families' must be a mapping in {source} "
            f"(got {type(families_block).__name__})"
        )
    added = 0
    for fid_raw, spec in families_block.items():
        fid = str(fid_raw)
        _validate_entry(fid, spec, source=source)
        _state["families"][fid] = {
            "label": str(spec["label"]),
            "inherits_from": spec.get("inherits_from"),
            "_doc": str(spec.get("_doc", "")),
        }
        if track_user_overrides:
            _state["user_overrides"][fid] = dict(spec)
        added += 1
    return added


# ---------------------------------------------------------------------------
# Public API: load / overlay
# ---------------------------------------------------------------------------


def load() -> int:
    """Load ``families_catalog.json`` into _state; return family count.

    Idempotent — repeated calls short-circuit on the cached state. Pattern
    mirrors ``ir.load`` / ``pd_groups.load``.
    """
    if _state["loaded"]:
        return len(_state["families"])

    payload = read_data(_CATALOG_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"families_catalog.json failed to parse: {_CATALOG_PATH}"
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

    Payload shape (matches ``<USER_CONFIG_DIR>/registers/families_custom.json``)::

        {
          "version": "1.0.0",
          "families": {
            "<fid>": {
              "label":          "<display string>",
              "inherits_from":  "<parent fid> | null",
              "_doc":           "<optional doc>"
            }
          }
        }

    Returns the number of family entries successfully applied (add or override).
    Per-entry validation (L3/L4/L5) raises RuntimeError on malformed overlay
    so a typo cannot silently destabilise the catalog (fail-loud invariant). Cross-
    family rules C2/C3 are re-checked by :func:`RegistryHub.validate`.
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


def get_family_meta(name: str) -> FamilyMeta:
    """Return the :class:`FamilyMeta` for ``name``.

    Raises :class:`FamiliesValidationError` if ``name`` is not in the catalog.
    NOT a soft Optional — callers with an unknown family are a bug we want
    to fail loud on per the no-silent-fallback invariant.
    """
    if not _state["loaded"]:
        load()
    fid = str(name or "")
    spec = _state["families"].get(fid)
    if spec is None:
        raise FamiliesValidationError(
            f"get_family_meta: {name!r} not in families_catalog.json; "
            f"declared families: {sorted(_state['families'].keys())}"
        )
    return FamilyMeta(
        id=fid,
        label=str(spec.get("label", "")),
        inherits_from=spec.get("inherits_from"),
        doc=str(spec.get("_doc", "")),
    )


def get_inheritance_chain(name: str) -> List[str]:
    """Return ``[name, parent, grandparent, ...]`` up to the root.

    Used by :func:`resolve_family_dispatch` when ``allow_fallback=True``.
    Cycles are impossible if :func:`RegistryHub.validate` has run (cycle
    detection there rejects circular catalogs); a defensive raise here
    handles hand-edited overlays / pre-validate usage.

    Raises :class:`FamiliesValidationError` if ``name`` is unknown or a
    cycle is detected mid-walk.
    """
    if not _state["loaded"]:
        load()
    fid = str(name or "")
    if fid not in _state["families"]:
        raise FamiliesValidationError(
            f"get_inheritance_chain: {name!r} not in families_catalog.json; "
            f"declared families: {sorted(_state['families'].keys())}"
        )
    chain: List[str] = []
    visited: set = set()
    cur: Optional[str] = fid
    while cur is not None:
        if cur in visited:
            raise FamiliesValidationError(
                f"get_inheritance_chain: cycle detected on {name!r} — "
                f"chain={chain + [cur]!r}; this should have been caught "
                f"by RegistryHub.validate, please re-run it"
            )
        if cur not in _state["families"]:
            # Off-catalog mid-walk — same root cause as C2, but here we are
            # in runtime API land so we must raise (defensive belt for
            # hand-edited overlays).
            raise FamiliesValidationError(
                f"get_inheritance_chain: chain from {name!r} reached "
                f"{cur!r} which is not in the catalog; "
                f"RegistryHub.validate should catch this — please re-run."
            )
        visited.add(cur)
        chain.append(cur)
        cur = _state["families"][cur].get("inherits_from")
    return chain


def resolve_family_dispatch(
    sku: str,
    table_name: str,
    table: Mapping[str, T],
    *,
    allow_fallback: bool = False,
) -> T:
    """Resolve a family-keyed dispatch lookup for ``sku`` against ``table``.

    Walks ``RobotSpec.families`` in list order; when
    ``allow_fallback=True``, each family additionally walks its
    :func:`get_inheritance_chain` (humanoid -> biped, etc.) before moving to
    the next family.

    Args:
        sku: Robot SKU (canonical key from ``registers.robots``). Must be a
            registered robot whose ``families`` list is non-empty — empty
            families is a Rule §8 violation we do not paper over.
        table_name: Caller-supplied identifier for error messages
            (e.g. ``"amp_obs_templates"``, ``"gait_command_kinds"``). Not used
            for lookup, only for fail-loud messaging.
        table: Caller-owned family-keyed dispatch table. Keys must be family
            ids from the catalog (verified at lookup time).
        allow_fallback: If False (default), only the SKU's declared families
            are searched. If True, after each declared family is missed, the
            inheritance chain of that family is walked. Visited families are
            deduped across the search (the humanoid chain may share biped
            with a later humanoid-derived family).

    Returns:
        The value from ``table`` for the first matching family.

    Raises:
        :class:`FamiliesValidationError` with an actionable message on any of:
          * sku not registered
          * sku has empty families
          * any family in sku.families is not in catalog
          * nothing matches (after fallback if enabled)
    """
    # Lazy import to avoid a load-time cycle (robots imports __init__ which
    # imports families).
    from . import robots as _robots

    if not _state["loaded"]:
        load()

    entry = _robots.get_robot(sku)
    if entry is None:
        raise FamiliesValidationError(
            f"resolve_family_dispatch: sku={sku!r} not in registers.robots; "
            f"check call site or run RegistryHub.reload()"
        )
    sku_families = list(entry.get("families", []) or [])
    if not sku_families:
        raise FamiliesValidationError(
            f"resolve_family_dispatch: sku={sku!r} has empty families; "
            f"populate `families` in robots_canonical.json or user overlay "
            f"(declare at least one family from families_catalog.json)"
        )
    for fam in sku_families:
        if fam not in _state["families"]:
            raise FamiliesValidationError(
                f"resolve_family_dispatch: sku={sku!r} declares family="
                f"{fam!r} which is not in families_catalog.json; add it to "
                f"the catalog or correct the SKU"
            )

    searched: List[str] = []
    for fam in sku_families:
        chain = get_inheritance_chain(fam) if allow_fallback else [fam]
        for node in chain:
            if node in searched:
                continue
            searched.append(node)
            if node in table:
                return table[node]

    raise FamiliesValidationError(
        f"resolve_family_dispatch: sku={sku!r} families={sku_families!r} "
        f"(allow_fallback={allow_fallback}, searched_chain={searched!r}) "
        f"found no key in table_name={table_name!r}; "
        f"available keys: {sorted(table.keys())}"
    )


# ---------------------------------------------------------------------------
# Cross-registry validation (called from RegistryHub.validate)
# ---------------------------------------------------------------------------


def validate() -> List[str]:
    """Cross-family integrity check (C2 + C3 of the PV fail-loud list).

    Returns a list of human-readable problem strings. Empty list = OK.
    The caller (:meth:`RegistryHub.validate`) wraps this list into a
    :class:`RegistryValidationError`.

    Checks:
      * C2: every non-null ``inherits_from`` targets an existing family in
            the catalog.
      * C3: the catalog has no inheritance cycle.

    C1 (per-entry schema) was already enforced at load time by
    :func:`_validate_entry`. C4 (SKU has non-empty families) and C5 (every
    SKU family is in the catalog) live in :meth:`RegistryHub.validate`'s
    main body — that is where SKU-side cross-registry checks already happen.
    """
    if not _state["loaded"]:
        load()

    problems: List[str] = []

    # C2: every non-null inherits_from target must be a declared family.
    for fid, spec in _state["families"].items():
        parent = spec.get("inherits_from")
        if parent is None:
            continue
        if parent not in _state["families"]:
            problems.append(
                f"  families: family={fid!r} inherits_from={parent!r} but "
                f"target is not declared in families_catalog.json; "
                f"fix typo or add the missing family"
            )

    # C3: DFS-based cycle detection with gray/black tri-state. We report the
    # specific cycle nodes (error messages must point at WHERE the
    # cycle is, not just that one exists).
    problems.extend(_detect_cycles())

    return problems


def _detect_cycles() -> List[str]:
    """Tri-state DFS cycle detection. Returns problem strings naming the
    cycle nodes; empty list = no cycles."""
    WHITE, GRAY, BLACK = 0, 1, 2
    families = _state["families"]
    color: Dict[str, int] = {fid: WHITE for fid in families}
    problems: List[str] = []
    found_cycles: set = set()  # dedupe across DFS entry points

    def dfs(node: str, path: List[str]) -> None:
        color[node] = GRAY
        path.append(node)
        parent = families[node].get("inherits_from")
        if parent and parent in families:
            if color[parent] == GRAY:
                cycle_start = path.index(parent)
                cycle = path[cycle_start:] + [parent]
                # Normalise the cycle for dedupe: rotate to start at the
                # lexicographically smallest node so "a->b->c->a" and
                # "b->c->a->b" report once.
                core = cycle[:-1]  # drop trailing repeat
                if core:
                    pivot = core.index(min(core))
                    rotated = core[pivot:] + core[:pivot]
                    key = tuple(rotated)
                    if key not in found_cycles:
                        found_cycles.add(key)
                        problems.append(
                            f"  families: inheritance cycle detected: "
                            f"{' -> '.join(rotated + [rotated[0]])}; "
                            f"break the cycle in families_catalog.json"
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
    "FamilyMeta",
    "FamiliesValidationError",
    "load",
    "merge_user_extensions",
    "list_families",
    "get_family_meta",
    "get_inheritance_chain",
    "resolve_family_dispatch",
    "validate",
    "get_version",
]
