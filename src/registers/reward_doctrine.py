# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.reward_doctrine — Reward-design DOCTRINE (pure teaching data).

Backed by ``data/reward_doctrine.json`` (factory defaults) plus an optional
user overlay at ``<USER_CONFIG_DIR>/registers/reward_doctrine_custom.json``.

Purpose
-------
The AI Build agent, and the reward Coverage/critic surfaces, only see each
reward term's ``name / min / max / default / polarity`` — the *design reasoning*
that lives in each reward script's docstring is stripped. This module restores
that knowledge as compact, queryable data so the agent can reason:

* **辩证面 (dialectical)** — pair a task reward with its penalty counterpart,
* **姿态 (posture)** — learn posture alongside command tracking,
* **局部解审视 (local-optima review)** — self-audit for degenerate optima.

This is **teaching only**: it never gates, blocks, or auto-injects. A missing
category/hint is a missing *teaching hint*, not a required input — the queries
degrade gracefully (return ``""`` / ``None``) rather than raise (§8 applies to
required inputs, not advisory hints). Exhaustive term coverage is guarded by a
unit test, not a boot-time gate.

Calling code
------------
* ``ai_orchestration.registry_projection`` surfaces ``category_of`` / ``hint_of``
  per term and the ``principles`` / ``degenerate_optima`` blocks in the agent
  vocabulary.
* the rewards-block self-review turn reads ``degenerate_optima``.
* the critic prompt restates the doctrine pathologies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_DEFAULTS_PATH = _DATA_DIR / "reward_doctrine.json"


def _user_overlay_path():
    """Lazy resolver for the user-overlay path (USER_CONFIG_DIR is not set until
    the install wizard runs — resolve at call time, never at import)."""
    return Paths.USER_CONFIG_DIR / "registers" / "reward_doctrine_custom.json"


# =============================================================================
# Module cache
# =============================================================================

_state: Dict[str, Any] = {
    "loaded": False,
    "doc": {},  # merged doctrine dict
}


def load() -> int:
    """Read factory + user doctrine into the module cache.

    Returns the number of categorized ``(term, backend)`` entries (a count for
    the RegistryHub summary; never raises on a missing/garbled overlay)."""
    if _state["loaded"]:
        return _categorized_count()
    doc = _read_doc(_DEFAULTS_PATH)
    overlay = _read_doc(_user_overlay_path())
    if overlay:
        doc = _merge(doc, overlay)
    _state["doc"] = doc
    _state["loaded"] = True
    return _categorized_count()


def reload() -> int:
    _state["loaded"] = False
    _state["doc"] = {}
    return load()


def _read_doc(path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_data(path)
    except Exception as exc:  # noqa: BLE001 — teaching data, degrade not crash
        log_warning(f"[registers.reward_doctrine] failed to read {path}: {exc}")
        return {}
    if isinstance(data, dict):
        return data
    log_warning(
        f"[registers.reward_doctrine] unexpected payload shape in {path}: "
        f"{type(data).__name__}"
    )
    return {}


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay wins. Dicts merge per-key (recursively); lists of ``{id|name:..}``
    merge by that identity key (user overrides matching, appends new); any other
    value is replaced wholesale."""
    out = dict(base)
    for key, oval in overlay.items():
        bval = out.get(key)
        if isinstance(bval, dict) and isinstance(oval, dict):
            out[key] = _merge(bval, oval)
        elif isinstance(bval, list) and isinstance(oval, list):
            out[key] = _merge_list(bval, oval)
        else:
            out[key] = oval
    return out


def _merge_list(base: List[Any], overlay: List[Any]) -> List[Any]:
    id_key = _list_id_key(base) or _list_id_key(overlay)
    if id_key is None:
        return list(overlay)  # opaque list → replace
    merged = {str(e[id_key]): dict(e) for e in base if isinstance(e, dict) and id_key in e}
    order = [str(e[id_key]) for e in base if isinstance(e, dict) and id_key in e]
    for e in overlay:
        if not (isinstance(e, dict) and id_key in e):
            continue
        k = str(e[id_key])
        if k not in merged:
            order.append(k)
        merged[k] = dict(e)
    return [merged[k] for k in order]


def _list_id_key(items: List[Any]) -> Optional[str]:
    for e in items:
        if isinstance(e, dict):
            if "id" in e:
                return "id"
            if "name" in e:
                return "name"
        return None
    return None


# =============================================================================
# Queries (graceful on miss)
# =============================================================================

def _doc() -> Dict[str, Any]:
    if not _state["loaded"]:
        load()
    return _state["doc"]


def _term_categories() -> Dict[str, Dict[str, str]]:
    tc = _doc().get("term_categories") or {}
    return tc if isinstance(tc, dict) else {}


def category_of(term: str, backend: str) -> str:
    """The semantic category of ``term`` for ``backend`` (``""`` on miss)."""
    per_backend = _term_categories().get(str(backend)) or {}
    if isinstance(per_backend, dict):
        return str(per_backend.get(str(term), "") or "")
    return ""


def hint_of(term: str, backend: Optional[str] = None) -> Optional[str]:
    """The one-line design hint for ``term`` (``None`` on miss).

    Hints are keyed by term name only (a term key means the same thing across
    backends; where names differ they are separate keys). ``backend`` is
    accepted for API symmetry with :func:`category_of` and ignored."""
    hints = _doc().get("term_hints") or {}
    if isinstance(hints, dict):
        h = hints.get(str(term))
        return str(h) if h else None
    return None


def categories() -> List[Dict[str, Any]]:
    cats = _doc().get("categories") or []
    return list(cats) if isinstance(cats, list) else []


def term_categories(backend: str) -> Dict[str, str]:
    """The full ``{term: category}`` map for ``backend`` (empty on miss)."""
    per_backend = _term_categories().get(str(backend)) or {}
    return dict(per_backend) if isinstance(per_backend, dict) else {}


def categorized_terms(backend: str) -> frozenset:
    """The set of term keys that carry a category for ``backend`` (test aid)."""
    return frozenset(term_categories(backend).keys())


def principles() -> List[Dict[str, Any]]:
    p = _doc().get("principles") or []
    return list(p) if isinstance(p, list) else []


def caveat() -> str:
    return str(_doc().get("caveat", "") or "")


def degenerate_optima() -> List[Dict[str, Any]]:
    d = _doc().get("degenerate_optima") or []
    return list(d) if isinstance(d, list) else []


def _categorized_count() -> int:
    return sum(len(v) for v in _term_categories().values() if isinstance(v, dict))


__all__ = [
    "load",
    "reload",
    "category_of",
    "hint_of",
    "categories",
    "term_categories",
    "categorized_terms",
    "principles",
    "caveat",
    "degenerate_optima",
]
