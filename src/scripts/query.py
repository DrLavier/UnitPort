# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting query helpers for preset registries.

Migrated from the tail of ``DEMO/src/system/training/task_module_registry.py``.
Exposes a unified view over four preset kinds:

* ``reward``         — SB3 + Isaac Lab reward presets
* ``termination``    — SB3 + Isaac Lab termination presets
* ``observation``    — Isaac Lab observation presets (no SB3 equivalent)
* ``discriminator``  — AMP discriminator method-override presets
                      (Isaac Lab + AMP algorithm only)

Callers use :func:`lookup` / :func:`query_registry` / :func:`validate_keys`
instead of indexing the per-kind dicts directly so that:

* a reward key like ``"base_height"`` never resolves to the termination
  entry of the same name (kind-namespaced lookup);
* the SB3 vs Isaac-Lab variants of the same key are disambiguated by
  ``backend`` filter.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from scripts.discriminator.registry import IL_DISC_REGISTRY
from scripts.observations.registry import IL_OBS_REGISTRY
from scripts.rewards.registry import IL_REWARD_REGISTRY, REWARD_REGISTRY
from scripts.task_module import ALG_ALL, TaskModuleItem
from scripts.terminations.registry import (
    IL_TERMINATION_REGISTRY,
    TERMINATION_REGISTRY,
)


# Keys are only unique WITHIN a kind. A reward and a termination may
# legitimately share a semantic name (e.g. ``base_height`` = reward
# penalty on height deviation, AND termination when height drops too
# low). All lookups go through ``lookup(key, kind=...)`` or
# ``query_registry(kind=...)``, both of which preserve the (kind, key)
# tuple namespace.
#
# Within a kind, multiple sub-registries may exist (SB3 + IL variants).
# They are iterated in order; ``query_registry`` filters by ``backend``
# so only the implementation matching the engine is returned.
#
# ``writable_registry(kind)`` returns the LAST sub-registry — by
# convention the IL/AMP variant. The discriminator kind has only one
# sub-registry (AMP-only); observation likewise (Isaac Lab only).
ALL_REGISTRIES: Dict[str, List[Dict[str, TaskModuleItem]]] = {
    "reward": [REWARD_REGISTRY, IL_REWARD_REGISTRY],
    "termination": [TERMINATION_REGISTRY, IL_TERMINATION_REGISTRY],
    "observation": [IL_OBS_REGISTRY],
    "discriminator": [IL_DISC_REGISTRY],
}


def iter_all_items() -> Iterator[Tuple[str, str, TaskModuleItem]]:
    """Yield ``(kind, key, item)`` for every registered module across all kinds."""
    for kind, subs in ALL_REGISTRIES.items():
        for sub in subs:
            for key, item in sub.items():
                yield (kind, key, item)


def lookup(
    key: str,
    *,
    kind: str,
    backend: Optional[str] = None,
) -> Optional[TaskModuleItem]:
    """Kind-aware single-key lookup.

    Returns the item whose key matches within the requested kind's
    sub-registries, or ``None``. Use this instead of a flat lookup — it
    guarantees that a reward key like ``"base_height"`` never resolves
    to the termination entry of the same name (and vice versa).

    When ``backend`` is provided, only items whose ``backends`` include
    that backend are considered — this resolves *within-kind* collisions
    where the same key exists as both an SB3 and an Isaac Lab variant
    (e.g. ``joint_accel_penalty`` is registered with an empty ``il_func``
    on the SB3 side and with ``_unitport_joint_acc_l2`` on the IL side).

    When ``backend`` is None, the LAST matching item wins — this mirrors
    the original dict-merge order where Isaac Lab entries (registered
    later) shadow earlier SB3 entries for the same key.
    """
    match: Optional[TaskModuleItem] = None
    for sub in ALL_REGISTRIES.get(kind, []):
        item = sub.get(key)
        if item is None:
            continue
        if backend is not None and backend not in item.backends:
            continue
        match = item
    return match


def writable_registry(kind: str) -> Dict[str, TaskModuleItem]:
    """Return the primary writable sub-registry for a kind.

    Used by user-facing editors (e.g. RewardEditorPanel) to persist
    custom modules. The last sub-registry in the list is treated as the
    writable target (IL_REWARD_REGISTRY for rewards, etc.).
    """
    subs = ALL_REGISTRIES.get(kind, [])
    if not subs:
        raise KeyError(f"unknown registry kind: {kind!r}")
    return subs[-1]


def query_registry(
    *,
    kind: Optional[str] = None,
    backend: Optional[str] = None,
    algorithm: Optional[str] = None,
    family: Optional[str] = None,
) -> Dict[str, TaskModuleItem]:
    """Filter modules by kind / backend / algorithm / family.

    Parameters
    ----------
    kind : "reward" | "termination" | "observation" | "discriminator" | None
        Filter by function kind. None = all kinds.
    backend : "sb3" | "isaac_lab" | None
        Filter by engine compatibility. None = all.
    algorithm : "PPO" | "AMP" | "SAC" | None
        Filter by algorithm compatibility. None = all.
    family : "quadruped" | "biped" | None
        Filter by robot family. None = all.

    Returns
    -------
    Dict[str, TaskModuleItem]
        Filtered copy. When ``kind`` is given, only that kind's sub-
        registries are consulted — cross-kind key collisions cannot occur.
        When ``kind`` is None, items from different kinds sharing a key
        will overwrite each other in the returned flat dict; callers that
        need cross-kind visibility should use :func:`iter_all_items`.
    """
    result: Dict[str, TaskModuleItem] = {}
    kinds = [kind] if kind is not None else list(ALL_REGISTRIES.keys())
    for k in kinds:
        for sub in ALL_REGISTRIES.get(k, []):
            for key, item in sub.items():
                if backend is not None and backend not in item.backends:
                    continue
                if algorithm is not None:
                    if ALG_ALL not in item.algorithms and algorithm not in item.algorithms:
                        continue
                if family is not None:
                    if item.applicable_families and family not in item.applicable_families:
                        continue
                result[key] = item
    return result


def validate_keys(
    keys,
    kind: str,
    backend: str,
) -> tuple:
    """Validate reward/termination keys against the unified registry.

    Returns
    -------
    (matched, unmatched) : (Dict[str, TaskModuleItem], Set[str])
        matched = items found in the registry for this kind+backend.
        unmatched = keys not found (unknown or wrong backend).
    """
    valid = query_registry(kind=kind, backend=backend)
    matched = {k: valid[k] for k in keys if k in valid}
    unmatched = {k for k in keys if k not in valid}
    return matched, unmatched


def emit_inline_overrides(
    kind: str,
    out_path,
    *,
    backend: Optional[str] = None,
) -> int:
    """Serialise sub-registries for ``kind`` to a JSON sidecar.

    Used to bridge the main process (where the editor mutates the
    in-memory registry) and the training subprocess (which re-imports
    the registry fresh and would otherwise lose user edits).

    Walks **all** sub-registries of the given kind (not just the
    writable one) so SB3-side reward edits in ``REWARD_REGISTRY``
    survive alongside IL-side edits in ``IL_REWARD_REGISTRY``.

    The sidecar format is intentionally minimal::

        {
          "kind": "discriminator",
          "entries": {
            "disc_predict_reward": {"il_inline": "def ..."},
            ...
          }
        }

    Parameters
    ----------
    kind:
        Registry kind ("reward" / "termination" / "observation" /
        "discriminator").
    out_path:
        File to write. May be str or Path; parent dirs created on demand.
    backend:
        Optional backend filter ("sb3" / "isaac_lab"). When supplied,
        only entries whose ``backends`` set contains this value are
        emitted — the subprocess on that backend will not see (and
        attempt to exec) source intended for the other backend.

    Returns the number of entries written.
    """
    import json
    from pathlib import Path

    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)

    entries: Dict[str, dict] = {}
    for sub in ALL_REGISTRIES.get(kind, []):
        for key, item in sub.items():
            if not item.il_inline:
                continue
            if backend is not None and backend not in item.backends:
                continue
            # Later sub-registries (e.g. IL_REWARD_REGISTRY) intentionally
            # override earlier ones (REWARD_REGISTRY) when the same key
            # is present — matches the editor's write-back precedence
            # ("write into whichever sub holds the key"), and keeps the
            # subprocess's view consistent with what the editor showed.
            entries[key] = {"il_inline": item.il_inline}

    payload = {"kind": kind, "entries": entries}
    if backend is not None:
        payload["backend"] = backend
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(entries)


__all__ = [
    "ALL_REGISTRIES",
    "iter_all_items",
    "lookup",
    "writable_registry",
    "query_registry",
    "validate_keys",
    "emit_inline_overrides",
]
