"""Reward preset aggregator — SB3 + Isaac Lab.

Each preset lives in its own file under ``sb3/`` or ``isaac_lab/`` and
exports a single ``ENTRY: TaskModuleItem`` constant. This module scans
both sub-packages at import time, materialises ``REWARD_REGISTRY`` and
``IL_REWARD_REGISTRY`` from the discovered entries, and preserves the
legacy public surface (helpers, recommended-terms tuple) unchanged.

The split was driven by user intent: ``src/scripts/`` is the canonical
home for training-time function source, and every function should have
its own file in the matching folder. The aggregator builds the dicts
once; downstream callers (UI editors, IL config compiler, resolver)
keep mutating the same dict instances they did before via
``dataclasses.replace(sub[key], il_inline=source)`` — moving the data
into files does not change identity.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Tuple

from scripts.task_module import TaskModuleItem


def _collect(subpkg: str) -> Dict[str, TaskModuleItem]:
    """Import every non-underscore module under ``rewards/<subpkg>/`` and
    pick up its ``ENTRY`` constant."""
    pkg = importlib.import_module(f"scripts.rewards.{subpkg}")
    out: Dict[str, TaskModuleItem] = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"scripts.rewards.{subpkg}.{m.name}")
        entry = getattr(mod, "ENTRY", None)
        if entry is None:
            continue
        if entry.key in out:
            raise RuntimeError(
                f"[scripts.rewards.{subpkg}] duplicate ENTRY.key "
                f"{entry.key!r} in {mod.__name__}"
            )
        out[entry.key] = entry
    return out


REWARD_REGISTRY:    Dict[str, TaskModuleItem] = _collect("sb3")
IL_REWARD_REGISTRY: Dict[str, TaskModuleItem] = _collect("isaac_lab")


def reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(REWARD_REGISTRY)


#: Reward terms that almost every legged-locomotion run benefits from.
#: Used both as the default seed when the canvas wires no rewards AND as
#: the keyset checked by ``warn_missing_recommended_reward_terms`` —
#: a partial canvas dict missing any of these will silently degrade to
#: weight 0 inside the env (``self._reward_terms.get(key, 0.0)``).
RECOMMENDED_LOCOMOTION_REWARD_TERMS: Tuple[str, ...] = (
    "velocity_tracking",
    "alive",
    "energy",
    "upright",
    "base_height_penalty",
    "action_smoothness",
    "dof_pos_limits",
)


def default_reward_terms() -> Dict[str, float]:
    """Default reward set seeded into the SB3 env when no canvas wiring exists."""
    return {
        key: REWARD_REGISTRY[key].default
        for key in RECOMMENDED_LOCOMOTION_REWARD_TERMS
    }


def warn_missing_recommended_reward_terms(
    user_terms: Optional[Dict[str, float]],
    *,
    logger=None,
) -> Tuple[str, ...]:
    """Emit a warning when the user-supplied reward dict skips a recommended term.

    Returns the tuple of missing keys (possibly empty). Pass ``logger`` to
    redirect output; otherwise prints to stderr via ``warnings.warn``.

    Only warns when the user *did* pass a non-empty dict — silence when
    the env will fall back to ``default_reward_terms()`` (which already
    includes everything) or when the user explicitly passed an empty
    dict (an explicit "I want zero reward" signal).
    """
    if not user_terms:
        return ()
    missing = tuple(
        k for k in RECOMMENDED_LOCOMOTION_REWARD_TERMS if k not in user_terms
    )
    if not missing:
        return ()
    msg = (
        "[task_module_registry] reward_terms is missing recommended "
        f"safety terms: {', '.join(missing)}. They will default to "
        "weight 0 — the policy may learn crouching / asymmetric gait "
        "to survive without these guards. Add them to the Canvas "
        "Rewards node or the user-passed reward_terms dict."
    )
    if logger is not None:
        try:
            logger.warning(msg)
        except Exception:
            print(msg, flush=True)
    else:
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return missing


def il_reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_REWARD_REGISTRY)


def default_il_reward_terms() -> Dict[str, float]:
    return {k: IL_REWARD_REGISTRY[k].default for k in IL_REWARD_REGISTRY}


__all__ = [
    "REWARD_REGISTRY",
    "IL_REWARD_REGISTRY",
    "RECOMMENDED_LOCOMOTION_REWARD_TERMS",
    "reward_registry",
    "il_reward_registry",
    "default_reward_terms",
    "default_il_reward_terms",
    "warn_missing_recommended_reward_terms",
]
