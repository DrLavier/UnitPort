"""Observation preset aggregator — Isaac Lab only.

Each observation lives in its own file under ``isaac_lab/`` and exports
a single ``ENTRY: TaskModuleItem``. There is no inline Python source
here, because Isaac Lab observation functions are picked from
``isaaclab.envs.mdp.observations`` by **key name** at compile time — see
``isaac_lab_config_compiler._obs_term_func`` for the binding step.

SB3 has no observation registry today (the env's obs builder is fixed);
this package is single-backend by design.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from scripts.task_module import TaskModuleItem


def _collect(subpkg: str) -> Dict[str, TaskModuleItem]:
    pkg = importlib.import_module(f"scripts.observations.{subpkg}")
    out: Dict[str, TaskModuleItem] = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"scripts.observations.{subpkg}.{m.name}")
        entry = getattr(mod, "ENTRY", None)
        if entry is None:
            continue
        if entry.key in out:
            raise RuntimeError(
                f"[scripts.observations.{subpkg}] duplicate ENTRY.key "
                f"{entry.key!r} in {mod.__name__}"
            )
        out[entry.key] = entry
    return out


IL_OBS_REGISTRY: Dict[str, TaskModuleItem] = _collect("isaac_lab")


def il_obs_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_OBS_REGISTRY)


def default_il_obs_terms() -> Dict[str, float]:
    """Default IL observation terms for a fresh canvas — every standard
    proprioceptive input enabled, plus the velocity command. Excludes
    ``height_scan`` because it requires a ray-caster sensor on the canvas
    (the IL compiler skips the term when no scanner is present, but
    starting it disabled avoids a silent surprise).
    """
    return {
        k: IL_OBS_REGISTRY[k].default
        for k in (
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_command",
            "joint_pos",
            "joint_vel",
            "last_action",
        )
    }


__all__ = [
    "IL_OBS_REGISTRY",
    "il_obs_registry",
    "default_il_obs_terms",
]
