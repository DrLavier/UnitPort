# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AMP discriminator preset aggregator (Isaac Lab only).

Three editable function bodies that drive ``AMPDiscriminator`` behaviour.
Each entry lives in its own file under ``isaac_lab/`` and exports a
single ``ENTRY: TaskModuleItem`` whose ``il_inline`` carries the default
Python source. The RewardEditorPanel (parameterised on kind) lets the
user inspect and override these bodies. ``AMPDiscriminator.__init__``
execs any user-edited source into a callable and binds it as a
method-replacement guard via the ``_OVERRIDE_SLOTS`` table.

Slot contract — DO NOT add new keys without first extending
``_OVERRIDE_SLOTS`` in
``application/training/amp/algorithms/discriminator.py``.
The three slot keys MUST match ``_OVERRIDE_SLOTS`` exactly:

    ("disc_forward",          "_user_fn_forward",         "forward")
    ("disc_compute_grad_pen", "_user_fn_grad_pen",        "compute_grad_pen")
    ("disc_predict_reward",   "_user_fn_predict_reward",  "predict_amp_reward")

These three are NOT per-robot presets — the GAN architecture and
loss shape are robot-agnostic by design.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from scripts.task_module import TaskModuleItem


def _collect(subpkg: str) -> Dict[str, TaskModuleItem]:
    pkg = importlib.import_module(f"scripts.discriminator.{subpkg}")
    out: Dict[str, TaskModuleItem] = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"scripts.discriminator.{subpkg}.{m.name}")
        entry = getattr(mod, "ENTRY", None)
        if entry is None:
            continue
        if entry.key in out:
            raise RuntimeError(
                f"[scripts.discriminator.{subpkg}] duplicate ENTRY.key "
                f"{entry.key!r} in {mod.__name__}"
            )
        out[entry.key] = entry
    return out


IL_DISC_REGISTRY: Dict[str, TaskModuleItem] = _collect("isaac_lab")


def il_disc_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_DISC_REGISTRY)


__all__ = [
    "IL_DISC_REGISTRY",
    "il_disc_registry",
]
