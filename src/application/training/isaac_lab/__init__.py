# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab subprocess training backend (RELEASE port).

Public surface (lazily imported — see below):
    IsaacLabConfig          — pure data, knows how to build subprocess args
    IsaacLabBackend         — subprocess orchestrator + stdout parser
    IsaacLabTrainingTask    — TrainingTask wrapper for canvas / TasksManager

Lazy package init (PEP 562): these three symbols pull in ``unitport_sdk``
(hence PyQt6) through the logging bridge, but the **launcher subprocess** runs
under Isaac Sim's bundled Python — which has no PyQt6 — and legitimately
imports sibling submodules of this package (``headed_experience``,
``reference_state_init``, …). Importing those must NOT drag the backend (and
PyQt6) in as a package-import side effect, or the subprocess dies with
``ModuleNotFoundError: No module named 'PyQt6'`` before Isaac Sim even starts.
So we resolve the public names only on first attribute access, mirroring the
lazy pattern in ``application/ui/dialogs/__init__.py``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-checker visibility without the runtime PyQt6 cost
    from .backend import IsaacLabBackend
    from .config import IsaacLabConfig
    from .task import IsaacLabTrainingTask

__all__ = [
    "IsaacLabConfig",
    "IsaacLabBackend",
    "IsaacLabTrainingTask",
]

# Public name → submodule that defines it. Imported on first access only.
_LAZY: dict[str, str] = {
    "IsaacLabConfig": ".config",
    "IsaacLabBackend": ".backend",
    "IsaacLabTrainingTask": ".task",
}


def __getattr__(name: str):
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(submodule, __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)
