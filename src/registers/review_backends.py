# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.review_backends — Review-engine catalogue (process-local).

DEMO 对应：``DEMO/src/system/training/review_backends.py``.

Separate from the *training-side* backend concept (SB3 vs Isaac Lab):
a **review backend** is the physics simulator that runs the trained policy
for inspection after training:

    * ``mujoco``     — local MJCF viewer, fast startup
    * ``isaac_sim``  — Isaac Lab Kit subprocess viewer
    * ``newton``     — (future) Nvidia Newton warp-based simulator

Consumed by:
    * Export 节点 ``review_backend_picker`` widget (param_rows.py
      ``ReviewBackendPickerRow._resolve_backend_choices``) — populates the
      dropdown choices + availability badge.
    * Export 节点 ``review_scene_picker`` widget — pairs with
      ``application.training.scene_registry.list_review_scenes`` to filter
      scenes by review-engine compatibility.

No Qt imports. Catalog is loaded at module import time. ``available`` 标志
随平台环境（Isaac Sim 安装、CUDA 等）变化时由调用方走 ``register_backend``
覆写；当前阶段为静态默认值（mujoco / isaac_sim ON, newton OFF）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Canonical backend ids
# ---------------------------------------------------------------------------

BACKEND_MUJOCO = "mujoco"
BACKEND_ISAAC_SIM = "isaac_sim"
BACKEND_NEWTON = "newton"


# ---------------------------------------------------------------------------
# ReviewBackend dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReviewBackend:
    """One registered review physics backend."""

    backend_id: str
    display_name: str
    description: str
    scene_file_kinds: Set[str] = field(default_factory=set)
    """Scene file extensions this backend can consume: mjcf / usd / urdf."""
    available: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "display_name": self.display_name,
            "description": self.description,
            "scene_file_kinds": sorted(self.scene_file_kinds),
            "available": bool(self.available),
        }


# ---------------------------------------------------------------------------
# Process-local registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, ReviewBackend] = {}


def register_backend(backend: ReviewBackend) -> None:
    """Register or overwrite a review backend."""
    if not backend.backend_id:
        raise ValueError("ReviewBackend.backend_id must be non-empty")
    _REGISTRY[backend.backend_id] = backend


def get_backend(backend_id: str) -> ReviewBackend:
    if backend_id not in _REGISTRY:
        raise KeyError(
            f"Review backend {backend_id!r} is not registered. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[backend_id]


def has_backend(backend_id: str) -> bool:
    return backend_id in _REGISTRY


# ---------------------------------------------------------------------------
# Availability mapping: review-backend → installed-engine row in
# backends_installed.json. The review backend display flag in the picker
# is the AND of (a) we know how to launch this backend AND (b) the
# underlying engine is currently installed on this machine.
#
# When a review backend has no engine dependency (e.g. mujoco today —
# the viewer ships with the SDK's required ``mujoco`` Python package),
# the entry maps to ``None`` and we treat it as always-available.
# ---------------------------------------------------------------------------

_BACKEND_ENGINE_DEPENDENCY: Dict[str, Optional[str]] = {
    BACKEND_MUJOCO: None,           # mujoco is a hard SDK dep — always available
    BACKEND_ISAAC_SIM: "isaac_lab", # USD viewer requires the registered Isaac Lab install
    BACKEND_NEWTON: None,           # placeholder; flips available=False below regardless
}


def _engine_available_now(engine_id: Optional[str]) -> bool:
    """Resolve a review backend's runtime availability via registers.backends.

    Returns True when there is no engine dependency (e.g. mujoco), or the
    dependent engine is marked available in
    ``backends_installed.json::engines.<id>.available``. Returns False
    otherwise — including when the registers.backends module fails to
    import (defensive: review_backends must not crash the canvas just
    because the engine catalog is mid-migration).
    """
    if engine_id is None:
        return True
    try:
        from registers import backends as _b
        return bool(_b.is_available(engine_id))
    except Exception:
        return False


def _refresh_availability_from_installed_table() -> None:
    """Sync every registered review backend's ``available`` flag with the
    underlying engine's installed-state. Called from
    :func:`list_review_backends` before returning, so every picker query
    sees the current truth.

    Newton stays available=False (placeholder) regardless — the gate is
    "wired into UnitPort", not "engine installed".
    """
    for backend in _REGISTRY.values():
        dep = _BACKEND_ENGINE_DEPENDENCY.get(backend.backend_id)
        if backend.backend_id == BACKEND_NEWTON:
            backend.available = False
            continue
        backend.available = _engine_available_now(dep)


def list_review_backends(available_only: bool = False) -> List[ReviewBackend]:
    """Return all registered review backends in a stable order.

    Order: mujoco first (default), isaac_sim second, others alphabetic.
    Always returns the full set unless ``available_only=True`` — the picker
    keeps unavailable backends visible (greyed) so users discover what's
    coming in future releases.

    Each call refreshes the ``available`` flag from
    ``backends_installed.json`` so the picker reflects the latest engine
    install state (e.g. user just registered an Isaac Lab path via the
    Engine Settings dialog and triggered ``refresh_engine_availability``).
    """
    _refresh_availability_from_installed_table()
    items = list(_REGISTRY.values())
    if available_only:
        items = [b for b in items if b.available]
    _priority = {BACKEND_MUJOCO: 0, BACKEND_ISAAC_SIM: 1, BACKEND_NEWTON: 2}
    return sorted(
        items,
        key=lambda b: (_priority.get(b.backend_id, 99), b.backend_id),
    )


def default_backend_id() -> str:
    """Return the user-facing default (Export node fresh-drop)."""
    return BACKEND_MUJOCO


# ---------------------------------------------------------------------------
# Bundled defaults
# ---------------------------------------------------------------------------


def _install_defaults() -> None:
    register_backend(ReviewBackend(
        backend_id=BACKEND_MUJOCO,
        display_name="MuJoCo",
        description=(
            "Local MJCF viewer — fast startup, runs in the main process, "
            "ideal for quick policy sanity checks and gamepad piloting."
        ),
        scene_file_kinds={"mjcf"},
        available=True,
    ))

    register_backend(ReviewBackend(
        backend_id=BACKEND_ISAAC_SIM,
        display_name="Isaac Sim",
        description=(
            "Isaac Lab Kit subprocess — high-fidelity USD viewer, matches "
            "the training renderer 1:1. Slower startup."
        ),
        scene_file_kinds={"usd"},
        available=True,
    ))

    register_backend(ReviewBackend(
        backend_id=BACKEND_NEWTON,
        display_name="Newton",
        description=(
            "Nvidia Newton — warp-based simulator. Placeholder; not yet "
            "wired into UnitPort."
        ),
        scene_file_kinds={"usd", "mjcf"},
        available=False,
    ))


_install_defaults()


__all__ = [
    "BACKEND_MUJOCO",
    "BACKEND_ISAAC_SIM",
    "BACKEND_NEWTON",
    "ReviewBackend",
    "register_backend",
    "get_backend",
    "has_backend",
    "list_review_backends",
    "default_backend_id",
]
