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
from typing import Dict, List, Set


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


def list_review_backends(available_only: bool = False) -> List[ReviewBackend]:
    """Return all registered review backends in a stable order.

    Order: mujoco first (default), isaac_sim second, others alphabetic.
    Always returns the full set unless ``available_only=True`` — the picker
    keeps unavailable backends visible (greyed) so users discover what's
    coming in future releases.
    """
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
