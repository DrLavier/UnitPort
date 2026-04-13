"""Review backend registry — the "use the trained brain in a new
environment" engine catalogue.

Separate from the training-side backend concept (SB3 vs Isaac Lab).
A review backend is the **physics simulator** that runs the trained
policy for inspection after training:

    * ``mujoco``     — local MJCF ray-cast viewer, fast startup
    * ``isaac_sim``  — Isaac Lab subprocess + persistent Kit viewer
    * ``newton``     — (future) Nvidia Newton warp-based simulator

A canvas scene is compatible with a review backend when the scene's
``review_backends`` set contains the backend id. The Export node's
``review_scene_picker`` widget filters the Scene registry by this
relation so users only see scenes the chosen engine can actually load.

No Qt imports. The catalogue is process-local and re-loaded at module
import time; future iterations will probe the environment at startup
to flip ``available`` based on whether the backend subprocess can
actually launch (e.g. Isaac Sim installed + CUDA present, Newton
installed, …).
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
    """One registered review physics backend.

    Identity is ``backend_id``; ``display_name`` is what the picker
    surfaces. ``scene_file_kinds`` enumerates the asset file types
    this backend can load (used for future auto-detection of scene
    compatibility when an explicit ``review_backends`` entry is
    missing on a Scene registry entry).

    ``available`` is True when the backend subprocess can actually
    launch in the current environment. Placeholder backends like
    Newton ship disabled so the picker still lists them (for
    discoverability) but greys out the button.
    """

    backend_id: str
    display_name: str
    description: str
    scene_file_kinds: Set[str] = field(default_factory=set)
    """Scene file extensions / markers this backend can consume:
    ``mjcf`` (\\*.xml), ``usd`` (\\*.usd, nucleus://), ``urdf``."""
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
# Registry
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


def list_backends(available_only: bool = False) -> List[ReviewBackend]:
    """Return all registered backends in a stable order.

    Order: mujoco first (default), isaac_sim second, others alphabetic.
    Always returns the full set unless ``available_only=True``; the
    picker keeps unavailable backends visible so users know what's
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
    """Return the user-facing default when the Export node is freshly
    dropped. Matches the "MuJoCo first" decision."""
    return BACKEND_MUJOCO


# ---------------------------------------------------------------------------
# Bundled defaults
# ---------------------------------------------------------------------------


def _install_defaults() -> None:
    register_backend(ReviewBackend(
        backend_id=BACKEND_MUJOCO,
        display_name="MuJoCo",
        description=(
            "Local MJCF viewer — fast startup, runs in the main "
            "process, ideal for quick policy sanity checks and "
            "gamepad piloting of the trained brain."
        ),
        scene_file_kinds={"mjcf"},
        available=True,
    ))

    register_backend(ReviewBackend(
        backend_id=BACKEND_ISAAC_SIM,
        display_name="Isaac Sim",
        description=(
            "Isaac Lab Kit subprocess — high-fidelity USD viewer, "
            "matches the training-time renderer 1:1. Slower startup "
            "but supports the persistent hot-swap workflow described "
            "in IsaacSim_design.yaml."
        ),
        scene_file_kinds={"usd"},
        available=True,
    ))

    register_backend(ReviewBackend(
        backend_id=BACKEND_NEWTON,
        display_name="Newton",
        description=(
            "Nvidia Newton — warp-based simulator. Placeholder entry; "
            "not yet wired into UnitPort."
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
    "list_backends",
    "default_backend_id",
]
