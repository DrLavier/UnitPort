"""Scene asset registry — single source of truth for training playgrounds.

Holds the canonical list of scenes/arenas that the canvas ``PlayGround
Setting`` node can pick from. Filtered by training backend (SB3 /
Isaac Lab) and robot family (quadruped / biped / wheeled) so the
picker only shows entries that are actually runnable in the current
canvas context.

Scene metadata is read-only at run time — user overrides (gravity,
friction, roughness, …) live on the canvas node, while identity /
file path / capability flags come from this module.

Process-local dict, loaded at import time from an in-code default
table. A future pass will add ``rescan()`` that walks project scene
directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class SceneValidationError(ValueError):
    """Raised when a candidate Scene fails validation."""


# ---------------------------------------------------------------------------
# Scene dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """Registered playground scene.

    Identity + capability flags are authoritative; user-editable knobs
    (gravity, friction, roughness, curriculum…) live on the
    ``PlayGroundSettingNode`` on the canvas and are written per-run.
    """

    scene_id: str
    name: str
    description: str
    scene_type: str                 # flat | rough | stairs | custom
    file_path: Optional[Path] = None
    file_url: str = ""              # nucleus:// or http:// (Isaac Lab)
    supported_backends: Set[str] = field(default_factory=set)
    """Training backend compatibility (sb3 / isaac_lab). Used by the
    PlayGround Setting scene picker during training setup."""
    supported_families: Set[str] = field(default_factory=set)
    """Empty set ⇒ supported on all families."""
    review_backends: Set[str] = field(default_factory=set)
    """Review engine compatibility (``mujoco`` / ``isaac_sim`` / future
    ``newton``). Used by the Export node's review_scene_picker widget
    to filter legal review destinations — most of the time a trained
    policy is reviewed in a DIFFERENT scene than it was trained in
    (e.g. an Isaac-trained brain validated against a flat MuJoCo arena),
    so the review picker does not care about the training-side
    supported_backends at all."""
    defaults: Dict[str, Any] = field(default_factory=dict)
    """Recommended parameter defaults that the picker auto-writes into
    the node parameters when the user selects this scene."""

    def supports(self, backend: Optional[str], family: Optional[str]) -> bool:
        if backend and backend not in self.supported_backends:
            return False
        if (
            family
            and self.supported_families
            and family not in self.supported_families
        ):
            return False
        return True

    def supports_review(
        self,
        review_backend: Optional[str],
        family: Optional[str] = None,
    ) -> bool:
        """True iff this scene can be loaded by the given review engine.

        Family filter is applied the same way as :meth:`supports`: a
        quadruped-only scene is rejected for biped review targets
        regardless of the review backend. ``None`` for either field
        skips that check.
        """
        if review_backend and review_backend not in self.review_backends:
            return False
        if (
            family
            and self.supported_families
            and family not in self.supported_families
        ):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "description": self.description,
            "scene_type": self.scene_type,
            "file_path": str(self.file_path) if self.file_path else "",
            "file_url": self.file_url,
            "supported_backends": sorted(self.supported_backends),
            "supported_families": sorted(self.supported_families),
            "review_backends": sorted(self.review_backends),
            "defaults": dict(self.defaults),
        }


# ---------------------------------------------------------------------------
# Process-local registry
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, Scene] = {}


def register_scene(scene: Scene) -> None:
    """Register or overwrite a Scene. Re-registration is intentional
    (hot-reload from future rescan paths)."""
    if not scene.scene_id:
        raise SceneValidationError("Scene.scene_id must be non-empty")
    if scene.scene_type not in ("flat", "rough", "stairs", "custom"):
        raise SceneValidationError(
            f"Scene.scene_type must be one of flat/rough/stairs/custom, "
            f"got {scene.scene_type!r}"
        )
    _REGISTRY[scene.scene_id] = scene


def get_scene(scene_id: str) -> Scene:
    if scene_id not in _REGISTRY:
        raise KeyError(
            f"Scene {scene_id!r} is not registered. "
            f"Known scenes: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[scene_id]


def has_scene(scene_id: str) -> bool:
    return scene_id in _REGISTRY


def list_scenes(
    backend: Optional[str] = None,
    family: Optional[str] = None,
) -> List[Scene]:
    """Return all registered scenes, optionally filtered by backend
    and/or family. Sorted by ``scene_id``."""
    items = [
        s for s in _REGISTRY.values()
        if s.supports(backend, family)
    ]
    return sorted(items, key=lambda s: s.scene_id)


def list_review_scenes(
    review_backend: Optional[str] = None,
    family: Optional[str] = None,
) -> List[Scene]:
    """Return scenes compatible with the given review engine.

    Intended for the Export node's ``review_scene_picker`` widget,
    which wants a "scenes this engine can load" list rather than
    "scenes this training backend was trained with". Review scene
    selection is deliberately **decoupled** from training scene
    selection — e.g. an Isaac-Lab trained policy can be reviewed in
    a MuJoCo flat arena as long as the scene has an MJCF file.
    """
    items = [
        s for s in _REGISTRY.values()
        if s.supports_review(review_backend, family)
    ]
    return sorted(items, key=lambda s: s.scene_id)


def clear_registry() -> None:
    """Drop all registered scenes. Test-only."""
    _REGISTRY.clear()


def rescan() -> None:
    """Re-apply the in-code default scene table.

    Placeholder for future filesystem discovery. Idempotent — safe to
    call from UI widget render callbacks.
    """
    _install_defaults()


# ---------------------------------------------------------------------------
# Built-in scene catalogue
# ---------------------------------------------------------------------------


def _install_defaults() -> None:
    register_scene(Scene(
        scene_id="flat_ground",
        name="Flat Ground",
        description=(
            "Infinite flat plane. Works with every backend and family — "
            "baseline arena for locomotion, standing, and pose tracking tasks."
        ),
        scene_type="flat",
        supported_backends={"sb3", "isaac_lab"},
        supported_families=set(),  # empty ⇒ all families
        review_backends={"mujoco", "isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 10.0,
            "arena_extent_y": 10.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "height_scan_enabled": "false",
        },
    ))

    register_scene(Scene(
        scene_id="rough_terrain",
        name="Rough Terrain",
        description=(
            "Procedural heightfield with configurable amplitude and slope. "
            "Pairs with a curriculum schedule so the policy ramps from easy "
            "to hard over training. Isaac Lab only — MJ side lacks "
            "heightfield randomisation today."
        ),
        scene_type="rough",
        supported_backends={"isaac_lab"},
        supported_families=set(),
        review_backends={"isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 100.0,
            "arena_extent_y": 100.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "roughness_amplitude": 0.08,
            "slope_max": 20.0,
            "curriculum_enabled": "true",
            "difficulty_levels": 10,
            "height_scan_enabled": "true",
            "scan_resolution": 0.1,
            "scan_size_x": 1.6,
            "scan_size_y": 1.0,
        },
    ))

    register_scene(Scene(
        scene_id="stairs",
        name="Stairs",
        description=(
            "Stair-case arena for biped locomotion. Fixed step geometry — "
            "not curriculum-controlled. Isaac Lab + biped family only."
        ),
        scene_type="stairs",
        supported_backends={"isaac_lab"},
        supported_families={"biped"},
        review_backends={"isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 20.0,
            "arena_extent_y": 20.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "height_scan_enabled": "true",
            "scan_resolution": 0.05,
            "scan_size_x": 1.2,
            "scan_size_y": 0.8,
        },
    ))


_install_defaults()


__all__ = [
    "Scene",
    "SceneValidationError",
    "register_scene",
    "get_scene",
    "has_scene",
    "list_scenes",
    "list_review_scenes",
    "clear_registry",
    "rescan",
]
