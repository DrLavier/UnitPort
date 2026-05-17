"""application.training.scene_registry — Playground scene catalogue.

DEMO 对应：``DEMO/src/system/training/scene_registry.py``.

Single source of truth for canvas Playground / Review scenes. Filtered by
training backend (sb3 / isaac_lab) and robot family (quadruped / biped /
wheeled) so pickers only show entries that are actually runnable.

Consumed by:
    * Export 节点 ``review_scene_picker`` widget (param_rows.py
      ``ReviewScenePickerRow._resolve_scene_choices``) → ``list_review_scenes``.
    * (Future) PlayGround Setting node — uses ``list_scenes`` for the
      training-side scene picker.

Process-local dict, in-code defaults installed at import time. Future
iteration will add filesystem rescan from ``registers/data/scenes_review.json``
+ ``~/UnitPort/registers/scenes_custom.json`` user overlay (CLAUDE.md §4
naming convention) — the dataclass + API remain backwards compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SceneValidationError(ValueError):
    """Raised when a candidate Scene fails validation."""


# ---------------------------------------------------------------------------
# Scene dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """Registered playground / review scene."""

    scene_id: str
    name: str
    description: str
    scene_type: str                 # flat | rough | stairs | custom
    file_path: Optional[Path] = None
    file_url: str = ""
    supported_backends: Set[str] = field(default_factory=set)
    """Training backend compatibility (sb3 / isaac_lab)."""
    supported_families: Set[str] = field(default_factory=set)
    """Empty set ⇒ supported on all families."""
    review_backends: Set[str] = field(default_factory=set)
    """Review-engine compatibility (mujoco / isaac_sim / newton).

    Used by Export node's review_scene_picker to filter legal review
    destinations — review scene selection is deliberately decoupled from
    training scene selection (e.g. an Isaac-trained brain validated against
    a flat MuJoCo arena).
    """
    defaults: Dict[str, Any] = field(default_factory=dict)
    """Recommended parameter defaults the picker auto-writes when selected."""

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
    """Register or overwrite a Scene. Re-registration is intentional."""
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
    """Return all registered scenes filtered by training backend + family."""
    items = [s for s in _REGISTRY.values() if s.supports(backend, family)]
    return sorted(items, key=lambda s: s.scene_id)


def list_review_scenes(
    review_backend: Optional[str] = None,
    family: Optional[str] = None,
) -> List[Scene]:
    """Scenes compatible with the given review engine + family.

    Used by Export node ``review_scene_picker``. Decoupled from
    ``list_scenes`` because review scene selection is independent of the
    training-side scene used during the original training run.
    """
    items = [
        s for s in _REGISTRY.values()
        if s.supports_review(review_backend, family)
    ]
    return sorted(items, key=lambda s: s.scene_id)


def clear_registry() -> None:
    """Drop all registered scenes (test-only)."""
    _REGISTRY.clear()


def rescan() -> None:
    """Re-apply the in-code default scene table. Idempotent."""
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
            "baseline arena for locomotion / standing / pose tracking."
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
            "Pairs with a curriculum schedule. Isaac Lab only — MJ side "
            "lacks heightfield randomisation today."
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

    # NOTE: every entry below maps to scene_type="rough" so the
    # play_ground_setting enum (["flat","rough"]) and
    # env_cfg_compiler dispatch (flat→plane / rough→ROUGH_TERRAINS_CFG)
    # remain happy. scene_id is the fine-grained discriminator carried
    # downstream — a future env_cfg_compiler pass may key richer
    # terrain_generator presets off it.

    register_scene(Scene(
        scene_id="pyramid_stairs",
        name="Pyramid Stairs",
        description=(
            "Concentric stepped pyramids — robot starts at the centre and "
            "must climb. Drives stance recovery + foot clearance. Isaac Lab "
            "rough generator (one of the four ROUGH_TERRAINS_CFG tiles)."
        ),
        scene_type="rough",
        supported_backends={"isaac_lab"},
        supported_families={"biped", "quadruped"},
        review_backends={"isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 20.0,
            "arena_extent_y": 20.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "roughness_amplitude": 0.0,
            "slope_max": 0.0,
            "curriculum_enabled": "true",
            "difficulty_levels": 10,
            "height_scan_enabled": "true",
            "scan_resolution": 0.05,
            "scan_size_x": 1.2,
            "scan_size_y": 0.8,
        },
    ))

    register_scene(Scene(
        scene_id="slopes",
        name="Slopes",
        description=(
            "Continuous inclined planes at progressively steeper angles. "
            "Pairs with curriculum to ramp slope difficulty over training. "
            "Isaac Lab rough generator."
        ),
        scene_type="rough",
        supported_backends={"isaac_lab"},
        supported_families=set(),
        review_backends={"isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 60.0,
            "arena_extent_y": 60.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "roughness_amplitude": 0.02,
            "slope_max": 30.0,
            "curriculum_enabled": "true",
            "difficulty_levels": 10,
            "height_scan_enabled": "true",
            "scan_resolution": 0.1,
            "scan_size_x": 1.6,
            "scan_size_y": 1.0,
        },
    ))

    register_scene(Scene(
        scene_id="stepping_stones",
        name="Stepping Stones",
        description=(
            "Discrete platforms separated by gaps. Trains precise foot "
            "placement and gap-traversal. Isaac Lab rough generator."
        ),
        scene_type="rough",
        supported_backends={"isaac_lab"},
        supported_families={"biped", "quadruped"},
        review_backends={"isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 30.0,
            "arena_extent_y": 30.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "roughness_amplitude": 0.0,
            "slope_max": 0.0,
            "curriculum_enabled": "true",
            "difficulty_levels": 8,
            "height_scan_enabled": "true",
            "scan_resolution": 0.05,
            "scan_size_x": 1.6,
            "scan_size_y": 1.0,
        },
    ))

    register_scene(Scene(
        scene_id="discrete_obstacles",
        name="Discrete Obstacles",
        description=(
            "Boxes / pillars scattered across an otherwise flat arena. "
            "Trains obstacle avoidance + recovery on push contact. Isaac "
            "Lab rough generator (also available to MuJoCo via geom props)."
        ),
        scene_type="rough",
        supported_backends={"isaac_lab"},
        supported_families=set(),
        review_backends={"isaac_sim", "mujoco"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 40.0,
            "arena_extent_y": 40.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "roughness_amplitude": 0.02,
            "slope_max": 0.0,
            "curriculum_enabled": "true",
            "difficulty_levels": 10,
            "height_scan_enabled": "true",
            "scan_resolution": 0.1,
            "scan_size_x": 1.6,
            "scan_size_y": 1.0,
        },
    ))

    register_scene(Scene(
        scene_id="gap_terrain",
        name="Gap Terrain",
        description=(
            "Wide flat tiles separated by chasms of varying width. "
            "Pushes the policy toward sustained jumps + landing recovery. "
            "Isaac Lab rough generator."
        ),
        scene_type="rough",
        supported_backends={"isaac_lab"},
        supported_families={"biped", "quadruped"},
        review_backends={"isaac_sim"},
        defaults={
            "gravity_z": -9.81,
            "arena_extent_x": 30.0,
            "arena_extent_y": 30.0,
            "friction_static": 1.0,
            "friction_dynamic": 0.8,
            "roughness_amplitude": 0.0,
            "slope_max": 0.0,
            "curriculum_enabled": "true",
            "difficulty_levels": 8,
            "height_scan_enabled": "true",
            "scan_resolution": 0.05,
            "scan_size_x": 1.6,
            "scan_size_y": 1.0,
        },
    ))

    register_scene(Scene(
        scene_id="mixed_rough",
        name="Mixed Rough Tiles",
        description=(
            "Standard ROUGH_TERRAINS_CFG mix — pyramid stairs + slopes + "
            "stepping stones + discrete obstacles, sampled per env. The "
            "fast-iteration default for general locomotion training."
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
