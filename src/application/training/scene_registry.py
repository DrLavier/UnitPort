# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
+ ``<USER_CONFIG_DIR>/registers/scenes_custom.json`` user overlay (CLAUDE.md §4
naming convention) — the dataclass + API remain backwards compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from unitport_sdk import Paths, log_warning, push_data, read_data

#: User-overlay catalogue of imported custom-terrain scenes, under the LIVE
#: USER_CONFIG_DIR (CLAUDE.md §4). Mirrors the registers/*_custom.json
#: convention. Resolved at call time so workspace hot-switches are picked up.
_USER_SCENES_REL = "registers/scenes_custom.json"
#: Where imported heightfield .npz assets are stored (USER_CONFIG_DIR-relative).
_TERRAIN_ASSET_DIR_REL = "registers/terrain_assets"


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
    """Re-apply the in-code default scene table + user overlay. Idempotent."""
    _install_defaults()
    load_user_scenes()


# ---------------------------------------------------------------------------
# User-overlay custom-terrain scenes (scenes_custom.json)
# ---------------------------------------------------------------------------


def _user_scenes_path() -> Path:
    """Lazy resolve of the user scene-overlay path inside the LIVE
    USER_CONFIG_DIR (mirrors registers.ir._user_custom_path)."""
    return Paths.USER_CONFIG_DIR / "registers" / "scenes_custom.json"


def _register_custom_scene_entry(entry: Dict[str, Any]) -> bool:
    """Register one custom-terrain scene from a scenes_custom.json entry.

    Pure (no I/O). The entry's ``file_path`` (canonical heightfield .npz)
    and ``vertical_scale`` are wired into the scene's ``defaults`` so that
    picking it in the canvas scene picker auto-sets scene_type='custom' +
    custom_terrain_path + custom_terrain_vertical_scale (the picker writes
    ``defaults`` into sibling params). Returns False on a malformed entry
    (logged, skipped) rather than aborting the whole overlay load.
    """
    sid = str(entry.get("scene_id", "") or "").strip()
    fpath = str(entry.get("file_path", "") or "").strip()
    if not sid or not fpath:
        log_warning(
            f"[scene_registry] skipping custom scene with empty scene_id / "
            f"file_path: {entry!r}"
        )
        return False
    vscale = entry.get("vertical_scale", 0.005)
    try:
        vscale = float(vscale)
    except (TypeError, ValueError):
        vscale = 0.005
    backends = set(entry.get("supported_backends") or {"sb3_mujoco", "isaac_lab"})
    review = set(entry.get("review_backends") or {"mujoco"})
    register_scene(Scene(
        scene_id=sid,
        name=str(entry.get("name", sid) or sid),
        description=str(entry.get("description", "") or ""),
        scene_type="custom",
        file_path=Path(fpath),
        supported_backends=backends,
        supported_families=set(entry.get("supported_families") or set()),
        review_backends=review,
        defaults={
            "scene_type": "custom",
            "custom_terrain_path": fpath,
            "custom_terrain_vertical_scale": vscale,
        },
    ))
    return True


def load_user_scenes() -> int:
    """Merge user-imported custom-terrain scenes from scenes_custom.json.

    No-op (returns 0) before the install wizard has established
    USER_CONFIG_DIR (CLAUDE.md §4 — no fallback). Returns the count of
    scenes registered.
    """
    if not Paths.is_user_config_dir_set():
        return 0
    path = _user_scenes_path()
    try:
        payload = read_data(path)
    except Exception as exc:  # noqa: BLE001 - missing/corrupt overlay → skip
        log_warning(f"[scene_registry] could not read {path}: {exc}")
        return 0
    if not isinstance(payload, dict):
        return 0
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return 0
    n = 0
    for entry in scenes:
        if isinstance(entry, dict) and _register_custom_scene_entry(entry):
            n += 1
    return n


def import_terrain_scene(
    source_path: Any,
    scene_id: str,
    name: str,
    *,
    description: str = "",
    size_x: Optional[float] = None,
    size_y: Optional[float] = None,
    elevation_z: Optional[float] = None,
    base_z: float = 0.0,
    vertical_scale: float = 0.005,
    supported_backends: Optional[Set[str]] = None,
    review_backends: Optional[Set[str]] = None,
) -> str:
    """Import an external height-map as a reusable custom-terrain scene.

    Converts ``source_path`` (PNG / .npy / .npz) to a canonical heightfield
    ``.npz`` stored under ``USER_CONFIG_DIR/registers/terrain_assets/``,
    appends an entry to ``scenes_custom.json``, and registers the scene so
    it appears in the canvas scene picker. Returns the absolute ``.npz``
    path (the value written into the canvas ``custom_terrain_path``).

    Requires USER_CONFIG_DIR to be established (raises via
    ``Paths.require_user_config_dir`` otherwise — §4, no fallback).
    """
    from application.training.terrain.asset_import import (
        convert_to_canonical_npz_bytes,
    )

    sid = str(scene_id).strip()
    if not sid:
        raise SceneValidationError("import_terrain_scene: scene_id must be non-empty")
    Paths.require_user_config_dir()  # fail-loud if not configured (§4)

    npz_bytes, meta = convert_to_canonical_npz_bytes(
        source_path,
        size_x=size_x, size_y=size_y, elevation_z=elevation_z, base_z=base_z,
    )

    asset_rel = f"{_TERRAIN_ASSET_DIR_REL}/{sid}.npz"
    if not push_data(asset_rel, npz_bytes):
        raise RuntimeError(
            f"import_terrain_scene: failed to write terrain asset {asset_rel}"
        )
    asset_abs = str((Paths.USER_CONFIG_DIR / asset_rel).resolve())

    entry = {
        "scene_id": sid,
        "name": str(name) or sid,
        "description": str(description),
        "file_path": asset_abs,
        "vertical_scale": float(vertical_scale),
        "sha256": meta["sha256"],
        "supported_backends": sorted(supported_backends or {"sb3_mujoco", "isaac_lab"}),
        "review_backends": sorted(review_backends or {"mujoco"}),
    }
    # Append/replace in scenes_custom.json (read-modify-write).
    payload = read_data(_user_scenes_path()) if Paths.is_user_config_dir_set() else None
    if not isinstance(payload, dict):
        payload = {"schema": "unitport.scenes_custom/v1", "scenes": []}
    scenes = [s for s in (payload.get("scenes") or []) if s.get("scene_id") != sid]
    scenes.append(entry)
    payload["scenes"] = scenes
    if not push_data(_USER_SCENES_REL, payload):
        raise RuntimeError(
            "import_terrain_scene: failed to write scenes_custom.json overlay"
        )

    _register_custom_scene_entry(entry)
    return asset_abs


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
        supported_backends={"sb3_mujoco", "isaac_lab"},
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
            "curriculum_enabled": "true",
            "difficulty_levels": 10,
            "height_scan_enabled": "true",
            "scan_resolution": 0.1,
            "scan_size_x": 1.6,
            "scan_size_y": 1.0,
        },
    ))


_install_defaults()
# Merge user-imported custom-terrain scenes (no-op until the install wizard
# establishes USER_CONFIG_DIR — §4).
load_user_scenes()


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
    "load_user_scenes",
    "import_terrain_scene",
]
