"""Built-in scene — rough terrain (Isaac Lab only)."""

from __future__ import annotations

from scripts.scenes.registry import Scene


ENTRY = Scene(
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
)
