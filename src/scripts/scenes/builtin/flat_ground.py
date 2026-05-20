"""Built-in scene — flat ground."""

from __future__ import annotations

from scripts.scenes.registry import Scene


ENTRY = Scene(
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
)
