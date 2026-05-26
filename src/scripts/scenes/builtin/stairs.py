# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Built-in scene — stairs (Isaac Lab + biped family only)."""

from __future__ import annotations

from scripts.scenes.registry import Scene


ENTRY = Scene(
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
)
