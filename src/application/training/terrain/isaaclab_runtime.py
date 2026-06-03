# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab Kit-worker runtime for the custom heightfield sub-terrain.

This module is imported by the **generated env_cfg inside the IsaacLab
Kit worker venv** (the same way the generated config imports
``application.training.amp.mdp_events``), NOT by the main app / unit
tests — it imports ``isaaclab`` / ``trimesh`` at load time, which only
exist in the Kit venv. The venv311 test suite only ``py_compile``s it.

It defines:

  * :func:`custom_heightfield_terrain` — a ``SubTerrainBaseCfg.function``
    implementing the ``(difficulty, cfg) -> ([trimesh], origin)`` contract.
    It loads the user heightfield ``.npz`` (canonical absolute metres,
    ``row→+X / col→+Y``), discretises to int16 via the shared
    :func:`application.training.terrain.isaaclab_lowering.build_int16_heights`
    and hands it to IsaacLab's ``convert_height_field_to_mesh`` — no
    transpose (IsaacLab's mesh convention already matches the canonical;
    only the MuJoCo side transposes).
  * :class:`CustomHeightFieldTerrainCfg` — the sub-terrain cfg carrying
    the npz path + scales, with ``function`` wired to the above.

``difficulty`` is accepted (the TerrainGenerator always passes it) but
unused this phase: custom terrain is a single fixed tile, not a
curriculum grid (施工规划 v2 §1 — curriculum deferred). The signature
keeps the hook so a future difficulty-scaled variant drops straight in.
"""
from __future__ import annotations

import numpy as np
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from application.training.terrain.isaaclab_lowering import build_int16_heights


def custom_heightfield_terrain(difficulty, cfg):
    """Build the user heightfield tile as a trimesh. ``(difficulty, cfg)``
    → ``([trimesh.Trimesh], origin)`` per the SubTerrainBaseCfg contract."""
    import trimesh
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

    npz = np.load(cfg.heightfield_path)
    if "heights" not in getattr(npz, "files", []):
        raise ValueError(
            f"custom_heightfield_terrain: {cfg.heightfield_path!r} has no "
            f"'heights' key (present: {getattr(npz, 'files', None)!r})."
        )
    heights_m = np.asarray(npz["heights"], dtype=np.float32)
    if heights_m.ndim != 2:
        raise ValueError(
            f"custom_heightfield_terrain: 'heights' must be 2-D, got shape "
            f"{heights_m.shape!r}"
        )

    hf_int = build_int16_heights(heights_m, cfg.vertical_scale)
    vertices, triangles = convert_height_field_to_mesh(
        hf_int, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    n_rows, n_cols = heights_m.shape
    cx = 0.5 * (n_rows - 1) * cfg.horizontal_scale
    cy = 0.5 * (n_cols - 1) * cfg.horizontal_scale
    cz = float(heights_m[n_rows // 2, n_cols // 2])
    origin = np.array([cx, cy, cz], dtype=np.float64)
    return [mesh], origin


@configclass
class CustomHeightFieldTerrainCfg(SubTerrainBaseCfg):
    """Sub-terrain cfg for a user-imported heightfield ``.npz``."""

    function = custom_heightfield_terrain

    #: Absolute path to the heightfield ``.npz`` (canonical metres).
    heightfield_path: str = ""
    #: Isotropic horizontal cell size (m). Set by the env_cfg compiler.
    horizontal_scale: float = 0.1
    #: Vertical discretisation (m) for the int16 height grid.
    vertical_scale: float = 0.005
    #: None keeps the grid faithful (no vertical-surface correction).
    slope_threshold = None


__all__ = ["custom_heightfield_terrain", "CustomHeightFieldTerrainCfg"]
