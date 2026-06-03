# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lower a :class:`HeightField` into an IsaacLab custom sub-terrain.

IsaacLab builds procedural terrain by calling a sub-terrain ``function``
that returns ``([trimesh], origin)``; a heightfield function returns a
2-D int16 array (units of ``vertical_scale``) which
``convert_height_field_to_mesh`` turns into a PhysX-collidable trimesh
(PV ① — proven the geological foundation). This module holds the **pure,
venv311-testable** half:

  * :func:`build_int16_heights` — absolute metres → int16 (vertical_scale
    units), with a fail-loud range check.
  * :func:`heightfield_to_isaaclab` — derive ``horizontal_scale`` /
    ``vertical_scale`` / size from a :class:`HeightField`, enforcing the
    isotropic-cell constraint IsaacLab imposes (one ``horizontal_scale``
    for both axes).
  * :func:`emit_custom_terrain_generator_cfg` — code-gen the
    ``TerrainGeneratorCfg`` literal the env_cfg compiler embeds (Step 5),
    referencing the Kit-worker runtime in
    ``application.training.terrain.isaaclab_runtime``.

Array convention (PV④): the canonical ``heights[i, j]`` (row→+X, col→+Y)
matches IsaacLab's ``convert_height_field_to_mesh`` exactly (vertex
``x=i*hscale, y=j*hscale``), so — unlike the MuJoCo side — **no transpose
is applied here**. The MuJoCo lowering transposes; this one does not. The
cross-engine gate (Step 3) is what proves the two stay byte-equivalent.

Isotropic-cell constraint
-------------------------
``convert_height_field_to_mesh`` uses a single ``horizontal_scale`` for
both axes, so a faithful IsaacLab tile needs square grid cells:
``size_x/(n_rows-1) == size_y/(n_cols-1)``. MuJoCo's ``<hfield>`` takes
independent ``radius_x``/``radius_y`` and does not need this — so the
constraint is enforced here (IsaacLab-specific), fail-loud, rather than
on the engine-agnostic contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from application.training.terrain.contract import HeightField


class IsaacLabTerrainLoweringError(RuntimeError):
    """Raised when a heightfield cannot be lowered into an IsaacLab
    sub-terrain (non-square cells, int16 overflow, bad ordering)."""


#: IsaacLab's stock heightfield vertical discretisation (m). int16 units.
DEFAULT_VERTICAL_SCALE = 0.005

#: Module-level config var name the generated env_cfg uses.
DEFAULT_GENERATOR_VAR = "UNITPORT_CUSTOM_TERRAIN_CFG"

#: int16 storage limit — heights beyond ±(this * vertical_scale) overflow.
_INT16_MAX = 32767


def build_int16_heights(heights_m: np.ndarray, vertical_scale: float) -> np.ndarray:
    """Absolute-metre heights → int16 array in ``vertical_scale`` units.

    Fail-loud (§8) on a vertical_scale that would overflow int16 — the
    caller must pick a coarser scale rather than letting the surface wrap.
    """
    if not (float(vertical_scale) > 0.0):
        raise IsaacLabTerrainLoweringError(
            f"build_int16_heights: vertical_scale must be > 0, got "
            f"{vertical_scale!r}"
        )
    arr = np.asarray(heights_m, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise IsaacLabTerrainLoweringError(
            "build_int16_heights: heights contain non-finite values (NaN/Inf)."
        )
    units = arr / np.float32(vertical_scale)
    peak = float(np.max(np.abs(units))) if units.size else 0.0
    if peak > _INT16_MAX:
        raise IsaacLabTerrainLoweringError(
            f"build_int16_heights: heights overflow int16 at vertical_scale="
            f"{vertical_scale} (peak {peak:.0f} units > {_INT16_MAX}). Use a "
            f"coarser vertical_scale."
        )
    return np.rint(units).astype(np.int16)


def heightfield_to_isaaclab(
    hf: HeightField,
    *,
    vertical_scale: float = DEFAULT_VERTICAL_SCALE,
    cell_tol: float = 1e-3,
) -> Dict[str, Any]:
    """Pure conversion: :class:`HeightField` → IsaacLab terrain params.

    Returns ``horizontal_scale`` / ``vertical_scale`` / ``size`` (the
    corner-origin tile extent IsaacLab expects) / ``heights_int16``.
    Raises :class:`IsaacLabTerrainLoweringError` on non-square cells or
    int16 overflow.
    """
    if hf.array_ordering != "row_major_xy":
        raise IsaacLabTerrainLoweringError(
            f"heightfield_to_isaaclab: unsupported array_ordering "
            f"{hf.array_ordering!r}; only 'row_major_xy' is implemented."
        )
    heights = np.ascontiguousarray(hf.heights, dtype=np.float32)
    if heights.ndim != 2 or heights.shape[0] < 2 or heights.shape[1] < 2:
        raise IsaacLabTerrainLoweringError(
            f"heightfield_to_isaaclab: heights must be 2-D >=2x2, got shape "
            f"{heights.shape!r}"
        )
    n_rows, n_cols = heights.shape
    cell_x = float(hf.size_x) / (n_rows - 1)
    cell_y = float(hf.size_y) / (n_cols - 1)
    if abs(cell_x - cell_y) > cell_tol * max(cell_x, cell_y):
        raise IsaacLabTerrainLoweringError(
            f"heightfield_to_isaaclab: IsaacLab uses one isotropic "
            f"horizontal_scale, so grid cells must be square, but "
            f"size_x/(n_rows-1)={cell_x:.6f} != size_y/(n_cols-1)={cell_y:.6f}. "
            f"Resample the heightfield to square cells before import."
        )
    hscale = cell_x
    heights_int16 = build_int16_heights(heights, vertical_scale)
    # The mesh IsaacLab builds spans [0, (n_rows-1)*hscale] x [0, (n_cols-1)*hscale].
    size = ((n_rows - 1) * hscale, (n_cols - 1) * hscale)
    return {
        "horizontal_scale": float(hscale),
        "vertical_scale": float(vertical_scale),
        "size": (float(size[0]), float(size[1])),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "heights_int16": heights_int16,
    }


def emit_custom_terrain_generator_cfg(
    npz_path: Union[Path, str],
    hf: HeightField,
    *,
    var_name: str = DEFAULT_GENERATOR_VAR,
    vertical_scale: float = DEFAULT_VERTICAL_SCALE,
) -> List[str]:
    """Emit the ``TerrainGeneratorCfg`` literal for a custom heightfield.

    The generated module (run in the Kit worker) imports the runtime
    sub-terrain cfg from ``application.training.terrain.isaaclab_runtime``
    and references ``npz_path`` so the heightfield is loaded at terrain
    build time. Validates the conversion up front (raises here, at compile
    time in venv311, rather than deep in the Kit worker).
    """
    conv = heightfield_to_isaaclab(hf, vertical_scale=vertical_scale)
    sx, sy = conv["size"]
    hscale = conv["horizontal_scale"]
    vscale = conv["vertical_scale"]
    path_literal = repr(str(Path(npz_path)))
    return [
        "# " + "=" * 70,
        "# UnitPort custom heightfield terrain — user-imported single source.",
        "# Built by application.training.terrain.isaaclab_runtime at terrain",
        "# build time from the embedded .npz; one isotropic horizontal_scale",
        "# (square cells enforced at compile). slope_threshold=None keeps the",
        "# grid faithful (no vertical-surface correction) so it matches the",
        "# MuJoCo <hfield> the cross-engine gate compares against.",
        "# " + "=" * 70,
        "from application.training.terrain.isaaclab_runtime import (",
        "    CustomHeightFieldTerrainCfg,",
        ")",
        f"{var_name} = TerrainGeneratorCfg(",
        f"    size=({sx!r}, {sy!r}),",
        "    border_width=0.0,",
        "    num_rows=1,",
        "    num_cols=1,",
        f"    horizontal_scale={hscale!r},",
        f"    vertical_scale={vscale!r},",
        "    slope_threshold=None,",
        "    use_cache=False,",
        "    sub_terrains={",
        '        "custom": CustomHeightFieldTerrainCfg(',
        "            proportion=1.0,",
        f"            size=({sx!r}, {sy!r}),",
        f"            horizontal_scale={hscale!r},",
        f"            vertical_scale={vscale!r},",
        "            slope_threshold=None,",
        f"            heightfield_path={path_literal},",
        "        ),",
        "    },",
        ")",
        "",
        "",
    ]


__all__ = [
    "IsaacLabTerrainLoweringError",
    "DEFAULT_VERTICAL_SCALE",
    "DEFAULT_GENERATOR_VAR",
    "build_int16_heights",
    "heightfield_to_isaaclab",
    "emit_custom_terrain_generator_cfg",
]
