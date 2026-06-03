# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lower a :class:`HeightField` into a MuJoCo ``<hfield>`` (SB3 side).

Derives a MuJoCo heightfield asset + geom from the cross-engine source
heightfield and injects it into an existing MJCF via ``mujoco.MjSpec``
(the ``MjSpec.from_file → inject → compile`` pattern proven reusable from
``service/runtime/simulation/mujoco/mj_field.py``). Unlike that
deployment-side composer, this path is **fail-loud** (PV 修正 1 /
CLAUDE.md §8): a user who asked to train on a custom terrain must never
be silently dropped onto flat ground — any failure raises.

Array convention (PV④, empirically pinned 2026-06-02)
-----------------------------------------------------
The canonical :class:`HeightField` uses ``heights[i, j]`` with row ``i``
along +X and col ``j`` along +Y (same as IsaacLab's
``convert_height_field_to_mesh``). MuJoCo's ``<hfield>`` is the
**transpose**: its ``userdata`` is row-major ``data[r, c]`` with row
``r`` → world +Y and col ``c`` → world +X (verified by compiling an
asymmetric ramp and ray-casting). So the canonical array is transposed
before flattening, and ``nrow``/``ncol`` swap accordingly:

    M = norm.T            # (n_cols, n_rows)
    nrow = n_cols         # MuJoCo rows span +Y
    ncol = n_rows         # MuJoCo cols span +X
    size = (size_x/2, size_y/2, elevation_z, base_thickness)

MuJoCo stores normalised data in ``[0, 1]`` and renders surface height as
``geom_pos_z + data * elevation_z``. Absolute metres are recovered by
normalising against the field's own span and placing the geom at the
field minimum: ``data = (h - hmin) / span``, ``elevation_z = span``,
``geom_pos_z = hmin`` → surface ``= hmin + (h - hmin) = h``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from application.training.terrain.contract import HeightField


class TerrainLoweringError(RuntimeError):
    """Raised when a heightfield cannot be lowered/injected into MuJoCo.

    Deliberately a hard error (not a warn-and-flat fallback): training on
    the wrong (flat) ground when the user asked for custom terrain is the
    exact "illusion of success" CLAUDE.md §8 forbids.
    """


#: Geom/asset name the injected terrain uses. Stable so callers (height
#: sampling in the consistency gate, review) can find it by name.
DEFAULT_TERRAIN_NAME = "unitport_custom_terrain"

#: Floor span (m) below the lowest surface point. Must be > 0 (MuJoCo
#: requires a positive base box). Purely structural — does not affect the
#: top surface geometry.
_BASE_THICKNESS = 1.0

#: elevation_z floor for a (near-)flat field — MuJoCo rejects a
#: zero-elevation hfield. A flat custom terrain is degenerate but legal.
_MIN_ELEVATION = 1e-4


def heightfield_to_mujoco_hfield(
    hf: HeightField,
    *,
    name: str = DEFAULT_TERRAIN_NAME,
    center: Tuple[float, float] = (0.0, 0.0),
    base_thickness: float = _BASE_THICKNESS,
) -> Dict[str, Any]:
    """Pure conversion: :class:`HeightField` → MuJoCo ``add_hfield`` kwargs.

    Returns a dict with ``name`` / ``size`` (4-tuple) / ``nrow`` / ``ncol``
    / ``userdata`` (flattened, normalised [0,1], transposed for MuJoCo) and
    the recommended geom ``pos`` ``(cx, cy, hmin)`` that places the surface
    at absolute metres with the tile centred on ``center``.

    No I/O, no engine handle — unit-testable on its own.
    """
    if hf.array_ordering != "row_major_xy":
        raise TerrainLoweringError(
            f"heightfield_to_mujoco_hfield: unsupported array_ordering "
            f"{hf.array_ordering!r}; only 'row_major_xy' is implemented."
        )
    heights = np.ascontiguousarray(hf.heights, dtype=np.float32)
    if heights.ndim != 2 or heights.shape[0] < 2 or heights.shape[1] < 2:
        raise TerrainLoweringError(
            f"heightfield_to_mujoco_hfield: heights must be 2-D >=2x2, got "
            f"shape {heights.shape!r}"
        )
    if not np.all(np.isfinite(heights)):
        raise TerrainLoweringError(
            "heightfield_to_mujoco_hfield: heights contain non-finite values "
            "(NaN/Inf) — refuse to build a holed terrain (§8)."
        )

    n_rows, n_cols = heights.shape
    hmin = float(heights.min())
    span = float(heights.max()) - hmin
    elevation_z = span if span > _MIN_ELEVATION else _MIN_ELEVATION
    norm = np.clip((heights - hmin) / elevation_z, 0.0, 1.0)

    # Transpose so MuJoCo row→+Y, col→+X (PV④). nrow/ncol swap.
    m = np.ascontiguousarray(norm.T, dtype=np.float64)  # (n_cols, n_rows)

    return {
        "name": name,
        "size": [
            float(hf.size_x) / 2.0,
            float(hf.size_y) / 2.0,
            float(elevation_z),
            float(base_thickness),
        ],
        "nrow": int(n_cols),
        "ncol": int(n_rows),
        "userdata": m.flatten().tolist(),
        "pos": [float(center[0]), float(center[1]), hmin],
    }


def inject_heightfield(
    mujoco_mod: Any,
    spec: Any,
    hf: HeightField,
    *,
    name: str = DEFAULT_TERRAIN_NAME,
    center: Tuple[float, float] = (0.0, 0.0),
    replace_planes: bool = True,
    rgba: Optional[List[float]] = None,
) -> str:
    """Inject the heightfield into an open ``MjSpec``. Returns the geom name.

    ``replace_planes`` removes existing worldbody plane geoms first (the
    menagerie ``scene.xml`` ships a flat ``floor`` plane that would
    otherwise coexist with the terrain). Raises :class:`TerrainLoweringError`
    on any MuJoCo-side failure — never silently leaves the spec on flat
    ground.
    """
    params = heightfield_to_mujoco_hfield(hf, name=name, center=center)

    worldbody = getattr(spec, "worldbody", None)
    if worldbody is None:
        raise TerrainLoweringError("inject_heightfield: spec has no worldbody")

    if replace_planes:
        try:
            plane_t = mujoco_mod.mjtGeom.mjGEOM_PLANE
            for g in list(getattr(spec, "geoms", []) or []):
                if getattr(g, "type", None) == plane_t:
                    spec.delete(g)
        except Exception as exc:  # noqa: BLE001
            raise TerrainLoweringError(
                f"inject_heightfield: failed to remove existing plane geoms: "
                f"{exc}"
            ) from exc

    try:
        spec.add_hfield(
            name=params["name"],
            size=params["size"],
            nrow=params["nrow"],
            ncol=params["ncol"],
            userdata=params["userdata"],
        )
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoweringError(
            f"inject_heightfield: MjSpec.add_hfield failed: {exc}"
        ) from exc

    try:
        geom = worldbody.add_geom()
        geom.name = f"{params['name']}_geom"
        geom.type = mujoco_mod.mjtGeom.mjGEOM_HFIELD
        geom.hfieldname = params["name"]
        geom.pos = params["pos"]
        if rgba is not None:
            geom.rgba = [float(c) for c in rgba]
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoweringError(
            f"inject_heightfield: add_geom (hfield) failed: {exc}"
        ) from exc

    return f"{params['name']}_geom"


def compose_model_with_terrain(
    mjcf_path: Union[Path, str],
    hf: HeightField,
    *,
    name: str = DEFAULT_TERRAIN_NAME,
    center: Tuple[float, float] = (0.0, 0.0),
    replace_planes: bool = True,
):
    """Load an MJCF, inject ``hf``, and return the compiled ``MjModel``.

    The end-to-end seam the SB3 env (Step 5) calls in place of
    ``MjModel.from_xml_path``. Fail-loud throughout (§8): a missing
    mujoco, unreadable MJCF, or compile failure raises rather than
    degrading to flat ground.
    """
    try:
        import mujoco  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - mujoco is a hard dep
        raise TerrainLoweringError(
            "compose_model_with_terrain: mujoco is required to build a "
            "custom-terrain model but is not importable."
        ) from exc

    p = Path(mjcf_path)
    if not p.is_file():
        raise TerrainLoweringError(
            f"compose_model_with_terrain: MJCF not found: {p}"
        )
    try:
        spec = mujoco.MjSpec.from_file(str(p))
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoweringError(
            f"compose_model_with_terrain: MjSpec.from_file failed for {p}: "
            f"{exc}"
        ) from exc

    inject_heightfield(
        mujoco, spec, hf, name=name, center=center, replace_planes=replace_planes
    )

    try:
        return spec.compile()
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoweringError(
            f"compose_model_with_terrain: spec.compile() failed: {exc}"
        ) from exc


__all__ = [
    "TerrainLoweringError",
    "DEFAULT_TERRAIN_NAME",
    "heightfield_to_mujoco_hfield",
    "inject_heightfield",
    "compose_model_with_terrain",
]
