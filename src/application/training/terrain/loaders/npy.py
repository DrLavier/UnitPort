# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``heightfield_npy`` — raw numpy height array loader.

Reads a ``.npy`` file holding a 2-D array of **absolute elevations in
metres** (shape ``(n_rows, n_cols)``). A bare ``.npy`` carries no
geometry metadata, so the tile's physical ``size_x`` / ``size_y`` are
supplied to the constructor (mirroring how ``NpyLoader`` for motion takes
``fps``) — never reverse-inferred.

For a self-contained array asset that bundles the geometry alongside the
heights, use :class:`NpzHeightFieldLoader` (``heightfield_npz``) instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from application.training.terrain.contract import (
    TERRAIN_SCHEMA_VERSION,
    HeightField,
    TerrainContract,
    TerrainSourceInfo,
    heightfield_sha256,
    validate_terrain_contract,
)
from application.training.terrain.loaders.base import (
    TerrainLoader,
    TerrainLoaderError,
)


class NpyHeightFieldLoader(TerrainLoader):
    """Load a ``.npy`` 2-D array of absolute elevations (metres).

    Parameters
    ----------
    size_x, size_y
        Physical tile extent in metres along the row (X) / column (Y)
        axes. Both required and strictly positive.
    array_ordering
        How the array indices map to world axes; defaults to the v1
        canonical ``"row_major_xy"`` (see contract module docstring).
    """

    format_id = "heightfield_npy"

    def __init__(
        self,
        size_x: float,
        size_y: float,
        array_ordering: str = "row_major_xy",
    ) -> None:
        if not (float(size_x) > 0.0) or not (float(size_y) > 0.0):
            raise TerrainLoaderError(
                f"{type(self).__name__}: size_x/size_y must be > 0 metres, "
                f"got size_x={size_x!r}, size_y={size_y!r}"
            )
        self._size_x = float(size_x)
        self._size_y = float(size_y)
        self._array_ordering = str(array_ordering)

    def load(self, path: Union[Path, str]) -> TerrainContract:
        p = Path(path)
        if not p.is_file():
            raise TerrainLoaderError(f"{type(self).__name__}: file not found: {p}")
        try:
            arr = np.load(str(p))
        except Exception as exc:  # noqa: BLE001
            raise TerrainLoaderError(
                f"{type(self).__name__}: np.load failed for {p}: {exc}"
            ) from exc

        heights = np.asarray(arr, dtype=np.float32)
        if heights.ndim != 2:
            raise TerrainLoaderError(
                f"{type(self).__name__}: expected 2-D array (n_rows, n_cols), "
                f"got shape {heights.shape} in {p}"
            )

        hf = HeightField(
            heights=heights,
            size_x=self._size_x,
            size_y=self._size_y,
            array_ordering=self._array_ordering,
        )
        contract = TerrainContract(
            schema_version=TERRAIN_SCHEMA_VERSION,
            height_field=hf,
            source_info=TerrainSourceInfo(
                source_path=str(p.resolve()),
                source_format=self.format_id,
                sha256=heightfield_sha256(heights),
            ),
        )
        validate_terrain_contract(contract)
        return contract


__all__ = ["NpyHeightFieldLoader"]
