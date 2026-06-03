# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``heightfield_npz`` — self-contained numpy heightfield asset.

Reads a ``.npz`` archive that bundles the height array **and** its
geometry metadata, so it needs no constructor geometry args. This is the
canonical portable form (CLAUDE.md §9 self-contained) — the shape a
bundle carries (Step 4) and what the importer re-saves a PNG/raw import
into.

Required archive keys:
  * ``heights``        — 2-D float array, absolute elevation in metres.
  * ``size_x``         — scalar, tile extent X (metres).
  * ``size_y``         — scalar, tile extent Y (metres).

Optional:
  * ``array_ordering`` — string; defaults to the v1 canonical
    ``"row_major_xy"`` when absent.
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


class NpzHeightFieldLoader(TerrainLoader):
    """Load a self-contained ``.npz`` heightfield asset (heights + size)."""

    format_id = "heightfield_npz"

    def load(self, path: Union[Path, str]) -> TerrainContract:
        p = Path(path)
        if not p.is_file():
            raise TerrainLoaderError(f"{type(self).__name__}: file not found: {p}")
        try:
            npz = np.load(str(p))
        except Exception as exc:  # noqa: BLE001
            raise TerrainLoaderError(
                f"{type(self).__name__}: np.load failed for {p}: {exc}"
            ) from exc

        try:
            keys = set(npz.files)
        except Exception as exc:  # noqa: BLE001
            raise TerrainLoaderError(
                f"{type(self).__name__}: {p} is not a .npz archive: {exc}"
            ) from exc

        missing = {"heights", "size_x", "size_y"} - keys
        if missing:
            raise TerrainLoaderError(
                f"{type(self).__name__}: {p} missing required key(s) "
                f"{sorted(missing)!r}; present: {sorted(keys)!r}"
            )

        heights = np.asarray(npz["heights"], dtype=np.float32)
        if heights.ndim != 2:
            raise TerrainLoaderError(
                f"{type(self).__name__}: 'heights' must be 2-D (n_rows, "
                f"n_cols), got shape {heights.shape} in {p}"
            )
        try:
            size_x = float(np.asarray(npz["size_x"]).reshape(()))
            size_y = float(np.asarray(npz["size_y"]).reshape(()))
        except Exception as exc:  # noqa: BLE001
            raise TerrainLoaderError(
                f"{type(self).__name__}: 'size_x'/'size_y' must be scalars "
                f"in {p}: {exc}"
            ) from exc

        array_ordering = "row_major_xy"
        if "array_ordering" in keys:
            array_ordering = str(np.asarray(npz["array_ordering"]).reshape(()).item())

        hf = HeightField(
            heights=heights,
            size_x=size_x,
            size_y=size_y,
            array_ordering=array_ordering,
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


__all__ = ["NpzHeightFieldLoader"]
