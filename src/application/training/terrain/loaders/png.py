# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``heightfield_png`` — grayscale image height-map loader.

Reads a single-channel (or to-be-grayscaled) PNG and maps pixel
intensity to absolute elevation in metres:

    elevation = base_z + (pixel / pixel_max) * elevation_z

where ``pixel_max`` is the dtype's full-scale (255 for 8-bit, 65535 for
16-bit), so a white pixel is ``base_z + elevation_z`` and black is
``base_z``. ``size_x`` / ``size_y`` / ``elevation_z`` (and optional
``base_z``) are constructor args because a PNG carries no physical scale.

Pillow is an OPTIONAL dependency: PNG import is a convenience over the
dependency-free ``.npy`` / ``.npz`` canonical formats. The import is
gated so a venv without Pillow simply cannot offer this format (fails
loud with an actionable message) rather than crashing at module import.
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

# WHY KEPT (§8(a)): Pillow is an optional third-party dep — PNG import is a
# convenience over the dependency-free .npy/.npz formats. Branch on absence
# so the loader fails loud at use-time with guidance instead of breaking the
# whole terrain subpackage import on a Pillow-less venv.
try:
    from PIL import Image as _PILImage  # type: ignore
except ImportError:  # pragma: no cover - exercised only on a Pillow-less venv
    _PILImage = None


#: Full-scale value per image dtype, used to normalise pixels to [0, 1].
_DTYPE_FULL_SCALE = {
    np.dtype(np.uint8): 255.0,
    np.dtype(np.uint16): 65535.0,
    np.dtype(np.int32): 65535.0,  # PIL 'I' mode (32-bit) commonly holds 16-bit data
}


class PngHeightFieldLoader(TerrainLoader):
    """Load a grayscale PNG height-map, scaled to absolute metres.

    Parameters
    ----------
    size_x, size_y
        Physical tile extent in metres (row / column axes). Positive.
    elevation_z
        Metres a full-scale (white) pixel maps to above ``base_z``.
        Strictly positive.
    base_z
        Metres a zero (black) pixel maps to. Default 0.0.
    array_ordering
        Index→world convention; defaults to v1 canonical ``"row_major_xy"``.
    """

    format_id = "heightfield_png"

    def __init__(
        self,
        size_x: float,
        size_y: float,
        elevation_z: float,
        base_z: float = 0.0,
        array_ordering: str = "row_major_xy",
    ) -> None:
        if not (float(size_x) > 0.0) or not (float(size_y) > 0.0):
            raise TerrainLoaderError(
                f"{type(self).__name__}: size_x/size_y must be > 0 metres, "
                f"got size_x={size_x!r}, size_y={size_y!r}"
            )
        if not (float(elevation_z) > 0.0):
            raise TerrainLoaderError(
                f"{type(self).__name__}: elevation_z must be > 0 metres "
                f"(white-pixel height), got {elevation_z!r}"
            )
        self._size_x = float(size_x)
        self._size_y = float(size_y)
        self._elevation_z = float(elevation_z)
        self._base_z = float(base_z)
        self._array_ordering = str(array_ordering)

    def load(self, path: Union[Path, str]) -> TerrainContract:
        if _PILImage is None:
            raise TerrainLoaderError(
                f"{type(self).__name__}: Pillow (PIL) is required to import "
                f"PNG height-maps but is not installed. Install Pillow, or "
                f"import the terrain as .npy / .npz instead."
            )
        p = Path(path)
        if not p.is_file():
            raise TerrainLoaderError(f"{type(self).__name__}: file not found: {p}")
        try:
            with _PILImage.open(str(p)) as img:
                # Keep the native single-channel depth (L=8-bit, I=32-bit,
                # I;16=16-bit). RGB/RGBA collapse to luminance via 'L'.
                if img.mode not in ("L", "I", "I;16"):
                    img = img.convert("L")
                pixels = np.asarray(img)
        except Exception as exc:  # noqa: BLE001
            raise TerrainLoaderError(
                f"{type(self).__name__}: failed to read PNG {p}: {exc}"
            ) from exc

        if pixels.ndim != 2:
            raise TerrainLoaderError(
                f"{type(self).__name__}: expected a single-channel grayscale "
                f"image, got array shape {pixels.shape} in {p}"
            )

        full_scale = _DTYPE_FULL_SCALE.get(pixels.dtype)
        if full_scale is None:
            raise TerrainLoaderError(
                f"{type(self).__name__}: unsupported PNG pixel dtype "
                f"{pixels.dtype!r} in {p}. Supported: uint8 / uint16."
            )

        norm = np.asarray(pixels, dtype=np.float32) / np.float32(full_scale)
        heights = (self._base_z + norm * self._elevation_z).astype(np.float32)

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


__all__ = ["PngHeightFieldLoader"]
