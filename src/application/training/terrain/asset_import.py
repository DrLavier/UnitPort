# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Import an external height-map → canonical heightfield ``.npz`` (Step 5e).

A user imports a PNG / ``.npy`` / ``.npz`` height-map; the canvas and the
whole pipeline consume only the canonical self-contained ``.npz`` (size
metadata embedded). This module is the format-conversion engine:

    convert_to_canonical_npz_bytes(src, size_x=..., size_y=..., ...) -> bytes

dispatching to the right Step-1 loader (PNG / npy need geometry the file
cannot carry; npz is self-contained) and re-serialising the validated
heightfield. Pure (no USER_CONFIG_DIR / I/O beyond reading ``src``), so it
is fully unit-testable; the UI dialog + scene-registry overlay write the
returned bytes under ``USER_CONFIG_DIR`` (see scene_registry).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from application.training.terrain.bundle_io import serialize_heightfield_npz_bytes
from application.training.terrain.loaders import (
    NpyHeightFieldLoader,
    NpzHeightFieldLoader,
    PngHeightFieldLoader,
    TerrainLoaderError,
)


#: file extension → terrain loader format_id
_EXT_FORMAT = {
    ".npz": "heightfield_npz",
    ".npy": "heightfield_npy",
    ".png": "heightfield_png",
}


def infer_format(source_path: Union[Path, str]) -> str:
    """Map a file extension to a terrain ``format_id``. Raises on unknown."""
    ext = Path(source_path).suffix.lower()
    fmt = _EXT_FORMAT.get(ext)
    if fmt is None:
        raise TerrainLoaderError(
            f"infer_format: unsupported terrain file extension {ext!r} "
            f"({source_path}). Supported: {sorted(_EXT_FORMAT)}."
        )
    return fmt


def convert_to_canonical_npz_bytes(
    source_path: Union[Path, str],
    *,
    source_format: Optional[str] = None,
    size_x: Optional[float] = None,
    size_y: Optional[float] = None,
    elevation_z: Optional[float] = None,
    base_z: float = 0.0,
    array_ordering: str = "row_major_xy",
) -> Tuple[bytes, Dict[str, Any]]:
    """Read an external height-map and return canonical ``.npz`` bytes + meta.

    ``size_x`` / ``size_y`` are REQUIRED for ``.npy`` / ``.png`` (the file
    carries no physical scale); ``elevation_z`` is additionally required for
    ``.png`` (white-pixel height in metres). ``.npz`` is self-contained and
    ignores them. Raises :class:`TerrainLoaderError` on a missing required
    arg, and the contract's errors on a malformed surface (§8).
    """
    fmt = source_format or infer_format(source_path)

    if fmt == "heightfield_npz":
        loader: Any = NpzHeightFieldLoader()
    elif fmt == "heightfield_npy":
        if size_x is None or size_y is None:
            raise TerrainLoaderError(
                "convert_to_canonical_npz_bytes: .npy import requires size_x "
                "and size_y (the raw array carries no physical scale)."
            )
        loader = NpyHeightFieldLoader(
            size_x=size_x, size_y=size_y, array_ordering=array_ordering
        )
    elif fmt == "heightfield_png":
        if size_x is None or size_y is None or elevation_z is None:
            raise TerrainLoaderError(
                "convert_to_canonical_npz_bytes: .png import requires size_x, "
                "size_y and elevation_z (white-pixel height in metres)."
            )
        loader = PngHeightFieldLoader(
            size_x=size_x, size_y=size_y, elevation_z=elevation_z,
            base_z=base_z, array_ordering=array_ordering,
        )
    else:
        raise TerrainLoaderError(
            f"convert_to_canonical_npz_bytes: unknown source_format {fmt!r}."
        )

    contract = loader.load(source_path)  # validates the surface (§8)
    npz_bytes = serialize_heightfield_npz_bytes(contract)
    hf = contract.height_field
    meta = {
        "sha256": contract.source_info.sha256,
        "size_x": float(hf.size_x),
        "size_y": float(hf.size_y),
        "n_rows": int(hf.n_rows),
        "n_cols": int(hf.n_cols),
        "array_ordering": hf.array_ordering,
        "min_elevation": hf.min_elevation,
        "max_elevation": hf.max_elevation,
    }
    return npz_bytes, meta


__all__ = [
    "infer_format",
    "convert_to_canonical_npz_bytes",
]
