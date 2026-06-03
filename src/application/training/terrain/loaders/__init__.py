# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Terrain loaders — ``format_id → TerrainContract``.

Three import formats ship out of the box:

- ``heightfield_npy`` — raw ``.npy`` 2-D array of absolute elevations
  (metres); tile ``size_x`` / ``size_y`` supplied to the constructor.
- ``heightfield_npz`` — self-contained ``.npz`` bundling heights + size
  metadata; the canonical portable form (CLAUDE.md §9), what a bundle
  carries. Dependency-free.
- ``heightfield_png`` — grayscale image height-map scaled to metres;
  needs ``size_x`` / ``size_y`` / ``elevation_z``. Requires Pillow
  (optional dep, §8(a)).

Registration is explicit (mirrors the motion loaders): built-in loaders
register at package-import time via the :func:`register_loader` calls
below — never via class side-effects. ``grep "register_loader("
src/application/training/terrain/loaders/__init__.py`` lists the set.
"""
from __future__ import annotations

from application.training.terrain.loaders.base import (
    LOADER_REGISTRY,
    TerrainLoader,
    TerrainLoaderError,
    get_loader,
    list_loader_formats,
    register_loader,
)
from application.training.terrain.loaders.npy import NpyHeightFieldLoader
from application.training.terrain.loaders.npz import NpzHeightFieldLoader
from application.training.terrain.loaders.png import PngHeightFieldLoader


# Explicit factory registration — see the module docstring for why this is
# preferred over class-side-effect (decorator / __init_subclass__).
register_loader(NpyHeightFieldLoader)
register_loader(NpzHeightFieldLoader)
register_loader(PngHeightFieldLoader)


__all__ = [
    "TerrainLoader",
    "TerrainLoaderError",
    "LOADER_REGISTRY",
    "register_loader",
    "list_loader_formats",
    "get_loader",
    "NpyHeightFieldLoader",
    "NpzHeightFieldLoader",
    "PngHeightFieldLoader",
]
