# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Custom terrain — heightfield single-source-of-truth + loaders.

A user-imported terrain is represented as a :class:`HeightField` (a 2-D
float array of absolute elevations in metres + the physical tile size)
wrapped in a :class:`TerrainContract` (schema version + provenance). The
heightfield is the **single cross-engine source**: MuJoCo (SB3) derives
a ``<hfield>`` from it and IsaacLab derives a ``@height_field_to_mesh``
sub-terrain from the SAME array, so both engines run byte-identical
geometry (sim2sim闭环一致性).

This subpackage is the *harvest + schema* layer only (施工规划 v2 Step 1):
it reads a user file into the canonical in-memory form and validates it.
Lowering into each engine (Step 2), the cross-engine consistency gate
(Step 3) and bundle self-containment (Step 4) live elsewhere and consume
:class:`TerrainContract`.

Loaders are registry-dispatched exactly like the motion loaders
(``application.training.motion.loaders``): one ``format_id`` per import
format, registered explicitly at package-import time.
"""
from __future__ import annotations

from application.training.terrain.contract import (
    ALLOWED_ARRAY_ORDERINGS,
    TERRAIN_SCHEMA_VERSION,
    HeightField,
    TerrainContract,
    TerrainContractError,
    TerrainSourceInfo,
    heightfield_sha256,
    validate_terrain_contract,
)
from application.training.terrain.loaders import (
    LOADER_REGISTRY,
    NpyHeightFieldLoader,
    NpzHeightFieldLoader,
    PngHeightFieldLoader,
    TerrainLoader,
    TerrainLoaderError,
    get_loader,
    list_loader_formats,
    register_loader,
)

__all__ = [
    # contract
    "TERRAIN_SCHEMA_VERSION",
    "ALLOWED_ARRAY_ORDERINGS",
    "HeightField",
    "TerrainContract",
    "TerrainContractError",
    "TerrainSourceInfo",
    "heightfield_sha256",
    "validate_terrain_contract",
    # loaders
    "LOADER_REGISTRY",
    "TerrainLoader",
    "TerrainLoaderError",
    "register_loader",
    "list_loader_formats",
    "get_loader",
    "NpyHeightFieldLoader",
    "NpzHeightFieldLoader",
    "PngHeightFieldLoader",
]
