# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Heightfield terrain contract — the cross-engine single source of truth.

A :class:`HeightField` carries the numeric payload (a 2-D array of
absolute elevations in metres + the tile's physical extent). A
:class:`TerrainContract` wraps it with:

  * ``schema_version`` — pinned to :data:`TERRAIN_SCHEMA_VERSION`.
  * ``source_info`` — provenance (resolved path, import format,
    sha256 of the height bytes) for diagnostics and reproducibility.

Why robot-agnostic (unlike the motion contract)
-----------------------------------------------
A heightfield is geometry, not robot data: the same array is valid for
any robot. So — deliberately — there is **no** ``target_sku`` /
``target_family`` binding here (the motion contract needs them because
joint roles must align with the robot's IR slice; terrain has no joints).

Single canonical array convention
---------------------------------
``heights[i, j]`` is the absolute elevation (metres) at the grid node

    x = i / (n_rows - 1) * size_x
    y = j / (n_cols - 1) * size_y

measured from the tile's min corner, with row index ``i`` running along
+X and column index ``j`` along +Y. This is recorded explicitly in
``array_ordering`` (the only value v1 emits is
``"row_major_xy"``) so the per-engine lowering (Step 2) can remap to each
engine's own convention WITHOUT guessing — MuJoCo ``<hfield>`` and
IsaacLab ``@height_field_to_mesh`` disagree on row/col→x/y direction, and
that remap is resolved against this declared ordering, never inferred.

Heights are stored as **absolute metres**, not normalised 0..1: this
keeps both engines' derivations symmetric (MuJoCo re-normalises against
its own ``elevation_z``; IsaacLab divides by its ``vertical_scale``) and
avoids a normalisation-basis drift between the two (施工规划 v2 §3).

No silent fallbacks (CLAUDE.md §8)
----------------------------------
:func:`validate_terrain_contract` raises on the first structural problem
— NaN/Inf elevations, non-2-D arrays, non-positive sizes, a provenance
sha256 that does not match the recomputed height bytes. A terrain that
cannot be trusted geometrically must fail loud here, never reach an
engine as a silently-zeroed or truncated surface.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TerrainContractError(ValueError):
    """Raised when a :class:`TerrainContract` / :class:`HeightField` fails
    schema validation (bad shape, NaN/Inf, non-positive size, version
    mismatch, provenance digest mismatch, …)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


TERRAIN_SCHEMA_VERSION = "1.0.0"

#: Array conventions the schema knows about. v1 emits only
#: ``"row_major_xy"`` (see module docstring). Kept as a set so a future
#: importer that genuinely needs a transposed/flipped source can declare
#: its ordering explicitly instead of silently mismatching the engines.
ALLOWED_ARRAY_ORDERINGS = frozenset({"row_major_xy"})


# ---------------------------------------------------------------------------
# Provenance digest
# ---------------------------------------------------------------------------


def heightfield_sha256(heights: np.ndarray) -> str:
    """SHA-256 of the height array in a canonical byte layout.

    The array is coerced to C-contiguous ``float32`` before hashing so
    the digest is stable across the dtype/layout a loader happened to
    produce. Loaders stamp this onto :class:`TerrainSourceInfo`;
    :func:`validate_terrain_contract` recomputes and compares it so a
    payload that was mutated after loading (or a hand-edited provenance
    field) fails loud rather than shipping a wrong-geometry terrain.
    """
    arr = np.ascontiguousarray(heights, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Heightfield payload
# ---------------------------------------------------------------------------


@dataclass
class HeightField:
    """A terrain tile as a grid of absolute elevations.

    Attributes
    ----------
    heights
        2-D ``float32`` array, shape ``(n_rows, n_cols)``, absolute
        elevation in metres. See module docstring for the index→world
        convention.
    size_x, size_y
        Physical extent of the tile in metres along the row (X) and
        column (Y) axes respectively. Both strictly positive.
    array_ordering
        One of :data:`ALLOWED_ARRAY_ORDERINGS`; records how ``heights``
        indices map to world axes (v1: ``"row_major_xy"``).

    Stored as a plain (non-frozen) dataclass to mirror
    :class:`application.training.motion.clip.MotionClip` — numpy arrays
    do not play well with the generated ``__eq__`` of a frozen
    dataclass, and like ``MotionClip`` this payload is never compared by
    value.
    """

    heights: np.ndarray
    size_x: float
    size_y: float
    array_ordering: str = "row_major_xy"

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return int(self.heights.shape[0]) if self.heights.ndim == 2 else 0

    @property
    def n_cols(self) -> int:
        return int(self.heights.shape[1]) if self.heights.ndim == 2 else 0

    @property
    def cell_size_x(self) -> float:
        """Grid spacing along X (metres). ``size_x / (n_rows - 1)``."""
        n = self.n_rows
        if n < 2:
            raise TerrainContractError(
                f"HeightField.cell_size_x undefined for n_rows={n} (<2)"
            )
        return float(self.size_x) / float(n - 1)

    @property
    def cell_size_y(self) -> float:
        """Grid spacing along Y (metres). ``size_y / (n_cols - 1)``."""
        n = self.n_cols
        if n < 2:
            raise TerrainContractError(
                f"HeightField.cell_size_y undefined for n_cols={n} (<2)"
            )
        return float(self.size_y) / float(n - 1)

    @property
    def min_elevation(self) -> float:
        return float(np.min(self.heights))

    @property
    def max_elevation(self) -> float:
        return float(np.max(self.heights))

    @property
    def elevation_span(self) -> float:
        """``max_elevation - min_elevation`` (metres)."""
        return self.max_elevation - self.min_elevation


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainSourceInfo:
    """Where the heightfield came from.

    Attributes
    ----------
    source_path
        Absolute path the loader resolved before reading. Never empty.
    source_format
        The loader ``format_id`` that produced this contract
        (e.g. ``"heightfield_png"``).
    sha256
        Digest of the height bytes via :func:`heightfield_sha256`,
        recomputed and verified by :func:`validate_terrain_contract`.
    """

    source_path: str
    source_format: str
    sha256: str


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainContract:
    """Self-describing wrapper around a :class:`HeightField`.

    Frozen dataclass to match the immutable-contract idiom used by
    :class:`application.training.motion.contract.ReferenceMotionContract`
    and the registry data contracts. Construction-time guarantees live in
    ``__post_init__`` (the cheap non-empty checks the type system cannot
    express); the full structural/semantic pass is
    :func:`validate_terrain_contract`.
    """

    schema_version: str
    height_field: HeightField
    source_info: TerrainSourceInfo

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise TerrainContractError(
                "TerrainContract.schema_version is required and must be a "
                "non-empty string (pinned to TERRAIN_SCHEMA_VERSION="
                f"{TERRAIN_SCHEMA_VERSION!r})."
            )
        if not isinstance(self.height_field, HeightField):
            raise TerrainContractError(
                "TerrainContract.height_field must be a HeightField, got "
                f"{type(self.height_field).__name__!r}."
            )
        if not isinstance(self.source_info, TerrainSourceInfo):
            raise TerrainContractError(
                "TerrainContract.source_info must be a TerrainSourceInfo, got "
                f"{type(self.source_info).__name__!r}."
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_terrain_contract(contract: TerrainContract) -> None:
    """Structural and semantic checks. Raises on the first problem.

    Layers (mirrors the motion contract validator):
      L1  type checks                              - dataclass shape
      L2  schema_version pinned                    - "1.0.0" only
      L3  heights is a 2-D float array, >= 2x2     - usable grid
      L4  heights finite (no NaN/Inf)              - §8 fail-loud
      L5  size_x / size_y strictly positive        - real extent
      L6  array_ordering ∈ ALLOWED                 - reject unknown
      L7  source provenance present + digest match - no anonymous /
          tampered loads
    """
    if not isinstance(contract, TerrainContract):
        raise TerrainContractError(
            f"validate_terrain_contract: expected TerrainContract, got "
            f"{type(contract).__name__!r}"
        )

    # L2 — schema version pin
    if contract.schema_version != TERRAIN_SCHEMA_VERSION:
        raise TerrainContractError(
            f"TerrainContract.schema_version={contract.schema_version!r} does "
            f"not match the supported version {TERRAIN_SCHEMA_VERSION!r}. "
            f"Re-import the terrain with the current loader."
        )

    hf = contract.height_field
    if not isinstance(hf, HeightField):
        raise TerrainContractError(
            f"TerrainContract.height_field must be HeightField, got "
            f"{type(hf).__name__!r}"
        )

    # L3 — array shape/dtype
    heights = hf.heights
    if not isinstance(heights, np.ndarray):
        raise TerrainContractError(
            f"HeightField.heights must be a numpy ndarray, got "
            f"{type(heights).__name__!r}"
        )
    if heights.ndim != 2:
        raise TerrainContractError(
            f"HeightField.heights must be 2-D (n_rows, n_cols), got shape "
            f"{heights.shape!r}"
        )
    if not np.issubdtype(heights.dtype, np.floating):
        raise TerrainContractError(
            f"HeightField.heights must be a floating dtype (absolute metres), "
            f"got {heights.dtype!r}"
        )
    if heights.shape[0] < 2 or heights.shape[1] < 2:
        raise TerrainContractError(
            f"HeightField.heights must be at least 2x2 to define a surface, "
            f"got shape {heights.shape!r}"
        )

    # L4 — finite values (§8: never feed a NaN/Inf surface to an engine)
    if not np.all(np.isfinite(heights)):
        n_bad = int(np.count_nonzero(~np.isfinite(heights)))
        raise TerrainContractError(
            f"HeightField.heights contains {n_bad} non-finite value(s) "
            f"(NaN/Inf). A terrain surface must be fully finite — fix the "
            f"source rather than letting a hole reach the simulator."
        )

    # L5 — physical extent
    if not (float(hf.size_x) > 0.0):
        raise TerrainContractError(
            f"HeightField.size_x must be > 0 metres, got {hf.size_x!r}"
        )
    if not (float(hf.size_y) > 0.0):
        raise TerrainContractError(
            f"HeightField.size_y must be > 0 metres, got {hf.size_y!r}"
        )

    # L6 — array ordering
    if hf.array_ordering not in ALLOWED_ARRAY_ORDERINGS:
        raise TerrainContractError(
            f"HeightField.array_ordering={hf.array_ordering!r} is not known. "
            f"Allowed: {sorted(ALLOWED_ARRAY_ORDERINGS)!r}."
        )

    # L7 — provenance
    si = contract.source_info
    if not isinstance(si, TerrainSourceInfo):
        raise TerrainContractError(
            f"TerrainContract.source_info must be TerrainSourceInfo, got "
            f"{type(si).__name__!r}"
        )
    if not si.source_path:
        raise TerrainContractError(
            "TerrainContract.source_info.source_path is required and may not "
            "be an empty string."
        )
    if not si.source_format:
        raise TerrainContractError(
            "TerrainContract.source_info.source_format is required and may "
            "not be an empty string."
        )
    if not si.sha256:
        raise TerrainContractError(
            "TerrainContract.source_info.sha256 is required and may not be "
            "an empty string."
        )
    recomputed = heightfield_sha256(heights)
    if si.sha256 != recomputed:
        raise TerrainContractError(
            f"TerrainContract.source_info.sha256 mismatch: stored "
            f"{si.sha256!r} != recomputed {recomputed!r}. The height payload "
            f"was modified after loading, or the provenance digest was "
            f"hand-edited — refuse to trust the geometry."
        )


__all__ = [
    "TERRAIN_SCHEMA_VERSION",
    "ALLOWED_ARRAY_ORDERINGS",
    "HeightField",
    "TerrainContract",
    "TerrainContractError",
    "TerrainSourceInfo",
    "heightfield_sha256",
    "validate_terrain_contract",
]
