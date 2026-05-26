# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""TorqueLookupTable — loader + evaluator for the v1 torque-lookup schema.

A remotized actuator's maximum output torque depends on the joint angle rather
than being a single scalar box limit. This module is the canonical, engine-
agnostic representation of that angle→max-torque curve. The schema it reads is
documented (with full rationale) in ``torque_lookup_v1.yaml`` next to this file.

Single source of truth, two consumers:
  - IsaacLab (training): ``to_isaaclab_array()`` emits the 3-column
    ``[angle, gear_ratio, max_torque]`` form ``RemotizedPDActuatorCfg`` expects.
  - MuJoCo (deploy): ``max_torque_at`` / ``max_torque_batch`` give the per-step
    torque ceiling that ``PDController`` clamps against at the current angle.

Pure Python + numpy + pyyaml — no SDK / app imports, so it loads identically in
the main app and in the IsaacLab compile venv. File I/O follows the
``application.training`` convention (``Path.open`` + ``yaml.safe_load``), not
``DataManager``: this code runs inside the training/compile path, where the SDK
is not guaranteed importable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# --- Schema constants -------------------------------------------------------

SCHEMA_VERSION = 1
_VALID_INTERPOLATION = ("linear", "step", "cubic_spline")  # enum advertised
_IMPLEMENTED_INTERPOLATION = ("linear",)                   # enum implemented in v1
_VALID_EXTRAPOLATION = ("clamp_to_endpoint", "error")
_AXIS_LABEL_DEFAULT = "joint_position_rad"
# Canonical float format used ONLY for the provenance hash, so the digest is
# stable regardless of YAML loader field order or float repr quirks. NOT used
# for to_dict (which must round-trip the exact loaded values byte-tight).
# `%.12g` is chosen for cross-platform hash stability: 12 significant digits
# is wide enough to distinguish any physically-meaningful torque/angle value,
# yet narrow enough to stay inside the region where every conformant libm
# formats identically. Widening past ~15 digits would expose the last
# bit(s) of the IEEE-754 double, where glibc (Linux) and msvcrt (Windows)
# can round the decimal repr differently — yielding different hashes for the
# same table on different OSes. Do not widen this without re-checking that.
_HASH_FLOAT_FMT = "%.12g"


class TorqueLookupSchemaError(ValueError):
    """Raised when a torque-lookup payload violates the v1 schema.

    A dedicated type so callers (compiler, deploy_contract) can distinguish a
    malformed table from other ValueErrors and surface a precise message.
    """


def _raise_version(version: Any) -> "TorqueLookupSchemaError":
    """Build a precise schema-version rejection with upgrade/downgrade steps."""
    if isinstance(version, int) and version > SCHEMA_VERSION:
        return TorqueLookupSchemaError(
            f"torque_lookup: this is a v{version} schema file, but this reader "
            f"only supports v{SCHEMA_VERSION}. Upgrade UnitPort to a release "
            f"that ships the v{version} torque-lookup reader, or downgrade the "
            f"data file to v{SCHEMA_VERSION} (a single symmetric `max_torque` "
            f"column). The reader refuses to reinterpret a future file under "
            f"v{SCHEMA_VERSION} rules because that would silently misread it."
        )
    return TorqueLookupSchemaError(
        f"torque_lookup: unsupported schema_version={version!r} "
        f"(this reader supports v{SCHEMA_VERSION}). Set `schema_version: "
        f"{SCHEMA_VERSION}` and provide a single symmetric `max_torque` column."
    )


class TorqueLookupTable:
    """Angle → maximum-torque lookup (v1 schema, symmetric, linear).

    Construct via :meth:`from_yaml` (data file) or :meth:`from_dict`
    (deserialization from a deploy_contract payload). Both run the full
    validation suite; an invalid table raises :class:`TorqueLookupSchemaError`
    rather than constructing a silently-wrong object (CLAUDE.md §8).
    """

    __slots__ = (
        "_axis", "_max_torque", "_axis_label",
        "_interpolation", "_symmetric", "_extrapolation",
    )

    def __init__(
        self,
        *,
        axis: np.ndarray,
        max_torque: np.ndarray,
        axis_label: str = _AXIS_LABEL_DEFAULT,
        interpolation: str = "linear",
        symmetric: bool = True,
        extrapolation: str = "clamp_to_endpoint",
    ) -> None:
        # Direct construction is allowed but still validated — there is no
        # "trusted" path that bypasses the invariants.
        axis_arr = np.asarray(axis, dtype=np.float64).reshape(-1)
        torque_arr = np.asarray(max_torque, dtype=np.float64).reshape(-1)
        _validate(
            axis=axis_arr,
            max_torque=torque_arr,
            interpolation=interpolation,
            symmetric=symmetric,
            extrapolation=extrapolation,
        )
        self._axis = axis_arr
        self._max_torque = torque_arr
        self._axis_label = str(axis_label)
        self._interpolation = str(interpolation)
        self._symmetric = bool(symmetric)
        self._extrapolation = str(extrapolation)

    # --- Constructors -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "TorqueLookupTable":
        """Load and validate a v1 torque-lookup data file."""
        import yaml  # local import: optional dep, matches training-tree convention

        p = Path(path)
        if not p.is_file():
            raise TorqueLookupSchemaError(
                f"torque_lookup: file not found: {p}"
            )
        with p.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        if not isinstance(payload, dict):
            raise TorqueLookupSchemaError(
                f"torque_lookup: {p} did not parse to a mapping "
                f"(got {type(payload).__name__})"
            )
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TorqueLookupTable":
        """Build from a parsed payload (YAML file or deploy_contract section).

        Inverse of :meth:`to_dict`; the round-trip is byte-tight (required by
        Step 4.2, which embeds tables into the deploy_contract).
        """
        if not isinstance(d, dict):
            raise TorqueLookupSchemaError(
                f"torque_lookup: expected a mapping, got {type(d).__name__}"
            )
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise _raise_version(version)

        interpolation = d.get("interpolation", "linear")
        symmetric = d.get("symmetric", True)
        extrapolation = d.get("extrapolation", "clamp_to_endpoint")
        axis_label = d.get("axis", _AXIS_LABEL_DEFAULT)

        table = d.get("table")
        if not isinstance(table, (list, tuple)):
            raise TorqueLookupSchemaError(
                "torque_lookup: `table` must be a list of [axis, max_torque] "
                f"rows (got {type(table).__name__})"
            )
        axis_vals: List[float] = []
        torque_vals: List[float] = []
        for i, row in enumerate(table):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise TorqueLookupSchemaError(
                    f"torque_lookup: table row {i} must be a 2-element "
                    f"[axis, max_torque] pair, got {row!r}"
                )
            try:
                axis_vals.append(float(row[0]))
                torque_vals.append(float(row[1]))
            except (TypeError, ValueError) as exc:
                raise TorqueLookupSchemaError(
                    f"torque_lookup: table row {i} has non-numeric values: "
                    f"{row!r}"
                ) from exc

        return cls(
            axis=np.asarray(axis_vals, dtype=np.float64),
            max_torque=np.asarray(torque_vals, dtype=np.float64),
            axis_label=axis_label,
            interpolation=interpolation,
            symmetric=symmetric,
            extrapolation=extrapolation,
        )

    # --- Serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Emit the v1 payload. Byte-tight inverse of :meth:`from_dict`."""
        return {
            "schema_version": SCHEMA_VERSION,
            "axis": self._axis_label,
            "interpolation": self._interpolation,
            "symmetric": self._symmetric,
            "extrapolation": self._extrapolation,
            "array_ordering": "monotonic_increasing_axis",
            "table": [
                [float(a), float(t)]
                for a, t in zip(self._axis.tolist(), self._max_torque.tolist())
            ],
        }

    # --- Evaluation ---------------------------------------------------------

    def max_torque_at(self, q: float) -> float:
        """Return the (positive) torque ceiling |τ_max| at joint angle *q*."""
        return float(self.max_torque_batch(np.asarray([q], dtype=np.float64))[0])

    def max_torque_batch(self, q: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`max_torque_at`. Same shape as *q*.

        Linear interpolation between rows. Out-of-range handling follows the
        table's ``extrapolation`` setting:
          - clamp_to_endpoint: nearest endpoint value (np.interp default;
            identical to IsaacLab's zero-order-hold).
          - error: raise on any query outside [axis_min, axis_max].
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if self._extrapolation == "error":
            lo, hi = self._axis[0], self._axis[-1]
            if np.any(q_arr < lo) or np.any(q_arr > hi):
                raise TorqueLookupSchemaError(
                    f"torque_lookup: query outside axis range [{lo}, {hi}] and "
                    f"extrapolation='error' (min={float(np.min(q_arr))}, "
                    f"max={float(np.max(q_arr))})"
                )
        # np.interp clamps to endpoints for out-of-range x, matching
        # clamp_to_endpoint / IsaacLab LinearInterpolation.
        return np.interp(q_arr, self._axis, self._max_torque)

    # --- Properties / exports ----------------------------------------------

    @property
    def axis_range(self) -> Tuple[float, float]:
        """(axis_min, axis_max) of the table's domain."""
        return (float(self._axis[0]), float(self._axis[-1]))

    @property
    def peak_torque(self) -> float:
        """Maximum |τ_max| across the table (for sanity checks / provenance)."""
        return float(np.max(self._max_torque))

    @property
    def min_torque(self) -> float:
        """Minimum |τ_max| across the table (for sanity checks / provenance)."""
        return float(np.min(self._max_torque))

    def to_isaaclab_array(self) -> np.ndarray:
        """Emit the 3-column ``[angle, gear_ratio, max_torque]`` array that
        IsaacLab's ``RemotizedPDActuatorCfg.joint_parameter_lookup`` expects.

        ``gear_ratio`` is the constant ``1.0``. This is NOT a magic number:
        per PV-2 (b), IsaacLab's ``RemotizedPDActuator.compute()`` exposes the
        transmission-ratio column as a property but NEVER references it — only
        the angle column and the max_torque column feed the torque clamp. The
        gear ratio is pure metadata that does not enter the PD law or the
        ceiling. ``1.0`` is therefore a semantically-null placeholder; UnitPort
        does not model gear_ratio at runtime on either engine.
        """
        n = self._axis.shape[0]
        gear_ratio = np.ones(n, dtype=np.float64)
        return np.column_stack([self._axis, gear_ratio, self._max_torque])

    def sha256(self) -> str:
        """Stable content hash for provenance.

        Hashes a CANONICAL form (sorted keys, fixed float format) so the digest
        is identical across reloads of the same logical table regardless of
        YAML field ordering or float-repr differences. This is intentionally
        independent of :meth:`to_dict` (which preserves exact values for the
        byte-tight round-trip); the hash normalizes formatting instead.
        """
        canon = {
            "schema_version": SCHEMA_VERSION,
            "axis": self._axis_label,
            "interpolation": self._interpolation,
            "symmetric": self._symmetric,
            "extrapolation": self._extrapolation,
            "table": [
                [_HASH_FLOAT_FMT % float(a), _HASH_FLOAT_FMT % float(t)]
                for a, t in zip(self._axis.tolist(), self._max_torque.tolist())
            ],
        }
        blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        lo, hi = self.axis_range
        return (
            f"TorqueLookupTable(n={self._axis.shape[0]}, "
            f"axis=[{lo:.3f},{hi:.3f}], peak={self.peak_torque:.1f}Nm, "
            f"interp={self._interpolation})"
        )


# --- Validation -------------------------------------------------------------

def _validate(
    *,
    axis: np.ndarray,
    max_torque: np.ndarray,
    interpolation: str,
    symmetric: bool,
    extrapolation: str,
) -> None:
    """Enforce every v1 invariant; raise TorqueLookupSchemaError on violation."""
    # interpolation
    if interpolation not in _VALID_INTERPOLATION:
        raise TorqueLookupSchemaError(
            f"torque_lookup: interpolation={interpolation!r} is not a "
            f"recognized v1 enum value {_VALID_INTERPOLATION}"
        )
    if interpolation not in _IMPLEMENTED_INTERPOLATION:
        raise TorqueLookupSchemaError(
            f"torque_lookup: interpolation={interpolation!r} is reserved but "
            f"not implemented in v1; only {_IMPLEMENTED_INTERPOLATION} is "
            f"supported. IsaacLab interpolates the same table linearly with "
            f"zero-order-hold; a non-linear method on the MuJoCo side would "
            f"diverge the two engines. Use `interpolation: linear`."
        )
    # symmetric
    if symmetric is not True:
        raise TorqueLookupSchemaError(
            "torque_lookup: v1 supports only `symmetric: true` (the table "
            "gives |τ_max|, lower bound = −|τ_max|). Asymmetric "
            "positive/negative ceilings are a v2 feature — bump the schema "
            "and use a v2 reader instead of `symmetric: false`."
        )
    # extrapolation
    if extrapolation not in _VALID_EXTRAPOLATION:
        raise TorqueLookupSchemaError(
            f"torque_lookup: extrapolation={extrapolation!r} not in "
            f"{_VALID_EXTRAPOLATION}"
        )
    # table length
    if axis.shape[0] < 2:
        raise TorqueLookupSchemaError(
            f"torque_lookup: table must have >= 2 rows, got {axis.shape[0]}"
        )
    if axis.shape[0] != max_torque.shape[0]:
        raise TorqueLookupSchemaError(
            f"torque_lookup: axis ({axis.shape[0]}) and max_torque "
            f"({max_torque.shape[0]}) length mismatch"
        )
    # finite
    if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(max_torque)):
        raise TorqueLookupSchemaError(
            "torque_lookup: axis / max_torque contain non-finite values"
        )
    # strictly increasing axis (array_ordering: monotonic_increasing_axis)
    diffs = np.diff(axis)
    if np.any(diffs <= 0.0):
        bad = int(np.argmax(diffs <= 0.0))
        raise TorqueLookupSchemaError(
            f"torque_lookup: axis column must be STRICTLY increasing "
            f"(array_ordering=monotonic_increasing_axis); violation at row "
            f"{bad + 1}: {float(axis[bad])} -> {float(axis[bad + 1])}"
        )
    # positive torque
    if np.any(max_torque <= 0.0):
        bad = int(np.argmax(max_torque <= 0.0))
        raise TorqueLookupSchemaError(
            f"torque_lookup: all max_torque values must be > 0; row {bad} = "
            f"{float(max_torque[bad])}"
        )
