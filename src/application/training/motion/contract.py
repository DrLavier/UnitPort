# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reference motion contract — schema wrapping a MotionClip with the
metadata required to route it to the right consumer.

A :class:`MotionClip` carries the numeric payload (joint angles, root
pose, velocities). The contract wraps that payload with:

  * ``consumption_mode`` — which downstream node may consume this clip
    (e.g. AMP discriminator vs tracking target). Mismatched routing
    fails loud at the consumer.
  * ``source_info`` — provenance for diagnostics and reproducibility:
    where the clip came from on disk, and whether it was loaded via a
    pack URI.
  * ``target_family`` / ``target_sku`` — REQUIRED binding to the robot
    the clip was prepared for. Both fields are non-empty strings; the
    loader receives them as explicit arguments from its caller and
    propagates them onto the contract. Joint roles must align with
    the registry's IR slice for the named family.

The contract is the value returned by :meth:`MotionLoader.load`. Direct
access to the underlying :class:`MotionClip` is available via
``contract.clip`` — existing AMP code paths (``MotionAmpData``,
``compute_amp_obs_from_motion``, RSI) accept ``MotionClip`` unchanged,
so callers extract it on the boundary.

Schema versioning
-----------------
Format ``MAJOR.MINOR.PATCH``. Loaders stamp ``schema_version`` on every
contract they construct. The validator pins ``v1`` to the literal string
``"1.0.0"``; minor / patch bumps add optional fields, major bump is
breaking and unsupported by this module. The contract is a pure
in-memory value (no disk serialization) — making the previously
``Optional`` ``target_sku`` / ``target_family`` fields required is a
construction-API tightening that does not touch the on-disk schema, so
the version pin stays unchanged.

Consumption mode enum
---------------------
Three values are declared:

  * ``"amp_discriminator"`` — H0 reference-motion path. Clips feed the
    AMP discriminator as the reference distribution.
  * ``"tracking_target"`` — reserved for tracking-style training where
    the policy reproduces the clip trajectory directly. Not implemented
    in H0; consumption mode raises on this value.
  * ``"hybrid"`` — reserved for mixed-mode training that combines AMP
    reward with explicit tracking. Not implemented in H0.

Loaders default to ``"amp_discriminator"``; future loaders that need
``tracking_target`` will set it explicitly once the consumer side lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from application.training.motion.clip import MotionClip


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReferenceMotionContractError(ValueError):
    """Raised when a :class:`ReferenceMotionContract` fails schema validation
    (missing required field, unknown enum value, version mismatch, etc.)."""


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


CONTRACT_SCHEMA_VERSION = "1.0.0"

#: Every consumption mode the schema knows about. Used to reject typos
#: and unknown values at validate time.
ALLOWED_CONSUMPTION_MODES = frozenset({
    "amp_discriminator",
    "tracking_target",
    "hybrid",
})

#: Subset that is actually implemented today. Contracts declaring a
#: reserved mode validate against the enum but fail the H0 gate.
H0_CONSUMPTION_MODES = frozenset({"amp_discriminator"})


# ---------------------------------------------------------------------------
# Source provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """Where the clip came from.

    Attributes
    ----------
    source_path
        Absolute path the loader resolved before reading. Never empty.
    source_pack_uri
        Original ``pack:<namespace>:<rel>`` URI when the load originated
        from a pack resolution; ``None`` when the loader was invoked
        with an absolute path directly.
    """

    source_path: str
    source_pack_uri: Optional[str] = None


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceMotionContract:
    """Self-describing wrapper around a :class:`MotionClip`.

    Required fields (all non-empty):
        ``schema_version`` — pinned to :data:`CONTRACT_SCHEMA_VERSION`.
        ``consumption_mode`` — :data:`ALLOWED_CONSUMPTION_MODES`.
        ``clip`` — the validated :class:`MotionClip` payload.
        ``source_info`` — :class:`SourceInfo` with non-empty ``source_path``.
        ``target_family`` — canonical family slug the clip was prepared
            for (matches ``registers.families`` vocabulary). Must be a
            non-empty string; loaders receive this from their caller.
        ``target_sku`` — canonical robot SKU
            (``registers.robots.build_sku(brand, model)``). Must be a
            non-empty string; joint roles cross-validate against the
            registry's IR slice for the named family.

    Construction-time guarantees are enforced by ``__post_init__``: an
    empty / missing ``target_family`` or ``target_sku`` raises
    :exc:`ReferenceMotionContractError` immediately at instantiation.
    Combined with the same guarantees on ``schema_version`` /
    ``consumption_mode`` they pin the no-silent-success rule
    (CLAUDE.md §1.8) at the contract surface — callers cannot construct
    a partially-wired contract that drifts into a downstream consumer.

    Frozen + dataclass to match the existing immutable registry data
    contract (FamilyMeta / SymmetryData / CanaryData).
    """

    schema_version: str
    consumption_mode: str
    clip: MotionClip
    source_info: SourceInfo
    target_family: str
    target_sku: str

    def __post_init__(self) -> None:
        # The four enum-shaped / object-shaped fields are verified at
        # depth by :func:`validate_contract` (L1-L6). This hook adds
        # only the non-empty / non-None checks the type system cannot
        # express, so constructing the dataclass with an empty
        # ``target_sku`` cannot silently slip past the loader boundary
        # and reappear as a wrong-robot bundle at deploy.
        if not isinstance(self.target_family, str) or not self.target_family.strip():
            raise ReferenceMotionContractError(
                "ReferenceMotionContract.target_family is required and "
                "must be a non-empty string (canonical family slug from "
                "registers.families, e.g. 'quadruped' / 'humanoid'). "
                "Fix the loader caller to plumb the family explicitly "
                "— do not synthesise an empty placeholder."
            )
        if not isinstance(self.target_sku, str) or not self.target_sku.strip():
            raise ReferenceMotionContractError(
                "ReferenceMotionContract.target_sku is required and "
                "must be a non-empty string (canonical robot SKU from "
                "registers.robots.build_sku(brand, model)). Fix the "
                "loader caller to plumb the SKU explicitly — do not "
                "synthesise an empty placeholder."
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_contract(contract: ReferenceMotionContract) -> None:
    """Structural and semantic checks. Raises on the first problem.

    Layers (mirrors registry validators):
      L1  type checks                            - dataclass shape
      L2  schema_version pinned                  - "1.0.0" only
      L3  consumption_mode ∈ ALLOWED             - reject unknown
      L4  consumption_mode ∈ H0 implemented set  - reject reserved
      L5  clip presence + type                   - must be MotionClip
      L6  source_info.source_path non-empty      - no anonymous loads
    """
    if not isinstance(contract, ReferenceMotionContract):
        raise ReferenceMotionContractError(
            f"validate_contract: expected ReferenceMotionContract, got "
            f"{type(contract).__name__!r}"
        )

    if contract.schema_version != CONTRACT_SCHEMA_VERSION:
        raise ReferenceMotionContractError(
            f"ReferenceMotionContract.schema_version={contract.schema_version!r} "
            f"does not match the supported version "
            f"{CONTRACT_SCHEMA_VERSION!r}. Re-export the clip with the "
            f"current loader."
        )

    mode = contract.consumption_mode
    if mode not in ALLOWED_CONSUMPTION_MODES:
        raise ReferenceMotionContractError(
            f"ReferenceMotionContract.consumption_mode={mode!r} is not a "
            f"known value. Allowed: {sorted(ALLOWED_CONSUMPTION_MODES)!r}."
        )
    if mode not in H0_CONSUMPTION_MODES:
        raise ReferenceMotionContractError(
            f"ReferenceMotionContract.consumption_mode={mode!r} is reserved "
            f"for a future workstream (only "
            f"{sorted(H0_CONSUMPTION_MODES)!r} is implemented). Producing a "
            f"clip with this mode requires the matching consumer node to "
            f"land first."
        )

    if not isinstance(contract.clip, MotionClip):
        raise ReferenceMotionContractError(
            f"ReferenceMotionContract.clip must be MotionClip, got "
            f"{type(contract.clip).__name__!r}"
        )

    if not isinstance(contract.source_info, SourceInfo):
        raise ReferenceMotionContractError(
            f"ReferenceMotionContract.source_info must be SourceInfo, got "
            f"{type(contract.source_info).__name__!r}"
        )
    if not contract.source_info.source_path:
        raise ReferenceMotionContractError(
            "ReferenceMotionContract.source_info.source_path is required "
            "and may not be an empty string."
        )


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "ALLOWED_CONSUMPTION_MODES",
    "H0_CONSUMPTION_MODES",
    "ReferenceMotionContract",
    "ReferenceMotionContractError",
    "SourceInfo",
    "validate_contract",
]
