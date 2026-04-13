"""RobotAsset validator — three-tier path b strategy.

Per AMP_design.yaml §7.risks.usd_parser_venv_dependency, validation tiers:

Tier 1 — User provides ``mjcf_path`` (with or without other paths):
    - mjcf_parser is the primary metadata source
    - Full validation in main venv: joint_names, dof_count, link tree
    - If usd_path also present, defer cross-check to launcher subprocess
      (a deferred-check warning is recorded in source_meta['warnings'])

Tier 2 — User provides only ``urdf_path``:
    - urdf_parser is the metadata source
    - Same coverage as tier 1 minus the cross-check

Tier 3 — User provides only ``usd_path``:
    - In main venv: warn-only (asset registers but joint_names is empty,
      ``source_meta['primary'] == 'none'``, warning recorded). The
      launcher subprocess will run the real usd_parser before training.

A combined tier 1+2 (mjcf + urdf both present) prefers mjcf as primary
and uses urdf only as a sanity diff.
"""
from __future__ import annotations

from typing import List

from src.system.training.robot_assets.registry import (
    RobotAsset,
    RobotAssetValidationError,
)
from src.system.training.robot_assets.parsers import (
    parse_urdf,
    parse_mjcf,
    UrdfParseError,
    MjcfParseError,
)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def validate_asset(asset: RobotAsset) -> None:
    """Validate ``asset`` in-place: populates parsed metadata + raises on error.

    On success: ``asset.joint_names`` / ``dof_count`` / ``base_link`` /
    ``mass`` / ``foot_link_names`` / ``default_joint_pos`` are filled in,
    and ``asset.source_meta`` records which file was used as primary.

    On failure: raises ``RobotAssetValidationError`` with a diff in the
    message. The asset is left in a partially-populated state — callers
    should not register it.
    """
    if not asset.asset_id:
        raise RobotAssetValidationError("RobotAsset.asset_id must be non-empty")

    if not (asset.usd_path or asset.mjcf_path or asset.urdf_path or asset.usd_url):
        raise RobotAssetValidationError(
            f"RobotAsset '{asset.asset_id}' must have at least one of "
            f"usd_path / mjcf_path / urdf_path / usd_url. Without any "
            f"source there is nothing to register."
        )
    # NB: urdf-only is allowed for metadata inspection, but
    # resolve_for_training() and resolve_for_mujoco() will still raise
    # because neither usd nor mjcf is present.
    # NB: usd_url (Nucleus) is accepted as a training-side reference
    # without an on-disk check — Isaac Sim resolves it at runtime.

    warnings: List[str] = []

    # ── Tier 1: prefer mjcf as primary ──
    if asset.mjcf_path is not None:
        if not asset.mjcf_path.is_file():
            raise RobotAssetValidationError(
                f"RobotAsset '{asset.asset_id}': mjcf_path does not exist: {asset.mjcf_path}"
            )
        try:
            mjcf_meta = parse_mjcf(asset.mjcf_path)
        except MjcfParseError as exc:
            raise RobotAssetValidationError(
                f"RobotAsset '{asset.asset_id}': mjcf parse failed — {exc}"
            ) from exc

        asset.joint_names = list(mjcf_meta.joint_names)
        asset.dof_count = mjcf_meta.dof_count
        asset.base_link = mjcf_meta.base_link
        asset.mass = mjcf_meta.mass
        # Foot links from heuristic; user override comes through node params
        # at a higher layer (RobotAssetNode merges param-supplied foot list).
        if not asset.foot_link_names:
            asset.foot_link_names = list(mjcf_meta.foot_link_names)
        if not asset.default_joint_pos:
            asset.default_joint_pos = dict(mjcf_meta.default_joint_pos)
        asset.source_meta["primary"] = "mjcf"

        # If urdf is also present, do a sanity diff (joint_names should match
        # in count and ideally in name; we tolerate name differences because
        # mjcf <-> urdf joint naming is inconsistent in many community robots).
        if asset.urdf_path is not None and asset.urdf_path.is_file():
            try:
                urdf_meta = parse_urdf(asset.urdf_path)
            except UrdfParseError as exc:
                warnings.append(f"urdf sanity parse failed: {exc}")
            else:
                if urdf_meta.dof_count != mjcf_meta.dof_count:
                    raise RobotAssetValidationError(
                        f"RobotAsset '{asset.asset_id}': mjcf/urdf dof mismatch — "
                        f"mjcf={mjcf_meta.dof_count} dof, urdf={urdf_meta.dof_count} dof. "
                        f"The two files describe structurally different robots."
                    )

        # If usd is also present, defer cross-check to launcher subprocess.
        if asset.usd_path is not None:
            if not asset.usd_path.is_file():
                raise RobotAssetValidationError(
                    f"RobotAsset '{asset.asset_id}': usd_path does not exist: {asset.usd_path}"
                )
            warnings.append(
                "usd present alongside mjcf — joint name cross-check is deferred "
                "to the Isaac Sim subprocess (path b)."
            )

        asset.source_meta["warnings"] = warnings
        _final_invariants(asset)
        return

    # ── Tier 2: urdf only ──
    if asset.urdf_path is not None:
        if not asset.urdf_path.is_file():
            raise RobotAssetValidationError(
                f"RobotAsset '{asset.asset_id}': urdf_path does not exist: {asset.urdf_path}"
            )
        try:
            urdf_meta = parse_urdf(asset.urdf_path)
        except UrdfParseError as exc:
            raise RobotAssetValidationError(
                f"RobotAsset '{asset.asset_id}': urdf parse failed — {exc}"
            ) from exc

        asset.joint_names = list(urdf_meta.joint_names)
        asset.dof_count = urdf_meta.dof_count
        asset.base_link = urdf_meta.base_link
        asset.mass = urdf_meta.mass
        if not asset.foot_link_names:
            asset.foot_link_names = list(urdf_meta.foot_link_names)
        if not asset.default_joint_pos:
            asset.default_joint_pos = dict(urdf_meta.default_joint_pos)
        asset.source_meta["primary"] = "urdf"

        if asset.usd_path is not None:
            if not asset.usd_path.is_file():
                raise RobotAssetValidationError(
                    f"RobotAsset '{asset.asset_id}': usd_path does not exist: {asset.usd_path}"
                )
            warnings.append(
                "usd present alongside urdf — joint name cross-check is deferred "
                "to the Isaac Sim subprocess (path b)."
            )

        asset.source_meta["warnings"] = warnings
        _final_invariants(asset)
        return

    # ── Tier 3: usd only — warn-only registration ──
    # Reached when neither mjcf nor urdf is available. Two sub-cases:
    #   (a) asset.usd_path set (on-disk .usd file)
    #   (b) asset.usd_url set (Nucleus URL, e.g. menagerie fallback)
    if asset.usd_path is not None:
        if not asset.usd_path.is_file():
            raise RobotAssetValidationError(
                f"RobotAsset '{asset.asset_id}': usd_path does not exist: {asset.usd_path}"
            )
    # usd_url doesn't need a filesystem check — Isaac Sim resolves it
    # at runtime via Nucleus.

    asset.source_meta["primary"] = "none"
    if asset.usd_url:
        warnings.append(
            f"Nucleus-only asset: usd_url={asset.usd_url}. Joint metadata "
            f"is not parsed in the main venv; Isaac Lab validates the URL "
            f"at training launch. MuJoCo sim2sim is unavailable without "
            f"a .xml MJCF file."
        )
    else:
        warnings.append(
            "USD-only asset: joint metadata will be verified at training launch "
            "by the Isaac Sim subprocess. This asset cannot be used for MuJoCo "
            "sim2sim until a .xml MJCF file is also provided."
        )
    asset.source_meta["warnings"] = warnings
    # No final_invariants here — joint_names is intentionally empty.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_invariants(asset: RobotAsset) -> None:
    """Sanity-check the populated metadata. Raises on contradiction."""
    if asset.dof_count <= 0:
        raise RobotAssetValidationError(
            f"RobotAsset '{asset.asset_id}': dof_count={asset.dof_count} (expected > 0). "
            f"The robot file appears to have no actuated joints."
        )
    if len(asset.joint_names) != asset.dof_count:
        # Should not happen given how parsers fill these in, but the
        # invariant is cheap to check and catches future regressions.
        raise RobotAssetValidationError(
            f"RobotAsset '{asset.asset_id}': len(joint_names)={len(asset.joint_names)} "
            f"!= dof_count={asset.dof_count}. Parser bug or corrupt file."
        )
    if not asset.base_link:
        raise RobotAssetValidationError(
            f"RobotAsset '{asset.asset_id}': base_link could not be determined."
        )
