"""Runtime helpers for the robot_assets overlay — read-side, MuJoCo spawn sites.

This module is imported from the three MuJoCo spawn paths (Review Pose,
generic_mujoco_env, MjSimEnv) so they don't depend on the heavier
:mod:`application.service.robot_assets.service` module directly. The split
also lets the runtime call sites stay synchronous and dependency-free of
PyQt / TasksManager — important because :class:`MjSimEnv` may instantiate
in headless deploy paths where the QApplication / Task plumbing isn't
available.

Public surface:

    read_mjcf_base_offset(sku, *, bundle_contract=None)
        -> (offset_z: float, calibrated: bool)

The ``bundle_contract`` optional input lets deploy-time PolicyRunner pass
the bundle's ``deploy_contract.mjcf_base_height_offset`` so a freshly
unpacked bundle on a machine without the user overlay still gets the
correct spawn z (CLAUDE.md §1.9 — portable artifacts must be self-contained).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from unitport_sdk import log_warning


# Per-SKU "we already warned about this in this process" dedup. The three
# spawn sites can each fire dozens to thousands of resets per session;
# without dedup the log would drown in repeated WARN lines for the same
# uncalibrated SKU.
_warned_uncalibrated: set = set()
_warned_stale: set = set()


def _coerce_offset(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    try:
        return float(payload.get("offset_z", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None


def read_mjcf_base_offset(
    sku: str,
    *,
    bundle_contract: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool]:
    """Return ``(offset_z, calibrated)`` for the given SKU.

    Priority order:

    1. ``bundle_contract`` if supplied (deploy-time, self-contained
       artifact). Wins over the local overlay so cross-machine bundle
       deploys are consistent.
    2. ``USER_CONFIG_DIR/robot_assets/state.json :: assets[sku].mjcf_base_height_offset``
       overlay computed by ``calibrate_mjcf_base_offset``.

    Failure modes:

    - SKU empty / no overlay / no bundle field → ``(0.0, False)`` plus a
      one-shot ``log_warning`` per SKU per process. The runtime caller
      still proceeds — the spawn z is just uncorrected. UI surfaces this
      via the Robot Node "uncalibrated" badge so the user knows.
    - Overlay status ``not_applicable`` (manipulator) → ``(0.0, True)``
      with no warn (the family has no spawn-z drift problem by design).
    - Overlay status ``calibrated`` but no ``offset_z`` field present →
      treated as uncalibrated (same WARN).

    Stale-detection (mjcf_sha256 / standing_pose_sha256 mismatch) is
    handled by the freshness gate in
    :func:`application.service.robot_assets.service.ensure_mjcf_base_offset_cached`
    — by the time we read here, a re-calibration task may be in flight;
    we use whatever the overlay says right now.
    """
    if not sku:
        return (0.0, False)

    # ---- (1) Bundle contract wins (deploy-time portable artifact) ----
    if isinstance(bundle_contract, dict):
        contract_payload = bundle_contract.get("mjcf_base_height_offset")
        if isinstance(contract_payload, dict):
            status = str(contract_payload.get("status", "") or "")
            if status == "not_applicable":
                return (0.0, True)
            off = _coerce_offset(contract_payload)
            if off is not None and status == "calibrated":
                return (off, True)
        elif isinstance(contract_payload, (int, float)):
            # Tolerate a bare float in the contract (older bundle layout)
            # so legacy bundles still get the lift.
            return (float(contract_payload), True)

    # ---- (2) User overlay ----
    try:
        from application.service.robot_assets.service import (
            get_robot_asset_service,
        )

        svc = get_robot_asset_service()
        overlay = svc.get_mjcf_base_offset(sku)
    except Exception as exc:
        # WHY KEPT (§1.8 (c)): service load can fail in deploy paths
        # (USER_CONFIG_DIR unset, PyQt absent). Warn once and proceed
        # with no correction rather than crashing the runtime.
        if sku not in _warned_uncalibrated:
            log_warning(
                f"[mjcf_base_offset] sku={sku!r}: cannot read overlay "
                f"({exc}); spawn z will not be corrected this session."
            )
            _warned_uncalibrated.add(sku)
        return (0.0, False)

    if not isinstance(overlay, dict):
        if sku not in _warned_uncalibrated:
            log_warning(
                f"[mjcf_base_offset] sku={sku!r}: no overlay (uncalibrated). "
                f"Spawn z will not be corrected. Reconnect the Robot Node so "
                f"the one-shot calibration task can run, or check that the "
                f"registry has bodies_per_format.MJCF + "
                f"default_init_joint_angles populated for this SKU."
            )
            _warned_uncalibrated.add(sku)
        return (0.0, False)

    status = str(overlay.get("status", "") or "")
    if status == "not_applicable":
        return (0.0, True)
    if status != "calibrated":
        if sku not in _warned_uncalibrated:
            log_warning(
                f"[mjcf_base_offset] sku={sku!r}: overlay status={status!r} "
                f"(expected 'calibrated' or 'not_applicable'); spawn z "
                f"uncorrected this session."
            )
            _warned_uncalibrated.add(sku)
        return (0.0, False)

    off = _coerce_offset(overlay)
    if off is None:
        if sku not in _warned_uncalibrated:
            log_warning(
                f"[mjcf_base_offset] sku={sku!r}: overlay has no usable "
                f"offset_z field; spawn z uncorrected this session."
            )
            _warned_uncalibrated.add(sku)
        return (0.0, False)
    return (off, True)


def clear_runtime_warn_cache() -> None:
    """Reset the per-process WARN dedup. Used by tests and by the service
    when the user manually re-calibrates a SKU so a fresh failure surfaces
    on the next spawn instead of being silently dedup'd."""
    _warned_uncalibrated.clear()
    _warned_stale.clear()


__all__ = ["read_mjcf_base_offset", "clear_runtime_warn_cache"]
