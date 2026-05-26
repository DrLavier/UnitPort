# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ConnectionPhase enum — all phase IDs the orchestrator emits.

Phase IDs are verbatim copies of DEMO's ``ros2_generic_connector`` phase
constants (2026-05-14, Phase 1) so RELEASE's signal payloads match what
the real connector will emit in Phase 3+ without UI churn.

Two groups:

* ``alpha`` (4): first-time provisioning — used only by the greenfield
  branch landing in Phase 4. Phase 1's fake state machine does NOT walk
  these; they are defined here so the enum is complete.
* ``beta`` (11): brownfield / steady-state lifecycle. The full ordered
  sequence is exposed as :data:`BROWNFIELD_SEQUENCE` for the Phase 1
  fake state machine to iterate.

The string values match DEMO 1:1; future connectors emit the same strings
on ``AppSignals.connection_phase_changed``.
"""

from __future__ import annotations

from enum import Enum


class ConnectionPhase(str, Enum):
    """All connection phase IDs the L2 orchestrator may emit."""

    # --- alpha (greenfield prefix, Phase 4) ---
    P_ALPHA_DETECT = "alpha_detect_provisioned"
    P_ALPHA_BOOTSTRAP = "alpha_bootstrap"
    P_ALPHA_REBOOT = "alpha_trigger_reboot"
    P_ALPHA_RECOVERY = "alpha_await_recovery"

    # --- beta (brownfield / steady-state, Phase 3) ---
    P0_USB_IDENTITY_PROBE = "usb_identity_probe"
    P1_SSH_HANDSHAKE = "ssh_handshake"
    P2_MODE_SWITCH = "mode_switch"
    P3_INSPECTOR = "inspector"
    P4_ROBOT_PROFILE_PUSH = "robot_profile_push"
    P4B_BRAND_RUNTIME = "brand_runtime_deploy"
    P4C_ENTER_ROS2_MODE = "enter_ros2_mode"
    P5_BRIDGE_UP = "bridge_up"
    P6_MAPPING_HASH_CHECK = "mapping_hash_check"
    P7_ACTIVATE = "activate"
    P8_CONFIRMATION = "confirmation"
    # Phase 3.5 — post-connect self-diagnose. Runs HOST_PROBES + per-brand
    # adapter.diagnostic_probes(); never rolls back the connected state.
    P9_DIAGNOSTICS = "diagnostics"


# Ordered brownfield walk. The Phase 1 fake state machine iterates this
# tuple; Phase 3's real connector executes the same order with real I/O.
BROWNFIELD_SEQUENCE: tuple[ConnectionPhase, ...] = (
    ConnectionPhase.P0_USB_IDENTITY_PROBE,
    ConnectionPhase.P1_SSH_HANDSHAKE,
    ConnectionPhase.P2_MODE_SWITCH,
    ConnectionPhase.P3_INSPECTOR,
    ConnectionPhase.P4_ROBOT_PROFILE_PUSH,
    ConnectionPhase.P4B_BRAND_RUNTIME,
    ConnectionPhase.P4C_ENTER_ROS2_MODE,
    ConnectionPhase.P5_BRIDGE_UP,
    ConnectionPhase.P6_MAPPING_HASH_CHECK,
    ConnectionPhase.P7_ACTIVATE,
    ConnectionPhase.P8_CONFIRMATION,
    ConnectionPhase.P9_DIAGNOSTICS,
)


__all__ = ["ConnectionPhase", "BROWNFIELD_SEQUENCE"]
