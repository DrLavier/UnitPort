"""ConnectionProfile — full per-connection profile dataclass.

Verbatim 15-field shape ported from DEMO's connector profile (Phase 1,
2026-05-14). Phase 1 UI in RealRobotConnectionCard only binds 5 of the
fields (``connector_type`` derived from the adapter dropdown,
``ros_domain_id`` from the domain combo, ``pupper_ip`` from the address
edit, ``ssh_port`` from the port edit, ``ros_distro`` left as default);
the remaining 10 fields carry DEMO defaults so the dataclass shape is
already final and Phase 3+ never has to migrate it.

The factory ``build_profile_from_info`` lives in ``ros2_connection_config``
(not here) to avoid a circular import — that module already imports from
this one to type-annotate the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConnectionProfile:
    """One concrete robot connection target.

    Fields mirror DEMO's ``ConnectionProfile`` 1:1 so adapter/connector
    code can be lifted in Phase 3 without re-shaping the call sites.

    Phase 1 only populates the subset wired through the UI; the rest stay
    at their default values until the corresponding feature lands:

      brand / robot                       — set in Phase 5 (brand SDK adapter)
      pupper_ip / ssh_port / ssh_user     — pupper_ip set in Phase 1 (=address)
      ros_domain_id                       — set in Phase 1 (=domain combo)
      busid                               — set in Phase 2 (USB transport)
      auto_reconnect                      — Phase 3+ (steady-state behaviour)
      last_connected_at                   — Phase 3+ (persistence)
      apt_mirror                          — Phase 4 (greenfield bootstrap)
      connector_type                      — set in Phase 1 ("ros2_generic" / "brand_sdk")
      robot_hostname                      — Phase 3 (post-handshake discovery)
      ros_distro                          — Phase 3 (mode-switch phase)
      ssh_timeout                         — Phase 3 (SSH transport)
      unitport_target_isolated            — Phase 4 (greenfield mode switch)
      middleware                          — Phase 3.9 (UI middleware combo:
                                            "Auto" / "DDS (CycloneDDS)" / ...).
                                            ``MangdangRmwInstalledProbe`` reads
                                            this to decide whether to ERROR on
                                            a robot that lacks cyclonedds.
      detected_rmw                        — Phase 3.9 (filled by
                                            ``MangdangRmwDetectionProbe`` when
                                            middleware == "Auto" and SSH is
                                            available; e.g. "rmw_fastrtps_cpp").
    """

    brand: str = ""
    robot: str = ""
    pupper_ip: str = ""
    ssh_port: int = 22
    ssh_user: str = "ubuntu"
    ros_domain_id: int = 42
    busid: str = ""
    auto_reconnect: bool = True
    last_connected_at: str = ""  # ISO-8601 UTC; empty = never connected
    apt_mirror: str = ""
    connector_type: str = "brand_sdk"
    robot_hostname: str = ""
    ros_distro: str = "humble"
    ssh_timeout: int = 60
    unitport_target_isolated: bool = False
    middleware: str = "Auto"
    detected_rmw: Optional[str] = None


__all__ = ["ConnectionProfile"]
