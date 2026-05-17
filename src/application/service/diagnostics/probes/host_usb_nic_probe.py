"""HostUsbNicProbe — verify the USB NCM tether interface is up on the host.

When a Mangdang / mini_pupper is plugged in via USB, the robot exposes
a USB NCM ethernet adapter that the host enumerates with an address in
the 192.168.55.0/24 range (robot uses .1, host gets .2 by default).
Without that interface, the rest of the brownfield connect path can't
even reach the robot — so this probe runs first and emits an actionable
finding when the NIC is missing.
"""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network
from typing import List, Tuple

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import DiagnosticFinding


_USB_TETHER_NETS: Tuple[IPv4Network, ...] = (
    IPv4Network("192.168.55.0/24"),  # mangdang / mini_pupper default
    IPv4Network("192.168.7.0/24"),   # beaglebone style USB-ether
)


class HostUsbNicProbe(DiagnosticProbe):
    """Look for a host NIC with an address in the USB-tether subnets."""

    id = "host_usb_nic"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        # Only meaningful when the user picked the USB transport. Skip on
        # other transports so a wifi connect doesn't get a noisy USB warning.
        try:
            kind = getattr(ctx.transport, "kind", None)
            kind_label = str(getattr(kind, "name", kind) or "")
        except Exception:
            kind_label = ""
        if kind_label and kind_label.upper() != "USB":
            return [self.info(f"skipped: transport is {kind_label}")]

        try:
            import psutil
        except ImportError:
            return [self.info("psutil unavailable; cannot enumerate NICs")]

        try:
            addrs_per_nic = psutil.net_if_addrs()
            stats_per_nic = psutil.net_if_stats()
        except Exception as exc:  # noqa: BLE001
            return [self.warning(
                "psutil enumeration failed",
                detail=str(exc),
            )]

        matches: List[Tuple[str, str, bool]] = []
        for nic, addrs in addrs_per_nic.items():
            stats = stats_per_nic.get(nic)
            up = bool(stats and stats.isup)
            for a in addrs:
                family = getattr(a.family, "name", str(a.family))
                if "AF_INET" not in family or family.endswith("INET6"):
                    continue
                try:
                    ip = IPv4Address(a.address)
                except Exception:
                    continue
                for net in _USB_TETHER_NETS:
                    if ip in net:
                        matches.append((nic, str(ip), up))

        if not matches:
            return [self.error(
                "USB tether NIC absent",
                detail=(
                    "No host network interface holds a 192.168.55.x or "
                    "192.168.7.x address. Plug in the robot via USB; if "
                    "already plugged in, check Device Manager for an "
                    "unrecognised 'USB NCM Host Device'."
                ),
            )]

        any_up = any(up for _, _, up in matches)
        if not any_up:
            return [self.error(
                "USB tether NIC present but down",
                detail=(
                    "Found "
                    + ", ".join(f"{n}={ip}" for n, ip, _ in matches)
                    + " but the interface is administratively down. Toggle "
                    "it via Network Connections -> Enable."
                ),
            )]

        return [self.ok(
            "USB tether NIC up",
            detail=", ".join(f"{n}={ip}" for n, ip, _ in matches),
        )]


__all__ = ["HostUsbNicProbe"]
