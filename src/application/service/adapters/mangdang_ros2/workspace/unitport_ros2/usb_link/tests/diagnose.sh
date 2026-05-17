#!/usr/bin/env bash
# UnitPort USB link — on-robot diagnostic.
# Prints every piece of state that matters for "why isn't handshake working"
# so a copy-paste of the output is enough to root-cause remotely.
#
#   sudo /opt/unitport/ros2_ws/src/unitport_ros2/usb_link/tests/diagnose.sh
#
# Safe to run any time — purely read-only (no installs, no service changes).

set -uo pipefail      # no -e: keep going past any one failed check

section() {
    printf '\n\033[36m=== %s ===\033[0m\n' "$*"
}

section "0. Environment"
echo "  uname:    $(uname -srmo)"
echo "  distro:   $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
echo "  uptime:   $(uptime -p 2>/dev/null || uptime)"
echo "  date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"

section "1. config.txt dwc2 line"
if [[ -f /boot/firmware/config.txt ]]; then
    grep -nE 'dtoverlay=dwc2' /boot/firmware/config.txt || echo "  !! no dtoverlay=dwc2 line found"
else
    echo "  !! /boot/firmware/config.txt not present"
fi

section "2. UDC (peripheral-mode USB controller)"
if [[ -d /sys/class/udc ]]; then
    ls -1 /sys/class/udc 2>/dev/null | sed 's/^/  /'
    if [[ -z "$(ls -A /sys/class/udc 2>/dev/null)" ]]; then
        echo "  !! /sys/class/udc is EMPTY — dr_mode=peripheral did not take effect."
        echo "     Check config.txt and reboot."
    fi
else
    echo "  !! /sys/class/udc missing — dwc2 module not loaded?"
fi

section "3. Gadget configfs state"
GADGET=/sys/kernel/config/usb_gadget/unitport
if [[ -d "$GADGET" ]]; then
    echo "  gadget present at $GADGET"
    [[ -r "$GADGET/UDC" ]] && echo "  UDC: '$(cat "$GADGET/UDC" 2>/dev/null)'"
    echo "  functions: $(ls "$GADGET/functions/" 2>/dev/null | tr '\n' ' ')"
    echo "  configs:   $(ls "$GADGET/configs/" 2>/dev/null | tr '\n' ' ')"
else
    echo "  !! $GADGET not present — unitport-usb-gadget.service did not run"
fi

section "4. usb0 / usb1 interfaces"
for iface in usb0 usb1; do
    if [[ -d "/sys/class/net/$iface" ]]; then
        echo "  [$iface] present"
        ip -br link show "$iface" 2>/dev/null | sed 's/^/    link: /'
        ip -4 -br addr show "$iface" 2>/dev/null | sed 's/^/    addr: /'
    else
        echo "  [$iface] not present"
    fi
done

section "5. systemd units"
for unit in unitport-usb-gadget.service \
            unitport-identity.service \
            unitport-dnsmasq.service \
            unitport-handshake.service \
            unitport.target; do
    state="$(systemctl is-active "$unit" 2>/dev/null || true)"
    enabled="$(systemctl is-enabled "$unit" 2>/dev/null || true)"
    printf '  %-35s active=%-10s enabled=%s\n' "$unit" "${state:-?}" "${enabled:-?}"
done

section "6. /etc/unitport tree (what the install dropped)"
if [[ -d /etc/unitport ]]; then
    find /etc/unitport -maxdepth 3 -type f -o -type l 2>/dev/null | sort | sed 's/^/  /'
else
    echo "  !! /etc/unitport missing — install did not complete"
fi

section "7. /etc/netplan + /etc/dnsmasq.d"
ls -la /etc/netplan/ 2>/dev/null | sed 's/^/  /'
echo "  ---"
ls -la /etc/dnsmasq.d/ 2>/dev/null | sed 's/^/  /' || echo "  !! /etc/dnsmasq.d missing"

section "8. Identity endpoint (localhost probe)"
if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 http://192.168.55.1:9999/identity 2>&1 | head -20 | sed 's/^/  /'
else
    python3 -c 'import urllib.request, sys
try:
    with urllib.request.urlopen("http://192.168.55.1:9999/identity", timeout=2) as r:
        sys.stdout.write(r.read().decode("utf-8", "replace"))
except Exception as e:
    print(f"  probe failed: {e}")' | sed 's/^/  /'
fi

section "9. dnsmasq leases (host-side IPs seen)"
for f in /var/lib/misc/dnsmasq.leases /var/lib/dhcp/dhcpd.leases /var/lib/dnsmasq/dnsmasq.leases; do
    if [[ -f "$f" ]]; then
        echo "  $f:"
        cat "$f" | sed 's/^/    /'
    fi
done

section "10. Recent journal lines — unitport-* (last 30 each)"
for unit in unitport-usb-gadget unitport-identity unitport-dnsmasq unitport-handshake; do
    echo "  --- $unit ---"
    journalctl -u "${unit}.service" -n 30 --no-pager 2>&1 | sed 's/^/    /' || true
done

section "11. Kernel — dwc2/gadget/rndis/ecm (last 30)"
dmesg 2>/dev/null | grep -iE 'dwc2|gadget|rndis|ecm|configfs' | tail -30 | sed 's/^/  /' || \
    echo "  (dmesg unreadable — run as root)"

echo
echo "=== diagnose complete ==="
