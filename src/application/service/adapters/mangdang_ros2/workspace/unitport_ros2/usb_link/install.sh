#!/usr/bin/env bash
# usb_link/install.sh — backward-compatibility shim. The full bootstrap
# (including the USB link stack: gadget, identity daemon, dnsmasq,
# handshake, netplan, dnsmasq config, dwc2 dr_mode peripheral patch) is
# now in scripts/bootstrap.sh. The USB-link logic is no longer optional
# — bootstrap.sh always installs it. The unitport-usb-gadget.service
# uses ConditionPathExists=/sys/class/udc, so on boards without USB OTG
# hardware it silently no-ops without affecting the rest of the install.
#
# UNITPORT_USB_ROLLBACK=1 keeps its historical meaning: tear down only
# the USB-link artifacts (units, scripts, netplan, dnsmasq, config.txt
# patch). For "remove all of UnitPort" use a separate uninstall path —
# rollback here is intentionally USB-link-only to match the script name.

set -euo pipefail

USB_LINK_SRC="$(cd "$(dirname "$0")" && pwd)"
ETC_DIR="/etc/unitport"
BACKUP_DIR="${ETC_DIR}/backup"
LOCAL_BIN="/usr/local/bin"
SYSTEMD_DIR="/etc/systemd/system"
NETPLAN_FILE="/etc/netplan/60-unitport-usb.yaml"
DNSMASQ_CONF="/etc/dnsmasq.d/unitport-usb.conf"
BOOT_CONFIG="/boot/firmware/config.txt"

log() { printf '[unitport-usb] %s\n' "$*"; }
err() { printf '[unitport-usb] ERROR: %s\n' "$*" >&2; }

require_root() {
    if [[ $EUID -ne 0 ]]; then
        err "must be run as root (sudo)"
        exit 1
    fi
}

rollback_main() {
    log "rolling back UnitPort USB link artifacts"
    systemctl disable --now unitport-usb-gadget.service unitport-identity.service \
        unitport-dnsmasq.service unitport-handshake.service 2>/dev/null || true
    rm -f "${SYSTEMD_DIR}/unitport-usb-gadget.service"
    rm -f "${SYSTEMD_DIR}/unitport-identity.service"
    rm -f "${SYSTEMD_DIR}/unitport-dnsmasq.service"
    rm -f "${SYSTEMD_DIR}/unitport-handshake.service"
    rm -f /etc/systemd/system/dnsmasq.service.d/unitport.conf
    rmdir /etc/systemd/system/dnsmasq.service.d 2>/dev/null || true
    systemctl unmask dnsmasq.service >/dev/null 2>&1 || true
    rm -f "${LOCAL_BIN}/unitport-usb-gadget.sh"
    rm -f "${LOCAL_BIN}/unitport-identity.py"
    rm -f "${LOCAL_BIN}/unitport-dds-select.sh"
    rm -f "${LOCAL_BIN}/unitport-handshake-probe.py"
    rm -f "$NETPLAN_FILE"
    rm -f "$DNSMASQ_CONF"
    rm -rf "${ETC_DIR}/dds"
    rm -f "${ETC_DIR}/robot_id"
    if [[ -f "${BACKUP_DIR}/$(basename "$BOOT_CONFIG").original" ]]; then
        cp -p "${BACKUP_DIR}/$(basename "$BOOT_CONFIG").original" "$BOOT_CONFIG"
        log "config.txt restored from original backup"
    else
        sed -i 's|^[[:space:]]*dtoverlay=dwc2,dr_mode=peripheral[[:space:]]*$|dtoverlay=dwc2|' \
            "$BOOT_CONFIG" 2>/dev/null || true
        log "config.txt: dr_mode=peripheral removed (no original backup found)"
    fi
    systemctl daemon-reload
    netplan generate 2>/dev/null || true
    log "rollback complete — reboot to fully release USB OTG"
}

require_root
if [[ "${UNITPORT_USB_ROLLBACK:-0}" == "1" ]]; then
    rollback_main
else
    # Forward to bootstrap.sh — the unified installer.
    exec "${USB_LINK_SRC}/../scripts/bootstrap.sh" "$@"
fi
