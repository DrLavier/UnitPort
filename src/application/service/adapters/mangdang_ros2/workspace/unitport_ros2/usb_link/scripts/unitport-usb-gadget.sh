#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# Assemble the UnitPort USB Ethernet gadget as a CDC-NCM function.
# Runs under unitport-usb-gadget.service as a oneshot at boot.
#
# Why NCM (Network Control Model) and nothing else:
#   * Windows 10 1703+ (April 2017) loads it natively via usbncm.sys — no
#     driver install, no OS Descriptor + Compat ID dance that RNDIS
#     requires and that Windows 11 has regressed on.
#   * macOS 10.10+ and every modern Linux kernel support it natively.
#   * Single function + single config → zero host-OS selection logic, no
#     multi-config enumeration flakiness.
#   * One MAC → one dnsmasq dhcp-host line for pinning the host to .2.
# The trade-off: pre-2017 Windows builds without NCM can't use the tether.
# That's a non-issue for any realistic Mini Pupper development host.

set -euo pipefail

GADGET=/sys/kernel/config/usb_gadget/unitport
CONFIGFS=/sys/kernel/config

log() { printf '[unitport-usb-gadget] %s\n' "$*"; }

# Ensure configfs is mounted and libcomposite is loaded.
if ! mountpoint -q "$CONFIGFS"; then
    modprobe configfs 2>/dev/null || true
    mount -t configfs none "$CONFIGFS" 2>/dev/null || true
fi
modprobe libcomposite 2>/dev/null || true

# Idempotent teardown: leave no partial gadget state from a previous run.
if [[ -d "$GADGET" ]]; then
    log "tearing down existing gadget"
    if [[ -w "$GADGET/UDC" ]]; then
        echo "" > "$GADGET/UDC" 2>/dev/null || true
    fi
    for cfg in "$GADGET"/configs/*/; do
        [[ -d "$cfg" ]] || continue
        for link in "$cfg"*; do
            [[ -L "$link" ]] && rm -f "$link"
        done
        [[ -d "${cfg}strings/0x409" ]] && rmdir "${cfg}strings/0x409" 2>/dev/null || true
        [[ -d "${cfg}strings" ]] && rmdir "${cfg}strings" 2>/dev/null || true
        rmdir "$cfg" 2>/dev/null || true
    done
    for fn in "$GADGET"/functions/*; do
        [[ -d "$fn" ]] && rmdir "$fn" 2>/dev/null || true
    done
    [[ -d "$GADGET/strings/0x409" ]] && rmdir "$GADGET/strings/0x409" 2>/dev/null || true
    [[ -d "$GADGET/strings" ]] && rmdir "$GADGET/strings" 2>/dev/null || true
    [[ -d "$GADGET/os_desc" ]] && rmdir "$GADGET/os_desc" 2>/dev/null || true
    rmdir "$GADGET" 2>/dev/null || true
fi

# Build gadget.
mkdir -p "$GADGET"
cd "$GADGET"

# USB IDs — Linux Foundation / Multifunction Composite Gadget.
echo 0x1d6b > idVendor
echo 0x0104 > idProduct
echo 0x0200 > bcdUSB
echo 0x0100 > bcdDevice

# bDeviceClass 0x02 = Communications Device Class. NCM is a CDC sub-protocol
# so this is the correct parent class. bDeviceSubClass + bDeviceProtocol 0
# means "see the interface descriptors" — which is where NCM advertises
# itself.
echo 0x02 > bDeviceClass
echo 0x00 > bDeviceSubClass
echo 0x00 > bDeviceProtocol

# Serial number is stable across reboots — derived from wlan0 MAC suffix.
MAC="$(cat /sys/class/net/wlan0/address 2>/dev/null || echo '00:00:00:00:00:00')"
SERIAL="minipupper-$(echo "$MAC" | tr -d ':' | tail -c 7)"

mkdir -p strings/0x409
echo "$SERIAL"                   > strings/0x409/serialnumber
echo "MangDang / UnitPort"       > strings/0x409/manufacturer
echo "Mini Pupper 2 (UnitPort)"  > strings/0x409/product

# NCM function — the single function this gadget exposes. The host-side and
# device-side MACs are stable across reboots so dnsmasq's dhcp-host line
# can pin the host to 192.168.55.2 deterministically.
mkdir -p functions/ncm.usb0
echo DE:AD:BE:EF:00:01 > functions/ncm.usb0/host_addr
echo DE:AD:BE:EF:00:02 > functions/ncm.usb0/dev_addr

# Single config — no multi-config selection logic, no OS Descriptor needed.
# Host OS drivers (Windows usbncm.sys / macOS AppleUSBNCMData / Linux
# cdc_ncm) bind unconditionally when they see a CDC-NCM function descriptor.
mkdir -p configs/c.1
echo 250 > configs/c.1/MaxPower
mkdir -p configs/c.1/strings/0x409
echo "UnitPort NCM" > configs/c.1/strings/0x409/configuration
ln -s functions/ncm.usb0 configs/c.1/

# Bind to the first available UDC (there's only one on CM4 OTG).
UDC_NAME="$(ls /sys/class/udc 2>/dev/null | head -n1 || true)"
if [[ -z "$UDC_NAME" ]]; then
    log "ERROR: no UDC — ensure dtoverlay=dwc2,dr_mode=peripheral and reboot"
    exit 1
fi
echo "$UDC_NAME" > UDC
log "gadget bound to UDC $UDC_NAME (NCM)"

# Bring usb0 up AND assign the static IP directly here.
# Why not netplan: some MangDang / Raspbian images use NetworkManager as
# the netplan renderer. Our ``renderer: networkd`` YAML is parsed but its
# generated systemd-networkd files are then ignored — usb0 stays DOWN
# with no IP, dnsmasq fails "unknown interface usb0", identity fails
# "Cannot assign requested address". Doing it inline here works regardless
# of which network stack is in charge, and since this script runs on every
# boot the assignment re-materialises automatically.
if [[ -e /sys/class/net/usb0 ]]; then
    # Wait briefly for the kernel to finish registering the netif — with
    # some libcomposite builds the add-event lags the UDC-bind write by a
    # few hundred milliseconds.
    for _try in 1 2 3 4 5; do
        if [[ -e /sys/class/net/usb0 ]]; then break; fi
        sleep 0.2
    done
    ip link set usb0 up 2>/dev/null || true
    # `ip addr add` is non-idempotent (fails with "File exists" when the
    # address is already assigned) — swallow that specific error so a
    # manual service restart doesn't 'fail'.
    if ! ip -4 addr show dev usb0 | grep -q '192\.168\.55\.1/24'; then
        ip addr add 192.168.55.1/24 dev usb0 2>/dev/null || true
    fi
    log "usb0 up, addr=$(ip -4 -br addr show usb0 | awk '{print $3}')"
fi
