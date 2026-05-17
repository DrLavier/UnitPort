#!/usr/bin/env bash
# Flip the UnitPort DDS active profile between usb and wifi.
#   unitport-dds-select.sh {usb|wifi|status}
# Restarts unitport-bridge.service on change so the new RMW env takes effect.

set -euo pipefail

DDS_DIR="/etc/unitport/dds"
ACTIVE="${DDS_DIR}/active.env"

action="${1:-status}"

case "$action" in
    usb)  target="${DDS_DIR}/usb_only.env"  ;;
    wifi) target="${DDS_DIR}/wifi_only.env" ;;
    status)
        if [[ -L "$ACTIVE" ]]; then
            readlink -f "$ACTIVE"
        else
            echo "(no active profile)"
        fi
        exit 0
        ;;
    *)
        echo "usage: $0 {usb|wifi|status}" >&2
        exit 2
        ;;
esac

if [[ ! -f "$target" ]]; then
    echo "DDS profile not found: $target" >&2
    exit 3
fi

ln -sfn "$target" "$ACTIVE"
echo "[unitport-dds] active.env → $(basename "$target")"

if systemctl is-active --quiet unitport-bridge.service 2>/dev/null; then
    systemctl restart unitport-bridge.service
    echo "[unitport-dds] unitport-bridge.service restarted"
fi
