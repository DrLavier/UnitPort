#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# UnitPort USB link — post-install verification. Runs on the robot after
# install + reboot + cable plugged in. Exits 0 only if every check passes.

set -uo pipefail

RED="\033[31m"; GREEN="\033[32m"; RESET="\033[0m"
fail=0

check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf "  ${GREEN}[OK]${RESET}   %s\n" "$name"
    else
        printf "  ${RED}[FAIL]${RESET} %s\n" "$name"
        fail=1
    fi
}

echo "UnitPort USB link — post-install checks"
echo

check "usb0 interface exists"                    ip link show usb0
check "usb0 has address 192.168.55.1"            sh -c 'ip -4 addr show usb0 | grep -q "192\.168\.55\.1"'
check "identity endpoint returns robot_id"       sh -c 'curl -fsS --max-time 3 http://192.168.55.1:9999/identity | grep -q robot_id'
check "wlan0 remains UP (WiFi unaffected)"       sh -c 'ip link show wlan0 | grep -q "state UP"'
check "joystick.service still active"            systemctl is-active --quiet joystick.service
check "usb0 is NOT a member of any bridge"       sh -c '! (bridge link 2>/dev/null | grep -qE "usb0.*master")'
check "dnsmasq is NOT bound to :53"              sh -c '! (ss -lntp 2>/dev/null | grep -qE ":53[[:space:]].*dnsmasq")'
check "Fast-DDS usb profile present"             test -f /etc/unitport/dds/fastdds_usb_only.xml
check "Cyclone DDS usb profile present"          test -f /etc/unitport/dds/cyclonedds_usb_only.xml
check "active.env symlink exists"                test -L /etc/unitport/dds/active.env
check "dtoverlay pinned to peripheral"           sh -c 'grep -qE "^[[:space:]]*dtoverlay=dwc2,dr_mode=peripheral[[:space:]]*$" /boot/firmware/config.txt'
check "unitport-usb-gadget.service active"       systemctl is-active --quiet unitport-usb-gadget.service
check "unitport-identity.service active"         systemctl is-active --quiet unitport-identity.service

echo
if [[ $fail -eq 0 ]]; then
    printf "${GREEN}All checks passed.${RESET}\n"
else
    printf "${RED}%d check(s) failed.${RESET}\n" "$fail"
fi
exit $fail
