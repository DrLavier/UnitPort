#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# Switch the UnitPort overlay services on or off — ADDITIVE, never exclusive.
#
#   sudo unitport_mode.sh on      # start UnitPort services; leave factory running
#   sudo unitport_mode.sh off     # stop UnitPort services; factory stays up
#   sudo unitport_mode.sh status
#
# The host's P2_MODE_SWITCH phase invokes this script. It's also safe to run
# by hand when debugging.
#
# HISTORY (important): earlier versions of this script called
# `systemctl isolate unitport.target` which Conflicted with default.target
# and killed sshd + the factory Flask panel on :8080 + the vendor ROS2
# bringup. That broke the Robot Control Panel (Start / Gait / Active
# Controller buttons all failed because Flask was gone) and made the robot
# unreachable over SSH until the next reboot. The new additive behaviour
# below preserves every vendor service. See
# memory/architecture_additive_not_exclusive.md in the host repo for the
# rule this script enforces.

set -euo pipefail

MODE_FILE="/etc/unitport/mode"
UNITPORT_TARGET="unitport.target"

log() { printf '[unitport-mode] %s\n' "$*"; }

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "unitport_mode.sh must be run as root." >&2
        exit 1
    fi
}

on() {
    require_root
    log "starting unitport.target (additive — default.target stays active)"
    # `systemctl start` activates the target without deactivating
    # default.target (unlike `isolate`). Because the refreshed
    # unitport.target declares no Conflicts clause, sshd / vendor ROS2 /
    # Flask keep running underneath.
    systemctl start "${UNITPORT_TARGET}"
    mkdir -p "$(dirname "${MODE_FILE}")"
    echo "on" > "${MODE_FILE}"
    log "UnitPort overlay is now ACTIVE alongside the vendor stack."
}

off() {
    require_root
    log "stopping unitport.target (vendor stack untouched)"
    # `stop` deactivates only the UnitPort overlay services. The factory
    # default.target remains active the whole time.
    systemctl stop "${UNITPORT_TARGET}" || true
    mkdir -p "$(dirname "${MODE_FILE}")"
    echo "off" > "${MODE_FILE}"
    log "UnitPort overlay OFF; vendor default.target continues unchanged."
}

status() {
    if [[ -f "${MODE_FILE}" ]]; then
        printf 'mode: %s\n' "$(cat "${MODE_FILE}")"
    else
        printf 'mode: unknown (marker missing)\n'
    fi
    systemctl is-active "${UNITPORT_TARGET}" || true
    # Also surface sshd status so operators can sanity-check that the
    # overlay has not taken the vendor stack down.
    systemctl is-active ssh 2>/dev/null || systemctl is-active sshd 2>/dev/null || true
}

case "${1:-status}" in
    on)     on ;;
    off)    off ;;
    status) status ;;
    *)
        echo "usage: $0 {on|off|status}" >&2
        exit 2
        ;;
esac
