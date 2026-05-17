#!/usr/bin/env bash
# UnitPort full uninstaller — removes every UnitPort-authored artifact from
# the robot and returns the boot path to the factory baseline. Safe to run
# multiple times (idempotent) and safe to run on a half-installed robot.
#
#   sudo /opt/unitport/ros2_ws/src/unitport_ros2/scripts/uninstall_robot.sh
#   sudo /opt/unitport/ros2_ws/src/unitport_ros2/scripts/uninstall_robot.sh --purge-deps
#
# What gets removed:
#   * unitport.target + unitport-* services under /etc/systemd/system/
#   * /usr/local/bin/unitport-* helper scripts
#   * /etc/unitport/ (config, env, dds profiles, backups, mode marker)
#   * /etc/netplan/60-unitport-usb.yaml + /etc/dnsmasq.d/unitport-usb.conf
#   * /opt/unitport/ (entire workspace, colcon build artifacts)
#   * /boot/firmware/config.txt: restored from our backup copy
#
# What is NOT removed by default (use --purge-deps to also remove):
#   * dnsmasq (apt package) — installed by us but also used by other stacks
#   * python3-colcon-common-extensions (apt package) — ~150 MB of build deps
#
# Factory assets NEVER touched:
#   * /etc/netplan/50-cloud-init.yaml
#   * joystick.service, mini_pupper_*, /opt/mangdang/*
#   * Any /etc/udev/rules.d/99-mini_pupper-*.rules

set -uo pipefail     # no -e: uninstall should march on past a missing item

WS_SRC="/opt/unitport/ros2_ws/src/unitport_ros2"
ETC_DIR="/etc/unitport"
LOCAL_BIN="/usr/local/bin"
SYSTEMD_DIR="/etc/systemd/system"
WS_ROOT="/opt/unitport"

PURGE_DEPS=0
for arg in "$@"; do
    case "$arg" in
        --purge-deps) PURGE_DEPS=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
    esac
done

log() { printf '[unitport-uninstall] %s\n' "$*"; }

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "uninstall_robot.sh must be run as root (sudo)." >&2
        exit 1
    fi
}

stop_and_disable_services() {
    log "stopping + disabling UnitPort services"
    # Additive stop — leaves default.target (and every vendor service
    # running underneath it: sshd, factory ROS2, Flask panel) untouched.
    # The old code ran `systemctl isolate default.target` which was also
    # harmless on well-configured robots but asymmetric with the install
    # path; keeping it symmetric means operators who run uninstall then
    # reinstall don't see spurious "default.target rebuilt" side-effects.
    log "  stopping unitport.target (vendor stack stays up)"
    systemctl stop unitport.target 2>/dev/null || true
    local services=(
        unitport.target
        unitport-bridge.service
        unitport-estop.service
        unitport-state-pub.service
        unitport-param.service
        unitport-banner.service
        unitport-usb-gadget.service
        unitport-identity.service
        unitport-dnsmasq.service
        unitport-handshake.service
        # Brand-specific runtime units. Stopping unitport-minipupper.service
        # triggers its ExecStopPost= that restarts mini_pupper_web.service,
        # so the factory Flask panel on :8080 comes back automatically on
        # the way out. Also safe to stop on robots that never had this unit
        # installed (systemctl stop on a missing unit is a no-op).
        unitport-minipupper.service
    )
    # One log line per service keeps the host-side idle-timeout watchdog happy:
    # even if `systemctl stop` takes 30s on a wedged unit, no silence window
    # exceeds the time between these prints + the 5s-per-call worst case.
    for svc in "${services[@]}"; do
        log "  stop + disable ${svc}"
        systemctl stop "$svc" 2>/dev/null || true
        systemctl disable "$svc" 2>/dev/null || true
    done
}

remove_systemd_units() {
    log "removing unit files under ${SYSTEMD_DIR}/"
    rm -f "${SYSTEMD_DIR}/unitport.target"
    rm -f "${SYSTEMD_DIR}/unitport-bridge.service"
    rm -f "${SYSTEMD_DIR}/unitport-estop.service"
    rm -f "${SYSTEMD_DIR}/unitport-state-pub.service"
    rm -f "${SYSTEMD_DIR}/unitport-param.service"
    rm -f "${SYSTEMD_DIR}/unitport-banner.service"
    rm -f "${SYSTEMD_DIR}/unitport-usb-gadget.service"
    rm -f "${SYSTEMD_DIR}/unitport-identity.service"
    rm -f "${SYSTEMD_DIR}/unitport-dnsmasq.service"
    rm -f "${SYSTEMD_DIR}/unitport-handshake.service"
    # Brand-specific runtime unit shipped by deploy_brand_runtime for
    # Mini Pupper v2 (ships dormant, started on-demand at session open).
    rm -f "${SYSTEMD_DIR}/unitport-minipupper.service"
    # Old drop-in path from the pre-owned-unit era.
    rm -f "${SYSTEMD_DIR}/dnsmasq.service.d/unitport.conf"
    rmdir "${SYSTEMD_DIR}/dnsmasq.service.d" 2>/dev/null || true
    # Drop-in we layered onto the factory mini_pupper_web.service to
    # enforce the PCA9685 I2C mutex. Removing it restores the vendor
    # unit's original, stand-alone behaviour byte-for-byte.
    rm -f "${SYSTEMD_DIR}/mini_pupper_web.service.d/unitport-conflicts.conf"
    rmdir "${SYSTEMD_DIR}/mini_pupper_web.service.d" 2>/dev/null || true
    # .wants symlinks systemctl enable created under our own target dir.
    rm -rf "${SYSTEMD_DIR}/unitport.target.wants"
    # Unmask the distro's dnsmasq.service we masked during install (leaving
    # it DISABLED — the user didn't ask for dnsmasq before we arrived).
    systemctl unmask dnsmasq.service 2>/dev/null || true
    # Narrow sudoers grant for mode-switching between UnitPort ROS2 mode
    # and the factory Flask panel. Safe to remove unconditionally — missing
    # file = nothing to do.
    rm -f /etc/sudoers.d/unitport-mode-switch
}

remove_scripts() {
    log "removing helper scripts from ${LOCAL_BIN}/"
    rm -f "${LOCAL_BIN}/unitport-usb-gadget.sh"
    rm -f "${LOCAL_BIN}/unitport-identity.py"
    rm -f "${LOCAL_BIN}/unitport-dds-select.sh"
    rm -f "${LOCAL_BIN}/unitport-handshake-probe.py"
    rm -f "${LOCAL_BIN}/unitport-banner.py"
}

remove_network_files() {
    log "removing UnitPort netplan / dnsmasq configs"
    rm -f /etc/netplan/60-unitport-usb.yaml
    rm -f /etc/dnsmasq.d/unitport-usb.conf
}

restore_config_txt() {
    local boot_config="/boot/firmware/config.txt"
    local backup="${ETC_DIR}/backup/$(basename "$boot_config").original"
    if [[ -f "$backup" && -f "$boot_config" ]]; then
        cp -p "$backup" "$boot_config"
        log "config.txt restored from ${backup}"
    elif [[ -f "$boot_config" ]]; then
        # Best-effort fallback: strip the dr_mode=peripheral suffix.
        sed -i 's|^[[:space:]]*dtoverlay=dwc2,dr_mode=peripheral[[:space:]]*$|dtoverlay=dwc2|' \
            "$boot_config" 2>/dev/null || true
        log "config.txt: dr_mode=peripheral stripped (no original backup found)"
    fi
}

remove_etc_and_opt() {
    log "removing ${ETC_DIR}/ and ${WS_ROOT}/"
    rm -rf "${ETC_DIR}"
    rm -rf "${WS_ROOT}"
}

purge_apt_deps() {
    if (( PURGE_DEPS != 1 )); then
        return 0
    fi
    log "--purge-deps set — removing dnsmasq + python3-colcon-common-extensions"
    apt-get -y remove --purge dnsmasq python3-colcon-common-extensions 2>/dev/null || true
    apt-get -y autoremove --purge 2>/dev/null || true
}

main() {
    require_root
    stop_and_disable_services
    remove_systemd_units
    remove_scripts
    remove_network_files
    restore_config_txt
    remove_etc_and_opt
    systemctl daemon-reload 2>/dev/null || true
    # Do NOT run `netplan apply` during a remote uninstall — it reloads
    # systemd-networkd, which briefly bounces wlan0 and kills the SSH
    # session the user is uninstalling through. `netplan generate` rewrites
    # the backing files without touching the live network; usb0 gets torn
    # down automatically on the next reboot.
    netplan generate 2>/dev/null || true
    purge_apt_deps
    log "uninstall complete"
    log ""
    log "Reboot to fully release USB OTG peripheral mode:"
    log "  sudo reboot"
    if (( PURGE_DEPS != 1 )); then
        log ""
        log "NOTE: these apt packages were installed by UnitPort and are still"
        log "      present. Remove with --purge-deps or manually:"
        log "        sudo apt-get -y remove --purge dnsmasq python3-colcon-common-extensions"
        log "        sudo apt-get -y autoremove"
    fi
}

main "$@"
