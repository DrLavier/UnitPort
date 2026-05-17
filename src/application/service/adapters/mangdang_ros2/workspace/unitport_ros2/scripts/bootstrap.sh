#!/usr/bin/env bash
# bootstrap.sh — UnitPort robot-side full bootstrap. ONE script. ALWAYS
# idempotent. Replaces the historical install_robot.sh + usb_link/install.sh
# split that left mask state, daemon constants, and partial installs as
# foot-guns for every operator who hit the wrong env var.
#
# Single-source-of-truth principles:
#   1. ALL 11 unitport-* units are unmasked + enabled by name in lockstep.
#      A stale mask from a prior install or manual debug never wedges the
#      system permanently.
#   2. UNITPORT_VERSION and UNITPORT_WORKSPACE_FINGERPRINT live ONLY in
#      /etc/unitport/identity.yaml. Daemon reads from disk. No RAM-stamped
#      module constants. No source-file sed at multiple stages.
#   3. USB tether stack is ALWAYS installed. The gadget service has
#      ConditionPathExists=/sys/class/udc, so on boards without USB OTG
#      hardware the unit silently no-ops. There is no UNITPORT_USB_LINK
#      env-var gate to forget.
#   4. Postcondition self-check fails the install if anything ended up in
#      a wrong state. Caller (host bridge_upgrade) reads exit 5 → surfaces
#      it. No silent partial success.
#
# Inputs (env vars set by caller; both optional, both fall back to
# package.xml lookup if unset for manual-SSH invocations):
#   UNITPORT_VERSION                 — workspace version string
#   UNITPORT_WORKSPACE_FINGERPRINT   — sha256-derived 16-hex content fingerprint
#   ROS_DISTRO                       — defaults to humble
#
# Usage:
#   sudo /opt/unitport/ros2_ws/src/unitport_ros2/scripts/bootstrap.sh
#   sudo UNITPORT_VERSION=0.3.0 UNITPORT_WORKSPACE_FINGERPRINT=abc123def4567890 \
#        bash /opt/unitport/ros2_ws/src/unitport_ros2/scripts/bootstrap.sh
#
# Exit codes:
#   0 — bootstrap completed and postcondition verified
#   1 — not running as root
#   2 — ROS_DISTRO not installed
#   3 — unitport.target ships an additive-violating directive
#   4 — colcon entrypoint verification failed
#   5 — postcondition failed (some unit masked / disabled, or identity.yaml incomplete)

set -euo pipefail

# ===== paths =====
WS_SRC="/opt/unitport/ros2_ws/src/unitport_ros2"
WS_ROOT="/opt/unitport/ros2_ws"
ETC_DIR="/etc/unitport"
BACKUP_DIR="${ETC_DIR}/backup"
LOCAL_BIN="/usr/local/bin"
SYSTEMD_DIR="/etc/systemd/system"
NETPLAN_FILE="/etc/netplan/60-unitport-usb.yaml"
DNSMASQ_CONF="/etc/dnsmasq.d/unitport-usb.conf"
BOOT_CONFIG="/boot/firmware/config.txt"
ROS_DISTRO="${ROS_DISTRO:-humble}"

USB_LINK_SRC="${WS_SRC}/usb_link"

TS="$(date -u +%Y%m%d-%H%M%SZ)"
REBOOT_REQUIRED=0

# UnitPort metadata from caller (env vars). Empty when invoked manually
# without the bridge_upgrade flow — we'll derive version from package.xml
# and leave fingerprint blank (host re-runs install when fingerprint
# matters).
UNITPORT_VERSION="${UNITPORT_VERSION:-}"
UNITPORT_WORKSPACE_FINGERPRINT="${UNITPORT_WORKSPACE_FINGERPRINT:-}"

# All 11 unitport-* systemd units. The exact list is hardcoded because
# discovering "all unitport-* units" via glob would also find disabled
# .wants symlinks and fail in unhelpful ways. Keep this list in lockstep
# with the .service files shipped under systemd/ and usb_link/systemd/.
ALL_UNITPORT_UNITS=(
    unitport.target
    unitport-estop.service
    unitport-bridge.service
    unitport-state-pub.service
    unitport-param.service
    unitport-banner.service
    unitport-minipupper.service
    unitport-usb-gadget.service
    unitport-identity.service
    unitport-dnsmasq.service
    unitport-handshake.service
)

# ===== logging =====
log() { printf '[unitport-bootstrap] %s\n' "$*"; }
err() { printf '[unitport-bootstrap] ERROR: %s\n' "$*" >&2; }

# ===== preflight =====
require_root() {
    if [[ $EUID -ne 0 ]]; then
        err "must be run as root (sudo)"
        exit 1
    fi
}

source_ros() {
    if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
        err "ROS2 ${ROS_DISTRO} not found at /opt/ros/${ROS_DISTRO}."
        err "Set ROS_DISTRO env var before invoking."
        exit 2
    fi
    # ROS2 setup.bash trips `set -u` on internal vars. Relax flags just
    # for the source and restore them after.
    set +u
    # shellcheck disable=SC1091
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    set -u
}

ensure_deps() {
    log "checking apt packages (python3-colcon-common-extensions, python3-yaml, dnsmasq)"
    if ! command -v colcon >/dev/null 2>&1; then
        apt-get update
        apt-get install -y python3-colcon-common-extensions
    fi
    if ! python3 -c 'import yaml' >/dev/null 2>&1; then
        apt-get install -y python3-yaml
    fi
    # dnsmasq is needed for unitport-dnsmasq.service. Apt creates
    # /etc/dnsmasq.d on install — required before our config drop-in.
    if ! command -v dnsmasq >/dev/null 2>&1; then
        log "dnsmasq not found — installing"
        apt-get update -qq || true
        apt-get install -y dnsmasq
    fi
}

backup_once() {
    local src="$1"
    mkdir -p "${BACKUP_DIR}"
    local base original
    base="$(basename "$src")"
    original="${BACKUP_DIR}/${base}.original"
    if [[ -f "$src" && ! -f "$original" ]]; then
        cp -p "$src" "$original"
    fi
    if [[ -f "$src" ]]; then
        cp -p "$src" "${BACKUP_DIR}/${base}.${TS}.bak"
    fi
}

# ===== mask self-heal — runs BEFORE we install any unit so the install
#       writes onto plain unit files, not over a /dev/null mask symlink =====
unmask_all_unitport_units() {
    log "unmasking all 11 unitport units (idempotent self-heal)"
    for u in "${ALL_UNITPORT_UNITS[@]}"; do
        systemctl unmask "$u" 2>/dev/null || true
    done
    systemctl daemon-reload
}

# ===== overlay tear-down (additive-only — never isolate) =====
stop_stale_overlay() {
    # Stop any in-flight unitport overlay so the install replaces the
    # active unit files cleanly. Conservative — only stops units we own.
    #
    # Self-foot-shooting guard: when we're invoked over SSH on the USB
    # tether (host=192.168.55.x), stopping unitport-usb-gadget.service
    # tears down the libcomposite UDC binding → kills usb0 → drops our
    # own SSH channel mid-install ("Socket exception 10054, 远程主机
    # 强迫关闭"). Detect the USB-SSH case and exclude the gadget stack
    # (gadget + identity + dnsmasq + handshake — all BindsTo-bound to
    # gadget). They survive the overlay refresh fine: install_systemd_*
    # rewrites the same unit files later, and reenable_all_units does the
    # daemon-reload. The gadget keeps running across this script and
    # picks up any unit-file changes on the next reboot.
    log "stopping any stale UnitPort overlay before installing fresh units"
    local skip_usb_link=0
    # Two signals — either is sufficient to skip the USB-link teardown.
    # SSH_CONNECTION is the cheap path when sudo preserves env; usb0
    # liveness is the fallback when sudo strips it (default sudoers).
    local ssh_src
    ssh_src="$(echo "${SSH_CONNECTION:-}" | awk '{print $1}')"
    if [[ "$ssh_src" == 192.168.55.* ]]; then
        skip_usb_link=1
        log "  SSH source $ssh_src is on USB tether → skip USB-link teardown"
    elif ip -o link show usb0 2>/dev/null | grep -q 'state UP'; then
        # usb0 is the libcomposite NCM interface our gadget service brings
        # up. If it's currently in 'state UP' the gadget is bound to a UDC
        # right now — stopping unitport-usb-gadget.service would write ""
        # to /sys/.../UDC, unbind, and tear down the interface. Whether or
        # not THIS install is going through usb0, anything else on it
        # (host-side ROS2, future automated tooling) takes a hit. Default
        # to "leave gadget running while it's up"; install_systemd_*
        # rewrites the unit files in place anyway and the next reboot
        # picks the new ones up cleanly.
        skip_usb_link=1
        log "  usb0 is UP → gadget is actively carrying traffic; skip USB-link teardown"
    fi
    if (( skip_usb_link == 1 )); then
        log "  → keeping unitport-usb-gadget/identity/dnsmasq/handshake running"
        log "  → install_systemd_usb_link will rewrite their unit files in place"
        log "  → reboot at the end of the install applies the new units cleanly"
    fi
    systemctl stop unitport.target 2>/dev/null || true
    for unit in "${ALL_UNITPORT_UNITS[@]}"; do
        if [[ "$unit" == "unitport.target" ]]; then
            continue
        fi
        if (( skip_usb_link == 1 )); then
            case "$unit" in
                unitport-usb-gadget.service|unitport-identity.service|\
                unitport-dnsmasq.service|unitport-handshake.service)
                    log "  skip stop $unit (carrying our SSH session)"
                    continue
                    ;;
            esac
        fi
        systemctl stop "$unit" 2>/dev/null || true
    done
}

# ===== legacy artifact removal =====
purge_legacy_artifacts() {
    log "purging legacy robot-side identity / bundle artifacts"
    local -a legacy_files=(
        "${ETC_DIR}/robot_id"
        "${ETC_DIR}/deploy.yaml"
        "${ETC_DIR}/deploy_manifest.json"
        "${ETC_DIR}/robot_mapping.json"
        "${ETC_DIR}/robot_profile.json"
    )
    local f
    for f in "${legacy_files[@]}"; do
        if [[ -e "$f" ]]; then
            rm -f "$f" && log "  removed $f"
        fi
    done
    local ssh_user="${SUDO_USER:-}"
    if [[ -z "$ssh_user" || "$ssh_user" == "root" ]]; then
        ssh_user=$(ls -1 /home 2>/dev/null | head -n 1 || true)
    fi
    if [[ -n "$ssh_user" ]]; then
        local home_fb="/home/${ssh_user}/.unitport/robot_profile.json"
        if [[ -e "$home_fb" ]]; then
            rm -f "$home_fb" && log "  removed $home_fb"
        fi
    fi
}

# ===== USB gadget hardware gate (Raspberry Pi config.txt) =====
patch_config_txt() {
    if [[ ! -f "$BOOT_CONFIG" ]]; then
        log "skip config.txt patch — $BOOT_CONFIG not found (non-Pi platform)"
        return 0
    fi
    if ! grep -qE '^[[:space:]]*dtoverlay=dwc2([[:space:]]*|,)' "$BOOT_CONFIG"; then
        log "skip config.txt patch — no dtoverlay=dwc2 line (non-USB-OTG board)"
        return 0
    fi
    if grep -qE '^[[:space:]]*dtoverlay=dwc2,dr_mode=peripheral[[:space:]]*$' "$BOOT_CONFIG"; then
        log "config.txt already pinned to dr_mode=peripheral"
        return 0
    fi
    backup_once "$BOOT_CONFIG"
    sed -i 's|^[[:space:]]*dtoverlay=dwc2[[:space:]]*$|dtoverlay=dwc2,dr_mode=peripheral|' "$BOOT_CONFIG"
    log "config.txt: dtoverlay=dwc2 → dtoverlay=dwc2,dr_mode=peripheral"
    REBOOT_REQUIRED=1
}

# ===== /etc/unitport/ defaults =====
stage_etc() {
    log "staging ${ETC_DIR}/ defaults"
    mkdir -p "${ETC_DIR}"
    [[ -f "${ETC_DIR}/env" ]] || cat > "${ETC_DIR}/env" <<EOF
# UnitPort runtime environment — edited by the host's onboarding pipeline.
ROS_DOMAIN_ID=42
ROS_LOG_DIR=/tmp/unitport-log
EOF
    grep -q '^ROS_DOMAIN_ID=' "${ETC_DIR}/env" || \
        echo 'ROS_DOMAIN_ID=42' >> "${ETC_DIR}/env"
    grep -q '^ROS_LOG_DIR=' "${ETC_DIR}/env" || \
        echo 'ROS_LOG_DIR=/tmp/unitport-log' >> "${ETC_DIR}/env"
    # No RMW_IMPLEMENTATION here on purpose: Mangdang factory image only
    # ships rmw_fastrtps_cpp; host's cyclonedds-python interops via RTPS.
    sed -i '/^RMW_IMPLEMENTATION=/d' "${ETC_DIR}/env" 2>/dev/null || true
    mkdir -p /tmp/unitport-log
    chmod 1777 /tmp/unitport-log
    [[ -f "${ETC_DIR}/adapter_registry.yaml" ]] || \
        install -m 0644 "${WS_SRC}/unitport_bringup/config/adapter_registry.yaml" \
                        "${ETC_DIR}/adapter_registry.yaml"
    [[ -f "${ETC_DIR}/tunable_params.yaml" ]] || \
        install -m 0644 "${WS_SRC}/unitport_bringup/config/tunable_params.yaml" \
                        "${ETC_DIR}/tunable_params.yaml"
}

# ===== ament workspace build =====
build_workspace() {
    log "building colcon workspace at ${WS_ROOT} (clean rebuild)"
    cd "${WS_ROOT}"
    # Clean rebuild + STANDARD install (NOT --symlink-install). The
    # --symlink-install layout puts ament_python entry-point scripts under
    # <prefix>/bin/<exe> instead of <prefix>/lib/<pkg>/<exe>, which
    # breaks `ros2 pkg executables`, `ros2 run`, and every systemd unit
    # ExecStart. Standard layout costs disk space; pays for itself.
    rm -rf install build log
    # event-handlers tweaks:
    #   console_direct+   - stream stdout/stderr LIVE per package instead
    #                       of buffering until package completion. Without
    #                       this, an SSH-driven install sees 60-120s
    #                       silences per package and the host's idle
    #                       timeout fires (was 90s — bumped to 240s on
    #                       host side too as belt-and-braces).
    #   console_cohesion- - disable the cohesion handler which is what
    #                       was buffering everything in the first place.
    #   console_start_end+- per-package start/end lines so the host log
    #                       gets a heartbeat at minimum every package.
    # PYTHONUNBUFFERED=1 covers ament_python build steps (setuptools'
    # ``running build`` etc.) that would otherwise sit in stdout buffers.
    PYTHONUNBUFFERED=1 colcon build \
        --executor sequential \
        --event-handlers console_direct+ console_cohesion- console_start_end+

    # Postcondition: each ros2cli sees the entrypoints we expect.
    log "verifying entrypoints via 'ros2 pkg executables'"
    set +u
    # shellcheck disable=SC1091
    source "${WS_ROOT}/install/setup.bash"
    set -u
    local exes ros2_rc
    exes="$(ros2 pkg executables 2>&1)" || ros2_rc=$?
    ros2_rc="${ros2_rc:-0}"
    local missing=()
    for spec in "unitport_estop estop_node" \
                "unitport_state_pub generic_state_pub_node" \
                "unitport_param_srv param_srv_node" \
                "unitport_bridge generic_bridge_node"; do
        if ! grep -qxF "${spec}" <<<"${exes}"; then
            missing+=("${spec}")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        echo "============================================================" >&2
        echo "FATAL: 'ros2 pkg executables' does not list these entrypoints" >&2
        echo "       after colcon build (workspace is in a broken state):"   >&2
        printf '  - %s\n' "${missing[@]}"                                    >&2
        echo                                                                 >&2
        echo "DIAGNOSTIC DUMP (so we can find where the exes actually went):" >&2
        echo "  ros2 pkg executables exit code: ${ros2_rc}"                  >&2
        echo "  ros2 pkg executables output (full, including stderr):"       >&2
        printf '%s\n' "${exes}" | sed 's/^/    /'                       >&2
        echo                                                                 >&2
        exit 4
    fi
    log "colcon build verified: all 4 unitport_* entrypoints visible"
}

# ===== systemd unit installation (main 7) =====
install_systemd_main() {
    log "installing 7 main systemd units"
    install -m 0644 "${WS_SRC}/systemd/unitport.target"              "${SYSTEMD_DIR}/unitport.target"
    install -m 0644 "${WS_SRC}/systemd/unitport-estop.service"       "${SYSTEMD_DIR}/unitport-estop.service"
    install -m 0644 "${WS_SRC}/systemd/unitport-bridge.service"      "${SYSTEMD_DIR}/unitport-bridge.service"
    install -m 0644 "${WS_SRC}/systemd/unitport-state-pub.service"   "${SYSTEMD_DIR}/unitport-state-pub.service"
    install -m 0644 "${WS_SRC}/systemd/unitport-param.service"       "${SYSTEMD_DIR}/unitport-param.service"
    install -m 0644 "${WS_SRC}/systemd/unitport-banner.service"      "${SYSTEMD_DIR}/unitport-banner.service"
    if [[ -r "${WS_SRC}/systemd/unitport-minipupper.service" ]]; then
        install -m 0644 "${WS_SRC}/systemd/unitport-minipupper.service" \
                        "${SYSTEMD_DIR}/unitport-minipupper.service"
    fi
    install -m 0755 "${WS_SRC}/scripts/unitport-banner.py"           "${LOCAL_BIN}/unitport-banner.py"
}

# ===== USB-link scripts to /usr/local/bin/ =====
install_usb_link_scripts() {
    log "installing USB-link scripts → ${LOCAL_BIN}/"
    install -m 0755 "${USB_LINK_SRC}/scripts/unitport-usb-gadget.sh"       "${LOCAL_BIN}/unitport-usb-gadget.sh"
    install -m 0755 "${USB_LINK_SRC}/scripts/unitport-identity.py"        "${LOCAL_BIN}/unitport-identity.py"
    install -m 0755 "${USB_LINK_SRC}/scripts/unitport-dds-select.sh"      "${LOCAL_BIN}/unitport-dds-select.sh"
    install -m 0755 "${USB_LINK_SRC}/scripts/unitport-handshake-probe.py" "${LOCAL_BIN}/unitport-handshake-probe.py"
}

# ===== systemd unit installation (USB-link 4) =====
install_systemd_usb_link() {
    log "installing 4 USB-link systemd units"
    install -m 0644 "${USB_LINK_SRC}/systemd/unitport-usb-gadget.service"  "${SYSTEMD_DIR}/unitport-usb-gadget.service"
    install -m 0644 "${USB_LINK_SRC}/systemd/unitport-identity.service"    "${SYSTEMD_DIR}/unitport-identity.service"
    install -m 0644 "${USB_LINK_SRC}/systemd/unitport-dnsmasq.service"     "${SYSTEMD_DIR}/unitport-dnsmasq.service"
    install -m 0644 "${USB_LINK_SRC}/systemd/unitport-handshake.service"   "${SYSTEMD_DIR}/unitport-handshake.service"
    # Disable + mask the distro's dnsmasq.service — apt enables it by
    # default; we run our own owned unit (unitport-dnsmasq) and want
    # nothing factory-adjacent serving DHCP at boot.
    systemctl disable --now dnsmasq.service >/dev/null 2>&1 || true
    systemctl mask dnsmasq.service >/dev/null 2>&1 || true
}

# ===== USB-link network config (netplan + dnsmasq) =====
install_network() {
    log "installing netplan + dnsmasq config (scoped to usb0)"
    mkdir -p "$(dirname "$DNSMASQ_CONF")"
    install -m 0600 "${USB_LINK_SRC}/netplan/60-unitport-usb.yaml" "$NETPLAN_FILE"
    install -m 0644 "${USB_LINK_SRC}/dnsmasq/unitport-usb.conf"    "$DNSMASQ_CONF"
    # Do NOT run `netplan apply` — it bounces every managed interface and
    # would drop our SSH session. `netplan generate` rewrites the
    # networkd files from yaml without touching the live network.
    netplan generate 2>/dev/null || log "netplan generate skipped (config may be stale)"
    log "netplan config written; takes effect on reboot"
}

# ===== DDS profile staging =====
install_dds() {
    log "staging DDS profiles"
    mkdir -p "${ETC_DIR}/dds"
    install -m 0644 "${USB_LINK_SRC}/dds/usb_only.env"            "${ETC_DIR}/dds/usb_only.env"
    install -m 0644 "${USB_LINK_SRC}/dds/wifi_only.env"           "${ETC_DIR}/dds/wifi_only.env"
    install -m 0644 "${USB_LINK_SRC}/dds/fastdds_usb_only.xml"    "${ETC_DIR}/dds/fastdds_usb_only.xml"
    install -m 0644 "${USB_LINK_SRC}/dds/cyclonedds_usb_only.xml" "${ETC_DIR}/dds/cyclonedds_usb_only.xml"
    if [[ ! -L "${ETC_DIR}/dds/active.env" && ! -e "${ETC_DIR}/dds/active.env" ]]; then
        ln -sfn "${ETC_DIR}/dds/wifi_only.env" "${ETC_DIR}/dds/active.env"
        log "active.env → wifi_only.env (default)"
    fi
}

# ===== legacy /etc/unitport/robot_id (back-compat for older host code) =====
write_robot_id_legacy() {
    mkdir -p "${ETC_DIR}"
    local hostname mac serial
    hostname="$(hostname)"
    mac="$(cat /sys/class/net/wlan0/address 2>/dev/null || echo '00:00:00:00:00:00')"
    serial="minipupper-$(echo "$mac" | tr -d ':' | tail -c 7)"
    cat > "${ETC_DIR}/robot_id" <<EOF
hostname=${hostname}
serial=${serial}
platform=raspberry_pi_cm4
carrier=mangdang/mini_pupper_2
form_factor=quadruped
transport=usb_ethernet_gadget
EOF
    chmod 0644 "${ETC_DIR}/robot_id"
}

# ===== reenable + enable all 11 units in lockstep =====
reenable_all_units() {
    log "daemon-reload + reenable all 11 unitport units"
    systemctl daemon-reload
    for u in "${ALL_UNITPORT_UNITS[@]}"; do
        # reenable = disable+enable: forces re-creation of .wants symlinks
        # from the current [Install] section, in case an old unit had a
        # different WantedBy (e.g. the additive-target migration).
        # Targets without [Install] are "static" — reenable will fail and
        # we fall through to enable, which also fails for static units —
        # both no-ops, hence the `|| true`.
        systemctl reenable "$u" >/dev/null 2>&1 \
            || systemctl enable "$u" >/dev/null 2>&1 \
            || true
    done
}

# ===== additive-target sanity check =====
verify_additive_target() {
    local target_file="${SYSTEMD_DIR}/unitport.target"
    if [[ -r "${target_file}" ]] && grep -qiE '^\s*Conflicts\s*=.*default\.target' "${target_file}"; then
        echo "=============================================================" >&2
        echo "FATAL: ${target_file} declares Conflicts=default.target."     >&2
        echo "       The additive-overlay design forbids this — it would"  >&2
        echo "       kill sshd / Flask / vendor ROS2 the moment unitport"  >&2
        echo "       .target is started."                                   >&2
        echo "=============================================================" >&2
        exit 3
    fi
    if [[ -r "${target_file}" ]] && grep -qiE '^\s*AllowIsolate\s*=\s*yes' "${target_file}"; then
        echo "WARN: ${target_file} has AllowIsolate=yes." >&2
    fi
    log "sanity check ok: unitport.target is additive (no Conflicts, no AllowIsolate)"
}

# ===== identity.yaml seed (single source of truth for version + fingerprint) =====
seed_identity_record() {
    # Resolve UnitPort metadata. Caller (host bridge_upgrade) sets env
    # vars; manual SSH invocations fall back to package.xml for version
    # and leave fingerprint blank (host re-runs install when fingerprint
    # actually matters for the Upgrade button).
    local ver="$UNITPORT_VERSION"
    local fp="$UNITPORT_WORKSPACE_FINGERPRINT"
    if [[ -z "$ver" ]]; then
        local pkg_xml="${WS_SRC}/unitport_bringup/package.xml"
        if [[ -r "$pkg_xml" ]]; then
            ver=$(grep -oE '<version>[^<]+</version>' "$pkg_xml" \
                  | head -n1 \
                  | sed -E 's|<version>(.*)</version>|\1|')
        fi
    fi
    log "seeding ${ETC_DIR}/identity.yaml (version=${ver:-?} fingerprint=${fp:0:8}…)"

    local script="${LOCAL_BIN}/unitport-identity.py"
    if [[ ! -x "$script" ]]; then
        err "${script} missing — install_usb_link_scripts must run first"
        return 1
    fi
    if ! python3 "$script" --probe \
            --unitport-version="$ver" \
            --workspace-fingerprint="$fp" 2>&1 | sed 's|^|    |'; then
        log "WARN: identity --probe exited non-zero (next Connect will retry)"
    fi
}

# ===== /opt/unitport ownership (SFTP-friendly) =====
grant_user_writable_unitport() {
    local ssh_user="${SUDO_USER:-}"
    if [[ -z "$ssh_user" || "$ssh_user" == "root" ]]; then
        ssh_user=$(ls -1 /home 2>/dev/null | head -n 1 || true)
    fi
    if [[ -n "$ssh_user" && "$ssh_user" != "root" ]]; then
        log "chown /opt/unitport → ${ssh_user}:${ssh_user} (enables SFTP writes)"
        chown -R "${ssh_user}:${ssh_user}" /opt/unitport 2>/dev/null || true
    else
        log "skip chown /opt/unitport — couldn't determine a non-root login user"
    fi
}

# ===== postcondition self-check =====
verify_postcondition() {
    log "verifying post-install state"
    local errors=0
    for u in "${ALL_UNITPORT_UNITS[@]}"; do
        local state
        state=$(systemctl is-enabled "$u" 2>/dev/null || true)
        case "$state" in
            masked)
                err "POSTCONDITION FAIL: $u is masked"
                errors=$((errors+1))
                ;;
            enabled|static|alias|enabled-runtime)
                : # ok
                ;;
            "")
                err "POSTCONDITION FAIL: $u not loadable (is-enabled returned empty)"
                errors=$((errors+1))
                ;;
            *)
                err "POSTCONDITION FAIL: $u is '$state' (expected enabled/static)"
                errors=$((errors+1))
                ;;
        esac
    done
    if [[ ! -f "${ETC_DIR}/identity.yaml" ]]; then
        err "POSTCONDITION FAIL: ${ETC_DIR}/identity.yaml missing"
        errors=$((errors+1))
    else
        if ! grep -q '^unitport_version:' "${ETC_DIR}/identity.yaml"; then
            err "POSTCONDITION FAIL: identity.yaml missing 'unitport_version:' line"
            errors=$((errors+1))
        fi
        if ! grep -q '^workspace_fingerprint:' "${ETC_DIR}/identity.yaml"; then
            err "POSTCONDITION FAIL: identity.yaml missing 'workspace_fingerprint:' line"
            errors=$((errors+1))
        fi
    fi
    if (( errors > 0 )); then
        err "$errors postcondition failure(s) — install state is broken"
        return 5
    fi
    log "postcondition OK: 11 units loadable, identity.yaml complete"
    return 0
}

print_reboot_message() {
    if [[ "${REBOOT_REQUIRED:-0}" == "1" ]]; then
        log "========================================"
        log "  REBOOT REQUIRED"
        log "  config.txt patched; USB OTG enters peripheral mode after reboot"
        log "  sudo reboot"
        log "========================================"
    fi
}

main() {
    log "starting UnitPort robot-side bootstrap"
    require_root
    source_ros
    ensure_deps
    unmask_all_unitport_units    # heal stale mask state BEFORE installing units
    stop_stale_overlay
    purge_legacy_artifacts
    patch_config_txt              # USB OTG hardware enable; non-fatal if non-Pi
    stage_etc                    # /etc/unitport/{env,adapter_registry,tunable_params}
    build_workspace               # colcon build + entrypoint verify
    install_systemd_main          # 7 main units + banner script
    install_usb_link_scripts      # 4 USB-link scripts → /usr/local/bin/
    install_systemd_usb_link      # 4 USB-link units + mask distro dnsmasq
    install_network               # netplan + dnsmasq config (no apply)
    install_dds                   # /etc/unitport/dds/*.env + active.env link
    write_robot_id_legacy         # back-compat /etc/unitport/robot_id
    reenable_all_units            # daemon-reload + reenable all 11 units
    verify_additive_target        # unitport.target must not Conflicts=default.target
    seed_identity_record          # /etc/unitport/identity.yaml with version + fp
    grant_user_writable_unitport
    if ! verify_postcondition; then
        exit 5
    fi
    print_reboot_message
    log "bootstrap complete."
    log "Next: sudo systemctl start unitport.target  (additive — sshd/Flask/vendor stay up)"
}

main "$@"
