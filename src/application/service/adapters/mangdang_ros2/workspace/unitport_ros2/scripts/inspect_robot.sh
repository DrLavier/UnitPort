#!/usr/bin/env bash
# Diagnostic helper — run on the robot to quickly dump what the inspector
# will see. Useful for debugging Layer A off-host.
#
#   bash inspect_robot.sh

set -u

log() { printf '=== %s ===\n' "$*"; }

SHELL_SRC="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [[ -f "${SHELL_SRC}" ]]; then
    # shellcheck disable=SC1090
    source "${SHELL_SRC}"
fi

log "hostname / os"
hostname
cat /etc/os-release | grep -E 'PRETTY_NAME|VERSION_ID' || true

log "cpu / ram"
uname -m
nproc
free -m | awk '/Mem:/ {print "ram_mb="$2}'

log "ros2 node list"
ros2 node list 2>/dev/null || echo "(ros2 cli unavailable)"

log "ros2 topic list -t"
ros2 topic list -t 2>/dev/null || true

log "ros2 service list -t | head"
ros2 service list -t 2>/dev/null | head -n 20 || true

log "robot_description (1st 2k chars)"
ros2 param get /robot_state_publisher robot_description 2>/dev/null | head -c 2000
echo

log "unitport files under /etc/unitport"
ls -la /etc/unitport/ 2>/dev/null || true
