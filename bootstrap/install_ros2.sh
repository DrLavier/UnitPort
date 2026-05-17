#!/usr/bin/env bash
# install_ros2.sh — Linux ROS2 bridge setup.
#
# Checks that Docker Engine is available, pulls the base ROS2 Humble
# image, and builds the UnitPort ros2_bridge image. Results are written
# to runtime/env/install_state.json.
#
# Optional first argument: --check-native-first
#   When set, runs bootstrap/detect_ros2.py first; if a native ROS2 Humble
#   install is found the script records ros2_mode=native in install_state.json
#   and exits 0 without touching Docker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${PROJECT_ROOT}/runtime/env"
STATE_FILE="${STATE_DIR}/install_state.json"
BRIDGE_IMAGE="${UNITPORT_ROS2_IMAGE:-unitport/ros2_bridge:humble}"

say() { printf '[install_ros2] %s\n' "$*"; }
fail() { printf '[install_ros2][ERROR] %s\n' "$*" >&2; exit 1; }

PY_EXE="${PROJECT_ROOT}/.venv311/bin/python"
if [ ! -x "$PY_EXE" ]; then
    PY_EXE="python3"
fi

if [ "${1:-}" = "--check-native-first" ]; then
    say "Probing for native ROS2 Humble..."
    mkdir -p "$STATE_DIR"
    ROS2_JSON=$("$PY_EXE" "${SCRIPT_DIR}/detect_ros2.py" 2>/dev/null || echo '{"found": false}')
    if "$PY_EXE" -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if (d.get('found') and d.get('distro')=='humble') else 1)" "$ROS2_JSON" >/dev/null 2>&1; then
        say "Native ROS2 Humble detected; skipping Docker bridge build."
        "$PY_EXE" - "$ROS2_JSON" "$STATE_FILE" <<'PY'
import json, sys, time, pathlib
d = json.loads(sys.argv[1])
sf = pathlib.Path(sys.argv[2])
s = {}
if sf.exists():
    try:
        s = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        s = {}
s["ros2_mode"] = "native"
s["ros2_distro"] = d.get("distro", "")
s["ros2_ros_root"] = d.get("ros_root", "")
s["ros2_verified_at"] = int(time.time())
sf.write_text(json.dumps(s, indent=2), encoding="utf-8")
print(f"[install_ros2] State written to {sf}")
PY
        exit 0
    fi
    say "No native ROS2 Humble found; proceeding with Docker bridge build."
fi

say "Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  fail "docker CLI not found. Install Docker Engine: https://docs.docker.com/engine/install/"
fi
if ! docker info >/dev/null 2>&1; then
  fail "docker info failed — daemon not running or permissions denied."
fi

say "Pulling ros:humble-ros-base (cache warm-up)..."
docker pull ros:humble-ros-base

say "Building ${BRIDGE_IMAGE}..."
docker build -t "${BRIDGE_IMAGE}" "${PROJECT_ROOT}/runtime/docker/ros2_bridge"

say "Smoke-testing image..."
docker run --rm "${BRIDGE_IMAGE}" bash -lc 'ros2 --help >/dev/null && echo ok'

mkdir -p "${STATE_DIR}"
"$PY_EXE" - <<PY
import json, os, time, pathlib
state_file = pathlib.Path(r"${STATE_FILE}")
state = {}
if state_file.exists():
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        state = {}
state["ros2_bridge_verified"] = True
state["ros2_bridge_image"] = r"${BRIDGE_IMAGE}"
state["ros2_bridge_verified_at"] = int(time.time())
state["ros2_mode"] = "both" if state.get("ros2_mode") == "native" else "docker"
state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(f"[install_ros2] State written to {state_file}")
PY

say "Done. Bridge image: ${BRIDGE_IMAGE}"
