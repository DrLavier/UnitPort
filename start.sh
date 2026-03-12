#!/bin/bash
# ==============================================================================
# UnitPort start.sh
# Launches UnitPort using the project-local Python environment.
#
# Priority order:
#   1. runtime/python/python (packaged runtime, if present)
#   2. .venv311/bin/python (project-local venv, default)
#
# Always injects project-local CYCLONEDDS_HOME if available.
# ==============================================================================

set -e

# Get script directory (resolve symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
RUNTIME_PYTHON="$SCRIPT_DIR/runtime/python/python"
VENV_PYTHON="$SCRIPT_DIR/.venv311/bin/python"
CDDS_DIR="$SCRIPT_DIR/runtime/cyclonedds"
INSTALL_STATE="$SCRIPT_DIR/runtime/env/install_state.json"

# ------------------------------------------------------------------------------
# Select Python executable
# ------------------------------------------------------------------------------
if [ -x "$RUNTIME_PYTHON" ]; then
    PYTHON_EXE="$RUNTIME_PYTHON"
    LAUNCH_MODE="packaged"
elif [ -x "$VENV_PYTHON" ]; then
    PYTHON_EXE="$VENV_PYTHON"
    LAUNCH_MODE="venv311"
else
    echo "[ERROR] No project-local Python found."
    echo "[ERROR] Checked: runtime/python/python"
    echo "[ERROR] Checked: .venv311/bin/python"
    echo "[ERROR] Run install.sh first."
    exit 1
fi

# ------------------------------------------------------------------------------
# Inject project-local CycloneDDS (best-effort, non-blocking)
# ------------------------------------------------------------------------------
if [ -d "$CDDS_DIR/lib" ] || [ -d "$CDDS_DIR/bin" ]; then
    export CYCLONEDDS_HOME="$CDDS_DIR"
    
    # Add library path based on architecture
    if [ -d "$CDDS_DIR/lib" ]; then
        export LD_LIBRARY_PATH="$CDDS_DIR/lib:$LD_LIBRARY_PATH"
    fi
    
    # Add bin path
    if [ -d "$CDDS_DIR/bin" ]; then
        export PATH="$CDDS_DIR/bin:$PATH"
    fi
    
    echo "[runtime:cyclonedds] CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
else
    echo "[runtime:cyclonedds] CycloneDDS not configured — Unitree SDK may be unavailable."
fi

# ------------------------------------------------------------------------------
# Detect display for Qt
# ------------------------------------------------------------------------------
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    # No display available, use offscreen platform
    export QT_QPA_PLATFORM="offscreen"
    echo "[runtime:display] No display detected, using offscreen mode"
elif [ -n "$WAYLAND_DISPLAY" ]; then
    export QT_QPA_PLATFORM="wayland"
    echo "[runtime:display] Using Wayland"
else
    export QT_QPA_PLATFORM="xcb"
    echo "[runtime:display] Using X11"
fi

# ------------------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------------------
echo "[start] UnitPort - mode: $LAUNCH_MODE"
echo "[start] Python: $PYTHON_EXE"
echo "[start] Platform: Linux"

# Pass all arguments to main.py
exec "$PYTHON_EXE" main.py "$@"
