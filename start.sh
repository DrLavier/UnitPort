#!/bin/bash
# ==============================================================================
# UnitPort start.sh
# Launches UnitPort using ONLY the project-local .venv311 environment.
# Never falls back to runtime/python or any global Python.
# Always injects project-local CYCLONEDDS_HOME if available.
# ==============================================================================

set -e

# Get script directory (resolve symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
VENV_PYTHON="$SCRIPT_DIR/.venv311/bin/python"
CDDS_DIR="$SCRIPT_DIR/runtime/cyclonedds"
INSTALL_STATE="$SCRIPT_DIR/runtime/env/install_state.json"

# ------------------------------------------------------------------------------
# Select Python executable
# ------------------------------------------------------------------------------
if [ -x "$VENV_PYTHON" ]; then
    PYTHON_EXE="$VENV_PYTHON"
    LAUNCH_MODE="venv311"
else
    echo "[ERROR] .venv311 Python not found."
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
