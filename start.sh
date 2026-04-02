#!/bin/bash
# ==============================================================================
# UnitPort start.sh - single entry point
# Checks .venv311; if missing or broken, runs install.sh first.
# Then launches the application.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv311/bin/python"
CDDS_DIR="$SCRIPT_DIR/runtime/cyclonedds"

# ------------------------------------------------------------------------------
# Auto-install if .venv311 is missing or broken
# ------------------------------------------------------------------------------

need_install=0

if [ ! -x "$VENV_PYTHON" ]; then
    echo "[start] .venv311 not found, running install.sh ..."
    need_install=1
elif ! "$VENV_PYTHON" -c "import sys; sys.exit(0)" 2>/dev/null; then
    echo "[start] .venv311 Python broken, running install.sh ..."
    need_install=1
fi

if [ "$need_install" -eq 1 ]; then
    bash "$SCRIPT_DIR/install.sh"
    if [ $? -ne 0 ]; then
        echo "[start] install.sh failed, cannot continue."
        exit 1
    fi
    if [ ! -x "$VENV_PYTHON" ]; then
        echo "[start] install.sh finished but .venv311 still missing, cannot continue."
        exit 1
    fi
fi

# Quick sanity: can we import PySide6?
if ! "$VENV_PYTHON" -c "import PySide6" 2>/dev/null; then
    echo "[start] PySide6 missing, re-running install.sh ..."
    bash "$SCRIPT_DIR/install.sh"
    if [ $? -ne 0 ]; then
        echo "[start] install.sh failed, cannot continue."
        exit 1
    fi
fi

PYTHON_EXE="$VENV_PYTHON"
LAUNCH_MODE="venv311"

# ------------------------------------------------------------------------------
# Inject project-local CycloneDDS (best-effort, non-blocking)
# ------------------------------------------------------------------------------
if [ -d "$CDDS_DIR/lib" ] || [ -d "$CDDS_DIR/bin" ]; then
    export CYCLONEDDS_HOME="$CDDS_DIR"
    [ -d "$CDDS_DIR/lib" ] && export LD_LIBRARY_PATH="$CDDS_DIR/lib:${LD_LIBRARY_PATH:-}"
    [ -d "$CDDS_DIR/bin" ] && export PATH="$CDDS_DIR/bin:$PATH"
    echo "[runtime:cyclonedds] CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
else
    echo "[runtime:cyclonedds] CycloneDDS not configured, Unitree SDK may be unavailable."
fi

# ------------------------------------------------------------------------------
# Detect display for Qt
# ------------------------------------------------------------------------------
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
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
echo "[start] Platform: $(uname -s)"

exec "$PYTHON_EXE" main.py "$@"
