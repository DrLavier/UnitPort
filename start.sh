#!/bin/bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# UnitPort RELEASE start.sh - single entry point
#
# 1) Verify .venv311 exists and Python boots
# 2) Verify the BOOTSTRAP imports (PyQt6 + loguru + unitport_sdk) so
#    main.py can paint the LoadingScreen. Everything else (httpx,
#    paramiko, mujoco, ...) is verified inside the loading screen by
#    DepCheckTask, which streams pip install progress to the log widget.
# 3) Inject project-local CycloneDDS (best-effort)
# 4) Launch main.py
# 5) On non-zero exit, run install.sh once and retry the launch
#    (defensive net for crashes that occur before DepCheckTask ran)
#
# At any failed gate, install.sh is invoked transparently.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# ASCII logo (120 chars wide). Inlined for self-contained launch.
# ------------------------------------------------------------------------------
print_ascii() {
    cat << 'ASCII_EOF'
========================================================================================================================
------------------------------------------------------------------------------------------------------------------------
         ####
     ####   ####                                          ####
 #####   ############      #####     #####                #####  #####   ############                            ####
##    ##########    ##     #####     #####                       #####   #############                          #####
##  #########       ##     #####     #####  ###########   ##### ######## #####    #####   #########   ####### #########
##  ######     #### ##     #####     #####  ############  ##### ######## #####    ##### ############  ####### #########
##  ######   ###### ##     #####     #####  ####    ####  #####  #####   #############  ####    ##### #####     #####
##  ######   ###### ##     #####     #####  ####    ####  #####  #####   ###########    ####    ##### #####     #####
##    ####   ###    ##     ######   ######  ####    ####  #####  #####   #####          #####   ##### #####     #####
##                ###       #############   ####    ####  #####  ####### #####           ###########  #####     #######
  #####      ######           ########      ####    ####  #####   ###### #####             #######    #####       #####
     #####   ###
         ####
------------------------------------------------------------------------------------------------------------------------
========================================================================================================================
ASCII_EOF
}
print_ascii
# Flag install.sh so it doesn't repaint the ASCII when start.sh falls through to it.
export UNITPORT_ASCII_PRINTED=1

VENV_PYTHON="$SCRIPT_DIR/.venv311/bin/python"
CDDS_DIR="$SCRIPT_DIR/runtime/cyclonedds"

# ------------------------------------------------------------------------------
# Gate 1 + 2: .venv311 present and Python boots
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

# ------------------------------------------------------------------------------
# Gate 3: BOOTSTRAP import check (PyQt6 + loguru + unitport_sdk only)
# ------------------------------------------------------------------------------
# These three are required to paint the LoadingScreen. If any is missing
# we cannot show the loading screen at all, so we fall back to the
# console-mode install.sh. Every other requirements.txt package is verified
# by DepCheckTask AFTER the loading screen is up, so the user sees pip
# install progress streamed live.

bootstrap_check() {
    "$VENV_PYTHON" -c "import sys; sys.path.insert(0, r'$SCRIPT_DIR/src'); import PyQt6, loguru, unitport_sdk" 2>/dev/null
}

if ! bootstrap_check; then
    echo "[start] bootstrap imports missing (PyQt6 / loguru / unitport_sdk), running install.sh ..."
    bash "$SCRIPT_DIR/install.sh"
    if [ $? -ne 0 ]; then
        echo "[start] install.sh failed, cannot continue."
        exit 1
    fi
    if ! bootstrap_check; then
        echo "[start] bootstrap imports still missing after install.sh; aborting."
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
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    export QT_QPA_PLATFORM="offscreen"
    echo "[runtime:display] No display detected, using offscreen mode"
elif [ -n "${WAYLAND_DISPLAY:-}" ]; then
    export QT_QPA_PLATFORM="wayland"
    echo "[runtime:display] Using Wayland"
else
    export QT_QPA_PLATFORM="xcb"
    echo "[runtime:display] Using X11"
fi

# ------------------------------------------------------------------------------
# Launch (with one defensive install+retry on non-zero exit)
# ------------------------------------------------------------------------------
echo "[start] UnitPort RELEASE - mode: $LAUNCH_MODE"
echo "[start] Python: $PYTHON_EXE"
echo "[start] Platform: $(uname -s)"

launch_retried=0

while true; do
    "$PYTHON_EXE" main.py "$@"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        exit 0
    fi

    if [ "$launch_retried" -eq 0 ]; then
        echo ""
        echo "[start] main.py exited with code $rc."
        echo "[start] Re-running install.sh to verify environment, then retrying launch ..."
        launch_retried=1
        bash "$SCRIPT_DIR/install.sh"
        if [ $? -ne 0 ]; then
            echo "[start] install.sh failed during recovery; surfacing original exit code $rc."
            exit "$rc"
        fi
        continue
    fi

    echo "[start] main.py exited with code $rc again after install recovery; aborting."
    exit "$rc"
done
