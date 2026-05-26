#!/bin/bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# UnitPort install.sh
# Strategy: project-local .venv311 (Python 3.11).
#
# Creates a project-local virtual environment at .venv311/
# and installs all dependencies into it.
# Never installs packages into global Python.
#
# Usage: ./install.sh
# ==============================================================================

set -e  # Exit on error

# Get script directory (resolve symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# ASCII logo (always) + LICENSE panel (first install only).
# First-install gate: $SCRIPT_DIR/runtime/env/install_state.json absence.
# reset.sh / reset.bat clears that file -> factory reset re-prompts the license.
#
# UNITPORT_ASCII_PRINTED=1 means our caller (start.sh) already painted the
# logo, so we skip it to avoid the double-print. Direct runs of install.sh
# still see the logo because the var is not set.
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
##  ######    ##### ##     #####     #####  ####    ####  #####  #####   #############  ####    ##### #####     #####
##  ######    ##### ##     #####     #####  ####    ####  #####  #####   ###########    ####    ##### #####     #####
##    ####    ##    ##     ######   ######  ####    ####  #####  #####   #####          #####   ##### #####     #####
###               ###       #############   ####    ####  #####  ####### #####           ###########  #####     #######
  #####      ######           ########      ####    ####  #####   ###### #####             #######    #####       #####
     #####   ###
         ####
------------------------------------------------------------------------------------------------------------------------
========================================================================================================================
ASCII_EOF
}

print_license() {
    cat << 'LICENSE_EOF'

========================================================================================================================
 | UnitPort Studio  -  License and Pre-install Notice                                                                  |
 |                                                                                                                     |
 | [LICENSE - Key Terms]                                                                                               |
 |   1. Data Security  : User data is processed and stored locally by default.                                         |
 |                       Cloud storage is hosted on Supabase, optional, and does not block normal usage.               |
 |   2. No Repackaging : Redistributing or commercially reselling this software (or derivatives) is prohibited.        |
 |   3. Copyright      : Protected under the EU Copyright Directive 2019/790. Violations will be prosecuted.           |
 |                                                                                                                     |
 | [Installation Notice]                                                                                               |
 |   First-time install may take 15 - 45 minutes, depending on selected components and network conditions.             |
 |   Components include Isaac Lab, loco-mujoco, and vendor SDKs.                                                       |
 |   Please keep the network stable and avoid power loss during installation.                                          |
========================================================================================================================
LICENSE_EOF
}

if [ -z "${UNITPORT_ASCII_PRINTED:-}" ]; then
    print_ascii
fi

if [ ! -f "$SCRIPT_DIR/runtime/env/install_state.json" ]; then
    print_license
    echo ""
    while true; do
        # set -e is on; tolerate non-zero `read` (EOF, Ctrl-D) without dying.
        LICENSE_ANSWER=""
        read -r -p "Have you read and agreed to the terms above? [Y/N]: " LICENSE_ANSWER || true
        case "$LICENSE_ANSWER" in
            [Yy]) echo ""; break ;;
            [Nn]) echo ""; echo "[install] Cancelled. Run install.sh again to review the terms."; exit 0 ;;
            *) echo "  Please enter Y or N."; echo "" ;;
        esac
    done
fi

echo "[install] UnitPort RELEASE environment setup"
echo "[install] Project root: $SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv311"
VENV_PYTHON="$VENV_DIR/bin/python"
WHEELS_DIR="$SCRIPT_DIR/runtime/wheels"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
ENV_DIR="$SCRIPT_DIR/runtime/env"
INSTALL_STATE="$ENV_DIR/install_state.json"

# ------------------------------------------------------------------------------
# Redirect pip temp + cache onto the project drive.
# pip downloads GB-scale wheels (torch/CUDA) through $TMPDIR; on systems with
# a small root partition this fills / and aborts install with ENOSPC. Keep
# everything on the same volume as the project.
# ------------------------------------------------------------------------------
PIP_TMP_DIR="$SCRIPT_DIR/.pip-tmp"
PIP_CACHE="$SCRIPT_DIR/.pip-cache"
mkdir -p "$PIP_TMP_DIR" "$PIP_CACHE"
export TMPDIR="$PIP_TMP_DIR"
export PIP_CACHE_DIR="$PIP_CACHE"
echo "[install] pip tmp     : $PIP_TMP_DIR"
echo "[install] pip cache   : $PIP_CACHE"

# ------------------------------------------------------------------------------
# Step 1: Resolve Python 3.11 for env creation
# ------------------------------------------------------------------------------
# Strategy: try every plausible candidate (command names + common paths) and
# keep the first one that reports version 3.11. Never exit early just because
# one candidate is the wrong version -- the next candidate may be correct.
BASE_PYTHON=""
INSTALL_MODE="venv311"

is_py311() {
    local candidate="$1"
    [ -n "$candidate" ] || return 1
    # command -v handles both commands and absolute paths
    command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ] || return 1
    local ver
    ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
    [ "$ver" = "3.11" ]
}

# 1) Command names in PATH (most common case)
for cmd in python3.11 python3 python; do
    if is_py311 "$cmd"; then
        BASE_PYTHON="$cmd"
        echo "[install] Mode: project-local .venv311 via $cmd in PATH"
        break
    fi
done

# 2) Common install paths (Linux distros, Homebrew, pyenv, etc.)
if [ -z "$BASE_PYTHON" ]; then
    CANDIDATE_PATHS=(
        /usr/bin/python3.11
        /usr/local/bin/python3.11
        /opt/python3.11/bin/python3.11
        /opt/homebrew/bin/python3.11
        /opt/homebrew/opt/python@3.11/bin/python3.11
        /usr/local/opt/python@3.11/bin/python3.11
        /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
    )
    # pyenv: expand glob for any 3.11.x install
    if [ -d "$HOME/.pyenv/versions" ]; then
        for pyenv_py in "$HOME/.pyenv/versions"/3.11*/bin/python3.11; do
            CANDIDATE_PATHS+=("$pyenv_py")
        done
    fi
    for path in "${CANDIDATE_PATHS[@]}"; do
        if [ -x "$path" ] && is_py311 "$path"; then
            BASE_PYTHON="$path"
            echo "[install] Mode: project-local .venv311 via $path"
            break
        fi
    done
fi

# 3) Conda environments (base + envs) - cross-platform
if [ -z "$BASE_PYTHON" ]; then
    CONDA_ROOTS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/miniforge3"
        "$HOME/mambaforge"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/opt/homebrew/Caskroom/miniconda/base"
        "/opt/homebrew/anaconda3"
    )
    CONDA_PY=""
    CONDA_NAME=""
    for root in "${CONDA_ROOTS[@]}"; do
        [ -d "$root" ] || continue
        # Base env
        if [ -z "$CONDA_PY" ] && [ -x "$root/bin/python" ] && is_py311 "$root/bin/python"; then
            CONDA_PY="$root/bin/python"
            CONDA_NAME="$(basename "$root")-base"
        fi
        # Sub-envs
        if [ -z "$CONDA_PY" ] && [ -d "$root/envs" ]; then
            for env_dir in "$root/envs"/*; do
                [ -d "$env_dir" ] || continue
                if [ -x "$env_dir/bin/python" ] && is_py311 "$env_dir/bin/python"; then
                    CONDA_PY="$env_dir/bin/python"
                    CONDA_NAME="$(basename "$env_dir")"
                    break
                fi
            done
        fi
        [ -n "$CONDA_PY" ] && break
    done

    if [ -n "$CONDA_PY" ]; then
        echo ""
        echo "[install] Found conda environment \"$CONDA_NAME\" with Python 3.11:"
        echo "[install]   $CONDA_PY"
        # read must not fail set -e on empty input
        ans=""
        read -r -p "Use this conda environment to create .venv311? [y/N]: " ans || true
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            BASE_PYTHON="$CONDA_PY"
            echo "[install] Mode: project-local .venv311 via conda env \"$CONDA_NAME\""
        else
            echo "[install] Skipping conda environment."
        fi
    fi
fi

if [ -z "$BASE_PYTHON" ]; then
    echo "[ERROR] Python 3.11 not found."
    echo "[ERROR] UnitPort requires Python 3.11 specifically (not 3.10, 3.12, or 3.13)."
    echo "[ERROR]"
    echo "[ERROR] Install Python 3.11:"
    echo "[ERROR]   Ubuntu/Debian: sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo "[ERROR]   Fedora/RHEL:   sudo dnf install python3.11 python3.11-devel"
    echo "[ERROR]   Arch:          sudo pacman -S python311"
    echo "[ERROR]   macOS:         brew install python@3.11"
    echo "[ERROR]   Any platform:  pyenv install 3.11 && pyenv local 3.11"
    echo "[ERROR]"
    echo "[ERROR] Then rerun ./install.sh"
    exit 1
fi

# Get Python version
PY_VER=$($BASE_PYTHON --version 2>&1 | awk '{print $2}')
echo "[install] Base Python: $PY_VER"

# ------------------------------------------------------------------------------
# Step 2: Create or verify .venv311
# ------------------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    if [ -x "$VENV_PYTHON" ]; then
        VENV_VER=$($VENV_PYTHON --version 2>&1 | awk '{print $2}')
        echo "[install] Existing .venv311: $VENV_VER"
    fi
else
    echo "[install] Creating .venv311 ..."
    $BASE_PYTHON -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create .venv311"
        exit 1
    fi
    echo "[install] .venv311 created."
fi
TARGET_PYTHON="$VENV_PYTHON"

# Verify target Python exists
if [ ! -x "$TARGET_PYTHON" ]; then
    echo "[ERROR] Target Python not found: $TARGET_PYTHON"
    exit 1
fi

# ------------------------------------------------------------------------------
# Step 3: Upgrade pip
# ------------------------------------------------------------------------------
echo "[install] Upgrading pip ..."
$TARGET_PYTHON -m pip install --upgrade pip setuptools wheel --quiet 2>/dev/null || {
    echo "[WARNING] pip upgrade failed. Continuing with existing version."
}

# ------------------------------------------------------------------------------
# Step 4: bootstrap install -- minimum to launch MainWindow
# ------------------------------------------------------------------------------
# All heavy provisioning (torch CUDA, requirements.txt, loco-mujoco, URL scheme,
# ROS2) is deferred to the in-app ProvisioningTask which streams pip stdout
# into the LoadingScreen. Bootstrap only needs the minimum to draw a Qt window
# and route logs:
#   PyQt6      -- Qt platform
#   loguru     -- log sink consumed by unitport_sdk
#   cyclonedds -- hard requirement of idl_messages/builtin_interfaces.py (class
#                 base IdlStruct); MainWindow's widget chain pulls it in eagerly
#                 via adapters -> ros2 native bridge. Without it the
#                 LoadingScreen itself cannot paint.
#
# --only-binary :all: on cyclonedds is non-negotiable: it has no source-build
# path that survives without the CycloneDDS C library + a C++17 toolchain. The
# flag makes pip refuse to build, so pip selects the newest matching wheel --
# or fails fast with "No matching distribution", which we surface as a hard
# error here so the user can install a compatible Python.

echo "[install] Installing bootstrap packages (PyQt6, loguru) ..."
$TARGET_PYTHON -m pip install --disable-pip-version-check PyQt6 loguru
if [ $? -ne 0 ]; then
    echo "[ERROR] Bootstrap install failed."
    exit 1
fi

echo "[install] Installing cyclonedds (--only-binary :all:; MainWindow import gate) ..."
$TARGET_PYTHON -m pip install --disable-pip-version-check --only-binary :all: --upgrade "cyclonedds>=0.10.2"
if [ $? -ne 0 ]; then
    echo "[ERROR] cyclonedds install failed."
    echo "[ERROR] pip could not find a binary wheel for cyclonedds on this Python."
    echo "[ERROR] MainWindow cannot import without it -- the app will not start."
    echo "[ERROR] Verify you are on Python 3.11 ($($TARGET_PYTHON --version)),"
    echo "[ERROR] or check https://pypi.org/project/cyclonedds/#files for a matching wheel."
    exit 1
fi

# ------------------------------------------------------------------------------
# Step 5: Write minimal install_state.json
# ------------------------------------------------------------------------------
# ProvisioningTask flips ``provisioning_pending`` to false and adds
# torch_cuda / loco_mujoco / url_scheme facts after it runs.

mkdir -p "$ENV_DIR"
INSTALL_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$INSTALL_STATE" << EOF
{
  "bootstrap": true,
  "provisioning_pending": true,
  "install_timestamp": "$INSTALL_TIMESTAMP",
  "install_mode": "$INSTALL_MODE",
  "python_version": "$PY_VER",
  "notes": "Written by RELEASE/install.sh (bootstrap stage)"
}
EOF
echo "[install] install_state.json written."

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "[install] Bootstrap complete. Heavy dependencies will install on first launch via LoadingScreen."
echo "[install] Mode   : $INSTALL_MODE"
echo "[install] Python : $TARGET_PYTHON"
echo "[install] Launch : ./start.sh"
