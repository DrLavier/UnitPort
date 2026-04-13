#!/bin/bash
# ==============================================================================
# UnitPort install.sh
# Strategy: project-local .venv311 (Python 3.11)
#
# Creates a project-local virtual environment at .venv311/
# and installs all dependencies into it.
# Never installs packages into global Python.
# runtime/python is ignored for dependency installation.
#
# Usage: ./install.sh
# ==============================================================================

set -e  # Exit on error

# Get script directory (resolve symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[install] UnitPort environment setup"
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
# Step 4a: Install PyTorch with CUDA (if NVIDIA GPU present)
# ------------------------------------------------------------------------------
# PyPI only hosts CPU-only torch builds.  The CUDA builds live on
# PyTorch's own index.  We detect NVIDIA hardware first and install
# the CUDA wheel so that the plain "torch>=2.0.0" line in
# requirements.txt is already satisfied when Step 4b runs.

if command -v nvidia-smi &> /dev/null; then
    echo "[install] NVIDIA GPU detected -- installing PyTorch with CUDA ..."
    $TARGET_PYTHON -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124 --quiet || {
        echo "[WARNING] CUDA PyTorch install failed; will fall back to CPU version."
    }
    echo "[install] PyTorch CUDA installed."
else
    echo "[install] No NVIDIA GPU detected -- PyTorch CPU will be installed."
fi

# ------------------------------------------------------------------------------
# Step 4b: Install packages
# ------------------------------------------------------------------------------
if [ ! -f "$REQUIREMENTS" ]; then
    echo "[ERROR] requirements.txt not found: $REQUIREMENTS"
    exit 1
fi

# Count available wheels
WHEEL_COUNT=0
if [ -d "$WHEELS_DIR" ]; then
    WHEEL_COUNT=$(find "$WHEELS_DIR" -maxdepth 1 -name "*.whl" 2>/dev/null | wc -l)
fi

if [ "$WHEEL_COUNT" -gt 0 ]; then
    echo "[install] Installing from local wheelhouse ($WHEEL_COUNT wheels) ..."
    $TARGET_PYTHON -m pip install --no-index --find-links="$WHEELS_DIR" -r "$REQUIREMENTS" 2>/dev/null || {
        echo "[install] Offline install incomplete, retrying with network ..."
        $TARGET_PYTHON -m pip install -r "$REQUIREMENTS"
    }
else
    echo "[install] Installing from network (no local wheels found) ..."
    $TARGET_PYTHON -m pip install -r "$REQUIREMENTS"
fi

if [ $? -ne 0 ]; then
    echo "[ERROR] Package install failed"
    exit 1
fi

# ------------------------------------------------------------------------------
# Step 5: Clone loco-mujoco if missing
# ------------------------------------------------------------------------------
LOCO_DIR="$SCRIPT_DIR/custom_mods/motions/loco-mujoco"
if [ ! -d "$LOCO_DIR/.git" ]; then
    echo "[install] Cloning loco-mujoco reference motion library ..."
    git clone --depth=1 --progress "https://github.com/robfiras/loco-mujoco.git" "$LOCO_DIR" || {
        echo "[WARNING] loco-mujoco clone failed (optional, reference motions unavailable)."
    }
else
    echo "[install] loco-mujoco already present."
fi

# ------------------------------------------------------------------------------
# Step 6: Verify critical imports
# ------------------------------------------------------------------------------
echo "[install] Verifying imports ..."

PYSIDE6_OK=$($TARGET_PYTHON -c "import PySide6; print(PySide6.__version__)" 2>/dev/null && echo "ok" || echo "failed")
if [ "$PYSIDE6_OK" != "ok" ]; then
    echo "[WARNING] PySide6 import failed."
else
    echo "[install] PySide6 OK: $PYSIDE6_OK"
fi

TORCH_INFO=$($TARGET_PYTHON -c "import torch; cuda='CUDA '+torch.version.cuda if torch.cuda.is_available() else 'CPU only'; print(torch.__version__, cuda)" 2>/dev/null || echo "import failed")
echo "[install] PyTorch : $TORCH_INFO"

MUJOCO_OK=$($TARGET_PYTHON -c "import mujoco; print(mujoco.__version__)" 2>/dev/null && echo "ok" || echo "failed")
if [ "$MUJOCO_OK" != "ok" ]; then
    echo "[WARNING] MuJoCo import failed (optional)."
else
    echo "[install] MuJoCo OK: $MUJOCO_OK"
fi

CDDS_OK=$($TARGET_PYTHON -c "import cyclonedds; print(cyclonedds.__file__)" 2>/dev/null && echo "ok" || echo "failed")
if [ "$CDDS_OK" != "ok" ]; then
    echo "[WARNING] CycloneDDS import failed. Attempting reinstall ..."
    $TARGET_PYTHON -m pip install --only-binary :all: "cyclonedds>=0.10.2" --quiet 2>/dev/null || true
    CDDS_OK=$($TARGET_PYTHON -c "import cyclonedds; print(cyclonedds.__file__)" 2>/dev/null && echo "ok" || echo "failed")
    if [ "$CDDS_OK" != "ok" ]; then
        echo "[WARNING] CycloneDDS still unavailable. Real-robot communication may not work."
    else
        echo "[install] CycloneDDS OK (retry): $CDDS_OK"
    fi
else
    echo "[install] CycloneDDS OK: $CDDS_OK"
fi

# ------------------------------------------------------------------------------
# Step 7: Write install_state.json
# ------------------------------------------------------------------------------
mkdir -p "$ENV_DIR"

INSTALL_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
WHEELS_BOOL="false"
if [ "$WHEEL_COUNT" -gt 0 ]; then
    WHEELS_BOOL="true"
fi

cat > "$INSTALL_STATE" << EOF
{
  "installed": true,
  "install_timestamp": "$INSTALL_TIMESTAMP",
  "install_mode": "$INSTALL_MODE",
  "python_version": "$PY_VER",
  "runtime_python_verified": false,
  "cyclonedds_verified": false,
  "wheels_installed": $WHEELS_BOOL,
  "notes": "Written by install.sh"
}
EOF
echo "[install] install_state.json written."

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "[install] Installation complete."
echo "[install] Mode   : $INSTALL_MODE"
echo "[install] Python : $TARGET_PYTHON"
echo "[install] Launch : ./start.sh"
