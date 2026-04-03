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
BASE_PYTHON=""

if command -v python3.11 &> /dev/null; then
    BASE_PYTHON="python3.11"
    INSTALL_MODE="venv311"
    echo "[install] Mode: project-local .venv311 via system Python 3.11"
elif command -v python3 &> /dev/null; then
    # Check if python3 is version 3.11
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$PY_VERSION" = "3.11" ]; then
        BASE_PYTHON="python3"
        INSTALL_MODE="venv311"
        echo "[install] Mode: project-local .venv311 via python3 (3.11)"
    else
        echo "[ERROR] Python 3.11 not found. Current version: $PY_VERSION"
        echo "[ERROR] Install Python 3.11."
        exit 1
    fi
elif command -v python &> /dev/null; then
    PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
    if [ "$PY_VERSION" = "3.11" ]; then
        BASE_PYTHON="python"
        INSTALL_MODE="venv311"
        echo "[install] Mode: project-local .venv311 via python (3.11)"
    else
        echo "[ERROR] Python 3.11 not found. Current version: $PY_VERSION"
        echo "[ERROR] Install Python 3.11."
        exit 1
    fi
else
    echo "[ERROR] Python not found."
    echo "[ERROR] Install Python 3.11."
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
