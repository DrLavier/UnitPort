#!/bin/bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# UnitPort RELEASE - Factory Reset
#
# Restores the project to its initial (pre-install) state.
# The next "./start.sh" will trigger a full install.
#
# Usage: chmod +x reset.sh && ./reset.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ============================================================"
echo "        UnitPort RELEASE - Factory Reset"
echo "  ============================================================"
echo ""
echo "  This will restore the project to its initial (pre-install)"
echo "  state.  The next './start.sh' will trigger a full install."
echo ""
echo "  WILL BE DELETED:"
echo ""
echo "    .venv311/                         Python virtual environment"
echo "    runtime/env/                      Install state"
echo "    custom_mods/motions/loco-mujoco/  Cloned reference motion library"
echo "    log/                              Log files"
echo "    __pycache__ / *.pyc               Python cache (everywhere)"
echo "    .pip-tmp/ / .pip-cache/           pip scratch + cache"
echo "    .pytest_cache/ / .cache/          Test/build caches"
echo ""
echo "  WILL BE NORMALIZED (back to factory state):"
echo ""
echo "    src/config/system.ini             Drop [Workspace] / [Session],"
echo "                                      blank [Resources] values,"
echo "                                      reset [Localisation] lang = EN"
echo ""
echo "  NOT TOUCHED:"
echo ""
echo "    src/                              Source tree (other than system.ini)"
echo "    custom_mods/nodes/                User-authored nodes"
echo "    projects/                         User project data"
echo "    runtime/factory_build/            Factory-built USD body dumps etc."
echo "                                      (expensive to rebuild; survives reset)"
echo "    \$HOME/UnitPort/                   Per-user state (user.ini, tokens, etc.)"
echo ""

read -r -p "  Proceed with factory reset? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "  Aborted."
    exit 0
fi

echo ""

# -- helpers --
del_file() {
    if [ -f "$SCRIPT_DIR/$1" ]; then
        rm -f "$SCRIPT_DIR/$1"
        echo "  [DEL]  $1"
    else
        echo "  [OK]   $1  (not present)"
    fi
}

del_dir() {
    if [ -d "$SCRIPT_DIR/$1" ]; then
        rm -rf "$SCRIPT_DIR/$1"
        echo "  [DEL]  $1/"
    else
        echo "  [OK]   $1/  (not present)"
    fi
}

# [1/6] system.ini normalize (pure stdlib, ok to run before venv removal)
echo "  [1/6] Normalizing src/config/system.ini to factory state ..."
PY_EXE=""
if [ -x "$SCRIPT_DIR/.venv311/bin/python" ]; then
    PY_EXE="$SCRIPT_DIR/.venv311/bin/python"
elif [ -x "$SCRIPT_DIR/.venv311/Scripts/python.exe" ]; then
    PY_EXE="$SCRIPT_DIR/.venv311/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PY_EXE="python3"
elif command -v python >/dev/null 2>&1; then
    PY_EXE="python"
fi
if [ -n "$PY_EXE" ]; then
    "$PY_EXE" "$SCRIPT_DIR/bootstrap/reset_system_ini.py" || true
else
    echo "  [SKIP] no Python interpreter found, system.ini left as-is"
fi

# [2/6] Virtual environment
echo "  [2/6] Removing virtual environment ..."
del_dir ".venv311"

# [3/6] Runtime install state
echo "  [3/6] Removing runtime install state ..."
del_dir "runtime/env"

# [4/6] Cloned reference libraries
echo "  [4/6] Removing cloned reference libraries ..."
del_dir "custom_mods/motions/loco-mujoco"

# [5/6] Cache and logs
echo "  [5/6] Clearing cache and logs ..."
del_dir "log"
del_dir ".pip-tmp"
del_dir ".pip-cache"
del_dir ".pytest_cache"
del_dir ".cache"
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  [DEL]  __pycache__ (all)"
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# [6/6] Per-user state hint
echo "  [6/6] Hint: per-user state (user.ini, auth tokens, telemetry caches) lives"
echo "        under \$HOME/UnitPort/ and is NOT touched by reset."
echo "        Delete it manually if you want a truly clean first-run experience."

echo ""
echo "  ============================================================"
echo "  Factory reset complete."
echo ""
echo "  Run './install.sh' then './start.sh' to begin fresh setup,"
echo "  or just run './start.sh' -- it auto-installs on a missing venv."
echo "  ============================================================"
echo ""
