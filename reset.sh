#!/bin/bash
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
echo "  NOT TOUCHED:"
echo ""
echo "    src/config/system.ini             SDK factory defaults (read-only)"
echo "    src/                              Source tree"
echo "    custom_mods/nodes/                User-authored nodes"
echo "    projects/                         User project data"
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

# [1/5] Virtual environment
echo "  [1/5] Removing virtual environment ..."
del_dir ".venv311"

# [2/5] Runtime install state
echo "  [2/5] Removing runtime install state ..."
del_dir "runtime/env"

# [3/5] Cloned reference libraries
echo "  [3/5] Removing cloned reference libraries ..."
del_dir "custom_mods/motions/loco-mujoco"

# [4/5] Cache and logs
echo "  [4/5] Clearing cache and logs ..."
del_dir "log"
del_dir ".pip-tmp"
del_dir ".pip-cache"
del_dir ".pytest_cache"
del_dir ".cache"
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  [DEL]  __pycache__ (all)"
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# [5/5] Per-user state hint
echo "  [5/5] Hint: per-user state (user.ini, auth tokens, telemetry caches) lives"
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
