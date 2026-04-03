#!/bin/bash
# ==============================================================================
# UnitPort - Factory Reset
#
# Restores the project to its initial (pre-install) state.
# The next "./start.sh" will trigger a full install + setup wizard.
#
# Usage: chmod +x reset.sh && ./reset.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ============================================================"
echo "        UnitPort - Factory Reset"
echo "  ============================================================"
echo ""
echo "  This will restore the project to its initial (pre-install)"
echo "  state.  The next './start.sh' will trigger a full install +"
echo "  setup wizard."
echo ""
echo "  WILL BE DELETED:"
echo ""
echo "    .venv311/                         Python virtual environment"
echo "    runtime/env/                      Install state"
echo "    src/system/runtime/sdk/*          Downloaded SDKs (git-cloned)"
echo "    src/system/runtime/simulation/    MuJoCo menagerie (submodule)"
echo "    custom_mods/                      All custom content"
echo "    training_workspaces/              Training workspace data"
echo "    logs/                             Log files"
echo "    __pycache__ / *.pyc               Python cache (everywhere)"
echo "    src/config/setup_state.json       Setup wizard state"
echo "    src/config/isaaclab.json          Isaac Lab registration"
echo "    src/config/rewards_presets.json   Reward preset overrides"
echo "    src/system/models/.sdk_install_state.json"
echo ""
echo "  NOT TOUCHED:"
echo ""
echo "    projects/                         User project data"
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

# [1/7] Virtual environment
echo "  [1/7] Removing virtual environment ..."
del_dir ".venv311"

# [2/7] Runtime state
echo "  [2/7] Removing runtime state ..."
del_dir "runtime/env"
del_file "src/system/models/.sdk_install_state.json"

# [3/7] Downloaded SDKs
echo "  [3/7] Removing downloaded SDKs ..."
if [ -d "src/system/runtime/sdk" ]; then
    for d in src/system/runtime/sdk/*/; do
        [ -d "$d" ] || continue
        rm -rf "$d"
        echo "  [DEL]  $d"
    done
fi
# MuJoCo menagerie (submodule)
if [ -d "src/system/runtime/simulation/mujoco/menagerie/.git" ]; then
    git submodule deinit -f src/system/runtime/simulation/mujoco/menagerie 2>/dev/null || true
    echo "  [DEL]  menagerie submodule (deinit)"
fi
del_dir "src/system/runtime/simulation/mujoco/menagerie"

# [4/7] Custom content
echo "  [4/7] Removing custom_mods content ..."
del_dir "custom_mods"
del_dir "training_workspaces"

# [5/7] Config registry
echo "  [5/7] Clearing config registry ..."
del_file "src/config/setup_state.json"
del_file "src/config/isaaclab.json"
del_file "src/config/rewards_presets.json"

# [6/7] Cache and logs
echo "  [6/7] Clearing cache and logs ..."
del_dir "logs"
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  [DEL]  __pycache__ (all)"
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
del_dir ".pytest_cache"
del_dir ".cache"

# [7/7] Restore git-tracked defaults
echo "  [7/7] Restoring git-tracked defaults ..."
git checkout -- src/config/system.ini 2>/dev/null && echo "  [RST]  src/config/system.ini" || echo "  [SKIP] src/config/system.ini (not in git)"
git checkout -- src/config/user.ini 2>/dev/null && echo "  [RST]  src/config/user.ini" || echo "  [SKIP] src/config/user.ini (not in git)"
git checkout -- src/config/ui.ini 2>/dev/null && echo "  [RST]  src/config/ui.ini" || echo "  [SKIP] src/config/ui.ini (not in git)"
git checkout -- src/config/node_registry.json 2>/dev/null && echo "  [RST]  src/config/node_registry.json" || echo "  [SKIP] src/config/node_registry.json (not in git)"
git checkout -- custom_mods/ 2>/dev/null && echo "  [RST]  custom_mods/ (git skeleton)" || echo "  [SKIP] custom_mods/ (not in git)"

echo ""
echo "  ============================================================"
echo "  Factory reset complete."
echo ""
echo "  Run './install.sh' then './start.sh' to begin fresh setup."
echo "  ============================================================"
echo ""
