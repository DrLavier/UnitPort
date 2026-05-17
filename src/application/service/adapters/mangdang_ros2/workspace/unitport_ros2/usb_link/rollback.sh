#!/usr/bin/env bash
# UnitPort USB link uninstaller — thin wrapper around install.sh with
# UNITPORT_USB_ROLLBACK=1 so both paths share a single source of truth.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec env UNITPORT_USB_ROLLBACK=1 bash "${SCRIPT_DIR}/install.sh" "$@"
