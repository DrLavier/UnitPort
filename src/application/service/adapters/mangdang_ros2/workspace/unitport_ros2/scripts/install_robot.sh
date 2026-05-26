#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# install_robot.sh — backward-compatibility shim. The real installer is
# bootstrap.sh in the same directory; this name is kept so external
# callers (older bridge_upgrade.py revisions, hand-rolled SSH scripts,
# documentation referencing the historical script name) still work.
#
# Forwards all arguments + environment to bootstrap.sh.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/bootstrap.sh" "$@"
