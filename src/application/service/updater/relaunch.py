# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Relaunch helper for the "Update and restart" flow.

After a successful in-app update, spawn a fresh UnitPort process via the
project-root launcher (``start.bat`` on Windows, ``start.sh`` elsewhere)
detached from the current process, so the caller can immediately quit and
the new process boots on the updated code.

``main.py`` self-re-execs under ``.venv311``, so launching through the
start script (rather than re-spawning the current interpreter) guarantees
the relaunched process uses the same bootstrap path a normal launch would.
"""

from __future__ import annotations

import os
import subprocess
import sys

from unitport_sdk import Paths, log_info, log_warning


def relaunch_app() -> bool:
    """Spawn a detached new UnitPort process. Returns True on spawn.

    The caller is responsible for quitting the current process afterwards
    (e.g. ``QApplication.quit()``); this function only starts the successor.
    """
    root = Paths.PROJECT_ROOT
    # WHY KEPT (§8 b): documented cross-platform branch — Windows uses the
    # .bat launcher, every other OS uses the .sh launcher.
    if os.name == "nt":
        script = root / "start.bat"
        args = ["cmd.exe", "/c", "start", "", str(script)]
        creationflags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
        popen_kw = {"creationflags": creationflags, "close_fds": True}
    else:
        script = root / "start.sh"
        args = ["/bin/sh", str(script)]
        popen_kw = {"start_new_session": True, "close_fds": True}

    if not script.exists():
        log_warning(
            f"[updater] relaunch script not found ({script}); the app will "
            f"close without relaunching — please start UnitPort manually"
        )
        return False

    try:
        subprocess.Popen(args, cwd=str(root), **popen_kw)  # noqa: S603
        log_info(f"[updater] relaunch spawned via {script.name}")
        return True
    except Exception as exc:  # noqa: BLE001 — defensive spawn
        log_warning(
            f"[updater] relaunch spawn failed: {exc}; please start "
            f"UnitPort manually"
        )
        return False


__all__ = ["relaunch_app"]
