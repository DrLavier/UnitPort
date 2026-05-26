# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Background tasks for the multi-version Isaac Lab installation registry.

Two ``TasksManager``-compatible tasks back the User-panel "Engines → Isaac Lab"
management dialog so the UI thread never blocks on a slow ``import isaaclab``
subprocess or a multi-GB ``rmtree``:

* :class:`IsaacInstallProbeTask` — re-probe every registered install's version /
  availability (one subprocess per root, each up to 30 s). Submitted when the
  management dialog opens and after a new root is registered.
* :class:`IsaacUninstallTask` — physically delete the project-owned *base*
  install (``Engines/isaac_lab/``) from disk, then drop it from the registry.
  External pins are *not* deleted on disk — those go through
  ``backends.unregister_isaac_installation`` directly (unbind only), no task.

Both translate registry / filesystem failures into Task semantics (raise →
``task_finished(ok=False)``) rather than swallowing them (CLAUDE.md §8).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from unitport_sdk import Task

from registers import backends


class IsaacInstallProbeTask(Task):
    """Re-probe every registered Isaac Lab install and persist cached version.

    Result dict: ``{"ok": True, "installations": [...]}`` where the list is the
    decorated output of :func:`registers.backends.list_isaac_installations`.
    """

    def __init__(self, *, name: str = "isaac_install_probe") -> None:
        super().__init__(name=name)

    def run(self) -> Any:  # type: ignore[override]
        self.log_info("probing registered Isaac Lab installations")
        items = backends.refresh_isaac_installations()
        self.log_success(f"probed {len(items)} Isaac Lab install(s)")
        return {"ok": True, "installations": items}


class IsaacUninstallTask(Task):
    """Delete the base Isaac Lab install tree, then unregister it.

    Only ever invoked for the project-owned base root (``Engines/isaac_lab/``);
    the dialog routes external roots to a registry-only unbind. ``rmtree``
    failures propagate (rule §8 — no silent "warn and continue").

    Result dict: ``{"ok": True, "root": <str>, "removed": <bool>}``. ``removed``
    is False when the directory was already gone (we still drop the stale
    registry entry so the UI reflects reality).
    """

    def __init__(self, *, root: str, name: str = "isaac_uninstall") -> None:
        super().__init__(name=name)
        self._root = Path(str(root)).expanduser()

    def run(self) -> Any:  # type: ignore[override]
        root = self._root
        if not backends.is_base_isaac_root(str(root)):
            # Defensive: the dialog must never route an external root here, but
            # if it does, refuse to rmtree someone else's directory.
            raise ValueError(
                f"refusing to uninstall non-base Isaac Lab root {root!r} "
                f"(external installs are unbound, not deleted)"
            )
        removed = False
        if root.exists():
            self.log_info(f"removing Isaac Lab install tree at {root}")
            self.check_cancelled()
            shutil.rmtree(root)
            removed = True
            self.log_success(f"removed Isaac Lab install tree at {root}")
        else:
            self.log_warning(
                f"Isaac Lab base root {root} already gone — dropping stale "
                f"registry entry only"
            )
        backends.unregister_isaac_installation(str(root))
        backends.refresh_engine_availability()
        return {"ok": True, "root": str(root), "removed": removed}


__all__ = ["IsaacInstallProbeTask", "IsaacUninstallTask"]
