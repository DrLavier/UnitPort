"""Heavy-backend installers (currently Isaac Lab only).

Owns the in-app installation of backends whose footprint and licence
machinery make them unfit for either:

* the project ``.venv311`` (Isaac Sim ~30 GB + GPU drivers + EULAs); or
* the existing ``application.service.models.sdk_manager`` package
  (brand SDKs cloned under ``custom_mods/runtime/sdk_extensions/`` —
  MB-scale, no EULA).

Public surface deliberately minimal — most consumers should reach the
installer via :class:`application.tools.isaac_lab_install_task.IsaacLabInstallTask`
rather than touching :class:`IsaacLabInstaller` directly. The Task wrapper
provides cancellation propagation and ties into the SDK ``TasksManager``
logging contract.

EULA acceptance lives outside this package's source — under
``Paths.USER_CONFIG_DIR / "eula_acceptance.json"`` — so it survives a
clean ``git pull`` per the project's user-state-never-in-repo rule
(``CLAUDE.md`` §1.4).
"""

from __future__ import annotations

from ._paths import (
    default_isaac_lab_install_root,
    engines_root,
    is_install_path_internal,
)
from .isaac_lab_installer import (
    InstallerCancelled,
    InstallerCloneError,
    InstallerError,
    InstallerNetworkError,
    InstallerPreflightError,
    InstallerSubprocessError,
    InstallPlan,
    InstallReport,
    IsaacLabInstaller,
)
from .custom_mods_manifest import (
    CustomModEntry,
    entry_already_installed,
    load_manifest_entries,
    manifest_path,
)

__all__ = [
    "default_isaac_lab_install_root",
    "engines_root",
    "is_install_path_internal",
    "IsaacLabInstaller",
    "InstallPlan",
    "InstallReport",
    "InstallerError",
    "InstallerCancelled",
    "InstallerCloneError",
    "InstallerNetworkError",
    "InstallerPreflightError",
    "InstallerSubprocessError",
    "CustomModEntry",
    "load_manifest_entries",
    "manifest_path",
    "entry_already_installed",
]
