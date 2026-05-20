"""application.ui.wizard -- the first-launch InstallConfigWizard.

Modal QDialog shown ON TOP of the LoadingScreen on first launch (or
when ``setup_state.json`` is missing / ``completed=false``). Four
pages collect the user's choice for:

1. :class:`DataDirectoryPage` -- where USER_CONFIG_DIR points (default
   ``PROJECT_ROOT/runtime/user_config``; custom paths persist via
   ``~/.unitport_paths.ini`` bootstrap shim). Applied BEFORE the rest of
   the wizard's writes so ``setup_state.json`` and everything downstream
   lands at the chosen location.
2. :class:`MenagerieSelectPage` -- which mujoco_menagerie packages to
   sparse-checkout.
3. :class:`SdkSelectPage` -- which brand SDK repos to clone.
4. :class:`BackendPage` -- MuJoCo / Loco-MuJoCo / ROS2 / Isaac Lab
   (locate / install / cloud-deploy) configuration.

The wizard is non-blocking (``open()`` not ``exec()``) so the
``ProvisioningTask`` running on a TasksManager worker can continue
installing requirements.txt in parallel. When the user finishes / skips,
the wizard emits ``completed: pyqtSignal(dict)`` carrying the
selections; ``UnitPortMain._maybe_submit_postsetup`` gates submission
of the heavy ``PostSetupTask`` on both the wizard AND provisioning
being done.

Public surface:

    from application.ui.wizard import (
        InstallConfigWizard,
        load_setup_state,
        save_setup_state,
        setup_completed,
    )
"""

from .install_config_wizard import (
    InstallConfigWizard,
    load_setup_state,
    save_setup_state,
    setup_completed,
)

__all__ = [
    "InstallConfigWizard",
    "load_setup_state",
    "save_setup_state",
    "setup_completed",
]
