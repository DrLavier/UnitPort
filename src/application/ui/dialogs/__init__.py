# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Modal dialogs opened from the sidebar panels and node ParamRow ⚙ buttons.

Lazy-loaded via PEP 562 ``__getattr__``: the dialog submodules in this
package transitively pull heavy startup-time forbidden deps —
``email_identity_dialog`` / ``identity_unlink_dialog`` import
``application.service.auth.get_auth_manager`` (which loads ``httpx`` via
``avatar_cache`` / ``supabase_client``); ``update_*_dialog`` pull the
``application.service.updater`` package (also httpx).

These deps are part of ``requirements.txt`` and are installed by
``ProvisioningTask`` at Stage 2. The InstallConfigWizard, however, is
constructed BEFORE ProvisioningTask runs (the wizard must paint first so
pip output doesn't flood the LoadingScreen pre-dialog), and the wizard's
import chain transitively pulls ``application.ui.dialogs`` via
``mission_control_panel → real_robot_connection_card →
connection_settings_dialog``. With eager imports here, that chain crashes
with ``ModuleNotFoundError: No module named 'httpx'`` on a first launch.

Each name in ``__all__`` is resolved on first attribute access, so
``from application.ui.dialogs.connection_settings_dialog import X``
still loads ``__init__`` (Python loads parent packages first), but
``__init__`` no longer eagerly drags in the auth/updater-dependent
submodules. Anyone doing ``from application.ui.dialogs import
EmailIdentityDialog`` triggers ``__getattr__`` only at that access
point, which on the actual call path happens at Stage 3+ when the user
opens the dialog — by then httpx is on disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-checker visibility without runtime cost.
    from .bar1_risk_dialog import Bar1RiskDialog
    from .canvas_error_dialog import (
        CanvasErrorDialog,
        CanvasIssue,
        clear_canvas_diagnostic_marks,
        issues_from_exception,
        show_canvas_error_dialog,
    )
    from .email_identity_dialog import EmailIdentityDialog
    from .eula_dialog import EulaDialog
    from .identity_unlink_dialog import IdentityUnlinkDialog
    from .isaac_install_progress_dialog import IsaacInstallProgressDialog
    from .joint_mapping_dialog import JointMappingDialog, JointMappingHealth
    from .language_picker_dialog import LanguagePickerDialog
    from .registry_module_editor_panel import (
        RegistryModuleEditorPanel,
        open_reward_function_editor,
        open_training_motion_editor,
    )
    from .update_available_dialog import UpdateAvailableDialog
    from .update_latest_dialog import UpdateLatestDialog
    from .update_progress_dialog import UpdateProgressDialog
    from .version_notice_dialog import VersionNoticeDialog

__all__ = [
    "Bar1RiskDialog",
    "CanvasErrorDialog",
    "CanvasIssue",
    "clear_canvas_diagnostic_marks",
    "EmailIdentityDialog",
    "EulaDialog",
    "IdentityUnlinkDialog",
    "IsaacInstallProgressDialog",
    "JointMappingDialog",
    "JointMappingHealth",
    "LanguagePickerDialog",
    "RegistryModuleEditorPanel",
    "UpdateAvailableDialog",
    "UpdateLatestDialog",
    "UpdateProgressDialog",
    "VersionNoticeDialog",
    "issues_from_exception",
    "open_reward_function_editor",
    "open_training_motion_editor",
    "show_canvas_error_dialog",
]


# Map exported name -> (submodule, attribute). Anything not listed here
# raises AttributeError so typos surface loudly instead of silently
# loading the wrong module.
_LAZY: dict[str, tuple[str, str]] = {
    "Bar1RiskDialog": ("bar1_risk_dialog", "Bar1RiskDialog"),
    "CanvasErrorDialog": ("canvas_error_dialog", "CanvasErrorDialog"),
    "CanvasIssue": ("canvas_error_dialog", "CanvasIssue"),
    "clear_canvas_diagnostic_marks": ("canvas_error_dialog", "clear_canvas_diagnostic_marks"),
    "issues_from_exception": ("canvas_error_dialog", "issues_from_exception"),
    "show_canvas_error_dialog": ("canvas_error_dialog", "show_canvas_error_dialog"),
    "EmailIdentityDialog": ("email_identity_dialog", "EmailIdentityDialog"),
    "EulaDialog": ("eula_dialog", "EulaDialog"),
    "IdentityUnlinkDialog": ("identity_unlink_dialog", "IdentityUnlinkDialog"),
    "IsaacInstallProgressDialog": (
        "isaac_install_progress_dialog", "IsaacInstallProgressDialog",
    ),
    "JointMappingDialog": ("joint_mapping_dialog", "JointMappingDialog"),
    "JointMappingHealth": ("joint_mapping_dialog", "JointMappingHealth"),
    "LanguagePickerDialog": ("language_picker_dialog", "LanguagePickerDialog"),
    "RegistryModuleEditorPanel": (
        "registry_module_editor_panel", "RegistryModuleEditorPanel",
    ),
    "open_reward_function_editor": (
        "registry_module_editor_panel", "open_reward_function_editor",
    ),
    "open_training_motion_editor": (
        "registry_module_editor_panel", "open_training_motion_editor",
    ),
    "UpdateAvailableDialog": ("update_available_dialog", "UpdateAvailableDialog"),
    "UpdateLatestDialog": ("update_latest_dialog", "UpdateLatestDialog"),
    "UpdateProgressDialog": ("update_progress_dialog", "UpdateProgressDialog"),
    "VersionNoticeDialog": ("version_notice_dialog", "VersionNoticeDialog"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(
            f"module 'application.ui.dialogs' has no attribute {name!r}"
        )
    module_name, attr = _LAZY[name]
    from importlib import import_module
    mod = import_module(f"{__name__}.{module_name}")
    value = getattr(mod, attr)
    globals()[name] = value  # cache so subsequent lookups are O(1)
    return value
