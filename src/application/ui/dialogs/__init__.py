"""Modal dialogs opened from the sidebar panels and node ParamRow ⚙ buttons."""

from __future__ import annotations

from .canvas_error_dialog import (
    CanvasErrorDialog,
    CanvasIssue,
    issues_from_exception,
    show_canvas_error_dialog,
)
from .email_identity_dialog import EmailIdentityDialog
from .identity_unlink_dialog import IdentityUnlinkDialog
from .registry_module_editor_panel import (
    RegistryModuleEditorPanel,
    open_reward_function_editor,
    open_training_motion_editor,
)
from .update_available_dialog import UpdateAvailableDialog
from .update_latest_dialog import UpdateLatestDialog
from .update_progress_dialog import UpdateProgressDialog

__all__ = [
    "CanvasErrorDialog",
    "CanvasIssue",
    "EmailIdentityDialog",
    "IdentityUnlinkDialog",
    "RegistryModuleEditorPanel",
    "UpdateAvailableDialog",
    "UpdateLatestDialog",
    "UpdateProgressDialog",
    "issues_from_exception",
    "open_reward_function_editor",
    "open_training_motion_editor",
    "show_canvas_error_dialog",
]
