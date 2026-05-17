"""Live input dispatcher (keyboard + gamepad).

Public entry points:

    from application.service.input import get_global_input_manager

    mgr = get_global_input_manager()
    mgr.set_mode("keyboard")
    mgr.forward_key_press(int(Qt.Key.Key_W))    # invoked by Controller panel
"""

from __future__ import annotations

from .manager import (
    GAMEPAD_BUTTON_ACTIONS,
    GlobalInputManager,
    get_global_input_manager,
)

__all__ = [
    "GlobalInputManager",
    "get_global_input_manager",
    "GAMEPAD_BUTTON_ACTIONS",
]
