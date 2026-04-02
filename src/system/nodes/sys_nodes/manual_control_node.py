#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Control Node — wraps UserInputManager as a Canvas behaviour source.

Allows the user to control a robot via keyboard/gamepad input, producing
actions in the same format as policy inference output.  This node can be
used in any Canvas DAG position where a BehaviorNode would normally appear.
"""

from typing import Any, Dict

from src.system.core.logger import log_info, log_warning

from .base_node import BaseNode


class ManualControlNode(BaseNode):
    """Canvas node for real-time manual robot control.

    Inputs:
        sim_env: SimEnvContext (optional, for simulation target)

    Outputs:
        result: Execution result dict

    Parameters:
        mode:           "keyboard" | "gamepad"
        profile:        Input profile name (e.g. "keyboard_quadruped")
        max_duration:   Maximum execution duration in seconds (0 = unlimited)
        max_steps:      Maximum control steps (0 = derived from duration + frequency)
        command_scale:  Scale factor applied to velocity commands
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "manual_control")
        self.inputs = {
            "sim_env": None,
        }
        self.outputs = {
            "result": None,
        }
        self.parameters = {
            "mode": "keyboard",
            "profile": "keyboard_quadruped",
            "max_duration": 30.0,
            "max_steps": 0,
            "command_scale": 1.0,
        }

    def get_display_name(self) -> str:
        return "Manual Control"

    def get_description(self) -> str:
        return "Real-time manual robot control via keyboard or gamepad input"

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manual control session.

        In a real execution context, this node enters a control loop that
        reads from UserInputManager each tick and dispatches actions through
        the ActionDispatcher.  The loop runs until max_duration/max_steps
        is reached or the user triggers stop.

        For DAG validation (non-interactive), returns a summary result.
        """
        sim_env = inputs.get("sim_env") or self.get_parameter("sim_env")
        mode = self.get_parameter("mode", "keyboard")
        profile_name = self.get_parameter("profile", "keyboard_quadruped")
        max_duration = float(self.get_parameter("max_duration", 30.0))
        max_steps = int(self.get_parameter("max_steps", 0))
        command_scale = float(self.get_parameter("command_scale", 1.0))

        try:
            from src.system.runtime.user_input_manager import UserInputManager

            manager = UserInputManager(profile_name=profile_name)

            # Determine action dim from sim_env if available
            if sim_env is not None:
                joint_names = getattr(sim_env, "joint_names", [])
                if joint_names:
                    manager.action_dim = len(joint_names)

            # Derive max_steps from duration + frequency
            if max_steps <= 0 and sim_env is not None:
                control_hz = getattr(sim_env, "control_frequency_hz", 50.0)
                max_steps = max(1, int(control_hz * max_duration))
            elif max_steps <= 0:
                max_steps = int(50.0 * max_duration)

            log_info(
                f"ManualControlNode: mode={mode}, profile={profile_name}, "
                f"max_steps={max_steps}, scale={command_scale}"
            )

            # In non-interactive context (DAG execution), return configuration
            # summary.  Real-time control loop is driven by the MissionRunThread
            # which calls get_velocity_command() / get_action() each tick.
            return {
                "status": "ready",
                "mode": mode,
                "profile": profile_name,
                "max_steps": max_steps,
                "command_scale": command_scale,
                "action_dim": manager.action_dim,
                "input_manager": manager,
            }

        except Exception as exc:
            log_warning(f"ManualControlNode setup failed: {exc}")
            return {
                "status": "error",
                "error": str(exc),
            }
