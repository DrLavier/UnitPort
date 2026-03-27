#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boston Dynamics Spot model with MuJoCo simulation support."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.base import BaseRobotModel
from bin.core.logger import log_debug, log_error, log_info, log_warning
from bin.core.config_manager import ConfigManager
from system.mujoco_asset_registry import resolve_mujoco_asset

SPOT_SDK_AVAILABLE = False
MUJOCO_AVAILABLE = False

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore[assignment]


class SpotModel(BaseRobotModel):
    def __init__(self, robot_type: str = "spot"):
        super().__init__(robot_type or "spot")
        self.config = ConfigManager()
        self.is_available = SPOT_SDK_AVAILABLE
        self.mujoco_available = MUJOCO_AVAILABLE
        self.model = None
        self.data = None
        self.viewer = None
        self.running = False
        self.stop_requested = False
        self.pause_requested = False
        self._register_actions()

    def _register_actions(self):
        self.register_action("stand", self._stand_action, "Stand", {})
        self.register_action("sit", self._sit_action, "Sit", {})
        self.register_action("walk", self._walk_action, "Walk", {})
        self.register_action("stop", self._stop_action, "Stop", {})

    def initialize(self) -> bool:
        if not self.mujoco_available:
            log_warning("MuJoCo unavailable; Spot model running in stub mode")
            return True
        return True

    def _find_model_file(self) -> Optional[Path]:
        location = resolve_mujoco_asset("bostiondynamics", self.robot_type)
        if location is not None:
            log_info(f"Spot model file found via {location.source}: {location.scene_path}")
            return location.scene_path
        return None

    def load_model(self) -> bool:
        if not self.mujoco_available:
            log_warning("MuJoCo unavailable; skipping Spot model loading")
            return True
        try:
            model_file = self._find_model_file()
            if model_file is None:
                log_error("Spot MuJoCo asset not found")
                return False
            self.model = mujoco.MjModel.from_xml_path(str(model_file))
            self.data = mujoco.MjData(self.model)
            log_info(f"Spot MuJoCo model loaded: {model_file}")
            return True
        except Exception as exc:
            log_error(f"Spot model loading failed: {exc}")
            return False

    def ensure_viewer(self) -> bool:
        if not self.mujoco_available:
            return False
        if self.model is None or self.data is None:
            if not self.load_model():
                return False
        if self.viewer is not None:
            return True
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return True
        except Exception as exc:
            log_error(f"Spot viewer launch failed: {exc}")
            self.viewer = None
            return False

    def _set_initial_pose(self):
        if self.model is None or self.data is None:
            return
        try:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            else:
                mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        except Exception as exc:
            log_warning(f"Spot initial pose setup failed: {exc}")

    def _should_stop(self, viewer=None) -> bool:
        if self.stop_requested:
            return True
        if viewer is None:
            return False
        try:
            return not viewer.is_running()
        except Exception:
            return False

    def velocity_move_mujoco(self, vx: float, vy: float, vyaw: float, duration: float) -> bool:
        if not self.mujoco_available:
            log_warning("MuJoCo unavailable; simulating Spot velocity_move_mujoco")
            time.sleep(min(float(duration), 0.1))
            return True
        try:
            if not self.ensure_viewer():
                return False
            viewer = self.viewer
            self._set_initial_pose()
            viewer.sync()
            return self._walk_trot_gait_spot(viewer, vx=float(vx), duration=float(duration))
        except Exception as exc:
            log_error(f"Spot velocity_move_mujoco failed: {exc}")
            return False

    def _walk_trot_gait_spot(self, viewer, vx: float, duration: float) -> bool:
        stand = {
            "fl_hx": 0.0, "fl_hy": 1.04, "fl_kn": -1.80,
            "fr_hx": 0.0, "fr_hy": 1.04, "fr_kn": -1.80,
            "hl_hx": 0.0, "hl_hy": 1.04, "hl_kn": -1.80,
            "hr_hx": 0.0, "hr_hy": 1.04, "hr_kn": -1.80,
        }
        dt = self.model.opt.timestep
        gait_period = 0.5
        half_cycle = max(2, int(gait_period / dt) // 2)
        num_cycles = max(1, int(max(float(duration), gait_period) / gait_period))
        speed_frac = max(0.15, min(1.0, abs(float(vx)) / 1.0))
        hip_swing = 0.25 * speed_frac
        knee_lift = 0.30 * speed_frac
        hip_abduct = 0.08 * speed_frac

        def apply(targets: Dict[str, float]):
            for joint_name, target in targets.items():
                try:
                    aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                    if aid < 0:
                        continue
                    self.data.ctrl[aid] = target
                except Exception:
                    continue
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(dt)

        for _ in range(int(0.3 / dt)):
            if self._should_stop(viewer):
                return False
            apply(stand)

        def swing_targets(front: str, rear: str) -> Dict[str, float]:
            t = dict(stand)
            for leg in (front, rear):
                # Positive hy = forward reach in Spot's MuJoCo joint convention
                t[f"{leg}_hy"] = stand[f"{leg}_hy"] + hip_swing
                t[f"{leg}_kn"] = stand[f"{leg}_kn"] + knee_lift
            support = {"fl", "fr", "hl", "hr"} - {front, rear}
            for leg in support:
                t[f"{leg}_hy"] = stand[f"{leg}_hy"] - 0.10 * speed_frac
                t[f"{leg}_kn"] = stand[f"{leg}_kn"] - 0.10 * speed_frac
            t[f"{front}_hx"] = hip_abduct
            t[f"{rear}_hx"] = -hip_abduct
            return t

        for _ in range(num_cycles):
            for _step in range(half_cycle):
                if self._should_stop(viewer):
                    return False
                apply(swing_targets("fr", "hl"))
            for _step in range(half_cycle):
                if self._should_stop(viewer):
                    return False
                apply(swing_targets("fl", "hr"))

        for _ in range(int(0.2 / dt)):
            if self._should_stop(viewer):
                return False
            apply(stand)
        return True

    def run_action(self, action_name: str, **kwargs) -> bool:
        info = self._actions.get(action_name)
        if not info:
            return False
        return bool(info["function"](**kwargs))

    def get_available_actions(self) -> List[str]:
        return list(self._actions.keys())

    def get_sensor_data(self) -> Dict[str, Any]:
        if self.data is None:
            return {"simulated": True, "session_open": self.viewer is not None}
        return {
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "time": float(self.data.time),
        }

    def stop(self):
        self.stop_requested = True
        self.running = False

    def _stand_action(self, **kwargs) -> bool:
        return True

    def _sit_action(self, **kwargs) -> bool:
        return True

    def _walk_action(self, **kwargs) -> bool:
        return self.velocity_move_mujoco(
            float(kwargs.get("vx", 0.5)),
            float(kwargs.get("vy", 0.0)),
            float(kwargs.get("vyaw", 0.0)),
            float(kwargs.get("duration", 1.0)),
        )

    def _stop_action(self, **kwargs) -> bool:
        self.stop()
        return True
