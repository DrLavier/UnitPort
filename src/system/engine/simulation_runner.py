#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime simulation runner."""

from typing import Any
from PySide6.QtCore import QThread, Signal
from src.system.core.logger import log_info, log_error


class SimulationRunner(QThread):
    """Run a robot action in an isolated worker thread.

    Phase 6 additions:
    - ``overlay_info``: mutable dict passed to the viewer for real-time overlay
      rendering (active skill name, transition state, input mode indicator).
    - ``overlay_updated`` signal: emitted when overlay_info changes so the UI
      can refresh the viewer decoration.
    """

    simulation_started = Signal(str)
    simulation_finished = Signal(str)
    error_occurred = Signal(str)
    progress_updated = Signal(int, str)
    overlay_updated = Signal(dict)   # Phase 6: viewer overlay info changed

    def __init__(self, robot_model: Any, action: str, **kwargs):
        super().__init__()
        self.robot_model = robot_model
        self.action = action
        self.kwargs = kwargs
        self.running = False
        self._stop_requested = False
        self.overlay_info: dict = {}  # Phase 6: {skill_name, transition_state, input_mode}

    def run(self):
        try:
            self.running = True
            self._stop_requested = False
            start_msg = f"Starting simulation: {self.robot_model.robot_type} - {self.action}"
            self.simulation_started.emit(start_msg)
            log_info(start_msg)
            self._run_simulation()
        except Exception as exc:
            error_msg = f"Simulation error: {exc}"
            log_error(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            self.running = False
            finish_msg = "Simulation complete" if not self._stop_requested else "Simulation stopped"
            self.simulation_finished.emit(finish_msg)
            log_info(finish_msg)

    def _run_simulation(self):
        success = self.robot_model.run_action(self.action, **self.kwargs)
        if not success:
            raise RuntimeError(f"Action execution failed: {self.action}")

    def stop(self):
        self._stop_requested = True
        if self.robot_model:
            self.robot_model.stop()

    def is_running(self) -> bool:
        return self.running and not self._stop_requested

    def set_overlay(self, **kwargs: str) -> None:
        """Update viewer overlay fields and emit signal.

        Common keys: ``skill_name``, ``transition_state``, ``input_mode``.
        """
        self.overlay_info.update(kwargs)
        self.overlay_updated.emit(dict(self.overlay_info))

