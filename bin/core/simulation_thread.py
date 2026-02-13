#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Thread Module
Run MuJoCo simulation in independent thread
"""

from PySide6.QtCore import QThread, Signal
from typing import Optional
from bin.core.logger import log_info, log_error, log_debug


class SimulationThread(QThread):
    """Simulation thread base class"""

    # Signal definitions
    simulation_started = Signal(str)
    simulation_finished = Signal(str)
    error_occurred = Signal(str)
    progress_updated = Signal(int, str)

    def __init__(self, robot_model, action: str, **kwargs):
        """
        Initialize simulation thread

        Args:
            robot_model: Robot model instance
            action: Action to execute
            **kwargs: Other parameters
        """
        super().__init__()
        self.robot_model = robot_model
        self.action = action
        self.kwargs = kwargs
        self.running = False
        self._stop_requested = False

    def run(self):
        """Run simulation"""
        try:
            self.running = True
            self._stop_requested = False

            # Emit start signal
            start_msg = f"Starting simulation: {self.robot_model.robot_type} - {self.action}"
            self.simulation_started.emit(start_msg)
            log_info(start_msg)

            # Execute actual simulation
            self._run_simulation()

        except Exception as e:
            error_msg = f"Simulation error: {str(e)}"
            log_error(error_msg)
            self.error_occurred.emit(error_msg)

        finally:
            self.running = False
            finish_msg = "Simulation complete" if not self._stop_requested else "Simulation stopped"
            self.simulation_finished.emit(finish_msg)
            log_info(finish_msg)

    def _run_simulation(self):
        """Execute simulation (subclasses should override this method)"""
        # Call robot model action execution method
        success = self.robot_model.run_action(self.action, **self.kwargs)

        if not success:
            raise RuntimeError(f"Action execution failed: {self.action}")

    def stop(self):
        """Stop simulation"""
        log_info("Stop simulation request received")
        self._stop_requested = True

        # Stop robot
        if self.robot_model:
            self.robot_model.stop()

    def is_running(self) -> bool:
        """Check if running"""
        return self.running and not self._stop_requested
