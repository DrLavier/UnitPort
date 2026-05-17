"""Isaac Lab subprocess training backend (RELEASE port).

Public surface:
    IsaacLabConfig          — pure data, knows how to build subprocess args
    IsaacLabBackend         — subprocess orchestrator + stdout parser
    IsaacLabTrainingTask    — TrainingTask wrapper for canvas / TasksManager
    detect_isaac_lab        — re-export of registers.backends Isaac probe
"""
from .config import IsaacLabConfig
from .backend import IsaacLabBackend
from .task import IsaacLabTrainingTask

__all__ = [
    "IsaacLabConfig",
    "IsaacLabBackend",
    "IsaacLabTrainingTask",
]
