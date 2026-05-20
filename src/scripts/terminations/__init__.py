"""Termination presets — sub-folder aggregator."""

from scripts.terminations.registry import (
    IL_TERMINATION_REGISTRY,
    TERMINATION_REGISTRY,
    default_il_termination_conditions,
    default_termination_conditions,
    il_termination_registry,
    termination_registry,
)


__all__ = [
    "TERMINATION_REGISTRY",
    "IL_TERMINATION_REGISTRY",
    "termination_registry",
    "il_termination_registry",
    "default_termination_conditions",
    "default_il_termination_conditions",
]
