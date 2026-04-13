"""AMP algorithms — discriminator + AMP-flavored PPO.

``discriminator`` is main-venv safe (pure torch).
``amp_ppo`` imports from ``rsl_rl`` (Isaac Sim venv only); it is NOT
re-exported here so ``import src.system.training.amp.algorithms`` does
not fail outside the Isaac venv. The launcher loads it lazily via
``from src.system.training.amp.algorithms.amp_ppo import AMPPPO``.
"""
from __future__ import annotations

from src.system.training.amp.algorithms.discriminator import AMPDiscriminator

__all__ = ["AMPDiscriminator"]
