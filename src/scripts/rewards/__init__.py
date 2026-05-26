# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reward presets — sub-folder aggregator.

This package contains one Python file per reward, organised by
backend (``sb3/`` vs ``isaac_lab/``). The :mod:`scripts.rewards.registry`
aggregator scans both sub-packages at import time and builds the legacy
``REWARD_REGISTRY`` / ``IL_REWARD_REGISTRY`` dicts; everything else in
the public surface below is preserved verbatim from the pre-split
monolith.
"""

from scripts.rewards.registry import (
    IL_RECOMMENDED_LOCOMOTION_REWARD_TERMS,
    IL_REWARD_REGISTRY,
    RECOMMENDED_LOCOMOTION_REWARD_TERMS,
    REWARD_REGISTRY,
    default_il_reward_terms,
    default_reward_terms,
    il_reward_registry,
    recommended_reward_terms_for_backend,
    reward_registry,
    warn_missing_recommended_reward_terms,
)


__all__ = [
    "REWARD_REGISTRY",
    "IL_REWARD_REGISTRY",
    "RECOMMENDED_LOCOMOTION_REWARD_TERMS",
    "IL_RECOMMENDED_LOCOMOTION_REWARD_TERMS",
    "recommended_reward_terms_for_backend",
    "reward_registry",
    "il_reward_registry",
    "default_reward_terms",
    "default_il_reward_terms",
    "warn_missing_recommended_reward_terms",
]
