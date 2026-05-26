# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Observation presets — sub-folder aggregator (Isaac Lab only)."""

from scripts.observations.registry import (
    IL_OBS_REGISTRY,
    default_il_obs_terms,
    il_obs_registry,
)


__all__ = [
    "IL_OBS_REGISTRY",
    "il_obs_registry",
    "default_il_obs_terms",
]
