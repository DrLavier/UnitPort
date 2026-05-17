"""IL Policy → MuJoCo sim2sim adapter — Phase 3 territory.

Phase 1 only delivers ``sim_env_context`` (the abstract base ``MjSimEnv``
inherits from). The bulk of the package — ``policy_runner``, ``obs_builder``,
``action_applier``, ``inference_engine``, ``bundle_loader``, ``deploy_contract``,
``joint_space``, ``sim2sim_compiler`` — lands in Phase 3 as 1:1 verbatim
copies (any restructuring would lose the sim2sim invariants documented in
``TRAINING_BACKEND_PLAN.yaml``).
"""

from __future__ import annotations

from .sim_env_context import SimEnvContext  # noqa: F401

__all__ = ["SimEnvContext"]
