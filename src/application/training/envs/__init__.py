"""application.training.envs — Generic MuJoCo + Gymnasium env package.

Stage 6 — brand-neutral training env. The env is driven entirely by
:class:`registers.robots.RobotSpec` (joint topology + MJCF asset) and
:class:`application.training.training_spec.SceneConfig`/
:class:`ObsActionContract` — no brand strings appear anywhere in this
subtree (the Stage 6 acceptance grep over the brand list must return zero).

Public surface:
    * :class:`GenericMujocoEnv`           — gymnasium.Env subclass
    * :func:`make_env`                    — TrainingSpec → env factory
    * :class:`AmpObsAlignmentError`       — Stage 8 helper used at boot
    * :func:`verify_alignment` / :func:`dump_alignment_report`

The Isaac Lab path does NOT consume this subtree — it uses Isaac Sim's
own env. Only the SB3 backend (Stage 7 + Stage 10) drives this env.
"""
from __future__ import annotations

from application.training.envs.generic_mujoco_env import (
    DEFAULT_QUADRUPED_MJCF,
    GenericMujocoEnv,
    make_env,
)
from application.training.envs.joint_alignment import (
    AlignmentReport,
    AmpObsAlignmentError,
    dump_alignment_report,
    verify_alignment,
)

__all__ = [
    "GenericMujocoEnv",
    "make_env",
    "DEFAULT_QUADRUPED_MJCF",
    "AlignmentReport",
    "AmpObsAlignmentError",
    "dump_alignment_report",
    "verify_alignment",
]
