# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UnitPort AMP backend — Phase 3 RELEASE layout.

This package holds the AMP algorithm core that BOTH training chains
build on:

    application/training/amp/
    ├── __init__.py                     (this file — safe re-exports only)
    ├── algorithms/
    │   ├── discriminator.py            (shared, main-venv safe)
    │   ├── amp_ppo.py                  (SB3 chain — pulls stable_baselines3)
    │   ├── rsl_amp_ppo.py              (IL chain — pulls rsl_rl, isaac-venv only)
    │   └── actor_critic.py             (IL chain — vendored RSL-RL actor-critic)
    ├── storage/
    │   ├── replay_buffer.py            (shared, main-venv safe)
    │   └── rollout_storage.py          (IL chain — RSL-RL rollout buffer)
    ├── utils/
    │   └── normalizer.py               (shared, main-venv safe)
    ├── runners/
    │   └── amp_on_policy_runner.py     (IL chain — isaac-venv only;
    │                                     imports rsl_amp_ppo, NOT amp_ppo)
    ├── wrappers/
    │   └── amp_rsl_rl_vec_env_wrapper.py   (IL chain — isaac-venv runtime)
    ├── data_provider.py                (shared — MotionClip → amp_data bridge)
    ├── joint_alignment.py              (shared — field order verification)
    ├── mdp_events.py                   (IL chain — RSI mdp_event)
    ├── tag_router.py                   (IL chain — task-conditional reward)
    └── obs_terms.py                    (shared — AMP obs term definitions)

**Backend boundary (Phase 3 plan rule §B6).** The two ``*_amp_ppo.py``
files are deliberately separate so each chain depends only on the libs
it owns:

  * SB3 chain → ``algorithms.amp_ppo.train_amp_ppo`` /
    ``AmpPpoTrainer`` (uses ``stable_baselines3`` PPO + custom AMP
    callback). Lives here because it shares ``discriminator`` /
    ``replay_buffer`` / ``normalizer`` / ``data_provider`` with the IL
    chain — extracting it under ``application/training/sb3_*`` would
    duplicate those shared modules per chain.

  * IL chain → ``algorithms.rsl_amp_ppo.AMPPPO`` (vendored RSL-RL
    style algorithm) + ``runners.amp_on_policy_runner.AMPOnPolicyRunner``.
    Lazily imported by ``isaac_lab/launcher/il_train_launcher.py`` —
    will ImportError in the main UnitPort venv (rsl_rl is only present
    inside Isaac Lab's venv).

Cross-chain rule:
  * ``rsl_amp_ppo``, ``amp_on_policy_runner``, and ``wrappers/`` MUST NOT
    import ``stable_baselines3``.
  * ``amp_ppo`` MUST NOT import ``rsl_rl``.
Verified by Phase 3 §Stage 8 grep audit.

This ``__init__.py`` only re-exports **main-venv-safe** symbols (pure
torch / pure python). The two ``*_amp_ppo`` modules + RSL-RL runner +
wrapper are loaded lazily by their respective callers.
"""
from __future__ import annotations

# ── Main-venv-safe re-exports ───────────────────────────────────────────────
# Everything below must be importable in the main UnitPort venv without
# pulling in rsl_rl or isaaclab_rl. Torch is allowed.

from application.training.amp.algorithms.discriminator import AMPDiscriminator
from application.training.amp.storage.replay_buffer import AmpReplayBuffer
from application.training.amp.utils.normalizer import Normalizer
from application.training.amp.data_provider import (
    MotionAmpData,
    build_amp_data,
    build_amp_data_from_files,
)
from application.training.amp.joint_alignment import (
    AmpObsAlignmentError,
    verify_alignment,
    dump_alignment_report,
)

__all__ = [
    "AMPDiscriminator",
    "AmpReplayBuffer",
    "Normalizer",
    "MotionAmpData",
    "build_amp_data",
    "build_amp_data_from_files",
    "AmpObsAlignmentError",
    "verify_alignment",
    "dump_alignment_report",
]
