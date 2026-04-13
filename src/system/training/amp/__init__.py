"""UnitPort AMP backend — phase_3 of AMP_design.yaml.

Port of the AMP_for_hardware ``rsl_rl`` fork into UnitPort's namespace,
plus UnitPort-specific glue (motion data bridge, joint alignment, vec
env wrapper). Per AMP_design.yaml §4.amp_backend.vendoring_strategy:

- Package path is ``src.system.training.amp.*`` — no collision with
  the ``rsl-rl-lib`` package installed in the Isaac Sim venv.
- Vendored files (``amp_ppo.py``, ``amp_on_policy_runner.py``) still
  import from ``rsl_rl`` at runtime; that resolves to the Isaac venv's
  installed ``rsl-rl-lib``. The only thing we forbid is a parallel
  ``rsl_rl`` *package* in our namespace, which would shadow the venv.

- This ``__init__.py`` re-exports only **main-venv-safe** symbols
  (pure torch / pure python). ``amp_ppo`` and the runner are loaded
  lazily by the launcher because they depend on isaac-venv-only
  ``rsl_rl`` modules and will ImportError in the main venv.

Layout:

    src/system/training/amp/
    ├── __init__.py                     (this file — safe re-exports)
    ├── algorithms/
    │   ├── __init__.py                 (empty)
    │   ├── discriminator.py            (vendored, main-venv safe)
    │   └── amp_ppo.py                  (vendored, isaac-venv only)
    ├── storage/
    │   ├── __init__.py                 (empty)
    │   └── replay_buffer.py            (vendored, main-venv safe)
    ├── utils/
    │   ├── __init__.py                 (empty)
    │   └── normalizer.py               (vendored, main-venv safe)
    ├── runners/
    │   ├── __init__.py                 (empty)
    │   └── amp_on_policy_runner.py     (vendored, isaac-venv only)
    ├── wrappers/
    │   ├── __init__.py                 (empty)
    │   └── amp_rsl_rl_vec_env_wrapper.py   (subclass, isaac-venv runtime)
    ├── data_provider.py                (new — MotionClip → amp_data bridge)
    └── joint_alignment.py              (new — field order verification)
"""
from __future__ import annotations

# ── Main-venv-safe re-exports ───────────────────────────────────────────────
# Everything below must be importable in the main UnitPort venv without
# pulling in rsl_rl or isaaclab_rl. Torch is allowed.

from src.system.training.amp.algorithms.discriminator import AMPDiscriminator
from src.system.training.amp.storage.replay_buffer import ReplayBuffer
from src.system.training.amp.utils.normalizer import Normalizer
from src.system.training.amp.data_provider import (
    MotionAmpData,
    build_amp_data,
    build_amp_data_from_files,
)
from src.system.training.amp.joint_alignment import (
    AmpObsAlignmentError,
    verify_alignment,
    dump_alignment_report,
)

__all__ = [
    "AMPDiscriminator",
    "ReplayBuffer",
    "Normalizer",
    "MotionAmpData",
    "build_amp_data",
    "build_amp_data_from_files",
    "AmpObsAlignmentError",
    "verify_alignment",
    "dump_alignment_report",
]
