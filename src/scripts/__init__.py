# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Default training scripts (presets) shipped with UnitPort.

Subpackages are organised by category — every category here defines
**executable script content** (Python source, regex rules, IR roles,
etc.). Pure catalog data without per-item Python lives under
:mod:`registers` instead (see ``registers.commands``).

Categories
----------

* :mod:`scripts.rewards`         — reward presets (SB3 + Isaac Lab)
* :mod:`scripts.terminations`    — termination presets (SB3 + Isaac Lab)
* :mod:`scripts.observations`    — observation presets (Isaac Lab only)
* :mod:`scripts.discriminator`   — AMP discriminator method-override presets
* :mod:`scripts.gait`            — gait presets (Walk These Ways style)
* :mod:`scripts.scenes`          — playground scene registry
* :mod:`scripts.il_envs`         — Isaac Lab environment presets
* :mod:`scripts.training_motion` — motion-clip labels / library / IR mapping

Top-level utility modules
-------------------------

Two flat ``.py`` files at this package's root hold cross-cutting code
shared by the kind-based sub-packages above (rewards / terminations /
observations / discriminator):

* :mod:`scripts.task_module` — :class:`TaskModuleItem` dataclass,
  backend / algorithm / family constants, and the four ``*_item()``
  factories.
* :mod:`scripts.query`       — ``ALL_REGISTRIES`` plus ``lookup``,
  ``query_registry``, ``validate_keys``, ``iter_all_items``,
  ``writable_registry``, ``emit_inline_overrides``.

Import surface
--------------

All public names are re-exported here so callers can do:

    from scripts import REWARD_REGISTRY, lookup, GaitPreset, infer_task_tag

The sub-packages themselves are namespace packages (no per-folder
``__init__.py``); module-specific imports still work where preferred:

    from scripts.rewards.registry import REWARD_REGISTRY
"""

# ── Shared types + factories ─────────────────────────────────────────
from scripts.task_module import (
    ALG_ALL,
    ALG_AMP,
    ALG_PPO,
    ALG_SAC,
    ALG_TD3,
    ALL_ALGORITHMS,
    ALL_BACKENDS,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    BACKEND_NEWTON,
    BACKEND_SB3,
    IL_MOD_INLINE,
    IL_MOD_MDP,
    IL_MOD_VEL,
    LEGGED_FAMILIES,
    LOCOMOTION_FAMILIES,
    TaskModuleItem,
    discriminator_item,
    observation_item,
    reward_item,
    termination_item,
)

# ── Cross-cutting query helpers ──────────────────────────────────────
from scripts.query import (
    ALL_REGISTRIES,
    emit_inline_overrides,
    iter_all_items,
    lookup,
    query_registry,
    validate_keys,
    writable_registry,
)

# ── Reward presets ───────────────────────────────────────────────────
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

# ── Termination presets ──────────────────────────────────────────────
from scripts.terminations.registry import (
    IL_TERMINATION_REGISTRY,
    TERMINATION_REGISTRY,
    default_il_termination_conditions,
    default_termination_conditions,
    il_termination_registry,
    termination_registry,
)

# ── Observation presets ──────────────────────────────────────────────
from scripts.observations.registry import (
    IL_OBS_REGISTRY,
    default_il_obs_terms,
    il_obs_registry,
)

# ── AMP discriminator presets ────────────────────────────────────────
from scripts.discriminator.registry import (
    IL_DISC_REGISTRY,
    il_disc_registry,
)

# ── Gait presets ─────────────────────────────────────────────────────
from scripts.gait.presets import (
    DEFAULT_PRESETS,
    GaitPreset,
    default_presets,
    default_presets_json,
)

# ── Scene presets ────────────────────────────────────────────────────
from scripts.scenes.registry import (
    Scene,
    SceneValidationError,
    clear_registry,
    get_scene,
    has_scene,
    list_review_scenes,
    list_scenes,
    register_scene,
    rescan,
)

# ── Isaac Lab environment presets ────────────────────────────────────
from scripts.il_envs.presets import (
    IL_PRESETS,
    ILPreset,
    get_il_preset,
    get_il_preset_task_name,
    list_il_presets,
)

# ── Training motion: labels ──────────────────────────────────────────
from scripts.training_motion.labels import (
    infer_task_tag,
    list_available_tags,
    load_manifest,
    load_manifest_weights,
    resolve_task_tags,
)

# ── Training motion: library ─────────────────────────────────────────
from scripts.training_motion.library import (
    CATEGORIES,
    MotionEntry,
    MotionPack,
    apply_clip_enabled_filter,
    build_pack_ref,
    expand_motion_pack,
    get_category,
    import_file,
    list_entries,
    list_motion_packs,
    resolve_motion_source_files,
    resolve_pack_ref,
)

# ── Training motion: IR mapping ──────────────────────────────────────
from scripts.training_motion.ir_mapping import (
    ClipIRStatus,
    FORMAT_IR_ROLES,
    QUADRUPED_AMP_IR_ROLES,
    get_ir_roles,
)


__all__ = [
    # Shared types + factories
    "TaskModuleItem",
    "reward_item",
    "termination_item",
    "observation_item",
    "discriminator_item",
    "BACKEND_SB3",
    "BACKEND_ISAAC",
    "BACKEND_NEWTON",
    "ALL_BACKENDS",
    "ALG_PPO",
    "ALG_SAC",
    "ALG_AMP",
    "ALG_TD3",
    "ALG_ALL",
    "ALL_ALGORITHMS",
    "IL_MOD_MDP",
    "IL_MOD_VEL",
    "IL_MOD_INLINE",
    "LOCOMOTION_FAMILIES",
    "LEGGED_FAMILIES",
    "ALL_FAMILIES",
    # Cross-cutting query helpers
    "ALL_REGISTRIES",
    "lookup",
    "query_registry",
    "validate_keys",
    "iter_all_items",
    "writable_registry",
    "emit_inline_overrides",
    # Rewards
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
    # Terminations
    "TERMINATION_REGISTRY",
    "IL_TERMINATION_REGISTRY",
    "termination_registry",
    "il_termination_registry",
    "default_termination_conditions",
    "default_il_termination_conditions",
    # Observations
    "IL_OBS_REGISTRY",
    "il_obs_registry",
    "default_il_obs_terms",
    # Discriminator
    "IL_DISC_REGISTRY",
    "il_disc_registry",
    # Gait
    "GaitPreset",
    "DEFAULT_PRESETS",
    "default_presets",
    "default_presets_json",
    # Scenes
    "Scene",
    "SceneValidationError",
    "register_scene",
    "get_scene",
    "has_scene",
    "list_scenes",
    "list_review_scenes",
    "clear_registry",
    "rescan",
    # IL envs
    "ILPreset",
    "IL_PRESETS",
    "list_il_presets",
    "get_il_preset",
    "get_il_preset_task_name",
    # Training motion labels
    "infer_task_tag",
    "load_manifest",
    "load_manifest_weights",
    "resolve_task_tags",
    "list_available_tags",
    # Training motion library
    "CATEGORIES",
    "MotionEntry",
    "MotionPack",
    "get_category",
    "list_entries",
    "list_motion_packs",
    "expand_motion_pack",
    "resolve_pack_ref",
    "build_pack_ref",
    "resolve_motion_source_files",
    "apply_clip_enabled_filter",
    "import_file",
    # Training motion IR mapping
    "QUADRUPED_AMP_IR_ROLES",
    "FORMAT_IR_ROLES",
    "ClipIRStatus",
    "get_ir_roles",
]
