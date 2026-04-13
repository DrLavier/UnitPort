"""MuJoCo runtime layer for the Mission sim2real harness.

This package provides the three-element runtime model declared in
``knowledge_base/mission_design.yaml``:

    ACTOR  (mj_actor.MjActor)    — MJCF embodiment
    FIELD  (mj_field.MjField)    — free-form scene
    BIND   (mj_sim_env.MjSimEnv) — actor ⊕ field, consumed by PolicyRunner

Bundles loaded by BehaviorNode run inside an MjSimEnv. The MjSimEnv class
is a concrete subclass of :class:`src.system.policy.sim_env_context.SimEnvContext`
— it does NOT replace the base dataclass; it implements the abstract
sim_step / reset / render / is_terminated methods.
"""

from .mj_actor import MjActor
from .mj_field import MjField
from .mj_sim_env import MjSimEnv
from .field_loader import (
    FieldNotFoundError,
    list_all_fields,
    list_project_fields,
    list_shared_fields,
    load_field,
    resolve_field_path,
    shared_root,
)

__all__ = [
    "MjActor",
    "MjField",
    "MjSimEnv",
    "FieldNotFoundError",
    "load_field",
    "list_all_fields",
    "list_shared_fields",
    "list_project_fields",
    "resolve_field_path",
    "shared_root",
]
