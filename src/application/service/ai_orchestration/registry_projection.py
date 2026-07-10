# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""C2 Registry Projection — the basis agents reason from (blueprint §3.C2).

A **deterministic, runtime-generated** projection of the live registry and node
manifests, scoped to the locked ``(backend, robot family)``. It is the single
source of the vocabulary an orchestration agent may emit; no agent keeps a
hand-maintained copy (which would drift — blueprint §7 single-source-of-truth).

Per ``schema_id`` it exposes:
  * **ports** — input/output :class:`PortSpec` (the basis for the C1
    interface-alignment check in :mod:`.mutation_api`).
  * **param schema** — per key: ``param_type`` / ``choices`` (enum) / numeric
    ``[min, max]`` / ``conditional_on`` backend gate. Params gated OFF for the
    locked backend are dropped here (backend-first, blueprint §3 / footgun
    "backend & family first").

Plus the closed registry enumerations the model must pick from:
  * reward function names with **hard weight bounds** (the #1 safety vocab —
    ``training_config.md`` §6), filtered by ``backend`` ∩ ``family``.
  * termination kinds with bounds, observation terms, command items, init-pose
    presets, trainable SKUs.
  * the **family-required node set** — reused from
    ``spec_validator._REQUIRED_NODES`` (single source, not re-listed).

:meth:`RegistryProjection.to_tool_schemas` renders these as LLM tool-call
JSON-schemas with ``enum`` lists inlined, so the model is constrained *as it
writes* (blueprint §3.C2). Per-reward numeric bounds vary by function and are
enforced by the C1 tier-1 check (:mod:`.mutation_api`) + surfaced in the agent
prompt via :attr:`RegistryProjection.rewards`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from unitport_sdk import log_warning

from application.compiler.nodes import ParamSpec, PortSpec
from application.compiler.term_payload import PAGE_GLOBAL
from application.training.spec_validator import (
    _REQUIRED_NODES,
    AlgorithmFamily,
)

from .contracts import AgentRole

__all__ = [
    "ParamFieldSchema",
    "RewardSchema",
    "TerminationSchema",
    "RegistryProjection",
    "build_projection",
    "build_projection_for_sku",
    "default_algorithm_family",
]


# =============================================================================
# Projected value shapes
# =============================================================================

@dataclass(frozen=True)
class ParamFieldSchema:
    """One node param's projected schema."""

    key: str
    param_type: str
    choices: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Any = None
    conditional_on: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RewardSchema:
    """One reward function with its HARD weight bounds (safety-critical).

    ``category`` (semantic role: command_tracking / command_penalty / posture /
    gait / …) and ``hint`` (a one-line design note) come from the
    ``registers.reward_doctrine`` teaching data — the pairing/degenerate-optimum
    knowledge the raw registry strips, restored so the agent can reason
    dialectically. Both default to empty (an uncategorized term simply carries
    no teaching hint — advisory, never required)."""

    name: str
    minimum: float
    maximum: float
    default: float
    polarity: str
    category: str = ""
    hint: str = ""


@dataclass(frozen=True)
class TerminationSchema:
    """One termination kind with its threshold bounds."""

    name: str
    minimum: float
    maximum: float
    default: float


# =============================================================================
# Backend / conditional helpers
# =============================================================================

def default_algorithm_family(backend: str) -> AlgorithmFamily:
    """The default trainer family for a backend (PPO path).

    The Design agent may override (e.g. ``IL_AMP_PPO``); this is the seed used
    when only the backend is known.
    """
    return AlgorithmFamily.IL_PPO if backend == "isaac_lab" else AlgorithmFamily.SB3_PPO


def _conditional_drops_for_backend(
    cond: Optional[Dict[str, Any]], backend: str
) -> bool:
    """Return True when a ``conditional_on`` clause gates a param/port OFF for
    ``backend``.

    Only the ``key == "backend"`` clause is evaluable from the projection
    (other clauses depend on sibling param values we don't have here, and are
    UI-display gates rather than hard validity — they are kept). Mirrors the
    semantics of ``spec_validator``'s port-active check for the backend axis.
    """
    if not isinstance(cond, dict):
        return False
    if str(cond.get("key", "")) != "backend":
        return False
    op = str(cond.get("op", "=="))
    val = cond.get("value")
    if op == "==":
        return backend != val
    if op == "!=":
        return backend == val
    if op == "in":
        return backend not in (val or [])
    if op == "not in":
        return backend in (val or [])
    # Unknown operator: do not silently drop — keep the param and warn.
    log_warning(f"[ai.projection] unknown conditional_on op {op!r}; keeping param")
    return False


# =============================================================================
# RegistryProjection
# =============================================================================

@dataclass
class RegistryProjection:
    """The vocabulary, scoped to ``(backend, families, algorithm_family)``."""

    backend: str
    families: FrozenSet[str]
    algorithm_family: AlgorithmFamily
    rewards: Dict[str, RewardSchema] = field(default_factory=dict)
    terminations: Dict[str, TerminationSchema] = field(default_factory=dict)
    obs_terms: List[str] = field(default_factory=list)
    command_items: List[str] = field(default_factory=list)
    init_poses: List[str] = field(default_factory=list)
    skus: List[str] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    required_nodes: Tuple[str, ...] = ()
    #: pd-group reward-page keys available for the family set (LIVE from the
    #: pd_groups registry). ``__global__`` is always available and is NOT in
    #: this list — the composition view prepends it. Empty for a family with no
    #: pd_groups (e.g. ``generic``) → only the global page exists.
    pd_group_pages: List[str] = field(default_factory=list)
    #: Available clip SEGMENTS (user-cut sub-ranges) the agent may reference on a
    #: training item's ``clip`` as ``<clip_ref>#seg=lo-hi``. Grouped by parent
    #: clip; LIVE from the assets-browser (list_pickable_motion_entries +
    #: list_segments). Empty when no segments have been cut. Discovery only — the
    #: compile gate still validates the ref/range the agent emits.
    motion_segments: List[Dict[str, Any]] = field(default_factory=list)

    # -- per-schema_id lookups (computed live from the node registry) --------

    def ports(self, schema_id: str) -> Dict[str, List[PortSpec]]:
        """Return ``{"inputs": [...], "outputs": [...]}`` PortSpecs, or empty
        lists for an unknown schema_id."""
        from registers import nodes as _nodes

        manifest = _nodes.get_manifest(schema_id)
        if manifest is None:
            return {"inputs": [], "outputs": []}
        return {"inputs": list(manifest.inputs), "outputs": list(manifest.outputs)}

    def port(self, schema_id: str, port_name: str) -> Optional[PortSpec]:
        """Return one PortSpec by name (input or output), or None."""
        spec = self.ports(schema_id)
        for p in spec["inputs"] + spec["outputs"]:
            if p.name == port_name:
                return p
        return None

    def param_schema(self, schema_id: str) -> Dict[str, ParamFieldSchema]:
        """Return ``{key: ParamFieldSchema}`` for a node, with params gated OFF
        for the locked backend dropped."""
        from registers import nodes as _nodes

        manifest = _nodes.get_manifest(schema_id)
        if manifest is None:
            return {}
        out: Dict[str, ParamFieldSchema] = {}
        for spec in manifest.parameters:
            cond = (spec.meta or {}).get("conditional_on") if spec.meta else None
            if _conditional_drops_for_backend(cond, self.backend):
                continue
            meta = spec.meta or {}
            out[spec.key] = ParamFieldSchema(
                key=spec.key,
                param_type=spec.type,
                choices=list(spec.choices) if spec.choices is not None else None,
                minimum=_as_float(meta.get("min")),
                maximum=_as_float(meta.get("max")),
                default=spec.default,
                conditional_on=dict(cond) if isinstance(cond, dict) else None,
            )
        return out

    # -- LLM tool schemas (blueprint §3.C2: enums inlined at generation) ------

    def to_tool_schemas(
        self,
        scope: AgentRole,
        *,
        allowed_schema_ids: Optional[FrozenSet[str]] = None,
    ) -> List[Dict[str, Any]]:
        """JSON-schema tool definitions for an agent role.

        Only producing roles (``ORCHESTRATOR`` in v1) get mutating tools; the
        read-only ``DESIGN`` / ``CRITIC`` / ``INTEGRATOR`` roles return [] (they
        act through structured findings, not writes). Node ids, reward names,
        command items, obs terms and termination kinds are inlined as ``enum``s
        so the model is constrained as it writes; per-reward numeric bounds are
        enforced by the C1 tier-1 check (they vary per function).

        ``allowed_schema_ids`` scopes the ``place_node`` node-type enum to a
        single canvas block (AI Build v2 block-by-block orchestration): the
        model can only place node types belonging to the active block, a hard
        constraint at generation time rather than an after-the-fact rejection.
        ``None`` (the default) offers the full node catalog.
        """
        if scope != AgentRole.ORCHESTRATOR:
            return []
        reward_names = sorted(self.rewards.keys())
        term_names = sorted(self.terminations.keys())
        place_node_ids = (
            sorted(set(self.node_ids) & set(allowed_schema_ids))
            if allowed_schema_ids is not None
            else self.node_ids
        )
        return [
            _tool(
                "place_node",
                "Place a new node of the given type onto the canvas.",
                {
                    "schema_id": _enum_prop(place_node_ids, "Node type to place."),
                    "instance_id": _str_prop("Optional explicit instance id."),
                },
                ["schema_id"],
            ),
            _tool(
                "remove_node",
                "Delete a node (and its edges) by instance id.",
                {"instance_id": _str_prop("Instance id of the node to remove.")},
                ["instance_id"],
            ),
            _tool(
                "connect",
                "Wire src_node.src_port -> dst_node.dst_port. Port types must "
                "be compatible (interface-alignment is enforced).",
                {
                    "src_node": _str_prop(
                        "Source node_id (a snapshot key or a place_node return; "
                        "NOT a schema/type name)."
                    ),
                    "src_port": _str_prop("Output port name on the source."),
                    "dst_node": _str_prop(
                        "Target node_id (a snapshot key or a place_node return; "
                        "NOT a schema/type name)."
                    ),
                    "dst_port": _str_prop("Input port name on the target."),
                },
                ["src_node", "src_port", "dst_node", "dst_port"],
            ),
            _tool(
                "disconnect",
                "Remove an edge by its edge id.",
                {"edge_id": _str_prop("Edge id to remove.")},
                ["edge_id"],
            ),
            _tool(
                "set_param",
                "Set one parameter on a node. Value is validated against the "
                "node's param schema, registry enums, and reward hard bounds.",
                {
                    "node": _str_prop(
                        "Node_id (a snapshot key or a place_node return; NOT a "
                        "schema/type name)."
                    ),
                    "key": _str_prop("Parameter key."),
                    "value": {"description": "Parameter value (type per schema)."},
                },
                ["node", "key", "value"],
            ),
            _tool(
                "assign_reward_bundle",
                "Assign a set of reward terms to a command item. Enables the "
                "item, ensures a rewards node for that item, writes the terms to "
                "the chosen page, and wires reward_in__<item>. Call once per "
                "command item to differentiate rewards per item (composite is "
                "the normal shape). Use each reward's 'category'/'hint' and the "
                "'reward_doctrine' block in the vocabulary to reason "
                "dialectically — pair task rewards with their penalty "
                "counterparts and include posture terms where the task warrants.",
                {
                    "item_id": _enum_prop(
                        self.command_items, "Command item to assign rewards to."
                    ),
                    "terms": {
                        "type": "object",
                        "description": (
                            "Map of reward function name -> weight. Allowed "
                            f"reward names: {reward_names}. Weights must respect "
                            "each function's documented hard bounds."
                        ),
                    },
                    "page": {
                        "type": "string",
                        "enum": [PAGE_GLOBAL, *self.pd_group_pages],
                        "description": (
                            f"Reward page to write to. {PAGE_GLOBAL!r} (default) "
                            "for shared/global terms; a pd-group page name for a "
                            "joint-scoped reward."
                        ),
                    },
                },
                ["item_id", "terms"],
            ),
        ]

    def block_schema_ids(self, schema_ids: FrozenSet[str]) -> List[str]:
        """A block's schema_ids narrowed to those valid for the locked backend.

        ``self.node_ids`` is already backend-scoped, so intersecting drops a
        block's other-backend nodes (e.g. the SB3-only ``env_assembler`` inside
        the actor_env block on an isaac_lab run). Used to scope the block's
        placeable-node prompt + tool enum so the model is never even told about a
        node it must not place."""
        return sorted(set(schema_ids) & set(self.node_ids))

    def vocabulary_summary(self) -> Dict[str, Any]:
        """A compact dict of the closed vocabularies, for the agent prompt."""
        summary: Dict[str, Any] = {
            "backend": self.backend,
            "families": sorted(self.families),
            "algorithm_family": self.algorithm_family.value,
            "required_nodes": list(self.required_nodes),
            "node_ids": list(self.node_ids),
            "rewards": {
                n: {
                    "min": s.minimum,
                    "max": s.maximum,
                    "default": s.default,
                    "polarity": s.polarity,
                    # category + one-line design hint (what it pairs with / what
                    # degenerate optimum it guards against) so the agent reasons
                    # dialectically instead of from name/bounds alone.
                    **({"category": s.category} if s.category else {}),
                    **({"hint": s.hint} if s.hint else {}),
                }
                for n, s in sorted(self.rewards.items())
            },
            "terminations": {
                n: {"min": s.minimum, "max": s.maximum, "default": s.default}
                for n, s in sorted(self.terminations.items())
            },
            "obs_terms": list(self.obs_terms),
            "command_items": list(self.command_items),
            "init_poses": list(self.init_poses),
            "skus": list(self.skus),
            # Clip sub-ranges the user cut in the Clip Motion Editor. Reference a
            # segment on a training item's `clip` by its exact `ref`
            # (`<clip>#seg=lo-hi`); [] means none exist — do not fabricate one.
            "motion_segments": list(self.motion_segments),
            "reward_composition": self._reward_composition(),
            "reward_doctrine": self._reward_doctrine(),
        }
        # Skills (trigger command channels) are an IsaacLab-only training path —
        # SB3 has no trigger training and rejects gated packages. Only surface the
        # affordance on that backend so the agent never authors a dead SB3 skill.
        if self.backend == "isaac_lab":
            summary["skill_system"] = self._skill_system()
        return summary

    def _reward_composition(self) -> Dict[str, Any]:
        """Composite-reward affordance (blueprint §6 reward paging).

        Surfaces, LIVE from the registry: the reward pages available for the
        locked family (``__global__`` + the pd-group pages), that per-item
        assignment is the primary differentiation mechanism (referencing the
        existing ``command_items``), and that multiple rewards nodes are normal.
        A family with no pd_groups degrades cleanly to just the global page.
        """
        return {
            "pages": [PAGE_GLOBAL, *self.pd_group_pages],
            "global_page": PAGE_GLOBAL,
            "pd_group_pages": list(self.pd_group_pages),
            "per_item_differentiation": "primary",
            "differentiate_across": "command_items",
            "multiple_rewards_nodes": "normal",
            "note": (
                "Reward configuration is normally COMPOSITE, not one flat "
                "global bundle. Put shared terms on the "
                f"{PAGE_GLOBAL!r} page; give each enabled command item (see "
                "'command_items') its own reward bundle via "
                "assign_reward_bundle(item_id=..., page=...) to differentiate "
                "emphasis per item; use a pd-group page (see 'pd_group_pages') "
                "for a reward that should apply only to a joint partition. "
                "Multiple rewards nodes — one per command item — are normal."
            ),
        }

    def _reward_doctrine(self) -> Dict[str, Any]:
        """Reward-design DOCTRINE surfaced for the agent to REASON with.

        The three principles (dialectical / posture / local-optima review), the
        judgment caveat, and the degenerate-optima self-audit checklist — live
        from ``registers.reward_doctrine`` teaching data. Token-bounded (prose +
        edges, never docstrings). Advisory: it teaches, it never forces — the
        per-term ``category``/``hint`` in the ``rewards`` block above say what
        each term pairs with and guards against."""
        from registers import reward_doctrine

        return {
            "principles": reward_doctrine.principles(),
            "caveat": reward_doctrine.caveat(),
            "degenerate_optima": reward_doctrine.degenerate_optima(),
        }

    def _skill_system(self) -> Dict[str, Any]:
        """Skill (trigger command channel) affordance — how to author a discrete,
        on-demand behavior (jump / wave) the user triggers, authored on the
        ``training_motion`` node via ``skill_items`` + a gated ``packages`` entry.

        IsaacLab-only (the caller gates this on backend). Teaches WHAT a skill is,
        WHEN to use one, the exact set_param WORKFLOW, the training SEMANTICS, and
        the hard CONSTRAINTS — so the agent produces a valid skill with the tools it
        already has (skill authoring involves judgment — which reward, which items,
        the gating shape — so this teaches the shape rather than hardcoding an op).
        Binding the trigger to a key/button is a separate per-user Controller-panel
        action, NEVER authored on the canvas."""
        # Anchor the worked example to the live reward vocab (accurate per family).
        jump_reward = "jump_height" if "jump_height" in self.rewards else (
            "<a skill task reward from the 'rewards' vocab>"
        )
        return {
            "what": (
                "A skill is a DISCRETE trigger command channel (e.g. jump), as "
                "opposed to a continuous velocity command. It is a training package "
                "whose command interface is a TRIGGER: the trigger value enters the "
                "policy obs, the policy keeps ONE whole-body action space, and the "
                "package's reward is GATED on the trigger. It is one network "
                "observing a trigger — NOT a separate action head — because the "
                "physical coupling (a jump's torque disturbs balance; the same "
                "network must compensate) only exists when one policy owns the whole "
                "body and sees the trigger."
            ),
            "when_to_use": (
                "Use for an on-demand, discrete behavior the user fires (jump, "
                "wave). Do NOT model continuous locomotion as a skill — that is a "
                "velocity command_item. Trigger-only for now (enum reserved)."
            ),
            "authoring_workflow": [
                "1) Declare the trigger channel: set_param(<training_motion>, "
                "'skill_items', {'<skill_id>': {'name': '<Display>', 'kind': "
                "'trigger', 'latch': 'pulse', 'window_s': <seconds>, 'enabled': "
                "true}}). latch 'pulse' = a one-shot decaying pulse (the ONLY "
                "trained mode; 'hold' is reserved — do not use). window_s = the "
                "post-pulse window over which the trigger obs + gated reward decay.",
                "2) Create the GATED reward package: set_param(<training_motion>, "
                "'packages', {'<pkg_id>': {'package_id': '<pkg_id>', 'name': "
                "'<Display>', 'enabled': true, 'package_weight': 1.0, 'gated_by': "
                "'<skill_id>', 'reward_terms': {'__global__': {'<skill_reward>': "
                "{'weight': <w>}}}}}). gated_by binds the package's reward to the "
                "skill, so it is active ONLY inside the trigger window; pick the "
                "skill's task reward from the 'rewards' vocab.",
                "3) Assign member items: set_param(<training_motion>, "
                "'training_items', {'<item>': {..., 'package_id': '<pkg_id>'}}) — "
                "the training items during which the skill can be triggered "
                "(usually the locomotion items).",
            ],
            "training_semantics": (
                "The trigger enters the policy obs; the curriculum issues it with "
                "RANDOMIZED timing during rollouts (like gait commands today), so "
                "the policy learns the conditional mapping: perform the skill inside "
                "the trigger window, hold posture otherwise. ENV-GATED — the skill "
                "only responds after the policy is trained; a trigger the policy "
                "ignores is fake-success, not a finished skill."
            ),
            "constraints": [
                "IsaacLab-only: SB3 has no trigger training path and rejects gated "
                "packages — never author a skill on an sb3_mujoco canvas.",
                "Do NOT combine gait and a skill on the same canvas (they collide "
                "in the deploy obs layout) — choose one.",
                "gated_by MUST name an enabled skill in skill_items.",
                "The skill reward must be a name in the 'rewards' vocab within its "
                "hard bounds; design it DIALECTICALLY (a jump wants an apex/height "
                "reward AND a penalty basket + posture terms so the policy does not "
                "just leap and topple — see reward_doctrine).",
            ],
            "binding_note": (
                "Binding the trigger to a key/button is a PER-USER action in the "
                "Controller panel AFTER training — it is NOT part of the canvas "
                "config and you must not try to author it here."
            ),
            "example_jump": {
                "skill_items": {
                    "jump": {"name": "Jump", "kind": "trigger",
                             "latch": "pulse", "window_s": 1.0, "enabled": True},
                },
                "packages": {
                    "jump_pkg": {
                        "package_id": "jump_pkg", "name": "Jump", "enabled": True,
                        "package_weight": 1.0, "gated_by": "jump",
                        "reward_terms": {"__global__": {jump_reward: {"weight": 5.0}}},
                    },
                },
                "membership": (
                    "assign the walk / stand command items to 'jump_pkg' via "
                    "training_items[<item>].package_id = 'jump_pkg'"
                ),
            },
        }


# =============================================================================
# Builders
# =============================================================================

def build_projection(
    backend: str,
    families: FrozenSet[str],
    *,
    algorithm_family: Optional[AlgorithmFamily] = None,
) -> RegistryProjection:
    """Build the C2 projection for a backend + robot family set.

    Args:
        backend: ``"isaac_lab"`` | ``"sb3_mujoco"`` (the canvas backend id).
        families: the robot's family tokens (e.g. ``{"quadruped", "legged"}``)
            used to intersect against reward/termination ``applicable_families``.
        algorithm_family: the locked trainer family; defaults to the backend's
            PPO path.
    """
    if not backend:
        raise ValueError("build_projection: backend is required (fail-loud §8)")
    fam = algorithm_family or default_algorithm_family(backend)

    proj = RegistryProjection(
        backend=backend,
        families=frozenset(families),
        algorithm_family=fam,
        required_nodes=tuple(_REQUIRED_NODES.get(fam, ())),
    )
    proj.rewards = _project_rewards(backend, proj.families)
    proj.terminations = _project_terminations(backend, proj.families)
    proj.obs_terms = _project_obs_terms(backend, proj.families)
    proj.command_items = _project_command_items()
    proj.init_poses = _project_init_poses(proj.families)
    proj.skus = _project_skus()
    proj.node_ids = _project_node_ids(backend)
    proj.pd_group_pages = _project_reward_pages(proj.families)
    proj.motion_segments = _project_motion_segments()
    return proj


def build_projection_for_sku(
    backend: str,
    sku: str,
    *,
    algorithm_family: Optional[AlgorithmFamily] = None,
) -> RegistryProjection:
    """Build a projection resolving the family set from a robot SKU."""
    from registers import robots as _robots

    spec = _robots.get_robot_spec(sku)
    if spec is None:
        raise ValueError(
            f"build_projection_for_sku: unknown robot SKU {sku!r} (fail-loud §8)"
        )
    return build_projection(
        backend, frozenset(spec.families), algorithm_family=algorithm_family
    )


# =============================================================================
# Enumeration projectors (each reads ONE registry — no hand-copied vocab)
# =============================================================================

def _project_rewards(
    backend: str, families: FrozenSet[str]
) -> Dict[str, RewardSchema]:
    from registers import reward_doctrine
    from scripts.rewards.registry import IL_REWARD_REGISTRY, REWARD_REGISTRY

    src = IL_REWARD_REGISTRY if backend == "isaac_lab" else REWARD_REGISTRY
    out: Dict[str, RewardSchema] = {}
    for name, item in src.items():
        if backend not in item.backends:
            continue
        if families and not (item.applicable_families & families):
            continue
        out[name] = RewardSchema(
            name=name,
            minimum=float(item.min_value),
            maximum=float(item.max_value),
            default=float(item.default),
            polarity=str(item.polarity),
            # Restore the stripped design knowledge for dialectical reasoning.
            category=reward_doctrine.category_of(name, backend),
            hint=reward_doctrine.hint_of(name, backend) or "",
        )
    return out


def _project_terminations(
    backend: str, families: FrozenSet[str]
) -> Dict[str, TerminationSchema]:
    from scripts.terminations.registry import (
        IL_TERMINATION_REGISTRY,
        TERMINATION_REGISTRY,
    )

    src = IL_TERMINATION_REGISTRY if backend == "isaac_lab" else TERMINATION_REGISTRY
    out: Dict[str, TerminationSchema] = {}
    for name, item in src.items():
        if backend not in item.backends:
            continue
        if families and not (item.applicable_families & families):
            continue
        out[name] = TerminationSchema(
            name=name,
            minimum=float(item.min_value),
            maximum=float(item.max_value),
            default=float(item.default),
        )
    return out


def _project_obs_terms(backend: str, families: FrozenSet[str]) -> List[str]:
    # Obs terms are an IsaacLab registry (SB3 consumes a subset of the same
    # keys); the SB3 path has no separate obs registry, so we project the IL
    # keys for both and let the compile gate police SB3-uncomputable terms.
    from scripts.observations.registry import IL_OBS_REGISTRY

    out: List[str] = []
    for name, item in IL_OBS_REGISTRY.items():
        if families and not (item.applicable_families & families):
            continue
        out.append(name)
    return sorted(out)


def _project_command_items() -> List[str]:
    from registers import commands as _commands

    return sorted(item.id for item in _commands.list_items())


def _project_init_poses(families: FrozenSet[str]) -> List[str]:
    from registers import init_pose_presets as _ipp
    from registers.init_pose_presets import InitPosePresetValidationError

    names: List[str] = []
    for fam in sorted(families):
        try:
            presets = _ipp.list_presets(fam)
        except InitPosePresetValidationError:
            # This family token has no init-pose catalog entry (e.g. a generic
            # "legged" tag); try the next. Narrow catch — enumeration discovery,
            # not papering over a required input (§8).
            continue
        for p in presets:
            if p.name not in names:
                names.append(p.name)
    return names


def _project_skus() -> List[str]:
    from registers import robots as _robots

    return sorted(_robots.list_skus())


def _project_node_ids(backend: str) -> List[str]:
    from registers import nodes as _nodes

    # Backend-scoped: SB3-only nodes never reach an isaac_lab run (and vice
    # versa). Universal nodes (empty manifest.backends) pass for every backend.
    return sorted(m.id for m in _nodes.list_nodes(backend=backend))


def _project_reward_pages(families: FrozenSet[str]) -> List[str]:
    """Ordered pd-group reward-page keys available across the family set.

    Read LIVE from the pd_groups registry (never hand-listed): the ordered union
    of ``pd_groups.list_group_ids(family)`` over the family set, de-duplicated
    preserving first-seen order. Excludes ``__global__`` (always available; the
    composition view prepends it). A family with no pd_groups (e.g. ``generic``,
    or a bare ``legged`` tag) contributes nothing → ``[]``.
    """
    from registers import pd_groups as _pd

    out: List[str] = []
    for fam in sorted(families):
        for gid in _pd.list_group_ids(fam):
            if gid not in out:
                out.append(gid)
    return out


#: Prompt-size guards for segment discovery (segments are user-cut, normally
#: few; these caps only bite pathological libraries — a hit is logged, §8).
_MAX_SEG_CLIPS = 40
_MAX_SEG_PER_CLIP = 20


def _project_motion_segments() -> List[Dict[str, Any]]:
    """Enumerate user-cut clip SEGMENTS for the agent to reference.

    Mirrors the UI Clip picker (``registry_module_editor_panel._segment_options``):
    for each pickable motion file, list its saved segments and build the
    canonical ``<clip_ref>#seg=lo-hi`` ref. Grouped by parent clip; only clips
    that actually have segments are included (whole-clip enumeration is a
    separate concern and would bloat the prompt). Fail-SOFT: this is optional
    discovery context, so a provider hiccup yields ``[]`` and a WARN — the
    compile gate still validates any ref the agent emits, so no ref slips
    through unchecked. Caps are logged, never silent (§8).
    """
    from unitport_sdk import Paths

    # Segments are user state under USER_CONFIG_DIR — none can exist before the
    # install wizard establishes it, so skip the library scan entirely then.
    if not Paths.is_user_config_dir_set():
        return []
    try:
        from application.service.assets_browser import (
            get_asset_browser_provider,
            list_pickable_motion_entries,
        )
        from application.training.motion.segment_ref import build_segment_ref

        entries = list_pickable_motion_entries()
        provider = get_asset_browser_provider()
    except Exception as exc:  # noqa: BLE001 — optional discovery; stays loud in log
        log_warning(f"[ai_orchestration] motion-segment projection skipped: {exc!r}")
        return []

    out: List[Dict[str, Any]] = []
    for e in entries:
        if len(out) >= _MAX_SEG_CLIPS:
            log_warning(
                f"[ai_orchestration] motion-segment projection capped at "
                f"{_MAX_SEG_CLIPS} clips — some clips-with-segments omitted "
                f"from the agent vocabulary."
            )
            break
        ref = str(getattr(e, "path", "") or "")
        if not ref:
            continue
        try:
            segs = provider.list_segments(ref)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not blank all
            log_warning(
                f"[ai_orchestration] list_segments failed for {ref!r}: {exc!r}"
            )
            continue
        if not segs:
            continue
        if len(segs) > _MAX_SEG_PER_CLIP:
            log_warning(
                f"[ai_orchestration] clip {ref!r} has {len(segs)} segments — "
                f"showing first {_MAX_SEG_PER_CLIP} to the agent."
            )
        seg_out: List[Dict[str, Any]] = []
        for s in segs[:_MAX_SEG_PER_CLIP]:
            try:
                seg_ref = build_segment_ref(ref, int(s.start_frame), int(s.end_frame))
            except (ValueError, TypeError):
                continue
            entry = {
                "ref": seg_ref,
                "name": str(getattr(s, "name", "") or ""),
                "start": int(s.start_frame),
                "end": int(s.end_frame),
            }
            if getattr(s, "task_tag", ""):
                entry["tag"] = str(s.task_tag)
            seg_out.append(entry)
        if seg_out:
            out.append({
                "clip": ref,
                "name": str(getattr(e, "name", "") or ""),
                "segments": seg_out,
            })
    return out


# =============================================================================
# Small helpers
# =============================================================================

def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tool(
    name: str, description: str, properties: Dict[str, Any], required: List[str]
) -> Dict[str, Any]:
    """Build one OpenAI-style function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _enum_prop(values: List[str], description: str) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values), "description": description}


def _str_prop(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}
