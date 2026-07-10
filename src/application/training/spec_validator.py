# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.spec_validator — TrainingSpec semantic validation.

Companion to :mod:`application.training.training_spec` and
:mod:`application.training.spec_compiler`. Enforces the cross-cutting rules
**R1–R7** documented in ``MIGRATION_MAP.md``. Two-phase usage:

    1. **Lowering-time**: cheap structural checks that early-fail the
       compile (missing required nodes, dangling ports, ambiguous
       algorithm source). ``check_topology(ir, family)``.

    2. **Spec-time**: deep semantic checks against a populated
       :class:`TrainingSpec` (range / choice / cross-field). ``check_spec(spec)``.

Both phases produce typed :class:`ValidationIssue` objects. The compiler
collects issues and raises :class:`SpecValidationError` if any are
``Severity.ERROR``; warnings are returned alongside the spec for the UI to
surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from application.compiler.lowering import WorkflowIR
    from application.training.training_spec import TrainingSpec


# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(Enum):
    """Stable identifiers for UI presentation; values are stable across versions."""

    MISSING_REQUIRED_NODE = "missing_required_node"
    MISSING_REQUIRED_PORT = "missing_required_port"
    DANGLING_PORT = "dangling_port"
    OBS_OUTPUT_UNWIRED = "obs_output_unwired"
    AMBIGUOUS_ALGORITHM_SOURCE = "ambiguous_algorithm_source"
    BACKEND_ALGORITHM_MISMATCH = "backend_algorithm_mismatch"
    BACKEND_REGISTRY_MISMATCH = "backend_registry_mismatch"
    SIM_DT_CONFLICT = "sim_dt_conflict"
    UNKNOWN_PARAM_VALUE = "unknown_param_value"
    PARAM_OUT_OF_RANGE = "param_out_of_range"
    UNMAPPED_ACTION_JOINTS = "unmapped_action_joints"
    UNKNOWN_ROBOT = "unknown_robot"
    INCOMPLETE_AMP_WIRING = "incomplete_amp_wiring"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INVALID_JSON_PARAM = "invalid_json_param"
    INVALID_PARAM_TYPE = "invalid_param_type"
    UNRESOLVED_ROBOT_ASSET = "unresolved_robot_asset"
    INVALID_MOTION_CLIP = "invalid_motion_clip"
    MISSING_CONSUMPTION_MODE = "missing_consumption_mode"
    INVALID_CONSUMPTION_MODE = "invalid_consumption_mode"
    NON_IR_JOINT_NAME = "non_ir_joint_name"
    INVALID_COMMAND_TEMPLATE = "invalid_command_template"
    INVALID_HEADING_COMMAND = "invalid_heading_command"
    REWARD_TERM_CONFLICT = "reward_term_conflict"
    BASE_ASSET_UNRESOLVED = "base_asset_unresolved"
    STAGE_SCHEDULE_RESERVED_FOR_H2 = "stage_schedule_reserved_for_h2"
    CONTACT_SENSOR_COVERAGE_INSUFFICIENT = "contact_sensor_coverage_insufficient"
    CONTACT_SENSOR_BODY_DATA_MISSING = "contact_sensor_body_data_missing"
    TERRAIN_CURRICULUM_INVALID = "terrain_curriculum_invalid"
    CUSTOM_TERRAIN_INVALID = "custom_terrain_invalid"
    SB3_FEATURE_UNSUPPORTED = "sb3_feature_unsupported"
    SB3_PARALLELISM_INVALID = "sb3_parallelism_invalid"
    GENERIC = "generic"


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding."""

    code: IssueCode
    severity: Severity
    message: str
    node_id: str = ""
    field: str = ""

    def __str__(self) -> str:
        prefix = f"[{self.severity.value}:{self.code.value}]"
        loc = f" node={self.node_id}" if self.node_id else ""
        fld = f" field={self.field}" if self.field else ""
        return f"{prefix}{loc}{fld} {self.message}"


class SpecValidationError(RuntimeError):
    """Raised when validation produces any ``Severity.ERROR`` issue."""

    def __init__(self, issues: List[ValidationIssue]):
        self.issues = list(issues)
        msg = "TrainingSpec validation failed:\n  " + "\n  ".join(str(i) for i in issues)
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Family classification — needed by both lowering and spec checks
# ---------------------------------------------------------------------------

class AlgorithmFamily(Enum):
    """Classification of which trainer path the canvas drives.

    Decided by :func:`classify_family` from the IR alone (no spec needed).
    """

    SB3_PPO = "sb3_ppo"
    SB3_OFFPOLICY = "sb3_offpolicy"      # SAC / TD3
    IL_PPO = "il_ppo"
    IL_AMP_PPO = "il_amp_ppo"
    UNKNOWN = "unknown"


def classify_family(ir: "WorkflowIR") -> AlgorithmFamily:
    """Decide which algorithm family the canvas wires.

    Rules (R1):
      * If both ``algorithm_config + train`` AND ``il_ppo_trainer`` (or
        ``amp_trainer``) are present → ambiguous → UNKNOWN.
      * ``il_ppo_trainer.training_mode == "AMP_PPO"`` → IL_AMP_PPO.
      * ``amp_trainer`` present (any mode) → IL_AMP_PPO.
      * ``il_ppo_trainer`` present, mode PPO → IL_PPO.
      * ``algorithm_config.algorithm in {SAC, TD3}`` → SB3_OFFPOLICY.
      * ``algorithm_config.algorithm == PPO`` (with ``train`` sink) → SB3_PPO.
      * else UNKNOWN.
    """
    by_id = {n.schema_id: n for n in ir.nodes}
    has_il_trainer = "il_ppo_trainer" in by_id
    has_amp_trainer = "amp_trainer" in by_id
    has_sb3_train = "train" in by_id
    algo_node = by_id.get("algorithm_config")

    if (has_il_trainer or has_amp_trainer) and has_sb3_train and algo_node:
        return AlgorithmFamily.UNKNOWN  # ambiguous; caller will emit issue

    if has_amp_trainer:
        return AlgorithmFamily.IL_AMP_PPO
    if has_il_trainer:
        mode = _param_value(by_id["il_ppo_trainer"], "training_mode", "PPO")
        return AlgorithmFamily.IL_AMP_PPO if mode == "AMP_PPO" else AlgorithmFamily.IL_PPO

    if algo_node is not None:
        algo = _param_value(algo_node, "algorithm", "PPO")
        if algo in ("SAC", "TD3"):
            return AlgorithmFamily.SB3_OFFPOLICY
        return AlgorithmFamily.SB3_PPO

    return AlgorithmFamily.UNKNOWN


# ---------------------------------------------------------------------------
# Topology checks (lowering-time; cheap)
# ---------------------------------------------------------------------------

# Required nodes per family — without these the spec cannot be assembled
# (no trainer, no robot to drive, etc.). Missing one is a hard ERROR.
_REQUIRED_NODES: dict = {
    AlgorithmFamily.SB3_PPO: (
        "algorithm_config", "robot", "actor_setting",
        "rewards", "terminations", "obs_action_config",
        "env_assembler", "train",
    ),
    AlgorithmFamily.SB3_OFFPOLICY: (
        "algorithm_config", "robot", "actor_setting",
        "rewards", "terminations", "obs_action_config",
        "env_assembler", "train",
    ),
    AlgorithmFamily.IL_PPO: (
        "il_ppo_trainer", "robot", "actor_setting",
        "rewards", "terminations", "il_observation",
        "il_policy_network", "play_ground_setting",
    ),
    AlgorithmFamily.IL_AMP_PPO: (
        "il_ppo_trainer", "robot", "actor_setting",
        "rewards", "terminations", "il_observation",
        "il_policy_network", "play_ground_setting",
        "training_motion", "discriminator",
    ),
}

# Recommended nodes per family — populators have sensible defaults so the
# spec compiles fine without them, but the user is missing tunables that
# the canvas exposes. Missing one is a WARNING (visible in the cmd log,
# does not block submit). Legacy DEMO canvases that predate these
# stand-alone nodes still play through.
#
# IL families do not recommend physics_config / task_config: env_cfg_compiler
# reads sim_dt from play_ground_setting and hardcodes control_dt=0.02 (50 Hz),
# and the velocity-command schema lives on training_motion. Recommending these
# nodes on IL canvases used to lure users into setting values that the Isaac
# Lab path silently ignores.
_RECOMMENDED_NODES: dict = {
    AlgorithmFamily.SB3_PPO: ("physics_config", "task_config"),
    AlgorithmFamily.SB3_OFFPOLICY: ("physics_config", "task_config"),
    AlgorithmFamily.IL_PPO: (),
    AlgorithmFamily.IL_AMP_PPO: (),
}


def check_topology(ir: "WorkflowIR") -> List[ValidationIssue]:
    """Lowering-time IR checks. Cheap; no spec needed.

    Catches:
      * Required-node misses for the inferred algorithm family.
      * Ambiguous algorithm wiring (R1).
      * Required input ports that are unwired AND not gated off by a
        ``conditional_on`` meta whose host param disables them.
    """
    issues: List[ValidationIssue] = []

    family = classify_family(ir)
    by_id = {n.schema_id: n for n in ir.nodes}

    # R1: ambiguous trainer source
    if "algorithm_config" in by_id and "train" in by_id and (
        "il_ppo_trainer" in by_id or "amp_trainer" in by_id
    ):
        issues.append(ValidationIssue(
            code=IssueCode.AMBIGUOUS_ALGORITHM_SOURCE,
            severity=Severity.ERROR,
            message=(
                "canvas wires both SB3 path (algorithm_config + train) and "
                "Isaac-Lab trainer (il_ppo_trainer/amp_trainer); pick one"
            ),
        ))
        return issues  # downstream classification meaningless

    if family is AlgorithmFamily.UNKNOWN:
        issues.append(ValidationIssue(
            code=IssueCode.MISSING_REQUIRED_NODE,
            severity=Severity.ERROR,
            message=(
                "no recognized trainer wired; expected algorithm_config+train "
                "OR il_ppo_trainer OR amp_trainer"
            ),
        ))
        return issues

    for nid in _REQUIRED_NODES[family]:
        if nid not in by_id:
            issues.append(ValidationIssue(
                code=IssueCode.MISSING_REQUIRED_NODE,
                severity=Severity.ERROR,
                message=f"family={family.value} requires node id {nid!r}",
            ))

    for nid in _RECOMMENDED_NODES.get(family, ()):
        if nid not in by_id:
            issues.append(ValidationIssue(
                code=IssueCode.MISSING_REQUIRED_NODE,
                severity=Severity.WARNING,
                message=(
                    f"family={family.value} works without node id {nid!r} "
                    f"(dataclass defaults flow through), but adding it on the "
                    f"canvas exposes the tunables this node owns"
                ),
            ))

    issues.extend(_check_required_ports(ir))
    issues.extend(_check_il_observation_wired(ir))  # 缺口② — obs consumed by edge
    issues.extend(_check_training_item_reward_coverage(ir))  # §8 — enabled item w/o reward edge = 0 reward
    issues.extend(_check_reward_term_conflicts(ir))
    issues.extend(_check_reward_weight_bounds(ir))  # §8 — registry-declared weight bounds
    issues.extend(_check_base_asset_resolvable(ir))
    issues.extend(_check_legacy_dr_fields(ir))  # R_DR1 — Stage H migration
    issues.extend(_check_contact_sensor_coverage(ir))  # R_CONTACT1 — WYSIWYG sensor coverage
    issues.extend(_check_sb3_reward_partition(ir))      # 缺口③ — SB3 can't honor joint-paged rewards
    return issues


def _check_base_asset_resolvable(ir: "WorkflowIR") -> List[ValidationIssue]:
    """Reject loud when the v2 base_asset start_point token can't be
    resolved against on-disk artifacts:

      - ``__latest__`` + ``last_run_id`` empty / run dir missing /
        ``checkpoints/`` empty → user picked "continue from this canvas"
        but the canvas has nothing to continue from. Loud over silent —
        no fallback to scratch.
      - ``__load__`` + ``checkpoint_id`` empty / not the
        ``run:<abs>`` or ``export:<abs>`` grammar / referenced path
        does not exist on disk → user picked "load specific checkpoint"
        but didn't actually pick one (or picked one that has since
        been deleted).
    """
    from pathlib import Path as _P
    out: List[ValidationIssue] = []
    for n in ir.nodes:
        if n.schema_id != "base_asset":
            continue
        sp_p = n.params.get("start_point")
        sp = sp_p.value if sp_p is not None else None
        if sp == "__latest__":
            rid_p = n.params.get("last_run_id")
            rid = (rid_p.value if rid_p is not None else "") or ""
            err = _resolvable_latest_error(str(rid))
            if err:
                out.append(ValidationIssue(
                    code=IssueCode.BASE_ASSET_UNRESOLVED,
                    severity=Severity.ERROR,
                    node_id=n.id,
                    field="start_point",
                    message=(
                        f"base_asset/{n.id}: start_point='__latest__' "
                        f"but {err}; pick Load or start a fresh New run "
                        f"first"
                    ),
                ))
        elif sp == "__load__":
            cid_p = n.params.get("checkpoint_id")
            cid = (cid_p.value if cid_p is not None else "") or ""
            err = _checkpoint_id_error(str(cid))
            if err:
                out.append(ValidationIssue(
                    code=IssueCode.BASE_ASSET_UNRESOLVED,
                    severity=Severity.ERROR,
                    node_id=n.id,
                    field="checkpoint_id",
                    message=f"base_asset/{n.id}: {err}",
                ))
    return out


def _resolvable_latest_error(run_id: str) -> str:
    """Return a human error string when ``run_id`` can't resolve to an
    on-disk run with at least one .pt, or ``""`` when it resolves
    cleanly."""
    if not run_id:
        return "this canvas has no recorded run (last_run_id empty)"
    from application.service.training_assets import get_training_assets
    asset = get_training_assets().find_run(run_id)
    if asset is None or not asset.path.is_dir():
        return f"recorded run {run_id!r} no longer exists on disk"
    cps = asset.path / "checkpoints"
    if not cps.is_dir() or not any(cps.glob("*.pt")):
        return f"run {run_id!r} has no checkpoint files (.pt)"
    return ""


def _checkpoint_id_error(checkpoint_id: str) -> str:
    """Return a human error string when ``checkpoint_id`` isn't a
    resolvable ``run:<abs>`` or ``export:<abs>`` token, else ``""``."""
    from pathlib import Path as _P
    if not checkpoint_id:
        return "checkpoint_id is empty; pick a checkpoint via the Load picker"
    if checkpoint_id.startswith("run:"):
        path = checkpoint_id[4:]
    elif checkpoint_id.startswith("export:"):
        path = checkpoint_id[7:]
    else:
        return (
            f"checkpoint_id {checkpoint_id!r} doesn't match the expected "
            f"'run:<abs>' or 'export:<abs>' grammar"
        )
    if not path:
        return f"checkpoint_id {checkpoint_id!r} has empty path component"
    if not _P(path).is_file():
        return f"checkpoint file does not exist on disk: {path}"
    return ""


def _check_sb3_reward_partition(ir: "WorkflowIR") -> List[ValidationIssue]:
    """缺口③ — SB3 cannot honor per-joint-group (paged) reward scoping.

    A non-global reward page is a pd_group joint subset: IsaacLab emits a
    per-page ``SceneEntityCfg(joint_names=[...])`` so the reward applies only to
    that partition. The SB3 MuJoCo reward functions are joint-unaware — the
    spec_compiler flattens every page into one union — so a joint-paged reward
    would be applied to ALL joints, silently dropping the partition the user
    drew. Fail loud (§8) rather than train a policy whose reward scope differs
    from the canvas. Move such rewards to the global page, or train on IsaacLab.
    """
    out: List[ValidationIssue] = []
    if (getattr(ir, "backend", "") or "").strip() != "sb3_mujoco":
        return out
    import json as _json
    from application.compiler.term_payload import (
        is_paged_reward_terms, iter_reward_pages, PAGE_GLOBAL,
    )
    for node in ir.nodes:
        if node.schema_id != "rewards":
            continue
        terms_raw = _param_value(node, "reward_terms", {})
        if isinstance(terms_raw, str):
            try:
                terms_raw = _json.loads(terms_raw)
            except (ValueError, TypeError):
                continue
        if not isinstance(terms_raw, dict) or not is_paged_reward_terms(terms_raw):
            continue
        joint_pages = [
            pid for pid, pterms in iter_reward_pages(terms_raw)
            if pid != PAGE_GLOBAL and pterms
        ]
        if joint_pages:
            out.append(ValidationIssue(
                code=IssueCode.SB3_FEATURE_UNSUPPORTED,
                severity=Severity.ERROR,
                node_id=node.id,
                field="reward_terms",
                message=(
                    f"per-joint-group reward pages {joint_pages} are "
                    f"IsaacLab-only — the SB3 MuJoCo reward functions are "
                    f"joint-unaware and would apply these rewards to ALL "
                    f"joints, silently dropping the pd-group partition scope. "
                    f"Move these rewards to the global page (__global__), or "
                    f"train on IsaacLab."
                ),
            ))
    return out


def _check_reward_term_conflicts(ir: "WorkflowIR") -> List[ValidationIssue]:
    """Reject the compile when two rewards nodes share an item but disagree
    on a term value. With ``reward_in__<item_id>`` now ``multi=True``, the
    spec_compiler unions multiple rewards' ``reward_terms`` into one
    per-item dict; same-key+different-value would be a non-deterministic
    silent overwrite. Surface it here so the UI can also paint the affected
    term rows ``danger_zone`` in the rewards-node editor.

    A malformed ``reward_terms`` JSON blob on any rewards node raises
    ``ValueError`` upstream — translated here into a typed
    ``INVALID_JSON_PARAM`` issue rather than letting the validator crash
    or silently treat the node as empty.
    """
    from application.compiler.reward_conflicts import compute_reward_term_conflicts

    out: List[ValidationIssue] = []
    try:
        conflicts = compute_reward_term_conflicts(ir)
    except ValueError as exc:
        out.append(ValidationIssue(
            code=IssueCode.INVALID_JSON_PARAM,
            severity=Severity.ERROR,
            field="reward_terms",
            message=str(exc),
        ))
        return out
    for rewards_id, by_key in conflicts.items():
        for key, info in by_key.items():
            sources = sorted(info.values.keys())
            others = [s for s in sources if s != rewards_id]
            out.append(ValidationIssue(
                code=IssueCode.REWARD_TERM_CONFLICT,
                severity=Severity.ERROR,
                node_id=rewards_id,
                field=key,
                message=(
                    f"reward term {key!r} on item {info.item_id!r} is "
                    f"defined with conflicting values across rewards nodes "
                    f"{sources}; conflicts with {others}"
                ),
            ))
    return out


def _check_reward_weight_bounds(ir: "WorkflowIR") -> List[ValidationIssue]:
    """Reward weights MUST stay within the registry-declared ``[min_value,
    max_value]`` for each reward (CLAUDE.md §8: fail loud rather than train a
    policy whose reward is quietly broken).

    The reward registry is the single source of truth for sane bounds (e.g.
    ``joint_accel_penalty`` ∈ [-0.001, 0], ``joint_vel_penalty`` ∈ [-1.0, 0]).
    A canvas weight outside it is almost always a units/typo error that silently
    sabotages training: a joint-velocity / -acceleration penalty orders of
    magnitude too large (e.g. -2.5 vs the -0.001 cap → ~1e4× / vs -2.5e-7 default
    → ~1e7×) makes ANY joint motion catastrophically costly, so the optimal
    policy FREEZES its joints — it stands, kneels, barely steps, and never learns
    to walk, no matter how correct the PD / obs / gait are. Nothing enforced
    these bounds before, so such values reached training unnoticed. Widen the
    registry ``min_value`` / ``max_value`` (the single source) if a value outside
    the slider range is genuinely intended.
    """
    import json as _json
    from application.compiler.term_payload import (
        iter_reward_pages, parse_term_payload,
    )
    try:
        from scripts import REWARD_REGISTRY, IL_REWARD_REGISTRY
    except Exception:                                   # pragma: no cover
        return []
    out: List[ValidationIssue] = []
    for node in ir.nodes:
        if node.schema_id != "rewards":
            continue
        terms_raw = _param_value(node, "reward_terms", {})
        if isinstance(terms_raw, str):
            try:
                terms_raw = _json.loads(terms_raw)
            except (ValueError, TypeError):
                continue
        if not isinstance(terms_raw, dict):
            continue
        for _pid, terms in iter_reward_pages(terms_raw):
            if not isinstance(terms, dict):
                continue
            for func, payload in terms.items():
                item = IL_REWARD_REGISTRY.get(func) or REWARD_REGISTRY.get(func)
                if item is None:
                    continue  # unknown reward — coverage/registry checks handle it
                weight, _variant, _applies = parse_term_payload(payload)
                lo, hi = float(item.min_value), float(item.max_value)
                if weight < lo or weight > hi:
                    out.append(ValidationIssue(
                        code=IssueCode.PARAM_OUT_OF_RANGE,
                        severity=Severity.ERROR,
                        node_id=node.id,
                        field=func,
                        message=(
                            f"reward {func!r} weight={weight:g} is outside its "
                            f"valid range [{lo:g}, {hi:g}] (the reward registry "
                            f"is the single source of truth for sane bounds). An "
                            f"out-of-range penalty quietly breaks training: a "
                            f"joint-velocity/acceleration weight orders of "
                            f"magnitude too large makes joint motion "
                            f"catastrophically costly, so the policy freezes its "
                            f"joints and never walks. Set {func!r} within "
                            f"[{lo:g}, {hi:g}] on the Rewards node (registry "
                            f"default {float(item.default):g}), or widen the "
                            f"registry min/max if this is genuinely intended."
                        ),
                    ))

    # Method A (Slice 1c): the loop above validates RAW weights on wired
    # ``rewards`` nodes. Inline-package rewards never reach a wired node, so
    # validate their EFFECTIVE product (package_weight × term_weight) here — the
    # joint-freeze guard must see the product, not the raw term weight, or a
    # non-unit package weight could push an in-bound term past the registry bound
    # undetected. Reuse the single fold site (_fold_package_weight); no re-derive.
    tm = next((n for n in ir.nodes if n.schema_id == "training_motion"), None)
    if tm is not None:
        from application.compiler.term_payload import migrate_reward_terms_to_paged
        from application.training.package_synthesis import (
            _fold_package_weight,
            packages_from_param,
        )
        try:
            packages = packages_from_param(_param_value(tm, "packages", {}))
        except Exception as exc:                              # malformed authored data
            out.append(ValidationIssue(
                code=IssueCode.GENERIC,
                severity=Severity.ERROR,
                node_id=tm.id,
                field="packages",
                message=f"training_motion 'packages' param is malformed: {exc}",
            ))
            packages = {}
        for pid, pkg in packages.items():
            if not pkg.reward_terms:
                continue
            folded = _fold_package_weight(
                migrate_reward_terms_to_paged(pkg.reward_terms), pkg.package_weight
            )
            for _page, terms in folded.items():
                if not isinstance(terms, dict):
                    continue
                for func, payload in terms.items():
                    item = IL_REWARD_REGISTRY.get(func) or REWARD_REGISTRY.get(func)
                    if item is None:
                        continue
                    weight, _variant, _applies = parse_term_payload(payload)
                    lo, hi = float(item.min_value), float(item.max_value)
                    if weight < lo or weight > hi:
                        out.append(ValidationIssue(
                            code=IssueCode.PARAM_OUT_OF_RANGE,
                            severity=Severity.ERROR,
                            node_id=tm.id,
                            field=f"packages.{pid}.{func}",
                            message=(
                                f"package {pid!r} reward {func!r}: effective weight "
                                f"package_weight×term = {weight:g} is outside the "
                                f"registry range [{lo:g}, {hi:g}]. package_weight "
                                f"({float(pkg.package_weight):g}) scales every term of "
                                f"this package; a value orders of magnitude too large "
                                f"makes the policy freeze its joints. Lower "
                                f"package_weight or the term weight, or widen the "
                                f"registry min/max if genuinely intended."
                            ),
                        ))
    return out


def _check_il_observation_wired(ir: "WorkflowIR") -> List[ValidationIssue]:
    """缺口② — IL family: each il_observation node's ``obs_vector`` output MUST
    be wired to il_policy_network.

    The compiler now consumes the Observation node BY EDGE (not by type), so an
    unwired Observation node produces NO policy obs. Core canvas contract: a
    node must be wired to be used — fail loud (§8) instead of silently compiling
    an Observation node whose output drives nothing. (The old design discovered
    obs nodes by ``_find_by_type`` and compiled them regardless of wiring, which
    let a dangling 'critic' node look dead while still being emitted.)
    """
    out: List[ValidationIssue] = []
    if classify_family(ir) not in (AlgorithmFamily.IL_PPO, AlgorithmFamily.IL_AMP_PPO):
        return out
    obs_ids = [n.id for n in ir.nodes if n.schema_id == "il_observation"]
    pol_ids = {n.id for n in ir.nodes if n.schema_id == "il_policy_network"}
    if not obs_ids or not pol_ids:
        return out  # presence handled by _REQUIRED_NODES
    wired = {
        e.source_node for e in ir.edges
        if e.source_port == "obs_vector"
        and e.target_node in pol_ids
        and e.target_port == "obs_vector"
    }
    for oid in obs_ids:
        if oid not in wired:
            out.append(ValidationIssue(
                code=IssueCode.OBS_OUTPUT_UNWIRED,
                severity=Severity.ERROR,
                node_id=oid,
                field="obs_vector",
                message=(
                    "il_observation.obs_vector must be wired to il_policy_network "
                    "— an unwired Observation node silently produces no policy "
                    "obs. Connect its obs_vector output to the Policy Network "
                    "node (canvas contract: a node must be wired to be used)."
                ),
            ))
    return out


def _check_training_item_reward_coverage(ir: "WorkflowIR") -> List[ValidationIssue]:
    """An enabled training_item with NO ``reward_in__<item>`` edge silently
    earns ZERO reward for its entire env share.

    The command term samples every enabled item by its weight, but the reward
    masks (``unitport_item_mask``) only fire for item indices a Rewards node is
    wired to via ``reward_in__<item_id>``. An enabled-but-unwired item therefore
    has its whole fleet slice trained on a constant-0 reward — wasted compute
    that also poisons the critic (a 1/N slice of constant-0 value targets) and
    biases advantage normalisation. This is exactly how the G1 'Turn' item went
    unnoticed: ``_check_per_item_reward_scale`` drops zero-budget items before
    its skew check, so nothing saw the orphan. Fail loud (§8) — wire the item to
    a Rewards node or disable it. Mirrors :func:`_check_il_observation_wired`
    (a node/item must be wired to be used).
    """
    import json as _json

    out: List[ValidationIssue] = []
    if classify_family(ir) not in (AlgorithmFamily.IL_PPO, AlgorithmFamily.IL_AMP_PPO):
        return out
    tm_nodes = [n for n in ir.nodes if n.schema_id == "training_motion"]
    if not tm_nodes:
        return out
    for tm in tm_nodes:
        p = tm.params.get("training_items")
        if p is None:
            continue
        raw = getattr(p, "value", p)
        if isinstance(raw, str):
            try:
                items = _json.loads(raw) if raw.strip() else {}
            except (ValueError, TypeError):
                continue  # malformed JSON surfaced by other checks
        else:
            items = raw or {}
        if not isinstance(items, dict):
            continue
        enabled = [
            iid for iid, cfg in items.items()
            if not isinstance(cfg, dict) or cfg.get("enabled", True)
        ]
        if not enabled:
            continue
        wired = {
            str(e.target_port)[len("reward_in__"):]
            for e in ir.edges
            if e.target_node == tm.id and str(e.target_port).startswith("reward_in__")
        }
        for iid in enabled:
            if iid not in wired:
                out.append(ValidationIssue(
                    code=IssueCode.GENERIC,
                    severity=Severity.ERROR,
                    node_id=tm.id,
                    field=f"training_items.{iid}",
                    message=(
                        f"training_item {iid!r} is enabled (the command term samples "
                        f"it by weight) but has NO reward_in__{iid} edge from any "
                        f"Rewards node — its entire env share would train on a "
                        f"constant-0 reward (wasted compute + critic poisoning; a §8 "
                        f"silent failure). Fix: connect a Rewards node's reward_pipe "
                        f"to this item's reward_in__{iid} port, or disable the item."
                    ),
                ))
    return out


def _check_required_ports(ir: "WorkflowIR") -> List[ValidationIssue]:
    """Walk every node's manifest input ports; flag unwired non-optional
    ports unless their ``conditional_on`` meta disables them given current
    host params.
    """
    from registers import nodes as nodes_registry

    issues: List[ValidationIssue] = []

    incoming_keys = {(e.target_node, e.target_port) for e in ir.edges}

    for n in ir.nodes:
        cls = nodes_registry.get_node_class(n.schema_id)
        if cls is None:
            issues.append(ValidationIssue(
                code=IssueCode.MISSING_REQUIRED_NODE,
                severity=Severity.ERROR,
                node_id=n.id,
                message=f"unknown schema_id {n.schema_id!r}",
            ))
            continue
        manifest = cls.manifest()
        host_params = {k: p.value for k, p in n.params.items()}
        for port in manifest.inputs:
            if port.optional:
                continue
            if not _port_active(port, host_params):
                continue
            if (n.id, port.name) in incoming_keys:
                continue
            issues.append(ValidationIssue(
                code=IssueCode.MISSING_REQUIRED_PORT,
                severity=Severity.ERROR,
                node_id=n.id,
                field=port.name,
                message=(
                    f"required input port {port.name!r} is unwired "
                    f"(schema={n.schema_id})"
                ),
            ))
    return issues


def _port_active(port, host_params: dict) -> bool:
    """Return False when port.meta declares a ``conditional_on`` that
    evaluates false for the current host params."""
    meta = getattr(port, "meta", None)
    if not isinstance(meta, dict):
        return True
    cond = meta.get("conditional_on")
    if not isinstance(cond, dict):
        return True
    key = cond.get("key")
    op = cond.get("op")
    expected = cond.get("value")
    if not key or not op:
        return True
    actual = host_params.get(key)
    try:
        if op == "==": return actual == expected
        if op == "!=": return actual != expected
        if op == "in":
            seq = expected if isinstance(expected, (list, tuple, set)) else (expected,)
            return actual in seq
        if op == "not in":
            seq = expected if isinstance(expected, (list, tuple, set)) else (expected,)
            return actual not in seq
    except Exception:
        return True
    return True


# ---------------------------------------------------------------------------
# Spec checks (post-compile; deep)
# ---------------------------------------------------------------------------


def check_spec(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Deep semantic checks against a populated :class:`TrainingSpec`."""
    issues: List[ValidationIssue] = []
    issues.extend(_check_backend_algorithm(spec))
    issues.extend(_check_backend_registries(spec))
    issues.extend(_check_sim_dt(spec))
    issues.extend(_check_action_joints(spec))
    issues.extend(_check_amp_wiring(spec))
    issues.extend(_check_robot(spec))
    issues.extend(_check_joint_ir_canonical(spec))   # R8 — Phase 5 IR-only contract
    issues.extend(_check_init_joint_angles_in_range(spec))  # R_INIT1 — joint limit pre-flight
    issues.extend(_check_command_template_contract(spec))
    issues.extend(_check_heading_command(spec))      # heading-command sanity
    issues.extend(_check_recommended_reward_terms(spec))
    issues.extend(_check_sb3_reward_term_kinds(spec))
    issues.extend(_check_sb3_termination_kinds(spec))
    issues.extend(_check_training_packages(spec))    # R_PKG1 — package membership
    issues.extend(_check_skill_gating(spec))         # R_SKILL — skill gate → real skill
    issues.extend(_check_skill_gait_deploy_conflict(spec))  # R_SKILL2 — gait+skill deploy
    issues.extend(_check_per_item_reward_scale(spec))  # R_REWARD_SCALE
    issues.extend(_check_pd_param(spec))             # R_PD1..R_PD4 — sim2sim PD
    issues.extend(_check_stage_schedule(spec))       # R_STAGE_H0 — H0 default lock
    issues.extend(_check_recurrent_policy(spec))     # 缺口① — recurrent policy
    issues.extend(_check_policy_contract_fields(spec))  # B1 — policy arch integrity
    issues.extend(_check_terrain_curriculum(spec))   # 缺口④ — terrain curriculum
    issues.extend(_check_custom_terrain(spec))       # custom heightfield terrain
    issues.extend(_check_sb3_obs_terms(spec))         # 缺口⑤ P1/F3 — SB3 obs alignment
    issues.extend(_check_sb3_unsupported(spec))       # 缺口⑤ F1/F2/F5 — SB3 fail-loud gates
    issues.extend(_check_adaptive_sampling(spec))     # Phase C — adaptive item sampling bounds
    issues.extend(_check_item_weights(spec))          # Weight Set — initial sampling weights
    issues.extend(_check_sb3_parallelism(spec))       # SB3 VecEnv parallelism mode
    return issues


def _check_sb3_parallelism(spec: "TrainingSpec") -> List[ValidationIssue]:
    """SB3 ``env_assembler`` vectorisation settings (fail-loud, §8).

    ``parallelism_mode`` resolves to a concrete ``(n_envs, vec_type)`` at
    launch via :mod:`application.training.envs.auto_parallelism` — auto on the
    training machine's hardware, manual from the canvas values. This validates
    the canvas inputs that auto-resolution / manual pass-through can't fix:
      * an unknown ``parallelism_mode`` (not auto / manual);
      * in manual mode, an unknown ``vec_type`` or ``n_envs < 1`` (ERROR), and
        ``n_envs == 1`` (WARN — single-env robot RL trains very slowly).
    Auto mode is NOT count-validated here: it may legitimately resolve to a
    small n_envs on a weak machine (auto_parallelism logs that at launch).
    """
    out: List[ValidationIssue] = []
    backend = (getattr(getattr(spec, "algorithm", None), "backend", "") or "").lower()
    if backend not in _SB3_BACKENDS:
        return out
    env_cfg = getattr(spec, "env", None)
    if env_cfg is None:
        return out

    mode = (getattr(env_cfg, "parallelism_mode", "auto") or "auto").strip().lower()
    if mode not in ("auto", "manual"):
        out.append(ValidationIssue(
            code=IssueCode.SB3_PARALLELISM_INVALID,
            severity=Severity.ERROR,
            field="env_assembler.parallelism_mode",
            message=(
                f"parallelism_mode={mode!r} is not 'auto' or 'manual'. "
                f"'auto' adapts n_envs/vec_type to this machine's CPU+RAM; "
                f"'manual' uses the canvas n_envs/vec_type."
            ),
        ))
        return out

    if mode == "manual":
        n_envs = int(getattr(env_cfg, "n_envs", 8) or 8)
        vec_type = (getattr(env_cfg, "vec_type", "subproc") or "subproc").strip().lower()
        if vec_type not in ("dummy", "subproc"):
            out.append(ValidationIssue(
                code=IssueCode.SB3_PARALLELISM_INVALID,
                severity=Severity.ERROR,
                field="env_assembler.vec_type",
                message=(
                    f"vec_type={vec_type!r} is not 'dummy' or 'subproc' "
                    f"(manual parallelism_mode)."
                ),
            ))
        if n_envs < 1:
            out.append(ValidationIssue(
                code=IssueCode.SB3_PARALLELISM_INVALID,
                severity=Severity.ERROR,
                field="env_assembler.n_envs",
                message=f"n_envs={n_envs} must be >= 1 (manual parallelism_mode).",
            ))
        elif n_envs == 1:
            out.append(ValidationIssue(
                code=IssueCode.SB3_PARALLELISM_INVALID,
                severity=Severity.WARNING,
                field="env_assembler.n_envs",
                message=(
                    "n_envs=1 — robot RL trains very slowly with a single env. "
                    "Use parallelism_mode='auto' to size it to your hardware, "
                    "or set n_envs higher."
                ),
            ))
    return out


def _check_terrain_curriculum(spec: "TrainingSpec") -> List[ValidationIssue]:
    """缺口④ — terrain curriculum + sub-terrain mix integrity (fail-loud, §8).

    Guards the four ways the play_ground_setting terrain knobs can be wired
    into a config that silently does the wrong thing:
      1. curriculum_enabled on a flat scene — terrain_levels needs a
         ``generator`` terrain, so it would never engage.
      2. all sub-terrain proportions zero — the generator has nothing to
         build.
      3. max_init_terrain_level >= difficulty_levels — initial row is out of
         the generated grid.
      4. SB3 backend + rough/curriculum — the MuJoCo gym env has no terrain
         generator at all, so the setting would be silently dropped (the
         broader SB3-parity gate is deferred; this narrow one ships with the
         terrain feature so it cannot create a new "用户设置≠实际运行").
    """
    out: List[ValidationIssue] = []
    scene = getattr(spec, "scene", None)
    if scene is None:
        return out
    scene_type = (getattr(scene, "scene_type", "flat") or "flat").strip().lower()
    rough = getattr(scene, "rough", None)
    curriculum_enabled = bool(getattr(rough, "curriculum_enabled", False)) if rough is not None else False
    backend = (getattr(getattr(spec, "algorithm", None), "backend", "") or "").lower()
    is_rough = scene_type == "rough"

    # (4) SB3 cannot honour terrain / terrain curriculum.
    if backend in _SB3_BACKENDS and (is_rough or curriculum_enabled):
        out.append(ValidationIssue(
            code=IssueCode.TERRAIN_CURRICULUM_INVALID,
            severity=Severity.ERROR,
            field="play_ground_setting.scene_type",
            message=(
                "rough terrain / terrain curriculum is IsaacLab-only; the SB3 "
                "(MuJoCo) backend has no terrain generator and would silently "
                "run on flat ground. Switch the training backend to IsaacLab, "
                "or set scene_type='flat' (and disable curriculum)."
            ),
        ))
        return out

    # (1) curriculum on flat terrain never engages.
    if curriculum_enabled and not is_rough:
        out.append(ValidationIssue(
            code=IssueCode.TERRAIN_CURRICULUM_INVALID,
            severity=Severity.ERROR,
            field="play_ground_setting.curriculum_enabled",
            message=(
                "terrain curriculum requires a 'generator' terrain — set "
                "play_ground_setting.scene_type='rough' or disable "
                "curriculum_enabled."
            ),
        ))

    if is_rough and rough is not None:
        # (2) at least one sub-terrain must have positive weight.
        props = getattr(rough, "proportions", {}) or {}
        if sum(float(v) for v in props.values()) <= 0.0:
            out.append(ValidationIssue(
                code=IssueCode.TERRAIN_CURRICULUM_INVALID,
                severity=Severity.ERROR,
                field="play_ground_setting.prop_*",
                message=(
                    "all rough sub-terrain proportions are zero — the terrain "
                    "generator has nothing to build. Give at least one "
                    "sub-terrain a positive proportion."
                ),
            ))
        # (3) initial difficulty row must be inside the generated grid.
        diff = int(getattr(rough, "difficulty_levels", 0) or 0)
        max_init = int(getattr(rough, "max_init_terrain_level", 0) or 0)
        if diff > 0 and max_init >= diff:
            out.append(ValidationIssue(
                code=IssueCode.TERRAIN_CURRICULUM_INVALID,
                severity=Severity.ERROR,
                field="play_ground_setting.max_init_terrain_level",
                message=(
                    f"max_init_terrain_level ({max_init}) must be < "
                    f"difficulty_levels ({diff}) — the initial row would be "
                    f"outside the generated terrain grid."
                ),
            ))
    return out


def _check_custom_terrain(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Custom heightfield terrain integrity (fail-loud, §8).

    Guards canvas-time the ways ``scene_type='custom'`` can be wired into a
    silently-wrong run. Both engines DO support custom heightfields (the
    MuJoCo <hfield> + IsaacLab terrain-function lowerings), so — unlike
    rough — custom is NOT SB3-rejected.

      1. scene_type='custom' but no terrain configured (enabled False /
         empty source_path) — nothing to build.
      2. custom + terrain curriculum — curriculum is deferred for custom
         terrain (施工规划 v2 §1 / PV 修正 3); a single fixed tile cannot
         drive a difficulty grid, so reject the combination loudly rather
         than silently ignore the curriculum flag.
      3. unknown source_format — not a registered terrain loader.
      4. source file missing on disk.

    Deep structural validity (2-D, finite, >=2x2, square cells for
    IsaacLab) is enforced fail-loud at the loader / lowering boundary
    (``validate_terrain_contract`` + ``heightfield_to_isaaclab``) when the
    asset is actually read at compile time — kept out of the validator so
    it does not load a megapixel array on every validate pass.
    """
    out: List[ValidationIssue] = []
    scene = getattr(spec, "scene", None)
    if scene is None:
        return out
    scene_type = (getattr(scene, "scene_type", "flat") or "flat").strip().lower()
    if scene_type != "custom":
        return out

    custom = getattr(scene, "custom", None)
    src = str(getattr(custom, "source_path", "") or "").strip() if custom else ""

    # (1) custom selected but nothing configured.
    if custom is None or not bool(getattr(custom, "enabled", False)) or not src:
        out.append(ValidationIssue(
            code=IssueCode.CUSTOM_TERRAIN_INVALID,
            severity=Severity.ERROR,
            field="play_ground_setting.custom_terrain_path",
            message=(
                "scene_type='custom' but no heightfield is imported. Import a "
                "terrain height-map (.npz / .npy / .png) on the Play Ground "
                "Setting node, or switch scene_type to 'flat' / 'rough'."
            ),
        ))
        return out

    # (2) custom + curriculum — deferred this phase.
    rough = getattr(scene, "rough", None)
    if rough is not None and bool(getattr(rough, "curriculum_enabled", False)):
        out.append(ValidationIssue(
            code=IssueCode.CUSTOM_TERRAIN_INVALID,
            severity=Severity.ERROR,
            field="play_ground_setting.curriculum_enabled",
            message=(
                "terrain curriculum is not supported for custom terrain (a "
                "single imported tile has no difficulty grid). Disable "
                "curriculum_enabled, or use scene_type='rough' for a "
                "curriculum-driven generator."
            ),
        ))

    # (3) format must be a registered terrain loader.
    fmt = str(getattr(custom, "source_format", "") or "").strip()
    try:
        from application.training.terrain.loaders import list_loader_formats
        known = list_loader_formats()
    except Exception:  # noqa: BLE001 - terrain pkg import failure surfaces below
        known = []
    if fmt and known and fmt not in known:
        out.append(ValidationIssue(
            code=IssueCode.CUSTOM_TERRAIN_INVALID,
            severity=Severity.ERROR,
            field="play_ground_setting.custom_terrain_format",
            message=(
                f"custom terrain source_format={fmt!r} is not a known terrain "
                f"loader. Known: {known!r}."
            ),
        ))

    # (4) source file must exist on disk.
    try:
        from pathlib import Path as _P
        exists = _P(src).is_file()
    except Exception:  # noqa: BLE001
        exists = False
    if not exists:
        out.append(ValidationIssue(
            code=IssueCode.CUSTOM_TERRAIN_INVALID,
            severity=Severity.ERROR,
            field="play_ground_setting.custom_terrain_path",
            message=(
                f"custom terrain file not found: {src!r}. Re-import the "
                f"height-map or fix the path."
            ),
        ))
    return out


# The SB3 backend has exactly ONE id. No "sb3" kind, no tolerant fallback.
_SB3_BACKENDS = ("sb3_mujoco",)


def _check_sb3_obs_terms(spec: "TrainingSpec") -> List[ValidationIssue]:
    """缺口⑤ P1/F3 — SB3 observation-term alignment (fail-loud, §8).

    The SB3 (MuJoCo gym) path now assembles its observation through the
    shared obs_term_engine, term-by-term from ``obs_action.il_terms`` — the
    same layout the deploy ObsBuilder replays, so "train == deploy" holds by
    construction. Terms the engine cannot compute from a MuJoCo state alone
    (``height_scan`` needs a ray-caster the gym env does not have;
    reference-motion / phase terms need an AMP adapter) would be silently
    dropped by the old hardcoded layout. Reject them here so the operator
    re-wires instead of training a policy whose obs contract it can never
    satisfy at deploy.
    """
    out: List[ValidationIssue] = []
    backend = (getattr(getattr(spec, "algorithm", None), "backend", "") or "").lower()
    if backend not in _SB3_BACKENDS:
        return out
    obs_action = getattr(spec, "obs_action", None)
    il_terms = getattr(obs_action, "il_terms", None) if obs_action is not None else None
    if not il_terms or not isinstance(il_terms, dict):
        return out  # empty → default proprio layout (all engine terms)

    from application.training.envs import obs_term_engine as _obs_eng

    for term_name in il_terms:
        if not _obs_eng.is_engine_term(str(term_name)):
            out.append(ValidationIssue(
                code=IssueCode.SB3_FEATURE_UNSUPPORTED,
                severity=Severity.ERROR,
                field="il_observation.obs_terms",
                message=(
                    f"observation term {term_name!r} cannot be computed by the "
                    f"SB3 (MuJoCo) backend — only proprioceptive terms "
                    f"({sorted(_obs_eng.ENGINE_TERMS)}) are supported. "
                    f"height_scan needs a ray-caster and reference/phase terms "
                    f"need an AMP adapter, both IsaacLab-only. Remove the term "
                    f"or switch the training backend to IsaacLab."
                ),
            ))
    return out


def _check_sb3_unsupported(spec: "TrainingSpec") -> List[ValidationIssue]:
    """缺口⑤ F1/F2/F5 — features the SB3 (MuJoCo gym) backend cannot honour.

    The SB3 path now genuinely consumes everything the MuJoCo engine can do
    (obs terms, PD, domain rand, terminations, command, multi-stage). The
    remainder is IsaacLab/rsl_rl-only and used to be silently dropped or
    "warn-and-continue" (the CLAUDE.md §8 anti-pattern the user called out).
    Each now fails loud at validation so the operator switches backend or
    removes the feature — never trains a policy whose config the runtime
    silently ignores.

      * F1 AMP / motion clips — the discriminator + reference-motion replay
        live in rsl_rl's AMPOnPolicyRunner; SB3 only has PPO/SAC/TD3. (The
        ``AMP_PPO`` training_mode is already gated by
        :func:`_check_backend_algorithm`; this also catches a wired
        ``motion_ref`` left on a PPO canvas.)
      * F2 gait — the Walk-These-Ways gait command/obs/rewards are emitted
        only on the IsaacLab side.
      * F5 RSI — ``init_pose.mode == "reference_frame_0"`` needs reference
        motion clips, same dependency as AMP.

    F3 (height_scan) is handled per-term in :func:`_check_sb3_obs_terms`;
    F4 (rough terrain / terrain curriculum) in :func:`_check_terrain_curriculum`.
    """
    out: List[ValidationIssue] = []
    backend = (getattr(getattr(spec, "algorithm", None), "backend", "") or "").lower()
    if backend not in _SB3_BACKENDS:
        return out

    # F6 — skill / trigger-gated packages (skill_command_path_design.md Slice 3)
    # are IsaacLab-only: the trigger command term + obs injection + reward gate
    # emit only on the IsaacLab side. Fail loud rather than train a package whose
    # trigger gate the MuJoCo runtime silently ignores (§8).
    _motion = getattr(spec, "motion", None)
    _gated = sorted(
        pid for pid, p in (getattr(_motion, "packages", None) or {}).items()
        if str(getattr(p, "gated_by", "") or "").strip()
    )
    if _gated:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.packages",
            message=(
                f"skill/trigger-gated packages {_gated} are IsaacLab-only — the "
                f"trigger command term, obs injection, and reward gate emit only on "
                f"the IsaacLab backend. Switch this canvas to IsaacLab or remove the "
                f"gated_by link (§8 fail-loud)."
            ),
        ))

    # P8 gate — the MuJoCo gym env only implements PD joint-position control.
    # A non-joint_position action_type used to warn-and-continue (running PD
    # anyway); fail loud so the operator's torque/velocity choice isn't
    # silently ignored.
    for _src, _obj in (
        ("obs_action.action_type", getattr(spec, "obs_action", None)),
        ("physics.action_type", getattr(spec, "physics", None)),
    ):
        at = (getattr(_obj, "action_type", "joint_position") or "joint_position").lower()
        if at not in ("joint_position", "position"):
            out.append(ValidationIssue(
                code=IssueCode.SB3_FEATURE_UNSUPPORTED,
                severity=Severity.ERROR,
                field=_src,
                message=(
                    f"action_type={at!r} is not supported by the SB3 backend — "
                    f"the MuJoCo gym env only runs PD joint-position control. "
                    f"Set action_type='joint_position', or switch to IsaacLab."
                ),
            ))

    # F1 — AMP / motion clips wired (independent of training_mode).
    il = getattr(spec, "il", None)
    motion_ref = getattr(il, "motion_ref", None) if il is not None else None
    if motion_ref is not None and getattr(motion_ref, "clip_paths", None):
        out.append(ValidationIssue(
            code=IssueCode.SB3_FEATURE_UNSUPPORTED,
            severity=Severity.ERROR,
            field="il.motion_ref",
            message=(
                "AMP / reference-motion clips are IsaacLab/rsl_rl-only — the "
                "SB3 backend has no discriminator or motion-replay runner and "
                "would silently train a plain PPO policy ignoring the clips. "
                "Switch the training backend to IsaacLab, or remove the "
                "TrainingMotion / AMP wiring."
            ),
        ))

    # F2 — gait (Walk These Ways) command/obs/rewards.
    gait = getattr(getattr(spec, "motion", None), "gait", None)
    if gait is not None and bool(getattr(gait, "enabled", False)):
        out.append(ValidationIssue(
            code=IssueCode.SB3_FEATURE_UNSUPPORTED,
            severity=Severity.ERROR,
            field="motion.gait.enabled",
            message=(
                "gait (Walk-These-Ways phase command / obs / rewards) is not "
                "supported by the SB3 backend yet — it would be silently "
                "dropped. Switch to IsaacLab, or disable gait on the "
                "TrainingMotion node."
            ),
        ))

    # F3 — height-scan sensor (ray-caster) has no MuJoCo gym equivalent. The
    # per-term form is caught in _check_sb3_obs_terms; this catches the scene
    # sensor being enabled on the play-ground node.
    height_scan = getattr(getattr(spec, "scene", None), "height_scan", None)
    if height_scan is not None and bool(getattr(height_scan, "enabled", False)):
        out.append(ValidationIssue(
            code=IssueCode.SB3_FEATURE_UNSUPPORTED,
            severity=Severity.ERROR,
            field="scene.height_scan.enabled",
            message=(
                "height_scan (terrain ray-caster) is IsaacLab-only — the SB3 "
                "backend has no ray-caster sensor. Disable height_scan, or "
                "switch the training backend to IsaacLab."
            ),
        ))

    # F5 — RSI init pose needs reference motion clips.
    init_pose = getattr(getattr(spec, "actor", None), "init_pose", None)
    mode = (getattr(init_pose, "mode", "default") or "default").strip().lower()
    if mode == "reference_frame_0":
        out.append(ValidationIssue(
            code=IssueCode.SB3_FEATURE_UNSUPPORTED,
            severity=Severity.ERROR,
            field="actor.init_pose.mode",
            message=(
                "init_pose.mode='reference_frame_0' (RSI) needs reference "
                "motion clips, which the SB3 backend cannot replay. Use "
                "'default' / 'keyframe' / 'custom', or switch to IsaacLab."
            ),
        ))

    # F6 — adaptive item sampling lives in the IsaacLab command term
    # (UnitportWeightedVelocityCommand). The SB3 MuJoCo env runs its own
    # command sampler with no per-item weight re-biasing, so an enabled
    # adaptive curriculum would be silently ignored.
    motion = getattr(spec, "motion", None)
    if motion is not None and bool(getattr(motion, "adaptive_motion_enabled", False)):
        out.append(ValidationIssue(
            code=IssueCode.SB3_FEATURE_UNSUPPORTED,
            severity=Severity.ERROR,
            field="motion.adaptive_motion_enabled",
            message=(
                "adaptive item sampling is IsaacLab-only — the SB3 MuJoCo env "
                "has no weighted multi-item command term to re-bias. Disable "
                "adaptive_motion_enabled on the TrainingMotion node, or switch "
                "the training backend to IsaacLab."
            ),
        ))
    return out


def _check_adaptive_sampling(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Phase C — adaptive item-sampling field sanity (fail-loud, §8).

    Only fires when ``motion.adaptive_motion_enabled``. Guards the bounds the
    env-side re-weighter assumes: a usable [floor, ceil] window, a positive
    update cadence, and at least two enabled training items (one item makes
    re-weighting a no-op — surfacing it prevents a silent "curriculum that
    never does anything").
    """
    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None or not bool(getattr(motion, "adaptive_motion_enabled", False)):
        return out

    floor = float(getattr(motion, "adaptive_weight_floor", 0.03))
    ceil = float(getattr(motion, "adaptive_weight_ceil", 0.30))
    interval = int(getattr(motion, "adaptive_update_interval", 50))
    items = getattr(motion, "training_items", {}) or {}
    n_enabled = sum(
        1 for v in items.values()
        if isinstance(v, dict) and v.get("enabled", False)
    )

    if not (0.0 <= floor < ceil <= 1.0):
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.adaptive_weight_floor",
            message=(
                f"adaptive sampling needs 0 <= weight_floor < weight_ceil <= 1, "
                f"got floor={floor}, ceil={ceil}."
            ),
        ))
    # Feasibility: every item clamped to >= floor must be able to coexist.
    if n_enabled > 0 and floor * n_enabled > 1.0:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.adaptive_weight_floor",
            message=(
                f"weight_floor={floor} × {n_enabled} enabled items = "
                f"{floor * n_enabled:.2f} > 1.0 — the per-item floor cannot be "
                f"satisfied. Lower weight_floor to <= {1.0 / n_enabled:.3f}."
            ),
        ))
    if interval < 1:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.adaptive_update_interval",
            message=(
                f"adaptive_update_interval must be >= 1 iteration, got {interval}."
            ),
        ))
    if n_enabled < 2:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.WARNING,
            field="motion.adaptive_motion_enabled",
            message=(
                f"adaptive item sampling is enabled but only {n_enabled} training "
                f"item is enabled — re-weighting needs >= 2 items to do anything. "
                f"Enable more TrainingMotion items or turn adaptive sampling off."
            ),
        ))
    return out


def _check_item_weights(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Weight Set — per-item initial sampling weight sanity (fail-loud, §8).

    The Weight Set pie editor stores an optional ``weight`` on each
    ``training_items[id]``. Both engines (Isaac-Lab ``initial_weights`` and the
    SB3 ``_cmd_item_weights``) read it, default a missing weight to the uniform
    share, then normalise. This guards the only ways the value can be malformed
    so it never silently normalises into garbage:

      * a negative or NaN/inf weight on an enabled item → ERROR (a typo /
        bad hand-edit; normalising it would produce a nonsense distribution),
      * every enabled item explicitly weighted 0 → ERROR (no item could ever
        be sampled; the uniform fallback would mask the user's intent).

    A weight that is simply *absent* is fine (uniform share). Weights on
    DISABLED items are ignored (the item never enters either engine's list),
    so they are not validated — disabling an item parks its weight harmlessly.
    """
    import math

    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None:
        return out
    items = getattr(motion, "training_items", {}) or {}
    if not isinstance(items, dict):
        return out

    enabled_weights: List[float] = []
    for item_id, cfg in items.items():
        if not (isinstance(cfg, dict) and cfg.get("enabled")):
            continue
        if "weight" not in cfg:
            continue
        raw = cfg.get("weight")
        try:
            w = float(raw)
        except (TypeError, ValueError):
            out.append(ValidationIssue(
                code=IssueCode.GENERIC,
                severity=Severity.ERROR,
                field=f"motion.training_items.{item_id}.weight",
                message=(
                    f"training item {item_id!r} has a non-numeric weight "
                    f"{raw!r}; weight must be a number >= 0 (or unset for the "
                    f"uniform share)."
                ),
            ))
            continue
        if math.isnan(w) or math.isinf(w) or w < 0.0:
            out.append(ValidationIssue(
                code=IssueCode.GENERIC,
                severity=Severity.ERROR,
                field=f"motion.training_items.{item_id}.weight",
                message=(
                    f"training item {item_id!r} has an invalid sampling weight "
                    f"{w}; weight must be a finite number >= 0 (or unset for "
                    f"the uniform share)."
                ),
            ))
            continue
        enabled_weights.append(w)

    # All enabled items explicitly weighted 0 → nothing could be sampled.
    if enabled_weights and sum(enabled_weights) <= 0.0:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.training_items",
            message=(
                "every enabled training item is weighted 0 — no command item "
                "could ever be sampled. Give at least one enabled item a "
                "positive Weight Set value (or leave them unset for uniform)."
            ),
        ))
    return out


def _check_recurrent_policy(spec: "TrainingSpec") -> List[ValidationIssue]:
    """缺口① — recurrent policy field integrity (fail-loud, §8).

    The il_policy_network enum already constrains ``rnn_type`` at the UI, and
    the compiler coerces; this is the belt-and-suspenders spec-level guard so a
    hand-edited / programmatically-built spec can't carry a bad recurrent
    config into training. ``rnn_type='none'`` (MLP) skips all checks.
    """
    out: List[ValidationIssue] = []
    pn = getattr(getattr(spec, "algorithm", None), "policy_net", None)
    if pn is None:
        return out
    rnn_type = (getattr(pn, "rnn_type", "none") or "none").strip().lower()
    if rnn_type not in ("none", "gru", "lstm"):
        out.append(ValidationIssue(
            code=IssueCode.UNKNOWN_PARAM_VALUE,
            severity=Severity.ERROR,
            field="rnn_type",
            message=(
                f"il_policy_network.rnn_type={rnn_type!r} is invalid; expected "
                f"one of 'none' / 'gru' / 'lstm'."
            ),
        ))
        return out
    if rnn_type != "none":
        if int(getattr(pn, "rnn_hidden_size", 0)) <= 0:
            out.append(ValidationIssue(
                code=IssueCode.UNKNOWN_PARAM_VALUE,
                severity=Severity.ERROR,
                field="rnn_hidden_size",
                message=(
                    f"il_policy_network.rnn_hidden_size must be a positive int "
                    f"when rnn_type={rnn_type!r} (got "
                    f"{getattr(pn, 'rnn_hidden_size', None)!r})."
                ),
            ))
        if int(getattr(pn, "rnn_num_layers", 0)) <= 0:
            out.append(ValidationIssue(
                code=IssueCode.UNKNOWN_PARAM_VALUE,
                severity=Severity.ERROR,
                field="rnn_num_layers",
                message=(
                    f"il_policy_network.rnn_num_layers must be a positive int "
                    f"when rnn_type={rnn_type!r} (got "
                    f"{getattr(pn, 'rnn_num_layers', None)!r})."
                ),
            ))
    return out


# Activation set accepted by either backend's network builder (union of
# sb3_trainer._activation_fn and isaac_lab.onnx_export._activation_module) — the
# validator rejects only genuinely-unknown values, not backend-specific subsets.
_VALID_POLICY_ACTIVATIONS = frozenset(
    {"elu", "relu", "tanh", "leaky_relu", "selu", "gelu", "silu", "swish", "sigmoid"}
)


def _check_policy_contract_fields(spec: "TrainingSpec") -> List[ValidationIssue]:
    """B1 — policy network arch integrity (fail-loud, §8).

    The arch feeds the deploy bundle's ``policy_contract`` snapshot; a garbage
    arch (empty hidden dims, non-positive noise std, unknown activation) must be
    caught at the spec level so it never reaches a shipped bundle. Mirrors
    ``_check_recurrent_policy`` (which owns the recurrent fields).
    """
    out: List[ValidationIssue] = []
    pn = getattr(getattr(spec, "algorithm", None), "policy_net", None)
    if pn is None:
        return out

    for field_name in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = getattr(pn, field_name, None)
        ok = isinstance(dims, (list, tuple)) and len(dims) > 0
        if ok:
            try:
                ok = all(int(d) > 0 for d in dims)
            except (TypeError, ValueError):
                ok = False
        if not ok:
            out.append(ValidationIssue(
                code=IssueCode.PARAM_OUT_OF_RANGE,
                severity=Severity.ERROR,
                field=field_name,
                message=(
                    f"il_policy_network.{field_name} must be a non-empty list of "
                    f"positive ints (got {dims!r})."
                ),
            ))

    init_std = getattr(pn, "init_noise_std", None)
    try:
        init_std_f = float(init_std)
    except (TypeError, ValueError):
        init_std_f = -1.0
    if init_std_f <= 0.0:
        out.append(ValidationIssue(
            code=IssueCode.PARAM_OUT_OF_RANGE,
            severity=Severity.ERROR,
            field="init_noise_std",
            message=(
                f"il_policy_network.init_noise_std must be a positive float "
                f"(direct action std, never exp()'d); got {init_std!r}."
            ),
        ))

    activation = str(getattr(pn, "activation", "") or "").strip().lower()
    if activation and activation not in _VALID_POLICY_ACTIVATIONS:
        out.append(ValidationIssue(
            code=IssueCode.UNKNOWN_PARAM_VALUE,
            severity=Severity.ERROR,
            field="activation",
            message=(
                f"il_policy_network.activation={activation!r} is unknown; expected "
                f"one of {sorted(_VALID_POLICY_ACTIVATIONS)}."
            ),
        ))
    return out


def _check_stage_schedule(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_STAGE_H0 — stage_schedule H0 default lock + schema_version contract.

    Delegates to :func:`training_spec.validate_stage_schedule_dict_h0`,
    which fail-loud raises with the three-part directive. Any raise
    converts to a single ``STAGE_SCHEDULE_RESERVED_FOR_H2`` issue so
    validation surfaces it uniformly alongside other R-rules.

    The dataclass populated by :func:`spec_compiler._populate_stage_schedule`
    always carries ``schema_version=STAGE_SCHEDULE_SCHEMA_VERSION`` and
    the populate-time helper already runs :func:`validate_stage_entry_h0`
    on each raw canvas entry; this check is the defensive in-memory
    second gate (after mutation through code paths that bypass populate).
    """
    from dataclasses import asdict
    from application.training.training_spec import (
        validate_stage_schedule_dict_h0,
    )

    out: List[ValidationIssue] = []
    sched = getattr(spec, "stage_schedule", None)
    if sched is None:
        return out
    try:
        validate_stage_schedule_dict_h0(asdict(sched))
    except ValueError as exc:
        out.append(ValidationIssue(
            code=IssueCode.STAGE_SCHEDULE_RESERVED_FOR_H2,
            severity=Severity.ERROR,
            field="stage_schedule",
            message=str(exc),
        ))
    return out


def _check_legacy_dr_fields(ir: "WorkflowIR") -> List[ValidationIssue]:
    """R_DR1 — surface legacy motor_strength_range / joint_damping_range.

    Stage H of the sim2sim PD framework deleted these two DR knobs in
    favor of ``omega_n_log_uniform`` / ``zeta_log_uniform``. Old
    canvases save them verbatim — silently dropping them would mask the
    user's expectation that "their DR was still applied"; loudly
    pointing at the renamed field is the only acceptable behavior
    (RELEASE/CLAUDE.md §1.8).
    """
    out: List[ValidationIssue] = []
    for n in ir.nodes:
        if n.schema_id != "domain_rand":
            continue
        # IRNode.params is Dict[str, IRParam] — membership test by key
        # is enough; we only need to know whether the legacy field was
        # persisted on this node.
        params = n.params if hasattr(n, "params") and isinstance(n.params, dict) else {}
        if "motor_strength_range" in params:
            out.append(ValidationIssue(
                code=IssueCode.UNKNOWN_PARAM_VALUE,
                severity=Severity.ERROR,
                node_id=n.id,
                field="motor_strength_range",
                message=(
                    "domain_rand.motor_strength_range was replaced by "
                    "omega_n_log_uniform (Stage H of the sim2sim PD "
                    "framework). Open the Domain Rand node and re-pick "
                    "the field — the new knob perturbs the canonical "
                    "(ωn, ζ) parameterization instead of MJCF actuator "
                    "gear, so the same DR semantics apply on IsaacLab/PhysX."
                ),
            ))
        if "joint_damping_range" in params:
            out.append(ValidationIssue(
                code=IssueCode.UNKNOWN_PARAM_VALUE,
                severity=Severity.ERROR,
                node_id=n.id,
                field="joint_damping_range",
                message=(
                    "domain_rand.joint_damping_range was replaced by "
                    "zeta_log_uniform (Stage H of the sim2sim PD "
                    "framework). Open the Domain Rand node and re-pick "
                    "the field — the new knob perturbs the canonical ζ "
                    "instead of MJCF dof_damping."
                ),
            ))
    return out


# Contact-sensor consumer enumeration. Keep this table aligned with
# scripts/rewards/isaac_lab/*.py il_params templates and the
# illegal_contact emit branch in
# isaac_lab/env_cfg_compiler.py (_terminations_cfg). New reward /
# termination that consumes contact_forces MUST be added here so the
# R_CONTACT1 coverage check stays exhaustive — otherwise the consumer
# starts at runtime against a sensor that may not cover its bodies.
#
# Format: {reward_key | termination_key: list of IR category names}
# illegal_contact is family-aware and handled inline (the categories
# resolve different role sets per family).
_CONTACT_REWARD_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "feet_air_time": ("feet",),
    "feet_slide": ("feet",),
    "gait": ("feet",),
    "track_gait_phase": ("feet",),
    "undesired_contacts": ("thighs", "hips", "base"),
}


def _illegal_contact_categories_for_families(families: Iterable[str]) -> Tuple[str, ...]:
    """Mirror env_cfg_compiler._terminations_cfg's family-aware illegal_contact
    IR category list (the body regex emit picks the same set)."""
    fam_set = set(families or [])
    if fam_set & {"biped", "humanoid"}:
        return ("base", "torso", "pelvis", "waist", "shoulders")
    if fam_set & {"quadruped", "wheeled"}:
        return ("base", "thighs", "calves")
    return ("base",)


def _check_contact_sensor_coverage(ir: "WorkflowIR") -> List[ValidationIssue]:
    """R_CONTACT1 — WYSIWYG sensor coverage ⊆ check.

    actor_setting.contact_body_names is the IR-role list the user picked
    in the ir_body_picker; env_cfg_compiler emits ContactSensorCfg's
    prim_path purely from those roles (no /Robot/.* fallback). Reward /
    termination nodes that consume the ``contact_forces`` sensor each
    declare a body subset via IR templates (``{ir:feet}`` /
    ``{ir:thighs_hips_base}``) or, for illegal_contact, a family-aware
    category set hardcoded in the compiler. The sensor's covered body
    set MUST be a superset of every consumer's required body set —
    otherwise the consumer reads an empty buffer at runtime and the
    training silently flat-lines (zero contact reward, no termination
    on body-slam, etc.) — the §1.8 silent-fallback failure mode this
    check forbids.

    Surfaces at canvas-time as
    ``CONTACT_SENSOR_COVERAGE_INSUFFICIENT`` with the three-part
    directive (which consumer needs which bodies + what the picker
    is missing + which IR roles to add).
    """
    out: List[ValidationIssue] = []
    # ContactSensor coverage is an IsaacLab concern: the SB3 MuJoCo env has no
    # ContactSensor (illegal_contact reads data.contact directly with a
    # hardcoded family body set, identical to IsaacLab by design), and
    # contact_body_names is hidden on SB3 canvases. Validating picker coverage
    # there would force the user to populate a field SB3 never reads.
    if (getattr(ir, "backend", "") or "").strip() == "sb3_mujoco":
        return out
    by_id = {n.schema_id: n for n in ir.nodes}

    actor_node = by_id.get("actor_setting")
    if actor_node is None:
        return out

    # Picker contents — IR roles list (post user commit).
    import json as _json

    def _get_param_value(node, key, fallback=""):
        p = node.params.get(key) if hasattr(node.params, "get") else None
        if p is None:
            return fallback
        return getattr(p, "value", fallback)

    raw_cbn = _get_param_value(actor_node, "contact_body_names", "[]")
    try:
        picker_roles = _json.loads(raw_cbn) if isinstance(raw_cbn, str) else list(raw_cbn)
    except (ValueError, TypeError):
        picker_roles = []
    if not isinstance(picker_roles, list):
        picker_roles = []
    picker_role_set = {str(r) for r in picker_roles}

    # Robot SKU → families → BodyIRMapper for IR-role → body resolution.
    robot_node = by_id.get("robot")
    if robot_node is None:
        return out
    asset_id_raw = _get_param_value(robot_node, "asset_id", "")
    asset_id = str(asset_id_raw or "").strip()
    if not asset_id:
        return out
    try:
        from registers.robots import resolve_id as _resolve_robot_id
        sku = _resolve_robot_id(asset_id) or asset_id
    except Exception:
        return out

    # Construct BodyIRMapper directly from the registry's per-format
    # body table (avoids the RobotAssetService dependency so this check
    # works in headless / pre-startup contexts too — e.g. unit tests).
    try:
        from registers.robots import get_robot
        entry = get_robot(sku)
    except Exception:
        return out
    if not entry:
        return out
    bodies_per_format = entry.get("bodies_per_format", {}) or {}
    fmt = None
    for candidate in ("USD", "MJCF", "URDF"):
        block = bodies_per_format.get(candidate)
        if isinstance(block, dict) and block:
            fmt = candidate
            break
    if fmt is None:
        # §8 fail-loud: the SKU has no body data in ANY format across the
        # three-layer merge (SDK ship < factory_build < user overlay).
        # Without it the ⊆ check cannot enumerate consumer needs, so a
        # silent return would let the canvas compile with the
        # ContactSensorCfg pointing at picker-empty roles and produce a
        # working-looking-but-flat-lined run (the §1.8 failure mode this
        # whole check exists to prevent).
        #
        # This usually means the IsaacLab auto-dump trigger has not yet
        # populated FACTORY_BUILD_DIR/robots_factory_build.json. The fix
        # path depends on which IsaacLab the user has:
        #   * Local install:  launch the app once with IsaacLab registered
        #                     and the data_load stage will dump every
        #                     canonical SKU automatically (~5-30s per SKU).
        #   * Cloud install:  the cloud-dump pathway is pending; install a
        #                     local IsaacLab to bootstrap, or wait.
        #   * No IsaacLab:    install one — the contact-sensor consumers
        #                     in this canvas need USD body topology to
        #                     wire reliably.
        out.append(ValidationIssue(
            code=IssueCode.CONTACT_SENSOR_BODY_DATA_MISSING,
            severity=Severity.ERROR,
            node_id=robot_node.id,
            field="asset_id",
            message=(
                f"Robot {asset_id!r} (sku={sku}) has no body data across "
                f"MJCF / USD / URDF after merging SDK ship + factory_build "
                f"+ user overlay. The contact-sensor ⊆ check cannot run, "
                f"and the ContactSensorCfg would silently equip an empty "
                f"body set (§1.8 silent-fallback ban).\n\n"
                f"Fix path:\n"
                f"  1. If you have a local IsaacLab installation, restart "
                f"the app — the data_load stage auto-dumps every standard "
                f"SKU's USD body table on launch (5-30s per SKU). The "
                f"dump lands in FACTORY_BUILD_DIR (not USER_CONFIG_DIR) "
                f"so it is shared across all users on this install.\n"
                f"  2. If you have only cloud IsaacLab, install a local "
                f"copy (Settings → Engines) — the cloud-dump RPC is "
                f"pending in this build.\n"
                f"  3. If you have no IsaacLab at all, install one — "
                f"this canvas's contact-sensor consumers (rewards / "
                f"terminations) need USD body topology to wire.\n\n"
                f"Once the dump completes, re-open this canvas; the ⊆ "
                f"check will run against the freshly populated data."
            ),
        ))
        return out
    body_block = bodies_per_format[fmt]
    body_names: List[str] = []
    roles_dict: Dict[str, Any] = {}
    for entry_val in body_block.values():
        if not isinstance(entry_val, dict):
            continue
        bn = str(entry_val.get("name", "") or "")
        rid = str(entry_val.get("ir_role", "") or "")
        if not bn:
            continue
        body_names.append(bn)
        if rid:
            roles_dict[rid] = {"body": bn}
    families = list(entry.get("families", []) or [])
    family = families[0] if families else "quadruped"

    # Construct the mapper without ``from_dict``'s legacy foot-auto-inject
    # path (that path appends a fake ``*_foot`` body name when the asset
    # only declares calves, intended to bridge old MJCF dumps — but for
    # ⊆ check it would invent IR roles the picker provably cannot select,
    # turning a true coverage hole into a spurious one).
    try:
        from application.training.body_ir import BodyIRMapper
        mapper = BodyIRMapper(body_names, family=family)
        mapper._build_roles_from_bodies()  # type: ignore[attr-defined]
        for rid, info in roles_dict.items():
            body = info.get("body")
            if not body:
                continue
            slot = mapper._by_id.get(rid)  # type: ignore[attr-defined]
            if slot is not None:
                slot.body = body
                slot.auto_from_asset = True
    except Exception:
        return out

    # Enumerate enabled contact consumers across reward / termination
    # nodes. Consumer set is fully knowable at canvas-time (no dynamic
    # body needs — PV穷举确认 5 reward + 1 termination 全可静态枚举).
    consumer_needs: Dict[str, set] = {}  # consumer_label → required IR roles

    for node in ir.nodes:
        if node.schema_id == "rewards":
            terms_raw = _get_param_value(node, "reward_terms", {})
            if isinstance(terms_raw, str):
                try:
                    terms_raw = _json.loads(terms_raw)
                except (ValueError, TypeError):
                    terms_raw = {}
            if not isinstance(terms_raw, dict):
                continue
            # 缺口③ — reward_terms is paged; flatten across pages so contact
            # rewards on any page (Global or joint) are seen.
            from application.compiler.term_payload import (
                is_paged_reward_terms, iter_reward_pages,
            )
            if is_paged_reward_terms(terms_raw):
                _flat: Dict[str, Any] = {}
                for _pid, _pt in iter_reward_pages(terms_raw):
                    for _fk, _pl in _pt.items():
                        _flat.setdefault(_fk, _pl)
                terms_raw = _flat
            for term_key, term_val in terms_raw.items():
                if term_key not in _CONTACT_REWARD_CATEGORIES:
                    continue
                # Treat missing or non-numeric weight as "enabled" (the
                # registry-backed editor stores raw payloads where the
                # weight key is sometimes nested in a dict).
                weight = (
                    term_val.get("weight")
                    if isinstance(term_val, dict)
                    else term_val
                )
                try:
                    if weight is not None and float(weight) == 0.0:
                        continue
                except (TypeError, ValueError):
                    pass
                categories = _CONTACT_REWARD_CATEGORIES[term_key]
                needed: set = set()
                for cat in categories:
                    for body in mapper.get_category_bodies(cat):
                        for role in mapper.roles:
                            if role.body == body and role.role_id:
                                needed.add(role.role_id)
                                break
                if needed:
                    consumer_needs.setdefault(
                        f"rewards.{term_key}", set()
                    ).update(needed)
        elif node.schema_id == "terminations":
            terms_raw = _get_param_value(node, "termination_conditions", {})
            if isinstance(terms_raw, str):
                try:
                    terms_raw = _json.loads(terms_raw)
                except (ValueError, TypeError):
                    terms_raw = {}
            if not isinstance(terms_raw, dict) or "illegal_contact" not in terms_raw:
                continue
            categories = _illegal_contact_categories_for_families(families)
            needed = set()
            for cat in categories:
                for body in mapper.get_category_bodies(cat):
                    for role in mapper.roles:
                        if role.body == body and role.role_id:
                            needed.add(role.role_id)
                            break
            if needed:
                consumer_needs.setdefault(
                    "terminations.illegal_contact", set()
                ).update(needed)

    if not consumer_needs:
        return out

    # ⊆ check: every consumer's needed IR roles must be picker-selected.
    missing_per_consumer: Dict[str, List[str]] = {}
    for consumer, needed in consumer_needs.items():
        missing = sorted(needed - picker_role_set)
        if missing:
            missing_per_consumer[consumer] = missing

    if not missing_per_consumer:
        return out

    # Compose the three-part directive message.
    lines = [
        "actor_setting.contact_body_names does not cover every body "
        "required by the canvas's contact-sensor consumers. The "
        "ContactSensorCfg only equips bodies the user picked, so any "
        "reward/termination reading an unequipped body will read an "
        "empty buffer at runtime (§1.8 silent-fallback ban).",
        "",
        "Consumers + their missing IR roles:",
    ]
    all_missing: set = set()
    for consumer in sorted(missing_per_consumer):
        roles = missing_per_consumer[consumer]
        lines.append(f"  - {consumer} needs: {roles}")
        all_missing.update(roles)
    lines.append("")
    lines.append(
        f"Open the actor_setting node, expand the Contact Bodies "
        f"picker, and add these IR roles: {sorted(all_missing)}. "
        f"Alternatively remove the consumer terms from the rewards "
        f"/ terminations node if they aren't intended for this run."
    )

    out.append(ValidationIssue(
        code=IssueCode.CONTACT_SENSOR_COVERAGE_INSUFFICIENT,
        severity=Severity.ERROR,
        node_id=actor_node.id,
        field="contact_body_names",
        message="\n".join(lines),
    ))
    return out


def _check_pd_param(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_PD1..R_PD4 — sim2sim PD parameterization integrity.

    Most rules are enforced upstream:
      * R_PD1 (unknown group for family) — ``spec_compiler._compile_pd_param``
        emits at compile time.
      * R_PD2 (omega_n / zeta out of range) — ``PDGroup.__post_init__``
        raises at construction time.
      * R_PD3 (at-most-one ActuatorPDNode) — IR layer's by_id collapses
        duplicates onto the first instance; topology check below would
        catch it if we add one, but spec-level we can only confirm the
        single survivor is well-formed.
      * R_PD4 (legacy stiffness/damping coexist with pd_param) — REMOVED
        2026-05: the scalar stiffness/damping fields no longer exist
        (raw kp/kd were dropped from the data model; PD is parameterized
        only by (omega_n, zeta), CLAUDE.md §10), so there is nothing to
        warn about.

    Returns the issues list (currently always empty — kept as a hook for
    future spec-level PD rules).
    """
    out: List[ValidationIssue] = []
    return out


def _check_training_packages(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_PKG1 — training package membership integrity (Method A, Slice 1).

    Runs the single source of truth ``resolve_effective_packages`` and surfaces
    any fail-loud membership violation as an ERROR the UI shows pre-export:
      * an enabled item naming a missing or disabled package;
      * an enabled item with no ``package_id`` when a package layer is authored;
      * an enabled package with zero enabled member items;
      * explicit use of the reserved implicit-default package id.
    An absent package layer resolves to the implicit default and is silent
    (byte-identical to pre-package behavior).
    """
    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None:
        return out
    from application.training.training_spec import resolve_effective_packages
    try:
        resolve_effective_packages(motion)
    except ValueError as exc:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.packages",
            message=f"training package membership invalid: {exc}",
        ))
    return out


def _check_skill_gating(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_SKILL — a package's ``gated_by`` must name an enabled skill_item (a trigger
    channel) on the training_motion node (skill_command_path_design.md Slice 3).

    A skill package (``gated_by`` set) has its reward gated on that skill's trigger
    command; if the named skill does not exist / is disabled the IsaacLab codegen
    would reference a command term that is never emitted. Fail loud pre-export.
    Presence-gated: a canvas with no gated package is silent.
    """
    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None:
        return out
    packages = getattr(motion, "packages", None) or {}
    skills = getattr(motion, "skill_items", None) or {}
    enabled_skills = {
        str(sid) for sid, sk in skills.items()
        if (not isinstance(sk, dict)) or bool(sk.get("enabled", True))
    }
    for pid, p in packages.items():
        g = str(getattr(p, "gated_by", "") or "").strip()
        if not g:
            continue
        if g not in enabled_skills:
            out.append(ValidationIssue(
                code=IssueCode.GENERIC,
                severity=Severity.ERROR,
                field="motion.packages",
                message=(
                    f"package {pid!r} is gated_by {g!r}, which is not an enabled "
                    f"skill_item (enabled skills: {sorted(enabled_skills)}). Author a "
                    f"trigger skill named {g!r} on the training_motion node, or fix "
                    f"the gated_by link (§8 fail-loud)."
                ),
            ))
    return out


def _check_skill_gait_deploy_conflict(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_SKILL2 — a canvas with BOTH gait AND a skill trigger is not deploy-supported yet.

    The deploy obs layout records the skill trigger obs term but NOT gait obs (a
    pre-existing gap). So at deploy the trigger obs would land at the wrong offset
    (gait present in training, absent in deploy) — a silent train≠deploy failure.
    Fail loud until the general command_slice deploy-obs mechanism also covers gait.
    Presence-gated: only fires when both are actually enabled.
    """
    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None:
        return out
    gait = getattr(motion, "gait", None)
    gait_on = bool(getattr(gait, "enabled", False)) if gait is not None else False
    skills = getattr(motion, "skill_items", None) or {}
    has_skill = any(
        (not isinstance(sk, dict)) or bool(sk.get("enabled", True))
        for sk in skills.values()
    )
    if gait_on and has_skill:
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.ERROR,
            field="motion.skill_items",
            message=(
                "a canvas with BOTH gait and a skill trigger channel is not "
                "deploy-supported yet: gait obs is not in the deploy obs layout, so "
                "the trigger obs would land at the wrong offset at deploy (a silent "
                "train≠deploy break). Disable gait or the skill for now — deploy-obs "
                "parity for gait+skill needs the general command_slice mechanism to "
                "also cover gait (§8 fail-loud)."
            ),
        ))
    return out


def _check_per_item_reward_scale(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_REWARD_SCALE — warn when WITHIN-PACKAGE per-item |Σ weight| diverges.

    Per-item composite rewards (``spec.rewards.terms_by_item``) define a
    separate reward bag for each motion item (stand / walk / run / …).
    When one item's summed weight is much larger than another's, the
    policy will preferentially chase the high-budget item even if it
    learns the low-budget one badly — exactly the scale-skew failure
    mode flagged in the audit report.

    Method A (Slice 1c): the skew is measured WITHIN each package, never across
    packages. Between-package balance is the user's deliberate ``package_weight``
    choice (the coarse global-rebalance lever) and must NOT be flagged as skew. A
    legacy canvas with no package layer resolves to one implicit default package
    (every item in one group), so the check is byte-identical to the pre-package
    cross-item behavior.

    The check is a WARNING (not ERROR): users may intentionally bias an
    item, but a 3:1 budget gap is almost always an oversight. The
    threshold lives in the function to keep the rule auditable from the
    code; bump it if real workloads need a different cut-off.
    """
    out: List[ValidationIssue] = []
    rewards = getattr(spec, "rewards", None)
    motion = getattr(spec, "motion", None)
    if rewards is None or motion is None:
        return out
    per_item = getattr(rewards, "terms_by_item", None) or {}
    if not isinstance(per_item, dict) or len(per_item) < 2:
        return out
    threshold = 3.0

    def _item_total(term_dict: Any) -> float:
        if not isinstance(term_dict, dict):
            return 0.0
        total = 0.0
        for val in term_dict.values():
            w = val.get("weight", 0.0) if isinstance(val, dict) else val
            try:
                total += abs(float(w))
            except (TypeError, ValueError):
                continue
        return total

    from application.training.training_spec import (
        DEFAULT_PACKAGE_ID,
        package_members,
        resolve_effective_packages,
    )
    try:
        packages = resolve_effective_packages(motion)
    except ValueError:
        # Membership is already reported by R_PKG1 (_check_training_packages);
        # do not re-report here — just skip the skew check.
        return out
    is_implicit_default = set(packages) == {DEFAULT_PACKAGE_ID}

    for pid in packages:
        item_totals: Dict[str, float] = {}
        for item_id in package_members(motion, pid):
            total = _item_total(per_item.get(item_id, {}))
            if total > 0.0:
                item_totals[item_id] = total
        if len(item_totals) < 2:
            continue
        max_id = max(item_totals, key=item_totals.get)
        min_id = min(item_totals, key=item_totals.get)
        ratio = item_totals[max_id] / max(item_totals[min_id], 1e-9)
        if ratio <= threshold:
            continue
        pkg_note = "" if is_implicit_default else f" within package {pid!r}"
        out.append(ValidationIssue(
            code=IssueCode.GENERIC,
            severity=Severity.WARNING,
            field="rewards.terms_by_item",
            message=(
                f"per-item reward budgets are skewed{pkg_note}: item {max_id!r} sums "
                f"|Σ weight|={item_totals[max_id]:.3g} vs {min_id!r} "
                f"={item_totals[min_id]:.3g} (ratio {ratio:.2f}× > "
                f"{threshold:.1f}×). The policy will preferentially chase "
                f"{max_id!r} even on commands intended to activate "
                f"{min_id!r}. Re-balance the weights or fold the cheap "
                f"item into the rich one's term set."
            ),
        ))
    return out


def _check_sb3_reward_term_kinds(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Reject SB3 canvases that wire reward kinds GenericMujocoEnv cannot
    compute. The runtime registry ``reward_terms._REWARD_FNS`` is a strict
    subset of ``scripts/rewards/registry.REWARD_REGISTRY`` (foot / contact /
    reference-tracking terms are deferred). Without this gate users get a
    one-line warn-and-skip at training start and their reward function
    silently misses terms they expected to drive policy behaviour.
    """
    out: List[ValidationIssue] = []
    backend = (getattr(spec.algorithm, "backend", "") or "").lower()
    if backend not in _SB3_BACKENDS:
        return out
    try:
        from application.training.envs.reward_terms import known_reward_kinds
    except Exception:
        return out
    known = set(known_reward_kinds())
    seen: set = set()
    terms = getattr(spec.rewards, "terms", None) or {}
    if isinstance(terms, dict):
        seen.update(terms.keys())
    per_item = getattr(spec.rewards, "terms_by_item", None) or {}
    if isinstance(per_item, dict):
        for item_terms in per_item.values():
            if isinstance(item_terms, dict):
                seen.update(item_terms.keys())
    unknown = sorted(k for k in seen if k and k not in known)
    if not unknown:
        return out
    out.append(ValidationIssue(
        code=IssueCode.UNKNOWN_PARAM_VALUE,
        severity=Severity.ERROR,
        field="rewards.terms",
        message=(
            f"reward kinds {unknown!r} are listed in the canvas registry "
            f"but the SB3 runtime (GenericMujocoEnv) does not compute them. "
            f"They would silently be dropped at training start. Either remove "
            f"these terms from the rewards node or implement them in "
            f"application/training/envs/reward_terms.py. Known kinds: "
            f"{sorted(known)}"
        ),
    ))
    return out


def _check_sb3_termination_kinds(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Reject SB3 canvases that wire termination kinds with no runtime
    implementation. Same rationale as :func:`_check_sb3_reward_term_kinds`.
    """
    out: List[ValidationIssue] = []
    backend = (getattr(spec.algorithm, "backend", "") or "").lower()
    if backend not in _SB3_BACKENDS:
        return out
    try:
        from application.training.envs.reward_terms import known_termination_kinds
    except Exception:
        return out
    known = set(known_termination_kinds())
    conditions = getattr(spec.terminations, "conditions", None) or {}
    if not isinstance(conditions, dict):
        return out
    unknown = sorted(k for k in conditions if k and k not in known)
    if not unknown:
        return out
    out.append(ValidationIssue(
        code=IssueCode.UNKNOWN_PARAM_VALUE,
        severity=Severity.ERROR,
        field="terminations.conditions",
        message=(
            f"termination kinds {unknown!r} are listed in the canvas registry "
            f"but the SB3 runtime (GenericMujocoEnv) does not evaluate them. "
            f"They would silently be dropped at training start. Remove them "
            f"or implement them in application/training/envs/reward_terms.py. "
            f"Known kinds: {sorted(known)}"
        ),
    ))
    return out


def _check_recommended_reward_terms(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Surface RECOMMENDED_LOCOMOTION_REWARD_TERMS that the canvas omitted.

    Background: the env's per-term lookup historically degraded missing
    keys to weight 0 (``self._reward_terms.get(key, 0.0)``), which
    silently produced crouching / asymmetric gaits when the user forgot
    to wire safety terms. Strict-canvas: surface the omission so the
    user sees it before the play submission. Reported as ``WARNING``
    (not ERROR) because non-locomotion canvases legitimately don't need
    these terms; the user can ignore the warning explicitly.
    """
    out: List[ValidationIssue] = []
    terms = getattr(spec.rewards, "terms", None) or {}
    # v2: per-item terms count toward "is term present" — if any item supplies
    # a recommended safety term, treat it as wired even if absent from the
    # global ``rewards.terms`` fallback. Empty intersection is still
    # acknowledged below.
    per_item = getattr(spec.rewards, "terms_by_item", None) or {}
    present: set = set()
    if isinstance(terms, dict):
        present.update(terms.keys())
    if isinstance(per_item, dict):
        for item_terms in per_item.values():
            if isinstance(item_terms, dict):
                present.update(item_terms.keys())
    if not present:
        return out  # empty terms is its own issue (caught elsewhere)
    try:
        from scripts import (
            recommended_reward_terms_for_backend,
        )
    except Exception:
        return out
    # IL canvases store IL term names (track_lin_vel_xy, alive_reward, …)
    # which don't overlap with the SB3 vocabulary (velocity_tracking,
    # alive, …). The pre-split single SB3-only list false-flagged every
    # IL canvas as "missing every safety term" — pick the list that
    # matches the canvas backend instead.
    backend = (getattr(spec.algorithm, "backend", "") or "").lower()
    recommended = recommended_reward_terms_for_backend(backend)
    missing = [k for k in recommended if k not in present]
    if not missing:
        return out
    out.append(ValidationIssue(
        code=IssueCode.GENERIC,
        severity=Severity.WARNING,
        field="rewards.terms",
        message=(
            f"reward_terms omits recommended locomotion safety terms: "
            f"{missing!r}. These default to weight 0 inside the env, which "
            f"historically led to crouching / asymmetric gait. Add them in "
            f"the Rewards node, or acknowledge by leaving this warning."
        ),
    ))
    return out


# Canonical command channel set / order — must match
# registers/data/commands_defaults.json and downstream consumers
# (generic_mujoco_env command vector, AMP tag_router, Isaac Lab reward
# tracking). The env unpacks the template into a fixed 3-tuple in this
# exact order, so dim != 3 or any unknown / missing key is a hard error.
_COMMAND_TEMPLATE_KEYS: Tuple[str, ...] = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def _check_command_template_contract(spec: "TrainingSpec") -> List[ValidationIssue]:
    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None:
        return out
    items = getattr(motion, "training_items", None) or {}
    expected = set(_COMMAND_TEMPLATE_KEYS)
    for item_id, item in items.items():
        if not isinstance(item, dict):
            continue
        tmpl = item.get("command_template")
        if tmpl is None:
            continue
        if not isinstance(tmpl, dict):
            out.append(ValidationIssue(
                code=IssueCode.INVALID_COMMAND_TEMPLATE,
                severity=Severity.ERROR,
                node_id="training_motion",
                field=f"training_items.{item_id}.command_template",
                message=(
                    f"command_template must be a dict keyed by "
                    f"{_COMMAND_TEMPLATE_KEYS!r}; got {type(tmpl).__name__}"
                ),
            ))
            continue
        actual = set(tmpl.keys())
        if actual != expected:
            out.append(ValidationIssue(
                code=IssueCode.INVALID_COMMAND_TEMPLATE,
                severity=Severity.ERROR,
                node_id="training_motion",
                field=f"training_items.{item_id}.command_template",
                message=(
                    f"command_template keys must be exactly "
                    f"{sorted(expected)} (canonical channel order "
                    f"{_COMMAND_TEMPLATE_KEYS!r}); got {sorted(actual)}"
                ),
            ))
    return out


def _check_heading_command(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Heading-command (legged_gym parity) sanity checks.

    Only fires when ``motion.heading_command`` is on. Guards the two
    closed-loop knobs that would silently misbehave: a non-positive
    stiffness (yaw never tracks the target → §8 illusion of a heading
    controller that does nothing) and a degenerate / out-of-range heading
    window. Applies to both engines (IsaacLab + SB3 both implement it).
    """
    out: List[ValidationIssue] = []
    motion = getattr(spec, "motion", None)
    if motion is None or not bool(getattr(motion, "heading_command", False)):
        return out
    stiffness = float(getattr(motion, "heading_control_stiffness", 0.5) or 0.0)
    if stiffness <= 0.0:
        out.append(ValidationIssue(
            code=IssueCode.INVALID_HEADING_COMMAND,
            severity=Severity.ERROR,
            node_id="training_motion",
            field="heading_control_stiffness",
            message=(
                "heading_command is on but heading_control_stiffness "
                f"{stiffness} <= 0 — the yaw channel would never track the "
                "target heading (closed loop with zero gain). Set a positive "
                "gain (legged_gym uses 0.5)."
            ),
        ))
    hr = getattr(motion, "heading_range", None)
    if isinstance(hr, (list, tuple)) and len(hr) == 2:
        lo, hi = float(hr[0]), float(hr[1])
        pi = 3.141592653589793
        if lo >= hi:
            out.append(ValidationIssue(
                code=IssueCode.INVALID_HEADING_COMMAND,
                severity=Severity.ERROR,
                node_id="training_motion",
                field="heading_range",
                message=(
                    f"heading_range lo ({lo}) must be < hi ({hi})."
                ),
            ))
        if lo < -pi - 1e-6 or hi > pi + 1e-6:
            out.append(ValidationIssue(
                code=IssueCode.INVALID_HEADING_COMMAND,
                severity=Severity.WARNING,
                node_id="training_motion",
                field="heading_range",
                message=(
                    f"heading_range [{lo}, {hi}] extends beyond [-π, π]; "
                    "headings wrap, so values outside this window alias onto "
                    "in-window targets. legged_gym uses [-π, π]."
                ),
            ))
    return out


def _check_backend_algorithm(spec: "TrainingSpec") -> List[ValidationIssue]:
    out: List[ValidationIssue] = []
    backend = spec.algorithm.backend
    algo = spec.algorithm.algorithm
    mode = spec.algorithm.training_mode
    if backend == "isaac_lab" and algo in ("SAC", "TD3"):
        out.append(ValidationIssue(
            code=IssueCode.BACKEND_ALGORITHM_MISMATCH,
            severity=Severity.ERROR,
            field="algorithm.backend",
            message=(
                f"backend=isaac_lab does not support {algo}; pin sb3_mujoco "
                f"or change algorithm to PPO"
            ),
        ))
    if backend == "isaac_lab" and mode == "AMP_PPO":
        # OK — Isaac Lab supports AMP via il_ppo_trainer.
        pass
    if backend == "sb3_mujoco" and mode == "AMP_PPO":
        # AMP-PPO on SB3 is not wired (no SB3 AMP runner); surface it *before*
        # submit so the play button doesn't burn a slot on a guaranteed crash.
        # (Also caught by _check_sb3_unsupported.) Requires spec.algorithm.backend
        # to be the canonical engine id sb3_mujoco — see the backend-id
        # unification (no "sb3" kind allowed).
        out.append(ValidationIssue(
            code=IssueCode.BACKEND_ALGORITHM_MISMATCH,
            severity=Severity.ERROR,
            field="algorithm.training_mode",
            message=(
                "AMP_PPO on sb3_mujoco backend is not yet wired (Stage 8); "
                "pin backend=isaac_lab for AMP_PPO, or change training_mode "
                "to PPO"
            ),
        ))
    return out


def _check_backend_registries(spec: "TrainingSpec") -> List[ValidationIssue]:
    """Rewards / terminations / domain_rand each carry a `backend` enum that
    must match the algorithm path, else the registry-keyed terms point at the
    wrong vocabulary."""
    out: List[ValidationIssue] = []
    algo_backend = spec.algorithm.backend
    expected = "isaac_lab" if algo_backend == "isaac_lab" else "sb3_mujoco"
    for path, actual in (
        ("rewards.backend", spec.rewards.backend),
        ("terminations.backend", spec.terminations.backend),
        ("domain_rand.backend", spec.domain_rand.backend),
    ):
        if actual != expected:
            out.append(ValidationIssue(
                code=IssueCode.BACKEND_REGISTRY_MISMATCH,
                severity=Severity.ERROR,
                field=path,
                message=(
                    f"{path}={actual!r} but algorithm.backend resolves to "
                    f"{expected!r}; registry-keyed terms will not resolve"
                ),
            ))
    return out


def _check_sim_dt(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R4: physics_config.sim_dt vs scene.sim_dt — warn on mismatch."""
    out: List[ValidationIssue] = []
    if abs(spec.physics.sim_dt - spec.scene.sim_dt) > 1e-9:
        out.append(ValidationIssue(
            code=IssueCode.SIM_DT_CONFLICT,
            severity=Severity.WARNING,
            field="physics.sim_dt",
            message=(
                f"physics.sim_dt={spec.physics.sim_dt} != "
                f"scene.sim_dt={spec.scene.sim_dt}; lowering picks per "
                f"backend (R4)"
            ),
        ))
    # control_dt must be an integer multiple of sim_dt.
    if spec.physics.sim_dt > 0:
        ratio = spec.physics.control_dt / spec.physics.sim_dt
        if abs(ratio - round(ratio)) > 1e-6:
            out.append(ValidationIssue(
                code=IssueCode.PARAM_OUT_OF_RANGE,
                severity=Severity.ERROR,
                field="physics.control_dt",
                message=(
                    f"control_dt={spec.physics.control_dt} must be an integer "
                    f"multiple of sim_dt={spec.physics.sim_dt} (got {ratio:.4f})"
                ),
            ))
    return out


def _check_action_joints(spec: "TrainingSpec") -> List[ValidationIssue]:
    """If actor.action_joint_names_expr is not empty, every *literal* name must
    be a recognised IR role in ``RobotSpec.joint_ir_roles``. Regex items
    (containing ``.*+?[](){}|^$\\``) and the bare catchall ``".*"`` are passed
    through to Isaac Lab as-is — matching the contract documented in
    ``_check_joint_ir_canonical``.

    Phase 5 IR-only contract (CLAUDE.md §1): the canvas carries IR roles only;
    ``joint_order`` holds *physical* joint names (``FL_hip_joint``), so the
    membership test MUST run in IR space against ``joint_ir_roles``
    (``hip_FL``). Comparing IR-role literals against physical ``joint_order``
    is an IR↔physical mixing bug — the Robot node's body/joint mapping is the
    only legal bridge between the two namespaces, and the env_cfg compiler
    already does that translation at the JointPositionActionCfg emit site via
    ``JointIRResolver.to_physical_list``."""
    out: List[ValidationIssue] = []
    expr = spec.actor.action_joint_names_expr
    if not expr:
        return out
    known = set(spec.robot.joint_ir_roles)
    if not known:
        return out  # robot not resolved / no joints; another check will flag
    regex_chars = set(".*+?[](){}|^$\\")
    missing = [
        n for n in expr
        if isinstance(n, str) and n and n != ".*"
        and not any(c in regex_chars for c in n)
        and n not in known
    ]
    if missing:
        out.append(ValidationIssue(
            code=IssueCode.UNMAPPED_ACTION_JOINTS,
            severity=Severity.ERROR,
            field="actor.action_joint_names_expr",
            message=(
                f"action joints {missing!r} not in RobotSpec.joint_ir_roles "
                f"(robot={spec.robot.sku}). List items must be IR roles "
                f"(e.g. 'hip_FL', 'calf_RR'), not physical joint names — the "
                f"Robot node's mapping translates them at compile time."
            ),
        ))
    return out


def _check_amp_wiring(spec: "TrainingSpec") -> List[ValidationIssue]:
    """AMP_PPO needs reference motion + discriminator hyperparams populated.

    Also gates the Isaac Lab path on the presence of the UnitPort launcher
    (``Engines.isaac_lab_unitport_launcher``). Without it, the run would
    fall back to Isaac Lab's stock ``train.py`` which has no AMP support —
    the canvas would silently degrade to plain PPO. Refusing at validation
    time prevents that "silent fake" pattern.
    """
    out: List[ValidationIssue] = []
    if spec.algorithm.training_mode != "AMP_PPO":
        return out
    # ``motion_ref`` is None when the compiler skipped populating it
    # (e.g. the canvas left ``consumption_mode`` blank — that emits
    # ``MISSING_CONSUMPTION_MODE`` upstream). Either situation is a
    # missing-reference-motion ERROR for AMP_PPO; emit the same
    # ``INCOMPLETE_AMP_WIRING`` issue rather than double-flagging.
    if spec.il.motion_ref is None or not spec.il.motion_ref.clip_paths:
        out.append(ValidationIssue(
            code=IssueCode.INCOMPLETE_AMP_WIRING,
            severity=Severity.ERROR,
            field="il.motion_ref",
            message=(
                "AMP_PPO mode requires reference motion clips "
                "(training_motion.training_items[*].clip)"
            ),
        ))
    if spec.algorithm.backend == "isaac_lab":
        # The launcher lives in the project source tree at a fixed location
        # anchored on ``Paths.APP_ROOT``; backends registry computes it.
        # There is no user override path — this is framework code, not
        # user config. A missing file is a project-integrity defect.
        from registers import backends as _backends
        launcher_path = _backends.train_launcher_path("isaac_lab")
        if launcher_path is None or not launcher_path.is_file():
            out.append(ValidationIssue(
                code=IssueCode.INCOMPLETE_AMP_WIRING,
                severity=Severity.ERROR,
                field="algorithm.backend",
                message=(
                    f"AMP_PPO on isaac_lab requires the in-tree UnitPort "
                    f"launcher at {launcher_path!s} — file missing, the "
                    f"source tree is incomplete. Re-checkout / restore the "
                    f"RELEASE/src/application/training/isaac_lab/launcher/ "
                    f"directory. (Stock RSL-RL train.py has no AMP support; "
                    f"the run would silently degrade to plain PPO without "
                    f"this launcher.)"
                ),
            ))
    return out


def _check_robot(spec: "TrainingSpec") -> List[ValidationIssue]:
    out: List[ValidationIssue] = []
    if not spec.robot.sku:
        out.append(ValidationIssue(
            code=IssueCode.UNKNOWN_ROBOT,
            severity=Severity.ERROR,
            field="robot",
            message="robot is not resolved; canvas robot.asset_id was rejected by registry",
        ))
        return out
    if not spec.robot.joint_order:
        out.append(ValidationIssue(
            code=IssueCode.UNKNOWN_ROBOT,
            severity=Severity.WARNING,
            field="robot",
            message=(
                f"robot {spec.robot.sku} has no declared joints; "
                f"action_dim will fall back to 0"
            ),
        ))
    return out


def _check_init_joint_angles_in_range(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_INIT1 — every ``actor.joint_init`` angle must fall inside the robot's
    physical joint limits.

    Reads ranges from the robot's on-disk MJCF (resolved via
    RobotAssetService → absolute path). The pipe carries USD data too,
    but USD-side range extraction would require loading the asset inside
    the IsaacLab venv; MJCF is the convenient cross-engine canonical
    today. Skipped (with WARN) when the SKU has no on-disk MJCF.

    Pinpointing on the canvas: the offending node is the
    ``actor_setting`` node — the validator surfaces ``node_id`` on the
    issue so the UI can highlight that node with the ``danger_zone``
    theme color. CLAUDE.md §1.8: this is fail-loud at the earliest
    possible boundary, preventing the silent corruption where the
    canvas's inline view shows correct values but the slider editor
    drops them and 0.0 gets emitted to env.yaml.
    """
    out: List[ValidationIssue] = []
    if spec.actor is None:
        return out
    ji = getattr(spec.actor, "joint_init", None)
    if not isinstance(ji, dict) or not ji:
        return out
    sku = str(getattr(spec.robot, "sku", "") or "")
    if not sku:
        return out

    # Resolve MJCF via the same path the canvas / runtime uses.
    try:
        from application.service.robot_assets.service import (
            get_robot_asset_service,
        )
        asset = get_robot_asset_service().resolve(sku)
    except Exception:  # noqa: BLE001
        return out
    if asset is None or asset.mjcf_path is None or not asset.mjcf_path.is_file():
        # No MJCF on disk — env_cfg_compiler's runtime validator still
        # protects, just with a less-actionable error. We can't enforce
        # the contract without the source-of-truth ranges.
        return out

    try:
        import mujoco  # type: ignore
    except Exception:  # noqa: BLE001
        return out
    try:
        m = mujoco.MjModel.from_xml_path(str(asset.mjcf_path))
    except Exception:  # noqa: BLE001
        return out

    # Map MJCF joint name → (lo, hi) when limited.
    mj_ranges: dict[str, tuple[float, float]] = {}
    for ji_id in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji_id)
        if not jname:
            continue
        if not bool(m.jnt_limited[ji_id]):
            continue
        mj_ranges[jname] = (float(m.jnt_range[ji_id][0]),
                            float(m.jnt_range[ji_id][1]))

    # IR role → physical joint name via JointIRResolver (MJCF format).
    try:
        from application.training.joint_ir import JointIRResolver
        resolver = JointIRResolver(spec.robot, active_format="MJCF")
    except Exception:  # noqa: BLE001
        return out

    violations: List[tuple[str, str, float, float, float]] = []
    for ir_role, angle in ji.items():
        try:
            phys = resolver.to_physical(str(ir_role))
        except Exception:  # noqa: BLE001
            continue
        if phys not in mj_ranges:
            continue
        lo, hi = mj_ranges[phys]
        try:
            a = float(angle)
        except (TypeError, ValueError):
            continue
        if a < lo or a > hi:
            violations.append((str(ir_role), phys, a, lo, hi))

    if violations:
        bullet_lines = "\n".join(
            f"  • IR role {ir!r} (physical {phys!r}): {a:.4f} rad "
            f"outside [{lo:.4f}, {hi:.4f}]"
            for ir, phys, a, lo, hi in violations
        )
        out.append(ValidationIssue(
            code=IssueCode.PARAM_OUT_OF_RANGE,
            severity=Severity.ERROR,
            node_id="actor_setting",
            field="init_joint_angles",
            message=(
                "actor_setting.init_joint_angles has angles outside the "
                "robot's MJCF joint limits:\n" + bullet_lines + "\n\n"
                "Open ActorSetting → init_joint_angles in the canvas and "
                "pick a pose inside every joint's physical range. "
                f"Reference for quadrupeds: hip ≈ 0, thigh ≈ 0.9, "
                f"knee ≈ -1.55 to -1.8 (Spot bends backward, Go2 forward; "
                f"check the per-joint range above)."
            ),
        ))
    return out


def _check_joint_ir_canonical(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R8 — Phase 5 IR-only joint contract.

    Every joint name reachable from the canvas (``actor.joint_init`` keys,
    ``actor.action_joint_names_expr`` items when an explicit list) MUST be
    a registered IR role for the bound robot's ``joint_ir_roles``.  Vendor
    abbreviations (``fl_hx``), USD physical names (``FL_hip_joint``), and
    arbitrary user typos all fail here — there is no tolerance.

    The IR-only rule prevents three classes of silent breakage:
      1. Isaac Lab ``Articulation._process_cfg`` regex-mismatching against
         USD joint names → training crashes at env construction.
      2. ``manifest.robot.joint_names`` carrying inconsistent names → deploy
         binding to MJCF/URDF fails or maps the wrong actuator.
      3. Cross-machine bundle reuse impossible — the IR layer is the only
         abstraction that lets a Go2 policy load on an A1.

    Tolerable: ``[]`` (empty list) or a single regex catchall like
    ``".*"`` are passed through to Isaac Lab as-is.
    """
    out: List[ValidationIssue] = []
    valid_ir = set(spec.robot.joint_ir_roles or [])
    if not valid_ir:
        # Robot has no joints declared — _check_robot already warned;
        # nothing meaningful to validate here.
        return out

    actor = spec.actor
    if actor is None:
        return out

    # joint_init keys must be IR roles
    ji = getattr(actor, "joint_init", None)
    if isinstance(ji, dict) and ji:
        bad = sorted(k for k in ji if k not in valid_ir)
        if bad:
            out.append(ValidationIssue(
                code=IssueCode.NON_IR_JOINT_NAME,
                severity=Severity.ERROR,
                field="actor.joint_init",
                message=(
                    f"actor.joint_init contains non-IR joint name(s): {bad}. "
                    f"Robot {spec.robot.sku!r} valid IR roles: {sorted(valid_ir)}. "
                    f"Fix: edit the canvas Actor Setting node so every joint "
                    f"key is an IR role from that list. Run "
                    f"`bootstrap/migrate_canvas_joint_names_to_ir.py` to "
                    f"auto-translate a legacy canvas."
                ),
            ))

    # action_joint_names_expr items: only validate when each item looks
    # like a literal IR role (not a regex). A heuristic: if the item is a
    # literal string with no regex metacharacters AND is not empty AND
    # does not equal ".*", treat it as an IR role candidate.
    aje = getattr(actor, "action_joint_names_expr", None)
    if isinstance(aje, list) and aje:
        regex_chars = set(".*+?[](){}|^$\\")
        bad_items: List[str] = []
        for item in aje:
            s = str(item)
            if not s or s == ".*" or any(c in regex_chars for c in s):
                continue
            if s not in valid_ir:
                bad_items.append(s)
        if bad_items:
            out.append(ValidationIssue(
                code=IssueCode.NON_IR_JOINT_NAME,
                severity=Severity.ERROR,
                field="actor.action_joint_names_expr",
                message=(
                    f"actor.action_joint_names_expr contains non-IR literal "
                    f"item(s): {sorted(bad_items)}. Robot {spec.robot.sku!r} "
                    f"valid IR roles: {sorted(valid_ir)}. Fix: edit the "
                    f"canvas Actor Setting node — list items must be either "
                    f"a regex (e.g. \"hip_.*\") or an exact IR role from the "
                    f"list above."
                ),
            ))

    return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def filter_errors(issues: Iterable[ValidationIssue]) -> List[ValidationIssue]:
    return [i for i in issues if i.severity is Severity.ERROR]


def raise_if_errors(issues: Iterable[ValidationIssue]) -> None:
    errs = filter_errors(issues)
    if errs:
        raise SpecValidationError(errs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param_value(node, key: str, default: Any = None) -> Any:
    p = node.params.get(key)
    if p is None:
        return default
    return getattr(p, "value", default)


__all__ = [
    "Severity",
    "IssueCode",
    "ValidationIssue",
    "SpecValidationError",
    "AlgorithmFamily",
    "classify_family",
    "check_topology",
    "check_spec",
    "filter_errors",
    "raise_if_errors",
]
