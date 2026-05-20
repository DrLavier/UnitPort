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
    NON_IR_JOINT_NAME = "non_ir_joint_name"
    INVALID_COMMAND_TEMPLATE = "invalid_command_template"
    REWARD_TERM_CONFLICT = "reward_term_conflict"
    BASE_ASSET_UNRESOLVED = "base_asset_unresolved"
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
    issues.extend(_check_reward_term_conflicts(ir))
    issues.extend(_check_base_asset_resolvable(ir))
    issues.extend(_check_legacy_dr_fields(ir))  # R_DR1 — Stage H migration
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
    issues.extend(_check_recommended_reward_terms(spec))
    issues.extend(_check_sb3_reward_term_kinds(spec))
    issues.extend(_check_sb3_termination_kinds(spec))
    issues.extend(_check_per_item_reward_scale(spec))  # R_REWARD_SCALE
    issues.extend(_check_pd_param(spec))             # R_PD1..R_PD4 — sim2sim PD
    return issues


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
      * R_PD4 (legacy stiffness/damping coexist with pd_param) — WARN
        when both are non-default to direct the user toward cleanup.

    The check below covers R_PD4 (the only rule that needs spec-level
    state). Returns the issues list (possibly empty).
    """
    out: List[ValidationIssue] = []
    actor = getattr(spec, "actor", None)
    if actor is None:
        return out
    pd_param = getattr(actor, "pd_param", None)
    if pd_param is None:
        return out

    # R_PD4: if pd_param is set, the legacy ActuatorConfig.stiffness/damping
    # fields are dead. Warn the user that any value they typed there is
    # ignored to avoid the "I edited stiffness and nothing changed"
    # debugging trap.
    legacy_actuator = getattr(actor, "actuator", None)
    if legacy_actuator is not None:
        # ActuatorConfig dataclass defaults: stiffness=25.0, damping=0.5.
        if (
            abs(float(getattr(legacy_actuator, "stiffness", 25.0)) - 25.0) > 1e-9
            or abs(float(getattr(legacy_actuator, "damping", 0.5)) - 0.5) > 1e-9
        ):
            out.append(ValidationIssue(
                code=IssueCode.UNKNOWN_PARAM_VALUE,
                severity=Severity.WARNING,
                node_id="actor_setting",
                field="stiffness/damping",
                message=(
                    "ActuatorPDNode is wired; ActorSetting.stiffness / "
                    "damping are ignored. Reset them to defaults (25.0 / "
                    "0.5) to remove this warning. PD now flows through "
                    "the canonical (omega_n, zeta) parameterization on "
                    "ActuatorPDNode."
                ),
            ))

    return out


def _check_per_item_reward_scale(spec: "TrainingSpec") -> List[ValidationIssue]:
    """R_REWARD_SCALE — warn when per-item total |Σ weight| diverges.

    Per-item composite rewards (``spec.rewards.terms_by_item``) define a
    separate reward bag for each motion item (stand / walk / run / …).
    When one item's summed weight is much larger than another's, the
    policy will preferentially chase the high-budget item even if it
    learns the low-budget one badly — exactly the scale-skew failure
    mode flagged in the audit report.

    The check is a WARNING (not ERROR): users may intentionally bias an
    item, but a 3:1 budget gap is almost always an oversight. The
    threshold lives in the function to keep the rule auditable from the
    code; bump it if real workloads need a different cut-off.
    """
    out: List[ValidationIssue] = []
    rewards = getattr(spec, "rewards", None)
    if rewards is None:
        return out
    per_item = getattr(rewards, "terms_by_item", None) or {}
    if not isinstance(per_item, dict) or len(per_item) < 2:
        return out
    threshold = 3.0
    item_totals: Dict[str, float] = {}
    for item_id, term_dict in per_item.items():
        if not isinstance(term_dict, dict):
            continue
        total = 0.0
        for val in term_dict.values():
            if isinstance(val, dict):
                w = val.get("weight", 0.0)
            else:
                w = val
            try:
                total += abs(float(w))
            except (TypeError, ValueError):
                continue
        if total > 0.0:
            item_totals[item_id] = total
    if len(item_totals) < 2:
        return out
    max_id = max(item_totals, key=item_totals.get)
    min_id = min(item_totals, key=item_totals.get)
    ratio = item_totals[max_id] / max(item_totals[min_id], 1e-9)
    if ratio <= threshold:
        return out
    out.append(ValidationIssue(
        code=IssueCode.GENERIC,
        severity=Severity.WARNING,
        field="rewards.terms_by_item",
        message=(
            f"per-item reward budgets are skewed: item {max_id!r} sums "
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
    if backend not in ("sb3", "sb3_mujoco"):
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
    if backend not in ("sb3", "sb3_mujoco"):
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
        from application.training.isaac_lab.task_module_registry import (
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
        # AMP-PPO on SB3 is Stage 8 (SB3 AMP launcher hasn't landed). The
        # subprocess will hard-fail at startup; surface this *before* submit
        # so the play button doesn't burn a slot on a guaranteed crash.
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
    expected = "isaac_lab" if algo_backend == "isaac_lab" else "sb3"
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
    be in RobotSpec.joint_order. Regex items (containing ``.*+?[](){}|^$\\``)
    and the bare catchall ``".*"`` are passed through to Isaac Lab as-is —
    matching the contract documented in ``_check_joint_ir_canonical``."""
    out: List[ValidationIssue] = []
    expr = spec.actor.action_joint_names_expr
    if not expr:
        return out
    known = set(spec.robot.joint_order)
    if not known:
        return out  # robot not resolved; another check will flag
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
                f"action joints {missing!r} not in RobotSpec.joint_order "
                f"(robot={spec.robot.sku})"
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
    if not spec.il.motion_ref.clip_paths:
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
