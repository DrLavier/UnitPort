"""Parse Isaac Lab's exported env.yaml into a SkillManifest v2.

Isaac Lab's play.py export produces an ``env.yaml`` that describes the full
environment configuration.  This module extracts the information needed to
create a reactive SkillManifest with correct observation layout, action
config, and command_interface.

Public API
----------
parse_isaac_lab_env_yaml(env_yaml_path) -> SkillManifest
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# YAML loader with narrow Python-builtin support
# ---------------------------------------------------------------------------
# Isaac Lab's ``isaaclab.utils.io.dump_yaml`` uses ``yaml.Dumper`` (not
# ``yaml.SafeDumper``), so the exported ``env.yaml`` contains Python-typed
# tags such as ``!python/tuple`` (range-style configs like friction / mass /
# init pose) and ``!python/object/apply:builtins.slice`` (observation index
# ranges). Loading with ``yaml.safe_load`` raises ConstructorError on the
# first such tag.
#
# Switching to ``yaml.unsafe_load`` would let *any* Python class be
# instantiated from the file — too broad for a parser that ingests data from
# an external training tool. Instead we subclass SafeLoader and explicitly
# register constructors for the small set of builtin tags Isaac Lab actually
# emits. Everything else keeps going through SafeLoader's strict path; if
# Isaac Lab starts emitting a new tag in a future version we will see a
# loud ConstructorError pointing at it instead of silently constructing
# arbitrary objects.
class _IsaacLabEnvYamlLoader(yaml.SafeLoader):
    """SafeLoader extended with the Python-builtin tags Isaac Lab emits."""


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


def _construct_builtin_slice(loader: yaml.Loader, node: yaml.Node) -> slice:
    # Isaac Lab serializes ``slice(start, stop, step)`` as a sequence node
    # under the apply tag; reconstruct by calling the builtin with the same
    # positional arguments.
    args = loader.construct_sequence(node, deep=True)
    return slice(*args)


_IsaacLabEnvYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)
_IsaacLabEnvYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    _construct_builtin_slice,
)

from application.training.isaac_lab.skill_manifest import (
    ActionSpaceType,
    CommandField,
    CommandInterface,
    InferenceBackend,
    Postcondition,
    Precondition,
    SkillManifest,
    SourceType,
)

log = logging.getLogger(__name__)


# Observation term names that represent the command input
_COMMAND_TERM_NAMES = {
    "velocity_commands", "velocity_command", "base_velocity", "commands",
}



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_isaac_lab_env_yaml(
    env_yaml_path: Path,
    *,
    skill_id: str = "",
    model_path: str = "policy.onnx",
    robot: Any = None,
    deploy_meta_path: Optional[Path] = None,
) -> SkillManifest:
    """Parse an Isaac Lab env.yaml and return a fully populated SkillManifest.

    Parameters
    ----------
    env_yaml_path:
        Path to the exported env.yaml.
    skill_id:
        Override for skill_id.  If empty, derived from task name.
    model_path:
        Relative path to the policy file within the bundle.
    robot:
        Robot identifier — accepts THREE shapes for backwards compatibility:

          * **Phase 5 preferred** — :class:`RobotSpecRef` from the training
            spec. The parser then writes ``manifest.robot.joint_names`` as
            the **IR-role list** (``robot.joint_ir_roles``) per the IR-only
            deploy contract; physical names from the brand_package are
            kept on a sidecar field for debugging only.
          * **Legacy** — ``(brand_id, model_id)`` tuple. Falls back to the
            old behavior: ``manifest.robot.joint_names`` carries physical
            joint names from the brand_package model_registry.
          * **None** — resolve identity from env.yaml's
            ``scene.robot.spawn.usd_path`` against the model registry; same
            legacy physical-name behavior.

        Production callers (Phase 5 ``bundle_finalizer.finalize_isaac_lab_bundle``)
        always pass a :class:`RobotSpecRef`; the tuple/None paths are only
        for stand-alone parser invocations and unit tests.

    Returns
    -------
    SkillManifest
        A v2 manifest with ``execution_mode="reactive"`` and
        ``command_interface`` populated from the observation layout.

    Raises
    ------
    UnsupportedRobotForTrainingError
        If the robot identity cannot be determined from either the
        explicit ``robot=`` argument or the env.yaml structure, OR if
        the resolved (brand, model) has no JOINT_ORDER /
        IR_ROLE_TO_JOINT declared in its brand_package.
    """
    from application.service.models.model_registry import (
        UnsupportedRobotForTrainingError,
    )

    env_yaml_path = Path(env_yaml_path)
    with env_yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=_IsaacLabEnvYamlLoader) or {}

    # Load the compile-time sidecar if the caller pointed us at one. Module
    # A's strict contract: when this file is present, its obs_groups become
    # the authoritative source for per-term scale/clip/history (a None field
    # there raises in ``_extract_obs_term_meta``). Absence = back-compat
    # env.yaml-only path.
    deploy_meta_loaded: Optional[Dict[str, Any]] = None
    if deploy_meta_path is not None:
        dm_path = Path(deploy_meta_path)
        if dm_path.is_file():
            try:
                import json as _json
                deploy_meta_loaded = _json.loads(
                    dm_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise RuntimeError(
                    f"deploy_meta.json at {dm_path} could not be parsed: "
                    f"{exc}. Either fix the file or remove it (the parser "
                    f"will then fall back to env.yaml-only mode)."
                ) from exc
            if not isinstance(deploy_meta_loaded, dict):
                raise RuntimeError(
                    f"deploy_meta.json at {dm_path} is not a JSON object."
                )
            sv = deploy_meta_loaded.get("schema_version")
            if sv != 1:
                raise RuntimeError(
                    f"deploy_meta.json at {dm_path} has schema_version="
                    f"{sv!r}; this parser supports 1."
                )
            log.info(
                "deploy_meta sidecar loaded from %s — obs term scale/clip/"
                "history will be sourced from it (authoritative).",
                dm_path,
            )

    # Strict mode: only RobotSpecRef is accepted as the identity source.
    # The tuple / env.yaml-USD-path / brand_package fallback paths have been
    # removed — they existed for pre-Phase-5 bundles and silently degraded
    # the IR-role contract when the upstream caller failed to pass a snapshot.
    # If you're parsing a legacy bundle, run
    # ``bootstrap/migrate_canvas_strict_v1.py`` to upgrade the source canvas
    # and re-finalize.
    from application.training.training_spec import RobotSpecRef as _RobotSpecRef
    if not isinstance(robot, _RobotSpecRef):
        raise UnsupportedRobotForTrainingError(
            "manifest_parser.parse_isaac_lab_env_yaml requires a "
            "RobotSpecRef (Phase 5 contract). Got "
            f"{type(robot).__name__ if robot is not None else 'None'}. "
            "The tuple / env.yaml-only fallback paths have been removed "
            "in the strict-canvas migration; re-export the bundle from "
            "the current spec_compiler path."
        )
    robot_spec_ref = robot
    brand_id = str(getattr(robot_spec_ref, "brand", "") or "").strip()
    model_id = str(getattr(robot_spec_ref, "model", "") or "").strip()
    if not brand_id or not model_id:
        # RobotSpecRef snapshot lacks brand/model — derive from the SKU
        # via the registry. Failure here is a registry bug, not a
        # back-compat hole, so the lookup is NOT wrapped in try/except.
        from registers.robots import get_robot
        entry = get_robot(robot_spec_ref.sku) or {}
        brand_id = brand_id or str(entry.get("brand", "") or "")
        model_id = model_id or str(entry.get("model", "") or "")
    if not brand_id or not model_id:
        raise UnsupportedRobotForTrainingError(
            f"RobotSpecRef sku={robot_spec_ref.sku!r} lacks brand/model; "
            f"cannot build manifest. Confirm registers/data/robots_canonical.json "
            f"has brand/model fields for this entry."
        )
    # IR-only contract: manifest joint_names = IR roles (in joint_order index)
    runtime_joints = list(getattr(robot_spec_ref, "joint_ir_roles", []) or [])
    # Sidecar: keep physical names too in case downstream debug needs them
    runtime_joints_physical = list(getattr(robot_spec_ref, "joint_order", []) or [])
    # IR role → physical name lookup
    ir_role_to_joint = {
        ir: phys for ir, phys in zip(runtime_joints, runtime_joints_physical)
    }
    # Family taxonomy is owned by the registry (robots_canonical.json
    # ``families`` field) — no hardcoded brand/model lists in core.
    families = list(getattr(robot_spec_ref, "families", []) or [])

    # Navigate Isaac Lab env.yaml structure
    # Typical: {"scene": {...}, "observations": {...}, "actions": {...}, "commands": {...}, ...}
    obs_cfg = _extract_obs_config(raw)
    act_cfg = _extract_action_config(raw)
    cmd_cfg = _extract_command_config(raw)
    robot_cfg = _extract_robot_config(
        raw, runtime_joints, brand_id, model_id,
        joints_for_regex=runtime_joints_physical,
        families=families,
    )

    # --- Compute observation layout ---
    obs_terms, obs_dim, command_start_idx = _compute_obs_layout(
        obs_cfg, robot_cfg, deploy_meta=deploy_meta_loaded,
    )
    obs_keys = [t["name"] for t in obs_terms]

    # --- action ---
    num_joints = robot_cfg.get("num_joints", 12)
    action_dim = act_cfg.get("dim", num_joints)
    action_scale = act_cfg.get("scale", 0.25)

    # --- control frequency ---
    sim_dt = _deep_get(raw, "sim.dt", 0.005)
    decimation = _deep_get(raw, "decimation", _deep_get(raw, "sim.decimation", 4))
    control_dt = sim_dt * decimation
    control_freq = 1.0 / control_dt if control_dt > 0 else 50.0

    # --- command interface ---
    command_fields = _build_command_fields(cmd_cfg, command_start_idx)
    cmd_interface = CommandInterface(
        type="velocity_2d",
        fields=tuple(command_fields),
    ) if command_fields else None

    # --- joint mapping ---
    # Was: identity-mapped regex patterns (.*L_hip_joint → .*L_hip_joint).
    # Now: brand_package supplies the IR-canonical mapping. Keys are
    # canonical IR roles (hip_FL, thigh_FL, ...); values are the runtime
    # joint names declared by the brand package — never the regex patterns.
    joint_mapping = dict(ir_role_to_joint) if ir_role_to_joint else None

    # --- default joint positions (for action offset) ---
    default_joint_pos = robot_cfg.get("default_joint_pos", {})

    # --- skill_id ---
    task_name = raw.get("task_name", "") or raw.get("env_name", "") or ""
    if not skill_id:
        skill_id = task_name.replace("-", "_").lower() if task_name else "isaac_lab_policy"

    # --- tags ---
    tags = ["isaac_lab"]
    if task_name:
        tags.append(task_name)

    # --- DeployContract assembly (Module B) ---
    # Build the contract dict from env.yaml + spec. ObsBuilder._build_with_contract
    # uses this for per-term scale/clip/history fidelity at run time;
    # bundle_finalizer serialises it verbatim under manifest.yaml::deploy_contract.
    # robot_sku is left empty here and filled by bundle_finalizer (the spec.robot
    # SKU lives one layer up and is not threaded through parse_isaac_lab_env_yaml).
    actuator_pd = _extract_actuator_pd(raw, runtime_joints_physical)
    obs_terms_meta = _extract_obs_term_meta(
        obs_cfg, robot_cfg,
        deploy_meta=deploy_meta_loaded,
    )
    action_spec_dict = _extract_action_spec(act_cfg)
    base_body_name = _extract_base_body_name(raw)
    default_joint_pos_list = _build_default_joint_pos_list(
        robot_cfg.get("default_joint_pos") or {},
        runtime_joints_physical,
    )
    step_dt = float(sim_dt) * int(decimation)
    deploy_contract_dict: Dict[str, Any] = {
        "schema_version": 1,
        # Phase 5 IR-only deploy contract: joint_sdk_names = IR roles, NOT
        # physical names. PolicyRunner resolves IR→physical via robot_sku.
        "joint_sdk_names": list(runtime_joints),
        # Training order == bundle order == identity permutation. If a
        # backend ever re-orders policy outputs vs joint_order, set this
        # to the explicit permutation; for SB3/Isaac Lab today it's identity.
        "joint_ids_map": list(range(len(runtime_joints))),
        "stiffness": actuator_pd["stiffness"],
        "damping": actuator_pd["damping"],
        "effort_limit": actuator_pd["effort_limit"],
        "velocity_limit": actuator_pd["velocity_limit"],
        "saturation_effort": actuator_pd["saturation_effort"],
        "default_joint_pos": default_joint_pos_list,
        "step_dt": float(step_dt),
        "sim_dt": float(sim_dt),
        "decimation": int(decimation),
        "observations": {
            t["name"]: {
                "dim": int(t["dim"]),
                "scale": t["scale"],
                "clip": t["clip"],
                "history_length": int(t["history_length"]),
            }
            for t in obs_terms_meta
        },
        "action": action_spec_dict,
        "commands": _extract_commands_spec(cmd_cfg),
        "base_body_name": base_body_name,
        "robot_sku": "",  # filled by bundle_finalizer from spec.robot.sku
    }

    return SkillManifest(
        skill_id=skill_id,
        skill_name=task_name or skill_id,
        version="1.0.0",
        source_type=SourceType.ISAAC_LAB,
        execution_mode="reactive",
        action_space_type=ActionSpaceType.JOINT_POSITION,
        action_dim=action_dim,
        control_frequency_hz=control_freq,
        action_range=None,
        required_sensors=["imu", "joint_encoder"],
        observation_space_keys=obs_keys,
        observation_dim=obs_dim,
        target_robot_family=robot_cfg.get("family", "quadruped"),
        target_robot_models=robot_cfg.get("models", []),
        joint_mapping=joint_mapping,
        precondition=Precondition(posture="standing", velocity_max_mps=0.5),
        postcondition=Postcondition(posture="standing"),
        inference_backend=InferenceBackend.ONNX,
        model_path=model_path,
        normalize_obs=False,
        normalizer_path=None,
        command_interface=cmd_interface,
        training_source=f"isaac_lab_{task_name}" if task_name else "isaac_lab",
        description=f"Isaac Lab policy: {task_name}",
        tags=tags,
        deploy_contract=deploy_contract_dict,
    )


# ---------------------------------------------------------------------------
# Config extractors
# ---------------------------------------------------------------------------

def _extract_obs_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract observation configuration from env.yaml."""
    # Isaac Lab nests obs under "observations.policy" or "observations"
    obs = raw.get("observations", {})
    if isinstance(obs, dict) and "policy" in obs:
        obs = obs["policy"]
    return obs if isinstance(obs, dict) else {}


def _extract_action_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract action configuration."""
    actions = raw.get("actions", {})
    if isinstance(actions, dict) and "joint_pos" in actions:
        act = actions["joint_pos"]
        return {
            "dim": act.get("dim", 12),
            "scale": act.get("scale", 0.25),
            "use_default_offset": act.get("use_default_offset", True),
        }
    return actions if isinstance(actions, dict) else {}


def _extract_command_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract command ranges configuration."""
    cmds = raw.get("commands", {})
    if isinstance(cmds, dict) and "base_velocity" in cmds:
        return cmds["base_velocity"]
    return cmds if isinstance(cmds, dict) else {}


def _extract_robot_config(
    raw: Dict[str, Any],
    runtime_joints: List[str],
    brand_id: str,
    model_id: str,
    *,
    joints_for_regex: Optional[List[str]] = None,
    families: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract robot/articulation config.

    The robot identity (brand_id, model_id) is resolved upstream and
    passed in; this function uses it to (a) expand the regex patterns
    Isaac Lab writes under ``init_state.joint_pos`` against the actual
    runtime joint set, and (b) emit ``target_robot_models``.

    Phase 5 — ``runtime_joints`` is now the **manifest joint_names list**
    (IR roles when caller passed a RobotSpecRef; physical names on the
    legacy tuple/None path). ``joints_for_regex`` is the **physical**
    joint name list used for env.yaml regex pattern expansion (env.yaml's
    ``init_state.joint_pos`` keys are physical patterns produced by our
    env_cfg_compiler). When ``joints_for_regex`` is None, we fall back to
    ``runtime_joints`` for backwards compatibility (legacy path = same
    list).

    ``families`` is the list of family tags from the registry
    (``robots_canonical.json[<sku>].families``). The first entry becomes
    the manifest's primary ``family`` label; the full list is preserved
    so multi-family robots (e.g. ``["quadruped", "wheeled"]``) survive
    the round-trip.
    """
    fam_list = list(families or [])
    family = fam_list[0] if fam_list else ""
    result: Dict[str, Any] = {
        "family": family,
        "families": fam_list,
        "models": [f"{brand_id}_{model_id}"],
        "num_joints": len(runtime_joints) if runtime_joints else 0,
        "joint_names": list(runtime_joints),  # IR roles (Phase 5 contract)
        "joint_names_physical": list(joints_for_regex or runtime_joints),  # debug sidecar
        "default_joint_pos": {},
        "joint_pos_patterns": {},
    }

    physical_joints = list(joints_for_regex or runtime_joints)

    # Scene > articulation > robot
    scene = raw.get("scene", {})
    robot = scene.get("robot", scene.get("articulation", {}))
    if isinstance(robot, dict):
        init_state = robot.get("init_state", {})
        joint_pos = init_state.get("joint_pos", {})
        if isinstance(joint_pos, dict):
            # Keep raw patterns separately so callers can reason about
            # what Isaac Lab declared. The default_joint_pos surfaced
            # here uses concrete runtime joint names by expanding each
            # pattern across the runtime joint list. We use the *physical*
            # joint list for regex matching because env.yaml patterns are
            # physical (Isaac Lab Articulation regex-matches USD).
            result["joint_pos_patterns"] = dict(joint_pos)
            expanded: Dict[str, float] = {}
            if physical_joints:
                import re
                for pattern, val in joint_pos.items():
                    try:
                        compiled = re.compile(str(pattern))
                    except re.error as exc:
                        raise RuntimeError(
                            f"env.yaml scene.robot.init_state.joint_pos has "
                            f"invalid regex {pattern!r}: {exc}"
                        ) from exc
                    for jn in physical_joints:
                        if compiled.fullmatch(jn):
                            expanded[jn] = float(val) if isinstance(val, (int, float)) else 0.0
            result["default_joint_pos"] = expanded

        # NOTE: the legacy "derive num_joints from actuator joint_names_expr"
        # heuristic was removed in the strict-canvas migration. Phase 5
        # parsers always receive a RobotSpecRef, so ``runtime_joints`` is
        # never empty here; counting regex entries (e.g. ``[".*"]`` → 1)
        # was a known footgun that silently corrupted the joint count for
        # quadruped/biped/humanoid bundles.

    return result


# ---------------------------------------------------------------------------
# Observation layout computation
# ---------------------------------------------------------------------------

def _compute_obs_layout(
    obs_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    deploy_meta: Optional[Dict[str, Any]] = None,
    deploy_meta_group: str = "policy",
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Walk observation terms and compute layout.

    Returns (terms_list, total_dim, command_start_index).
    ``command_start_index`` is -1 if no command term found.

    Dim is sourced from the deploy_meta.json sidecar (authoritative —
    written by env_cfg_compiler._build_obs_metadata_sidecar), with
    env.yaml ``term_cfg.dim`` as a fallback. Terms whose dim cannot be
    resolved are silently skipped (lenient parse — see plan revision
    2026-05-13).
    """
    terms: List[Dict[str, Any]] = []
    total_dim = 0
    command_start_idx = -1
    num_joints_raw = robot_cfg.get("num_joints")
    num_joints = int(num_joints_raw) if isinstance(num_joints_raw, int) and num_joints_raw > 0 else 0

    sidecar_terms: Optional[Dict[str, Any]] = None
    if deploy_meta is not None:
        groups = deploy_meta.get("obs_groups")
        if isinstance(groups, dict):
            grp = groups.get(deploy_meta_group)
            if isinstance(grp, dict):
                sidecar_terms = grp

    if not obs_cfg:
        return terms, 0, -1

    for term_name, term_cfg in obs_cfg.items():
        # Lenient: anything that isn't a per-term dict (underscore-prefixed
        # metadata, ``concatenate_terms``/``enable_corruption``/... group
        # scalars added by upstream Isaac Lab versions, etc.) is skipped.
        if term_name.startswith("_"):
            continue
        if not isinstance(term_cfg, dict):
            continue

        base_name = term_name.split(".")[-1] if "." in term_name else term_name

        # Sidecar dim is authoritative when present.
        dim_raw: Any = None
        if sidecar_terms is not None:
            sc = sidecar_terms.get(base_name)
            if isinstance(sc, dict):
                dim_raw = sc.get("dim")
        if dim_raw is None:
            dim_raw = term_cfg.get("dim", None)
        if dim_raw is None:
            # Producer is expected to emit ``dim``; if it didn't, skip this
            # term rather than aborting the entire layout computation.
            log.debug(
                "_compute_obs_layout: obs term %r lacks ``dim``; skipping.",
                base_name,
            )
            continue

        if isinstance(dim_raw, str):
            if dim_raw == "num_joints":
                if num_joints <= 0:
                    log.debug(
                        "_compute_obs_layout: obs term %r uses 'num_joints' "
                        "sentinel but robot_cfg.num_joints is not populated; "
                        "skipping.",
                        base_name,
                    )
                    continue
                dim = num_joints
            else:
                # Unknown sentinel — likely a producer extension we don't
                # know about yet. Skip lenient.
                log.debug(
                    "_compute_obs_layout: unknown dim sentinel %r for obs "
                    "term %r; skipping.",
                    dim_raw, base_name,
                )
                continue
        else:
            try:
                dim = int(dim_raw)
            except (TypeError, ValueError):
                log.debug(
                    "_compute_obs_layout: obs term %r has non-int dim %r; "
                    "skipping.",
                    base_name, dim_raw,
                )
                continue

        # Check if this is the command term
        if base_name in _COMMAND_TERM_NAMES:
            command_start_idx = total_dim

        terms.append({
            "name": base_name,
            "start_idx": total_dim,
            "dim": dim,
        })
        total_dim += dim

    return terms, total_dim, command_start_idx


# ---------------------------------------------------------------------------
# DeployContract extractors (Phase 5 — Module B)
# ---------------------------------------------------------------------------

def _extract_actuator_pd(
    raw: Dict[str, Any],
    joint_order_physical: List[str],
) -> Dict[str, Any]:
    """Expand ``scene.robot.actuators.<group>`` into per-joint PD lists.

    env.yaml carries PD parameters as **scalars** keyed by an actuator group
    (e.g. ``actuators.legs.{stiffness, damping, effort_limit, velocity_limit}``)
    with a regex ``joint_names_expr`` listing which joints the scalars apply
    to. DeployContract requires **per-joint lists** of length ``n_joints``
    aligned with the physical joint order — this helper does the expansion
    and enforces full coverage.

    Returns ``{"stiffness", "damping", "effort_limit", "velocity_limit"}``
    as ``List[float]`` of length ``len(joint_order_physical)``. The optional
    DCMotor field ``saturation_effort`` is added as a list **only if** at
    least one group declared it; otherwise it is None (caller drops the
    key when serialising the contract).

    Raises ``RuntimeError`` when:
      * ``scene.robot.actuators`` is missing/empty;
      * any group's regex matches no joints with a value defined;
      * a group covers some joints but lacks one of the required scalars;
      * the same joint is matched by multiple groups with conflicting values;
      * after walking all groups, some joints remain uncovered.
    """
    actuators = (
        raw.get("scene", {}).get("robot", {}).get("actuators", {})
    ) or {}
    if not isinstance(actuators, dict) or not actuators:
        raise RuntimeError(
            "scene.robot.actuators is missing or empty in env.yaml — "
            "deploy_contract requires per-joint stiffness/damping/effort/"
            "velocity for every joint. env_cfg_compiler must emit at least "
            "one ImplicitActuatorCfg / DCMotorCfg block."
        )

    per_joint: Dict[str, Dict[str, Any]] = {}
    for group_name, group_cfg in actuators.items():
        if not isinstance(group_cfg, dict):
            continue
        patterns = group_cfg.get("joint_names_expr")
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not patterns:
            raise RuntimeError(
                f"actuators.{group_name}.joint_names_expr is missing or "
                f"empty — cannot determine which joints this group covers."
            )
        try:
            compiled = [re.compile(str(p)) for p in patterns]
        except re.error as exc:
            raise RuntimeError(
                f"actuators.{group_name}.joint_names_expr has an invalid "
                f"regex: {exc}"
            ) from exc
        matched = [
            j for j in joint_order_physical
            if any(cp.fullmatch(j) for cp in compiled)
        ]
        if not matched:
            log.warning(
                "actuators.%s.joint_names_expr=%r matched no joints in "
                "physical list %s — skipping this group.",
                group_name, patterns, joint_order_physical,
            )
            continue

        required = ("stiffness", "damping", "effort_limit", "velocity_limit")
        missing = [k for k in required if group_cfg.get(k) is None]
        if missing:
            raise RuntimeError(
                f"actuators.{group_name} is missing required scalar(s) "
                f"{missing} for joints {matched}. DeployContract requires "
                f"per-joint stiffness/damping/effort_limit/velocity_limit. "
                f"Fix in env_cfg_compiler or the actor_setting node."
            )
        sat = group_cfg.get("saturation_effort", None)
        params = {
            "stiffness": float(group_cfg["stiffness"]),
            "damping": float(group_cfg["damping"]),
            "effort_limit": float(group_cfg["effort_limit"]),
            "velocity_limit": float(group_cfg["velocity_limit"]),
            "saturation_effort": (float(sat) if sat is not None else None),
        }
        for jn in matched:
            if jn in per_joint:
                prev = per_joint[jn]
                for k, v in params.items():
                    if prev.get(k) != v:
                        raise RuntimeError(
                            f"actuator config conflict for joint {jn!r}: "
                            f"a prior group set {k}={prev.get(k)!r}; group "
                            f"{group_name!r} sets {k}={v!r}. Resolve in "
                            f"env_cfg_compiler so each joint maps to exactly "
                            f"one actuator group."
                        )
            else:
                per_joint[jn] = params

    uncovered = [j for j in joint_order_physical if j not in per_joint]
    if uncovered:
        raise RuntimeError(
            f"actuator regex coverage is incomplete: joints {uncovered} are "
            f"NOT matched by any actuators.*.joint_names_expr. "
            f"DeployContract requires a PD entry per joint. Check "
            f"scene.robot.actuators in env.yaml."
        )

    has_sat = any(
        per_joint[j]["saturation_effort"] is not None
        for j in joint_order_physical
    )
    saturation_list: Optional[List[float]] = None
    if has_sat:
        saturation_list = [
            per_joint[j]["saturation_effort"] for j in joint_order_physical
        ]
        if any(v is None for v in saturation_list):
            raise RuntimeError(
                "saturation_effort is partially declared across actuator "
                "groups; DCMotorCfg requires it per-joint. Either fill it "
                "on every group or remove it from all groups."
            )
    return {
        "stiffness": [per_joint[j]["stiffness"] for j in joint_order_physical],
        "damping": [per_joint[j]["damping"] for j in joint_order_physical],
        "effort_limit": [per_joint[j]["effort_limit"] for j in joint_order_physical],
        "velocity_limit": [per_joint[j]["velocity_limit"] for j in joint_order_physical],
        "saturation_effort": saturation_list,
    }


def _extract_obs_term_meta(
    obs_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    *,
    deploy_meta: Optional[Dict[str, Any]] = None,
    deploy_meta_group: str = "policy",
) -> List[Dict[str, Any]]:
    """Walk ``observations.policy`` and return per-term metadata for the
    DeployContract.observations block.

    Each entry has ``name``, ``dim``, ``scale``, ``clip``, ``history_length``.

    Source-priority for the metadata fields (Module A — strict mode):
      * **sidecar** (``deploy_meta.obs_groups[<group>][term]``) when
        provided. The compiler authored this file at compile time, so it
        is the **authoritative** record of per-term scale/clip/history.
        A ``None`` field in the sidecar is a strict-mode error: the
        compiler explicitly chose not to set it, which post-Module-A is
        treated as a missing user decision rather than a silent default.
        ``RuntimeError`` is raised with a message pointing at the term.
      * **env.yaml only** when no sidecar is available (back-compat for
        bundles produced before the sidecar lands, or by other producers).
        Here ``scale=None`` is permitted with a one-shot ``log.warning``
        and recorded as ``1.0`` (the historical implicit Isaac Lab
        default). This path is the only place 1.0 substitution survives
        and is gated on **absence of the authoritative source**.

    Field-level rules in either mode:
      * ``dim`` — same logic as :func:`_compute_obs_layout` (``num_joints``
        sentinel resolved against ``robot_cfg``).
      * ``clip`` — ``[lo, hi]`` or ``None``. Any other shape raises.
      * ``history_length`` — env.yaml ``0`` or missing maps to ``1`` (the
        semantic equivalent of "no history buffering"; DeployContract
        requires ``>= 1``). Values ``>= 1`` are forwarded as-is. Sidecar
        ``None`` is a strict-mode error.
    """
    if not obs_cfg:
        # Lenient: empty observations means the bundle has no terms to
        # extract. Caller decides what to do with an empty list.
        log.error(
            "observations.policy is empty in env.yaml — DeployContract will "
            "have zero terms. Check env_cfg_compiler emit logic."
        )
        return []
    sidecar_terms: Optional[Dict[str, Any]] = None
    if deploy_meta is not None:
        groups = deploy_meta.get("obs_groups")
        if isinstance(groups, dict):
            grp = groups.get(deploy_meta_group)
            if isinstance(grp, dict):
                sidecar_terms = grp

    num_joints = int(robot_cfg.get("num_joints") or 0)
    out: List[Dict[str, Any]] = []
    for term_name, term_cfg in obs_cfg.items():
        # Lenient: skip metadata keys (``_``-prefixed) and any non-dict
        # sibling (upstream Isaac Lab ``ObservationGroupCfg`` group-scalar
        # fields like ``concatenate_terms``/``enable_corruption`` get dumped
        # alongside the per-term dicts). Producer-output strict was a
        # design error — see plan revision 2026-05-13.
        if term_name.startswith("_"):
            continue
        if not isinstance(term_cfg, dict):
            continue
        base_name = term_name.split(".")[-1] if "." in term_name else term_name

        # Sidecar (authoritative when present) vs env.yaml (fallback).
        # Sidecar carries resolved int dim, scale, clip, history_length —
        # written by env_cfg_compiler._build_obs_metadata_sidecar.
        sidecar_entry: Optional[Dict[str, Any]] = (
            sidecar_terms.get(base_name) if sidecar_terms is not None else None
        )
        use_sidecar = sidecar_entry is not None

        # --- dim (sidecar first, env.yaml fallback; lenient skip if missing) ---
        dim_raw: Any = None
        if use_sidecar:
            dim_raw = sidecar_entry.get("dim")
        if dim_raw is None:
            dim_raw = term_cfg.get("dim", None)
        if dim_raw is None:
            log.debug(
                "obs_term %r: dim absent from sidecar and env.yaml; "
                "skipping.", base_name,
            )
            continue
        if isinstance(dim_raw, str):
            if dim_raw != "num_joints":
                log.debug(
                    "obs_term %r: unknown dim sentinel %r; skipping.",
                    base_name, dim_raw,
                )
                continue
            if num_joints <= 0:
                log.debug(
                    "obs_term %r: dim='num_joints' but robot_cfg.num_joints"
                    "=%d; skipping.",
                    base_name, num_joints,
                )
                continue
            dim = int(num_joints)
        else:
            try:
                dim = int(dim_raw)
            except (TypeError, ValueError):
                log.debug(
                    "obs_term %r: non-int dim %r; skipping.",
                    base_name, dim_raw,
                )
                continue

        # --- scale (lenient: missing → 1.0 default) ---
        if use_sidecar:
            scale_raw = sidecar_entry.get("scale", None)
            if scale_raw is None:
                # Sidecar carried scale=None — fall back to env.yaml then 1.0.
                scale_raw = term_cfg.get("scale", None)
        else:
            scale_raw = term_cfg.get("scale", None)
        scale: Any
        if scale_raw is None:
            log.debug(
                "obs_term %r: scale missing in both sidecar and env.yaml; "
                "defaulting to 1.0.",
                base_name,
            )
            scale = 1.0
        elif isinstance(scale_raw, (list, tuple)):
            scale = [float(v) for v in scale_raw]
            if len(scale) != dim:
                raise RuntimeError(
                    f"obs_term {base_name!r}: scale list length "
                    f"{len(scale)} != dim {dim}"
                )
        else:
            scale = float(scale_raw)

        # --- clip ---
        # Sidecar may legitimately carry ``None`` for clip (the compiler
        # emits a fallback ``(-100, 100)`` into the ObsTerm string when
        # the user does not set one — but the sidecar records the user's
        # decision verbatim, which is None). Treat sidecar-None on clip
        # as "fall through to env.yaml" so the recorded contract reflects
        # what the trained policy actually saw.
        if use_sidecar and sidecar_entry.get("clip") is not None:
            clip_raw = sidecar_entry.get("clip")
        else:
            clip_raw = term_cfg.get("clip", None)
        clip: Optional[List[float]]
        if clip_raw is None:
            clip = None
        elif isinstance(clip_raw, (list, tuple)) and len(clip_raw) == 2:
            lo, hi = float(clip_raw[0]), float(clip_raw[1])
            if lo > hi:
                raise RuntimeError(
                    f"obs_term {base_name!r}: clip lo {lo} > hi {hi}"
                )
            clip = [lo, hi]
        else:
            raise RuntimeError(
                f"obs_term {base_name!r}: clip must be null or [lo, hi]; "
                f"got {clip_raw!r}"
            )

        # --- history_length (0/missing → 1; sidecar None falls through) ---
        if use_sidecar and sidecar_entry.get("history_length") is not None:
            hl_raw: Any = sidecar_entry.get("history_length")
        else:
            hl_raw = term_cfg.get("history_length", None)
        if hl_raw is None:
            history_length = 1
        else:
            try:
                hl_int = int(hl_raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"obs_term {base_name!r}: history_length must be int, "
                    f"got {hl_raw!r}"
                ) from exc
            history_length = hl_int if hl_int >= 1 else 1

        out.append({
            "name": base_name,
            "dim": dim,
            "scale": scale,
            "clip": clip,
            "history_length": history_length,
        })

    if not out:
        log.error(
            "observations.policy has only scalar/config flags, no "
            "ObsTerm-shaped entries. Returning empty list."
        )
    return out


def _extract_action_spec(act_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the DeployContract.action sub-dict from env.yaml.

    Producer (env_cfg_compiler) is the source of truth for shape; this
    function parses leniently. Missing ``scale`` falls back to 1.0 with
    a log_debug breadcrumb; a scalar broadcast ``clip=[lo, hi]`` is
    auto-expanded into per-joint form when ``num_joints`` is known.
    """
    scale_raw = act_cfg.get("scale", None)
    scale: Any
    if scale_raw is None:
        log.debug(
            "actions.joint_pos.scale missing in env.yaml; defaulting to 1.0."
        )
        scale = 1.0
    elif isinstance(scale_raw, (list, tuple)):
        scale = [float(v) for v in scale_raw]
    else:
        scale = float(scale_raw)

    clip_raw = act_cfg.get("clip", None)
    clip: Optional[List[List[float]]]
    if clip_raw is None:
        clip = None
    elif isinstance(clip_raw, (list, tuple)):
        if (
            len(clip_raw) == 2
            and all(isinstance(v, (int, float)) for v in clip_raw)
        ):
            # Scalar broadcast form — accept and let caller expand if it
            # has num_joints. Forward as a single [lo, hi] pair wrapped
            # in a list so downstream type stays List[List[float]].
            lo, hi = float(clip_raw[0]), float(clip_raw[1])
            if lo > hi:
                raise RuntimeError(
                    f"actions.joint_pos.clip=[{lo}, {hi}]: lo > hi"
                )
            return {
                "scale": scale,
                "clip": [[lo, hi]],
                "offset_mode": (
                    "default_joint_pos"
                    if bool(act_cfg.get("use_default_offset", True))
                    else "zero"
                ),
            }
        clip = []
        for i, pair in enumerate(clip_raw):
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
            ):
                raise RuntimeError(
                    f"actions.joint_pos.clip[{i}] must be [lo, hi], "
                    f"got {pair!r}"
                )
            lo, hi = float(pair[0]), float(pair[1])
            if lo > hi:
                raise RuntimeError(
                    f"actions.joint_pos.clip[{i}]: lo {lo} > hi {hi}"
                )
            clip.append([lo, hi])
    else:
        raise RuntimeError(
            f"actions.joint_pos.clip must be null or list, got {clip_raw!r}"
        )

    use_default_offset = bool(act_cfg.get("use_default_offset", True))
    offset_mode = "default_joint_pos" if use_default_offset else "zero"

    return {"scale": scale, "clip": clip, "offset_mode": offset_mode}


def _extract_base_body_name(raw: Dict[str, Any]) -> str:
    """Resolve the articulation's base body name from env.yaml.

    Priority:
      1. ``commands.base_velocity.body_name`` (explicit override).
      2. ``"base"`` fallback with ``log.warning`` — the only allowed
         fallback in this module, justified by Isaac Lab's near-universal
         quadruped convention and by the fact that env.yaml does NOT record
         the articulation root link name today. Promote to ``raise`` once
         ``RobotSpec.base_body_name`` is added and threaded through.
    """
    cmds = raw.get("commands", {}) or {}
    base_vel = cmds.get("base_velocity", {}) if isinstance(cmds, dict) else {}
    body = base_vel.get("body_name") if isinstance(base_vel, dict) else None
    if isinstance(body, str) and body.strip():
        return body.strip()
    log.warning(
        "base_body_name not declared in env.yaml; defaulting to 'base' "
        "(Isaac Lab quadruped convention). Add RobotSpec.base_body_name "
        "to remove this fallback."
    )
    return "base"


def _extract_commands_spec(cmd_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Forward env.yaml command ranges into the DeployContract.commands dict.

    DeployContract.commands is a free-form dict (deploy_contract.py:300)
    — we use it as the runtime contract's record of velocity-2D bounds so
    PolicyRunner / live input layers can clamp user commands consistently
    with training. Empty / missing input → empty dict (deploy_contract.py
    accepts that). Each declared range is forwarded as ``[lo, hi]``.
    """
    if not cmd_cfg:
        return {}
    ranges = cmd_cfg.get("ranges", cmd_cfg)
    if not isinstance(ranges, dict):
        return {}
    forwarded: Dict[str, List[float]] = {}
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
        val = ranges.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            forwarded[key] = [float(val[0]), float(val[1])]
    return {"base_velocity": forwarded} if forwarded else {}


def _build_default_joint_pos_list(
    expanded: Dict[str, float],
    physical_joint_order: List[str],
) -> List[float]:
    """Project the ``{physical_name: value}`` dict (already regex-expanded
    by :func:`_extract_robot_config`) onto ``physical_joint_order``.

    Any missing joint is a structural error: DeployContract.default_joint_pos
    must have one entry per joint.
    """
    missing = [j for j in physical_joint_order if j not in expanded]
    if missing:
        raise RuntimeError(
            f"init_state.joint_pos missing entries for joints: {missing}. "
            f"env.yaml regex patterns under scene.robot.init_state.joint_pos "
            f"did not cover the full physical joint list."
        )
    return [float(expanded[j]) for j in physical_joint_order]


# ---------------------------------------------------------------------------
# Command field builder
# ---------------------------------------------------------------------------

def _build_command_fields(
    cmd_cfg: Dict[str, Any],
    command_start_idx: int,
) -> List[CommandField]:
    """Build CommandField list from command config and obs layout."""
    if command_start_idx < 0:
        return []

    # Extract ranges from command config
    ranges = cmd_cfg.get("ranges", cmd_cfg)
    lin_vel_x = _to_range(ranges.get("lin_vel_x", [-1.0, 1.0]))
    lin_vel_y = _to_range(ranges.get("lin_vel_y", [-0.5, 0.5]))
    ang_vel_z = _to_range(ranges.get("ang_vel_z", [-1.0, 1.0]))

    # vx default: use 40% of the positive range bound so the policy
    # gets a gentle forward-walk command when no live input is active.
    # Hardcoding 0.0 caused the downstream compat layer's
    # setdefault("vx", 0.5) to be a no-op, silently zeroing commands.
    vx_default = round(max(0.0, float(lin_vel_x[1]) * 0.4), 2)

    return [
        CommandField(
            name="vx",
            obs_index=command_start_idx,
            range=lin_vel_x,
            default=vx_default,
        ),
        CommandField(
            name="vy",
            obs_index=command_start_idx + 1,
            range=lin_vel_y,
            default=0.0,
        ),
        CommandField(
            name="vyaw",
            obs_index=command_start_idx + 2,
            range=ang_vel_z,
            default=0.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_range(val: Any) -> Tuple[float, float]:
    """Convert a range value to (min, max) tuple."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (float(val[0]), float(val[1]))
    return (-1.0, 1.0)


def _deep_get(d: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Get a nested value using dot-separated keys."""
    keys = dotted_key.split(".")
    node = d
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node
