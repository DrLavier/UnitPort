"""Isaac Lab → DeployContract compiler (single source of truth).

This module is the canonical translator from an Isaac Lab training bundle
(``env.yaml`` + ``manifest.yaml`` + optional MJCF) into the
:class:`DeployContract` intermediate representation consumed by the
runtime playback stack (``ObsBuilder`` / ``ActionApplier`` / ``PDController``
/ ``JointSpace``).

Prior to this module, two parallel Isaac Lab parsers existed — the runtime
shim ``il_manifest_compat.py`` and the skill-system entry
``skill/isaac_lab_manifest_parser.py`` — each with independent obs / action
/ PD / joint-order / height_scan logic. Any fix to one had to be duplicated
in the other or silent divergence occurred (see the 2026-04-11 height_scan
grid incident). ``sim2sim_compiler`` consolidates every bit of Isaac Lab
parsing into one module; every sim2sim consumer reads from the
``DeployContract`` this module produces, never directly from ``env.yaml``.

Public API
----------

``compile(env_yaml, joint_names, default_joint_pos, sim_dt, decimation, bundle_path) -> DeployContract | None``
    Top-level entry point. Returns a fully-assembled DeployContract or
    ``None`` when env.yaml is too sparse to populate a meaningful
    contract. Callers should fall back to legacy heuristics in that case.

The module also exposes the lower-level helpers (``recover_joint_order``,
``expand_actuator_field_per_joint``, ``extract_observations_for_contract``,
``extract_command_ranges``, ``extract_height_scan_grid_params``, …) so
other callers with their own context (a pre-built joint list, a hand-built
stub env.yaml, a custom robot) can skip the convenience wrapper and drive
the compilation step-by-step.

Design notes
------------

Single-purpose: this module produces ``DeployContract`` only. Skill
orchestration metadata (preconditions / postconditions / command interface /
source type) belongs on :class:`SkillManifest` and is NOT carried here.

Stateless + pure: no global state beyond the MJCF joint-name cache (which
is a pure memoization of disk reads). Every function takes its inputs
explicitly and returns a plain value — no hidden dependencies on the
surrounding module.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.system.policy.deploy_contract import DeployContract

log = logging.getLogger(__name__)


# ============================================================================
# MJCF joint-name cache (the only piece of state in the module)
# ============================================================================

_MJCF_JOINT_CACHE: Dict[str, List[str]] = {}


def load_mjcf_joint_names(robot_type: str) -> List[str]:
    """Return the MJCF joint name list for *robot_type* in URDF order.

    Uses ``unitree_gym_env._resolve_registered_scene_xml`` to find the
    registered scene MJCF for the brand+type, loads it via the mujoco
    Python bindings, and walks ``mj_id2name(mjOBJ_JOINT, jid)`` for
    every joint that is not a free joint or ball joint (matching the
    qpos-space construction in ``joint_space.joint_spaces_from_mj_model``).

    Cached per ``robot_type`` so the disk read + mujoco load happens at
    most once per session per robot. Returns an empty list (and logs a
    warning) if any step fails — the caller is expected to fall back to
    a hardcoded list or accept that joint_names will be empty.
    """
    key = str(robot_type or "").strip().lower()
    if not key:
        return []
    cached = _MJCF_JOINT_CACHE.get(key)
    if cached is not None:
        return list(cached)
    try:
        from src.system.training.unitree_gym_env import _resolve_registered_scene_xml
    except Exception as exc:
        log.warning("IL joint recovery: scene resolver unavailable (%s)", exc)
        _MJCF_JOINT_CACHE[key] = []
        return []
    scene_xml = _resolve_registered_scene_xml(key, "flat")
    if not scene_xml:
        log.warning(
            "IL joint recovery: no registered scene XML for robot_type=%r — "
            "non-Go2 IL bundles will fall back to empty joint_names",
            key,
        )
        _MJCF_JOINT_CACHE[key] = []
        return []
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(scene_xml)
    except Exception as exc:
        log.warning(
            "IL joint recovery: mujoco failed to load %s (%s)",
            scene_xml, exc,
        )
        _MJCF_JOINT_CACHE[key] = []
        return []
    names: List[str] = []
    try:
        for jid in range(int(getattr(model, "njnt", 0) or 0)):
            jtype = int(model.jnt_type[jid])
            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name:
                names.append(str(name))
    except Exception as exc:
        log.warning("IL joint recovery: walk failed for %s (%s)", scene_xml, exc)
        _MJCF_JOINT_CACHE[key] = []
        return []
    _MJCF_JOINT_CACHE[key] = list(names)
    return names


# ============================================================================
# Joint-order recovery from actuator regexes
# ============================================================================

def recover_joint_order(
    env_yaml: Dict[str, Any],
    mjcf_joints: List[str],
) -> List[str]:
    """Reconstruct the IL training joint order from env.yaml + MJCF joints.

    Walks ``scene.robot.actuators`` in dict insertion order, applies each
    actuator group's ``joint_names_expr`` regex list to ``mjcf_joints``,
    and concatenates the matches in MJCF order. Duplicates (joints
    matched by more than one regex) are skipped on subsequent
    occurrences so the resulting list is bijective with the URDF joint
    set the policy controls.

    Returns an empty list when ``mjcf_joints`` is empty or when the
    actuator config is missing — callers should fall back to a robot-
    specific hardcode (Go2) or accept the limitation.
    """
    if not mjcf_joints or not isinstance(env_yaml, dict):
        return []
    scene = env_yaml.get("scene") or {}
    robot = scene.get("robot") if isinstance(scene, dict) else None
    if not isinstance(robot, dict):
        robot = scene.get("articulation") if isinstance(scene, dict) else None
    if not isinstance(robot, dict):
        return []
    actuators = robot.get("actuators") or {}
    if not isinstance(actuators, dict):
        return []

    seen: set = set()
    out: List[str] = []
    for _group_name, act_cfg in actuators.items():
        if not isinstance(act_cfg, dict):
            continue
        patterns = act_cfg.get("joint_names_expr") or []
        if not isinstance(patterns, list):
            continue
        for pat in patterns:
            try:
                rx = re.compile(str(pat))
            except re.error:
                continue
            for jname in mjcf_joints:
                if jname in seen:
                    continue
                if rx.fullmatch(jname) or rx.match(jname):
                    out.append(jname)
                    seen.add(jname)
    return out


# ============================================================================
# Per-joint actuator field expansion
# ============================================================================

def expand_actuator_field_per_joint(
    env_yaml: Dict[str, Any],
    joint_names: List[str],
    field_name: str,
    default_value: float,
) -> List[float]:
    """Expand a per-group actuator field into a per-joint list aligned to joint_names.

    Isaac Lab env.yaml stores actuators as named groups, each holding ONE
    numeric value for ``stiffness``, ``damping``, ``effort_limit`` etc plus a
    ``joint_names_expr`` regex list selecting which joints the value applies
    to. This helper expands that group→regex layout into a flat per-joint
    list aligned with ``joint_names`` (the bundle/articulation order).

    Joints with no matching group fall back to ``default_value`` (and the
    caller logs a warning so the user notices).

    Returns a list of floats with len == len(joint_names).
    """
    out: List[float] = [float(default_value)] * len(joint_names)
    matched: List[bool] = [False] * len(joint_names)

    if not isinstance(env_yaml, dict):
        return out
    scene = env_yaml.get("scene") or {}
    robot = scene.get("robot") if isinstance(scene, dict) else None
    if not isinstance(robot, dict):
        robot = scene.get("articulation") if isinstance(scene, dict) else None
    if not isinstance(robot, dict):
        return out
    actuators = robot.get("actuators") or {}
    if not isinstance(actuators, dict):
        return out

    for _group_name, act_cfg in actuators.items():
        if not isinstance(act_cfg, dict):
            continue
        if field_name not in act_cfg:
            continue
        try:
            value = float(act_cfg[field_name])
        except (TypeError, ValueError):
            continue
        patterns = act_cfg.get("joint_names_expr") or []
        if not isinstance(patterns, list):
            continue
        for pat in patterns:
            try:
                rx = re.compile(str(pat))
            except re.error:
                continue
            for i, jname in enumerate(joint_names):
                if matched[i]:
                    continue
                if rx.fullmatch(jname) or rx.match(jname):
                    out[i] = value
                    matched[i] = True

    if not all(matched):
        unmatched = [j for j, m in zip(joint_names, matched) if not m]
        log.warning(
            "expand_actuator_field_per_joint(%s): %d joint(s) did not match "
            "any actuator group regex; defaulting to %s. Unmatched: %s",
            field_name,
            len(unmatched),
            default_value,
            unmatched,
        )
    return out


# ============================================================================
# Observation term dim resolution
# ============================================================================

def resolve_obs_term_dim(
    term_name: str,
    term_cfg: Dict[str, Any],
    env_yaml: Dict[str, Any],
    n_joints: int,
) -> Optional[int]:
    """Compute the flattened dim of one Isaac Lab observations.policy term.

    Priority order (first hit wins):

      1. Explicit ``dim`` field on the term. Isaac Lab does NOT currently
         serialize this — the dim is computed at env construction time by
         invoking the obs function — but we check anyway so that user-hand-
         edited env.yaml or future exporter versions can short-circuit us.

      2. ``ObsBuilder._DIM_ALIASES`` — the same table the runtime uses.
         This is the authoritative source for anything the runtime can
         consume. ``"num_joints"`` / ``"action_dim"`` sentinels resolve
         against the bundle's joint count.

      3. For ``height_scan`` / ``heightfield_scan``, compute from the
         referenced RayCaster's ``pattern_cfg`` in ``env_yaml['scene']``
         (grid_pattern size/resolution → (nx+1) * (ny+1)). Falls back to
         the 187 default if the pattern is missing or unrecognized.

      4. Name-based heuristic (joint → n_joints, action → n_joints,
         everything else → 3). Last resort; if this is hit on a term the
         caller should treat the whole contract as unreliable.

    Returns ``None`` when even the heuristic can't make sense of the term.
    """
    # ── (1) explicit dim on the term ──
    if isinstance(term_cfg, dict) and "dim" in term_cfg:
        try:
            d = int(term_cfg["dim"])
            if d > 0:
                return d
        except (TypeError, ValueError):
            pass

    # ── (2) ObsBuilder._DIM_ALIASES ──
    try:
        from src.system.policy.obs_builder import ObsBuilder
        aliases = getattr(ObsBuilder, "_DIM_ALIASES", {}) or {}
        alias = aliases.get(str(term_name))
    except Exception:
        alias = None
    if alias is not None:
        if alias == "num_joints" or alias == "action_dim":
            return int(n_joints) if n_joints > 0 else None
        try:
            val = int(alias)
            return val if val > 0 else None
        except (TypeError, ValueError):
            pass

    # ── (3) height_scan from RayCaster grid_pattern ──
    if str(term_name) in ("height_scan", "heightfield_scan"):
        dim = compute_height_scan_dim(env_yaml, term_cfg)
        if dim:
            return dim
        return 187  # Isaac Lab legged-robot default grid

    # ── (4) name-based heuristic (last resort) ──
    base = str(term_name).split(".")[-1].lower()
    if "joint" in base or "action" in base:
        return int(n_joints) if n_joints > 0 else None
    if "vel" in base or "gravity" in base or "command" in base:
        return 3
    return None


def compute_height_scan_dim(
    env_yaml: Dict[str, Any],
    term_cfg: Dict[str, Any],
) -> Optional[int]:
    """Derive height_scan dim from the sensor's grid_pattern size/resolution.

    Isaac Lab's grid_pattern emits ``(round(size_x/res)+1) * (round(size_y/res)+1)``
    rays. We walk ``scene.<name>.pattern_cfg`` where ``<name>`` is pulled
    from ``term_cfg['params']['sensor_cfg']['name']``. Returns ``None``
    when any link in the chain is missing or not a grid_pattern.
    """
    if not isinstance(env_yaml, dict):
        return None
    params = term_cfg.get("params") or {}
    sensor_cfg = params.get("sensor_cfg") if isinstance(params, dict) else None
    sensor_name = ""
    if isinstance(sensor_cfg, dict):
        sensor_name = str(sensor_cfg.get("name") or "")
    if not sensor_name:
        return None
    scene = env_yaml.get("scene") or {}
    sensor = scene.get(sensor_name) if isinstance(scene, dict) else None
    if not isinstance(sensor, dict):
        return None
    pattern = sensor.get("pattern_cfg")
    if not isinstance(pattern, dict):
        return None
    func = str(pattern.get("func") or "")
    if "grid_pattern" not in func:
        return None
    size = pattern.get("size")
    res = pattern.get("resolution")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    try:
        sx, sy = float(size[0]), float(size[1])
        r = float(res)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return None
    nx = int(round(sx / r)) + 1
    ny = int(round(sy / r)) + 1
    return nx * ny if nx > 0 and ny > 0 else None


def extract_height_scan_grid_params(
    env_yaml: Dict[str, Any],
    term_cfg: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Extract the height_scan ray-caster grid params from env.yaml.

    Returns a dict with ``nx``, ``ny``, ``resolution``, ``offset_z``,
    ``target_offset`` so the MuJoCo ObsBuilder can reproduce the exact
    same grid the policy was trained against. ``None`` when any required
    field is missing.

    ``offset_z`` is the sensor mounting z offset above the base body
    (Isaac Lab's ``height_scanner.offset.pos[2]``, typically 20 m — a
    positioning hack to ensure rays start above any possible terrain).
    ``target_offset`` is the term's ``params.offset`` (the canonical
    Isaac Lab ``mdp.height_scan`` subtracts this from the obs value).
    """
    if not isinstance(env_yaml, dict):
        return None
    params = term_cfg.get("params") or {}
    if not isinstance(params, dict):
        return None
    sensor_cfg = params.get("sensor_cfg")
    sensor_name = ""
    if isinstance(sensor_cfg, dict):
        sensor_name = str(sensor_cfg.get("name") or "")
    if not sensor_name:
        return None
    scene = env_yaml.get("scene") or {}
    sensor = scene.get(sensor_name) if isinstance(scene, dict) else None
    if not isinstance(sensor, dict):
        return None
    pattern = sensor.get("pattern_cfg")
    if not isinstance(pattern, dict):
        return None
    func = str(pattern.get("func") or "")
    if "grid_pattern" not in func:
        return None
    size = pattern.get("size")
    res = pattern.get("resolution")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    try:
        sx, sy = float(size[0]), float(size[1])
        r = float(res)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return None
    nx = int(round(sx / r)) + 1
    ny = int(round(sy / r)) + 1
    if nx <= 0 or ny <= 0:
        return None

    # Sensor mounting height (ray origin offset above base).
    offset_z = 20.0
    try:
        offset = sensor.get("offset") or {}
        pos = offset.get("pos") if isinstance(offset, dict) else None
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            offset_z = float(pos[2])
    except (TypeError, ValueError):
        pass

    # Term-level target_offset (``mdp.height_scan`` subtracts this).
    target_offset = 0.5
    try:
        raw_target = params.get("offset")
        if raw_target is not None:
            target_offset = float(raw_target)
    except (TypeError, ValueError):
        pass

    return {
        "nx": float(nx),
        "ny": float(ny),
        "resolution": float(r),
        "offset_z": float(offset_z),
        "target_offset": float(target_offset),
    }


# ============================================================================
# Canonical obs term defaults and aliases
# ============================================================================
#
# Isaac Lab's env.yaml exporter writes ``scale: null`` / ``clip: null`` even
# when the running config had explicit values — the serializer fails to
# distinguish "unset" from "explicitly set to default" for ObsTerm instances.
# These canonical tables recover the actual training-time values from the
# reference robot_lab / beyondAMP AMP locomotion configs for Unitree Go2.

CANONICAL_OBS_SCALE: Dict[str, float] = {
    "base_ang_vel":     0.2,   # beyondAMP + robot_lab default
    "base_angular_velocity": 0.2,
    "joint_vel":        0.05,  # beyondAMP + robot_lab default
    "joint_vel_rel":    0.05,
    "joint_velocities": 0.05,
    # Others default to 1.0
}

CANONICAL_OBS_CLIP: Dict[str, Tuple[float, float]] = {
    "last_action":      (-100.0, 100.0),
    "actions":          (-100.0, 100.0),
    "previous_action":  (-100.0, 100.0),
    "base_lin_vel":     (-100.0, 100.0),
    "base_linear_velocity": (-100.0, 100.0),
    "base_ang_vel":     (-100.0, 100.0),
    "base_angular_velocity": (-100.0, 100.0),
    "joint_pos":        (-100.0, 100.0),
    "joint_pos_rel":    (-100.0, 100.0),
    "joint_positions":  (-100.0, 100.0),
    "joint_vel":        (-100.0, 100.0),
    "joint_vel_rel":    (-100.0, 100.0),
    "joint_velocities": (-100.0, 100.0),
    # height_scan training-time clip is (-1, 1) in robot_lab.
    "height_scan":      (-1.0, 1.0),
    "heightfield_scan": (-1.0, 1.0),
}

# Canonical declaration order used by Unitree Go2 AMP reference configs
# (robot_lab + beyondAMP both declare in this exact sequence). Isaac Lab's
# env.yaml exporter is known to alphabetize obs terms when dumping, which
# destroys the declaration order the policy was trained against. When the
# set of terms in env.yaml matches one of these canonical signatures, we
# reorder the extracted dict to match the training-time order.
CANONICAL_OBS_ORDER_8TERM: Tuple[str, ...] = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_command",   # also velocity_commands
    "joint_pos",          # also joint_pos_rel
    "joint_vel",          # also joint_vel_rel
    "last_action",        # also actions, previous_action
    "height_scan",
)

TERM_NAME_ALIASES: Dict[str, Tuple[str, ...]] = {
    "base_lin_vel":       ("base_lin_vel", "base_linear_velocity"),
    "base_ang_vel":       ("base_ang_vel", "base_angular_velocity"),
    "projected_gravity":  ("projected_gravity", "gravity_vec"),
    "velocity_command":   ("velocity_command", "velocity_commands"),
    "joint_pos":          ("joint_pos", "joint_pos_rel", "joint_positions"),
    "joint_vel":          ("joint_vel", "joint_vel_rel", "joint_velocities"),
    "last_action":        ("last_action", "actions", "previous_action"),
    "height_scan":        ("height_scan", "heightfield_scan"),
}


def canonicalize_term_category(term_name: str) -> Optional[str]:
    """Map a concrete obs term name to its canonical category key."""
    lname = str(term_name)
    for canonical, aliases in TERM_NAME_ALIASES.items():
        if lname in aliases:
            return canonical
    return None


# ============================================================================
# Observations / commands / base body extraction
# ============================================================================

def extract_observations_for_contract(
    env_yaml: Dict[str, Any],
    n_joints: int,
) -> Dict[str, Dict[str, Any]]:
    """Extract observations.policy terms with scale/clip/history_length.

    Returns an ordered dict where each value is a dict with the keys
    deploy_contract.ObsTermSpec.from_dict expects: ``dim``, ``scale``,
    ``clip``, ``history_length``, and (for height_scan) ``grid_params``.
    Strips Isaac Lab internals (``func``, ``params``, ``noise``,
    ``modifiers``, ``flatten_history_dim``).

    When env.yaml's ``scale`` / ``clip`` fields are null (Isaac Lab's
    exporter commonly elides them even when the training config set
    non-default values), fall back to the canonical defaults from the
    robot_lab / beyondAMP AMP reference configs. When the set of term
    names matches the canonical 8-term Unitree Go2 AMP signature, the
    output is REORDERED from env.yaml's alphabetical-by-exporter order
    into the canonical declaration order the policy was actually trained
    against. This is the single biggest source of sim2sim divergence
    for Isaac Lab AMP bundles.
    """
    obs_root = env_yaml.get("observations") if isinstance(env_yaml, dict) else None
    if isinstance(obs_root, dict) and "policy" in obs_root:
        obs_block = obs_root["policy"]
    else:
        obs_block = obs_root if isinstance(obs_root, dict) else {}
    if not isinstance(obs_block, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for term_name, term_cfg in obs_block.items():
        if not isinstance(term_cfg, dict):
            continue
        if str(term_name).startswith("_"):
            continue

        # Dim resolution: shared authoritative table with runtime ObsBuilder.
        dim = resolve_obs_term_dim(str(term_name), term_cfg, env_yaml, n_joints)
        if dim is None:
            log.warning(
                "sim2sim_compiler: could not determine dim for obs term %r "
                "(not in _DIM_ALIASES, not height_scan, heuristic failed); "
                "skipping term.",
                term_name,
            )
            continue
        if dim <= 0:
            continue

        # Scale: env.yaml is the authoritative source when set. When null,
        # fall back to the canonical per-term default (base_ang_vel=0.2,
        # joint_vel=0.05) that matches the reference Isaac Lab AMP config.
        # Using 1.0 here instead is the single biggest cause of sim2sim
        # divergence: the policy sees obs values 5-20× larger than training
        # and produces action magnitudes that saturate the PD effort limit
        # on the first step, then diverge into thrashing via feedback.
        raw_scale = term_cfg.get("scale")
        if raw_scale is None:
            scale: Any = CANONICAL_OBS_SCALE.get(str(term_name), 1.0)
            if scale != 1.0:
                log.info(
                    "sim2sim_compiler: obs term %r has scale=null in env.yaml; "
                    "falling back to canonical default %.3f (matches reference "
                    "Isaac Lab AMP config).",
                    term_name, scale,
                )
        elif isinstance(raw_scale, (list, tuple)):
            try:
                scale = [float(v) for v in raw_scale]
            except (TypeError, ValueError):
                scale = 1.0
        else:
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError):
                scale = 1.0

        # Clip: same story — canonical defaults (±100 for most, ±1 for
        # height_scan) when env.yaml is null.
        raw_clip = term_cfg.get("clip")
        clip: Any = None
        if isinstance(raw_clip, (list, tuple)) and len(raw_clip) == 2:
            try:
                clip = [float(raw_clip[0]), float(raw_clip[1])]
            except (TypeError, ValueError):
                clip = None
        if clip is None:
            fallback = CANONICAL_OBS_CLIP.get(str(term_name))
            if fallback is not None:
                clip = [fallback[0], fallback[1]]

        # history_length: Isaac Lab uses 0 to mean "no history" but our
        # contract uses >=1 (1 == single frame). Coerce 0 → 1.
        history_length = int(term_cfg.get("history_length") or 0)
        if history_length < 1:
            history_length = 1

        term_entry: Dict[str, Any] = {
            "dim": dim,
            "scale": scale,
            "clip": clip,
            "history_length": history_length,
        }

        # height_scan: capture the training-time ray-caster grid so the
        # MuJoCo runtime can reproduce exactly the same number of rays
        # with the same geometric layout. Without this, obs_builder
        # falls back to its 17×11 hardcoded default, which silently
        # produces the wrong obs dim for any non-default grid and
        # shifts every downstream obs term by the dim delta.
        if str(term_name) in ("height_scan", "heightfield_scan"):
            grid_params = extract_height_scan_grid_params(env_yaml, term_cfg)
            if grid_params is not None:
                term_entry["grid_params"] = grid_params
                expected = int(grid_params["nx"]) * int(grid_params["ny"])
                if expected != dim:
                    log.warning(
                        "sim2sim_compiler: height_scan dim (%d) disagrees "
                        "with nx*ny from grid_params (%d); trusting grid_params "
                        "so the runtime ray cast matches the training-time grid.",
                        dim, expected,
                    )
                    term_entry["dim"] = expected

        out[str(term_name)] = term_entry

    # ── Canonical reorder ─────────────────────────────────────────────
    # When the term set matches the canonical 8-term Unitree Go2 AMP
    # signature, reorder to the declaration order used by the reference
    # configs. env.yaml's alphabetical-by-exporter order does NOT match
    # the policy's training-time input order.
    categories = [canonicalize_term_category(n) for n in out.keys()]
    if None not in categories and set(categories) == set(CANONICAL_OBS_ORDER_8TERM):
        name_by_category = {
            canonicalize_term_category(n): n for n in out.keys()
        }
        reordered: Dict[str, Dict[str, Any]] = {}
        for canonical in CANONICAL_OBS_ORDER_8TERM:
            actual_name = name_by_category[canonical]
            reordered[actual_name] = out[actual_name]
        if list(reordered.keys()) != list(out.keys()):
            log.info(
                "sim2sim_compiler: detected canonical 8-term Go2 AMP obs "
                "signature; reordering from env.yaml order %s to canonical "
                "training order %s.",
                list(out.keys()), list(reordered.keys()),
            )
        out = reordered
    return out


def extract_command_ranges(env_yaml: Dict[str, Any]) -> Dict[str, Any]:
    """Extract commands.base_velocity.ranges from env.yaml.

    Returns ``{"base_velocity": {"ranges": {"lin_vel_x": [lo, hi], ...}}}``
    or an empty dict when nothing is found. The deploy_contract preserves
    these as a free-form dict — it does not currently validate the inner
    structure beyond ``isinstance(commands, dict)``.
    """
    cmds_root = env_yaml.get("commands") if isinstance(env_yaml, dict) else None
    if not isinstance(cmds_root, dict):
        return {}
    base_vel = cmds_root.get("base_velocity")
    if not isinstance(base_vel, dict):
        return {}
    # Isaac Lab uses either .ranges (BaseVelocityCommand) or .limit_ranges
    # (curriculum-wrapped versions).
    ranges = base_vel.get("ranges") or base_vel.get("limit_ranges") or {}
    if not isinstance(ranges, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        val = ranges.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                out[key] = [float(val[0]), float(val[1])]
            except (TypeError, ValueError):
                continue
    if not out:
        return {}
    return {"base_velocity": {"ranges": out}}


def infer_base_body_name(env_yaml: Dict[str, Any]) -> str:
    """Best-effort base body name extraction from env.yaml.

    Isaac Lab stores the floating base body indirectly via prim_path (e.g.
    ``/World/envs/env_.*/Robot/base``). The leaf name is what MuJoCo's
    ``mj_name2id`` will look up later. Returns an empty string when nothing
    plausible is found — callers should fall back to MJCF root inference at
    consume time.
    """
    if not isinstance(env_yaml, dict):
        return ""
    scene = env_yaml.get("scene") or {}
    robot = scene.get("robot") if isinstance(scene, dict) else None
    if not isinstance(robot, dict):
        robot = scene.get("articulation") if isinstance(scene, dict) else None
    if not isinstance(robot, dict):
        return ""
    prim_path = robot.get("prim_path") or ""
    if isinstance(prim_path, str) and "/" in prim_path:
        leaf = prim_path.rsplit("/", 1)[-1]
        if leaf:
            return str(leaf)
    return ""


# ============================================================================
# init_state joint_pos pattern expansion
# ============================================================================

def expand_joint_pos_pattern_dict(
    pattern_dict: Dict[str, Any],
    joint_names: List[str],
) -> List[float]:
    """Expand an Isaac Lab regex-keyed init_state.joint_pos dict.

    Isaac Lab's env.yaml stores ``init_state.joint_pos`` as a dict whose
    keys are *regex patterns* (e.g. ``.*L_hip_joint``, ``F[L,R]_thigh_joint``)
    rather than concrete joint names. Each pattern matches one or more
    joints; the value is applied to every match. Unmatched joints fall
    back to ``0.0``.

    Returns a list aligned with ``joint_names``.
    """
    out = [0.0] * len(joint_names)
    if not isinstance(pattern_dict, dict):
        return out
    for pattern, value in pattern_dict.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        try:
            rx = re.compile(str(pattern))
        except re.error:
            continue
        for i, jname in enumerate(joint_names):
            if rx.fullmatch(jname) or rx.match(jname):
                out[i] = v
    return out


# ============================================================================
# Utility: env.yaml read + brand/type parsing + ONNX cross-check
# ============================================================================

def read_env_yaml(bundle_path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort load of the Isaac Lab env.yaml stored next to the manifest.

    Returns ``None`` when the file is missing or unparseable. Uses the same
    custom YAML loader the IL manifest parser registers (handles
    ``!python/tuple`` / ``!python/object/apply:builtins.slice``).
    """
    env_yaml = bundle_path / "env.yaml"
    if not env_yaml.is_file():
        return None
    try:
        import yaml as _yaml
        from src.system.skill.isaac_lab_manifest_parser import _IsaacLabEnvYamlLoader
        with env_yaml.open("r", encoding="utf-8") as fh:
            return _yaml.load(fh, Loader=_IsaacLabEnvYamlLoader)
    except Exception:
        return None


def normalize_robot_brand_and_type(target_models: Any) -> Tuple[str, str]:
    """Split a target_robot_models entry into ``(brand, type)``.

    ``parse_isaac_lab_env_yaml`` writes entries like ``"unitree_go2"`` or
    ``"boston_dynamics_spot"``. UnitPort's v1 layout wants brand and type
    separately so the runtime adapter can dispatch.
    """
    brand = ""
    rtype = ""
    if isinstance(target_models, list) and target_models:
        raw = str(target_models[0] or "").strip().lower()
        for prefix in ("unitree_", "boston_dynamics_", "bostondynamics_", "xiaomi_"):
            if raw.startswith(prefix):
                brand = prefix.rstrip("_")
                rtype = raw[len(prefix):]
                break
        if not brand:
            # No prefix → assume the entire string is the type and pick a
            # brand by family. Best guess: Unitree (only IL preset right now).
            brand = "unitree"
            rtype = raw
    return brand, rtype


def read_policy_onnx_obs_dim(bundle_path: Optional[Path]) -> Optional[int]:
    """Return the ``obs`` input dim of ``<bundle_path>/policy.onnx`` or None.

    Failures are swallowed — this is a best-effort cross-check used to
    decide whether to *trust* a heuristically-assembled deploy_contract,
    not to gate bundle loading. If onnx isn't installed, or the file is
    missing, or the first input isn't 2-D, just return None and the
    caller will assume the contract is correct.
    """
    if bundle_path is None:
        return None
    p = Path(bundle_path) / "policy.onnx"
    if not p.is_file():
        return None
    try:
        import onnx  # type: ignore
        model = onnx.load(str(p))
        for inp in model.graph.input:
            shape = inp.type.tensor_type.shape
            dims = [d.dim_value for d in shape.dim]
            if len(dims) >= 2:
                return int(dims[-1])
    except Exception:
        return None
    return None


# ============================================================================
# Deploy contract assembly
# ============================================================================

def assemble_deploy_contract_dict(
    env_yaml: Dict[str, Any],
    joint_names: List[str],
    default_joint_pos: List[float],
    sim_dt: Optional[float],
    decimation: Optional[int],
    bundle_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Build a deploy_contract dict from a parsed Isaac Lab env.yaml.

    Returns None when env.yaml lacks the minimum data needed to populate a
    meaningful contract — callers should NOT crash, just skip the contract
    section and let the legacy v1 path handle the bundle.

    The returned dict is shaped exactly like ``DeployContract.to_dict()``
    output (and will round-trip through ``DeployContract.from_dict``):

      {
        "schema_version": 1,
        "joint_sdk_names": [...],         # == joint_names (bundle order)
        "joint_ids_map":   [0, 1, ..., n-1],  # identity for IL imports
        "stiffness":       [...],          # per-joint, expanded from groups
        "damping":         [...],
        "effort_limit":    [...],
        "default_joint_pos": [...],        # already in bundle order from caller
        "sim_dt": float,
        "step_dt": sim_dt * decimation,
        "decimation": int,
        "observations": { name: {dim, scale, clip, history_length, grid_params?}, ... },
        "action": {scale, clip, offset_mode},
        "commands": {"base_velocity": {"ranges": {...}}},
        "base_body_name": str,
      }

    Why joint_ids_map is identity
    -----------------------------
    The unitree_rl_lab exporter uses
    ``resolve_matching_names(asset.data.joint_names, joint_sdk_names)``
    to compute joint_ids_map. ``joint_sdk_names`` is set in the Isaac Lab
    Python config class (NOT in env.yaml), so an importer reading only
    env.yaml cannot recover the original mapping. We instead set
    ``joint_sdk_names = joint_names`` (== bundle/articulation order, the
    only ordering env.yaml gives us) and use the identity map. This makes
    the contract internally consistent: stiffness[i] always corresponds to
    the joint at joint_sdk_names[i]. The trade-off is that downstream code
    that wants the *real* hardware SDK order has to look it up elsewhere
    (e.g. service/adapters/unitree_sdk2). Sim2sim does not need the wire
    order — it only needs the policy's training order, which IS the
    bundle order, which IS the contract.
    """
    if not joint_names:
        return None
    n = len(joint_names)

    if not isinstance(env_yaml, dict):
        return None

    # Decimation must be a positive integer; sim_dt must be > 0.
    if sim_dt is None or sim_dt <= 0:
        return None
    try:
        decimation_int = int(decimation) if decimation is not None else 0
    except (TypeError, ValueError):
        decimation_int = 0
    if decimation_int < 1:
        return None
    step_dt = float(sim_dt) * decimation_int

    # Per-joint actuator fields. Defaults match Isaac Lab Go2 stock values
    # so the resulting contract is at least sane when actuators are absent.
    stiffness = expand_actuator_field_per_joint(
        env_yaml, joint_names, "stiffness", default_value=25.0
    )
    damping = expand_actuator_field_per_joint(
        env_yaml, joint_names, "damping", default_value=0.5
    )
    effort_limit = expand_actuator_field_per_joint(
        env_yaml, joint_names, "effort_limit", default_value=33.5
    )

    # default_joint_pos comes from the caller (already expanded from the
    # init_state regex dict via expand_joint_pos_pattern_dict).
    default_pos_list = [float(v) for v in default_joint_pos[:n]]
    while len(default_pos_list) < n:
        default_pos_list.append(0.0)

    # Observations — must be non-empty for the contract to be meaningful.
    observations = extract_observations_for_contract(env_yaml, n)
    if not observations:
        log.warning(
            "sim2sim_compiler: env.yaml has no observations.policy terms; "
            "skipping deploy_contract assembly. Bundle will load via "
            "the legacy path."
        )
        return None

    # Action term — extract from actions.joint_pos when present (the IL
    # locomotion default). Other action types fall back to scale=1, no clip.
    action_block = env_yaml.get("actions") if isinstance(env_yaml, dict) else None
    if isinstance(action_block, dict) and "joint_pos" in action_block:
        joint_pos_action = action_block["joint_pos"] or {}
        try:
            action_scale: Any = float(joint_pos_action.get("scale", 0.25))
        except (TypeError, ValueError):
            action_scale = 0.25
        # Isaac Lab actions use_default_offset=True is the locomotion norm.
        offset_mode = (
            "default_joint_pos"
            if joint_pos_action.get("use_default_offset", True)
            else "zero"
        )
        action_dict: Dict[str, Any] = {
            "scale": action_scale,
            "clip": None,  # IL locomotion typically does not clip raw actions
            "offset_mode": offset_mode,
        }
    else:
        action_dict = {
            "scale": 1.0,
            "clip": None,
            "offset_mode": "default_joint_pos",
        }

    commands = extract_command_ranges(env_yaml)
    base_body_name = infer_base_body_name(env_yaml)

    contract: Dict[str, Any] = {
        "schema_version": 1,
        "joint_sdk_names": list(joint_names),
        "joint_ids_map": list(range(n)),
        "stiffness": stiffness,
        "damping": damping,
        "effort_limit": effort_limit,
        "default_joint_pos": default_pos_list,
        "sim_dt": float(sim_dt),
        "step_dt": step_dt,
        "decimation": decimation_int,
        "observations": observations,
        "action": action_dict,
        "commands": commands,
        "base_body_name": base_body_name,
    }

    # ── Post-assembly cross-check against policy.onnx ─────────────────
    # If we can read the ONNX and its obs input dim disagrees with the
    # sum of contract term dims, the contract is demonstrably wrong —
    # drop it instead of shipping numbers that will fail compat checks
    # downstream. The legacy IL path will then handle the bundle.
    contract_total = sum(
        int(t.get("dim", 0)) * max(1, int(t.get("history_length", 1) or 1))
        for t in observations.values()
    )
    onnx_obs_dim = read_policy_onnx_obs_dim(bundle_path)
    if onnx_obs_dim is not None and contract_total != onnx_obs_dim:
        log.warning(
            "sim2sim_compiler: contract total %d does not match "
            "policy.onnx obs input dim %d — contract is unreliable, dropping "
            "it so the legacy IL path handles this bundle. Term dims: %s",
            contract_total,
            onnx_obs_dim,
            {k: v.get("dim") for k, v in observations.items()},
        )
        return None

    return contract


# ============================================================================
# Top-level convenience API
# ============================================================================

def compile(
    env_yaml: Dict[str, Any],
    joint_names: List[str],
    default_joint_pos: List[float],
    sim_dt: Optional[float],
    decimation: Optional[int],
    bundle_path: Optional[Path] = None,
) -> Optional[DeployContract]:
    """Compile an Isaac Lab env.yaml into a ``DeployContract`` object.

    Convenience wrapper over :func:`assemble_deploy_contract_dict` that
    returns a validated :class:`DeployContract` dataclass instead of a
    raw dict. Callers who need the dict form (e.g. to write it into a
    manifest.yaml before ``DeployContract.from_dict`` runs) should use
    :func:`assemble_deploy_contract_dict` directly.

    Returns ``None`` when env.yaml is too sparse to produce a meaningful
    contract — the caller must then fall back to legacy heuristics.

    Parameters
    ----------
    env_yaml:
        Parsed Isaac Lab env.yaml dict. May be empty / invalid; we
        return ``None`` in that case.
    joint_names:
        Bundle-order joint name list, typically produced by
        :func:`recover_joint_order` from env.yaml + MJCF joints.
    default_joint_pos:
        Per-joint default position, aligned with ``joint_names``,
        typically produced by :func:`expand_joint_pos_pattern_dict`.
    sim_dt:
        Physics step size from ``env_yaml['sim']['dt']``.
    decimation:
        Control decimation from ``env_yaml['decimation']`` — policy
        runs every ``decimation`` sim steps.
    bundle_path:
        Optional bundle directory for ONNX cross-check of the assembled
        contract's observation dim against the trained policy's actual
        input shape. When they disagree, the contract is dropped.
    """
    contract_dict = assemble_deploy_contract_dict(
        env_yaml=env_yaml,
        joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        sim_dt=sim_dt,
        decimation=decimation,
        bundle_path=bundle_path,
    )
    if contract_dict is None:
        return None
    try:
        return DeployContract.from_dict(contract_dict)
    except ValueError as exc:
        log.error(
            "sim2sim_compiler.compile: DeployContract validation failed "
            "on assembled dict (%s). Falling back to None so caller uses "
            "the legacy IL path.",
            exc,
        )
        return None
