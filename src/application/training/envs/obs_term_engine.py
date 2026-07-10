# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Shared MuJoCo observation-term engine — the single source of truth for
how a named observation term is computed from a MuJoCo ``(model, data)``
state.

Why this module exists
----------------------
Two code paths used to compute the proprioceptive obs terms **independently**:

  * training  — ``GenericMujocoEnv._build_obs`` (hardcoded 6-segment layout,
    absolute joint_pos, world-frame velocity, and a sign-flipped
    ``projected_gravity``);
  * deploy    — ``ObsBuilder._build_component`` (contract-driven, relative
    joint_pos, body-frame velocity, correct gravity).

Whenever the base tilted off level, the two disagreed on
``projected_gravity`` (x/y sign), on ``joint_pos`` (absolute vs.
default-relative) and on ``base_lin_vel`` (world vs. body frame). A policy
trained on one layout and deployed on the other is "almost right but
actually wrong" — the exact failure mode CLAUDE.md §8 forbids.

This module is the single implementation both paths now call, so
"train == deploy" holds **by construction** for every term it owns. The math
mirrors the deploy-side (Isaac-Lab-correct) convention.

Scope
-----
The engine owns the *proprioceptive* terms shared by training and deploy:
``base_ang_vel``, ``base_lin_vel``, ``projected_gravity``, ``joint_pos``
(default-relative), ``joint_vel``, ``last_action``, ``velocity_command`` /
``command`` and ``imu``. Deploy-only terms that need a ray-caster or a
reference-motion adapter (``height_scan``, ``reference_joint_*``,
``phase_sin_cos``) stay in :class:`ObsBuilder` — :func:`is_engine_term`
returns ``False`` for them and :func:`compute_term` raises, so the SB3
training path (which cannot supply that machinery) fails loud rather than
zero-filling (CLAUDE.md §8). The SB3 spec validator rejects such terms up
front (rule F3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Term naming — canonical ids + the alias table both backends share
# ---------------------------------------------------------------------------

# Canvas ``il_observation.obs_terms`` keys (env_cfg_compiler._obs_term_func),
# deploy ``ObsBuilder`` component names, and SB3 legacy names all collapse to
# one canonical id here. Keep in lockstep with ObsBuilder._DIM_ALIASES.
_TERM_ALIASES = {
    "base_angular_velocity": "base_ang_vel",
    "base_ang_vel": "base_ang_vel",
    "base_linear_velocity": "base_lin_vel",
    "base_lin_vel": "base_lin_vel",
    "projected_gravity": "projected_gravity",
    "gravity_vec": "projected_gravity",
    "joint_positions": "joint_pos",
    "joint_pos": "joint_pos",
    "joint_pos_rel": "joint_pos",
    "joint_velocities": "joint_vel",
    "joint_vel": "joint_vel",
    "joint_vel_rel": "joint_vel",
    "last_action": "last_action",
    "previous_action": "last_action",
    "actions": "last_action",
    "velocity_command": "velocity_command",
    "velocity_commands": "velocity_command",
    "command": "command",
    "imu": "imu",
}

# Canonical id → dimension. Strings resolve against the live robot at call
# time (see :func:`term_dim`); ints are fixed.
_TERM_DIM = {
    "base_ang_vel": 3,
    "base_lin_vel": 3,
    "projected_gravity": 3,
    "joint_pos": "num_joints",
    "joint_vel": "num_joints",
    "last_action": "action_dim",
    "velocity_command": 3,
    "command": 4,
    "imu": 6,
}

#: Canonical term ids this engine can compute from a MuJoCo state alone.
ENGINE_TERMS = frozenset(_TERM_DIM)


def canonical_term(name: str) -> Optional[str]:
    """Return the canonical term id for *name*, or ``None`` if unknown."""
    return _TERM_ALIASES.get(str(name))


def is_engine_term(name: str) -> bool:
    """True iff *name* (any alias) is a term this engine owns."""
    return canonical_term(name) in ENGINE_TERMS


def term_dim(name: str, *, num_joints: int, action_dim: int) -> int:
    """Resolve the flat dimension of *name* against the live robot.

    Raises ``KeyError`` for terms the engine does not own — callers must
    not silently default a dimension (CLAUDE.md §8).
    """
    canon = canonical_term(name)
    if canon not in _TERM_DIM:
        raise KeyError(
            f"obs_term_engine.term_dim: term {name!r} is not an engine term "
            f"(known: {sorted(ENGINE_TERMS)}). Deploy-only terms "
            f"(height_scan / reference_* / phase) carry their dim elsewhere."
        )
    spec = _TERM_DIM[canon]
    if spec == "num_joints":
        return int(num_joints)
    if spec == "action_dim":
        return int(action_dim)
    return int(spec)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class MujocoObsInputs:
    """Everything the per-term functions need, decoupled from either caller.

    ``mj_data`` / ``mj_model`` are a live MuJoCo state. ``joint_permute``
    maps a per-joint vector that is in MuJoCo ``qpos`` order to the order the
    consumer expects (bundle order at deploy; identity in training where the
    env already runs in actuator order). When ``None`` the engine slices the
    first ``num_joints`` entries and pads with zeros — the historical
    identity contract.
    """

    mj_model: Any
    mj_data: Any
    num_joints: int
    action_dim: int
    last_action: Optional[np.ndarray] = None
    command: Optional[Sequence[float]] = None
    default_joint_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    # "isaac_lab" → base_lin_vel rotated to body frame + joint_pos made
    # default-relative; "sb3" → world-frame lin vel + absolute joint_pos.
    # joint_pos relative-ness ALSO turns on whenever default_joint_pos is
    # present and the consumer is contract-driven (matches ObsBuilder).
    convention: str = "isaac_lab"
    wants_relative_joint_pos: bool = True
    joint_permute: Optional[Callable[[np.ndarray], np.ndarray]] = None


# ---------------------------------------------------------------------------
# Math helpers (verbatim convention from the deploy-side ObsBuilder)
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` → 3x3 rotation matrix."""
    w, x, y, z = quat.astype(float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def project_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction ``[0, 0, -1]`` expressed in the base frame.

    ``g_b = R^T @ g_w`` — the Isaac-Lab ``projected_gravity_b`` convention.
    This is the **correct** formula; the pre-2026-06 training-side helper
    flipped the x/y signs (it returned ``+R[2, :2]`` instead of ``-R[2, :2]``)
    so SB3 train and deploy disagreed the moment the base tilted.
    """
    R = _quat_to_rotation_matrix(quat_wxyz)
    return (R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)


def _unit_quat(qpos: np.ndarray) -> np.ndarray:
    quat = qpos[3:7].astype(np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("base quaternion norm is near-zero")
    return quat / norm


def world_to_body(quat_wxyz: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the base/body frame: ``v_b = R^T v_w``.

    SINGLE SOURCE for MuJoCo body-frame velocity in the SB3 stack — used by
    the obs engine (``base_lin_vel``) AND the SB3 reward
    (``velocity_tracking``), so the reward compares the actual base velocity
    against the body-frame command in the SAME frame the policy observes it
    (and the same frame Isaac Lab's ``root_lin_vel_b`` uses). SB3-only; the
    Isaac Lab reward path (env_cfg_compiler MDP terms) is independent.
    """
    R = _quat_to_rotation_matrix(np.asarray(quat_wxyz, dtype=np.float32))
    return (R.T @ np.asarray(world_vec, dtype=np.float32)).astype(np.float32)


def _world_to_body(qpos: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
    return world_to_body(_unit_quat(qpos), world_vec)


def _joint_block_in_consumer_order(
    block: np.ndarray, inp: MujocoObsInputs
) -> np.ndarray:
    """Map a qpos-order per-joint vector into the consumer's joint order."""
    if inp.joint_permute is not None:
        return np.asarray(inp.joint_permute(block), dtype=np.float32)
    n = int(inp.num_joints)
    arr = np.asarray(block, dtype=np.float32)[:n]
    if arr.shape[0] < n:
        arr = np.pad(arr, (0, n - arr.shape[0]))
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-term computation
# ---------------------------------------------------------------------------

def _t_base_ang_vel(inp: MujocoObsInputs) -> np.ndarray:
    # MuJoCo freejoint qvel[3:6] is ALREADY body-frame angular velocity —
    # returned unrotated (matches Isaac Lab root_ang_vel_b and ObsBuilder).
    qvel = np.asarray(inp.mj_data.qvel, dtype=np.float32).flatten()
    return qvel[3:6].astype(np.float32)


def _t_base_lin_vel(inp: MujocoObsInputs) -> np.ndarray:
    qvel = np.asarray(inp.mj_data.qvel, dtype=np.float32).flatten()
    world = qvel[0:3]
    if inp.convention == "isaac_lab":
        qpos = np.asarray(inp.mj_data.qpos, dtype=np.float32).flatten()
        return _world_to_body(qpos, world)
    return world.astype(np.float32)


def _t_projected_gravity(inp: MujocoObsInputs) -> np.ndarray:
    qpos = np.asarray(inp.mj_data.qpos, dtype=np.float32).flatten()
    if qpos.shape[0] < 7:
        raise ValueError("qpos too short for a base quaternion")
    return project_gravity(_unit_quat(qpos))


def _t_imu(inp: MujocoObsInputs) -> np.ndarray:
    return np.concatenate(
        [_t_base_ang_vel(inp), _t_projected_gravity(inp)]
    ).astype(np.float32)


def _t_joint_pos(inp: MujocoObsInputs) -> np.ndarray:
    qpos = np.asarray(inp.mj_data.qpos, dtype=np.float32).flatten()
    block = _joint_block_in_consumer_order(qpos[7:], inp)
    n = int(inp.num_joints)
    relative = inp.wants_relative_joint_pos or inp.convention == "isaac_lab"
    dft = np.asarray(inp.default_joint_pos, dtype=np.float32)
    if relative and dft.shape[0] == n and block.shape[0] == n:
        block = block - dft
    return block.astype(np.float32)


def _t_joint_vel(inp: MujocoObsInputs) -> np.ndarray:
    qvel = np.asarray(inp.mj_data.qvel, dtype=np.float32).flatten()
    return _joint_block_in_consumer_order(qvel[6:], inp)


def _t_last_action(inp: MujocoObsInputs) -> np.ndarray:
    n = int(inp.action_dim)
    if inp.last_action is None:
        # First-frame sentinel — every trainer initialises last_action=0 at
        # episode start, so zeros are the only in-distribution value here.
        return np.zeros(n, dtype=np.float32)
    arr = np.asarray(inp.last_action, dtype=np.float32).flatten()
    if arr.shape[0] < n:
        arr = np.pad(arr, (0, n - arr.shape[0]))
    return arr[:n].astype(np.float32)


def _command_vec(inp: MujocoObsInputs, dim: int) -> np.ndarray:
    if inp.command is None:
        raise ValueError(
            f"obs_term_engine: command term requested but command is None — "
            f"callers must pass an explicit length-{dim} command "
            f"(np.zeros({dim}) for stand-still). Implicit-zero is forbidden "
            f"(CLAUDE.md §8)."
        )
    arr = np.asarray(inp.command, dtype=np.float32).flatten()
    if arr.shape[0] < dim:
        arr = np.pad(arr, (0, dim - arr.shape[0]))
    return arr[:dim].astype(np.float32)


def command_slice_vec(command: Any, offset: int, dim: int) -> np.ndarray:
    """Read ``dim`` entries of the command vector starting at absolute ``offset``.

    The deploy-side SSOT for a command-DERIVED obs term (a gait/skill-trigger channel
    that lives past the front velocity trio, skill_command_path_design.md Slice 4). At
    training the IsaacLab command term feeds the obs directly; at deploy the whole
    command is one flat vector ordered exactly as ``deploy_contract.commands.channels``
    (velocity trio, then gait, then triggers), so the same value sits at ``offset`` =
    that channel's index. Short vectors zero-pad to reach the window (a channel the
    provider hasn't driven yet reads its ``default`` 0.0). ``command is None`` raises
    (§8 — never an implicit zero).
    """
    if command is None:
        raise ValueError(
            f"obs_term_engine.command_slice_vec: command is None but a command-slice "
            f"obs (offset={offset}, dim={dim}) was requested — pass an explicit "
            f"command vector (zeros for stand-still). Implicit-zero is forbidden (§8)."
        )
    off = int(offset)
    d = int(dim)
    if off < 0 or d <= 0:
        raise ValueError(
            f"obs_term_engine.command_slice_vec: bad offset={offset!r}/dim={dim!r}"
        )
    arr = np.asarray(command, dtype=np.float32).flatten()
    end = off + d
    if arr.shape[0] < end:
        arr = np.pad(arr, (0, end - arr.shape[0]))
    return arr[off:end].astype(np.float32)


_TERM_FN = {
    "base_ang_vel": _t_base_ang_vel,
    "base_lin_vel": _t_base_lin_vel,
    "projected_gravity": _t_projected_gravity,
    "imu": _t_imu,
    "joint_pos": _t_joint_pos,
    "joint_vel": _t_joint_vel,
    "last_action": _t_last_action,
    "velocity_command": lambda inp: _command_vec(inp, 3),
    "command": lambda inp: _command_vec(inp, 4),
}


#: Default obs layout for an SB3 canvas with NO ``il_observation`` node
#: (empty ``il_terms``). Same term set / order / dims as the pre-2026-06
#: hardcoded ``GenericMujocoEnv._build_obs`` layout, so ``obs_dim`` and the
#: deploy-contract key set are unchanged for legacy-style canvases. The
#: per-term numerics are now the engine's (correct) convention — adopting
#: this layout silently FIXES the old train≠deploy gravity-sign /
#: absolute-joint_pos / world-frame-velocity discrepancies.
DEFAULT_SB3_OBS_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "last_action",
    "velocity_command",
)


@dataclass
class ResolvedObsTerm:
    """One obs term resolved for layout: name + dim + per-term modifiers."""

    name: str
    dim: int
    scale: Any = 1.0                       # float | List[float]
    clip: Optional[tuple] = None           # (lo, hi) or None
    history_length: int = 1


def normalize_obs_layout(
    il_terms: Any,
    *,
    num_joints: int,
    action_dim: int,
) -> list:
    """Resolve a canvas ``il_terms`` mapping into an ordered term layout.

    ``il_terms`` is the ``obs_action.il_terms`` dict (term name → scale
    shorthand or ``{scale, clip, history_length}``). Insertion order is the
    policy input order. An empty / falsy mapping yields the
    :data:`DEFAULT_SB3_OBS_TERMS` layout (with a caller-emitted WARN).

    Fails loud (CLAUDE.md §8): a term the engine cannot compute from a
    MuJoCo state (``height_scan`` etc.) raises ``KeyError`` here at env-build
    time rather than being silently dropped — the SB3 spec validator also
    rejects it up front (rule F3).
    """
    import json as _json

    raw_items: list
    if not il_terms:
        raw_items = [(name, 1.0) for name in DEFAULT_SB3_OBS_TERMS]
    elif isinstance(il_terms, dict):
        raw_items = list(il_terms.items())
    else:
        raise TypeError(
            f"normalize_obs_layout: il_terms must be a dict, got "
            f"{type(il_terms).__name__}."
        )

    out: list = []
    for key, value in raw_items:
        name = str(key).strip()
        if not name:
            raise ValueError("normalize_obs_layout: empty obs term name.")
        # dim resolution doubles as the fail-loud gate for non-engine terms.
        dim = term_dim(name, num_joints=num_joints, action_dim=action_dim)

        scale: Any = 1.0
        clip: Optional[tuple] = None
        history_length = 1

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                value = _json.loads(stripped)
            else:
                value = float(stripped)

        if isinstance(value, bool) or value is None:
            raise ValueError(
                f"normalize_obs_layout: obs_terms[{name!r}] value "
                f"{type(value).__name__} is not a valid scale/config."
            )
        if isinstance(value, (int, float)):
            scale = float(value)
        elif isinstance(value, (list, tuple)):
            scale = [float(v) for v in value]
        elif isinstance(value, dict):
            sc = value.get("scale", 1.0)
            scale = (
                [float(v) for v in sc]
                if isinstance(sc, (list, tuple)) else float(sc)
            )
            clip_raw = value.get("clip")
            if clip_raw is not None:
                clip = (float(clip_raw[0]), float(clip_raw[1]))
            history_length = int(value.get("history_length", 1) or 1)
        else:
            raise ValueError(
                f"normalize_obs_layout: obs_terms[{name!r}] has unsupported "
                f"value type {type(value).__name__}."
            )

        if isinstance(scale, list) and len(scale) != dim:
            raise ValueError(
                f"normalize_obs_layout: obs_terms[{name!r}] per-component "
                f"scale length {len(scale)} != term dim {dim}."
            )
        if history_length < 1:
            raise ValueError(
                f"normalize_obs_layout: obs_terms[{name!r}].history_length "
                f"must be >= 1, got {history_length}."
            )
        out.append(
            ResolvedObsTerm(
                name=name, dim=dim, scale=scale, clip=clip,
                history_length=history_length,
            )
        )
    return out


def compute_term(name: str, inp: MujocoObsInputs) -> np.ndarray:
    """Compute observation term *name* from the MuJoCo state in *inp*.

    Raises ``KeyError`` for any term the engine does not own (deploy-only
    terms or genuinely unknown names) — never zero-fills (CLAUDE.md §8).
    """
    canon = canonical_term(name)
    if canon not in _TERM_FN:
        raise KeyError(
            f"obs_term_engine.compute_term: term {name!r} is not computable "
            f"from a MuJoCo state by this engine (known: {sorted(ENGINE_TERMS)}). "
            f"height_scan / reference_* / phase need a ray-caster or "
            f"reference-motion adapter and are handled by ObsBuilder; the SB3 "
            f"path rejects them at validation time."
        )
    return _TERM_FN[canon](inp)


__all__ = [
    "MujocoObsInputs",
    "ResolvedObsTerm",
    "ENGINE_TERMS",
    "DEFAULT_SB3_OBS_TERMS",
    "canonical_term",
    "is_engine_term",
    "term_dim",
    "normalize_obs_layout",
    "compute_term",
    "project_gravity",
    "world_to_body",
]
