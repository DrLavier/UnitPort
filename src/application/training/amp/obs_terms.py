# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AMP observation registry — the term-level single source of truth.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/amp_obs_terms.py``
(618 LoC). Registers each AMP observation term **once** with both a
motion-side slice fn (clip → numpy) and an env-side reader fn (Isaac
Lab wrapper → torch). The same six terms drive both producers, so the
expert and policy AMP obs vectors are guaranteed to share field order /
units / frame conventions — the most common silent killer of AMP
training (disc separating the two distributions purely on index drift).

Main-venv safe: only ``numpy`` is imported at module load. ``torch`` is
imported lazily inside :func:`compute_amp_obs_from_env` and the env-side
readers, so this module remains importable from the main UnitPort venv
(SB3 chain side) without dragging the heavy DL stack in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# Slice fn signature: (clip, frame_indices) -> (B, term_dim) float32
MotionSliceFn = Callable[[Any, np.ndarray], np.ndarray]
# Env-side reader signature: wrapper -> torch.Tensor (num_envs, term_dim)
EnvReadFn = Callable[[Any], Any]
# Dim resolver: ctx -> int (ctx carries num_dofs / num_feet etc.)
DimFn = Callable[[Dict[str, Any]], int]


@dataclass(frozen=True)
class AmpObsTerm:
    """One entry in the AMP obs registry.

    ``env_fn`` is optional only because main-venv callers may register
    motion-only terms during testing; production runs must use terms
    that provide both sides — :func:`compute_amp_obs_from_env` raises
    when a term used at runtime lacks ``env_fn``.
    """

    name: str
    dim_fn: DimFn
    motion_slice_fn: MotionSliceFn
    env_fn: Optional[EnvReadFn] = None
    doc: str = ""


_REGISTRY: Dict[str, AmpObsTerm] = {}


def register(term: AmpObsTerm) -> AmpObsTerm:
    if term.name in _REGISTRY:
        raise ValueError(
            f"AMP obs term {term.name!r} is already registered. "
            f"Names must be unique."
        )
    _REGISTRY[term.name] = term
    return term


def get_term(name: str) -> AmpObsTerm:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown AMP obs term {name!r}. Registered: {sorted(_REGISTRY)}"
        ) from exc


def list_terms() -> List[str]:
    return sorted(_REGISTRY)


def compute_amp_obs_dim(term_names: Sequence[str], context: Dict[str, Any]) -> int:
    """Sum the dims of *term_names* using *context* to resolve each."""
    return sum(get_term(n).dim_fn(context) for n in term_names)


def compute_amp_obs_from_motion(
    term_names: Sequence[str],
    clip: Any,
    times_sec: np.ndarray,
) -> np.ndarray:
    """Build the AMP obs vector for a batch of continuous times.

    Linear interpolation between adjacent frames (matches DEMO).
    """
    times_sec = np.asarray(times_sec, dtype=np.float64)
    n = int(times_sec.shape[0])
    frame_dt = float(getattr(clip, "frame_dt", 0.0))
    if frame_dt <= 0.0:
        raise ValueError(
            f"MotionClip must have positive frame_dt; got {frame_dt!r}"
        )

    n_frames = int(getattr(clip, "n_frames"))
    idx_f = times_sec / frame_dt
    i0 = np.clip(np.floor(idx_f).astype(np.int64), 0, n_frames - 1)
    i1 = np.minimum(i0 + 1, n_frames - 1)
    alpha = (idx_f - i0).astype(np.float32)
    alpha_col = alpha[:, None]

    pieces: List[np.ndarray] = []
    for name in term_names:
        term = get_term(name)
        a = term.motion_slice_fn(clip, i0)
        b = term.motion_slice_fn(clip, i1)
        if a.shape != b.shape:
            raise RuntimeError(
                f"Term {name!r} returned inconsistent shapes "
                f"i0={a.shape} vs i1={b.shape}"
            )
        if a.ndim != 2 or a.shape[0] != n:
            raise RuntimeError(
                f"Term {name!r} returned shape {a.shape}; expected "
                f"({n}, dim) with 2-D layout"
            )
        lerped = ((1.0 - alpha_col) * a + alpha_col * b).astype(
            np.float32, copy=False
        )
        pieces.append(lerped)
    return np.concatenate(pieces, axis=-1)


# ---------------------------------------------------------------------------
# Env-side AMP obs assembly
# ---------------------------------------------------------------------------

#: Clamp range applied per-element after concatenation. Defeats single-env
#: numerical glitches (NaN / Inf / explode after a sim instability) that
#: would otherwise poison the AMP normalizer running mean for the rest of
#: training. Matches DEMO's value.
_AMP_OBS_CLIP: float = 100.0


def compute_amp_obs_from_env(term_names: Sequence[str], wrapper: Any) -> Any:
    """Build the AMP obs tensor for the running env.

    Returns a torch tensor ``(num_envs, total_dim)`` on the env's
    device. The call delegates to each term's ``env_fn`` and
    concatenates in the order given. NaN/Inf values are replaced with
    zeros and the result is clamped to ``±_AMP_OBS_CLIP`` so a single
    bad env cannot poison the discriminator or the AMP normalizer.
    """
    import torch as _torch  # local: keep the module main-venv safe

    pieces = []
    for name in term_names:
        term = get_term(name)
        if term.env_fn is None:
            raise RuntimeError(
                f"AMP obs term {name!r} has no env_fn — cannot be used in "
                f"compute_amp_obs_from_env. Registered env_fn-less terms "
                f"are motion-side only."
            )
        x = term.env_fn(wrapper)
        if x.dim() != 2:
            raise RuntimeError(
                f"Term {name!r} env_fn returned tensor with ndim={x.dim()}, "
                f"expected 2 (num_envs, dim). Shape: {tuple(x.shape)}"
            )
        pieces.append(x)
    out = _torch.cat(pieces, dim=-1)
    out = _torch.nan_to_num(
        out, nan=0.0, posinf=_AMP_OBS_CLIP, neginf=-_AMP_OBS_CLIP
    )
    return out.clamp_(-_AMP_OBS_CLIP, _AMP_OBS_CLIP)


# ---------------------------------------------------------------------------
# Motion-side slice fns
# ---------------------------------------------------------------------------


def _joint_pos_slice(clip: Any, idx: np.ndarray) -> np.ndarray:
    arr = getattr(clip, "joint_pos", None)
    if arr is None:
        dof = int(getattr(clip, "dof", 0))
        return np.zeros((len(idx), dof), dtype=np.float32)
    return arr[idx]


def _joint_vel_slice(clip: Any, idx: np.ndarray) -> np.ndarray:
    arr = getattr(clip, "joint_vel", None)
    if arr is None:
        dof = int(getattr(clip, "dof", 0))
        return np.zeros((len(idx), dof), dtype=np.float32)
    return arr[idx]


def _toe_pos_local_slice(clip: Any, idx: np.ndarray) -> np.ndarray:
    arr = getattr(clip, "toe_pos_local", None)
    if arr is None:
        return np.zeros((len(idx), 12), dtype=np.float32)
    return arr[idx]


def _lin_vel_slice(clip: Any, idx: np.ndarray) -> np.ndarray:
    arr = getattr(clip, "lin_vel", None)
    if arr is None:
        return np.zeros((len(idx), 3), dtype=np.float32)
    return arr[idx]


def _ang_vel_slice(clip: Any, idx: np.ndarray) -> np.ndarray:
    arr = getattr(clip, "ang_vel", None)
    if arr is None:
        return np.zeros((len(idx), 3), dtype=np.float32)
    return arr[idx]


def _root_height_slice(clip: Any, idx: np.ndarray) -> np.ndarray:
    arr = getattr(clip, "root_pos", None)
    if arr is None:
        return np.zeros((len(idx), 1), dtype=np.float32)
    # Shape (T, 3) → take z column, reshape to (B, 1).
    return arr[idx, 2:3].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Env-side helpers
# ---------------------------------------------------------------------------
#
# Each env_fn is called once per rollout step with the wrapper instance.
# It reads from ``wrapper.env.unwrapped.scene["robot"]`` using Isaac
# Lab's articulation data surface. To avoid re-resolving foot body
# indices / joint permutation every step we cache them on the wrapper
# under ``_amp_term_cache``.


def _wrapper_cache(wrapper: Any) -> Dict[str, Any]:
    cache = getattr(wrapper, "_amp_term_cache", None)
    if cache is None:
        cache = {}
        try:
            wrapper._amp_term_cache = cache
        except AttributeError:
            # Some wrapper subclasses freeze attribute access; fall
            # back to a per-call cache (slow path but still correct).
            return {}
    return cache


def _robot_data(wrapper: Any) -> Any:
    return wrapper.env.unwrapped.scene["robot"].data


#: Canonical quadruped leg order. Public — this is the shared contract
#: between the motion loader (``_reorder_quad_legs`` /
#: ``_reorder_quad_feet`` in ``application/training/motion/loader.py``)
#: and the policy-side env extraction below. Both producers MUST emit
#: joint / foot tensors in ``CANONICAL_QUAD_LEGS × CANONICAL_QUAD_JOINT_SLOTS``
#: order or the discriminator trivially separates them on index position,
#: pinning ``disc_acc`` at ~1.00 regardless of policy quality.
#:
#: If you add a new per-DoF or per-foot AMP term, route its ``env_fn``
#: through ``_resolve_joint_perm`` / ``_resolve_foot_ids`` so the same
#: permutation is applied — do NOT read from ``data.joint_pos`` /
#: ``data.body_pos_w`` directly.
CANONICAL_QUAD_LEGS: tuple = ("FL", "FR", "RL", "RR")
CANONICAL_QUAD_JOINT_SLOTS: tuple = ("hip", "thigh", "calf")

# Backwards-compatible private aliases.
_CANONICAL_LEGS = CANONICAL_QUAD_LEGS
_CANONICAL_SLOTS = CANONICAL_QUAD_JOINT_SLOTS


# Stage 4: per-format IR-role → joint-name override registered by the
# launcher before training starts. When set, ``resolve_canonical_joint_perm``
# uses this table instead of the hardcoded ``{leg}_{slot}_joint`` suffix
# pattern. Lets Spot (joint ``fl_hy``, no `_joint` suffix) and other
# robots whose USD naming diverges from Unitree convention build the
# canonical permutation correctly.
_JOINT_NAMES_BY_IR_OVERRIDE: Optional[Dict[str, str]] = None


def set_canonical_joint_perm_override(
    joint_names_by_ir: Optional[Dict[str, str]],
) -> None:
    """Register an ``{ir_role: joint_name}`` table used by
    :func:`resolve_canonical_joint_perm` to look up which articulation
    joint each canonical IR role maps to.

    Pass ``None`` to clear. Set once at launcher startup from the
    spec's ``joints_role_map_for(active_format)`` (inverted).
    """
    global _JOINT_NAMES_BY_IR_OVERRIDE
    if joint_names_by_ir:
        _JOINT_NAMES_BY_IR_OVERRIDE = dict(joint_names_by_ir)
    else:
        _JOINT_NAMES_BY_IR_OVERRIDE = None


# Canonical IR role order, parallel to ``CANONICAL_QUAD_LEGS × CANONICAL_QUAD_JOINT_SLOTS``.
_CANONICAL_QUAD_IR_ROLES: tuple = (
    "hip_FL",   "thigh_FL", "calf_FL",
    "hip_FR",   "thigh_FR", "calf_FR",
    "hip_RL",   "thigh_RL", "calf_RL",
    "hip_RR",   "thigh_RR", "calf_RR",
)


def resolve_canonical_joint_perm(articulation: Any) -> List[int]:
    """Return the permutation from articulation-native joint order to
    canonical ``FL/FR/RL/RR × hip/thigh/calf`` order.

    Shared helper consumed by both the AMP obs env_fn path
    (``_resolve_joint_perm`` below, which wraps this for wrapper-level
    caching) and the RSI reset event in
    :mod:`application.training.amp.mdp_events`. Keeping one source
    avoids leg-order drift between obs computation and reset writes.

    Resolution priority (Stage 4):
      1. If :func:`set_canonical_joint_perm_override` registered a
         per-format ``{ir_role: joint_name}`` table — look up each
         canonical IR role directly in that table. Works for any joint
         naming convention (Spot, Anymal, custom rigs).
      2. Otherwise — fall back to the Unitree-style ``{leg}_{slot}_joint``
         suffix pattern (legacy default; works for Go2/A1/H1/G1).
    """
    joint_names = list(articulation.joint_names)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    perm: List[int] = []
    missing: List[str] = []

    if _JOINT_NAMES_BY_IR_OVERRIDE:
        for ir_role in _CANONICAL_QUAD_IR_ROLES:
            target = _JOINT_NAMES_BY_IR_OVERRIDE.get(ir_role, "")
            if target and target in name_to_idx:
                perm.append(name_to_idx[target])
            else:
                missing.append(f"{ir_role}->{target!r}")
    else:
        for leg in _CANONICAL_LEGS:
            for slot in _CANONICAL_SLOTS:
                target = f"{leg}_{slot}_joint"
                if target in name_to_idx:
                    perm.append(name_to_idx[target])
                else:
                    missing.append(target)

    if missing or len(perm) != 12:
        raise RuntimeError(
            "amp_obs_terms: cannot build canonical FL/FR/RL/RR x "
            "hip/thigh/calf joint permutation. Missing joint names: "
            f"{missing}. Articulation joint_names = {joint_names}. "
            "If your robot uses non-standard joint naming (e.g. BD Spot "
            "fl_hy/fl_kn), make sure the launcher calls "
            "set_canonical_joint_perm_override() with the per-format "
            "{ir_role: joint_name} table before training starts."
        )
    return perm


def _resolve_joint_perm(wrapper: Any) -> Any:
    """Wrapper-cached LongTensor view of :func:`resolve_canonical_joint_perm`.

    Without this permutation, policy-side joint_pos / joint_vel feed
    the discriminator in a different element order than expert-side
    clips (which the loader explicitly reorders via
    ``_reorder_quad_legs``). The disc then separates the two
    distributions trivially on index position, pinning disc_acc at
    ~1.00 from iter 0.
    """
    import torch as _torch
    cache = _wrapper_cache(wrapper)
    if "joint_perm" in cache:
        return cache["joint_perm"]

    robot = wrapper.env.unwrapped.scene["robot"]
    perm = resolve_canonical_joint_perm(robot)
    device = _robot_data(wrapper).joint_pos.device
    perm_tensor = _torch.as_tensor(perm, dtype=_torch.long, device=device)
    cache["joint_perm"] = perm_tensor
    print(
        f"[UnitPort][AMP] amp_obs_terms joint_perm: "
        f"asset_order={list(robot.joint_names)} → "
        f"canonical=[FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_calf] "
        f"perm={perm}",
        flush=True,
    )
    return perm_tensor


def _resolve_foot_ids(wrapper: Any) -> List[int]:
    """Resolve foot body ids in canonical FL/FR/RL/RR order.

    Matches the loader's ``_reorder_quad_feet`` output so toe_pos_local
    is bit-aligned between policy and expert sides.
    """
    cache = _wrapper_cache(wrapper)
    if "foot_ids" in cache:
        return cache["foot_ids"]
    robot = wrapper.env.unwrapped.scene["robot"]

    # Look up each canonical leg's foot body individually so the order
    # is deterministic. ``find_bodies`` with a single-leg regex returns
    # at most one hit per leg for standard quadrupeds (A1, Go2, Spot,
    # Anymal).
    canonical_foot_ids: List[int] = []
    canonical_foot_names: List[str] = []
    missing: List[str] = []
    for leg in _CANONICAL_LEGS:
        for suffix in ("_foot", "_toe"):
            ids, names = robot.find_bodies(f"{leg}{suffix}")
            if ids:
                canonical_foot_ids.append(ids[0])
                canonical_foot_names.append(names[0])
                break
        else:
            missing.append(f"{leg}_foot|{leg}_toe")

    if missing or len(canonical_foot_ids) != 4:
        raise RuntimeError(
            "amp_obs_terms: cannot find canonical FL/FR/RL/RR foot links. "
            f"Missing patterns: {missing}. body_names={robot.body_names}. "
            "Add a custom hook if your robot uses non-standard foot naming."
        )

    cache["foot_ids"] = canonical_foot_ids
    cache["foot_names"] = canonical_foot_names
    print(
        f"[UnitPort][AMP] amp_obs_terms resolved feet (canonical "
        f"FL/FR/RL/RR order): {canonical_foot_names} "
        f"→ indices {canonical_foot_ids}",
        flush=True,
    )
    return canonical_foot_ids


def _joint_pos_env(wrapper: Any):
    perm = _resolve_joint_perm(wrapper)
    return _robot_data(wrapper).joint_pos[:, perm]


def _joint_vel_env(wrapper: Any):
    perm = _resolve_joint_perm(wrapper)
    return _robot_data(wrapper).joint_vel[:, perm]


def _toe_pos_local_env(wrapper: Any):
    data = _robot_data(wrapper)
    foot_ids = _resolve_foot_ids(wrapper)
    foot_pos_w = data.body_pos_w[:, foot_ids, :]           # (N, num_feet, 3)
    root_pos_w = data.root_pos_w                           # (N, 3)
    # Translation-only body-frame convention — matches the
    # AMP_for_hardware mocap files where toe positions are stored the
    # same way. A rotated body frame would desync from the recorded
    # distribution and silently destroy training.
    foot_pos_local = foot_pos_w - root_pos_w.unsqueeze(1)
    return foot_pos_local.reshape(foot_pos_local.shape[0], -1)


def _lin_vel_env(wrapper: Any):
    data = _robot_data(wrapper)
    v = getattr(data, "root_lin_vel_b", None)
    if v is None:
        v = data.root_lin_vel_w
    return v


def _ang_vel_env(wrapper: Any):
    data = _robot_data(wrapper)
    w = getattr(data, "root_ang_vel_b", None)
    if w is None:
        w = data.root_ang_vel_w
    return w


def _root_height_env(wrapper: Any):
    data = _robot_data(wrapper)
    return data.root_pos_w[:, 2:3]


# ---------------------------------------------------------------------------
# Default term set — 43-dim amp_legged_gym layout
# ---------------------------------------------------------------------------
#
# These six terms reproduce exactly the layout that
# ``MotionClip._sample_amp_obs_batch`` and the legacy
# ``_amp_obs_extractor`` were computing before the registry existed.
# Future asset families (bipeds, humanoids) can register additional
# terms without touching the core lerp logic.

def _required_ctx(ctx: Dict[str, Any], key: str, term_name: str) -> int:
    """Read a positive-int field from the AMP obs ctx, raise on missing.

    CLAUDE.md §1.8: the previous ``ctx.get("num_dofs", 12)`` /
    ``ctx.get("num_feet", 4)`` patterns substituted Go2 morphology
    defaults for ANY robot whose ctx didn't carry the field — silently
    producing 12-DoF expert observations against a 43-joint humanoid
    policy. AMP discriminator then separated the distributions on
    pure index drift instead of motion plausibility, and the user
    saw "expert reward stuck near 0" with no diagnostic.

    Caller (motion loader / env builder) MUST populate ctx with the
    real morphology counts. Refusing to substitute defaults catches
    the wiring bug at the term registration boundary, not deep inside
    the discriminator loss.
    """
    val = ctx.get(key)
    if val is None:
        raise ValueError(
            f"[amp.obs_terms.{term_name}] required context field {key!r} "
            f"is missing. AMP obs term dimensions MUST be plumbed from "
            f"the actual robot morphology (num_dofs / num_feet) — refusing "
            f"to substitute Go2-class default. Fix the AMP env builder "
            f"to populate ctx[{key!r}] before resolving term dims "
            f"(CLAUDE.md §1.8)."
        )
    try:
        n = int(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[amp.obs_terms.{term_name}] ctx[{key!r}]={val!r} is not "
            f"a valid integer count."
        ) from exc
    if n <= 0:
        raise ValueError(
            f"[amp.obs_terms.{term_name}] ctx[{key!r}]={n} must be > 0."
        )
    return n


register(AmpObsTerm(
    name="joint_pos",
    dim_fn=lambda ctx: _required_ctx(ctx, "num_dofs", "joint_pos"),
    motion_slice_fn=_joint_pos_slice,
    env_fn=_joint_pos_env,
    doc="Per-DoF joint positions in the canonical joint order.",
))

register(AmpObsTerm(
    name="joint_vel",
    dim_fn=lambda ctx: _required_ctx(ctx, "num_dofs", "joint_vel"),
    motion_slice_fn=_joint_vel_slice,
    env_fn=_joint_vel_env,
    doc="Per-DoF joint velocities in the canonical joint order.",
))

register(AmpObsTerm(
    name="toe_pos_local",
    dim_fn=lambda ctx: 3 * _required_ctx(ctx, "num_feet", "toe_pos_local"),
    motion_slice_fn=_toe_pos_local_slice,
    env_fn=_toe_pos_local_env,
    doc="Per-foot toe position in body frame (N feet × 3 axes).",
))

register(AmpObsTerm(
    name="base_lin_vel_local",
    dim_fn=lambda ctx: 3,
    motion_slice_fn=_lin_vel_slice,
    env_fn=_lin_vel_env,
    doc="Root linear velocity in base/body frame (xyz).",
))

register(AmpObsTerm(
    name="base_ang_vel_local",
    dim_fn=lambda ctx: 3,
    motion_slice_fn=_ang_vel_slice,
    env_fn=_ang_vel_env,
    doc="Root angular velocity in base/body frame (xyz).",
))

register(AmpObsTerm(
    name="root_height",
    dim_fn=lambda ctx: 1,
    motion_slice_fn=_root_height_slice,
    env_fn=_root_height_env,
    doc="Root body height above world z=0.",
))


#: Term ordering for the 43-dim amp_legged_gym layout. Matches the DEMO
#: ``DEFAULT_QUADRUPED_TERMS`` constant exactly so existing motion files
#: + checkpoints stay bit-compatible.
DEFAULT_QUADRUPED_TERMS: List[str] = [
    "joint_pos",
    "toe_pos_local",
    "base_lin_vel_local",
    "base_ang_vel_local",
    "joint_vel",
    "root_height",
]


# ---------------------------------------------------------------------------
# Launcher-level preflight
# ---------------------------------------------------------------------------


def preflight_canonical_mapping(wrapper: Any) -> Dict[str, Any]:
    """Eagerly resolve joint permutation + canonical foot ids.

    Intended to be called **once at launcher startup** right after the
    AMP env wrapper is built. Serves two structural purposes:

    1. Surface any joint/foot naming mismatch **before** the runner
       starts training — naming drift (new robot brand, renamed URDF,
       forgotten reorder in a new term) becomes a loud crash at step 0
       instead of a silent disc-acc=1.00 degenerate run.

    2. Produce an auditable snapshot of the asset→canonical mapping.
       Callers (e.g. ``il_train_launcher``) should thread this dict
       into ``amp_alignment.json`` so the mapping is preserved for
       post-mortem and bundle export.

    Returns
    -------
    dict with keys:
        ``asset_joint_names``        — robot articulation's native order
        ``canonical_joint_perm``     — indices into ``asset_joint_names``
                                       that produce canonical
                                       FL/FR/RL/RR × hip/thigh/calf
        ``canonical_foot_names``     — foot body names in canonical order
        ``canonical_foot_ids``       — foot body ids in canonical order
    """
    perm = _resolve_joint_perm(wrapper)
    foot_ids = _resolve_foot_ids(wrapper)
    cache = _wrapper_cache(wrapper)
    robot = wrapper.env.unwrapped.scene["robot"]
    return {
        "asset_joint_names": list(robot.joint_names),
        "canonical_joint_perm": (
            perm.tolist() if hasattr(perm, "tolist") else list(perm)
        ),
        "canonical_foot_names": list(cache.get("foot_names", [])),
        "canonical_foot_ids": list(foot_ids),
    }


__all__ = [
    "AmpObsTerm",
    "CANONICAL_QUAD_LEGS",
    "CANONICAL_QUAD_JOINT_SLOTS",
    "DEFAULT_QUADRUPED_TERMS",
    "compute_amp_obs_dim",
    "compute_amp_obs_from_env",
    "compute_amp_obs_from_motion",
    "get_term",
    "list_terms",
    "preflight_canonical_mapping",
    "register",
    "resolve_canonical_joint_perm",
]
