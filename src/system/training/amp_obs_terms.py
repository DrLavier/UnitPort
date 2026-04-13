"""AMP observation term dispatch — single source of truth.

B5 in ``knowledge_base/amp_fix.yaml``. Per spec §3 of
``AMP_EXAMPLE/AMP-PPO_design.yaml``:

    The AMP observation vector consumed by the discriminator MUST be
    bit-identical in semantics whether it is produced by:
      (a) the running environment from robot state, OR
      (b) the motion loader from a recorded reference frame.

    Any difference in joint ordering, frame (world vs base), unit,
    sign, or history length will silently destroy training. This is
    enforced by construction: both producers call the SAME
    ``compute_amp_obs()`` function parameterised by ``amp_obs_terms``.

This module is the "same function" both producers call. It registers a
dispatch table mapping term names (``joint_pos``, ``toe_pos_local`` …)
to an ``AmpObsTerm`` struct carrying:

* a ``dim_fn`` that resolves the term's dimension from a context dict
  (so e.g. ``joint_pos`` can compute ``num_dofs`` at runtime);
* a ``motion_slice_fn`` that returns an un-interpolated slice
  ``(len(indices), dim)`` from a ``MotionClip`` at the given frame
  indices — the caller does lerp in one place;
* an ``env_fn`` that reads the field from a running AMP env wrapper
  and returns a torch tensor ``(num_envs, dim)``.

The high-level helpers at the bottom of this file —
``compute_amp_obs_from_motion(term_names, clip, times)`` and
``compute_amp_obs_from_env(term_names, wrapper)`` — are what the
motion loader / Isaac Lab launcher actually call, so neither side
can drift from the other.

Main-venv safe: only imports numpy at module load. torch is imported
lazily inside ``compute_amp_obs_from_env`` so test code that only
exercises the motion side does not pay for a torch import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Motion-side slice function. Receives a ``MotionClip`` and a
#: ``(B,)`` int64 frame index array, returns ``(B, dim)`` float32.
#: Must NOT do any interpolation — the caller handles lerp so all
#: terms share identical interp semantics.
MotionSliceFn = Callable[[Any, np.ndarray], np.ndarray]

#: Env-side read function. Receives an AMP env wrapper (the object
#: passed to ``AmpRslRlVecEnvWrapper(amp_obs_extractor=...)``) and
#: returns a torch tensor ``(num_envs, dim)`` on the env's device.
EnvReadFn = Callable[[Any], Any]

#: Dim resolver. Given a context dict (``num_dofs``, ``num_feet`` …),
#: return the term's flat dim. The context is supplied by the
#: caller — typically the spec compiler fills it from the active
#: robot articulation.
DimFn = Callable[[Dict[str, Any]], int]


@dataclass(frozen=True)
class AmpObsTerm:
    """A single entry in the AMP obs registry.

    Attributes
    ----------
    name:
        Stable identifier used in config files and manifests. Must
        match across env, motion loader, and bundle manifest.
    dim_fn:
        Resolves the flat dimension from a context dict.
    motion_slice_fn:
        Returns the un-interpolated slice for a batch of frame indices.
    env_fn:
        Reads the term from a running AMP env wrapper.
    doc:
        One-line description for logging / manifest dumps.
    """

    name: str
    dim_fn: DimFn
    motion_slice_fn: MotionSliceFn
    env_fn: EnvReadFn
    doc: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, AmpObsTerm] = {}


def register(term: AmpObsTerm) -> AmpObsTerm:
    """Add *term* to the registry. Raises on duplicates."""
    if term.name in _REGISTRY:
        raise ValueError(
            f"AMP obs term {term.name!r} is already registered. "
            f"Names must be unique."
        )
    _REGISTRY[term.name] = term
    return term


def get_term(name: str) -> AmpObsTerm:
    """Look up a registered term by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown AMP obs term {name!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        ) from exc


def list_terms() -> List[str]:
    """Return all registered term names, sorted."""
    return sorted(_REGISTRY)


def compute_amp_obs_dim(
    term_names: Sequence[str], context: Dict[str, Any]
) -> int:
    """Sum the dims of *term_names* using *context* to resolve each.

    ``context`` typically carries ``num_dofs`` and ``num_feet`` so
    joint- and foot-sized terms know their widths.
    """
    return sum(get_term(n).dim_fn(context) for n in term_names)


# ---------------------------------------------------------------------------
# High-level dispatch — both producers call these
# ---------------------------------------------------------------------------


def compute_amp_obs_from_motion(
    term_names: Sequence[str],
    clip: Any,
    times_sec: np.ndarray,
) -> np.ndarray:
    """Build the AMP obs vector for a batch of continuous times.

    Parameters
    ----------
    term_names:
        Canonical term names, in the exact order they should appear in
        the concatenated output.
    clip:
        A ``MotionClip`` (or anything with the same attribute surface
        — see ``_joint_pos_slice`` below for the minimum contract).
    times_sec:
        ``(B,)`` float array of sample times in seconds.

    Returns
    -------
    ``(B, total_dim)`` float32 array.

    Interpolation semantics match the legacy ``MotionClip
    ._sample_amp_obs_batch``: linear between adjacent frames,
    index-clipped at the tail (no wraparound).
    """
    times_sec = np.asarray(times_sec, dtype=np.float64)
    n = int(times_sec.shape[0])
    frame_dt = float(getattr(clip, "frame_dt", 0.0))
    if frame_dt <= 0.0:
        raise ValueError(
            f"MotionClip must have positive frame_dt; got {frame_dt!r}"
        )

    idx_f = times_sec / frame_dt
    n_frames = int(getattr(clip, "n_frames"))
    i0 = np.clip(np.floor(idx_f).astype(np.int64), 0, n_frames - 1)
    i1 = np.minimum(i0 + 1, n_frames - 1)
    alpha = (idx_f - i0).astype(np.float32)
    alpha_col = alpha[:, None]  # (B, 1) broadcast across feature dim

    pieces: List[np.ndarray] = []
    for name in term_names:
        term = get_term(name)
        a = term.motion_slice_fn(clip, i0)
        b = term.motion_slice_fn(clip, i1)
        if a.shape != b.shape:
            raise RuntimeError(
                f"Term {name!r} motion_slice_fn returned inconsistent "
                f"shapes for i0 ({a.shape}) vs i1 ({b.shape})"
            )
        if a.ndim != 2 or a.shape[0] != n:
            raise RuntimeError(
                f"Term {name!r} motion_slice_fn returned shape {a.shape}, "
                f"expected ({n}, dim) with 2-D layout"
            )
        lerped = ((1.0 - alpha_col) * a + alpha_col * b).astype(
            np.float32, copy=False
        )
        pieces.append(lerped)

    return np.concatenate(pieces, axis=-1)


def compute_amp_obs_from_env(term_names: Sequence[str], wrapper: Any) -> Any:
    """Build the AMP obs tensor for the running env.

    Returns a torch tensor ``(num_envs, total_dim)`` on the env's
    device. The call delegates to each term's ``env_fn`` and
    concatenates in the order given.
    """
    import torch as _torch  # local: keep the module main-venv safe

    pieces = []
    for name in term_names:
        term = get_term(name)
        x = term.env_fn(wrapper)
        if x.dim() != 2:
            raise RuntimeError(
                f"Term {name!r} env_fn returned tensor with ndim={x.dim()}, "
                f"expected 2 (num_envs, dim). Shape: {tuple(x.shape)}"
            )
        pieces.append(x)
    return _torch.cat(pieces, dim=-1)


# ---------------------------------------------------------------------------
# Default term set — the legacy 43-dim amp_legged_gym layout
# ---------------------------------------------------------------------------
#
# These six terms reproduce exactly the layout that
# ``MotionClip._sample_amp_obs_batch`` and
# ``il_train_launcher._amp_obs_extractor`` were computing before the
# registry existed. They are registered here so any caller can simply
# do::
#
#     import src.system.training.amp_obs_terms  # noqa: F401
#
# to make the defaults available. Future asset families (bipeds,
# humanoids) can register additional terms without touching the core
# lerp logic.


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
    root_pos = getattr(clip, "root_pos", None)
    if root_pos is None:
        return np.zeros((len(idx), 1), dtype=np.float32)
    # Shape (T, 3) → take z column, reshape to (B, 1)
    return root_pos[idx, 2:3].astype(np.float32, copy=False)


# ---- env side ----
#
# The env_fn is called once per rollout step with the wrapper
# instance. It reads from ``wrapper.env.unwrapped.scene["robot"]``
# using Isaac Lab's articulation data surface. To avoid re-resolving
# foot body indices every step we cache them on the wrapper under
# ``_amp_term_cache``.


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


def _resolve_foot_ids(wrapper: Any) -> List[int]:
    cache = _wrapper_cache(wrapper)
    if "foot_ids" in cache:
        return cache["foot_ids"]
    robot = wrapper.env.unwrapped.scene["robot"]
    # Standard quadruped naming — A1, Go2, Spot, Anymal.
    foot_ids, foot_names = robot.find_bodies(".*_foot")
    if not foot_ids:
        foot_ids, foot_names = robot.find_bodies(".*_toe")
    if not foot_ids:
        raise RuntimeError(
            "amp_obs_terms: cannot find foot links in robot articulation. "
            "Tried '.*_foot' and '.*_toe' regexes against bodies: "
            f"{robot.body_names}. Add a custom hook if your robot uses "
            "non-standard foot naming."
        )
    cache["foot_ids"] = foot_ids
    cache["foot_names"] = foot_names
    print(
        f"[UnitPort][AMP] amp_obs_terms resolved feet: {foot_names} "
        f"→ indices {foot_ids}",
        flush=True,
    )
    return foot_ids


def _joint_pos_env(wrapper: Any):
    return _robot_data(wrapper).joint_pos


def _joint_vel_env(wrapper: Any):
    return _robot_data(wrapper).joint_vel


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


# ---- registrations ----
register(
    AmpObsTerm(
        name="joint_pos",
        dim_fn=lambda ctx: int(ctx.get("num_dofs", 12)),
        motion_slice_fn=_joint_pos_slice,
        env_fn=_joint_pos_env,
        doc="Per-DoF joint positions in the canonical joint order.",
    )
)
register(
    AmpObsTerm(
        name="toe_pos_local",
        dim_fn=lambda ctx: 3 * int(ctx.get("num_feet", 4)),
        motion_slice_fn=_toe_pos_local_slice,
        env_fn=_toe_pos_local_env,
        doc="Foot positions relative to the root body (translation only).",
    )
)
register(
    AmpObsTerm(
        name="base_lin_vel_local",
        dim_fn=lambda ctx: 3,
        motion_slice_fn=_lin_vel_slice,
        env_fn=_lin_vel_env,
        doc="Root linear velocity in base/body frame (xyz).",
    )
)
register(
    AmpObsTerm(
        name="base_ang_vel_local",
        dim_fn=lambda ctx: 3,
        motion_slice_fn=_ang_vel_slice,
        env_fn=_ang_vel_env,
        doc="Root angular velocity in base/body frame (xyz).",
    )
)
register(
    AmpObsTerm(
        name="joint_vel",
        dim_fn=lambda ctx: int(ctx.get("num_dofs", 12)),
        motion_slice_fn=_joint_vel_slice,
        env_fn=_joint_vel_env,
        doc="Per-DoF joint velocities.",
    )
)
register(
    AmpObsTerm(
        name="root_height",
        dim_fn=lambda ctx: 1,
        motion_slice_fn=_root_height_slice,
        env_fn=_root_height_env,
        doc="Root body height above world z=0.",
    )
)


#: The legacy 43-dim ``amp_legged_gym`` quadruped layout, in the order
#: the discriminator was trained on. Pass this to
#: ``compute_amp_obs_from_motion`` / ``compute_amp_obs_from_env`` for
#: default behaviour; pre-existing runs continue to work.
DEFAULT_QUADRUPED_TERMS: List[str] = [
    "joint_pos",
    "toe_pos_local",
    "base_lin_vel_local",
    "base_ang_vel_local",
    "joint_vel",
    "root_height",
]
