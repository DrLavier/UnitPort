# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AMP Reference-State Initialization (RSI) — Isaac Lab MDP event.

RSI resets each env to a randomly sampled frame of the expert mocap
(root pose + velocity + joint state) instead of a fixed default pose
plus uniform noise. Without RSI, early-training policy trajectories
barely overlap the expert distribution — the discriminator saturates
at ~1.00 acc from iter 0 and gives the policy essentially no gradient.
With RSI, every episode starts mid-gait inside the expert manifold,
so the disc immediately sees informative policy-vs-expert pairs.

Module layout
-------------
- ``register_rsi_pool`` / ``get_rsi_pool`` / ``clear_rsi_pools`` —
  main-venv-safe registry helpers, no torch at module scope.
- ``reset_from_reference_motion`` — Isaac Lab ``EventTermCfg(mode="reset")``
  callback. Imports torch lazily and calls the asset ``write_*_to_sim``
  API, so it is only invocable inside the Isaac Sim venv. Unit tests
  that exercise the module (e.g. ``tests/unit/test_rsi_pool.py``) touch
  the registry only, never this function.

Lifecycle
---------
1. Launcher builds the numpy pool via
   ``data_provider.build_rsi_pool(amp_data.clips, n_frames=N)`` and
   calls :func:`register_rsi_pool` under the pool_id string that the
   compiler baked into the generated ``EventCfg`` (default
   ``"default"``).
2. Isaac Lab's EventManager invokes :func:`reset_from_reference_motion`
   with the env, the env_ids to reset, and the pool_id from the
   EventTerm params. Missing pool → warn once and no-op so test /
   sanity runs don't hard-crash.
3. Joint permutation is resolved from
   ``amp_obs_terms.resolve_canonical_joint_perm`` (shared with the AMP
   obs extractor) and cached per articulation id so the first reset
   pays the resolve cost and subsequent resets are O(1).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


_POOL_SCHEMA: Tuple[Tuple[str, int], ...] = (
    ("root_pos", 3),
    ("root_quat_wxyz", 4),
    ("root_lin_vel_w", 3),
    ("root_ang_vel_w", 3),
    ("joint_pos_canonical", 12),
    ("joint_vel_canonical", 12),
)

_RSI_POOLS: Dict[str, Dict[str, np.ndarray]] = {}
#: Device-cached torch copies of the numpy pool. Key = (pool_id, str(device));
#: value = dict with the same keys as the numpy pool but float32 tensors on
#: the target device. Populated lazily on first reset.
_TORCH_POOL_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_WARNED_MISSING: set = set()
#: Joint-perm cache keyed on ``id(articulation)``. Avoids re-walking
#: joint_names every reset call.
_JOINT_PERM_CACHE: Dict[int, Any] = {}


# ---------------------------------------------------------------------------
# Registry helpers (main-venv safe — no torch)
# ---------------------------------------------------------------------------


def register_rsi_pool(pool_id: str, pool: Dict[str, np.ndarray]) -> None:
    """Register a numpy pool under ``pool_id``.

    Re-registering the same pool_id invalidates any cached torch copies so
    a restart-and-reload cycle picks up fresh frames without a process
    restart.
    """
    if not isinstance(pool_id, str) or not pool_id:
        raise ValueError(f"register_rsi_pool: pool_id must be a non-empty str, got {pool_id!r}")
    if not isinstance(pool, dict):
        raise TypeError(f"register_rsi_pool: pool must be a dict, got {type(pool).__name__}")

    expected = {k for k, _ in _POOL_SCHEMA}
    missing = expected - set(pool.keys())
    extra = set(pool.keys()) - expected
    if missing or extra:
        raise ValueError(
            f"register_rsi_pool: pool keys mismatch. missing={sorted(missing)} "
            f"extra={sorted(extra)}. Expected exactly {sorted(expected)}."
        )

    n_frames: Optional[int] = None
    cleaned: Dict[str, np.ndarray] = {}
    for key, expected_dim in _POOL_SCHEMA:
        arr = np.asarray(pool[key], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(
                f"register_rsi_pool: field {key!r} has shape {arr.shape}, "
                f"expected (N, {expected_dim})."
            )
        if n_frames is None:
            n_frames = arr.shape[0]
        elif arr.shape[0] != n_frames:
            raise ValueError(
                f"register_rsi_pool: field {key!r} has N={arr.shape[0]}, "
                f"expected N={n_frames} (must match across all fields)."
            )
        cleaned[key] = arr

    if not n_frames:
        raise ValueError("register_rsi_pool: pool is empty (N=0)")

    _RSI_POOLS[pool_id] = cleaned
    # Drop any stale torch copies for this pool_id across devices.
    stale_keys = [k for k in _TORCH_POOL_CACHE if k[0] == pool_id]
    for k in stale_keys:
        _TORCH_POOL_CACHE.pop(k, None)
    _WARNED_MISSING.discard(pool_id)


def get_rsi_pool(pool_id: str) -> Optional[Dict[str, np.ndarray]]:
    return _RSI_POOLS.get(pool_id)


def clear_rsi_pools() -> None:
    _RSI_POOLS.clear()
    _TORCH_POOL_CACHE.clear()
    _WARNED_MISSING.clear()
    _JOINT_PERM_CACHE.clear()


# ---------------------------------------------------------------------------
# Isaac Lab event callback
# ---------------------------------------------------------------------------


def _ensure_torch_pool(pool_id: str, device: Any) -> Optional[Dict[str, Any]]:
    """Return a device-resident torch view of the pool, or None if missing."""
    import torch as _torch
    cache_key = (pool_id, str(device))
    cached = _TORCH_POOL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    pool = _RSI_POOLS.get(pool_id)
    if pool is None:
        return None
    converted = {
        key: _torch.as_tensor(arr, dtype=_torch.float32, device=device)
        for key, arr in pool.items()
    }
    _TORCH_POOL_CACHE[cache_key] = converted
    return converted


def _resolve_joint_perm_from_env(env: Any, asset_cfg: Any) -> Any:
    import torch as _torch
    from application.training.amp.obs_terms import resolve_canonical_joint_perm

    articulation = env.scene[asset_cfg.name]
    key = id(articulation)
    cached = _JOINT_PERM_CACHE.get(key)
    if cached is not None:
        return cached
    perm_list = resolve_canonical_joint_perm(articulation)
    device = articulation.data.joint_pos.device
    perm = _torch.as_tensor(perm_list, dtype=_torch.long, device=device)
    _JOINT_PERM_CACHE[key] = perm
    return perm


def _coerce_env_ids(env: Any, env_ids: Any) -> Any:
    import torch as _torch
    device = env.device
    if env_ids is None:
        return _torch.arange(env.num_envs, device=device, dtype=_torch.long)
    if isinstance(env_ids, slice):
        return _torch.arange(env.num_envs, device=device, dtype=_torch.long)[env_ids]
    if isinstance(env_ids, _torch.Tensor):
        return env_ids.to(device=device, dtype=_torch.long).reshape(-1)
    return _torch.as_tensor(env_ids, device=device, dtype=_torch.long).reshape(-1)


def reset_from_reference_motion(
    env: Any,
    env_ids: Any,
    asset_cfg: Any,
    pool_id: str = "default",
    rsi_prob: float = 1.0,
    joint_noise: float = 0.0,
) -> None:
    """Isaac Lab MDP event: reset each env in ``env_ids`` to a random
    frame drawn from the RSI pool ``pool_id``.

    ``rsi_prob < 1.0`` turns this into a stochastic override: per-env
    Bernoulli draws decide which rows get RSI writes and which keep
    whatever the prior reset events (``reset_base`` / ``reset_joints``)
    already set. Skipped rows are NOT written — we subset env_ids BEFORE
    the write rather than writing then undoing, so prior resets persist.

    ``joint_noise`` is a small Gaussian added to joint POSITIONS only
    after the reference write. Applying it to velocities too would
    desync pos/vel energy on a balanced pose and produce fast
    falls; keeping it pos-only matches how legged_gym's AMP code
    treats DR on top of RSI.
    """
    import torch as _torch

    pool_t = _ensure_torch_pool(pool_id, env.device)
    if pool_t is None:
        if pool_id not in _WARNED_MISSING:
            _WARNED_MISSING.add(pool_id)
            print(
                f"[UnitPort][AMP][RSI][WARN] pool {pool_id!r} not registered — "
                f"skipping RSI for this reset. Falling back to default reset "
                f"terms. (This warning is printed once per missing pool_id.)",
                flush=True,
            )
        return

    env_ids_long = _coerce_env_ids(env, env_ids)
    n = int(env_ids_long.numel())
    if n == 0:
        return

    asset = env.scene[asset_cfg.name]

    n_pool = pool_t["root_pos"].shape[0]
    frame_idx = _torch.randint(0, n_pool, (n,), device=env.device)

    root_pos = pool_t["root_pos"][frame_idx]
    # Apply the per-env world-space origin. Motion clips store root_pos
    # in a local, origin-centered frame (all clips start near [0, 0, ~0.3]
    # and drift as the robot walks forward in clip time). Isaac Lab's
    # ``write_root_pose_to_sim`` expects WORLD-frame coordinates, and
    # ``reset_root_state_uniform`` (which fires before this event) already
    # placed each env at its own ``scene.env_origins[i]`` in the scene
    # grid. If we write the raw clip root_pos here, every RSI-reset env
    # teleports on top of the same world-space coordinate (~origin) — so
    # env 0 lands correctly by accident, and envs 1..N collide / tunnel
    # through neighbouring envs' terrain on step 0, producing the exact
    # NaN-velocity + value-function explosion pattern that crashed Go2
    # AMP runs at ~iter 230. Adding env_origins re-grounds each env in
    # its own cell while preserving the clip's pose/phase information.
    try:
        env_origins = env.scene.env_origins[env_ids_long]
    except Exception:
        env_origins = None
    if env_origins is not None:
        root_pos = root_pos + env_origins
    root_quat_wxyz = pool_t["root_quat_wxyz"][frame_idx]
    lin_vel = pool_t["root_lin_vel_w"][frame_idx]
    ang_vel = pool_t["root_ang_vel_w"][frame_idx]
    canonical_joint_pos = pool_t["joint_pos_canonical"][frame_idx]
    canonical_joint_vel = pool_t["joint_vel_canonical"][frame_idx]

    perm = _resolve_joint_perm_from_env(env, asset_cfg)
    # Scatter canonical → asset-native: asset[:, perm[k]] = canonical[:, k].
    # Building a fresh tensor avoids touching whatever the prior reset
    # wrote for envs we later subset away.
    asset_joint_pos = _torch.empty_like(canonical_joint_pos)
    asset_joint_pos[:, perm] = canonical_joint_pos
    asset_joint_vel = _torch.empty_like(canonical_joint_vel)
    asset_joint_vel[:, perm] = canonical_joint_vel

    if joint_noise > 0.0:
        asset_joint_pos = asset_joint_pos + _torch.randn_like(asset_joint_pos) * float(joint_noise)

    pose7 = _torch.cat([root_pos, root_quat_wxyz], dim=-1)
    vel6 = _torch.cat([lin_vel, ang_vel], dim=-1)

    if rsi_prob < 1.0:
        mask = _torch.rand(n, device=env.device) < float(rsi_prob)
        if not bool(mask.any()):
            return
        env_ids_write = env_ids_long[mask]
        pose7 = pose7[mask]
        vel6 = vel6[mask]
        asset_joint_pos = asset_joint_pos[mask]
        asset_joint_vel = asset_joint_vel[mask]
    else:
        env_ids_write = env_ids_long

    # Defensive clamp + nan-sanitize before writing to the sim. The pool
    # builder already filters obvious garbage, but joint_noise can push a
    # borderline pose out of range, and future pool sources may not be
    # pre-filtered. A single NaN root_pos written here propagates to every
    # physics read thereafter and is the hardest failure mode to recover
    # from — cheap to guard, expensive to debug.
    pose7 = _torch.nan_to_num(pose7, nan=0.0, posinf=0.0, neginf=0.0)
    pose7[:, 2].clamp_(min=0.15, max=1.5)  # root z must be above contact radius
    vel6 = _torch.nan_to_num(vel6, nan=0.0, posinf=0.0, neginf=0.0)
    vel6[:, 0:3].clamp_(-10.0, 10.0)  # lin_vel
    vel6[:, 3:6].clamp_(-20.0, 20.0)  # ang_vel
    asset_joint_pos = _torch.nan_to_num(
        asset_joint_pos, nan=0.0, posinf=0.0, neginf=0.0
    )
    asset_joint_vel = _torch.nan_to_num(
        asset_joint_vel, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_(-30.0, 30.0)

    asset.write_root_pose_to_sim(pose7, env_ids=env_ids_write)
    asset.write_root_velocity_to_sim(vel6, env_ids=env_ids_write)
    asset.write_joint_state_to_sim(
        asset_joint_pos, asset_joint_vel, joint_ids=None, env_ids=env_ids_write
    )
