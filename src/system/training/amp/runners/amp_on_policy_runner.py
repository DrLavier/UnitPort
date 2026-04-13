# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Vendored into UnitPort from:
#   custom_mods/archives/AMP_for_hardware/rsl_rl/rsl_rl/runners/amp_on_policy_runner.py
# See src/system/training/amp/algorithms/discriminator.py for the full
# BSD-3 header (Copyright 2021 NVIDIA / ETH Zurich / Nikita Rudin).
#
# Changes from upstream:
#   - Imports the UnitPort-vendored AMPDiscriminator, Normalizer, and
#     AMPPPO (all under src.system.training.amp.*) instead of the
#     AMP_for_hardware rsl_rl fork's copies.
#   - ActorCritic and VecEnv are still pulled from the system rsl_rl
#     (Isaac venv) so we inherit whatever Isaac Lab ships.
#   - The rollout inner loop reads the pre-step and corrected-post-step
#     amp_obs from ``infos["amp"]`` as produced by our
#     AmpRslRlVecEnvWrapper (see §4.amp_backend.vec_env_wrapper), rather
#     than calling ``env.get_amp_observations()`` twice per step. This
#     is the clean-room reconciliation with our wrapper contract.
#   - Heavy rsl_rl imports are deferred to __init__ so this module can
#     be *loaded* from the main venv (tests) even though it cannot be
#     *instantiated* there.

"""AMP-flavored on-policy runner — top-level training loop.

Responsibilities
----------------
1. Build the actor_critic, AMPDiscriminator, Normalizer, and AMPPPO
   from the compiled config.
2. Drive rollout / compute_returns / update for each iteration.
3. Log to TensorBoard + stdout.
4. Save / load checkpoints (actor, discriminator, optimizer, normalizer).

Only usable inside the Isaac Sim venv. Main-venv tests may import this
module (the heavy rsl_rl imports are deferred) but MUST NOT instantiate
``AMPOnPolicyRunner``.
"""
from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Any, Dict, List, Optional

import torch

from src.system.training.amp.algorithms.amp_ppo import AMPPPO
from src.system.training.amp.algorithms.actor_critic import ActorCritic
from src.system.training.amp.algorithms.discriminator import AMPDiscriminator
from src.system.training.amp.utils.normalizer import Normalizer


def _normalize_obs(value):
    """Coerce whatever the env returned into a plain ``torch.Tensor``.

    The vendored AMP runner expects ``actor_critic.act(obs)`` to receive
    a vanilla ``torch.Tensor`` of shape ``(num_envs, num_obs)``. Modern
    Isaac Lab + RslRlVecEnvWrapper can return any of:

      - ``torch.Tensor``                  (the happy path)
      - ``(tensor, extras_dict)``         (rsl_rl 2.x ``get_observations``)
      - ``dict[str, torch.Tensor]``       (un-concatenated ObsGroup)
      - ``TensorDict`` / similar wrapper  (some Isaac Lab releases)
      - any of the above wrapped in another tuple

    All of these break ``nn.Linear``'s ``__torch_function__`` dispatch
    with the cryptic ``Multiple dispatch failed for 'torch.nn.linear'``
    when fed straight in. This helper unwraps recursively, sorts the
    dict's keys for determinism, concatenates along the feature dim,
    and finally casts to a plain ``torch.Tensor`` (no subclass) so
    every downstream op gets the standard dispatch.

    The result is also ``.clone().contiguous()``-ed to detach from any
    underlying obs-manager buffer that might get reused / cleared on the
    next env operation. We've seen Isaac Lab's obs manager hand back
    tensor *views* into buffers that get rezeroed when the env steps,
    which manifested as ``obs`` having shape ``(N, 48)`` at the start
    of ``learn()`` and ``(N, 0)`` at the first ``act()`` call.
    """
    if value is None:
        return None
    # tuple → first element (rsl_rl 2.x get_observations() returns
    # ``(obs_tensor, extras_dict)``)
    if isinstance(value, tuple):
        value = value[0]
    # dict / dict-like → concat sorted keys along feature dim
    if isinstance(value, dict):
        keys = sorted(value.keys())
        value = torch.cat([value[k] for k in keys], dim=-1)
    # Force a plain torch.Tensor — drops any subclass that might have
    # an obstructive __torch_function__.
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    elif type(value) is not torch.Tensor:
        # ``isinstance`` is True for subclasses; ``type`` checks the
        # exact class. Subclasses (e.g. TensorDict / wrapped) get
        # rebuilt as plain Tensors via ``.as_subclass``.
        value = value.as_subclass(torch.Tensor)
    # Defensive copy — see docstring above for the buffer-reuse footgun.
    return value.clone().contiguous()


def _resolve_env_dim(env, attrs, group_key=None):
    """Resolve a dimension (obs / actions) from a VecEnv across rsl_rl versions.

    Tries each name in *attrs* via ``getattr`` first (rsl_rl 1.x style),
    then falls back to:
      1. ``env.observation_space`` / ``env.action_space`` shape lookup
      2. ``env.unwrapped.observation_manager.group_obs_dim[group_key]``
         (Isaac Lab 2.x style — only for obs, requires group_key)

    Returns ``None`` when nothing matches; callers handle the fallback.
    """
    for name in attrs:
        val = getattr(env, name, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    # Fall through to gym Spaces
    if "obs" in attrs[0] or "observation" in attrs[0]:
        space = getattr(env, "observation_space", None)
        if space is not None:
            try:
                shape = getattr(space, "shape", None)
                if shape is not None and len(shape) > 0:
                    return int(shape[-1])
                # Dict space — pick the named group
                if hasattr(space, "spaces") and group_key:
                    sub = space.spaces.get(group_key)
                    if sub is not None and hasattr(sub, "shape"):
                        return int(sub.shape[-1])
            except Exception:
                pass
    else:
        space = getattr(env, "action_space", None)
        if space is not None:
            shape = getattr(space, "shape", None)
            if shape is not None and len(shape) > 0:
                try:
                    return int(shape[-1])
                except Exception:
                    pass

    # Last resort: query the underlying ManagerBasedRLEnv directly
    if group_key is not None:
        try:
            inner = getattr(env, "unwrapped", env)
            obs_mgr = getattr(inner, "observation_manager", None)
            if obs_mgr is not None:
                group_dims = getattr(obs_mgr, "group_obs_dim", {}) or {}
                dim = group_dims.get(group_key)
                if dim is not None:
                    if hasattr(dim, "__iter__"):
                        return int(list(dim)[-1])
                    return int(dim)
        except Exception:
            pass

    return None


def _resolve_num_obs(env) -> int:
    """Find policy-side obs dimension. Hard-error if all heuristics fail."""
    dim = _resolve_env_dim(
        env,
        attrs=["num_obs", "obs_dim", "observation_dim"],
        group_key="policy",
    )
    if dim is None or dim <= 0:
        raise AttributeError(
            "AMPOnPolicyRunner: cannot determine the policy obs dim from "
            "the env. Tried env.num_obs / env.obs_dim / env.observation_dim, "
            "env.observation_space.shape, and "
            "env.unwrapped.observation_manager.group_obs_dim['policy']."
        )
    return dim


def _resolve_num_actions(env) -> int:
    """Find action dim. Hard-error if all heuristics fail."""
    dim = _resolve_env_dim(env, attrs=["num_actions", "action_dim"])
    if dim is None or dim <= 0:
        raise AttributeError(
            "AMPOnPolicyRunner: cannot determine the action dim from the env. "
            "Tried env.num_actions / env.action_dim and env.action_space.shape."
        )
    return dim


def _resolve_num_envs(env) -> int:
    """Find num_envs. Hard-error if missing — fundamental for batching."""
    n = getattr(env, "num_envs", None)
    if n is None:
        try:
            n = int(env.unwrapped.scene.num_envs)
        except Exception:
            pass
    if n is None or n <= 0:
        raise AttributeError(
            "AMPOnPolicyRunner: cannot determine env.num_envs."
        )
    return int(n)


def _resolve_actor_critic_classes():
    """Return the vendored UnitPort actor-critic classes.

    Originally this helper iterated rsl_rl import paths trying to find
    the moving ``ActorCritic`` class across versions. Modern rsl_rl 2.x
    dropped that god-class entirely (see the diagnostic dump in any
    pre-vendor commit's error message — only ``CNN``, ``MLP``, ``RNN``,
    ``GaussianDistribution``, etc. survive). Rather than chase rsl_rl's
    moving target, UnitPort now ships the AMP_for_hardware-style
    ``ActorCritic`` in-tree at ``amp/algorithms/actor_critic.py``. The
    helper kept its name + signature so the runner constructor below
    didn't need to change.

    ``ActorCriticRecurrent`` is intentionally ``None`` — the recurrent
    variant isn't vendored. AMP training in phase_3 uses the feed-forward
    MLP path only; setting ``policy_class_name = "ActorCriticRecurrent"``
    in the runner cfg will raise a clear error from
    ``AMPOnPolicyRunner.__init__``'s ``_local_classes`` lookup.
    """
    return ActorCritic, None


class AMPOnPolicyRunner:
    """Orchestrates an AMP-PPO training run on a VecEnv-compatible env.

    Parameters
    ----------
    env:
        ``AmpRslRlVecEnvWrapper`` instance (or any env that exposes
        ``num_amp_obs``, ``get_amp_observations()``, and returns
        ``infos["amp"] = {"obs", "obs_next"}`` from ``step()``).
    train_cfg:
        Runner configuration dict. Expected shape mirrors the AMP_for_
        hardware format — ``runner`` / ``algorithm`` / ``policy`` sub-
        dicts. The launcher builds this from AMPConfig + AlgorithmConfig.
    amp_data:
        ``MotionAmpData`` from :mod:`src.system.training.amp.data_provider`.
    log_dir:
        Where to write TensorBoard events and checkpoints. ``None``
        disables logging (useful for sanity tests).
    device:
        Torch device.
    """

    def __init__(
        self,
        env,
        train_cfg: Dict[str, Any],
        amp_data,
        log_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        # Deferred rsl_rl imports — only succeed inside Isaac venv.
        # ``ActorCritic`` lives at one of two paths depending on the
        # rsl_rl version:
        #   - rsl_rl 1.x  (AMP_for_hardware fork era):
        #         from rsl_rl.modules import ActorCritic
        #   - rsl_rl 2.x  (current Isaac Lab default):
        #         from rsl_rl.modules.actor_critic import ActorCritic
        # The 2.x version stopped re-exporting at the package root, so
        # the legacy path raises ImportError. Try both, in newest-first
        # order so production setups don't pay for the fallback.
        ActorCritic, ActorCriticRecurrent = _resolve_actor_critic_classes()
        from torch.utils.tensorboard import SummaryWriter as _SW  # noqa: F401

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self._top_train_cfg = train_cfg  # kept for staged training dispatch
        self.device = device
        self.env = env

        # ── Defensive env attribute reads ──
        # The vendored AMP_for_hardware runner was written against the
        # rsl_rl 1.x VecEnv contract which exposed direct attributes:
        #   num_obs, num_actions, num_privileged_obs, num_envs,
        #   include_history_steps, dof_pos_limits, episode_length_buf,
        #   max_episode_length, get_observations,
        #   get_privileged_observations, reset, step
        #
        # Modern Isaac Lab's ``RslRlVecEnvWrapper`` (and our subclass
        # ``AmpRslRlVecEnvWrapper``) exposes ONLY: ``num_envs``,
        # ``device``, ``observation_space``, ``action_space``. The
        # legacy direct dim attributes are gone — they're now derived
        # from the obs manager's group dims.
        #
        # _resolve_num_obs / _resolve_num_actions / _resolve_num_envs
        # try every known path to extract these dims, raising a clear
        # error when all fail. _resolve_env_dim with group_key="critic"
        # finds the privileged obs dim if a "critic" group exists.
        num_obs_local = _resolve_num_obs(self.env)
        num_actions_local = _resolve_num_actions(self.env)
        num_envs_local = _resolve_num_envs(self.env)
        # Cache them so the rest of the runner doesn't have to re-resolve.
        self._num_obs = num_obs_local
        self._num_actions = num_actions_local
        self._num_envs = num_envs_local

        # Privileged (critic) obs — None means symmetric AC (critic uses
        # the same obs vector as the actor). RolloutStorage handles None
        # by skipping the privileged_observations buffer entirely.
        num_privileged_obs = _resolve_env_dim(
            self.env,
            attrs=["num_privileged_obs", "num_critic_obs"],
            group_key="critic",
        )
        if num_privileged_obs is not None:
            num_critic_obs = num_privileged_obs
        else:
            num_critic_obs = num_obs_local

        # ``policy_class_name`` is a string from the runner cfg dict
        # (e.g. "ActorCritic" / "ActorCriticRecurrent"). Resolve it
        # against the locally-imported class objects above instead of
        # ``eval()``-ing into module globals — eval would fail because
        # the names live inside this method's scope, not module scope.
        _local_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
        }
        policy_class_name = self.cfg["policy_class_name"]
        actor_critic_class = _local_classes.get(policy_class_name)
        if actor_critic_class is None:
            raise ValueError(
                f"AMPOnPolicyRunner: unknown policy_class_name "
                f"{policy_class_name!r}. Known: {list(_local_classes)}"
            )
        if getattr(self.env, "include_history_steps", None) is not None:
            num_actor_obs = num_obs_local * int(self.env.include_history_steps)
        else:
            num_actor_obs = num_obs_local

        actor_critic = actor_critic_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions_local,
            **self.policy_cfg,
        ).to(self.device)

        # ── Optional preload to burn expert transitions into memory ──
        # The default ``num_preload_transitions = 2_000_000`` from
        # AMP_for_hardware was sized for their multi-clip dog dataset
        # (~50k frames). For our typical 6-clip pool of ~240 frames
        # this is gross overkill — and even with the vectorised
        # MotionClip._sample_amp_obs_batch path the preload still does
        # 2M numpy lookups. Print a sentinel so the user can see what's
        # happening (otherwise looks like Isaac Sim is hanging) and
        # report elapsed time.
        preload_n = int(self.cfg.get("amp_num_preload_transitions", 0))
        if preload_n > 0 and hasattr(amp_data, "preload_transitions"):
            print(
                f"[UnitPort][AMP] Preloading {preload_n:,} expert AMP "
                f"transitions from {amp_data.num_motions} clip(s)…",
                flush=True,
            )
            _t0 = time.time()
            amp_data.preload_transitions(preload_n)
            print(
                f"[UnitPort][AMP] Preload done in {time.time() - _t0:.1f}s "
                f"(observation_dim={amp_data.observation_dim}).",
                flush=True,
            )

        amp_normalizer = Normalizer(amp_data.observation_dim)

        discriminator = AMPDiscriminator(
            input_dim=amp_data.observation_dim * 2,
            amp_reward_coef=self.cfg["amp_reward_coef"],
            hidden_layer_sizes=self.cfg["amp_discr_hidden_dims"],
            device=self.device,
            task_reward_lerp=self.cfg.get("amp_task_reward_lerp", 0.0),
        ).to(self.device)

        min_std = None
        if "min_normalized_std" in self.cfg and hasattr(self.env, "dof_pos_limits"):
            min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device)
                * torch.abs(
                    self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]
                )
            )

        self.alg = AMPPPO(
            actor_critic=actor_critic,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
        )

        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg["save_interval"])

        # ── Lerp schedule (RSI-AMP staged training) ──
        # Optional linear anneal for task_reward_lerp. When present the
        # runner updates discriminator.task_reward_lerp every iteration.
        self._lerp_schedule = None
        raw_sched = self.cfg.get("amp_lerp_schedule", "")
        if raw_sched:
            import json as _json
            try:
                s = _json.loads(raw_sched) if isinstance(raw_sched, str) else raw_sched
                if isinstance(s, dict) and "start" in s and "end" in s:
                    self._lerp_schedule = {
                        "start": float(s["start"]),
                        "end": float(s["end"]),
                        "over_iters": int(s.get("over_iters", 1)),
                    }
            except Exception:
                pass

        # ``num_privileged_obs`` may be None — RolloutStorage's
        # ``privileged_observations`` buffer is then skipped (see the
        # ``if privileged_obs_shape[0] is not None:`` branch in the
        # vendored rollout_storage.py).
        self.alg.init_storage(
            num_envs_local,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_privileged_obs],   # may be None — handled downstream
            [num_actions_local],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # ── Staged training dispatch ──
        # When the top-level train_cfg carries a stage_schedule, the runner
        # iterates through stages in order, applying each stage's overrides
        # on top of the baseline config. Otherwise, single-stage legacy path.
        _top_cfg = getattr(self, "_top_train_cfg", None)
        if _top_cfg and isinstance(_top_cfg.get("stage_schedule"), dict):
            sched = _top_cfg["stage_schedule"]
            stages = sched.get("stages", [])
            if isinstance(stages, list) and stages:
                self._learn_staged(stages, sched, init_at_random_ep_len)
                return

        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf"):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # ``get_observations`` API drift across rsl_rl versions:
        #   rsl_rl 1.x  → returns a single tensor
        #   rsl_rl 2.x  → returns a tuple ``(obs_tensor, extras_dict)``
        # Handle both. Modern Isaac Lab's RslRlVecEnvWrapper is on the
        # 2.x contract, so this normally takes the second branch.
        # ── Verbose obs diagnostic dump (first-call only) ──
        # Three independent reads to triangulate the source of any
        # shape/type mismatch:
        #   (a) env.get_observations() — what the wrapper exposes
        #   (b) obs_manager.compute() — raw output of the manager
        #   (c) obs_manager.group_obs_dim — declared shape per group
        diag_lines: List[str] = []

        # (a) wrapper output
        try:
            raw_obs = self.env.get_observations()
            diag_lines.append(
                f"[UnitPort][AMP][diag] (a) env.get_observations() type="
                f"{type(raw_obs).__name__}"
            )
            if isinstance(raw_obs, tuple):
                diag_lines.append(
                    f"[UnitPort][AMP][diag]     tuple len={len(raw_obs)} "
                    f"first_type={type(raw_obs[0]).__name__}"
                )
                if hasattr(raw_obs[0], "shape"):
                    diag_lines.append(
                        f"[UnitPort][AMP][diag]     tuple[0].shape={tuple(raw_obs[0].shape)}"
                    )
                if isinstance(raw_obs[0], dict):
                    for k, v in raw_obs[0].items():
                        s = tuple(v.shape) if hasattr(v, "shape") else "?"
                        diag_lines.append(f"[UnitPort][AMP][diag]     tuple[0][{k!r}].shape={s}")
            elif isinstance(raw_obs, dict):
                for k, v in raw_obs.items():
                    s = tuple(v.shape) if hasattr(v, "shape") else "?"
                    diag_lines.append(f"[UnitPort][AMP][diag]     dict[{k!r}].shape={s}")
            elif hasattr(raw_obs, "shape"):
                diag_lines.append(
                    f"[UnitPort][AMP][diag]     tensor.shape={tuple(raw_obs.shape)}"
                )
        except Exception as exc:
            raw_obs = None
            diag_lines.append(f"[UnitPort][AMP][diag] (a) get_observations() FAILED: {exc}")

        # (b) raw obs manager output — bypass the wrapper entirely
        mgr_compute_result = None
        try:
            inner = getattr(self.env, "unwrapped", self.env)
            obs_mgr = getattr(inner, "observation_manager", None)
            if obs_mgr is not None:
                mgr_compute_result = obs_mgr.compute()
                diag_lines.append(
                    f"[UnitPort][AMP][diag] (b) obs_manager.compute() type="
                    f"{type(mgr_compute_result).__name__}"
                )
                if isinstance(mgr_compute_result, dict):
                    for gk, gv in mgr_compute_result.items():
                        if hasattr(gv, "shape"):
                            diag_lines.append(
                                f"[UnitPort][AMP][diag]     [{gk!r}] tensor shape={tuple(gv.shape)}"
                            )
                        elif isinstance(gv, dict):
                            for tk, tv in gv.items():
                                s = tuple(tv.shape) if hasattr(tv, "shape") else "?"
                                diag_lines.append(
                                    f"[UnitPort][AMP][diag]     [{gk!r}][{tk!r}] shape={s}"
                                )
                        else:
                            diag_lines.append(
                                f"[UnitPort][AMP][diag]     [{gk!r}] type={type(gv).__name__}"
                            )
        except Exception as exc:
            diag_lines.append(f"[UnitPort][AMP][diag] (b) obs_manager.compute() FAILED: {exc}")

        # (c) declared dimensions
        try:
            inner = getattr(self.env, "unwrapped", self.env)
            obs_mgr = getattr(inner, "observation_manager", None)
            if obs_mgr is not None:
                group_dims = getattr(obs_mgr, "group_obs_dim", None)
                diag_lines.append(
                    f"[UnitPort][AMP][diag] (c) obs_manager.group_obs_dim={group_dims}"
                )
                # Also list active obs term names per group if exposed
                active_terms = getattr(obs_mgr, "active_terms", None)
                if active_terms is not None:
                    diag_lines.append(
                        f"[UnitPort][AMP][diag]     active_terms={active_terms}"
                    )
        except Exception as exc:
            diag_lines.append(f"[UnitPort][AMP][diag] (c) group_obs_dim FAILED: {exc}")

        # Also dump observation_space for completeness
        try:
            obs_space = getattr(self.env, "observation_space", None)
            diag_lines.append(
                f"[UnitPort][AMP][diag] (d) env.observation_space={obs_space}"
            )
        except Exception:
            pass

        for line in diag_lines:
            print(line, flush=True)

        # ── Wrapper bypass ──
        # The Isaac Lab RslRlVecEnvWrapper / AmpRslRlVecEnvWrapper has been
        # observed to return a TensorDict of shape ``(num_envs,)`` from both
        # ``get_observations()`` AND from ``step()`` in this Isaac Lab
        # release — see triangulation dump above. The underlying
        # observation_manager.compute() returns the *correct* dict
        # ``{"policy": tensor(num_envs, num_obs), ...}``. Rather than
        # repeatedly try to massage the wrapper's broken output, we bypass
        # it: read obs directly from the obs manager every time we need
        # them. This closure is used at init AND after every env.step().
        inner_env = getattr(self.env, "unwrapped", self.env)
        obs_mgr_ref = getattr(inner_env, "observation_manager", None)
        has_critic_group = (
            obs_mgr_ref is not None
            and "critic" in (getattr(obs_mgr_ref, "group_obs_dim", {}) or {})
        )

        def _read_obs_direct():
            """Bypass the wrapper; read obs straight from obs_manager.

            Returns ``(policy_obs, critic_obs_or_None)``. Both are plain
            ``torch.Tensor`` instances of shape ``(num_envs, dim)``.
            """
            result = obs_mgr_ref.compute()
            if not isinstance(result, dict) or "policy" not in result:
                raise RuntimeError(
                    f"[UnitPort][AMP] obs_manager.compute() returned "
                    f"unexpected shape: type={type(result).__name__}"
                )
            pol = _normalize_obs(result["policy"])
            crit = _normalize_obs(result["critic"]) if has_critic_group else None
            return pol, crit

        # Prefer the wrapper's output if it actually returned a sane tensor;
        # otherwise read straight from the obs manager. In every Isaac Lab
        # release we've tested, the wrapper path is broken, so the bypass
        # is the de-facto default.
        obs = _normalize_obs(raw_obs)
        if obs is None or len(obs.shape) < 2 or obs.shape[-1] == 0:
            print(
                "[UnitPort][AMP][diag] Wrapper produced empty obs — "
                "bypassing wrapper, reading from obs_manager directly.",
                flush=True,
            )
            if obs_mgr_ref is not None:
                obs, _crit_init = _read_obs_direct()

        # Cross-check against the dim the runner used to build ActorCritic.
        # If they don't match, the actor MLP will fail at the first matmul
        # with the cryptic ``mat1 and mat2 shapes cannot be multiplied``
        # error — surface the mismatch here with a clear message that
        # also embeds the diagnostic dump so it can't be missed.
        if obs is None or len(obs.shape) < 2 or obs.shape[-1] != self._num_obs:
            actual = tuple(obs.shape) if obs is not None else None
            raise RuntimeError(
                f"\n[UnitPort][AMP] Observation shape mismatch!\n"
                f"  Resolved at construction time: num_obs={self._num_obs}\n"
                f"  Actual at first get_observations(): shape={actual}\n"
                f"\n"
                + "\n".join(diag_lines)
                + "\n\n"
                f"  Most likely causes:\n"
                f"  1. The compiled UnitPortEnvCfg's PolicyCfg has 0 obs terms\n"
                f"     emitted (obs_terms JSON in il_observation node was empty\n"
                f"     or all terms got filtered).\n"
                f"  2. concatenate_terms is False so obs comes back as a dict\n"
                f"     and _normalize_obs produced a wrong concatenation.\n"
                f"  3. The wrapper's observation_space lies about the shape.\n"
            )

        # ``get_privileged_observations`` was rsl_rl 1.x only — modern
        # versions removed it (privileged data is fetched via the obs
        # manager's "critic" group instead). Prefer the obs manager
        # bypass when a "critic" group is declared; fall back to the
        # legacy method if present; otherwise reuse policy obs.
        privileged_obs = None
        if has_critic_group and obs_mgr_ref is not None:
            try:
                _, privileged_obs = _read_obs_direct()
            except Exception:
                privileged_obs = None
        if privileged_obs is None and hasattr(self.env, "get_privileged_observations"):
            try:
                privileged_obs = _normalize_obs(self.env.get_privileged_observations())
            except Exception:
                privileged_obs = None

        amp_obs = _normalize_obs(self.env.get_amp_observations())
        critic_obs = privileged_obs if privileged_obs is not None else obs

        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)

        print(
            f"[UnitPort][AMP] First obs:     type={type(obs).__name__} "
            f"shape={tuple(obs.shape)} device={obs.device} dtype={obs.dtype}",
            flush=True,
        )
        print(
            f"[UnitPort][AMP] First amp_obs: type={type(amp_obs).__name__} "
            f"shape={tuple(amp_obs.shape)} device={amp_obs.device} dtype={amp_obs.dtype}",
            flush=True,
        )
        print(
            f"[UnitPort][AMP] Resolved dims: num_obs={self._num_obs} "
            f"num_actions={self._num_actions} num_envs={self._num_envs}",
            flush=True,
        )

        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos: list = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        # B4 (AMP fix plan): separate deques for style-only and task-only
        # episodic means. These drive Train/mean_style_reward and
        # Train/mean_task_reward so users can see if style is dominating
        # or if task signal is actually flowing through the mix.
        style_rewbuffer: deque = deque(maxlen=100)
        task_rewbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        cur_style_sum = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        cur_task_sum = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )

        # One-time pre-rollout shape sanity check — last line of defence
        # before we feed the actor. If the shape is wrong here, we hard-error
        # with an unmissable diagnostic instead of letting torch's cryptic
        # ``mat1 and mat2 shapes cannot be multiplied`` bubble up.
        print(
            f"[UnitPort][AMP] Pre-rollout obs check: "
            f"obs.shape={tuple(obs.shape)} dtype={obs.dtype} device={obs.device}",
            flush=True,
        )
        if obs.shape[-1] != self._num_obs:
            raise RuntimeError(
                f"\n[UnitPort][AMP] Obs shape changed between init check and rollout!\n"
                f"  num_obs (used to build ActorCritic): {self._num_obs}\n"
                f"  obs.shape at first rollout step:    {tuple(obs.shape)}\n"
                f"\n"
                f"  This usually means a torch tensor view that was clone()d at\n"
                f"  init time got rebuilt as a non-clone reference somewhere\n"
                f"  between the init check and the rollout loop. Check\n"
                f"  _normalize_obs() returns and any obs.to(device) calls.\n"
            )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ── Lerp anneal ──
            if self._lerp_schedule is not None:
                s = self._lerp_schedule
                alpha = min(1.0, float(it) / max(1, s["over_iters"]))
                new_lerp = s["start"] + (s["end"] - s["start"]) * alpha
                self.alg.discriminator.task_reward_lerp = new_lerp

            # ── Rollout ──
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # ── Per-step shape watchdog ──
                    # The matmul error from torch.nn.linear is unreadable
                    # ("mat1 and mat2 shapes cannot be multiplied"). Wrap
                    # the actor call so any shape mismatch comes back with
                    # the actual obs/critic/amp_obs sizes inline. Also
                    # prints the first-step shapes unconditionally so the
                    # user (and us) can see them in the log without
                    # scrolling.
                    if it == self.current_learning_iteration and i == 0:
                        print(
                            f"[UnitPort][AMP] @first act(): "
                            f"obs.shape={tuple(obs.shape)} "
                            f"obs.dtype={obs.dtype} "
                            f"obs.device={obs.device} "
                            f"obs.is_contiguous={obs.is_contiguous()} "
                            f"obs.data_ptr={obs.data_ptr():#x}",
                            flush=True,
                        )
                        print(
                            f"[UnitPort][AMP] @first act(): "
                            f"critic.shape={tuple(critic_obs.shape)} "
                            f"amp.shape={tuple(amp_obs.shape)}",
                            flush=True,
                        )
                        # Also peek at the actor's first Linear layer to
                        # confirm the resolved num_obs matches what was
                        # baked into the network.
                        try:
                            first_linear = next(
                                m for m in self.alg.actor_critic.actor
                                if hasattr(m, "in_features")
                            )
                            print(
                                f"[UnitPort][AMP] @first act(): "
                                f"actor.first_linear.in_features="
                                f"{first_linear.in_features} "
                                f"out_features={first_linear.out_features}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(f"[UnitPort][AMP] (could not introspect actor: {exc})", flush=True)

                    try:
                        actions = self.alg.act(obs, critic_obs, amp_obs)
                    except RuntimeError as exc:
                        # Re-raise with shape context inline so the failure
                        # message in the log can't be missed even if the
                        # diagnostic prints above were trimmed somewhere.
                        raise RuntimeError(
                            f"\n[UnitPort][AMP] alg.act() crashed at "
                            f"iter={it} step={i}\n"
                            f"  obs.shape       = {tuple(obs.shape)}\n"
                            f"  critic.shape    = {tuple(critic_obs.shape)}\n"
                            f"  amp.shape       = {tuple(amp_obs.shape)}\n"
                            f"  resolved num_obs= {self._num_obs}\n"
                            f"  resolved n_acts = {self._num_actions}\n"
                            f"  underlying err  : {exc}\n"
                        ) from exc

                    # Our AmpRslRlVecEnvWrapper returns a 4-tuple with
                    # ``infos["amp"] = {"obs": pre, "obs_next": corrected}``.
                    # The first element ``_step_obs`` is intentionally
                    # discarded — we re-read obs straight from the
                    # observation_manager via _read_obs_direct() because
                    # the wrapper's obs return is unreliable in this
                    # Isaac Lab version (see init bypass above).
                    _step_obs, rewards, dones, infos = self.env.step(actions)
                    del _step_obs  # never trust this in this Isaac Lab release

                    amp_info = infos.get("amp", {}) if isinstance(infos, dict) else {}
                    pre_amp_obs = amp_info.get("obs", amp_obs)
                    next_amp_obs = amp_info.get(
                        "obs_next", self.env.get_amp_observations()
                    )

                    # Re-read policy + critic obs from the obs manager
                    # directly. ``_read_obs_direct()`` is closure-bound to
                    # the unwrapped env's observation_manager and always
                    # returns plain (num_envs, dim) tensors.
                    if obs_mgr_ref is not None:
                        obs, critic_from_mgr = _read_obs_direct()
                    else:
                        obs = _normalize_obs(_step_obs) if False else obs
                        critic_from_mgr = None

                    pre_amp_obs = _normalize_obs(pre_amp_obs)
                    next_amp_obs = _normalize_obs(next_amp_obs)
                    if critic_from_mgr is not None:
                        critic_obs = critic_from_mgr
                    else:
                        critic_obs = obs

                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    # B3 (AMP fix plan): reward mixing happens HERE in
                    # the runner, not inside the discriminator. The env
                    # stays AMP-agnostic; `rewards` from env.step() is
                    # the pure task reward. We compute the style reward
                    # from the discriminator, then blend with the spec
                    # formula `r = (1-l)*style + l*task`.
                    task_r = rewards
                    style_r, _d = self.alg.discriminator.predict_amp_reward(
                        pre_amp_obs,
                        next_amp_obs,
                        normalizer=self.alg.amp_normalizer,
                    )
                    # Shape alignment: discriminator returns (B,) but env
                    # rewards may arrive as (B,) or (B, 1). Collapse any
                    # trailing singleton to match.
                    if style_r.dim() > task_r.dim():
                        style_r = style_r.squeeze(-1)
                    elif task_r.dim() > style_r.dim():
                        task_r = task_r.squeeze(-1)
                    lerp = float(self.alg.discriminator.task_reward_lerp)
                    mixed_r = (1.0 - lerp) * style_r + lerp * task_r

                    amp_obs = next_amp_obs.clone()
                    self.alg.process_env_step(mixed_r, dones, infos, next_amp_obs)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += mixed_r
                        cur_style_sum += style_r
                        cur_task_sum += task_r
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        style_rewbuffer.extend(
                            cur_style_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        task_rewbuffer.extend(
                            cur_task_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_style_sum[new_ids] = 0
                        cur_task_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # ── Learning step ──
                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self._log(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, locs: Dict[str, Any], width: int = 80, pad: int = 35) -> None:
        self.tot_timesteps += self.num_steps_per_env * self._num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        if self.writer is not None:
            # B4 (AMP fix plan): scalar names aligned with spec §5.
            # Old names kept commented for traceability:
            #   Loss/AMP          → Loss/amp_loss
            #   Loss/AMP_grad     → Loss/grad_pen_loss
            #   AMP/policy_pred   → Loss/policy_pred
            #   AMP/expert_pred   → Loss/expert_pred
            # New: Loss/accuracy_policy, Loss/accuracy_expert,
            #      Train/mean_style_reward, Train/mean_task_reward.
            self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
            self.writer.add_scalar("Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"])
            self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
            self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
            self.writer.add_scalar("Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"])
            self.writer.add_scalar("Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"])
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
            self.writer.add_scalar(
                "AMP/task_reward_lerp",
                float(self.alg.discriminator.task_reward_lerp),
                locs["it"],
            )
            if len(locs["rewbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length",
                    statistics.mean(locs["lenbuffer"]),
                    locs["it"],
                )
            if len(locs["style_rewbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_style_reward",
                    statistics.mean(locs["style_rewbuffer"]),
                    locs["it"],
                )
            if len(locs["task_rewbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_task_reward",
                    statistics.mean(locs["task_rewbuffer"]),
                    locs["it"],
                )

        # Pull running mean reward / episode length out of the deques so the
        # parent process's metric parser has them. Empty deque → NaN sentinel
        # which the parser treats as "no data this iter" (carry-forward).
        rewbuf = locs.get("rewbuffer")
        style_buf = locs.get("style_rewbuffer")
        task_buf = locs.get("task_rewbuffer")
        lenbuf = locs.get("lenbuffer")
        mean_reward = statistics.mean(rewbuf) if rewbuf and len(rewbuf) > 0 else float("nan")
        mean_style = statistics.mean(style_buf) if style_buf and len(style_buf) > 0 else float("nan")
        mean_task = statistics.mean(task_buf) if task_buf and len(task_buf) > 0 else float("nan")
        mean_ep_len = statistics.mean(lenbuf) if lenbuf and len(lenbuf) > 0 else float("nan")
        tot_iter_local = locs.get("tot_iter", 0)

        # Single-line, machine-parseable. The parent UI's
        # ``isaac_lab_backend._AMP_LINE_RE`` matches this exact format —
        # if you change the field names or order, update that regex AND
        # ``parse_amp_line()`` AND ``TrainingPanel.on_metrics()`` in the
        # same commit (see knowledge_base/APP_PPO_TOKNOW.yaml §3).
        print(
            f"[AMP] it={locs['it']:>4d}/{tot_iter_local}  "
            f"rew={mean_reward:.3f}  "
            f"style={mean_style:.3f}  "
            f"task={mean_task:.3f}  "
            f"len={mean_ep_len:.1f}  "
            f"vf={locs['mean_value_loss']:.3f}  "
            f"surr={locs['mean_surrogate_loss']:.3f}  "
            f"amp={locs['mean_amp_loss']:.3f}  "
            f"gp={locs['mean_grad_pen_loss']:.3f}  "
            f"pol={locs['mean_policy_pred']:+.2f}  "
            f"exp={locs['mean_expert_pred']:+.2f}  "
            f"acc_p={locs['mean_accuracy_policy']:.2f}  "
            f"acc_e={locs['mean_accuracy_expert']:.2f}  "
            f"t={iteration_time:.2f}s",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Staged training (STAGE_NODE_DESIGN.yaml §2.5)
    # ------------------------------------------------------------------

    def _learn_staged(
        self,
        stages: List[Dict[str, Any]],
        schedule: Dict[str, Any],
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run multi-stage training loop.

        Each stage applies parameter overrides, trains for its configured
        number of iterations (or until advance criteria are met), saves a
        checkpoint, then moves to the next stage.
        """
        import json as _json

        ckpt_strategy = str(schedule.get("checkpoint_strategy", "both"))
        global_iter = self.current_learning_iteration
        total_all_stages = sum(
            int(s.get("iterations", 0)) for s in stages if isinstance(s, dict)
        )

        print(
            f"\n{'=' * 70}\n"
            f"[UnitPort][STAGE] Staged training: {len(stages)} stages, "
            f"{total_all_stages} total iters\n"
            f"{'=' * 70}",
            flush=True,
        )

        # ── Snapshot baseline config for override/restore ──
        self._baseline_lerp = float(self.alg.discriminator.task_reward_lerp)
        self._baseline_amp_reward_coef = float(self.alg.discriminator.amp_reward_coef)
        self._baseline_lr = float(self.alg.learning_rate)
        self._baseline_entropy_coef = float(self.alg.entropy_coef)
        self._baseline_clip_param = float(self.alg.clip_param)
        self._baseline_disc_lr = float(self.alg_cfg.get("amp_disc_lr", 1e-4))
        self._baseline_disc_grad_penalty = float(
            self.alg_cfg.get("amp_disc_grad_penalty", 10.0)
        )
        self._baseline_disc_label_smoothing = float(
            self.alg_cfg.get("amp_disc_label_smoothing", 0.9)
        )
        self._stage_best_reward: Dict[int, float] = {}

        for stage_idx, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            stage_id = stage.get("stage_id", f"stage_{stage_idx}")
            display = stage.get("display_name", stage_id)
            stage_iters = int(stage.get("iterations", 1000))
            eval_interval = max(1, int(stage.get("eval_interval", 50)))
            overrides = stage.get("overrides", {})
            if isinstance(overrides, str):
                try:
                    overrides = _json.loads(overrides)
                except Exception:
                    overrides = {}

            print(
                f"\n{'─' * 70}\n"
                f"[UnitPort][STAGE] Stage {stage_idx}/{len(stages) - 1}: "
                f"{display}  ({stage_iters} iters)\n"
                f"[UnitPort][STAGE] Overrides: {list(overrides.keys())}\n"
                f"{'─' * 70}",
                flush=True,
            )

            # ── Apply overrides ──
            self._apply_stage_overrides(overrides, stage_idx)

            # ── Log stage transition to TensorBoard ──
            if self.writer is not None:
                self.writer.add_scalar(
                    "Stage/current_stage", stage_idx, global_iter
                )
                self.writer.add_text(
                    "Stage/transition",
                    f"Stage {stage_idx}: {display} "
                    f"(iters={stage_iters}, overrides={list(overrides.keys())})",
                    global_iter,
                )

            # ── Stage training loop ──
            best_reward_this_stage = float("-inf")
            stage_start_iter = global_iter

            for local_iter in range(stage_iters):
                # Run one iteration of the existing learn() inner loop.
                # We call the iteration directly rather than re-entering
                # learn() to avoid re-initializing buffers.
                self._train_one_iteration(global_iter, total_all_stages)
                global_iter += 1

                # ── Track best reward for this stage ──
                if hasattr(self, "_last_mean_reward"):
                    if self._last_mean_reward > best_reward_this_stage:
                        best_reward_this_stage = self._last_mean_reward
                        if ckpt_strategy in ("save_best_per_stage", "both"):
                            if self.log_dir:
                                self.save(
                                    os.path.join(
                                        self.log_dir,
                                        f"model_best_stage{stage_idx}.pt",
                                    ),
                                    infos={"stage_id": stage_id, "stage_idx": stage_idx},
                                )

                # ── Check advance criteria ──
                if (
                    local_iter > 0
                    and local_iter % eval_interval == 0
                    and self._check_advance_criteria(stage, local_iter)
                ):
                    print(
                        f"[UnitPort][STAGE] Auto-advancing from stage "
                        f"{stage_idx} at local_iter={local_iter}",
                        flush=True,
                    )
                    break

            # ── Stage complete — save checkpoint ──
            self._stage_best_reward[stage_idx] = best_reward_this_stage
            if ckpt_strategy in ("save_on_advance", "both"):
                if self.log_dir:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_stage{stage_idx}_end.pt",
                        ),
                        infos={"stage_id": stage_id, "stage_idx": stage_idx},
                    )

            print(
                f"[UnitPort][STAGE] Stage {stage_idx} ({display}) complete. "
                f"best_reward={best_reward_this_stage:.3f}  "
                f"iters={global_iter - stage_start_iter}",
                flush=True,
            )

        # ── All stages done — final save ──
        self.current_learning_iteration = global_iter
        if self.log_dir:
            self.save(
                os.path.join(self.log_dir, f"model_{global_iter}.pt"),
                infos={"staged_training_complete": True},
            )

        print(
            f"\n{'=' * 70}\n"
            f"[UnitPort][STAGE] All {len(stages)} stages complete. "
            f"Total iters: {global_iter}\n"
            f"{'=' * 70}",
            flush=True,
        )

    def _train_one_iteration(self, it: int, tot_iter: int) -> None:
        """Execute a single training iteration (rollout + update + log).

        Extracted from the inner loop of ``learn()`` so both single-stage
        and staged paths share the same iteration logic.
        """
        start = time.time()

        # ── Lerp anneal (per-stage schedule applied in _apply_stage_overrides) ──
        if self._lerp_schedule is not None:
            s = self._lerp_schedule
            # Use stage-local iteration count for lerp annealing
            stage_local = it - getattr(self, "_lerp_schedule_start_iter", 0)
            alpha = min(1.0, float(stage_local) / max(1, s["over_iters"]))
            new_lerp = s["start"] + (s["end"] - s["start"]) * alpha
            self.alg.discriminator.task_reward_lerp = new_lerp

        obs = self._last_obs
        critic_obs = self._last_critic_obs
        amp_obs = self._last_amp_obs

        # ── Rollout ──
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                try:
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"[UnitPort][AMP] act() failed at it={it} step={i}: "
                        f"obs={tuple(obs.shape)} critic={tuple(critic_obs.shape)} "
                        f"amp={tuple(amp_obs.shape)}: {exc}"
                    ) from exc

                ret = self.env.step(actions)
                obs_raw, privileged_obs_raw, rewards, dones, infos = (
                    ret if len(ret) == 5 else (*ret[:3], ret[3], ret[3])
                )

                def _normalize_obs(t):
                    if isinstance(t, dict):
                        return t.get("policy", next(iter(t.values())))
                    return t

                obs = _normalize_obs(obs_raw).to(self.device)
                if privileged_obs_raw is not None:
                    privileged_obs = _normalize_obs(privileged_obs_raw).to(self.device)
                else:
                    privileged_obs = None
                critic_obs = privileged_obs if privileged_obs is not None else obs

                rewards = rewards.to(self.device)
                dones = dones.to(self.device)

                # AMP observations from wrapper
                amp_info = infos.get("amp", {}) if isinstance(infos, dict) else {}
                pre_amp_obs = amp_info.get("obs", amp_obs).to(self.device)
                next_amp_obs = amp_info.get("obs_next", pre_amp_obs).to(self.device)
                amp_obs = next_amp_obs

                # AMP reward blending
                task_r = rewards
                style_r, _d = self.alg.discriminator.predict_amp_reward(
                    pre_amp_obs,
                    next_amp_obs,
                    normalizer=self.alg.amp_normalizer,
                )
                if style_r.dim() > task_r.dim():
                    style_r = style_r.squeeze(-1)
                elif task_r.dim() > style_r.dim():
                    task_r = task_r.squeeze(-1)
                lerp = float(self.alg.discriminator.task_reward_lerp)
                mixed_r = (1.0 - lerp) * style_r + lerp * task_r

                self.alg.process_env_step(mixed_r, dones, infos)

                self._cur_reward_sum += mixed_r
                self._cur_style_sum += style_r
                self._cur_task_sum += task_r
                self._cur_episode_length += 1

                new_ids = (dones > 0.5).nonzero(as_tuple=False)
                for idx in new_ids:
                    idx = idx.item()
                    self._rewbuffer.append(self._cur_reward_sum[idx].item())
                    self._lenbuffer.append(self._cur_episode_length[idx].item())
                    self._style_rewbuffer.append(self._cur_style_sum[idx].item())
                    self._task_rewbuffer.append(self._cur_task_sum[idx].item())
                    self._cur_reward_sum[idx] = 0
                    self._cur_style_sum[idx] = 0
                    self._cur_task_sum[idx] = 0
                    self._cur_episode_length[idx] = 0

        stop = time.time()
        collection_time = stop - start

        # ── Learning step ──
        start = stop
        self.alg.compute_returns(critic_obs)
        (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
        ) = self.alg.update()
        stop = time.time()
        learn_time = stop - start

        # ── Track last mean reward for stage criteria ──
        self._last_mean_reward = (
            statistics.mean(self._rewbuffer)
            if self._rewbuffer and len(self._rewbuffer) > 0
            else float("-inf")
        )
        self._last_mean_ep_len = (
            statistics.mean(self._lenbuffer)
            if self._lenbuffer and len(self._lenbuffer) > 0
            else 0.0
        )
        self._last_disc_accuracy = (
            (mean_accuracy_policy + mean_accuracy_expert) / 2.0
        )

        # ── Logging ──
        if self.log_dir is not None:
            self._log({
                "it": it,
                "tot_iter": tot_iter,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_amp_loss": mean_amp_loss,
                "mean_grad_pen_loss": mean_grad_pen_loss,
                "mean_policy_pred": mean_policy_pred,
                "mean_expert_pred": mean_expert_pred,
                "mean_accuracy_policy": mean_accuracy_policy,
                "mean_accuracy_expert": mean_accuracy_expert,
                "rewbuffer": self._rewbuffer,
                "lenbuffer": self._lenbuffer,
                "style_rewbuffer": self._style_rewbuffer,
                "task_rewbuffer": self._task_rewbuffer,
            })

        if it % self.save_interval == 0 and self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # Store obs for next iteration
        self._last_obs = obs
        self._last_critic_obs = critic_obs
        self._last_amp_obs = amp_obs

    def _init_staged_buffers(self) -> None:
        """Initialize the per-episode tracking buffers used by _train_one_iteration.

        Called once before the staged training loop starts, replicating
        the buffer setup from the original learn() method.
        """
        self._rewbuffer: deque = deque(maxlen=100)
        self._lenbuffer: deque = deque(maxlen=100)
        self._style_rewbuffer: deque = deque(maxlen=100)
        self._task_rewbuffer: deque = deque(maxlen=100)
        self._cur_reward_sum = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        self._cur_style_sum = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        self._cur_task_sum = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        self._cur_episode_length = torch.zeros(
            self._num_envs, dtype=torch.float, device=self.device
        )
        self._last_mean_reward = float("-inf")
        self._last_mean_ep_len = 0.0
        self._last_disc_accuracy = 0.5

        # Initial observations
        def _normalize_obs(t):
            if isinstance(t, dict):
                return t.get("policy", next(iter(t.values())))
            return t

        obs_raw = self.env.get_observations()
        if isinstance(obs_raw, tuple):
            obs_raw, extras = obs_raw
        obs = _normalize_obs(obs_raw).to(self.device)

        privileged_obs = None
        if hasattr(self.env, "get_privileged_observations"):
            po = self.env.get_privileged_observations()
            if po is not None:
                privileged_obs = _normalize_obs(po).to(self.device)

        amp_obs = _normalize_obs(self.env.get_amp_observations()).to(self.device)
        critic_obs = privileged_obs if privileged_obs is not None else obs

        self._last_obs = obs
        self._last_critic_obs = critic_obs
        self._last_amp_obs = amp_obs

        self.alg.actor_critic.train()
        self.alg.discriminator.train()

    def _apply_stage_overrides(
        self, overrides: Dict[str, Any], stage_idx: int
    ) -> None:
        """Apply a stage's parameter overrides to the running training state.

        Called at the beginning of each stage. Modifies the discriminator,
        optimizer, and algorithm parameters in-place.
        """
        import json as _json

        # Initialize buffers on first stage
        if stage_idx == 0:
            self._init_staged_buffers()

        # ── AMP reward coefficient ──
        if "amp_reward_coef" in overrides:
            val = float(overrides["amp_reward_coef"])
            self.alg.discriminator.amp_reward_coef = val
            print(f"[UnitPort][STAGE]   amp_reward_coef → {val}", flush=True)

        # ── Task reward lerp (initial value for this stage) ──
        if "task_reward_lerp" in overrides:
            val = float(overrides["task_reward_lerp"])
            self.alg.discriminator.task_reward_lerp = val
            print(f"[UnitPort][STAGE]   task_reward_lerp → {val}", flush=True)

        # ── Lerp schedule (per-stage, resets relative to stage start) ──
        if "lerp_schedule" in overrides:
            raw = overrides["lerp_schedule"]
            if raw == "none" or not raw:
                self._lerp_schedule = None
                print("[UnitPort][STAGE]   lerp_schedule → disabled", flush=True)
            else:
                try:
                    s = _json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(s, dict) and "start" in s and "end" in s:
                        self._lerp_schedule = {
                            "start": float(s["start"]),
                            "end": float(s["end"]),
                            "over_iters": int(s.get("over_iters", 1)),
                        }
                        # Record the global iter at which this schedule starts
                        self._lerp_schedule_start_iter = self.current_learning_iteration
                        # Set initial lerp value
                        self.alg.discriminator.task_reward_lerp = float(s["start"])
                        print(
                            f"[UnitPort][STAGE]   lerp_schedule → "
                            f"{s['start']}→{s['end']} over {s.get('over_iters',1)} iters",
                            flush=True,
                        )
                except Exception:
                    self._lerp_schedule = None

        # ── Discriminator learning rate ──
        if "disc_lr" in overrides:
            val = float(overrides["disc_lr"])
            for pg in self.alg.optimizer.param_groups:
                if pg.get("name", "").startswith("amp"):
                    pg["lr"] = val
            print(f"[UnitPort][STAGE]   disc_lr → {val}", flush=True)

        # ── Discriminator gradient penalty ──
        if "disc_grad_penalty" in overrides:
            val = float(overrides["disc_grad_penalty"])
            self.alg.amp_disc_grad_penalty = val
            print(f"[UnitPort][STAGE]   disc_grad_penalty → {val}", flush=True)

        # ── Discriminator label smoothing ──
        if "disc_label_smoothing" in overrides:
            val = float(overrides["disc_label_smoothing"])
            self.alg.amp_disc_label_smoothing = val
            print(f"[UnitPort][STAGE]   disc_label_smoothing → {val}", flush=True)

        # ── PPO learning rate ──
        if "learning_rate" in overrides:
            val = float(overrides["learning_rate"])
            # Update base + non-discriminator param groups
            self.alg.learning_rate = val
            for pg in self.alg.optimizer.param_groups:
                if not pg.get("name", "").startswith("amp"):
                    pg["lr"] = val
            print(f"[UnitPort][STAGE]   learning_rate → {val}", flush=True)

        # ── Entropy coefficient ──
        if "entropy_coef" in overrides:
            val = float(overrides["entropy_coef"])
            self.alg.entropy_coef = val
            print(f"[UnitPort][STAGE]   entropy_coef → {val}", flush=True)

        # ── Clip param ──
        if "clip_param" in overrides:
            val = float(overrides["clip_param"])
            self.alg.clip_param = val
            print(f"[UnitPort][STAGE]   clip_param → {val}", flush=True)

        # ── Training mode (PPO ↔ AMP_PPO) ──
        # NOTE: Full PPO→AMP mode switch (initializing discriminator from
        # scratch) is complex and deferred to Phase 2.2. For now, the
        # training_mode override controls whether the discriminator's style
        # reward contributes to the mix:
        #   "PPO" → set task_reward_lerp = 1.0 (pure task, no style)
        #   "AMP_PPO" → use the configured lerp value
        if "training_mode" in overrides:
            mode = str(overrides["training_mode"]).strip()
            if mode == "PPO":
                self.alg.discriminator.task_reward_lerp = 1.0
                self._lerp_schedule = None  # disable lerp annealing
                print(
                    "[UnitPort][STAGE]   training_mode → PPO "
                    "(lerp=1.0, AMP discriminator frozen)",
                    flush=True,
                )
            elif mode == "AMP_PPO":
                # Restore configured lerp (may be overridden by explicit
                # task_reward_lerp or lerp_schedule above)
                if "task_reward_lerp" not in overrides:
                    self.alg.discriminator.task_reward_lerp = self._baseline_lerp
                print(
                    f"[UnitPort][STAGE]   training_mode → AMP_PPO "
                    f"(lerp={self.alg.discriminator.task_reward_lerp})",
                    flush=True,
                )

    def _check_advance_criteria(
        self, stage: Dict[str, Any], local_iter: int
    ) -> bool:
        """Evaluate whether to advance to the next stage.

        Returns True if all criteria are met and the mode allows auto-advance.
        """
        mode = str(stage.get("advance_mode", "auto")).strip().lower()
        if mode == "manual":
            return False

        criteria = stage.get("advance_criteria", {})
        if isinstance(criteria, str):
            import json as _json
            try:
                criteria = _json.loads(criteria)
            except Exception:
                criteria = {}
        if not isinstance(criteria, dict):
            return False

        # ── min_iterations ──
        min_iter = int(criteria.get("min_iterations", 0))
        if local_iter < min_iter:
            return False

        # ── min_mean_episode_length ──
        min_ep_len = float(criteria.get("min_mean_episode_length", 0))
        if min_ep_len > 0 and self._last_mean_ep_len < min_ep_len:
            return False

        # ── min_mean_reward ──
        min_rew = criteria.get("min_mean_reward")
        if min_rew is not None and self._last_mean_reward < float(min_rew):
            return False

        # ── disc_accuracy_range ──
        disc_range = criteria.get("disc_accuracy_range")
        if disc_range and isinstance(disc_range, (list, tuple)) and len(disc_range) == 2:
            lo, hi = float(disc_range[0]), float(disc_range[1])
            if not (lo <= self._last_disc_accuracy <= hi):
                return False

        # ── Hybrid: visual check required → log and don't advance ──
        if mode == "hybrid" and criteria.get("visual_check_required"):
            print(
                f"[UnitPort][STAGE] Criteria met but visual_check_required=true. "
                f"Waiting for manual advance.",
                flush=True,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str, infos: Optional[Dict[str, Any]] = None) -> None:
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "discriminator_state_dict": self.alg.discriminator.state_dict(),
                "amp_normalizer": self.alg.amp_normalizer.state_dict()
                if hasattr(self.alg.amp_normalizer, "state_dict")
                else self.alg.amp_normalizer,
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    @staticmethod
    def _translate_rsl_rl_ppo_state_dict(loaded: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Translate a stock RSL-RL PPO checkpoint to this runner's key layout.

        Stock RSL-RL (the ``OnPolicyRunner`` / ``MLPModel`` path) saves
        the actor and critic under separate top-level keys with MLP
        weights named ``mlp.N.weight`` / ``mlp.N.bias``:

            {
              "actor_state_dict":  {"mlp.0.weight": ..., "mlp.0.bias": ...,
                                    ..., "distribution.std_param": ...},
              "critic_state_dict": {"mlp.0.weight": ..., ...},
              "optimizer_state_dict": ...,
              "iter": ...,
            }

        This runner's ``ActorCritic`` uses ``nn.Sequential`` for actor /
        critic, so its state_dict keys are ``actor.N.weight``,
        ``critic.N.weight``, and a top-level ``std`` parameter.

        Returns a merged ``model_state_dict``-shaped dict when the input
        looks like the stock format, or ``None`` when it does not (in
        which case the caller should fall back to the native format).
        """
        if not isinstance(loaded, dict):
            return None
        if "actor_state_dict" not in loaded or "critic_state_dict" not in loaded:
            return None

        actor_sd = loaded["actor_state_dict"]
        critic_sd = loaded["critic_state_dict"]
        merged: Dict[str, torch.Tensor] = {}
        for k, v in actor_sd.items():
            if k.startswith("mlp."):
                merged[f"actor.{k[len('mlp.'):]}"] = v
            elif k == "distribution.std_param":
                # RSL-RL stores the learned std here; this runner uses
                # a top-level ``std`` Parameter of the same shape.
                merged["std"] = v
        for k, v in critic_sd.items():
            if k.startswith("mlp."):
                merged[f"critic.{k[len('mlp.'):]}"] = v
        return merged

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        *,
        warm_start_actor: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint.

        Parameters
        ----------
        path
            Absolute path to a ``.pt`` produced by either this runner
            (full AMP state) or by a pure PPO ``OnPolicyRunner``.
        load_optimizer
            Whether to restore optimizer state. Ignored when
            ``warm_start_actor`` is True.
        warm_start_actor
            When True, load only the actor-critic weights and keep the
            discriminator, amp_normalizer, optimizer, and iteration
            counter freshly initialized. This is the path used to seed
            an AMP-PPO run from a pure PPO checkpoint (which has no
            ``discriminator_state_dict``), and also to reset optimizer
            momentum when switching tasks / reward layouts while
            keeping the actor weights.

            The actor-critic load uses ``strict=False`` so extra keys
            in the checkpoint (e.g. unused heads from an older code
            path) do not abort the load. Shape mismatch is still
            raised — that is a genuine architecture error the user
            must fix by aligning ``actor_hidden_dims`` / observation
            layout with the source checkpoint.
        """
        loaded = torch.load(path)

        # Detect checkpoint format. Two shapes are accepted:
        #
        #   (a) Native AMP runner format — ``model_state_dict`` key
        #       (produced by this runner's own ``save()``).
        #   (b) Stock RSL-RL PPO format — separate ``actor_state_dict``
        #       and ``critic_state_dict`` keys with ``mlp.N.*`` naming.
        #       Translated to (a) via ``_translate_rsl_rl_ppo_state_dict``.
        #       Only valid for warm-start (no discriminator / amp_normalizer).
        model_sd: Optional[Dict[str, torch.Tensor]] = None
        if "model_state_dict" in loaded:
            model_sd = loaded["model_state_dict"]
        else:
            translated = self._translate_rsl_rl_ppo_state_dict(loaded)
            if translated is not None:
                if not warm_start_actor:
                    raise KeyError(
                        f"Checkpoint at {path!r} is a stock RSL-RL PPO "
                        f"checkpoint (actor_state_dict / critic_state_dict "
                        f"format). Full resume into AMP-PPO is not possible "
                        f"because the checkpoint has no discriminator / "
                        f"optimizer-for-discriminator state. Set Start Point "
                        f"load_mode='warm_start_actor' to seed the actor "
                        f"from this checkpoint instead."
                    )
                model_sd = translated
                print(
                    f"[UnitPort][AMP] Translated RSL-RL PPO checkpoint "
                    f"(mlp.N.* → actor.N.* / critic.N.*), "
                    f"{len(model_sd)} tensors.",
                    flush=True,
                )

        if model_sd is None:
            raise KeyError(
                f"Checkpoint at {path!r} has no 'model_state_dict' key and "
                f"is not in stock RSL-RL PPO format — cannot be loaded by "
                f"AmpOnPolicyRunner. Keys present: "
                f"{sorted(loaded.keys()) if isinstance(loaded, dict) else type(loaded).__name__}"
            )

        try:
            self.alg.actor_critic.load_state_dict(
                model_sd, strict=not warm_start_actor
            )
        except RuntimeError as exc:
            # Shape mismatch — almost always an architecture mismatch
            # between the canvas ``il_policy_network`` and the source
            # run's saved dims. Surface a clear, actionable message.
            raise RuntimeError(
                f"[AMP] actor-critic load_state_dict failed against "
                f"checkpoint {path!r}. Typical cause: the current canvas's "
                f"actor/critic hidden dims or the observation layout do "
                f"not match the source run. Underlying error: {exc}"
            ) from exc

        if warm_start_actor:
            # Actor-only warm start: discriminator, amp_normalizer, and
            # optimizer stay freshly initialized; iteration counter
            # resets so the new run's logs start at 0.
            self.current_learning_iteration = 0
            return loaded.get("infos")

        # ── Full resume (AMP ckpt expected) ─────────────────────────
        disc_state = loaded.get("discriminator_state_dict")
        if disc_state is None:
            raise KeyError(
                f"Checkpoint {path!r} has no 'discriminator_state_dict' — "
                f"this looks like a pure PPO checkpoint, not an AMP run. "
                f"Use warm_start_actor=True (Start Point load_mode="
                f"'warm_start_actor') to seed an AMP run from a PPO "
                f"checkpoint."
            )
        self.alg.discriminator.load_state_dict(disc_state)

        norm_state = loaded.get("amp_normalizer")
        if isinstance(norm_state, dict) and hasattr(self.alg.amp_normalizer, "load_state_dict"):
            self.alg.amp_normalizer.load_state_dict(norm_state)
        elif norm_state is not None:
            # Legacy checkpoints stored the whole Normalizer object
            self.alg.amp_normalizer = norm_state

        if load_optimizer and "optimizer_state_dict" in loaded:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = int(loaded.get("iter", 0))
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
