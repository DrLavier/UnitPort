"""AMP vec-env wrapper — subclass of isaaclab_rl's RslRlVecEnvWrapper.

Per AMP_design.yaml §4.amp_backend.vec_env_wrapper.

This wrapper is the *only* piece of AMP infra that runs inside
``env.step()``. Its job is to make the episode-boundary handling of
the AMP transition pair ``(s_t, s_{t+1})`` correct when an env
auto-resets between ``t`` and ``t+1``:

1. Before calling ``super().step()``, cache the current amp_obs.
2. Call the wrapped env's step — inside that call some envs reset.
3. For envs whose ``dones[i] == True``, the "post-step" amp_obs the
   inner env now reports is from the **fresh** episode, not from the
   end of the previous one. The correct ``s_{t+1}`` is the CACHED
   pre-step amp_obs indexed at that env.
4. Pack the corrected ``(s_t, s_{t+1})`` pair onto ``infos["amp"]``
   so the runner can insert it into the replay buffer.

Without this fix the discriminator gets fake transitions that look
like "dynamic motion instantly teleports to a spawn pose", and
training silently degrades (no exception, no log warning, just bad
policies).

Importability
-------------
The wrapper subclasses ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` which
only exists inside the Isaac Sim venv. In the main UnitPort venv that
import raises ``ModuleNotFoundError``; this module catches it and
falls back to a stub base class with the same public API surface so
the module still imports cleanly and unit tests can exercise the
``cache_and_correct_amp_obs`` helper.

The constant ``AMP_WRAPPER_HAS_REAL_BASE`` tells callers whether the
class they get back is the real subclass or the main-venv stub.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base class resolution
# ---------------------------------------------------------------------------

AMP_WRAPPER_HAS_REAL_BASE: bool
_Base: Any  # the RslRlVecEnvWrapper class object we subclass

try:
    # Inside the Isaac Sim venv this resolves to the real wrapper.
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as _Base  # type: ignore
    AMP_WRAPPER_HAS_REAL_BASE = True
except Exception:
    # Main venv fallback: define a minimal stub with the pieces our
    # subclass touches. ``step`` just forwards to the inner env.
    AMP_WRAPPER_HAS_REAL_BASE = False

    class _Base:  # type: ignore[no-redef]
        """Stand-in base class for environments without isaaclab_rl."""

        def __init__(self, env: Any, clip_actions: float = 1.0) -> None:
            self.env = env
            self.clip_actions = float(clip_actions)
            # Forward a handful of attributes that AMP code expects on
            # the wrapper itself.
            self.num_envs = getattr(env, "num_envs", 0)
            self.device = getattr(env, "device", "cpu")
            self.num_obs = getattr(env, "num_obs", 0)
            self.num_actions = getattr(env, "num_actions", 0)
            self.num_privileged_obs = getattr(env, "num_privileged_obs", None)

        def step(self, actions: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
            # Stub: just forward. Real wrapper translates obs_dict →
            # tensor form etc; we don't need that for unit tests.
            return self.env.step(actions)  # pragma: no cover

        def reset(self) -> Any:
            return self.env.reset()  # pragma: no cover

        def get_observations(self) -> Any:
            return self.env.get_observations()  # pragma: no cover

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


# Type alias for the amp-obs extractor the launcher injects. The extractor
# maps the wrapper instance to a tensor of shape (num_envs, amp_obs_dim).
AmpObsExtractor = Callable[["AmpRslRlVecEnvWrapper"], Any]


class AmpRslRlVecEnvWrapper(_Base):
    """AMP-aware replacement for the stock ``RslRlVecEnvWrapper``.

    Use this INSTEAD of (not on top of) ``RslRlVecEnvWrapper``. Per
    AMP_design.yaml §4.amp_backend.vec_env_wrapper.replacement_semantics:

        AMP path:  env = AmpRslRlVecEnvWrapper(gym_env, ...)
        PPO path:  env = RslRlVecEnvWrapper(gym_env, ...)

    Never nest.

    Parameters
    ----------
    env:
        The ManagerBasedRLEnv (or main-venv stub) to wrap.
    clip_actions:
        Forwarded to the base class.
    num_amp_obs:
        Dimension of the AMP observation the extractor produces.
    amp_obs_fields:
        Ordered list of canonical field names — passed through
        :func:`application.training.amp.joint_alignment.verify_alignment`
        at launcher time.
    amp_obs_extractor:
        Callable returning a tensor of shape ``(num_envs, num_amp_obs)``
        when given the wrapper instance. The launcher builds this from
        the env config + the user's ``amp_obs_fields`` selection.
    """

    def __init__(
        self,
        env: Any,
        *,
        clip_actions: float = 1.0,
        num_amp_obs: int,
        amp_obs_fields: List[str],
        amp_obs_extractor: AmpObsExtractor,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)  # type: ignore[arg-type]
        self._num_amp_obs = int(num_amp_obs)
        self._amp_obs_fields: List[str] = list(amp_obs_fields)
        self._amp_obs_extractor = amp_obs_extractor

        # ``include_history_steps`` is an attribute the vendored
        # AmpOnPolicyRunner reads on ``env`` to decide whether actor
        # observations are stacked across time. AMP_for_hardware uses
        # this for recurrent policies; our default is None (no stacking)
        # and the launcher can override via set_include_history_steps.
        self.include_history_steps: Optional[int] = None

        # Pre-step cache for correct (s, s') pairs at episode boundaries.
        # Lazily allocated on first step so we don't need to know the
        # device / dtype at construction time.
        self._pre_step_amp_obs: Any = None

    # ------------------------------------------------------------------
    # Public attributes the runner reads
    # ------------------------------------------------------------------

    @property
    def num_amp_obs(self) -> int:
        return self._num_amp_obs

    @property
    def amp_obs_fields(self) -> List[str]:
        return list(self._amp_obs_fields)

    def get_amp_observations(self) -> Any:
        """Compute the current AMP observation tensor for all envs.

        Uses the extractor supplied at construction. This is the one
        place in the wrapper where AMP-specific env layout assumptions
        live — everything else is format-agnostic.
        """
        return self._amp_obs_extractor(self)

    def set_include_history_steps(self, n: Optional[int]) -> None:
        """Optional: enable amp_obs history stacking for recurrent runs."""
        self.include_history_steps = n

    # ------------------------------------------------------------------
    # Stage Switch hot-swap API (P0-2b)
    # ------------------------------------------------------------------
    #
    # The AMP runner calls these between stages to reconfigure the env
    # pipeline without rebuilding anything. The wrapper itself stores no
    # config — it just routes the payload to whichever layer actually
    # owns the setting:
    #
    #   1. Inner env's own ``swap_<x>`` (e.g. UnitreeGymEnv exposes the
    #      full family). Preferred path — the env knows its own fields.
    #   2. Vector-container sub-envs (``.envs`` list on SyncVectorEnv /
    #      DummyVecEnv). Called on each sub-env.
    #   3. Isaac Lab manager APIs for reward / termination / command
    #      weight updates, when the inner env exposes the managers.
    #   4. Otherwise a single warning — swap is a no-op for this stage.

    def _iter_swap_targets(self) -> List[Any]:
        """Collect candidate objects to dispatch a ``swap_*`` call onto.

        Returns a list so a single wrapper over a vectorised env still
        touches every underlying gym instance. The order is inner-first
        (direct env) then sub-envs; callers stop after any target
        handles the call.
        """
        inner = getattr(self, "env", None)
        targets: List[Any] = []
        if inner is not None:
            targets.append(getattr(inner, "unwrapped", inner))
            sub = getattr(inner, "envs", None)
            if sub is not None:
                for e in sub:
                    targets.append(getattr(e, "unwrapped", e))
        return targets

    def _delegate_swap(self, method_name: str, payload: Any) -> bool:
        """Route a ``swap_*`` call to the first target that implements it.

        Returns True when at least one target handled the call. Sub-env
        iteration calls every env that has the method (so all parallel
        instances in a SyncVectorEnv / DummyVecEnv get the update).
        """
        handled = False
        for target in self._iter_swap_targets():
            fn = getattr(target, method_name, None)
            if callable(fn):
                try:
                    fn(payload)
                    handled = True
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"{method_name} on {type(target).__name__} raised "
                        f"{type(exc).__name__}: {exc}",
                        stacklevel=2,
                    )
        return handled

    def _isaac_lab_reward_swap(self, reward_terms: Dict[str, float]) -> bool:
        """Isaac Lab fallback for ``swap_reward_terms``.

        Updates each named reward term's ``.weight`` via the standard
        ``reward_manager.get_term_cfg`` / ``set_term_cfg`` pair. Silently
        skips terms the manager doesn't know about — user-authored
        reward-term names may not match the env's registry exactly.
        """
        for target in self._iter_swap_targets():
            mgr = getattr(target, "reward_manager", None)
            if mgr is None:
                continue
            get_cfg = getattr(mgr, "get_term_cfg", None)
            set_cfg = getattr(mgr, "set_term_cfg", None)
            if not (callable(get_cfg) and callable(set_cfg)):
                continue
            for name, weight in reward_terms.items():
                try:
                    cfg = get_cfg(name)
                    cfg.weight = float(weight)
                    set_cfg(name, cfg)
                except Exception:
                    # Unknown term name — skip rather than abort the
                    # whole swap; log at the call site already mentions
                    # which sections were attempted.
                    continue
            return True
        return False

    def swap_reward_terms(self, reward_terms: Any) -> bool:
        if not isinstance(reward_terms, dict) or not reward_terms:
            return False
        if self._delegate_swap("swap_reward_terms", reward_terms):
            return True
        return self._isaac_lab_reward_swap(reward_terms)

    def swap_termination(self, termination_cfg: Any) -> bool:
        return self._delegate_swap("swap_termination", termination_cfg)

    def swap_domain_rand(self, dr_cfg: Any) -> bool:
        return self._delegate_swap("swap_domain_rand", dr_cfg)

    def swap_reference_motion(self, motion_cfg: Any) -> bool:
        return self._delegate_swap("swap_reference_motion", motion_cfg)

    def swap_commands(self, commands_cfg: Any) -> bool:
        return self._delegate_swap("swap_commands", commands_cfg)

    def swap_noise(self, noise_cfg: Any) -> bool:
        return self._delegate_swap("swap_noise", noise_cfg)

    def swap_obs_action_config(self, oac_cfg: Any) -> bool:
        return self._delegate_swap("swap_obs_action_config", oac_cfg)

    # ------------------------------------------------------------------
    # Per-iteration curriculum hook (cold-start ramps)
    # ------------------------------------------------------------------
    #
    # Called from AMPOnPolicyRunner._update_curriculum at the top of
    # each outer training iteration. The runner resolves each active
    # schedule to a concrete scalar and hands the dict in; this method
    # pushes the scalars onto the appropriate Isaac Lab managers.
    #
    # Contract with the runner:
    #   scales["command_velocity"]        → multiplier ∈ [0, 1+] on
    #       each command-term range (lin_vel_x / lin_vel_y / ang_vel_z).
    #   scales["action_scale"]            → multiplier on the joint
    #       action-term scale (applied on top of the static canvas
    #       setting so 1.0 means "full configured scale").
    #   scales["termination_base_height"] → absolute threshold (metres)
    #       for the base_height termination term.
    #   scales["obs_noise_scale"]         → multiplier on every obs
    #       group's noise model std (Gaussian/Uniform/constant).
    #
    # Entropy is applied directly to ``alg.entropy_coef`` by the runner
    # (no env touch needed) so it is *not* in this dict.
    #
    # Implementation strategy: walk _iter_swap_targets() and for each
    # target that has the relevant Isaac Lab manager, mutate the
    # matching cfg field. Targets without the manager (e.g. MuJoCo
    # backend) are skipped silently — curriculum then becomes a no-op
    # on that path, which is the correct cross-backend fallback.

    def apply_curriculum_scales(self, scales: Dict[str, float]) -> Dict[str, bool]:
        """Apply resolved curriculum scalars to the inner env's managers.

        Returns a dict mapping dimension_name → applied_success so the
        runner can log which levers actually landed. A False entry
        indicates the backend has no matching manager / term and the
        scalar was silently dropped (non-fatal — other dimensions
        still applied).

        Safe to call every iteration; no allocations on the hot path
        aside from the small dict literals and the manager lookups.
        """
        applied: Dict[str, bool] = {
            "command_velocity": False,
            "action_scale": False,
            "termination_base_height": False,
            "obs_noise_scale": False,
        }

        if "command_velocity" in scales:
            applied["command_velocity"] = self._apply_command_scale(
                float(scales["command_velocity"])
            )
        if "action_scale" in scales:
            applied["action_scale"] = self._apply_action_scale(
                float(scales["action_scale"])
            )
        if "termination_base_height" in scales:
            applied["termination_base_height"] = self._apply_termination_height(
                float(scales["termination_base_height"])
            )
        if "obs_noise_scale" in scales:
            applied["obs_noise_scale"] = self._apply_obs_noise_scale(
                float(scales["obs_noise_scale"])
            )
        return applied

    # ---- per-dimension implementations ---------------------------------

    @staticmethod
    def _scale_range(rng: Any, factor: float) -> Any:
        """Multiply a (lo, hi) tuple/list by *factor*, preserving type."""
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except Exception:
            return rng
        scaled = (lo * factor, hi * factor)
        if isinstance(rng, list):
            return [scaled[0], scaled[1]]
        return scaled

    def _apply_command_scale(self, factor: float) -> bool:
        """Scale every command-term range by *factor* on the fly.

        Isaac Lab's UniformVelocityCommandCfg exposes a nested ``ranges``
        object with ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` and
        optionally ``heading``. We stash the original ranges on first
        call so repeated applications always lerp against the static
        baseline (otherwise consecutive factor=0.5 calls would compound
        into 0.25 → 0.125 → ...).
        """
        handled = False
        for target in self._iter_swap_targets():
            mgr = getattr(target, "command_manager", None)
            if mgr is None:
                continue
            get_cfg = getattr(mgr, "get_term_cfg", None)
            set_cfg = getattr(mgr, "set_term_cfg", None)
            term_names = getattr(mgr, "active_terms", None) or []
            if not (callable(get_cfg) and callable(set_cfg) and term_names):
                continue
            cache = getattr(self, "_command_baseline_ranges", None)
            if cache is None:
                cache = {}
                self._command_baseline_ranges = cache
            for name in term_names:
                try:
                    cfg = get_cfg(name)
                except Exception:
                    continue
                ranges = getattr(cfg, "ranges", None)
                if ranges is None:
                    continue
                key = (id(target), name)
                if key not in cache:
                    cache[key] = {
                        field: tuple(getattr(ranges, field))
                        for field in ("lin_vel_x", "lin_vel_y",
                                      "ang_vel_z", "heading")
                        if hasattr(ranges, field)
                    }
                for field, base in cache[key].items():
                    scaled = (base[0] * factor, base[1] * factor)
                    try:
                        setattr(ranges, field, scaled)
                    except Exception:
                        continue
                try:
                    set_cfg(name, cfg)
                except Exception:
                    pass
                handled = True
        return handled

    def _apply_action_scale(self, factor: float) -> bool:
        """Scale every action-term ``scale`` by *factor* against its baseline."""
        handled = False
        for target in self._iter_swap_targets():
            mgr = getattr(target, "action_manager", None)
            if mgr is None:
                continue
            get_cfg = getattr(mgr, "get_term_cfg", None)
            set_cfg = getattr(mgr, "set_term_cfg", None)
            term_names = getattr(mgr, "active_terms", None) or []
            if not (callable(get_cfg) and callable(set_cfg) and term_names):
                continue
            cache = getattr(self, "_action_baseline_scale", None)
            if cache is None:
                cache = {}
                self._action_baseline_scale = cache
            for name in term_names:
                try:
                    cfg = get_cfg(name)
                except Exception:
                    continue
                if not hasattr(cfg, "scale"):
                    continue
                key = (id(target), name)
                if key not in cache:
                    cache[key] = float(getattr(cfg, "scale", 1.0))
                try:
                    cfg.scale = cache[key] * factor
                    set_cfg(name, cfg)
                except Exception:
                    continue
                handled = True
        return handled

    def _apply_termination_height(self, threshold: float) -> bool:
        """Set the ``base_height`` termination-term threshold absolutely."""
        handled = False
        for target in self._iter_swap_targets():
            mgr = getattr(target, "termination_manager", None)
            if mgr is None:
                continue
            get_cfg = getattr(mgr, "get_term_cfg", None)
            set_cfg = getattr(mgr, "set_term_cfg", None)
            if not (callable(get_cfg) and callable(set_cfg)):
                continue
            for name in ("base_height", "min_height", "body_height"):
                try:
                    cfg = get_cfg(name)
                except Exception:
                    continue
                params = getattr(cfg, "params", None)
                if isinstance(params, dict):
                    for pkey in ("minimum_height", "threshold", "min_height"):
                        if pkey in params:
                            params[pkey] = float(threshold)
                            try:
                                set_cfg(name, cfg)
                                handled = True
                            except Exception:
                                pass
                            break
        return handled

    def _apply_obs_noise_scale(self, factor: float) -> bool:
        """Multiply every obs-term noise ``std`` / ``n_max``-``n_min`` by *factor*."""
        handled = False
        for target in self._iter_swap_targets():
            mgr = getattr(target, "observation_manager", None)
            if mgr is None:
                continue
            cache = getattr(self, "_obs_noise_baseline", None)
            if cache is None:
                cache = {}
                self._obs_noise_baseline = cache
            group_term_names = getattr(mgr, "active_terms", None)
            if not isinstance(group_term_names, dict):
                continue
            get_cfg = getattr(mgr, "get_term_cfg", None)
            set_cfg = getattr(mgr, "set_term_cfg", None)
            if not (callable(get_cfg) and callable(set_cfg)):
                continue
            for group, names in group_term_names.items():
                for name in (names or []):
                    try:
                        cfg = get_cfg(group, name)
                    except TypeError:
                        try:
                            cfg = get_cfg(name)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    noise = getattr(cfg, "noise", None)
                    if noise is None:
                        continue
                    key = (id(target), group, name)
                    if key not in cache:
                        cache[key] = {
                            attr: float(getattr(noise, attr))
                            for attr in ("std", "mean", "n_min", "n_max")
                            if hasattr(noise, attr) and isinstance(
                                getattr(noise, attr), (int, float)
                            )
                        }
                    for attr, base in cache[key].items():
                        try:
                            setattr(noise, attr, base * factor)
                        except Exception:
                            continue
                    try:
                        set_cfg(group, name, cfg)
                    except TypeError:
                        try:
                            set_cfg(name, cfg)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    handled = True
        return handled

    # ------------------------------------------------------------------
    # step() override — the whole reason this class exists
    # ------------------------------------------------------------------

    def step(self, actions: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """Wrap the inner step with (s, s') edge-case handling.

        Order of operations:
        1. Cache current amp_obs (this is ``s_t``).
        2. Call the real base-class step → gives us obs / rewards /
           dones / infos. Inner env auto-resets envs whose episode
           ended *during* this step.
        3. Query amp_obs again → this is ``post_step_amp_obs``. For
           envs whose dones[i] is True, the observation is from the
           **new** episode, not the end of the old one.
        4. Fix by replacing post_step rows where dones[i] is True with
           the cached pre-step rows. Those are the true final frames
           of the previous episode, which is what the discriminator
           should learn against.
        5. Attach the corrected pair to ``infos["amp"]`` so the runner
           can insert it into the replay buffer.
        """
        # 1. cache
        pre = self.get_amp_observations()
        self._pre_step_amp_obs = pre

        # 2. forward to base class
        result = super().step(actions)  # type: ignore[misc]
        # Unpack whatever shape the base class produced. Both the
        # isaac-venv and main-venv stubs can return a 4-tuple.
        if isinstance(result, tuple) and len(result) == 4:
            obs, rewards, dones, infos = result
        elif isinstance(result, tuple) and len(result) == 5:
            # Isaac Lab newer API: (obs, rewards, terminated, truncated, infos)
            obs, rewards, terminated, truncated, infos = result
            dones = self._combine_dones(terminated, truncated)
        else:
            # Unknown shape — let the caller unpack as-is; skip the
            # amp info entirely for safety.
            return result

        # 3. post-step amp_obs
        post = self.get_amp_observations()

        # 4. fix done rows
        corrected_next = self.cache_and_correct_amp_obs(pre, post, dones)

        # 5. attach for the runner
        if not isinstance(infos, dict):
            infos = {}
        infos["amp"] = {
            "obs": pre,
            "obs_next": corrected_next,
        }

        # Return in the shape the base class produced (prefer 4-tuple
        # for the AMP runner, which was written against the legacy API).
        return obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Helpers — kept as separate methods so tests can exercise them
    # without needing a real base class
    # ------------------------------------------------------------------

    @staticmethod
    def cache_and_correct_amp_obs(pre: Any, post: Any, dones: Any) -> Any:
        """Replace ``post`` rows where ``dones`` is True with ``pre`` rows.

        Works for both numpy arrays and torch tensors without importing
        torch — we rely on duck typing on ``__getitem__`` / index
        assignment.
        """
        if pre is None or post is None or dones is None:
            return post
        # Prefer a boolean mask; accept anything with the right length.
        try:
            out = post.clone() if hasattr(post, "clone") else post.copy()
        except Exception:
            out = post  # last-resort: no-op

        # Build a boolean index. torch tensors, numpy arrays, python
        # lists and tuples all work via enumerate + scalar bool.
        try:
            for i, d in enumerate(dones):
                flag = bool(d.item()) if hasattr(d, "item") else bool(d)
                if flag:
                    out[i] = pre[i]
        except Exception:
            # If anything goes wrong we return post unchanged — better
            # to silently run with a slightly-wrong ``s_{t+1}`` than to
            # crash a training run that's already hours in.
            return post
        return out

    @staticmethod
    def _combine_dones(terminated: Any, truncated: Any) -> Any:
        """Merge terminated/truncated masks into a single ``dones`` array.

        Pure elementwise OR. Prefers torch ops when available.
        """
        try:
            import torch  # lazy
            if isinstance(terminated, torch.Tensor) or isinstance(truncated, torch.Tensor):
                return torch.logical_or(
                    terminated.bool() if hasattr(terminated, "bool") else terminated,
                    truncated.bool() if hasattr(truncated, "bool") else truncated,
                )
        except Exception:
            pass
        return np.logical_or(np.asarray(terminated), np.asarray(truncated))
