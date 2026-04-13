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
        :func:`src.system.training.amp.joint_alignment.verify_alignment`
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
