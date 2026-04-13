"""ReactiveLoop — real-time policy execution loop for reactive skills.

Unlike episode-based PolicyRunner (reset → N steps → done), ReactiveLoop
runs continuously: each tick reads sensor state + commands, runs inference,
and sends actions.  There is no episode concept or reset.

The loop terminates when:
  - External stop signal (cancel_check returns True)
  - Safety interception triggers
  - Timeout (if configured)
  - Termination condition callback returns True

Placed in engine/ (alongside runtime_engine.py, workflow_runner.py,
node_executor.py) because it is an execution path, not a simulation component.

Usage::

    loop = ReactiveLoop(manifest, inference_engine, obs_assembler,
                         action_applier, command_bus)
    loop.run(env, cancel_check=lambda: False)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.system.skill.skill_manifest import SkillManifest

log = logging.getLogger(__name__)


class ReactiveLoop:
    """Real-time policy execution loop for reactive SkillManifest skills.

    Parameters
    ----------
    manifest:
        SkillManifest v2 with execution_mode="reactive".
    inference_engine:
        Object with ``infer(obs: np.ndarray) -> np.ndarray`` method.
    obs_assembler:
        ObsAssembler instance configured for this manifest.
    action_applier:
        ActionApplier instance (optionally with PD controller).
    command_bus:
        CommandBus instance for reading real-time command values.
    """

    def __init__(
        self,
        manifest: SkillManifest,
        inference_engine: Any,
        obs_assembler: Any,
        action_applier: Any,
        command_bus: Any,
    ) -> None:
        self._manifest = manifest
        self._inference = inference_engine
        self._obs_assembler = obs_assembler
        self._action_applier = action_applier
        self._command_bus = command_bus

        self._control_dt = 1.0 / max(manifest.control_frequency_hz, 1.0)
        self._last_action: Optional[np.ndarray] = None
        self._tick_count = 0
        self._running = False
        # Decimation = number of physics substeps per policy decision.
        # Derived lazily from ``env.mj_model.opt.timestep`` on the first
        # tick (we don't have ``env`` at construction time). Cached so
        # the per-tick cost is one attribute lookup. ``None`` means we
        # haven't probed yet; ``1`` means real-hardware / single-step
        # path (no inner physics loop).
        self._decimation: Optional[int] = None

        # Callbacks
        self.on_tick: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None
        self.on_stop: Optional[Callable[[str], None]] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def last_action(self) -> Optional[np.ndarray]:
        return self._last_action

    def run(
        self,
        env: Any,
        *,
        cancel_check: Optional[Callable[[], bool]] = None,
        termination_check: Optional[Callable[[], bool]] = None,
        timeout_seconds: float = 0.0,
        max_ticks: int = 0,
    ) -> Dict[str, Any]:
        """Execute the reactive loop until termination.

        Parameters
        ----------
        env:
            SimEnvContext or SDK adapter providing sensor data.
        cancel_check:
            Returns True to stop the loop (e.g. user pressed Stop).
        termination_check:
            Returns True when a task-level condition is met (e.g. reached target).
        timeout_seconds:
            Maximum wall-clock time. 0 = no timeout.
        max_ticks:
            Maximum number of policy ticks. 0 = unlimited.

        Returns
        -------
        dict
            Summary with keys: ticks, duration_s, stop_reason.
        """
        if cancel_check is None:
            cancel_check = lambda: False
        if termination_check is None:
            termination_check = lambda: False

        self._running = True
        self._tick_count = 0
        self._last_action = None
        stop_reason = "completed"
        start_time = time.monotonic()

        log.info(
            "ReactiveLoop started: skill=%s, freq=%.1fHz, dt=%.4fs",
            self._manifest.skill_id,
            self._manifest.control_frequency_hz,
            self._control_dt,
        )

        try:
            while self._running:
                tick_start = time.monotonic()

                # --- Check termination conditions ---
                if cancel_check():
                    stop_reason = "cancelled"
                    break

                if termination_check():
                    stop_reason = "termination_condition"
                    break

                if timeout_seconds > 0:
                    elapsed = tick_start - start_time
                    if elapsed >= timeout_seconds:
                        stop_reason = "timeout"
                        break

                if max_ticks > 0 and self._tick_count >= max_ticks:
                    stop_reason = "max_ticks"
                    break

                # --- Core tick ---
                try:
                    obs, action = self._tick(env)
                except Exception:
                    log.exception("ReactiveLoop tick error at tick %d", self._tick_count)
                    stop_reason = "tick_error"
                    break

                self._tick_count += 1

                # Callback
                if self.on_tick is not None:
                    try:
                        self.on_tick(self._tick_count, obs, action)
                    except Exception:
                        log.debug("on_tick callback error", exc_info=True)

                # --- Timing: sleep to maintain control frequency ---
                tick_elapsed = time.monotonic() - tick_start
                sleep_time = self._control_dt - tick_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self._running = False
            duration = time.monotonic() - start_time
            log.info(
                "ReactiveLoop stopped: reason=%s, ticks=%d, duration=%.2fs",
                stop_reason, self._tick_count, duration,
            )
            if self.on_stop is not None:
                try:
                    self.on_stop(stop_reason)
                except Exception:
                    pass

        return {
            "ticks": self._tick_count,
            "duration_s": time.monotonic() - start_time,
            "stop_reason": stop_reason,
        }

    def stop(self) -> None:
        """Signal the loop to stop after the current tick."""
        self._running = False

    def _resolve_decimation(self, env: Any) -> int:
        """Compute physics substeps per policy decision for *env*.

        Mirrors PolicyRunner.run_episode's decimation handling: one
        policy forward pass per outer tick, ``decimation`` PD refresh +
        ``mj_step`` substeps inside. Without this the inner physics
        loop runs at the policy rate (50 Hz) instead of the sim rate
        (200 Hz on stock IL Go2), which on its own caps the achievable
        PD bandwidth and reproduces the "soft / slippery legs" symptom
        even after Bug 1/2 are fixed.

        Resolution order:
          1. ``env.mj_model.opt.timestep`` (sim path) → ``round(control_dt / sim_dt)``
          2. Real-hardware adapters with no ``mj_model`` → ``1``

        Cached on the instance after the first successful resolve.
        """
        if self._decimation is not None:
            return self._decimation
        decimation = 1
        try:
            mj_model = getattr(env, "mj_model", None)
            if mj_model is not None:
                sim_dt = float(mj_model.opt.timestep)
                if sim_dt > 0:
                    decimation = max(1, int(round(self._control_dt / sim_dt)))
        except Exception:
            log.debug("ReactiveLoop: failed to resolve decimation; using 1", exc_info=True)
            decimation = 1
        self._decimation = decimation
        if decimation > 1:
            log.info(
                "ReactiveLoop: decimation=%d (control_dt=%.4fs, sim_dt=%.4fs)",
                decimation, self._control_dt, self._control_dt / decimation,
            )
        return decimation

    def _tick(self, env: Any) -> tuple:
        """Execute a single tick: observe → infer → (apply + step) × decimation.

        Mirrors the structure PolicyRunner.run_episode uses for IL
        bundles: one obs build + one policy forward pass per outer
        tick, then ``decimation`` physics substeps. Each substep
        re-invokes ``apply_with_pd`` so the PD controller sees the
        live ``mj_data.qpos/qvel`` instead of a frozen torque from the
        top of the tick. Frozen torques across substeps are the
        structural sim2sim mismatch that previously made Mission
        canvas IL replays "soft" — Isaac Lab's ImplicitActuator
        recomputes the joint drive every PhysX substep and we have to
        match that on the MuJoCo side.

        Real-hardware adapters (no ``mj_model``) collapse to
        ``decimation=1`` and behave exactly like the legacy single-shot
        path — there is no physics step to refresh against.

        Returns (obs, raw_action) tuple, where ``raw_action`` is the
        policy output BEFORE scale/PD/permutation (the order Isaac
        Lab's ``previous_action`` observation term expects).
        """
        # 1. Assemble observation (control rate, once per outer tick)
        obs = self._obs_assembler.assemble(
            env,
            command_bus=self._command_bus,
            last_action=self._last_action,
        )

        # 2. Policy inference (once per outer tick)
        action = self._inference.infer(obs)
        action = np.asarray(action, dtype=np.float32).flatten()

        # 3. Decimation loop: refresh PD torque each physics substep.
        decimation = self._resolve_decimation(env)
        has_sim_step = hasattr(env, "sim_step")
        use_pd = hasattr(self._action_applier, "apply_with_pd")

        for _ in range(decimation):
            if use_pd:
                ctrl = self._action_applier.apply_with_pd(action, env)
            else:
                ctrl = self._action_applier.apply(action, env)
            self._send_action(env, ctrl)
            if has_sim_step:
                try:
                    env.sim_step()
                except NotImplementedError:
                    # Real adapter — there is no inner physics step to
                    # iterate, so the loop degenerates to a single
                    # apply+send. Break to avoid pointlessly recomputing
                    # the same ctrl ``decimation`` times against an
                    # unchanging state.
                    break

        # ``last_action`` is the raw policy output in bundle order
        # (pre-scale, pre-PD) — what Isaac Lab feeds back as
        # ``previous_action``. Storing the ctrl-order torque here would
        # leak the wrong joint ordering into the next obs.
        self._last_action = action
        return obs, action

    @staticmethod
    def _send_action(env: Any, ctrl: np.ndarray) -> None:
        """Write control signal to the environment."""
        try:
            if hasattr(env, "mj_data") and env.mj_data is not None:
                nctrl = min(len(ctrl), env.mj_data.ctrl.shape[0])
                env.mj_data.ctrl[:nctrl] = ctrl[:nctrl]
        except Exception:
            log.debug("Failed to write ctrl to mj_data", exc_info=True)
