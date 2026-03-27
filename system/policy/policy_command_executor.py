#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle 7 — PolicyCommandExecutor: brand-agnostic runtime bridge.

Translates a RUN_POLICY node payload + execution context into a PolicyRunner
invocation, returning a structured result dict the workflow runtime can surface.

No brand-specific imports here.  SimEnvContext is built by probing the
robot_model handle defensively.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from system.policy.policy_runner import PolicyRunner
from system.policy.sim_env_context import SimEnvContext

# CheckpointRegistry lives in system.service; importing it at module level creates a
# circular dependency because system.policy.__init__ re-exports policy types which
# are imported by system.service layers.  Defer to call time instead.
if TYPE_CHECKING:
    from system.service.checkpoint_registry import CheckpointRegistry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _failure(code: str, message: str, policy_id: str = "") -> Dict[str, Any]:
    """Return a structured failure dict."""
    return {
        "status": "failure",
        "result_kind": "run_policy",
        "policy_id": policy_id,
        "message": message,
        "error_code": code,
        "compat_status": "",
        "steps_run": 0,
        "terminated": False,
    }


def _build_sim_env_context(robot_model: Any, render: bool) -> SimEnvContext:
    """Build a SimEnvContext by probing ``robot_model`` defensively.

    Does not import brand-specific classes.  Tries common attribute names and
    wraps each MuJoCo call in a closure, falling back to no-op lambdas when
    the adapter does not expose them.

    Raises
    ------
    ValueError
        When required MuJoCo handles (mj_model / mj_data) are absent.
    """
    mj_model = (
        getattr(robot_model, "mj_model", None)
        or getattr(robot_model, "model", None)
    )
    mj_data = (
        getattr(robot_model, "mj_data", None)
        or getattr(robot_model, "data", None)
    )

    if mj_model is None or mj_data is None:
        raise ValueError(
            "robot_model is missing mj_model or mj_data; "
            "MuJoCo must be initialized before running a policy"
        )

    joint_names = (
        getattr(robot_model, "joint_names", None)
        or getattr(robot_model, "joint_name_list", None)
        or []
    )
    if not joint_names:
        try:
            import mujoco as _mj
            joint_names = []
            for actuator_id in range(int(getattr(mj_model, "nu", 0) or 0)):
                name = _mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                if name:
                    joint_names.append(str(name))
        except Exception:
            joint_names = []
    control_frequency_hz = float(
        getattr(robot_model, "control_frequency_hz", None) or 50.0
    )

    ctx = SimEnvContext(
        mj_model=mj_model,
        mj_data=mj_data,
        joint_names=list(joint_names),
        control_frequency_hz=control_frequency_hz,
        adapter=robot_model,
    )

    # Bind sim_step: prefer robot_model.sim_step; fall back to mujoco.mj_step.
    if callable(getattr(robot_model, "sim_step", None)):
        ctx.sim_step = robot_model.sim_step
    else:
        try:
            import mujoco as _mj  # deferred — avoids hard dep in headless tests
            _m, _d = mj_model, mj_data
            ctx.sim_step = lambda: _mj.mj_step(_m, _d)
        except ImportError:
            ctx.sim_step = lambda: None  # type: ignore[method-assign]

    # Bind reset
    if callable(getattr(robot_model, "reset", None)):
        ctx.reset = robot_model.reset
    else:
        try:
            import mujoco as _mj
            _m, _d = mj_model, mj_data
            ctx.reset = lambda: _mj.mj_resetData(_m, _d)
        except ImportError:
            ctx.reset = lambda: None  # type: ignore[method-assign]

    # Bind render. Prefer a dedicated robot_model.render() when available.
    # Otherwise fall back to ensure_viewer()+viewer.sync() so policy-driven
    # behavior runs can still open and refresh the passive MuJoCo viewer.
    if render and callable(getattr(robot_model, "render", None)):
        ctx.render = robot_model.render
    elif render:
        ensure_viewer = getattr(robot_model, "ensure_viewer", None)
        viewer_getter = lambda: getattr(robot_model, "viewer", None)

        def _render_via_viewer() -> None:
            if callable(ensure_viewer):
                ok = ensure_viewer()
                if ok is False:
                    return
            viewer = viewer_getter()
            if viewer is not None and callable(getattr(viewer, "sync", None)):
                viewer.sync()

        ctx.render = _render_via_viewer  # type: ignore[method-assign]
    else:
        ctx.render = lambda: None  # type: ignore[method-assign]

    # Bind is_terminated
    if callable(getattr(robot_model, "is_terminated", None)):
        ctx.is_terminated = robot_model.is_terminated
    else:
        ctx.is_terminated = lambda: False  # type: ignore[method-assign]

    return ctx


# ---------------------------------------------------------------------------
# PolicyCommandExecutor
# ---------------------------------------------------------------------------

class PolicyCommandExecutor:
    """Brand-agnostic runtime bridge for executing a RUN_POLICY workflow node.

    Reads params from the lowered node payload, resolves the policy bundle
    via CheckpointRegistry, builds SimEnvContext from the runtime robot_model,
    and invokes PolicyRunner.

    Circle 7 scope only — no Phase 2 telemetry, safety-per-step, or JIT.
    """

    def __init__(
        self,
        registry_factory: Optional[Callable[[], Any]] = None,
        runner_factory: Optional[Callable[[], PolicyRunner]] = None,
    ):
        """
        Parameters
        ----------
        registry_factory:
            Zero-arg callable that returns a CheckpointRegistry instance.
            Defaults to ``CheckpointRegistry`` (new instance each call).
            Override in tests to supply a pre-seeded registry.
        runner_factory:
            Zero-arg callable that returns a PolicyRunner instance.
            Defaults to ``PolicyRunner``.
            Override in tests to supply a mock runner.
        """
        # Deferred default: CheckpointRegistry import happens here to avoid the
        # system.service ↔ system.policy circular import at module load time.
        if registry_factory is None:
            from system.service.checkpoint_registry import CheckpointRegistry
            registry_factory = CheckpointRegistry
        self._registry_factory: Callable[[], Any] = registry_factory
        self._runner_factory: Callable[[], PolicyRunner] = (
            runner_factory if runner_factory is not None else PolicyRunner
        )

    def execute(
        self,
        node_payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a RUN_POLICY node and return a structured result dict.

        All predictable failure modes (missing params, registry miss,
        compatibility failure, episode failure) are translated into a
        failure result dict — no exceptions propagate out of this method.

        Parameters
        ----------
        node_payload:
            Flat dict with keys ``policy_id``, ``max_steps``,
            ``velocity_command``, ``render`` (as lowered by Circle 6).
        context:
            Runtime context dict.  ``context["robot_model"]`` must be present
            and carry ``mj_model`` / ``mj_data`` handles.

        Returns
        -------
        dict with keys:
          ``status``        — ``"success"`` | ``"failure"``
          ``result_kind``   — ``"run_policy"``
          ``policy_id``     — str
          ``message``       — human-readable summary
          ``error_code``    — str (empty on success)
          ``compat_status`` — CompatStatus value string (when available)
          ``steps_run``     — int
          ``terminated``    — bool
        """
        # 1. Validate required params
        policy_id = (node_payload.get("policy_id") or "").strip()
        if not policy_id:
            return _failure(
                "missing_policy_id",
                "run_policy: policy_id is required but was not provided",
            )

        try:
            max_steps = int(node_payload.get("max_steps") or 1000)
        except (ValueError, TypeError):
            max_steps = 1000

        velocity_command = node_payload.get("velocity_command") or [0.0, 0.0, 0.0]
        render = bool(node_payload.get("render", False))

        # 2. Resolve robot_model from context (mirrors existing executor pattern)
        robot_model = (
            context.get("robot_model")
            or (context.get("scenario") or {}).get("robot_model")
        )
        if robot_model is None:
            return _failure(
                "missing_robot_model",
                "run_policy: context['robot_model'] is absent; "
                "a robot model with MuJoCo handles is required",
                policy_id=policy_id,
            )

        # 3. Resolve bundle path through CheckpointRegistry (no raw path execution)
        try:
            registry = self._registry_factory()
            bundle_path: Path = registry.get_bundle_path(policy_id)
        except KeyError:
            return _failure(
                "unknown_policy_id",
                f"run_policy: policy_id {policy_id!r} not found in registry",
                policy_id=policy_id,
            )
        except Exception as exc:  # noqa: BLE001
            return _failure(
                "registry_error",
                f"run_policy: registry lookup failed: {exc}",
                policy_id=policy_id,
            )

        # 4. Build SimEnvContext by probing robot_model
        try:
            env = _build_sim_env_context(robot_model, render=render)
        except (ValueError, AttributeError, TypeError) as exc:
            return _failure(
                "mujoco_init_failed",
                f"run_policy: failed to build SimEnvContext: {exc}",
                policy_id=policy_id,
            )

        # 5. Load PolicyRunner (compat check)
        runner = self._runner_factory()
        try:
            from system.policy.policy_runner import IncompatibleWeightError
            runner.load(bundle_path, env)
        except IncompatibleWeightError as exc:
            return _failure(
                "compat_failed",
                f"run_policy: policy compatibility check failed: {exc}",
                policy_id=policy_id,
            )
        except Exception as exc:  # noqa: BLE001
            return _failure(
                "load_failed",
                f"run_policy: PolicyRunner.load() raised: {exc}",
                policy_id=policy_id,
            )

        # 6. Run episode
        try:
            episode = runner.run_episode(
                env,
                max_steps=max_steps,
                command=velocity_command,
                render=render,
            )
        except Exception as exc:  # noqa: BLE001
            return _failure(
                "episode_error",
                f"run_policy: unexpected error during episode: {exc}",
                policy_id=policy_id,
            )

        # 7. Translate EpisodeResult → result dict
        compat_val = (
            episode.compat_status.value
            if episode.compat_status is not None
            else ""
        )

        if not episode.success:
            return {
                "status": "failure",
                "result_kind": "run_policy",
                "policy_id": policy_id,
                "message": episode.message or "Policy episode did not succeed",
                "error_code": "episode_failed",
                "compat_status": compat_val,
                "steps_run": episode.steps_run,
                "terminated": episode.terminated,
            }

        return {
            "status": "success",
            "result_kind": "run_policy",
            "policy_id": policy_id,
            "message": f"Policy completed: {episode.steps_run} steps",
            "error_code": "",
            "compat_status": compat_val,
            "steps_run": episode.steps_run,
            "terminated": episode.terminated,
        }
