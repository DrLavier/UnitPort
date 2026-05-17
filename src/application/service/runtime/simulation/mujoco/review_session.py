"""MuJoCo review subprocess — Export 节点 ▶ Launch Review 的执行端.

DEMO 对应：``DEMO/bin/pages/training/training_workspace_window.py:_on_review_launch_requested``
经 ``_on_export_review_requested`` 走的 in-process 路径。

Architecture:
    Export 节点点击 ▶ Launch Review
        → CanvasScene.review_launch_requested.emit(payload)
        → MainWindow._on_review_launch_requested(payload)
        → MujocoReviewTask.submit() to TasksManager
        → MujocoReviewTask.run() (本文件) on SDK worker thread
            ├── MjActor.from_sku(sku)               (复用)
            ├── MjField.flat_ground()               (复用，Stage F-1 仅 flat)
            ├── MjSimEnv.build(actor, field)        (复用)
            ├── mujoco.viewer.launch_passive(model, data)
            ├── attach viewer to env._viewer / _viewer_sync
            ├── PolicyRunner.load_runner(bundle, env, robot_sku=sku)  (复用)
            ├── runner.run_episode(env, max_steps=N, render=True)
            └── post-episode: viewer kept alive until user closes window

Viewer lifecycle:
    The viewer window stays open across episodes. Each call to
    ``runner.run_episode`` invokes ``env.reset()`` at the start which
    drives ``MjSimEnv.reset`` (``mj_resetDataKeyframe('home')`` +
    bundle default-pose overlay), so every episode starts from a
    fresh standing pose. After an episode ends (terminated by fall,
    max_steps, or external interruption) we pause ~2s with
    ``viewer.sync()`` pumping so the user can inspect the final pose,
    then auto-restart the next episode. The outer loop exits only
    when the user closes the viewer window or Task.cancel() fires.

    User pressing MuJoCo viewer's Reset button (or Backspace) calls
    ``mj_resetData`` which writes ``qpos = mj_model.qpos0``. We patch
    ``qpos0`` to the home keyframe at ``MjSimEnv.build`` time, so any
    reset path lands on the same standing pose. Physics is advanced
    inside ``run_episode``; between episodes it is frozen (viewer
    sync only). Restart latency from "user clicks Reset" to "new
    episode begins" is bounded by the inter-episode pause (default 2s).

Cancellation policy:
    Stage F-1 不在 episode 内做 chunked cancel ——
    ``PolicyRunner.run_episode`` 把 ``step_telemetry_fn`` 异常 swallow
    （policy_runner.py:771-774），无法借它中断；自己重写 episode 循环
    会和 PolicyRunner 的 decimation/PD 不变量重复。所以：
      * 用户关 viewer 窗口 → ``viewer.is_running()`` → False，
        env.render() 内部 sync 会停 → 物理循环跑完后退出
      * env.is_terminated() → True (机器人摔倒) → episode 自然结束 →
        但 viewer 仍保持开启,等用户手动关窗
      * Task.cancel() 在 episode 之间 + post-episode 等待循环里生效
    若 Stage G 需要更激进的 cancel，把 episode 循环搬到本 Task 里
    自管理，绕过 PolicyRunner.run_episode 的封装层。

Headless degradation:
    缺 ``mujoco.viewer`` 模块（CI / WSL）时降级 log_warning 但不跑 episode
    —— "review" 的意义就是肉眼观察，没窗口跑空步意义不大。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from unitport_sdk import Task, log_info, log_warning

from application.service.robot_init_poses import InitPoseOverride


class MujocoReviewTask(Task):
    """SDK Task: launch a passive MuJoCo viewer + run trained policy episode.

    Cancellable only between episode runs (see module docstring). Returns
    a result dict ``{steps_run, viewer_used, bundle_path, sku, scene_id}``
    via ``task_finished`` signal.

    Default command policy:
        Review starts with ``command=None`` ⇒ ObsBuilder feeds the
        policy a zero ``velocity_command`` obs, so the robot stays
        still. Isaac Lab UnitreeFlatVelocity trains 5% of envs at zero
        command (``env.yaml rel_standing_envs=0.05``), so the policy
        has seen this distribution and the standing pose is in-domain.
        This matches user expectation that "Review = inspect spawn
        pose, no auto-walk".

        To drive a non-zero command, pass ``review_command=[vx, vy,
        vyaw]`` (callable path: canvas → ``review_launch_requested``
        payload includes a ``command`` field → MainWindow forwards as
        kwarg). Callers that want the bundle's training-time default
        instead can call :func:`_resolve_review_command(runner)` after
        load and pass the result here.
    """

    def __init__(
        self,
        bundle_path: Path,
        sku: str,
        scene_id: str,
        *,
        max_steps: int = 2000,
        review_command: Optional[Sequence[float]] = None,
        inter_episode_pause: float = 2.0,
        init_pose_override: Optional[InitPoseOverride] = None,
    ) -> None:
        super().__init__(f"Review {bundle_path.name} ({sku})")
        self._bundle_path = Path(bundle_path)
        self._sku = str(sku)
        self._scene_id = str(scene_id)
        self._max_steps = int(max_steps)
        self._review_command: Optional[List[float]] = (
            [float(v) for v in review_command]
            if review_command is not None else None
        )
        # Seconds to pause between episodes — gives the user time to see
        # the final pose before the next reset. ``0`` for back-to-back
        # restart. The viewer remains responsive during the pause.
        self._inter_episode_pause = max(0.0, float(inter_episode_pause))
        # UI-supplied init pose override (None = no override, use the
        # bundle's contract.init_base_pos + default_joint_pos).
        self._init_pose_override = init_pose_override

    def run(self) -> Dict[str, Any]:
        # Lazy imports — heavy (mujoco / onnxruntime / numpy) and
        # absent on headless CI.  We want the failure to surface as a
        # clean log line, not a Task-construction-time crash.
        from .mj_actor import MjActor
        from .mj_sim_env import MjSimEnv
        from application.service.runtime.policy.policy_runner import PolicyRunner

        self.check_cancelled()
        actor = MjActor.from_sku(self._sku)
        field = _resolve_field(self._scene_id)
        env = MjSimEnv.build(actor=actor, field=field)
        self.check_cancelled()

        # Passive viewer — defensive: missing mujoco.viewer (headless) →
        # warn + skip episode entirely (review without window is pointless).
        viewer = None
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.mj_model, env.mj_data)
            env._viewer = viewer
            env._viewer_sync = viewer.sync
            log_info(f"[review/mujoco] viewer launched for {self._sku}")
        except Exception as exc:
            log_warning(
                f"[review/mujoco] viewer unavailable — review aborted: {exc}"
            )
            return {
                "steps_run": 0,
                "viewer_used": False,
                "bundle_path": str(self._bundle_path),
                "sku": self._sku,
                "scene_id": self._scene_id,
                "aborted": "no viewer",
            }

        try:
            self.check_cancelled()
            runner = PolicyRunner.load_runner(
                self._bundle_path, env, robot_sku=self._sku,
            )

            # ── UI-supplied init pose override ────────────────────────
            # PolicyRunner.load() already pushed contract.init_base_pos
            # into env via set_il_init_base_pos. If the UI passed an
            # override, overwrite both base_pos and joint_pos here so
            # the first env.reset() inside run_episode picks them up.
            # IR role order comes from the deploy contract (normalized
            # to by-type for Isaac Lab bundles); fallback joint values
            # are the bundle's default_joint_pos so partial overrides
            # don't break the spawn.
            if self._init_pose_override is not None:
                try:
                    bundle = getattr(runner, "_bundle", None)
                    contract = (
                        getattr(bundle, "deploy_contract", None)
                        if bundle is not None
                        else None
                    )
                    if contract is not None:
                        ir_roles = list(getattr(contract, "joint_sdk_names", []) or [])
                        default_jp = list(
                            getattr(contract, "default_joint_pos", []) or []
                        )
                    else:
                        ir_roles = []
                        default_jp = []
                    qpos_bundle = self._init_pose_override.to_bundle_order(
                        bundle_ir_roles=ir_roles,
                        default_joint_pos=default_jp,
                    )
                    env.set_il_init_base_pos(self._init_pose_override.base_pos)
                    env.set_init_joint_pos_override(qpos_bundle)
                    log_info(
                        f"[review/mujoco] init_pose_override applied · "
                        f"base_pos={list(self._init_pose_override.base_pos)} · "
                        f"joints={len(qpos_bundle)} · ir_roles={len(ir_roles)}"
                    )
                except Exception as exc:
                    log_warning(
                        f"[review/mujoco] init_pose_override apply failed: "
                        f"{exc} (falling back to bundle defaults)"
                    )

            # ── Live controller input bridge ──────────────────────────
            # Pull (vx, vy, vyaw) from the process-singleton
            # GlobalInputManager each policy step. Gamepad source polls
            # in a 60Hz QThread; keyboard source fires on Qt key events
            # while the main window has focus. Both write to a
            # thread-safe CommandBus we read from this SDK worker thread
            # (CommandBus uses its own threading.Lock). When the manager
            # is unavailable / mode='off' the bus returns 0.0 for every
            # field — equivalent to "stand", which is what we want until
            # the user starts driving.
            _static_fallback = self._review_command

            def _live_command_provider() -> Optional[List[float]]:
                try:
                    from application.service.input.manager import (
                        get_global_input_manager,
                    )
                    mgr = get_global_input_manager()
                    if mgr is None:
                        return _static_fallback
                    live = mgr.get_live_values()
                    return [
                        float(live.get("vx", 0.0)),
                        float(live.get("vy", 0.0)),
                        float(live.get("vyaw", 0.0)),
                    ]
                except Exception:
                    return _static_fallback

            log_info(
                f"[review/mujoco] review loop starting · max_steps_per_episode="
                f"{self._max_steps} · inter_episode_pause={self._inter_episode_pause:.1f}s "
                f"· command_provider=GlobalInputManager "
                f"(static_fallback={_static_fallback if _static_fallback is not None else 'zero'})"
            )

            # ── Episode loop ─────────────────────────────────────────
            # Run episodes back-to-back. Each call to run_episode invokes
            # env.reset() at the start, which goes through MjSimEnv.reset
            # → mj_resetDataKeyframe('home') + bundle default_qpos
            # overlay, putting the robot in the standing pose for a
            # fresh start. After an episode ends (terminated by fall,
            # max_steps, or user pressing MuJoCo viewer's Reset which
            # triggers env.is_terminated indirectly via qpos jump), we
            # pause briefly so the user can inspect the final pose,
            # then loop back. Loop exits only when the user closes the
            # viewer window or the SDK Task is cancelled — matching the
            # user expectation that "Reset = restart episode".
            episode_idx = 0
            total_steps = 0
            last_term = False
            last_reason = ""
            while True:
                try:
                    self.check_cancelled()
                except Exception:
                    break
                if not viewer.is_running():
                    break
                episode_idx += 1
                log_info(
                    f"[review/mujoco] starting episode #{episode_idx}"
                )
                result = runner.run_episode(
                    env,
                    max_steps=self._max_steps,
                    render=True,
                    command=_static_fallback,
                    command_provider=_live_command_provider,
                )
                steps = int(getattr(result, "steps_run", 0))
                term = bool(getattr(result, "terminated", False))
                reason = str(getattr(result, "termination_reason", ""))
                msg = str(getattr(result, "message", "") or "")
                tail = f" · {msg}" if reason == "error" and msg else ""
                total_steps += steps
                last_term = term
                last_reason = reason
                log_info(
                    f"[review/mujoco] episode #{episode_idx} done · "
                    f"steps={steps} terminated={term} reason={reason}{tail} "
                    f"— auto-restart in {self._inter_episode_pause:.1f}s "
                    f"(close viewer to exit)"
                )
                if not viewer.is_running():
                    break
                # Brief inspection pause (viewer pumped at 60Hz, physics
                # frozen). Returns True if user closed the viewer during
                # the pause → exit the loop; False on timeout → restart.
                if self._wait_for_viewer_close(
                    viewer, max_seconds=self._inter_episode_pause
                ):
                    break

            return {
                "steps_run": total_steps,
                "episodes_run": episode_idx,
                "viewer_used": True,
                "bundle_path": str(self._bundle_path),
                "sku": self._sku,
                "scene_id": self._scene_id,
                "terminated": last_term,
                "termination_reason": last_reason,
            }
        finally:
            # By now the user has closed the window (or the Task was
            # cancelled); calling close() is a no-op on an already-closed
            # passive viewer but keeps the cleanup path symmetric.
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

    def _wait_for_viewer_close(
        self,
        viewer: Any,
        max_seconds: Optional[float] = None,
    ) -> bool:
        """Pump viewer sync at ~60 Hz until one of:

          * the user closes the viewer window — returns ``True``;
          * the SDK Task is cancelled — returns ``True``;
          * ``max_seconds`` elapsed (when provided) — returns ``False``.

        When ``max_seconds`` is ``None`` the wait is indefinite.
        Returning ``True`` is the caller's signal that the review loop
        should terminate; ``False`` means "timeout — keep looping".

        ``viewer.sync()`` must be called periodically so mouse rotate,
        scroll zoom, keyboard pan etc. on the passive viewer thread
        stay responsive while we're between episodes.
        """
        import time as _time

        period = 1.0 / 60.0
        deadline = (
            None if max_seconds is None else _time.perf_counter() + float(max_seconds)
        )
        while True:
            try:
                if not viewer.is_running():
                    return True
            except Exception:
                return True
            try:
                viewer.sync()
            except Exception:
                return True
            try:
                self.check_cancelled()
            except Exception:
                # Task.cancel() raised — outer try/finally closes the
                # viewer. Caller treats this the same as "viewer closed".
                return True
            if deadline is not None and _time.perf_counter() >= deadline:
                return False
            _time.sleep(period)


def _resolve_review_command(runner: Any) -> Optional[list]:
    """Read ``command_interface.fields[*].default`` from the loaded bundle.

    Returns a flat list ordered by ``obs_index`` (the policy-input layout)
    or ``None`` if the manifest has no command_interface.

    Opt-in helper — :class:`MujocoReviewTask` no longer calls this
    automatically. Pass the returned list as ``review_command=`` to the
    Task constructor if you want the bundle's training-time default
    command (e.g. ``vx=0.8`` forward walk). The default Review path
    feeds ``None`` so the robot stands still until the user provides
    explicit input.
    """
    bundle = getattr(runner, "_bundle", None)
    if bundle is None:
        return None
    raw = getattr(bundle, "raw_manifest", None) or {}
    ci = raw.get("command_interface") if isinstance(raw, dict) else None
    if not isinstance(ci, dict):
        return None
    fields = ci.get("fields")
    if not isinstance(fields, list) or not fields:
        return None
    indexed: list = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        try:
            idx = int(f.get("obs_index", -1))
            default = float(f.get("default", 0.0))
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        indexed.append((idx, default))
    if not indexed:
        return None
    indexed.sort(key=lambda pair: pair[0])
    # Caller passes the result to ``run_episode(command=...)`` which is
    # forwarded to ``ObsBuilder._get_command``; that consumer truncates /
    # pads to the term's declared dim, so we hand over the values in
    # obs-index order without trying to align to a frame here.
    return [val for _idx, val in indexed]


def _resolve_field(scene_id: str):
    """``scene_id`` → ``MjField`` instance.

    Stage F-1 only knows ``flat_ground``; everything else degrades to
    flat with a warning log. Stage G will add ``MjField.from_scene_registry(scene_id)``
    backed by ``application.training.scene_registry`` MJCF assets
    (rough_terrain / stairs / ...).
    """
    from .mj_field import MjField

    if scene_id and scene_id != "flat_ground":
        log_warning(
            f"[review/mujoco] scene_id={scene_id!r} not yet implemented — "
            "falling back to flat_ground"
        )
    return MjField.flat_ground()


__all__ = ["MujocoReviewTask"]
