#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisCheckRunner — MuJoCo passive-viewer visualization for Training Ground.

run_vis_check_episodes(model, spec, vis_cfg, log_fn)
    Opens the MuJoCo passive viewer and runs *vis_episodes* episodes using the
    current trained model weights.  Blocks until all episodes finish or the
    user closes the viewer window.

    Called from a background daemon thread (spawned by TrainRunThread) so that
    the training thread can block on a threading.Event while the user watches.
    The MuJoCo passive viewer manages its own render thread internally, so
    calling this from any non-main thread is safe.

Dependencies: mujoco >= 3.0, gymnasium (via UnitreeGymEnv)
Heavy deps guarded by ImportError — vis check is silently skipped when mujoco
or gymnasium are unavailable.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from src.system.training.training_config import VisCheckConfig
    from src.system.training.training_spec import TrainingJobSpec

LogFn = Callable[[str], None]


def _resolve_bundle_lineage_spec(bundle_path, fallback_spec=None, log_fn: Optional[LogFn] = None):
    """Best-effort: reconstruct the source TrainingJobSpec from bundle lineage."""
    try:
        from src.system.core.utils.path_helper import get_project_root
        from src.system.training.training_spec import TrainingSpecCompiler
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Lineage spec loader unavailable; using current canvas spec: {exc}")
        return fallback_spec

    try:
        bundle_dir = Path(bundle_path)
        source_path = bundle_dir / "source.json"
        if not source_path.exists():
            return fallback_spec

        with source_path.open("r", encoding="utf-8") as fh:
            source = json.load(fh) or {}

        parent_policy_id = str(source.get("parent_policy_id", "") or "").strip()
        experiment_id = str(source.get("experiment_id", "") or "").strip()
        if not parent_policy_id or not experiment_id:
            return fallback_spec

        canvas_path = (
            get_project_root()
            / "training_workspaces"
            / parent_policy_id
            / "experiments"
            / f"{experiment_id}.canvas.json"
        )
        if not canvas_path.exists():
            if log_fn:
                log_fn(
                    f"[review] Lineage canvas not found for bundle '{bundle_dir.name}': "
                    f"{canvas_path}"
                )
            return fallback_spec

        with canvas_path.open("r", encoding="utf-8") as fh:
            graph = json.load(fh) or {}

        spec = TrainingSpecCompiler().compile(
            graph,
            policy_id=parent_policy_id,
            experiment_id=experiment_id,
        )
        if log_fn:
            log_fn(
                f"[review] Using bundle lineage spec: "
                f"policy={parent_policy_id} experiment={experiment_id}"
            )
        return spec
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Failed to reconstruct lineage spec; using current canvas spec: {exc}")
        return fallback_spec


def _build_replay_env(spec: "TrainingJobSpec", commands, *, use_domain_rand: bool = False):
    from src.system.training.sb3_trainer import get_obs_action_dims
    from src.system.training.training_spec import resolve_effective_task_terms
    from src.system.training.unitree_gym_env import (
        UnitreeGymEnv,
        _resolve_scene_xml,
        resolve_action_clip_range,
        resolve_obs_components,
    )

    robot = spec.robot_spec
    physics = spec.physics_config
    env_cfg = getattr(spec, "env_config", None)
    obs_cfg = getattr(spec, "obs_action_config", None)
    task_cfg = getattr(spec, "task_config", None)
    reward_terms, termination_conditions = resolve_effective_task_terms(spec)
    obs_dim, action_dim = get_obs_action_dims(spec)
    scene_xml = _resolve_scene_xml(
        getattr(spec, "scene_config", None),
        robot.mjcf_path or "",
        getattr(robot, "robot_type", ""),
    )
    gravity_z = getattr(getattr(spec, "scene_config", None), "gravity_z", -9.81)
    obs_components = resolve_obs_components(obs_cfg)
    frame_stack = int(getattr(obs_cfg, "frame_stack", 1) or 1)
    action_clip = resolve_action_clip_range(obs_cfg, env_cfg)
    action_scale = float(getattr(obs_cfg, "action_scale", 1.0) or 1.0)
    action_type = (
        getattr(physics, "action_type", None)
        or getattr(obs_cfg, "action_type", None)
        or "joint_position"
    )
    episode_max_steps = int(getattr(env_cfg, "time_limit_override", 0) or 0)
    if episode_max_steps <= 0:
        episode_max_steps = max(100, int(getattr(physics, "episode_max_steps", 100) or 100))

    return UnitreeGymEnv(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_episode_steps=episode_max_steps,
        mjcf_path=robot.mjcf_path or None,
        scene_xml_path=scene_xml,
        sim_dt=getattr(physics, "sim_dt", 0.002),
        control_dt=getattr(physics, "control_dt", 0.02),
        gravity_z=gravity_z,
        commands=commands,
        reward_terms=reward_terms,
        termination_conditions=termination_conditions,
        obs_components=obs_components,
        frame_stack=frame_stack,
        action_type=action_type,
        action_scale=action_scale,
        action_clip=action_clip,
        joint_config=getattr(robot, "joint_config", {}),
        domain_rand_config=(getattr(spec, "domain_rand_config", None) if use_domain_rand else None),
        reference_motion_config=getattr(spec, "reference_motion_config", None),
        init_pose_config=getattr(spec, "init_pose_config", None),
        command_mode=getattr(task_cfg, "command_mode", "fixed"),
        curriculum=getattr(task_cfg, "curriculum", False),
        success_threshold=getattr(task_cfg, "success_threshold", 0.8),
        truncation_max_steps=getattr(task_cfg, "truncation_max_steps", 0),
        curriculum_schedule=getattr(task_cfg, "curriculum_schedule", {}),
    )


def _bundle_command_defaults(bundle) -> np.ndarray:
    raw_manifest = getattr(bundle, "raw_manifest", {}) or {}
    runtime_cfg = raw_manifest.get("runtime") or {}
    defaults = runtime_cfg.get("command_defaults") or {}
    if not isinstance(defaults, dict):
        return np.zeros(3, dtype=np.float32)
    try:
        return np.array(
            [
                float(defaults.get("vx", 0.0) or 0.0),
                float(defaults.get("vy", 0.0) or 0.0),
                float(defaults.get("wz", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _resolve_export_review_command(bundle, spec, log_fn: Optional[LogFn] = None) -> np.ndarray:
    command = _bundle_command_defaults(bundle)
    if np.linalg.norm(command) > 1e-8:
        if log_fn:
            log_fn(f"[review] Using bundle command defaults: {tuple(command.astype(float).tolist())}")
        return command

    try:
        from src.system.training.unitree_gym_env import resolve_task_commands
        fallback = resolve_task_commands(getattr(spec, "task_config", None))
    except Exception:
        fallback = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    if log_fn:
        log_fn(
            f"[review] Bundle has no runtime.command_defaults; "
            f"falling back to task_config command {tuple(fallback.astype(float).tolist())}"
        )
    return fallback


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _draw_command_arrows(mujoco, viewer, env, log_fn: Optional[LogFn] = None) -> None:
    """Draw review-only arrows that visualize the effective command vector."""
    try:
        viewer.user_scn.ngeom = 0
        base = np.array(getattr(env._data, "qpos", [0.0, 0.0, 0.0])[:3], dtype=float)
        commands = np.array(getattr(env, "_commands", [0.0, 0.0, 0.0]), dtype=float)

        vx, vy, wz = float(commands[0]), float(commands[1]), float(commands[2])
        planar_norm = float(np.linalg.norm([vx, vy]))
        if planar_norm > 1e-6 and viewer.user_scn.ngeom < len(viewer.user_scn.geoms):
            start = base + np.array([0.0, 0.0, 0.18], dtype=float)
            scale = 0.6
            end = start + np.array([vx, vy, 0.0], dtype=float) * scale
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.array([0.25, 0.75, 1.0, 0.95], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.035,
                start.astype(np.float64),
                end.astype(np.float64),
            )
            viewer.user_scn.ngeom += 1

        if abs(wz) > 1e-6 and viewer.user_scn.ngeom < len(viewer.user_scn.geoms):
            start = base + np.array([0.0, 0.0, 0.34], dtype=float)
            end = start + np.array([0.0, np.sign(wz) * 0.28, 0.0], dtype=float)
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.array([0.45, 1.0, 0.45, 0.95], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.02 + min(abs(wz), 4.0) * 0.006,
                start.astype(np.float64),
                end.astype(np.float64),
            )
            viewer.user_scn.ngeom += 1
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"[review] Command-arrow overlay failed: {exc}")


def _try_activate_mujoco_window(log_fn: Optional[LogFn] = None) -> None:
    """
    Attempt to bring the MuJoCo viewer window to the foreground.
    Best-effort: silently skips if ctypes / win32 are unavailable or window
    cannot be found within a short timeout.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        found_hwnd = [None]

        def _enum_cb(hwnd, _lParam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if "mujoco" in title.lower() or "MuJoCo" in title:
                    found_hwnd[0] = hwnd
                    return False  # stop enumeration
            return True  # continue

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        cb = EnumWindowsProc(_enum_cb)

        # MuJoCo viewer window may take a moment to appear — retry briefly
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            user32.EnumWindows(cb, 0)
            if found_hwnd[0] is not None:
                hwnd = found_hwnd[0]
                SW_RESTORE = 9
                user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetForegroundWindow(hwnd)
                if log_fn:
                    log_fn("[vis] MuJoCo viewer window activated.")
                return
            time.sleep(0.1)
            found_hwnd[0] = None  # reset for next enum pass

        if log_fn:
            log_fn("[vis] Could not locate MuJoCo viewer window to activate (non-fatal).")
    except Exception:
        pass  # never crash training due to window-activation failure


def run_vis_check_episodes(
    model,
    spec: "TrainingJobSpec",
    vis_cfg: "VisCheckConfig",
    log_fn: Optional[LogFn] = None,
) -> None:
    """
    Open a MuJoCo passive viewer and run *vis_cfg.vis_episodes* episodes.

    Parameters
    ----------
    model:
        Trained SB3 model (must support ``model.predict(obs)``).
    spec:
        Compiled TrainingJobSpec supplying robot/physics/env config.
    vis_cfg:
        VisCheckConfig with vis_episodes and deterministic flag.
    log_fn:
        Optional callable for progress/warning messages.

    The function blocks until:
      - all *vis_cfg.vis_episodes* episodes have completed, OR
      - the user closes the viewer window.

    Silently returns (no exception raised) if mujoco / gymnasium are absent or
    if the viewer cannot be opened.
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        if log_fn:
            log_fn("[vis] mujoco not available — visualization skipped")
        return

    try:
        from src.system.training.unitree_gym_env import resolve_task_commands
    except ImportError as exc:
        if log_fn:
            log_fn(f"[vis] UnitreeGymEnv not available — visualization skipped: {exc}")
        return

    physics = spec.physics_config
    commands = resolve_task_commands(spec.task_config)
    n_episodes = max(1, vis_cfg.vis_episodes)
    deterministic = vis_cfg.deterministic
    target_dt = max(1.0 / 60.0, float(getattr(physics, "control_dt", 0.02) or 0.02))
    episode_pause_sec = 0.75

    if log_fn:
        log_fn(
            f"[vis] Opening MuJoCo viewer — {n_episodes} episode(s), "
            f"deterministic={deterministic}"
        )

    try:
        env = _build_replay_env(spec, commands, use_domain_rand=False)
    except Exception as exc:
        if log_fn:
            log_fn(f"[vis] Failed to build environment — visualization skipped: {exc}")
        return

    episodes_shown = 0
    try:
        obs, _ = env.reset()
        # launch_passive opens the viewer in its own render thread and returns
        # immediately; we drive the sim loop from here.
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            _try_activate_mujoco_window(log_fn)
            while episodes_shown < n_episodes and viewer.is_running():
                frame_start = time.perf_counter()
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(action)
                viewer.sync()

                elapsed = time.perf_counter() - frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

                if terminated or truncated:
                    episodes_shown += 1
                    if log_fn:
                        log_fn(
                            f"[vis] Episode {episodes_shown}/{n_episodes} finished "
                            f"({'terminated' if terminated else 'truncated'})"
                        )
                    pause_deadline = time.perf_counter() + episode_pause_sec
                    while viewer.is_running() and time.perf_counter() < pause_deadline:
                        viewer.sync()
                        time.sleep(1.0 / 60.0)
                    if episodes_shown < n_episodes:
                        obs, _ = env.reset()
                    # else: loop condition exits naturally on next iteration
    except Exception as exc:
        if log_fn:
            log_fn(f"[vis] Viewer error (non-fatal): {exc}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    if log_fn:
        log_fn(f"[vis] Visualization complete: {episodes_shown}/{n_episodes} episode(s) shown")


def _run_export_bundle_episode(
    bundle_path,
    spec: "TrainingJobSpec",
    command_scale: float = 1.0,
    hold_viewer_open: bool = False,
    log_fn: Optional[LogFn] = None,
    on_episode_complete: Optional[Callable] = None,
):
    import mujoco
    import mujoco.viewer

    from pathlib import Path
    from src.system.policy.bundle_loader import BundleLoader
    from src.system.policy.policy_runner import PolicyRunner
    from src.system.policy.sim_env_context import SimEnvContext
    from src.system.training.unitree_gym_env import UnitreeGymEnv

    bundle_path = Path(bundle_path)
    if log_fn:
        log_fn(f"[review] Loading bundle '{bundle_path.name}' from {bundle_path}")
    bundle = BundleLoader().load(bundle_path)
    replay_spec = _resolve_bundle_lineage_spec(bundle_path, fallback_spec=spec, log_fn=log_fn)
    physics = replay_spec.physics_config
    command = _resolve_export_review_command(bundle, replay_spec, log_fn=log_fn)
    try:
        scale = float(command_scale)
    except Exception:
        scale = 1.0
    command = (np.asarray(command, dtype=np.float32) * np.float32(scale)).astype(np.float32)
    env = _build_replay_env(replay_spec, command, use_domain_rand=False)
    sim_dt = float(getattr(physics, "sim_dt", 0.002) or 0.002)
    wall_step_dt = max(sim_dt, 1.0 / 240.0)

    try:
        if log_fn:
            log_fn(f"[review] Replay env ready: obs={getattr(bundle, 'obs_dim', '?')} action={getattr(bundle, 'action_dim', '?')} contract={((bundle.raw_manifest.get('observation_space') or {}).get('contract_preset', ''))}")
        env.reset(seed=getattr(replay_spec.algorithm_config, "seed", 42))
        runner = PolicyRunner()
        sim_env = SimEnvContext(
            mj_model=env._model,
            mj_data=env._data,
            joint_names=list(
                getattr(env, "joint_names", None)
                or getattr(env, "_joint_names", [])
                or []
            ),
            control_frequency_hz=float(
                getattr(physics, "control_dt", 0.02)
                and (1.0 / float(getattr(physics, "control_dt", 0.02)))
                or 50.0
            ),
            adapter=env,
        )
        def _paced_sim_step() -> None:
            mujoco.mj_step(env._model, env._data)
            time.sleep(wall_step_dt)

        sim_env.sim_step = _paced_sim_step  # type: ignore[method-assign]
        sim_env.reset = lambda: env.reset(seed=getattr(replay_spec.algorithm_config, "seed", 42))  # type: ignore[method-assign]
        sim_env.render = lambda: None  # type: ignore[method-assign]
        sim_env.is_terminated = lambda: bool(env._is_terminated())  # type: ignore[method-assign]
        if log_fn:
            log_fn("[review] Loading policy into PolicyRunner")
        runner.load(bundle_path, sim_env)
        command_list = command.astype(float).tolist()
        max_steps = int(((bundle.raw_manifest.get("runtime") or {}).get("max_steps", 0) or 0))
        if max_steps <= 0:
            max_steps = max(1, int(getattr(env, "_max_steps", 150)))

        if log_fn:
            log_fn(
                f"[review] Running exported bundle '{bundle_path.name}' "
                f"cmd={tuple(command_list)} steps={max_steps}"
            )

        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            _try_activate_mujoco_window(log_fn)
            sim_env.render = lambda: viewer.sync()  # type: ignore[method-assign]
            sim_env.is_terminated = lambda: (not viewer.is_running()) or bool(env._is_terminated())  # type: ignore[method-assign]
            if log_fn:
                log_fn("[review] Viewer opened; starting PolicyRunner.run_episode()")
            episode = runner.run_episode(
                sim_env,
                max_steps=max_steps,
                command=command_list,
                render=True,
            )
            if log_fn:
                log_fn(
                    f"[review] Episode finished: success={episode.success} steps={episode.steps_run} "
                    f"terminated={episode.terminated} reason={episode.termination_reason}"
                )
                if episode.termination_reason == "env_terminated":
                    try:
                        term_info = env.get_termination_debug_info()
                        log_fn(
                            "[review] Termination detail: "
                            f"reason={term_info.get('reason', '')} "
                            f"step={term_info.get('step_count', '')} "
                            f"base_height={term_info.get('base_height', '')} "
                            f"min_height={term_info.get('min_height', '')} "
                            f"roll={term_info.get('roll', '')} "
                            f"pitch={term_info.get('pitch', '')} "
                            f"contact_impulse={term_info.get('contact_impulse', '')}"
                        )
                    except Exception as exc:
                        log_fn(f"[review] Termination detail unavailable: {exc}")
            # Notify caller that episode result is ready (before the hold loop).
            # Used by _run_export_bundle_episode_nonblocking to decouple the
            # episode result from the viewer lifetime.
            if on_episode_complete is not None:
                try:
                    on_episode_complete(episode)
                except Exception:
                    pass

            if hold_viewer_open:
                if log_fn:
                    log_fn("[review] Holding viewer open with live physics until the window is closed")
                while viewer.is_running():
                    mujoco.mj_step(env._model, env._data)
                    viewer.sync()
                    time.sleep(wall_step_dt)
                if log_fn:
                    log_fn("[review] Viewer closed; replay returning to caller")
            return episode
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_export_bundle_episode_nonblocking(
    bundle_path,
    spec: "TrainingJobSpec",
    command_scale: float = 1.0,
    log_fn: Optional[LogFn] = None,
    episode_timeout_sec: float = 300.0,
):
    """
    Run a policy episode and return the result without waiting for the viewer
    to close.

    Spawns a daemon thread that calls ``_run_export_bundle_episode`` with
    ``hold_viewer_open=True``.  The calling thread blocks only until the
    episode itself finishes (via ``on_episode_complete`` callback), then
    returns the result immediately.  The daemon thread continues to run live
    physics in the passive viewer until the user closes the window.

    Used by ``BehaviorNode`` so that Mission execution completes and the UI
    unlocks while the MuJoCo window remains open for inspection.
    """
    import threading

    _result: list = [None]        # EpisodeResult or Exception
    _ready = threading.Event()

    def _on_episode_complete(episode):
        _result[0] = episode
        _ready.set()

    def _thread_fn():
        try:
            ep = _run_export_bundle_episode(
                bundle_path,
                spec=spec,
                command_scale=command_scale,
                hold_viewer_open=True,
                log_fn=log_fn,
                on_episode_complete=_on_episode_complete,
            )
            # on_episode_complete already set the event, but guard against
            # implementations that skip the callback path.
            if not _ready.is_set():
                _result[0] = ep
                _ready.set()
        except Exception as exc:
            _result[0] = exc
            _ready.set()

    t = threading.Thread(target=_thread_fn, daemon=True)
    t.start()

    # Wait only until episode result is ready, not until viewer closes.
    _ready.wait(timeout=episode_timeout_sec)

    result = _result[0]
    if result is None:
        raise TimeoutError(
            f"_run_export_bundle_episode_nonblocking: episode did not complete "
            f"within {episode_timeout_sec}s"
        )
    if isinstance(result, Exception):
        raise result
    return result


def run_export_bundle_review(
    bundle_path,
    spec: "TrainingJobSpec",
    log_fn: Optional[LogFn] = None,
) -> None:
    """
    Open a MuJoCo passive viewer and run one real episode through PolicyRunner
    using the exported runtime bundle.
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        if log_fn:
            log_fn("[review] mujoco not available - export review skipped")
        return

    try:
        from pathlib import Path
    except ImportError as exc:
        if log_fn:
            log_fn(f"[review] Export review dependencies unavailable: {exc}")
        return

    bundle_path = Path(bundle_path)
    try:
        _run_export_bundle_episode(
            bundle_path,
            spec,
            command_scale=1.0,
            hold_viewer_open=True,
            log_fn=log_fn,
        )
    except Exception as exc:
        if log_fn:
            msg = str(exc)
            log_fn(f"[review] Export bundle review failed: {msg}")
            if "incompatible" in msg.lower():
                log_fn(
                    "[review] HINT: The exported bundle was trained with a "
                    "different environment configuration than the current canvas. "
                    "If you changed the canvas after training (e.g. scene_config, "
                    "robot_type, obs_components), the bundle cannot be reviewed "
                    "against the new config. Re-train with the updated canvas."
                )


def run_environment_review(
    spec: "TrainingJobSpec",
    log_fn: Optional[LogFn] = None,
) -> None:
    """
    Open a MuJoCo passive viewer using the same environment init path as training.

    The environment is constructed and reset exactly once using the effective
    TrainingJobSpec, then held in the passive viewer until the user closes it.
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        if log_fn:
            log_fn("[review] mujoco not available - review skipped")
        return

    try:
        from src.system.training.unitree_gym_env import resolve_task_commands
    except ImportError as exc:
        if log_fn:
            log_fn(f"[review] Training env unavailable - review skipped: {exc}")
        return

    robot = spec.robot_spec
    physics = spec.physics_config
    task = spec.task_config
    commands = resolve_task_commands(task)

    try:
        env = _build_replay_env(spec, commands, use_domain_rand=False)
        env.reset(seed=getattr(spec.algorithm_config, "seed", 42))
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Failed to build review environment: {exc}")
        return

    preview_dt = max(1.0 / 60.0, float(getattr(physics, "control_dt", 0.02) or 0.02))
    preview_duration_sec = 3.0
    preview_pause_sec = 1.0
    initial_qpos = np.array(env._data.qpos, dtype=np.float64).copy()
    initial_qvel = np.array(env._data.qvel, dtype=np.float64).copy()

    if log_fn:
        gravity = tuple(float(v) for v in getattr(env._model.opt, "gravity", [0.0, 0.0, -9.81]))
        commands = tuple(float(v) for v in getattr(env, "_commands", [0.0, 0.0, 0.0]))
        log_fn(
            f"[review] robot={robot.robot_type} scene={getattr(spec.robot_spec, 'mjcf_path', '') or 'embedded'} "
            f"gravity={gravity} cmd={commands} task={task.task_type}/{task.command_mode}"
        )

    try:
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            _try_activate_mujoco_window(log_fn)
            loop_start = time.perf_counter()
            pause_until = 0.0
            while viewer.is_running():
                now = time.perf_counter()
                elapsed = now - loop_start

                if pause_until > now:
                    env._data.qpos[:] = initial_qpos
                    env._data.qvel[:] = 0.0
                    mujoco.mj_forward(env._model, env._data)
                    viewer.sync()
                    time.sleep(min(1.0 / 60.0, pause_until - now))
                    continue

                if elapsed >= preview_duration_sec:
                    env._data.qpos[:] = initial_qpos
                    env._data.qvel[:] = initial_qvel
                    mujoco.mj_forward(env._model, env._data)
                    pause_until = time.perf_counter() + preview_pause_sec
                    loop_start = pause_until
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
                    continue

                vx, vy, wz = (float(env._commands[0]), float(env._commands[1]), float(env._commands[2]))
                env._data.qpos[:] = initial_qpos
                env._data.qvel[:] = 0.0
                env._data.qpos[0] = initial_qpos[0] + vx * elapsed
                env._data.qpos[1] = initial_qpos[1] + vy * elapsed
                if env._model.nq >= 7:
                    env._data.qpos[3:7] = _quat_from_yaw(wz * elapsed)
                mujoco.mj_forward(env._model, env._data)
                viewer.sync()
                time.sleep(preview_dt)
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Viewer error (non-fatal): {exc}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    if log_fn:
        log_fn("[review] Viewer closed.")


def run_scene_config_preview(
    spec: "TrainingJobSpec",
    log_fn: Optional[LogFn] = None,
) -> None:
    """Open the current scene and preview free-fall under the configured gravity."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        if log_fn:
            log_fn("[preview] mujoco not available - preview skipped")
        return

    try:
        from src.system.training.sb3_trainer import get_obs_action_dims
        from src.system.training.unitree_gym_env import UnitreeGymEnv, _resolve_scene_xml
    except ImportError as exc:
        if log_fn:
            log_fn(f"[preview] Training env unavailable - preview skipped: {exc}")
        return

    robot = spec.robot_spec
    physics = spec.physics_config
    obs_dim, action_dim = get_obs_action_dims(spec)
    scene_xml = _resolve_scene_xml(
        getattr(spec, "scene_config", None),
        robot.mjcf_path or "",
        getattr(robot, "robot_type", ""),
    )
    gravity_z = getattr(getattr(spec, "scene_config", None), "gravity_z", -9.81)

    try:
        env = UnitreeGymEnv(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_episode_steps=max(100, physics.episode_max_steps),
            mjcf_path=robot.mjcf_path or None,
            scene_xml_path=scene_xml,
            sim_dt=getattr(physics, "sim_dt", 0.002),
            control_dt=getattr(physics, "control_dt", 0.02),
            gravity_z=gravity_z,
            commands=np.zeros(3, dtype=np.float32),
            reward_terms={},
            termination_conditions={},
        )
        env.reset(seed=getattr(spec.algorithm_config, "seed", 42))
    except Exception as exc:
        if log_fn:
            log_fn(f"[preview] Failed to build preview environment: {exc}")
        return

    preview_dt = max(1.0 / 60.0, float(getattr(physics, "control_dt", 0.02) or 0.02))
    preview_pause_sec = 1.0
    drop_height = 0.8
    initial_qpos = np.array(env._data.qpos, dtype=np.float64).copy()
    sim_dt = max(1e-4, float(getattr(env._model.opt, "timestep", 0.002) or 0.002))
    sim_steps_per_frame = max(1, int(round(preview_dt / sim_dt)))

    def _reset_drop_state() -> None:
        mujoco.mj_resetData(env._model, env._data)
        env._data.qpos[:] = initial_qpos
        env._data.qvel[:] = 0.0
        if env._model.nq >= 3:
            env._data.qpos[2] = initial_qpos[2] + drop_height
        mujoco.mj_forward(env._model, env._data)

    _reset_drop_state()

    if log_fn:
        gravity = tuple(float(v) for v in getattr(env._model.opt, "gravity", [0.0, 0.0, -9.81]))
        log_fn(
            f"[preview] robot={robot.robot_type} scene={scene_xml or robot.mjcf_path or 'embedded'} "
            f"gravity={gravity} sim_dt={sim_dt:.4f} frame_steps={sim_steps_per_frame} mode=free_fall"
        )

    try:
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            _try_activate_mujoco_window(log_fn)
            pause_until = 0.0
            while viewer.is_running():
                now = time.perf_counter()
                if pause_until > now:
                    viewer.sync()
                    time.sleep(min(1.0 / 60.0, pause_until - now))
                    continue

                for _ in range(sim_steps_per_frame):
                    mujoco.mj_step(env._model, env._data)
                viewer.sync()

                hit_ground = False
                try:
                    hit_ground = bool(env._data.ncon > 0)
                except Exception:
                    hit_ground = False
                if hit_ground or float(env._data.qpos[2]) <= float(initial_qpos[2]):
                    pause_until = time.perf_counter() + preview_pause_sec
                    _reset_drop_state()

                time.sleep(preview_dt)
    except Exception as exc:
        if log_fn:
            log_fn(f"[preview] Viewer error (non-fatal): {exc}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    if log_fn:
        log_fn("[preview] Viewer closed.")


def run_init_pose_preview(
    spec: "TrainingJobSpec",
    log_fn: Optional[LogFn] = None,
) -> None:
    """Open the MuJoCo viewer showing the robot frozen in the configured init pose.

    Gravity is zeroed so the pose is held static — the user can rotate the camera
    to inspect joint angles, base height and orientation without the robot falling.
    Close the viewer window to return.
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        if log_fn:
            log_fn("[init_pose] mujoco not available — preview skipped")
        return

    try:
        from src.system.training.sb3_trainer import get_obs_action_dims
        from src.system.training.unitree_gym_env import UnitreeGymEnv, _resolve_scene_xml
    except ImportError as exc:
        if log_fn:
            log_fn(f"[init_pose] Training env unavailable — preview skipped: {exc}")
        return

    robot = spec.robot_spec
    physics = spec.physics_config
    obs_dim, action_dim = get_obs_action_dims(spec)
    scene_xml = _resolve_scene_xml(
        getattr(spec, "scene_config", None),
        robot.mjcf_path or "",
        getattr(robot, "robot_type", ""),
    )
    gravity_z = getattr(getattr(spec, "scene_config", None), "gravity_z", -9.81)
    init_pose_config = getattr(spec, "init_pose_config", None)

    try:
        env = UnitreeGymEnv(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_episode_steps=max(100, physics.episode_max_steps),
            mjcf_path=robot.mjcf_path or None,
            scene_xml_path=scene_xml,
            sim_dt=getattr(physics, "sim_dt", 0.002),
            control_dt=getattr(physics, "control_dt", 0.02),
            gravity_z=gravity_z,
            commands=np.zeros(3, dtype=np.float32),
            reward_terms={},
            termination_conditions={},
            init_pose_config=init_pose_config,
        )
        env.reset(seed=getattr(spec.algorithm_config, "seed", 42))
    except Exception as exc:
        if log_fn:
            log_fn(f"[init_pose] Failed to build preview environment: {exc}")
        return

    # Zero gravity so the configured pose is held static in the viewer.
    saved_gravity = np.array(env._model.opt.gravity, dtype=np.float64).copy()
    env._model.opt.gravity[:] = 0.0
    mujoco.mj_forward(env._model, env._data)

    mode_str = str(getattr(init_pose_config, "mode", "default") or "default") if init_pose_config else "default"
    if log_fn:
        log_fn(
            f"[init_pose] mode={mode_str} robot={robot.robot_type} "
            f"scene={scene_xml or robot.mjcf_path or 'embedded'} — close window to exit"
        )

    try:
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            _try_activate_mujoco_window(log_fn)
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)
    except Exception as exc:
        if log_fn:
            log_fn(f"[init_pose] Viewer error (non-fatal): {exc}")
    finally:
        env._model.opt.gravity[:] = saved_gravity
        try:
            env.close()
        except Exception:
            pass

    if log_fn:
        log_fn("[init_pose] Viewer closed.")
