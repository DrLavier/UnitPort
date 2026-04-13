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

import os
import sys
import time
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from src.system.training.training_config import VisCheckConfig
    from src.system.training.training_spec import TrainingJobSpec

LogFn = Callable[[str], None]


# =====================================================================
# Public: run_policy_in_viewer — unified MuJoCo interactive review
# =====================================================================
# Single implementation shared by Mission (BehaviorNode) and Training
# (Export Review).  Viewer runs in a daemon thread; caller blocks
# until the episode completes, viewer stays open for inspection.

def run_policy_in_viewer(
    runner: Any,
    sim_env: Any,
    max_steps: int,
    command: list,
    *,
    command_provider: Optional[Callable[[], Any]] = None,
    step_telemetry_fn: Optional[Callable] = None,
    episode_timeout_sec: float = 600.0,
    log_fn: Optional[LogFn] = None,
) -> Any:
    """Run a PolicyRunner episode inside a MuJoCo passive viewer.

    Identical to BehaviorNode._run_episode_with_viewer but lives at
    module level so Training Review can call it without importing the
    Mission node graph.

    Key behaviours (matching Mission):
      - Viewer + episode run in a daemon thread
      - GLFW key events bridge into GlobalInputManager
      - Early termination disabled (viewer close = exit)
      - Auto-arm GlobalInputManager if not already active
      - Caller blocks until episode finishes; viewer stays open after

    Returns the EpisodeResult. Raises on timeout or episode crash.
    Falls back to headless run_episode when MuJoCo viewer is unavailable.
    """
    try:
        import mujoco
        import mujoco.viewer
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] mujoco.viewer unavailable ({exc}); running headless")
        return runner.run_episode(
            sim_env, max_steps=max_steps, command=command, render=False,
            step_telemetry_fn=step_telemetry_fn,
            command_provider=command_provider,
        )

    import threading
    import mujoco
    import mujoco.viewer

    # ── Auto-arm input ────────────────────────────────────────────
    _gim = None
    try:
        from src.system.runtime.global_input_manager import (
            get_global_input_manager,
        )
        _gim = get_global_input_manager()
        if not _gim.is_active():
            if log_fn:
                log_fn("[review] Input not active — auto-arming (gamepad → keyboard)")
            if not _gim.set_mode("gamepad"):
                _gim.set_mode("keyboard")
        if _gim.is_active() and command_provider is None:
            command_provider = _gim.make_command_provider()
            if log_fn:
                log_fn(f"[review] Live input ARMED ({_gim.mode})")
    except Exception:
        pass

    # ── GLFW → GlobalInputManager key bridge ──────────────────────
    _GLFW_TO_QT = {
        87: 0x57, 65: 0x41, 83: 0x53, 68: 0x44,   # WASD
        81: 0x51, 69: 0x45,                         # QE
        265: 0x01000013, 264: 0x01000015,           # Up/Down
        263: 0x01000012, 262: 0x01000014,           # Left/Right
        340: 0x01000020, 344: 0x01000020,           # Shift
        256: 0x01000000, 257: 0x01000004,           # Esc / Enter
    }
    _release_timers: dict = {}
    _RELEASE_DELAY = 0.20

    def _viewer_key_cb(key: int) -> None:
        if _gim is None:
            return
        qt_key = _GLFW_TO_QT.get(key, key)
        try:
            old = _release_timers.pop(qt_key, None)
            if old is not None:
                old.cancel()
            _gim.forward_key_press(qt_key)
            t = threading.Timer(
                _RELEASE_DELAY,
                lambda qk=qt_key: _safe_rel(qk),
            )
            t.daemon = True
            t.start()
            _release_timers[qt_key] = t
        except Exception:
            pass

    def _safe_rel(qt_key: int) -> None:
        _release_timers.pop(qt_key, None)
        if _gim is not None:
            try:
                _gim.forward_key_release(qt_key)
            except Exception:
                pass

    # ── Thread: viewer + episode + hold ───────────────────────────
    result_holder: list = [None]
    ready_evt = threading.Event()

    def _thread_fn() -> None:
        try:
            with mujoco.viewer.launch_passive(
                sim_env.mj_model, sim_env.mj_data,
                key_callback=_viewer_key_cb,
            ) as viewer:
                # Attach viewer for render() calls
                if hasattr(sim_env, "attach_viewer"):
                    sim_env.attach_viewer(viewer)
                else:
                    sim_env.render = lambda: viewer.sync()

                # Disable early termination — viewer close is the exit
                _orig_is_terminated = getattr(sim_env, "is_terminated", None)
                sim_env.is_terminated = lambda: not viewer.is_running()

                try:
                    episode = runner.run_episode(
                        sim_env,
                        max_steps=max_steps,
                        command=command,
                        render=True,
                        step_telemetry_fn=step_telemetry_fn,
                        command_provider=command_provider,
                    )
                    result_holder[0] = episode
                except Exception as exc:
                    result_holder[0] = exc
                finally:
                    if _orig_is_terminated is not None:
                        sim_env.is_terminated = _orig_is_terminated
                    ready_evt.set()

                # Hold viewer open with live physics
                try:
                    while viewer.is_running():
                        mujoco.mj_step(sim_env.mj_model, sim_env.mj_data)
                        viewer.sync()
                        time.sleep(0.005)
                except Exception:
                    pass
        except Exception as exc:
            if result_holder[0] is None:
                result_holder[0] = exc
            ready_evt.set()

    t = threading.Thread(target=_thread_fn, daemon=True)
    t.start()
    ready_evt.wait(timeout=episode_timeout_sec)

    result = result_holder[0]
    if result is None:
        raise TimeoutError(
            f"run_policy_in_viewer: episode did not complete "
            f"within {episode_timeout_sec}s"
        )
    if isinstance(result, Exception):
        raise result
    return result


def _synthesize_default_replay_spec(bundle_path, log_fn: Optional[LogFn] = None):
    """Build a minimal TrainingJobSpec good enough for ``_build_replay_env``.

    Used as the last-resort fallback when neither bundle lineage nor a
    caller-supplied spec is available — most notably for Isaac Lab bundles
    (whose ``source.json`` only records ``{type, src, env_yaml}`` and has
    no parent_policy_id) and for ``BehaviorNode`` replays in the Mission
    canvas (which call ``_run_export_bundle_episode_nonblocking(spec=None)``
    because they have no training canvas at hand).

    Strategy:
      - Read the bundle's ``manifest.yaml`` to discover the trained robot
        family / model — that's the only piece UnitreeGymEnv truly cannot
        guess (it picks the MJCF from ``robot_type``).
      - Construct a default ``TrainingJobSpec`` and fill ``robot_spec``
        only. Every other config sub-object keeps its dataclass default
        and ``_build_replay_env``'s ``getattr(..., default)`` shims will
        fall through to the same defaults the SB3 fresh-canvas template
        uses.

    Returns ``None`` only if the manifest can't be read at all — callers
    should treat that as an unrecoverable failure.
    """
    try:
        import yaml
        from src.system.training.training_spec import TrainingJobSpec
        from src.system.training.training_config import RobotSpec
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Synthesizer unavailable: {exc}")
        return None

    bundle_dir = Path(bundle_path)
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.exists():
        if log_fn:
            log_fn(f"[review] Cannot synthesize spec: {manifest_path} missing")
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Manifest parse failed: {exc}")
        return None

    # SkillManifest v2 lives under the "skill" key; v1 is flat at the root.
    skill = raw.get("skill") if isinstance(raw, dict) else None
    if not isinstance(skill, dict):
        skill = raw if isinstance(raw, dict) else {}

    # robot_type: prefer the first concrete model, fall back to family,
    # finally to "go2" (only IL preset right now). Normalise the name to
    # the bare keys ``_resolve_registered_scene_xml`` recognises ("go2",
    # "h1", "spot", …) — the IL manifest parser tends to write descriptive
    # names like "unitree_go2" / "boston_dynamics_spot" that the SB3
    # scene resolver doesn't accept directly.
    def _normalize_robot_type(name: str) -> str:
        n = str(name or "").strip().lower()
        if not n:
            return ""
        # Strip known brand prefixes once.
        for prefix in ("unitree_", "boston_dynamics_", "bostondynamics_", "xiaomi_"):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break
        # Some manifests carry hyphenated variants ("go-2"); collapse to
        # the registry's underscore-free key.
        return n.replace("-", "")

    robot_models = skill.get("target_robot_models") or []
    robot_type = ""
    if isinstance(robot_models, list) and robot_models:
        robot_type = _normalize_robot_type(robot_models[0])
    if not robot_type:
        # Family is usually a category like "quadruped" — not a usable
        # MJCF key on its own, but kept as a low-priority fallback.
        robot_type = _normalize_robot_type(skill.get("target_robot_family"))
    if not robot_type:
        robot_type = "go2"

    spec = TrainingJobSpec(policy_id=str(skill.get("skill_id", "") or bundle_dir.name))
    spec.robot_spec = RobotSpec(robot_type=robot_type, mjcf_path="")

    if log_fn:
        log_fn(
            f"[review] Synthesized default replay spec for bundle "
            f"'{bundle_dir.name}' (robot_type={robot_type})"
        )
    return spec


def _resolve_bundle_lineage_spec(bundle_path, fallback_spec=None, log_fn: Optional[LogFn] = None):
    """Best-effort: reconstruct the source TrainingJobSpec from bundle lineage.

    Resolution order:
      1. ``source.json`` parent_policy_id/experiment_id → recompile original
         canvas (works for SB3 bundles UnitPort produced).
      2. The caller's ``fallback_spec`` (the live canvas spec — only valid
         when the user is reviewing from the same backend that produced
         the bundle).
      3. ``_synthesize_default_replay_spec`` — reads the bundle's own
         ``manifest.yaml`` to extract robot_type, fills the rest with
         dataclass defaults. This is what makes Isaac Lab bundles and
         BehaviorNode replays work without lineage metadata.
    """
    try:
        from src.system.core.utils.path_helper import get_project_root
        from src.system.training.training_spec import TrainingSpecCompiler
    except Exception as exc:
        if log_fn:
            log_fn(f"[review] Lineage spec loader unavailable; using current canvas spec: {exc}")
        return fallback_spec or _synthesize_default_replay_spec(bundle_path, log_fn=log_fn)

    try:
        bundle_dir = Path(bundle_path)
        source_path = bundle_dir / "source.json"
        if not source_path.exists():
            return fallback_spec or _synthesize_default_replay_spec(bundle_path, log_fn=log_fn)

        with source_path.open("r", encoding="utf-8") as fh:
            source = json.load(fh) or {}

        parent_policy_id = str(source.get("parent_policy_id", "") or "").strip()
        experiment_id = str(source.get("experiment_id", "") or "").strip()
        project_id = str(source.get("project_id", "") or "").strip()
        if not parent_policy_id or not experiment_id:
            # No lineage metadata (typical for Isaac Lab bundles whose
            # source.json only carries {type, src, env_yaml}).
            return fallback_spec or _synthesize_default_replay_spec(bundle_path, log_fn=log_fn)

        graph = None

        # Try loading from ProjectStore first (new project workspace layout)
        if project_id and experiment_id:
            try:
                from src.system.core.project_store import ProjectStore
                store = ProjectStore()
                graph = store.load_experiment(project_id, experiment_id)
            except Exception:
                pass

        # Fallback: resolve via TrainingWorkspaceStore (legacy compat wrapper).
        # Historically a workspace was named after its parent policy, so the
        # parent_policy_id doubles as a workspace_name for ProjectMeta.name lookup.
        if graph is None:
            try:
                from src.system.training.training_workspace_store import TrainingWorkspaceStore
                ws_store = TrainingWorkspaceStore()
                pid = ws_store._resolve_project_id(parent_policy_id)
                if pid:
                    from src.system.core.project_store import ProjectStore
                    store = ProjectStore()
                    graph = store.load_experiment(pid, experiment_id)
            except Exception:
                pass

        # Final fallback: legacy training_workspaces path
        if graph is None:
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
                return fallback_spec or _synthesize_default_replay_spec(bundle_path, log_fn=log_fn)
            with canvas_path.open("r", encoding="utf-8") as fh:
                graph = json.load(fh) or {}

        # Experiments are persisted as a multi-backend envelope:
        #   {"schema": "training_canvas_v1",
        #    "active_backend": "<name>",
        #    "layers": {"<backend>": <canvas_with_nodes>}}
        # Unwrap so the compiler receives the flat canvas it expects. Older
        # experiments saved before the envelope existed are passed through
        # untouched.
        if (
            isinstance(graph, dict)
            and "layers" in graph
            and isinstance(graph.get("layers"), dict)
        ):
            active_backend = str(graph.get("active_backend", "") or "").strip()
            layers = graph["layers"]
            inner = layers.get(active_backend) if active_backend else None
            if not isinstance(inner, dict) or not inner.get("nodes"):
                # Active backend is empty — pick any non-empty layer.
                for candidate in layers.values():
                    if isinstance(candidate, dict) and candidate.get("nodes"):
                        inner = candidate
                        break
            if isinstance(inner, dict):
                graph = inner

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
            # Be explicit that the failure is in the SAVED lineage snapshot,
            # not the user's current canvas — the message used to be ambiguous
            # and made users think their live canvas was broken.
            log_fn(
                f"[review] Saved lineage snapshot for this bundle could not be "
                f"recompiled ({exc}); falling back to current canvas spec."
            )
        return fallback_spec or _synthesize_default_replay_spec(bundle_path, log_fn=log_fn)


def _resolve_replay_domain_rand(spec: "TrainingJobSpec", *, use_domain_rand: bool):
    """Decide what DomainRandConfig the MuJoCo replay env should see.

    Sim2sim parity rule: obs_noise_std must always match training, otherwise
    the policy is asked to act on a *cleaner* observation distribution than it
    was trained on and behaves erratically. Physics-side randomization
    (mass/friction/motor/push) is gated separately because per-reset variation
    breaks reproducibility for visualization runs.

    - use_domain_rand=True  -> pass the full DomainRandConfig from the spec
    - use_domain_rand=False -> return a slim copy with ``enabled=False``
                               (skips physics rand at unitree_gym_env L1371)
                               but keeps ``obs_noise_std`` so the obs adder at
                               L1074 still fires.
    """
    src_cfg = getattr(spec, "domain_rand_config", None)
    if src_cfg is None:
        return None
    if use_domain_rand:
        return src_cfg
    try:
        from src.system.training.training_config import DomainRandConfig
        return DomainRandConfig(
            enabled=False,
            obs_noise_std=float(getattr(src_cfg, "obs_noise_std", 0.0) or 0.0),
        )
    except Exception:
        # Fallback: a duck-typed shim with just the two attributes the env reads.
        class _ObsNoiseOnly:
            enabled = False
            obs_noise_std = float(getattr(src_cfg, "obs_noise_std", 0.0) or 0.0)
        return _ObsNoiseOnly()


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
        domain_rand_config=_resolve_replay_domain_rand(spec, use_domain_rand=use_domain_rand),
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


def _neutralize_replay_spec_for_sim2real(spec: "TrainingJobSpec") -> None:
    """Strip the training scene/curriculum off *spec* in-place so the bundle
    runs on a generic flat MuJoCo testbed.

    The Mission canvas wants Behavior nodes to behave like a sim2real harness:
    a neutral environment that exercises the *policy* without dragging the
    training scene's terrain, domain randomisation, init pose, reference
    motion or curriculum logic along with it. The robot model, physics, and
    obs/action contract are intentionally preserved — those define the policy
    I/O surface and would break loading otherwise.

    Mutates the spec in-place; the caller is expected to pass a throwaway
    copy (or accept that subsequent reads see the neutralised values).
    """
    # Force flat scene auto-detect (_resolve_scene_xml() defaults to the
    # robot's registered "flat" scene when scene_config is None).
    try:
        spec.scene_config = None
    except Exception:
        pass

    # Drop everything that recreates the training environment shape.
    for _attr in (
        "domain_rand_config",
        "reference_motion_config",
        "init_pose_config",
    ):
        try:
            setattr(spec, _attr, None)
        except Exception:
            pass

    # Disable training curriculum / time-limit overrides on the task. The
    # bundle's own runtime.max_steps still bounds the episode length.
    task_cfg = getattr(spec, "task_config", None)
    if task_cfg is not None:
        for _attr, _default in (
            ("curriculum", False),
            ("curriculum_schedule", {}),
            ("truncation_max_steps", 0),
        ):
            try:
                setattr(task_cfg, _attr, _default)
            except Exception:
                pass

    env_cfg = getattr(spec, "env_config", None)
    if env_cfg is not None:
        try:
            env_cfg.time_limit_override = 0
        except Exception:
            pass


def _run_export_bundle_episode(
    bundle_path,
    spec: "TrainingJobSpec",
    command_scale: float = 1.0,
    hold_viewer_open: bool = False,
    log_fn: Optional[LogFn] = None,
    on_episode_complete: Optional[Callable] = None,
    neutral_replay: bool = False,
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
    if neutral_replay and replay_spec is not None:
        # Mission-side BehaviorNode path: replace the bundle's training
        # scene/curriculum with a neutral flat MuJoCo testbed (sim2real-style).
        # This deliberately ignores the bundle's lineage scene_config and any
        # Isaac-Sim-only assumptions (height_scan terrain, etc.) so the same
        # bundle can be evaluated on a clean default surface.
        _neutralize_replay_spec_for_sim2real(replay_spec)
        if log_fn:
            log_fn(
                "[review] Sim2real mode: replaying on neutral flat MuJoCo "
                "testbed (training scene + curriculum stripped)."
            )
    if replay_spec is None:
        # Both lineage AND the synthesizer returned nothing — usually means
        # the bundle's manifest.yaml is missing or unreadable. Surface a
        # specific error instead of crashing on ``None.physics_config``.
        raise RuntimeError(
            f"Cannot review bundle '{bundle_path.name}': no source.json "
            f"lineage, no caller-provided spec, and the bundle's "
            f"manifest.yaml could not be parsed. The bundle is corrupt or "
            f"was produced by an unsupported exporter."
        )
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

        # Delegate to the unified run_policy_in_viewer — same daemon-
        # thread + GLFW key bridge + auto-arm + no-early-termination
        # pipeline that Mission's BehaviorNode uses.
        episode = run_policy_in_viewer(
            runner,
            sim_env,
            max_steps=max(max_steps, 100_000),
            command=command_list,
            log_fn=log_fn,
        )

        if on_episode_complete is not None:
            try:
                on_episode_complete(episode)
            except Exception:
                pass
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
    neutral_replay: bool = False,
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
                neutral_replay=neutral_replay,
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
    # The intermediate "MuJoCo couldn't replay this bundle" state is NOT an
    # error condition — it just means the lightweight backend can't host
    # this particular bundle (e.g. Isaac height_scan obs) and we should try
    # the heavier Isaac Sim path. Use neutral wording (no "failed"/"error"
    # keywords) so the panel's keyword-sniffing log router doesn't escalate
    # this transitional state to ERROR. Only the final "both backends gave
    # up" state below is logged via log_error.
    try:
        _run_export_bundle_episode(
            bundle_path,
            spec,
            command_scale=1.0,
            hold_viewer_open=True,
            log_fn=log_fn,
        )
        return
    except Exception as exc:
        mujoco_err = exc
        if log_fn:
            log_fn(
                f"[review] MuJoCo cannot host this bundle "
                f"(reason: {_summarize_for_warning(exc)}); "
                f"attempting Isaac Sim viewport fallback."
            )

    # ── Fallback: Isaac Sim viewport ────────────────────────────────────
    # MuJoCo is the lightweight default, but Isaac Lab bundles trained with
    # height_scan / Isaac-only sensors will always be incompatible with the
    # MuJoCo replay env. When that happens, transparently retry through
    # play.py in Isaac Sim where the env matches the training config. If
    # Isaac Sim is also unavailable / breaks, THEN log_error.
    isaac_err: Optional[Exception] = None
    try:
        launched = _try_isaac_sim_review_fallback(bundle_path, log_fn=log_fn)
        if launched:
            return
        # Not an Isaac bundle, or no usable checkpoint dir → not a fallback
        # candidate. Drop into the final log_error block below with the
        # original MuJoCo cause.
    except Exception as exc:
        isaac_err = exc
        if log_fn:
            log_fn(
                f"[review] Isaac Sim fallback could not start "
                f"(reason: {_summarize_for_warning(exc)})."
            )

    # Final state: both backends are out. THIS is the only place we
    # legitimately log_error.
    try:
        from src.system.core.logger import log_error
        if isaac_err is not None:
            log_error(
                f"[review] Bundle review failed in both backends. "
                f"MuJoCo: {mujoco_err}. Isaac Sim: {isaac_err}."
            )
        else:
            log_error(f"[review] Bundle review failed: {mujoco_err}")
    except Exception:
        pass

    if log_fn and "incompatible" in str(mujoco_err).lower():
        log_fn(
            "[review] HINT: This bundle was trained with an environment "
            "that the lightweight MuJoCo replay cannot reproduce "
            "(e.g. height_scan / terrain perception). Isaac Sim viewport "
            "is the canonical reviewer for such bundles — register Isaac "
            "Lab via the setup wizard to enable the fallback."
        )


def _summarize_for_warning(exc: Exception) -> str:
    """Compress an exception message into a short, keyword-safe blurb.

    The training panel's log router (``_append_review_log``) routes any
    message containing "failed" / "error" / "skipped" / "unavailable" to
    the error sink. We deliberately scrub those words from intermediate
    fallback warnings so the user doesn't see a red ERROR line for a
    transitional state that we are about to recover from.
    """
    msg = str(exc).strip()
    # Heuristic: if the wrapper raised a CompatibilityChecker incompatibility,
    # boil it down to the failing check codes (much less noisy than the full
    # "Bundle '...' is incompatible with the current environment. Failing
    # checks: [...]" sentence).
    lower = msg.lower()
    if "incompatible" in lower and "failing checks" in lower:
        # Extract just the bracketed check list, e.g. "['obs_dim_mismatch']"
        try:
            start = msg.index("[")
            end = msg.index("]", start) + 1
            return f"obs/action contract mismatch {msg[start:end]}"
        except ValueError:
            return "obs/action contract mismatch"
    if len(msg) > 160:
        msg = msg[:157] + "..."
    # Scrub trigger words so the panel router doesn't escalate the warning.
    for trigger in ("failed", "Failed", "FAILED",
                    "error",  "Error",  "ERROR",
                    "skipped","Skipped",
                    "unavailable", "Unavailable"):
        msg = msg.replace(trigger, "issue")
    return msg


def _try_isaac_sim_review_fallback(
    bundle_path,
    log_fn: Optional[LogFn] = None,
) -> bool:
    """Attempt to replay an Isaac-Lab-trained bundle in the Isaac Sim viewport.

    Returns ``True`` when the Isaac Sim subprocess was successfully spawned
    (the user closes the window manually). Returns ``False`` when the bundle
    is not an Isaac Lab bundle, or when the original training run directory
    cannot be located. Raises on infrastructure failures (Isaac Lab not
    installed, subprocess spawn errors, …) so the caller can distinguish a
    skipped fallback from a failed fallback.
    """
    from pathlib import Path

    bundle_path = Path(bundle_path)
    source_file = bundle_path / "source.json"
    if not source_file.exists():
        return False

    try:
        source_meta = json.loads(source_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return False

    if str(source_meta.get("type", "")).lower() != "isaac_lab":
        return False  # SB3 / HuggingFace / local bundle — fallback doesn't apply

    # ``src`` is the absolute path to the original Isaac Lab run/exported dir
    # at import time (see CheckpointRegistry.import_isaac_lab_bundle).  RSL-RL's
    # play.py wants the run directory containing model_*.pt — try ``src`` first
    # and walk one level up if needed.
    src_path_str = str(source_meta.get("src", "") or "").strip()
    if not src_path_str:
        if log_fn:
            log_fn("[review] Isaac fallback: source.json has no 'src' field")
        return False

    candidate = Path(src_path_str)
    if not candidate.exists():
        if log_fn:
            log_fn(
                f"[review] Isaac fallback: original run directory no longer "
                f"exists at {candidate} (was the training run deleted?)"
            )
        return False

    checkpoint_dir = candidate
    has_checkpoints = (
        any(checkpoint_dir.glob("model_*.pt"))
        or (checkpoint_dir / "checkpoints").is_dir()
    )
    if not has_checkpoints and checkpoint_dir.parent.exists():
        # ``src`` may have pointed at an ``exported/`` subdir; the actual run
        # dir is its parent.
        parent = checkpoint_dir.parent
        if any(parent.glob("model_*.pt")) or (parent / "checkpoints").is_dir():
            checkpoint_dir = parent
            has_checkpoints = True

    if not has_checkpoints:
        if log_fn:
            log_fn(
                f"[review] Isaac fallback: no model_*.pt found near "
                f"{candidate}; cannot reconstruct --load_run for play.py"
            )
        return False

    # Detect Isaac Lab installation
    from src.system.training.isaac_lab_backend import (
        IsaacLabBackend,
        detect_isaac_lab,
    )
    from src.system.training.isaac_lab_config import IsaacLabConfig

    detection = detect_isaac_lab()
    if detection is None:
        if log_fn:
            log_fn(
                "[review] Isaac fallback: Isaac Lab installation not "
                "registered. Run the setup wizard or set ISAAC_LAB_PATH."
            )
        return False

    # Recover the original task name. Resolution order:
    #   1. source.json.task_name           (written at import time, post-fix)
    #   2. manifest.yaml task_name fields  (rarely populated; defensive)
    #   3. policy_id name + manifest robot type heuristic
    #   4. hardcoded default (Flat Go2)
    task_name, task_source = _resolve_isaac_review_task_name(
        bundle_path, source_meta, log_fn=log_fn
    )

    config = IsaacLabConfig(
        task_name=task_name,
        isaac_lab_path=detection.get("path", ""),
        isaac_lab_python=detection.get("python", ""),
        isaac_lab_launcher=detection.get("launcher", ""),
    )
    backend = IsaacLabBackend(config)

    if log_fn:
        log_fn(
            f"[review] Falling back to Isaac Sim viewport "
            f"(task={task_name} [{task_source}], run={checkpoint_dir})"
        )

    # Capture stdout/stderr to a temp log so the user can diagnose silent
    # exits ("Isaac Sim launched then closed without showing anything"
    # is otherwise impossible to debug). We override the backend's detached
    # Popen with a logged variant.
    import subprocess
    import tempfile
    cmd = config.build_review_command(str(checkpoint_dir))
    log_handle = tempfile.NamedTemporaryFile(
        prefix="unitport_isaac_review_",
        suffix=".log",
        delete=False,
        mode="w",
        encoding="utf-8",
    )
    log_path = log_handle.name
    log_handle.close()
    if log_fn:
        log_fn(f"[review] Isaac Sim subprocess log: {log_path}")
    try:
        log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        env_vars = dict(os.environ)
        env_vars["PYTHONUNBUFFERED"] = "1"
        env_vars.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        subprocess.Popen(
            cmd,
            cwd=config.isaac_lab_path or None,
            env=env_vars,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        # If anything in our logged-spawn path breaks, fall back to the
        # backend's original detached spawn so the user still gets *some*
        # attempt at the review.
        backend.launch_review(str(checkpoint_dir))

    if log_fn:
        log_fn(
            "[review] Isaac Sim viewport launching (detached). If the "
            "window closes immediately, tail the subprocess log above."
        )
    return True


# A small whitelist of built-in RSL-RL tasks UnitPort knows how to map onto.
# Used by the heuristic fallback below when source.json doesn't carry a
# task_name. Order matters within each robot family — most specific first.
_ISAAC_BUILTIN_TASK_MAP = {
    "go2":     ("Isaac-Velocity-Rough-Unitree-Go2-v0",
                "Isaac-Velocity-Flat-Unitree-Go2-v0"),
    "go1":     ("Isaac-Velocity-Rough-Unitree-Go1-v0",
                "Isaac-Velocity-Flat-Unitree-Go1-v0"),
    "anymal_d":("Isaac-Velocity-Rough-Anymal-D-v0",
                "Isaac-Velocity-Flat-Anymal-D-v0"),
    "anymal":  ("Isaac-Velocity-Rough-Anymal-C-v0",
                "Isaac-Velocity-Flat-Anymal-C-v0"),
    "h1":      ("Isaac-Velocity-Rough-H1-v0",
                "Isaac-Velocity-Flat-H1-v0"),
    "g1":      ("Isaac-Velocity-Rough-G1-v0",
                "Isaac-Velocity-Flat-G1-v0"),
    "spot":    ("Isaac-Velocity-Rough-Spot-v0",
                "Isaac-Velocity-Flat-Spot-v0"),
}

# Tokens in the bundle name (or manifest tags) that strongly imply a
# rough/uneven terrain training run.
_ROUGH_TERRAIN_TOKENS = (
    "rough", "stair", "stairs", "terrain", "uneven", "hills", "slope",
)


def _resolve_isaac_review_task_name(
    bundle_path,
    source_meta: dict,
    log_fn: Optional[LogFn] = None,
) -> tuple:
    """Best-effort resolver for the gym task ID to hand to play.py.

    Returns ``(task_name, source_label)`` where *source_label* is one of
    ``source_json`` / ``manifest`` / ``heuristic`` / ``default`` so the
    caller can surface it in the log message.
    """
    from pathlib import Path

    bundle_path = Path(bundle_path)

    # 1. source.json (written at import time by checkpoint_registry)
    tn = str(source_meta.get("task_name", "") or "").strip()
    if tn:
        return tn, "source_json"

    # 2. manifest.yaml — defensive read; rarely populated for IL bundles.
    manifest_robot_models: list = []
    manifest_robot_family: str = ""
    try:
        import yaml as _yaml
        manifest_file = bundle_path / "manifest.yaml"
        if manifest_file.exists():
            md = _yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
            for key_path in (
                ("isaac_lab", "task_name"),
                ("source", "task_name"),
                ("task_name",),
            ):
                cur: Any = md
                ok = True
                for k in key_path:
                    if not isinstance(cur, dict) or k not in cur:
                        ok = False
                        break
                    cur = cur[k]
                if ok and cur:
                    return str(cur), "manifest"
            manifest_robot_models = list(md.get("target_robot_models") or [])
            manifest_robot_family = str(md.get("target_robot_family") or "")
            # tags sometimes carry the task name string (parser appends it)
            for tag in md.get("tags") or []:
                tag_s = str(tag)
                if tag_s.startswith("Isaac-"):
                    return tag_s, "manifest_tag"
    except Exception:
        pass

    # 3. Heuristic from policy_id + robot family.
    name_lc = bundle_path.name.lower()
    is_rough = any(tok in name_lc for tok in _ROUGH_TERRAIN_TOKENS)

    def _scan_for_robot_key(text: str) -> str:
        text = text.lower()
        # Order matters: check the more specific keys first so "anymal_d"
        # matches before "anymal", and "go2" before any "go" prefix.
        for key in ("anymal_d", "go2", "go1", "h1", "g1", "spot", "anymal"):
            if key in text:
                return key
        return ""

    robot_key = ""
    for model in manifest_robot_models:
        robot_key = _scan_for_robot_key(str(model))
        if robot_key:
            break
    if not robot_key and manifest_robot_family:
        robot_key = _scan_for_robot_key(manifest_robot_family)
    if not robot_key:
        robot_key = _scan_for_robot_key(name_lc)
    if not robot_key:
        # Last-ditch default: UnitPort's IL pipeline is Go2-centric (every
        # built-in preset targets Unitree Go2). Falling through to "go2"
        # here matches what _synthesize_default_replay_spec already does
        # for the MuJoCo replay path, and gives the heuristic a fighting
        # chance instead of dropping straight to the Flat-Go2 default.
        robot_key = "go2"
        if log_fn:
            log_fn(
                "[review] Isaac fallback: no robot key in manifest or "
                "bundle name; assuming go2 (UnitPort default). Re-export "
                "the bundle to persist the exact task ID."
            )

    rough_task, flat_task = _ISAAC_BUILTIN_TASK_MAP[robot_key]
    chosen = rough_task if is_rough else flat_task
    if log_fn:
        log_fn(
            f"[review] Isaac fallback: inferred task={chosen} from "
            f"bundle name (robot={robot_key}, rough={is_rough}). "
            f"Re-export the bundle to persist the exact task ID."
        )
    return chosen, "heuristic"


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
    """Open the MuJoCo viewer showing the robot in the configured init pose
    with live physics enabled.

    The robot is held at the init pose via PD control (same gains used during
    training), so the user can observe how the pose interacts with gravity and
    the ground plane.  Close the viewer window to return.
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
        from src.system.training.unitree_gym_env import (
            UnitreeGymEnv,
            _resolve_scene_xml,
            _PD_KP,
            _PD_KD,
        )
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

    sim_dt = getattr(physics, "sim_dt", 0.002)
    control_dt = getattr(physics, "control_dt", 0.02)

    try:
        env = UnitreeGymEnv(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_episode_steps=max(100, physics.episode_max_steps),
            mjcf_path=robot.mjcf_path or None,
            scene_xml_path=scene_xml,
            sim_dt=sim_dt,
            control_dt=control_dt,
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

    # ── Capture the init-pose joint targets for PD hold ────────────────
    n_act = min(env._act_dim, env._model.nu)
    n_qpos_joints = max(env._act_dim, env._model.nq - 7)
    hold_qpos = np.asarray(env._data.qpos[7:7 + n_qpos_joints], dtype=np.float32).copy()

    # Per-actuator gear ratio (first column).
    raw_gear = np.asarray(env._model.actuator_gear[:n_act, 0], dtype=np.float32)
    gear = np.where(raw_gear > 0.0, raw_gear, 1.0)

    # actuator-to-qpos mapping
    c2q = env._ctrl_to_qpos_idx

    decimation = int(round(control_dt / sim_dt)) if sim_dt > 0 else 10
    viewer_dt = 1.0 / 60.0

    mode_str = str(getattr(init_pose_config, "mode", "default") or "default") if init_pose_config else "default"
    if log_fn:
        log_fn(
            f"[init_pose] mode={mode_str} robot={robot.robot_type} "
            f"scene={scene_xml or robot.mjcf_path or 'embedded'} "
            f"— live physics ON — close window to exit"
        )

    try:
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            _try_activate_mujoco_window(log_fn)
            while viewer.is_running():
                # ── PD hold: drive actuators toward init-pose joints ───
                joint_pos = np.asarray(
                    env._data.qpos[7:7 + n_qpos_joints], dtype=np.float32,
                )
                joint_vel = np.asarray(
                    env._data.qvel[6:6 + max(env._act_dim, env._model.nv - 6)],
                    dtype=np.float32,
                )
                for ai in range(n_act):
                    qi = int(c2q[ai])
                    env._data.ctrl[ai] = (
                        _PD_KP * (hold_qpos[qi] - joint_pos[qi])
                        - _PD_KD * joint_vel[qi]
                    ) / gear[ai]

                for _ in range(decimation):
                    mujoco.mj_step(env._model, env._data)

                viewer.sync()
                time.sleep(viewer_dt)
    except Exception as exc:
        if log_fn:
            log_fn(f"[init_pose] Viewer error (non-fatal): {exc}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    if log_fn:
        log_fn("[init_pose] Viewer closed.")
