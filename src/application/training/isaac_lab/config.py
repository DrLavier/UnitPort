"""IsaacLabConfig — pure data describing one Isaac Lab training run.

Ported from DEMO ``src/system/training/isaac_lab_config.py`` with two
RELEASE-specific changes:

1. ``isaac_lab_path`` / ``isaac_lab_python`` / ``isaac_lab_launcher``
   default to whatever ``registers.backends._detect_isaac_lab()``
   discovered at engine refresh time. Callers can override per-run.

2. **No stock-task fallback.** ``_train_script()`` and
   ``build_command()`` both raise when ``unitport_launcher_path`` /
   ``config_file`` are unset. The Phase-2 MVP convenience of falling
   back to Isaac Lab's stock ``Isaac-Velocity-Flat-Unitree-Go2-v0`` was
   removed 2026-05-10 because it silently produced training runs that
   ignored the user's canvas (rewards / terminations / num_envs) — the
   user reported "RELEASE training is broken vs DEMO" and the root
   cause turned out to be DEMO had been silently running the stock
   task while the user thought they were comparing canvas vs canvas.
   ``from_registers()`` always wires ``unitport_launcher_path`` to the
   in-tree ``launcher/il_train_launcher.py``; if that path is empty
   we fail loud rather than degrade to stock.

No Isaac Lab imports — pure config / argv builder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class IsaacLabConfig:
    """Configuration for a single Isaac Lab training run."""

    # --- task ---
    # Empty default — every legitimate IL training routes through
    # ``IsaacLabTrainingTask.__init__`` which compiles ``UnitPortEnvCfg``
    # from the canvas and forces ``--task UnitPort-Custom-v0`` in
    # ``build_command``. ``task_name`` is now only used as a display
    # label for the run; it must NEVER reach the subprocess CLI as a
    # task selector.
    task_name: str = ""
    num_envs: int = 4096
    max_iterations: int = 1500
    seed: int = 42

    # --- output ---
    log_dir: str = ""       # set by caller (<project>/training/runs/<backend_id>/<run_id>/)
    run_id: str = ""        # unique run identifier

    # --- hardware ---
    headless: bool = True
    gpu_id: int = 0

    # --- Isaac Lab installation (auto-filled from registers.backends) ---
    isaac_lab_path: str = ""        # Isaac Lab root directory
    isaac_lab_python: str = ""      # path to Isaac Lab's python interpreter
    isaac_lab_launcher: str = ""    # path to isaaclab.bat / isaaclab.sh

    # --- launcher selection ---
    # When empty (MVP default), use Isaac Lab's stock train.py — only
    # built-in tasks (``--task Isaac-...``) are runnable. When set to an
    # absolute path, use that script (e.g. UnitPort's il_train_launcher
    # once ported in Phase 2.5).
    unitport_launcher_path: str = ""

    # --- advanced ---
    extra_args: List[str] = field(default_factory=list)
    resume_checkpoint: Optional[str] = None
    warm_start_actor: bool = False
    config_file: str = ""           # compiled @configclass file (Phase 2.5)

    # --- algorithm (RSL-RL default) ---
    algorithm: str = "ppo"          # ppo | sac | AMP_PPO

    # --- AMP-specific (Phase 2.5; inert when algorithm == "ppo") ---
    amp_motion_files: List[str] = field(default_factory=list)
    amp_reward_coef: float = 2.0
    amp_num_preload_transitions: int = 2_000_000
    amp_transition_dt: float = 0.02
    amp_task_reward_lerp: float = 0.5
    amp_replay_buffer_size: int = 1_000_000
    amp_disc_grad_penalty: float = 10.0
    amp_disc_lr: float = 1e-4
    amp_disc_label_smoothing: float = 0.9
    amp_disc_logit_clamp_max: float = 4.0
    amp_reward_clamp_per_step: float = 50.0
    amp_policy_std_clamp_max: float = 1.5
    amp_disc_overrides_file: str = ""
    amp_lerp_schedule: str = ""
    amp_lerp_schedule_json: str = ""
    amp_obs_fields: str = ""
    amp_auto_inject_ref_obs: bool = True
    # training_motion playback knobs — forwarded so the launcher / amp
    # data_provider can use them once the build_*_pool signatures grow
    # support. Today only stamped onto run_meta as provenance.
    amp_phase_mode: str = "loop"
    amp_random_start_phase: bool = True
    amp_task_filter: str = ""
    amp_task_labels: str = ""
    amp_task_items: str = ""

    # --- Staged training pipeline (Phase 2.5) ---
    stage_schedule_json: str = ""
    stage_checkpoint_strategy: str = "both"

    # --- Post-training eval (from eval_config node) ---
    eval_episodes: int = 0                 # 0 = skip post-train eval
    eval_deterministic: bool = True
    eval_success_threshold: float = 0.8
    eval_save_best_model: bool = False
    eval_video_dir: str = ""

    # --- RSI (Phase 2.5) ---
    init_pose_mode: str = "default"
    init_pose_rsi_prob: float = 1.0
    init_pose_rsi_sample_mode: str = "frame_0"
    init_pose_keyframe_name: str = "home"
    body_mapping_json: str = ""
    robot_asset_id: str = ""
    amp_rsi_enabled: bool = False
    amp_rsi_prob: float = 0.0
    amp_rsi_joint_noise: float = 0.02
    amp_rsi_n_frames: int = 20000

    # ------------------------------------------------------------------
    # Auto-fill installation paths from RELEASE registers.backends
    # ------------------------------------------------------------------

    @classmethod
    def from_registers(cls, **overrides) -> "IsaacLabConfig":
        """Build a config pre-populated with the registered Isaac Lab
        installation (root + python + launcher).

        Reads ``~/UnitPort/engines/isaac_lab.json`` via
        ``registers.backends._detect_isaac_lab`` to discover the root, then
        derives launcher (``isaaclab.bat|sh``) and python paths under it.
        """
        from registers.backends import _detect_isaac_lab, _find_isaac_python

        det = _detect_isaac_lab()
        root_str = det.get("path") or ""
        if not root_str:
            raise RuntimeError(
                "Isaac Lab not registered — call EngineService."
                "register_isaac_local(<root>) or import_isaac_lab_path_from_demo() first"
            )
        root = Path(root_str)
        launcher = ""
        for name in ("isaaclab.bat", "isaaclab.sh"):
            p = root / name
            if p.exists():
                launcher = str(p)
                break
        python = _find_isaac_python(root) or ""
        cfg = cls(
            isaac_lab_path=str(root),
            isaac_lab_python=python,
            isaac_lab_launcher=launcher,
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"IsaacLabConfig has no field {k!r}")
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_training_spec(cls, spec) -> "IsaacLabConfig":
        """Build a config from a :class:`TrainingSpec` (or its ``to_dict``).

        ``TrainingSpec.to_dict()`` is a nested dict — the legacy
        ``IsaacLabBackendAdapter.build_task`` did flat ``spec.get("num_envs")``
        lookups and silently dropped every canvas value. This classmethod
        reads from the real nested paths::

            task.isaac_task_name              -> task_name (else default)
            env.n_envs                        -> num_envs
            algorithm.il_ppo.max_iterations   -> max_iterations
            algorithm.seed                    -> seed
            algorithm.il_ppo.headless         -> headless
            algorithm.training_mode           -> algorithm ("AMP_PPO" | "ppo")

        When ``training_mode == "AMP_PPO"`` (H2), additionally forwards
        AMP / motion / stage / RSI fields onto the launcher contract::

            il.amp                            -> amp_* fields
            il.motion_ref.clip_paths          -> amp_motion_files
            stage_schedule                    -> stage_schedule_json (b64 JSON)
            actor.init_pose                   -> init_pose_mode + rsi_*
            robot.sku                         -> robot_asset_id
            registers.backends.train_launcher_path("isaac_lab")  -> unitport_launcher_path

        The UnitPort launcher (1382-line ``il_train_launcher.py``) is Phase
        2.5 territory — until it is ported in / registered, AMP_PPO Isaac
        Lab runs are blocked by :func:`spec_validator._check_amp_wiring`
        which raises an INCOMPLETE_AMP_WIRING ERROR before submit (so the
        user does not silently get a stock-train.py PPO run instead).

        Accepts either a :class:`TrainingSpec` instance or its ``to_dict()``
        output — both round-trip through :meth:`TrainingSpec.from_dict`.
        """
        import json
        from dataclasses import asdict

        from application.training.training_spec import TrainingSpec

        if isinstance(spec, dict):
            spec_obj = TrainingSpec.from_dict(spec)
        else:
            spec_obj = spec

        # Display label only — never reaches the subprocess CLI
        # (build_command always emits ``--task UnitPort-Custom-v0``).
        # The previous ``or "Isaac-Velocity-Flat-Unitree-Go2-v0"`` fallback
        # was misleading: it suggested the stock task could still run, but
        # build_command refuses to launch without config_file anyway.
        task_name = (
            getattr(getattr(spec_obj, "task", None), "isaac_task_name", "") or ""
        ).strip()

        algo = getattr(spec_obj, "algorithm", None)
        il_ppo = getattr(algo, "il_ppo", None) if algo is not None else None

        n_envs = int(getattr(getattr(spec_obj, "env", None), "n_envs", 0) or 0) or 4096
        seed = int(getattr(algo, "seed", 42) if algo is not None else 42)
        max_iters = int(
            getattr(il_ppo, "max_iterations", 1500) if il_ppo is not None else 1500
        )
        headless = bool(
            getattr(il_ppo, "headless", True) if il_ppo is not None else True
        )

        training_mode = (
            str(getattr(algo, "training_mode", "PPO") if algo is not None else "PPO")
            .strip()
            .upper()
        )
        # IsaacLabConfig.algorithm vocabulary: lower-case "ppo"/"sac" or the
        # exact string "AMP_PPO" — see build_command()'s gating.
        if training_mode == "AMP_PPO":
            algorithm = "AMP_PPO"
        else:
            algorithm = (
                str(getattr(algo, "algorithm", "PPO") if algo is not None else "PPO")
                .strip()
                .lower() or "ppo"
            )

        cfg = cls.from_registers(
            task_name=task_name,
            num_envs=n_envs,
            max_iterations=max_iters,
            seed=seed,
            headless=headless,
            algorithm=algorithm,
        )

        # Always forward robot SKU + init_pose mode — non-default values
        # are inert when ``unitport_launcher_path`` is empty (build_command
        # gates them behind the launcher) but get carried through once the
        # launcher is registered, so the canvas configuration is preserved.
        robot_ref = getattr(spec_obj, "robot", None)
        cfg.robot_asset_id = str(getattr(robot_ref, "sku", "") or "")

        # Body mapping (Robot node body_mapping) must travel to the
        # launcher whenever init_pose_mode=reference_frame_0 or any other
        # path needs IR-role → physical-joint resolution for the active
        # robot. Previously this field defaulted to "" and the launcher's
        # RSI branch silently skipped applying the reference frame:
        #     "RSI: no --unitport_body_mapping — cannot route clip joints
        #      through IR layer. Skipped."
        # Serialize the RobotSpecRef.body_role_map to JSON so the launcher
        # can base64-decode and feed JointIRResolver-equivalent lookup.
        body_role_map = getattr(robot_ref, "body_role_map", None) if robot_ref is not None else None
        if isinstance(body_role_map, dict) and body_role_map:
            cfg.body_mapping_json = json.dumps(body_role_map)

        init_pose = getattr(getattr(spec_obj, "actor", None), "init_pose", None)
        if init_pose is not None:
            cfg.init_pose_mode = str(getattr(init_pose, "mode", "default") or "default")
            cfg.init_pose_rsi_prob = float(getattr(init_pose, "rsi_prob", 1.0) or 1.0)
            cfg.init_pose_rsi_sample_mode = str(
                getattr(init_pose, "rsi_sample_mode", "frame_0") or "frame_0"
            )
            cfg.init_pose_keyframe_name = str(
                getattr(init_pose, "keyframe_name", "home") or "home"
            )

        # Post-training eval — eval_config node fields surface here so the
        # launcher's eval phase (added in this pass) can run N deterministic
        # episodes after runner.learn and write eval_results.json. Without
        # this forwarding the canvas's eval section was silently dropped.
        eval_cfg = getattr(spec_obj, "eval", None)
        if eval_cfg is not None:
            cfg.eval_episodes = int(getattr(eval_cfg, "eval_episodes", 0) or 0)
            cfg.eval_deterministic = bool(getattr(eval_cfg, "deterministic", True))
            cfg.eval_success_threshold = float(
                getattr(eval_cfg, "success_threshold", 0.8) or 0.0
            )
            cfg.eval_save_best_model = bool(getattr(eval_cfg, "save_best_model", False))
            cfg.eval_video_dir = str(getattr(eval_cfg, "video_dir", "") or "")

        # Stage schedule — base64-encoded JSON of the dataclass dict, consumed
        # by the launcher's ``--unitport_stage_schedule`` flag. The previous
        # ``try / except: stage_schedule_json = ""`` silently masked
        # dataclass corruption — strict-mode contract now lets the failure
        # surface as the real exception (asdict raises TypeError on a
        # malformed dataclass instance).
        stage_sched = getattr(spec_obj, "stage_schedule", None)
        if stage_sched is not None:
            cfg.stage_schedule_json = json.dumps(asdict(stage_sched))
            cfg.stage_checkpoint_strategy = str(
                getattr(stage_sched, "checkpoint_strategy", "both") or "both"
            )

        # Motion clips are also consumed on the PPO path when the user
        # picks ``init_pose_mode = reference_frame_0`` on actor_setting:
        # the launcher's RSI block loads the first clip frame as the
        # episode-start joint pose. Setting amp_motion_files unconditionally
        # (when motion_ref carries clip_paths) lets PPO + RSI work without
        # forcing the user onto AMP_PPO mode. The AMP-only consumer
        # ignores it when algorithm != AMP_PPO.
        il_all = getattr(spec_obj, "il", None)
        motion_ref_all = getattr(il_all, "motion_ref", None) if il_all is not None else None
        if motion_ref_all is not None:
            _all_clip_paths = getattr(motion_ref_all, "clip_paths", None) or {}
            if _all_clip_paths:
                cfg.amp_motion_files = [str(p) for p in _all_clip_paths.values() if p]

        # AMP_PPO-only payload (silent on non-AMP runs).
        if algorithm == "AMP_PPO":
            il = getattr(spec_obj, "il", None)
            amp = getattr(il, "amp", None) if il is not None else None
            motion_ref = getattr(il, "motion_ref", None) if il is not None else None
            # motion_fps drives the (s, s') transition gap fed to the
            # discriminator: amp_transition_dt = 1 / motion_fps. Previously
            # the launcher always saw the IsaacLabConfig default (0.02 s = 50
            # Hz) regardless of what the user wrote on training_motion.
            motion_fps = float(getattr(motion_ref, "motion_fps", 50.0) or 50.0) if motion_ref is not None else 50.0
            if motion_fps > 0:
                cfg.amp_transition_dt = 1.0 / motion_fps
            if motion_ref is not None:
                cfg.amp_phase_mode = str(getattr(motion_ref, "phase_mode", "loop") or "loop")
                cfg.amp_random_start_phase = bool(
                    getattr(motion_ref, "random_start_phase", True)
                )
            if amp is not None:
                cfg.amp_reward_coef = float(getattr(amp, "amp_reward_coef", 2.0))
                cfg.amp_task_reward_lerp = float(getattr(amp, "task_reward_lerp", 0.5))
                cfg.amp_disc_grad_penalty = float(getattr(amp, "disc_grad_penalty", 10.0))
                cfg.amp_disc_label_smoothing = float(getattr(amp, "disc_label_smoothing", 0.9))
                cfg.amp_replay_buffer_size = int(getattr(amp, "amp_replay_buffer_size", 1_000_000))
                cfg.amp_num_preload_transitions = int(getattr(amp, "num_preload_transitions", 2_000_000))
                cfg.amp_disc_lr = float(getattr(amp, "disc_lr", 1e-4))
                cfg.amp_lerp_schedule = str(getattr(amp, "lerp_schedule", "") or "")
                disc = getattr(amp, "disc", None)
                if disc is not None:
                    cfg.amp_disc_logit_clamp_max = float(getattr(disc, "disc_logit_clamp_max", 4.0))
                    cfg.amp_reward_clamp_per_step = float(getattr(disc, "reward_clamp_per_step", 50.0))
                    cfg.amp_policy_std_clamp_max = float(getattr(disc, "policy_std_clamp_max", 1.5))
                    # discriminator-specific knobs that the trainer-level amp
                    # block does not carry (the trainer was being read as a
                    # fallback only). Without these forwards the canvas's
                    # discriminator node settings for custom lerp / obs
                    # routing were silently dropped on the way to launcher.
                    cfg.amp_lerp_schedule_json = str(getattr(disc, "lerp_schedule_json", "") or "")
                    cfg.amp_obs_fields = str(getattr(disc, "amp_obs_fields", "") or "")
                    cfg.amp_auto_inject_ref_obs = bool(getattr(disc, "auto_inject_ref_obs", True))
                    # When the user picked the "custom" preset on
                    # discriminator.lerp_schedule + filled lerp_schedule_json,
                    # the launcher's --unitport_amp_lerp_schedule expects a
                    # JSON anneal dict ({"start": ..., "end": ..., "over_iters": ...}).
                    # Override the enum-name forwarding so the JSON actually
                    # reaches the launcher; otherwise the user's custom
                    # schedule was a no-op.
                    _disc_lerp_mode = str(getattr(disc, "lerp_schedule", "") or "")
                    if _disc_lerp_mode == "custom" and cfg.amp_lerp_schedule_json:
                        cfg.amp_lerp_schedule = cfg.amp_lerp_schedule_json
                    elif _disc_lerp_mode and _disc_lerp_mode != "none":
                        cfg.amp_lerp_schedule = _disc_lerp_mode
            # amp_motion_files was already set above (motion_ref_all branch
            # — works for PPO + RSI too); no need to reset here.

            # Command-conditioned AMP sampling: the launcher
            # (--unitport_amp_task_items) accepts a base64-encoded JSON list
            # of TaskItem dicts and, when present, draws expert transitions
            # from the matching sub-buffer per env-current command instead
            # of uniform sampling. Without this forward the canvas's
            # training_items configuration was silently ignored and AMP
            # fell back to uniform sampling regardless of how the user
            # split items in training_motion.
            motion = getattr(spec_obj, "motion", None)
            training_items = (
                getattr(motion, "training_items", None) or {}
                if motion is not None else {}
            )
            if training_items:
                import base64
                task_items_list = []
                for item_id, payload in training_items.items():
                    if not isinstance(payload, dict):
                        continue
                    if not payload.get("enabled", False):
                        continue
                    entry = {"id": str(item_id), **{
                        k: v for k, v in payload.items() if k != "enabled"
                    }}
                    task_items_list.append(entry)
                if task_items_list:
                    encoded = base64.b64encode(
                        json.dumps(task_items_list).encode("utf-8")
                    ).decode("ascii")
                    cfg.amp_task_items = encoded

        # Checkpoint forwarding (base_asset node → IsaacLab launcher).
        # spec_compiler._populate_algorithm fills spec.algorithm.checkpoint
        # from the base_asset node (start_point + checkpoint_id + load_mode).
        # Without this read the IL backend silently ignored "Latest" / "Load"
        # selections — users got a scratch run instead of resume / warm-start.
        # checkpoint_file is "" for start_point=__new__; non-empty for the
        # other two branches (path already split out of the run:/export:
        # token by spec_compiler).
        checkpoint = getattr(algo, "checkpoint", None) if algo is not None else None
        if checkpoint is not None:
            ckpt_path = str(getattr(checkpoint, "checkpoint_file", "") or "").strip()
            if ckpt_path:
                cfg.resume_checkpoint = ckpt_path
                load_mode = str(
                    getattr(checkpoint, "load_mode", "scratch") or ""
                ).strip()
                cfg.warm_start_actor = (load_mode == "warm_start_actor")

        # Wire the in-tree UnitPort launcher. The path is a function of
        # where ``RELEASE/src/application/`` lives on disk — resolved by
        # the backends registry, not user config. No override key, no
        # settings dialog: the launcher is part of the source tree.
        # Stock RSL-RL train.py rejects ``--log_dir`` and ``--unitport_*``
        # flags, so without this launcher run artifacts can't be routed
        # into the project tree at all. ``spec_validator._check_amp_wiring``
        # already raises INCOMPLETE_AMP_WIRING when the file is missing,
        # so leaving ``unitport_launcher_path`` empty here is impossible
        # on the canonical Play-button path; we still skip the assignment
        # for non-Play call sites (e.g. spec inspection) when the file
        # is absent rather than carrying a bogus path forward.
        from registers import backends as _backends
        launcher_path = _backends.train_launcher_path("isaac_lab")
        if launcher_path is not None and launcher_path.is_file():
            cfg.unitport_launcher_path = str(launcher_path)

        return cfg

    @property
    def control_dt(self) -> float:
        """Approximate control dt for velocity tasks (sim_dt * decimation)."""
        return 0.02  # 50 Hz default; overridden by env.yaml after training

    # ------------------------------------------------------------------
    # Script resolution
    # ------------------------------------------------------------------

    def _train_script(self) -> str:
        """Return the absolute path to the Python script we launch.

        Always the in-tree UnitPort launcher; fail if it isn't wired.
        Stock Isaac-Lab ``train.py`` fallback is intentionally removed —
        without the UnitPort launcher there is no way to thread the
        compiled ``UnitPortEnvCfg`` into RSL-RL, and the stock script
        would silently run a built-in task instead of the user's canvas.
        """
        if not self.unitport_launcher_path:
            raise RuntimeError(
                "[IsaacLabConfig] unitport_launcher_path is empty. "
                "RELEASE refuses to fall back to Isaac Lab's stock "
                "train.py because the canvas would be silently ignored. "
                "Use IsaacLabConfig.from_registers() (which auto-wires "
                "application/training/isaac_lab/launcher/il_train_launcher.py) "
                "or set unitport_launcher_path explicitly."
            )
        return str(Path(self.unitport_launcher_path).resolve())

    def _play_script(self) -> str:
        """Return the absolute path to the play / export script.

        For MVP, fall back to Isaac Lab's stock ``play.py`` next to
        ``train.py``. UnitPort's il_play_launcher (which routes
        UnitPort run-dirs around RSL-RL's hardcoded log_root_path
        regex) lands in Phase 2.5.
        """
        if self.unitport_launcher_path:
            base = Path(self.unitport_launcher_path).parent
            cand = base / "il_play_launcher.py"
            if cand.exists():
                return str(cand)
        if not self.isaac_lab_path:
            raise RuntimeError("isaac_lab_path is empty — cannot resolve stock play.py")
        return str(
            Path(self.isaac_lab_path)
            / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
        )

    # ------------------------------------------------------------------
    # Subprocess command builder (verbatim from DEMO except for
    # _train_script default + the robot_asset_id branch which is gated
    # behind UnitPort launcher because the stock train.py has no
    # --unitport_robot_asset_id flag)
    # ------------------------------------------------------------------

    def _build_launcher_cmd(self, script: str, args: List[str]) -> List[str]:
        """Build a subprocess command that uses Isaac Lab's own Python.

        On Linux/Mac: ``<launcher> -p <script> <args>``
        On Windows:   writes a tiny temp .bat wrapper that quotes the Python
                      path correctly (works around isaaclab.bat's unquoted
                      ``call !python_exe!`` bug with spaces in path).
        """
        launcher = self.isaac_lab_launcher
        if launcher and Path(launcher).exists():
            if os.name == "nt" and launcher.lower().endswith(".bat"):
                wrapper = self._write_win_wrapper(script, args)
                return ["cmd", "/c", wrapper]
            return [launcher, "-p", script, *args]

        if self.isaac_lab_python and Path(self.isaac_lab_python).exists():
            return [self.isaac_lab_python, script, *args]
        return ["python", script, *args]

    def _write_win_wrapper(self, script: str, args: List[str]) -> str:
        """Create a temp .bat that invokes isaaclab.bat's Python with quoting."""
        import tempfile

        lab_root = self.isaac_lab_path.replace("/", "\\")
        args_str = " ".join(f'"{a}"' if " " in a else a for a in args)
        content = (
            "@echo off\r\n"
            "setlocal EnableExtensions EnableDelayedExpansion\r\n"
            f'set "ISAACLAB_PATH={lab_root}"\r\n'
            'rem --- discover python (same logic as isaaclab.bat) ---\r\n'
            'if not "%CONDA_PREFIX%"=="" (\r\n'
            '    set "python_exe=%CONDA_PREFIX%\\python.exe"\r\n'
            ') else (\r\n'
            '    set "python_exe=%ISAACLAB_PATH%\\_isaac_sim\\python.bat"\r\n'
            ')\r\n'
            'if not exist "!python_exe!" (\r\n'
            '    for /f "delims=" %%i in (\'where python\') do (\r\n'
            '        if not defined python_exe_found (\r\n'
            '            set "python_exe_found=%%i"\r\n'
            '        )\r\n'
            '    )\r\n'
            '    if defined python_exe_found set "python_exe=!python_exe_found!"\r\n'
            ')\r\n'
            'if not exist "!python_exe!" (\r\n'
            '    echo [ERROR] No Python found for Isaac Lab\r\n'
            '    exit /b 1\r\n'
            ')\r\n'
            'echo [INFO] Using python from: !python_exe!\r\n'
            f'"!python_exe!" "{script}" {args_str}\r\n'
        )
        fd, path = tempfile.mkstemp(suffix=".bat", prefix="unitport_il_run_")
        os.close(fd)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def build_command(self) -> List[str]:
        """Build the subprocess launch command for RSL-RL training.

        Refuses to launch unless **both** ``config_file`` (the compiled
        ``UnitPortEnvCfg`` path) and ``unitport_launcher_path`` (the
        in-tree ``il_train_launcher.py``) are set. The previous fallback
        to ``self.task_name`` silently selected Isaac Lab's stock
        ``Isaac-Velocity-Flat-Unitree-Go2-v0`` task whenever either
        was missing, ignoring the user's canvas entirely.
        """
        if not self.config_file or not self.unitport_launcher_path:
            raise RuntimeError(
                "[IsaacLabConfig] cannot build launch command — "
                f"config_file={self.config_file!r}, "
                f"unitport_launcher_path={self.unitport_launcher_path!r}. "
                "Both must be set; RELEASE refuses to fall back to Isaac "
                "Lab's stock task because it would silently ignore the "
                "user's canvas. The legitimate path is: hit the top Play "
                "button → submit_canvas_training → IsaacLabTrainingTask "
                "compiles the canvas to UnitPortEnvCfg and sets config_file. "
                "If you are calling build_command from a custom code path, "
                "set both fields explicitly."
            )

        task = "UnitPort-Custom-v0"
        # num_envs / max_iterations / seed are baked into the compiled
        # UnitPortEnvCfg + PPORunnerCfg by env_cfg_compiler — emitting them
        # as CLI overrides would clobber the canvas-driven values.
        args = ["--task", task]
        args.extend(["--unitport_config", self.config_file])
        if self.log_dir:
            args.extend(["--log_dir", self.log_dir])
        if self.headless:
            args.append("--headless")
        if self.resume_checkpoint:
            ckpt = Path(self.resume_checkpoint)
            if ckpt.is_dir():
                args.extend(["--load_run", str(ckpt)])
            else:
                args.extend(["--load_run", str(ckpt.parent)])
                if ckpt.name:
                    args.extend(["--checkpoint", ckpt.name])
            if self.warm_start_actor and self.unitport_launcher_path:
                args.append("--unitport_warm_start_actor")

        # The flags below are UnitPort-launcher-only — the stock
        # Isaac train.py rejects unknown args, so suppress them unless
        # we're routed through the UnitPort launcher.
        if self.unitport_launcher_path:
            if self.robot_asset_id:
                args.extend(["--unitport_robot_asset_id", self.robot_asset_id])

            if self.algorithm == "AMP_PPO":
                args.extend([
                    "--unitport_algorithm", "AMP_PPO",
                    "--unitport_amp_reward_coef", str(self.amp_reward_coef),
                    "--unitport_amp_preload", str(self.amp_num_preload_transitions),
                    "--unitport_amp_transition_dt", str(self.amp_transition_dt),
                    "--unitport_amp_task_reward_lerp", str(self.amp_task_reward_lerp),
                    "--unitport_amp_replay_buffer_size", str(self.amp_replay_buffer_size),
                    "--unitport_amp_disc_grad_penalty", str(self.amp_disc_grad_penalty),
                    "--unitport_amp_disc_lr", str(self.amp_disc_lr),
                    "--unitport_amp_disc_label_smoothing", str(self.amp_disc_label_smoothing),
                    "--unitport_amp_disc_logit_clamp_max", str(self.amp_disc_logit_clamp_max),
                    "--unitport_amp_reward_clamp_per_step", str(self.amp_reward_clamp_per_step),
                    "--unitport_amp_policy_std_clamp_max", str(self.amp_policy_std_clamp_max),
                ])
                if self.amp_disc_overrides_file:
                    args.extend(["--unitport_amp_disc_overrides_file", str(self.amp_disc_overrides_file)])
                if self.amp_lerp_schedule:
                    args.extend(["--unitport_amp_lerp_schedule", str(self.amp_lerp_schedule)])
                if self.amp_task_filter:
                    args.extend(["--unitport_amp_task_filter", str(self.amp_task_filter)])
                if self.amp_task_labels:
                    args.extend(["--unitport_amp_task_labels", str(self.amp_task_labels)])
                if self.amp_task_items:
                    args.extend(["--unitport_amp_task_items", str(self.amp_task_items)])
                if self.amp_motion_files:
                    args.extend([
                        "--unitport_amp_motion_files",
                        ",".join(str(p) for p in self.amp_motion_files),
                    ])
                if self.amp_obs_fields:
                    args.extend([
                        "--unitport_amp_obs_fields", self.amp_obs_fields,
                    ])
                # auto_inject_ref_obs is a bool — emit a single --flag when
                # disabled (launcher treats its absence as default True).
                if not self.amp_auto_inject_ref_obs:
                    args.append("--unitport_amp_no_auto_inject_ref_obs")
                # Metadata-only flags (data_provider does not consume these
                # yet — launcher stamps them onto run_meta so deploy /
                # post-mortem tooling can recover the user's intent).
                args.extend([
                    "--unitport_amp_phase_mode", self.amp_phase_mode,
                ])
                if not self.amp_random_start_phase:
                    args.append("--unitport_amp_no_random_start_phase")
                if self.amp_rsi_enabled and self.amp_rsi_prob > 0.0:
                    args.extend([
                        "--unitport_amp_rsi_prob", str(self.amp_rsi_prob),
                        "--unitport_amp_rsi_n_frames", str(self.amp_rsi_n_frames),
                    ])

            if self.init_pose_mode and self.init_pose_mode != "default":
                args.extend([
                    "--unitport_init_pose_mode", self.init_pose_mode,
                    "--unitport_rsi_prob", str(self.init_pose_rsi_prob),
                    "--unitport_rsi_sample_mode", self.init_pose_rsi_sample_mode,
                ])
                if self.init_pose_mode == "keyframe" and self.init_pose_keyframe_name:
                    args.extend([
                        "--unitport_init_pose_keyframe_name",
                        self.init_pose_keyframe_name,
                    ])
                if self.body_mapping_json:
                    import base64
                    encoded = base64.b64encode(
                        self.body_mapping_json.encode("utf-8")
                    ).decode("ascii")
                    args.extend(["--unitport_body_mapping", encoded])
                # reference_frame_0 RSI needs the motion clip; carry the
                # files even on the PPO path (the AMP block above only
                # forwards them when algorithm == AMP_PPO).
                if (
                    self.init_pose_mode == "reference_frame_0"
                    and self.amp_motion_files
                    and self.algorithm != "AMP_PPO"
                ):
                    args.extend([
                        "--unitport_amp_motion_files",
                        ",".join(str(p) for p in self.amp_motion_files),
                    ])

            if self.stage_schedule_json:
                import base64
                encoded = base64.b64encode(
                    self.stage_schedule_json.encode("utf-8")
                ).decode("ascii")
                args.extend([
                    "--unitport_stage_schedule", encoded,
                    "--unitport_stage_checkpoint_strategy", self.stage_checkpoint_strategy,
                ])

            # Post-training eval — only emit when eval_episodes > 0 so the
            # launcher can short-circuit and skip the eval phase entirely
            # when the user did not wire an eval_config node.
            if self.eval_episodes > 0:
                args.extend([
                    "--unitport_eval_episodes", str(self.eval_episodes),
                    "--unitport_eval_success_threshold", str(self.eval_success_threshold),
                ])
                if not self.eval_deterministic:
                    args.append("--unitport_eval_stochastic")
                if self.eval_save_best_model:
                    args.append("--unitport_save_best_model")
                if self.eval_video_dir:
                    args.extend(["--unitport_eval_video_dir", self.eval_video_dir])

        args.extend(self.extra_args)
        return self._build_launcher_cmd(self._train_script(), args)

    def build_export_command(self, checkpoint_path: str) -> List[str]:
        """Build command to export a trained policy via play.py."""
        args = [
            "--task", self.task_name,
            "--num_envs", "1",
            "--load_run", str(Path(checkpoint_path).parent),
            "--load_checkpoint", str(Path(checkpoint_path).name),
            "--headless",
        ]
        return self._build_launcher_cmd(self._play_script(), args)

    def build_review_command(self, checkpoint_dir: str) -> List[str]:
        """Build command to visually replay a trained policy (Isaac Sim viewport)."""
        args = [
            "--task", self.task_name,
            "--num_envs", "1",
            "--load_run", checkpoint_dir,
        ]
        return self._build_launcher_cmd(self._play_script(), args)
