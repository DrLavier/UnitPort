"""Isaac Lab training configuration dataclass.

Captures everything UnitPort needs to launch an Isaac Lab training run
as an external subprocess.  No Isaac Lab imports — this is pure config.
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
    task_name: str = "Isaac-Velocity-Flat-Go2-v0"
    num_envs: int = 4096
    max_iterations: int = 1500
    seed: int = 42

    # --- output ---
    log_dir: str = ""       # set by caller (project training/runs/<run_id>/)
    run_id: str = ""        # unique run identifier

    # --- hardware ---
    headless: bool = True
    gpu_id: int = 0

    # --- Isaac Lab installation ---
    isaac_lab_path: str = ""        # Isaac Lab root directory
    isaac_lab_python: str = ""      # path to Isaac Lab's python interpreter
    isaac_lab_launcher: str = ""    # path to isaaclab.bat / isaaclab.sh

    # --- advanced ---
    extra_args: List[str] = field(default_factory=list)
    resume_checkpoint: Optional[str] = None
    warm_start_actor: bool = False
    """When True AND ``resume_checkpoint`` is set, the launcher loads only
    the actor-critic weights from the checkpoint (via ``strict=False``)
    and keeps the optimizer / discriminator / amp_normalizer freshly
    initialized. Required when seeding an AMP-PPO run from a pure PPO
    checkpoint, and also the right mode when reusing an old actor with
    a new reward layout or a new task."""
    config_file: str = ""           # compiled @configclass Python file from IsaacLabConfigCompiler

    # --- algorithm (RSL-RL default) ---
    algorithm: str = "ppo"          # ppo | sac | AMP_PPO

    # --- AMP-specific config (phase_3 of AMP_design.yaml §4) ---
    # Populated only when algorithm == "AMP_PPO". Fed into the launcher
    # via --unitport_amp_* CLI args — see build_command() below.
    amp_motion_files: List[str] = field(default_factory=list)
    """Absolute paths to amp_legged_gym motion .txt/.json files. At least
    one file is required when algorithm == AMP_PPO."""

    amp_reward_coef: float = 2.0
    """Forwarded as --unitport_amp_reward_coef. Matches AMPConfig default."""

    amp_num_preload_transitions: int = 2_000_000
    """Forwarded as --unitport_amp_preload. 0 = stream sampling."""

    amp_transition_dt: float = 0.02
    """(s, s') gap passed to MotionClip.sample_transitions. Typical = control_dt."""

    amp_task_reward_lerp: float = 0.5
    """Forwarded as --unitport_amp_task_reward_lerp. Blend coefficient for
    ``r_total = (1-l)*style + l*task``. Default 0.5 per AMP-PPO_design.yaml
    §1.reward_mixing."""

    amp_replay_buffer_size: int = 1_000_000
    """Forwarded as --unitport_amp_replay_buffer_size."""

    amp_disc_grad_penalty: float = 10.0
    """Forwarded as --unitport_amp_disc_grad_penalty. R1 gradient penalty lambda."""

    amp_disc_lr: float = 1e-4
    """Forwarded as --unitport_amp_disc_lr. Separate learning rate for the
    discriminator. When 0 the discriminator shares the policy LR."""

    amp_disc_label_smoothing: float = 0.9
    """Forwarded as --unitport_amp_disc_label_smoothing. Soft expert BCE
    target (Salimans 2016 one-sided label smoothing).  1.0 = no
    smoothing, 0.9 = standard GAN default."""

    amp_lerp_schedule: str = ""
    """JSON linear anneal schedule for task_reward_lerp. Empty = constant.
    Example: '{"start":0.75,"end":0.3,"over_iters":2000}'"""

    # --- Staged training pipeline ---
    stage_schedule_json: str = ""
    """JSON-serialized StageSchedule. Empty = single-stage (backward compat).
    When non-empty, forwarded as --unitport_stage_schedule to the launcher."""

    stage_checkpoint_strategy: str = "both"
    """Checkpoint strategy at stage transitions: save_on_advance | save_best_per_stage | both."""

    robot_asset_id: str = ""
    """Optional: when set, forwarded as --unitport_robot_asset_id and
    resolve_for_training is called to produce --unitport_robot_usd. Used
    by both PPO and AMP paths."""

    @property
    def control_dt(self) -> float:
        """Approximate control dt for velocity tasks (sim_dt * decimation)."""
        return 0.02  # 50 Hz default; overridden by env.yaml after training

    def _train_script(self) -> str:
        """Resolve the path to UnitPort's custom training launcher."""
        return str(Path(__file__).resolve().parent / "il_train_launcher.py")

    def _play_script(self) -> str:
        """Resolve the path to UnitPort's custom play launcher.

        We do NOT use Isaac Lab's stock ``rsl_rl/play.py`` because it calls
        ``get_checkpoint_path(log_root_path, agent_cfg.load_run, ...)`` —
        ``log_root_path`` is hardcoded to ``<isaac_root>/logs/rsl_rl/<exp>``
        and ``load_run`` is treated as a regex against subdirectory names,
        not as an absolute path. UnitPort's checkpoints live under the
        project tree (``projects/<slug>/training/runs/...``), outside Isaac
        Lab's expected log layout, so the stock launcher can never find
        them. ``il_play_launcher.py`` bypasses ``get_checkpoint_path``
        and feeds the absolute ``.pt`` path straight into
        ``OnPolicyRunner.load()``.
        """
        return str(Path(__file__).resolve().parent / "il_play_launcher.py")

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
                # Write a wrapper .bat that discovers Python from isaaclab.bat
                # environment, then calls it WITH quotes around the exe path.
                wrapper = self._write_win_wrapper(script, args)
                return ["cmd", "/c", wrapper]
            return [launcher, "-p", script, *args]

        # Fallback: explicit python path (if detection filled it)
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

        Always uses UnitPort's ``il_train_launcher.py`` which supports both:
          - Built-in tasks (``--task Isaac-Velocity-Flat-...``, no config file)
          - Custom tasks (``--task UnitPort-Custom-v0 --unitport_config <path>``)
        """
        # If we have a compiled config file, use custom task ID
        task = self.task_name
        if self.config_file and not task.startswith("Isaac-"):
            task = "UnitPort-Custom-v0"

        args = [
            "--task", task,
            "--num_envs", str(self.num_envs),
            "--max_iterations", str(self.max_iterations),
            "--seed", str(self.seed),
        ]
        if self.config_file:
            args.extend(["--unitport_config", self.config_file])
        if self.log_dir:
            args.extend(["--log_dir", self.log_dir])
        if self.headless:
            args.append("--headless")
        if self.resume_checkpoint:
            # RSL-RL expects --load_run <run_dir> [--checkpoint <file_name>].
            # BaseAssetNode gives us a resolved .pt / .zip file path, so split
            # it into (parent, name). If the caller passed an existing
            # directory, use it directly.
            ckpt = Path(self.resume_checkpoint)
            if ckpt.is_dir():
                args.extend(["--load_run", str(ckpt)])
            else:
                args.extend(["--load_run", str(ckpt.parent)])
                if ckpt.name:
                    args.extend(["--checkpoint", ckpt.name])
            if self.warm_start_actor:
                args.append("--unitport_warm_start_actor")

        # Phase_1 of AMP_design.yaml §3: if the canvas wired a robot asset
        # through the phase_1 registry, forward it. The launcher uses the
        # resolved USD path to override env_cfg.scene.robot.spawn.usd_path.
        if self.robot_asset_id:
            from src.system.training.robot_assets import (
                rescan, resolve_for_training, RobotAssetValidationError,
            )
            rescan()  # ensure registry is populated
            try:
                usd_ref = resolve_for_training(self.robot_asset_id)
                args.extend([
                    "--unitport_robot_asset_id", self.robot_asset_id,
                    "--unitport_robot_usd", usd_ref,
                ])
            except RobotAssetValidationError as exc:
                # No USD source for this asset — abort here with a clear
                # error message instead of silently dropping the override
                # and letting the launcher crash deep inside PhysX with
                # FileNotFoundError on whatever stale string the canvas
                # had in il_robot_asset.usd_path.
                raise RuntimeError(
                    f"\n\n[UnitPort] Cannot launch Isaac Lab training — "
                    f"IL Robot Asset {self.robot_asset_id!r} has no USD "
                    f"source.\n\n"
                    f"  Reason: {exc}\n\n"
                    f"  Fix: pick a menagerie asset from the IL Robot Asset "
                    f"dropdown that ships a Nucleus USD URL (unitree_go2, "
                    f"unitree_a1, unitree_g1, unitree_h1, "
                    f"boston_dynamics_spot, …).\n\n"
                    f"  Archive assets like AMP_for_hardware_a1 / "
                    f"MetalHead_a1 only ship URDF/MJCF and CANNOT be used "
                    f"directly for Isaac Lab training without first "
                    f"converting their URDF to USD.\n"
                ) from exc

        # Phase_3 of AMP_design.yaml §4: when the canvas wired an
        # amp_ppo_trainer, the window layer sets self.algorithm to
        # "AMP_PPO" and populates self.amp_*. Forward all of it to the
        # launcher's AMP branch. The PPO branch default is preserved
        # when algorithm != "AMP_PPO".
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
            ])
            if self.amp_lerp_schedule:
                args.extend([
                    "--unitport_amp_lerp_schedule", str(self.amp_lerp_schedule),
                ])
            if self.amp_motion_files:
                args.extend([
                    "--unitport_amp_motion_files",
                    ",".join(str(p) for p in self.amp_motion_files),
                ])

        # Staged training pipeline (STAGE_NODE_DESIGN.yaml §2)
        if self.stage_schedule_json:
            import base64
            encoded = base64.b64encode(
                self.stage_schedule_json.encode("utf-8")
            ).decode("ascii")
            args.extend([
                "--unitport_stage_schedule", encoded,
                "--unitport_stage_checkpoint_strategy", self.stage_checkpoint_strategy,
            ])

        args.extend(self.extra_args)
        return self._build_launcher_cmd(self._train_script(), args)

    def build_remote_command(
        self,
        remote_launcher: str,
        remote_script: str,
        remote_log_dir: str = "",
        remote_config_file: str = "",
        remote_motion_files: Optional[List[str]] = None,
    ) -> str:
        """Build a shell command string for remote (Linux) execution via SSH.

        Bypasses the local ``_build_launcher_cmd`` / Windows .bat logic and
        produces a single ``bash -c '...'`` compatible command using the
        remote server's Isaac Lab launcher and paths.

        Parameters
        ----------
        remote_launcher:
            Absolute path to ``isaaclab.sh`` on the remote host.
        remote_script:
            Absolute path to the uploaded ``il_train_launcher.py`` on remote.
        remote_log_dir:
            Remote directory for training logs / checkpoints.
        remote_config_file:
            Remote path to the uploaded compiled @configclass Python file.
        remote_motion_files:
            Remote paths to AMP motion files (if any).
        """
        task = self.task_name
        if (remote_config_file or self.config_file) and not task.startswith("Isaac-"):
            task = "UnitPort-Custom-v0"

        args = [
            "--task", task,
            "--num_envs", str(self.num_envs),
            "--max_iterations", str(self.max_iterations),
            "--seed", str(self.seed),
        ]
        if remote_config_file:
            args.extend(["--unitport_config", remote_config_file])
        if remote_log_dir:
            args.extend(["--log_dir", remote_log_dir])

        # Always headless on remote servers.
        args.append("--headless")

        if self.resume_checkpoint:
            ckpt = self.resume_checkpoint
            args.extend(["--load_run", ckpt])
            if self.warm_start_actor:
                args.append("--unitport_warm_start_actor")

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
            ])
            if self.amp_lerp_schedule:
                args.extend(["--unitport_amp_lerp_schedule", str(self.amp_lerp_schedule)])
            if remote_motion_files:
                args.extend([
                    "--unitport_amp_motion_files",
                    ",".join(remote_motion_files),
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

        args.extend(self.extra_args)

        # Build: <launcher> -p <script> <args...>
        parts = [remote_launcher, "-p", remote_script] + args
        # Shell-quote each part for safe SSH exec.
        def _sq(s: str) -> str:
            if " " in s or "'" in s or '"' in s or "$" in s:
                return "'" + s.replace("'", "'\\''") + "'"
            return s

        return " ".join(_sq(p) for p in parts)

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
        """Build command to visually replay a trained policy (Isaac Sim viewport).

        Opens UnitPort's il_play_launcher with rendering enabled so the user
        can watch the robot execute the trained policy in real time. Only a
        single env is spawned — the review path is purely a visual sanity
        check, not a benchmark, so 50 parallel robots is overkill and just
        clutters the viewport.
        """
        args = [
            "--task", self.task_name,
            "--num_envs", "1",
            "--load_run", checkpoint_dir,
        ]
        return self._build_launcher_cmd(self._play_script(), args)
