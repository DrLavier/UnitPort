# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacLabConfig — pure data describing one Isaac Lab training run.

Ported from DEMO ``src/system/training/isaac_lab_config.py`` with two
RELEASE-specific changes:

1. ``isaac_lab_path`` / ``isaac_lab_python`` / ``isaac_lab_launcher``
   default to the user's **active selection**
   (``application.service.engines.service.resolve_active_local_isaac`` —
   the user.ini[Training].local_isaac_root pin, with loud fallback to base).
   The launcher must start the install the user picked, not the
   auto-discovered base. Callers can override per-run.

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


def _flatten_clip_paths_ranges(clip_paths: dict, clip_ranges: dict):
    """Flatten ``{item_id: path}`` (+ ``{item_id: (lo, hi)}``) into two
    positionally-aligned lists ``(files, ranges)``.

    Empty paths are skipped in BOTH lists together (so files ↔ ranges stay
    aligned across the item_id-key-dropping flatten). Each range entry is
    ``"lo:hi"`` for a segment or ``"-"`` for a whole file. This is the single
    place the item_id keying is lost, so the range MUST be captured here in
    lockstep — it cannot be re-derived downstream.
    """
    files: List[str] = []
    ranges: List[str] = []
    for item_id, p in clip_paths.items():
        if not p:
            continue
        files.append(str(p))
        rng = (clip_ranges or {}).get(item_id)
        ranges.append(f"{int(rng[0])}:{int(rng[1])}" if rng else "-")
    return files, ranges


def _amp_motion_cli_args(files: List[str], ranges: List[str]) -> List[str]:
    """Build the ``--unitport_amp_motion_files`` (+ optional ``_ranges``) args.

    The range list rides positionally with the file list. The ranges arg is
    emitted only when at least one file is a segment (any entry != ``"-"``), so
    whole-file-only runs stay byte-identical to the pre-segment CLI. Shared by
    both emission sites (AMP_PPO path + PPO reference_frame_0 RSI path) so they
    can never drift.
    """
    out: List[str] = []
    if not files:
        return out
    out.extend(["--unitport_amp_motion_files", ",".join(str(p) for p in files)])
    if len(ranges) == len(files) and any(r != "-" for r in ranges):
        out.extend(["--unitport_amp_motion_ranges", ",".join(ranges)])
    return out


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
    # Positionally aligned to ``amp_motion_files``: each entry is either
    # ``"lo:hi"`` (an inclusive frame sub-range → the launcher slices that clip
    # to a segment) or ``"-"`` (whole file). Built in lockstep with
    # ``amp_motion_files`` from ``motion_ref.clip_ranges`` because the clip-path
    # dict is flattened to a bare list here (item_id keys are dropped), so the
    # range MUST ride positionally — it cannot be re-keyed downstream. Crosses
    # to the Kit worker as a plain ``--unitport_amp_motion_ranges`` CLI arg.
    amp_motion_ranges: List[str] = field(default_factory=list)
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
    # Stage 3 per-format schema: launcher receives the spec's active
    # format + a JSON {ir_role: joint_name} table for the active format
    # so RSI can map clip IR roles directly to physical joint names
    # without the body+suffix heuristic (which fails on Spot etc.).
    active_format: str = ""
    joint_names_by_ir_json: str = ""
    # Canonical SKU + primary family for the canvas-selected robot.
    # ``robot_sku`` is ``registers.robots.RobotSpec.sku`` (the
    # ``build_sku(brand, model)`` hash). ``robot_family`` is the documented
    # ``families[0]`` primary family (see ``registers/families.py`` —
    # ``families[0] = primary_family``). Both travel to the launcher as
    # explicit subprocess flags so the launcher's USD / MJCF / RSI hooks
    # resolve identity through the registry without re-reading TrainingSpec.
    robot_sku: str = ""
    robot_family: str = ""
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

        Resolves the install root via the **user's active selection**
        (``application.service.engines.service.resolve_active_local_isaac`` —
        user.ini[Training].local_isaac_root pin → base → first → loud fallback),
        then derives launcher (``isaaclab.bat|sh``) and python paths under it.

        This MUST route through the same resolver the User panel writes:
        the training launcher has to start the Isaac install the user picked,
        not the auto-discovered base. The prior ``_detect_isaac_lab()`` path
        read only ``backends_installed.json::local_root`` and silently launched
        the base regardless of the user's choice (CLAUDE.md §8 — surface the
        selection, never silently wrong).
        """
        from registers.backends import _find_isaac_python
        from application.service.engines.service import resolve_active_local_isaac

        active = resolve_active_local_isaac()
        if not active or not str(active.get("root") or "").strip():
            raise RuntimeError(
                "Isaac Lab not registered — register an install via the User "
                "panel (EngineService.register_isaac_local(<root>)) or "
                "import_isaac_lab_path_from_demo() first"
            )
        if not active.get("exists", True):
            raise RuntimeError(
                f"Selected Isaac Lab root {active.get('root')!r} no longer has "
                "markers (isaaclab.sh|isaaclab.bat + source/). Re-select an "
                "install in the User panel before training."
            )
        root = Path(str(active["root"]))
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
            robot.sku                         -> robot_sku
            robot.families[0]                 -> robot_family
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
        if algo is None:
            raise ValueError(
                "[IsaacLabConfig.from_training_spec] spec.algorithm is None — "
                "the canvas algorithm_config node is missing. Refusing to "
                "substitute PPO + seed=42 + max_iters=1500 defaults "
                "(CLAUDE.md §1.8); fix the canvas wiring first."
            )
        il_ppo = getattr(algo, "il_ppo", None)
        if il_ppo is None:
            raise ValueError(
                "[IsaacLabConfig.from_training_spec] spec.algorithm.il_ppo "
                "is None — the canvas il_ppo / il_ppo_trainer config is "
                "missing. Refusing to substitute max_iters=1500 / "
                "headless=True defaults (CLAUDE.md §1.8)."
            )

        # CLAUDE.md §1.8: n_envs / seed / max_iters are training-corrupting
        # when wrong — fail loud rather than substitute Go2-class defaults
        # (4096 envs / seed 42 / 1500 iters). The previous chained
        # `getattr(..., default) or default` masked missing canvas wiring.
        env_ref = getattr(spec_obj, "env", None)
        if env_ref is None:
            raise ValueError(
                "[IsaacLabConfig.from_training_spec] spec.env is None — "
                "the canvas env_assembler node is missing. n_envs cannot "
                "be inferred; refusing 4096 default (CLAUDE.md §1.8)."
            )
        n_envs_raw = getattr(env_ref, "n_envs", None)
        if not n_envs_raw or int(n_envs_raw) <= 0:
            raise ValueError(
                f"[IsaacLabConfig.from_training_spec] spec.env.n_envs="
                f"{n_envs_raw!r} is missing or non-positive. Fix the "
                f"env_assembler node's n_envs ParamSpec."
            )
        n_envs = int(n_envs_raw)

        seed_raw = getattr(algo, "seed", None)
        if seed_raw is None:
            raise ValueError(
                "[IsaacLabConfig.from_training_spec] spec.algorithm.seed is "
                "missing — required for run reproducibility. Set the "
                "algorithm_config node's seed parameter."
            )
        seed = int(seed_raw)

        max_iters_raw = getattr(il_ppo, "max_iterations", None)
        if not max_iters_raw or int(max_iters_raw) <= 0:
            raise ValueError(
                f"[IsaacLabConfig.from_training_spec] spec.algorithm.il_ppo."
                f"max_iterations={max_iters_raw!r} is missing or non-positive. "
                f"Training has no stopping condition; set this on the "
                f"il_ppo_trainer node."
            )
        max_iters = int(max_iters_raw)

        # headless is a UI/display preference; True is a sane fallback
        # (training subprocess defaults to headless, the canvas
        # il_ppo_trainer node's ``headless`` ParamSpec ships True). Keep
        # the default but coerce explicitly so the field is always bool.
        # WHY KEPT: not a data-corrupting field; default matches the
        # ParamSpec ship value, so the substitution is visible at the
        # canvas-edit level.
        headless = bool(getattr(il_ppo, "headless", True))

        training_mode_raw = getattr(algo, "training_mode", None)
        if not training_mode_raw:
            raise ValueError(
                "[IsaacLabConfig.from_training_spec] spec.algorithm.training_mode "
                "is missing — must be 'PPO' or 'AMP_PPO'. Fix algorithm_config "
                "node."
            )
        training_mode = str(training_mode_raw).strip().upper()
        # IsaacLabConfig.algorithm vocabulary: lower-case "ppo"/"sac" or the
        # exact string "AMP_PPO" — see build_command()'s gating.
        if training_mode == "AMP_PPO":
            algorithm = "AMP_PPO"
        else:
            algorithm_raw = getattr(algo, "algorithm", None)
            if not algorithm_raw:
                raise ValueError(
                    "[IsaacLabConfig.from_training_spec] spec.algorithm.algorithm "
                    "is missing — required to pick the specific algorithm "
                    "variant (e.g. 'PPO', 'SAC'). Fix algorithm_config node."
                )
            algorithm = str(algorithm_raw).strip().lower()
            if not algorithm:
                raise ValueError(
                    "[IsaacLabConfig.from_training_spec] spec.algorithm.algorithm "
                    "is empty after strip — fix algorithm_config node."
                )

        cfg = cls.from_registers(
            task_name=task_name,
            num_envs=n_envs,
            max_iterations=max_iters,
            seed=seed,
            headless=headless,
            algorithm=algorithm,
        )

        # Always forward robot SKU + primary family + init_pose mode —
        # non-default values are inert when ``unitport_launcher_path`` is
        # empty (build_command gates them behind the launcher) but get
        # carried through once the launcher is registered, so the canvas
        # configuration is preserved.
        # ``families[0] = primary_family`` is the documented contract
        # surfaced by ``registers/families.py`` (used by ~8 production
        # sites: env_cfg_compiler, spec_compiler, pd_preview, body_ir,
        # …); reading index 0 here is the single-source path, not a
        # heuristic. None-guarded so non-Play inspection paths
        # (``spec.robot is None``) leave the fields empty rather than
        # raising on ``[0]`` of an absent list.
        robot_ref = getattr(spec_obj, "robot", None)
        cfg.robot_sku = str(getattr(robot_ref, "sku", "") or "")
        _families = getattr(robot_ref, "families", None) if robot_ref is not None else None
        cfg.robot_family = (
            str(_families[0]) if _families else ""
        )

        # Body mapping (Robot node body_mapping) must travel to the
        # launcher whenever init_pose_mode=reference_frame_0 or any other
        # path needs IR-role → physical-joint resolution for the active
        # robot. Previously this field defaulted to "" and the launcher's
        # RSI branch silently skipped applying the reference frame.
        # Stage 4: in addition to the body_role_map (joint→ir_role
        # legacy mapping), we now serialise the active format and an
        # IR-role-keyed joint name table so the launcher can resolve
        # ``clip_ir_role[i]`` → physical joint name directly, bypassing
        # the body+suffix heuristic that fails on Spot.
        body_role_map = getattr(robot_ref, "body_role_map", None) if robot_ref is not None else None
        if isinstance(body_role_map, dict) and body_role_map:
            cfg.body_mapping_json = json.dumps(body_role_map)

        if robot_ref is not None:
            cfg.active_format = str(getattr(robot_ref, "active_format", "") or "")
            # Build {ir_role: joint_name} for the active format. Uses
            # RobotSpecRef.joints_role_map_for which returns
            # {joint_name: ir_role}; we invert here.
            if cfg.active_format and hasattr(robot_ref, "joints_role_map_for"):
                joint_role_map = robot_ref.joints_role_map_for(cfg.active_format)
                if joint_role_map:
                    by_ir = {ir: jn for jn, ir in joint_role_map.items() if ir}
                    cfg.joint_names_by_ir_json = json.dumps(by_ir)

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
        # malformed dataclass instance). The H0 guard re-validates the
        # populated payload before it crosses the IPC boundary so a stale
        # or hand-mutated dataclass cannot inject a non-default stage role
        # / obs_profile / init_from / distill_loss past the compiler check.
        stage_sched = getattr(spec_obj, "stage_schedule", None)
        if stage_sched is not None:
            from application.training.training_spec import (
                validate_stage_schedule_dict_h0,
            )
            stage_sched_dict = asdict(stage_sched)
            validate_stage_schedule_dict_h0(stage_sched_dict)
            cfg.stage_schedule_json = json.dumps(stage_sched_dict)
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
                # Flatten clip_paths (item_id → path) to a bare list AND build a
                # positionally-aligned range list in the SAME pass, so a
                # segment's [lo,hi] stays matched to its file across the
                # item_id-key-dropping boundary here (and the later CLI hop).
                _all_clip_ranges = getattr(motion_ref_all, "clip_ranges", None) or {}
                cfg.amp_motion_files, cfg.amp_motion_ranges = (
                    _flatten_clip_paths_ranges(_all_clip_paths, _all_clip_ranges)
                )

        # AMP_PPO-only payload (silent on non-AMP runs).
        if algorithm == "AMP_PPO":
            il = getattr(spec_obj, "il", None)
            amp = getattr(il, "amp", None) if il is not None else None
            motion_ref = getattr(il, "motion_ref", None) if il is not None else None
            # CLAUDE.md §1.8: AMP_PPO requires both motion_ref AND amp
            # configs wired on the canvas. Substituting Go2-class defaults
            # (50 Hz motion / 2.0 reward coef / 10.0 grad penalty / etc.)
            # silently corrupts the discriminator when the user intended
            # different hyperparams.
            if motion_ref is None:
                raise ValueError(
                    "[IsaacLabConfig.from_training_spec] AMP_PPO requires "
                    "spec.algorithm.il.motion_ref but got None — the "
                    "canvas training_motion node is missing. Refusing to "
                    "substitute 50 Hz default."
                )
            motion_fps_raw = getattr(motion_ref, "motion_fps", None)
            if not motion_fps_raw or float(motion_fps_raw) <= 0:
                raise ValueError(
                    f"[IsaacLabConfig.from_training_spec] motion_ref.motion_fps"
                    f"={motion_fps_raw!r} is missing or non-positive. The "
                    f"discriminator's (s,s') transition gap is derived from "
                    f"this; refusing 50 Hz default."
                )
            motion_fps = float(motion_fps_raw)
            cfg.amp_transition_dt = 1.0 / motion_fps
            cfg.amp_phase_mode = str(getattr(motion_ref, "phase_mode", "loop") or "loop")
            cfg.amp_random_start_phase = bool(
                getattr(motion_ref, "random_start_phase", True)
            )
            # AMP RSI — thread training_motion's reference-state-init config so
            # the launcher builds + registers the "default" RSI pool
            # (--unitport_amp_rsi_prob > 0 is the gate at _build_launcher_cmd).
            # CLAUDE.md §1.11: env_cfg_compiler already emits the
            # reset_from_reference_motion EventTerm (pool_id="default") from the
            # SAME canvas fields; without threading the launcher half the pool
            # is never populated → "pool 'default' not registered" WARN and RSI
            # is silently skipped every reset (the §1.8 illusion-of-success the
            # user hit). Both halves now read the one canvas source via the spec.
            cfg.amp_rsi_enabled = bool(
                getattr(motion_ref, "reference_state_init_enabled", False)
            )
            cfg.amp_rsi_prob = float(getattr(motion_ref, "rsi_prob", 0.0) or 0.0)
            cfg.amp_rsi_joint_noise = float(
                getattr(motion_ref, "rsi_joint_noise", 0.02) or 0.02
            )
            if amp is None:
                raise ValueError(
                    "[IsaacLabConfig.from_training_spec] AMP_PPO requires "
                    "spec.algorithm.il.amp but got None — the canvas "
                    "amp_helper / discriminator node is missing. Refusing "
                    "to substitute Go2-class AMP hyperparams (reward_coef=2, "
                    "grad_penalty=10, etc.)."
                )

            def _required_amp_field(field_name: str, caster):
                val = getattr(amp, field_name, None)
                if val is None:
                    raise ValueError(
                        f"[IsaacLabConfig.from_training_spec] AMP_PPO "
                        f"requires spec.algorithm.il.amp.{field_name} but "
                        f"the field is None. Fix the canvas amp_helper "
                        f"node's ParamSpec (CLAUDE.md §1.8)."
                    )
                try:
                    return caster(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"[IsaacLabConfig.from_training_spec] amp.{field_name}="
                        f"{val!r} is not coercible to {caster.__name__}."
                    ) from exc

            cfg.amp_reward_coef = _required_amp_field("amp_reward_coef", float)
            cfg.amp_task_reward_lerp = _required_amp_field("task_reward_lerp", float)
            cfg.amp_disc_grad_penalty = _required_amp_field("disc_grad_penalty", float)
            cfg.amp_disc_label_smoothing = _required_amp_field("disc_label_smoothing", float)
            cfg.amp_replay_buffer_size = _required_amp_field("amp_replay_buffer_size", int)
            cfg.amp_num_preload_transitions = _required_amp_field("num_preload_transitions", int)
            cfg.amp_disc_lr = _required_amp_field("disc_lr", float)
            cfg.amp_lerp_schedule = str(getattr(amp, "lerp_schedule", "") or "")
            disc = getattr(amp, "disc", None)
            if disc is not None:
                # disc subblock fields stay getattr-with-default since the
                # disc subnode is itself optional (default behaviour =
                # canonical clamps when no discriminator node is dropped).
                # These ARE clamp/limit safety values, not training-corrupting
                # hyperparams; keep current behavior.
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

    # ------------------------------------------------------------------
    # Subprocess command builder (verbatim from DEMO except for
    # _train_script default + the robot_sku / robot_family branches
    # which are gated behind the UnitPort launcher because the stock
    # train.py has no --unitport_robot_sku / --unitport_robot_family
    # flags)
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
        """Create a temp .bat that invokes the registered Isaac Lab Python
        with proper quoting.

        Uses ``self.isaac_lab_python`` — resolved by
        ``registers.backends._find_isaac_python`` at ``from_registers()`` time,
        which checks ``_isaac_sim/``, ``.venv/Scripts/``, and ``CONDA_PREFIX``
        in that order. Baking the already-resolved path in (instead of
        re-discovering inside the .bat) avoids the silent fallback to
        ``where python`` → system Python that occurs for pip-installed Isaac
        Lab installs where ``_isaac_sim/`` does not exist (per CLAUDE.md §1.8
        "no silent fallbacks — fail loud, always").
        """
        import tempfile

        if not self.isaac_lab_python or not Path(self.isaac_lab_python).exists():
            raise RuntimeError(
                "[IsaacLabConfig] cannot launch Isaac Lab: isaac_lab_python "
                f"is unset or missing (got {self.isaac_lab_python!r}). "
                "Re-register the Isaac Lab install via "
                "EngineService.register_isaac_local(<root>) — "
                "registers.backends._find_isaac_python must locate "
                "_isaac_sim/python.bat OR .venv/Scripts/python.exe OR "
                "CONDA_PREFIX/python.exe under the root."
            )

        python_exe = self.isaac_lab_python.replace("/", "\\")
        # Write the args to a sibling response file (one token per line) and
        # pass ``@<argsfile>`` as a single token to python. The launcher's
        # ArgumentParser is constructed with ``fromfile_prefix_chars='@'``
        # (il_train_launcher.py) so argparse expands the file contents in
        # place. This bypasses cmd.exe's 8191-char-per-line limit, which
        # truncated AMP runs mid-arg (large stage_schedule base64 +
        # per-item amp_task_items + N motion file paths easily push the
        # single-line invocation past 8 KB and corrupt argv silently).
        fd_args, args_path = tempfile.mkstemp(
            suffix=".args.txt", prefix="unitport_il_run_"
        )
        os.close(fd_args)
        Path(args_path).write_text(
            "\n".join(args) + "\n", encoding="utf-8"
        )
        # ``@`` token must not be parsed as a cmd metacharacter — wrap the
        # whole token in quotes (covers spaces in %TEMP% paths too).
        argsfile_token = f'"@{args_path}"'
        content = (
            "@echo off\r\n"
            "setlocal EnableExtensions EnableDelayedExpansion\r\n"
            f'set "python_exe={python_exe}"\r\n'
            'if not exist "!python_exe!" (\r\n'
            f'    echo [ERROR] Isaac Python vanished: !python_exe!\r\n'
            '    exit /b 1\r\n'
            ')\r\n'
            'echo [INFO] Using python from: !python_exe!\r\n'
            f'echo [INFO] Args file: {args_path}\r\n'
            f'"!python_exe!" "{script}" {argsfile_token}\r\n'
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
            if self.robot_sku:
                args.extend(["--unitport_robot_sku", self.robot_sku])
            if self.robot_family:
                args.extend(["--unitport_robot_family", self.robot_family])

            # Headed run → hand the launcher the install root so it can
            # generate a minimal viewport-only experience under <root>/apps/
            # and select it, avoiding the full-GUI ``omni.kit.menu.utils``
            # crash and cutting GPU BAR1 pressure. The launcher itself gates
            # on ``not --video``. Headless runs pass nothing (no viewport).
            if not self.headless and self.isaac_lab_path:
                args.extend(
                    ["--unitport_headed_experience", self.isaac_lab_path]
                )

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
                args.extend(_amp_motion_cli_args(
                    self.amp_motion_files, self.amp_motion_ranges
                ))
                # AMP obs term list is resolved HERE (app-side: this builder
                # runs in .venv311, which has registers + PyQt6) and ALWAYS
                # threaded to the launcher. The Isaac Kit venv has NO PyQt6, so
                # the launcher MUST NOT import registers.amp_obs_templates to
                # derive a family default — registers/__init__ pulls
                # unitport_sdk → logger → PyQt6 and the worker dies with
                # ModuleNotFoundError before training (CLAUDE.md §1.11: resolve
                # at the boundary that owns the dependency, not the Kit side).
                # Canvas discriminator override wins; empty → family default.
                _amp_obs_fields = str(self.amp_obs_fields or "").strip()
                if not _amp_obs_fields:
                    if not self.robot_family:
                        raise ValueError(
                            "[isaac_lab/config] cannot resolve the AMP obs "
                            "template: discriminator.amp_obs_fields is empty "
                            "AND robot_family is unset. Fix the canvas Robot "
                            "node (family must resolve) or set the "
                            "discriminator node's amp_obs_fields explicitly."
                        )
                    from registers.amp_obs_templates import get_obs_template
                    _amp_obs_fields = ",".join(
                        get_obs_template(self.robot_family).template_terms
                    )
                args.extend(["--unitport_amp_obs_fields", _amp_obs_fields])
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

            # Active format + IR-role→joint-name table. Threaded whenever
            # available (NOT only under a custom init_pose_mode): the AMP
            # env-side joint permutation for humanoids resolves each IR role
            # to its physical joint name via this table
            # (obs_terms.set_canonical_joint_perm_override) — without it a
            # G1 AMP run falls back to using the IR role slug AS the joint
            # name, which never matches `left_hip_pitch_joint` etc. and the
            # humanoid perm raises at step 0. Inert for plain-PPO / quadruped
            # runs (the AMP obs path never consults the override there).
            if self.active_format:
                args.extend(["--unitport_active_format", self.active_format])
            if self.joint_names_by_ir_json:
                import base64
                encoded = base64.b64encode(
                    self.joint_names_by_ir_json.encode("utf-8")
                ).decode("ascii")
                args.extend(["--unitport_joint_names_by_ir", encoded])

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
                    args.extend(_amp_motion_cli_args(
                        self.amp_motion_files, self.amp_motion_ranges
                    ))

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

    # Note: ``build_export_command`` and ``build_review_command`` were
    # removed (CLAUDE.md §1.8/§1.9). The legacy export path spawned a
    # second Isaac Sim subprocess via ``il_play_launcher.py`` to write
    # ``policy.onnx`` as a side-effect; that path was redundant
    # (bundle_finalizer converts ``.pt`` to ONNX in-process via
    # ``export_rsl_rl_actor_to_onnx``) and a documented source of RTX
    # scenedb crashes. The review path now goes through bundle-only
    # ``IsaacSimReviewTask`` (see
    # ``application/service/runtime/simulation/isaac_sim/review_session.py``)
    # which reads ONNX + deploy_contract from the bundle directly,
    # honouring CLAUDE.md §1.9 portable-artifact rule (no reach-back
    # into local-only run_dir state).
