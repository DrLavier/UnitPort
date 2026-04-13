#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior Node
Executes the trained policy of its bound checkpoint in MuJoCo simulation.

P3 (mission_design.yaml §5): sim_env is now a first-class input port.
StartNode constructs the default MjSimEnv, NodeExecutor threads it
through DFS, and BehaviorNode reads it from ``inputs["sim_env"]``.
The legacy BUG-1 fail-fast guard has been removed because the executor
also auto-injects sim_env from any upstream StartNode for legacy
canvases that lack an explicit edge.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.system.core.logger import log_error, log_info, log_warning

from .base_node import BaseNode


class BehaviorNode(BaseNode):
    """
    Behavior node that executes a checkpoint policy.

    Inputs
    ------
    - ``checkpoint`` : either a CheckpointNode dict, a CheckpointBundle
                       instance, or absent (in which case the node falls
                       back to its ``policy_id`` parameter).
    - ``sim_env``    : MjSimEnv (P3 — required port; provided by
                       StartNode through the canvas, with auto-injection
                       fallback for legacy canvases without explicit
                       wiring).

    Outputs
    -------
    - ``result``  : episode summary dict (status, policy_id, steps_run, …)
    - ``sim_env`` : the same MjSimEnv, threaded through so downstream
                    nodes can keep operating on the same world.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "behavior")
        self.inputs = {
            "checkpoint": None,
            "sim_env": None,    # P3: required input port
        }
        self.outputs = {
            "result": None,
            "sim_env": None,    # P3: thread sim_env downstream
        }
        self.parameters = {
            "policy_id": "",
            "command_scale": 1.0,
            "sim_env": None,    # legacy parameter-bridge fallback (P3)
        }

    @staticmethod
    def _resolve_command(bundle: Any, scale: float = 1.0) -> list[float]:
        raw_manifest = getattr(bundle, "raw_manifest", {}) or {}
        runtime_cfg = raw_manifest.get("runtime") or {}
        defaults = runtime_cfg.get("command_defaults") or {}
        if not isinstance(defaults, dict):
            return [0.0, 0.0, 0.0]
        try:
            safe_scale = float(scale)
            return [
                float(defaults.get("vx", 0.0) or 0.0) * safe_scale,
                float(defaults.get("vy", 0.0) or 0.0) * safe_scale,
                float(defaults.get("wz", 0.0) or 0.0) * safe_scale,
            ]
        except (TypeError, ValueError):
            return [0.0, 0.0, 0.0]

    @staticmethod
    def _resolve_max_steps(bundle: Any, sim_env: Any) -> int:
        raw_manifest = getattr(bundle, "raw_manifest", {}) or {}
        runtime_cfg = raw_manifest.get("runtime") or {}
        try:
            max_steps = int(runtime_cfg.get("max_steps", 0) or 0)
        except (TypeError, ValueError):
            max_steps = 0
        if max_steps > 0:
            return max_steps
        # Interactive viewer sessions should run until the user closes
        # the window.  Use the episode_length from env.yaml when
        # available, otherwise default to 30 minutes — long enough that
        # the user never hits the limit in practice.
        control_hz = float(getattr(sim_env, "control_frequency_hz", 50.0) or 50.0)
        render_enabled = bool(getattr(sim_env, "render_enabled", False))
        if render_enabled:
            episode_s = 30 * 60  # 30 minutes
            try:
                env_episode_s = float(
                    raw_manifest.get("runtime", {}).get("episode_length_s", 0) or 0
                )
                if env_episode_s > 0:
                    episode_s = max(episode_s, env_episode_s)
            except (TypeError, ValueError):
                pass
            return max(1, int(control_hz * episode_s))
        return max(1, int(control_hz * 20.0))

    def _failed_result(
        self,
        sim_env: Any,
        policy_id: str,
        reason: str,
        status: str = "failed",
        obs_remap_warnings: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Build a uniform failure result that still threads sim_env through.

        sim_env is preserved on the output port even on failure so
        downstream nodes (e.g. RobustnessReportNode in P5) can still
        observe / tear down the world they were given. obs_remap_warnings
        from the P4 embodiment matching pre-flight are also threaded
        through so the executor can populate diagnostics even when the
        load itself failed.
        """
        return {
            "result": {
                "status": status,
                "policy_id": policy_id,
                "steps_run": 0,
                "terminated": False,
                "engine_type": "",
                "termination_reason": reason,
                "obs_remap_warnings": obs_remap_warnings or [],
            },
            "sim_env": sim_env,
        }

    def _run_embodiment_matching(
        self,
        policy_id: str,
        bundle_path: str,
        sim_env: Any,
    ) -> tuple[list[str], str]:
        """P4 pre-flight: validate the actor + field against the bundle's
        SkillManifest.

        Returns
        -------
        ``(warnings, fatal_reason)``
            - ``warnings`` is the list of warning strings to surface in
              ``RuntimeResult.diagnostics`` regardless of outcome.
            - ``fatal_reason`` (P4-T2) is a non-empty short reason string
              when match_actor_to_manifest returns ``CompatStatus.FAIL``
              (e.g. action_dim mismatch). The caller MUST short-circuit
              with ``embodiment_match_failed: <reason>`` BEFORE invoking
              PolicyRunner.load when fatal_reason is non-empty.

        Field matcher FAILs are NOT fatal (currently field matcher only
        emits WARN-level issues — keeping the door open for stress-test
        runs against intentionally bad fields). Only actor matcher
        FAILs short-circuit.

        The function is best-effort: any failure to load the manifest
        is logged and treated as "no manifest available", meaning empty
        warnings + empty fatal_reason. P4 checks are skipped cleanly
        for bundles that genuinely lack a SkillManifest.
        """
        warnings: list[str] = []
        try:
            manifest = self._load_skill_manifest(policy_id, bundle_path)
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"BehaviorNode '{self.node_id}': could not load SkillManifest "
                f"for policy_id={policy_id!r} bundle={bundle_path!r}: {exc}"
            )
            return warnings, ""

        if manifest is None:
            return warnings, ""

        try:
            from src.system.policy.compatibility_checker import (
                CompatStatus,
                match_actor_to_manifest,
                match_field_to_manifest,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"BehaviorNode '{self.node_id}': compatibility_checker import "
                f"failed: {exc}"
            )
            return warnings, ""

        actor = getattr(sim_env, "actor", None)
        field_obj = getattr(sim_env, "field_spec", None)

        fatal_reason = ""
        try:
            actor_report = match_actor_to_manifest(actor, manifest)
            for issue in actor_report.issues:
                warnings.append(f"actor.{issue.code}: {issue.message}")
            if actor_report.status is CompatStatus.FAIL:
                fail_codes = [
                    i.code for i in actor_report.issues
                    if i.severity is CompatStatus.FAIL
                ]
                fatal_reason = ",".join(fail_codes) or "actor_match_fail"
                log_warning(
                    f"BehaviorNode '{self.node_id}': actor↔manifest match "
                    f"FAILED — {fail_codes} (short-circuiting before "
                    f"PolicyRunner.load per P4-T2)"
                )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"BehaviorNode '{self.node_id}': match_actor_to_manifest "
                f"raised: {exc}"
            )

        try:
            field_report = match_field_to_manifest(field_obj, manifest)
            for issue in field_report.issues:
                warnings.append(f"field.{issue.code}: {issue.message}")
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"BehaviorNode '{self.node_id}': match_field_to_manifest "
                f"raised: {exc}"
            )

        return warnings, fatal_reason

    # GLFW → Qt key-code mapping for the viewer keyboard bridge.
    # MuJoCo's passive viewer fires GLFW key codes; GlobalInputManager
    # expects Qt key codes.  Printable ASCII keys share the same values
    # (A-Z = 65-90, 0-9 = 48-57, Space = 32) so only the special keys
    # need an explicit translation entry.
    _GLFW_TO_QT_KEY = {
        265: 0x01000013,  # GLFW_KEY_UP    → Qt.Key_Up
        264: 0x01000015,  # GLFW_KEY_DOWN  → Qt.Key_Down
        263: 0x01000012,  # GLFW_KEY_LEFT  → Qt.Key_Left
        262: 0x01000014,  # GLFW_KEY_RIGHT → Qt.Key_Right
        340: 0x01000020,  # GLFW_KEY_LEFT_SHIFT  → Qt.Key_Shift
        344: 0x01000020,  # GLFW_KEY_RIGHT_SHIFT → Qt.Key_Shift
        256: 0x01000000,  # GLFW_KEY_ESCAPE → Qt.Key_Escape
        257: 0x01000004,  # GLFW_KEY_ENTER  → Qt.Key_Return
    }

    @staticmethod
    def _run_episode_with_viewer(
        runner: Any,
        sim_env: Any,
        max_steps: int,
        command: list[float],
        episode_timeout_sec: float = 600.0,
        step_telemetry_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        command_provider: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """Run a PolicyRunner episode with a passive MuJoCo viewer attached.

        Delegates to the shared ``run_policy_in_viewer`` in
        vis_check_runner so Mission and Training Review share the
        exact same daemon-thread + GLFW key bridge + auto-arm +
        no-early-termination pipeline.
        """
        from src.system.training.vis_check_runner import run_policy_in_viewer

        return run_policy_in_viewer(
            runner,
            sim_env,
            max_steps=max_steps,
            command=list(command),
            command_provider=command_provider,
            step_telemetry_fn=step_telemetry_fn,
            episode_timeout_sec=episode_timeout_sec,
        )

    @staticmethod
    def _load_skill_manifest(policy_id: str, bundle_path: str) -> Any:
        """Best-effort SkillManifest loader with two fallbacks.

        Resolution order:
          1. CheckpointRegistry.get(policy_id) — preferred for bundles
             that have been registered (training/exported/<id>/ or
             custom_mods/training/checkpoints/<id>/).
          2. Direct file load via :func:`load_skill_manifest` against
             ``bundle_path/manifest.yaml`` — covers ad-hoc / synthetic
             bundles that aren't in any registry (P4-T1 fixture, plus
             scripted callers that hand BehaviorNode an arbitrary
             on-disk path).
          3. ``None`` — caller treats this as "no manifest, skip P4
             checks". This must NEVER raise.
        """
        # (1) registry path — preferred when policy_id is known
        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry

            registry = CheckpointRegistry()
            entry = registry.get(policy_id) if policy_id else None
            if entry is not None:
                manifest = getattr(entry, "skill_manifest", None)
                if manifest is not None:
                    return manifest
        except Exception:
            pass

        # (2) direct file path fallback — used by P4-T1 synthetic bundle
        # and any other "I have a bundle on disk but it isn't registered"
        # caller. Wrapped in a try block so missing files / bad yaml
        # silently degrade to None instead of raising.
        try:
            if bundle_path:
                from pathlib import Path as _Path
                from src.system.skill.manifest_loader import load_skill_manifest

                bp = _Path(bundle_path)
                if bp.is_dir() and (bp / "manifest.yaml").is_file():
                    return load_skill_manifest(bp)
        except Exception:
            pass

        return None

    def _resolve_sim_env(self, inputs: Dict[str, Any]) -> Any:
        """Pull sim_env from the canvas port first, then the legacy
        parameter bridge as a backstop.
        """
        port_value = inputs.get("sim_env") if isinstance(inputs, dict) else None
        if port_value is not None:
            return port_value
        return self.get_parameter("sim_env", None)

    def _resolve_checkpoint(
        self,
        inputs: Dict[str, Any],
    ) -> tuple[Any, str, str, Dict[str, Any] | None]:
        """Walk the three legacy ``checkpoint`` input shapes.

        Returns ``(bundle, bundle_path, policy_id, error_result_or_None)``.
        ``error_result_or_None`` is non-None when the upstream
        CheckpointNode reported an error and the BehaviorNode should
        fail fast.
        """
        from src.system.policy.manifest_schema import CheckpointBundle

        checkpoint_in = inputs.get("checkpoint") if isinstance(inputs, dict) else None
        bundle: Any = None
        bundle_path = ""
        policy_id = ""

        if checkpoint_in is None:
            policy_id = self.get_parameter("policy_id", "")
        elif isinstance(checkpoint_in, dict):
            if checkpoint_in.get("status") == "error":
                return None, "", "", {
                    "status": "failed",
                    "policy_id": "",
                    "steps_run": 0,
                    "terminated": False,
                    "engine_type": "",
                    "termination_reason": "checkpoint_error",
                }
            bundle = checkpoint_in.get("bundle")
            bundle_path = str(checkpoint_in.get("bundle_path", "") or "")
            policy_id = str(checkpoint_in.get("policy_id", "") or "")
        elif isinstance(checkpoint_in, CheckpointBundle):
            bundle = checkpoint_in
            bundle_path = str(checkpoint_in.bundle_path)
            policy_id = checkpoint_in.name

        return bundle, bundle_path, policy_id, None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # ── Step 1: resolve sim_env (port first, parameter fallback) ───
        sim_env = self._resolve_sim_env(inputs)

        # ── Step 2: resolve checkpoint (3 legacy input shapes) ─────────
        bundle, bundle_path, policy_id, err = self._resolve_checkpoint(inputs)
        if err is not None:
            return {"result": err, "sim_env": sim_env}

        if not policy_id and not bundle_path and bundle is None:
            return self._failed_result(sim_env, "", "no_checkpoint")

        # ── Step 3: registry resolution if only policy_id given ────────
        if bundle is None and not bundle_path and policy_id:
            try:
                from src.system.service.checkpoint_registry import CheckpointRegistry

                bundle_path = str(CheckpointRegistry().get_bundle_path(policy_id))
            except Exception as exc:  # noqa: BLE001
                return self._failed_result(
                    sim_env, policy_id, f"registry_error: {exc}"
                )

        try:
            command_scale = float(self.get_parameter("command_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            command_scale = 1.0

        # ── Step 4: PRIMARY path — PolicyRunner against the live MjSimEnv
        if sim_env is not None and bundle_path:
            from src.system.policy.policy_runner import IncompatibleWeightError, PolicyRunner

            # P4 (mission_design.yaml §6) — embodiment matching pre-flight.
            # We resolve the bundle's SkillManifest, then run the actor /
            # field matchers BEFORE PolicyRunner.load(). Warnings are
            # collected and threaded through the result dict so the
            # executor can fold them into the diagnostics.
            #
            # P4-T2: actor matcher FAIL (e.g. action_dim mismatch) is a
            # hard pre-flight stop — we short-circuit BEFORE attempting
            # PolicyRunner.load to avoid the redundant load + cryptic
            # IncompatibleWeightError.
            obs_remap_warnings, embodiment_fatal = self._run_embodiment_matching(
                policy_id=policy_id,
                bundle_path=bundle_path,
                sim_env=sim_env,
            )
            if embodiment_fatal:
                return self._failed_result(
                    sim_env,
                    policy_id,
                    f"embodiment_match_failed: {embodiment_fatal}",
                    obs_remap_warnings=obs_remap_warnings,
                )

            runner = PolicyRunner()
            try:
                runner.load(Path(bundle_path), env=sim_env)
                bundle = bundle or getattr(runner, "_bundle", None)
            except IncompatibleWeightError as exc:
                return self._failed_result(
                    sim_env, policy_id, f"incompatible_bundle: {exc}",
                    obs_remap_warnings=obs_remap_warnings,
                )
            except Exception as exc:  # noqa: BLE001
                return self._failed_result(
                    sim_env, policy_id, f"load_failed: {exc}",
                    obs_remap_warnings=obs_remap_warnings,
                )

            command = self._resolve_command(bundle, scale=command_scale)
            max_steps = self._resolve_max_steps(bundle, sim_env)

            # ── F1: sim2sim mode detection & dispatch enforcement ──────
            # The user can ASK for sim2sim explicitly via the
            # ``execution_target`` node parameter (values: "mujoco_sim2sim"
            # or "mujoco"). When set, we hard-require the bundle to ship a
            # deploy_contract — failing loud is much better than silently
            # falling back to the legacy heuristic path. When unset, we
            # auto-detect: a bundle that has a contract IS in sim2sim mode.
            execution_target = str(
                self.get_parameter("execution_target", "") or ""
            ).strip().lower()
            has_contract = False
            try:
                has_contract = (bundle is not None and bundle.deploy_contract is not None)
            except Exception:
                # bundle.deploy_contract failed validation — compat checker
                # already raised, but defensive just in case.
                has_contract = False

            sim2sim_mode = (
                execution_target in ("mujoco_sim2sim", "mujoco") or has_contract
            )

            if execution_target in ("mujoco_sim2sim", "mujoco") and not has_contract:
                return self._failed_result(
                    sim_env,
                    policy_id,
                    (
                        f"sim2sim_requested_but_no_contract: "
                        f"execution_target={execution_target!r} requires the "
                        f"bundle to ship a deploy_contract section, but the "
                        f"loaded bundle has none. Re-import the bundle so the "
                        f"IL importer assembles the contract from env.yaml, "
                        f"or remove the execution_target parameter to use the "
                        f"legacy path."
                    ),
                    obs_remap_warnings=obs_remap_warnings,
                )

            # ── F3 wiring: per-step telemetry collector ────────────────
            # When sim2sim mode is active, build a tiny collector that
            # snapshots the latest step's obs/action/command and counts
            # the steps. The summary fields land in the result dict using
            # DiagnosticsKey constants so downstream consumers (UI live
            # charts, integration tests, mission_runtime_diagnostics) can
            # read them without parsing logs.
            sim2sim_telemetry_state: Dict[str, Any] = {
                "steps_healthy": 0,
                "last_obs": None,
                "last_action": None,
                "last_command": None,
                "last_step_index": -1,
            }

            step_telemetry_fn: Optional[Callable[[Dict[str, Any]], None]]
            if sim2sim_mode:
                from src.system.engine.contracts import DiagnosticsKey as _DK

                def _collect_step(payload: Dict[str, Any]) -> None:
                    sim2sim_telemetry_state["steps_healthy"] += 1
                    sim2sim_telemetry_state["last_step_index"] = payload.get(
                        _DK.MUJOCO_SIM2SIM_STEP_INDEX,
                        sim2sim_telemetry_state["last_step_index"],
                    )
                    if _DK.MUJOCO_SIM2SIM_LAST_OBS in payload:
                        sim2sim_telemetry_state["last_obs"] = payload[
                            _DK.MUJOCO_SIM2SIM_LAST_OBS
                        ]
                    if _DK.MUJOCO_SIM2SIM_LAST_ACTION in payload:
                        sim2sim_telemetry_state["last_action"] = payload[
                            _DK.MUJOCO_SIM2SIM_LAST_ACTION
                        ]
                    if _DK.MUJOCO_SIM2SIM_COMMAND in payload:
                        sim2sim_telemetry_state["last_command"] = payload[
                            _DK.MUJOCO_SIM2SIM_COMMAND
                        ]

                step_telemetry_fn = _collect_step
            else:
                step_telemetry_fn = None

            # ── Live input provider ─────────────────────────────────
            # When GlobalInputManager is armed (keyboard or gamepad),
            # create a command_provider so the policy receives live
            # velocity commands each tick instead of the static bundle
            # default.  Same pattern used by vis_check_runner.
            live_command_provider = None
            try:
                from src.system.runtime.global_input_manager import (
                    get_global_input_manager,
                )
                _gim = get_global_input_manager()
                if _gim.is_active():
                    live_command_provider = _gim.make_command_provider()
                    log_info(
                        f"BehaviorNode '{self.node_id}': live input ARMED "
                        f"({_gim.mode}) — feeding policy from CommandBus"
                    )
            except Exception:
                live_command_provider = None

            try:
                # Pick the run path based on the StartNode's
                # ``render_enabled`` flag, which StartNode stamps onto
                # the sim_env at construction time. Interactive Mission
                # canvases get a passive viewer + post-episode hold;
                # headless callers (tests, RobustnessMatrix, scripted
                # eval) get a plain run_episode call.
                if bool(getattr(sim_env, "render_enabled", False)):
                    episode = self._run_episode_with_viewer(
                        runner, sim_env, max_steps=max_steps, command=command,
                        step_telemetry_fn=step_telemetry_fn,
                        command_provider=live_command_provider,
                    )
                else:
                    episode = runner.run_episode(
                        sim_env,
                        max_steps=max_steps,
                        command=command,
                        render=True,
                        step_telemetry_fn=step_telemetry_fn,
                        command_provider=live_command_provider,
                    )
            except Exception as exc:  # noqa: BLE001
                return self._failed_result(
                    sim_env, policy_id, f"episode_failed: {exc}",
                    obs_remap_warnings=obs_remap_warnings,
                )

            result_dict: Dict[str, Any] = {
                "status": "success" if episode.success else "failed",
                "policy_id": policy_id,
                "steps_run": episode.steps_run,
                "terminated": episode.terminated,
                "engine_type": episode.engine_type,
                "termination_reason": episode.termination_reason,
                "obs_remap_warnings": obs_remap_warnings,
            }

            # Attach sim2sim diagnostics when the contract path was active.
            if sim2sim_mode:
                from src.system.engine.contracts import DiagnosticsKey as _DK
                result_dict[_DK.MUJOCO_SIM2SIM_STEPS_HEALTHY] = (
                    sim2sim_telemetry_state["steps_healthy"]
                )
                if sim2sim_telemetry_state["last_obs"] is not None:
                    result_dict[_DK.MUJOCO_SIM2SIM_LAST_OBS] = (
                        sim2sim_telemetry_state["last_obs"]
                    )
                if sim2sim_telemetry_state["last_action"] is not None:
                    result_dict[_DK.MUJOCO_SIM2SIM_LAST_ACTION] = (
                        sim2sim_telemetry_state["last_action"]
                    )
                if sim2sim_telemetry_state["last_command"] is not None:
                    result_dict[_DK.MUJOCO_SIM2SIM_COMMAND] = (
                        sim2sim_telemetry_state["last_command"]
                    )

            return {
                "result": result_dict,
                "sim_env": sim_env,
            }

        # ── Step 5: FALLBACK path — neutral_replay when sim_env is missing
        # but a bundle is on disk. Lets legacy callers (Training Ground
        # export review, scripts that don't go through the engine) keep
        # working without a StartNode-rooted MjSimEnv.
        replay_path = None
        if bundle_path:
            try:
                replay_path = Path(bundle_path)
            except Exception:
                replay_path = None

        if replay_path is not None and replay_path.exists():
            try:
                from src.system.training.vis_check_runner import _run_export_bundle_episode_nonblocking

                log_warning(
                    f"BehaviorNode '{self.node_id}': sim_env unavailable, "
                    f"falling back to neutral_replay for bundle "
                    f"'{replay_path.name}'. This path is for legacy / "
                    f"out-of-engine callers — Mission canvases should "
                    f"always provide sim_env via StartNode."
                )
                episode = _run_export_bundle_episode_nonblocking(
                    replay_path,
                    spec=None,
                    command_scale=command_scale,
                    log_fn=lambda message: log_info(f"BehaviorNode replay: {message}"),
                    neutral_replay=True,
                )
                return {
                    "result": {
                        "status": "success" if episode.success else "failed",
                        "policy_id": policy_id,
                        "steps_run": episode.steps_run,
                        "terminated": episode.terminated,
                        "engine_type": episode.engine_type,
                        "termination_reason": episode.termination_reason,
                    },
                    "sim_env": sim_env,
                }
            except Exception as exc:
                log_error(
                    f"BehaviorNode '{self.node_id}': neutral_replay fallback "
                    f"failed: {exc}"
                )
                return self._failed_result(
                    sim_env, policy_id, f"replay_failed: {exc}"
                )

        # Nothing to run with: no sim_env AND no on-disk bundle.
        return self._failed_result(
            sim_env, policy_id, "no_sim_env_and_no_bundle"
        )

    def get_display_name(self) -> str:
        return "Behavior"

    def get_description(self) -> str:
        return "Execute checkpoint policy in MuJoCo simulation"

    def get_input_ports(self) -> list[str]:
        return ["checkpoint", "sim_env"]

    def get_output_ports(self) -> list[str]:
        return ["result", "sim_env"]

    # ── Phase 7 A5: SkillManifest summary for Canvas display ──────────

    def skill_manifest_summary(self) -> Dict[str, Any]:
        """Return a summary dict of the bound checkpoint's SkillManifest.

        Used by Canvas UI to render manifest badges (action_space_type,
        frequency, pre/postcondition posture, compatibility status).

        Returns an empty dict if no manifest is available.
        """
        policy_id = self.get_parameter("policy_id", "")
        if not policy_id:
            return {}

        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry

            registry = CheckpointRegistry()
            entry = registry.get(policy_id)
            sm = entry.skill_manifest
            if sm is None:
                return {}

            return {
                "skill_id": getattr(sm, "skill_id", ""),
                "action_space_type": (
                    sm.action_space_type.value
                    if hasattr(sm.action_space_type, "value")
                    else str(sm.action_space_type)
                ),
                "control_frequency_hz": getattr(sm, "control_frequency_hz", 0.0),
                "precondition_posture": getattr(
                    getattr(sm, "precondition", None), "posture", "any"
                ),
                "postcondition_posture": getattr(
                    getattr(sm, "postcondition", None), "posture", "standing"
                ),
                "target_robot_family": getattr(sm, "target_robot_family", ""),
                "inference_backend": (
                    sm.inference_backend.value
                    if hasattr(sm.inference_backend, "value")
                    else str(sm.inference_backend)
                ),
            }
        except Exception:
            return {}
