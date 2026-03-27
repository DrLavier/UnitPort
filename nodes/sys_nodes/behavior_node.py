#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior Node
Executes the trained policy of its bound checkpoint in MuJoCo simulation.
"""

from pathlib import Path
from typing import Any, Dict

from bin.core.logger import log_error, log_info

from .base_node import BaseNode


class BehaviorNode(BaseNode):
    """
    Behavior node that executes a checkpoint policy.

    When a real exported bundle path is available, it prefers the same
    bundle-lineage replay path used by Export Review so Mission runtime
    behavior matches Training Ground replay more closely.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "behavior")
        self.inputs = {
            "checkpoint": None,
        }
        self.outputs = {
            "result": None,
        }
        self.parameters = {
            "policy_id": "",
            "command_scale": 1.0,
            "sim_env": None,
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
        control_hz = getattr(sim_env, "control_frequency_hz", 50.0)
        return max(1, int(float(control_hz) * 3.0))

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from system.policy.manifest_schema import CheckpointBundle

        checkpoint_in = inputs.get("checkpoint")
        bundle: Any = None
        bundle_path = ""
        policy_id = ""

        if checkpoint_in is None:
            policy_id = self.get_parameter("policy_id", "")
        elif isinstance(checkpoint_in, dict):
            if checkpoint_in.get("status") == "error":
                return {
                    "result": {
                        "status": "failed",
                        "policy_id": "",
                        "steps_run": 0,
                        "terminated": False,
                        "engine_type": "",
                        "termination_reason": "checkpoint_error",
                    }
                }
            bundle = checkpoint_in.get("bundle")
            bundle_path = str(checkpoint_in.get("bundle_path", "") or "")
            policy_id = str(checkpoint_in.get("policy_id", "") or "")
        elif isinstance(checkpoint_in, CheckpointBundle):
            bundle = checkpoint_in
            bundle_path = str(checkpoint_in.bundle_path)
            policy_id = checkpoint_in.name

        if not policy_id and not bundle_path and bundle is None:
            return {
                "result": {
                    "status": "failed",
                    "policy_id": "",
                    "steps_run": 0,
                    "terminated": False,
                    "engine_type": "",
                    "termination_reason": "no_checkpoint",
                }
            }

        sim_env = self.get_parameter("sim_env", None)
        if sim_env is None:
            replay_ready = False
            try:
                replay_ready = bool(bundle_path) and Path(bundle_path).exists()
            except Exception:
                replay_ready = False
            if not replay_ready:
                return {
                    "result": {
                        "status": "sim_env_required",
                        "policy_id": policy_id,
                        "steps_run": 0,
                        "terminated": False,
                        "engine_type": "",
                        "termination_reason": "sim_env_required",
                    }
                }

        if bundle is None and not bundle_path and policy_id:
            try:
                from system.service.checkpoint_registry import CheckpointRegistry

                bundle_path = str(CheckpointRegistry().get_bundle_path(policy_id))
            except Exception as exc:  # noqa: BLE001
                return {
                    "result": {
                        "status": "failed",
                        "policy_id": policy_id,
                        "steps_run": 0,
                        "terminated": False,
                        "engine_type": "",
                        "termination_reason": f"registry_error: {exc}",
                    }
                }

        try:
            command_scale = float(self.get_parameter("command_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            command_scale = 1.0

        replay_error = ""
        if bundle_path:
            try:
                replay_path = Path(bundle_path)
            except Exception:
                replay_path = None
            if replay_path is not None and replay_path.exists():
                try:
                    from system.training.vis_check_runner import _run_export_bundle_episode_nonblocking

                    log_info(
                        f"BehaviorNode replay start: policy={policy_id or replay_path.name} "
                        f"bundle={replay_path} scale={command_scale}"
                    )
                    episode = _run_export_bundle_episode_nonblocking(
                        replay_path,
                        spec=None,
                        command_scale=command_scale,
                        log_fn=lambda message: log_info(f"BehaviorNode replay: {message}"),
                    )
                    log_info(
                        f"BehaviorNode replay end: policy={policy_id or replay_path.name} "
                        f"status={'success' if episode.success else 'failed'} "
                        f"steps={episode.steps_run} reason={episode.termination_reason}"
                    )
                    return {
                        "result": {
                            "status": "success" if episode.success else "failed",
                            "policy_id": policy_id,
                            "steps_run": episode.steps_run,
                            "terminated": episode.terminated,
                            "engine_type": episode.engine_type,
                            "termination_reason": episode.termination_reason,
                        }
                    }
                except Exception as exc:
                    replay_error = str(exc)
                    log_error(
                        f"BehaviorNode replay failed: policy={policy_id or replay_path.name} "
                        f"bundle={replay_path} error={exc}"
                    )

        # BUG-1 guard: if no sim_env was injected, PolicyRunner cannot run.
        # The only execution path that could have succeeded was replay above;
        # if it failed (or bundle_path was empty), we must stop here.
        if sim_env is None:
            return {
                "result": {
                    "status": "failed",
                    "policy_id": policy_id,
                    "steps_run": 0,
                    "terminated": False,
                    "engine_type": "",
                    "termination_reason": (
                        f"replay_failed: {replay_error}" if replay_error else "sim_env_required"
                    ),
                }
            }

        from system.policy.policy_runner import IncompatibleWeightError, PolicyRunner

        runner = PolicyRunner()
        try:
            runner.load(Path(bundle_path), env=sim_env)
            bundle = bundle or getattr(runner, "_bundle", None)
        except IncompatibleWeightError as exc:
            suffix = f"; replay_failed: {replay_error}" if replay_error else ""
            return {
                "result": {
                    "status": "failed",
                    "policy_id": policy_id,
                    "steps_run": 0,
                    "terminated": False,
                    "engine_type": "",
                    "termination_reason": f"incompatible_bundle: {exc}{suffix}",
                }
            }
        except Exception as exc:  # noqa: BLE001
            suffix = f"; replay_failed: {replay_error}" if replay_error else ""
            return {
                "result": {
                    "status": "failed",
                    "policy_id": policy_id,
                    "steps_run": 0,
                    "terminated": False,
                    "engine_type": "",
                    "termination_reason": f"load_failed: {exc}{suffix}",
                }
            }

        command = self._resolve_command(bundle, scale=command_scale)
        max_steps = self._resolve_max_steps(bundle, sim_env)

        try:
            episode = runner.run_episode(
                sim_env,
                max_steps=max_steps,
                command=command,
                render=True,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "result": {
                    "status": "failed",
                    "policy_id": policy_id,
                    "steps_run": 0,
                    "terminated": False,
                    "engine_type": "",
                    "termination_reason": f"episode_failed: {exc}",
                }
            }

        return {
            "result": {
                "status": "success" if episode.success else "failed",
                "policy_id": policy_id,
                "steps_run": episode.steps_run,
                "terminated": episode.terminated,
                "engine_type": episode.engine_type,
                "termination_reason": episode.termination_reason,
            }
        }

    def get_display_name(self) -> str:
        return "Behavior"

    def get_description(self) -> str:
        return "Execute checkpoint policy in MuJoCo simulation"
