"""ReactiveLocomotionNode — Canvas node for reactive locomotion control.

A specialized BehaviorNode that enters a ReactiveLoop when executed.
The policy runs continuously, receiving real-time velocity commands
from a gamepad, keyboard, or other input source via CommandBus.

Properties (visible in Canvas inspector):
  - checkpoint: reference to CheckpointRegistry entry
  - command_source: gamepad | keyboard | canvas_input | api
  - termination_condition: manual_stop | timeout
  - timeout_seconds: float (when termination=timeout)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base_node import BaseNode

log = logging.getLogger(__name__)


class ReactiveLocomotionNode(BaseNode):
    """Reactive locomotion node — enters real-time control loop on execute().

    Unlike sequential BehaviorNode (which runs for N steps and returns),
    this node blocks until an external termination signal is received.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "reactive_loco")
        self.inputs = {
            "checkpoint": None,
        }
        self.outputs = {
            "result": None,
        }
        self.parameters = {
            "policy_id": "",
            "command_source": "gamepad",      # gamepad | keyboard | canvas_input | api
            "input_profile": "quadruped_gamepad",
            "termination_condition": "manual_stop",  # manual_stop | timeout
            "timeout_seconds": 60.0,
            "sim_env": None,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reactive locomotion loop.

        Loads the checkpoint, sets up CommandBus + input source,
        creates ReactiveLoop, and blocks until termination.
        """
        try:
            return self._execute_impl(inputs)
        except Exception as exc:
            log.exception("ReactiveLocomotionNode execution failed")
            return {"result": {"status": "error", "error": str(exc)}}

    def _execute_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from src.system.runtime.command_bus import CommandBus
        from src.system.engine.reactive_loop import ReactiveLoop
        from src.system.policy.obs_assembler import ObsAssembler
        from src.system.skill.manifest_loader import load_skill_manifest

        # --- Resolve checkpoint ---
        checkpoint_in = inputs.get("checkpoint")
        bundle_path = ""
        if isinstance(checkpoint_in, dict):
            bundle_path = checkpoint_in.get("bundle_path", "")
        elif checkpoint_in is not None:
            bundle_path = str(getattr(checkpoint_in, "bundle_path", ""))

        if not bundle_path:
            policy_id = self.get_parameter("policy_id", "")
            if policy_id:
                try:
                    from src.system.service.checkpoint_registry import CheckpointRegistry
                    registry = CheckpointRegistry()
                    entry = registry.get(policy_id)
                    bundle_path = str(entry.bundle_path)
                except Exception as exc:
                    return {"result": {"status": "error", "error": f"Checkpoint not found: {exc}"}}

        if not bundle_path:
            return {"result": {"status": "error", "error": "No checkpoint specified"}}

        # --- Load manifest ---
        from pathlib import Path
        manifest = load_skill_manifest(Path(bundle_path))

        if manifest.execution_mode != "reactive":
            log.warning(
                "Checkpoint %s has execution_mode='%s', expected 'reactive'. Proceeding anyway.",
                bundle_path, manifest.execution_mode,
            )

        # --- Set up inference engine ---
        inference = self._create_inference(Path(bundle_path), manifest)
        if inference is None:
            return {"result": {"status": "error", "error": "Failed to create inference engine"}}

        # --- Set up components ---
        command_bus = CommandBus()
        obs_assembler = ObsAssembler(manifest)

        # sim_env must be resolved BEFORE building the action applier so
        # we can construct mj_qpos / mj_ctrl JointSpaces from the live
        # mj_model. Without those spaces, ``apply_with_pd`` falls through
        # to the warning branch and IL torques are silently miscomputed
        # (the "Mission canvas IL replay flailing" symptom).
        sim_env = self.get_parameter("sim_env")
        if sim_env is None:
            return {"result": {"status": "error", "error": "No sim_env provided"}}

        # Action applier with optional PD controller (and joint spaces)
        action_applier = self._create_action_applier(
            Path(bundle_path), manifest, sim_env
        )

        # --- Set up input source ---
        input_source = self._create_input_source(command_bus)

        # --- Create and run loop ---
        loop = ReactiveLoop(
            manifest=manifest,
            inference_engine=inference,
            obs_assembler=obs_assembler,
            action_applier=action_applier,
            command_bus=command_bus,
        )

        termination = self.get_parameter("termination_condition", "manual_stop")
        timeout = float(self.get_parameter("timeout_seconds", 60.0))

        # Start input source
        if input_source is not None and hasattr(input_source, "start"):
            input_source.start()

        try:
            result = loop.run(
                sim_env,
                cancel_check=lambda: getattr(self, "_cancel_requested", False),
                timeout_seconds=timeout if termination == "timeout" else 0.0,
            )
        finally:
            if input_source is not None and hasattr(input_source, "stop"):
                input_source.stop()
            command_bus.clear()

        return {"result": {"status": "success", **result}}

    def _create_inference(self, bundle_path: Any, manifest: Any) -> Optional[Any]:
        """Create ONNX/TorchScript inference engine."""
        try:
            from src.system.policy.inference_engine import InferenceEngine
            model_file = bundle_path / manifest.model_path
            return InferenceEngine(str(model_file), backend=manifest.inference_backend.value)
        except Exception as exc:
            log.error("Failed to create inference engine: %s", exc)
            return None

    def _create_action_applier(self, bundle_path: Any, manifest: Any, sim_env: Any) -> Any:
        """Create action applier, with PD controller for Isaac Lab policies.

        Mirrors the construction PolicyRunner.load() does for the
        vis_check / mission preview path: builds bundle/qpos/ctrl
        JointSpaces from the live mj_model and threads them into the
        ActionApplier so apply_with_pd takes the convention-correct path
        instead of falling through to plain apply().
        """
        from src.system.policy.action_applier import ActionApplier
        from src.system.policy.manifest_schema import CheckpointBundle

        bundle_joint_names = (
            list(manifest.joint_mapping.keys()) if manifest.joint_mapping else []
        )

        # Build a minimal CheckpointBundle for ActionApplier
        bundle = CheckpointBundle(
            name=manifest.skill_name,
            version=manifest.version,
            bundle_path=bundle_path,
            policy_file=bundle_path / manifest.model_path,
            policy_format=manifest.inference_backend.value,
            obs_dim=manifest.observation_dim,
            action_dim=manifest.action_dim,
            action_type=manifest.action_space_type.value,
            control_frequency_hz=manifest.control_frequency_hz,
            decimation=1,
            robot_brand="",
            num_joints=manifest.action_dim,
            joint_names=bundle_joint_names,
            raw_manifest={},
        )

        pd_controller = None
        if manifest.source_type.value == "isaac_lab":
            try:
                import yaml
                # Isaac Lab's env.yaml contains !python/tuple tags for
                # range-style configs (friction/mass/init pose). Reuse the
                # parser-side loader so SafeLoader's strict path stays the
                # default everywhere except for this one trusted file.
                from src.system.skill.isaac_lab_manifest_parser import (
                    _IsaacLabEnvYamlLoader,
                )
                env_yaml = bundle_path / "env.yaml"
                if env_yaml.exists():
                    with env_yaml.open("r", encoding="utf-8") as fh:
                        raw_env = yaml.load(fh, Loader=_IsaacLabEnvYamlLoader) or {}
                    from src.system.sim.pd_controller import PDController
                    # ``joint_names`` is REQUIRED for the regex-pattern
                    # ``init_state.joint_pos`` expansion in env.yaml — without
                    # it the IL default standing pose lands in the wrong slots
                    # and the robot starts the episode in an OOD configuration.
                    pd_controller = PDController.from_env_yaml(
                        raw_env,
                        num_joints=manifest.action_dim,
                        joint_names=bundle_joint_names,
                    )
            except Exception as exc:
                log.warning("Could not create PD controller: %s", exc)

        # Build JointSpaces from the live MuJoCo model so apply_with_pd
        # can permute bundle ↔ qpos ↔ ctrl orderings explicitly. Without
        # these the IL torque path falls through to plain apply() with a
        # warning, which is the silent-miscompute path Mission canvas
        # used to hit.
        bundle_space = None
        qpos_space = None
        ctrl_space = None
        try:
            from src.system.policy.joint_space import (
                JointSpace,
                joint_spaces_from_mj_model,
            )
            if bundle_joint_names:
                bundle_space = JointSpace(
                    label="bundle", joints=tuple(bundle_joint_names)
                )
            mj_model = getattr(sim_env, "mj_model", None)
            if mj_model is not None:
                qpos_space, ctrl_space = joint_spaces_from_mj_model(mj_model)
        except Exception as exc:
            log.warning(
                "ReactiveLocomotionNode: failed to build JointSpaces "
                "(falling back to legacy apply path): %s",
                exc,
            )
            bundle_space = None
            qpos_space = None
            ctrl_space = None

        applier = ActionApplier(
            bundle,
            pd_controller=pd_controller,
            bundle_space=bundle_space,
            ctrl_space=ctrl_space,
        )
        if qpos_space is not None:
            applier.attach_qpos_space(qpos_space)
        return applier

    def _create_input_source(self, command_bus: Any) -> Optional[Any]:
        """Create the appropriate input source."""
        source_type = self.get_parameter("command_source", "gamepad")
        profile_name = self.get_parameter("input_profile", "quadruped_gamepad")

        if source_type == "gamepad":
            try:
                from src.system.runtime.input_sources.gamepad_source import GamepadInputSource
                return GamepadInputSource(command_bus)
            except Exception:
                log.info("Gamepad not available, falling back to keyboard")
                source_type = "keyboard"

        if source_type == "keyboard":
            try:
                from src.system.runtime.input_sources.keyboard_source import KeyboardInputSource
                return KeyboardInputSource(command_bus)
            except Exception:
                log.warning("Keyboard input source creation failed")

        return None

    def request_cancel(self) -> None:
        """Signal the reactive loop to stop."""
        self._cancel_requested = True
