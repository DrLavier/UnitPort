"""PolicyPreviewRunner — loads a checkpoint policy and replays it in MuJoCo.

Runs in a QThread so the training main thread is unaffected.
The MuJoCo viewer blocks until the user closes it, then the thread exits.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class PolicyPreviewRunner(QThread):
    """Background thread that opens a MuJoCo viewer with a trained policy.

    Signals
    -------
    preview_started()
        Emitted when the viewer has opened.
    preview_stopped()
        Emitted when the viewer is closed and the thread exits.
    preview_error(str)
        Emitted on any error during setup or execution.
    """

    preview_started = Signal()
    preview_stopped = Signal()
    preview_error = Signal(str)

    def __init__(self, checkpoint_path: str, parent=None) -> None:
        super().__init__(parent)
        self._checkpoint_path = Path(checkpoint_path)
        self._cancel_event = threading.Event()

    def request_stop(self) -> None:
        """Ask the preview to stop on the next tick."""
        self._cancel_event.set()

    def run(self) -> None:
        """Thread entry — build pipeline, open viewer, run loop."""
        try:
            self._run_preview()
        except Exception as exc:
            log.exception("PolicyPreviewRunner error")
            self.preview_error.emit(str(exc))
        finally:
            self.preview_stopped.emit()

    def _run_preview(self) -> None:
        checkpoint_dir = self._checkpoint_path.parent
        policy_path = self._find_policy(checkpoint_dir)
        if policy_path is None:
            self.preview_error.emit(
                f"No policy file (policy.onnx / policy_jit.pt) found near {self._checkpoint_path}"
            )
            return

        env_yaml = self._find_env_yaml(checkpoint_dir)

        # 1. Build SkillManifest from env.yaml (if available)
        manifest = None
        if env_yaml is not None:
            try:
                from src.system.skill.isaac_lab_manifest_parser import parse_isaac_lab_env_yaml
                manifest = parse_isaac_lab_env_yaml(env_yaml, model_path=str(policy_path))
            except Exception as exc:
                log.warning("Failed to parse env.yaml, falling back to defaults: %s", exc)

        # 2. Load inference engine
        if policy_path.suffix == ".onnx":
            from src.system.policy.inference_engine import ONNXEngine
            engine = ONNXEngine()
        else:
            from src.system.policy.inference_engine import JITEngine
            engine = JITEngine()
        engine.load(str(policy_path))

        # 3. Load MuJoCo model
        import mujoco
        mjcf_path = self._find_mjcf(checkpoint_dir)
        if mjcf_path is None:
            self.preview_error.emit("No MJCF/XML model found for MuJoCo preview")
            return
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)

        # 4. Build ObsAssembler
        from src.system.policy.obs_assembler import ObsAssembler
        from src.system.policy.joint_reorder import compute_reorder_indices
        import numpy as np

        # Get joint names from MuJoCo model (hinge joints only = actuated DOFs)
        mj_joint_names = [
            model.joint(i).name for i in range(model.njnt)
            if model.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE
        ]

        obs_reorder = None   # MuJoCo → policy order (for obs assembly)
        act_reorder = None   # policy → MuJoCo order (for action application)
        if manifest is not None:
            if hasattr(manifest, "joint_names") and manifest.joint_names:
                obs_reorder = compute_reorder_indices(
                    mj_joint_names, manifest.joint_names
                )  # mj_values[obs_reorder] = policy order
                act_reorder = compute_reorder_indices(
                    manifest.joint_names, mj_joint_names
                )  # il_values[act_reorder] = mujoco order
            obs_assembler = ObsAssembler(
                manifest,
                reorder_indices=obs_reorder,
            )
        else:
            obs_assembler = None

        # Wrap MuJoCo data for ObsAssembler (readers expect env.mj_data, env.joint_names)
        class _MjEnvContext:
            """Minimal adapter so ObsAssembler sensor readers can access MuJoCo state."""
            def __init__(self, mj_model, mj_data, joint_names):
                self.mj_model = mj_model
                self.mj_data = mj_data
                self.joint_names = joint_names
        env_ctx = _MjEnvContext(model, data, mj_joint_names)

        # 5. Build PD controller
        from src.system.sim.pd_controller import PDController
        num_act = model.nu
        pd = PDController(kp=25.0, kd=0.5, num_joints=num_act)

        # 6. Command bus + input source
        from src.system.runtime.command_bus import CommandBus
        command_bus = CommandBus()

        # Try gamepad first, fall back to keyboard
        input_source = None
        try:
            from src.system.runtime.input_sources.gamepad_source import GamepadInputSource
            input_source = GamepadInputSource(command_bus)
            log.info("Using gamepad input for preview")
        except Exception:
            pass
        if input_source is None:
            try:
                from src.system.runtime.input_sources.keyboard_source import KeyboardInputSource
                input_source = KeyboardInputSource(command_bus)
                log.info("Using keyboard input for preview")
            except Exception:
                log.warning("No input source available for preview")

        # 7. Open MuJoCo viewer and run loop
        self.preview_started.emit()
        log.info("Opening MuJoCo preview: %s", policy_path)

        import mujoco.viewer
        last_action = np.zeros(num_act)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not self._cancel_event.is_set():
                # Read commands
                if input_source is not None and hasattr(input_source, "poll"):
                    try:
                        input_source.poll()
                    except Exception:
                        pass

                # Assemble obs
                if obs_assembler is not None:
                    try:
                        obs = obs_assembler.assemble(
                            env_ctx, command_bus=command_bus, last_action=last_action
                        )
                    except Exception:
                        obs = np.zeros(engine._input_size if hasattr(engine, "_input_size") else 48)
                else:
                    obs = np.zeros(48)

                # Inference
                try:
                    action = engine.predict(obs)
                except Exception:
                    action = np.zeros(num_act)
                last_action = action

                # PD control -> torques
                if act_reorder is not None:
                    from src.system.policy.joint_reorder import reorder_array
                    action = reorder_array(action, act_reorder)

                torques = pd.compute(
                    target_pos=action,
                    current_pos=data.qpos[-num_act:],
                    current_vel=data.qvel[-num_act:],
                )
                data.ctrl[:num_act] = torques

                mujoco.mj_step(model, data)
                viewer.sync()

        log.info("MuJoCo preview closed")

    # ------------------------------------------------------------------
    # File discovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_policy(checkpoint_dir: Path) -> Optional[Path]:
        """Look for policy.onnx or policy_jit.pt near the checkpoint."""
        for name in ("policy.onnx", "policy_jit.pt"):
            for d in [checkpoint_dir, checkpoint_dir.parent]:
                p = d / name
                if p.exists():
                    return p
        # Maybe the checkpoint itself is the policy
        for ext in (".onnx", ".pt"):
            for f in checkpoint_dir.glob(f"*{ext}"):
                if "model_" not in f.name:
                    return f
        return None

    @staticmethod
    def _find_env_yaml(checkpoint_dir: Path) -> Optional[Path]:
        """Look for env.yaml near the checkpoint."""
        for d in [checkpoint_dir, checkpoint_dir.parent]:
            p = d / "env.yaml"
            if p.exists():
                return p
            p = d / "params" / "env.yaml"
            if p.exists():
                return p
        return None

    @staticmethod
    def _find_mjcf(checkpoint_dir: Path) -> Optional[Path]:
        """Find a MuJoCo XML model file."""
        # Check for Go2 as default
        from src.system.models.mujoco_asset_registry import resolve_mujoco_asset
        try:
            loc = resolve_mujoco_asset("unitree", "go2")
            if loc is not None:
                return loc.scene_path
        except Exception:
            pass
        # Fallback: search near checkpoint
        for d in [checkpoint_dir, checkpoint_dir.parent, checkpoint_dir.parent.parent]:
            for xml in d.glob("*.xml"):
                return xml
        return None
