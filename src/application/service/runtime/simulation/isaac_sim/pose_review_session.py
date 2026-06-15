# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacSim **pose-only** review subprocess driver.

Counterpart of :class:`IsaacSimReviewTask`. Where the latter loads a
trained bundle and runs ONNX inference in Isaac Sim, this task only
needs:

    * an SKU (resolves to a registered USD asset),
    * a base pose ``(x, y, z)``,
    * an IR-role keyed dict of joint angles (radians).

It then spawns ``il_pose_review_launcher.py`` under Isaac Lab's Python,
which builds a minimal scene around the USD articulation, writes the
override pose into the initial state, and holds the robot under
implicit-PD until the viewport is closed.

Use case: tune Actor Setting init_pos / init_joint_angles against the
IsaacSim renderer **before** training, instead of being forced to train
first just to see what the pose will look like in the high-fidelity
viewer. Mirrors the Actor Setting node's MuJoCo
:class:`RobotReviewTask` pose path 1:1.

Bundle-free contract: there is no bundle, no policy, no deploy_contract.
The launcher reads only the canvas-supplied SKU + override.

Severity routing matches ``IsaacSimReviewTask`` — see
``application.service.runtime.simulation.isaac_sim._log_utils``.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from unitport_sdk import (
    Task,
    log_error,
    log_info,
    log_warning,
)

from application.service.robot_init_poses import InitPoseOverride
from application.service.runtime.simulation.isaac_sim._log_utils import (
    classify_severity,
)


class IsaacSimPoseReviewTask(Task):
    """SDK Task: spawn ``il_pose_review_launcher.py`` for a pose-only view.

    Parameters parallel :class:`application.service.runtime.simulation.
    mujoco.robot_review_session.RobotReviewTask` so callers can dispatch
    on engine without reshaping the argv. Unlike its bundle-driven sibling
    :class:`IsaacSimReviewTask`, there is **no** bundle / ONNX / deploy
    contract — the subprocess builds the scene purely from the SKU
    registry + override.
    """

    def __init__(
        self,
        sku: str,
        scene_id: str,
        *,
        init_pose_override: Optional[InitPoseOverride] = None,
        hold_seconds: float = 0.0,
    ) -> None:
        super().__init__(name=f"IsaacSimPoseReview {sku}")
        self._sku: str = str(sku or "").strip()
        self._scene_id: str = str(scene_id or "flat_ground").strip()
        self._override: Optional[InitPoseOverride] = init_pose_override
        self._hold_seconds: float = float(hold_seconds)
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Task contract
    # ------------------------------------------------------------------

    def cancel(self) -> None:  # type: ignore[override]
        """Cooperative cancel — kill the Isaac Sim subprocess."""
        super().cancel()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except Exception as exc:
                log_warning(
                    f"[isaac_sim_pose_review] cancel signal failed: {exc}; "
                    f"forcing kill"
                )
                try:
                    proc.kill()
                except Exception:
                    pass

    def run(self) -> Dict[str, Any]:
        if not self._sku:
            raise ValueError(
                "IsaacSimPoseReviewTask: sku is empty — set a Robot Node "
                "asset on the canvas before requesting an Isaac Sim pose "
                "review."
            )

        from application.training.isaac_lab.config import IsaacLabConfig
        cfg = IsaacLabConfig.from_registers()

        # The pose-review launcher lives next to il_review_launcher.py
        # under <RELEASE>/src/application/training/isaac_lab/launcher/.
        # parents: [0]=isaac_sim [1]=simulation [2]=runtime [3]=service
        # [4]=application — so the training/ tree hangs off parents[4].
        launcher_root = Path(__file__).resolve().parents[4] / \
            "training" / "isaac_lab" / "launcher"
        pose_launcher = launcher_root / "il_pose_review_launcher.py"
        if not pose_launcher.is_file():
            raise FileNotFoundError(
                f"il_pose_review_launcher.py not found at {pose_launcher}"
            )

        base_pos: Tuple[float, float, float]
        joint_pos_by_ir_json: str
        if self._override is not None:
            bp = self._override.base_pos or (0.0, 0.0, 0.4)
            base_pos = (float(bp[0]), float(bp[1]), float(bp[2]))
            cleaned: Dict[str, float] = {}
            for k, v in (self._override.joint_pos_by_ir or {}).items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            joint_pos_by_ir_json = json.dumps(cleaned, separators=(",", ":"))
        else:
            base_pos = (0.0, 0.0, 0.4)
            joint_pos_by_ir_json = "{}"

        cmd = cfg._build_launcher_cmd(
            str(pose_launcher),
            [
                "--sku", self._sku,
                "--scene", self._scene_id,
                "--base_pos", f"{base_pos[0]},{base_pos[1]},{base_pos[2]}",
                "--joint_pos_by_ir", joint_pos_by_ir_json,
                "--hold_seconds", f"{self._hold_seconds:.3f}",
            ],
        )

        log_info(
            f"[isaac_sim_pose_review] launching: sku={self._sku} "
            f"scene={self._scene_id} base_pos={base_pos} "
            f"joints={joint_pos_by_ir_json[:80]}"
            f"{'...' if len(joint_pos_by_ir_json) > 80 else ''}"
        )
        log_info(f"[isaac_sim_pose_review] cmd: {' '.join(cmd)}")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        kwargs: Dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "cwd": cfg.isaac_lab_path or None,
            "env": env,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(cmd, **kwargs)
        proc = self._proc
        assert proc.stdout is not None

        for line in proc.stdout:
            self.check_cancelled()
            line = line.rstrip()
            if not line:
                continue
            sev = classify_severity(line)
            tagged = f"[isaac_sim_pose_review] {line}"
            if sev == "error":
                log_error(tagged)
            elif sev == "warning":
                log_warning(tagged)
            else:
                log_info(tagged)

        proc.wait()
        retcode = proc.returncode
        if retcode != 0:
            log_error(
                f"[isaac_sim_pose_review] subprocess exited with code={retcode}"
            )
            raise RuntimeError(
                f"il_pose_review_launcher.py exited with code={retcode}; "
                f"see error lines above for the root cause"
            )

        log_info(
            f"[isaac_sim_pose_review] subprocess exited cleanly (code=0)"
        )
        return {
            "sku": self._sku,
            "scene_id": self._scene_id,
            "exit_code": retcode,
        }


__all__ = ["IsaacSimPoseReviewTask"]
