# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacSim review subprocess driver — Export 节点 ▶ Launch Review (review_backend=isaac_sim).

Architecture:
    Export node Launch Review (review_backend=isaac_sim)
        → CanvasScene.review_launch_requested.emit(payload)
        → MainWindow._on_review_launch_requested(payload)
        → IsaacSimReviewTask.submit() to TasksManager
        → IsaacSimReviewTask.run() (本文件) on SDK worker thread
            ├── SKU hard-check: caller SKU == manifest.robot.sku
            ├── Build subprocess command (isaac python + il_review_launcher.py + bundle)
            ├── Popen + tail stdout with severity-aware log routing
            └── Wait for subprocess exit → done

Bundle-only contract (RELEASE/CLAUDE.md §1.9):
    The subprocess loads everything from ``<bundle>/manifest.yaml`` +
    ``<bundle>/policy.onnx``. It does NOT reach back into
    ``<project>/training/runs/`` or the canvas. Bundle is portable across
    machines (cloud-sync compatible).

Severity routing (CLAUDE.md §1.8):
    Subprocess stdout is classified per line:
      * ``[Fatal] / [Error] / [Critical]`` → log_error (red in CmdLogWidget)
      * ``[Warning]``                       → log_warning (yellow)
      * everything else                      → log_info
    The user sees Isaac Sim crash backtraces in the correct color slot
    instead of buried in INFO spam.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from unitport_sdk import (
    Task,
    log_error,
    log_info,
    log_warning,
    read_data,
)

from application.service.robot_init_poses import InitPoseOverride
from application.service.runtime.simulation.isaac_sim._log_utils import (
    classify_severity,
)


def _read_bundle_sku(bundle_dir: Path) -> str:
    """Extract ``robot.sku`` from ``<bundle_dir>/manifest.yaml``.

    Raises FileNotFoundError when manifest is missing — bundle without
    a manifest is not a bundle. Raises ValueError when ``robot.sku``
    field is absent — manifest without SKU is unusable per CLAUDE.md
    §1.7 SKU-only contract.
    """
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"bundle has no manifest.yaml: {manifest_path}"
        )
    raw = read_data(manifest_path)
    if not isinstance(raw, dict):
        raise ValueError(
            f"manifest.yaml is not a mapping: {manifest_path}"
        )
    robot = raw.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(
            f"manifest.yaml has no robot section: {manifest_path}"
        )
    sku = str(robot.get("sku") or "").strip()
    if not sku:
        raise ValueError(
            f"manifest.robot.sku is empty: {manifest_path} "
            f"(re-export the bundle from a project where the canvas "
            f"Robot Node has a registered SKU)"
        )
    return sku


class IsaacSimReviewTask(Task):
    """SDK Task: spawn ``il_review_launcher.py`` to replay a bundle in Isaac Sim.

    Sibling of :class:`MujocoReviewTask`. Same constructor signature so the
    UI dispatcher can pick either backend uniformly.
    """

    def __init__(
        self,
        bundle_path: Path,
        sku: str,
        scene_id: str,
        *,
        max_play_steps: int = 3000,
        init_pose_override: Optional[InitPoseOverride] = None,
    ) -> None:
        super().__init__(name=f"IsaacSimReview {Path(bundle_path).name}")
        self._bundle_path: Path = Path(bundle_path).resolve()
        self._caller_sku: str = str(sku or "").strip()
        self._scene_id: str = str(scene_id or "flat_ground").strip()
        self._max_play_steps: int = int(max_play_steps)
        # UI-supplied init pose override (None = no override → the launcher
        # spawns at the bundle's frozen default_joint_pos). The override
        # changes only the spawn pose; the policy's action neutral offset
        # stays the contract's default_joint_pos (parity w/ MujocoReviewTask).
        self._override: Optional[InitPoseOverride] = init_pose_override
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
                    f"[isaac_sim_review] cancel signal failed: {exc}; "
                    f"forcing kill"
                )
                try:
                    proc.kill()
                except Exception:
                    pass

    def run(self) -> Dict[str, Any]:
        if not self._bundle_path.is_dir():
            raise FileNotFoundError(
                f"bundle dir does not exist: {self._bundle_path}"
            )

        # SKU hard-check: caller (canvas-currently-bound SKU) must
        # match the bundle's frozen training-time SKU. Same shape as
        # PolicyRunner.load — bundles are SKU-locked.
        bundle_sku = _read_bundle_sku(self._bundle_path)
        if self._caller_sku and self._caller_sku != bundle_sku:
            raise RuntimeError(
                f"Bundle '{self._bundle_path.name}' is SKU-locked to "
                f"robot_sku={bundle_sku!r}; canvas Robot Config is currently "
                f"bound to robot_sku={self._caller_sku!r}. Bundles cannot be "
                f"replayed on a different robot — switch the canvas Robot "
                f"Node to match the bundle, or pick a different bundle."
            )

        # Resolve Isaac Lab install (python + launcher .py).
        from application.training.isaac_lab.config import IsaacLabConfig
        cfg = IsaacLabConfig.from_registers()

        # The il_review_launcher.py lives next to il_train_launcher.py
        # under <RELEASE>/src/application/training/isaac_lab/launcher/.
        # parents: [0]=isaac_sim [1]=simulation [2]=runtime [3]=service
        # [4]=application — so the training/ tree hangs off parents[4].
        launcher_root = Path(__file__).resolve().parents[4] / \
            "training" / "isaac_lab" / "launcher"
        review_launcher = launcher_root / "il_review_launcher.py"
        if not review_launcher.is_file():
            raise FileNotFoundError(
                f"il_review_launcher.py not found at {review_launcher}"
            )

        # Build the subprocess command. Use the same launcher-wrapper
        # shape as IsaacLabConfig._build_launcher_cmd so isaaclab.bat
        # quoting works on Windows. We piggy-back on a private method
        # rather than duplicate the .bat wrapper logic.
        review_args = [
            "--bundle", str(self._bundle_path),
            "--scene", self._scene_id,
            "--max_play_steps", str(self._max_play_steps),
        ]
        # UI init pose override → pass spawn pose to the launcher (parity
        # with MujocoReviewTask). Serialised exactly like the pose-review
        # task: base_pos clamped to a 3-tuple, joint_pos_by_ir JSON-encoded.
        # The launcher converts the IR-keyed dict into the bundle's joint
        # order via InitPoseOverride.to_bundle_order. Absent override → omit
        # both args so the launcher spawns at the bundle default.
        if self._override is not None:
            bp = self._override.base_pos or (0.0, 0.0, 0.4)
            base_pos = (float(bp[0]), float(bp[1]), float(bp[2]))
            cleaned: Dict[str, float] = {}
            for k, v in (self._override.joint_pos_by_ir or {}).items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            review_args += [
                "--init_base_pos",
                f"{base_pos[0]},{base_pos[1]},{base_pos[2]}",
                "--init_joint_pos_by_ir",
                json.dumps(cleaned, separators=(",", ":")),
            ]
        # Headed replay → minimal viewport-only experience (avoids the full
        # Kit GUI omni.kit.menu.utils crash; see headed_experience.py). Hand
        # the launcher the install root so it generates/selects it.
        if cfg.isaac_lab_path:
            review_args += ["--unitport_headed_experience", cfg.isaac_lab_path]
        cmd = cfg._build_launcher_cmd(str(review_launcher), review_args)

        log_info(
            f"[isaac_sim_review] launching: bundle={self._bundle_path.name} "
            f"sku={bundle_sku} scene={self._scene_id} "
            f"max_steps={self._max_play_steps}"
        )
        log_info(f"[isaac_sim_review] cmd: {' '.join(cmd)}")

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
            tagged = f"[isaac_sim_review] {line}"
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
                f"[isaac_sim_review] subprocess exited with code={retcode}"
            )
            raise RuntimeError(
                f"il_review_launcher.py exited with code={retcode}; "
                f"see error lines above for the root cause"
            )

        log_info(f"[isaac_sim_review] subprocess exited cleanly (code=0)")
        return {
            "bundle_path": str(self._bundle_path),
            "sku": bundle_sku,
            "scene_id": self._scene_id,
            "exit_code": retcode,
        }


__all__ = ["IsaacSimReviewTask"]
