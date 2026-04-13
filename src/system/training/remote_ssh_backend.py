"""RemoteSSHBackend — orchestrates Isaac Lab training on a remote server via SSH.

Same interface as ``IsaacLabBackend``:
    backend = RemoteSSHBackend(config, server)
    backend.on_message = my_callback   # receives MSG_* dicts
    backend.start()
    backend.cancel()

Reuses ``MetricsAccumulator``, ``parse_log_line``, ``parse_amp_line`` from
``isaac_lab_backend.py`` so the training panel receives identical messages
regardless of local vs. remote execution.
"""

from __future__ import annotations

import logging
import os
import posixpath
import stat
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from src.system.training.isaac_lab_backend import (
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_FINISHED,
    MSG_LOG,
    MSG_METRICS,
    MSG_PROGRESS,
    MetricsAccumulator,
    parse_amp_line,
    parse_log_line,
)
from src.system.training.isaac_lab_config import IsaacLabConfig
from src.system.training.remote_ssh_config import RemoteServerConfig

log = logging.getLogger(__name__)

# Training done sentinel (same as isaac_lab_backend.py).
_TRAINING_DONE_SENTINEL = "[UnitPort] TRAINING_LOOP_DONE"

# Iteration line regex for progress bar (reuse from isaac_lab_backend).
import re
_ITER_LINE_RE = re.compile(
    r"[Ll]earning\s+[Ii]teration\s+(\d+)\s*/\s*(\d+)"
)
_SAVED_RE = re.compile(
    r"(?:[Ss]aving\s+model\s+to|model_\d+\.pt)\s*[:]?\s*(.+?)(?:\s|$)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class RemoteSSHBackend:
    """Orchestrates Isaac Lab training on a remote server via SSH/SFTP."""

    def __init__(
        self,
        config: IsaacLabConfig,
        server: RemoteServerConfig,
    ) -> None:
        self.config = config
        self.server = server
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None

        self._thread: Optional[threading.Thread] = None
        self._cancelled = threading.Event()
        self._export_path: Optional[Path] = None
        self._remote_pid: Optional[int] = None
        self._ssh_client: Optional[Any] = None  # paramiko.SSHClient
        self._last_iter_total: int = 0

    # ------------------------------------------------------------------
    # Public API (mirrors IsaacLabBackend)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch remote training in a background thread."""
        if self._thread and self._thread.is_alive():
            log.warning("RemoteSSHBackend already running")
            return
        self._cancelled.clear()
        self._remote_pid = None
        self._thread = threading.Thread(
            target=self._run, name="remote_ssh_train", daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the running remote training."""
        self._cancelled.set()
        pid = self._remote_pid
        client = self._ssh_client
        if pid and client:
            try:
                cmd = f"kill -TERM {pid} 2>/dev/null; kill -TERM -{pid} 2>/dev/null"
                client.exec_command(cmd, timeout=5)
                log.info("Sent SIGTERM to remote PID %d", pid)
            except Exception:
                log.exception("Failed to send kill to remote process")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def export_path(self) -> Optional[Path]:
        return self._export_path

    # ------------------------------------------------------------------
    # Internal: emit helper
    # ------------------------------------------------------------------

    def _emit(self, msg: Dict[str, Any]) -> None:
        if self.on_message:
            try:
                self.on_message(msg)
            except Exception:
                log.exception("Error in on_message callback")

    # ------------------------------------------------------------------
    # Internal: SSH connection
    # ------------------------------------------------------------------

    def _connect(self):
        """Create and return a connected paramiko.SSHClient."""
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: Dict[str, Any] = {
            "hostname": self.server.host,
            "port": self.server.port,
            "username": self.server.username,
            "timeout": 15,
        }
        if self.server.auth_method == "key" and self.server.private_key_path:
            connect_kwargs["key_filename"] = self.server.private_key_path
        elif self.server.auth_method == "password" and self.server.password:
            connect_kwargs["password"] = self.server.password
        else:
            raise ValueError("No valid SSH authentication credentials configured.")

        client.connect(**connect_kwargs)

        # Enable keepalive to detect dead connections.
        transport = client.get_transport()
        if transport:
            transport.set_keepalive(30)

        return client

    # ------------------------------------------------------------------
    # Internal: SFTP helpers
    # ------------------------------------------------------------------

    def _sftp_mkdirs(self, sftp, remote_path: str) -> None:
        """Recursively create remote directories (like mkdir -p)."""
        parts = PurePosixPath(remote_path).parts
        current = ""
        for part in parts:
            current = posixpath.join(current, part) if current else part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)

    def _sftp_put_with_log(self, sftp, local_path: str, remote_path: str) -> None:
        """Upload a file via SFTP with a log message."""
        local_size = os.path.getsize(local_path)
        self._emit({
            "type": MSG_LOG,
            "data": f"[UnitPort] Uploading {Path(local_path).name} "
                    f"({local_size / 1024:.0f} KB) -> {remote_path}",
        })
        sftp.put(local_path, remote_path)

    def _sftp_download_recursive(
        self,
        sftp,
        remote_dir: str,
        local_dir: Path,
        pattern: str = "",
    ) -> List[str]:
        """Recursively download files from remote_dir to local_dir.

        Returns list of downloaded local paths.
        """
        downloaded: List[str] = []
        try:
            entries = sftp.listdir_attr(remote_dir)
        except FileNotFoundError:
            return downloaded

        local_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            remote_path = posixpath.join(remote_dir, entry.filename)
            local_path = local_dir / entry.filename

            if stat.S_ISDIR(entry.st_mode or 0):
                sub = self._sftp_download_recursive(sftp, remote_path, local_path, pattern)
                downloaded.extend(sub)
            else:
                if pattern and not _matches_pattern(entry.filename, pattern):
                    continue
                try:
                    sftp.get(remote_path, str(local_path))
                    downloaded.append(str(local_path))
                except Exception as exc:
                    self._emit({
                        "type": MSG_LOG,
                        "data": f"[UnitPort] Download failed: {remote_path} -> {exc}",
                    })

        return downloaded

    # ------------------------------------------------------------------
    # Internal: upload training files
    # ------------------------------------------------------------------

    def _upload_training_files(self, sftp, remote_run_dir: str) -> Dict[str, str]:
        """Upload config file, launcher script, and AMP motion files.

        Returns dict mapping logical names to remote paths.
        """
        paths: Dict[str, str] = {}

        # 1) Compiled @configclass Python file.
        if self.config.config_file and Path(self.config.config_file).is_file():
            remote_cfg = posixpath.join(remote_run_dir, "unitport_config.py")
            self._sftp_put_with_log(sftp, self.config.config_file, remote_cfg)
            paths["config_file"] = remote_cfg

        # 2) UnitPort's il_train_launcher.py (the remote has Isaac Lab but
        #    not UnitPort source — this single-file script is the bridge).
        launcher_src = Path(__file__).resolve().parent / "il_train_launcher.py"
        if launcher_src.is_file():
            remote_launcher = posixpath.join(remote_run_dir, "il_train_launcher.py")
            self._sftp_put_with_log(sftp, str(launcher_src), remote_launcher)
            paths["launcher_script"] = remote_launcher
        else:
            log.warning("il_train_launcher.py not found at %s", launcher_src)

        # 3) AMP motion files.
        if self.config.amp_motion_files:
            motion_dir = posixpath.join(remote_run_dir, "motion")
            self._sftp_mkdirs(sftp, motion_dir)
            remote_motion_paths: List[str] = []
            for local_motion in self.config.amp_motion_files:
                if Path(local_motion).is_file():
                    fname = Path(local_motion).name
                    rpath = posixpath.join(motion_dir, fname)
                    self._sftp_put_with_log(sftp, str(local_motion), rpath)
                    remote_motion_paths.append(rpath)
            if remote_motion_paths:
                paths["motion_files"] = ",".join(remote_motion_paths)

        return paths

    # ------------------------------------------------------------------
    # Internal: process line (mirrors IsaacLabBackend._process_line)
    # ------------------------------------------------------------------

    def _process_line(
        self,
        line: str,
        metrics_acc: MetricsAccumulator,
        best_reward: float,
    ) -> tuple:
        """Process a single log line: emit, parse progress/metrics.

        Returns (best_reward, checkpoint_path_or_None).
        """
        self._emit({"type": MSG_LOG, "data": line})
        checkpoint_path = None

        clean = _ANSI_RE.sub("", line)

        # Iteration header → progress.
        m = _ITER_LINE_RE.search(clean)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            self._last_iter_total = total
            display_step = total if step >= total - 1 else step
            self._emit({
                "type": MSG_PROGRESS,
                "data": (display_step, total, best_reward, best_reward, 0.0, "training"),
            })

        # Training done sentinel.
        if _TRAINING_DONE_SENTINEL in clean:
            total = self._last_iter_total or 1
            self._emit({
                "type": MSG_PROGRESS,
                "data": (total, total, best_reward, best_reward, 0.0, "training"),
            })

        parsed = parse_log_line(clean)
        if parsed:
            self._emit(parsed)

        # AMP single-line metrics.
        amp_parsed = parse_amp_line(clean)
        if amp_parsed:
            self._emit(amp_parsed)
            amp_data = amp_parsed.get("data", {})
            it = int(amp_data.get("iteration", 0))
            total = int(amp_data.get("total", 0))
            if total > 0:
                self._last_iter_total = total
                rew = float(amp_data.get("reward_mean", 0.0))
                if amp_data.get("reward_valid") and rew > best_reward:
                    best_reward = rew
                display_step = total if it >= total - 1 else it + 1
                self._emit({
                    "type": MSG_PROGRESS,
                    "data": (display_step, total, rew, best_reward, 0.0, "training"),
                })

        # RSL-RL multi-line metrics block.
        metrics_msg = metrics_acc.feed(clean)
        if metrics_msg:
            self._emit(metrics_msg)
            reward = metrics_msg.get("data", {}).get("reward_mean", best_reward)
            if reward > best_reward:
                best_reward = reward

        # Checkpoint saved.
        m = _SAVED_RE.search(clean)
        if m:
            checkpoint_path = m.group(1).strip()

        return best_reward, checkpoint_path

    # ------------------------------------------------------------------
    # Internal: main run loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread target: connect, upload, execute, tail, download."""
        import paramiko

        run_id = self.config.run_id or f"cloud_{uuid.uuid4().hex[:8]}"
        # Upload dir: Docker mode uses the host side of the volume mount;
        # bare-metal uses remote_work_dir.
        upload_base = self.server.effective_upload_dir
        remote_run_dir = f"{upload_base}/{run_id}"
        remote_log_dir = f"{remote_run_dir}/logs"

        client = None
        try:
            # ── 1. Connect ──────────────────────────────────────────────
            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Connecting to {self.server.host}:{self.server.port}..."})
            client = self._connect()
            self._ssh_client = client
            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Connected as {self.server.username}@{self.server.host}"})

            # ── 2. Create remote directories ────────────────────────────
            sftp = client.open_sftp()
            self._sftp_mkdirs(sftp, remote_run_dir)
            self._sftp_mkdirs(sftp, remote_log_dir)

            # ── 3. Upload training files ────────────────────────────────
            self._emit({"type": MSG_LOG, "data": "[UnitPort] Uploading training files..."})
            uploaded = self._upload_training_files(sftp, remote_run_dir)
            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Upload complete. Files: {list(uploaded.keys())}"})

            # ── 4. Build remote command ─────────────────────────────────
            use_docker = self.server.docker_enabled and self.server.docker_image

            # Paths the launcher / config / motions were uploaded to on the
            # HOST filesystem. In Docker mode these paths live on the host
            # and are volume-mounted into the container.
            host_script = uploaded.get("launcher_script", "")
            if not host_script:
                self._emit({"type": MSG_ERROR, "data": "Failed to upload il_train_launcher.py"})
                return

            host_config = uploaded.get("config_file", "")
            host_motion = uploaded.get("motion_files", "").split(",") if uploaded.get("motion_files") else None

            if use_docker:
                # Docker mode.
                # The host data dir is volume-mounted into the container.
                # Paths uploaded to the host need remapping to the
                # container mount point.
                #
                # Example (matching user's cloud panel config):
                #   host upload dir:  /home/trooperai/isaaclab_data/{run_id}/
                #   container sees:   /data/{run_id}/
                #   isaaclab.sh:      /workspace/isaaclab/isaaclab.sh (built into image)

                host_data = self.server.docker_host_data_dir.rstrip("/")
                container_data = self.server.docker_container_data_dir.rstrip("/")
                container_run = f"{container_data}/{run_id}"
                container_log_dir = f"{container_run}/logs"

                def _remap(host_path: str) -> str:
                    """Remap host path under upload dir to container mount."""
                    if host_path.startswith(host_data):
                        return container_data + host_path[len(host_data):]
                    return host_path

                container_script = _remap(host_script)
                container_config = _remap(host_config) if host_config else ""
                container_motion = [_remap(p) for p in host_motion] if host_motion else None

                # Isaac Lab launcher inside the official container
                container_launcher = "/workspace/isaaclab/isaaclab.sh"

                cmd_str = self.config.build_remote_command(
                    remote_launcher=container_launcher,
                    remote_script=container_script,
                    remote_log_dir=container_log_dir,
                    remote_config_file=container_config,
                    remote_motion_files=container_motion,
                )

                docker_extra = self.server.docker_extra_args.strip()
                inner_cmd = f"echo PID=$$ && exec {cmd_str}"

                if self.server.docker_use_exec:
                    # Existing container managed by cloud panel (docker exec)
                    cname = self.server.docker_container_name
                    full_cmd = (
                        f"docker exec "
                        f"-e PYTHONUNBUFFERED=1 -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y "
                        f"{docker_extra + ' ' if docker_extra else ''}"
                        f"{cname} "
                        f"bash -c '{inner_cmd}'"
                    )
                else:
                    # Ephemeral container (docker run --rm)
                    full_cmd = (
                        f"docker run --rm --gpus all "
                        f"-v {host_data}:{container_data} "
                        f"-e PYTHONUNBUFFERED=1 -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y "
                        f"{docker_extra + ' ' if docker_extra else ''}"
                        f"{self.server.docker_image} "
                        f"bash -c '{inner_cmd}'"
                    )
            else:
                # Bare-metal mode (original path)
                remote_launcher = self.server.remote_isaac_lab_launcher or (
                    self.server.remote_isaac_lab_path.rstrip("/") + "/isaaclab.sh"
                )

                cmd_str = self.config.build_remote_command(
                    remote_launcher=remote_launcher,
                    remote_script=host_script,
                    remote_log_dir=remote_log_dir,
                    remote_config_file=host_config,
                    remote_motion_files=host_motion,
                )

                env_prefix = ""
                if self.server.conda_env:
                    env_prefix = (
                        f"source activate {self.server.conda_env} 2>/dev/null || "
                        f"conda activate {self.server.conda_env} && "
                    )

                full_cmd = (
                    f"export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y && "
                    f"{env_prefix}"
                    f"echo PID=$$ && exec {cmd_str}"
                )

            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Remote command: {cmd_str}"})
            if use_docker:
                self._emit({"type": MSG_LOG, "data": f"[UnitPort] Docker image: {self.server.docker_image}"})
            self._emit({"type": MSG_LOG, "data": "[UnitPort] Starting remote training..."})

            # ── 5. Execute via SSH ──────────────────────────────────────
            transport = client.get_transport()
            channel = transport.open_session()
            channel.get_pty()  # allocate PTY for proper line buffering
            channel.exec_command(full_cmd)

            # ── 6. Tail stdout ──────────────────────────────────────────
            best_reward = float("-inf")
            last_checkpoint_path: Optional[str] = None
            metrics_acc = MetricsAccumulator()
            line_buffer = ""

            while not channel.exit_status_ready() or channel.recv_ready():
                if self._cancelled.is_set():
                    self._emit({"type": MSG_LOG, "data": "[UnitPort] Cancellation requested..."})
                    self.cancel()  # sends kill
                    channel.close()
                    self._emit({"type": MSG_CANCELLED})
                    return

                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    line_buffer += chunk

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        line = line.rstrip("\r")

                        # Capture PID from our echo.
                        if line.startswith("PID=") and self._remote_pid is None:
                            try:
                                self._remote_pid = int(line.split("=", 1)[1].strip())
                                self._emit({
                                    "type": MSG_LOG,
                                    "data": f"[UnitPort] Remote PID: {self._remote_pid}",
                                })
                            except ValueError:
                                pass
                            continue

                        bw, cp = self._process_line(line, metrics_acc, best_reward)
                        best_reward = bw
                        if cp:
                            last_checkpoint_path = cp
                else:
                    time.sleep(0.05)

            # Process remaining buffer.
            if line_buffer.strip():
                bw, cp = self._process_line(line_buffer.strip(), metrics_acc, best_reward)
                best_reward = bw
                if cp:
                    last_checkpoint_path = cp

            # Flush metrics accumulator.
            metrics_msg = metrics_acc.flush()
            if metrics_msg:
                self._emit(metrics_msg)

            exit_status = channel.recv_exit_status()
            self._emit({"type": MSG_LOG, "data": f"[UnitPort] Remote process exited with code {exit_status}"})

            if self._cancelled.is_set():
                self._emit({"type": MSG_CANCELLED})
                return

            if exit_status != 0 and exit_status != 55:
                self._emit({
                    "type": MSG_ERROR,
                    "data": f"Remote training failed (exit code {exit_status})",
                })
                return

            # ── 7. Download results ─────────────────────────────────────
            local_log_dir = Path(self.config.log_dir) if self.config.log_dir else None
            if local_log_dir:
                local_log_dir.mkdir(parents=True, exist_ok=True)
                self._emit({"type": MSG_LOG, "data": "[UnitPort] Downloading training results..."})
                downloaded = self._sftp_download_recursive(
                    sftp, remote_log_dir, local_log_dir,
                )
                self._emit({
                    "type": MSG_LOG,
                    "data": f"[UnitPort] Downloaded {len(downloaded)} files to {local_log_dir}",
                })

                # Find the export directory.
                exported_dir = local_log_dir / "exported"
                if not exported_dir.is_dir():
                    # Look for any subdirectory named "exported".
                    for d in local_log_dir.rglob("exported"):
                        if d.is_dir():
                            exported_dir = d
                            break

                if exported_dir.is_dir():
                    self._export_path = exported_dir

            # ── 8. Emit finished ────────────────────────────────────────
            bundle_path = str(self._export_path or local_log_dir or remote_log_dir)
            self._emit({
                "type": MSG_FINISHED,
                "data": {
                    "bundle_path": bundle_path,
                    "artifact_path": bundle_path,
                    "step": 0,
                    "reward_mean": best_reward if best_reward != float("-inf") else 0.0,
                    "best_reward": best_reward if best_reward != float("-inf") else 0.0,
                    "ep_len_mean": 0.0,
                },
            })

        except ImportError:
            self._emit({
                "type": MSG_ERROR,
                "data": "paramiko is not installed. Run: pip install paramiko",
            })
        except Exception as exc:
            log.exception("RemoteSSHBackend error")
            if "Authentication" in str(type(exc).__name__):
                self._emit({
                    "type": MSG_ERROR,
                    "data": f"SSH authentication failed: {exc}",
                })
            elif "SSH" in str(type(exc).__name__) or "socket" in str(type(exc).__name__):
                self._emit({
                    "type": MSG_ERROR,
                    "data": (
                        f"SSH connection lost: {exc}\n"
                        "Training may still be running on the remote server. "
                        f"Check {self.server.host} manually."
                    ),
                })
            else:
                self._emit({"type": MSG_ERROR, "data": str(exc)})
        finally:
            self._ssh_client = None
            if client:
                try:
                    client.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _matches_pattern(filename: str, pattern: str) -> bool:
    """Simple glob-like pattern match for file downloads."""
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)
