"""SSH-based Isaac Lab cloud deployer.

Uploads ``install_script.py`` to a remote Linux server via SFTP,
executes it, and streams structured progress back to the UI.

Usage::

    deployer = IsaacCloudDeployer(
        host="10.0.0.1", username="user",
        auth_method="key", private_key_path="~/.ssh/id_rsa",
        remote_install_dir="/home/user/UnitPort/engines/isaac",
    )
    deployer.progress.connect(on_progress)
    deployer.finished_ok.connect(on_done)     # receives remote IsaacLab root
    deployer.failed.connect(on_error)
    deployer.start()
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)

# install_script.py lives next to this file.
_INSTALL_SCRIPT = Path(__file__).resolve().parent / "install_script.py"

# Structured markers (same protocol as install_script.py)
_STEP_RE = re.compile(r"^\[STEP (\d+)/(\d+)\]\s*(.+)")
_OK_RE = re.compile(r"^\[OK\]\s*(.*)")
_WARN_RE = re.compile(r"^\[WARN\]\s*(.*)")
_ERROR_RE = re.compile(r"^\[ERROR\]\s*(.*)")
_ROOT_RE = re.compile(r"^\[ISAAC_LAB_ROOT\]\s*(.*)")
_DONE_RE = re.compile(r"^\[DONE\]")


class IsaacCloudDeployer(QThread):
    """QThread that deploys Isaac Lab to a remote server via SSH.

    Signals
    -------
    progress(str)
        Human-readable status text.
    step_changed(int, int, str)
        (current_step, total_steps, description).
    finished_ok(str)
        Remote Isaac Lab root path on success.
    failed(str)
        Error message on failure.
    log_line(str)
        Raw output line from remote execution.
    """

    progress = Signal(str)
    step_changed = Signal(int, int, str)
    finished_ok = Signal(str)
    failed = Signal(str)
    log_line = Signal(str)

    def __init__(
        self,
        host: str,
        username: str,
        auth_method: str = "key",
        private_key_path: str = "",
        password: str = "",
        port: int = 22,
        remote_install_dir: str = "",
        server_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._username = username
        self._auth_method = auth_method
        self._private_key_path = private_key_path
        self._password = password
        self._port = port
        self._remote_install_dir = remote_install_dir or "~/UnitPort/engines/isaac"
        self._server_name = server_name or host
        self._cancelled = False
        self._isaac_lab_root: Optional[str] = None

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def isaac_lab_root(self) -> Optional[str]:
        return self._isaac_lab_root

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            import paramiko
        except ImportError:
            self.failed.emit(
                "paramiko is not installed. Run:\n"
                "  .venv311/Scripts/pip install paramiko"
            )
            return

        if not _INSTALL_SCRIPT.is_file():
            self.failed.emit(f"Install script not found: {_INSTALL_SCRIPT}")
            return

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # ── Connect ──────────────────────────────────────────────────
            self.progress.emit(f"Connecting to {self._host}...")
            connect_kw: Dict[str, Any] = {
                "hostname": self._host,
                "port": self._port,
                "username": self._username,
                "timeout": 15,
            }
            if self._auth_method == "key" and self._private_key_path:
                key_path = os.path.expanduser(self._private_key_path)
                connect_kw["key_filename"] = key_path
            elif self._auth_method == "password" and self._password:
                connect_kw["password"] = self._password
            else:
                self.failed.emit("No valid SSH credentials configured.")
                return

            client.connect(**connect_kw)
            self.progress.emit(f"Connected to {self._host}")

            if self._cancelled:
                self.failed.emit("Cancelled.")
                return

            # ── Upload install script ────────────────────────────────────
            self.progress.emit("Uploading install script...")
            sftp = client.open_sftp()
            try:
                # Ensure remote directory exists
                remote_dir = self._remote_install_dir.rstrip("/")
                self._sftp_mkdirs(sftp, remote_dir)

                remote_script = f"{remote_dir}/_unitport_install_isaac.py"
                sftp.put(str(_INSTALL_SCRIPT), remote_script)
                sftp.chmod(remote_script, 0o755)
                self.progress.emit("Install script uploaded")
            finally:
                sftp.close()

            if self._cancelled:
                self.failed.emit("Cancelled.")
                return

            # ── Find remote Python 3.11 ──────────────────────────────────
            self.progress.emit("Detecting remote Python 3.11...")
            python_cmd = self._find_remote_python(client)
            if not python_cmd:
                self.failed.emit(
                    "Python 3.11 not found on remote server.\n"
                    "Install it first:\n"
                    "  sudo apt install python3.11 python3.11-venv python3.11-dev"
                )
                return
            self.progress.emit(f"Remote Python: {python_cmd}")

            # ── Execute install ──────────────────────────────────────────
            cmd = (
                f'{python_cmd} "{remote_script}" '
                f'--install-dir "{remote_dir}"'
            )
            self.progress.emit(f"Starting remote installation...")

            # Use get_pty for line-buffered output
            transport = client.get_transport()
            if transport is None:
                self.failed.emit("SSH transport lost.")
                return
            channel = transport.open_session()
            channel.get_pty()
            channel.exec_command(cmd)

            # Stream output
            buf = ""
            while not channel.exit_status_ready() or channel.recv_ready():
                if self._cancelled:
                    channel.close()
                    self.failed.emit("Cancelled by user.")
                    return
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.rstrip("\r")
                        if line:
                            self.log_line.emit(line)
                            self._parse_line(line)
                else:
                    self.msleep(50)

            # Flush remaining
            if buf.strip():
                self.log_line.emit(buf.strip())
                self._parse_line(buf.strip())

            exit_code = channel.recv_exit_status()
            channel.close()

            if exit_code != 0:
                self.failed.emit(
                    f"Remote installation failed (exit code {exit_code})."
                )
                return

            if not self._isaac_lab_root:
                self.failed.emit(
                    "Installation completed but Isaac Lab root was not reported."
                )
                return

            # ── Register server ──────────────────────────────────────────
            self._register_server()
            self.finished_ok.emit(self._isaac_lab_root)

        except Exception as exc:
            self.failed.emit(f"Cloud deploy error: {exc}")
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sftp_mkdirs(sftp: Any, remote_path: str) -> None:
        """Recursively create remote directories (like mkdir -p)."""
        import stat as stat_mod
        parts = remote_path.split("/")
        current = ""
        for part in parts:
            if not part:
                current = "/"
                continue
            current = current.rstrip("/") + "/" + part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)

    def _find_remote_python(self, client: Any) -> Optional[str]:
        """Find Python 3.11 on the remote server."""
        candidates = [
            "python3.11",
            "/usr/bin/python3.11",
            "/usr/local/bin/python3.11",
            "python3",
        ]
        for candidate in candidates:
            cmd = (
                f'{candidate} -c '
                f'"import sys; print(f\'{{sys.version_info.major}}.'
                f'{{sys.version_info.minor}}\')" 2>/dev/null'
            )
            _, stdout, _ = client.exec_command(cmd, timeout=10)
            result = stdout.read().decode("utf-8", errors="replace").strip()
            if result == "3.11":
                return candidate
        return None

    def _parse_line(self, line: str) -> None:
        """Parse structured output markers from remote install_script.py."""
        m = _STEP_RE.match(line)
        if m:
            cur, total, desc = int(m.group(1)), int(m.group(2)), m.group(3)
            self.step_changed.emit(cur, total, desc)
            self.progress.emit(f"[{cur}/{total}] {desc}")
            return
        m = _OK_RE.match(line)
        if m:
            self.progress.emit(f"OK: {m.group(1)}")
            return
        m = _WARN_RE.match(line)
        if m:
            self.progress.emit(f"WARN: {m.group(1)}")
            return
        m = _ERROR_RE.match(line)
        if m:
            self.progress.emit(f"ERROR: {m.group(1)}")
            return
        m = _ROOT_RE.match(line)
        if m:
            self._isaac_lab_root = m.group(1).strip()
            return
        if _DONE_RE.match(line):
            self.progress.emit("Remote installation complete!")
            return

    def _register_server(self) -> None:
        """Register the deployed server in the unified engine registry."""
        try:
            from src.system.engines.registry import get_engine_registry

            isaac_root = self._isaac_lab_root or self._remote_install_dir
            launcher = isaac_root.rstrip("/") + "/IsaacLab/isaaclab.sh"

            server_dict = {
                "name": self._server_name,
                "host": self._host,
                "port": self._port,
                "username": self._username,
                "auth_method": self._auth_method,
                "private_key_path": self._private_key_path,
                "remote_isaac_lab_path": isaac_root.rstrip("/") + "/IsaacLab",
                "remote_isaac_lab_launcher": launcher,
                "remote_python": "",
                "remote_work_dir": "/tmp/unitport_training",
                "conda_env": "",
                "upload_timeout": 120,
                "download_timeout": 300,
            }
            registry = get_engine_registry()
            registry.add_server("isaac_lab", server_dict)
            log.info(
                "[cloud-deploy] Server '%s' registered in engine registry",
                self._server_name,
            )
        except Exception as exc:
            log.warning("[cloud-deploy] Failed to register server: %s", exc)
