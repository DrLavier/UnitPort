# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SSHSession — paramiko facade used by Phase 3.5 diagnostics & repair.

Ported from ``DEMO/src/system/runtime/ros2/onboarding/ssh_session.py`` with
two RELEASE-specific additions:

* :meth:`exec_sudo` — feeds a sudo password to stdin so non-interactive
  shells can run privileged commands without a TTY.
* :meth:`write_file` — uploads bytes/text to a remote path; with
  ``sudo=True`` it pipes through ``sudo tee`` so writes under
  ``/etc/...`` succeed without owning the file as the SSH user.

The class is brand-agnostic — credentials are resolved by the caller via
:class:`SecureCredentialStore` (per ``server_name`` slot) and passed in.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import paramiko
except ImportError as exc:  # pragma: no cover — hard dependency
    raise RuntimeError(
        "paramiko is required for connection diagnostics SSH; "
        "run `pip install paramiko>=3.0`"
    ) from exc


class SshError(RuntimeError):
    """Single typed exception SSH consumers see — wraps paramiko internals."""

    def __init__(self, error_code: str, message: str = "") -> None:
        super().__init__(message or error_code)
        self.error_code = error_code


@dataclass
class SSHResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class SSHSession:
    """Thin paramiko facade.

    Holds one SSHClient + reuses it for exec/put/get. Callers are expected
    to close the session via :meth:`close` or the context-manager protocol.
    """

    def __init__(
        self,
        host: str,
        user: str = "ubuntu",
        password: Optional[str] = None,
        port: int = 22,
        key_filename: Optional[str] = None,
        timeout: float = 30.0,
        sudo_password: Optional[str] = None,
    ) -> None:
        self.host = host
        self.user = user
        self.port = int(port)
        self._password = password
        self._sudo_password = sudo_password
        self._key_filename = key_filename
        self._timeout = float(timeout)
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._client is not None:
            return
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=self._password,
                key_filename=self._key_filename,
                timeout=self._timeout,
                allow_agent=True,
                look_for_keys=True,
            )
        except paramiko.AuthenticationException as exc:
            raise SshError("ssh_auth_failed", str(exc)) from exc
        except paramiko.SSHException as exc:
            raise SshError("ssh_protocol_error", str(exc)) from exc
        except OSError as exc:
            raise SshError("ssh_unreachable", str(exc)) from exc
        # Application-level keepalive — see DEMO history note.
        try:
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
        except Exception:
            pass
        self._client = client

    def is_alive(self) -> bool:
        if self._client is None:
            return False
        try:
            transport = self._client.get_transport()
            return bool(transport and transport.is_active())
        except Exception:
            return False

    def reconnect(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        self.connect()

    def close(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "SSHSession":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # exec
    # ------------------------------------------------------------------

    def exec(self, command: str, timeout: float = 30.0) -> SSHResult:
        """Run a shell command; return its stdout/stderr/exit_code."""
        if self._client is None:
            self.connect()
        assert self._client is not None
        try:
            _, stdout, stderr = self._client.exec_command(
                command, timeout=timeout,
            )
        except paramiko.SSHException as exc:
            raise SshError("ssh_exec_failed", str(exc)) from exc
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return SSHResult(stdout=out, stderr=err, exit_code=int(exit_code))

    def exec_sudo(
        self,
        command: str,
        sudo_password: Optional[str] = None,
        timeout: float = 30.0,
    ) -> SSHResult:
        """Run a command via ``sudo -S`` and feed the password to stdin.

        Falls back to ``self._sudo_password`` when no password argument is
        passed. Raises :class:`SshError` with ``ssh_no_sudo_password`` when
        neither is available — diagnostics path can convert this to an
        INFO finding asking the user to populate the credential.
        """
        pw = sudo_password if sudo_password is not None else self._sudo_password
        if not pw:
            raise SshError(
                "ssh_no_sudo_password",
                "sudo password not provided. Save it via the sec1 SSH "
                "credentials section.",
            )
        if self._client is None:
            self.connect()
        assert self._client is not None
        # ``sudo -S -p ''`` reads password from stdin without echoing a prompt.
        full = f"sudo -S -p '' {command}"
        try:
            stdin, stdout, stderr = self._client.exec_command(
                full, timeout=timeout, get_pty=False,
            )
        except paramiko.SSHException as exc:
            raise SshError("ssh_exec_failed", str(exc)) from exc
        try:
            stdin.write(pw + "\n")
            stdin.flush()
        except Exception:
            pass
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        # sudo prints "[sudo] password for ..." to stderr on failure; trim
        # leading prompts so the repair-task log shows the actual error.
        err = "\n".join(
            line for line in err.splitlines()
            if not line.startswith("[sudo]")
            and "password for" not in line
        )
        return SSHResult(stdout=out, stderr=err, exit_code=int(exit_code))

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _sftp_client(self) -> paramiko.SFTPClient:
        if self._client is None:
            self.connect()
        assert self._client is not None
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def put(self, local_path: str, remote_path: str) -> None:
        sftp = self._sftp_client()
        sftp.put(local_path, remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        sftp = self._sftp_client()
        sftp.get(remote_path, local_path)

    def put_file_bytes(
        self,
        remote_path: str,
        data: bytes,
        *,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Upload an in-memory bytes blob to ``remote_path`` via SFTP.

        Used by the unitport deploy path to push a tarred workspace
        without touching the local filesystem. ``progress_cb(sent, total)``
        is invoked roughly every 5% so the UI / cmd_log can report
        progress over the typical 30-60s tar push.
        """
        sftp = self._sftp_client()
        total = len(data)
        with sftp.file(remote_path, "wb") as fh:
            fh.set_pipelined(True)
            chunk = max(1, total // 20)  # ~5% slices for progress callback
            sent = 0
            while sent < total:
                end = min(sent + chunk, total)
                fh.write(data[sent:end])
                sent = end
                if progress_cb is not None:
                    try:
                        progress_cb(sent, total)
                    except Exception:
                        pass

    def exec_stream(
        self,
        command: str,
        line_callback: Optional[Callable[[str], None]] = None,
        *,
        sudo_password: Optional[str] = None,
        timeout: float = 900.0,
        idle_timeout: float = 180.0,
    ) -> SSHResult:
        """Run a long-lived command, streaming stdout lines as they arrive.

        Required for the bootstrap.sh path: colcon build can spend 60-120s
        silently compiling a single package, which would trip the default
        :meth:`exec` channel timeout. This method polls the channel and
        only fails the command when **no output** has arrived for
        ``idle_timeout`` seconds — colcon's per-package "Starting" /
        "Finished" lines happen often enough to keep us alive.

        ``line_callback`` is invoked with each newline-terminated stdout
        line as it arrives; useful for streaming progress to cmd_log.

        ``sudo_password`` triggers ``sudo -S -p '' <command>`` and feeds
        the password to stdin, mirroring :meth:`exec_sudo`.
        """
        if self._client is None:
            self.connect()
        assert self._client is not None

        full = command
        if sudo_password is not None:
            full = f"sudo -S -p '' {command}"

        try:
            stdin, stdout, stderr = self._client.exec_command(
                full, timeout=timeout, get_pty=False,
            )
        except paramiko.SSHException as exc:
            raise SshError("ssh_exec_failed", str(exc)) from exc

        if sudo_password is not None:
            try:
                stdin.write(sudo_password + "\n")
                stdin.flush()
            except Exception:
                pass

        chan = stdout.channel
        chan.settimeout(0.5)
        deadline = time.monotonic() + max(1.0, timeout)
        last_activity = time.monotonic()
        out_buf = bytearray()
        err_buf = bytearray()
        line_pending = bytearray()

        while True:
            now = time.monotonic()
            if now > deadline:
                try:
                    chan.close()
                except Exception:
                    pass
                raise SshError(
                    "ssh_exec_timeout",
                    f"command exceeded total timeout {timeout:.0f}s",
                )
            if (now - last_activity) > idle_timeout:
                try:
                    chan.close()
                except Exception:
                    pass
                raise SshError(
                    "ssh_exec_idle_timeout",
                    f"no output for {idle_timeout:.0f}s; command "
                    "appears stalled",
                )

            progressed = False
            if chan.recv_ready():
                data = chan.recv(65536)
                if data:
                    out_buf += data
                    line_pending += data
                    last_activity = now
                    progressed = True
                    if line_callback is not None:
                        # Flush every complete line so the UI sees colcon
                        # progress as it streams.
                        while True:
                            nl = line_pending.find(b"\n")
                            if nl < 0:
                                break
                            line = bytes(line_pending[:nl]).decode(
                                "utf-8", errors="replace",
                            )
                            del line_pending[: nl + 1]
                            try:
                                line_callback(line)
                            except Exception:
                                pass

            if chan.recv_stderr_ready():
                edata = chan.recv_stderr(65536)
                if edata:
                    err_buf += edata
                    last_activity = now
                    progressed = True

            if chan.exit_status_ready() and not chan.recv_ready() \
                    and not chan.recv_stderr_ready():
                break

            if not progressed:
                # Brief sleep to avoid burning CPU on the busy loop.
                time.sleep(0.05)

        # Drain anything left after exit.
        try:
            while chan.recv_ready():
                out_buf += chan.recv(65536)
            while chan.recv_stderr_ready():
                err_buf += chan.recv_stderr(65536)
        except Exception:
            pass

        # Flush any trailing partial line through the callback.
        if line_callback is not None and line_pending:
            try:
                line_callback(
                    bytes(line_pending).decode("utf-8", errors="replace")
                )
            except Exception:
                pass

        exit_code = chan.recv_exit_status()
        out = bytes(out_buf).decode("utf-8", errors="replace")
        err = bytes(err_buf).decode("utf-8", errors="replace")
        # Strip the inevitable "[sudo] password for <user>:" prompt from stderr
        # so log lines show only the actual command's diagnostics.
        if sudo_password is not None:
            err = "\n".join(
                line for line in err.splitlines()
                if not line.startswith("[sudo]")
                and "password for" not in line
            )
        return SSHResult(stdout=out, stderr=err, exit_code=int(exit_code))

    def write_file(
        self,
        remote_path: str,
        content: str | bytes,
        *,
        sudo: bool = False,
        sudo_password: Optional[str] = None,
        mode: Optional[int] = None,
    ) -> None:
        """Upload ``content`` to ``remote_path``.

        ``sudo=False`` uses SFTP — the SSH user must already own the path.
        ``sudo=True`` pipes through ``sudo tee`` so writes under
        ``/etc/`` / ``/usr/`` succeed; uses the persisted ``sudo_password``
        unless overridden.

        ``mode`` (octal int) is applied via ``chmod`` after the write when
        provided. Useful for service drop-ins (``0o644``).
        """
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        if sudo:
            # ``cat << 'UNITPORT_EOF' | sudo tee remote_path`` keeps the
            # heredoc literal so $variables / backticks in the content do
            # not expand. tee writes to stdout too — discard it.
            heredoc = (
                "cat << 'UNITPORT_EOF' | sudo -S -p '' tee "
                + shlex.quote(remote_path)
                + " > /dev/null\n"
                + text
                + "\nUNITPORT_EOF"
            )
            if self._client is None:
                self.connect()
            assert self._client is not None
            pw = (
                sudo_password
                if sudo_password is not None
                else self._sudo_password
            )
            if not pw:
                raise SshError(
                    "ssh_no_sudo_password",
                    "sudo password not provided for write_file(sudo=True)",
                )
            try:
                stdin, stdout, stderr = self._client.exec_command(
                    heredoc, timeout=30.0, get_pty=False,
                )
            except paramiko.SSHException as exc:
                raise SshError("ssh_exec_failed", str(exc)) from exc
            try:
                # sudo reads password from stdin first (we used -S -p '');
                # only after sudo authenticates does ``cat`` read the
                # heredoc. Sending the password before the heredoc is fine
                # — sudo grabs only the first newline-terminated line.
                stdin.write(pw + "\n")
                stdin.flush()
            except Exception:
                pass
            rc = stdout.channel.recv_exit_status()
            err = stderr.read().decode("utf-8", errors="replace")
            if rc != 0:
                raise SshError(
                    "ssh_write_file_failed",
                    f"sudo tee {remote_path} exit={rc}: {err.strip()}",
                )
            if mode is not None:
                # chmod via sudo as well — file is now root-owned.
                self.exec_sudo(
                    f"chmod {oct(mode)[2:]} {shlex.quote(remote_path)}",
                    sudo_password=pw,
                )
        else:
            sftp = self._sftp_client()
            with sftp.file(remote_path, "wb") as fh:
                fh.write(text.encode("utf-8"))
            if mode is not None:
                try:
                    sftp.chmod(remote_path, mode)
                except Exception:
                    pass


__all__ = ["SSHSession", "SSHResult", "SshError"]
