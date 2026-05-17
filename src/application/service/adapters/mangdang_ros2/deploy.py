"""mangdang_ros2/deploy.py — first-time deploy of unitport bringup.

Phase 3.7 invasive RepairAction. Fires when
:class:`MangdangServiceActiveProbe` reports ``unit_not_deployed`` (the
robot is in factory state with no UnitPort installation). Replaces the
"please SSH in and run apt + colcon yourself" hint with a one-click
install path.

Adapted from DEMO ``src/system/runtime/ros2/onboarding/bridge_upgrade.py``
with the following minimum-viable simplifications:

* No USB ``/upgrade`` HTTP fast-path — go straight to SSH+tar.
* No workspace-fingerprint computation / post-reboot fingerprint verify
  (the simple "USB identity daemon answers again" check is sufficient
  for v1; full fingerprint verification lands with the Phase 4 wizard).
* No background credential prompt — the SshCredentialPromptDialog
  upstream of AutoRepairLoop already collected the SSH password before
  we got here.

Steps:

1. Pack the brand-colocated ``workspace/unitport_ros2`` directory
   (sibling of this file) into an in-memory tar.gz with root:root
   ownership + executable bits set on ``scripts/*.sh``. CRLF
   normalisation for shell / yaml / service / target / sudoers files
   (Windows checkouts).
2. SFTP upload tar to ``/tmp/unitport_ros2_ws.tar`` on the robot.
3. ``sudo`` extract into ``/opt/unitport/ros2_ws/src/`` with
   rename-aside to survive partial prior installs.
4. ``sudo bash /opt/unitport/ros2_ws/src/unitport_ros2/scripts/bootstrap.sh``
   via ``exec_stream`` with a 600s idle timeout (colcon build can sit
   silent for 60-120s per package).
5. ``sudo reboot`` (transport drop is the success signal for this step).
6. Poll the USB identity daemon for up to 5min until it answers again.
7. Return success — Ros2ConnectionController auto-reconnects.

The deploy payload lives at
``application/service/adapters/mangdang_ros2/workspace/unitport_ros2/``
— brand-colocated per the Multi-Brand Inclusiveness rule. It is NOT
under ``RELEASE/runtime/`` (that is ``Paths.RUNTIME_DIR``, reserved
for dynamically-built cache artifacts) nor under ``src/runtime/``
(same — cache directory, not source).
"""

from __future__ import annotations

import io
import shlex
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from unitport_sdk import (
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
)

from application.service.connection.ssh_session import SSHSession, SshError
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import RepairAction


_LOG_TAG = "[deploy]"

# Robot-side install paths — match bootstrap.sh's expectation. DO NOT change
# without updating workspace/unitport_ros2/scripts/bootstrap.sh (sibling of
# this file) at the same time.
_REMOTE_WS_ROOT = "/opt/unitport/ros2_ws"
_REMOTE_WS_SRC = f"{_REMOTE_WS_ROOT}/src"
_REMOTE_TAR_STAGING = "/tmp/unitport_ros2_ws.tar"
_REMOTE_INSTALLER = (
    f"{_REMOTE_WS_ROOT}/src/unitport_ros2/scripts/bootstrap.sh"
)

# Timeouts — bootstrap.sh wraps a CM4 colcon build that can run 5-10min.
_BOOTSTRAP_TOTAL_TIMEOUT = 900.0   # 15 min hard cap
_BOOTSTRAP_IDLE_TIMEOUT = 600.0    # 10 min between any output line
_EXTRACT_TIMEOUT = 60.0
_REBOOT_GRACE_S = 8.0
_RECOVERY_TIMEOUT_S = 300.0
_RECOVERY_POLL_S = 3.0

# Files whose contents must be LF-only on the target. Windows checkouts
# with autocrlf=true would otherwise ship CRLF that bash + systemd refuse
# to parse (`line 14: $'\r': command not found`, etc.).
_LF_REQUIRED_SUFFIXES = (
    ".sh", ".bash",
    ".service", ".target", ".socket", ".timer",
    ".yaml", ".yml", ".json",
    ".sudoers",
    ".conf",
)

# USB identity probe — used both as a pre-flight liveness check and as
# the "robot is back from reboot" signal in _await_recovery.
_USB_IDENTITY_HOST = "192.168.55.1"
_USB_IDENTITY_PORT = 9999


# ---------------------------------------------------------------------- core


def _locate_workspace() -> Optional[Path]:
    """Return the brand-colocated ``workspace/unitport_ros2`` directory.

    The deploy payload is shipped alongside this module under
    ``application/service/adapters/mangdang_ros2/workspace/unitport_ros2/``
    — Mangdang-specific colcon packages + systemd units, kept inside the
    brand adapter per the Multi-Brand Inclusiveness rule (CLAUDE.md §1.1).
    """
    candidate = Path(__file__).resolve().parent / "workspace" / "unitport_ros2"
    return candidate if candidate.is_dir() else None


def _pack_workspace(local: Path) -> bytes:
    """Tar the workspace with root:root + 0755 on shell scripts.

    The robot bootstrap runs as root and bootstrap.sh has no chown step;
    files land under /opt/unitport with whatever ownership the tar
    declares. We force root:root + LF normalisation so a Windows-checked-
    out workspace still extracts cleanly on the robot.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:

        def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo:
            if ti.name.endswith(".sh") or "/scripts/" in ti.name:
                ti.mode = 0o755
            elif ti.isdir():
                ti.mode = 0o755
            else:
                ti.mode = 0o644
            ti.uid = 0
            ti.gid = 0
            ti.uname = "root"
            ti.gname = "root"
            return ti

        local_abs = local.resolve()

        def _needs_lf(path: Path) -> bool:
            return (
                path.is_file()
                and path.suffix.lower() in _LF_REQUIRED_SUFFIXES
            )

        def _add(path: Path, arcname: str) -> None:
            ti = tf.gettarinfo(str(path), arcname=arcname)
            if ti is None:
                return
            ti = _filter(ti)
            if path.is_dir():
                tf.addfile(ti)
                return
            if path.is_symlink() or not path.is_file():
                tf.add(str(path), arcname=arcname, filter=_filter)
                return
            if _needs_lf(path):
                raw = path.read_bytes()
                normalised = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                ti.size = len(normalised)
                tf.addfile(ti, io.BytesIO(normalised))
                return
            with path.open("rb") as fh:
                tf.addfile(ti, fh)

        # Exclude host-side build / IDE / VCS artifacts — they are useless
        # on the robot (Python 3.11 bytecode won't match the robot's
        # 3.10 install, colcon will re-build everything) and bloat the
        # tar.
        _SKIP_DIR_NAMES = {
            "__pycache__", ".git", ".venv", ".venv311", ".idea", ".vscode",
            "build", "install", "log",
        }
        _SKIP_FILE_SUFFIXES = (".pyc", ".pyo")

        def _skip(rel: Path) -> bool:
            for part in rel.parts:
                if part in _SKIP_DIR_NAMES:
                    return True
            return rel.suffix.lower() in _SKIP_FILE_SUFFIXES

        _add(local_abs, "unitport_ros2")
        for entry in sorted(local_abs.rglob("*")):
            try:
                rel = entry.relative_to(local_abs)
            except ValueError:
                continue
            if _skip(rel):
                continue
            arcname = "unitport_ros2/" + rel.as_posix()
            _add(entry, arcname)
    return buf.getvalue()


def _usb_identity_alive(host: str, timeout: float = 1.5) -> bool:
    """One-shot HTTP GET ``http://host:9999/identity`` — True iff 2xx."""
    url = f"http://{host}:{_USB_IDENTITY_PORT}/identity"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _await_recovery(
    usb_ip: str,
    *,
    timeout_s: float = _RECOVERY_TIMEOUT_S,
) -> bool:
    """Poll the USB identity daemon until the robot returns or we time out.

    Returns True when the daemon answers, False on timeout. We sleep an
    initial grace period for sshd to actually drop before polling, then
    poll every ``_RECOVERY_POLL_S`` seconds.
    """
    log_info(
        f"{_LOG_TAG} waiting {int(_REBOOT_GRACE_S)}s grace period for "
        "sshd to drop ..."
    )
    time.sleep(_REBOOT_GRACE_S)
    deadline = time.monotonic() + max(30.0, float(timeout_s))
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        if _usb_identity_alive(usb_ip):
            log_success(
                f"{_LOG_TAG} USB identity back at {usb_ip} "
                f"(after {iteration} probe(s))"
            )
            return True
        remaining = max(0, int(deadline - time.monotonic()))
        if iteration % 10 == 1:  # log every ~30s
            log_debug(
                f"{_LOG_TAG} robot still rebooting ({remaining}s remaining) ..."
            )
        time.sleep(_RECOVERY_POLL_S)
    log_error(
        f"{_LOG_TAG} robot did not return within {int(timeout_s)}s — "
        "manual investigation required"
    )
    return False


def _resolve_target(ctx: DiagnosticContext) -> tuple:
    """Pick (host, ssh_user, ssh_port) for the SSH install session.

    Order of preference:
    1. USB tether if the identity daemon already answers — directly use
       192.168.55.1:22 (sshd is on the same gadget interface).
    2. Else fall back to ``profile.pupper_ip`` over LAN if set.
    """
    profile = ctx.profile
    if profile is None:
        raise SshError("ssh_no_profile", "no ConnectionProfile in context")
    user = (getattr(profile, "ssh_user", "") or "ubuntu").strip() or "ubuntu"
    if _usb_identity_alive(_USB_IDENTITY_HOST):
        return _USB_IDENTITY_HOST, user, 22
    pupper_ip = (getattr(profile, "pupper_ip", "") or "").strip()
    if pupper_ip:
        return pupper_ip, user, 22
    raise SshError(
        "ssh_no_target",
        "neither USB identity daemon nor pupper_ip is reachable; "
        "check the USB cable / power",
    )


# ---------------------------------------------------------------------- repair


def _deploy_unitport(ctx: DiagnosticContext) -> None:
    """The repair-action body. Raises on any failure."""
    workspace = _locate_workspace()
    if workspace is None:
        raise SshError(
            "deploy_no_workspace",
            "application/service/adapters/mangdang_ros2/workspace/"
            "unitport_ros2 not found in tree",
        )
    log_info(f"{_LOG_TAG} starting first-time unitport deploy")

    host, user, port = _resolve_target(ctx)
    log_info(f"{_LOG_TAG} target: {user}@{host}:{port}")

    # Pack workspace.
    tar_bytes = _pack_workspace(workspace)
    size_kb = len(tar_bytes) // 1024
    log_info(
        f"{_LOG_TAG} packed workspace: {size_kb} KB "
        f"({len(workspace.rglob('*') and list(workspace.rglob('*')))} files)"
    )

    # Open a fresh SSH session — we do NOT reuse ctx.ssh because the
    # diagnose path opened it for short-lived probes; this install runs
    # for ~10 min and must own its session lifecycle.
    sudo_pw = getattr(ctx.ssh, "_sudo_password", None) if ctx.ssh else None
    pw = getattr(ctx.ssh, "_password", None) if ctx.ssh else None
    if not pw:
        raise SshError(
            "deploy_no_ssh_password",
            "SSH password unavailable — save it in the connection card "
            "and re-run",
        )
    if not sudo_pw:
        sudo_pw = pw  # default: same as login password

    ssh = SSHSession(
        host=host, user=user, password=pw, port=port,
        timeout=30.0, sudo_password=sudo_pw,
    )
    try:
        ssh.connect()
        log_debug(f"{_LOG_TAG} SSH connected")

        # Upload tar.
        last_pct = [-5]

        def _scp_progress(sent: int, total: int) -> None:
            if total <= 0:
                return
            pct = int(sent * 100 / total)
            if pct >= last_pct[0] + 10 or pct == 100:
                last_pct[0] = pct
                log_debug(
                    f"{_LOG_TAG}   scp {pct:3d}%  "
                    f"({sent // 1024}/{total // 1024} KB)"
                )

        log_info(
            f"{_LOG_TAG} uploading {size_kb} KB tar to {_REMOTE_TAR_STAGING} ..."
        )
        ssh.put_file_bytes(
            _REMOTE_TAR_STAGING, tar_bytes, progress_cb=_scp_progress,
        )
        log_success(f"{_LOG_TAG} upload complete")

        # Extract with rename-aside so partial prior installs don't break us.
        extract_script = (
            "set -e; "
            f"mkdir -p {_REMOTE_WS_SRC}; "
            f"if [ -e {_REMOTE_WS_SRC}/unitport_ros2 ]; then "
            "  ts=$(date +%s)_$$; "
            f"  mv {_REMOTE_WS_SRC}/unitport_ros2 "
            f"     {_REMOTE_WS_SRC}/.unitport_ros2.prev.$ts; "
            f"  chattr -R -i {_REMOTE_WS_SRC}/.unitport_ros2.prev.$ts "
            "     2>/dev/null || true; "
            f"  rm -rf {_REMOTE_WS_SRC}/.unitport_ros2.prev.$ts "
            "     2>/dev/null || true; "
            "fi; "
            f"tar --no-same-owner --touch -xf {_REMOTE_TAR_STAGING} "
            f"    -C {_REMOTE_WS_SRC}; "
            f"chown -R root:root {_REMOTE_WS_SRC}/unitport_ros2; "
            f"chmod +x {_REMOTE_WS_SRC}/unitport_ros2/scripts/*.sh; "
            f"rm -f {_REMOTE_TAR_STAGING}"
        )
        log_info(f"{_LOG_TAG} extracting on robot ...")
        res = ssh.exec_stream(
            f"bash -c {shlex.quote(extract_script)}",
            sudo_password=sudo_pw,
            timeout=_EXTRACT_TIMEOUT,
            idle_timeout=_EXTRACT_TIMEOUT,
        )
        if res.exit_code != 0:
            raise SshError(
                "deploy_extract_failed",
                f"extract failed (exit {res.exit_code}): "
                f"{(res.stderr or res.stdout)[-400:]}",
            )
        log_success(f"{_LOG_TAG} extract complete")

        # Run bootstrap.sh — long, streams output line-by-line.
        log_info(
            f"{_LOG_TAG} running bootstrap.sh — colcon build can take 5-10 "
            "minutes; output streams below"
        )

        def _on_line(line: str) -> None:
            text = line.rstrip()
            if text:
                log_debug(f"{_LOG_TAG}   {text}")

        res = ssh.exec_stream(
            f"bash {_REMOTE_INSTALLER}",
            line_callback=_on_line,
            sudo_password=sudo_pw,
            timeout=_BOOTSTRAP_TOTAL_TIMEOUT,
            idle_timeout=_BOOTSTRAP_IDLE_TIMEOUT,
        )
        if res.exit_code != 0:
            tail = (res.stderr or res.stdout or "")[-800:]
            raise SshError(
                "deploy_bootstrap_failed",
                f"bootstrap.sh failed (exit {res.exit_code}); tail:\n{tail}",
            )
        log_success(f"{_LOG_TAG} bootstrap.sh completed cleanly")

        # Trigger reboot. The session WILL drop mid-command — that's the
        # success signal that reboot was actually issued.
        log_info(
            f"{_LOG_TAG} rebooting robot (sshd will drop momentarily) ..."
        )
        try:
            ssh.exec_stream(
                "/bin/sh -c 'nohup /sbin/reboot </dev/null "
                ">/dev/null 2>&1 &'",
                sudo_password=sudo_pw,
                timeout=10.0,
                idle_timeout=10.0,
            )
        except SshError as exc:
            # Connection drop is expected — log at debug level.
            log_debug(
                f"{_LOG_TAG} reboot command terminated SSH (expected): {exc}"
            )
    finally:
        try:
            ssh.close()
        except Exception:
            pass

    # Wait for the robot to come back up.
    log_info(
        f"{_LOG_TAG} waiting up to {int(_RECOVERY_TIMEOUT_S)}s for robot to "
        "reboot and re-expose the USB identity daemon ..."
    )
    if not _await_recovery(_USB_IDENTITY_HOST):
        raise SshError(
            "deploy_recovery_timeout",
            f"robot did not return within {int(_RECOVERY_TIMEOUT_S)}s after "
            "reboot — check the robot's serial console for boot errors",
        )
    log_success(
        f"{_LOG_TAG} unitport bringup deployed and robot is back online"
    )


_REPAIR_DEPLOY_UNITPORT = RepairAction(
    name="mangdang.deploy_unitport_bringup",
    describe=(
        "First-time install of unitport bringup on robot: SFTP push "
        "workspace, sudo bootstrap.sh (colcon build, ~10 min), reboot, "
        "wait for recovery"
    ),
    run=_deploy_unitport,
    safe=False,  # invasive: modifies /opt/unitport/, /etc/systemd/, reboots
)


__all__ = ["_REPAIR_DEPLOY_UNITPORT"]
