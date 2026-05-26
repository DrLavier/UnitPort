# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Mangdang-specific diagnostic probes.

These all require SSH to the robot. They cover the three deployment
issues the user hit on a real mini_pupper after Phase 3:

* Service inactive — ``mangdang_unitport.service`` not running.
* RMW missing — ``ros-humble-rmw-cyclonedds-cpp`` not installed (so the
  service uses fastrtps and host CycloneDDS can't see anything).
* Service env empty — ``RMW_IMPLEMENTATION`` / ``ROS_DOMAIN_ID`` not
  injected via systemd ``Environment=`` so the colcon hooks erase them.

Each probe pairs the failure with a typed :class:`RepairAction` consumed
by :class:`RepairTask`.
"""

from __future__ import annotations

from typing import List, Optional

from application.service.adapters import topic_registry
from application.service.adapters.mangdang_ros2.conflict_probe import (
    MangdangServiceConflictProbe,
)
from application.service.adapters.mangdang_ros2.deploy import (
    _REPAIR_DEPLOY_UNITPORT,
)
from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import (
    DiagnosticFinding,
    RepairAction,
)


_SERVICE = "unitport-minipupper.service"
_RMW_PKG = "ros-humble-rmw-cyclonedds-cpp"
_DROPIN_PATH = f"/etc/systemd/system/{_SERVICE}.d/unitport.conf"
_DROPIN_TEMPLATE = """[Service]
Environment=RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
Environment=ROS_DOMAIN_ID={domain_id}
"""


# ----------------------------------------------------------------------
# Repair actions (module-level so they can be referenced from findings)
# ----------------------------------------------------------------------


def _restart_service(ctx: DiagnosticContext) -> None:
    if ctx.ssh is None:
        raise RuntimeError("ssh unavailable")
    res = ctx.ssh.exec_sudo(f"systemctl restart {_SERVICE}")
    if not res.ok:
        raise RuntimeError(
            f"systemctl restart exit={res.exit_code}: {res.stderr.strip()}"
        )


def _enable_and_start_service(ctx: DiagnosticContext) -> None:
    if ctx.ssh is None:
        raise RuntimeError("ssh unavailable")
    res = ctx.ssh.exec_sudo(f"systemctl enable --now {_SERVICE}")
    if not res.ok:
        raise RuntimeError(
            f"systemctl enable --now exit={res.exit_code}: {res.stderr.strip()}"
        )


def _install_cyclonedds(ctx: DiagnosticContext) -> None:
    if ctx.ssh is None:
        raise RuntimeError("ssh unavailable")
    # apt-get update first so the package index is fresh; -qq keeps log noise low.
    upd = ctx.ssh.exec_sudo("apt-get update -qq", timeout=120.0)
    if not upd.ok:
        raise RuntimeError(
            f"apt-get update exit={upd.exit_code}: {upd.stderr.strip()}"
        )
    inst = ctx.ssh.exec_sudo(
        f"apt-get install -y --no-install-recommends {_RMW_PKG}",
        timeout=300.0,
    )
    if not inst.ok:
        raise RuntimeError(
            f"apt-get install exit={inst.exit_code}: {inst.stderr.strip()}"
        )


def _write_dropin_and_restart(ctx: DiagnosticContext) -> None:
    if ctx.ssh is None:
        raise RuntimeError("ssh unavailable")
    domain_id = int(getattr(ctx.profile, "ros_domain_id", 42) or 42)
    content = _DROPIN_TEMPLATE.format(domain_id=domain_id)
    # Ensure the drop-in directory exists.
    mkd = ctx.ssh.exec_sudo(f"mkdir -p /etc/systemd/system/{_SERVICE}.d")
    if not mkd.ok:
        raise RuntimeError(
            f"mkdir drop-in dir exit={mkd.exit_code}: {mkd.stderr.strip()}"
        )
    ctx.ssh.write_file(_DROPIN_PATH, content, sudo=True, mode=0o644)
    rel = ctx.ssh.exec_sudo("systemctl daemon-reload")
    if not rel.ok:
        raise RuntimeError(
            f"daemon-reload exit={rel.exit_code}: {rel.stderr.strip()}"
        )
    _restart_service(ctx)


_REPAIR_RESTART = RepairAction(
    name="mangdang.restart_service",
    describe=f"sudo systemctl restart {_SERVICE}",
    run=_restart_service,
    safe=True,
)

_REPAIR_ENABLE_NOW = RepairAction(
    name="mangdang.enable_now_service",
    describe=f"sudo systemctl enable --now {_SERVICE}",
    run=_enable_and_start_service,
    safe=True,
)

_REPAIR_INSTALL_RMW = RepairAction(
    name="mangdang.install_cyclonedds",
    describe=(
        f"sudo apt-get install -y {_RMW_PKG} (also runs apt-get update first)"
    ),
    run=_install_cyclonedds,
    safe=False,
)

_REPAIR_WRITE_DROPIN = RepairAction(
    name="mangdang.write_dds_dropin",
    describe=(
        f"write {_DROPIN_PATH} with RMW_IMPLEMENTATION + ROS_DOMAIN_ID, "
        "daemon-reload, restart service"
    ),
    run=_write_dropin_and_restart,
    safe=True,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_state(ctx: DiagnosticContext, unit: str) -> str:
    """Return systemd ``LoadState`` for ``unit`` over SSH; empty on failure.

    Possible values (systemd 247+):

    * ``loaded``    — unit file present and parsed; may be active/inactive
    * ``not-found`` — no unit file by this name on the robot
    * ``masked``    — admin-disabled
    * ``error``     — unit file present but failed to parse

    Distinguishing ``not-found`` from ``loaded`` is what lets us avoid
    queueing a doomed restart/enable repair for a robot that simply has
    no unitport bringup deployed.
    """
    if ctx.ssh is None:
        return ""
    try:
        res = ctx.ssh.exec(f"systemctl show -p LoadState --value {unit}")
    except Exception:
        return ""
    return (res.stdout or "").strip()


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------


class MangdangServiceActiveProbe(DiagnosticProbe):
    """Verify ``mangdang_unitport.service`` is active (running)."""

    id = "mangdang_service_active"
    requires_ssh = True

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        if ctx.ssh is None:
            return [self.info("ssh_unavailable")]

        load_state = _load_state(ctx, _SERVICE)
        if load_state == "not-found":
            # Unit file isn't on the robot at all -> factory state. Attach
            # the invasive deploy repair: SFTP push the bundled workspace,
            # sudo bootstrap.sh (~10 min colcon build), reboot, await
            # recovery. Marked safe=False so it only runs through the
            # ResultDialog [Apply Invasive Fixes] confirmation path.
            return [self.error(
                f"{_SERVICE} not deployed on robot (LoadState=not-found)",
                detail=(
                    "The unitport bringup unit isn't installed on this "
                    "robot — it's in factory state. The invasive 'deploy "
                    "unitport bringup' fix bundles a fresh workspace, "
                    "uploads it via SFTP, runs bootstrap.sh under sudo "
                    "(colcon build takes ~5-10 min on a CM4), then reboots "
                    "the robot and waits for it to come back online.\n\n"
                    "After deploy completes, the connection auto-retries "
                    "and the rest of the auto-repair loop (RMW install, "
                    "env drop-in, etc.) takes over."
                ),
                repair=_REPAIR_DEPLOY_UNITPORT,
            )]
        if load_state == "masked":
            return [self.error(
                f"{_SERVICE} is masked on the robot",
                detail=(
                    "An admin disabled this unit with `systemctl mask`. "
                    f"Run `sudo systemctl unmask {_SERVICE}` on the robot "
                    "before re-trying connect."
                ),
                repair=None,
            )]
        if load_state and load_state not in {"loaded"}:
            return [self.warning(
                f"{_SERVICE} LoadState={load_state}",
                detail="Unexpected systemd state; manual inspection required.",
            )]

        # Unit IS loaded — fall through to the active/inactive check.
        res = ctx.ssh.exec(f"systemctl is-active {_SERVICE}")
        state = (res.stdout or "").strip()
        if state == "active":
            return [self.ok(f"{_SERVICE} active")]
        # Distinguish "not enabled" from "failed" so the dialog suggests
        # the right repair: enable+start vs restart.
        if state in {"inactive", "dead", "failed"}:
            enabled = ctx.ssh.exec(f"systemctl is-enabled {_SERVICE}")
            ena = (enabled.stdout or "").strip()
            repair = (
                _REPAIR_RESTART if ena == "enabled" else _REPAIR_ENABLE_NOW
            )
            return [self.error(
                f"{_SERVICE} is {state} (enabled={ena})",
                detail=(
                    "The Mangdang ROS2 controller service is not running. "
                    "Without it the robot publishes nothing on DDS — host "
                    "sees the bridge alive but no traffic."
                ),
                repair=repair,
            )]
        return [self.warning(
            f"{_SERVICE} state unknown: {state or '(empty)'}",
            detail=res.stderr.strip(),
        )]


class MangdangRmwDetectionProbe(DiagnosticProbe):
    """Phase 3.9 — Auto middleware: detect robot's actual RMW via SSH.

    Writes the discovered ``RMW_IMPLEMENTATION`` value (e.g.
    ``rmw_fastrtps_cpp``) into ``ctx.profile.detected_rmw`` so downstream
    probes — notably :class:`MangdangRmwInstalledProbe` — can soften
    their severity when the host is in Auto mode and the robot is
    already running some functioning RMW.

    Skipped (returns INFO, never ERROR) when:
      * profile.middleware != "Auto" (user explicitly picked an RMW;
        strict probes apply).
      * ctx.ssh is None (no creds — Auto stays in pure tolerant mode).
    """

    id = "mangdang_rmw_detection"
    requires_ssh = True

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        middleware = str(getattr(ctx.profile, "middleware", "") or "")
        if middleware != "Auto":
            return [self.info("middleware not Auto; detection skipped")]
        if ctx.ssh is None:
            return [self.info("ssh unavailable; Auto stays in tolerant mode")]
        res = ctx.ssh.exec(f"systemctl show -p Environment {_SERVICE}")
        body = (res.stdout or "").strip()
        rmw = ""
        if body.startswith("Environment="):
            for tok in body[len("Environment="):].split():
                k, _, v = tok.partition("=")
                if k == "RMW_IMPLEMENTATION":
                    rmw = v
                    break
        if not rmw:
            return [self.info("RMW_IMPLEMENTATION not set in service env")]
        # Mutate the profile so MangdangRmwInstalledProbe (running later)
        # can read the detected RMW. Profile is a single instance per task.
        try:
            ctx.profile.detected_rmw = rmw
        except Exception:
            pass
        return [self.info(f"Auto detected RMW: {rmw}")]


class MangdangRmwInstalledProbe(DiagnosticProbe):
    """Verify ``ros-humble-rmw-cyclonedds-cpp`` is installed.

    Phase 3.9: when the host's middleware is "Auto" the missing-package
    case is demoted from ERROR to INFO and the invasive apt-install
    repair is dropped — RTPS wire interop between host cyclonedds-python
    and robot factory FastDDS is acceptable, and pushing the user to
    apt-install on the robot violates the AutoConnect tolerance contract.
    Strict CycloneDDS mode (user explicitly picked it) keeps ERROR.
    """

    id = "mangdang_rmw_installed"
    requires_ssh = True

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        if ctx.ssh is None:
            return [self.info("ssh_unavailable")]
        res = ctx.ssh.exec(
            f"dpkg -s {_RMW_PKG} 2>/dev/null | grep -E '^Status:'"
        )
        line = (res.stdout or "").strip()
        if "install ok installed" in line:
            return [self.ok(f"{_RMW_PKG} installed")]

        middleware = str(getattr(ctx.profile, "middleware", "") or "")
        detected = str(getattr(ctx.profile, "detected_rmw", "") or "")
        if middleware == "Auto":
            return [self.info(
                f"{_RMW_PKG} not installed; Auto mode tolerates this "
                f"(robot RMW: {detected or 'unknown'})",
            )]
        return [self.error(
            f"{_RMW_PKG} not installed",
            detail=(
                "The robot's ROS 2 stack will use the default RMW "
                "(typically FastDDS). Host CycloneDDS bridge can still "
                "reach it over RTPS but interop is brittle. Either "
                "install the package on the robot, or switch the host's "
                "middleware to 'Auto' to suppress this warning."
            ),
            repair=_REPAIR_INSTALL_RMW,
        )]


class MangdangChampSubscriberProbe(DiagnosticProbe):
    """Verify the robot has CHAMP nodes subscribing the declared cmd topics.

    DEMO and Phase 3.5 service-active probe both checked whether the
    systemd unit is running. But systemd reports "active" even when the
    inner ``ros2 launch`` chain crashes inside ExecStart — the unit's
    main process is the launcher, not CHAMP, so CHAMP can be dead while
    the unit still reads as active.

    This probe asks the DDS wire directly: does ANY foreign DataReader
    exist for our declared cmd topics? Empty = CHAMP isn't actually up,
    regardless of what systemd thinks. Repair = restart the unit, which
    re-runs the launcher (idempotent + safe).
    """

    id = "mangdang_champ_subscriber"
    # SSH is optional for the DDS check itself, but when present we use
    # it to pull a journal snippet so the user sees CHAMP's actual error
    # inside the dialog without having to ssh in manually. Mark True so
    # the SSH credentials section auto-expands when this finding fires.
    requires_ssh = True

    # Tail length for the journal snippet — short enough to fit in the
    # dialog detail, long enough to capture a Python traceback.
    _JOURNAL_TAIL = 30

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        bridge = ctx.bridge
        adapter = ctx.adapter
        if bridge is None or adapter is None:
            return [self.info("bridge or adapter not active")]
        brand_id = getattr(adapter, "BRAND_ID", "") or ""
        cmd_specs = topic_registry.list_for_brand(brand_id, role="cmd")
        if not cmd_specs:
            return [self.info("no cmd-role topics declared")]

        # Lazy snapshot: only fetch journal if at least one cmd topic is
        # missing a subscriber AND ssh is available. Avoids a 100-200ms
        # SSH round trip on the happy path.
        journal_snippet: Optional[str] = None
        ssh_journal_attempted = False

        findings: List[DiagnosticFinding] = []
        for spec in cmd_specs:
            try:
                foreign = bridge.foreign_subscribers(spec.topic)
            except Exception as exc:
                findings.append(self.warning(
                    f"could not enumerate subscribers on {spec.topic}",
                    detail=str(exc),
                ))
                continue
            if foreign:
                findings.append(self.ok(
                    f"{spec.topic} subscribed by {len(foreign)} robot "
                    "node(s)",
                ))
                continue

            if not ssh_journal_attempted:
                journal_snippet = self._fetch_journal(ctx)
                ssh_journal_attempted = True

            detail = (
                f"The host's view of the DDS domain has 0 remote "
                f"DataReader on '{spec.topic}'. Even though "
                f"{_SERVICE} may report 'active' to systemd, its inner "
                f"ros2 launch chain (CHAMP / mini_pupper_bringup) is "
                f"not exposing this subscriber on the wire.\n\n"
                f"Safe fix restarts the service so the launch chain "
                f"re-runs from scratch."
            )
            if journal_snippet:
                detail += (
                    f"\n\n--- Recent journal ({_SERVICE}, last "
                    f"{self._JOURNAL_TAIL} lines) ---\n{journal_snippet}"
                )
            elif ctx.ssh is None:
                detail += (
                    "\n\nProvide SSH credentials in the connection card to "
                    "auto-include the recent service journal here, or run "
                    f"manually: sudo journalctl -u {_SERVICE} -n 100 "
                    "--no-pager"
                )
            else:
                detail += (
                    f"\n\nCould not retrieve recent journal over SSH; run "
                    f"manually: sudo journalctl -u {_SERVICE} -n 100 "
                    "--no-pager"
                )

            findings.append(self.error(
                f"no robot subscriber on {spec.topic} — CHAMP not running?",
                detail=detail,
                repair=_REPAIR_RESTART,
                requires_ssh=True,
            ))
        return findings

    def _fetch_journal(self, ctx: DiagnosticContext) -> Optional[str]:
        """Return last N lines of the service journal; None on failure."""
        if ctx.ssh is None:
            return None
        cmd = (
            f"journalctl -u {_SERVICE} -n {self._JOURNAL_TAIL} "
            "--no-pager 2>/dev/null"
        )
        try:
            res = ctx.ssh.exec(cmd)
        except Exception:
            return None
        out = (res.stdout or "").strip()
        if not out:
            # Some distros restrict journalctl to root for service logs;
            # fall back to sudo if the unprivileged read came back empty.
            try:
                res = ctx.ssh.exec_sudo(cmd, timeout=8.0)
            except Exception:
                return None
            out = (res.stdout or "").strip()
        return out or None


class MangdangServiceEnvProbe(DiagnosticProbe):
    """Verify systemd injects RMW_IMPLEMENTATION + ROS_DOMAIN_ID."""

    id = "mangdang_service_env"
    requires_ssh = True

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        if ctx.ssh is None:
            return [self.info("ssh_unavailable")]
        # Skip the env check entirely when the unit isn't deployed —
        # writing a drop-in for a non-existent unit is harmless but the
        # repair's daemon-reload + restart step would fail and clutter
        # the result dialog. Service-active probe already surfaces the
        # "unit_not_deployed" finding for the user to act on.
        if _load_state(ctx, _SERVICE) == "not-found":
            return [self.info(
                f"{_SERVICE} not deployed; env check skipped",
                detail=(
                    "See the mangdang_service_active finding for the "
                    "deployment hint. Once the unit is in place, re-run "
                    "[Connect] and this probe will validate the env."
                ),
            )]
        res = ctx.ssh.exec(f"systemctl show -p Environment {_SERVICE}")
        env = (res.stdout or "").strip()
        # Format: "Environment=KEY=VAL KEY=VAL ..." or "Environment="
        body = env.partition("=")[2]
        kv = {}
        for tok in body.split():
            k, _, v = tok.partition("=")
            if k:
                kv[k] = v
        rmw = kv.get("RMW_IMPLEMENTATION", "")
        domain = kv.get("ROS_DOMAIN_ID", "")
        wanted_domain = str(int(getattr(ctx.profile, "ros_domain_id", 42) or 42))
        problems = []
        if rmw != "rmw_cyclonedds_cpp":
            problems.append(
                f"RMW_IMPLEMENTATION={rmw or '(unset)'} (want rmw_cyclonedds_cpp)"
            )
        if domain != wanted_domain:
            problems.append(
                f"ROS_DOMAIN_ID={domain or '(unset)'} (want {wanted_domain})"
            )
        if not problems:
            return [self.ok(
                "service Environment carries cyclonedds + correct domain"
            )]
        return [self.error(
            "service Environment missing required vars",
            detail=(
                "Issues: " + "; ".join(problems) + ". "
                "The colcon setup hook chained inside ExecStart= overwrites "
                "EnvironmentFile= entries; we install a drop-in that uses "
                "Environment= so systemd injects the values into the exec'd "
                "process directly (the hooks then see them as already-set "
                "and skip)."
            ),
            repair=_REPAIR_WRITE_DROPIN,
        )]


def mangdang_probes() -> List[DiagnosticProbe]:
    """Return one fresh instance of each Mangdang probe.

    Order matters: ``MangdangRmwDetectionProbe`` runs before
    ``MangdangRmwInstalledProbe`` so the latter can read the detected
    RMW value from ``ctx.profile.detected_rmw`` when in Auto mode.
    """
    return [
        MangdangServiceActiveProbe(),
        MangdangRmwDetectionProbe(),
        MangdangRmwInstalledProbe(),
        MangdangServiceEnvProbe(),
        MangdangServiceConflictProbe(),
        MangdangChampSubscriberProbe(),
    ]


__all__ = [
    "MangdangServiceActiveProbe",
    "MangdangRmwDetectionProbe",
    "MangdangRmwInstalledProbe",
    "MangdangServiceEnvProbe",
    "MangdangServiceConflictProbe",
    "MangdangChampSubscriberProbe",
    "mangdang_probes",
]
