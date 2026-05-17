"""MangdangServiceConflictProbe — detect mini_pupper_web racing /cmd_vel.

The Mini Pupper factory image ships ``mini_pupper_web.service``: a Flask
control panel that owns the PCA9685 I2C bus and publishes its own zero
Twists on /cmd_vel as part of its idle loop. When it stays alive
alongside the unitport bringup unit, both publishers race on /cmd_vel and
the robot ignores teleop frames coming from the host.

DEMO solved this in two layers: (1) the unitport systemd unit declares
``Conflicts=mini_pupper_web.service`` so systemd auto-stops the web on
unit start; (2) ``ExecStopPost=`` restarts the web on unit stop for a
clean handoff. This probe verifies both halves of that contract on the
deployed robot, and offers safe repairs to install whichever piece is
missing.

The deployed systemd unit name is not consistent across deployment
revisions: DEMO's unit file is ``unitport-minipupper.service`` while
RELEASE's earlier Phase 3.5 probes referenced ``mangdang_unitport.service``.
We probe both candidates and report whichever is actually active.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import (
    DiagnosticFinding,
    RepairAction,
)


_FACTORY_WEB_UNIT = "mini_pupper_web.service"
_CANDIDATE_UNITS = (
    "unitport-minipupper.service",   # DEMO convention — what deploy.py installs
    "mangdang_unitport.service",     # Phase 3.5 legacy alias (kept for back-compat)
)


# ---------------------------------------------------------------------- repairs


def _stop_factory_web(ctx: DiagnosticContext) -> None:
    """Idempotent ``systemctl stop`` on the factory web service.

    ``stop`` on an already-inactive unit is a no-op (exit 0), so re-runs are
    safe. We do NOT ``disable`` — that would stop it from coming back on
    reboot, which the user might still want for unrelated reasons.
    """
    if ctx.ssh is None:
        raise RuntimeError("ssh unavailable")
    res = ctx.ssh.exec_sudo(f"systemctl stop {_FACTORY_WEB_UNIT}")
    if not res.ok:
        raise RuntimeError(
            f"systemctl stop {_FACTORY_WEB_UNIT} exit={res.exit_code}: "
            f"{res.stderr.strip() or res.stdout.strip()}"
        )


def _make_stop_factory_web_repair() -> RepairAction:
    return RepairAction(
        name="mangdang.stop_factory_web",
        describe=f"sudo systemctl stop {_FACTORY_WEB_UNIT}",
        run=_stop_factory_web,
        safe=True,
    )


_DROPIN_TEMPLATE = """[Unit]
Conflicts={web_unit}
"""


def _make_write_conflict_dropin_repair(unit: str) -> RepairAction:
    """Build a repair that adds Conflicts= to the detected unit's drop-in dir.

    The drop-in lives at ``/etc/systemd/system/<unit>.d/unitport-conflict.conf``
    and only declares the Conflicts= relationship — no ExecStop/Restart
    overrides — so it stays compatible with whatever the deployed unit file
    already does. After writing we ``daemon-reload`` (cheap) but do NOT
    restart the unit (the user may not want a reconnect right now;
    Conflicts= takes effect on the next start).
    """
    dropin_path = f"/etc/systemd/system/{unit}.d/unitport-conflict.conf"
    content = _DROPIN_TEMPLATE.format(web_unit=_FACTORY_WEB_UNIT)

    def _run(ctx: DiagnosticContext) -> None:
        if ctx.ssh is None:
            raise RuntimeError("ssh unavailable")
        mkd = ctx.ssh.exec_sudo(f"mkdir -p /etc/systemd/system/{unit}.d")
        if not mkd.ok:
            raise RuntimeError(
                f"mkdir drop-in dir exit={mkd.exit_code}: "
                f"{mkd.stderr.strip()}"
            )
        ctx.ssh.write_file(dropin_path, content, sudo=True, mode=0o644)
        rel = ctx.ssh.exec_sudo("systemctl daemon-reload")
        if not rel.ok:
            raise RuntimeError(
                f"daemon-reload exit={rel.exit_code}: {rel.stderr.strip()}"
            )

    return RepairAction(
        name=f"mangdang.write_conflict_dropin[{unit}]",
        describe=(
            f"write {dropin_path} with Conflicts={_FACTORY_WEB_UNIT}, "
            "then daemon-reload"
        ),
        run=_run,
        safe=True,
    )


# ---------------------------------------------------------------------- probe


class MangdangServiceConflictProbe(DiagnosticProbe):
    """Detect mini_pupper_web racing /cmd_vel and verify Conflicts= directive."""

    id = "mangdang_service_conflict"
    requires_ssh = True

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        if ctx.ssh is None:
            return [self.info("ssh_unavailable")]

        detected = self._detect_active_unit(ctx)
        findings: List[DiagnosticFinding] = []

        # (1) Is the factory web racing right now?
        web_state = self._unit_state(ctx, _FACTORY_WEB_UNIT)
        if web_state == "active":
            findings.append(self.error(
                f"{_FACTORY_WEB_UNIT} active alongside unitport bringup",
                detail=(
                    f"The Mini Pupper factory web service ({_FACTORY_WEB_UNIT}) "
                    f"is running on the robot. It publishes zero Twists on "
                    f"/cmd_vel as part of its idle loop, racing our teleop "
                    f"frames -> the robot ignores joystick input even though "
                    f"the bridge looks healthy.\n\n"
                    f"Detected unitport unit: {detected or '(none active)'}.\n"
                    f"Safe fix stops the factory web service "
                    f"(idempotent — re-runs are no-ops)."
                ),
                repair=_make_stop_factory_web_repair(),
            ))
        else:
            findings.append(self.ok(
                f"{_FACTORY_WEB_UNIT} not active "
                f"(state={web_state or 'unknown'})",
            ))

        # (2) Does the deployed unit declare Conflicts=mini_pupper_web.service?
        if detected is None:
            findings.append(self.info(
                "no unitport unit detected; skipping Conflicts= check",
                detail=(
                    "Tried: " + ", ".join(_CANDIDATE_UNITS) + ". "
                    "Either the unit is named differently on this robot or "
                    "the unitport stack is not deployed."
                ),
            ))
            return findings

        conflicts = self._unit_property(ctx, detected, "Conflicts")
        if conflicts is None:
            findings.append(self.warning(
                f"could not read Conflicts= for {detected}",
                detail="systemctl show returned no output",
            ))
            return findings
        # systemd Conflicts= is space-separated when multiple are listed.
        listed = set(conflicts.split())
        if _FACTORY_WEB_UNIT in listed:
            findings.append(self.ok(
                f"{detected} declares Conflicts={_FACTORY_WEB_UNIT}",
            ))
        else:
            findings.append(self.warning(
                f"{detected} missing Conflicts={_FACTORY_WEB_UNIT}",
                detail=(
                    f"Without the Conflicts= directive, systemd will let the "
                    f"factory web service come back up alongside the unitport "
                    f"unit on the next reboot. The race condition will "
                    f"silently re-emerge.\n\n"
                    f"Current Conflicts= value: {conflicts or '(empty)'}.\n"
                    f"Safe fix installs a drop-in at "
                    f"/etc/systemd/system/{detected}.d/unitport-conflict.conf"
                    f" that adds the directive without touching the rest of "
                    f"the unit definition."
                ),
                repair=_make_write_conflict_dropin_repair(detected),
            ))

        return findings

    # --- helpers ---------------------------------------------------------

    def _detect_active_unit(self, ctx: DiagnosticContext) -> Optional[str]:
        """Return whichever candidate unit is currently active, else None."""
        for unit in _CANDIDATE_UNITS:
            if self._unit_state(ctx, unit) == "active":
                return unit
        return None

    def _unit_state(self, ctx: DiagnosticContext, unit: str) -> str:
        """``systemctl is-active <unit>`` -> trimmed stdout (e.g. ``active``).

        Note: ``is-active`` returns exit 3 for inactive/dead, but stdout is
        still informative ("inactive" / "failed" / ...). We use stdout
        irrespective of exit code.
        """
        if ctx.ssh is None:
            return ""
        res = ctx.ssh.exec(f"systemctl is-active {unit}")
        return (res.stdout or "").strip()

    def _unit_property(
        self, ctx: DiagnosticContext, unit: str, prop: str,
    ) -> Optional[str]:
        """Read one ``systemctl show -p <prop>`` value; returns trimmed body."""
        if ctx.ssh is None:
            return None
        res = ctx.ssh.exec(f"systemctl show -p {prop} {unit}")
        line = (res.stdout or "").strip()
        if not line:
            return None
        # Format: "Conflicts=foo.service bar.service"
        return line.partition("=")[2].strip()


__all__ = ["MangdangServiceConflictProbe"]
