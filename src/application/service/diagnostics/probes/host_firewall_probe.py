# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""HostFirewallProbe — verify Windows Defender Firewall does not block DDS.

CycloneDDS uses UDP for SPDP / SEDP discovery in the 7400-7500 range
(domain-tag offsets). When Windows Defender blocks inbound UDP on those
ports the host participant never sees the robot's announcements; the
bridge looks alive but stays dark.

This probe enumerates active inbound block rules covering UDP 7400-7500
and offers a safe repair: ``netsh advfirewall firewall add rule`` to
allow that port range explicitly.

macOS and Linux skip this probe with an INFO finding; their default
firewalls don't block the DDS discovery range out of the box.
"""

from __future__ import annotations

import platform
import subprocess
from typing import List

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import (
    DiagnosticFinding,
    RepairAction,
)


_DDS_RULE_NAME = "UnitPort DDS Discovery"
_DDS_PORT_RANGE = "7400-7500"


class _LocalisedNetshError(RuntimeError):
    """Raised when netsh output uses non-English (locale-translated) keys."""


def _run_netsh(cmd: list, *, timeout: int):
    """Run netsh with locale-tolerant decoding.

    netsh on Chinese Windows emits GBK-encoded text; Python's default
    text=True picks up the system codepage and crashes with
    UnicodeDecodeError on non-ASCII rule names. Read the streams as bytes
    and decode with errors='replace' so a localised system never aborts
    the probe.
    """
    proc = subprocess.run(  # noqa: S603 — netsh is well-known
        cmd, capture_output=True, timeout=timeout,
    )
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace") \
        if not _looks_gbk(proc.stdout) else \
        (proc.stdout or b"").decode("gbk", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace") \
        if not _looks_gbk(proc.stderr) else \
        (proc.stderr or b"").decode("gbk", errors="replace")

    class _Result:
        pass
    r = _Result()
    r.returncode = proc.returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _looks_gbk(data: bytes) -> bool:
    """Heuristic: payload has high bytes and is not valid UTF-8."""
    if not data or not any(b > 0x7F for b in data[:4096]):
        return False
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def _is_windows_admin() -> bool:
    """True when the current process has admin token on Windows; else False."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _run_netsh_elevated(args: list, timeout: int = 30) -> int:
    """Run ``netsh <args>`` under UAC elevation; return its exit code.

    Used when the running process lacks the admin token needed to modify
    Windows Defender Firewall rules. The standard subprocess path returns
    exit=1 with "operation requires elevation" stderr; this fallback uses
    ShellExecuteExW + verb='runas' to trigger the UAC consent prompt, then
    waits for the elevated child to finish.

    Exit code is the netsh return: 0 = ok, non-zero = failure. The UAC
    cancellation case (user clicked "No") raises RuntimeError so the
    repair surfaces a clear message instead of silently no-op'ing.
    """
    import ctypes
    from ctypes import wintypes

    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    SEE_MASK_NO_CONSOLE = 0x00008000
    SW_HIDE = 0
    INFINITE = 0xFFFFFFFF
    ERROR_CANCELLED = 1223  # user clicked "No" on UAC prompt

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    # netsh argv list -> single param string with quoting for the rule name.
    parts = []
    for arg in args:
        if " " in arg or "=" in arg and " " in arg.partition("=")[2]:
            k, eq, v = arg.partition("=")
            if eq:
                parts.append(f'{k}="{v}"')
                continue
        parts.append(arg)
    params = " ".join(parts)

    sei = SHELLEXECUTEINFOW()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NO_CONSOLE
    sei.lpVerb = "runas"
    sei.lpFile = "netsh"
    sei.lpParameters = params
    sei.nShow = SW_HIDE

    shell32 = ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL

    if not shell32.ShellExecuteExW(ctypes.byref(sei)):
        err = ctypes.GetLastError()
        if err == ERROR_CANCELLED:
            raise RuntimeError(
                "user declined UAC elevation prompt — firewall rule not added"
            )
        raise RuntimeError(f"ShellExecuteExW failed (GetLastError={err})")

    if not sei.hProcess:
        raise RuntimeError("ShellExecuteExW returned no process handle")

    try:
        wait_ms = int(max(1.0, float(timeout)) * 1000)
        kernel32.WaitForSingleObject(sei.hProcess, wait_ms)
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(sei.hProcess, ctypes.byref(code))
        return int(code.value)
    finally:
        kernel32.CloseHandle(sei.hProcess)


def _add_dds_allow_rule(_: DiagnosticContext) -> None:
    """Add an inbound allow rule for UDP 7400-7500 via netsh.

    Idempotent: ``netsh ... add rule`` simply appends another matching rule
    if one already exists; the duplicate is harmless. We don't bother
    deleting first so the action stays safe to re-run.

    Windows Defender Firewall rule modification requires admin privileges.
    When the running process is not elevated we re-dispatch via
    :func:`_run_netsh_elevated` so the user sees the standard UAC consent
    prompt; declining the prompt fails the repair cleanly.
    """
    args = [
        "advfirewall", "firewall", "add", "rule",
        f"name={_DDS_RULE_NAME}",
        "dir=in",
        "action=allow",
        "protocol=UDP",
        f"localport={_DDS_PORT_RANGE}",
    ]

    if _is_windows_admin():
        proc = _run_netsh(["netsh"] + args, timeout=15)
        if proc.returncode != 0:
            raise RuntimeError(
                f"netsh add rule failed (exit={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return

    # Non-elevated path — UAC consent prompts the user.
    exit_code = _run_netsh_elevated(args, timeout=30)
    if exit_code != 0:
        raise RuntimeError(
            f"elevated netsh add rule failed (exit={exit_code}); "
            "check Windows Defender Firewall console for details"
        )


class HostFirewallProbe(DiagnosticProbe):
    id = "host_firewall"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        if platform.system() != "Windows":
            return [self.info(
                f"firewall probe skipped on {platform.system()}",
            )]

        # Check for an existing block rule against UDP 7400-7500.
        try:
            blocking = self._enumerate_blocks()
        except _LocalisedNetshError as exc:
            # Localised Windows (CN/JP/...) -- netsh returns translated
            # keys we can't parse. Don't pretend we know the firewall
            # state; offer the safe allow rule for the user to opt into.
            return [self.info(
                "firewall enumeration not available on this locale",
                detail=str(exc),
                repair=RepairAction(
                    name="firewall.add_dds_allow_rule",
                    describe=(
                        f'netsh advfirewall firewall add rule '
                        f'name="{_DDS_RULE_NAME}" dir=in action=allow '
                        f'protocol=UDP localport={_DDS_PORT_RANGE}'
                    ),
                    run=_add_dds_allow_rule,
                    safe=True,
                ),
            )]
        except Exception as exc:  # noqa: BLE001
            return [self.warning(
                "firewall enumeration failed",
                detail=str(exc),
            )]

        repair = RepairAction(
            name="firewall.add_dds_allow_rule",
            describe=(
                f'netsh advfirewall firewall add rule name="{_DDS_RULE_NAME}" '
                f"dir=in action=allow protocol=UDP localport={_DDS_PORT_RANGE}"
            ),
            run=_add_dds_allow_rule,
            safe=True,
        )

        if blocking:
            return [self.warning(
                "firewall may block DDS discovery",
                detail=(
                    "Found inbound rule(s) potentially affecting UDP "
                    f"{_DDS_PORT_RANGE}: " + "; ".join(blocking)
                ),
                repair=repair,
            )]

        # No active block rule found. We still surface the repair as an
        # OK finding so the user can opt-in to add the explicit allow rule
        # — Defender's defaults vary across SKUs and a missing block rule
        # doesn't prove DDS will pass. But severity stays OK so we don't
        # spam the dialog.
        return [self.ok("no firewall block rule detected for DDS range")]

    def _enumerate_blocks(self) -> List[str]:
        proc = _run_netsh(
            [
                "netsh", "advfirewall", "firewall", "show", "rule",
                "name=all", "dir=in",
            ],
            timeout=15,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"netsh show rule exit={proc.returncode}: "
                f"{proc.stderr.strip()}"
            )
        out = proc.stdout
        # Detect localised output: if none of the English keys ever appear,
        # we cannot parse this safely. Surface as a typed error so the
        # caller offers an opt-in repair instead of pretending all is well.
        if not any(
            tok in out for tok in ("Action:", "Protocol:", "LocalPort:")
        ):
            raise _LocalisedNetshError(
                "netsh emitted localised keys; install the safe DDS allow "
                "rule manually if you suspect blocking"
            )
        blocks: List[str] = []
        # Rules are paragraph-separated. Look for blocks where Action=Block,
        # Protocol mentions UDP, and LocalPort intersects 7400-7500.
        for chunk in out.split("\n\n"):
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            if not lines:
                continue
            d = {}
            for ln in lines:
                if ":" not in ln:
                    continue
                k, _, v = ln.partition(":")
                d[k.strip()] = v.strip()
            action = d.get("Action", "").lower()
            protocol = d.get("Protocol", "").lower()
            local_port = d.get("LocalPort", "").lower()
            if action != "block":
                continue
            if "udp" not in protocol and protocol not in {"any", ""}:
                continue
            if not _ports_overlap(local_port):
                continue
            name = d.get("Rule Name", "?")
            blocks.append(f"{name} ({local_port})")
        return blocks


def _ports_overlap(local_port: str) -> bool:
    """Return True when ``local_port`` (netsh format) intersects 7400-7500."""
    if not local_port or local_port in {"any", "*"}:
        return True
    target_lo, target_hi = 7400, 7500
    for tok in local_port.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            try:
                lo, hi = (int(x) for x in tok.split("-", 1))
            except ValueError:
                continue
        else:
            try:
                lo = hi = int(tok)
            except ValueError:
                continue
        if not (hi < target_lo or lo > target_hi):
            return True
    return False


__all__ = ["HostFirewallProbe"]
