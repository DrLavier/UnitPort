# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Subprocess dispatcher for the Dump MJCF / Dump USD buttons.

The MJCF path runs in-process via mujoco (already a hard dep of the main
app), so we just call :func:`BodyIRMapper.discover_from_mjcf` directly.
The USD path needs Isaac Sim's pxr/Omniverse stack which we never import
into the main venv — so we spawn ``bootstrap/discover_usd_bodies.py``
inside the Isaac Lab venv as a subprocess, read the resulting JSON, and
hand it back to the caller.

Returned shape from both paths::

    {"bodies": {body_name: ir_role, ...}, "joints": {joint_name: ir_role, ...}}

IR-role assignment for USD-discovered bodies is intentionally **left to
the caller** in the USD path — discover_usd_bodies.py returns raw names
only because USD has no canonical IR role hint. The caller (Robot Asset
card or :meth:`RobotAssetService.set_discovered_bodies`) seeds joints
from the existing MJCF table when name patterns match, then surfaces the
unmapped ones in the UI for user assignment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_error, log_info, log_warning


_DISCOVERY_TIMEOUT_S = 180  # pxr path ~5s, kit fallback cold start ~60-120s


class DiscoveryError(RuntimeError):
    """Raised when MJCF/USD asset discovery fails."""


# ---------------------------------------------------------------------------
# MJCF — in-process via mujoco (already in main venv)
# ---------------------------------------------------------------------------


COSMETIC_TOKENS: frozenset = frozenset({
    # Suffix stems that strongly imply "cosmetic / passive accessory
    # link" rather than a real actuated joint or kinematic limb.
    # When the tokeniser would otherwise classify e.g. `FL_hip_protector`
    # as `hip_FL` (because "hip" matches), these tokens override the
    # guess to ``misc`` — the user can later refine via the IR-role
    # assignment dialog (pick the actual limb role, sensor_*, or
    # Out of Scope) but ``misc`` is a safe default that means
    # "physical link but not actuated".
    "protector", "cover", "shell", "guard", "shroud", "fairing",
    "housing", "skin", "decoration", "decor", "logo", "label",
    "black", "white",       # Go2 USD: base_black / base_white colour-swap shells
    "calflower",            # Go2 USD: FL_calflower / FL_calflower1 cosmetic calf sleeves
    "assembler", "fixed",   # NVIDIA USD wraps fixed-joint helpers as `AssemblerFixedJoint`
    "site",                 # mujoco "site" sensor markers
})


def _is_non_unique_role(role: str) -> bool:
    """Return True for IR roles where the registry legitimately holds
    multiple bodies pointing at the same role_id — ``sensor`` /
    ``sensor_*`` (one robot, many sensors) and ``misc`` (one robot,
    many decorative covers). Used by both the dump-time suggester and
    the boot-time dedupe pass to skip uniqueness guards."""
    if not role:
        return False
    return role == "misc" or role.startswith("sensor")


def _is_cosmetic_name(name: str) -> bool:
    """True iff a body/joint name contains any cosmetic-token stem as
    a substring of its lowercased form.

    Substring (not token-exact) match because USD names often append
    numeric suffixes — ``FL_calflower1`` and ``FL_calflower2`` both
    need to match the ``calflower`` cosmetic stem, but ``split()``
    would only see ``"calflower1"`` as a single token. Cosmetic stems
    are long and unique enough (``protector`` / ``shroud`` /
    ``calflower`` / ``assembler``) that false positives are unlikely.
    """
    if not name:
        return False
    low = str(name).lower()
    return any(t in low for t in COSMETIC_TOKENS)


def _suggest_ir_roles(
    names: List[str], existing_map: Dict[str, str],
    *,
    family: str = "generic",
) -> Dict[str, str]:
    """Assign IR roles to a list of joint or body names.

    Priority per name:
      1. ``existing_map[name]`` if present and non-empty (preferred —
         canonical knowledge from MJCF or a previous Dump). Always wins,
         even if it conflicts with a later auto-suggestion.
      2. Cosmetic-token veto: names containing tokens like ``protector``
         / ``cover`` / ``shroud`` / ``logo`` get an empty ir_role even
         when the tokeniser would otherwise classify them — they are
         accessory links the user must classify manually (typically as
         ``sensor_*`` or Out-of-Scope). Skips the veto when the tokeniser
         result is a ``sensor_*`` role (logos/labels can legitimately be
         sensor mounts).
      3. ``body_ir._suggest_role_id(name, family)`` — keyword tokeniser
         with family-aware multi-DOF recognition (biped: hip_pitch_L /
         ankle_roll_R / waist_yaw / thumb_2_L ...; quadruped: hip_FL /
         thigh_FL / calf_FL / foot_FL ...).
      4. **Uniqueness guard** — if step 3 would assign the same
         non-sensor ir_role to a second body, the second+ entries get
         empty ir_role instead. Sensor roles (``sensor`` / ``sensor_*``)
         are exempt because a robot legitimately has many sensors all
         mapping to ``sensor`` if the type token is absent. The first
         body to "win" a role keeps it; conflicts fall back to user
         resolution via the IR-role assignment dialog.
      5. Empty string — surfaces in the UI BodyMappingTable for the
         user to manually assign before training.
    """
    import re as _re

    from application.training.body_ir import _suggest_role_id

    # Track role_ids already used by step 1 (seed) or step 3 (tokeniser)
    # so we never auto-assign the same actuated-joint role to two
    # different bodies — the registry only has one slot per role_id and
    # BodyIRMapper's last-write-wins eviction would otherwise spam the
    # unmapped list with bodies that "lost" the race.
    used_roles: set = set()

    out: Dict[str, str] = {}
    # First pass — seeded entries (always win, register them as used so
    # auto-suggestions in the second pass can't re-use the same role).
    for nm in names:
        seeded = (existing_map.get(nm) or "").strip()
        if seeded:
            out[nm] = seeded
            used_roles.add(seeded)

    # Second pass — auto-suggest only for not-yet-resolved names.
    for nm in names:
        if nm in out:
            continue
        try:
            inferred = _suggest_role_id(nm, family)
        except Exception:
            inferred = ""
        if inferred:
            # Cosmetic-token override: a name like FL_hip_protector or
            # FL_calflower1 tokenises to a limb role (hip / calf
            # matches) but the cosmetic substring signals "decorative
            # shell". Re-route to ``misc``. Sensor_* roles are exempt
            # (a cosmetic-named link CAN legitimately be a sensor).
            if not inferred.startswith("sensor") and _is_cosmetic_name(nm):
                inferred = "misc"
        # Uniqueness guard — exempts sensor_* and misc roles since
        # both legitimately repeat across many bodies on one robot.
        if inferred and not _is_non_unique_role(inferred):
            if inferred in used_roles:
                inferred = ""
        out[nm] = inferred or ""
        if inferred:
            used_roles.add(inferred)
    return out


def discover_mjcf(
    mjcf_path: Path,
    joints_role_map: Dict[str, str],
    *,
    family: str = "generic",
) -> Dict[str, Any]:
    """Parse an MJCF file in-process and return body/joint discovery payload.

    Args:
        mjcf_path: path to the MJCF scene.xml on disk
        joints_role_map: existing ``{joint_name: ir_role}`` to seed the
            discovery — usually the robot's existing MJCF block when
            re-running Dump, or empty when bootstrapping a previously
            undeclared robot (A1/G1/H1 before any Dump). Empty seed
            triggers keyword-tokeniser fallback.

    Returns:
        ``{"bodies": {name: ir_role}, "joints": {name: ir_role}}``.
        Joints + bodies are IR-role-stamped via (1) the seed map then
        (2) keyword tokeniser. Unresolvable entries get an empty role
        which surfaces in the canvas BodyMappingTable for the user.
    """
    from application.training.body_ir import BodyIRMapper

    if not mjcf_path.exists():
        raise DiscoveryError(f"MJCF not found: {mjcf_path}")

    # Joints: extract from MJCF, then auto-suggest IR roles for any
    # joint missing from the seed map.
    import mujoco  # type: ignore
    # MuJoCo's tinyxml2 C++ parser opens files via fopen(), which on
    # Windows interprets bytes through the system ANSI codepage. UTF-8
    # paths containing CJK / accented characters fail there even though
    # pathlib.exists() above passed. Route through the Windows short-path
    # API so the load works for users with non-ASCII project paths.
    from application.physics.mujoco_path_compat import safe_mjcf_path
    m = mujoco.MjModel.from_xml_path(safe_mjcf_path(mjcf_path))
    raw_joints: List[str] = []
    for i in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jn:
            raw_joints.append(jn)
    joints_with_ir = _suggest_ir_roles(raw_joints, joints_role_map, family=family)

    # Bodies: BodyIRMapper.discover_from_mjcf uses joint→body via
    # jnt_bodyid + keyword fallback for feet/base. Feed it the
    # enriched joint map so all the actuated bodies pick up IR roles
    # (even when the input seed was empty).
    try:
        bodies_with_ir = BodyIRMapper.discover_from_mjcf(mjcf_path, joints_with_ir)
    except Exception as exc:
        raise DiscoveryError(f"MJCF body discovery failed: {exc}") from exc

    # Also auto-suggest for any body the joint+keyword path missed.
    import mujoco as _mu  # type: ignore
    all_bodies: List[str] = []
    for i in range(1, m.nbody):
        bn = _mu.mj_id2name(m, _mu.mjtObj.mjOBJ_BODY, i)
        if bn and bn not in all_bodies:
            all_bodies.append(bn)
    bodies_with_ir = _suggest_ir_roles(all_bodies, bodies_with_ir, family=family)

    return {"bodies": bodies_with_ir, "joints": joints_with_ir}


# ---------------------------------------------------------------------------
# USD — subprocess into Isaac Lab venv
# ---------------------------------------------------------------------------


def _resolve_isaac_python() -> Optional[str]:
    """Locate the Isaac Lab Python interpreter via the registry."""
    try:
        from registers.backends import _detect_isaac_lab, _find_isaac_python
        det = _detect_isaac_lab()
        root_str = (det or {}).get("path", "")
        if not root_str:
            return None
        py = _find_isaac_python(Path(root_str))
        return py
    except Exception as exc:
        log_warning(f"[discovery] cannot resolve isaac python: {exc}")
        return None


def _discover_usd_script() -> Path:
    """Resolve ``bootstrap/discover_usd_bodies.py`` on disk."""
    project_root = Paths.PROJECT_ROOT
    return project_root / "bootstrap" / "discover_usd_bodies.py"


def discover_usd(
    usd_ref: str,
    joints_role_map_mjcf: Dict[str, str],
    *,
    family: str = "generic",
) -> Dict[str, Any]:
    """Parse a USD asset by spawning Isaac Lab's Python.

    Args:
        usd_ref: local path or Nucleus URL (``nucleus:Robots/...``).
        joints_role_map_mjcf: ``{joint_name: ir_role}`` from the MJCF
            block of the same robot. When USD joint names match MJCF
            names exactly (the common case for menagerie-derived USDs)
            we use this to pre-fill IR roles; mismatches surface as
            empty roles for the UI to resolve.

    Returns:
        ``{"bodies": {name: ir_role_or_empty}, "joints": {name: ir_role_or_empty}}``.

    Raises:
        :class:`DiscoveryError` on subprocess failure / unparseable output.
    """
    isaac_py = _resolve_isaac_python()
    if isaac_py is None:
        raise DiscoveryError(
            "Isaac Lab Python interpreter not found. Register Isaac Lab "
            "via the Engine Settings dialog (Settings → Engines → Isaac Lab) "
            "before using Dump USD."
        )

    script = _discover_usd_script()
    if not script.exists():
        raise DiscoveryError(f"discover_usd_bodies.py not found at {script}")

    out_fd, out_path = tempfile.mkstemp(suffix=".json", prefix="unitport_usd_dump_")
    os.close(out_fd)

    cmd = [isaac_py, str(script), str(usd_ref), out_path]
    log_info(f"[discovery] spawning USD dump: {' '.join(cmd[:3])} ...")

    # Auto-accept Omniverse Kit EULA in the child process; without this
    # the kit fallback inside discover_usd_bodies.py blocks forever on
    # an interactive Yes/No prompt (subprocess has no TTY → EOF → kit
    # bootstrap fails with "Unable to bootstrap inner kit kernel").
    # Mirrors what il_train_launcher.py does for the training subprocess.
    #
    # Force UTF-8 for both the child process stdout (PYTHONIOENCODING /
    # PYTHONUTF8) and our pipe decoder (encoding="utf-8", errors="replace").
    # Without this, Isaac Sim's print statements emit UTF-8 bytes that
    # our Popen reader tries to decode as GBK on a Windows zh_CN host
    # and crashes with ``UnicodeDecodeError: 'gbk' codec can't decode
    # byte 0x80…`` — which is what happened when a user with a Chinese
    # install path tried to refresh joint mappings against the new
    # ``<install_dir>/.venv`` Isaac interpreter.
    child_env = os.environ.copy()
    child_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env.setdefault("PYTHONUTF8", "1")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DISCOVERY_TIMEOUT_S,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise DiscoveryError(
            f"USD discovery timed out after {_DISCOVERY_TIMEOUT_S}s "
            f"(ref={usd_ref!r}). If this is a remote Nucleus URL the "
            f"first fetch can be slow — retry once."
        )

    # Pre-build the stderr tail once — every error path below quotes it.
    # The kit subprocess produces a LOT of stdout/stderr (system info
    # banner ~30 lines, extension load progress ~100 lines), so a naive
    # "last 10 lines" mostly catches the banner footer and hides the real
    # error. We grep for high-signal lines first (our own script's
    # ``[discover_usd_bodies]`` markers, ``[Error]`` / ``[Critical]`` Kit
    # log levels, ``Failed to load plugin`` from the ANSI-codepage DLL
    # load failure pattern, ``Traceback`` / ``Exception`` from Python
    # crashes) and only fall back to plain last-N lines if no signal lines
    # were found.
    _SIGNAL_PATTERNS: tuple = (
        "[discover_usd_bodies]",   # our own script's diagnostic prefix
        "[Error]",                 # carb log level
        "[Critical]",              # carb log level
        "[Fatal]",                 # carb log level
        "Failed to load plugin",   # the LoadLibraryA non-ASCII DLL pattern
        "Coding Error",            # USD/Plug runtime errors
        "Traceback",               # Python exception block start
        "Exception:",              # bare exception repr
        "Error:",                  # generic error tail (carb / kit / pxr)
        "RuntimeError",            # our own script's RuntimeError
        "ModuleNotFoundError",     # pxr unavailable in isaac venv
    )

    def _diag_tail() -> str:
        stderr = (result.stderr or "").splitlines()
        stdout = (result.stdout or "").splitlines()
        all_lines: List[str] = stderr + stdout
        signal: List[str] = []
        for line in all_lines:
            if any(p in line for p in _SIGNAL_PATTERNS):
                stripped = line.rstrip()
                if stripped and (not signal or signal[-1] != stripped):
                    signal.append(stripped)
        if signal:
            # Keep the last 15 signal lines — enough to show a full
            # traceback or several Kit DLL-load failures, not so many
            # that the error panel becomes unreadable.
            return "\n".join(signal[-15:])
        # No signal lines surfaced; fall back to plain tail so the user
        # still sees *something*. This typically means the subprocess
        # died before producing any structured diagnostic.
        non_empty: List[str] = []
        for line in all_lines:
            stripped = line.rstrip()
            if stripped and (not non_empty or non_empty[-1] != stripped):
                non_empty.append(stripped)
        return "\n".join(non_empty[-10:])

    # Always read the output file first — the script writes a structured
    # ``_error`` sentinel for every known failure mode (non-ASCII Isaac
    # Lab path, pxr unavailable, kit boot crash, Stage.Open returned
    # None, traversal returned nothing, etc.) regardless of the exit
    # code, so the sentinel is a more reliable signal than ``returncode``
    # alone. The exit code is used as a tie-breaker: if the file is
    # unreadable / empty / not JSON AND the script exited non-zero, we
    # report the script as crashed; if both the file is bad AND exit is
    # zero, the script died so abruptly it could not even write the
    # sentinel (Kit's ``os._exit(0)`` mid-traversal).
    try:
        raw_text = Path(out_path).read_text(encoding="utf-8")
    except OSError as exc:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise DiscoveryError(
            f"USD dump output unreadable at {out_path!r}: {exc}\n"
            f"Subprocess exit={result.returncode}.\n"
            f"stderr tail:\n{_diag_tail()}"
        ) from exc

    try:
        payload = json.loads(raw_text) if raw_text.strip() else {}
    except Exception as exc:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise DiscoveryError(
            f"USD dump output unparseable "
            f"({type(exc).__name__}: {exc}). Raw output was "
            f"{len(raw_text)} byte(s). Subprocess exit={result.returncode}.\n"
            f"stderr tail:\n{_diag_tail()}"
        ) from exc
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    if isinstance(payload, dict) and payload.get("_error"):
        # Structured failure from the script — the sentinel field
        # carries the script's own diagnosis (including the non-ASCII
        # Isaac Lab path case, which short-circuits with a clear "move
        # Isaac Lab to ASCII path" directive). Surface that AND the
        # filtered stderr tail so the user has both the actionable fix
        # and the underlying Kit error trail.
        raise DiscoveryError(
            f"USD discovery did not complete: {payload['_error']}\n"
            f"Subprocess exit={result.returncode}.\n"
            f"stderr tail:\n{_diag_tail()}"
        )

    if result.returncode != 0:
        # Non-zero exit AND no sentinel — the script crashed before it
        # could write any structured information. All we have is stderr.
        raise DiscoveryError(
            f"USD discovery subprocess failed (exit {result.returncode}).\n"
            f"stderr tail:\n{_diag_tail()}"
        )

    if not raw_text.strip():
        # Exit 0 but empty file — Kit's os._exit(0) fired before any
        # write, including our up-front sentinel. Should be rare now
        # that the sentinel runs as the first thing in main().
        raise DiscoveryError(
            f"USD discovery produced no output (subprocess exited 0 but "
            f"the result file is empty).\n"
            f"stderr tail:\n{_diag_tail()}"
        )

    raw_bodies: List[str] = list(payload.get("bodies", []) or [])
    raw_joints: List[str] = list(payload.get("joints", []) or [])

    # Joints: prefer MJCF seed (same name) then keyword tokeniser, so
    # robots without an MJCF baseline (A1 / G1 / H1: data gap, no MJCF
    # joint table declared) still get IR roles auto-assigned from
    # Unitree-style joint naming. Unrecognised names surface as empty
    # roles for the UI to resolve.
    joints_out = _suggest_ir_roles(raw_joints, joints_role_map_mjcf, family=family)

    # Bodies: tokeniser only (no MJCF body seed in this code path).
    # The Robot Asset card → canvas BodyMappingTable will let the user
    # assign IR roles to any body the tokeniser couldn't classify.
    bodies_out = _suggest_ir_roles(raw_bodies, {}, family=family)

    return {"bodies": bodies_out, "joints": joints_out}


# ---------------------------------------------------------------------------
# Unified entry — picks the right path per format
# ---------------------------------------------------------------------------


def dump_format(
    asset: Any,
    fmt: str,
    joints_role_map_mjcf: Dict[str, str],
    *,
    family: str = "generic",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Run the appropriate discovery path for ``fmt`` and return
    ``(joints, bodies)`` flat dicts the caller can hand to
    :meth:`RobotAssetService.set_discovered_bodies`.

    ``family`` is the robot morphology family (``biped`` / ``humanoid`` /
    ``quadruped`` / ...). Threaded into :func:`_suggest_ir_roles` so the
    tokeniser can pick up multi-DOF axis-split joints (G1's
    ``left_hip_pitch_joint`` → ``hip_pitch_L``) and finger roles (G1
    with_hands ``left_thumb_2_joint`` → ``thumb_2_L``). For non-biped
    families the family-specific rules are a no-op.
    """
    fmt_u = str(fmt).upper()
    if fmt_u == "MJCF":
        mjcf_path = getattr(asset, "mjcf_path", None)
        if mjcf_path is None:
            raise DiscoveryError("No MJCF path declared for this robot")
        payload = discover_mjcf(Path(mjcf_path), joints_role_map_mjcf, family=family)
    elif fmt_u == "USD":
        # Prefer local path; fall back to Nucleus URL
        usd_path = getattr(asset, "usd_path", None)
        usd_url = getattr(asset, "usd_url", None)
        ref = str(usd_path) if usd_path else (usd_url or "")
        if not ref:
            raise DiscoveryError("No USD path or Nucleus URL declared for this robot")
        payload = discover_usd(ref, joints_role_map_mjcf, family=family)
    elif fmt_u == "URDF":
        raise DiscoveryError("URDF discovery not yet implemented")
    else:
        raise DiscoveryError(f"Unknown format: {fmt!r}")

    return dict(payload.get("joints", {})), dict(payload.get("bodies", {}))


__all__ = [
    "DiscoveryError",
    "discover_mjcf",
    "discover_usd",
    "dump_format",
]
