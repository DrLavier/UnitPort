"""Verify every package listed in requirements.txt is importable in this venv.

Two consumers share this module:

1. **Command line** (start.bat / start.sh / manual diagnostic): exits 0 if all
   required packages import, 1 if anything is missing or broken.

2. **In-process** (``application.tools.startup_tasks.ProvisioningTask``):
   imports ``iter_required_packages`` and ``probe_imports`` directly to gate
   the LoadingScreen-first startup pipeline. Both call sites share the same
   parser and probe so requirements.txt has exactly one source of truth.

Run manually:
    python bootstrap/_check_requirements.py             # uses ../requirements.txt
    python bootstrap/_check_requirements.py PATH        # uses PATH
"""

from __future__ import annotations

import importlib
import re
import sys
import traceback
from pathlib import Path
from typing import Iterator, NamedTuple


class ImportFailure(NamedTuple):
    """One package's failed import probe.

    Carries enough context for the caller to render an actionable error:
    distribution name (as in requirements.txt), import name (the bare
    ``import_module`` argument), the exception type/message, and the
    head of the traceback. The traceback head is typically what tells a
    DLL load failure apart from a plain ModuleNotFoundError, which in
    turn tells torch-CUDA-mismatch apart from torch-not-installed.
    """

    pkg: str
    import_name: str
    exc_type: str
    exc_msg: str
    traceback_head: str

# Distribution name (as it appears in requirements.txt) -> import name.
# Only entries that differ from the lowercased / dash-to-underscore default
# need to be listed.
NAME_MAP: dict[str, str] = {
    "PyYAML": "yaml",
    "stable-baselines3": "stable_baselines3",
    "rlottie-python": "rlottie_python",
    "mcap-ros2-support": "mcap_ros2",
    "huggingface_hub": "huggingface_hub",
    "tomli_w": "tomli_w",
}

# Distributions that ship CLI/test tooling only and aren't required for the
# UnitPort process to launch. Skipped to avoid noisy "missing" warnings on
# headless / minimal installs.
SKIP_PACKAGES: set[str] = {
    "pytest",
    "tensorboard",
}


_NAME_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.\-]*)")


def _marker_applies(marker_text: str) -> bool:
    """Return True if the requirements.txt environment marker evaluates True
    on the current interpreter.

    Uses ``packaging.markers.Marker`` when available (pip ships it). Falls
    back to True (i.e. the entry is treated as required) so we never silently
    skip a real package on minimal venvs.
    """
    marker_text = marker_text.strip()
    if not marker_text:
        return True
    try:
        from packaging.markers import Marker
    except Exception:
        return True
    try:
        return bool(Marker(marker_text).evaluate())
    except Exception:
        return True


def _extract_package_name(line: str) -> str | None:
    """Return the bare distribution name from one requirements.txt line, or
    None when the line is blank, a comment, or its environment marker
    evaluates to False on this interpreter (e.g. ``tomli; python_version <
    "3.11"`` skips on Python 3.11+).
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # Split off and evaluate environment markers ("; python_version < '3.11'").
    if ";" in s:
        s, marker = s.split(";", 1)
        if not _marker_applies(marker):
            return None
        s = s.strip()
    # Strip extras suffix ("foo[bar]>=1.0")
    if "[" in s:
        s = s.split("[", 1)[0]
    m = _NAME_RE.match(s)
    return m.group(1) if m else None


def _to_import_name(pkg: str) -> str:
    return NAME_MAP.get(pkg, pkg.replace("-", "_"))


def iter_required_packages(req_path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(distribution_name, import_name)`` for every required package.

    Packages in :data:`SKIP_PACKAGES` and duplicates are filtered out. The
    iterator is single-pass (caller should ``list()`` it if multiple passes
    are needed).
    """
    seen: set[str] = set()
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        pkg = _extract_package_name(raw)
        if not pkg or pkg in SKIP_PACKAGES or pkg in seen:
            continue
        seen.add(pkg)
        yield pkg, _to_import_name(pkg)


def probe_imports(
    pairs: list[tuple[str, str]] | Iterator[tuple[str, str]],
) -> list[ImportFailure]:
    """Try importing each ``(dist, import)`` pair; return failures only.

    Returns a list of :class:`ImportFailure` records (empty if every
    package imports cleanly). The traceback head is the last 3 frames of
    the exception's traceback joined with the type+msg line, suitable
    for one-shot error reporting in a UI log without a full stack dump.
    """
    missing: list[ImportFailure] = []
    for pkg, import_name in pairs:
        try:
            importlib.import_module(import_name)
        except BaseException as exc:  # noqa: BLE001
            # BaseException catches ImportError, ModuleNotFoundError, plus
            # rarer cases like SyntaxError / OSError from a half-installed
            # wheel (e.g. CUDA wheel whose _C.pyd can't load its DLL deps).
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            head = "".join(tb_lines[-4:]).strip()
            missing.append(
                ImportFailure(
                    pkg=pkg,
                    import_name=import_name,
                    exc_type=type(exc).__name__,
                    exc_msg=str(exc),
                    traceback_head=head,
                )
            )
    return missing


def main() -> int:
    if len(sys.argv) >= 2:
        req_path = Path(sys.argv[1])
    else:
        req_path = Path(__file__).resolve().parent.parent / "requirements.txt"

    if not req_path.exists():
        print(f"[check] requirements.txt not found: {req_path}", file=sys.stderr)
        return 2

    pairs = list(iter_required_packages(req_path))
    missing = probe_imports(pairs)

    if missing:
        print(f"[check] {len(missing)} required package(s) missing or broken:")
        for f in missing:
            print(f"  - {f.pkg} (import {f.import_name})  ->  {f.exc_type}: {f.exc_msg}")
        return 1

    print(f"[check] all {len(pairs)} required packages importable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
