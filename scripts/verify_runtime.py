#!/usr/bin/env python3
"""
UnitPort - scripts/verify_runtime.py
Full runtime verification pass (Phase 2).

Checks that the packaged runtime payload is correctly assembled and usable
before distribution or after install.bat completes.

Run with:
  runtime/python/python.exe scripts/verify_runtime.py        (full check)
  runtime/python/python.exe scripts/verify_runtime.py --json (machine-readable)

Exit codes:
  0 = all critical checks passed (warnings allowed)
  1 = one or more critical checks failed
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
RESULTS: list[dict] = []


# ── Check decorator ────────────────────────────────────────────────────────

def check(name: str, critical: bool = True, category: str = "general"):
    """Register and run a named verification check."""
    def decorator(fn):
        def wrapper():
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"exception: {exc}"
            if ok:
                status = "PASS"
            elif critical:
                status = "FAIL"
            else:
                status = "WARN"
            RESULTS.append({
                "check": name,
                "status": status,
                "detail": detail,
                "critical": critical,
                "category": category,
            })
            label = f"[{status:<4}]"
            print(f"  {label} {name}")
            if detail:
                print(f"         {detail}")
            return ok
        return wrapper
    return decorator


# ── Category: Python runtime ───────────────────────────────────────────────

@check("runtime/python/python.exe present", critical=True, category="python")
def check_python_exe():
    p = RUNTIME_DIR / "python" / "python.exe"
    return p.exists(), str(p)


@check("Python version is 3.10 or 3.11", critical=True, category="python")
def check_python_version():
    v = sys.version_info
    ok = v.major == 3 and v.minor in (10, 11)
    return ok, f"{v.major}.{v.minor}.{v.micro} — {'OK' if ok else 'UNSUPPORTED version'}"


@check("pip is available in runtime Python", critical=True, category="python")
def check_pip_available():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        ok = result.returncode == 0
        return ok, result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return False, str(exc)


@check("site-packages enabled (import site)", critical=True, category="python")
def check_site_packages():
    try:
        import site
        paths = site.getsitepackages()
        return bool(paths), f"{len(paths)} site-packages path(s)"
    except Exception as exc:
        return False, str(exc)


# ── Category: Core imports ─────────────────────────────────────────────────

@check("PySide6 importable", critical=True, category="imports")
def check_pyside6():
    try:
        import PySide6
        return True, f"version {PySide6.__version__}"
    except ImportError as exc:
        return False, str(exc)


@check("numpy importable", critical=True, category="imports")
def check_numpy():
    try:
        import numpy
        return True, f"version {numpy.__version__}"
    except ImportError as exc:
        return False, str(exc)


@check("MuJoCo importable", critical=False, category="imports")
def check_mujoco():
    try:
        import mujoco
        return True, f"version {mujoco.__version__}"
    except ImportError as exc:
        return False, str(exc)


@check("cyclonedds Python package importable", critical=False, category="imports")
def check_cyclonedds_python():
    try:
        import cyclonedds
        ver = getattr(cyclonedds, "__version__", "unknown")
        return True, f"version {ver}"
    except ImportError as exc:
        return False, str(exc)


@check("unitree_sdk2py importable", critical=False, category="imports")
def check_unitree_sdk():
    try:
        import unitree_sdk2py  # noqa: F401
        return True, "ok"
    except ImportError as exc:
        return False, str(exc)


# ── Category: CycloneDDS native ───────────────────────────────────────────

@check("runtime/cyclonedds/ directory present", critical=False, category="cyclonedds")
def check_cyclonedds_dir():
    p = RUNTIME_DIR / "cyclonedds"
    has_bin = (p / "bin").is_dir()
    if p.is_dir() and has_bin:
        return True, str(p)
    if p.is_dir() and not has_bin:
        return False, f"directory exists but bin/ is missing: {p}"
    return False, f"not found: {p}"


@check("CYCLONEDDS_HOME points to project-local path", critical=False, category="cyclonedds")
def check_cyclonedds_home():
    home = os.environ.get("CYCLONEDDS_HOME", "")
    if not home:
        return False, "CYCLONEDDS_HOME not set — start via start.bat or run configure_cyclonedds_env()"
    expected = RUNTIME_DIR / "cyclonedds"
    try:
        ok = Path(home).resolve() == expected.resolve()
    except Exception:
        ok = False
    return ok, f"CYCLONEDDS_HOME={home}"


@check("CycloneDDS DLL discoverable in PATH", critical=False, category="cyclonedds")
def check_cyclonedds_dll():
    cdds_bin = RUNTIME_DIR / "cyclonedds" / "bin"
    if not cdds_bin.is_dir():
        return False, "runtime/cyclonedds/bin/ not present"

    candidates = list(cdds_bin.glob("ddsc*.dll")) + list(cdds_bin.glob("CycloneDDS*.dll"))
    if not candidates:
        return False, f"no ddsc*.dll or CycloneDDS*.dll found in {cdds_bin}"

    # Attempt to load the first candidate via ctypes
    dll_path = candidates[0]
    try:
        ctypes.CDLL(str(dll_path))
        return True, f"loaded: {dll_path.name}"
    except OSError as exc:
        return False, f"ctypes load failed for {dll_path.name}: {exc}"


# ── Category: Wheelhouse ──────────────────────────────────────────────────

@check("runtime/wheels/ directory present", critical=False, category="wheels")
def check_wheels_dir():
    p = RUNTIME_DIR / "wheels"
    if not p.is_dir():
        return False, f"not found: {p}"
    wheel_files = list(p.glob("*.whl"))
    return bool(wheel_files), f"{len(wheel_files)} wheel(s)"


@check("runtime/env/install_state.json shows installed=true", critical=False, category="wheels")
def check_install_state():
    p = RUNTIME_DIR / "env" / "install_state.json"
    if not p.exists():
        return False, "not found — install.bat has not run"
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        ok = data.get("installed") is True
        mode = data.get("install_mode", "unknown")
        ts = data.get("install_timestamp", "unknown")
        return ok, f"installed={ok}, mode={mode}, timestamp={ts}"
    except Exception as exc:
        return False, str(exc)


# ── Category: Manifest ────────────────────────────────────────────────────

@check("runtime/env/manifest.json readable and populated", critical=False, category="manifest")
def check_manifest():
    p = RUNTIME_DIR / "env" / "manifest.json"
    if not p.exists():
        return False, "not found — run scripts/build_runtime.bat"
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        py_ver = data.get("python_version") or "not set"
        cdds_ver = data.get("cyclonedds_version") or "not set"
        wheels_n = len(data.get("wheel_packages", []))
        populated = py_ver != "not set"
        return populated, f"python={py_ver}, cyclonedds={cdds_ver}, wheels={wheels_n}"
    except Exception as exc:
        return False, str(exc)


# ── Category: Project imports ─────────────────────────────────────────────

@check("UnitPort project root in sys.path", critical=True, category="project")
def check_project_in_path():
    root_str = str(PROJECT_ROOT)
    ok = root_str in sys.path
    if not ok:
        sys.path.insert(0, root_str)
        ok = True  # we just fixed it — warn so user knows start.bat should set PYTHONPATH
        return False, f"project root not in sys.path at launch; PYTHONPATH may not be set by start.bat"
    return True, str(PROJECT_ROOT)


@check("bin.core.config_manager importable", critical=True, category="project")
def check_config_manager():
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from bin.core.config_manager import ConfigManager  # noqa: F401
        return True, "ok"
    except ImportError as exc:
        return False, str(exc)


@check("models.sdk_manager importable", critical=True, category="project")
def check_sdk_manager():
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from models.sdk_manager import verify_registered_sdks, configure_cyclonedds_env  # noqa: F401
        return True, "ok"
    except ImportError as exc:
        return False, str(exc)


# ── Runner ────────────────────────────────────────────────────────────────

_ALL_CHECKS = [
    # Python runtime
    check_python_exe,
    check_python_version,
    check_pip_available,
    check_site_packages,
    # Core imports
    check_pyside6,
    check_numpy,
    check_mujoco,
    check_cyclonedds_python,
    check_unitree_sdk,
    # CycloneDDS native
    check_cyclonedds_dir,
    check_cyclonedds_home,
    check_cyclonedds_dll,
    # Wheelhouse
    check_wheels_dir,
    check_install_state,
    # Manifest
    check_manifest,
    # Project
    check_project_in_path,
    check_config_manager,
    check_sdk_manager,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="UnitPort runtime verification")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--category", default=None,
        help="Run only checks in this category (python/imports/cyclonedds/wheels/manifest/project)"
    )
    args = parser.parse_args()

    if not args.json:
        print("=" * 60)
        print("UnitPort Runtime Verification")
        print(f"Project root : {PROJECT_ROOT}")
        print(f"Runtime dir  : {RUNTIME_DIR}")
        print(f"Python       : {sys.version}")
        print("=" * 60)

    checks_to_run = _ALL_CHECKS
    if args.category:
        checks_to_run = [c for c in _ALL_CHECKS if True]  # run all, filter by decorator attribute below

    current_cat = None
    for fn in _ALL_CHECKS:
        # derive category from RESULTS after each call via the decorator
        before = len(RESULTS)
        fn()
        if RESULTS and not args.json:
            cat = RESULTS[-1].get("category", "")
            if cat != current_cat:
                current_cat = cat
                # print category header before this result
                idx = len(RESULTS) - 1
                # reprint with header (already printed inline; print separator before next category)
                pass

    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    warned = sum(1 for r in RESULTS if r["status"] == "WARN")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total = len(RESULTS)

    if args.json:
        output = {
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "total": total,
            "overall": "PASS" if failed == 0 else "FAIL",
            "checks": RESULTS,
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print(f"Result: {passed} PASS  {warned} WARN  {failed} FAIL  (total {total})")
        if failed > 0:
            critical_failures = [r["check"] for r in RESULTS if r["status"] == "FAIL"]
            print(f"FAILED checks: {critical_failures}")
            print("VERIFICATION FAILED — fix critical errors before distribution.")
        elif warned > 0:
            warn_list = [r["check"] for r in RESULTS if r["status"] == "WARN"]
            print(f"WARNINGS (non-critical): {warn_list}")
            print("VERIFICATION PASSED WITH WARNINGS — optional features may be unavailable.")
        else:
            print("VERIFICATION PASSED — runtime is fully assembled.")
        print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
