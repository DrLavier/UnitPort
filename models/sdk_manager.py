#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDK registry and hot-plug bootstrap for models/ brands.

Startup path (production):
  verify_registered_sdks()  — lightweight import check, no network/build ops.
  configure_cyclonedds_env() — injects project-local CYCLONEDDS_HOME.

Developer/CI path only (requires UNITPORT_DEV_MODE=1):
  ensure_registered_sdks()  — may clone repos and run pip install.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from system.model_registry import (
    canonical_brand_ids,
    get_brand_spec,
    sdk_project_keys_for_brand,
    sdk_project_url,
)


ProgressCallback = Callable[[str, str], None]

# Project root inferred relative to this file (models/sdk_manager.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_DIR = _PROJECT_ROOT / "runtime"
_BRANDS_SDK_DIR = _PROJECT_ROOT / "brands_sdk"
_MENAGERIE_DIR = _PROJECT_ROOT / "models" / "mujoco_menagerie"
_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def _prepend_sys_path(path: Path) -> bool:
    """Insert *path* at the front of sys.path once.

    Returns True when the path was added, False when it was absent or already
    present.
    """
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    if not path.exists():
        return False
    existing = {str(Path(p).resolve()) if p else "" for p in sys.path}
    if resolved in existing:
        return False
    sys.path.insert(0, resolved)
    return True


def _runtime_python_executable() -> Optional[Path]:
    """Return the packaged Python executable when present."""
    system = platform.system()
    if system == "Windows":
        candidate = _RUNTIME_DIR / "python" / "python.exe"
        return candidate if candidate.exists() else None
    for name in ("python", "python3"):
        candidate = _RUNTIME_DIR / "python" / name
        if candidate.exists():
            return candidate
    return None


def _probe_module_with_runtime_python(
    module_name: str,
    extra_paths: Optional[List[Path]] = None,
) -> bool:
    """Probe whether *module_name* is importable in the packaged runtime."""
    runtime_python = _runtime_python_executable()
    if runtime_python is None:
        return False

    env = os.environ.copy()
    path_lines: List[str] = []
    for path in extra_paths or []:
        if path.exists():
            path_lines.append(f"sys.path.insert(0, {str(path)!r})")

    probe_code = textwrap.dedent(
        f"""
        import sys
        {'; '.join(path_lines)}
        import {module_name}
        print(getattr({module_name}, "__file__", "ok"))
        """
    ).strip()
    completed = subprocess.run(
        [str(runtime_python), "-c", probe_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    return completed.returncode == 0


def _emit(message: str, level: str = "info", callback: Optional[ProgressCallback] = None) -> None:
    """Send bootstrap progress to stdout and, if available, the in-app CMD log."""
    if callback is not None:
        callback(level, message)

    print(f"[sdk-bootstrap:{level}] {message}")

    try:
        from bin.core.logger import log_debug, log_error, log_info, log_success, log_warning

        logger_map = {
            "debug": log_debug,
            "info": log_info,
            "warning": log_warning,
            "error": log_error,
            "success": log_success,
        }
        logger_map.get(level, log_info)(message)
        # Startup runs synchronously in the main Qt thread, blocking the event loop.
        # processEvents() forces an immediate repaint so the loading screen shows this line.
        from PySide6.QtWidgets import QApplication
        _app = QApplication.instance()
        if _app is not None:
            _app.processEvents()
    except Exception:
        pass


@dataclass(frozen=True)
class SdkProject:
    brand: str
    brand_dir: Path
    name: str
    url: str

    @property
    def path(self) -> Path:
        return self.brand_dir / self.name


@dataclass(frozen=True)
class InstallTarget:
    kind: str
    path: Path


class SdkManager:
    """Load SDK registrations from the canonical model registry."""

    _instance: Optional["SdkManager"] = None

    def __new__(cls) -> "SdkManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.models_dir = Path(__file__).resolve().parent
        self.brands_sdk_dir = _BRANDS_SDK_DIR
        self.state_path = self.models_dir / ".sdk_install_state.json"
        self._projects: List[SdkProject] = []
        self._project_index: Dict[str, SdkProject] = {}
        self._loaded = False
        self._initialized = True

    def load_registry(self, force_reload: bool = False) -> List[SdkProject]:
        if self._loaded and not force_reload:
            return list(self._projects)

        self._projects = []
        self._project_index = {}

        for brand_id in canonical_brand_ids():
            brand_spec = get_brand_spec(brand_id)
            brand_dir_name = brand_spec.display_name if brand_spec is not None else brand_id
            brand_dir = self.brands_sdk_dir / brand_dir_name
            for name in sdk_project_keys_for_brand(brand_id):
                project = SdkProject(
                    brand=brand_id,
                    brand_dir=brand_dir,
                    name=name,
                    url=sdk_project_url(brand_id, name),
                )
                self._projects.append(project)
                self._project_index[self._normalize_key(name)] = project

        self._loaded = True
        return list(self._projects)

    def get_projects(self) -> List[SdkProject]:
        return self.load_registry()

    def get_project(self, name: str) -> Optional[SdkProject]:
        self.load_registry()
        return self._project_index.get(self._normalize_key(name))

    def get_registered_brand_names(self) -> List[str]:
        projects = self.load_registry()
        seen = []
        for project in projects:
            if project.brand not in seen:
                seen.append(project.brand)
        return seen

    def get_registered_brand_dirs(self) -> List[Path]:
        return [self.brands_sdk_dir / brand for brand in self.get_registered_brand_names()]

    def resolve_path(self, key: str, fallback: Optional[Path] = None) -> Optional[Path]:
        """Resolve a logical SDK path key to a concrete directory under brands_sdk/."""
        self.load_registry()
        normalized = self._normalize_key(key)
        project = self._project_index.get(normalized)
        if project is not None:
            return project.path

        if normalized == "models_root":
            return self.models_dir

        if normalized == "unitree_sdk":
            project = self._project_index.get("unitree_sdk2_python")
            return project.path if project is not None else fallback

        if normalized == "unitree_mujoco":
            project = self._project_index.get("unitree_mujoco")
            return project.path if project is not None else fallback

        if normalized == "unitree_robots":
            mujoco = self.resolve_path("unitree_mujoco")
            return (mujoco / "unitree_robots") if mujoco is not None else fallback

        return fallback

    def ensure_registered_sdks(
        self,
        *,
        progress: Optional[ProgressCallback] = None,
        strict: bool = False,
    ) -> List[Path]:
        """Ensure all SDK repos declared in the canonical model registry are present."""
        ensured: List[Path] = []
        for project in self.load_registry():
            if project.path.exists():
                ensured.append(project.path)
            else:
                try:
                    self._clone_project(project, progress=progress)
                    ensured.append(project.path)
                except Exception as exc:
                    _emit(
                        f"SDK download failed for {project.brand}/{project.name}: {exc}",
                        "error",
                        progress,
                    )
                    if strict:
                        raise

            try:
                self._ensure_project_dependencies(project, progress=progress)
            except Exception as exc:
                _emit(
                    f"SDK dependency install failed for {project.brand}/{project.name}: {exc}",
                    "error",
                    progress,
                )
                if strict:
                    raise

        return ensured

    def _clone_project(
        self,
        project: SdkProject,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        project.brand_dir.mkdir(parents=True, exist_ok=True)
        _emit(
            f"Downloading SDK {project.brand}/{project.name} from {project.url}",
            "info",
            progress,
        )

        command = [
            "git",
            "clone",
            "--progress",
            project.url,
            str(project.path),
        ]

        process = subprocess.Popen(
            command,
            cwd=str(project.brand_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            message = line.strip()
            if message:
                _emit(f"{project.brand}/{project.name}: {message}", "info", progress)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"`{' '.join(command)}` exited with code {return_code}")

        _emit(f"SDK ready: {project.brand}/{project.name}", "success", progress)

    def _ensure_project_dependencies(
        self,
        project: SdkProject,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        targets = self._discover_install_targets(project)
        if not targets:
            return

        state = self._load_install_state()
        project_key = f"{project.brand}/{project.name}"
        project_state = state.get(project_key, {})

        changed_targets = [
            target
            for target in targets
            if project_state.get(str(target.path)) != self._fingerprint_path(target.path)
        ]
        if not changed_targets:
            return

        for target in changed_targets:
            self._install_target(project, target, progress=progress)
            project_state[str(target.path)] = self._fingerprint_path(target.path)

        state[project_key] = project_state
        self._save_install_state(state)

    def _discover_install_targets(self, project: SdkProject) -> List[InstallTarget]:
        candidates: List[InstallTarget] = []
        requirements_path = project.path / "requirements.txt"

        if requirements_path.exists():
            candidates.append(InstallTarget(kind="requirements", path=requirements_path))
            return candidates

        setup_path = project.path / "setup.py"
        pyproject_path = project.path / "pyproject.toml"
        if setup_path.exists() or pyproject_path.exists():
            candidates.append(InstallTarget(kind="package", path=project.path))

        return candidates

    def _install_target(
        self,
        project: SdkProject,
        target: InstallTarget,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        if target.kind == "requirements":
            command = [sys.executable, "-m", "pip", "install", "-r", str(target.path)]
            workdir = target.path.parent
            description = f"Installing requirements from {target.path.relative_to(project.path)}"
        elif target.kind == "package":
            command = [sys.executable, "-m", "pip", "install", "-e", str(target.path)]
            workdir = project.path
            description = "Installing SDK package in editable mode"
        else:
            raise ValueError(f"Unsupported install target type: {target.kind}")

        _emit(
            f"{project.brand}/{project.name}: {description}",
            "info",
            progress,
        )
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        for stream in (completed.stdout, completed.stderr):
            for line in (stream or "").splitlines():
                message = line.strip()
                if message:
                    _emit(f"{project.brand}/{project.name}: {message}", "info", progress)

        if completed.returncode != 0:
            raise RuntimeError(f"`{' '.join(command)}` exited with code {completed.returncode}")

        _emit(
            f"{project.brand}/{project.name}: dependency installation complete",
            "success",
            progress,
        )

    def _load_install_state(self) -> Dict[str, Dict[str, str]]:
        if not self.state_path.exists():
            return {}

        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return {}

        return data if isinstance(data, dict) else {}

    def _save_install_state(self, state: Dict[str, Dict[str, str]]) -> None:
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

    @staticmethod
    def _fingerprint_path(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    @staticmethod
    def _normalize_key(value: str) -> str:
        return value.strip().lower().replace("-", "_")



def check_mujoco_menagerie() -> bool:
    """Return True if models/mujoco_menagerie exists and is non-empty."""
    return _MENAGERIE_DIR.is_dir() and any(_MENAGERIE_DIR.iterdir())


def ensure_mujoco_menagerie(
    *,
    progress: Optional[ProgressCallback] = None,
) -> bool:
    """Ensure models/mujoco_menagerie is present; clone from GitHub if missing.

    Uses ``--depth=1`` for a shallow clone (fast, ~300 MB).
    Returns True on success; False when git is unavailable or the clone fails.
    Startup continues in degraded mode on failure — never raises.
    """
    if check_mujoco_menagerie():
        _emit("[menagerie] mujoco_menagerie already present.", "info", progress)
        return True

    _MENAGERIE_DIR.parent.mkdir(parents=True, exist_ok=True)
    _emit(
        f"[menagerie] models/mujoco_menagerie not found — cloning from {_MENAGERIE_URL} ...",
        "info",
        progress,
    )

    command = [
        "git", "clone",
        "--depth=1",
        "--progress",
        _MENAGERIE_URL,
        str(_MENAGERIE_DIR),
    ]

    try:
        process = subprocess.Popen(
            command,
            cwd=str(_MENAGERIE_DIR.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            msg = line.strip()
            if msg:
                _emit(f"[menagerie] {msg}", "info", progress)
        return_code = process.wait()
    except Exception as exc:
        _emit(f"[menagerie] clone failed — {exc}", "error", progress)
        return False

    if return_code != 0:
        _emit(
            f"[menagerie] git clone exited with code {return_code} — MuJoCo assets unavailable.",
            "error",
            progress,
        )
        return False

    _emit("[menagerie] mujoco_menagerie ready.", "success", progress)
    return True


def configure_cyclonedds_env(project_root: Optional[Path] = None) -> str:
    """Inject project-local CYCLONEDDS_HOME into os.environ, overriding any global path.

    Prefers ``runtime/cyclonedds/`` inside the project.  Only overwrites the
    env var when the current value does not already point to the project-local
    directory (so callers that pre-set it via start.bat are not disrupted).

    Returns the resolved CYCLONEDDS_HOME string that is now active.
    """
    root = Path(project_root) if project_root else _PROJECT_ROOT
    local_cdds = root / "runtime" / "cyclonedds"

    current = os.environ.get("CYCLONEDDS_HOME", "")
    if current:
        try:
            if Path(current).resolve() == local_cdds.resolve():
                return current  # already correct
        except Exception:
            pass

    if local_cdds.is_dir():
        os.environ["CYCLONEDDS_HOME"] = str(local_cdds)
        cdds_bin = local_cdds / "bin"
        if cdds_bin.is_dir():
            path_now = os.environ.get("PATH", "")
            if str(cdds_bin) not in path_now:
                os.environ["PATH"] = str(cdds_bin) + os.pathsep + path_now
        cdds_lib = local_cdds / "lib"
        if platform.system() == "Linux" and cdds_lib.is_dir():
            lib_path_now = os.environ.get("LD_LIBRARY_PATH", "")
            if str(cdds_lib) not in lib_path_now:
                os.environ["LD_LIBRARY_PATH"] = str(cdds_lib) + os.pathsep + lib_path_now
        _emit(f"CYCLONEDDS_HOME → {local_cdds} (project-local)", "info")
        return str(local_cdds)

    if current:
        _emit(
            f"Project-local CycloneDDS not found; keeping existing CYCLONEDDS_HOME={current}",
            "warning",
        )
        return current

    _emit("CycloneDDS not configured (runtime/cyclonedds/ absent, CYCLONEDDS_HOME not set)", "warning")
    return ""


def configure_registered_sdk_imports(
    *,
    progress: Optional[ProgressCallback] = None,
) -> List[Path]:
    """Expose vendored SDK package roots to Python import resolution.

    This is a lightweight bootstrap step for startup verification and local
    runtime use. It does not build or install anything; it only prepends known
    project-local SDK roots to ``sys.path`` when they exist.
    """
    manager = SdkManager()
    added: List[Path] = []

    known_import_roots: List[Path] = []
    runtime_site_packages = _RUNTIME_DIR / "python" / "Lib" / "site-packages"
    if runtime_site_packages.is_dir():
        known_import_roots.append(runtime_site_packages)

    unitree_sdk = manager.resolve_path("unitree_sdk")
    if unitree_sdk is not None:
        known_import_roots.append(unitree_sdk)

    for root in known_import_roots:
        if _prepend_sys_path(root):
            added.append(root)
            _emit(f"SDK import path enabled: {root}", "info", progress)

    return added


def verify_registered_sdks(
    *,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, object]:
    """Lightweight startup verification — no network calls, no builds.

    Checks whether registered SDK directories exist on disk and attempts
    key imports.  Returns a status dict suitable for startup logging.

    Never raises; failures are captured in the returned dict.
    """
    manager = SdkManager()
    projects = manager.load_registry()
    configure_registered_sdk_imports(progress=progress)

    results: Dict[str, object] = {
        "mode": "verify",
        "sdk_dirs_found": [],
        "sdk_dirs_missing": [],
        "import_checks": {},
        "degraded": False,
    }

    for project in projects:
        key = f"{project.brand}/{project.name}"
        if project.path.exists():
            results["sdk_dirs_found"].append(key)  # type: ignore[attr-defined]
        else:
            results["sdk_dirs_missing"].append(key)  # type: ignore[attr-defined]
            _emit(f"Optional SDK not present (degraded mode): {key}", "warning", progress)

    runtime_site_packages = _RUNTIME_DIR / "python" / "Lib" / "site-packages"
    unitree_sdk = manager.resolve_path("unitree_sdk")

    # Attempt lightweight import probes for known optional SDK packages
    _import_probes: List[Tuple[str, str]] = [
        ("unitree_sdk2py", "unitree"),
        ("cyclonedds", "cyclonedds"),
    ]
    for module_name, label in _import_probes:
        try:
            __import__(module_name)
            results["import_checks"][label] = "ok"  # type: ignore[index]
            continue
        except Exception:
            pass

        fallback_paths: List[Path] = []
        if runtime_site_packages.is_dir():
            fallback_paths.append(runtime_site_packages)
        if module_name == "unitree_sdk2py" and unitree_sdk is not None:
            fallback_paths.append(unitree_sdk)

        if _probe_module_with_runtime_python(module_name, extra_paths=fallback_paths):
            results["import_checks"][label] = "ok"  # type: ignore[index]
            _emit(
                f"Optional SDK import verified via packaged runtime: {label} ({module_name})",
                "info",
                progress,
            )
        else:
            results["import_checks"][label] = "unavailable"  # type: ignore[index]
            _emit(f"Optional SDK import unavailable: {label} ({module_name})", "warning", progress)

    if results["sdk_dirs_missing"] or any(
        v != "ok" for v in results["import_checks"].values()  # type: ignore[union-attr]
    ):
        results["degraded"] = True

    return results


def ensure_registered_sdks(
    *,
    progress: Optional[ProgressCallback] = None,
    strict: bool = False,
) -> List[Path]:
    """Clone missing SDK repos and install their dependencies.

    DEVELOPER / CI USE ONLY.
    This function performs network I/O (git clone) and runs pip install.
    It must NOT be called from normal application startup in production.

    Gate with UNITPORT_DEV_MODE=1 at the call site before invoking this.
    """
    return SdkManager().ensure_registered_sdks(progress=progress, strict=strict)


def get_sdk_manager() -> SdkManager:
    return SdkManager()
