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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


ProgressCallback = Callable[[str, str], None]

# Project root inferred relative to this file (models/sdk_manager.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_DIR = _PROJECT_ROOT / "runtime"


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
    """Load SDK registrations and ensure required repos exist under models/."""

    _instance: Optional["SdkManager"] = None

    def __new__(cls) -> "SdkManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.models_dir = Path(__file__).resolve().parent
        self.registry_path = self.models_dir / "brands_list.json"
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

        raw = self._read_registry_json()
        for brand, repos in raw.items():
            brand_dir = self.models_dir / brand
            if not isinstance(repos, dict):
                continue
            for name, url in repos.items():
                if not isinstance(name, str) or not isinstance(url, str):
                    continue
                clean_url = url.strip()
                if not clean_url:
                    continue
                project = SdkProject(
                    brand=brand,
                    brand_dir=brand_dir,
                    name=name.strip(),
                    url=clean_url,
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
        return [self.models_dir / brand for brand in self.get_registered_brand_names()]

    def resolve_path(self, key: str, fallback: Optional[Path] = None) -> Optional[Path]:
        """Resolve a logical SDK path key to a concrete directory under models/."""
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
        """Ensure all repos listed in brands_list.json are present."""
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

    def _read_registry_json(self) -> Dict[str, Dict[str, str]]:
        if not self.registry_path.exists():
            return {}

        with open(self.registry_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            return {}
        return data

    @staticmethod
    def _normalize_key(value: str) -> str:
        return value.strip().lower().replace("-", "_")



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

    # Attempt lightweight import probes for known optional SDK packages
    _import_probes: List[Tuple[str, str]] = [
        ("unitree_sdk2py", "unitree"),
        ("cyclonedds", "cyclonedds"),
    ]
    for module_name, label in _import_probes:
        try:
            __import__(module_name)
            results["import_checks"][label] = "ok"  # type: ignore[index]
        except ImportError:
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
