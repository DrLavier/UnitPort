# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SdkManager -- per-brand SDK clone & dependency-install manager.

Port of DEMO ``src/system/models/sdk_manager.py``, scoped down to the
operations the wizard + ``PostSetupTask`` actually need:

- :meth:`load_registry` -- enumerate ``(brand, project_key, url)`` tuples
  from ``registers.brands.brands_packages.json``.
- :meth:`get_project` -- look up a single ``SdkProject`` by project key.
- :meth:`ensure_selected_sdks` -- clone + pip-install only the
  user-picked subset (wizard happy path).
- :meth:`ensure_registered_sdks` -- clone + pip-install everything in
  the registry (skipped-wizard fallback).
- :meth:`register_isaaclab_path` -- delegate to ``EngineService`` for
  the wizard's "Locate existing Isaac Lab" branch.

What we do NOT port in this round (deferred to follow-up work):

- ``configure_cyclonedds_env`` -- separate concern, lives wherever DDS
  bring-up lands in RELEASE.
- ``configure_registered_sdk_imports`` / ``verify_registered_sdks`` --
  ``sys.path`` injection + lightweight import probing belong on a
  dedicated start-up step, not on the SDK manager.
- Packaged-runtime probing (``_runtime_python_executable`` etc.) --
  RELEASE always runs inside ``.venv311``; no second-interpreter probe.

Storage:
- Repo clones live under ``custom_mods/runtime/sdk_extensions/<brand
  display name>/<project key>/`` -- bit-exact match to DEMO so a user
  who already cloned via DEMO doesn't redo the work.
- Per-project install fingerprint (avoids repeat `pip install`) lives
  outside the project tree, under ``Paths.USER_CONFIG_DIR / "sdk" /
  install_state.json`` -- written via ``push_data`` (project tree stays
  clean per the no-user-state-in-source rule).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from registers import brands
from unitport_sdk import (
    Paths,
    log_error,
    log_info,
    log_success,
    log_warning,
    push_data,
    read_data,
    save_data,
)


ProgressCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SdkProject:
    """One brand-SDK repo entry. ``path`` is the on-disk clone target."""

    brand: str
    brand_dir: Path
    name: str
    url: str
    install_subdir: str = ""

    @property
    def path(self) -> Path:
        return self.brand_dir / self.name

    @property
    def install_path(self) -> Path:
        """Directory pip should target when installing this SDK.

        Most upstream repos expose ``setup.py`` / ``requirements.txt`` at
        the repo root, so this defaults to :attr:`path`. A few upstream
        repos (e.g. ``tfoldi/go2-webrtc``) keep the Python distribution
        under a subdirectory; ``brands_packages.json`` declares the
        per-project override via ``sdk_project_install_subdir``.
        """
        if self.install_subdir:
            return self.path / self.install_subdir
        return self.path


@dataclass(frozen=True)
class _InstallTarget:
    kind: str   # "requirements" or "package"
    path: Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _sdk_extensions_root() -> Path:
    return Paths.CUSTOM_MODS_DIR / "runtime" / "sdk_extensions"


def _install_state_path() -> Path:
    """Absolute path to ``install_state.json`` — machine-level shared state.

    The SDK install fingerprint is a per-machine fact (whether Isaac Lab,
    SB3, etc. are physically installed on this host); not per-account.
    Storing it under USER_CONFIG_DIR would force re-detection on every
    account login. We write it under the WORKSPACE root via DataManager
    directly (NOT push_data, which would prepend USER_CONFIG_DIR).
    """
    from application.service.user_workspace import machine_config_dir
    return machine_config_dir() / "sdk" / "install_state.json"


# ---------------------------------------------------------------------------
# Logging shim -- emits to LogSignal AND to the optional callback
# ---------------------------------------------------------------------------

def _emit(message: str, level: str = "info", callback: Optional[ProgressCallback] = None) -> None:
    if callback is not None:
        try:
            callback(level, message)
        except Exception:  # pragma: no cover -- never let the UI shim raise
            pass
    sink = {
        "info": log_info,
        "success": log_success,
        "warning": log_warning,
        "error": log_error,
    }.get(level, log_info)
    sink(message)


# ---------------------------------------------------------------------------
# SdkManager
# ---------------------------------------------------------------------------

class SdkManager:
    """Singleton facade over ``registers.brands`` + git clone + pip install.

    Strict naming per the migration plan: ``class SdkManager``.
    """

    _instance: Optional["SdkManager"] = None

    def __new__(cls) -> "SdkManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._sdk_root = _sdk_extensions_root()
        self._projects: List[SdkProject] = []
        self._project_index: Dict[str, SdkProject] = {}
        self._loaded = False
        self._initialized = True

    # ------------------------------------------------------------------
    # Registry enumeration
    # ------------------------------------------------------------------

    def load_registry(self, force_reload: bool = False) -> List[SdkProject]:
        """Read ``registers.brands.brands_packages.json`` and build the
        ``SdkProject`` list. Idempotent.
        """
        if self._loaded and not force_reload:
            return list(self._projects)

        brands.load()
        self._projects = []
        self._project_index = {}

        for brand_dict in brands.list_brands():
            brand_id = str(brand_dict.get("id", "")).strip()
            if not brand_id:
                continue
            display_name = str(brand_dict.get("display_name", brand_id))
            brand_dir = self._sdk_root / display_name
            url_map = dict(brand_dict.get("sdk_project_urls", {}) or {})
            subdir_map = dict(brand_dict.get("sdk_project_install_subdir", {}) or {})
            for name in brand_dict.get("sdk_project_keys", []) or []:
                url = str(url_map.get(name, "") or "")
                if not url:
                    log_warning(
                        f"[sdk] {brand_id}/{name}: missing sdk_project_urls "
                        f"entry in brands_packages.json -- skipping"
                    )
                    continue
                project = SdkProject(
                    brand=brand_id,
                    brand_dir=brand_dir,
                    name=str(name),
                    url=url,
                    install_subdir=str(subdir_map.get(name, "") or ""),
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
        seen: List[str] = []
        for project in self.load_registry():
            if project.brand not in seen:
                seen.append(project.brand)
        return seen

    def get_registered_brand_dirs(self) -> List[Path]:
        return [
            self._sdk_root / brand
            for brand in self.get_registered_brand_names()
        ]

    # ------------------------------------------------------------------
    # Bulk + selective ensure
    # ------------------------------------------------------------------

    def ensure_registered_sdks(
        self,
        *,
        progress: Optional[ProgressCallback] = None,
        strict: bool = False,
    ) -> List[Path]:
        """Clone (if missing) + dep-install every project in the registry.

        Used when the user skipped the wizard and we should fall back to
        DEMO's "install everything" default. Each per-project failure
        only logs unless ``strict=True``.
        """
        ensured: List[Path] = []
        for project in self.load_registry():
            if project.path.exists():
                _emit(
                    f"[sdk] {project.brand}/{project.name} already present",
                    "info", progress,
                )
                ensured.append(project.path)
            else:
                try:
                    self._clone_project(project, progress=progress)
                    ensured.append(project.path)
                except Exception as exc:
                    _emit(
                        f"[sdk] download failed for {project.brand}/{project.name}: {exc}",
                        "error", progress,
                    )
                    if strict:
                        raise
                    continue
            try:
                self._ensure_project_dependencies(project, progress=progress)
            except Exception as exc:
                _emit(
                    f"[sdk] dep install failed for {project.brand}/{project.name}: {exc}",
                    "error", progress,
                )
                if strict:
                    raise
        return ensured

    def ensure_selected_sdks(
        self,
        selections: List[Dict[str, str]],
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Path]:
        """Clone + dep-install only the user-picked subset.

        Each selection dict has keys ``brand``, ``key``, ``url`` (the
        wizard fills these from :meth:`get_projects`). Missing keys are
        treated as "skip and warn" rather than fatal -- partial
        wizard-state is better than aborting the whole post-setup.
        """
        if not selections:
            _emit("[sdk] no SDK projects selected -- skipping", "info", progress)
            return []

        self.load_registry()
        ensured: List[Path] = []
        for sel in selections:
            brand_id = str(sel.get("brand", ""))
            project_key = str(sel.get("key", ""))
            project = self.get_project(project_key)
            if project is None:
                _emit(
                    f"[sdk] unknown project: {brand_id}/{project_key} -- skipped",
                    "warning", progress,
                )
                continue
            if project.path.exists():
                _emit(
                    f"[sdk] {brand_id}/{project_key} already present",
                    "info", progress,
                )
                ensured.append(project.path)
            else:
                try:
                    self._clone_project(project, progress=progress)
                    ensured.append(project.path)
                except Exception as exc:
                    _emit(
                        f"[sdk] download failed for {brand_id}/{project_key}: {exc}",
                        "error", progress,
                    )
                    continue
            try:
                self._ensure_project_dependencies(project, progress=progress)
            except Exception as exc:
                _emit(
                    f"[sdk] dep install failed for {brand_id}/{project_key}: {exc}",
                    "error", progress,
                )
        return ensured

    # ------------------------------------------------------------------
    # Isaac Lab path bridge
    # ------------------------------------------------------------------

    def register_isaaclab_path(
        self,
        path: str,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> bool:
        """Wizard's "Locate existing Isaac Lab" finalizer.

        Delegates to :class:`EngineService` (the modernized RELEASE
        replacement for DEMO's ``EngineRegistry``). Returns True iff
        the path validates AND the registry write succeeded.
        """
        p = Path(path)
        if not p.is_dir():
            _emit(f"[isaaclab] path does not exist: {path}", "error", progress)
            return False

        from application.service.engines import get_engine_service
        ok = get_engine_service().register_isaac_local(str(p.resolve()))
        if ok:
            _emit(f"[isaaclab] registered: {p.resolve()}", "success", progress)
        else:
            _emit(
                f"[isaaclab] registration failed for {p.resolve()} (validation rejected)",
                "warning", progress,
            )
        return ok

    # ==================================================================
    # Internal: clone + dep install
    # ==================================================================

    def _clone_project(
        self,
        project: SdkProject,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        project.brand_dir.mkdir(parents=True, exist_ok=True)
        _emit(
            f"[sdk] downloading {project.brand}/{project.name} from {project.url}",
            "info", progress,
        )
        command = [
            "git", "clone", "--progress", project.url, str(project.path),
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
        rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"`{' '.join(command)}` exited with code {rc}")
        _emit(f"[sdk] ready: {project.brand}/{project.name}", "success", progress)

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
        project_state = dict(state.get(project_key, {}))

        changed = [
            t for t in targets
            if project_state.get(str(t.path)) != self._fingerprint_path(t.path)
        ]
        if not changed:
            return

        for target in changed:
            self._install_target(project, target, progress=progress)
            project_state[str(target.path)] = self._fingerprint_path(target.path)

        state[project_key] = project_state
        self._save_install_state(state)

    def _discover_install_targets(self, project: SdkProject) -> List[_InstallTarget]:
        """Enumerate pip targets at the project's install path.

        Most upstream SDKs ship a single artifact (requirements.txt OR
        setup.py / pyproject.toml). A few (e.g. ``tfoldi/go2-webrtc``)
        ship both, with an incomplete ``install_requires`` in setup.py
        and the full dependency list in requirements.txt — installing
        only one of the two leaves the env unusable. Order is fixed:
        requirements first so pip can resolve concrete deps before the
        package itself is wired up in editable mode.
        """
        install_path = project.install_path
        if project.install_subdir and not install_path.is_dir():
            log_warning(
                f"[sdk] {project.brand}/{project.name}: install_subdir "
                f"{project.install_subdir!r} does not exist under "
                f"{project.path} -- nothing to install"
            )
            return []

        candidates: List[_InstallTarget] = []
        requirements_path = install_path / "requirements.txt"
        if requirements_path.exists():
            candidates.append(_InstallTarget(kind="requirements", path=requirements_path))
        if (install_path / "setup.py").exists() or (install_path / "pyproject.toml").exists():
            candidates.append(_InstallTarget(kind="package", path=install_path))
        return candidates

    def _install_target(
        self,
        project: SdkProject,
        target: _InstallTarget,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        if target.kind == "requirements":
            command = [sys.executable, "-m", "pip", "install", "-r", str(target.path)]
            workdir = target.path.parent
            description = (
                f"installing requirements from {target.path.relative_to(project.path)}"
            )
        elif target.kind == "package":
            # --no-deps: vendored SDK setup.py / pyproject.toml often pin
            # outdated transitive deps (cyclonedds==0.10.2 etc.) that
            # conflict with the venv's authoritative requirements.txt.
            # The matching requirements.txt (when present) is installed
            # first by :meth:`_discover_install_targets` so deps are
            # already resolved by the time we reach this branch.
            command = [
                sys.executable, "-m", "pip", "install",
                "--no-deps", "-e", str(target.path),
            ]
            workdir = target.path
            description = "installing SDK package in editable mode (--no-deps)"
        else:
            raise ValueError(f"unsupported install target type: {target.kind}")

        _emit(f"{project.brand}/{project.name}: {description}", "info", progress)
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
                msg = line.strip()
                if msg:
                    _emit(f"{project.brand}/{project.name}: {msg}", "info", progress)
        if completed.returncode != 0:
            raise RuntimeError(
                f"`{' '.join(command)}` exited with code {completed.returncode}"
            )
        _emit(
            f"{project.brand}/{project.name}: dep installation complete",
            "success", progress,
        )

    # ------------------------------------------------------------------
    # Install-state fingerprint cache
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return _install_state_path()

    def _load_install_state(self) -> Dict[str, Dict[str, str]]:
        path = self._state_path()
        if not path.exists():
            return {}
        data = read_data(path)
        return data if isinstance(data, dict) else {}

    def _save_install_state(self, state: Dict[str, Dict[str, str]]) -> None:
        if not save_data(self._state_path(), state):
            log_warning("[sdk] failed to persist install_state.json")

    @staticmethod
    def _fingerprint_path(path: Path) -> str:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"

    @staticmethod
    def _normalize_key(value: str) -> str:
        return value.strip().lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Module-level convenience aliases (mirror DEMO's surface)
# ---------------------------------------------------------------------------

def get_sdk_manager() -> SdkManager:
    return SdkManager()


def ensure_selected_sdks(
    selections: List[Dict[str, str]],
    *,
    progress: Optional[ProgressCallback] = None,
) -> List[Path]:
    return SdkManager().ensure_selected_sdks(selections, progress=progress)


def ensure_registered_sdks(
    *,
    progress: Optional[ProgressCallback] = None,
    strict: bool = False,
) -> List[Path]:
    return SdkManager().ensure_registered_sdks(progress=progress, strict=strict)


def register_isaaclab_path(
    path: str,
    *,
    progress: Optional[ProgressCallback] = None,
) -> bool:
    return SdkManager().register_isaaclab_path(path, progress=progress)


__all__ = [
    "SdkManager",
    "SdkProject",
    "ProgressCallback",
    "get_sdk_manager",
    "ensure_selected_sdks",
    "ensure_registered_sdks",
    "register_isaaclab_path",
]
