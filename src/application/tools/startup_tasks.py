# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""TasksManager-compatible Task subclasses used by ``UnitPortMain`` startup.

Three classes are exposed:

- :class:`FuncTask` -- generic ``Task`` wrapper that runs an arbitrary
  zero-arg callable. Used for the existing ``init_verification`` /
  ``data_load`` / ``project_load`` startup phases whose bodies live as
  methods on ``UnitPortMain``.

- :class:`ProvisioningTask` -- the post-bootstrap minimum-viable gate.
  ``install.bat`` / ``install.sh`` only install the bare set needed to
  draw a Qt window (``PyQt6`` + ``loguru``); this task brings the venv
  the rest of the way up: every package in ``requirements.txt``, plus a
  CUDA torch upgrade if an NVIDIA GPU is present, plus ``unitport://``
  URL-scheme registration. **Order matters:** requirements run FIRST
  (installing CPU torch among everything else), then the CUDA wheel
  force-reinstalls torch on top. Doing CUDA first and requirements
  second meant requirements could downgrade the just-installed CUDA
  wheel to a broken half-install -- the bug that produced
  ``"pip install completed but 3 package(s) still unimportable:
  ['torch','stable-baselines3','pyqtgraph']"``. Each subprocess streams
  stdout line-by-line into the on-screen LoadingScreen log via
  ``self.log_info``.

- :class:`PostSetupTask` -- runs after both ``ProvisioningTask`` AND
  the ``InstallConfigWizard`` finish. Drives the heavy / interactive
  installs deferred from ``install.bat``: brand-SDK clones, MuJoCo
  Menagerie sparse-checkout, loco-mujoco clone, Isaac Lab
  install/locate/cloud-deploy, ROS2 native-detect or Docker bridge,
  then writes the final ``install_state.json``. Selections come from
  the wizard; ``skipped: True`` means "do the legacy default" (full
  Menagerie clone, all registered SDKs, no Isaac Lab).

All three obey the SDK's TasksManager contract: log via ``self.log_*``
(thread-safe LogSignal marshalling), use ``self.check_cancelled()`` at
loop boundaries so cancellation propagates, never call ``time.sleep``.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

from unitport_sdk import Paths, Task, up_data

# bootstrap/_check_requirements.py is the single source of truth for the
# requirements parse + probe logic; loading it directly guarantees the
# command-line path (start.bat) and the in-process path (this Task) stay
# in lock-step. We load by file path rather than ``from bootstrap._check_requirements
# import ...`` because PROJECT_ROOT is intentionally kept off sys.path -- only
# ``src/`` is added -- so the training-pipeline package under ``src/scripts/``
# stays unambiguous. spec_from_file_location is robust regardless of sys.path
# order.
def _load_check_requirements():
    path = Paths.PROJECT_ROOT / "bootstrap" / "_check_requirements.py"
    spec = importlib.util.spec_from_file_location(
        "_unitport_check_requirements", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_check_requirements = _load_check_requirements()
ImportFailure = _check_requirements.ImportFailure
iter_required_packages = _check_requirements.iter_required_packages
probe_imports = _check_requirements.probe_imports


_INSTALL_STATE_REL = Path("env") / "install_state.json"
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"

# sys.modules entries that must be flushed after a pip install so the
# subsequent re-probe sees the new wheel rather than the stale failure
# (importlib short-circuits on cached failures). Includes torch and the
# packages most likely to have already been imported (and thus cached as
# None) by an earlier probe in the same Python process.
_TORCH_AND_DEPENDENTS = (
    "torch",
    "stable_baselines3",
    "pyqtgraph",
    "torchvision",
    "torchaudio",
)


def _flush_module_cache(also: Optional[List[str]] = None) -> None:
    """Drop only **failed-import sentinels** from ``sys.modules`` so a
    re-probe after pip install can actually retry the import.

    Why not pop successful modules too: many target packages here
    (torch, mujoco, onnx, pyqtgraph, ...) are C extensions that
    register types/enums globally via nanobind / pybind11 at module
    init. Re-importing such a module in the same process re-runs that
    init and the second registration aborts with::

        Critical nanobind error: refusing to add duplicate key ...
        nanobind: type 'OpSchema' was already registered!

    So the only thing we need to clear are entries that ``importlib``
    cached as a failure (stored as ``None`` in ``sys.modules``);
    ``importlib.invalidate_caches()`` plus removing those sentinels is
    enough to let a re-probe pick up the freshly-installed wheel.
    Already-imported modules are left alone -- if they were importable
    before the pip run, they still are, and re-importing them risks the
    nanobind crash above.
    """
    importlib.invalidate_caches()
    targets = set(_TORCH_AND_DEPENDENTS)
    if also:
        targets.update(also)
    for mod in list(sys.modules):
        for prefix in targets:
            if mod == prefix or mod.startswith(prefix + "."):
                if sys.modules.get(mod) is None:
                    sys.modules.pop(mod, None)
                break


class FuncTask(Task):
    """Run a zero-arg callable as a TasksManager task.

    Promoted out of ``main.py`` so the bootstrap sequence can submit
    bodies that live on ``UnitPortMain`` without each one needing its
    own ``Task`` subclass. New phase logic with non-trivial bodies should
    still grow into dedicated subclasses (like :class:`ProvisioningTask`).
    """

    def __init__(self, name: str, fn: Callable[[], Any]) -> None:
        super().__init__(name=name)
        self._fn = fn

    def run(self) -> Any:  # type: ignore[override]
        return self._fn()


class ProvisioningTask(Task):
    """Bring the venv from "bootstrap minimum" to "main UI can import".

    Runs three idempotent sub-steps in series:

    1. **requirements.txt** -- probe every entry, ``pip install -r`` if
       anything is missing. CPU torch gets installed here as part of the
       full set. ``--no-cache-dir`` avoids reusing a stale / truncated
       wheel left by an earlier interrupted install. After install,
       ``sys.modules`` is flushed for torch and known-cascading
       dependents (stable_baselines3, pyqtgraph) so the re-probe sees
       fresh state. Re-probe failure raises with per-package detail so
       the LoadingScreen log shows actionable next-step pip commands.
    2. **torch CUDA** -- if an NVIDIA GPU is present, force-reinstall
       torch from the cu124 index, replacing the CPU torch from step 1.
       Failure here is non-fatal: log a warning and keep the CPU torch.
       Also handles the cu124-vs-driver-mismatch case: if the new wheel
       installs but ``import torch`` then fails with a DLL-load error,
       fall back to PyPI CPU torch.
    3. **URL scheme** -- register ``unitport://`` if absent. Optional;
       failure logs a warning and continues. OAuth redirects need it.

    Step 1 is the only blocking one (raises on hard failure); steps 2
    and 3 only ``log_warning``. ``UnitPortMain._on_task_finished``
    handles the raised exception by logging an error and stopping
    startup -- no auto-quit, the LoadingScreen has a window frame so
    the user can read the message and close the window themselves.
    """

    def __init__(self, name: str = "provision") -> None:
        super().__init__(name=name)

    def run(self) -> dict:  # type: ignore[override]
        report: dict = {"steps": []}
        report["requirements"] = self._step_requirements(report)
        report["torch_cuda"] = self._step_torch_cuda(report)
        report["url_scheme"] = self._step_url_scheme(report)
        self._step_provisioning_state_partial(report)
        self.log_success("ready for wizard finalization")
        return report

    # ------------------------------------------------------------------
    # Step 1 -- requirements.txt (CPU torch + full set)
    # ------------------------------------------------------------------
    def _step_requirements(self, report: dict) -> dict:
        req_path = Paths.PROJECT_ROOT / "requirements.txt"
        if not req_path.exists():
            raise RuntimeError(f"requirements.txt not found at {req_path}")

        pairs = list(iter_required_packages(req_path))
        self.log_info(f"checking {len(pairs)} required packages")

        missing = probe_imports(pairs)
        if not missing:
            self.log_success("all required packages importable")
            report["steps"].append({"requirements": "skipped"})
            return {"status": "skipped", "checked": len(pairs), "installed": []}

        # Prefer the pinned lockfile when present so end-user installs
        # land on the exact versions the maintainer tested. Falls back to
        # requirements.txt when the lockfile is absent (dev clones / very
        # old release branches). Plan P1-2.
        lock_path = Paths.PROJECT_ROOT / "requirements.lock"
        install_target = lock_path if lock_path.exists() else req_path

        missing_names = [f.pkg for f in missing]
        self.log_warning(
            f"{len(missing)} package(s) missing: {missing_names}; "
            f"running pip install -r {install_target.name}"
        )

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "-r",
            str(install_target),
        ]
        self._run_streaming(cmd)

        # Failed imports get cached as None in sys.modules; without the
        # flush a successful pip install would still re-probe as
        # "missing" because importlib short-circuits cached failures.
        # The flush also covers the cascade case: if torch failed
        # earlier, every torch-dependent (sb3, pyqtgraph) was cached as
        # failed too.
        extra = [import_name for _, import_name in pairs]
        _flush_module_cache(also=extra)

        still_missing = probe_imports(pairs)
        if still_missing:
            self._report_unimportable(still_missing)
            still_names = [f.pkg for f in still_missing]
            raise RuntimeError(
                f"pip install completed but {len(still_missing)} package(s) "
                f"still unimportable: {still_names} (see log for per-package detail)"
            )

        installed_names = [f.pkg for f in missing]
        self.log_success(
            f"requirements complete; installed "
            f"{len(installed_names)} package(s)"
        )
        report["steps"].append({"requirements": "installed"})
        return {
            "status": "installed",
            "checked": len(pairs),
            "installed": installed_names,
        }

    # ------------------------------------------------------------------
    # Step 2 -- torch CUDA (replaces CPU torch from step 1)
    # ------------------------------------------------------------------
    def _step_torch_cuda(self, report: dict) -> dict:
        """Upgrade torch to the CUDA build if an NVIDIA GPU is present."""
        self.log_info("checking torch CUDA")

        # Already on a CUDA build? Decide by wheel variant
        # (``torch.version.cuda`` is the CUDA toolkit string for CUDA
        # wheels and ``None`` for CPU wheels) rather than by
        # ``cuda.is_available()``. The runtime check can transiently
        # return False (driver paging, no GPU attached this session,
        # WSL passthrough not active, ...) even when the correct CUDA
        # wheel is already installed -- and reinstalling a multi-GB
        # wheel won't fix a driver-side issue. Only fall through to
        # install when the installed variant is CPU.
        try:
            import torch  # noqa: WPS433
            installed_cuda = getattr(torch.version, "cuda", None)
            if installed_cuda:
                runtime_ok = False
                try:
                    runtime_ok = bool(torch.cuda.is_available())
                except Exception:
                    runtime_ok = False
                if runtime_ok:
                    self.log_info(
                        f"torch CUDA already available "
                        f"(torch={torch.__version__}, cuda={installed_cuda})"
                    )
                else:
                    self.log_warning(
                        f"torch CUDA wheel already installed "
                        f"(torch={torch.__version__}, cuda={installed_cuda}) "
                        f"but cuda.is_available()=False -- likely a driver/"
                        f"runtime issue, NOT reinstalling"
                    )
                report["steps"].append({"torch_cuda": "skipped"})
                return {"status": "skipped", "cuda": runtime_ok}
        except ImportError:
            # Step 1 should have installed CPU torch; if it didn't,
            # don't blow up here -- requirements step already raised on
            # hard failure. We're in the "torch importable but CPU" or
            # "torch missing entirely" branch; either way fall through.
            pass

        # NVIDIA detection. shutil.which is enough -- nvidia-smi only
        # exists when the NVIDIA driver is installed, which is the same
        # gate the CUDA wheel cares about.
        if shutil.which("nvidia-smi") is None:
            self.log_info(
                "no NVIDIA GPU detected; CPU torch (from step 1) is final"
            )
            report["steps"].append({"torch_cuda": "no_gpu"})
            return {"status": "no_gpu", "cuda": False}

        self.log_info(
            "NVIDIA GPU detected; force-reinstalling torch CUDA wheel"
        )
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "--force-reinstall",
            "torch",
            "--index-url",
            _TORCH_CUDA_INDEX,
        ]
        try:
            self._run_streaming(cmd)
        except RuntimeError as exc:
            self.log_warning(
                f"CUDA torch install failed ({exc}); keeping CPU torch"
            )
            report["steps"].append({"torch_cuda": "failed_keeping_cpu"})
            return {"status": "failed_keeping_cpu", "cuda": False}

        _flush_module_cache()

        # Verify the CUDA wheel actually loads. cu124 wheels need a
        # recent NVIDIA driver (>= 525); on older drivers the install
        # succeeds but ``import torch._C`` raises a DLL-load error. In
        # that case we fall back to CPU torch from PyPI.
        try:
            import torch  # noqa: WPS433
            cuda_ok = bool(torch.cuda.is_available())
        except BaseException as exc:  # noqa: BLE001
            self.log_warning(
                f"CUDA torch installed but import failed ({type(exc).__name__}: {exc}); "
                f"falling back to CPU torch from PyPI"
            )
            self._reinstall_cpu_torch_fallback()
            report["steps"].append({"torch_cuda": "fallback_cpu"})
            return {"status": "fallback_cpu", "cuda": False}

        if cuda_ok:
            self.log_success(
                f"torch CUDA installed "
                f"(torch={torch.__version__}, cuda={torch.version.cuda})"
            )
            report["steps"].append({"torch_cuda": "installed"})
            return {"status": "installed", "cuda": True}

        self.log_warning(
            "CUDA torch installed but cuda.is_available() is False; "
            "leaving CUDA build in place for further diagnosis"
        )
        report["steps"].append({"torch_cuda": "installed_but_unavailable"})
        return {"status": "installed_but_unavailable", "cuda": False}

    def _reinstall_cpu_torch_fallback(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "--force-reinstall",
            "torch",
        ]
        try:
            self._run_streaming(cmd)
            _flush_module_cache()
        except RuntimeError as exc:
            # Even the CPU fallback failed. log_warning and let the
            # main app boot -- the user will see torch errors when they
            # try to do anything that needs it, and at least the UI is
            # usable.
            self.log_warning(
                f"CPU torch fallback install also failed ({exc}); "
                f"torch is in an inconsistent state"
            )

    # ------------------------------------------------------------------
    # Step 3 -- unitport:// URL scheme (optional)
    # ------------------------------------------------------------------
    def _step_url_scheme(self, report: dict) -> dict:
        script = Paths.PROJECT_ROOT / "bootstrap" / "register_url_scheme.py"
        if not script.exists():
            self.log_info(
                "bootstrap/register_url_scheme.py absent; skipping"
            )
            report["steps"].append({"url_scheme": "no_script"})
            return {"status": "no_script"}

        if self._url_scheme_already_registered():
            self.log_info("unitport:// URL scheme already registered")
            report["steps"].append({"url_scheme": "skipped"})
            return {"status": "skipped"}

        self.log_info("registering unitport:// URL scheme")
        cmd = [sys.executable, str(script)]
        try:
            self._run_streaming(cmd)
        except RuntimeError as exc:
            self.log_warning(
                f"URL scheme registration failed ({exc}); "
                f"OAuth redirects may not work"
            )
            report["steps"].append({"url_scheme": "failed"})
            return {"status": "failed", "error": str(exc)}

        self.log_success("URL scheme registered")
        report["steps"].append({"url_scheme": "registered"})
        return {"status": "registered"}

    # ------------------------------------------------------------------
    # Step 4 -- write a partial install_state. Final write is by
    # PostSetupTask; we just record what the provisioning step learned.
    # ------------------------------------------------------------------
    def _step_provisioning_state_partial(self, report: dict) -> None:
        path = Paths.RUNTIME_DIR / _INSTALL_STATE_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        patch = {
            "provisioning_in_progress": False,
            "provisioning_done_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "torch_cuda": (report.get("torch_cuda") or {}).get("cuda", False),
            "url_scheme": (report.get("url_scheme") or {}).get("status")
                in {"skipped", "registered"},
        }
        try:
            up_data(path, data=patch, merge=True)
        except Exception as exc:  # pragma: no cover -- defensive
            self.log_warning(
                f"failed to update install_state.json: {exc}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _report_unimportable(self, failures: List[ImportFailure]) -> None:
        """Per-package detail in the LoadingScreen log.

        For each failure: distribution name, import name, exception
        type+msg, head of the traceback, and a copy-pasteable pip
        command the user can run to retry that single package.
        """
        self.log_error(
            f"{len(failures)} package(s) still unimportable after pip install:"
        )
        for f in failures:
            self.log_error(
                f"  - {f.pkg} (import {f.import_name}): {f.exc_type}: {f.exc_msg}"
            )
            if f.traceback_head:
                # Indent traceback so it groups with the bullet visually.
                for line in f.traceback_head.splitlines():
                    self.log_error(f"      {line}")
            self.log_error(
                f"      try: pip install --no-cache-dir --force-reinstall {f.pkg}"
            )

    def _url_scheme_already_registered(self) -> bool:
        """Cheap probe; never raises."""
        try:
            if platform.system() == "Windows":
                import winreg  # noqa: WPS433 -- stdlib, Windows-only
                try:
                    with winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER, r"Software\Classes\unitport"
                    ):
                        return True
                except OSError:
                    return False
            if platform.system() == "Linux":
                desktop = (
                    Path(os.path.expanduser("~"))
                    / ".local" / "share" / "applications" / "unitport.desktop"
                )
                return desktop.exists()
        except Exception:
            return False
        return False

    def _run_streaming(self, cmd: List[str]) -> None:
        """Run ``cmd`` and stream stdout (merged stderr) into log_info.

        bufsize=1 + text=True forces line-buffered text mode so progress
        lines reach the LoadingScreen as soon as the child flushes them.
        encoding='utf-8' / errors='replace' tolerates the occasional
        non-UTF-8 byte that pip emits on Windows for some package names.
        Raises ``RuntimeError`` on non-zero exit so caller decides
        whether to bubble or downgrade to a warning.
        """
        self.log_info("$ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None  # PIPE guarantees this
        try:
            for line in proc.stdout:
                self.check_cancelled()
                stripped = line.rstrip()
                if stripped:
                    self.log_info(stripped)
        finally:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"command exited with code {rc}: {cmd[0]}")


class PostSetupTask(Task):
    """Heavy / interactive installs deferred to after the wizard.

    Submitted by ``UnitPortMain`` once both ``ProvisioningTask`` and the
    ``InstallConfigWizard`` have finished. Reads the wizard's
    ``selections`` dict and drives the actions DEMO performed
    synchronously in ``install.bat`` (Steps 5, 7b, 8) plus the wizard's
    own post-actions: brand SDKs, MuJoCo Menagerie sparse-checkout,
    loco-mujoco clone, Isaac Lab (locate / install / cloud-deploy),
    ROS2 (native-detect or Docker bridge build), then writes the final
    ``install_state.json``.

    On a warm start (everything already installed) every step's probe
    short-circuits and the task returns in seconds.

    Currently only stub bodies log the selections. The implementation
    fills in over the migration, but ``UnitPortMain`` already wires
    submission so the orchestration can be tested independently.
    """

    def __init__(self, selections: dict, name: str = "post-setup") -> None:
        super().__init__(name=name)
        self._sel = selections or {}

    def run(self) -> dict:  # type: ignore[override]
        report: dict = {"steps": [], "selections": dict(self._sel)}
        skipped = bool(self._sel.get("skipped", False))
        self.log_info(
            f"starting (skipped={skipped}, "
            f"keys={sorted(self._sel.keys())})"
        )
        # NOTE: real bodies land in subsequent migration steps. For now
        # each step is a stub that logs what it WOULD do, so the
        # orchestration in main.py can be verified end-to-end.
        self._step_sdks(report)
        self._step_menagerie(report)
        self._step_loco_mujoco(report)
        self._step_custom_mods(report)
        self._step_isaac_lab(report)
        self._step_ros2(report)
        self._step_finalize_install_state(report)
        self.log_success("complete")
        return report

    # ------------------------------------------------------------------
    # Step bodies
    # ------------------------------------------------------------------
    def _step_sdks(self, report: dict) -> None:
        """Clone + dep-install brand SDKs.

        - skipped wizard -> install everything in the registry (DEMO default)
        - Finish path with selections -> install only the picked subset
        """
        from application.service.models.sdk_manager import SdkManager

        manager = SdkManager()
        progress_cb = self._sdk_progress_callback()

        try:
            if self._sel.get("skipped"):
                ensured = manager.ensure_registered_sdks(progress=progress_cb)
            else:
                ensured = manager.ensure_selected_sdks(
                    list(self._sel.get("sdks") or []),
                    progress=progress_cb,
                )
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"SDK step failed: {exc}")
            report["steps"].append({"sdks": "failed"})
            return

        report["steps"].append({"sdks": "done", "count": len(ensured)})
        self.log_success(f"SDKs ready ({len(ensured)} project(s))")

    def _step_menagerie(self, report: dict) -> None:
        """Sparse-checkout selected packages or full clone (skipped path)."""
        from application.service.models import menagerie_manager as mm
        on_output = lambda msg: self.log_info(f"[menagerie] {msg}")  # noqa: E731

        if self._sel.get("skipped"):
            # Full shallow clone -- match DEMO ``ensure_mujoco_menagerie``.
            if mm.menagerie_root().exists() and any(mm.menagerie_root().iterdir()):
                self.log_info("menagerie already present")
                report["steps"].append({"menagerie": "skipped"})
                return
            self.log_info("cloning full mujoco_menagerie (skipped path)")
            try:
                mm.bootstrap_with_packages([], on_output=on_output)
            except Exception as exc:  # noqa: BLE001
                self.log_warning(f"menagerie clone failed: {exc}")
                report["steps"].append({"menagerie": "failed"})
                return
            report["steps"].append({"menagerie": "full_clone"})
            return

        folders = list(self._sel.get("menagerie_folders") or [])
        if not folders:
            self.log_info("no menagerie folders selected -- skipping")
            report["steps"].append({"menagerie": "no_selection"})
            return

        try:
            if mm.has_git_clone():
                self.log_info(
                    f"menagerie sparse-checkout already initialised; "
                    f"adding {len(folders)} folder(s)"
                )
                mm.add_packages(folders, on_output=on_output)
            else:
                self.log_info(
                    f"bootstrapping menagerie with {len(folders)} folder(s)"
                )
                mm.bootstrap_with_packages(folders, on_output=on_output)
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"menagerie sparse-checkout failed: {exc}")
            report["steps"].append({"menagerie": "failed"})
            return

        report["steps"].append({"menagerie": "sparse_checkout", "count": len(folders)})
        self.log_success(
            f"menagerie ready ({len(folders)} folder(s) sparse-checked-out)"
        )

    def _step_loco_mujoco(self, report: dict) -> None:
        """Shallow-clone loco-mujoco reference motion library."""
        backend = self._sel.get("backend") or {}
        if not self._sel.get("skipped") and not backend.get("loco_mujoco", True):
            self.log_info("loco-mujoco disabled by user")
            report["steps"].append({"loco_mujoco": "disabled"})
            return

        loco_dir = Paths.CUSTOM_MODS_DIR / "motions" / "loco-mujoco"
        if (loco_dir / ".git").exists():
            self.log_info("loco-mujoco already present")
            report["steps"].append({"loco_mujoco": "skipped"})
            return

        if shutil.which("git") is None:
            self.log_warning(
                "git not on PATH; skipping loco-mujoco "
                "(reference motions will be unavailable)"
            )
            report["steps"].append({"loco_mujoco": "no_git"})
            return

        loco_dir.parent.mkdir(parents=True, exist_ok=True)
        self.log_info(f"cloning loco-mujoco -> {loco_dir}")
        cmd = [
            "git", "clone", "--depth=1", "--progress",
            "https://github.com/robfiras/loco-mujoco.git",
            str(loco_dir),
        ]
        try:
            self._run_streaming(cmd)
        except RuntimeError as exc:
            self.log_warning(f"loco-mujoco clone failed ({exc})")
            report["steps"].append({"loco_mujoco": "failed"})
            return

        report["steps"].append({"loco_mujoco": "cloned"})
        self.log_success("loco-mujoco cloned")

    def _step_custom_mods(self, report: dict) -> None:
        """Shallow-clone reference-motion packs from the built-in catalog.

        The catalog is code-defined in ``custom_mods_manifest.py`` (there
        is no ``installs.txt`` to read). Selection comes from
        ``selections.backend.custom_mods`` (list of entry keys) on the
        normal path; on the wizard-skipped ("legacy default") path the
        ``default_install`` subset is cloned -- this keeps shipped canvas
        templates (e.g. ``Go2_AMP``, which hard-refs ``pack:amp_go2:``
        clips) working out of the box. Each entry is cloned into
        ``Paths.CUSTOM_MODS_DIR / <relative_path>`` (every entry under
        ``motions/``).
        """
        from application.service.installers.custom_mods_manifest import (
            entry_already_installed,
            load_manifest_entries,
        )

        entries = load_manifest_entries()
        by_key = {e.key: e for e in entries}

        if self._sel.get("skipped"):
            selected_keys = [e.key for e in entries if e.default_install]
            self.log_info(
                f"custom_mods: wizard skipped -> default set "
                f"({len(selected_keys)} entr{'y' if len(selected_keys) == 1 else 'ies'})"
            )
        else:
            backend = self._sel.get("backend") or {}
            selected_keys = list(backend.get("custom_mods") or [])

        if not selected_keys:
            self.log_info("custom_mods: no entries selected")
            report["steps"].append({"custom_mods": "none_selected"})
            return

        results: list[dict] = []

        if shutil.which("git") is None:
            self.log_warning(
                "git not on PATH; skipping custom_mods clones "
                "(selected entries will be unavailable)"
            )
            report["steps"].append({"custom_mods": "no_git"})
            return

        for key in selected_keys:
            entry = by_key.get(key)
            if entry is None:
                self.log_warning(
                    f"custom_mods: selection {key!r} is not in the "
                    f"reference-motion catalog; skipping"
                )
                results.append({"key": key, "status": "missing_manifest_entry"})
                continue

            if entry_already_installed(entry):
                self.log_info(f"custom_mods: {entry.key} already installed")
                results.append({"key": entry.key, "status": "already_installed"})
                continue

            target = entry.target_dir
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and any(target.iterdir()):
                self.log_warning(
                    f"custom_mods: target {target} exists and is non-empty "
                    f"but has no git metadata; skipping {entry.key}"
                )
                results.append({"key": entry.key, "status": "target_dirty"})
                continue

            cmd = ["git", "clone", "--depth=1", "--progress", entry.url, str(target)]
            if entry.ref:
                cmd[3:3] = ["--branch", entry.ref]
            self.log_info(f"cloning custom_mod {entry.key} -> {target}")
            try:
                self._run_streaming(cmd)
            except RuntimeError as exc:
                self.log_warning(
                    f"custom_mods: clone of {entry.key} failed ({exc})"
                )
                results.append({"key": entry.key, "status": "failed"})
                continue
            results.append({"key": entry.key, "status": "cloned"})
            self.log_success(f"custom_mod {entry.key} cloned")

        report["steps"].append({"custom_mods": results})

    def _step_isaac_lab(self, report: dict) -> None:
        """Isaac Lab register / install / cloud-deploy — fail loud, never silent.

        Branches and their exit conventions:

        - **disabled** (``isaaclab_enabled=False``): legitimate user opt-out,
          returns normally; PostSetupTask continues with later steps.
        - **locate ok**: returns normally.
        - **install dispatched**: returns normally; the long-running
          IsaacLabInstallTask reports its own success / failure via
          ``AppSignals.isaac_install_*`` and the progress dialog.
        - **ANY other branch** (locate failed, cloud-deploy chosen but
          unimplemented, no-mode-selected wizard state, install dispatch
          rejected by EULA / path guards): :meth:`_fail_isaac_install`
          which ``log_error``s, broadcasts a failure signal so the
          progress dialog opens with the error view, and **raises**.

        The raise is deliberate. PostSetupTask propagates the exception
        to the SDK Task framework, which lands in
        ``UnitPortMain._on_task_finished`` with ``ok=False``; that handler
        sets the fatal flag and freezes the LoadingScreen on the error.
        The user cannot mistake a failed Isaac install for a successful
        one, and the boot does NOT march on through ros2 / finalize as
        if nothing happened.
        """
        backend = self._sel.get("backend") or {}
        if not backend.get("isaaclab_enabled"):
            self.log_info("isaac_lab disabled by user")
            report["steps"].append({"isaac_lab": "disabled"})
            return

        # Fundamental idempotency gate. Whatever the wizard's recorded
        # mode (install / locate / cloud_deploy), if Isaac Lab is already
        # present on disk AND registered with EngineService, this step
        # is a no-op. Re-running the installer when the install is
        # already done is the user-reported bug "Engines\isaac_lab 明明
        # 已经有 isaac 了，但是每次启动程序都会再走一次安装进程": the
        # wizard saved isaaclab_install=True on first launch, the
        # installer succeeded, but nothing flipped setup_state.json back
        # to locate mode, so every subsequent boot re-dispatched the
        # 30 GB install. The precheck below is the root fix — it makes
        # _step_isaac_lab idempotent regardless of which mode setup_state
        # claims, so the same selection dict can be replayed safely on
        # every boot. main.py also flips setup_state on install success
        # (state-machine closure); this gate is the belt that catches
        # the case where the flip never happened (older installs, manual
        # placement of Isaac Lab under Engines/, hand-edited setup_state,
        # crash between install success and main.py's complete handler).
        if self._isaac_lab_already_usable(backend):
            report["steps"].append({"isaac_lab": "already_registered"})
            return

        if backend.get("isaaclab_locate"):
            from application.service.models.sdk_manager import register_isaaclab_path
            path = str(backend.get("isaaclab_path", "")).strip()
            if not path:
                self._fail_isaac_install(
                    "Isaac Lab locate requested but isaaclab_path is "
                    "empty in setup_state.json. Re-run the wizard and "
                    "either type/browse the Isaac Lab root, or pick a "
                    "different mode.",
                    report,
                )
            # Refuse to register a skeleton-valid but venv-broken install.
            # "Locate" semantically asserts "this install is ready to use";
            # silently registering one whose .venv has the wrong torch
            # flavor or missing RL framework would surface as the same
            # _C.pyd PyInit__C ACCESS_VIOLATION at first training. Fail
            # loud with the precise reason and tell the user what to do.
            healthy, reason = self._isaac_lab_venv_healthy(Path(path).expanduser())
            if not healthy:
                self._fail_isaac_install(
                    f"Isaac Lab at {path!r} is registered as locatable but "
                    f"its venv is unhealthy: {reason}. Re-run the wizard "
                    f"and choose 'Install Isaac Lab' (the installer will "
                    f"repair the venv — ensure_cuda_torch + "
                    f"isaaclab_rl[all]), or fix the venv manually via "
                    f"`{path}\\isaaclab.bat -i` before retrying locate.",
                    report,
                )
            ok = register_isaaclab_path(
                path, progress=self._sdk_progress_callback(),
            )
            if not ok:
                self._fail_isaac_install(
                    f"Isaac Lab locate failed: {path!r} is not a valid "
                    f"installation (missing isaaclab.sh / isaaclab.bat "
                    f"and/or source/ directory).",
                    report,
                )
            report["steps"].append({"isaac_lab": "located"})
            return

        if backend.get("isaaclab_install"):
            # _dispatch_isaac_install raises on its own guards (path
            # empty / EULA missing / EULA stale); successful dispatch
            # returns and PostSetupTask finishes — the actual long-running
            # install reports its own success/failure via signals.
            self._dispatch_isaac_install(backend, report)
            return

        if backend.get("isaaclab_cloud_deploy"):
            self._fail_isaac_install(
                "Isaac Lab cloud-deploy is not implemented in this "
                "build (deferred post-1.0). Your SSH config has been "
                "recorded in setup_state.json for future use, but no "
                "remote install will run. Re-run the wizard and choose "
                "'Install Isaac Lab' or 'Locate existing installation'.",
                report,
            )

        # User enabled Isaac Lab but didn't pick install / locate / cloud.
        # Radio buttons should be mutually exclusive in the wizard — if
        # we get here, either setup_state.json was hand-edited or the
        # wizard's get_config built a malformed dict. Either way it's
        # a hard failure, not a "use defaults" condition.
        self._fail_isaac_install(
            "Isaac Lab enabled but no install mode (install / locate / "
            "cloud_deploy) was selected. Setup state is corrupted; "
            "delete the [App].setup_state pointer or re-run install.bat "
            "to redo the wizard from scratch.",
            report,
        )

    def _fail_isaac_install(self, message: str, report: dict) -> None:
        """Log error, broadcast failure signal, ABORT PostSetupTask.

        Three responsibilities, all loud:

        1. ``log_error`` with the searchable ``[ISAAC INSTALL FAILED]``
           sentinel so the user can grep CmdLogWidget / file log.
        2. Emit ``isaac_install_complete(False, msg)`` so the
           ``IsaacInstallProgressDialog`` auto-opens with the error
           view (main.py's race-recovery handler covers the case where
           no phase signal preceded this).
        3. ``raise RuntimeError`` — this is what NEVER returns. The
           previous version's ``return`` after logging let PostSetupTask
           continue to ``_step_ros2`` etc., making the failure look
           like a non-event in the boot report. Re-raising propagates
           to ``UnitPortMain._on_task_finished`` which sets the boot
           fatal flag and stops the rest of the pipeline.

        Does not return. The annotation NoReturn would be more honest
        but we don't import typing.NoReturn elsewhere in this module.
        """
        self.log_error(f"=== [ISAAC INSTALL FAILED] {message} ===")
        try:
            from application.service.signals import get_app_signals
            get_app_signals().isaac_install_complete.emit(False, message)
        except Exception as exc:  # noqa: BLE001
            # Never let a UX-only side-effect mask the real failure —
            # surface this on log_error too, then re-raise the original.
            self.log_error(
                f"(also: could not broadcast isaac_install_complete signal: "
                f"{exc!r})"
            )
        report["steps"].append({"isaac_lab": "failed", "reason": message})
        raise RuntimeError(f"Isaac Lab install aborted: {message}")

    def _dispatch_isaac_install(self, backend: dict, report: dict) -> None:
        """Submit an :class:`IsaacLabInstallTask` and return without waiting.

        Every guard before submission RAISES via
        :meth:`_fail_isaac_install` on failure — never returns silently.
        See its docstring for why we abort PostSetupTask on guard failure
        instead of just logging a warning and marching on.

        On successful submission, returns; the long-running install
        task drives its own success/failure via ``AppSignals.isaac_install_*``.
        """
        install_path = str(backend.get("isaaclab_path", "")).strip()
        if not install_path:
            self._fail_isaac_install(
                "Isaac Lab fresh-install requested but isaaclab_path is "
                "empty in setup_state.json. Re-run the wizard and either "
                "type / browse an install directory, or pick a different "
                "mode.",
                report,
            )

        # EULA gate: only ``setup_state.json.selections.backend.eula_accepted_ids``
        # is the authoritative source. That list comes from the wizard's
        # EulaDialog where the user explicitly ticked + clicked Accept,
        # and is the bit of state PostSetupTask actually needs to verify.
        #
        # We deliberately do NOT also require ``<USER_CONFIG_DIR>/eula_acceptance.json``
        # to contain matching records. That persistent store is a UX
        # cache so the wizard can skip re-prompting on subsequent runs;
        # it is NOT a security boundary (anyone who can edit
        # setup_state.json can edit the persistent store too). Gating on
        # both produced a brittle flow where deleting the cache invalidated
        # an otherwise-valid wizard acceptance.
        from application.service.installers import eula as _eula
        claimed_ids = list(backend.get("eula_accepted_ids") or [])
        required_ids = {spec.eula_id for spec in _eula.list_required_eulas()}
        if not required_ids.issubset(set(claimed_ids)):
            missing = sorted(required_ids - set(claimed_ids))
            self._fail_isaac_install(
                f"Isaac Lab fresh-install requested without complete EULA "
                f"acceptance. Missing from setup_state.json: {missing}. "
                f"Re-run the wizard, tick every tab in the EULA dialog, "
                f"and click Accept.",
                report,
            )
        # Best-effort: top up the persistent store from the wizard's
        # claim so future wizard runs skip the modal. Failures here are
        # purely a future-UX issue (user gets re-prompted next time);
        # they MUST NOT block the install.
        try:
            for spec in _eula.list_required_eulas():
                if spec.eula_id in set(claimed_ids):
                    _eula.record_acceptance(spec.eula_id, spec.version)
        except Exception as exc:  # noqa: BLE001
            self.log_warning(
                f"could not back-fill <USER_CONFIG_DIR>/eula_acceptance.json "
                f"from setup_state claim (non-fatal): {exc!r}"
            )

        # Fire-and-forget: submit to TasksManager and return.
        # Cancellation / progress flow through the SDK Task framework
        # + AppSignals; PostSetupTask is unblocked immediately. ANY
        # failure inside the install task surfaces via the progress
        # dialog + ``isaac_install_complete(False, …)``; it does NOT
        # propagate back here (this method has already returned by then).
        from application.service.installers import is_install_path_internal
        from application.tools.isaac_lab_install_task import IsaacLabInstallTask
        from unitport_sdk import get_tasks_manager

        mode = "internal" if is_install_path_internal(install_path) else "external"
        task = IsaacLabInstallTask(
            install_dir=Path(install_path),
            eula_ids=claimed_ids,
        )
        task_id = get_tasks_manager().submit(task)
        self.log_info(
            f"isaac install dispatched as task {task_id} "
            f"(install_dir={install_path}, mode={mode})"
        )
        report["steps"].append({
            "isaac_lab": "install_dispatched",
            "task_id": task_id,
            "install_dir": install_path,
            "mode": mode,
        })

    def _isaac_lab_venv_healthy(self, root: Path) -> tuple[bool, str]:
        """Probe the Isaac Lab venv for runtime-critical state.

        Skeleton markers (isaaclab.bat + source/) say nothing about
        whether the venv inside ``<root>/.venv`` actually has a CUDA
        torch and the RL frameworks the launcher requires. If those
        are missing, training fails at first launch with either:

          * ``_C.pyd!PyInit__C`` ACCESS_VIOLATION (exit 0xC0000005)
            — torch is CPU-only, Isaac Sim's native ext can't bind
            CUDA symbols. Root cause for the post-``reset.bat`` case:
            reset wipes ``runtime/env/`` but leaves
            ``<USER_CONFIG_DIR>/engines/isaac_lab.json`` and the
            Engine venv intact, so ``_isaac_lab_already_usable``
            would short-circuit on the stale-but-skeleton-valid
            registration and the installer's ``_ensure_cuda_torch``
            step would never run.

          * "Isaac venv is missing rsl_rl" — older installer that
            pip-installed isaaclab_rl bare (no [all] extras).

        Returns (ok, reason). When ok=False the reason is logged so
        the user / log reader sees *why* we're re-running the
        installer rather than just "looks stale, retrying" without
        context. External / conda-managed installs that don't put the
        venv at ``<root>/.venv`` are returned as healthy (we have no
        way to probe them and we don't own them anyway).
        """
        from sys import platform as _platform
        venv_dir = root / ".venv"
        if not venv_dir.is_dir():
            return True, ""
        py = (
            venv_dir / "Scripts" / "python.exe"
            if _platform == "win32"
            else venv_dir / "bin" / "python"
        )
        if not py.exists():
            return False, f".venv interpreter missing at {py}"

        import subprocess as _sub
        # Probe torch flavor — CUDA build sets ``torch.version.cuda``
        # to a non-empty string ("12.8" etc.); CPU build leaves it as
        # None. ImportError (torch missing) and ``cuda is None`` are
        # the same fault class — both reach the launcher as the same
        # PyInit__C crash. Also probe tensordict's version line — the
        # 0.12.x wheels are built against torch 2.8 ABI and crash with
        # the same ACCESS_VIOLATION at ``tensordict._C`` load inside an
        # Isaac Sim Kit subprocess even though standalone Python imports
        # them fine. ``isaac_lab_installer._ensure_cuda_torch`` pins it
        # to 0.11.x; surface the same constraint here so a stray ``pip
        # install`` that yanks tensordict forward triggers a repair.
        probe = (
            "import sys\n"
            "try:\n"
            "    import torch\n"
            "    cuda = torch.version.cuda\n"
            "    print('TORCH', torch.__version__, '|', cuda or 'cpu')\n"
            "except Exception as e:\n"
            "    print('TORCH_FAIL', type(e).__name__, e); sys.exit(1)\n"
            "import importlib.util as u\n"
            "import importlib.metadata as m\n"
            "for n in ('rsl_rl',):\n"
            "    print('PKG', n, 'OK' if u.find_spec(n) else 'MISSING')\n"
            "try:\n"
            "    print('TD', m.version('tensordict'))\n"
            "except m.PackageNotFoundError:\n"
            "    print('TD', 'MISSING')\n"
        )
        try:
            res = _sub.run(
                [str(py), "-c", probe],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, _sub.TimeoutExpired) as exc:
            return False, f"venv probe failed to spawn: {exc!r}"
        if res.returncode != 0:
            return False, (
                f"venv probe non-zero (rc={res.returncode}): "
                f"{(res.stderr or res.stdout).strip()[:200]}"
            )
        out = res.stdout or ""
        torch_line = next(
            (ln for ln in out.splitlines() if ln.startswith("TORCH ")),
            "",
        )
        if "| cpu" in torch_line or "| None" in torch_line:
            return False, (
                f"venv has CPU-only torch ({torch_line.strip()}) — "
                f"Isaac Sim 5.x needs CUDA torch"
            )
        missing_pkgs = [
            ln.split()[1]
            for ln in out.splitlines()
            if ln.startswith("PKG ") and ln.endswith("MISSING")
        ]
        if missing_pkgs:
            return False, f"venv missing packages: {', '.join(missing_pkgs)}"
        td_line = next(
            (ln for ln in out.splitlines() if ln.startswith("TD ")),
            "",
        )
        td_ver = td_line.split(" ", 1)[1].strip() if " " in td_line else ""
        if td_ver == "MISSING":
            return False, "venv missing package: tensordict"
        # Reject the torch-2.8-ABI line (tensordict >= 0.12) when paired
        # with our pinned torch 2.7 — wheel ABI mismatch crashes Kit at
        # tensordict._C load with ACCESS_VIOLATION.
        if td_ver and not td_ver.startswith("0.11."):
            return False, (
                f"tensordict=={td_ver} is incompatible with torch 2.7 "
                f"under Isaac Sim Kit (need 0.11.x)"
            )
        return True, ""

    def _isaac_lab_already_usable(self, backend: dict) -> bool:
        """Probe whether Isaac Lab is already installed AND registered.

        Returns True iff after this method runs:

        - ``EngineService.get_local("isaac_lab")`` reports
          ``registered=True`` with a ``root`` that on disk still has
          the markers (``isaaclab.sh`` or ``isaaclab.bat`` + ``source/``
          directory); OR
        - the wizard-declared ``isaaclab_path`` validates as an Isaac
          Lab root and is freshly registered through ``EngineService``
          on this call; OR
        - the machine-level ``sdk/install_state.json`` records a prior
          successful install whose ``root`` validates and is registered
          on this call.

        This is the single source of truth for "Isaac Lab is ready to
        use, do not run any installer/locate work." It treats the
        on-disk install + ``EngineService.local`` registration as the
        physical fact and ignores what ``setup_state.json`` claims —
        the wizard's selection is intent, not state.

        Re-registration uses ``source="boot_repair"`` so audit tools
        can distinguish first-time wizard-driven registration from
        idempotent re-registration during boot.
        """
        # Lazy imports keep this method off the cold path when isaaclab
        # is disabled by the user — those imports pull in user_workspace
        # which reads machine-level state, and EngineService which loads
        # engines/*.json. No reason to pay that on disabled boots.
        from application.service.engines import get_engine_service
        from application.service.user_workspace import machine_config_dir
        from unitport_sdk import read_data

        def _markers_ok(root: Path) -> bool:
            try:
                if not root.is_dir():
                    return False
                has_launcher = (
                    (root / "isaaclab.sh").exists()
                    or (root / "isaaclab.bat").exists()
                )
                has_source = (root / "source").is_dir()
                return has_launcher and has_source
            except OSError:
                return False

        _venv_healthy = self._isaac_lab_venv_healthy

        svc = get_engine_service()
        local = svc.get_local("isaac_lab")

        # Signal 1 — EngineService says we registered it. Trust only if
        # the markers still exist AND the venv passes the runtime-health
        # probe. The skeleton check (isaaclab.bat + source/) survives a
        # reset.bat but says nothing about whether the venv inside has
        # the right torch flavor / RL frameworks — see `_venv_healthy`
        # for why this matters.
        if local.get("registered") and local.get("root"):
            root = Path(str(local["root"]))
            if _markers_ok(root):
                healthy, reason = _venv_healthy(root)
                if healthy:
                    self.log_info(
                        f"isaac_lab already registered at {root}; "
                        f"skipping locate/install step"
                    )
                    return True
                self.log_warning(
                    f"isaac_lab registered at {root} but venv is "
                    f"unhealthy ({reason}); re-running install_lab to "
                    f"repair (ensure_cuda_torch + isaaclab_rl[all])"
                )
            else:
                self.log_warning(
                    f"isaac_lab engines state claims root={root} but the "
                    f"markers (isaaclab.sh|isaaclab.bat + source/) are "
                    f"missing; treating registration as stale and probing "
                    f"other locations"
                )

        # Signal 2 — wizard recorded a path; if Isaac Lab is in fact
        # installed there (manual placement under Engines/, or the
        # installer succeeded on a prior boot but no one updated the
        # engines state), repair the EngineService registration in
        # place and short-circuit. This is what catches the canonical
        # user-reported case: Isaac Lab sitting at the wizard's chosen
        # path with all markers, EngineService still empty.
        declared = str(backend.get("isaaclab_path", "")).strip()
        if declared:
            root = Path(declared).expanduser()
            if _markers_ok(root):
                healthy, reason = _venv_healthy(root)
                if not healthy:
                    self.log_warning(
                        f"isaac_lab skeleton at {root} (wizard path) but "
                        f"venv unhealthy ({reason}); falling through to "
                        f"install_lab so it can be repaired"
                    )
                else:
                    resolved = str(root.resolve())
                    if svc.register_isaac_local(resolved, source="boot_repair"):
                        self.log_info(
                            f"isaac_lab found at {root} (wizard path); "
                            f"re-registered without re-installing"
                        )
                        return True

        # Signal 3 — install_state.json records a prior successful
        # install. Same repair flow as signal 2 but the path source is
        # the install record, which survives wizard state corruption.
        try:
            state_path = machine_config_dir() / "sdk" / "install_state.json"
            if state_path.exists():
                data = read_data(state_path)
                if isinstance(data, dict):
                    isaac_block = data.get("isaac_lab") or {}
                    recorded_root = str(isaac_block.get("root", "")).strip()
                    if recorded_root:
                        root = Path(recorded_root)
                        if _markers_ok(root):
                            healthy, reason = _venv_healthy(root)
                            if not healthy:
                                self.log_warning(
                                    f"isaac_lab skeleton at {root} (from "
                                    f"install_state.json) but venv unhealthy "
                                    f"({reason}); falling through to "
                                    f"install_lab so it can be repaired"
                                )
                            else:
                                resolved = str(root.resolve())
                                if svc.register_isaac_local(
                                    resolved, source="boot_repair"
                                ):
                                    self.log_info(
                                        f"isaac_lab found at {root} (from "
                                        f"sdk/install_state.json); re-registered "
                                        f"without re-installing"
                                    )
                                    return True
        except Exception as exc:  # noqa: BLE001
            # Probe is best-effort: a corrupt install_state.json must
            # not block the normal locate/install flow. Log and fall
            # through so the wizard's recorded mode still runs.
            self.log_warning(
                f"isaac_lab install_state.json probe failed (non-fatal): "
                f"{exc!r}"
            )

        return False

    def _step_ros2(self, report: dict) -> None:
        """ROS2 native-detect + Docker bridge build.

        Native-detect via ``bootstrap/detect_ros2.py`` is cheap and safe
        to run in this build. Docker build (``bootstrap/install_ros2.bat``
        / ``.sh``) is left as a deferred step -- the user's choice is
        recorded in setup_state.json so the eventual in-app trigger has
        the data it needs.
        """
        backend = self._sel.get("backend") or {}
        if not self._sel.get("skipped") and not backend.get("ros2_enabled", True):
            self.log_info("ros2 disabled by user")
            report["steps"].append({"ros2": "disabled"})
            return

        detect_script = Paths.PROJECT_ROOT / "bootstrap" / "detect_ros2.py"
        if not detect_script.exists():
            self.log_warning(
                f"bootstrap/detect_ros2.py absent at {detect_script}; "
                f"skipping native ROS2 probe"
            )
            report["steps"].append({"ros2": "no_detect_script"})
            return

        self.log_info("probing for native ROS2")
        try:
            completed = subprocess.run(
                [sys.executable, str(detect_script)],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"ros2 native-detect crashed: {exc}")
            report["steps"].append({"ros2": "detect_crashed"})
            return

        try:
            import json as _json
            payload = _json.loads((completed.stdout or "").strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            payload = {"found": False}

        if payload.get("found"):
            self.log_success(
                f"native ROS2 detected: distro={payload.get('distro')!r} "
                f"root={payload.get('ros_root')!r}"
            )
            report["steps"].append({"ros2": "native", "info": payload})
            return

        self.log_warning(
            "no native ROS2 detected; Docker bridge build is "
            "available via bootstrap/install_ros2.bat (deferred to follow-up build)"
        )
        report["steps"].append({"ros2": "docker_deferred"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sdk_progress_callback(self):
        """Return a (level, message) callback that pipes into ``self.log_*``.

        SdkManager / loco-mujoco helpers expect the DEMO-style callback
        signature ``Callable[[level: str, message: str], None]``. We
        adapt it here so progress lines flow into the LoadingScreen
        log via this Task's name-tagged sinks.
        """
        sinks = {
            "info": self.log_info,
            "success": self.log_success,
            "warning": self.log_warning,
            "error": self.log_error,
            "debug": self.log_info,
        }

        def _cb(level: str, message: str) -> None:
            (sinks.get(level, self.log_info))(message)

        return _cb

    def _run_streaming(self, cmd: List[str]) -> None:
        """Same shape as :class:`ProvisioningTask._run_streaming`.

        Duplicated rather than shared as a module helper because each
        Task instance owns its own ``log_*`` / ``check_cancelled`` bound
        methods, and a free function would have to take them as args
        anyway -- two-line duplication is cheaper than the indirection.
        """
        self.log_info("$ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self.check_cancelled()
                stripped = line.rstrip()
                if stripped:
                    self.log_info(stripped)
        finally:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"command exited with code {rc}: {cmd[0]}")

    def _step_finalize_install_state(self, report: dict) -> None:
        path = Paths.RUNTIME_DIR / _INSTALL_STATE_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        patch = {
            "provisioning_pending": False,
            "post_setup_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "post_setup_steps": report["steps"],
        }
        try:
            up_data(path, data=patch, merge=True)
            self.log_info(f"install_state.json updated at {path}")
        except Exception as exc:  # pragma: no cover -- defensive
            self.log_warning(
                f"failed to update install_state.json: {exc}"
            )


__all__ = ["FuncTask", "ProvisioningTask", "PostSetupTask"]
