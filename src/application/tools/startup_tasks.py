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

        missing_names = [f.pkg for f in missing]
        self.log_warning(
            f"{len(missing)} package(s) missing: {missing_names}; "
            f"running pip install -r requirements.txt"
        )

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "-r",
            str(req_path),
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

        loco_dir = Paths.PROJECT_ROOT / "custom_mods" / "motions" / "loco-mujoco"
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

    def _step_isaac_lab(self, report: dict) -> None:
        """Isaac Lab register / install / cloud-deploy.

        - locate-existing -> :class:`SdkManager.register_isaaclab_path`
          (validates + writes to ``EngineService``)
        - fresh install / cloud-deploy -> implementations land in a
          follow-up turn (Task #28 in the migration plan). For now we
          log_warning and report the deferral rather than raising,
          since the user's choice is already persisted in setup_state.json.
        """
        backend = self._sel.get("backend") or {}
        if not backend.get("isaaclab_enabled"):
            self.log_info("isaac_lab disabled by user")
            report["steps"].append({"isaac_lab": "disabled"})
            return

        if backend.get("isaaclab_locate"):
            from application.service.models.sdk_manager import register_isaaclab_path
            ok = register_isaaclab_path(
                str(backend.get("isaaclab_path", "")).strip(),
                progress=self._sdk_progress_callback(),
            )
            report["steps"].append({"isaac_lab": "located" if ok else "locate_failed"})
            return

        if backend.get("isaaclab_install"):
            self.log_warning(
                "isaac_lab fresh-install path not yet wired in this build; "
                f"the chosen install_dir={backend.get('isaaclab_path')!r} is recorded in "
                f"setup_state.json -- run bootstrap/install_isaac_lab.bat (or its eventual "
                f"in-app trigger) to perform the ~30 GB install."
            )
            report["steps"].append({"isaac_lab": "install_deferred"})
            return

        if backend.get("isaaclab_cloud_deploy"):
            self.log_warning(
                "isaac_lab cloud-deploy path not yet wired in this build; "
                "the SSH config is recorded in setup_state.json -- run the cloud "
                "deployer manually to provision the remote server."
            )
            report["steps"].append({"isaac_lab": "cloud_deferred"})
            return

        self.log_info("isaac_lab enabled but no mode selected")
        report["steps"].append({"isaac_lab": "no_mode"})

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
