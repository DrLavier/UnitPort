"""Core in-app installer for Isaac Lab + Isaac Sim.

Pipeline (pip-based; see ``_constants.py`` for the pinned versions —
Isaac Sim 5.1.0.0 + Isaac Lab v0.54.3):

  preflight → install_sim → clone → install_lab → register → done

Each stage emits :pyattr:`AppSignals.isaac_install_phase` on entry and
pumps :pyattr:`AppSignals.isaac_install_progress` while it runs. The
final terminal signal is :pyattr:`AppSignals.isaac_install_complete`.

Venv policy — **always dedicated, never shared.**
``pip install isaacsim[all,extscache]==5.1.0.0`` ships hard pins on
many packages the UnitPort project also depends on (click, numpy,
pyyaml, charset-normalizer, …). When pip resolves those into a venv
that already contains the project's own pins, it silently uninstalls
the project versions and replaces them with isaacsim's. By the time
exit codes show anything, ``.venv311`` is corrupted and the main app
is broken. The only safe answer is: never let pip touch ``.venv311``.

The installer therefore unconditionally creates a fresh venv at
``<install_dir>/.venv/`` (``.venv`` so :func:`_find_isaac_python` in
``registers/backends.py`` discovers it automatically — same convention
the legacy "locate" path uses) and installs isaacsim + every Isaac Lab
sub-package as editable into THAT venv. ``.venv311`` is never modified.
The :class:`InstallPlan.mode` field is preserved for caller compatibility
but ignored — every install is effectively external. ``mode`` in the
final :class:`InstallReport` always reads ``"dedicated"``.

Cancellation: caller-supplied ``check_cancelled`` is polled at every
stage boundary plus inside pip / git stdout streaming loops. On
cancel we raise :class:`InstallerCancelled` and leave the partial-state
marker on disk for a future "Retry" to skip already-finished stages.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from unitport_sdk import (
    Paths,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
    up_data,
)

from application.service.engines.service import get_engine_service
from application.service.signals import get_app_signals

from ._constants import (
    EULA_TEXTS,
    INSTALLER_ACCEPT_ENV,
    INSTALLER_AUTO_YES_STDIN,
    ISAAC_LAB_RELEASE,
    ISAAC_SIM_RELEASE,
    PREFLIGHT_FREE_BYTES_MIN,
    PREFLIGHT_WIN_PATH_LEN_MAX,
)
from ._paths import is_install_path_internal


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class InstallerError(RuntimeError):
    """Base for hard failures — converts to ``complete(False, …)`` in the Task wrapper."""


class InstallerPreflightError(InstallerError):
    """Disk space / path length / EULA missing — caught early."""


class InstallerNetworkError(InstallerError):
    """pip / git network failure (DNS, TLS, HTTP non-200)."""


class InstallerCloneError(InstallerError):
    """``git clone`` failed or the Isaac Lab markers are missing afterwards."""


class InstallerSubprocessError(InstallerError):
    """``isaaclab.{bat,sh} -i`` or ``pip install`` returned non-zero."""


class InstallerCancelled(InstallerError):
    """Wrapping Task's ``check_cancelled`` signalled stop."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class InstallPlan:
    install_dir: Path
    mode: str    # "internal" | "external"
    eula_ids: List[str] = field(default_factory=list)
    check_cancelled: Optional[Callable[[], None]] = None


@dataclass
class InstallReport:
    install_dir: Path
    mode: str
    isaac_sim_version: str
    isaac_lab_tag: str
    venv_python: str        # absolute path to the python that owns isaacsim
    installed_at: str


_INSTALL_STATE_REL = Path("sdk") / "install_state.json"

# Per-stage progress envelopes. ``install_sim`` is the longest in
# wall-clock (several GB of wheels); we widen its band accordingly.
_PHASE_RANGES: dict[str, tuple[float, float]] = {
    "preflight":    (0.00, 0.02),
    "install_sim":  (0.02, 0.65),
    "clone":        (0.65, 0.70),
    "install_lab":  (0.70, 0.95),
    "register":     (0.95, 1.00),
}


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


class IsaacLabInstaller:
    """Synchronous, signal-emitting Isaac Lab installer.

    Construct once per attempt; not reusable after :meth:`execute`.
    """

    def __init__(self, plan: InstallPlan) -> None:
        self._plan = plan
        self._signals = get_app_signals()
        self._partial = plan.install_dir / ".unitport.partial"
        # Tracked by ``_emit_phase``; consumed by ``_run_subprocess`` to
        # decide which slice of the global progress band to wiggle in
        # while a long subprocess streams stdout.
        self._current_phase: str = "preflight"

    # ------------------------------------------------------------------
    def execute(self) -> InstallReport:
        log_info(
            "=== [ISAAC INSTALL START] "
            f"sim={ISAAC_SIM_RELEASE.version_label} "
            f"lab={ISAAC_LAB_RELEASE.branch_or_tag} "
            f"mode={self._plan.mode} dir={self._plan.install_dir} ==="
        )
        try:
            self._stage_preflight()
            self._stage_install_sim()
            self._stage_clone()
            self._stage_install_lab()
            report = self._stage_register()
            self._emit_phase("done", "Installation complete.")
            self._emit_progress(1.0, "Done")
            self._signals.isaac_install_complete.emit(
                True, f"Installed at {self._plan.install_dir}"
            )
            log_success(
                f"=== [ISAAC INSTALL SUCCESS] {self._plan.install_dir} ==="
            )
            return report
        except InstallerCancelled:
            log_error("=== [ISAAC INSTALL FAILED] cancelled by user ===")
            self._signals.isaac_install_complete.emit(False, "cancelled")
            raise
        except InstallerError as exc:
            # Sentinel banner ABOVE the structured error so the user can
            # scroll up in CmdLogWidget and search for the exact tag
            # without wading through pip / git / isaaclab.bat stdout
            # (which is now log_debug-filtered, but file log still has it).
            log_error(f"=== [ISAAC INSTALL FAILED] {type(exc).__name__}: {exc} ===")
            log_error(
                "Diagnostic hints: check Paths.LOGS_DIR for the full "
                "pip/git output (log_debug-routed); verify EULA acceptance "
                "in <USER_CONFIG_DIR>/eula_acceptance.json; confirm the install "
                "drive has ≥20 GB free; for pip dependency clashes the "
                "installer auto-falls-back to external venv mode."
            )
            self._signals.isaac_install_phase.emit("error", str(exc))
            self._signals.isaac_install_complete.emit(False, str(exc))
            raise
        except Exception as exc:  # noqa: BLE001
            log_error(
                f"=== [ISAAC INSTALL FAILED] unexpected {type(exc).__name__}: {exc!r} ==="
            )
            self._signals.isaac_install_phase.emit("error", repr(exc))
            self._signals.isaac_install_complete.emit(False, repr(exc))
            raise InstallerError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Stage 1 — Preflight
    # ------------------------------------------------------------------
    def _stage_preflight(self) -> None:
        self._emit_phase("preflight", "Verifying environment...")

        # EULA gate: caller (PostSetupTask) is responsible for verifying
        # against ``setup_state.json.selections.backend.eula_accepted_ids``.
        # Here we just re-verify the eula_ids the caller passed — it's
        # cheap insurance against a coding mistake at the call site.
        # We deliberately do NOT read ``<USER_CONFIG_DIR>/eula_acceptance.json``
        # again; that file is a wizard-side cache, not a security boundary.
        required_ids = {spec.eula_id for spec in EULA_TEXTS}
        claimed = set(self._plan.eula_ids or [])
        if not required_ids.issubset(claimed):
            raise InstallerPreflightError(
                f"Caller passed an incomplete EULA acceptance set. "
                f"Missing from claim: {sorted(required_ids - claimed)}"
            )

        # Disk space against the target's anchor.
        target = self._plan.install_dir
        anchor = Path(target.anchor) if target.anchor else Path("/")
        try:
            probe_path = str(anchor) if anchor.exists() else str(target.parent)
            usage = shutil.disk_usage(probe_path)
        except OSError as exc:
            raise InstallerPreflightError(
                f"Cannot determine free disk space for {anchor}: {exc}"
            ) from exc
        if usage.free < PREFLIGHT_FREE_BYTES_MIN:
            need_gb = PREFLIGHT_FREE_BYTES_MIN / (1024 ** 3)
            have_gb = usage.free / (1024 ** 3)
            raise InstallerPreflightError(
                f"Insufficient disk space at {anchor}: need ≥{need_gb:.0f} GB, "
                f"have {have_gb:.1f} GB."
            )

        # Path length (Windows soft check).
        if os.name == "nt" and len(str(target.resolve(strict=False))) > PREFLIGHT_WIN_PATH_LEN_MAX:
            log_warning(
                f"[isaac-install] install path is "
                f"{len(str(target))} chars long — enable Windows "
                f"LongPathsEnabled registry key if pip install fails."
            )

        # Make sure the install root exists. Different from a zip-based
        # pipeline (which only created dirs at extract time) — pip+git
        # both need the target to be already-creatable.
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InstallerPreflightError(
                f"Cannot create install directory {target}: {exc}"
            ) from exc

        # WHY KEPT: provision pip scratch dirs on the install drive so a
        # small system %TEMP% does not abort isaacsim's multi-GB wheel
        # downloads halfway. _acceptance_env() plumbs these into
        # TMP/TEMP/TMPDIR/PIP_CACHE_DIR for every pip subprocess. Without
        # this redirect, pip's cachecontrol streams downloads through
        # tempfile (default %TEMP% on Windows = C:) and isaacsim's 3.4 GB
        # extscache wheel raises OSError(28) when C: runs out.
        for sub in (self._pip_tmp_dir(), self._pip_cache_dir()):
            try:
                sub.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise InstallerPreflightError(
                    f"Cannot create installer scratch directory {sub}: {exc}"
                ) from exc
        log_info(
            f"[isaac-install] pip scratch redirected onto install drive: "
            f"TMP={self._pip_tmp_dir()}  PIP_CACHE_DIR={self._pip_cache_dir()}"
        )

        # Re-judge internal vs external against the actual path. The
        # wizard's pre-judgement can drift between submission and run
        # (project moved, install path edited in setup_state.json).
        actual_mode = "internal" if is_install_path_internal(target) else "external"
        if actual_mode != self._plan.mode:
            log_warning(
                f"[isaac-install] caller declared mode={self._plan.mode!r} "
                f"but current path judgement is {actual_mode!r}; using "
                f"current judgement"
            )
            self._plan.mode = actual_mode

        if self._partial.exists():
            log_info(
                f"[isaac-install] detected partial install marker at "
                f"{self._partial}; resuming where possible"
            )

        self._check_cancelled()
        self._emit_progress(_PHASE_RANGES["preflight"][1], "Preflight OK")

    # ------------------------------------------------------------------
    # Stage 2 — pip install isaacsim (always into the dedicated venv)
    # ------------------------------------------------------------------
    def _stage_install_sim(self) -> None:
        self._emit_phase(
            "install_sim",
            f"Installing Isaac Sim {ISAAC_SIM_RELEASE.version_label} via pip...",
        )

        # Always use a dedicated venv at <install_dir>/.venv/. Internal
        # mode (sharing .venv311) was attempted in earlier versions and
        # is structurally incompatible with isaacsim's hard pins — pip
        # silently uninstalls our project deps to satisfy them. Don't
        # reinvent the wheel: Python's venv mechanism already gives us
        # full isolation, use it.
        self._plan.mode = "dedicated"
        target_python = self._ensure_dedicated_venv()

        # ``--upgrade`` so a partial-state retry picks up the latest
        # wheels; ``--extra-index-url`` so isaacsim is found in NVIDIA's
        # index while transitive deps still resolve against PyPI. EULA
        # env so the Omniverse Kit kernel that runs during wheel
        # post-install hooks doesn't try to prompt.
        env = self._acceptance_env()
        cmd = [
            str(target_python), "-m", "pip", "install", "--upgrade",
            "--disable-pip-version-check",
            "--extra-index-url", ISAAC_SIM_RELEASE.pypi_extra_index,
            ISAAC_SIM_RELEASE.pip_spec,
        ]
        self._run_subprocess(cmd, "pip install isaacsim", env=env)

        # Quick smoke-check the install. The Omniverse Kit kernel kicks
        # in during ``import isaacsim`` and will prompt for the NVOLA
        # EULA if it doesn't see ``OMNI_KIT_ACCEPT_EULA=YES``; we also
        # pipe ``Yes\n`` x5 into stdin as a fallback for kit versions
        # that ignore the env var (without it, kernel reads EOF and
        # aborts with "Unable to bootstrap inner kit kernel: EOF when
        # reading a line").
        #
        # NOTE: Isaac Sim 5.1 does NOT expose a top-level ``__version__``
        # attribute on the ``isaacsim`` package — we read the pip-side
        # version via ``importlib.metadata`` instead. The import itself
        # is what we actually need to validate (it triggers the Kit
        # kernel bootstrap that does the real wheel-vs-runtime sanity
        # check).
        verify_cmd = [
            str(target_python), "-c",
            (
                "import isaacsim, importlib.metadata as m; "
                "print('isaacsim', m.version('isaacsim'))"
            ),
        ]
        try:
            self._run_subprocess(
                verify_cmd,
                "verify isaacsim import",
                env=env,
                stdin_payload=INSTALLER_AUTO_YES_STDIN,
            )
        except InstallerSubprocessError as exc:
            raise InstallerSubprocessError(
                f"isaacsim installed but import failed: {exc}"
            ) from exc

        self._write_partial_marker(stage="install_sim_done", python=str(target_python))
        self._emit_progress(_PHASE_RANGES["install_sim"][1], "Isaac Sim installed")

    # ------------------------------------------------------------------
    # Stage 3 — Clone Isaac Lab
    # ------------------------------------------------------------------
    def _stage_clone(self) -> None:
        self._emit_phase(
            "clone",
            f"Cloning Isaac Lab {ISAAC_LAB_RELEASE.branch_or_tag}...",
        )
        target = self._plan.install_dir

        if _isaac_lab_clone_complete(target):
            log_info(
                f"[isaac-install] Isaac Lab already cloned at {target}; skipping"
            )
            self._emit_progress(_PHASE_RANGES["clone"][1], "Clone skipped")
            return

        if shutil.which("git") is None:
            raise InstallerCloneError(
                "git is not on PATH — cannot clone Isaac Lab. "
                "Install Git for Windows and retry."
            )

        # Init-then-fetch rather than ``git clone`` so we can land into
        # a directory that already has the external-mode venv/ next to it.
        stale_lock = target / ".git" / "index.lock"
        if stale_lock.exists():
            log_warning(f"[isaac-install] removing stale git lock {stale_lock}")
            try:
                stale_lock.unlink()
            except OSError:
                pass

        if not (target / ".git").exists():
            self._run_subprocess(["git", "init", str(target)], "git init")
        # Idempotent remote add.
        try:
            self._run_subprocess(
                [
                    "git", "-C", str(target), "remote", "add", "origin",
                    ISAAC_LAB_RELEASE.git_url,
                ],
                "git remote add",
                strict=False,
            )
        except InstallerSubprocessError:
            pass

        ref = ISAAC_LAB_RELEASE.branch_or_tag
        # Fetch tag first (depth=1); fall back to a branch fetch if the
        # ref isn't a tag.
        fetch_tag_cmd = [
            "git", "-C", str(target), "fetch", "--depth=1", "origin",
            f"refs/tags/{ref}:refs/tags/{ref}",
        ]
        try:
            self._run_subprocess(fetch_tag_cmd, f"git fetch tag {ref}")
            checkout_target = f"tags/{ref}"
        except InstallerSubprocessError:
            log_info(
                f"[isaac-install] ref {ref!r} is not a tag; trying as branch"
            )
            self._run_subprocess(
                [
                    "git", "-C", str(target), "fetch", "--depth=1", "origin", ref,
                ],
                f"git fetch branch {ref}",
            )
            checkout_target = "FETCH_HEAD"

        self._run_subprocess(
            ["git", "-C", str(target), "checkout", checkout_target],
            "git checkout",
        )

        if not _isaac_lab_clone_complete(target):
            raise InstallerCloneError(
                f"Clone reported success but isaaclab.{{bat,sh}} or "
                f"source/ are missing under {target}"
            )

        self._write_partial_marker(stage="clone_done")
        self._emit_progress(_PHASE_RANGES["clone"][1], "Clone complete")

    # ------------------------------------------------------------------
    # Stage 4 — Install Isaac Lab packages (editable, into owning venv)
    # ------------------------------------------------------------------
    def _stage_install_lab(self) -> None:
        """Install every ``source/*/pyproject.toml`` package as editable.

        This deliberately avoids ``isaaclab.bat -i`` / ``isaaclab.sh -i``.
        The .bat launcher's internal contract (which env vars it honours
        to override Python, whether it expects an Omniverse Launcher
        layout, etc.) varies across releases and is sparsely documented.
        Pip-installing each sub-package as editable using the same
        Python that owns isaacsim is the canonical, fully-deterministic
        path:

          * It works the same on Windows + Linux without launcher quirks.
          * Editable mode keeps the clone authoritative — any patches
            the user makes under ``source/isaaclab*/`` take effect
            without a re-install.
          * Failures map 1:1 to pip errors we can show in the error view.
        """
        self._emit_phase("install_lab", "Installing Isaac Lab packages...")
        target = self._plan.install_dir
        source_dir = target / "source"
        if not source_dir.is_dir():
            raise InstallerCloneError(
                f"Isaac Lab source/ directory missing at {source_dir} — "
                f"the clone stage produced an incomplete layout."
            )

        # Enumerate installable sub-packages. Isaac Lab v0.54.3 ships
        # ``source/isaaclab/``, ``source/isaaclab_assets/``,
        # ``source/isaaclab_mimic/``, ``source/isaaclab_rl/``,
        # ``source/isaaclab_tasks/``; each has its own pyproject.toml.
        sub_pkgs = sorted(
            child for child in source_dir.iterdir()
            if child.is_dir() and (child / "pyproject.toml").exists()
        )
        if not sub_pkgs:
            raise InstallerSubprocessError(
                f"No Isaac Lab sub-packages with pyproject.toml found "
                f"under {source_dir}. Either the clone is incomplete or "
                f"Isaac Lab v{ISAAC_LAB_RELEASE.branch_or_tag} ships a "
                f"different layout than this installer expects."
            )

        target_python = self._owning_python()
        if target_python is None:
            raise InstallerSubprocessError(
                "No interpreter recorded for this install — "
                "stage_install_sim must have run successfully first."
            )

        log_info(
            f"[isaac-install] installing {len(sub_pkgs)} Isaac Lab package(s) "
            f"editable into {target_python}"
        )

        # Ensure CUDA PyTorch BEFORE editable installs. ``isaaclab_rl`` (and
        # several transitive deps under the [all] RL extras) declare
        # ``torch>=2.7`` against PyPI, which on Windows resolves to the
        # CPU-only build by default. A CPU torch is fatal at training time:
        # Isaac Sim 5.1's ``_C.pyd`` initialiser dereferences CUDA-bound
        # symbols during ``PyInit__C`` and aborts the process with
        # ACCESS_VIOLATION (exit 0xC0000005). Upstream ``isaaclab.bat -i``
        # calls ``:ensure_cuda_torch`` for exactly the same reason. We run
        # it twice — once before the editable pass to seed the right wheel,
        # once after to repair any CPU-torch replacement pip may have done
        # while resolving the [all] RL extras (rl_games' transitive pins).
        self._ensure_cuda_torch(target_python)

        # Single pip invocation with all sub-packages so pip's resolver
        # sees them at once and resolves cross-package deps in one pass.
        #
        # ``isaaclab_rl`` carries an ``extras_require`` table for each RL
        # framework (rsl_rl, sb3, skrl, rl_games). Editable-installing it
        # bare pulls only base deps — the framework wheels (most notably
        # ``rsl-rl-lib`` which the UnitPort launcher's precheck requires)
        # are silently omitted, causing every freshly-installed Isaac Lab
        # to fail at first training with "Isaac venv is missing rsl_rl".
        # Upstream ``isaaclab.bat -i`` defaults to ``[all]`` for exactly
        # the same reason — match that contract here so the UnitPort
        # installer ends in a runnable state, not a half-installed one.
        cmd = [
            str(target_python), "-m", "pip", "install",
            "--disable-pip-version-check",
            "--upgrade-strategy", "only-if-needed",
        ]
        _RL_EXTRAS = "[all]"
        for pkg in sub_pkgs:
            spec = str(pkg)
            if pkg.name == "isaaclab_rl":
                spec = f"{spec}{_RL_EXTRAS}"
            cmd.extend(["--editable", spec])

        # Isaac Lab pins legacy sdists (e.g. ``flatdict==4.0.1``) whose
        # ``setup.py`` does ``import pkg_resources`` at module load.
        # Setuptools >=81 dropped the bundled ``pkg_resources`` package,
        # so pip's build-isolation env (which downloads the latest
        # setuptools by default) crashes with
        # ``ModuleNotFoundError: No module named 'pkg_resources'``.
        # Writing a constraints file and pointing ``PIP_CONSTRAINT`` at
        # it makes pip apply the cap to the build-isolation env too
        # (pip 22.x+ honours PIP_CONSTRAINT for build deps).
        constraints_file = target / ".unitport_build_constraints.txt"
        try:
            constraints_file.write_text(
                "# Auto-generated by IsaacLabInstaller — caps setuptools so\n"
                "# legacy sdists (flatdict==4.0.1, etc.) can still import\n"
                "# pkg_resources during their build phase.\n"
                "setuptools<81\n",
                encoding="utf-8",
            )
        except OSError as exc:
            log_warning(
                f"[isaac-install] could not write build constraints "
                f"file {constraints_file}: {exc}"
            )

        env = self._acceptance_env()
        env["PIP_CONSTRAINT"] = str(constraints_file)
        self._run_subprocess(
            cmd, "pip install Isaac Lab (editable)", env=env,
        )

        # Second pass — see _ensure_cuda_torch comment above. The [all]
        # extras resolve includes rl_games' transitive deps which can
        # quietly downgrade torch back to CPU. Re-asserting after pip
        # mirrors upstream isaaclab.bat exactly.
        self._ensure_cuda_torch(target_python)

        # Sanity: any of the sub-packages should be importable now.
        # Pick ``isaaclab`` if present, else the first package. Same
        # env + stdin treatment as the isaacsim smoke check — most
        # isaaclab modules import isaacsim transitively, so we need
        # to silence the Omniverse Kit EULA prompt here too.
        verify_pkg_name = "isaaclab" if any(
            p.name == "isaaclab" for p in sub_pkgs
        ) else sub_pkgs[0].name
        verify_cmd = [
            str(target_python), "-c", f"import {verify_pkg_name}",
        ]
        try:
            self._run_subprocess(
                verify_cmd,
                f"verify import {verify_pkg_name}",
                env=env,
                stdin_payload=INSTALLER_AUTO_YES_STDIN,
            )
        except InstallerSubprocessError as exc:
            raise InstallerSubprocessError(
                f"Isaac Lab installed but ``import {verify_pkg_name}`` "
                f"failed: {exc}"
            ) from exc

        self._write_partial_marker(stage="install_lab_done")
        self._emit_progress(_PHASE_RANGES["install_lab"][1], "Isaac Lab installed")

    # ------------------------------------------------------------------
    # Stage 5 — Register
    # ------------------------------------------------------------------
    def _stage_register(self) -> InstallReport:
        self._emit_phase("register", "Registering with EngineService...")
        target = self._plan.install_dir
        ok = get_engine_service().register_isaac_local(
            str(target), source="install"
        )
        if not ok:
            raise InstallerError(
                f"EngineService refused to register {target} — installation "
                f"layout looks malformed (missing isaaclab.sh/bat or source/)."
            )

        now = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        owning_python = self._owning_python() or self._project_venv_python()
        report = InstallReport(
            install_dir=target,
            mode=self._plan.mode,
            isaac_sim_version=ISAAC_SIM_RELEASE.version_label,
            isaac_lab_tag=ISAAC_LAB_RELEASE.branch_or_tag,
            venv_python=str(owning_python) if owning_python else "",
            installed_at=now,
        )

        try:
            from application.service.user_workspace import machine_config_dir
            state_path = machine_config_dir() / _INSTALL_STATE_REL
            state_path.parent.mkdir(parents=True, exist_ok=True)
            up_data(
                state_path,
                data={
                    "isaac_lab": {
                        "version": ISAAC_SIM_RELEASE.version_label,
                        "tag": ISAAC_LAB_RELEASE.branch_or_tag,
                        "root": str(target),
                        "venv_mode": self._plan.mode,
                        "venv_python": report.venv_python,
                        "installed_at": now,
                    }
                },
                merge=True,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[isaac-install] could not update install_state.json: {exc}"
            )

        try:
            self._partial.unlink(missing_ok=True)
        except OSError:
            pass

        self._emit_progress(_PHASE_RANGES["register"][1], "Registered")
        log_success(
            f"[isaac-install] complete: {target} "
            f"(sim={ISAAC_SIM_RELEASE.version_label}, "
            f"lab={ISAAC_LAB_RELEASE.branch_or_tag}, mode={self._plan.mode})"
        )
        return report

    # ==================================================================
    # Helpers
    # ==================================================================

    def _check_cancelled(self) -> None:
        if self._plan.check_cancelled is not None:
            try:
                self._plan.check_cancelled()
            except Exception as exc:  # noqa: BLE001
                raise InstallerCancelled(str(exc)) from exc

    def _emit_phase(self, phase: str, label: str) -> None:
        log_info(f"[isaac-install] {phase}: {label}")
        self._current_phase = phase
        self._signals.isaac_install_phase.emit(phase, label)

    def _emit_progress(self, fraction: float, label: str) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self._signals.isaac_install_progress.emit(clamped, label)

    def _write_partial_marker(self, *, stage: str, python: Optional[str] = None) -> None:
        """Persist a tiny ini-shaped marker so a Retry can skip stages."""
        try:
            self._partial.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"stage={stage}",
                f"updated={_dt.datetime.utcnow().isoformat()}Z",
                f"mode={self._plan.mode}",
            ]
            if python:
                lines.append(f"python={python}")
            elif self._owning_python():
                lines.append(f"python={self._owning_python()}")
            self._partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            log_warning(f"[isaac-install] could not write partial marker: {exc}")

    def _owning_python(self) -> Optional[Path]:
        """Resolve the python that owns this install's isaacsim.

        Always points at the dedicated ``<install_dir>/.venv/``.
        Falls back to the partial marker (for a Retry resuming after a
        prior run) when the venv hasn't been created yet on this attempt.
        """
        dedicated_python = self._dedicated_venv_python()
        if dedicated_python is not None and dedicated_python.exists():
            return dedicated_python
        # Marker fallback (Retry case — venv may already exist from a
        # prior run that crashed before stage_install_sim could create
        # the marker on the new attempt).
        try:
            if self._partial.exists():
                for line in self._partial.read_text(encoding="utf-8").splitlines():
                    if line.startswith("python="):
                        p = Path(line.split("=", 1)[1].strip())
                        if p.exists():
                            return p
        except OSError:
            pass
        return None

    # ------------------------------------------------------------------
    # CUDA PyTorch guarantee
    # ------------------------------------------------------------------
    # Pinned to match upstream ``isaaclab.bat :ensure_cuda_torch`` (Isaac
    # Lab v2.3.2). Bump alongside ``ISAAC_SIM_RELEASE`` / ``ISAAC_LAB_RELEASE``
    # whenever upstream rotates the supported torch matrix.
    _TORCH_VER = "2.7.0"
    _TORCHVISION_VER = "0.22.0"
    _CUDA_TAG = "cu128"
    _PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"
    # tensordict ships compiled wheels tied to a specific torch ABI.
    # ``rsl-rl-lib==3.1.2`` (pulled by ``isaaclab_rl[rsl_rl]``) lists
    # ``tensordict`` with no version pin, so pip resolves to the latest
    # (0.12.x at time of writing) — those wheels are built against torch
    # 2.8+ and crash with ACCESS_VIOLATION (exit 0xC0000005) at
    # ``_C.pyd!PyInit__C`` when ``tensordict._C`` is loaded inside an
    # Isaac Sim Kit subprocess (standalone Python imports them fine — the
    # crash only manifests under Kit's DLL-search-order quirks). The 0.11.x
    # line is the last release built for torch 2.7. Pin here so reinstalls
    # are reproducible; bump when ``_TORCH_VER`` is bumped to 2.8+.
    _TENSORDICT_VER = "0.11.0"

    def _ensure_cuda_torch(self, target_python: Path) -> None:
        """Ensure the install venv has CUDA-enabled torch, not the CPU build.

        Direct port of upstream ``isaaclab.bat :ensure_cuda_torch`` (with
        a tensordict pin added). Isaac Sim 5.1's native extensions
        hard-link against CUDA-bound symbols in ``torch._C`` — a CPU
        torch build (which is what PyPI's default ``torch>=2.7`` resolves
        to on Windows) crashes the process with ACCESS_VIOLATION (exit
        0xC0000005) inside ``_C.pyd!PyInit__C`` during Isaac Sim bootstrap.
        We additionally pin ``tensordict`` to the torch-2.7-compatible
        line because rsl-rl-lib pulls it transitively with no version
        constraint and the latest (0.12.x, torch-2.8-ABI) wheels trigger
        the same ACCESS_VIOLATION at ``tensordict._C`` load inside Kit.

        Called twice from :meth:`_stage_install_lab` (before + after the
        editable pass), matching upstream so a transitive [all]-extras
        downgrade is repaired immediately rather than surfacing at first
        training.
        """
        expected = f"{self._TORCH_VER}+{self._CUDA_TAG}"
        env = self._acceptance_env()

        # Probe current torch + tensordict — capture_output, no failure
        # if either is missing.
        probe = subprocess.run(
            [
                str(target_python), "-c",
                "import importlib.metadata as m\n"
                "def v(name):\n"
                "    try: return m.version(name)\n"
                "    except m.PackageNotFoundError: return ''\n"
                "print('TORCH', v('torch'))\n"
                "print('TD', v('tensordict'))\n",
            ],
            capture_output=True, text=True, env=env, check=False,
        )
        out_lines = (probe.stdout or "").splitlines()
        cur_torch = next(
            (ln.split(" ", 1)[1] for ln in out_lines if ln.startswith("TORCH ")),
            "",
        ).strip()
        cur_td = next(
            (ln.split(" ", 1)[1] for ln in out_lines if ln.startswith("TD ")),
            "",
        ).strip()

        torch_ok = cur_torch == expected
        td_ok = cur_td == self._TENSORDICT_VER
        if torch_ok and td_ok:
            log_info(
                f"[isaac-install] PyTorch {expected} + tensordict "
                f"{self._TENSORDICT_VER} already installed — skipping "
                f"CUDA torch ensure."
            )
            return

        if not torch_ok:
            if cur_torch:
                log_info(
                    f"[isaac-install] replacing PyTorch {cur_torch} -> {expected}..."
                )
                self._run_subprocess(
                    [
                        str(target_python), "-m", "pip", "uninstall", "-y",
                        "torch", "torchvision", "torchaudio",
                    ],
                    "pip uninstall torch/torchvision/torchaudio",
                    env=env,
                    strict=False,
                )
            else:
                log_info(
                    f"[isaac-install] installing PyTorch {self._TORCH_VER} "
                    f"with CUDA {self._CUDA_TAG}..."
                )
            self._run_subprocess(
                [
                    str(target_python), "-m", "pip", "install",
                    "--disable-pip-version-check",
                    "--index-url", self._PYTORCH_INDEX,
                    f"torch=={self._TORCH_VER}",
                    f"torchvision=={self._TORCHVISION_VER}",
                ],
                f"pip install torch=={self._TORCH_VER}+{self._CUDA_TAG}",
                env=env,
            )

        if not td_ok:
            log_info(
                f"[isaac-install] pinning tensordict "
                f"{cur_td or '(missing)'} -> {self._TENSORDICT_VER} "
                f"(torch-{self._TORCH_VER} ABI match)..."
            )
            # No --index-url: tensordict wheels live on PyPI, not on the
            # PyTorch wheel index. ``--force-reinstall`` rather than
            # uninstall+install so the wheel is re-fetched even if a
            # later pip pass coincidentally re-resolves to the same
            # version (covers the case where a stale partial-install
            # left mismatched on-disk state).
            self._run_subprocess(
                [
                    str(target_python), "-m", "pip", "install",
                    "--disable-pip-version-check",
                    "--force-reinstall", "--no-deps",
                    f"tensordict=={self._TENSORDICT_VER}",
                ],
                f"pip install tensordict=={self._TENSORDICT_VER}",
                env=env,
            )

    def _ensure_dedicated_venv(self) -> Path:
        """Create ``<install_dir>/.venv/`` if absent and return its python.

        ``.venv`` (not ``venv``) matches the convention
        :func:`registers.backends._find_isaac_python` already searches —
        so the post-install ``_detect_isaac_lab`` probe and the
        ``application/training/isaac_lab/backend.py`` launcher both
        discover this interpreter automatically with zero further wiring.
        """
        venv_python = self._dedicated_venv_python()
        venv_dir = self._plan.install_dir / ".venv"
        if venv_python is not None and venv_python.exists():
            log_info(f"[isaac-install] reusing existing dedicated venv at {venv_dir}")
            return venv_python
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        log_info(f"[isaac-install] creating dedicated venv at {venv_dir}")
        # Use sys.executable as the base — the project's interpreter is
        # a fine base for ``python -m venv`` (the new venv is standalone
        # by default, no shared site-packages unless we asked for
        # ``--system-site-packages``, which we don't).
        import sys as _sys
        self._run_subprocess(
            [_sys.executable, "-m", "venv", str(venv_dir)],
            "python -m venv",
            env=self._acceptance_env(),
        )
        venv_python = self._dedicated_venv_python()
        if venv_python is None or not venv_python.exists():
            raise InstallerSubprocessError(
                f"venv creation reported success but {venv_dir} has no python"
            )
        # Bring pip up to date in the new venv. ``setuptools<81`` keeps
        # the bundled ``pkg_resources`` available so legacy sdists Isaac
        # Lab depends on (``flatdict==4.0.1`` etc.) can still build —
        # the matching ``PIP_CONSTRAINT`` is applied at install_lab
        # time to cap pip's build-isolation env too.
        self._run_subprocess(
            [str(venv_python), "-m", "pip", "install", "--upgrade",
             "--disable-pip-version-check", "pip", "setuptools<81", "wheel"],
            "pip upgrade (dedicated venv)",
            env=self._acceptance_env(),
        )
        return venv_python

    def _dedicated_venv_python(self) -> Optional[Path]:
        venv_dir = self._plan.install_dir / ".venv"
        if platform.system() == "Windows":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    def _pip_tmp_dir(self) -> Path:
        # Per-install scratch for tempfile-backed pip downloads.
        return self._plan.install_dir / ".unitport_pip_tmp"

    def _pip_cache_dir(self) -> Path:
        # Per-install pip cache so retries can resume warm without
        # touching the system pip cache on a tight system drive.
        return self._plan.install_dir / ".unitport_pip_cache"

    def _acceptance_env(self) -> dict:
        """Parent env + every NVIDIA/Omniverse 'EULA accepted' flag we know.

        ``OMNI_KIT_ACCEPT_EULA=YES`` is the documented one; the rest are
        belt-and-suspenders for Isaac Sim / Kit versions that renamed the
        variable or honour separate flags for privacy / telemetry consent.
        Set in the env passed to every pip + verify subprocess so the
        kernel never tries to read an interactive prompt.

        Also forces ``PYTHONIOENCODING=utf-8`` so child Python's stdout /
        stderr emit UTF-8 regardless of the host console codepage. Without
        this, a Windows zh_CN console + ``encoding="utf-8"`` on our pipe
        side produces three replacement chars per Chinese byte triple
        (the ``A:\\???\\.venv311\\…`` mojibake we see when the install
        path contains non-ASCII characters).
        """
        env = {**os.environ, **INSTALLER_ACCEPT_ENV}
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        # Force pip's download buffer + cache onto the install drive.
        # Provisioned in _stage_preflight; see the WHY KEPT comment there.
        pip_tmp = str(self._pip_tmp_dir())
        pip_cache = str(self._pip_cache_dir())
        env["TMP"] = pip_tmp
        env["TEMP"] = pip_tmp
        env["TMPDIR"] = pip_tmp
        env["PIP_CACHE_DIR"] = pip_cache
        return env

    def _run_subprocess(
        self,
        cmd: List[str],
        label: str,
        *,
        env: Optional[dict] = None,
        strict: bool = True,
        stdin_payload: Optional[bytes] = None,
    ) -> None:
        """Run ``cmd``, stream stdout, surface failures loudly.

        Per-line stdout goes to ``log_debug`` (DEBUG_FLAG=False by
        default ⇒ filtered out of CmdLogWidget but kept in the file
        log). This stops a 1000-line ``pip install isaacsim`` from
        burying the actual install error in the on-screen log.

        We also pump each interesting stdout line through
        ``isaac_install_progress`` (with a synthesized oscillating
        fraction inside the current phase's band) so the
        :class:`IsaacInstallProgressDialog` shows live pip activity
        instead of freezing at the phase-entry fraction for the entire
        duration of a 5-minute pip resolve. The fraction itself only
        wiggles slightly — the real signal here is the changing label,
        which gives the user concrete feedback that the installer is
        making progress and isn't stuck.

        On non-zero exit we re-promote the LAST chunk of stdout to
        ``log_error`` so the user can read the actual reason without
        flipping DEBUG_FLAG on and re-running. The full output is
        always available in the file log under ``Paths.LOGS_DIR``.
        """
        log_info(f"[isaac-install] $ {' '.join(cmd)}  # {label}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_payload else subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        # Ring buffer of recent stdout lines so we can echo a summary
        # to log_error on non-zero exit without flooding the widget.
        from collections import deque
        tail: deque[str] = deque(maxlen=40)
        # Pre-compute the progress envelope for the current phase so
        # heartbeat updates stay inside the band the dialog expects.
        lo, hi = _PHASE_RANGES.get(self._current_phase, (0.0, 1.0))
        # Anchor the heartbeat fraction near the phase entry; we wiggle
        # within the first third of the band so we never overshoot
        # whatever the next phase boundary will set.
        anchor = lo + (hi - lo) * 0.15
        wiggle = (hi - lo) * 0.10
        line_count = 0
        try:
            if stdin_payload and proc.stdin is not None:
                try:
                    proc.stdin.write(stdin_payload.decode("ascii"))
                    proc.stdin.flush()
                    proc.stdin.close()
                except OSError:
                    pass
            assert proc.stdout is not None
            for line in proc.stdout:
                self._check_cancelled()
                stripped = line.rstrip()
                if not stripped:
                    continue
                tail.append(stripped)
                log_debug(stripped)
                line_count += 1
                # Emit a heartbeat every N lines so the dialog status
                # stays responsive without flooding the signal bus.
                if line_count % 5 == 0:
                    fraction = anchor + wiggle * ((line_count // 5) % 2)
                    # Trim the line to something dialog-readable.
                    short = stripped if len(stripped) <= 100 else stripped[:97] + "..."
                    self._emit_progress(fraction, f"{label}: {short}")
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            # Promote the tail so the error is immediately visible in
            # the widget. Cap at the last 20 lines — anything more is
            # usually just download progress.
            log_error(
                f"[isaac-install] {label} exited with code {rc}; "
                f"last {min(len(tail), 20)} lines of output:"
            )
            for line in list(tail)[-20:]:
                log_error(f"    {line}")
            if strict:
                raise InstallerSubprocessError(
                    f"{label} exited with code {rc}"
                )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _isaac_lab_clone_complete(root: Path) -> bool:
    """Markers EngineService.register_isaac_local also checks."""
    return (
        ((root / "isaaclab.sh").exists() or (root / "isaaclab.bat").exists())
        and (root / "source").is_dir()
    )


__all__ = [
    "IsaacLabInstaller",
    "InstallPlan",
    "InstallReport",
    "InstallerError",
    "InstallerPreflightError",
    "InstallerNetworkError",
    "InstallerCloneError",
    "InstallerSubprocessError",
    "InstallerCancelled",
]
