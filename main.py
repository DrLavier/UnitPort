"""UnitPort main entry point.

Single process, single top-level window. ``UnitPortMain`` is the only
top-level controller; ``MainWindow`` (under ``application.ui``) hosts
both the loading-stage page and the main UI in a single
``QStackedWidget`` -- no separate splash window.

Startup workflow (LoadingScreen-first):

    Stage 0  Bootstrap (sync, pre-Qt)
        venv enforcement -> crash hook -> AppUserModelID -> QApplication
        -> stdlib-logging bridge so 3rd-party libs surface in CmdLogWidget.

    Stage 1  Show LoadingScreen (sync, <100 ms target)
        ``MainWindow()`` constructs ONLY the loading page; the main page
        (Sidebar | work_zone) is deferred so heavy import chains
        (auth -> httpx, training -> torch) do not block the first paint.
        ``window.show()`` makes the logo pulse + log wallpaper visible.

    Stage 2  Background tasks (TasksManager workers; logs stream live)
        a) ``ProvisioningTask`` -- five idempotent sub-steps:
           torch CUDA, requirements.txt, loco-mujoco clone, URL scheme
           registration, install_state finalize. Each subprocess streams
           stdout line-by-line into the loading screen via ``log_info``.
        b) ``init_verification`` -- paths and config sanity.
        c) ``data_load`` -- registry hydration + project directory scan
           (primes the ProjectStore snapshot consumed by the sidebar).
        d) ``project_load`` -- runs after ``data_load`` finishes, so the
           "first project" fallback can read the primed snapshot. Resolves
           [Project] last_path with first-project fallback. Never touch UI.

    Stage 3  Swap to main page (main thread)
        ``_finalize`` calls ``MainWindow.build_main_page_now()`` to
        construct page 1 (this is when httpx, paramiko, etc. finally
        import; they are guaranteed installed by Stage 2a). Then
        ``finish_loading()`` fades the loading page out.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Process wall-clock at module import. Used by the exit auto-push hook to
# limit the upload set to files modified during THIS launch — pushing every
# file in USER_CONFIG_DIR on every quit was the documented "Phase 1 trade
# bandwidth for simplicity" path and is now a strict bug: it wastes cloud
# quota and shutdown time on bytes that are byte-for-byte already remote.
_SESSION_START_TS: float = time.time()

# ---- Project root + sys.path injection ---------------------------------------
# Only ``src/`` goes on sys.path -- it owns the ``application``, ``unitport_sdk``,
# ``registers``, and ``scripts`` (training pipeline) packages. The project root
# is intentionally NOT added: PROJECT_ROOT/bootstrap/ holds startup helpers
# (``_check_requirements.py``, ``register_url_scheme.py``, ``detect_ros2.py``)
# that we keep off sys.path so the package name stays unambiguous. In-process
# callers that need PROJECT_ROOT/bootstrap/ files load them by file path via
# ``importlib.util.spec_from_file_location`` (see
# ``application.tools.startup_tasks._load_check_requirements``).
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Keep ProvisioningTask's pip cache on the project drive, matching the path
# install.bat uses for its bootstrap install. This avoids re-downloading the
# wheels bootstrap already fetched (PyQt6, loguru) and keeps the GB-scale
# torch wheel out of %TEMP% on small system drives.
os.environ.setdefault("PIP_CACHE_DIR", str(_PROJECT_ROOT / ".pip-cache"))


# ---- Venv enforcement (bootstrap; mirrors DEMO/src/system/core/utils/project_python.py)
# RELEASE has its own .venv311 (physically isolated from DEMO/.venv311). If
# main.py is launched from any other interpreter and the project venv exists,
# we transparently re-exec under it. If it does not exist we tell the user to
# run install.bat (which start.bat does automatically on a missing venv).
_REEXEC_ENV_KEY = "UNITPORT_VENV_REEXEC"


def _expected_project_python() -> Path:
    if platform.system() == "Windows":
        return (_PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe").resolve()
    return (_PROJECT_ROOT / ".venv311" / "bin" / "python").resolve()


def _ensure_project_venv() -> None:
    expected = _expected_project_python()
    actual = Path(sys.executable).resolve()
    if actual == expected:
        return
    if not expected.exists():
        raise RuntimeError(
            "UnitPort RELEASE must be launched with the project-local .venv311 "
            "interpreter, but that interpreter does not exist.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "Run install.bat (or just start.bat -- it auto-installs)."
        )
    if os.environ.get(_REEXEC_ENV_KEY) == "1":
        raise RuntimeError(
            "UnitPort RELEASE attempted to re-execute under .venv311 but the "
            "process is still running under a different interpreter.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env[_REEXEC_ENV_KEY] = "1"
    raise SystemExit(subprocess.call([str(expected), str(script), *sys.argv[1:]], env=env))


_ensure_project_venv()

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from unitport_sdk import (
    Assets,
    Paths,
    get_data_value,
    get_task_signal,
    get_tasks_manager,
    install_crash_hook,
    load_data,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
)

from application.tools.startup_tasks import FuncTask, PostSetupTask, ProvisioningTask
from application.ui.wizard import (
    InstallConfigWizard,
    load_setup_state,
    save_setup_state,
    setup_completed,
)

# Imported lazily inside ``run`` — the auth module's transitive imports
# (httpx, keyring) must not load before Stage 1 paints the LoadingScreen.
# See run() for the actual import + wiring.


# Stable AppUserModelID for the Windows shell. Without this, the
# Windows taskbar groups our window under ``python.exe`` and shows the
# Python interpreter icon instead of the icon set via setWindowIcon.
# Setting it BEFORE QApplication.show() ensures the shell associates
# our window with this app id from the moment it appears.
_APP_USER_MODEL_ID = "Anthropic.UnitPort.Studio"


def _set_windows_app_user_model_id() -> None:
    """Detach the taskbar group from python.exe so our window icon shows.

    Windows-only. No-op on other platforms or if shell32 is unavailable.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _APP_USER_MODEL_ID
        )
    except Exception as exc:  # pragma: no cover - defensive on minimal Windows
        log_warning(f"[boot] could not set AppUserModelID: {exc}")


# ---- stdlib-logging bridge --------------------------------------------------
# unitport_sdk's log_info / log_warning / log_error already marshal across
# threads onto LogSignal, but third-party libraries imported during the
# ProvisioningTask phase (httpx, paramiko, supabase-py, ...) emit through the
# stdlib ``logging`` module. Without this bridge those records would only
# reach stderr and never appear in the on-screen CmdLogWidget. The handler
# is intentionally minimal: format = "name: message", level = INFO+, route
# by levelno to the matching log_* function.

_LOG_LEVEL_MAP = {
    logging.DEBUG: log_debug,
    logging.INFO: log_info,
    logging.WARNING: log_warning,
    logging.ERROR: log_error,
    logging.CRITICAL: log_error,
}


# Loggers whose records are noise-by-design (per-request status lines,
# connection bookkeeping). Dropped before reaching CmdLogWidget at any
# level. httpx emits "HTTP Request: POST ... 200 OK" once per call; httpcore
# emits connection open/close. Business code that needs to flag a real
# failure (StorageError, AuthError) does its own log_error / log_warning,
# so nothing important hides behind this filter.
_SILENCED_LOGGER_PREFIXES = ("httpx", "httpcore")


class _StdlibToLogSignal(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if any(
                record.name == p or record.name.startswith(p + ".")
                for p in _SILENCED_LOGGER_PREFIXES
            ):
                return
            sink = _LOG_LEVEL_MAP.get(record.levelno, log_info)
            sink(self.format(record))
        except Exception:
            # Never raise from logging.Handler.emit -- it would corrupt the
            # caller's control flow and lose the original log line anyway.
            pass


def _install_stdlib_log_bridge() -> None:
    handler = _StdlibToLogSignal()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    # Idempotent: do not double-attach if a previous run hot-reloaded main.
    for existing in root.handlers:
        if isinstance(existing, _StdlibToLogSignal):
            return
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)


class UnitPortMain:
    """UnitPort single-process controller with a parallel startup."""

    MIN_PY = (3, 10)

    def __init__(self) -> None:
        self._app: Optional[QApplication] = None
        self._main_window = None
        self._task_master = None  # TasksManager singleton (semantic alias)

        # Orchestration state -- one task id per gate.
        self._provision_task_id: str = ""
        self._postsetup_task_id: str = ""
        self._init_task_id: str = ""
        self._data_task_id: str = ""
        self._project_task_id: str = ""
        self._loaded_project_path: str = ""
        # Resolved by ``_project_load_body`` from user.ini ``[Project] last_canvas``.
        # When set + non-empty, ``_finalize`` auto-opens this canvas after
        # ``open_project`` so the user lands directly back where they left off.
        self._loaded_canvas_file_id: str = ""
        # Populated by ``_data_load_body`` via RobotAssetSelfCheckTask; if
        # non-empty, _finalize opens IRRoleAssignmentDialog after
        # finish_loading. List[PendingAssignment]; default empty so _finalize
        # never NoneType-faceplants if data_load raised before populating it.
        self._pending_ir_assignments: list = []
        self._fatal: bool = False

        # Parallel wizard + provisioning gating. Both must finish before
        # ``PostSetupTask`` runs; either branch may finish first.
        self._wizard: Optional[InstallConfigWizard] = None
        self._wizard_done: bool = False
        self._provision_done: bool = False
        self._wizard_selections: dict = {}

        # Auth deeplink (unitport://auth-callback) plumbing. The DeeplinkHandler
        # is the primary process's QLocalServer; if another instance is already
        # running we sys.exit(0) in ``run`` after forwarding the URL.
        self._deeplink_handler: Optional[Any] = None
        self._pending_deeplink_url: Optional[str] = None

    # -------------------------------------------------------------------
    # Entry point -- 4-stage LoadingScreen-first pipeline
    # -------------------------------------------------------------------
    def run(self) -> int:
        # Stage 0 -- bootstrap (sync, pre-window).
        # Order matters on Windows: AppUserModelID must be set before any
        # top-level window is shown, and QApplication's icon must be set
        # before MainWindow is constructed so the title-bar/taskbar icon
        # is consistent from the very first paint. The stdlib-logging
        # bridge is installed AFTER QApplication exists so any subsequent
        # logging.getLogger() emits route through LogSignal -> CmdLogWidget.
        install_crash_hook()
        _set_windows_app_user_model_id()
        self._app = QApplication(sys.argv)
        self._apply_app_icon()
        _install_stdlib_log_bridge()

        # Stage 0a -- workspace housekeeping. All workspace configuration
        # lives in ``system.ini`` (the SDK's authoritative on-disk config).
        # The legacy bootstrap shim (``~/.unitport_paths.ini``) is forbidden
        # and removed on every boot.
        #
        # Three idempotent steps:
        #  1. Legacy root → _guest/: pre-isolation installs kept data
        #     directly under <workspace>/. Move it into <workspace>/_guest/.
        #  2. Per-user setup_state / install_state → machine-level: older
        #     iterations stored these under USER_CONFIG_DIR (per-account);
        #     elevate them to the WORKSPACE root so wizard / SDK install
        #     state is shared across accounts.
        #  3. Always normalise ``system.ini[Resources].user_config_dir``,
        #     ``system.ini[Workspace].root`` and ``system.ini[Session].*``,
        #     materialise the per-account directory if missing, and remove
        #     the bootstrap shim file if any caller wrote one.
        try:
            from application.service.user_workspace import (
                apply_machine_locale_preference,
                ensure_default_workspace_config,
                migrate_legacy_root_to_guest,
                migrate_user_state_to_machine_level,
                reload_paths,
            )
            reloaded = False
            if migrate_legacy_root_to_guest():
                log_info("[boot] legacy USER_CONFIG_DIR layout migrated to _guest/")
                reloaded = True
            if migrate_user_state_to_machine_level():
                log_info("[boot] elevated per-user install state to machine level")
            if ensure_default_workspace_config():
                log_info(
                    "[boot] normalised system.ini workspace config "
                    "(removed any legacy shim, ensured per-account dir exists)"
                )
                reloaded = True
            if reloaded:
                reload_paths()
            # Reconcile the SDK's in-memory language (loaded from
            # user.ini / system.ini at SDK import time) with the
            # device-level ``WORKSPACE/locale.ini`` preference. The user
            # last picked a language; that pick is machine-level and
            # must NEVER be overwritten by an account switch.
            apply_machine_locale_preference()
        except Exception as exc:
            log_warning(f"[boot] workspace housekeeping failed: {exc}")

        # Stage 0b -- single-instance + deeplink guard.
        # Must run after QApplication (QLocalServer needs Qt) but before
        # MainWindow paints — a secondary instance launched from an OAuth
        # callback should forward its URL and exit silently without ever
        # showing a window. Imported directly from the submodule so the
        # auth_manager -> httpx chain does NOT load on Stage 0; that import
        # is deferred to Stage 3 (_finalize), keeping the LoadingScreen
        # paint budget intact.
        from application.service.auth.deeplink_handler import (
            install_single_instance_guard,
            find_deeplink_url,
        )
        # The guard returns (handler, should_exit). should_exit is True
        # ONLY when this process was launched with a unitport:// URL and
        # a primary was already running to receive the forwarded URL —
        # i.e. the OS-level OAuth callback case. A parallel start.bat
        # launch (no URL in argv) gets (None, False) and proceeds as a
        # standalone window; the previous behaviour of unconditionally
        # exiting was a bug that broke side-by-side install testing.
        self._deeplink_handler, should_exit = install_single_instance_guard(sys.argv)
        if should_exit:
            log_info(
                "[auth:startup] secondary instance — forwarded deep-link and exiting"
            )
            return 0
        self._pending_deeplink_url = find_deeplink_url(sys.argv)
        if self._deeplink_handler is not None:
            log_info("[auth:startup] single-instance guard installed (primary)")
        else:
            log_info(
                "[auth:startup] running as parallel secondary — "
                "OAuth callbacks will route to the primary"
            )

        # Stage 1 -- show the loading screen NOW.
        # MainWindow.__init__ only constructs the loading page (logo +
        # CmdLogWidget that auto-subscribes to LogSignal). Page 1 is built
        # in Stage 3 once dependencies are confirmed, so heavy import
        # chains (auth -> httpx, training -> torch) cannot block the
        # first paint.
        from application.ui.main_window import MainWindow
        self._main_window = MainWindow()
        self._main_window.show()

        # Isaac Lab installer signal wiring. MUST happen before
        # PostSetupTask is submitted (Stage 2), because that's when
        # IsaacLabInstallTask gets queued — and the install task's
        # worker thread can start emitting ``isaac_install_phase``
        # before the main thread reaches _finalize. If we connected in
        # _finalize, the preflight signal (and possibly a fast
        # preflight-failure complete signal) would be lost, leaving the
        # progress dialog forever unopened and the install silently
        # failing.
        self._wire_isaac_install_handlers()

        # Stage 2 -- parallel: ProvisioningTask runs on a worker, the
        # InstallConfigWizard opens (non-blocking) on the main thread.
        # Both must finish before PostSetupTask is submitted (heavy
        # menagerie / SDK / Isaac Lab / ROS2 installs that depend on the
        # user's wizard selections AND on requirements.txt being installed).
        #
        # ORDER MATTERS: the wizard is constructed + open()ed BEFORE
        # ProvisioningTask is submitted, and the provisioning submit itself
        # is deferred to the next event-loop tick via QTimer.singleShot(0).
        # This guarantees the wizard's show event paints first, so the user
        # sees the InstallConfigWizard dialog up front rather than watching
        # a wall of pip logs scroll past on the LoadingScreen for several
        # seconds while the wizard is being constructed in the background.
        # (Wizard construction touches disk: MenagerieSelectPage scans the
        # menagerie folder, SdkSelectPage loads the SDK registry + brands.)
        self._task_master = get_tasks_manager()
        get_task_signal().task_finished.connect(self._on_task_finished)

        # Clean shutdown: cancel running tasks and wait for QThread workers
        # before QApplication tears down. Without this, closing the window
        # while ProvisioningTask is mid-pip-install leaves a live QThread,
        # producing "QThread: Destroyed while thread is still running" and
        # a non-zero process exit code.
        self._app.aboutToQuit.connect(self._shutdown_tasks)

        # Wizard branch -- decided synchronously at startup, opened
        # non-blocking. If the user already completed (or skipped) on a
        # previous launch, we skip the dialog and reuse the persisted
        # selections.
        if setup_completed():
            persisted = load_setup_state()
            self._wizard_selections = persisted.get("selections") or {
                "skipped": bool(persisted.get("skipped", False))
            }
            self._wizard_done = True
            log_info("[boot] setup already completed; skipping wizard")
        else:
            # First-launch: blocking language picker BEFORE the wizard
            # opens, so all wizard widgets (and the "Installation in
            # progress" tag on the LoadingScreen) render in the user's
            # preferred language from the first frame. exec() blocks the
            # bootstrap thread but the LoadingScreen behind the picker
            # is already painted and keeps streaming logs.
            self._prompt_initial_language()
            # Show the "Installation in progress" tag below the logo for
            # the entire install phase; cleared in _on_task_finished once
            # PostSetupTask completes (the last heavy step driven by the
            # wizard's selections).
            self._main_window.set_install_message_visible(True)
            log_info("[boot] first launch -- opening InstallConfigWizard")
            self._wizard = InstallConfigWizard(self._main_window)
            self._wizard.completed.connect(self._on_wizard_completed)
            # rejected fires when the user closes the dialog via the
            # window-frame X. Treat it as "skipped" so PostSetupTask can
            # still run (and the LoadingScreen doesn't get stuck waiting).
            self._wizard.rejected.connect(self._on_wizard_rejected)
            self._wizard.open()

        # Provisioning is submitted on the next event-loop tick so the
        # wizard's show event is processed first. The worker thread spawned
        # by submit() runs in parallel; deferring by one tick only delays
        # the start of pip subprocesses by a few milliseconds and is well
        # worth the UX gain of an immediately-visible setup dialog.
        QTimer.singleShot(0, self._submit_provisioning)

        return self._app.exec()

    def _prompt_initial_language(self) -> None:
        """Show the modal language picker on the very first launch.

        Called only when ``setup_completed()`` is False, i.e. there is no
        ``setup_state.json`` on disk yet. The picker enumerates every
        sub-directory under ``localisation/`` and persists the user's
        pick to both ``user.ini[Localisation].lang`` (via
        ``I18n.set_language``) and the machine-level ``locale.ini``.
        After it closes, ``I18n.language_changed`` has already fired so
        every ``i18n_bind``-attached widget on the LoadingScreen has
        retranslated. Failures (no localisation dirs, picker dismissed)
        fall back to whatever the SDK already loaded — boot is never
        blocked on this dialog.
        """
        try:
            from application.ui.dialogs.language_picker_dialog import (
                LanguagePickerDialog,
            )
            dlg = LanguagePickerDialog(self._main_window)
            dlg.exec()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[boot] language picker failed: {exc}")

    def _submit_provisioning(self) -> None:
        """Submit ProvisioningTask after the wizard's show event has been
        processed. Deferred via QTimer.singleShot(0, ...) from ``run`` so
        the InstallConfigWizard dialog paints before pip subprocesses begin
        flooding the LoadingScreen log.

        NOTE: CheckUpdateTask is intentionally NOT submitted here. Importing
        ``application.service.updater`` pulls in ``release_api`` -> ``httpx``,
        which is part of ``requirements.txt`` and therefore not yet present on
        a first launch (only the install.bat bootstrap minimum — PyQt6,
        loguru, cyclonedds — is guaranteed at this point). Submission is
        deferred to ``_submit_update_check`` after ProvisioningTask completes.
        """
        if self._fatal:
            return
        log_info("[boot] starting provisioning on background worker")
        self._provision_task_id = self._task_master.submit(ProvisioningTask())

    def _submit_update_check(self) -> None:
        """Submit CheckUpdateTask once ProvisioningTask has installed httpx.

        Its result lands in UpdateService's cache + user.ini[App] and is
        broadcast via AppSignals.update_check_complete; the Sidebar Update
        button subscribes to that signal on construction. Failure is purely
        a log_warning — the task id is intentionally NOT added to
        ``boot_ids`` so a network outage does not abort startup.
        """
        try:
            from application.service.updater import CheckUpdateTask
            self._task_master.submit(CheckUpdateTask())
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[boot] update check submission failed: {exc}")

    def _shutdown_tasks(self) -> None:
        """Cancel + join all task workers before Qt event loop tears down."""
        # Stop the global input manager first: its gamepad poll thread
        # must join before pygame's atexit handler deinits the video
        # subsystem, otherwise the still-running loop hits
        # "video system not initialized" and tries to log through an
        # already-destroyed LogSignal.
        try:
            from application.service.input import get_global_input_manager
            get_global_input_manager().shutdown()
        except Exception as exc:
            log_warning(f"[boot] input shutdown raised: {exc}")

        # Cloud-sync auto-push on shutdown. Runs only when both
        #   (a) the user is signed in, AND
        #   (b) the auto-push toggle in user.ini[Cloud] auto_push is on.
        # Scope: ONLY files whose mtime is at-or-after ``_SESSION_START_TS``
        # (process module-import wall clock). The manual Push button still
        # uploads the full include set; this hook is the "save my deltas
        # before I quit" path and must never re-upload bytes that haven't
        # changed since this launch began. Synchronous: the TasksManager
        # is about to be cancelled so we cannot submit a CloudSyncTask.
        # plan_push catches IO errors internally and execute() reports
        # per-file failures in its summary rather than raising. Guest
        # sessions skip both checks.
        try:
            from application.service.auth import get_auth_manager
            if get_auth_manager().is_signed_in():
                from unitport_sdk import Config
                if Config.get_value(
                    "Cloud", "auto_push", False, value_type=bool,
                ):
                    from application.service.cloud_sync import (
                        get_cloud_sync_service,
                    )
                    svc = get_cloud_sync_service()
                    plan = svc.plan_push(since_mtime=_SESSION_START_TS)
                    n = len(plan.entries)
                    if n == 0:
                        log_info(
                            f"[cloud-sync] exit auto-push: no files changed "
                            f"this session "
                            f"(skipped {plan.skipped_unchanged} unchanged)"
                        )
                    else:
                        log_info(
                            f"[cloud-sync] exit auto-push: uploading "
                            f"{n} changed file(s) "
                            f"(skipped {plan.skipped_unchanged} unchanged) — "
                            f"please do NOT close the window"
                        )

                        def _on_progress(done: int, total: int, key: str) -> None:
                            ratio = (done / total) if total else 1.0
                            bar_w = 30
                            filled = int(ratio * bar_w)
                            bar = "#" * filled + "-" * (bar_w - filled)
                            short = key if len(key) <= 40 else "..." + key[-37:]
                            line = (
                                f"[cloud-sync] exit auto-push: "
                                f"[{bar}] {ratio * 100:5.1f}% "
                                f"({done}/{total}) {short}"
                            )
                            log_info(line, wrap=(done >= total))

                        summary = svc.execute(plan, progress_cb=_on_progress)
                        log_info(
                            f"[cloud-sync] exit auto-push: "
                            f"ok={int(summary.get('ok', 0) or 0)}/{n} "
                            f"failed={int(summary.get('failed', 0) or 0)} "
                            f"skipped={int(summary.get('skipped', 0) or 0)}"
                        )
        except Exception as exc:
            log_warning(f"[boot] cloud auto-push at shutdown raised: {exc}")

        # Cancel any in-flight auth worker QThread + armed restore-retry
        # timer. AuthManager's _AuthWorker threads belong to the auth
        # facade, not TasksManager, so the master shutdown below would
        # not wait for them — we'd get "QThread: Destroyed while thread
        # is still running" when an OAuth/sign-out call is mid-flight at
        # quit. shutdown() also sets the manager's _shutting_down flag
        # so the retry timer does not re-arm during teardown.
        try:
            from application.service.auth import get_auth_manager
            get_auth_manager().shutdown()
        except Exception as exc:
            log_warning(f"[boot] auth shutdown raised: {exc}")

        if self._task_master is None:
            return
        try:
            self._task_master.shutdown(timeout_ms=3000)
        except Exception as exc:
            log_warning(f"[boot] task shutdown raised: {exc}")

    def _apply_app_icon(self) -> None:
        """Set ``icon_app.svg`` as the QApplication-wide default icon.

        Windows uses this for the title bar and (paired with the
        AppUserModelID set above) for the taskbar entry. Falls back
        silently if the asset is missing.
        """
        if self._app is None:
            return
        path = Assets.find_icon("icon_app")
        if path is None:
            log_warning("[boot] icon_app.svg missing; window will use default icon")
            return
        self._app.setWindowIcon(QIcon(str(path)))

    # -------------------------------------------------------------------
    # Task orchestration (runs on the main thread; signal is queued)
    # -------------------------------------------------------------------
    def _on_task_finished(self, task_id: str, ok: bool, _result: Any) -> None:
        if self._fatal:
            return

        # task_master.task_finished is a global signal that fires for
        # every Task — including user-triggered work submitted long after
        # boot is complete (training runs, MujocoReviewTask, ...). The
        # boot pipeline only owns the five IDs below; failures of any
        # other task are the responsibility of their originator and must
        # not flip the boot _fatal flag or print "startup aborted".
        boot_ids = {
            self._provision_task_id,
            self._postsetup_task_id,
            self._init_task_id,
            self._data_task_id,
            self._project_task_id,
        }
        boot_ids.discard("")
        if task_id not in boot_ids:
            return

        if not ok:
            # No auto-quit. The loading screen has a system window frame
            # (title bar + close button), so the user reads the error and
            # closes the window themselves. crash_*.log under Paths.LOGS_DIR
            # captures the traceback for post-mortem.
            crash_dir = Paths.LOGS_DIR
            log_error(
                f"[boot] task {task_id!r} failed; startup aborted. "
                f"See {crash_dir} for crash logs. Close this window when done."
            )
            self._fatal = True
            return

        if task_id == self._provision_task_id:
            self._provision_done = True
            # ProvisioningTask just finished installing requirements.txt,
            # so httpx (transitive dep of the updater package) is now
            # importable. Fire the non-gating update check here.
            self._submit_update_check()
            self._maybe_submit_postsetup()
            return

        if task_id == self._postsetup_task_id:
            # Install phase is done — hide the "Installation in progress"
            # tag on the LoadingScreen. Subsequent stages (init_verify /
            # data_load / project_load) are sub-second and don't need
            # a marker.
            try:
                self._main_window.set_install_message_visible(False)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[boot] could not hide install label: {exc}")
            self._init_task_id = self._task_master.submit(
                FuncTask("init_verification", self._init_verification_body)
            )
            return

        if task_id == self._init_task_id:
            self._data_task_id = self._task_master.submit(
                FuncTask("data_load", self._data_load_body)
            )
            return

        if task_id == self._data_task_id:
            # project_load depends on the project snapshot primed by
            # data_load (first-project fallback reads it). Sequential.
            self._project_task_id = self._task_master.submit(
                FuncTask("project_load", self._project_load_body)
            )
            return

        if task_id == self._project_task_id:
            # Hop to a clean main-thread tick before triggering the
            # main-page construction + loading -> main-page transition.
            QTimer.singleShot(0, self._finalize)
            return

    # -------------------------------------------------------------------
    # Wizard <-> task gating
    # -------------------------------------------------------------------
    def _on_wizard_completed(self, selections: dict) -> None:
        """Called when the user clicks Finish or Skip in the wizard."""
        self._wizard_selections = dict(selections or {})
        self._wizard_done = True
        log_success(
            f"[boot] wizard completed (skipped={self._wizard_selections.get('skipped', False)}); "
            f"provision_done={self._provision_done}"
        )
        self._maybe_submit_postsetup()

    def _on_wizard_rejected(self) -> None:
        """User closed the wizard via the X button -- treat as skipped.

        Without this branch ``PostSetupTask`` would never submit and the
        app would stay on the LoadingScreen indefinitely. ``rejected``
        fires AFTER ``accepted`` (which we trigger on Finish/Skip), so
        the ``_wizard_done`` guard prevents a double-fire.

        Edge case: if the user closes the dialog BEFORE picking a data
        directory, ``USER_CONFIG_DIR`` is still unset and no boot
        progress is possible — there is no fallback location. Log a
        loud error and refuse to advance; the user can re-open the
        wizard by killing and re-launching the app.
        """
        if self._wizard_done:
            return
        try:
            from unitport_sdk import Paths
            if not Paths.is_user_config_dir_set():
                log_error(
                    "[boot] wizard closed without picking a workspace "
                    "data directory. UnitPort cannot run without one — "
                    "USER_CONFIG_DIR has no built-in default. Close this "
                    "window and re-launch UnitPort to retry the wizard."
                )
                return
        except Exception as exc:                                      # noqa: BLE001
            log_warning(f"[boot] post-wizard sanity check failed: {exc}")
        log_warning("[boot] wizard closed without finishing -- treating as skipped")
        self._on_wizard_completed({"skipped": True})

    def _maybe_submit_postsetup(self) -> None:
        """Submit PostSetupTask iff both gates are ready and we haven't already."""
        if self._fatal:
            return
        if not (self._provision_done and self._wizard_done):
            return
        if self._postsetup_task_id:
            return
        log_info("[boot] both gates ready; submitting PostSetupTask")
        self._postsetup_task_id = self._task_master.submit(
            PostSetupTask(self._wizard_selections)
        )

    def _finalize(self) -> None:
        # Stage 3 -- build the main page on the main thread now that every
        # dependency is installed and every registry is hydrated. This is
        # where Sidebar/UserPanel/auth/httpx finally import; they would
        # have crashed MainWindow.__init__ before Stage 2a installed them.
        self._main_window.build_main_page_now()

        # Auth lifecycle: now that UserPanel exists and is listening on
        # AuthManager signals, wire the deeplink router and trigger silent
        # session restore. Both are pushed onto the next event-loop tick so
        # they do not block the loading-screen fade-out.
        from application.service.auth import get_auth_manager
        auth_mgr = get_auth_manager()
        if self._deeplink_handler is not None:
            self._deeplink_handler.url_received.connect(auth_mgr.handle_oauth_callback)
            if self._pending_deeplink_url:
                pending = self._pending_deeplink_url
                self._pending_deeplink_url = None
                QTimer.singleShot(
                    0, lambda: self._deeplink_handler.dispatch(pending)
                )
        # try_restore_session reads refresh_token from OS keyring and
        # exchanges it via /token?grant_type=refresh_token. On success it
        # emits authenticated(UserProfile); on token-invalid it emits
        # signed_out(); on network failure it logs + signed_out() but
        # PRESERVES the keyring entry so the next launch retries.
        QTimer.singleShot(0, auth_mgr.try_restore_session)

        # Cloud-sync passive self-check. Deferred ~1.2 s so the session
        # restore has a chance to land (it runs on the next tick + a
        # network round-trip). The check just lists the user's cloud
        # prefix; no files are transferred. Guest accounts no-op.
        QTimer.singleShot(1200, self._cloud_self_check_on_startup)

        # Isaac Lab installer signal wiring is already done at Stage 1
        # (right after MainWindow). The progress dialog now uses
        # self._main_window as parent if the wire fired before MainWindow
        # was visible.

        if self._loaded_project_path:
            self._main_window.open_project(self._loaded_project_path)
            # Auto-open last_canvas (if any) so the user lands directly in
            # the canvas they had open last session. If the file_id no
            # longer resolves on disk, the auto-load handler logs and
            # falls back to the empty NewCanvasForm — no fatal.
            if self._loaded_canvas_file_id:
                self._main_window.open_canvas(self._loaded_canvas_file_id)
        else:
            self._main_window.show_project_picker()
        # Fade the loading page out and swap the central stack to the
        # main content page (Sidebar | main_row + work_zone).
        self._main_window.finish_loading()

        # IR-role assignment pop-up — opens iff RobotAssetSelfCheckTask
        # (in _data_load_body) found entries whose tokeniser couldn't
        # auto-resolve and that the user must pick a role for. Modal,
        # blocks training until resolved (or cancelled — next launch
        # will re-detect and re-open).
        self._maybe_open_ir_assignment_dialog()

    def _wire_isaac_install_handlers(self) -> None:
        """Wire the in-app Isaac Lab installer's signals to UI/backend refresh.

        Two responsibilities:

        1. Open :class:`IsaacInstallProgressDialog` the first time
           ``isaac_install_phase`` fires (= the installer task has
           reached its preflight stage on the worker thread). We don't
           pre-open the dialog because most launches do NOT trigger an
           install — only first-launches where the user picked "fresh
           install" + accepted the EULA. Opening on first phase signal
           keeps the dialog out of the path of every other launch.

        2. On a successful ``isaac_install_complete``, force the
           backends registry to re-detect Isaac Lab so the sidebar /
           project picker can immediately show it as available. The
           progress dialog handles its own UI close on success.

        Both connections are single-shot per process — re-installs in
        the same session (retry path) re-use the same connections.
        """
        from application.service.signals import get_app_signals

        signals = get_app_signals()
        # Lazy reference; populated by _open_isaac_install_dialog on
        # first phase signal so a re-install retry can re-open after
        # the prior dialog's accept().
        self._isaac_install_dialog: Optional[Any] = None
        signals.isaac_install_phase.connect(self._on_isaac_install_phase)
        signals.isaac_install_complete.connect(self._on_isaac_install_complete)

    def _on_isaac_install_phase(self, phase: str, label: str) -> None:
        """Open the progress dialog on the first phase signal we see.

        Opens for EVERY phase value including ``"error"`` and EVEN when
        ``self._fatal`` is set. Rationale: the Isaac install failure is
        usually the REASON the boot was marked fatal, so suppressing the
        dialog there would hide the actual cause from the user. They
        deserve to see the error view regardless of what else the boot
        pipeline did.
        """
        if self._isaac_install_dialog is None:
            self._open_isaac_install_dialog(initial_phase=phase, initial_label=label)

    def _open_isaac_install_dialog(
        self, *, initial_phase: str = "", initial_label: str = "",
    ) -> None:
        try:
            from application.ui.dialogs.isaac_install_progress_dialog import (
                IsaacInstallProgressDialog,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[isaac-install] could not open progress dialog: {exc!r}"
            )
            return
        try:
            install_dir = self._isaac_install_dir_hint()
        except Exception:  # noqa: BLE001
            install_dir = ""
        dlg = IsaacInstallProgressDialog(
            install_dir=install_dir, parent=self._main_window,
        )
        dlg.finished.connect(self._on_isaac_install_dialog_closed)
        # Plumb the Retry / Locate buttons to actionable handlers.
        dlg.retry_requested.connect(self._on_isaac_install_retry)
        dlg.locate_mode_requested.connect(self._on_isaac_install_switch_locate)
        self._isaac_install_dialog = dlg
        # Forward the phase that triggered the open so the dialog
        # doesn't start with the stale "Waiting for installer to
        # start..." status. Qt does not re-deliver the signal we are
        # currently handling to the slot the dialog just connected, so
        # we have to push it explicitly.
        if initial_phase:
            try:
                dlg._on_phase(initial_phase, initial_label)
            except Exception:                                       # noqa: BLE001
                pass
        dlg.show()

    def _on_isaac_install_dialog_closed(self, _result: int) -> None:
        self._isaac_install_dialog = None

    def _on_isaac_install_complete(self, ok: bool, message: str) -> None:
        """Refresh backends on success; ensure error visibility on failure.

        The ``_fatal`` flag is NOT checked: an Isaac install failure
        usually IS what triggered the fatal flag (via PostSetupTask
        raising), and suppressing the error dialog in that case would
        hide the cause from the user.
        """
        # If a fast preflight failure raced ahead of the phase signal
        # (or no phase signal was emitted at all — e.g. PostSetupTask's
        # _fail_isaac_install path that fires complete(False) directly),
        # open the dialog now so the user sees what happened. The
        # dialog's own complete handler then transitions to the error
        # view; we back-fill since it missed the signal that just fired.
        if not ok and self._isaac_install_dialog is None:
            log_warning(
                "[isaac-install] complete(False) arrived before any phase "
                "signal opened the dialog — opening it now for visibility"
            )
            self._open_isaac_install_dialog()
            try:
                if self._isaac_install_dialog is not None:
                    self._isaac_install_dialog._on_complete(False, message)
            except Exception as exc:  # noqa: BLE001
                log_warning(
                    f"[isaac-install] could not back-fill error view: {exc}"
                )

        if not ok:
            return

        # State-machine closure: transition setup_state.json's recorded
        # selection from "install" to "locate" now that the install has
        # actually finished. Without this flip the wizard's intent
        # ("please install") keeps firing on every subsequent boot's
        # PostSetupTask._step_isaac_lab, which then re-dispatches the
        # 30 GB installer over a perfectly good install. Pair with the
        # idempotency precheck in PostSetupTask._isaac_lab_already_usable:
        # the precheck is the belt that catches stale states regardless
        # of how they arose, this flip is the suspenders that prevents
        # the state from going stale in the first place.
        try:
            persisted = load_setup_state()
            selections = persisted.get("selections") or {}
            backend = selections.get("backend") or {}
            if backend.get("isaaclab_install"):
                backend["isaaclab_install"] = False
                backend["isaaclab_locate"] = True
                # isaaclab_path was set by the wizard and is still the
                # install root — do NOT clobber it. Keeping it lets the
                # locate branch on next boot re-validate the same path.
                selections["backend"] = backend
                persisted["selections"] = selections
                save_setup_state(persisted)
                log_info(
                    "[isaac-install] setup_state.json updated: "
                    "install → locate (path preserved)"
                )
        except Exception as exc:  # noqa: BLE001
            # Best-effort: if the flip fails, the precheck in
            # _step_isaac_lab still catches the duplicate install on
            # next boot. Log and continue so we do not mask the
            # successful install with a bookkeeping warning.
            log_warning(
                f"[isaac-install] could not flip setup_state.json to "
                f"locate mode after successful install: {exc!r}"
            )

        try:
            from registers.backends import refresh_engine_availability
            refresh_engine_availability()
            log_success(
                "[isaac-install] backends registry refreshed after install"
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[isaac-install] backends refresh after install failed: {exc}"
            )

        # Now that Isaac Lab is registered and the dedicated ``.venv``
        # has a working ``import isaacsim``, re-run the robot asset
        # self-check so any USD-side joint table that the boot-time
        # self-check skipped (because no Isaac was available then) gets
        # dumped now. If the re-check surfaces unmapped roles, open the
        # IR-role assignment dialog so the user can resolve them — same
        # flow as the boot-time path through ``_maybe_open_ir_assignment_dialog``.
        try:
            from application.tools.robot_asset_selfcheck import (
                RobotAssetSelfCheckTask,
            )
            pending = RobotAssetSelfCheckTask().run()
            self._pending_ir_assignments = pending or []
            if self._pending_ir_assignments:
                log_info(
                    f"[isaac-install] post-install self-check produced "
                    f"{len(self._pending_ir_assignments)} pending IR "
                    f"assignment(s); opening dialog"
                )
                self._maybe_open_ir_assignment_dialog()
            else:
                log_info(
                    "[isaac-install] post-install self-check found no "
                    "unmapped joints — IR tables are complete"
                )
        except Exception as exc:                                      # noqa: BLE001
            log_warning(
                f"[isaac-install] post-install self-check failed: {exc}"
            )

    def _on_isaac_install_retry(self) -> None:
        """Re-submit the Isaac Lab install task using the wizard's selections.

        The IsaacLabInstaller's partial-state marker lets the retry skip
        already-completed stages, so this can be safely called after a
        download / extract / clone failure.
        """
        if self._fatal or self._task_master is None:
            return
        try:
            persisted = load_setup_state()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[isaac-install] retry: could not read setup_state: {exc}")
            return
        selections = persisted.get("selections") or {}
        backend = selections.get("backend") or {}
        if not backend.get("isaaclab_install"):
            log_warning(
                "[isaac-install] retry pressed but wizard selections no longer "
                "indicate fresh install — ignoring"
            )
            return
        install_path = str(backend.get("isaaclab_path", "")).strip()
        eula_ids = list(backend.get("eula_accepted_ids") or [])
        if not install_path or not eula_ids:
            log_warning(
                "[isaac-install] retry pressed but install_path or eula_ids "
                "missing in setup_state — ignoring"
            )
            return
        try:
            from application.tools.isaac_lab_install_task import IsaacLabInstallTask
            task = IsaacLabInstallTask(
                install_dir=Path(install_path),
                eula_ids=eula_ids,
            )
            self._task_master.submit(task)
            log_info("[isaac-install] retry: install task re-submitted")
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[isaac-install] retry submission failed: {exc!r}")

    def _on_isaac_install_switch_locate(self) -> None:
        """Stub — opens the legacy locate path. Surfaced for the user
        as the "Switch to Locate mode" button on the install-failed view.

        The full UX (a focused mini-wizard reopening the BackendPage with
        the locate radio pre-selected) is deferred; for now we log a
        clear instruction so the user knows the path forward.
        """
        log_info(
            "[isaac-install] switch-to-locate requested. The wizard's "
            "Locate option is available on next launch — close UnitPort, "
            "delete <USER_CONFIG_DIR>/setup_state.json (or its [App] entry), "
            "and relaunch to re-run the wizard with Locate selected."
        )

    def _isaac_install_dir_hint(self) -> str:
        """Read the install dir from the persisted wizard state (best-effort)."""
        try:
            persisted = load_setup_state()
        except Exception:  # noqa: BLE001
            return ""
        selections = persisted.get("selections") or {}
        backend = selections.get("backend") or {}
        return str(backend.get("isaaclab_path", "") or "").strip()

    def _maybe_open_ir_assignment_dialog(self) -> None:
        pending = list(getattr(self, "_pending_ir_assignments", []) or [])
        if not pending:
            return
        try:
            from application.ui.dialogs.ir_role_assignment_dialog import (
                IRRoleAssignmentDialog,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[boot] ir_role assignment dialog import failed: {exc}")
            return
        log_info(
            f"[boot] opening IR-role assignment dialog for {len(pending)} "
            f"pending entr{'ies' if len(pending) != 1 else 'y'}"
        )
        dlg = IRRoleAssignmentDialog(pending, parent=self._main_window)
        dlg.exec()

    def _cloud_self_check_on_startup(self) -> None:
        """Submit a CloudSyncTask("self_check") if the user is signed in.

        List-only — no files are transferred. Guest sessions skip; failed
        list calls log a warning inside CloudSyncService and surface zero
        objects, which is the desired UX (silent fallback, no popup).
        """
        try:
            from application.service.auth import get_auth_manager
            if not get_auth_manager().is_signed_in():
                return
            from application.tools.cloud_sync_task import CloudSyncTask
            if self._task_master is None:
                return
            self._task_master.submit(CloudSyncTask("self_check"))
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[boot] cloud self-check at startup raised: {exc}")

    # -------------------------------------------------------------------
    # 1. Environment / path / version / registry sanity checks
    # -------------------------------------------------------------------
    def _init_verification_body(self) -> None:
        log_info("[init] verifying environment...")

        if sys.version_info < self.MIN_PY:
            raise RuntimeError(
                f"Python >= {'.'.join(map(str, self.MIN_PY))} required, "
                f"got {sys.version_info[:2]}"
            )

        # Auto-create writable directories.
        for label, p in [
            ("LOGS_DIR", Paths.LOGS_DIR),
            ("RUNTIME_DIR", Paths.RUNTIME_DIR),
        ]:
            p.mkdir(parents=True, exist_ok=True)
            log_debug(f"[init] {label} = {p}")

        # Required pre-existing directories.
        for label, p in [
            ("CONFIG_DIR", Paths.CONFIG_DIR),
            ("ASSETS_DIR", Paths.ASSETS_DIR),
            ("REGISTERS_DIR", Paths.REGISTERS_DIR),
        ]:
            if not p.exists():
                raise RuntimeError(f"{label} missing: {p}")
            log_debug(f"[init] {label} = {p}")

        # Required config files.
        if not Paths.SYSTEM_INI.exists():
            raise RuntimeError(f"system.ini missing: {Paths.SYSTEM_INI}")
        if not Paths.USER_INI.exists():
            log_info(
                f"[init] user.ini absent (first-run normal); will be created on first "
                f"Config.set_value at {Paths.USER_INI}"
            )

        log_success("[init] verification passed")

    # -------------------------------------------------------------------
    # 2. Hydrate persisted state via DataManager
    # -------------------------------------------------------------------
    def _data_load_body(self) -> None:
        log_info("[data] loading persisted state...")

        # system.ini is owned by SDK Config; business code never loads it directly.
        # user.ini is read via DataManager (top-level alias load_data).
        if Paths.USER_INI.exists():
            load_data(Paths.USER_INI)
            log_debug("[data] user.ini loaded")

        # Registry layer (stub today; real catalogues land in stage B).
        from registers import RegistryHub
        RegistryHub.load_all()
        RegistryHub.validate()
        log_info(f"[data] registers loaded: {RegistryHub.summary()}")

        # Auto-discover local USD/URDF/XACRO from custom_mods/models/ (incl.
        # menagerie sparse-checkout) and merge into registers.robots via the
        # <USER_CONFIG_DIR>/registers/robots_custom.json overlay. Must run *after*
        # RegistryHub.load_all so SKU resolution works; results are merged
        # back into the hub by scan_and_merge_assets's RegistryHub.reload().
        # Single source of truth: discovery writes only into the registry
        # overlay, never into state.json.
        try:
            from application.service.robot_assets import (
                get_robot_asset_service,
            )
            n = get_robot_asset_service().scan_and_merge_assets()
            log_info(f"[data] robot_assets discovery merged {n} (sku, kind) pair(s)")
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[data] robot_assets discovery failed: {exc}")

        # Hot-scan projects/<dir>/project.yaml. The snapshot is consumed by
        # ProjectsPanel on construction (no I/O on the UI thread) and by
        # _project_load_body for the first-project fallback below.
        from application.service.projects import get_project_store
        store = get_project_store()
        snapshot = store.refresh_snapshot()
        log_info(
            f"[data] scanned {len(snapshot)} project(s) under {store.projects_root()}"
        )

        # Register built-in training backends (Isaac Lab + SB3+MuJoCo) so
        # algorithm_config.backend selector can route train_pipe at submit
        # time. Backends are advisory until a node calls select_backend(...);
        # ensure_default_backends is idempotent.
        from application.training.backend import ensure_default_backends
        ensure_default_backends()

        # Robot Asset Self-Check — auto-dump any robot whose enabled MJCF/USD
        # per-format table is empty, then collect entries where the tokeniser
        # left ir_role=""; main.py opens the IRRoleAssignmentDialog from
        # _finalize() if this list is non-empty. Runs synchronously on the
        # data_load worker thread; log lines stream through LogSignal like
        # every other startup step.
        #
        # CLAUDE.md §1.8: the previous `except Exception → log_warning +
        # _pending_ir_assignments = []` was a silent degradation — when the
        # self-check crashed (bad overlay, Isaac kit missing, registry
        # validation raising mid-sweep), the user saw zero indication and
        # the IR-role assignment dialog never opened, even though any of
        # those failure modes leave the registry in a state where
        # subsequent training is likely broken. Replace with log_error
        # carrying the full traceback so the LogSignal pane surfaces it
        # loudly; still keep _data_load_body proceeding because the rest
        # of startup (project load, canvas open) can succeed even when
        # self-check is broken, but the failure is now visible.
        from application.tools.robot_asset_selfcheck import (
            RobotAssetSelfCheckTask,
        )
        try:
            self._pending_ir_assignments = RobotAssetSelfCheckTask().run()
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            log_error(
                f"[data] robot_asset self-check crashed — IR-role "
                f"assignment dialog will NOT open this session, and any "
                f"empty per-format tables remain undumped. Investigate "
                f"before training. Exception: {type(exc).__name__}: {exc}\n"
                f"{_tb.format_exc()}"
            )
            self._pending_ir_assignments = []

        log_success("[data] data_load complete")

    # -------------------------------------------------------------------
    # 3. Resolve last-opened project (no UI side-effects)
    # -------------------------------------------------------------------
    def _project_load_body(self) -> None:
        last = ""
        last_canvas = ""
        if Paths.USER_INI.exists():
            # NOTE: ``fallback`` MUST be keyword -- positional args are
            # consumed as additional key lookups by DataManager.
            last = (
                get_data_value(
                    Paths.USER_INI, "Project", "last_path", fallback=""
                )
                or ""
            ).strip()
            last_canvas = (
                get_data_value(
                    Paths.USER_INI, "Project", "last_canvas", fallback=""
                )
                or ""
            ).strip()

        # ``last_path`` must resolve to a *registered* project in the snapshot
        # primed by ``_data_load_body`` — not just any existing directory on
        # disk. After the projects/ → <USER_CONFIG_DIR>/projects/ migration, the
        # old D:\Unitport\EXE\RELEASE\projects\new_test\ may still be on disk
        # while the registered copy lives at the new location; in that case
        # Path(last).exists() would lie and we'd boot into an unbound project
        # + a stale canvas (the "dead screen" — _bind_project fails silently
        # but open_canvas still flips _MainPanel into canvas mode).
        from application.service.projects import get_project_store
        store = get_project_store()
        registered: Optional[Any] = None
        if last:
            registered = store.find_by_path(Path(last))

        if registered is not None:
            log_debug(f"[project] last_path = {last}")
            self._loaded_project_path = last
            # Only carry last_canvas forward when last_path resolved cleanly —
            # a canvas under a different (fallback) project is meaningless.
            if last_canvas:
                # Verify the file still exists on disk; if not, drop it so the
                # picker form can take over instead of showing a stale entry.
                abs_canvas = Path(last) / last_canvas
                if abs_canvas.exists():
                    self._loaded_canvas_file_id = last_canvas
                    log_debug(f"[project] last_canvas = {last_canvas}")
                else:
                    log_warning(
                        f"[project] last_canvas no longer exists on disk: "
                        f"{abs_canvas}"
                    )
                    self._loaded_canvas_file_id = ""
            else:
                self._loaded_canvas_file_id = ""
        else:
            if last:
                if Path(last).exists():
                    log_warning(
                        f"[project] last_path exists on disk but is not a "
                        f"registered project (likely stale after migration): "
                        f"{last}"
                    )
                else:
                    log_warning(
                        f"[project] last_path no longer exists on disk: {last}"
                    )
            # Fallback: cache-load the first project from the snapshot
            # primed by _data_load_body (sorted by manifest.updated_at desc).
            first = store.first_project()
            if first is not None:
                self._loaded_project_path = str(first.path)
                log_info(
                    f"[project] auto-selected first project: {first.name} "
                    f"({first.path})"
                )
            else:
                self._loaded_project_path = ""
                log_info("[project] no resolvable last project; picker will be shown")
            # Don't carry last_canvas across a fallback — different project.
            self._loaded_canvas_file_id = ""

        log_success("[project] project_load complete")


if __name__ == "__main__":
    sys.exit(UnitPortMain().run())
