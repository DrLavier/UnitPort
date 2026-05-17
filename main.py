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
from pathlib import Path
from typing import Any, Optional

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
        self._deeplink_handler = install_single_instance_guard(sys.argv)
        if self._deeplink_handler is None:
            # Primary already running — URL forwarded by the guard. Exit
            # before any window is constructed.
            log_info("[auth:startup] secondary instance — forwarded deep-link and exiting")
            return 0
        self._pending_deeplink_url = find_deeplink_url(sys.argv)
        log_info("[auth:startup] single-instance guard installed")

        # Stage 1 -- show the loading screen NOW.
        # MainWindow.__init__ only constructs the loading page (logo +
        # CmdLogWidget that auto-subscribes to LogSignal). Page 1 is built
        # in Stage 3 once dependencies are confirmed, so heavy import
        # chains (auth -> httpx, training -> torch) cannot block the
        # first paint.
        from application.ui.main_window import MainWindow
        self._main_window = MainWindow()
        self._main_window.show()

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

    def _submit_provisioning(self) -> None:
        """Submit ProvisioningTask after the wizard's show event has been
        processed. Deferred via QTimer.singleShot(0, ...) from ``run`` so
        the InstallConfigWizard dialog paints before pip subprocesses begin
        flooding the LoadingScreen log."""
        if self._fatal:
            return
        log_info("[boot] starting provisioning on background worker")
        self._provision_task_id = self._task_master.submit(ProvisioningTask())
        # Fire a non-gating update check in parallel. Its result lands in
        # UpdateService's cache + user.ini[App] and is broadcast via
        # AppSignals.update_check_complete; the Sidebar Update button
        # subscribes to that signal on construction. Submitted here (not
        # in Stage 3) so the check happens while ProvisioningTask runs,
        # not in series with it. Failure is purely a log_warning — the
        # task id is intentionally NOT added to ``boot_ids`` so a network
        # outage does not abort startup.
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
        # The exit self-check (list-only round-trip) was removed in
        # favour of this opt-in upload — users who never flip the
        # toggle pay no shutdown cost. Synchronous: the TasksManager is
        # about to be cancelled so we cannot submit a CloudSyncTask.
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
                    plan = svc.plan_push()
                    n = len(plan.entries)
                    if n == 0:
                        log_info(
                            "[cloud-sync] exit auto-push: nothing to upload"
                        )
                    else:
                        log_info(
                            f"[cloud-sync] exit auto-push: uploading "
                            f"{n} file(s)"
                        )
                        summary = svc.execute(plan)
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
            self._maybe_submit_postsetup()
            return

        if task_id == self._postsetup_task_id:
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
        """
        if self._wizard_done:
            return
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
        # ~/UnitPort/registers/robots_custom.json overlay. Must run *after*
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
        # disk. After the projects/ → ~/UnitPort/projects/ migration, the
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
