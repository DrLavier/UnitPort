#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnitPort - Robot Visual Programming Platform
Main entry file
"""

import logging
import multiprocessing
import os
import platform
import sys
from pathlib import Path
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QObject
from PySide6.QtWidgets import QApplication

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.system.core.utils.project_python import ensure_project_venv

ensure_project_venv(PROJECT_ROOT, Path(__file__))

# ============================================================================
# Config Paths - Central definition for all config file access
# ============================================================================
CONFIG_DIR = PROJECT_ROOT / "src" / "config"
SYSTEM_CONFIG_PATH = CONFIG_DIR / "system.ini"
USER_CONFIG_PATH = CONFIG_DIR / "user.ini"
UI_CONFIG_PATH = CONFIG_DIR / "ui.ini"
LOCALISATION_DIR = PROJECT_ROOT / "localisation"

# Export paths for other modules
CONFIG_PATHS = {
    'config_dir': CONFIG_DIR,
    'system': SYSTEM_CONFIG_PATH,
    'user': USER_CONFIG_PATH,
    'ui': UI_CONFIG_PATH,
    'localisation': LOCALISATION_DIR
}


def get_config_path(config_type: str) -> Path:
    """
    Get config file path by type

    Args:
        config_type: 'system', 'user', 'ui', or 'localisation'

    Returns:
        Path object
    """
    return CONFIG_PATHS.get(config_type, CONFIG_DIR)


# Now import modules (after path setup)
from src.system.core.config_manager import ConfigManager
from src.system.core.theme_manager import init_theme_manager, set_theme
from src.system.core.localisation import get_localisation
from bin.pages.layout.ui import MainWindow
from src.system.models import get_model
from src.system.models.sdk_manager import (
    configure_cyclonedds_env,
    ensure_loco_mujoco,
    ensure_mujoco_menagerie,
    ensure_mujoco_menagerie_selective,
    ensure_selected_sdks,
    register_isaaclab_path,
    verify_registered_sdks,
)
from bin.pages.setup.setup_wizard import SetupWizard, load_setup_state, setup_completed
from src.system.core.utils.logger import setup_logger

_DEV_MODE = os.environ.get("UNITPORT_DEV_MODE", "0") == "1"


class _LogSignalHandler(logging.Handler):
    """Bridges the startup stdlib logger → Qt LogSignal + forces a repaint per record.

    Attached to the 'Celebrimbor' logger during the startup sequence so that
    every logger.info / logger.warning / logger.error call also appears on the
    loading-screen log wallpaper in real time.
    """

    _LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from src.system.core.logger import get_log_signal
            msg = record.getMessage()
            log_type = self._LEVEL_MAP.get(record.levelno, "info")
            get_log_signal().emit_log(msg, log_type)
            _app = QApplication.instance()
            if _app is not None:
                _app.processEvents()
        except Exception:
            pass


def _configure_linux_runtime_env() -> None:
    """Best-effort Linux runtime defaults for direct ``python main.py`` launches."""
    if platform.system() != "Linux":
        return

    if "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
        else:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _log_runtime_banner(logger) -> None:
    # Cross-platform: check for both python.exe (Windows) and python (Linux)
    system = platform.system()
    
    if system == "Windows":
        runtime_python = PROJECT_ROOT / "runtime" / "python" / "python.exe"
    else:
        # Linux/Unix: check for python or python3
        runtime_python = PROJECT_ROOT / "runtime" / "python" / "python"
        if not runtime_python.exists():
            runtime_python = PROJECT_ROOT / "runtime" / "python" / "python3"
    
    if runtime_python.exists():
        logger.info("[runtime:packaged] Project-local Python runtime detected.")
    else:
        if _DEV_MODE:
            logger.warning("[runtime:developer] Packaged runtime absent — running from system Python (dev mode).")
        else:
            logger.warning("[runtime:unpackaged] Packaged runtime absent — distribution integrity may be compromised.")

    logger.info(f"[runtime:python] executable={sys.executable}")
    logger.info(f"[runtime:python] version={sys.version.splitlines()[0]}")


def _run_sdk_verification(logger) -> bool:
    """Lightweight SDK verification (production startup path).

    Returns True if all optional SDKs are available, False if degraded.
    Never triggers network I/O or builds.
    """
    try:
        status = verify_registered_sdks()
    except Exception as exc:
        logger.error(f"[sdk:error] SDK verification raised an exception: {exc}")
        return False

    if status.get("degraded"):
        missing = status.get("sdk_dirs_missing", [])
        unavailable = [k for k, v in status.get("import_checks", {}).items() if v != "ok"]
        if missing:
            logger.warning(f"[sdk:unavailable] SDK directories missing: {missing}")
        if unavailable:
            logger.warning(f"[sdk:unavailable] Optional SDK imports unavailable: {unavailable}")
        logger.warning("[mode:degraded] Starting in degraded mode — hardware SDK features disabled.")
        return False

    logger.info("[sdk:ok] All registered SDKs verified.")
    return True


def _run_sdk_dev_bootstrap(logger) -> None:
    """Developer-only heavy bootstrap (git clone + pip install).

    Only executed when UNITPORT_DEV_MODE=1. Not called in production.
    """
    from src.system.models.sdk_manager import ensure_registered_sdks
    logger.warning("[sdk:dev-bootstrap] Running full SDK bootstrap (developer mode only).")
    try:
        ensure_registered_sdks()
        logger.info("[sdk:dev-bootstrap] SDK bootstrap complete.")
    except Exception as exc:
        logger.error(f"[sdk:dev-bootstrap] Bootstrap failed: {exc}")


def main():
    """Main function"""
    _configure_linux_runtime_env()
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("UnitPort starting...")
    logger.info("=" * 60)

    # ── Runtime banner ─────────────────────────────────────────────────────
    _log_runtime_banner(logger)

    # ── Core config + theme (no Qt widgets; safe before QApplication) ──────
    config = ConfigManager()
    logger.info("Config files loaded")

    init_theme_manager(str(UI_CONFIG_PATH))

    # Resolve theme early so loading screen / homepage use the correct palette
    _raw_theme = config.get('PREFERENCES', 'theme', fallback='dark', config_type='user') or 'dark'
    _theme = _raw_theme.lower() if _raw_theme.lower() in ('light', 'dark') else 'dark'
    set_theme(_theme)

    # ── Qt application ─────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Fix: Apply global stylesheet for Qt native dialogs (file dialogs, etc.)
    # This fixes text color issues in Linux where native dialogs don't show text
    _dialog_style = """
        QWidget {
            color: #e5e7eb;
        }
        QMessageBox, QFileDialog {
            background-color: #2d2d2d;
        }
        QFileDialog QListView, QFileDialog QTreeView {
            color: #e5e7eb;
            background-color: #2d2d2d;
        }
        QFileDialog QListView::item, QFileDialog QTreeView::item {
            color: #e5e7eb;
            background-color: #2d2d2d;
        }
        QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
            background-color: #3d3d3d;
        }
        QLineEdit {
            color: #e5e7eb;
            background-color: #1e1e1e;
        }
    """
    app.setStyleSheet(_dialog_style)

    # ── Main window + in-window startup overlay ────────────────────────────
    window = MainWindow(config)
    window.showFullScreen()

    # ── Mandatory-task worker (runs in background thread) ────────────────

    class _MandatoryWorker(QThread):
        """Runs mandatory (non-optional) startup tasks in a background thread.

        These tasks proceed in parallel while the setup wizard is open so the
        user's configuration time is not wasted.
        """
        finished = Signal()

        def run(self) -> None:
            try:
                # CycloneDDS
                cdds_home = configure_cyclonedds_env(PROJECT_ROOT)
                if not cdds_home:
                    logger.warning("[runtime:cyclonedds] CycloneDDS not configured – Unitree SDK may be unavailable.")

                # SDK verification (lightweight; no network)
                if _DEV_MODE:
                    _run_sdk_dev_bootstrap(logger)
                else:
                    _run_sdk_verification(logger)

                # Localisation
                loc = get_localisation()
                loc.set_localisation_dir(str(LOCALISATION_DIR))
                loc.load_language("en")

                logger.info("[startup] Mandatory tasks complete.")
            except Exception as exc:
                logger.error(f"[startup] Mandatory task error: {exc}")
            finally:
                self.finished.emit()

    # ── Orchestrator: wizard + mandatory worker in parallel ───────────────

    _mandatory_done = False
    _wizard_selections: dict = {}

    def _on_mandatory_finished() -> None:
        nonlocal _mandatory_done
        _mandatory_done = True
        logger.info("[startup] Background mandatory tasks finished.")

    def _get_wizard_selections() -> dict:
        """Show wizard on first launch OR reload previous selections."""
        if setup_completed():
            state = load_setup_state()
            prev = state.get("selections", {})
            if prev:
                logger.info("[setup] Using saved setup selections from previous run.")
            return prev
        logger.info("[setup] First launch detected — opening setup wizard.")
        wizard = SetupWizard(window)
        result_code = wizard.exec()          # modal — blocks this path only
        if result_code == wizard.DialogCode.Accepted:
            sel = wizard.get_selections()
            logger.info(f"[setup] Wizard completed (skipped={sel.get('skipped', False)}).")
            return sel
        logger.info("[setup] Wizard dismissed — using defaults.")
        return {}

    def _run_optional_tasks(selections: dict) -> None:
        """Run user-selected optional downloads (menagerie, SDKs, loco, isaaclab)."""
        skipped = not selections or selections.get("skipped")

        # ── SDK downloads (selective) ─────────────────────────────────────
        if not skipped:
            sdk_list = selections.get("sdks", [])
            if sdk_list:
                logger.info(f"[sdk] Installing {len(sdk_list)} selected SDK package(s) …")
                ensure_selected_sdks(sdk_list)
            else:
                logger.info("[sdk] No SDK packages selected.")

        # ── MuJoCo Menagerie (selective or full) ──────────────────────────
        if not skipped:
            menagerie_folders = selections.get("menagerie_folders", [])
            if menagerie_folders:
                menagerie_ok = ensure_mujoco_menagerie_selective(menagerie_folders)
            else:
                logger.info("[menagerie] No menagerie models selected — skipping.")
                menagerie_ok = True
        else:
            menagerie_ok = ensure_mujoco_menagerie()

        if menagerie_ok:
            logger.info("[menagerie:ok] mujoco_menagerie assets available.")
        else:
            logger.warning(
                "[menagerie:unavailable] mujoco_menagerie not available – "
                "MuJoCo simulation assets disabled."
            )

        # ── Loco-MuJoCo ──────────────────────────────────────────────────
        backend_cfg = (selections or {}).get("backend", {})
        skip_loco = (not skipped) and not backend_cfg.get("loco_mujoco", True)

        if skip_loco:
            logger.info("[loco-mujoco] Skipped by user selection.")
        else:
            loco_ok = ensure_loco_mujoco()
            if loco_ok:
                logger.info("[loco-mujoco:ok] Reference motion library available.")
            else:
                logger.warning(
                    "[loco-mujoco:unavailable] loco-mujoco not available – "
                    "community reference motions disabled."
                )

        # ── Isaac Lab registration ────────────────────────────────────────
        if backend_cfg.get("isaaclab_locate") and backend_cfg.get("isaaclab_path"):
            register_isaaclab_path(backend_cfg["isaaclab_path"])

    def _finalize_startup() -> None:
        """Load robot model and finish the loading screen."""
        # Re-run SDK verification now that optional SDKs may have been installed
        _run_sdk_verification(logger)

        default_robot = config.get('SIMULATION', 'default_robot', fallback='go2')
        logger.info(f"Loading default robot: {default_robot}")

        try:
            model = get_model('unitree')
            if model:
                robot_instance = model(default_robot)
                window.set_robot_model(robot_instance)
                logger.info("[model:ok] Unitree model loaded successfully.")
            else:
                logger.warning("[model:unavailable] Unitree model not found – simulation mode active.")
        except Exception as exc:
            logger.error(f"[model:error] Model loading failed: {exc}")
            logger.warning("[mode:degraded] Continuing with simulation mode.")

        window.run_startup_prewarm()
        window.finish_startup_loading()
        logger.info("App shell displayed (Homepage)")

    # ── Top-level startup entry point ─────────────────────────────────────

    def _run_startup_sequence() -> None:
        try:
            _run_startup_sequence_inner()
        except Exception as exc:
            import traceback
            logger.error(f"[startup:fatal] Startup sequence crashed: {exc}")
            logger.error(f"[startup:fatal] {traceback.format_exc()}")
            QTimer.singleShot(3000, QApplication.instance().quit)

    def _run_startup_sequence_inner() -> None:
        _bridge = _LogSignalHandler()
        logger.addHandler(_bridge)

        try:
            # 1. Start mandatory tasks in a background thread
            worker = _MandatoryWorker()
            worker.finished.connect(_on_mandatory_finished)
            worker.start()

            # 2. Show wizard (modal) — mandatory tasks keep running behind it.
            #    If wizard was completed before, this returns immediately.
            selections = _get_wizard_selections()

            # 3. If the mandatory worker is still running, wait for it.
            #    processEvents() keeps the UI alive (loading log updates).
            if not _mandatory_done:
                logger.info("[startup] Waiting for mandatory tasks to finish …")
            while not _mandatory_done:
                QApplication.processEvents()
                worker.wait(50)

            # 4. Run optional tasks (user-selected downloads) — appended at the end
            _run_optional_tasks(selections)

            # 5. Finalize
            _finalize_startup()
        finally:
            logger.removeHandler(_bridge)

    QTimer.singleShot(50, _run_startup_sequence)
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
