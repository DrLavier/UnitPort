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
from PySide6.QtCore import QTimer, Qt
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
from src.system.models.sdk_manager import configure_cyclonedds_env, ensure_loco_mujoco, ensure_mujoco_menagerie, verify_registered_sdks
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

    def _run_startup_sequence() -> None:
        try:
            _run_startup_sequence_inner()
        except Exception as exc:
            import traceback
            logger.error(f"[startup:fatal] Startup sequence crashed: {exc}")
            logger.error(f"[startup:fatal] {traceback.format_exc()}")
            # Delay exit slightly so the error message renders in the loading overlay
            QTimer.singleShot(3000, QApplication.instance().quit)

    def _run_startup_sequence_inner() -> None:
        # Bridge stdlib logger → LogSignal so all logger.info/warning/error
        # messages appear on the loading-screen log wallpaper in real time.
        _bridge = _LogSignalHandler()
        logger.addHandler(_bridge)

        try:
            _run_startup_sequence_body()
        finally:
            logger.removeHandler(_bridge)

    def _run_startup_sequence_body() -> None:
        # ── Project-local CycloneDDS env injection ─────────────────────────
        cdds_home = configure_cyclonedds_env(PROJECT_ROOT)
        if not cdds_home:
            logger.warning("[runtime:cyclonedds] CycloneDDS not configured – Unitree SDK may be unavailable.")

        # ── SDK verification (lightweight; no builds) ──────────────────────
        if _DEV_MODE:
            _run_sdk_dev_bootstrap(logger)
        else:
            _run_sdk_verification(logger)

        # ── MuJoCo Menagerie asset check / auto-download ───────────────────
        menagerie_ok = ensure_mujoco_menagerie()
        if menagerie_ok:
            logger.info("[menagerie:ok] mujoco_menagerie assets available.")
        else:
            logger.warning(
                "[menagerie:unavailable] mujoco_menagerie not available – "
                "MuJoCo simulation assets disabled."
            )

        # ── Loco-MuJoCo reference motion library check / auto-clone ─────────
        loco_ok = ensure_loco_mujoco()
        if loco_ok:
            logger.info("[loco-mujoco:ok] Reference motion library available.")
        else:
            logger.warning(
                "[loco-mujoco:unavailable] loco-mujoco not available – "
                "community reference motions disabled."
            )

        # ── Localisation ───────────────────────────────────────────────────
        loc = get_localisation()
        loc.set_localisation_dir(str(LOCALISATION_DIR))
        loc.load_language("en")

        # ── Robot model ────────────────────────────────────────────────────
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

    QTimer.singleShot(50, _run_startup_sequence)
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
