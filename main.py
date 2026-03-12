#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnitPort - Robot Visual Programming Platform
Main entry file
"""

import os
import platform
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Config Paths - Central definition for all config file access
# ============================================================================
CONFIG_DIR = PROJECT_ROOT / "config"
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
from bin.core.config_manager import ConfigManager
from bin.core.theme_manager import init_theme_manager
from bin.core.localisation import get_localisation
from bin.ui import MainWindow
from models import get_model
from models.sdk_manager import configure_cyclonedds_env, verify_registered_sdks
from utils.logger import setup_logger

_DEV_MODE = os.environ.get("UNITPORT_DEV_MODE", "0") == "1"


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
    from models.sdk_manager import ensure_registered_sdks
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

    # ── Project-local CycloneDDS env injection ─────────────────────────────
    # Must happen before any SDK imports that probe CYCLONEDDS_HOME.
    cdds_home = configure_cyclonedds_env(PROJECT_ROOT)
    if cdds_home:
        logger.info(f"[runtime:cyclonedds] CYCLONEDDS_HOME={cdds_home}")
    else:
        logger.warning("[runtime:cyclonedds] CycloneDDS not configured — Unitree SDK may be unavailable.")

    # ── SDK verification (lightweight; no builds) ──────────────────────────
    if _DEV_MODE:
        _run_sdk_dev_bootstrap(logger)
    else:
        _run_sdk_verification(logger)

    # ── Core framework init ────────────────────────────────────────────────
    config = ConfigManager()
    logger.info("Config files loaded")

    init_theme_manager(str(UI_CONFIG_PATH))

    loc = get_localisation()
    loc.set_localisation_dir(str(LOCALISATION_DIR))
    loc.load_language("en")

    # ── Qt application ─────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Fix: Apply global stylesheet for Qt native dialogs (file dialogs, etc.)
    # This fixes text color issues in Linux where native dialogs don't show text
    # Default to dark theme colors, will be updated after theme is loaded
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

    window = MainWindow(config)

    # ── Robot model ────────────────────────────────────────────────────────
    default_robot = config.get('SIMULATION', 'default_robot', fallback='go2')
    logger.info(f"Loading default robot: {default_robot}")

    try:
        model = get_model('unitree')
        if model:
            robot_instance = model(default_robot)
            window.set_robot_model(robot_instance)
            logger.info("[model:ok] Unitree model loaded successfully.")
        else:
            logger.warning("[model:unavailable] Unitree model not found — simulation mode active.")
    except Exception as exc:
        logger.error(f"[model:error] Model loading failed: {exc}")
        logger.warning("[mode:degraded] Continuing with simulation mode.")

    # ── Launch ─────────────────────────────────────────────────────────────
    window.show()
    logger.info("Main window displayed")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
