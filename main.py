#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnitPort - Robot Visual Programming Platform
Main entry file
"""

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
from utils.logger import setup_logger


def main():
    """Main function"""
    # Setup file logger
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("UnitPort starting...")
    logger.info("=" * 60)

    # Initialize config manager
    config = ConfigManager()
    logger.info("Config files loaded")

    # Initialize theme manager with UI config path
    init_theme_manager(str(UI_CONFIG_PATH))

    # Initialize localisation
    loc = get_localisation()
    loc.set_localisation_dir(str(LOCALISATION_DIR))
    loc.load_language("en")

    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create main window
    window = MainWindow(config)

    # Load default robot model
    default_robot = config.get('SIMULATION', 'default_robot', fallback='go2')
    logger.info(f"Loading default robot: {default_robot}")

    try:
        model = get_model('unitree')
        if model:
            robot_instance = model(default_robot)
            window.set_robot_model(robot_instance)
            logger.info("Unitree model loaded successfully")
        else:
            logger.warning("Unitree model not found, using simulation mode")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.warning("Continuing with simulation mode")

    # Show window
    window.show()
    logger.info("Main window displayed")

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
