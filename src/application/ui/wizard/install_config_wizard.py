"""InstallConfigWizard -- first-launch 3-page setup dialog (PyQt6 port).

Strict naming per the migration plan: ``class InstallConfigWizard``.

Port of DEMO ``bin/pages/setup/setup_wizard.py`` to PyQt6 + RELEASE
conventions:

- Theme reads via :class:`Config` (``Config.get_color`` overlay) rather
  than DEMO's ``theme_manager.get_color``.
- ``setup_state.json`` lives outside the project tree at
  ``Paths.USER_CONFIG_DIR / "setup_state.json"`` (per RELEASE rule
  "user state never lives in the project tree"). Read/written via
  ``read_data`` / ``push_data`` so the file goes through DataManager's
  atomic write path automatically.
- Modal opened with ``open()`` not ``exec()`` so the parallel
  ``ProvisioningTask`` worker keeps installing requirements while the
  user clicks through pages.
- Emits ``completed: pyqtSignal(dict)`` carrying the selections;
  ``UnitPortMain._on_wizard_completed`` consumes the dict and gates
  ``PostSetupTask`` submission on both wizard-done AND provision-done.

Schema bit-exact with DEMO ``setup_state.json``::

    {
      "completed": true,
      "skipped": false,
      "selections": {
        "skipped": false,
        "menagerie_folders": ["unitree_go2", ...],
        "sdks": [{"brand": "unitree", "key": "...", "url": "..."}, ...],
        "backend": {
          "mujoco_pip": true,
          "loco_mujoco": true,
          "ros2_enabled": true,
          "isaaclab_enabled": false,
          "isaaclab_install": false,
          "isaaclab_locate": false,
          "isaaclab_cloud_deploy": false,
          "isaaclab_path": "...",
          "cloud_ssh": { ... }    # only when isaaclab_cloud_deploy=true
        }
      }
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config, Paths, log_info, log_warning, push_data, read_data, save_data,
)

from application.service.user_workspace import (
    clear_shim as _clear_user_config_shim,
    default_user_config_dir,
    reload_paths as _reload_paths,
    write_shim as _write_user_config_shim,
)

from application.service.models import menagerie_manager as mm
from application.service.models import sdk_manager as sm

from .menagerie_card import IconFetchWorker, MenagerieCardGrid


# ---------------------------------------------------------------------------
# Setup state persistence (machine-level — shared across all accounts).
#
# Wizard completion is a per-machine fact (the OS-level dependencies were
# either installed or not; the user's answers on the menagerie / Isaac Lab
# / ROS2 dialogs apply to this install). Storing it under USER_CONFIG_DIR
# would force the wizard to re-run every time the user signs into a new
# account. We write it directly under the WORKSPACE root via DataManager
# (NOT push_data, which would prepend USER_CONFIG_DIR).
# ---------------------------------------------------------------------------

_SETUP_STATE_REL = "setup_state.json"


def _setup_state_path() -> Path:
    from application.service.user_workspace import machine_config_dir
    return machine_config_dir() / _SETUP_STATE_REL


def load_setup_state() -> Dict[str, Any]:
    path = _setup_state_path()
    if not path.exists():
        return {}
    data = read_data(path)
    return data if isinstance(data, dict) else {}


def save_setup_state(state: Dict[str, Any]) -> None:
    if not save_data(_setup_state_path(), state):
        log_warning("[wizard] failed to persist setup_state.json")


def setup_completed() -> bool:
    """Return True iff the user has already completed (or skipped) the wizard."""
    return bool(load_setup_state().get("completed", False))


# ---------------------------------------------------------------------------
# Background worker: live menagerie list -- snapshot fallback
# ---------------------------------------------------------------------------

class _MenagerieFetchWorker(QThread):
    items_loaded = pyqtSignal(list)

    def run(self) -> None:
        try:
            names = mm.fetch_remote_packages()
        except Exception:  # noqa: BLE001
            names = list(mm.MENAGERIE_PACKAGES_SNAPSHOT)
        self.items_loaded.emit(names)


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def _wizard_stylesheet() -> str:
    bg = Config.get_color("row_1", "#2a2c33")
    bg_main = Config.get_color("bg_1", "#272829")
    text = Config.get_color("main_t2", "#ffffff")
    text2 = Config.get_color("sub_t1", "#cccccc")
    muted = Config.get_color("sub_t2", "#888888")
    border = Config.get_color("border_1", "#444444")
    btn_bg = Config.get_color("btn_1", "#2a2c33")
    btn_hover = Config.get_color("hover_1", "#525252")
    btn_border = Config.get_color("border_1", "#4b5563")
    btn_text = Config.get_color("main_t1", "#e5e7eb")
    input_bg = Config.get_color("bg_3", "#0f1115")
    accent = "#FF7844"

    return f"""
        QDialog#InstallConfigWizard {{
            background: {bg_main};
        }}

        QLabel#wizardTitle {{
            color: {text};
            font-size: 20px;
            font-weight: 600;
        }}
        QLabel#wizardSubtitle {{
            color: {text2};
            font-size: 13px;
        }}
        QLabel#pageTitle {{
            color: {text};
            font-size: 16px;
            font-weight: 600;
        }}
        QLabel#pageHint {{
            color: {muted};
            font-size: 12px;
        }}
        QLabel#sectionLabel {{
            color: {text2};
            font-size: 13px;
            font-weight: 500;
        }}

        QScrollArea {{
            background: transparent;
            border: 1px solid {border};
            border-radius: 6px;
        }}
        QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}

        QCheckBox {{
            color: {text};
            spacing: 8px;
            font-size: 13px;
            padding: 4px 2px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {btn_border};
            border-radius: 3px;
            background: {input_bg};
        }}
        QCheckBox::indicator:checked {{
            background: {accent};
            border-color: {accent};
        }}
        QCheckBox::indicator:hover {{
            border-color: {accent};
        }}

        QGroupBox {{
            color: {text};
            font-size: 13px;
            font-weight: 500;
            border: 1px solid {border};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 18px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }}

        QPushButton {{
            background: {btn_bg};
            color: {btn_text};
            border: 1px solid {btn_border};
            border-radius: 4px;
            padding: 8px 20px;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background: {btn_hover};
        }}
        QPushButton#primaryButton {{
            background: {accent};
            color: #ffffff;
            border: none;
            font-weight: 600;
        }}
        QPushButton#primaryButton:hover {{
            background: #e06830;
        }}

        QLineEdit {{
            background: {input_bg};
            color: {text};
            border: 1px solid {btn_border};
            border-radius: 4px;
            padding: 8px 10px;
            font-size: 13px;
        }}
        QLineEdit:focus {{
            border-color: {accent};
        }}

        QLabel#dotActive {{
            color: {accent};
            font-size: 18px;
        }}
        QLabel#dotInactive {{
            color: {muted};
            font-size: 18px;
        }}

        QFrame#separator {{
            background: {border};
            max-height: 1px;
        }}
    """


# ===========================================================================
# Page 1: Data directory (USER_CONFIG_DIR picker)
# ===========================================================================
# Shim read/write + path reload helpers live in
# ``application.service.user_workspace`` so the post-install relocate
# (UserPanel gear button) and the first-launch picker stay in sync.


class DataDirectoryPage(QWidget):
    """Pick where UnitPort stores user data (USER_CONFIG_DIR).

    Default lands at ``~/UnitPort`` (the platform-neutral home anchor).
    Choosing a custom location writes ``~/.unitport_paths.ini`` — a tiny
    bootstrap shim consulted by ``Paths._resolve_dynamic`` before
    ``system.ini`` so the choice survives ``git pull`` of the repo.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._default_dir = default_user_config_dir()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Data Directory")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel(
            "Choose where UnitPort stores all per-user data: projects, "
            "exported policies, login tokens, telemetry caches, downloaded "
            "engines, and the user.ini overlay.\n\n"
            "The repository under D:\\Unitport\\EXE\\RELEASE is never written "
            "to at runtime, so this directory is what survives across "
            "redeploys / git pulls."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Path row
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("UserConfig Dir:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setText(str(self._current_value_for_init()))
        self._dir_edit.setPlaceholderText(str(self._default_dir))
        row.addWidget(self._dir_edit, 1)
        self._btn_browse = QPushButton("Browse ...")
        self._btn_browse.setFixedWidth(90)
        self._btn_browse.clicked.connect(self._browse)
        row.addWidget(self._btn_browse)
        self._btn_reset = QPushButton("Reset to default")
        self._btn_reset.setFixedWidth(140)
        self._btn_reset.clicked.connect(self._reset_to_default)
        row.addWidget(self._btn_reset)
        layout.addLayout(row)

        note = QLabel(
            "Default: ~/UnitPort (resolves to your user home — e.g. "
            "C:\\Users\\<name>\\UnitPort on Windows). Pick a different "
            "drive if your C: drive is low on space."
        )
        note.setObjectName("pageHint")
        note.setStyleSheet("color: #888; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._status = QLabel("")
        self._status.setObjectName("pageHint")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._refresh_status()
        self._dir_edit.textChanged.connect(lambda _t: self._refresh_status())

        layout.addStretch(1)

    def _current_value_for_init(self) -> Path:
        """If a shim already exists (re-running wizard manually), pre-fill
        the line edit with the user's prior choice; otherwise default."""
        try:
            existing = Paths._read_bootstrap_user_config_dir()
            if existing:
                return Path(existing).expanduser()
        except Exception:
            pass
        return self._default_dir

    def _browse(self) -> None:
        start = self._dir_edit.text().strip() or str(self._default_dir)
        path = QFileDialog.getExistingDirectory(
            self, "Choose UnitPort Data Directory", start
        )
        if path:
            self._dir_edit.setText(path)

    def _reset_to_default(self) -> None:
        self._dir_edit.setText(str(self._default_dir))

    def _refresh_status(self) -> None:
        try:
            chosen = Path(self._dir_edit.text().strip()).expanduser()
        except Exception:
            self._status.setText("Invalid path")
            self._status.setStyleSheet("color: #FFC107; font-size: 12px;")
            return
        if chosen == self._default_dir:
            self._status.setText("Using default location.")
            self._status.setStyleSheet("color: #888; font-size: 12px;")
        elif chosen.exists():
            self._status.setText(
                f"Custom location — already exists at {chosen}"
            )
            self._status.setStyleSheet("color: #00D26A; font-size: 12px;")
        else:
            self._status.setText(
                f"Custom location — will be created at {chosen}"
            )
            self._status.setStyleSheet("color: #888; font-size: 12px;")

    def get_data_dir(self) -> Path:
        """Return the user-chosen directory (absolute, ~ expanded)."""
        raw = self._dir_edit.text().strip()
        if not raw:
            return self._default_dir
        return Path(raw).expanduser()

    def is_default(self) -> bool:
        return self.get_data_dir() == self._default_dir


# ===========================================================================
# Page 2: Menagerie selection
# ===========================================================================

class MenagerieSelectPage(QWidget):
    """4-column grid of Menagerie packages (page 1 of the wizard)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items_loaded = False
        self._registered_dirs = mm.registered_package_dirs()
        self._installed_dirs = mm.scan_installed_packages()
        self._fetch_worker: Optional[_MenagerieFetchWorker] = None
        self._icon_worker: Optional[IconFetchWorker] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("MuJoCo Menagerie Models")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel(
            "Select the robot models you need from mujoco_menagerie.\n"
            "UnitPort-supported models are pre-selected. Others are optional."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._btn_select_all = QPushButton("Select All")
        self._btn_deselect_all = QPushButton("Deselect All")
        self._btn_refresh = QPushButton("Refresh from GitHub")
        for btn in (self._btn_select_all, self._btn_deselect_all, self._btn_refresh):
            btn.setFixedHeight(30)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_select_all.clicked.connect(self._select_all)
        self._btn_deselect_all.clicked.connect(self._deselect_all)
        self._btn_refresh.clicked.connect(self._start_fetch)
        toolbar.addWidget(self._btn_select_all)
        toolbar.addWidget(self._btn_deselect_all)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_refresh)
        layout.addLayout(toolbar)

        self._grid = MenagerieCardGrid(self)
        layout.addWidget(self._grid, 1)

        self._loading_label = QLabel("Loading model list from GitHub …")
        self._loading_label.setObjectName("pageHint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_label)
        self._loading_label.hide()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._items_loaded:
            self._start_fetch()

    def _start_fetch(self) -> None:
        if self._fetch_worker is not None and self._fetch_worker.isRunning():
            return
        self._loading_label.show()
        self._btn_refresh.setEnabled(False)
        self._fetch_worker = _MenagerieFetchWorker()
        self._fetch_worker.items_loaded.connect(self._on_items_loaded)
        self._fetch_worker.start()

    @pyqtSlot(list)
    def _on_items_loaded(self, items: List[Any]) -> None:
        self._loading_label.hide()
        self._btn_refresh.setEnabled(True)
        names: List[str] = []
        for entry in items or []:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict) and entry.get("name"):
                names.append(str(entry["name"]))
        self._installed_dirs = mm.scan_installed_packages()
        self._grid.populate(
            names,
            installed=self._installed_dirs,
            registered=self._registered_dirs,
            pre_checked=self._registered_dirs,
        )
        self._kick_icon_fetch()
        self._items_loaded = True

    def _kick_icon_fetch(self) -> None:
        missing = self._grid.cards_missing_icon()
        if not missing:
            return
        if self._icon_worker is not None and self._icon_worker.isRunning():
            return
        self._icon_worker = IconFetchWorker(missing, parent=self)
        self._icon_worker.icon_ready.connect(self._grid.update_card_icon_from_path)
        self._icon_worker.start()

    def _select_all(self) -> None:
        self._grid.select_all()

    def _deselect_all(self) -> None:
        self._grid.deselect_all()

    def get_selected_folders(self) -> List[str]:
        return self._grid.selected_packages()

    def hideEvent(self, event) -> None:  # noqa: N802
        if self._icon_worker is not None and self._icon_worker.isRunning():
            self._icon_worker.cancel()
        super().hideEvent(event)


# ===========================================================================
# Page 2: SDK selection
# ===========================================================================

class SdkSelectPage(QWidget):
    """Per-brand SDK checkbox list (page 2 of the wizard)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._brand_groups: Dict[str, Dict[str, QCheckBox]] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("SDK Packages")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel(
            "Select the robot SDK packages to download.\n"
            "Only checked items will be cloned. You can install more later from Settings."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(12)

        # Read brand list from registers via SdkManager (single source of
        # truth -- DEMO read it from a separate model_registry, RELEASE
        # routes everything through registers.brands).
        manager = sm.SdkManager()
        manager.load_registry()
        by_brand: Dict[str, List[sm.SdkProject]] = {}
        for project in manager.get_projects():
            by_brand.setdefault(project.brand, []).append(project)

        from registers import brands
        brands.load()

        for brand_id, projects in by_brand.items():
            brand_dict = brands.get_brand(brand_id) or {}
            display_name = str(brand_dict.get("display_name", brand_id))

            group = QGroupBox(display_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)

            self._brand_groups[brand_id] = {}
            for project in projects:
                label = project.name.replace("_", " ").replace("-", " ").title()
                cb = QCheckBox(label)
                cb.setToolTip(project.url)
                cb.setProperty("brand", brand_id)
                cb.setProperty("project_key", project.name)
                cb.setChecked(True)
                self._brand_groups[brand_id][project.name] = cb
                group_layout.addWidget(cb)

            content_layout.addWidget(group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def get_selected_sdks(self) -> List[Tuple[str, str, str]]:
        """Return list of (brand_id, project_key, url) for checked items."""
        result: List[Tuple[str, str, str]] = []
        manager = sm.SdkManager()
        for brand_id, cbs in self._brand_groups.items():
            for key, cb in cbs.items():
                if cb.isChecked():
                    project = manager.get_project(key)
                    url = project.url if project else ""
                    result.append((brand_id, key, url))
        return result


# ===========================================================================
# Page 3: Backend (MuJoCo / Loco-MuJoCo / ROS2 / Isaac Lab)
# ===========================================================================

class BackendPage(QWidget):
    """Configure backend simulation environments (page 3 of the wizard)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Backend Environment")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel(
            "Configure simulation backends for training and execution.\n"
            "Uncheck items you don't need to speed up setup."
        )
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(16)

        # ---- MuJoCo --------------------------------------------------
        mujoco_group = QGroupBox("MuJoCo")
        mujoco_layout = QVBoxLayout(mujoco_group)
        mujoco_layout.setSpacing(6)
        self._cb_mujoco = QCheckBox("Install MuJoCo (pip package)")
        self._cb_mujoco.setChecked(True)
        self._cb_mujoco.setToolTip("Required for MuJoCo-based simulation and training.")
        mujoco_layout.addWidget(self._cb_mujoco)
        self._cb_loco_mujoco = QCheckBox("Download loco-mujoco reference motion library")
        self._cb_loco_mujoco.setChecked(True)
        self._cb_loco_mujoco.setToolTip(
            "Community reference motion library for locomotion training. "
            "Cloned from github.com/robfiras/loco-mujoco."
        )
        mujoco_layout.addWidget(self._cb_loco_mujoco)
        content_layout.addWidget(mujoco_group)

        # ---- ROS2 ----------------------------------------------------
        ros2_group = QGroupBox("ROS2")
        ros2_layout = QVBoxLayout(ros2_group)
        ros2_layout.setSpacing(6)
        self._cb_ros2 = QCheckBox("Install ROS2 support")
        self._cb_ros2.setChecked(True)
        self._cb_ros2.setToolTip(
            "Enables ROS2-based brands (Unitree Go2, Mangdang Mini Pupper, …).\n"
            "If a native ROS2 Humble install is found on this machine it is "
            "reused; otherwise the Docker bridge image is built."
        )
        ros2_layout.addWidget(self._cb_ros2)
        ros2_hint = QLabel(
            "Auto-detects a native ROS2 install (Humble preferred). When none "
            "is present, the UnitPort Docker bridge image is built instead — "
            "Docker Desktop / Docker Engine must be running."
        )
        ros2_hint.setObjectName("pageHint")
        ros2_hint.setWordWrap(True)
        ros2_hint.setStyleSheet("color: #888; font-size: 11px;")
        ros2_layout.addWidget(ros2_hint)
        content_layout.addWidget(ros2_group)

        # ---- Isaac Lab ----------------------------------------------
        isaac_group = QGroupBox("Isaac Lab (NVIDIA)")
        isaac_layout = QVBoxLayout(isaac_group)
        isaac_layout.setSpacing(6)

        self._cb_isaaclab = QCheckBox("Enable Isaac Lab integration")
        self._cb_isaaclab.setChecked(False)
        self._cb_isaaclab.setToolTip(
            "Isaac Lab provides GPU-accelerated simulation via NVIDIA Isaac Sim.\n"
            "You can install it now or point to an existing installation."
        )
        self._cb_isaaclab.toggled.connect(self._on_isaaclab_toggled)
        isaac_layout.addWidget(self._cb_isaaclab)

        # Sub-options panel (shown after enable)
        self._isaac_options = QWidget()
        opts_layout = QVBoxLayout(self._isaac_options)
        opts_layout.setContentsMargins(24, 4, 0, 0)
        opts_layout.setSpacing(8)

        # Option A: Install from scratch
        self._rb_isaac_install = QRadioButton(
            "Install Isaac Lab + Isaac Sim (from scratch)"
        )
        self._rb_isaac_install.setToolTip(
            "Automatically download and install Isaac Lab + Isaac Sim 5.1\n"
            "into a dedicated directory with its own Python virtual environment.\n"
            "Requires ~30 GB disk space, NVIDIA GPU, and internet connection."
        )
        self._rb_isaac_install.toggled.connect(self._on_isaac_mode_toggled)
        opts_layout.addWidget(self._rb_isaac_install)

        self._install_path_panel = QWidget()
        install_path_layout = QVBoxLayout(self._install_path_panel)
        install_path_layout.setContentsMargins(24, 0, 0, 0)
        install_path_layout.setSpacing(4)
        install_hint = QLabel(
            "Choose the installation directory. Isaac Lab + Isaac Sim will be "
            "installed with a dedicated venv inside this directory."
        )
        install_hint.setObjectName("pageHint")
        install_hint.setWordWrap(True)
        install_path_layout.addWidget(install_hint)

        install_row = QHBoxLayout()
        install_row.setSpacing(6)
        self._install_dir_edit = QLineEdit()
        self._install_dir_edit.setText(
            str(Path.home() / "UnitPort" / "engines" / "isaac")
        )
        self._install_dir_edit.setPlaceholderText("Installation directory ...")
        install_row.addWidget(self._install_dir_edit, 1)
        self._btn_browse_install = QPushButton("Browse ...")
        self._btn_browse_install.setFixedWidth(90)
        self._btn_browse_install.clicked.connect(self._browse_install_dir)
        install_row.addWidget(self._btn_browse_install)
        install_path_layout.addLayout(install_row)

        install_note = QLabel(
            "Isaac Sim 5.1.0 | IsaacLab (pinned) | PyTorch 2.7+cu128\n"
            "Disk: ~30 GB | Requires: NVIDIA GPU with CUDA 12.x driver"
        )
        install_note.setObjectName("pageHint")
        install_note.setStyleSheet("color: #888; font-size: 11px;")
        install_note.setWordWrap(True)
        install_path_layout.addWidget(install_note)
        self._install_path_panel.hide()
        opts_layout.addWidget(self._install_path_panel)

        # Option B: Locate existing
        self._rb_isaac_locate = QRadioButton(
            "Locate existing Isaac Lab installation"
        )
        self._rb_isaac_locate.setToolTip(
            "Point to an existing Isaac Lab root directory.\n"
            "The path will be registered for training calls."
        )
        self._rb_isaac_locate.toggled.connect(self._on_isaac_mode_toggled)
        opts_layout.addWidget(self._rb_isaac_locate)

        self._locate_path_panel = QWidget()
        locate_path_layout = QVBoxLayout(self._locate_path_panel)
        locate_path_layout.setContentsMargins(24, 0, 0, 0)
        locate_path_layout.setSpacing(4)
        locate_row = QHBoxLayout()
        locate_row.setSpacing(6)
        self._isaac_path_edit = QLineEdit()
        self._isaac_path_edit.setPlaceholderText("Isaac Lab root directory ...")
        locate_row.addWidget(self._isaac_path_edit, 1)
        self._btn_browse_isaac = QPushButton("Browse ...")
        self._btn_browse_isaac.setFixedWidth(90)
        self._btn_browse_isaac.clicked.connect(self._browse_isaaclab)
        locate_row.addWidget(self._btn_browse_isaac)
        locate_path_layout.addLayout(locate_row)

        self._isaac_path_status = QLabel("")
        self._isaac_path_status.setObjectName("pageHint")
        locate_path_layout.addWidget(self._isaac_path_status)
        self._locate_path_panel.hide()
        opts_layout.addWidget(self._locate_path_panel)

        # Option C: Cloud SSH deploy
        self._rb_isaac_cloud = QRadioButton(
            "Deploy Isaac Lab to remote Linux server (SSH)"
        )
        self._rb_isaac_cloud.setToolTip(
            "Automatically install Isaac Lab + Isaac Sim on a remote GPU server\n"
            "via SSH. The server will be registered for cloud training.\n"
            "Requires: SSH access, Python 3.11 + NVIDIA GPU on remote host."
        )
        self._rb_isaac_cloud.toggled.connect(self._on_isaac_mode_toggled)
        opts_layout.addWidget(self._rb_isaac_cloud)

        self._cloud_panel = QWidget()
        cloud_layout = QVBoxLayout(self._cloud_panel)
        cloud_layout.setContentsMargins(24, 0, 0, 0)
        cloud_layout.setSpacing(4)
        cloud_hint = QLabel(
            "Enter SSH connection details for the remote GPU server.\n"
            "Isaac Lab will be installed via SSH + SFTP."
        )
        cloud_hint.setObjectName("pageHint")
        cloud_hint.setWordWrap(True)
        cloud_layout.addWidget(cloud_hint)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(QLabel("Server name:"))
        self._cloud_name_edit = QLineEdit()
        self._cloud_name_edit.setPlaceholderText("My GPU Server")
        name_row.addWidget(self._cloud_name_edit, 1)
        cloud_layout.addLayout(name_row)

        host_row = QHBoxLayout()
        host_row.setSpacing(6)
        host_row.addWidget(QLabel("Host:"))
        self._cloud_host_edit = QLineEdit()
        self._cloud_host_edit.setPlaceholderText("10.0.0.1 or gpu-server.example.com")
        host_row.addWidget(self._cloud_host_edit, 1)
        host_row.addWidget(QLabel("Port:"))
        self._cloud_port_edit = QLineEdit()
        self._cloud_port_edit.setText("22")
        self._cloud_port_edit.setFixedWidth(60)
        host_row.addWidget(self._cloud_port_edit)
        cloud_layout.addLayout(host_row)

        user_row = QHBoxLayout()
        user_row.setSpacing(6)
        user_row.addWidget(QLabel("Username:"))
        self._cloud_user_edit = QLineEdit()
        self._cloud_user_edit.setPlaceholderText("root")
        user_row.addWidget(self._cloud_user_edit, 1)
        cloud_layout.addLayout(user_row)

        auth_row = QHBoxLayout()
        auth_row.setSpacing(6)
        auth_row.addWidget(QLabel("Auth:"))
        self._cloud_auth_key = QRadioButton("SSH Key")
        self._cloud_auth_key.setChecked(True)
        self._cloud_auth_pwd = QRadioButton("Password")
        auth_row.addWidget(self._cloud_auth_key)
        auth_row.addWidget(self._cloud_auth_pwd)
        auth_row.addStretch()
        cloud_layout.addLayout(auth_row)

        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        key_row.addWidget(QLabel("Key file:"))
        self._cloud_key_edit = QLineEdit()
        self._cloud_key_edit.setPlaceholderText("~/.ssh/id_rsa")
        self._cloud_key_edit.setText("~/.ssh/id_rsa")
        key_row.addWidget(self._cloud_key_edit, 1)
        self._btn_browse_key = QPushButton("Browse ...")
        self._btn_browse_key.setFixedWidth(90)
        self._btn_browse_key.clicked.connect(self._browse_ssh_key)
        key_row.addWidget(self._btn_browse_key)
        cloud_layout.addLayout(key_row)

        self._cloud_pwd_row = QWidget()
        pwd_row_layout = QHBoxLayout(self._cloud_pwd_row)
        pwd_row_layout.setContentsMargins(0, 0, 0, 0)
        pwd_row_layout.setSpacing(6)
        pwd_row_layout.addWidget(QLabel("Password:"))
        self._cloud_pwd_edit = QLineEdit()
        self._cloud_pwd_edit.setEchoMode(QLineEdit.EchoMode.Password)
        pwd_row_layout.addWidget(self._cloud_pwd_edit, 1)
        self._cloud_pwd_row.hide()
        cloud_layout.addWidget(self._cloud_pwd_row)

        self._cloud_auth_key.toggled.connect(self._on_cloud_auth_toggled)

        remote_row = QHBoxLayout()
        remote_row.setSpacing(6)
        remote_row.addWidget(QLabel("Remote path:"))
        self._cloud_remote_dir_edit = QLineEdit()
        self._cloud_remote_dir_edit.setText("~/UnitPort/engines/isaac")
        self._cloud_remote_dir_edit.setPlaceholderText("~/UnitPort/engines/isaac")
        remote_row.addWidget(self._cloud_remote_dir_edit, 1)
        cloud_layout.addLayout(remote_row)

        cloud_note = QLabel(
            "Isaac Sim 5.1.0 | IsaacLab (pinned) | PyTorch 2.7+cu128\n"
            "Requires: Ubuntu 22.04+, NVIDIA GPU, CUDA 12.x, ~30 GB disk"
        )
        cloud_note.setObjectName("pageHint")
        cloud_note.setStyleSheet("color: #888; font-size: 11px;")
        cloud_note.setWordWrap(True)
        cloud_layout.addWidget(cloud_note)
        self._cloud_panel.hide()
        opts_layout.addWidget(self._cloud_panel)

        self._isaac_options.hide()
        isaac_layout.addWidget(self._isaac_options)

        content_layout.addWidget(isaac_group)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def _on_isaaclab_toggled(self, checked: bool) -> None:
        self._isaac_options.setVisible(checked)
        if not checked:
            self._rb_isaac_install.setChecked(False)
            self._rb_isaac_locate.setChecked(False)
            self._rb_isaac_cloud.setChecked(False)

    def _on_isaac_mode_toggled(self, _checked: bool) -> None:
        self._install_path_panel.setVisible(self._rb_isaac_install.isChecked())
        self._locate_path_panel.setVisible(self._rb_isaac_locate.isChecked())
        self._cloud_panel.setVisible(self._rb_isaac_cloud.isChecked())

    def _on_cloud_auth_toggled(self, key_checked: bool) -> None:
        self._cloud_key_edit.setVisible(key_checked)
        self._btn_browse_key.setVisible(key_checked)
        self._cloud_pwd_row.setVisible(not key_checked)

    def _browse_ssh_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Private Key",
            str(Path.home() / ".ssh"),
            "All Files (*)",
        )
        if path:
            self._cloud_key_edit.setText(path)

    def _browse_install_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose Isaac Lab Installation Directory",
            self._install_dir_edit.text() or str(Path.home()),
        )
        if path:
            self._install_dir_edit.setText(path)

    def _browse_isaaclab(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Isaac Lab Root Directory", ""
        )
        if path:
            self._isaac_path_edit.setText(path)
            self._validate_isaaclab_path(path)

    def _validate_isaaclab_path(self, path: str) -> None:
        p = Path(path)
        markers = ["setup.py", "pyproject.toml", "isaaclab", "source",
                   "isaaclab.sh", "isaaclab.bat"]
        found = [m for m in markers if (p / m).exists()]
        if found:
            self._isaac_path_status.setText("Valid Isaac Lab installation detected")
            self._isaac_path_status.setStyleSheet("color: #00D26A; font-size: 12px;")
            self._cb_isaaclab.setChecked(True)
            self._rb_isaac_locate.setChecked(True)
        else:
            self._isaac_path_status.setText(
                "Could not verify Isaac Lab at this path -- "
                "expected isaaclab.sh/bat or source/ directory"
            )
            self._isaac_path_status.setStyleSheet("color: #FFC107; font-size: 12px;")

    def get_config(self) -> Dict[str, Any]:
        is_install = self._rb_isaac_install.isChecked()
        is_locate = self._rb_isaac_locate.isChecked()
        is_cloud = self._rb_isaac_cloud.isChecked()

        cfg: Dict[str, Any] = {
            "mujoco_pip": self._cb_mujoco.isChecked(),
            "loco_mujoco": self._cb_loco_mujoco.isChecked(),
            "ros2_enabled": self._cb_ros2.isChecked(),
            "isaaclab_enabled": self._cb_isaaclab.isChecked(),
            "isaaclab_install": is_install,
            "isaaclab_locate": is_locate,
            "isaaclab_cloud_deploy": is_cloud,
            "isaaclab_path": (
                self._install_dir_edit.text().strip() if is_install
                else self._isaac_path_edit.text().strip()
            ),
        }

        if is_cloud:
            try:
                port = int(self._cloud_port_edit.text() or 22)
            except ValueError:
                port = 22
            cfg["cloud_ssh"] = {
                "server_name": (
                    self._cloud_name_edit.text().strip()
                    or self._cloud_host_edit.text().strip()
                ),
                "host": self._cloud_host_edit.text().strip(),
                "port": port,
                "username": self._cloud_user_edit.text().strip(),
                "auth_method": "key" if self._cloud_auth_key.isChecked() else "password",
                "private_key_path": self._cloud_key_edit.text().strip(),
                "password": self._cloud_pwd_edit.text(),
                "remote_install_dir": (
                    self._cloud_remote_dir_edit.text().strip()
                    or "~/UnitPort/engines/isaac"
                ),
            }

        return cfg


# ===========================================================================
# Main wizard dialog
# ===========================================================================

class InstallConfigWizard(QDialog):
    """Three-page first-launch installation wizard.

    Strict naming per the migration plan: ``class InstallConfigWizard``
    (DEMO called the equivalent ``SetupWizard``).

    Open with ``open()`` (non-blocking modal). On finish/skip emits the
    ``completed`` signal carrying the selections dict; the dialog also
    persists ``setup_state.json`` itself so a crash mid-postsetup
    doesn't re-trigger the wizard on next launch.
    """

    PAGE_TITLES = [
        "Data Directory",
        "Menagerie Models",
        "SDK Packages",
        "Backend Environment",
    ]

    # selections dict (see module docstring for the schema)
    completed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("InstallConfigWizard")
        self.setWindowTitle("UnitPort — Custom Setup")
        self.setMinimumSize(1320, 720)
        self.resize(1340, 760)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )

        self._result_data: Dict[str, Any] = {}
        self._init_ui()
        self.setStyleSheet(_wizard_stylesheet())

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(0)

        # ---- Header --------------------------------------------------
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("UnitPort Setup")
        title.setObjectName("wizardTitle")
        header.addWidget(title)
        subtitle = QLabel(
            "Choose which components to install. You can change these later in Settings."
        )
        subtitle.setObjectName("wizardSubtitle")
        header.addWidget(subtitle)
        root.addLayout(header)
        root.addSpacing(12)

        # ---- Separator ----------------------------------------------
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        root.addWidget(sep)
        root.addSpacing(12)

        # ---- Page stack ----------------------------------------------
        self._stack = QStackedWidget()
        self._data_dir_page = DataDirectoryPage()
        self._menagerie_page = MenagerieSelectPage()
        self._sdk_page = SdkSelectPage()
        self._backend_page = BackendPage()
        self._stack.addWidget(self._data_dir_page)
        self._stack.addWidget(self._menagerie_page)
        self._stack.addWidget(self._sdk_page)
        self._stack.addWidget(self._backend_page)
        root.addWidget(self._stack, 1)
        root.addSpacing(12)

        # ---- Footer: dots + nav buttons -----------------------------
        footer = QHBoxLayout()
        footer.setSpacing(8)

        self._dots: List[QLabel] = []
        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(6)
        for i in range(len(self.PAGE_TITLES)):
            dot = QLabel("●")
            dot.setObjectName("dotActive" if i == 0 else "dotInactive")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._dots.append(dot)
            dots_layout.addWidget(dot)
        footer.addLayout(dots_layout)
        footer.addStretch()

        self._btn_skip = QPushButton("Skip Setup")
        self._btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_skip.clicked.connect(self._on_skip)
        footer.addWidget(self._btn_skip)

        self._btn_back = QPushButton("Back")
        self._btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_back.clicked.connect(self._go_back)
        self._btn_back.setEnabled(False)
        footer.addWidget(self._btn_back)

        self._btn_next = QPushButton("Next")
        self._btn_next.setObjectName("primaryButton")
        self._btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_next.clicked.connect(self._go_next)
        footer.addWidget(self._btn_next)

        root.addLayout(footer)

    # ---- Navigation --------------------------------------------------
    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        total = self._stack.count()
        self._btn_back.setEnabled(idx > 0)
        is_last = idx == total - 1
        self._btn_next.setText("Finish" if is_last else "Next")

        for i, dot in enumerate(self._dots):
            dot.setObjectName("dotActive" if i == idx else "dotInactive")
            dot.setStyleSheet(dot.styleSheet())
        self.setStyleSheet(self.styleSheet())

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        total = self._stack.count()
        if idx < total - 1:
            self._stack.setCurrentIndex(idx + 1)
            self._update_nav()
        else:
            self._finish()

    def _apply_data_dir_choice(self) -> None:
        """Persist the DataDirectoryPage selection BEFORE any USER_CONFIG
        write happens. Writes ``~/.unitport_paths.ini`` for custom paths,
        deletes it for default. Then reloads ``Paths`` + ``Config`` so
        ``save_setup_state`` (the very next write) lands at the new
        location."""
        chosen = self._data_dir_page.get_data_dir()
        if self._data_dir_page.is_default():
            _clear_user_config_shim()
            log_info(
                f"[wizard] data directory: default ({chosen}) — "
                "no shim written"
            )
        else:
            try:
                chosen.mkdir(parents=True, exist_ok=True)
                _write_user_config_shim(chosen)
                log_info(f"[wizard] data directory set to {chosen}")
            except OSError as exc:
                log_warning(
                    f"[wizard] could not initialise {chosen}: {exc}; "
                    "falling back to default"
                )
                _clear_user_config_shim()
        # Re-resolve Tier-B paths so USER_CONFIG_DIR / USER_INI / etc.
        # reflect the new choice; flush Config's user.ini parser cache
        # so a later overlay read doesn't return values from the old
        # path. Safe to call multiple times (idempotent).
        _reload_paths()

    def _on_skip(self) -> None:
        """User skipped the wizard -- fall back to legacy 'install
        everything' default. PostSetupTask treats ``skipped=True`` as
        the trigger for full menagerie clone + all registered SDKs.
        """
        self._apply_data_dir_choice()
        self._result_data = {"skipped": True}
        save_setup_state({"completed": True, "skipped": True})
        self.completed.emit(self._result_data)
        self.accept()

    def _finish(self) -> None:
        """Collect selections and close."""
        self._apply_data_dir_choice()
        self._result_data = {
            "skipped": False,
            "user_config_dir": str(self._data_dir_page.get_data_dir()),
            "menagerie_folders": self._menagerie_page.get_selected_folders(),
            "sdks": [
                {"brand": b, "key": k, "url": u}
                for b, k, u in self._sdk_page.get_selected_sdks()
            ],
            "backend": self._backend_page.get_config(),
        }
        save_setup_state({
            "completed": True,
            "skipped": False,
            "selections": self._result_data,
        })
        self.completed.emit(self._result_data)
        self.accept()

    def get_selections(self) -> Dict[str, Any]:
        return self._result_data


__all__ = [
    "InstallConfigWizard",
    "DataDirectoryPage",
    "MenagerieSelectPage",
    "SdkSelectPage",
    "BackendPage",
    "load_setup_state",
    "save_setup_state",
    "setup_completed",
]
