#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom installation wizard shown during first startup.

Three pages:
  1. MuJoCo Menagerie model selector  (sparse-checkout from GitHub)
  2. SDK project selector              (per-brand SDK repos)
  3. Backend environment               (MuJoCo / Isaac Lab path registration)
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.system.core.theme_manager import get_color, get_font
from src.system.models.model_registry import (
    canonical_brand_ids,
    get_brand_spec,
    sdk_project_keys_for_brand,
    sdk_project_url,
)

# ── Paths ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # → NEW/
_SRC_ROOT = _PROJECT_ROOT / "src" / "system"
_CONFIG_DIR = _PROJECT_ROOT / "src" / "config"
_SETUP_STATE_PATH = _CONFIG_DIR / "setup_state.json"
_MENAGERIE_DIR = _SRC_ROOT / "runtime" / "simulation" / "mujoco" / "menagerie"
_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
_MENAGERIE_API_URL = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents"

# Known menagerie robot folders (fallback if GitHub API is unreachable)
_MENAGERIE_FALLBACK_ITEMS: List[Dict[str, str]] = [
    {"name": "agility_cassie",   "description": "Agility Robotics Cassie"},
    {"name": "agility_digit",    "description": "Agility Robotics Digit"},
    {"name": "anybotics_anymal_b", "description": "ANYbotics ANYmal B"},
    {"name": "anybotics_anymal_c", "description": "ANYbotics ANYmal C"},
    {"name": "franka_emika_panda", "description": "Franka Emika Panda"},
    {"name": "franka_fr3",       "description": "Franka FR3"},
    {"name": "google_barkour_v0", "description": "Google Barkour v0"},
    {"name": "google_barkour_vb", "description": "Google Barkour vB"},
    {"name": "google_robot",     "description": "Google Robot"},
    {"name": "hello_robot_stretch", "description": "Hello Robot Stretch"},
    {"name": "kinova_gen3",      "description": "Kinova Gen3"},
    {"name": "kuka_iiwa_14",     "description": "KUKA iiwa 14"},
    {"name": "robotiq_2f85",     "description": "Robotiq 2F-85"},
    {"name": "robotis_op3",      "description": "ROBOTIS OP3"},
    {"name": "shadow_hand",      "description": "Shadow Dexterous Hand"},
    {"name": "trossen_vx300s",   "description": "Trossen VX-300s"},
    {"name": "ufactory_lite6",   "description": "UFactory Lite 6"},
    {"name": "ufactory_xarm7",   "description": "UFactory xArm 7"},
    {"name": "unitree_a1",       "description": "Unitree A1"},
    {"name": "unitree_go1",      "description": "Unitree Go1"},
    {"name": "unitree_go2",      "description": "Unitree Go2"},
    {"name": "unitree_g1",       "description": "Unitree G1"},
    {"name": "unitree_h1",       "description": "Unitree H1"},
    {"name": "unitree_z1",       "description": "Unitree Z1"},
    {"name": "universal_robots_ur5e", "description": "Universal Robots UR5e"},
    {"name": "wonik_allegro",    "description": "Wonik Allegro Hand"},
]

# Models that UnitPort directly supports — pre-checked by default
_UNITPORT_SUPPORTED_MENAGERIE = {
    "unitree_go2", "unitree_a1", "unitree_g1", "unitree_h1",
}


# ═══════════════════════════════════════════════════════════════════════════
# Background worker: fetch menagerie directory listing from GitHub
# ═══════════════════════════════════════════════════════════════════════════

class _MenagerieFetchWorker(QThread):
    """Fetch the top-level directory listing of mujoco_menagerie from GitHub API."""
    finished = Signal(list)  # List[Dict[str, str]]  name + description

    def run(self) -> None:
        items = self._fetch_from_api()
        if not items:
            items = list(_MENAGERIE_FALLBACK_ITEMS)
        self.finished.emit(items)

    @staticmethod
    def _fetch_from_api() -> List[Dict[str, str]]:
        """Query GitHub Contents API; return list of dicts {name, description}."""
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                _MENAGERIE_API_URL,
                headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "UnitPort"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            # Filter: directories only, skip hidden / metadata dirs
            skip_prefixes = (".", "_", "test")
            items: List[Dict[str, str]] = []
            for entry in data:
                if entry.get("type") != "dir":
                    continue
                name = entry["name"]
                if any(name.startswith(p) for p in skip_prefixes):
                    continue
                # derive a human-readable description from folder name
                description = name.replace("_", " ").title()
                items.append({"name": name, "description": description})
            items.sort(key=lambda x: x["name"])
            return items
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════
# Setup state persistence
# ═══════════════════════════════════════════════════════════════════════════

def load_setup_state() -> Dict[str, Any]:
    if _SETUP_STATE_PATH.exists():
        try:
            return json.loads(_SETUP_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_setup_state(state: Dict[str, Any]) -> None:
    _SETUP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETUP_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def setup_completed() -> bool:
    """Return True if the user has already completed the setup wizard."""
    return load_setup_state().get("completed", False)


# ═══════════════════════════════════════════════════════════════════════════
# Shared stylesheet helpers
# ═══════════════════════════════════════════════════════════════════════════

def _wizard_stylesheet() -> str:
    bg = get_color("panel_bg", "#2a2c33")
    bg_main = get_color("bg", "#272829")
    text = get_color("text_primary", "#ffffff")
    text2 = get_color("text_secondary", "#cccccc")
    muted = get_color("text_muted", "#888888")
    border = get_color("border", "#444444")
    btn_bg = get_color("button_bg", "#2a2c33")
    btn_hover = get_color("button_hover", "#525252")
    btn_border = get_color("button_border", "#4b5563")
    btn_text = get_color("button_text", "#e5e7eb")
    input_bg = get_color("input_bg", "#0f1115")
    accent = "#FF7844"

    return f"""
        QDialog#SetupWizard {{
            background: {bg_main};
        }}

        /* ── Title / Subtitle ─────────────────────────── */
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

        /* ── Scroll area ──────────────────────────────── */
        QScrollArea {{
            background: transparent;
            border: 1px solid {border};
            border-radius: 6px;
        }}
        QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}

        /* ── Checkboxes ───────────────────────────────── */
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

        /* ── Group boxes ──────────────────────────────── */
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

        /* ── Buttons ──────────────────────────────────── */
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

        /* ── Input ────────────────────────────────────── */
        QLineEdit {{
            background: {input_bg};
            color: {text};
            border: 1px solid {btn_border};
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 13px;
        }}
        QLineEdit:focus {{
            border-color: {accent};
        }}

        /* ── Progress bar ─────────────────────────────── */
        QProgressBar {{
            background: {input_bg};
            border: 1px solid {border};
            border-radius: 4px;
            text-align: center;
            color: {text};
            font-size: 11px;
            height: 18px;
        }}
        QProgressBar::chunk {{
            background: {accent};
            border-radius: 3px;
        }}

        /* ── Navigation dots ──────────────────────────── */
        QLabel#dotActive {{
            color: {accent};
            font-size: 18px;
        }}
        QLabel#dotInactive {{
            color: {muted};
            font-size: 18px;
        }}

        /* ── Separator ────────────────────────────────── */
        QFrame#separator {{
            background: {border};
            max-height: 1px;
        }}
    """


# ═══════════════════════════════════════════════════════════════════════════
# Page 1: MuJoCo Menagerie Selector
# ═══════════════════════════════════════════════════════════════════════════

class MenageriePage(QWidget):
    """Select which mujoco_menagerie model folders to download."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checkboxes: Dict[str, QCheckBox] = {}
        self._items_loaded = False
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

        # Toolbar: select all / deselect all / refresh
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

        # Scroll area for checkboxes
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_layout.setSpacing(2)
        self._scroll_layout.addStretch()
        self._scroll.setWidget(self._scroll_content)
        layout.addWidget(self._scroll, 1)

        # Loading indicator
        self._loading_label = QLabel("Loading model list from GitHub …")
        self._loading_label.setObjectName("pageHint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_label)
        self._loading_label.hide()

        # Worker
        self._worker: Optional[_MenagerieFetchWorker] = None

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not self._items_loaded:
            self._start_fetch()

    def _start_fetch(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._loading_label.show()
        self._btn_refresh.setEnabled(False)
        self._worker = _MenagerieFetchWorker()
        self._worker.finished.connect(self._on_items_loaded)
        self._worker.start()

    @Slot(list)
    def _on_items_loaded(self, items: List[Dict[str, str]]) -> None:
        self._loading_label.hide()
        self._btn_refresh.setEnabled(True)
        self._populate(items)
        self._items_loaded = True

    def _populate(self, items: List[Dict[str, str]]) -> None:
        # Clear existing
        for cb in self._checkboxes.values():
            self._scroll_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

        # Remove the stretch
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for entry in items:
            name = entry["name"]
            desc = entry.get("description", name)
            cb = QCheckBox(f"{desc}  ({name})")
            cb.setProperty("folder_name", name)
            # Pre-check UnitPort-supported models
            if name in _UNITPORT_SUPPORTED_MENAGERIE:
                cb.setChecked(True)
            self._checkboxes[name] = cb
            self._scroll_layout.addWidget(cb)

        self._scroll_layout.addStretch()

    def _select_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def get_selected_folders(self) -> List[str]:
        return [name for name, cb in self._checkboxes.items() if cb.isChecked()]


# ═══════════════════════════════════════════════════════════════════════════
# Page 2: SDK Selector
# ═══════════════════════════════════════════════════════════════════════════

class SdkPage(QWidget):
    """Select which SDK repos to clone / install."""

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

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(12)

        for brand_id in canonical_brand_ids():
            brand_spec = get_brand_spec(brand_id)
            if brand_spec is None:
                continue
            project_keys = sdk_project_keys_for_brand(brand_id)
            if not project_keys:
                continue

            group = QGroupBox(brand_spec.display_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)

            self._brand_groups[brand_id] = {}
            for key in project_keys:
                url = sdk_project_url(brand_id, key)
                label = key.replace("_", " ").replace("-", " ").title()
                cb = QCheckBox(f"{label}")
                cb.setToolTip(url)
                cb.setProperty("brand", brand_id)
                cb.setProperty("project_key", key)
                cb.setChecked(True)  # default: all checked
                self._brand_groups[brand_id][key] = cb
                group_layout.addWidget(cb)

            content_layout.addWidget(group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def get_selected_sdks(self) -> List[Tuple[str, str, str]]:
        """Return list of (brand_id, project_key, url) for checked items."""
        result: List[Tuple[str, str, str]] = []
        for brand_id, cbs in self._brand_groups.items():
            for key, cb in cbs.items():
                if cb.isChecked():
                    url = sdk_project_url(brand_id, key)
                    result.append((brand_id, key, url))
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Page 3: Backend Environment
# ═══════════════════════════════════════════════════════════════════════════

class BackendPage(QWidget):
    """Configure backend simulation environments: MuJoCo, Isaac Lab, loco-mujoco."""

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

        # ── MuJoCo ────────────────────────────────────────────────────────
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

        # ── Isaac Lab ─────────────────────────────────────────────────────
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

        # Option A: Install from scratch (radio button)
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

        # Install-path sub-panel (shown when "install" radio is selected)
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
        self._btn_browse_install.setStyleSheet("padding: 0px;")
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

        # Option B: Locate existing (radio button)
        self._rb_isaac_locate = QRadioButton(
            "Locate existing Isaac Lab installation"
        )
        self._rb_isaac_locate.setToolTip(
            "Point to an existing Isaac Lab root directory.\n"
            "The path will be registered for training calls."
        )
        self._rb_isaac_locate.toggled.connect(self._on_isaac_mode_toggled)
        opts_layout.addWidget(self._rb_isaac_locate)

        # Locate-path sub-panel (shown when "locate" radio is selected)
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
        self._btn_browse_isaac.setStyleSheet("padding: 0px;")
        self._btn_browse_isaac.setFixedWidth(90)
        self._btn_browse_isaac.clicked.connect(self._browse_isaaclab)
        locate_row.addWidget(self._btn_browse_isaac)
        locate_path_layout.addLayout(locate_row)

        self._isaac_path_status = QLabel("")
        self._isaac_path_status.setObjectName("pageHint")
        locate_path_layout.addWidget(self._isaac_path_status)

        self._locate_path_panel.hide()
        opts_layout.addWidget(self._locate_path_panel)

        # Option C: Deploy to remote server via SSH (radio button)
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

        # Cloud-deploy sub-panel (shown when "cloud" radio is selected)
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

        # Server name
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(QLabel("Server name:"))
        self._cloud_name_edit = QLineEdit()
        self._cloud_name_edit.setPlaceholderText("My GPU Server")
        name_row.addWidget(self._cloud_name_edit, 1)
        cloud_layout.addLayout(name_row)

        # Host + port
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

        # Username
        user_row = QHBoxLayout()
        user_row.setSpacing(6)
        user_row.addWidget(QLabel("Username:"))
        self._cloud_user_edit = QLineEdit()
        self._cloud_user_edit.setPlaceholderText("root")
        user_row.addWidget(self._cloud_user_edit, 1)
        cloud_layout.addLayout(user_row)

        # Auth method
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

        # Key path
        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        key_row.addWidget(QLabel("Key file:"))
        self._cloud_key_edit = QLineEdit()
        self._cloud_key_edit.setPlaceholderText("~/.ssh/id_rsa")
        self._cloud_key_edit.setText("~/.ssh/id_rsa")
        key_row.addWidget(self._cloud_key_edit, 1)
        self._btn_browse_key = QPushButton("Browse ...")
        self._btn_browse_key.setStyleSheet("padding: 0px;")
        self._btn_browse_key.setFixedWidth(90)
        self._btn_browse_key.clicked.connect(self._browse_ssh_key)
        key_row.addWidget(self._btn_browse_key)
        cloud_layout.addLayout(key_row)

        # Password (shown only when password auth selected)
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

        # Toggle key/password visibility
        self._cloud_auth_key.toggled.connect(self._on_cloud_auth_toggled)

        # Remote install dir
        remote_row = QHBoxLayout()
        remote_row.setSpacing(6)
        remote_row.addWidget(QLabel("Remote path:"))
        self._cloud_remote_dir_edit = QLineEdit()
        self._cloud_remote_dir_edit.setText("~/UnitPort/engines/isaac")
        self._cloud_remote_dir_edit.setPlaceholderText(
            "~/UnitPort/engines/isaac"
        )
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

    def _on_isaac_mode_toggled(self, _checked: bool) -> None:
        """Show/hide sub-panels based on which radio button is active."""
        self._install_path_panel.setVisible(self._rb_isaac_install.isChecked())
        self._locate_path_panel.setVisible(self._rb_isaac_locate.isChecked())
        self._cloud_panel.setVisible(self._rb_isaac_cloud.isChecked())

    def _on_cloud_auth_toggled(self, key_checked: bool) -> None:
        """Toggle key-file vs password fields."""
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
        # Validate: look for IsaacLab marker files
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
            "isaaclab_enabled": self._cb_isaaclab.isChecked(),
            "isaaclab_install": is_install,
            "isaaclab_locate": is_locate,
            "isaaclab_cloud_deploy": is_cloud,
            # For "install" mode: the local target directory
            # For "locate" mode: the existing Isaac Lab root directory
            "isaaclab_path": (
                self._install_dir_edit.text().strip() if is_install
                else self._isaac_path_edit.text().strip()
            ),
        }

        # Cloud deploy SSH config (only populated when Option C is selected)
        if is_cloud:
            cfg["cloud_ssh"] = {
                "server_name": self._cloud_name_edit.text().strip()
                               or self._cloud_host_edit.text().strip(),
                "host": self._cloud_host_edit.text().strip(),
                "port": int(self._cloud_port_edit.text() or 22),
                "username": self._cloud_user_edit.text().strip(),
                "auth_method": "key" if self._cloud_auth_key.isChecked()
                               else "password",
                "private_key_path": self._cloud_key_edit.text().strip(),
                "password": self._cloud_pwd_edit.text(),
                "remote_install_dir": (
                    self._cloud_remote_dir_edit.text().strip()
                    or "~/UnitPort/engines/isaac"
                ),
            }

        return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Main Wizard Dialog
# ═══════════════════════════════════════════════════════════════════════════

class SetupWizard(QDialog):
    """Three-page custom installation wizard shown on first launch."""

    PAGE_TITLES = ["Menagerie Models", "SDK Packages", "Backend Environment"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SetupWizard")
        self.setWindowTitle("UnitPort — Custom Setup")
        self.setMinimumSize(720, 560)
        self.resize(800, 620)
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

        # ── Header ────────────────────────────────────────────────────────
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("UnitPort Setup")
        title.setObjectName("wizardTitle")
        header.addWidget(title)

        subtitle = QLabel("Choose which components to install. You can change these later in Settings.")
        subtitle.setObjectName("wizardSubtitle")
        header.addWidget(subtitle)
        root.addLayout(header)
        root.addSpacing(12)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        root.addWidget(sep)
        root.addSpacing(12)

        # ── Page stack ────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._menagerie_page = MenageriePage()
        self._sdk_page = SdkPage()
        self._backend_page = BackendPage()
        self._stack.addWidget(self._menagerie_page)
        self._stack.addWidget(self._sdk_page)
        self._stack.addWidget(self._backend_page)
        root.addWidget(self._stack, 1)
        root.addSpacing(12)

        # ── Footer: dots + nav buttons ────────────────────────────────────
        footer = QHBoxLayout()
        footer.setSpacing(8)

        # Page indicator dots
        self._dots: List[QLabel] = []
        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(6)
        for i in range(3):
            dot = QLabel("●")
            dot.setObjectName("dotActive" if i == 0 else "dotInactive")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._dots.append(dot)
            dots_layout.addWidget(dot)
        footer.addLayout(dots_layout)
        footer.addStretch()

        # Skip button
        self._btn_skip = QPushButton("Skip Setup")
        self._btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_skip.clicked.connect(self._on_skip)
        footer.addWidget(self._btn_skip)

        # Back / Next / Finish
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

    # ── Navigation ────────────────────────────────────────────────────────

    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        total = self._stack.count()
        self._btn_back.setEnabled(idx > 0)
        is_last = idx == total - 1
        self._btn_next.setText("Finish" if is_last else "Next")

        for i, dot in enumerate(self._dots):
            dot.setObjectName("dotActive" if i == idx else "dotInactive")
            # Force stylesheet re-evaluation
            dot.setStyleSheet(dot.styleSheet())
        # Reapply stylesheet for dots
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

    def _on_skip(self) -> None:
        """Skip wizard — use defaults (download everything)."""
        self._result_data = {"skipped": True}
        save_setup_state({"completed": True, "skipped": True})
        self.accept()

    def _finish(self) -> None:
        """Collect selections and close."""
        self._result_data = {
            "skipped": False,
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
        self.accept()

    def get_selections(self) -> Dict[str, Any]:
        return self._result_data
