# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
    Config, I18n, Paths, i18n_bind, log_info, log_warning, push_data,
    read_data, save_data, tr,
)

from application.service.user_workspace import (
    apply_machine_locale_preference as _apply_machine_locale_preference,
    reload_paths as _reload_paths,
    set_workspace_root as _set_workspace_root,
)

from application.service.installers import (
    default_isaac_lab_install_root,
)
from application.service.installers import eula as _eula
from application.service.installers.custom_mods_manifest import (
    CustomModEntry,
    entry_already_installed,
    load_manifest_entries,
)
from application.service.models import menagerie_manager as mm
from application.service.models import sdk_manager as sm

from .menagerie_card import IconFetchWorker, MenagerieCardGrid


# Menagerie folders pre-checked on first install. Chosen to cover the
# robots most likely to be touched on a fresh UnitPort setup; the user
# can deselect any of them inside MenagerieSelectPage. Names match the
# top-level directory names in ``google-deepmind/mujoco_menagerie``.
_DEFAULT_PRE_CHECKED_MENAGERIE = frozenset({
    "unitree_a1",
    "unitree_g1",
    "unitree_go1",
    "unitree_go2",
    "boston_dynamics_spot",
})


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


def _setup_state_path() -> Optional[Path]:
    """Resolve the on-disk setup_state.json location.

    Returns ``None`` when the workspace root is not yet configured — a
    fresh install has no place to store wizard state until the user picks
    a data directory. Callers must handle ``None`` by treating the wizard
    as not-yet-run (``setup_completed()`` returns False, ``load_setup_state``
    returns an empty dict). ``save_setup_state`` is only ever called from
    the wizard's ``_finish``/``_on_skip``, both of which establish the
    workspace via ``_apply_data_dir_choice`` first.
    """
    from application.service.user_workspace import read_workspace_root
    root = read_workspace_root()
    if root is None:
        return None
    return root / _SETUP_STATE_REL


def load_setup_state() -> Dict[str, Any]:
    path = _setup_state_path()
    if path is None or not path.exists():
        return {}
    data = read_data(path)
    return data if isinstance(data, dict) else {}


def save_setup_state(state: Dict[str, Any]) -> None:
    path = _setup_state_path()
    if path is None:
        log_warning(
            "[wizard] cannot persist setup_state.json: workspace root not "
            "configured. Wizard's DataDirectoryPage must run first."
        )
        return
    if not save_data(path, state):
        log_warning("[wizard] failed to persist setup_state.json")


def setup_completed() -> bool:
    """Return True iff the user has already completed (or skipped) the wizard.

    Returns False when the workspace root is not yet configured — that
    state, by definition, means the wizard has never run on this install.
    """
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
    """Pick where UnitPort stores user data (WORKSPACE root → USER_CONFIG_DIR).

    There is **no default** here — the user must enter (or browse to) a path
    explicitly. The SDK does not fall back to ``~/UnitPort`` anywhere, and
    refuses to materialise a default workspace anywhere else either: any
    code that would need a fallback path is a boot-order bug.

    On finish the chosen path is written into ``system.ini[Workspace].root``
    (and ``[Resources].user_config_dir`` is derived from it), then
    ``Paths`` is reloaded so the very next write (``save_setup_state``)
    lands at the right location.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel()
        title.setObjectName("pageTitle")
        i18n_bind(title, "setText", "setup.data_dir.title", default="Data Directory")
        layout.addWidget(title)

        hint = QLabel()
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        i18n_bind(
            hint,
            "setText",
            "setup.data_dir.hint",
            default=(
                "Choose where UnitPort stores all per-user data: projects, "
                "exported policies, login tokens, telemetry caches, "
                "downloaded engines, and the user.ini overlay. The "
                "application install folder is never written to at "
                "runtime, so this directory is what survives across "
                "redeploys / git pulls. You must pick an absolute path — "
                "there is no built-in default."
            ),
        )
        layout.addWidget(hint)

        # Path row
        row = QHBoxLayout()
        row.setSpacing(6)
        label_dir = QLabel()
        i18n_bind(
            label_dir, "setText", "setup.data_dir.label_dir", default="UserConfig Dir:"
        )
        row.addWidget(label_dir)
        self._dir_edit = QLineEdit()
        self._dir_edit.setText(self._current_value_for_init())
        self._dir_edit.setPlaceholderText(
            tr(
                "setup.data_dir.placeholder",
                "Pick an absolute path (e.g. D:\\UnitPort)",
            )
        )
        row.addWidget(self._dir_edit, 1)
        self._btn_browse = QPushButton()
        self._btn_browse.setFixedWidth(90)
        self._btn_browse.clicked.connect(self._browse)
        i18n_bind(
            self._btn_browse, "setText", "setup.data_dir.btn_browse", default="Browse ..."
        )
        row.addWidget(self._btn_browse)
        layout.addLayout(row)

        note = QLabel()
        note.setObjectName("pageHint")
        note.setStyleSheet("color: #888; font-size: 11px;")
        note.setWordWrap(True)
        i18n_bind(
            note,
            "setText",
            "setup.data_dir.note",
            default=(
                "This path is the WORKSPACE root: per-account folders live "
                "underneath it (one per signed-in user, plus _guest/). "
                "Pick an absolute path with enough space (engine downloads "
                "may exceed 30 GB)."
            ),
        )
        layout.addWidget(note)

        self._status = QLabel("")
        self._status.setObjectName("pageHint")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._refresh_status()
        self._dir_edit.textChanged.connect(lambda _t: self._refresh_status())

        layout.addStretch(1)

    def _current_value_for_init(self) -> str:
        """Pre-fill from prior wizard runs.

        Prefer ``system.ini[Workspace].root`` (the authoritative location
        under the new design); fall back to a legacy shim only if it still
        exists from an upgrade-in-progress install. Empty string when
        neither has ever been set.
        """
        try:
            from application.service.user_workspace import read_workspace_root
            existing = read_workspace_root()
            if existing is not None:
                return str(existing)
        except Exception:
            pass
        try:
            existing = Paths._read_bootstrap_user_config_dir()
            if existing:
                return str(Path(existing).expanduser())
        except Exception:
            pass
        return ""

    def _browse(self) -> None:
        start = self._dir_edit.text().strip()
        if not start:
            start = str(Path.cwd())
        path = QFileDialog.getExistingDirectory(
            self, "Choose UnitPort Data Directory", start
        )
        if path:
            self._dir_edit.setText(path)

    def _refresh_status(self) -> None:
        raw = self._dir_edit.text().strip()
        if not raw:
            self._status.setText(
                tr(
                    "setup.data_dir.status_required",
                    "Required — pick an absolute path for the workspace root.",
                )
            )
            self._status.setStyleSheet("color: #FF6B6B; font-size: 12px;")
            return
        try:
            chosen = Path(raw).expanduser()
        except Exception:
            self._status.setText(tr("setup.data_dir.status_invalid", "Invalid path"))
            self._status.setStyleSheet("color: #FFC107; font-size: 12px;")
            return
        if not chosen.is_absolute():
            self._status.setText(
                tr(
                    "setup.data_dir.status_relative",
                    "Path must be absolute (got: {path}).",
                ).replace("{path}", str(chosen))
            )
            self._status.setStyleSheet("color: #FFC107; font-size: 12px;")
            return
        if chosen.exists():
            self._status.setText(
                tr(
                    "setup.data_dir.status_custom_exists",
                    "Workspace location — already exists at {path}",
                ).replace("{path}", str(chosen))
            )
            self._status.setStyleSheet("color: #00D26A; font-size: 12px;")
        else:
            self._status.setText(
                tr(
                    "setup.data_dir.status_custom_create",
                    "Workspace location — will be created at {path}",
                ).replace("{path}", str(chosen))
            )
            self._status.setStyleSheet("color: #888; font-size: 12px;")

    def get_data_dir(self) -> Optional[Path]:
        """Return the user-chosen absolute directory, or ``None`` when blank.

        ``None`` means the user has not picked anything; the wizard's
        ``_apply_data_dir_choice`` refuses to advance in that state — there
        is no fallback default.
        """
        raw = self._dir_edit.text().strip()
        if not raw:
            return None
        try:
            chosen = Path(raw).expanduser()
        except Exception:
            return None
        if not chosen.is_absolute():
            return None
        return chosen


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

        title = QLabel()
        title.setObjectName("pageTitle")
        i18n_bind(
            title, "setText", "setup.menagerie.title", default="MuJoCo Menagerie Models"
        )
        layout.addWidget(title)

        hint = QLabel()
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        i18n_bind(
            hint,
            "setText",
            "setup.menagerie.hint",
            default=(
                "Select the robot models you need from mujoco_menagerie. "
                "UnitPort-supported models are pre-selected. Others are optional."
            ),
        )
        layout.addWidget(hint)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._btn_select_all = QPushButton()
        self._btn_deselect_all = QPushButton()
        self._btn_refresh = QPushButton()
        i18n_bind(
            self._btn_select_all,
            "setText",
            "setup.menagerie.btn_select_all",
            default="Select All",
        )
        i18n_bind(
            self._btn_deselect_all,
            "setText",
            "setup.menagerie.btn_deselect_all",
            default="Deselect All",
        )
        i18n_bind(
            self._btn_refresh,
            "setText",
            "setup.menagerie.btn_refresh",
            default="Refresh from GitHub",
        )
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

        self._loading_label = QLabel()
        self._loading_label.setObjectName("pageHint")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        i18n_bind(
            self._loading_label,
            "setText",
            "setup.menagerie.loading",
            default="Loading model list from GitHub …",
        )
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
        # Pre-check everything UnitPort already maps a robot for, PLUS the
        # "default install" set (Unitree A1/G1/GO1/GO2 + Boston Dynamics
        # Spot). Filtered against ``names`` so we don't pre-check a folder
        # that doesn't actually exist in the upstream tree.
        available = set(names)
        pre_checked = (
            self._registered_dirs | _DEFAULT_PRE_CHECKED_MENAGERIE
        ) & available
        self._grid.populate(
            names,
            installed=self._installed_dirs,
            registered=self._registered_dirs,
            pre_checked=pre_checked,
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

        title = QLabel()
        title.setObjectName("pageTitle")
        i18n_bind(title, "setText", "setup.sdk.title", default="SDK Packages")
        layout.addWidget(title)

        hint = QLabel()
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        i18n_bind(
            hint,
            "setText",
            "setup.sdk.hint",
            default=(
                "Select the robot SDK packages to download. "
                "Only checked items will be cloned. You can install more later from Settings."
            ),
        )
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
        # EULA ids the user has accepted for the Isaac Lab fresh-install
        # path during this wizard session. Populated by the EulaDialog
        # bound to ``_rb_isaac_install.toggled``; echoed into
        # ``selections.backend.eula_accepted_ids`` from :meth:`get_config`
        # so PostSetupTask can defensively re-verify before dispatching
        # the long-running install task. Persisted records live in
        # ``Paths.USER_CONFIG_DIR/eula_acceptance.json`` (see
        # ``application.service.installers.eula``); this list is the
        # short-lived in-session echo, not the source of truth.
        self._isaac_eula_accepted_ids: List[str] = []
        # Re-entrancy guard for the install radio handler — without this
        # the programmatic ``setChecked(False)`` we issue when the user
        # rejects EULA re-enters _on_isaac_install_radio_toggled and
        # loops.
        self._suppress_install_radio_handler: bool = False
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel()
        title.setObjectName("pageTitle")
        i18n_bind(title, "setText", "setup.backend.title", default="Backend Environment")
        layout.addWidget(title)

        hint = QLabel()
        hint.setObjectName("pageHint")
        hint.setWordWrap(True)
        i18n_bind(
            hint,
            "setText",
            "setup.backend.hint",
            default=(
                "Configure simulation backends for training and execution. "
                "Uncheck items you don't need to speed up setup."
            ),
        )
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(16)

        # ---- MuJoCo --------------------------------------------------
        mujoco_group = QGroupBox()
        i18n_bind(
            mujoco_group, "setTitle", "setup.backend.group_mujoco", default="MuJoCo"
        )
        mujoco_layout = QVBoxLayout(mujoco_group)
        mujoco_layout.setSpacing(6)
        self._cb_mujoco = QCheckBox()
        self._cb_mujoco.setChecked(True)
        i18n_bind(
            self._cb_mujoco,
            "setText",
            "setup.backend.cb_mujoco",
            default="Install MuJoCo (pip package)",
        )
        i18n_bind(
            self._cb_mujoco,
            "setToolTip",
            "setup.backend.cb_mujoco_tip",
            default="Required for MuJoCo-based simulation and training.",
        )
        mujoco_layout.addWidget(self._cb_mujoco)
        self._cb_loco_mujoco = QCheckBox()
        self._cb_loco_mujoco.setChecked(True)
        i18n_bind(
            self._cb_loco_mujoco,
            "setText",
            "setup.backend.cb_loco",
            default="Download loco-mujoco reference motion library",
        )
        i18n_bind(
            self._cb_loco_mujoco,
            "setToolTip",
            "setup.backend.cb_loco_tip",
            default=(
                "Community reference motion library for locomotion training. "
                "Cloned from github.com/robfiras/loco-mujoco."
            ),
        )
        mujoco_layout.addWidget(self._cb_loco_mujoco)
        content_layout.addWidget(mujoco_group)

        # ---- Reference Motion Libraries (custom_mods/installs.txt) ----
        # Hot-pluggable third-party clones. Manifest lives at
        # custom_mods/installs.txt; each entry deploys under
        # custom_mods/motions/<name>/. All checkboxes default off --
        # these are optional extras, not part of the default install.
        self._custom_mod_checkboxes: Dict[str, QCheckBox] = {}
        self._custom_mod_entries: List[CustomModEntry] = []
        try:
            self._custom_mod_entries = load_manifest_entries()
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[wizard] failed to load custom_mods manifest: {exc}"
            )
            self._custom_mod_entries = []

        if self._custom_mod_entries:
            mods_group = QGroupBox()
            i18n_bind(
                mods_group,
                "setTitle",
                "setup.backend.group_custom_mods",
                default="Reference Motion Libraries (optional)",
            )
            mods_layout = QVBoxLayout(mods_group)
            mods_layout.setSpacing(6)

            mods_hint = QLabel()
            mods_hint.setObjectName("pageHint")
            mods_hint.setWordWrap(True)
            mods_hint.setStyleSheet("color: #888; font-size: 11px;")
            i18n_bind(
                mods_hint,
                "setText",
                "setup.backend.custom_mods_hint",
                default=(
                    "Optional third-party motion / AMP reference repos. "
                    "Each is shallow-cloned into custom_mods/motions/. "
                    "Leave unchecked to skip — you can install later "
                    "by editing custom_mods/installs.txt."
                ),
            )
            mods_layout.addWidget(mods_hint)

            for entry in self._custom_mod_entries:
                cb = QCheckBox()
                already = entry_already_installed(entry)
                templates_csv = ", ".join(entry.required_for)
                required_suffix = ""
                if entry.is_required:
                    suffix_tpl = tr(
                        "setup.backend.custom_mod_required_suffix",
                        default="(required for {templates})",
                    )
                    required_suffix = "  " + suffix_tpl.replace(
                        "{templates}", templates_csv
                    )
                if already:
                    already_tag = tr(
                        "setup.backend.custom_mod_already_installed",
                        default="(already installed)",
                    )
                    label_text = (
                        f"{entry.display_name}  {already_tag}{required_suffix}"
                    )
                    cb.setChecked(False)
                    cb.setEnabled(False)
                else:
                    label_text = f"{entry.display_name}{required_suffix}"
                    cb.setChecked(entry.is_required)
                cb.setText(label_text)
                tooltip = f"{entry.url}\n→ custom_mods/{entry.relative_path}"
                if entry.is_required:
                    tip_tpl = tr(
                        "setup.backend.custom_mod_required_tip",
                        default=(
                            "Required by canvas template(s): {templates}. "
                            "Uncheck only if you do not plan to use the "
                            "listed template — otherwise the template "
                            "will fail to load."
                        ),
                    )
                    tooltip += "\n" + tip_tpl.replace(
                        "{templates}", templates_csv
                    )
                cb.setToolTip(tooltip)
                cb.setProperty("custom_mod_key", entry.key)
                self._custom_mod_checkboxes[entry.key] = cb
                mods_layout.addWidget(cb)

            content_layout.addWidget(mods_group)

        # ---- ROS2 ----------------------------------------------------
        ros2_group = QGroupBox()
        i18n_bind(ros2_group, "setTitle", "setup.backend.group_ros2", default="ROS2")
        ros2_layout = QVBoxLayout(ros2_group)
        ros2_layout.setSpacing(6)
        self._cb_ros2 = QCheckBox()
        self._cb_ros2.setChecked(True)
        i18n_bind(
            self._cb_ros2,
            "setText",
            "setup.backend.cb_ros2",
            default="Install ROS2 support",
        )
        i18n_bind(
            self._cb_ros2,
            "setToolTip",
            "setup.backend.cb_ros2_tip",
            default=(
                "Enables ROS2-based brands (Unitree Go2, Mangdang Mini Pupper, …). "
                "A native ROS2 Humble install is reused when available; "
                "otherwise the Docker bridge image is built."
            ),
        )
        ros2_layout.addWidget(self._cb_ros2)
        ros2_hint = QLabel()
        ros2_hint.setObjectName("pageHint")
        ros2_hint.setWordWrap(True)
        ros2_hint.setStyleSheet("color: #888; font-size: 11px;")
        i18n_bind(
            ros2_hint,
            "setText",
            "setup.backend.ros2_note",
            default=(
                "Auto-detects a native ROS2 install (Humble preferred). When none "
                "is present, the UnitPort Docker bridge image is built instead — "
                "Docker Desktop / Docker Engine must be running."
            ),
        )
        ros2_layout.addWidget(ros2_hint)
        content_layout.addWidget(ros2_group)

        # ---- Isaac Lab ----------------------------------------------
        isaac_group = QGroupBox()
        i18n_bind(
            isaac_group,
            "setTitle",
            "setup.backend.group_isaac",
            default="Isaac Lab (NVIDIA)",
        )
        isaac_layout = QVBoxLayout(isaac_group)
        isaac_layout.setSpacing(6)

        self._cb_isaaclab = QCheckBox()
        self._cb_isaaclab.setChecked(False)
        i18n_bind(
            self._cb_isaaclab,
            "setText",
            "setup.backend.cb_isaac",
            default="Enable Isaac Lab integration",
        )
        i18n_bind(
            self._cb_isaaclab,
            "setToolTip",
            "setup.backend.cb_isaac_tip",
            default=(
                "Isaac Lab provides GPU-accelerated simulation via NVIDIA Isaac Sim. "
                "You can install it now or point to an existing installation."
            ),
        )
        self._cb_isaaclab.toggled.connect(self._on_isaaclab_toggled)
        isaac_layout.addWidget(self._cb_isaaclab)

        # Sub-options panel (shown after enable)
        self._isaac_options = QWidget()
        opts_layout = QVBoxLayout(self._isaac_options)
        opts_layout.setContentsMargins(24, 4, 0, 0)
        opts_layout.setSpacing(8)

        # Option A: Install from scratch
        self._rb_isaac_install = QRadioButton()
        i18n_bind(
            self._rb_isaac_install,
            "setText",
            "setup.backend.rb_isaac_install",
            default="Install Isaac Lab + Isaac Sim (from scratch)",
        )
        self._rb_isaac_install.setToolTip(
            "Automatically download and install Isaac Lab + Isaac Sim 5.1\n"
            "into a dedicated directory with its own Python virtual environment.\n"
            "Requires ~30 GB disk space, NVIDIA GPU, and internet connection."
        )
        self._rb_isaac_install.toggled.connect(self._on_isaac_mode_toggled)
        # Connect EULA gate AFTER the visibility handler so the panel is
        # already shown when the modal appears (or hidden on rejection).
        self._rb_isaac_install.toggled.connect(self._on_isaac_install_radio_toggled)
        opts_layout.addWidget(self._rb_isaac_install)

        self._install_path_panel = QWidget()
        install_path_layout = QVBoxLayout(self._install_path_panel)
        install_path_layout.setContentsMargins(24, 0, 0, 0)
        install_path_layout.setSpacing(4)
        install_hint = QLabel()
        install_hint.setObjectName("pageHint")
        install_hint.setWordWrap(True)
        i18n_bind(
            install_hint,
            "setText",
            "setup.backend.install_hint",
            default=(
                "Choose the installation directory. Isaac Lab + Isaac Sim will be "
                "installed with a dedicated venv inside this directory."
            ),
        )
        install_path_layout.addWidget(install_hint)

        install_row = QHBoxLayout()
        install_row.setSpacing(6)
        self._install_dir_edit = QLineEdit()
        # Default: PROJECT_ROOT/Engines/isaac_lab. See plan §A — the
        # "Engines/<backend_id>/" convention is for heavy backends (Isaac
        # Sim is ~30 GB) distinct from custom_mods/runtime/sdk_extensions/
        # which holds MB-scale brand SDK clones.
        self._install_dir_edit.setText(str(default_isaac_lab_install_root()))
        self._install_dir_edit.setPlaceholderText(
            tr("setup.backend.isaac_install_dir_ph", "Installation directory ...")
        )
        install_row.addWidget(self._install_dir_edit, 1)
        self._btn_browse_install = QPushButton()
        self._btn_browse_install.setFixedWidth(90)
        self._btn_browse_install.clicked.connect(self._browse_install_dir)
        i18n_bind(
            self._btn_browse_install,
            "setText",
            "setup.data_dir.btn_browse",
            default="Browse ...",
        )
        install_row.addWidget(self._btn_browse_install)
        install_path_layout.addLayout(install_row)

        install_note = QLabel()
        install_note.setObjectName("pageHint")
        install_note.setStyleSheet("color: #888; font-size: 11px;")
        install_note.setWordWrap(True)
        i18n_bind(
            install_note,
            "setText",
            "setup.backend.install_note",
            default=(
                "Isaac Sim 5.1.0 | IsaacLab (pinned) | PyTorch 2.7+cu128\n"
                "Disk: ~30 GB | Requires: NVIDIA GPU with CUDA 12.x driver"
            ),
        )
        install_path_layout.addWidget(install_note)
        self._install_path_panel.hide()
        opts_layout.addWidget(self._install_path_panel)

        # Option B: Locate existing
        self._rb_isaac_locate = QRadioButton()
        i18n_bind(
            self._rb_isaac_locate,
            "setText",
            "setup.backend.rb_isaac_locate",
            default="Locate existing Isaac Lab installation",
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
        self._isaac_path_edit.setPlaceholderText(
            tr("setup.backend.isaac_locate_dir_ph", "Isaac Lab root directory ...")
        )
        locate_row.addWidget(self._isaac_path_edit, 1)
        self._btn_browse_isaac = QPushButton()
        self._btn_browse_isaac.setFixedWidth(90)
        self._btn_browse_isaac.clicked.connect(self._browse_isaaclab)
        i18n_bind(
            self._btn_browse_isaac,
            "setText",
            "setup.data_dir.btn_browse",
            default="Browse ...",
        )
        locate_row.addWidget(self._btn_browse_isaac)
        locate_path_layout.addLayout(locate_row)

        self._isaac_path_status = QLabel("")
        self._isaac_path_status.setObjectName("pageHint")
        locate_path_layout.addWidget(self._isaac_path_status)
        self._locate_path_panel.hide()
        opts_layout.addWidget(self._locate_path_panel)

        # Option C: Cloud SSH deploy
        self._rb_isaac_cloud = QRadioButton()
        i18n_bind(
            self._rb_isaac_cloud,
            "setText",
            "setup.backend.rb_isaac_cloud",
            default="Deploy Isaac Lab to remote Linux server (SSH)",
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
        cloud_hint = QLabel()
        cloud_hint.setObjectName("pageHint")
        cloud_hint.setWordWrap(True)
        i18n_bind(
            cloud_hint,
            "setText",
            "setup.backend.cloud_hint",
            default=(
                "Enter SSH connection details for the remote GPU server. "
                "Isaac Lab will be installed via SSH + SFTP."
            ),
        )
        cloud_layout.addWidget(cloud_hint)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        _lbl_srv = QLabel()
        i18n_bind(
            _lbl_srv,
            "setText",
            "setup.backend.cloud_server_name",
            default="Server name:",
        )
        name_row.addWidget(_lbl_srv)
        self._cloud_name_edit = QLineEdit()
        self._cloud_name_edit.setPlaceholderText("My GPU Server")
        name_row.addWidget(self._cloud_name_edit, 1)
        cloud_layout.addLayout(name_row)

        host_row = QHBoxLayout()
        host_row.setSpacing(6)
        _lbl_host = QLabel()
        i18n_bind(
            _lbl_host, "setText", "setup.backend.cloud_host", default="Host:"
        )
        host_row.addWidget(_lbl_host)
        self._cloud_host_edit = QLineEdit()
        self._cloud_host_edit.setPlaceholderText("10.0.0.1 or gpu-server.example.com")
        host_row.addWidget(self._cloud_host_edit, 1)
        _lbl_port = QLabel()
        i18n_bind(
            _lbl_port, "setText", "setup.backend.cloud_port", default="Port:"
        )
        host_row.addWidget(_lbl_port)
        self._cloud_port_edit = QLineEdit()
        self._cloud_port_edit.setText("22")
        self._cloud_port_edit.setFixedWidth(60)
        host_row.addWidget(self._cloud_port_edit)
        cloud_layout.addLayout(host_row)

        user_row = QHBoxLayout()
        user_row.setSpacing(6)
        _lbl_user = QLabel()
        i18n_bind(
            _lbl_user, "setText", "setup.backend.cloud_user", default="Username:"
        )
        user_row.addWidget(_lbl_user)
        self._cloud_user_edit = QLineEdit()
        self._cloud_user_edit.setPlaceholderText("root")
        user_row.addWidget(self._cloud_user_edit, 1)
        cloud_layout.addLayout(user_row)

        auth_row = QHBoxLayout()
        auth_row.setSpacing(6)
        _lbl_auth = QLabel()
        i18n_bind(
            _lbl_auth, "setText", "setup.backend.cloud_auth", default="Auth:"
        )
        auth_row.addWidget(_lbl_auth)
        self._cloud_auth_key = QRadioButton()
        i18n_bind(
            self._cloud_auth_key,
            "setText",
            "setup.backend.cloud_auth_key",
            default="SSH Key",
        )
        self._cloud_auth_key.setChecked(True)
        self._cloud_auth_pwd = QRadioButton()
        i18n_bind(
            self._cloud_auth_pwd,
            "setText",
            "setup.backend.cloud_auth_pwd",
            default="Password",
        )
        auth_row.addWidget(self._cloud_auth_key)
        auth_row.addWidget(self._cloud_auth_pwd)
        auth_row.addStretch()
        cloud_layout.addLayout(auth_row)

        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        _lbl_key = QLabel()
        i18n_bind(
            _lbl_key,
            "setText",
            "setup.backend.cloud_key_file",
            default="Key file:",
        )
        key_row.addWidget(_lbl_key)
        self._cloud_key_edit = QLineEdit()
        self._cloud_key_edit.setPlaceholderText("~/.ssh/id_rsa")
        self._cloud_key_edit.setText("~/.ssh/id_rsa")
        key_row.addWidget(self._cloud_key_edit, 1)
        self._btn_browse_key = QPushButton()
        self._btn_browse_key.setFixedWidth(90)
        self._btn_browse_key.clicked.connect(self._browse_ssh_key)
        i18n_bind(
            self._btn_browse_key,
            "setText",
            "setup.data_dir.btn_browse",
            default="Browse ...",
        )
        key_row.addWidget(self._btn_browse_key)
        cloud_layout.addLayout(key_row)

        self._cloud_pwd_row = QWidget()
        pwd_row_layout = QHBoxLayout(self._cloud_pwd_row)
        pwd_row_layout.setContentsMargins(0, 0, 0, 0)
        pwd_row_layout.setSpacing(6)
        _lbl_pwd = QLabel()
        i18n_bind(
            _lbl_pwd,
            "setText",
            "setup.backend.cloud_password",
            default="Password:",
        )
        pwd_row_layout.addWidget(_lbl_pwd)
        self._cloud_pwd_edit = QLineEdit()
        self._cloud_pwd_edit.setEchoMode(QLineEdit.EchoMode.Password)
        pwd_row_layout.addWidget(self._cloud_pwd_edit, 1)
        self._cloud_pwd_row.hide()
        cloud_layout.addWidget(self._cloud_pwd_row)

        self._cloud_auth_key.toggled.connect(self._on_cloud_auth_toggled)

        remote_row = QHBoxLayout()
        remote_row.setSpacing(6)
        _lbl_remote = QLabel()
        i18n_bind(
            _lbl_remote,
            "setText",
            "setup.backend.cloud_remote_path",
            default="Remote path:",
        )
        remote_row.addWidget(_lbl_remote)
        self._cloud_remote_dir_edit = QLineEdit()
        # Remote SSH target path (the user's *remote* server's filesystem,
        # not this machine). Suggest a neutral install root that doesn't
        # imply UnitPort owns the remote user's home.
        self._cloud_remote_dir_edit.setText("~/isaac_lab")
        self._cloud_remote_dir_edit.setPlaceholderText("~/isaac_lab")
        remote_row.addWidget(self._cloud_remote_dir_edit, 1)
        cloud_layout.addLayout(remote_row)

        cloud_note = QLabel()
        cloud_note.setObjectName("pageHint")
        cloud_note.setStyleSheet("color: #888; font-size: 11px;")
        cloud_note.setWordWrap(True)
        i18n_bind(
            cloud_note,
            "setText",
            "setup.backend.cloud_note",
            default=(
                "Isaac Sim 5.1.0 | IsaacLab (pinned) | PyTorch 2.7+cu128\n"
                "Requires: Ubuntu 22.04+, NVIDIA GPU, CUDA 12.x, ~30 GB disk"
            ),
        )
        cloud_layout.addWidget(cloud_note)
        self._cloud_panel.hide()
        opts_layout.addWidget(self._cloud_panel)

        self._isaac_options.hide()
        isaac_layout.addWidget(self._isaac_options)

        # Visually slot Isaac Lab right after MuJoCo so the "Enable Isaac Lab"
        # toggle isn't buried at the bottom of the page and skipped over.
        content_layout.insertWidget(1, isaac_group)
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

    def _on_isaac_install_radio_toggled(self, checked: bool) -> None:
        """Gate the Isaac Lab fresh-install radio behind the EULA modal.

        Fires on both check + uncheck via ``QRadioButton.toggled``; we
        only act on the rising edge. The flow:

        1. If the user just *unchecked* (e.g. switched to Locate) → clear
           the in-session accepted-ids cache (different mode, no consent
           needed) and return.
        2. If the user just *checked* AND the persisted acceptance store
           already covers every required EULA at the pinned version →
           record the ids in this session's cache and return (no modal).
        3. Otherwise → open ``EulaDialog.exec()``. Acceptance: cache ids
           and let the radio stay checked. Rejection: programmatically
           uncheck the install radio (under a re-entrancy guard) and
           clear the install path so the user has a clean slate to
           pick another option.
        """
        if self._suppress_install_radio_handler:
            return

        if not checked:
            self._isaac_eula_accepted_ids = []
            return

        # Already covered by <USER_CONFIG_DIR>/eula_acceptance.json? skip modal.
        if _eula.required_eula_ids_satisfied(require_specific_versions=True):
            self._isaac_eula_accepted_ids = [
                spec.eula_id for spec in _eula.list_required_eulas()
            ]
            log_info(
                "[wizard] Isaac install: EULA acceptance already on file; "
                "skipping modal"
            )
            return

        # Lazy import: avoids dialogs/__init__ touching the heavy
        # dialog graph at wizard-construction time (PEP 562 path).
        from application.ui.dialogs.eula_dialog import EulaDialog

        log_info("[wizard] Isaac install: opening EULA dialog")
        dlg = EulaDialog(parent=self)
        accepted: List[str] = []

        def _capture_ids(ids: List[str]) -> None:
            nonlocal accepted
            accepted = list(ids or [])

        dlg.accepted_ids.connect(_capture_ids)
        result = dlg.exec()

        if result == QDialog.DialogCode.Accepted and accepted:
            self._isaac_eula_accepted_ids = accepted
            log_info(
                f"[wizard] Isaac install: EULA accepted "
                f"({len(accepted)} licence(s))"
            )
            return

        # Rejected — bounce the radio back and tidy state. Guard against
        # the toggled() that the programmatic uncheck will fire.
        log_info("[wizard] Isaac install: EULA rejected; reverting radio")
        self._isaac_eula_accepted_ids = []
        self._suppress_install_radio_handler = True
        try:
            self._rb_isaac_install.setChecked(False)
        finally:
            self._suppress_install_radio_handler = False
        # _on_isaac_mode_toggled hides _install_path_panel on its own
        # via the unchecked toggled. We do NOT clear _install_dir_edit —
        # the user may simply re-tick the radio later and we should not
        # discard their typed path on a transient cancel.

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
            self._isaac_path_status.setText(
                tr(
                    "setup.backend.isaac_valid",
                    "Valid Isaac Lab installation detected",
                )
            )
            self._isaac_path_status.setStyleSheet("color: #00D26A; font-size: 12px;")
            self._cb_isaaclab.setChecked(True)
            self._rb_isaac_locate.setChecked(True)
        else:
            self._isaac_path_status.setText(
                tr(
                    "setup.backend.isaac_invalid",
                    "Could not verify Isaac Lab at this path -- "
                    "expected isaaclab.sh/bat or source/ directory",
                )
            )
            self._isaac_path_status.setStyleSheet("color: #FFC107; font-size: 12px;")

    def get_config(self) -> Dict[str, Any]:
        is_install = self._rb_isaac_install.isChecked()
        is_locate = self._rb_isaac_locate.isChecked()
        is_cloud = self._rb_isaac_cloud.isChecked()

        custom_mods_selected: List[str] = [
            key for key, cb in self._custom_mod_checkboxes.items()
            if cb.isEnabled() and cb.isChecked()
        ]

        cfg: Dict[str, Any] = {
            "mujoco_pip": self._cb_mujoco.isChecked(),
            "loco_mujoco": self._cb_loco_mujoco.isChecked(),
            "custom_mods": custom_mods_selected,
            "ros2_enabled": self._cb_ros2.isChecked(),
            "isaaclab_enabled": self._cb_isaaclab.isChecked(),
            "isaaclab_install": is_install,
            "isaaclab_locate": is_locate,
            "isaaclab_cloud_deploy": is_cloud,
            "isaaclab_path": (
                self._install_dir_edit.text().strip() if is_install
                else self._isaac_path_edit.text().strip()
            ),
            # Only emit eula_accepted_ids when the user is actually going
            # to install — the locate and cloud_deploy paths skip the
            # in-app installer entirely and have no EULA gate of their
            # own (locate trusts the user's pre-existing Isaac Lab licence
            # acceptance; cloud_deploy is stub this release).
            "eula_accepted_ids": (
                list(self._isaac_eula_accepted_ids) if is_install else []
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
                    or "~/isaac_lab"
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
        i18n_bind(
            self,
            "setWindowTitle",
            "setup.wizard.window_title",
            default="UnitPort — Custom Setup",
        )
        self.setMinimumSize(1320, 720)
        self.resize(1340, 760)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )

        self._result_data: Dict[str, Any] = {}
        self._init_ui()
        # Initial nav label ("Next" on page 0, switches to "Finish" via
        # _update_nav once the user reaches the last page).
        self._update_nav()
        self.setStyleSheet(_wizard_stylesheet())
        # Re-apply the QSS-driven page-title / button styles when the user
        # switches language while the wizard is open — the underlying
        # widget text is refreshed by i18n_bind, but the dot indicators
        # and primary-button styling depend on objectName state.
        I18n.instance().language_changed.connect(
            lambda _lang: (self.setStyleSheet(_wizard_stylesheet()), self._update_nav())
        )

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(0)

        # ---- Header --------------------------------------------------
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel()
        title.setObjectName("wizardTitle")
        i18n_bind(
            title, "setText", "setup.wizard.header_title", default="UnitPort Setup"
        )
        header.addWidget(title)
        subtitle = QLabel()
        subtitle.setObjectName("wizardSubtitle")
        subtitle.setWordWrap(True)
        i18n_bind(
            subtitle,
            "setText",
            "setup.wizard.header_subtitle",
            default=(
                "Choose which components to install. You can change these later in Settings."
            ),
        )
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

        self._btn_skip = QPushButton()
        self._btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_skip.clicked.connect(self._on_skip)
        i18n_bind(
            self._btn_skip, "setText", "setup.wizard.btn_skip", default="Skip Setup"
        )
        footer.addWidget(self._btn_skip)

        self._btn_back = QPushButton()
        self._btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_back.clicked.connect(self._go_back)
        self._btn_back.setEnabled(False)
        i18n_bind(
            self._btn_back, "setText", "setup.wizard.btn_back", default="Back"
        )
        footer.addWidget(self._btn_back)

        # btn_next's label is "Next" on intermediate pages and "Finish" on
        # the last page; ``_update_nav`` re-applies the correct tr() each
        # time the page changes (and is also re-fired on language_changed
        # by the connect block in __init__).
        self._btn_next = QPushButton()
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
        self._btn_next.setText(
            tr("setup.wizard.btn_finish", "Finish")
            if is_last
            else tr("setup.wizard.btn_next", "Next")
        )

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
        if idx == 0:
            # Leaving the DataDirectoryPage — persist the workspace root
            # BEFORE any later page can run code that depends on
            # USER_CONFIG_DIR being set. There is no fallback, so refuse
            # to advance if the user hasn't picked a path.
            if not self._apply_data_dir_choice():
                return
        if idx < total - 1:
            self._stack.setCurrentIndex(idx + 1)
            self._update_nav()
        else:
            self._finish()

    def _apply_data_dir_choice(self) -> bool:
        """Persist the DataDirectoryPage selection BEFORE any USER_CONFIG
        write happens.

        Writes the chosen absolute path into ``system.ini[Workspace].root``
        and derives ``[Resources].user_config_dir`` from it (via
        :func:`set_workspace_root`, which also calls ``reload_paths``).
        After this returns successfully, ``Paths.USER_CONFIG_DIR`` is set
        and the next write (``save_setup_state``) lands at the right place.

        Returns ``False`` if the user has not picked a valid path; the
        wizard must NOT advance in that case — there is no fallback.
        """
        chosen = self._data_dir_page.get_data_dir()
        if chosen is None:
            log_warning(
                "[wizard] data directory not chosen — wizard cannot "
                "advance. USER_CONFIG_DIR has no built-in default; the "
                "user must enter an absolute path."
            )
            return False
        try:
            _set_workspace_root(chosen)
        except OSError as exc:
            log_warning(
                f"[wizard] could not initialise workspace at {chosen}: {exc}"
            )
            return False
        log_info(f"[wizard] workspace root set to {chosen}")
        # set_workspace_root already calls reload_paths internally, but
        # call it again defensively in case the implementation changes.
        _reload_paths()
        # The pre-wizard language picker called Config.set_value AND
        # write_machine_locale BEFORE the workspace existed; both wrote
        # only to in-memory state (disk persistence deferred to avoid
        # materialising the sentinel path). Flush them now.
        try:
            Config.flush_pending_overlay_writes()
        except Exception as exc:                                      # noqa: BLE001
            log_warning(f"[wizard] could not flush user.ini overlay: {exc}")
        try:
            _apply_machine_locale_preference()
        except Exception as exc:                                      # noqa: BLE001
            log_warning(f"[wizard] could not back-fill machine locale: {exc}")
        return True

    def _on_skip(self) -> None:
        """User skipped the wizard.

        We still require a WORKSPACE root — skipping the menagerie / SDK
        / backend pages is fine, but without a workspace root the SDK has
        nowhere to write ``setup_state.json`` and refusing to materialise
        a default is a hard rule. If the user clicks Skip without picking
        a path, the dialog stays open.
        """
        if not self._apply_data_dir_choice():
            return
        self._result_data = {"skipped": True}
        save_setup_state({"completed": True, "skipped": True})
        self.completed.emit(self._result_data)
        self.accept()

    def _finish(self) -> None:
        """Collect selections and close."""
        if not self._apply_data_dir_choice():
            # The DataDirectoryPage status label already explains what's
            # missing. Keep the dialog open so the user can fix it.
            return
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
