"""MainWindow -- pyramid main console (single shared window).

LoadingScreen and the main UI live in the **same** top-level window:
the central widget is a ``QStackedWidget`` whose page 0 is the
``LoadingScreen`` (log wallpaper + dim + pulsing logo) and page 1 is the
main content (Sidebar | main_row + work_zone). When startup tasks are
done, ``finish_loading()`` fades the loading page out and swaps to the
main page in place -- no separate window, no re-show.

Layout of the main page::

    Sidebar | ( main_row                                    )
            | ( canvas_panel (main_panel) | cmd_column      )
            |   <----------- work_zone splitter ----------> |

* ``Sidebar`` is a thin left rail; clicks dispatch to feature windows
  (stub today; real windows arrive in later stages).
* ``main_row`` is a fixed-height empty placeholder (future: breadcrumb
  / project header / global search).
* ``work_zone`` hosts a horizontal ``QSplitter``:
    - ``canvas_panel`` (left, flex) wraps an empty ``main_panel``
      placeholder -- real canvas widgets land here in later stages.
    - ``cmd_column`` (right) wraps the ``CmdLogWidget`` console; min
      width 100px, edge-draggable splitter handle (DEMO-style).

Lifecycle (5-step contract):
    init        construction-time properties + state holders. All
                global data is loaded by ``UnitPortMain`` while the
                loading page is showing, so this method does **not**
                load data.
    load_data   project switch / open: re-load project-scoped data via
                DataManager. Not called at startup.
    build_ui    construct the central stack and both pages.
    refresh     reflect current state (window title, status bar).
    apply_theme apply Config colors / fonts; called on init and on
                theme switch.

Public methods invoked by ``UnitPortMain``:
    open_project(path)       load a project + refresh
    show_project_picker()    stub (later: open a feature window)
    finish_loading()         fade the loading page out, swap to main
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    Qt,
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QGuiApplication, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    CmdLogWidget,
    Config,
    DataManager,
    I18n,
    LaviProgressBar,
    Paths,
    get_data_value,
    get_task_signal,
    get_tasks_manager,
    i18n_bind,
    log_debug,
    log_error,
    log_info,
    log_warning,
    read_data,
    save_data,
    tr,
)

from application.service.projects import (
    ProjectInfo,
    current_project_info,
    get_project_store,
    list_canvas_groups,
    resolve_file,
    set_current_project_info,
)
from application.service.training_assets import get_training_assets
from registers import backends as backends_registry

from .canvas import CanvasPage
from .loading_screen import LoadingScreen
from .sidebar import Sidebar

# MissionControlPanel / SysMonitorWidget are deliberately NOT imported at
# module load. Their transitive import chain (mission_control_panel ->
# real_robot_connection_card -> adapters -> ros2 native bridge -> cyclonedds)
# is heavy and must not block Stage 1 (LoadingScreen-first paint, see
# main.py docstring). They are imported locally inside the methods that
# instantiate them (page 1 construction in ``build_main_page_now``-reachable
# code paths). ``from __future__ import annotations`` (line 44) makes their
# names usable in type annotations without the symbols at module scope.


class _MainPanel(QWidget):
    """Host widget with two **mutually exclusive** top-level modes::

        mode = "canvas"   → CanvasPage (bottom) + MissionControlPanel (overlay).
                            Picker is hidden.
        mode = "picker"   → HomepagePage fills the whole panel.
                            Canvas + MC are hidden so they cannot bleed
                            through underneath.

    The two views are **siblings** in this widget — one is always
    completely erased while the other is shown, so there is no chance of
    Mission Control's overlay or the canvas leaking visually behind the
    picker (or vice versa).

    The MC overlay (in canvas mode) still sizes to either the full panel
    rect (Mission Control mode — chart over canvas) or just the top-row
    strip (Training Canva mode — canvas fully exposed below), driven by
    :attr:`MissionControlPanel.overlay_compact_changed`.
    """

    MODE_CANVAS = "canvas"
    MODE_PICKER = "picker"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("mainPanel")
        self._canvas: Optional[CanvasPage] = None
        self._mc: Optional[MissionControlPanel] = None
        self._picker: Optional[QWidget] = None
        self._mode: str = self.MODE_PICKER

    def set_children(
        self,
        canvas: CanvasPage,
        mc: MissionControlPanel,
        picker: QWidget,
    ) -> None:
        self._canvas = canvas
        self._mc = mc
        self._picker = picker
        mc.overlay_compact_changed.connect(lambda _b: self._relayout())
        # Initial mode applied + relayout so the right children are visible.
        self.set_mode(self._mode)

    def current_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch between picker and canvas modes.

        Both children are explicitly shown/hidden — never just stacked
        — so there is no z-order leak. Idempotent.
        """
        if mode not in (self.MODE_CANVAS, self.MODE_PICKER):
            raise ValueError(f"_MainPanel.set_mode: unknown mode {mode!r}")
        self._mode = mode
        is_picker = (mode == self.MODE_PICKER)
        if self._canvas is not None:
            self._canvas.setVisible(not is_picker)
        if self._mc is not None:
            self._mc.setVisible(not is_picker)
        if self._picker is not None:
            self._picker.setVisible(is_picker)
        self._relayout()

    def resizeEvent(self, ev) -> None:  # noqa: D401
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self) -> None:
        w = max(0, self.width())
        h = max(0, self.height())
        if self._mode == self.MODE_PICKER:
            if self._picker is not None:
                self._picker.setGeometry(0, 0, w, h)
                self._picker.raise_()
            return
        # canvas mode
        if self._canvas is None or self._mc is None:
            return
        # Canvas always fills the host.
        self._canvas.setGeometry(0, 0, w, h)
        # MC overlay: full when not compact, top_row strip when compact.
        if self._mc.is_overlay_compact():
            mc_h = self._mc.top_row_height()
        else:
            mc_h = h
        self._mc.setGeometry(0, 0, w, mc_h)
        self._mc.raise_()


# Trainer tasks emit progress text like "step 100/6000  reward=..." (SB3) or
# "iter 100/6000  reward=..." (Isaac Lab). We pull (current, total) out of
# that text so the inline LaviProgressBar can show real iter counts instead
# of the constructor-default 0/100. Match the *first* N/M pair to avoid
# accidentally grabbing reward values that happen to contain a slash.
_TRAINING_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


class MainWindow(QMainWindow):
    """UnitPort top-level window -- pyramid main console + loading page."""

    loading_finished = pyqtSignal()

    # Progress-row dropdown signals — UI-only stubs; backend wiring lands in
    # Stage C/D when projects/training services are populated.
    # ``history_requested`` was retired when History became a canvas-snapshot
    # archive (per-row delete + Clear All) — clicks now feed
    # ``_on_history_run_clicked`` directly via the HistoryMenu widget.
    template_requested = pyqtSignal(str)        # template path/id
    policy_requested = pyqtSignal(str, str)     # (backend_id, policy_id)

    # Inline training controls (formerly on the floating ControlBar).
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    backend_changed = pyqtSignal(str)           # "local" | "cloud"

    _DEFAULT_W = 1920
    _DEFAULT_H = 1080
    _MAIN_ROW_H = 40
    _FADE_MS = 250

    # work_zone splitter: canvas (flex) | cmd (resizable).
    _CMD_MIN_W = 300
    _CMD_DEFAULT_W = 340

    def __init__(self) -> None:
        super().__init__()
        self.init()
        self.build_ui()
        self.apply_theme()
        self.refresh()
        # Window title + status bar carry composite strings ("Studio - <project>")
        # that can't be expressed as a single i18n_bind. refresh() recomputes
        # them; piggyback on language_changed to keep them in sync. The slot
        # also fans out to sidebar.refresh() (panels that read tr() at build
        # time and don't bind individually).
        I18n.instance().language_changed.connect(self.refresh)
        log_debug("[ui] MainWindow constructed")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self) -> None:
        self.setWindowTitle("UnitPort Studio")
        icon_path = Assets.find_icon("icon_app")
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(self._DEFAULT_W, self._DEFAULT_H)
        self._center_on_primary_screen()

        # Widget slots (instantiated in build_ui).
        self._central_stack: Optional[QStackedWidget] = None
        self._loading_page: Optional[LoadingScreen] = None
        self._main_page: Optional[QWidget] = None
        self._sidebar: Optional[Sidebar] = None
        self._main_row: Optional[QFrame] = None
        self._sys_monitor: Optional[SysMonitorWidget] = None
        self._progress_bar: Optional[LaviProgressBar] = None
        # main_row label rendered before [Templates ▾]: shows
        # "[<engine_tag>] <canvas_name>" (engine tag tinted with the engine's
        # theme slot, name in `highlight`) when a canvas is selected/loaded,
        # or "-- --" in `main_c1` when idle. Driven by _update_canvas_label.
        self._canvas_label: Optional[QLabel] = None
        self._btn_new: Optional[QToolButton] = None
        self._btn_templates: Optional[QToolButton] = None
        self._btn_history: Optional[QToolButton] = None
        # HistoryMenu widget hosted by _btn_history — built in _setup_main_row.
        # Lazy-typed via Any to avoid a top-level import of the widget (kept
        # local to the build_ui callsite for the same reason MissionControlPanel
        # imports are deferred — see file docstring).
        self._history_menu: Optional[Any] = None
        self._btn_policies: Optional[QToolButton] = None
        self._start_btn: Optional[QPushButton] = None
        self._stop_btn: Optional[QPushButton] = None
        self._target_combo: Optional[QComboBox] = None
        self._work_zone: Optional[QWidget] = None
        self._work_splitter: Optional[QSplitter] = None
        self._canvas_panel: Optional[QWidget] = None
        self._main_panel: Optional[_MainPanel] = None
        # main_panel hosts the canvas at the bottom and MissionControlPanel
        # as an overlay raised above it. The overlay's geometry flips
        # between full-cover (Mission Control mode — chart over canvas)
        # and top-row strip (Training Canva mode — canvas exposed below).
        self._canvas_page: Optional[CanvasPage] = None
        self._mission_control_panel: Optional[MissionControlPanel] = None
        # Empty-state picker; sibling of canvas+mission_control under
        # _main_panel. Mutually exclusive with them — see _MainPanel.
        self._picker_panel: Optional[QWidget] = None
        self._cmd_column: Optional[QWidget] = None
        self._cmd_log: Optional[CmdLogWidget] = None

        # Fade animation state (owned by finish_loading).
        self._fade_effect: Optional[QGraphicsOpacityEffect] = None
        self._fade_anim: Optional[QPropertyAnimation] = None

        # Project state.
        self._project_path: str = ""
        # Resolved ProjectInfo for the currently loaded project; populated
        # by _bind_project from open_project.
        self._current_project: Optional[ProjectInfo] = None

        # Currently-active training task id (empty = idle). Set by
        # _on_start_training, cleared by _on_training_finished. Drives the
        # ▶ / ■ button enabled-state via set_training_running.
        self._active_task_id: str = ""
        # Slot index of the active training task (-1 = idle). Captured
        # synchronously from TasksManager.get_slot_status() right after
        # submit, because TaskSignal.status_changed("running") is emitted
        # *inside* submit (DirectConnection) — too early for our handler
        # to see _active_task_id. Used by _on_training_progress_updated
        # to filter the global progress_updated signal.
        self._active_slot_idx: int = -1

    def load_data(self, project_path: str) -> None:
        """Reload project-scoped data on project open / switch.

        Startup-phase data is owned by ``UnitPortMain``; this method is
        only invoked from ``open_project`` (and future project-switch
        flows), never from ``__init__``.
        """
        log_debug(f"[ui] load_data: {project_path}")
        self._project_path = project_path
        # Stage C/D: hydrate project metadata, node graph, settings via
        # DataManager once the project format lands.

    def build_ui(self) -> None:
        # Only the loading page is built up-front. The main page (Sidebar +
        # work_zone + canvas) is constructed in :meth:`build_main_page_now`,
        # invoked from ``UnitPortMain._finalize`` after every startup Task
        # has finished. This keeps MainWindow.__init__ free of heavy import
        # chains (auth → httpx, training → torch, etc.) so the LoadingScreen
        # can paint within ~100 ms of show().
        self._central_stack = QStackedWidget(self)
        self._central_stack.addWidget(self._build_loading_page())
        self._central_stack.setCurrentWidget(self._loading_page)
        self.setCentralWidget(self._central_stack)

        self.setStatusBar(QStatusBar(self))

        # F11 toggles full-screen (paired with the Sidebar bottom button).
        QShortcut(
            QKeySequence(Qt.Key.Key_F11),
            self,
            activated=self.toggle_fullscreen,
        )

        # Ctrl+S dispatches based on the MissionControlPanel mode:
        # Scripts mode with a script loaded → save the script via the
        # editor's resolver path; otherwise → save the active canvas
        # via page.save_to_project (project-internal path policy).
        QShortcut(
            QKeySequence.StandardKey.Save,
            self,
            activated=self._handle_save_shortcut,
        )

        # PageUp / PageDown cycle the MissionControlPanel SliderSwitch
        # (Mission Control | Training Canva | Scripts). No-op when the
        # switch is hidden (no canvas loaded).
        QShortcut(
            QKeySequence(Qt.Key.Key_PageUp),
            self,
            activated=lambda: self._cycle_mc_mode(-1),
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_PageDown),
            self,
            activated=lambda: self._cycle_mc_mode(+1),
        )

    def refresh(self) -> None:
        if self._project_path:
            name = Path(self._project_path).name
            self.setWindowTitle(f"UnitPort Studio - {name}")
            self.statusBar().showMessage(f"Project: {name}")
        else:
            self.setWindowTitle("UnitPort Studio")
            self.statusBar().showMessage(tr("status.ready", "UnitPort ready"))
        # Hot-reload sidebar panels (Project Files re-scans projects/).
        if self._sidebar is not None:
            self._sidebar.refresh()

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_1", "#1E1E1E")
        border = Config.get_color("border_1", "#2C2C2C")
        btn_text = Config.get_color("main_t1", "#D6D3C7")
        btn_border = Config.get_color("border_2", "#3d3d3d")
        btn_hover = Config.get_color("hover_1", "#525252")
        btn_pressed = Config.get_color("hover_2", "#212121")
        sep_color = Config.get_color("sub_t2", "#777777")
        start_color = Config.get_color("safe_zone", "#36E38E")
        stop_color = Config.get_color("danger_zone", "#FF6B6B")
        muted = Config.get_color("sub_t2", "#777777")
        font_small = Config.get_font_size("size_small", 12)
        font_normal = Config.get_font_size("size_normal", 14)
        self.setStyleSheet(
            f"QMainWindow {{ background-color: {bg}; }}"
            f"QFrame#mainRow {{ "
            f"background-color: {bg}; "
            f"border-bottom: 1px solid {border}; "
            f"}}"
            f"QLabel#mainRowCanvasLabel {{ "
            f"background: transparent; font-size: {font_normal}px; }}"
            f"QWidget#workZone {{ background-color: {bg}; }}"
            f"QWidget#canvasPanel {{ background-color: {bg}; }}"
            f"QWidget#mainPanel {{ background-color: {bg}; }}"
            f"QWidget#cmdColumn {{ "
            f"background-color: {bg}; "
            f"border-left: 1px solid {border}; "
            f"}}"
            f"QToolButton#progressRowDropdown {{ "
            f"background: transparent; color: {btn_text}; "
            f"border: 1px solid {btn_border}; font-size: {font_small}px; "
            f"padding: 1px 8px; border-radius: 4px; }}"
            f"QToolButton#progressRowDropdown:hover {{ background: {btn_hover}; }}"
            f"QToolButton#progressRowDropdown:pressed {{ background: {btn_pressed}; }}"
            f"QToolButton#progressRowDropdown::menu-indicator {{ image: none; }}"
            f"QLabel#progressRowSep {{ color: {sep_color}; "
            f"background: transparent; padding: 0 2px; "
            f"font-size: {font_small}px; }}"
            f"QPushButton#progressRowStart, QPushButton#progressRowStop {{ "
            f"background: transparent; border: none; border-radius: 4px; "
            f"color: {btn_text}; font-size: {font_small}px; }}"
            f"QPushButton#progressRowStart:hover, QPushButton#progressRowStop:hover {{ "
            f"background: {btn_hover}; }}"
            f"QPushButton#progressRowStart:enabled {{ color: {start_color}; }}"
            f"QPushButton#progressRowStop:enabled {{ color: {stop_color}; }}"
            f"QPushButton#progressRowStart:disabled, "
            f"QPushButton#progressRowStop:disabled {{ color: {muted}; }}"
            f"QComboBox#progressRowTarget {{ "
            f"padding: 1px 16px 1px 8px; background: transparent; "
            f"color: {btn_text}; border: 1px solid {start_color}; "
            f"border-radius: 4px; font-size: {font_small}px; }}"
            f"QComboBox#progressRowTarget:hover {{ background: {btn_hover}; }}"
            f"QComboBox#progressRowTarget::drop-down {{ border: none; width: 14px; }}"
            f"QComboBox#progressRowTarget QAbstractItemView {{ "
            f"background: {bg}; color: {btn_text}; "
            f"border: 1px solid {start_color}; "
            f"selection-background-color: {btn_hover}; }}"
        )
        if self._progress_bar is not None:
            self._progress_bar.refresh_style()
        # Re-stamp the inline RichText colors on theme change.
        self._update_canvas_label()
        if self._loading_page is not None:
            self._loading_page.apply_theme()
        if self._sidebar is not None:
            self._sidebar.apply_theme()
        if self._canvas_page is not None:
            self._canvas_page.apply_theme()
        if self._mission_control_panel is not None:
            self._mission_control_panel.apply_theme()
        if self._sys_monitor is not None:
            self._sys_monitor.apply_theme()

    def closeEvent(self, event) -> None:  # noqa: D401
        if self._sys_monitor is not None:
            self._sys_monitor.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Project entry points (called by UnitPortMain._finalize)
    # ------------------------------------------------------------------
    def open_project(self, path: str) -> None:
        self.load_data(path)
        self._bind_project(path)
        # ``_bind_project`` may set ``_current_project = None`` when the
        # supplied path isn't a registered project (stale ``last_path`` in
        # user.ini, deep-link to a moved project, sidebar click on a
        # deleted entry, …). Pivot to the picker instead of leaving the
        # window in a half-bound state — downstream calls like
        # ``open_canvas`` would otherwise flip into canvas mode with no
        # ProjectInfo and render the "dead screen".
        if self._current_project is None:
            log_warning(
                f"[ui] open_project: {path!r} is not a registered project; "
                "returning to picker"
            )
            self._project_path = ""
            self.refresh()
            self.show_project_picker()
            return
        if self._mission_control_panel is not None:
            self._mission_control_panel.notify_project_loaded()
        # Default to picker mode after a project switch — the new project
        # has no canvas loaded yet. open_canvas (called from _finalize on
        # startup, or by the user via the picker / sidebar) will flip
        # back to canvas mode through ``_on_canvas_loaded``.
        self._show_picker_view()
        self.refresh()

    def open_project_in_place(self, path: str) -> bool:
        """Bind a project living under another user's workspace.

        Used by the cross-user Open-in-place flow: B picks A's canvas →
        chooses "Open in place" → we need to bind A's project even though
        A's project tree is intentionally excluded from B's ProjectStore
        snapshot (it scans only ``Paths.PROJECTS_DIR``, i.e. the active
        user's projects). We read ``project.yaml`` ourselves, mint an
        ad-hoc :class:`ProjectInfo`, and bind through the snapshot-free
        ``_bind_project_info`` path. Returns True on success, False if
        the manifest is missing / malformed (caller falls back to picker).

        Audit lifecycle stays intact: ``_current_project.path`` points
        at A's directory, so ``predicted_target`` in
        ``_save_current_canvas`` resolves to A's file, ``classify_target``
        recognises it as cross-user, and ``record_overwrite`` fires.
        """
        proj_path = Path(path)
        manifest_path = proj_path / "project.yaml"
        if not manifest_path.exists():
            log_warning(
                f"[ui] open_project_in_place: missing manifest at {manifest_path}"
            )
            return False
        try:
            manifest = DataManager.load(manifest_path, force_reload=True)
        except Exception as exc:                              # noqa: BLE001
            log_warning(
                f"[ui] open_project_in_place: manifest parse failed: {exc!r}"
            )
            return False
        if not isinstance(manifest, dict):
            log_warning(
                f"[ui] open_project_in_place: manifest not a mapping at {manifest_path}"
            )
            return False
        name = str(manifest.get("name") or proj_path.name)
        project_id = str(manifest.get("project_id") or proj_path.name)
        updated_at = manifest.get("updated_at")
        try:
            updated_ts = float(updated_at) if updated_at is not None else 0.0
        except (TypeError, ValueError):
            updated_ts = 0.0
        info = ProjectInfo(
            path=proj_path,
            name=name,
            project_id=project_id,
            updated_at=updated_ts,
            manifest=dict(manifest),
        )
        # Mirror open_project's tail — load_data warm + bind + UI refresh.
        try:
            self.load_data(path)
        except Exception:                                     # noqa: BLE001
            pass
        self._bind_project_info(info)
        if self._mission_control_panel is not None:
            self._mission_control_panel.notify_project_loaded()
        self._show_picker_view()
        self.refresh()
        log_info(
            f"[ui] open_project_in_place: bound cross-user project "
            f"{info.name!r} at {info.path}"
        )
        return True

    def _bind_project(self, path: Optional[str]) -> None:
        """Rebind MissionControlPanel to the project at ``path`` (or clear).

        Snapshot-bound binding: ``path`` must resolve via the active
        user's ProjectStore snapshot. For cross-user open-in-place
        (where the target lives under another user's workspace and is
        therefore *intentionally* absent from the active snapshot), use
        :meth:`_bind_cross_user_project` instead — it constructs an
        ad-hoc ProjectInfo from the source manifest and binds directly.
        """
        info: Optional[ProjectInfo] = None
        if path:
            info = get_project_store().find_by_path(Path(path))
            if info is None:
                log_warning(f"[ui] open_project: no ProjectInfo for {path!r}")
        self._bind_project_info(info)

    def _bind_project_info(self, info: Optional[ProjectInfo]) -> None:
        """Bind ``info`` (or unbind when ``None``) without snapshot lookup.

        Common tail of :meth:`_bind_project` and the cross-user open-in-
        place path. Mutates ``_current_project``, broadcasts via
        ``set_current_project_info`` / ``AppSignals.project_changed``,
        resets the canvas-backend global, and rebuilds the
        MissionControlPanel groups + sidebar dropdown selection.
        """
        self._current_project = info
        # Broadcast the change. ``set_current_project_info`` is the canonical
        # write side: it updates the module-level state every consumer reads
        # via ``current_project_info()`` (training tasks, bundle exporter)
        # and emits ``AppSignals.project_changed`` for live-refresh
        # subscribers (top-row History/Policies dropdowns).
        set_current_project_info(info)
        # Reset the canvas-backend global on every project rebind. If the
        # newly-bound project auto-opens a canvas, ``_on_canvas_loaded``
        # will overwrite this with the real backend id immediately.
        try:
            from application.service.signals import set_current_backend
            set_current_backend("")
        except Exception:                                  # pragma: no cover
            pass
        if self._mission_control_panel is None:
            return  # main page not built yet; defensive guard
        if info is None:
            self._mission_control_panel.set_project(None)
            self._push_canvas_groups_to_projects_panel([])
            self._push_project_to_scripts_training_panel(None)
            return
        canvas_groups = list_canvas_groups(info)
        # Bind the project context first so any selection signal that
        # fires during rebuild has an anchor for resolve_file.
        self._mission_control_panel.set_project(info)
        self._push_canvas_groups_to_projects_panel(canvas_groups)
        self._push_project_to_scripts_training_panel(info)
        n_canvas = sum(
            len(sg.get("items") or []) for tg in canvas_groups for sg in (tg.get("groups") or [])
        )
        log_info(
            f"[ui] bound MissionControlPanel -> {info.name} ({n_canvas} canvas)"
        )
        # Sync the sidebar's project dropdown so the active project stays
        # visible after open_project flows that didn't originate there
        # (startup last_path, ProjectPicker, deep-link, ...).
        self._sync_projects_dropdown(info.path)

    def _sync_projects_dropdown(self, path) -> None:
        if self._sidebar is None:
            return
        projects_panel = self._sidebar.panel_widget("projects")
        if projects_panel is None:
            return
        setter = getattr(projects_panel, "set_current_project", None)
        if callable(setter):
            setter(path)

    def _on_workspace_changed(self, _old_dir: str, _new_dir: str) -> None:
        """USER_CONFIG_DIR was relocated (manual move) OR hot-swapped by sign-in.

        The project store now scans the new ``Paths.PROJECTS_DIR``; the
        currently bound project (if any) sits at a new absolute path. We:

        1. Snapshot the **whole UI state** that depends on disk paths
           BEFORE refreshing — currently-bound project id, currently-open
           canvas file_id, and whether we were in canvas mode. This is
           what makes a guest → user sign-in invisible to the workflow:
           the user was looking at canvas A in ``_guest``, and after the
           hot-switch we re-bind the same project + re-open the same
           canvas from its new physical path.
        2. Refresh ProjectStore snapshot.
        3. Re-bind the project by ``project_id`` (content-stable across
           directory renames).
        4. If we were in canvas mode and the canvas survived, re-open it
           so the view doesn't flip to the picker / new-canvas screen.
        5. Fallback to the picker only when the project genuinely didn't
           survive the relocate.
        """
        prior = self._current_project
        prior_id = prior.project_id if prior is not None else ""

        # Snapshot canvas state before any refresh.
        prior_canvas_id = ""
        was_in_canvas_mode = False
        if self._main_panel is not None:
            was_in_canvas_mode = (
                getattr(self._main_panel, "_mode", "") == _MainPanel.MODE_CANVAS
            )
        if self._mission_control_panel is not None:
            prior_canvas_id = (
                self._mission_control_panel.current_canvas_file_id or ""
            ).strip()

        from application.service.projects import get_project_store
        store = get_project_store()
        snapshot = store.refresh_snapshot()

        # Resolution order for "which project to land on after the switch":
        #   1. The project the user was working on right before the switch,
        #      if its project_id is in the new snapshot (handles dir-rename
        #      under the same workspace).
        #   2. The new workspace's user.ini ``[Project] last_path`` (with
        #      ``last_canvas``). This is what makes login restore the user's
        #      prior session — mirrors main.py ``_project_load_body``.
        #   3. First registered project in the snapshot (newest by manifest
        #      ``updated_at``) — so a fresh-login workspace with projects
        #      lands on *something* instead of a blank picker.
        target_path = ""
        target_canvas = ""
        restore_canvas = False

        if prior_id:
            for info in snapshot:
                if info.project_id == prior_id:
                    target_path = str(info.path)
                    if was_in_canvas_mode and prior_canvas_id:
                        target_canvas = prior_canvas_id
                        restore_canvas = True
                    log_info(
                        f"[ui] workspace relocate: re-binding {info.name} at "
                        f"new path {info.path}"
                    )
                    break

        if not target_path and Paths.USER_INI.exists():
            # Config.reload() has already been called inside reload_paths()
            # → get_data_value reflects the NEW workspace's user.ini.
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
            if last:
                registered = store.find_by_path(Path(last))
                if registered is not None:
                    target_path = str(registered.path)
                    if last_canvas:
                        abs_canvas = Path(target_path) / last_canvas
                        if abs_canvas.exists():
                            target_canvas = last_canvas
                            restore_canvas = True
                    log_info(
                        f"[ui] workspace switch: restoring last project "
                        f"{registered.name} ({registered.path})"
                    )

        if not target_path:
            first = store.first_project()
            if first is not None:
                target_path = str(first.path)
                log_info(
                    f"[ui] workspace switch: no last_path; auto-selecting "
                    f"first registered project {first.name}"
                )

        if target_path:
            self.open_project(target_path)
            if restore_canvas and target_canvas:
                # open_canvas validates file_id resolves under the bound
                # project; gracefully falls back to picker if not.
                self.open_canvas(target_canvas)
            # Belt-and-braces: re-stamp the User rail icon from the live
            # (post-switch) auth state. Signal-based paths sometimes miss
            # this — e.g. account-switch where cached_user is None for a
            # tick while session.json is mid-merge.
            if self._sidebar is not None:
                self._sidebar.refresh_user_icon()
            # Defer audit review one event-loop tick so the project re-bind
            # finishes painting before the modal lands on top.
            QTimer.singleShot(0, self._maybe_show_audit_review)
            return

        # New workspace has zero projects — clean picker state.
        self._project_path = ""
        self._current_project = None
        try:
            from application.service.projects import set_current_project_info
            set_current_project_info(None)
        except Exception:                                          # pragma: no cover
            pass
        if self._mission_control_panel is not None:
            self._mission_control_panel.set_project(None)
            self._push_canvas_groups_to_projects_panel([])
            self._push_project_to_scripts_training_panel(None)
        self.refresh()
        self.show_project_picker()
        if self._sidebar is not None:
            self._sidebar.refresh_user_icon()
        # Even a zero-project new workspace may have inherited audit
        # entries from a prior session on this machine — fire the review
        # trigger before returning.
        QTimer.singleShot(0, self._maybe_show_audit_review)

    def _maybe_show_audit_review(self) -> None:
        """Surface any pending cross-user audit entries for the active user.

        Called from ``_on_workspace_changed`` via ``QTimer.singleShot(0,
        ...)`` so the project re-bind finishes painting before the modal
        appears on top. Guards: skip on guest / signed-out workspaces;
        skip when the pending queue is empty.
        """
        try:
            from application.service.user_workspace import read_active_user
            from application.ui.dialogs.cross_user_audit_review_dialog import (
                show_audit_review_if_pending,
            )
        except Exception as exc:                              # noqa: BLE001
            log_warning(f"[ui] audit-review: import failed: {exc!r}")
            return
        uid = (read_active_user() or "").strip()
        if not uid or uid == "_guest":
            return
        try:
            show_audit_review_if_pending(uid, parent=self)
        except Exception as exc:                              # noqa: BLE001
            log_warning(f"[ui] audit-review: show failed: {exc!r}")

    def show_project_picker(self) -> None:
        """Switch the host panel to the picker (HomepagePage).

        Erases canvas+MissionControlPanel from the view; the picker
        becomes the sole top-level child of ``_main_panel``. Used at
        startup when no project / no canvas resolves, and from the
        main_row "New" button.
        """
        log_debug("[ui] show_project_picker -> picker mode")
        self.statusBar().showMessage(
            tr("status.no_project", "No project loaded")
        )
        self._show_picker_view()

    # ------------------------------------------------------------------
    # Picker / Canvas transition primitives
    # ------------------------------------------------------------------
    # MainRow 左上的 canvas label、_main_panel 当前页、Sidebar 导航按钮显隐
    # 必须保持「绝对统一」。任何把 _main_panel 翻到 PICKER/CANVAS 的入口都
    # 必须走下面这两个原语，避免出现「label 显示 canvas 已加载 / 实际页面是
    # picker / Sidebar 卡在 TC 样式」这类三态漂移（登出→重登场景的根因）。
    def _show_picker_view(self) -> None:
        if self._main_panel is not None:
            self._main_panel.set_mode(_MainPanel.MODE_PICKER)
        self._update_canvas_label()
        self._update_start_btn_enabled()
        self._apply_main_row_canvas_mode(False)
        if self._sidebar is not None and self._mission_control_panel is not None:
            self._sidebar.apply_view_mode(
                self._mission_control_panel._effective_mode()
            )

    def _show_canvas_view(self) -> None:
        if self._main_panel is not None:
            self._main_panel.set_mode(_MainPanel.MODE_CANVAS)
        self._update_canvas_label()
        self._update_start_btn_enabled()
        self._apply_main_row_canvas_mode(True)
        if self._sidebar is not None and self._mission_control_panel is not None:
            self._sidebar.apply_view_mode(
                self._mission_control_panel._effective_mode()
            )

    def _apply_main_row_canvas_mode(self, in_canvas: bool) -> None:
        """Toggle the canvas-scoped main_row dropdowns.

        The three dropdowns (Templates / History / Policies) only operate
        on a loaded canvas — Templates filters by the canvas's backend,
        History/Policies query the project's training assets cache. In
        picker mode there is no canvas, so hide them rather than show
        useless placeholders. The [New] button + inline training controls
        stay visible across modes.
        """
        for btn in (
            self._btn_templates,
            self._btn_history,
            self._btn_policies,
        ):
            if btn is not None:
                btn.setVisible(in_canvas)

    def open_canvas(self, file_id: str) -> None:
        """Programmatically open ``file_id`` inside the current project.

        Called from ``UnitPortMain._finalize`` to auto-jump back into
        ``[Project] last_canvas`` on next launch, and from
        :meth:`_create_and_open_canvas` after a fresh canvas is written
        to disk. Triggers the existing MissionControlPanel auto-load
        path (and the canvas-mode flip) via ``canvas_loaded`` →
        :meth:`_on_canvas_loaded`.
        """
        if not file_id:
            return
        if self._mission_control_panel is None:
            log_warning("[ui] open_canvas: MissionControlPanel not built yet")
            return
        # Refuse to flip into canvas mode without a bound project — that's
        # the "dead screen" path (canvas frame shown but nothing loads
        # because MissionControlPanel has no ProjectInfo). Pivot back to
        # the picker instead so the user can choose / create a project.
        if self._current_project is None:
            log_warning(
                f"[ui] open_canvas: refusing {file_id!r} — no project bound; "
                "returning to picker"
            )
            self.show_project_picker()
            return
        # Make sure the canvas + MC overlay are visible before driving
        # the LaviTabTable selection — _files_table lives under
        # MissionControlPanel which is hidden in picker mode, and a
        # hidden widget cannot accept programmatic selection cleanly.
        self._show_canvas_view()
        self._mission_control_panel.open_canvas(file_id)

    def _on_canvas_loaded(self, file_id: str) -> None:
        """Successful canvas load: persist + ensure canvas mode is shown."""
        # Belt-and-braces: open_canvas already flipped to canvas mode,
        # but a stray load triggered from elsewhere (sidebar click while
        # picker is open, e.g.) must also flip.
        self._show_canvas_view()
        self._persist_last_canvas(file_id)
        # Push the canvas's bound backend into the AppSignals global so
        # the training pipeline (run dirs, exported bundles), the
        # TrainingAssetsCache (filtering), and any future deploy panel
        # can read it via ``current_backend()`` without poking CanvasPage
        # directly. Empty string = no backend bound (defensive — every
        # well-formed canvas declares a backend, but a fresh one might
        # arrive here mid-init).
        try:
            from application.service.signals import set_current_backend
            page = self._canvas_page
            backend_id = (getattr(page, "backend_id", None) if page else None) or ""
            set_current_backend(backend_id)
        except Exception as exc:                           # pragma: no cover
            log_warning(f"[ui] _on_canvas_loaded: backend broadcast failed: {exc}")
        # Sync the sidebar Project Files highlight to the just-loaded canvas
        # so the row stays selected even when the load came from the picker /
        # last-canvas auto-open (rather than a row click).
        if self._sidebar is not None:
            panel = self._sidebar.panel_widget("projects")
            setter = getattr(panel, "set_current_canvas", None) if panel else None
            if callable(setter):
                try:
                    setter(file_id)
                except Exception:
                    pass
        # Sidebar may have just been re-painted; if the backend swap changed
        # the SB3-vs-IL Observs visibility, re-run apply_view_mode.
        if self._sidebar is not None and self._mission_control_panel is not None:
            try:
                self._sidebar.apply_view_mode(
                    self._mission_control_panel._effective_mode()
                )
            except Exception:
                pass
        # Update start button + canvas label after load completes.
        self._update_start_btn_enabled()
        self._update_canvas_label()

    def _persist_last_canvas(self, file_id: str) -> None:
        """Persist the (project_path, canvas_file_id) pair to user.ini.

        Fires only on successful CanvasPage loads, so a failed auto-load
        won't blow away ``last_canvas``. ``Config.set_value`` writes to
        ``user.ini`` only and invalidates DataManager's cache for that
        path.
        """
        if not self._project_path or not file_id:
            return
        try:
            Config.set_value("Project", "last_path", self._project_path)
            Config.set_value("Project", "last_canvas", file_id)
        except Exception as exc:  # noqa: BLE001 — never fail user navigation on persist
            log_warning(f"[ui] persist last_canvas failed: {exc!r}")

    def _create_and_open_canvas(
        self,
        project_path: str,
        canvas_name: str,
        engine_id: str,
        template_path: str = "",
    ) -> None:
        """Materialize a new ``<project>/canvas/<subdir>/<name>.canvas.json``
        and auto-open it.

        Called from the HomepagePage create card. The picker has already
        validated the project name + canvas name + engine id and
        guaranteed the project directory exists (via
        ``ProjectStore.create_project``).

        ``template_path`` (optional) selects the seed source:
            * ""           → blank canvas (empty nodes/edges, default view)
            * "<abs path>" → read that ``.canvas.json`` and write its
                             contents as the seed (so the new project
                             starts with a fully-wired template). The
                             ``backend`` field of the resulting file is
                             defensively forced to ``engine_id`` so a
                             template whose backend was hand-edited
                             can't poison the new canvas.
        """
        from application.service.projects import get_project_store
        from registers import backends as backends_registry

        info = get_project_store().find_by_path(Path(project_path))
        if info is None:
            log_error(f"[ui] new canvas: project not found at {project_path!r}")
            return

        # Open the project (no-op if it's already the active project).
        if self._project_path != str(info.path):
            self.open_project(str(info.path))

        # Compose the on-disk path. canvas_subdir maps engine_id → folder
        # name (e.g. sb3_mujoco → "SB3"); the file_id used by the sidebar
        # files list is "canvas/<subdir>/<name>.canvas.json".
        subdir = backends_registry.canvas_subdir(engine_id)
        target = Path(info.path) / "canvas" / subdir / f"{canvas_name}.canvas.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        # Decide the seed payload. Template path wins when readable;
        # otherwise fall back to the empty seed (schema mirrors
        # CanvasPage.to_workflow_dict so load_from_file accepts it
        # without a migration path).
        seed = None
        if template_path:
            try:
                loaded = read_data(Path(template_path))
            except Exception as exc:                          # noqa: BLE001
                log_warning(
                    f"[ui] new canvas: template read failed {template_path}: {exc!r}"
                )
                loaded = None
            if isinstance(loaded, dict):
                seed = dict(loaded)
                # Defensive override: trust the engine_id picked in the
                # UI over whatever the template file claims.
                seed["backend"] = engine_id
            else:
                log_warning(
                    f"[ui] new canvas: template not a dict, falling back to "
                    f"empty seed: {template_path!r}"
                )
        if seed is None:
            seed = {
                "version": "1.0.0",
                "backend": engine_id,
                "metadata": {
                    "view": {"zoom": 1.0, "center_x": 0.0, "center_y": 0.0}
                },
                "nodes": [],
                "edges": [],
            }
        try:
            save_data(target, seed)
        except Exception as exc:  # noqa: BLE001
            log_error(f"[ui] new canvas: write failed at {target}: {exc!r}")
            return
        if template_path:
            log_info(
                f"[ui] new canvas created from template {Path(template_path).name}: {target}"
            )
        else:
            log_info(f"[ui] new canvas created: {target}")

        # Refresh the canvas-tab groups so the freshly-written file shows
        # up in the sidebar list.
        self._bind_project(str(info.path))

        # Auto-select / load it. Drives canvas_loaded -> _persist_last_canvas.
        file_id = f"canvas/{subdir}/{canvas_name}.canvas.json"
        self.open_canvas(file_id)

    def _on_homepage_canvas_open(
        self, project_path: str, canvas_file_id: str
    ) -> None:
        """Workspace-row Load → bind project then auto-open the canvas.

        Called from ``HomepagePage.canvas_open_requested`` when the user
        clicks "Load" (or double-clicks) on a row in the local workspaces
        table. When the picked canvas lives under another user's
        workspace, we route through the cross-user choice dialog first:

        * **Copy** — duplicate the canvas (and its project shell) into
          the active user's workspace and open the copy. Pure within-
          workspace work, no audit trail.
        * **Open in place** — open the original. Any later save / delete
          flows through the audit hook in ``_save_with_audit`` and the
          delete handler in projects_card, both of which classify_target
          again at action time.
        * **Cancel** — no-op.

        Non-cross-user opens take the straight path identical to the
        previous behaviour.
        """
        if not project_path or not canvas_file_id:
            return

        # Resolve the absolute canvas path so cross_user_audit can match
        # it against the workspace root.
        canvas_abs = Path(project_path) / canvas_file_id
        try:
            from application.service import cross_user_audit
            target_uid, _rel = cross_user_audit.classify_target(canvas_abs)
        except Exception as exc:                              # noqa: BLE001
            log_warning(f"[ui] classify_target failed: {exc!r}")
            target_uid = None

        if target_uid:
            from application.ui.dialogs.cross_user_open_choice_dialog import (
                CrossUserOpenChoiceDialog,
                OpenChoice,
            )
            owner_label = self._cross_user_owner_label(target_uid)
            choice = CrossUserOpenChoiceDialog.pick(
                owner_label=owner_label,
                canvas_name=canvas_abs.name,
                parent=self,
            )
            if choice == OpenChoice.CANCEL:
                return
            if choice == OpenChoice.COPY:
                copied = self._copy_cross_user_canvas_to_active(
                    src_project_path=Path(project_path),
                    canvas_file_id=canvas_file_id,
                    owner_label=owner_label,
                )
                if copied is None:
                    return
                new_project_path, new_canvas_file_id = copied
                self.open_project(str(new_project_path))
                self.open_canvas(new_canvas_file_id)
                return
            # OpenChoice.IN_PLACE: the cross-user project is intentionally
            # NOT in the active ProjectStore snapshot, so open_project
            # would reject the path. Bind ad-hoc via open_project_in_place
            # so _current_project.path points at the source — the audit
            # hook in _save_current_canvas will then classify saves as
            # cross-user and record them.
            if not self.open_project_in_place(project_path):
                self.show_project_picker()
                return
            self.open_canvas(canvas_file_id)
            return

        # open_project pivots back to the picker on failure (stale path,
        # non-active workspace, etc.) — we still call open_canvas
        # unconditionally because it guards on _current_project itself.
        self.open_project(project_path)
        self.open_canvas(canvas_file_id)

    def _cross_user_owner_label(self, target_uid: str) -> str:
        """Return a display label for the user owning ``target_uid``."""
        try:
            from application.service.user_workspace import read_workspace_root
            from application.ui.widgets.homepage.projects_card import (
                _user_display_label,
            )
            root = read_workspace_root()
            return _user_display_label(root / target_uid) or target_uid
        except Exception:                                     # noqa: BLE001
            return target_uid

    def _copy_cross_user_canvas_to_active(
        self,
        *,
        src_project_path: Path,
        canvas_file_id: str,
        owner_label: str,                                     # noqa: ARG002
    ) -> Optional[Tuple[Path, str]]:
        """Materialise the picked canvas into the active user's workspace.

        Strategy (the "preserve the outer project name" approach the
        user explicitly asked for): drop the canvas at
        ``<active_uid>/projects/<src_project_id>/<canvas_file_id>``,
        re-using the source's directory name verbatim. Two sub-cases:

        * **Destination project doesn't exist** — create it by copying
          the source's ``project.yaml`` so the new project appears in
          the active user's ProjectStore snapshot under the same name.
        * **Destination project already exists** (the active user
          happens to have a same-named project) — drop the canvas into
          the existing project. If the canvas filename clashes inside
          it, suffix the stem with ``_copy`` / ``_copy_N`` until free.

        Either way the result is a regular intra-workspace project that
        ``open_project`` can bind via the normal snapshot lookup — no
        ad-hoc binding required. Returns ``(dst_project_path,
        dst_canvas_file_id)`` on success, ``None`` on failure (user is
        informed via a QMessageBox).
        """
        try:
            from application.service.projects import get_project_store
        except Exception as exc:                              # noqa: BLE001
            log_error(f"[ui] copy: import failed: {exc!r}")
            return None

        store = get_project_store()
        src_canvas = src_project_path / canvas_file_id
        if not src_canvas.exists():
            self._copy_failed_dialog(
                canvas_file_id,
                f"source canvas missing on disk: {src_canvas}",
            )
            return None

        src_project_id = src_project_path.name
        dst_project_path = Paths.PROJECTS_DIR / src_project_id

        # ---- ensure the destination project exists ---------------------
        # We DO NOT use ``format="bytes"`` for these copies: that would
        # cache raw ``bytes`` under the destination's .yaml / .canvas.json
        # path key in DataManager. The canvas page then loads the dst path
        # without a format hint, hits the cache, gets bytes instead of a
        # dict, and crashes with "content not a dict" + renders an empty
        # canvas (only a restart clears the cache). Round-tripping
        # through each extension's native handler caches the typed object
        # the consumer expects.
        if not dst_project_path.exists():
            try:
                dst_project_path.mkdir(parents=True, exist_ok=True)
                src_manifest = src_project_path / "project.yaml"
                if src_manifest.exists():
                    manifest = DataManager.load(src_manifest, force_reload=True)
                    if isinstance(manifest, dict):
                        DataManager.write(
                            dst_project_path / "project.yaml", manifest,
                        )
                    else:
                        log_warning(
                            f"[ui] copy: source manifest read returned "
                            f"{type(manifest).__name__}, skipping manifest copy"
                        )
            except Exception as exc:                          # noqa: BLE001
                log_error(f"[ui] copy: dst project init failed: {exc!r}")
                self._copy_failed_dialog(canvas_file_id, str(exc))
                return None

        # ---- pick a non-clashing destination canvas path ---------------
        dst_canvas_rel = canvas_file_id
        dst_canvas = dst_project_path / dst_canvas_rel
        if dst_canvas.exists():
            # Split the file_id into "<parent>/" + "<stem>.canvas.json"
            # so we can suffix only the stem.
            if "/" in canvas_file_id:
                parent_rel, leaf = canvas_file_id.rsplit("/", 1)
                parent_rel += "/"
            else:
                parent_rel, leaf = "", canvas_file_id
            stem = leaf
            if stem.lower().endswith(".canvas.json"):
                stem = stem[: -len(".canvas.json")]
            placed = False
            for i in range(1, 100):
                cand_leaf = (
                    f"{stem}_copy.canvas.json"
                    if i == 1 else f"{stem}_copy_{i}.canvas.json"
                )
                cand_rel = f"{parent_rel}{cand_leaf}"
                cand_abs = dst_project_path / cand_rel
                if not cand_abs.exists():
                    dst_canvas_rel = cand_rel
                    dst_canvas = cand_abs
                    placed = True
                    break
            if not placed:
                self._copy_failed_dialog(
                    canvas_file_id,
                    "could not find a free canvas filename (99 attempts)",
                )
                return None

        # ---- copy the canvas via the JSON handler ----------------------
        # See the note above ``ensure the destination project exists``:
        # the bytes-format detour breaks the canvas loader by caching
        # bytes under the dst .canvas.json key. Read + write as a dict
        # so the cache stores what every consumer expects.
        try:
            dst_canvas.parent.mkdir(parents=True, exist_ok=True)
            canvas_doc = DataManager.load(src_canvas, force_reload=True)
            if not isinstance(canvas_doc, dict):
                raise TypeError(
                    f"source canvas is not a JSON object "
                    f"(got {type(canvas_doc).__name__})"
                )
            DataManager.write(dst_canvas, canvas_doc)
        except Exception as exc:                              # noqa: BLE001
            log_error(f"[ui] copy: canvas write failed: {exc!r}")
            self._copy_failed_dialog(canvas_file_id, str(exc))
            return None

        log_info(
            f"[ui] copied cross-user canvas {src_canvas} → {dst_canvas}"
        )
        try:
            store.refresh_snapshot()
        except Exception:                                     # noqa: BLE001
            pass
        return (dst_project_path, dst_canvas_rel)

    def _copy_failed_dialog(self, canvas_file_id: str, err_text: str) -> None:
        """Surface a Copy failure to the user with the underlying reason."""
        QMessageBox.warning(
            self,
            tr(
                "homepage.projects.cross_user_copy_failed_title",
                "Copy failed",
            ),
            tr(
                "homepage.projects.cross_user_copy_failed_body",
                "Could not copy '{name}' into your workspace:\n{err}",
            ).format(name=canvas_file_id, err=err_text),
        )

    # ------------------------------------------------------------------
    # Sidebar Project Files signal handling
    # ------------------------------------------------------------------
    def _on_mc_mode_changed(self, mode: str) -> None:
        """MissionControl mode 切换 → Sidebar nav 按钮显隐重排."""
        if self._sidebar is None:
            return
        self._sidebar.apply_view_mode(mode)

    def _on_mc_canvas_loaded_resync(self, _file_id: str) -> None:
        """Canvas 加载完成 → 用 effective mode 再同步一次 Sidebar."""
        if self._sidebar is None or self._mission_control_panel is None:
            return
        self._sidebar.apply_view_mode(
            self._mission_control_panel._effective_mode()
        )

    def _on_sidebar_node_requested(self, node_id: str) -> None:
        """Sidebar 弹出面板里双击节点 → 当前 canvas 视口中心 spawn.

        Gates:
            - canvas page 必须存在
            - canvas 当前必须 interactive(MC 模式 / 无 canvas 加载时静默拒绝)
        """
        page = self._canvas_page
        if page is None:
            return
        if not getattr(page, "interactive", True):
            log_debug(
                f"[main_window] sidebar node_requested '{node_id}' suppressed "
                f"(canvas not interactive)"
            )
            return
        try:
            view = getattr(page, "view", None)
            if view is None:
                return
            vp_center = view.viewport().rect().center()
            scene_pos = view.mapToScene(vp_center)
            page.spawn_node(node_id, x=scene_pos.x(), y=scene_pos.y())
        except Exception as exc:
            log_warning(
                f"[main_window] sidebar spawn '{node_id}' failed: {exc!r}"
            )

    def _on_projects_canvas_selected(self, _file_id: str) -> None:
        """ProjectsPanel selection → load canvas + re-evaluate ▶ / label.

        Forwards to ``open_canvas`` (which drives MC's auto-load path) and
        also re-evaluates the start button + canvas label so the UI tracks
        even when the user clicks a row but the load is still in flight.
        """
        if _file_id:
            self.open_canvas(_file_id)
        self._update_start_btn_enabled()
        self._update_canvas_label()

    def _cycle_mc_mode(self, direction: int) -> None:
        """PageUp/PageDown handler — advance MissionControlPanel mode tab.

        Direction is -1 (PageUp / previous) or +1 (PageDown / next), with
        wrap-around. Silently no-ops when the panel is absent or the
        SliderSwitch is hidden (canvas not yet loaded).
        """
        mc = self._mission_control_panel
        if mc is None:
            return
        mc.cycle_mode(int(direction))

    def _handle_save_shortcut(self) -> None:
        """Ctrl+S dispatcher: route to script save or canvas save.

        When the MissionControlPanel is in Scripts mode with a script
        loaded in the compiler editor, the keypress targets the script
        (saved via the resolver / file path established when the script
        was loaded). Otherwise it falls through to the canvas save.

        The unsaved-changes prompt still calls ``_save_current_canvas``
        directly — it only ever needs the canvas-side save.
        """
        mc = self._mission_control_panel
        if mc is not None and mc.try_save_active_script():
            return
        self._save_current_canvas()

    def _save_current_canvas(self) -> bool:
        """Save the active canvas via page.save_to_project.

        Reached via :meth:`_handle_save_shortcut` (when MC is not in
        Scripts mode) and directly from the unsaved-changes prompt in
        :meth:`_confirm_discard_or_save_current_canvas`.

        Returns True on a successful save, False on any soft failure (no
        project / no canvas / malformed file_id / save_to_project rejected).
        The Ctrl+S shortcut path ignores the return value; the
        unsaved-changes prompt consumes it to keep the user on the
        current canvas when a save fails.

        Cross-user audit: when the target path lives under another user's
        workspace (caller chose "Open in place" on the cross-user prompt
        earlier), we ``capture_pre_state`` before the write and
        ``record_overwrite`` after — so the owner sees the change on their
        next sign-in via :meth:`_maybe_show_audit_review`.
        """
        if self._current_project is None:
            log_warning("[ui] save: no project loaded; ignoring Ctrl+S")
            return False
        mc = self._mission_control_panel
        page = self._canvas_page
        if mc is None or page is None:
            return False
        file_id = (mc.current_canvas_file_id or "").strip()
        if not file_id:
            log_warning("[ui] save: no canvas file loaded; ignoring Ctrl+S")
            return False
        stem = file_id
        if stem.lower().startswith("canvas/"):
            stem = stem[len("canvas/"):]
        if stem.lower().endswith(".canvas.json"):
            stem = stem[: -len(".canvas.json")]
        if not stem:
            log_warning(f"[ui] save: malformed file_id {file_id!r}; ignoring Ctrl+S")
            return False

        # Predict the on-disk target so we can classify + capture BEFORE
        # save_to_project actually writes. The path is deterministic from
        # (project.path, file_id), which is exactly how page.save_to_project
        # builds it internally.
        predicted_target = Path(self._current_project.path) / file_id
        from application.service import cross_user_audit
        try:
            target_uid, _rel = cross_user_audit.classify_target(predicted_target)
        except Exception as exc:                              # noqa: BLE001
            log_warning(f"[ui] save: classify_target failed: {exc!r}")
            target_uid = None
        pre_state = (
            cross_user_audit.capture_pre_state(predicted_target)
            if target_uid else None
        )

        target = page.save_to_project(self._current_project, stem)
        if target is None:
            log_error(f"[ui] save: page.save_to_project returned None for {stem!r}")
            return False

        if target_uid:
            # Re-read post bytes for the sha — cheap on canvas files (KBs).
            post_bytes: Optional[bytes] = None
            try:
                raw = DataManager.load(
                    Path(target), force_reload=True, format="bytes",
                )
                if isinstance(raw, (bytes, bytearray)):
                    post_bytes = bytes(raw)
            except Exception as exc:                          # noqa: BLE001
                log_warning(f"[ui] save: post-read failed: {exc!r}")
            cross_user_audit.record_overwrite(
                Path(target), pre_state or {}, post_bytes, note="canvas_save",
            )
            owner_label = self._cross_user_owner_label(target_uid)
            self.statusBar().showMessage(
                tr(
                    "homepage.projects.cross_user_audit_recorded",
                    "Logged for {owner}'s review",
                ).format(owner=owner_label),
                3000,
            )
        else:
            self.statusBar().showMessage(
                tr(
                    "status.canvas_saved", "Canvas saved: {path}",
                ).format(path=target),
                3000,
            )
        return True

    # ------------------------------------------------------------------
    # Export node §3 — SafetyReview gate + Review subprocess dispatch
    # ------------------------------------------------------------------
    def _run_safety_review(self, *, blocking: bool) -> bool:
        """Run TrainingContext.safety_review against the current canvas.

        DEMO 对应：``TrainingWorkspaceWindow._run_safety_review``
        (training_workspace_window.py:4977-5013).

        Called by ``ReviewLaunchButtonRow._handle_click`` before emitting
        ``scene.review_launch_requested``. ``blocking=True`` means error-
        severity issues abort the launch (returns False); ``blocking=False``
        runs in advisory mode (returns True regardless of severity).
        """
        page = self._canvas_page
        if page is None:
            log_warning("[ui] safety_review: no canvas page mounted")
            return False
        try:
            graph = page.to_workflow_dict()
        except Exception as exc:
            log_error(f"[ui] safety_review: serialize failed: {exc}")
            return False

        try:
            from application.training.training_context import (
                get_training_context,
            )
        except Exception as exc:
            log_warning(f"[ui] safety_review: TrainingContext import failed: {exc}")
            return not blocking

        pid = ""
        try:
            pid = str(self._current_project.id) if self._current_project else ""
        except Exception:
            pid = ""
        ctx = get_training_context(pid)
        result = ctx.safety_review(graph)

        # Surface issues to the cmd log. Future: paint red borders on
        # offending nodes via ``_apply_safety_result`` (DEMO line 4985).
        for issue in result.issues:
            line = (
                f"[safety_review/{issue.section}/{issue.code}] {issue.message}"
            )
            if issue.fix_hint:
                line += f"  ({issue.fix_hint})"
            if issue.severity == "error":
                log_error(line)
            else:
                log_warning(line)

        if blocking and not result.ok:
            log_error(
                f"[ui] safety_review: BLOCKED by "
                f"{len(result.errors)} error(s) — Launch Review aborted."
            )
            return False
        return True

    def _on_review_launch_requested(self, payload: dict) -> None:
        """Handle ``scene.review_launch_requested`` from Export node Launch button.

        DEMO 对应：``TrainingWorkspaceWindow._on_review_launch_requested``
        (training_workspace_window.py:4709+).

        Stage D scope: log the payload + dispatch by backend. Stage E+ wires
        the actual MuJoCo / Isaac Sim subprocess spawning — for now the
        mujoco branch logs a "would launch" notice so end-to-end smoke
        testing can verify the wire-up without needing a viewer.
        """
        if not isinstance(payload, dict):
            log_warning(f"[ui] review_launch: bad payload: {payload!r}")
            return
        backend = str(payload.get("backend", "") or "").strip()
        scene_id = str(payload.get("scene_id", "") or "").strip()
        bundle = str(payload.get("bundle_name", "") or "").strip()
        version = str(payload.get("version", "v1") or "v1")
        overwrite = bool(payload.get("overwrite", True))
        log_info(
            f"[ui] review_launch: backend={backend} scene={scene_id} "
            f"bundle={bundle} version={version} overwrite={overwrite}"
        )
        if backend == "mujoco":
            # Stage F-1: in-process MuJoCo passive viewer + PolicyRunner via
            # MujocoReviewTask submitted to TasksManager.
            sku = self._resolve_canvas_robot_sku()
            if not sku:
                log_error(
                    "[ui] review_launch/mujoco: no Robot node / asset_id on "
                    "canvas — drag a Robot node and pick an asset before review."
                )
                return
            bundle_dir = self._resolve_bundle_dir(
                bundle, version, overwrite, review_backend=backend,
            )
            if bundle_dir is None:
                return
            if not bundle_dir.exists():
                return
            try:
                from application.service.runtime.simulation.mujoco.review_session import (
                    MujocoReviewTask,
                )
                from unitport_sdk import get_tasks_manager
            except Exception as exc:
                log_error(f"[ui] review_launch/mujoco: import failed: {exc}")
                return
            task = MujocoReviewTask(
                bundle_path=bundle_dir,
                sku=sku,
                scene_id=scene_id,
            )
            try:
                tid = get_tasks_manager().submit(task)
            except Exception as exc:
                log_error(f"[ui] review_launch/mujoco: task submit failed: {exc}")
                return
            log_info(
                f"[ui] review_launch/mujoco: submitted task {tid} "
                f"(bundle={bundle_dir.name}, sku={sku}, scene={scene_id})"
            )
        elif backend == "isaac_sim":
            # Bundle-only IsaacSim review (CLAUDE.md §1.9). The subprocess
            # loads policy + deploy_contract from the bundle directory and
            # builds a minimal Isaac Lab scene from the SKU registry — no
            # reach-back into run_dir or canvas state. Same shape as the
            # mujoco branch above; the only difference is the Task class.
            sku = self._resolve_canvas_robot_sku()
            if not sku:
                log_error(
                    "[ui] review_launch/isaac_sim: no Robot node / asset_id "
                    "on canvas — drag a Robot node and pick an asset "
                    "before review."
                )
                return
            bundle_dir = self._resolve_bundle_dir(
                bundle, version, overwrite, review_backend=backend,
            )
            if bundle_dir is None:
                return
            if not bundle_dir.exists():
                return
            try:
                from application.service.runtime.simulation.isaac_sim import (
                    IsaacSimReviewTask,
                )
                from unitport_sdk import get_tasks_manager
            except Exception as exc:
                log_error(
                    f"[ui] review_launch/isaac_sim: import failed: {exc}"
                )
                return
            task = IsaacSimReviewTask(
                bundle_path=bundle_dir,
                sku=sku,
                scene_id=scene_id,
            )
            try:
                tid = get_tasks_manager().submit(task)
            except Exception as exc:
                log_error(
                    f"[ui] review_launch/isaac_sim: task submit failed: {exc}"
                )
                return
            log_info(
                f"[ui] review_launch/isaac_sim: submitted task {tid} "
                f"(bundle={bundle_dir.name}, sku={sku}, scene={scene_id})"
            )
        elif backend == "newton":
            log_warning(
                "[ui] review_launch/newton: backend is a placeholder — no-op"
            )
        else:
            log_warning(f"[ui] review_launch: unknown backend {backend!r}")

    def _resolve_canvas_robot_sku(self) -> str:
        """Walk current canvas for a Robot node → return normalised SKU or ''.

        Stage F-1 helper for the mujoco review path. The Robot node carries
        ``asset_id`` which may be either a short name (``go2``) or full SKU
        (``unitree.go2``). ``registers.robots.resolve_id`` normalises both.
        Returns ``""`` when no Robot / asset_id / unresolvable SKU.
        """
        page = self._canvas_page
        if page is None:
            return ""
        try:
            graph = page.to_workflow_dict()
        except Exception:
            return ""
        for n in graph.get("nodes", []) or []:
            if str(n.get("schema_id") or "") != "robot":
                continue
            params = n.get("params") or {}
            spec = params.get("asset_id")
            raw = spec.get("value") if isinstance(spec, dict) else spec
            raw = str(raw or "").strip()
            if not raw:
                continue
            try:
                from registers import robots as _r
                resolved = _r.resolve_id(raw)
                return str(resolved or raw)
            except Exception:
                return raw
        return ""

    # Maps the review-side backend id (Export node's ``review_backend``
    # picker — mujoco / isaac_sim / newton) to the training-side backend
    # id that owns the ``<project>/training/exported/<backend_id>/`` folder
    # where bundles are written. Bundles trained under one training backend
    # can sometimes be reviewed under another (e.g. an IL-trained policy can
    # also run in MuJoCo via deploy_contract), so the mapping is one-to-many:
    # the FIRST entry is the preferred location, the rest are fallbacks
    # scanned when the preferred path is missing.
    _REVIEW_BACKEND_TO_TRAIN_BACKENDS: Dict[str, tuple] = {
        "mujoco": ("sb3_mujoco", "isaac_lab"),
        "isaac_sim": ("isaac_lab",),
        "newton": (),
    }

    def _resolve_bundle_dir(
        self,
        bundle_name: str,
        version: str,
        overwrite: bool,
        review_backend: str = "",
    ) -> Optional[Path]:
        """Compute the bundle output directory for ``(name, version, overwrite)``.

        Strict project-scoped: resolves under
        ``<project>/training/exported/<train_backend_id>/<name>/``.

        When ``review_backend`` is supplied, the train-backend layer is
        derived from :data:`_REVIEW_BACKEND_TO_TRAIN_BACKENDS` (preferred
        training backend for each review backend). If that location is
        empty but the bundle exists under another training backend, the
        cross-backend hit is returned with a WARNING — the launcher's
        own compat checks then decide whether the policy actually runs.
        When ``review_backend`` is empty (legacy callers), falls back to
        the canvas-bound backend (``current_backend()``).
        Overwrite=True → ``<name>``; Overwrite=False → ``<name>_<version>``.
        Mirrors ``application.ui.canvas.param_rows._bundle_root_for``.
        Returns ``None`` (and logs an error) when no project is open.
        """
        proj = current_project_info()
        if proj is None:
            log_error(
                "[ui] review_launch: no project is open — open a project "
                "first so the trained bundle can be located under "
                "<project>/training/exported/."
            )
            return None
        name = (bundle_name or "").strip()
        if not overwrite and version:
            name = f"{name}_{version}"
        if not name:
            log_error(
                "[ui] review_launch: empty bundle name — set a bundle "
                "name on the Export node."
            )
            return None

        exported_root = proj.path / "training" / "exported"
        review_backend = (review_backend or "").strip()

        if review_backend:
            preferred = self._REVIEW_BACKEND_TO_TRAIN_BACKENDS.get(
                review_backend, ()
            )
            if not preferred:
                log_error(
                    f"[ui] review_launch: review_backend={review_backend!r} "
                    f"has no known training-backend mapping — known: "
                    f"{sorted(self._REVIEW_BACKEND_TO_TRAIN_BACKENDS)}"
                )
                return None
            for tbid in preferred:
                cand = exported_root / tbid / name
                if cand.exists():
                    if tbid != preferred[0]:
                        log_warning(
                            f"[ui] review_launch/{review_backend}: bundle "
                            f"{name!r} not found under preferred train "
                            f"backend {preferred[0]!r}; using fallback "
                            f"{tbid!r} ({cand}). Compatibility will be "
                            f"checked by the launcher."
                        )
                    return cand
            available = self._scan_exported_backends(exported_root, name)
            if available:
                log_error(
                    f"[ui] review_launch/{review_backend}: bundle {name!r} "
                    f"was trained under {sorted(available)!r}, but "
                    f"review_backend={review_backend!r} needs a bundle "
                    f"trained under one of {list(preferred)!r}. "
                    f"Either switch the Export node's review_backend, or "
                    f"re-train on the matching canvas."
                )
            else:
                log_error(
                    f"[ui] review_launch/{review_backend}: bundle {name!r} "
                    f"not found anywhere under {exported_root}. "
                    f"Train and export first."
                )
            return exported_root / preferred[0] / name

        from application.service.signals import current_backend
        bid = (current_backend() or "").strip() or "unknown"
        return exported_root / bid / name

    @staticmethod
    def _scan_exported_backends(exported_root: Path, bundle_name: str) -> list:
        """Return the list of training-backend subdir names where
        ``<exported_root>/<subdir>/<bundle_name>/`` exists. Empty list when
        nothing matches (caller decides how to report).
        """
        hits: list = []
        if not exported_root.exists():
            return hits
        try:
            for child in exported_root.iterdir():
                if not child.is_dir():
                    continue
                if (child / bundle_name).exists():
                    hits.append(child.name)
        except OSError:
            return hits
        return hits

    # ------------------------------------------------------------------
    # Loading -> main page transition
    # ------------------------------------------------------------------
    def build_main_page_now(self) -> None:
        """Construct the main page on-demand and re-apply theme.

        Called by ``UnitPortMain._finalize`` once every background startup
        Task has completed. Idempotent: a second call is a no-op so the
        method is safe to invoke from cancellation / retry paths.

        The second :meth:`apply_theme` call is required because the first
        call (from ``__init__``) ran when every page-1 widget slot was
        ``None``; the main page's sidebar / work_zone / etc. need their
        styles applied now that they exist.
        """
        if self._main_page is not None or self._central_stack is None:
            return
        self._central_stack.addWidget(self._build_main_page())
        self.apply_theme()
        # Wire the inline training controls now that the play/stop buttons
        # exist (they are constructed inside ``_build_main_page`` →
        # ``_populate_main_row_controls``). The buttons emit start_clicked /
        # stop_clicked; we drive submit_canvas_training and TasksManager.cancel
        # here. task_finished resets the running state for any task we own.
        self.start_clicked.connect(self._on_start_training)
        self.stop_clicked.connect(self._on_stop_training)
        ts = get_task_signal()
        ts.task_finished.connect(self._on_training_finished)
        # Initial ▶ state: no canvas selected yet, no canvas mounted → disabled.
        # ProjectsPanel.canvas_selected (routed via _on_projects_canvas_selected)
        # plus _on_canvas_loaded will flip it on once a canvas is picked / loaded.
        self._update_start_btn_enabled()
        # We can't reuse LaviProgressBar.bind_slot here: it routes
        # progress_updated(slot_idx, ratio, text) into set_progress(ratio=ratio)
        # only — the count label stays at constructor default 0/100. Wire our
        # own handler that parses "N/M" out of the trainer's progress text and
        # drives set_total + set_progress(current=N).
        ts.progress_updated.connect(self._on_training_progress_updated)

        # Policies dropdowns: connect click signals to handlers.
        # The dropdowns themselves rebuild on aboutToShow (see
        # _populate_history_menu / _populate_policies_menu) so a project
        # switch is picked up the next time the user opens the menu.
        # History click/delete/clear are wired directly on the HistoryMenu
        # widget in _setup_main_row (no MainWindow-level Qt signal hop).
        self.policy_requested.connect(self._on_policy_pick)
        self.template_requested.connect(self._on_template_picked)

        # Canvas-snapshot hook: every training run gets a content-addressed
        # snapshot of the live canvas, stored under <project>/canvas/_runs_/
        # and pointed to from <run_dir>/canvas_snapshot.txt so History can
        # later restore that exact canvas state. Signal fires on the main
        # thread (queued connection from sb3_task / isaac_lab task threads).
        from application.service.signals import get_app_signals
        get_app_signals().training_run_started.connect(
            self._on_training_run_started
        )

    def finish_loading(self) -> None:
        """Stop the loading logo, fade the loading page out, then swap
        the central stack to the main page. Idempotent."""
        if (
            self._central_stack is None
            or self._loading_page is None
            or self._main_page is None
        ):
            return
        if self._central_stack.currentWidget() is self._main_page:
            return

        # Halt the pulse and detach LogoPulse's inner opacity effect so
        # the parent-level fade below does not nest opacity effects.
        self._loading_page.stop_logo()

        effect = QGraphicsOpacityEffect(self._loading_page)
        effect.setOpacity(1.0)
        self._loading_page.setGraphicsEffect(effect)
        self._fade_effect = effect

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(self._FADE_MS)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._on_loading_fade_finished)
        self._fade_anim = anim
        anim.start()

    def _on_loading_fade_finished(self) -> None:
        if self._central_stack is not None and self._main_page is not None:
            self._central_stack.setCurrentWidget(self._main_page)
        if self._loading_page is not None:
            self._loading_page.setGraphicsEffect(None)
        self._fade_effect = None
        self._fade_anim = None
        # The embedded canvas inside MissionControlPanel populates from a
        # real .canvas.json the moment the user selects a file in the
        # sidebar's Project Files list — no debug populate_demo needed.
        self.loading_finished.emit()

    # ------------------------------------------------------------------
    # Full-screen toggle (F11 + Sidebar bottom button both route here)
    # ------------------------------------------------------------------
    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ------------------------------------------------------------------
    # Sidebar dispatch (stub -- feature windows arrive in later stages)
    # ------------------------------------------------------------------
    def _open_feature_window(self, key: str) -> None:
        log_debug(f"[ui] feature_requested: {key} (window not yet implemented)")
        self.statusBar().showMessage(
            tr(
                "status.feature_pending",
                f"Feature window '{key}' not yet implemented",
            )
        )

    # ------------------------------------------------------------------
    # Sidebar Update button -> in-app updater dialogs
    # ------------------------------------------------------------------
    def _on_update_requested(self) -> None:
        """Dispatch the Update click to one of three dialogs.

        - No cached release -> tell the user they're up to date.
        - Cached release without body (throttled-replay) -> force a
          fresh check, then re-dispatch with the result.
        - Cached release with a body -> open the available-update
          dialog with release notes and the 3-button footer.
        """
        from application.service.updater import (
            ApplyUpdateTask,
            get_update_service,
        )
        from .dialogs import (
            UpdateAvailableDialog,
            UpdateLatestDialog,
            UpdateProgressDialog,
        )

        svc = get_update_service()
        release = svc.latest_release()
        current_version = svc.current_version()

        # The cached-replay path (throttled hit) returns a ReleaseInfo
        # with no body / html_url; re-fetch synchronously so the dialog
        # has something to show. The user clicked Update — they're
        # actively waiting, so a one-time blocking call is acceptable.
        if release is not None and not release.body and not release.html_url:
            log_debug("[ui:update] cached replay missing notes; forcing fresh check")
            release = svc.check(force=True)

        if release is None:
            UpdateLatestDialog(current_version, parent=self).exec()
            return

        avail = UpdateAvailableDialog(release, current_version, parent=self)
        avail.apply_requested.connect(
            lambda info: self._launch_update_apply(info)
        )
        avail.exec()

    def _launch_update_apply(self, release) -> None:
        """Open the progress modal and submit ApplyUpdateTask."""
        from application.service.updater import ApplyUpdateTask
        from .dialogs import UpdateProgressDialog

        progress = UpdateProgressDialog(release.version, parent=self)
        progress.show()
        try:
            # Submit through the task master if MainWindow owns it via
            # the running QApplication. Falls back to the SDK singleton
            # otherwise (cli scripts mocking out MainWindow).
            from unitport_sdk import get_tasks_manager
            get_tasks_manager().submit(ApplyUpdateTask(release))
        except Exception as exc:                                # noqa: BLE001
            log_warning(f"[ui:update] failed to submit ApplyUpdateTask: {exc}")

    # ------------------------------------------------------------------
    # Page constructors
    # ------------------------------------------------------------------
    def _build_loading_page(self) -> QWidget:
        self._loading_page = LoadingScreen(self._central_stack)
        return self._loading_page

    def set_install_message_visible(self, visible: bool) -> None:
        """Toggle the "Installation in progress" tag on the loading page.

        Thin passthrough used by ``UnitPortMain`` on first-time install
        (shown when the wizard opens, hidden once PostSetupTask finishes).
        Safe to call before / after ``build_main_page_now`` — the loading
        page exists for the entire boot lifecycle.
        """
        if self._loading_page is not None:
            self._loading_page.set_install_message_visible(visible)

    def _build_main_page(self) -> QWidget:
        self._main_page = QWidget(self._central_stack)
        page_layout = QHBoxLayout(self._main_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        # Left: Sidebar.
        self._sidebar = Sidebar(self._main_page)
        self._sidebar.feature_requested.connect(self._open_feature_window)
        self._sidebar.fullscreen_toggle_requested.connect(self.toggle_fullscreen)
        self._sidebar.update_requested.connect(self._on_update_requested)
        # Bridge auth signals -> sidebar rail icon refresh. Sidebar already
        # subscribes to these directly inside _wire_user_icon, but routing a
        # second copy through MainWindow makes the refresh resilient to any
        # edge case where the direct connection misses (slot timing inside
        # a modal LoginDialog flush, etc.). refresh_user_icon is idempotent.
        try:
            from application.service.auth import get_auth_manager
            _auth = get_auth_manager()
            _auth.authenticated.connect(
                lambda _u: self._sidebar.refresh_user_icon()
            )
            _auth.signed_out.connect(self._sidebar.refresh_user_icon)
            _auth.avatar_updated.connect(
                lambda _uid, _pm: self._sidebar.refresh_user_icon()
            )
        except Exception as exc:                                # noqa: BLE001
            log_warning(f"[ui] sidebar auth-bridge wiring failed: {exc!r}")
        # Bridge ProjectsPanel activation -> open_project (load_data + refresh).
        projects_panel = self._sidebar.panel_widget("projects")
        if projects_panel is not None:
            sig = getattr(projects_panel, "file_activated", None)
            if sig is not None:
                sig.connect(lambda p: self.open_project(str(p)))
        # Bridge UserPanel.workspace_changed -> re-bind project from new
        # USER_CONFIG_DIR + refresh sidebar snapshots so the relocate
        # operation is reflected without a restart.
        user_panel = self._sidebar.panel_widget("user")
        if user_panel is not None:
            sig = getattr(user_panel, "workspace_changed", None)
            if sig is not None:
                sig.connect(self._on_workspace_changed)
        page_layout.addWidget(self._sidebar)

        # Right: main_row (top, empty) + work_zone (bottom, console).
        right = QWidget(self._main_page)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._main_row = QFrame(right)
        self._main_row.setObjectName("mainRow")
        self._main_row.setFixedHeight(self._MAIN_ROW_H)
        mr_layout = QHBoxLayout(self._main_row)
        mr_layout.setContentsMargins(10, 0, 10, 0)
        mr_layout.setSpacing(8)
        # Left: migrated dropdowns + inline training controls + progress bar.
        self._populate_main_row_controls(self._main_row, mr_layout)
        # Right: live system monitor. Local import: see module-top comment
        # — defers the cyclonedds / adapters chain off the Stage 1 paint path.
        from .widgets import SysMonitorWidget
        self._sys_monitor = SysMonitorWidget(self._main_row)
        mr_layout.addWidget(
            self._sys_monitor,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        right_layout.addWidget(self._main_row)

        self._work_zone = QWidget(right)
        self._work_zone.setObjectName("workZone")
        wz_layout = QVBoxLayout(self._work_zone)
        wz_layout.setContentsMargins(0, 0, 0, 0)
        wz_layout.setSpacing(0)
        wz_layout.addWidget(self._build_work_splitter(self._work_zone), 1)
        right_layout.addWidget(self._work_zone, 1)

        page_layout.addWidget(right, 1)

        # Sidebar panels are lazy-built — subscribe to ``panel_built`` so
        # we (re)seed each one with the live project + selection state the
        # first time it gets constructed. Also wire whatever's already
        # present (ProjectsPanel is built eagerly at sidebar construction).
        if self._sidebar is not None:
            self._sidebar.panel_built.connect(self._on_sidebar_panel_built)
            for key in (
                "projects",
                "training",
                "rewards",
                "terminations",
                "observations",
            ):
                panel = self._sidebar.panel_widget(key)
                if panel is not None:
                    self._on_sidebar_panel_built(key, panel)
        # Wire the mission_panel left card to the top main_row controls so
        # the card's [开始/停止] buttons mirror state + behaviour of the
        # progress-row pair, and the card's link combo stays in sync with
        # the [Local|Cloud] selector.
        if self._mission_control_panel is not None:
            try:
                self._mission_control_panel.bind_run_buttons(
                    self._start_btn, self._stop_btn
                )
                self._mission_control_panel.bind_link_combo(self._target_combo)
            except Exception as exc:
                log_warning(f"[main_window] mission_panel bind failed: {exc!r}")
        return self._main_page

    def _on_sidebar_panel_built(self, key: str, widget: QWidget) -> None:
        """Reseed a freshly-built sidebar panel with live state.

        Sidebar panels are lazy: the first time the user opens
        ``Training``, the panel widget is constructed AFTER project bind
        already ran. This hook plugs the panel into the data-flow it
        missed (project info, canvas selection, script-load → MC).
        """
        if self._mission_control_panel is None:
            return
        if key == "projects":
            # Push current canvas groups + canvas-selected wire.
            self._push_canvas_groups_to_projects_panel(
                list_canvas_groups(self._current_project)
                if self._current_project else []
            )
            sig = getattr(widget, "canvas_selected", None)
            if sig is not None:
                try:
                    sig.connect(self._on_projects_canvas_selected)
                except Exception:
                    pass
            # Re-highlight the currently loaded canvas if any.
            cur_id = (
                self._mission_control_panel.current_canvas_file_id or ""
            ).strip()
            if cur_id and hasattr(widget, "set_current_canvas"):
                try:
                    widget.set_current_canvas(cur_id)
                except Exception:
                    pass
            return
        if key == "training":
            self._push_project_to_scripts_training_panel(self._current_project)
            sig = getattr(widget, "script_selected", None)
            if sig is not None:
                try:
                    sig.connect(self._mission_control_panel.load_script)
                except Exception:
                    pass
            return
        if key in ("rewards", "terminations", "observations"):
            sig = getattr(widget, "script_selected", None)
            if sig is not None:
                try:
                    sig.connect(self._mission_control_panel.load_script)
                except Exception:
                    pass
            return

    def _push_canvas_groups_to_projects_panel(
        self, groups: list,
    ) -> None:
        if self._sidebar is None:
            return
        panel = self._sidebar.panel_widget("projects")
        setter = getattr(panel, "set_canvas_groups", None) if panel else None
        if callable(setter):
            try:
                setter(groups)
            except Exception as exc:                          # noqa: BLE001
                log_warning(f"[ui] projects_panel.set_canvas_groups failed: {exc!r}")

    def _push_project_to_scripts_training_panel(
        self, info: Optional[ProjectInfo],
    ) -> None:
        if self._sidebar is None:
            return
        panel = self._sidebar.panel_widget("training")
        setter = getattr(panel, "set_project", None) if panel else None
        if callable(setter):
            try:
                setter(info)
            except Exception as exc:                          # noqa: BLE001
                log_warning(f"[ui] training_panel.set_project failed: {exc!r}")

    _CTRL_BTN_SIZE = 24
    _CTRL_ICON_SIZE = QSize(16, 16)

    def _populate_main_row_controls(self, parent: QWidget, lay: QHBoxLayout) -> None:
        """Populate ``main_row``'s layout with the migrated controls::

            [Templates ▾] [History ▾] [Policies ▾]  |  [▶] [■] [Local/Cloud]  <bar>

        Replaces the previously separate progress row that lived above the
        canvas; sys_monitor is appended by the caller after this populator.
        """
        # "[engine] canvas_name" / "-- --" label, rendered as RichText so the
        # two segments can carry distinct theme colors. Filled in by
        # _update_canvas_label; called once below for the initial idle state.
        self._canvas_label = QLabel(parent)
        self._canvas_label.setObjectName("mainRowCanvasLabel")
        self._canvas_label.setTextFormat(Qt.TextFormat.RichText)
        self._canvas_label.setContentsMargins(4, 0, 8, 0)
        lay.addWidget(self._canvas_label)

        # "New" button — switches the canvas-tab body to the homepage
        # (four-card landing UI: account, news, local workspaces, create
        # project + template). Reuses the progressRowDropdown QSS so it
        # sits flush next to the dropdown row visually; clicking does
        # not open a menu, just calls show_project_picker.
        self._btn_new = QToolButton(parent)
        self._btn_new.setObjectName("progressRowDropdown")
        i18n_bind(self._btn_new, "setText", "progressrow.btn.new", "New")
        self._btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._btn_new, "setToolTip",
            "progressrow.tooltip.new", "Open the new-canvas picker",
        )
        self._btn_new.clicked.connect(self._on_new_clicked)
        lay.addWidget(self._btn_new)

        # Three dropdowns (renamed: Templates / History / Policies).
        # Templates scans ``<PROJECT_ROOT>/custom_mods/canvas/*.canvas.json``
        # on every aboutToShow — no index file (per project convention:
        # dynamic loading, never persist a list of templates).
        # History/Policies populate from the current project's TrainingStore
        # on every aboutToShow — that keeps the menu items in sync with
        # runs.json / bundles.json without bookkeeping side-channels.
        self._btn_templates = self._make_dropdown_btn(
            "progressrow.btn.templates", "Templates", parent,
            self.template_requested,
            populator=self._populate_templates_menu,
        )
        self._btn_history = self._make_dropdown_btn(
            "progressrow.btn.history", "History", parent,
            None,
            populator=None,
        )
        # Swap the default QMenu for HistoryMenu — rows with delete buttons +
        # Clear All footer. aboutToShow rebuilds the rows from the
        # TrainingAssetsCache; click signals route directly to handlers.
        from application.ui.widgets.history_menu import HistoryMenu
        self._history_menu = HistoryMenu(self._btn_history)
        self._history_menu.aboutToShow.connect(self._populate_history_menu)
        self._history_menu.run_clicked.connect(self._on_history_run_clicked)
        self._history_menu.delete_clicked.connect(self._on_history_run_delete)
        self._history_menu.clear_all_clicked.connect(self._on_history_clear_all)
        self._btn_history.setMenu(self._history_menu)
        self._btn_policies = self._make_dropdown_btn(
            "progressrow.btn.policies", "Policies", parent,
            self.policy_requested,
            populator=self._populate_policies_menu,
        )
        lay.addWidget(self._btn_templates)
        lay.addWidget(self._btn_history)
        lay.addWidget(self._btn_policies)
        # Hidden until a canvas is loaded — see _apply_main_row_canvas_mode.
        # Picker is the default startup mode; _show_canvas_view will reveal
        # them on the first successful canvas load.
        self._btn_templates.setVisible(False)
        self._btn_history.setVisible(False)
        self._btn_policies.setVisible(False)
        self._update_canvas_label()

        # ' | ' text separator between dropdowns and inline training controls.
        sep = QLabel(" | ", parent)
        sep.setObjectName("progressRowSep")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sep)

        # Inline play / stop / [Local|Cloud] (formerly ControlBar contents).
        self._start_btn = QPushButton(parent)
        self._start_btn.setObjectName("progressRowStart")
        self._start_btn.setFixedSize(self._CTRL_BTN_SIZE, self._CTRL_BTN_SIZE)
        self._start_btn.setIconSize(self._CTRL_ICON_SIZE)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._start_btn, "setToolTip",
            "progressrow.tooltip.start", "Start Training",
        )
        self._start_btn.clicked.connect(self.start_clicked.emit)

        self._stop_btn = QPushButton(parent)
        self._stop_btn.setObjectName("progressRowStop")
        self._stop_btn.setFixedSize(self._CTRL_BTN_SIZE, self._CTRL_BTN_SIZE)
        self._stop_btn.setIconSize(self._CTRL_ICON_SIZE)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        i18n_bind(
            self._stop_btn, "setToolTip",
            "progressrow.tooltip.stop", "Stop Training",
        )
        self._stop_btn.clicked.connect(self.stop_clicked.emit)

        self._target_combo = QComboBox(parent)
        self._target_combo.setObjectName("progressRowTarget")
        self._target_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._target_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._target_combo.addItem(
            tr("progressrow.target.local", "Local"), "local"
        )
        self._target_combo.addItem(
            tr("progressrow.target.cloud", "Cloud"), "cloud"
        )
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)

        lay.addWidget(self._start_btn)
        lay.addWidget(self._stop_btn)
        lay.addWidget(self._target_combo)

        self._apply_ctrl_icons()

        # Empty task_key + empty default_name → no "Training" label inside the bar.
        self._progress_bar = LaviProgressBar(
            task_key="",
            default_name="",
            total=100,
            parent=parent,
        )
        lay.addWidget(self._progress_bar, 1)

    def _make_dropdown_btn(
        self,
        i18n_key: str,
        default_text: str,
        parent: QWidget,
        signal: pyqtSignal,
        *,
        populator: Optional[Any] = None,
    ) -> QToolButton:
        """Build a small chevroned dropdown button matching the DEMO toolbar.

        Without a ``populator`` the menu starts with a disabled '(none)'
        placeholder (Templates uses this — items land later). When a
        ``populator`` is supplied it is invoked on every ``aboutToShow``
        so the menu always reflects the latest project state.
        """
        btn = QToolButton(parent)
        btn.setObjectName("progressRowDropdown")
        # 自带 "  ▾" 后缀，不能直接 i18n_bind setText（会被覆写为纯翻译）。
        # 自己包一层闭包挂 language_changed，并用同 i18n_bind 一致的
        # alive-flag 防 dead-widget 触发 Qt qWarning。
        sig = I18n.instance().language_changed
        state = {"alive": True}

        def _disconnect():
            if not state["alive"]:
                return
            state["alive"] = False
            try:
                sig.disconnect(_set_dd_text)
            except (TypeError, RuntimeError):
                pass

        def _set_dd_text(*_, _btn=btn, _k=i18n_key, _d=default_text):
            if not state["alive"]:
                return
            try:
                _btn.setText(tr(_k, _d) + "  ▾")
            except RuntimeError:
                _disconnect()

        _set_dd_text()
        sig.connect(_set_dd_text)
        btn.destroyed.connect(lambda *_: _disconnect())
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        menu = QMenu(btn)
        if populator is None:
            placeholder = menu.addAction(tr("common.empty", "(none)"))
            placeholder.setEnabled(False)
        else:
            menu.aboutToShow.connect(populator)
        btn.setMenu(menu)
        # Stash the bound signal so future populators can wire actions to it.
        btn._unitport_signal = signal  # type: ignore[attr-defined]
        return btn

    # ------------------------------------------------------------------
    # History / Policies dropdown populators (driven by aboutToShow)
    # ------------------------------------------------------------------
    def _populate_history_menu(self) -> None:
        """Rebuild the History menu from the TrainingAssetsCache.

        Reads the in-memory list — no per-open disk scan. The cache itself
        refreshes on project switch (subscribed to ``project_changed``)
        and after every training run finishes. Forwarded to the
        ``HistoryMenu`` widget, which builds row widgets with per-row delete
        buttons plus a Clear All footer button.
        """
        if self._history_menu is None:
            return
        if self._current_project is None:
            self._history_menu.populate(
                [],
                placeholder_text=tr(
                    "progressrow.menu.no_project", "(no project)"
                ),
            )
            return
        runs = get_training_assets().runs()
        self._history_menu.populate(runs)

    def _populate_policies_menu(self) -> None:
        """Rebuild the Policies menu from the TrainingAssetsCache.

        Reads the in-memory list — no per-open disk scan. Manifest-derived
        ``(algo, obs/action)`` suffix comes from the cached PolicyAsset
        fields (the cache parsed manifest.yaml at scan time).
        """
        if self._btn_policies is None:
            return
        menu = self._btn_policies.menu()
        if menu is None:
            return
        menu.clear()

        if self._current_project is None:
            act = menu.addAction(tr("progressrow.menu.no_project", "(no project)"))
            act.setEnabled(False)
            return

        policies = get_training_assets().policies()
        if not policies:
            act = menu.addAction(tr("progressrow.menu.empty", "(empty)"))
            act.setEnabled(False)
            return

        for p in policies:
            suffix = (
                f"  ({p.algorithm}, {p.obs_dim}/{p.action_dim})"
                if (p.algorithm or p.obs_dim or p.action_dim) else ""
            )
            label = f"[{p.backend_id}] {p.policy_id}{suffix}"
            act = menu.addAction(label)
            act.triggered.connect(
                lambda _checked=False, bid=p.backend_id, pid=p.policy_id:
                    self.policy_requested.emit(bid, pid)
            )

    # ------------------------------------------------------------------
    # Templates dropdown — dynamic scan of custom_mods/canvas/<backend>/
    # ------------------------------------------------------------------
    @staticmethod
    def _templates_root(backend_id: str = "") -> Path:
        """Return ``custom_mods/canvas[/<backend_id>]`` (backend-scoped when given).

        Templates are partitioned per backend so the dropdown can scan only
        the entries that match the currently-open canvas without reading
        each JSON file's ``backend`` field. The caller passes an empty
        string to get the parent dir (used by housekeeping / migrations).
        """
        root = Paths.CUSTOM_MODS_DIR / "canvas"
        bid = (backend_id or "").strip()
        return root / bid if bid else root

    def _current_canvas_backend(self) -> str:
        """Resolve the backend id of the currently-loaded canvas, or ''.

        Source order:
          1. ``CanvasPage.backend_id`` — authoritative once a canvas is loaded.
          2. ``signals.current_backend()`` — module-level mirror (set by
             ``_on_canvas_loaded``); covers the edge case where the canvas
             reference isn't wired yet but the backend was broadcast.
        """
        page = self._canvas_page
        bid = (getattr(page, "backend_id", None) if page else None) or ""
        bid = str(bid).strip()
        if bid:
            return bid
        try:
            from application.service.signals import current_backend
            return (current_backend() or "").strip()
        except Exception:                                       # pragma: no cover
            return ""

    def _populate_templates_menu(self) -> None:
        """Rebuild the Templates menu from disk on every aboutToShow.

        Dynamic-loading convention: no index file. We glob the
        ``custom_mods/canvas/<backend_id>/`` directory for
        ``*.canvas.json`` (where ``backend_id`` is the currently-open
        canvas's engine) and use each file's stem as the menu label.
        Click → ``template_requested`` with the absolute path.
        """
        if self._btn_templates is None:
            return
        menu = self._btn_templates.menu()
        if menu is None:
            return
        menu.clear()

        backend_id = self._current_canvas_backend()
        if not backend_id:
            act = menu.addAction(
                tr("progressrow.menu.no_canvas", "(no canvas)")
            )
            act.setEnabled(False)
            return

        root = self._templates_root(backend_id)
        if not root.exists():
            act = menu.addAction(tr("progressrow.menu.empty", "(empty)"))
            act.setEnabled(False)
            return

        try:
            files = sorted(root.glob("*.canvas.json"))
        except OSError as exc:                                  # pragma: no cover
            log_warning(f"[ui] templates scan failed at {root}: {exc!r}")
            files = []

        if not files:
            act = menu.addAction(tr("progressrow.menu.empty", "(empty)"))
            act.setEnabled(False)
            return

        for p in files:
            stem = p.name[: -len(".canvas.json")] if p.name.endswith(
                ".canvas.json"
            ) else p.stem
            act = menu.addAction(stem)
            path_str = str(p)
            act.triggered.connect(
                lambda _checked=False, _p=path_str: self.template_requested.emit(_p)
            )

    def _on_template_picked(self, template_path: str) -> None:
        """Load a template's content into the *currently open* canvas in place.

        "Load" means replace: the active canvas's scene + backend bind are
        rebuilt from the template, while the save target stays on the
        canvas file the user already had open. The canvas is left dirty so
        Ctrl+S writes the template content to that current file — we do
        NOT materialize a new file under ``<project>/canvas/<subdir>/``.

        Gates:
          * A project must be bound (else: pivot to picker).
          * A canvas must currently be loaded (else: status hint — there
            is nothing to "load into" without an open canvas).
          * The template's ``backend`` must match the current canvas's
            backend; otherwise the file would live under one backend's
            subdir while the canvas claims another (save_to_project
            rejects this), so we reject up front with a clear message.
          * Unsaved edits on the current canvas trigger the standard
            Save / Discard / Cancel prompt before we clobber the scene.
        """
        if not template_path:
            return
        src = Path(template_path)
        if not src.exists():
            log_warning(f"[ui] template missing on disk: {template_path}")
            self.statusBar().showMessage(
                tr("status.template_missing", "Template not found on disk"),
                3000,
            )
            return

        if self._current_project is None:
            self.statusBar().showMessage(
                tr(
                    "status.template_no_project",
                    "Open a project and a canvas before loading a template",
                ),
                3000,
            )
            self.show_project_picker()
            return

        mc = self._mission_control_panel
        page = self._canvas_page
        if mc is None or page is None:
            return

        current_file_id = (mc.current_canvas_file_id or "").strip()
        if not current_file_id:
            self.statusBar().showMessage(
                tr(
                    "status.template_no_canvas",
                    "Open a canvas before loading a template",
                ),
                3000,
            )
            return

        try:
            data = read_data(src)
        except Exception as exc:                                # noqa: BLE001
            log_error(f"[ui] template read failed {src}: {exc!r}")
            self.statusBar().showMessage(
                tr("status.template_read_failed", "Failed to read template"),
                3000,
            )
            return
        if not isinstance(data, dict):
            log_error(f"[ui] template file is not a dict: {src}")
            return

        template_backend = str(data.get("backend") or "").strip()
        if not template_backend:
            log_error(f"[ui] template has no backend field: {src}")
            self.statusBar().showMessage(
                tr(
                    "status.template_no_backend",
                    "Template is missing a backend field",
                ),
                3000,
            )
            return

        current_backend = (page.backend_id or "").strip()
        if current_backend and current_backend != template_backend:
            log_error(
                f"[ui] template backend {template_backend!r} != current "
                f"canvas backend {current_backend!r}"
            )
            self.statusBar().showMessage(
                tr(
                    "status.template_backend_mismatch",
                    "Template backend '{tb}' does not match current canvas "
                    "backend '{cb}'",
                ).format(tb=template_backend, cb=current_backend),
                5000,
            )
            return

        # Two-step guard:
        #   1. Dirty edits → standard Save / Discard / Cancel prompt.
        #   2. Clean but non-empty canvas → warn that load will wipe the
        #      current scene. Skipped for an already-empty canvas (no
        #      content to lose).
        if page.is_dirty():
            if not self._confirm_discard_or_save_current_canvas():
                return
        elif page.instances:
            if not self._confirm_replace_canvas_content(src.name):
                return

        try:
            page.replace_content_from_dict(data)
        except Exception as exc:                                # noqa: BLE001
            log_error(f"[ui] template load failed: {exc!r}")
            self.statusBar().showMessage(
                tr(
                    "status.template_apply_failed",
                    "Failed to apply template content",
                ),
                3000,
            )
            return

        stem = src.name[: -len(".canvas.json")] if src.name.endswith(
            ".canvas.json"
        ) else src.stem
        log_info(
            f"[ui] template '{stem}' loaded into canvas {current_file_id} "
            f"(in-memory; Ctrl+S to persist)"
        )
        self.statusBar().showMessage(
            tr(
                "status.template_loaded",
                "Template '{name}' loaded into current canvas — Ctrl+S to save",
            ).format(name=stem),
            4000,
        )

    # ------------------------------------------------------------------
    # New-canvas / dirty-canvas helpers
    # ------------------------------------------------------------------
    def _on_new_clicked(self) -> None:
        """Main-row [New] click: guard unsaved edits + unbind current canvas.

        UX:
          * If the canvas is clean (or no canvas loaded), unbind and flip
            to the picker immediately.
          * If the canvas has unsaved edits, prompt Save / Discard / Cancel:
              Save    → save_to_project, then unbind + show picker;
              Discard → unbind + show picker;
              Cancel  → stay on the current canvas.
        """
        if not self._confirm_discard_or_save_current_canvas():
            return
        if self._mission_control_panel is not None:
            self._mission_control_panel.unbind_canvas()
        # Clear the persisted last_canvas so a re-launch after picking New
        # doesn't snap back into the canvas the user just unbound.
        try:
            Config.set_value("Project", "last_canvas", "")
        except Exception as exc:                                # noqa: BLE001
            log_warning(f"[ui] clear last_canvas failed: {exc!r}")
        self.show_project_picker()

    def _confirm_discard_or_save_current_canvas(self) -> bool:
        """Ask the user what to do about unsaved edits.

        Returns True when the caller should proceed (no dirty state, or
        the user picked Save/Discard); False when the user picked Cancel.
        """
        page = self._canvas_page
        if page is None:
            return True
        try:
            dirty = bool(page.is_dirty())
        except Exception:
            dirty = False
        if not dirty:
            return True

        mc = self._mission_control_panel
        file_id = (mc.current_canvas_file_id if mc is not None else "") or (
            mc.current_canvas if mc is not None else ""
        ) or ""
        display = self._canvas_display_name(file_id) or tr(
            "common.current_canvas", "current canvas"
        )

        box = QMessageBox(self)
        box.setWindowTitle(tr("dialog.unsaved_title", "Unsaved changes"))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            tr(
                "dialog.unsaved_text",
                "{name} has unsaved changes. Save before continuing?",
            ).format(name=display)
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()

        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            return self._save_current_canvas()
        return True  # Discard

    def _confirm_replace_canvas_content(self, template_file_name: str) -> bool:
        """Warn before clobbering a clean-but-non-empty canvas with a template.

        Returns True if the user accepts (proceed with replace), False on
        Cancel. Defaults to Cancel for safety.
        """
        mc = self._mission_control_panel
        file_id = (mc.current_canvas_file_id if mc is not None else "") or ""
        display = self._canvas_display_name(file_id) or tr(
            "common.current_canvas", "current canvas"
        )
        tpl_name = template_file_name
        if tpl_name.endswith(".canvas.json"):
            tpl_name = tpl_name[: -len(".canvas.json")]

        box = QMessageBox(self)
        box.setWindowTitle(
            tr("dialog.replace_canvas_title", "Replace canvas content")
        )
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            tr(
                "dialog.replace_canvas_text",
                "Loading template '{tpl}' will replace all nodes and edges "
                "in '{name}'. Continue?",
            ).format(tpl=tpl_name, name=display)
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Ok

    @staticmethod
    def _canvas_display_name(file_id: str) -> str:
        if not file_id:
            return ""
        name = file_id.replace("\\", "/").rsplit("/", 1)[-1]
        if name.endswith(".canvas.json"):
            return name[: -len(".canvas.json")]
        return name

    # ------------------------------------------------------------------
    # History / Policies click handlers
    # ------------------------------------------------------------------
    def _on_training_run_started(self, run_id: str, _label: str) -> None:
        """Snapshot the current canvas + write a pointer in the new run dir.

        Fired by ``AppSignals.training_run_started`` (queued connection from
        the trainer task thread, delivered on the main thread). Failing to
        snapshot is logged but never aborts the training run — the training
        path is owned by the trainer task, not this hook.
        """
        page = self._canvas_page
        project = self._current_project
        if page is None or project is None or not run_id:
            return
        backend_id = self._current_canvas_backend()
        if not backend_id:
            log_warning(
                f"[ui] training_run_started: cannot snapshot canvas for "
                f"run {run_id!r} — no canvas backend bound"
            )
            return
        run_dir = (
            project.path / "training" / "runs" / backend_id / str(run_id)
        )
        try:
            from application.service.canvas_snapshots import CanvasSnapshotStore
            data = page.to_workflow_dict()
            digest = CanvasSnapshotStore(project).write(data, run_dir)
            log_info(
                f"[ui] canvas snapshot saved for run {run_id} → "
                f"{digest[:12]}..."
            )
        except Exception as exc:                                # noqa: BLE001
            log_warning(
                f"[ui] canvas snapshot failed for run {run_id}: {exc!r}"
            )

    def _on_history_run_clicked(self, backend_id: str, run_id: str) -> None:
        """Restore the canvas snapshot captured at the start of ``run_id``.

        Two-step guard mirrors the template-load path:
            1. dirty canvas    → Save / Discard / Cancel prompt;
            2. clean non-empty → "this will overwrite current canvas" prompt.
        """
        if self._history_menu is not None:
            self._history_menu.close()
        if not run_id:
            return
        page = self._canvas_page
        project = self._current_project
        if page is None or project is None:
            return

        cache = get_training_assets()
        asset = cache.find_run(run_id, backend_id=backend_id)
        if asset is None or not asset.path.exists():
            cache.refresh()
            asset = cache.find_run(run_id, backend_id=backend_id)
        if asset is None:
            self.statusBar().showMessage(
                tr(
                    "status.history_missing",
                    "Run [{}] {} no longer exists on disk",
                ).format(backend_id, run_id),
                4000,
            )
            return

        from application.service.canvas_snapshots import CanvasSnapshotStore
        snap = CanvasSnapshotStore(project)
        try:
            digest = snap.digest_for_run(asset.path)
            data = snap.load_snapshot(digest)
        except (FileNotFoundError, ValueError) as exc:
            log_warning(
                f"[ui] history click: snapshot unavailable for run "
                f"{run_id}: {exc}"
            )
            self.statusBar().showMessage(
                tr(
                    "status.history_snapshot_missing",
                    "Canvas snapshot for run [{}] {} is missing on disk",
                ).format(backend_id, run_id),
                5000,
            )
            return

        # Two-step guard — identical to the template-load path so users get
        # the same "save unsaved edits?" + "overwrite current scene?" UX.
        if page.is_dirty():
            if not self._confirm_discard_or_save_current_canvas():
                return
        elif page.instances:
            if not self._confirm_replace_canvas_content(f"run/{run_id}"):
                return

        try:
            page.replace_content_from_dict(data)
        except Exception as exc:                                # noqa: BLE001
            log_error(f"[ui] history snapshot load failed: {exc!r}")
            self.statusBar().showMessage(
                tr(
                    "status.template_apply_failed",
                    "Failed to apply template content",
                ),
                3000,
            )
            return
        self.statusBar().showMessage(
            tr(
                "status.history_loaded",
                "Canvas restored from run {} — Ctrl+S to persist",
            ).format(run_id),
            4000,
        )

    def _on_history_run_delete(self, backend_id: str, run_id: str) -> None:
        """Delete a single run dir + GC its snapshot if no other run uses it."""
        import shutil
        if not run_id or self._current_project is None:
            return
        cache = get_training_assets()
        asset = cache.find_run(run_id, backend_id=backend_id)
        if asset is None:
            cache.refresh()
            asset = cache.find_run(run_id, backend_id=backend_id)
        if asset is None:
            return
        try:
            shutil.rmtree(asset.path)
        except OSError as exc:
            log_error(
                f"[ui] history delete: rmtree {asset.path} failed: {exc!r}"
            )
            self.statusBar().showMessage(
                tr(
                    "status.history_delete_failed",
                    "Failed to delete run {}",
                ).format(run_id),
                4000,
            )
            return
        cache.refresh()
        try:
            from application.service.canvas_snapshots import (
                CanvasSnapshotStore,
                collect_referenced_digests,
            )
            surviving = collect_referenced_digests(self._current_project)
            CanvasSnapshotStore(self._current_project).gc(surviving)
        except Exception as exc:                                # noqa: BLE001
            log_warning(f"[ui] snapshot gc after delete failed: {exc!r}")
        # Rebuild the menu immediately so the row disappears without the
        # user having to re-open the popup.
        self._populate_history_menu()
        self.statusBar().showMessage(
            tr("status.history_deleted", "Run {} deleted").format(run_id),
            3000,
        )

    def _on_history_clear_all(self) -> None:
        """Wipe every leaf run directory + every canvas snapshot.

        Only leaf run dirs are removed; backend partition dirs (e.g.
        ``<runs>/sb3_mujoco/``) are left in place so a future run can still
        land in them without an extra mkdir.
        """
        import shutil
        if self._current_project is None:
            return
        runs = get_training_assets().runs()
        if not runs:
            return
        box = QMessageBox(self)
        box.setWindowTitle(
            tr("history.confirm.clear_title", "Clear all training history?")
        )
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            tr(
                "history.confirm.clear_body",
                "This deletes every run directory under "
                "<project>/training/runs/ and every canvas snapshot in "
                "_runs_/. This cannot be undone.",
            )
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if box.exec() != QMessageBox.StandardButton.Ok:
            return
        for asset in runs:
            try:
                shutil.rmtree(asset.path)
            except OSError as exc:
                log_warning(
                    f"[ui] history clear: rmtree {asset.path} failed: {exc!r}"
                )
        get_training_assets().refresh()
        try:
            from application.service.canvas_snapshots import CanvasSnapshotStore
            CanvasSnapshotStore(self._current_project).gc_all()
        except Exception as exc:                                # noqa: BLE001
            log_warning(f"[ui] snapshot gc_all failed: {exc!r}")
        self._populate_history_menu()
        self.statusBar().showMessage(
            tr("status.history_cleared", "Training history cleared"),
            3000,
        )

    def _on_policy_pick(self, backend_id: str, policy_id: str) -> None:
        """Reveal the bundle dir + persist the active policy.

        Without a deploy panel today we surface the bundle two ways:
          1. Open the bundle directory in the OS file manager so the user
             can confirm the artifact exists where it should.
          2. Persist ``[App] active_policy`` (and ``active_policy_backend``)
             in user.ini so the future deploy panel can pick up the user's
             last selection unambiguously.

        Path resolution goes through the TrainingAssetsCache, scoped by
        ``backend_id`` for exact partition match, with the same
        refresh-on-miss policy as ``_on_history_run_clicked``.
        """
        if not policy_id:
            return
        if self._current_project is None:
            self.statusBar().showMessage(
                tr("status.policy_no_project", "No project loaded"), 3000
            )
            return

        cache = get_training_assets()
        asset = cache.find_policy(policy_id, backend_id=backend_id)
        if asset is None or not asset.path.exists():
            cache.refresh()
            asset = cache.find_policy(policy_id, backend_id=backend_id)

        try:
            Config.set_value("App", "active_policy", policy_id)
            Config.set_value("App", "active_policy_backend", backend_id)
        except Exception as exc:                           # pragma: no cover
            log_warning(f"[ui] policy pick: persist active_policy failed: {exc}")

        if asset is not None and asset.path.exists():
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(asset.path)))
            self.statusBar().showMessage(
                tr("status.policy_selected", "Policy selected: [{}] {}").format(
                    backend_id, policy_id,
                ),
                3000,
            )
        else:
            self.statusBar().showMessage(
                tr(
                    "status.policy_missing",
                    "Policy [{}] {} selected, but bundle path is missing on disk",
                ).format(backend_id, policy_id),
                4000,
            )

    def _apply_ctrl_icons(self) -> None:
        """Bind icon_play / icon_stop SVGs (text fallback if missing)."""
        for btn, icon_name, fallback in (
            (self._start_btn, "icon_play", ">"),
            (self._stop_btn,  "icon_stop", "[]"),
        ):
            if btn is None:
                continue
            p = Assets.find_icon(icon_name)
            if p is not None:
                btn.setIcon(QIcon(str(p)))
                btn.setText("")
            else:
                btn.setIcon(QIcon())
                btn.setText(fallback)

    def _on_target_changed(self, _index: int) -> None:
        if self._target_combo is None:
            return
        data = str(self._target_combo.currentData() or "").strip()
        if not data:
            return
        self.backend_changed.emit(data)

    def set_training_running(self, running: bool) -> None:
        """Toggle the start/stop pair from outside (training backend).

        ■ tracks ``running`` directly. ▶ is *not* a simple ``not running``:
        even when idle we keep ▶ disabled if there is no canvas selected
        (per spec: ▶ only enables when there is a real training target).
        Delegate that to :meth:`_update_start_btn_enabled`.
        """
        if self._stop_btn is not None:
            self._stop_btn.setEnabled(running)
        self._update_start_btn_enabled()

    def _update_canvas_label(self) -> None:
        """Refresh the main_row "[engine] canvas_name" label.

        Source for the active file_id, in priority order:
          1. ``mission_control_panel.current_canvas_file_id`` — set by the
             embedded canvas after a successful auto-load from disk;
          2. ``mission_control_panel.current_canvas`` — the row currently
             selected in the sidebar's Project Files Canvas tab.

        When neither is set: render ``-- --`` in ``main_c1``.
        Otherwise parse the file_id (``[canvas/]<backend_subdir>/.../<stem>.canvas.json``)
        to derive the engine_id via ``backends_registry.resolve_engine_id_from_subdir``,
        and emit RichText with two color spans:
          * engine tag → ``Config.get_color(get_theme_slot(engine_id))`` —
            same per-engine tint used to prefix Canvas-tab rows in
            MissionControlPanel;
          * canvas name → ``Config.get_color("highlight")``.

        Re-called from ``apply_theme`` so theme switches re-stamp the colors.
        """
        if self._canvas_label is None:
            return

        file_id = ""
        if self._mission_control_panel is not None:
            file_id = (
                self._mission_control_panel.current_canvas_file_id
                or self._mission_control_panel.current_canvas
                or ""
            )

        if not file_id:
            idle_color = Config.get_color("main_c1")
            self._canvas_label.setText(
                f'<span style="color:{idle_color};">-- --</span>'
            )
            return

        # Parse "<canvas>/<backend_subdir>/.../<stem>.canvas.json" → (subdir, stem).
        parts = file_id.replace("\\", "/").strip("/").split("/")
        if parts and parts[0] == "canvas":
            parts = parts[1:]
        subdir = parts[0] if len(parts) >= 2 else ""
        stem = parts[-1] if parts else file_id
        if stem.endswith(".canvas.json"):
            stem = stem[: -len(".canvas.json")]

        engine_id = (
            backends_registry.resolve_engine_id_from_subdir(subdir)
            if subdir
            else None
        )
        if engine_id:
            engine_label = backends_registry.get_display_name(engine_id)
            engine_color = Config.get_color(
                backends_registry.get_theme_slot(engine_id)
            )
        else:
            engine_label = subdir or "?"
            engine_color = Config.get_color("main_t2")

        name_color = Config.get_color("highlight")
        engine_html = html.escape(engine_label)
        stem_html = html.escape(stem)
        self._canvas_label.setText(
            f'<span style="color:{engine_color};">[{engine_html}]</span> '
            f'<span style="color:{name_color};">{stem_html}</span>'
        )

    def _update_start_btn_enabled(self) -> None:
        """Re-evaluate the ▶ button's enabled state from current state.

        Enabled iff:
          * no training run is currently active (``_active_task_id`` empty), and
          * a canvas file_id is selected in the sidebar's Project Files list
            (mirrors :meth:`_resolve_training_canvas_dict`'s file_id source,
            without doing the disk read).

        Called: after the main page is built, on ProjectsPanel.canvas_selected
        and MissionControlPanel.canvas_loaded, and at every set_training_running edge.
        """
        if self._start_btn is None:
            return
        if self._active_task_id:
            self._start_btn.setEnabled(False)
            return
        has_target = (
            self._mission_control_panel is not None
            and bool(self._mission_control_panel.current_canvas)
        )
        self._start_btn.setEnabled(has_target)

    def set_progress(self, current: int, total: int, state: str = "running") -> None:
        """Drive the standalone progress row from the outside (training backend)."""
        if self._progress_bar is None:
            return
        self._progress_bar.set_total(total)
        self._progress_bar.set_progress(current=current)
        self._progress_bar.set_state(state)

    def _build_work_splitter(self, parent: QWidget) -> QSplitter:
        """work_zone splitter: canvas_panel (flex) | cmd_column (resizable, min 100).

        Left pane is ``canvas_panel`` -- a thin container whose only child
        today is the empty ``main_panel`` placeholder. Real canvas widgets
        will be added inside ``main_panel`` in later stages without
        re-plumbing the splitter. Right pane is ``cmd_column`` wrapping
        ``CmdLogWidget``; it has a min width of ``_CMD_MIN_W`` and the
        splitter handle is edge-draggable in the DEMO style.
        """
        splitter = QSplitter(Qt.Orientation.Horizontal, parent)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            "QSplitter::handle { background: transparent; border: none; }"
        )

        self._canvas_panel = QWidget(splitter)
        self._canvas_panel.setObjectName("canvasPanel")
        canvas_layout = QVBoxLayout(self._canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        # main_panel hosts the canvas at the bottom (layout-managed via
        # _MainPanel.resizeEvent → setGeometry on canvas) and the
        # MissionControlPanel as an overlay raised above it. The overlay's
        # geometry switches between full-cover (MC mode) and top_row strip
        # (TC mode) on overlay_compact_changed.
        self._main_panel = _MainPanel(self._canvas_panel)
        self._canvas_page = CanvasPage(self._main_panel, embedded=True)
        # Wire Export 节点 ▶ Launch Review 信号 → review subprocess 调度。
        # CanvasPage._scene 在 CanvasPage.__init__ 内已构造（参见 page.py）。
        try:
            scene = getattr(self._canvas_page, "_scene", None)
            if scene is not None and hasattr(scene, "review_launch_requested"):
                scene.review_launch_requested.connect(self._on_review_launch_requested)
        except Exception as exc:
            log_warning(f"[ui] review_launch wire-up failed: {exc}")
        # Local import: see module-top comment — defers the cyclonedds /
        # adapters chain off the Stage 1 paint path.
        from .widgets import MissionControlPanel
        self._mission_control_panel = MissionControlPanel(self._main_panel)
        self._mission_control_panel.set_canvas(self._canvas_page)
        # Persist the (project, canvas) pair to user.ini on every successful
        # canvas load + flip the host panel into canvas mode (the New
        # button might have left it in picker mode).
        self._mission_control_panel.canvas_loaded.connect(
            self._on_canvas_loaded
        )
        # Sidebar nav 按钮的显隐跟随 MissionControl 的 effective mode：
        # Training Canva → 只显示 Node Library；
        # Mission Control → 只显示 Project Files / Robot Asset / Controller。
        # spawn 路由：sidebar.node_requested → 当前 canvas 视口中心 spawn_node。
        if self._sidebar is not None:
            self._mission_control_panel.mode_changed.connect(
                self._on_mc_mode_changed
            )
            # canvas_loaded 也触发一次同步：如果用户上次会话保存的 _mode 已经
            # 是 TC，slider 不会再发 mode_changed,但 _effective_mode 在 canvas
            # 加载瞬间从 forced-MC 翻回 TC,需要补一次刷新。
            self._mission_control_panel.canvas_loaded.connect(
                self._on_mc_canvas_loaded_resync
            )
            self._sidebar.node_requested.connect(
                self._on_sidebar_node_requested
            )
            # Initial sync — use the current effective mode (forced-MC if no
            # canvas is loaded yet).
            self._sidebar.apply_view_mode(
                self._mission_control_panel._effective_mode()
            )
        # Empty-state picker — sibling of canvas+mission_control, mutually
        # exclusive with them via _MainPanel.set_mode. The picker is now a
        # full HomepagePage (four-card landing UI) instead of the legacy
        # NewCanvasForm; the create card's submitted signal still routes
        # directly to MainWindow's _create_and_open_canvas, and a new
        # open_project_requested signal wires workspace-row clicks straight
        # into open_project so the user can jump into an existing project
        # without going through the file picker.
        from application.ui.widgets.homepage import HomepagePage
        self._picker_panel = HomepagePage(parent=self._main_panel)
        self._picker_panel.submitted.connect(self._create_and_open_canvas)
        self._picker_panel.canvas_open_requested.connect(
            self._on_homepage_canvas_open
        )
        # Community card's Download button → reuse the sidebar's
        # apply-update flow with the card-selected ReleaseInfo. Skips
        # UpdateAvailableDialog because the user has already seen the
        # release version + description inline on the card.
        self._picker_panel.community_apply_requested.connect(
            self._launch_update_apply
        )
        self._main_panel.set_children(
            self._canvas_page,
            self._mission_control_panel,
            self._picker_panel,
        )
        canvas_layout.addWidget(self._main_panel, 1)
        splitter.addWidget(self._canvas_panel)

        self._cmd_column = QWidget(splitter)
        self._cmd_column.setObjectName("cmdColumn")
        cmd_layout = QVBoxLayout(self._cmd_column)
        cmd_layout.setContentsMargins(0, 0, 0, 0)
        cmd_layout.setSpacing(0)
        self._cmd_log = CmdLogWidget(self._cmd_column)
        cmd_layout.addWidget(self._cmd_log, 1)
        self._cmd_column.setMinimumWidth(self._CMD_MIN_W)
        splitter.addWidget(self._cmd_column)

        # Initial split: canvas_panel takes flex, cmd_column starts at default.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([
            max(self._DEFAULT_W - self._CMD_DEFAULT_W, self._CMD_MIN_W),
            self._CMD_DEFAULT_W,
        ])
        # Both panes collapsible=False so the user can't accidentally
        # zero-out either side via the splitter handle.
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        self._work_splitter = splitter
        return splitter

    # ------------------------------------------------------------------
    # Training controls (▶ / ■) — Stage 12 wiring
    # ------------------------------------------------------------------
    def _on_start_training(self) -> None:
        if self._active_task_id:
            log_warning(
                tr("training.already_running", "A training run is already active")
            )
            return
        canvas_dict = self._resolve_training_canvas_dict()
        if canvas_dict is None:
            log_warning(tr("training.no_canvas", "No canvas loaded"))
            return
        # Reward × MotionPhase coverage pre-flight (see plan
        # custom-mods-canvas-issaclab-go2-ppo). Blocks training in strict
        # mode when any phase is missing required reward isolation —
        # protects against the Go2 idle-limb-flailing pattern where every
        # reward is unconditional and the static phase has no negative
        # signal to counter unconditional positives like feet_air_time.
        if not self._coverage_preflight_ok(canvas_dict):
            return
        # Deploy-target coverage pre-flight: BEFORE consuming compute,
        # show the user which deploy targets the resulting bundle will
        # support given the current registry state (MJCF / USD per-format
        # tables). If there's a cross-format gap, surface a blocking
        # modal so the user knows they're about to train a run that
        # won't deploy to one of the targets — this is the safety net
        # the Robot Node UX is missing today. User explicitly OK'd
        # proceeding ⇒ continue; cancelled ⇒ abort silently.
        if not self._deploy_coverage_preflight_ok(canvas_dict):
            return
        # Clear any leftover danger_zone marks from a previous failed
        # submit so the user only sees marks relevant to THIS attempt.
        try:
            from application.ui.dialogs import clear_canvas_diagnostic_marks
            clear_canvas_diagnostic_marks(self)
        except Exception:  # noqa: BLE001
            pass
        try:
            from application.training.trainer_runtime import submit_canvas_training
            result = submit_canvas_training(canvas_dict)
        except Exception as exc:  # noqa: BLE001 — split: canvas issues → dialog, real bugs → cmd log
            # Canvas self-check failures (SpecValidationError /
            # CanvasConfigError) are user-actionable misconfigurations,
            # not application crashes — route them to the unified popup
            # so the user sees a structured "which node, which key, why"
            # message instead of a traceback. Anything else (programming
            # bugs, transient IO, etc.) still lands in the cmd log with
            # a full traceback so it stays debuggable.
            from application.ui.dialogs import show_canvas_error_dialog
            if show_canvas_error_dialog(exc, parent=self):
                log_warning(f"[play] canvas check failed: {exc}")
                return
            import traceback
            log_error("[play] submit failed:\n" + traceback.format_exc())
            return
        task_id = str(result.get("task_id") or "")
        self._active_task_id = task_id
        # Writeback: stamp the base_asset node's ``last_run_id`` so the
        # Start Point picker can surface Latest on the next open
        # (cumulative-training affordance). Lives in the canvas's normal
        # save lifecycle — _refresh_dirty_state flags the canvas dirty
        # so the user gets the standard unsaved-changes indicator.
        rid = str(result.get("run_id") or "").strip()
        if rid:
            self._persist_base_asset_last_run(rid)
        # Capture the slot index synchronously from the public manager API.
        # submit() has already routed the task into a slot (or the queue) by
        # the time it returns; get_slot_status() exposes the current task_id
        # per slot. Queued case (-1) falls back to ratio-only progress in the
        # progress_updated handler.
        self._active_slot_idx = -1
        for entry in get_tasks_manager().get_slot_status():
            if entry.get("task_id") == task_id:
                self._active_slot_idx = int(entry.get("index", -1))
                break
        # Disable ▶ + enable ■. set_training_running re-evaluates ▶ via
        # _update_start_btn_enabled (which sees _active_task_id is now set
        # and forces ▶ off), and flips ■ to enabled.
        self.set_training_running(True)
        # Jump to Mission Control mode so the chart + bottom panel are
        # visible while the run progresses.
        if self._mission_control_panel is not None:
            self._mission_control_panel.enter_mission_control_mode()
        # Reset the cmd column to a known width so the Mission Control
        # chart has predictable horizontal room on every run start.
        if self._work_splitter is not None:
            total = self._work_splitter.width()
            cmd_w = 585
            self._work_splitter.setSizes([max(total - cmd_w, 1), cmd_w])
        # Reset the bar so a stale 100% from a prior run isn't visible while
        # we wait for the first MSG_PROGRESS. set_progress will auto-flip
        # IDLE → RUNNING on the first non-zero ratio (LaviProgressBar.set_progress).
        if self._progress_bar is not None:
            self._progress_bar.reset()
        log_info(
            tr(
                "training.started",
                "Training started: run={run} backend={backend} algo={algo}",
            ).format(
                run=result.get("run_id", ""),
                backend=result.get("backend", ""),
                algo=result.get("algorithm", ""),
            )
        )

    def _persist_base_asset_last_run(self, run_id: str) -> None:
        """Stamp ``base_asset.last_run_id`` on the active CanvasPage so
        the Start Point picker surfaces ``Latest`` (cumulative training)
        on the next open. Lives in the canvas's normal save lifecycle —
        ``_refresh_dirty_state`` is invoked so the user gets the standard
        unsaved-changes indicator and can persist via the usual Save
        path.
        """
        page = self._canvas_page
        if page is None:
            return
        # Local import — items.py imports main_window indirectly through
        # a chain that would make a top-level import circular.
        from application.ui.canvas.items import NodeItem
        for ni in page._instances.values():
            if not isinstance(ni, NodeItem):
                continue
            try:
                if ni.manifest.id != "base_asset":
                    continue
            except AttributeError:
                continue
            if ni.params.get("last_run_id") == run_id:
                return
            ni.params["last_run_id"] = run_id
            on_changed = getattr(ni, "on_param_changed", None)
            if callable(on_changed):
                on_changed("last_run_id", run_id)
            try:
                page._refresh_dirty_state()
            except Exception as exc:  # noqa: BLE001
                log_warning(
                    f"[play] base_asset.last_run_id writeback: "
                    f"_refresh_dirty_state failed: {exc!r}"
                )
            return

    def _coverage_preflight_ok(self, canvas_dict: dict) -> bool:
        """Reward × MotionPhase coverage gate before training submits.

        Returns ``True`` when training may proceed, ``False`` when the
        user cancelled out of a critical-coverage modal. Behavior is
        gated by ``[Canvas] reward_coverage_mode``:

        * ``strict``   — critical coverage opens a modal block dialog.
                          Cancel halts ▶; *Continue anyway* proceeds.
        * ``warn``     — critical / warning coverage logs a warning; ▶ proceeds.
        * ``off``      — no check at all.

        The check pulls the first ``rewards`` node out of the canvas
        dict, deserializes the ``reward_terms`` IRParam, runs the
        evaluator against the canonical phase list, and decides from
        the resulting severity. Any registry-layer or schema mismatch
        degrades to "no check" rather than blocking — better to let a
        flaky check pass than to wedge ▶ on an upstream bug.
        """
        try:
            mode = str(
                Config.get_value("Canvas", "reward_coverage_mode", "strict")
                or "strict"
            ).strip().lower()
        except Exception:
            mode = "strict"
        if mode == "off":
            return True
        try:
            from registers import motion_phases as _mp
            from registers import rewards_coverage as _rc
        except Exception as exc:
            log_warning(f"[play] coverage preflight unavailable: {exc!r}")
            return True
        reward_terms: dict = {}
        rewards_node_id: str = ""
        for n in canvas_dict.get("nodes", []) or []:
            if (n.get("schema_id") or "") != "rewards":
                continue
            params = n.get("params", {}) or {}
            entry = params.get("reward_terms")
            raw_val = None
            if isinstance(entry, dict) and "value" in entry:
                raw_val = entry.get("value")
            else:
                raw_val = entry
            if isinstance(raw_val, str):
                try:
                    reward_terms = (
                        json.loads(raw_val) if raw_val.strip() else {}
                    )
                except Exception:
                    reward_terms = {}
            elif isinstance(raw_val, dict):
                reward_terms = dict(raw_val)
            rewards_node_id = str(n.get("id") or "")
            break
        if not reward_terms:
            return True
        phases = [
            {
                "id": p.id,
                "display_name": p.display_name_default,
                "polarity_required": p.polarity_required,
            }
            for p in _mp.list_phases()
        ]
        if not phases:
            return True
        backend = str(canvas_dict.get("backend", "") or "isaac_lab")
        try:
            report = _rc.evaluate(reward_terms, phases, backend=backend)
        except Exception as exc:
            log_warning(f"[play] coverage evaluation failed: {exc!r}")
            return True
        if not report.blocking:
            if report.severity == "warning":
                names = ", ".join(p.display_name for p in report.warning_phases())
                log_warning(
                    f"[play] reward coverage warning on phase(s): {names}"
                )
            return True
        if mode == "warn":
            names = ", ".join(p.display_name for p in report.critical_phases())
            log_warning(
                f"[play] reward coverage CRITICAL on phase(s): {names} "
                f"(mode=warn — proceeding)"
            )
            return True
        # Strict mode — block until user explicitly confirms.
        return self._show_coverage_block_dialog(report, rewards_node_id)

    def _show_coverage_block_dialog(self, report, rewards_node_id: str) -> bool:
        """Modal block dialog. Returns True if user clicked "Continue anyway"."""
        from PyQt6.QtWidgets import QMessageBox
        crit = report.critical_phases()
        warn = report.warning_phases()
        body_lines: List[str] = []
        body_lines.append(tr(
            "training.coverage.blocked.body",
            "Reward coverage is critical — the policy is likely to learn "
            "undesired behaviors when the affected phases are active.",
        ))
        body_lines.append("")
        for ph in crit:
            body_lines.append(f"  ✗  {ph.display_name}: {ph.message}")
        if warn:
            body_lines.append("")
            for ph in warn:
                body_lines.append(f"  ⚠  {ph.display_name}: {ph.message}")
        body_lines.append("")
        body_lines.append(tr(
            "training.coverage.blocked.hint",
            "Double-click the badge on the Rewards node to open the "
            "Coverage Inspector for suggested fixes.",
        ))
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(tr(
            "training.coverage.blocked.title", "Reward Coverage Issue",
        ))
        msg.setText(tr(
            "training.coverage.blocked.header",
            "Cannot start training: reward coverage check failed.",
        ))
        msg.setInformativeText("\n".join(body_lines))
        proceed_btn = msg.addButton(
            tr("training.coverage.blocked.proceed", "Continue anyway"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_btn = msg.addButton(
            tr("training.coverage.blocked.cancel", "Cancel"),
            QMessageBox.ButtonRole.RejectRole,
        )
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        return msg.clickedButton() is proceed_btn

    def _deploy_coverage_preflight_ok(self, canvas_dict: dict) -> bool:
        """Surface deploy-target coverage to the user BEFORE submit consumes compute.

        Pulls ``ir.robot_id`` from the canvas dict, runs
        :func:`application.training.trainer_runtime.compute_deploy_coverage`,
        and if the cross-format IR-role sets disagree, presents a modal
        confirmation listing which deploy targets the trained bundle will
        and won't support. Returns:

          * ``True`` — no gap, OR user explicitly confirmed proceeding;
            training submit may continue.
          * ``False`` — user cancelled. Caller must abort.

        This is the safety net that closes the "training silently
        produces a non-deployable bundle" footgun the Robot Node UX
        doesn't surface (CLAUDE.md §1.8 — deploy-target unavailability
        is a real user-visible consequence, not just a warning to log).
        """
        # Resolve the SKU once. canvas_dict is the IR-shape JSON; SKU is
        # carried at top-level as ``robot_id`` (set by
        # CanvasPage.to_workflow_dict from the Robot node's asset_id).
        sku = str(canvas_dict.get("robot_id") or "").strip()
        if not sku:
            # No robot bound on the canvas — spec_compiler will raise the
            # appropriate UNRESOLVED_ROBOT_ASSET issue downstream. Not our
            # gate to fire.
            return True
        try:
            from application.training.trainer_runtime import (
                compute_deploy_coverage,
            )
            report = compute_deploy_coverage(sku)
        except Exception as exc:  # noqa: BLE001
            # Compute failure shouldn't wedge ▶ — log and let downstream
            # raise if there's a real problem. (Mirrors the
            # _coverage_preflight_ok degradation policy.)
            log_warning(f"[play] deploy-coverage compute failed: {exc!r}")
            return True
        if not report.has_gap:
            return True
        return self._show_deploy_coverage_dialog(report)

    def _show_deploy_coverage_dialog(self, report) -> bool:
        """Modal confirmation listing deploy-target consequences. Returns True to proceed."""
        from PyQt6.QtWidgets import QMessageBox

        # Build a compact summary the user can act on without scrolling.
        affected = "\n".join(f"  ✗  {t}" for t in report.affected_targets)
        # Truncate IR-role lists to keep the dialog readable — full lists
        # are still in the log via _pre_flight_warn_cross_format_coverage.
        def _truncate(items, n=6):
            if len(items) <= n:
                return ", ".join(items)
            shown = ", ".join(items[:n])
            return f"{shown}, … (+{len(items) - n} more)"

        body_lines: list[str] = [
            tr(
                "training.deploy_coverage.body",
                "The trained bundle will be deployable to ONLY a subset of "
                "the supported targets, because the robot's MJCF and USD "
                "asset tables don't declare the same joint set.",
            ),
            "",
            tr("training.deploy_coverage.affected", "Affected deploy targets:"),
            affected,
            "",
        ]
        if report.missing_in_mjcf:
            body_lines.append(tr(
                "training.deploy_coverage.miss_mjcf",
                "Roles declared in USD but missing from MJCF: {roles}",
            ).format(roles=_truncate(report.missing_in_mjcf)))
        if report.missing_in_usd:
            body_lines.append(tr(
                "training.deploy_coverage.miss_usd",
                "Roles declared in MJCF but missing from USD: {roles}",
            ).format(roles=_truncate(report.missing_in_usd)))
        body_lines.append("")
        body_lines.append(tr(
            "training.deploy_coverage.hint",
            "To restore both deploy targets: cancel, open the Robot Assets "
            "sidebar, repoint the smaller-DOF asset to a matching variant "
            "(e.g. for Unitree G1 use scene_with_hands.xml alongside g1.usd), "
            "re-Dump, then re-run Play.",
        ))

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(tr(
            "training.deploy_coverage.title",
            "Reduced deploy-target coverage",
        ))
        msg.setText(tr(
            "training.deploy_coverage.header",
            "Training will produce a bundle with reduced deploy coverage "
            "for robot ‘{name}’ (sku={sku}).",
        ).format(name=report.robot_name, sku=report.sku))
        msg.setInformativeText("\n".join(body_lines))
        proceed_btn = msg.addButton(
            tr("training.deploy_coverage.proceed",
               "Proceed anyway (don't waste compute later)"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_btn = msg.addButton(
            tr("training.deploy_coverage.cancel", "Cancel"),
            QMessageBox.ButtonRole.RejectRole,
        )
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        chosen = msg.clickedButton() is proceed_btn
        if chosen:
            log_warning(
                f"[play] user proceeded despite deploy-coverage gap on "
                f"sku={report.sku!r}: missing_in_mjcf={report.missing_in_mjcf} "
                f"missing_in_usd={report.missing_in_usd}"
            )
        else:
            log_info(
                f"[play] cancelled by user at deploy-coverage modal "
                f"(sku={report.sku!r})"
            )
        return chosen

    def _resolve_training_canvas_dict(self) -> Optional[dict]:
        """Return the IR-shape dict to feed submit_canvas_training, or None.

        Two sources, in priority order:
          1. The host-level ``CanvasPage`` when it has a file loaded
             (``mission_control_panel.current_canvas_file_id`` set) —
             preferred so unsaved edits are picked up.
          2. ``mission_control_panel.current_canvas`` — the file_id selected
             in the sidebar's Project Files list. Read straight from disk
             so the user can hit ▶ without first triggering an auto-load.
        """
        mc = self._mission_control_panel
        page = self._canvas_page
        if mc is None:
            return None
        if page is not None and mc.current_canvas_file_id:
            try:
                return page.to_workflow_dict()
            except Exception as exc:  # noqa: BLE001
                log_error(f"[play] to_workflow_dict failed: {exc!r}")
                return None
        file_id = mc.current_canvas or ""
        if not file_id or self._current_project is None:
            return None
        try:
            abs_path = resolve_file(self._current_project, file_id)
        except ValueError as exc:
            log_error(f"[play] cannot resolve canvas {file_id!r}: {exc}")
            return None
        try:
            data = read_data(Path(abs_path))
        except Exception as exc:  # noqa: BLE001
            log_error(f"[play] read canvas failed {abs_path}: {exc!r}")
            return None
        if not isinstance(data, dict):
            log_error(f"[play] canvas content is not a dict: {abs_path}")
            return None
        return data

    def _on_training_progress_updated(
        self, slot_idx: int, ratio: float, text: str
    ) -> None:
        """Filter progress_updated to our active slot, then drive total+current.

        Every task in the system fires this signal; we ignore those that
        don't belong to the active training run. When the trainer's text
        carries an ``N/M`` pair (SB3: ``step N/M``, Isaac Lab: ``iter N/M``)
        we update both ``set_total`` and ``set_progress(current=N)`` so the
        count label reflects real iter counts instead of 0/100. If no pair
        is present (e.g., init phase) we fall back to ratio-only.
        """
        if self._progress_bar is None:
            return
        if self._active_slot_idx < 0 or slot_idx != self._active_slot_idx:
            return
        m = _TRAINING_PROGRESS_RE.search(text or "")
        if m is not None:
            current = int(m.group(1))
            total = max(1, int(m.group(2)))
            self._progress_bar.set_total(total)
            self._progress_bar.set_progress(current=current)
        else:
            self._progress_bar.set_progress(ratio=ratio)

    def _on_stop_training(self) -> None:
        if not self._active_task_id:
            return
        ok = get_tasks_manager().cancel(self._active_task_id)
        log_info(
            f"[play] cancel requested ok={ok} task_id={self._active_task_id}"
        )

    def _on_training_finished(
        self, task_id: str, success: bool, result: Any
    ) -> None:
        if task_id != self._active_task_id:
            return
        self._active_task_id = ""
        self._active_slot_idx = -1
        self.set_training_running(False)
        log_info(
            f"[play] training finished task_id={task_id} success={success}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _center_on_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        self.move(
            avail.x() + (avail.width() - self._DEFAULT_W) // 2,
            avail.y() + (avail.height() - self._DEFAULT_H) // 2,
        )
