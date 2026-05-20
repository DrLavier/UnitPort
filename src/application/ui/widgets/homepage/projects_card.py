"""HomepageLocalFiles — multi-user × multi-project × multi-canvas table.

Rows are sourced by walking the **workspace root** (the base path that
holds one sub-directory per user — ``_guest`` for the guest workspace,
and a slug like ``alpharius-unitport-ai`` for each signed-in account):

    <workspace_root>/
        _guest/
            projects/<project_id>/
                project.yaml
                canvas/<engine_subdir>/*.canvas.json
        <user_slug>/
            projects/<project_id>/
                project.yaml
                canvas/<engine_subdir>/*.canvas.json

Each row corresponds to **one canvas file** — a project that holds three
canvases yields three rows. Columns:

    User | Project | Canvas | Robot | Engine | Last Modified | Action

Card chrome (top → bottom):

    title row     [ Local Files               | search box ]
    headers row   [ User Project Canvas ...               ]
    rows scroll   [ row, row, row, ...                    ]
    bottom row    [                                Refresh ]

The card re-scans on:

* ``ProjectStore.snapshot_changed`` (active user's project list changed)
* ``AuthManager.workspace_changed``  (active user switched, e.g. sign-in)
* explicit ``refresh()`` (used by language switch and Refresh button)

The search box filters the visible rows by canvas name. Typing is
debounced — the filter applies 50 ms after the user stops typing so
each keystroke does not trigger a full re-layout.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QFontMetrics, QMouseEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from unitport_sdk import (
    Config,
    DataManager,
    I18n,
    Paths,
    SwitchButton,
    i18n_bind,
    log_info,
    log_warning,
    read_data,
    setButton,
    tr,
)

from application.service.auth import get_auth_manager
from application.service.projects import get_project_store
from application.service.user_workspace import (
    migrate_workspace_root,
    read_workspace_root,
    reload_workspace_data,
)

from registers import backends as backends_registry
from registers import robots as robots_registry

from .card_base import HomepageCard


# Column widths — User/Engine/Last Modified/Action are fixed at the
# constants below. Project / Canvas / Robot use ``_COL_*_MIN`` as a
# floor; the actual width is recomputed each refresh to fit the widest
# cell content (header text included), capped at ``_COL_FIT_CAP``.
_COL_USER_W = 96
_COL_ENGINE_W = 90
_COL_TIME_W = 120
_COL_ACTION_W = 96
# Fit-content floors for Project / Canvas / Robot. The starting values
# match the old fixed widths so a freshly-launched empty workspace still
# renders with familiar column proportions.
_COL_PROJECT_MIN = 130
_COL_CANVAS_MIN = 150
_COL_ROBOT_MIN = 110
# Hard ceiling so a pathologically long project / canvas / robot name
# does not push the Action column off the visible area.
_COL_FIT_CAP = 280
# Horizontal padding added to each cell's natural text width so the
# content does not sit flush against the next column.
_COL_FIT_PADDING = 16

# Reserved directory names at the workspace root that are NOT per-user
# workspaces — machine-level state shared by all accounts.
_NON_USER_DIR_NAMES = frozenset({
    "sdk", "registers", "engines", "runtime", "robot_assets",
    "avatars",
})


@dataclass(frozen=True)
class _Row:
    """One scanned canvas — what the table actually renders."""
    user_dir: Path
    user_label: str
    project_path: Path
    project_name: str
    project_id: str
    canvas_path: Path
    canvas_name: str
    canvas_file_id: str         # "canvas/<subdir>/<name>.canvas.json"
    robot_label: str
    engine_subdir: str          # "IsaacLab" / "SB3" / ...
    last_modified: float        # epoch seconds
    is_active_user: bool        # True when user_dir == active USER_CONFIG_DIR


def _user_display_label(user_dir: Path) -> str:
    """Human-readable label for a user directory.

    ``_guest`` → "Guest"; slug dirs → tries the cached ``session.json``
    inside the dir for a display name / email, falling back to the slug.
    """
    if user_dir.name == "_guest":
        return tr("homepage.projects.user_guest", "Guest")
    session_file = user_dir / "session.json"
    if session_file.exists():
        try:
            cached = DataManager.load(session_file)
        except Exception:                                     # noqa: BLE001
            cached = None
        if isinstance(cached, dict):
            label = (
                str(cached.get("display_name") or "").strip()
                or str(cached.get("email") or "").strip()
            )
            if label:
                return label
    return user_dir.name


def _canvas_robot_label(canvas_path: Path) -> str:
    """Read ``robot_id`` from a canvas file and resolve to display name."""
    try:
        data = read_data(canvas_path)
    except Exception as exc:                                  # noqa: BLE001
        log_warning(
            f"[homepage.projects] cannot read canvas {canvas_path}: {exc!r}"
        )
        return tr("homepage.projects.robot_unset", "—")
    if not isinstance(data, dict):
        return tr("homepage.projects.robot_unset", "—")
    sku = str(data.get("robot_id") or "").strip()
    if not sku:
        return tr("homepage.projects.robot_unset", "—")
    entry = robots_registry.get_robot(sku)
    if entry is None:
        resolved = robots_registry.resolve_to_sku(sku)
        if resolved:
            entry = robots_registry.get_robot(resolved)
    if entry is None:
        return sku
    name = str(entry.get("name") or "").strip()
    return name or sku


def _human_relative(ts: float) -> str:
    if not ts or ts <= 0:
        return tr("homepage.projects.time_unknown", "—")
    now = time.time()
    delta = max(0.0, now - ts)
    if delta < 60:
        return tr("homepage.projects.time_just_now", "just now")
    if delta < 3600:
        n = int(delta // 60)
        key = "homepage.projects.time_min_one" if n == 1 else "homepage.projects.time_min_n"
        default = "1 minute ago" if n == 1 else "{n} minutes ago"
        return tr(key, default).format(n=n)
    if delta < 86400:
        n = int(delta // 3600)
        key = "homepage.projects.time_hr_one" if n == 1 else "homepage.projects.time_hr_n"
        default = "1 hour ago" if n == 1 else "{n} hours ago"
        return tr(key, default).format(n=n)
    if delta < 86400 * 2:
        return tr("homepage.projects.time_yesterday", "yesterday")
    if delta < 86400 * 7:
        n = int(delta // 86400)
        return tr("homepage.projects.time_day_n", "{n} days ago").format(n=n)
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _scan_all_workspaces() -> List[_Row]:
    """Walk every user dir under the workspace root and yield one row per canvas."""
    try:
        root = read_workspace_root()
    except Exception as exc:                                  # noqa: BLE001
        log_warning(f"[homepage.projects] read_workspace_root failed: {exc!r}")
        return []
    if not root.exists():
        return []

    try:
        active_user_dir = Path(Paths.USER_CONFIG_DIR).resolve()
    except OSError:
        active_user_dir = Path(Paths.USER_CONFIG_DIR)

    rows: List[_Row] = []
    try:
        user_dirs = sorted(
            (d for d in root.iterdir() if d.is_dir()),
            key=lambda p: p.name.lower(),
        )
    except OSError as exc:
        log_warning(f"[homepage.projects] cannot list {root}: {exc}")
        return []

    for user_dir in user_dirs:
        if user_dir.name in _NON_USER_DIR_NAMES:
            continue
        projects_dir = user_dir / "projects"
        if not projects_dir.exists() or not projects_dir.is_dir():
            continue
        user_label = _user_display_label(user_dir)
        try:
            user_dir_resolved = user_dir.resolve()
        except OSError:
            user_dir_resolved = user_dir
        is_active = user_dir_resolved == active_user_dir

        try:
            project_dirs = sorted(
                (d for d in projects_dir.iterdir() if d.is_dir()),
                key=lambda p: p.name.lower(),
            )
        except OSError as exc:
            log_warning(f"[homepage.projects] cannot list {projects_dir}: {exc}")
            continue

        for project_dir in project_dirs:
            manifest_path = project_dir / "project.yaml"
            if not manifest_path.exists():
                continue
            try:
                manifest = DataManager.load(manifest_path)
            except Exception as exc:                          # noqa: BLE001
                log_warning(
                    f"[homepage.projects] cannot parse {manifest_path}: {exc}"
                )
                continue
            if not isinstance(manifest, dict):
                continue
            project_name = str(manifest.get("name") or project_dir.name)
            project_id = str(manifest.get("project_id") or project_dir.name)

            canvas_root = project_dir / "canvas"
            if not canvas_root.exists() or not canvas_root.is_dir():
                continue
            try:
                engine_dirs = sorted(
                    (d for d in canvas_root.iterdir() if d.is_dir()),
                    key=lambda p: p.name.lower(),
                )
            except OSError:
                continue
            for engine_dir in engine_dirs:
                try:
                    canvas_files = sorted(
                        engine_dir.glob("*.canvas.json"),
                        key=lambda p: p.name.lower(),
                    )
                except OSError:
                    continue
                for cf in canvas_files:
                    name = cf.name
                    if name.lower().endswith(".canvas.json"):
                        stem = name[: -len(".canvas.json")]
                    else:
                        stem = cf.stem
                    try:
                        mtime = cf.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    robot_label = _canvas_robot_label(cf)
                    rows.append(_Row(
                        user_dir=user_dir,
                        user_label=user_label,
                        project_path=project_dir,
                        project_name=project_name,
                        project_id=project_id,
                        canvas_path=cf,
                        canvas_name=stem,
                        canvas_file_id=f"canvas/{engine_dir.name}/{name}",
                        robot_label=robot_label,
                        engine_subdir=engine_dir.name,
                        last_modified=mtime,
                        is_active_user=is_active,
                    ))
    # Active-user rows surface first; within each group newest canvas
    # wins, ties broken by user / project / canvas name.
    rows.sort(key=lambda r: (
        0 if r.is_active_user else 1,
        -r.last_modified,
        r.user_label.lower(),
        r.project_name.lower(),
        r.canvas_name.lower(),
    ))
    return rows


class _ElidedLabel(QLabel):
    """QLabel that elides its text with ``ElideRight`` when too narrow."""

    _PADDING = 4

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_text = text or ""
        super().setText(self._full_text)
        self.setToolTip(self._full_text)

    def setText(self, text: str) -> None:                  # type: ignore[override]
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._apply_elide()

    def text(self) -> str:                                  # type: ignore[override]
        return self._full_text

    def resizeEvent(self, ev) -> None:                      # noqa: D401
        super().resizeEvent(ev)
        self._apply_elide()

    def _apply_elide(self) -> None:
        if not self._full_text:
            super().setText("")
            return
        fm = self.fontMetrics()
        budget = max(0, self.width() - self._PADDING)
        super().setText(
            fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight, budget)
        )


def _engine_color_slot(subdir: str) -> str:
    eid = backends_registry.resolve_engine_id_from_subdir(subdir or "")
    return backends_registry.get_theme_slot(eid or "")


class _LocalFilesRow(QFrame):
    """One row in the local-files table."""

    load_clicked = pyqtSignal(str, str)       # project_path, canvas_file_id
    reveal_clicked = pyqtSignal(str)          # canvas absolute path
    delete_clicked = pyqtSignal(str)          # canvas absolute path

    # Item height — pinned so the trailing action buttons can compute a
    # deterministic ``_ROW_HEIGHT - 2`` height (per spec: "button height
    # equals item height minus 2"). Inner vertical margins shrink to 1 px
    # accordingly so the tall buttons fit without forcing the row taller.
    _ROW_HEIGHT = 36

    def __init__(
        self,
        row: _Row,
        widths: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._row = row
        self.setObjectName("homepageLocalFilesRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(self._ROW_HEIGHT)

        engine_color = Config.get_color(_engine_color_slot(row.engine_subdir))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 1, 10, 1)
        layout.setSpacing(8)

        # User — safe_zone tint for the active workspace (the rows that
        # are pinned to the top), muted tone for cross-workspace rows.
        layout.addWidget(self._make_cell(
            row.user_label, _COL_USER_W,
            object_name=(
                "homepageLfCellUserActive" if row.is_active_user
                else "homepageLfCellMuted"
            ),
            elide=True,
        ))
        # Project — primary tone. Width is fit-to-content per refresh
        # (computed by the host card and passed via ``widths``); no
        # elide needed since the column is sized to its widest entry.
        layout.addWidget(self._make_cell(
            row.project_name, widths["project"],
            object_name="homepageLfCellName",
            elide=False,
        ))
        # Canvas — primary tone, fit-to-content. This is the column the
        # search box filters against.
        layout.addWidget(self._make_cell(
            row.canvas_name, widths["canvas"],
            object_name="homepageLfCellCanvas",
            elide=False,
        ))
        # Robot — muted tone, fit-to-content.
        layout.addWidget(self._make_cell(
            row.robot_label, widths["robot"],
            object_name="homepageLfCell",
            elide=False,
        ))
        # Engine — coloured per-backend via the registry's theme slot
        # mapping; inline stylesheet because the colour is per-row.
        engine_label = self._make_cell(
            row.engine_subdir, _COL_ENGINE_W,
            object_name="homepageLfCellEngine",
            elide=True,
        )
        engine_label.setStyleSheet(
            f"QLabel#homepageLfCellEngine {{"
            f"  color: {engine_color};"
            f"  background: transparent;"
            f"  font-size: {Config.get_font_size('size_small')}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
        )
        layout.addWidget(engine_label)
        # Last Modified — muted tone.
        layout.addWidget(self._make_cell(
            _human_relative(row.last_modified), _COL_TIME_W,
            object_name="homepageLfCell",
            elide=True,
        ))

        layout.addStretch(1)

        layout.addLayout(self._build_action_box())

    def _make_cell(
        self,
        text: str,
        width: int,
        *,
        object_name: str,
        elide: bool = False,
    ) -> QLabel:
        lbl: QLabel = _ElidedLabel(text) if elide else QLabel(text)
        lbl.setObjectName(object_name)
        lbl.setFixedWidth(width)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        return lbl

    def _build_action_box(self) -> QHBoxLayout:
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        # Three icon-only action buttons: Load (▶) · Reveal (📁) · Delete (×).
        # All three are always enabled — cross-user gating moved from a UI
        # block to an accountability model (CrossUserAuditService) in the
        # save / delete pipelines. See application.service.cross_user_audit
        # for the why; the visual "this row belongs to another user" hint
        # is the active-user colour on the User cell.

        load_btn = self._make_icon_btn(
            object_name="homepageLfLoadBtn",
            icon_stem="icon_play_alt",
            enabled=True,
            tooltip=tr("homepage.projects.load_tooltip", "Open this canvas"),
        )
        load_btn.clicked.connect(
            lambda: self.load_clicked.emit(
                str(self._row.project_path), self._row.canvas_file_id
            )
        )
        box.addWidget(load_btn)

        reveal_btn = self._make_icon_btn(
            object_name="homepageLfRevealBtn",
            icon_stem="icon_prj",
            enabled=True,
            tooltip=tr(
                "homepage.projects.reveal_tooltip", "Show in folder"
            ),
        )
        reveal_btn.clicked.connect(
            lambda: self.reveal_clicked.emit(str(self._row.canvas_path))
        )
        box.addWidget(reveal_btn)

        del_btn = self._make_icon_btn(
            object_name="homepageLfDeleteBtn",
            icon_stem="icon_delete",
            enabled=True,
            tooltip=tr(
                "homepage.projects.delete_tooltip", "Delete this canvas file"
            ),
        )
        del_btn.clicked.connect(
            lambda: self.delete_clicked.emit(str(self._row.canvas_path))
        )
        box.addWidget(del_btn)
        return box

    def _make_icon_btn(
        self,
        *,
        object_name: str,
        icon_stem: str,
        enabled: bool,
        tooltip: str,
    ) -> QPushButton:
        """Icon-only action button — unified ``light`` kind (transparent
        bg + transparent border, ``hover_1`` lift on hover) at a height
        of ``_ROW_HEIGHT - 2`` so the trio reads as part of the row chrome
        rather than as nested boxed controls.
        """
        btn = setButton(
            f"homepage.projects.action.{object_name}",
            32, self._ROW_HEIGHT - 2,
            kind="light",
            icon=icon_stem,
            icon_only=True,
            disabled=not enabled,
        )
        btn.setObjectName(object_name)
        # setButton derives icon size from height (≈ height-12), which at
        # 34 px height balloons the 16-px SVGs to 22 px and looks crowded
        # against the row labels. Pin to 16 to keep the icon weight even
        # across the trio.
        btn.setIconSize(QSize(16, 16))
        btn.setToolTip(tooltip)
        return btn

    @property
    def canvas_name(self) -> str:
        return self._row.canvas_name

    @property
    def is_snapshot(self) -> bool:
        """True when this row belongs to a project's ``_runs_`` snapshot dir."""
        return self._row.engine_subdir == "_runs_"

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:  # noqa: D401
        if ev.button() == Qt.MouseButton.LeftButton:
            self.load_clicked.emit(
                str(self._row.project_path), self._row.canvas_file_id
            )
        super().mouseDoubleClickEvent(ev)


class HomepageLocalFiles(HomepageCard):
    """Bottom-left homepage card. Multi-user canvas-file table.

    Layout (top → bottom):

    * **Title row** — "Local Files" on the left, a debounced search box
      on the right. The search box filters visible rows by canvas name
      (the ``Canvas`` column) 50 ms after the user stops typing.
    * **Headers row** — column labels in ``size_small`` / ``main_t1``,
      no background.
    * **Rows scroll** — one ``_LocalFilesRow`` per canvas file. Rows
      belonging to the currently-active workspace (the signed-in
      account, or guest when signed-out) are pinned to the top and have
      their ``User`` cell tinted ``safe_zone``. Each row paints a 1-px,
      6-px-radius ``btn_1`` border with no fill; rows are separated by
      5 px of layout spacing.
    * **Bottom row** — ``Refresh`` button at the right edge.
    """

    canvas_open_requested = pyqtSignal(str, str)   # project_path, canvas_file_id

    _SEARCH_DEBOUNCE_MS = 50

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        embedded: bool = False,
    ) -> None:
        super().__init__(
            title_key="homepage.projects.title",
            title_default="Local Files",
            parent=parent,
            embedded=embedded,
        )

        self._store = get_project_store()
        self._auth = get_auth_manager()

        # Filter state — last-applied query, lowercased.
        self._filter_text = ""

        # Title-row right cluster: [ Refresh-icon ] [ Search ] [ hidden action_btn ].
        # The base class added an addStretch(1) between the title and
        # the trailing action slot, so widgets inserted just before
        # ``_action_btn`` land at the right edge.
        self._search_edit = QLineEdit()
        self._search_edit.setObjectName("homepageLfSearch")
        self._search_edit.setPlaceholderText(tr(
            "homepage.projects.search_placeholder", "Search canvases…"
        ))
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setFixedWidth(180)
        self._search_edit.setFixedHeight(26)
        insert_at = max(0, self.title_row.count() - 1)
        self.title_row.insertWidget(
            insert_at, self._search_edit,
            0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        # Refresh button — icon-only LaviButton in ``light`` kind so it
        # paints flat against the card chrome. The factory does the
        # i18n key plumbing for the (suppressed) text; we bind the
        # tooltip separately so language switches still re-translate it.
        #
        # The LaviButton factory derives ``iconSize`` from ``height``
        # (``max(12, h-12)``), which for the homepage chrome's small
        # icon buttons would render the SVG too thin to read. Override
        # to a more legible 18 px after construction.
        self._refresh_btn = setButton(
            "homepage.projects.refresh",
            width=28, height=28,
            kind="light", spec="none",
            icon="icon_refresh", icon_only=True,
            default="Refresh",
        )
        self._refresh_btn.setObjectName("homepageLfRefreshBtn")
        self._refresh_btn.setIconSize(QSize(18, 18))
        self._refresh_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        i18n_bind(
            self._refresh_btn, "setToolTip",
            "homepage.projects.refresh_tooltip", "Refresh local files",
        )
        self._refresh_btn.clicked.connect(self.refresh)
        # Inserted before the search edit so the title row reads:
        #   title …stretch… [refresh] [search]  (action_btn stays hidden)
        refresh_at = max(0, self.title_row.count() - 2)
        self.title_row.insertWidget(
            refresh_at, self._refresh_btn,
            0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        # Snapshot toggle — "Snapshot: [SwitchButton]" sits to the left of
        # the refresh button with a leading spacer so the cluster reads as
        # a distinct control rather than crowding the refresh / search
        # pair. Off (default) hides rows whose canvas lives under the
        # per-project ``_runs_/`` snapshot directory; on reveals them.
        self._show_snapshots = False
        self._snapshot_label = QLabel(
            tr("homepage.projects.snapshot_label", "Snapshot:")
        )
        self._snapshot_label.setObjectName("homepageLfSnapshotLabel")
        self._snapshot_label.setProperty(
            "i18n_key", "homepage.projects.snapshot_label"
        )
        self._snapshot_label.setProperty("i18n_default", "Snapshot:")
        self._snapshot_switch = SwitchButton(checked=self._show_snapshots)
        self._snapshot_switch.toggled.connect(self._on_snapshot_toggled)

        ref_idx = self.title_row.indexOf(self._refresh_btn)
        # Insert in reverse so the visible order ends up as
        #   [stretch] [Snapshot:] [Switch] [gap] [refresh] [search] …
        self.title_row.insertSpacing(ref_idx, 16)
        self.title_row.insertWidget(
            ref_idx, self._snapshot_switch,
            0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self.title_row.insertWidget(
            ref_idx, self._snapshot_label,
            0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        # Debounce timer — 50 ms after the user stops typing the filter
        # is re-applied. Using a single-shot timer means rapid keystrokes
        # restart the clock instead of triggering a filter per character.
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(self._SEARCH_DEBOUNCE_MS)
        self._search_timer.timeout.connect(self._apply_filter)
        self._search_edit.textChanged.connect(self._on_search_text_changed)

        # Content spacing: title-row → table → bottom-row sit on the
        # outer ``content_layout`` with a generous gap so the three
        # card sections read as distinct strips. The table itself
        # (headers + rows) lives in a nested host with tight spacing so
        # the header line stays visually anchored to its column data.
        # The 8-px top margin matches the bump applied between sections
        # — the base ``root`` layout already places the title row 12 px
        # above ``content_host``, and the extra 8 brings it in line.
        self.content_layout.setContentsMargins(0, 8, 0, 0)
        self.content_layout.setSpacing(18)

        # ---- table host (headers + rows scroll + empty label) ----
        self._table_host = QWidget(self)
        self._table_host.setObjectName("homepageLfTableHost")
        table_v = QVBoxLayout(self._table_host)
        table_v.setContentsMargins(0, 0, 0, 0)
        table_v.setSpacing(6)
        self._build_header(table_v)

        self._scroll = QScrollArea(self._table_host)
        self._scroll.setObjectName("homepageLfScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._rows_host = QWidget()
        self._rows_host.setObjectName("homepageLfRowsHost")
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(5)
        self._rows_layout.addStretch(1)

        self._scroll.setWidget(self._rows_host)
        table_v.addWidget(self._scroll, 1)

        self._empty_label = QLabel(tr(
            "homepage.projects.empty",
            "No local workspaces yet — create your first project on the right.",
        ))
        self._empty_label.setObjectName("homepageLfEmpty")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setVisible(False)
        table_v.addWidget(self._empty_label)

        self.content_layout.addWidget(self._table_host, 1)

        # ---- bottom row (Migrate + Open Local Workspace, right-aligned) ----
        self._build_bottom_row()

        # React to background changes.
        self._store.snapshot_changed.connect(self._on_snapshot_changed)
        self._auth.workspace_changed.connect(self._on_workspace_changed)
        I18n.instance().language_changed.connect(self.refresh)

        self._apply_table_theme()
        self.refresh()

    # ------------------------------------------------------------------

    def _build_header(self, parent_layout: QVBoxLayout) -> None:
        host = QFrame(self._table_host)
        host.setObjectName("homepageLfHeader")
        host.setFrameShape(QFrame.Shape.NoFrame)
        host.setFixedHeight(24)
        # Header host must refuse horizontal compression — otherwise the
        # cells inside (which inherit QLabel's default Preferred policy)
        # would let the parent layout shrink them under pressure, while
        # the row widgets (with Fixed-policy cells, see ``_make_cell``)
        # keep their natural width inside the scroll area. The
        # asymmetry shows up as misaligned columns when the card is
        # squeezed.
        host.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(host)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        def _make(key: str, default: str, width: int) -> QLabel:
            lbl = QLabel(tr(key, default))
            lbl.setObjectName("homepageLfHeaderCell")
            lbl.setFixedWidth(width)
            # Match the row cell policy in ``_make_cell`` — Fixed horizontal
            # so the layout cannot squeeze the header cells below their
            # column width.
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            lbl.setProperty("i18n_key", key)
            lbl.setProperty("i18n_default", default)
            return lbl

        self._h_user = _make("homepage.projects.col_user", "User", _COL_USER_W)
        # Project / Canvas / Robot start at their floor widths; the
        # refresh path recomputes them to fit the widest cell content.
        self._h_project = _make(
            "homepage.projects.col_project", "Project", _COL_PROJECT_MIN,
        )
        self._h_canvas = _make(
            "homepage.projects.col_canvas", "Canvas", _COL_CANVAS_MIN,
        )
        self._h_robot = _make(
            "homepage.projects.col_robot", "Robot", _COL_ROBOT_MIN,
        )
        self._h_engine = _make("homepage.projects.col_engine", "Engine", _COL_ENGINE_W)
        self._h_time = _make("homepage.projects.col_time", "Last Modified", _COL_TIME_W)
        self._h_action = _make("homepage.projects.col_action", "Action", _COL_ACTION_W)

        for w in (
            self._h_user, self._h_project, self._h_canvas, self._h_robot,
            self._h_engine, self._h_time,
        ):
            layout.addWidget(w)
        layout.addStretch(1)
        layout.addWidget(self._h_action)

        parent_layout.addWidget(host)

    def _build_bottom_row(self) -> None:
        """Bottom row: Migrate · Open Local Workspace →, left-aligned.

        Both buttons share the ``light`` + ``safe`` kind/spec pair — the
        SDK's LaviButton factory paints a flat, safe-zone-tinted control
        with no background fill, matching the rest of the homepage
        chrome. Refresh moved to the title row (icon-only).
        """
        host = QWidget(self)
        host.setObjectName("homepageLfBottomRow")
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._migrate_btn = setButton(
            "homepage.projects.btn_migrate",
            width=92, height=28,
            kind="light", spec="save",
            default="Migrate",
        )
        self._migrate_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._migrate_btn.clicked.connect(self._on_migrate_clicked)
        self._migrate_btn.setVisible(False)
        layout.addWidget(self._migrate_btn)

        # Open Local Workspace — minimal hyperlink-style text button.
        # Deliberately NOT a LaviButton: no fill, no border, no hover
        # state — the link should read as inline text the user can
        # click, not a CTA competing with Migrate. Re-translation on
        # language switch is handled in ``refresh()``.
        self._open_ws_btn = QPushButton(
            tr("homepage.projects.btn_open_workspace", "Open Local Workspace →")
        )
        self._open_ws_btn.setObjectName("homepageLfOpenWsLink")
        self._open_ws_btn.setFlat(True)
        self._open_ws_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._open_ws_btn.setProperty(
            "i18n_key", "homepage.projects.btn_open_workspace"
        )
        self._open_ws_btn.setProperty(
            "i18n_default", "Open Local Workspace →"
        )
        self._open_ws_btn.clicked.connect(self._on_open_workspace_clicked)
        layout.addWidget(self._open_ws_btn)

        layout.addStretch(1)

        self.content_layout.addWidget(host)

    # ------------------------------------------------------------------
    # Bottom-row button handlers
    # ------------------------------------------------------------------

    def _on_open_workspace_clicked(self) -> None:
        """Reveal the workspace root in the OS file explorer.

        The workspace root is the multi-user parent dir (one sub-dir per
        signed-in account + ``_guest``); opening it lets the user manage
        files outside the homepage table.
        """
        try:
            root = read_workspace_root()
        except Exception as exc:                              # noqa: BLE001
            log_warning(
                f"[homepage.projects] open workspace failed: {exc!r}"
            )
            return
        if not root.exists():
            log_warning(
                f"[homepage.projects] workspace root missing: {root}"
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _on_migrate_clicked(self) -> None:
        """Relocate the workspace root to a user-chosen folder.

        Flow: pick folder → confirm → ``migrate_workspace_root`` (moves
        the tree + rewrites ``system.ini[Workspace].root`` + re-anchors
        ``USER_CONFIG_DIR``) → ``reload_workspace_data`` (re-hydrate
        DataManager / registries / ProjectStore from the new tree) →
        ``refresh()`` (rescan the table). All branches are toast-style:
        the heavy lifting is in the service layer; this method is just
        glue + user dialogs.
        """
        try:
            current = read_workspace_root()
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[homepage.projects] migrate: read_workspace_root failed: {exc!r}"
            )
            return

        start_dir = str(current.parent) if current.exists() else str(current)
        chosen = QFileDialog.getExistingDirectory(
            self.window(),
            tr("homepage.projects.migrate_title", "Choose new workspace root"),
            start_dir,
        )
        if not chosen:
            return
        new_root = Path(chosen)

        ans = QMessageBox.question(
            self.window(),
            tr("homepage.projects.migrate_confirm_title", "Migrate workspace"),
            tr(
                "homepage.projects.migrate_confirm_body",
                "Move the workspace root from\n  {old}\nto\n  {new}?\n\n"
                "All projects, canvases, and per-user data will be moved. "
                "The app will reload its workspace state when done.",
            ).format(old=str(current), new=str(new_root)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        ok, err = migrate_workspace_root(new_root)
        if not ok:
            log_warning(
                f"[homepage.projects] workspace migration failed: {err}"
            )
            QMessageBox.warning(
                self.window(),
                tr("homepage.projects.migrate_failed_title", "Migration failed"),
                tr(
                    "homepage.projects.migrate_failed_body",
                    "Could not migrate workspace:\n{err}",
                ).format(err=err),
            )
            return

        # Re-hydrate caches so the new tree's contents are seen by the
        # ProjectStore / registries / DataManager. Done unconditionally
        # even if it raises — at worst the user sees stale rows until a
        # manual Refresh and a log line points at the root cause.
        try:
            reload_workspace_data()
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[homepage.projects] reload_workspace_data failed: {exc!r}"
            )

        self.refresh()

        try:
            final_root = read_workspace_root()
        except Exception:                                         # noqa: BLE001
            final_root = new_root
        log_info(
            f"[homepage.projects] workspace migrated to {final_root}"
        )
        QMessageBox.information(
            self.window(),
            tr("homepage.projects.migrate_success_title", "Migration complete"),
            tr(
                "homepage.projects.migrate_success_body",
                "Workspace root is now:\n{path}",
            ).format(path=str(final_root)),
        )

    # ------------------------------------------------------------------

    def _on_snapshot_changed(self, _snapshot: object) -> None:
        self.refresh()

    def _on_workspace_changed(self, _uid: str) -> None:
        # Active user dir flipped — re-scan so the pinned-rows order and
        # the safe_zone tinting follow the new active workspace.
        self.refresh()

    def _on_search_text_changed(self, _text: str) -> None:
        # Restart the debounce clock. The actual filter runs once 50 ms
        # of keyboard silence elapses (see ``_apply_filter``).
        self._search_timer.start()

    def _on_snapshot_toggled(self, checked: bool) -> None:
        self._show_snapshots = bool(checked)
        self._update_row_visibility()

    def _apply_filter(self) -> None:
        self._filter_text = self._search_edit.text().strip().lower()
        self._update_row_visibility()

    def _update_row_visibility(self) -> None:
        """Show/hide each row based on the current canvas-name filter."""
        needle = self._filter_text
        show_snaps = self._show_snapshots
        any_visible = False
        # _rows_layout's last entry is the trailing stretch; skip it.
        for i in range(self._rows_layout.count() - 1):
            item = self._rows_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if not isinstance(w, _LocalFilesRow):
                continue
            visible = (not needle) or (needle in w.canvas_name.lower())
            if visible and w.is_snapshot and not show_snaps:
                visible = False
            w.setVisible(visible)
            if visible:
                any_visible = True

        # When the filter wipes the table empty, surface the placeholder
        # (re-using the empty-state label is fine — the wording is
        # appropriate either way).
        has_rows = self._rows_layout.count() > 1
        if has_rows and not any_visible:
            self._scroll.setVisible(True)
            self._empty_label.setVisible(False)
        elif has_rows:
            self._scroll.setVisible(True)
            self._empty_label.setVisible(False)
        else:
            self._scroll.setVisible(False)
            self._empty_label.setVisible(True)

    def refresh(self) -> None:
        # Re-translate strings that live outside the row widgets.
        for w in (
            self._h_user, self._h_project, self._h_canvas, self._h_robot,
            self._h_engine, self._h_time, self._h_action,
        ):
            key = w.property("i18n_key") or ""
            default = w.property("i18n_default") or ""
            if key:
                w.setText(tr(str(key), str(default)))
        self._empty_label.setText(tr(
            "homepage.projects.empty",
            "No local workspaces yet — create your first project on the right.",
        ))
        self._search_edit.setPlaceholderText(tr(
            "homepage.projects.search_placeholder", "Search canvases…"
        ))
        self._snapshot_label.setText(tr(
            "homepage.projects.snapshot_label", "Snapshot:"
        ))
        self._open_ws_btn.setText(tr(
            "homepage.projects.btn_open_workspace",
            "Open Local Workspace →",
        ))

        while self._rows_layout.count() > 1:
            it = self._rows_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        rows = _scan_all_workspaces()
        if not rows:
            self._scroll.setVisible(False)
            self._empty_label.setVisible(True)
            self._apply_table_theme()
            return
        self._scroll.setVisible(True)
        self._empty_label.setVisible(False)

        # Resize Project / Canvas / Robot columns to fit the widest
        # entry in this batch (plus the localised header text). Headers
        # and rows are kept in lockstep so the column boundaries align.
        widths = self._compute_fit_widths(rows)
        self._h_project.setFixedWidth(widths["project"])
        self._h_canvas.setFixedWidth(widths["canvas"])
        self._h_robot.setFixedWidth(widths["robot"])

        for row in rows:
            widget = _LocalFilesRow(row, widths, parent=self._rows_host)
            widget.load_clicked.connect(self.canvas_open_requested.emit)
            widget.reveal_clicked.connect(self._on_reveal_clicked)
            widget.delete_clicked.connect(self._on_delete_clicked)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, widget)

        # Re-apply the current filter — refresh() may be triggered by a
        # workspace change while the user has a search term pending; we
        # do not want a stale filter to silently un-filter the table.
        self._update_row_visibility()

        self._apply_table_theme()

    # ------------------------------------------------------------------
    # Column width fitting
    # ------------------------------------------------------------------
    def _compute_fit_widths(self, rows: List[_Row]) -> dict:
        """Return per-column pixel widths sized to the widest cell.

        Walks the row dataset + the localised header strings, measuring
        each candidate text via the body font's :class:`QFontMetrics`.
        Each column's width is ``max(floor, widest_natural + padding)``
        clamped at :data:`_COL_FIT_CAP` so a pathologically long entry
        cannot push the Action column out of view.
        """
        fm = QFontMetrics(self._h_project.font())

        def fit(values, floor):
            widest = max((fm.horizontalAdvance(v or "") for v in values),
                         default=0)
            return max(floor, min(widest + _COL_FIT_PADDING, _COL_FIT_CAP))

        project_w = fit(
            [tr("homepage.projects.col_project", "Project")]
            + [r.project_name for r in rows],
            _COL_PROJECT_MIN,
        )
        canvas_w = fit(
            [tr("homepage.projects.col_canvas", "Canvas")]
            + [r.canvas_name for r in rows],
            _COL_CANVAS_MIN,
        )
        robot_w = fit(
            [tr("homepage.projects.col_robot", "Robot")]
            + [r.robot_label for r in rows],
            _COL_ROBOT_MIN,
        )
        return {
            "project": project_w,
            "canvas": canvas_w,
            "robot": robot_w,
        }

    # ------------------------------------------------------------------
    # Reveal flow
    # ------------------------------------------------------------------

    def _on_reveal_clicked(self, canvas_path: str) -> None:
        """Open the canvas file's location in the OS file explorer.

        On Windows we shell out to ``explorer /select,<path>`` so the
        target file is highlighted inside its folder — a noticeably
        better UX than opening the parent dir blind. On other platforms
        we fall back to ``QDesktopServices.openUrl`` on the parent dir
        (macOS Finder / Linux file manager will open the folder).
        """
        if not canvas_path:
            return
        target = Path(canvas_path)
        if not target.exists():
            log_warning(
                f"[homepage.projects] reveal target missing: {canvas_path}"
            )
            return
        if sys.platform == "win32":
            try:
                # ``explorer /select,<path>`` highlights the file. We avoid
                # check=True because explorer.exe often returns non-zero
                # exit codes even when the window opens successfully.
                import subprocess
                subprocess.Popen(["explorer", f"/select,{target}"])
                return
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[homepage.projects] explorer /select failed: {exc!r} — "
                    "falling back to QDesktopServices"
                )
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    # ------------------------------------------------------------------
    # Delete flow
    # ------------------------------------------------------------------

    def _on_delete_clicked(self, canvas_path: str) -> None:
        if not canvas_path:
            return
        target = Path(canvas_path)
        if not target.exists():
            log_warning(f"[homepage.projects] delete target missing: {canvas_path}")
            self.refresh()
            return

        # Classify cross-user FIRST — affects both the confirm copy and
        # whether we need to pre-snapshot for the audit log.
        from application.service import cross_user_audit
        target_uid, rel_path = cross_user_audit.classify_target(target)
        is_cross_user = bool(target_uid and rel_path)

        if is_cross_user:
            owner_label = self._owner_label_for_uid(target_uid)
            confirm_title = tr(
                "homepage.projects.cross_user_delete_warn_title",
                "Delete another user's canvas",
            )
            confirm_body = tr(
                "homepage.projects.cross_user_delete_warn_body",
                ("'{name}' belongs to {owner}. Deleting it now will be logged "
                 "so {owner} can review or restore it on their next "
                 "sign-in.\n\nDelete anyway?"),
            ).format(name=target.name, owner=owner_label)
        else:
            confirm_title = tr(
                "homepage.projects.delete_confirm_title", "Delete canvas",
            )
            confirm_body = tr(
                "homepage.projects.delete_confirm_body",
                ("Permanently delete '{name}'?\n\nThis only removes the canvas "
                 "file — the project itself is kept."),
            ).format(name=target.name)

        ans = QMessageBox.question(
            self.window(),
            confirm_title, confirm_body,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        # Capture pre-state BEFORE unlink — the bytes themselves stay in
        # memory only (no local snapshot, see cross_user_audit Phase 1
        # scope) but pre_sha256 + pre_size are diagnostic in the entry.
        pre_state = (
            cross_user_audit.capture_pre_state(target) if is_cross_user else None
        )

        try:
            target.unlink()
            log_info(f"[homepage.projects] deleted canvas {target}")
        except Exception as exc:                              # noqa: BLE001
            log_warning(
                f"[homepage.projects] delete failed {target}: {exc!r}"
            )
            QMessageBox.warning(
                self.window(),
                tr("homepage.projects.delete_failed_title", "Delete failed"),
                tr(
                    "homepage.projects.delete_failed_body",
                    "Could not delete '{name}':\n{err}",
                ).format(name=target.name, err=str(exc)),
            )
            return

        if is_cross_user and pre_state is not None:
            cross_user_audit.record_delete(
                target, pre_state, note="homepage_delete",
            )

        try:
            self._store.refresh_snapshot()
        except Exception:                                     # noqa: BLE001
            pass
        self.refresh()

    def _owner_label_for_uid(self, target_uid: str) -> str:
        """Resolve a display name for the cross-user audit confirm copy.

        Reuses ``_user_display_label`` which reads ``session.json`` from
        the owner's workspace dir. Falls back to the raw uid string when
        the dir is gone or the cache is missing.
        """
        try:
            from application.service.user_workspace import read_workspace_root
            root = read_workspace_root()
            return _user_display_label(root / target_uid) or target_uid
        except Exception:                                     # noqa: BLE001
            return target_uid

    # ------------------------------------------------------------------

    def apply_theme(self) -> None:                            # type: ignore[override]
        super().apply_theme()
        self._apply_table_theme()

    def _apply_table_theme(self) -> None:
        header_fg = Config.get_color("main_t1")
        # Rows: no fill, btn_1 border, hover lift via row_1.
        row_border = Config.get_color("btn_1")
        row_hover = Config.get_color("row_1")
        text = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        active_user_fg = Config.get_color("safe_zone")
        search_bg = Config.get_color("bg_3")
        search_border = Config.get_color("border_2")
        search_focus = Config.get_color("hover_1")
        size_small = Config.get_font_size("size_small")
        size_mini = Config.get_font_size("size_mini")

        qss = (
            # Title-row search input.
            f"QLineEdit#homepageLfSearch {{"
            f"  background: {search_bg};"
            f"  color: {text};"
            f"  border: 1px solid {search_border};"
            f"  border-radius: 6px;"
            f"  padding: 2px 8px;"
            f"  font-size: {size_small}px;"
            f"}}"
            f"QLineEdit#homepageLfSearch:focus {{"
            f"  border: 1px solid {search_focus};"
            f"}}"
            # Snapshot toggle label — small muted caption next to the
            # switch ("Snapshot: [○]"). Same tier as the search input.
            f"QLabel#homepageLfSnapshotLabel {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            # Headers — size_small, no background, main_t1.
            f"QFrame#homepageLfHeader {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageLfHeaderCell {{"
            f"  color: {header_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QScrollArea#homepageLfScroll, QWidget#homepageLfRowsHost {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            # Rows — no fill, btn_1 border, 6px radius.
            f"QFrame#homepageLocalFilesRow {{"
            f"  background: transparent;"
            f"  border: 1px solid {row_border};"
            f"  border-radius: 6px;"
            f"}}"
            f"QFrame#homepageLocalFilesRow:hover {{"
            f"  background: {row_hover};"
            f"}}"
            # Active-user 'User' cell — safe_zone tint, marks rows that
            # belong to the signed-in account (the pinned-to-top group).
            f"QLabel#homepageLfCellUserActive {{"
            f"  color: {active_user_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageLfCellMuted {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageLfCellName {{"
            f"  color: {text};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 500;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageLfCellCanvas {{"
            f"  color: {text};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 500;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageLfCell {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_mini}px;"
            f"  border: none;"
            f"}}"
            # Row action trio (Load · Reveal · Delete) is owned by
            # LaviButton (``kind="light"`` via setButton in
            # ``_LocalFilesRow._make_icon_btn``). No id-selector rules
            # here — they would defeat the SDK button's own QSS.
            # Bottom-row + table host — transparent passthroughs.
            # Migrate is a LaviButton (factory-owned QSS); the title-row
            # Refresh is similarly factory-styled. Open Local Workspace
            # is a plain QPushButton styled here as a flat hyperlink:
            # no fill, no border, no padding, and we explicitly pin the
            # ``:hover`` block to the same palette as the default state
            # so Qt's built-in hover tint does not kick in.
            f"QWidget#homepageLfBottomRow, QWidget#homepageLfTableHost {{"
            f"  background: transparent;"
            f"}}"
            f"QPushButton#homepageLfOpenWsLink {{"
            f"  color: {Config.get_color('safe_zone')};"
            f"  background: transparent;"
            f"  border: none;"
            f"  padding: 0;"
            f"  text-align: left;"
            f"  font-size: {size_small}px;"
            f"}}"
            f"QPushButton#homepageLfOpenWsLink:hover, "
            f"QPushButton#homepageLfOpenWsLink:pressed, "
            f"QPushButton#homepageLfOpenWsLink:focus {{"
            f"  color: {Config.get_color('safe_zone')};"
            f"  background: transparent;"
            f"  border: none;"
            f"  text-decoration: none;"
            f"}}"
            f"QLabel#homepageLfEmpty {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  padding: 24px;"
            f"  border: none;"
            f"}}"
        )
        self.setStyleSheet(self.styleSheet() + qss)
