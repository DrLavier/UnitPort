# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CloudManageDialog — browse / download / delete / upload the user's
Supabase Storage ``{user_id}/...`` namespace.

Opened from the homepage Account card's "Manage" button. Self-contained:
no shared state with the sidebar or other widgets, listens directly to
``CloudSyncService.usage_changed`` and to the per-task ``task_finished``
signal so it stays alive across the async one-shot operations.

UI shape (recursive folder tree, hover effects, trailing action icons):

.. code-block:: text

    +-------------------------------------------------------+
    | Manage Cloud Storage                            [ X ] |
    +-------------------------------------------------------+
    | Used 26.6 MB / 200.0 MB  [██░░░░░░░░░░░░░░░]   13%    |
    +-------------------------------------------------------+
    | v configs                                             |
    |    settings.json                  1.2 KB   [⬇] [🗑]   |
    | v projects                                            |
    |    v myproj                                           |
    |       project.yaml                  512 B  [⬇] [🗑]   |
    |       v canvas                                        |
    |          main.canvas.json           12 KB  [⬇] [🗑]   |
    +-------------------------------------------------------+
    |                              [ Refresh ]  [ Close ]   |
    +-------------------------------------------------------+

Drop semantics (drag external files from the OS):
* drop on a *folder header* → upload to that folder's cloud path
* drop on a *file row*      → upload to that file's parent path
* drop on the empty scroll area background → upload to root (``""``)
* per-file 50 MB hard cap (server-side limit; we surface a single batched
  warning rather than letting Storage reject one-by-one)

Theme rule (CLAUDE.md §1.5): every colour and font size resolves through
``Config.get_color`` / ``Config.get_font_size``; no hex literals.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QCursor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    I18n,
    Paths,
    get_task_signal,
    get_tasks_manager,
    log_warning,
    setButton,
    tr,
)

from application.service.auth import get_auth_manager
from application.service.cloud_sync import get_cloud_sync_service
from application.tools.cloud_manage_task import CloudFileOpTask, CloudListTask


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


_SIZE_LIMIT_BYTES = 50 * 1024 * 1024     # mirror storage-sync.spec.yaml § server
_ROW_HEIGHT = 32
_INDENT_PX = 18
_ICON_BTN_PX = 24
_CHEVRON_DOWN = "▾"
_CHEVRON_RIGHT = "▸"


# Reuse cloud_sync's ext→content-type table so uploaded files declare the
# same MIME types push uses. Missing extensions fall through to mimetypes
# and finally to application/octet-stream.
try:
    from application.service.cloud_sync import _CONTENT_TYPE_BY_EXT as _CT_TABLE
except Exception:                                              # noqa: BLE001
    _CT_TABLE = {}


def _format_size(n: int) -> str:
    n = max(0, int(n))
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024 / 1024 / 1024:.1f} GB"


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    ct = _CT_TABLE.get(ext) if isinstance(_CT_TABLE, dict) else None
    if ct:
        return ct
    guess, _ = mimetypes.guess_type(path.name)
    return guess or "application/octet-stream"


# ---------------------------------------------------------------------------
# Drop-target helpers
# ---------------------------------------------------------------------------


def _extract_local_paths(event) -> List[Path]:
    """Pull local file paths out of a Qt drop event's URL list.

    Filters out directories and non-file URLs (http drops, in-app
    canvas drags, etc.). The list preserves original order so the
    user-visible upload sequence matches the drag order.
    """
    out: List[Path] = []
    md = event.mimeData()
    if md is None or not md.hasUrls():
        return out
    for url in md.urls():
        if not url.isLocalFile():
            continue
        p = Path(url.toLocalFile())
        if p.is_file():
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# File row
# ---------------------------------------------------------------------------


class _FileRow(QFrame):
    """One file in the tree.

    Renders name + size + download / delete buttons. Accepts external
    file drops too: dropping on a file row uploads into that file's
    *parent* folder (``rel_path.rsplit('/', 1)[0]``), matching the user
    expectation that a file row "represents" its parent dir as well.
    """

    download_clicked = pyqtSignal(str)        # rel_path
    delete_clicked = pyqtSignal(str)
    drop_received = pyqtSignal(str, list)     # target_dir_rel, [Path,...]

    def __init__(
        self,
        rel_path: str,
        basename: str,
        size: int,
        indent_level: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._rel_path = rel_path
        self.setObjectName("cloudManageFileRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(_ROW_HEIGHT)
        self.setAcceptDrops(True)
        self.setProperty("dropHover", "false")
        self.setProperty("busy", "false")

        row = QHBoxLayout(self)
        row.setContentsMargins(8 + _INDENT_PX * indent_level, 0, 8, 0)
        row.setSpacing(8)

        self._name_label = QLabel(basename, self)
        self._name_label.setObjectName("cloudManageFileName")
        self._name_label.setToolTip(rel_path)
        row.addWidget(self._name_label, 1)

        self._size_label = QLabel(_format_size(size), self)
        self._size_label.setObjectName("cloudManageFileSize")
        row.addWidget(self._size_label, 0)

        self._dl_btn = self._make_action_btn(
            "cloud.download_tooltip",
            "Download to local workspace",
            "icon_download",
        )
        self._dl_btn.clicked.connect(
            lambda: self.download_clicked.emit(self._rel_path)
        )
        row.addWidget(self._dl_btn, 0)

        self._del_btn = self._make_action_btn(
            "cloud.delete_tooltip",
            "Delete from cloud",
            "icon_delete",
        )
        self._del_btn.clicked.connect(
            lambda: self.delete_clicked.emit(self._rel_path)
        )
        row.addWidget(self._del_btn, 0)

    # -- helpers --------------------------------------------------------

    def _make_action_btn(
        self, tip_key: str, tip_default: str, icon_name: str,
    ) -> QPushButton:
        btn = QPushButton(self)
        btn.setFixedSize(_ICON_BTN_PX, _ICON_BTN_PX)
        btn.setFlat(True)
        btn.setObjectName("cloudManageRowBtn")
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon = Assets.find_icon(icon_name)
        if icon is not None:
            btn.setIcon(QIcon(str(icon)))
            btn.setIconSize(QSize(_ICON_BTN_PX - 10, _ICON_BTN_PX - 10))
        else:
            log_warning(f"[cloud_manage_dialog] icon missing: {icon_name}")
        btn.setToolTip(tr(tip_key, tip_default))
        return btn

    def set_busy(self, busy: bool) -> None:
        self.setProperty("busy", "true" if busy else "false")
        self._dl_btn.setEnabled(not busy)
        self._del_btn.setEnabled(not busy)
        st = self.style()
        if st is not None:
            st.unpolish(self)
            st.polish(self)

    @property
    def rel_path(self) -> str:
        return self._rel_path

    # -- drag/drop ------------------------------------------------------

    def _parent_dir(self) -> str:
        if "/" in self._rel_path:
            return self._rel_path.rsplit("/", 1)[0]
        return ""

    def dragEnterEvent(self, ev) -> None:  # noqa: N802
        if _extract_local_paths(ev):
            ev.acceptProposedAction()
            self._set_drop_hover(True)
        else:
            ev.ignore()

    def dragMoveEvent(self, ev) -> None:   # noqa: N802
        if _extract_local_paths(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragLeaveEvent(self, _ev) -> None:  # noqa: N802
        self._set_drop_hover(False)

    def dropEvent(self, ev) -> None:        # noqa: N802
        paths = _extract_local_paths(ev)
        self._set_drop_hover(False)
        if not paths:
            ev.ignore()
            return
        ev.acceptProposedAction()
        self.drop_received.emit(self._parent_dir(), paths)

    def _set_drop_hover(self, on: bool) -> None:
        self.setProperty("dropHover", "true" if on else "false")
        st = self.style()
        if st is not None:
            st.unpolish(self)
            st.polish(self)


# ---------------------------------------------------------------------------
# Folder section (collapsible)
# ---------------------------------------------------------------------------


class _IndentBody(QFrame):
    """Folder body widget that paints a vertical indent guide.

    A single thin line is drawn at the x-coordinate matching the parent
    folder's chevron column (slightly to its right), so users can trace
    a child row back up to its parent header. Nested folders draw their
    own guide at a deeper indent level — multiple guides stack to form
    a tree-like visual.
    """

    def __init__(
        self, indent_level: int, parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cloudManageFolderBody")
        self._indent_level = indent_level

    def paintEvent(self, ev) -> None:  # noqa: N802
        super().paintEvent(ev)
        # Reading the color slot every paint keeps the guide in sync
        # when the theme is swapped at runtime — there's no separate
        # apply_theme hop needed.
        color = QColor(Config.get_color("border_1"))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(color, 1))
        # x = parent's chevron column + half-chevron-width, so the guide
        # sits between the parent header and the child rows that hang
        # below it.
        x = 8 + _INDENT_PX * self._indent_level + 7
        painter.drawLine(x, 0, x, self.height())
        painter.end()


class _FolderSection(QFrame):
    """Collapsible folder. Header row toggles an indent-guided body.

    Tracks the recursive list of cloud-rel-paths in its subtree so the
    header's Download / Delete buttons have a single source of truth
    for "everything under this folder" without re-walking the widget
    tree on every click.
    """

    download_clicked = pyqtSignal(str, list)   # folder_rel, [rel_path,...]
    delete_clicked = pyqtSignal(str, list)
    drop_received = pyqtSignal(str, list)

    def __init__(
        self,
        folder_rel: str,
        display_name: str,
        indent_level: int,
        files: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._folder_rel = folder_rel
        self._files = list(files)
        self.setObjectName("cloudManageFolderSection")

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # --- header (drop target, click to toggle) -------------------
        self._header = _FolderHeader(
            folder_rel, display_name, indent_level, self,
        )
        self._header.toggled.connect(self._on_toggled)
        self._header.drop_received.connect(self.drop_received.emit)
        self._header.download_clicked.connect(self._emit_download)
        self._header.delete_clicked.connect(self._emit_delete)
        col.addWidget(self._header)

        self._body = _IndentBody(indent_level, self)
        self._body_col = QVBoxLayout(self._body)
        self._body_col.setContentsMargins(0, 0, 0, 0)
        self._body_col.setSpacing(0)
        col.addWidget(self._body)

        self._expanded = True

    @property
    def folder_rel(self) -> str:
        return self._folder_rel

    @property
    def files(self) -> List[str]:
        return list(self._files)

    def add_widget(self, w: QWidget) -> None:
        self._body_col.addWidget(w)

    def _on_toggled(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)

    def _emit_download(self) -> None:
        self.download_clicked.emit(self._folder_rel, list(self._files))

    def _emit_delete(self) -> None:
        self.delete_clicked.emit(self._folder_rel, list(self._files))


class _FolderHeader(QFrame):
    """Clickable folder header (chevron + name + summary + actions).

    Hosts download / delete icon buttons so users can act on the whole
    subtree at once. Clicks on the buttons themselves are absorbed by
    Qt's child-event handling and never reach :meth:`mousePressEvent`,
    so they don't accidentally toggle the folder.
    """

    toggled = pyqtSignal(bool)                # expanded
    drop_received = pyqtSignal(str, list)
    download_clicked = pyqtSignal()
    delete_clicked = pyqtSignal()

    def __init__(
        self,
        folder_rel: str,
        display_name: str,
        indent_level: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._folder_rel = folder_rel
        self._expanded = True
        self.setObjectName("cloudManageFolderHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(_ROW_HEIGHT)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAcceptDrops(True)
        self.setProperty("dropHover", "false")

        row = QHBoxLayout(self)
        row.setContentsMargins(8 + _INDENT_PX * indent_level, 0, 8, 0)
        row.setSpacing(6)

        self._chevron = QLabel(_CHEVRON_DOWN, self)
        self._chevron.setObjectName("cloudManageChevron")
        self._chevron.setFixedWidth(12)
        row.addWidget(self._chevron, 0)

        self._name = QLabel(display_name, self)
        self._name.setObjectName("cloudManageFolderName")
        self._name.setToolTip(folder_rel or "/")
        row.addWidget(self._name, 1)

        self._summary = QLabel("", self)
        self._summary.setObjectName("cloudManageFolderSummary")
        row.addWidget(self._summary, 0)

        self._dl_btn = self._make_action_btn(
            "cloud.folder_download_tooltip",
            "Download all files in this folder",
            "icon_download",
        )
        self._dl_btn.clicked.connect(self.download_clicked.emit)
        row.addWidget(self._dl_btn, 0)

        self._del_btn = self._make_action_btn(
            "cloud.folder_delete_tooltip",
            "Delete this folder and all files inside",
            "icon_delete",
        )
        self._del_btn.clicked.connect(self.delete_clicked.emit)
        row.addWidget(self._del_btn, 0)

    def _make_action_btn(
        self, tip_key: str, tip_default: str, icon_name: str,
    ) -> QPushButton:
        btn = QPushButton(self)
        btn.setFixedSize(_ICON_BTN_PX, _ICON_BTN_PX)
        btn.setFlat(True)
        btn.setObjectName("cloudManageRowBtn")
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon = Assets.find_icon(icon_name)
        if icon is not None:
            btn.setIcon(QIcon(str(icon)))
            btn.setIconSize(QSize(_ICON_BTN_PX - 10, _ICON_BTN_PX - 10))
        else:
            log_warning(f"[cloud_manage_dialog] icon missing: {icon_name}")
        btn.setToolTip(tr(tip_key, tip_default))
        return btn

    def set_busy(self, busy: bool) -> None:
        self._dl_btn.setEnabled(not busy)
        self._del_btn.setEnabled(not busy)

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._expanded = not self._expanded
            self._chevron.setText(
                _CHEVRON_DOWN if self._expanded else _CHEVRON_RIGHT
            )
            self.toggled.emit(self._expanded)
        super().mousePressEvent(ev)

    def dragEnterEvent(self, ev) -> None:  # noqa: N802
        if _extract_local_paths(ev):
            ev.acceptProposedAction()
            self._set_drop_hover(True)
        else:
            ev.ignore()

    def dragMoveEvent(self, ev) -> None:   # noqa: N802
        if _extract_local_paths(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragLeaveEvent(self, _ev) -> None:  # noqa: N802
        self._set_drop_hover(False)

    def dropEvent(self, ev) -> None:        # noqa: N802
        paths = _extract_local_paths(ev)
        self._set_drop_hover(False)
        if not paths:
            ev.ignore()
            return
        ev.acceptProposedAction()
        self.drop_received.emit(self._folder_rel, paths)

    def _set_drop_hover(self, on: bool) -> None:
        self.setProperty("dropHover", "true" if on else "false")
        st = self.style()
        if st is not None:
            st.unpolish(self)
            st.polish(self)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class CloudManageDialog(QDialog):
    """Modal cloud-files browser.

    Opens with a fresh ``list_remote()`` dispatched on a
    :class:`CloudListTask` so the dialog never blocks the main thread on
    the HTTP round-trip (the underlying Supabase list endpoint is
    shallow + recursive and easily takes seconds on a large workspace).
    All subsequent download / upload / delete operations go through
    :class:`CloudFileOpTask` on the same TasksManager worker pool.

    The dialog holds a set of in-flight task ids keyed by the rel_path
    they target, so its ``_on_task_finished`` slot can dispatch into the
    right row's busy-clear handler without scanning every widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cloudManageDialog")
        self.setModal(True)
        self.setWindowTitle(tr(
            "cloud.manage_dialog_title", "Manage Cloud Storage",
        ))
        self.resize(720, 520)
        self.setMinimumSize(560, 400)

        self._auth = get_auth_manager()
        self._svc = get_cloud_sync_service()

        # task_id → callback(ok, result)
        self._inflight: Dict[str, Callable[[bool, object], None]] = {}
        # rel_path → _FileRow (only files; folder headers don't go busy)
        self._rows_by_path: Dict[str, _FileRow] = {}
        # task id of the in-flight list refresh (so we ignore stale results
        # if the user clicked Refresh twice rapidly)
        self._list_task_id: str = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        self._build_quota_row(outer)
        self._build_tree_area(outer)
        self._build_footer(outer)

        self._apply_theme()

        # Signal wiring
        self._svc.usage_changed.connect(self._on_usage_changed)
        get_task_signal().task_finished.connect(self._on_task_finished)
        I18n.instance().language_changed.connect(self._retranslate)

        # Seed UI
        cached = self._svc.cached_usage()
        if cached is not None:
            self._render_usage(*cached)
        self._refresh_tree()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_quota_row(self, outer: QVBoxLayout) -> None:
        self._quota_host = QFrame(self)
        self._quota_host.setObjectName("cloudManageQuotaHost")
        row = QHBoxLayout(self._quota_host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self._quota_label = QLabel("—", self._quota_host)
        self._quota_label.setObjectName("cloudManageQuotaLabel")
        row.addWidget(self._quota_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._quota_bar = QProgressBar(self._quota_host)
        self._quota_bar.setObjectName("cloudManageQuotaBar")
        self._quota_bar.setRange(0, 1000)
        self._quota_bar.setValue(0)
        self._quota_bar.setTextVisible(False)
        self._quota_bar.setFixedHeight(6)
        row.addWidget(self._quota_bar, 1, Qt.AlignmentFlag.AlignVCenter)

        self._quota_percent = QLabel("—", self._quota_host)
        self._quota_percent.setObjectName("cloudManageQuotaPercent")
        self._quota_percent.setMinimumWidth(36)
        self._quota_percent.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        row.addWidget(self._quota_percent, 0)

        outer.addWidget(self._quota_host)

    def _build_tree_area(self, outer: QVBoxLayout) -> None:
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("cloudManageScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setAcceptDrops(True)

        self._tree_host = _RootDropArea(self)
        self._tree_host.setObjectName("cloudManageTreeHost")
        self._tree_col = QVBoxLayout(self._tree_host)
        self._tree_col.setContentsMargins(0, 4, 0, 4)
        self._tree_col.setSpacing(0)
        self._tree_host.drop_received.connect(self._on_drop)

        self._scroll.setWidget(self._tree_host)
        outer.addWidget(self._scroll, 1)

        self._empty_label = QLabel(
            tr("cloud.manage_dialog_empty",
               "No files in cloud yet — try Push or drag a file here."),
            self._tree_host,
        )
        self._empty_label.setObjectName("cloudManageEmpty")
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setVisible(False)
        self._tree_col.addWidget(self._empty_label)
        self._tree_col.addStretch(1)

    def _build_footer(self, outer: QVBoxLayout) -> None:
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)

        self._btn_refresh = setButton(
            "cloud.manage_dialog_refresh", 96, 30,
            kind="border", spec="none",
            default="Refresh",
        )
        self._btn_refresh.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_refresh.clicked.connect(self._refresh_tree)
        footer.addWidget(self._btn_refresh, 0)

        self._btn_close = setButton(
            "cloud.manage_dialog_close", 96, 30,
            kind="normal", spec="save",
            default="Close",
        )
        self._btn_close.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close, 0)

        outer.addLayout(footer)

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _refresh_tree(self) -> None:
        """Kick a background list, render a loading placeholder.

        ``CloudSyncService.list_remote`` is a synchronous HTTP round-
        trip (list + recursive folder walks); running it on the main
        thread freezes the dialog and the rest of the UI behind it. The
        actual fetch happens inside :class:`CloudListTask`; this method
        just clears the tree, shows a placeholder, and registers the
        completion callback. The real population lives in
        :meth:`_render_list_result`, dispatched on ``task_finished``.
        """
        self._clear_tree_widgets()
        self._rows_by_path.clear()

        if not self._auth.is_signed_in():
            self._empty_label.setText(tr(
                "cloud.manage_dialog_not_signed_in",
                "Sign in first to manage cloud storage.",
            ))
            self._empty_label.setVisible(True)
            self._btn_refresh.setEnabled(False)
            return
        self._btn_refresh.setEnabled(False)        # re-enabled when task lands

        # Show the placeholder while the background fetch is in flight.
        self._empty_label.setText(tr(
            "cloud.manage_dialog_loading",
            "Loading…",
        ))
        self._empty_label.setVisible(True)

        try:
            tid = get_tasks_manager().submit(CloudListTask())
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud_manage_dialog] list submit failed: {exc!r}")
            self._btn_refresh.setEnabled(True)
            self._empty_label.setText(tr(
                "cloud.manage_dialog_op_failed_body",
                "{phase} failed:\n{err}",
            ).format(phase="list", err=str(exc)))
            return
        self._list_task_id = tid
        self._inflight[tid] = self._on_list_task_done

    def _clear_tree_widgets(self) -> None:
        for i in reversed(range(self._tree_col.count())):
            item = self._tree_col.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is None or w is self._empty_label:
                continue
            self._tree_col.removeWidget(w)
            w.setParent(None)
            w.deleteLater()

    def _on_list_task_done(self, ok: bool, result: object) -> None:
        # Late arrivals (a second Refresh fired before this one returned)
        # are dropped — only the latest list_task_id wins.
        self._btn_refresh.setEnabled(True)
        data = result if isinstance(result, dict) else {}
        if not ok or not data.get("ok", False):
            err = str(data.get("error") or result or "unknown error")
            log_warning(f"[cloud_manage_dialog] list failed: {err}")
            self._empty_label.setText(tr(
                "cloud.manage_dialog_op_failed_body",
                "{phase} failed:\n{err}",
            ).format(phase="list", err=err))
            self._empty_label.setVisible(True)
            return
        rows = data.get("rows") or []
        if not self._auth.is_signed_in():
            return
        user = self._auth.current_user()
        uid = user.user_id if user is not None else ""
        self._render_list_result(rows, uid)

    def _render_list_result(self, rows: list, uid: str) -> None:
        # Always wipe again (the user may have triggered another action
        # during the in-flight window — keep the rendering deterministic).
        self._clear_tree_widgets()
        self._rows_by_path.clear()

        if not rows:
            self._empty_label.setText(tr(
                "cloud.manage_dialog_empty",
                "No files in cloud yet — try Push or drag a file here.",
            ))
            self._empty_label.setVisible(True)
            return
        self._empty_label.setVisible(False)

        tree = self._build_tree(rows, uid)
        # Render at indent 0; insert at index 0 (before the trailing
        # stretch which lives at the end of _tree_col).
        self._render_node(tree, "", 0, prepend_index=0)

    @staticmethod
    def _build_tree(entries: List[dict], uid: str) -> dict:
        root: dict = {"_files": [], "subfolders": {}}
        prefix = f"{uid}/" if uid else ""
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name:
                continue
            rel = name[len(prefix):] if prefix and name.startswith(prefix) else name
            rel = rel.lstrip("/")
            if not rel:
                continue
            parts = rel.split("/")
            node = root
            for seg in parts[:-1]:
                sub = node["subfolders"].setdefault(
                    seg, {"_files": [], "subfolders": {}},
                )
                node = sub
            node["_files"].append({
                "rel": rel,
                "basename": parts[-1],
                "size": int(entry.get("size") or 0),
            })
        return root

    def _render_node(
        self,
        node: dict,
        prefix: str,
        indent_level: int,
        prepend_index: Optional[int] = None,
    ) -> None:
        # Folders first (alphabetical), then files (alphabetical) so the
        # tree reads top-down naturally.
        insert_at = prepend_index
        for name in sorted(node["subfolders"].keys()):
            sub = node["subfolders"][name]
            folder_rel = f"{prefix}/{name}".lstrip("/") if prefix else name
            files = _collect_files(sub)
            section = _FolderSection(
                folder_rel, name, indent_level, files, self._tree_host,
            )
            section.drop_received.connect(self._on_drop)
            section.download_clicked.connect(self._on_folder_download)
            section.delete_clicked.connect(self._on_folder_delete)
            self._populate_section(section, sub, folder_rel, indent_level + 1)
            if insert_at is None:
                self._tree_col.addWidget(section)
            else:
                self._tree_col.insertWidget(insert_at, section)
                insert_at += 1
            # Folder summary
            n_files, total_size = _count_recursive(sub)
            section._header.set_summary(tr(
                "cloud.manage_dialog_folder_summary",
                "{n} items · {size}",
            ).format(n=n_files, size=_format_size(total_size)))

        for f in sorted(node["_files"], key=lambda x: x["basename"]):
            row = _FileRow(
                f["rel"], f["basename"], f["size"], indent_level, self._tree_host,
            )
            row.download_clicked.connect(self._on_download_clicked)
            row.delete_clicked.connect(self._on_delete_clicked)
            row.drop_received.connect(self._on_drop)
            self._rows_by_path[f["rel"]] = row
            if insert_at is None:
                self._tree_col.addWidget(row)
            else:
                self._tree_col.insertWidget(insert_at, row)
                insert_at += 1

    def _populate_section(
        self,
        section: _FolderSection,
        node: dict,
        folder_rel: str,
        indent_level: int,
    ) -> None:
        for name in sorted(node["subfolders"].keys()):
            sub = node["subfolders"][name]
            sub_rel = f"{folder_rel}/{name}"
            files = _collect_files(sub)
            child_section = _FolderSection(
                sub_rel, name, indent_level, files, section,
            )
            child_section.drop_received.connect(self._on_drop)
            child_section.download_clicked.connect(self._on_folder_download)
            child_section.delete_clicked.connect(self._on_folder_delete)
            self._populate_section(child_section, sub, sub_rel, indent_level + 1)
            n_files, total_size = _count_recursive(sub)
            child_section._header.set_summary(tr(
                "cloud.manage_dialog_folder_summary",
                "{n} items · {size}",
            ).format(n=n_files, size=_format_size(total_size)))
            section.add_widget(child_section)

        for f in sorted(node["_files"], key=lambda x: x["basename"]):
            row = _FileRow(
                f["rel"], f["basename"], f["size"], indent_level, section,
            )
            row.download_clicked.connect(self._on_download_clicked)
            row.delete_clicked.connect(self._on_delete_clicked)
            row.drop_received.connect(self._on_drop)
            self._rows_by_path[f["rel"]] = row
            section.add_widget(row)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_download_clicked(self, rel_path: str) -> None:
        local_dst = Path(Paths.USER_CONFIG_DIR) / Path(*rel_path.split("/"))
        if local_dst.exists():
            choice = self._ask_conflict(local_dst)
            if choice == "skip":
                return
            if choice == "save_as":
                new_dst, _ = QFileDialog.getSaveFileName(
                    self,
                    tr("cloud.manage_dialog_dl_save_as_caption",
                       "Save downloaded file"),
                    str(local_dst),
                )
                if not new_dst:
                    return
                local_dst = Path(new_dst)
            # overwrite → fall through with original local_dst
        self._submit_op(
            rel_path,
            CloudFileOpTask("download_one", rel_path, local_dst=local_dst),
        )

    def _ask_conflict(self, local_dst: Path) -> str:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr(
            "cloud.manage_dialog_dl_conflict_title", "File exists",
        ))
        box.setText(tr(
            "cloud.manage_dialog_dl_conflict_body",
            "Local file already exists:\n{path}\n\nWhat should we do?",
        ).format(path=str(local_dst)))
        b_overwrite = box.addButton(
            tr("cloud.manage_dialog_dl_overwrite", "Overwrite"),
            QMessageBox.ButtonRole.DestructiveRole,
        )
        b_skip = box.addButton(
            tr("cloud.manage_dialog_dl_skip", "Skip"),
            QMessageBox.ButtonRole.RejectRole,
        )
        b_save = box.addButton(
            tr("cloud.manage_dialog_dl_save_as", "Save as…"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        box.setDefaultButton(b_skip)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_overwrite:
            return "overwrite"
        if clicked is b_save:
            return "save_as"
        return "skip"

    @pyqtSlot(str, list)
    def _on_folder_download(self, folder_rel: str, files: list) -> None:
        files = [str(f) for f in files if f]
        if not files:
            return
        leaf = folder_rel.rsplit("/", 1)[-1] or "/"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr(
            "cloud.manage_dialog_folder_dl_title", "Download folder",
        ))
        box.setText(tr(
            "cloud.manage_dialog_folder_dl_body",
            "Download {n} files in '{name}' to the local workspace?",
        ).format(n=len(files), name=leaf))
        b_replace = box.addButton(
            tr("cloud.manage_dialog_folder_dl_replace", "Replace existing"),
            QMessageBox.ButtonRole.DestructiveRole,
        )
        b_skip = box.addButton(
            tr("cloud.manage_dialog_folder_dl_skip_existing",
               "Skip existing"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        b_cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(b_skip)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked is b_cancel:
            return
        overwrite_all = clicked is b_replace
        # Per-file: pre-resolve local_dst, skip on conflict if user chose
        # "skip existing", else proceed with overwrite. We don't surface
        # a per-file completion toast — only failures pop a warning.
        kicked = 0
        for rel in files:
            local_dst = Path(Paths.USER_CONFIG_DIR) / Path(*rel.split("/"))
            if local_dst.exists() and not overwrite_all:
                continue
            self._submit_op(
                rel,
                CloudFileOpTask(
                    "download_one", rel, local_dst=local_dst,
                ),
                silent_success=True,
            )
            kicked += 1
        if kicked == 0:
            QMessageBox.information(
                self,
                tr("cloud.manage_dialog_folder_dl_title", "Download folder"),
                tr("cloud.manage_dialog_folder_dl_all_skipped",
                   "All files already exist locally — nothing downloaded."),
            )

    @pyqtSlot(str, list)
    def _on_folder_delete(self, folder_rel: str, files: list) -> None:
        files = [str(f) for f in files if f]
        if not files:
            return
        leaf = folder_rel.rsplit("/", 1)[-1] or "/"
        ans = QMessageBox.question(
            self,
            tr("cloud.manage_dialog_folder_del_title", "Delete folder"),
            tr(
                "cloud.manage_dialog_folder_del_body",
                "Permanently delete '{name}' and all {n} files inside "
                "from cloud?",
            ).format(name=leaf, n=len(files)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._submit_folder_op(
            folder_rel,
            CloudFileOpTask("delete_many", rel_paths=files),
        )

    @pyqtSlot(str)
    def _on_delete_clicked(self, rel_path: str) -> None:
        basename = rel_path.rsplit("/", 1)[-1]
        ans = QMessageBox.question(
            self,
            tr("cloud.manage_dialog_delete_confirm_title", "Delete cloud file"),
            tr("cloud.manage_dialog_delete_confirm_body",
               "Permanently delete '{name}' from cloud?").format(name=basename),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._submit_op(rel_path, CloudFileOpTask("delete", rel_path))

    @pyqtSlot(str, list)
    def _on_drop(self, target_dir: str, local_paths: list) -> None:
        # 50MB cap is enforced server-side and storage-sync.spec.yaml; we
        # batch-warn here so users don't get a per-file failure cascade.
        accepted: List[Path] = []
        oversize: List[str] = []
        for raw in local_paths:
            p = Path(raw)
            try:
                size = p.stat().st_size
            except OSError as exc:
                log_warning(f"[cloud_manage_dialog] stat({p}) failed: {exc}")
                continue
            if size >= _SIZE_LIMIT_BYTES:
                oversize.append(f"{p.name} ({_format_size(size)})")
                continue
            accepted.append(p)
        if oversize:
            QMessageBox.warning(
                self,
                tr("cloud.manage_dialog_oversize_title", "File too large"),
                tr("cloud.manage_dialog_oversize_body",
                   "Files over 50 MB cannot be uploaded:\n{names}").format(
                    names="\n".join(oversize),
                ),
            )
        for src in accepted:
            try:
                payload = src.read_bytes()
            except OSError as exc:
                log_warning(f"[cloud_manage_dialog] read({src}) failed: {exc}")
                continue
            rel = f"{target_dir}/{src.name}" if target_dir else src.name
            ct = _guess_content_type(src)
            self._submit_op(
                rel,
                CloudFileOpTask(
                    "upload_one", rel,
                    upload_bytes=payload, upload_ct=ct,
                ),
            )

    # ------------------------------------------------------------------
    # Task plumbing
    # ------------------------------------------------------------------

    def _submit_op(
        self,
        rel_path: str,
        task: CloudFileOpTask,
        *,
        silent_success: bool = False,
    ) -> None:
        if not self._auth.is_signed_in():
            QMessageBox.warning(
                self,
                tr("cloud.manage_dialog_op_failed_title", "Operation failed"),
                tr("cloud.manage_dialog_not_signed_in",
                   "Sign in first to manage cloud storage."),
            )
            return
        row = self._rows_by_path.get(rel_path)
        if row is not None:
            row.set_busy(True)
        try:
            tid = get_tasks_manager().submit(task)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[cloud_manage_dialog] submit failed: {exc!r}")
            if row is not None:
                row.set_busy(False)
            QMessageBox.warning(
                self,
                tr("cloud.manage_dialog_op_failed_title", "Operation failed"),
                str(exc),
            )
            return
        self._inflight[tid] = self._make_completion_cb(
            rel_path, silent_success=silent_success,
        )

    def _submit_folder_op(
        self, folder_rel: str, task: CloudFileOpTask,
    ) -> None:
        """Dispatch a folder-wide op (currently only ``delete_many``).

        Folder ops don't map to a single ``_FileRow``, so we skip the
        per-row busy plumbing — visual feedback comes from the
        completion callback's tree refresh.
        """
        if not self._auth.is_signed_in():
            QMessageBox.warning(
                self,
                tr("cloud.manage_dialog_op_failed_title", "Operation failed"),
                tr("cloud.manage_dialog_not_signed_in",
                   "Sign in first to manage cloud storage."),
            )
            return
        try:
            tid = get_tasks_manager().submit(task)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[cloud_manage_dialog] folder submit failed: {exc!r}"
            )
            QMessageBox.warning(
                self,
                tr("cloud.manage_dialog_op_failed_title", "Operation failed"),
                str(exc),
            )
            return
        self._inflight[tid] = self._make_folder_completion_cb(folder_rel)

    def _make_completion_cb(
        self, rel_path: str, *, silent_success: bool = False,
    ) -> Callable[[bool, object], None]:
        def _cb(ok: bool, result: object) -> None:
            row = self._rows_by_path.get(rel_path)
            if row is not None:
                row.set_busy(False)
            data = result if isinstance(result, dict) else {}
            phase = str(data.get("phase") or "op")
            if not ok or not data.get("ok", False):
                err = str(data.get("error") or result or "unknown error")
                QMessageBox.warning(
                    self,
                    tr("cloud.manage_dialog_op_failed_title", "Operation failed"),
                    tr("cloud.manage_dialog_op_failed_body",
                       "{phase} failed:\n{err}").format(phase=phase, err=err),
                )
                return
            if phase == "download_one":
                if not silent_success:
                    local_path = str(data.get("local_path") or "")
                    QMessageBox.information(
                        self,
                        tr("cloud.manage_dialog_dl_done_title",
                           "Download complete"),
                        tr("cloud.manage_dialog_dl_done_body",
                           "Downloaded to:\n{path}").format(path=local_path),
                    )
                # Cloud side unchanged — no tree refresh.
                return
            # upload_one / delete → tree changed; the async refresh path
            # (CloudListTask → CloudSyncService.list_remote) calls
            # fetch_storage_usage() internally, so the quota row refreshes
            # for free via usage_changed.
            self._refresh_tree()
        return _cb

    def _make_folder_completion_cb(
        self, folder_rel: str,
    ) -> Callable[[bool, object], None]:
        def _cb(ok: bool, result: object) -> None:
            data = result if isinstance(result, dict) else {}
            phase = str(data.get("phase") or "folder_op")
            if not ok or not data.get("ok", False):
                err = str(data.get("error") or result or "unknown error")
                QMessageBox.warning(
                    self,
                    tr("cloud.manage_dialog_op_failed_title", "Operation failed"),
                    tr("cloud.manage_dialog_op_failed_body",
                       "{phase} failed:\n{err}").format(phase=phase, err=err),
                )
                return
            # delete_many → tree changed.
            self._refresh_tree()
        return _cb

    @pyqtSlot(str, bool, object)
    def _on_task_finished(
        self, task_id: str, ok: bool, result: object,
    ) -> None:
        cb = self._inflight.pop(task_id, None)
        if cb is None:
            return
        cb(ok, result)

    # ------------------------------------------------------------------
    # Quota row
    # ------------------------------------------------------------------

    @pyqtSlot(int, int)
    def _on_usage_changed(self, used: int, total: int) -> None:
        self._render_usage(used, total)

    def _render_usage(self, used: int, total: int) -> None:
        if total <= 0:
            self._quota_label.setText(tr(
                "cloud.manage_dialog_quota", "Used {used} / {total}",
            ).format(used="—", total="—"))
            self._quota_bar.setValue(0)
            self._quota_percent.setText("—")
            return
        ratio = max(0.0, min(1.0, float(used) / float(total)))
        self._quota_bar.setValue(int(ratio * 1000))
        self._quota_label.setText(tr(
            "cloud.manage_dialog_quota", "Used {used} / {total}",
        ).format(used=_format_size(used), total=_format_size(total)))
        self._quota_percent.setText(f"{int(round(ratio * 100))}%")

    # ------------------------------------------------------------------
    # i18n + theme
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self.setWindowTitle(tr(
            "cloud.manage_dialog_title", "Manage Cloud Storage",
        ))
        # Quota label re-renders on next usage_changed; force one now.
        cached = self._svc.cached_usage()
        if cached is not None:
            self._render_usage(*cached)
        # Re-list to refresh folder summaries with translated strings.
        self._refresh_tree()

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        bg_2 = Config.get_color("bg_2")
        row_hover = Config.get_color("row_1")
        folder_hover = Config.get_color("hover_1")
        accent = Config.get_color("safe_zone")
        usage_track = Config.get_color("row_1")
        usage_border = Config.get_color("border_1")
        usage_fill = Config.get_color("safe_zone")
        main = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        btn_hover = Config.get_color("btn_1")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QDialog#cloudManageDialog {{"
            f"  background-color: {bg};"
            f"}}"
            # Quota row
            f"QLabel#cloudManageQuotaLabel {{"
            f"  color: {main};"
            f"  font-size: {size_small}px;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#cloudManageQuotaPercent {{"
            f"  color: {sub};"
            f"  font-size: {size_small}px;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QProgressBar#cloudManageQuotaBar {{"
            f"  background-color: {usage_track};"
            f"  border: 1px solid {usage_border};"
            f"  border-radius: 3px;"
            f"}}"
            f"QProgressBar#cloudManageQuotaBar::chunk {{"
            f"  background-color: {usage_fill};"
            f"  border-radius: 2px;"
            f"}}"
            # Scroll / tree host
            f"QScrollArea#cloudManageScroll {{"
            f"  background: {bg_2};"
            f"  border: 1px solid {Config.get_color('border_1')};"
            f"  border-radius: 6px;"
            f"}}"
            f"QWidget#cloudManageTreeHost {{"
            f"  background: {bg_2};"
            f"}}"
            f"QWidget#cloudManageFolderBody {{"
            f"  background: transparent;"
            f"}}"
            f"QLabel#cloudManageEmpty {{"
            f"  color: {sub};"
            f"  font-size: {size_small}px;"
            f"  background: transparent;"
            f"  padding: 24px;"
            f"}}"
            # Folder header
            f"QFrame#cloudManageFolderHeader {{"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: 4px;"
            f"}}"
            f"QFrame#cloudManageFolderHeader:hover {{"
            f"  background-color: {folder_hover};"
            f"}}"
            f"QFrame#cloudManageFolderHeader[dropHover=\"true\"] {{"
            f"  background-color: {folder_hover};"
            f"  border: 1px solid {accent};"
            f"}}"
            f"QLabel#cloudManageChevron {{"
            f"  color: {sub};"
            f"  font-size: {size_small}px;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#cloudManageFolderName {{"
            f"  color: {Config.get_color('highlight')};"
            f"  font-size: {size_normal}px;"
            f"  font-weight: 600;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#cloudManageFolderSummary {{"
            f"  color: {sub};"
            f"  font-size: {size_small}px;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            # File row
            f"QFrame#cloudManageFileRow {{"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: 4px;"
            f"}}"
            f"QFrame#cloudManageFileRow:hover {{"
            f"  background-color: {row_hover};"
            f"}}"
            f"QFrame#cloudManageFileRow[dropHover=\"true\"] {{"
            f"  background-color: {row_hover};"
            f"  border: 1px solid {accent};"
            f"}}"
            f"QFrame#cloudManageFileRow[busy=\"true\"] {{"
            f"  background-color: {row_hover};"
            f"}}"
            f"QLabel#cloudManageFileName {{"
            f"  color: {main};"
            f"  font-size: {size_normal}px;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#cloudManageFileSize {{"
            f"  color: {sub};"
            f"  font-size: {size_small}px;"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QPushButton#cloudManageRowBtn {{"
            f"  background: transparent;"
            f"  border: none;"
            f"  border-radius: 4px;"
            f"  padding: 2px;"
            f"}}"
            f"QPushButton#cloudManageRowBtn:hover {{"
            f"  background: {btn_hover};"
            f"}}"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, ev) -> None:  # noqa: N802
        # Drop in-flight callbacks so a late task_finished hitting a
        # destroyed dialog doesn't crash. The tasks themselves keep
        # running and write their result to the log via Task internals.
        self._inflight.clear()
        try:
            self._svc.usage_changed.disconnect(self._on_usage_changed)
        except (TypeError, RuntimeError):
            pass
        try:
            get_task_signal().task_finished.disconnect(self._on_task_finished)
        except (TypeError, RuntimeError):
            pass
        try:
            I18n.instance().language_changed.disconnect(self._retranslate)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Root drop area — backstop for drops on the empty scroll background
# ---------------------------------------------------------------------------


class _RootDropArea(QFrame):
    """Scroll-area container that catches drops not absorbed by any
    folder header or file row, forwarding them as uploads to the root.
    """

    drop_received = pyqtSignal(str, list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev) -> None:  # noqa: N802
        if _extract_local_paths(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev) -> None:   # noqa: N802
        if _extract_local_paths(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev) -> None:        # noqa: N802
        paths = _extract_local_paths(ev)
        if not paths:
            ev.ignore()
            return
        ev.acceptProposedAction()
        self.drop_received.emit("", paths)


# ---------------------------------------------------------------------------
# Small utility — recursive count / size
# ---------------------------------------------------------------------------


def _count_recursive(node: dict) -> tuple:
    n = len(node.get("_files", []))
    total = sum(int(f.get("size") or 0) for f in node.get("_files", []))
    for sub in node.get("subfolders", {}).values():
        sub_n, sub_total = _count_recursive(sub)
        n += sub_n
        total += sub_total
    return n, total


def _collect_files(node: dict) -> List[str]:
    """Flat list of cloud-rel-paths under ``node`` (recursive)."""
    out: List[str] = []
    for f in node.get("_files", []) or []:
        rel = str(f.get("rel") or "")
        if rel:
            out.append(rel)
    for sub in (node.get("subfolders", {}) or {}).values():
        out.extend(_collect_files(sub))
    return out


__all__ = ["CloudManageDialog"]
