"""FilesRow: persistent file tab bar above the main workspace.

Renders one row of tabs representing project-workspace files of the current
workspace kind (mission or training). Each tab shows [filename, dirty dot,
close X]. A leading '+' button creates a new file, followed by a vertical
separator and a 2 px gap before the tab strip.

The row is a single persistent instance — its kind switches between
"mission" and "training" as the user navigates, and `set_files` refreshes
the tabs for the active workspace type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.system.core.theme_manager import get_color


@dataclass
class FileTabEntry:
    """Data for one tab in the FilesRow.

    ``is_pinned`` hides the close button (used for the pending ``new_mission``
    draft which should only vanish via save or explicit discard).

    ``backend_tag`` is a short prefix label rendered before the title with a
    coloured pill (used by training tabs to surface SB3 vs Isaac Lab).

    ``robot_model`` and ``saved_date`` are optional metadata surfaced by the
    ProjectFileBrowserPanel sidebar item (e.g. "Go2 - 2026-04-13").
    """
    file_id: str
    title: str
    is_dirty: bool = False
    is_pinned: bool = False
    backend_tag: str = ""  # "" / "SB3" / "Isaac"
    robot_model: str = ""  # e.g. "Go2"
    saved_date: str = ""   # e.g. "2026-04-13"


class _FileTabWidget(QWidget):
    """Single tab: [marker] [title] [dirty dot] [close ×], self-painted bg.

    Tab height equals the parent FilesRow height so checked/unchecked tabs
    sit flush with the row (no top notch / vertical seam). The leading
    *marker* block (training tabs only) is a square coloured pill carrying
    the backend tag text (e.g. "IL", "SB3").
    """

    clicked = Signal(str)
    close_requested = Signal(str)

    _DIRTY_DIAMETER = 7
    _CLOSE_EXTENT = 12
    # Match FilesRow._HEIGHT — kept in sync intentionally so the tab
    # body fills the row with no vertical gap above or below.
    _TAB_HEIGHT = 48
    # Square edge for the leading backend marker block.
    _MARKER_EXTENT = 32
    # Left inset between the tab's left edge and its first content cell
    # (whether the backend marker block or the title label).
    _CONTENT_LEFT_INSET = 5

    def __init__(self, entry: FileTabEntry, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._active = False
        self._hovered = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(self._TAB_HEIGHT)

        layout = QHBoxLayout(self)
        # 5 px between the tab's left edge and its first content cell.
        layout.setContentsMargins(self._CONTENT_LEFT_INSET, 0, 8, 0)
        layout.setSpacing(8)

        # Backend tag marker — square block (training tabs only). Empty
        # backend_tag (mission tabs) just leaves the contentsMargins inset
        # in place so the title text doesn't kiss the tab's left edge.
        self._tag_label = QLabel(self._entry.backend_tag or "", self)
        self._tag_label.setObjectName("fileTabBackendTag")
        self._tag_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._tag_label.setFixedSize(self._MARKER_EXTENT, self._MARKER_EXTENT)
        self._tag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tag_label.setVisible(bool(self._entry.backend_tag))
        layout.addWidget(self._tag_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._label = QLabel(self._entry.title, self)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Bump the title font 3 pt above the application default for legibility
        # in the now-taller tab. Use pointSize when available, else pixelSize.
        _font = self._label.font()
        if _font.pointSize() > 0:
            _font.setPointSize(_font.pointSize() + 3)
        else:
            _font.setPixelSize(max(8, _font.pixelSize()) + 3)
        self._label.setFont(_font)
        layout.addWidget(self._label, 0, Qt.AlignmentFlag.AlignVCenter)

        # Reserved slot for the dirty dot so the layout does not jump when
        # dirtiness toggles. Painting happens in paintEvent over this slot.
        self._dot_slot = QWidget(self)
        self._dot_slot.setFixedSize(self._DIRTY_DIAMETER + 2, self._DIRTY_DIAMETER + 2)
        self._dot_slot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._dot_slot, 0, Qt.AlignmentFlag.AlignVCenter)

        self._close_btn = QPushButton("×", self)
        self._close_btn.setObjectName("fileTabClose")
        self._close_btn.setFixedSize(20,20)
        self._close_btn.setFixedSize(self._CLOSE_EXTENT, self._CLOSE_EXTENT)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setFlat(True)
        self._close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._close_btn.clicked.connect(
            lambda: self.close_requested.emit(self._entry.file_id)
        )
        self._close_btn.setVisible(not self._entry.is_pinned)
        layout.addWidget(self._close_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    # ── accessors ────────────────────────────────────────────────────

    def file_id(self) -> str:
        return self._entry.file_id

    def set_active(self, active: bool) -> None:
        if self._active != active:
            self._active = active
            self.apply_theme()
            self.update()

    def set_dirty(self, dirty: bool) -> None:
        if self._entry.is_dirty != dirty:
            self._entry.is_dirty = dirty
            self.update()

    def set_title(self, title: str) -> None:
        self._entry.title = title
        self._label.setText(title)

    # ── painting ─────────────────────────────────────────────────────

    def paintEvent(self, event):  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Full-bleed body — tab height equals row height, so the fill goes
        # edge-to-edge with no rounded notch and no vertical seam against
        # the row background.
        body = self.rect()

        if self._active:
            bg = QColor(get_color("card_bg", "#272829"))
        elif self._hovered:
            bg = QColor(get_color("tab_bg_hover", get_color("hover_bg", "#3d3d3d")))
        else:
            # Match the row background so inactive tabs blend seamlessly.
            bg = QColor(get_color("main_row_bg", "#1f2937"))

        painter.fillRect(body, bg)

        # Dirty dot centred in the reserved slot.
        if self._entry.is_dirty:
            dot_color = QColor(get_color("accent", "#3b82f6"))
            slot = self._dot_slot.geometry()
            cx = slot.center().x()
            cy = slot.center().y()
            d = self._DIRTY_DIAMETER
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(dot_color)
            painter.drawEllipse(cx - d // 2, cy - d // 2, d, d)

    def enterEvent(self, event):  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            # Only fire when the press did not land on the close button.
            if not self._close_btn.geometry().contains(event.position().toPoint()):
                self.clicked.emit(self._entry.file_id)
        super().mousePressEvent(event)

    # ── theme ────────────────────────────────────────────────────────

    def apply_theme(self) -> None:
        text_col = (
            get_color("text_primary", "#e2e8f0")
            if self._active
            else get_color("text_secondary", "#9ca3af")
        )
        self._label.setStyleSheet(
            f"color: {text_col}; background: transparent; border: none;"
        )
        # Backend tag marker — only training tabs set a non-empty backend_tag.
        # Square block carrying short tag text ("IL", "SB3", "Isaac").
        # SB3 / Isaac use distinct accent colors so the backend is glanceable.
        tag = (self._entry.backend_tag or "").strip()
        if tag:
            tag_lower = tag.lower()
            if "isaac" in tag_lower or tag_lower == "il":
                tag_bg = get_color("tab_bg_training", "#62E3E2")
            else:
                tag_bg = get_color("tab_bg_checked", "#FF7844")
            tag_text = get_color("tab_text_checked", "#000000")
            self._tag_label.setStyleSheet(
                f"QLabel#fileTabBackendTag {{"
                f" color: {tag_text};"
                f" background-color: {tag_bg};"
                f" border: none;"
                f" border-radius: 0px;"
                f" padding: 0px;"
                f" font-size: 11px;"
                f" font-weight: 800;"
                f" qproperty-alignment: AlignCenter;"
                f" }}"
            )
            self._tag_label.setText(tag)
            self._tag_label.setVisible(True)
        else:
            self._tag_label.setVisible(False)
        close_col = get_color("text_secondary", "#9ca3af")
        close_hover_text = get_color("text_primary", "#e2e8f0")
        close_hover_bg = get_color("hover_bg", "#3d3d3d")
        # 12 × 12 square button with 2 px rounded corners. Font size tuned
        # so the × glyph reads as solid against the hover background fill.
        self._close_btn.setStyleSheet(f"""
            QPushButton#fileTabClose {{
                color: {close_col};
                background: transparent;
                border: none;
                border-radius: 2px;
                font-size: 13px;
                font-weight: bold;
                padding: 0px 0px 1px 0px;
            }}
            QPushButton#fileTabClose:hover {{
                background: {close_hover_bg};
                color: {close_hover_text};
            }}
        """)
        self.update()


class FilesRow(QWidget):
    """Persistent file-tab row above the main workspace.

    - Fixed height 48 px (same as MainRow).
    - Layout: [+ button] [8 px] [| separator] [2 px] [tab strip] [stretch].
    - `kind` is "mission" or "training"; background swaps accent accordingly.
    """

    _OBJECT_NAME = "filesRow"
    _HEIGHT = 48
    # '+' button leaves a 2 px gap on every side of the row → square edge
    # equals row height minus 2× margin.
    _NEW_BTN_MARGIN = 2
    _NEW_BTN_SIZE = _HEIGHT - 2 * _NEW_BTN_MARGIN

    new_requested = Signal()
    tab_activated = Signal(str)
    tab_close_requested = Signal(str)
    tab_restore_requested = Signal(str)

    def __init__(self, theme: str = "dark", kind: str = "mission", parent=None):
        super().__init__(parent)
        self._theme = (theme or "dark").lower()
        self._kind = (kind or "mission").lower()
        self._tabs: List[_FileTabWidget] = []
        self._entries: List[FileTabEntry] = []
        # Hidden file_ids (per kind) survive set_files when ids persist.
        self._hidden_by_kind: Dict[str, Set[str]] = {}
        self._active_id: str = ""

        self.setObjectName(self._OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(self._HEIGHT)

        outer = QHBoxLayout(self)
        # 2 px gap on all four edges around the '+' button (and the same
        # 2 px right-edge gap mirroring the left).
        outer.setContentsMargins(self._NEW_BTN_MARGIN, 0, self._NEW_BTN_MARGIN, 0)
        outer.setSpacing(0)

        # '+' new-file button (leftmost, inside the row container).
        self._new_btn = QPushButton("+", self)
        self._new_btn.setObjectName("filesRowNewBtn")
        self._new_btn.setFixedSize(self._NEW_BTN_SIZE, self._NEW_BTN_SIZE)
        self._new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._new_btn.setToolTip("New file")
        self._new_btn.clicked.connect(self._on_new_btn_clicked)
        outer.addWidget(self._new_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        
        outer.addSpacing(3)

        # Tab strip host. Zero spacing so adjacent tabs sit flush — the
        # active/inactive bg contrast handles visual separation.
        self._tab_host = QWidget(self)
        self._tab_layout = QHBoxLayout(self._tab_host)
        self._tab_layout.setContentsMargins(0, 0, 0, 0)
        self._tab_layout.setSpacing(0)
        self._tab_layout.addStretch(1)
        outer.addWidget(self._tab_host, 1)

        self.apply_theme()

    # ── public API ───────────────────────────────────────────────────

    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str) -> None:
        kind = (kind or "mission").lower()
        if kind == self._kind:
            return
        self._kind = kind
        self.apply_theme()

    def set_files(
        self,
        entries: List[FileTabEntry],
        kind: Optional[str] = None,
        active_id: Optional[str] = None,
    ) -> None:
        """Replace the full file list. Hidden state is preserved for file
        ids that survive between calls (so a hidden file stays hidden across
        refreshes). If *active_id* is provided it becomes the new active
        tab; otherwise the previous active_id is retained."""
        if kind is not None:
            self.set_kind(kind)
        if active_id is not None:
            self._active_id = str(active_id or "")

        self._entries = list(entries)

        # Trim hidden set to surviving ids only.
        kind_key = self._kind
        surviving = {e.file_id for e in self._entries}
        self._hidden_by_kind[kind_key] = {
            fid for fid in self._hidden_by_kind.get(kind_key, set())
            if fid in surviving
        }

        self._rerender()

    def _rerender(self) -> None:
        """Rebuild tab widgets from ``self._entries``, skipping hidden ones."""
        for tab in self._tabs:
            self._tab_layout.removeWidget(tab)
            tab.setParent(None)
            tab.deleteLater()
        self._tabs = []

        hidden = self._hidden_by_kind.get(self._kind, set())
        for entry in self._entries:
            if entry.file_id in hidden:
                continue
            tab = _FileTabWidget(entry, self._tab_host)
            tab.clicked.connect(self.tab_activated.emit)
            tab.close_requested.connect(self._on_tab_close_clicked)
            tab.set_active(entry.file_id == self._active_id)
            tab.apply_theme()
            self._tab_layout.insertWidget(self._tab_layout.count() - 1, tab)
            self._tabs.append(tab)

    def set_active(self, file_id: str) -> None:
        self._active_id = str(file_id or "")
        for tab in self._tabs:
            tab.set_active(tab.file_id() == self._active_id)

    def active_id(self) -> str:
        return self._active_id

    def mark_dirty(self, file_id: str, dirty: bool = True) -> None:
        # NB: do NOT pre-mutate entry.is_dirty before calling tab.set_dirty.
        # The tab holds a reference to the same FileTabEntry object stored
        # in self._entries, so set_dirty's "old != new" guard would return
        # False after a pre-mutation and the visual repaint would be
        # skipped — leaving a stale dot on the tab even after a save.
        for tab in self._tabs:
            if tab.file_id() == file_id:
                tab.set_dirty(dirty)
                return
        # Tab not currently rendered (e.g. file is hidden) — just update
        # the underlying entry so the next set_files / restore reflects it.
        for entry in self._entries:
            if entry.file_id == file_id:
                entry.is_dirty = dirty
                return

    def clear(self) -> None:
        self.set_files([], active_id="")

    # ── hidden-files management ──────────────────────────────────────

    def hide_file(self, file_id: str) -> None:
        """Hide a tab from the strip (keeps it in the underlying list)."""
        file_id = str(file_id or "")
        if not file_id:
            return
        bucket = self._hidden_by_kind.setdefault(self._kind, set())
        if file_id in bucket:
            return
        bucket.add(file_id)
        self._rerender()

    def unhide_file(self, file_id: str) -> None:
        bucket = self._hidden_by_kind.get(self._kind)
        if not bucket or file_id not in bucket:
            return
        bucket.discard(file_id)
        self._rerender()

    def hidden_entries(self) -> List[FileTabEntry]:
        hidden = self._hidden_by_kind.get(self._kind, set())
        return [e for e in self._entries if e.file_id in hidden]

    def _on_tab_close_clicked(self, file_id: str) -> None:
        """A tab's × was clicked. Hide it locally first, then notify the
        host so it can layer additional policy (dirty-check, delete…)."""
        self.hide_file(file_id)
        self.tab_close_requested.emit(file_id)

    # ── '+' button with hidden-files dropdown ────────────────────────

    def _on_new_btn_clicked(self) -> None:
        hidden_entries = self.hidden_entries()
        if not hidden_entries:
            self.new_requested.emit()
            return

        menu = QMenu(self)
        menu.setObjectName("filesRowNewMenu")
        self._style_menu(menu)

        new_action = QAction("<New>", menu)
        new_action.triggered.connect(self.new_requested.emit)
        menu.addAction(new_action)
        menu.addSeparator()

        for entry in hidden_entries:
            label = entry.title or entry.file_id
            act = QAction(label, menu)
            act.setData(entry.file_id)
            act.triggered.connect(
                lambda _checked=False, fid=entry.file_id: self._restore_hidden(fid)
            )
            menu.addAction(act)

        # Pop menu directly below the '+' button.
        btn_bottom_left = self._new_btn.mapToGlobal(
            self._new_btn.rect().bottomLeft()
        )
        menu.exec(btn_bottom_left)

    def _restore_hidden(self, file_id: str) -> None:
        self.unhide_file(file_id)
        self.tab_restore_requested.emit(file_id)

    def _style_menu(self, menu: QMenu) -> None:
        bg = get_color("button_bg", "#2a2a2a")
        hover = get_color("hover_bg", "#3d3d3d")
        text = get_color("text_primary", "#e2e8f0")
        border = get_color("border", "#475569")
        menu.setStyleSheet(f"""
            QMenu#filesRowNewMenu {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                padding: 4px 0px;
            }}
            QMenu#filesRowNewMenu::item {{
                padding: 6px 20px;
            }}
            QMenu#filesRowNewMenu::item:selected {{
                background-color: {hover};
            }}
            QMenu#filesRowNewMenu::separator {{
                height: 1px;
                background: {border};
                margin: 4px 8px;
            }}
        """)

    # ── theme ────────────────────────────────────────────────────────

    def apply_theme(self) -> None:
        if self._kind == "training":
            bg = get_color(
                "main_row_training_bg",
                get_color("main_row_bg", "#1f2937"),
            )
        else:
            bg = get_color("main_row_bg", "#1f2937")
        border = get_color("border", "#475569")
        # No bottom border — files row sits flush against the toolbar row
        # below it, regardless of mission/training kind.
        self.setStyleSheet(
            f"#{self._OBJECT_NAME} {{"
            f" background-color: {bg};"
            f" border: none;"
            f" }}"
        )

        # Hover color matches the file tab hover (tab_bg_hover with hover_bg
        # fallback) so the '+' affordance reads as part of the tab strip.
        hover = get_color("tab_bg_hover", get_color("hover_bg", "#3d3d3d"))
        text = get_color("text_primary", "#e2e8f0")
        # 2 px corner radius, no border, no fill — only a hover background.
        # Glyph runs 1 px above the previous default for stronger presence
        # in the now-larger square.
        self._new_btn.setStyleSheet(f"""
            QPushButton#filesRowNewBtn {{
                background-color: transparent;
                color: {text};
                border: none;
                border-radius: 2px;
                font-size: 19px;
                font-weight: bold;
                padding: 0px 0px 2px 0px;
            }}
            QPushButton#filesRowNewBtn:hover {{
                background-color: {hover};
            }}
        """)
        for tab in self._tabs:
            tab.apply_theme()


# ===========================================================================
# ProjectFileBrowserPanel — sidebar panel for browsing project canvas files
# ===========================================================================

class _CanvasFileItem(QWidget):
    """Single row in the project file browser.

    Layout: [checkbox] [text block]
    Text block line 1: ``[TAG] title``  (tag only for training items)
    Text block line 2: ``robot_model - saved_date``

    Click anywhere on the item toggles the checkbox.
    """

    toggled = Signal(str, bool)  # file_id, checked

    def __init__(self, entry: FileTabEntry, checked: bool = True, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._checked = checked
        self._hovered = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        # Checkbox indicator
        self._checkbox = QCheckBox(self)
        self._checkbox.setChecked(checked)
        self._checkbox.setFixedSize(16, 16)
        self._checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._checkbox.toggled.connect(self._on_checkbox_toggled)
        layout.addWidget(self._checkbox, 0, Qt.AlignmentFlag.AlignVCenter)

        # Text block
        text_widget = QWidget(self)
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        # Line 1: [TAG] title
        tag = self._entry.backend_tag
        line1_text = f"[{tag}] {self._entry.title}" if tag else self._entry.title
        self._line1 = QLabel(line1_text, text_widget)
        self._line1.setObjectName("canvasFileItemTitle")
        self._line1.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_layout.addWidget(self._line1)

        # Line 2: robot_model - saved_date
        meta_parts = []
        if self._entry.robot_model:
            meta_parts.append(self._entry.robot_model)
        if self._entry.saved_date:
            meta_parts.append(self._entry.saved_date)
        line2_text = " - ".join(meta_parts) if meta_parts else ""
        self._line2 = QLabel(line2_text, text_widget)
        self._line2.setObjectName("canvasFileItemMeta")
        self._line2.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if not line2_text:
            self._line2.setVisible(False)
        text_layout.addWidget(self._line2)

        layout.addWidget(text_widget, 1)

        self.apply_theme()

    def file_id(self) -> str:
        return self._entry.file_id

    def is_checked(self) -> bool:
        return self._checked

    def set_checked(self, checked: bool) -> None:
        if self._checked == checked:
            return
        self._checked = checked
        self._checkbox.blockSignals(True)
        self._checkbox.setChecked(checked)
        self._checkbox.blockSignals(False)

    def _on_checkbox_toggled(self, checked: bool) -> None:
        self._checked = checked
        self.toggled.emit(self._entry.file_id, checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Don't double-fire if click landed on the checkbox itself.
            if not self._checkbox.geometry().contains(event.position().toPoint()):
                self._checkbox.toggle()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self._hovered = True
        self._apply_hover_style()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_hover_style()
        super().leaveEvent(event)

    def _apply_hover_style(self) -> None:
        if self._hovered:
            bg = get_color("hover_bg", "#3d3d3d")
        else:
            bg = "transparent"
        self.setStyleSheet(f"_CanvasFileItem {{ background-color: {bg}; }}")

    def apply_theme(self) -> None:
        title_color = get_color("text_primary", "#e2e8f0")
        meta_color = get_color("text_secondary", "#9ca3af")
        self._line1.setStyleSheet(
            f"QLabel#canvasFileItemTitle {{ color: {title_color}; font-size: 13px; font-weight: 600; background: transparent; border: none; }}"
        )
        self._line2.setStyleSheet(
            f"QLabel#canvasFileItemMeta {{ color: {meta_color}; font-size: 11px; background: transparent; border: none; }}"
        )
        accent = get_color("accent", "#3b82f6")
        border = get_color("border", "#475569")
        self._checkbox.setStyleSheet(f"""
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {border};
                border-radius: 3px;
                background: transparent;
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
                border-color: {accent};
            }}
        """)
        self._apply_hover_style()


class ProjectFileBrowserPanel(QWidget):
    """Sidebar panel listing project canvas files with check-to-show behaviour.

    Checked items correspond to visible FilesRow tabs; unchecking hides the
    tab, checking restores it. The panel observes the current shell mode
    (mission / training) and filters items accordingly.

    Signals
    -------
    file_checked(file_id, checked)
        Emitted when the user toggles a canvas item.
    """

    file_checked = Signal(str, bool)  # file_id, is_now_checked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("projectFileBrowserPanel")
        self._entries: List[FileTabEntry] = []
        self._items: List[_CanvasFileItem] = []
        self._hidden_ids: set = set()
        self._project_name: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Project name header button ────────────────────────────────
        self._project_btn = QPushButton("No Project", self)
        self._project_btn.setObjectName("projectFileBrowserHeader")
        self._project_btn.setFlat(True)
        self._project_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._project_btn.setFixedHeight(36)
        root.addWidget(self._project_btn)

        # ── 1 px divider ─────────────────────────────────────────────
        self._divider = QFrame(self)
        self._divider.setFrameShape(QFrame.Shape.HLine)
        self._divider.setFixedHeight(1)
        root.addWidget(self._divider)

        # ── Scrollable item list ──────────────────────────────────────
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch(1)

        self._scroll.setWidget(self._list_host)
        root.addWidget(self._scroll, 1)

        self.apply_theme()

    # ── Public API ────────────────────────────────────────────────────

    def set_project_name(self, name: str) -> None:
        self._project_name = name or ""
        self._project_btn.setText(self._project_name or "No Project")

    def set_entries(
        self,
        entries: List[FileTabEntry],
        hidden_ids: Optional[set] = None,
    ) -> None:
        """Replace the full file list. *hidden_ids* are the file_ids that
        should appear unchecked (i.e. not shown in FilesRow)."""
        self._entries = list(entries)
        self._hidden_ids = set(hidden_ids or set())
        self._rebuild_items()

    def sync_hidden(self, hidden_ids: set) -> None:
        """Update check states from an external hidden-set change without
        rebuilding the full item list."""
        self._hidden_ids = set(hidden_ids or set())
        for item in self._items:
            item.set_checked(item.file_id() not in self._hidden_ids)

    # ── Internal ──────────────────────────────────────────────────────

    def _rebuild_items(self) -> None:
        for item in self._items:
            self._list_layout.removeWidget(item)
            item.setParent(None)
            item.deleteLater()
        self._items = []

        for entry in self._entries:
            checked = entry.file_id not in self._hidden_ids
            item = _CanvasFileItem(entry, checked=checked, parent=self._list_host)
            item.toggled.connect(self._on_item_toggled)
            item.apply_theme()
            self._list_layout.insertWidget(self._list_layout.count() - 1, item)
            self._items.append(item)

    def _on_item_toggled(self, file_id: str, checked: bool) -> None:
        if checked:
            self._hidden_ids.discard(file_id)
        else:
            self._hidden_ids.add(file_id)
        self.file_checked.emit(file_id, checked)

    # ── Theme ─────────────────────────────────────────────────────────

    def apply_theme(self) -> None:
        text = get_color("text_primary", "#e2e8f0")
        border = get_color("border", "#475569")
        hover = get_color("hover_bg", "#3d3d3d")
        self._project_btn.setStyleSheet(f"""
            QPushButton#projectFileBrowserHeader {{
                color: {text};
                background: transparent;
                border: none;
                font-weight: bold;
                font-size: 14px;
                padding: 6px 12px;
                text-align: left;
            }}
            QPushButton#projectFileBrowserHeader:hover {{
                background-color: {hover};
            }}
        """)
        self._divider.setStyleSheet(f"background-color: {border}; border: none;")
        scroll_bg = get_color("card_bg", "#2E2E2E")
        self._scroll.setStyleSheet(f"QScrollArea {{ background: {scroll_bg}; border: none; }}")
        self._list_host.setStyleSheet(f"background: {scroll_bg};")
        for item in self._items:
            item.apply_theme()
