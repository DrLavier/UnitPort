"""Lookbook UI widgets for browsing MuJoCo Menagerie packages (PyQt6 port).

Used by:
- :class:`MenagerieSelectPage` (wizard Page 1)
- :class:`MenagerieBrowserDialog` (Robot Asset sidebar -- not built this round)

Layout: 4-column grid of 300x100 cards. Each card:
    [checkbox] [80x80 preview icon] [title / subtitle]

Status encoding:
    - installed -> green-tinted background, checkbox locked-checked
    - registered -> blue bold title (matches a UnitPort robot)
    - checked -> accent border + highlight (user-selected for download)
    - hover -> soft border + background tint

Bit-exact port of DEMO ``bin/pages/layout/menagerie_card.py`` modulo
PySide6 -> PyQt6 (Signal -> pyqtSignal, Qt enum dotted form, etc.) and
theme reads via ``Config.get_color`` rather than DEMO's
``theme_manager.get_color``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config

from application.service.models import menagerie_manager as mm


CARD_W = 300
CARD_H = 100
ICON_W = 80
ICON_H = 80
GRID_COLUMNS = 4
GRID_HSPACE = 12
GRID_VSPACE = 12
GRID_MARGIN = 14


def _stylesheet() -> str:
    bg_card = Config.get_color("card_bg", "#23262b")
    bg_hover = Config.get_color("hover_bg", "#2d3036")
    bg_main = Config.get_color("bg_2", "#1a1c20")
    text_main = Config.get_color("main_t2", "#ffffff")
    text_muted = Config.get_color("sub_t2", "#9ca3af")
    border = Config.get_color("border_1", "#3a3a3a")
    accent = Config.get_color("highlight", "#FF7844")
    accent_blue = "#3b82f6"
    installed_bg = "#1f3a2e"
    installed_border = "#2ea36b"
    checked_bg = "#332620"

    return f"""
        QScrollArea#menagerieGrid {{
            background: {bg_main};
            border: 0;
        }}
        QWidget#menagerieGridContent {{
            background: {bg_main};
        }}

        QFrame#menagerieCard {{
            background: {bg_card};
            border: 1px solid {border};
            border-radius: 10px;
        }}
        QFrame#menagerieCard:hover {{
            background: {bg_hover};
            border-color: {accent};
        }}
        QFrame#menagerieCard[checked="true"] {{
            background: {checked_bg};
            border: 2px solid {accent};
        }}
        QFrame#menagerieCard[installed="true"] {{
            background: {installed_bg};
            border: 1px solid {installed_border};
        }}
        QFrame#menagerieCard[installed="true"][checked="true"] {{
            background: {installed_bg};
            border: 2px solid {installed_border};
        }}

        QLabel#menagerieCardIcon {{
            background: transparent;
            border: 1px solid {border};
            border-radius: 6px;
        }}

        QLabel#menagerieCardTitle {{
            color: {text_main};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
            border: 0;
        }}
        QLabel#menagerieCardTitle[registered="true"] {{
            color: {accent_blue};
        }}
        QLabel#menagerieCardSubtitle {{
            color: {text_muted};
            font-size: 11px;
            background: transparent;
            border: 0;
        }}

        QCheckBox#menagerieCardCheckbox {{
            background: transparent;
            spacing: 0;
        }}
        QCheckBox#menagerieCardCheckbox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {border};
            border-radius: 4px;
            background: {bg_main};
        }}
        QCheckBox#menagerieCardCheckbox::indicator:hover {{
            border-color: {accent};
        }}
        QCheckBox#menagerieCardCheckbox::indicator:checked {{
            background: {accent};
            border-color: {accent};
            image: none;
        }}
        QCheckBox#menagerieCardCheckbox::indicator:disabled {{
            background: {installed_border};
            border-color: {installed_border};
        }}
    """


class MenagerieCard(QFrame):
    """One 300x100 lookbook card for a single menagerie package."""

    toggled_changed = pyqtSignal(str, bool)   # name, checked

    def __init__(
        self,
        name: str,
        *,
        registered: bool,
        installed: bool,
        description: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("menagerieCard")
        self.setFixedSize(CARD_W, CARD_H)
        self.setProperty("checked", False)
        self.setProperty("installed", installed)
        self.setProperty("registered", registered)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._name = name
        self._installed = installed
        self._registered = registered

        if not installed:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)

        self._cb = QCheckBox(self)
        self._cb.setObjectName("menagerieCardCheckbox")
        self._cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if installed:
            self._cb.setChecked(True)
            self._cb.setEnabled(False)
            self._set_checked_property(True)
        self._cb.toggled.connect(self._on_cb_toggled)
        outer.addWidget(self._cb, 0, Qt.AlignmentFlag.AlignVCenter)

        self._icon_label = QLabel(self)
        self._icon_label.setObjectName("menagerieCardIcon")
        self._icon_label.setFixedSize(ICON_W, ICON_H)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setScaledContents(False)
        outer.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignVCenter)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(2)
        text_box.addStretch(1)

        self._title = QLabel(description or name, self)
        self._title.setObjectName("menagerieCardTitle")
        self._title.setProperty("registered", registered)
        self._title.setWordWrap(True)
        text_box.addWidget(self._title)

        self._subtitle = QLabel(self._build_subtitle(), self)
        self._subtitle.setObjectName("menagerieCardSubtitle")
        self._subtitle.setWordWrap(False)
        text_box.addWidget(self._subtitle)
        text_box.addStretch(1)

        outer.addLayout(text_box, 1)

        tt: List[str] = []
        if installed:
            tt.append("Already installed locally — locked.")
        if registered:
            tt.append("Matches a UnitPort-registered robot.")
        if tt:
            self.setToolTip(" ".join(tt))

    # ---- public ------------------------------------------------------
    def package_name(self) -> str:
        return self._name

    def is_checked(self) -> bool:
        return self._cb.isChecked()

    def is_installed(self) -> bool:
        return self._installed

    def set_checked(self, state: bool) -> None:
        if self._installed:
            return
        self._cb.setChecked(bool(state))

    def set_icon_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap is None or pixmap.isNull():
            return
        scaled = pixmap.scaled(
            ICON_W, ICON_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._icon_label.setPixmap(scaled)

    def has_icon(self) -> bool:
        pix = self._icon_label.pixmap()
        return pix is not None and not pix.isNull()

    # ---- internal ----------------------------------------------------
    def _build_subtitle(self) -> str:
        bits: List[str] = [self._name]
        if self._installed:
            bits.append("installed")
        if self._registered:
            bits.append("registered")
        return "  ·  ".join(bits)

    def _set_checked_property(self, state: bool) -> None:
        self.setProperty("checked", bool(state))
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_cb_toggled(self, state: bool) -> None:
        self._set_checked_property(state)
        self.toggled_changed.emit(self._name, bool(state))

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if (
            ev.button() == Qt.MouseButton.LeftButton
            and not self._installed
            and not self._cb.geometry().contains(ev.position().toPoint())
        ):
            self._cb.toggle()
            ev.accept()
            return
        super().mousePressEvent(ev)


class MenagerieCardGrid(QScrollArea):
    """Scrollable 4-column grid of :class:`MenagerieCard`."""

    selection_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("menagerieGrid")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._content = QWidget()
        self._content.setObjectName("menagerieGridContent")
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(GRID_MARGIN, GRID_MARGIN, GRID_MARGIN, GRID_MARGIN)
        self._grid.setHorizontalSpacing(GRID_HSPACE)
        self._grid.setVerticalSpacing(GRID_VSPACE)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setWidget(self._content)

        self._cards: Dict[str, MenagerieCard] = {}
        self._filter_text = ""

        self.setStyleSheet(_stylesheet())

    # ---- populate ----------------------------------------------------
    def populate(
        self,
        names: List[str],
        *,
        installed: Set[str],
        registered: Set[str],
        pre_checked: Optional[Set[str]] = None,
    ) -> None:
        for card in self._cards.values():
            self._grid.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        pre_checked = pre_checked or set()

        for name in names:
            description = name.replace("_", " ").title()
            is_installed = name in installed
            card = MenagerieCard(
                name,
                registered=name in registered,
                installed=is_installed,
                description=description,
                parent=self._content,
            )
            if not is_installed and name in pre_checked:
                card.set_checked(True)
            card.toggled_changed.connect(self._on_card_toggled)
            self._apply_synchronous_icon(card)
            self._cards[name] = card

        self._reflow()

    # ---- queries -----------------------------------------------------
    def selected_packages(self) -> List[str]:
        return [
            name for name, c in self._cards.items()
            if c.is_checked() and not c.is_installed()
        ]

    def all_card_names(self) -> List[str]:
        return list(self._cards.keys())

    def cards_missing_icon(self) -> List[str]:
        return [name for name, c in self._cards.items() if not c.has_icon()]

    # ---- mutators ----------------------------------------------------
    def set_filter(self, text: str) -> None:
        self._filter_text = (text or "").strip().lower()
        self._reflow()

    def select_all(self) -> None:
        for card in self._cards.values():
            card.set_checked(True)

    def deselect_all(self) -> None:
        for card in self._cards.values():
            card.set_checked(False)

    def update_card_icon(self, name: str, pixmap: QPixmap) -> None:
        card = self._cards.get(name)
        if card is not None:
            card.set_icon_pixmap(pixmap)

    def update_card_icon_from_path(self, name: str, path: str) -> None:
        if not path:
            return
        pix = QPixmap(path)
        if not pix.isNull():
            self.update_card_icon(name, pix)

    # ---- internal ----------------------------------------------------
    def _on_card_toggled(self, _name: str, _state: bool) -> None:
        self.selection_changed.emit()

    def _apply_synchronous_icon(self, card: MenagerieCard) -> None:
        name = card.package_name()
        cached = mm.cached_icon_path(name)
        if cached.exists():
            pix = QPixmap(str(cached))
            if not pix.isNull():
                card.set_icon_pixmap(pix)
                return
        local = mm.package_preview_image(name)
        if local is not None and local.exists():
            pix = QPixmap(str(local))
            if not pix.isNull():
                card.set_icon_pixmap(pix)

    def _reflow(self) -> None:
        needle = self._filter_text
        visible_order = [
            (name, card) for name, card in self._cards.items()
            if not needle or needle in name.lower()
        ]
        for card in self._cards.values():
            card.setVisible(False)

        for i, (_name, card) in enumerate(visible_order):
            row = i // GRID_COLUMNS
            col = i % GRID_COLUMNS
            self._grid.addWidget(card, row, col)
            card.setVisible(True)


class IconFetchWorker(QThread):
    """Background worker -- fills the icon cache for a list of names.

    Steps mirror DEMO 1:1: load the cached ``_index.json``, refresh
    from GitHub Trees API if the index is incomplete, then download
    each missing PNG via ``ensure_cached_icon`` on a thread pool.
    """

    icon_ready = pyqtSignal(str, str)        # name, cache_path
    progress = pyqtSignal(int, int)          # done, total
    failed = pyqtSignal(str)                 # human-readable error
    finished_all = pyqtSignal()

    MAX_WORKERS = 8

    def __init__(self, names: List[str], parent=None) -> None:
        super().__init__(parent)
        self._names = list(names)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            index = mm.load_icon_index()
            need_index_refresh = (
                not index
                or any(n not in index for n in self._names)
            )
            if need_index_refresh:
                try:
                    fresh = mm.fetch_icon_index()
                    if fresh:
                        index = fresh
                        mm.save_icon_index(index)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(f"icon index fetch failed: {exc}")

            todo = [
                n for n in self._names
                if not mm.has_cached_icon(n)
            ]
            total = len(todo)
            done = 0
            if total == 0:
                self.finished_all.emit()
                return

            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
                futures = {
                    pool.submit(
                        mm.ensure_cached_icon,
                        name,
                        index.get(name),
                    ): name
                    for name in todo
                }
                for fut in as_completed(futures):
                    if self._cancelled:
                        break
                    name = futures[fut]
                    try:
                        path = fut.result()
                    except Exception:  # noqa: BLE001
                        path = None
                    done += 1
                    self.progress.emit(done, total)
                    if path is not None:
                        self.icon_ready.emit(name, str(path))

            self.finished_all.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


__all__ = [
    "CARD_W", "CARD_H", "ICON_W", "ICON_H", "GRID_COLUMNS",
    "MenagerieCard", "MenagerieCardGrid", "IconFetchWorker",
]
