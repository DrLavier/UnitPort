"""HomepagePage — homepage container for the empty-state landing page.

Hosted by ``_MainPanel.MODE_PICKER`` in place of the legacy
``NewCanvasForm``. Four cards under a ``QGridLayout``:

    +-----------+----------------+
    | UserCard  | CommunityCard  |
    +-----------+----------------+
    | Create    | LocalFiles     |   (Create has fixed width; LocalFiles
    | (fixed)   |  (stretch)     |    absorbs the remaining horizontal room)
    +-----------+----------------+

Three signals are forwarded to ``MainWindow``:

* :attr:`submitted` — ``(project_path, canvas_name, engine_id,
  template_path)``. Connected to ``MainWindow._create_and_open_canvas``
  which writes the new canvas file (using ``template_path`` as the seed
  when non-empty) and auto-opens it.
* :attr:`canvas_open_requested` — ``(project_path, canvas_file_id)``.
  Connected to a thin MainWindow handler that does
  ``open_project(project_path)`` + ``open_canvas(canvas_file_id)`` so a
  workspace-row Load button jumps directly into a specific canvas.
* :attr:`community_apply_requested` — ``(ReleaseInfo)``. The Community
  card's Download button surfaces a user-selected release here;
  ``MainWindow`` routes it into the existing ``ApplyUpdateTask`` flow
  (the user has already seen release notes inline on the card, so the
  intermediate ``UpdateAvailableDialog`` is skipped).

The class itself does no I/O or auth work — every card owns its own
manager singleton and signal wiring.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)

from unitport_sdk import Config, log_debug

from .community_card import HomepageCommunityCard
from .create_card import HomepageCreateCanvas
from .home_account_card import HomeAccountCard
from .projects_card import HomepageLocalFiles


class HomepagePage(QWidget):
    """2x2 homepage. Replaces NewCanvasForm as the picker widget."""

    # (project_path, canvas_name, engine_id, template_path)
    submitted = pyqtSignal(str, str, str, str)
    # (project_path, canvas_file_id) — Load button / row double-click on
    # the workspaces table. The host (MainWindow) is expected to drive
    # open_project + open_canvas in sequence.
    canvas_open_requested = pyqtSignal(str, str)
    # (ReleaseInfo) — Community card's Download button. MainWindow plumbs
    # this into the existing ApplyUpdateTask + UpdateProgressDialog flow,
    # bypassing UpdateAvailableDialog (release notes were shown inline).
    community_apply_requested = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("homepagePage")
        # WA_StyledBackground lets ``setStyleSheet("QWidget#homepagePage { ... }")``
        # actually paint a background — otherwise a plain QWidget stays
        # transparent and the canvas behind would bleed through.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._build_ui()
        self._wire_signals()
        self._apply_theme()
        log_debug("[ui] HomepagePage constructed")

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        # Outer breathing room doubled from 16 → 32 so the four cards
        # read as floating tiles on the bg_3 canvas.
        layout.setContentsMargins(32, 32, 32, 32)
        # Horizontal gap between left/right columns is widened more than
        # the vertical gap — the cards are visually distinct sections,
        # not a uniform grid, and the extra horizontal air keeps each
        # column's content from crowding the divider zone.
        layout.setHorizontalSpacing(28)
        layout.setVerticalSpacing(14)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        # Account + News on top have limited content (3 CTA buttons /
        # static news placeholder); workspaces + create-project rows below
        # take the bulk of the vertical real estate. 2 : 5 keeps the top
        # row dense but readable.
        layout.setRowStretch(0, 2)
        layout.setRowStretch(1, 5)

        self._user_card = HomeAccountCard(parent=self)
        self._community_card = HomepageCommunityCard(parent=self)
        # Bottom row holds two independent cards: the Create form
        # (fixed-width) on the left and the Local Files browser on the
        # right (absorbs all remaining width).
        self._create_card = HomepageCreateCanvas(parent=self)
        self._local_files_card = HomepageLocalFiles(parent=self)

        # The top two cards keep the equal-weight column-stretch sizing
        # rule (Ignored H-policy + min-width floor).
        for card in (
            self._user_card,
            self._community_card,
        ):
            card.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
            )
            card.setMinimumWidth(240)

        # Create card pins to a fixed width — wide enough for the form
        # contents (380-px form column + ~80 px of card chrome padding)
        # and small enough that the Local Files table next door owns the
        # rest of the homepage row.
        self._create_card.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        self._create_card.setFixedWidth(460)
        self._local_files_card.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        self._local_files_card.setMinimumWidth(420)

        layout.addWidget(self._user_card, 0, 0)
        layout.addWidget(self._community_card, 0, 1)
        # Bottom row: nest the two cards in a QHBoxLayout host so the
        # grid's equal column stretch (used by the top row) doesn't
        # force them to share width 50/50 — the create card stays at
        # its fixed width and the local-files card eats the remainder.
        bottom_host = QWidget(self)
        bottom_host.setObjectName("homepageBottomRowHost")
        bottom_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bottom_row = QHBoxLayout(bottom_host)
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(layout.horizontalSpacing())
        bottom_row.addWidget(self._create_card, 0)
        bottom_row.addWidget(self._local_files_card, 1)
        layout.addWidget(bottom_host, 1, 0, 1, 2)

    def _wire_signals(self) -> None:
        self._create_card.submitted.connect(self.submitted.emit)
        # Local Files emits its canvas-open requests directly now that
        # it lives in its own card — no more forwarding through Create.
        self._local_files_card.canvas_open_requested.connect(
            self.canvas_open_requested.emit
        )
        # Community card → MainWindow apply flow.
        self._community_card.apply_requested.connect(
            self.community_apply_requested.emit
        )

    # ------------------------------------------------------------------
    # Theme (re-applied by MainWindow.apply_theme if it dispatches)
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        self._apply_theme()
        # Cards are independent — let them re-style too.
        for card in (
            self._user_card,
            self._community_card,
            self._create_card,
            self._local_files_card,
        ):
            try:
                card.apply_theme()
            except Exception:                                 # noqa: BLE001
                pass

    def _apply_theme(self) -> None:
        # Homepage canvas tone is bg_3 (deeper than card body bg_1) so the
        # four rounded cards read as raised tiles on the canvas instead of
        # blending into it.
        bg = Config.get_color("bg_3")
        self.setStyleSheet(
            f"QWidget#homepagePage {{ background: {bg}; }}"
        )
