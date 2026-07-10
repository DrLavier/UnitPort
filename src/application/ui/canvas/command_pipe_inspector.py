# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Command Pipe Inspector — read-only popup over the derived contract.

Opened by clicking the ``training_motion`` node's ``command_pipe`` output
port. Renders the channel table (name / kind badge / range+unit / source
pill / obs dims), the footer sums (command dims vs obs dims — two
DIFFERENT sums, the command-space vs obs-space distinction), the sin/cos
footnote (only when a non-identity encoding is present, text derived from
the layouts), the empty state (``NO_ENABLED_ITEMS``) and the unresolved
notices (``FAMILY_UNKNOWN``, ...).

Strictly render-only: every value comes from
:class:`CommandContractModel.result` — no channel-name literals, no
dimension constants, no encoding rules live here (A4 purity guard).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, tr

from application.training.command_schema import (
    UNRESOLVED_NO_ENABLED_ITEMS,
)

from .command_contract_model import CommandContractModel


_POPUP_MIN_W = 380
_POPUP_MAX_W = 520


def _font(tier: str, fallback: int, *, bold: bool = False) -> QFont:
    f = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
    f.setPixelSize(int(Config.get_font_size(tier, fallback)))
    f.setBold(bold)
    return f


def _badge_label(text: str, bg_slot: str, fg_slot: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(_font("size_mini", 10, bold=True))
    lbl.setStyleSheet(
        f"background:{Config.get_color(bg_slot, '#3A3A3A')};"
        f"color:{Config.get_color(fg_slot, '#F5F5F5')};"
        f"border-radius:4px; padding:1px 6px;"
    )
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


class CommandPipeInspector(QWidget):
    """Floating read-only popup anchored at the Command Pipe port."""

    def __init__(self, model: CommandContractModel) -> None:
        # Top-level Qt Popup (no parent) — same convention as
        # canvas popups.py: parenting to the view breaks the popup grab.
        super().__init__(None)
        self._model = model
        self.setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumWidth(_POPUP_MIN_W)
        self.setMaximumWidth(_POPUP_MAX_W)
        self.setStyleSheet(
            f"QWidget {{ background:{Config.get_color('canvas_contract_popup_bg', '#202020')}; }}"
        )
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(12, 10, 12, 10)
        self._root.setSpacing(6)
        model.contract_changed.connect(self._rebuild)
        self._rebuild()

    # ------------------------------------------------------------------

    def _clear(self) -> None:
        while self._root.count():
            item = self._root.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            lay = item.layout()
            if lay is not None:
                while lay.count():
                    sub = lay.takeAt(0)
                    sw = sub.widget()
                    if sw is not None:
                        sw.deleteLater()

    def _rebuild(self) -> None:
        self._clear()
        title = QLabel(tr("canvas.cmd_inspector.title", "Command Contract"))
        title.setFont(_font("size_normal", 14, bold=True))
        title.setStyleSheet(f"color:{Config.get_color('main_t1', '#E8E6DA')};")
        self._root.addWidget(title)

        result = self._model.result
        if result is None:
            return
        contract = result.contract
        unresolved = list(result.unresolved)
        codes = {u.code for u in unresolved}

        # Unresolved notices — every entry surfaces (design §2: collection
        # instead of raising is only acceptable because the UI renders it).
        for u in unresolved:
            notice = QLabel(
                tr(f"canvas.cmd_inspector.unresolved.{u.code}", u.message)
            )
            notice.setWordWrap(True)
            notice.setFont(_font("size_small", 12))
            notice.setStyleSheet(
                f"color:{Config.get_color('canvas_contract_notice_fg', '#F59E0B')};"
            )
            self._root.addWidget(notice)

        if UNRESOLVED_NO_ENABLED_ITEMS in codes and not contract.channels:
            # Empty state — no fake rows (frontend spec).
            empty_t = QLabel(
                tr("canvas.cmd_inspector.empty_title", "Command Contract Empty")
            )
            empty_t.setFont(_font("size_small", 12, bold=True))
            empty_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_t.setStyleSheet(
                f"color:{Config.get_color('canvas_node_param_text', '#9CA3AF')};"
            )
            empty_h = QLabel(tr(
                "canvas.cmd_inspector.empty_hint",
                "Enable at least one Training Item.\n"
                "Command channels are automatically derived.",
            ))
            empty_h.setFont(_font("size_small", 12))
            empty_h.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_h.setStyleSheet(
                f"color:{Config.get_color('canvas_node_param_text', '#9CA3AF')};"
            )
            self._root.addSpacing(10)
            self._root.addWidget(empty_t)
            self._root.addWidget(empty_h)
            self._root.addSpacing(10)
            self.adjustSize()
            return

        # ---- channel table -------------------------------------------
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        headers = (
            tr("canvas.cmd_inspector.col_name", "Name"),
            tr("canvas.cmd_inspector.col_kind", "Kind"),
            tr("canvas.cmd_inspector.col_range", "Range"),
            tr("canvas.cmd_inspector.col_source", "Source"),
            tr("canvas.cmd_inspector.col_obs", "Obs Dims"),
        )
        hdr_color = Config.get_color("canvas_node_param_text", "#9CA3AF")
        for col, text in enumerate(headers):
            h = QLabel(text)
            h.setFont(_font("size_mini", 10, bold=True))
            h.setStyleSheet(f"color:{hdr_color};")
            grid.addWidget(h, 0, col)

        stripe = Config.get_color("canvas_contract_stripe_bg", "#262626")
        for row, ch in enumerate(contract.channels, start=1):
            name = QLabel(ch.name)
            name.setFont(_font("size_small", 12))
            name.setStyleSheet(f"color:{Config.get_color('main_t1', '#E8E6DA')};")
            kind = _badge_label(
                tr(f"canvas.cmd_inspector.kind.{ch.kind}", ch.kind),
                f"canvas_contract_badge_{ch.kind}_bg",
                "canvas_contract_badge_fg",
            )
            unit = f" {ch.unit}" if ch.unit else ""
            rng = QLabel(f"[{ch.low:g}, {ch.high:g}]{unit}")
            rng.setFont(_font("size_small", 12))
            rng.setStyleSheet(f"color:{Config.get_color('main_t1', '#E8E6DA')};")
            source = _badge_label(
                tr(f"canvas.cmd_inspector.source.{ch.source}", ch.source),
                "canvas_contract_pill_bg",
                "canvas_contract_pill_fg",
            )
            obs = QLabel(str(ch.obs_encoding.obs_dims))
            obs.setFont(_font("size_small", 12))
            obs.setAlignment(Qt.AlignmentFlag.AlignCenter)
            obs.setStyleSheet(f"color:{Config.get_color('main_t1', '#E8E6DA')};")
            if row % 2 == 0:
                for w in (name, rng, obs):
                    w.setStyleSheet(w.styleSheet() + f"background:{stripe};")
            grid.addWidget(name, row, 0)
            grid.addWidget(kind, row, 1)
            grid.addWidget(rng, row, 2)
            grid.addWidget(source, row, 3)
            grid.addWidget(obs, row, 4)
        self._root.addLayout(grid)

        # ---- footer (always visible) ----------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(
            f"color:{Config.get_color('canvas_contract_popup_border', '#3A3A3A')};"
        )
        self._root.addWidget(sep)
        footer = QLabel(
            f"{tr('canvas.cmd_inspector.footer_cmd', 'Command Dims')}: "
            f"{contract.total_command_dim()}    "
            f"{tr('canvas.cmd_inspector.footer_obs', 'Observation Dims')}: "
            f"{contract.total_obs_dim()}"
        )
        footer.setFont(_font("size_small", 12, bold=True))
        footer.setStyleSheet(f"color:{Config.get_color('main_t1', '#E8E6DA')};")
        self._root.addWidget(footer)

        # Footnote only when a non-identity encoding is present; text
        # derives from the layouts (frontend correction §7.5).
        layouts = contract.layouts_present()
        if layouts:
            names = ", ".join(
                tr(f"canvas.cmd_inspector.layout.{lay}", lay) for lay in layouts
            )
            note = QLabel(tr(
                "canvas.cmd_inspector.footnote_encoding",
                "(phase channels encoded as {layouts})",
            ).format(layouts=names))
            note.setFont(_font("size_mini", 10))
            note.setStyleSheet(
                f"color:{Config.get_color('canvas_node_param_text', '#9CA3AF')};"
            )
            self._root.addWidget(note)
        self.adjustSize()


def open_command_pipe_inspector(scene, node_item, global_pos: QPoint) -> Optional[CommandPipeInspector]:
    """Open the Inspector for ``node_item`` anchored at ``global_pos``.

    Retains the popup on the scene (``_cmd_inspector_ref``) so it is not
    garbage-collected before Qt shows it; Qt.Popup semantics close it on
    any outside click, WA_DeleteOnClose destroys it.
    """
    from .command_contract_model import get_contract_model

    model = get_contract_model(node_item)
    popup = CommandPipeInspector(model)
    scene._cmd_inspector_ref = popup
    popup.move(global_pos)
    popup.show()
    return popup


__all__ = ["CommandPipeInspector", "open_command_pipe_inspector"]
