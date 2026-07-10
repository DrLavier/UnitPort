# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Skill authoring dialog — skill_command_path_design.md Slice 2.

A modal editor for the ``training_motion`` node's ``skill_items`` param. Each skill
is a discrete policy command channel (a TRIGGER for now; enum reserved). Authoring
here produces the ``skill_items`` dict that ``derive_command_contract`` turns into a
``ContractChannel(kind="trigger", source="skill")`` on the v1 command contract.

Skill schema: ``{skill_id: {name, kind:"trigger", latch:"pulse"|"hold", enabled,
suggested_binding?}}`` — ``skill_id`` is the stable channel name; ``name`` is display.
All colors/fonts come from ``Config`` (§5). Fixed-width, self-contained.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, tr

from application.training.command_schema import (
    CONTRACT_KIND_TRIGGER,
    CONTRACT_LATCH_HOLD,
    CONTRACT_LATCH_PULSE,
)

__all__ = ["SkillAuthoringDialog"]

_LATCH_MODES = (CONTRACT_LATCH_PULSE, CONTRACT_LATCH_HOLD)


def _parse_json_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return s or "skill"


def _norm_skill(sid: str, raw: Any) -> Dict[str, Any]:
    d = raw if isinstance(raw, dict) else {}
    latch = str(d.get("latch", CONTRACT_LATCH_PULSE)).strip().lower()
    if latch not in _LATCH_MODES:
        latch = CONTRACT_LATCH_PULSE
    try:
        window_s = float(d.get("window_s", 1.0))
    except (TypeError, ValueError):
        window_s = 1.0
    if window_s <= 0.0:
        window_s = 1.0
    return {
        "name": str(d.get("name", sid)),
        "kind": CONTRACT_KIND_TRIGGER,      # trigger-only for now (enum reserved)
        "latch": latch,
        "window_s": window_s,
        "enabled": bool(d.get("enabled", True)),
        # NOTE: no key/button binding is authored here. Binding a skill to a
        # controller input is a per-user action done in the Controller panel
        # (registry-driven, per controller type), never in this training dialog.
    }


class SkillAuthoringDialog(QDialog):
    """Edit the skill (trigger) channels. Read ``result_skill_items`` after
    ``exec() == Accepted``."""

    def __init__(
        self,
        skill_items_value: Any,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        raw = _parse_json_obj(skill_items_value)
        self._skills: Dict[str, Dict[str, Any]] = {
            str(sid): _norm_skill(str(sid), v) for sid, v in raw.items()
        }
        self.result_skill_items: Dict[str, Dict[str, Any]] = dict(self._skills)

        self.setObjectName("skillAuthoringDialog")
        self.setModal(True)
        self.setWindowTitle(tr("training.skills.title", "Skill Channels"))
        self.setFixedWidth(520)

        self._list: Optional[QListWidget] = None
        self._name_edit: Optional[QLineEdit] = None
        self._latch_combo: Optional[QComboBox] = None
        self._window_edit: Optional[QLineEdit] = None
        self._enabled_chk: Optional[QCheckBox] = None
        self._preview: Optional[QPlainTextEdit] = None
        self._status: Optional[QLabel] = None

        self._init_ui()
        self.setStyleSheet(self._stylesheet())
        self._reload_list()
        self._refresh_preview()

    # ------------------------------------------------------------------ UI
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        header = QLabel(tr("training.skills.title", "Skill Channels"), self)
        header.setObjectName("skHeader")
        root.addWidget(header)

        hint = QLabel(self)
        hint.setObjectName("skHint")
        hint.setWordWrap(True)
        hint.setText(tr(
            "training.skills.hint",
            "Each skill is a discrete policy command channel (a trigger). It "
            "enters the policy obs and rides the command contract; a policy must "
            "be trained to respond to it. Binding it to a key or button is done "
            "per-user in the Controller panel after training — not here.",
        ))
        root.addWidget(hint)

        body = QGridLayout()
        body.setHorizontalSpacing(8)
        body.setVerticalSpacing(6)

        self._list = QListWidget(self)
        self._list.setObjectName("skList")
        self._list.setFixedHeight(150)
        self._list.currentItemChanged.connect(lambda *_: self._on_selected())
        body.addWidget(self._list, 0, 0, 5, 1)

        add_btn = QPushButton(tr("training.skills.add", "Add skill"), self)
        add_btn.setObjectName("skBtn")
        add_btn.clicked.connect(self._add_skill)
        body.addWidget(add_btn, 0, 1)
        rm_btn = QPushButton(tr("training.skills.remove", "Remove"), self)
        rm_btn.setObjectName("skBtn")
        rm_btn.clicked.connect(self._remove_skill)
        body.addWidget(rm_btn, 1, 1)

        form = QWidget(self)
        fl = QGridLayout(form)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setHorizontalSpacing(8)
        fl.setVerticalSpacing(6)
        fl.addWidget(self._label(tr("training.skills.name", "Name")), 0, 0)
        self._name_edit = QLineEdit(form)
        self._name_edit.setObjectName("skInput")
        self._name_edit.textEdited.connect(self._on_name_changed)
        fl.addWidget(self._name_edit, 0, 1)
        fl.addWidget(self._label(tr("training.skills.latch", "Latch")), 1, 0)
        self._latch_combo = QComboBox(form)
        self._latch_combo.setObjectName("skCombo")
        self._latch_combo.addItem(tr("training.skills.latch.pulse", "pulse (one-shot)"), CONTRACT_LATCH_PULSE)
        self._latch_combo.addItem(tr("training.skills.latch.hold", "hold (while held)"), CONTRACT_LATCH_HOLD)
        self._latch_combo.currentIndexChanged.connect(self._on_latch_changed)
        fl.addWidget(self._latch_combo, 1, 1)
        fl.addWidget(self._label(tr("training.skills.window", "Window (s)")), 2, 0)
        self._window_edit = QLineEdit(form)
        self._window_edit.setObjectName("skInput")
        self._window_edit.setPlaceholderText("1.0")
        self._window_edit.setToolTip(tr(
            "training.skills.window.tip",
            "Post-pulse decay window: the trigger obs + reward gate decay 1→0 "
            "over this many seconds. The deploy runtime rebuilds the same window.",
        ))
        self._window_edit.editingFinished.connect(self._on_window_changed)
        fl.addWidget(self._window_edit, 2, 1)
        self._enabled_chk = QCheckBox(tr("training.skills.enabled", "Enabled"), form)
        self._enabled_chk.setObjectName("skCheck")
        self._enabled_chk.toggled.connect(self._on_enabled_changed)
        fl.addWidget(self._enabled_chk, 3, 1)
        body.addWidget(form, 0, 2, 4, 1)
        body.setColumnStretch(0, 3)
        body.setColumnStretch(2, 4)
        root.addLayout(body)

        root.addWidget(self._divider())
        cap = QLabel(tr("training.skills.preview", "Emitted trigger channels"), self)
        cap.setObjectName("skCaption")
        root.addWidget(cap)
        self._preview = QPlainTextEdit(self)
        self._preview.setObjectName("skPreview")
        self._preview.setReadOnly(True)
        self._preview.setFixedHeight(84)
        root.addWidget(self._preview)

        self._status = QLabel(self)
        self._status.setObjectName("skStatus")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok = QPushButton(tr("common.ok", "OK"), self)
        ok.setObjectName("skBtnPrimary")
        ok.clicked.connect(self._on_accept)
        cancel = QPushButton(tr("common.cancel", "Cancel"), self)
        cancel.setObjectName("skBtn")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        root.addLayout(btn_row)

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text, self)
        lbl.setObjectName("skFieldLabel")
        return lbl

    def _divider(self) -> QFrame:
        line = QFrame(self)
        line.setObjectName("skDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    # ----------------------------------------------------- selection
    def _current_sid(self) -> str:
        if self._list is None:
            return ""
        it = self._list.currentItem()
        return str(it.data(Qt.ItemDataRole.UserRole)) if it is not None else ""

    def _reload_list(self) -> None:
        if self._list is None:
            return
        self._list.blockSignals(True)
        self._list.clear()
        for sid, sk in self._skills.items():
            label = sk.get("name") or sid
            item = QListWidgetItem(str(label))
            item.setData(Qt.ItemDataRole.UserRole, sid)
            self._list.addItem(item)
        self._list.blockSignals(False)
        if self._list.count():
            self._list.setCurrentRow(0)
        self._on_selected()

    def _on_selected(self) -> None:
        sid = self._current_sid()
        sk = self._skills.get(sid)
        has = sk is not None
        for w in (self._name_edit, self._latch_combo, self._enabled_chk,
                  self._window_edit):
            if w is not None:
                w.setEnabled(has)
        if not has:
            for w in (self._name_edit, self._window_edit):
                if w is not None:
                    w.setText("")
            return
        if self._name_edit:
            self._name_edit.setText(str(sk.get("name", "")))
        if self._window_edit:
            self._window_edit.setText(f"{float(sk.get('window_s', 1.0)):g}")
        if self._latch_combo:
            self._latch_combo.blockSignals(True)
            idx = max(0, self._latch_combo.findData(sk.get("latch", CONTRACT_LATCH_PULSE)))
            self._latch_combo.setCurrentIndex(idx)
            self._latch_combo.blockSignals(False)
        if self._enabled_chk:
            self._enabled_chk.blockSignals(True)
            self._enabled_chk.setChecked(bool(sk.get("enabled", True)))
            self._enabled_chk.blockSignals(False)

    # ----------------------------------------------------- mutations
    def _unique_sid(self, base: str) -> str:
        sid = _slug(base)
        if sid not in self._skills:
            return sid
        n = 2
        while f"{sid}_{n}" in self._skills:
            n += 1
        return f"{sid}_{n}"

    def _add_skill(self) -> None:
        n = len(self._skills) + 1
        name = tr("training.skills.new_name", "skill {n}").replace("{n}", str(n))
        sid = self._unique_sid(name)
        self._skills[sid] = _norm_skill(sid, {"name": name})
        self._reload_list()
        self._select_sid(sid)
        self._refresh_preview()

    def _remove_skill(self) -> None:
        sid = self._current_sid()
        if sid in self._skills:
            del self._skills[sid]
            self._reload_list()
            self._refresh_preview()

    def _select_sid(self, sid: str) -> None:
        if self._list is None:
            return
        for row in range(self._list.count()):
            if str(self._list.item(row).data(Qt.ItemDataRole.UserRole)) == sid:
                self._list.setCurrentRow(row)
                return

    def _on_name_changed(self, text: str) -> None:
        sid = self._current_sid()
        if sid in self._skills:
            self._skills[sid]["name"] = str(text)
            it = self._list.currentItem() if self._list else None
            if it is not None:
                it.setText(str(text) or sid)
            self._refresh_preview()

    def _on_latch_changed(self, _idx: int) -> None:
        sid = self._current_sid()
        if sid in self._skills and self._latch_combo is not None:
            self._skills[sid]["latch"] = str(self._latch_combo.currentData())
            self._refresh_preview()

    def _on_window_changed(self) -> None:
        sid = self._current_sid()
        if sid not in self._skills or self._window_edit is None:
            return
        raw = self._window_edit.text().strip()
        try:
            w = float(raw)
            if w <= 0.0:
                raise ValueError
        except ValueError:
            self._window_edit.setText(f"{float(self._skills[sid].get('window_s', 1.0)):g}")
            self._set_status(
                tr("training.skills.bad_window", "window must be a positive number"),
                error=True,
            )
            return
        self._skills[sid]["window_s"] = w
        self._set_status("", error=False)
        self._refresh_preview()

    def _on_enabled_changed(self, on: bool) -> None:
        sid = self._current_sid()
        if sid in self._skills:
            self._skills[sid]["enabled"] = bool(on)
            self._refresh_preview()

    # ----------------------------------------------------- preview + accept
    def _refresh_preview(self) -> None:
        if self._preview is None:
            return
        lines = []
        for sid, sk in self._skills.items():
            if not bool(sk.get("enabled", True)):
                continue
            lines.append(
                f"{sid}  (trigger, latch={sk.get('latch', 'pulse')}, "
                f"window={float(sk.get('window_s', 1.0)):g}s)"
            )
        self._preview.setPlainText(
            "\n".join(lines) or tr("training.skills.preview_empty", "No enabled skills.")
        )

    def _on_accept(self) -> None:
        # Light validation: non-empty unique ids (the emitter re-validates loudly).
        if any(not str(sid).strip() for sid in self._skills):
            self._set_status(tr("training.skills.bad_id", "every skill needs an id"), error=True)
            return
        self.result_skill_items = {
            sid: dict(sk) for sid, sk in self._skills.items()
        }
        self.accept()

    def _set_status(self, text: str, *, error: bool) -> None:
        if self._status is None:
            return
        self._status.setText(text or "")
        slot = "danger_zone" if error else "sub_t2"
        size = Config.get_font_size("size_small", 12)
        self._status.setStyleSheet(
            f"color: {Config.get_color(slot, '#C0504D' if error else '#8A8778')}; "
            f"font-size: {size}px; background: transparent;"
        )

    # ----------------------------------------------------- theme (§5)
    def _stylesheet(self) -> str:
        bg = Config.get_color("bg_1", "#15171C")
        bg2 = Config.get_color("bg_2", "#1A1C22")
        text = Config.get_color("main_t2", "#FFFFFF")
        t1 = Config.get_color("main_t1", "#D6D3C7")
        muted = Config.get_color("sub_t2", "#8A8778")
        accent = Config.get_color("canvas_contract_badge_trigger_bg", "#B0722D")
        border = Config.get_color("border_1", "#2A2D35")
        size_normal = Config.get_font_size("size_normal", 14)
        size_small = Config.get_font_size("size_small", 12)
        return f"""
            QDialog#skillAuthoringDialog {{ background: {bg}; }}
            QLabel#skHeader {{ color: {accent}; font-size: {size_normal}px; font-weight: 600; }}
            QLabel#skHint, QLabel#skCaption {{ color: {muted}; font-size: {size_small}px; }}
            QLabel#skFieldLabel {{ color: {t1}; font-size: {size_small}px; }}
            QFrame#skDivider {{ color: {border}; background: {border}; max-height: 1px; }}
            QListWidget#skList, QPlainTextEdit#skPreview {{
                background: {bg2}; color: {t1}; border: 1px solid {border};
                font-size: {size_small}px;
            }}
            QLineEdit#skInput, QComboBox#skCombo {{
                background: {bg2}; color: {text}; border: 1px solid {border};
                padding: 2px 4px; font-size: {size_small}px;
            }}
            QCheckBox#skCheck {{ color: {t1}; font-size: {size_small}px; }}
            QPushButton#skBtn, QPushButton#skBtnPrimary {{
                background: {bg2}; color: {text}; border: 1px solid {border};
                padding: 4px 10px; font-size: {size_small}px;
            }}
            QPushButton#skBtnPrimary {{ background: {accent}; border: 1px solid {accent}; }}
            QLabel#skStatus {{ font-size: {size_small}px; }}
        """
