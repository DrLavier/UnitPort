# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AiApiSettingsDialog — register the LLM agent(s) AI Build calls.

The user may register several **agents** (one connection each: base URL / API
key / model + a cloud-upload switch) and pick one as the *default*. The left
column is the agent list (``+`` / ``-`` to add / remove, click to switch); the
right column edits the selected agent. **Save** writes the whole registry, marks
the selected agent the default, and mirrors that choice to
``user.ini[AiBuild].active_agent_id`` so it takes effect on every launch
(persistence lives in
:mod:`application.service.ai_orchestration.api_config`, under
``USER_CONFIG_DIR`` — NOT ``src/`` and NOT ``system.ini``, CLAUDE.md §4).

Each agent carries an ``upload_to_cloud`` marker (its switch) that defaults
**off**; the cloud-sync service refuses to upload the config file unless every
agent opts in, so an API key stays on the machine unless the user opts in.

The token may be given two ways per agent: a literal key, or the NAME of an
environment variable holding it (the switch picks which the field edits). In env
mode only the variable NAME is stored; the secret is read from the environment
at run time and never written to disk.

Opened from the AI-Build top-row agent button
(:attr:`AiBuildPanel.settings_requested`) — for editing, and as the entry gate
when no usable connection is saved yet.

§5 compliance: colors via ``Config.get_color`` against ``system.ini[Theme]``;
fonts via ``Config.get_font_size`` (input fields pinned to the ``size_small``
tier so their text matches the surrounding labels). UI strings through ``tr`` /
``i18n_bind``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18nLabel,
    LaviLineEdit,
    SwitchButton,
    i18n_bind,
    log_info,
    setButton,
    setLineEdit,
    tr,
)

from application.service.ai_orchestration.api_config import (
    ENV_TOKEN_MIN_LEN,
    AgentRegistry,
    ApiConfig,
    is_env_token_valid,
    load_agent_registry,
    new_agent_id,
    save_agent_registry,
)


class _AgentRow(QFrame):
    """One registered agent in the left list: name + model subtitle.

    Single-click selects. Full hover + selected visuals (the ``active`` dynamic
    property toggles the accent border); colors are read at use-time (§5) via the
    dialog's stylesheet, so the row tracks the active palette."""

    clicked = pyqtSignal(str)

    def __init__(
        self, agent_id: str, name: str, subtitle: str, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._agent_id = agent_id
        self.setObjectName("aiAgentRow")
        self.setProperty("active", "false")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(2)
        self._name = QLabel(name, self)
        self._name.setObjectName("aiAgentRowName")
        lay.addWidget(self._name)
        self._sub = QLabel(subtitle, self)
        self._sub.setObjectName("aiAgentRowSub")
        self._sub.setVisible(bool(subtitle))
        lay.addWidget(self._sub)

    def agent_id(self) -> str:
        return self._agent_id

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._agent_id)
        super().mousePressEvent(event)


class AiApiSettingsDialog(QDialog):
    """Modal: agent list (left) + connection editor (right) for AI Build."""

    _LIST_WIDTH = 190

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aiApiSettingsDialog")
        self.setModal(True)
        self.setWindowTitle(
            tr("canvas.ai_build.settings_title", default="AI Build — API Settings")
        )
        self.resize(720, 470)

        # Registry (never empty in the dialog — a fresh blank agent seeds an
        # otherwise-empty registry so there is always something to edit).
        self._registry: AgentRegistry = load_agent_registry()
        if not self._registry.agents:
            seed = ApiConfig(
                id=new_agent_id(),
                name=tr("canvas.ai_build.settings_agent_new", default="New Agent"),
            )
            self._registry.agents.append(seed)
            self._registry.active_id = seed.id
        self._selected_id: str = self._registry.active_id or self._registry.agents[0].id
        self._agent_rows: Dict[str, _AgentRow] = {}

        # Per-session literal/env buffers so toggling the switch never loses text.
        self._buf_key = ""
        self._buf_env = ""

        self._build_ui()
        self._rebuild_agent_list()
        self._load_agent_into_form(self._registry.active())
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        intro = I18nLabel(
            "canvas.ai_build.settings_intro",
            default="Register the model connection(s) AI Build calls. The selected "
            "agent becomes the default; settings are saved under your user config "
            "folder.",
            parent=self,
        )
        intro.setObjectName("aiApiSettingsIntro")
        intro.setWordWrap(True)
        outer.addWidget(intro, 0)

        main = QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(12)
        main.addWidget(self._build_agent_panel(), 0)
        main.addWidget(self._build_form_panel(), 1)
        outer.addLayout(main, 1)

        # --- Validation status -----------------------------------------
        self._status = QLabel("", self)
        self._status.setObjectName("aiApiSettingsStatus")
        self._status.setWordWrap(True)
        outer.addWidget(self._status, 0)

        # --- Footer ----------------------------------------------------
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        self._btn_cancel = setButton(
            "canvas.ai_build.settings_cancel", 96, 32,
            kind="normal", spec="none", default="Cancel", parent=self,
        )
        self._btn_cancel.clicked.connect(self.reject)
        footer.addWidget(self._btn_cancel, 0)
        self._btn_save = setButton(
            "canvas.ai_build.settings_save", 96, 32,
            kind="normal", spec="save", default="Save", parent=self,
        )
        self._btn_save.clicked.connect(self._on_save)
        footer.addWidget(self._btn_save, 0)
        outer.addLayout(footer)

    def _build_agent_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("aiAgentPanel")
        panel.setFixedWidth(self._LIST_WIDTH)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        title = QLabel(
            tr("canvas.ai_build.settings_agents_title", default="Agents"), panel
        )
        title.setObjectName("aiAgentPanelTitle")
        lay.addWidget(title, 0)

        scroll = QScrollArea(panel)
        scroll.setObjectName("aiAgentScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._agent_host = QWidget()
        self._agent_host.setObjectName("aiAgentHost")
        self._agent_list_layout = QVBoxLayout(self._agent_host)
        self._agent_list_layout.setContentsMargins(0, 0, 0, 0)
        self._agent_list_layout.setSpacing(6)
        self._agent_list_layout.addStretch(1)
        scroll.setWidget(self._agent_host)
        lay.addWidget(scroll, 1)

        # +/- controls.
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self._btn_add = QPushButton("＋", panel)
        self._btn_add.setObjectName("aiAgentAdd")
        self._btn_add.setFixedSize(28, 26)
        self._btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._btn_add, "setToolTip",
            "canvas.ai_build.settings_agent_add_tip", default="Add agent",
        )
        self._btn_add.clicked.connect(self._on_add_agent)
        controls.addWidget(self._btn_add, 0)
        self._btn_remove = QPushButton("－", panel)
        self._btn_remove.setObjectName("aiAgentRemove")
        self._btn_remove.setFixedSize(28, 26)
        self._btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        i18n_bind(
            self._btn_remove, "setToolTip",
            "canvas.ai_build.settings_agent_remove_tip", default="Remove agent",
        )
        self._btn_remove.clicked.connect(self._on_remove_agent)
        controls.addWidget(self._btn_remove, 0)
        controls.addStretch(1)
        lay.addLayout(controls, 0)
        return panel

    def _build_form_panel(self) -> QWidget:
        block = QFrame(self)
        block.setObjectName("aiApiSettingsBlock")
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(12, 10, 12, 12)
        block_layout.setSpacing(10)

        # Agent display name.
        self._le_name: LaviLineEdit = setLineEdit(
            placeholder=tr(
                "canvas.ai_build.settings_agent_name_ph", default="e.g. GPT-4o"
            ),
            parent=block,
        )
        self._le_name.setFixedHeight(28)
        self._pin_field_font(self._le_name)
        block_layout.addWidget(
            self._kv_row(
                block,
                tr("canvas.ai_build.settings_agent_name", default="Name"),
                self._le_name,
            ),
            0,
        )

        self._le_base_url: LaviLineEdit = setLineEdit(
            placeholder="https://api.openai.com/v1",
            parent=block,
        )
        self._le_base_url.setFixedHeight(28)
        self._pin_field_font(self._le_base_url)
        block_layout.addWidget(
            self._kv_row(
                block,
                tr("canvas.ai_build.settings_base_url", default="Base URL"),
                self._le_base_url,
            ),
            0,
        )

        # API key: a literal token OR an environment-variable NAME. One field,
        # the switch picks which it edits; both buffers are kept so toggling
        # back and forth never loses typed text.
        self._le_api_key: LaviLineEdit = setLineEdit(parent=block)
        self._le_api_key.setFixedHeight(28)
        self._pin_field_font(self._le_api_key)
        self._le_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._le_api_key.textChanged.connect(self._on_key_text_changed)
        self._btn_reveal = setButton(
            "canvas.ai_build.settings_reveal",
            64,
            28,
            kind="border",
            checkable=True,
            default="Show",
            parent=block,
        )
        self._btn_reveal.toggled.connect(self._on_reveal_toggled)
        block_layout.addWidget(
            self._kv_row(
                block,
                tr("canvas.ai_build.settings_api_key", default="API Key"),
                self._inline(block, self._le_api_key, self._btn_reveal),
            ),
            0,
        )

        # Switch: read the token from an environment variable name.
        self._sw_use_env = SwitchButton(parent=block, checked=False)
        block_layout.addWidget(
            self._switch_row(
                block,
                tr(
                    "canvas.ai_build.settings_use_env",
                    default="Read token from environment variable",
                ),
                tr(
                    "canvas.ai_build.settings_use_env_tip",
                    default="Store only the variable NAME; the token is read from "
                    "your environment at run time and never written to disk.",
                ),
                self._sw_use_env,
            ),
            0,
        )
        self._sw_use_env.toggled.connect(self._on_use_env_toggled)

        # Live env-resolution preview (only meaningful in env mode).
        self._env_preview = QLabel("", block)
        self._env_preview.setObjectName("aiApiSettingsEnvPreview")
        self._env_preview.setWordWrap(True)
        block_layout.addWidget(self._env_preview, 0)

        self._le_model: LaviLineEdit = setLineEdit(
            placeholder=tr(
                "canvas.ai_build.settings_model_ph",
                default="e.g. gpt-4o / claude-opus-4-8",
            ),
            parent=block,
        )
        self._le_model.setFixedHeight(28)
        self._pin_field_font(self._le_model)
        block_layout.addWidget(
            self._kv_row(
                block,
                tr("canvas.ai_build.settings_model", default="Model"),
                self._le_model,
            ),
            0,
        )

        # --- Cloud-upload switch ---------------------------------------
        self._sw_cloud = SwitchButton(parent=block, checked=False)
        block_layout.addWidget(
            self._switch_row(
                block,
                tr("canvas.ai_build.settings_cloud", default="Upload this API info to cloud"),
                tr(
                    "canvas.ai_build.settings_cloud_tip",
                    default="Off by default. While off, the online sync feature never "
                    "uploads your API key.",
                ),
                self._sw_cloud,
            ),
            0,
        )

        block_layout.addStretch(1)
        return block

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pin_field_font(self, widget: QWidget) -> None:
        """Pin an input's font to the ``size_small`` tier so its text matches the
        surrounding labels (LaviLineEdit ships at ``size_normal``, one tier too
        big for this dense form). Fonts via ``Config.get_font_size`` (§5)."""
        font = widget.font()
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        widget.setFont(font)

    def _key_placeholder(self, use_env: bool) -> str:
        if use_env:
            return tr("canvas.ai_build.settings_env_ph", default="e.g. OPENAI_API_KEY")
        return tr("canvas.ai_build.settings_api_key_ph", default="sk-…")

    def _switch_row(
        self, parent: QWidget, label_text: str, tip_text: str, switch: QWidget
    ) -> QWidget:
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        lbl = QLabel(label_text, row)
        lbl.setObjectName("aiApiSettingsCloudLabel")
        lbl.setWordWrap(True)
        tip = QLabel(tip_text, row)
        tip.setObjectName("aiApiSettingsCloudTip")
        tip.setWordWrap(True)
        text.addWidget(lbl, 0)
        text.addWidget(tip, 0)
        layout.addLayout(text, 1)
        layout.addWidget(switch, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def _kv_row(self, parent: QWidget, label_text: str, control: QWidget) -> QWidget:
        row = QWidget(parent)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        lbl = QLabel(label_text, row)
        lbl.setObjectName("aiApiSettingsKv")
        lbl.setFixedWidth(84)
        rl.addWidget(lbl, 0)
        rl.addWidget(control, 1)
        return row

    def _inline(self, parent: QWidget, edit: QWidget, btn: QWidget) -> QWidget:
        host = QWidget(parent)
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(edit, 1)
        layout.addWidget(btn, 0)
        return host

    # ------------------------------------------------------------------
    # Agent list
    # ------------------------------------------------------------------

    def _rebuild_agent_list(self) -> None:
        """(Re)build the left list from ``self._registry``, marking the selected
        agent active. Called on open, add, remove, and after a switch (so the
        stashed model subtitle refreshes)."""
        lay = self._agent_list_layout
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._agent_rows = {}
        untitled = tr("canvas.ai_build.settings_agent_untitled", "Untitled agent")
        no_model = tr("canvas.ai_build.settings_agent_no_model", "(no model)")
        for agent in self._registry.agents:
            name = (agent.name or "").strip() or agent.model.strip() or untitled
            subtitle = agent.model.strip() or no_model
            row = _AgentRow(agent.id, name, subtitle, parent=self._agent_host)
            row.clicked.connect(self._on_agent_row_clicked)
            row.set_active(agent.id == self._selected_id)
            lay.addWidget(row)
            self._agent_rows[agent.id] = row
        lay.addStretch(1)

    def _load_agent_into_form(self, agent: ApiConfig) -> None:
        """Seed the right-hand form from ``agent`` (signals blocked so the switch
        handler doesn't fight the direct field writes)."""
        use_env = bool(agent.api_key_env.strip())
        self._buf_key = agent.api_key
        self._buf_env = agent.api_key_env
        self._le_name.setText(agent.name)
        self._le_base_url.setText(agent.base_url)
        self._le_model.setText(agent.model)
        self._sw_use_env.blockSignals(True)
        self._sw_use_env.setChecked(use_env)
        self._sw_use_env.blockSignals(False)
        self._le_api_key.setText(self._buf_env if use_env else self._buf_key)
        self._le_api_key.setEchoMode(
            QLineEdit.EchoMode.Normal if use_env else QLineEdit.EchoMode.Password
        )
        self._btn_reveal.blockSignals(True)
        self._btn_reveal.setChecked(False)
        self._btn_reveal.blockSignals(False)
        self._btn_reveal.setVisible(not use_env)  # no secret to reveal in env mode
        self._le_api_key.setPlaceholderText(self._key_placeholder(use_env))
        self._sw_cloud.setChecked(agent.upload_to_cloud)
        self._status.setText("")
        self._update_env_preview()

    def _stash_form_into(self, agent: ApiConfig) -> None:
        """Capture the current (unvalidated) form into ``agent`` in memory, so a
        switch / add / remove keeps partial edits. Persistence + validation
        happen only on Save."""
        use_env = self._sw_use_env.isChecked()
        field = str(self._le_api_key.text()).strip()
        agent.name = str(self._le_name.text()).strip()
        agent.base_url = str(self._le_base_url.text()).strip()
        agent.model = str(self._le_model.text()).strip()
        agent.upload_to_cloud = bool(self._sw_cloud.isChecked())
        if use_env:
            agent.api_key_env = field
            agent.api_key = ""
        else:
            agent.api_key = field
            agent.api_key_env = ""

    def _on_agent_row_clicked(self, agent_id: str) -> None:
        if agent_id == self._selected_id:
            return
        current = self._registry.get(self._selected_id)
        if current is not None:
            self._stash_form_into(current)
        self._selected_id = agent_id
        target = self._registry.get(agent_id)
        if target is not None:
            self._load_agent_into_form(target)
        self._rebuild_agent_list()

    def _on_add_agent(self) -> None:
        current = self._registry.get(self._selected_id)
        if current is not None:
            self._stash_form_into(current)
        agent = ApiConfig(
            id=new_agent_id(),
            name=tr("canvas.ai_build.settings_agent_new", default="New Agent"),
        )
        self._registry.agents.append(agent)
        self._selected_id = agent.id
        self._load_agent_into_form(agent)
        self._rebuild_agent_list()
        self._le_name.setFocus(Qt.FocusReason.OtherFocusReason)
        self._le_name.selectAll()

    def _on_remove_agent(self) -> None:
        agent = self._registry.get(self._selected_id)
        if agent is None:
            return
        # Confirm only when the agent actually holds a connection worth losing.
        if agent.base_url.strip() or agent.model.strip() or agent.api_key.strip() \
                or agent.api_key_env.strip():
            reply = QMessageBox.question(
                self,
                tr("canvas.ai_build.settings_remove_title", "Remove agent"),
                tr(
                    "canvas.ai_build.settings_remove_body",
                    "Remove this agent registration? This cannot be undone.",
                ),
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        idx = self._registry.agents.index(agent)
        self._registry.agents.remove(agent)
        # Never leave an empty registry — reseed a blank agent to edit.
        if not self._registry.agents:
            fresh = ApiConfig(
                id=new_agent_id(),
                name=tr("canvas.ai_build.settings_agent_new", default="New Agent"),
            )
            self._registry.agents.append(fresh)
            self._selected_id = fresh.id
        else:
            neighbour = self._registry.agents[min(idx, len(self._registry.agents) - 1)]
            self._selected_id = neighbour.id
        selected = self._registry.get(self._selected_id)
        if selected is not None:
            self._load_agent_into_form(selected)
        self._rebuild_agent_list()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_reveal_toggled(self, checked: bool) -> None:
        # Reveal only applies to a literal token; in env mode the field holds a
        # variable NAME shown in clear already.
        if self._sw_use_env.isChecked():
            return
        self._le_api_key.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )

    def _on_use_env_toggled(self, checked: bool) -> None:
        # Save the current field into its buffer, then swap to the other form.
        if checked:
            self._buf_key = str(self._le_api_key.text())
            self._le_api_key.setText(self._buf_env)
            self._le_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self._btn_reveal.setVisible(False)
        else:
            self._buf_env = str(self._le_api_key.text())
            self._le_api_key.setText(self._buf_key)
            self._btn_reveal.setChecked(False)
            self._btn_reveal.setVisible(True)
            self._le_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._le_api_key.setPlaceholderText(self._key_placeholder(checked))
        self._update_env_preview()

    def _on_key_text_changed(self, _text: str) -> None:
        if self._sw_use_env.isChecked():
            self._update_env_preview()

    def _update_env_preview(self) -> None:
        """Show whether the named env var resolves to a usable token (>N chars).

        Never displays the secret itself — only the variable name + char count.
        Color is read at use-time from the theme (CLAUDE.md §5)."""
        if not self._sw_use_env.isChecked():
            self._env_preview.setText("")
            self._env_preview.setStyleSheet("")
            return
        name = str(self._le_api_key.text()).strip()
        if not name:
            self._env_preview.setText("")
            self._env_preview.setStyleSheet("")
            return
        if is_env_token_valid(name):
            n = len(os.environ.get(name, ""))
            color = Config.get_color("log_success", "#2CD620")
            self._env_preview.setStyleSheet(f"color: {color}; background: transparent;")
            self._env_preview.setText(
                tr(
                    "canvas.ai_build.settings_env_ok",
                    default="✓ {name} resolved ({n} chars)",
                ).format(name=name, n=n)
            )
        else:
            color = Config.get_color("log_warning", "#D9AB41")
            self._env_preview.setStyleSheet(f"color: {color}; background: transparent;")
            self._env_preview.setText(
                tr(
                    "canvas.ai_build.settings_env_bad",
                    default="✗ {name} is not set, or its value is ≤ {n} chars.",
                ).format(name=name, n=ENV_TOKEN_MIN_LEN)
            )

    def _on_save(self) -> None:
        agent = self._registry.get(self._selected_id)
        if agent is None:
            return
        use_env = self._sw_use_env.isChecked()
        base_url = str(self._le_base_url.text()).strip()
        model = str(self._le_model.text()).strip()
        field = str(self._le_api_key.text()).strip()
        name = str(self._le_name.text()).strip()
        upload = bool(self._sw_cloud.isChecked())

        if use_env:
            # Validate (>N chars) and store ONLY the variable name — the secret
            # is resolved at run time and never written to disk.
            if not is_env_token_valid(field):
                self._status.setText(
                    tr(
                        "canvas.ai_build.settings_env_invalid",
                        default="The environment variable must exist and hold a "
                        "value longer than {n} characters.",
                    ).format(n=ENV_TOKEN_MIN_LEN)
                )
                return
            agent.api_key = ""
            agent.api_key_env = field
        else:
            agent.api_key = field
            agent.api_key_env = ""
        agent.base_url = base_url
        agent.model = model
        agent.name = name or model
        agent.upload_to_cloud = upload

        if not agent.is_configured:
            self._status.setText(
                tr(
                    "canvas.ai_build.settings_incomplete",
                    default="Base URL, API Key and Model are all required.",
                )
            )
            return

        # The selected agent is the one being saved → make it the default.
        self._registry.active_id = agent.id
        if not save_agent_registry(self._registry):
            self._status.setText(
                tr(
                    "canvas.ai_build.settings_save_failed",
                    default="Could not write the settings file. Check the log.",
                )
            )
            return
        log_info("[ai.api_settings] saved AI Build agent registry")
        self.accept()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1", "#1E1E1E")
        bg_alt = Config.get_color("bg_2", "#1A1A1A")
        sub = Config.get_color("sub_t2", "#777777")
        main = Config.get_color("main_t1", "#D6D3C7")
        border = Config.get_color("border_2", "#3d3d3d")
        block_bg = Config.get_color("bg_3", "#101010")
        warn = Config.get_color("log_warning", "#D9AB41")
        accent = Config.get_color("safe_zone", "#36E38E")
        accent_text = Config.get_color("alt_t1", "#1E1E1E")
        hover = Config.get_color("hover_2", "#212121")
        font_normal = Config.get_font_size("size_normal", 16)
        font_small = Config.get_font_size("size_small", 12)
        self.setStyleSheet(
            f"QDialog#aiApiSettingsDialog {{ background-color: {bg}; }}"
            f"QLabel#aiApiSettingsIntro {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            # Agent list panel + rows (hover + selected visuals).
            f"QFrame#aiAgentPanel {{ background-color: {bg_alt}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
            f"QWidget#aiAgentHost, QScrollArea#aiAgentScroll {{ background: transparent; }}"
            f"QLabel#aiAgentPanelTitle {{ color: {sub}; font-size: {font_small}px; "
            f"font-weight: 600; background: transparent; }}"
            f"QFrame#aiAgentRow {{ background-color: {block_bg}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
            f"QFrame#aiAgentRow:hover {{ background-color: {hover}; }}"
            f'QFrame#aiAgentRow[active="true"] {{ border: 1px solid {accent}; }}'
            f"QLabel#aiAgentRowName {{ color: {main}; font-size: {font_small}px; "
            f"font-weight: 600; background: transparent; }}"
            f"QLabel#aiAgentRowSub {{ color: {sub}; font-size: {font_small}px; "
            f"background: transparent; }}"
            f"QPushButton#aiAgentAdd, QPushButton#aiAgentRemove {{ "
            f"background-color: {block_bg}; color: {main}; border: 1px solid {border}; "
            f"border-radius: 6px; font-size: {font_normal}px; font-weight: 600; }}"
            f"QPushButton#aiAgentAdd:hover, QPushButton#aiAgentRemove:hover {{ "
            f"border-color: {accent}; color: {accent}; }}"
            # Connection editor block.
            f"QFrame#aiApiSettingsBlock {{ background-color: {block_bg}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
            f"QLabel#aiApiSettingsKv {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#aiApiSettingsCloudLabel {{ color: {main}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#aiApiSettingsCloudTip {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#aiApiSettingsStatus {{ color: {warn}; "
            f"font-size: {font_small}px; background: transparent; }}"
            # Env preview: font from the theme; ok/bad color is applied per-update
            # (widget-level setStyleSheet) from log_success / log_warning slots.
            f"QLabel#aiApiSettingsEnvPreview {{ "
            f"font-size: {font_small}px; background: transparent; }}"
        )


__all__ = ["AiApiSettingsDialog"]
