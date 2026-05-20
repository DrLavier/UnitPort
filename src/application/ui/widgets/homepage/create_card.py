"""HomepageCreateCanvas — homepage "Create New Project" card.

Layout (the whole card is :class:`HomepageCreateCanvas`, fixed-width
on the homepage so the sibling Local Files card can absorb the
remaining horizontal real estate):

.. code:: text

    +----------------------------------+
    | title row (HomepageCard chrome)  |
    +----------------------------------+
    | a) [ New Project ] [ To Project ]|
    | b) project-name input *OR*       |
    |    existing-project picker       |
    | c) Canvas Name input             |
    |    Backend SliderSwitch          |
    |    Choose a template (2×2 grid + |
    |                       Free Build)|
    | d) [ + Create ]                  |
    +----------------------------------+

Mode toggle (a) is a segmented :class:`SliderSwitch` with two
options (``New Project`` / ``To Project``). The widget paints its
own animated highlight pill so we never wire a manual exclusive
group — we just listen to ``current_changed`` and route the new key
to :meth:`_set_mode`.

Backend picker (c) is a :class:`_BackendSwitch` (segmented
``SliderSwitch``). One segment per ``backends_registry``
canvas-owning engine; the active pill is repainted on every
selection change with that backend's theme slot
(``registers.backends.get_theme_slot``) over ``alt_t1`` text.
Unavailable backends remain visible (segment label suffixed with
``" (unavailable)"``) so the user can see what isn't wired yet.

Create button (d) is a ``LaviButton`` (kind=border, spec=save) with a
``icon_add.svg`` tinted to ``safe_zone`` (the same hue as the border
and the label) at ``size_normal`` font size, stretched edge-to-edge
across the card.

The ``submitted(project_path, canvas_name, engine_id, template_path)``
signal contract is unchanged — ``MainWindow._create_and_open_canvas``
remains the wired consumer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    I18n,
    LaviButton,
    LaviComboBox,
    SliderSwitch,
    log_warning,
    tr,
)

from application.service.engines import get_engine_service
from application.service.projects import ProjectInfo, get_project_store
from registers import backends as backends_registry

from .card_base import HomepageCard
from .templates import Template, list_templates_for_backend


_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

_MODE_NEW = "new"
_MODE_EXISTING = "existing"


# Engine-id → tuple of component import names that must all resolve via
# ``importlib.util.find_spec`` for the engine to be considered usable
# without a full :func:`backends_registry.refresh_engine_availability`
# pass. Only composite training engines need this — IsaacLab requires a
# subprocess probe (too expensive to run during widget construction) so
# we trust whatever's already in ``backends_installed.json`` for it.
_LIGHT_PROBE_MODULES: dict = {
    "sb3_mujoco": ("stable_baselines3", "mujoco", "gymnasium"),
}


def _light_probe_engine_available(engine_id: str) -> bool:
    """In-process probe for whether ``engine_id``'s component deps are
    importable. Returns False for engines without a registered probe.

    Used as a fallback when the registry's ``backends_installed.json``
    row for ``engine_id`` is missing or stale — common right after a
    fresh install, before any UI has triggered ``refresh_engine_availability``.
    """
    mods = _LIGHT_PROBE_MODULES.get(engine_id)
    if not mods:
        return False
    from importlib.util import find_spec
    try:
        return all(find_spec(m) is not None for m in mods)
    except Exception:                                         # noqa: BLE001
        return False


def _tinted_pixmap(
    name: str, side: int, tint: Optional[str] = None,
) -> Optional[QPixmap]:
    """Render ``name`` SVG to a square QPixmap.

    Pass ``tint=None`` to keep the SVG's intrinsic stroke/fill colours
    (used for icons like ``icon_build`` whose source colour is already
    correct). Pass a hex string to recolour every visible pixel via
    ``CompositionMode_SourceIn`` (preserves the alpha mask). Returns
    ``None`` when the asset is missing or QtSvg is unavailable.
    """
    svg_path = Assets.find_icon(name)
    if svg_path is None:
        return None
    try:
        from PyQt6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(str(svg_path))
    except Exception:                                         # noqa: BLE001
        return None
    pm = QPixmap(side, side)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    try:
        renderer.render(painter)
        if tint:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(pm.rect(), QColor(tint))
    finally:
        painter.end()
    return pm


def _tinted_icon(name: str, side: int, tint: str) -> Optional[QIcon]:
    """Convenience wrapper around :func:`_tinted_pixmap` returning a
    QIcon for use with ``QPushButton.setIcon``.
    """
    pm = _tinted_pixmap(name, side=side, tint=tint)
    return QIcon(pm) if pm is not None else None


def _template_icon_spec(template: "Template") -> tuple:
    """Resolve ``(icon_name, theme_slot)`` for a template's prefix icon.

    ``theme_slot`` is ``None`` when the SVG should render with its
    intrinsic colours — only the Free Build entry uses that path. File
    templates pick their icon from the robot form prefix (``Go2`` →
    quadruped, ``G1``/``G2`` → humanoid) and their tint from the algo
    suffix (``AMP`` / ``PPO``), giving each preset a distinct
    (form × algo) hue in the homepage grid.
    """
    if template.is_blank:
        return ("icon_build", None)
    title = template.title or ""
    if title.startswith("Go2"):
        icon = "icon_quadruped"
        if "AMP" in title:
            return (icon, "misc_1")
        if "PPO" in title:
            return (icon, "misc_2")
        return (icon, "misc_1")
    if title.startswith("G1") or title.startswith("G2"):
        icon = "icon_humanoid"
        if "AMP" in title:
            return (icon, "misc_3")
        if "PPO" in title:
            return (icon, "misc_4")
        return (icon, "misc_3")
    if title.startswith("H1"):
        # H1_PPO inherits the (humanoid, misc_3) tint G1_AMP used before it
        # was retired — keeps the homepage grid visually anchored to a
        # warm-humanoid slot in the row G1_AMP previously occupied.
        return ("icon_humanoid", "misc_3")
    return ("", None)


class _TemplateCard(QFrame):
    """One clickable template entry in the right-side grid.

    Layout: ``[icon][text]`` — a square icon column on the left and a
    vertical title/subtitle stack on the right. The icon is resolved
    via :func:`_template_icon_spec` (per-template SVG + theme-slot
    tint); Free Build keeps its SVG's intrinsic colour.
    """

    _ICON_SIDE = 28

    clicked = pyqtSignal(str)   # template_path or "blank:<engine>"

    def __init__(
        self,
        template: Template,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._template = template
        self._selected = False
        self.setObjectName("homepageTemplateCard")
        self.setProperty("selected", "0")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(72)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # ---- icon column ---------------------------------------------
        icon_name, tint_slot = _template_icon_spec(template)
        if icon_name:
            tint = Config.get_color(tint_slot) if tint_slot else None
            pm = _tinted_pixmap(icon_name, side=self._ICON_SIDE, tint=tint)
            if pm is not None:
                icon_lbl = QLabel(self)
                icon_lbl.setObjectName("homepageTemplateCardIcon")
                icon_lbl.setPixmap(pm)
                icon_lbl.setFixedSize(self._ICON_SIDE, self._ICON_SIDE)
                layout.addWidget(
                    icon_lbl, 0, Qt.AlignmentFlag.AlignVCenter
                )

        # ---- text column ---------------------------------------------
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        self._title_lbl = QLabel(template.title)
        self._title_lbl.setObjectName("homepageTemplateCardTitle")
        self._title_lbl.setWordWrap(False)
        text_col.addWidget(self._title_lbl)

        if template.subtitle:
            self._subtitle_lbl = QLabel(template.subtitle)
            self._subtitle_lbl.setObjectName("homepageTemplateCardSubtitle")
            text_col.addWidget(self._subtitle_lbl)

        layout.addLayout(text_col, 1)

    def template(self) -> Template:
        return self._template

    def key(self) -> str:
        return self._template.template_path or f"blank:{self._template.engine_id}"

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty("selected", "1" if selected else "0")
        st = self.style()
        if st is not None:
            st.unpolish(self)
            st.polish(self)

    def mousePressEvent(self, ev):  # noqa: D401
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key())
        super().mousePressEvent(ev)


class _BackendSwitch(SliderSwitch):
    """SliderSwitch whose active pill takes its bg from the currently
    selected backend's theme slot (and ``alt_t1`` for the pill text).

    The underlying SliderSwitch repaints from the public ``_active_pill_bg``
    and ``_active_text`` attributes each frame, so swapping them and
    calling :meth:`update` is enough to retint the pill — no painter
    override needed. We re-resolve the slot on every selection change
    so different backends paint with their own catalogued theme tone.
    """

    def __init__(
        self,
        options: List,
        *,
        parent: Optional[QWidget] = None,
        height: int = 36,
        min_segment_width: int = 110,
    ) -> None:
        super().__init__(
            options,
            parent=parent,
            height=height,
            min_segment_width=min_segment_width,
        )
        self.current_changed.connect(
            lambda _i, _k: self._refresh_pill_color()
        )
        self._refresh_pill_color()

    def _refresh_pill_color(self) -> None:
        key = self.currentKey()
        if not key:
            self.update()
            return
        slot = backends_registry.get_theme_slot(key)
        color = Config.get_color(slot) if slot else None
        if color:
            self._active_pill_bg = color
            self._active_text = Config.get_color("alt_t1")
        self.update()

    def apply_theme(self) -> None:                            # type: ignore[override]
        # Let the base class re-read its catalogue slots (track bg /
        # inactive text / etc.), then re-stamp the backend-specific
        # active-pill tint so a theme switch never reverts us to the
        # generic ``highlight`` slot.
        super().apply_theme()
        self._refresh_pill_color()


class HomepageCreateCanvas(HomepageCard):
    """Bottom-right homepage card — creates a new canvas in a new or
    existing project.

    Holds its own form state; emits :attr:`submitted` once validated.
    """

    # (project_path, canvas_name, engine_id, template_path) — forwarded
    # by the create form on submit.
    submitted = pyqtSignal(str, str, str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            title_key="homepage.create.title",
            title_default="Create New Project",
            parent=parent,
        )
        self._store = get_project_store()
        self._template_cards: List[_TemplateCard] = []
        self._selected_key: str = ""
        self._mode: str = _MODE_NEW

        # All form rows go straight into the inherited ``content_layout``;
        # there is no inner split any more. ``HomepagePage`` pins this
        # card to a fixed width and gives the remaining homepage row
        # width to its sibling :class:`HomepageLocalFiles` card.
        self._left_layout = self.content_layout
        # Overall vertical spacing between sections in the form column.
        self._left_layout.setContentsMargins(0, 8, 0, 4)
        self._left_layout.setSpacing(20)

        # Rows, top-to-bottom — templates sits between Backend (inside
        # _build_canvas_inputs) and the Create button.
        self._build_project_block()
        self._build_canvas_inputs()
        self._build_template_section()
        self._build_error_label()
        self._build_create_row()
        # Stretch sentinel so the rows hug the top instead of spreading.
        self._left_layout.addStretch(1)

        # Reactions
        I18n.instance().language_changed.connect(self._retranslate)
        self._store.snapshot_changed.connect(self._on_snapshot_changed)
        # Engine availability can flip mid-session (Isaac Lab self-install,
        # manual register-local in the user panel, ...). When it does the
        # EngineService emits ``changed("")`` — relabel the backend pills
        # so newly-installed engines drop their "(unavailable)" suffix and
        # any flip-back is visible immediately.
        get_engine_service().changed.connect(self._on_engine_availability_changed)

        # Initial populate (backends are baked into the SliderSwitch at
        # construction time — see _build_canvas_inputs — so no separate
        # populator runs here).
        self._populate_existing_projects()
        # Default the mode to "To Project" whenever the user already has at
        # least one project on disk — the common path for return visits is
        # adding a canvas to an existing project, not spinning up a brand
        # new one. Only fall back to "New Project" when the store is empty.
        if self._store.snapshot():
            self._mode_switch.setCurrentKey(
                "homepage.create.mode_existing", animated=False, emit=False,
            )
            self._set_mode(_MODE_EXISTING)
        self._populate_templates(self._current_backend())
        self._apply_create_theme()

    # ------------------------------------------------------------------
    # Left column builders
    # ------------------------------------------------------------------

    def _build_project_block(self) -> None:
        """(a+b) Project section — label / mode SliderSwitch / input.

        Single tight-spacing column so the label, the New-or-To toggle,
        and the resulting input read as one grouped field rather than
        three loose rows. The SliderSwitch sits between the label and
        the input, flipping the stacked editor (text input vs existing
        picker) below it.
        """
        block = QWidget(self)
        block.setObjectName("homepageCreateProjectBlock")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        # Tight inner spacing — three components form one logical field.
        layout.setSpacing(8)

        # Adaptive label: "Project Name" when creating fresh, "Pick a
        # project" when picking from disk. Stays in the same DOM slot so
        # the layout doesn't reflow on mode switch.
        self._project_label = QLabel(
            tr("homepage.create.label_project_name", "Project Name")
        )
        self._project_label.setObjectName("homepageCreateLabel")
        layout.addWidget(self._project_label)

        # --- SliderSwitch (mode toggle) --------------------------------
        self._mode_switch = SliderSwitch(
            [
                ("homepage.create.mode_new", "New Project"),
                # Spec pins this label to "To Project" — DO NOT rename
                # back to "Use Existing"; the user explicitly insisted
                # on this wording.
                ("homepage.create.mode_existing", "To Project"),
            ],
            parent=block,
            height=35,
            min_segment_width=110,
        )
        self._mode_switch.current_changed.connect(self._on_mode_switch_changed)
        # Wrap so the slider hugs the left edge instead of stretching.
        sw_row = QHBoxLayout()
        sw_row.setContentsMargins(0, 0, 0, 0)
        sw_row.setSpacing(0)
        sw_row.addWidget(self._mode_switch, 0)
        sw_row.addStretch(1)
        layout.addLayout(sw_row)

        # --- Stacked input (text input | existing-project combo) -------
        self._project_stack = QStackedWidget(block)
        self._project_stack.setObjectName("homepageCreateProjectStack")

        self._name_edit = QLineEdit(self._project_stack)
        self._name_edit.setObjectName("homepageCreateNameEdit")
        self._name_edit.setFixedHeight(35)
        self._name_edit.setPlaceholderText(tr(
            "homepage.create.placeholder_name",
            "New Project Name",
        ))
        self._name_edit.textEdited.connect(self._on_name_edited)
        self._project_stack.addWidget(self._name_edit)

        self._existing_combo = LaviComboBox([], parent=self._project_stack)
        self._existing_combo.setObjectName("homepageCreateExistingCombo")
        self._existing_combo.setFixedHeight(35)
        self._project_stack.addWidget(self._existing_combo)

        layout.addWidget(self._project_stack)
        self._left_layout.addWidget(block)

    def _on_mode_switch_changed(self, _index: int, key: str) -> None:
        if key == "homepage.create.mode_new":
            self._set_mode(_MODE_NEW)
        elif key == "homepage.create.mode_existing":
            self._set_mode(_MODE_EXISTING)

    def _build_canvas_inputs(self) -> None:
        """(c) Canvas Name (top-down) + Backend SliderSwitch (top-down).

        The Backend segments are spun up from the registry at build
        time — we read :func:`backends_registry.list_canvas_owning_engines`
        first, then construct the SliderSwitch with one segment per
        engine. Each segment's active-pill tint follows that engine's
        theme slot via :class:`_BackendSwitch`.
        """
        # --- Canvas Name (label above, input below) --------------------
        canvas_block = QWidget(self)
        canvas_layout = QVBoxLayout(canvas_block)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(8)
        canvas_lbl = QLabel(tr("homepage.create.label_canvas_name", "Canvas Name"))
        canvas_lbl.setObjectName("homepageCreateLabel")
        canvas_layout.addWidget(canvas_lbl)
        self._canvas_name_edit = QLineEdit(canvas_block)
        self._canvas_name_edit.setObjectName("homepageCreateCanvasNameEdit")
        self._canvas_name_edit.setFixedHeight(35)
        self._canvas_name_edit.setPlaceholderText(tr(
            "homepage.create.placeholder_canvas", "e.g. main"
        ))
        self._canvas_name_edit.setText("main")
        self._canvas_name_edit.textEdited.connect(self._on_name_edited)
        canvas_layout.addWidget(self._canvas_name_edit)
        self._left_layout.addWidget(canvas_block)

        # --- Backend (label above, segmented switch below) -------------
        backend_block = QWidget(self)
        backend_layout = QVBoxLayout(backend_block)
        backend_layout.setContentsMargins(0, 0, 0, 0)
        backend_layout.setSpacing(8)
        backend_lbl = QLabel(tr("homepage.create.label_backend", "Backend"))
        backend_lbl.setObjectName("homepageCreateLabel")
        backend_layout.addWidget(backend_lbl)

        # Pull the registered canvas-owning engine list now so the
        # SliderSwitch is born with the correct segment count. Each
        # entry's label folds in availability ("(unavailable)") so the
        # user can still see (but be informed about) backends that
        # aren't wired up yet.
        self._backend_ids = backends_registry.list_canvas_owning_engines()
        installed = backends_registry.list_installed()
        # Cache per-engine availability for both option-label and default
        # selection — light probe fills in for engines whose row hasn't
        # been written to backends_installed.json yet (composite engines
        # like sb3_mujoco are commonly missing right after install).
        availability: dict = {}
        options: List[tuple] = []
        for eid in self._backend_ids:
            row = installed.get(eid) or {}
            available = bool(row.get("available", False))
            if not available:
                available = _light_probe_engine_available(eid)
            availability[eid] = available
            display = backends_registry.get_display_name(eid)
            if not available:
                display = (
                    f"{display} "
                    f"({tr('homepage.create.engine_unavailable', 'unavailable')})"
                )
            options.append((eid, display))

        self._backend_switch = _BackendSwitch(
            options,
            parent=backend_block,
            height=35,
            min_segment_width=120,
        )
        self._backend_switch.current_changed.connect(
            lambda _i, key: self._on_backend_changed(key)
        )
        # Default to the first available backend, falling back to the
        # first registered one when nothing is installed yet.
        first_available = next(
            (eid for eid in self._backend_ids if availability.get(eid)),
            None,
        )
        if first_available:
            self._backend_switch.setCurrentKey(
                first_available, animated=False, emit=False
            )
        # Wrap so the switch hugs the left edge — SliderSwitch sizes
        # itself to its segments, and we don't want a stray expand to
        # stretch the pill widths.
        switch_row = QHBoxLayout()
        switch_row.setContentsMargins(0, 0, 0, 0)
        switch_row.setSpacing(0)
        switch_row.addWidget(self._backend_switch, 0)
        switch_row.addStretch(1)
        backend_layout.addLayout(switch_row)
        self._left_layout.addWidget(backend_block)

    def _build_error_label(self) -> None:
        self._error_label = QLabel("", self)
        self._error_label.setObjectName("homepageCreateError")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        self._left_layout.addWidget(self._error_label)

    def _build_create_row(self) -> None:
        """(d) Create button — LaviButton border+save, full-width, with
        an ``icon_add.svg`` tinted to ``safe_zone`` so the icon matches
        the border/label colour.
        """
        # border + save → bg transparent, fg/border safe_zone. The icon
        # SVG ships with a white stroke, so we re-render it with the
        # safe_zone tint to keep visual unity with the label.
        safe = Config.get_color("safe_zone")
        self._create_btn = LaviButton(
            "homepage.create.btn",
            default="Create",
            kind="border",
            spec="save",
            parent=self,
        )
        # LaviButton hard-codes font-size: size_small in its own QSS, so
        # bumping QFont alone is ignored — we override via an id-selector
        # rule in ``_apply_create_theme`` against this objectName.
        self._create_btn.setObjectName("homepageCreateBtn")
        # Stretch horizontally and stay readable vertically. Expanding
        # H-policy + addWidget at stretch=1 lets the button fill the
        # full column width while still centring its label/icon group.
        self._create_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._create_btn.setFixedHeight(50)
        self._create_btn.clicked.connect(self._on_create)
        # Tint the SVG to safe_zone and pin icon size to the (px) value
        # of size_normal so it matches the QSS-driven text height (the
        # stylesheet font-size wins over QFont, so fontMetrics() can't
        # be trusted at this point).
        icon_side = Config.get_font_size("size_normal")
        tinted = _tinted_icon("icon_add", side=icon_side, tint=safe)
        if tinted is not None:
            self._create_btn.setIcon(tinted)
            self._create_btn.setIconSize(QSize(icon_side, icon_side))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._create_btn, 1)
        self._left_layout.addLayout(row)

    # ------------------------------------------------------------------
    # Template picker (left column, between Backend and Create button)
    # ------------------------------------------------------------------

    def _build_template_section(self) -> None:
        """Templates block — section label + 2×2 preset grid + Free
        Build full-width row, packed into the left form column.

        The whole block lives in a self-owned QVBoxLayout so the inner
        spacing stays tight regardless of the outer left-column 20-px
        cadence (which targets section-to-section, not label-to-grid).
        """
        block = QWidget(self)
        block.setObjectName("homepageCreateTemplateBlock")
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(8)

        tpl_lbl = QLabel(tr("homepage.create.label_template", "Choose a template"))
        tpl_lbl.setObjectName("homepageCreateLabel")
        block_layout.addWidget(tpl_lbl)

        self._grid_host = QWidget(block)
        self._grid_host.setObjectName("homepageCreateGridHost")
        self._grid_layout = QGridLayout(self._grid_host)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(8)
        # Vertical 8 keeps the preset rows + Free Build close together —
        # the larger 40-px gap looked airy in the wide right column, but
        # inside a fixed 380-px left column it crowds out everything
        # below.
        self._grid_layout.setVerticalSpacing(8)
        self._grid_layout.setColumnStretch(0, 1)
        self._grid_layout.setColumnStretch(1, 1)
        block_layout.addWidget(self._grid_host)

        self._left_layout.addWidget(block)

    # ------------------------------------------------------------------
    # Populators
    # ------------------------------------------------------------------

    def _populate_existing_projects(self) -> None:
        snapshot: List[ProjectInfo] = list(self._store.snapshot())
        items: List[tuple] = [(str(p.path), p.name) for p in snapshot]
        self._existing_combo.setItems(items)
        if snapshot:
            self._existing_combo.setCurrentKey(str(snapshot[0].path))

    def _populate_templates(self, backend_id: str) -> None:
        # Wipe the grid completely. Row stretch factors are sticky across
        # ``clear()`` so reset them too, otherwise a previous backend's
        # bottom-stretch sentinel will keep pushing rows around.
        while self._grid_layout.count():
            it = self._grid_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for r in range(self._grid_layout.rowCount() + 1):
            self._grid_layout.setRowStretch(r, 0)
        self._template_cards = []

        templates = list_templates_for_backend(backend_id)
        if not templates:
            empty = QLabel(tr(
                "homepage.create.no_templates",
                "No templates available for this backend.",
            ))
            empty.setObjectName("homepageCreateEmpty")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            self._grid_layout.addWidget(empty, 0, 0, 1, 2)
            self._selected_key = ""
            return

        presets = [t for t in templates if not t.is_blank]
        blank = next((t for t in templates if t.is_blank), None)

        # 2×2 preset matrix, row-major. Beyond 4 presets the extra rows
        # extend downwards (still 2 per row); fewer than 4 just leaves
        # empty slots in the matrix so the Free Build row stays anchored
        # below at a stable position.
        for i, tpl in enumerate(presets):
            row, col = divmod(i, 2)
            card = _TemplateCard(tpl, parent=self._grid_host)
            card.clicked.connect(self._on_template_clicked)
            self._template_cards.append(card)
            self._grid_layout.addWidget(card, row, col)

        preset_rows = (len(presets) + 1) // 2 if presets else 0

        if blank is not None:
            blank_card = _TemplateCard(blank, parent=self._grid_host)
            blank_card.clicked.connect(self._on_template_clicked)
            self._template_cards.append(blank_card)
            # Free Build spans the whole row so its width matches the
            # 2-column preset matrix above.
            self._grid_layout.addWidget(blank_card, preset_rows, 0, 1, 2)

        used_rows = preset_rows + (1 if blank is not None else 0)
        self._grid_layout.setRowStretch(used_rows, 1)

        # Default to Go2_PPO when present (the canonical quadruped PPO
        # quickstart), otherwise fall back to the first card so the
        # picker always has something selected.
        default_card = next(
            (c for c in self._template_cards
             if c.template().title == "Go2_PPO"),
            self._template_cards[0] if self._template_cards else None,
        )
        if default_card is not None:
            self._selected_key = default_card.key()
            default_card.set_selected(True)
        else:
            self._selected_key = ""
        self._apply_create_theme()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        if mode not in (_MODE_NEW, _MODE_EXISTING):
            return
        self._mode = mode
        if mode == _MODE_NEW:
            self._project_stack.setCurrentIndex(0)
            self._project_label.setText(
                tr("homepage.create.label_project_name", "Project Name")
            )
        else:
            self._project_stack.setCurrentIndex(1)
            self._project_label.setText(
                tr("homepage.create.label_pick_project", "Pick a project")
            )
            # Snapshot may have changed since the card was built — pull
            # the latest list before showing the existing-project combo.
            self._populate_existing_projects()
        self._clear_error()

    def _on_backend_changed(self, backend_id: str) -> None:
        self._populate_templates(backend_id or self._current_backend())
        self._clear_error()

    def _on_template_clicked(self, key: str) -> None:
        self._selected_key = key
        for card in self._template_cards:
            card.set_selected(card.key() == key)
        self._clear_error()

    def _on_name_edited(self, _text: str) -> None:
        self._clear_error()

    def _on_snapshot_changed(self, _snapshot: object) -> None:
        self._populate_existing_projects()

    def _on_engine_availability_changed(self, _engine_id: str) -> None:
        """Rebuild Backend SliderSwitch segment labels from the freshly
        refreshed ``backends_installed.json`` so a just-installed engine
        loses its ``(unavailable)`` suffix without us recreating the widget.

        Preserves the user's current selection by key. If the active key
        was unavailable and is now available, no re-selection is needed —
        the segment stays put and just sheds the suffix.
        """
        installed = backends_registry.list_installed()
        options: List[tuple] = []
        for eid in self._backend_ids:
            row = installed.get(eid) or {}
            available = bool(row.get("available", False))
            if not available:
                available = _light_probe_engine_available(eid)
            display = backends_registry.get_display_name(eid)
            if not available:
                display = (
                    f"{display} "
                    f"({tr('homepage.create.engine_unavailable', 'unavailable')})"
                )
            options.append((eid, display))

        prev_key = self._current_backend()
        self._backend_switch.set_options(options)
        if prev_key:
            self._backend_switch.setCurrentKey(
                prev_key, animated=False, emit=False,
            )

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _current_backend(self) -> str:
        try:
            return str(self._backend_switch.currentKey() or "").strip()
        except Exception:                                     # noqa: BLE001
            return ""

    def _selected_template(self) -> Optional[Template]:
        for card in self._template_cards:
            if card.key() == self._selected_key:
                return card.template()
        return None

    def _resolve_project(self) -> Optional[ProjectInfo]:
        if self._mode == _MODE_NEW:
            raw_name = self._name_edit.text().strip()
            if not raw_name:
                self._show_error(tr(
                    "homepage.create.error_no_name",
                    "Type a project name first.",
                ))
                return None
            if not _NAME_RE.match(raw_name):
                self._show_error(tr(
                    "homepage.create.error_invalid_name",
                    "Project name may only contain letters, digits, '_' and '-'.",
                ))
                return None
            info = self._store.create_project(raw_name)
            if info is None:
                self._show_error(tr(
                    "homepage.create.error_create_failed",
                    "Could not create project '{name}' (name may clash with an existing one — check the log).",
                ).format(name=raw_name))
                return None
            return info
        try:
            key = str(self._existing_combo.currentKey() or "").strip()
        except Exception:                                     # noqa: BLE001
            key = ""
        if not key:
            self._show_error(tr(
                "homepage.create.error_no_existing",
                "Pick an existing project from the list, or switch to 'New Project'.",
            ))
            return None
        info = self._store.find_by_path(Path(key))
        if info is None:
            self._show_error(tr(
                "homepage.create.error_existing_missing",
                "That project is no longer on disk — refresh the list and pick again.",
            ))
            return None
        return info

    def _on_create(self) -> None:
        engine_id = self._current_backend()
        if not engine_id:
            self._show_error(tr(
                "homepage.create.error_no_backend",
                "Pick a backend before creating.",
            ))
            return
        tpl = self._selected_template()
        if tpl is None:
            self._show_error(tr(
                "homepage.create.error_no_template",
                "Pick a template (or 'Free Build') before creating.",
            ))
            return

        raw_canvas = self._canvas_name_edit.text().strip()
        if not raw_canvas:
            self._show_error(tr(
                "homepage.create.error_no_canvas_name",
                "Type a canvas name (e.g. 'main').",
            ))
            return
        if not _NAME_RE.match(raw_canvas):
            self._show_error(tr(
                "homepage.create.error_invalid_canvas_name",
                "Canvas name may only contain letters, digits, '_' and '-'.",
            ))
            return

        info = self._resolve_project()
        if info is None:
            return

        canvas_name = raw_canvas
        try:
            subdir = backends_registry.canvas_subdir(engine_id) or engine_id
        except Exception as exc:                              # noqa: BLE001
            log_warning(
                f"[homepage.create] canvas_subdir({engine_id!r}) failed: {exc!r}"
            )
            subdir = engine_id
        canvas_dir = Path(info.path) / "canvas" / subdir
        target = canvas_dir / f"{canvas_name}.canvas.json"
        if target.exists():
            for i in range(2, 100):
                cand = f"{canvas_name}_{i}"
                if not (canvas_dir / f"{cand}.canvas.json").exists():
                    canvas_name = cand
                    break

        self._clear_error()
        if self._mode == _MODE_NEW:
            self._name_edit.clear()
        self.submitted.emit(
            str(info.path), canvas_name, engine_id, tpl.template_path,
        )

    # ------------------------------------------------------------------
    # Strings + error display
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self._name_edit.setPlaceholderText(tr(
            "homepage.create.placeholder_name",
            "New Project Name",
        ))
        self._canvas_name_edit.setPlaceholderText(tr(
            "homepage.create.placeholder_canvas", "e.g. main"
        ))
        # The adaptive project label depends on the current mode, so it
        # has to be re-resolved on language switch (the SliderSwitch
        # auto-retranslates its own segments).
        if self._mode == _MODE_NEW:
            self._project_label.setText(
                tr("homepage.create.label_project_name", "Project Name")
            )
        else:
            self._project_label.setText(
                tr("homepage.create.label_pick_project", "Pick a project")
            )
        self._populate_existing_projects()
        self._populate_templates(self._current_backend())

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)
        log_warning(f"[homepage.create] {msg}")

    def _clear_error(self) -> None:
        self._error_label.clear()
        self._error_label.setVisible(False)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:                            # type: ignore[override]
        super().apply_theme()
        self._apply_create_theme()
        self._mode_switch.refresh_style()
        self._create_btn.refresh_style()
        self._backend_switch.refresh_style()
        self._existing_combo.refresh_style()

    def _apply_create_theme(self) -> None:
        text = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        bg_field = Config.get_color("bg_3")
        border_field = Config.get_color("border_2")
        focus_border = Config.get_color("hover_1")
        danger = Config.get_color("danger_zone")
        # Template cards mirror the Account-card sign-in button palette:
        # row_1 fill + 1px border_2 in idle, btn_1 fill on hover. The
        # selected state adds a safe_zone border so the current pick
        # still reads as a primed CTA.
        tpl_bg = Config.get_color("row_1")
        tpl_bg_hover = Config.get_color("btn_1")
        tpl_border = Config.get_color("border_2")
        tpl_bg_sel = Config.get_color("btn_1")
        tpl_border_sel = Config.get_color("safe_zone")
        tpl_title = Config.get_color("main_t1")
        tpl_sub = Config.get_color("main_c2")
        size_small = Config.get_font_size("size_small")
        size_normal = Config.get_font_size("size_normal")
        size_mini = Config.get_font_size("size_mini")

        qss = (
            f"QLabel#homepageCreateSectionLabel {{"
            f"  color: {text};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 700;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageCreateLabel {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            f"QLineEdit#homepageCreateNameEdit, "
            f"QLineEdit#homepageCreateCanvasNameEdit {{"
            f"  background: {bg_field};"
            f"  color: {text};"
            f"  border: 1px solid {border_field};"
            f"  border-radius: 6px;"
            f"  padding: 6px 10px;"
            f"  font-size: {size_normal}px;"
            f"}}"
            f"QLineEdit#homepageCreateNameEdit:focus, "
            f"QLineEdit#homepageCreateCanvasNameEdit:focus {{"
            f"  border: 1px solid {focus_border};"
            f"}}"
            f"QWidget#homepageCreateGridHost, "
            f"QWidget#homepageCreateTemplateBlock, "
            f"QWidget#homepageCreateProjectBlock {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QFrame#homepageTemplateCard {{"
            f"  background: {tpl_bg};"
            f"  border: 1px solid {tpl_border};"
            f"  border-radius: 6px;"
            f"}}"
            f"QFrame#homepageTemplateCard:hover {{"
            f"  background: {tpl_bg_hover};"
            f"}}"
            f"QFrame#homepageTemplateCard[selected=\"1\"] {{"
            f"  background: {tpl_bg_sel};"
            f"  border: 1px solid {tpl_border_sel};"
            f"}}"
            f"QLabel#homepageTemplateCardIcon {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageTemplateCardTitle {{"
            f"  color: {tpl_title};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageTemplateCardSubtitle {{"
            f"  color: {tpl_sub};"
            f"  background: transparent;"
            f"  font-size: {size_mini}px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageCreateEmpty {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  padding: 16px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homepageCreateError {{"
            f"  color: {danger};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            # Bump the CTA to size_normal — LaviButton's own QSS pins
            # font-size: size_small, but an id-selector rule from this
            # ancestor wins on specificity.
            f"QPushButton#homepageCreateBtn {{"
            f"  font-size: {size_normal}px;"
            f"}}"
        )
        self.setStyleSheet(self.styleSheet() + qss)


# Legacy alias — older imports (page.py, etc.) addressed this card as
# ``NewProjectCard``. Keep the name in scope so nothing breaks until all
# call sites migrate to :class:`HomepageCreateCanvas`.
NewProjectCard = HomepageCreateCanvas
