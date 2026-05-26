# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Sidebar panels for the Scripts work-mode.

In Scripts mode the sidebar rail flips to four buttons:
``[Training] [Rewards] [Termins] [Observs]``. Each opens one of the panels
in this module. Clicking an item in any panel emits
``script_selected(virtual_id)`` which ``MainWindow`` routes to
``MissionControlPanel.load_script`` for display in the compiler editor.

Virtual id forms:

* ``project:<rel>`` / ``system:<rel>`` — real ``.py`` files (TrainingScriptsPanel,
  resolvable via :func:`application.service.projects.resolve_file`).
* ``registry:<kind>:<key>`` — in-memory ``TaskModuleItem`` from
  ``src/scripts/{rewards,terminations,observations}/registry.py``. The MC
  panel renders the item's ``il_inline`` source (rewards/observations) or a
  generated stub (terminations have no inline source).

All four panels share a tab-less single-tab ``LaviTabTable`` host so the
visuals match the flattened Project Files panel.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from unitport_sdk import (
    Config,
    LaviTabTable,
    log_warning,
    setLaviTabTable,
)

from application.service.projects import (
    ProjectInfo,
    list_script_groups,
)
from application.service.signals import current_backend, get_app_signals


# ---------------------------------------------------------------------------
# LaviTabTable restyle helper (sidebar-local)
# ---------------------------------------------------------------------------
# Geometry constants mirrored from ``unitport_sdk.ui`` so the override
# paintEvents below match the SDK's layout exactly. The only behavioural
# delta: NO background fill (we want the sidebar panel's bg_3 to show
# through entirely), and the highlight slot tints the group-header text
# / caret instead of being applied as a fill.
_LTT_CARET_SIZE = 8
_LTT_LABEL_GAP = 6


def _patch_header_paint(header) -> None:
    """Replace ``header.paintEvent``: no fill, highlight text/caret.

    Idle: caret + label drawn in ``highlight`` so the header reads as a
    visually distinct row against the transparent list surface.
    Hover: text becomes bold; colour stays ``highlight``.
    """

    def paintEvent(event) -> None:                              # noqa: ARG001
        painter = QPainter(header)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # No fillRect — header background is fully transparent.

        text_col = QColor(Config.get_color("highlight", "#F6D393"))

        rect = header.rect()
        cs = _LTT_CARET_SIZE
        cx = 6 + header._indent
        cy = (rect.height() - cs) // 2
        path = QPainterPath()
        if header._expanded:
            path.moveTo(cx, cy + 1)
            path.lineTo(cx + cs, cy + 1)
            path.lineTo(cx + cs / 2.0, cy + cs - 1)
        else:
            path.moveTo(cx + 1, cy)
            path.lineTo(cx + 1, cy + cs)
            path.lineTo(cx + cs - 1, cy + cs / 2.0)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(text_col))
        painter.drawPath(path)

        font = QFont(header._font)
        if header._hover:
            font.setBold(True)
        painter.setPen(QPen(text_col))
        painter.setFont(font)
        text_x = cx + cs + _LTT_LABEL_GAP
        text_rect = QRectF(text_x, 0, rect.width() - text_x - 2, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            header._label_text,
        )
        painter.end()

    header.paintEvent = paintEvent
    header.update()


def _patch_item_paint(item) -> None:
    """Replace ``item.paintEvent``: no fill; keep indicator, prefix, label.

    Drops the SDK's ``list_bg`` / ``hover_bg`` rectangle so the list
    surface stays fully transparent. The selection indicator box,
    optional coloured prefix badge, and main label all keep the SDK's
    original colour logic (hover → ``highlight``, checked → ``safe_zone``,
    idle → ``main_t1``).
    """
    # Cache QFontMetrics import once; QFontMetrics lives in QtGui too.
    from PyQt6.QtGui import QFontMetrics

    def paintEvent(event) -> None:                              # noqa: ARG001
        painter = QPainter(item)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # No fillRect — row background is fully transparent.

        idle_text = QColor(Config.get_color("main_t1", "#D6D3C7"))
        muted = QColor(Config.get_color("sub_t1", "#A0A0A0"))
        accent = QColor(Config.get_color("safe_zone", "#36E38E"))
        hover_text = QColor(Config.get_color("highlight", "#F6D393"))

        rect = item.rect()
        box_x = item._indent + 1
        box_y = (rect.height() - item._box_side) // 2
        box_rect = QRectF(box_x, box_y, item._box_side, item._box_side)
        radius = 2.0
        if item._checked:
            painter.setPen(QPen(accent, 1.0))
            painter.setBrush(QBrush(accent))
            painter.drawRoundedRect(box_rect, radius, radius)
        else:
            painter.setPen(QPen(muted, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box_rect, radius, radius)

        if item._hover:
            text_color = hover_text
        elif item._checked:
            text_color = accent
        else:
            text_color = idle_text
        font = QFont(item._font)
        if item._hover:
            font.setBold(True)
        painter.setFont(font)
        text_x = box_x + item._box_side + _LTT_LABEL_GAP

        if item._prefix_text and item._prefix_color_slot:
            prefix_color = QColor(Config.get_color(item._prefix_color_slot))
            fm = QFontMetrics(font)
            prefix_w = fm.horizontalAdvance(item._prefix_text)
            prefix_rect = QRectF(text_x, 0, prefix_w, rect.height())
            painter.setPen(QPen(prefix_color))
            painter.drawText(
                prefix_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                item._prefix_text,
            )
            text_x += prefix_w + _LTT_LABEL_GAP

        painter.setPen(QPen(text_color))
        text_rect = QRectF(text_x, 0, rect.width() - text_x - 2, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            item._label_text,
        )
        painter.end()

    item.paintEvent = paintEvent
    item.update()


def _strip_widget_fill(w) -> None:
    """Disable autoFillBackground + clear palette window/base colour."""
    if w is None:
        return
    w.setAutoFillBackground(False)
    # SDK's _apply_list_bg sets Window role; clear it back to transparent
    # so a stray paint engine fallback doesn't fill with the old colour.
    pal = w.palette()
    transparent = QColor(0, 0, 0, 0)
    from PyQt6.QtGui import QPalette
    pal.setColor(QPalette.ColorRole.Window, transparent)
    pal.setColor(QPalette.ColorRole.Base, transparent)
    w.setPalette(pal)


def restyle_tab_table_for_sidebar(table: LaviTabTable) -> None:
    """Completely strip the LaviTabTable's background fills.

    Three changes vs. the SDK default:

    1. The list surface widgets (every nested QWidget under the table)
       drop their palette window fills + autoFillBackground so the panel's
       background shows through.
    2. Each group header (top-level + sub-) is re-painted with NO fill;
       its caret + label are tinted with the ``highlight`` theme slot to
       give it visual identity without a coloured bar.
    3. Each item row is re-painted with NO fill — only the indicator box,
       optional prefix badge, and label remain. Hover / checked states
       still tint the text (highlight / safe_zone) per SDK behaviour.

    Idempotent — safe to call after every LaviTabTable rebuild and on
    every theme switch.
    """
    # 1) Recursively strip background fills on every child widget. SDK's
    # ``_apply_list_bg`` only touches a handful of widgets; sweeping the
    # entire subtree catches anything they missed (and is harmless on
    # widgets that already had transparent backgrounds).
    from PyQt6.QtWidgets import QWidget
    _strip_widget_fill(table)
    for child in table.findChildren(QWidget):
        _strip_widget_fill(child)

    # 2) + 3) Replace paintEvent on every group header and item row that
    # currently lives in the table. _tab_groups stores
    #   {tab_id: [(group_id, header, [item_rows]), ...]}
    # — both top-level and sub-groups (sub headers are appended to the
    # same list, see ui.py:3707).
    tab_groups = getattr(table, "_tab_groups", {}) or {}
    for entries in tab_groups.values():
        for entry in entries:
            try:
                _gid, header, rows = entry
            except (ValueError, TypeError):
                continue
            if header is not None:
                _patch_header_paint(header)
            for row in rows or ():
                if row is not None:
                    _patch_item_paint(row)


# ---------------------------------------------------------------------------
# Shared single-tab list skeleton
# ---------------------------------------------------------------------------
class _SingleTabListPanel(QWidget):
    """Base widget hosting a single-tab :class:`LaviTabTable`.

    The tab strip is hidden so the list reads as a flat grouped tree.
    Subclasses feed groups via :meth:`set_groups` and listen to selection
    by connecting :attr:`script_selected`.
    """

    script_selected = pyqtSignal(str)

    def __init__(
        self,
        *,
        tab_id: str,
        tab_default: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tab_id = tab_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._groups: List[Dict] = []
        self._current_selection: str = ""

        self._table: LaviTabTable = setLaviTabTable(
            [{"id": tab_id, "default": tab_default, "groups": []}],
            kind="single",
            parent=self,
        )
        # Hide the tab strip — degenerate 1-tab case; LaviTabTable always
        # paints `_tab_bar`. Private-attr access acknowledged; follow-up
        # SDK request: add a public ``tab_bar_visible`` constructor flag.
        try:
            self._table._tab_bar.setVisible(False)
        except AttributeError:
            pass
        self._table.selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, 1)

        self._apply_transparent_bg()
        restyle_tab_table_for_sidebar(self._table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_groups(self, groups: List[Dict]) -> None:
        """Rebuild the LaviTabTable with new top-level groups."""
        self._groups = list(groups or [])
        spec = [{
            "id": self._tab_id,
            "default": self._tab_id,
            "groups": list(self._groups),
        }]
        # No public setter exists on LaviTabTable for groups-mode tabs, so
        # replace the widget entirely (same pattern as MissionControlPanel
        # ``_rebuild_files_table``).
        self._table.selectionChanged.disconnect(self._on_selection_changed)
        layout = self.layout()
        layout.removeWidget(self._table)
        self._table.setParent(None)
        self._table.deleteLater()

        self._table = setLaviTabTable(
            spec, kind="single", parent=self,
        )
        try:
            self._table._tab_bar.setVisible(False)
        except AttributeError:
            pass
        self._table.selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, 1)
        self._apply_transparent_bg()
        restyle_tab_table_for_sidebar(self._table)

        # Try to restore selection if the previously-selected id is still
        # present in the new groups.
        if self._current_selection:
            self._table.setSelection(self._tab_id, [self._current_selection])
            survivors = self._table.selectionMap().get(self._tab_id, [])
            if not survivors:
                self._current_selection = ""

    def set_current_selection(self, item_id: str) -> None:
        """Programmatically drive selection without emitting."""
        if not item_id:
            return
        self._table.setSelection(self._tab_id, [item_id])
        survivors = self._table.selectionMap().get(self._tab_id, [])
        self._current_selection = survivors[0] if survivors else ""

    def refresh_style(self) -> None:
        try:
            self._table.refresh_style()
        except Exception:
            pass
        self._apply_transparent_bg()
        # SDK refresh_style re-enables autofill on the list surface; re-strip
        # + re-tint group headers so theme switches stay consistent.
        restyle_tab_table_for_sidebar(self._table)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_transparent_bg(self) -> None:
        # The LaviTabTable reads the ``lavi_tab_table_bg`` slot for its
        # list surface; setting that to "transparent" in the theme would
        # disable autofill repo-wide. Locally enforce transparency on the
        # wrapper QWidget so the sidebar TitleGroupBox shows through.
        self.setStyleSheet("background: transparent;")

    def _on_selection_changed(self, _tab_id: str, items: list) -> None:
        item_id = items[0] if items else ""
        self._current_selection = item_id
        if item_id:
            self.script_selected.emit(item_id)


# ---------------------------------------------------------------------------
# Training scripts panel — drives off file_index.list_script_groups
# ---------------------------------------------------------------------------
class TrainingScriptsPanel(_SingleTabListPanel):
    """Lists project + global ``.py`` files under ``scripts/`` directories.

    Reuses :func:`list_script_groups` so the bucket layout (Project Script /
    Global Script) is identical to the legacy Script tab. Item ids are
    ``project:<rel>`` / ``system:<rel>`` and resolve through
    :func:`application.service.projects.resolve_file` unchanged.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            tab_id="sidebar.training.tab",
            tab_default="Training",
            parent=parent,
        )
        self._project_info: Optional[ProjectInfo] = None

    def set_project(self, info: Optional[ProjectInfo]) -> None:
        self._project_info = info
        if info is None:
            self.set_groups([])
            return
        try:
            groups = list_script_groups(info)
        except Exception as exc:
            log_warning(f"[scripts_panels] list_script_groups failed: {exc!r}")
            groups = []
        self.set_groups(groups)


# ---------------------------------------------------------------------------
# Registry preset panels — rewards / terminations / observations
# ---------------------------------------------------------------------------
_REGISTRY_TITLE = {
    "reward": "Rewards",
    "termination": "Termins",
    "observation": "Observs",
}

# Family → theme slot for the per-variant prefix badge. Families not in
# this map fall back to ``script_family_any``. Slots are defined in
# ``src/config/system.ini[Theme]``; do not hex-literal here (rule §1.5).
_FAMILY_SLOT: Dict[str, str] = {
    "quadruped": "script_family_quadruped",
    "humanoid": "script_family_humanoid",
    "biped": "script_family_biped",
    "wheeled": "script_family_wheeled",
    "manipulator": "script_family_manipulator",
}


def _family_color_slot(families) -> str:
    """First-known family → theme slot; empty / unknown → ``any`` slot."""
    if not families:
        return "script_family_any"
    for fam in families:
        if fam in _FAMILY_SLOT:
            return _FAMILY_SLOT[fam]
    return "script_family_any"


def _family_badge(families) -> str:
    """Short prefix badge text — ``[fam]`` or ``[any]`` for empty."""
    if not families:
        return "[any]"
    if len(families) == 1:
        return f"[{next(iter(families))}]"
    return "[" + ",".join(sorted(families)[:2]) + "+]"


class RegistryPresetPanel(_SingleTabListPanel):
    """Per-key fold-out of registry presets + user variants.

    Each function key is a *top-level* group on the LaviTabTable — there
    is no outer Rewards / Termins / Observs bucket header, because the
    sidebar rail already routes the user into one of those modes before
    the panel ever paints. Visual structure::

        action_rate_penalty                  ← top-level group (per key)
            [P]    Preset
            [quadruped] aggressive
            [humanoid]  slow
            +    + new variant               ← synthetic action item
        alive_reward
            [P]    Preset
            ...

    Item ids:

    * ``registry:<kind>:<key>:preset``      — factory preset row
    * ``registry:<kind>:<key>:<variant>``   — one user variant
    * ``registry:<kind>:<key>:__new__``     — synthetic; the panel
      intercepts this in :meth:`_on_selection_changed`, opens
      :class:`VariantCreateDialog`, and on Cancel reverts the visible
      selection back to whatever was active before the click.

    Refresh triggers:

    * :data:`AppSignals.current_backend_changed`   — engine swap
    * :data:`AppSignals.user_scripts_changed`      — variant added /
      deleted (only re-renders when the change touches this kind)
    """

    def __init__(
        self, *, kind: str, parent: Optional[QWidget] = None
    ) -> None:
        title = _REGISTRY_TITLE.get(kind, kind.title())
        super().__init__(
            tab_id=f"sidebar.{kind}.tab",
            tab_default=title,
            parent=parent,
        )
        self._kind = kind
        self._backend: str = ""
        # Track the last *real* selection (non-``__new__``) so we can
        # revert when the user cancels the variant-create dialog.
        self._last_real_selection: str = ""
        # Re-entrancy guard for programmatic ``setSelection`` — the SDK
        # re-emits ``selectionChanged`` for programmatic mutations, and
        # without this guard ``_revert_to_last_real_selection`` would
        # ping-pong through ``_on_selection_changed``.
        self._suppress_selection_signal: bool = False
        get_app_signals().current_backend_changed.connect(self.set_backend)
        get_app_signals().user_scripts_changed.connect(self._on_variants_changed)
        # Cold-start paint — pick up the backend that may already be set
        # from a canvas opened before this panel was lazy-built.
        self.set_backend(current_backend())

    # ── refresh hooks ──────────────────────────────────────────────

    def set_backend(self, backend: str) -> None:
        self._backend = backend or ""
        self._rebuild()

    def _on_variants_changed(self, kind: str, _key: str) -> None:
        if kind == self._kind:
            self._rebuild()

    def _rebuild(self) -> None:
        groups = self._build_groups(self._kind, self._backend)
        self.set_groups(groups)

    # ── group spec construction ────────────────────────────────────

    def _build_groups(self, kind: str, backend: str):
        """Build the flat per-key group spec for this panel.

        The outer Rewards/Termins/Observs bucket header was removed —
        each function key becomes a directly-foldable top-level group.
        Returns an empty list when the backend has no presets and no
        user-only keys (the table just stays empty).
        """
        # Local import avoids a startup-time import cycle:
        # scripts.* loads heavy registries; we lazy-load on panel render.
        try:
            from application.service.scripts import resolver
            from scripts.query import lookup as preset_lookup
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[scripts_panels] resolver import failed: {exc!r}"
            )
            return []

        be = resolver.map_engine_to_backend(backend)
        keys = resolver.list_keys(kind, backend=backend)
        groups = []
        for key in keys:
            preset_item = preset_lookup(key, kind=kind, backend=be)
            title = getattr(preset_item, "title", None) or key
            row_items = []
            # Preset row (only emitted when a preset exists for this
            # backend — purely user-defined keys still show variants).
            if preset_item is not None:
                row_items.append({
                    "id": f"registry:{kind}:{key}:preset",
                    "default": "Preset",
                    "prefix": "[P]",
                    "prefix_color": "script_family_preset",
                })
            for v in resolver.list_variants(kind, key):
                row_items.append({
                    "id": f"registry:{kind}:{key}:{v.name}",
                    "default": v.name,
                    "prefix": _family_badge(v.families),
                    "prefix_color": _family_color_slot(v.families),
                })
            # Synthetic create-row — always present so the user can grow
            # the list without leaving the sidebar.
            row_items.append({
                "id": f"registry:{kind}:{key}:__new__",
                "default": "+ new variant",
                "prefix": "+",
                "prefix_color": "highlight",
            })
            groups.append({
                "id": f"scripts.{kind}.key.{key}",
                "default": title,
                "expanded": False,
                "items": row_items,
            })
        return groups

    # ── selection handling — intercept ``__new__`` sentinel ─────────

    def _on_selection_changed(self, _tab_id: str, items: list) -> None:
        if self._suppress_selection_signal:
            return
        item_id = items[0] if items else ""
        # ``__new__`` ⇒ open the create dialog and decide whether to
        # promote the selection (accept) or revert (cancel) — bypasses
        # the normal ``script_selected`` emit so the editor doesn't
        # flicker into a non-existent virtual id.
        if item_id.endswith(":__new__"):
            self._handle_new_variant_click(item_id)
            return
        # Normal row — record as the last real selection so the next
        # ``__new__`` click can revert here if cancelled.
        self._current_selection = item_id
        if item_id:
            self._last_real_selection = item_id
            self.script_selected.emit(item_id)

    def _handle_new_variant_click(self, virtual_id: str) -> None:
        """Open :class:`VariantCreateDialog`; revert selection on cancel."""
        parts = virtual_id.split(":", 3)
        if len(parts) != 4 or parts[0] != "registry":
            return
        kind, key = parts[1], parts[2]
        try:
            from application.ui.dialogs.variant_create_dialog import (
                VariantCreateDialog,
            )
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[scripts_panels] VariantCreateDialog import: {exc!r}"
            )
            self._revert_to_last_real_selection()
            return
        dlg = VariantCreateDialog(kind=kind, key=key, parent=self.window())

        def _on_accepted(k: str, ky: str, vname: str) -> None:
            # After accept, the resolver fires user_scripts_changed →
            # this panel rebuilds; we then drive the new variant as
            # the active selection so the editor swaps to it.
            new_id = f"registry:{k}:{ky}:{vname}"
            self._last_real_selection = new_id
            self.set_current_selection(new_id)
            self.script_selected.emit(new_id)

        dlg.accepted_variant.connect(_on_accepted)
        result = dlg.exec()
        if not result:
            # Cancelled / closed without accept → restore prior row.
            self._revert_to_last_real_selection()

    def _revert_to_last_real_selection(self) -> None:
        """Move the visible row indicator back to the last real selection.

        Programmatic ``setSelection`` re-emits ``selectionChanged``; the
        :attr:`_suppress_selection_signal` guard short-circuits the
        re-entrant call so the revert is purely visual.
        """
        self._suppress_selection_signal = True
        try:
            if self._last_real_selection:
                self.set_current_selection(self._last_real_selection)
            else:
                # No prior selection to revert to — clear any sticky state.
                try:
                    self._table.setSelection(self._tab_id, [])
                except Exception:                                 # noqa: BLE001
                    pass
                self._current_selection = ""
        finally:
            self._suppress_selection_signal = False


# Legacy name retained for callers that imported the panel's old
# backend→registry-dict helper. Routes through the new resolver so its
# behaviour stays consistent with what the panel shows.
def _resolve_registry(kind: str, backend: str) -> Dict[str, object]:
    """Deprecated: prefer ``application.service.scripts.resolver``. Returns
    the preset dict for ``(kind, backend)``; user variants are ignored."""
    try:
        from application.service.scripts import resolver
        from scripts.query import lookup as preset_lookup
    except Exception as exc:                                      # noqa: BLE001
        log_warning(
            f"[scripts_panels] legacy _resolve_registry import "
            f"failed: {exc!r}"
        )
        return {}
    be = resolver.map_engine_to_backend(backend)
    out: Dict[str, object] = {}
    for key in resolver.list_keys(kind, backend=backend):
        item = preset_lookup(key, kind=kind, backend=be)
        if item is not None:
            out[key] = item
    return out


__all__ = [
    "TrainingScriptsPanel",
    "RegistryPresetPanel",
]
