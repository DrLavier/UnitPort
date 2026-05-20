"""Variant resolver — preset + user-variant lookup for training scripts.

See :mod:`application.service.scripts` for the package overview.

Public surface (re-exported from the package):

- :class:`ResolvedScript`, :class:`VariantMeta`
- :func:`resolve` / :func:`list_variants` / :func:`list_keys`
- :func:`save_variant` / :func:`delete_variant`
- :func:`family_filter` / :func:`map_engine_to_backend`

All on-disk I/O routes through :class:`unitport_sdk.DataManager` and
:class:`unitport_sdk.Paths`. Variants live under two search roots,
queried in this order — user overrides system:

  1. ``Paths.USER_CONFIG_DIR / scripts / <kind_dir> / <key>`` — user
     overlay; writable via :func:`save_variant` / :func:`delete_variant`.
  2. ``Paths.SRC_ROOT / scripts / <kind_dir> / <key>`` — system factory
     variants shipped with the SDK; read-only, never written from
     Python code. Used to deliver per-family overlays (biped /
     humanoid / quadruped) without forcing first-run scaffolding.

Each root may contain:

- ``<variant>.py``  — Python source (text I/O via :func:`save_data` with
                       format=``.txt``); the resolver compile-checks this
                       source on write to fail loud on bad input.
- ``variants.toml`` — metadata for every variant under this key. Schema::

      [variant.<name>]
      families = ["quadruped"]      # [] = any
      description = "..."
      based_on = "preset"           # or another variant name
      created_at = "<ISO-Z>"
      last_modified_at = "<ISO-Z>"
      il_params_override = "..."    # optional, per-variant override of
                                     # the preset's ``il_params`` template
                                     # (e.g. swap the sensor body regex).

The factory presets (``scripts.<kind>s.registry.*_REGISTRY``) remain
single-source-of-truth for the **default** behaviour. System variants
under ``src/scripts/`` and user variants under ``USER_CONFIG_DIR`` can
shadow that default for specific ``(robot_family, variant_name)``
combinations; ``family_filter`` enforces the family scope at
``resolve()`` time when ``robot_sku`` is passed. ``resolve(variant=None
|"preset")`` still reads the preset directly via
:func:`scripts.query.lookup` — no variant can override the preset path.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from unitport_sdk import (
    DataManager,
    Paths,
    load_data,
    log_warning,
    save_data,
)

from scripts.query import lookup as _preset_lookup
from scripts.query import query_registry as _preset_query
from scripts.task_module import TaskModuleItem

# ─── Kind / dir / backend mappings ────────────────────────────────────

# Singular kind → plural directory name (matches ``src/scripts/<plural>/``).
_KIND_TO_DIR: Dict[str, str] = {
    "reward": "rewards",
    "termination": "terminations",
    "observation": "observations",
    "discriminator": "discriminator",
}
_VALID_KINDS = frozenset(_KIND_TO_DIR.keys())

# Engine ID (from ``current_backend()``) → ``TaskModuleItem.backends`` tag.
# Engine IDs live in ``registers.backends``; backend tags in
# ``scripts.task_module``. The two namespaces are deliberately separate;
# this map is the only sanctioned translation. Identity entries (``sb3``,
# ``isaac_lab``) let callers that already hold the kind-form string
# (e.g. ``spec_compiler``'s ``canvas_backend_kind``) feed it in
# directly without an extra branch.
_ENGINE_TO_BACKEND: Dict[str, str] = {
    "sb3_mujoco": "sb3",
    "isaac_lab": "isaac_lab",
    "newton": "newton",
    "sb3": "sb3",
}


def map_engine_to_backend(engine_id: Optional[str]) -> Optional[str]:
    """Translate a UI/engine ID to the backend tag used inside
    :class:`TaskModuleItem.backends`. Unknown / empty → ``None``."""
    if not engine_id:
        return None
    return _ENGINE_TO_BACKEND.get(engine_id)


# ─── Dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class VariantMeta:
    """One entry from ``variants.toml``'s ``[variant.<name>]`` table."""

    name: str
    families: FrozenSet[str]
    description: str
    based_on: str
    created_at: str
    last_modified_at: str
    # Optional per-variant override of the preset's ``il_params`` template
    # (e.g. swap ``body_names={ir:feet}`` for ``body_names={ir:ankles}``
    # on a biped variant of ``feet_air_time``). When None, the env_cfg
    # compiler falls back to the preset's il_params.
    il_params_override: Optional[str] = None
    # Provenance — which search root the variant was found in. Set at
    # parse time; not stored in variants.toml. Used by callers that need
    # to distinguish "shipped with SDK" vs "user authored".
    origin_root: str = "user"  # "user" | "system"


@dataclass(frozen=True)
class ResolvedScript:
    """Resolved Python source + provenance for one ``(kind, key, variant)``."""

    kind: str
    key: str
    variant: Optional[str]
    source: str
    item: Optional[TaskModuleItem]
    # "preset" (no variant) | "user_variant" (USER_CONFIG_DIR overlay)
    # | "system_variant" (factory variant under SRC_ROOT/scripts)
    origin: str
    families: FrozenSet[str]
    description: str
    # Forwarded from the variant's ``variants.toml`` entry. None for
    # preset path or when the variant didn't declare an override.
    # ``env_cfg_compiler`` reads this to splice a per-variant
    # ``il_params`` template into the emitted RewTerm/DoneTerm.
    il_params_override: Optional[str] = None


# ─── Path helpers ─────────────────────────────────────────────────────

def _user_scripts_root(kind: str, key: str) -> Optional[Path]:
    """User overlay root. ``None`` when ``USER_CONFIG_DIR`` is not yet
    configured (pre-wizard boot) — caller treats this as "no user
    variants present" rather than raising."""
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"unknown kind: {kind!r}; expected one of {sorted(_VALID_KINDS)}"
        )
    if not Paths.is_user_config_dir_set():
        return None
    return Paths.USER_CONFIG_DIR / "scripts" / _KIND_TO_DIR[kind] / key


def _system_scripts_root(kind: str, key: str) -> Path:
    """System (factory) variant root — always available, ``__file__``
    derived, no wizard dependency. Read-only at runtime; lives in the
    repo under ``src/scripts/<kind_dir>/<key>/``."""
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"unknown kind: {kind!r}; expected one of {sorted(_VALID_KINDS)}"
        )
    return Paths.SRC_ROOT / "scripts" / _KIND_TO_DIR[kind] / key


def _iter_variant_roots(kind: str, key: str) -> List[Tuple[str, Path]]:
    """Search roots in priority order (user first, system fallback).

    Returns ``[(origin_tag, root_path), ...]``. ``origin_tag`` is
    ``"user"`` or ``"system"`` — propagates to :class:`VariantMeta`'s
    ``origin_root`` so the UI can render shipped vs user-authored
    variants differently. User entries override system entries with
    the same variant name.
    """
    out: List[Tuple[str, Path]] = []
    user = _user_scripts_root(kind, key)
    if user is not None:
        out.append(("user", user))
    out.append(("system", _system_scripts_root(kind, key)))
    return out


def _variants_toml_path(kind: str, key: str) -> Optional[Path]:
    """Path of the *user* variants.toml (write target). ``None`` when
    USER_CONFIG_DIR is not set — ``save_variant`` raises in that case
    so the user knows their write would be lost. Reads consult both
    user and system tomls via :func:`_iter_variant_roots`."""
    root = _user_scripts_root(kind, key)
    return root / "variants.toml" if root is not None else None


def _variant_py_path(kind: str, key: str, variant: str) -> Optional[Path]:
    """Path of the *user* variant .py (write target). See
    :func:`_variants_toml_path` for the None semantics."""
    root = _user_scripts_root(kind, key)
    return root / f"{variant}.py" if root is not None else None


# ─── Family filter ────────────────────────────────────────────────────

def family_filter(
    variant_families: FrozenSet[str],
    robot_families: FrozenSet[str],
) -> bool:
    """True iff this variant is applicable to a robot with the given families.

    Rules:
        - ``variant_families`` empty → "any robot" → True
        - ``robot_families`` empty   → caller does not know the robot, accept
        - else                       → non-empty intersection required
    """
    if not variant_families:
        return True
    if not robot_families:
        return True
    return bool(variant_families & robot_families)


# ─── Robot family lookup ──────────────────────────────────────────────

def _robot_families(robot_sku: str) -> FrozenSet[str]:
    """Look up ``RobotSpec.families`` for a SKU; empty on miss."""
    if not robot_sku:
        return frozenset()
    try:
        from registers.robots import get_robot_spec
    except Exception:                                             # noqa: BLE001
        return frozenset()
    try:
        spec = get_robot_spec(robot_sku)
    except Exception:                                             # noqa: BLE001
        return frozenset()
    if spec is None:
        return frozenset()
    fams = getattr(spec, "families", ()) or ()
    return frozenset(str(f) for f in fams)


# ─── Reading variants (user + system) ─────────────────────────────────

def _read_one_variants_toml(toml_path: Path) -> Dict[str, Dict]:
    """Read the ``variant.<name>`` table from a single ``variants.toml`` file
    or ``{}`` if the file is missing / malformed."""
    if not toml_path.exists():
        return {}
    try:
        data = load_data(toml_path, force_reload=True) or {}
    except Exception as exc:                                      # noqa: BLE001
        log_warning(
            f"[scripts.resolver] variants.toml parse failed: "
            f"{toml_path}: {exc!r}"
        )
        return {}
    raw = data.get("variant", {}) if isinstance(data, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _read_variants_toml(kind: str, key: str) -> List[Tuple[str, str, Dict]]:
    """All ``variant.<name>`` entries from all search roots.

    Returns ``[(origin_tag, variant_name, raw_meta_dict), ...]`` in
    search-root priority order (user first, system second). Same-named
    variants from different roots both appear — caller decides priority
    (typically: keep first occurrence per name). Skips entries whose
    matching ``<variant>.py`` source file is missing.
    """
    out: List[Tuple[str, str, Dict]] = []
    for origin_tag, root in _iter_variant_roots(kind, key):
        toml_path = root / "variants.toml"
        for name, raw in _read_one_variants_toml(toml_path).items():
            if not isinstance(raw, dict):
                continue
            if not (root / f"{name}.py").exists():
                continue  # orphan; log at list_variants() level
            out.append((origin_tag, name, raw))
    return out


def _parse_meta(name: str, raw: Dict, origin_root: str = "user") -> VariantMeta:
    families_list = raw.get("families") or []
    if not isinstance(families_list, list):
        families_list = []
    il_params = raw.get("il_params_override")
    il_params = str(il_params) if isinstance(il_params, str) and il_params else None
    return VariantMeta(
        name=name,
        families=frozenset(
            str(f) for f in families_list if isinstance(f, str)
        ),
        description=str(raw.get("description") or ""),
        based_on=str(raw.get("based_on") or "preset"),
        created_at=str(raw.get("created_at") or ""),
        last_modified_at=str(raw.get("last_modified_at") or ""),
        il_params_override=il_params,
        origin_root=origin_root,
    )


def list_variants(kind: str, key: str) -> List[VariantMeta]:
    """All variants (user + system) under ``(kind, key)`` with ``.py``
    source files present. User overlay wins on name collision.

    Entries in ``variants.toml`` without a corresponding ``.py`` are
    dropped with a warning at read time. Result is alphabetically
    sorted by variant name.
    """
    seen: Dict[str, VariantMeta] = {}
    # _iter_variant_roots is user-first, so the first occurrence wins.
    for origin_tag, root in _iter_variant_roots(kind, key):
        toml_path = root / "variants.toml"
        raw_table = _read_one_variants_toml(toml_path)
        for name in sorted(raw_table.keys()):
            if name in seen:
                continue
            if not isinstance(raw_table[name], dict):
                continue
            if not (root / f"{name}.py").exists():
                log_warning(
                    f"[scripts.resolver] orphaned variant "
                    f"{kind}/{key}/{name} ({origin_tag}) — meta present "
                    f"but source file missing; skipped"
                )
                continue
            seen[name] = _parse_meta(name, raw_table[name], origin_root=origin_tag)
    return [seen[name] for name in sorted(seen)]


def _list_variant_keys(kind: str) -> List[str]:
    """Directory listing across user + system roots.

    A key is exposed if at least one search root contains
    ``<kind_dir>/<key>/variants.toml``. User and system contributions
    union together.
    """
    if kind not in _VALID_KINDS:
        return []
    keys: Set[str] = set()
    roots: List[Path] = []
    if Paths.is_user_config_dir_set():
        roots.append(Paths.USER_CONFIG_DIR / "scripts" / _KIND_TO_DIR[kind])
    roots.append(Paths.SRC_ROOT / "scripts" / _KIND_TO_DIR[kind])
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / "variants.toml").exists():
                keys.add(child.name)
    return sorted(keys)


def list_keys(kind: str, backend: Optional[str] = None) -> List[str]:
    """Union of preset keys (filtered by backend) and variant keys
    (user + system). ``backend`` is the engine-ID form (translated to
    :class:`TaskModuleItem.backends` tag via
    :func:`map_engine_to_backend` before preset filtering)."""
    be = map_engine_to_backend(backend) if backend else None
    preset_keys = set(_preset_query(kind=kind, backend=be).keys())
    variant_keys = set(_list_variant_keys(kind))
    return sorted(preset_keys | variant_keys)


# ─── Source resolution ────────────────────────────────────────────────

def _synthesize_preset_stub(kind: str, key: str, item: TaskModuleItem) -> str:
    """Build an editable stub when ``item.il_inline`` is empty.

    Replicates the inline-stub formatting that used to live in
    ``mission_control_panel._load_registry_virtual_into_editor`` so
    "clone-from-preset" gets the exact source the editor showed.
    """
    return (
        f'"""{item.title}\n\n{item.desc}\n"""\n\n'
        f"# Default: {item.default}  Range: "
        f"[{item.min_value}, {item.max_value}]  Step: {item.step}\n"
        f"def {key}():\n"
        f"    return {item.default!r}\n"
    )


def _preset_source(kind: str, key: str, item: TaskModuleItem) -> str:
    if item.il_inline:
        return item.il_inline
    return _synthesize_preset_stub(kind, key, item)


def _read_variant_source(
    kind: str, key: str, variant: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Search user → system for the variant's ``.py`` source.

    Returns ``(source, origin_tag)`` where ``origin_tag`` is ``"user"``
    or ``"system"``; both are ``None`` when the variant doesn't exist
    in either root.
    """
    for origin_tag, root in _iter_variant_roots(kind, key):
        py_path = root / f"{variant}.py"
        if not py_path.exists():
            continue
        # ``.py`` is not in DataManager's handler dispatch table; use the
        # text handler explicitly so we get a ``str`` back (handlers
        # default to binary for unknown extensions).
        try:
            data = load_data(py_path, force_reload=True, format=".txt")
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[scripts.resolver] variant source read failed: "
                f"{py_path}: {exc!r}"
            )
            continue
        if data is None:
            return "", origin_tag
        if isinstance(data, bytes):
            # Defensive — TextHandler should have decoded already.
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                data = data.decode("utf-8", errors="replace")
        # Normalise CRLF → LF so source body compares cleanly across
        # platforms (Windows newlines survive the atomic temp+replace).
        return str(data).replace("\r\n", "\n"), origin_tag
    return None, None


def _read_variant_meta(
    kind: str, key: str, variant: str,
) -> Tuple[Optional[VariantMeta], Optional[str]]:
    """Search user → system for the variant's ``variants.toml`` entry.

    Returns ``(meta, origin_tag)`` paired the same way as
    :func:`_read_variant_source`. ``meta`` is None when no root carries
    the entry (caller should still attempt source lookup — orphan
    sources without metadata are still treated as preset-default).
    """
    for origin_tag, root in _iter_variant_roots(kind, key):
        raw_table = _read_one_variants_toml(root / "variants.toml")
        if variant in raw_table and isinstance(raw_table[variant], dict):
            return _parse_meta(variant, raw_table[variant], origin_tag), origin_tag
    return None, None


def resolve(
    kind: str,
    key: str,
    *,
    variant: Optional[str] = None,
    robot_sku: Optional[str] = None,
    backend: Optional[str] = None,
) -> Optional[ResolvedScript]:
    """Resolve ``(kind, key, variant)`` to runnable Python source + meta.

    Cases:

    - ``variant in (None, "", "preset")`` → preset source (``il_inline``
      or stub if empty).
    - else → user variant source under ``USER_CONFIG_DIR``; missing →
      returns ``None`` so caller can decide whether to fall back to preset.

    Family filter:
        When ``robot_sku`` is provided, the resolved variant's families
        must intersect the robot's families per :func:`family_filter`.
        Mismatch → returns ``None``. (Caller can re-call with
        ``variant=None`` to grab the preset fallback.)

    ``backend`` is the engine ID ("sb3_mujoco" / "isaac_lab"); used to
    disambiguate within-kind preset collisions (e.g. SB3 vs IL variant
    of the same key).
    """
    if kind not in _VALID_KINDS:
        log_warning(f"[scripts.resolver] unknown kind: {kind!r}")
        return None

    be = map_engine_to_backend(backend) if backend else None
    item = _preset_lookup(key, kind=kind, backend=be)

    # Preset path
    if variant in (None, "", "preset"):
        if item is None:
            return None
        return ResolvedScript(
            kind=kind,
            key=key,
            variant=None,
            source=_preset_source(kind, key, item),
            item=item,
            origin="preset",
            families=frozenset(item.applicable_families),
            description=item.desc,
        )

    # User/system variant path — both roots consulted, user first.
    meta, meta_origin = _read_variant_meta(kind, key, variant)
    source, src_origin = _read_variant_source(kind, key, variant)
    if source is None or meta is None:
        return None
    # Source + meta may legitimately come from different roots when a
    # user overlay edits only the meta block but leaves the .py alone
    # (or vice versa). Trust the meta's origin tag for the
    # ``origin`` field; the .py search was source-of-truth for the
    # actual Python body.

    if robot_sku:
        if not family_filter(meta.families, _robot_families(robot_sku)):
            return None

    origin_label = "user_variant" if meta_origin == "user" else "system_variant"
    return ResolvedScript(
        kind=kind,
        key=key,
        variant=variant,
        source=source,
        item=item,
        origin=origin_label,
        families=meta.families,
        description=meta.description or (item.desc if item else ""),
        il_params_override=meta.il_params_override,
    )


# ─── Mutations ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_changed(kind: str, key: str) -> None:
    """Fire ``AppSignals.user_scripts_changed(kind, key)`` if available.

    During unit tests / early bootstrap the signal bus may not be wired
    yet — silently no-op rather than fail the save.
    """
    try:
        from application.service.signals import get_app_signals

        emitter = getattr(get_app_signals(), "user_scripts_changed", None)
        if emitter is not None:
            emitter.emit(kind, key)
    except Exception:                                             # noqa: BLE001
        pass


def _is_safe_name(name: str) -> bool:
    """A safe filename token: Python-identifier-shaped, no path traversal."""
    return bool(name) and name.isidentifier()


def save_variant(
    kind: str,
    key: str,
    variant: str,
    source: str,
    *,
    families: Optional[List[str]] = None,
    description: str = "",
    based_on: str = "preset",
    il_params_override: Optional[str] = None,
) -> Path:
    """Persist a user variant to ``<USER_CONFIG_DIR>/scripts/<kind>/<key>/``.

    Steps:

    1. ``compile(source, ...)`` — surface ``SyntaxError`` before writing,
       wrapped in :class:`ValueError` so callers don't need to import the
       Python builtin.
    2. Write ``<variant>.py`` via :func:`save_data` (atomic temp+replace
       courtesy of :class:`DataManager`).
    3. Read+merge ``variants.toml`` — preserves other variants under the
       same key. ``created_at`` is set on first save; ``last_modified_at``
       updates each time.
    4. Emit :data:`AppSignals.user_scripts_changed` for the sidebar to
       refresh.

    Returns the ``.py`` path on success. Raises ``ValueError`` for bad
    input or syntax errors; ``OSError`` if either write fails.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown kind: {kind!r}")
    if not _is_safe_name(key):
        raise ValueError(f"bad key: {key!r} (must be identifier-shaped)")
    if variant == "preset" or not _is_safe_name(variant):
        raise ValueError(f"bad variant name: {variant!r}")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty Python string")

    try:
        compile(source, f"<variant:{kind}/{key}/{variant}>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"variant source has SyntaxError: {exc}") from exc

    py_path = _variant_py_path(kind, key, variant)
    toml_path = _variants_toml_path(kind, key)
    if py_path is None or toml_path is None:
        # USER_CONFIG_DIR not yet established by the install wizard —
        # CLAUDE.md §1.4 forbids the previous fallback to a default
        # location (would silently produce a write the rest of the app
        # never reads back from).
        Paths.require_user_config_dir()  # raises RuntimeError

    if not save_data(py_path, source, format=".txt"):
        raise OSError(f"failed to write variant source: {py_path}")

    # Read+merge: preserve sibling variants under the same key.
    if toml_path.exists():
        try:
            all_meta = dict(load_data(toml_path, force_reload=True) or {})
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[scripts.resolver] could not read existing variants.toml "
                f"(treating as empty): {toml_path}: {exc!r}"
            )
            all_meta = {}
    else:
        all_meta = {}

    variants_table = dict(all_meta.get("variant") or {})
    prior = variants_table.get(variant) or {}
    now = _now_iso()
    fam_list = sorted(set(families)) if families else []
    entry: Dict[str, object] = {
        "families": fam_list,
        "description": description,
        "based_on": based_on or "preset",
        "created_at": str(prior.get("created_at") or now),
        "last_modified_at": now,
    }
    # Only emit il_params_override when non-empty so legacy variants
    # (and freshly cloned-from-preset variants) keep their TOML clean.
    if il_params_override:
        entry["il_params_override"] = str(il_params_override)
    variants_table[variant] = entry
    all_meta["variant"] = variants_table

    if not save_data(toml_path, all_meta, format=".toml"):
        raise OSError(f"failed to write variants.toml: {toml_path}")

    _emit_changed(kind, key)
    return py_path


def delete_variant(kind: str, key: str, variant: str) -> bool:
    """Remove a user variant's source + its entry in ``variants.toml``.

    Returns True iff at least one of source / meta entry existed and was
    removed. The empty key directory is left in place — cheap on disk
    and avoids races with concurrent reads.
    """
    if kind not in _VALID_KINDS or not variant or variant == "preset":
        return False
    if not _is_safe_name(variant):
        return False

    py_path = _variant_py_path(kind, key, variant)
    toml_path = _variants_toml_path(kind, key)
    if py_path is None or toml_path is None:
        # USER_CONFIG_DIR unset → no user variants exist to delete;
        # system variants are read-only and never deleted from here.
        return False
    removed_any = False

    if py_path.exists():
        try:
            py_path.unlink()
            DataManager.clear(py_path)
            removed_any = True
        except OSError as exc:
            log_warning(
                f"[scripts.resolver] failed to unlink {py_path}: {exc!r}"
            )

    if toml_path.exists():
        try:
            all_meta = dict(load_data(toml_path, force_reload=True) or {})
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[scripts.resolver] could not read variants.toml during "
                f"delete: {toml_path}: {exc!r}"
            )
            all_meta = {}
        variants_table = dict(all_meta.get("variant") or {})
        if variant in variants_table:
            del variants_table[variant]
            removed_any = True
            if variants_table:
                all_meta["variant"] = variants_table
                save_data(toml_path, all_meta, format=".toml")
            else:
                # Last variant gone — drop the file rather than leaving an
                # empty ``[variant]`` table behind.
                try:
                    toml_path.unlink()
                    DataManager.clear(toml_path)
                except OSError as exc:
                    log_warning(
                        f"[scripts.resolver] failed to unlink "
                        f"{toml_path}: {exc!r}"
                    )

    if removed_any:
        _emit_changed(kind, key)
    return removed_any
