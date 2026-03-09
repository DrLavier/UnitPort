"""Mission save/load helpers and schema validation.

STAGE-05: Mission Persistence and Snapshot Flow
-----------------------------------------------
Pure-Python module (no Qt dependency) so all helpers are fully
unit-testable without a display.

Public API
----------
validate_mission_schema(data) -> (ok: bool, reason: str)
inject_snapshot_metadata(data) -> data   (mutates + returns)
migrate_mission_payload(data) -> dict    (migration info; does not mutate data)

Constants
---------
MISSION_SCHEMA_VERSION  — current schema version string
UNITPORT_VERSION        — current application version tag
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Version constants ─────────────────────────────────────────────────────────

MISSION_SCHEMA_VERSION: str = "1.4"   # Step 5: protocol sub_dot contract; behavior_in_port type="protocol"
UNITPORT_VERSION: str = "cycle3"

# Required top-level keys in every loadable mission dict.
# Extra keys are always tolerated (forward compat).
_REQUIRED_KEYS: Tuple[str, ...] = ("nodes", "connections")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_mission_schema(data: Any) -> Tuple[bool, str]:
    """Check that *data* is a loadable mission dict.

    Returns
    -------
    (True, "")
        Valid — safe to pass to ``GraphScene.load_workflow()``.
    (False, reason_str)
        Invalid — *reason_str* is a user-readable explanation.

    Design notes
    ------------
    - Missing ``schema_version`` is **not** an error; pre-STAGE-05 files
      omit it and must remain loadable (backward compatibility).
    - Extra unknown keys are tolerated for forward compatibility.
    - ``nodes`` must be a ``list``, not a ``dict``; a dict signals an
      exec-graph payload (wrong format) rather than a saved mission.
    """
    if not isinstance(data, dict):
        return False, "Mission file is not a JSON object."

    for key in _REQUIRED_KEYS:
        if key not in data:
            return False, f"Missing required key: '{key}'."
        if not isinstance(data[key], list):
            return (
                False,
                f"Key '{key}' must be a list, got {type(data[key]).__name__}.",
            )

    return True, ""


# ── Metadata injection ────────────────────────────────────────────────────────

def inject_snapshot_metadata(
    data: Dict[str, Any],
    source: str = "mission_editor_ui",
) -> Dict[str, Any]:
    """Stamp *data* with lightweight snapshot metadata.

    Mutates *data* in-place **and** returns it so callers can chain:

        saved = inject_snapshot_metadata(scene.serialize_workflow())

    Added keys
    ----------
    schema_version   : str  — MISSION_SCHEMA_VERSION
    unitport_version : str  — UNITPORT_VERSION
    saved_at         : str  — UTC ISO-8601 timestamp ending with "Z"
    source           : str  — origin context that triggered the save
                              (default: ``"mission_editor_ui"``; callers may
                              pass e.g. ``"test_harness"`` or ``"cli_export"``
                              to distinguish save origins in log/audit trails)
    """
    data["schema_version"] = MISSION_SCHEMA_VERSION
    data["unitport_version"] = UNITPORT_VERSION
    data["saved_at"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "Z"
    data["source"] = source
    return data


# ── Settings payload helpers (Cycle 3 STAGE-02) ───────────────────────────────


def build_settings_payload(
    brand: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical ``"settings"`` payload for mission files.

    Returns
    -------
    ``{"brand": brand, "config": {...}}``

    The returned dict is a shallow copy of *config* so callers cannot
    accidentally mutate the saved payload after this call.
    """
    return {
        "brand": brand,
        "config": dict(config) if config else {},
    }


def build_scenario_payload(
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical ``"scenario_settings"`` payload for mission files.

    Returns a shallow copy of *settings* so callers cannot accidentally mutate
    the saved payload after this call.

    Added by Cycle 3 STAGE-06 to persist advanced MuJoCo / scenario settings
    alongside SDK settings in the mission file.
    """
    return dict(settings) if settings else {}


def extract_scenario_payload(
    data: Any,
) -> Optional[Dict[str, Any]]:
    """Extract scenario/MuJoCo settings from a loaded mission dict.

    Returns
    -------
    ``dict``
        When a valid ``"scenario_settings"`` block is present.
    ``None``
        When the key is absent or the block is malformed.

    Design notes
    ------------
    - Missing ``"scenario_settings"`` (pre-Cycle-3-STAGE-06 files) → ``None``.
    - Present but not a dict → ``None``; never raises so callers can fall
      through to a silent no-restore path.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("scenario_settings")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def extract_settings_payload(
    data: Any,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract SDK settings from a loaded mission dict.

    Returns
    -------
    ``(brand, config)``
        When a valid ``"settings"`` block is present.
    ``None``
        When the ``"settings"`` key is absent or the block is malformed.

    Design notes
    ------------
    - Missing ``"settings"`` key (pre-Cycle-3 files) → ``None`` — no error.
    - ``"settings"`` present but malformed → ``None`` — no exception raised,
      so callers can always fall through to a silent no-restore path.
    - Missing ``"config"`` inside a valid block → treated as empty dict so
      the brand is still used to rebuild the form with default values.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("settings")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    brand = raw.get("brand")
    if not isinstance(brand, str) or not brand.strip():
        return None
    config = raw.get("config")
    if not isinstance(config, dict):
        config = {}
    return brand, config


# ── Behavior drafts payload helpers (Circle 1 Step 1.6) ──────────────────────


def build_behavior_drafts_payload(
    drafts: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical ``"behavior_drafts"`` payload for mission files.

    Parameters
    ----------
    drafts : Dict[str, Any]
        Per-node draft state from ``BehaviorPanel.get_behavior_drafts_state()``.
        Keys are node IDs (int or str); values are raw draft dicts
        ``{"core": str, "hb": {...}}``.

    Returns
    -------
    A shallow copy with all keys stringified (JSON requires string keys).
    Never raises.
    """
    out: Dict[str, Any] = {}
    for k, v in drafts.items():
        try:
            out[str(k)] = dict(v)
        except Exception:  # noqa: BLE001
            pass
    return out


def extract_behavior_drafts_payload(
    data: Any,
) -> Optional[Dict[str, Any]]:
    """Extract per-node behavior/heartbeat drafts from a loaded mission dict.

    Returns
    -------
    ``Dict[str, Any]``
        When a valid ``"behavior_drafts"`` block is present.
        Keys are node-ID strings; values are raw draft dicts.
    ``None``
        When the key is absent or the block is not a dict.

    Design notes
    ------------
    - Missing key (pre-Step-1.6 files) → ``None`` — silent no-restore.
    - Present but not a dict → ``None`` — never raises.
    - Individual malformed entries are silently skipped by the caller.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("behavior_drafts")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


# ── Behavior timeline payload helpers (Phase 1 redesign) ─────────────────────


def build_behavior_timeline_payload(
    timelines: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical ``"behavior_timelines"`` payload for mission files.

    Parameters
    ----------
    timelines : Dict[str, Any]
        Per-node structured timeline state.
        Keys are node IDs (int or str); values are BehaviorTimeline.to_dict()
        plain dicts.

    Returns
    -------
    A shallow copy with all keys stringified (JSON requires string keys).
    Never raises.
    """
    out: Dict[str, Any] = {}
    for k, v in timelines.items():
        try:
            out[str(k)] = dict(v)
        except Exception:  # noqa: BLE001
            pass
    return out


def extract_behavior_timeline_payload(
    data: Any,
) -> Optional[Dict[str, Any]]:
    """Extract per-node BehaviorTimeline data from a loaded mission dict.

    Returns
    -------
    ``Dict[str, Any]``
        When a valid ``"behavior_timelines"`` block is present.
        Keys are node-ID strings; values are BehaviorTimeline.to_dict() dicts.
    ``None``
        When the key is absent or the block is not a dict.

    Design notes
    ------------
    - Missing key (pre-Phase-1 files) → ``None`` — silent no-restore.
    - Present but not a dict → ``None`` — never raises.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("behavior_timelines")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


# ── Package metadata helpers (Circle 3) ──────────────────────────────────────


def build_package_metadata_payload(
    metadata: Any,
) -> Dict[str, Any]:
    """Serialise a PackageMetadata (or plain dict) for inclusion in a mission file.

    Parameters
    ----------
    metadata : PackageMetadata or dict or None
        When a ``PackageMetadata`` instance is supplied, its ``to_dict()``
        method is used.  A plain dict is accepted as-is (shallow copy).
        ``None`` produces an empty dict.

    Returns
    -------
    A dict safe for JSON serialisation.  Never raises.
    """
    if metadata is None:
        return {}
    if hasattr(metadata, "to_dict"):
        try:
            return dict(metadata.to_dict())
        except Exception:
            return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def extract_package_metadata_payload(
    data: Any,
    subgraph_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Extract package metadata for a specific subgraph from a loaded mission dict.

    Looks for ``data["subgraphs"][subgraph_id]["package_metadata"]`` when
    *subgraph_id* is provided; otherwise returns the raw ``"package_metadata"``
    key at the top level if present.

    Parameters
    ----------
    data        : Any        — loaded mission dict.
    subgraph_id : str        — target subgraph ID; empty → top-level lookup.

    Returns
    -------
    dict when the key is found and valid; ``None`` otherwise.  Never raises.
    """
    if not isinstance(data, dict):
        return None

    if subgraph_id:
        subgraphs = data.get("subgraphs")
        if isinstance(subgraphs, dict):
            sg = subgraphs.get(subgraph_id)
            if isinstance(sg, dict):
                raw = sg.get("package_metadata")
                if isinstance(raw, dict):
                    return dict(raw)
        return None

    raw = data.get("package_metadata")
    if isinstance(raw, dict):
        return dict(raw)
    return None


def needs_package_metadata_migration(data: Any) -> bool:
    """Return True when *data* contains subgraphs that lack package_metadata.

    Used by ``migrate_mission_payload()`` to detect older files whose
    subgraph records pre-date the Circle 3 package-metadata contract.
    Never raises.
    """
    if not isinstance(data, dict):
        return False
    subgraphs = data.get("subgraphs")
    if not subgraphs:
        return False
    items: list
    if isinstance(subgraphs, dict):
        items = list(subgraphs.values())
    elif isinstance(subgraphs, list):
        items = subgraphs
    else:
        return False
    for sg in items:
        if isinstance(sg, dict) and "package_metadata" not in sg:
            return True
    return False


# ── Protocol migration helpers (Step 5) ───────────────────────────────────────

# Schema versions that pre-date the protocol sub_dot contract (Step 4).
# Files at these versions may have Behavior "condition" connections typed "bool"
# rather than "protocol".  They must still load; the canvas will show a
# protocol_invalid border to prompt the user to rewire.
_PROTOCOL_UPGRADE_REQUIRED_BELOW = "1.4"


def _version_lt(v1: str, v2: str) -> bool:
    """Return True when version string v1 is strictly less than v2.

    Compares dot-separated integer segments; non-integer segments sort as 0.
    """
    def _parts(v: str):
        return [int(x) if x.isdigit() else 0 for x in v.split(".")]
    return _parts(v1) < _parts(v2)


def migrate_mission_payload(data: Any) -> Dict[str, Any]:
    """Inspect *data* for protocol-contract migration requirements.

    Does **not** mutate *data*.  Returns a migration-info dict so the caller
    can decide whether to log, warn, or adjust load behaviour.

    Returns
    -------
    dict with keys:
      ``"needs_protocol_upgrade"`` : bool
          True when the file predates the Step 4 protocol sub_dot contract
          (schema_version < "1.4" or absent).  Behaviour node "condition"
          connections may be typed "bool" and will load in migration-compat
          mode, showing a ``protocol_invalid`` border until rewired.
      ``"prior_schema_version"``   : str
          The schema_version found in *data*, or ``"unknown"`` when absent.
      ``"warnings"``               : List[str]
          Human-readable migration notices (empty when up-to-date).
    """
    if not isinstance(data, dict):
        return {
            "needs_protocol_upgrade": False,
            "prior_schema_version":   "unknown",
            "warnings": [],
        }

    prior = str(data.get("schema_version") or "").strip() or "unknown"
    needs_upgrade = (prior == "unknown") or _version_lt(prior, _PROTOCOL_UPGRADE_REQUIRED_BELOW)

    warnings: List[str] = []
    if needs_upgrade:
        warnings.append(
            f"Mission file schema_version={prior!r} predates the protocol sub_dot "
            f"contract (introduced in schema 1.4).  Behavior node 'condition' "
            f"connections will load in migration-compat mode and show a "
            f"protocol_invalid border until rewired to a protocol-typed output."
        )

    # Circle 3: detect missing package_metadata on subgraph records.
    needs_pkg_meta = needs_package_metadata_migration(data) if isinstance(data, dict) else False
    if needs_pkg_meta:
        warnings.append(
            "Mission file contains subgraphs without package_metadata (pre-Circle-3 format).  "
            "Subgraphs will load correctly; package traceability fields will be absent until "
            "the file is re-saved with the current editor version."
        )

    return {
        "needs_protocol_upgrade":         needs_upgrade,
        "needs_package_metadata_upgrade": needs_pkg_meta,   # Circle 3
        "prior_schema_version":           prior,
        "warnings":                       warnings,
    }
