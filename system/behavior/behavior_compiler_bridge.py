#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior compiler bridge — Phase 2 STAGE-02.

Extends the bridge from a thin MissionPlanner wrapper into a full
behavior_ref → artifact resolution pipeline:

  compile()       — source → BehaviorArtifact (diagnostics + trace_id)
  register()      — explicit artifact registration in in-memory registry
  resolve()       — behavior_ref → BehaviorResolveResult (structured; never bare None)
  lookup()        — behavior_ref → Optional[BehaviorArtifact] (valid only; delegates to resolve())
  lookup_any()    — behavior_ref → Optional[BehaviorArtifact] (valid or not)
  list_refs()     — enumerate all registered behavior_refs
  clear_registry()— flush in-memory registry (testing / hot-reload)

Resolution order (Phase 2 — in-memory only; persistence added Phase 4+):
  1. In-memory registry keyed by behavior_ref
     a. Not found   → BehaviorResolveResult with reason=BEHAVIOR_REF_NOT_FOUND
     b. Found, invalid → BehaviorResolveResult with reason=ARTIFACT_INVALID
     c. Found, valid   → BehaviorResolveResult.ok=True, artifact=<artifact>
  2. Phase 4+: persisted artifact store (not yet implemented)

Structured resolver return schema (BehaviorResolveResult)
---------------------------------------------------------
  ok          : bool   — True iff artifact is valid and usable
  reason      : str    — BehaviorErrorCode constant; "" on success
  artifact    : Optional[BehaviorArtifact] — None when missing
  diagnostics : List[BehaviorDiagnostic]   — resolution + compile diagnostics
  trace_id    : str    — correlation ID (caller-supplied or auto-generated)

Reason-code contract
--------------------
  missing behavior_ref → BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND
  invalid artifact     → BehaviorErrorCode.ARTIFACT_INVALID
  success              → reason == ""

Backward compatibility
----------------------
The original from_source() signature is preserved unchanged so that all
existing callers continue to work without modification.
lookup() still returns Optional[BehaviorArtifact] for legacy callers.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from bin.core.logger import log_warning
except (ImportError, AttributeError):
    def log_warning(*a, **k): pass  # type: ignore[misc]

from compiler.semantic.diagnostics import Diagnostic, DiagnosticLevel
from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, IRParam, NodeKind
from system.mission.mission_planner import MissionPlanner

from .behavior_artifact import (
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,      # noqa: F401 — re-exported for callers
    BehaviorResolveResult,  # noqa: F401 — re-exported for callers
)
from .editor_ir import build_behavior_editor_ir
from .heartbeat_policy import HeartbeatPolicy  # Circle 2 Step 2.1

# Phase 1 behavior redesign: action_profile is pure Python; import lazily to
# avoid hard dependency if the module is absent in trimmed distributions.
try:
    from .action_profile import BehaviorTimeline, validate_timeline  # type: ignore[attr-defined]
    _ACTION_PROFILE_AVAILABLE = True
except ImportError:
    _ACTION_PROFILE_AVAILABLE = False
    BehaviorTimeline = None  # type: ignore[assignment,misc]

try:
    from .timeline_migration import migrate_timeline, validate_timeline_topology  # type: ignore[attr-defined]
    _TIMELINE_MIGRATION_AVAILABLE = True
except ImportError:
    _TIMELINE_MIGRATION_AVAILABLE = False
    migrate_timeline = None  # type: ignore[assignment,misc]
    validate_timeline_topology = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_diagnostic(d: Diagnostic, trace_id: str) -> BehaviorDiagnostic:
    """Map a compiler Diagnostic to a BehaviorDiagnostic.

    Level mapping
    -------------
    DiagnosticLevel.ERROR   → "error"
    DiagnosticLevel.WARNING → "warning"  (compiler spells it "warn")
    DiagnosticLevel.INFO    → "info"

    Location: node_id takes precedence over line number.
    """
    _LEVEL_MAP = {
        DiagnosticLevel.ERROR:   "error",
        DiagnosticLevel.WARNING: "warning",
        DiagnosticLevel.INFO:    "info",
    }
    level = _LEVEL_MAP.get(d.level, "info")

    location: Optional[str] = None
    if d.location is not None:
        if d.location.node_id is not None:
            location = f"node:{d.location.node_id}"
        elif d.location.line is not None:
            location = f"line:{d.location.line}"

    return BehaviorDiagnostic(
        level=level,
        code=d.code,
        message=d.message,
        location=location,
        trace_id=trace_id,
    )


def _source_hash(source: str) -> str:
    """Return SHA-256 hex digest of a source string."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _brand_for_robot_type(robot_type: str) -> str:
    """Return best-effort brand name for a robot_type."""
    try:
        from system.model_registry import get_robot_brand_map
        mapping = get_robot_brand_map()
        brand = mapping.get(robot_type)
        if brand:
            return str(brand).lower()
    except Exception:
        pass
    fallback = {
        "go2": "unitree",
        "go2w": "unitree",
        "a1": "unitree",
        "b1": "unitree",
        "b2": "unitree",
        "h1": "unitree",
        "h1_2": "unitree",
        "g1": "unitree",
        "spot": "bostiondynamics",
        "cyberdog": "xiaomi",
        "cyberdog2": "xiaomi",
    }
    return fallback.get(str(robot_type or "").strip().lower(), "unknown")


def _has_executable_nodes(ir: WorkflowIR) -> bool:
    """Return True when the IR already contains executable behavior nodes."""
    executable_kinds = {
        NodeKind.ACTION,
        NodeKind.TIMER,
        NodeKind.STOP,
        NodeKind.WAIT,
        NodeKind.BEHAVIOR_CALL,
    }
    return any(node.kind in executable_kinds for node in ir.nodes)


def _timeline_to_behavior_ir(timeline: Any, robot_type: str) -> WorkflowIR:
    """Build a sequential WorkflowIR from a structured BehaviorTimeline."""
    brand = _brand_for_robot_type(robot_type)
    ir = WorkflowIR(
        robot_type=robot_type,
        brand=brand,
    )

    previous_node_id: Optional[str] = None
    ordered_segments = list(getattr(timeline, "action_segments", []) or [])
    editor_ir = build_behavior_editor_ir(timeline, brand=brand, robot_type=robot_type)
    modules_by_action: Dict[str, List[Dict[str, Any]]] = {}
    for action_seg in ordered_segments:
        action_id = str(getattr(action_seg, "action_id", "") or "")
        modules_by_action[action_id] = [
            module.to_dict() for module in editor_ir.modules_for_action(action_id)
        ]

    for index, seg in enumerate(ordered_segments):
        seg_name = str(getattr(seg, "name", "") or "").strip()
        seg_kind = str(getattr(seg, "kind", "movement") or "movement").strip().lower()
        seg_duration = float(getattr(seg, "duration", 1.0) or 1.0)
        seg_action_id = str(getattr(seg, "action_id", "") or "")
        seg_intent_id = str(getattr(seg, "intent_id", "") or "")
        seg_params = dict(getattr(seg, "params", {}) or {})
        seg_modules = modules_by_action.get(seg_action_id, [])

        if not seg_name:
            continue

        node_id = f"tl_{index}"
        node: Optional[IRNode] = None

        if seg_name == "wait":
            node = IRNode(
                id=node_id,
                schema_id="builtin.timer",
                kind=NodeKind.TIMER,
                params={
                    "duration": IRParam("duration", seg_duration, "float"),
                    "unit": IRParam("unit", "seconds", "string"),
                    "timeline_action_id": IRParam("timeline_action_id", seg_action_id, "string"),
                    "body_part_modules": IRParam("body_part_modules", seg_modules, "object"),
                },
            )
        elif seg_name == "stop":
            node = IRNode(
                id=node_id,
                schema_id="builtin.stop",
                kind=NodeKind.STOP,
                params={
                    "timeline_action_id": IRParam("timeline_action_id", seg_action_id, "string"),
                    "body_part_modules": IRParam("body_part_modules", seg_modules, "object"),
                },
            )
        elif seg_kind == "behavior":
            node = IRNode(
                id=node_id,
                schema_id="behavior_call",
                kind=NodeKind.BEHAVIOR_CALL,
                params={
                    "behavior_ref": IRParam("behavior_ref", seg_name, "string"),
                    "duration": IRParam("duration", seg_duration, "float"),
                    "timeline_action_id": IRParam("timeline_action_id", seg_action_id, "string"),
                    "body_part_modules": IRParam("body_part_modules", seg_modules, "object"),
                },
            )
        else:
            node = IRNode(
                id=node_id,
                schema_id="builtin.action_execution",
                kind=NodeKind.ACTION,
                params={
                    "action": IRParam("action", seg_name, "string"),
                    "duration": IRParam("duration", seg_duration, "float"),
                    "timeline_action_id": IRParam("timeline_action_id", seg_action_id, "string"),
                    "intent_id": IRParam("intent_id", seg_intent_id, "string"),
                    "action_params": IRParam("action_params", seg_params, "object"),
                    "body_part_modules": IRParam("body_part_modules", seg_modules, "object"),
                },
            )

        ir.add_node(node)
        if previous_node_id is not None:
            ir.add_edge(
                IREdge(
                    from_node=previous_node_id,
                    from_port="flow_out",
                    to_node=node_id,
                    to_port="flow_in",
                )
            )
        previous_node_id = node_id

    return ir


# ---------------------------------------------------------------------------
# BehaviorCompilerBridge
# ---------------------------------------------------------------------------

class BehaviorCompilerBridge:
    """Compile behavior sources and resolve behavior_ref → BehaviorArtifact.

    Registry
    --------
    The in-memory registry is a dict keyed by behavior_ref (str).
    compile() auto-registers valid artifacts.
    register() supports explicit external registration.

    lookup() filters out invalid artifacts (is_valid == False) so that
    STAGE-03 runtime dispatch always receives a usable subgraph or None.
    Use lookup_any() when diagnostic inspection of invalid artifacts is needed.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._planner = MissionPlanner()
        self._registry: Dict[str, BehaviorArtifact] = {}
        self._allow_disk_lookup: bool = True
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(__file__).resolve().parents[2] / "__cache__" / "behavior_artifacts"
        )
        self._resolved_cache_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Legacy API — preserved unchanged for backward compatibility
    # ------------------------------------------------------------------

    def from_source(
        self, source: str, robot_type: str = "go2"
    ) -> Tuple[WorkflowIR, List[Diagnostic]]:
        """Original bridge method — unchanged signature."""
        return self._planner.from_source(source, robot_type=robot_type)

    # ------------------------------------------------------------------
    # Phase 2: compile → BehaviorArtifact
    # ------------------------------------------------------------------

    def compile(
        self,
        source: str,
        behavior_ref: str,
        robot_type: str = "go2",
        trace_id: Optional[str] = None,
        heartbeat_policy: Optional[Dict[str, Any]] = None,
        timeline: Optional[Any] = None,
        is_simulation: bool = False,
        brand: Optional[str] = None,
    ) -> BehaviorArtifact:
        """Compile a behavior source string into a BehaviorArtifact.

        Steps
        -----
        1. Run compiler pipeline via MissionPlanner.from_source().
        2. Map compiler Diagnostics → BehaviorDiagnostics (shared trace_id).
        3. Create BehaviorArtifact with artifact_id, source_hash, trace_id.
        4. Populate heartbeat_policy: caller-supplied dict takes precedence;
           defaults to HeartbeatPolicy.default().to_dict() so v2 artifacts
           always carry a fully populated policy — never None (Circle 2 Step 2.1).
        5. Phase 1 behavior redesign: if *timeline* is a BehaviorTimeline,
           validate it (safe-range enforcement) and attach warning-level
           BehaviorDiagnostics for any out-of-range motor parameters.
           ``timeline.to_dict()`` is stored as ``artifact.timeline_data`` so
           that audit logs can always reference the structured timeline that
           produced this artifact.  Timeline validation is **non-blocking**:
           warnings do not affect ``is_valid`` or auto-registration.
        6. Auto-register in registry iff is_valid (no error-level diagnostics).

        The artifact is always returned regardless of is_valid so that callers
        can inspect diagnostics on compile failure.  Callers should check
        artifact.is_valid before relying on artifact.behavior_ir.

        Error code on compile failure: BehaviorErrorCode.BEHAVIOR_COMPILE_ERROR
        (callers decide whether to surface this as a BehaviorInvokeOutput).

        Args:
            heartbeat_policy: Optional policy dict override (Circle 2 Step 2.1).
                When None, defaults to HeartbeatPolicy.default().to_dict().
            timeline: Optional BehaviorTimeline instance (Phase 1 behavior redesign).
                When supplied, safe-range violations are appended as warning-level
                BehaviorDiagnostics (non-blocking).  ``timeline.to_dict()`` is
                stored in the artifact as ``timeline_data``.
            is_simulation: Passed to ``validate_timeline()``; uses sim_min/sim_max
                bounds instead of safe_min/safe_max when True.
            brand: Robot brand (e.g. ``"unitree"``).  When None, resolved via
                ``_brand_for_robot_type(robot_type)``.  Used for Step 9
                topology validation so that canonical track/param checks use
                the correct model-specific topology.
        """
        tid = trace_id or str(uuid.uuid4())
        effective_brand = brand if brand is not None else _brand_for_robot_type(robot_type)
        behavior_ir, compiler_diags = self._planner.from_source(
            source, robot_type=robot_type
        )
        behavior_diags = [_map_diagnostic(d, tid) for d in compiler_diags]

        # Circle 2 Step 2.1: always populate heartbeat_policy so v2 artifacts
        # carry a non-None policy dict from the moment of compilation.
        resolved_policy: Dict[str, Any] = (
            heartbeat_policy
            if heartbeat_policy is not None
            else HeartbeatPolicy.default().to_dict()
        )

        # Phase 1 behavior redesign: validate structured timeline if provided.
        # Emit warning-level diagnostics for out-of-range motor parameters;
        # these are non-blocking and do not affect is_valid.
        timeline_data: Optional[Dict[str, Any]] = None
        effective_timeline = timeline
        if timeline is not None and _ACTION_PROFILE_AVAILABLE:
            try:
                if _TIMELINE_MIGRATION_AVAILABLE and migrate_timeline is not None:
                    migrated = migrate_timeline(
                        timeline,
                        brand=effective_brand,
                        robot_type=robot_type,
                    )
                    effective_timeline = migrated.timeline
                    for warning in migrated.warnings:
                        log_warning(
                            f"BehaviorCompilerBridge: timeline migration warning for "
                            f"'{behavior_ref}': {warning}"
                        )
                motor_diags = validate_timeline(
                    effective_timeline,
                    is_simulation=is_simulation,
                )
                for md in motor_diags:
                    behavior_diags.append(BehaviorDiagnostic(
                        level=md.level,
                        code=f"timeline.motor_{md.param_key}_out_of_range",
                        message=md.message,
                        location=f"track:{md.track_name}" if md.track_name else None,
                        trace_id=tid,
                    ))
                # Step 9: topology validation — unknown tracks are ERROR-level
                # (block is_valid); param-key mismatches are WARNING-level.
                if _TIMELINE_MIGRATION_AVAILABLE and validate_timeline_topology is not None:
                    topo_diags = validate_timeline_topology(
                        effective_timeline,
                        brand=effective_brand,
                        robot_type=robot_type,
                    )
                    for td in topo_diags:
                        location = f"track:{td.track_name}" if td.track_name else None
                        behavior_diags.append(BehaviorDiagnostic(
                            level=td.level,
                            code=td.code,
                            message=td.message,
                            location=location,
                            trace_id=tid,
                        ))
                timeline_data = effective_timeline.to_dict()
            except Exception as _tl_exc:  # noqa: BLE001
                log_warning(
                    f"BehaviorCompilerBridge: timeline validation failed for "
                    f"'{behavior_ref}': {type(_tl_exc).__name__}: {_tl_exc}"
                )

        if effective_timeline is not None and not _has_executable_nodes(behavior_ir):
            try:
                behavior_ir = _timeline_to_behavior_ir(effective_timeline, robot_type)
            except Exception as _ir_exc:  # noqa: BLE001
                log_warning(
                    f"BehaviorCompilerBridge: timeline IR fallback failed for "
                    f"'{behavior_ref}': {type(_ir_exc).__name__}: {_ir_exc}"
                )

        artifact = BehaviorArtifact.create(
            behavior_ref=behavior_ref,
            behavior_ir=behavior_ir,
            diagnostics=behavior_diags,
            trace_id=tid,
            source_hash=_source_hash(source),
            heartbeat_policy=resolved_policy,
            timeline_data=timeline_data,
        )

        # Only register if no compile errors — prevents stale invalid artifacts
        if artifact.is_valid:
            self.register(artifact)

        return artifact

    # ------------------------------------------------------------------
    # Phase 2: registry operations
    # ------------------------------------------------------------------

    def register(self, artifact: BehaviorArtifact) -> None:
        """Explicitly register a pre-built artifact by behavior_ref.

        Overwrites any existing entry for the same behavior_ref.
        Callers are responsible for validating artifact.is_valid if they
        want to avoid registering invalid artifacts.
        """
        self._allow_disk_lookup = True
        self._registry[artifact.behavior_ref] = artifact
        self._persist_artifact(artifact)

    def resolve(
        self, behavior_ref: str, trace_id: Optional[str] = None
    ) -> BehaviorResolveResult:
        """Structured resolution: behavior_ref → BehaviorResolveResult.

        Always returns a fully-populated result — never bare None.  The
        caller receives an explicit ok/reason/diagnostics on every path.

        Resolution order (Phase 2 — in-memory only):
          1. behavior_ref not in registry
             → BehaviorResolveResult.missing()   reason=BEHAVIOR_REF_NOT_FOUND
          2. artifact found but is_valid=False
             → BehaviorResolveResult.invalid()   reason=ARTIFACT_INVALID
          3. artifact found and is_valid=True
             → BehaviorResolveResult.success()   ok=True

        trace_id is generated fresh if not supplied by the caller.
        """
        tid = trace_id or str(uuid.uuid4())
        artifact = self._registry.get(behavior_ref)
        if artifact is None and self._allow_disk_lookup:
            artifact = self._load_latest_persisted_artifact(behavior_ref)
            if artifact is not None:
                self._registry[behavior_ref] = artifact
        if artifact is None:
            return BehaviorResolveResult.missing(behavior_ref, tid)
        if not artifact.is_valid:
            return BehaviorResolveResult.invalid(artifact, tid)
        return BehaviorResolveResult.success(artifact, tid)

    def lookup(self, behavior_ref: str) -> Optional[BehaviorArtifact]:
        """Resolve behavior_ref → valid BehaviorArtifact or None.

        Delegates to resolve() internally; returns None on any failure.
        Signature and semantics are unchanged for legacy callers.
        """
        result = self.resolve(behavior_ref)
        return result.artifact if result.ok else None

    def lookup_any(self, behavior_ref: str) -> Optional[BehaviorArtifact]:
        """Like lookup() but returns invalid artifacts too.

        Useful for diagnostics inspection when a compile produced errors
        and the caller wants to surface them to the user.
        """
        artifact = self._registry.get(behavior_ref)
        if artifact is not None:
            return artifact
        if not self._allow_disk_lookup:
            return None
        artifact = self._load_latest_persisted_artifact(behavior_ref)
        if artifact is not None:
            self._registry[behavior_ref] = artifact
        return artifact

    def list_refs(self) -> List[str]:
        """Return all registered behavior_ref keys (valid and invalid)."""
        return list(self._registry.keys())

    def clear_registry(self) -> None:
        """Remove in-memory artifacts and disable disk fallback for this instance."""
        self._registry.clear()
        self._allow_disk_lookup = False

    def duplicate_ref(
        self,
        source_ref: str,
        new_ref: Optional[str] = None,
    ) -> Optional[str]:
        """Duplicate a registered/persisted behavior_ref into a new ref.

        Returns the new ref on success, otherwise None.
        """
        src = str(source_ref or "").strip()
        if not src:
            return None
        source = self.lookup_any(src)
        if source is None:
            return None

        if new_ref is not None and str(new_ref).strip():
            target = str(new_ref).strip()
        else:
            base = f"{src}_copy"
            target = base
            suffix = 2
            while self.lookup_any(target) is not None or self._ref_cache_dir(target).exists():
                target = f"{base}{suffix}"
                suffix += 1

        if not target or target == src:
            return None
        if self.lookup_any(target) is not None or self._ref_cache_dir(target).exists():
            return None

        cloned = BehaviorArtifact.create(
            behavior_ref=target,
            behavior_ir=source.behavior_ir,
            diagnostics=list(source.diagnostics),
            source_hash=source.source_hash,
            heartbeat_ir=source.heartbeat_ir,
            heartbeat_policy=(
                dict(source.heartbeat_policy)
                if isinstance(source.heartbeat_policy, dict)
                else source.heartbeat_policy
            ),
            heartbeat_script=source.heartbeat_script,
        )
        self.register(cloned)
        return target

    def delete_ref(self, behavior_ref: str) -> bool:
        """Delete behavior_ref from in-memory registry and persisted cache."""
        ref = str(behavior_ref or "").strip()
        if not ref:
            return False

        removed = self._registry.pop(ref, None) is not None
        ref_dir = self._ref_cache_dir(ref)
        if ref_dir.exists():
            try:
                shutil.rmtree(ref_dir)
                removed = True
            except Exception as exc:
                log_warning(
                    f"BehaviorCompilerBridge: failed to delete cached ref '{ref}' "
                    f"under '{ref_dir}': {type(exc).__name__}: {exc}"
                )
        return removed

    # ------------------------------------------------------------------
    # Persistence (cache-backed artifact recovery)
    # ------------------------------------------------------------------

    @staticmethod
    def _behavior_ref_hash(behavior_ref: str) -> str:
        return hashlib.sha1(behavior_ref.encode("utf-8")).hexdigest()

    def _ref_cache_dir(self, behavior_ref: str) -> Path:
        return self._get_persistence_root() / self._behavior_ref_hash(behavior_ref)

    def _get_persistence_root(self) -> Path:
        """Return a writable cache root for artifact persistence.

        Some restricted environments expose a temp/cache path that exists but
        is not writable for nested files or directories. When that happens,
        fall back to a deterministic project-local cache path derived from the
        requested cache_dir so multiple bridge instances still share artifacts.
        """
        if self._resolved_cache_dir is not None:
            return self._resolved_cache_dir

        requested = self._cache_dir
        if self._is_writable_dir(requested):
            self._resolved_cache_dir = requested
            return requested

        fallback_root = Path(__file__).resolve().parents[2] / "__cache__" / "behavior_artifacts_fallback"
        fallback_root.mkdir(parents=True, exist_ok=True)
        fallback_dir = fallback_root / self._behavior_ref_hash(str(requested))
        fallback_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_cache_dir = fallback_dir
        log_warning(
            f"BehaviorCompilerBridge: cache_dir '{requested}' is not writable; "
            f"using fallback cache '{fallback_dir}'"
        )
        return fallback_dir

    @staticmethod
    def _is_writable_dir(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _artifact_to_record(artifact: BehaviorArtifact) -> Dict[str, object]:
        ir_dict = artifact.behavior_ir.to_dict()
        record: Dict[str, object] = {
            "artifact_id": artifact.artifact_id,
            "behavior_ref": artifact.behavior_ref,
            # Circle 2 Step 2.1: persist core_ir alongside behavior_ir for
            # forward compat. New readers use core_ir; old readers fall back to
            # behavior_ir.  Both keys always hold the same WorkflowIR dict.
            "behavior_ir": ir_dict,
            "core_ir":     ir_dict,
            "diagnostics": [d.to_dict() for d in artifact.diagnostics],
            "trace_id": artifact.trace_id,
            "compiled_at": artifact.compiled_at,
            "source_hash": artifact.source_hash,
            # v2 — heartbeat reserved (Circle 1 Step 1.2 / Circle 2 Step 2.1)
            "heartbeat_ir": (
                artifact.heartbeat_ir.to_dict()
                if artifact.heartbeat_ir is not None
                else None
            ),
            "heartbeat_policy": artifact.heartbeat_policy,
            "heartbeat_script": artifact.heartbeat_script,
            # Phase 1 behavior redesign — structured timeline snapshot (None for legacy)
            "timeline_data": artifact.timeline_data,
        }
        return record

    @staticmethod
    def _record_to_artifact(record: Dict[str, object]) -> Optional[BehaviorArtifact]:
        try:
            # Circle 2 Step 2.1: prefer core_ir (v2 key); fall back to
            # behavior_ir (v1 key) so old persisted records load without migration.
            ir_dict = record.get("core_ir") or record.get("behavior_ir")
            if not isinstance(ir_dict, dict):
                return None
            diagnostics: List[BehaviorDiagnostic] = []
            for item in record.get("diagnostics", []) or []:
                if not isinstance(item, dict):
                    continue
                diagnostics.append(
                    BehaviorDiagnostic(
                        level=str(item.get("level", "info")),
                        code=str(item.get("code", "")),
                        message=str(item.get("message", "")),
                        location=(
                            None
                            if item.get("location") is None
                            else str(item.get("location"))
                        ),
                        trace_id=str(item.get("trace_id", str(uuid.uuid4()))),
                    )
                )
            # v2 — restore heartbeat_ir if persisted (Circle 1 Step 1.2)
            hb_ir_dict = record.get("heartbeat_ir")
            heartbeat_ir = (
                WorkflowIR.from_dict(hb_ir_dict)
                if isinstance(hb_ir_dict, dict)
                else None
            )
            hb_policy = record.get("heartbeat_policy")
            hb_script = record.get("heartbeat_script")
            # Phase 1 behavior redesign: restore timeline_data if persisted.
            tl_data = record.get("timeline_data")
            return BehaviorArtifact(
                artifact_id=str(record.get("artifact_id", str(uuid.uuid4()))),
                behavior_ref=str(record.get("behavior_ref", "")),
                behavior_ir=WorkflowIR.from_dict(ir_dict),
                diagnostics=diagnostics,
                trace_id=str(record.get("trace_id", str(uuid.uuid4()))),
                compiled_at=str(record.get("compiled_at", "")),
                source_hash=(
                    None
                    if record.get("source_hash") is None
                    else str(record.get("source_hash"))
                ),
                heartbeat_ir=heartbeat_ir,
                heartbeat_policy=hb_policy if isinstance(hb_policy, dict) else None,
                heartbeat_script=(
                    str(hb_script) if isinstance(hb_script, str) else None
                ),
                timeline_data=tl_data if isinstance(tl_data, dict) else None,
            )
        except Exception:
            return None

    def _persist_artifact(self, artifact: BehaviorArtifact) -> None:
        try:
            ref_dir = self._ref_cache_dir(artifact.behavior_ref)
            ref_dir.mkdir(parents=True, exist_ok=True)
            record = self._artifact_to_record(artifact)
            payload = json.dumps(record, ensure_ascii=False, indent=2)

            artifact_path = ref_dir / f"{artifact.artifact_id}.json"
            artifact_path.write_text(payload, encoding="utf-8")

            latest_path = ref_dir / "latest.json"
            latest_path.write_text(payload, encoding="utf-8")
        except Exception as exc:
            # Persistence failures must never break compile/resolve flow, but they
            # must be diagnosable.  Log a warning so operators can detect cache
            # path permission issues (e.g. restricted CI/sandbox temp dirs) without
            # having to add debug instrumentation.
            log_warning(
                f"BehaviorCompilerBridge: failed to persist artifact "
                f"'{artifact.behavior_ref}' (id={artifact.artifact_id}) "
                f"to '{self._cache_dir}': {type(exc).__name__}: {exc}"
            )

    def _load_latest_persisted_artifact(
        self,
        behavior_ref: str,
    ) -> Optional[BehaviorArtifact]:
        ref_dir = self._ref_cache_dir(behavior_ref)
        if not ref_dir.exists():
            return None

        candidate_paths: List[Path] = []
        latest_path = ref_dir / "latest.json"
        if latest_path.exists():
            candidate_paths.append(latest_path)
        candidate_paths.extend(
            sorted(ref_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        )

        seen: set = set()
        for path in candidate_paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            artifact = self._record_to_artifact(raw)
            if artifact is None:
                continue
            if artifact.behavior_ref != behavior_ref:
                continue
            return artifact
        return None
