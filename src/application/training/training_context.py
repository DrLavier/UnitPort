"""application.training.training_context — TrainingContext (validation + compat).

DEMO 对应：``DEMO/src/system/training/training_context.py``.

Two responsibilities for the Export node UI:

  1. ``compat_report(graph)``  — Start Point compatibility diff for the §2
     panel (StartPointCompatPanelRow). 4 fixed CompatField rows
     (Framework / Action dim / Decimation / Joint order) + a summary line.

  2. ``safety_review(graph)``  — pre-flight validation gate for the
     "▶ Launch Review" button (ReviewLaunchButtonRow). Returns
     ``RefreshResult(ok, issues)``; when ``ok=False`` the launch is blocked
     and the issue list paints red borders on offending nodes.

Stage D scope: ports the *contracts* (dataclasses + method signatures) so
the UI is correctly wired today. The full DEMO compat/validation rule
set is large (~700 LOC) and reaches into bundle parsing + spec compiling;
this module ships a **minimal viable** implementation that:

  * Walks the graph for a ``base_asset`` / ``robot`` / ``train`` /
    ``export`` node so the panel + button do something coherent.
  * Returns 4 CompatField placeholders with status ``"none"`` when no
    Start Point is wired up.
  * ``safety_review`` returns ``ok=True`` unless a ``hard`` rule trips
    (currently only "no ExportNode on canvas").

Future expansion (by stage):
  * Stage E+ — Wire ``ExportedBundleRegistry`` to load the upstream
    bundle's manifest + drive the 4 compat fields with real diffs.
  * Stage E+ — Port DEMO's ``_check_start_point`` (S2/S3/S4/S5 rules) +
    the per-section refresh helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# CompatField + StartPointCompatReport — §2 panel data
# ---------------------------------------------------------------------------

#: Allowed status tokens. UI maps:
#:    ok      → ✓ (green)
#:    warning → ⚠ (orange) — warm-start will reindex etc.
#:    error   → ✗ (red)
#:    none    → · (muted) — N/A, no Start Point connected
COMPAT_STATUSES = ("ok", "warning", "error", "none")


@dataclass(frozen=True)
class CompatField:
    """One row in the Start Point Compatibility panel."""

    label: str
    """Display label — must be one of StartPointCompatPanelRow._SPC_FIELDS."""
    old_value: str = ""
    """Value loaded from the upstream bundle manifest."""
    new_value: str = ""
    """Value from the current canvas configuration."""
    status: str = "none"
    """One of COMPAT_STATUSES."""
    detail: str = ""
    """Optional explanation surfaced in tooltips / log."""


@dataclass
class StartPointCompatReport:
    """Aggregate report for the §2 panel."""

    has_start_point: bool = False
    """True iff the Export node's upstream train_pipe carries a Start Point."""
    start_point_label: str = ""
    """asset_id or checkpoint filename — short label for the panel header."""
    load_mode: str = "scratch"
    """One of: scratch | resume | warm_start_actor."""
    manifest_loaded: bool = False
    """True iff the bundle manifest parsed cleanly."""
    manifest_error: str = ""
    """Non-empty when ``manifest_loaded=False``."""
    fields: List[CompatField] = field(default_factory=list)
    """4 CompatField entries — Framework / Action dim / Decimation / Joint order."""
    overall_status: str = "none"
    """Aggregate status for the footer summary (worst of fields[*].status)."""
    summary: str = ""
    """One-liner footer text."""


# ---------------------------------------------------------------------------
# ValidationIssue + RefreshResult — safety_review output
# ---------------------------------------------------------------------------

#: Severity levels. UI maps:
#:    error   → red border on node, blocks Launch / Train
#:    warning → orange border, advisory only
ISSUE_SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class ValidationIssue:
    """One issue surfaced by safety_review."""

    severity: str
    """One of ISSUE_SEVERITIES."""
    section: str
    """Logical group — robot | scene | compile | safety | cross | export | ..."""
    code: str
    """Machine-readable id, e.g. ``S3_start_point_action_dim_mismatch``."""
    message: str
    """Human-readable log text."""
    node_id: Optional[str] = None
    """Canvas node to highlight; None ⇒ canvas-global issue."""
    node_type: Optional[str] = None
    """Fallback when node_id unavailable (e.g. "no ExportNode on canvas")."""
    fix_hint: Optional[str] = None
    """Optional remediation hint."""


@dataclass
class RefreshResult:
    """Outcome of a safety_review pass."""

    ok: bool
    """True iff there are zero error-severity issues."""
    issues: List[ValidationIssue] = field(default_factory=list)
    revision: int = 0
    """Monotonic counter — increments per safety_review call within a context."""

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# TrainingContext
# ---------------------------------------------------------------------------


# Field labels must match StartPointCompatPanelRow._SPC_FIELDS in
# application.ui.canvas.param_rows — they are paired by string label.
_COMPAT_LABELS = ("Framework", "Action dim", "Decimation", "Joint order")


def _aggregate_status(statuses: List[str]) -> str:
    """Worst-of: error > warning > ok > none. Empty input → 'none'."""
    priority = {"error": 3, "warning": 2, "ok": 1, "none": 0}
    if not statuses:
        return "none"
    return max(statuses, key=lambda s: priority.get(s, 0))


def _str_or_dash(v: Any) -> str:
    """Display helper — render None / empty as '—'."""
    if v is None or v == "":
        return "—"
    return str(v)


class TrainingContext:
    """Per-workspace training state holder.

    Stage D: just enough state to drive the Export node's §2 panel + the
    "▶ Launch Review" button gating. Future stages will fold the IR
    compile result + bundle manifest cache + RefreshResult history here.
    """

    def __init__(self, project_id: str = "") -> None:
        self.project_id = str(project_id or "")
        self._revision: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nodes_by_kind(graph: Optional[Dict[str, Any]], schema_id: str) -> List[Dict[str, Any]]:
        """Return every node in ``graph`` whose ``schema_id`` matches.

        Tolerant to both DEMO ``node_type`` and RELEASE ``schema_id`` keys
        so the same helper works against either dict shape.
        """
        if not isinstance(graph, dict):
            return []
        out: List[Dict[str, Any]] = []
        for n in graph.get("nodes", []) or []:
            if not isinstance(n, dict):
                continue
            sid = str(n.get("schema_id") or n.get("node_type") or "")
            if sid == schema_id:
                out.append(n)
        return out

    @staticmethod
    def _node_param(node: Dict[str, Any], key: str, default: Any = "") -> Any:
        """Read a param value from an IR-shape node dict.

        Params are dict[name → IRParam-shape] where IRParam = {"name", "value", "param_type"}.
        Tolerant to plain values too (DEMO sometimes inlines).
        """
        params = node.get("params") if isinstance(node, dict) else None
        if not isinstance(params, dict):
            return default
        spec = params.get(key)
        if spec is None:
            return default
        if isinstance(spec, dict) and "value" in spec:
            return spec["value"]
        return spec

    @staticmethod
    def _detect_load_mode(graph: Optional[Dict[str, Any]]) -> tuple:
        """Return ``(load_mode, label)`` from the graph's base_asset node.

        Mirrors DEMO ``_resolve_start_point`` (training_context.py:391-450)
        in spirit: looks for a base_asset node, reads its ``start_point``
        token, derives load_mode + a display label.
        """
        for ba in TrainingContext._nodes_by_kind(graph, "base_asset"):
            sp = str(TrainingContext._node_param(ba, "start_point", "") or "").strip()
            if not sp or sp == "__new__":
                return "scratch", ""
            if sp == "__latest_export__":
                return "resume", "latest export"
            if sp.startswith("run:"):
                return "resume", sp[4:].rsplit("/", 1)[-1] or sp
            if sp.startswith("asset:"):
                return "warm_start_actor", sp[6:]
            return "warm_start_actor", sp
        return "scratch", ""

    # ------------------------------------------------------------------
    # compat_report — §2 panel data (Stage F-2: real bundle vs canvas diff)
    # ------------------------------------------------------------------

    # ---- bundle-side: load manifest from start_point token --------------

    def _load_start_point_manifest(
        self, graph: Optional[Dict[str, Any]]
    ) -> Optional[tuple]:
        """Resolve base_asset.start_point → ``(manifest_dict, label, bundle_path)``.

        Returns None when no Start Point or the bundle can't be loaded.

        Token grammar (mirrors ``_start_point_param_patch`` in param_rows.py):
          * ``run:<abs_path_to_.pt>``  → bundle_dir = parent(.pt)
          * ``asset:<policy_id>``      → ExportedBundleRegistry().discover()
                                          找匹配 entry
          * ``__latest_export__``      → 取最新 mtime 的有效 ExportedBundleEntry
        Then YAML-parses ``<bundle_dir>/manifest.yaml`` via BundleLoader.
        Defensive: any IO / parse error returns None (caller renders the
        "bundle could not be loaded" branch).
        """
        for ba in self._nodes_by_kind(graph, "base_asset"):
            sp = str(self._node_param(ba, "start_point", "") or "").strip()
            if not sp or sp == "__new__":
                return None
            from pathlib import Path

            bundle_dir: Optional[Path] = None
            label: str = ""
            try:
                if sp.startswith("run:"):
                    ckpt = Path(sp[4:])
                    if ckpt.is_file():
                        bundle_dir = ckpt.parent
                        label = ckpt.name
                elif sp.startswith("asset:"):
                    pid = sp[6:]
                    label = pid
                    from application.service.exported_bundle_registry import (
                        ExportedBundleRegistry,
                    )
                    for entry in ExportedBundleRegistry().discover():
                        if entry.is_valid and entry.policy_id == pid:
                            bundle_dir = Path(entry.bundle_path)
                            break
                elif sp == "__latest_export__":
                    from application.service.exported_bundle_registry import (
                        ExportedBundleRegistry,
                    )
                    valid = [
                        e for e in ExportedBundleRegistry().discover() if e.is_valid
                    ]
                    if valid:
                        latest = max(
                            valid,
                            key=lambda e: (
                                Path(e.bundle_path).stat().st_mtime
                                if Path(e.bundle_path).exists() else 0.0
                            ),
                        )
                        bundle_dir = Path(latest.bundle_path)
                        label = latest.policy_id
            except Exception:
                return None
            if bundle_dir is None or not bundle_dir.is_dir():
                return None
            try:
                from application.service.runtime.policy.bundle_loader import (
                    BundleLoader,
                )
                manifest = BundleLoader._parse_manifest(bundle_dir)
            except Exception:
                return None
            return manifest, (label or bundle_dir.name), bundle_dir
        return None

    # ---- canvas-side helpers -------------------------------------------

    def _canvas_backend(self, graph: Optional[Dict[str, Any]]) -> str:
        """Read algorithm_config.backend; fallback algo_config / train, else 'sb3'.

        Different node names exist for the algorithm-config role across
        canvas backends — try the most specific (algorithm_config) first.
        """
        for sid in ("algorithm_config", "algo_config"):
            for n in self._nodes_by_kind(graph, sid):
                v = str(self._node_param(n, "backend", "") or "").strip()
                if v:
                    return v
        for tn in self._nodes_by_kind(graph, "train"):
            v = str(self._node_param(tn, "backend", "") or "").strip()
            if v:
                return v
        return "sb3"

    def _canvas_robot_asset(self, graph: Optional[Dict[str, Any]]):
        """robot.asset_id → registers.robots.resolve_id → RobotAssetService.resolve.

        Returns the RobotAsset or None (no Robot node / unresolved SKU /
        asset service unavailable). All exceptions degrade to None.
        """
        for rn in self._nodes_by_kind(graph, "robot"):
            raw = str(self._node_param(rn, "asset_id", "") or "").strip()
            if not raw:
                return None
            try:
                from registers import robots as _robots_registry
                sku = (
                    raw if _robots_registry.get_robot(raw) is not None
                    else _robots_registry.resolve_id(raw)
                )
                if not sku:
                    return None
                from application.service.robot_assets.service import (
                    get_robot_asset_service,
                )
                return get_robot_asset_service().resolve(str(sku))
            except Exception:
                return None
        return None

    def _canvas_action_dim(self, graph: Optional[Dict[str, Any]]) -> Optional[int]:
        """``len(asset.joints)`` — joints is Dict[str,str] preserving insertion order."""
        asset = self._canvas_robot_asset(graph)
        if asset is None:
            return None
        joints = getattr(asset, "joints", None)
        if not isinstance(joints, dict) or not joints:
            return None
        return int(len(joints))

    def _canvas_decimation(self, graph: Optional[Dict[str, Any]]) -> Optional[int]:
        """``round(physics_config.control_dt / physics_config.sim_dt)``.

        Returns None when physics_config missing or either dt is non-positive.
        """
        for pn in self._nodes_by_kind(graph, "physics_config"):
            try:
                control_dt = float(self._node_param(pn, "control_dt", 0) or 0)
                sim_dt = float(self._node_param(pn, "sim_dt", 0) or 0)
            except (TypeError, ValueError):
                return None
            if control_dt > 0 and sim_dt > 0:
                return int(round(control_dt / sim_dt))
            return None
        return None

    def _canvas_joint_order(self, graph: Optional[Dict[str, Any]]) -> List[str]:
        """``list(asset.joints.keys())`` — canonical joint order from RobotAsset."""
        asset = self._canvas_robot_asset(graph)
        if asset is None:
            return []
        joints = getattr(asset, "joints", None)
        if not isinstance(joints, dict):
            return []
        return list(joints.keys())

    # ---- diff helpers (each returns one CompatField) -------------------

    @staticmethod
    def _diff_framework(old: Any, new: Any) -> CompatField:
        old_s = str(old or "").strip()
        new_s = str(new or "").strip()
        if old_s and new_s and old_s == new_s:
            status, detail = "ok", ""
        elif not old_s or not new_s:
            status, detail = "warning", "framework brand missing on one side"
        else:
            status, detail = (
                "warning",
                "framework differs — cross-train compatibility not guaranteed",
            )
        return CompatField(
            label="Framework",
            old_value=old_s or "—",
            new_value=new_s or "—",
            status=status,
            detail=detail,
        )

    @staticmethod
    def _diff_action_dim(old: Any, new: Any) -> CompatField:
        try:
            old_i = int(old) if old is not None and old != "" else None
        except (TypeError, ValueError):
            old_i = None
        try:
            new_i = int(new) if new is not None and new != "" else None
        except (TypeError, ValueError):
            new_i = None
        if old_i is not None and new_i is not None and old_i == new_i:
            status, detail = "ok", ""
        elif old_i is None or new_i is None:
            status, detail = "warning", "action_dim unknown on one side"
        else:
            status, detail = (
                "error",
                "policy outputs wrong shape — bundle/canvas action_dim mismatch",
            )
        return CompatField(
            label="Action dim",
            old_value=_str_or_dash(old_i),
            new_value=_str_or_dash(new_i),
            status=status,
            detail=detail,
        )

    @staticmethod
    def _diff_decimation(old: Any, new: Any) -> CompatField:
        try:
            old_i = int(old) if old is not None and old != "" else None
        except (TypeError, ValueError):
            old_i = None
        try:
            new_i = int(new) if new is not None and new != "" else None
        except (TypeError, ValueError):
            new_i = None
        if old_i is not None and new_i is not None and old_i == new_i:
            status, detail = "ok", ""
        elif old_i is None or new_i is None:
            status, detail = "warning", "decimation unknown on one side"
        else:
            # DEMO: mismatch is warning (policy still runs, just at different rate).
            status, detail = "warning", f"control rate differs ({old_i}→{new_i})"
        return CompatField(
            label="Decimation",
            old_value=_str_or_dash(old_i),
            new_value=_str_or_dash(new_i),
            status=status,
            detail=detail,
        )

    @staticmethod
    def _diff_joint_order(
        old: List[str], new: List[str], load_mode: str,
    ) -> CompatField:
        old_l = list(old or [])
        new_l = list(new or [])
        old_disp = f"{len(old_l)} joints" if old_l else "—"
        new_disp = f"{len(new_l)} joints" if new_l else "—"
        if old_l and new_l and old_l == new_l:
            status, detail = "ok", ""
        elif not old_l or not new_l:
            status, detail = "warning", "joint list missing on one side"
        elif load_mode == "resume":
            status, detail = (
                "error",
                "resume mode requires byte-exact joint order match",
            )
        else:  # warm_start_actor / scratch
            status, detail = (
                "warning",
                f"warm_start_actor will reindex ({len(set(old_l) ^ set(new_l))} differ)",
            )
        return CompatField(
            label="Joint order",
            old_value=old_disp,
            new_value=new_disp,
            status=status,
            detail=detail,
        )

    # ---- compat_report orchestrator ------------------------------------

    def compat_report(self, graph: Optional[Dict[str, Any]]) -> StartPointCompatReport:
        """Return a StartPointCompatReport for the Export node §2 panel.

        Stage F-2 implementation: real bundle vs canvas diff.
          * Detects Start Point token via base_asset node.
          * Loads bundle manifest via BundleLoader._parse_manifest.
          * Diffs 4 fields against canvas-side values (RobotAsset joints,
            algorithm_config.backend, physics_config.control_dt/sim_dt).
          * Status policy follows DEMO compat_report (training_context.py:391-625):
            framework/decimation mismatch → warning; action_dim mismatch →
            error; joint_order mismatch + resume → error, else warning.
        """
        load_mode, label = self._detect_load_mode(graph)
        has_sp = load_mode != "scratch"

        if not has_sp:
            return StartPointCompatReport(
                has_start_point=False,
                start_point_label="",
                load_mode="scratch",
                manifest_loaded=False,
                fields=[
                    CompatField(label=lbl, status="none") for lbl in _COMPAT_LABELS
                ],
                overall_status="none",
                summary="No Start Point connected — connect train_pipe to populate",
            )

        loaded = self._load_start_point_manifest(graph)
        if loaded is None:
            return StartPointCompatReport(
                has_start_point=True,
                start_point_label=label,
                load_mode=load_mode,
                manifest_loaded=False,
                manifest_error="bundle not found / unparseable",
                fields=[
                    CompatField(
                        label=lbl, status="warning", detail="bundle missing",
                    )
                    for lbl in _COMPAT_LABELS
                ],
                overall_status="warning",
                summary=(
                    f"Start Point: {label or '(unnamed)'} — bundle could not be loaded"
                ),
            )

        manifest, sp_label, bundle_path = loaded
        # Bundle-side values (defensive .get — legacy bundles may miss keys)
        robot_block = manifest.get("robot") if isinstance(manifest, dict) else None
        runtime_block = manifest.get("runtime") if isinstance(manifest, dict) else None
        old_brand = (robot_block or {}).get("brand", "")
        old_dim = (robot_block or {}).get("num_joints")
        old_dec = (runtime_block or {}).get("decimation")
        old_joints = list((robot_block or {}).get("joint_names", []) or [])

        # Canvas-side values
        new_backend = self._canvas_backend(graph)
        new_dim = self._canvas_action_dim(graph)
        new_dec = self._canvas_decimation(graph)
        new_joints = self._canvas_joint_order(graph)

        fields = [
            self._diff_framework(old_brand, new_backend),
            self._diff_action_dim(old_dim, new_dim),
            self._diff_decimation(old_dec, new_dec),
            self._diff_joint_order(old_joints, new_joints, load_mode),
        ]
        overall = _aggregate_status([f.status for f in fields])
        return StartPointCompatReport(
            has_start_point=True,
            start_point_label=sp_label,
            load_mode=load_mode,
            manifest_loaded=True,
            fields=fields,
            overall_status=overall,
            summary=(
                f"Start Point: {sp_label} (load_mode={load_mode}) · {overall}"
            ),
        )

    # ------------------------------------------------------------------
    # safety_review — Launch Review gate
    # ------------------------------------------------------------------

    def safety_review(
        self,
        graph: Optional[Dict[str, Any]],
        *,
        policy_id: str = "",
        experiment_id: str = "",
    ) -> RefreshResult:
        """Pre-flight validation. Block Launch when ``ok=False``.

        Stage D minimal rules (all errors block, all warnings advisory):
          * E1 (error)   — no ExportNode on canvas.
          * E2 (error)   — Export node's bundle_name is the ``<NEW>`` sentinel.
          * E3 (warning) — no Robot node on canvas (Launch will degrade).
          * E4 (warning) — review_backend selects ``newton`` (placeholder).
        """
        self._revision += 1
        issues: List[ValidationIssue] = []

        exports = self._nodes_by_kind(graph, "export")
        if not exports:
            issues.append(ValidationIssue(
                severity="error",
                section="export",
                code="E1_no_export_node",
                message="No ExportNode on canvas — Launch Review cannot proceed.",
                node_type="export",
                fix_hint="Drag an Export node onto the canvas and connect train_pipe.",
            ))
        else:
            for ex in exports:
                bn = str(self._node_param(ex, "bundle_name", "<NEW>") or "<NEW>").strip()
                if bn in ("", "<NEW>"):
                    issues.append(ValidationIssue(
                        severity="error",
                        section="export",
                        code="E2_export_bundle_name_sentinel",
                        message="ExportNode.bundle_name is still the <NEW> sentinel — "
                                "set a concrete name before launching review.",
                        node_id=str(ex.get("id") or ex.get("node_id") or ""),
                        node_type="export",
                        fix_hint="Rename the bundle in the Export node's Checkpoint field.",
                    ))
                rb = str(self._node_param(ex, "review_backend", "mujoco") or "mujoco")
                if rb == "newton":
                    issues.append(ValidationIssue(
                        severity="warning",
                        section="export",
                        code="E4_newton_not_available",
                        message="Newton backend is a placeholder — Launch Review will no-op.",
                        node_id=str(ex.get("id") or ex.get("node_id") or ""),
                        node_type="export",
                        fix_hint="Switch review_backend to mujoco or isaac_sim.",
                    ))

        if not self._nodes_by_kind(graph, "robot"):
            issues.append(ValidationIssue(
                severity="warning",
                section="robot",
                code="E3_no_robot_node",
                message="No Robot node on canvas — review subprocess will run with default morphology.",
                node_type="robot",
                fix_hint="Drag a Robot node and pick an asset_id.",
            ))

        ok = not any(i.severity == "error" for i in issues)
        return RefreshResult(ok=ok, issues=issues, revision=self._revision)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_CONTEXTS: Dict[str, TrainingContext] = {}


def get_training_context(project_id: str = "") -> TrainingContext:
    """Return the per-project TrainingContext singleton.

    Empty ``project_id`` (no project loaded) maps to the ``""`` slot so
    headless / pre-project usage still works.
    """
    pid = str(project_id or "")
    ctx = _CONTEXTS.get(pid)
    if ctx is None:
        ctx = TrainingContext(project_id=pid)
        _CONTEXTS[pid] = ctx
    return ctx


def reset_training_contexts() -> None:
    """Test-only — drop all cached contexts."""
    _CONTEXTS.clear()


__all__ = [
    "COMPAT_STATUSES",
    "ISSUE_SEVERITIES",
    "CompatField",
    "StartPointCompatReport",
    "ValidationIssue",
    "RefreshResult",
    "TrainingContext",
    "get_training_context",
    "reset_training_contexts",
]
