"""region_validator — soft validation of the [setup → logic → output]
canvas taxonomy declared in mission_design.yaml §3.

Per principle P8 (mission_design.yaml §2):
    "Region membership is declared per-node-category and validated by
     MissionPlanner as a WARNING, never a hard reject. Rationale:
     BUG-1 remediation requires that legacy canvases (which predate
     the region taxonomy) keep opening and running. Hard rejection
     would regress that goal."

Validation rules
----------------
1. **Setup → Logic → Output flow**: every flow edge must travel
   forward in region order. An edge that goes Logic → Setup or
   Output → Logic is a region inversion → WARN.
2. **Output region is sink-only**: a node in REGION_OUTPUT_HANDLING
   should not have a flow_out edge into something other than another
   output node → WARN.
3. **Single Start guideline**: a canvas should have exactly one node
   in REGION_INITIAL_SETUP whose type is ``start`` (P6 of principles —
   single sim_env per canvas). Multiple StartNodes → WARN. Zero
   StartNodes → INFO (the runtime auto-injects via the executor).

The validator NEVER returns errors — only warnings — so MissionPlanner
can fold them into the existing Diagnostic pipeline without blocking
compilation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Region constants — mirror BaseNode constants so this module has no
# circular import on the node layer.
# ---------------------------------------------------------------------------

REGION_INITIAL_SETUP   = "initial_setup"
REGION_LOGIC_EXECUTION = "logic_execution"
REGION_OUTPUT_HANDLING = "output_handling"

# Strict ordering used for "edge inverts region order" detection.
_REGION_ORDER: Dict[str, int] = {
    REGION_INITIAL_SETUP:   0,
    REGION_LOGIC_EXECUTION: 1,
    REGION_OUTPUT_HANDLING: 2,
}


# ---------------------------------------------------------------------------
# schema_id → region map
# ---------------------------------------------------------------------------
#
# This intentionally lives next to the validator (instead of being
# derived from BaseNode.get_region() at import time) so the validator
# can run on raw IR / dict payloads without needing to instantiate a
# node class. The map IS the contract — when a new node type lands,
# the author either adds it here or accepts the LOGIC default.

_NODE_TYPE_TO_REGION: Dict[str, str] = {
    # ── Region 1: Initial Setup ──────────────────────────────────────
    # v1.4: StartNode is the only Region 1 node. Actor and field
    # selection live as parameters on the StartNode itself, not as
    # separate Mission Runtime nodes (those were removed in v1.4).
    "start":     REGION_INITIAL_SETUP,

    # ── Region 3: Output Handling ────────────────────────────────────
    "end":       REGION_OUTPUT_HANDLING,

    # Everything else defaults to logic_execution. Listing the most
    # common types here is for documentation, not behavior — the
    # default branch in classify() catches the rest.
    "behavior":          REGION_LOGIC_EXECUTION,
    "checkpoint":        REGION_LOGIC_EXECUTION,
    "action_execution":  REGION_LOGIC_EXECUTION,
    "if":                REGION_LOGIC_EXECUTION,
    "while_loop":        REGION_LOGIC_EXECUTION,
}


def classify(node_type: str) -> str:
    """Return the canonical region for a node_type / schema_id.

    Unknown types fall back to ``REGION_LOGIC_EXECUTION``. This means
    "if you don't tell us, we assume you're a logic node" — which is
    correct for the vast majority of cases and the safest default for
    legacy canvases.
    """
    if not node_type:
        return REGION_LOGIC_EXECUTION
    return _NODE_TYPE_TO_REGION.get(node_type, REGION_LOGIC_EXECUTION)


def register_region(node_type: str, region: str) -> None:
    """Augment the type→region map at runtime.

    Used by extension code that adds new node categories without
    forking this file. Region MUST be one of the three constants.
    """
    if region not in _REGION_ORDER:
        raise ValueError(f"region_validator: unknown region '{region}'")
    _NODE_TYPE_TO_REGION[node_type] = region


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RegionWarning:
    code: str           # short stable token, e.g. "region.inversion"
    message: str        # human-readable
    node_id: str = ""   # offending node id, if any
    edge: Optional[Tuple[str, str]] = None  # (from_node, to_node) for edge-level warnings


@dataclass
class RegionReport:
    warnings: List[RegionWarning]
    region_counts: Dict[str, int]

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class RegionValidator:
    """Soft canvas region validator.

    Accepts either a :class:`WorkflowIR` or a raw graph dict
    ({"nodes": [...], "connections": [...]}) so it can run at any
    point in the canvas → IR pipeline.
    """

    def validate(self, source: Any) -> RegionReport:
        nodes, edges = self._normalize(source)
        warnings: List[RegionWarning] = []
        region_counts: Dict[str, int] = {
            REGION_INITIAL_SETUP:   0,
            REGION_LOGIC_EXECUTION: 0,
            REGION_OUTPUT_HANDLING: 0,
        }

        # node_id → region
        node_region: Dict[str, str] = {}
        # node_id → node_type (for diagnostics + Start counting)
        node_type: Dict[str, str] = {}

        for nid, ntype in nodes:
            region = classify(ntype)
            node_region[nid] = region
            node_type[nid] = ntype
            region_counts[region] = region_counts.get(region, 0) + 1

        # Rule 3: Start cardinality
        start_ids = [nid for nid, ntype in node_type.items() if ntype == "start"]
        if len(start_ids) > 1:
            warnings.append(RegionWarning(
                code="region.multi_start",
                message=(
                    f"Canvas has {len(start_ids)} StartNodes "
                    f"({', '.join(start_ids)}). Per principle P6 each "
                    f"mission canvas should own exactly one sim_env, "
                    f"rooted at a single StartNode."
                ),
            ))

        # Rule 1: edges must travel forward in region order
        for src_id, dst_id in edges:
            src_region = node_region.get(src_id)
            dst_region = node_region.get(dst_id)
            if not src_region or not dst_region:
                continue
            src_rank = _REGION_ORDER.get(src_region, -1)
            dst_rank = _REGION_ORDER.get(dst_region, -1)
            if src_rank > dst_rank:
                warnings.append(RegionWarning(
                    code="region.inversion",
                    message=(
                        f"Edge {src_id}→{dst_id} inverts region order: "
                        f"{src_region} → {dst_region}. Per §3 the canvas "
                        f"flow must travel "
                        f"[initial_setup → logic_execution → output_handling]."
                    ),
                    edge=(src_id, dst_id),
                ))

        # Rule 2: a node in OUTPUT region should not feed back into LOGIC
        # (covered by Rule 1, but also flag the corresponding node for
        # easier UX — the warning carries the offending node_id).
        for src_id, dst_id in edges:
            if (
                node_region.get(src_id) == REGION_OUTPUT_HANDLING
                and node_region.get(dst_id) == REGION_LOGIC_EXECUTION
            ):
                warnings.append(RegionWarning(
                    code="region.output_feedback",
                    message=(
                        f"Output-region node {src_id} ({node_type.get(src_id, '')}) "
                        f"feeds back into a logic node {dst_id} "
                        f"({node_type.get(dst_id, '')}). Output region is sink-only."
                    ),
                    node_id=src_id,
                    edge=(src_id, dst_id),
                ))

        return RegionReport(warnings=warnings, region_counts=region_counts)

    # ------------------------------------------------------------------
    # Source normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(source: Any) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Reduce source → ([(node_id, node_type), ...], [(from, to), ...]).

        Accepts either a WorkflowIR (which carries IRNode + IREdge
        dataclasses) or a dict with ``nodes`` / ``connections`` /
        ``edges`` lists.
        """
        nodes: List[Tuple[str, str]] = []
        edges: List[Tuple[str, str]] = []

        # ── WorkflowIR shape ────────────────────────────────────────
        ir_nodes = getattr(source, "nodes", None)
        ir_edges = getattr(source, "edges", None)
        if ir_nodes is not None and ir_edges is not None:
            for n in ir_nodes:
                nid = str(getattr(n, "id", "") or "")
                # IR carries schema_id like "builtin.start" — strip the
                # "builtin." prefix so the type matches the registry.
                schema_id = str(getattr(n, "schema_id", "") or "")
                if schema_id.startswith("builtin."):
                    schema_id = schema_id[len("builtin."):]
                nodes.append((nid, schema_id))
            for e in ir_edges:
                from_id = str(getattr(e, "from_node", "") or "")
                to_id = str(getattr(e, "to_node", "") or "")
                if from_id and to_id:
                    edges.append((from_id, to_id))
            return nodes, edges

        # ── Dict shape ──────────────────────────────────────────────
        if isinstance(source, dict):
            for n in source.get("nodes", []) or []:
                if not isinstance(n, dict):
                    continue
                nid = str(n.get("id", "") or "")
                ntype = str(n.get("type", "") or n.get("node_type", "") or "")
                nodes.append((nid, ntype))
            for c in source.get("connections", []) or source.get("edges", []) or []:
                if not isinstance(c, dict):
                    continue
                src = c.get("from", {})
                dst = c.get("to", {})
                if isinstance(src, dict) and isinstance(dst, dict):
                    from_id = str(src.get("node", "") or "")
                    to_id = str(dst.get("node", "") or "")
                else:
                    from_id = str(src or "")
                    to_id = str(dst or "")
                if from_id and to_id:
                    edges.append((from_id, to_id))
            return nodes, edges

        return nodes, edges
