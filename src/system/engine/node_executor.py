#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime node executor."""

import threading as _threading
import time as _time
import uuid as _uuid

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

try:
    from src.system.core.logger import log_info, log_error, log_debug, log_warning
except (ImportError, AttributeError):
    # Fallback for test/standalone environments where PySide6 / Qt is unavailable.
    def log_info(*a, **k): pass    # type: ignore[misc]
    def log_error(*a, **k): pass   # type: ignore[misc]
    def log_debug(*a, **k): pass   # type: ignore[misc]
    def log_warning(*a, **k): pass # type: ignore[misc]

from .result_inspector import is_failure_result


# ---------------------------------------------------------------------------
# Step 8 — Executor dispatch helpers
# ---------------------------------------------------------------------------

def _select_executor(brand: str, backend: str):
    """Return the appropriate OperationExecutor for (brand, backend), or None.

    Returns None when no executor is available; callers must fall through to
    the legacy raw-action path in that case.
    """
    from src.system.binding.executors import UnitreeSDKExecutor, Go2MujocoExecutor, SpotSDKExecutor, SpotMujocoExecutor
    from src.system.binding.output import BACKEND_SDK, BACKEND_MUJOCO
    b = (brand or "").lower()
    if b == "unitree":
        if backend == BACKEND_SDK:
            return UnitreeSDKExecutor()
        if backend == BACKEND_MUJOCO:
            return Go2MujocoExecutor()
    if b == "boston_dynamics":
        if backend == BACKEND_SDK:
            return SpotSDKExecutor()
        if backend == BACKEND_MUJOCO:
            return SpotMujocoExecutor()
    return None


def _dispatch_via_executor(binding_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a pre-resolved BindingOutput through the appropriate executor.

    Returns a node-result dict compatible with the action_execution node format:
      - success     → ``{'flow_out': {'status': 'success', ...}}``
      - unsupported → ``{'flow_out': {'status': 'unsupported', ...}}``  (graceful)
      - failed      → ``{'error': reason, 'flow_out': {'status': 'failed', ...}}``

    If no executor is found for the brand/backend combination, returns an
    'unsupported' result so the node degrades gracefully without blocking.
    """
    executor = _select_executor(binding_output.brand, binding_output.backend)
    if executor is None:
        return {
            "flow_out": {
                "status": "unsupported",
                "diag_code": f"executor.unsupported.{binding_output.operation}",
                "reason": (
                    f"No executor for brand={binding_output.brand!r} "
                    f"backend={binding_output.backend!r}"
                ),
                "binding_name": None,
                "binding_output": binding_output.to_dict() if hasattr(binding_output, "to_dict") else {},
                "executor_name": None,
                "executor_path": True,
            }
        }

    from src.system.binding.executors.base import ExecutorResult
    exec_result: ExecutorResult = executor.execute(binding_output, context)
    executor_name = type(executor).__name__

    flow = {
        "status": "success" if exec_result.success else (
            "unsupported"
            if exec_result.diag_code.startswith("executor.unsupported.")
            else "failed"
        ),
        "operation": exec_result.operation,
        "diag_code": exec_result.diag_code,
        "reason": exec_result.reason,
        "intent_id": binding_output.intent_id,
        "binding_name": f"{binding_output.intent_id}:{binding_output.backend}",
        "binding_output": binding_output.to_dict() if hasattr(binding_output, "to_dict") else {},
        "executor_name": executor_name,
        "execution_result": "success" if exec_result.success else (
            "unsupported"
            if exec_result.diag_code.startswith("executor.unsupported.")
            else "failed"
        ),
        "executor_path": True,
    }
    if exec_result.success:
        return {"flow_out": flow}
    if exec_result.diag_code.startswith("executor.unsupported."):
        return {"flow_out": flow}
    # executor.failed — surface as node error so invoker can detect it
    return {"error": exec_result.reason, "flow_out": flow}


class NodeExecutor:
    """Execute a DAG/workflow node graph.

    Execution modes
    ---------------
    Primary  : flow-aware traversal  — respects FLOW/DATA edge types, conditional
               branching, loop repetition, and stop/abort semantics.
    Fallback : topological sort      — used when the flow graph yields no entry
               nodes (e.g. pure data-only graphs or isolated node sets).

    Node execution failure policy
    ------------------------------
    - Unknown node type   → result = {"status": "skipped", "reason": "unknown_type:<t>"}
                            No exception raised; execution continues.
    - AbortWorkflow raise → sets _abort=True/_abort_reason, stops traversal;
                            node's {"error": ...} is recorded in results.
    - Any other exception → result = {"error": str(exc)}, traversal continues.

    Loop policy
    -----------
    max_loop_iterations (default 100) caps while-loop repetitions per node.
    Configured by RuntimeEngine from SafetyPolicy.from_dict() before execution.
    When the limit is hit: warning logged, _loop_limit_exceeded=True, traversal
    continues from loop_end edges (no silent discard).
    """

    # Node types that need robot_model injected (mirrors WorkflowRunner).
    _ROBOT_AWARE_TYPES: FrozenSet[str] = frozenset(
        {"action_execution", "sensor_input", "stop"}
    )

    # Node types that need sim_env injected (Phase F — checkpoint-runner path).
    _SIM_ENV_AWARE_TYPES: FrozenSet[str] = frozenset({"behavior"})

    # Output port names that identify a loop node.
    _LOOP_PORTS: FrozenSet[str] = frozenset({"loop_body", "loop_end"})

    # Conditional-branch output port names / prefixes.
    _BRANCH_PORTS: FrozenSet[str] = frozenset(
        {"out_if", "out_else", "out_pass", "out_block"}
    )

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.connections: List[Dict[str, Any]] = []
        self.execution_order: List[str] = []

        # Policy / config — set by RuntimeEngine before each call to execute().
        self.max_loop_iterations: int = 100
        # Migration switch (STEP-05): True → flow-aware DFS (new, default);
        # False → topological sort fallback (legacy compat path).
        # Set by RuntimeEngine from MigrationFlags.from_env() before execution.
        self.use_flow_aware_execution: bool = True

        # Behavior dispatch (Phase 2 STAGE-03).
        # Set by RuntimeEngine before execution when behavior nodes are present.
        # If _behavior_bridge is None, behavior nodes return a "skipped" result
        # (backward-compatible: no error raised, non-behavior nodes unaffected).
        self._behavior_invoker: Optional[Any] = None   # BehaviorSubgraphInvoker
        self._behavior_bridge: Optional[Any] = None    # BehaviorCompilerBridge
        self._behavior_policy: Optional[Any] = None    # SafetyPolicy

        # Runtime state — reset by execute() and clear().
        self._abort: bool = False
        self._abort_reason: str = ""
        self._last_executed_count: int = 0
        self._loop_limit_exceeded: bool = False
        # Optional per-node lifecycle callback set by caller before execute().
        # Signature: callback(node_id: str, status: str) where
        # status ∈ {"running", "success", "failed"}.  None → no-op.
        self._node_status_callback = None
        # Cycle 2 STAGE-06: optional cancel check injected by RuntimeEngine.
        # Signature: () -> bool.  Returns True when cancellation is requested.
        # Checked at the start of each node visit — None means no cancel check.
        self._cancel_fn = None
        self._cancelled: bool = False

    # ------------------------------------------------------------------
    # Public graph-building API
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, node_type: str, node_data: Dict[str, Any]):
        node = {
            "id": node_id,
            "type": node_type,
            "data": node_data,
            "inputs": [],
            "outputs": [],
        }
        self.nodes.append(node)
        log_debug(f"add node: {node_id} ({node_type})")

    def add_connection(
        self,
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str,
        edge_type: str = "flow",
    ):
        """Add a directed edge between nodes.

        edge_type:
          "flow" — control-flow edge; determines traversal order and which
                   downstream nodes are visited (default, backward-compatible).
          "data" — data-flow edge; carries port values between nodes but does
                   not itself trigger node visitation.
        """
        connection = {
            "from": {"node": from_node, "port": from_port},
            "to": {"node": to_node, "port": to_port},
            "edge_type": edge_type,
        }
        self.connections.append(connection)
        log_debug(
            f"add connection: {from_node}.{from_port} -> {to_node}.{to_port}"
            f" [{edge_type}]"
        )

    # ------------------------------------------------------------------
    # Topological sort — used by to_code() and as a fallback
    # ------------------------------------------------------------------

    def build_execution_order(self) -> bool:
        graph = {node["id"]: [] for node in self.nodes}
        in_degree = {node["id"]: 0 for node in self.nodes}

        for conn in self.connections:
            from_id = conn["from"]["node"]
            to_id = conn["to"]["node"]
            if from_id in graph and to_id in graph:
                graph[from_id].append(to_id)
                in_degree[to_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: List[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            log_error("cycle detected in node graph")
            return False

        self.execution_order = result
        log_info(f"execution order: {' -> '.join(result)}")
        return True

    # ------------------------------------------------------------------
    # Primary execution entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        node_status_callback=None,
    ) -> Dict[str, Any]:
        """Execute the node graph; return per-node results.

        Uses flow-aware traversal as primary strategy.
        Falls back to topological sort when the flow graph has no entry nodes.

        Args:
            node_status_callback: Optional callable(node_id, status) fired for
                each node lifecycle event.  status ∈ {"running", "success",
                "failed"}.  When None, behaviour is unchanged (backward-compat).
        """
        context = context or {}
        self._abort = False
        self._abort_reason = ""
        self._last_executed_count = 0
        self._loop_limit_exceeded = False
        self._cancelled = False   # Cycle 2 STAGE-06
        # Store callback for use by traversal helpers (closures access via self).
        self._node_status_callback = node_status_callback

        if not self.use_flow_aware_execution:
            # Compat fallback: topological sort (pre-STEP-04 legacy path).
            # Preserved as a controlled rollback option; set via MigrationFlags.
            return self._topological_execute_inner(context)

        return self._flow_aware_execute(context)

    # ------------------------------------------------------------------
    # Flow-aware execution (primary)
    # ------------------------------------------------------------------

    def _build_flow_graph(
        self,
    ) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, int]]:
        """Build FLOW-edge adjacency map and in-degree counts.

        Returns
        -------
        flow_out     : {node_id: [(from_port, to_node_id), ...]}
        flow_in_count: {node_id: int}   — FLOW in-degree only
        """
        flow_out: Dict[str, List[Tuple[str, str]]] = {}
        flow_in_count: Dict[str, int] = {n["id"]: 0 for n in self.nodes}

        for conn in self.connections:
            fn = conn["from"]["node"]
            tn = conn["to"]["node"]
            fp = conn["from"]["port"]
            if conn.get("edge_type", "flow") == "flow":
                if fn in flow_in_count and tn in flow_in_count:
                    flow_out.setdefault(fn, []).append((fp, tn))
                    flow_in_count[tn] += 1

        return flow_out, flow_in_count

    def _flow_aware_execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """DFS traversal following active FLOW edges.

        Entry nodes  : FLOW in-degree == 0, excluding pure data-provider kinds
                       ("comparison", "end") which are invoked on-demand via
                       DATA edges.
        Branching    : conditional nodes (if/switch/gate) — only non-None
                       output ports lead to downstream visitation.
        Loops        : while/for — handled by _handle_loop(); capped at
                       max_loop_iterations; loop_end edges run after loop exits.
        Abort        : AbortNode raises RuntimeError("AbortWorkflow: …");
                       _abort=True stops all further visitation.
        """
        flow_out, flow_in_count = self._build_flow_graph()

        # Pure data-provider nodes should not be entry points:
        # comparison/checkpoint nodes are invoked on-demand via DATA edges;
        # end nodes have no outgoing flow by definition.
        _SKIP_AS_ENTRY: FrozenSet[str] = frozenset({"comparison", "checkpoint", "end"})
        entry_nodes = [
            n["id"]
            for n in self.nodes
            if flow_in_count.get(n["id"], 0) == 0
            and n["data"].get("kind") not in _SKIP_AS_ENTRY
        ]

        if not entry_nodes:
            log_warning(
                "flow graph has no entry nodes; falling back to topological sort"
            )
            return self._topological_execute_inner(context)

        results: Dict[str, Any] = {}
        visited: set = set()

        # ── inner helpers (closures over results / visited / self) ──────

        def visit(node_id: str, loop_depth: int = 0) -> None:
            if self._abort or node_id in visited:
                return
            # Cycle 2 STAGE-06: stop traversal when cancellation is requested.
            if self._cancel_fn is not None and self._cancel_fn():
                self._cancelled = True
                return
            node = self._find_node(node_id)
            if not node:
                log_warning(f"node not found: {node_id}")
                return

            # Resolve DATA-edge dependencies before executing this node.
            # Mirrors WorkflowRunner._evaluate_condition() on-demand strategy:
            # condition/data provider nodes execute when their consumer needs them,
            # not based on flow order alone.
            for conn in self.connections:
                if (
                    conn["to"]["node"] == node_id
                    and conn.get("edge_type", "flow") == "data"
                    and conn["from"]["node"] not in visited
                ):
                    visit(conn["from"]["node"], loop_depth)

            visited.add(node_id)
            self._last_executed_count += 1
            log_info(f"execute node: {node_id} ({node['type']})")

            # Notify caller that this node is now running (Cycle 1 STAGE-03).
            _cb = self._node_status_callback
            if _cb is not None:
                _cb(node_id, "running")

            inputs = self._collect_inputs(node_id, results)
            try:
                output = self._execute_node(node, inputs, context)
                results[node_id] = output
                if is_failure_result(output):
                    log_error(f"node {node_id} output indicates failure: {output}")
                    if _cb is not None:
                        _cb(node_id, "failed")
                else:
                    log_debug(f"node {node_id} output: {output}")
                    if _cb is not None:
                        _cb(node_id, "success")
            except RuntimeError as exc:
                msg = str(exc)
                results[node_id] = {"error": msg}
                log_error(f"node {node_id} failed: {exc}")
                if _cb is not None:
                    _cb(node_id, "failed")
                if "AbortWorkflow" in msg:
                    self._abort = True
                    self._abort_reason = msg
                return
            except Exception as exc:
                log_error(f"node {node_id} failed: {exc}")
                results[node_id] = {"error": str(exc)}
                if _cb is not None:
                    _cb(node_id, "failed")
                return

            if self._abort:
                return

            out_edges = flow_out.get(node_id, [])
            if not out_edges:
                return

            output_ports = set(output.keys()) if isinstance(output, dict) else set()

            # ── loop node: has loop_body / loop_end output ports ────────
            if output_ports & self._LOOP_PORTS:
                _handle_loop(node_id, node, output, out_edges, loop_depth)
                return

            # ── try/catch node ─────────────────────────────────────────
            if "try_body" in output_ports and "except_body" in output_ports:
                _handle_try_catch(node_id, node, output, out_edges, loop_depth)
                return

            # ── parallel branch node ───────────────────────────────────
            if any(p.startswith("branch_") for p in output_ports):
                _handle_parallel(node_id, node, output, out_edges, loop_depth)
                return

            # ── retry node ─────────────────────────────────────────────
            if "retry_body" in output_ports:
                _handle_retry(node_id, node, output, out_edges, loop_depth)
                return

            # ── timeout node ───────────────────────────────────────────
            if "out_timeout" in output_ports:
                _handle_timeout(node_id, node, output, out_edges, loop_depth)
                return

            # ── conditional node: follow only non-None output ports ─────
            is_conditional = bool(output_ports & self._BRANCH_PORTS) or any(
                p.startswith("case_") for p, _ in out_edges
            )
            if is_conditional:
                for fp, tn in out_edges:
                    if isinstance(output, dict) and output.get(fp) is not None:
                        visit(tn, loop_depth)
                return

            # ── regular node: follow all downstream flow edges ──────────
            # WorkflowRunner follows flow_out edges unconditionally for
            # non-conditional nodes; port output values are irrelevant here.
            for _fp, tn in out_edges:
                visit(tn, loop_depth)

        def _handle_loop(
            node_id: str,
            node: Dict[str, Any],
            initial_output: Dict[str, Any],
            out_edges: List[Tuple[str, str]],
            loop_depth: int,
        ) -> None:
            """Drive loop-body iteration; cap while loops at max_loop_iterations."""
            _MAX_NESTING = 8
            if loop_depth >= _MAX_NESTING:
                log_warning(
                    f"max loop nesting depth ({_MAX_NESTING}) exceeded at"
                    f" node {node_id}; aborting loop"
                )
                return

            body_targets = [tn for fp, tn in out_edges if fp == "loop_body"]
            end_targets  = [tn for fp, tn in out_edges if fp == "loop_end"]

            params = node["data"].get("params", {})

            def get_param(name: str, default: Any) -> Any:
                p = params.get(name)
                if p is None:
                    return default
                return p.get("value", default) if isinstance(p, dict) else p

            loop_type = get_param("loop_type", "while")

            if loop_type == "for":
                try:
                    start = int(get_param("for_start", 0))
                    end   = int(get_param("for_end",   1))
                    step  = int(get_param("for_step",  1)) or 1
                except (ValueError, TypeError):
                    start, end, step = 0, 1, 1
                log_debug(f"for loop at {node_id}: range({start}, {end}, {step})")
                for i in range(start, end, step):
                    if self._abort or self._cancelled:
                        break
                    results[f"{node_id}._i"] = i
                    for tn in body_targets:
                        visited.discard(tn)
                        visit(tn, loop_depth + 1)
            else:
                # While loop — re-evaluate loop node each iteration.
                iteration = 0
                current_output = initial_output
                while (
                    isinstance(current_output, dict)
                    and current_output.get("loop_body") is not None
                    and iteration < self.max_loop_iterations
                ):
                    if self._abort or self._cancelled:
                        break
                    iteration += 1
                    for tn in body_targets:
                        visited.discard(tn)
                        visit(tn, loop_depth + 1)

                    # Re-evaluate the loop condition for next iteration.
                    re_inputs = self._collect_inputs(node_id, results)
                    try:
                        current_output = self._execute_node(node, re_inputs, context)
                        results[node_id] = current_output
                    except Exception as exc:
                        log_error(
                            f"loop re-evaluation at {node_id} failed: {exc}"
                        )
                        break

                if iteration >= self.max_loop_iterations:
                    log_warning(
                        f"while loop at {node_id} reached max_loop_iterations"
                        f"={self.max_loop_iterations}; exiting loop"
                    )
                    self._loop_limit_exceeded = True

            # Always follow loop_end edges after the loop exits.
            for tn in end_targets:
                visit(tn, loop_depth)

        def _get_node_param(node: Dict[str, Any], name: str, default: Any) -> Any:
            """Extract a parameter value from node data (handles IRParam dicts)."""
            params = node["data"].get("params", {})
            p = params.get(name)
            if p is None:
                return default
            return p.get("value", default) if isinstance(p, dict) else p

        def _handle_try_catch(
            node_id: str,
            node: Dict[str, Any],
            output: Dict[str, Any],
            out_edges: List[Tuple[str, str]],
            loop_depth: int,
        ) -> None:
            """Execute try body; on failure route to except; always run finally."""
            try_targets = [tn for fp, tn in out_edges if fp == "try_body"]
            except_targets = [tn for fp, tn in out_edges if fp == "except_body"]
            finally_targets = [tn for fp, tn in out_edges if fp == "finally_body"]
            flow_targets = [tn for fp, tn in out_edges if fp == "flow_out"]

            # Snapshot error state before try body.
            pre_errors = {
                nid for nid, r in results.items() if is_failure_result(r)
            }
            had_abort = self._abort

            # Execute try body branch.
            for tn in try_targets:
                visit(tn, loop_depth)

            # Detect failures that occurred during try body.
            post_errors = {
                nid for nid, r in results.items() if is_failure_result(r)
            }
            new_errors = post_errors - pre_errors
            caught_abort = self._abort and not had_abort

            if new_errors or caught_abort:
                # Suppress abort so except/finally can run.
                if caught_abort:
                    self._abort = False
                # Store caught error info for except body nodes to access.
                results[f"{node_id}._caught_error"] = {
                    "failed_nodes": list(new_errors),
                    "abort_reason": self._abort_reason if caught_abort else "",
                }
                for tn in except_targets:
                    visited.discard(tn)
                    visit(tn, loop_depth)

            # Always run finally body (unless hard-cancelled).
            if not self._cancelled:
                for tn in finally_targets:
                    visited.discard(tn)
                    visit(tn, loop_depth)

            # Continue to flow_out.
            if not self._abort and not self._cancelled:
                for tn in flow_targets:
                    visit(tn, loop_depth)

        def _handle_parallel(
            node_id: str,
            node: Dict[str, Any],
            output: Dict[str, Any],
            out_edges: List[Tuple[str, str]],
            loop_depth: int,
        ) -> None:
            """Fork execution: run branches concurrently, then follow join_out."""
            branch_targets = [
                (fp, tn) for fp, tn in out_edges if fp.startswith("branch_")
            ]
            join_targets = [
                tn for fp, tn in out_edges
                if fp == "join_out" or fp == "flow_out"
            ]

            if not branch_targets:
                return

            if len(branch_targets) == 1:
                # Single branch — no need for threads.
                _fp, tn = branch_targets[0]
                visit(tn, loop_depth)
            else:
                # Multiple branches — execute in parallel threads.
                # CPython GIL ensures dict/set bytecode ops are atomic;
                # branches should target independent subgraphs.
                branch_errors: list = []

                def _run_branch(target_id: str, port: str) -> None:
                    try:
                        visit(target_id, loop_depth)
                    except Exception as exc:
                        branch_errors.append((port, str(exc)))

                threads = []
                for fp, tn in branch_targets:
                    visited.discard(tn)
                    t = _threading.Thread(
                        target=_run_branch, args=(tn, fp), daemon=True,
                    )
                    threads.append(t)

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                if branch_errors:
                    log_warning(
                        f"parallel node {node_id}: branch errors: {branch_errors}"
                    )

            # Aggregate branch results for downstream JoinNode.
            results[f"{node_id}._parallel"] = {
                "branch_count": len(branch_targets),
                "completed": True,
            }

            for tn in join_targets:
                visit(tn, loop_depth)

        def _handle_retry(
            node_id: str,
            node: Dict[str, Any],
            output: Dict[str, Any],
            out_edges: List[Tuple[str, str]],
            loop_depth: int,
        ) -> None:
            """Re-execute body up to max_attempts; route to success or failed."""
            body_targets = [tn for fp, tn in out_edges if fp == "retry_body"]
            success_targets = [tn for fp, tn in out_edges if fp == "out_success"]
            failed_targets = [tn for fp, tn in out_edges if fp == "out_failed"]

            max_attempts = int(_get_node_param(node, "max_attempts", 3))
            backoff_sec = float(_get_node_param(node, "backoff_sec", 1.0))
            strategy = _get_node_param(node, "strategy", "fixed")

            succeeded = False
            attempt = 0
            for attempt in range(1, max_attempts + 1):
                if self._abort or self._cancelled:
                    break

                pre_errors = {
                    nid for nid, r in results.items() if is_failure_result(r)
                }

                for tn in body_targets:
                    visited.discard(tn)
                    visit(tn, loop_depth)

                post_errors = {
                    nid for nid, r in results.items() if is_failure_result(r)
                }
                if not (post_errors - pre_errors):
                    succeeded = True
                    break

                # Backoff before next attempt.
                if attempt < max_attempts:
                    if strategy == "exp":
                        delay = backoff_sec * (2 ** (attempt - 1))
                    else:
                        delay = backoff_sec
                    _time.sleep(min(delay, 30))

            results[f"{node_id}._retry"] = {
                "attempts": attempt,
                "succeeded": succeeded,
            }

            if succeeded:
                for tn in success_targets:
                    visit(tn, loop_depth)
            else:
                for tn in failed_targets:
                    visit(tn, loop_depth)

        def _handle_timeout(
            node_id: str,
            node: Dict[str, Any],
            output: Dict[str, Any],
            out_edges: List[Tuple[str, str]],
            loop_depth: int,
        ) -> None:
            """Run body with a deadline; route to out_timeout on expiry."""
            body_targets = [tn for fp, tn in out_edges if fp == "flow_out"]
            timeout_targets = [tn for fp, tn in out_edges if fp == "out_timeout"]

            duration = float(_get_node_param(node, "duration", 5.0))
            duration = max(0.1, min(duration, 300))

            body_done = _threading.Event()

            def _run_body() -> None:
                try:
                    for tn in body_targets:
                        visit(tn, loop_depth)
                finally:
                    body_done.set()

            t = _threading.Thread(target=_run_body, daemon=True)
            t.start()

            if body_done.wait(timeout=duration):
                # Body completed within deadline — results already recorded.
                pass
            else:
                # Deadline exceeded — follow timeout branch.
                log_warning(
                    f"timeout node {node_id}: body exceeded {duration}s deadline"
                )
                results[f"{node_id}._timed_out"] = True
                for tn in timeout_targets:
                    visited.discard(tn)
                    visit(tn, loop_depth)
                # Allow body thread to finish in background.
                t.join(timeout=1.0)

        # ── start DFS from entry nodes ──────────────────────────────────
        for entry_id in entry_nodes:
            visit(entry_id)

        return results

    # ------------------------------------------------------------------
    # Topological sort execution (fallback)
    # ------------------------------------------------------------------

    def _topological_execute_inner(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: execute nodes in topological order.

        Raises RuntimeError with reason "node_graph_cycle" if a cycle is detected.
        """
        if not self.execution_order:
            if not self.build_execution_order():
                raise RuntimeError(
                    "node_graph_cycle: cycle detected;"
                    " topological execution is impossible"
                )

        results: Dict[str, Any] = {}
        _cb = self._node_status_callback
        for node_id in self.execution_order:
            node = self._find_node(node_id)
            if not node:
                log_warning(f"node not found: {node_id}")
                continue
            log_info(f"execute node: {node_id} ({node['type']})")
            if _cb is not None:
                _cb(node_id, "running")
            inputs = self._collect_inputs(node_id, results)
            try:
                output = self._execute_node(node, inputs, context)
                results[node_id] = output
                self._last_executed_count += 1
                if is_failure_result(output):
                    log_error(f"node {node_id} output indicates failure: {output}")
                    if _cb is not None:
                        _cb(node_id, "failed")
                else:
                    log_debug(f"node {node_id} output: {output}")
                    if _cb is not None:
                        _cb(node_id, "success")
            except Exception as exc:
                log_error(f"node {node_id} failed: {exc}")
                results[node_id] = {"error": str(exc)}
                if _cb is not None:
                    _cb(node_id, "failed")

        return results

    # ------------------------------------------------------------------
    # Node finding and input collection
    # ------------------------------------------------------------------

    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None

    def _collect_inputs(
        self, node_id: str, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect inputs from ALL upstream connections (FLOW and DATA).

        Both edge types carry their from_port value into the target's to_port.
        This matches WorkflowRunner behaviour where incoming data + flow tokens
        are both available as inputs to a node.

        Condition-port extraction
        ------------------------
        WorkflowRunner._evaluate_condition() extracts the raw boolean from
        upstream results via `source_result.get("result",{}).get("value", False)`.
        IfNode / WhileLoopNode.execute() expect `inputs["condition"]` to be a
        raw bool-compatible value, not a wrapped dict.

        To align: when the destination port is "condition" and the value is a
        single-key {"value": X} dict, extract X directly.  This covers both
        ComparisonNode output (`{"result": {"value": bool}}` → port "result"
        carries `{"value": bool}`) and VariableNode output similarly.
        """
        inputs: Dict[str, Any] = {}
        for conn in self.connections:
            if conn["to"]["node"] == node_id:
                from_id   = conn["from"]["node"]
                from_port = conn["from"]["port"]
                to_port   = conn["to"]["port"]
                if from_id in results:
                    src = results[from_id]
                    val = src.get(from_port) if isinstance(src, dict) else src
                    # Align with WorkflowRunner condition extraction:
                    # unwrap {"value": X} on the condition port so IfNode /
                    # WhileLoopNode receive a plain bool-compatible value.
                    if to_port == "condition" and isinstance(val, dict) and "value" in val:
                        val = val["value"]
                    inputs[to_port] = val
        return inputs

    # ------------------------------------------------------------------
    # Node dispatch (schema_id → registry lookup + execute)
    # ------------------------------------------------------------------

    def _execute_node(
        self,
        node: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch to the registered node class and execute.

        Behavior node dispatch (Phase 2 STAGE-03):
          A node is treated as a behavior node when its schema_id is "behavior"
          OR its node_data carries external_kind="behavior" (Canvas-origin nodes).
          Behavior nodes are routed to _execute_behavior_node() before the
          normal registry path; non-behavior nodes are completely unaffected.

        Normal resolution order (node["type"] is the schema_id stored by add_node):
          1. Direct registry lookup  → exact match (e.g. "action_execution")
          2. Strip "builtin." prefix → Canvas-IR format (e.g. "builtin.if" → "if")
          3. node_data["schema_id"]  → legacy dict-payload fallback

        Failure policy:
          Unknown type   → {"status": "skipped", "reason": "unknown_type:<t>"} (no raise)
          Runtime error  → re-raise; caught by visit() or _topological_execute_inner()
        """
        node_type = node["type"]
        node_data = node["data"]

        # ── Behavior-call fast-path (Circle 1 Step 1.3, Phase F updated) ────
        # Route old behavior_call / external_kind="behavior" nodes to the
        # Circle 6 BehaviorSubgraphInvoker.  The new "behavior" node type
        # (BehaviorNode — Phase F checkpoint-runner) is NOT intercepted here
        # so it falls through to the registry lookup and executes normally.
        _schema_id = str(node_data.get("schema_id", "") or "").strip().lower()
        _external_kind = str(node_data.get("external_kind", "") or "").strip().lower()
        _is_new_behavior_node = node_type == "behavior" or _schema_id == "behavior"
        _is_behavior_call = (
            node_type == "behavior_call"
            or _schema_id == "behavior_call"
            or (_external_kind == "behavior" and not _is_new_behavior_node)
        )
        if _is_behavior_call:
            return self._execute_behavior_node(node, inputs, context)
        # ────────────────────────────────────────────────────────────────────

        # Circle 7: RUN_POLICY fast path — dispatch through PolicyCommandExecutor.
        # Detected by schema_id "run_policy" (Circle 6 canonical form) or the
        # node_data kind field for direct dict payloads.
        _is_run_policy = (
            node_type == "run_policy"
            or node_data.get("kind") == "run_policy"
            or node_data.get("schema_id") == "run_policy"
        )
        if _is_run_policy:
            return self._execute_run_policy_node(node, inputs, context)
        # ────────────────────────────────────────────────────────────────────

        registry_type = self._resolve_registry_type(node_type, node_data)
        if registry_type is None:
            log_warning(f"no registered class for node type '{node_type}', skipping")
            return {"status": "skipped", "reason": f"unknown_type:{node_type}"}

        from src.system.nodes import get_node_class
        node_class = get_node_class(registry_type)
        if node_class is None:
            log_warning(
                f"registry class vanished for type '{registry_type}', skipping"
            )
            return {"status": "skipped", "reason": f"unknown_type:{registry_type}"}

        logic_node = node_class(node["id"])

        # Set parameters from IRNode serialised params dict:
        # {"param_name": {"name": str, "value": Any, "param_type": str}}
        for param_name, param_info in node_data.get("params", {}).items():
            if isinstance(param_info, dict) and "value" in param_info:
                logic_node.set_parameter(param_name, param_info["value"])
            else:
                logic_node.set_parameter(param_name, param_info)

        if registry_type == "start":
            robot_model = (
                (context.get("scenario") or {}).get("robot_model")
                or context.get("robot_model")
            )
            current_robot_type = getattr(robot_model, "robot_type", "")
            if current_robot_type:
                logic_node.set_parameter("robot_type", current_robot_type)

        # Inject robot_model for robot-aware types (mirrors WorkflowRunner).
        if registry_type in self._ROBOT_AWARE_TYPES:
            robot_model = (context.get("scenario") or {}).get("robot_model")
            if robot_model is not None:
                logic_node.set_parameter("robot_model", robot_model)

        # Inject sim_env for behavior nodes (Phase F checkpoint-runner path).
        if registry_type in self._SIM_ENV_AWARE_TYPES:
            sim_env = (
                (context.get("scenario") or {}).get("sim_env")
                or context.get("sim_env")
            )
            if sim_env is not None:
                logic_node.set_parameter("sim_env", sim_env)

        # Step 8 — BindingOutput → Executor dispatch (canonical path).
        #
        # node_executor only dispatches using already-resolved BindingOutputs.
        # The old resolved_actions raw-action rewrite is completely removed.
        #
        # Dispatch rules for action_execution nodes:
        #
        #   1. action in resolved_bindings
        #        → _dispatch_via_executor() — primary path for ALL semantic intents.
        #
        #   2. action in resolved_actions but NOT in resolved_bindings
        #        → graceful unsupported (binding was missing at Step 2e; warning was
        #          already recorded by the invoker).  No SDK call is made.
        #
        #   3. action in neither map (raw action, non-semantic, non-behavior path)
        #        → fall through to logic_node.execute() below (unchanged legacy).
        #
        # Result shapes:
        #   success     → {'flow_out': {'status': 'success', ...}}
        #   unsupported → {'flow_out': {'status': 'unsupported', ...}}  (no 'error' key)
        #   failed      → {'error': ..., 'flow_out': {'status': 'failed', ...}}
        if registry_type == "action_execution":
            current_action = logic_node.get_parameter("action", "")
            resolved_bindings = context.get("resolved_bindings", None)
            resolved_actions  = context.get("resolved_actions")  or {}

            # Step 8 behavior path is active only when the invoker explicitly
            # injected the resolved_bindings key. Non-behavior / legacy paths
            # may still carry resolved_actions without any binding layer.
            if resolved_bindings is not None:
                if current_action in resolved_bindings:
                    # Primary path: executor dispatch (raw action name matches key).
                    return _dispatch_via_executor(resolved_bindings[current_action], context)

                # Timeline-generated IR nodes store raw_action in "action" but
                # resolved_bindings is keyed by intent_id.  Fall back to intent_id
                # lookup so the binding (with user-edited speed/duration) is found.
                _intent_id_param = logic_node.get_parameter("intent_id", "")
                if _intent_id_param and _intent_id_param in resolved_bindings:
                    return _dispatch_via_executor(resolved_bindings[_intent_id_param], context)

                if current_action in resolved_actions:
                    # Semantic intent but no BindingOutput — degrade gracefully.
                    return {
                        "flow_out": {
                            "status": "unsupported",
                            "diag_code": f"binding.no_binding.{current_action}",
                            "reason": f"No BindingOutput for intent {current_action!r}",
                            "executor_path": True,
                        }
                    }

        # Legacy path: when Step 8 binding dispatch is not active, retain the
        # older semantic-intent -> raw-action rewrite behaviour so non-behavior
        # contexts and pre-Step-8 callers keep working.
        if registry_type == "action_execution":
            if "resolved_bindings" not in context:
                resolved_actions = context.get("resolved_actions") or {}
                if resolved_actions:
                    current_action = logic_node.get_parameter("action", "")
                    if current_action in resolved_actions:
                        logic_node.set_parameter("action", resolved_actions[current_action])

        return logic_node.execute(inputs)

    # ------------------------------------------------------------------
    # Behavior node execution (Phase 2 STAGE-03)
    # ------------------------------------------------------------------

    def _execute_behavior_node(
        self,
        node: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch a behavior-type Mission node to BehaviorSubgraphInvoker.

        If _behavior_bridge is None (no bridge configured), returns a "skipped"
        result — not an error — so that workflows without behavior dispatch
        continue to function normally (backward compatibility).

        Result mapping
        --------------
        BehaviorInvokeOutput.status == "success"
            → {"status": "success", "behavior_ref": ..., "outputs": ...,
               "node_results": ..., "trace_id": ...}
              (no "error" key → RuntimeEngine does NOT add to failed_nodes)
        BehaviorInvokeOutput.status in ("failed", "blocked")
            → {"error": reason, "status": ..., "behavior_ref": ...,
               "diagnostics": [...], "trace_id": ...}
              ("error" key → RuntimeEngine adds node to failed_nodes)
        """
        node_data = node["data"]
        behavior_ref = self._extract_behavior_ref(node_data)

        if self._behavior_invoker is None or self._behavior_bridge is None:
            log_warning(
                f"behavior node '{node['id']}' (ref='{behavior_ref}') has no "
                "invoker/bridge configured; returning skipped"
            )
            return {
                "status": "skipped",
                "reason": f"no_behavior_invoker:{behavior_ref}",
                "compat_path": True,
            }

        from src.system.behavior.behavior_artifact import BehaviorInvokeInput

        # Prefer mission_trace_id (injected by RuntimeEngine) so all behavior
        # invocations within the same mission run share the same root trace ID.
        trace_id = (
            context.get("mission_trace_id")
            or context.get("trace_id")
            or str(_uuid.uuid4())
        )

        # Circle 2 Step 2.3: extract sdk_settings from scenario or direct context
        # so that BehaviorInvokeInput explicitly carries it to the invoker.
        sdk_settings = (
            (context.get("scenario") or {}).get("sdk_settings")
            or context.get("sdk_settings")
        )

        # Step 1: extract protocol_payload from upstream inputs (sub_dot port).
        # The sub_dot port is registered as slot "condition" on the Behavior canvas node
        # (graph_scene.py _mk_port "condition", data_type="protocol").  When the canvas
        # wires a data edge into that port, the executor receives it as inputs["condition"].
        protocol_payload = inputs.get("condition") if isinstance(inputs, dict) else None
        if not isinstance(protocol_payload, dict):
            protocol_payload = None  # guard against non-dict sentinel values

        invoke_input = BehaviorInvokeInput(
            behavior_ref=behavior_ref,
            inputs=inputs,
            context=context,
            trace_id=trace_id,
            sdk_settings=sdk_settings,       # Circle 2 Step 2.3
            protocol_payload=protocol_payload,  # Step 1
        )

        # Create a fresh sub-executor with the same policy as this executor.
        # Passing it as a parameter avoids a circular import between
        # behavior_invoker and node_executor.
        sub_executor = NodeExecutor()
        sub_executor.max_loop_iterations = self.max_loop_iterations
        sub_executor.use_flow_aware_execution = self.use_flow_aware_execution
        # Propagate behavior bridge so nested behavior nodes (if any) resolve too.
        sub_executor._behavior_invoker = self._behavior_invoker
        sub_executor._behavior_bridge = self._behavior_bridge
        sub_executor._behavior_policy = self._behavior_policy
        # Circle 2 Step 2.3: propagate cancel function so that cancellation
        # requested at the mission level propagates into behavior subgraph traversal.
        sub_executor._cancel_fn = self._cancel_fn

        invoke_result = self._behavior_invoker.invoke(
            invoke_input=invoke_input,
            bridge=self._behavior_bridge,
            sub_executor=sub_executor,
            policy=self._behavior_policy,
        )

        # Circle 2 Step 2.3: heartbeat section keys included in both success and
        # failure result dicts so _collect_behavior_tracing() can aggregate them.
        _hb_diags = [d.to_dict() for d in invoke_result.heartbeat_diagnostics]
        # Fix 4: extract timeline motor diagnostics from the artifact's diagnostics
        # (entries whose code starts with "timeline.").
        _all_diags = [d.to_dict() for d in invoke_result.diagnostics]
        _timeline_diags = [d for d in _all_diags if str(d.get("code", "")).startswith("timeline.")]
        # Step 1: surface protocol diagnostics
        _proto_diags = [d.to_dict() for d in invoke_result.protocol_diagnostics]
        # Circle 7: surface semantic resolution diagnostics
        _res_diags = [d.to_dict() for d in invoke_result.resolution_diagnostics]

        if invoke_result.status == "success":
            return {
                "status": "success",
                "behavior_ref": behavior_ref,
                "outputs": invoke_result.outputs,
                "node_results": invoke_result.node_results,
                "trace_id": invoke_result.trace_id,
                "diagnostics": _all_diags,
                # heartbeat section
                "heartbeat_status": invoke_result.heartbeat_status,
                "heartbeat_tick_count": invoke_result.heartbeat_tick_count,
                "heartbeat_diagnostics": _hb_diags,
                "heartbeat_policy": invoke_result.heartbeat_policy,
                # timeline motor diagnostics (Fix 4)
                "timeline_diagnostics": _timeline_diags,
                # Step 1: protocol ingest result
                "protocol_status": invoke_result.protocol_status,
                "protocol_diagnostics": _proto_diags,
                # Circle 1 Step 2: apply result counts
                "apply_success_count": invoke_result.apply_success_count,
                "apply_skipped_count": invoke_result.apply_skipped_count,
                "apply_failure_reasons": invoke_result.apply_failure_reasons,
                # Circle 7: semantic resolution diagnostics
                "resolution_diagnostics": _res_diags,
            }
        else:
            # "error" key triggers failed_nodes tracking in RuntimeEngine.
            return {
                "error": invoke_result.reason,
                "status": invoke_result.status,
                "behavior_ref": behavior_ref,
                "diagnostics": _all_diags,
                "trace_id": invoke_result.trace_id,
                # heartbeat section
                "heartbeat_status": invoke_result.heartbeat_status,
                "heartbeat_tick_count": invoke_result.heartbeat_tick_count,
                "heartbeat_diagnostics": _hb_diags,
                "heartbeat_policy": invoke_result.heartbeat_policy,
                # timeline motor diagnostics (Fix 4)
                "timeline_diagnostics": _timeline_diags,
                # Step 1: protocol ingest result
                "protocol_status": invoke_result.protocol_status,
                "protocol_diagnostics": _proto_diags,
                # Circle 1 Step 2: apply result counts (0 on non-success paths)
                "apply_success_count": invoke_result.apply_success_count,
                "apply_skipped_count": invoke_result.apply_skipped_count,
                "apply_failure_reasons": invoke_result.apply_failure_reasons,
                # Circle 7: semantic resolution diagnostics
                "resolution_diagnostics": _res_diags,
            }

    def _execute_run_policy_node(
        self,
        node: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch a RUN_POLICY node to PolicyCommandExecutor.

        Circle 7: connects the compiled RUN_POLICY IR node to real policy
        execution via a brand-agnostic bridge.

        Failure containment rule: any failure in PolicyCommandExecutor returns
        a failure result dict (with "error" key) without raising — the workflow
        continues from the node's downstream flow edges normally.

        Result mapping
        --------------
        success → plain result dict (no "error" key)
                  RuntimeEngine does NOT add node to failed_nodes
        failure → result dict with "error" key
                  RuntimeEngine adds node to failed_nodes; traversal continues
        """
        node_data = node["data"]
        node_payload = self._extract_run_policy_params(node_data)
        log_info(
            f"run_policy node '{node['id']}': policy_id={node_payload.get('policy_id')!r}"
        )

        from src.system.policy.policy_command_executor import PolicyCommandExecutor
        executor = PolicyCommandExecutor()
        result = executor.execute(node_payload, context)

        if result.get("status") == "success":
            return result  # no "error" key → not counted as failed
        else:
            # "error" key triggers failed_nodes tracking in RuntimeEngine;
            # returning (not raising) keeps the workflow alive.
            return {"error": result.get("message", "run_policy failed"), **result}

    @staticmethod
    def _extract_run_policy_params(node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract run_policy execution params from a node data dict.

        Handles two formats:
          - WorkflowIR-origin: node_data["params"]["<key>"]["value"]
          - Flat dict: node_data["<key>"] (top-level)
        """
        params = node_data.get("params", {})

        def _get(name: str, default: Any) -> Any:
            p = params.get(name)
            if p is None:
                return node_data.get(name, default)
            if isinstance(p, dict) and "value" in p:
                return p["value"]
            return p

        return {
            "policy_id":        _get("policy_id",        ""),
            "max_steps":        _get("max_steps",        1000),
            "velocity_command": _get("velocity_command", [0.0, 0.0, 0.0]),
            "render":           _get("render",           False),
        }

    @staticmethod
    def _extract_behavior_ref(node_data: Dict[str, Any]) -> str:
        """Extract the behavior_ref string from a behavior node's data dict.

        Handles two formats:
          - WorkflowIR origin: node_data["params"]["behavior_ref"]["value"]
          - Canvas-dict origin: node_data["behavior_ref"] (top-level string)
        """
        # WorkflowIR-origin nodes: params dict with IRParam serialised dicts.
        params = node_data.get("params", {})
        ref_info = params.get("behavior_ref")
        if ref_info is not None:
            if isinstance(ref_info, dict):
                return str(ref_info.get("value", ""))
            return str(ref_info)

        # Canvas-origin nodes: top-level "behavior_ref" key.
        top_level = node_data.get("behavior_ref")
        if top_level is not None:
            return str(top_level)

        return ""

    @staticmethod
    def _resolve_registry_type(
        node_type: str, node_data: Dict[str, Any]
    ) -> Optional[str]:
        """Map a schema_id / node_type string to the node registry key.

        Canvas WorkflowIR uses schema_id "builtin.<node_type>";
        the registry uses bare "<node_type>" keys.
        """
        from src.system.nodes import get_node_class

        if get_node_class(node_type) is not None:
            return node_type

        if node_type.startswith("builtin."):
            candidate = node_type[len("builtin."):]
            if get_node_class(candidate) is not None:
                return candidate

        schema_id = node_data.get("schema_id", "")
        if schema_id and schema_id != node_type:
            candidate = (
                schema_id[len("builtin."):] if schema_id.startswith("builtin.") else schema_id
            )
            if get_node_class(candidate) is not None:
                return candidate

        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self):
        self.nodes.clear()
        self.connections.clear()
        self.execution_order.clear()
        self._abort = False
        self._abort_reason = ""
        self._last_executed_count = 0
        self._loop_limit_exceeded = False
        self._cancelled = False   # Cycle 2 STAGE-06
        log_info("node executor cleared")

    def to_code(self) -> str:
        if not self.execution_order:
            self.build_execution_order()

        code_lines = [
            "# Auto-generated workflow code",
            "",
            "def execute_workflow():",
        ]
        for node_id in self.execution_order:
            node = self._find_node(node_id)
            if node:
                code_lines.append(f"    # {node['type']}: {node_id}")
                code_lines.append(f"    # TODO: implement {node['type']}")
                code_lines.append("")

        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    execute_workflow()")
        return "\n".join(code_lines)
