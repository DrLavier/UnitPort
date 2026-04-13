"""Non-blocking mission execution thread (Cycle 2 STAGE-06).

Wraps ``RuntimeEngine.execute()`` on a background QThread so the main UI
thread remains responsive during mission runs.

Signals
-------
node_status_changed(node_id: object, status: str)
    Emitted per-node lifecycle event.  Delivered to the main thread via Qt's
    auto (queued) connection when connected to a slot on a QObject that lives
    on the main thread.  status ∈ {"running", "success", "failed"}.

execution_finished(run_result: dict)
    Emitted once when RuntimeEngine.execute() returns, carrying the complete
    run_result dict.  Always emitted — even on cancellation or unexpected
    exception — so the UI can always restore its state.

Usage::

    thread = MissionRunThread(exec_graph, scenario, runtime_engine)
    thread.node_status_changed.connect(self._on_mission_node_status)
    thread.execution_finished.connect(self._on_mission_finished)
    thread.start()

Cancellation::

    thread.request_cancel()   # signals engine to stop at next node boundary
    thread.wait(5000)         # wait for thread to finish (optional, with timeout)

Contract / thread-safety
------------------------
- exec_graph and scenario must not be mutated by the main thread while the
  thread is running (read-only access only from the background thread).
- Qt's queued signal connections ensure all signal deliveries to main-thread
  slots happen on the main thread — no explicit locking needed in slots.
- request_cancel() is safe to call from any thread (RuntimeEngine sets a bool
  flag, which Python's GIL makes effectively atomic for single assignments).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal


class MissionRunThread(QThread):
    """Executes ``RuntimeEngine.execute()`` on a background thread.

    Cycle 3 STAGE-03 addition
    -------------------------
    Accepts an optional *run_policy* argument.  When provided, the policy is
    activated as the thread-local run-scoped policy at the start of ``run()``
    via ``RobotContext.set_run_scoped_policy()``.  This means all
    ``RobotContext.run_action / get_sensor_data / stop`` calls from within the
    worker thread use the per-run policy **without** mutating the class-level
    ``_lifecycle_policy``.  The slot is always cleared in the ``finally``
    block so no policy leaks between runs.
    """

    # Qt signals — auto connection type queues delivery to main-thread slots.
    node_status_changed = Signal(object, str)  # (node_id, status)
    execution_finished  = Signal(dict)          # run_result

    def __init__(
        self,
        exec_graph: Any,  # Circle 1: accepts WorkflowIR or exec_graph dict
        scenario: Dict[str, Any],
        runtime_engine: Any,
        run_policy: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._exec_graph  = exec_graph
        self._scenario    = scenario
        self._engine      = runtime_engine
        self._run_policy  = run_policy  # Cycle 3 STAGE-03: per-run scoped policy

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Thread entry point — called automatically by QThread.start()."""
        # Cycle 3 STAGE-03: activate run-scoped policy on this thread so all
        # RobotContext lifecycle calls use it without touching class-level state.
        if self._run_policy is not None:
            from src.system.core.robot_context import RobotContext
            RobotContext.set_run_scoped_policy(self._run_policy)

        # Clear the per-process "user closed the MuJoCo viewer" sticky
        # flag so this fresh mission run can pop its own viewer. Without
        # this, once the user closes a viewer to abort one mission, every
        # subsequent mission run would fail at the first ensure_viewer
        # call. Brand-specific reset hooks are gathered here so the
        # MissionRunThread stays brand-agnostic at the call site.
        try:
            from src.system.brand_packages.unitree.unitree_model import UnitreeModel
            UnitreeModel.reset_viewer_session()
        except Exception:
            # Brand package may be absent in trimmed builds — ignore.
            pass

        def _status_cb(node_id: object, status: str) -> None:
            # Emit via Qt signal; queued delivery to the main thread.
            self.node_status_changed.emit(node_id, status)

        try:
            result = self._engine.execute(
                self._exec_graph,
                self._scenario,
                node_status_callback=_status_cb,
            )
        except Exception as exc:
            # Last-resort safety net: the UI must always receive
            # execution_finished so it can restore its state regardless of
            # unexpected engine errors.
            result = {
                "status": "failed",
                "reason": "mission_run_thread_error",
                "node_count": 0,
                "results": {},
                "diagnostics": {
                    "executed_count": 0,
                    "failed_nodes": [],
                    "compat_path": False,
                    "error": str(exc),
                },
            }
        finally:
            # Always clear the thread-local run-scoped policy on thread exit,
            # even on cancellation or unexpected exception, so the slot is
            # never leaked between runs.
            if self._run_policy is not None:
                from src.system.core.robot_context import RobotContext
                RobotContext.clear_run_scoped_policy()

        self.execution_finished.emit(result)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def request_cancel(self) -> None:
        """Request best-effort cancellation of the in-progress run.

        Delegates to ``RuntimeEngine.request_cancel()`` which sets a flag
        checked between node executions.  Non-blocking and thread-safe.
        Does nothing if the engine has no ``request_cancel`` method (forward
        compatibility guard).
        """
        cancel_fn = getattr(self._engine, "request_cancel", None)
        if cancel_fn is not None:
            cancel_fn()
