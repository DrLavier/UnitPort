"""Task scheduling for the runtime layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import uuid


@dataclass
class Scheduler:
    """Manages scheduling and cancellation of runtime tasks."""

    _tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def schedule(self, task: Any) -> str:
        """Schedule a task and return its task ID."""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {"task": task, "status": "scheduled"}
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task. Returns True if successfully cancelled."""
        if task_id not in self._tasks:
            return False
        self._tasks[task_id]["status"] = "cancelled"
        return True

    def get_status(self, task_id: str) -> dict:
        """Return the current status of a scheduled task."""
        return self._tasks.get(task_id, {"status": "unknown"})
