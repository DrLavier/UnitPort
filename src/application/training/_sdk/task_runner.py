"""TrainingTask — base ``unitport_sdk.Task`` for long-running training work.

Replaces DEMO's raw ``QThread`` patterns in ``sb3_trainer.py`` /
``isaac_lab_backend.py`` etc. The SDK's ``Task`` already provides:

- ``self.sleep(s)``        — interruptible (raises on cancel during sleep)
- ``self.check_cancelled()`` — raises ``TaskCancelledException``
- ``self.progress_line(ratio, text)`` — UI progress bar update
- thread-safe ``self.log_*`` (auto-prefixed with ``[task.name]``)

Subclasses MUST:

1. Resolve the active project before calling ``super().__init__`` and pass
   the fully-resolved ``run_dir = <project>/training/runs/<backend>/<run_id>/``.
   Unbound training is rejected at submit time — write paths under
   ``Paths.USER_CONFIG_DIR`` are no longer accepted (RELEASE/CLAUDE.md §1.4).
2. Call ``self.check_cancelled()`` at every rollout / iteration boundary.
3. Use ``self.sleep(s)`` — never ``time.sleep`` (breaks ``cancel()``).
4. Write artifacts under ``self.run_dir`` via ``checkpoint_io`` (atomic writes).
5. Fail loudly: raise on any state that breaks the training contract; the SDK
   catches it and emits ``task_finished(success=False, ...)``.
"""

from __future__ import annotations

from pathlib import Path

from unitport_sdk import Task

from .logging_bridge import train_log_info


class TrainingTask(Task):
    """Base class for any long-running training/eval/export job.

    Args:
        name:    Human-readable task name shown in TaskSlot UI.
        run_id:  Unique run identifier.
        run_dir: Absolute path to the run's I/O anchor — must be
                 ``<project>/training/runs/<backend_id>/<run_id>/`` and
                 already exist (subclass mkdirs it before calling super).
    """

    def __init__(self, name: str, run_id: str, run_dir: Path):
        super().__init__(name)
        rd = Path(run_dir)
        if not rd.is_absolute():
            raise ValueError(
                f"TrainingTask.run_dir must be absolute (got {rd!r}); "
                "subclass should resolve <project>/training/runs/<backend>/<run_id>/ "
                "before calling super().__init__()."
            )
        self.run_id: str = str(run_id)
        self.run_dir: Path = rd

    # Convenience wrappers — subclasses can use the SDK's self.log_* directly,
    # but train_log_* prefixes with [run_id] which is filterable in CmdLogWidget.
    def info(self, msg: str) -> None:
        train_log_info(self.run_id, msg)

    # NOTE: subclasses override ``run()`` from Task. The implementation must
    # honour the cancellation contract described in this module's docstring.


__all__ = ["TrainingTask"]
