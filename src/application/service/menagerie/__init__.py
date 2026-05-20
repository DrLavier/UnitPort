"""Sidebar-side Menagerie tasks.

The pure-Python git/network surface lives in
:mod:`application.service.models.menagerie_manager`; this package wraps it
into ``Task`` subclasses so the Robot Asset sidebar dialog dispatches via
``get_tasks_manager().submit(...)`` (cancellable, progress-bridged into
``CmdLogWidget``) instead of raw ``QThread`` workers.
"""

from .tasks import (
    MenagerieIconFetchTask,
    MenagerieRefreshTask,
    MenagerieSparseAddTask,
)

__all__ = [
    "MenagerieIconFetchTask",
    "MenagerieRefreshTask",
    "MenagerieSparseAddTask",
]
