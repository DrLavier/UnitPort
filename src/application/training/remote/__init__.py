# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Remote (cloud) training submission — SSH-based job launch to a user server.

See :mod:`application.training.remote.submit_task` for the submit Task and
:mod:`application.training.remote_backend` for the hot-pluggable backend
registry that routes a compiled spec here when the train target is "cloud".
"""

from application.training.remote.submit_task import (
    RemoteSubmitConfigError,
    RemoteSubmitTask,
)

__all__ = ["RemoteSubmitTask", "RemoteSubmitConfigError"]
