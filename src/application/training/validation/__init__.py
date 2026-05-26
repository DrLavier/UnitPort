# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.validation — Pre-export validation steps.

Hosts the sim2sim PD calibration node + future validation pre-flight
checks that run between policy training and bundle export. The bundle
finalizer calls into here to gate on verdict before writing artifacts
to disk.
"""

from .sim2sim_calibration import (
    CalibrationReport,
    CalibrationVerdict,
    JointCalibrationResult,
    run_calibration,
)


__all__ = [
    "CalibrationReport",
    "CalibrationVerdict",
    "JointCalibrationResult",
    "run_calibration",
]
