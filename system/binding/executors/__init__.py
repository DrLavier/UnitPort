"""system.binding.executors — OperationExecutor interfaces and implementations.

    base.py                  -- ExecutorResult dataclass + OperationExecutor ABC
    sdk_executor.py          -- SDKOperationExecutor base (backend="sdk")
    mujoco_executor.py       -- MujocoOperationExecutor base (backend="mujoco")
    unitree_sdk_executor.py  -- UnitreeSDKExecutor (go2 + go2w, SDK path, Step 7)
    go2_mujoco_executor.py   -- Go2MujocoExecutor (go2 MuJoCo; go2w unsupported, Step 7)
"""

from system.binding.executors.base import ExecutorResult, OperationExecutor
from system.binding.executors.sdk_executor import SDKOperationExecutor
from system.binding.executors.mujoco_executor import MujocoOperationExecutor
from system.binding.executors.unitree_sdk_executor import UnitreeSDKExecutor
from system.binding.executors.go2_mujoco_executor import Go2MujocoExecutor
from system.binding.executors.spot_sdk_executor import SpotSDKExecutor
from system.binding.executors.spot_mujoco_executor import SpotMujocoExecutor

__all__ = [
    "ExecutorResult",
    "OperationExecutor",
    "SDKOperationExecutor",
    "MujocoOperationExecutor",
    "UnitreeSDKExecutor",
    "Go2MujocoExecutor",
    "SpotSDKExecutor",
    "SpotMujocoExecutor",
]
