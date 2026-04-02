"""src.system.binding.bindings — concrete ActionBinding implementations.

Importing this package registers all built-in bindings into
``src.system.binding.registry.default_registry`` at import time.
Each binding module is responsible for its own registration call.

Registered at import time
-------------------------
- ``locomotion.forward`` (ForwardBinding) — sdk + mujoco, unitree wildcard
"""

from src.system.binding.bindings.locomotion import ForwardBinding
from src.system.binding.registry import default_registry

_forward = ForwardBinding()

# Register for Unitree brand (model wildcard) on both backends.
# Step 7 may add exact registrations for go2 / go2w if model-specific
# behaviour is needed; the model-wildcard entries act as the shared base.
default_registry.register(_forward, brand="unitree", robot_type="", backend="sdk")
default_registry.register(_forward, brand="unitree", robot_type="", backend="mujoco")

# Register for Boston Dynamics Spot on both backends.
default_registry.register(_forward, brand="bostiondynamics", robot_type="", backend="sdk")
default_registry.register(_forward, brand="bostiondynamics", robot_type="", backend="mujoco")
