"""application.service.models -- robot model & asset managers.

Members:

- :class:`MenagerieManager` -- MuJoCo Menagerie sparse-checkout manager
  (GitHub Trees API + sparse clone + per-package add). Ported 1:1 from
  ``DEMO/src/system/models/menagerie_manager.py``.
- :class:`SdkManager` -- per-brand SDK clone/install manager. Ports
  DEMO ``ensure_selected_sdks`` + ``ensure_registered_sdks`` plus the
  Isaac Lab path register helper.

Both ports preserve DEMO's storage schema bit-exact so a user moving
from DEMO -> RELEASE keeps custom_mods/models/menagerie/ + their SDK
clone state without manual migration.
"""

from .menagerie_manager import MenagerieManager
from .sdk_manager import SdkManager

__all__ = ["MenagerieManager", "SdkManager"]
