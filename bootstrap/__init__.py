"""Project-level bootstrap helpers.

The .py files here (``_check_requirements``, ``detect_ros2``,
``register_url_scheme``, the one-shot migrators) are individually runnable
from the command line and are also loaded by file path from
``application.tools.startup_tasks`` via ``importlib.util.spec_from_file_location``.
``PROJECT_ROOT`` is intentionally kept off ``sys.path`` (see ``main.py``),
so ``from bootstrap.X import Y`` will not work -- load by file path instead.
"""
