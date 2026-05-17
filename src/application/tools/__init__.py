"""application.tools — pure-compute helpers consumed by application code.

Members today:
- :class:`startup_tasks.FuncTask` — generic ``Task`` wrapper around a callable.
- :class:`startup_tasks.ProvisioningTask` — post-bootstrap gate that
  installs torch CUDA, the rest of requirements.txt, the loco-mujoco
  reference motion library, and the ``unitport://`` URL scheme. Streams
  every subprocess line to the on-screen LoadingScreen log.
"""
