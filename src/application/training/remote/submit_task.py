# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""RemoteSubmitTask — submit one training run to a cloud SSH server.

Activated when the user trains with the Cloud target. The chain is:

    Play → submit_canvas_training (target=="cloud")
         → RemoteTrainingBackend.build_task(spec, run_id)
         → RemoteSubmitTask(spec, engine_name, run_id)     # __init__ = config gate
         → TasksManager.submit(task) → task.run()          # run() = transport

Scope is "submittable": this Task connects over SSH, pushes the compiled spec +
canvas to the server's run directory, launches the remote entrypoint detached,
captures its PID as the job handle, and records the handle locally. The remote
*execution* (the ``unitport_remote_train.sh`` worker), live progress streaming,
and artifact pull-back are the explicit next iteration — this file only fixes
the contract those layers consume (see ``REMOTE_ENTRYPOINT`` /
``--engine/--run-id/--run-dir/--spec/--canvas`` below).

Fail-loud split (CLAUDE.md §8 — NEVER fall back to a local run):
    * Construction-time (``__init__``, runs synchronously inside build_task so
      it surfaces in the Play button's try/except as a dialog): no open project,
      no default server, missing SSH credential / key file → RemoteSubmitConfigError.
    * Run-time (``run()`` → task_finished(success=False)): SSH connect failure,
      remote_install_dir / entrypoint missing, launch returned no PID.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

from unitport_sdk import log_info

from application.training._sdk.task_runner import TrainingTask

# Remote entrypoint the server is expected to provide under ``remote_install_dir``.
# Provisioning it is the deferred installer's job; we only depend on the calling
# contract documented in the module docstring. The probe in run() checks it
# exists and fails loud when absent.
REMOTE_ENTRYPOINT = "unitport_remote_train.sh"


class RemoteSubmitConfigError(RuntimeError):
    """User-actionable misconfiguration of the cloud submit (raised in __init__).

    Distinct from a generic RuntimeError so the Play button can route it to a
    clear warning dialog instead of a cmd-log traceback. Never indicates a
    fallback to local — the submit is aborted and the user is told what to fix.
    """


class RemoteSubmitTask(TrainingTask):
    """Submit a compiled spec to a cloud SSH server and capture the job handle."""

    def __init__(
        self,
        spec: Dict[str, Any],
        *,
        engine_name: str,
        run_id: str,
        engine_id: Optional[str] = None,
    ) -> None:
        if not isinstance(spec, dict):
            raise RemoteSubmitConfigError(
                "RemoteSubmitTask requires the TrainingSpec.to_dict() payload "
                f"(got {type(spec).__name__})."
            )
        self._spec: Dict[str, Any] = spec
        self.engine_name: str = str(engine_name)
        # Cloud servers are namespaced per engine in EngineService; the active
        # cloud path is Isaac Lab, whose servers carry the run. Default the
        # lookup namespace to the spec engine.
        self.engine_id: str = str(engine_id or engine_name)

        # --- resolve project for the local run-dir anchor (fail-loud) --------
        from application.service.projects import current_project_info

        proj = current_project_info()
        if proj is None:
            raise RemoteSubmitConfigError(
                "Cloud training requires an open project — the local run "
                "directory (logs + job handle) lands under the project tree."
            )
        run_dir = proj.path / "training" / "runs" / "remote" / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- resolve the default cloud server (fail-loud (a)) ----------------
        from application.service.engines.service import get_engine_service

        server = get_engine_service().get_default_server(self.engine_id)
        if not server or not str(server.get("host") or "").strip():
            raise RemoteSubmitConfigError(
                "Cloud target is selected but no SSH server is configured for "
                f"engine {self.engine_id!r}. Add one and set it as default in "
                "Engine Settings → Cloud servers, or switch the train target "
                "back to Local."
            )
        self._server: Dict[str, Any] = dict(server)

        # --- resolve credentials up-front (fail-loud (b)) --------------------
        from application.service.auth.secure import get_secure_store

        store = get_secure_store()
        name = str(self._server.get("name") or "")
        auth_method = str(self._server.get("auth_method") or "key").strip().lower()
        self._password: Optional[str] = store.ssh_password(name).get()
        self._sudo_password: Optional[str] = store.ssh_sudo_password(name).get()
        self._key_filename: Optional[str] = (
            str(self._server.get("private_key_path") or "").strip() or None
        )
        if auth_method == "password":
            if not self._password:
                raise RemoteSubmitConfigError(
                    f"Server {name!r} uses password authentication but no "
                    "password is saved in the OS keyring. Re-enter it in "
                    "Engine Settings → Cloud servers."
                )
        elif auth_method == "key":
            if not self._key_filename:
                raise RemoteSubmitConfigError(
                    f"Server {name!r} uses key authentication but no private "
                    "key path is set."
                )
            if not Path(self._key_filename).expanduser().is_file():
                raise RemoteSubmitConfigError(
                    f"Server {name!r} private key not found: "
                    f"{self._key_filename!r}."
                )

        display = f"Remote {self.engine_name} #{str(run_id)[-8:]}"
        super().__init__(display, run_id, run_dir)

    # ------------------------------------------------------------------
    # run() — the transport. Any raise here → task_finished(success=False).
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        from application.service.connection.ssh_session import SSHSession
        from application.service.engines.remote_jobs import (
            STATUS_SUBMITTED,
            RemoteJobRecord,
            append_remote_job,
            now_iso,
        )

        srv = self._server
        name = str(srv.get("name") or "")
        host = str(srv.get("host") or "")
        user = str(srv.get("user") or "ubuntu")
        port = int(srv.get("port") or 22)
        install_dir = str(srv.get("remote_install_dir") or "").strip()
        work_dir = str(srv.get("remote_work_dir") or "").strip() or install_dir
        conda_env = str(srv.get("conda_env") or "").strip()

        self.info(
            f"cloud submit start engine={self.engine_name} server={name!r} "
            f"({user}@{host}:{port})"
        )

        if not install_dir:
            # Fail-loud (d): no install dir to host the entrypoint.
            raise RuntimeError(
                f"Server {name!r} has an empty remote_install_dir — cannot "
                "locate the remote training entrypoint. Set it in Engine "
                "Settings → Cloud servers."
            )

        session = SSHSession(
            host=host,
            user=user,
            password=self._password,
            port=port,
            key_filename=self._key_filename,
            sudo_password=self._sudo_password,
        )
        # connect() raises SshError on auth/unreachable/protocol — let it
        # propagate (fail-loud (c), NO fallback to a local run).
        session.connect()
        try:
            self.check_cancelled()

            # --- probe: install dir + entrypoint present (fail-loud (d)) -----
            entry = f"{install_dir.rstrip('/')}/{REMOTE_ENTRYPOINT}"
            probe = session.exec(
                f"test -d {shlex.quote(install_dir)} && "
                f"test -f {shlex.quote(entry)} && echo OK",
                timeout=20.0,
            )
            if "OK" not in probe.stdout:
                raise RuntimeError(
                    f"Remote install dir {install_dir!r} is missing or its "
                    f"{REMOTE_ENTRYPOINT!r} entrypoint is not present on "
                    f"{host}. Provision the remote engine before submitting "
                    "(remote installer is a separate step). "
                    f"(stderr: {probe.stderr.strip()[:200]})"
                )

            # --- create remote run dir + push spec/canvas --------------------
            remote_run_dir = f"{work_dir.rstrip('/')}/unitport_runs/{self.run_id}"
            mk = session.exec(
                f"mkdir -p {shlex.quote(remote_run_dir)}", timeout=20.0
            )
            if not mk.ok:
                raise RuntimeError(
                    f"Failed to create remote run dir {remote_run_dir!r} "
                    f"(exit={mk.exit_code}): {mk.stderr.strip()[:200]}"
                )

            self.check_cancelled()
            spec_bytes = json.dumps(self._spec).encode("utf-8")
            canvas_obj = (self._spec.get("meta") or {}).get("__canvas_dict__") or {}
            canvas_bytes = json.dumps(canvas_obj).encode("utf-8")
            session.put_file_bytes(f"{remote_run_dir}/spec.json", spec_bytes)
            session.put_file_bytes(f"{remote_run_dir}/canvas.json", canvas_bytes)

            # --- launch detached; capture PID as the job handle --------------
            self.check_cancelled()
            conda_flag = (
                f"--conda-env {shlex.quote(conda_env)} " if conda_env else ""
            )
            launch = (
                f"cd {shlex.quote(install_dir)} && "
                f"setsid nohup bash {shlex.quote(REMOTE_ENTRYPOINT)} "
                f"--engine {shlex.quote(self.engine_name)} "
                f"--run-id {shlex.quote(self.run_id)} "
                f"--run-dir {shlex.quote(remote_run_dir)} "
                f"--spec {shlex.quote(remote_run_dir + '/spec.json')} "
                f"--canvas {shlex.quote(remote_run_dir + '/canvas.json')} "
                f"{conda_flag}"
                f"> {shlex.quote(remote_run_dir + '/launch.log')} 2>&1 "
                f"& echo $!"
            )
            res = session.exec(launch, timeout=60.0)
            pid_str = res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else ""
            if not res.ok or not pid_str.isdigit():
                raise RuntimeError(
                    f"Remote launch failed (exit={res.exit_code}): "
                    f"{(res.stderr.strip() or res.stdout.strip())[:300]}"
                )
            remote_pid = int(pid_str)

            # --- record the handle locally (loud-warn, never raise) ----------
            record = RemoteJobRecord(
                run_id=self.run_id,
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                server_name=name,
                host=host,
                remote_run_dir=remote_run_dir,
                remote_pid=remote_pid,
                status=STATUS_SUBMITTED,
                submitted_at=now_iso(),
            )
            append_remote_job(record)

            # Mirror the handle into the local run dir for offline reference.
            try:
                (self.run_dir / "remote_job.json").write_text(
                    json.dumps(
                        {
                            "run_id": self.run_id,
                            "server": name,
                            "host": host,
                            "remote_run_dir": remote_run_dir,
                            "remote_pid": remote_pid,
                            "status": STATUS_SUBMITTED,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

            self.info(
                f"cloud submit OK pid={remote_pid} dir={remote_run_dir}"
            )
            log_info(
                f"[remote-submit] run_id={self.run_id} pid={remote_pid} "
                f"server={name!r} dir={remote_run_dir}"
            )
            return {
                "remote_pid": remote_pid,
                "remote_run_dir": remote_run_dir,
                "server": name,
                "status": STATUS_SUBMITTED,
            }
        finally:
            session.close()


__all__ = ["RemoteSubmitTask", "RemoteSubmitConfigError", "REMOTE_ENTRYPOINT"]
