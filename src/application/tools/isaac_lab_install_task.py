# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``TasksManager``-compatible wrapper around :class:`IsaacLabInstaller`.

Submitted by ``PostSetupTask._step_isaac_lab`` when the user picked the
fresh-install path and accepted the EULA. The wrapper exists so the
30 GB install can:

* run in a managed worker thread (cancellation, slot counting),
* surface logs through the standard ``LogSignal`` route into
  CmdLogWidget without bypassing the SDK contract, and
* present a single ``TaskSignal.task_finished`` to the wizard / sidebar
  exactly the same way the SB3 trainer task does.

Progress and per-stage labels are emitted by the installer directly
through ``AppSignals.isaac_install_*``; the Task wrapper does not
duplicate them through ``progress_line`` because the dialog binding is
keyed on those signals, and routing the same progress through two
channels would cause double-renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from unitport_sdk import Task

from application.service.installers import (
    InstallerCancelled,
    InstallerError,
    InstallPlan,
    IsaacLabInstaller,
    is_install_path_internal,
)


class IsaacLabInstallTask(Task):
    """Run :class:`IsaacLabInstaller` under the SDK ``TasksManager``.

    Construct with the path the user typed in the wizard (or the wizard's
    default) plus the list of EULA ids returned by the wizard's
    ``EulaDialog`` so the installer's preflight can reject hand-edited
    setup_state.json that claims accepted licences the user never saw.

    The installer's pipeline is synchronous and entirely event-driven
    via ``AppSignals``; the Task's ``run`` body is therefore tiny — its
    only job is to translate the installer's exceptions into Task
    semantics (``check_cancelled`` callback / non-zero return).
    """

    def __init__(
        self,
        *,
        install_dir: Path,
        eula_ids: List[str],
        name: str = "isaac_lab_install",
    ) -> None:
        super().__init__(name=name)
        # The wizard / PostSetupTask sees just a string path; resolve to
        # Path here so the installer can rely on Path semantics.
        self._install_dir = Path(install_dir).expanduser()
        # Re-judge internal vs external at submission time. PostSetupTask
        # may pre-declare a mode but it can drift if the user moved the
        # project between wizard completion and task dispatch.
        self._mode = (
            "internal"
            if is_install_path_internal(self._install_dir)
            else "external"
        )
        self._eula_ids = list(eula_ids or [])

    def run(self) -> Any:  # type: ignore[override]
        self.log_info(
            f"starting Isaac Lab install at {self._install_dir} "
            f"(mode={self._mode}, eula_ids={len(self._eula_ids)})"
        )

        plan = InstallPlan(
            install_dir=self._install_dir,
            mode=self._mode,
            eula_ids=list(self._eula_ids),
            # Bridge to the Task framework's cooperative cancellation.
            # IsaacLabInstaller calls this at every stage boundary and
            # at chunk loops inside download / extract.
            check_cancelled=self.check_cancelled,
        )

        installer = IsaacLabInstaller(plan)
        try:
            report = installer.execute()
        except InstallerCancelled:
            # The installer has already emitted the complete(False, ...)
            # signal; Task framework just needs to know we exited.
            self.log_warning("Isaac Lab install cancelled")
            return None
        except InstallerError as exc:
            self.log_error(f"Isaac Lab install failed: {exc}")
            raise
        self.log_success(
            f"Isaac Lab installed at {report.install_dir} "
            f"(mode={report.mode}, sim={report.isaac_sim_version}, "
            f"lab={report.isaac_lab_tag})"
        )
        return {
            "install_dir": str(report.install_dir),
            "mode": report.mode,
            "isaac_sim_version": report.isaac_sim_version,
            "isaac_lab_tag": report.isaac_lab_tag,
            "installed_at": report.installed_at,
        }


__all__ = ["IsaacLabInstallTask"]
