# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab USD Body Auto-Dump — boot-time trigger for the factory_build layer.

Runs in ``main.py._data_load_body`` before
:class:`application.tools.robot_asset_selfcheck.RobotAssetSelfCheckTask`.
Populates ``<FACTORY_BUILD_DIR>/robots_factory_build.json`` with the per-SKU
USD body tables that R5 ⊆-coverage / canvas pre-realization / sensor wiring
all read at runtime — so the data is shared across every user on this
install and treated as a factory fact (§9 portable artifacts).

Three routes, evaluated in order:

  * **Route A** — local IsaacLab is registered & available. Iterate every
    canonical SKU; for each whose ``bodies_per_format["USD"]`` is missing in
    the factory_build layer, call
    :meth:`RobotAssetService.dump_and_persist(sku, "USD",
    target_layer="factory_build")`. The user-overlay layer (if any) is
    untouched — its data still wins on conflict per the three-layer merge
    order (SDK ship < factory_build < user overlay).

  * **Route B** — no local IsaacLab, but at least one cloud Isaac server is
    configured (``EngineService.list_servers("isaac_lab")`` non-empty).
    The task does NOT block startup — it returns ``requires_cloud_prompt=True``
    in its report; ``main.py._finalize`` (or another UI hook) presents a
    dialog whose action button is currently a §8 fail-loud no-op
    (``log_error("cloud dump pathway not yet wired")`` + return). The
    dialog framework is in place so wiring the cloud action later is a
    handler-only edit.

  * **Route C** — every canonical SKU already has USD body data in either
    factory_build or user overlay. Skip (no work).

§8 fail-loud:
  * Route A dump failure (Isaac kit subprocess crash, Nucleus URL fetch
    error, MJCF asset missing, ...) → ``WARN`` with the full DumpResult
    error + SKU + kind, and the SKU is added to ``failed_skus`` so
    downstream code (R5 ⊆ check) can surface the missing-USD-data state
    rather than silently substituting an empty body list.
  * The factory_build file is NEVER touched with empty-data-as-success;
    a failed dump means the SKU's entry stays absent from factory_build,
    and the load-time R5 check fail-loud raises directing the user to
    re-trigger the dump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from unitport_sdk import Task


@dataclass
class AutoDumpReport:
    """Outcome of :class:`IsaacLabBodyAutoDumpTask.run`. ``main.py`` consumes
    this to decide whether to (a) open the cloud-IsaacLab prompt dialog
    and (b) re-emit any post-dump signals so other startup steps see the
    freshly populated registry.
    """

    # SKUs successfully dumped this run (USD body table now in factory_build).
    dumped_skus: List[str] = field(default_factory=list)
    # SKUs whose dump failed; downstream code (R5 ⊆) fail-loud raises rather
    # than silently substituting empty data. Each tuple = (sku, error_str).
    failed_skus: List[Tuple[str, str]] = field(default_factory=list)
    # SKUs that already had USD data in either factory_build or user overlay
    # (no dump needed).
    skipped_skus: List[str] = field(default_factory=list)
    # Set by Route B when local IsaacLab is absent but cloud servers exist.
    # ``main.py`` presents the cloud-prompt dialog when True.
    requires_cloud_prompt: bool = False
    # Diagnostic — surfaces in startup logs so users know which route ran.
    # One of: "local_available", "cloud_only", "no_isaac", "skipped".
    isaac_status: str = "no_isaac"


class IsaacLabBodyAutoDumpTask(Task):
    """Run the three-route auto-dump described in the module docstring.

    Cancellable at every per-SKU boundary via :meth:`Task.check_cancelled`.
    Subprocess-bound USD dump (5-30s per SKU) runs serially — parallelising
    would require multiple Isaac kit subprocesses which the Nucleus library
    is not licensed for in unattended workloads.
    """

    def __init__(self) -> None:
        super().__init__(name="Isaac Lab USD Body Auto-Dump")

    def run(self) -> AutoDumpReport:
        from registers import robots as _r
        from application.service.robot_assets import get_robot_asset_service

        report = AutoDumpReport()

        # Resolve local IsaacLab availability via the same path the engines
        # service uses (so detection is consistent with what the User panel
        # shows under "Engines → Local").
        from application.service.engines.service import resolve_active_local_isaac
        local_install = resolve_active_local_isaac()
        local_available = local_install is not None

        # Determine cloud-only mode: no local + at least one cloud server
        # configured for isaac_lab. Read the cloud-state file DIRECTLY
        # (not via EngineService) — EngineService is a QObject and
        # constructing it on a worker thread emits a Qt thread-affinity
        # warning. Schema mirrors EngineService._load_cloud_state.
        cloud_servers: List[dict] = []
        if not local_available:
            try:
                from unitport_sdk import Paths, read_data
                cloud_path = Paths.USER_CONFIG_DIR / "engines" / "isaac_lab.json"
                if cloud_path.exists():
                    state = read_data(cloud_path) or {}
                    if isinstance(state, dict):
                        cloud_servers = list(
                            (state.get("cloud") or {}).get("servers") or []
                        )
            except Exception as exc:  # noqa: BLE001
                self.log_warning(
                    f"cloud-server lookup failed (non-fatal): "
                    f"{type(exc).__name__}: {exc}"
                )
                cloud_servers = []

        if local_available:
            report.isaac_status = "local_available"
        elif cloud_servers:
            report.isaac_status = "cloud_only"
        else:
            report.isaac_status = "no_isaac"
            self.log_info(
                "no local IsaacLab and no cloud servers configured — "
                "skipping USD body auto-dump (R5 will surface any "
                "consumer needs)"
            )
            return report

        # Standard SKU set — only canonical (SDK-ship) SKUs are eligible.
        # User-private SKUs are the owner's responsibility to dump.
        standard_skus = [s for s in _r.list_skus() if _r.is_canonical_sku(s)]
        self.log_info(
            f"isaac_status={report.isaac_status}, {len(standard_skus)} "
            f"canonical SKU(s) eligible for USD body auto-dump"
        )

        # Route B: cloud-only — defer to the prompt dialog and return.
        # We do NOT attempt a cloud dump here (the cloud RPC path is a
        # §8 fail-loud no-op stub; running it would just log noise).
        if report.isaac_status == "cloud_only":
            report.requires_cloud_prompt = True
            for sku in standard_skus:
                report.skipped_skus.append(sku)
            self.log_info(
                "cloud-only mode — deferring dump to user-initiated cloud "
                "action; cloud_prompt will fire in finalize stage"
            )
            return report

        # Route A: local IsaacLab present — dump every SKU whose USD body
        # table is missing in the factory_build layer.
        svc = get_robot_asset_service()
        for sku in standard_skus:
            self.check_cancelled()
            if self._has_usd_bodies(sku):
                report.skipped_skus.append(sku)
                continue

            entry = _r.get_robot(sku) or {}
            name = str(entry.get("name") or sku)
            self.log_info(f"dumping USD bodies for {name!r} (sku={sku})")
            try:
                result = svc.dump_and_persist(
                    sku, "USD", target_layer="factory_build"
                )
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                self.log_warning(
                    f"dump crashed for {name!r}: {err}"
                )
                report.failed_skus.append((sku, err))
                continue

            if not result.ok:
                err = f"{result.error_kind}: {result.error}"
                self.log_warning(
                    f"dump failed for {name!r}: {err}"
                )
                report.failed_skus.append((sku, err))
                continue

            report.dumped_skus.append(sku)

        if report.dumped_skus:
            self.log_info(
                f"route A complete — dumped {len(report.dumped_skus)} SKU(s) "
                f"into factory_build layer; "
                f"{len(report.skipped_skus)} already had data; "
                f"{len(report.failed_skus)} failed"
            )
        else:
            self.log_info(
                "route A complete — no canonical SKU needed a USD dump"
            )
        return report

    @staticmethod
    def _has_usd_bodies(sku: str) -> bool:
        """True iff the merged registry view of ``sku`` already carries a
        non-empty ``bodies_per_format["USD"]`` block (from EITHER the
        factory_build OR the user-overlay layer)."""
        from registers import robots as _r
        entry = _r.get_robot(sku) or {}
        bpf = (entry.get("bodies_per_format") or {})
        block = bpf.get("USD")
        return isinstance(block, dict) and bool(block)
