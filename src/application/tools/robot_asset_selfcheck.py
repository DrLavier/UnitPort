# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Robot Asset Self-Check — boot-time scan + auto-dump pipeline.

Runs as part of the ``_data_load_body`` startup stage. Two **independent**
passes over every registered robot × enabled format (MJCF / USD):

**Pass A — Dump-if-empty (slow path, triggers Isaac kit subprocess):**

  Only runs for ``(sku, fmt)`` where BOTH ``joints_per_format[fmt]`` and
  ``bodies_per_format[fmt]`` are None / empty. Invokes
  :meth:`RobotAssetService.dump_and_persist` to populate them. Already
  populated tables are skipped — no re-dump, no clobbering of manual
  assignments. Failures (Isaac unavailable, Nucleus down, kit crash)
  log a warning and move on.

**Pass B — Collect pending (fast, always runs):**

  After Pass A finishes (so freshly-dumped tables are visible), iterate
  every robot × enabled format again. For each format that has ANY
  entry with empty ``ir_role`` — regardless of whether Pass A touched
  it — gather ALL entries (joints + bodies, auto-matched AND empty)
  into the pending list. The dialog shows the complete table so users
  can review tokeniser auto-matches alongside the empty rows that
  need their input.

  Robots whose per-format table is fully populated (every entry has a
  non-empty ir_role) contribute nothing to the pending list — no
  dialog noise for already-good robots.

The two passes are independent because the dialog must fire whenever
*any* assignment is missing, not just when a dump just ran. A robot
that was dumped weeks ago but still has empty entries (because the
tokeniser couldn't classify them and the user never opened the canvas
Body Mapping table) must still trigger the dialog on every boot until
resolved.

The task returns the pending list — ``main.py._maybe_open_ir_assignment_dialog``
opens :class:`IRRoleAssignmentDialog` iff this list is non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unitport_sdk import Task, log_info, log_warning


@dataclass
class PendingAssignment:
    """One row of "needs user IR-role assignment" data.

    Produced by :class:`RobotAssetSelfCheckTask`; consumed by
    :class:`IRRoleAssignmentDialog`. Each instance identifies exactly
    one entry in the registry (a single joint OR body for a single
    robot in a single format) whose ``ir_role`` is currently empty.
    The dialog will let the user pick a non-empty role for it; on
    Confirm the dialog persists via
    :meth:`RobotAssetService.update_ir_role`.
    """

    sku: str             # robot SKU
    robot_name: str      # display name for the list ("Boston Dynamics Spot")
    family: str          # IR family — drives the dropdown's valid-role list
    fmt: str             # "MJCF" / "USD"
    kind: str            # "joint" or "body"
    uid: str             # registry key in joints_per_format[fmt] / bodies_per_format[fmt]
    name: str            # raw joint/body name from the asset (UI display)
    current: str = ""    # current ir_role after auto-dump (empty = needs user choice)


class RobotAssetSelfCheckTask(Task):
    """Boot-time registry sweep — auto-dump empty per-format tables and
    return a list of pending IR-role assignments for the UI to resolve.

    See module docstring for the full algorithm.
    """

    def __init__(self) -> None:
        super().__init__(name="Robot Asset Self-Check")

    def run(self) -> List[PendingAssignment]:
        from registers import robots as _r
        from application.service.robot_assets import get_robot_asset_service

        svc = get_robot_asset_service()
        all_skus = list(_r.list_skus())
        self.log_info(f"scanning {len(all_skus)} registered robot(s)")

        # ── Pass A: dump-if-empty ────────────────────────────────────────
        # Only touches robot × fmt where the per-format table is null/empty.
        # Slow path (Isaac kit subprocess); skipped robots cost ~0.
        n_dumped = 0
        for sku in all_skus:
            self.check_cancelled()
            asset = svc.resolve(sku)
            if asset is None:
                continue
            robot_name = self._robot_display_name(sku, asset)
            for fmt in ("MJCF", "USD"):
                if not self._fmt_enabled(asset, fmt):
                    continue
                if self._dump_if_empty(svc, sku, robot_name, fmt):
                    n_dumped += 1
        if n_dumped:
            self.log_info(f"pass A: dumped {n_dumped} fmt-table(s)")
        else:
            self.log_info("pass A: no empty fmt-tables found, nothing to dump")

        # ── Pass A.5: retokenise empty ir_roles in already-dumped tables ─
        # Previous dumps may have left ir_role="" on entries the old
        # tokeniser couldn't classify (e.g. BD Spot's uleg/lleg/ank
        # before v1.2.0). The current tokeniser may handle them, so
        # re-suggest in-process and patch the empties via
        # bulk_update_ir_roles. Idempotent: entries that still come back
        # empty stay empty for Pass B to surface. Uniqueness-aware:
        # never assigns a role_id already used in the same per-format
        # table.
        n_retok = self._retokenise_empties(svc, all_skus)
        if n_retok:
            self.log_info(f"pass A.5: re-tokenised {n_retok} empty ir_role(s) "
                          f"using the upgraded keyword dictionary")
        else:
            self.log_info("pass A.5: no empty ir_role(s) needed re-tokenising")

        # ── Pass A.6: dedupe duplicate ir_role assignments ───────────────
        # Pre-v1.2.0 dumps (and any third-party-written user overlay)
        # may have assigned the SAME non-sensor ir_role to multiple
        # bodies — e.g. Go2 USD's FL_hip + FL_hip_protector both
        # tokenised to hip_FL. BodyIRMapper's per-role slot model
        # silently evicts the first-assigned body, surfacing it as
        # "unmapped" in the canvas table even though the registry data
        # is "filled". Detect those collisions, keep the most-likely
        # canonical entry (no cosmetic-suffix tokens), blank the rest.
        n_dedupe = self._dedupe_ir_roles(svc, all_skus)
        if n_dedupe:
            self.log_info(f"pass A.6: blanked {n_dedupe} duplicate ir_role "
                          f"assignment(s) (kept canonical, demoted cosmetic)")
        else:
            self.log_info("pass A.6: no duplicate ir_role assignments found")

        # ── Pass A.7: cross-format joint-count consistency check ──────────
        # When a robot has BOTH MJCF and USD per-format tables populated,
        # the two MUST describe the same joint set — otherwise Isaac Lab
        # → MuJoCo deploy is broken before training even starts (bundle
        # finalize's ``_derive_mujoco_pd_gains_for_bundle`` walks the
        # trained policy's IR roles against the MJCF table; any role that
        # exists in USD but not MJCF raises). CLAUDE.md §1.10 mandates
        # both engines' gains derived from the same canonical PD source,
        # which is only possible when both formats agree on joint set.
        #
        # We don't try to AUTO-FIX this: the divergence is almost always
        # an asset-path mismatch (e.g. G1's ``scene.xml`` is 29-DOF but
        # ``g1.usd`` is 43-DOF — re-dumping the same wrong MJCF asset
        # would loop forever). Loud log_error + a directive to swap the
        # asset path is the right escalation.
        n_drift = self._check_cross_format_drift(all_skus)
        if n_drift:
            self.log_info(
                f"pass A.7: {n_drift} robot(s) have MJCF/USD joint-count "
                f"divergence (see log_error lines above for the asset-path "
                f"fix)"
            )
        else:
            self.log_info("pass A.7: every dual-format robot agrees on joint count")

        # ── Pass B: collect pending (independent of Pass A outcome) ──────
        # Walks every robot × enabled fmt and surfaces ANY (sku, fmt) that
        # has at least one entry with empty ir_role — auto-matched entries
        # come along too so the dialog can show the complete picture.
        pending: List[PendingAssignment] = []
        for sku in all_skus:
            self.check_cancelled()
            asset = svc.resolve(sku)
            if asset is None:
                continue
            robot_name = self._robot_display_name(sku, asset)
            family = self._robot_family(asset)
            for fmt in ("MJCF", "USD"):
                if not self._fmt_enabled(asset, fmt):
                    continue
                self._collect_if_pending(
                    sku=sku, robot_name=robot_name, family=family, fmt=fmt,
                    pending=pending,
                )

        if pending:
            n_empty = sum(1 for p in pending if not p.current)
            n_skus = len({p.sku for p in pending})
            self.log_info(
                f"pass B: {n_empty} empty ir_role(s) across {n_skus} robot(s) "
                f"({len(pending)} total rows surfaced to dialog)"
            )
        else:
            self.log_info("pass B: every enabled fmt-table fully assigned")
        return pending

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _robot_display_name(sku: str, asset: Any) -> str:
        name = str(getattr(asset, "name", "") or "").strip()
        if name:
            return name
        brand = str(getattr(asset, "brand", "") or "").strip()
        model = str(getattr(asset, "model", "") or "").strip()
        if brand and model:
            return f"{brand} {model}"
        return sku

    @staticmethod
    def _robot_family(asset: Any) -> str:
        families = list(getattr(asset, "families", []) or [])
        return str(families[0]) if families else "generic"

    @staticmethod
    def _fmt_enabled(asset: Any, fmt: str) -> bool:
        f = fmt.lower()
        if f == "mjcf":
            p = getattr(asset, "mjcf_path", None)
            return p is not None and p.exists()
        if f == "usd":
            p = getattr(asset, "usd_path", None)
            if p is not None and p.exists():
                return True
            return bool(getattr(asset, "usd_url", None))
        if f == "urdf":
            p = getattr(asset, "urdf_path", None)
            return p is not None and p.exists()
        return False

    def _retokenise_empties(self, svc: Any, all_skus: List[str]) -> int:
        """Pass A.5 — run the current tokeniser over every empty ir_role
        in every already-populated per-format table, in-process.

        Uniqueness-aware: never assigns a role_id already used by
        another entry in the SAME (sku, fmt, kind) table. Cosmetic-token
        veto: names like ``*_protector`` / ``*_cover`` get an empty
        result rather than colliding with the canonical limb body. Both
        guards mirror the dump-side ``_suggest_ir_roles`` so freshly
        dumped tables and re-tokenised legacy tables stay symmetric.

        Cheap (no subprocess, single registry write + reload at the
        end via :meth:`RobotAssetService.bulk_update_ir_roles`).
        """
        from registers import robots as _r
        from application.training.body_ir import _suggest_role_id
        from application.service.robot_assets.discovery_subprocess import (
            _is_cosmetic_name,
            _is_non_unique_role,
        )

        patches: List[tuple] = []
        for sku in all_skus:
            self.check_cancelled()
            asset = svc.resolve(sku)
            if asset is None:
                continue
            family = self._robot_family(asset)
            entry = _r.get_robot(sku) or {}
            for fmt in ("MJCF", "USD"):
                if not self._fmt_enabled(asset, fmt):
                    continue
                for kind, block_key in (("joint", "joints_per_format"),
                                        ("body",  "bodies_per_format")):
                    table = (entry.get(block_key) or {}).get(fmt) or {}
                    # Pre-compute the role_ids already in use in this
                    # (kind, fmt) table so we can refuse to re-assign
                    # any of them to an empty entry.
                    used: set = {
                        str((spec or {}).get("ir_role") or "").strip()
                        for spec in table.values()
                        if isinstance(spec, dict)
                        and str((spec or {}).get("ir_role") or "").strip()
                    }
                    for uid, spec in table.items():
                        if not isinstance(spec, dict):
                            continue
                        if str(spec.get("ir_role") or "").strip():
                            continue  # already assigned; don't disturb
                        name = str(spec.get("name") or "").strip()
                        if not name:
                            continue
                        # Cosmetic names get misc directly — no need to
                        # run the tokeniser which would only return
                        # something we'd then override.
                        if _is_cosmetic_name(name):
                            patches.append((sku, fmt, kind, str(uid), "misc"))
                            used.add("misc")
                            continue
                        try:
                            new_role = _suggest_role_id(name, family)
                        except Exception:
                            continue
                        if not new_role:
                            continue
                        # Uniqueness guard — exempts sensor_* and misc
                        # (both legitimately cover many bodies).
                        if (not _is_non_unique_role(new_role)
                                and new_role in used):
                            continue
                        patches.append((sku, fmt, kind, str(uid), new_role))
                        used.add(new_role)

        if not patches:
            return 0
        try:
            return int(svc.bulk_update_ir_roles(patches))
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[robot_asset_selfcheck] retokenise bulk patch crashed: "
                f"{type(exc).__name__}: {exc}"
            )
            return 0

    def _dedupe_ir_roles(self, svc: Any, all_skus: List[str]) -> int:
        """Pass A.6 — for every per-format table, find entries that all
        carry the same non-sensor ir_role and blank all but the most
        canonical one.

        "Most canonical" = no cosmetic-suffix tokens in the body/joint
        name (e.g. ``FL_hip`` over ``FL_hip_protector``); tiebreak on
        shorter name. Sensor roles (``sensor`` / ``sensor_*``) and the
        Out-of-Scope sentinel are exempt because they can legitimately
        cover multiple entries.

        Required because pre-v1.2.0 dumps assigned the same role_id to
        every body whose tokenised stem matched — BodyIRMapper's
        per-role slot model then silently evicted all but the
        last-written entry, surfacing the rest as "unmapped" even
        though the registry says they're filled.
        """
        from collections import defaultdict
        from registers import robots as _r
        from registers.robots import OUT_OF_SCOPE_IR_ROLE
        from application.service.robot_assets.discovery_subprocess import (
            _is_cosmetic_name,
            _is_non_unique_role,
        )

        def _cosmeticness(name: str) -> int:
            """0 = canonical (no cosmetic substring), 1 = cosmetic. The
            entry sorted to 0 wins; cosmetic ones get demoted to misc."""
            return 1 if _is_cosmetic_name(name) else 0

        patches: List[tuple] = []
        for sku in all_skus:
            self.check_cancelled()
            asset = svc.resolve(sku)
            if asset is None:
                continue
            entry = _r.get_robot(sku) or {}
            for fmt in ("MJCF", "USD"):
                if not self._fmt_enabled(asset, fmt):
                    continue
                for kind, block_key in (("joint", "joints_per_format"),
                                        ("body",  "bodies_per_format")):
                    table = (entry.get(block_key) or {}).get(fmt) or {}
                    # Group entries by ir_role.
                    by_role: Dict[str, List[tuple]] = defaultdict(list)
                    for uid, spec in table.items():
                        if not isinstance(spec, dict):
                            continue
                        role = str(spec.get("ir_role") or "").strip()
                        if not role:
                            continue
                        # sensors / misc legitimately repeat (one robot
                        # has many sensor mounts, many cosmetic shells);
                        # OOS sentinel also legitimately repeats. Only
                        # one-slot-per-role limb categories need dedupe.
                        if _is_non_unique_role(role) or role == OUT_OF_SCOPE_IR_ROLE:
                            continue
                        name = str(spec.get("name") or "")
                        by_role[role].append((str(uid), name))
                    # Within each group of size > 1, keep the most
                    # canonical (lowest cosmeticness; then shortest name;
                    # then alphabetic). DEMOTE the rest to ``misc``
                    # rather than blank — they're physically present, the
                    # tokeniser just guessed too aggressively. User can
                    # later re-pick the actual limb / sensor role for
                    # any specific entry via the dialog.
                    for role, group in by_role.items():
                        if len(group) <= 1:
                            continue
                        group.sort(key=lambda u_n: (
                            _cosmeticness(u_n[1]),
                            len(u_n[1]),
                            u_n[1].lower(),
                        ))
                        # group[0] is the winner; group[1:] get demoted to misc.
                        for uid, _name in group[1:]:
                            patches.append((sku, fmt, kind, uid, "misc"))

        if not patches:
            return 0
        try:
            return int(svc.bulk_update_ir_roles(patches))
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[robot_asset_selfcheck] dedupe bulk patch crashed: "
                f"{type(exc).__name__}: {exc}"
            )
            return 0

    def _dump_if_empty(
        self, svc: Any, sku: str, robot_name: str, fmt: str,
    ) -> bool:
        """Pass A — if joints+bodies per-format table is empty, dump it.

        Returns True iff dump_and_persist was called AND succeeded. Does
        not write anything to the pending list — Pass B handles that.
        """
        from registers import robots as _r

        entry = _r.get_robot(sku) or {}
        joints_table = (entry.get("joints_per_format") or {}).get(fmt) or {}
        bodies_table = (entry.get("bodies_per_format") or {}).get(fmt) or {}
        if joints_table or bodies_table:
            return False

        self.log_info(
            f"dumping {fmt} for sku={sku!r} ({robot_name}) — "
            f"per-format table is empty"
        )
        try:
            result = svc.dump_and_persist(sku, fmt)
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[robot_asset_selfcheck] dump_and_persist crashed "
                f"sku={sku!r} fmt={fmt!r}: {type(exc).__name__}: {exc}; "
                f"skipping"
            )
            return False
        if not result.ok:
            log_warning(
                f"[robot_asset_selfcheck] dump failed "
                f"sku={sku!r} fmt={fmt!r} kind={result.error_kind}: "
                f"{result.error}; skipping (user can retry from canvas Refresh)"
            )
            return False
        return True

    def _check_cross_format_drift(self, all_skus: List[str]) -> int:
        """Pass A.7 — for every robot with BOTH MJCF and USD tables
        populated, log_warning when the joint counts disagree.

        This is ADVISORY, not a failure: cross-format coverage gaps only
        affect which deploy targets the bundle will support, not whether
        training itself is possible. Training uses the active format
        only (IL → USD, SB3 → MJCF); the non-active format's coverage
        determines deploy-target availability. The boot self-check
        surfaces this up-front so the user knows before training what
        deploy targets will be available, but never blocks.

        Returns the number of robots flagged.
        """
        from registers import robots as _r

        n_flagged = 0
        for sku in all_skus:
            self.check_cancelled()
            entry = _r.get_robot(sku) or {}
            joints_pf = entry.get("joints_per_format") or {}
            mjcf_tbl = joints_pf.get("MJCF") or {}
            usd_tbl = joints_pf.get("USD") or {}
            if not (isinstance(mjcf_tbl, dict) and mjcf_tbl):
                continue
            if not (isinstance(usd_tbl, dict) and usd_tbl):
                continue
            n_mjcf = len(mjcf_tbl)
            n_usd = len(usd_tbl)
            if n_mjcf == n_usd:
                continue
            name = entry.get("name") or sku
            # Identify which deploy target is the one losing coverage —
            # the format with fewer joints can't fully replay the
            # trained policy's action vector at deploy time.
            if n_mjcf < n_usd:
                affected = "MuJoCo deploy (MJCF asset has fewer joints)"
            else:
                affected = "IsaacSim deploy (USD asset has fewer joints)"
            log_warning(
                f"[robot_asset_selfcheck] sku={sku!r} ({name}): "
                f"joints_per_format[\"MJCF\"]={n_mjcf} ≠ "
                f"joints_per_format[\"USD\"]={n_usd}. {affected} will be "
                f"unavailable for bundles trained against the format that "
                f"covers more joints — training itself is unaffected. If "
                f"both deploy targets are needed, repoint the smaller-DOF "
                f"asset to a matching variant (for Unitree G1: use "
                f"``menagerie/unitree_g1/scene_with_hands.xml`` alongside "
                f"the IsaacLab G1 USD, both 43-DOF) and re-Dump."
            )
            n_flagged += 1
        return n_flagged

    def _collect_if_pending(
        self, *,
        sku: str, robot_name: str, family: str, fmt: str,
        pending: List[PendingAssignment],
    ) -> None:
        """Pass B — if this (sku, fmt) table contains ANY entry with empty
        ir_role, surface ALL entries (matched + empty) to the dialog.

        Fully-assigned tables contribute zero rows — the user sees
        nothing for robots that are already good.
        """
        from registers import robots as _r

        entry = _r.get_robot(sku) or {}
        joints_table = (entry.get("joints_per_format") or {}).get(fmt) or {}
        bodies_table = (entry.get("bodies_per_format") or {}).get(fmt) or {}
        if not joints_table and not bodies_table:
            # Pass A failed to populate this (sku, fmt) — nothing to assign.
            return

        # Quick check: does this fmt have ANY empty ir_role?
        def _has_empty(table: Dict[str, Any]) -> bool:
            for spec in table.values():
                if isinstance(spec, dict):
                    if not str(spec.get("ir_role") or "").strip():
                        return True
            return False

        if not _has_empty(joints_table) and not _has_empty(bodies_table):
            return  # fully assigned, no dialog noise needed

        # Has gaps → surface every entry (auto-matched ones too, for
        # context + the user can re-pick if the tokeniser guessed wrong).
        for uid, spec in joints_table.items():
            if not isinstance(spec, dict):
                continue
            pending.append(PendingAssignment(
                sku=sku, robot_name=robot_name, family=family,
                fmt=fmt, kind="joint", uid=str(uid),
                name=str(spec.get("name") or uid),
                current=str(spec.get("ir_role") or "").strip(),
            ))
        for uid, spec in bodies_table.items():
            if not isinstance(spec, dict):
                continue
            pending.append(PendingAssignment(
                sku=sku, robot_name=robot_name, family=family,
                fmt=fmt, kind="body", uid=str(uid),
                name=str(spec.get("name") or uid),
                current=str(spec.get("ir_role") or "").strip(),
            ))


__all__ = ["PendingAssignment", "RobotAssetSelfCheckTask"]
