# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Bundle ↔ env compatibility checker (Phase 3 port from DEMO).

Verbatim port of ``CompatibilityChecker.check`` (the bundle/env shape +
deploy_contract validation). DEMO's ``match_actor_to_manifest`` and
``match_field_to_manifest`` SkillManifest helpers — and the
``classify_required_sensors`` import they pull in via obs_adapter — are
NOT ported in Phase 3 because (a) the SkillManifest layer hasn't landed
in RELEASE yet and (b) the round-trip smoke goes straight through the
PolicyRunner.load → run_episode path that only consults
``CompatibilityChecker.check``. They land back in along with
``application/skill/``.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from application.training.manifest_schema import CheckpointBundle

from .joint_name_utils import canonicalize_joint_names
from .sim_env_context import SimEnvContext

if TYPE_CHECKING:
    from .action_applier import ActionApplier
    from .obs_builder import ObsBuilder


class CompatStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CompatIssue:
    code: str
    message: str
    severity: CompatStatus
    field: Optional[str] = None


@dataclass
class CompatReport:
    status: CompatStatus
    issues: List[CompatIssue]
    expected_obs_dim: int
    actual_obs_dim: int
    expected_action_dim: int
    actual_action_dim: int

    @property
    def ok(self) -> bool:
        return self.status is not CompatStatus.FAIL


def _aggregate(issues: List[CompatIssue]) -> CompatStatus:
    if any(issue.severity is CompatStatus.FAIL for issue in issues):
        return CompatStatus.FAIL
    if any(issue.severity is CompatStatus.WARN for issue in issues):
        return CompatStatus.WARN
    return CompatStatus.PASS


_SUPPORTED_ACTION_TYPES = {"joint_position", "torque"}


class CompatibilityChecker:
    """Reports compatibility between a CheckpointBundle and a SimEnvContext."""

    def check(
        self,
        bundle: CheckpointBundle,
        env: SimEnvContext,
        obs_builder: "ObsBuilder",
        action_applier: "ActionApplier",
        *,
        robot_sku: str,
    ) -> CompatReport:
        issues: List[CompatIssue] = []

        actual_obs = obs_builder.expected_dim(env)
        if actual_obs != bundle.obs_dim:
            note = (
                "runtime will truncate"
                if actual_obs > bundle.obs_dim
                else "runtime will zero-pad missing components — see "
                     "obs_remap_warnings for the degraded sensor list"
            )
            issues.append(CompatIssue(
                code="obs_dim_mismatch",
                message=(
                    f"Observation dimension mismatch: bundle expects {bundle.obs_dim}, "
                    f"builder produces {actual_obs} ({note})"
                ),
                severity=CompatStatus.WARN,
                field="observation_space.dim",
            ))

        actual_action = action_applier.expected_dim()
        if actual_action != bundle.action_dim:
            issues.append(CompatIssue(
                code="action_dim_mismatch",
                message=(
                    f"Action dimension mismatch: bundle expects {bundle.action_dim}, "
                    f"applier reports {actual_action}"
                ),
                severity=CompatStatus.FAIL,
                field="action_space.dim",
            ))

        if bundle.action_type not in _SUPPORTED_ACTION_TYPES:
            issues.append(CompatIssue(
                code="unsupported_action_type",
                message=(
                    f"Unsupported action type '{bundle.action_type}'. "
                    f"Runtime supports: {sorted(_SUPPORTED_ACTION_TYPES)}"
                ),
                severity=CompatStatus.FAIL,
                field="action_space.type",
            ))

        env_joint_names = getattr(env, "joint_names", None)
        if not env_joint_names:
            issues.append(CompatIssue(
                code="missing_env_joint_names",
                message="Environment provides no joint_names; cannot verify joint compatibility.",
                severity=CompatStatus.FAIL,
                field="joint_names",
            ))
        else:
            # Phase 5: bundle.joint_names carries IR roles. Translate to
            # physical via the bundle's brand+model SKU lookup before
            # comparing with env's MJCF joint names. Translation failure
            # falls back to direct comparison (legacy bundle).
            # SKU-only contract: caller plumbs the canonical SKU; we
            # translate bundle's IR-role joint_names to physical via
            # ir_roles_to_physical_names. NO reverse-derivation from
            # manifest brand/model strings — that was an antipattern
            # that broke whenever the brand string was the human name.
            bundle_joints_for_compare = list(bundle.joint_names)
            try:
                from .joint_name_utils import ir_roles_to_physical_names
                bundle_joints_for_compare = ir_roles_to_physical_names(
                    list(bundle.joint_names), robot_sku
                )
            except ValueError as exc:
                issues.append(CompatIssue(
                    code="ir_role_translation_failed",
                    message=(
                        f"Cannot translate bundle IR joint roles to "
                        f"physical names for robot_sku={robot_sku!r}: {exc}"
                    ),
                    severity=CompatStatus.FAIL,
                    field="robot.joint_names",
                ))

            bundle_canonical = canonicalize_joint_names(bundle_joints_for_compare)
            env_canonical = canonicalize_joint_names(env_joint_names)
            bundle_set = set(bundle_canonical)
            env_set = set(env_canonical)

            if bundle_set != env_set:
                extra_in_bundle = sorted(bundle_set - env_set)
                extra_in_env = sorted(env_set - bundle_set)
                if extra_in_bundle:
                    issues.append(CompatIssue(
                        code="joint_name_mismatch",
                        message=(
                            f"Bundle requires joints not present in env: {extra_in_bundle}. "
                            f"Remapping is impossible."
                        ),
                        severity=CompatStatus.FAIL,
                        field="robot.joint_names",
                    ))
                else:
                    issues.append(CompatIssue(
                        code="joint_order_mismatch",
                        message=(
                            "Bundle joints are a canonical subset of env joints; "
                            f"runtime will remap the overlapping joints. Extra env joints: {extra_in_env}"
                        ),
                        severity=CompatStatus.WARN,
                        field="robot.joint_names",
                    ))
            elif bundle_canonical != env_canonical or list(bundle.joint_names) != list(env_joint_names):
                issues.append(CompatIssue(
                    code="joint_order_mismatch",
                    message=(
                        "Bundle and env joint names are compatible after canonical remapping; "
                        "runtime will reorder actions to match the env."
                    ),
                    severity=CompatStatus.WARN,
                    field="robot.joint_names",
                ))

        try:
            specs = obs_builder.get_component_specs(env)
            known = {spec.name for spec in specs}
            unknown = [name for name in obs_builder._component_order if name not in known]
            for name in unknown:
                issues.append(CompatIssue(
                    code=f"unknown_obs_component.{name}",
                    message=f"Unknown observation component '{name}'; will be zero-filled.",
                    severity=CompatStatus.WARN,
                    field="observation_space",
                ))
        except Exception:
            pass

        # ── deploy_contract checks ─────────────────────────────────────
        self._check_deploy_contract(bundle, env, issues, robot_sku=robot_sku)

        return CompatReport(
            status=_aggregate(issues),
            issues=issues,
            expected_obs_dim=bundle.obs_dim,
            actual_obs_dim=actual_obs,
            expected_action_dim=bundle.action_dim,
            actual_action_dim=actual_action,
        )

    # ------------------------------------------------------------------
    # deploy_contract validation
    # ------------------------------------------------------------------

    def _check_deploy_contract(
        self,
        bundle: CheckpointBundle,
        env: SimEnvContext,
        issues: List[CompatIssue],
        *,
        robot_sku: str,
    ) -> None:
        """Validate the bundle's deploy_contract against bundle/env.

        DC1 (FAIL): contract section present but DeployContract.from_dict raises.
        DC2 (FAIL): len(contract.joint_sdk_names) != bundle.action_dim.
        DC3 (FAIL): sum(t.dim * t.history_length) != bundle.obs_dim.
        DC4 (FAIL): mj_model available AND any joint_sdk_name fails mj_name2id.
                    (Translates IR roles → physical names via ``robot_sku``.)
        DC5 (WARN): any obs term has history_length > 1 (advisory).

        ``robot_sku`` is the caller-supplied canonical SKU (= manifest.robot.sku);
        DC4 needs it to translate the contract's IR-role joint names to physical
        MJCF names. It was previously read as a free variable here — a latent
        NameError that only the ``mj_model is not None`` path (real deploy /
        sim2sim, not unit tests with mj_model=None) would hit.
        """
        try:
            contract = bundle.deploy_contract
        except ValueError as exc:
            issues.append(CompatIssue(
                code="deploy_contract_invalid",
                message=(
                    f"deploy_contract section present but failed validation: "
                    f"{exc}"
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract",
            ))
            return
        except Exception as exc:
            issues.append(CompatIssue(
                code="deploy_contract_load_error",
                message=(
                    f"deploy_contract load raised {type(exc).__name__}: {exc}"
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract",
            ))
            return

        if contract is None:
            return

        # DC2
        if len(contract.joint_sdk_names) != bundle.action_dim:
            issues.append(CompatIssue(
                code="deploy_contract_joint_count_mismatch",
                message=(
                    f"deploy_contract has {len(contract.joint_sdk_names)} "
                    f"joints but bundle.action_dim={bundle.action_dim}"
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract.joint_sdk_names",
            ))

        # DC3 — use the contract's own ``total_obs_dim`` (the single source of
        # truth) so the per-item obs TAIL is counted. SB3 per-item-reward
        # bundles append ``[cmd_norm, weight per item]`` (per_item_obs.tail_dim
        # = 1 + n_items) to the policy obs, so the network input (bundle.obs_dim)
        # is ``sum(observation terms) + tail``. Summing only the observation
        # terms here falsely flagged every per-item bundle as a dim mismatch.
        base_obs_dim = sum(
            t.dim * t.history_length for t in contract.observations.values()
        )
        contract_obs_dim = contract.total_obs_dim
        if contract_obs_dim != bundle.obs_dim:
            tail = contract_obs_dim - base_obs_dim
            issues.append(CompatIssue(
                code="deploy_contract_obs_dim_mismatch",
                message=(
                    f"deploy_contract obs total {contract_obs_dim} "
                    f"(terms {base_obs_dim} dim × history_length + per-item "
                    f"tail {tail}) but bundle.obs_dim={bundle.obs_dim}"
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract.observations",
            ))

        # DC4 — joint_sdk_names carries IR role names (Phase 5 contract).
        # Translate to physical names via the caller-supplied robot_sku
        # (= manifest.robot.sku, the single source of truth) before MJCF
        # lookup; mirrors joint_space.joint_spaces_from_deploy_contract.
        # The contract itself no longer stores a SKU.
        mj_model = getattr(env, "mj_model", None)
        if mj_model is not None:
            try:
                import mujoco
                contract_sku = (robot_sku or "").strip()
                names_to_check: List[str] = list(contract.joint_sdk_names)
                if contract_sku:
                    try:
                        from application.service.runtime.policy.joint_name_utils import (
                            ir_roles_to_physical_names,
                        )
                        names_to_check = ir_roles_to_physical_names(
                            contract.joint_sdk_names, contract_sku
                        )
                    except ValueError:
                        # IR translation failed — fall through to direct
                        # MJCF lookup so the existing error message still
                        # surfaces the missing names.
                        names_to_check = list(contract.joint_sdk_names)
                missing: List[str] = []
                for name in names_to_check:
                    jid = mujoco.mj_name2id(
                        mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    if jid < 0:
                        missing.append(name)
                if missing:
                    issues.append(CompatIssue(
                        code="deploy_contract_mjcf_joint_missing",
                        message=(
                            f"deploy_contract names {len(missing)} joint(s) "
                            f"that are not in the env's MJCF: {missing} "
                            f"(robot_sku={contract_sku!r})"
                        ),
                        severity=CompatStatus.FAIL,
                        field="deploy_contract.joint_sdk_names",
                    ))
            except ImportError:
                pass
            except Exception:
                pass

        # DC5
        history_terms = [
            name for name, t in contract.observations.items()
            if t.history_length > 1
        ]
        if history_terms:
            issues.append(CompatIssue(
                code="deploy_contract_history_length",
                message=(
                    f"deploy_contract uses history_length>1 on terms: "
                    f"{history_terms}. ObsBuilder supports this — verify "
                    f"the rollout matches training before trusting the policy."
                ),
                severity=CompatStatus.WARN,
                field="deploy_contract.observations",
            ))

        # ── B1 policy contract (PC*) ────────────────────────────────────────
        # Load-time half of the policy-contract two-gate (export-strict half is
        # bundle_exporter.assert_policy_contract_consistent). DENYLIST → FAIL,
        # WARNING → WARN; an absent snapshot (legacy bundle) → skip + WARN
        # (UniLab "no snapshot → skip"; §8(c)). FAILs flow to CompatStatus.FAIL,
        # which PolicyRunner.load already raises on — no new raise site.
        pc = getattr(contract, "policy_contract", None)
        if pc is None:
            issues.append(CompatIssue(
                code="policy_contract_absent",
                message=(
                    "deploy_contract carries no policy_contract snapshot (bundle "
                    "predates B1). Load-checkable policy I/O enforcement is "
                    "skipped; re-export through the current pipeline to enable it."
                ),
                severity=CompatStatus.WARN,
                field="deploy_contract.policy_contract",
            ))
            return

        # PC1 (FAIL): snapshot obs input dim != bundle.obs_dim.
        if int(pc.policy_input_dim) != int(bundle.obs_dim):
            issues.append(CompatIssue(
                code="policy_contract_obs_dim_mismatch",
                message=(
                    f"policy_contract.policy_input_dim {pc.policy_input_dim} != "
                    f"bundle.obs_dim {bundle.obs_dim} — re-export the bundle."
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract.policy_contract.policy_input_dim",
            ))

        # PC2 (FAIL): snapshot action output dim != bundle.action_dim.
        if int(pc.policy_output_dim) != int(bundle.action_dim):
            issues.append(CompatIssue(
                code="policy_contract_action_dim_mismatch",
                message=(
                    f"policy_contract.policy_output_dim {pc.policy_output_dim} != "
                    f"bundle.action_dim {bundle.action_dim} — re-export the bundle."
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract.policy_contract.policy_output_dim",
            ))

        # PC3 (FAIL): recurrent_shape mirror must agree with contract.recurrent
        # (both present-and-equal, or both absent).
        rec = getattr(contract, "recurrent", None)
        mirror = pc.recurrent_shape
        if (rec is not None) != (mirror is not None):
            issues.append(CompatIssue(
                code="policy_contract_recurrent_mismatch",
                message=(
                    f"policy_contract.recurrent_shape present={mirror is not None} "
                    f"disagrees with deploy_contract.recurrent present="
                    f"{rec is not None} — re-export the bundle."
                ),
                severity=CompatStatus.FAIL,
                field="deploy_contract.policy_contract.recurrent_shape",
            ))
        elif rec is not None and mirror is not None:
            if (
                str(mirror.get("rnn_type", "")) != str(rec.rnn_type)
                or int(mirror.get("hidden_size", 0)) != int(rec.hidden_size)
                or int(mirror.get("num_layers", 0)) != int(rec.num_layers)
            ):
                issues.append(CompatIssue(
                    code="policy_contract_recurrent_mismatch",
                    message=(
                        f"policy_contract.recurrent_shape {mirror} != "
                        f"deploy_contract.recurrent (rnn_type={rec.rnn_type}, "
                        f"hidden_size={rec.hidden_size}, num_layers={rec.num_layers}) "
                        f"— re-export the bundle."
                    ),
                    severity=CompatStatus.FAIL,
                    field="deploy_contract.policy_contract.recurrent_shape",
                ))

        # PCW3 (WARN): convention / joint-format mirrors disagree with the
        # manifest top-level (a frozen graph can't be load-checked for these,
        # so drift is advisory, not fatal).
        raw_manifest = getattr(bundle, "raw_manifest", None) or {}
        manifest_conv = str(raw_manifest.get("inference_convention", "") or "")
        manifest_fmt = str(raw_manifest.get("joint_array_format", "") or "")
        if pc.inference_convention and manifest_conv and pc.inference_convention != manifest_conv:
            issues.append(CompatIssue(
                code="policy_contract_convention_drift",
                message=(
                    f"policy_contract.inference_convention {pc.inference_convention!r} "
                    f"!= manifest.inference_convention {manifest_conv!r}."
                ),
                severity=CompatStatus.WARN,
                field="deploy_contract.policy_contract.inference_convention",
            ))
        if pc.joint_array_format and manifest_fmt and pc.joint_array_format != manifest_fmt:
            issues.append(CompatIssue(
                code="policy_contract_joint_format_drift",
                message=(
                    f"policy_contract.joint_array_format {pc.joint_array_format!r} "
                    f"!= manifest.joint_array_format {manifest_fmt!r}."
                ),
                severity=CompatStatus.WARN,
                field="deploy_contract.policy_contract.joint_array_format",
            ))
