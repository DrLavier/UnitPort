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
        self._check_deploy_contract(bundle, env, issues)

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
    ) -> None:
        """Validate the bundle's deploy_contract against bundle/env.

        DC1 (FAIL): contract section present but DeployContract.from_dict raises.
        DC2 (FAIL): len(contract.joint_sdk_names) != bundle.action_dim.
        DC3 (FAIL): sum(t.dim * t.history_length) != bundle.obs_dim.
        DC4 (FAIL): mj_model available AND any joint_sdk_name fails mj_name2id.
        DC5 (WARN): any obs term has history_length > 1 (advisory).
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

        # DC3
        contract_obs_dim = sum(
            t.dim * t.history_length for t in contract.observations.values()
        )
        if contract_obs_dim != bundle.obs_dim:
            issues.append(CompatIssue(
                code="deploy_contract_obs_dim_mismatch",
                message=(
                    f"deploy_contract observations sum to {contract_obs_dim} "
                    f"(dim × history_length) but bundle.obs_dim={bundle.obs_dim}"
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
