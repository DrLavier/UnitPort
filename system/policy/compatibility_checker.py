from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from .joint_name_utils import canonicalize_joint_names
from .manifest_schema import CheckpointBundle
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
    ) -> CompatReport:
        issues: List[CompatIssue] = []

        actual_obs = obs_builder.expected_dim(env)
        if actual_obs != bundle.obs_dim:
            issues.append(CompatIssue(
                code="obs_dim_mismatch",
                message=(
                    f"Observation dimension mismatch: bundle expects {bundle.obs_dim}, "
                    f"builder produces {actual_obs}"
                ),
                severity=CompatStatus.FAIL,
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
            bundle_canonical = canonicalize_joint_names(bundle.joint_names)
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

        return CompatReport(
            status=_aggregate(issues),
            issues=issues,
            expected_obs_dim=bundle.obs_dim,
            actual_obs_dim=actual_obs,
            expected_action_dim=bundle.action_dim,
            actual_action_dim=actual_action,
        )
