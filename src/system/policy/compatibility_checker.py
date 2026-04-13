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
            # P4 obs_remap rationale: an Isaac-trained bundle that lists
            # ``height_scan`` (or any other Isaac-only obs) in its
            # observation_space.components has bundle.obs_dim much larger
            # than what a flat-ground MjSimEnv can natively produce
            # (45-ish vs 235). The compat checker historically FAILed
            # this hard, which prevented every Isaac bundle from running
            # in Mission. v1.4: downgrade to WARN — ObsBuilder.build()
            # already zero-pads the obs vector at runtime (lines 164-175
            # of obs_builder.py), and the P4 embodiment matcher
            # surfaces the missing-sensor list as obs_remap_warnings
            # before the run starts. The user therefore SEES the gap
            # without the policy crashing on a hard FAIL.
            #
            # The "more dims than bundle expects" direction stays a WARN
            # too — runtime truncates safely.
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

        # ── deploy_contract checks (sim2sim_design.yaml Stage E1) ──────
        # When the bundle ships a deploy_contract, validate it against the
        # bundle / env so contract violations surface as crisp load-time
        # errors instead of garbage at runtime. The validation runs
        # whenever a contract is present — there is no runtime cost when
        # it's absent (legacy bundles).
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
    # deploy_contract validation (Stage E1)
    # ------------------------------------------------------------------

    def _check_deploy_contract(
        self,
        bundle: CheckpointBundle,
        env: SimEnvContext,
        issues: List[CompatIssue],
    ) -> None:
        """Validate the bundle's deploy_contract against the bundle/env.

        Rules (each adds one issue at the indicated severity):

          DC1 (FAIL): contract section is present but DeployContract.from_dict
              raises — schema/length/decimation mismatch.
          DC2 (FAIL): len(contract.joint_sdk_names) != bundle.action_dim.
          DC3 (FAIL): sum(t.dim * t.history_length for t in contract.observations)
              != bundle.obs_dim.
          DC4 (FAIL): mj_model is available AND any name in joint_sdk_names
              fails to resolve via mj_name2id.
          DC5 (WARN): any obs term has history_length > 1. ObsBuilder now
              supports per-term history (Phase 2.A3), but this WARN stays
              as a heads-up that history adds latency; downgrade to nothing
              once we ship a regression test for the multi-history path
              against a real bundle.

        For bundles WITHOUT a contract this method is a no-op — the
        legacy compat rules above already cover those bundles.
        """
        # ── DC1: contract loads at all ────────────────────────────────
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
            # Unexpected error — log as FAIL with the type for debug.
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
            # No contract — legacy bundle, nothing more to check here.
            return

        # ── DC2: joint count agrees with bundle.action_dim ─────────────
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

        # ── DC3: observations sum to bundle.obs_dim ────────────────────
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

        # ── DC4: every contract joint resolves in MJCF (when available) ─
        mj_model = getattr(env, "mj_model", None)
        if mj_model is not None:
            try:
                import mujoco
                missing: List[str] = []
                for name in contract.joint_sdk_names:
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
                            f"that are not in the env's MJCF: {missing}"
                        ),
                        severity=CompatStatus.FAIL,
                        field="deploy_contract.joint_sdk_names",
                    ))
            except ImportError:
                # mujoco not importable in this process — skip the check
                # rather than fabricating a FAIL.
                pass
            except Exception:
                # Defensive: any other model introspection failure should
                # not crash the checker.
                pass

        # ── DC5: history_length > 1 advisory ──────────────────────────
        history_terms = [
            name for name, t in contract.observations.items()
            if t.history_length > 1
        ]
        if history_terms:
            issues.append(CompatIssue(
                code="deploy_contract_history_length",
                message=(
                    f"deploy_contract uses history_length>1 on terms: "
                    f"{history_terms}. ObsBuilder supports this (Phase 2.A3) "
                    f"but the path is newly written — verify the rollout "
                    f"matches training before trusting the policy."
                ),
                severity=CompatStatus.WARN,
                field="deploy_contract.observations",
            ))


# ---------------------------------------------------------------------------
# P4 (mission_design.yaml §6) — embodiment matching
#
# These helpers extend CompatibilityChecker rather than introducing a
# parallel checker framework (per design intent: "EXTEND existing
# module ... Do NOT introduce a parallel checker framework"). They
# operate at the SkillManifest level — a coarser contract than the
# bundle ↔ env check above — and are called by BehaviorNode BEFORE
# PolicyRunner.load() so a mismatch surfaces with structured warnings
# rather than as a cryptic runtime error.
# ---------------------------------------------------------------------------


# Brand prefixes that may appear in SkillManifest.target_robot_models
# (e.g. ``unitree_go2``). The matcher strips these before comparing
# against an actor's bare ``robot_id`` so naming conventions across
# Isaac Lab / menagerie / SDK don't cause spurious WARNs.
_BRAND_PREFIXES = (
    "unitree_",
    "boston_dynamics_",
    "bostondynamics_",
    "spot_",   # already-bare; harmless
)


def _strip_brand_prefix(name: str) -> str:
    """Return *name* with any known ``brand_`` prefix removed.

    ``unitree_go2`` → ``go2``, ``boston_dynamics_spot`` → ``spot``.
    Names without a known prefix are returned unchanged.
    """
    n = (name or "").strip().lower()
    for prefix in _BRAND_PREFIXES:
        if n.startswith(prefix) and len(n) > len(prefix):
            return n[len(prefix):]
    return n


# Heuristic mapping from menagerie robot id → SkillManifest target_robot_family.
# Used to detect "wrong family" mismatches without requiring the user to
# annotate every actor explicitly.
_ROBOT_ID_TO_FAMILY = {
    "go2":  "quadruped",
    "go2w": "quadruped",
    "a1":   "quadruped",
    "b1":   "quadruped",
    "b2":   "quadruped",
    "spot": "quadruped",
    "h1":   "biped",
    "h1_2": "biped",
    "g1":   "biped",
}


def match_actor_to_manifest(actor: Any, manifest: Any) -> CompatReport:
    """Validate an :class:`MjActor` against a :class:`SkillManifest`.

    Checks:
      - target_robot_family matches the actor's robot family (heuristic)
      - target_robot_models, when non-empty, contains the actor's robot_id
      - action_dim from manifest matches actor.n_ctrl

    Returns a :class:`CompatReport`. Severity is FAIL for hard
    incompatibilities (action dim mismatch) and WARN for soft ones
    (family heuristic, model whitelist).
    """
    issues: List[CompatIssue] = []

    if actor is None:
        issues.append(CompatIssue(
            code="actor_missing",
            message="MjActor is None; cannot validate manifest compatibility.",
            severity=CompatStatus.FAIL,
            field="actor",
        ))
        return CompatReport(
            status=CompatStatus.FAIL,
            issues=issues,
            expected_obs_dim=0,
            actual_obs_dim=0,
            expected_action_dim=0,
            actual_action_dim=0,
        )

    actor_id = str(getattr(actor, "robot_id", "") or "").lower()
    n_ctrl = int(getattr(actor, "n_ctrl", 0) or 0)

    if manifest is None:
        # Bundles without a SkillManifest fall back to the bundle-level
        # check above; nothing to validate at the manifest layer.
        return CompatReport(
            status=CompatStatus.PASS,
            issues=[],
            expected_obs_dim=0,
            actual_obs_dim=0,
            expected_action_dim=n_ctrl,
            actual_action_dim=n_ctrl,
        )

    # ---- target_robot_family heuristic ----
    expected_family = str(getattr(manifest, "target_robot_family", "") or "").lower()
    actor_family = _ROBOT_ID_TO_FAMILY.get(actor_id, "")
    if expected_family and actor_family and expected_family != actor_family:
        issues.append(CompatIssue(
            code="robot_family_mismatch",
            message=(
                f"Manifest targets '{expected_family}' but actor '{actor_id}' "
                f"is a '{actor_family}'. Policy may produce out-of-distribution "
                f"actions."
            ),
            severity=CompatStatus.WARN,
            field="target_robot_family",
        ))

    # ---- target_robot_models whitelist ----
    # Naming-convention normalization: Isaac Lab manifests prefix the
    # robot id with the brand (``unitree_go2``) while the menagerie
    # actor's robot_id is the bare model name (``go2``). Strip known
    # brand prefixes from both sides before comparing so the two
    # conventions interoperate cleanly.
    target_models = getattr(manifest, "target_robot_models", None) or []
    if target_models and actor_id:
        normalized_whitelist = {
            _strip_brand_prefix(str(m).lower())
            for m in target_models
        }
        normalized_actor = _strip_brand_prefix(actor_id)
        if normalized_actor not in normalized_whitelist:
            issues.append(CompatIssue(
                code="robot_model_not_in_whitelist",
                message=(
                    f"Manifest whitelists {list(target_models)} but actor is "
                    f"'{actor_id}'. Behaviour may differ from training."
                ),
                severity=CompatStatus.WARN,
                field="target_robot_models",
            ))

    # ---- action_dim ----
    expected_action_dim = int(getattr(manifest, "action_dim", 0) or 0)
    if expected_action_dim > 0 and n_ctrl > 0 and expected_action_dim != n_ctrl:
        issues.append(CompatIssue(
            code="manifest_action_dim_mismatch",
            message=(
                f"Manifest action_dim={expected_action_dim} does not match "
                f"actor.n_ctrl={n_ctrl}. The policy will write to a "
                f"differently-sized control vector."
            ),
            severity=CompatStatus.FAIL,
            field="action_dim",
        ))

    return CompatReport(
        status=_aggregate(issues),
        issues=issues,
        expected_obs_dim=int(getattr(manifest, "observation_dim", 0) or 0),
        actual_obs_dim=0,
        expected_action_dim=expected_action_dim,
        actual_action_dim=n_ctrl,
    )


def match_field_to_manifest(field_obj: Any, manifest: Any) -> CompatReport:
    """Validate an :class:`MjField` against a :class:`SkillManifest`.

    Walks ``manifest.required_sensors`` through :func:`obs_adapter.classify`
    and surfaces every degraded sensor as a WARN-level issue. The field
    is never a hard FAIL on its own — even a fully degraded sensor list
    is acceptable (the policy will just see zero vectors and behave
    OOD), so users can deliberately stress-test bundles against fields
    that lack the training-time sensors.
    """
    from .obs_adapter import classify_required_sensors

    issues: List[CompatIssue] = []
    if manifest is None:
        return CompatReport(
            status=CompatStatus.PASS,
            issues=[],
            expected_obs_dim=0,
            actual_obs_dim=0,
            expected_action_dim=0,
            actual_action_dim=0,
        )

    required = list(getattr(manifest, "required_sensors", []) or [])
    report = classify_required_sensors(required, field_obj=field_obj)
    for decision in report.decisions:
        if decision.warning:
            issues.append(CompatIssue(
                code=decision.warning.split(":", 1)[0],
                message=decision.warning,
                severity=CompatStatus.WARN,
                field=f"required_sensors.{decision.sensor}",
            ))

    return CompatReport(
        status=_aggregate(issues),
        issues=issues,
        expected_obs_dim=int(getattr(manifest, "observation_dim", 0) or 0),
        actual_obs_dim=0,
        expected_action_dim=int(getattr(manifest, "action_dim", 0) or 0),
        actual_action_dim=0,
    )
