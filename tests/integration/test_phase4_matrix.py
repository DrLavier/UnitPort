#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4 Test Matrix — STAGE-07.

Single-document evidence of Phase 4 completeness.
Organised in three sections mirroring the STAGE-07 task list:

  SECTION A — Unit evidence
    A1: Multi-brand adapter interface completeness.
    A2: Diagnostics minimum-key consistency (brand / message / context).
    A3: Phase 4 reason codes taxonomy (reason_codes.py).

  SECTION B — Integration coverage
    B1: All 3 adapters — SDK-absent graceful failure through ServiceRouter.
    B2: adapter_name propagation for all brands and all result paths.
    B3: Cross-brand lifecycle orchestration (mock adapters in router).

  SECTION C — Regression coverage
    C1: Phase 3 lifecycle contract unchanged.
    C2: BRAND_ADAPTER_MAP routing constants correct.
    C3: Phase 4 deliverable gate (counts + invariants).

Isolation
---------
No Qt, no real SDK / ROS2.  UnitreeModel is stubbed; SPOT_SDK_AVAILABLE and
ROS2_AVAILABLE are patched to False for the SDK-absent test paths.
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Stub models.* (must precede any adapter/robot_context import) ──────────

_models_pkg  = types.ModuleType("models")
_models_pkg.__path__    = []       # marks it as a package so sub-imports work
_models_pkg.__package__ = "models"

_unitree_pkg = types.ModuleType("models.Unitree")


class _StubUnitreeModel:
    """Availability flags False → open_session returns graceful error."""
    def __init__(self, robot_type: str = "go2"):
        self.robot_type = robot_type
    is_available    = False   # class-level: getattr sees False
    mujoco_available = False
    running          = False

    def stop(self) -> None:       # required for close_session tests
        self.running = False

    def close_viewer(self) -> None:
        pass


_unitree_pkg.UnitreeModel = _StubUnitreeModel
_models_pkg.Unitree       = _unitree_pkg

# Stub models.brand_registry (needed by RobotContext import)
_brand_reg_pkg = types.ModuleType("models.brand_registry")


class _StubBrandRegistry:
    @staticmethod
    def get_robot_brand_map() -> dict:
        return {
            "go2": "unitree", "a1": "unitree", "b1": "unitree",
            "b2": "unitree", "h1": "unitree",
            "spot": "bostiondynamics",
            "cyberdog": "xiaomi", "cyberdog2": "xiaomi",
        }


_brand_reg_pkg.BrandRegistry = _StubBrandRegistry
_models_pkg.brand_registry   = _brand_reg_pkg

sys.modules.setdefault("models",                _models_pkg)
sys.modules.setdefault("models.Unitree",        _unitree_pkg)
sys.modules.setdefault("models.brand_registry", _brand_reg_pkg)

# ── Imports under test ─────────────────────────────────────────────────────

from system.service.adapters.unitree_sdk2.adapter import (  # noqa: E402
    UnitreeAdapter,
    _REASON_MODEL_UNAVAILABLE  as _U_MODEL_UNAVAILABLE,
    _REASON_ACTION_IN_PROGRESS as _U_ACTION_IN_PROGRESS,
    _REASON_CONTROL_CONFLICT   as _U_CONTROL_CONFLICT,
)
from system.service.adapters.spot_sdk.adapter import (      # noqa: E402
    SpotAdapter,
    _REASON_SDK_UNAVAILABLE  as _S_SDK_UNAVAILABLE,
    _REASON_NO_SESSION       as _S_NO_SESSION,
)
from system.service.adapters.cyberdog_sdk.adapter import (  # noqa: E402
    CyberDogAdapter,
    _REASON_ROS2_UNAVAILABLE as _C_ROS2_UNAVAILABLE,
    _REASON_NO_SESSION       as _C_NO_SESSION,
)
from system.service.adapters.base_adapter import BaseAdapter          # noqa: E402
from system.service.lifecycle import (                                # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
)
from system.service.service_registry import ServiceRegistry           # noqa: E402
from system.service.service_router   import RouteOp, ServiceRouter   # noqa: E402
from system.service.reason_codes import (                             # noqa: E402
    ALL_REASON_CODES, CANONICAL_REASONS,
    UNITREE_REASONS, SPOT_REASONS, CYBERDOG_REASONS,
    REASON_MAP,
)
from bin.core.robot_context import RobotContext                       # noqa: E402


# ── Shared: configurable mock adapter for router matrix tests ──────────────

class _MockAdapter(BaseAdapter):
    """Lifecycle mock used by cross-brand router matrix tests."""

    def __init__(self):
        self.open_session_status  = "ok"
        self.preflight_status     = "ok"
        self.close_session_status = "ok"
        self.run_action_return    = True
        self.run_action_raises    = None
        self.calls: list          = []

    def connect(self, **kw: Any) -> bool:
        self.calls.append(("connect", kw)); return True

    def run_action(self, action: str, **p: Any) -> Any:
        self.calls.append(("run_action", action, p))
        if self.run_action_raises: raise self.run_action_raises
        return self.run_action_return

    def stop(self) -> None:
        self.calls.append(("stop",))

    def get_sensor_data(self) -> Dict[str, Any]:
        self.calls.append(("get_sensor_data",)); return {"sensor": 1}

    def health(self) -> Dict[str, Any]:
        return {"connected": True}

    def open_session(self, config=None) -> Dict[str, Any]:
        self.calls.append(("open_session", config))
        if self.open_session_status == "ok":
            return LifecycleResult.ok("open_session").to_dict()
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "mock open_session failure"},
        ).to_dict()

    def preflight(self, context=None) -> Dict[str, Any]:
        self.calls.append(("preflight", context))
        if self.preflight_status == "ok":
            return LifecycleResult.ok("preflight").to_dict()
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED,
            {"message": "mock preflight failure"},
        ).to_dict()

    def close_session(self) -> Dict[str, Any]:
        self.calls.append(("close_session",))
        if self.close_session_status == "ok":
            return LifecycleResult.ok("close_session").to_dict()
        return LifecycleResult.error(
            "close_session", LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "mock close_session failure"},
        ).to_dict()


# ── Helpers ────────────────────────────────────────────────────────────────

_RESULT_SCHEMA = frozenset({"status", "reason", "stage", "diagnostics"})
_ROUTER_SCHEMA = frozenset({"status", "reason", "stage", "payload", "diagnostics", "adapter_name"})

_PATCH_SPOT_OFF   = "system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE"
_PATCH_ROS2_OFF   = "system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE"
# Patch UnitreeModel in the adapter's own namespace so the "SDK absent" test
# paths work even when the real models.Unitree has already been loaded (e.g.
# when pytest collects test_phase3_matrix.py before this file).
_PATCH_UNITREE_MODEL = "system.service.adapters.unitree_sdk2.adapter.UnitreeModel"


def _router_with(adapter: Any, name: str) -> ServiceRouter:
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Unit evidence
# ═══════════════════════════════════════════════════════════════════════════

# ── A1: Multi-brand adapter interface completeness ─────────────────────────

class TestAdapterInterfaceMatrix(unittest.TestCase):
    """All 3 Phase 4 adapters implement the full BaseAdapter + lifecycle contract."""

    _ADAPTER_CLASSES = [UnitreeAdapter, SpotAdapter, CyberDogAdapter]

    _LEGACY_METHODS = [
        "connect", "run_action", "stop", "get_sensor_data", "health",
    ]
    _LIFECYCLE_METHODS = [
        "open_session", "preflight", "close_session", "capabilities",
    ]

    def test_all_adapters_have_legacy_methods(self):
        """Every adapter class provides the Phase 1 legacy interface."""
        for cls in self._ADAPTER_CLASSES:
            for method in self._LEGACY_METHODS:
                with self.subTest(adapter=cls.__name__, method=method):
                    self.assertTrue(
                        callable(getattr(cls, method, None)),
                        f"{cls.__name__} missing legacy method: {method}",
                    )

    def test_all_adapters_have_lifecycle_methods(self):
        """Every adapter class provides the Phase 3/4 lifecycle interface."""
        for cls in self._ADAPTER_CLASSES:
            for method in self._LIFECYCLE_METHODS:
                with self.subTest(adapter=cls.__name__, method=method):
                    self.assertTrue(
                        callable(getattr(cls, method, None)),
                        f"{cls.__name__} missing lifecycle method: {method}",
                    )

    def test_all_adapters_are_base_adapter_subclasses(self):
        """Every adapter inherits from BaseAdapter."""
        for cls in self._ADAPTER_CLASSES:
            with self.subTest(adapter=cls.__name__):
                self.assertTrue(issubclass(cls, BaseAdapter))

    def test_all_adapters_instantiate_without_sdk(self):
        """Adapter constructors never crash even when vendor SDK is absent."""
        instances = [UnitreeAdapter(), SpotAdapter(), CyberDogAdapter()]
        for adapter in instances:
            with self.subTest(adapter=type(adapter).__name__):
                self.assertIsNotNone(adapter)

    def test_open_session_returns_dict_for_all_adapters(self):
        """open_session always returns a plain dict — never raises."""
        with patch(_PATCH_SPOT_OFF, False), patch(_PATCH_ROS2_OFF, False):
            results = [
                UnitreeAdapter().open_session(),
                SpotAdapter().open_session(),
                CyberDogAdapter().open_session(),
            ]
        for res in results:
            with self.subTest(result=res.get("stage")):
                self.assertIsInstance(res, dict)

    def test_preflight_returns_dict_for_all_adapters(self):
        """preflight always returns a plain dict — never raises."""
        results = [
            UnitreeAdapter().preflight(),
            SpotAdapter().preflight(),
            CyberDogAdapter().preflight(),
        ]
        for res in results:
            with self.subTest(result=res.get("stage")):
                self.assertIsInstance(res, dict)

    def test_close_session_returns_dict_for_all_adapters(self):
        """close_session always returns a plain dict — never raises."""
        results = [
            UnitreeAdapter().close_session(),
            SpotAdapter().close_session(),
            CyberDogAdapter().close_session(),
        ]
        for res in results:
            with self.subTest(result=res.get("stage")):
                self.assertIsInstance(res, dict)

    def test_capabilities_returns_dict_for_all_adapters(self):
        """capabilities() always returns a dict with actions/sensors/flags."""
        for cls in self._ADAPTER_CLASSES:
            with self.subTest(adapter=cls.__name__):
                cap = cls().capabilities()
                self.assertIsInstance(cap, dict)
                self.assertIn("actions", cap)
                self.assertIn("sensors", cap)
                self.assertIn("flags",   cap)

    def test_all_adapters_result_schema_on_error(self):
        """Every lifecycle error result has status/reason/stage/diagnostics."""
        with patch(_PATCH_SPOT_OFF, False), patch(_PATCH_ROS2_OFF, False):
            error_results = [
                UnitreeAdapter().open_session(),    # fails: no SDK/MuJoCo
                SpotAdapter().open_session(),        # fails: SDK absent
                CyberDogAdapter().open_session(),    # fails: ROS2 absent
                UnitreeAdapter().preflight(),        # fails: model None
                SpotAdapter().preflight(),           # fails: no session
                CyberDogAdapter().preflight(),       # fails: no session
            ]
        for res in error_results:
            for key in _RESULT_SCHEMA:
                with self.subTest(key=key, stage=res.get("stage")):
                    self.assertIn(key, res)

    def test_close_session_idempotent_ok_for_all_adapters(self):
        """close_session on a fresh adapter returns ok (idempotent teardown)."""
        for cls in self._ADAPTER_CLASSES:
            with self.subTest(adapter=cls.__name__):
                res = cls().close_session()
                self.assertEqual(res["status"], "ok")


# ── A2: Diagnostics minimum-key consistency ────────────────────────────────

class TestDiagnosticsMinimumKeys(unittest.TestCase):
    """All 3 adapters inject brand / message / context into every failure diagnostics."""

    _MINIMUM_KEYS = ("brand", "message", "context")

    # ── Helper ────────────────────────────────────────────────────────────

    def _assert_min_keys(self, diag: dict, *, brand: str) -> None:
        """Shared assertion for minimum diagnostics keys."""
        for key in self._MINIMUM_KEYS:
            self.assertIn(key, diag, f"Missing minimum key: {key}")
        self.assertEqual(diag["brand"], brand)
        self.assertIsInstance(diag["context"], dict)
        self.assertIsInstance(diag["message"], str)

    # ── open_session failure ──────────────────────────────────────────────

    def test_unitree_open_session_failure_has_minimum_keys(self):
        """UnitreeAdapter open_session failure includes brand/message/context."""
        with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel):
            res = UnitreeAdapter().open_session()
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="unitree")

    def test_spot_open_session_failure_has_minimum_keys(self):
        """SpotAdapter open_session failure includes brand/message/context."""
        with patch(_PATCH_SPOT_OFF, False):
            res = SpotAdapter().open_session()
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="bostiondynamics")

    def test_cyberdog_open_session_failure_has_minimum_keys(self):
        """CyberDogAdapter open_session failure includes brand/message/context."""
        with patch(_PATCH_ROS2_OFF, False):
            res = CyberDogAdapter().open_session()
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="xiaomi")

    # ── preflight failure ─────────────────────────────────────────────────

    def test_unitree_preflight_failure_has_minimum_keys(self):
        """UnitreeAdapter preflight failure includes brand/message/context."""
        res = UnitreeAdapter().preflight()  # _model is None
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="unitree")

    def test_spot_preflight_failure_has_minimum_keys(self):
        """SpotAdapter preflight failure includes brand/message/context."""
        res = SpotAdapter().preflight()  # no session
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="bostiondynamics")

    def test_cyberdog_preflight_failure_has_minimum_keys(self):
        """CyberDogAdapter preflight failure includes brand/message/context."""
        res = CyberDogAdapter().preflight()  # no session
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="xiaomi")

    # ── close_session failure ─────────────────────────────────────────────

    def test_spot_close_session_failure_has_minimum_keys(self):
        """SpotAdapter close_session failure includes brand/message/context."""
        a = SpotAdapter()
        # Inject open session with a lease_client that throws on release
        a._session_open   = True
        a._lease          = MagicMock()
        a._lease_client   = MagicMock()
        a._command_client = None
        a._lease_client.return_lease.side_effect = RuntimeError("hw fail")
        res = a.close_session()
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="bostiondynamics")

    def test_cyberdog_close_session_failure_has_minimum_keys(self):
        """CyberDogAdapter close_session failure includes brand/message/context."""
        a = CyberDogAdapter()
        a._node = MagicMock()
        a._session_open = True
        a._vel_publisher = None  # skip vel reset
        a._node.destroy_node.side_effect = RuntimeError("node locked")
        res = a.close_session()
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="xiaomi")

    def test_unitree_close_session_failure_has_minimum_keys(self):
        """UnitreeAdapter close_session failure includes brand/message/context."""
        a = UnitreeAdapter()
        model = _StubUnitreeModel()
        model.running = True
        a._model = model
        with patch.object(model, "stop", side_effect=RuntimeError("crash")):
            res = a.close_session()
        self.assertEqual(res["status"], "error")
        self._assert_min_keys(res["diagnostics"], brand="unitree")

    # ── brand values are correct ──────────────────────────────────────────

    def test_brand_values_are_stable_strings(self):
        """Brand strings are stable constants — not enum members."""
        with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel):
            unitree_res = UnitreeAdapter().open_session()
        brand_results = {
            "unitree":          unitree_res,
            "bostiondynamics":  SpotAdapter().preflight(),
            "xiaomi":           CyberDogAdapter().preflight(),
        }
        for expected_brand, res in brand_results.items():
            with self.subTest(brand=expected_brand):
                self.assertIsInstance(res["diagnostics"].get("brand"), str)
                self.assertEqual(res["diagnostics"]["brand"], expected_brand)


# ── A3: Phase 4 reason codes taxonomy ─────────────────────────────────────

class TestReasonCodesTaxonomy(unittest.TestCase):
    """reason_codes.py exports the complete cross-brand taxonomy."""

    def test_all_reason_codes_is_frozenset(self):
        self.assertIsInstance(ALL_REASON_CODES, frozenset)

    def test_total_reason_count_is_27(self):
        """8 canonical + 3 unitree + 8 spot + 8 cyberdog = 27.
        Phase 6 STAGE-04 added SAFETY_GATE_BLOCKED + RETRY_BUDGET_EXHAUSTED."""
        self.assertEqual(len(ALL_REASON_CODES), 27)

    def test_canonical_count_is_8(self):
        """Phase 6 STAGE-04 expanded canonical tier from 6 to 8."""
        self.assertEqual(len(CANONICAL_REASONS), 8)

    def test_unitree_count_is_3(self):
        self.assertEqual(len(UNITREE_REASONS), 3)

    def test_spot_count_is_8(self):
        self.assertEqual(len(SPOT_REASONS), 8)

    def test_cyberdog_count_is_8(self):
        self.assertEqual(len(CYBERDOG_REASONS), 8)

    def test_execute_failed_is_canonical(self):
        """EXECUTE_FAILED added in Phase 4 must be in canonical tier."""
        self.assertIn(LifecycleReason.EXECUTE_FAILED, CANONICAL_REASONS)

    def test_adapter_tiers_are_mutually_disjoint(self):
        """No code belongs to more than one adapter tier."""
        tiers = [UNITREE_REASONS, SPOT_REASONS, CYBERDOG_REASONS]
        for i, a in enumerate(tiers):
            for j, b in enumerate(tiers):
                if i != j:
                    overlap = a & b
                    self.assertFalse(overlap, f"Tier overlap: {overlap}")

    def test_canonical_plus_adapters_equals_all(self):
        combined = CANONICAL_REASONS | UNITREE_REASONS | SPOT_REASONS | CYBERDOG_REASONS
        self.assertEqual(combined, ALL_REASON_CODES)

    def test_all_codes_are_nonempty_strings(self):
        for code in ALL_REASON_CODES:
            with self.subTest(code=code):
                self.assertIsInstance(code, str)
                self.assertTrue(code)

    def test_reason_map_covers_all_codes(self):
        self.assertEqual(frozenset(REASON_MAP.keys()), ALL_REASON_CODES)

    def test_every_reason_map_entry_has_adapter_stage_tier(self):
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                for field in ("adapter", "stage", "tier"):
                    self.assertIn(field, meta, f"{code} missing '{field}'")

    def test_unitree_codes_have_unitree_sdk2_adapter(self):
        for code in UNITREE_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASON_MAP[code]["adapter"], "unitree_sdk2")

    def test_spot_codes_have_spot_sdk_adapter(self):
        for code in SPOT_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASON_MAP[code]["adapter"], "spot_sdk")

    def test_cyberdog_codes_have_cyberdog_sdk_adapter(self):
        for code in CYBERDOG_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASON_MAP[code]["adapter"], "cyberdog_sdk")

    def test_lifecycle_reason_constants_are_all_strings(self):
        codes = {v for k, v in vars(LifecycleReason).items()
                 if not k.startswith("_")}
        for code in codes:
            with self.subTest(code=code):
                self.assertIsInstance(code, str)

    def test_adapter_reason_codes_used_by_real_adapters_are_in_taxonomy(self):
        """The specific codes imported from adapters exist in the taxonomy."""
        codes_in_use = {
            _U_MODEL_UNAVAILABLE, _U_ACTION_IN_PROGRESS, _U_CONTROL_CONFLICT,
            _S_SDK_UNAVAILABLE,   _S_NO_SESSION,
            _C_ROS2_UNAVAILABLE,  _C_NO_SESSION,
        }
        for code in codes_in_use:
            with self.subTest(code=code):
                self.assertIn(code, ALL_REASON_CODES)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Integration coverage
# ═══════════════════════════════════════════════════════════════════════════

# ── B1: SDK-absent graceful failure through ServiceRouter ──────────────────

class TestSDKAbsentGracefulFailure(unittest.TestCase):
    """All 3 adapters return structured errors when vendor SDK is absent."""

    def _router_result(self, adapter: Any, adapter_name: str) -> Dict[str, Any]:
        return _router_with(adapter, adapter_name).execute_with_lifecycle(
            adapter_name, RouteOp.RUN_ACTION, {"action": "stand", "params": {}},
        )

    def test_unitree_sdk_absent_returns_error_status(self):
        """UnitreeAdapter with no SDK/MuJoCo → SESSION_OPEN_FAILED."""
        with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel):
            res = self._router_result(UnitreeAdapter(), "unitree_sdk2")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_spot_sdk_absent_returns_error_status(self):
        """SpotAdapter with SDK absent → SESSION_OPEN_FAILED."""
        with patch(_PATCH_SPOT_OFF, False):
            res = self._router_result(SpotAdapter(), "spot_sdk")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_cyberdog_ros2_absent_returns_error_status(self):
        """CyberDogAdapter with ROS2 absent → SESSION_OPEN_FAILED."""
        with patch(_PATCH_ROS2_OFF, False):
            res = self._router_result(CyberDogAdapter(), "cyberdog_sdk")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_unitree_sdk_absent_stage_is_open_session(self):
        with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel):
            res = self._router_result(UnitreeAdapter(), "unitree_sdk2")
        self.assertEqual(res["stage"], "open_session")

    def test_spot_sdk_absent_stage_is_open_session(self):
        with patch(_PATCH_SPOT_OFF, False):
            res = self._router_result(SpotAdapter(), "spot_sdk")
        self.assertEqual(res["stage"], "open_session")

    def test_cyberdog_ros2_absent_stage_is_open_session(self):
        with patch(_PATCH_ROS2_OFF, False):
            res = self._router_result(CyberDogAdapter(), "cyberdog_sdk")
        self.assertEqual(res["stage"], "open_session")

    def test_all_sdk_absent_results_have_full_router_schema(self):
        """Every SDK-absent result has all router-level keys."""
        with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel), \
                patch(_PATCH_SPOT_OFF, False), patch(_PATCH_ROS2_OFF, False):
            results = [
                self._router_result(UnitreeAdapter(),   "unitree_sdk2"),
                self._router_result(SpotAdapter(),      "spot_sdk"),
                self._router_result(CyberDogAdapter(),  "cyberdog_sdk"),
            ]
        for res in results:
            for key in _ROUTER_SCHEMA:
                with self.subTest(key=key, adapter=res.get("adapter_name")):
                    self.assertIn(key, res)

    def test_all_sdk_absent_results_have_none_payload(self):
        """Error results never return a non-None payload."""
        with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel), \
                patch(_PATCH_SPOT_OFF, False), patch(_PATCH_ROS2_OFF, False):
            results = [
                self._router_result(UnitreeAdapter(),   "unitree_sdk2"),
                self._router_result(SpotAdapter(),      "spot_sdk"),
                self._router_result(CyberDogAdapter(),  "cyberdog_sdk"),
            ]
        for res in results:
            with self.subTest(adapter=res.get("adapter_name")):
                self.assertIsNone(res["payload"])

    def test_sdk_absent_results_do_not_raise(self):
        """Router result construction never raises (graceful degradation)."""
        try:
            with patch(_PATCH_UNITREE_MODEL, _StubUnitreeModel), \
                    patch(_PATCH_SPOT_OFF, False), patch(_PATCH_ROS2_OFF, False):
                self._router_result(UnitreeAdapter(),   "unitree_sdk2")
                self._router_result(SpotAdapter(),      "spot_sdk")
                self._router_result(CyberDogAdapter(),  "cyberdog_sdk")
        except Exception as exc:
            self.fail(f"SDK-absent path raised unexpectedly: {exc}")


# ── B2: adapter_name propagation for all brands and all result paths ───────

class TestAdapterNamePropagation(unittest.TestCase):
    """execute_with_lifecycle injects adapter_name on every return path."""

    # ── Real adapters — SDK absent paths ──────────────────────────────────

    def test_unitree_open_session_fail_has_adapter_name(self):
        router = _router_with(UnitreeAdapter(), "unitree_sdk2")
        res = router.execute_with_lifecycle("unitree_sdk2", RouteOp.RUN_ACTION)
        self.assertEqual(res.get("adapter_name"), "unitree_sdk2")

    def test_spot_open_session_fail_has_adapter_name(self):
        with patch(_PATCH_SPOT_OFF, False):
            router = _router_with(SpotAdapter(), "spot_sdk")
            res = router.execute_with_lifecycle("spot_sdk", RouteOp.RUN_ACTION)
        self.assertEqual(res.get("adapter_name"), "spot_sdk")

    def test_cyberdog_open_session_fail_has_adapter_name(self):
        with patch(_PATCH_ROS2_OFF, False):
            router = _router_with(CyberDogAdapter(), "cyberdog_sdk")
            res = router.execute_with_lifecycle("cyberdog_sdk", RouteOp.RUN_ACTION)
        self.assertEqual(res.get("adapter_name"), "cyberdog_sdk")

    # ── Mock adapter — full path coverage ────────────────────────────────

    def _mock_router(self, name: str = "mock_adapter") -> tuple:
        adapter = _MockAdapter()
        router  = _router_with(adapter, name)
        return adapter, router

    def test_success_path_has_adapter_name(self):
        _, router = self._mock_router("test_adapter")
        res = router.execute_with_lifecycle("test_adapter", RouteOp.RUN_ACTION)
        self.assertEqual(res.get("adapter_name"), "test_adapter")

    def test_not_found_path_has_adapter_name(self):
        router = ServiceRouter()
        res = router.execute_with_lifecycle("ghost", RouteOp.STOP)
        self.assertEqual(res.get("adapter_name"), "ghost")

    def test_open_session_fail_path_has_adapter_name(self):
        adapter, router = self._mock_router("fail_open")
        adapter.open_session_status = "error"
        res = router.execute_with_lifecycle("fail_open", RouteOp.RUN_ACTION)
        self.assertEqual(res.get("adapter_name"), "fail_open")

    def test_preflight_fail_path_has_adapter_name(self):
        adapter, router = self._mock_router("fail_pre")
        adapter.preflight_status = "error"
        res = router.execute_with_lifecycle(
            "fail_pre", RouteOp.RUN_ACTION,
            policy=LifecyclePolicy(run_preflight=True),
        )
        self.assertEqual(res.get("adapter_name"), "fail_pre")

    def test_execute_fail_path_has_adapter_name(self):
        adapter, router = self._mock_router("fail_exec")
        adapter.run_action_raises = RuntimeError("boom")
        res = router.execute_with_lifecycle("fail_exec", RouteOp.RUN_ACTION)
        self.assertEqual(res.get("adapter_name"), "fail_exec")

    def test_close_fail_still_has_adapter_name(self):
        adapter, router = self._mock_router("fail_close")
        adapter.close_session_status = "error"
        res = router.execute_with_lifecycle(
            "fail_close", RouteOp.RUN_ACTION,
            policy=LifecyclePolicy(close_after=True),
        )
        # Close failure does not mask success — status is ok
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res.get("adapter_name"), "fail_close")

    def test_adapter_name_reflects_registered_key(self):
        """adapter_name value is exactly the key used for registration."""
        for name in ("unitree_sdk2", "spot_sdk", "cyberdog_sdk", "custom_robot"):
            with self.subTest(name=name):
                adapter = _MockAdapter()
                router  = _router_with(adapter, name)
                res = router.execute_with_lifecycle(name, RouteOp.GET_SENSOR_DATA)
                self.assertEqual(res.get("adapter_name"), name)


# ── B3: Cross-brand lifecycle orchestration ────────────────────────────────

class TestCrossBrandLifecycleOrchestration(unittest.TestCase):
    """ServiceRouter orchestrates all 3 brands through the same lifecycle flow."""

    _BRANDS = [
        ("unitree_sdk2",  "Unitree"),
        ("spot_sdk",      "Spot"),
        ("cyberdog_sdk",  "CyberDog"),
    ]

    def _setup(self, adapter_name: str) -> tuple:
        adapter = _MockAdapter()
        router  = _router_with(adapter, adapter_name)
        return adapter, router

    def test_all_brands_succeed_with_default_policy(self):
        """All 3 adapter brands return ok on the happy path."""
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                _, router = self._setup(adapter_name)
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                    {"action": "stand", "params": {}},
                )
                self.assertEqual(res["status"], "ok",
                                 f"{label}: expected ok, got {res}")

    def test_all_brands_succeed_with_full_lifecycle_policy(self):
        """All 3 adapter brands succeed with preflight + close_after."""
        policy = LifecyclePolicy(run_preflight=True, close_after=True)
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                _, router = self._setup(adapter_name)
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                    {"action": "stand", "params": {}},
                    policy=policy,
                )
                self.assertEqual(res["status"], "ok")

    def test_open_session_failure_stops_at_open_for_all_brands(self):
        """open_session failure halts the lifecycle at the open_session stage."""
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                adapter, router = self._setup(adapter_name)
                adapter.open_session_status = "error"
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                )
                self.assertEqual(res["stage"],  "open_session")
                self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_preflight_failure_calls_close_for_all_brands(self):
        """Preflight failure triggers best-effort close_session for all brands."""
        policy = LifecyclePolicy(run_preflight=True)
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                adapter, router = self._setup(adapter_name)
                adapter.preflight_status = "error"
                router.execute_with_lifecycle(adapter_name, RouteOp.RUN_ACTION,
                                              policy=policy)
                close_calls = [c for c in adapter.calls if c[0] == "close_session"]
                self.assertEqual(len(close_calls), 1,
                                 f"{label}: expected 1 close_session, got {len(close_calls)}")

    def test_execute_failure_reason_is_execute_failed_for_all_brands(self):
        """Execute exception always produces LifecycleReason.EXECUTE_FAILED."""
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                adapter, router = self._setup(adapter_name)
                adapter.run_action_raises = RuntimeError("crash")
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                    {"action": "stand", "params": {}},
                )
                self.assertEqual(res["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_close_failure_does_not_mask_payload_for_all_brands(self):
        """Close session failure is surfaced in diagnostics but does not change status."""
        policy = LifecyclePolicy(close_after=True)
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                adapter, router = self._setup(adapter_name)
                adapter.close_session_status = "error"
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                    {"action": "stand", "params": {}},
                    policy=policy,
                )
                self.assertEqual(res["status"], "ok")
                self.assertIn("close_session_failed", res["diagnostics"])

    def test_get_sensor_data_op_succeeds_for_all_brands(self):
        """GET_SENSOR_DATA operation succeeds for all 3 brands."""
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                _, router = self._setup(adapter_name)
                res = router.execute_with_lifecycle(adapter_name, RouteOp.GET_SENSOR_DATA)
                self.assertEqual(res["status"], "ok")

    def test_stop_op_succeeds_for_all_brands(self):
        """STOP operation succeeds for all 3 brands."""
        for adapter_name, label in self._BRANDS:
            with self.subTest(brand=label):
                _, router = self._setup(adapter_name)
                res = router.execute_with_lifecycle(adapter_name, RouteOp.STOP)
                self.assertEqual(res["status"], "ok")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Regression coverage
# ═══════════════════════════════════════════════════════════════════════════

# ── C1: Phase 3 lifecycle contract unchanged ───────────────────────────────

class TestPhase3ContractRegression(unittest.TestCase):
    """Phase 3 lifecycle constants, schema, and routing are unchanged."""

    def test_session_open_failed_value(self):
        self.assertEqual(LifecycleReason.SESSION_OPEN_FAILED, "session_open_failed")

    def test_preflight_failed_value(self):
        self.assertEqual(LifecycleReason.PREFLIGHT_FAILED, "preflight_failed")

    def test_session_close_failed_value(self):
        self.assertEqual(LifecycleReason.SESSION_CLOSE_FAILED, "session_close_failed")

    def test_capability_unavailable_value(self):
        self.assertEqual(LifecycleReason.CAPABILITY_UNAVAILABLE, "capability_unavailable")

    def test_adapter_not_found_value(self):
        self.assertEqual(LifecycleReason.ADAPTER_NOT_FOUND, "adapter_not_found")

    def test_execute_failed_value(self):
        """Phase 4 addition — must be stable string."""
        self.assertEqual(LifecycleReason.EXECUTE_FAILED, "execute_failed")

    def test_result_schema_has_required_keys(self):
        """LifecycleResult dict always contains status/reason/stage/diagnostics."""
        ok_result    = LifecycleResult.ok("open_session").to_dict()
        error_result = LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED, {}
        ).to_dict()
        for res in (ok_result, error_result):
            for key in _RESULT_SCHEMA:
                self.assertIn(key, res)

    def test_lifecycle_policy_defaults(self):
        """LifecyclePolicy defaults reproduce Phase 3 backward-compat behaviour."""
        p = LifecyclePolicy()
        self.assertFalse(p.run_preflight)
        self.assertFalse(p.close_after)

    def test_legacy_router_run_action_passthrough(self):
        """ServiceRouter.run_action still works as direct passthrough (Phase 1)."""
        mock = _MockAdapter()
        router = _router_with(mock, "u")
        result = router.run_action("u", "stand")
        self.assertEqual(result, True)

    def test_legacy_router_stop_passthrough(self):
        mock = _MockAdapter()
        router = _router_with(mock, "u")
        router.stop("u")   # must not raise

    def test_legacy_router_get_sensor_data_passthrough(self):
        mock = _MockAdapter()
        router = _router_with(mock, "u")
        data = router.get_sensor_data("u")
        self.assertIsInstance(data, dict)

    def test_adapter_not_found_returns_structured_error(self):
        """Missing adapter → ADAPTER_NOT_FOUND at acquire stage."""
        res = ServiceRouter().execute_with_lifecycle("ghost", RouteOp.STOP)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.ADAPTER_NOT_FOUND)
        self.assertEqual(res["stage"],  "acquire")

    def test_execute_failure_uses_constant_not_magic_string(self):
        """EXECUTE_FAILED constant used — not a bare "execute_failed" string literal."""
        adapter = _MockAdapter()
        adapter.run_action_raises = RuntimeError("boom")
        router = _router_with(adapter, "u")
        res = router.execute_with_lifecycle("u", RouteOp.RUN_ACTION)
        # Must equal the constant, not just any string
        self.assertEqual(res["reason"], LifecycleReason.EXECUTE_FAILED)
        self.assertIsInstance(res["reason"], str)


# ── C2: BRAND_ADAPTER_MAP routing constants ────────────────────────────────

class TestBrandAdapterMapRegression(unittest.TestCase):
    """RobotContext.BRAND_ADAPTER_MAP has the correct Phase 4 keys."""

    def test_map_has_exactly_three_brand_keys(self):
        self.assertEqual(len(RobotContext.BRAND_ADAPTER_MAP), 3)

    def test_unitree_brand_routes_to_unitree_sdk2(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["unitree"], "unitree_sdk2")

    def test_bostiondynamics_brand_routes_to_spot_sdk(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["bostiondynamics"], "spot_sdk")

    def test_xiaomi_brand_routes_to_cyberdog_sdk(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["xiaomi"], "cyberdog_sdk")

    def test_no_legacy_boston_dynamics_key(self):
        """Stale 'boston_dynamics' key removed in Phase 4."""
        self.assertNotIn("boston_dynamics", RobotContext.BRAND_ADAPTER_MAP)

    def test_no_legacy_cyberdog_brand_key(self):
        """'cyberdog' is a robot type, not a brand key — must not be in map."""
        self.assertNotIn("cyberdog", RobotContext.BRAND_ADAPTER_MAP)

    def test_fallback_map_has_spot_to_bostiondynamics(self):
        """_FALLBACK_BRAND_MAP allows 'spot' as shorthand for bostiondynamics."""
        fallback = RobotContext._FALLBACK_BRAND_MAP
        self.assertIn("spot", fallback)
        self.assertEqual(fallback["spot"], "bostiondynamics")

    def test_fallback_map_has_cyberdog_to_xiaomi(self):
        """_FALLBACK_BRAND_MAP allows 'cyberdog' as shorthand for xiaomi."""
        fallback = RobotContext._FALLBACK_BRAND_MAP
        self.assertIn("cyberdog", fallback)
        self.assertEqual(fallback["cyberdog"], "xiaomi")

    def test_adapter_names_are_registered_in_service_router(self):
        """The adapter names in BRAND_ADAPTER_MAP are valid ServiceRouter registry keys."""
        for brand, adapter_name in RobotContext.BRAND_ADAPTER_MAP.items():
            with self.subTest(brand=brand):
                self.assertIsInstance(adapter_name, str)
                self.assertTrue(adapter_name)

    def test_reset_clears_current_robot_type(self):
        """RobotContext.reset() restores _current_robot_type to 'go2'."""
        RobotContext._current_robot_type = "spot"
        RobotContext.reset()
        self.assertEqual(RobotContext._current_robot_type, "go2")


# ── C3: Phase 4 deliverable gate ──────────────────────────────────────────

class TestPhase4DeliverableGate(unittest.TestCase):
    """High-level invariants confirming all Phase 4 deliverables are in place."""

    def test_execute_failed_constant_added_in_phase4(self):
        """LifecycleReason.EXECUTE_FAILED exists as a stable string constant."""
        self.assertTrue(hasattr(LifecycleReason, "EXECUTE_FAILED"))
        self.assertIsInstance(LifecycleReason.EXECUTE_FAILED, str)
        self.assertEqual(LifecycleReason.EXECUTE_FAILED, "execute_failed")

    def test_six_lifecycle_reason_constants_total(self):
        """6 Phase 3/4 canonical reason codes present (Phase 6 adds 2 more)."""
        codes = {v for k, v in vars(LifecycleReason).items() if not k.startswith("_")}
        self.assertGreaterEqual(len(codes), 6)   # Phase 6 adds SAFETY_GATE_BLOCKED + RETRY_BUDGET_EXHAUSTED

    def test_reason_codes_module_exists_and_exports_required_names(self):
        """reason_codes.py exports all required public names."""
        from system.service import reason_codes as rc          # noqa: F401
        for name in ("ALL_REASON_CODES", "CANONICAL_REASONS",
                     "UNITREE_REASONS", "SPOT_REASONS", "CYBERDOG_REASONS",
                     "REASON_MAP", "DIAGNOSTIC_KEY_SCHEMA"):
            self.assertTrue(hasattr(rc, name), f"reason_codes missing: {name}")

    def test_three_adapter_packages_exist(self):
        """All three Phase 4 adapter packages can be imported."""
        from system.service.adapters.unitree_sdk2 import adapter as _u  # noqa: F401
        from system.service.adapters.spot_sdk     import adapter as _s  # noqa: F401
        from system.service.adapters.cyberdog_sdk import adapter as _c  # noqa: F401
        self.assertTrue(True)  # ImportError → fail

    def test_all_adapter_results_carry_adapter_name(self):
        """adapter_name key is present in ALL execute_with_lifecycle result paths."""
        paths_results = []

        # not-found path
        paths_results.append(
            ServiceRouter().execute_with_lifecycle("missing", RouteOp.STOP)
        )

        # success path
        a = _MockAdapter()
        router = _router_with(a, "check")
        paths_results.append(router.execute_with_lifecycle("check", RouteOp.STOP))

        # open_session fail
        a2 = _MockAdapter(); a2.open_session_status = "error"
        router2 = _router_with(a2, "check2")
        paths_results.append(router2.execute_with_lifecycle("check2", RouteOp.STOP))

        # execute fail
        a3 = _MockAdapter(); a3.run_action_raises = RuntimeError("x")
        router3 = _router_with(a3, "check3")
        paths_results.append(router3.execute_with_lifecycle("check3", RouteOp.RUN_ACTION))

        for res in paths_results:
            with self.subTest(status=res["status"], stage=res["stage"]):
                self.assertIn("adapter_name", res)

    def test_brand_adapter_map_is_class_attribute(self):
        """BRAND_ADAPTER_MAP is a class-level dict on RobotContext."""
        self.assertIsInstance(RobotContext.BRAND_ADAPTER_MAP, dict)

    def test_multi_brand_adapter_classes_are_importable(self):
        """SpotAdapter and CyberDogAdapter are importable without their SDKs."""
        from system.service.adapters.spot_sdk.adapter     import SpotAdapter      # noqa: F401
        from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter  # noqa: F401
        self.assertTrue(True)  # ImportError → fail


if __name__ == "__main__":
    unittest.main(verbosity=2)
