# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""USD body / joint name dump — runs inside the Isaac Lab venv.

Used by the Robot Asset card's "Dump USD" button to harvest the actual
body and joint names from a USD asset (local path or Nucleus URL) without
booting the full Isaac Sim kit. The result is a small JSON file the main
app reads back to populate ``bodies_per_format.USD`` /
``joints_per_format.USD`` in the user-overlay registry.

Usage::

    python discover_usd_bodies.py <usd_path_or_nucleus_url> <out_json_path>

Output JSON shape::

    {
        "bodies": ["body", "fl_hip", "fl_uleg", ...],     # ordered by USD prim traversal
        "joints": ["fl_hx", "fl_hy", "fl_kn", ...],
        "body_inertials": {                                # keyed by USD body name
            "body": {
                "mass": 15.234,                            # SI kg, or null if unauthored
                "mass_source": "authored",                 # authored | absent
                "com": [0.001, 0.0, -0.05],                # SI m, prim-local, or null
                "diagonal_inertia": [0.12, 1.30, 1.33],    # SI kg·m², or null if unauthored
                "principal_axes": [1.0, 0.0, 0.0, 0.0],    # quat (w,x,y,z), or null
                "inertia_source": "authored"               # authored | absent
            },
            ...
        },
        "joint_frames": {                                  # keyed by USD joint name
            "fl_hy": {body0, body1, local_pos0/1, local_rot0/1, axis}
        },
        "joint_drives": {                                  # keyed by USD joint name
            "fl_hy": {
                "kind": "revolute",                        # revolute | prismatic
                "drive": {
                    "type": "angular",                     # angular | linear (DriveAPI token)
                    "target_type": "force",                # force | acceleration | null
                    "stiffness": 60.0,                     # SI: Nm/rad (rev) / N/m (prism), or null
                    "damping": 1.5,                        # SI: Nm·s/rad / N·s/m, or null
                    "max_force": 80.0,                     # SI: Nm (rev) / N (prism), or null
                },
                "limits": {                                # SI: rad (rev) / m (prism), or null
                    "lower": -0.898, "upper": 2.295,
                },
                "max_velocity": 20.0,                      # SI: rad/s (rev) / m/s (prism), or null
                "provenance": {
                    "attribute_present": {                 # True = authored in USD; False = absent
                        "drive_stiffness": true, "drive_damping": true,
                        "drive_max_force": true, "drive_target_type": true,
                        "limit_lower": true, "limit_upper": true,
                        "max_velocity": false,
                    }
                }
            }
        }
    }

The ``body_inertials`` block is harvested from ``UsdPhysics.MassAPI`` and
SI-normalised via the stage's ``metersPerUnit`` / ``kilogramsPerUnit``. A field
that is not authored in the USD is recorded as ``null`` with its ``*_source`` set
to ``"absent"`` — never silently zeroed. The downstream calibration layer
(``usd_mjcf_inertia_calibration``) fails loud on ``inertia_source == "absent"``
rather than producing a half-correct overlay.

The ``joint_drives`` block is harvested from ``UsdPhysics.DriveAPI`` (token
``angular`` for revolute, ``linear`` for prismatic), the joint's
``RevoluteJoint``/``PrismaticJoint`` schema (position limits), and
``PhysxSchema.PhysxJointAPI`` (``maxJointVelocity``). Same fail-loud convention:
absent attributes are ``null`` with ``provenance.attribute_present[<name>] =
false`` so consumers can distinguish "USD did not author this" from "USD authored
zero". Angular units are converted to radians at dump time (USD authors revolute
limits and ``maxJointVelocity`` in degrees / deg·s⁻¹ by convention); linear units
are SI-normalised via ``metersPerUnit``. This is what the PD/Actuator AUTO
button in the Robot Node feeds off (``pd_preview.resolve_recommended_actuators``).

Two parsing paths, tried in order:

1. **``pxr.Usd`` direct** (lightweight, ~3-5 s) — preferred. Works for
   local USDs and for Nucleus URLs when the Isaac Sim venv has the
   Omniverse connector configured. No kit boot.
2. **``omni.isaac.core.utils.stage.open_stage``** (heavyweight, ~30 s) —
   fallback when pxr alone can't open the asset (typical for Nucleus
   URLs without the connector). Boots a minimal kit.

Exit codes:
    0  — wrote out_json successfully
    1  — argv error / unwritable output
    2  — pxr import failed AND kit fallback failed
    3  — stage opened but no rigid bodies / joints found (corrupt asset)
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List

# Auto-accept Omniverse Kit EULA — required for SimulationApp bootstrap when
# the subprocess has no TTY. Mirrors il_train_launcher.py line 49. Belt-and-
# suspenders: the parent (discovery_subprocess.py) also injects this env;
# setting it here too means manual `python discover_usd_bodies.py ...`
# invocations work without a wrapper.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _isaac_venv_path_is_ascii_safe() -> bool:
    """Return False iff ``sys.executable`` (the Isaac Lab venv Python) sits
    under a path containing non-ASCII characters.

    Kit's plugin loader (``omni.usd``'s ``Plug_Load`` in plugin.cpp) opens
    DLLs via ``LoadLibraryA`` — the ANSI byte-path variant. On a Windows
    install whose system codepage is non-UTF-8 (CP936 for zh_CN, CP1252
    for de_DE/fr_FR, …), a UTF-8 path containing characters outside that
    codepage gets mis-translated and ``LoadLibraryA`` returns
    ``ERROR_MOD_NOT_FOUND``::

        Failed to load plugin 'Omniverse USD Plugin':
            找不到指定的模块。
         in 'a:/测试/.../omni.usd_resolver-.../bin/omni_usd_resolver.dll'

    Once that single DLL load fails, Kit silently runs without
    ``OmniUsdResolver`` and ``UsdUsdResolver``, so any ``Stage.Open`` call
    produces an empty stage. Traversal returns zero bodies/joints; the
    subprocess exits 0 (because Kit's teardown calls ``os._exit(0)``)
    leaving no real diagnostic for the caller.

    The user's options are documented in the sentinel emitted alongside
    this probe. The check is Windows-only because POSIX ``dlopen`` is
    locale-neutral; on non-Windows we always return True.
    """
    if os.name != "nt":
        return True
    try:
        sys.executable.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _resolve_nucleus_url(usd_ref: str) -> str:
    """Translate UnitPort's ``nucleus:Robots/...`` shorthand to a real
    Isaac cloud URL.

    Only resolvable from inside Kit (carb settings are populated by
    AppLauncher), so this helper assumes the kit is already booted —
    call it from ``_dump_via_kit`` after ``AppLauncher(...)``.
    """
    if not usd_ref.startswith("nucleus:"):
        return usd_ref
    suffix = usd_ref[len("nucleus:"):].lstrip("/")
    # Preferred: isaaclab's helper (== `${asset_root_cloud}/Isaac`).
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
        if ISAAC_NUCLEUS_DIR:
            return f"{ISAAC_NUCLEUS_DIR}/{suffix}"
    except Exception:
        pass
    # Fallback: carb directly (works without isaaclab installed).
    try:
        import carb  # type: ignore
        root = carb.settings.get_settings().get(
            "/persistent/isaac/asset_root/cloud"
        )
        if root:
            return f"{root}/Isaac/{suffix}"
    except Exception:
        pass
    # Last resort: pass through — kit will produce a clear error.
    return usd_ref


def _read_stage_units(stage) -> "tuple[float, float]":
    """Return ``(meters_per_unit, kilograms_per_unit)`` from stage metadata.

    UsdPhysics quantities are authored in stage units, so we must read these to
    normalise to SI (kg, m, kg·m²). Isaac stages are normally 1.0/1.0 but we do
    not assume. Non-positive / unreadable metadata falls back to 1.0.
    """
    from pxr import UsdGeom, UsdPhysics  # type: ignore
    try:
        mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        mpu = 1.0
    try:
        kpu = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    except Exception:
        kpu = 1.0
    if not (mpu > 0.0):
        mpu = 1.0
    if not (kpu > 0.0):
        kpu = 1.0
    return mpu, kpu


def _attr_is_authored(attr) -> bool:
    """Version-robust 'has an authored opinion' check for a USD attribute."""
    if attr is None:
        return False
    try:
        if not attr.IsValid():
            return False
    except Exception:
        pass
    try:
        return bool(attr.HasAuthoredValue())
    except Exception:
        try:
            return attr.Get() is not None
        except Exception:
            return False


def _read_body_inertial(prim, mpu: float, kpu: float) -> Dict[str, Any]:
    """Harvest UsdPhysics.MassAPI mass props for a rigid-body prim → SI dict.

    Returns ``{mass, mass_source, com, diagonal_inertia, principal_axes,
    inertia_source}``. SI units (kg, m, kg·m²). An unauthored field is recorded
    as ``None`` with its ``*_source`` = ``"absent"`` — never zeroed. USD's
    ``diagonalInertia == (0,0,0)`` is also "absent" (PhysX auto-computes at sim
    load); the calibration layer must fail loud on absent inertia rather than
    leave a silent training↔sim gap.
    """
    from pxr import UsdPhysics  # type: ignore
    mass_api = UsdPhysics.MassAPI(prim)

    mass_attr = mass_api.GetMassAttr()
    diag_attr = mass_api.GetDiagonalInertiaAttr()
    com_attr = mass_api.GetCenterOfMassAttr()
    axes_attr = mass_api.GetPrincipalAxesAttr()

    mass = None
    mass_source = "absent"
    if _attr_is_authored(mass_attr):
        v = mass_attr.Get()
        if v is not None and float(v) > 0.0:
            mass = float(v) * kpu
            mass_source = "authored"

    diag = None
    inertia_source = "absent"
    if _attr_is_authored(diag_attr):
        v = diag_attr.Get()
        if v is not None:
            d = [float(v[0]), float(v[1]), float(v[2])]
            if any(x > 0.0 for x in d):  # (0,0,0) = USD "let PhysX compute" sentinel
                scale = kpu * mpu * mpu
                diag = [x * scale for x in d]
                inertia_source = "authored"

    com = None
    if _attr_is_authored(com_attr):
        v = com_attr.Get()
        if v is not None:
            com = [float(v[0]) * mpu, float(v[1]) * mpu, float(v[2]) * mpu]

    axes = None
    if _attr_is_authored(axes_attr):
        q = axes_attr.Get()
        if q is not None:
            im = q.GetImaginary()
            axes = [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]

    return {
        "mass": mass,
        "mass_source": mass_source,
        "com": com,
        "diagonal_inertia": diag,
        "principal_axes": axes,
        "inertia_source": inertia_source,
    }


def _read_joint_frame(prim, mpu: float) -> Dict[str, Any]:
    """Harvest a joint's kinematic anchor for the USD↔MJCF frame-alignment check.

    Returns the raw bits needed to express the joint origin + axis in the
    PARENT-local frame (the geometry/rotation math is done downstream in the
    calibration, which has numpy): body0/body1 names, ``localPos0/1`` (SI m),
    ``localRot0/1`` (quat w,x,y,z, identity default), and the ``axis`` token.
    """
    from pxr import UsdPhysics  # type: ignore
    joint = UsdPhysics.Joint(prim)

    def _rel_name(rel) -> str:
        try:
            tgts = rel.GetTargets()
            if tgts:
                return str(tgts[0].name)
        except Exception:
            pass
        return ""

    def _vec3(attr, scale: float):
        try:
            v = attr.Get()
        except Exception:
            v = None
        if v is None:
            return None
        return [float(v[0]) * scale, float(v[1]) * scale, float(v[2]) * scale]

    def _quat(attr):
        try:
            q = attr.Get()
        except Exception:
            q = None
        if q is None:
            return [1.0, 0.0, 0.0, 0.0]
        im = q.GetImaginary()
        return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]

    axis_token = ""
    for cls in (UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint):
        try:
            a = cls(prim).GetAxisAttr().Get()
            if a:
                axis_token = str(a)
                break
        except Exception:
            pass

    return {
        "body0": _rel_name(joint.GetBody0Rel()),
        "body1": _rel_name(joint.GetBody1Rel()),
        "local_pos0": _vec3(joint.GetLocalPos0Attr(), mpu),
        "local_rot0": _quat(joint.GetLocalRot0Attr()),
        "local_pos1": _vec3(joint.GetLocalPos1Attr(), mpu),
        "local_rot1": _quat(joint.GetLocalRot1Attr()),
        "axis": axis_token,
    }


def _read_joint_drive(prim, mpu: float) -> Dict[str, Any]:
    """Harvest a joint's drive params + position limits + max velocity cap.

    Output is SI:
      * ``drive.stiffness`` — N·m/rad (revolute) / N/m   (prismatic), or ``None``.
        IsaacLab DriveAPI carries mass-weighted gains in SI torque units
        (§10): for a revolute joint with ``ImplicitActuatorCfg.stiffness=K``,
        the runtime ``τ = K · (q_target - q) - D · q̇``. We store ``K`` verbatim
        so the downstream AUTO consumer can divide by ``m_eff`` to recover ``ωn``.
      * ``drive.damping`` — N·m·s/rad / N·s/m, or ``None``.
      * ``drive.max_force`` — N·m / N (per-joint torque ceiling), or ``None``.
        ``0.0`` (authored) means "PhysX no-limit" in USD convention — consumer
        treats that as ``None`` (no useful cap) but ``provenance`` keeps the
        authored bit so callers can distinguish "0 = explicitly no limit" from
        "absent = USD did not author".
      * ``drive.target_type`` — ``"force"`` | ``"acceleration"`` | ``None``.
      * ``limits.lower/upper`` — rad / m. USD authors revolute limits in
        DEGREES; we convert to radians here so consumers see SI throughout.
        ``None`` means unlimited (USD ``-inf`` / ``+inf`` or unauthored).
      * ``max_velocity`` — rad/s / m/s. From ``PhysxSchema.PhysxJointAPI``;
        revolute is degrees/sec in USD → rad/s here. ``None`` if PhysxSchema
        unavailable (vanilla pxr) or the attribute is unauthored.

    ``provenance.attribute_present`` records which fields were *authored* in
    the USD as opposed to absent. Same fail-loud contract as
    :func:`_read_body_inertial` — consumers must check this rather than guess
    from ``None`` (which collapses both "absent" and "explicitly null").
    """
    from pxr import UsdPhysics  # type: ignore

    is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
    is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
    if not (is_revolute or is_prismatic):
        return {}

    drive_token = "angular" if is_revolute else "linear"
    drive_payload: Dict[str, Any] = {
        "type": drive_token,
        "target_type": None,
        "stiffness": None,
        "damping": None,
        "max_force": None,
    }
    present: Dict[str, bool] = {
        "drive_stiffness": False,
        "drive_damping": False,
        "drive_max_force": False,
        "drive_target_type": False,
        "limit_lower": False,
        "limit_upper": False,
        "max_velocity": False,
    }

    # DriveAPI is applied per-token. ``Get(prim, "angular")`` returns a typed
    # DriveAPI whether or not the API has been *applied* — we then check each
    # attribute's authoring state individually.
    try:
        drive_api = UsdPhysics.DriveAPI.Get(prim, drive_token)
    except Exception:
        drive_api = None
    if drive_api:
        def _read_float(attr) -> "tuple[bool, float | None]":
            if not _attr_is_authored(attr):
                return (False, None)
            try:
                v = attr.Get()
            except Exception:
                return (False, None)
            if v is None:
                return (False, None)
            try:
                return (True, float(v))
            except (TypeError, ValueError):
                return (False, None)

        try:
            ok, v = _read_float(drive_api.GetStiffnessAttr())
            if ok:
                drive_payload["stiffness"] = v
                present["drive_stiffness"] = True
        except Exception:
            pass
        try:
            ok, v = _read_float(drive_api.GetDampingAttr())
            if ok:
                drive_payload["damping"] = v
                present["drive_damping"] = True
        except Exception:
            pass
        try:
            ok, v = _read_float(drive_api.GetMaxForceAttr())
            if ok:
                drive_payload["max_force"] = v
                present["drive_max_force"] = True
        except Exception:
            pass
        try:
            tt_attr = drive_api.GetTypeAttr()
            if _attr_is_authored(tt_attr):
                v = tt_attr.Get()
                if v is not None:
                    drive_payload["target_type"] = str(v)
                    present["drive_target_type"] = True
        except Exception:
            pass

    # Position limits. USD revolute limits are in DEGREES; convert to rad.
    # Prismatic limits are in stage units; convert via mpu.
    limits_payload: Dict[str, Any] = {"lower": None, "upper": None}
    typed_cls = UsdPhysics.RevoluteJoint if is_revolute else UsdPhysics.PrismaticJoint
    try:
        typed = typed_cls(prim)
        for key, getter in (
            ("lower", typed.GetLowerLimitAttr),
            ("upper", typed.GetUpperLimitAttr),
        ):
            try:
                a = getter()
                if not _attr_is_authored(a):
                    continue
                v = a.Get()
                if v is None:
                    continue
                fv = float(v)
                # USD encodes "no limit" as +/-inf — treat as None (unlimited)
                # but still record as authored. Otherwise convert units.
                if math.isinf(fv):
                    limits_payload[key] = None
                else:
                    limits_payload[key] = (
                        math.radians(fv) if is_revolute else fv * mpu
                    )
                present[f"limit_{key}"] = True
            except Exception:
                pass
    except Exception:
        pass

    # PhysxSchema.maxJointVelocity. Authored in degrees/sec for revolute by
    # USD convention; in stage-units/sec for prismatic. PhysxSchema only ships
    # with the full Omniverse stack — the vanilla pxr (used by _dump_via_pxr
    # on machines where standalone usd-core is enough) typically lacks it.
    # Absent module = absent attribute, with ``present.max_velocity = False``.
    max_velocity: "float | None" = None
    try:
        from pxr import PhysxSchema  # type: ignore

        physx_api = PhysxSchema.PhysxJointAPI.Get(
            prim.GetStage(), prim.GetPath()
        )
        if physx_api:
            mv_attr = physx_api.GetMaxJointVelocityAttr()
            if _attr_is_authored(mv_attr):
                v = mv_attr.Get()
                if v is not None:
                    fv = float(v)
                    if math.isinf(fv):
                        max_velocity = None
                    elif is_revolute:
                        max_velocity = math.radians(fv)
                    else:
                        max_velocity = fv * mpu
                    present["max_velocity"] = True
    except ImportError:
        pass
    except Exception:
        pass

    return {
        "kind": "revolute" if is_revolute else "prismatic",
        "drive": drive_payload,
        "limits": limits_payload,
        "max_velocity": max_velocity,
        "provenance": {"attribute_present": present},
    }


def _dump_via_pxr(usd_ref: str) -> Dict[str, Any]:
    """Direct pxr.Usd traversal (no kit). Returns the full discovery payload.

    Payload keys: ``bodies``, ``joints``, ``body_inertials``, ``joint_frames``,
    ``joint_drives``. See the module docstring for the per-joint shape of
    ``joint_drives`` (DriveAPI / position limits / maxJointVelocity).
    """
    from pxr import Usd, UsdPhysics  # type: ignore

    stage = Usd.Stage.Open(usd_ref)
    if stage is None:
        raise RuntimeError(f"pxr.Usd.Stage.Open returned None for {usd_ref!r}")

    mpu, kpu = _read_stage_units(stage)
    bodies: List[str] = []
    joints: List[str] = []
    body_inertials: Dict[str, Any] = {}
    joint_frames: Dict[str, Any] = {}
    joint_drives: Dict[str, Any] = {}
    seen_b: set = set()
    seen_j: set = set()
    skipped_non_dof = 0

    for prim in stage.Traverse():
        name = str(prim.GetName() or "")
        if not name:
            continue
        # Rigid bodies: tagged with UsdPhysics.RigidBodyAPI
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if name not in seen_b:
                    bodies.append(name)
                    seen_b.add(name)
                    try:
                        body_inertials[name] = _read_body_inertial(prim, mpu, kpu)
                    except Exception as exc:
                        print(
                            f"[discover_usd_bodies] inertial read failed for "
                            f"{name!r}: {exc!r}",
                            file=sys.stderr, flush=True,
                        )
        except Exception:
            pass
        # Joints: ONLY actuated articulation DOFs — RevoluteJoint or
        # PrismaticJoint. Isaac Lab's articulation API exposes exactly
        # those as controllable joints; FixedJoint (rigid attachment),
        # SphericalJoint (3-DOF ball, decomposed elsewhere), and the
        # abstract UsdPhysics.Joint base class contribute zero DOFs and
        # MUST be excluded — otherwise the registry gets polluted with
        # phantom joint names that policy code later cannot resolve.
        # BD Spot's USD attaches feet via FixedJoint named `*_ank`,
        # Go2's USD attaches cosmetic shells / sensor mounts / feet via
        # FixedJoint — all of these previously leaked through.
        try:
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                if name not in seen_j:
                    joints.append(name)
                    seen_j.add(name)
                    try:
                        joint_frames[name] = _read_joint_frame(prim, mpu)
                    except Exception as exc:
                        print(
                            f"[discover_usd_bodies] joint-frame read failed for "
                            f"{name!r}: {exc!r}",
                            file=sys.stderr, flush=True,
                        )
                    try:
                        drive_block = _read_joint_drive(prim, mpu)
                        if drive_block:
                            joint_drives[name] = drive_block
                    except Exception as exc:
                        print(
                            f"[discover_usd_bodies] joint-drive read failed for "
                            f"{name!r}: {exc!r}",
                            file=sys.stderr, flush=True,
                        )
            elif prim.IsA(UsdPhysics.Joint):
                skipped_non_dof += 1
        except Exception:
            pass

    if skipped_non_dof:
        print(
            f"[discover_usd_bodies] skipped {skipped_non_dof} non-DOF joint(s) "
            f"(FixedJoint / SphericalJoint / abstract Joint) — these are not "
            f"actuated and are excluded from the articulation joint list.",
            file=sys.stderr, flush=True,
        )

    return {
        "bodies": bodies, "joints": joints,
        "body_inertials": body_inertials, "joint_frames": joint_frames,
        "joint_drives": joint_drives,
    }


def _dump_via_kit(usd_ref: str, out_path: str) -> Dict[str, Any]:
    """Fallback path: boot a minimal Isaac Sim kit to open the stage.

    Heavier (~30-120 s cold startup) but mandatory when pxr is shipped
    only inside the isaacsim package (typical Isaac Sim 5.x pip install)
    rather than as a standalone usd-core wheel. Also required for any
    Nucleus URL pxr can't resolve without the Omniverse connector.

    **Uses isaaclab's AppLauncher** rather than a raw ``SimulationApp``.
    Mirrors what ``il_train_launcher.py`` does for the training subprocess,
    so we inherit the same headless boot path that's known to work on the
    user's GPU/driver/Hydra stack. Raw ``SimulationApp({"headless": True})``
    is known to access-violation inside ``omni.usd.dll`` during Hydra
    engine creation on some Isaac Sim 5.x installs; AppLauncher's
    ``args_cli`` plumbing wires the kit experience config the kit kernel
    actually expects.

    Stage-open helpers prefer Isaac Sim 5.x's ``isaacsim.core.utils.stage``
    and fall back to the legacy 4.x ``omni.isaac.core.utils.stage`` path.
    """
    # Boot kit the same way training does. AppLauncher accepts either a
    # parsed argparse Namespace or kwargs; the kwargs form is fine for our
    # single-shot use case.
    from isaaclab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        # After Kit has booted the connectors / OmniUsdResolver, ``pxr``
        # is importable from this very Python process and
        # ``Usd.Stage.Open`` is the simplest *synchronous* way to read
        # any URL (local / http / nucleus / omniverse). Avoid
        # ``isaacsim.core.utils.stage.open_stage`` here — that helper
        # routes through the kit context which loads the stage
        # asynchronously and our subsequent ``Traverse`` runs before the
        # asset is hydrated, yielding an empty body/joint list.
        from pxr import Usd, UsdPhysics  # type: ignore

        # Resolve UnitPort's `nucleus:Robots/...` shorthand to the real
        # NVIDIA Isaac cloud URL. The `nucleus:` scheme is our own
        # marker stored in robots_canonical.json; the actual asset lives
        # under `ISAAC_NUCLEUS_DIR` (=
        # `<carb.../asset_root/cloud>/Isaac`). Without this expansion
        # OmniUsdResolver treats `nucleus:Robots/...` as a local
        # relative path and silently fails.
        resolved_ref = _resolve_nucleus_url(usd_ref)
        if resolved_ref != usd_ref:
            print(
                f"[discover_usd_bodies] resolved {usd_ref!r} -> {resolved_ref!r}",
                file=sys.stderr, flush=True,
            )

        stage = Usd.Stage.Open(resolved_ref)
        if stage is None:
            raise RuntimeError(
                f"Usd.Stage.Open returned None for {resolved_ref!r}"
            )

        mpu, kpu = _read_stage_units(stage)
        bodies: List[str] = []
        joints: List[str] = []
        body_inertials: Dict[str, Any] = {}
        joint_frames: Dict[str, Any] = {}
        joint_drives: Dict[str, Any] = {}
        seen_b: set = set()
        seen_j: set = set()
        skipped_non_dof = 0
        for prim in stage.Traverse():
            name = str(prim.GetName() or "")
            if not name:
                continue
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and name not in seen_b:
                    bodies.append(name)
                    seen_b.add(name)
                    try:
                        body_inertials[name] = _read_body_inertial(prim, mpu, kpu)
                    except Exception as exc:
                        print(
                            f"[discover_usd_bodies] inertial read failed for "
                            f"{name!r}: {exc!r}",
                            file=sys.stderr, flush=True,
                        )
            except Exception:
                pass
            # See _dump_via_pxr — only RevoluteJoint / PrismaticJoint are
            # real articulation DOFs. FixedJoints (foot attachments, sensor
            # mounts, cosmetic shells) and SphericalJoints contribute zero
            # DOFs and must never reach the registry. Kit's pxr ships
            # PhysxSchema, so the kit path always populates max_velocity
            # when it is authored (vs the pxr-only path which may lack it).
            try:
                if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                    if name not in seen_j:
                        joints.append(name)
                        seen_j.add(name)
                        try:
                            joint_frames[name] = _read_joint_frame(prim, mpu)
                        except Exception as exc:
                            print(
                                f"[discover_usd_bodies] joint-frame read failed for "
                                f"{name!r}: {exc!r}",
                                file=sys.stderr, flush=True,
                            )
                        try:
                            drive_block = _read_joint_drive(prim, mpu)
                            if drive_block:
                                joint_drives[name] = drive_block
                        except Exception as exc:
                            print(
                                f"[discover_usd_bodies] joint-drive read failed for "
                                f"{name!r}: {exc!r}",
                                file=sys.stderr, flush=True,
                            )
                elif prim.IsA(UsdPhysics.Joint):
                    skipped_non_dof += 1
            except Exception:
                pass
        result = {
            "bodies": bodies, "joints": joints,
            "body_inertials": body_inertials, "joint_frames": joint_frames,
            "joint_drives": joint_drives,
        }
        print(
            f"[discover_usd_bodies] kit traversal: "
            f"{len(bodies)} bodies, {len(joints)} joints "
            f"(+{skipped_non_dof} non-DOF joints skipped)",
            file=sys.stderr, flush=True,
        )
        # Write the output JSON BEFORE simulation_app.close() runs in
        # the finally block. Kit's shutdown path can terminate the
        # Python interpreter directly (no return to main()), so any
        # cleanup that must happen after a successful traversal needs
        # to run here, while kit is still alive.
        if bodies or joints:
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                print(
                    f"[discover_usd_bodies] wrote {out_path}: "
                    f"{len(bodies)} bodies, {len(joints)} joints",
                    file=sys.stderr, flush=True,
                )
            except OSError as exc:
                print(
                    f"[discover_usd_bodies] cannot write {out_path!r}: {exc}",
                    file=sys.stderr, flush=True,
                )
                raise
        return result
    finally:
        simulation_app.close()


def _write_sentinel(out_path: str, error: str) -> None:
    """Write a non-empty ``_error`` JSON to ``out_path`` so the parent
    can distinguish 'subprocess exited with code 0 but produced no
    traversal' from 'subprocess crashed mid-way'.

    The most subtle case this guards against: ``simulation_app.close()``
    on some Isaac Sim 5.x builds calls ``os._exit(0)`` to skip
    Hydra / COM cleanup. If that fires inside the ``finally`` block of
    ``_dump_via_kit`` BEFORE we have written a successful result, the
    subprocess exits with code 0 and an empty tempfile, and the parent
    sees ``json.loads("")`` raise ``Expecting value`` — with no way to
    tell whether the failure was a missing asset, a kit boot error, or
    a Nucleus auth error. By writing this sentinel *before* doing
    anything that can trigger ``os._exit``, the parent always reads a
    well-formed JSON whose ``_error`` field is the structured cause.

    On success the sentinel is overwritten with the real result; the
    parent never sees ``_error`` in that case.
    """
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"bodies": [], "joints": [], "_error": error},
                f,
                indent=2,
            )
    except OSError as exc:
        # Best-effort: if we cannot write the sentinel either, the parent
        # will still see an empty file but the stderr below carries the
        # information it needs. Do not raise — we are already on an
        # error path or about to enter a risky kit boot.
        print(
            f"[discover_usd_bodies] could not write sentinel "
            f"{out_path!r}: {exc}",
            file=sys.stderr, flush=True,
        )


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(
            "usage: discover_usd_bodies.py <usd_path_or_nucleus_url> <out_json_path>",
            file=sys.stderr,
        )
        return 1

    usd_ref = argv[1]
    out_path = argv[2]

    # Prime the output with a hard error sentinel up front. Every
    # success path below overwrites it; every failure path leaves it
    # in place so the parent has structured information even when
    # the kit teardown short-circuits ``sys.exit`` with ``os._exit(0)``.
    _write_sentinel(
        out_path,
        f"discover_usd_bodies.py did not reach a successful traversal "
        f"for {usd_ref!r}. Check the subprocess stderr for the underlying "
        f"pxr / kit error.",
    )

    # Hard fail when the Isaac Lab venv lives under a non-ASCII path.
    # Kit's plugin loader uses LoadLibraryA, which mis-translates UTF-8
    # path bytes through the system ANSI codepage on Chinese / Cyrillic /
    # accented Windows installs. Plugins (omni.usd_resolver, usd_usd,
    # carb_*) silently fail to load; ``Stage.Open`` then yields an empty
    # stage. We cannot fix this from Python — ``LoadLibraryA`` is hard-
    # coded in Kit's C++ — so the only path forward is to abort early
    # with a clear directive instead of wasting 30+ s on a Kit boot
    # that's structurally guaranteed to produce nothing useful.
    #
    # Directory junctions and ``subst`` drive letters DO NOT help: Kit
    # canonicalises module paths through ``GetFinalPathNameByHandle``
    # which always returns the underlying real path. Only physically
    # moving Isaac Lab to an ASCII directory works.
    if not _isaac_venv_path_is_ascii_safe():
        _write_sentinel(
            out_path,
            "Isaac Lab is installed at a path containing non-ASCII "
            "characters ({!r}). Kit's plugin loader uses Windows "
            "LoadLibraryA (ANSI byte path), which mis-translates the "
            "UTF-8 bytes through the system codepage and fails to load "
            "omni_usd_resolver.dll / usd_usd.dll. Without these plugins "
            "Kit cannot open any USD asset. Directory junctions and "
            "subst drive letters do NOT work — Kit canonicalises paths "
            "through GetFinalPathNameByHandle. Fix: move Isaac Lab to "
            "an ASCII-only path (e.g. C:/IsaacLab/) and re-register via "
            "Settings → Engines → Isaac Lab → Locate.".format(sys.executable)
        )
        print(
            f"[discover_usd_bodies] aborting: Isaac Lab Python at non-ASCII "
            f"path {sys.executable!r}; Kit's LoadLibraryA cannot resolve "
            f"its own plugin DLLs.",
            file=sys.stderr, flush=True,
        )
        return 4

    # Try the lightweight pxr-only path first (no kit boot, ~1s); falls
    # back to the kit path if standalone pxr isn't importable in this
    # interpreter (typical Isaac Sim 5.x pip install).
    kit_wrote_output = False
    result: Dict[str, List[str]]
    try:
        result = _dump_via_pxr(usd_ref)
    except Exception as pxr_exc:
        print(f"[discover_usd_bodies] pxr path failed: {pxr_exc!r}", file=sys.stderr, flush=True)
        print("[discover_usd_bodies] falling back to kit-based path...", file=sys.stderr, flush=True)
        # Refresh the sentinel to record that we are about to enter the
        # kit path — if it crashes, the parent reads exactly this
        # message instead of "subprocess produced no traversal".
        _write_sentinel(
            out_path,
            f"pxr path failed ({pxr_exc!r}); kit fallback was entered "
            f"but did not produce a traversal. Possible causes: "
            f"`isaaclab.app` not importable from the Isaac Lab venv; "
            f"AppLauncher boot failure; ``Usd.Stage.Open`` returned None "
            f"or raised inside Kit (Nucleus auth / wrong URL / missing "
            f"asset). See subprocess stderr for the original kit error.",
        )
        try:
            # _dump_via_kit writes the JSON itself before kit teardown
            # (kit's close() may terminate the interpreter without
            # returning), so we mark the output as already written.
            result = _dump_via_kit(usd_ref, out_path)
            kit_wrote_output = bool(result.get("bodies") or result.get("joints"))
        except Exception as kit_exc:
            print(
                f"[discover_usd_bodies] kit path also failed: {kit_exc!r}",
                file=sys.stderr, flush=True,
            )
            # Overwrite the sentinel with the actual kit exception so the
            # parent can show it to the user verbatim (kit_exc may be
            # ImportError, RuntimeError, OmniUsdResolverError, …).
            _write_sentinel(
                out_path,
                f"kit fallback raised: {kit_exc!r}",
            )
            return 2

    if not result.get("bodies") and not result.get("joints"):
        print(
            f"[discover_usd_bodies] stage opened but no rigid bodies or joints "
            f"found in {usd_ref!r}",
            file=sys.stderr, flush=True,
        )
        _write_sentinel(
            out_path,
            f"stage opened for {usd_ref!r} but reported zero rigid bodies "
            f"and zero RevoluteJoint/PrismaticJoint prims. Asset is likely "
            f"corrupt or uses a non-UsdPhysics schema this build does not "
            f"recognise.",
        )
        return 3

    if not kit_wrote_output:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except OSError as exc:
            print(
                f"[discover_usd_bodies] cannot write {out_path!r}: {exc}",
                file=sys.stderr, flush=True,
            )
            _write_sentinel(
                out_path,
                f"traversal succeeded but writing output file "
                f"{out_path!r} failed: {exc}",
            )
            return 1
        print(
            f"[discover_usd_bodies] wrote {out_path}: "
            f"{len(result['bodies'])} bodies, {len(result['joints'])} joints",
            file=sys.stderr, flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
