# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MjField — scene composer for the Mission runtime layer.

A field is everything that is NOT the robot: the ground, terrain features,
props, lighting, and off-body sensors. P1 ships only the simplest variant
— the *flat ground field* — which assumes the actor's MJCF already
includes a flat groundplane (true for every menagerie ``scene.xml``).

The reason ``MjField`` is a first-class concept even in this minimal form
is to lock in the API surface that P5 needs:

    - ``compose(actor) -> mj_model``  : returns a model where the actor
      and the field share one mj_model. P1 short-circuits to
      ``actor.mj_model`` because the menagerie scenes are
      pre-composed; P5 will replace this with mjSpec / xml stitching.
    - ``declare_offbody_sensors() -> List[str]``: names that
      ``MjSimEnv.configure_sensors()`` forwards to the existing
      ``application.service.runtime.simulation.sensor_manager.SensorManager``.
      P1 ships an empty list (the default flat field has no off-body sensors).
    - ``randomize(rand_cfg)`` : domain-randomization hook. P1 no-op.

P5 will introduce richer fields (heightfield, stairs, prop obstacles,
low-light) and a yaml schema for project-local / shared field assets.

Ported verbatim from DEMO/src/system/runtime/simulation/mujoco/mj_field.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MjField:
    """A scene specification consumed by :class:`MjSimEnv`.

    P1 has only one production constructor — :meth:`flat_ground` — which
    is the StartNode default. P5 introduces richer fields: stair_climb
    (geom-based stairs), prop_obstacle_field (random box obstacles),
    mixed_props_arena (every prop primitive), low_light (lighting tweak),
    and rough_terrain (procedural heightfield via mjSpec.add_hfield).

    The ``heightfield`` block (P5-T1) declares a procedural noise
    heightfield. Schema::

        heightfield:
          nrow: 32
          ncol: 32
          size: [3.0, 3.0, 0.15, 0.1]   # radius_x, radius_y, max_h, base
          pos:  [3.0, 0.0, 0.0]          # geom origin
          seed: 42                        # numpy seed
          amplitude: 0.5                  # max normalized height
    """

    field_id: str
    base_terrain: str = "flat"
    props: List[Dict[str, Any]] = field(default_factory=list)
    lighting: Dict[str, Any] = field(default_factory=dict)
    offbody_sensor_specs: List[Dict[str, Any]] = field(default_factory=list)
    randomization: Optional[Dict[str, Any]] = None
    heightfield: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def flat_ground(cls) -> "MjField":
        """The StartNode default. Pass-through composition over the
        actor's own scene.xml, which already provides flat ground +
        ambient lighting in every menagerie mirror.
        """
        return cls(field_id="flat_ground", base_terrain="flat")

    # ------------------------------------------------------------------
    # P1/P5 composition surface
    # ------------------------------------------------------------------

    # Terrains that compose() handles in P5. Anything outside this set
    # falls back to flat (with no extra props injected) so unknown
    # terrains do not crash a Mission run.
    _SUPPORTED_TERRAINS = frozenset({"flat", "stairs", "obstacles"})

    def compose(self, actor: Any) -> Any:
        """Return the mj_model that pairs ``actor`` with this field.

        Pass-through for the flat-ground field (the menagerie scene
        already provides flat ground + lighting). For richer fields we
        load the actor's MJCF as an :class:`mujoco.MjSpec`, inject
        each ``self.props`` entry as a static geom in the worldbody,
        optionally apply lighting tweaks (P5-T2), and recompile.

        Unsupported terrains degrade to mjSpec composition with no
        props — Mission runs MUST keep working even when a P5+ field
        type lands later than the loader.
        """
        # Fast path: nothing to compose. Use the actor's pre-built model.
        if (
            self.base_terrain == "flat"
            and not self.props
            and not self.lighting
            and not self.heightfield
        ):
            return actor.mj_model

        return self._compose_via_mjspec(actor, props=self.props)

    # ------------------------------------------------------------------
    # mjSpec composition path
    # ------------------------------------------------------------------

    def _compose_via_mjspec(self, actor: Any, props: List[Dict[str, Any]]) -> Any:
        """Build a fresh mj_model = actor's MJCF + this field's geoms.

        Failure mode: silently degrade to flat ground rather than break.
        """
        import logging

        log = logging.getLogger(__name__)

        try:
            import mujoco
        except ImportError:
            log.warning(
                "MjField.compose: mujoco not importable; falling back "
                "to actor's pre-built model"
            )
            return actor.mj_model

        actor_mjcf = getattr(actor, "mjcf_path", None)
        if actor_mjcf is None:
            log.warning(
                "MjField.compose: actor has no mjcf_path; falling back"
            )
            return actor.mj_model

        try:
            spec = mujoco.MjSpec.from_file(str(actor_mjcf))
        except Exception as exc:
            log.warning(
                "MjField.compose: MjSpec.from_file failed for %s: %s",
                actor_mjcf, exc,
            )
            return actor.mj_model

        worldbody = getattr(spec, "worldbody", None)
        if worldbody is None:
            log.warning("MjField.compose: spec has no worldbody")
            return actor.mj_model

        for idx, prop in enumerate(props or []):
            if not isinstance(prop, dict):
                continue
            self._add_prop_geom(mujoco, worldbody, prop, default_index=idx)

        if self.lighting:
            self._apply_lighting(mujoco, spec, self.lighting)

        if self.heightfield:
            self._add_heightfield(mujoco, spec, self.heightfield)

        try:
            return spec.compile()
        except Exception as exc:
            log.warning(
                "MjField.compose: spec.compile() failed for field "
                "'%s': %s — falling back to actor's pre-built model",
                self.field_id, exc,
            )
            return actor.mj_model

    @staticmethod
    def _add_heightfield(mujoco_mod: Any, spec: Any, hf_spec: Dict[str, Any]) -> None:
        """P5-T1: procedural noise heightfield + geom referencing it."""
        import logging
        log = logging.getLogger(__name__)
        try:
            import numpy as np
        except ImportError:
            log.warning("MjField._add_heightfield: numpy unavailable")
            return

        try:
            nrow = int(hf_spec.get("nrow", 32) or 32)
            ncol = int(hf_spec.get("ncol", 32) or 32)
            size = list(hf_spec.get("size") or [3.0, 3.0, 0.15, 0.1])
            pos  = list(hf_spec.get("pos")  or [3.0, 0.0, 0.0])
            seed = int(hf_spec.get("seed", 42) or 42)
            amplitude = float(hf_spec.get("amplitude", 0.5) or 0.5)
            name = str(hf_spec.get("name", "field_hf") or "field_hf")
            rgba = list(hf_spec.get("rgba", [0.45, 0.55, 0.35, 1.0]) or [0.45, 0.55, 0.35, 1.0])
        except Exception as exc:
            log.warning("MjField._add_heightfield: bad spec: %s", exc)
            return

        rng = np.random.default_rng(seed)
        heights = (rng.random((nrow, ncol)) * float(amplitude)).flatten().tolist()

        try:
            spec.add_hfield(
                name=name,
                size=[float(s) for s in size[:4]],
                nrow=nrow,
                ncol=ncol,
                userdata=heights,
            )
        except Exception as exc:
            log.warning(
                "MjField._add_heightfield: add_hfield failed: %s", exc
            )
            return

        try:
            spec.worldbody.add_geom(
                name=f"{name}_geom",
                type=mujoco_mod.mjtGeom.mjGEOM_HFIELD,
                hfieldname=name,
                pos=pos,
                rgba=rgba,
            )
        except Exception as exc:
            log.warning(
                "MjField._add_heightfield: add_geom (hfield) failed: %s", exc
            )

    @staticmethod
    def _apply_lighting(mujoco_mod: Any, spec: Any, lighting: Dict[str, Any]) -> None:
        """P5-T2: apply field-level lighting tweaks via mjSpec."""
        import logging
        log = logging.getLogger(__name__)

        try:
            visual = getattr(spec, "visual", None)
            headlight = getattr(visual, "headlight", None) if visual is not None else None
            if headlight is None:
                log.warning("MjField._apply_lighting: spec has no visual.headlight")
                return
            for key in ("ambient", "diffuse", "specular"):
                value = lighting.get(key)
                if value is None:
                    continue
                if not isinstance(value, (list, tuple)) or len(value) < 3:
                    log.warning(
                        "MjField._apply_lighting: lighting.%s must be a "
                        "3-element [r, g, b] list, got %r — ignored",
                        key, value,
                    )
                    continue
                try:
                    setattr(headlight, key, [float(value[0]), float(value[1]), float(value[2])])
                except Exception as exc:
                    log.warning(
                        "MjField._apply_lighting: setattr headlight.%s "
                        "failed: %s", key, exc,
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("MjField._apply_lighting: top-level failure: %s", exc)

    @staticmethod
    def _add_prop_geom(mujoco_mod: Any, worldbody: Any, prop: Dict[str, Any], default_index: int = 0) -> None:
        """Inject a single prop dict as a worldbody geom (P5-T3)."""
        ptype = str(prop.get("type", "box") or "box").lower()

        _GEOM_TYPE_MAP = {
            "box":      mujoco_mod.mjtGeom.mjGEOM_BOX,
            "sphere":   mujoco_mod.mjtGeom.mjGEOM_SPHERE,
            "cylinder": mujoco_mod.mjtGeom.mjGEOM_CYLINDER,
            "capsule":  mujoco_mod.mjtGeom.mjGEOM_CAPSULE,
            "plane":    mujoco_mod.mjtGeom.mjGEOM_PLANE,
        }
        if ptype not in _GEOM_TYPE_MAP:
            import logging
            logging.getLogger(__name__).warning(
                "MjField._add_prop_geom: unknown prop type '%s' "
                "(supported: %s); skipping",
                ptype, sorted(_GEOM_TYPE_MAP),
            )
            return

        _DEFAULT_SIZE = {
            "box":      [0.1, 0.1, 0.05],
            "sphere":   [0.1],
            "cylinder": [0.08, 0.15],
            "capsule":  [0.06, 0.15],
            "plane":    [1.0, 1.0, 0.1],
        }

        name = str(prop.get("name", f"field_prop_{default_index}") or f"field_prop_{default_index}")
        pos = list(prop.get("pos", [0.0, 0.0, 0.05]) or [0.0, 0.0, 0.05])
        size = list(prop.get("size") or _DEFAULT_SIZE[ptype])
        rgba = list(prop.get("rgba", [0.6, 0.6, 0.6, 1.0]) or [0.6, 0.6, 0.6, 1.0])

        while len(size) < 3:
            size.append(0.0)

        kwargs: Dict[str, Any] = {
            "name": name,
            "type": _GEOM_TYPE_MAP[ptype],
            "pos": pos,
            "size": size,
            "rgba": rgba,
        }
        quat = prop.get("quat")
        if quat is not None:
            kwargs["quat"] = list(quat)

        try:
            worldbody.add_geom(**kwargs)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "MjField._add_prop_geom: add_geom failed for prop "
                "'%s' (type=%s): %s", name, ptype, exc,
            )

    def declare_offbody_sensors(self) -> List[str]:
        """Return the list of off-body sensor names this field exposes."""
        return [str(spec.get("name", "")) for spec in self.offbody_sensor_specs if spec]

    def randomize(self, rand_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Apply domain randomization to the field. P1 no-op."""
        return None
