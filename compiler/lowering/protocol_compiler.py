"""Protocol payload compiler — Mission-output → motor-weight protocol envelope.

Circle 4: Minimum compilation rules for mapping script/node outputs to the
motor-weight protocol payload expected by BehaviorSubgraphInvoker.

Public API
----------
ProtocolTargetSpec       — one motor param write intent (before envelope).
ProtocolPayloadCompiler  — builds the full, validate-ready protocol payload dict.

Signal output conventions  (Circle 4 Step 3)
--------------------------------------------
SIGNAL_ENABLE  ("event_enable")  → param_key="stiffness", value=1.0
SIGNAL_DISABLE ("event_disable") → param_key="stiffness", value=0.0
SIGNAL_SWITCH  ("event_switch")  → param_key="mode",      value=<caller-supplied>

Use ``for_signal(signal, motor_key, ...)`` for the signal convention path.
Use ``from_sensor_imu(sensor_data, motor_key, ...)`` for the sensor→protocol path.
Use ``build_payload(targets, ...)`` directly for full control.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Envelope constants (mirrors motor_weight_protocol._PROTOCOL_VERSION
# without creating an import cycle through the compiler → system path).
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: str = "1.0"
DEFAULT_MAX_AGE_MS: int = 1_000
DEFAULT_MODE: str = "constant"

# ---------------------------------------------------------------------------
# Signal output conventions  (Circle 4 Step 3)
# ---------------------------------------------------------------------------

SIGNAL_ENABLE  = "event_enable"
SIGNAL_DISABLE = "event_disable"
SIGNAL_SWITCH  = "event_switch"

# Canonical (param_key, fixed_value) per signal.
# fixed_value=None means the caller must supply switch_value.
_SIGNAL_PARAM: Dict[str, tuple] = {
    SIGNAL_ENABLE:  ("stiffness", 1.0),
    SIGNAL_DISABLE: ("stiffness", 0.0),
    SIGNAL_SWITCH:  ("mode",      None),
}

VALID_SIGNALS = frozenset(_SIGNAL_PARAM)


# ---------------------------------------------------------------------------
# ProtocolTargetSpec
# ---------------------------------------------------------------------------

@dataclass
class ProtocolTargetSpec:
    """One motor-param write intent, prior to protocol envelope construction.

    Fields
    ------
    motor_key  : Logical motor identifier (e.g. "leg_fl_hip").
    param_key  : Controlled parameter name (e.g. "gain", "stiffness", "mode").
    value      : Constant numeric value (scalar or array).
    address    : Runtime route key (int index or symbolic string).
    dtype      : "float32" | "float64".
    shape      : Dimension list; scalar = [1].
    limits     : Optional clamping bounds {"min": float, "max": float, "clamp": bool}.
    source_key : When set, generates kind="external" source with this external key.
    note       : Optional human-readable annotation carried to the payload.
    """

    motor_key:  str
    param_key:  str
    value:      Union[float, List[float]]
    address:    Union[int, str] = 0
    dtype:      str = "float32"
    shape:      List[int] = field(default_factory=lambda: [1])
    limits:     Optional[Dict[str, float]] = None
    source_key: Optional[str] = None   # None → kind="constant"; str → kind="external"
    note:       str = ""

    def to_target_dict(self) -> Dict[str, Any]:
        """Render as the per-target dict expected by validate_protocol_payload()."""
        if self.source_key is not None:
            src: Dict[str, Any] = {
                "kind":           "external",
                "external_key":   self.source_key,
                "external_value": self.value,
            }
        else:
            src = {"kind": "constant", "constant": self.value}
        d: Dict[str, Any] = {
            "motor_key": self.motor_key,
            "param_key": self.param_key,
            "address":   self.address,
            "dtype":     self.dtype,
            "shape":     self.shape,
            "source":    src,
        }
        if self.limits:
            d["limits"] = dict(self.limits)
        if self.note:
            d["note"] = self.note
        return d


# ---------------------------------------------------------------------------
# ProtocolPayloadCompiler
# ---------------------------------------------------------------------------

class ProtocolPayloadCompiler:
    """Compile structured target specs into a valid motor-weight protocol payload.

    Handles all envelope fields (protocol_version, trace_id, timestamp_ms,
    mode, max_age_ms) so callers only need to specify *what* to write.

    Instance method
    ---------------
    build_payload(targets, mode, max_age_ms, trace_id, timestamp_ms)
        → complete protocol payload dict

    Class-level convenience methods (all return a payload dict)
    -----------------------------------------------------------
    from_sensor_imu(sensor_data, motor_key, ...)
        Build from IMU / gyro sensor_data dict; uses kind="constant" source
        with the numeric value extracted from sensor_data[sensor_field].
    for_signal(signal, motor_key, ...)
        Build from a canonical signal name (SIGNAL_ENABLE/DISABLE/SWITCH).
    """

    def build_payload(
        self,
        targets: List[ProtocolTargetSpec],
        mode: str = DEFAULT_MODE,
        max_age_ms: int = DEFAULT_MAX_AGE_MS,
        trace_id: str = "",
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a complete, validate-ready protocol payload dict.

        Parameters
        ----------
        targets      : One or more ProtocolTargetSpec instances.
        mode         : Top-level mode; "constant" or "external".
        max_age_ms   : Payload freshness window in milliseconds (≥ 1).
        trace_id     : Caller-supplied trace id; auto-UUID when empty.
        timestamp_ms : Epoch-millisecond timestamp; defaults to now.

        Returns
        -------
        Dict suitable for ``validate_protocol_payload()``.  Never raises.
        """
        return {
            "protocol_version": PROTOCOL_VERSION,
            "trace_id":         trace_id or str(uuid.uuid4()),
            "timestamp_ms":     timestamp_ms if timestamp_ms is not None
                                else int(time.time() * 1000),
            "mode":             mode,
            "max_age_ms":       max_age_ms,
            "targets":          [t.to_target_dict() for t in (targets or [])],
        }

    @classmethod
    def from_sensor_imu(
        cls,
        sensor_data: Dict[str, Any],
        motor_key: str,
        param_key: str = "gain",
        sensor_field: str = "accel_z",
        address: Union[int, str] = 0,
        limits: Optional[Dict[str, float]] = None,
        max_age_ms: int = DEFAULT_MAX_AGE_MS,
        trace_id: str = "",
    ) -> Dict[str, Any]:
        """Build a constant-mode payload from an IMU/gyro sensor_data dict.

        Extracts ``sensor_data[sensor_field]`` as the motor param value.
        Non-numeric or missing fields fall back to 0.0 so downstream
        validation produces a clean diagnostic rather than raising here.

        Parameters
        ----------
        sensor_data  : Output dict from SensorInputNode (e.g. {"accel_z": 0.12}).
        motor_key    : Target motor logical id.
        param_key    : Parameter to write.  Default "gain".
        sensor_field : Key in sensor_data to read.  Default "accel_z".
        address      : Motor address / route key.
        limits       : Optional clamping bounds dict.
        max_age_ms   : Freshness window in ms.
        trace_id     : Correlation trace id.
        """
        raw = (sensor_data or {}).get(sensor_field, 0.0)
        value = float(raw) if isinstance(raw, (int, float)) else 0.0
        spec = ProtocolTargetSpec(
            motor_key=motor_key,
            param_key=param_key,
            value=value,
            address=address,
            limits=limits,
        )
        return cls().build_payload([spec], max_age_ms=max_age_ms, trace_id=trace_id)

    @classmethod
    def for_signal(
        cls,
        signal: str,
        motor_key: str,
        address: Union[int, str] = 0,
        switch_value: Optional[float] = None,
        max_age_ms: int = DEFAULT_MAX_AGE_MS,
        trace_id: str = "",
    ) -> Dict[str, Any]:
        """Build a protocol payload from a canonical signal name.

        Signal conventions (Circle 4 Step 3)
        -------------------------------------
        SIGNAL_ENABLE  ("event_enable")   → stiffness=1.0
        SIGNAL_DISABLE ("event_disable")  → stiffness=0.0
        SIGNAL_SWITCH  ("event_switch")   → mode=<switch_value>

        Raises
        ------
        ValueError  if signal not in VALID_SIGNALS.
        ValueError  if signal==SIGNAL_SWITCH and switch_value is None.
        """
        if signal not in _SIGNAL_PARAM:
            raise ValueError(
                f"Unknown signal {signal!r}; valid signals: {sorted(VALID_SIGNALS)}"
            )
        param_key, fixed_value = _SIGNAL_PARAM[signal]
        if fixed_value is None:
            if switch_value is None:
                raise ValueError(
                    "switch_value is required for SIGNAL_SWITCH ('event_switch')"
                )
            value: float = float(switch_value)
        else:
            value = fixed_value
        spec = ProtocolTargetSpec(
            motor_key=motor_key,
            param_key=param_key,
            value=value,
            address=address,
        )
        return cls().build_payload([spec], max_age_ms=max_age_ms, trace_id=trace_id)
