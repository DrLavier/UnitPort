"""application.service.connection — real-robot connection layer (Phase 1+).

Phase 1 ships type-only contracts: ``transport.base`` (Transport Protocol +
Capability/Role/Kind enums + result dataclasses + typed exceptions),
``profile.ConnectionProfile`` (15-field profile dataclass mirrored from DEMO),
and ``phases.ConnectionPhase`` (4 alpha + 11 beta phase IDs that real
connectors will emit through ``AppSignals.connection_phase_changed``).

No transport implementations or DDS bridge live here yet — those land in
Phase 2/3 once the UI + signal contracts are validated end-to-end against
the fake state machine in :mod:`application.service.ros2_connection_config`.
"""
