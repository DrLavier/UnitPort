#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Boundary Nodes
Start and End nodes that define workflow entry/exit points.

mission_design.yaml v1.4: StartNode is the SOLE Region 1 setup node.
There are no separate MjActorNode / MjFieldNode — actor selection is
the existing ``robot_brand`` / ``robot_type`` parameter (which doubles
as the SDK target identifier), and field selection is the new
``field_id`` parameter. The execute() method constructs an MjSimEnv
that downstream BehaviorNodes receive via canvas edges (with the
auto-injection backward-compat shim in node_executor).
"""

import logging
from typing import Dict, Any, List


from .base_node import BaseNode


log = logging.getLogger(__name__)


class StartNode(BaseNode):
    """Start node — workflow entry point.

    Three responsibilities, all on this single node:

    1. **Legacy side effect** (preserved): set
       ``RobotContext.robot_type`` from the ``robot_brand`` /
       ``robot_type`` parameters. Old canvases that only consume this
       side-effect must keep working.

    2. **Actor selection** (v1.4): the same ``robot_brand`` /
       ``robot_type`` parameters ARE the Mission actor selection.
       ``robot_type`` (e.g. ``"go2"``) is fed straight into
       :meth:`MjActor.from_menagerie` — there is no separate ActorNode.

    3. **Field selection + sim_env construction** (v1.4): the
       ``field_id`` parameter (e.g. ``"flat_ground"`` /
       ``"stair_climb"`` / a project-local id) is loaded via
       :func:`load_field` and combined with the actor into an
       :class:`MjSimEnv`. The sim_env is exposed on the ``sim_env``
       output port for downstream BehaviorNodes — there is no
       separate FieldNode.

    Output ports
    ------------
    - ``flow_out``  : preserved from legacy StartNode
    - ``sim_env``   : MjSimEnv instance for downstream BehaviorNode
    - ``actor``     : MjActor wired into the sim_env
    - ``field``     : MjField wired into the sim_env
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "start")
        # Actor selection — these double as the SDK target identifier
        # via RobotContext.set_robot_type().
        self.parameters['robot_brand'] = 'unitree'
        self.parameters['robot_type'] = 'go2'
        # Field selection — resolved through field_loader (project-local
        # first, shared fallback). Defaults to flat_ground.
        self.parameters['field_id'] = 'flat_ground'
        # Render the MuJoCo viewer by default — Mission canvases are
        # interactive, the user expects to SEE the simulation. Tests
        # and headless callers either set this to False or build the
        # sim_env outside the StartNode entirely.
        self.parameters['render_enabled'] = True
        self.parameters['max_episode_steps'] = 1000

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # ── Legacy side effect: keeps RobotContext / SDK target in sync
        from src.system.core.robot_context import RobotContext
        robot_type = str(self.parameters.get('robot_type', 'go2') or 'go2')
        RobotContext.set_robot_type(robot_type)

        # ── Build the Mission sim_env from actor (= robot_type) + field.
        # Wrapped in a try/except so a missing mujoco wheel or a broken
        # MJCF / field cannot regress canvases that don't care about
        # sim_env — the legacy flow_out path always returns.
        sim_env = None
        actor = None
        field = None
        try:
            from src.system.runtime.simulation.mujoco import (
                FieldNotFoundError,
                MjActor,
                MjField,
                MjSimEnv,
                load_field,
            )

            # Backward-compat: tolerate the legacy ``default_field_id``
            # parameter key on canvases saved before v1.4.
            field_id = str(
                self.parameters.get('field_id')
                or self.parameters.get('default_field_id')
                or 'flat_ground'
            )

            # robot_type IS the actor id — there is no separate
            # ``default_actor_id`` parameter in v1.4.
            actor = MjActor.from_menagerie(robot_type)

            # Resolve field via project-local + shared library; the
            # active project (if any) wins over shared.
            project_root = None
            try:
                from src.system.core import project_session
                project_root = project_session.get_active_project_root()
            except Exception:
                project_root = None

            try:
                field = load_field(field_id, project_root=project_root)
            except FieldNotFoundError:
                log.warning(
                    "StartNode '%s': field_id '%s' not found in project-local "
                    "(%s) or shared library. Falling back to flat_ground.",
                    self.node_id, field_id, project_root,
                )
                field = MjField.flat_ground()

            sim_env = MjSimEnv.build(actor, field)
            # Stamp the render_enabled flag onto the constructed
            # sim_env so downstream BehaviorNodes know whether to
            # spawn the passive viewer thread. Off-canvas callers
            # who build sim_env directly never set this attribute,
            # which getattr defaults to False — they stay headless.
            try:
                sim_env.render_enabled = bool(self.parameters.get('render_enabled', True))
            except Exception:
                pass
        except Exception as exc:
            log.warning(
                "StartNode '%s': failed to construct sim_env (%s). "
                "Downstream BehaviorNodes will fall through to the "
                "no_sim_env_and_no_bundle path.",
                self.node_id,
                exc,
            )

        return {
            'flow_out': None,
            'sim_env': sim_env,
            'actor': actor,
            'field': field,
        }

    def get_display_name(self) -> str:
        return "Start"

    def get_description(self) -> str:
        return (
            "Workflow entry point — picks robot brand+model (= the "
            "Mission actor) AND the field, and constructs the sim_env "
            "for downstream BehaviorNodes."
        )

    def get_output_ports(self) -> List[str]:
        return ["flow_out", "sim_env", "actor", "field"]

    def get_input_ports(self) -> List[str]:
        return []

    def get_region(self) -> str:
        return self.REGION_INITIAL_SETUP

    def to_code(self) -> str:
        robot_type = self.parameters.get('robot_type', 'go2')
        field_id = self.parameters.get('field_id', 'flat_ground')
        return (
            f"RobotContext.set_robot_type('{robot_type}')\n"
            f"# Mission sim_env: actor='{robot_type}', field='{field_id}'"
        )


class EndNode(BaseNode):
    """End node - workflow exit point.

    Collects all upstream input data and produces a workflow completion
    summary.  The ``status`` parameter allows the canvas author to label
    the exit condition (e.g. "success", "error", "timeout").
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "end")
        self.parameters = {
            'status': 'success',       # user-defined exit label
            'message': '',             # optional exit message
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        status = self.get_parameter('status', 'success')
        message = self.get_parameter('message', '')
        return {
            'workflow_status': status,
            'workflow_message': message,
            'collected_inputs': {k: v for k, v in inputs.items() if v is not None},
        }

    def get_display_name(self) -> str:
        return "End"

    def get_description(self) -> str:
        return "Workflow exit point"

    def get_output_ports(self) -> List[str]:
        return []

    def get_input_ports(self) -> List[str]:
        return ["flow_in"]

    def get_region(self) -> str:
        return self.REGION_OUTPUT_HANDLING

    def to_code(self) -> str:
        return "# End of workflow"
