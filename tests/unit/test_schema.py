#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the Schema system."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.schema.node_schema import NodeSchema, PortSchema, ParamSchema, ParamConstraint
from compiler.schema.registry import SchemaRegistry
from compiler.ir.types import IRType, PortDirection


class TestParamConstraint(unittest.TestCase):
    def test_round_trip(self):
        c = ParamConstraint(min_value=0.0, max_value=60.0, choices=["a", "b"])
        d = c.to_dict()
        c2 = ParamConstraint.from_dict(d)
        self.assertEqual(c.min_value, c2.min_value)
        self.assertEqual(c.max_value, c2.max_value)
        self.assertEqual(c.choices, c2.choices)

    def test_empty(self):
        c = ParamConstraint()
        d = c.to_dict()
        self.assertEqual(d, {})


class TestPortSchema(unittest.TestCase):
    def test_round_trip(self):
        p = PortSchema(name="flow_in", direction=PortDirection.INPUT,
                       data_type=IRType.VOID)
        d = p.to_dict()
        p2 = PortSchema.from_dict(d)
        self.assertEqual(p.name, p2.name)
        self.assertEqual(p.direction, p2.direction)
        self.assertEqual(p.data_type, p2.data_type)


class TestParamSchema(unittest.TestCase):
    def test_round_trip_with_constraints(self):
        p = ParamSchema(
            name="duration", param_type=IRType.FLOAT, default=1.0,
            constraints=ParamConstraint(min_value=0, max_value=60),
            unit="seconds",
        )
        d = p.to_dict()
        p2 = ParamSchema.from_dict(d)
        self.assertEqual(p.name, p2.name)
        self.assertEqual(p.param_type, p2.param_type)
        self.assertEqual(p.default, p2.default)
        self.assertIsNotNone(p2.constraints)
        self.assertEqual(p2.constraints.max_value, 60)
        self.assertEqual(p2.unit, "seconds")


class TestNodeSchema(unittest.TestCase):
    def test_round_trip(self):
        schema = NodeSchema(
            schema_id="builtin.action_execution",
            display_name="Action Execution",
            node_type="action_execution",
            kind="action",
            ports=[
                PortSchema("flow_in", PortDirection.INPUT, IRType.VOID),
                PortSchema("flow_out", PortDirection.OUTPUT, IRType.VOID),
            ],
            parameters=[
                ParamSchema("action", IRType.STRING, "stand",
                            ParamConstraint(choices=["stand", "sit", "walk"])),
            ],
            code_template="RobotContext.run_action('{action}')",
            robot_compat=["go2", "a1"],
        )
        d = schema.to_dict()
        s2 = NodeSchema.from_dict(d)
        self.assertEqual(s2.schema_id, "builtin.action_execution")
        self.assertEqual(len(s2.ports), 2)
        self.assertEqual(len(s2.parameters), 1)
        self.assertEqual(s2.parameters[0].constraints.choices, ["stand", "sit", "walk"])

    def test_get_ports(self):
        schema = NodeSchema(
            schema_id="test", display_name="Test", node_type="test", kind="test",
            ports=[
                PortSchema("in1", PortDirection.INPUT),
                PortSchema("in2", PortDirection.INPUT),
                PortSchema("out1", PortDirection.OUTPUT),
            ],
        )
        self.assertEqual(len(schema.get_input_ports()), 2)
        self.assertEqual(len(schema.get_output_ports()), 1)

    def test_get_parameter(self):
        schema = NodeSchema(
            schema_id="test", display_name="Test", node_type="test", kind="test",
            parameters=[
                ParamSchema("action", IRType.STRING, "stand"),
                ParamSchema("speed", IRType.FLOAT, 1.0),
            ],
        )
        self.assertIsNotNone(schema.get_parameter("action"))
        self.assertIsNone(schema.get_parameter("nonexistent"))


class TestSchemaRegistry(unittest.TestCase):
    def setUp(self):
        SchemaRegistry.reset()

    def test_register_and_get(self):
        schema = NodeSchema(
            schema_id="test.my_node",
            display_name="My Node",
            node_type="my_node",
            kind="custom",
        )
        SchemaRegistry.register(schema)
        result = SchemaRegistry.get("test.my_node")
        self.assertIsNotNone(result)
        self.assertEqual(result.display_name, "My Node")

    def test_get_by_node_type(self):
        schema = NodeSchema(
            schema_id="test.action",
            display_name="Action",
            node_type="action_execution",
            kind="action",
        )
        SchemaRegistry.register(schema)
        result = SchemaRegistry.get_by_node_type("action_execution")
        self.assertIsNotNone(result)

    def test_get_by_display_name(self):
        schema = NodeSchema(
            schema_id="test.action",
            display_name="Action Execution",
            node_type="action_execution",
            kind="action",
        )
        SchemaRegistry.register(schema)
        result = SchemaRegistry.get_by_display_name("Action Execution")
        self.assertIsNotNone(result)

    def test_load_builtins(self):
        SchemaRegistry.load_builtins()
        ids = SchemaRegistry.list_schema_ids()
        self.assertIn("builtin.action_execution", ids)
        self.assertIn("builtin.stop", ids)
        self.assertIn("builtin.if", ids)
        self.assertIn("builtin.while_loop", ids)
        self.assertIn("builtin.comparison", ids)
        self.assertIn("builtin.sensor_input", ids)
        self.assertIn("builtin.math", ids)
        self.assertIn("builtin.timer", ids)
        self.assertIn("builtin.variable", ids)
        self.assertIn("builtin.opaque_code", ids)

    def test_builtin_action_schema(self):
        SchemaRegistry.load_builtins()
        schema = SchemaRegistry.get("builtin.action_execution")
        self.assertIsNotNone(schema)
        self.assertEqual(schema.kind, "action")
        self.assertEqual(len(schema.get_input_ports()), 1)
        self.assertEqual(len(schema.get_output_ports()), 1)
        action_param = schema.get_parameter("action")
        self.assertIsNotNone(action_param)
        self.assertIn("stand", action_param.constraints.choices)

    def test_builtin_timer_schema(self):
        SchemaRegistry.load_builtins()
        schema = SchemaRegistry.get("builtin.timer")
        self.assertIsNotNone(schema)
        duration_param = schema.get_parameter("duration")
        self.assertIsNotNone(duration_param)
        self.assertEqual(duration_param.constraints.min_value, 0.0)
        self.assertEqual(duration_param.constraints.max_value, 60.0)

    def test_reset(self):
        SchemaRegistry.load_builtins()
        self.assertGreater(len(SchemaRegistry.list_schema_ids()), 0)
        SchemaRegistry.reset()
        # After reset, _loaded is False, so next access triggers reload
        # But the cache is cleared
        self.assertEqual(len(SchemaRegistry._schemas), 0)


if __name__ == "__main__":
    unittest.main()
