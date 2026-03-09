#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry for Script Box built-in helper functions."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ScriptBuiltinSpec:
    """Built-in function metadata shown in ScriptEditor."""

    name: str
    example: str
    description: str


SCRIPT_BUILTIN_FUNCTIONS: List[ScriptBuiltinSpec] = [
    ScriptBuiltinSpec(
        name="getInput",
        example="speed = getInput(TYPE='float')",
        description=(
            "Declare/read one input. NAME is optional (auto from assignment). "
            "Supported TYPE: any/bool/int/float/str/list/dict/number/function/protocol."
        ),
    ),
    ScriptBuiltinSpec(
        name="setOutput",
        example="setOutput(result, TYPE='int')",
        description=(
            "Declare one output. NAME is optional (auto from output variable/expression). "
            "Supported TYPE: any/bool/int/float/str/list/dict/number/function/protocol."
        ),
    ),
]
