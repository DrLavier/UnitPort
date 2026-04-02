# IR Workbench

This workspace is used for iterative IR development in UnitPort.

## Purpose

- host IR authoring and transpilation code,
- hold readable IR artifacts for review,
- keep temporary files isolated from stable modules.

## Layout

- `code/`: development code for IR parsing, normalization, transpilation, validation.
- `readable/`: human-readable IR files generated from Canvas/Compiler inputs.
- `templates/`: canonical readable IR templates used as authoring references.
- `tmp/`: temporary outputs, test fragments, scratch files.

## Workflow (current)

1. Start from `templates/ir_template_v0_1.yaml`.
2. Produce/adjust readable IR in `readable/`.
3. Implement translators/validators in `code/`.
4. Keep disposable artifacts in `tmp/`.

## Notes

- This folder is intentionally iterative, not final.
- Keep all docs and templates in English for agent interoperability.
