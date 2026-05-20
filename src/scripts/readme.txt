UnitPort training-script archive — one-function-per-file convention
====================================================================

This package holds the canonical "training-time function content" UnitPort
ships with. Eight categories live here, each in its own sub-package:

    rewards/         — SB3 + Isaac Lab reward presets
    terminations/    — SB3 + Isaac Lab termination presets
    observations/    — Isaac Lab observation presets
    discriminator/   — AMP discriminator method-override presets
    gait/            — Walk These Ways gait presets (quadruped)
    scenes/          — training playground scene catalogue
    il_envs/         — Isaac Lab environment preset templates
    training_motion/ — motion-clip labels / library / IR mapping

Convention for the first four ("kind-based" categories) and gait/scenes/il_envs
---------------------------------------------------------------------------

    <category>/
        registry.py          — aggregator only (scans sub-package, builds dict)
        __init__.py          — re-exports the public names
        sb3/                 — one file per SB3 entry (only where SB3 exists)
            <key>.py         — `ENTRY = <factory>_item(key="<key>", ...)`
        isaac_lab/           — one file per Isaac Lab entry
            <key>.py         — `ENTRY = <factory>_item(key="<key>", ...)`

Each entry file is self-contained:

  * imports only the constants it actually uses from scripts.task_module
    (BACKEND_*, ALG_*, *_FAMILIES, IL_MOD_INLINE, <factory>_item)
  * the Python source the IL compiler embeds verbatim is stored locally as
    `INLINE_SOURCE = '''...'''` — never pulled from a sibling file
  * the file's name (lowercase, snake_case) equals `ENTRY.key`

Aggregator behaviour
--------------------

`<category>/registry.py` uses `pkgutil.iter_modules` to discover every
non-underscore module under the backend sub-folders at import time. The
discovered ENTRY constants land in the legacy module-level dicts
(`REWARD_REGISTRY` / `IL_REWARD_REGISTRY` / `TERMINATION_REGISTRY` /
`IL_TERMINATION_REGISTRY` / `IL_OBS_REGISTRY` / `IL_DISC_REGISTRY`) and
in `scripts.query.ALL_REGISTRIES` — same names, same types, same
object identity as before the split. UI editors mutate these dicts
in-place via `dataclasses.replace(sub[key], il_inline=...)`; the IL
compiler reads `il_inline` and `il_func` / `il_module` / `il_params`
on a per-entry basis. None of that path changed.

Special cases:

  * `gait/presets_data/<name>.py` exports `ENTRY: GaitPreset` AND an
    integer `ORDER` field — the aggregator builds `DEFAULT_PRESETS` as
    an ordered list, lowest ORDER first.
  * `scenes/builtin/<scene_id>.py` exports `ENTRY: Scene`; the
    aggregator's `_install_defaults()` runs at module import and
    populates the process-local `_REGISTRY` dict (the same import-time
    side effect the pre-split version had).
  * `il_envs/presets_data/<slug>.py` exports `NAME` (the user-visible
    preset name, e.g. "Go2 Flat Velocity"), `TASK_NAME` (Isaac Lab
    registered task name), `ORDER` and `ENTRY` (the full preset dict).
  * `training_motion/` is unchanged — it is already organised by
    functional sub-module (labels / library / ir_mapping) rather than
    by per-key files, which fits its filesystem-scanning role better.

Adding a new entry
------------------

  1. Drop a new file `<category>/<backend>/<your_key>.py` matching the
     template above.
  2. Re-import the project. The aggregator picks it up automatically —
     no edits to `registry.py` / `__init__.py` are needed.

Removing or renaming an entry
-----------------------------

Removing the file removes the entry; renaming the file renames the
key implicitly (but make sure `ENTRY.key` matches the new filename).

Do NOT put any user-modifiable state inside this tree
-----------------------------------------------------

This package ships with the SDK and is read-only at runtime — the same
rule as `src/config/system.ini`. User edits to an entry's il_inline
source come in via the Registry Module Editor and are persisted as a
JSON sidecar under `~/UnitPort/scripts/<kind>_<backend>.json` (via
`scripts.query.emit_inline_overrides`), never written back into the
files here.
