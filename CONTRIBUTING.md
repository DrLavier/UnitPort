# Contributing to UnitPort

Thanks for your interest in contributing — UnitPort is a community project and
all sensible improvements are welcome.

This document covers the practical bits: how to file a useful issue, how to
get a development environment running, and what we look for in a pull
request. The architectural ground rules (multi-brand inclusiveness, no silent
fallbacks, portable artifacts) are documented in the README and enforced in
code review.

## Ways to contribute

- **Bug reports** — open a [GitHub issue](https://github.com/DrLavier/UnitPort/issues).
- **Feature requests / design discussion** — open a
  [GitHub discussion](https://github.com/DrLavier/UnitPort/discussions).
- **Documentation fixes** — PRs against `README.md`, `CHANGELOG.md`, or the
  `localisation/` catalogs are always welcome.
- **New robot adapters / brand packages** — see "Adding a robot" below.
- **Localisation** — drop a new `localisation/<LANG>/` folder mirroring `EN/`
  and open a PR.
- **Security issues** — please follow `SECURITY.md` instead of filing a
  public issue.

## Before you file an issue

Please include:

1. Your OS + Python version (UnitPort enforces Python 3.11).
2. UnitPort version (`system.ini[System].version`, or the git tag / commit
   you're on).
3. The full traceback if it's a crash. If it's a behavioural issue, a minimal
   canvas (`.canvas.json`) that reproduces the problem.
4. Whether you're using the SB3 backend or the Isaac Lab backend.

For Mission Control issues, also include the robot brand + model and the
adapter logs (sidebar → diagnostics).

## Development setup

```bash
# Windows
install.bat
start.bat

# Linux
chmod +x install.sh start.sh
./install.sh
./start.sh
```

The launcher creates a project-local `.venv311` using Python 3.11. Do not
install dependencies into your global Python — UnitPort assumes the venv
exists and re-execs under it.

### Running tests

```bash
# Windows
.venv311\Scripts\python.exe -m pytest

# Linux
.venv311/bin/python -m pytest
```

### Type check / syntax check

```bash
.venv311\Scripts\python.exe -m py_compile <changed_file.py>
```

## Code style

- **Python 3.11**. Type hints on new public functions.
- **Bilingual codebase** — CN + EN comments and docstrings are both welcome.
  UI strings must go through `tr()` / `i18n_bind`, never hardcoded.
- **PyQt6 only**. No PySide imports.
- **No silent fallbacks**. When a required input is missing or malformed,
  raise with a clear message. Read the "Fail loud, never silent" principle
  in the README before adding any `except: pass`-style code path.
- **Theme**. Colours and font sizes go through
  `Config.get_color(slot, fallback)` / `Config.get_font_size(slot)`. Hex
  literals in widgets will be rejected in review.
- **Paths**. Path constants come from `Paths`; on-disk I/O goes through
  `DataManager`. Do not call `open()` / `Path(...)` / `ConfigParser()`
  directly in business code.
- **Threading**. Cross-thread communication uses `pyqtSignal`. Inside
  `Task.run`, use `self.sleep()` and `self.check_cancelled()` — never
  `time.sleep`.

## Adding a robot (canonical registry)

UnitPort's core is brand-agnostic. To add a robot:

1. Open an issue with the URDF / MJCF (or a link), the robot's family
   (quadruped / humanoid / biped / wheeled / manipulator), and a body-name
   → IR-role mapping. The body-name mapping is the one thing the importer
   can't auto-derive.
2. Add the registry entry to `src/registers/data/robots_canonical.json` via
   PR. Each robot gets an immutable SKU.
3. If the robot needs a vendor SDK adapter, add it under
   `src/application/service/adapters/<vendor>/`.
4. If the robot needs brand-package hooks (icons, default canvases), add
   them under `src/application/service/brands/<vendor>/`.

**No brand strings in core paths.** All brand-specific code lives in
adapter / brand-package directories.

## Pull request checklist

- [ ] Your branch is up to date with `main`.
- [ ] Tests pass (`pytest`).
- [ ] No new hex colour literals or hardcoded paths.
- [ ] New UI strings are routed through `tr()` / `i18n_bind`.
- [ ] If you added a feature flag or temporary skip, the PR description
      documents when it should be removed.
- [ ] If you changed a public API surface (canvas spec, IR, bundle format,
      registers catalogs), the change is mentioned in `CHANGELOG.md`.
- [ ] The PR title is descriptive (no "fix" / "update" alone).

## Code review

Reviews focus on:

- **Correctness** — does it actually do what the description says.
- **Brand-inclusiveness** — could this assumption break for another vendor's
  robot.
- **Failure mode** — do errors surface loudly with actionable messages.
- **Artifact portability** — bundles / exports must remain self-contained.

Maintainers may merge with squash-and-commit. By submitting a PR you license
your contribution under the Apache License 2.0 (per Apache §5).

## Community

- Website: [uniport.ai](https://uniport.ai)
- Issues: [GitHub Issues](https://github.com/DrLavier/UnitPort/issues)
- Discussions: [GitHub Discussions](https://github.com/DrLavier/UnitPort/discussions)
