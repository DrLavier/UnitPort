# AGENT.md — Agent & Contributor Entry Point

This file is the single entry point for **AI coding agents** and **human contributors** working in the UnitPort Studio source tree. It states the non-negotiable engineering principles and indexes the project knowledge base — it deliberately does **not** hold deep content itself.

- Deep, area-specific knowledge lives under [`knowledge_base/`](knowledge_base/README.md) — one self-contained, retrieval-friendly document per functional area.
- The authoritative, fully-detailed rule text (with rule numbers `§1–§12`) is in [`CLAUDE.md`](CLAUDE.md). When this file and `CLAUDE.md` disagree, `CLAUDE.md` wins.
- End-user and feature documentation is in [`README.md`](README.md). Contribution process: [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Project at a glance

UnitPort Studio is a **universal** robot training/deploy platform: one canvas, many robots, two simulation engines (MuJoCo + IsaacLab/PhysX), one click to a deployable policy bundle. "Universal" is load-bearing — the core stays brand-agnostic; anything brand-specific is a hot-pluggable adapter.

- Python 3.11.9 (enforced), PyQt6 desktop app, project-local venv at `.venv311`.
- Source tree: `src/` with five namespaces (`unitport_sdk`, `config`, `registers`, `runtime`, `application`) plus `src/nodes/` (canvas node catalog) and root-level `bootstrap/`, `localisation/`, `tests/`.

## Ground rules

Each rule is one line here; the linked knowledge-base doc carries the full mechanics. Rule numbers refer to `CLAUDE.md §1`.

1. **No brand/model in the core** (§1). Brand-aware code lives only under `application/service/{adapters,brands}/` or `registers/brands/`. → [registers](knowledge_base/registers.md)
2. **Fail loud — no silent fallbacks** (§8). Missing/malformed input ⇒ raise. Never substitute defaults, zero-fill, or "warn and continue". → [architecture](knowledge_base/architecture.md#fail-loud-doctrine)
3. **Solve at the root, ship the full framework** (§11). Every change covers all layers it crosses (harvest → persist → schema → consume → migrate → validate → UI); producer and consumer land together.
4. **The canonical IR catalog is hand-edited** (§2). Unmapped robot bodies surface an error to the UI — never auto-append roles. → [registers](knowledge_base/registers.md)
5. **Code is English** (§3). All identifiers, comments, docstrings. UI strings go through `tr()` / `i18n_bind` — never bare literals. → [ui-conventions](knowledge_base/ui-conventions.md)
6. **User state lives only under `Paths.USER_CONFIG_DIR`** (§4) — no fallback locations, ever. → [configuration-and-paths](knowledge_base/configuration-and-paths.md)
7. **All UI colors and font sizes come from `system.ini`** (§5). No hex literals or ad-hoc font keys in code. → [ui-conventions](knowledge_base/ui-conventions.md)
8. **App version is single-sourced** in `system.ini[System].version` (§6); compare with `packaging.version`, never string ordering. → [build-run-verify](knowledge_base/build-run-verify.md)
9. **Robot identity in artifacts is SKU-only** (§7). Display names are derived at the boundary via the registry, never carried as free-form strings. → [artifacts-and-deploy](knowledge_base/artifacts-and-deploy.md)
10. **Artifacts are portable and self-contained** (§9). Loaders read only the artifact — never reach back into local project state. → [artifacts-and-deploy](knowledge_base/artifacts-and-deploy.md)
11. **PD is parameterized by `(ωn, ζ)`, never raw `kp`/`kd`** (§10). Both engines derive real-unit gains from the same effective inertia. → [physics-and-pd](knowledge_base/physics-and-pd.md)
12. **Agents never run git write operations** (§12). Make working-tree changes, report them, and stop — the human contributor owns commits, branches, and pushes.

## Knowledge base index

Read the doc for the area you are about to touch **before** writing code there.

| Document | Covers | Read when… |
|---|---|---|
| [knowledge_base/architecture.md](knowledge_base/architecture.md) | Source-tree layout, the five `src/` namespaces, naming caveats, fail-loud doctrine | deciding where new code goes |
| [knowledge_base/sdk-contract.md](knowledge_base/sdk-contract.md) | `unitport_sdk` API surface: Paths, Config, DataManager, TasksManager/Task, import rules | touching any SDK primitive, file I/O, or background tasks |
| [knowledge_base/configuration-and-paths.md](knowledge_base/configuration-and-paths.md) | `system.ini` ⊕ `user.ini` overlay, `USER_CONFIG_DIR`, path tiers, per-user storage | reading/writing settings or any user-produced state |
| [knowledge_base/registers.md](knowledge_base/registers.md) | The global registry (`RegistryHub`), IR roles, SKUs, families, data JSON conventions, brand adapters | robot catalogs, IR mapping, families, node manifests |
| [knowledge_base/training-pipeline.md](knowledge_base/training-pipeline.md) | Canvas → TrainingSpec → compile → validate → launch; SB3 vs IsaacLab parity; AMP/motion; rewards paging; observations; gait commands | anything under `src/application/training/` or `src/nodes/` |
| [knowledge_base/physics-and-pd.md](knowledge_base/physics-and-pd.md) | `(ωn, ζ)` PD doctrine, gain solvers, the two calibration gates, torque ceilings | actuator gains, sim2sim behavior, PD panels |
| [knowledge_base/artifacts-and-deploy.md](knowledge_base/artifacts-and-deploy.md) | Policy bundles, `deploy_contract`, manifest rules, obs parity at deploy, portability | exporting, loading, or deploying policies |
| [knowledge_base/ui-conventions.md](knowledge_base/ui-conventions.md) | Theming, fonts, i18n, sidebar/widget rules, canvas internals pointers | any PyQt6 UI work |
| [knowledge_base/build-run-verify.md](knowledge_base/build-run-verify.md) | First-time setup, launch, tests, localisation rebuild, release SOP | environment setup, running tests, cutting a release |

Conventions for writing and maintaining these documents (frontmatter schema, chunking rules, how to add a new area): [`knowledge_base/README.md`](knowledge_base/README.md).

## Quick verification

```powershell
.\install.bat                              # first-time setup (Linux: ./install.sh)
.\.venv311\Scripts\python.exe main.py      # launch
.\.venv311\Scripts\python.exe -m pytest tests\   # run the test suite
.\localisation.bat                         # rebuild i18n catalogs after editing localisation/
```

Full commands, path sanity checks, and release SOP: [knowledge_base/build-run-verify.md](knowledge_base/build-run-verify.md).
