<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

<!-- version: 1.2.0 -->

# v1.2.0

### Major: AI Build — an agent team now builds your training canvas

This release introduces **AI Build**, an agent-orchestration system built into the canvas. Describe the training you want in plain language — *"make Go2 walk at 1 m/s and turn in place"* — and a team of specialized agents designs, wires, parameterizes and validates the whole canvas for you: robot, actor & environment, rewards, training and export. You watch every edit land live on the canvas, and nothing is ever saved unless the result compiles clean.

### How the harness is arranged

AI Build is not a single free-running model — it is a **deterministic harness** that schedules small, verifiable agent passes and keeps every hard decision out of the model's hands:

- **Silent intent gate.** Your message is first classified *without* any project knowledge attached: is it a canvas build/modify request that needs the knowledge base? If yes, the build pipeline engages; if not, the assistant simply answers from its own knowledge in a chat bubble — no canvas access, nothing committed. Conversation history rides along either way, so follow-ups keep their referents.
- **Understanding.** A Design agent reads the request (plus the thread so far) and emits a short architecture brief — command items, locomotion emphasis, special features.
- **Locked requirements, zero tokens.** Cheap decisions you already made — robot SKU, training length, compute intensity, reference motion — are applied deterministically before any agent runs, then locked in every prompt so no tokens are spent re-deciding them. PD gains are auto-derived from the robot asset the same way.
- **Configuration, block by block.** An Orchestrator agent builds the canvas in five scoped passes — Robot → Actor & Env → Rewards → Training → Export — each with only that block's node vocabulary. Every single write goes through a validated mutation API: an out-of-bounds weight or a bad wire is rejected on the spot and the agent corrects itself. The rewards pass ends with a self-review against known degenerate optima (reward hacking) before moving on.
- **Validate → repair → commit, deterministically.** A compile gate, cross-node integrator and a critic review the finished canvas; blocking issues are routed back to the block that owns them for a bounded number of repair rounds. The commit decision is never the model's: **zero errors within budget = save; anything else = nothing persisted**, with the remaining issues reported. A final no-token pass guarantees the delivered canvas is at least structurally complete and tidily laid out.

### Also in AI Build

- **Checkpoint, Retry and config versions.** Every run snapshots the canvas first — Reset reverts it, Retry re-runs the last prompt from that point, and each run that changed the canvas is saved as a selectable configuration version (with *Origin* always available).
- **Per-canvas conversation threads** with persistent history, pause/resume mid-run, live per-phase token metering, and node highlighting that follows the agents around the canvas.
- **Bring your own model.** AI Build speaks to any OpenAI-compatible endpoint; configure base URL, model and key in the panel settings — the key can be read from an environment variable and never written to disk.

### Training Motion node — clip editing & segment marking, built in

The **Clip Motion Editor** now lives directly inside the Training Motion node, so choosing and trimming a reference motion no longer means a detour to the Resources panel. Pick a clip from the dropdown and it previews immediately on your canvas robot in an embedded viewer — scrub the timeline, orbit and zoom the render, and read the exact frame you are on.

- **Mark segments without a second tool.** Set an in/out range on the timeline (or type Start/End frames), then **Crop** for a one-click quick-cut or **Mark Segment** to name and tag it. Each clip and its saved segments appear in the same Clip dropdown, so a trimmed sub-motion is trainable without producing a new file.
- **Pick what trains, right in the table.** The segment list has a checkbox column — tick a segment to train on just that sub-range, or leave every box clear to train on the whole clip. The active choice is highlighted so the item's reference is always obvious.
- **Robot-aware.** The preview always uses the robot wired on your canvas; a clip whose joints that robot mostly lacks is flagged in red in the dropdown *before* you commit to it, so a mismatched motion can't slip through.
- **Never blocks the UI.** Clips are scanned and joint-matched on a background thread — the editor opens right away with a small loading indicator and fills in when ready.
