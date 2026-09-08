# Validation and acceptance

Validation must test the physical proposition and artifact integrity, not merely program execution.

## Evidence hierarchy

| Evidence | What it proves | What it does not prove |
| --- | --- | --- |
| Schema/unit tests | Code contracts and deterministic helpers | Visual quality or real simulator behavior |
| Blender handoff-only build | Geometry/export path works | Render quality or physical behavior |
| Blender full render | Initial scene can be rendered and inspected | Genesis migration correctness |
| Genesis no-render smoke | Solver can build and step; state can be recorded | Event-level realism or visual acceptance |
| Rendered nominal preview | Candidate appearance and dynamics are inspectable | Dataset-scale robustness |
| Small randomized preview | Randomization reaches real execution paths for sampled cases | Full batch validity or throughput |
| Completed resumable batch | Artifacts and identity are durable | Training readiness without aggregate physical checks |

## Phase 1 gate

Require all of the following for a reviewed nominal scene:

- source hash, metadata, reference frame and event evidence exist;
- `scene.blend` reopens in a fresh process;
- `scene_spec.json` and `scene_manifest.json` conform to the copied schema;
- each semantic entity has one visual owner and correct physics representation;
- exported meshes use body-local meters and the manifest transform is applied exactly once;
- deformable physics mesh is closed/watertight and required selectors are nonempty;
- dynamic bodies have positive initial clearance unless contact at `t=0` is explicitly observed;
- projection targets and camera framing are within declared tolerance;
- preview/overview are nonblank and were actually viewed;
- `render_review.json` hashes match the reviewed images;
- `physics_validated=false` and `training_ready=false` remain until later evidence exists.

Initial-frame pixel similarity is useful but not sufficient. Inspect silhouettes, occlusion, scale, depth, support/contact topology and material plausibility.

## Phase 2 gate

Define checks from scene semantics rather than fixed entity names. A typical dynamic scene should verify:

- exact or bounded `t=0` position and orientation error after relaxation;
- initial linear/angular velocities match the handoff;
- relaxed deformable surface remains within a justified tolerance of the intended initial surface;
- all rigid/deformable states remain finite;
- required contacts/releases occur within observed time windows;
- both sides of a claimed interaction respond physically;
- no support tunneling or unacceptable interpenetration;
- task-specific outcome, such as tipping, rebound, retention, rolling, flow or joint travel;
- encoded video dimensions, fps, duration/frame count and artifact hashes;
- no source-video access, keyframe replay or runtime correction after `t=0`.

Thresholds must come from observable event scale, geometry scale, solver resolution or a documented engineering tolerance. Do not loosen a threshold solely to make a run pass.

当前名义运行把场景专属要求写入 `physics_validation.checks`。`soft_impact` 检查接触、事件时序、软体响应、支撑穿透和末态；`rolling_transition` 检查跨支撑进入时序、位移方向、连续支撑、滚动滑移率、横向漂移、竖直波动和速度界限。新增物理命题应新增有测试的类型，不得在 `simulate.py` 中按实体名分支。

Separate gates:

- `passed`: requested artifact and preview-level checks pass;
- `physics_validated`: initial state, events, interactions and outcome pass for this run;
- `training_ready`: additionally requires trustworthy parameter provenance, stable numerics, sufficient coverage and aggregate batch validation.

A user may accept a visually convincing preview while a numerical issue remains a training blocker. Record that issue under diagnostics; never silently discard it.

## Phase 3 gate

For every declared randomization parameter, retain:

- name and JSON pointer;
- requested range/distribution and provenance;
- execution lane (`per_environment` or `per_batch`);
- requested and effective value;
- application mechanism;
- evidence that the actual scene/solver value changed.

For every batch, verify:

- deterministic sample IDs and seeds;
- run identity matches the dataset manifest;
- `_SUCCESS` hash points to the current `run_manifest.json`;
- trajectory arrays have expected `[T, B, ...]` shapes and finite values;
- `parameters.jsonl` has one row per environment;
- per-sample validation and outcome labels exist;
- invalid samples remain visible in aggregate counts;
- rendered preview, when requested, covers more than one environment;
- throughput and resource metrics are measured, not guessed.

Dataset completion requires the requested sample count and all expected batches. `valid_count == sample_count` is desirable but not automatically required; the acceptance policy must state whether invalid samples are retained, regenerated or block delivery.

## Visual review checklist

View actual artifacts rather than filenames:

- first frame: camera, scale, pose, visibility, support and initial gaps;
- event frame: contact location, deformation, shadows, motion direction and any occlusion;
- final frame: settling, penetration, unsupported floating, implausible damping or energy gain;
- overview: hidden collision geometry, support thickness and off-camera structures;
- randomized grid: real shape/material/physics diversity without repeated clones.

Reject toy-like results caused by raw primitives, razor-sharp edges, flat colors, default lighting, scale mismatch or collision geometry visibly disagreeing with the mesh.

## Anti-patterns

Never count these as success:

- a process exists but no final marker/manifest was written;
- videos exist while required stills, traces or validation files are missing;
- semantic final state matches but initial frame/camera/geometry is wrong;
- source footage is composited behind simulated objects;
- motion is replayed from observed frame positions;
- a nonrigid body is rendered as deforming but simulated as a rigid proxy;
- a random parameter appears in JSON but is shared or ignored by the solver;
- a very short smoke is reported as physical validation;
- output files are counted without reading per-sample validity;
- legacy directories are accepted to hide a broken canonical path.

## Minimum regression suite

After code changes, normally run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run python .agent/video2scene/scripts/inspect_repo.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id>
uv run python .agent/video2scene/scripts/evolution_audit.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id> \
  --fail-on-findings
```

Then run the smallest real Blender/Genesis scenario that exercises the changed path. For renderer or dynamics changes, tests alone are insufficient: render and inspect at least one representative scene.
