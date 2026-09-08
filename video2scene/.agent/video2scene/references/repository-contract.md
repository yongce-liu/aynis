# Repository contract

此文件描述当前仓库接口与扩展位置。执行任务时以实际代码、测试和 `inspect_repo.py` 输出为准；接口改变时同步更新本文件。

## Canonical layout

所有阶段共享一个 `outputs/<scene-id>/` 根目录，由 `video2scene/layout.py` 管理：

```text
outputs/<scene-id>/
├── frames/
│   ├── video_metadata.json
│   ├── evidence_sheet.jpg
│   └── frame_XXXX.png
├── blender/
│   ├── scene.blend
│   ├── scene_spec.json
│   ├── scene_manifest.json
│   ├── scene.schema.json
│   ├── validation.json
│   ├── render_review.json
│   ├── preview.png
│   ├── comparison.jpg
│   ├── overview.png
│   ├── assets/{visual,physics}/
│   └── variants/seed_XXXXXX/
├── sim/
│   ├── migration_plan.json
│   ├── trace.{json,npz}
│   ├── validation.json
│   ├── run_manifest.json
│   ├── run.log
│   ├── source_simulation_grid.jpg
│   └── rendered artifacts...
└── batches/
    ├── dataset_manifest.json
    ├── generate.log
    └── batch_XXXXXX/
        ├── _INCOMPLETE or _SUCCESS
        └── sim/
```

不要重新引入 `reference/`、`main/`、`genesis/` 或 batch 内 `blender/` 等旧布局。路径变化只能通过 `SceneLayout`/`BatchLayout` 完成。

## Entrypoints

| Stage | Command/module | Responsibility |
| --- | --- | --- |
| Evidence | `video2scene.evidence` | Probe video and extract review frames into `frames/` |
| Phase 1 | `video2scene.build` | Resolve spec, build Blender scene, render/export/validate handoff |
| Phase 1 validation | `video2scene.validate` | Reopen saved `.blend` and verify scene/assets/projections |
| Phase 2 planning | `video2scene.genesis_plan` | Convert manifest semantics to explicit solver decisions; reject unsupported physics |
| Phase 2 runtime | `video2scene.simulate` | Run one Genesis scene, record trace/render/validation |
| Phase 3 | `video2scene.generate_dataset` | Plan deterministic samples, build/reuse batch handoffs, run and resume batches |
| Contract inspection | `.agent/video2scene/scripts/inspect_repo.py` | Import the live repository code and summarize spec, lanes, solver plan and artifacts |
| Self-evolution audit | `.agent/video2scene/scripts/evolution_audit.py` | Detect structural drift, failed evidence, broken skill links and scene-specific coupling before/after edits |

## Module ownership

- `spec.py`: schema validation, JSON pointers, deterministic constrained sampling. General physical feasibility belongs here only when it is truly scene-independent.
- `geometry.py`, `materials.py`: reusable Blender primitives and material construction. Geometry functions return semantic root objects; decorative children remain owned by that root.
- `build.py`: orchestration and geometry dispatch. Collection routing, material kind, optional sphere surface markings and active-object selection are driven by the scene contract rather than fixed entity names.
- `assets.py`: export evaluated meshes in body-local meters, preserve vertex ordering and semantic ownership. Extend metadata when a new solver needs additional rest-state data.
- `validate.py`: saved Blender scene and handoff validation. Projection targets establish execution consistency, not true 3D accuracy.
- `genesis_plan.py`: source-of-truth routing from `physics_type`/collision representation to solver/material. Current implemented routes are analytic rigid/static bodies and closed-volume soft bodies through FEM; cloth/fluid intentionally fail.
- `simulation_runtime.py`: actual Genesis scene, morph, material, coupler, camera/light and entity construction. A plan entry is not implemented until this module applies it.
- `simulate.py`: nominal state capture and physical acceptance. It consumes `physics_validation.checks`; implemented nominal validators are `soft_impact` and `rolling_transition`, while new physical claims require a new typed validator rather than entity-name branches.
- `domain_randomization.py`: deterministic seeds, constrained batch plans, per-environment versus per-batch execution lanes, runtime entity resolution.
- `dataset_simulation.py`: batched state application, trajectory capture and per-sample physical validation. It consumes the same typed `physics_validation` rules as the nominal path and records generic per-rule outcome labels.
- `dataset_artifacts.py`: completeness and identity checks for resumable batches.
- `simulation_artifacts.py`: video encoding, source-reference comparison and contact sheets.
- `simulation_config.py`: shared typed Genesis options; defaults are calibrated for the reference scene, not universal physical constants.
- `tests/`: executable contracts. Prefer behavior/invariant tests over checking exact prose or implementation details.
- `.agent/video2scene/evolution-baseline.json`: reviewed structural snapshot of public module APIs, schema enums and required skill resources. Refresh only after verified intentional changes.

## Current schema boundary

`schemas/scene.schema.json` and `scenes/*.json` use `video2scene.scene.v1`. The current Blender geometry enum is limited to `sphere`, `box`, `cushion`, and `clamp`; the current Genesis compiler accepts rigid/static `sphere` or `box`, and soft `closed_rest_surface` with FEM parameters. Do not declare a new type in JSON without implementing its full path.

A Phase-1 entity should preserve at least:

- stable semantic name and label;
- geometry generator/asset and metric dimensions;
- world transform and coordinate conventions;
- appearance parameters and provenance;
- physics type, collision representation and material priors;
- initial linear/angular state;
- visual and physics asset mapping in the resolved manifest;
- deformable rest representation and selectors when applicable.

The scene contract should also contain source hash/frame, physical semantics, event windows, relations, camera, lighting, assumptions, quality flags and domain-randomization ranges. Nominal Phase-2 acceptance lives under `physics_validation.checks`: each item has a stable `id`, a typed validator (`soft_impact` or `rolling_transition`), semantic entity/support references, event linkage and scale-derived thresholds. Appearance may select reusable procedural `material_kind`, optional sphere `surface_detail`, and an explicit `render_collection`; these fields affect authored assets only and never introduce source-frame textures.

## Extension checklist

### Add a geometry generator

1. Extend the schema enum and required fields.
2. Implement a reusable constructor in `geometry.py` or a focused new module.
3. Add dispatch in `build.py` without entity-name conditionals where possible.
4. Export visual/physics representations and any required rest data.
5. Add post-save structural and metric checks.
6. Decide whether Genesis uses an analytic morph, mesh, MJCF, particles or an unsupported failure.
7. Add nominal and randomized tests.

### Add a physical mechanism

1. Express the mechanism in the scene contract, including states, constraints and observed event.
2. Add an explicit `choose_solver()` route and migration-plan fields.
3. Verify the installed Genesis API from the environment before coding against it.
4. Implement construction and state initialization in `simulation_runtime.py`.
5. Record states sufficient to test the physical proposition.
6. Add scene-independent checks where possible and a scene-specific validator only where necessary.
7. Run no-render smoke, rendered preview and source-event comparison.

### Add a randomized parameter

1. Declare a physically plausible range and provenance in the scene spec.
2. Keep sampling deterministic and constrained; impossible samples must fail after bounded attempts.
3. Classify the finest truthful execution lane in `parameter_execution()`.
4. Apply the value to actual Blender/Genesis objects.
5. Persist requested and effective values in batch artifacts.
6. Test that changing the parameter changes geometry, mass, material, initial state or rendering as claimed.

## Known reference-scene assumptions

Before using a new scene, search at least:

```bash
rg -n 'ball|block|cushion|falls_onto|contact.*window|tipped|retained' \
  video2scene tests
```

Do not merely rename new entities to `ball` or `block` to satisfy existing checks. Add a typed `physics_validation` rule, move reusable logic to rule/event-driven code, keep unavoidable task semantics in the scene spec, and update both nominal and batch paths before claiming Phase 3 support.
