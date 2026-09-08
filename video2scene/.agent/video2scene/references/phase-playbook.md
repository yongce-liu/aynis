# Phase playbook

## 0. Establish the target and evidence

Run from the repository root. Use a scene id derived from the video filename unless the user specifies one.

```bash
ffprobe -v error -show_streams -show_format inputs/<scene-id>.mp4
uv run python -m video2scene.evidence \
  inputs/<scene-id>.mp4 \
  --output-root outputs/<scene-id>
```

Inspect `frames/evidence_sheet.jpg` and individual frames with an image viewer. Sampling should cover:

- the reference `t=0` frame;
- at least two frames before and after every important contact/release/occlusion event;
- one or more settled/end-state frames;
- camera changes, cuts or motion blur that affect inference.

Create a compact evidence table before modeling:

| Field | Required content |
| --- | --- |
| Physical proposition | One falsifiable sentence describing bodies, interaction and expected response |
| Core entities | Shape, material class, dynamic/static/nonrigid role, relative scale |
| Initial state | Reference frame, pose, linear/angular velocity and confidence |
| Events | Frame/time interval and confidence |
| Constraints | Support, joints, drive laws, contact surfaces, occlusion |
| Assumptions | Absolute scale, hidden geometry, material/solver priors |
| Validation-only evidence | Later frames used only to compare timing/outcome |

If `t=0` is already in motion, estimate velocity from early frames and record uncertainty. Do not reset it to zero merely because the Blender scene is static.

## 1. Phase 1: scene contract and Blender handoff

### Implement

- Add or revise `scenes/<scene-id>.json`.
- Extend `schemas/scene.schema.json` only with fields that the code will consume.
- Reuse or add focused geometry/material modules; preserve metric scale and local coordinates.
- For soft bodies, export a watertight, consistently wound rest surface and selectors needed for support/contact/render binding.
- Keep source observations, inferred values, priors and randomization ranges distinguishable.

### Build and review

```bash
uv run python -m video2scene.build \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id>
```

Check `preview.png`, `comparison.jpg` and `overview.png` visually. Review silhouette, depth ordering, contact geometry, object scale, camera, material response, lighting and whether every physical body exists independently. Update `render_review.json` through the repository workflow rather than claiming review from file existence.

For fast follow-up only:

```bash
uv run python -m video2scene.build \
  --spec scenes/<scene-id>.json \
  --output-root /tmp/<scene-id>-handoff \
  --handoff-only --samples 1 --threads 1
```

A handoff-only success does not pass visual acceptance.

## 2. Phase 2: compile and validate Genesis physics

First inspect the live handoff and solver plan:

```bash
uv run python .agent/video2scene/scripts/inspect_repo.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id>
```

Use temporary outputs for experiments. Genesis cache directories should be writable and isolated:

```bash
export UV_CACHE_DIR=/tmp/skill2scene-uv-cache
export MPLCONFIGDIR=/tmp/skill2scene-mpl
export XDG_CACHE_HOME=/tmp/skill2scene-xdg-cache
```

Run a short no-render smoke first:

```bash
uv run python -m video2scene.simulate \
  --output-root /tmp/<scene-id>-smoke \
  --duration-s 0.1 --no-render --overwrite
```

The smoke root must already contain a Phase-1 handoff, or build it there first. Check `run.log`, `migration_plan.json`, `trace.npz` and `validation.json`; do not infer success from exit code alone.

Then render a meaningful single-environment preview:

```bash
uv run python -m video2scene.simulate \
  --output-root outputs/<scene-id> \
  --duration-s <event-covering-duration> \
  --overwrite
```

Review the first frame, each event frame and final frame. Tune one mechanism at a time with bounded parameter sweeps. Keep the accepted parameters in typed config or the scene contract, and document why they are scene-specific or general.

For deformable support, a valid initialization pattern is:

1. temporarily keep dynamic rigid bodies away from the contact region;
2. let the deformable body seat under gravity for a short, recorded duration;
3. zero deformable velocities if required by the intended initial state;
4. restore rigid poses and velocities exactly from the handoff;
5. start trace time at the restored `t=0` state.

This is initialization, not future-frame replay.

## 3. Phase 3: randomization and production

Audit lanes and ranges before GPU work:

```bash
uv run python .agent/video2scene/scripts/inspect_repo.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id>
```

Generate a small rendered preview first:

```bash
uv run python -m video2scene.generate_dataset \
  --spec scenes/<scene-id>.json \
  --asset-mode sampled \
  --samples 4 --num-envs 4 \
  --duration-s <event-covering-duration> \
  --render-preview \
  --output-root outputs/<scene-id>-preview \
  --overwrite
```

Inspect multiple environments, not only environment 0. Confirm that visible geometry, physical parameters, initial conditions and outcomes vary as declared, while every sample remains feasible.

Before a large run, benchmark at least two safe `num_envs` values using the full intended simulation duration or a clearly labelled throughput-only test. Observe host RAM, VRAM, build time and environment steps/s; choose the measured setting rather than the largest launchable batch.

Production is resumable:

```bash
uv run python -m video2scene.generate_dataset \
  --spec scenes/<scene-id>.json \
  --asset-mode sampled \
  --samples <count> --num-envs <measured-batch-size> \
  --seed <root-seed> \
  --duration-s <duration> \
  --output-root outputs/<scene-id> \
  --resume
```

After exit, count expected batches, `_SUCCESS` markers, manifests, trajectories and per-sample validation outcomes. A launched or detached process is not a completed dataset.

## 4. Focused repair workflow

When fixing an existing result:

1. reproduce the visible/physical defect from current artifacts;
2. quantify it from trace or image ROIs;
3. locate the owning layer: spec, Blender geometry/material, migration plan, solver/runtime, renderer, validation, or layout;
4. run the smallest meaningful parameter/code experiment;
5. inspect the candidate artifact;
6. add a regression check tied to the actual failure;
7. regenerate only the affected canonical artifacts and hashes.

Examples of proper separation:

- dark Genesis output: calibrate renderer light/color mapping, do not touch masses or contact;
- excessive rigid-soft embedding: compare step size/contact band/coupler settings, do not rewrite the final pose;
- geometry randomization absent in simulation: fix the execution lane and actual morph, not only `parameters.jsonl`;
- changed output layout: update `layout.py`, producers, consumers and tests together; do not add fallback search paths.
