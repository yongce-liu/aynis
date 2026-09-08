#!/usr/bin/env python
"""Inspect the live repository contract and canonical scene artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
import tyro

from video2scene import __version__
from video2scene.dataset_artifacts import batch_complete
from video2scene.domain_randomization import parameter_execution
from video2scene.genesis_plan import MigrationError, build_plan
from video2scene.layout import SceneLayout, resolve_scene_layout
from video2scene.spec import DEFAULT_SPEC, digest, read_json, validate


@dataclass(frozen=True)
class Args:
    """Inspect a scene specification and any existing canonical artifacts."""

    spec: Path = DEFAULT_SPEC
    output_root: Path | None = None
    fail_on_invalid: bool = False


def validation_summary(path: Path) -> dict[str, Any] | None:
    """Return compact validation status without hiding failed checks."""
    if not path.is_file():
        return None
    payload = read_json(path)
    checks = payload.get("checks", {})
    return {
        "path": str(path),
        "passed": payload.get("passed"),
        "physics_validated": payload.get("physics_validated"),
        "training_ready": payload.get("training_ready"),
        "check_count": len(checks),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def artifact_state(root: Path, names: tuple[str, ...]) -> dict[str, bool]:
    """Report artifact presence relative to one stage root."""
    return {name: (root / name).is_file() for name in names}


def solver_summary(blender_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Compile the current handoff when available and summarize solver routing."""
    if not (blender_dir / "scene_manifest.json").is_file():
        return None, None
    try:
        plan = build_plan(blender_dir)
    except (MigrationError, OSError, ValueError) as error:
        return None, str(error)
    entities = {
        name: {
            "source_physics_type": item["source_physics_type"],
            "role": item["role"],
            "solver": item["solver"],
            "material": item["material"],
        }
        for name, item in plan["entities"].items()
    }
    return {
        "schema_version": plan["schema_version"],
        "solver_stack": plan["solver_stack"],
        "policy": plan["policy"],
        "entities": entities,
    }, None


def dataset_summary(layout: SceneLayout) -> dict[str, Any]:
    """Summarize durable batch markers and the aggregate manifest."""
    batches = sorted(path for path in layout.batches.glob("batch_*") if path.is_dir())
    marked_success = [path for path in batches if (path / "_SUCCESS").is_file()]
    verified_success = []
    corrupt_success = []
    for path in marked_success:
        try:
            complete = batch_complete(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            complete = False
        (verified_success if complete else corrupt_success).append(path)
    incomplete = [path for path in batches if (path / "_INCOMPLETE").is_file()]
    manifest_path = layout.batches / "dataset_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    manifest_fields = None
    if manifest is not None:
        manifest_fields = {
            key: manifest.get(key)
            for key in (
                "complete",
                "planned_samples",
                "completed_samples",
                "valid_samples",
                "invalid_samples",
                "training_ready",
            )
            if key in manifest
        }
    return {
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_status": manifest_fields,
        "batch_count": len(batches),
        "marked_success_batches": len(marked_success),
        "verified_success_batches": len(verified_success),
        "corrupt_success_batches": len(corrupt_success),
        "incomplete_batches": len(incomplete),
        "verified_success_paths": [str(path) for path in verified_success],
        "corrupt_success_paths": [str(path) for path in corrupt_success],
        "incomplete_paths": [str(path) for path in incomplete],
    }


def inspect(args: Args) -> tuple[dict[str, Any], bool]:
    """Build a JSON-serializable snapshot and return whether invalid evidence exists."""
    spec_path = args.spec.resolve()
    spec = read_json(spec_path)
    validate(spec)
    layout = resolve_scene_layout(spec["scene_id"], args.output_root)
    execution = parameter_execution(spec)
    lanes = Counter(item["lane"] for item in execution.values())
    physics_types = Counter(
        entity["physics"]["type"] for entity in spec["entities"].values()
    )
    generators = Counter(
        entity["geometry"]["generator"] for entity in spec["entities"].values()
    )
    solver, solver_error = solver_summary(layout.blender)
    blender_validation = validation_summary(layout.blender / "validation.json")
    sim_validation = validation_summary(layout.sim / "validation.json")
    dataset = dataset_summary(layout)
    invalid = (
        any(
            summary is not None and summary.get("passed") is False
            for summary in (blender_validation, sim_validation)
        )
        or solver_error is not None
        or dataset["corrupt_success_batches"] > 0
    )

    result = {
        "repository": {
            "version": __version__,
            "cwd": str(Path.cwd().resolve()),
        },
        "spec": {
            "path": str(spec_path),
            "sha256": digest(spec),
            "schema_version": spec["schema_version"],
            "scene_id": spec["scene_id"],
            "source": spec["source"],
            "entity_count": len(spec["entities"]),
            "physics_types": dict(sorted(physics_types.items())),
            "geometry_generators": dict(sorted(generators.items())),
            "events": spec.get("semantics", {}).get("events", []),
            "relations": spec.get("semantics", {}).get("relations", []),
        },
        "randomization": {
            "parameter_count": len(execution),
            "lane_counts": dict(sorted(lanes.items())),
            "parameters": execution,
        },
        "layout": {
            "root": str(layout.root),
            "frames": str(layout.frames),
            "blender": str(layout.blender),
            "sim": str(layout.sim),
            "batches": str(layout.batches),
        },
        "artifacts": {
            "evidence": artifact_state(
                layout.frames,
                ("video_metadata.json", "evidence_sheet.jpg"),
            ),
            "blender": artifact_state(
                layout.blender,
                (
                    "scene.blend",
                    "scene_spec.json",
                    "scene_manifest.json",
                    "scene.schema.json",
                    "validation.json",
                    "render_review.json",
                    "preview.png",
                    "comparison.jpg",
                    "overview.png",
                ),
            ),
            "simulation": artifact_state(
                layout.sim,
                (
                    "migration_plan.json",
                    "trace.json",
                    "trace.npz",
                    "validation.json",
                    "run_manifest.json",
                    "run.log",
                    "simulation.mp4",
                    "contact_sheet.jpg",
                    "source_simulation_grid.jpg",
                ),
            ),
        },
        "validation": {
            "blender": blender_validation,
            "simulation": sim_validation,
        },
        "solver_plan": solver,
        "solver_plan_error": solver_error,
        "dataset": dataset,
    }
    return result, invalid


def main(args: Args) -> int:
    """Print the current contract snapshot and optionally fail on invalid artifacts."""
    try:
        result, invalid = inspect(args)
    except Exception:
        logger.exception("Repository contract inspection failed")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.fail_on_invalid and invalid:
        logger.error("Existing artifacts contain failed validation or migration errors")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
