"""Generate a domain-randomized dataset with Genesis parallel environments."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import tyro
from loguru import logger

from .dataset_artifacts import batch_complete, remove_legacy_batch_artifacts
from .dataset_simulation import run_parallel_batch
from .domain_randomization import (
    derive_seed,
    parameter_execution,
    plan_batch,
)
from .filesystem import remove_path
from .hashing import sha256_file
from .layout import BatchLayout, PROJECT_ROOT, SceneLayout, resolve_scene_layout
from .logging_utils import stage_log
from .simulation_config import SimulationRunConfig
from .spec import DEFAULT_SPEC, digest, read_json, sample, write_json


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a resumable Phase-3 dataset generation run."""

    spec: Path = DEFAULT_SPEC
    output_root: Path | None = None
    samples: int = 64
    num_envs: int = 4
    seed: int = 0
    start_sample: int = 0
    asset_mode: Literal["sampled", "reuse"] = "sampled"
    blender_render_samples: int = 8
    blender_threads: int = 12
    backend: Literal["gpu", "cpu"] = "gpu"
    coupler: Literal["legacy", "sap"] = "legacy"
    duration_s: float = 1.0
    relax_s: float = 0.0125
    simulation_hz: int = 1920
    record_hz: int = 30
    render_preview: bool = False
    force_retet: bool = False
    damping_scale: float = 2.5
    coupling_softness_m: float = 0.015
    coupling_friction_scale: float = 1.0
    use_sampled_restitution: Annotated[
        bool, tyro.conf.arg(name="sampled-restitution")
    ] = True
    resume: bool = False
    overwrite: bool = False

    @property
    def layout(self) -> SceneLayout:
        scene_id = read_json(self.spec)["scene_id"]
        return resolve_scene_layout(scene_id, self.output_root)

    @property
    def blender_dir(self) -> Path:
        return self.layout.blender

    @property
    def batches_dir(self) -> Path:
        return self.layout.batches


def _run_blender_variant(
    config: DatasetConfig,
    batch_seed: int,
) -> Path:
    blender_dir = config.layout.blender_variant(batch_seed)
    manifest_path = blender_dir / "scene_manifest.json"
    spec_path = blender_dir / "scene_spec.json"
    expected_spec = sample(read_json(config.spec), batch_seed)
    expected_spec["$schema"] = "scene.schema.json"
    required_artifacts = [manifest_path, spec_path, blender_dir / "scene.blend"]
    if config.render_preview:
        required_artifacts.append(blender_dir / "preview.png")
    if all(path.is_file() for path in required_artifacts) and digest(
        read_json(spec_path)
    ) == digest(expected_spec):
        return blender_dir
    command = [
        sys.executable,
        "-m",
        "video2scene.build",
        "--spec",
        str(config.spec),
        "--output-root",
        str(config.layout.root),
        "--seed",
        str(batch_seed),
        "--samples",
        str(config.blender_render_samples),
        "--threads",
        str(config.blender_threads),
    ]
    if not config.render_preview:
        command.append("--handoff-only")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    actual = read_json(spec_path)
    if digest(actual) != digest(expected_spec):
        raise RuntimeError(
            "Blender variant does not match the planned batch specification"
        )
    return blender_dir


def _dataset_identity(config: DatasetConfig) -> dict[str, Any]:
    identity = {
        "spec_sha256": sha256_file(config.spec),
        "samples": config.samples,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "start_sample": config.start_sample,
        "asset_mode": config.asset_mode,
        "backend": config.backend,
        "coupler": config.coupler,
        "duration_s": config.duration_s,
        "relax_s": config.relax_s,
        "simulation_hz": config.simulation_hz,
        "record_hz": config.record_hz,
        "render_preview": config.render_preview,
        "force_retet": config.force_retet,
        "damping_scale": config.damping_scale,
        "coupling_softness_m": config.coupling_softness_m,
        "coupling_friction_scale": config.coupling_friction_scale,
        "use_sampled_restitution": config.use_sampled_restitution,
    }
    if config.asset_mode == "reuse":
        identity["blender_manifest_sha256"] = sha256_file(
            config.blender_dir / "scene_manifest.json"
        )
    identity["sha256"] = digest(identity)
    return identity


def _collect_dataset_manifest(
    config: DatasetConfig,
    execution: dict[str, dict[str, str]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    batches: list[dict[str, Any]] = []
    parameter_values: dict[str, list[float]] = {name: [] for name in execution}
    completed = valid = invalid = 0
    failure_counts: dict[str, int] = {}
    for batch_dir in sorted(
        config.batches_dir.glob("batch_[0-9][0-9][0-9][0-9][0-9][0-9]")
    ):
        if not batch_complete(batch_dir):
            continue
        batch_layout = BatchLayout(batch_dir)
        run_manifest = read_json(batch_layout.sim / "run_manifest.json")
        batches.append(
            {
                "batch_index": run_manifest["batch_index"],
                "path": str(batch_dir.relative_to(config.batches_dir)),
                "sample_ids": run_manifest["sample_ids"],
                "sample_count": run_manifest["sample_count"],
                "valid_count": run_manifest["validation"]["valid_count"],
                "invalid_count": run_manifest["validation"]["invalid_count"],
                "environment_steps_per_second": run_manifest["timing"][
                    "environment_steps_per_second"
                ],
            }
        )
        completed += int(run_manifest["sample_count"])
        valid += int(run_manifest["validation"]["valid_count"])
        invalid += int(run_manifest["validation"]["invalid_count"])
        validation = read_json(batch_layout.sim / "validation.json")
        for sample_report in validation["samples"]:
            for check_name, passed in sample_report["checks"].items():
                if not passed:
                    failure_counts[check_name] = failure_counts.get(check_name, 0) + 1
        with (batch_layout.sim / "parameters.jsonl").open() as stream:
            for line in stream:
                item = json.loads(line)
                for name, value in item["parameters"].items():
                    parameter_values[name].append(float(value))
    coverage = {
        name: {
            "lane": execution[name]["lane"],
            "count": len(values),
            "unique": len(set(values)),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
        for name, values in parameter_values.items()
    }
    serialized_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    serialized_config["output_root"] = str(config.layout.root)
    return {
        "schema_version": "video2scene.domain-randomized-dataset.v1",
        "phase": 3,
        "scene_id": read_json(config.spec)["scene_id"],
        "config": serialized_config,
        "run_identity": identity,
        "artifact_layout": {
            "scene_root": "..",
            "batches": ".",
            "blender_variants": "../blender/variants",
            "batch_simulation": "batch_XXXXXX/sim",
        },
        "parameter_execution": execution,
        "coverage": coverage,
        "batches": batches,
        "planned_samples": config.samples,
        "completed_samples": completed,
        "valid_samples": valid,
        "invalid_samples": invalid,
        "failure_counts": dict(sorted(failure_counts.items())),
        "complete": completed == config.samples,
        "physics_validated": completed == config.samples and invalid == 0,
        "training_ready": False,
        "training_blockers": [
            "The sampled ranges remain engineering priors from monocular reconstruction.",
            "Preview review and downstream task-specific acceptance are still required.",
            "Filter samples whose per-environment simulation_valid flag is false.",
        ],
    }


def _generate_dataset(
    config: DatasetConfig,
    base_spec: dict[str, Any],
    identity: dict[str, Any],
    execution: dict[str, dict[str, str]],
) -> dict[str, Any]:
    write_json(
        config.batches_dir / "dataset_manifest.json",
        _collect_dataset_manifest(config, execution, identity),
    )

    batch_count = math.ceil(config.samples / config.num_envs)
    for batch_index in range(batch_count):
        sample_start = config.start_sample + batch_index * config.num_envs
        batch_size = min(
            config.num_envs, config.samples - batch_index * config.num_envs
        )
        batch_layout = config.layout.batch(batch_index)
        batch_dir = batch_layout.root
        expected_sample_ids = list(range(sample_start, sample_start + batch_size))
        if config.resume and batch_complete(
            batch_dir,
            expected_identity_sha256=identity["sha256"],
            expected_sample_ids=expected_sample_ids,
        ):
            logger.info("[Phase 3] Skip complete batch {:06d}", batch_index)
            continue
        batch_dir.mkdir(parents=True, exist_ok=True)
        remove_legacy_batch_artifacts(batch_dir)
        batch_layout.incomplete_marker.write_text("generation in progress\n")

        if config.asset_mode == "sampled":
            batch_seed = derive_seed(config.seed, "batch", batch_index)
            blender_dir = _run_blender_variant(config, batch_seed)
            batch_spec = read_json(blender_dir / "scene_spec.json")
            plan = plan_batch(
                base_spec,
                root_seed=config.seed,
                batch_index=batch_index,
                num_envs=batch_size,
                sample_start=sample_start,
                batch_spec=batch_spec,
                batch_seed=batch_seed,
            )
        else:
            blender_dir = config.blender_dir
            batch_spec = read_json(blender_dir / "scene_spec.json")
            plan = plan_batch(
                base_spec,
                root_seed=config.seed,
                batch_index=batch_index,
                num_envs=batch_size,
                sample_start=sample_start,
                batch_spec=batch_spec,
                batch_seed=None,
            )

        anchor_manifest = read_json(blender_dir / "scene_manifest.json")
        nominal_exposure = float(base_spec["lighting"]["exposure_ev"])
        batch_exposure = float(plan.batch_spec["lighting"]["exposure_ev"])
        exposure_scale = 2.0 ** (batch_exposure - nominal_exposure)
        render_defaults = base_spec["lighting"].get("genesis", {})
        simulation_config = SimulationRunConfig(
            blender_dir=blender_dir.resolve(),
            source_frames_dir=config.layout.frames.resolve(),
            sim_dir=batch_layout.sim.resolve(),
            backend=config.backend,
            coupler=config.coupler,
            duration_s=config.duration_s,
            relax_s=config.relax_s,
            fps=config.record_hz,
            simulation_hz=config.simulation_hz,
            num_envs=batch_size,
            seed=int(
                (plan.batch_seed or derive_seed(config.seed, "reuse", batch_index))
                % (2**31 - 1)
            ),
            render=config.render_preview,
            overwrite=True,
            force_retet=config.force_retet,
            damping_scale=config.damping_scale,
            coupling_softness_m=config.coupling_softness_m,
            coupling_friction_scale=config.coupling_friction_scale,
            ambient_light=float(render_defaults.get("ambient_light", 0.85))
            * exposure_scale,
            key_light_scale=float(render_defaults.get("key_light_scale", 0.35))
            * exposure_scale,
            fill_light_scale=float(render_defaults.get("fill_light_scale", 0.45))
            * exposure_scale,
            render_shadows=bool(render_defaults.get("render_shadows", False)),
            surface_color_space=render_defaults.get("surface_color_space", "linear"),
            approximate_area_lights=bool(
                render_defaults.get("approximate_area_lights", True)
            ),
        )
        logger.info(
            "[Phase 3] Batch {}/{}: {} parallel environments, samples {}..{}",
            batch_index + 1,
            batch_count,
            batch_size,
            sample_start,
            sample_start + batch_size - 1,
        )
        result = run_parallel_batch(
            simulation_config,
            plan,
            anchor_manifest,
            use_sampled_restitution=config.use_sampled_restitution,
        )
        batch_layout.success_marker.write_text(
            json.dumps(
                {
                    "batch_index": batch_index,
                    "sample_count": batch_size,
                    "valid_count": result.validation["valid_count"],
                    "run_identity_sha256": identity["sha256"],
                    "run_manifest_sha256": sha256_file(
                        batch_layout.sim / "run_manifest.json"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        if batch_layout.incomplete_marker.exists():
            batch_layout.incomplete_marker.unlink()
        dataset_manifest = _collect_dataset_manifest(config, execution, identity)
        write_json(config.batches_dir / "dataset_manifest.json", dataset_manifest)
        logger.info(
            "[Phase 3] Completed batch {:06d}: {}/{} simulation-valid",
            batch_index,
            result.validation["valid_count"],
            batch_size,
        )

    dataset_manifest = _collect_dataset_manifest(config, execution, identity)
    write_json(config.batches_dir / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest


def run_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Generate or resume a complete collection of parallel Genesis batches."""
    if config.samples < 1 or config.num_envs < 1:
        raise ValueError("samples and num_envs must be positive")
    if config.start_sample < 0:
        raise ValueError("start_sample must be non-negative")
    if config.asset_mode not in {"sampled", "reuse"}:
        raise ValueError("asset_mode must be sampled or reuse")
    if config.resume and config.overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if config.simulation_hz % config.record_hz:
        raise ValueError("simulation_hz must be divisible by record_hz")

    config.layout.prepare()
    base_spec = read_json(config.spec)
    identity = _dataset_identity(config)
    existing_manifest_path = config.batches_dir / "dataset_manifest.json"
    if config.resume and existing_manifest_path.is_file():
        existing_identity = read_json(existing_manifest_path).get("run_identity", {})
        if existing_identity.get("sha256") != identity["sha256"]:
            raise ValueError(
                "Resume configuration differs from the existing dataset manifest"
            )
    execution = parameter_execution(base_spec)

    if config.batches_dir.exists() and config.overwrite:
        remove_path(config.batches_dir)
    config.batches_dir.mkdir(parents=True, exist_ok=True)
    if any(config.batches_dir.iterdir()) and not (config.resume or config.overwrite):
        raise FileExistsError(
            f"Output is not empty: {config.batches_dir}. Pass --resume or --overwrite."
        )
    with stage_log(
        config.batches_dir / "generate.log", mode="a" if config.resume else "w"
    ):
        return _generate_dataset(config, base_spec, identity, execution)


def main() -> None:
    config = tyro.cli(DatasetConfig)
    config = replace(
        config,
        spec=config.spec.resolve(),
        output_root=config.output_root.resolve() if config.output_root else None,
    )
    try:
        result = run_dataset(config)
    except Exception:
        logger.exception("[Phase 3] Generation failed")
        raise
    logger.info(
        "[Phase 3] Dataset complete={} completed={} valid={} invalid={} output={}",
        result["complete"],
        result["completed_samples"],
        result["valid_samples"],
        result["invalid_samples"],
        config.batches_dir,
    )


if __name__ == "__main__":
    main()
