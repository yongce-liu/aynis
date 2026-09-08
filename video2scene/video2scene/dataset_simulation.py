"""Execution and validation of one parallel Phase-3 simulation batch."""

from __future__ import annotations

import json
import shutil
import time
from types import SimpleNamespace
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .filesystem import remove_path
from .domain_randomization import (
    BatchPlan,
    resolve_runtime_entities,
    serializable_batch_plan,
)
from .hashing import sha256_file
from .logging_utils import stage_log
from .simulation_artifacts import (
    encode_video,
    validate_simulation_inputs,
    write_comparison,
    write_contact_sheet,
)
from .simulation_config import SimulationRunConfig, validate_simulation_options
from .simulate import _validate_rolling_transition, _validate_soft_impact
from .simulation_geometry import rigid_bottom_z
from .simulation_runtime import (
    as_numpy,
    build_runtime,
    render_frame,
    sync_visual_followers,
)
from .spec import bounds, digest, write_json


@dataclass
class BatchRuntimeResult:
    validation: dict[str, Any]
    run_manifest: dict[str, Any]


def _normalize_batch(value: Any, num_envs: int) -> np.ndarray:
    array = as_numpy(value)
    if num_envs == 1 and array.ndim >= 1 and array.shape[0] != 1:
        return array[None, ...]
    return array


def _set_rigid_batch(
    entity: Any, positions: np.ndarray, quaternions: np.ndarray, velocities: np.ndarray
) -> None:
    if len(positions) == 1:
        entity.set_pos(positions[0], zero_velocity=True)
        entity.set_quat(quaternions[0], zero_velocity=True)
        entity.set_dofs_velocity(velocities[0])
    else:
        entity.set_pos(positions, zero_velocity=True)
        entity.set_quat(quaternions, zero_velocity=True)
        entity.set_dofs_velocity(velocities)


def _soft_batch_state(entity: Any, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    state = entity.get_state()
    positions = _normalize_batch(state.pos, num_envs).astype(np.float64)
    velocities = _normalize_batch(state.vel, num_envs).astype(np.float64)
    return positions, velocities


def _relax_and_initialize_batch(
    runtime: Any, config: SimulationRunConfig
) -> dict[str, Any]:
    env_entities = runtime.env_entities
    if env_entities is None:
        raise ValueError(
            "Parallel initialization requires per-environment entity descriptions"
        )
    num_envs = len(env_entities)
    dynamic_names = [
        name
        for name, entity in runtime.manifest["resolved_entities"].items()
        if entity["physics_type"] == "rigid"
    ]
    if not runtime.deformable:
        for name in dynamic_names:
            positions = np.asarray(
                [entities[name]["position_m"] for entities in env_entities],
                dtype=np.float64,
            )
            quaternions = np.asarray(
                [entities[name]["quaternion_wxyz"] for entities in env_entities],
                dtype=np.float64,
            )
            velocities = np.asarray(
                [
                    [
                        *entities[name]["initial_state"]["linear_velocity_m_s"],
                        *entities[name]["initial_state"]["angular_velocity_rad_s"],
                    ]
                    for entities in env_entities
                ],
                dtype=np.float64,
            )
            _set_rigid_batch(runtime.physical[name], positions, quaternions, velocities)
        sync_visual_followers(runtime)
        return {
            "steps": 0,
            "duration_s": 0.0,
            "method": "exact batched rigid t0 initialization; no deformable relaxation required",
            "deformables": {},
        }

    for rigid_index, name in enumerate(dynamic_names):
        positions = np.asarray(
            [entities[name]["position_m"] for entities in env_entities],
            dtype=np.float64,
        )
        positions[:, 2] = 1.8 + 0.25 * rigid_index
        quaternions = np.asarray(
            [entities[name]["quaternion_wxyz"] for entities in env_entities],
            dtype=np.float64,
        )
        velocities = np.zeros((num_envs, 6), dtype=np.float64)
        _set_rigid_batch(runtime.physical[name], positions, quaternions, velocities)

    relax_steps = max(0, round(config.relax_s / config.dt))
    for _ in range(relax_steps):
        runtime.scene.step(update_visualizer=False)

    relaxed: dict[str, Any] = {}
    for name, entity in runtime.deformable.items():
        positions, velocities = _soft_batch_state(entity, num_envs)
        indices = runtime.surface_indices[name]
        surface = positions[:, indices]
        source_surface = runtime.source_surfaces[name]
        displacement = np.linalg.norm(surface - source_surface[None, :, :], axis=2)
        relaxed[name] = {
            "surface_rms_displacement_m": np.sqrt(
                np.mean(displacement**2, axis=1)
            ).tolist(),
            "surface_p95_displacement_m": np.quantile(
                displacement, 0.95, axis=1
            ).tolist(),
            "surface_max_displacement_m": displacement.max(axis=1).tolist(),
            "pre_reset_max_speed_m_s": np.linalg.norm(velocities, axis=2)
            .max(axis=1)
            .tolist(),
            "velocity_reset_at_t0": [True] * num_envs,
            "bounds_m": np.stack(
                (surface.min(axis=1), surface.max(axis=1)), axis=1
            ).tolist(),
        }
        zeros = np.zeros_like(velocities)
        entity.set_velocity(zeros[0] if num_envs == 1 else zeros)

    for name in dynamic_names:
        positions = np.asarray(
            [entities[name]["position_m"] for entities in env_entities],
            dtype=np.float64,
        )
        quaternions = np.asarray(
            [entities[name]["quaternion_wxyz"] for entities in env_entities],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [
                [
                    *entities[name]["initial_state"]["linear_velocity_m_s"],
                    *entities[name]["initial_state"]["angular_velocity_rad_s"],
                ]
                for entities in env_entities
            ],
            dtype=np.float64,
        )
        _set_rigid_batch(runtime.physical[name], positions, quaternions, velocities)
    sync_visual_followers(runtime)
    return {
        "steps": relax_steps,
        "duration_s": relax_steps * config.dt,
        "method": "batched gravity seating followed by zero FEM velocity and exact per-environment rigid t0 restore",
        "deformables": relaxed,
    }


def _rigid_bottoms(
    entities: list[dict[str, dict[str, Any]]],
    name: str,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            rigid_bottom_z(items[name], positions[index], quaternions[index])
            for index, items in enumerate(entities)
        ],
        dtype=np.float64,
    )


def _make_patch_indices(
    runtime: Any, env_entities: list[dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    num_envs = len(env_entities)
    patches: dict[str, dict[str, Any]] = {}
    for rule in runtime.manifest.get("physics_validation", {}).get("checks", []):
        if rule["type"] != "soft_impact":
            continue
        body_name = rule["entity"]
        target_name = rule["target"]
        positions, _ = _soft_batch_state(runtime.deformable[target_name], num_envs)
        surfaces = positions[:, runtime.surface_indices[target_name]]
        selected_per_env: list[np.ndarray] = []
        baseline: list[float] = []
        radii: list[float] = []
        for env_index, items in enumerate(env_entities):
            body = items[body_name]
            xy = np.asarray(body["position_m"][:2], dtype=np.float64)
            if body["collision"]["representation"] == "sphere":
                radius = max(0.065, 1.7 * float(body["collision"]["radius_m"]))
            else:
                radius = max(0.055, 1.8 * max(body["collision"]["dimensions_m"][:2]))
            surface = surfaces[env_index]
            top_cut = float(surface[:, 2].max() - 0.04)
            selected = np.flatnonzero(
                (np.linalg.norm(surface[:, :2] - xy, axis=1) <= radius)
                & (surface[:, 2] >= top_cut)
            )
            if len(selected) < 16:
                top = np.flatnonzero(surface[:, 2] >= np.quantile(surface[:, 2], 0.7))
                selected = top[
                    np.argsort(np.linalg.norm(surface[top, :2] - xy, axis=1))[:64]
                ]
            selected_per_env.append(selected)
            baseline.append(float(np.quantile(surface[selected, 2], 0.9)))
            radii.append(radius)
        patches[rule["id"]] = {
            "entity": body_name,
            "target": target_name,
            "surface_indices": selected_per_env,
            "baseline_top_z_m": np.asarray(baseline, dtype=np.float64),
            "radius_m": np.asarray(radii, dtype=np.float64),
        }
    return patches


def _capture_batch_state(
    runtime: Any,
    time_s: float,
    patches: dict[str, dict[str, Any]],
    env_entities: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    num_envs = len(env_entities)
    record: dict[str, Any] = {
        "time_s": float(time_s),
        "rigid": {},
        "deformable": {},
        "relations": {},
    }
    soft_cache: dict[str, np.ndarray] = {}

    for name, entity in runtime.deformable.items():
        positions, velocities = _soft_batch_state(entity, num_envs)
        surface = positions[:, runtime.surface_indices[name]]
        surface_velocities = velocities[:, runtime.surface_indices[name]]
        soft_cache[name] = surface
        tet = positions[:, runtime.tetrahedra[name], :]
        signed_six_volume = np.einsum(
            "bei,bei->be",
            np.cross(tet[:, :, 1] - tet[:, :, 0], tet[:, :, 2] - tet[:, :, 0]),
            tet[:, :, 3] - tet[:, :, 0],
        )
        oriented_volume = (
            signed_six_volume * runtime.reference_tet_signs[name][None, :] / 6.0
        )
        record["deformable"][name] = {
            "bounds_m": np.stack((surface.min(axis=1), surface.max(axis=1)), axis=1),
            "center_m": surface.mean(axis=1),
            "max_surface_speed_m_s": np.linalg.norm(surface_velocities, axis=2).max(
                axis=1
            ),
            "minimum_oriented_tet_volume_m3": oriented_volume.min(axis=1),
            "inverted_tetrahedra": np.count_nonzero(oriented_volume <= 0.0, axis=1),
        }

    for name, entity in runtime.manifest["resolved_entities"].items():
        if entity["physics_type"] != "rigid":
            continue
        physical = runtime.physical[name]
        positions = _normalize_batch(physical.get_pos(), num_envs).astype(np.float64)
        quaternions = _normalize_batch(physical.get_quat(), num_envs).astype(np.float64)
        velocities = _normalize_batch(physical.get_dofs_velocity(), num_envs).astype(
            np.float64
        )
        record["rigid"][name] = {
            "position_m": positions,
            "quaternion_wxyz": quaternions,
            "linear_velocity_m_s": velocities[:, :3],
            "angular_velocity_rad_s": velocities[:, 3:6],
            "bottom_z_m": _rigid_bottoms(env_entities, name, positions, quaternions),
        }

    for rule_id, patch in patches.items():
        body_name = patch["entity"]
        surfaces = soft_cache[patch["target"]]
        patch_top = np.asarray(
            [
                np.quantile(surfaces[index, selected, 2], 0.9)
                for index, selected in enumerate(patch["surface_indices"])
            ],
            dtype=np.float64,
        )
        record["relations"][rule_id] = {
            "entity": body_name,
            "target": patch["target"],
            "patch_top_z_m": patch_top,
            "patch_indentation_m": patch["baseline_top_z_m"] - patch_top,
            "vertical_gap_m": record["rigid"][body_name]["bottom_z_m"] - patch_top,
        }
    return record


def _stack_records(
    records: list[dict[str, Any]], plan: BatchPlan
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "time_s": np.asarray(
            [record["time_s"] for record in records], dtype=np.float64
        ),
        "sample_id": np.asarray(
            [item.sample_id for item in plan.samples], dtype=np.int64
        ),
        "sample_seed": np.asarray(
            [item.sample_seed for item in plan.samples], dtype=np.uint64
        ),
    }
    for name in records[0]["rigid"]:
        for field in (
            "position_m",
            "quaternion_wxyz",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
            "bottom_z_m",
        ):
            arrays[f"rigid/{name}/{field}"] = np.stack(
                [record["rigid"][name][field] for record in records], axis=0
            )
    for name in records[0]["deformable"]:
        for field in (
            "bounds_m",
            "center_m",
            "max_surface_speed_m_s",
            "minimum_oriented_tet_volume_m3",
            "inverted_tetrahedra",
        ):
            arrays[f"deformable/{name}/{field}"] = np.stack(
                [record["deformable"][name][field] for record in records], axis=0
            )
    for name in records[0]["relations"]:
        for field in ("patch_top_z_m", "patch_indentation_m", "vertical_gap_m"):
            arrays[f"relation/{name}/{field}"] = np.stack(
                [record["relations"][name][field] for record in records], axis=0
            )
    parameter_names = sorted(plan.parameter_execution)
    arrays["parameter_names"] = np.asarray(parameter_names)
    arrays["parameter_values"] = np.asarray(
        [
            [
                item.spec["sampling"]["parameter_values"][name]
                for name in parameter_names
            ]
            for item in plan.samples
        ],
        dtype=np.float64,
    )
    return arrays


def _quat_error_batch(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    dots = np.clip(np.abs(np.sum(actual * expected, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def _support_top(spec: dict[str, Any], cushion_name: str) -> float:
    support_name = spec["entities"][cushion_name]["physics"]["deformable"][
        "boundary_conditions"
    ]["support_entity"]
    return float(bounds(spec["entities"][support_name])[1][2])


def _records_for_environment(
    arrays: dict[str, np.ndarray], env_index: int
) -> list[dict[str, Any]]:
    rigid_names = sorted(
        {
            key.split("/")[1]
            for key in arrays
            if key.startswith("rigid/") and key.endswith("/position_m")
        }
    )
    deformable_names = sorted(
        {
            key.split("/")[1]
            for key in arrays
            if key.startswith("deformable/") and key.endswith("/bounds_m")
        }
    )
    relation_ids = sorted(
        {
            key.split("/")[1]
            for key in arrays
            if key.startswith("relation/") and key.endswith("/vertical_gap_m")
        }
    )
    records: list[dict[str, Any]] = []
    for time_index, time_s in enumerate(arrays["time_s"]):
        record: dict[str, Any] = {
            "time_s": float(time_s),
            "rigid": {},
            "deformable": {},
            "relations": {},
        }
        for name in rigid_names:
            record["rigid"][name] = {
                field: arrays[f"rigid/{name}/{field}"][time_index, env_index].tolist()
                if arrays[f"rigid/{name}/{field}"][time_index, env_index].ndim
                else float(arrays[f"rigid/{name}/{field}"][time_index, env_index])
                for field in (
                    "position_m",
                    "quaternion_wxyz",
                    "linear_velocity_m_s",
                    "angular_velocity_rad_s",
                    "bottom_z_m",
                )
            }
        for name in deformable_names:
            record["deformable"][name] = {
                field: arrays[f"deformable/{name}/{field}"][
                    time_index, env_index
                ].tolist()
                if arrays[f"deformable/{name}/{field}"][time_index, env_index].ndim
                else float(arrays[f"deformable/{name}/{field}"][time_index, env_index])
                for field in (
                    "bounds_m",
                    "center_m",
                    "max_surface_speed_m_s",
                    "minimum_oriented_tet_volume_m3",
                    "inverted_tetrahedra",
                )
            }
        for rule_id in relation_ids:
            record["relations"][rule_id] = {
                field: float(
                    arrays[f"relation/{rule_id}/{field}"][time_index, env_index]
                )
                for field in (
                    "patch_top_z_m",
                    "patch_indentation_m",
                    "vertical_gap_m",
                )
            }
        records.append(record)
    return records


def _rule_outcome_labels(
    rules: list[dict[str, Any]], checks: dict[str, bool], metrics: dict[str, Any]
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for rule in rules:
        rule_id = rule["id"]
        family = (
            "soft_impacts" if rule["type"] == "soft_impact" else "rolling_transitions"
        )
        labels[rule_id] = {
            "type": rule["type"],
            "valid": all(
                passed
                for name, passed in checks.items()
                if name.startswith(rule_id + ".")
            ),
            "metrics": metrics.get(family, {}).get(rule_id, {}),
        }
    return labels


def _validate_batch(
    config: SimulationRunConfig,
    plan: BatchPlan,
    runtime: Any,
    relaxation: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    num_envs = len(plan.samples)
    sample_reports: list[dict[str, Any]] = []
    dynamic_names = tuple(
        dict.fromkeys(
            key.split("/")[1]
            for key in arrays
            if key.startswith("rigid/") and key.endswith("/position_m")
        )
    )

    for env_index, planned in enumerate(plan.samples):
        spec = planned.spec
        rules = list(spec.get("physics_validation", {}).get("checks", []))
        checks: dict[str, bool] = {
            "validation.contract_present": bool(rules),
            "policy.pure_simulation_after_t0": bool(
                runtime.plan["policy"]["initial_state_only_from_video"]
            )
            and not runtime.plan["policy"]["source_video_access_during_rollout"]
            and not runtime.plan["policy"]["future_pose_replay"],
        }
        physics_keys = list(checks)
        metrics: dict[str, Any] = {
            "initial_state_error": {},
            "deformables": {},
        }
        numeric: list[np.ndarray] = []

        for name in dynamic_names:
            positions = arrays[f"rigid/{name}/position_m"][:, env_index]
            quaternions = arrays[f"rigid/{name}/quaternion_wxyz"][:, env_index]
            velocities = arrays[f"rigid/{name}/linear_velocity_m_s"][:, env_index]
            angular = arrays[f"rigid/{name}/angular_velocity_rad_s"][:, env_index]
            expected = runtime.env_entities[env_index][name]
            position_error = float(
                np.linalg.norm(positions[0] - np.asarray(expected["position_m"]))
            )
            quaternion_error = float(
                _quat_error_batch(
                    quaternions[:1],
                    np.asarray(expected["quaternion_wxyz"], dtype=float)[None],
                )[0]
            )
            linear_velocity_error = float(
                np.linalg.norm(
                    velocities[0]
                    - np.asarray(
                        expected["initial_state"]["linear_velocity_m_s"], dtype=float
                    )
                )
            )
            angular_velocity_error = float(
                np.linalg.norm(
                    angular[0]
                    - np.asarray(
                        expected["initial_state"]["angular_velocity_rad_s"],
                        dtype=float,
                    )
                )
            )
            checks[f"{name}.initial_position"] = position_error <= 1e-5
            checks[f"{name}.initial_orientation"] = quaternion_error <= 1e-5
            checks[f"{name}.initial_linear_velocity"] = linear_velocity_error <= 1e-5
            checks[f"{name}.initial_angular_velocity"] = angular_velocity_error <= 1e-5
            physics_keys.extend(
                (
                    f"{name}.initial_position",
                    f"{name}.initial_orientation",
                    f"{name}.initial_linear_velocity",
                    f"{name}.initial_angular_velocity",
                )
            )
            metrics["initial_state_error"][name] = {
                "position_m": position_error,
                "rotation_rad": quaternion_error,
                "linear_velocity_m_s": linear_velocity_error,
                "angular_velocity_rad_s": angular_velocity_error,
            }
            numeric.extend((positions, quaternions, velocities, angular))

        for name in runtime.deformable:
            bounds_over_time = arrays[f"deformable/{name}/bounds_m"][:, env_index]
            speeds = arrays[f"deformable/{name}/max_surface_speed_m_s"][:, env_index]
            inversions = arrays[f"deformable/{name}/inverted_tetrahedra"][:, env_index]
            minimum_volumes = arrays[
                f"deformable/{name}/minimum_oriented_tet_volume_m3"
            ][:, env_index]
            p95 = float(
                relaxation["deformables"][name]["surface_p95_displacement_m"][env_index]
            )
            inversion_fraction = float(
                inversions.max(initial=0) / max(1, runtime.deformable[name].n_elements)
            )
            support_top = _support_top(spec, name)
            minimum_surface_z = float(bounds_over_time[:, 0, 2].min())
            checks[f"{name}.initial_surface_alignment"] = p95 <= 0.01
            checks[f"{name}.t0_velocity_reset"] = True
            checks[f"{name}.support_penetration"] = (
                minimum_surface_z >= support_top - 0.01
            )
            checks[f"{name}.tetrahedral_stability"] = inversion_fraction <= 0.02
            physics_keys.extend(
                (
                    f"{name}.initial_surface_alignment",
                    f"{name}.t0_velocity_reset",
                    f"{name}.support_penetration",
                    f"{name}.tetrahedral_stability",
                )
            )
            metrics["deformables"][name] = {
                "surface_p95_displacement_m": p95,
                "minimum_surface_z_m": minimum_surface_z,
                "support_top_z_m": support_top,
                "maximum_inverted_tetrahedra": int(inversions.max(initial=0)),
                "maximum_inverted_fraction": inversion_fraction,
                "minimum_oriented_tet_volume_m3": float(minimum_volumes.min()),
            }
            numeric.extend((bounds_over_time, speeds, minimum_volumes))

        validation_manifest = dict(runtime.manifest)
        validation_manifest.update(
            source=spec["source"],
            semantics=spec["semantics"],
            physics_validation=spec["physics_validation"],
            resolved_entities=runtime.env_entities[env_index],
        )
        validation_runtime = SimpleNamespace(manifest=validation_manifest)
        records = _records_for_environment(arrays, env_index)
        for rule in rules:
            if rule["type"] == "soft_impact":
                _validate_soft_impact(
                    validation_runtime,
                    config,
                    records,
                    rule,
                    checks,
                    physics_keys,
                    metrics,
                )
            elif rule["type"] == "rolling_transition":
                _validate_rolling_transition(
                    validation_runtime,
                    records,
                    rule,
                    checks,
                    physics_keys,
                    metrics,
                )
            else:
                raise ValueError(
                    f"Unsupported Phase-3 validation type: {rule['type']!r}"
                )

        finite = all(np.isfinite(value).all() for value in numeric)
        bounded = all(np.abs(value).max(initial=0.0) < 50.0 for value in numeric)
        checks["states.finite"] = bool(finite)
        checks["states.bounded"] = bool(bounded)
        physics_keys.extend(("states.finite", "states.bounded"))
        simulation_valid = bool(physics_keys) and all(
            checks[key] for key in physics_keys
        )
        sample_reports.append(
            {
                "sample_id": planned.sample_id,
                "env_index": env_index,
                "sample_seed": planned.sample_seed,
                "simulation_valid": simulation_valid,
                "checks": checks,
                "metrics": metrics,
                "outcome_labels": _rule_outcome_labels(rules, checks, metrics),
            }
        )

    valid_count = sum(item["simulation_valid"] for item in sample_reports)
    blockers = list(
        plan.batch_spec.get("physics_validation", {}).get(
            "training_blockers",
            [
                "Parameter intervals are engineering priors inferred from one monocular video, not identified distributions.",
            ],
        )
    )
    blockers.extend(
        (
            "Review preview batches and dataset-wide failure/coverage statistics before training use.",
            "Any sample with simulation_valid=false must be filtered or regenerated.",
        )
    )
    return {
        "schema_version": "video2scene.domain-randomized-validation.v1",
        "passed": valid_count == num_envs,
        "physics_validated": valid_count == num_envs,
        "sample_count": num_envs,
        "valid_count": valid_count,
        "invalid_count": num_envs - valid_count,
        "samples": sample_reports,
        "training_ready": False,
        "training_blockers": list(dict.fromkeys(blockers)),
    }


def _write_parameters(
    path: Path,
    plan: BatchPlan,
    validation: dict[str, Any],
    env_entities: list[dict[str, dict[str, Any]]],
) -> None:
    reports = {item["sample_id"]: item for item in validation["samples"]}
    with path.open("w") as stream:
        for planned in plan.samples:
            payload = {
                "sample_id": planned.sample_id,
                "env_index": planned.env_index,
                "sample_seed": planned.sample_seed,
                "attempt": planned.attempt,
                "spec_sha256": digest(planned.spec),
                "parameters": planned.spec["sampling"]["parameter_values"],
                "parameter_lanes": planned.spec["sampling"]["parameter_lanes"],
                "resolved_rigid": {
                    name: {
                        "mass_kg": entity.get("mass_kg"),
                        "collision": entity["collision"],
                        "position_m": entity["position_m"],
                        "quaternion_wxyz": entity["quaternion_wxyz"],
                        "initial_state": entity["initial_state"],
                        "coupling_friction": entity["physics_parameters"].get(
                            "coupling_friction"
                        ),
                    }
                    for name, entity in env_entities[planned.env_index].items()
                    if entity["physics_type"] == "rigid"
                },
                "simulation_valid": reports[planned.sample_id]["simulation_valid"],
                "outcome_labels": reports[planned.sample_id]["outcome_labels"],
            }
            stream.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _render_review(
    batch_output: Path, plan: BatchPlan, validation: dict[str, Any]
) -> None:
    names = ("preview.png", "comparison.jpg", "contact_sheet.jpg", "simulation.mp4")
    write_json(
        batch_output / "render_review.json",
        {
            "schema_version": "video2scene.domain-randomized-render-review.v1",
            "status": "pending_review_of_preview_environment",
            "preview_environment": {
                "env_index": 0,
                "sample_id": plan.samples[0].sample_id,
                "sample_seed": plan.samples[0].sample_seed,
            },
            "scope": "One representative environment from a parallel Phase-3 batch",
            "automated_validation": {
                "batch_passed": validation["passed"],
                "preview_sample_valid": validation["samples"][0]["simulation_valid"],
            },
            "artifact_sha256": {
                name: sha256_file(batch_output / name)
                for name in names
                if (batch_output / name).is_file()
            },
        },
    )


def _run_parallel_batch_prepared(
    config: SimulationRunConfig,
    plan: BatchPlan,
    anchor_manifest: dict[str, Any],
    frames_dir: Path,
    *,
    use_sampled_restitution: bool = True,
) -> BatchRuntimeResult:
    runtime = None
    started = time.perf_counter()
    simulation_started = None
    try:
        effective_specs = [item.spec for item in plan.samples]
        env_entities = resolve_runtime_entities(anchor_manifest, effective_specs)
        runtime = build_runtime(
            config,
            env_entities=env_entities,
            use_sampled_coupling_restitution=use_sampled_restitution,
        )
        runtime.plan["phase3"] = {
            "parallel_environments": len(plan.samples),
            "heterogeneous_rigid_geometry": True,
            "batch_shared_fem_geometry_and_material": True,
            "sampled_coupling_restitution": use_sampled_restitution,
            "parameter_execution": plan.parameter_execution,
        }
        write_json(config.sim_dir / "migration_plan.json", runtime.plan)
        write_json(config.sim_dir / "batch_plan.json", serializable_batch_plan(plan))

        relaxation = _relax_and_initialize_batch(runtime, config)
        patches = _make_patch_indices(runtime, env_entities)
        records = [_capture_batch_state(runtime, 0.0, patches, env_entities)]
        frame_paths: list[Path] = []
        times = [0.0]
        if config.render:
            sync_visual_followers(runtime)
            frame = frames_dir / "frame_0000.png"
            render_frame(runtime, frame)
            shutil.copyfile(frame, config.sim_dir / "preview.png")
            frame_paths.append(frame)

        n_steps = round(config.duration_s / config.dt)
        record_every = config.simulation_hz // config.fps
        simulation_started = time.perf_counter()
        for step in range(1, n_steps + 1):
            runtime.scene.step(update_visualizer=False)
            if step % record_every == 0 or step == n_steps:
                time_s = step * config.dt
                records.append(
                    _capture_batch_state(runtime, time_s, patches, env_entities)
                )
                times.append(time_s)
                if config.render:
                    sync_visual_followers(runtime)
                    frame = frames_dir / f"frame_{len(frame_paths):04d}.png"
                    render_frame(runtime, frame)
                    frame_paths.append(frame)
        simulation_seconds = time.perf_counter() - simulation_started

        arrays = _stack_records(records, plan)
        np.savez_compressed(config.sim_dir / "trajectories.npz", **arrays)
        write_json(
            config.sim_dir / "trace_schema.json",
            {
                "schema_version": "video2scene.domain-randomized-trace.v1",
                "layout": "time-major arrays; environment axis is second for trajectories and first for parameters",
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in arrays.items()
                },
            },
        )

        validation = _validate_batch(config, plan, runtime, relaxation, arrays)
        write_json(config.sim_dir / "validation.json", validation)
        _write_parameters(
            config.sim_dir / "parameters.jsonl", plan, validation, env_entities
        )

        video_probe = None
        if config.render:
            video_probe = encode_video(
                frames_dir, config.sim_dir / "simulation.mp4", config.fps
            )
            write_contact_sheet(config, frame_paths, times)
            if (config.blender_dir / "preview.png").is_file():
                write_comparison(config)
            _render_review(config.sim_dir, plan, validation)

        artifact_names = [
            "migration_plan.json",
            "batch_plan.json",
            "parameters.jsonl",
            "trajectories.npz",
            "trace_schema.json",
            "validation.json",
        ]
        if config.render:
            artifact_names.extend(
                (
                    "preview.png",
                    "contact_sheet.jpg",
                    "simulation.mp4",
                    "render_review.json",
                )
            )
            if (config.sim_dir / "comparison.jpg").is_file():
                artifact_names.append("comparison.jpg")
        elapsed = time.perf_counter() - started
        run_manifest = {
            "schema_version": "video2scene.domain-randomized-run.v1",
            "phase": 3,
            "scene_id": anchor_manifest["scene_id"],
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "batch_index": plan.batch_index,
            "batch_seed": plan.batch_seed,
            "sample_ids": [item.sample_id for item in plan.samples],
            "sample_count": len(plan.samples),
            "parallel_environments": len(plan.samples),
            "genesis_version": version("genesis-world"),
            "timing": {
                "build_seconds": runtime.build_seconds,
                "simulation_seconds": simulation_seconds,
                "total_seconds": elapsed,
                "simulated_steps": n_steps,
                "environment_steps": n_steps * len(plan.samples),
                "environment_steps_per_second": n_steps
                * len(plan.samples)
                / max(simulation_seconds, 1e-9),
            },
            "validation": {
                "passed": validation["passed"],
                "valid_count": validation["valid_count"],
                "invalid_count": validation["invalid_count"],
            },
            "video_probe": video_probe,
            "artifacts": {
                name: {"path": name, "sha256": sha256_file(config.sim_dir / name)}
                for name in artifact_names
                if (config.sim_dir / name).is_file()
            },
            "training_ready": False,
        }
        write_json(config.sim_dir / "run_manifest.json", run_manifest)
        return BatchRuntimeResult(validation=validation, run_manifest=run_manifest)
    finally:
        if runtime is not None:
            runtime.gs.destroy()


def run_parallel_batch(
    config: SimulationRunConfig,
    plan: BatchPlan,
    anchor_manifest: dict[str, Any],
    *,
    use_sampled_restitution: bool = True,
) -> BatchRuntimeResult:
    """Run one Genesis scene whose environments use the effective batch plan."""
    validate_simulation_options(config)
    validate_simulation_inputs(config)

    config.sim_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "migration_plan.json",
        "batch_plan.json",
        "run_manifest.json",
        "parameters.jsonl",
        "trajectories.npz",
        "trace_schema.json",
        "validation.json",
        "preview.png",
        "comparison.jpg",
        "contact_sheet.jpg",
        "simulation.mp4",
        "render_review.json",
        "run.log",
    ):
        path = config.sim_dir / name
        if path.exists():
            path.unlink()
    frames_dir = config.sim_dir / "frames"
    remove_path(frames_dir)
    if config.render:
        frames_dir.mkdir(parents=True)

    with stage_log(config.sim_dir / "run.log"):
        logger.info(
            "[Phase 3] Start batch {}: output={} environments={}",
            plan.batch_index,
            config.sim_dir,
            config.num_envs,
        )
        result = _run_parallel_batch_prepared(
            config,
            plan,
            anchor_manifest,
            frames_dir,
            use_sampled_restitution=use_sampled_restitution,
        )
        logger.info(
            "[Phase 3] Complete batch {}: valid={}/{}",
            plan.batch_index,
            result.validation["valid_count"],
            result.validation["sample_count"],
        )
        return result
