"""Plan deterministic domain-randomized batches for Genesis parallel simulation."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .spec import digest, pointer_parent, sample, validate

PER_ENV = "per_environment"
PER_BATCH = "per_batch"


@dataclass(frozen=True)
class PlannedSample:
    """One effective scene configuration assigned to a Genesis environment."""

    sample_id: int
    env_index: int
    sample_seed: int
    attempt: int
    spec: dict[str, Any]


@dataclass(frozen=True)
class BatchPlan:
    """A batch with one shared build configuration and per-environment overrides."""

    batch_index: int
    batch_seed: int | None
    batch_spec: dict[str, Any]
    samples: tuple[PlannedSample, ...]
    parameter_execution: dict[str, dict[str, str]]


def derive_seed(root_seed: int, *coordinates: int | str) -> int:
    """Derive a stable non-negative 63-bit seed without depending on batch size."""
    payload = ":".join(
        [str(root_seed), *(str(value) for value in coordinates)]
    ).encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) & ((1 << 63) - 1)


def pointer_value(data: dict[str, Any], pointer: str) -> Any:
    """Read a JSON-pointer value."""
    parent, key = pointer_parent(data, pointer)
    return parent[key]


def set_pointer_value(data: dict[str, Any], pointer: str, value: Any) -> None:
    """Write a JSON-pointer value."""
    parent, key = pointer_parent(data, pointer)
    parent[key] = copy.deepcopy(value)


def parameter_execution(base_spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Classify parameters by the finest physically faithful Genesis execution lane."""
    result: dict[str, dict[str, str]] = {}
    entities = base_spec["entities"]
    for name, parameter in sorted(
        base_spec["domain_randomization"]["parameters"].items()
    ):
        path = parameter["path"]
        parts = path.lstrip("/").split("/")
        lane = PER_BATCH
        reason = "Shared by the homogeneous Genesis scene build."
        application = "batch_scene_build"
        if len(parts) >= 3 and parts[0] == "entities":
            entity = entities[parts[1]]
            tail = "/".join(parts[2:])
            is_dynamic_rigid = entity["physics"]["type"] == "rigid"
            if is_dynamic_rigid and tail.startswith("initial_state/"):
                lane = PER_ENV
                reason = "Applied with a batched rigid-state setter after FEM gravity seating."
                application = "batched_rigid_velocity"
            elif is_dynamic_rigid and tail.startswith("transform/"):
                lane = PER_ENV
                reason = (
                    "Applied with batched rigid pose setters after FEM gravity seating."
                )
                application = "batched_rigid_pose"
            elif is_dynamic_rigid and tail in {
                "geometry/radius_m",
                "geometry/dimensions_m/0",
                "geometry/dimensions_m/1",
                "geometry/dimensions_m/2",
            }:
                lane = PER_ENV
                reason = "Applied through a Genesis heterogeneous rigid morph."
                application = "heterogeneous_rigid_geometry"
            elif is_dynamic_rigid and tail == "physics/parameters/density_kg_m3":
                lane = PER_ENV
                reason = (
                    "Converted to mass and inertia and applied with batched rigid info."
                )
                application = "batched_rigid_mass_and_inertia"
            elif entity["physics"]["type"] == "soft":
                reason = "FEM rest geometry and constitutive parameters are compiled once and therefore vary per batch."
                application = (
                    "blender_handoff_and_fem_tetrahedralization"
                    if tail.startswith("geometry/")
                    else "shared_fem_material_or_damping"
                    if tail.startswith("physics/")
                    else "shared_render_surface"
                )
            elif tail.startswith("appearance/"):
                reason = "Genesis surfaces are shared by cloned environments and therefore vary per batch."
                application = "shared_render_surface"
            elif tail.startswith("physics/"):
                reason = "Coupling/contact material coefficients are shared by cloned environments and vary per batch."
                application = "shared_rigid_material_or_coupler"
        elif parts[0] == "camera":
            reason = "The camera is shared by the batch renderer and therefore varies per batch."
            application = "shared_batch_camera"
        elif parts[0] == "lighting":
            reason = "Lighting is shared by the batch renderer and therefore varies per batch."
            application = "shared_batch_lighting"
        result[name] = {
            "lane": lane,
            "reason": reason,
            "application": application,
            "path": path,
            "scope": parameter["scope"],
        }
    return result


def _effective_spec(
    base_spec: dict[str, Any],
    batch_spec: dict[str, Any],
    candidate: dict[str, Any],
    execution: dict[str, dict[str, str]],
) -> dict[str, Any]:
    result = copy.deepcopy(batch_spec)
    for name, item in execution.items():
        if item["lane"] == PER_ENV:
            set_pointer_value(
                result, item["path"], pointer_value(candidate, item["path"])
            )
    validate(result, check_schema=False)
    return result


def plan_batch(
    base_spec: dict[str, Any],
    *,
    root_seed: int,
    batch_index: int,
    num_envs: int,
    sample_start: int,
    batch_spec: dict[str, Any] | None = None,
    batch_seed: int | None = None,
    max_attempts: int = 128,
) -> BatchPlan:
    """Create deterministic effective specs while preserving batch-shared solver parameters."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    execution = parameter_execution(base_spec)
    if batch_spec is None:
        batch_seed = (
            derive_seed(root_seed, "batch", batch_index)
            if batch_seed is None
            else batch_seed
        )
        batch_spec = sample(base_spec, batch_seed)
    else:
        batch_spec = copy.deepcopy(batch_spec)
        validate(batch_spec, check_schema=False)

    planned: list[PlannedSample] = []
    for env_index in range(num_envs):
        for attempt in range(max_attempts):
            sample_seed = derive_seed(
                root_seed, "sample", sample_start + env_index, attempt
            )
            candidate = sample(base_spec, sample_seed)
            try:
                effective = _effective_spec(base_spec, batch_spec, candidate, execution)
            except ValueError:
                continue
            effective["sampling"] = {
                "root_seed": root_seed,
                "batch_index": batch_index,
                "batch_seed": batch_seed,
                "sample_id": sample_start + env_index,
                "sample_seed": sample_seed,
                "attempt": attempt,
                "base_spec_sha256": digest(base_spec),
                "batch_spec_sha256": digest(batch_spec),
                "parameter_values": {
                    name: pointer_value(effective, item["path"])
                    for name, item in execution.items()
                },
                "parameter_lanes": {
                    name: item["lane"] for name, item in execution.items()
                },
                "source_alignment_required": False,
            }
            effective["quality"]["visual_review_status"] = "unreviewed_domain_variant"
            planned.append(
                PlannedSample(
                    sample_id=sample_start + env_index,
                    env_index=env_index,
                    sample_seed=sample_seed,
                    attempt=attempt,
                    spec=effective,
                )
            )
            break
        else:
            raise ValueError(
                f"Unable to combine batch {batch_index} with environment {env_index} after {max_attempts} attempts"
            )

    return BatchPlan(
        batch_index=batch_index,
        batch_seed=batch_seed,
        batch_spec=batch_spec,
        samples=tuple(planned),
        parameter_execution=execution,
    )


def serializable_batch_plan(plan: BatchPlan) -> dict[str, Any]:
    """Return the compact, inspectable part of a batch plan."""
    return {
        "schema_version": "video2scene.domain-randomization-plan.v1",
        "batch_index": plan.batch_index,
        "batch_seed": plan.batch_seed,
        "batch_spec_sha256": digest(plan.batch_spec),
        "parameter_execution": plan.parameter_execution,
        "batch_parameters": {
            name: pointer_value(plan.batch_spec, item["path"])
            for name, item in plan.parameter_execution.items()
            if item["lane"] == PER_BATCH
        },
        "samples": [
            {
                "sample_id": item.sample_id,
                "env_index": item.env_index,
                "sample_seed": item.sample_seed,
                "attempt": item.attempt,
                "spec_sha256": digest(item.spec),
                "parameters": item.spec["sampling"]["parameter_values"],
            }
            for item in plan.samples
        ],
    }


def _wxyz_from_euler(euler_xyz_deg: list[float]) -> list[float]:
    xyzw = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _geometry_scale(anchor: dict[str, Any], authored: dict[str, Any]) -> np.ndarray:
    anchor_geometry = anchor["geometry"]
    geometry = authored["geometry"]
    if geometry["generator"] == "sphere":
        ratio = float(geometry["radius_m"]) / float(anchor_geometry["radius_m"])
        return np.repeat(ratio, 3)
    if geometry["generator"] == "box":
        return np.asarray(geometry["dimensions_m"], dtype=float) / np.asarray(
            anchor_geometry["dimensions_m"], dtype=float
        )
    return np.ones(3, dtype=float)


def resolve_runtime_entities(
    anchor_manifest: dict[str, Any], effective_specs: list[dict[str, Any]]
) -> list[dict[str, dict[str, Any]]]:
    """Resolve per-environment rigid overrides against one reviewed Blender handoff."""
    anchor_authored = anchor_manifest["entities"]
    resolved_batches: list[dict[str, dict[str, Any]]] = []
    for spec in effective_specs:
        resolved = copy.deepcopy(anchor_manifest["resolved_entities"])
        for name, entity in resolved.items():
            authored = spec["entities"][name]
            entity["appearance"] = copy.deepcopy(authored["appearance"])
            entity["physics_parameters"] = copy.deepcopy(
                authored["physics"]["parameters"]
            )
            entity["initial_state"] = copy.deepcopy(authored["initial_state"])
            entity["position_m"] = [
                float(value) for value in authored["transform"]["position_m"]
            ]
            entity["quaternion_wxyz"] = _wxyz_from_euler(
                authored["transform"]["rotation_euler_xyz_deg"]
            )
            if entity["physics_type"] != "rigid":
                continue
            scale = _geometry_scale(anchor_authored[name], authored)
            entity["visual_scale_xyz"] = scale.tolist()
            collision = copy.deepcopy(authored["physics"]["collision"])
            if collision["representation"] == "sphere":
                collision["radius_m"] = float(authored["geometry"]["radius_m"])
            elif collision["representation"] == "box":
                collision["dimensions_m"] = [
                    float(value) for value in authored["geometry"]["dimensions_m"]
                ]
            entity["collision"] = collision
            if entity.get("mass_kg") is not None:
                anchor_density = float(
                    anchor_authored[name]["physics"]["parameters"]["density_kg_m3"]
                )
                density = float(authored["physics"]["parameters"]["density_kg_m3"])
                volume_scale = float(np.prod(scale))
                entity["mass_kg"] = float(
                    entity["mass_kg"] * volume_scale * density / anchor_density
                )
            if entity.get("rest_bounds_local_m") is not None:
                entity["rest_bounds_local_m"] = (
                    np.asarray(entity["rest_bounds_local_m"], dtype=float)
                    * scale[None, :]
                ).tolist()
        targets = {
            relation["subject"]: relation["object"]
            for relation in spec["semantics"]["relations"]
            if relation["relation"] == "falls_onto"
        }
        for body_name, target_name in targets.items():
            if body_name not in resolved:
                continue
            body_friction = float(
                spec["entities"][body_name]["physics"]["parameters"].get(
                    "friction", 0.5
                )
            )
            target_friction = float(
                spec["entities"][target_name]["physics"]["parameters"].get(
                    "friction", 0.5
                )
            )
            resolved[body_name]["physics_parameters"]["coupling_friction"] = math.sqrt(
                body_friction * target_friction
            )
        resolved_batches.append(resolved)
    return resolved_batches
