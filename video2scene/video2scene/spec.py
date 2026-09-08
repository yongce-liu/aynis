"""Scene contracts and deterministic, constrained configuration sampling."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation

from .layout import PROJECT_ROOT

DEFAULT_SPEC = PROJECT_ROOT / "scenes/ball-and-block-fall.json"


@dataclass(frozen=True)
class SamplingConfig:
    """Write deterministic sampled scene configurations."""

    output: Path
    spec: Path = DEFAULT_SPEC
    seed: int = 0
    count: int = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def pointer_parent(data, pointer):
    if not pointer.startswith("/"):
        raise ValueError(f"Expected JSON pointer, got {pointer}")
    parts = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]
    for part in parts[:-1]:
        data = data[int(part)] if isinstance(data, list) else data[part]
    key = int(parts[-1]) if isinstance(data, list) else parts[-1]
    return data, key


def bounds(entity):
    geo = entity["geometry"]
    center = np.asarray(entity["transform"]["position_m"], dtype=float)
    if geo["generator"] == "sphere":
        half = np.repeat(geo["radius_m"], 3)
    else:
        half = np.asarray(geo["dimensions_m"]) / 2
        rotation = Rotation.from_euler(
            "xyz", entity["transform"]["rotation_euler_xyz_deg"], degrees=True
        ).as_matrix()
        half = abs(rotation) @ half
    return center - half, center + half


def validate(spec, check_schema=True):
    if check_schema:
        import jsonschema

        jsonschema.Draft202012Validator(
            read_json(PROJECT_ROOT / "schemas/scene.schema.json")
        ).validate(spec)
    json.dumps(spec, allow_nan=False)
    entities = spec["entities"]
    for name, entity in entities.items():
        geo, physics = entity["geometry"], entity["physics"]
        if any(v <= 0 for v in geo.get("dimensions_m", [geo.get("radius_m", 1)])):
            raise ValueError(f"{name}: dimensions must be positive")
        for key in ("density_kg_m3", "young_modulus_pa"):
            if key in physics["parameters"] and physics["parameters"][key] <= 0:
                raise ValueError(f"{name}: {key} must be positive")
        if not -1 < physics["parameters"].get("poisson_ratio", 0.2) < 0.5:
            raise ValueError(f"{name}: invalid Poisson ratio")
        for key in ("friction", "rolling_friction", "damping_ratio"):
            if physics["parameters"].get(key, 0) < 0:
                raise ValueError(f"{name}: {key} cannot be negative")
        if not 0 <= physics["parameters"].get("restitution", 0) <= 1:
            raise ValueError(f"{name}: restitution must be between zero and one")
        if (
            geo.get("bevel_m", 0) < 0
            or geo.get("bevel_m", 0) >= min(geo.get("dimensions_m", [1])) / 2
        ):
            raise ValueError(f"{name}: invalid bevel size")
        if physics["type"] in ("soft", "cloth", "fluid"):
            if "deformable" not in physics or physics["collision"][
                "representation"
            ] in ("box", "sphere"):
                raise ValueError(
                    f"{name}: nonrigid body needs an explicit deformable representation"
                )
        if physics["type"] == "soft":
            support = physics["deformable"]["boundary_conditions"]["support_entity"]
            if support not in entities:
                raise ValueError(f"{name}: missing support {support}")
            body_lo, _ = bounds(entity)
            _, support_hi = bounds(entities[support])
            if abs(body_lo[2] - support_hi[2]) > 0.002:
                raise ValueError(f"{name}: deformable body must rest on its support")
    for name in spec["semantics"]["nonrigid_entities"]:
        if name not in entities or entities[name]["physics"]["type"] not in (
            "soft",
            "cloth",
            "fluid",
        ):
            raise ValueError(f"{name}: lost deformable classification")
    for relation in spec["semantics"]["relations"]:
        if relation["subject"] not in entities or relation["object"] not in entities:
            raise ValueError("Relation references unknown entity")

    event_names = {event["name"] for event in spec["semantics"]["events"]}
    rules = spec.get("physics_validation", {}).get("checks", [])
    rule_ids = [rule["id"] for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Physics validation check IDs must be unique")
    for rule in rules:
        rule_type = rule["type"]
        body_name = rule["entity"]
        if (
            body_name not in entities
            or entities[body_name]["physics"]["type"] != "rigid"
        ):
            raise ValueError(f"{rule['id']}: validation entity must be dynamic rigid")
        if rule.get("event") not in (None, *event_names):
            raise ValueError(f"{rule['id']}: validation event is not declared")
        body_lo, body_hi = bounds(entities[body_name])
        if rule_type == "soft_impact":
            target_name = rule["target"]
            if (
                target_name not in entities
                or entities[target_name]["physics"]["type"] != "soft"
            ):
                raise ValueError(f"{rule['id']}: target must be deformable")
            target_lo, target_hi = bounds(entities[target_name])
            clearance = float(rule.get("minimum_initial_clearance_m", 0.005))
            if body_lo[2] <= target_hi[2] + clearance:
                raise ValueError("Dynamic body must begin above its target")
            margin = float(rule.get("initial_xy_margin_m", 0.02))
            if np.any(body_lo[:2] <= target_lo[:2] + margin) or np.any(
                body_hi[:2] >= target_hi[:2] - margin
            ):
                raise ValueError("Impact footprint must fit within its target")
            floor_name = rule.get("floor_support")
            if floor_name is not None and floor_name not in entities:
                raise ValueError(f"{rule['id']}: floor support is unknown")
        elif rule_type == "rolling_transition":
            supports = rule["supports"]
            if not supports or any(name not in entities for name in supports):
                raise ValueError(f"{rule['id']}: rolling supports are invalid")
            if rule["target_support"] not in supports:
                raise ValueError(f"{rule['id']}: target support must be in supports")
            if rule["motion_axis"] not in (0, 1) or rule["angular_axis"] not in (
                0,
                1,
                2,
            ):
                raise ValueError(f"{rule['id']}: invalid motion or angular axis")
            support_lo, support_hi = bounds(entities[supports[0]])
            center = np.asarray(entities[body_name]["transform"]["position_m"])
            gap = float(body_lo[2] - support_hi[2])
            gap_range = rule.get("support_gap_range_m", [-0.005, 0.01])
            if not gap_range[0] <= gap <= gap_range[1]:
                raise ValueError(f"{rule['id']}: dynamic body must rest on its support")
            if not np.all(center[:2] >= support_lo[:2]) or not np.all(
                center[:2] <= support_hi[:2]
            ):
                raise ValueError(
                    f"{rule['id']}: initial center must lie on its support"
                )
        else:
            raise ValueError(f"{rule['id']}: unsupported validation type {rule_type!r}")

    soft_bounds = [
        bounds(entity)
        for entity in entities.values()
        if entity["physics"]["type"] == "soft"
    ]
    for index, (lower, upper) in enumerate(soft_bounds):
        for other_lower, other_upper in soft_bounds[index + 1 :]:
            if np.all(
                np.minimum(upper, other_upper) - np.maximum(lower, other_lower) > 0
            ):
                raise ValueError("Deformable bounding volumes overlap")
    camera = spec["camera"]
    if np.linalg.norm(np.subtract(camera["position_m"], camera["target_m"])) < 0.01:
        raise ValueError("Camera target coincides with camera")
    for name, parameter in spec["domain_randomization"]["parameters"].items():
        parent, key = pointer_parent(spec, parameter["path"])
        if not isinstance(parent[key], (int, float)):
            raise ValueError(f"{name}: numeric sampler requires numeric target")
        low, high = parameter["range"]
        if low > high or (parameter["distribution"] == "log_uniform" and low <= 0):
            raise ValueError(f"{name}: invalid sampling interval")


def sample(spec, seed, max_attempts=128):
    validate(spec)
    rng = np.random.Generator(np.random.PCG64(seed))
    for attempt in range(max_attempts):
        result, values = copy.deepcopy(spec), {}
        for name, parameter in sorted(
            spec["domain_randomization"]["parameters"].items()
        ):
            low, high = parameter["range"]
            distribution = parameter["distribution"]
            if distribution == "integer":
                value = int(rng.integers(low, high + 1))
            elif distribution == "log_uniform":
                value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                value = float(rng.uniform(low, high))
            parent, key = pointer_parent(result, parameter["path"])
            parent[key] = value
            values[name] = value
        # Keep resized supported deformables resting on their declared support plane.
        for entity in result["entities"].values():
            if entity["physics"]["type"] != "soft":
                continue
            support_id = entity["physics"]["deformable"]["boundary_conditions"][
                "support_entity"
            ]
            _, support_hi = bounds(result["entities"][support_id])
            entity["transform"]["position_m"][2] = float(
                support_hi[2] + entity["geometry"]["dimensions_m"][2] / 2
            )
        for rule in result.get("physics_validation", {}).get("checks", []):
            if rule["type"] != "rolling_transition":
                continue
            entity = result["entities"][rule["entity"]]
            radius = float(entity["geometry"]["radius_m"])
            motion_axis = int(rule["motion_axis"])
            angular_axis = int(rule["angular_axis"])
            linear_speed = float(
                entity["initial_state"]["linear_velocity_m_s"][motion_axis]
            )
            entity["initial_state"]["angular_velocity_rad_s"][angular_axis] = (
                float(rule.get("rolling_sign", 1.0)) * linear_speed / radius
            )
        result["sampling"] = {
            "seed": seed,
            "attempt": attempt,
            "base_spec_sha256": digest(spec),
            "values": values,
            "derived": [
                "supported deformable center height follows support top",
                "rolling angular velocity follows sampled linear speed and radius",
                "mass follows density times evaluated volume",
            ],
            "source_alignment_required": False,
        }
        result["quality"]["visual_review_status"] = "unreviewed_domain_variant"
        try:
            validate(result, check_schema=False)
        except ValueError:
            continue
        return result
    raise ValueError(
        f"No feasible sample after {max_attempts} attempts; narrow ranges or revise constraints"
    )


def main():
    args = tyro.cli(SamplingConfig)
    if args.count < 1:
        raise ValueError("count must be positive")
    spec = read_json(args.spec)
    write_json(
        args.output / "scene.schema.json",
        read_json(PROJECT_ROOT / "schemas/scene.schema.json"),
    )
    for seed in range(args.seed, args.seed + args.count):
        result = sample(spec, seed)
        result["$schema"] = "scene.schema.json"
        write_json(args.output / f"scene_{seed:06d}.json", result)
    logger.info(
        "Wrote {} deterministic scene configurations to {}", args.count, args.output
    )


if __name__ == "__main__":
    main()
