"""Compile a Phase-1 Blender handoff into an explicit Genesis solver plan."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .hashing import sha256_file
from .spec import read_json


class MigrationError(ValueError):
    """Raised when a scene cannot be migrated without changing its physics."""


def linear_to_srgb(
    color: list[float] | tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Convert Blender linear RGB values to display-space RGB for Genesis surfaces."""

    def convert(value: float) -> float:
        value = min(1.0, max(0.0, float(value)))
        return (
            12.92 * value
            if value <= 0.0031308
            else 1.055 * value ** (1.0 / 2.4) - 0.055
        )

    return (*[convert(value) for value in color[:3]], 1.0)


def vertical_fov_deg(camera: dict[str, Any]) -> float:
    """Convert Blender's horizontal sensor/lens pair to a vertical field of view."""
    width, height = camera["resolution_px"]
    horizontal_fov = 2.0 * math.atan(
        camera["sensor_width_mm"] / (2.0 * camera["lens_mm"])
    )
    vertical_fov = 2.0 * math.atan(math.tan(horizontal_fov / 2.0) * height / width)
    return math.degrees(vertical_fov)


def rayleigh_damping(entities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map solver-neutral damping ratios to one shared FEM Rayleigh pair."""
    frequencies = []
    ratios = []
    for entity in entities.values():
        if entity["physics_type"] != "soft":
            continue
        parameters = entity["physics_parameters"]
        dimensions = np.ptp(
            np.asarray(entity["rest_bounds_local_m"], dtype=float), axis=0
        )
        characteristic_length = max(float(np.min(dimensions)), 1e-4)
        wave_speed = math.sqrt(
            parameters["young_modulus_pa"] / parameters["density_kg_m3"]
        )
        frequencies.append(wave_speed / characteristic_length)
        ratios.append(parameters.get("damping_ratio", 0.0))
    if not frequencies:
        return {
            "reference_angular_frequency_rad_s": 0.0,
            "damping_ratio": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
        }
    omega = float(np.median(frequencies))
    ratio = float(np.median(ratios))
    return {
        "reference_angular_frequency_rad_s": omega,
        "damping_ratio": ratio,
        "alpha": ratio * omega,
        "beta": ratio / omega,
        "mapping": "zeta(omega)=alpha/(2*omega)+beta*omega/2; equal mass/stiffness contributions at reference omega",
    }


def _asset_record(
    blender_dir: Path, relative_path: str | None
) -> dict[str, Any] | None:
    if relative_path is None:
        return None
    path = (blender_dir / relative_path).resolve()
    if not path.is_file():
        raise MigrationError(f"Missing Phase-1 asset: {path}")
    record: dict[str, Any] = {
        "path": str(path),
        "relative_path": relative_path,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".obj":
        mesh = trimesh.load(path, force="mesh", process=False)
        record.update(
            vertices=int(len(mesh.vertices)),
            triangles=int(len(mesh.faces)),
            watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            volume_m3=float(mesh.volume),
        )
    return record


def choose_solver(name: str, entity: dict[str, Any]) -> dict[str, Any]:
    """Choose a solver without silently changing an entity's physical class."""
    physics_type = entity["physics_type"]
    collision = entity["collision"]["representation"]
    parameters = entity.get("physics_parameters", {})

    if physics_type in {"rigid", "static"}:
        if collision == "none":
            return {
                "role": "visual_only",
                "solver": None,
                "material": None,
                "reason": "Phase-1 marked this fixture outside the intended contact path.",
            }
        if collision not in {"sphere", "box"}:
            raise MigrationError(
                f"{name}: unsupported rigid collision representation {collision!r}"
            )
        return {
            "role": "dynamic_rigid" if physics_type == "rigid" else "static_rigid",
            "solver": "RigidSolver",
            "material": "gs.materials.Rigid",
            "collision": collision,
            "parameters": {
                "density_kg_m3": parameters.get("density_kg_m3"),
                "friction": parameters.get("friction", 0.5),
                "rolling_friction": parameters.get("rolling_friction", 0.0),
                "restitution": parameters.get("restitution", 0.0),
            },
            "reason": "Rigid/static semantic type and an explicit analytic collision primitive are available.",
        }

    if physics_type == "soft":
        deformable = entity.get("deformable") or {}
        if deformable.get("representation") != "closed_rest_surface":
            raise MigrationError(
                f"{name}: soft body is not a closed volumetric rest surface"
            )
        required = {"young_modulus_pa", "poisson_ratio", "density_kg_m3"}
        missing = sorted(required - parameters.keys())
        if missing:
            raise MigrationError(
                f"{name}: FEM migration is missing {', '.join(missing)}"
            )
        if collision != "deformable_surface":
            raise MigrationError(
                f"{name}: refusing to replace a deformable surface with {collision!r}"
            )
        return {
            "role": "deformable",
            "solver": "FEMSolver",
            "material": "gs.materials.FEM.Elastic",
            "parameters": {
                "E": parameters["young_modulus_pa"],
                "nu": parameters["poisson_ratio"],
                "rho": parameters["density_kg_m3"],
                "friction_mu": parameters.get("friction", 0.5),
                "damping_ratio": parameters.get("damping_ratio", 0.0),
                "model": "linear",
            },
            "discretization": {
                "source": "Phase-1 closed physics OBJ",
                "surface_decimation": False,
                "tetgen": {
                    "order": 1,
                    "quality": True,
                    "mindihedral": 0,
                    "minratio": 10.0,
                    "nobisect": True,
                    "maxvolume": -1.0,
                },
            },
            "reason": (
                "The object is a filled compliant volume with E/nu/rho estimates. Volumetric FEM preserves the "
                "closed Blender surface and maps those constitutive parameters directly. The accepted nominal preview "
                "uses Genesis linear FEM; element inversion is retained as a non-gating diagnostic."
            ),
        }

    if physics_type in {"cloth", "fluid"}:
        raise MigrationError(
            f"{name}: {physics_type} requires a dedicated shell/fluid compiler; refusing a rigid or generic-soft fallback"
        )
    raise MigrationError(f"{name}: unsupported physics type {physics_type!r}")


def build_plan(blender_dir: str | Path) -> dict[str, Any]:
    blender_dir = Path(blender_dir).resolve()
    manifest_path = blender_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise MigrationError(f"Missing Phase-1 manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "video2scene.scene.v1":
        raise MigrationError(
            f"Unsupported Phase-1 schema: {manifest.get('schema_version')!r}"
        )
    if not manifest.get("quality", {}).get("no_motion_keyframes", False):
        raise MigrationError(
            "Phase-1 scene contains or may contain future motion keyframes"
        )

    decisions: dict[str, Any] = {}
    for name, entity in manifest["resolved_entities"].items():
        decision = choose_solver(name, entity)
        decision["source_physics_type"] = entity["physics_type"]
        decision["visual_asset"] = _asset_record(blender_dir, entity.get("visual_mesh"))
        decision["physics_asset"] = _asset_record(
            blender_dir, entity.get("physics_mesh")
        )
        if decision["role"] == "deformable":
            physics_asset = decision["physics_asset"]
            if (
                physics_asset is None
                or not physics_asset["watertight"]
                or not physics_asset["winding_consistent"]
            ):
                raise MigrationError(
                    f"{name}: FEM requires a watertight, consistently wound Phase-1 physics mesh"
                )
        decision["position_m"] = entity["position_m"]
        decision["quaternion_wxyz"] = entity["quaternion_wxyz"]
        decision["initial_state"] = entity["initial_state"]
        decisions[name] = decision

    soft_entities = {
        name: entity
        for name, entity in manifest["resolved_entities"].items()
        if entity["physics_type"] == "soft"
    }
    damping = rayleigh_damping(soft_entities)
    return {
        "schema_version": "video2scene.genesis-plan.v1",
        "scene_id": manifest["scene_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_scene_spec_sha256": manifest.get("build", {}).get("spec_sha256"),
        "policy": {
            "initial_state_only_from_video": True,
            "source_video_access_during_rollout": False,
            "future_pose_replay": False,
            "nonrigid_fallback_to_rigid": False,
            "visual_geometry_source": "Phase-1 exported visual OBJ",
            "physics_geometry_source": "Phase-1 collision metadata and closed deformable OBJ",
        },
        "solver_stack": {
            "rigid": "RigidSolver",
            "deformable": "FEMSolver",
            "coupler": "LegacyCoupler",
            "coupler_reason": (
                "Legacy coupling is the validated default when rigid-FEM interaction is present; "
                "pure-rigid scenes retain the same deterministic runtime configuration."
            ),
            "precision": "64",
            "recommended_simulation_hz": 1920,
            "fem_self_contact": False,
            "rayleigh_damping": damping,
        },
        "entities": decisions,
        "camera": {
            "position_m": manifest["camera"]["position_m"],
            "target_m": manifest["camera"]["target_m"],
            "resolution_px": manifest["camera"]["resolution_px"],
            "vertical_fov_deg": vertical_fov_deg(manifest["camera"]),
        },
        "validation_contract": manifest.get("physics_validation", {}),
        "quality_gates": [
            "exact rigid t0 pose and velocity after optional deformable relaxation",
            "finite and bounded simulated state",
            *manifest.get("genesis_handoff", {}).get("required_phase2_validation", []),
            "no source frame, trajectory, or keyframe access after t0",
        ],
    }
