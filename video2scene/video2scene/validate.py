"""Validate saved Blender geometry, exported rest states and initial projection."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import bpy
import tyro
from loguru import logger
import numpy as np
import trimesh
from bpy_extras.object_utils import world_to_camera_view
from PIL import Image

from .hashing import sha256_file
from .simulation_geometry import rigid_bottom_z, world_bounds
from .spec import digest, read_json, validate, write_json


@dataclass(frozen=True)
class ValidationConfig:
    """Validate one saved Blender handoff."""

    blender_dir: Annotated[Path, tyro.conf.Positional]
    no_image_checks: bool = False


def run(output, *, require_images=True):
    output = Path(output).resolve()
    spec = read_json(output / "scene_spec.json")
    manifest = read_json(output / "scene_manifest.json")
    validate(spec)
    bpy.ops.wm.open_mainfile(filepath=str(output / "scene.blend"))
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    checks = {}
    geometry = {}
    checks["metric_scene"] = (
        scene.unit_settings.scale_length == 1 and scene.unit_settings.system == "METRIC"
    )
    checks["manifest_embedded"] = (
        json.loads(bpy.data.texts["scene_manifest.json"].as_string()) == manifest
    )
    checks["spec_hash_matches"] = digest(spec) == manifest["build"]["spec_sha256"]
    checks["no_motion_keyframes"] = len(bpy.data.actions) == 0 and all(
        obj.animation_data is None for obj in scene.objects
    )
    checks["no_source_image_textures"] = all(
        node.type != "TEX_IMAGE"
        for mat in bpy.data.materials
        if mat.use_nodes
        for node in mat.node_tree.nodes
    )
    checks["reference_camera_active"] = scene.camera.name == "ReferenceCamera"
    checks["all_transforms_finite"] = all(
        np.isfinite(np.array(obj.matrix_world)).all() for obj in scene.objects
    )
    checks["physics_claims_remain_unvalidated"] = (
        manifest["quality"]["physics_validated"] is False
        and manifest["quality"]["training_ready"] is False
    )
    for name, entity in manifest["resolved_entities"].items():
        obj = bpy.data.objects[name]
        checks[f"{name}.physics_type"] = (
            obj["physics_type"] == spec["entities"][name]["physics"]["type"]
        )
        checks[f"{name}.transform_export"] = bool(
            np.allclose(np.asarray(obj.matrix_world), entity["world_matrix"], atol=1e-7)
        )
        if "physics_mesh" not in entity:
            continue
        mesh = trimesh.load(
            output / entity["physics_mesh"], force="mesh", process=False
        )
        good = bool(
            mesh.is_watertight
            and mesh.is_winding_consistent
            and mesh.volume > 0
            and np.isfinite(mesh.vertices).all()
        )
        checks[f"{name}.closed_positive_mesh"] = good
        geometry[name] = {
            "watertight": bool(mesh.is_watertight),
            "consistent_winding": bool(mesh.is_winding_consistent),
            "volume_m3": float(mesh.volume),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
        }
        if "mass_kg" in entity:
            checks[f"{name}.mass_density_consistent"] = bool(
                np.isclose(
                    mesh.volume * entity["physics_parameters"]["density_kg_m3"],
                    entity["mass_kg"],
                    rtol=1e-5,
                )
            )
        if entity["physics_type"] == "soft":
            with np.load(output / entity["rest_state_npz"]) as rest:
                checks[f"{name}.rest_vertex_correspondence"] = bool(
                    np.allclose(rest["vertices_local_m"], mesh.vertices, atol=1e-7)
                    and np.array_equal(rest["triangles"], mesh.faces)
                )
                checks[f"{name}.boundary_selectors_present"] = all(
                    len(rest[key]) > 0
                    for key in ("support_candidate", "contact_surface", "seam_binding")
                )
            checks[f"{name}.no_rigid_substitution"] = (
                obj.rigid_body is None
                and entity["collision"]["representation"] == "deformable_surface"
            )
    projection = {}
    width, height = spec["camera"]["resolution_px"]
    nominal = spec.get("sampling", {}).get("source_alignment_required", True)
    for name, landmark in spec["evidence"]["landmarks_px"].items():
        uv = world_to_camera_view(
            scene, scene.camera, bpy.data.objects[name].matrix_world.translation
        )
        pixel = np.array([uv.x * width, (1 - uv.y) * height])
        error = float(np.linalg.norm(pixel - landmark["center"]))
        projection[name] = {
            "render_center_px": pixel.tolist(),
            "source_annotation_px": landmark["center"],
            "center_error_px": error,
            "interpretation": "Consistency with manual projection targets, not independent 3D reconstruction accuracy.",
        }
        checks[f"{name}.in_camera"] = bool(
            0 <= uv.x <= 1 and 0 <= uv.y <= 1 and uv.z > 0
        )
        if nominal:
            checks[f"{name}.reference_projection"] = error <= landmark["tolerance_px"]
    initial_relationships = {}
    for rule in spec.get("physics_validation", {}).get("checks", []):
        rule_id = rule["id"]
        body_name = rule["entity"]
        body = manifest["resolved_entities"][body_name]
        if rule["type"] == "soft_impact":
            target_name = rule["target"]
            target = manifest["resolved_entities"][target_name]
            body_mesh = trimesh.load(
                output / body["physics_mesh"], force="mesh", process=False
            )
            body_mesh.apply_transform(body["world_matrix"])
            target_mesh = trimesh.load(
                output / target["physics_mesh"], force="mesh", process=False
            )
            target_mesh.apply_transform(target["world_matrix"])
            clearance = float(body_mesh.bounds[0, 2] - target_mesh.bounds[1, 2])
            initial_relationships[rule_id] = {
                "entity": body_name,
                "target": target_name,
                "clearance_m": clearance,
            }
            checks[f"{rule_id}.initial_clearance"] = clearance > float(
                rule.get("minimum_initial_clearance_m", 0.005)
            )
        elif rule["type"] == "rolling_transition":
            support_name = rule["supports"][0]
            support = manifest["resolved_entities"][support_name]
            position = np.asarray(body["position_m"], dtype=float)
            quaternion = np.asarray(body["quaternion_wxyz"], dtype=float)
            support_aabb = world_bounds(support)
            gap = rigid_bottom_z(body, position, quaternion) - support_aabb[1, 2]
            center_inside = bool(
                np.all(position[:2] >= support_aabb[0, :2])
                and np.all(position[:2] <= support_aabb[1, :2])
            )
            initial_relationships[rule_id] = {
                "entity": body_name,
                "target": support_name,
                "vertical_gap_m": float(gap),
                "center_inside_support_xy": center_inside,
            }
            gap_range = rule.get("support_gap_range_m", [-0.005, 0.01])
            checks[f"{rule_id}.initial_support_gap"] = bool(
                gap_range[0] <= gap <= gap_range[1]
            )
            checks[f"{rule_id}.initial_support_footprint"] = center_inside
    image_stats = {}
    if require_images:
        for name in ("preview.png", "overview.png"):
            image = np.asarray(Image.open(output / name).convert("RGB"))
            image_stats[name] = {
                "shape": list(image.shape),
                "std_rgb": float(image.std()),
                "mean_rgb": image.mean(axis=(0, 1)).tolist(),
            }
            checks[f"{name}.nonblank"] = bool(
                image.std() > 8 and 10 < image.mean() < 245
            )
    review_path = output / "render_review.json"
    if review_path.exists():
        review = read_json(review_path)
        for name, sha256 in review.get("reviewed_image_sha256", {}).items():
            checks[f"{name}.review_matches_render"] = (
                sha256_file(output / name) == sha256
            )
    assets = sorted(path for path in (output / "assets").rglob("*") if path.is_file())
    report = {
        "schema_version": "video2scene.validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "geometry": geometry,
        "projection": projection,
        "initial_relationships": initial_relationships,
        "images": image_stats,
        "asset_sha256": {
            str(path.relative_to(output)): sha256_file(path) for path in assets
        },
        "physics_validated": False,
        "training_ready": False,
        "not_validated": [
            "True scale/depth",
            "unseen geometry",
            "global triangle self-intersection",
            "soft material identification",
            "contact dynamics",
            "Genesis import/stepping",
            "multi-environment scaling",
        ],
    }
    write_json(output / "validation.json", report)
    logger.info(
        "{}",
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(checks),
                "failures": [key for key, ok in checks.items() if not ok],
                "projection": projection,
                "initial_relationships": initial_relationships,
            },
            indent=2,
        ),
    )
    if not report["passed"]:
        raise ValueError("Saved scene validation failed; see validation.json")
    return report


def main():
    args = tyro.cli(ValidationConfig)
    run(args.blender_dir, require_images=not args.no_image_checks)


if __name__ == "__main__":
    main()
