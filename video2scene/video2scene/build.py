"""Build and render the reviewed scene specification using Blender Python."""

import copy
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
import tyro
from loguru import logger
from mathutils import Vector
from PIL import Image, ImageDraw

from . import geometry as geo
from .assets import export_entity
from .evidence import extract, probe
from .filesystem import remove_path
from .hashing import sha256_file
from .layout import PROJECT_ROOT, resolve_scene_layout
from .logging_utils import stage_log
from .materials import material
from .spec import DEFAULT_SPEC, digest, read_json, sample, validate, write_json


@dataclass(frozen=True)
class BuildConfig:
    """Build a reviewed or sampled Blender handoff."""

    spec: Path = DEFAULT_SPEC
    output_root: Path | None = None
    samples: int = 64
    threads: int = 12
    seed: int | None = None
    handoff_only: bool = False


def aim(obj, target):
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def configure_scene(spec, samples, threads):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.gravity = spec["environment"]["gravity_m_s2"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 7
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x, scene.render.resolution_y = spec["camera"][
        "resolution_px"
    ]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = spec["source"]["fps"]
    scene.frame_start = scene.frame_end = 1
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = spec["lighting"]["exposure_ev"]
    scene.world = bpy.data.worlds.new("Neutral indoor environment")
    scene.world.use_nodes = True
    world_color = spec["lighting"].get("world_color_linear", [0.72, 0.76, 0.80])
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        *world_color,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = spec[
        "lighting"
    ]["world_strength"]
    collections = {}
    for name in (
        "DynamicObjects",
        "DeformableObjects",
        "Environment",
        "Fixtures",
        "VisualDetails",
        "CollisionGeometry",
        "CamerasLights",
    ):
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
        collections[name] = collection
    return scene, collections


def _collection_for_entity(entity, collections):
    requested = entity.get("render_collection")
    if requested is not None:
        if requested not in collections:
            raise ValueError(f"Unknown render collection: {requested}")
        return collections[requested]
    physics = entity["physics"]
    if physics["type"] == "soft":
        return collections["DeformableObjects"]
    if physics["type"] == "rigid":
        return collections["DynamicObjects"]
    if physics["collision"]["representation"] == "none":
        return collections["Fixtures"]
    return collections["Environment"]


def _entity_material(name, entity):
    appearance = entity["appearance"]
    return material(
        name + "_material",
        appearance["base_color_linear"],
        appearance["roughness"],
        appearance.get("material_kind", "plain"),
        int(appearance.get("texture_seed", 0)),
        metallic=appearance.get("metallic"),
        transmission_weight=float(appearance.get("transmission_weight", 0.0)),
        ior=float(appearance.get("ior", 1.45)),
        alpha=float(appearance.get("alpha", 1.0)),
    )


def _detail_material(name, entity, fallback_color, fallback_roughness, fallback_kind):
    appearance = entity["appearance"]
    return material(
        name + "_detail_material",
        appearance.get("detail_color_linear", fallback_color),
        float(appearance.get("detail_roughness", fallback_roughness)),
        appearance.get("detail_material_kind", fallback_kind),
    )


def construct(spec, scene, collections):
    roots = {}
    for name, entity in spec["entities"].items():
        params = entity["geometry"]
        kind = params["generator"]
        collection = _collection_for_entity(entity, collections)
        mat = _entity_material(name, entity)
        pos = entity["transform"]["position_m"]
        if kind == "sphere":
            detail_mat = _detail_material(
                name, entity, (0.36, 0.28, 0.16), 0.78, "plain"
            )
            obj = geo.ball(
                name,
                params["radius_m"],
                pos,
                mat,
                detail_mat,
                collection,
                collections["VisualDetails"],
                params.get("surface_detail"),
            )
        elif kind == "cushion":
            detail_mat = _detail_material(
                name, entity, (0.30, 0.29, 0.26), 0.92, "fabric"
            )
            obj = geo.cushion(
                name,
                params,
                pos,
                mat,
                detail_mat,
                collection,
                collections["VisualDetails"],
            )
        elif kind == "clamp":
            detail_mat = _detail_material(
                name, entity, (0.010, 0.012, 0.014), 0.37, "metal"
            )
            obj = geo.clamp(
                name,
                pos,
                mat,
                detail_mat,
                collection,
                params.get("model_scale", 1.0),
            )
        elif kind == "box":
            obj = geo.cube(
                name,
                params["dimensions_m"],
                pos,
                mat,
                collection,
                params.get("bevel_m", 0),
            )
        else:
            raise ValueError(f"Unsupported geometry generator: {kind}")
        obj.rotation_euler = [
            math.radians(v) for v in entity["transform"]["rotation_euler_xyz_deg"]
        ]
        physics = entity["physics"]["type"]
        obj["entity_id"] = name
        obj["physics_type"] = physics
        obj["semantic_label"] = entity["label"]
        obj["collision_representation"] = entity["physics"]["collision"][
            "representation"
        ]
        obj["initial_linear_velocity_m_s"] = entity["initial_state"][
            "linear_velocity_m_s"
        ]
        obj["initial_angular_velocity_rad_s"] = entity["initial_state"][
            "angular_velocity_rad_s"
        ]
        obj["physics_parameters_json"] = json.dumps(
            entity["physics"], ensure_ascii=False
        )
        obj["physics_validated"] = False
        roots[name] = obj
        if obj.type == "MESH" and obj["collision_representation"] != "none":
            proxy = obj.copy()
            proxy.data = obj.data.copy()
            proxy.name = name + "__collision"
            collections["CollisionGeometry"].objects.link(proxy)
            proxy["physics_owner"] = name
            proxy["is_duplicate_reference_only"] = True
            proxy.hide_render = True
            proxy.hide_set(True)
            proxy.display_type = "WIRE"
    camera_data = bpy.data.cameras.new("ReferenceCamera")
    camera = bpy.data.objects.new("ReferenceCamera", camera_data)
    collections["CamerasLights"].objects.link(camera)
    camera.location = spec["camera"]["position_m"]
    camera_data.lens = spec["camera"]["lens_mm"]
    camera_data.sensor_width = spec["camera"]["sensor_width_mm"]
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start, camera_data.clip_end = spec["camera"]["clip_m"]
    aim(camera, spec["camera"]["target_m"])
    scene.camera = camera
    for name in ("key", "fill"):
        settings = spec["lighting"][name]
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size, data.color = (
            settings["power_w"],
            settings["size_m"],
            settings["color_linear"],
        )
        obj = bpy.data.objects.new(name, data)
        collections["CamerasLights"].objects.link(obj)
        obj.location = settings["position_m"]
        aim(obj, settings["target_m"])
    bpy.context.view_layer.update()
    return roots


def render(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _resolve_build_target(config: BuildConfig):
    spec = read_json(config.spec)
    validate(spec)
    if config.seed is not None:
        spec = sample(spec, config.seed)
    spec["$schema"] = "scene.schema.json"
    layout = resolve_scene_layout(spec["scene_id"], config.output_root)
    layout.prepare()
    output = (
        layout.blender_variant(config.seed)
        if config.seed is not None
        else layout.blender
    )
    output.mkdir(parents=True, exist_ok=True)
    return spec, layout, output


def build(config: BuildConfig):
    spec, layout, output = _resolve_build_target(config)
    with stage_log(output / "build.log"):
        return _build(config, spec, layout, output)


def _build(config: BuildConfig, spec, layout, output):
    handoff_only = config.handoff_only
    assets_dir = output / "assets"
    remove_path(assets_dir)
    for legacy_dir in (output / "reference", output / "main"):
        remove_path(legacy_dir)
    for name in (
        "scene.blend",
        "scene.schema.json",
        "scene_spec.json",
        "scene_manifest.json",
        "validation.json",
        "preview.png",
        "comparison.jpg",
        "overview.png",
        "render_review.json",
    ):
        stale = output / name
        remove_path(stale)
    write_json(
        output / "scene.schema.json",
        read_json(PROJECT_ROOT / "schemas/scene.schema.json"),
    )
    video = Path(spec["source"]["path"])
    if not video.is_absolute():
        video = PROJECT_ROOT / video
    if sha256_file(video) != spec["source"]["sha256"]:
        raise ValueError(
            "Video hash differs from the analyzed source; create a new evidence-backed spec"
        )
    if handoff_only:
        logger.info("[Phase 1] Reading source metadata for handoff-only build")
        metadata = probe(video)
        for stale_name in (
            "preview.png",
            "comparison.jpg",
            "overview.png",
            "render_review.json",
        ):
            stale = output / stale_name
            if stale.exists():
                stale.unlink()
    else:
        logger.info("[Phase 1] Extracting source evidence")
        metadata = extract(video, layout.frames)
    logger.info("[Phase 1] Building metric scene and exporting local assets")
    scene, collections = configure_scene(spec, config.samples, config.threads)
    roots = construct(spec, scene, collections)
    manifest = copy.deepcopy(spec)
    manifest["quality"]["visual_review_status"] = (
        "not_rendered_handoff_only" if handoff_only else "pending_review_of_this_render"
    )
    if not handoff_only:
        write_json(
            output / "render_review.json", {"status": "pending_review_of_this_render"}
        )
    manifest["source_metadata"] = metadata
    manifest["artifact_layout"] = {
        "base": "scene_root",
        "source_frames": layout.frames.relative_to(layout.root).as_posix(),
        "blender_handoff": output.relative_to(layout.root).as_posix(),
        "simulation": layout.sim.relative_to(layout.root).as_posix(),
        "batches": layout.batches.relative_to(layout.root).as_posix(),
    }
    manifest["build"] = {
        "spec_sha256": digest(spec),
        "bpy_version": bpy.app.version_string,
        "genesis_version": importlib.metadata.version("genesis-world"),
        "render_engine": "CYCLES",
        "render_device": "CPU",
        "samples": config.samples,
        "rendered": not handoff_only,
        "handoff_only": handoff_only,
        "simulation_executed": False,
    }
    manifest["resolved_entities"] = {
        name: export_entity(name, root, spec["entities"][name], output)
        for name, root in roots.items()
    }
    camera = scene.camera
    width, height = spec["camera"]["resolution_px"]
    focal = width * camera.data.lens / camera.data.sensor_width
    manifest["resolved_camera"] = {
        "K_px": [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        "world_from_camera_blender": [list(row) for row in camera.matrix_world],
        "image_coordinates": "x right, y down; K uses a camera with +Z forward, +Y down",
        "blender_to_cv_camera": [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
    }
    write_json(output / "scene_spec.json", spec)
    write_json(output / "scene_manifest.json", manifest)
    for name, data in (("scene_spec.json", spec), ("scene_manifest.json", manifest)):
        text = bpy.data.texts.new(name)
        text.write(json.dumps(data, ensure_ascii=False, indent=2))
    scene["phase"] = 1
    scene["physics_validated"] = False
    scene["scene_id"] = spec["scene_id"]
    scene["source_frame"] = spec["source"]["reference_frame"]
    scene["source_video_used_as_texture"] = False
    scene["status"] = (
        "Reconstructed initial state; no simulation or future motion keyframes"
    )
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    selected_name = next(
        (
            name
            for name, entity in spec["entities"].items()
            if entity["physics"]["type"] == "rigid"
        ),
        next(iter(roots)),
    )
    bpy.context.view_layer.objects.active = roots[selected_name]
    roots[selected_name].select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))
    if not handoff_only:
        logger.info("[Phase 1] Rendering reference camera")
        render(scene, output / "preview.png")
        overview_data = camera.data.copy()
        overview = bpy.data.objects.new("OverviewCamera", overview_data)
        collections["CamerasLights"].objects.link(overview)
        overview.location = spec["camera"]["overview_position_m"]
        overview.data.lens = 45
        aim(overview, spec["camera"]["overview_target_m"])
        scene.camera = overview
        scene.render.resolution_x, scene.render.resolution_y = 960, 540
        logger.info("[Phase 1] Rendering structural overview")
        render(scene, output / "overview.png")
        scene.camera = camera
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.filepath = str(output / "preview.png")
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))
        with (
            Image.open(
                layout.reference_frame(spec["source"]["reference_frame"])
            ) as source,
            Image.open(output / "preview.png") as preview,
        ):
            comparison = Image.new("RGB", (width * 2, height + 32), "#202326")
            comparison.paste(source.convert("RGB"), (0, 32))
            comparison.paste(preview.convert("RGB"), (width, 32))
            draw = ImageDraw.Draw(comparison)
            draw.text(
                (12, 10),
                f"SOURCE | frame {spec['source']['reference_frame']}",
                fill="white",
            )
            draw.text(
                (width + 12, 10), "BLENDER | reconstructed initial state", fill="white"
            )
            comparison.save(output / "comparison.jpg", quality=95)
    logger.info("[Phase 1] Reopening saved blend in a separate process for validation")
    validate_command = [sys.executable, "-m", "video2scene.validate", str(output)]
    if handoff_only:
        validate_command.append("--no-image-checks")
    subprocess.run(validate_command, check=True)
    logger.info("[Phase 1] Complete: {}", output / "scene.blend")


def main():
    args = tyro.cli(BuildConfig)
    if args.samples < 1 or not 1 <= args.threads <= 256:
        raise ValueError("Require samples >= 1 and 1 <= threads <= 256")
    build(args)


if __name__ == "__main__":
    main()
