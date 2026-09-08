"""Shared Genesis runtime construction and rendering primitives."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .genesis_plan import MigrationError, build_plan, linear_to_srgb
from .simulation_config import SimulationRunConfig
from .spec import read_json


@dataclass
class RuntimeScene:
    gs: Any
    scene: Any
    camera: Any
    manifest: dict[str, Any]
    plan: dict[str, Any]
    physical: dict[str, Any]
    visual_followers: dict[str, Any]
    deformable: dict[str, Any]
    surface_indices: dict[str, np.ndarray]
    source_surfaces: dict[str, np.ndarray]
    tetrahedra: dict[str, np.ndarray]
    reference_tet_signs: dict[str, np.ndarray]
    build_seconds: float
    env_entities: list[dict[str, dict[str, Any]]] | None = None


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _fem_damping(plan: dict[str, Any]) -> tuple[float, float]:
    damping = plan["solver_stack"]["rayleigh_damping"]
    return float(damping["alpha"]), float(damping["beta"])


def _surface(
    gs: Any,
    entity: dict[str, Any],
    *,
    deformable: bool = False,
    color_space: str = "linear",
) -> Any:
    base_color = entity["appearance"]["base_color_linear"]
    if color_space == "linear":
        color = (*[float(value) for value in base_color[:3]], 1.0)
    elif color_space == "srgb":
        color = linear_to_srgb(base_color)
    else:
        raise ValueError("--surface-color-space must be linear or srgb")
    appearance = entity["appearance"]
    parameters: dict[str, Any] = {
        "color": color,
        "roughness": float(appearance.get("roughness", 0.8)),
        "opacity": float(appearance.get("alpha", 1.0)),
        "ior": float(appearance.get("ior", 1.0)),
    }
    if deformable:
        parameters["vis_mode"] = "visual"
    parameters["metallic"] = float(appearance.get("metallic", 0.0))
    transmission = float(appearance.get("transmission_weight", 0.0))
    if appearance.get("material_kind") == "glass":
        # Rasterized refraction differs from the Cycles handoff. A neutral,
        # translucent surface preserves plate visibility without fake source pixels.
        parameters["specular_trans"] = 0.0
        parameters["diffuse_trans"] = 0.0
        return gs.surfaces.Default(**parameters)
    parameters["specular_trans"] = transmission
    return gs.surfaces.Default(**parameters)


def _rigid_material(
    gs: Any,
    entity: dict[str, Any],
    *,
    coupled: bool,
    coupling_softness_m: float = 0.006,
    coupling_friction_scale: float = 1.0,
    coupling_restitution: bool = False,
    sphere_softness_radius_m: float | None = None,
) -> Any:
    parameters = entity.get("physics_parameters", {})
    kwargs: dict[str, Any] = {
        "friction": max(0.01, float(parameters.get("friction", 0.5))),
        "needs_coup": coupled,
    }
    if coupled:
        collision = entity["collision"]["representation"]
        friction = coupling_friction_scale * float(
            parameters.get("coupling_friction", parameters.get("friction", 0.5))
        )
        softness = coupling_softness_m
        if collision == "sphere":
            # A wider normal influence prevents a fast sphere from crossing a vertex-sampled deformable surface.
            radius = sphere_softness_radius_m or float(entity["collision"]["radius_m"])
            softness = max(softness, 0.375 * radius)
            friction = max(0.2, 0.4 * friction)
        kwargs.update(
            coup_friction=min(5.0, friction),
            coup_softness=softness,
            # The accepted nominal path keeps this at zero; Phase 3 may opt into the sampled prior.
            coup_restitution=float(parameters.get("restitution", 0.0))
            if coupling_restitution
            else 0.0,
            sdf_cell_size=0.004,
            sdf_min_res=20,
            sdf_max_res=72,
        )
    if parameters.get("density_kg_m3") is not None:
        kwargs["rho"] = float(parameters["density_kg_m3"])
    if parameters.get("rolling_friction") is not None:
        kwargs["friction_rolling"] = float(parameters["rolling_friction"])
    return gs.materials.Rigid(**kwargs)


def _physics_morph(gs: Any, entity: dict[str, Any]) -> Any:
    collision = entity["collision"]
    common = {
        "pos": tuple(entity["position_m"]),
        "quat": tuple(entity["quaternion_wxyz"]),
        "fixed": entity["physics_type"] == "static",
        "visualization": False,
        "collision": True,
    }
    if collision["representation"] == "sphere":
        return gs.morphs.Sphere(radius=float(collision["radius_m"]), **common)
    if collision["representation"] == "box":
        return gs.morphs.Box(size=tuple(collision["dimensions_m"]), **common)
    raise MigrationError(
        f"Unsupported rigid collision representation: {collision['representation']}"
    )


def _visual_morph(gs: Any, path: Path, entity: dict[str, Any], *, batched: bool) -> Any:
    return gs.morphs.Mesh(
        file=str(path),
        scale=tuple(entity.get("visual_scale_xyz", (1.0, 1.0, 1.0))),
        pos=tuple(entity["position_m"]),
        quat=tuple(entity["quaternion_wxyz"]),
        fixed=True,
        collision=False,
        visualization=True,
        align=False,
        convexify=False,
        decimate=False,
        watertighten=None,
        file_meshes_are_zup=True,
        batch_fixed_verts=batched,
    )


def build_runtime(
    config: SimulationRunConfig,
    *,
    env_entities: list[dict[str, dict[str, Any]]] | None = None,
    use_sampled_coupling_restitution: bool = False,
) -> RuntimeScene:
    os.environ.setdefault("UV_CACHE_DIR", "/tmp/skill2scene-uv-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/skill2scene-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/skill2scene-xdg-cache")

    import genesis as gs

    plan = build_plan(config.blender_dir)
    manifest = read_json(config.blender_dir / "scene_manifest.json")
    heterogeneous = env_entities is not None
    if env_entities is None:
        env_entities = [manifest["resolved_entities"] for _ in range(config.num_envs)]
    if len(env_entities) != config.num_envs:
        raise ValueError(
            f"Expected {config.num_envs} environment descriptions, got {len(env_entities)}"
        )
    if config.backend not in {"cpu", "gpu"}:
        raise ValueError("--backend must be cpu or gpu")
    gs.init(
        backend=getattr(gs, config.backend),
        precision="64",
        seed=config.seed,
        use_deterministic_algorithms=True,
    )

    alpha, beta = _fem_damping(plan)
    alpha *= config.damping_scale
    beta *= config.damping_scale
    plan["solver_stack"].update(
        coupler="LegacyCoupler" if config.coupler == "legacy" else "SAPCoupler",
        coupler_reason=(
            plan["solver_stack"]["coupler_reason"]
            if config.coupler == "legacy"
            else "Explicit diagnostic override; SAP is not the default validated coupling path."
        ),
        simulation_hz=config.simulation_hz,
        fem_self_contact=config.coupler == "sap",
        effective_rayleigh_damping={
            "alpha": alpha,
            "beta": beta,
            "scale": config.damping_scale,
        },
        coupling_softness_m=config.coupling_softness_m,
        coupling_friction_scale=config.coupling_friction_scale,
        lighting_transfer={
            "source": "Phase-1 Blender world and area-light metadata",
            "light_model": config.light_model,
            "ambient_light": config.ambient_light,
            "key_light_scale": config.key_light_scale,
            "fill_light_scale": config.fill_light_scale,
            "render_shadows": config.render_shadows,
            "surface_color_space": config.surface_color_space,
            "approximate_area_lights": config.approximate_area_lights,
        },
        sphere_contact={
            "softness_m": max(
                config.coupling_softness_m,
                0.375
                * max(
                    (
                        float(entity["collision"]["radius_m"])
                        for entities in env_entities
                        for entity in entities.values()
                        if entity["collision"]["representation"] == "sphere"
                    ),
                    default=0.0,
                ),
            ),
            "friction_factor": 0.4,
            "reason": "Resolve fast curved contact against vertex-sampled deformable surfaces without trajectory control.",
        },
    )
    max_modulus = max(
        float(entity["physics_parameters"].get("young_modulus_pa", 0.0))
        for entity in manifest["resolved_entities"].values()
    )
    contact_stiffness = max(2.0e5, 10.0 * max_modulus)
    lights = []
    for name, scale in (
        ("key", config.key_light_scale),
        ("fill", config.fill_light_scale),
    ):
        settings = manifest["lighting"][name]
        intensity = scale * float(settings["power_w"])
        if config.light_model == "point":
            positions = [np.asarray(settings["position_m"], dtype=float)]
            if config.approximate_area_lights:
                direction = np.asarray(settings["target_m"], dtype=float) - positions[0]
                direction /= np.linalg.norm(direction)
                basis_u = np.cross(direction, np.asarray((0.0, 0.0, 1.0)))
                if np.linalg.norm(basis_u) < 1e-6:
                    basis_u = np.cross(direction, np.asarray((0.0, 1.0, 0.0)))
                basis_u /= np.linalg.norm(basis_u)
                half_span = 0.25 * float(settings.get("size_m", 0.0))
                center = positions[0]
                positions = [center - half_span * basis_u, center + half_span * basis_u]
            for position in positions:
                lights.append(
                    gs.options.vis.PointLight(
                        pos=tuple(position),
                        color=tuple(settings["color_linear"]),
                        intensity=intensity / len(positions),
                    )
                )
        elif config.light_model == "directional":
            direction = np.asarray(settings["target_m"], dtype=float) - np.asarray(
                settings["position_m"], dtype=float
            )
            direction /= np.linalg.norm(direction)
            lights.append(
                gs.options.vis.DirectionalLight(
                    dir=tuple(direction),
                    color=tuple(settings["color_linear"]),
                    intensity=intensity,
                )
            )
        else:
            raise ValueError("--light-model must be point or directional")
    if config.coupler == "sap":
        coupler_options = gs.options.SAPCouplerOptions(
            n_sap_iterations=6,
            n_pcg_iterations=80,
            n_linesearch_iterations=6,
            hydroelastic_stiffness=contact_stiffness,
            point_contact_stiffness=contact_stiffness,
            fem_floor_contact_type="none",
            # Genesis 1.4.0 creates the shared FEM BVH only on this path; this is also useful for the touching cushions.
            enable_fem_self_tet_contact=True,
            rigid_floor_contact_type="none",
            enable_rigid_fem_contact=True,
            rigid_rigid_contact_type="tet",
        )
    elif config.coupler == "legacy":
        coupler_options = gs.options.LegacyCouplerOptions(rigid_fem=True)
    else:
        raise ValueError("--coupler must be sap or legacy")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=config.dt,
            substeps=1,
            gravity=tuple(manifest["environment"]["gravity_m_s2"]),
            floor_height=-2.0,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=config.dt,
            enable_collision=True,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
            max_collision_pairs=256,
            max_contacts=512,
            iterations=80,
            batch_links_info=heterogeneous and config.num_envs > 1,
        ),
        fem_options=gs.options.FEMOptions(
            dt=config.dt,
            use_implicit_solver=True,
            n_newton_iterations=2,
            n_pcg_iterations=120,
            n_linesearch_iterations=3,
            damping_alpha=alpha,
            damping_beta=beta,
        ),
        coupler_options=coupler_options,
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            shadow=config.render_shadows,
            background_color=(0.18, 0.18, 0.18),
            ambient_light=(config.ambient_light,) * 3,
            lights=lights,
            n_support_neighbors=16,
        ),
        renderer=gs.renderers.Rasterizer(),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=False,
    )

    physical: dict[str, Any] = {}
    visual_followers: dict[str, Any] = {}
    deformable: dict[str, Any] = {}

    for name, entity in manifest["resolved_entities"].items():
        decision = plan["entities"][name]
        if decision["role"] in {"dynamic_rigid", "static_rigid"}:
            variants = [entities[name] for entities in env_entities]
            heterogeneous_morph = (
                heterogeneous
                and decision["role"] == "dynamic_rigid"
                and config.num_envs > 1
            )
            morph = (
                [_physics_morph(gs, variant) for variant in variants]
                if heterogeneous_morph
                else _physics_morph(gs, entity)
            )
            sphere_radius = None
            if entity["collision"]["representation"] == "sphere":
                sphere_radius = max(
                    float(variant["collision"]["radius_m"]) for variant in variants
                )
            physical[name] = scene.add_entity(
                morph=morph,
                material=_rigid_material(
                    gs,
                    variants[0],
                    coupled=True,
                    coupling_softness_m=config.coupling_softness_m,
                    coupling_friction_scale=config.coupling_friction_scale,
                    coupling_restitution=use_sampled_coupling_restitution,
                    sphere_softness_radius_m=sphere_radius,
                ),
                name=f"{name}-physics",
            )
        elif decision["role"] == "deformable":
            parameters = decision["parameters"]
            tetgen = dict(decision["discretization"]["tetgen"])
            tetgen["force_retet"] = config.force_retet
            deformable[name] = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=decision["physics_asset"]["path"],
                    pos=tuple(entity["position_m"]),
                    quat=tuple(entity["quaternion_wxyz"]),
                    file_meshes_are_zup=True,
                    **tetgen,
                ),
                material=gs.materials.FEM.Elastic(
                    E=float(parameters["E"]),
                    nu=float(parameters["nu"]),
                    rho=float(parameters["rho"]),
                    friction_mu=float(parameters["friction_mu"]),
                    hydroelastic_modulus=contact_stiffness,
                    model=parameters["model"],
                ),
                surface=_surface(
                    gs, entity, deformable=True, color_space=config.surface_color_space
                ),
                name=name,
            )

    for name, entity in manifest["resolved_entities"].items():
        visual_asset = plan["entities"][name]["visual_asset"]
        if name in deformable or visual_asset is None or not config.render:
            continue
        variants = [entities[name] for entities in env_entities]
        heterogeneous_morph = (
            heterogeneous
            and plan["entities"][name]["role"] == "dynamic_rigid"
            and config.num_envs > 1
        )
        morph = (
            [
                _visual_morph(gs, Path(visual_asset["path"]), variant, batched=True)
                for variant in variants
            ]
            if heterogeneous_morph
            else _visual_morph(
                gs, Path(visual_asset["path"]), entity, batched=config.num_envs > 1
            )
        )
        visual_followers[name] = scene.add_entity(
            morph=morph,
            material=_rigid_material(gs, variants[0], coupled=False),
            surface=_surface(gs, variants[0], color_space=config.surface_color_space),
            name=f"{name}-visual",
        )

    camera_spec = manifest["camera"]
    camera = scene.add_camera(
        model="pinhole",
        res=tuple(camera_spec["resolution_px"]),
        pos=tuple(camera_spec["position_m"]),
        lookat=tuple(camera_spec["target_m"]),
        up=(0.0, 0.0, 1.0),
        fov=float(plan["camera"]["vertical_fov_deg"]),
        near=float(camera_spec["clip_m"][0]),
        far=float(camera_spec["clip_m"][1]),
        env_idx=0 if config.num_envs > 1 else None,
    )

    start = time.perf_counter()
    scene.build(
        n_envs=config.num_envs if config.num_envs > 1 else 0,
        env_spacing=(2.2, 1.8),
    )
    build_seconds = time.perf_counter() - start

    for name, entity in manifest["resolved_entities"].items():
        if (
            name in physical
            and entity.get("mass_kg") is not None
            and entity["physics_type"] == "rigid"
        ):
            if heterogeneous and config.num_envs > 1:
                physical[name].set_mass(
                    np.asarray(
                        [items[name]["mass_kg"] for items in env_entities],
                        dtype=np.float64,
                    )
                )
            else:
                physical[name].set_mass(float(entity["mass_kg"]))

    surface_indices: dict[str, np.ndarray] = {}
    source_surfaces: dict[str, np.ndarray] = {}
    tetrahedra: dict[str, np.ndarray] = {}
    reference_tet_signs: dict[str, np.ndarray] = {}
    for name, entity in deformable.items():
        indices = np.unique(
            np.concatenate([np.asarray(vgeom.sim_verts_idx) for vgeom in entity.vgeoms])
        ).astype(np.int64)
        positions = as_numpy(entity.get_state().pos)
        if positions.ndim == 3:
            positions = positions[0]
        surface_indices[name] = indices
        source_surfaces[name] = positions[indices].copy()
        elements = np.asarray(entity.elems, dtype=np.int64)
        tet = positions[elements]
        signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(tet[:, 1] - tet[:, 0], tet[:, 2] - tet[:, 0]),
            tet[:, 3] - tet[:, 0],
        )
        signs = np.sign(signed_six_volume)
        if np.any(signs == 0):
            raise MigrationError(
                f"{name}: generated FEM mesh contains zero-volume tetrahedra"
            )
        tetrahedra[name] = elements
        reference_tet_signs[name] = signs

    return RuntimeScene(
        gs=gs,
        scene=scene,
        camera=camera,
        manifest=manifest,
        plan=plan,
        physical=physical,
        visual_followers=visual_followers,
        deformable=deformable,
        surface_indices=surface_indices,
        source_surfaces=source_surfaces,
        tetrahedra=tetrahedra,
        reference_tet_signs=reference_tet_signs,
        build_seconds=build_seconds,
        env_entities=env_entities if heterogeneous else None,
    )


def set_rigid_pose(
    entity: Any, state: dict[str, Any], *, zero_velocity: bool = True
) -> None:
    entity.set_pos(state["position_m"], zero_velocity=zero_velocity)
    entity.set_quat(state["quaternion_wxyz"], zero_velocity=zero_velocity)


def sync_visual_followers(runtime: RuntimeScene) -> None:
    for name, visual in runtime.visual_followers.items():
        if name not in runtime.physical:
            continue
        physical = runtime.physical[name]
        visual.set_pos(physical.get_pos(), zero_velocity=True)
        visual.set_quat(physical.get_quat(), zero_velocity=True)


def render_frame(runtime: RuntimeScene, path: Path) -> None:
    rgb = runtime.camera.render(rgb=True, force_render=True)[0]
    image = as_numpy(rgb)
    if image.ndim == 4:
        image = image[0]
    Image.fromarray(image.astype(np.uint8)).save(path)
