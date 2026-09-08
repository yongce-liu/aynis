"""Export body-local, triangulated assets without losing physics ownership."""

import copy
from pathlib import Path

import bpy
import numpy as np
import trimesh


def evaluated_mesh(obj, body_matrix):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        matrix = body_matrix.inverted() @ obj.matrix_world
        vertices = np.array(
            [tuple(matrix @ vertex.co) for vertex in mesh.vertices], dtype=np.float64
        )
        faces = np.array(
            [tuple(face.vertices) for face in mesh.loop_triangles], dtype=np.int32
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    finally:
        evaluated.to_mesh_clear()


def save_obj(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep vertex order so rest-surface vertex sets and later skin bindings remain usable.
    with path.open("w") as stream:
        stream.write(
            "# Local meters, Z up; no source video texture. Apply manifest transform once.\n"
        )
        for x, y, z in mesh.vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in mesh.faces:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")


def export_entity(name, root, spec, output):
    visual_objects = [
        obj for obj in [root, *root.children_recursive] if obj.type in ("MESH", "CURVE")
    ]
    visual_meshes = [evaluated_mesh(obj, root.matrix_world) for obj in visual_objects]
    combined = trimesh.util.concatenate(visual_meshes)
    visual_path = f"assets/visual/{name}.obj"
    save_obj(output / visual_path, combined)
    physics = spec["physics"]
    physics_mesh = (
        evaluated_mesh(root, root.matrix_world) if root.type == "MESH" else None
    )
    result = {
        "blender_object": root.name,
        "physics_type": physics["type"],
        "position_m": list(root.matrix_world.translation),
        "quaternion_wxyz": list(root.matrix_world.to_quaternion()),
        "world_matrix": [list(row) for row in root.matrix_world],
        "visual_mesh": visual_path,
        "visual_parts": [obj.name for obj in visual_objects],
        "visual_material_note": "OBJ is geometry only. Full procedural BSDFs are self-contained in scene.blend; appearance parameters are in this manifest.",
        "collision": copy.deepcopy(physics["collision"]),
        "appearance": spec["appearance"],
        "initial_state": spec["initial_state"],
        "physics_parameters": dict(physics["parameters"]),
    }
    if physics_mesh is not None:
        physics_path = f"assets/physics/{name}.obj"
        save_obj(output / physics_path, physics_mesh)
        result.update(
            physics_mesh=physics_path,
            volume_m3=float(physics_mesh.volume),
            rest_bounds_local_m=physics_mesh.bounds.tolist(),
        )
        if "density_kg_m3" in physics["parameters"]:
            mass = float(physics_mesh.volume * physics["parameters"]["density_kg_m3"])
            result["mass_kg"] = mass
            result["mass_source"] = "assumed_density_times_evaluated_closed_mesh_volume"
            root["mass_kg"] = mass
        if physics["type"] == "soft":
            groups = {}
            for group in root.vertex_groups:
                groups[group.name] = [
                    v.index
                    for v in root.data.vertices
                    if any(g.group == group.index and g.weight > 0.5 for g in v.groups)
                ]
            result["deformable"] = spec["physics"]["deformable"]
            result["vertex_sets"] = {
                name: len(indices) for name, indices in groups.items()
            }
            path = f"assets/physics/{name}_rest.npz"
            np.savez_compressed(
                output / path,
                vertices_local_m=physics_mesh.vertices,
                triangles=physics_mesh.faces,
                **{
                    name: np.asarray(indices, dtype=np.int32)
                    for name, indices in groups.items()
                },
            )
            result["rest_state_npz"] = path
    if spec["geometry"]["generator"] == "sphere":
        result["collision"]["radius_m"] = spec["geometry"]["radius_m"]
    elif spec["geometry"]["generator"] == "box":
        result["collision"]["dimensions_m"] = spec["geometry"]["dimensions_m"]
    return result
