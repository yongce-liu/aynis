"""Metric, closed geometry with separate visual detail and physics ownership."""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def move_to(obj, collection):
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def finish(obj, name, mat, collection):
    obj.name = name
    if mat:
        obj.data.materials.append(mat)
    move_to(obj, collection)
    return obj


def cube(name, dimensions, position, mat, collection, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = finish(bpy.context.object, name, mat, collection)
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new("Manufactured edge radius", "BEVEL")
        modifier.width = bevel
        modifier.segments = 4
        modifier = obj.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
        modifier.keep_sharp = True
    return obj


def curve(name, points, radius, mat, collection, cyclic=False, parent=None):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 2
    data.bevel_depth = radius
    data.bevel_resolution = 3
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj["physics_type"] = "visual_detail"
        obj["physics_owner"] = parent.name
    return obj


def signed_power(value, exponent):
    return math.copysign(abs(value) ** exponent, value)


def cushion_point(theta, phi, dims, wrinkles, seed, roundness=0.56):
    a, b, c = (d / 2 for d in dims)
    ct = max(0, math.cos(theta)) ** 0.78
    x = a * ct * signed_power(math.cos(phi), roundness)
    y = b * ct * signed_power(math.sin(phi), roundness)
    z = c * signed_power(math.sin(theta), 0.78)
    # A sewn perimeter gently pulls the filling inward.
    cinch = 1 - 0.012 * math.exp(-((theta / 0.075) ** 2))
    x *= cinch
    y *= cinch
    if theta > 0:
        u, v = x / a, y / b
        # Local gathers radiate from the sewn edges; avoid a uniformly corrugated rim.
        side = math.exp(-(((abs(u) - 0.82) / 0.20) ** 2)) * math.exp(
            -(((v - 0.48) / 0.48) ** 2)
        )
        back = math.exp(-(((v - 0.80) / 0.20) ** 2)) * (
            0.25 + 0.75 * math.sin(3 * u + seed) ** 6
        )
        folds = side * math.sin(45 * v + 8 * u + seed) + back * math.sin(
            51 * u + 10 * v + seed
        )
        z += wrinkles * folds * math.sin(theta) ** 0.6 * min(1, ct * 5)
        z += wrinkles * 0.22 * math.sin(5 * u + seed) * math.sin(4 * v + 1) * ct
    return (x, y, z)


def cushion(name, params, position, mat, seam_mat, collection, detail_collection):
    nphi, ntheta = 160, 72
    dims, amplitude, seed = (
        params["dimensions_m"],
        params["wrinkle_amplitude_m"],
        params["wrinkle_seed"],
    )
    vertices = [(0, 0, -dims[2] / 2)]
    for j in range(1, ntheta):
        theta = -math.pi / 2 + math.pi * j / ntheta
        for i in range(nphi):
            vertices.append(
                cushion_point(
                    theta,
                    2 * math.pi * i / nphi,
                    dims,
                    amplitude,
                    seed,
                    params.get("roundness", 0.56),
                )
            )
    vertices.append((0, 0, dims[2] / 2))
    faces = []
    for i in range(nphi):
        faces.append((0, 1 + (i + 1) % nphi, 1 + i))
    for j in range(ntheta - 2):
        for i in range(nphi):
            a = 1 + j * nphi + i
            b = 1 + j * nphi + (i + 1) % nphi
            faces.append((a, b, b + nphi, a + nphi))
    top = len(vertices) - 1
    last = 1 + (ntheta - 2) * nphi
    for i in range(nphi):
        faces.append((last + i, last + (i + 1) % nphi, top))
    mesh = bpy.data.meshes.new(name + "_rest_surface")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = position
    mesh.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    # These vertex sets are selectors, not silently enabled fixation constraints.
    for group_name, predicate in [
        ("support_candidate", lambda p: p[2] < -dims[2] * 0.40),
        ("contact_surface", lambda p: p[2] > dims[2] * 0.20),
        ("seam_binding", lambda p: abs(p[2]) < dims[2] * 0.035),
    ]:
        group = obj.vertex_groups.new(name=group_name)
        indices = [i for i, p in enumerate(vertices) if predicate(p)]
        if indices:
            group.add(indices, 1.0, "REPLACE")
    points = [
        cushion_point(
            0, 2 * math.pi * i / 320, dims, 0, seed, params.get("roundness", 0.56)
        )
        for i in range(320)
    ]
    piping = curve(
        name + "__piping",
        points,
        dims[2] * 0.0047,
        seam_mat,
        detail_collection,
        True,
        obj,
    )
    piping["deformation_binding"] = (
        "Rebind to the soft owner surface; never a separate rigid collider."
    )
    return obj


def ball(
    name,
    radius,
    position,
    mat,
    detail_mat,
    collection,
    detail_collection,
    surface_detail=None,
):
    """Create a smooth sphere with optional source-observed great-circle markings."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96, ring_count=64, radius=radius, location=position
    )
    obj = finish(bpy.context.object, name, mat, collection)
    for face in obj.data.polygons:
        face.use_smooth = True

    detail = surface_detail or {
        "type": "great_circle_bands",
        "angles_deg": [-25.8, 77.3],
        "normal_z": 0.18,
        "radius_fraction": 0.012,
    }
    if detail.get("type") == "none":
        return obj
    if detail.get("type") != "great_circle_bands":
        raise ValueError(f"Unsupported sphere surface detail: {detail.get('type')!r}")
    for index, angle_deg in enumerate(detail.get("angles_deg", [])):
        angle = math.radians(float(angle_deg))
        normal = Vector(
            (math.cos(angle), math.sin(angle), float(detail.get("normal_z", 0.18)))
        ).normalized()
        a = normal.cross(Vector((0, 0, 1))).normalized()
        b = normal.cross(a).normalized()
        points = []
        for parameter in np.linspace(0, 2 * math.pi, 256, endpoint=False):
            point = (a * math.cos(parameter) + b * math.sin(parameter)) * (
                radius * 0.9989
            )
            points.append(tuple(point))
        curve(
            name + f"__marking_{index}",
            points,
            radius * float(detail.get("radius_fraction", 0.012)),
            detail_mat,
            detail_collection,
            True,
            obj,
        )
    return obj


def clamp(name, position, mat, metal, collection, scale=1.0):
    root = bpy.data.objects.new(name, None)
    collection.objects.link(root)
    root.location = position
    root["physics_type"] = "static"
    root["semantic_role"] = "release_fixture_inferred_shape"
    # Curved molded handle and upper steel jaw are separate, editable geometry.
    path = [
        (-0.135, 0, -0.045),
        (-0.123, 0, -0.015),
        (-0.09, 0, 0.013),
        (-0.045, 0, 0.033),
        (0.025, 0, 0.042),
        (0.072, 0, 0.035),
        (0.093, 0, 0.012),
        (0.095, 0, -0.037),
    ]
    path = [tuple(v * scale for v in p) for p in path]
    handle = curve(
        name + "__rubber_handle", path, 0.020 * scale, mat, collection, parent=root
    )
    handle.data.bevel_resolution = 5
    arm = cube(
        name + "__steel_arm",
        tuple(v * scale for v in (0.16, 0.039, 0.035)),
        tuple(v * scale for v in (0, 0.02, 0.075)),
        metal,
        collection,
        0.007 * scale,
    )
    arm.parent = root
    arm.rotation_euler[1] = -0.38
    arm["physics_type"] = "visual_detail"
    arm["physics_owner"] = name
    jaw = cube(
        name + "__jaw",
        tuple(v * scale for v in (0.032, 0.061, 0.097)),
        tuple(v * scale for v in (0.092, 0.017, 0.025)),
        metal,
        collection,
        0.006 * scale,
    )
    jaw.parent = root
    jaw.rotation_euler[1] = -0.18
    jaw["physics_type"] = "visual_detail"
    jaw["physics_owner"] = name
    return root
