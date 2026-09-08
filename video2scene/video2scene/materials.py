"""Procedural, source-independent materials in scene metric coordinates."""

from __future__ import annotations

from typing import Any

import bpy


def _shader_input(shader: Any, *names: str) -> Any | None:
    """Return the first available Principled input across Blender versions."""
    for name in names:
        value = shader.inputs.get(name)
        if value is not None:
            return value
    return None


def material(
    name,
    color,
    roughness=0.5,
    kind="plain",
    seed=0,
    *,
    metallic=None,
    transmission_weight=0.0,
    ior=1.45,
    alpha=1.0,
):
    """Create a procedural Cycles material without source-image textures."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*color, alpha)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    metallic_input = _shader_input(shader, "Metallic")
    if metallic_input is not None:
        metallic_input.default_value = (
            0.75 if metallic is None and kind == "metal" else float(metallic or 0.0)
        )
    transmission_input = _shader_input(shader, "Transmission Weight", "Transmission")
    if transmission_input is not None:
        transmission_input.default_value = transmission_weight
    ior_input = _shader_input(shader, "IOR")
    if ior_input is not None:
        ior_input.default_value = ior
    alpha_input = _shader_input(shader, "Alpha")
    if alpha_input is not None:
        alpha_input.default_value = alpha

    if kind == "glass":
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = roughness
        if transmission_input is not None:
            transmission_input.default_value = max(0.9, transmission_weight)
        return mat

    tex = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = {
        "fabric": 18,
        "leather": 220,
        "rubber": 70,
        "plastic": 110,
        "wall": 3,
        "wood": 7,
    }.get(kind, 90)
    noise.inputs["Detail"].default_value = 3
    noise.noise_dimensions = "4D"
    noise.inputs["W"].default_value = seed
    links.new(tex.outputs["Object"], noise.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    spread = {
        "wall": 0.065,
        "fabric": 0.12,
        "leather": 0.30,
        "rubber": 0.09,
        "plastic": 0.045,
        "wood": 0.18,
    }.get(kind, 0.04)
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[0].color = (*(c * (1 - spread) for c in color), 1)
    ramp.color_ramp.elements[1].color = (
        *(min(1, c * (1 + spread)) for c in color),
        1,
    )

    if kind == "wood":
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.inputs["Scale"].default_value = 26
        wave.inputs["Distortion"].default_value = 3.5
        wave.inputs["Detail"].default_value = 4
        links.new(tex.outputs["Object"], wave.inputs["Vector"])
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 0.34
        links.new(noise.outputs["Fac"], mix.inputs[1])
        links.new(wave.outputs["Color"], mix.inputs[2])
        links.new(mix.outputs["Color"], ramp.inputs[0])
    else:
        links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = {
        "rubber": 0.12,
        "plastic": 0.05,
        "wood": 0.18,
    }.get(kind, 0.23)
    bump.inputs["Distance"].default_value = {
        "fabric": 0.00035,
        "leather": 0.00028,
        "rubber": 0.00012,
        "plastic": 0.00004,
        "wall": 0.0006,
        "wood": 0.00035,
    }.get(kind, 0.00007)
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    if kind == "fabric":
        sheen = _shader_input(shader, "Sheen Weight", "Sheen")
        if sheen is not None:
            sheen.default_value = 0.23
        waves = []
        for direction in ("X", "Y"):
            wave = nodes.new("ShaderNodeTexWave")
            wave.bands_direction = direction
            wave.inputs["Scale"].default_value = 780
            wave.inputs["Distortion"].default_value = 0.7
            links.new(tex.outputs["Object"], wave.inputs["Vector"])
            waves.append(wave)
        mul = nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        links.new(waves[0].outputs["Color"], mul.inputs[0])
        links.new(waves[1].outputs["Color"], mul.inputs[1])
        links.new(mul.outputs[0], bump.inputs["Height"])
    elif kind == "leather":
        pores = nodes.new("ShaderNodeTexVoronoi")
        pores.inputs["Scale"].default_value = 620
        links.new(tex.outputs["Object"], pores.inputs["Vector"])
        links.new(pores.outputs["Distance"], bump.inputs["Height"])
    else:
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
    return mat
