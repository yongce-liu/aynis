"""Typed configuration for nominal and batched Genesis runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from .layout import DEFAULT_SCENE_ROOT, SceneLayout


@dataclass(frozen=True, kw_only=True)
class SimulationOptions:
    """Physics and rendering options shared by all simulation entry points."""

    backend: Literal["gpu", "cpu"] = "gpu"
    coupler: Literal["legacy", "sap"] = "legacy"
    duration_s: float = 1.0
    relax_s: float = 0.0125
    fps: int = 30
    simulation_hz: int = 1920
    num_envs: int = 1
    seed: int = 0
    render: bool = True
    overwrite: bool = False
    force_retet: bool = False
    damping_scale: float = 2.5
    coupling_softness_m: float = 0.015
    coupling_friction_scale: float = 1.0
    light_model: Literal["point", "directional"] = "point"
    ambient_light: float = 0.85
    key_light_scale: float = 0.35
    fill_light_scale: float = 0.45
    render_shadows: bool = False
    surface_color_space: Literal["linear", "srgb"] = "linear"
    approximate_area_lights: bool = False

    @property
    def dt(self) -> float:
        return 1.0 / self.simulation_hz


@dataclass(frozen=True, kw_only=True)
class SimulationRunConfig(SimulationOptions):
    """Resolved input and output directories for one Genesis run."""

    blender_dir: Path
    source_frames_dir: Path
    sim_dir: Path


@dataclass(frozen=True, kw_only=True)
class SimulationCommandConfig(SimulationOptions):
    """Run Phase 2 inside one canonical scene output root."""

    output_root: Path = DEFAULT_SCENE_ROOT


def resolve_simulation_command(config: SimulationCommandConfig) -> SimulationRunConfig:
    """Map the user-facing root to explicit stage paths."""
    layout = SceneLayout(config.output_root.resolve())
    layout.prepare()
    options = {
        field.name: getattr(config, field.name) for field in fields(SimulationOptions)
    }
    return SimulationRunConfig(
        blender_dir=layout.blender,
        source_frames_dir=layout.frames,
        sim_dir=layout.sim,
        **options,
    )


def validate_simulation_options(config: SimulationOptions) -> None:
    """Reject invalid timing and contact settings before touching output data."""
    if config.duration_s <= 0 or config.relax_s < 0:
        raise ValueError("Durations must be positive, with non-negative relaxation")
    if config.fps <= 0 or config.simulation_hz <= 0:
        raise ValueError("Simulation and record rates must be positive")
    if config.simulation_hz % config.fps:
        raise ValueError("simulation_hz must be divisible by the output frame rate")
    if config.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if config.damping_scale < 0:
        raise ValueError("damping_scale must be non-negative")
    if config.coupling_softness_m < 0 or config.coupling_friction_scale < 0:
        raise ValueError("Coupling softness and friction scale must be non-negative")
