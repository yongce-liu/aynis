"""Filesystem and media artifacts produced by Genesis simulation runs."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .filesystem import remove_path
from .simulation_config import SimulationRunConfig
from .spec import read_json


def prepare_simulation_output(output: Path, overwrite: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    known_files = (
        "migration_plan.json",
        "run_manifest.json",
        "trace.json",
        "trace.npz",
        "validation.json",
        "preview.png",
        "comparison.jpg",
        "contact_sheet.jpg",
        "source_simulation_grid.jpg",
        "simulation.mp4",
        "render_review.json",
        "run.log",
    )
    existing = [output / name for name in known_files if (output / name).exists()]
    frames_dir = output / "frames"
    if (existing or frames_dir.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already contains Phase-2 artifacts: {output}. Pass --overwrite to replace them."
        )
    for path in existing:
        remove_path(path)
    remove_path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


def encode_video(frames_dir: Path, output: Path, fps: int) -> dict[str, Any]:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(probe.stdout)["streams"][0]


def _label_image(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(20, 24, 27))
    draw.text((8, 6), label, fill=(245, 245, 245))
    return image


def source_reference_frame(config: SimulationRunConfig) -> Path:
    manifest = read_json(config.blender_dir / "scene_manifest.json")
    frame_index = int(manifest["source"].get("reference_frame", 0))
    frame = config.source_frames_dir / f"frame_{frame_index:04d}.png"
    if not frame.is_file():
        raise FileNotFoundError(
            f"Missing canonical source reference frame: {frame}. "
            "Run video2scene.evidence or video2scene.build first."
        )
    return frame


def validate_simulation_inputs(config: SimulationRunConfig) -> None:
    """Fail before simulation when the canonical Phase-1 inputs are incomplete."""
    required = (
        config.blender_dir / "scene_manifest.json",
        config.blender_dir / "scene_spec.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required Phase-1 artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    if config.render:
        preview = config.blender_dir / "preview.png"
        if not preview.is_file():
            missing.append(preview)
        source_reference_frame(config)
    if missing:
        raise FileNotFoundError(
            "Missing required Phase-1 artifacts: "
            + ", ".join(str(path) for path in missing)
        )


def write_comparison(config: SimulationRunConfig) -> None:
    manifest = read_json(config.blender_dir / "scene_manifest.json")
    frame_index = int(manifest["source"].get("reference_frame", 0))
    source = source_reference_frame(config)
    blender = config.blender_dir / "preview.png"
    genesis = config.sim_dir / "preview.png"
    images = [
        _label_image(Image.open(source), f"SOURCE | frame {frame_index}"),
        _label_image(Image.open(blender), "BLENDER | Phase 1"),
        _label_image(Image.open(genesis), "GENESIS | simulated t=0"),
    ]
    height = max(image.height for image in images)
    canvas = Image.new(
        "RGB", (sum(image.width for image in images), height), (24, 24, 24)
    )
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    canvas.save(config.sim_dir / "comparison.jpg", quality=95)


def _chosen_frame_indices(config: SimulationRunConfig, times: list[float]) -> list[int]:
    desired = [0.0, 0.1, 0.2, 0.4, min(0.7, config.duration_s), config.duration_s]
    chosen: list[int] = []
    for target in desired:
        index = min(range(len(times)), key=lambda i: abs(times[i] - target))
        if index not in chosen:
            chosen.append(index)
    return chosen


def write_contact_sheet(
    config: SimulationRunConfig, frame_paths: list[Path], times: list[float]
) -> None:
    if not frame_paths:
        return
    chosen = _chosen_frame_indices(config, times)
    tiles = [
        _label_image(Image.open(frame_paths[index]), f"t={times[index]:.3f}s")
        for index in chosen
    ]
    width = max(image.width for image in tiles)
    height = max(image.height for image in tiles)
    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (24, 24, 24))
    for index, image in enumerate(tiles):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(config.sim_dir / "contact_sheet.jpg", quality=94)


def write_source_simulation_grid(
    config: SimulationRunConfig, frame_paths: list[Path], times: list[float]
) -> None:
    """Write matched source/simulation frames for held-out visual review."""
    if not frame_paths:
        return
    manifest = read_json(config.blender_dir / "scene_manifest.json")
    source_fps = float(manifest["source"]["fps"])
    pairs = []
    for index in _chosen_frame_indices(config, times):
        source_index = round(times[index] * source_fps)
        source_path = config.source_frames_dir / f"frame_{source_index:04d}.png"
        if source_path.is_file():
            pairs.append((source_index, index))
    if not pairs:
        return

    tile_size = (320, 180)
    row_height = tile_size[1] + 24
    canvas = Image.new("RGB", (tile_size[0] * len(pairs), row_height * 2), (24, 24, 24))
    for column, (source_index, simulation_index) in enumerate(pairs):
        source = Image.open(
            config.source_frames_dir / f"frame_{source_index:04d}.png"
        ).convert("RGB")
        simulation = Image.open(frame_paths[simulation_index]).convert("RGB")
        source.thumbnail(tile_size, Image.Resampling.LANCZOS)
        simulation.thumbnail(tile_size, Image.Resampling.LANCZOS)
        x = column * tile_size[0]
        canvas.paste(
            _label_image(
                source,
                f"SOURCE | f={source_index:04d} t={source_index / source_fps:.3f}s",
            ),
            (x, 0),
        )
        canvas.paste(
            _label_image(simulation, f"GENESIS | t={times[simulation_index]:.3f}s"),
            (x, row_height),
        )
    canvas.save(config.sim_dir / "source_simulation_grid.jpg", quality=94)
