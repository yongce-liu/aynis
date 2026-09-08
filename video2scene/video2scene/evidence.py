"""Extract source evidence without using video pixels as scene textures."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Annotated

from PIL import Image, ImageDraw
import tyro
from loguru import logger

from .filesystem import remove_path
from .hashing import sha256_file
from .layout import resolve_scene_layout


@dataclass(frozen=True)
class EvidenceConfig:
    """Extract source-video evidence for reconstruction."""

    video: Annotated[Path, tyro.conf.Positional]
    output_root: Path | None = None


def probe(video: Path) -> dict:
    """Read source video metadata without extracting image evidence."""
    video = video.resolve()
    result = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    stream = result["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    count = int(
        stream.get("nb_frames", round(float(result["format"]["duration"]) * fps))
    )
    return {
        "source_file": video.name,
        "sha256": sha256_file(video),
        "width": stream["width"],
        "height": stream["height"],
        "fps": fps,
        "frame_count": count,
        "duration_s": float(result["format"]["duration"]),
        "frames": [],
        "sampling": "Metadata only; no source frames extracted for a simulation handoff build.",
    }


def extract(video: Path, frames_dir: Path) -> dict:
    video = video.resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("sample_*.png", "frame_*.png"):
        for stale in frames_dir.glob(pattern):
            remove_path(stale)
    for name in ("evidence_sheet.jpg", "video_metadata.json"):
        remove_path(frames_dir / name)
    metadata = probe(video)
    count = metadata["frame_count"]
    width, height = metadata["width"], metadata["height"]
    probe_result = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    timestamps = probe_result["frames"]
    indices = sorted(
        set(
            list(range(min(16, count)))
            + [
                min(count - 1, round((count - 1) * t))
                for t in (0.16, 0.27, 0.5, 0.75, 1.0)
            ]
        )
    )
    selection = "+".join(f"eq(n\\,{i})" for i in indices)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "vfr",
            "-start_number",
            "0",
            str(frames_dir / "sample_%04d.png"),
        ],
        check=True,
    )
    rows = (len(indices) + 3) // 4
    sheet = Image.new("RGB", (4 * width, rows * (height + 28)), "#202326")
    draw = ImageDraw.Draw(sheet)
    records = []
    for k, index in enumerate(indices):
        frame_path = frames_dir / f"frame_{index:04d}.png"
        (frames_dir / f"sample_{k:04d}.png").replace(frame_path)
        time_s = float(timestamps[index]["best_effort_timestamp_time"])
        records.append({"index": index, "time_s": time_s, "file": frame_path.name})
        x, y = (k % 4) * width, (k // 4) * (height + 28)
        with Image.open(frame_path) as im:
            sheet.paste(im, (x, y))
        draw.text(
            (x + 9, y + height + 7),
            f"Frame {index:03d}  |  {time_s:.3f} s",
            fill="white",
        )
    sheet.save(frames_dir / "evidence_sheet.jpg", quality=93)
    metadata.update(
        frames=records,
        sampling="Dense first 16 frames plus full-clip coverage; timestamps from source PTS.",
    )
    (frames_dir / "video_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main():
    args = tyro.cli(EvidenceConfig)
    layout = resolve_scene_layout(args.video.stem, args.output_root)
    layout.prepare()
    logger.info("{}", json.dumps(extract(args.video, layout.frames), indent=2))


if __name__ == "__main__":
    main()
