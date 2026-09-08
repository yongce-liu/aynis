"""Loguru helpers for durable per-stage logs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from loguru import logger


@contextmanager
def stage_log(path: Path, *, mode: str = "w") -> Iterator[None]:
    """Mirror Loguru records to a deterministic stage-owned log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        path,
        mode=mode,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )
    try:
        yield
    except Exception:
        logger.exception("Stage failed")
        raise
    finally:
        logger.remove(sink_id)
