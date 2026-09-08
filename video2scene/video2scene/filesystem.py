"""Small filesystem primitives shared by pipeline stages."""

from __future__ import annotations

import shutil
from pathlib import Path


def remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory when it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
