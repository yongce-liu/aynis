"""Geometry helpers shared by nominal and batched simulation validation."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion to a 3x3 rotation matrix."""
    return Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    ).as_matrix()


def rigid_bottom_z(
    entity: dict[str, Any], position: np.ndarray, quaternion: np.ndarray
) -> float:
    """Return the world-space lowest point of a supported analytic rigid shape."""
    collision = entity["collision"]
    if collision["representation"] == "sphere":
        return float(position[2] - collision["radius_m"])
    half = np.asarray(collision["dimensions_m"], dtype=float) / 2.0
    z_extent = float(np.abs(quaternion_matrix_wxyz(quaternion)[2]) @ half)
    return float(position[2] - z_extent)


def world_bounds(entity: dict[str, Any]) -> np.ndarray:
    """Return a conservative world-space AABB from exported local bounds."""
    local = np.asarray(entity["rest_bounds_local_m"], dtype=float)
    corners = np.asarray(
        list(product(*zip(local[0], local[1], strict=True))), dtype=float
    )
    rotation = quaternion_matrix_wxyz(
        np.asarray(entity["quaternion_wxyz"], dtype=float)
    )
    position = np.asarray(entity["position_m"], dtype=float)
    world = corners @ rotation.T + position
    return np.stack((world.min(axis=0), world.max(axis=0)))


def horizontal_half_extent(entity: dict[str, Any]) -> np.ndarray:
    """Return conservative X/Y half extents for an analytic rigid body."""
    collision = entity["collision"]
    if collision["representation"] == "sphere":
        return np.repeat(float(collision["radius_m"]), 2)
    half = np.asarray(collision["dimensions_m"], dtype=float) / 2.0
    rotation = quaternion_matrix_wxyz(
        np.asarray(entity["quaternion_wxyz"], dtype=float)
    )
    return np.abs(rotation[:2]) @ half
