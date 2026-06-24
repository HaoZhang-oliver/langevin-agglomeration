"""Boundary conditions and minimum-image geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Boundary:
    """Periodic or finite cubic/rectangular boundary."""

    mode: str = "periodic"
    box_size: float | tuple[float, float, float] = 1.0
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None
    absorbing_floor: bool = False
    floor_z: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"periodic", "finite", "none"}:
            raise ValueError("boundary mode must be 'periodic', 'finite', or 'none'")

    @property
    def size(self) -> np.ndarray:
        if np.isscalar(self.box_size):
            return np.full(3, float(self.box_size))
        return np.asarray(self.box_size, dtype=float)

    @property
    def low(self) -> np.ndarray:
        if self.bounds_min is not None:
            return np.asarray(self.bounds_min, dtype=float)
        return np.zeros(3)

    @property
    def high(self) -> np.ndarray:
        if self.bounds_max is not None:
            return np.asarray(self.bounds_max, dtype=float)
        return self.low + self.size


def minimum_image(
    displacement: np.ndarray,
    box_size: float | np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Apply the 3D minimum-image convention to a displacement vector."""

    size = np.full(3, float(box_size)) if np.isscalar(box_size) else np.asarray(box_size, dtype=float)
    disp = np.asarray(displacement, dtype=float)
    return disp - size * np.round(disp / size)


def wrap_position(position: np.ndarray, boundary: Boundary) -> np.ndarray:
    """Wrap a COM position into a periodic box."""

    if boundary.mode != "periodic":
        return np.asarray(position, dtype=float)
    low = boundary.low
    size = boundary.size
    return low + np.mod(np.asarray(position, dtype=float) - low, size)


def apply_boundary_to_cluster(cluster, boundary: Boundary):
    """Apply periodic wrapping or finite reflecting boundaries to a cluster."""

    if boundary.mode == "none":
        return cluster, False
    if boundary.mode == "periodic":
        cluster.position = wrap_position(cluster.position, boundary)
        return cluster, False

    absorbed = False
    low = boundary.low
    high = boundary.high
    centers = cluster.absolute_centers
    radii = cluster.radii
    for axis in range(3):
        surface_min = float(np.min(centers[:, axis] - radii))
        surface_max = float(np.max(centers[:, axis] + radii))
        if axis == 2 and boundary.absorbing_floor and surface_min <= boundary.floor_z:
            absorbed = True
        if surface_min < low[axis]:
            cluster.position[axis] += low[axis] - surface_min
            if cluster.velocity[axis] < 0.0:
                cluster.velocity[axis] *= -1.0
            centers = cluster.absolute_centers
        if surface_max > high[axis]:
            cluster.position[axis] -= surface_max - high[axis]
            if cluster.velocity[axis] > 0.0:
                cluster.velocity[axis] *= -1.0
            centers = cluster.absolute_centers
    return cluster, absorbed
