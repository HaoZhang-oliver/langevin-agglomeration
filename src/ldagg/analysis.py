"""Analysis helpers for agglomeration runs."""

from __future__ import annotations

from collections import Counter

import h5py
import numpy as np

from ldagg.clusters import Cluster


def cluster_size_counts(clusters: list[Cluster]) -> dict[int, int]:
    """Return a size-distribution count keyed by primary count."""

    return dict(Counter(cluster.n_primary for cluster in clusters))


def aggregate_table(cluster: Cluster) -> list[dict[str, float | int]]:
    """Rows describing final primary sphere positions in an aggregate."""

    centers = cluster.absolute_centers
    rows = []
    for i, (center, radius, mass) in enumerate(zip(centers, cluster.radii, cluster.masses, strict=True)):
        rows.append(
            {
                "primary": i,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "radius": float(radius),
                "mass": float(mass),
            }
        )
    return rows


def summarize_h5(path: str) -> dict:
    """Read summary attributes from a run HDF5 file."""

    with h5py.File(path, "r") as h5:
        return {key: _jsonable(value) for key, value in h5.attrs.items()}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    return value
