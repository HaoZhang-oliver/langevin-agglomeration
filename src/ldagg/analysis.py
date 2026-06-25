"""Analysis helpers for agglomeration runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

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


def cluster_geometry_nm_array(cluster: Cluster) -> np.ndarray:
    """Return primary-sphere centers and diameters as ``x,y,z,d`` in nm."""

    centers_nm = cluster.absolute_centers * 1.0e9
    diameters_nm = (2.0 * cluster.radii * 1.0e9)[:, None]
    return np.column_stack([centers_nm, diameters_nm])


def export_cluster_geometry_nm_csv(cluster: Cluster, path: str | Path) -> Path:
    """Write ``x,y,z,diameter`` rows in nm with no header."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, cluster_geometry_nm_array(cluster), delimiter=",", fmt="%.10g")
    return out


def summarize_h5(path: str) -> dict:
    """Read summary attributes from a run HDF5 file."""

    with h5py.File(path, "r") as h5:
        return {key: _jsonable(value) for key, value in h5.attrs.items()}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    return value
