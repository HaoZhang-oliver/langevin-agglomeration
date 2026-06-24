"""Collision detection and irreversible sticking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ldagg.boundaries import Boundary, minimum_image, wrap_position
from ldagg.clusters import Cluster


@dataclass(frozen=True, slots=True)
class CollisionInfo:
    cluster_a: int
    cluster_b: int
    primary_a: int
    primary_b: int
    distance: float
    collision_point: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    time: float
    cluster_a: int
    cluster_b: int
    size_a: int
    size_b: int
    primary_a: int
    primary_b: int
    distance: float
    collision_x: float
    collision_y: float
    collision_z: float
    new_cluster_id: int
    new_size: int

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "cluster_a": self.cluster_a,
            "cluster_b": self.cluster_b,
            "size_a": self.size_a,
            "size_b": self.size_b,
            "primary_a": self.primary_a,
            "primary_b": self.primary_b,
            "distance": self.distance,
            "collision_x": self.collision_x,
            "collision_y": self.collision_y,
            "collision_z": self.collision_z,
            "new_cluster_id": self.new_cluster_id,
            "new_size": self.new_size,
        }


def pair_displacement(
    center_a: np.ndarray,
    center_b: np.ndarray,
    boundary: Boundary | None = None,
) -> np.ndarray:
    disp = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
    if boundary is not None and boundary.mode == "periodic":
        disp = minimum_image(disp, boundary.size)
    return disp


def find_collision(
    cluster_a: Cluster,
    cluster_b: Cluster,
    *,
    boundary: Boundary | None = None,
    capture_tolerance: float = 0.0,
) -> CollisionInfo | None:
    """Return the deepest primary-sphere contact between two clusters."""

    centers_a = cluster_a.absolute_centers
    centers_b = cluster_b.absolute_centers
    best: CollisionInfo | None = None
    best_overlap = -np.inf
    for i, ca in enumerate(centers_a):
        for j, cb in enumerate(centers_b):
            disp = pair_displacement(ca, cb, boundary)
            distance = float(np.linalg.norm(disp))
            threshold = float(cluster_a.radii[i] + cluster_b.radii[j] + capture_tolerance)
            if distance <= threshold:
                normal = disp / distance if distance > 0.0 else np.array([1.0, 0.0, 0.0])
                point = ca + normal * cluster_a.radii[i]
                overlap = threshold - distance
                info = CollisionInfo(
                    cluster_a=cluster_a.cluster_id,
                    cluster_b=cluster_b.cluster_id,
                    primary_a=i,
                    primary_b=j,
                    distance=distance,
                    collision_point=point,
                    normal=normal,
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = info
    return best


def first_collision(
    clusters: list[Cluster],
    *,
    boundary: Boundary | None = None,
    capture_tolerance: float = 0.0,
) -> tuple[int, int, CollisionInfo] | None:
    """Return indices and contact info for the first cluster pair in contact."""

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            info = find_collision(
                clusters[i], clusters[j], boundary=boundary, capture_tolerance=capture_tolerance
            )
            if info is not None:
                return i, j, info
    return None


def nearest_surface_gap(
    clusters: list[Cluster],
    *,
    boundary: Boundary | None = None,
) -> float:
    """Smallest primary-sphere surface gap among all cluster pairs."""

    if len(clusters) < 2:
        return np.inf
    gap = np.inf
    for i in range(len(clusters)):
        centers_i = clusters[i].absolute_centers
        for j in range(i + 1, len(clusters)):
            centers_j = clusters[j].absolute_centers
            for pi, ci in enumerate(centers_i):
                for pj, cj in enumerate(centers_j):
                    distance = float(np.linalg.norm(pair_displacement(ci, cj, boundary)))
                    gap = min(gap, distance - clusters[i].radii[pi] - clusters[j].radii[pj])
    return float(gap)


def _absolute_centers_for_merge(
    cluster_a: Cluster,
    cluster_b: Cluster,
    boundary: Boundary | None,
) -> tuple[np.ndarray, np.ndarray]:
    centers_a = cluster_a.absolute_centers
    if boundary is not None and boundary.mode == "periodic":
        disp_ab = minimum_image(cluster_b.position - cluster_a.position, boundary.size)
        pos_b = cluster_a.position + disp_ab
        centers_b = pos_b[None, :] + cluster_b.rel_positions
    else:
        centers_b = cluster_b.absolute_centers
    return centers_a, centers_b


def merge_clusters(
    cluster_a: Cluster,
    cluster_b: Cluster,
    *,
    collision: CollisionInfo | None = None,
    new_cluster_id: int,
    time: float = 0.0,
    boundary: Boundary | None = None,
    project_to_contact: bool = True,
) -> tuple[Cluster, CollisionEvent]:
    """Irreversibly stick two clusters while conserving mass and momentum."""

    centers_a, centers_b = _absolute_centers_for_merge(cluster_a, cluster_b, boundary)
    if collision is None:
        collision = find_collision(cluster_a, cluster_b, boundary=boundary)
    if collision is not None and project_to_contact:
        ia = collision.primary_a
        ib = collision.primary_b
        disp = centers_b[ib] - centers_a[ia]
        distance = float(np.linalg.norm(disp))
        normal = collision.normal if distance == 0.0 else disp / distance
        target = float(cluster_a.radii[ia] + cluster_b.radii[ib])
        # The collision predicate can trigger either from timestep overshoot
        # (slight overlap) or from capture_tolerance (small positive gap).
        # In both cases, project the triggering primary pair to true contact.
        if not np.isclose(distance, target, rtol=0.0, atol=1.0e-18):
            centers_b = centers_b + normal[None, :] * (target - distance)

    all_centers = np.vstack([centers_a, centers_b])
    all_radii = np.concatenate([cluster_a.radii, cluster_b.radii])
    all_masses = np.concatenate([cluster_a.masses, cluster_b.masses])
    all_frictions = np.concatenate([cluster_a.primary_frictions, cluster_b.primary_frictions])
    all_primary_ids = np.concatenate([cluster_a.primary_ids, cluster_b.primary_ids])
    total_mass = float(np.sum(all_masses))
    new_position = np.average(all_centers, axis=0, weights=all_masses)
    new_velocity = (cluster_a.mass * cluster_a.velocity + cluster_b.mass * cluster_b.velocity) / total_mass
    new_cluster = Cluster(
        cluster_id=new_cluster_id,
        rel_positions=all_centers - new_position[None, :],
        radii=all_radii,
        masses=all_masses,
        primary_frictions=all_frictions,
        position=new_position,
        velocity=new_velocity,
        friction_model=cluster_a.friction_model,
        gas=cluster_a.gas,
        metadata=dict(cluster_a.metadata),
        primary_ids=all_primary_ids,
    )
    if boundary is not None and boundary.mode == "periodic":
        new_cluster.position = wrap_position(new_cluster.position, boundary)

    if collision is None:
        point = 0.5 * (cluster_a.position + cluster_b.position)
        primary_a = primary_b = -1
        distance = float(np.linalg.norm(cluster_b.position - cluster_a.position))
    else:
        point = collision.collision_point
        primary_a = collision.primary_a
        primary_b = collision.primary_b
        distance = collision.distance
    event = CollisionEvent(
        time=time,
        cluster_a=cluster_a.cluster_id,
        cluster_b=cluster_b.cluster_id,
        size_a=cluster_a.n_primary,
        size_b=cluster_b.n_primary,
        primary_a=primary_a,
        primary_b=primary_b,
        distance=distance,
        collision_x=float(point[0]),
        collision_y=float(point[1]),
        collision_z=float(point[2]),
        new_cluster_id=new_cluster_id,
        new_size=new_cluster.n_primary,
    )
    return new_cluster, event
