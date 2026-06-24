from __future__ import annotations

import numpy as np

from ldagg.clusters import Cluster
from ldagg.collisions import find_collision, merge_clusters
from ldagg.gas import sphere_friction


def monomer(cluster_id: int, x: float, velocity: np.ndarray | None = None) -> Cluster:
    return Cluster.monomer(
        cluster_id,
        np.array([x, 0.0, 0.0]),
        np.zeros(3) if velocity is None else velocity,
        100.0e-9,
    )


def test_collision_contact_and_noncontact() -> None:
    a = monomer(0, 0.0)
    b = monomer(1, 100.0e-9)
    c = monomer(2, 101.0e-9)
    assert find_collision(a, b) is not None
    assert find_collision(a, c) is None


def test_merge_conserves_mass_and_momentum() -> None:
    a = monomer(0, 0.0, np.array([1.0, 0.0, 0.0]))
    b = monomer(1, 99.0e-9, np.array([-0.5, 0.0, 0.0]))
    total_mass = a.mass + b.mass
    total_momentum = a.mass * a.velocity + b.mass * b.velocity
    info = find_collision(a, b)
    merged, event = merge_clusters(a, b, collision=info, new_cluster_id=3, project_to_contact=True)
    assert np.isclose(merged.mass, total_mass)
    assert np.allclose(merged.mass * merged.velocity, total_momentum)
    assert merged.n_primary == 2
    assert merged.friction_model == "equivalent_sphere"
    assert np.isclose(merged.friction, sphere_friction(merged.volume_equivalent_diameter, merged.gas))
    distance = np.linalg.norm(merged.absolute_centers[1] - merged.absolute_centers[0])
    assert distance >= merged.radii[0] + merged.radii[1] - 1.0e-15
    assert event.new_size == 2


def test_merge_projects_capture_tolerance_gap_to_contact() -> None:
    a = monomer(0, 0.0)
    b = monomer(1, 104.0e-9)
    info = find_collision(a, b, capture_tolerance=5.0e-9)
    assert info is not None
    merged, _event = merge_clusters(a, b, collision=info, new_cluster_id=4, project_to_contact=True)
    distance = np.linalg.norm(merged.absolute_centers[1] - merged.absolute_centers[0])
    assert np.isclose(distance, merged.radii[0] + merged.radii[1], atol=1.0e-15)
