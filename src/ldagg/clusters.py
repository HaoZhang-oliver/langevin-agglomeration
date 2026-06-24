"""Rigid aggregate representation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ldagg.constants import DEFAULT_PARTICLE_DENSITY
from ldagg.gas import Gas, particle_mass, sphere_friction

DEFAULT_CLUSTER_FRICTION_MODEL = "equivalent_sphere"


@dataclass(slots=True)
class Cluster:
    """A rigid translating aggregate of primary spheres.

    Primary centers are stored relative to the aggregate center of mass.
    Rotation, restructuring, sintering, and hydrodynamic interactions are not
    included in this first implementation.
    """

    cluster_id: int
    rel_positions: np.ndarray
    radii: np.ndarray
    masses: np.ndarray
    primary_frictions: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    friction_model: str = DEFAULT_CLUSTER_FRICTION_MODEL
    gas: Gas = field(default_factory=Gas)
    metadata: dict = field(default_factory=dict)
    primary_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rel_positions = np.asarray(self.rel_positions, dtype=float).reshape((-1, 3))
        self.radii = np.asarray(self.radii, dtype=float).reshape((-1,))
        self.masses = np.asarray(self.masses, dtype=float).reshape((-1,))
        self.primary_frictions = np.asarray(self.primary_frictions, dtype=float).reshape((-1,))
        self.position = np.asarray(self.position, dtype=float).reshape((3,))
        self.velocity = np.asarray(self.velocity, dtype=float).reshape((3,))
        if not (
            len(self.rel_positions)
            == len(self.radii)
            == len(self.masses)
            == len(self.primary_frictions)
        ):
            raise ValueError("primary arrays must have matching lengths")
        if self.primary_ids is None:
            self.primary_ids = np.arange(len(self.radii), dtype=np.int64)
        else:
            self.primary_ids = np.asarray(self.primary_ids, dtype=np.int64).reshape((-1,))
            if len(self.primary_ids) != len(self.radii):
                raise ValueError("primary_ids must match primary arrays")
        if self.friction_model not in {"free_draining", "equivalent_sphere"}:
            raise ValueError("friction_model must be 'free_draining' or 'equivalent_sphere'")
        self.recenter()

    @classmethod
    def monomer(
        cls,
        cluster_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        diameter: float,
        *,
        density: float = DEFAULT_PARTICLE_DENSITY,
        gas: Gas | None = None,
        friction_model: str = DEFAULT_CLUSTER_FRICTION_MODEL,
    ) -> Cluster:
        gas = gas or Gas()
        return cls(
            cluster_id=cluster_id,
            rel_positions=np.zeros((1, 3), dtype=float),
            radii=np.array([0.5 * diameter], dtype=float),
            masses=np.array([particle_mass(diameter, density)], dtype=float),
            primary_frictions=np.array([sphere_friction(diameter, gas)], dtype=float),
            position=np.asarray(position, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            friction_model=friction_model,
            gas=gas,
            metadata={"diameter": diameter, "density": density},
            primary_ids=np.array([cluster_id], dtype=np.int64),
        )

    @property
    def n_primary(self) -> int:
        return len(self.radii)

    @property
    def mass(self) -> float:
        return float(np.sum(self.masses))

    @property
    def volume_equivalent_radius(self) -> float:
        """Radius of a sphere with the same total primary-sphere volume."""

        return float(np.sum(self.radii**3) ** (1.0 / 3.0))

    @property
    def volume_equivalent_diameter(self) -> float:
        """Diameter used by the default Cunningham-corrected cluster drag model."""

        return 2.0 * self.volume_equivalent_radius

    @property
    def friction(self) -> float:
        if self.friction_model == "free_draining":
            return float(np.sum(self.primary_frictions))
        return float(sphere_friction(self.volume_equivalent_diameter, self.gas))

    @property
    def diffusion(self) -> float:
        from ldagg.constants import BOLTZMANN

        return BOLTZMANN * self.gas.temperature / self.friction

    @property
    def absolute_centers(self) -> np.ndarray:
        return self.position[None, :] + self.rel_positions

    @property
    def bounding_radius(self) -> float:
        return float(np.max(np.linalg.norm(self.rel_positions, axis=1) + self.radii))

    @property
    def radius_of_gyration(self) -> float:
        rel = self.rel_positions
        return float(np.sqrt(np.sum(self.masses * np.sum(rel * rel, axis=1)) / self.mass))

    def recenter(self) -> None:
        """Ensure relative coordinates are centered on the mass COM."""

        com_offset = np.average(self.rel_positions, axis=0, weights=self.masses)
        self.rel_positions = self.rel_positions - com_offset[None, :]
        self.position = self.position + com_offset

    def copy(self, *, cluster_id: int | None = None) -> Cluster:
        return Cluster(
            cluster_id=self.cluster_id if cluster_id is None else cluster_id,
            rel_positions=self.rel_positions.copy(),
            radii=self.radii.copy(),
            masses=self.masses.copy(),
            primary_frictions=self.primary_frictions.copy(),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            friction_model=self.friction_model,
            gas=self.gas,
            metadata=dict(self.metadata),
            primary_ids=self.primary_ids.copy(),
        )


def dimer_seed(
    cluster_id: int,
    diameter: float,
    *,
    density: float = DEFAULT_PARTICLE_DENSITY,
    gas: Gas | None = None,
    friction_model: str = DEFAULT_CLUSTER_FRICTION_MODEL,
) -> Cluster:
    """Create a two-monomer touching dimer centered at the origin."""

    gas = gas or Gas()
    mass = particle_mass(diameter, density)
    friction = sphere_friction(diameter, gas)
    return Cluster(
        cluster_id=cluster_id,
        rel_positions=np.array([[-0.5 * diameter, 0.0, 0.0], [0.5 * diameter, 0.0, 0.0]]),
        radii=np.array([0.5 * diameter, 0.5 * diameter]),
        masses=np.array([mass, mass]),
        primary_frictions=np.array([friction, friction]),
        position=np.zeros(3),
        velocity=np.zeros(3),
        friction_model=friction_model,
        gas=gas,
        metadata={"diameter": diameter, "density": density},
        primary_ids=np.array([0, 1], dtype=np.int64),
    )
