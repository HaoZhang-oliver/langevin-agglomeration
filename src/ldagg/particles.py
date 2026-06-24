"""Primary spherical particle definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from ldagg.constants import DEFAULT_PARTICLE_DENSITY
from ldagg.gas import Gas, particle_mass, sphere_diffusion, sphere_friction


@dataclass(frozen=True, slots=True)
class PrimaryParticle:
    """A spherical primary particle with SI properties."""

    diameter: float
    density: float = DEFAULT_PARTICLE_DENSITY
    gas: Gas = field(default_factory=Gas)

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def mass(self) -> float:
        return particle_mass(self.diameter, self.density)

    @property
    def friction(self) -> float:
        return sphere_friction(self.diameter, self.gas)

    @property
    def diffusion(self) -> float:
        return sphere_diffusion(self.diameter, self.gas)
