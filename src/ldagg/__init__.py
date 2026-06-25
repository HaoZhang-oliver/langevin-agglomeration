"""Langevin Dynamics aerosol agglomeration."""

from ldagg.constants import BOLTZMANN, DEFAULT_GRAVITY, EPS0
from ldagg.diagnostics import DiagnosticsConfig
from ldagg.electric import ClusterDipoleResult, ElectricFieldConfig, MaterialProperties
from ldagg.gas import Gas, particle_mass, sphere_diffusion, sphere_friction
from ldagg.particles import PrimaryParticle

__all__ = [
    "BOLTZMANN",
    "ClusterDipoleResult",
    "DEFAULT_GRAVITY",
    "DiagnosticsConfig",
    "EPS0",
    "ElectricFieldConfig",
    "Gas",
    "MaterialProperties",
    "PrimaryParticle",
    "particle_mass",
    "sphere_diffusion",
    "sphere_friction",
]

__version__ = "0.1.0"
