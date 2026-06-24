"""Gas properties and Cunningham-corrected sphere friction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ldagg.constants import (
    BOLTZMANN,
    DEFAULT_CUNNINGHAM_C1,
    DEFAULT_CUNNINGHAM_C2,
    DEFAULT_CUNNINGHAM_C3,
    DEFAULT_GAS_MEAN_FREE_PATH,
    DEFAULT_GAS_VISCOSITY,
    DEFAULT_PRESSURE,
    DEFAULT_TEMPERATURE,
)


@dataclass(slots=True)
class Gas:
    """Gas state used by the LD model.

    Parameters are SI values. The Cunningham constants match the
    Suresh-Gopalakrishnan MATLAB settling example by default.
    """

    temperature: float = DEFAULT_TEMPERATURE
    pressure: float = DEFAULT_PRESSURE
    viscosity: float = DEFAULT_GAS_VISCOSITY
    mean_free_path: float = DEFAULT_GAS_MEAN_FREE_PATH
    cunningham_c1: float = DEFAULT_CUNNINGHAM_C1
    cunningham_c2: float = DEFAULT_CUNNINGHAM_C2
    cunningham_c3: float = DEFAULT_CUNNINGHAM_C3

    @classmethod
    def from_mapping(cls, data: dict | None) -> Gas:
        if data is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict:
        return asdict(self)


def cunningham_correction(diameter: float | np.ndarray, gas: Gas | None = None) -> float | np.ndarray:
    """Return Cunningham slip correction for a sphere diameter."""

    gas = gas or Gas()
    diameter = np.asarray(diameter, dtype=float)
    radius = 0.5 * diameter
    if np.any(radius <= 0):
        raise ValueError("particle diameter must be positive")
    kn = gas.mean_free_path / radius
    cc = 1.0 + kn * (
        gas.cunningham_c1 + gas.cunningham_c2 * np.exp(-gas.cunningham_c3 / kn)
    )
    return float(cc) if cc.ndim == 0 else cc


def particle_mass(diameter: float | np.ndarray, density: float) -> float | np.ndarray:
    """Mass of a spherical particle, kg."""

    diameter = np.asarray(diameter, dtype=float)
    if np.any(diameter <= 0):
        raise ValueError("particle diameter must be positive")
    mass = density * (np.pi / 6.0) * diameter**3
    return float(mass) if mass.ndim == 0 else mass


def sphere_friction(diameter: float | np.ndarray, gas: Gas | None = None) -> float | np.ndarray:
    """Cunningham-corrected scalar friction factor ``f = 3*pi*mu*d/Cc``."""

    gas = gas or Gas()
    diameter = np.asarray(diameter, dtype=float)
    cc = cunningham_correction(diameter, gas)
    friction = 3.0 * np.pi * gas.viscosity * diameter / cc
    return float(friction) if friction.ndim == 0 else friction


def sphere_diffusion(diameter: float | np.ndarray, gas: Gas | None = None) -> float | np.ndarray:
    """Stokes-Einstein diffusivity ``D = k_B*T/f`` for a Cunningham sphere."""

    gas = gas or Gas()
    friction = sphere_friction(diameter, gas)
    diffusion = BOLTZMANN * gas.temperature / friction
    return float(diffusion) if np.ndim(diffusion) == 0 else diffusion
