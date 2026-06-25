"""Physical constants and default tutorial parameters."""

from __future__ import annotations

import numpy as np

BOLTZMANN: float = 1.380649e-23
"""Boltzmann constant, J/K."""

EPS0: float = 8.8541878128e-12
"""Vacuum permittivity, F/m."""

DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
"""Default gravitational acceleration vector, m/s^2."""

DEFAULT_PARTICLE_DENSITY: float = 1000.0
"""Default particle density, kg/m^3."""

DEFAULT_TEMPERATURE: float = 300.0
DEFAULT_PRESSURE: float = 101_325.0
DEFAULT_GAS_VISCOSITY: float = 1.8258e-5
DEFAULT_GAS_MEAN_FREE_PATH: float = 66.7e-9
DEFAULT_CUNNINGHAM_C1: float = 1.257
DEFAULT_CUNNINGHAM_C2: float = 0.4
DEFAULT_CUNNINGHAM_C3: float = 1.1
