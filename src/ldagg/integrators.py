"""Ermak-Buckholz first-order Langevin Dynamics integrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ldagg.constants import BOLTZMANN


@dataclass(frozen=True, slots=True)
class EBStepResult:
    position: np.ndarray
    velocity: np.ndarray
    dt: float
    sigma_v2: float
    sigma_r2: float


def eb_factors(beta: float, dt: float) -> tuple[float, float, float, float]:
    """Return ``e, fac4, fac3, fac5`` for the EB update."""

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    x = beta * dt
    e = float(np.exp(-x))
    fac4 = float(np.tanh(0.5 * x))
    fac3 = 1.0 - e * e
    if abs(x) < 1.0e-3:
        fac5 = x**3 / 6.0 - x**5 / 60.0 + 17.0 * x**7 / 10080.0
    else:
        fac5 = 2.0 * x - 4.0 * fac4
    return e, fac4, fac3, fac5


def positive_eb_dt(
    mass: float,
    friction: float,
    dt: float,
    *,
    shrink: float = 0.8,
    dt_min: float = 1.0e-30,
) -> tuple[float, float, float, float, float]:
    """Reduce ``dt`` until the EB displacement variance is positive."""

    beta = friction / mass
    trial_dt = dt
    while trial_dt >= dt_min:
        e, fac4, fac3, fac5 = eb_factors(beta, trial_dt)
        sigma_r2_factor = fac5 / beta**2
        if sigma_r2_factor > 0.0 and np.isfinite(sigma_r2_factor):
            return trial_dt, e, fac4, fac3, fac5
        trial_dt *= shrink
    raise RuntimeError("EB timestep underflow while enforcing positive displacement variance")


def eb_step(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    friction: float,
    force: np.ndarray,
    dt: float,
    temperature: float,
    rng: np.random.Generator,
    *,
    brownian: bool = True,
    dt_min: float = 1.0e-30,
) -> EBStepResult:
    """Advance one particle or rigid cluster with the Ermak-Buckholz update.

    The stochastic velocity and displacement normals are independent, matching
    the zero-covariance first-order tutorial implementation.
    """

    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    force = np.asarray(force, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,) or force.shape != (3,):
        raise ValueError("position, velocity, and force must be 3-vectors")
    if mass <= 0.0 or friction <= 0.0:
        raise ValueError("mass and friction must be positive")

    actual_dt, e, fac4, fac3, fac5 = positive_eb_dt(mass, friction, dt, dt_min=dt_min)
    beta = friction / mass
    thermal = BOLTZMANN * temperature / mass
    sigma_v2 = thermal * fac3
    sigma_r2 = thermal / beta**2 * fac5

    new_velocity = velocity * e + (force / friction) * (1.0 - e)
    if brownian:
        new_velocity = new_velocity + np.sqrt(sigma_v2) * rng.normal(size=3)

    new_position = (
        position
        + (new_velocity + velocity - 2.0 * force / friction) * (fac4 / beta)
        + (force / friction) * actual_dt
    )
    if brownian:
        new_position = new_position + np.sqrt(sigma_r2) * rng.normal(size=3)

    return EBStepResult(new_position, new_velocity, actual_dt, sigma_v2, sigma_r2)


def eb_step_many(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    frictions: np.ndarray,
    forces: np.ndarray,
    dt: float,
    temperature: float,
    rng: np.random.Generator,
    *,
    brownian: bool = True,
    dt_min: float = 1.0e-30,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Advance arrays of objects with a common timestep."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    forces = np.asarray(forces, dtype=float)
    masses = np.asarray(masses, dtype=float)
    frictions = np.asarray(frictions, dtype=float)
    if positions.shape != velocities.shape or positions.shape != forces.shape or positions.shape[1] != 3:
        raise ValueError("positions, velocities, and forces must have shape (n, 3)")

    actual_dt = min(
        positive_eb_dt(float(m), float(f), dt, dt_min=dt_min)[0]
        for m, f in zip(masses, frictions, strict=True)
    )

    beta = frictions / masses
    x = beta * actual_dt
    e = np.exp(-x)
    fac4 = np.tanh(0.5 * x)
    fac3 = 1.0 - e * e
    fac5 = np.where(
        np.abs(x) < 1.0e-3,
        x**3 / 6.0 - x**5 / 60.0 + 17.0 * x**7 / 10080.0,
        2.0 * x - 4.0 * fac4,
    )
    thermal = BOLTZMANN * temperature / masses
    sigma_v2 = thermal * fac3
    sigma_r2 = thermal / beta**2 * fac5

    new_velocities = velocities * e[:, None] + (forces / frictions[:, None]) * (1.0 - e)[:, None]
    if brownian:
        new_velocities = new_velocities + np.sqrt(sigma_v2)[:, None] * rng.normal(size=velocities.shape)
    new_positions = (
        positions
        + (new_velocities + velocities - 2.0 * forces / frictions[:, None])
        * (fac4 / beta)[:, None]
        + (forces / frictions[:, None]) * actual_dt
    )
    if brownian:
        new_positions = new_positions + np.sqrt(sigma_r2)[:, None] * rng.normal(size=positions.shape)
    return new_positions, new_velocities, actual_dt
