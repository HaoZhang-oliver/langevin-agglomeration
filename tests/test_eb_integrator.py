from __future__ import annotations

import numpy as np

from ldagg.constants import BOLTZMANN
from ldagg.integrators import eb_step, eb_step_many


def test_eb_equilibrium_velocity_variance() -> None:
    rng = np.random.default_rng(1)
    mass = 1.0
    friction = 1.0
    temperature = 1.0 / BOLTZMANN
    n = 20000
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    forces = np.zeros((n, 3))
    _, new_v, _ = eb_step_many(
        positions,
        velocities,
        np.full(n, mass),
        np.full(n, friction),
        forces,
        10.0,
        temperature,
        rng,
    )
    variance = np.var(new_v[:, 0])
    assert np.isclose(variance, BOLTZMANN * temperature / mass, rtol=0.05)


def test_force_free_brownian_msd() -> None:
    rng = np.random.default_rng(2)
    mass = 1.0
    friction = 1.0
    temperature = 1.0 / BOLTZMANN
    n = 30000
    velocity_std = np.sqrt(BOLTZMANN * temperature / mass)
    positions = np.zeros((n, 3))
    velocities = velocity_std * rng.normal(size=(n, 3))
    forces = np.zeros((n, 3))
    dt = 100.0
    new_r, _, _ = eb_step_many(
        positions,
        velocities,
        np.full(n, mass),
        np.full(n, friction),
        forces,
        dt,
        temperature,
        rng,
    )
    msd = np.mean(np.sum(new_r * new_r, axis=1))
    expected = 6.0 * (BOLTZMANN * temperature / friction) * dt
    assert np.isclose(msd, expected, rtol=0.08)


def test_deterministic_gravity_terminal_velocity() -> None:
    rng = np.random.default_rng(3)
    mass = 2.0
    friction = 5.0
    force = np.array([0.0, 0.0, -9.0])
    position = np.zeros(3)
    velocity = np.zeros(3)
    dt = 0.05
    for _ in range(300):
        result = eb_step(
            position,
            velocity,
            mass,
            friction,
            force,
            dt,
            300.0,
            rng,
            brownian=False,
        )
        position = result.position
        velocity = result.velocity
    assert np.isclose(velocity[2], force[2] / friction, rtol=1.0e-3)
