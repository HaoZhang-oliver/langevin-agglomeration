from __future__ import annotations

import numpy as np

from ldagg.gas import Gas, particle_mass, sphere_friction
from ldagg.settling import SettlingConfig, run_settling, settling_timestep


def test_settling_timestep_matches_tutorial_formula() -> None:
    gas = Gas()
    diameter = 1.0e-6
    mass = particle_mass(diameter, 1000.0)
    friction = sphere_friction(diameter, gas)
    force = np.array([0.0, 0.0, -mass * 9.81])
    z = 0.1
    expected = 0.01 * min(
        z * z * friction / (6.0 * 1.380649e-23 * gas.temperature),
        z * friction / np.linalg.norm(force),
    )
    assert np.isclose(settling_timestep(z, friction, force, gas.temperature), expected)


def test_short_settling_run_completes() -> None:
    config = SettlingConfig(
        diameters_nm=(1000.0,),
        height=2.0e-5,
        trials=2,
        max_steps=10000,
        seed=22,
        save_trajectory_steps=20,
    )
    result = run_settling(config)
    assert result["summary"][0]["completed"] >= 1
