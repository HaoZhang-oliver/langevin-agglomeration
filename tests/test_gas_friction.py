from __future__ import annotations

import numpy as np

from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL, dimer_seed
from ldagg.constants import BOLTZMANN
from ldagg.gas import Gas, cunningham_correction, particle_mass, sphere_diffusion, sphere_friction


def test_friction_factor_matches_matlab_formula() -> None:
    gas = Gas()
    diameter = 100.0e-9
    kn = gas.mean_free_path / (0.5 * diameter)
    expected_cc = 1.0 + kn * (
        gas.cunningham_c1 + gas.cunningham_c2 * np.exp(-gas.cunningham_c3 / kn)
    )
    expected = 3.0 * np.pi * gas.viscosity * diameter / expected_cc
    assert np.isclose(cunningham_correction(diameter, gas), expected_cc)
    assert np.isclose(sphere_friction(diameter, gas), expected)


def test_mass_and_diffusion_are_si_positive() -> None:
    gas = Gas()
    diameter = 100.0e-9
    mass = particle_mass(diameter, 1000.0)
    diffusion = sphere_diffusion(diameter, gas)
    assert mass > 0.0
    assert np.isclose(diffusion, BOLTZMANN * gas.temperature / sphere_friction(diameter, gas))


def test_default_cluster_friction_uses_volume_equivalent_sphere() -> None:
    gas = Gas()
    diameter = 100.0e-9
    dimer = dimer_seed(0, diameter, gas=gas)
    expected_diameter = diameter * 2.0 ** (1.0 / 3.0)

    assert DEFAULT_CLUSTER_FRICTION_MODEL == "equivalent_sphere"
    assert dimer.friction_model == "equivalent_sphere"
    assert np.isclose(dimer.volume_equivalent_diameter, expected_diameter)
    assert np.isclose(dimer.friction, sphere_friction(expected_diameter, gas))
    assert dimer.friction < 2.0 * sphere_friction(diameter, gas)


def test_free_draining_cluster_friction_remains_available() -> None:
    gas = Gas()
    diameter = 100.0e-9
    dimer = dimer_seed(0, diameter, gas=gas, friction_model="free_draining")

    assert np.isclose(dimer.friction, 2.0 * sphere_friction(diameter, gas))
