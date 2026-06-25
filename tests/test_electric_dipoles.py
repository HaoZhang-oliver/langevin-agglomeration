from __future__ import annotations

import numpy as np

from ldagg.clusters import Cluster
from ldagg.constants import EPS0
from ldagg.electric import (
    ElectricFieldConfig,
    dipole_forces_on_clusters,
    dipole_pair_force_on_b,
    induced_dipole,
    load_material_properties,
    resolve_polarizability,
)
from ldagg.sequential_growth import SequentialGrowthConfig, run_sequential_growth
from ldagg.simulation import CoagulationConfig


def monomer(cluster_id: int, position) -> Cluster:
    return Cluster.monomer(cluster_id, np.asarray(position, dtype=float), np.zeros(3), 100.0e-9)


def field_config(**kwargs) -> ElectricFieldConfig:
    data = {
        "enabled": True,
        "vector": (0.0, 0.0, 1.0e6),
        "medium_relative_permittivity": 1.0,
        "polarizability_SI": 1.0e-32,
    }
    data.update(kwargs)
    return ElectricFieldConfig.from_mapping(data)


def test_disabled_field_returns_zero_forces() -> None:
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 200.0e-9])]
    forces = dipole_forces_on_clusters(clusters, ElectricFieldConfig(enabled=False), 100.0e-9)
    assert np.allclose(forces, 0.0)


def test_zero_field_vector_returns_zero_forces() -> None:
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 200.0e-9])]
    config = ElectricFieldConfig.from_mapping({"enabled": True, "vector": [0.0, 0.0, 0.0]})
    forces = dipole_forces_on_clusters(clusters, config, 100.0e-9)
    assert np.allclose(forces, 0.0)


def test_head_to_tail_parallel_dipoles_attract() -> None:
    p = induced_dipole(1.0e-32, np.array([0.0, 0.0, 1.0e6]))
    force = dipole_pair_force_on_b(np.array([0.0, 0.0, 200.0e-9]), p, p)
    assert force[2] < 0.0


def test_side_by_side_parallel_dipoles_repulse() -> None:
    p = induced_dipole(1.0e-32, np.array([0.0, 0.0, 1.0e6]))
    force = dipole_pair_force_on_b(np.array([200.0e-9, 0.0, 0.0]), p, p)
    assert force[0] > 0.0


def test_newtons_third_law_cluster_forces() -> None:
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 200.0e-9])]
    forces = dipole_forces_on_clusters(clusters, field_config(), 100.0e-9)
    assert np.allclose(np.sum(forces, axis=0), 0.0, atol=1.0e-30)


def test_force_decays_as_r_to_minus_four() -> None:
    p = induced_dipole(1.0e-32, np.array([0.0, 0.0, 1.0e6]))
    force_r = np.linalg.norm(dipole_pair_force_on_b(np.array([0.0, 0.0, 200.0e-9]), p, p))
    force_2r = np.linalg.norm(dipole_pair_force_on_b(np.array([0.0, 0.0, 400.0e-9]), p, p))
    assert np.isclose(force_r / force_2r, 16.0)


def test_material_yaml_loader(tmp_path) -> None:
    path = tmp_path / "material.yml"
    path.write_text(
        "name: test_metal\npolarizability_SI: 2.5e-32\ndensity: 19300\nnotes: test\n",
        encoding="utf-8",
    )
    material = load_material_properties(path)
    assert material.name == "test_metal"
    assert material.polarizability_SI == 2.5e-32
    assert material.density == 19300.0


def test_conducting_sphere_polarizability_model() -> None:
    config = ElectricFieldConfig.from_mapping(
        {
            "enabled": True,
            "vector": [1.0e6, 0.0, 0.0],
            "medium_relative_permittivity": 1.0,
            "polarizability_model": "conducting_sphere",
        }
    )
    alpha = resolve_polarizability(config, 100.0e-9)
    expected = 4.0 * np.pi * EPS0 * (50.0e-9) ** 3
    assert np.isclose(alpha, expected)


def test_config_parsing_with_electric_field(tmp_path) -> None:
    material = tmp_path / "material.yml"
    material.write_text("name: test\npolarizability_SI: 1.2e-32\n", encoding="utf-8")
    electric = {
        "enabled": True,
        "vector": [1.0e6, 0.0, 0.0],
        "medium_relative_permittivity": 1.00058,
        "material_file": str(material),
        "polarizability_model": "provided",
    }
    coag = CoagulationConfig.from_mapping({"electric_field": electric})
    growth = SequentialGrowthConfig.from_mapping({"electric_field": electric})

    assert coag.electric_field.enabled
    assert growth.electric_field.enabled
    assert coag.electric_field.vector == (1.0e6, 0.0, 0.0)
    assert growth.electric_field.material_file == str(material)


def test_direct_polarizability_overrides_material(tmp_path) -> None:
    material = tmp_path / "material.yml"
    material.write_text("name: test\npolarizability_SI: 1.0e-32\n", encoding="utf-8")
    config = field_config(material_file=str(material), polarizability_SI=3.0e-32)
    assert resolve_polarizability(config, 100.0e-9) == 3.0e-32


def test_short_dipole_enabled_growth_smoke() -> None:
    config = SequentialGrowthConfig(
        target_size=3,
        seed_type="dimer",
        launch_gap=0.0,
        kill_gap=250.0e-9,
        max_steps_per_trial=500,
        max_trials=20,
        dt_max=2.0e-7,
        seed=19,
        capture_tolerance=80.0e-9,
        electric_field=field_config(vector=[1.0e6, 0.0, 0.0]),
    )
    result = run_sequential_growth(config)
    assert result["aggregate"].n_primary >= 2
