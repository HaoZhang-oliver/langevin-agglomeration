from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ldagg.clusters import Cluster, dimer_seed
from ldagg.diagnostics import DiagnosticsConfig, cluster_dipole_diagnostics
from ldagg.electric import (
    ElectricFieldConfig,
    coupled_cluster_dipoles,
    dipole_forces_on_clusters,
    dipole_pair_force_on_b,
    induced_dipole,
    resolve_polarizability,
)
from ldagg.simulation import CoagulationConfig, run_coagulation

DIAMETER = 30.0e-9
FIELD = np.array([0.0, 0.0, 1.0e6])


def monomer(cluster_id: int, position) -> Cluster:
    return Cluster.monomer(
        cluster_id,
        np.asarray(position, dtype=float),
        np.zeros(3),
        DIAMETER,
    )


def dimer(cluster_id: int, axis: str, separation: float = DIAMETER) -> Cluster:
    cluster = dimer_seed(cluster_id, DIAMETER)
    offsets = {
        "x": np.array([0.5 * separation, 0.0, 0.0]),
        "z": np.array([0.0, 0.0, 0.5 * separation]),
    }[axis]
    cluster.rel_positions = np.vstack((-offsets, offsets))
    cluster.recenter()
    return cluster


def field_config(**kwargs) -> ElectricFieldConfig:
    data = {
        "enabled": True,
        "vector": FIELD,
        "medium_relative_permittivity": 1.0,
        "polarizability_model": "conducting_sphere",
        "dipole_force_model": "cluster_coupled_dipole",
        "regularization_gap": 0.0,
        "coupled_internal_regularization_gap": 0.0,
    }
    data.update(kwargs)
    return ElectricFieldConfig.from_mapping(data)


def test_old_electric_configs_default_to_primary_pair_fixed() -> None:
    config = ElectricFieldConfig.from_mapping(
        {
            "enabled": True,
            "vector": [1.0e6, 0.0, 0.0],
            "polarizability_SI": 1.0e-32,
        }
    )
    coag = CoagulationConfig.from_mapping(
        {
            "electric_field": {
                "enabled": True,
                "vector": [1.0e6, 0.0, 0.0],
                "polarizability_SI": 1.0e-32,
            }
        }
    )

    assert config.dipole_force_model == "primary_pair_fixed"
    assert coag.electric_field.dipole_force_model == "primary_pair_fixed"


def test_invalid_dipole_force_model_raises() -> None:
    with pytest.raises(ValueError, match="dipole_force_model"):
        ElectricFieldConfig.from_mapping({"dipole_force_model": "not_a_model"})


def test_monomer_cdm_limit_is_alpha_e() -> None:
    config = field_config()
    alpha = resolve_polarizability(config, DIAMETER)
    result = coupled_cluster_dipoles(monomer(0, [0.0, 0.0, 0.0]), config, DIAMETER)

    assert result.n_primary == 1
    assert np.allclose(result.primary_dipoles[0], induced_dipole(alpha, FIELD))
    assert np.allclose(result.total_dipole, induced_dipole(alpha, FIELD))


def test_zero_field_cdm_dipoles_and_forces_are_zero() -> None:
    config = field_config(vector=[0.0, 0.0, 0.0])
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 5.0 * DIAMETER])]
    result = coupled_cluster_dipoles(clusters[0], config, DIAMETER)
    forces = dipole_forces_on_clusters(clusters, config, DIAMETER)

    assert np.allclose(result.primary_dipoles, 0.0)
    assert np.allclose(result.total_dipole, 0.0)
    assert np.allclose(forces, 0.0)


def test_far_separated_two_primary_cluster_approaches_independent_sum() -> None:
    config = field_config()
    alpha = resolve_polarizability(config, DIAMETER)
    result = coupled_cluster_dipoles(dimer(0, "z", separation=100.0 * DIAMETER), config, DIAMETER)
    expected = 2.0 * induced_dipole(alpha, FIELD)

    assert np.allclose(result.total_dipole, expected, rtol=2.0e-6, atol=0.0)


def test_touching_dimer_cdm_anisotropy() -> None:
    config = field_config()
    alpha = resolve_polarizability(config, DIAMETER)
    independent_norm = 2.0 * alpha * np.linalg.norm(FIELD)

    parallel = coupled_cluster_dipoles(dimer(0, "z"), config, DIAMETER)
    perpendicular = coupled_cluster_dipoles(dimer(1, "x"), config, DIAMETER)

    assert np.linalg.norm(parallel.total_dipole) > independent_norm
    assert np.linalg.norm(perpendicular.total_dipole) < independent_norm


def test_e_reversal_reverses_cluster_dipole_but_not_pair_force() -> None:
    positive = field_config(vector=[0.0, 0.0, 1.0e6])
    negative = field_config(vector=[0.0, 0.0, -1.0e6])
    cluster = dimer(0, "z")
    p_positive = coupled_cluster_dipoles(cluster, positive, DIAMETER).total_dipole
    p_negative = coupled_cluster_dipoles(cluster, negative, DIAMETER).total_dipole

    clusters = [dimer(0, "z"), monomer(1, [0.0, 0.0, 8.0 * DIAMETER])]
    force_positive = dipole_forces_on_clusters(clusters, positive, DIAMETER)
    force_negative = dipole_forces_on_clusters(clusters, negative, DIAMETER)

    assert np.allclose(p_negative, -p_positive)
    assert np.allclose(force_negative, force_positive)


def test_cluster_coupled_two_monomers_reduce_to_point_dipoles() -> None:
    config = field_config()
    alpha = resolve_polarizability(config, DIAMETER)
    p = induced_dipole(alpha, FIELD)
    separation = 8.0 * DIAMETER
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, separation])]

    forces = dipole_forces_on_clusters(clusters, config, DIAMETER)
    expected_on_b = dipole_pair_force_on_b(
        np.array([0.0, 0.0, separation]),
        p,
        p,
        eps_r=config.medium_relative_permittivity,
        min_distance=DIAMETER,
    )

    assert np.allclose(forces[1], expected_on_b)
    assert np.allclose(forces[0], -expected_on_b)


def test_cluster_coupled_newtons_third_law_for_three_clusters() -> None:
    config = field_config()
    clusters = [
        monomer(0, [0.0, 0.0, 0.0]),
        monomer(1, [8.0 * DIAMETER, 0.0, 0.0]),
        dimer(2, "x"),
    ]
    clusters[2].position = np.array([0.0, 9.0 * DIAMETER, 0.0])

    forces = dipole_forces_on_clusters(clusters, config, DIAMETER)
    total_norm = float(np.sum(np.linalg.norm(forces, axis=1)))

    assert np.linalg.norm(np.sum(forces, axis=0)) < 1.0e-12 * total_norm


def test_cluster_coupled_force_scales_as_r_to_minus_four() -> None:
    config = field_config()
    clusters_r = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 8.0 * DIAMETER])]
    clusters_2r = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 16.0 * DIAMETER])]

    force_r = np.linalg.norm(dipole_forces_on_clusters(clusters_r, config, DIAMETER)[1])
    force_2r = np.linalg.norm(dipole_forces_on_clusters(clusters_2r, config, DIAMETER)[1])

    assert np.isclose(force_r / force_2r, 16.0)


def test_cluster_dipole_diagnostics_columns_and_output(tmp_path) -> None:
    electric = field_config()
    diagnostic = cluster_dipole_diagnostics([dimer(0, "z")], electric, DIAMETER)
    assert diagnostic["dipole_force_model"] == "cluster_coupled_dipole"
    assert np.isfinite(diagnostic["largest_cluster_P_over_N_alpha_E"])
    assert diagnostic["largest_cluster_P_over_N_alpha_E"] > 1.0

    config = CoagulationConfig(
        n_particles=2,
        diameter=DIAMETER,
        box_size=3.0e-7,
        boundary_mode="finite",
        t_end=2.0e-9,
        max_steps=2,
        dt_max=1.0e-9,
        save_every=1,
        seed=12,
        electric_field=electric,
        diagnostics=DiagnosticsConfig(enabled=True, every=1),
    )
    run_coagulation(config, tmp_path, make_plots=False)
    diagnostics_csv = pd.read_csv(tmp_path / "diagnostics.csv")
    final_diagnostics = pd.read_csv(tmp_path / "final_diagnostics.csv")

    expected_columns = {
        "dipole_force_model",
        "largest_cluster_P_x",
        "largest_cluster_P_y",
        "largest_cluster_P_z",
        "largest_cluster_P_norm",
        "largest_cluster_P_over_N_alpha_E",
        "max_cluster_P_norm",
        "mean_cluster_P_norm",
        "max_cluster_cdm_condition",
        "mean_cluster_cdm_condition",
    }
    assert expected_columns.issubset(diagnostics_csv.columns)
    assert expected_columns.issubset(final_diagnostics.columns)
    assert diagnostics_csv["dipole_force_model"].iloc[0] == "cluster_coupled_dipole"
    assert np.isfinite(diagnostics_csv["largest_cluster_P_over_N_alpha_E"].iloc[0])
