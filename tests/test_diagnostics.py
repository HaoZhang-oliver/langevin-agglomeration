from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from ldagg.boundaries import Boundary
from ldagg.clusters import Cluster, dimer_seed
from ldagg.constants import BOLTZMANN, EPS0
from ldagg.diagnostics import (
    DiagnosticsConfig,
    electric_field_scalars,
    force_diagnostics,
    morphology_diagnostics,
    transport_diagnostics,
)
from ldagg.electric import (
    ElectricFieldConfig,
    dipole_force_energy_diagnostics_on_clusters,
    dipole_pair_energy,
    induced_dipole,
)
from ldagg.sequential_growth import SequentialGrowthConfig, sequential_force_components
from ldagg.simulation import (
    CoagulationConfig,
    deterministic_force_components,
    run_coagulation,
)


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


def test_electric_field_scalars_contact_gamma() -> None:
    config = field_config()
    scalars = electric_field_scalars(config, 100.0e-9, 300.0)
    p_norm = 1.0e-32 * 1.0e6
    c_dipole = p_norm**2 / (4.0 * np.pi * EPS0)
    expected_gamma = abs(-2.0 * c_dipole / (100.0e-9) ** 3) / (BOLTZMANN * 300.0)

    assert scalars["E_z"] == 1.0e6
    assert scalars["p_z"] == p_norm
    assert np.isclose(scalars["Gamma_dd_contact"], expected_gamma)

    disabled = electric_field_scalars(ElectricFieldConfig(enabled=False), 100.0e-9, 300.0)
    assert disabled["alpha"] == 0.0
    assert disabled["p_norm"] == 0.0
    assert disabled["Gamma_dd_contact"] == 0.0


def test_dipole_pair_energy_signs() -> None:
    p = induced_dipole(1.0e-32, np.array([0.0, 0.0, 1.0e6]))
    head_to_tail = dipole_pair_energy(np.array([0.0, 0.0, 200.0e-9]), p, p)
    side_by_side = dipole_pair_energy(np.array([200.0e-9, 0.0, 0.0]), p, p)

    assert head_to_tail < 0.0
    assert side_by_side > 0.0


def test_dipole_force_energy_cluster_summary() -> None:
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 200.0e-9])]
    forces, energy, evaluated, skipped, min_distance, min_gap = (
        dipole_force_energy_diagnostics_on_clusters(clusters, field_config(), 100.0e-9)
    )

    assert forces.shape == (2, 3)
    assert energy < 0.0
    assert evaluated == 1
    assert skipped == 0
    assert np.isclose(min_distance, 200.0e-9)
    assert np.isclose(min_gap, 100.0e-9)
    residual = force_diagnostics(forces, np.zeros_like(forces), forces)["newton_residual_dipole"]
    assert residual < 1.0e-12


def test_diagnostics_config_parsing_for_both_modes() -> None:
    diagnostics = {
        "enabled": True,
        "every": 2,
        "store_snapshot_forces": False,
        "store_event_metrics": True,
        "store_pair_summary": False,
    }
    coag = CoagulationConfig.from_mapping({"diagnostics": diagnostics})
    growth = SequentialGrowthConfig.from_mapping({"diagnostics": diagnostics})

    assert isinstance(coag.diagnostics, DiagnosticsConfig)
    assert coag.diagnostics.enabled
    assert coag.diagnostics.every == 2
    assert not growth.diagnostics.store_snapshot_forces
    assert not growth.diagnostics.store_pair_summary


def test_force_component_helpers_return_total_without_changing_components() -> None:
    clusters = [monomer(0, [0.0, 0.0, 0.0]), monomer(1, [0.0, 0.0, 200.0e-9])]
    config = CoagulationConfig(
        n_particles=2,
        diameter=100.0e-9,
        gravity_enabled=True,
        electric_field=field_config(),
    )
    components = deterministic_force_components(
        config,
        clusters,
        Boundary("finite", 1.0e-6),
        include_pair_summary=True,
    )

    assert set(components) == {"gravity", "dipole", "total", "dipole_pair_summary"}
    assert np.allclose(components["total"], components["gravity"] + components["dipole"])
    assert components["dipole_pair_summary"]["dipole_evaluated_pairs"] == 1

    seq_components = sequential_force_components(
        SequentialGrowthConfig(diameter=100.0e-9, electric_field=field_config()),
        clusters[0],
        clusters[1],
        include_pair_summary=True,
    )
    assert np.allclose(seq_components["total"], seq_components["gravity"] + seq_components["dipole"])


def test_transport_and_morphology_diagnostics() -> None:
    dimer = dimer_seed(0, 100.0e-9)
    dimer.rel_positions = np.array([[0.0, 0.0, -50.0e-9], [0.0, 0.0, 50.0e-9]])
    dimer.recenter()
    morph = morphology_diagnostics(
        dimer,
        electric_field_vector=np.array([0.0, 0.0, 1.0]),
        electric_field_enabled=True,
    )
    assert np.isclose(morph["radius_of_gyration"], 50.0e-9)
    assert morph["aspect_ratio"] > 1.0e100
    assert np.isclose(morph["alignment_order_S"], 1.0)
    assert morph["max_coordination"] == 1.0
    assert morph["mean_coordination"] == 1.0

    transport = transport_diagnostics([dimer], np.array([[1.0e-15, 0.0, 0.0]]), 1.0e-6, 300.0)
    assert transport["max_drift_step"] > 0.0
    assert transport["max_brownian_rms_step"] > 0.0
    assert transport["max_drift_to_brownian_ratio"] > 0.0


def test_coagulation_diagnostics_outputs(tmp_path) -> None:
    config = CoagulationConfig(
        n_particles=2,
        diameter=100.0e-9,
        box_size=6.0e-7,
        boundary_mode="finite",
        t_end=2.0e-9,
        max_steps=2,
        dt_max=1.0e-9,
        save_every=1,
        seed=2,
        electric_field=field_config(),
        diagnostics=DiagnosticsConfig(enabled=True, every=1),
    )
    result = run_coagulation(config, tmp_path, make_plots=False)

    assert result.diagnostics
    diagnostics_csv = pd.read_csv(tmp_path / "diagnostics.csv")
    assert {"E_norm", "p_norm", "Gamma_dd_contact", "newton_residual_dipole"}.issubset(
        diagnostics_csv.columns
    )
    assert (tmp_path / "final_diagnostics.json").exists()
    with h5py.File(tmp_path / "run.h5", "r") as h5:
        assert "diagnostics" in h5
        assert "final_diagnostics" in h5
        first_snapshot = h5["snapshots"]["000000"]
        assert "force_dipole" in first_snapshot
