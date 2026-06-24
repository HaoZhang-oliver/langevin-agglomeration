from __future__ import annotations

from ldagg.sequential_growth import SequentialGrowthConfig, run_sequential_growth
from ldagg.simulation import CoagulationConfig, random_nonoverlap_monomers, run_coagulation


def test_finite_box_initializes_monomers_inside_walls() -> None:
    import numpy as np

    config = CoagulationConfig(
        n_particles=8,
        diameter=100.0e-9,
        box_size=6.0e-7,
        boundary_mode="finite",
        seed=17,
    )
    clusters = random_nonoverlap_monomers(config, np.random.default_rng(config.seed))
    radius = 0.5 * config.diameter

    for cluster in clusters:
        centers = cluster.absolute_centers
        assert np.all(centers - radius >= 0.0)
        assert np.all(centers + radius <= config.box_size)


def test_short_coagulate_smoke() -> None:
    config = CoagulationConfig(
        n_particles=5,
        diameter=100.0e-9,
        box_size=5.0e-7,
        t_end=5.0e-6,
        max_steps=200,
        dt_max=5.0e-7,
        seed=7,
        capture_tolerance=5.0e-9,
    )
    result = run_coagulation(config)
    assert len(result.clusters) <= config.n_particles
    assert result.stats[-1]["cluster_count"] == len(result.clusters)


def test_short_grow_smoke_reaches_target() -> None:
    config = SequentialGrowthConfig(
        target_size=4,
        launch_gap=0.0,
        kill_gap=300.0e-9,
        max_steps_per_trial=1500,
        max_trials=80,
        dt_max=1.0e-6,
        seed=5,
        capture_tolerance=80.0e-9,
    )
    result = run_sequential_growth(config)
    assert result["reached_target"]
    assert result["aggregate"].n_primary >= config.target_size
