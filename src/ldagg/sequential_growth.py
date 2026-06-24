"""Sequential monomer growth mode."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from ldagg.analysis import aggregate_table
from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL, Cluster, dimer_seed
from ldagg.collisions import find_collision, merge_clusters, nearest_surface_gap
from ldagg.constants import BOLTZMANN, DEFAULT_PARTICLE_DENSITY
from ldagg.gas import Gas
from ldagg.integrators import eb_step
from ldagg.plotting import plot_cluster_stats_csv, plot_final_aggregate_csv


@dataclass(slots=True)
class SequentialGrowthConfig:
    gas: Gas = field(default_factory=Gas)
    target_size: int = 8
    seed_type: str = "dimer"
    diameter: float = 100.0e-9
    particle_density: float = DEFAULT_PARTICLE_DENSITY
    launch_gap: float = 20.0e-9
    kill_gap: float = 500.0e-9
    max_steps_per_trial: int = 5000
    max_trials: int = 200
    dt_factor: float = 5.0e-2
    dt_min: float = 1.0e-12
    dt_max: float = 1.0e-5
    gap_floor_fraction: float = 1.0e-2
    capture_tolerance: float = 0.0
    gravity_enabled: bool = False
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    brownian: bool = True
    friction_model: str = DEFAULT_CLUSTER_FRICTION_MODEL
    seed: int = 123
    move_seed: bool = False
    project_to_contact: bool = True

    @classmethod
    def from_mapping(cls, data: dict | None) -> SequentialGrowthConfig:
        data = dict(data or {})
        gas = Gas.from_mapping(data.pop("gas", None))
        return cls(gas=gas, **data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gas"] = self.gas.to_dict()
        return data


def uniform_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    return vec / np.linalg.norm(vec)


def make_seed_cluster(config: SequentialGrowthConfig) -> Cluster:
    if config.seed_type == "monomer":
        return Cluster.monomer(
            0,
            np.zeros(3),
            np.zeros(3),
            config.diameter,
            density=config.particle_density,
            gas=config.gas,
            friction_model=config.friction_model,
        )
    if config.seed_type == "dimer":
        return dimer_seed(
            0,
            config.diameter,
            density=config.particle_density,
            gas=config.gas,
            friction_model=config.friction_model,
        )
    raise ValueError("seed_type must be 'monomer' or 'dimer'")


def growth_timestep(config: SequentialGrowthConfig, aggregate: Cluster, monomer: Cluster) -> float:
    gap = nearest_surface_gap([aggregate, monomer])
    gap_floor = config.gap_floor_fraction * min(float(np.min(aggregate.radii)), float(np.min(monomer.radii)))
    gap_scale = max(gap, gap_floor)
    f_min = min(aggregate.friction, monomer.friction)
    dt_diff = gap_scale * gap_scale * f_min / (6.0 * BOLTZMANN * config.gas.temperature)
    if config.gravity_enabled:
        force_max = max(aggregate.mass, monomer.mass) * np.linalg.norm(config.gravity)
        dt_force = np.inf if force_max == 0.0 else gap_scale * f_min / force_max
    else:
        dt_force = np.inf
    dt = config.dt_factor * min(dt_diff, dt_force, config.dt_max)
    if not np.isfinite(dt):
        dt = config.dt_max
    return float(np.clip(dt, config.dt_min, config.dt_max))


def simulate_sequential_growth(config: SequentialGrowthConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    aggregate = make_seed_cluster(config)
    next_cluster_id = aggregate.n_primary
    events = []
    stats = [
        {
            "time": 0.0,
            "trial": 0,
            "cluster_count": 1,
            "largest_cluster": aggregate.n_primary,
            "events": 0,
            "mean_cluster_size": float(aggregate.n_primary),
        }
    ]
    total_time = 0.0
    trials = 0
    speed_std_cache = None
    gravity = np.asarray(config.gravity, dtype=float)

    while aggregate.n_primary < config.target_size and trials < config.max_trials:
        trials += 1
        radius = aggregate.bounding_radius
        launch_radius = radius + config.launch_gap + 0.5 * config.diameter
        kill_radius = launch_radius + config.kill_gap
        direction = uniform_unit_vector(rng)
        position = aggregate.position + direction * launch_radius
        if speed_std_cache is None:
            probe = Cluster.monomer(
                -1,
                position,
                np.zeros(3),
                config.diameter,
                density=config.particle_density,
                gas=config.gas,
                friction_model=config.friction_model,
            )
            speed_std_cache = np.sqrt(BOLTZMANN * config.gas.temperature / probe.mass)
        velocity = speed_std_cache * rng.normal(size=3)
        monomer = Cluster.monomer(
            next_cluster_id,
            position,
            velocity,
            config.diameter,
            density=config.particle_density,
            gas=config.gas,
            friction_model=config.friction_model,
        )
        hit_info = None
        for _step in range(config.max_steps_per_trial):
            hit_info = find_collision(aggregate, monomer, capture_tolerance=config.capture_tolerance)
            if hit_info is not None:
                break
            if np.linalg.norm(monomer.position - aggregate.position) > kill_radius:
                break
            dt = growth_timestep(config, aggregate, monomer)
            if config.move_seed:
                # The seed aggregate translates as one rigid cluster using its
                # COM, total mass, and cluster-level scalar friction.
                force_a = aggregate.mass * gravity if config.gravity_enabled else np.zeros(3)
                result_a = eb_step(
                    aggregate.position,
                    aggregate.velocity,
                    aggregate.mass,
                    aggregate.friction,
                    force_a,
                    dt,
                    config.gas.temperature,
                    rng,
                    brownian=config.brownian,
                    dt_min=config.dt_min * 1.0e-6,
                )
                aggregate.position = result_a.position
                aggregate.velocity = result_a.velocity
            force_m = monomer.mass * gravity if config.gravity_enabled else np.zeros(3)
            result_m = eb_step(
                monomer.position,
                monomer.velocity,
                monomer.mass,
                monomer.friction,
                force_m,
                dt,
                config.gas.temperature,
                rng,
                brownian=config.brownian,
                dt_min=config.dt_min * 1.0e-6,
            )
            monomer.position = result_m.position
            monomer.velocity = result_m.velocity
            total_time += dt
        if hit_info is None:
            hit_info = find_collision(aggregate, monomer, capture_tolerance=config.capture_tolerance)
        if hit_info is not None:
            aggregate, event = merge_clusters(
                aggregate,
                monomer,
                collision=hit_info,
                new_cluster_id=next_cluster_id + 1,
                time=total_time,
                project_to_contact=config.project_to_contact,
            )
            if not config.move_seed:
                aggregate.position = np.zeros(3)
                aggregate.velocity = np.zeros(3)
                aggregate.recenter()
            events.append(event)
            next_cluster_id += 2
            stats.append(
                {
                    "time": total_time,
                    "trial": trials,
                    "cluster_count": 1,
                    "largest_cluster": aggregate.n_primary,
                    "events": len(events),
                    "mean_cluster_size": float(aggregate.n_primary),
                }
            )

    return {
        "aggregate": aggregate,
        "events": events,
        "stats": stats,
        "time": total_time,
        "trials": trials,
        "config": config.to_dict(),
        "reached_target": aggregate.n_primary >= config.target_size,
    }


def write_growth_outputs(result: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    with (out / "config_used.yml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result["config"], fh, sort_keys=False)
    summary = {
        "mode": "grow",
        "time": result["time"],
        "trials": result["trials"],
        "target_size": result["config"]["target_size"],
        "final_size": result["aggregate"].n_primary,
        "events": len(result["events"]),
        "reached_target": bool(result["reached_target"]),
    }
    with (out / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame([event.to_dict() for event in result["events"]]).to_csv(out / "events.csv", index=False)
    pd.DataFrame(result["stats"]).to_csv(out / "cluster_stats.csv", index=False)
    pd.DataFrame(aggregate_table(result["aggregate"])).to_csv(out / "final_aggregate.csv", index=False)
    with h5py.File(out / "run.h5", "w") as h5:
        for key, value in summary.items():
            if isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating)):
                h5.attrs[key] = value
        centers = result["aggregate"].absolute_centers
        h5.create_dataset("final_centers", data=centers)
        h5.create_dataset("final_radii", data=result["aggregate"].radii)
    plot_cluster_stats_csv(out / "cluster_stats.csv", plots)
    plot_final_aggregate_csv(out / "final_aggregate.csv", plots)


def run_sequential_growth(
    config: SequentialGrowthConfig,
    out_dir: str | Path | None = None,
) -> dict:
    result = simulate_sequential_growth(config)
    if out_dir is not None:
        write_growth_outputs(result, out_dir)
    return result
