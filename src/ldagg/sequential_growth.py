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
from ldagg.diagnostics import (
    DiagnosticsConfig,
    cluster_dipole_diagnostics,
    electric_field_scalars,
    force_diagnostics,
    morphology_diagnostics,
    transport_diagnostics,
)
from ldagg.electric import (
    ElectricFieldConfig,
    dipole_force_energy_diagnostics_on_clusters,
    dipole_forces_on_clusters,
)
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
    electric_field: ElectricFieldConfig = field(default_factory=ElectricFieldConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    seed: int = 123
    move_seed: bool = False
    project_to_contact: bool = True

    @classmethod
    def from_mapping(cls, data: dict | None) -> SequentialGrowthConfig:
        data = dict(data or {})
        gas = Gas.from_mapping(data.pop("gas", None))
        electric_field = ElectricFieldConfig.from_mapping(data.pop("electric_field", None))
        diagnostics = DiagnosticsConfig.from_mapping(data.pop("diagnostics", None))
        return cls(gas=gas, electric_field=electric_field, diagnostics=diagnostics, **data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gas"] = self.gas.to_dict()
        data["electric_field"] = self.electric_field.to_dict()
        data["diagnostics"] = self.diagnostics.to_dict()
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


def sequential_deterministic_forces(
    config: SequentialGrowthConfig,
    aggregate: Cluster,
    monomer: Cluster,
) -> np.ndarray:
    """Return gravity plus optional induced-dipole force for aggregate and monomer."""

    return np.asarray(sequential_force_components(config, aggregate, monomer)["total"], dtype=float)


def sequential_force_components(
    config: SequentialGrowthConfig,
    aggregate: Cluster,
    monomer: Cluster,
    *,
    include_pair_summary: bool = False,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Return gravity, dipole, and total deterministic forces for aggregate and monomer."""

    clusters = [aggregate, monomer]
    gravity_forces = np.zeros((2, 3), dtype=float)
    dipole_forces = np.zeros((2, 3), dtype=float)
    pair_summary = _empty_pair_summary()
    if config.gravity_enabled:
        gravity = np.asarray(config.gravity, dtype=float)
        gravity_forces = np.asarray([cluster.mass * gravity for cluster in clusters], dtype=float)
    if config.electric_field.enabled:
        if include_pair_summary:
            (
                dipole_forces,
                energy,
                evaluated,
                skipped,
                min_distance,
                min_gap,
            ) = dipole_force_energy_diagnostics_on_clusters(
                clusters,
                config.electric_field,
                config.diameter,
                boundary=None,
            )
            pair_summary = _pair_summary_dict(energy, evaluated, skipped, min_distance, min_gap)
        else:
            dipole_forces = dipole_forces_on_clusters(
                clusters,
                config.electric_field,
                config.diameter,
                boundary=None,
            )
    total = gravity_forces + dipole_forces
    return {
        "gravity": gravity_forces,
        "dipole": dipole_forces,
        "total": total,
        "dipole_pair_summary": pair_summary,
    }


def _empty_pair_summary() -> dict[str, float]:
    return {
        "dipole_energy": 0.0,
        "dipole_evaluated_pairs": 0,
        "dipole_skipped_cutoff_pairs": 0,
        "dipole_min_center_distance": float("inf"),
        "dipole_min_surface_gap": float("inf"),
    }


def _pair_summary_dict(
    energy: float,
    evaluated: int,
    skipped: int,
    min_distance: float,
    min_gap: float,
) -> dict[str, float]:
    return {
        "dipole_energy": float(energy),
        "dipole_evaluated_pairs": int(evaluated),
        "dipole_skipped_cutoff_pairs": int(skipped),
        "dipole_min_center_distance": float(min_distance),
        "dipole_min_surface_gap": float(min_gap),
    }


def _diagnostics_due(config: SequentialGrowthConfig, step: int) -> bool:
    return config.diagnostics.enabled and step % config.diagnostics.every == 0


def _diagnostic_row(
    config: SequentialGrowthConfig,
    aggregate: Cluster,
    monomer: Cluster,
    *,
    time: float,
    trial: int,
    step: int,
    dt: float,
    components: dict[str, np.ndarray | dict[str, float]],
    events_count: int,
    transport_forces: np.ndarray,
) -> dict[str, float | int | str]:
    clusters = [aggregate, monomer]
    row: dict[str, float | int] = {
        "time": float(time),
        "trial": int(trial),
        "step": int(step),
        "dt": float(dt),
        "cluster_count": 2,
        "largest_cluster": int(aggregate.n_primary),
        "events": int(events_count),
    }
    row.update(electric_field_scalars(config.electric_field, config.diameter, config.gas.temperature))
    if config.electric_field.enabled:
        row.update(cluster_dipole_diagnostics(clusters, config.electric_field, config.diameter))
    row.update(
        force_diagnostics(
            np.asarray(components["dipole"], dtype=float),
            np.asarray(components["gravity"], dtype=float),
            np.asarray(components["total"], dtype=float),
        )
    )
    row.update(
        transport_diagnostics(
            clusters,
            transport_forces,
            dt,
            config.gas.temperature,
            boundary=None,
        )
    )
    if config.diagnostics.store_pair_summary:
        row.update(components.get("dipole_pair_summary", _empty_pair_summary()))
    row.update(
        morphology_diagnostics(
            aggregate,
            electric_field_vector=config.electric_field.vector,
            electric_field_enabled=config.electric_field.enabled,
            contact_tolerance=max(1.0e-12, config.capture_tolerance),
        )
    )
    return row


def _collision_event_metric_row(
    event,
    merged: Cluster,
    aggregate: Cluster,
    monomer: Cluster,
    info,
    config: SequentialGrowthConfig,
    *,
    trial: int,
) -> dict[str, float | int]:
    radius_sum = float(aggregate.radii[info.primary_a] + monomer.radii[info.primary_b])
    row = event.to_dict()
    row.update(
        {
            "trial": int(trial),
            "trigger_surface_gap": float(info.distance - radius_sum),
            "contact_radius_sum": radius_sum,
            "normal_x": float(info.normal[0]),
            "normal_y": float(info.normal[1]),
            "normal_z": float(info.normal[2]),
        }
    )
    row.update(
        morphology_diagnostics(
            merged,
            electric_field_vector=config.electric_field.vector,
            electric_field_enabled=config.electric_field.enabled,
            contact_tolerance=max(1.0e-12, config.capture_tolerance),
        )
    )
    return row


def growth_timestep(
    config: SequentialGrowthConfig,
    aggregate: Cluster,
    monomer: Cluster,
    forces: np.ndarray | None = None,
) -> float:
    gap = nearest_surface_gap([aggregate, monomer])
    gap_floor = config.gap_floor_fraction * min(float(np.min(aggregate.radii)), float(np.min(monomer.radii)))
    gap_scale = max(gap, gap_floor)
    clusters = [aggregate, monomer]
    if forces is None:
        forces = sequential_deterministic_forces(config, aggregate, monomer)
        if not config.move_seed:
            forces[0] = 0.0
    max_diffusion = max(BOLTZMANN * config.gas.temperature / cluster.friction for cluster in clusters)
    dt_diff = gap_scale * gap_scale / (6.0 * max_diffusion)
    drift_speeds = [
        np.linalg.norm(force) / cluster.friction
        for force, cluster in zip(np.asarray(forces, dtype=float), clusters, strict=True)
    ]
    max_drift_speed = max(drift_speeds, default=0.0)
    dt_force = gap_scale / max_drift_speed if max_drift_speed > 0.0 else np.inf
    dt = config.dt_factor * min(dt_diff, dt_force, config.dt_max)
    if not np.isfinite(dt):
        dt = config.dt_max
    return float(np.clip(dt, config.dt_min, config.dt_max))


def simulate_sequential_growth(config: SequentialGrowthConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    aggregate = make_seed_cluster(config)
    next_cluster_id = aggregate.n_primary
    events = []
    event_metrics = []
    diagnostics_rows = []
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
            components = sequential_force_components(
                config,
                aggregate,
                monomer,
                include_pair_summary=config.diagnostics.store_pair_summary
                and _diagnostics_due(config, _step),
            )
            forces = np.asarray(components["total"], dtype=float).copy()
            if not config.move_seed:
                forces[0] = 0.0
            dt = growth_timestep(config, aggregate, monomer, forces=forces)
            if _diagnostics_due(config, _step):
                diagnostics_rows.append(
                    _diagnostic_row(
                        config,
                        aggregate,
                        monomer,
                        time=total_time,
                        trial=trials,
                        step=_step,
                        dt=dt,
                        components=components,
                        events_count=len(events),
                        transport_forces=forces,
                    )
                )
            if config.move_seed:
                # The seed aggregate translates as one rigid cluster using its
                # COM, total mass, and cluster-level scalar friction.
                result_a = eb_step(
                    aggregate.position,
                    aggregate.velocity,
                    aggregate.mass,
                    aggregate.friction,
                    forces[0],
                    dt,
                    config.gas.temperature,
                    rng,
                    brownian=config.brownian,
                    dt_min=config.dt_min * 1.0e-6,
                )
                aggregate.position = result_a.position
                aggregate.velocity = result_a.velocity
            result_m = eb_step(
                monomer.position,
                monomer.velocity,
                monomer.mass,
                monomer.friction,
                forces[1],
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
            old_aggregate = aggregate
            aggregate, event = merge_clusters(
                old_aggregate,
                monomer,
                collision=hit_info,
                new_cluster_id=next_cluster_id + 1,
                time=total_time,
                project_to_contact=config.project_to_contact,
            )
            if config.diagnostics.enabled and config.diagnostics.store_event_metrics:
                event_metrics.append(
                    _collision_event_metric_row(
                        event,
                        aggregate,
                        old_aggregate,
                        monomer,
                        hit_info,
                        config,
                        trial=trials,
                    )
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
        "diagnostics": diagnostics_rows,
        "event_metrics": event_metrics,
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
    if result["diagnostics"]:
        pd.DataFrame(result["diagnostics"]).to_csv(out / "diagnostics.csv", index=False)
    if result["event_metrics"]:
        pd.DataFrame(result["event_metrics"]).to_csv(out / "event_metrics.csv", index=False)
    pd.DataFrame(aggregate_table(result["aggregate"])).to_csv(out / "final_aggregate.csv", index=False)
    final_diagnostics = morphology_diagnostics(
        result["aggregate"],
        electric_field_vector=result["config"]["electric_field"]["vector"],
        electric_field_enabled=bool(result["config"]["electric_field"]["enabled"]),
        contact_tolerance=max(1.0e-12, float(result["config"]["capture_tolerance"])),
    )
    if bool(result["config"]["electric_field"]["enabled"]):
        final_diagnostics.update(
            cluster_dipole_diagnostics(
                [result["aggregate"]],
                ElectricFieldConfig.from_mapping(result["config"]["electric_field"]),
                float(result["config"]["diameter"]),
            )
        )
    with (out / "final_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(final_diagnostics, fh, indent=2)
    pd.DataFrame([final_diagnostics]).to_csv(out / "final_diagnostics.csv", index=False)
    with h5py.File(out / "run.h5", "w") as h5:
        for key, value in summary.items():
            if isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating)):
                h5.attrs[key] = value
        centers = result["aggregate"].absolute_centers
        h5.create_dataset("final_centers", data=centers)
        h5.create_dataset("final_radii", data=result["aggregate"].radii)
        _write_rows_group(h5, "diagnostics", result["diagnostics"])
        _write_rows_group(h5, "event_metrics", result["event_metrics"])
        _write_rows_group(h5, "final_diagnostics", [final_diagnostics])
    plot_cluster_stats_csv(out / "cluster_stats.csv", plots)
    plot_final_aggregate_csv(out / "final_aggregate.csv", plots)


def _write_rows_group(h5, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    group = h5.create_group(name)
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [row.get(key, np.nan) for row in rows]
        if any(isinstance(value, str) for value in values):
            group.create_dataset(key, data=np.asarray(values, dtype="S"))
        else:
            group.create_dataset(key, data=np.asarray(values, dtype=float))


def run_sequential_growth(
    config: SequentialGrowthConfig,
    out_dir: str | Path | None = None,
) -> dict:
    result = simulate_sequential_growth(config)
    if out_dir is not None:
        write_growth_outputs(result, out_dir)
    return result
