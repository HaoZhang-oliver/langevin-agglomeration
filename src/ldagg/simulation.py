"""N-cluster Brownian coagulation/agglomeration simulation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from ldagg.analysis import aggregate_table, cluster_size_counts
from ldagg.boundaries import Boundary, apply_boundary_to_cluster, minimum_image
from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL, Cluster
from ldagg.collisions import first_collision, merge_clusters, nearest_surface_gap
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
class CoagulationConfig:
    gas: Gas = field(default_factory=Gas)
    n_particles: int = 32
    diameter: float = 100.0e-9
    particle_density: float = DEFAULT_PARTICLE_DENSITY
    box_size: float = 2.0e-6
    boundary_mode: str = "periodic"
    t_end: float = 1.0e-3
    target_clusters: int = 1
    target_size: int | None = None
    max_events: int = 1000
    max_steps: int = 100_000
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
    seed: int = 2024
    save_every: int = 20
    project_to_contact: bool = True

    @classmethod
    def from_mapping(cls, data: dict | None) -> CoagulationConfig:
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


@dataclass(slots=True)
class CoagulationResult:
    clusters: list[Cluster]
    events: list
    stats: list[dict]
    snapshots: list[dict]
    diagnostics: list[dict]
    event_metrics: list[dict]
    config: dict
    time: float
    steps: int


def random_nonoverlap_monomers(config: CoagulationConfig, rng: np.random.Generator) -> list[Cluster]:
    """Initialize monomers uniformly in the box without overlaps."""

    clusters: list[Cluster] = []
    radius = 0.5 * config.diameter
    boundary = Boundary(config.boundary_mode, config.box_size)
    speed_std = None
    attempts = 0
    max_attempts = max(10_000, config.n_particles * 2_000)
    if boundary.mode == "finite":
        low = boundary.low + radius
        high = boundary.high - radius
        if np.any(high <= low):
            raise ValueError("box_size is too small to contain the requested monomer diameter")
    else:
        low = boundary.low
        high = boundary.high
    while len(clusters) < config.n_particles and attempts < max_attempts:
        attempts += 1
        position = rng.uniform(low, high, size=3)
        ok = True
        for cluster in clusters:
            disp = position - cluster.position
            if boundary.mode == "periodic":
                disp = minimum_image(disp, boundary.size)
            if np.linalg.norm(disp) < 2.05 * radius:
                ok = False
                break
        if not ok:
            continue
        if speed_std is None:
            probe = Cluster.monomer(
                -1,
                position,
                np.zeros(3),
                config.diameter,
                density=config.particle_density,
                gas=config.gas,
                friction_model=config.friction_model,
            )
            speed_std = np.sqrt(BOLTZMANN * config.gas.temperature / probe.mass)
        velocity = speed_std * rng.normal(size=3)
        clusters.append(
            Cluster.monomer(
                len(clusters),
                position,
                velocity,
                config.diameter,
                density=config.particle_density,
                gas=config.gas,
                friction_model=config.friction_model,
            )
        )
    if len(clusters) < config.n_particles:
        raise RuntimeError("could not place non-overlapping monomers; increase box_size")
    return clusters


def deterministic_forces(
    config: CoagulationConfig,
    clusters: list[Cluster],
    boundary: Boundary,
) -> np.ndarray:
    """Return gravity plus optional induced-dipole force on each cluster."""

    return deterministic_force_components(config, clusters, boundary)["total"]


def deterministic_force_components(
    config: CoagulationConfig,
    clusters: list[Cluster],
    boundary: Boundary,
    *,
    include_pair_summary: bool = False,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Return gravity, dipole, and total deterministic force components."""

    gravity_forces = np.zeros((len(clusters), 3), dtype=float)
    dipole_forces = np.zeros((len(clusters), 3), dtype=float)
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
                boundary=boundary,
            )
            pair_summary = _pair_summary_dict(energy, evaluated, skipped, min_distance, min_gap)
        else:
            dipole_forces = dipole_forces_on_clusters(
                clusters,
                config.electric_field,
                config.diameter,
                boundary=boundary,
            )
    total = gravity_forces + dipole_forces
    return {
        "gravity": gravity_forces,
        "dipole": dipole_forces,
        "total": total,
        "dipole_pair_summary": pair_summary,
    }


def agglomeration_timestep(
    config: CoagulationConfig,
    clusters: list[Cluster],
    boundary: Boundary,
    forces: np.ndarray | None = None,
) -> float:
    """Conservative gap-based timestep for agglomeration."""

    if len(clusters) < 2:
        return min(config.dt_max, max(config.dt_min, config.t_end))
    if forces is None:
        forces = deterministic_forces(config, clusters, boundary)
    gap = nearest_surface_gap(clusters, boundary=boundary)
    min_radius = min(float(np.min(cluster.radii)) for cluster in clusters)
    gap_scale = max(gap, config.gap_floor_fraction * min_radius)
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


def record_stats(time: float, steps: int, clusters: list[Cluster], events_count: int) -> dict:
    counts = cluster_size_counts(clusters)
    largest = max(counts) if counts else 0
    return {
        "time": float(time),
        "step": int(steps),
        "cluster_count": int(len(clusters)),
        "largest_cluster": int(largest),
        "events": int(events_count),
        "mean_cluster_size": float(np.mean([cluster.n_primary for cluster in clusters])) if clusters else 0.0,
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


def _diagnostics_due(config: CoagulationConfig, step: int) -> bool:
    return config.diagnostics.enabled and step % config.diagnostics.every == 0


def _largest_cluster(clusters: list[Cluster]) -> Cluster:
    return max(clusters, key=lambda cluster: cluster.n_primary)


def _diagnostic_row(
    config: CoagulationConfig,
    clusters: list[Cluster],
    boundary: Boundary,
    *,
    time: float,
    step: int,
    dt: float,
    components: dict[str, np.ndarray | dict[str, float]],
    events_count: int,
    transport_forces: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    total_for_transport = (
        np.asarray(components["total"], dtype=float)
        if transport_forces is None
        else np.asarray(transport_forces, dtype=float)
    )
    row: dict[str, float | int] = {
        "time": float(time),
        "step": int(step),
        "dt": float(dt),
        "cluster_count": int(len(clusters)),
        "largest_cluster": int(max(cluster.n_primary for cluster in clusters)) if clusters else 0,
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
            total_for_transport,
            dt,
            config.gas.temperature,
            boundary=boundary,
        )
    )
    if config.diagnostics.store_pair_summary:
        row.update(components.get("dipole_pair_summary", _empty_pair_summary()))
    if clusters:
        row.update(
            morphology_diagnostics(
                _largest_cluster(clusters),
                electric_field_vector=config.electric_field.vector,
                electric_field_enabled=config.electric_field.enabled,
                contact_tolerance=max(1.0e-12, config.capture_tolerance),
            )
        )
    return row


def _collision_event_metric_row(
    event,
    merged: Cluster,
    cluster_a: Cluster,
    cluster_b: Cluster,
    info,
    config: CoagulationConfig,
    *,
    step: int,
) -> dict[str, float | int]:
    radius_sum = float(cluster_a.radii[info.primary_a] + cluster_b.radii[info.primary_b])
    row = event.to_dict()
    row.update(
        {
            "step": int(step),
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


def snapshot(
    time: float,
    clusters: list[Cluster],
    force_components: dict[str, np.ndarray | dict[str, float]] | None = None,
) -> dict:
    primary_centers = []
    primary_radii = []
    primary_ids = []
    primary_cluster_ids = []
    for cluster in clusters:
        primary_centers.append(cluster.absolute_centers)
        primary_radii.append(cluster.radii)
        primary_ids.append(cluster.primary_ids)
        primary_cluster_ids.append(np.full(cluster.n_primary, cluster.cluster_id, dtype=np.int64))
    snap = {
        "time": float(time),
        "ids": np.asarray([cluster.cluster_id for cluster in clusters], dtype=np.int64),
        "positions": np.asarray([cluster.position for cluster in clusters], dtype=float),
        "velocities": np.asarray([cluster.velocity for cluster in clusters], dtype=float),
        "sizes": np.asarray([cluster.n_primary for cluster in clusters], dtype=np.int64),
        "primary_centers": np.vstack(primary_centers) if primary_centers else np.empty((0, 3)),
        "primary_radii": np.concatenate(primary_radii) if primary_radii else np.empty((0,)),
        "primary_ids": np.concatenate(primary_ids) if primary_ids else np.empty((0,), dtype=np.int64),
        "primary_cluster_ids": np.concatenate(primary_cluster_ids)
        if primary_cluster_ids
        else np.empty((0,), dtype=np.int64),
    }
    if force_components is not None:
        for key, value in (
            ("force_gravity", force_components.get("gravity")),
            ("force_dipole", force_components.get("dipole")),
            ("force_total", force_components.get("total")),
        ):
            if value is not None:
                snap[key] = np.asarray(value, dtype=float)
    return snap


def simulate_coagulation(config: CoagulationConfig) -> CoagulationResult:
    rng = np.random.default_rng(config.seed)
    boundary = Boundary(config.boundary_mode, config.box_size)
    clusters = random_nonoverlap_monomers(config, rng)
    events = []
    event_metrics = []
    stats = [record_stats(0.0, 0, clusters, 0)]
    diagnostics_rows = []
    initial_components = None
    if config.diagnostics.enabled:
        initial_components = deterministic_force_components(
            config,
            clusters,
            boundary,
            include_pair_summary=config.diagnostics.store_pair_summary,
        )
        diagnostics_rows.append(
            _diagnostic_row(
                config,
                clusters,
                boundary,
                time=0.0,
                step=0,
                dt=0.0,
                components=initial_components,
                events_count=0,
            )
        )
    snapshots = [
        snapshot(
            0.0,
            clusters,
            initial_components
            if config.diagnostics.enabled and config.diagnostics.store_snapshot_forces
            else None,
        )
    ]
    time = 0.0
    next_cluster_id = config.n_particles
    step = 0

    for step in range(1, config.max_steps + 1):
        while True:
            hit = first_collision(clusters, boundary=boundary, capture_tolerance=config.capture_tolerance)
            if hit is None:
                break
            i, j, info = hit
            merged, event = merge_clusters(
                clusters[i],
                clusters[j],
                collision=info,
                new_cluster_id=next_cluster_id,
                time=time,
                boundary=boundary,
                project_to_contact=config.project_to_contact,
            )
            if config.diagnostics.enabled and config.diagnostics.store_event_metrics:
                event_metrics.append(
                    _collision_event_metric_row(
                        event,
                        merged,
                        clusters[i],
                        clusters[j],
                        info,
                        config,
                        step=step,
                    )
                )
            next_cluster_id += 1
            for idx in sorted([i, j], reverse=True):
                del clusters[idx]
            clusters.append(merged)
            events.append(event)
        if time >= config.t_end:
            break
        if len(clusters) <= config.target_clusters:
            break
        if config.target_size is not None and max(c.n_primary for c in clusters) >= config.target_size:
            break
        if len(events) >= config.max_events:
            break

        include_pair_summary = config.diagnostics.store_pair_summary and _diagnostics_due(config, step)
        components = deterministic_force_components(
            config,
            clusters,
            boundary,
            include_pair_summary=include_pair_summary,
        )
        forces = np.asarray(components["total"], dtype=float)
        remaining = config.t_end - time
        if remaining <= config.dt_min:
            time = config.t_end
            break
        dt = min(agglomeration_timestep(config, clusters, boundary, forces), remaining)
        if _diagnostics_due(config, step):
            diagnostics_rows.append(
                _diagnostic_row(
                    config,
                    clusters,
                    boundary,
                    time=time,
                    step=step,
                    dt=dt,
                    components=components,
                    events_count=len(events),
                )
            )
        for cluster, force in zip(clusters, forces, strict=True):
            # Each aggregate translates as one rigid body: the EB update uses
            # the cluster COM, total mass, and cluster-level scalar friction.
            result = eb_step(
                cluster.position,
                cluster.velocity,
                cluster.mass,
                cluster.friction,
                force,
                dt,
                config.gas.temperature,
                rng,
                brownian=config.brownian,
                dt_min=config.dt_min * 1.0e-6,
            )
            cluster.position = result.position
            cluster.velocity = result.velocity
            apply_boundary_to_cluster(cluster, boundary)
        time += dt
        if step % config.save_every == 0:
            stats.append(record_stats(time, step, clusters, len(events)))
            snapshots.append(
                snapshot(
                    time,
                    clusters,
                    components
                    if config.diagnostics.enabled and config.diagnostics.store_snapshot_forces
                    else None,
                )
            )

    stats.append(record_stats(time, step, clusters, len(events)))
    final_components = None
    if config.diagnostics.enabled:
        final_components = deterministic_force_components(
            config,
            clusters,
            boundary,
            include_pair_summary=config.diagnostics.store_pair_summary,
        )
    snapshots.append(
        snapshot(
            time,
            clusters,
            final_components
            if config.diagnostics.enabled and config.diagnostics.store_snapshot_forces
            else None,
        )
    )
    return CoagulationResult(
        clusters=clusters,
        events=events,
        stats=stats,
        snapshots=snapshots,
        diagnostics=diagnostics_rows,
        event_metrics=event_metrics,
        config=config.to_dict(),
        time=time,
        steps=step,
    )


def write_coagulation_outputs(
    result: CoagulationResult,
    out_dir: str | Path,
    *,
    make_plots: bool = True,
) -> None:
    out = Path(out_dir)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    with (out / "config_used.yml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result.config, fh, sort_keys=False)
    summary = {
        "mode": "coagulate",
        "time": result.time,
        "steps": result.steps,
        "cluster_count": len(result.clusters),
        "largest_cluster": max(cluster.n_primary for cluster in result.clusters),
        "events": len(result.events),
        "box_size": result.config["box_size"],
        "diameter": result.config["diameter"],
        "boundary_mode": result.config["boundary_mode"],
    }
    with (out / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame([event.to_dict() for event in result.events]).to_csv(out / "events.csv", index=False)
    stats_df = pd.DataFrame(result.stats).drop_duplicates(subset=["time", "cluster_count", "events"])
    stats_df.to_csv(out / "cluster_stats.csv", index=False)
    if result.diagnostics:
        pd.DataFrame(result.diagnostics).to_csv(out / "diagnostics.csv", index=False)
    if result.event_metrics:
        pd.DataFrame(result.event_metrics).to_csv(out / "event_metrics.csv", index=False)
    largest = max(result.clusters, key=lambda c: c.n_primary)
    pd.DataFrame(aggregate_table(largest)).to_csv(out / "final_aggregate.csv", index=False)
    final_diagnostics = morphology_diagnostics(
        largest,
        electric_field_vector=result.config["electric_field"]["vector"],
        electric_field_enabled=bool(result.config["electric_field"]["enabled"]),
        contact_tolerance=max(1.0e-12, float(result.config["capture_tolerance"])),
    )
    if bool(result.config["electric_field"]["enabled"]):
        final_diagnostics.update(
            cluster_dipole_diagnostics(
                [largest],
                ElectricFieldConfig.from_mapping(result.config["electric_field"]),
                float(result.config["diameter"]),
            )
        )
    with (out / "final_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(final_diagnostics, fh, indent=2)
    pd.DataFrame([final_diagnostics]).to_csv(out / "final_diagnostics.csv", index=False)

    _write_run_h5(out / "run.h5", result, summary)
    if make_plots:
        plot_cluster_stats_csv(out / "cluster_stats.csv", plots)
        plot_final_aggregate_csv(out / "final_aggregate.csv", plots)


def _write_run_h5(path: Path, result: CoagulationResult, summary: dict) -> None:
    with h5py.File(path, "w") as h5:
        for key, value in summary.items():
            if isinstance(value, (str, int, float, np.integer, np.floating)):
                h5.attrs[key] = value
        dtype = np.dtype(
            [
                ("time", "f8"),
                ("step", "i8"),
                ("cluster_count", "i8"),
                ("largest_cluster", "i8"),
                ("events", "i8"),
                ("mean_cluster_size", "f8"),
            ]
        )
        arr = np.zeros(len(result.stats), dtype=dtype)
        for i, row in enumerate(result.stats):
            for name in dtype.names:
                arr[name][i] = row[name]
        h5.create_dataset("cluster_stats", data=arr)
        snapshots = h5.create_group("snapshots")
        for i, snap in enumerate(result.snapshots):
            group = snapshots.create_group(f"{i:06d}")
            group.attrs["time"] = snap["time"]
            group.create_dataset("ids", data=snap["ids"])
            group.create_dataset("positions", data=snap["positions"])
            group.create_dataset("velocities", data=snap["velocities"])
            group.create_dataset("sizes", data=snap["sizes"])
            group.create_dataset("primary_centers", data=snap["primary_centers"])
            group.create_dataset("primary_radii", data=snap["primary_radii"])
            group.create_dataset("primary_ids", data=snap["primary_ids"])
            group.create_dataset("primary_cluster_ids", data=snap["primary_cluster_ids"])
            for key in ("force_gravity", "force_dipole", "force_total"):
                if key in snap:
                    group.create_dataset(key, data=snap[key])
        _write_rows_group(h5, "diagnostics", result.diagnostics)
        _write_rows_group(h5, "event_metrics", result.event_metrics)
        largest = max(result.clusters, key=lambda c: c.n_primary)
        final_diagnostics = morphology_diagnostics(
            largest,
            electric_field_vector=result.config["electric_field"]["vector"],
            electric_field_enabled=bool(result.config["electric_field"]["enabled"]),
            contact_tolerance=max(1.0e-12, float(result.config["capture_tolerance"])),
        )
        if bool(result.config["electric_field"]["enabled"]):
            final_diagnostics.update(
                cluster_dipole_diagnostics(
                    [largest],
                    ElectricFieldConfig.from_mapping(result.config["electric_field"]),
                    float(result.config["diameter"]),
                )
            )
        _write_rows_group(h5, "final_diagnostics", [final_diagnostics])


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


def run_coagulation(
    config: CoagulationConfig,
    out_dir: str | Path | None = None,
    *,
    make_plots: bool = True,
) -> CoagulationResult:
    result = simulate_coagulation(config)
    if out_dir is not None:
        write_coagulation_outputs(result, out_dir, make_plots=make_plots)
    return result
