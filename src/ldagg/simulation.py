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
    seed: int = 2024
    save_every: int = 20
    project_to_contact: bool = True

    @classmethod
    def from_mapping(cls, data: dict | None) -> CoagulationConfig:
        data = dict(data or {})
        gas = Gas.from_mapping(data.pop("gas", None))
        return cls(gas=gas, **data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gas"] = self.gas.to_dict()
        return data


@dataclass(slots=True)
class CoagulationResult:
    clusters: list[Cluster]
    events: list
    stats: list[dict]
    snapshots: list[dict]
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


def agglomeration_timestep(config: CoagulationConfig, clusters: list[Cluster], boundary: Boundary) -> float:
    """Conservative gap-based timestep for agglomeration."""

    if len(clusters) < 2:
        return min(config.dt_max, max(config.dt_min, config.t_end))
    gap = nearest_surface_gap(clusters, boundary=boundary)
    min_radius = min(float(np.min(cluster.radii)) for cluster in clusters)
    gap_scale = max(gap, config.gap_floor_fraction * min_radius)
    f_min = min(cluster.friction for cluster in clusters)
    dt_diff = gap_scale * gap_scale * f_min / (6.0 * BOLTZMANN * config.gas.temperature)
    if config.gravity_enabled:
        forces = [cluster.mass * np.linalg.norm(config.gravity) for cluster in clusters]
        force_max = max(max(forces), 1.0e-300)
        dt_force = gap_scale * f_min / force_max
    else:
        dt_force = np.inf
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


def snapshot(time: float, clusters: list[Cluster]) -> dict:
    primary_centers = []
    primary_radii = []
    primary_ids = []
    primary_cluster_ids = []
    for cluster in clusters:
        primary_centers.append(cluster.absolute_centers)
        primary_radii.append(cluster.radii)
        primary_ids.append(cluster.primary_ids)
        primary_cluster_ids.append(np.full(cluster.n_primary, cluster.cluster_id, dtype=np.int64))
    return {
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


def simulate_coagulation(config: CoagulationConfig) -> CoagulationResult:
    rng = np.random.default_rng(config.seed)
    boundary = Boundary(config.boundary_mode, config.box_size)
    clusters = random_nonoverlap_monomers(config, rng)
    events = []
    stats = [record_stats(0.0, 0, clusters, 0)]
    snapshots = [snapshot(0.0, clusters)]
    time = 0.0
    next_cluster_id = config.n_particles
    gravity = np.asarray(config.gravity, dtype=float)
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

        dt = min(agglomeration_timestep(config, clusters, boundary), config.t_end - time)
        for cluster in clusters:
            # Each aggregate translates as one rigid body: the EB update uses
            # the cluster COM, total mass, and cluster-level scalar friction.
            force = cluster.mass * gravity if config.gravity_enabled else np.zeros(3)
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
            snapshots.append(snapshot(time, clusters))

    stats.append(record_stats(time, step, clusters, len(events)))
    snapshots.append(snapshot(time, clusters))
    return CoagulationResult(
        clusters=clusters,
        events=events,
        stats=stats,
        snapshots=snapshots,
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
    largest = max(result.clusters, key=lambda c: c.n_primary)
    pd.DataFrame(aggregate_table(largest)).to_csv(out / "final_aggregate.csv", index=False)

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
