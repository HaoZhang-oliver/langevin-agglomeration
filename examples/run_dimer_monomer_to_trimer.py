"""Simulate one dimer and one monomer until they stick into a trimer.

Edit the variables in the USER INPUTS section to tune the run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import yaml

from ldagg.analysis import aggregate_table
from ldagg.boundaries import Boundary, apply_boundary_to_cluster
from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL, Cluster, dimer_seed
from ldagg.collisions import find_collision, merge_clusters, nearest_surface_gap
from ldagg.constants import BOLTZMANN, DEFAULT_PARTICLE_DENSITY
from ldagg.gas import Gas
from ldagg.integrators import eb_step
from ldagg.plotting import (
    plot_agglomeration_snapshot,
    plot_agglomeration_trajectories,
    save_agglomeration_snapshots,
    save_agglomeration_video,
)
from ldagg.simulation import record_stats, snapshot


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ----------------------------- USER INPUTS -----------------------------

PRIMARY_DIAMETER_M = 100.0e-9
PARTICLE_DENSITY_KG_M3 = DEFAULT_PARTICLE_DENSITY
FRICTION_MODEL = DEFAULT_CLUSTER_FRICTION_MODEL
BOX_SIZE_M = 9.0e-7

INITIAL_GAP_M = 200.0e-9
INITIAL_APPROACH_SPEED_M_S = 0.8
MOVE_DIMER = True
BROWNIAN = True
RANDOM_SEED = 20260623

MAX_TIME_S = 5.0e-4
MAX_STEPS = 100_000
DT_FACTOR = 0.05
DT_MIN_S = 1.0e-12
DT_MAX_S = 1.0e-9
GAP_FLOOR_FRACTION = 0.01
CAPTURE_TOLERANCE_M = 0.0

SAVE_EVERY_STEPS = 1
DIAGNOSTIC_EVERY_STEPS = 100
OUTPUT_DIR = Path("outputs/dimer_monomer_to_trimer")

RENDER_FIGURES = env_bool("LDAGG_RENDER", True)
VISUALIZATION_BACKEND = os.getenv("LDAGG_VIS_BACKEND", "pyvista")
SAVE_PNG_SNAPSHOTS = env_bool("LDAGG_SAVE_SNAPSHOTS", True)
SAVE_VIDEO = env_bool("LDAGG_SAVE_VIDEO", True)
MAX_RENDERED_FRAMES = 120
VIDEO_FPS = 10
VIEW_ELEVATION_DEG = 30.0
VIEW_AZIMUTH_DEG = 30.0

# -----------------------------------------------------------------------


def make_initial_clusters(gas: Gas, rng: np.random.Generator) -> list[Cluster]:
    center = np.full(3, 0.5 * BOX_SIZE_M)
    dimer = dimer_seed(
        0,
        PRIMARY_DIAMETER_M,
        density=PARTICLE_DENSITY_KG_M3,
        gas=gas,
        friction_model=FRICTION_MODEL,
    )
    dimer.position = center.copy()
    dimer.primary_ids = np.array([0, 1], dtype=np.int64)

    monomer_x = center[0] + 1.5 * PRIMARY_DIAMETER_M + INITIAL_GAP_M
    monomer_position = np.array([monomer_x, center[1], center[2]], dtype=float)
    monomer_surface_max = monomer_position[0] + 0.5 * PRIMARY_DIAMETER_M
    boundary_tol = 1.0e-18
    if monomer_surface_max > BOX_SIZE_M + boundary_tol:
        raise ValueError(
            "BOX_SIZE_M is too small for the requested INITIAL_GAP_M; "
            f"need at least {monomer_surface_max:.3e} m, got {BOX_SIZE_M:.3e} m"
        )
    if monomer_surface_max > BOX_SIZE_M:
        monomer_position[0] -= monomer_surface_max - BOX_SIZE_M

    mass_probe = Cluster.monomer(
        2,
        monomer_position,
        np.zeros(3),
        PRIMARY_DIAMETER_M,
        density=PARTICLE_DENSITY_KG_M3,
        gas=gas,
    )
    thermal_speed = np.sqrt(BOLTZMANN * gas.temperature / mass_probe.mass)
    velocity = thermal_speed * rng.normal(size=3)
    velocity[0] -= INITIAL_APPROACH_SPEED_M_S
    monomer = Cluster.monomer(
        2,
        monomer_position,
        velocity,
        PRIMARY_DIAMETER_M,
        density=PARTICLE_DENSITY_KG_M3,
        gas=gas,
        friction_model=FRICTION_MODEL,
    )
    monomer.primary_ids = np.array([2], dtype=np.int64)
    return [dimer, monomer]


def choose_timestep(gas: Gas, clusters: list[Cluster], boundary: Boundary) -> float:
    gap = nearest_surface_gap(clusters, boundary=boundary)
    min_radius = min(float(np.min(cluster.radii)) for cluster in clusters)
    gap_scale = max(gap, GAP_FLOOR_FRACTION * min_radius)
    f_min = min(cluster.friction for cluster in clusters)
    dt_diff = gap_scale * gap_scale * f_min / (6.0 * BOLTZMANN * gas.temperature)
    dt = min(DT_FACTOR * dt_diff, DT_MAX_S)
    return float(np.clip(dt, DT_MIN_S, DT_MAX_S))


def write_run_h5(path: Path, stats: list[dict], snapshots: list[dict], summary: dict) -> Path:
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            path = path.with_name(f"{path.stem}_{os.getpid()}{path.suffix}")
            print(f"  existing run.h5 is locked; writing {path.name} instead")
    with h5py.File(path, "w") as h5:
        for key, value in summary.items():
            if isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating)):
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
        arr = np.zeros(len(stats), dtype=dtype)
        for i, row in enumerate(stats):
            for name in dtype.names:
                arr[name][i] = row[name]
        h5.create_dataset("cluster_stats", data=arr)
        group = h5.create_group("snapshots")
        for i, snap in enumerate(snapshots):
            sg = group.create_group(f"{i:06d}")
            sg.attrs["time"] = snap["time"]
            for key in (
                "ids",
                "positions",
                "velocities",
                "sizes",
                "primary_centers",
                "primary_radii",
                "primary_ids",
                "primary_cluster_ids",
            ):
                sg.create_dataset(key, data=snap[key])
    return path


def main() -> None:
    gas = Gas()
    rng = np.random.default_rng(RANDOM_SEED)
    boundary = Boundary("finite", BOX_SIZE_M)
    clusters = make_initial_clusters(gas, rng)
    events = []
    stats = [record_stats(0.0, 0, clusters, 0)]
    snapshots = [snapshot(0.0, clusters)]
    time = 0.0

    print("Dimer + monomer LD run")
    print(f"  primary diameter: {PRIMARY_DIAMETER_M:.3e} m")
    print(f"  initial surface gap: {INITIAL_GAP_M:.3e} m")
    print(f"  cluster friction model: {FRICTION_MODEL}")
    print(f"  box size: {BOX_SIZE_M:.3e} m")
    print(f"  dt_max: {DT_MAX_S:.3e} s")
    print(f"  capture_tolerance: {CAPTURE_TOLERANCE_M:.3e} m")
    print(f"  seed: {RANDOM_SEED}")
    print(f"  output: {OUTPUT_DIR}")

    for step in range(1, MAX_STEPS + 1):
        hit = find_collision(clusters[0], clusters[1], boundary=boundary, capture_tolerance=CAPTURE_TOLERANCE_M)
        if hit is not None:
            merged, event = merge_clusters(
                clusters[0],
                clusters[1],
                collision=hit,
                new_cluster_id=3,
                time=time,
                boundary=None,
                project_to_contact=True,
            )
            clusters = [merged]
            events.append(event)
            stats.append(record_stats(time, step, clusters, len(events)))
            snapshots.append(snapshot(time, clusters))
            print(
                "Collision detected: "
                f"t={time:.6e} s, step={step}, "
                f"{event.size_a}+{event.size_b}->{event.new_size}, "
                f"trigger distance={event.distance:.3e} m"
            )
            break

        if time >= MAX_TIME_S:
            print(f"Reached MAX_TIME_S={MAX_TIME_S:.3e} s before contact")
            break

        gap = nearest_surface_gap(clusters, boundary=boundary)
        dt = min(choose_timestep(gas, clusters, boundary), MAX_TIME_S - time)
        for i, cluster in enumerate(clusters):
            if i == 0 and not MOVE_DIMER:
                continue
            result = eb_step(
                cluster.position,
                cluster.velocity,
                cluster.mass,
                cluster.friction,
                np.zeros(3),
                dt,
                gas.temperature,
                rng,
                brownian=BROWNIAN,
                dt_min=DT_MIN_S * 1.0e-6,
            )
            cluster.position = result.position
            cluster.velocity = result.velocity
            apply_boundary_to_cluster(cluster, boundary)

        time += dt
        if step % SAVE_EVERY_STEPS == 0:
            stats.append(record_stats(time, step, clusters, len(events)))
            snapshots.append(snapshot(time, clusters))
        if step % DIAGNOSTIC_EVERY_STEPS == 0:
            print(
                f"  step={step:6d}, t={time:.3e} s, "
                f"gap={gap:.3e} m, dt={dt:.3e} s"
            )
    else:
        print(f"Reached MAX_STEPS={MAX_STEPS} before contact")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    snapshots_dir = OUTPUT_DIR / "snapshots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    largest = max(cluster.n_primary for cluster in clusters)
    summary = {
        "mode": "dimer_monomer_to_trimer",
        "time": time,
        "steps": step,
        "cluster_count": len(clusters),
        "largest_cluster": largest,
        "events": len(events),
        "reached_trimer": bool(largest >= 3),
        "box_size": BOX_SIZE_M,
        "diameter": PRIMARY_DIAMETER_M,
        "initial_gap": INITIAL_GAP_M,
        "capture_tolerance": CAPTURE_TOLERANCE_M,
        "friction_model": FRICTION_MODEL,
    }

    config_used = {
        "primary_diameter_m": PRIMARY_DIAMETER_M,
        "particle_density_kg_m3": PARTICLE_DENSITY_KG_M3,
        "box_size_m": BOX_SIZE_M,
        "initial_gap_m": INITIAL_GAP_M,
        "initial_approach_speed_m_s": INITIAL_APPROACH_SPEED_M_S,
        "move_dimer": MOVE_DIMER,
        "brownian": BROWNIAN,
        "random_seed": RANDOM_SEED,
        "max_time_s": MAX_TIME_S,
        "max_steps": MAX_STEPS,
        "dt_factor": DT_FACTOR,
        "dt_min_s": DT_MIN_S,
        "dt_max_s": DT_MAX_S,
        "capture_tolerance_m": CAPTURE_TOLERANCE_M,
        "friction_model": FRICTION_MODEL,
        "gas": gas.to_dict(),
    }
    with (OUTPUT_DIR / "config_used.yml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_used, fh, sort_keys=False)
    with (OUTPUT_DIR / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame([event.to_dict() for event in events]).to_csv(OUTPUT_DIR / "events.csv", index=False)
    pd.DataFrame(stats).to_csv(OUTPUT_DIR / "cluster_stats.csv", index=False)
    final_cluster = max(clusters, key=lambda cluster: cluster.n_primary)
    pd.DataFrame(aggregate_table(final_cluster)).to_csv(OUTPUT_DIR / "final_aggregate.csv", index=False)
    run_h5 = OUTPUT_DIR / "run.h5"
    run_h5 = write_run_h5(run_h5, stats, snapshots, summary)

    print("Run complete")
    print(f"  simulated time: {time:.6e} s")
    print(f"  integration steps: {step}")
    print(f"  collision events: {len(events)}")
    print(f"  final cluster count: {len(clusters)}")
    print(f"  largest cluster size: {largest}")
    print(f"  reached trimer: {largest >= 3}")

    if RENDER_FIGURES:
        print("Rendering trajectory, snapshots, and video...")
        plot_agglomeration_trajectories(
            run_h5,
            plots_dir / "primary_trajectories.png",
            backend=VISUALIZATION_BACKEND,
            max_frames=MAX_RENDERED_FRAMES,
            box_size=BOX_SIZE_M,
            elev=VIEW_ELEVATION_DEG,
            azim=VIEW_AZIMUTH_DEG,
        )
        snapshot_result = plot_agglomeration_snapshot(
            snapshots[-1],
            plots_dir / "final_snapshot.png",
            backend=VISUALIZATION_BACKEND,
            box_size=BOX_SIZE_M,
            elev=VIEW_ELEVATION_DEG,
            azim=VIEW_AZIMUTH_DEG,
            title=f"Final state, largest cluster = {largest}",
            show_com=False,
        )
        if VISUALIZATION_BACKEND == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close(snapshot_result[0])
        closeup_result = plot_agglomeration_snapshot(
            snapshots[-1],
            plots_dir / "final_snapshot_closeup.png",
            backend=VISUALIZATION_BACKEND,
            box_size=None,
            elev=VIEW_ELEVATION_DEG,
            azim=VIEW_AZIMUTH_DEG,
            show_box=False,
            show_com=False,
            sphere_resolution=24,
            title="Final trimer close-up",
        )
        if VISUALIZATION_BACKEND == "matplotlib":
            plt.close(closeup_result[0])

        if SAVE_PNG_SNAPSHOTS:
            paths = save_agglomeration_snapshots(
                run_h5,
                snapshots_dir,
                backend=VISUALIZATION_BACKEND,
                max_frames=MAX_RENDERED_FRAMES,
                box_size=BOX_SIZE_M,
                elev=VIEW_ELEVATION_DEG,
                azim=VIEW_AZIMUTH_DEG,
            )
            print(f"  saved {len(paths)} PNG snapshots to {snapshots_dir}")
        if SAVE_VIDEO:
            video_path = save_agglomeration_video(
                run_h5,
                plots_dir / "dimer_monomer_to_trimer.mp4",
                backend=VISUALIZATION_BACKEND,
                max_frames=MAX_RENDERED_FRAMES,
                fps=VIDEO_FPS,
                box_size=BOX_SIZE_M,
                elev=VIEW_ELEVATION_DEG,
                azim=VIEW_AZIMUTH_DEG,
            )
            print(f"  saved video to {video_path}")
    else:
        print("Rendering disabled by LDAGG_RENDER=0")

    print(f"Summary JSON: {OUTPUT_DIR / 'run_summary.json'}")
    print(f"Events CSV:   {OUTPUT_DIR / 'events.csv'}")
    print(f"Run HDF5:     {run_h5}")


if __name__ == "__main__":
    main()
