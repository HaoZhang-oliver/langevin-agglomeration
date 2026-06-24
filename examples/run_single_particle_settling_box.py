"""Simulate one Brownian particle settling under gravity in a finite box.

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
from ldagg.clusters import Cluster
from ldagg.constants import BOLTZMANN, DEFAULT_GRAVITY, DEFAULT_PARTICLE_DENSITY
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

PRIMARY_DIAMETER_M = 1.0e-6
PARTICLE_DENSITY_KG_M3 = DEFAULT_PARTICLE_DENSITY
BOX_SIZE_M = 20.0e-6
INITIAL_POSITION_M = np.array([10.0e-6, 10.0e-6, 18.0e-6])

GRAVITY_M_S2 = DEFAULT_GRAVITY
BROWNIAN = True
RANDOM_SEED = 20260623

MAX_TIME_S = 0.8
MAX_STEPS = 50_000
DT_FACTOR = 0.05
DT_MIN_S = 1.0e-10
DT_MAX_S = 5.0e-4
FLOOR_CAPTURE_TOLERANCE_M = 1.0e-9
SAVE_EVERY_STEPS = 1
DIAGNOSTIC_EVERY_STEPS = 1000

OUTPUT_DIR = Path("outputs/single_particle_settling_box")

RENDER_FIGURES = env_bool("LDAGG_RENDER", True)
VISUALIZATION_BACKEND = os.getenv("LDAGG_VIS_BACKEND", "pyvista")
SAVE_PNG_SNAPSHOTS = env_bool("LDAGG_SAVE_SNAPSHOTS", True)
SAVE_VIDEO = env_bool("LDAGG_SAVE_VIDEO", True)
MAX_RENDERED_FRAMES = 120
VIDEO_FPS = 12
VIEW_ELEVATION_DEG = 30.0
VIEW_AZIMUTH_DEG = 30.0

# -----------------------------------------------------------------------


def choose_timestep(
    z_gap: float,
    friction: float,
    force: np.ndarray,
    temperature: float,
) -> float:
    """Tutorial-style settling timestep, capped for smooth visualization."""

    z_gap = max(float(z_gap), 1.0e-30)
    force_mag = float(np.linalg.norm(force))
    dt_diff = z_gap * z_gap * friction / (6.0 * BOLTZMANN * temperature)
    dt_force = np.inf if force_mag == 0.0 else z_gap * friction / force_mag
    dt = DT_FACTOR * min(dt_diff, dt_force, DT_MAX_S)
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


def plot_settling_diagnostics(history: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True)
    axes[0].plot(history["time_s"], history["z_m"] * 1.0e6, color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("z (um)")
    axes[0].set_title("Single-particle settling")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(history["time_s"], history["vz_m_s"] * 1.0e6, color="tab:red", linewidth=1.0)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("v_z (um/s)")
    axes[1].grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "settling_z_velocity.png", dpi=160)
    plt.close(fig)


def main() -> None:
    gas = Gas()
    rng = np.random.default_rng(RANDOM_SEED)
    gravity = np.asarray(GRAVITY_M_S2, dtype=float)
    boundary = Boundary("finite", BOX_SIZE_M, absorbing_floor=True, floor_z=0.0)

    particle = Cluster.monomer(
        0,
        INITIAL_POSITION_M.copy(),
        np.zeros(3),
        PRIMARY_DIAMETER_M,
        density=PARTICLE_DENSITY_KG_M3,
        gas=gas,
    )
    particle.primary_ids = np.array([0], dtype=np.int64)
    thermal_speed = np.sqrt(BOLTZMANN * gas.temperature / particle.mass)
    particle.velocity = thermal_speed * rng.normal(size=3)

    force = particle.mass * gravity
    terminal_velocity = force / particle.friction
    diffusion = BOLTZMANN * gas.temperature / particle.friction
    relaxation_time = particle.mass / particle.friction
    radius = 0.5 * PRIMARY_DIAMETER_M
    initial_floor_gap = float(particle.position[2] - radius)

    clusters = [particle]
    stats = [record_stats(0.0, 0, clusters, 0)]
    snapshots = [snapshot(0.0, clusters)]
    history_rows = [
        {
            "time_s": 0.0,
            "x_m": particle.position[0],
            "y_m": particle.position[1],
            "z_m": particle.position[2],
            "vx_m_s": particle.velocity[0],
            "vy_m_s": particle.velocity[1],
            "vz_m_s": particle.velocity[2],
            "dt_s": 0.0,
            "absorbed": False,
        }
    ]

    print("Single-particle Brownian settling in a finite box")
    print(f"  primary diameter: {PRIMARY_DIAMETER_M:.3e} m")
    print(f"  box size: {BOX_SIZE_M:.3e} m")
    print(f"  initial position: {INITIAL_POSITION_M}")
    print(f"  Brownian enabled: {BROWNIAN}")
    print(f"  gravity: {gravity} m/s^2")
    print(f"  terminal velocity: {terminal_velocity} m/s")
    print(f"  diffusion coefficient: {diffusion:.3e} m^2/s")
    print(f"  velocity relaxation time m/f: {relaxation_time:.3e} s")
    print(f"  deterministic floor-contact estimate: {initial_floor_gap / abs(terminal_velocity[2]):.3e} s")
    print(f"  floor capture tolerance: {FLOOR_CAPTURE_TOLERANCE_M:.3e} m")
    print(f"  seed: {RANDOM_SEED}")
    print(f"  output: {OUTPUT_DIR}")

    time = 0.0
    absorbed = False
    for step in range(1, MAX_STEPS + 1):
        floor_gap = float(particle.position[2] - radius - boundary.floor_z)
        if floor_gap <= FLOOR_CAPTURE_TOLERANCE_M:
            particle.position[2] = boundary.floor_z + radius
            absorbed = True
            stats.append(record_stats(time, step, clusters, 0))
            snapshots.append(snapshot(time, clusters))
            history_rows.append(
                {
                    "time_s": time,
                    "x_m": particle.position[0],
                    "y_m": particle.position[1],
                    "z_m": particle.position[2],
                    "vx_m_s": particle.velocity[0],
                    "vy_m_s": particle.velocity[1],
                    "vz_m_s": particle.velocity[2],
                    "dt_s": 0.0,
                    "absorbed": True,
                }
            )
            print(
                f"Particle reached absorbing floor tolerance: "
                f"t={time:.6e} s, step={step}, gap={floor_gap:.3e} m"
            )
            break
        if time >= MAX_TIME_S:
            print(f"Reached MAX_TIME_S={MAX_TIME_S:.3e} s before floor contact")
            break

        dt = min(choose_timestep(floor_gap, particle.friction, force, gas.temperature), MAX_TIME_S - time)
        result = eb_step(
            particle.position,
            particle.velocity,
            particle.mass,
            particle.friction,
            force,
            dt,
            gas.temperature,
            rng,
            brownian=BROWNIAN,
            dt_min=DT_MIN_S * 1.0e-6,
        )
        particle.position = result.position
        particle.velocity = result.velocity
        time += result.dt
        _cluster, absorbed = apply_boundary_to_cluster(particle, boundary)

        if step % SAVE_EVERY_STEPS == 0 or absorbed:
            stats.append(record_stats(time, step, clusters, 0))
            snapshots.append(snapshot(time, clusters))
            history_rows.append(
                {
                    "time_s": time,
                    "x_m": particle.position[0],
                    "y_m": particle.position[1],
                    "z_m": particle.position[2],
                    "vx_m_s": particle.velocity[0],
                    "vy_m_s": particle.velocity[1],
                    "vz_m_s": particle.velocity[2],
                    "dt_s": result.dt,
                    "absorbed": bool(absorbed),
                }
            )
        if step % DIAGNOSTIC_EVERY_STEPS == 0 or absorbed:
            print(
                f"  step={step:6d}, t={time:.3e} s, "
                f"z={particle.position[2]:.3e} m, "
                f"floor_gap={max(particle.position[2] - radius, 0.0):.3e} m, "
                f"dt={result.dt:.3e} s, absorbed={absorbed}"
            )
        if absorbed:
            print(f"Particle reached absorbing floor: t={time:.6e} s, step={step}")
            break
    else:
        print(f"Reached MAX_STEPS={MAX_STEPS} before floor contact")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    snapshots_dir = OUTPUT_DIR / "snapshots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(history_rows)
    summary = {
        "mode": "single_particle_settling_box",
        "time": time,
        "steps": step,
        "absorbed": bool(absorbed),
        "cluster_count": 1,
        "largest_cluster": 1,
        "events": 0,
        "box_size": BOX_SIZE_M,
        "diameter": PRIMARY_DIAMETER_M,
        "floor_capture_tolerance": FLOOR_CAPTURE_TOLERANCE_M,
        "terminal_velocity_z_m_s": float(terminal_velocity[2]),
        "diffusion_m2_s": float(diffusion),
        "relaxation_time_s": float(relaxation_time),
    }
    config_used = {
        "primary_diameter_m": PRIMARY_DIAMETER_M,
        "particle_density_kg_m3": PARTICLE_DENSITY_KG_M3,
        "box_size_m": BOX_SIZE_M,
        "initial_position_m": [float(value) for value in INITIAL_POSITION_M],
        "gravity_m_s2": [float(value) for value in gravity],
        "brownian": BROWNIAN,
        "random_seed": RANDOM_SEED,
        "max_time_s": MAX_TIME_S,
        "max_steps": MAX_STEPS,
        "dt_factor": DT_FACTOR,
        "dt_min_s": DT_MIN_S,
        "dt_max_s": DT_MAX_S,
        "floor_capture_tolerance_m": FLOOR_CAPTURE_TOLERANCE_M,
        "gas": gas.to_dict(),
    }
    with (OUTPUT_DIR / "config_used.yml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_used, fh, sort_keys=False)
    with (OUTPUT_DIR / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame().to_csv(OUTPUT_DIR / "events.csv", index=False)
    pd.DataFrame(stats).to_csv(OUTPUT_DIR / "cluster_stats.csv", index=False)
    pd.DataFrame(aggregate_table(particle)).to_csv(OUTPUT_DIR / "final_aggregate.csv", index=False)
    history.to_csv(OUTPUT_DIR / "particle_trajectory.csv", index=False)
    run_h5 = write_run_h5(OUTPUT_DIR / "run.h5", stats, snapshots, summary)

    print("Run complete")
    print(f"  simulated time: {time:.6e} s")
    print(f"  integration steps: {step}")
    print(f"  absorbed by floor: {absorbed}")
    print(f"  final position: {particle.position}")
    print(f"  saved trajectory rows: {len(history)}")

    if RENDER_FIGURES:
        print("Rendering PyVista trajectory, snapshots, and video...")
        plot_settling_diagnostics(history, plots_dir)
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
            title=f"Final state, absorbed = {absorbed}",
            show_com=False,
        )
        if VISUALIZATION_BACKEND == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close(snapshot_result[0])
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
                plots_dir / "single_particle_settling_box.mp4",
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
    print(f"Trajectory CSV: {OUTPUT_DIR / 'particle_trajectory.csv'}")
    print(f"Run HDF5: {run_h5}")


if __name__ == "__main__":
    main()
