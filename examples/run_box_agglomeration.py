"""Random monomer agglomeration in a 3D box.

Edit the variables in the USER INPUTS section to tune the run.
"""

from __future__ import annotations

import os
from pathlib import Path

from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL
from ldagg.constants import DEFAULT_GRAVITY, DEFAULT_PARTICLE_DENSITY
from ldagg.plotting import (
    plot_agglomeration_snapshot,
    plot_agglomeration_trajectories,
    save_agglomeration_snapshots,
    save_agglomeration_video,
)
from ldagg.simulation import CoagulationConfig, run_coagulation


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ----------------------------- USER INPUTS -----------------------------

N_PARTICLES = 10
PRIMARY_DIAMETER_M = 30.0e-9
PARTICLE_DENSITY_KG_M3 = DEFAULT_PARTICLE_DENSITY
BOX_SIZE_M = 1.0e-6
BOUNDARY_MODE = "periodic"  # "finite" reflecting box, or "periodic" cube

BROWNIAN = True
GRAVITY_ENABLED = True
GRAVITY_M_S2 = tuple(float(value) for value in DEFAULT_GRAVITY)
FRICTION_MODEL = DEFAULT_CLUSTER_FRICTION_MODEL  # volume-equivalent sphere drag by default
RANDOM_SEED = 20260623

T_END_S = 2.0e-4
MAX_STEPS = 50_000
MAX_EVENTS = 50
TARGET_CLUSTERS = 1
TARGET_SIZE = None  # set an integer to stop once largest cluster reaches that size
DT_FACTOR = 0.08
DT_MIN_S = 1.0e-12
DT_MAX_S = 2.0e-7
GAP_FLOOR_FRACTION = 0.01
CAPTURE_TOLERANCE_M = 8.0e-9
SAVE_EVERY_STEPS = 5
PROJECT_TO_CONTACT = True

OUTPUT_DIR = Path("outputs/box_agglomeration")

RENDER_FIGURES = env_bool("LDAGG_RENDER", True)
VISUALIZATION_BACKEND = os.getenv("LDAGG_VIS_BACKEND", "pyvista")
SAVE_PNG_SNAPSHOTS = env_bool("LDAGG_SAVE_SNAPSHOTS", True)
SAVE_VIDEO = env_bool("LDAGG_SAVE_VIDEO", True)
SNAPSHOT_EVERY_SAVED_FRAME = 1
MAX_RENDERED_FRAMES = 120
VIDEO_FPS = 10
VIEW_ELEVATION_DEG = 30.0
VIEW_AZIMUTH_DEG = 30.0

# -----------------------------------------------------------------------


def build_config() -> CoagulationConfig:
    return CoagulationConfig(
        n_particles=N_PARTICLES,
        diameter=PRIMARY_DIAMETER_M,
        particle_density=PARTICLE_DENSITY_KG_M3,
        box_size=BOX_SIZE_M,
        boundary_mode=BOUNDARY_MODE,
        t_end=T_END_S,
        target_clusters=TARGET_CLUSTERS,
        target_size=TARGET_SIZE,
        max_events=MAX_EVENTS,
        max_steps=MAX_STEPS,
        dt_factor=DT_FACTOR,
        dt_min=DT_MIN_S,
        dt_max=DT_MAX_S,
        gap_floor_fraction=GAP_FLOOR_FRACTION,
        capture_tolerance=CAPTURE_TOLERANCE_M,
        gravity_enabled=GRAVITY_ENABLED,
        gravity=GRAVITY_M_S2,
        brownian=BROWNIAN,
        friction_model=FRICTION_MODEL,
        seed=RANDOM_SEED,
        save_every=SAVE_EVERY_STEPS,
        project_to_contact=PROJECT_TO_CONTACT,
    )


def print_event_diagnostics(result) -> None:
    if not result.events:
        print("Collision diagnostics: no collisions recorded")
        return
    print("Collision diagnostics:")
    for event in result.events:
        print(
            "  "
            f"t={event.time:.3e} s, "
            f"{event.size_a}+{event.size_b}->{event.new_size}, "
            f"clusters {event.cluster_a}/{event.cluster_b} -> {event.new_cluster_id}, "
            f"trigger distance={event.distance:.3e} m"
        )


def print_final_cluster_diagnostics(result) -> None:
    print("Final cluster diagnostics:")
    for cluster in sorted(result.clusters, key=lambda item: item.cluster_id):
        print(
            "  "
            f"cluster={cluster.cluster_id}, "
            f"n_primary={cluster.n_primary}, "
            f"mass={cluster.mass:.3e} kg, "
            f"d_ve={cluster.volume_equivalent_diameter:.3e} m, "
            f"friction={cluster.friction:.3e} kg/s, "
            f"COM=({cluster.position[0]:.3e}, {cluster.position[1]:.3e}, {cluster.position[2]:.3e})"
        )


def main() -> None:
    config = build_config()

    print("Random box agglomeration run")
    print(f"  monomers: {N_PARTICLES}")
    print(f"  primary diameter: {PRIMARY_DIAMETER_M:.3e} m")
    print(f"  box size: {BOX_SIZE_M:.3e} m")
    print(f"  boundary mode: {BOUNDARY_MODE}")
    print(f"  Brownian enabled: {BROWNIAN}")
    print(f"  gravity enabled: {GRAVITY_ENABLED}, gravity={GRAVITY_M_S2} m/s^2")
    print(f"  cluster friction model: {FRICTION_MODEL}")
    print(f"  capture tolerance: {CAPTURE_TOLERANCE_M:.3e} m")
    print(f"  t_end: {T_END_S:.3e} s, dt_max: {DT_MAX_S:.3e} s")
    print(f"  stop target clusters: {TARGET_CLUSTERS}, target size: {TARGET_SIZE}")
    print(f"  seed: {RANDOM_SEED}")
    print(f"  output: {OUTPUT_DIR}")
    print("Running LD coagulation...")

    result = run_coagulation(config, OUTPUT_DIR, make_plots=False)
    run_h5 = OUTPUT_DIR / "run.h5"
    plots_dir = OUTPUT_DIR / "plots"
    snapshots_dir = OUTPUT_DIR / "snapshots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    largest = max(cluster.n_primary for cluster in result.clusters)
    print("Run complete")
    print(f"  simulated time: {result.time:.6e} s")
    print(f"  integration steps: {result.steps}")
    print(f"  collision events: {len(result.events)}")
    print(f"  initial cluster count: {N_PARTICLES}")
    print(f"  final cluster count: {len(result.clusters)}")
    print(f"  largest cluster size: {largest}")
    print_event_diagnostics(result)
    print_final_cluster_diagnostics(result)

    if RENDER_FIGURES:
        print("Rendering trajectory, snapshots, and video...")
        trajectory_result = plot_agglomeration_trajectories(
            run_h5,
            plots_dir / "primary_trajectories.png",
            backend=VISUALIZATION_BACKEND,
            max_frames=MAX_RENDERED_FRAMES,
            box_size=BOX_SIZE_M,
            elev=VIEW_ELEVATION_DEG,
            azim=VIEW_AZIMUTH_DEG,
        )
        if VISUALIZATION_BACKEND == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close(trajectory_result[0])

        if result.snapshots:
            snapshot_result = plot_agglomeration_snapshot(
                result.snapshots[-1],
                plots_dir / "final_snapshot.png",
                backend=VISUALIZATION_BACKEND,
                box_size=BOX_SIZE_M,
                elev=VIEW_ELEVATION_DEG,
                azim=VIEW_AZIMUTH_DEG,
                title=f"Final state, largest cluster = {largest}",
                show_com=False,
            )
            if VISUALIZATION_BACKEND == "matplotlib":
                plt.close(snapshot_result[0])

        if SAVE_PNG_SNAPSHOTS:
            paths = save_agglomeration_snapshots(
                run_h5,
                snapshots_dir,
                backend=VISUALIZATION_BACKEND,
                every=SNAPSHOT_EVERY_SAVED_FRAME,
                max_frames=MAX_RENDERED_FRAMES,
                box_size=BOX_SIZE_M,
                elev=VIEW_ELEVATION_DEG,
                azim=VIEW_AZIMUTH_DEG,
            )
            print(f"  saved {len(paths)} PNG snapshots to {snapshots_dir}")

        if SAVE_VIDEO:
            video_path = save_agglomeration_video(
                run_h5,
                plots_dir / "box_agglomeration.mp4",
                backend=VISUALIZATION_BACKEND,
                every=SNAPSHOT_EVERY_SAVED_FRAME,
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
