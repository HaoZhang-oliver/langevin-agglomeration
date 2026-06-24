"""Agglomerate monomers until a pentamer forms, then render snapshots and video.

Edit the variables in the USER INPUTS section to tune the run.
"""

from __future__ import annotations

import os
from pathlib import Path

from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL
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
TARGET_SIZE = 5
PRIMARY_DIAMETER_M = 30.0e-9
PARTICLE_DENSITY_KG_M3 = 1000.0
FRICTION_MODEL = DEFAULT_CLUSTER_FRICTION_MODEL
BOX_SIZE_M = 1000e-9

T_END_S = 3.0e-4
MAX_STEPS = 50_000
MAX_EVENTS = 20
DT_FACTOR = 0.08
DT_MAX_S = 2.0e-7
CAPTURE_TOLERANCE_M = 5.0e-9

RANDOM_SEED = 0
SAVE_EVERY_STEPS = 1
OUTPUT_DIR = Path("outputs/pentamer_agglomeration")

RENDER_FIGURES = env_bool("LDAGG_RENDER", True)
VISUALIZATION_BACKEND = os.getenv("LDAGG_VIS_BACKEND", "pyvista")
SAVE_PNG_SNAPSHOTS = env_bool("LDAGG_SAVE_SNAPSHOTS", True)
SAVE_VIDEO = env_bool("LDAGG_SAVE_VIDEO", True)
SNAPSHOT_EVERY_SAVED_FRAME = 1
MAX_RENDERED_FRAMES = 80
VIDEO_FPS = 8
VIEW_ELEVATION_DEG = 30.0
VIEW_AZIMUTH_DEG = 30.0

# -----------------------------------------------------------------------


def main() -> None:
    config = CoagulationConfig(
        n_particles=N_PARTICLES,
        diameter=PRIMARY_DIAMETER_M,
        particle_density=PARTICLE_DENSITY_KG_M3,
        box_size=BOX_SIZE_M,
        boundary_mode="periodic",
        t_end=T_END_S,
        target_clusters=1,
        target_size=TARGET_SIZE,
        max_events=MAX_EVENTS,
        max_steps=MAX_STEPS,
        dt_factor=DT_FACTOR,
        dt_max=DT_MAX_S,
        capture_tolerance=CAPTURE_TOLERANCE_M,
        gravity_enabled=False,
        brownian=True,
        friction_model=FRICTION_MODEL,
        seed=RANDOM_SEED,
        save_every=SAVE_EVERY_STEPS,
    )

    print("Pentamer agglomeration run")
    print(f"  target largest cluster size: {TARGET_SIZE}")
    print(f"  monomers: {N_PARTICLES}")
    print(f"  primary diameter: {PRIMARY_DIAMETER_M:.3e} m")
    print(f"  cluster friction model: {FRICTION_MODEL}")
    print(f"  box size: {BOX_SIZE_M:.3e} m")
    print(f"  t_end: {T_END_S:.3e} s, dt_max: {DT_MAX_S:.3e} s")
    print(f"  seed: {RANDOM_SEED}")
    print(f"  output: {OUTPUT_DIR}")
    print("Running LD coagulation...")

    result = run_coagulation(config, OUTPUT_DIR, make_plots=False)
    run_h5 = OUTPUT_DIR / "run.h5"
    plots_dir = OUTPUT_DIR / "plots"
    snapshots_dir = OUTPUT_DIR / "snapshots"

    largest = max(cluster.n_primary for cluster in result.clusters)
    print("Run complete")
    print(f"  simulated time: {result.time:.6e} s")
    print(f"  integration steps: {result.steps}")
    print(f"  collision events: {len(result.events)}")
    print(f"  final cluster count: {len(result.clusters)}")
    print(f"  largest cluster size: {largest}")
    print(f"  reached pentamer: {largest >= TARGET_SIZE}")

    if result.events:
        print("Collision diagnostics:")
        for event in result.events:
            print(
                "  "
                f"t={event.time:.3e} s, "
                f"{event.size_a}+{event.size_b}->{event.new_size}, "
                f"clusters {event.cluster_a}/{event.cluster_b} -> {event.new_cluster_id}"
            )
    else:
        print("Collision diagnostics: no collisions recorded")

    if RENDER_FIGURES:
        print("Rendering trajectory and aggregate figures...")
        plot_agglomeration_trajectories(
            run_h5,
            plots_dir / "primary_trajectories.png",
            backend=VISUALIZATION_BACKEND,
            max_frames=MAX_RENDERED_FRAMES,
            box_size=BOX_SIZE_M,
            elev=VIEW_ELEVATION_DEG,
            azim=VIEW_AZIMUTH_DEG,
        )
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
                import matplotlib.pyplot as plt

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
                plots_dir / "agglomeration.mp4",
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

    print(f"Summary CSV: {OUTPUT_DIR / 'cluster_stats.csv'}")
    print(f"Events CSV:  {OUTPUT_DIR / 'events.csv'}")
    print(f"Run HDF5:    {run_h5}")


if __name__ == "__main__":
    main()
