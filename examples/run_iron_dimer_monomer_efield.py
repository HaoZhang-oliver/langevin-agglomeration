"""Iron dimer + monomer agglomeration in a constant DC electric field.

Edit the variables in the USER INPUTS section to tune the run. The material
properties are defined directly in this script rather than loaded from YAML.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

from ldagg.boundaries import Boundary, apply_boundary_to_cluster
from ldagg.clusters import DEFAULT_CLUSTER_FRICTION_MODEL, Cluster, dimer_seed
from ldagg.collisions import find_collision, merge_clusters, nearest_surface_gap
from ldagg.constants import BOLTZMANN, EPS0
from ldagg.electric import ElectricFieldConfig, dipole_forces_on_clusters
from ldagg.gas import Gas
from ldagg.integrators import eb_step
from ldagg.plotting import (
    plot_agglomeration_snapshot,
    plot_agglomeration_trajectories,
    save_agglomeration_snapshots,
    save_agglomeration_video,
)
from ldagg.simulation import CoagulationResult, record_stats, snapshot, write_coagulation_outputs


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ----------------------------- USER INPUTS -----------------------------

PRIMARY_DIAMETER_M = 30.0e-9
BOX_SIZE_M = 5.0e-7
INITIAL_GAP_M = 20.0e-9

# Iron properties. Density is used for mass; polarizability uses the neutral
# metal/conducting-sphere approximation for the primary sphere in air.
IRON_DENSITY_KG_M3 = 7874.0
MEDIUM_RELATIVE_PERMITTIVITY = 1.00058
IRON_PRIMARY_POLARIZABILITY_SI = (
    4.0 * np.pi * EPS0 * MEDIUM_RELATIVE_PERMITTIVITY * (0.5 * PRIMARY_DIAMETER_M) ** 3
)

# 30 kV/cm = 3.0e6 V/m, directed along +z.
ELECTRIC_FIELD_STRENGTH_V_M = 30.0e3 * 100.0
ELECTRIC_FIELD_VECTOR_V_M = (0.0, 0.0, ELECTRIC_FIELD_STRENGTH_V_M)

FRICTION_MODEL = DEFAULT_CLUSTER_FRICTION_MODEL
MOVE_DIMER = True
BROWNIAN = True
RANDOM_SEED = 20260624

MAX_TIME_S = 5.0e-4
MAX_STEPS = 20_000
DT_FACTOR = 0.2
DT_MIN_S = 1.0e-12
DT_MAX_S = 5.0e-9
GAP_FLOOR_FRACTION = 0.01
CAPTURE_TOLERANCE_M = 0.0

SAVE_EVERY_STEPS = 5
DIAGNOSTIC_EVERY_STEPS = 100
OUTPUT_DIR = Path("outputs/iron_dimer_monomer_efield")

RENDER_FIGURES = env_bool("LDAGG_RENDER", True)
VISUALIZATION_BACKEND = os.getenv("LDAGG_VIS_BACKEND", "pyvista")
SAVE_PNG_SNAPSHOTS = env_bool("LDAGG_SAVE_SNAPSHOTS", True)
SAVE_VIDEO = env_bool("LDAGG_SAVE_VIDEO", True)
MAX_RENDERED_FRAMES = 120
VIDEO_FPS = 10
VIEW_ELEVATION_DEG = 25.0
VIEW_AZIMUTH_DEG = 35.0

# -----------------------------------------------------------------------


def make_z_aligned_dimer(gas: Gas, center: np.ndarray) -> Cluster:
    dimer = dimer_seed(
        0,
        PRIMARY_DIAMETER_M,
        density=IRON_DENSITY_KG_M3,
        gas=gas,
        friction_model=FRICTION_MODEL,
    )
    dimer.rel_positions = np.array(
        [
            [0.0, 0.0, -0.5 * PRIMARY_DIAMETER_M],
            [0.0, 0.0, 0.5 * PRIMARY_DIAMETER_M],
        ],
        dtype=float,
    )
    dimer.position = np.asarray(center, dtype=float)
    dimer.velocity = np.zeros(3)
    dimer.primary_ids = np.array([0, 1], dtype=np.int64)
    dimer.recenter()
    return dimer


def make_initial_clusters(gas: Gas, rng: np.random.Generator) -> list[Cluster]:
    dimer_center = np.array([0.5 * BOX_SIZE_M, 0.5 * BOX_SIZE_M, 0.5 * BOX_SIZE_M])
    dimer = make_z_aligned_dimer(gas, dimer_center)

    monomer_z = dimer_center[2] + 1.5 * PRIMARY_DIAMETER_M + INITIAL_GAP_M
    monomer_position = np.array([dimer_center[0], dimer_center[1], monomer_z], dtype=float)
    if monomer_position[2] + 0.5 * PRIMARY_DIAMETER_M > BOX_SIZE_M:
        raise ValueError("BOX_SIZE_M is too small for the requested INITIAL_GAP_M")

    velocity = np.zeros(3)
    if BROWNIAN:
        probe = Cluster.monomer(
            2,
            monomer_position,
            np.zeros(3),
            PRIMARY_DIAMETER_M,
            density=IRON_DENSITY_KG_M3,
            gas=gas,
            friction_model=FRICTION_MODEL,
        )
        thermal_speed = np.sqrt(BOLTZMANN * gas.temperature / probe.mass)
        velocity = thermal_speed * rng.normal(size=3)

    monomer = Cluster.monomer(
        2,
        monomer_position,
        velocity,
        PRIMARY_DIAMETER_M,
        density=IRON_DENSITY_KG_M3,
        gas=gas,
        friction_model=FRICTION_MODEL,
    )
    monomer.primary_ids = np.array([2], dtype=np.int64)
    return [dimer, monomer]


def choose_timestep(gas: Gas, clusters: list[Cluster], boundary: Boundary, forces: np.ndarray) -> float:
    gap = nearest_surface_gap(clusters, boundary=boundary)
    min_radius = min(float(np.min(cluster.radii)) for cluster in clusters)
    gap_scale = max(gap, GAP_FLOOR_FRACTION * min_radius)
    max_diffusion = max(BOLTZMANN * gas.temperature / cluster.friction for cluster in clusters)
    dt_diff = gap_scale * gap_scale / (6.0 * max_diffusion)
    drift_speeds = [
        np.linalg.norm(force) / cluster.friction
        for force, cluster in zip(np.asarray(forces, dtype=float), clusters, strict=True)
    ]
    max_drift_speed = max(drift_speeds, default=0.0)
    dt_force = gap_scale / max_drift_speed if max_drift_speed > 0.0 else np.inf
    dt = DT_FACTOR * min(dt_diff, dt_force, DT_MAX_S)
    if not np.isfinite(dt):
        dt = DT_MAX_S
    return float(np.clip(dt, DT_MIN_S, DT_MAX_S))


def main() -> None:
    gas = Gas()
    rng = np.random.default_rng(RANDOM_SEED)
    boundary = Boundary("finite", BOX_SIZE_M)
    clusters = make_initial_clusters(gas, rng)
    electric_field = ElectricFieldConfig(
        enabled=True,
        vector=ELECTRIC_FIELD_VECTOR_V_M,
        medium_relative_permittivity=MEDIUM_RELATIVE_PERMITTIVITY,
        polarizability_SI=IRON_PRIMARY_POLARIZABILITY_SI,
        regularization_gap=0.0,
    )

    events = []
    stats = [record_stats(0.0, 0, clusters, 0)]
    snapshots = [snapshot(0.0, clusters)]
    time = 0.0
    step = 0

    initial_forces = dipole_forces_on_clusters(
        clusters,
        electric_field,
        PRIMARY_DIAMETER_M,
        boundary=boundary,
    )
    initial_gap = nearest_surface_gap(clusters, boundary=boundary)

    print("Iron dimer + monomer LD run with constant DC electric field")
    print(f"  primary diameter: {PRIMARY_DIAMETER_M:.3e} m")
    print(f"  iron density: {IRON_DENSITY_KG_M3:.1f} kg/m^3")
    print(f"  alpha: {IRON_PRIMARY_POLARIZABILITY_SI:.6e} C m^2/V")
    print(f"  E0: {ELECTRIC_FIELD_VECTOR_V_M} V/m = 30 kV/cm along +z")
    print(f"  initial surface gap: {initial_gap:.3e} m")
    print(f"  initial dimer force: {initial_forces[0]} N")
    print(f"  initial monomer force: {initial_forces[1]} N")
    print(f"  Brownian enabled: {BROWNIAN}")
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
                boundary=boundary,
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

        forces = dipole_forces_on_clusters(
            clusters,
            electric_field,
            PRIMARY_DIAMETER_M,
            boundary=boundary,
        )
        if not MOVE_DIMER:
            forces[0] = 0.0
        dt = min(choose_timestep(gas, clusters, boundary, forces), MAX_TIME_S - time)

        for cluster, force in zip(clusters, forces, strict=True):
            if cluster.cluster_id == 0 and not MOVE_DIMER:
                continue
            result = eb_step(
                cluster.position,
                cluster.velocity,
                cluster.mass,
                cluster.friction,
                force,
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
            gap = nearest_surface_gap(clusters, boundary=boundary)
            force_norms = [float(np.linalg.norm(force)) for force in forces]
            print(
                f"  step={step:6d}, t={time:.3e} s, "
                f"gap={gap:.3e} m, dt={dt:.3e} s, |F|={force_norms}"
            )
    else:
        print(f"Reached MAX_STEPS={MAX_STEPS} before contact")

    largest = max(cluster.n_primary for cluster in clusters)
    config_used = {
        "mode": "iron_dimer_monomer_efield",
        "gas": gas.to_dict(),
        "diameter": PRIMARY_DIAMETER_M,
        "particle_density": IRON_DENSITY_KG_M3,
        "box_size": BOX_SIZE_M,
        "boundary_mode": boundary.mode,
        "initial_gap": INITIAL_GAP_M,
        "brownian": BROWNIAN,
        "move_dimer": MOVE_DIMER,
        "friction_model": FRICTION_MODEL,
        "electric_field": electric_field.to_dict(),
        "material": {
            "name": "iron",
            "density_kg_m3": IRON_DENSITY_KG_M3,
            "polarizability_model": "conducting_sphere",
            "polarizability_SI": IRON_PRIMARY_POLARIZABILITY_SI,
        },
        "seed": RANDOM_SEED,
    }
    result = CoagulationResult(
        clusters=clusters,
        events=events,
        stats=stats,
        snapshots=snapshots,
        config=config_used,
        time=time,
        steps=step,
    )
    write_coagulation_outputs(result, OUTPUT_DIR, make_plots=False)

    run_h5 = OUTPUT_DIR / "run.h5"
    plots_dir = OUTPUT_DIR / "plots"
    snapshots_dir = OUTPUT_DIR / "snapshots"

    summary = {
        "reached_trimer": bool(largest >= 3),
        "largest_cluster": largest,
        "events": len(events),
        "time": time,
        "steps": step,
        "electric_field_V_m": ELECTRIC_FIELD_VECTOR_V_M,
        "electric_field_kV_cm": 30.0,
        "iron_polarizability_SI": IRON_PRIMARY_POLARIZABILITY_SI,
    }
    with (OUTPUT_DIR / "iron_efield_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Run complete")
    print(f"  simulated time: {time:.6e} s")
    print(f"  integration steps: {step}")
    print(f"  collision events: {len(events)}")
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
            title=f"Iron + E field, largest cluster = {largest}",
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
                plots_dir / "iron_dimer_monomer_efield.mp4",
                backend=VISUALIZATION_BACKEND,
                max_frames=MAX_RENDERED_FRAMES,
                fps=VIDEO_FPS,
                box_size=BOX_SIZE_M,
                elev=VIEW_ELEVATION_DEG,
                azim=VIEW_AZIMUTH_DEG,
            )
            print(f"  saved video to {video_path}")
    else:
        print("Rendering disabled. Set LDAGG_RENDER=1 or RENDER_FIGURES=True to save plots/video.")

    print(f"Summary JSON: {OUTPUT_DIR / 'iron_efield_summary.json'}")
    print(f"Run HDF5:     {run_h5}")


if __name__ == "__main__":
    main()
