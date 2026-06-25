# Usage

## Environment

```bash
conda env create -f environment.yml
conda activate langevin-agglomeration
pytest
```

If developing without installing, commands can also run with `PYTHONPATH=src`.

## Settling

```bash
ldag settling --config examples/configs/settling_air_300K.yml --out outputs/settling
```

This reproduces the tutorial-style gravity settling setup for 10, 100, and 1000 nm particles. The example config uses a small trial count so it runs quickly; increase `trials` for smoother histograms.

## Box Agglomeration

```bash
ldag coagulate --config examples/configs/box_agglomeration.yml --out outputs/coagulation
```

This initializes non-overlapping monomers in a periodic cube, integrates Brownian LD motion, and sticks clusters on contact.

For an editable visualization script, run:

```bash
python examples/run_box_agglomeration.py
```

The script variables at the top control the number of monomers, box size, finite or periodic boundaries, Brownian motion, gravity, stopping criteria, and rendering outputs. It randomizes monomer positions, then allows monomer-monomer, monomer-cluster, and cluster-cluster sticking. After every collision, the merged aggregate continues as one rigid cluster using its COM, total mass, preserved primary-sphere geometry, and volume-equivalent sphere drag.

Important config fields:

- `n_particles`: number of initial monomers.
- `box_size`: periodic cube side length in meters.
- `diameter`: primary diameter in meters.
- `t_end`, `target_clusters`, `target_size`, `max_events`: stopping criteria.
- `capture_tolerance`: optional contact tolerance in meters.
- `friction_model`: `equivalent_sphere` by default, or `free_draining` for summed primary-sphere friction.

For a self-contained particle-level visualization workflow, run:

```bash
python examples/run_box_agglomeration.py
python examples/run_single_particle_settling_box.py
python examples/run_dimer_monomer_to_trimer.py
python examples/run_pentamer_agglomeration.py
```

`run_single_particle_settling_box.py` tracks one primary particle in a finite box with Brownian motion, drag, gravity, reflecting side/top walls, and an absorbing floor. It prints settling diagnostics and writes `primary_trajectories.png`, `final_snapshot.png`, `settling_z_velocity.png`, PNG frames, and an MP4/GIF video.

`run_dimer_monomer_to_trimer.py` is the minimal debugging example. It starts from a dimer and one monomer, runs until true contact makes a trimer, prints gap and timestep diagnostics, saves plots, writes PNG snapshots, and creates an MP4 or GIF animation.

The tunable inputs are at the top of each script. The pentamer script stops when the largest aggregate reaches five primary spheres.

The script renders by default. Set `LDAGG_RENDER=0` to run only the simulation and numeric outputs, or set `LDAGG_SAVE_SNAPSHOTS=0` / `LDAGG_SAVE_VIDEO=0` to disable only those render products. When invoking the Windows conda Python from WSL, use `cmd.exe` so Windows environment variables are visible, for example:

```bash
cmd.exe /c "cd /d D:\Research\Mars\langevin-agglomeration && set LDAGG_RENDER=0&& C:\Users\hzhang29\miniconda3\envs\langevin-agglomeration\python.exe examples\run_pentamer_agglomeration.py"
```

## Sequential Growth

```bash
ldag grow --config examples/configs/sequential_growth.yml --out outputs/growth
```

This starts from a monomer or dimer, launches one monomer near the aggregate, and repeats trials until monomers stick or escape. The default keeps the seed aggregate fixed for speed.

## Constant DC Electric Field

Neutral metal primary spheres can be given fixed induced dipoles in a uniform DC electric field. Example runs:

```bash
ldag coagulate --config examples/configs/dipole_box_agglomeration.yml --out outputs/dipole_box
ldag grow --config examples/configs/dipole_sequential_growth.yml --out outputs/dipole_growth
```

The config section is:

```yaml
electric_field:
  enabled: true
  vector: [0.0, 0.0, 1.0e6]
  medium_relative_permittivity: 1.00058
  material_file: examples/materials/demo_metal_100nm.yml
  polarizability_model: provided
  dipole_cutoff: null
  regularization_gap: 0.0
```

You can also pass `polarizability_SI` directly in the config; that direct value overrides the material file. For a conducting sphere estimate, use `polarizability_model: conducting_sphere` without `polarizability_SI`. The field model adds only induced dipole-dipole forces between different clusters. It does not add net charge, `qE` drift, Coulomb forces, torques, or aggregate rotation.

## Diagnostics

Coagulation and sequential-growth configs can request extra electric-field,
force, transport, collision, and morphology diagnostics:

```yaml
diagnostics:
  enabled: true
  every: 1
  store_snapshot_forces: true
  store_event_metrics: true
  store_pair_summary: true
```

When enabled, `diagnostics.csv` records fields such as `E_norm`, `alpha`,
`p_norm`, `Gamma_dd_contact`, `dipole_energy`, `max_F_dipole`,
`newton_residual_dipole`, `min_surface_gap`, drift/Brownian step estimates,
and morphology metrics for the largest aggregate. `event_metrics.csv` is
written when collisions occur and stores trigger geometry plus post-merge
morphology. The same tables are also stored in `run.h5` groups.

## Output Files

Simulation outputs include:

- `config_used.yml`
- `run_summary.json`
- `run.h5` or `trajectory_sample.h5`
- `events.csv`
- `diagnostics.csv` when diagnostics are enabled
- `event_metrics.csv` when diagnostics are enabled and collisions occur
- `cluster_stats.csv`
- `final_aggregate.csv`
- `final_diagnostics.json` and `final_diagnostics.csv`
- `plots/*.png`

Use:

```bash
ldag summarize outputs/growth/run.h5
ldag plot outputs/growth/run.h5 --out outputs/growth/plots
```

## Visualization Helpers

The main functions are in `ldagg.plotting`:

- `plot_agglomeration_snapshot(snapshot, out_path, box_size=..., backend="pyvista")`
- `plot_agglomeration_trajectories(run_h5, out_path, box_size=...)`
- `save_agglomeration_snapshots(run_h5, out_dir, max_frames=..., backend="pyvista")`
- `save_agglomeration_video(run_h5, out_path, fps=..., backend="pyvista")`

Use the PyVista backend for agglomeration figures when physical particle size matters. It renders each primary as true 3D sphere geometry using the stored SI-unit radius, which avoids the misleading screen-size markers used by plain 3D scatter plots. `max_frames` samples the whole saved trajectory, including the beginning and end, instead of only taking the first frames.

`save_agglomeration_video` writes MP4 when ffmpeg is available. If ffmpeg is missing and the Matplotlib backend is used with a requested `.mp4`, it writes a GIF fallback beside the requested file.

PyVista is the default in-package backend because it is a scriptable VTK wrapper that fits NumPy-style workflows and can render offscreen snapshots/videos from Python. OVITO is also a strong community tool for particle and molecular dynamics visualization, especially for interactive inspection of larger exported trajectories, but it is better used as an external viewer rather than the package's built-in rendering dependency.
