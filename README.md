# ldagg: Langevin aerosol agglomeration

`ldagg` is a small Python scientific-computing package for translational Langevin Dynamics (LD) aerosol simulations. It follows the Suresh and Gopalakrishnan 2021 tutorial update form for Brownian motion, drag, gravity settling, periodic wrapping, and first-passage-style collision checks, then extends the code path to irreversible rigid agglomerate growth. It also supports an optional constant DC electric field that induces identical neutral primary-sphere dipoles and adds pairwise inter-cluster dipole-dipole forces.

## Installation

Create the requested conda environment:

```bash
conda env create -f environment.yml
conda activate langevin-agglomeration
pytest
```

The console command is installed as `ldag` by the editable package entry in `environment.yml`.
`pyvista`, `imageio`, and `imageio-ffmpeg` are included for higher-quality
true-radius 3D particle snapshots and MP4/GIF videos.

## Quickstart

Run the three supplied examples:

```bash
python examples/run_settling.py
python examples/run_box_agglomeration.py
python examples/run_sequential_growth.py
python examples/run_single_particle_settling_box.py
python examples/run_dimer_monomer_to_trimer.py
python examples/run_pentamer_agglomeration.py
```

Or use the CLI:

```bash
ldag settling --config examples/configs/settling_air_300K.yml --out outputs/settling
ldag coagulate --config examples/configs/box_agglomeration.yml --out outputs/coagulation
ldag grow --config examples/configs/sequential_growth.yml --out outputs/growth
ldag coagulate --config examples/configs/dipole_box_agglomeration.yml --out outputs/dipole_box
ldag grow --config examples/configs/dipole_sequential_growth.yml --out outputs/dipole_growth
ldag summarize outputs/growth/run.h5
ldag plot outputs/growth/run.h5 --out outputs/growth/plots
```

Each run writes `config_used.yml`, `run_summary.json`, tabular CSV files, HDF5 trajectory or snapshot data, and plots under the output directory.
When `diagnostics.enabled=true`, coagulation and growth runs also write `diagnostics.csv`,
`final_diagnostics.json`, and HDF5 diagnostic groups. These include electric-field
scalars, induced dipole magnitude, contact dipole energy ratio, dipole-force
Newton residual, drift/Brownian step estimates, pair-loop summaries, and aggregate
morphology metrics.

`examples/run_box_agglomeration.py` randomly places tunable monomers in a 3D box, then lets monomers and clusters agglomerate under Brownian motion, drag, and optional gravity while writing PyVista snapshots and video. `examples/run_single_particle_settling_box.py` is a one-particle Brownian settling visualization in a finite box with reflecting walls and an absorbing floor. `examples/run_dimer_monomer_to_trimer.py` is the minimal agglomeration visualization example: it starts from one dimer and one monomer, runs until true primary-sphere contact, then writes trajectory plots, snapshots, and a video. `examples/run_pentamer_agglomeration.py` is a larger tunable script that stops once the largest aggregate reaches five primary spheres.

## Implemented Model

The package uses SI units throughout. Each particle or rigid aggregate obeys

```text
m dv/dt = -f v + F_ext + F_B(t)
dr/dt = v
```

where `F_ext = m*g` if gravity is enabled and zero otherwise. Spherical primary particles use Cunningham-corrected friction and `D = k_B*T/f`. Clusters are rigid sets of primary spheres; the default cluster friction uses a Cunningham-corrected volume-equivalent sphere diameter, `d_ve = 2*(sum(r_primary^3))^(1/3)`. A `free_draining` model remains available as an explicit option.

When `electric_field.enabled=true`, neutral metal primary spheres acquire a fixed induced dipole `p = alpha E0`. Dipole-dipole forces are computed only between primary spheres belonging to different rigid clusters and are summed onto each cluster COM as deterministic forces. No net charge, `qE` drift, Coulomb force, torque, rotation, self-consistent polarization, or multipole model is included.

Collisions are detected when any primary sphere pair from two clusters reaches `distance <= r_i + r_j + capture_tolerance`. Colliding clusters stick irreversibly, conserve mass and linear momentum, and preserve primary-sphere geometry except for optional small contact projection after timestep overshoot.

Agglomeration `run.h5` files store sparse process snapshots containing primary-sphere centers, radii, primary IDs, and cluster IDs. Visualization helpers in `ldagg.plotting` can render 3D snapshots, primary-particle trajectory traces, PNG frame sequences, and MP4/GIF videos. The default agglomeration examples use the PyVista backend (`LDAGG_VIS_BACKEND=pyvista`) so primary spheres are rendered as actual 3D geometry with physical radii instead of Matplotlib screen-size markers. Set `LDAGG_VIS_BACKEND=matplotlib` to use the lighter fallback.

## Limitations

This version does not include net charge, Coulomb forces, image forces, van der Waals forces, sintering, restructuring, rotation/torques, self-consistent polarization, hydrodynamic interaction perturbations, measured mobility diameter, or dynamic-shape-factor drag. The default volume-equivalent sphere drag is still a scalar approximation. Periodic boundaries with gravity are a modeling convenience, not a literal settling chamber.
