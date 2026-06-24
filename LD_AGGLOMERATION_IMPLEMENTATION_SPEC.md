# Implementation spec: Langevin Dynamics agglomeration without electric field

## Objective

Build a Python package that simulates aerosol particle agglomeration using the Langevin Dynamics (LD) methodology described by Suresh & Gopalakrishnan (2021) (in folder summary_Suresh2021), but with **no electric field and no electrostatics** in the first implementation. Include Brownian stochastic forcing, gas drag, and optional gravity. Collisions between primary spheres or rigid aggregates should cause irreversible sticking.

This first implementation should be useful for:

1. reproducing the tutorial-style gravity settling example;
2. simulating Brownian coagulation/agglomeration of monodisperse spherical primary particles in a 3D box;
3. running a cheap sequential-growth mode: start from a seed aggregate and add one monomer at a time.

Do not implement external electric fields, induced dipoles, charges, van der Waals forces, sintering, restructuring, or rotational Brownian dynamics in this pass. Make the architecture easy to extend later.

---

## Physical model

For a particle or rigid cluster `i`, integrate the translational Langevin equation

```text
m_i dv_i/dt = -f_i v_i + F_ext,i + F_B,i(t)
dr_i/dt = v_i
```

where:

- `r_i` is the center-of-mass position.
- `v_i` is the center-of-mass velocity.
- `m_i` is mass.
- `f_i` is scalar friction factor.
- `F_ext,i = m_i g` for gravity, or zero if gravity is disabled.
- `F_B,i(t)` is the Brownian stochastic force.

For the first agglomeration model, ignore hydrodynamic interaction perturbations between particles. Use isolated-particle scalar friction. This corresponds to the dilute or cheap baseline model. Later we may add Corson-style hydrodynamic interactions.

### Primary particle properties

For a spherical primary particle with diameter `d_p`, radius `a_p = d_p/2`, density `rho_p`, gas viscosity `mu_g`, gas mean free path `lambda_g`, and gas temperature `T_g`:

```text
m_p = rho_p * (pi/6) * d_p^3
Kn = lambda_g / a_p
C_c = 1 + Kn * (C1 + C2 * exp(-C3 / Kn))
f_p = 3*pi*mu_g*d_p / C_c
D_p = k_B*T_g / f_p
```

Use defaults matching the tutorial MATLAB settling example unless the user overrides them:

```text
T_g = 300 K
p_g = 101325 Pa
mu_g = 1.8258e-5 kg/(m s)
lambda_g = 66.7e-9 m
C1 = 1.257
C2 = 0.4
C3 = 1.1
g = [0, 0, -9.81] m/s^2
```

### Cluster properties

Represent an agglomerate as a rigid cluster of primary spheres. Store:

```text
cluster_id
primary sphere centers relative to cluster COM, shape (n_primary, 3)
primary radii
cluster COM position
cluster velocity
mass
friction
```

Initial friction model:

```text
f_cluster = sum(f_primary_k)
m_cluster = sum(m_primary_k)
```

This is a free-draining approximation. Also implement an optional `equivalent_sphere` friction model for later comparison:

```text
volume-equivalent radius: a_eq = a_p * n_primary^(1/3)
friction from Cunningham-corrected sphere with diameter 2*a_eq
```

For this version, translate clusters as rigid objects. Do not rotate clusters and do not relax internal coordinates after sticking.

---

## EB time integration

Implement the Ermak-Buckholz first-order update used in the tutorial. For each moving object with scalar mass `m`, scalar friction `f`, deterministic force `F`, velocity `v`, position `r`, timestep `dt`, gas temperature `T`, and `beta = f/m`:

```text
e = exp(-beta*dt)
fac4 = (1 - e)/(1 + e)

sigma_v^2 = (k_B*T/m) * (1 - e^2)
sigma_r^2 = (k_B*T/m) / beta^2 * (2*beta*dt - 4*fac4)

v_new = v*e + (F/f)*(1 - e) + sqrt(sigma_v^2)*N_v
r_new = r + (v_new + v - 2*F/f)*(fac4/beta) + (F/f)*dt + sqrt(sigma_r^2)*N_r
```

`N_v` and `N_r` are independent standard normal 3-vectors. This is the same zero covariance variant used in the tutorial code. If `sigma_r^2 <= 0`, repeatedly reduce `dt` until it is positive.

Add a `brownian=False` option for deterministic tests. With Brownian disabled, omit the random terms but retain drag and deterministic force.

---

## Timestep strategy

Implement adaptive timestep selection. Keep it conservative and testable.

For gravity settling validation, use the tutorial form:

```text
dt = dt_factor * min(z^2*f/(6*k_B*T), z*f/|m*g|)
```

where `z` is distance to the settling plane. Use `dt_factor = 0.01` by default.

For agglomeration, define `gap_min` as the smallest surface-to-surface distance between any two clusters, computed over all primary sphere pairs. Use a lower bound such as `gap_floor = 0.01*a_p` to avoid zero-length timesteps before collision handling. Then choose:

```text
dt_diff  = gap_scale^2 * f_min/(6*k_B*T)
dt_force = gap_scale * f_min/(max(|F_ext|, eps_force))
dt = dt_factor * min(dt_diff, dt_force, dt_max)
```

where `gap_scale = max(gap_min, gap_floor)`. If gravity is disabled, `dt_force = inf`. Also cap `dt` by user-specified `dt_max` and `dt_min`.

Make the code robust: avoid NaNs, avoid negative random displacement variance, and stop with a clear error if `dt` underflows.

---

## Boundaries

Implement two boundary modes:

1. `periodic`: 3D periodic cube, using minimum-image convention for distance calculations. Best for Brownian coagulation without gravity or with weak gravity.
2. `finite`: finite box. Use reflecting boundaries by default, with optional absorbing floor for settling validation.

For periodic boundary, wrap COM positions after every step.

For collision checks under periodic boundaries, use minimum-image displacement between primary sphere centers.

---

## Collision and sticking

Collision between two clusters occurs when any pair of primary spheres, one from each cluster, satisfies:

```text
center_distance <= radius_i + radius_j + capture_tolerance
```

Default `capture_tolerance = 0.0`.

When a collision occurs:

1. Merge the two clusters into one rigid aggregate.
2. Preserve the absolute positions of all primary sphere centers at the instant of collision. If slight overlap exists due to timestep overshoot, optionally project the newly colliding pair to contact along their centerline; make this behavior configurable, default `project_to_contact=True`.
3. Set new cluster velocity by mass-weighted momentum average:

```text
v_new = (m_a*v_a + m_b*v_b)/(m_a + m_b)
```

4. Recompute COM, relative primary centers, mass, friction, and metadata.
5. Record a collision event with time, cluster ids, sizes before collision, collision pair, collision point, and new cluster size.

Do not model bounce or non-sticking collisions.

---

## Simulation modes

### 1. Settling validation mode

Replicate the paper's MATLAB example in Python:

```text
particle diameters: 10 nm, 100 nm, 1000 nm
rho_p = 1000 kg/m^3
T_g = 300 K
p_g = 101325 Pa
height H = 0.1 m
trials = configurable, default smaller in tests and 1000 in examples
```

Initialize position `[0, 0, H]` and velocity sampled from Maxwell-Boltzmann:

```text
v0_component ~ Normal(0, sqrt(k_B*T_g/m_p))
```

Integrate until `z <= a_p`. Save settling time distribution, mean, standard deviation, and a sample trajectory.

### 2. N-particle agglomeration mode

Initialize `N` monomers randomly in a cubic box with no overlaps. Evolve all clusters under LD. When clusters collide, merge them. Stop when one of these occurs:

```text
time >= t_end
number of clusters <= target_clusters
largest cluster size >= target_size
number of collision events >= max_events
```

Record cluster-size distribution vs time and final aggregate geometry.

### 3. Sequential monomer growth mode

This is the cheap morphology-growth mode for later electric-field work:

```text
seed: monomer, dimer, or user-specified aggregate
growth: inject one monomer at a launch boundary, simulate until it attaches or escapes, then freeze it into the aggregate and repeat
```

For no electric field, start with uniform random injection on a spherical shell around the seed aggregate:

```text
R_launch = R_aggregate + launch_gap
R_kill   = R_launch + kill_gap
```

A trial ends when the monomer collides with the aggregate, escapes beyond `R_kill`, or exceeds `max_steps_per_trial`. If it escapes, restart a new monomer without changing the aggregate. If it collides, attach it and proceed to the next target size.

In this mode, it is acceptable to keep the aggregate fixed and move only the monomer for the first implementation. Include an option `move_seed=True` later, but default `False` for speed.

---

## Package structure

Create a clean Python project:

```text
langevin-agglomeration/
  environment.yml
  pyproject.toml
  README.md
  docs/
    theory.md
    usage.md
  src/ldagg/
    __init__.py
    constants.py
    gas.py
    particles.py
    clusters.py
    integrators.py
    boundaries.py
    collisions.py
    simulation.py
    sequential_growth.py
    settling.py
    analysis.py
    plotting.py
    cli.py
  examples/
    run_settling.py
    run_box_agglomeration.py
    run_sequential_growth.py
    configs/
      settling_air_300K.yml
      box_agglomeration.yml
      sequential_growth.yml
  tests/
    test_gas_friction.py
    test_eb_integrator.py
    test_settling.py
    test_collisions.py
    test_agglomeration_smoke.py
```

Use `numpy`, `scipy`, `numba`, `matplotlib`, `pandas`, `h5py`, `pyyaml`, `typer`, `rich`, and `pytest`. Use Numba only where it materially improves pair-distance or collision loops. Keep pure NumPy fallback behavior clear.

---

## CLI requirements

Implement a CLI executable `ldag` using Typer:

```bash
ldag settling --config examples/configs/settling_air_300K.yml --out outputs/settling
ldag coagulate --config examples/configs/box_agglomeration.yml --out outputs/coagulation
ldag grow --config examples/configs/sequential_growth.yml --out outputs/growth
ldag summarize outputs/growth/run.h5
ldag plot outputs/growth/run.h5 --out outputs/growth/plots
```

All simulations must accept a random seed and produce reproducible results.

---

## Output files

For each run, write:

```text
config_used.yml
run_summary.json
trajectory_sample.h5 or run.h5
events.csv
cluster_stats.csv
final_aggregate.csv
plots/*.png
```

For HDF5 trajectory data, do not save every timestep by default. Save every `save_every` steps or save only event snapshots.

---

## Validation and tests

Implement tests that can run quickly on a laptop.

### Unit tests

1. Friction factor returns the MATLAB-equivalent value for known diameter and gas parameters.
2. EB integrator with `F=0` and Brownian enabled gives approximately correct velocity variance `k_B*T/m` after many samples.
3. EB integrator with gravity and Brownian disabled approaches terminal velocity `v_t = m*g/f`.
4. Position MSD for force-free Brownian motion is approximately `6Dt` at long time for an ensemble.
5. Collision detection catches contact between two equal spheres at distance `2a` and does not catch non-contact at `2a + gap`.
6. Merging conserves mass and momentum and produces non-overlapping relative geometry where possible.

### Example validation

1. `examples/run_settling.py` should reproduce the qualitative paper result: 10 nm particles wander strongly; 1000 nm particles settle more deterministically.
2. `examples/run_box_agglomeration.py` should show decreasing cluster count over time for Brownian particles.
3. `examples/run_sequential_growth.py` should grow from a dimer to a user-defined size and output the final aggregate geometry.

---

## Engineering constraints

- Do not use a black-box molecular dynamics engine. Implement the LD update directly.
- Keep units explicit and SI-based.
- Use dataclasses or Pydantic models for configuration.
- Use vectorized arrays for positions, velocities, masses, and frictions where possible.
- Keep code readable; numerical correctness matters more than cleverness.
- Include docstrings with equations and variable definitions.
- Write a README with installation, conda environment creation, quickstart, and model limitations.
- Run `pytest` and at least one short example before declaring completion.

---

## Known limitations to document

Document these clearly in `README.md` and `docs/theory.md`:

- No electric field, charge, dipoles, van der Waals, sintering, restructuring, rotation, or hydrodynamic interaction perturbations in the first version.
- Free-draining cluster friction is a cheap approximation.
- Gravity has little effect on relative motion for identical monomers if cluster friction is also free-draining; use `equivalent_sphere` friction or polydispersity to see gravitational differential settling.
- Periodic boundaries with gravity are a modeling convenience, not a literal finite-settling chamber.
- LD assumes particles are sufficiently massive compared with gas molecules and in thermal equilibrium with the gas.

