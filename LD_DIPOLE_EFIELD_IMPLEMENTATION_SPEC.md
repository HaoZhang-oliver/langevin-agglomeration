# Implementation Spec: Constant-Field Induced Dipole-Dipole Interaction for `ldagg`

## Purpose

Extend the current `ldagg` Langevin agglomeration code so metal nanoparticles in a constant external DC electric field acquire field-induced dipoles and interact through dipole-dipole forces. The implementation should preserve the existing no-field behavior and remain cheap enough for the current monomer, dimer-monomer, sequential-growth, and small box-coagulation workflows.

The current code already solves translational Langevin Dynamics using the Ermak-Buckholz update with deterministic forces passed into `eb_step`. The new work is therefore mostly a deterministic force-model extension, not a new integrator.

## Current code structure to work with

Important files in the uploaded repository:

- `src/ldagg/integrators.py`
  - Contains `eb_step(...)` and `eb_step_many(...)`.
  - These already accept a deterministic 3-vector `force` for each moving object.
- `src/ldagg/clusters.py`
  - Defines rigid translating `Cluster` objects.
  - A cluster stores primary sphere centers in `rel_positions` and exposes `absolute_centers`.
  - It does not rotate, restructure, sinter, or compute torques.
- `src/ldagg/collisions.py`
  - Detects primary-sphere contact between clusters.
  - Merges clusters irreversibly with `merge_clusters(...)`.
- `src/ldagg/simulation.py`
  - N-cluster agglomeration in a box.
  - `CoagulationConfig` currently includes Brownian motion, gravity, friction model, box size, collision tolerance, and timestep settings.
  - Current deterministic force is only gravity: `force = cluster.mass * gravity` when gravity is enabled.
- `src/ldagg/sequential_growth.py`
  - Sequential growth mode: fixed or moving aggregate plus one launched monomer.
  - `SequentialGrowthConfig` currently mirrors the no-field physics.
- `examples/run_dimer_monomer_to_trimer.py`
  - Minimal debugging script for dimer + monomer.
- `docs/theory.md`
  - Explicitly says electric fields, charge, induced dipoles, van der Waals, etc. are excluded in the current version.

## Physical model for this pass

### Assumptions

Use the simplest field-induced dipole model:

1. Particles are identical metal primary spheres.
2. The external electric field is constant in space and time:

   \[
   \mathbf{E}_0 = (E_x, E_y, E_z)
   \]

3. Particles are neutral unless future work adds net charge. Do **not** add a single-particle `q E` force in this pass.
4. Each primary sphere has a fixed induced dipole aligned with the imposed field:

   \[
   \mathbf{p} = \alpha \mathbf{E}_0
   \]

   where `alpha` is the primary-particle electric polarizability in SI units, `C m^2 / V`.

5. Use pairwise point-dipole interactions between primary spheres in **different clusters**. Do not compute internal forces between primary spheres inside the same rigid cluster.
6. Do not implement local-field self-consistency, multipoles, charges, AC response, torques, rotations, sintering, or van der Waals forces in this pass.

### Dipole-dipole potential

For two primary dipoles `p_i` and `p_j`, define

- `r_vec = r_j - r_i`, the vector from primary `i` to primary `j`.
- `r = ||r_vec||`.
- `r_hat = r_vec / r`.
- `eps0 = 8.8541878128e-12 F/m`.
- `eps_r_medium`, the relative permittivity of the gas or medium.
- `k = 1 / (4*pi*eps0*eps_r_medium)`.

The interaction energy is

\[
U_{dd} = \frac{1}{4\pi\epsilon_0\epsilon_r r^3}
\left[\mathbf{p}_i\cdot\mathbf{p}_j
-3(\mathbf{p}_i\cdot\hat{\mathbf r})(\mathbf{p}_j\cdot\hat{\mathbf r})\right]
\]

### Dipole-dipole force

Implement the general vector force on primary `j` due to primary `i` as

\[
\mathbf{F}_{j\leftarrow i} =
\frac{3}{4\pi\epsilon_0\epsilon_r r^4}
\left[
(\mathbf{p}_i\cdot\hat{\mathbf r})\mathbf{p}_j
+(\mathbf{p}_j\cdot\hat{\mathbf r})\mathbf{p}_i
+(\mathbf{p}_i\cdot\mathbf{p}_j)\hat{\mathbf r}
-5(\mathbf{p}_i\cdot\hat{\mathbf r})(\mathbf{p}_j\cdot\hat{\mathbf r})\hat{\mathbf r}
\right]
\]

where `r_vec` points from `i` to `j`.

Important sign tests:

- If `E0` is along `+z`, primary `i` is at the origin, and primary `j` is at `+z`, then `F_on_j.z < 0`; the pair is head-to-tail attractive.
- If `E0` is along `+z`, primary `i` is at the origin, and primary `j` is at `+x`, then `F_on_j.x > 0`; the pair is side-by-side repulsive.
- Newton's third law must hold exactly at the pair level: `F_on_i = -F_on_j`.

### Regularization and cutoff

The point-dipole model diverges at short range. In normal operation, collision detection should merge clusters before serious overlap occurs, but numerical overshoot can happen. Implement a conservative distance floor:

\[
r_{eff} = \max(r, r_i + r_j + h_{reg})
\]

where `h_reg` is a configurable regularization gap, default `0.0` or a small value such as `0.0e-9` to `1.0e-9 m`.

Use `r_hat` from the actual displacement, but use `r_eff` in the `r^-4` force magnitude.

Add an optional cutoff distance. If `dipole_cutoff` is `None`, compute all cross-cluster primary pairs. If it is a float, skip pairs with `r > dipole_cutoff`.

## Configuration and material data

Use YAML as the primary material format because the code already uses YAML configs.

Add a new nested config section to both `CoagulationConfig` and `SequentialGrowthConfig`:

```yaml
electric_field:
  enabled: true
  vector: [0.0, 0.0, 1.0e6]   # V/m
  medium_relative_permittivity: 1.00058
  material_file: examples/materials/demo_metal_100nm.yml
  polarizability_model: provided  # provided | conducting_sphere
  dipole_cutoff: null             # m; optional
  regularization_gap: 0.0         # m
```

Add a material file example:

```yaml
name: demo_conducting_metal_100nm_primary
polarizability_SI: 1.3908125693e-32  # C m^2 / V, conducting sphere alpha=4*pi*eps0*a^3 for a=50 nm in eps_r=1
notes: Demo value for 100 nm diameter conducting primary sphere. Replace with measured or literature value for real materials.
```

Also support a direct config override:

```yaml
electric_field:
  enabled: true
  vector: [1.0e6, 0.0, 0.0]
  medium_relative_permittivity: 1.0
  polarizability_SI: 1.3908125693e-32
```

If both `polarizability_SI` and `material_file` are present, the direct config value should override the file value and issue no error.

If `polarizability_model: conducting_sphere` and `polarizability_SI` is absent, compute

\[
\alpha = 4\pi\epsilon_0\epsilon_r a^3
\]

where `a = diameter / 2` and `eps_r` is `medium_relative_permittivity`.

Validation rules:

- If `enabled=false`, no polarizability is required and all dipole forces are zero.
- If `enabled=true`, `vector` must be a finite 3-vector.
- If `enabled=true` and `||vector|| == 0`, allow it but return zero forces.
- If `enabled=true`, polarizability must be available either from direct config, material YAML, or `conducting_sphere` calculation.
- Reject nonpositive polarizability.

## Files to add or modify

### 1. Add `src/ldagg/electric.py`

Implement:

```python
EPS0 = 8.8541878128e-12

@dataclass(slots=True)
class MaterialProperties:
    name: str = "unknown"
    polarizability_SI: float | None = None
    density: float | None = None
    notes: str | None = None

@dataclass(slots=True)
class ElectricFieldConfig:
    enabled: bool = False
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    medium_relative_permittivity: float = 1.0
    material_file: str | None = None
    polarizability_SI: float | None = None
    polarizability_model: str = "provided"  # "provided" or "conducting_sphere"
    dipole_cutoff: float | None = None
    regularization_gap: float = 0.0
```

Functions:

```python
ElectricFieldConfig.from_mapping(data: dict | None) -> ElectricFieldConfig
ElectricFieldConfig.to_dict() -> dict
load_material_properties(path: str | Path) -> MaterialProperties
resolve_polarizability(config: ElectricFieldConfig, diameter: float) -> float
induced_dipole(alpha: float, electric_field_vector: np.ndarray) -> np.ndarray
dipole_pair_force_on_b(r_vec, p_a, p_b, eps_r=1.0, min_distance=None) -> np.ndarray
dipole_forces_between_clusters(cluster_a, cluster_b, field_config, primary_diameter, boundary=None) -> tuple[np.ndarray, np.ndarray]
dipole_forces_on_clusters(clusters, field_config, primary_diameter, boundary=None) -> np.ndarray
```

Design notes:

- The `dipole_forces_on_clusters` function returns an array with shape `(len(clusters), 3)` containing total force on each cluster COM.
- Use primary centers (`cluster.absolute_centers`) and primary radii.
- Use the `pair_displacement(...)` helper or `minimum_image(...)` logic for periodic boundaries.
- Do not mutate clusters in the force functions.
- Keep this O(N_cluster^2 * N_primary^2) for now. Current use cases are small.

### 2. Modify `src/ldagg/constants.py`

Add:

```python
EPS0: float = 8.8541878128e-12
```

and export it if useful.

### 3. Modify `src/ldagg/simulation.py`

Add to `CoagulationConfig`:

```python
electric_field: ElectricFieldConfig = field(default_factory=ElectricFieldConfig)
```

Update `from_mapping` and `to_dict`.

Add a helper:

```python
def deterministic_forces(config: CoagulationConfig, clusters: list[Cluster], boundary: Boundary) -> np.ndarray:
    forces = np.zeros((len(clusters), 3))
    if config.gravity_enabled:
        forces += np.array([cluster.mass * gravity for cluster in clusters])
    if config.electric_field.enabled:
        forces += dipole_forces_on_clusters(clusters, config.electric_field, config.diameter, boundary)
    return forces
```

Modify `agglomeration_timestep` so it can receive `forces` and include deterministic drift from all deterministic forces:

```python
max_diffusion = max(k_B*T / cluster.friction)
dt_diff = gap_scale**2 / (6*max_diffusion)
max_drift_speed = max(norm(F_i)/cluster_i.friction)
dt_force = gap_scale/max_drift_speed if max_drift_speed > 0 else inf
dt = dt_factor * min(dt_diff, dt_force, dt_max)
```

Then in `simulate_coagulation(...)`:

1. detect and merge existing contacts,
2. compute deterministic forces at current positions,
3. choose timestep using those forces,
4. pass each cluster's total deterministic force into `eb_step`.

Do not add a separate `qE` force; these are neutral metal particles with induced dipoles.

### 4. Modify `src/ldagg/sequential_growth.py`

Add `electric_field` to `SequentialGrowthConfig`, with parsing and serialization.

Update `growth_timestep(...)` to include deterministic drift forces.

In `simulate_sequential_growth(...)`, compute the two-cluster dipole force between the aggregate and monomer:

```python
forces = dipole_forces_on_clusters([aggregate, monomer], config.electric_field, config.diameter, boundary=None)
force_a = forces[0] + gravity_force_a
force_m = forces[1] + gravity_force_m
```

Apply `force_a` only if `move_seed=True`. Always apply `force_m` to the moving monomer.

Important limitation to document: because the current aggregate does not rotate, dipole-dipole forces can bias translation/attachment but cannot rotate a dimer or larger aggregate into the field direction. That is acceptable for this pass.

### 5. Modify example scripts and configs

Add:

- `examples/materials/demo_metal_100nm.yml`
- `examples/configs/dipole_box_agglomeration.yml`
- `examples/configs/dipole_sequential_growth.yml`
- optionally `examples/run_dimer_monomer_dipole.py`

For the dimer-monomer debug example, choose `E0` along the existing dimer axis if the script initializes the dimer along `x`:

```python
E0 = [1.0e6, 0.0, 0.0]
```

That makes head-to-tail attraction easy to see without implementing rotation.

### 6. Update docs

Update `docs/theory.md` with a short section:

- neutral metal particles in uniform DC field acquire induced dipoles,
- no net `qE` force is applied unless charge physics is later added,
- dipole-dipole force is pairwise, anisotropic, attractive head-to-tail and repulsive side-by-side,
- current model has no torque/rotation/self-consistent polarization.

Update `docs/usage.md` with a command example:

```bash
ldag coagulate --config examples/configs/dipole_box_agglomeration.yml --out outputs/dipole_box
ldag grow --config examples/configs/dipole_sequential_growth.yml --out outputs/dipole_growth
```

## Tests to add

Add `tests/test_electric_dipoles.py`.

Required tests:

1. `test_disabled_field_returns_zero_forces`
   - Build two monomer clusters.
   - `enabled=false` returns all-zero force array.

2. `test_zero_field_vector_returns_zero_forces`
   - `enabled=true`, `vector=[0,0,0]` returns zero forces.

3. `test_head_to_tail_parallel_dipoles_attract`
   - Particle `a` at origin, particle `b` at `[0,0,r]`, field `[0,0,E]`.
   - `dipole_pair_force_on_b(...).z < 0`.

4. `test_side_by_side_parallel_dipoles_repulse`
   - Particle `a` at origin, particle `b` at `[r,0,0]`, field `[0,0,E]`.
   - `dipole_pair_force_on_b(...).x > 0`.

5. `test_newtons_third_law_cluster_forces`
   - Two monomer clusters.
   - Sum of cluster forces is approximately zero.

6. `test_force_decays_as_r_to_minus_four`
   - Compare same geometry at `r` and `2r`.
   - Magnitude ratio should be approximately `16`.

7. `test_material_yaml_loader`
   - Create temp YAML with `polarizability_SI` and check loaded value.

8. `test_conducting_sphere_polarizability_model`
   - For diameter `100 nm`, `eps_r=1`, verify `alpha ~= 4*pi*eps0*(50e-9)^3`.

9. `test_config_parsing_with_electric_field`
   - `CoagulationConfig.from_mapping(...)` and `SequentialGrowthConfig.from_mapping(...)` parse nested `electric_field`.

10. Smoke tests:
   - Existing no-field tests must still pass.
   - A short dipole-enabled sequential growth run must complete without exceptions. Do not require deterministic collision in the smoke test unless the initial geometry ensures it.

## Acceptance criteria

Run these before declaring done:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python examples/run_dimer_monomer_to_trimer.py
PYTHONPATH=src ldag coagulate --config examples/configs/dipole_box_agglomeration.yml --out outputs/dipole_box_smoke
PYTHONPATH=src ldag grow --config examples/configs/dipole_sequential_growth.yml --out outputs/dipole_growth_smoke
```

If the environment lacks CLI entry points, run the Python module equivalent:

```bash
PYTHONPATH=src python -m ldagg coagulate --config examples/configs/dipole_box_agglomeration.yml --out outputs/dipole_box_smoke
PYTHONPATH=src python -m ldagg grow --config examples/configs/dipole_sequential_growth.yml --out outputs/dipole_growth_smoke
```

## Performance notes

- For now, use direct all-pairs primary interactions. The current examples are small.
- Add a `dipole_cutoff` option so later runs can limit cost.
- Do not implement neighbor lists in this pass unless tests become too slow.

## Explicit non-goals for this pass

Do not implement:

- net charge or `qE` drift,
- Coulomb forces,
- image-charge forces,
- van der Waals forces,
- AC fields or complex polarizability,
- self-consistent mutual polarization,
- torques or rotation,
- sintering/restructuring,
- hydrodynamic interactions beyond the existing scalar cluster friction.

