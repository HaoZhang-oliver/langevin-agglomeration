# Theory

## Variables

All quantities use SI units. For object `i`, `r_i` is center-of-mass position, `v_i` is velocity, `m_i` is mass, `f_i` is scalar translational friction, `T_g` is gas temperature, and `k_B` is Boltzmann's constant.

## Langevin Equation

The implemented translational LD equation is

```text
m_i dv_i/dt = -f_i v_i + F_ext,i + F_B,i(t)
dr_i/dt = v_i
```

`F_ext,i = m_i g` for gravity and zero when gravity is disabled. The Brownian force has zero mean and is tied to friction through fluctuation-dissipation.

## Gas Drag

For a spherical primary particle:

```text
m_p = rho_p * (pi/6) * d_p^3
a_p = d_p/2
Kn = lambda_g/a_p
C_c = 1 + Kn*(C1 + C2*exp(-C3/Kn))
f_p = 3*pi*mu_g*d_p/C_c
D_p = k_B*T_g/f_p
```

The defaults match the Suresh and Gopalakrishnan MATLAB settling example: `T_g=300 K`, `p_g=101325 Pa`, `rho_p=1000 kg/m^3`, `mu_g=1.8258e-5 kg/(m s)`, `lambda_g=66.7e-9 m`, `C1=1.257`, `C2=0.4`, `C3=1.1`, and `g=[0,0,-9.81] m/s^2`.

## Ermak-Buckholz Update

For `beta = f/m`, timestep `dt`, and `e = exp(-beta*dt)`:

```text
fac4 = (1 - e)/(1 + e)
sigma_v2 = (k_B*T/m) * (1 - e^2)
sigma_r2 = (k_B*T/m)/beta^2 * (2*beta*dt - 4*fac4)

v_new = v*e + (F/f)*(1 - e) + sqrt(sigma_v2)*N_v
r_new = r + (v_new + v - 2*F/f)*(fac4/beta) + (F/f)*dt + sqrt(sigma_r2)*N_r
```

`N_v` and `N_r` are independent standard normal 3-vectors, matching the zero-covariance tutorial code. If the displacement variance is not positive, the integrator reduces `dt`.

## Aggregates

Agglomerates are rigid clusters of primary spheres. The package stores primary centers relative to the cluster COM. In this version, clusters translate only; they do not rotate or restructure. The default friction uses a Cunningham-corrected volume-equivalent sphere diameter:

```text
m_cluster = sum(m_primary)
d_ve = 2*(sum(r_primary^3))^(1/3)
f_cluster = 3*pi*mu_g*d_ve / C_c(d_ve)
```

The EB update advances the cluster COM using `m_cluster`, `f_cluster`, and `F_ext = m_cluster*g` when gravity is enabled. A `free_draining` option remains available and computes `f_cluster = sum(f_primary)`. Neither option is a true measured mobility diameter or dynamic-shape-factor model.

## Collisions and Sticking

A collision occurs when any primary sphere pair across two clusters satisfies:

```text
distance <= r_i + r_j + capture_tolerance
```

The clusters then merge irreversibly. Linear momentum is conserved:

```text
v_new = (m_a*v_a + m_b*v_b)/(m_a + m_b)
```

The merged aggregate keeps the absolute primary-sphere positions at collision, except that a small overlap can be projected back to contact if `project_to_contact=True`.

## Boundaries

`periodic` boundaries wrap COM positions in a cube and use the minimum-image convention for collision distances. `finite` boundaries reflect clusters off the box walls and can optionally absorb at a floor for settling-style problems.

## Exclusions

Electric fields, charge, induced dipoles, van der Waals forces, sintering, restructuring, rotation, and hydrodynamic interactions are not included in this first implementation.
