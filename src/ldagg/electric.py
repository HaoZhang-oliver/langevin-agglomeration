"""Uniform-field induced dipole interactions."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ldagg.boundaries import Boundary
from ldagg.collisions import pair_displacement
from ldagg.constants import EPS0


@dataclass(slots=True)
class MaterialProperties:
    name: str = "unknown"
    polarizability_SI: float | None = None
    density: float | None = None
    notes: str | None = None


@dataclass(slots=True)
class ClusterDipoleResult:
    primary_dipoles: np.ndarray
    total_dipole: np.ndarray
    condition_number: float
    n_primary: int


@dataclass(slots=True)
class ElectricFieldConfig:
    enabled: bool = False
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    medium_relative_permittivity: float = 1.0
    material_file: str | None = None
    polarizability_SI: float | None = None
    polarizability_model: str = "provided"
    dipole_cutoff: float | None = None
    regularization_gap: float = 0.0
    dipole_force_model: str = "primary_pair_fixed"
    coupled_internal_regularization_gap: float | None = None
    coupled_condition_warning: float = 1.0e12

    @classmethod
    def from_mapping(cls, data: dict | None) -> ElectricFieldConfig:
        if isinstance(data, cls):
            return data
        values = dict(data or {})
        if "vector" in values:
            vector = np.asarray(values["vector"], dtype=float)
            if vector.shape != (3,):
                raise ValueError("electric_field.vector must be a finite 3-vector")
            values["vector"] = tuple(float(value) for value in vector)
        for key in (
            "medium_relative_permittivity",
            "polarizability_SI",
            "dipole_cutoff",
            "regularization_gap",
            "coupled_internal_regularization_gap",
            "coupled_condition_warning",
        ):
            if values.get(key) is not None:
                values[key] = float(values[key])
        config = cls(**values)
        config.validate()
        return config

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        vector = np.asarray(self.vector, dtype=float)
        if vector.shape != (3,):
            raise ValueError("electric_field.vector must be a finite 3-vector")
        if self.enabled and not np.all(np.isfinite(vector)):
            raise ValueError("electric_field.vector must be finite when enabled")
        if self.medium_relative_permittivity <= 0.0 or not np.isfinite(
            self.medium_relative_permittivity
        ):
            raise ValueError("medium_relative_permittivity must be positive")
        if self.polarizability_model not in {"provided", "conducting_sphere"}:
            raise ValueError("polarizability_model must be 'provided' or 'conducting_sphere'")
        if self.polarizability_SI is not None and (
            self.polarizability_SI <= 0.0 or not np.isfinite(self.polarizability_SI)
        ):
            raise ValueError("polarizability_SI must be positive when provided")
        if self.dipole_cutoff is not None and (
            self.dipole_cutoff <= 0.0 or not np.isfinite(self.dipole_cutoff)
        ):
            raise ValueError("dipole_cutoff must be positive when provided")
        if self.regularization_gap < 0.0 or not np.isfinite(self.regularization_gap):
            raise ValueError("regularization_gap must be nonnegative")
        if self.dipole_force_model not in {"primary_pair_fixed", "cluster_coupled_dipole"}:
            raise ValueError(
                "dipole_force_model must be 'primary_pair_fixed' or "
                "'cluster_coupled_dipole'"
            )
        if self.coupled_internal_regularization_gap is not None and (
            self.coupled_internal_regularization_gap < 0.0
            or not np.isfinite(self.coupled_internal_regularization_gap)
        ):
            raise ValueError("coupled_internal_regularization_gap must be nonnegative")
        if self.coupled_condition_warning <= 0.0 or not np.isfinite(
            self.coupled_condition_warning
        ):
            raise ValueError("coupled_condition_warning must be positive")


def load_material_properties(path: str | Path) -> MaterialProperties:
    """Load material polarizability metadata from a YAML file."""

    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    polarizability = data.get("polarizability_SI")
    density = data.get("density")
    return MaterialProperties(
        name=str(data.get("name", "unknown")),
        polarizability_SI=None if polarizability is None else float(polarizability),
        density=None if density is None else float(density),
        notes=data.get("notes"),
    )


def _field_vector(config: ElectricFieldConfig) -> np.ndarray:
    return np.asarray(config.vector, dtype=float)


def resolve_polarizability(config: ElectricFieldConfig, diameter: float) -> float:
    """Resolve the primary-sphere polarizability in SI units."""

    config.validate()
    field = _field_vector(config)
    if not config.enabled or np.linalg.norm(field) == 0.0:
        return 0.0
    if config.polarizability_SI is not None:
        return float(config.polarizability_SI)
    if config.polarizability_model == "conducting_sphere":
        radius = 0.5 * float(diameter)
        if radius <= 0.0:
            raise ValueError("diameter must be positive for conducting_sphere polarizability")
        return float(4.0 * np.pi * EPS0 * config.medium_relative_permittivity * radius**3)
    if config.material_file is not None:
        material = load_material_properties(config.material_file)
        if material.polarizability_SI is None:
            raise ValueError(f"material file {config.material_file!r} lacks polarizability_SI")
        if material.polarizability_SI <= 0.0 or not np.isfinite(material.polarizability_SI):
            raise ValueError("material polarizability_SI must be positive")
        return float(material.polarizability_SI)
    raise ValueError(
        "enabled electric_field requires polarizability_SI, material_file, "
        "or polarizability_model='conducting_sphere'"
    )


def induced_dipole(alpha: float, electric_field_vector: np.ndarray) -> np.ndarray:
    """Return the induced primary-sphere dipole vector ``p = alpha E0``."""

    alpha = float(alpha)
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be nonnegative")
    field = np.asarray(electric_field_vector, dtype=float)
    if field.shape != (3,) or not np.all(np.isfinite(field)):
        raise ValueError("electric_field_vector must be a finite 3-vector")
    return alpha * field


def dipole_field_tensor(
    r_vec,
    eps_r: float = 1.0,
    min_distance: float | None = None,
) -> np.ndarray:
    """Return the dipole field tensor mapping a source dipole to its field.

    ``r_vec`` points from the source dipole to the field-evaluation point.
    The optional distance floor regularizes only the radial magnitude. A zero
    displacement returns a zero tensor because no direction is defined.
    """

    if eps_r <= 0.0 or not np.isfinite(eps_r):
        raise ValueError("eps_r must be positive")
    r_vec = np.asarray(r_vec, dtype=float)
    if r_vec.shape != (3,) or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    distance = float(np.linalg.norm(r_vec))
    if distance == 0.0:
        return np.zeros((3, 3), dtype=float)
    r_hat = r_vec / distance
    if min_distance is None:
        r_eff = distance
    else:
        r_eff = max(distance, float(min_distance))
    if r_eff <= 0.0 or not np.isfinite(r_eff):
        raise ValueError("effective dipole distance must be positive")
    return (3.0 * np.outer(r_hat, r_hat) - np.eye(3)) / (
        4.0 * np.pi * EPS0 * eps_r * r_eff**3
    )


def dipole_pair_force_on_b(
    r_vec,
    p_a,
    p_b,
    eps_r: float = 1.0,
    min_distance: float | None = None,
) -> np.ndarray:
    """Force on dipole ``b`` from dipole ``a``.

    ``r_vec`` points from ``a`` to ``b``. The direction uses the actual
    displacement, while the optional distance floor only regularizes the
    ``r^-4`` magnitude.
    """

    if eps_r <= 0.0 or not np.isfinite(eps_r):
        raise ValueError("eps_r must be positive")
    r_vec = np.asarray(r_vec, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if r_vec.shape != (3,) or p_a.shape != (3,) or p_b.shape != (3,):
        raise ValueError("r_vec, p_a, and p_b must be 3-vectors")
    distance = float(np.linalg.norm(r_vec))
    if distance == 0.0:
        return np.zeros(3)
    r_hat = r_vec / distance
    if min_distance is None:
        r_eff = distance
    else:
        r_eff = max(distance, float(min_distance))
    if r_eff <= 0.0 or not np.isfinite(r_eff):
        raise ValueError("effective dipole distance must be positive")

    pa_r = float(np.dot(p_a, r_hat))
    pb_r = float(np.dot(p_b, r_hat))
    pa_pb = float(np.dot(p_a, p_b))
    bracket = pa_r * p_b + pb_r * p_a + pa_pb * r_hat - 5.0 * pa_r * pb_r * r_hat
    prefactor = 3.0 / (4.0 * np.pi * EPS0 * eps_r * r_eff**4)
    return prefactor * bracket


def dipole_pair_energy(
    r_vec,
    p_a,
    p_b,
    eps_r: float = 1.0,
    min_distance: float | None = None,
) -> float:
    """Potential energy of two point dipoles.

    ``r_vec`` points from ``a`` to ``b``. The optional distance floor mirrors
    the force regularization and changes only the radial magnitude.
    """

    if eps_r <= 0.0 or not np.isfinite(eps_r):
        raise ValueError("eps_r must be positive")
    r_vec = np.asarray(r_vec, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if r_vec.shape != (3,) or p_a.shape != (3,) or p_b.shape != (3,):
        raise ValueError("r_vec, p_a, and p_b must be 3-vectors")
    distance = float(np.linalg.norm(r_vec))
    if distance == 0.0:
        return 0.0
    r_hat = r_vec / distance
    if min_distance is None:
        r_eff = distance
    else:
        r_eff = max(distance, float(min_distance))
    if r_eff <= 0.0 or not np.isfinite(r_eff):
        raise ValueError("effective dipole distance must be positive")

    pa_pb = float(np.dot(p_a, p_b))
    pa_r = float(np.dot(p_a, r_hat))
    pb_r = float(np.dot(p_b, r_hat))
    prefactor = 1.0 / (4.0 * np.pi * EPS0 * eps_r * r_eff**3)
    return float(prefactor * (pa_pb - 3.0 * pa_r * pb_r))


def _effective_primary_dipole(
    field_config: ElectricFieldConfig,
    primary_diameter: float,
) -> np.ndarray:
    alpha = resolve_polarizability(field_config, primary_diameter)
    return induced_dipole(alpha, _field_vector(field_config))


def coupled_cluster_dipoles(
    cluster,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
) -> ClusterDipoleResult:
    """Solve the internal coupled-dipole model for one rigid cluster."""

    field_config.validate()
    n_primary = int(cluster.n_primary)
    zeros = np.zeros((n_primary, 3), dtype=float)
    field = _field_vector(field_config)
    if n_primary == 0 or not field_config.enabled or np.linalg.norm(field) == 0.0:
        return ClusterDipoleResult(
            primary_dipoles=zeros,
            total_dipole=np.zeros(3, dtype=float),
            condition_number=1.0,
            n_primary=n_primary,
        )

    alpha = resolve_polarizability(field_config, primary_diameter)
    if alpha == 0.0:
        return ClusterDipoleResult(
            primary_dipoles=zeros,
            total_dipole=np.zeros(3, dtype=float),
            condition_number=1.0,
            n_primary=n_primary,
        )

    rel_positions = np.asarray(cluster.rel_positions, dtype=float).reshape((n_primary, 3))
    radii = np.asarray(cluster.radii, dtype=float).reshape((n_primary,))
    internal_gap = (
        field_config.regularization_gap
        if field_config.coupled_internal_regularization_gap is None
        else field_config.coupled_internal_regularization_gap
    )
    if internal_gap < 0.0 or not np.isfinite(internal_gap):
        raise ValueError("coupled internal regularization gap must be nonnegative")

    matrix = np.eye(3 * n_primary, dtype=float)
    rhs = np.tile(alpha * field, n_primary)
    for i in range(n_primary):
        row = slice(3 * i, 3 * i + 3)
        for j in range(n_primary):
            if i == j:
                continue
            col = slice(3 * j, 3 * j + 3)
            min_distance = float(radii[i] + radii[j] + internal_gap)
            tensor = dipole_field_tensor(
                rel_positions[i] - rel_positions[j],
                eps_r=field_config.medium_relative_permittivity,
                min_distance=min_distance,
            )
            matrix[row, col] = -alpha * tensor

    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise RuntimeError("coupled-dipole matrix condition number is not finite")
    if condition_number > field_config.coupled_condition_warning:
        warnings.warn(
            "coupled-dipole matrix is poorly conditioned "
            f"(condition number {condition_number:.3e})",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        dipoles_flat = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("coupled-dipole matrix solve failed") from exc
    if not np.all(np.isfinite(dipoles_flat)):
        raise RuntimeError("coupled-dipole solution contains non-finite values")

    primary_dipoles = dipoles_flat.reshape((n_primary, 3))
    return ClusterDipoleResult(
        primary_dipoles=primary_dipoles,
        total_dipole=np.sum(primary_dipoles, axis=0),
        condition_number=condition_number,
        n_primary=n_primary,
    )


def cluster_effective_dipole(
    cluster,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
) -> np.ndarray:
    """Return the effective cluster dipole for the configured force model."""

    if field_config.dipole_force_model == "cluster_coupled_dipole":
        return coupled_cluster_dipoles(cluster, field_config, primary_diameter).total_dipole
    return float(cluster.n_primary) * _effective_primary_dipole(field_config, primary_diameter)


def dipole_forces_between_clusters(
    cluster_a,
    cluster_b,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
    boundary: Boundary | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total induced-dipole forces on two clusters."""

    forces, *_summary = dipole_force_energy_diagnostics_on_clusters(
        [cluster_a, cluster_b],
        field_config,
        primary_diameter,
        boundary=boundary,
    )
    return forces[0], forces[1]


def dipole_forces_on_clusters(
    clusters,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
    boundary: Boundary | None = None,
) -> np.ndarray:
    """Return total deterministic induced-dipole force on each cluster COM."""

    forces, *_summary = dipole_force_energy_diagnostics_on_clusters(
        clusters,
        field_config,
        primary_diameter,
        boundary=boundary,
    )
    return forces


def dipole_force_energy_diagnostics_on_clusters(
    clusters,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
    boundary: Boundary | None = None,
) -> tuple[np.ndarray, float, int, int, float, float]:
    """Return dipole forces plus energy and pair-loop diagnostics."""

    field_config.validate()
    if field_config.dipole_force_model == "cluster_coupled_dipole":
        return _cluster_coupled_dipole_force_energy_diagnostics_on_clusters(
            clusters,
            field_config,
            primary_diameter,
            boundary=boundary,
        )
    return _primary_pair_fixed_force_energy_diagnostics_on_clusters(
        clusters,
        field_config,
        primary_diameter,
        boundary=boundary,
    )


def _primary_pair_fixed_force_energy_diagnostics_on_clusters(
    clusters,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
    boundary: Boundary | None = None,
) -> tuple[np.ndarray, float, int, int, float, float]:
    """Fixed-primary-dipole force using all inter-cluster primary pairs."""

    forces = np.zeros((len(clusters), 3), dtype=float)
    if len(clusters) < 2 or not field_config.enabled:
        return forces, 0.0, 0, 0, np.inf, np.inf
    p = _effective_primary_dipole(field_config, primary_diameter)
    if np.linalg.norm(p) == 0.0:
        return forces, 0.0, 0, 0, np.inf, np.inf

    total_energy = 0.0
    evaluated_pairs = 0
    skipped_cutoff_pairs = 0
    min_center_distance = np.inf
    min_surface_gap = np.inf
    cutoff = field_config.dipole_cutoff
    for a_index in range(len(clusters)):
        cluster_a = clusters[a_index]
        centers_a = cluster_a.absolute_centers
        for b_index in range(a_index + 1, len(clusters)):
            cluster_b = clusters[b_index]
            centers_b = cluster_b.absolute_centers
            for i, center_a in enumerate(centers_a):
                for j, center_b in enumerate(centers_b):
                    r_vec = pair_displacement(center_a, center_b, boundary=boundary)
                    distance = float(np.linalg.norm(r_vec))
                    if cutoff is not None and distance > cutoff:
                        skipped_cutoff_pairs += 1
                        continue
                    evaluated_pairs += 1
                    radius_sum = float(cluster_a.radii[i]) + float(cluster_b.radii[j])
                    min_distance = (
                        radius_sum + field_config.regularization_gap
                    )
                    min_center_distance = min(min_center_distance, distance)
                    min_surface_gap = min(min_surface_gap, distance - radius_sum)
                    force_on_b = dipole_pair_force_on_b(
                        r_vec,
                        p,
                        p,
                        eps_r=field_config.medium_relative_permittivity,
                        min_distance=min_distance,
                    )
                    total_energy += dipole_pair_energy(
                        r_vec,
                        p,
                        p,
                        eps_r=field_config.medium_relative_permittivity,
                        min_distance=min_distance,
                    )
                    forces[a_index] -= force_on_b
                    forces[b_index] += force_on_b
    return (
        forces,
        float(total_energy),
        int(evaluated_pairs),
        int(skipped_cutoff_pairs),
        float(min_center_distance),
        float(min_surface_gap),
    )


def _cluster_coupled_dipole_force_energy_diagnostics_on_clusters(
    clusters,
    field_config: ElectricFieldConfig,
    primary_diameter: float,
    boundary: Boundary | None = None,
) -> tuple[np.ndarray, float, int, int, float, float]:
    """Cluster-CDM force using one effective point dipole per cluster.

    In this mode, ``dipole_cutoff`` is interpreted as a cluster-COM distance
    cutoff rather than a primary-primary distance cutoff.
    The approximation self-consistently polarizes primaries only inside each
    rigid cluster. It does not solve mutual inter-cluster polarization, torque,
    rotation, charge forces, restructuring, sintering, or a full conducting-body
    electrostatic boundary-value problem.
    """

    forces = np.zeros((len(clusters), 3), dtype=float)
    if len(clusters) < 2 or not field_config.enabled:
        return forces, 0.0, 0, 0, np.inf, np.inf
    field = _field_vector(field_config)
    if np.linalg.norm(field) == 0.0:
        return forces, 0.0, 0, 0, np.inf, np.inf

    dipoles = [
        coupled_cluster_dipoles(cluster, field_config, primary_diameter).total_dipole
        for cluster in clusters
    ]
    if not any(np.linalg.norm(dipole) > 0.0 for dipole in dipoles):
        return forces, 0.0, 0, 0, np.inf, np.inf

    total_energy = 0.0
    evaluated_pairs = 0
    skipped_cutoff_pairs = 0
    min_center_distance = np.inf
    min_surface_gap = np.inf
    cutoff = field_config.dipole_cutoff
    for a_index in range(len(clusters)):
        cluster_a = clusters[a_index]
        dipole_a = dipoles[a_index]
        for b_index in range(a_index + 1, len(clusters)):
            cluster_b = clusters[b_index]
            r_vec = pair_displacement(cluster_a.position, cluster_b.position, boundary=boundary)
            distance = float(np.linalg.norm(r_vec))
            if cutoff is not None and distance > cutoff:
                skipped_cutoff_pairs += 1
                continue
            evaluated_pairs += 1
            radius_sum = float(cluster_a.bounding_radius + cluster_b.bounding_radius)
            min_distance = radius_sum + field_config.regularization_gap
            min_center_distance = min(min_center_distance, distance)
            min_surface_gap = min(min_surface_gap, distance - radius_sum)
            force_on_b = dipole_pair_force_on_b(
                r_vec,
                dipole_a,
                dipoles[b_index],
                eps_r=field_config.medium_relative_permittivity,
                min_distance=min_distance,
            )
            total_energy += dipole_pair_energy(
                r_vec,
                dipole_a,
                dipoles[b_index],
                eps_r=field_config.medium_relative_permittivity,
                min_distance=min_distance,
            )
            forces[a_index] -= force_on_b
            forces[b_index] += force_on_b
    return (
        forces,
        float(total_energy),
        int(evaluated_pairs),
        int(skipped_cutoff_pairs),
        float(min_center_distance),
        float(min_surface_gap),
    )
