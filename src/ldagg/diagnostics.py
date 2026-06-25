"""Diagnostic helpers for force, transport, and aggregate morphology."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ldagg.clusters import Cluster
from ldagg.collisions import nearest_surface_gap
from ldagg.constants import BOLTZMANN, EPS0
from ldagg.electric import (
    ElectricFieldConfig,
    cluster_effective_dipole,
    coupled_cluster_dipoles,
    induced_dipole,
    resolve_polarizability,
)

_EPS = 1.0e-300


@dataclass(slots=True)
class DiagnosticsConfig:
    enabled: bool = False
    every: int = 1
    store_snapshot_forces: bool = True
    store_event_metrics: bool = True
    store_pair_summary: bool = True

    @classmethod
    def from_mapping(cls, data: dict | None) -> DiagnosticsConfig:
        if isinstance(data, cls):
            return data
        values = dict(data or {})
        config = cls(**values)
        if config.every < 1:
            raise ValueError("diagnostics.every must be >= 1")
        return config

    def to_dict(self) -> dict:
        return asdict(self)


def electric_field_scalars(
    field_config: ElectricFieldConfig,
    diameter: float,
    temperature: float,
) -> dict[str, float]:
    """Return scalar diagnostics for the external field and induced dipole."""

    field = np.asarray(field_config.vector, dtype=float)
    alpha = resolve_polarizability(field_config, diameter)
    p = induced_dipole(alpha, field)
    e_norm = float(np.linalg.norm(field))
    p_norm = float(np.linalg.norm(p))
    gamma = 0.0
    if field_config.enabled and e_norm > 0.0 and p_norm > 0.0:
        r_contact = float(diameter) + field_config.regularization_gap
        c_dipole = p_norm * p_norm / (
            4.0 * np.pi * EPS0 * field_config.medium_relative_permittivity
        )
        u_head_to_tail = -2.0 * c_dipole / r_contact**3
        gamma = abs(u_head_to_tail) / (BOLTZMANN * float(temperature))
    return {
        "E_x": float(field[0]),
        "E_y": float(field[1]),
        "E_z": float(field[2]),
        "E_norm": e_norm,
        "alpha": float(alpha),
        "p_x": float(p[0]),
        "p_y": float(p[1]),
        "p_z": float(p[2]),
        "p_norm": p_norm,
        "Gamma_dd_contact": float(gamma),
    }


def force_diagnostics(
    dipole_forces: np.ndarray,
    gravity_forces: np.ndarray,
    total_forces: np.ndarray,
) -> dict[str, float]:
    """Return scalar diagnostics for deterministic force components."""

    dipole = np.asarray(dipole_forces, dtype=float).reshape((-1, 3))
    gravity = np.asarray(gravity_forces, dtype=float).reshape((-1, 3))
    total = np.asarray(total_forces, dtype=float).reshape((-1, 3))
    dipole_norms = np.linalg.norm(dipole, axis=1) if len(dipole) else np.empty(0)
    gravity_norms = np.linalg.norm(gravity, axis=1) if len(gravity) else np.empty(0)
    total_norms = np.linalg.norm(total, axis=1) if len(total) else np.empty(0)
    dipole_denominator = float(np.sum(dipole_norms)) + _EPS
    return {
        "max_F_dipole": float(np.max(dipole_norms)) if len(dipole_norms) else 0.0,
        "mean_F_dipole": float(np.mean(dipole_norms)) if len(dipole_norms) else 0.0,
        "max_F_gravity": float(np.max(gravity_norms)) if len(gravity_norms) else 0.0,
        "max_F_total": float(np.max(total_norms)) if len(total_norms) else 0.0,
        "mean_F_total": float(np.mean(total_norms)) if len(total_norms) else 0.0,
        "newton_residual_dipole": float(np.linalg.norm(np.sum(dipole, axis=0)) / dipole_denominator)
        if len(dipole)
        else 0.0,
    }


def cluster_dipole_diagnostics(
    clusters: list[Cluster],
    field_config: ElectricFieldConfig,
    primary_diameter: float,
) -> dict[str, float | str]:
    """Return effective cluster-dipole diagnostics for electric-field runs."""

    row: dict[str, float | str] = {
        "dipole_force_model": field_config.dipole_force_model,
        "largest_cluster_P_x": float("nan"),
        "largest_cluster_P_y": float("nan"),
        "largest_cluster_P_z": float("nan"),
        "largest_cluster_P_norm": float("nan"),
        "largest_cluster_P_over_N_alpha_E": float("nan"),
        "max_cluster_P_norm": float("nan"),
        "mean_cluster_P_norm": float("nan"),
        "max_cluster_cdm_condition": float("nan"),
        "mean_cluster_cdm_condition": float("nan"),
    }
    if not clusters:
        return row
    field = np.asarray(field_config.vector, dtype=float)
    field_norm = float(np.linalg.norm(field))
    alpha = resolve_polarizability(field_config, primary_diameter)
    if not field_config.enabled or field_norm == 0.0 or alpha == 0.0:
        return row

    dipoles = []
    conditions = []
    for cluster in clusters:
        if field_config.dipole_force_model == "cluster_coupled_dipole":
            result = coupled_cluster_dipoles(cluster, field_config, primary_diameter)
            dipoles.append(result.total_dipole)
            conditions.append(result.condition_number)
        else:
            dipoles.append(cluster_effective_dipole(cluster, field_config, primary_diameter))
            conditions.append(float("nan"))

    dipoles_array = np.asarray(dipoles, dtype=float).reshape((len(clusters), 3))
    dipole_norms = np.linalg.norm(dipoles_array, axis=1)
    sizes = np.asarray([cluster.n_primary for cluster in clusters], dtype=float)
    largest_index = int(np.argmax(sizes))
    largest_dipole = dipoles_array[largest_index]
    largest_norm = float(dipole_norms[largest_index])
    denominator = float(sizes[largest_index] * alpha * field_norm)
    finite_conditions = np.asarray(
        [value for value in conditions if np.isfinite(value)],
        dtype=float,
    )
    row.update(
        {
            "largest_cluster_P_x": float(largest_dipole[0]),
            "largest_cluster_P_y": float(largest_dipole[1]),
            "largest_cluster_P_z": float(largest_dipole[2]),
            "largest_cluster_P_norm": largest_norm,
            "largest_cluster_P_over_N_alpha_E": float(largest_norm / denominator)
            if denominator > 0.0
            else float("nan"),
            "max_cluster_P_norm": float(np.max(dipole_norms)),
            "mean_cluster_P_norm": float(np.mean(dipole_norms)),
            "max_cluster_cdm_condition": float(np.max(finite_conditions))
            if len(finite_conditions)
            else float("nan"),
            "mean_cluster_cdm_condition": float(np.mean(finite_conditions))
            if len(finite_conditions)
            else float("nan"),
        }
    )
    return row


def transport_diagnostics(
    clusters: list[Cluster],
    total_forces: np.ndarray,
    dt: float,
    temperature: float,
    *,
    boundary=None,
) -> dict[str, float]:
    """Return timestep-scale deterministic drift and Brownian displacement diagnostics."""

    if not clusters:
        return {
            "min_surface_gap": float("nan"),
            "max_drift_step": 0.0,
            "mean_drift_step": 0.0,
            "max_brownian_rms_step": 0.0,
            "mean_brownian_rms_step": 0.0,
            "max_drift_to_brownian_ratio": 0.0,
            "mean_drift_to_brownian_ratio": 0.0,
        }
    forces = np.asarray(total_forces, dtype=float).reshape((len(clusters), 3))
    frictions = np.asarray([cluster.friction for cluster in clusters], dtype=float)
    drift_steps = np.linalg.norm(forces, axis=1) / frictions * float(dt)
    diffusion = BOLTZMANN * float(temperature) / frictions
    brownian_rms = np.sqrt(6.0 * diffusion * float(dt))
    ratios = drift_steps / (brownian_rms + _EPS)
    return {
        "min_surface_gap": float(nearest_surface_gap(clusters, boundary=boundary)),
        "max_drift_step": float(np.max(drift_steps)),
        "mean_drift_step": float(np.mean(drift_steps)),
        "max_brownian_rms_step": float(np.max(brownian_rms)),
        "mean_brownian_rms_step": float(np.mean(brownian_rms)),
        "max_drift_to_brownian_ratio": float(np.max(ratios)),
        "mean_drift_to_brownian_ratio": float(np.mean(ratios)),
    }


def morphology_diagnostics(
    cluster: Cluster,
    *,
    electric_field_vector: np.ndarray | tuple[float, float, float] | None = None,
    electric_field_enabled: bool = False,
    contact_tolerance: float = 1.0e-12,
) -> dict[str, float]:
    """Return simple morphology metrics for one rigid aggregate."""

    rel = np.asarray(cluster.rel_positions, dtype=float)
    masses = np.asarray(cluster.masses, dtype=float)
    if len(rel) == 0:
        return {
            "radius_of_gyration": 0.0,
            "aspect_ratio": 0.0,
            "alignment_order_S": float("nan"),
            "branching_fraction": 0.0,
            "max_coordination": 0.0,
            "mean_coordination": 0.0,
        }
    weights = masses / float(np.sum(masses))
    gyration_tensor = np.einsum("n,ni,nj->ij", weights, rel, rel)
    eigenvalues, eigenvectors = np.linalg.eigh(gyration_tensor)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    radius_of_gyration = float(np.sqrt(np.sum(eigenvalues)))
    aspect_ratio = float(np.sqrt(eigenvalues[0] / max(eigenvalues[-1], _EPS)))

    alignment = float("nan")
    if electric_field_enabled and electric_field_vector is not None:
        field = np.asarray(electric_field_vector, dtype=float)
        field_norm = float(np.linalg.norm(field))
        if field_norm > 0.0:
            e_hat = field / field_norm
            e1 = eigenvectors[:, 0]
            alignment = float((3.0 * float(np.dot(e1, e_hat)) ** 2 - 1.0) / 2.0)

    coordination = primary_coordination_numbers(cluster, contact_tolerance=contact_tolerance)
    return {
        "radius_of_gyration": radius_of_gyration,
        "aspect_ratio": aspect_ratio,
        "alignment_order_S": alignment,
        "branching_fraction": float(np.mean(coordination >= 3)) if len(coordination) else 0.0,
        "max_coordination": float(np.max(coordination)) if len(coordination) else 0.0,
        "mean_coordination": float(np.mean(coordination)) if len(coordination) else 0.0,
    }


def primary_coordination_numbers(
    cluster: Cluster,
    *,
    contact_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Count touching-neighbor primary spheres inside one aggregate."""

    centers = cluster.absolute_centers
    coordination = np.zeros(cluster.n_primary, dtype=np.int64)
    for i in range(cluster.n_primary):
        for j in range(i + 1, cluster.n_primary):
            distance = float(np.linalg.norm(centers[j] - centers[i]))
            threshold = float(cluster.radii[i] + cluster.radii[j] + contact_tolerance)
            if distance <= threshold:
                coordination[i] += 1
                coordination[j] += 1
    return coordination
