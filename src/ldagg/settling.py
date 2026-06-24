"""Tutorial-style gravity settling validation mode."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from ldagg.constants import BOLTZMANN, DEFAULT_PARTICLE_DENSITY
from ldagg.gas import Gas, particle_mass, sphere_friction
from ldagg.integrators import eb_step
from ldagg.plotting import plot_settling_outputs


@dataclass(slots=True)
class SettlingConfig:
    gas: Gas = field(default_factory=Gas)
    particle_density: float = DEFAULT_PARTICLE_DENSITY
    diameters_nm: tuple[float, ...] = (10.0, 100.0, 1000.0)
    height: float = 0.1
    trials: int = 1000
    dt_factor: float = 1.0e-2
    dt_min: float = 1.0e-12
    dt_max: float = np.inf
    max_steps: int = 2_000_000
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    brownian: bool = True
    seed: int = 12345
    save_trajectory_steps: int = 10_000

    @classmethod
    def from_mapping(cls, data: dict | None) -> SettlingConfig:
        data = dict(data or {})
        gas = Gas.from_mapping(data.pop("gas", None))
        return cls(gas=gas, **data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gas"] = self.gas.to_dict()
        return data


def settling_timestep(
    z: float,
    friction: float,
    force: np.ndarray,
    temperature: float,
    *,
    dt_factor: float = 1.0e-2,
    dt_min: float = 1.0e-12,
    dt_max: float = np.inf,
) -> float:
    """Tutorial timestep ``dt = factor*min(z^2*f/(6kT), z*f/|mg|)``."""

    z = max(float(z), 1.0e-30)
    force_mag = float(np.linalg.norm(force))
    dt_diff = z * z * friction / (6.0 * BOLTZMANN * temperature)
    dt_force = np.inf if force_mag == 0.0 else z * friction / force_mag
    dt = dt_factor * min(dt_diff, dt_force, dt_max)
    return max(dt_min, dt)


def run_settling(config: SettlingConfig, out_dir: str | Path | None = None) -> dict:
    """Run the MATLAB-style settling ensemble."""

    rng = np.random.default_rng(config.seed)
    gravity = np.asarray(config.gravity, dtype=float)
    settling_times: dict[float, list[float]] = {}
    sample_trajectories: dict[float, np.ndarray] = {}
    summary_rows = []

    for diameter_nm in config.diameters_nm:
        diameter = float(diameter_nm) * 1.0e-9
        radius = 0.5 * diameter
        mass = particle_mass(diameter, config.particle_density)
        friction = sphere_friction(diameter, config.gas)
        force = mass * gravity
        speed_std = np.sqrt(BOLTZMANN * config.gas.temperature / mass)
        times = []
        sample = []
        for trial in range(config.trials):
            position = np.array([0.0, 0.0, config.height], dtype=float)
            velocity = speed_std * rng.normal(size=3)
            time = 0.0
            for _step in range(config.max_steps):
                if trial == 0 and len(sample) < config.save_trajectory_steps:
                    sample.append([time, *position, *velocity])
                dt = settling_timestep(
                    position[2],
                    friction,
                    force,
                    config.gas.temperature,
                    dt_factor=config.dt_factor,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                )
                result = eb_step(
                    position,
                    velocity,
                    mass,
                    friction,
                    force,
                    dt,
                    config.gas.temperature,
                    rng,
                    brownian=config.brownian,
                    dt_min=config.dt_min * 1.0e-6,
                )
                position = result.position
                velocity = result.velocity
                time += result.dt
                if position[2] <= radius:
                    times.append(time)
                    break
            else:
                times.append(np.nan)
        settling_times[float(diameter_nm)] = times
        sample_trajectories[float(diameter_nm)] = np.asarray(sample, dtype=float)
        finite = np.asarray(times, dtype=float)
        finite = finite[np.isfinite(finite)]
        summary_rows.append(
            {
                "diameter_nm": float(diameter_nm),
                "trials": int(config.trials),
                "completed": int(len(finite)),
                "mean_time_s": float(np.mean(finite)) if len(finite) else np.nan,
                "std_time_s": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "mass_kg": float(mass),
                "friction_kg_s": float(friction),
            }
        )

    result = {
        "config": config.to_dict(),
        "summary": summary_rows,
        "settling_times": {k: np.asarray(v, dtype=float) for k, v in settling_times.items()},
        "sample_trajectories": sample_trajectories,
    }
    if out_dir is not None:
        write_settling_outputs(result, out_dir)
    return result


def write_settling_outputs(result: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "config_used.yml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result["config"], fh, sort_keys=False)
    with (out / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"summary": result["summary"]}, fh, indent=2)
    pd.DataFrame(result["summary"]).to_csv(out / "cluster_stats.csv", index=False)
    pd.DataFrame().to_csv(out / "events.csv", index=False)
    pd.DataFrame().to_csv(out / "final_aggregate.csv", index=False)

    with h5py.File(out / "trajectory_sample.h5", "w") as h5:
        h5.attrs["mode"] = "settling"
        for diameter_nm, times in result["settling_times"].items():
            group = h5.create_group(f"dp_{diameter_nm:g}_nm")
            group.create_dataset("settling_times", data=np.asarray(times, dtype=float))
            group.create_dataset("sample_trajectory", data=result["sample_trajectories"][diameter_nm])

    plot_settling_outputs(result["settling_times"], result["summary"], plots)
