"""Typer command line interface for ldagg."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich import print

from ldagg.analysis import summarize_h5
from ldagg.plotting import plot_h5_summary
from ldagg.sequential_growth import SequentialGrowthConfig, run_sequential_growth
from ldagg.settling import SettlingConfig, run_settling
from ldagg.simulation import CoagulationConfig, run_coagulation

app = typer.Typer(help="Langevin Dynamics aerosol agglomeration.")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@app.command()
def settling(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    out: Annotated[Path, typer.Option(..., "--out", help="Output directory.")],
) -> None:
    cfg = SettlingConfig.from_mapping(load_yaml(config))
    result = run_settling(cfg, out)
    print(f"settling complete: {out} ({len(result['summary'])} particle sizes)")


@app.command()
def coagulate(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    out: Annotated[Path, typer.Option(..., "--out", help="Output directory.")],
) -> None:
    cfg = CoagulationConfig.from_mapping(load_yaml(config))
    result = run_coagulation(cfg, out)
    print(
        f"coagulation complete: {out} ({len(result.clusters)} clusters, {len(result.events)} events)"
    )


@app.command()
def grow(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    out: Annotated[Path, typer.Option(..., "--out", help="Output directory.")],
) -> None:
    cfg = SequentialGrowthConfig.from_mapping(load_yaml(config))
    result = run_sequential_growth(cfg, out)
    print(f"growth complete: {out} (final size {result['aggregate'].n_primary})")


@app.command()
def summarize(path: Path) -> None:
    print(json.dumps(summarize_h5(str(path)), indent=2))


@app.command()
def plot(
    path: Path,
    out: Annotated[Path, typer.Option(..., "--out", help="Plot output directory.")],
) -> None:
    plot_h5_summary(path, out)
    print(f"plots written to {out}")


if __name__ == "__main__":
    app()
