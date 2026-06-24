from __future__ import annotations

import sys

import h5py
import pytest

from ldagg.plotting import (
    AGGLOMERATE_COLOR,
    MONOMER_COLOR,
    _box_bounds,
    _pyvista_add_box,
    _pyvista_set_camera,
    _snapshot_colors,
    _trajectory_segments,
    load_agglomeration_snapshots,
    plot_agglomeration_snapshot,
    plot_agglomeration_trajectories,
    save_agglomeration_snapshots,
)
from ldagg.simulation import CoagulationConfig, run_coagulation


def test_snapshot_colors_monomers_blue_and_agglomerates_gray() -> None:
    snapshot = {
        "ids": [10, 11],
        "sizes": [1, 2],
        "primary_cluster_ids": [10, 11, 11],
    }
    assert _snapshot_colors(snapshot, snapshot["primary_cluster_ids"]) == [
        MONOMER_COLOR,
        AGGLOMERATE_COLOR,
        AGGLOMERATE_COLOR,
    ]


def test_box_bounds_are_fixed_domain_bounds() -> None:
    assert _box_bounds(2.0) == (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    assert _box_bounds((1.0, 2.0, 3.0)) == (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)


def test_periodic_trajectory_segments_break_wrap_jumps() -> None:
    centers = [
        [0.85, 0.2, 0.2],
        [0.95, 0.2, 0.2],
        [0.05, 0.2, 0.2],
        [0.15, 0.2, 0.2],
    ]
    periodic = _trajectory_segments(centers, box_size=1.0, boundary_mode="periodic")
    finite = _trajectory_segments(centers, box_size=1.0, boundary_mode="finite")

    assert len(periodic) == 2
    assert [len(segment) for segment in periodic] == [2, 2]
    assert len(finite) == 1
    assert len(finite[0]) == 4


def test_pyvista_box_camera_ignores_actor_bounds() -> None:
    if sys.platform == "win32":
        pytest.skip("PyVista offscreen rendering is unstable when Windows Python is launched from WSL")
    pyvista = pytest.importorskip("pyvista")

    cameras = []
    for center in ([0.5, 0.5, 0.5], [1.6, 0.5, 0.5]):
        plotter = pyvista.Plotter(off_screen=True, window_size=(640, 480))
        plotter.add_mesh(pyvista.Sphere(radius=0.1, center=center))
        _pyvista_add_box(plotter, pyvista, 1.0)
        _pyvista_set_camera(plotter, 1.0)
        cameras.append(
            (
                plotter.camera.position,
                plotter.camera.focal_point,
                plotter.camera.up,
                plotter.camera.parallel_scale,
            )
        )
        plotter.close()

    for first, second in zip(cameras[0], cameras[1], strict=True):
        assert first == pytest.approx(second)


def test_agglomeration_snapshot_visualization_outputs(tmp_path) -> None:
    config = CoagulationConfig(
        n_particles=4,
        diameter=100.0e-9,
        box_size=4.5e-7,
        t_end=2.0e-6,
        max_steps=50,
        dt_max=5.0e-7,
        seed=11,
        save_every=1,
        capture_tolerance=5.0e-9,
    )
    run_coagulation(config, tmp_path, make_plots=False)
    h5_path = tmp_path / "run.h5"

    with h5py.File(h5_path, "r") as h5:
        first = h5["snapshots"]["000000"]
        assert "primary_centers" in first
        assert "primary_radii" in first
        assert "primary_ids" in first
        assert "primary_cluster_ids" in first

    if sys.platform == "win32":
        pytest.skip("Matplotlib rendering is unstable when Windows Python is launched from WSL")

    plot_path = tmp_path / "plots" / "trajectories.png"
    fig, _ax = plot_agglomeration_trajectories(h5_path, plot_path)
    fig.clf()
    assert plot_path.exists()

    frames = save_agglomeration_snapshots(h5_path, tmp_path / "frames", max_frames=2)
    assert len(frames) == 2
    assert all(path.exists() for path in frames)

    pytest.importorskip("pyvista")
    image = plot_agglomeration_trajectories(
        h5_path,
        tmp_path / "plots" / "trajectories_pyvista.png",
        backend="pyvista",
        max_frames=3,
    )
    assert image.ndim == 3
    assert (tmp_path / "plots" / "trajectories_pyvista.png").exists()

    sampled = load_agglomeration_snapshots(h5_path, max_frames=2)
    all_snaps = load_agglomeration_snapshots(h5_path)
    assert sampled[0]["time"] == all_snaps[0]["time"]
    assert sampled[-1]["time"] == all_snaps[-1]["time"]


def test_pyvista_snapshot_backend_outputs_true_radius_image(tmp_path) -> None:
    if sys.platform == "win32":
        pytest.skip("PyVista offscreen rendering is unstable when Windows Python is launched from WSL")
    pytest.importorskip("pyvista")

    snapshot = {
        "time": 0.0,
        "primary_centers": [
            [0.0, 0.0, 0.0],
            [100.0e-9, 0.0, 0.0],
            [200.0e-9, 0.0, 0.0],
        ],
        "primary_radii": [50.0e-9, 50.0e-9, 50.0e-9],
        "primary_cluster_ids": [0, 0, 0],
    }
    out_path = tmp_path / "trimer_pyvista.png"
    image = plot_agglomeration_snapshot(
        snapshot,
        out_path,
        backend="pyvista",
        box_size=None,
        title="test trimer",
    )
    assert out_path.exists()
    assert image.ndim == 3
    assert image.shape[-1] == 3
    assert image.sum() > 0
