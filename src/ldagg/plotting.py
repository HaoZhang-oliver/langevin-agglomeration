"""Plotting utilities for examples and CLI output."""

from __future__ import annotations

from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

MONOMER_COLOR = "#1f77b4"
AGGLOMERATE_COLOR = "#cfcfcf"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _apply_margins(fig) -> None:
    """Use stable fixed margins instead of Matplotlib tight_layout."""

    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.9)


def _box_size_array(box_size: float | tuple[float, float, float] | np.ndarray | None) -> np.ndarray | None:
    if box_size is None:
        return None
    if np.isscalar(box_size):
        return np.full(3, float(box_size))
    return np.asarray(box_size, dtype=float)


def _box_bounds(box_size: float | tuple[float, float, float] | np.ndarray | None):
    size = _box_size_array(box_size)
    if size is None:
        return None
    return (0.0, float(size[0]), 0.0, float(size[1]), 0.0, float(size[2]))


def _draw_box(ax, box_size: float | tuple[float, float, float] | np.ndarray | None) -> None:
    size = _box_size_array(box_size)
    if size is None:
        return
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=float,
    )
    corners *= size[None, :]
    ax.plot(corners[:5, 0], corners[:5, 1], corners[:5, 2], color="0.7", linewidth=0.8)
    ax.plot(corners[5:, 0], corners[5:, 1], corners[5:, 2], color="0.7", linewidth=0.8)
    for i in range(4):
        ax.plot(
            [corners[i, 0], corners[i + 5, 0]],
            [corners[i, 1], corners[i + 5, 1]],
            [corners[i, 2], corners[i + 5, 2]],
            color="0.7",
            linewidth=0.8,
        )


def _set_equal_axes(
    ax,
    points: np.ndarray,
    radii: np.ndarray | None = None,
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
) -> None:
    size = _box_size_array(box_size)
    if size is not None:
        ax.set_xlim(0.0, size[0])
        ax.set_ylim(0.0, size[1])
        ax.set_zlim(0.0, size[2])
        try:
            ax.set_box_aspect(size)
        except AttributeError:
            pass
        return
    if len(points) == 0:
        return
    pad = 0.0 if radii is None or len(radii) == 0 else float(np.max(radii))
    pmin = np.min(points, axis=0) - pad
    pmax = np.max(points, axis=0) + pad
    center = 0.5 * (pmin + pmax)
    span = max(float(np.max(pmax - pmin)), 1.0e-30)
    half = 0.55 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _snapshot_points(snapshot: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "primary_centers" in snapshot:
        points = np.asarray(snapshot["primary_centers"], dtype=float)
        radii = np.asarray(snapshot.get("primary_radii", np.ones(len(points))), dtype=float)
        labels = np.asarray(snapshot.get("primary_cluster_ids", np.arange(len(points))), dtype=np.int64)
    else:
        points = np.asarray(snapshot.get("positions", np.empty((0, 3))), dtype=float)
        radii = np.ones(len(points), dtype=float)
        labels = np.asarray(snapshot.get("ids", np.arange(len(points))), dtype=np.int64)
    return points, radii, labels


def _snapshot_cluster_sizes(snapshot: dict, labels: np.ndarray) -> dict[int, int]:
    ids = np.asarray(snapshot.get("ids", []), dtype=np.int64)
    sizes = np.asarray(snapshot.get("sizes", []), dtype=np.int64)
    if len(ids) == len(sizes) and len(ids):
        cluster_sizes = {int(cid): int(size) for cid, size in zip(ids, sizes, strict=True)}
    else:
        cluster_sizes = {}

    if len(labels):
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts, strict=True):
            cluster_sizes.setdefault(int(label), int(count))
    return cluster_sizes


def _cluster_size_color(size: int) -> str:
    return MONOMER_COLOR if size <= 1 else AGGLOMERATE_COLOR


def _snapshot_colors(snapshot: dict, labels: np.ndarray) -> list[str]:
    cluster_sizes = _snapshot_cluster_sizes(snapshot, labels)
    return [_cluster_size_color(cluster_sizes.get(int(label), 1)) for label in labels]


def _primary_colors_from_final_snapshot(
    primary_ids: list[int],
    final_snapshot: dict,
) -> list[str]:
    if not final_snapshot or "primary_ids" not in final_snapshot or "primary_cluster_ids" not in final_snapshot:
        return [MONOMER_COLOR for _pid in primary_ids]

    primary_snapshot_ids = np.asarray(final_snapshot["primary_ids"], dtype=np.int64)
    primary_cluster_ids = np.asarray(final_snapshot["primary_cluster_ids"], dtype=np.int64)
    cluster_sizes = _snapshot_cluster_sizes(final_snapshot, primary_cluster_ids)
    primary_to_size = {
        int(pid): cluster_sizes.get(int(cid), 1)
        for pid, cid in zip(primary_snapshot_ids, primary_cluster_ids, strict=True)
    }
    return [_cluster_size_color(primary_to_size.get(int(pid), 1)) for pid in primary_ids]


def _sphere_xyz(center: np.ndarray, radius: float, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _plot_physical_spheres(
    ax,
    centers: np.ndarray,
    radii: np.ndarray,
    colors: list,
    *,
    resolution: int = 16,
    alpha: float = 0.9,
) -> None:
    for center, radius, color in zip(centers, radii, colors, strict=True):
        x, y, z = _sphere_xyz(center, float(radius), resolution)
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            edgecolor="0.25",
            linewidth=0.12,
            alpha=alpha,
            shade=True,
            rstride=1,
            cstride=1,
        )


def _read_snapshot_group(group) -> dict:
    snap = {"time": float(group.attrs.get("time", np.nan))}
    for key in (
        "ids",
        "positions",
        "velocities",
        "sizes",
        "primary_centers",
        "primary_radii",
        "primary_ids",
        "primary_cluster_ids",
    ):
        if key in group:
            snap[key] = group[key][:]
    return snap


def _sample_names(names: list[str], every: int = 1, max_frames: int | None = None) -> list[str]:
    selected = names[:: max(1, every)]
    if max_frames is not None and len(selected) > max_frames:
        if max_frames <= 0:
            return []
        indices = np.linspace(0, len(selected) - 1, max_frames, dtype=int)
        selected = [selected[int(index)] for index in indices]
    return selected


def load_agglomeration_snapshots(
    h5_path: str | Path,
    *,
    every: int = 1,
    max_frames: int | None = None,
) -> list[dict]:
    """Load saved agglomeration snapshots from a run HDF5 file.

    When ``max_frames`` is provided, frames are sampled over the full saved
    trajectory instead of taking only the beginning of the run.
    """

    snapshots = []
    with h5py.File(h5_path, "r") as h5:
        if "snapshots" not in h5:
            return snapshots
        names = sorted(h5["snapshots"], key=lambda item: int(item))
        for name in _sample_names(names, every=every, max_frames=max_frames):
            snapshots.append(_read_snapshot_group(h5["snapshots"][name]))
    return snapshots


def h5_box_size(h5_path: str | Path) -> float | None:
    """Return the box size stored in a run file, if present."""

    with h5py.File(h5_path, "r") as h5:
        value = h5.attrs.get("box_size")
        return None if value is None else float(value)


def h5_boundary_mode(h5_path: str | Path) -> str | None:
    """Return the boundary mode stored in a run file, if present."""

    with h5py.File(h5_path, "r") as h5:
        value = h5.attrs.get("boundary_mode")
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)


def _pyvista_module():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista visualization requires pyvista. Install it with "
            "`pip install pyvista imageio imageio-ffmpeg`."
        ) from exc
    pv.OFF_SCREEN = True
    return pv


def _pyvista_add_box(plotter, pv, box_size) -> None:
    size = _box_size_array(box_size)
    if size is None:
        return
    box = pv.Box(bounds=(0.0, size[0], 0.0, size[1], 0.0, size[2]))
    plotter.add_mesh(box, style="wireframe", color="gray", line_width=1.0, opacity=0.55)


def _pyvista_add_axes_labels(plotter, pv, box_size, points) -> None:
    size = _box_size_array(box_size)
    if size is None and len(points):
        pmin = np.min(points, axis=0)
        pmax = np.max(points, axis=0)
        center = 0.5 * (pmin + pmax)
        span = max(float(np.max(pmax - pmin)), 1.0e-30)
        pmin = center - 0.65 * span
        pmax = center + 0.65 * span
    elif size is not None:
        pmin = np.zeros(3)
        pmax = size
    else:
        return
    label_points = np.array(
        [
            [pmax[0], pmin[1], pmin[2]],
            [pmin[0], pmax[1], pmin[2]],
            [pmin[0], pmin[1], pmax[2]],
        ]
    )
    plotter.add_point_labels(
        label_points,
        ["x", "y", "z"],
        font_size=14,
        text_color="black",
        shape_opacity=0.0,
        always_visible=True,
    )


def _pyvista_set_camera(
    plotter,
    box_size: float | tuple[float, float, float] | np.ndarray | None,
    *,
    closeup: bool = False,
    camera_position=None,
) -> None:
    if camera_position is not None:
        plotter.camera_position = camera_position
        return

    bounds = None if closeup else _box_bounds(box_size)
    if bounds is None:
        plotter.view_isometric()
        plotter.reset_camera()
    else:
        # Keep the domain fixed in videos. PyVista's view_isometric() and
        # VTK reset_camera() consult actor bounds, which jitter for periodic
        # particles that temporarily render outside [0, L]. Compute a fixed
        # orthographic isometric camera directly from the simulation box.
        size = np.asarray(_box_size_array(box_size), dtype=float)
        center = 0.5 * size
        direction = np.array([1.0, 1.0, 1.0], dtype=float)
        direction /= np.linalg.norm(direction)
        viewup = np.array([0.0, 0.0, 1.0], dtype=float)
        viewup -= np.dot(viewup, direction) * direction
        viewup /= np.linalg.norm(viewup)
        right = np.cross(direction, viewup)

        corners = np.array(
            [
                [x, y, z]
                for x in (0.0, size[0])
                for y in (0.0, size[1])
                for z in (0.0, size[2])
            ],
            dtype=float,
        )
        offsets = corners - center
        half_height = float(np.max(np.abs(offsets @ viewup)))
        half_width = float(np.max(np.abs(offsets @ right)))
        width, height = plotter.window_size
        aspect = max(float(width) / max(float(height), 1.0), 1.0e-12)
        margin = 1.08
        zoom = 1.05
        parallel_scale = margin * max(half_height, half_width / aspect) / zoom
        distance = 3.0 * max(float(np.linalg.norm(size)), 1.0e-30)

        plotter.camera_position = (
            tuple(center + direction * distance),
            tuple(center),
            tuple(viewup),
        )
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
        plotter.camera.clipping_range = (
            max(distance - 2.5 * np.linalg.norm(size), 1.0e-30),
            distance + 2.5 * np.linalg.norm(size),
        )
        return
    plotter.camera.zoom(1.35 if closeup else 1.05)


def _pyvista_snapshot_image(
    snapshot: dict,
    *,
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    title: str | None = None,
    show_box: bool = True,
    show_com: bool = False,
    closeup: bool = False,
    window_size: tuple[int, int] = (1200, 900),
    sphere_theta_resolution: int = 48,
    sphere_phi_resolution: int = 24,
    background: str = "white",
    camera_position=None,
) -> np.ndarray:
    pv = _pyvista_module()
    points, radii, labels = _snapshot_points(snapshot)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    colors = _snapshot_colors(snapshot, labels) if len(points) else []
    for center, radius, color in zip(points, radii, colors, strict=True):
        sphere = pv.Sphere(
            radius=float(radius),
            center=tuple(float(v) for v in center),
            theta_resolution=sphere_theta_resolution,
            phi_resolution=sphere_phi_resolution,
        )
        plotter.add_mesh(
            sphere,
            color=color,
            smooth_shading=True,
            specular=0.35,
            specular_power=18.0,
            ambient=0.25,
        )
    if show_com and "positions" in snapshot and len(snapshot["positions"]):
        com = pv.PolyData(np.asarray(snapshot["positions"], dtype=float))
        plotter.add_mesh(com, color="black", point_size=9.0, render_points_as_spheres=True)
    if show_box and not closeup:
        _pyvista_add_box(plotter, pv, box_size)
    _pyvista_add_axes_labels(plotter, pv, box_size if not closeup else None, points)
    if title:
        plotter.add_text(title, position="upper_left", font_size=10, color="black")
    plotter.add_light(pv.Light(light_type="headlight", intensity=0.45))
    _pyvista_set_camera(plotter, box_size, closeup=closeup, camera_position=camera_position)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image


def plot_agglomeration_snapshot_pyvista(
    snapshot: dict,
    out_path: str | Path | None = None,
    *,
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    title: str | None = None,
    show_box: bool = True,
    show_com: bool = False,
    closeup: bool = False,
    window_size: tuple[int, int] = (1200, 900),
    sphere_theta_resolution: int = 48,
    sphere_phi_resolution: int = 24,
    background: str = "white",
    camera_position=None,
) -> np.ndarray:
    """Render one agglomeration snapshot with PyVista/VTK true-radius spheres."""

    image = _pyvista_snapshot_image(
        snapshot,
        box_size=box_size,
        title=title,
        show_box=show_box,
        show_com=show_com,
        closeup=closeup,
        window_size=window_size,
        sphere_theta_resolution=sphere_theta_resolution,
        sphere_phi_resolution=sphere_phi_resolution,
        background=background,
        camera_position=camera_position,
    )
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, image)
    return image


def plot_agglomeration_snapshot(
    snapshot: dict,
    out_path: str | Path | None = None,
    *,
    ax=None,
    backend: str = "matplotlib",
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    title: str | None = None,
    elev: float = 30.0,
    azim: float = 30.0,
    show_box: bool = True,
    show_com: bool = False,
    render_spheres: bool = True,
    sphere_resolution: int = 16,
    marker_scale: float = 180.0,
    dpi: int = 160,
):
    """Plot one agglomeration snapshot in 3D.

    The visual style mirrors the tutorial MATLAB settling visualizer: 3D axes,
    fixed view angle, grid, and trajectory-friendly limits.
    """

    if backend == "pyvista":
        if ax is not None:
            raise ValueError("PyVista backend does not accept a Matplotlib ax")
        return plot_agglomeration_snapshot_pyvista(
            snapshot,
            out_path,
            box_size=box_size,
            title=title,
            show_box=show_box,
            show_com=show_com,
            closeup=box_size is None,
            sphere_theta_resolution=max(16, 2 * sphere_resolution),
            sphere_phi_resolution=max(8, sphere_resolution),
        )
    if backend != "matplotlib":
        raise ValueError("backend must be 'matplotlib' or 'pyvista'")

    plt = _pyplot()
    created = ax is None
    if created:
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    points, radii, labels = _snapshot_points(snapshot)
    if len(points):
        colors = _snapshot_colors(snapshot, labels)
        if render_spheres and np.all(np.isfinite(radii)) and np.max(radii) > 0.0:
            _plot_physical_spheres(
                ax,
                points,
                radii,
                colors,
                resolution=sphere_resolution,
                alpha=0.92,
            )
        else:
            scale = np.maximum(radii / max(float(np.max(radii)), 1.0e-30), 0.15)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=marker_scale * scale**2,
                c=colors,
                edgecolors="k",
                linewidths=0.35,
                alpha=0.92,
                depthshade=True,
            )
    if show_com and "positions" in snapshot and len(snapshot["positions"]):
        com = np.asarray(snapshot["positions"], dtype=float)
        ax.scatter(com[:, 0], com[:, 1], com[:, 2], s=20, c="k", marker="x", linewidths=0.7)
    if show_box:
        _draw_box(ax, box_size)
    _set_equal_axes(ax, points, radii, box_size)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if title is None:
        time = snapshot.get("time")
        count = len(np.unique(labels)) if len(labels) else len(snapshot.get("ids", []))
        title = f"t = {time:.3e} s, clusters = {count}" if time is not None else "Agglomeration"
    ax.set_title(title)
    if out_path is not None:
        _apply_margins(fig)
        fig.savefig(out_path, dpi=dpi)
    if created:
        return fig, ax
    return ax


def _trajectory_tracks_from_snapshots(
    snapshots: list[dict],
    primary_ids: list[int] | None = None,
) -> tuple[dict[int, list[np.ndarray]], dict]:
    tracks: dict[int, list[np.ndarray]] = {}
    final_snapshot = snapshots[-1] if snapshots else {}
    for snap in snapshots:
        if "primary_centers" in snap and "primary_ids" in snap:
            for pid, center in zip(snap["primary_ids"], snap["primary_centers"], strict=True):
                if primary_ids is None or int(pid) in primary_ids:
                    tracks.setdefault(int(pid), []).append(np.asarray(center, dtype=float))
        elif "positions" in snap and "ids" in snap:
            for pid, center in zip(snap["ids"], snap["positions"], strict=True):
                if primary_ids is None or int(pid) in primary_ids:
                    tracks.setdefault(int(pid), []).append(np.asarray(center, dtype=float))
    return tracks, final_snapshot


def _trajectory_points(tracks: dict[int, list[np.ndarray]]) -> np.ndarray:
    arrays = [np.asarray(centers, dtype=float) for centers in tracks.values() if len(centers)]
    if not arrays:
        return np.empty((0, 3))
    return np.vstack(arrays)


def _trajectory_segments(
    centers: np.ndarray,
    *,
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    boundary_mode: str | None = None,
) -> list[np.ndarray]:
    arr = np.asarray(centers, dtype=float)
    if len(arr) < 2:
        return []
    size = _box_size_array(box_size)
    if boundary_mode != "periodic" or size is None:
        return [arr]

    jumps = np.any(np.abs(np.diff(arr, axis=0)) > 0.5 * size[None, :], axis=1)
    segments = []
    start = 0
    for index, jump in enumerate(jumps):
        if jump:
            if index + 1 - start >= 2:
                segments.append(arr[start : index + 1])
            start = index + 1
    if len(arr) - start >= 2:
        segments.append(arr[start:])
    return segments


def plot_agglomeration_trajectories_pyvista(
    h5_path: str | Path,
    out_path: str | Path | None = None,
    *,
    primary_ids: list[int] | None = None,
    every: int = 1,
    max_frames: int | None = None,
    boundary_mode: str | None = None,
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    title: str = "Primary-particle trajectories",
    window_size: tuple[int, int] = (1200, 900),
    tube_radius: float | None = None,
    sphere_theta_resolution: int = 48,
    sphere_phi_resolution: int = 24,
    background: str = "white",
) -> np.ndarray:
    """Render primary-particle trajectory traces with PyVista/VTK."""

    pv = _pyvista_module()
    snapshots = load_agglomeration_snapshots(h5_path, every=every, max_frames=max_frames)
    if box_size is None:
        box_size = h5_box_size(h5_path)
    if boundary_mode is None:
        boundary_mode = h5_boundary_mode(h5_path)
    tracks, final_snapshot = _trajectory_tracks_from_snapshots(snapshots, primary_ids)
    points_for_limits = _trajectory_points(tracks)
    if tube_radius is None:
        size = _box_size_array(box_size)
        if size is not None:
            span = float(np.max(size))
        elif len(points_for_limits):
            span = max(float(np.max(np.ptp(points_for_limits, axis=0))), 1.0e-30)
        else:
            span = 1.0
        tube_radius = 0.0035 * span

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    sorted_tracks = sorted(tracks.items())
    track_ids = [pid for pid, _centers in sorted_tracks]
    track_colors = _primary_colors_from_final_snapshot(track_ids, final_snapshot)
    for color, (_pid, centers) in zip(track_colors, sorted_tracks, strict=True):
        arr = np.asarray(centers, dtype=float)
        for segment in _trajectory_segments(arr, box_size=box_size, boundary_mode=boundary_mode):
            line = pv.PolyData(segment)
            line.lines = np.concatenate(([len(segment)], np.arange(len(segment), dtype=np.int64)))
            plotter.add_mesh(
                line.tube(radius=float(tube_radius), n_sides=12),
                color=color,
                smooth_shading=True,
                opacity=0.85,
            )

    if final_snapshot:
        points, radii, labels = _snapshot_points(final_snapshot)
        colors = _snapshot_colors(final_snapshot, labels) if len(points) else []
        for center, radius, color in zip(points, radii, colors, strict=True):
            sphere = pv.Sphere(
                radius=float(radius),
                center=tuple(float(v) for v in center),
                theta_resolution=sphere_theta_resolution,
                phi_resolution=sphere_phi_resolution,
            )
            plotter.add_mesh(
                sphere,
                color=color,
                smooth_shading=True,
                specular=0.35,
                specular_power=18.0,
                ambient=0.25,
            )

    _pyvista_add_box(plotter, pv, box_size)
    label_points = points_for_limits
    if final_snapshot:
        final_points, _radii, _labels = _snapshot_points(final_snapshot)
        if len(final_points):
            label_points = np.vstack([label_points, final_points]) if len(label_points) else final_points
    _pyvista_add_axes_labels(plotter, pv, box_size, label_points)
    plotter.add_text(title, position="upper_left", font_size=10, color="black")
    plotter.add_light(pv.Light(light_type="headlight", intensity=0.45))
    _pyvista_set_camera(plotter, box_size)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, image)
    return image


def plot_agglomeration_trajectories(
    h5_path: str | Path,
    out_path: str | Path | None = None,
    *,
    primary_ids: list[int] | None = None,
    every: int = 1,
    max_frames: int | None = None,
    boundary_mode: str | None = None,
    backend: str = "matplotlib",
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    elev: float = 30.0,
    azim: float = 30.0,
    dpi: int = 160,
):
    """Plot primary-particle trajectories from saved agglomeration snapshots."""

    if backend == "pyvista":
        return plot_agglomeration_trajectories_pyvista(
            h5_path,
            out_path,
            primary_ids=primary_ids,
            every=every,
            max_frames=max_frames,
            boundary_mode=boundary_mode,
            box_size=box_size,
        )
    if backend != "matplotlib":
        raise ValueError("backend must be 'matplotlib' or 'pyvista'")

    plt = _pyplot()
    snapshots = load_agglomeration_snapshots(h5_path, every=every, max_frames=max_frames)
    if box_size is None:
        box_size = h5_box_size(h5_path)
    if boundary_mode is None:
        boundary_mode = h5_boundary_mode(h5_path)
    tracks, final_snapshot = _trajectory_tracks_from_snapshots(snapshots, primary_ids)

    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    sorted_tracks = sorted(tracks.items())
    track_ids = [pid for pid, _centers in sorted_tracks]
    track_colors = _primary_colors_from_final_snapshot(track_ids, final_snapshot)
    for color, (pid, centers) in zip(track_colors, sorted_tracks, strict=True):
        arr = np.asarray(centers, dtype=float)
        labeled = False
        for segment in _trajectory_segments(arr, box_size=box_size, boundary_mode=boundary_mode):
            label = str(pid) if not labeled else None
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                linewidth=0.9,
                color=color,
                label=label,
            )
            labeled = True
    if final_snapshot:
        points, radii, labels = _snapshot_points(final_snapshot)
        if len(points):
            colors = _snapshot_colors(final_snapshot, labels)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=80, c=colors, alpha=0.8)
    points_for_limits = _trajectory_points(tracks)
    _draw_box(ax, box_size)
    _set_equal_axes(ax, points_for_limits, box_size=box_size)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Primary-particle trajectories")
    if len(tracks) <= 12:
        ax.legend(title="primary", fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _apply_margins(fig)
    if out_path is not None:
        fig.savefig(out_path, dpi=dpi)
    return fig, ax


def save_agglomeration_snapshots(
    h5_path: str | Path,
    out_dir: str | Path,
    *,
    every: int = 1,
    max_frames: int | None = None,
    prefix: str = "snapshot",
    backend: str = "matplotlib",
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    elev: float = 30.0,
    azim: float = 30.0,
    dpi: int = 160,
    sphere_resolution: int = 16,
) -> list[Path]:
    """Save PNG snapshot frames from a coagulation HDF5 run."""

    if backend == "pyvista":
        return save_agglomeration_snapshots_pyvista(
            h5_path,
            out_dir,
            every=every,
            max_frames=max_frames,
            prefix=prefix,
            box_size=box_size,
            sphere_theta_resolution=max(16, 2 * sphere_resolution),
            sphere_phi_resolution=max(8, sphere_resolution),
        )
    if backend != "matplotlib":
        raise ValueError("backend must be 'matplotlib' or 'pyvista'")

    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshots = load_agglomeration_snapshots(h5_path, every=every, max_frames=max_frames)
    if box_size is None:
        box_size = h5_box_size(h5_path)
    paths = []
    for frame, snap in enumerate(snapshots):
        path = out / f"{prefix}_{frame:04d}.png"
        fig, _ax = plot_agglomeration_snapshot(
            snap,
            path,
            box_size=box_size,
            elev=elev,
            azim=azim,
            dpi=dpi,
            sphere_resolution=sphere_resolution,
        )
        plt.close(fig)
        paths.append(path)
    return paths


def save_agglomeration_snapshots_pyvista(
    h5_path: str | Path,
    out_dir: str | Path,
    *,
    every: int = 1,
    max_frames: int | None = None,
    prefix: str = "snapshot",
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    sphere_theta_resolution: int = 48,
    sphere_phi_resolution: int = 24,
) -> list[Path]:
    """Save PNG snapshot frames with PyVista true-radius sphere rendering."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshots = load_agglomeration_snapshots(h5_path, every=every, max_frames=max_frames)
    if box_size is None:
        box_size = h5_box_size(h5_path)
    paths = []
    for frame, snap in enumerate(snapshots):
        path = out / f"{prefix}_{frame:04d}.png"
        plot_agglomeration_snapshot_pyvista(
            snap,
            path,
            box_size=box_size,
            title=f"t = {snap.get('time', np.nan):.3e} s",
            sphere_theta_resolution=sphere_theta_resolution,
            sphere_phi_resolution=sphere_phi_resolution,
        )
        paths.append(path)
    return paths


def save_agglomeration_video(
    h5_path: str | Path,
    out_path: str | Path,
    *,
    every: int = 1,
    max_frames: int | None = None,
    fps: int = 8,
    backend: str = "matplotlib",
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    elev: float = 30.0,
    azim: float = 30.0,
    dpi: int = 140,
    sphere_resolution: int = 12,
) -> Path:
    """Save an MP4 or GIF animation from saved agglomeration snapshots.

    If an MP4 is requested but ffmpeg is unavailable, the function falls back
    to a GIF next to the requested path.
    """

    if backend == "pyvista":
        return save_agglomeration_video_pyvista(
            h5_path,
            out_path,
            every=every,
            max_frames=max_frames,
            fps=fps,
            box_size=box_size,
            sphere_theta_resolution=max(16, 2 * sphere_resolution),
            sphere_phi_resolution=max(8, sphere_resolution),
        )
    if backend != "matplotlib":
        raise ValueError("backend must be 'matplotlib' or 'pyvista'")

    plt = _pyplot()
    from matplotlib import animation

    snapshots = load_agglomeration_snapshots(h5_path, every=every, max_frames=max_frames)
    if not snapshots:
        raise ValueError(f"no snapshots found in {h5_path}")
    if box_size is None:
        box_size = h5_box_size(h5_path)

    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index: int):
        ax.clear()
        plot_agglomeration_snapshot(
            snapshots[frame_index],
            ax=ax,
            box_size=box_size,
            elev=elev,
            azim=azim,
            show_box=True,
            sphere_resolution=sphere_resolution,
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / fps)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
        actual = out
    elif animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps)
        actual = out
    else:
        writer = animation.PillowWriter(fps=fps)
        actual = out.with_suffix(".gif")
    anim.save(actual, writer=writer, dpi=dpi)
    plt.close(fig)
    return actual


def save_agglomeration_video_pyvista(
    h5_path: str | Path,
    out_path: str | Path,
    *,
    every: int = 1,
    max_frames: int | None = None,
    fps: int = 8,
    box_size: float | tuple[float, float, float] | np.ndarray | None = None,
    sphere_theta_resolution: int = 32,
    sphere_phi_resolution: int = 16,
) -> Path:
    """Save an MP4/GIF animation using PyVista true-radius sphere rendering."""

    snapshots = load_agglomeration_snapshots(h5_path, every=every, max_frames=max_frames)
    if not snapshots:
        raise ValueError(f"no snapshots found in {h5_path}")
    if box_size is None:
        box_size = h5_box_size(h5_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    actual = out
    if actual.suffix.lower() not in {".gif", ".mp4"}:
        actual = actual.with_suffix(".mp4")
    writer = imageio.get_writer(actual, fps=fps)
    try:
        for snap in snapshots:
            image = _pyvista_snapshot_image(
                snap,
                box_size=box_size,
                title=f"t = {snap.get('time', np.nan):.3e} s",
                sphere_theta_resolution=sphere_theta_resolution,
                sphere_phi_resolution=sphere_phi_resolution,
                window_size=(960, 720),
            )
            writer.append_data(image)
    finally:
        writer.close()
    return actual


def plot_settling_outputs(
    settling_times: dict[float, np.ndarray],
    summary_rows: list[dict],
    out_dir: str | Path,
) -> None:
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(settling_times), 1, figsize=(6, 2.6 * len(settling_times)))
    if len(settling_times) == 1:
        axes = [axes]
    for ax, (diameter_nm, times) in zip(axes, settling_times.items(), strict=True):
        times = np.asarray(times, dtype=float)
        times = times[np.isfinite(times) & (times > 0.0)]
        if len(times):
            ax.hist(np.log(times), bins=min(12, max(3, len(times))))
        ax.set_title(f"{diameter_nm:g} nm")
        ax.set_xlabel("log(settling time / s)")
        ax.set_ylabel("count")
    _apply_margins(fig)
    fig.savefig(out / "settling_histograms.png", dpi=160)
    plt.close(fig)

    rows = sorted(summary_rows, key=lambda row: row["diameter_nm"])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.loglog([row["diameter_nm"] / 1000.0 for row in rows], [row["mean_time_s"] for row in rows], "o-")
    ax.set_xlabel("d_p (um)")
    ax.set_ylabel("mean settling time (s)")
    ax.set_title("Settling time vs particle size")
    _apply_margins(fig)
    fig.savefig(out / "settling_mean_times.png", dpi=160)
    plt.close(fig)


def plot_cluster_stats_csv(csv_path: str | Path, out_dir: str | Path) -> None:
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(df["time"], df["cluster_count"], label="clusters")
    if "largest_cluster" in df:
        ax.plot(df["time"], df["largest_cluster"], label="largest")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.legend()
    _apply_margins(fig)
    fig.savefig(out / "cluster_stats.png", dpi=160)
    plt.close(fig)


def plot_final_aggregate_csv(csv_path: str | Path, out_dir: str | Path) -> None:
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    sizes = (df["radius"].to_numpy() / df["radius"].max()) ** 2 * 120
    color = MONOMER_COLOR if len(df) == 1 else AGGLOMERATE_COLOR
    ax.scatter(df["x"], df["y"], df["z"], s=sizes, c=color, edgecolors="0.25", linewidths=0.35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Final aggregate")
    _apply_margins(fig)
    fig.savefig(out / "final_aggregate.png", dpi=160)
    plt.close(fig)


def plot_h5_summary(h5_path: str | Path, out_dir: str | Path) -> None:
    """Plot available HDF5 summary datasets."""

    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        if "cluster_stats" not in h5:
            return
        stats = h5["cluster_stats"][:]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(stats["time"], stats["cluster_count"], label="clusters")
    ax.plot(stats["time"], stats["largest_cluster"], label="largest")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.legend()
    _apply_margins(fig)
    fig.savefig(out / "h5_cluster_stats.png", dpi=160)
    plt.close(fig)
