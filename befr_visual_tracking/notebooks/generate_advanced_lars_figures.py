#!/usr/bin/env python3
"""Plotly figures for the advanced calibration slides."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras
from befr_visual_tracking.advanced.initial_pose_estimation import (
    estimate_initial_camera_poses,
    relative_pose_estimates_to_cameras,
)
from befr_visual_tracking.advanced.observations import (
    group_measurements_by_timestamp,
    load_visible_measurements,
)
from befr_visual_tracking.camera_model import Camera, cameras_from_yaml
from befr_visual_tracking.experiments.replay import load_ground_truth_for_replay
from befr_visual_tracking.result_writer import load_metadata_json
from befr_visual_tracking.triangulation import PixelObservation, triangulate_from_observations

# Reuse Lars notebook helpers by executing the notebook's helper cells once.
NOTEBOOK_PATH = REPO_ROOT / "befr_visual_tracking" / "notebooks" / "final_vis.ipynb"
OUT_DIR = REPO_ROOT / "befr_visual_tracking" / "notebooks" / "final_vis_figures" / "calibration_figure_eight_3d_advanced"
DATA_DIR = REPO_ROOT / "results" / "calibration_figure_eight_3d"
TARGET_TIME_S = 7.30
ACTIVE = ["camera_0", "camera_1", "camera_2"]

CAMERA_COLORS = {
    "camera_0": "#f75ab0",
    "camera_1": "#ae3ae8",
    "camera_2": "#f68665",
}
TRUE_COLOR = "#20c05c"
TRIANGULATION_POINT_COLOR = "#b45309"
DETECTION_COLOR = "red"


def load_lars_helpers() -> dict:
    import json

    nb = json.loads(NOTEBOOK_PATH.read_text())
    namespace: dict = {"np": np, "go": go, "Path": Path}
    for idx in [1, 3, 5, 6, 7]:
        exec("".join(nb["cells"][idx]["source"]), namespace)
    namespace["REPO_ROOT"] = REPO_ROOT
    exec("".join(nb["cells"][3]["source"]), namespace)
    return namespace


def camera_list_by_id(cameras: dict[str, Camera], active: list[str]) -> list[Camera]:
    return [cameras[camera_id] for camera_id in active]


def add_camera_geometry_custom(fig: go.Figure, cameras: list[Camera], helpers: dict, *, dashed: bool = False) -> None:
    for camera in cameras:
        color = CAMERA_COLORS.get(camera.name, helpers["DEFAULT_CAMERA_COLOR"])
        if dashed:
            origin = camera.extrinsics.t_camera_to_world
            corners = helpers["image_plane_corners"](camera, helpers["FRUSTUM_DEPTH_M"])
            fig.add_trace(
                helpers["line_trace"](
                    [*corners, corners[0]],
                    color=color,
                    width=4,
                    name=f"{camera.name} initial",
                    opacity=0.85,
                )
            )
            frustum_lines: list[np.ndarray | None] = []
            for corner in corners:
                frustum_lines.extend([origin, corner, None])
            fig.add_trace(
                helpers["line_trace"](
                    frustum_lines,
                    color=color,
                    width=2,
                    name=f"{camera.name} initial frustum",
                    opacity=0.85,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0]],
                    y=[origin[1]],
                    z=[origin[2]],
                    mode="markers",
                    marker={"size": 5, "color": color, "symbol": "circle-open"},
                    name=f"{camera.name} initial",
                )
            )
        else:
            helpers["add_camera_frustum"](fig, camera, color, helpers["FRUSTUM_DEPTH_M"])
            if helpers["SHOW_PIXEL_GRID"]:
                helpers["add_pixel_grid"](fig, camera, helpers["FRUSTUM_DEPTH_M"], helpers["GRID_STEP_PX"], color)


def scene_axis_ranges_from_cameras(cameras: list[Camera], points: list[np.ndarray], helpers: dict) -> dict[str, list[float]]:
    all_points: list[np.ndarray] = []
    for camera in cameras:
        all_points.append(camera.extrinsics.t_camera_to_world)
        all_points.extend(helpers["image_plane_corners"](camera, helpers["FRUSTUM_DEPTH_M"]))
    all_points.extend(points)
    arr = np.asarray(all_points, dtype=float)
    padding = float(helpers["SCENE_AXIS_PADDING_M"])
    low = np.min(arr, axis=0) - padding
    high = np.max(arr, axis=0) + padding
    return {
        "x": [float(low[0]), float(high[0])],
        "y": [float(low[1]), float(high[1])],
        "z": [float(low[2]), float(high[2])],
    }


def load_scene(helpers: dict):
    metadata = load_metadata_json(DATA_DIR / "metadata.json")
    _, true_cameras = cameras_from_yaml(DATA_DIR / "camera_config.yaml")
    true_by_id = apply_gauge_fix_to_cameras({camera.name: camera for camera in true_cameras})
    true_by_id = {camera_id: true_by_id[camera_id] for camera_id in ACTIVE}

    measurements = load_visible_measurements(str(DATA_DIR / "detections.csv"))
    grouped = group_measurements_by_timestamp(measurements)
    timestamp = min(grouped.keys(), key=lambda ts: abs(ts - TARGET_TIME_S))
    gt_by_time = load_ground_truth_for_replay(DATA_DIR)
    gt_pos, gt_vel = gt_by_time[timestamp]

    pixel_obs = [
        PixelObservation(measurement.camera_id, measurement.u, measurement.v)
        for measurement in grouped[timestamp].values()
        if measurement.camera_id in ACTIVE
    ]

    initial_poses = estimate_initial_camera_poses(
        true_by_id,
        grouped,
        active_cameras=ACTIVE,
        use_opencv=False,
        translation_scale=1.0,
    )
    initial_by_id = relative_pose_estimates_to_cameras(initial_poses, true_by_id)

    _, est_cameras = cameras_from_yaml(DATA_DIR / "calibrated_cameras.yaml")
    est_by_id = {camera.name: camera for camera in est_cameras if camera.name in ACTIVE}

    tri = triangulate_from_observations(
        true_by_id,
        pixel_obs,
        estimate_covariance=True,
        pixel_noise_std=float(metadata.get("pixel_noise_std", 1.0)),
        monte_carlo_samples=40,
    )

    helpers["SCENE_AXIS_RANGES"] = scene_axis_ranges_from_cameras(
        camera_list_by_id(true_by_id, ACTIVE),
        [gt_pos, tri.point if tri.point is not None else gt_pos],
        helpers,
    )

    return {
        "timestamp": timestamp,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "true_by_id": true_by_id,
        "initial_by_id": initial_by_id,
        "est_by_id": est_by_id,
        "pixel_obs": pixel_obs,
        "tri": tri,
        "metadata": metadata,
    }


def export_figure(fig: go.Figure, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return path


def figure_unknown_setup(scene: dict, helpers: dict) -> go.Figure:
    fig = go.Figure()
    add_camera_geometry_custom(fig, [scene["true_by_id"]["camera_0"]], helpers)
    add_camera_geometry_custom(
        fig,
        [scene["initial_by_id"][camera_id] for camera_id in ["camera_1", "camera_2"]],
        helpers,
        dashed=True,
    )
    helpers["add_state_marker"](
        fig,
        scene["gt_pos"],
        scene["gt_vel"],
        color=TRUE_COLOR,
        name="ground truth",
        symbol="diamond",
        size=9,
        show_velocity=True,
    )
    helpers["apply_layout"](fig, "Advanced setup — Camera 0 known, others unknown", scene["timestamp"])
    return fig


def figure_calibration_target(scene: dict, helpers: dict) -> go.Figure:
    fig = go.Figure()
    add_camera_geometry_custom(fig, camera_list_by_id(scene["true_by_id"], ACTIVE), helpers)
    helpers["add_state_marker"](
        fig,
        scene["gt_pos"],
        scene["gt_vel"],
        color=TRUE_COLOR,
        name="ground truth",
        symbol="diamond",
        size=9,
        show_velocity=True,
    )
    helpers["apply_layout"](fig, "Calibration target state", scene["timestamp"])
    return fig


def figure_detections(scene: dict, helpers: dict) -> go.Figure:
    fig = go.Figure()
    add_camera_geometry_custom(fig, camera_list_by_id(scene["true_by_id"], ACTIVE), helpers)
    for obs in scene["pixel_obs"]:
        camera = scene["true_by_id"][obs.camera_id]
        helpers["add_image_marker"](
            fig,
            camera,
            (obs.u, obs.v),
            color="limegreen",
            name=f"{camera.name} detection",
            size=4,
        )
        helpers["add_pixel_covariance_ellipses"](
            fig,
            camera,
            (obs.u, obs.v),
            camera.noise_model.measurement_noise_covariance,
            helpers["MEASUREMENT_SIGMA_LEVELS"],
            color=DETECTION_COLOR,
            name_prefix="measurement covariance",
        )
    helpers["add_state_marker"](
        fig,
        scene["gt_pos"],
        scene["gt_vel"],
        color=TRUE_COLOR,
        name="ground truth",
        symbol="diamond",
        size=9,
        show_velocity=True,
    )
    helpers["apply_layout"](fig, "Noisy detections during calibration", scene["timestamp"])
    return fig


def figure_triangulation(scene: dict, helpers: dict, *, show_samples: bool = False) -> go.Figure:
    fig = go.Figure()
    add_camera_geometry_custom(fig, camera_list_by_id(scene["true_by_id"], ACTIVE), helpers)
    record = {
        "detections": [],
        "triangulation_result": scene["tri"],
        "triangulation_samples": np.empty((0, 3)),
    }
    for obs in scene["pixel_obs"]:
        camera = scene["true_by_id"][obs.camera_id]
        origin, direction = camera.pixel_to_world_ray(obs.u, obs.v)
        endpoint = origin + helpers["TRIANGULATION_DETECTION_RAY_LENGTH_M"] * direction
        fig.add_trace(
            helpers["line_trace"](
                [origin, endpoint],
                color=helpers["KF_COLOR"],
                width=4,
                name=f"{camera.name} triangulation ray",
            )
        )
    if scene["tri"].point is not None:
        point = scene["tri"].point
        fig.add_trace(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers",
                marker={"size": 8, "color": TRIANGULATION_POINT_COLOR},
                name="triangulation point",
            )
        )
    helpers["add_state_marker"](
        fig,
        scene["gt_pos"],
        np.zeros(3),
        color=TRUE_COLOR,
        name="ground truth",
        symbol="diamond",
        size=9,
        show_velocity=False,
    )
    title = "Triangulation rays during calibration"
    helpers["apply_layout"](fig, title, scene["timestamp"])
    return fig


def figure_bundle_adjustment(scene: dict, helpers: dict) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("before BA", "after BA"),
    )

    def draw_panel(col: int, cameras: dict[str, Camera], title_prefix: str) -> None:
        for camera_id in ACTIVE:
            camera = cameras[camera_id]
            color = CAMERA_COLORS[camera_id]
            origin = camera.extrinsics.t_camera_to_world
            corners = helpers["image_plane_corners"](camera, helpers["FRUSTUM_DEPTH_M"])
            fig.add_trace(
                helpers["line_trace"]([*corners, corners[0]], color=color, width=4, name=f"{title_prefix} {camera_id}"),
                row=1,
                col=col,
            )
            frustum_lines: list[np.ndarray | None] = []
            for corner in corners:
                frustum_lines.extend([origin, corner, None])
            fig.add_trace(
                helpers["line_trace"](frustum_lines, color=color, width=2, name=f"{title_prefix} {camera_id} frustum"),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0]],
                    y=[origin[1]],
                    z=[origin[2]],
                    mode="markers",
                    marker={"size": 5, "color": color},
                    name=f"{title_prefix} {camera_id}",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        gt = scene["gt_pos"]
        fig.add_trace(
            go.Scatter3d(
                x=[gt[0]],
                y=[gt[1]],
                z=[gt[2]],
                mode="markers",
                marker={"size": 8, "color": TRUE_COLOR, "symbol": "diamond"},
                name="ground truth",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

    draw_panel(1, scene["initial_by_id"], "initial")
    draw_panel(2, scene["est_by_id"], "calibrated")

    ranges = helpers["SCENE_AXIS_RANGES"]
    for col in (1, 2):
        scene_name = "scene" if col == 1 else "scene2"
        fig.update_layout(
            {
                scene_name: {
                    "xaxis": {"title": "X [m]", "range": ranges["x"]},
                    "yaxis": {"title": "Y [m]", "range": ranges["y"]},
                    "zaxis": {"title": "Z [m]", "range": ranges["z"]},
                    "aspectmode": helpers["PLOT_ASPECTMODE"],
                    "camera": helpers["PLOT_SCENE_CAMERA"],
                }
            }
        )
    fig.update_layout(
        title=f"Bundle adjustment — camera poses (t = {scene['timestamp']:.2f} s)",
        height=helpers["PLOT_HEIGHT"],
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig


def main() -> None:
    helpers = load_lars_helpers()
    scene = load_scene(helpers)

    export_figure(figure_unknown_setup(scene, helpers), "01_advanced_unknown_setup.html")
    export_figure(figure_calibration_target(scene, helpers), "02_calibration_target_state.html")
    export_figure(figure_detections(scene, helpers), "03_noisy_detections.html")
    export_figure(figure_triangulation(scene, helpers), "04_triangulation_rays.html")
    export_figure(figure_bundle_adjustment(scene, helpers), "05_bundle_adjustment_before_after.html")

    print(f"\nOpen figures in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
