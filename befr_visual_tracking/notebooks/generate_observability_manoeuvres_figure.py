#!/usr/bin/env python3
"""Observability slide: compare calibration manoeuvres (Exp 101)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras
from befr_visual_tracking.camera_model import Camera, cameras_from_yaml

NOTEBOOK_PATH = REPO_ROOT / "befr_visual_tracking" / "notebooks" / "final_vis.ipynb"
OUT_DIR = REPO_ROOT / "befr_visual_tracking" / "notebooks" / "final_vis_figures" / "observability_manoeuvres"
ACTIVE = ["camera_0", "camera_1", "camera_2"]

CAMERA_COLORS = {
    "camera_0": "#f75ab0",
    "camera_1": "#ae3ae8",
    "camera_2": "#f68665",
}

MANOEUVRES = [
    {
        "panel": "A",
        "folder": "calibration_stationary_hover",
        "label": "Stationary hover",
        "tag": "No motion",
        "observability": "Poor — no parallax",
        "trajectory_color": "#ef4444",
        "calibration_exp": "experiment_008",
    },
    {
        "panel": "B",
        "folder": "calibration_straight_line",
        "label": "Straight line",
        "tag": "1D motion",
        "observability": "Limited depth cues",
        "trajectory_color": "#f59e0b",
        "calibration_exp": "experiment_010",
    },
    {
        "panel": "C",
        "folder": "calibration_planar_figure_eight",
        "label": "Planar figure-eight",
        "tag": "2D motion (XY)",
        "observability": "Better — lateral parallax",
        "trajectory_color": "#3b82f6",
        "calibration_exp": "experiment_012",
    },
    {
        "panel": "D",
        "folder": "calibration_figure_eight_3d",
        "label": "3D figure-eight",
        "tag": "Full 3D motion",
        "observability": "Best — rich parallax",
        "trajectory_color": "#20c05c",
        "calibration_exp": "experiment_014",
    },
]


def load_lars_helpers() -> dict:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    namespace: dict = {"np": np, "go": go, "Path": Path, "REPO_ROOT": REPO_ROOT}
    for idx in [1, 3, 5, 6, 7]:
        exec("".join(nb["cells"][idx]["source"]), namespace)
    exec("".join(nb["cells"][3]["source"]), namespace)
    return namespace


def load_trajectory(results_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(results_dir / "ground_truth.csv")
    pos = df[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
    vel = df[["gt_vx", "gt_vy", "gt_vz"]].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=float)
    return ts, pos, vel


def load_cameras(results_dir: Path) -> list[Camera]:
    _, cameras = cameras_from_yaml(results_dir / "camera_config.yaml")
    by_id = apply_gauge_fix_to_cameras({camera.name: camera for camera in cameras})
    return [by_id[camera_id] for camera_id in ACTIVE]


def ba_reprojection_px(calibration_exp: str) -> float | None:
    report_path = REPO_ROOT / "results" / calibration_exp / "calibration_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    return float(report["bundle_adjustment"]["reprojection_rmse"])


def trajectory_stats(pos: np.ndarray) -> dict[str, float]:
    span = np.max(pos, axis=0) - np.min(pos, axis=0)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return {
        "span_x": float(span[0]),
        "span_y": float(span[1]),
        "span_z": float(span[2]),
        "path_length": float(np.sum(speed)),
        "max_step": float(np.max(speed)) if len(speed) else 0.0,
    }


def add_cameras_to_subplot(fig: go.Figure, cameras: list[Camera], helpers: dict, row: int, col: int) -> None:
    for camera in cameras:
        color = CAMERA_COLORS.get(camera.name, helpers["DEFAULT_CAMERA_COLOR"])
        origin = camera.extrinsics.t_camera_to_world
        corners = helpers["image_plane_corners"](camera, helpers["FRUSTUM_DEPTH_M"])
        frustum_lines: list[np.ndarray | None] = []
        for corner in corners:
            frustum_lines.extend([origin, corner, None])
        fig.add_trace(
            helpers["line_trace"](frustum_lines, color=color, width=1.5, name=camera.name, opacity=0.55),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                mode="markers",
                marker={"size": 3, "color": color},
                name=camera.name,
                showlegend=False,
                hovertemplate=f"{camera.name}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def add_velocity_arrow_subplot(
    fig: go.Figure,
    origin: np.ndarray,
    velocity: np.ndarray,
    *,
    color: str,
    row: int,
    col: int,
    time_horizon_s: float = 0.5,
) -> None:
    origin = np.asarray(origin, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    arrow = float(time_horizon_s) * velocity
    end = origin + arrow
    if float(np.linalg.norm(arrow)) < 1e-6:
        return
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode="lines",
            line={"color": color, "width": 6},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def add_trajectory_to_subplot(
    fig: go.Figure,
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    color: str,
    name: str,
    row: int,
    col: int,
) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines+markers" if len(pos) < 4 else "lines",
            line={"color": color, "width": 7},
            marker={"size": 4 if len(pos) < 4 else 0, "color": color},
            name=name,
            showlegend=False,
            hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>",
        ),
        row=row,
        col=col,
    )
    mid = len(pos) // 2
    add_velocity_arrow_subplot(fig, pos[mid], vel[mid], color=color, row=row, col=col)


def scene_ranges(all_positions: list[np.ndarray], cameras: list[Camera], helpers: dict) -> dict[str, list[float]]:
    points: list[np.ndarray] = []
    for pos in all_positions:
        points.extend(pos)
    for camera in cameras:
        points.append(camera.extrinsics.t_camera_to_world)
        points.extend(helpers["image_plane_corners"](camera, helpers["FRUSTUM_DEPTH_M"]))
    arr = np.asarray(points, dtype=float)
    pad = float(helpers["SCENE_AXIS_PADDING_M"])
    low = np.min(arr, axis=0) - pad
    high = np.max(arr, axis=0) + pad
    return {
        "x": [float(low[0]), float(high[0])],
        "y": [float(low[1]), float(high[1])],
        "z": [float(low[2]), float(high[2])],
    }


def build_figure(helpers: dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}], [{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[f"{m['panel']}. {m['label']}" for m in MANOEUVRES],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    reference_cameras = load_cameras(REPO_ROOT / "results" / MANOEUVRES[0]["folder"])
    all_positions: list[np.ndarray] = []
    annotations: list[str] = []

    for index, manoeuvre in enumerate(MANOEUVRES):
        row = index // 2 + 1
        col = index % 2 + 1
        results_dir = REPO_ROOT / "results" / manoeuvre["folder"]
        _, pos, vel = load_trajectory(results_dir)
        all_positions.append(pos)
        stats = trajectory_stats(pos)
        reproj = ba_reprojection_px(manoeuvre["calibration_exp"])

        add_cameras_to_subplot(fig, reference_cameras, helpers, row, col)
        add_trajectory_to_subplot(
            fig,
            pos,
            vel,
            color=manoeuvre["trajectory_color"],
            name=manoeuvre["label"],
            row=row,
            col=col,
        )

        reproj_text = f"{reproj:.2f} px BA reproj." if reproj is not None else "BA n/a"
        annotations.append(
            f"{manoeuvre['tag']} | Δz={stats['span_z']:.2f} m | "
            f"path={stats['path_length']:.1f} m | {reproj_text}<br>"
            f"<span style='font-size:11px'>{manoeuvre['observability']}</span>"
        )

    ranges = scene_ranges(all_positions, reference_cameras, helpers)
    scene_names = ["scene", "scene2", "scene3", "scene4"]
    camera_view = helpers["PLOT_SCENE_CAMERA"]
    for scene_name in scene_names:
        fig.update_layout(
            {
                scene_name: {
                    "xaxis": {"title": "X [m]", "range": ranges["x"], "autorange": False},
                    "yaxis": {"title": "Y [m]", "range": ranges["y"], "autorange": False},
                    "zaxis": {"title": "Z [m]", "range": ranges["z"], "autorange": False},
                    "aspectmode": helpers["PLOT_ASPECTMODE"],
                    "camera": camera_view,
                }
            }
        )

    fig.update_layout(
        title=(
            "Observability — calibration manoeuvres (Exp 101)<br>"
            "<sup>Motion creates parallax; richer 3D paths improve camera calibration</sup>"
        ),
        height=920,
        margin={"l": 0, "r": 0, "t": 110, "b": 0},
        paper_bgcolor="#eef3f8",
        showlegend=False,
    )

    # Panel footnotes
    x_positions = [0.23, 0.77, 0.23, 0.77]
    y_positions = [0.52, 0.52, 0.02, 0.02]
    for annotation, x, y in zip(annotations, x_positions, y_positions, strict=True):
        fig.add_annotation(
            text=annotation,
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font={"size": 11},
        )

    return fig


def main() -> None:
    helpers = load_lars_helpers()
    fig = build_figure(helpers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    html_path = OUT_DIR / "observability_manoeuvres_2x2.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"saved {html_path.relative_to(REPO_ROOT)}")

    png_path = OUT_DIR / "observability_manoeuvres_2x2.png"
    try:
        fig.write_image(png_path, width=1600, height=920, scale=2)
        print(f"saved {png_path.relative_to(REPO_ROOT)}")
    except Exception as exc:
        print(f"PNG export skipped ({exc}). Open HTML and use Plotly download button.")


if __name__ == "__main__":
    main()
