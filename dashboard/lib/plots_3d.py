"""Plotly 3D scene helpers for the dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from befr_visual_tracking.camera_model import Camera
from befr_visual_tracking.experiments.metrics import StepRecord


def _camera_frustum_lines(camera: Camera, depth: float = 2.0) -> tuple[list, list, list]:
    corners = [
        camera.pixel_to_world_ray(0.0, 0.0),
        camera.pixel_to_world_ray(camera.intrinsics.width_px, 0.0),
        camera.pixel_to_world_ray(camera.intrinsics.width_px, camera.intrinsics.height_px),
        camera.pixel_to_world_ray(0.0, camera.intrinsics.height_px),
    ]
    xs, ys, zs = [], [], []
    origin = camera.extrinsics.t_camera_to_world
    for _origin, direction in corners:
        end = origin + depth * direction[1]
        xs.extend([origin[0], end[0], None])
        ys.extend([origin[1], end[1], None])
        zs.extend([origin[2], end[2], None])
    return xs, ys, zs


def _covariance_ellipsoid(center: np.ndarray, covariance: np.ndarray, scale: float = 2.0):
    position_cov = covariance[:3, :3]
    eigvals, eigvecs = np.linalg.eigh(position_cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 10)
    xs, ys, zs = [], [], []
    for ui in u:
        for vi in v:
            point = np.array(
                [
                    np.cos(ui) * np.sin(vi),
                    np.sin(ui) * np.sin(vi),
                    np.cos(vi),
                ],
                dtype=float,
            )
            scaled = eigvecs @ (np.sqrt(eigvals) * scale * point)
            xs.append(center[0] + scaled[0])
            ys.append(center[1] + scaled[1])
            zs.append(center[2] + scaled[2])
    return xs, ys, zs


def build_scene_figure(
    *,
    cameras_by_id: dict[str, Camera],
    ground_truth_positions: np.ndarray,
    estimated_positions: np.ndarray | None = None,
    current_step: StepRecord | None = None,
    active_camera_ids: list[str] | None = None,
    detections: dict[str, tuple[float, float]] | None = None,
) -> go.Figure:
    fig = go.Figure()
    active_ids = set(active_camera_ids or cameras_by_id.keys())

    for camera_id, camera in cameras_by_id.items():
        origin = camera.extrinsics.t_camera_to_world
        color = "royalblue" if camera_id in active_ids else "lightgray"
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                mode="markers+text",
                marker={"size": 4, "color": color},
                text=[camera_id],
                name=camera_id,
            )
        )
        fx, fy, fz = _camera_frustum_lines(camera)
        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="lines",
                line={"color": color, "width": 2},
                showlegend=False,
            )
        )
        z_axis = origin + camera.extrinsics.R_camera_to_world @ np.array([0.0, 0.0, 0.8])
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], z_axis[0]],
                y=[origin[1], z_axis[1]],
                z=[origin[2], z_axis[2]],
                mode="lines",
                line={"color": color, "width": 4},
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=ground_truth_positions[:, 0],
            y=ground_truth_positions[:, 1],
            z=ground_truth_positions[:, 2],
            mode="lines",
            line={"color": "green", "width": 5},
            name="Ground truth",
        )
    )

    if estimated_positions is not None and len(estimated_positions):
        fig.add_trace(
            go.Scatter3d(
                x=estimated_positions[:, 0],
                y=estimated_positions[:, 1],
                z=estimated_positions[:, 2],
                mode="lines",
                line={"color": "orange", "width": 4},
                name="Estimate",
            )
        )

    if current_step is not None:
        est = current_step.state[:3]
        gt = current_step.ground_truth_position
        fig.add_trace(
            go.Scatter3d(
                x=[est[0], gt[0]],
                y=[est[1], gt[1]],
                z=[est[2], gt[2]],
                mode="lines+markers",
                line={"color": "crimson", "width": 6},
                marker={"size": 5},
                name="Current error",
            )
        )
        ex, ey, ez = _covariance_ellipsoid(est, current_step.covariance)
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="markers",
                marker={"size": 1.5, "color": "orange", "opacity": 0.25},
                name="Uncertainty",
            )
        )

        if detections:
            for camera_id, (u, v) in detections.items():
                camera = cameras_by_id[camera_id]
                origin, direction = camera.pixel_to_world_ray(u, v)
                end = origin + 3.0 * direction
                fig.add_trace(
                    go.Scatter3d(
                        x=[origin[0], end[0]],
                        y=[origin[1], end[1]],
                        z=[origin[2], end[2]],
                        mode="lines",
                        line={"color": "gold", "width": 3},
                        showlegend=False,
                    )
                )

    fig.update_layout(
        scene={
            "xaxis_title": "X [m]",
            "yaxis_title": "Y [m]",
            "zaxis_title": "Z [m]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=650,
    )
    return fig
