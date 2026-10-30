#!/usr/bin/env python3
"""
Generate Task-1-style 3D animation frames for the advanced calibration task.

Style matches camera_model.ipynb / presentation_anim (1).pptx:
  coloured camera frustums, grid image planes, green ground truth,
  detection ellipses, triangulation rays, Monte Carlo samples.

Output: presentation_anim/advanced_frames/*.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from befr_visual_tracking.triangulation import PixelObservation, triangulate_from_observations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_anim" / "advanced_frames"
DATA = ROOT / "results" / "calibration_figure_eight_3d"

CAM_COLORS = {
    "camera_0": "#e377c2",
    "camera_1": "#9467bd",
    "camera_2": "#ff7f0e",
    "camera_3": "#17becf",
}
ACTIVE = ["camera_0", "camera_1", "camera_2"]
T_SCENE = 7.30
FRUSTUM_DEPTH = 2.0
NOISE_SIGMA = 8.0


def set_axes_equal_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def pixel_to_plane_point(camera: Camera, u: float, v: float, depth: float = FRUSTUM_DEPTH) -> np.ndarray:
    cx, cy = camera.intrinsics.principal_point
    x = (u - cx) * depth / camera.intrinsics.fx
    y = (v - cy) * depth / camera.intrinsics.fy
    point_camera = np.array([x, y, depth], dtype=float)
    return camera.extrinsics.t_camera_to_world + camera.extrinsics.R_camera_to_world @ point_camera


def image_plane_corners(camera: Camera, depth: float = FRUSTUM_DEPTH) -> np.ndarray:
    w, h = camera.intrinsics.width_px, camera.intrinsics.height_px
    uv = [(0, 0), (w, 0), (w, h), (0, h)]
    return np.array([pixel_to_plane_point(camera, u, v, depth) for u, v in uv])


def draw_camera_frustum(
    ax,
    camera: Camera,
    *,
    color: str,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: str | None = None,
    grid: bool = True,
) -> None:
    origin = camera.extrinsics.t_camera_to_world
    corners = image_plane_corners(camera)
    closed = np.vstack([corners, corners[0]])

    ax.scatter([origin[0]], [origin[1]], [origin[2]], color=color, s=35, label=label)
    for corner in corners:
        ax.plot(
            [origin[0], corner[0]],
            [origin[1], corner[1]],
            [origin[2], corner[2]],
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=1.0,
        )
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        closed[:, 2],
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=1.4,
    )

    if grid:
        w, h = camera.intrinsics.width_px, camera.intrinsics.height_px
        for u in np.linspace(0, w, 9):
            p0 = pixel_to_plane_point(camera, u, 0)
            p1 = pixel_to_plane_point(camera, u, h)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, alpha=0.25, linewidth=0.6)
        for v in np.linspace(0, h, 7):
            p0 = pixel_to_plane_point(camera, 0, v)
            p1 = pixel_to_plane_point(camera, w, v)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, alpha=0.25, linewidth=0.6)

    # u/v axes on first corner
    p00 = pixel_to_plane_point(camera, 0, 0)
    pu = pixel_to_plane_point(camera, 80, 0)
    pv = pixel_to_plane_point(camera, 0, 60)
    ax.plot([p00[0], pu[0]], [p00[1], pu[1]], [p00[2], pu[2]], color="k", linewidth=1.2)
    ax.plot([p00[0], pv[0]], [p00[1], pv[1]], [p00[2], pv[2]], color="k", linewidth=1.2)


def draw_detection(
    ax,
    camera: Camera,
    u: float,
    v: float,
    *,
    color: str = "#2ca02c",
    sigma: float = NOISE_SIGMA,
) -> None:
    point = pixel_to_plane_point(camera, u, v)
    ax.scatter([point[0]], [point[1]], [point[2]], color=color, s=40, zorder=5)

    # Uncertainty ellipse in image plane, lifted to 3D
    corners = image_plane_corners(camera)
    u_vec = corners[1] - corners[0]
    v_vec = corners[3] - corners[0]
    u_hat = u_vec / np.linalg.norm(u_vec)
    v_hat = v_vec / np.linalg.norm(v_vec)
    theta = np.linspace(0, 2 * np.pi, 40)
    ellipse = []
    for angle in theta:
        offset = sigma * (np.cos(angle) * u_hat + np.sin(angle) * v_hat) * (FRUSTUM_DEPTH / camera.intrinsics.fx)
        ellipse.append(point + offset)
    ellipse = np.array(ellipse)
    ax.plot(ellipse[:, 0], ellipse[:, 1], ellipse[:, 2], color=color, alpha=0.35, linewidth=2)


def draw_gt(ax, position: np.ndarray, velocity: np.ndarray, *, label: str = "ground truth") -> None:
    ax.scatter([position[0]], [position[1]], [position[2]], color="#2ca02c", s=70, marker="D", label=label)
    vel = np.asarray(velocity, dtype=float)
    if np.linalg.norm(vel) > 1e-6:
        vel = 0.35 * vel / np.linalg.norm(vel)
        ax.quiver(
            position[0],
            position[1],
            position[2],
            vel[0],
            vel[1],
            vel[2],
            color="#2ca02c",
            linewidth=2,
            arrow_length_ratio=0.2,
        )


def draw_triangulation_rays(
    ax,
    cameras_by_id: dict[str, Camera],
    observations: list[PixelObservation],
    target: np.ndarray,
    *,
    color: str = "#ff7f0e",
) -> None:
    for obs in observations:
        camera = cameras_by_id[obs.camera_id]
        origin, direction = camera.pixel_to_world_ray(obs.u, obs.v)
        end = origin + 4.5 * direction
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=2,
        )
    ax.scatter([target[0]], [target[1]], [target[2]], color="#8c4a00", s=60, label="triangulation point")


def draw_mc_samples(
    ax,
    samples: np.ndarray,
    mean: np.ndarray,
) -> None:
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color="#ffd699", s=8, alpha=0.5, label="Monte Carlo samples")
    ax.scatter([mean[0]], [mean[1]], [mean[2]], color="#8c4a00", s=70, label="triangulation point")


def new_scene(title: str):
    fig = plt.figure(figsize=(10, 8), facecolor="#eef3f8")
    ax = fig.add_subplot(111, projection="3d", facecolor="#eef3f8")
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, alpha=0.25)
    return fig, ax


def save_frame(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {path.name}")


def load_scene_data():
    _, true_cameras = cameras_from_yaml(DATA / "camera_config.yaml")
    true_by_id = apply_gauge_fix_to_cameras({c.name: c for c in true_cameras})
    true_by_id = {k: true_by_id[k] for k in ACTIVE}

    measurements = load_visible_measurements(DATA / "detections.csv")
    grouped = group_measurements_by_timestamp(measurements)
    timestamps = sorted(grouped.keys())
    timestamp = min(timestamps, key=lambda ts: abs(ts - T_SCENE))

    gt_rows = []
    with (DATA / "ground_truth.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            gt_rows.append(row)
    gt_row = min(gt_rows, key=lambda row: abs(float(row["timestamp"]) - timestamp))
    position = np.array([float(gt_row["gt_x"]), float(gt_row["gt_y"]), float(gt_row["gt_z"])])
    velocity = np.array([float(gt_row["gt_vx"]), float(gt_row["gt_vy"]), float(gt_row["gt_vz"])])

    timestamp = min(grouped.keys(), key=lambda ts: abs(ts - T_SCENE))
    measurements_at_t = grouped[timestamp]
    pixel_obs = [
        PixelObservation(m.camera_id, m.u, m.v)
        for m in measurements_at_t.values()
        if m.camera_id in ACTIVE
    ]

    tri = triangulate_from_observations(
        true_by_id,
        pixel_obs,
        estimate_covariance=True,
        pixel_noise_std=1.0,
        monte_carlo_samples=120,
    )

    initial_poses = estimate_initial_camera_poses(
        true_by_id,
        grouped,
        active_cameras=ACTIVE,
        use_opencv=False,
        translation_scale=1.0,
    )
    initial_by_id = relative_pose_estimates_to_cameras(initial_poses, true_by_id)
    initial_by_id = {k: initial_by_id[k] for k in ACTIVE}

    _, est_cameras = cameras_from_yaml(DATA / "calibrated_cameras.yaml")
    est_by_id = {c.name: c for c in est_cameras if c.name in ACTIVE}

    return {
        "timestamp": timestamp,
        "true_by_id": true_by_id,
        "initial_by_id": initial_by_id,
        "est_by_id": est_by_id,
        "position": position,
        "velocity": velocity,
        "pixel_obs": pixel_obs,
        "tri": tri,
    }


def frame_01_unknown_setup(data) -> None:
    fig, ax = new_scene("1. Advanced setup — Camera 0 known, others unknown")
    draw_camera_frustum(ax, data["true_by_id"]["camera_0"], color=CAM_COLORS["camera_0"], label="camera_0 (known)")
    for cam_id in ["camera_1", "camera_2"]:
        draw_camera_frustum(
            ax,
            data["initial_by_id"][cam_id],
            color=CAM_COLORS[cam_id],
            linestyle="--",
            alpha=0.85,
            label=f"{cam_id} (initial)",
        )
    draw_gt(ax, data["position"], data["velocity"])
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "01_unknown_setup.png")


def frame_02_gt_velocity(data) -> None:
    fig, ax = new_scene(f"2. Calibration target state — t = {T_SCENE:.2f} s")
    for cam_id in ACTIVE:
        draw_camera_frustum(ax, data["true_by_id"][cam_id], color=CAM_COLORS[cam_id], label=cam_id)
    draw_gt(ax, data["position"], data["velocity"])
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "02_calibration_target_state.png")


def frame_03_projection(data) -> None:
    fig, ax = new_scene(f"3. Ideal projection during calibration — t = {T_SCENE:.2f} s")
    for cam_id in ACTIVE:
        cam = data["true_by_id"][cam_id]
        draw_camera_frustum(ax, cam, color=CAM_COLORS[cam_id], label=cam_id)
        uv = cam.project_world_point(data["position"])
        if uv:
            origin, direction = cam.pixel_to_world_ray(*uv)
            end = data["position"]
            ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color="#2ca02c", linewidth=1.5)
            draw_detection(ax, cam, uv[0], uv[1], color="#2ca02c", sigma=0)
    draw_gt(ax, data["position"], data["velocity"])
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "03_ideal_projection.png")


def frame_04_detections(data) -> None:
    fig, ax = new_scene(f"4. Noisy detections during calibration — t = {T_SCENE:.2f} s")
    for cam_id in ACTIVE:
        cam = data["true_by_id"][cam_id]
        draw_camera_frustum(ax, cam, color=CAM_COLORS[cam_id], label=cam_id)
    for obs in data["pixel_obs"]:
        draw_detection(ax, data["true_by_id"][obs.camera_id], obs.u, obs.v)
    draw_gt(ax, data["position"], data["velocity"])
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "04_noisy_detections.png")


def frame_05_triangulation(data) -> None:
    fig, ax = new_scene(f"5. Triangulation rays — t = {T_SCENE:.2f} s")
    for cam_id in ACTIVE:
        draw_camera_frustum(ax, data["true_by_id"][cam_id], color=CAM_COLORS[cam_id], label=cam_id)
    draw_triangulation_rays(ax, data["true_by_id"], data["pixel_obs"], data["tri"].point)
    draw_gt(ax, data["position"], np.zeros(3), label="ground truth")
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "05_triangulation_rays.png")


def frame_06_monte_carlo(data) -> None:
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(120):
        noisy = []
        for obs in data["pixel_obs"]:
            noisy.append(
                PixelObservation(
                    obs.camera_id,
                    obs.u + rng.normal(0, 1.0),
                    obs.v + rng.normal(0, 1.0),
                )
            )
        result = triangulate_from_observations(data["true_by_id"], noisy)
        if result.point is not None:
            samples.append(result.point)
    samples = np.array(samples)

    fig, ax = new_scene(f"6. Monte Carlo triangulation samples — t = {T_SCENE:.2f} s")
    for cam_id in ACTIVE:
        draw_camera_frustum(ax, data["true_by_id"][cam_id], color=CAM_COLORS[cam_id], label=cam_id)
    draw_mc_samples(ax, samples, data["tri"].point)
    draw_gt(ax, data["position"], np.zeros(3), label="ground truth")
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "06_monte_carlo_triangulation.png")


def frame_07_essential_matrix(data) -> None:
    fig, ax = new_scene("7. Essential-matrix init — relative to Camera 0")
    ref = data["true_by_id"]["camera_0"]
    tgt = data["initial_by_id"]["camera_1"]
    draw_camera_frustum(ax, ref, color=CAM_COLORS["camera_0"], label="camera_0")
    draw_camera_frustum(ax, tgt, color=CAM_COLORS["camera_1"], linestyle="--", label="camera_1 initial")
    mid = 0.5 * (ref.extrinsics.t_camera_to_world + tgt.extrinsics.t_camera_to_world)
    ax.plot(
        [ref.extrinsics.t_camera_to_world[0], tgt.extrinsics.t_camera_to_world[0]],
        [ref.extrinsics.t_camera_to_world[1], tgt.extrinsics.t_camera_to_world[1]],
        [ref.extrinsics.t_camera_to_world[2], tgt.extrinsics.t_camera_to_world[2]],
        color="#64748b",
        linewidth=2,
    )
    ax.text(mid[0], mid[1], mid[2], "  E", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "07_essential_matrix_init.png")


def frame_08_bundle_adjustment(data) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"}, facecolor="#eef3f8")
    for ax, cameras, title in (
        (ax1, data["initial_by_id"], "before BA"),
        (ax2, data["est_by_id"], "after BA"),
    ):
        ax.set_facecolor("#eef3f8")
        ax.set_title(title, fontsize=12)
        for cam_id in ACTIVE:
            draw_camera_frustum(ax, cameras[cam_id], color=CAM_COLORS[cam_id], label=cam_id)
        draw_gt(ax, data["position"], np.zeros(3))
        set_axes_equal_3d(ax)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("8. Bundle adjustment — camera poses", fontsize=13)
    save_frame(fig, "08_bundle_adjustment_before_after.png")


def frame_09_tracking(data) -> None:
    fig, ax = new_scene(f"9. Online tracking with calibrated cameras — t = {T_SCENE:.2f} s")
    for cam_id in ACTIVE:
        draw_camera_frustum(ax, data["est_by_id"][cam_id], color=CAM_COLORS[cam_id], label=cam_id)
    est = data["tri"].point
    ax.scatter([est[0]], [est[1]], [est[2]], color="#1f77b4", s=70, label="EKF prior")
    draw_gt(ax, data["position"], data["velocity"])
    ax.legend(loc="lower left", fontsize=9)
    set_axes_equal_3d(ax)
    save_frame(fig, "09_calibrated_tracking.png")


def main() -> None:
    data = load_scene_data()
    frame_01_unknown_setup(data)
    frame_02_gt_velocity(data)
    frame_03_projection(data)
    frame_04_detections(data)
    frame_05_triangulation(data)
    frame_06_monte_carlo(data)
    frame_07_essential_matrix(data)
    frame_08_bundle_adjustment(data)
    frame_09_tracking(data)
    print(f"\nDone — insert PNGs from:\n  {OUT}")


if __name__ == "__main__":
    main()
