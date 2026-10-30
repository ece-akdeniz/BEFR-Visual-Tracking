#!/usr/bin/env python3
"""Generate minimal presentation figures for the advanced task."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_anim" / "advanced_plots"


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_unknown_poses() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Advanced task — unknown camera poses", fontsize=14, pad=12)

    centers = {
        "camera_0": (0.0, -3.0),
        "camera_1": (3.0, 0.0),
        "camera_2": (-3.0, 0.0),
        "camera_3": (2.2, 2.2),
    }
    for name, (x, y) in centers.items():
        known = name == "camera_0"
        color = "#2563eb" if known else "#94a3b8"
        ax.scatter([x], [y], s=180, c=color, zorder=3)
        label = name if known else f"{name}  ?"
        ax.text(x, y - 0.45, label, ha="center", va="top", fontsize=11)
        ax.plot([x, 0], [y, 0], color=color, linewidth=1.5, alpha=0.7)

    ax.scatter([0], [0], s=120, c="#16a34a", marker="^", zorder=4)
    ax.text(0, 0.35, "flight volume", ha="center", fontsize=11)
    ax.text(
        0,
        -4.6,
        "known: Camera 0 pose + intrinsics    |    unknown: cameras 1–3",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(OUT / "01_unknown_poses.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Calibration pipeline", fontsize=14, pad=12)

    steps = [
        "detections.csv",
        "essential-matrix init",
        "trajectory init",
        "bundle adjustment",
        "calibrated_cameras.yaml",
    ]
    y = 8.5
    for i, text in enumerate(steps):
        box = FancyBboxPatch(
            (1.5, y - 0.55),
            7.0,
            1.0,
            boxstyle="round,pad=0.08",
            linewidth=1.2,
            edgecolor="#2563eb",
            facecolor="#eff6ff",
        )
        ax.add_patch(box)
        ax.text(5, y, text, ha="center", va="center", fontsize=12)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(5, y - 0.75),
                xytext=(5, y - 1.15),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#64748b"},
            )
        y -= 1.6

    ax.text(5, 0.8, "gauge fixed by Camera 0", ha="center", fontsize=11, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "02_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_cal_reports() -> dict[tuple[int, str], dict]:
    reports: dict[tuple[int, str], dict] = {}
    for path in sorted((ROOT / "results").glob("experiment_*/calibration_report.json")):
        meta_path = path.parent / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        exp_id = meta.get("experiment_id")
        variant = meta.get("variant")
        if exp_id is None or variant is None:
            continue
        reports[(int(exp_id), str(variant))] = json.loads(path.read_text())
    return reports


def plot_manoeuvre_comparison(reports: dict) -> None:
    variants = [
        ("hover", "hover"),
        ("line", "straight line"),
        ("planar_eight", "planar eight"),
        ("figure_eight_3d", "3D eight"),
    ]
    reproj = []
    cam_err = []
    labels = []
    for key, label in variants:
        report = reports.get((101, key))
        if report is None:
            continue
        after = report["after_optimization"]
        reproj.append(after["reprojection_rmse"])
        cam_err.append(after["camera_translation_rmse"])
        labels.append(label)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = ["#94a3b8", "#64748b", "#2563eb", "#16a34a"]
    axes[0].bar(labels, reproj, color=colors[: len(labels)])
    axes[0].set_ylabel("reproj. RMSE [px]")
    axes[0].set_title("Exp. 101 — reprojection")
    _style_axes(axes[0])

    axes[1].bar(labels, cam_err, color=colors[: len(labels)])
    axes[1].set_ylabel("camera trans. RMSE [m]")
    axes[1].set_title("Exp. 101 — camera poses")
    _style_axes(axes[1])
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Manoeuvre observability", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "03_manoeuvre_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_camera_count(reports: dict) -> None:
    variants = [("two_cameras", "2 cams"), ("three_cameras", "3 cams")]
    reproj = []
    traj = []
    labels = []
    for key, label in variants:
        report = reports.get((102, key))
        if report is None:
            continue
        after = report["after_optimization"]
        reproj.append(after["reprojection_rmse"])
        traj.append(after["trajectory_rmse_aligned"] * 1000.0)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, reproj, width, label="reproj. [px]", color="#2563eb")
    ax.bar(x + width / 2, traj, width, label="traj. [mm]", color="#16a34a")
    ax.set_xticks(x, labels)
    ax.set_title("Exp. 102 — camera count")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "04_camera_count.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noise(reports: dict) -> None:
    variants = [("noise_0p5", "0.5 px"), ("noise_1p0", "1.0 px"), ("noise_2p0", "2.0 px")]
    reproj = []
    labels = []
    for key, label in variants:
        report = reports.get((103, key))
        if report is None:
            continue
        reproj.append(report["after_optimization"]["reprojection_rmse"])
        labels.append(label)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(labels, reproj, marker="o", linewidth=2, color="#2563eb", markersize=8)
    ax.set_ylabel("reproj. RMSE [px]")
    ax.set_title("Exp. 103 — pixel noise")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "05_noise.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_trajectory() -> None:
    csv_path = ROOT / "results" / "calibration_figure_eight_3d" / "ground_truth.csv"
    xs, ys, zs = [], [], []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["gt_x"]))
            ys.append(float(row["gt_y"]))
            zs.append(float(row["gt_z"]))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color="#16a34a", linewidth=2)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Calibration manoeuvre — 3D figure-eight")
    fig.tight_layout()
    fig.savefig(OUT / "06_calibration_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_camera_poses_3d() -> None:
    from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras
    from befr_visual_tracking.camera_model import cameras_from_yaml

    exp_dir = ROOT / "results" / "calibration_figure_eight_3d"
    _, true_cameras = cameras_from_yaml(exp_dir / "camera_config.yaml")
    true_by_id = apply_gauge_fix_to_cameras({c.name: c for c in true_cameras})
    _, est_cameras = cameras_from_yaml(exp_dir / "calibrated_cameras.yaml")
    est_by_id = {c.name: c for c in est_cameras}

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for label, cameras, color in (
        ("true", true_by_id, "#16a34a"),
        ("optimised", est_by_id, "#f97316"),
    ):
        xs = [cam.extrinsics.t_camera_to_world[0] for cam in cameras.values()]
        ys = [cam.extrinsics.t_camera_to_world[1] for cam in cameras.values()]
        zs = [cam.extrinsics.t_camera_to_world[2] for cam in cameras.values()]
        ax.scatter(xs, ys, zs, s=50, c=color, label=label)
        for name, cam in cameras.items():
            ax.text(
                cam.extrinsics.t_camera_to_world[0],
                cam.extrinsics.t_camera_to_world[1],
                cam.extrinsics.t_camera_to_world[2],
                name,
                fontsize=8,
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Camera poses — true vs optimised")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "07_cameras_true_vs_optimised.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_essential_matrix_schematic() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Essential-matrix initialisation", fontsize=14, pad=10)

    ax.add_patch(FancyBboxPatch((0.5, 2.2), 2.2, 1.4, boxstyle="round", fc="#eff6ff", ec="#2563eb"))
    ax.text(1.6, 2.9, "camera 0\n(fixed)", ha="center", va="center", fontsize=11)

    ax.add_patch(FancyBboxPatch((7.3, 2.2), 2.2, 1.4, boxstyle="round", fc="#fff7ed", ec="#f97316"))
    ax.text(8.4, 2.9, "camera i\n(unknown)", ha="center", va="center", fontsize=11)

    arrow = FancyArrowPatch((2.8, 2.9), (7.1, 2.9), arrowstyle="<->", mutation_scale=14, lw=1.8, color="#64748b")
    ax.add_patch(arrow)
    ax.text(5.0, 3.25, "essential matrix E", ha="center", fontsize=11)

    ax.text(5.0, 1.2, "relative pose  →  batch bundle adjustment", ha="center", fontsize=11, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "08_essential_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = _load_cal_reports()
    plot_unknown_poses()
    plot_pipeline()
    plot_essential_matrix_schematic()
    plot_manoeuvre_comparison(reports)
    plot_camera_count(reports)
    plot_noise(reports)
    plot_calibration_trajectory()
    plot_camera_poses_3d()
    print(f"Saved advanced presentation plots to {OUT}")


if __name__ == "__main__":
    main()
