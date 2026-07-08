"""Load saved experiment folders for dashboard Mode A."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.experiments.metrics import StepRecord, compute_tracking_metrics, tracking_metrics_to_dict
from befr_visual_tracking.result_writer import load_detections_csv, load_ground_truth_csv, load_metadata_json


@dataclass
class LoadedExperiment:
    path: Path
    metadata: dict
    metrics: dict
    cameras_by_id: dict
    ground_truth: list[dict]
    detections: list[dict]
    steps: list[StepRecord]


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def list_saved_experiments(results_root: str | Path | None = None) -> list[Path]:
    root = Path(results_root or default_results_dir())
    if not root.is_dir():
        return []
    experiments = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file() and (path / "estimates.csv").is_file()
    ]
    return sorted(experiments, key=lambda path: path.name)


def load_estimates_csv(path: str | Path) -> list[StepRecord]:
    steps: list[StepRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            state = np.array(
                [float(row[f"est_{axis}"]) for axis in ("x", "y", "z", "vx", "vy", "vz")],
                dtype=float,
            )
            gt_position = state[:3] - np.array(
                [float(row["pos_err"]) for _ in range(3)],
                dtype=float,
            )
            # Reconstruct GT from stored error magnitude along each axis is unavailable;
            # load ground truth separately when possible.
            gt_position = np.zeros(3, dtype=float)
            gt_velocity = np.zeros(3, dtype=float)
            covariance = np.diag(
                [
                    float(row["cov_px"]),
                    float(row["cov_py"]),
                    float(row["cov_pz"]),
                    float(row["cov_vx"]),
                    float(row["cov_vy"]),
                    float(row["cov_vz"]),
                ]
            )
            nis_values = [float(row["nis_mean"])] if row.get("nis_mean") else []
            steps.append(
                StepRecord(
                    timestamp=float(row["timestamp"]),
                    state=state,
                    covariance=covariance,
                    ground_truth_position=gt_position,
                    ground_truth_velocity=gt_velocity,
                    num_visible_cameras=int(row["num_visible_cameras"]),
                    update_time_sec=float(row["update_time_sec"]),
                    nis_values=nis_values,
                )
            )
    return steps


def attach_ground_truth(steps: list[StepRecord], ground_truth_rows: list[dict]) -> list[StepRecord]:
    gt_by_time = {float(row["timestamp"]): row for row in ground_truth_rows}
    enriched: list[StepRecord] = []
    for step in steps:
        row = gt_by_time.get(step.timestamp)
        if row is None:
            enriched.append(step)
            continue
        enriched.append(
            StepRecord(
                timestamp=step.timestamp,
                state=step.state,
                covariance=step.covariance,
                ground_truth_position=np.array(
                    [float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])],
                    dtype=float,
                ),
                ground_truth_velocity=np.array(
                    [float(row["gt_vx"]), float(row["gt_vy"]), float(row["gt_vz"])],
                    dtype=float,
                ),
                num_visible_cameras=step.num_visible_cameras,
                update_time_sec=step.update_time_sec,
                nis_values=step.nis_values,
            )
        )
    return enriched


def load_saved_experiment(path: str | Path) -> LoadedExperiment:
    experiment_path = Path(path)
    metadata = load_metadata_json(experiment_path / "metadata.json")
    metrics_path = experiment_path / "metrics.json"
    metrics = load_metadata_json(metrics_path) if metrics_path.is_file() else {}

    _, cameras = cameras_from_yaml(experiment_path / "camera_config.yaml")
    cameras_by_id = {camera.name: camera for camera in cameras}
    ground_truth = load_ground_truth_csv(experiment_path / "ground_truth.csv")
    detections = load_detections_csv(experiment_path / "detections.csv")
    steps = attach_ground_truth(load_estimates_csv(experiment_path / "estimates.csv"), ground_truth)

    if not metrics and steps:
        metrics = tracking_metrics_to_dict(compute_tracking_metrics(steps))

    return LoadedExperiment(
        path=experiment_path,
        metadata=metadata,
        metrics=metrics,
        cameras_by_id=cameras_by_id,
        ground_truth=ground_truth,
        detections=detections,
        steps=steps,
    )
