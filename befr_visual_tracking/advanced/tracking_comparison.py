"""Compare online filter performance with true vs calibrated camera poses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from befr_visual_tracking.advanced.calibrated_config import load_calibrated_cameras
from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_points
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.ekf_tracker import run_ekf_tracker_on_dataset
from befr_visual_tracking.kalman_tracker import run_kalman_tracker_on_dataset
from befr_visual_tracking.result_writer import load_ground_truth_csv, write_metadata_json
from befr_visual_tracking.ukf_tracker import run_ukf_tracker_on_dataset


@dataclass
class TrackingComparisonResult:
    filter_type: str
    camera_source: str
    position_rmse: float
    velocity_rmse: float
    num_steps: int


def _position_rmse(result) -> float:
    errors = [
        np.linalg.norm(state[:3] - gt)
        for state, gt in zip(result.states, result.ground_truth_positions, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def _velocity_rmse(result) -> float:
    errors = [
        np.linalg.norm(state[3:] - gt)
        for state, gt in zip(result.states, result.ground_truth_velocities, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def _run_filter(
    filter_type: str,
    experiment_dir: Path,
    cameras_by_id: dict,
    *,
    active_cameras: list[str] | None,
):
    kwargs = {
        "experiment_dir": experiment_dir,
        "cameras_by_id": cameras_by_id,
        "active_cameras": active_cameras,
    }
    if filter_type == "kf":
        return run_kalman_tracker_on_dataset(**kwargs)
    if filter_type == "ekf":
        return run_ekf_tracker_on_dataset(**kwargs)
    if filter_type == "ukf":
        return run_ukf_tracker_on_dataset(**kwargs)
    raise ValueError(filter_type)


def compare_tracking_with_calibrated_cameras(
    experiment_dir: str | Path,
    *,
    calibrated_config_path: str | Path | None = None,
    filter_types: list[str] | None = None,
    active_cameras: list[str] | None = None,
    reference_camera_id: str = "camera_0",
) -> list[TrackingComparisonResult]:
    """
    Replay KF/EKF/UKF with true simulation poses and calibrated poses.

    Ground-truth positions are transformed into the camera_0 gauge for a fair
    comparison when calibrated cameras are expressed in that reference frame.
    """
    experiment_path = Path(experiment_dir)
    filter_types = filter_types or ["kf", "ekf", "ukf"]

    _, true_cameras = cameras_from_yaml(experiment_path / "camera_config.yaml")
    true_by_id = {camera.name: camera for camera in true_cameras}

    calibrated_path = Path(calibrated_config_path or experiment_path / "calibrated_cameras.yaml")
    _, calibrated_by_id = load_calibrated_cameras(calibrated_path)

    gt_rows = load_ground_truth_csv(experiment_path / "ground_truth.csv")
    reference = true_by_id[reference_camera_id]

    # Patch ground truth in detections replay by relying on tracker result GT loading.
    # The offline runners read GT from CSV in world frame; for calibrated cameras we
    # need gauge-fixed GT. We compare using transformed GT after each run below.

    results: list[TrackingComparisonResult] = []
    for filter_type in filter_types:
        for label, cameras in (("true_poses", true_by_id), ("calibrated_poses", calibrated_by_id)):
            tracker_result = _run_filter(
                filter_type,
                experiment_path,
                cameras,
                active_cameras=active_cameras,
            )
            if label == "calibrated_poses":
                tracker_result.ground_truth_positions = [
                    apply_gauge_fix_to_points(position.reshape(1, 3), reference.extrinsics)[0]
                    for position in tracker_result.ground_truth_positions
                ]
            results.append(
                TrackingComparisonResult(
                    filter_type=filter_type,
                    camera_source=label,
                    position_rmse=_position_rmse(tracker_result),
                    velocity_rmse=_velocity_rmse(tracker_result),
                    num_steps=len(tracker_result.timestamps),
                )
            )
    return results


def save_tracking_comparison_report(
    experiment_dir: str | Path,
    results: list[TrackingComparisonResult],
) -> Path:
    path = Path(experiment_dir) / "tracking_comparison_report.json"
    payload = [
        {
            "filter_type": result.filter_type,
            "camera_source": result.camera_source,
            "position_rmse": result.position_rmse,
            "velocity_rmse": result.velocity_rmse,
            "num_steps": result.num_steps,
        }
        for result in results
    ]
    write_metadata_json(path, {"comparisons": payload})
    return path
