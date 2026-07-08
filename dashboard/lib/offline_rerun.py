"""Interactive offline rerun pipeline for dashboard Mode B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import Camera, CameraNoiseModel, cameras_from_yaml
from befr_visual_tracking.dataset_generator import CanonicalDataset, DetectionRow, generate_detection_rows
from befr_visual_tracking.experiments.dropout import apply_dropout_to_detections
from befr_visual_tracking.experiments.metrics import StepRecord, compute_tracking_metrics, tracking_metrics_to_dict
from befr_visual_tracking.experiments.replay import TrackingReplayResult, replay_tracker_on_detections
from befr_visual_tracking.kalman_tracker import group_detections_by_timestamp, load_ground_truth_by_timestamp
from befr_visual_tracking.trajectory_design import build_trajectory, sample_trajectory
from befr_visual_tracking.tracker_base import TrackerDetection


TRAJECTORY_ALIASES = {
    "circle": "slow_circle",
    "square": "square",
    "planar figure-eight": "planar_figure_eight",
    "3d figure-eight": "calibration_figure_eight_3d",
}


@dataclass
class InteractiveRunConfig:
    filter_type: str
    trajectory: str
    active_cameras: list[str]
    pixel_noise_std: float
    dropout_mode: str = "none"
    acceleration_noise_std: float = 0.5
    random_seed: int = 42
    sample_rate_hz: float = 30.0


@dataclass
class InteractiveRunResult:
    config: InteractiveRunConfig
    dataset: CanonicalDataset
    cameras_by_id: dict[str, Camera]
    replay: TrackingReplayResult
    metrics: dict
    detections_by_time: dict[float, list[TrackerDetection]]
    ground_truth_by_time: dict[float, tuple[np.ndarray, np.ndarray]]


def _apply_dropout(dataset: CanonicalDataset, mode: str, camera_ids: list[str]) -> CanonicalDataset:
    if mode == "none":
        return dataset
    if mode == "one camera":
        keep = {camera_ids[0]}
    elif mode == "multiple cameras":
        keep = set(camera_ids[: max(2, len(camera_ids) - 1)])
    elif mode == "all cameras temporarily":
        return CanonicalDataset(
            detections=apply_dropout_to_detections(dataset.detections, camera_ids=camera_ids),
            ground_truth=dataset.ground_truth,
            random_seed=dataset.random_seed,
        )
    else:
        keep = set(camera_ids)

    rows: list[DetectionRow] = []
    for row in dataset.detections:
        if row.camera_id not in keep:
            rows.append(
                DetectionRow(
                    timestamp=row.timestamp,
                    camera_id=row.camera_id,
                    u=None,
                    v=None,
                    u_ideal=row.u_ideal,
                    v_ideal=row.v_ideal,
                    visible=False,
                    gt_x=row.gt_x,
                    gt_y=row.gt_y,
                    gt_z=row.gt_z,
                    gt_vx=row.gt_vx,
                    gt_vy=row.gt_vy,
                    gt_vz=row.gt_vz,
                )
            )
        else:
            rows.append(row)
    return CanonicalDataset(
        detections=tuple(rows),
        ground_truth=dataset.ground_truth,
        random_seed=dataset.random_seed,
    )


def _configure_cameras(pixel_noise_std: float) -> dict[str, Camera]:
    _, cameras = cameras_from_yaml(default_cameras_yaml_path())
    configured: dict[str, Camera] = {}
    for camera in cameras:
        camera.noise_model = CameraNoiseModel(
            pixel_noise_std_u=pixel_noise_std,
            pixel_noise_std_v=pixel_noise_std,
            discretize=True,
            detection_probability=camera.noise_model.detection_probability,
        )
        configured[camera.name] = camera
    return configured


def run_interactive_tracking(config: InteractiveRunConfig) -> InteractiveRunResult:
    """Regenerate noisy detections and run the selected filter offline."""
    trajectory_key = TRAJECTORY_ALIASES.get(config.trajectory, config.trajectory)
    trajectory = build_trajectory(trajectory_key, sample_rate_hz=config.sample_rate_hz)
    samples = sample_trajectory(trajectory)

    cameras_by_id = _configure_cameras(config.pixel_noise_std)
    cameras = list(cameras_by_id.values())

    dataset = generate_detection_rows(
        cameras=cameras,
        trajectory_samples=samples,
        random_seed=config.random_seed,
    )
    dataset = _apply_dropout(dataset, config.dropout_mode, list(cameras_by_id.keys()))

    rows = load_detections_csv_rows(dataset)
    detections_by_time = group_detections_by_timestamp(rows)
    ground_truth_by_time = {
        sample.timestamp: (sample.position.copy(), sample.velocity.copy()) for sample in samples
    }

    replay = replay_tracker_on_detections(
        filter_type=config.filter_type,
        cameras_by_id=cameras_by_id,
        detections_by_time=detections_by_time,
        ground_truth_by_time=ground_truth_by_time,
        active_cameras=config.active_cameras,
        acceleration_noise_std=config.acceleration_noise_std,
        pixel_noise_std=config.pixel_noise_std,
        random_seed=config.random_seed,
    )
    metrics = tracking_metrics_to_dict(
        compute_tracking_metrics(replay.steps, tracker_stats=replay.tracker_stats)
    )

    return InteractiveRunResult(
        config=config,
        dataset=dataset,
        cameras_by_id=cameras_by_id,
        replay=replay,
        metrics=metrics,
        detections_by_time=detections_by_time,
        ground_truth_by_time=ground_truth_by_time,
    )


def load_detections_csv_rows(dataset: CanonicalDataset) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in dataset.detections:
        rows.append(
            {
                "timestamp": f"{row.timestamp:.8f}",
                "camera_id": row.camera_id,
                "u": "" if row.u is None else f"{row.u:.8f}",
                "v": "" if row.v is None else f"{row.v:.8f}",
                "visible": str(row.visible).lower(),
            }
        )
    return rows


def compare_filters(config: InteractiveRunConfig, filter_types: list[str]) -> list[InteractiveRunResult]:
    return [
        run_interactive_tracking(
            InteractiveRunConfig(
                filter_type=filter_type,
                trajectory=config.trajectory,
                active_cameras=config.active_cameras,
                pixel_noise_std=config.pixel_noise_std,
                dropout_mode=config.dropout_mode,
                acceleration_noise_std=config.acceleration_noise_std,
                random_seed=config.random_seed,
                sample_rate_hz=config.sample_rate_hz,
            )
        )
        for filter_type in filter_types
    ]
