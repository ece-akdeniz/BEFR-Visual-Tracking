"""Compute standard experiment metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class StepRecord:
    timestamp: float
    state: np.ndarray
    covariance: np.ndarray
    ground_truth_position: np.ndarray
    ground_truth_velocity: np.ndarray
    num_visible_cameras: int
    update_time_sec: float
    nis_values: list[float] = field(default_factory=list)


@dataclass
class TrackingMetrics:
    position_rmse: float
    velocity_rmse: float
    position_max_error: float
    velocity_max_error: float
    total_runtime_sec: float
    mean_update_time_sec: float
    p95_update_time_sec: float
    max_update_time_sec: float
    filter_frequency_hz: float
    convergence_time_sec: float | None
    recovery_time_sec: float | None
    num_steps: int
    num_visible_cameras_mean: float
    num_rejected_detections: int
    covariance_position_mean: list[float]
    covariance_velocity_mean: list[float]
    nis_mean: float | None
    nis_p95: float | None
    reprojection_rmse: float | None = None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def position_errors(steps: list[StepRecord]) -> np.ndarray:
    return np.array(
        [np.linalg.norm(step.state[:3] - step.ground_truth_position) for step in steps],
        dtype=float,
    )


def velocity_errors(steps: list[StepRecord]) -> np.ndarray:
    return np.array(
        [np.linalg.norm(step.state[3:] - step.ground_truth_velocity) for step in steps],
        dtype=float,
    )


def compute_convergence_time(
    steps: list[StepRecord],
    *,
    error_threshold: float = 0.05,
    hold_steps: int = 5,
) -> float | None:
    """Seconds until position error stays below threshold for hold_steps updates."""
    if len(steps) < hold_steps:
        return None

    errors = position_errors(steps)
    for index in range(hold_steps - 1, len(steps)):
        window = errors[index - hold_steps + 1 : index + 1]
        if np.all(window <= error_threshold):
            return float(steps[index].timestamp - steps[0].timestamp)
    return None


def compute_recovery_time(
    steps: list[StepRecord],
    *,
    dropout_threshold_cameras: int = 2,
    recovery_ratio: float = 1.2,
) -> float | None:
    """
    Time from the end of a low-camera segment until position error recovers.

    Recovery means returning to recovery_ratio times the pre-dropout error level.
    """
    if len(steps) < 3:
        return None

    visible = np.array([step.num_visible_cameras for step in steps], dtype=int)
    errors = position_errors(steps)
    dropout_indices = np.where(visible <= dropout_threshold_cameras)[0]
    if len(dropout_indices) == 0:
        return None

    dropout_start = int(dropout_indices[0])
    if dropout_start == 0:
        baseline = errors[0]
    else:
        baseline = float(np.mean(errors[max(0, dropout_start - 5) : dropout_start]))

    recovery_target = max(baseline * recovery_ratio, 1e-3)
    end_dropout = int(dropout_indices[-1])
    for index in range(end_dropout + 1, len(steps)):
        if visible[index] >= dropout_threshold_cameras + 1 and errors[index] <= recovery_target:
            return float(steps[index].timestamp - steps[end_dropout].timestamp)
    return None


def compute_tracking_metrics(
    steps: list[StepRecord],
    *,
    tracker_stats: dict | None = None,
    reprojection_rmse: float | None = None,
) -> TrackingMetrics:
    if not steps:
        return TrackingMetrics(
            position_rmse=float("inf"),
            velocity_rmse=float("inf"),
            position_max_error=float("inf"),
            velocity_max_error=float("inf"),
            total_runtime_sec=0.0,
            mean_update_time_sec=0.0,
            p95_update_time_sec=0.0,
            max_update_time_sec=0.0,
            filter_frequency_hz=0.0,
            convergence_time_sec=None,
            recovery_time_sec=None,
            num_steps=0,
            num_visible_cameras_mean=0.0,
            num_rejected_detections=0,
            covariance_position_mean=[0.0, 0.0, 0.0],
            covariance_velocity_mean=[0.0, 0.0, 0.0],
            nis_mean=None,
            nis_p95=None,
            reprojection_rmse=reprojection_rmse,
        )

    pos_err = position_errors(steps)
    vel_err = velocity_errors(steps)
    update_times = [step.update_time_sec for step in steps]
    duration = steps[-1].timestamp - steps[0].timestamp
    all_nis = [value for step in steps for value in step.nis_values]

    cov_positions = np.array([step.covariance[:3, :3].diagonal() for step in steps], dtype=float)
    cov_velocities = np.array([step.covariance[3:, 3:].diagonal() for step in steps], dtype=float)

    rejected = 0
    if tracker_stats:
        rejected = int(
            tracker_stats.get("num_rejected_innovations", 0)
            + tracker_stats.get("num_rejected_projections", 0)
            + tracker_stats.get("num_rejected_triangulations", 0)
            + tracker_stats.get("num_rejected_sigma_points", 0)
        )

    return TrackingMetrics(
        position_rmse=float(np.sqrt(np.mean(np.square(pos_err)))),
        velocity_rmse=float(np.sqrt(np.mean(np.square(vel_err)))),
        position_max_error=float(np.max(pos_err)),
        velocity_max_error=float(np.max(vel_err)),
        total_runtime_sec=float(np.sum(update_times)),
        mean_update_time_sec=float(np.mean(update_times)),
        p95_update_time_sec=_percentile(update_times, 95.0),
        max_update_time_sec=float(np.max(update_times)),
        filter_frequency_hz=float(len(steps) / duration) if duration > 0 else 0.0,
        convergence_time_sec=compute_convergence_time(steps),
        recovery_time_sec=compute_recovery_time(steps),
        num_steps=len(steps),
        num_visible_cameras_mean=float(np.mean([step.num_visible_cameras for step in steps])),
        num_rejected_detections=rejected,
        covariance_position_mean=[float(v) for v in cov_positions.mean(axis=0)],
        covariance_velocity_mean=[float(v) for v in cov_velocities.mean(axis=0)],
        nis_mean=float(np.mean(all_nis)) if all_nis else None,
        nis_p95=_percentile(all_nis, 95.0) if all_nis else None,
        reprojection_rmse=reprojection_rmse,
    )


def tracking_metrics_to_dict(metrics: TrackingMetrics) -> dict:
    return asdict(metrics)
