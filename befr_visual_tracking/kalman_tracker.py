"""Triangulation-based linear Kalman filter baseline tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_model import Camera
from befr_visual_tracking.motion_model import STATE_DIM
from befr_visual_tracking.result_writer import load_detections_csv, load_ground_truth_csv
from befr_visual_tracking.tracker_base import TrackerBase, TrackerDetection
from befr_visual_tracking.triangulation import PixelObservation, triangulate_from_observations


MEASUREMENT_DIM = 3
H_POSITION = np.zeros((MEASUREMENT_DIM, STATE_DIM), dtype=float)
H_POSITION[:, :3] = np.eye(MEASUREMENT_DIM, dtype=float)


@dataclass
class KalmanTrackerStats:
    num_updates: int = 0
    num_rejected_triangulations: int = 0
    num_prediction_only_steps: int = 0
    num_initializations: int = 0


@dataclass
class KalmanTrackerResult:
    timestamps: list[float] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    covariances: list[np.ndarray] = field(default_factory=list)
    ground_truth_positions: list[np.ndarray] = field(default_factory=list)
    ground_truth_velocities: list[np.ndarray] = field(default_factory=list)


def linear_position_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a linear Kalman update with z = H x + noise, H = [I3 | 0]."""
    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    covariance = np.asarray(covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
    measurement = np.asarray(measurement, dtype=float).reshape(MEASUREMENT_DIM)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float).reshape(
        MEASUREMENT_DIM, MEASUREMENT_DIM
    )

    innovation = measurement - H_POSITION @ state
    innovation_covariance = H_POSITION @ covariance @ H_POSITION.T + measurement_covariance
    kalman_gain = covariance @ H_POSITION.T @ np.linalg.inv(innovation_covariance)

    updated_state = state + kalman_gain @ innovation
    identity = np.eye(STATE_DIM, dtype=float)
    updated_covariance = (identity - kalman_gain @ H_POSITION) @ covariance
    return updated_state, updated_covariance


class KalmanTracker(TrackerBase):
    """
    Baseline tracker: multi-ray triangulation followed by a linear Kalman filter.

    Rules:
        - update only when at least two cameras are available
        - prediction only with zero or one camera
        - triangulation covariance is used as R_k
        - weak triangulations are rejected
        - first valid triangulation initialises the filter
    """

    def __init__(
        self,
        cameras_by_id: dict[str, Camera],
        *,
        acceleration_noise_std: float = 0.5,
        active_cameras: list[str] | None = None,
        pixel_noise_std: float = 1.0,
        min_angle_deg: float = 2.0,
        max_condition_number: float = 1e8,
        max_reprojection_rmse: float = 10.0,
        max_ray_intersection_residual: float = 1.0,
        monte_carlo_samples: int = 500,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(acceleration_noise_std=acceleration_noise_std)
        self._cameras_by_id = cameras_by_id
        self._active_cameras = set(active_cameras) if active_cameras is not None else None
        self._pixel_noise_std = float(pixel_noise_std)
        self._min_angle_deg = float(min_angle_deg)
        self._max_condition_number = float(max_condition_number)
        self._max_reprojection_rmse = float(max_reprojection_rmse)
        self._max_ray_intersection_residual = float(max_ray_intersection_residual)
        self._monte_carlo_samples = int(monte_carlo_samples)
        self._rng = rng if rng is not None else np.random.default_rng()
        self.stats = KalmanTrackerStats()

    def _filter_detections(self, detections: list[TrackerDetection]) -> list[TrackerDetection]:
        if self._active_cameras is None:
            return list(detections)
        return [detection for detection in detections if detection.camera_id in self._active_cameras]

    def _to_observations(self, detections: list[TrackerDetection]) -> list[PixelObservation]:
        return [
            PixelObservation(camera_id=detection.camera_id, u=detection.u, v=detection.v)
            for detection in detections
        ]

    def _triangulate(self, detections: list[TrackerDetection]):
        return triangulate_from_observations(
            self._cameras_by_id,
            self._to_observations(detections),
            min_angle_deg=self._min_angle_deg,
            max_condition_number=self._max_condition_number,
            estimate_covariance=True,
            pixel_noise_std=self._pixel_noise_std,
            monte_carlo_samples=self._monte_carlo_samples,
            rng=self._rng,
        )

    def _triangulation_acceptable(self, result) -> bool:
        if result.point is None or not result.quality.is_valid:
            return False
        if result.covariance is None:
            return False
        if result.quality.reprojection_rmse > self._max_reprojection_rmse:
            return False
        if result.quality.ray_intersection_residual > self._max_ray_intersection_residual:
            return False
        return True

    def try_initialize(self, timestamp: float, detections: list[TrackerDetection]) -> bool:
        """Initialise from a valid triangulation if the tracker is not yet ready."""
        if self.is_initialized:
            return True

        filtered = self._filter_detections(detections)
        if len(filtered) < 2:
            return False

        result = self._triangulate(filtered)
        if not self._triangulation_acceptable(result):
            self.stats.num_rejected_triangulations += 1
            return False

        assert result.point is not None
        assert result.covariance is not None
        self.initialize(result.point, result.covariance, timestamp)
        self.stats.num_initializations += 1
        return True

    def update(self, detections: list[TrackerDetection]) -> None:
        """Apply a triangulation measurement update when at least two cameras are visible."""
        filtered = self._filter_detections(detections)
        if len(filtered) < 2:
            self.stats.num_prediction_only_steps += 1
            return

        if not self.is_initialized:
            if self._timestamp is None:
                raise RuntimeError("Timestamp must be set before the first update")
            initialized = self.try_initialize(self._timestamp, filtered)
            if not initialized:
                self.stats.num_prediction_only_steps += 1
            return

        result = self._triangulate(filtered)
        if not self._triangulation_acceptable(result):
            self.stats.num_rejected_triangulations += 1
            return

        assert result.point is not None
        assert result.covariance is not None
        assert self._state is not None
        assert self._covariance is not None

        self._state, self._covariance = linear_position_update(
            self._state,
            self._covariance,
            result.point,
            result.covariance,
        )
        self.stats.num_updates += 1

    def process_timestep(self, timestamp: float, detections: list[TrackerDetection]) -> None:
        """Predict to ``timestamp`` and apply an update when enough cameras are available."""
        if not self.is_initialized:
            self.try_initialize(timestamp, detections)
            return

        self.predict(timestamp)
        self.update(detections)


def group_detections_by_timestamp(
    detections: list[dict[str, str]],
) -> dict[float, list[TrackerDetection]]:
    grouped: dict[float, list[TrackerDetection]] = {}
    for row in detections:
        if row.get("visible", "").lower() != "true":
            continue
        if not row.get("u") or not row.get("v"):
            continue

        timestamp = float(row["timestamp"])
        grouped.setdefault(timestamp, []).append(
            TrackerDetection(
                timestamp=timestamp,
                camera_id=row["camera_id"],
                u=float(row["u"]),
                v=float(row["v"]),
            )
        )
    return grouped


def load_ground_truth_by_timestamp(path: str | Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    rows = load_ground_truth_csv(path)
    ground_truth: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        timestamp = float(row["timestamp"])
        position = np.array(
            [float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])],
            dtype=float,
        )
        velocity = np.array(
            [float(row["gt_vx"]), float(row["gt_vy"]), float(row["gt_vz"])],
            dtype=float,
        )
        ground_truth[timestamp] = (position, velocity)
    return ground_truth


def run_kalman_tracker_on_dataset(
    *,
    experiment_dir: str | Path,
    cameras_by_id: dict[str, Camera],
    acceleration_noise_std: float = 0.5,
    pixel_noise_std: float = 1.0,
    monte_carlo_samples: int = 500,
    active_cameras: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> KalmanTrackerResult:
    """Replay a saved detection dataset through the baseline Kalman tracker."""
    experiment_path = Path(experiment_dir)
    detections_by_time = group_detections_by_timestamp(
        load_detections_csv(experiment_path / "detections.csv")
    )
    ground_truth_by_time = load_ground_truth_by_timestamp(experiment_path / "ground_truth.csv")

    tracker = KalmanTracker(
        cameras_by_id,
        acceleration_noise_std=acceleration_noise_std,
        pixel_noise_std=pixel_noise_std,
        monte_carlo_samples=monte_carlo_samples,
        active_cameras=active_cameras,
        rng=rng,
    )

    result = KalmanTrackerResult()
    for timestamp in sorted(detections_by_time.keys()):
        tracker.process_timestep(timestamp, detections_by_time[timestamp])
        if not tracker.is_initialized:
            continue

        gt = ground_truth_by_time.get(timestamp)
        result.timestamps.append(timestamp)
        result.states.append(tracker.get_state())
        result.covariances.append(tracker.get_covariance())
        if gt is not None:
            result.ground_truth_positions.append(gt[0])
            result.ground_truth_velocities.append(gt[1])

    return result


def compute_position_rmse(result: KalmanTrackerResult) -> float:
    if not result.states or not result.ground_truth_positions:
        return float("inf")

    errors = [
        np.linalg.norm(state[:3] - gt_position)
        for state, gt_position in zip(result.states, result.ground_truth_positions, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def compute_velocity_rmse(result: KalmanTrackerResult) -> float:
    if not result.states or not result.ground_truth_velocities:
        return float("inf")

    errors = [
        np.linalg.norm(state[3:] - gt_velocity)
        for state, gt_velocity in zip(result.states, result.ground_truth_velocities, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))
