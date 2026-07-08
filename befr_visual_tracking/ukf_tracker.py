"""Raw-pixel Unscented Kalman Filter tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_model import Camera
from befr_visual_tracking.ekf_tracker import (
    PIXEL_MEASUREMENT_DIM,
    normalized_innovation_squared,
)
from befr_visual_tracking.kalman_tracker import (
    group_detections_by_timestamp,
    load_ground_truth_by_timestamp,
)
from befr_visual_tracking.motion_model import STATE_DIM, normalize_initial_covariance, position_from_state
from befr_visual_tracking.tracker_base import TrackerBase, TrackerDetection
from befr_visual_tracking.triangulation import PixelObservation, triangulate_from_observations
from befr_visual_tracking.ukf_core import (
    cap_covariance_eigenvalues,
    generate_sigma_points,
    ukf_weights,
)


@dataclass
class UkfTrackerStats:
    num_camera_updates: int = 0
    num_rejected_triangulations: int = 0
    num_rejected_sigma_points: int = 0
    num_rejected_innovations: int = 0
    num_initializations: int = 0


@dataclass
class UkfTrackerResult:
    timestamps: list[float] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    covariances: list[np.ndarray] = field(default_factory=list)
    ground_truth_positions: list[np.ndarray] = field(default_factory=list)
    ground_truth_velocities: list[np.ndarray] = field(default_factory=list)


def project_state_to_pixel(
    camera: Camera,
    state: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray | None:
    position = position_from_state(state)
    projected = camera.project_world_point_for_filter(position, epsilon=epsilon)
    if projected is None:
        return None
    return np.array([projected[0], projected[1]], dtype=float)


def ukf_pixel_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    camera: Camera,
    measurement_covariance: np.ndarray,
    *,
    innovation_gate: float = 9.21,
    projection_epsilon: float = 1e-6,
    max_invalid_sigma_points: int = 6,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool, float, int]:
    """
    Apply one raw-pixel UKF update.

    Returns (state, covariance, accepted, nis, invalid_sigma_count).
    """
    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    covariance = np.asarray(covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
    measurement = np.asarray(measurement, dtype=float).reshape(PIXEL_MEASUREMENT_DIM)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float).reshape(
        PIXEL_MEASUREMENT_DIM, PIXEL_MEASUREMENT_DIM
    )

    sigma_points, lambda_, alpha, beta = generate_sigma_points(
        state,
        covariance,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    mean_weights, cov_weights = ukf_weights(lambda_, alpha=alpha, beta=beta)

    projected_points: list[np.ndarray | None] = []
    for sigma_state in sigma_points:
        projected_points.append(
            project_state_to_pixel(camera, sigma_state, epsilon=projection_epsilon)
        )

    invalid_count = sum(point is None for point in projected_points)
    if invalid_count > max_invalid_sigma_points:
        return state, covariance, False, 0.0, invalid_count

    valid_indices = [index for index, point in enumerate(projected_points) if point is not None]
    if not valid_indices or 0 not in valid_indices:
        return state, covariance, False, 0.0, invalid_count

    renormalized_mean_weights = mean_weights[valid_indices]
    renormalized_cov_weights = cov_weights[valid_indices]
    renormalized_mean_weights /= np.sum(renormalized_mean_weights)
    renormalized_cov_weights /= np.sum(renormalized_cov_weights)

    measurement_mean = np.zeros(PIXEL_MEASUREMENT_DIM, dtype=float)
    for weight, index in zip(renormalized_mean_weights, valid_indices, strict=True):
        assert projected_points[index] is not None
        measurement_mean += weight * projected_points[index]

    measurement_covariance_ukf = np.zeros(
        (PIXEL_MEASUREMENT_DIM, PIXEL_MEASUREMENT_DIM), dtype=float
    )
    cross_covariance = np.zeros((STATE_DIM, PIXEL_MEASUREMENT_DIM), dtype=float)
    for weight, index in zip(renormalized_cov_weights, valid_indices, strict=True):
        assert projected_points[index] is not None
        delta_z = projected_points[index] - measurement_mean
        delta_x = sigma_points[index] - state
        measurement_covariance_ukf += weight * np.outer(delta_z, delta_z)
        cross_covariance += weight * np.outer(delta_x, delta_z)

    innovation_covariance = measurement_covariance_ukf + measurement_covariance
    innovation = measurement - measurement_mean
    nis = normalized_innovation_squared(innovation, innovation_covariance)
    if nis > innovation_gate:
        return state, covariance, False, nis, invalid_count

    kalman_gain = cross_covariance @ np.linalg.inv(innovation_covariance)
    updated_state = state + kalman_gain @ innovation
    updated_covariance = covariance - kalman_gain @ innovation_covariance @ kalman_gain.T
    return updated_state, updated_covariance, True, nis, invalid_count


class UkfTracker(TrackerBase):
    """Raw-pixel UKF with sequential per-camera updates."""

    def __init__(
        self,
        cameras_by_id: dict[str, Camera],
        *,
        acceleration_noise_std: float = 0.5,
        active_cameras: list[str] | None = None,
        pixel_noise_std: float | None = None,
        innovation_gate: float = 9.21,
        min_angle_deg: float = 2.0,
        max_condition_number: float = 1e8,
        monte_carlo_samples: int = 500,
        projection_epsilon: float = 1e-6,
        max_invalid_sigma_points: int = 6,
        max_initial_covariance_eigenvalue: float = 100.0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(acceleration_noise_std=acceleration_noise_std)
        self._cameras_by_id = cameras_by_id
        self._active_cameras = set(active_cameras) if active_cameras is not None else None
        self._pixel_noise_std = pixel_noise_std
        self._innovation_gate = float(innovation_gate)
        self._min_angle_deg = float(min_angle_deg)
        self._max_condition_number = float(max_condition_number)
        self._monte_carlo_samples = int(monte_carlo_samples)
        self._projection_epsilon = float(projection_epsilon)
        self._max_invalid_sigma_points = int(max_invalid_sigma_points)
        self._max_initial_covariance_eigenvalue = float(max_initial_covariance_eigenvalue)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._kappa = float(kappa)
        self._rng = rng if rng is not None else np.random.default_rng()
        self.stats = UkfTrackerStats()
        self.last_step_nis: list[float] = []

    def initialize(
        self,
        position: np.ndarray,
        covariance: np.ndarray,
        timestamp: float,
        velocity: np.ndarray | None = None,
    ) -> None:
        capped = cap_covariance_eigenvalues(
            normalize_initial_covariance(covariance),
            max_eigenvalue=self._max_initial_covariance_eigenvalue,
        )
        super().initialize(position, capped, timestamp, velocity=velocity)

    def _filter_detections(self, detections: list[TrackerDetection]) -> list[TrackerDetection]:
        if self._active_cameras is None:
            return list(detections)
        return [detection for detection in detections if detection.camera_id in self._active_cameras]

    def _measurement_covariance(self, camera: Camera) -> np.ndarray:
        if self._pixel_noise_std is not None:
            sigma = float(self._pixel_noise_std)
            return np.diag([sigma**2, sigma**2])
        return camera.noise_model.measurement_noise_covariance

    def _monte_carlo_pixel_noise(self) -> float:
        if self._pixel_noise_std is not None:
            return float(self._pixel_noise_std)
        camera = next(iter(self._cameras_by_id.values()))
        return 0.5 * (
            camera.noise_model.pixel_noise_std_u + camera.noise_model.pixel_noise_std_v
        )

    def _triangulate_for_initialization(self, detections: list[TrackerDetection]):
        observations = [
            PixelObservation(camera_id=detection.camera_id, u=detection.u, v=detection.v)
            for detection in detections
        ]
        return triangulate_from_observations(
            self._cameras_by_id,
            observations,
            min_angle_deg=self._min_angle_deg,
            max_condition_number=self._max_condition_number,
            estimate_covariance=True,
            pixel_noise_std=self._monte_carlo_pixel_noise(),
            monte_carlo_samples=self._monte_carlo_samples,
            rng=self._rng,
        )

    def try_initialize(self, timestamp: float, detections: list[TrackerDetection]) -> bool:
        if self.is_initialized:
            return True

        filtered = self._filter_detections(detections)
        if len(filtered) < 2:
            return False

        result = self._triangulate_for_initialization(filtered)
        if result.point is None or not result.quality.is_valid or result.covariance is None:
            self.stats.num_rejected_triangulations += 1
            return False

        self.initialize(result.point, result.covariance, timestamp)
        self.stats.num_initializations += 1
        return True

    def update_one_camera(self, detection: TrackerDetection) -> bool:
        self._ensure_initialized()
        assert self._state is not None
        assert self._covariance is not None

        camera = self._cameras_by_id[detection.camera_id]
        measurement = np.array([detection.u, detection.v], dtype=float)
        self._state, self._covariance, accepted, nis, invalid_count = ukf_pixel_update(
            self._state,
            self._covariance,
            measurement,
            camera,
            self._measurement_covariance(camera),
            innovation_gate=self._innovation_gate,
            projection_epsilon=self._projection_epsilon,
            max_invalid_sigma_points=self._max_invalid_sigma_points,
            alpha=self._alpha,
            beta=self._beta,
            kappa=self._kappa,
        )
        self.last_step_nis.append(nis)
        if invalid_count > self._max_invalid_sigma_points:
            self.stats.num_rejected_sigma_points += 1
            return False
        if accepted:
            self.stats.num_camera_updates += 1
        else:
            self.stats.num_rejected_innovations += 1
        return accepted

    def update(self, detections: list[TrackerDetection]) -> None:
        if not self.is_initialized:
            if self._timestamp is None:
                raise RuntimeError("Timestamp must be set before the first update")
            self.try_initialize(self._timestamp, detections)
            return

        for detection in self._filter_detections(detections):
            self.update_one_camera(detection)

    def process_timestep(self, timestamp: float, detections: list[TrackerDetection]) -> None:
        self.last_step_nis = []
        if not self.is_initialized:
            self.try_initialize(timestamp, detections)
            if self.is_initialized and detections:
                self.update(detections)
            return

        self.predict(timestamp)
        self.update(detections)


def run_ukf_tracker_on_dataset(
    *,
    experiment_dir: str | Path,
    cameras_by_id: dict[str, Camera],
    acceleration_noise_std: float = 0.5,
    pixel_noise_std: float = 1.0,
    innovation_gate: float = 9.21,
    monte_carlo_samples: int = 500,
    active_cameras: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> UkfTrackerResult:
    """Replay a saved detection dataset through the raw-pixel UKF."""
    from befr_visual_tracking.result_writer import load_detections_csv

    experiment_path = Path(experiment_dir)
    detections_by_time = group_detections_by_timestamp(
        load_detections_csv(experiment_path / "detections.csv")
    )
    ground_truth_by_time = load_ground_truth_by_timestamp(experiment_path / "ground_truth.csv")

    tracker = UkfTracker(
        cameras_by_id,
        acceleration_noise_std=acceleration_noise_std,
        pixel_noise_std=pixel_noise_std,
        innovation_gate=innovation_gate,
        monte_carlo_samples=monte_carlo_samples,
        active_cameras=active_cameras,
        rng=rng,
    )

    result = UkfTrackerResult()
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


def compute_position_rmse(result: UkfTrackerResult) -> float:
    if not result.states or not result.ground_truth_positions:
        return float("inf")
    errors = [
        np.linalg.norm(state[:3] - gt_position)
        for state, gt_position in zip(result.states, result.ground_truth_positions, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def compute_velocity_rmse(result: UkfTrackerResult) -> float:
    if not result.states or not result.ground_truth_velocities:
        return float("inf")
    errors = [
        np.linalg.norm(state[3:] - gt_velocity)
        for state, gt_velocity in zip(result.states, result.ground_truth_velocities, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))
