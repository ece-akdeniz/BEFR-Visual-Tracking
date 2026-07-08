"""Initial 3D trajectory from approximate camera poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.advanced.observations import TimestampObservations
from befr_visual_tracking.camera_model import Camera
from befr_visual_tracking.triangulation import PixelObservation, triangulate_from_observations


@dataclass
class TrajectoryInitialization:
    timestamps: list[float]
    positions: list[np.ndarray]
    valid_mask: list[bool]
    reprojection_rmse: list[float]


def initialize_trajectory(
    observations: list[TimestampObservations],
    cameras_by_id: dict[str, Camera],
    *,
    min_angle_deg: float = 2.0,
    max_condition_number: float = 1e8,
    max_reprojection_rmse: float = 15.0,
) -> TrajectoryInitialization:
    """Triangulate one 3D position per timestamp and reject poor geometry."""
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    valid_mask: list[bool] = []
    reprojection_errors: list[float] = []

    for observation in observations:
        pixel_obs = [
            PixelObservation(camera_id=m.camera_id, u=m.u, v=m.v)
            for m in observation.measurements
        ]
        result = triangulate_from_observations(
            cameras_by_id,
            pixel_obs,
            min_angle_deg=min_angle_deg,
            max_condition_number=max_condition_number,
        )
        timestamps.append(observation.timestamp)
        if result.point is None or not result.quality.is_valid:
            positions.append(np.full(3, np.nan, dtype=float))
            valid_mask.append(False)
            reprojection_errors.append(float("inf"))
            continue

        rmse = result.quality.reprojection_rmse
        if rmse > max_reprojection_rmse:
            positions.append(np.full(3, np.nan, dtype=float))
            valid_mask.append(False)
            reprojection_errors.append(rmse)
            continue

        positions.append(result.point.copy())
        valid_mask.append(True)
        reprojection_errors.append(rmse)

    return TrajectoryInitialization(
        timestamps=timestamps,
        positions=positions,
        valid_mask=valid_mask,
        reprojection_rmse=reprojection_errors,
    )


def interpolate_missing_positions(initialization: TrajectoryInitialization) -> np.ndarray:
    """Linearly interpolate NaN trajectory samples in time."""
    timestamps = np.asarray(initialization.timestamps, dtype=float)
    positions = np.asarray(initialization.positions, dtype=float)
    valid = np.asarray(initialization.valid_mask, dtype=bool)

    if not np.any(valid):
        raise RuntimeError("No valid triangulated positions for trajectory initialization")

    filled = positions.copy()
    valid_indices = np.flatnonzero(valid)
    for axis in range(3):
        filled[:, axis] = np.interp(
            timestamps,
            timestamps[valid_indices],
            positions[valid_indices, axis],
        )
    return filled
