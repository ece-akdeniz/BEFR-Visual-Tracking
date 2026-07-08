"""Load synchronized multi-camera observations for calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.result_writer import load_detections_csv, load_ground_truth_csv


@dataclass(frozen=True)
class PixelMeasurement:
    timestamp: float
    camera_id: str
    u: float
    v: float


@dataclass(frozen=True)
class TimestampObservations:
    timestamp: float
    measurements: tuple[PixelMeasurement, ...]


def _measurement_from_row(row: dict[str, str]) -> PixelMeasurement | None:
    u_value = row.get("u") or row.get("u_ideal")
    v_value = row.get("v") or row.get("v_ideal")
    if not u_value or not v_value:
        return None
    return PixelMeasurement(
        timestamp=float(row["timestamp"]),
        camera_id=row["camera_id"],
        u=float(u_value),
        v=float(v_value),
    )


def load_visible_measurements(path: str) -> list[PixelMeasurement]:
    """Load all visible pixel detections from a detections CSV."""
    rows = load_detections_csv(path)
    measurements: list[PixelMeasurement] = []
    for row in rows:
        if row.get("visible", "").lower() != "true":
            continue
        measurement = _measurement_from_row(row)
        if measurement is not None:
            measurements.append(measurement)
    return measurements


def group_measurements_by_timestamp(
    measurements: list[PixelMeasurement],
) -> dict[float, dict[str, PixelMeasurement]]:
    grouped: dict[float, dict[str, PixelMeasurement]] = {}
    for measurement in measurements:
        grouped.setdefault(measurement.timestamp, {})[measurement.camera_id] = measurement
    return grouped


def build_timestamp_observations(
    grouped: dict[float, dict[str, PixelMeasurement]],
    *,
    active_cameras: list[str] | None = None,
    min_cameras: int = 2,
) -> list[TimestampObservations]:
    """Keep timestamps with at least ``min_cameras`` active observations."""
    observations: list[TimestampObservations] = []
    for timestamp in sorted(grouped.keys()):
        available = grouped[timestamp]
        camera_ids = active_cameras if active_cameras is not None else sorted(available.keys())
        selected = [available[camera_id] for camera_id in camera_ids if camera_id in available]
        if len(selected) < min_cameras:
            continue
        observations.append(
            TimestampObservations(
                timestamp=timestamp,
                measurements=tuple(selected),
            )
        )
    return observations


def load_ground_truth_positions(path: str) -> dict[float, np.ndarray]:
    rows = load_ground_truth_csv(path)
    return {
        float(row["timestamp"]): np.array(
            [float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])],
            dtype=float,
        )
        for row in rows
    }


def collect_camera_correspondences(
    grouped: dict[float, dict[str, PixelMeasurement]],
    reference_camera_id: str,
    target_camera_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect synchronized pixel pairs between two cameras."""
    points_ref: list[list[float]] = []
    points_tgt: list[list[float]] = []
    for measurements in grouped.values():
        if reference_camera_id not in measurements or target_camera_id not in measurements:
            continue
        ref = measurements[reference_camera_id]
        tgt = measurements[target_camera_id]
        points_ref.append([ref.u, ref.v])
        points_tgt.append([tgt.u, tgt.v])
    if not points_ref:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.asarray(points_ref, dtype=float), np.asarray(points_tgt, dtype=float)
