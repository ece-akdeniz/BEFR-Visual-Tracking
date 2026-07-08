"""Generate canonical detection datasets from trajectories and camera models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.camera_model import Camera
from befr_visual_tracking.trajectory_design import TrajectorySample


@dataclass(frozen=True)
class DetectionRow:
    timestamp: float
    camera_id: str
    u: float | None
    v: float | None
    u_ideal: float | None
    v_ideal: float | None
    visible: bool
    gt_x: float
    gt_y: float
    gt_z: float
    gt_vx: float
    gt_vy: float
    gt_vz: float


@dataclass(frozen=True)
class GroundTruthRow:
    timestamp: float
    gt_x: float
    gt_y: float
    gt_z: float
    gt_vx: float
    gt_vy: float
    gt_vz: float


@dataclass(frozen=True)
class CanonicalDataset:
    detections: tuple[DetectionRow, ...]
    ground_truth: tuple[GroundTruthRow, ...]
    random_seed: int


def generate_detection_rows(
    *,
    cameras: list[Camera],
    trajectory_samples: list[TrajectorySample],
    random_seed: int = 42,
) -> CanonicalDataset:
    """
    Generate synchronized ideal and noisy detections for a trajectory.

    Each timestamp uses one shared time stamp for all cameras. Noisy detections
    follow the same camera.measure() path as the live ROS sensor simulator.
    """
    rng = np.random.default_rng(random_seed)
    detection_rows: list[DetectionRow] = []
    ground_truth_rows: list[GroundTruthRow] = []

    for sample in trajectory_samples:
        position = sample.position
        velocity = sample.velocity
        ground_truth_rows.append(
            GroundTruthRow(
                timestamp=sample.timestamp,
                gt_x=float(position[0]),
                gt_y=float(position[1]),
                gt_z=float(position[2]),
                gt_vx=float(velocity[0]),
                gt_vy=float(velocity[1]),
                gt_vz=float(velocity[2]),
            )
        )

        for camera in cameras:
            ideal = camera.project_world_point(position)
            visible = ideal is not None
            u_ideal = float(ideal[0]) if ideal is not None else None
            v_ideal = float(ideal[1]) if ideal is not None else None

            measurement = camera.measure(position, sample.timestamp, rng=rng)
            if measurement is None:
                u_noisy = None
                v_noisy = None
            else:
                u_noisy = float(measurement.u)
                v_noisy = float(measurement.v)

            detection_rows.append(
                DetectionRow(
                    timestamp=sample.timestamp,
                    camera_id=camera.name,
                    u=u_noisy,
                    v=v_noisy,
                    u_ideal=u_ideal,
                    v_ideal=v_ideal,
                    visible=visible,
                    gt_x=float(position[0]),
                    gt_y=float(position[1]),
                    gt_z=float(position[2]),
                    gt_vx=float(velocity[0]),
                    gt_vy=float(velocity[1]),
                    gt_vz=float(velocity[2]),
                )
            )

    return CanonicalDataset(
        detections=tuple(detection_rows),
        ground_truth=tuple(ground_truth_rows),
        random_seed=random_seed,
    )
