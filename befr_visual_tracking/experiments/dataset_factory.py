"""Generate experiment datasets with controlled parameter changes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import Camera, CameraNoiseModel, cameras_from_yaml
from befr_visual_tracking.dataset_generator import CanonicalDataset, generate_detection_rows
from befr_visual_tracking.experiments.dropout import apply_dropout_to_detections
from befr_visual_tracking.result_writer import save_canonical_dataset
from befr_visual_tracking.trajectory_design import TrajectoryDefinition, build_trajectory, sample_trajectory


@dataclass(frozen=True)
class DatasetSpec:
    trajectory: str = "slow_circle"
    trajectory_kwargs: dict | None = None
    pixel_noise_std: float = 1.0
    pixel_noise_std_u: float | None = None
    pixel_noise_std_v: float | None = None
    quantize: bool = True
    random_seed: int = 42
    cameras_config_path: str | Path | None = None
    dropout_scenario: str | None = None
    use_zero_noise: bool = False


def _apply_noise_to_cameras(
    cameras: list[Camera],
    *,
    pixel_noise_std: float,
    pixel_noise_std_u: float | None,
    pixel_noise_std_v: float | None,
    quantize: bool,
) -> list[Camera]:
    sigma_u = pixel_noise_std_u if pixel_noise_std_u is not None else pixel_noise_std
    sigma_v = pixel_noise_std_v if pixel_noise_std_v is not None else pixel_noise_std
    modified: list[Camera] = []
    for camera in cameras:
        updated = copy.copy(camera)
        updated.noise_model = CameraNoiseModel(
            pixel_noise_std_u=sigma_u,
            pixel_noise_std_v=sigma_v,
            discretize=quantize,
            detection_probability=camera.noise_model.detection_probability,
        )
        modified.append(updated)
    return modified


def build_trajectory_from_spec(spec: DatasetSpec) -> TrajectoryDefinition:
    kwargs = dict(spec.trajectory_kwargs or {})
    return build_trajectory(spec.trajectory, **kwargs)


def generate_dataset_from_spec(spec: DatasetSpec) -> tuple[CanonicalDataset, list[Camera], TrajectoryDefinition]:
    cameras_path = Path(spec.cameras_config_path or default_cameras_yaml_path())
    _, cameras = cameras_from_yaml(cameras_path)

    noise_std = 0.0 if spec.use_zero_noise else spec.pixel_noise_std
    quantize = False if spec.use_zero_noise else spec.quantize
    cameras = _apply_noise_to_cameras(
        cameras,
        pixel_noise_std=noise_std,
        pixel_noise_std_u=spec.pixel_noise_std_u,
        pixel_noise_std_v=spec.pixel_noise_std_v,
        quantize=quantize,
    )

    trajectory = build_trajectory_from_spec(spec)
    samples = sample_trajectory(trajectory)
    dataset = generate_detection_rows(
        cameras=cameras,
        trajectory_samples=samples,
        random_seed=spec.random_seed,
    )

    if spec.dropout_scenario == "planned_sequence":
        camera_ids = [camera.name for camera in cameras]
        detections = apply_dropout_to_detections(dataset.detections, camera_ids=camera_ids)
        dataset = CanonicalDataset(
            detections=detections,
            ground_truth=dataset.ground_truth,
            random_seed=dataset.random_seed,
        )

    if spec.use_zero_noise:
        # Replace noisy pixels with ideal projections for a true zero-noise sanity test.
        rows = []
        for row in dataset.detections:
            rows.append(
                type(row)(
                    timestamp=row.timestamp,
                    camera_id=row.camera_id,
                    u=row.u_ideal,
                    v=row.v_ideal,
                    u_ideal=row.u_ideal,
                    v_ideal=row.v_ideal,
                    visible=row.visible,
                    gt_x=row.gt_x,
                    gt_y=row.gt_y,
                    gt_z=row.gt_z,
                    gt_vx=row.gt_vx,
                    gt_vy=row.gt_vy,
                    gt_vz=row.gt_vz,
                )
            )
        dataset = CanonicalDataset(
            detections=tuple(rows),
            ground_truth=dataset.ground_truth,
            random_seed=dataset.random_seed,
        )

    return dataset, cameras, trajectory


def save_dataset_to_directory(
    output_dir: str | Path,
    dataset: CanonicalDataset,
    *,
    metadata: dict,
    cameras_config_path: str | Path,
) -> Path:
    return save_canonical_dataset(
        output_dir,
        dataset,
        metadata=metadata,
        camera_config_path=cameras_config_path,
    )
