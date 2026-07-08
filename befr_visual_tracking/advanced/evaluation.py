"""Evaluation metrics for advanced calibration results."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from befr_visual_tracking.advanced.bundle_adjustment import project_point
from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras, apply_gauge_fix_to_points
from befr_visual_tracking.advanced.observations import TimestampObservations
from befr_visual_tracking.camera_model import Camera


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class CalibrationEvaluation:
    camera_translation_rmse: float
    camera_rotation_rmse_deg: float
    trajectory_rmse: float
    reprojection_rmse: float
    trajectory_rmse_aligned: float
    camera_translation_rmse_aligned: float
    scale_estimate: float
    num_timesteps: int
    num_cameras: int


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Estimate Sim(3) alignment mapping source -> target."""
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError("Source and target must have the same shape")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target

    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    variance = np.sum(source_centered**2) / source.shape[0]
    scale = float(np.trace(np.diag(singular) @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])) / variance)
    translation = mu_target - scale * rotation @ mu_source
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def apply_similarity(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return (transform.scale * (transform.rotation @ points.T)).T + transform.translation


def rotation_rmse_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    delta = Rotation.from_matrix(R_gt.T @ R_est)
    return float(np.degrees(np.linalg.norm(delta.as_rotvec())))


def evaluate_calibration(
    estimated_cameras: dict[str, Camera],
    ground_truth_cameras: dict[str, Camera],
    estimated_positions: np.ndarray,
    ground_truth_positions: np.ndarray,
    observations: list[TimestampObservations],
    *,
    reference_camera_id: str = "camera_0",
) -> CalibrationEvaluation:
    """Compare estimated calibration against gauge-fixed ground truth."""
    gt_fixed = apply_gauge_fix_to_cameras(ground_truth_cameras, reference_camera_id)
    est_ids = sorted(camera_id for camera_id in estimated_cameras if camera_id != reference_camera_id)

    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for camera_id in est_ids:
        gt = gt_fixed[camera_id]
        est = estimated_cameras[camera_id]
        translation_errors.append(
            float(np.linalg.norm(est.extrinsics.t_camera_to_world - gt.extrinsics.t_camera_to_world))
        )
        rotation_errors.append(
            rotation_rmse_deg(est.extrinsics.R_camera_to_world, gt.extrinsics.R_camera_to_world)
        )

    gt_positions = apply_gauge_fix_to_points(
        ground_truth_positions,
        ground_truth_cameras[reference_camera_id].extrinsics,
    )
    estimated_positions = np.asarray(estimated_positions, dtype=float).reshape(-1, 3)
    trajectory_rmse = float(np.sqrt(np.mean(np.sum((estimated_positions - gt_positions) ** 2, axis=1))))

    sim = umeyama_similarity(estimated_positions, gt_positions)
    aligned_positions = apply_similarity(estimated_positions, sim)
    trajectory_rmse_aligned = float(
        np.sqrt(np.mean(np.sum((aligned_positions - gt_positions) ** 2, axis=1)))
    )

    aligned_translation_errors: list[float] = []
    for camera_id in est_ids:
        gt = gt_fixed[camera_id]
        est = estimated_cameras[camera_id]
        aligned_t = apply_similarity(est.extrinsics.t_camera_to_world.reshape(1, 3), sim)[0]
        aligned_translation_errors.append(float(np.linalg.norm(aligned_t - gt.extrinsics.t_camera_to_world)))

    reprojection_errors: list[float] = []
    intrinsics_k = next(iter(estimated_cameras.values())).intrinsics.K
    for time_index, observation in enumerate(observations):
        point = estimated_positions[time_index]
        for measurement in observation.measurements:
            camera = estimated_cameras[measurement.camera_id]
            R = camera.extrinsics.R_world_to_camera
            t = camera.extrinsics.t_world_to_camera
            projected = project_point(R, t, intrinsics_k, point)
            if np.all(np.isfinite(projected)):
                error = projected - np.array([measurement.u, measurement.v], dtype=float)
                reprojection_errors.append(float(np.linalg.norm(error)))

    reprojection_rmse = float(np.sqrt(np.mean(np.square(reprojection_errors)))) if reprojection_errors else float(
        "inf"
    )

    return CalibrationEvaluation(
        camera_translation_rmse=float(np.sqrt(np.mean(np.square(translation_errors))))
        if translation_errors
        else 0.0,
        camera_rotation_rmse_deg=float(np.sqrt(np.mean(np.square(rotation_errors))))
        if rotation_errors
        else 0.0,
        trajectory_rmse=trajectory_rmse,
        reprojection_rmse=reprojection_rmse,
        trajectory_rmse_aligned=trajectory_rmse_aligned,
        camera_translation_rmse_aligned=float(np.sqrt(np.mean(np.square(aligned_translation_errors))))
        if aligned_translation_errors
        else 0.0,
        scale_estimate=sim.scale,
        num_timesteps=len(estimated_positions),
        num_cameras=len(est_ids) + 1,
    )


def evaluation_to_dict(evaluation: CalibrationEvaluation) -> dict:
    return asdict(evaluation)
