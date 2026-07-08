"""End-to-end advanced calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from befr_visual_tracking.advanced.bundle_adjustment import (
    build_problem,
    bundle_adjustment_to_cameras,
    run_bundle_adjustment,
)
from befr_visual_tracking.advanced.calibrated_config import save_calibrated_cameras
from befr_visual_tracking.advanced.evaluation import CalibrationEvaluation, evaluate_calibration, evaluation_to_dict
from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras, apply_gauge_fix_to_points
from befr_visual_tracking.advanced.initial_pose_estimation import (
    estimate_initial_camera_poses,
    infer_translation_scale_from_ground_truth,
    relative_pose_estimates_to_cameras,
)
from befr_visual_tracking.advanced.observations import (
    build_timestamp_observations,
    group_measurements_by_timestamp,
    load_ground_truth_positions,
    load_visible_measurements,
)
from befr_visual_tracking.advanced.trajectory_initialization import (
    initialize_trajectory,
    interpolate_missing_positions,
)
from befr_visual_tracking.camera_config import load_camera_system_config
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.result_writer import load_metadata_json, write_metadata_json


@dataclass
class CalibrationPipelineResult:
    experiment_dir: Path
    calibrated_config_path: Path
    before_optimization: CalibrationEvaluation
    after_optimization: CalibrationEvaluation
    bundle_adjustment_success: bool
    bundle_adjustment_reprojection_rmse: float
    active_cameras: list[str]


def run_calibration_pipeline(
    experiment_dir: str | Path,
    *,
    active_cameras: list[str] | None = None,
    reference_camera_id: str = "camera_0",
    smoothness_weight: float = 0.05,
    use_opencv_init: bool = True,
    use_metric_scale_from_ground_truth: bool = False,
    max_nfev: int = 200,
    use_ideal_detections: bool = False,
    subsample_stride: int = 1,
    max_timesteps: int | None = None,
) -> CalibrationPipelineResult:
    """
    Run initial pose estimation, trajectory initialization, and bundle adjustment.

    When ``use_metric_scale_from_ground_truth`` is True the first non-reference
    camera baseline from simulation is used to set essential-matrix translation
    scale. Otherwise translation direction is unit-normalised and evaluation
    should use similarity alignment to account for global scale ambiguity.
    """
    experiment_path = Path(experiment_dir)
    camera_config_path = experiment_path / "camera_config.yaml"
    detections_name = "detections_ideal.csv" if use_ideal_detections else "detections.csv"
    detections_path = experiment_path / detections_name

    _, ground_truth_cameras = cameras_from_yaml(camera_config_path)
    ground_truth_by_id = {camera.name: camera for camera in ground_truth_cameras}
    gt_gauge = apply_gauge_fix_to_cameras(ground_truth_by_id, reference_camera_id)

    if active_cameras is None:
        metadata = load_metadata_json(experiment_path / "metadata.json")
        active_cameras = list(metadata.get("active_cameras", sorted(ground_truth_by_id.keys())))

    measurements = load_visible_measurements(detections_path)
    grouped = group_measurements_by_timestamp(measurements)
    observations = build_timestamp_observations(grouped, active_cameras=active_cameras)
    if subsample_stride > 1:
        observations = observations[::subsample_stride]
    if max_timesteps is not None:
        observations = observations[:max_timesteps]

    translation_scale = None
    if use_metric_scale_from_ground_truth:
        other_ids = [cid for cid in active_cameras if cid != reference_camera_id]
        if other_ids:
            translation_scale = infer_translation_scale_from_ground_truth(
                ground_truth_by_id[reference_camera_id],
                ground_truth_by_id[other_ids[0]],
            )

    pose_estimates = estimate_initial_camera_poses(
        gt_gauge,
        grouped,
        reference_camera_id=reference_camera_id,
        active_cameras=active_cameras,
        translation_scale=translation_scale,
        use_opencv=use_opencv_init,
    )
    initial_cameras = relative_pose_estimates_to_cameras(pose_estimates, gt_gauge)

    trajectory_init = initialize_trajectory(observations, initial_cameras)
    initial_positions = interpolate_missing_positions(trajectory_init)

    gt_positions_dict = load_ground_truth_positions(experiment_path / "ground_truth.csv")
    gt_positions = np.array(
        [gt_positions_dict[ts] for ts in trajectory_init.timestamps],
        dtype=float,
    )
    gt_positions_gauge = apply_gauge_fix_to_points(
        gt_positions,
        ground_truth_by_id[reference_camera_id].extrinsics,
    )

    before = evaluate_calibration(
        initial_cameras,
        ground_truth_by_id,
        initial_positions,
        gt_positions_gauge,
        observations,
        reference_camera_id=reference_camera_id,
    )

    problem = build_problem(
        observations,
        initial_cameras,
        initial_positions,
        reference_camera_id=reference_camera_id,
        active_cameras=active_cameras,
    )
    ba_result = run_bundle_adjustment(
        problem,
        smoothness_weight=smoothness_weight,
        max_nfev=max_nfev,
    )
    calibrated_cameras = bundle_adjustment_to_cameras(
        ba_result,
        {camera_id: gt_gauge[camera_id] for camera_id in active_cameras},
        reference_camera_id=reference_camera_id,
    )

    after = evaluate_calibration(
        calibrated_cameras,
        ground_truth_by_id,
        ba_result.positions,
        gt_positions_gauge,
        observations,
        reference_camera_id=reference_camera_id,
    )

    calibrated_path = experiment_path / "calibrated_cameras.yaml"
    save_calibrated_cameras(
        calibrated_path,
        load_camera_system_config(camera_config_path),
        calibrated_cameras,
        reference_camera_id=reference_camera_id,
        fallback_cameras=gt_gauge,
        metadata={
            "source_experiment": str(experiment_path.name),
            "reference_camera_id": reference_camera_id,
            "active_cameras": active_cameras,
            "use_ideal_detections": use_ideal_detections,
            "use_metric_scale_from_ground_truth": use_metric_scale_from_ground_truth,
            "evaluation_before": evaluation_to_dict(before),
            "evaluation_after": evaluation_to_dict(after),
            "bundle_adjustment": {
                "success": ba_result.success,
                "cost": ba_result.cost,
                "nfev": ba_result.nfev,
                "reprojection_rmse": ba_result.reprojection_rmse,
            },
            "scale_ambiguity_note": (
                "Without a metric baseline constraint, reconstruction is defined up to "
                "a global similarity transform. Use evaluation *_aligned metrics and "
                "scale_estimate when comparing against simulation ground truth."
            ),
        },
    )

    report_path = experiment_path / "calibration_report.json"
    write_metadata_json(
        report_path,
        {
            "experiment": experiment_path.name,
            "active_cameras": active_cameras,
            "before_optimization": evaluation_to_dict(before),
            "after_optimization": evaluation_to_dict(after),
            "bundle_adjustment": {
                "success": ba_result.success,
                "cost": ba_result.cost,
                "nfev": ba_result.nfev,
                "reprojection_rmse": ba_result.reprojection_rmse,
            },
        },
    )

    return CalibrationPipelineResult(
        experiment_dir=experiment_path,
        calibrated_config_path=calibrated_path,
        before_optimization=before,
        after_optimization=after,
        bundle_adjustment_success=ba_result.success,
        bundle_adjustment_reprojection_rmse=ba_result.reprojection_rmse,
        active_cameras=active_cameras,
    )
