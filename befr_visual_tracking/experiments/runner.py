"""Run controlled experiments and save standard result folders."""

from __future__ import annotations

from pathlib import Path

from befr_visual_tracking.advanced.calibration_pipeline import run_calibration_pipeline
from befr_visual_tracking.advanced.evaluation import evaluation_to_dict
from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras
from befr_visual_tracking.advanced.initial_pose_estimation import (
    estimate_initial_camera_poses,
    relative_pose_estimates_to_cameras,
)
from befr_visual_tracking.advanced.observations import (
    group_measurements_by_timestamp,
    load_visible_measurements,
)
from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.experiments.dataset_factory import DatasetSpec, generate_dataset_from_spec
from befr_visual_tracking.experiments.metrics import compute_tracking_metrics, tracking_metrics_to_dict
from befr_visual_tracking.experiments.registry import ExperimentPlan, ExperimentVariant, get_experiment_plan
from befr_visual_tracking.experiments.replay import (
    load_detections_for_replay,
    load_ground_truth_for_replay,
    replay_tracker_on_detections,
)
from befr_visual_tracking.result_writer import allocate_experiment_dir, save_tracking_experiment
from befr_visual_tracking.triangulation import PixelObservation, triangulate_from_observations


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def _build_dataset_spec(variant: ExperimentVariant, plan: ExperimentPlan) -> DatasetSpec:
    defaults = {
        "trajectory": "slow_circle",
        "pixel_noise_std": 1.0,
        "random_seed": 42,
        "cameras_config_path": default_cameras_yaml_path(),
    }
    if plan.experiment_id in {101, 102, 103} and "trajectory" not in variant.dataset_overrides:
        defaults["trajectory"] = "calibration_figure_eight_3d"
    defaults.update(variant.dataset_overrides)
    return DatasetSpec(**defaults)


def _create_shared_dataset(plan: ExperimentPlan, variant: ExperimentVariant, results_root: Path) -> Path:
    dataset_spec = _build_dataset_spec(variant, plan)
    dataset, _, trajectory = generate_dataset_from_spec(dataset_spec)
    dataset_dir = allocate_experiment_dir(results_root)
    save_tracking_experiment(
        dataset_dir,
        dataset=dataset,
        metadata={
            "experiment_id": plan.experiment_id,
            "experiment_title": plan.title,
            "stage": "shared_dataset",
            "trajectory": trajectory.name,
            "pixel_noise_std": dataset_spec.pixel_noise_std,
        },
        steps=[],
        metrics={},
        camera_config_path=Path(dataset_spec.cameras_config_path or default_cameras_yaml_path()),
    )
    return dataset_dir


def _triangulation_sanity_rmse(cameras_by_id: dict, experiment_dir: Path) -> float:
    detections = load_visible_measurements(experiment_dir / "detections.csv")
    grouped = group_measurements_by_timestamp(detections)
    if not grouped:
        return float("inf")
    timestamp = sorted(grouped.keys())[0]
    observations = [
        PixelObservation(camera_id=camera_id, u=measurement.u, v=measurement.v)
        for camera_id, measurement in grouped[timestamp].items()
    ]
    result = triangulate_from_observations(cameras_by_id, observations)
    if result.point is None:
        return float("inf")
    return float(result.quality.reprojection_rmse)


def _run_tracking_variant(
    *,
    plan: ExperimentPlan,
    variant: ExperimentVariant,
    filter_type: str,
    results_root: Path,
    shared_dataset_dir: Path | None,
) -> Path:
    dataset_spec = _build_dataset_spec(variant, plan)
    cameras_config_path = Path(dataset_spec.cameras_config_path or default_cameras_yaml_path())
    _, cameras = cameras_from_yaml(cameras_config_path)
    cameras_by_id = {camera.name: camera for camera in cameras}

    dataset = None
    source_dataset_dir = shared_dataset_dir
    if source_dataset_dir is None:
        dataset, cameras, _trajectory = generate_dataset_from_spec(dataset_spec)
        cameras_by_id = {camera.name: camera for camera in cameras}
        replay_source = None
    else:
        replay_source = source_dataset_dir

    detections_by_time = None
    ground_truth_by_time = None
    if replay_source is not None:
        detections_by_time = load_detections_for_replay(
            replay_source,
            use_ideal_detections=dataset_spec.use_zero_noise,
        )
        ground_truth_by_time = load_ground_truth_for_replay(replay_source)
    else:
        assert dataset is not None
        from befr_visual_tracking.kalman_tracker import group_detections_by_timestamp, load_ground_truth_by_timestamp
        from befr_visual_tracking.result_writer import write_detections_csv, write_ground_truth_csv
        import tempfile

        temp_root = Path(tempfile.mkdtemp(prefix="befr_replay_"))
        write_detections_csv(temp_root / "detections.csv", dataset.detections)
        write_ground_truth_csv(temp_root / "ground_truth.csv", dataset.ground_truth)
        replay_source = temp_root
        detections_by_time = load_detections_for_replay(
            replay_source,
            use_ideal_detections=dataset_spec.use_zero_noise,
        )
        ground_truth_by_time = load_ground_truth_by_timestamp(temp_root / "ground_truth.csv")

    tracking_cfg = {
        "active_cameras": ["camera_0", "camera_1", "camera_2"],
        "acceleration_noise_std": 0.5,
        "pixel_noise_std": dataset_spec.pixel_noise_std,
        "innovation_gate": 9.21,
        "random_seed": dataset_spec.random_seed,
    }
    tracking_cfg.update(variant.tracking_overrides)

    replay = replay_tracker_on_detections(
        filter_type=filter_type,
        cameras_by_id=cameras_by_id,
        detections_by_time=detections_by_time,
        ground_truth_by_time=ground_truth_by_time,
        active_cameras=tracking_cfg["active_cameras"],
        acceleration_noise_std=tracking_cfg["acceleration_noise_std"],
        pixel_noise_std=tracking_cfg["pixel_noise_std"],
        innovation_gate=tracking_cfg["innovation_gate"],
        random_seed=tracking_cfg["random_seed"],
    )

    metrics = tracking_metrics_to_dict(
        compute_tracking_metrics(replay.steps, tracker_stats=replay.tracker_stats)
    )
    metrics["triangulation_reprojection_rmse"] = _triangulation_sanity_rmse(cameras_by_id, replay_source)

    metadata = {
        "experiment_id": plan.experiment_id,
        "experiment_title": plan.title,
        "variant": variant.name,
        "filter": filter_type,
        "trajectory": dataset_spec.trajectory,
        "active_cameras": tracking_cfg["active_cameras"],
        "pixel_noise_std": dataset_spec.pixel_noise_std,
        "dropout_scenario": variant.metadata_overrides.get("dropout_scenario", "none"),
        "process_noise_std": tracking_cfg["acceleration_noise_std"],
        "experiment_type": plan.experiment_type,
    }
    metadata.update(variant.metadata_overrides)

    output_dir = allocate_experiment_dir(results_root)
    return save_tracking_experiment(
        output_dir,
        dataset=dataset if shared_dataset_dir is None else None,
        metadata=metadata,
        steps=replay.steps,
        metrics=metrics,
        camera_config_path=cameras_config_path,
        source_dataset_dir=shared_dataset_dir,
    )


def _run_calibration_variant(
    *,
    plan: ExperimentPlan,
    variant: ExperimentVariant,
    results_root: Path,
) -> Path:
    dataset_spec = _build_dataset_spec(variant, plan)
    dataset, _, trajectory = generate_dataset_from_spec(dataset_spec)
    cameras_config_path = Path(dataset_spec.cameras_config_path or default_cameras_yaml_path())

    dataset_dir = allocate_experiment_dir(results_root)
    save_tracking_experiment(
        dataset_dir,
        dataset=dataset,
        metadata={
            "experiment_id": plan.experiment_id,
            "experiment_title": plan.title,
            "variant": variant.name,
            "trajectory": trajectory.name,
            "pixel_noise_std": dataset_spec.pixel_noise_std,
            "experiment_type": "calibration",
        },
        steps=[],
        metrics={},
        camera_config_path=cameras_config_path,
    )

    active_cameras = variant.metadata_overrides.get(
        "active_cameras",
        ["camera_0", "camera_1", "camera_2"],
    )
    calibration = run_calibration_pipeline(
        dataset_dir,
        active_cameras=list(active_cameras),
        use_opencv_init=False,
        use_metric_scale_from_ground_truth=True,
        use_ideal_detections=False,
        subsample_stride=5,
        max_timesteps=80,
    )

    metrics = {
        "calibration_before": evaluation_to_dict(calibration.before_optimization),
        "calibration_after": evaluation_to_dict(calibration.after_optimization),
        "bundle_adjustment_reprojection_rmse": calibration.bundle_adjustment_reprojection_rmse,
    }
    output_dir = allocate_experiment_dir(results_root)
    return save_tracking_experiment(
        output_dir,
        dataset=None,
        metadata={
            "experiment_id": plan.experiment_id,
            "experiment_title": plan.title,
            "variant": variant.name,
            "trajectory": trajectory.name,
            "active_cameras": active_cameras,
            "pixel_noise_std": dataset_spec.pixel_noise_std,
            "experiment_type": "calibration",
            "calibrated_config_path": str(calibration.calibrated_config_path),
        },
        steps=[],
        metrics=metrics,
        camera_config_path=calibration.calibrated_config_path,
        source_dataset_dir=dataset_dir,
    )


def _camera_pose_source_for_advanced_tracking(
    source: str,
    *,
    experiment_dir: Path,
    active_cameras: list[str],
) -> dict:
    _, true_cameras = cameras_from_yaml(experiment_dir / "camera_config.yaml")
    true_by_id = {camera.name: camera for camera in true_cameras}
    gt_gauge = apply_gauge_fix_to_cameras(true_by_id)
    if source == "true":
        return {camera_id: gt_gauge[camera_id] for camera_id in active_cameras if camera_id in gt_gauge}

    measurements = load_visible_measurements(experiment_dir / "detections.csv")
    grouped = group_measurements_by_timestamp(measurements)
    estimates = estimate_initial_camera_poses(
        gt_gauge,
        grouped,
        reference_camera_id="camera_0",
        active_cameras=active_cameras,
        use_opencv=False,
        translation_scale=1.0,
    )
    initial = relative_pose_estimates_to_cameras(estimates, gt_gauge)
    if source == "initial":
        return initial

    calibration = run_calibration_pipeline(
        experiment_dir,
        active_cameras=active_cameras,
        use_opencv_init=False,
        use_metric_scale_from_ground_truth=True,
        subsample_stride=5,
        max_timesteps=80,
    )
    _, calibrated = cameras_from_yaml(calibration.calibrated_config_path)
    return {camera.name: camera for camera in calibrated}


def _run_advanced_tracking_variant(
    *,
    plan: ExperimentPlan,
    variant: ExperimentVariant,
    filter_type: str,
    results_root: Path,
    shared_dataset_dir: Path,
) -> Path:
    active_cameras = ["camera_0", "camera_1", "camera_2"]
    pose_source = variant.metadata_overrides.get("camera_pose_source", "true")
    cameras_by_id = _camera_pose_source_for_advanced_tracking(
        pose_source,
        experiment_dir=shared_dataset_dir,
        active_cameras=active_cameras,
    )

    detections_by_time = load_detections_for_replay(shared_dataset_dir)
    ground_truth_by_time = load_ground_truth_for_replay(shared_dataset_dir)
    replay = replay_tracker_on_detections(
        filter_type=filter_type,
        cameras_by_id=cameras_by_id,
        detections_by_time=detections_by_time,
        ground_truth_by_time=ground_truth_by_time,
        active_cameras=active_cameras,
    )
    metrics = tracking_metrics_to_dict(
        compute_tracking_metrics(replay.steps, tracker_stats=replay.tracker_stats)
    )
    metrics["camera_pose_source"] = pose_source

    output_dir = allocate_experiment_dir(results_root)
    return save_tracking_experiment(
        output_dir,
        dataset=None,
        metadata={
            "experiment_id": plan.experiment_id,
            "experiment_title": plan.title,
            "variant": variant.name,
            "filter": filter_type,
            "camera_pose_source": pose_source,
            "active_cameras": active_cameras,
            "experiment_type": plan.experiment_type,
        },
        steps=replay.steps,
        metrics=metrics,
        camera_config_path=shared_dataset_dir / "camera_config.yaml",
        source_dataset_dir=shared_dataset_dir,
    )


def run_experiment(
    experiment_id: int,
    *,
    results_root: str | Path | None = None,
    filters: list[str] | None = None,
    variants: list[str] | None = None,
) -> list[Path]:
    """Execute all variants (and filters) for one experiment plan."""
    plan = get_experiment_plan(experiment_id)
    root = Path(results_root or default_results_dir())
    selected_variants = [
        variant for variant in plan.variants if variants is None or variant.name in variants
    ]
    selected_filters = list(filters or plan.filters or ("ekf",))
    outputs: list[Path] = []
    shared_dataset_dir: Path | None = None

    if plan.shared_dataset and selected_variants:
        shared_dataset_dir = _create_shared_dataset(plan, selected_variants[0], root)

    if plan.experiment_type == "advanced_tracking" and shared_dataset_dir is None:
        shared_dataset_dir = _create_shared_dataset(
            plan,
            ExperimentVariant(name="shared", dataset_overrides={"trajectory": "calibration_figure_eight_3d"}),
            root,
        )

    for variant in selected_variants:
        if plan.experiment_type == "calibration":
            outputs.append(_run_calibration_variant(plan=plan, variant=variant, results_root=root))
            continue

        if plan.experiment_type == "advanced_tracking":
            assert shared_dataset_dir is not None
            for filter_type in selected_filters:
                outputs.append(
                    _run_advanced_tracking_variant(
                        plan=plan,
                        variant=variant,
                        filter_type=filter_type,
                        results_root=root,
                        shared_dataset_dir=shared_dataset_dir,
                    )
                )
            continue

        for filter_type in selected_filters:
            outputs.append(
                _run_tracking_variant(
                    plan=plan,
                    variant=variant,
                    filter_type=filter_type,
                    results_root=root,
                    shared_dataset_dir=shared_dataset_dir,
                )
            )

    return outputs
