"""Run advanced calibration on a recorded dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from befr_visual_tracking.advanced.calibration_pipeline import run_calibration_pipeline
from befr_visual_tracking.advanced.tracking_comparison import (
    compare_tracking_with_calibrated_cameras,
    save_tracking_comparison_report,
)
from befr_visual_tracking.generate_canonical_dataset import default_results_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run batch joint calibration (Phase 11).")
    parser.add_argument(
        "experiment_dir",
        help="Path to a recorded experiment folder (contains detections.csv)",
    )
    parser.add_argument(
        "--active-cameras",
        nargs="*",
        default=None,
        help="Camera IDs to use (default: metadata active_cameras)",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.05,
        help="Trajectory second-difference regularisation weight",
    )
    parser.add_argument(
        "--use-ideal-detections",
        action="store_true",
        help="Optimise against detections_ideal.csv instead of noisy detections",
    )
    parser.add_argument(
        "--metric-scale-from-gt",
        action="store_true",
        help="Use simulation baseline to set essential-matrix translation scale",
    )
    parser.add_argument(
        "--no-opencv-init",
        action="store_true",
        help="Skip OpenCV essential-matrix initialisation",
    )
    parser.add_argument(
        "--compare-tracking",
        action="store_true",
        help="After calibration, compare KF/EKF/UKF with true vs calibrated poses",
    )
    args = parser.parse_args(argv)

    result = run_calibration_pipeline(
        args.experiment_dir,
        active_cameras=args.active_cameras,
        smoothness_weight=args.smoothness_weight,
        use_opencv_init=not args.no_opencv_init,
        use_metric_scale_from_ground_truth=args.metric_scale_from_gt,
        use_ideal_detections=args.use_ideal_detections,
    )

    print(f"Calibrated cameras saved to: {result.calibrated_config_path}")
    print("Before optimisation:")
    print(f"  trajectory RMSE (aligned): {result.before_optimization.trajectory_rmse_aligned:.4f} m")
    print(f"  camera translation RMSE:   {result.before_optimization.camera_translation_rmse:.4f} m")
    print("After optimisation:")
    print(f"  trajectory RMSE (aligned): {result.after_optimization.trajectory_rmse_aligned:.4f} m")
    print(f"  camera translation RMSE:   {result.after_optimization.camera_translation_rmse:.4f} m")
    print(f"  reprojection RMSE:         {result.after_optimization.reprojection_rmse:.4f} px")
    print(f"  estimated scale factor:    {result.after_optimization.scale_estimate:.4f}")

    if args.compare_tracking:
        comparisons = compare_tracking_with_calibrated_cameras(
            args.experiment_dir,
            active_cameras=result.active_cameras,
        )
        report_path = save_tracking_comparison_report(args.experiment_dir, comparisons)
        print(f"Tracking comparison report: {report_path}")
        for item in comparisons:
            print(
                f"  {item.filter_type} / {item.camera_source}: "
                f"position RMSE={item.position_rmse:.4f} m"
            )


if __name__ == "__main__":
    main()
