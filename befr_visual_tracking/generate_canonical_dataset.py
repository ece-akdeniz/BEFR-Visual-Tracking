"""Generate and save the canonical offline dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.dataset_generator import generate_detection_rows
from befr_visual_tracking.result_writer import load_experiment_config, save_canonical_dataset
from befr_visual_tracking.trajectory_design import build_trajectory, sample_trajectory


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def default_experiments_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "experiments.yaml"


def build_trajectory_from_experiment(experiment: dict):
    trajectory_name = experiment.get("trajectory", "slow_circle")
    kwargs = {
        key: experiment[key]
        for key in (
            "center",
            "radius",
            "angular_rate",
            "sample_rate_hz",
            "duration",
            "start",
            "end",
            "scale_x",
            "scale_y",
            "vertical_amplitude",
        )
        if key in experiment
    }
    if "duration" in kwargs and kwargs["duration"] in (None, "null"):
        kwargs["duration"] = None
    if "center" in kwargs:
        kwargs["center"] = tuple(kwargs["center"])
    if "start" in kwargs:
        kwargs["start"] = tuple(kwargs["start"])
    if "end" in kwargs:
        kwargs["end"] = tuple(kwargs["end"])
    return build_trajectory(trajectory_name, **kwargs)


def generate_canonical_dataset(
    *,
    experiment_name: str = "canonical_slow_circle",
    experiments_path: str | Path | None = None,
    cameras_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    experiments_file = Path(experiments_path or default_experiments_path())
    experiment_cfg = load_experiment_config(experiments_file)[experiment_name]

    cameras_path = Path(cameras_config_path or default_cameras_yaml_path())
    system_config, cameras = cameras_from_yaml(cameras_path)

    trajectory = build_trajectory_from_experiment(experiment_cfg)
    samples = sample_trajectory(trajectory)
    random_seed = int(experiment_cfg.get("random_seed", 42))
    dataset = generate_detection_rows(
        cameras=cameras,
        trajectory_samples=samples,
        random_seed=random_seed,
    )

    output_name = experiment_cfg.get("output_name", experiment_name)
    destination = Path(output_dir or default_results_dir()) / output_name

    metadata = {
        "experiment": experiment_name,
        "trajectory": trajectory.name,
        "duration_s": trajectory.duration,
        "sample_rate_hz": trajectory.sample_rate_hz,
        "source": "offline_generator",
        "active_cameras": [camera.name for camera in cameras],
        "pixel_noise_std": system_config.noise.sigma_u,
        "dropout_scenario": "none",
    }

    return save_canonical_dataset(
        destination,
        dataset,
        metadata=metadata,
        camera_config_path=cameras_path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the canonical offline dataset.")
    parser.add_argument(
        "--experiment",
        default="canonical_slow_circle",
        help="Experiment name from config/experiments.yaml",
    )
    parser.add_argument(
        "--experiments-config",
        default=str(default_experiments_path()),
        help="Path to experiments.yaml",
    )
    parser.add_argument(
        "--cameras-config",
        default=str(default_cameras_yaml_path()),
        help="Path to cameras.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_results_dir()),
        help="Base directory for saved experiment folders",
    )
    args = parser.parse_args(argv)

    output_path = generate_canonical_dataset(
        experiment_name=args.experiment,
        experiments_path=args.experiments_config,
        cameras_config_path=args.cameras_config,
        output_dir=args.output_dir,
    )
    print(f"Canonical dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
