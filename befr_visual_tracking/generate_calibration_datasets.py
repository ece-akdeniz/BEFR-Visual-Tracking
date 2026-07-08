"""Generate calibration trajectory datasets for advanced Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from befr_visual_tracking.generate_canonical_dataset import (
    default_experiments_path,
    default_results_dir,
    generate_canonical_dataset,
)


CALIBRATION_EXPERIMENTS = [
    "calibration_stationary_hover",
    "calibration_straight_line",
    "calibration_planar_figure_eight",
    "calibration_figure_eight_3d",
]


def generate_calibration_datasets(
    *,
    experiments: list[str] | None = None,
    experiments_path: str | Path | None = None,
    cameras_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all calibration manoeuvre datasets."""
    names = experiments or CALIBRATION_EXPERIMENTS
    outputs: list[Path] = []
    for experiment_name in names:
        outputs.append(
            generate_canonical_dataset(
                experiment_name=experiment_name,
                experiments_path=experiments_path,
                cameras_config_path=cameras_config_path,
                output_dir=output_dir,
            )
        )
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration trajectory datasets for advanced Task 1."
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=CALIBRATION_EXPERIMENTS,
        help="Experiment names from config/experiments.yaml",
    )
    parser.add_argument(
        "--experiments-config",
        default=str(default_experiments_path()),
    )
    parser.add_argument(
        "--cameras-config",
        default=None,
        help="Optional override for cameras.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_results_dir()),
    )
    args = parser.parse_args(argv)

    outputs = generate_calibration_datasets(
        experiments=args.experiments,
        experiments_path=args.experiments_config,
        cameras_config_path=args.cameras_config,
        output_dir=args.output_dir,
    )
    for output in outputs:
        print(f"Calibration dataset saved to: {output}")


if __name__ == "__main__":
    main()
