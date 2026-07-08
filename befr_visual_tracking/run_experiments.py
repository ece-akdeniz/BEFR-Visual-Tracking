"""Run controlled one-variable-at-a-time experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from befr_visual_tracking.experiments.registry import EXPERIMENT_PLANS, get_experiment_plan
from befr_visual_tracking.experiments.runner import default_results_dir, run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run controlled BEFR tracking experiments.")
    parser.add_argument("--list", action="store_true", help="List available experiment plans")
    parser.add_argument("--experiment", type=int, help="Experiment id (0-7, 101-104)")
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        help="Filter type override (kf, ekf, ukf). Repeat for multiple.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Run only the named variant(s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_results_dir()),
        help="Base results directory for experiment_NNN folders",
    )
    args = parser.parse_args(argv)

    if args.list:
        for experiment_id in sorted(EXPERIMENT_PLANS):
            plan = get_experiment_plan(experiment_id)
            print(f"{experiment_id:3d}  {plan.title}  [{plan.experiment_type}]")
        return

    if args.experiment is None:
        parser.error("--experiment is required unless --list is used")

    outputs = run_experiment(
        args.experiment,
        results_root=args.output_dir,
        filters=args.filters,
        variants=args.variants,
    )

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
