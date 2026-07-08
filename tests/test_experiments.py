"""Tests for the controlled experiment framework."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from befr_visual_tracking.experiments.metrics import StepRecord, compute_tracking_metrics
from befr_visual_tracking.experiments.runner import run_experiment
from befr_visual_tracking.result_writer import allocate_experiment_dir


class TestExperimentMetrics(unittest.TestCase):
    def test_rmse_from_steps(self) -> None:
        steps = [
            StepRecord(
                timestamp=0.0,
                state=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                covariance=__import__("numpy").eye(6),
                ground_truth_position=__import__("numpy").zeros(3),
                ground_truth_velocity=__import__("numpy").zeros(3),
                num_visible_cameras=3,
                update_time_sec=0.001,
            )
        ]
        metrics = compute_tracking_metrics(steps)
        self.assertAlmostEqual(metrics.position_rmse, 1.0)


class TestExperimentRunner(unittest.TestCase):
    def test_experiment_zero_writes_standard_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = run_experiment(
                0,
                results_root=temp_dir,
                filters=["ekf"],
                variants=["sanity"],
            )
            self.assertEqual(len(outputs), 1)
            output = outputs[0]
            for filename in (
                "metadata.json",
                "detections.csv",
                "estimates.csv",
                "ground_truth.csv",
                "camera_config.yaml",
                "metrics.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["experiment_id"], 0)
            self.assertEqual(metadata["filter"], "ekf")

            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertLess(metrics["position_rmse"], 0.05)
            self.assertLess(metrics["triangulation_reprojection_rmse"], 1e-3)

    def test_allocate_experiment_dir_increments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = allocate_experiment_dir(temp_dir)
            second = allocate_experiment_dir(temp_dir)
            self.assertEqual(first.name, "experiment_001")
            self.assertEqual(second.name, "experiment_002")


if __name__ == "__main__":
    unittest.main()
