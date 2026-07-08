"""Tests for dashboard offline rerun."""

from __future__ import annotations

import unittest

from dashboard.lib.offline_rerun import InteractiveRunConfig, run_interactive_tracking


class TestDashboardOfflineRerun(unittest.TestCase):
    def test_interactive_ekf_runs_quickly(self) -> None:
        result = run_interactive_tracking(
            InteractiveRunConfig(
                filter_type="ekf",
                trajectory="circle",
                active_cameras=["camera_0", "camera_1", "camera_2"],
                pixel_noise_std=1.0,
                sample_rate_hz=15.0,
            )
        )
        self.assertGreater(len(result.replay.steps), 10)
        self.assertLess(result.metrics["position_rmse"], 0.5)


if __name__ == "__main__":
    unittest.main()
