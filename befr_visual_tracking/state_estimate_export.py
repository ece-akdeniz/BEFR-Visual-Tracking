"""Pure-Python export of tracker state for ROS, Streamlit, and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from befr_visual_tracking.tracker_base import TrackerBase


@dataclass(frozen=True)
class StateEstimateExport:
    """Position, velocity, and covariance blocks from a tracker."""

    position: np.ndarray
    velocity: np.ndarray
    pose_covariance: np.ndarray
    twist_covariance: np.ndarray


def state_estimate_from_tracker(
    tracker: TrackerBase,
    *,
    orientation_covariance_value: float = 1e6,
) -> StateEstimateExport:
    """Extract state estimate arrays from the current tracker state."""
    state = tracker.get_state()
    covariance = tracker.get_covariance()

    pose_covariance = np.zeros(36, dtype=float)
    pose_covariance[0] = float(covariance[0, 0])
    pose_covariance[7] = float(covariance[1, 1])
    pose_covariance[14] = float(covariance[2, 2])
    pose_covariance[21] = orientation_covariance_value
    pose_covariance[28] = orientation_covariance_value
    pose_covariance[35] = orientation_covariance_value

    twist_covariance = np.zeros(36, dtype=float)
    twist_covariance[0] = float(covariance[3, 3])
    twist_covariance[7] = float(covariance[4, 4])
    twist_covariance[14] = float(covariance[5, 5])

    return StateEstimateExport(
        position=np.asarray(state[:3], dtype=float),
        velocity=np.asarray(state[3:], dtype=float),
        pose_covariance=pose_covariance,
        twist_covariance=twist_covariance,
    )
