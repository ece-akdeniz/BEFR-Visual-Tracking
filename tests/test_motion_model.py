"""Tests for the constant-velocity motion model."""

from __future__ import annotations

import unittest

import numpy as np

from befr_visual_tracking.motion_model import (
    predict_state,
    process_noise_covariance,
    state_from_position_velocity,
    transition_matrix,
)


class TestTransitionMatrix(unittest.TestCase):
    def test_constant_velocity_propagation(self) -> None:
        dt = 0.1
        state = state_from_position_velocity([1.0, 2.0, 3.0], [0.5, -0.2, 0.1])
        predicted = transition_matrix(dt) @ state
        np.testing.assert_allclose(predicted[:3], state[:3] + dt * state[3:], atol=1e-12)
        np.testing.assert_allclose(predicted[3:], state[3:], atol=1e-12)

    def test_zero_dt_is_identity(self) -> None:
        F = transition_matrix(0.0)
        np.testing.assert_allclose(F, np.eye(6))


class TestProcessNoise(unittest.TestCase):
    def test_symmetric_positive_semidefinite(self) -> None:
        Q = process_noise_covariance(0.05, 0.5)
        np.testing.assert_allclose(Q, Q.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(Q)
        self.assertTrue(np.all(eigenvalues >= -1e-12))

    def test_scales_with_sigma_a_squared(self) -> None:
        Q1 = process_noise_covariance(0.1, 1.0)
        Q2 = process_noise_covariance(0.1, 2.0)
        np.testing.assert_allclose(Q2, 4.0 * Q1, atol=1e-12)


class TestPredictState(unittest.TestCase):
    def test_predict_increases_position_uncertainty(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, 1.5], [0.0, 0.0, 0.0])
        P = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        _, P_pred = predict_state(state, P, 0.2, 0.5)
        self.assertGreater(P_pred[0, 0], P[0, 0])


if __name__ == "__main__":
    unittest.main()
