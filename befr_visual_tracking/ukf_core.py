"""Unscented Kalman Filter utilities for 6D constant-velocity state."""

from __future__ import annotations

import numpy as np

from befr_visual_tracking.motion_model import STATE_DIM


def ukf_scaling_parameters(
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return lambda and the UKF tuning parameters."""
    lambda_ = alpha**2 * (STATE_DIM + kappa) - STATE_DIM
    return lambda_, alpha, beta, kappa


def generate_sigma_points(
    state: np.ndarray,
    covariance: np.ndarray,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, float, float, float]:
    """Generate 2n+1 sigma points for the current state distribution."""
    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    covariance = np.asarray(covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
    lambda_, alpha, beta, _ = ukf_scaling_parameters(alpha=alpha, beta=beta, kappa=kappa)

    try:
        chol = np.linalg.cholesky((STATE_DIM + lambda_) * covariance)
    except np.linalg.LinAlgError:
        regularized = covariance + 1e-9 * np.eye(STATE_DIM)
        chol = np.linalg.cholesky((STATE_DIM + lambda_) * regularized)

    sigma_points = np.zeros((2 * STATE_DIM + 1, STATE_DIM), dtype=float)
    sigma_points[0] = state
    for index in range(STATE_DIM):
        sigma_points[index + 1] = state + chol[:, index]
        sigma_points[index + 1 + STATE_DIM] = state - chol[:, index]
    return sigma_points, lambda_, alpha, beta


def ukf_weights(
    lambda_: float,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and covariance weights for 2n+1 sigma points."""
    mean_weights = np.full(2 * STATE_DIM + 1, 0.5 / (STATE_DIM + lambda_))
    cov_weights = mean_weights.copy()
    mean_weights[0] = lambda_ / (STATE_DIM + lambda_)
    cov_weights[0] = lambda_ / (STATE_DIM + lambda_) + (1.0 - alpha**2 + beta)
    return mean_weights, cov_weights


def cap_covariance_eigenvalues(
    covariance: np.ndarray,
    max_eigenvalue: float = 100.0,
    min_eigenvalue: float = 1e-9,
) -> np.ndarray:
    """Limit extreme covariance growth for stable UKF sigma points."""
    covariance = np.asarray(covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, min_eigenvalue, max_eigenvalue)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T
