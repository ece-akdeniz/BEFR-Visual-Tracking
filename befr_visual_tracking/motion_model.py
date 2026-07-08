"""Constant-velocity motion model shared by all trackers."""

from __future__ import annotations

import numpy as np

STATE_DIM = 6
POSITION_DIM = 3
VELOCITY_DIM = 3


def state_from_position_velocity(
    position: np.ndarray,
    velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Build [p_x, p_y, p_z, v_x, v_y, v_z]^T."""
    position = np.asarray(position, dtype=float).reshape(3)
    if velocity is None:
        velocity = np.zeros(3, dtype=float)
    else:
        velocity = np.asarray(velocity, dtype=float).reshape(3)
    return np.concatenate([position, velocity])


def position_from_state(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=float).reshape(STATE_DIM)[:POSITION_DIM]


def velocity_from_state(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=float).reshape(STATE_DIM)[POSITION_DIM:]


def transition_matrix(dt: float) -> np.ndarray:
    """Return F(dt) for the constant-velocity model."""
    if dt < 0.0:
        raise ValueError("Time step must be non-negative")

    F = np.eye(STATE_DIM, dtype=float)
    F[0:3, 3:6] = float(dt) * np.eye(POSITION_DIM)
    return F


def process_noise_covariance(dt: float, acceleration_noise_std: float) -> np.ndarray:
    """
    Return Q(dt) for unknown acceleration modelled as white noise with std sigma_a.
    """
    if dt < 0.0:
        raise ValueError("Time step must be non-negative")

    sigma_a = float(acceleration_noise_std)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    I3 = np.eye(POSITION_DIM, dtype=float)

    Q = sigma_a**2 * np.block(
        [
            [0.25 * dt4 * I3, 0.5 * dt3 * I3],
            [0.5 * dt3 * I3, dt2 * I3],
        ]
    )
    return Q


def normalize_initial_covariance(covariance: np.ndarray) -> np.ndarray:
    """
    Accept a 6x6 state covariance or a 3x3 position covariance.

    For a 3x3 input, velocity is initialised with zero mean and large variance
    derived from the position uncertainty scale.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (STATE_DIM, STATE_DIM):
        return covariance.copy()

    if covariance.shape == (POSITION_DIM, POSITION_DIM):
        P = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        P[:POSITION_DIM, :POSITION_DIM] = covariance
        position_scale = max(float(np.trace(covariance)), 1.0)
        P[POSITION_DIM:, POSITION_DIM:] = position_scale * np.eye(POSITION_DIM)
        return P

    raise ValueError("Covariance must have shape (3, 3) or (6, 6)")


def predict_state(
    state: np.ndarray,
    covariance: np.ndarray,
    dt: float,
    acceleration_noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate mean and covariance through the constant-velocity model."""
    if dt < 0.0:
        raise ValueError("Time step must be non-negative")

    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    covariance = np.asarray(covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)

    if dt == 0.0:
        return state.copy(), covariance.copy()

    F = transition_matrix(dt)
    Q = process_noise_covariance(dt, acceleration_noise_std)
    predicted_state = F @ state
    predicted_covariance = F @ covariance @ F.T + Q
    return predicted_state, predicted_covariance
